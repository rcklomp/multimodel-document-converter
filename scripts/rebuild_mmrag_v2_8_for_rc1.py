#!/usr/bin/env python
"""Phase I (Plan v2.9, 2026-05-11) — rebuild a Qdrant collection from
the canonical 34-doc corpus.

Original v2.9.0-rc1 use: rebuild `mmrag_v2_8` from local Ollama llava.

v2.11 extensions (Phase 1, 2026-05-17):
- `--collection` overrides the collection name (default `mmrag_v2_8`).
- `--provider {ollama,dashscope}` selects the embedding provider.
- `--model` overrides the embedding model (provider-aware default).
- `--api-key` for Dashscope (defaults to DASHSCOPE_API_KEY env var).
- `--resume` skips docs already fully ingested into the target collection
  (chunk_id-presence check via Qdrant scroll). Survives the kind of
  Ollama-mid-rebuild crash that hit v2.10 Phase 8.
- `--no-recreate` skips the implicit `--recreate` on doc 1 (use when
  resuming or layering into an existing collection).

Per-doc invocation now passes the provider/model/api-key flags through
to `scripts/ingest_to_qdrant.py`.

Run from project root, examples:

    # v2.10 ship-state rebuild (unchanged from original behavior):
    conda run -n mmrag-v2 --no-capture-output \\
      python scripts/rebuild_mmrag_v2_8_for_rc1.py

    # v2.11 Phase 1 challenger rebuild against Dashscope cloud:
    conda run -n mmrag-v2 --no-capture-output \\
      python scripts/rebuild_mmrag_v2_8_for_rc1.py \\
      --collection mmrag_v2_8__qwen3_dashscope \\
      --provider dashscope \\
      --model text-embedding-v4

    # Resume after a partial run (skip docs already in the collection):
    conda run -n mmrag-v2 --no-capture-output \\
      python scripts/rebuild_mmrag_v2_8_for_rc1.py --resume

Expected wall time: 4-8 hours on local Ollama llava; faster against
Dashscope cloud (no local model compute, just API throughput).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
LOG_DEFAULT = OUTPUT / "_logs" / "phase_i_rebuild.log"
COLLECTION_DEFAULT = "mmrag_v2_8__qwen3_dashscope"
COLLECTION_LEGACY = "mmrag_v2_8"  # v2.10 baseline (llava 4096-dim), retained for 30-day rollback
QDRANT_URL_DEFAULT = "http://localhost:6333"

CANONICAL_34 = [
    "HarryPotter_and_the_Sorcerers_Stone",
    "Form_0013_invoice",
    "Form_betwistingsformulier",
    "CarOK_voorraadtelling",
    "AIOS_LLM_Agent_Operating_System",
    "A_comprehensive_review_on_hybrid_electri",
    "Hybrid_electric_vehicles",
    "IRJET_Modeling_of_Solar_PV",
    "Recent_Trends_in_Transportation",
    "Combat_Aircraft_August_2025",
    "PCWorld_July_2025",
    "ATZ_Elektronik_German",
    "Kimothi_RAG_Guide",
    "Integra_manual",
    "Jungjun_AI_Agent",
    "Bourne_RAG_2024",
    "Devlin_LLM_Agents",
    "Raieli_AI_Agents",
    "Adedeji_GenAI_Google_Cloud",
    "Cronin_GenAI_Models",
    "Hao_ML_Platform",
    "Nagasubramanian_Agentic_AI",
    "Sekar_MCP_Standard",
    "Python_Cookbook",
    "ArcGIS_Python_Cookbook",
    "Fluent_Python",
    "Python_Distilled",
    "Ayeva_Python_Patterns",
    "Chaubal_PyTorch_Projects",
    "Earthship_Vol1",
    "Firearms",
    "Greenhouse_Design",
    "ChatGPT_Praktijk_handboek",
    "KI_En_ChatGPT_Praktische_Gids",
]


def _log_path(custom: Path | None) -> Path:
    return custom if custom is not None else LOG_DEFAULT


def log(message: str, log_path: Path) -> None:
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _doc_chunk_ids(jsonl_path: Path) -> set[str]:
    ids: set[str] = set()
    if not jsonl_path.exists():
        return ids
    for i, line in enumerate(jsonl_path.open("r", encoding="utf-8")):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if i == 0 and obj.get("object_type") == "ingestion_metadata":
            continue
        cid = obj.get("chunk_id")
        if cid:
            ids.add(cid)
    return ids


def _qdrant_count_chunks_for_doc(qdrant_url: str, collection: str,
                                  doc_chunk_ids: set[str]) -> int:
    """How many of this doc's chunk_ids are already present in the
    target collection.

    Uses the Qdrant scroll API with a filter on the `chunk_id` payload
    field. Conservative: if Qdrant is unreachable, returns 0 (treats
    the doc as not ingested, forces re-ingest).
    """
    if not doc_chunk_ids:
        return 0
    body = {
        "filter": {
            "must": [{
                "key": "chunk_id",
                "match": {"any": list(doc_chunk_ids)},
            }],
        },
        "limit": 1,
        "with_payload": False,
        "with_vector": False,
    }
    url = f"{qdrant_url}/collections/{collection}/points/count"
    count_body = {
        "filter": body["filter"],
        "exact": True,
    }
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(count_body).encode(),
            method="POST",
        )
        req.add_header("Content-Type", "application/json")
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read())
        return int((data.get("result") or {}).get("count") or 0)
    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, OSError, TimeoutError):
        return 0


def _doc_is_fully_ingested(jsonl_path: Path, qdrant_url: str,
                            collection: str) -> tuple[bool, int, int]:
    """Returns (is_fully_ingested, chunks_present, chunks_expected)."""
    expected = _doc_chunk_ids(jsonl_path)
    if not expected:
        return False, 0, 0
    present = _qdrant_count_chunks_for_doc(qdrant_url, collection, expected)
    # Allow a small tolerance — ingest_to_qdrant.py filters ~0.44 % of chunks
    # (empty content, validator rejections), so "fully ingested" means
    # "present >= 90 % of expected" rather than "present == expected".
    threshold = max(1, int(0.90 * len(expected)))
    return (present >= threshold, present, len(expected))


def _ingest_one(jsonl_path: Path, collection: str, provider: str,
                model: str | None, api_key: str | None,
                qdrant_url: str, recreate: bool,
                retries: int = 3) -> int:
    cmd = [
        sys.executable, "scripts/ingest_to_qdrant.py", str(jsonl_path),
        "--collection", collection,
        "--qdrant-url", qdrant_url,
        "--provider", provider,
    ]
    if model:
        cmd += ["--model", model]
    if api_key:
        cmd += ["--api-key", api_key]
    if recreate:
        cmd.append("--recreate")

    for attempt in range(1, retries + 1):
        proc = subprocess.run(cmd, cwd=ROOT)
        if proc.returncode == 0:
            return 0
        # Retry on transient failure (the v2.10 Phase 8 Ollama
        # URLError pattern). Exponential backoff: 30s, 60s, 120s.
        if attempt < retries:
            backoff = 30 * (2 ** (attempt - 1))
            print(
                f"  ! ingest exited rc={proc.returncode} (attempt {attempt}/{retries}); "
                f"sleeping {backoff}s before retry",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(backoff)
            continue
        return proc.returncode
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collection", type=str, default=COLLECTION_DEFAULT,
                        help=f"Target Qdrant collection (default: {COLLECTION_DEFAULT})")
    parser.add_argument("--qdrant-url", type=str, default=QDRANT_URL_DEFAULT)
    parser.add_argument("--provider", type=str, default="dashscope",
                        choices=["ollama", "dashscope"],
                        help="Embedding provider (default: dashscope as of v2.11.0; pass "
                             "ollama for legacy llava rebuild against mmrag_v2_8)")
    parser.add_argument("--model", type=str, default=None,
                        help="Embedding model. Default 'text-embedding-v4' for dashscope; "
                             "'llava' for ollama.")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for dashscope provider. "
                             "Defaults to DASHSCOPE_API_KEY env var.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip docs already fully ingested (≥90 %% of "
                             "chunk_ids present in the target collection). "
                             "Implies --no-recreate.")
    parser.add_argument("--no-recreate", action="store_true",
                        help="Do not pass --recreate on doc 1 (use when "
                             "layering into an existing collection).")
    parser.add_argument("--log-path", type=Path, default=None,
                        help="Override log path (default: output/_logs/phase_i_rebuild.log)")
    args = parser.parse_args()

    if args.provider == "dashscope" and not args.api_key:
        args.api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not args.api_key:
            print("ERROR: --provider dashscope requires --api-key or DASHSCOPE_API_KEY env var",
                  file=sys.stderr)
            return 2

    log_path = _log_path(args.log_path)
    skip_recreate = args.no_recreate or args.resume

    header = (
        f"=== REBUILD {args.collection} provider={args.provider} "
        f"model={args.model or '(default)'} "
        f"resume={args.resume} recreate_doc1={not skip_recreate} ==="
    )
    log(header, log_path)
    total_start = time.time()

    for idx, name in enumerate(CANONICAL_34, start=1):
        jsonl = OUTPUT / name / "ingestion.jsonl"
        if not jsonl.exists():
            log(f"[{idx}/34] MISSING {name} — aborting", log_path)
            return 1

        if args.resume:
            done, present, expected = _doc_is_fully_ingested(
                jsonl, args.qdrant_url, args.collection
            )
            if done:
                log(
                    f"[{idx}/34] SKIP {name} (resume: {present}/{expected} chunks present)",
                    log_path,
                )
                continue
            if expected > 0:
                log(
                    f"[{idx}/34] PARTIAL {name} ({present}/{expected} present) — re-ingesting",
                    log_path,
                )

        log(f"[{idx}/34] START {name}", log_path)
        recreate = (idx == 1) and not skip_recreate
        t0 = time.time()
        rc = _ingest_one(
            jsonl_path=jsonl,
            collection=args.collection,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            qdrant_url=args.qdrant_url,
            recreate=recreate,
        )
        elapsed = time.time() - t0
        if rc != 0:
            log(f"[{idx}/34] FAIL {name} (rc={rc}, {elapsed:.0f}s)", log_path)
            return rc
        log(f"[{idx}/34] DONE {name} ({elapsed:.0f}s)", log_path)

    total = time.time() - total_start
    log(f"=== ALL DONE — total wall time {total/60:.1f} min ===", log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
