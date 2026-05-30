#!/usr/bin/env python3
"""v2.10 synthetic-soak harness — LLM-judged retrieval quality eval.

Substitutes a "two weeks of real RAG usage" soak with an automated
LLM-as-judge protocol that:

1. SAMPLE — stratified-sample ~300 text chunks across the 34-doc
   `mmrag_v2_8` corpus. Heuristics: ≥ 150 chars, not pure-code, not
   advertisement.
2. GENERATE — for each sampled chunk, ask Dashscope qwen-max to write
   2 natural user queries whose answer is in the chunk (~600 queries).
3. RETRIEVE — embed each query via Ollama llava, search `mmrag_v2_8`
   top-5.
4. JUDGE — ask Dashscope qwen-max to grade each top-1 result on
   relevance / format / faithfulness (0-2 each).
5. REPORT — aggregate per-doc and corpus-wide metrics + a list of the
   worst (lowest-scoring) (query, chunk) pairs as v2.10.x candidate
   defects. Writes
   `docs/QUALITY_SNAPSHOT_<DATE>_v2.10_soak.md`.

Pinned design choices (2026-05-16):
- Judge provider: Dashscope `qwen-max` (best judgment quality in this
  pipeline; matches Phase 5b enrichment provider).
- Query count: 300 chunks × 2 queries = 600.
- Threshold: report-only. No ship gate; humans read the numbers.
- Cadence: run on every tagged release.

Resumable. Re-running the script picks up where the work file left
off so an API hiccup mid-stage doesn't waste prior calls. Stages
can also be run individually:

  python scripts/synthetic_soak.py --stage sample
  python scripts/synthetic_soak.py --stage generate
  python scripts/synthetic_soak.py --stage retrieve
  python scripts/synthetic_soak.py --stage judge
  python scripts/synthetic_soak.py --stage report
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from search_qdrant import embed as _embed_ollama, search  # noqa: E402
from ingest_to_qdrant import embed_text_dashscope, embed_text_omlx  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "output" / "soak" / "v2.10"
DEFAULT_WORK_PATH = OUTPUT_DIR / "work.jsonl"

# Disk-headroom precheck (v2.14 Phase 5 — incident-improvement from
# the v2.13 P1 disk-full crash that killed Qdrant mid-judge). A soak's
# retrieve+judge passes only add a few MB to the work file, but Qdrant
# (running on the same disk) writes WAL + segments during retrieval
# and crashes hard if the disk fills. 10 GB free is the floor below
# which we refuse to start a stage. Override via the
# `SOAK_DISK_HEADROOM_FLOOR_GB` env var (e.g. `=2.0` for a tight CI run).
DISK_HEADROOM_FLOOR_GB = float(
    os.environ.get("SOAK_DISK_HEADROOM_FLOOR_GB", "10.0")
)

# v2.15 Phase 3 [F] — document-class telemetry sink. Soak harness
# writes one JSON line per query into this rolling log; analyzed
# by scripts/analyze_doc_class_telemetry.py at cycle-open per
# [DEPRECATED: See V3_EXECUTION_MANDATE.md]. Override via env var for test/CI
# runs that shouldn't pollute the production rolling log.
TELEMETRY_LOG_PATH = Path(
    os.environ.get(
        "MMRAG_TELEMETRY_LOG",
        str(REPO_ROOT / "output/telemetry/document_class_hits.jsonl"),
    )
)
DEFAULT_REPORT_PATH = REPO_ROOT / "docs" / (
    f"QUALITY_SNAPSHOT_{datetime.now().strftime('%Y-%m-%d')}_v2.10_soak.md"
)
DOCS_ROOT = REPO_ROOT / "output"
COLLECTION_DEFAULT_DASHSCOPE = "mmrag_v2_8__qwen3_dashscope"
COLLECTION_DEFAULT_OLLAMA = "mmrag_v2_8"
COLLECTION_DEFAULT_OMLX = "mmrag_v2_8__qwen3_local"
EMBED_MODEL_OLLAMA = "llava"
EMBED_MODEL_DASHSCOPE = "text-embedding-v4"
EMBED_MODEL_OMLX = "Qwen3-Embedding-8B-mxfp8"
OMLX_DEFAULT_URL = "http://10.0.10.246:8000/v1/embeddings"
TOP_K = 5

# Dashscope OpenAI-compatible endpoint (matches scripts/convert_all.sh /
# refiner.py pattern).
DASHSCOPE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
JUDGE_MODEL = "qwen-max"
GENERATOR_MODEL = "qwen-max"

# v2.14 Phase 4c: local-LLM query-generation provider. Re-uses the
# GX10 vLLM endpoint defaults from `mmrag_v2.retrieval.hyde` so a
# single source of truth tracks any future endpoint swap.
# Query generation is leniency-trap-immune (it isn't judging), so the
# vllm path is safe even on the all-axes-RESTRICTED 27B verdict.
from mmrag_v2.retrieval.hyde import (  # noqa: E402
    VLLM_DEFAULT_MODEL as _VLLM_GEN_DEFAULT_MODEL,
    VLLM_DEFAULT_URL as _VLLM_GEN_DEFAULT_URL,
)
VLLM_GEN_DEFAULT_URL = _VLLM_GEN_DEFAULT_URL
VLLM_GEN_DEFAULT_MODEL = _VLLM_GEN_DEFAULT_MODEL

# Heuristic content filters for chunk sampling.
MIN_CHUNK_CHARS = 150
MAX_CODE_RATIO = 0.4
ADVERT_KEYWORDS = ("subscribe", "buy now", "click here", "limited time", "discount")

# Canonical doc directories. Mirrors scripts/rebuild_mmrag_v2_8_for_rc1.py.
# v2.16 Phase 0: renamed from CANONICAL_34 → CANONICAL_DOCS and extended
# from 34 to 38 entries (7 PDFs ingested from data/raw/; 4 passed strict
# gate, 3 dropped per DECISIONS "v2.16 Phase 0 Strict-Gate Honest
# Reduction"). Name describes semantic role (the canonical docs list),
# not cardinality.
CANONICAL_DOCS = [
    "HarryPotter_and_the_Sorcerers_Stone", "Form_0013_invoice", "Form_betwistingsformulier",
    "CarOK_voorraadtelling", "AIOS_LLM_Agent_Operating_System",
    "A_comprehensive_review_on_hybrid_electri", "Hybrid_electric_vehicles",
    "IRJET_Modeling_of_Solar_PV", "Recent_Trends_in_Transportation",
    "Combat_Aircraft_August_2025", "PCWorld_July_2025", "ATZ_Elektronik_German",
    "Kimothi_RAG_Guide", "Integra_manual", "Jungjun_AI_Agent", "Bourne_RAG_2024",
    "Devlin_LLM_Agents", "Raieli_AI_Agents", "Adedeji_GenAI_Google_Cloud",
    "Cronin_GenAI_Models", "Hao_ML_Platform", "Nagasubramanian_Agentic_AI",
    "Sekar_MCP_Standard", "Python_Cookbook", "ArcGIS_Python_Cookbook",
    "Fluent_Python", "Python_Distilled", "Ayeva_Python_Patterns",
    "Chaubal_PyTorch_Projects", "Earthship_Vol1", "Firearms", "Greenhouse_Design",
    "ChatGPT_Praktijk_handboek", "KI_En_ChatGPT_Praktische_Gids",
    # v2.16 Phase 0 additions (data/raw/, ingested 2026-05-25; only the
    # 4 that PASS strict gate are canonical):
    "ATZ_Aerodynamik_Nutzfahrzeugen",
    "ATZ_ESF_Mercedes_2009",
    "Schwungradspeicher",
    "Eliasz_Zephyr_RTOS",
]


def _load_chunks(doc_name: str) -> list[dict]:
    jsonl = DOCS_ROOT / doc_name / "ingestion.jsonl"
    if not jsonl.exists():
        return []
    chunks: list[dict] = []
    for i, line in enumerate(jsonl.open("r", encoding="utf-8")):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if i == 0 and obj.get("object_type") == "ingestion_metadata":
            continue
        chunks.append(obj)
    return chunks


def _is_eligible_text_chunk(chunk: dict) -> bool:
    if chunk.get("modality") != "text":
        return False
    content = (chunk.get("content") or "").strip()
    if len(content) < MIN_CHUNK_CHARS:
        return False
    # Reject mostly-code chunks (heuristic: many short indented lines)
    lines = content.splitlines()
    if lines:
        code_like = sum(1 for ln in lines if ln.startswith(("    ", "\t", "  ")) or ln.strip().startswith((">>>", "...")))
        if code_like / max(1, len(lines)) > MAX_CODE_RATIO:
            return False
    lowered = content.lower()
    if any(kw in lowered for kw in ADVERT_KEYWORDS):
        return False
    return True


def stage_sample(seed: int, n_chunks: int, work_path: Path) -> None:
    if work_path.exists():
        print(f"  sample: work file already exists at {work_path}; skip (delete to re-sample)")
        return
    rng = random.Random(seed)
    print(f"  sample: stratified across {len(CANONICAL_DOCS)} docs, target n={n_chunks}, seed={seed}")
    per_doc_target = max(1, n_chunks // len(CANONICAL_DOCS))
    sampled: list[dict] = []
    for doc_name in CANONICAL_DOCS:
        chunks = [c for c in _load_chunks(doc_name) if _is_eligible_text_chunk(c)]
        if not chunks:
            print(f"    {doc_name}: 0 eligible (skip)")
            continue
        take = min(per_doc_target, len(chunks))
        picks = rng.sample(chunks, take)
        for p in picks:
            sampled.append({
                "doc_dir": doc_name,
                "gold_chunk_id": p.get("chunk_id"),
                "gold_doc_id": p.get("doc_id"),
                "gold_source_file": (p.get("metadata") or {}).get("source_file"),
                "gold_page_number": (p.get("metadata") or {}).get("page_number"),
                "gold_content": (p.get("content") or "").strip(),
                "queries": [],
            })
        print(f"    {doc_name}: sampled {take}/{len(chunks)}")
    # Cap to exactly n_chunks if oversampled (rare with the floor)
    rng.shuffle(sampled)
    if len(sampled) > n_chunks:
        sampled = sampled[:n_chunks]
    # Assign deterministic sample_ids in shuffled order
    for i, row in enumerate(sampled, start=1):
        row["sample_id"] = f"S{i:04d}"
    work_path.parent.mkdir(parents=True, exist_ok=True)
    with work_path.open("w", encoding="utf-8") as fh:
        for row in sampled:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  sample: wrote {len(sampled)} rows to {work_path}")


def _check_disk_headroom(work_path: Path, floor_gb: float = DISK_HEADROOM_FLOOR_GB) -> None:
    """v2.14 Phase 5: refuse to start a write-heavy stage when free
    disk is below `floor_gb`. Prevents the v2.13 P1 incident where
    the disk filled mid-judge and crashed Qdrant's container overlayfs.

    Checks the partition holding `work_path` (which is the same partition
    Qdrant writes its segments to in the standard dev setup).
    """
    try:
        target = work_path.parent if work_path.parent.exists() else REPO_ROOT
        usage = shutil.disk_usage(target)
    except OSError as e:
        print(f"  WARNING: disk-headroom precheck couldn't stat {target}: {e}",
              file=sys.stderr)
        return
    free_gb = usage.free / (1024 ** 3)
    if free_gb < floor_gb:
        print(
            f"\nERROR: Insufficient disk headroom — {free_gb:.1f} GB free, "
            f"floor is {floor_gb:.1f} GB. Aborting before this stage can\n"
            f"crash Qdrant (incident reference: v2.13 P1 soak, 2026-05-22).\n"
            f"Free space and retry, or override via DISK_HEADROOM_FLOOR_GB=<smaller>.\n",
            file=sys.stderr,
        )
        raise SystemExit(3)
    if free_gb < floor_gb * 2:
        print(f"  NOTE: disk headroom tight ({free_gb:.1f} GB free; floor {floor_gb:.1f} GB)")


def _read_work(work_path: Path) -> list[dict]:
    if not work_path.exists():
        return []
    rows: list[dict] = []
    for line in work_path.open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _write_work(work_path: Path, rows: list[dict]) -> None:
    tmp = work_path.with_suffix(work_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(work_path)


def _append_telemetry(query_text: str, top: list[dict]) -> None:
    """v2.15 Phase 3 [F]: append one telemetry record per query into
    the rolling document-class-hits log. No-op if the import or write
    fails (telemetry is best-effort; never breaks the soak).

    Per `docs/PLAN_V2.15.md` §Phase 3 [F] step 3 + Round-4 Finding 1's
    `analyze_doc_class_telemetry.py` deliverable contract."""
    try:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from mmrag_v2.retrieval.documented_limitations import class_names
        from mmrag_v2.retrieval.telemetry import build_telemetry_record
    except ImportError:
        return
    try:
        record = build_telemetry_record(
            query_text, top, class_names(), top_k=5,
        )
        TELEMETRY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TELEMETRY_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except (OSError, ValueError) as e:
        print(f"    ! telemetry write failed (non-blocking): {e}",
              file=sys.stderr)


def _call_dashscope(api_key: str, model: str, messages: list[dict],
                    temperature: float = 0.0, max_tokens: int = 600,
                    timeout: int = 60, retries: int = 3) -> str | None:
    """OpenAI-compatible chat-completions call against Dashscope. Returns
    response text or None on failure."""
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(DASHSCOPE_URL, data=body, method="POST")
            req.add_header("Authorization", f"Bearer {api_key}")
            req.add_header("Content-Type", "application/json")
            resp = urllib.request.urlopen(req, timeout=timeout)
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            try:
                detail = e.read().decode("utf-8", errors="replace")[:200]
            except Exception:
                detail = ""
            print(f"    ! Dashscope HTTP {e.code}: {detail}", file=sys.stderr)
            return None
        except (urllib.error.URLError, ConnectionError, OSError, TimeoutError) as e:
            last_err = e
            time.sleep(2 ** attempt)
            continue
        except Exception as e:
            print(f"    ! Dashscope error: {e}", file=sys.stderr)
            return None
    print(f"    ! Dashscope failed after {retries} retries: {last_err}", file=sys.stderr)
    return None


def _call_vllm(url: str, model: str, messages: list[dict], *,
               api_key: str | None = None,
               temperature: float = 0.0, max_tokens: int = 600,
               timeout: int = 60, retries: int = 3) -> str | None:
    """OpenAI-compatible chat-completions call against a local vLLM
    endpoint. Returns response text or None on failure.

    v2.14 Phase 4c. Sends the `chat_template_kwargs.enable_thinking=False`
    extension defensively per memory/feedback_qwen3_thinking_payload —
    Qwen3 models served with `--reasoning-parser qwen3` default to
    thinking mode and route output to `message.reasoning`, starving
    `message.content`. The kwarg is a no-op on non-thinking templates
    so it's safe across all GX10 model swaps.
    """
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=body, method="POST")
            req.add_header("Content-Type", "application/json")
            if api_key:
                req.add_header("Authorization", f"Bearer {api_key}")
            resp = urllib.request.urlopen(req, timeout=timeout)
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            try:
                detail = e.read().decode("utf-8", errors="replace")[:200]
            except Exception:
                detail = ""
            print(f"    ! vLLM HTTP {e.code}: {detail}", file=sys.stderr)
            return None
        except (urllib.error.URLError, ConnectionError, OSError, TimeoutError) as e:
            last_err = e
            time.sleep(2 ** attempt)
            continue
        except Exception as e:
            print(f"    ! vLLM error: {e}", file=sys.stderr)
            return None
    print(f"    ! vLLM failed after {retries} retries: {last_err}", file=sys.stderr)
    return None


_JSON_ARRAY_RE = re.compile(r"\[\s*(?:\".*?\"\s*,?\s*)+\]", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _extract_json(text: str, expect: str) -> Any:
    """Robust JSON extraction (LLMs sometimes wrap output in prose/markdown)."""
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if expect == "array":
        m = _JSON_ARRAY_RE.search(text)
    else:
        m = _JSON_OBJECT_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


GENERATE_SYSTEM = (
    "You write evaluation queries for a retrieval-augmented generation system. "
    "Return ONLY a JSON array of exactly 2 short natural questions (5-15 words each). "
    "Do not include any prose around the JSON."
)

GENERATE_USER_TEMPLATE = """Given this passage, write 2 natural user questions whose answer is in this passage.
Requirements:
- Questions should be specific enough that this passage (or a closely related passage from the same document) would be a strong answer.
- Do not quote the passage verbatim.
- Make the two questions DIFFERENT in shape — e.g. one fact-seeking ("what is..."), one explanatory or how-to.
- Use the same language as the passage (English, Dutch, or German).

Passage:
\"\"\"
{content}
\"\"\"

Return ONLY a JSON array of 2 strings."""


def stage_generate(work_path: Path, api_key: str, *,
                   gen_provider: str = "dashscope",
                   gen_url: str = VLLM_GEN_DEFAULT_URL,
                   gen_model: str | None = None) -> None:
    """Generate 2 queries per sampled chunk.

    `gen_provider` selects the backend:
      - "dashscope" (default): cloud `qwen-max`. Requires `api_key`.
      - "vllm": local GX10 endpoint (Phase 4c, leniency-trap-immune
        because query generation isn't judging). Reuses the default
        endpoint + model from `mmrag_v2.retrieval.hyde`.
    """
    rows = _read_work(work_path)
    if not rows:
        print("  generate: no work file; run --stage sample first", file=sys.stderr)
        return
    if gen_model is None:
        gen_model = VLLM_GEN_DEFAULT_MODEL if gen_provider == "vllm" else GENERATOR_MODEL
    print(f"  generate: using gen_provider={gen_provider} model={gen_model}")
    if gen_provider == "vllm":
        print(f"  generate: local vLLM at {gen_url} ($0; ~3-9s per query pair)")
    needed = sum(1 for r in rows if not r.get("queries"))
    print(f"  generate: {needed}/{len(rows)} rows need queries")
    done = 0
    for i, row in enumerate(rows):
        if row.get("queries"):
            continue
        content = row["gold_content"][:2500]  # cap input size
        messages = [
            {"role": "system", "content": GENERATE_SYSTEM},
            {"role": "user", "content": GENERATE_USER_TEMPLATE.format(content=content)},
        ]
        if gen_provider == "vllm":
            result = _call_vllm(
                gen_url, gen_model, messages=messages,
                temperature=0.3, max_tokens=300,
            )
        else:
            result = _call_dashscope(
                api_key, gen_model, messages=messages,
                temperature=0.3, max_tokens=300,
            )
        queries: list[str] = []
        if result:
            parsed = _extract_json(result, "array")
            if isinstance(parsed, list):
                queries = [str(q).strip() for q in parsed if isinstance(q, (str, int))]
        if not queries or len(queries) < 2:
            print(f"    [{row['sample_id']}] generation failed; skipping", file=sys.stderr)
            row["queries"] = []  # leave empty so a re-run can retry
            continue
        row["queries"] = [
            {"query_id": f"{row['sample_id']}.Q{j+1}", "query_text": q.strip()}
            for j, q in enumerate(queries[:2])
        ]
        done += 1
        if done % 10 == 0 or i + 1 == len(rows):
            _write_work(work_path, rows)
            print(f"    [{i+1}/{len(rows)}] generated {done} so far (work flushed)")
    _write_work(work_path, rows)
    print(f"  generate: wrote queries for {done} rows")


def _embed_query(text: str, provider: str, model: str,
                 ollama_url: str, api_key: str,
                 omlx_url: str = OMLX_DEFAULT_URL) -> list[float]:
    if provider == "dashscope":
        return embed_text_dashscope(text, model, api_key)
    if provider == "omlx":
        return embed_text_omlx(text, model, api_key, url=omlx_url)
    return _embed_ollama(text, model=model, ollama_url=ollama_url)


def stage_retrieve(work_path: Path, qdrant_url: str, ollama_url: str,
                   collection: str, provider: str, embed_model: str,
                   api_key: str, *,
                   rerank_backend: str | None = None,
                   top_k_retrieve: int = TOP_K,
                   top_n_return: int = TOP_K,
                   hybrid: bool = False,
                   sparse_collection: str = "mmrag_v2_8__bm25_sparse",
                   bm25_index_path: str = "tests/fixtures/bm25_index_v2_12.json",
                   use_hyde: bool = False,
                   auto_intent_hyde: bool = False,
                   hyde_provider: str = "dashscope") -> None:
    """Retrieve top-K candidates per query and store them on the row.

    When `rerank_backend` is None (default), behavior is identical to
    pre-v2.12: Qdrant top-`top_n_return` is stored verbatim.

    When `rerank_backend` is set (`dashscope` or `omlx`), the pipeline
    becomes: embed → Qdrant top-`top_k_retrieve` → reranker → top-
    `top_n_return`. The stored entry includes `rerank_score` and
    `rerank_index` on each chunk so the Phase 1 soak can compare
    reranker output across backends.
    """
    _check_disk_headroom(work_path)
    rows = _read_work(work_path)
    if not rows:
        print("  retrieve: no work file", file=sys.stderr)
        return
    queries_total = sum(len(r.get("queries") or []) for r in rows)
    queries_done = sum(1 for r in rows for q in (r.get("queries") or []) if q.get("retrieval"))
    rerank_desc = f" rerank={rerank_backend}" if rerank_backend else " (no rerank)"
    print(f"  retrieve: {queries_done}/{queries_total} queries already retrieved "
          f"(collection={collection}, provider={provider}, model={embed_model}{rerank_desc}, "
          f"top_k_retrieve={top_k_retrieve}, top_n_return={top_n_return})")

    # Lazy-construct the reranker so the import path stays optional.
    reranker = None
    if rerank_backend:
        # Import via the public mmrag_v2.retrieval API.
        try:
            sys.path.insert(0, str(REPO_ROOT / "src"))
            from mmrag_v2.retrieval import get_reranker  # noqa: E402
            reranker = get_reranker(rerank_backend)
            print(f"    reranker constructed: backend={reranker.name} model={reranker.model}")
        except Exception as e:
            print(f"    ! could not construct reranker '{rerank_backend}': {e}",
                  file=sys.stderr)
            return

    # Hybrid retrieve uses the full mmrag_v2.retrieval pipeline so we get
    # consistent dense + sparse + RRF + rerank composition.
    hybrid_retrieve = None
    if hybrid:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from mmrag_v2.retrieval import retrieve_hybrid_reranked  # noqa: E402
        hybrid_retrieve = retrieve_hybrid_reranked
        print(f"    hybrid mode: dense={collection} + sparse={sparse_collection} + RRF")

    flushed = 0
    for r in rows:
        for q in (r.get("queries") or []):
            if q.get("retrieval"):
                continue

            # Branch A: hybrid retrieval (does its own embed + dense + sparse + RRF + rerank).
            if hybrid_retrieve is not None:
                try:
                    reranked = hybrid_retrieve(
                        q["query_text"],
                        dense_collection=collection,
                        sparse_collection=sparse_collection,
                        bm25_index_path=bm25_index_path,
                        top_k_retrieve=top_k_retrieve,
                        top_n_fuse=top_k_retrieve,
                        top_n_return=top_n_return,
                        embed_provider=provider,
                        embed_model=embed_model,
                        embed_api_key=api_key,
                        qdrant_url=qdrant_url,
                        reranker=reranker,
                        use_hyde=use_hyde,
                        auto_intent_hyde=auto_intent_hyde,
                        hyde_provider=hyde_provider,
                    )
                except Exception as e:
                    print(f"    ! hybrid retrieval failed for {q['query_id']}: {e}",
                          file=sys.stderr)
                    continue
                top = []
                for hit in reranked:
                    payload = hit.get("payload") or {}
                    top.append({
                        "chunk_id": payload.get("chunk_id") or str(hit.get("id")),
                        "doc_id": payload.get("doc_id"),
                        "source_file": payload.get("source_file"),
                        "page_number": payload.get("page_number"),
                        "modality": payload.get("modality"),
                        "score": round(float(hit.get("score") or 0.0), 6),
                        "rerank_score": round(float(hit.get("rerank_score") or 0.0), 6),
                        "rerank_index": int(hit.get("rerank_index", -1)),
                        "content": (payload.get("content") or "").strip()[:1500],
                    })
                q["retrieval"] = {
                    "top_k": top,
                    "rerank_backend": rerank_backend,
                    "top_k_retrieve": top_k_retrieve,
                    "hybrid": True,
                    "sparse_collection": sparse_collection,
                }
                _append_telemetry(q["query_text"], top)
                flushed += 1
                if flushed % 20 == 0:
                    _write_work(work_path, rows)
                continue

            # Branch B (legacy): dense-only retrieve + optional rerank.
            embed_text = q["query_text"]
            if use_hyde:
                # Lazy import — only when needed.
                sys.path.insert(0, str(REPO_ROOT / "src"))
                from mmrag_v2.retrieval.hyde import generate_with_fallback  # noqa: E402
                embed_text = generate_with_fallback(q["query_text"], api_key)
            try:
                vec = _embed_query(embed_text, provider, embed_model,
                                    ollama_url, api_key)
                results = search(vec, collection, limit=top_k_retrieve,
                                 qdrant_url=qdrant_url)
            except Exception as e:
                print(f"    ! retrieval failed for {q['query_id']}: {e}", file=sys.stderr)
                continue

            # Optionally rerank the Qdrant top-K.
            if reranker is not None:
                rerank_inputs = []
                for i, hit in enumerate(results):
                    payload = hit.get("payload") or {}
                    rerank_inputs.append({
                        "chunk_id": payload.get("chunk_id") or str(hit.get("id")),
                        "content": (payload.get("content") or "")[:1500],
                        "_hit": hit,
                    })
                try:
                    reranked = reranker.rerank(
                        q["query_text"], rerank_inputs, top_n=top_n_return
                    )
                except Exception as e:
                    print(f"    ! rerank failed for {q['query_id']}: {e}; "
                          f"falling back to vector-rank order", file=sys.stderr)
                    reranked = [
                        {**rerank_inputs[i], "rerank_score": 0.0, "rerank_index": i}
                        for i in range(min(top_n_return, len(rerank_inputs)))
                    ]
                top = []
                for r_item in reranked:
                    hit = r_item.get("_hit") or {}
                    payload = hit.get("payload") or {}
                    top.append({
                        "chunk_id": payload.get("chunk_id") or str(hit.get("id")),
                        "doc_id": payload.get("doc_id"),
                        "source_file": payload.get("source_file"),
                        "page_number": payload.get("page_number"),
                        "modality": payload.get("modality"),
                        "score": round(float(hit.get("score") or 0.0), 6),
                        "rerank_score": round(float(r_item.get("rerank_score") or 0.0), 6),
                        "rerank_index": int(r_item.get("rerank_index", -1)),
                        "content": (payload.get("content") or "").strip()[:1500],
                    })
            else:
                top = []
                for hit in results[:top_n_return]:
                    payload = hit.get("payload") or {}
                    top.append({
                        "chunk_id": payload.get("chunk_id") or str(hit.get("id")),
                        "doc_id": payload.get("doc_id"),
                        "source_file": payload.get("source_file"),
                        "page_number": payload.get("page_number"),
                        "modality": payload.get("modality"),
                        "score": round(float(hit.get("score") or 0.0), 6),
                        "content": (payload.get("content") or "").strip()[:1500],
                    })
            q["retrieval"] = {
                "top_k": top,
                "rerank_backend": rerank_backend,
                "top_k_retrieve": top_k_retrieve,
            }
            _append_telemetry(q["query_text"], top)
            flushed += 1
            if flushed % 20 == 0:
                _write_work(work_path, rows)
    _write_work(work_path, rows)
    print(f"  retrieve: completed {flushed} new retrievals")


JUDGE_SYSTEM = (
    "You evaluate retrieval-augmented generation (RAG) quality. "
    "Return ONLY a JSON object with integer fields relevance, format, faithfulness "
    "(each 0, 1, or 2) and a short string field rationale. No prose outside the JSON."
)

JUDGE_USER_TEMPLATE = """Grade how well the RETRIEVED chunk answers the USER QUERY.

USER QUERY:
{query}

GOLD PASSAGE (the chunk that was used to generate the query — for context only, do not penalize the retrieved chunk for being a different chunk from the same document):
\"\"\"
{gold}
\"\"\"

RETRIEVED CHUNK (top-1 from the retrieval system):
source_file: {source_file}
page: {page}
\"\"\"
{retrieved}
\"\"\"

Score on three axes, each 0/1/2:

1. relevance: Does the retrieved chunk's content answer the user query?
   2 = answers it directly. 1 = same topic but doesn't really answer. 0 = wrong domain.

2. format: Is the retrieved chunk content well-formed prose / code / table?
   2 = clean and readable. 1 = minor issues (some truncation, odd whitespace).
   0 = broken (leaked markup, garbled OCR, marker artifacts, severe truncation).

3. faithfulness: Would a user reading ONLY this chunk get a correct answer?
   2 = self-contained correct answer. 1 = partial / needs more context.
   0 = misleading or wrong.

Return ONLY: {{"relevance": <0-2>, "format": <0-2>, "faithfulness": <0-2>, "rationale": "<one sentence>"}}"""


def stage_judge(
    work_path: Path,
    api_key: str,
    *,
    judge_provider: str = "dashscope",
    judge_url: str = VLLM_GEN_DEFAULT_URL,
    judge_model: str | None = None,
) -> None:
    _check_disk_headroom(work_path)
    rows = _read_work(work_path)
    if not rows:
        print("  judge: no work file", file=sys.stderr)
        return
    if judge_model is None:
        judge_model = (
            VLLM_GEN_DEFAULT_MODEL if judge_provider == "vllm" else JUDGE_MODEL
        )
    print(
        f"  judge: provider={judge_provider} model={judge_model}"
        + (f" url={judge_url}" if judge_provider == "vllm" else "")
    )
    queries_total = sum(len(r.get("queries") or []) for r in rows)
    queries_done = sum(
        1 for r in rows for q in (r.get("queries") or [])
        if q.get("judgment") and q["judgment"].get("relevance") is not None
    )
    print(f"  judge: {queries_done}/{queries_total} queries already judged")
    flushed = 0
    for r in rows:
        for q in (r.get("queries") or []):
            judgment = q.get("judgment") or {}
            if judgment.get("relevance") is not None:
                continue
            retrieval = q.get("retrieval") or {}
            top = (retrieval.get("top_k") or [])
            if not top:
                continue
            top1 = top[0]
            content = top1.get("content") or ""
            messages = [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                    query=q["query_text"],
                    gold=r["gold_content"][:1500],
                    source_file=top1.get("source_file") or "",
                    page=top1.get("page_number"),
                    retrieved=content[:1500],
                )},
            ]
            if judge_provider == "vllm":
                result = _call_vllm(
                    judge_url, judge_model, messages,
                    temperature=0.0, max_tokens=200,
                )
            else:
                result = _call_dashscope(
                    api_key, judge_model, messages=messages,
                    temperature=0.0, max_tokens=200,
                )
            parsed = _extract_json(result or "", "object") if result else None
            if not isinstance(parsed, dict) or "relevance" not in parsed:
                print(f"    ! judge parse failed for {q['query_id']}; skipping", file=sys.stderr)
                continue
            try:
                q["judgment"] = {
                    "relevance": int(parsed.get("relevance", 0)),
                    "format": int(parsed.get("format", 0)),
                    "faithfulness": int(parsed.get("faithfulness", 0)),
                    "rationale": str(parsed.get("rationale", ""))[:300],
                }
            except (ValueError, TypeError):
                print(f"    ! judge cast failed for {q['query_id']}; skipping", file=sys.stderr)
                continue
            flushed += 1
            if flushed % 20 == 0:
                _write_work(work_path, rows)
                print(f"    judged {flushed} so far (work flushed)")
    _write_work(work_path, rows)
    print(f"  judge: completed {flushed} new judgments")


def stage_report(work_path: Path, report_path: Path,
                 collection: str, provider: str, embed_model: str,
                 *, gen_provider: str = "dashscope",
                 gen_model: str | None = None,
                 judge_provider: str = "dashscope",
                 judge_model: str | None = None) -> None:
    rows = _read_work(work_path)
    if not rows:
        print("  report: no work file", file=sys.stderr)
        return

    n_chunks = len(rows)
    queries: list[dict] = []
    for r in rows:
        for q in (r.get("queries") or []):
            queries.append({"row": r, "query": q})

    # Recall metrics (deterministic, no LLM)
    recall_at_1_chunk = 0
    recall_at_5_chunk = 0
    recall_at_5_doc = 0
    judged = 0
    rel_sum = fmt_sum = fait_sum = 0
    rel_max = fmt_max = fait_max = 0
    per_doc: dict[str, dict[str, int]] = {}
    weakest: list[tuple[int, dict]] = []  # (total_score, query+row pair)

    for entry in queries:
        r = entry["row"]
        q = entry["query"]
        gold_chunk = r["gold_chunk_id"]
        gold_doc = r["gold_doc_id"]
        doc_dir = r["doc_dir"]
        slot = per_doc.setdefault(doc_dir, {"queries": 0, "r1_chunk": 0, "r5_chunk": 0, "r5_doc": 0,
                                            "rel_sum": 0, "fmt_sum": 0, "fait_sum": 0, "judged": 0})
        slot["queries"] += 1
        retrieval = (q.get("retrieval") or {}).get("top_k") or []
        if retrieval:
            top_ids = [hit["chunk_id"] for hit in retrieval]
            top_docs = [hit.get("doc_id") for hit in retrieval]
            if top_ids and top_ids[0] == gold_chunk:
                recall_at_1_chunk += 1
                slot["r1_chunk"] += 1
            if gold_chunk in top_ids[:5]:
                recall_at_5_chunk += 1
                slot["r5_chunk"] += 1
            if gold_doc in top_docs[:5]:
                recall_at_5_doc += 1
                slot["r5_doc"] += 1
        judgment = q.get("judgment") or {}
        if judgment.get("relevance") is not None:
            judged += 1
            rel = int(judgment["relevance"]); fmt = int(judgment["format"]); fait = int(judgment["faithfulness"])
            rel_sum += rel; fmt_sum += fmt; fait_sum += fait
            rel_max += 2; fmt_max += 2; fait_max += 2
            slot["rel_sum"] += rel; slot["fmt_sum"] += fmt; slot["fait_sum"] += fait
            slot["judged"] += 1
            total = rel + fmt + fait
            weakest.append((total, entry))

    weakest.sort(key=lambda x: x[0])
    weakest_top = weakest[:15]

    def pct(num, denom):
        return f"{(num / denom * 100):.1f}%" if denom else "n/a"

    now = datetime.now(timezone.utc)
    lines: list[str] = []
    lines.append(f"# Quality Snapshot {now.strftime('%Y-%m-%d')} — SOAK ({collection})")
    lines.append("")
    lines.append("> **Status:** synthetic-soak report.")
    try:
        rel_src = work_path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        rel_src = work_path
    lines.append(f"> Source: `{rel_src}`.")
    # Inspect what reranker was used (if any) from the work-file metadata.
    rerank_seen: set[str] = set()
    for r in rows:
        for q in (r.get("queries") or []):
            ret = q.get("retrieval") or {}
            rb = ret.get("rerank_backend")
            if rb:
                rerank_seen.add(rb)
    rerank_desc = (
        f"`{','.join(sorted(rerank_seen))}`" if rerank_seen
        else "(none — vector-rank only)"
    )
    resolved_gen_model = gen_model or (
        VLLM_GEN_DEFAULT_MODEL if gen_provider == "vllm" else GENERATOR_MODEL
    )
    gen_desc = (
        f"Dashscope `{resolved_gen_model}`" if gen_provider == "dashscope"
        else f"local vLLM `{resolved_gen_model}` (Phase 4c)"
    )
    resolved_judge_model = judge_model or (
        VLLM_GEN_DEFAULT_MODEL if judge_provider == "vllm" else JUDGE_MODEL
    )
    judge_desc = (
        f"Dashscope `{resolved_judge_model}`" if judge_provider == "dashscope"
        else f"local vLLM `{resolved_judge_model}` (GX10)"
    )
    lines.append(f"> Judge: {judge_desc}. Generator: {gen_desc}. Embedder: `{embed_model}` (provider={provider}). Collection: `{collection}`. Reranker: {rerank_desc}.")
    lines.append("> No QA threshold; this snapshot is informational.")
    lines.append("")
    lines.append("## 1. Corpus summary")
    lines.append("")
    lines.append(f"- Sampled chunks: **{n_chunks}** across {len(per_doc)} docs.")
    lines.append(f"- Queries generated: **{len(queries)}**.")
    lines.append(f"- Queries judged: **{judged}/{len(queries)}** ({pct(judged, len(queries))}).")
    lines.append("")
    lines.append("## 2. Headline metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Recall@1 (gold chunk_id is top-1) | {pct(recall_at_1_chunk, len(queries))} ({recall_at_1_chunk}/{len(queries)}) |")
    lines.append(f"| Recall@5 (gold chunk_id in top-5) | {pct(recall_at_5_chunk, len(queries))} ({recall_at_5_chunk}/{len(queries)}) |")
    lines.append(f"| Recall@5 (gold doc_id in top-5)   | {pct(recall_at_5_doc, len(queries))} ({recall_at_5_doc}/{len(queries)}) |")
    lines.append(f"| Relevance score                   | {pct(rel_sum, rel_max)} ({rel_sum}/{rel_max}) |")
    lines.append(f"| Format score                      | {pct(fmt_sum, fmt_max)} ({fmt_sum}/{fmt_max}) |")
    lines.append(f"| Faithfulness score                | {pct(fait_sum, fait_max)} ({fait_sum}/{fait_max}) |")
    lines.append("")
    lines.append("## 3. Per-document metrics")
    lines.append("")
    lines.append("| Doc | Queries | R@1 | R@5 (chunk) | R@5 (doc) | Relevance | Format | Faith |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for doc_dir in sorted(per_doc.keys()):
        s = per_doc[doc_dir]
        lines.append(
            f"| {doc_dir} | {s['queries']} | "
            f"{pct(s['r1_chunk'], s['queries'])} | "
            f"{pct(s['r5_chunk'], s['queries'])} | "
            f"{pct(s['r5_doc'], s['queries'])} | "
            f"{pct(s['rel_sum'], s['judged']*2)} | "
            f"{pct(s['fmt_sum'], s['judged']*2)} | "
            f"{pct(s['fait_sum'], s['judged']*2)} |"
        )
    lines.append("")
    lines.append("## 4. Weakest 15 (query, top-1) pairs — v2.10.x defect candidates")
    lines.append("")
    for total, entry in weakest_top:
        q = entry["query"]; r = entry["row"]
        retrieval = (q.get("retrieval") or {}).get("top_k") or []
        top1 = retrieval[0] if retrieval else {}
        jud = q.get("judgment") or {}
        lines.append(
            f"- **{q['query_id']}** total={total}/6 (r={jud.get('relevance')}, "
            f"f={jud.get('format')}, faith={jud.get('faithfulness')})"
        )
        lines.append(f"  - Query: {q['query_text']!r}")
        lines.append(f"  - Gold doc: `{r['doc_dir']}` (chunk `{r['gold_chunk_id']}`)")
        lines.append(f"  - Top-1: `{top1.get('source_file')}` p={top1.get('page_number')} score={top1.get('score')}")
        lines.append(f"  - Judge rationale: {jud.get('rationale')}")
    lines.append("")
    lines.append("## 5. Methodology")
    lines.append("")
    lines.append(f"- Sampled {n_chunks} text chunks (≥ {MIN_CHUNK_CHARS} chars, ≤ {int(MAX_CODE_RATIO*100)}% code-like lines, no advertisement keywords). Stratified across the 34-doc canonical corpus.")
    lines.append(f"- Each chunk → 2 queries generated by {gen_desc} (temperature 0.3).")
    lines.append(f"- Each query → top-{TOP_K} retrieved from `{collection}` via `{provider}` provider, model `{embed_model}`.")
    lines.append(f"- Each top-1 chunk → graded by `{resolved_judge_model}` (temperature 0.0) on relevance / format / faithfulness, each 0-2.")
    lines.append("- Gold passage is shown to the judge for context; the judge is instructed NOT to penalize a different-chunk same-document retrieval.")
    lines.append("")
    lines.append("## 6. Revision log")
    lines.append("")
    lines.append("| Date | Change |")
    lines.append("|---|---|")
    lines.append(f"| {now.strftime('%Y-%m-%d')} | Initial v2.10.0-rc1 soak snapshot. |")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  report: wrote {report_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=["sample", "generate", "retrieve", "judge", "report", "all"],
                        default="all")
    parser.add_argument("--n-chunks", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--work-path", default=str(DEFAULT_WORK_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--collection", default=None,
                        help="Qdrant collection to retrieve from. Provider-aware defaults: "
                             f"'{COLLECTION_DEFAULT_DASHSCOPE}' (dashscope), "
                             f"'{COLLECTION_DEFAULT_OLLAMA}' (ollama), "
                             f"'{COLLECTION_DEFAULT_OMLX}' (omlx).")
    parser.add_argument("--provider", default="dashscope", choices=["ollama", "dashscope", "omlx"],
                        help="Embedding provider for query-side (default: dashscope as of v2.11.0). "
                             "Must match how the target collection was built. "
                             "'omlx' (v2.13 P1 candidate) = local Qwen3-Embedding-8B-mxfp8 via "
                             "omlx-server (4096-dim, text-only).")
    parser.add_argument("--embed-model", default=None,
                        help="Query-side embedding model. Default 'text-embedding-v4' for dashscope; "
                             "'llava' for ollama; 'Qwen3-Embedding-8B-mxfp8' for omlx.")
    parser.add_argument("--omlx-url", default=os.environ.get("OMLX_URL", OMLX_DEFAULT_URL),
                        help=f"omlx embeddings endpoint (default: {OMLX_DEFAULT_URL})")
    parser.add_argument("--rerank-backend", default=None,
                        choices=["dashscope", "omlx", "null", None],
                        help="When set, the retrieve stage runs the Qdrant top-K candidates through "
                             "the named reranker before storing the top-N. Default off (pre-v2.12 "
                             "behavior). 'dashscope' = cloud gte-rerank; 'omlx' = local "
                             "gte-reranker-modernbert-base-mlx; 'null' = pass-through (debug).")
    parser.add_argument("--top-k-retrieve", type=int, default=None,
                        help="Number of Qdrant candidates the reranker sees per query. "
                             "Only used when --rerank-backend is set. Default 25 in that case; "
                             "ignored when reranker is off (top-K and top-N collapse to TOP_K=5).")
    parser.add_argument("--hybrid", action="store_true",
                        help="Use hybrid retrieval (dense + BM25 sparse + RRF fusion). "
                             "Requires the sparse side-collection populated via "
                             "scripts/ingest_bm25_sparse.py. Implies --rerank-backend if "
                             "no reranker is specified (defaults to omlx).")
    parser.add_argument("--sparse-collection", default="mmrag_v2_8__bm25_sparse",
                        help="Sparse side-collection for hybrid mode.")
    parser.add_argument("--bm25-index-path", default="tests/fixtures/bm25_index_v2_12.json",
                        help="Path to the BM25 index JSON for hybrid mode.")
    parser.add_argument("--hyde", action="store_true",
                        help="Use HyDE: generate a hypothetical answer via qwen-max and embed "
                             "that instead of the literal query. Adds ~1s + ~$0.001 per query "
                             "but improves retrieval when the question and answer have different "
                             "vocabulary.")
    parser.add_argument("--auto-intent-hyde", action="store_true",
                        help="v2.14 Phase 2 / v2.15 Phase 1: enable TARGETED HyDE — query-intent "
                             "classifier decides per-query whether HyDE fires (code + minority-"
                             "language intents only). Overrides --hyde to per-intent gating. "
                             "Implies --hybrid (the auto_intent_hyde knob lives on "
                             "retrieve_hybrid_reranked).")
    parser.add_argument("--hyde-provider", default="dashscope",
                        choices=["dashscope", "vllm"],
                        help="HyDE generation provider when --hyde or --auto-intent-hyde is set. "
                             "'dashscope' (default) = cloud qwen-max; 'vllm' = local GX10 endpoint "
                             "at $0/call but adds ~9s wall-clock per query.")
    # v2.14 Phase 4c: local-LLM query generation. Query generation is
    # NOT judging — leniency-trap rules don't apply. Safe to default to
    # vllm once the GX10 endpoint is the user's stable choice.
    parser.add_argument("--gen-provider", default="dashscope",
                        choices=["dashscope", "vllm"],
                        help="Query-generation provider (v2.14 Phase 4c). 'dashscope' (default) "
                             "uses cloud `qwen-max` ($). 'vllm' uses the local GX10 endpoint "
                             "($0). Generation isn't judging, so the leniency-trap rules don't "
                             "apply; safe to use the local path even on an all-RESTRICTED Phase 0 "
                             "verdict.")
    parser.add_argument("--gen-url", default=VLLM_GEN_DEFAULT_URL,
                        help=f"Override vLLM chat/completions URL (default: {VLLM_GEN_DEFAULT_URL}). "
                             "Only used when --gen-provider=vllm.")
    parser.add_argument("--gen-model", default=None,
                        help="Override generation model id. Default: qwen-max for dashscope, "
                             f"{VLLM_GEN_DEFAULT_MODEL} for vllm.")
    parser.add_argument("--judge-provider", choices=["dashscope", "vllm"],
                        default="dashscope",
                        help="LLM-as-a-Judge provider for relevance/format/faithfulness. "
                             "'dashscope' (default) uses cloud qwen-max. 'vllm' uses the local "
                             "GX10 endpoint ($0). Phase 0 calibration applies to vllm "
                             f"(current model: {VLLM_GEN_DEFAULT_MODEL}).")
    parser.add_argument("--judge-url", default=VLLM_GEN_DEFAULT_URL,
                        help=f"Override vLLM judge URL (default: {VLLM_GEN_DEFAULT_URL}). "
                             "Only used when --judge-provider=vllm.")
    parser.add_argument("--judge-model", default=None,
                        help="Override judge model id. Default: qwen-max for dashscope, "
                             f"{VLLM_GEN_DEFAULT_MODEL} for vllm.")
    parser.add_argument("--docs-root", type=Path, default=None,
                        help="Override the per-doc baseline root (default: "
                             "REPO_ROOT/output). Sample stage looks for "
                             "<docs-root>/<canonical_name>/ingestion.jsonl. "
                             "Set to output/v3_canonical/ for the V3 soak.")
    args = parser.parse_args()
    if args.docs_root is not None:
        global DOCS_ROOT
        DOCS_ROOT = args.docs_root.resolve()
        print(f"  docs-root override: {DOCS_ROOT}")

    work_path = Path(args.work_path)
    report_path = Path(args.report_path)
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    # Generation needs a dashscope key ONLY when --gen-provider=dashscope
    # (vllm path is local-LLM, $0, no api_key).
    generate_needs_key = (
        args.stage in ("generate", "all") and args.gen_provider == "dashscope"
    )
    judge_needs_key = (
        args.stage in ("judge", "all") and args.judge_provider == "dashscope"
    )
    needs_key = generate_needs_key or judge_needs_key or (
        args.stage in ("retrieve", "all") and args.provider == "dashscope"
    )
    if needs_key and not api_key:
        print("ERROR: DASHSCOPE_API_KEY env var is not set; required for "
              "judge, --gen-provider=dashscope, and --provider=dashscope retrieve.",
              file=sys.stderr)
        return 2

    # omlx provider needs MLX_API_KEY (for the local omlx-server, not Dashscope).
    omlx_api_key = os.environ.get("MLX_API_KEY", "").strip()
    if args.stage in ("retrieve", "all") and args.provider == "omlx" and not omlx_api_key:
        print("ERROR: MLX_API_KEY env var is not set; required for --provider omlx.",
              file=sys.stderr)
        return 2

    if args.collection is None:
        args.collection = (
            COLLECTION_DEFAULT_DASHSCOPE if args.provider == "dashscope"
            else COLLECTION_DEFAULT_OMLX if args.provider == "omlx"
            else COLLECTION_DEFAULT_OLLAMA
        )
    if args.embed_model is None:
        args.embed_model = (
            EMBED_MODEL_DASHSCOPE if args.provider == "dashscope"
            else EMBED_MODEL_OMLX if args.provider == "omlx"
            else EMBED_MODEL_OLLAMA
        )

    if args.stage in ("sample", "all"):
        print("[stage] sample")
        stage_sample(args.seed, args.n_chunks, work_path)
    if args.stage in ("generate", "all"):
        print("[stage] generate")
        stage_generate(
            work_path, api_key,
            gen_provider=args.gen_provider,
            gen_url=args.gen_url,
            gen_model=args.gen_model,
        )
    if args.stage in ("retrieve", "all"):
        print("[stage] retrieve")
        # Hybrid mode implies reranking; default reranker if user
        # didn't pass one explicitly.
        effective_rerank_backend = args.rerank_backend
        if args.hybrid and effective_rerank_backend is None:
            effective_rerank_backend = "omlx"
        top_k_retrieve_resolved = (
            args.top_k_retrieve if args.top_k_retrieve is not None
            else (25 if effective_rerank_backend else TOP_K)
        )
        embed_api_key = (
            api_key if args.provider == "dashscope"
            else omlx_api_key if args.provider == "omlx"
            else ""
        )
        stage_retrieve(
            work_path, args.qdrant_url, args.ollama_url,
            args.collection, args.provider, args.embed_model, embed_api_key,
            rerank_backend=effective_rerank_backend,
            top_k_retrieve=top_k_retrieve_resolved,
            top_n_return=TOP_K,
            hybrid=args.hybrid,
            sparse_collection=args.sparse_collection,
            bm25_index_path=args.bm25_index_path,
            use_hyde=args.hyde,
            auto_intent_hyde=args.auto_intent_hyde,
            hyde_provider=args.hyde_provider,
        )
    if args.stage in ("judge", "all"):
        print("[stage] judge")
        stage_judge(
            work_path, api_key,
            judge_provider=args.judge_provider,
            judge_url=args.judge_url,
            judge_model=args.judge_model,
        )
    if args.stage in ("report", "all"):
        print("[stage] report")
        stage_report(work_path, report_path,
                     args.collection, args.provider, args.embed_model,
                     gen_provider=args.gen_provider,
                     gen_model=args.gen_model,
                     judge_provider=args.judge_provider,
                     judge_model=args.judge_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
