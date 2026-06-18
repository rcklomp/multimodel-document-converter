"""One-shot: ingest Devlin + all converted backlog code books into PRODUCTION,
rebuild sparse, validate retrieval. Idempotent (skips doc_ids already in dense).

DRY RUN (default, read-only, no Qdrant writes): prints a readiness table per book
(doc_id, chunks, R3 + verdict, prod status, jsonl present). Use this to confirm
everything is staged while the production-write permission is unavailable.

REAL RUN: set INGEST_DO_IT=1. Per book: copy its jsonl into output/phase5_reextract/
(so the sparse rebuild indexes it), ingest dense (omlx), then after all books rebuild
sparse (phase5_ingest_bm25) and run a code-query retrieval spot-check. The dense
ingest + sparse rebuild are the only steps that need the Qdrant-write permission.

Run:
  python scripts/_ingest_all_codebooks.py            # dry run (read-only)
  INGEST_DO_IT=1 python scripts/_ingest_all_codebooks.py   # real (needs write perm)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, "scripts")
import _code_quality as cq  # noqa: E402

PYBIN = "/Users/Shared/miniforge3/envs/mmrag-v2/bin/python"
QDRANT = "http://localhost:6333"
DENSE = "mmrag_v3__qwen3_local"
REEXTRACT = Path("output/phase5_reextract")
DO_IT = os.environ.get("INGEST_DO_IT") == "1"

# Devlin's reextract dir is already updated with the Fix-b output; backlog books are
# auto-discovered from output/backlog/.
DEVLIN_DIR = REEXTRACT / "Devlin_M_Building_LLM_Agents_with_RAG_Knowledge_"


def _http(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{QDRANT}{path}", data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


def prod_doc_ids() -> set:
    ids, off = set(), None
    while True:
        b = {"limit": 1000, "with_payload": ["doc_id"], "with_vector": False}
        if off:
            b["offset"] = off
        r = _http("POST", f"/collections/{DENSE}/points/scroll", b)["result"]
        for p in r["points"]:
            ids.add((p.get("payload") or {}).get("doc_id"))
        off = r.get("next_page_offset")
        if not off:
            break
    return ids


def book_dirs() -> list:
    dirs = []
    if (DEVLIN_DIR / "ingestion.jsonl").exists():
        dirs.append(DEVLIN_DIR)
    for d in sorted(Path("output/backlog").glob("*")):
        if (d / "ingestion.jsonl").exists():
            dirs.append(d)
    return dirs


def book_stats(jsonl: Path):
    rows = [json.loads(l) for l in open(jsonl)]
    hdr = rows[0]
    m = cq.code_quality([{"content": d.get("content") or "",
                          "modality": (d.get("metadata") or {}).get("modality") or d.get("modality"),
                          "metadata": d.get("metadata") or {}} for d in rows])
    return hdr, len(rows), m


def ingest_dense(jsonl: Path) -> bool:
    r = subprocess.run([PYBIN, "scripts/ingest_to_qdrant.py", str(jsonl),
                        "--provider", "omlx", "--collection", DENSE, "--qdrant-url", QDRANT],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    INGEST FAIL rc={r.returncode}: {r.stderr[-400:]}", flush=True)
    return r.returncode == 0


def main() -> int:
    prod = prod_doc_ids()
    dirs = book_dirs()
    print(f"{'BOOK':52s} {'doc_id':14s} {'chunks':>6} {'R3':>6} {'verdict':12s} {'prod?':8s}")
    plan = []
    for d in dirs:
        jl = d / "ingestion.jsonl"
        hdr, n, m = book_stats(jl)
        did = hdr.get("doc_id")
        verdict = cq.gate_verdict(m).upper()
        in_prod = did in prod
        print(f"{d.name[:52]:52s} {str(did):14s} {n:6d} {m.indentation_fidelity:6.3f} {verdict:12s} "
              f"{'IN_PROD' if in_prod else 'NEW':8s}")
        if not in_prod:
            plan.append((d, did))
    print(f"\nready to ingest (NEW): {len(plan)} book(s); already in prod: {len(dirs)-len(plan)}")

    if not DO_IT:
        print("\nDRY RUN (read-only). Set INGEST_DO_IT=1 to ingest + rebuild sparse.")
        return 0

    # REAL RUN (needs Qdrant-write permission)
    for d, did in plan:
        dest = REEXTRACT / d.name
        if d != dest:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(d / "ingestion.jsonl", dest / "ingestion.jsonl")
            if (d / "assets").exists():
                if (dest / "assets").exists():
                    shutil.rmtree(dest / "assets")
                shutil.copytree(d / "assets", dest / "assets")
        print(f"  ingesting {d.name} ({did}) ...", flush=True)
        ingest_dense(dest / "ingestion.jsonl")
    print("  rebuilding sparse ...", flush=True)
    subprocess.run([PYBIN, "scripts/phase5_ingest_bm25.py", "--qdrant-url", QDRANT])
    print("INGEST_ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
