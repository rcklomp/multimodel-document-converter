#!/usr/bin/env python3
"""v2.12 retrieval-regression harness — hybrid + rerank.

Pins the v2.12.0 production retrieval shape against the live stack
(Dashscope text-embedding-v4 + `mmrag_v2_8__qwen3_dashscope` dense +
`mmrag_v2_8__bm25_sparse` sparse + ModernBERT rerank via omlx-server).

Same 20-query workload as the v2.11 fingerprint, but the captured
top-5 reflects the FULL production pipeline rather than dense-only.
The two fingerprints coexist:

  retrieval_regression_v2_11_qwen3.json   v2.11.0 (dense + Qdrant only)
  retrieval_regression_v2_12_hybrid.json  v2.12.0 (hybrid + rerank)

Both regression tests must pass on the live stack pre-tag.

Two modes:

  --capture     run the full pipeline, write the new fingerprint
  default       verify against the tracked fingerprint
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Reuse the 20-query workload from the v2.11 regression harness.
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_rr_mod", SCRIPTS / "retrieval_regression.py"
)
_rr_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rr_mod)
QUERIES = _rr_mod.QUERIES


FIXTURE_PATH_DEFAULT = REPO_ROOT / "tests" / "fixtures" / "retrieval_regression_v2_12_hybrid.json"
DENSE_COLLECTION = "mmrag_v2_8__qwen3_dashscope"
SPARSE_COLLECTION = "mmrag_v2_8__bm25_sparse"
BM25_INDEX = "tests/fixtures/bm25_index_v2_12.json"
TOP_K_RETRIEVE = 25
TOP_N_RETURN = 5
STRICT_K = 3


def _summarize(hit: dict) -> dict:
    payload = hit.get("payload") or {}
    return {
        "chunk_id": payload.get("chunk_id") or str(hit.get("id")),
        "doc_id": payload.get("doc_id"),
        "source_file": payload.get("source_file"),
        "modality": payload.get("modality"),
        "page_number": payload.get("page_number"),
        "score": round(float(hit.get("score") or 0.0), 6),
        "rerank_score": round(float(hit.get("rerank_score") or 0.0), 6),
    }


def _run_query(query_text: str) -> list[dict]:
    from mmrag_v2.retrieval import retrieve_hybrid_reranked
    return retrieve_hybrid_reranked(
        query=query_text,
        dense_collection=DENSE_COLLECTION,
        sparse_collection=SPARSE_COLLECTION,
        bm25_index_path=BM25_INDEX,
        top_k_retrieve=TOP_K_RETRIEVE,
        top_n_return=TOP_N_RETURN,
        # Pin to v2.12.0 contract: cloud Dashscope text-embedding-v4
        # against the 1024-dim dashscope collection. (v2.13.0 flipped
        # the library defaults to omlx + 4096-dim collection; this
        # script preserves the v2.12 release shape as archaeology.)
        embed_provider="dashscope",
        embed_model="text-embedding-v4",
        reranker_backend="omlx",
        use_hyde=False,
    )


def capture(fixture_path: Path) -> None:
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {
        "schema": 1,
        "engine_version": "2.12.0",
        "pipeline": "hybrid+rerank",
        "dense_collection": DENSE_COLLECTION,
        "sparse_collection": SPARSE_COLLECTION,
        "bm25_index": BM25_INDEX,
        "embed_model": "text-embedding-v4",
        "reranker_model": "gte-reranker-modernbert-base-mlx",
        "top_k_retrieve": TOP_K_RETRIEVE,
        "top_n_return": TOP_N_RETURN,
        "strict_k": STRICT_K,
        "queries": [],
    }
    for qid, qtext in QUERIES:
        print(f"  capture {qid:14s} {qtext!r}", flush=True)
        hits = _run_query(qtext)
        out["queries"].append({
            "id": qid,
            "text": qtext,
            "top_k": [_summarize(h) for h in hits[:TOP_N_RETURN]],
        })
    fixture_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nWrote v2.12 fingerprint: {fixture_path}")
    print(f"Queries captured:  {len(out['queries'])}")


def _summarize_top1(entry_top: list[dict]) -> str:
    if not entry_top:
        return "<empty>"
    r = entry_top[0]
    src = (r.get("source_file") or "").split("/")[-1][:40]
    return (f"rerank={r.get('rerank_score'):.3f} dense={r.get('score'):.3f} "
            f"doc={r.get('doc_id')} p={r.get('page_number')} src={src!r}")


def verify(fixture_path: Path) -> int:
    if not fixture_path.exists():
        print(f"ERROR: fingerprint missing at {fixture_path}; run with --capture first",
              file=sys.stderr)
        return 2
    baseline = json.loads(fixture_path.read_text())
    failures: list[str] = []
    drifts: list[str] = []
    rows: list[str] = []

    for entry in baseline["queries"]:
        qid = entry["id"]
        qtext = entry["text"]
        baseline_top = entry["top_k"]
        baseline_chunk_ids = [r["chunk_id"] for r in baseline_top]
        baseline_top1_doc = baseline_top[0]["doc_id"] if baseline_top else None

        hits = _run_query(qtext)
        current_top = [_summarize(h) for h in hits[:TOP_N_RETURN]]
        current_chunk_ids = [r["chunk_id"] for r in current_top]
        current_top1_doc = current_top[0]["doc_id"] if current_top else None

        strict_hit = current_chunk_ids[:STRICT_K] == baseline_chunk_ids[:STRICT_K]
        loose_hit = current_top1_doc == baseline_top1_doc

        if not loose_hit:
            status = "FAIL"
            failures.append(
                f"{qid}: top-1 doc_id changed\n"
                f"  baseline: {baseline_top1_doc} ({_summarize_top1(baseline_top)})\n"
                f"  current : {current_top1_doc} ({_summarize_top1(current_top)})"
            )
        elif not strict_hit:
            status = "DRIFT"
            drifts.append(
                f"{qid}: top-{STRICT_K} chunk_ids reshuffled (top-1 doc stable)\n"
                f"  baseline: {baseline_chunk_ids[:STRICT_K]}\n"
                f"  current : {current_chunk_ids[:STRICT_K]}"
            )
        else:
            status = "PASS"
        rows.append(f"  {qid:14s} {status:5s} {_summarize_top1(current_top)}")

    print("=" * 78)
    print(f"v2.12 retrieval regression (hybrid+rerank, top-{TOP_N_RETURN}, strict={STRICT_K})")
    print(f"Baseline: {fixture_path}")
    print("=" * 78)
    for row in rows:
        print(row)
    print()
    if drifts:
        print(f"DRIFT (top-{STRICT_K} chunk_ids reshuffled, top-1 doc stable) — {len(drifts)} queries:")
        for d in drifts:
            print(f"  - {d}")
        print()
    if failures:
        print(f"FAILURES (top-1 doc_id changed) — {len(failures)} queries:")
        for f in failures:
            print(f"  - {f}")
        print("\nRETRIEVAL_REGRESSION_V2_12: FAIL")
        return 1
    if drifts:
        print("RETRIEVAL_REGRESSION_V2_12: PASS_WITH_DRIFT")
        return 0
    print("RETRIEVAL_REGRESSION_V2_12: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--capture", action="store_true",
                        help="capture fingerprint (overwrites the target fixture)")
    parser.add_argument("--fixture", default=str(FIXTURE_PATH_DEFAULT))
    args = parser.parse_args()
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("ERROR: DASHSCOPE_API_KEY env var required", file=sys.stderr)
        return 2
    if not os.environ.get("MLX_API_KEY"):
        print("ERROR: MLX_API_KEY env var required (for omlx reranker)", file=sys.stderr)
        return 2
    fixture_path = Path(args.fixture)
    if args.capture:
        capture(fixture_path)
        return 0
    return verify(fixture_path)


if __name__ == "__main__":
    raise SystemExit(main())
