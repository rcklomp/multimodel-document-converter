#!/usr/bin/env python3
"""v2.14 retrieval-regression harness — hybrid + rerank (omlx embedder).

Pins the v2.14.0 production retrieval shape against the live stack
(local `Qwen3-Embedding-8B-mxfp8` via omlx-server +
`mmrag_v2_8__qwen3_local` dense + `mmrag_v2_8__bm25_sparse` sparse +
ModernBERT rerank via omlx-server).

Same 20-query workload as v2.11 / v2.12 fingerprints. The three
fingerprints coexist as cycle-close archaeology:

  retrieval_regression_v2_11_qwen3.json   v2.11.0 (dense + Qdrant only)
  retrieval_regression_v2_12_hybrid.json  v2.12.0 (hybrid + rerank, dashscope)
  retrieval_regression_v2_14_hybrid.json  v2.14.0 (hybrid + rerank, omlx)

The v2.12 fingerprint stays pinned to dashscope (contract for that
release). The v2.14 fingerprint pins the new omlx-based production
shape. Both regression tests run on the live stack pre-tag; v2.14 is
the active production gate.

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


FIXTURE_PATH_DEFAULT = REPO_ROOT / "tests" / "fixtures" / "retrieval_regression_v2_14_hybrid.json"
DENSE_COLLECTION = "mmrag_v2_8__qwen3_local"
SPARSE_COLLECTION = "mmrag_v2_8__bm25_sparse"
BM25_INDEX = "tests/fixtures/bm25_index_v2_12.json"
EMBED_PROVIDER = "omlx"
EMBED_MODEL = "Qwen3-Embedding-8B-mxfp8"
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
        embed_provider=EMBED_PROVIDER,
        embed_model=EMBED_MODEL,
        reranker_backend="omlx",
        use_hyde=False,
    )


def capture(fixture_path: Path) -> None:
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {
        "schema": 1,
        "engine_version": "2.13.0",
        "pipeline": "hybrid+rerank",
        "dense_collection": DENSE_COLLECTION,
        "sparse_collection": SPARSE_COLLECTION,
        "bm25_index": BM25_INDEX,
        "embed_provider": EMBED_PROVIDER,
        "embed_model": EMBED_MODEL,
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
    print(f"\nWrote v2.14 fingerprint: {fixture_path}")
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
        print(f"ERROR: fixture missing — run --capture first ({fixture_path})", file=sys.stderr)
        return 2
    expected = json.loads(fixture_path.read_text())
    if expected.get("engine_version") != "2.13.0":
        print(f"ERROR: fixture engine_version != 2.13.0", file=sys.stderr)
        return 2

    pass_count = 0
    fail_count = 0
    for q in expected["queries"]:
        qid = q["id"]
        qtext = q["text"]
        expected_top = q["top_k"]
        actual_hits = _run_query(qtext)
        actual_top = [_summarize(h) for h in actual_hits[:TOP_N_RETURN]]
        # Strict check: top-STRICT_K chunk_ids must match exactly in order.
        expected_ids = [r["chunk_id"] for r in expected_top[:STRICT_K]]
        actual_ids = [r["chunk_id"] for r in actual_top[:STRICT_K]]
        if expected_ids == actual_ids:
            pass_count += 1
            print(f"  PASS {qid}")
        else:
            fail_count += 1
            print(f"  FAIL {qid} — {qtext!r}")
            print(f"    expected top-{STRICT_K}: {expected_ids}")
            print(f"    actual   top-{STRICT_K}: {actual_ids}")
            print(f"    expected top1: {_summarize_top1(expected_top)}")
            print(f"    actual   top1: {_summarize_top1(actual_top)}")
    total = pass_count + fail_count
    print(f"\nv2.14 hybrid regression: {pass_count}/{total} PASS")
    return 0 if fail_count == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", action="store_true",
                   help="Capture a new fingerprint (overwrites existing).")
    p.add_argument("--fixture", default=str(FIXTURE_PATH_DEFAULT),
                   help="Path to the JSON fingerprint file.")
    args = p.parse_args()
    fixture_path = Path(args.fixture)
    if args.capture:
        capture(fixture_path)
        return 0
    return verify(fixture_path)


if __name__ == "__main__":
    raise SystemExit(main())
