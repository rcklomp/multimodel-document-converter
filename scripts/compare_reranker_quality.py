#!/usr/bin/env python3
"""v2.12 Phase 1 pre-work — side-by-side reranker quality comparison.

For each of the 20 v2.11 retrieval-regression queries:
  1. Embed query, fetch top-K candidates from Qdrant (production collection).
  2. Send (query, K-candidates) to both rerankers:
     - Cloud:  Dashscope `gte-rerank` (intl endpoint)
     - Local:  omlx `mlx-community_Qwen3-Reranker-8B-mxfp8` on 10.0.10.246:8000
  3. Capture top-5 indices + relevance scores from each.
  4. Compute per-query overlap metrics + corpus-wide agreement.

The candidate set is identical across the two rerankers (same embed +
Qdrant search), so disagreement comes from the reranker model + its
scoring head, not from upstream variance.

Quality signals reported:
  - top_1_agreement      fraction of queries where both rerankers picked
                         the same top-1 (high = the two models agree on
                         the strongest signal)
  - top_5_jaccard_mean   mean |A∩B|/|A∪B| over top-5 sets per query
                         (0.0 = no overlap, 1.0 = identical top-5)
  - top_5_overlap_count  histogram: how many queries had N items in
                         common between the two top-5 sets

Usage:
    DASHSCOPE_API_KEY=... MLX_API_KEY=... \\
        python scripts/compare_reranker_quality.py \\
        --output-json tests/fixtures/reranker_quality_2026-05-21.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from ingest_to_qdrant import embed_text_dashscope  # noqa: E402
from search_qdrant import search  # noqa: E402
from measure_reranker_latency import (  # noqa: E402
    QUERIES,
    rerank_call,
    DEFAULT_RERANK_URL_DASHSCOPE,
    DEFAULT_RERANK_MODEL_DASHSCOPE,
    DEFAULT_RERANK_URL_OMLX,
    DEFAULT_RERANK_MODEL_OMLX,
)

# Optional override for the local model — by default uses what's in
# measure_reranker_latency.DEFAULT_RERANK_MODEL_OMLX; pass --local-model
# to compare a different one.
LOCAL_MODEL_OVERRIDE: str | None = None


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collection", default="mmrag_v2_8__qwen3_dashscope")
    parser.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--top-k-retrieve", type=int, default=25)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--max-chunk-chars", type=int, default=1500)
    parser.add_argument("--embed-model", default="text-embedding-v4")
    parser.add_argument("--local-model", default=None,
                        help="Override the local reranker model name "
                             f"(default: {DEFAULT_RERANK_MODEL_OMLX}).")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()
    local_model = args.local_model or DEFAULT_RERANK_MODEL_OMLX

    dashscope_key = os.environ.get("DASHSCOPE_API_KEY", "")
    mlx_key = os.environ.get("MLX_API_KEY", "")
    if not dashscope_key:
        print("ERROR: DASHSCOPE_API_KEY required (embedding + cloud reranker).", file=sys.stderr)
        return 2
    if not mlx_key:
        print("ERROR: MLX_API_KEY required (local reranker).", file=sys.stderr)
        return 2

    print(f"=== Reranker quality comparison ===")
    print(f"Collection:        {args.collection}")
    print(f"top_k_retrieve:    {args.top_k_retrieve}")
    print(f"top_n returned:    {args.top_n}")
    print(f"Cloud reranker:    {DEFAULT_RERANK_MODEL_DASHSCOPE} @ Dashscope intl")
    print(f"Local reranker:    {local_model} @ {DEFAULT_RERANK_URL_OMLX}")
    print()

    per_query: list[dict] = []
    for qid, qtext in QUERIES:
        print(f"--- {qid} ---")
        # 1. Embed + Qdrant top-K (single call; both rerankers see the same candidates).
        vec = embed_text_dashscope(qtext, args.embed_model, dashscope_key)
        cands = search(vec, args.collection, limit=args.top_k_retrieve, qdrant_url=args.qdrant_url)
        docs = [((c.get("payload") or {}).get("content") or "")[:args.max_chunk_chars]
                for c in cands]
        doc_ids = [(c.get("payload") or {}).get("doc_id") for c in cands]
        chunk_ids = [(c.get("payload") or {}).get("chunk_id") or str(c.get("id")) for c in cands]

        # 2. Cloud rerank.
        try:
            t0 = time.perf_counter()
            cloud_results, cloud_elapsed = rerank_call(
                qtext, docs, dashscope_key,
                DEFAULT_RERANK_URL_DASHSCOPE, DEFAULT_RERANK_MODEL_DASHSCOPE,
                top_n=args.top_n,
            )
            cloud_top = [r.get("index") for r in cloud_results[:args.top_n]]
            cloud_scores = [r.get("relevance_score") for r in cloud_results[:args.top_n]]
        except Exception as e:
            print(f"    ! cloud rerank failed: {e}", file=sys.stderr)
            continue

        # 3. Local rerank.
        try:
            local_results, local_elapsed = rerank_call(
                qtext, docs, mlx_key,
                DEFAULT_RERANK_URL_OMLX, local_model,
                top_n=args.top_n,
            )
            local_top = [r.get("index") for r in local_results[:args.top_n]]
            local_scores = [r.get("relevance_score") for r in local_results[:args.top_n]]
        except Exception as e:
            print(f"    ! local rerank failed: {e}", file=sys.stderr)
            continue

        # 4. Compute overlap.
        cloud_set = set(cloud_top)
        local_set = set(local_top)
        overlap = len(cloud_set & local_set)
        top1_agree = cloud_top[0] == local_top[0] if cloud_top and local_top else False
        # Chunk-level overlap (in case of duplicate doc_ids in the candidate set;
        # use chunk_id for the real apples-to-apples).
        cloud_chunks = {chunk_ids[i] for i in cloud_top if i is not None and i < len(chunk_ids)}
        local_chunks = {chunk_ids[i] for i in local_top if i is not None and i < len(chunk_ids)}

        print(f"  cloud  top-5 idx={cloud_top}  scores={[round(s or 0.0,3) for s in cloud_scores]}  ({cloud_elapsed*1000:.0f}ms)")
        print(f"  local  top-5 idx={local_top}  scores={[round(s or 0.0,3) for s in local_scores]}  ({local_elapsed*1000:.0f}ms)")
        print(f"  overlap: {overlap}/5  top-1 agree: {top1_agree}  jaccard: {jaccard(cloud_set, local_set):.2f}")

        per_query.append({
            "query_id": qid,
            "query_text": qtext,
            "candidate_count": len(cands),
            "cloud_top_indices": cloud_top,
            "cloud_top_scores": cloud_scores,
            "cloud_elapsed_seconds": cloud_elapsed,
            "local_top_indices": local_top,
            "local_top_scores": local_scores,
            "local_elapsed_seconds": local_elapsed,
            "top1_agree": top1_agree,
            "top5_overlap_count": overlap,
            "top5_jaccard": jaccard(cloud_set, local_set),
            "cloud_top5_chunk_ids": list(cloud_chunks),
            "local_top5_chunk_ids": list(local_chunks),
        })

    # Aggregate.
    n = len(per_query)
    if not n:
        print("\nERROR: no queries scored", file=sys.stderr)
        return 1
    top1_agreement = sum(1 for r in per_query if r["top1_agree"]) / n
    jaccards = [r["top5_jaccard"] for r in per_query]
    overlap_hist = {i: sum(1 for r in per_query if r["top5_overlap_count"] == i) for i in range(6)}
    cloud_latencies = [r["cloud_elapsed_seconds"] for r in per_query]
    local_latencies = [r["local_elapsed_seconds"] for r in per_query]

    print()
    print("=" * 78)
    print(f"Reranker quality comparison ({n} queries, top-{args.top_n} of top-{args.top_k_retrieve})")
    print("=" * 78)
    print(f"top-1 agreement rate:   {top1_agreement*100:.1f}% ({int(top1_agreement*n)}/{n} queries)")
    print(f"top-5 mean Jaccard:     {mean(jaccards):.3f}")
    print(f"top-5 overlap histogram (queries with N items in common):")
    for i in range(6):
        bar = '#' * overlap_hist[i]
        print(f"  {i}/5: {overlap_hist[i]:>2}  {bar}")
    print()
    print(f"Cloud reranker latency:  mean={mean(cloud_latencies)*1000:.0f}ms  max={max(cloud_latencies)*1000:.0f}ms")
    print(f"Local reranker latency:  mean={mean(local_latencies)*1000:.0f}ms  max={max(local_latencies)*1000:.0f}ms")
    print(f"Local/Cloud ratio:       {mean(local_latencies)/mean(cloud_latencies):.1f}×")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "collection": args.collection,
            "top_k_retrieve": args.top_k_retrieve,
            "top_n": args.top_n,
            "cloud_model": DEFAULT_RERANK_MODEL_DASHSCOPE,
            "local_model": local_model,
            "summary": {
                "queries_scored": n,
                "top1_agreement_rate": top1_agreement,
                "top5_jaccard_mean": mean(jaccards),
                "top5_overlap_histogram": overlap_hist,
                "cloud_latency_mean_seconds": mean(cloud_latencies),
                "cloud_latency_max_seconds": max(cloud_latencies),
                "local_latency_mean_seconds": mean(local_latencies),
                "local_latency_max_seconds": max(local_latencies),
                "local_to_cloud_latency_ratio": mean(local_latencies)/mean(cloud_latencies),
            },
            "per_query": per_query,
        }, indent=2))
        print(f"\nFull data written to {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
