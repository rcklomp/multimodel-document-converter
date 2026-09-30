#!/usr/bin/env python3
"""Dense vs hybrid (dense+BM25+RRF) retrieval recall, fully local. Resumable.

For each soak query, retrieves the gold chunk under (a) dense + rerank and (b) hybrid
(dense + BM25 + RRF + rerank), and records the gold chunk's rank under each. Recall
against the known gold needs no LLM. Appends per query to hybrid_compare.jsonl.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mmrag_v2.retrieval import (  # noqa: E402
    get_reranker,
    retrieve_hybrid_reranked,
    retrieve_reranked,
)

WORK = Path("output/v3_soak_code/work.jsonl")
OUT = Path("output/v3_soak_code/hybrid_compare.jsonl")
DENSE = "mmrag_v3__qwen3_local"
SPARSE = "mmrag_v3__bm25_sparse"
IDX = "tests/fixtures/bm25_index_v3.json"
EM = "Qwen3-Embedding-8B-mxfp8"


def _rank(hits: list[dict], gold: str) -> int:
    ids = [(h.get("payload") or {}).get("chunk_id") for h in hits]
    return ids.index(gold) + 1 if gold in ids else 9999


def main() -> None:
    key = os.environ["MLX_API_KEY"]
    rr = get_reranker("omlx")
    done = set()
    if OUT.exists():
        for line in OUT.open():
            try:
                done.add(json.loads(line)["qid"])
            except Exception:
                pass
    rows = [json.loads(line) for line in WORK.open() if line.strip()]
    i = 0
    with OUT.open("a") as fh:
        for r in rows:
            gold = r["gold_chunk_id"]
            for q in r.get("queries", []):
                qt, qid = q.get("query_text"), q.get("query_id")
                if not qt or qid in done:
                    continue
                try:
                    d = retrieve_reranked(
                        qt,
                        collection=DENSE,
                        top_k_retrieve=50,
                        top_n_return=10,
                        embed_provider="omlx",
                        embed_model=EM,
                        embed_api_key=key,
                        reranker=rr,
                    )
                    h = retrieve_hybrid_reranked(
                        qt,
                        dense_collection=DENSE,
                        sparse_collection=SPARSE,
                        bm25_index_path=IDX,
                        top_k_retrieve=50,
                        top_n_fuse=50,
                        top_n_return=10,
                        embed_provider="omlx",
                        embed_model=EM,
                        embed_api_key=key,
                        reranker=rr,
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"! {qid}: {e}", file=sys.stderr)
                    continue
                fh.write(
                    json.dumps({"qid": qid, "dense": _rank(d, gold), "hybrid": _rank(h, gold)})
                    + "\n"
                )
                fh.flush()
                i += 1
                if i % 50 == 0:
                    print(f"  {i} done", flush=True)
    report()


def report() -> None:
    rs = [json.loads(line) for line in OUT.open() if line.strip()]

    def at(k: int, key: str) -> float:
        return 100 * sum(1 for x in rs if x[key] <= k) / len(rs)

    print(f"\nn={len(rs)}")
    print(
        "dense : R@1 %.1f  R@5 %.1f  R@10 %.1f" % (at(1, "dense"), at(5, "dense"), at(10, "dense"))
    )
    print(
        "HYBRID: R@1 %.1f  R@5 %.1f  R@10 %.1f"
        % (at(1, "hybrid"), at(5, "hybrid"), at(10, "hybrid"))
    )


if __name__ == "__main__":
    report() if "--report-only" in sys.argv else main()
