#!/usr/bin/env python3
"""Retrieval-config sweep: which config puts the gold chunk in the top-5/10?

Fully local (oMLX embed + Qdrant + oMLX rerank, no cloud, no LLM judge). Retrieval
recall against the known gold chunk needs no judge. Reuses the production pipeline
(retrieve_reranked) so the result is faithful, not a reimplementation.

Reads the soak work.jsonl (queries + gold_chunk_id). For each config, retrieves +
reranks and records the gold chunk's rank. Resumable: appends per (config,query).

Usage: python scripts/retrieval_sweep.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mmrag_v2.retrieval import get_reranker, retrieve_reranked  # noqa: E402

WORK = Path("output/v3_soak_code/work.jsonl")
OUT = Path("output/v3_soak_code/retrieval_sweep.jsonl")
COLL = "mmrag_v3__qwen3_local"
EMBED_MODEL = "Qwen3-Embedding-8B-mxfp8"
RETURN_N = 10

# config label -> top_k_retrieve (candidates the reranker sees)
CONFIGS = {"dense_rr25": 25, "dense_rr50": 50, "dense_rr100": 100}


def _queries():
    for r in (json.loads(line) for line in WORK.open(encoding="utf-8") if line.strip()):
        for q in r.get("queries", []):
            if q.get("query_text"):
                yield r["gold_chunk_id"], q["query_id"], q["query_text"]


def run() -> None:
    key = os.environ.get("MLX_API_KEY", "").strip()
    if not key:
        print("ERROR: MLX_API_KEY not set", file=sys.stderr)
        raise SystemExit(2)
    reranker = get_reranker("omlx")
    done = set()
    if OUT.exists():
        for line in OUT.open(encoding="utf-8"):
            try:
                d = json.loads(line)
                done.add((d["config"], d["query_id"]))
            except Exception:
                pass
    qs = list(_queries())
    with OUT.open("a", encoding="utf-8") as fh:
        for label, topk in CONFIGS.items():
            n = 0
            for gold, qid, qt in qs:
                if (label, qid) in done:
                    continue
                try:
                    hits = retrieve_reranked(
                        qt,
                        collection=COLL,
                        top_k_retrieve=topk,
                        top_n_return=RETURN_N,
                        embed_provider="omlx",
                        embed_model=EMBED_MODEL,
                        embed_api_key=key,
                        reranker=reranker,
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"  ! {label} {qid}: {e}", file=sys.stderr)
                    continue
                ids = [(h.get("payload") or {}).get("chunk_id") for h in hits]
                rank = ids.index(gold) + 1 if gold in ids else 9999
                fh.write(json.dumps({"config": label, "query_id": qid, "rank": rank}) + "\n")
                fh.flush()
                n += 1
                if n % 50 == 0:
                    print(f"  {label}: {n} done", flush=True)
            print(f"{label}: complete", flush=True)
    report()


def report() -> None:
    by = {}
    for line in OUT.open(encoding="utf-8"):
        d = json.loads(line)
        by.setdefault(d["config"], []).append(d["rank"])
    print("\n=== RETRIEVAL SWEEP (gold-chunk recall after rerank, top-10) ===")
    print(f"{'config':<12} | {'n':>4} | {'R@1':>6} | {'R@5':>6} | {'R@10':>6}")
    for label in CONFIGS:
        rs = by.get(label, [])
        if not rs:
            continue

        def at(k):
            return 100 * sum(1 for x in rs if x <= k) / len(rs)

        print(f"{label:<12} | {len(rs):>4} | {at(1):>5.1f}% | {at(5):>5.1f}% | {at(10):>5.1f}%")


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        report()
    else:
        run()
