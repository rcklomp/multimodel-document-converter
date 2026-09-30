"""Retrieval-only gold-chunk@10 over the full gold set against current production.

Reads output/v3_soak_code/work.jsonl (263 gold chunks / 514 queries, 29 docs), runs
the production hybrid path (omlx dense + bm25 sparse + RRF + ModernBERT rerank, hnsw_ef
512) per query, and reports whether the gold chunk lands in top-10. No generation/
judging (that's rag_answer_eval). Confirms the corpus expansion didn't regress the
original docs' retrieval. Prints aggregate + any docs below 0.80.
"""
from __future__ import annotations

import json
from collections import defaultdict

from mmrag_v2.retrieval import get_reranker, retrieve_hybrid_reranked

DENSE, SPARSE = "mmrag_v3__qwen3_local", "mmrag_v3__bm25_sparse"
rr = get_reranker("omlx")
rows = [json.loads(l) for l in open("output/v3_soak_code/work.jsonl") if l.strip()]

hit = tot = 0
per_doc = defaultdict(lambda: [0, 0])  # doc_id -> [hits, total]
for r in rows:
    gid = r.get("gold_chunk_id")
    doc = r.get("gold_doc_id") or (gid or "").split("_")[0]
    for q in r.get("queries", []):
        qt = q if isinstance(q, str) else (q.get("query_text") or q.get("query") or q.get("text") or "")
        if not qt:
            continue
        try:
            hits = retrieve_hybrid_reranked(
                qt, dense_collection=DENSE, sparse_collection=SPARSE,
                top_k_retrieve=50, top_n_return=10, rrf_weights=(1.0, 0.25),
                embed_provider="omlx", reranker=rr)
        except Exception as e:  # noqa: BLE001
            print(f"! {gid}: {e}")
            continue
        ids = [(h.get("payload") or {}).get("chunk_id") for h in hits]
        ok = gid in ids
        hit += ok
        tot += 1
        per_doc[doc][0] += ok
        per_doc[doc][1] += 1

print(f"\nGOLD_CHUNK@10 (full gold set, current production): {hit}/{tot} = {hit/tot:.3f}")
weak = sorted(((h / t, d, h, t) for d, (h, t) in per_doc.items() if t and h / t < 0.80))
print(f"docs below 0.80 ({len(weak)} of {len(per_doc)}):")
for rate, d, h, t in weak:
    print(f"  {d}: {h}/{t} = {rate:.2f}")
