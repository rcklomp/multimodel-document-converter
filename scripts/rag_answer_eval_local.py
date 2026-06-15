#!/usr/bin/env python3
"""End-to-end RAG answer-quality confirmation, FULLY LOCAL (GX10 14B, direct).

Does the retrieval improvement translate to better ANSWERS? Two arms:
  A = dense + rerank, top-5            (the current baseline)
  B = hybrid w(dense=1,sparse=0.25), top-10  (German-safe hybrid + feed-more)
For each query: retrieve -> generate an answer from the passages -> judge the
answer's correctness vs the gold. Generation + judging on GX10 Qwen2.5-14B (local,
direct endpoint, no relay, no cloud). Recall/embeds via oMLX. Resumable.

Note: the local 14B judge is "directional" (weaker than qwen-max on rel/faith), so
the ABSOLUTE numbers shift vs the cloud run; the A-vs-B DELTA is what's trustworthy
(same judge both arms, bias cancels).
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mmrag_v2.retrieval import (
    get_reranker,
    retrieve_hybrid_reranked,
    retrieve_reranked,
)  # noqa: E402
from rag_answer_eval import _gen_messages, _judge_messages, _parse_score  # noqa: E402
from synthetic_soak import _call_vllm  # noqa: E402

WORK = Path("output/v3_soak_code/work.jsonl")
OUT = Path("output/v3_soak_code/answers_local.jsonl")
REPORT = Path("output/v3_soak_code/answer_local_report.md")
DENSE = "mmrag_v3__qwen3_local"
SPARSE = "mmrag_v3__bm25_sparse"
IDX = "tests/fixtures/bm25_index_v3.json"
EM = "Qwen3-Embedding-8B-mxfp8"
GX10_URL = "http://10.0.10.239:8000/v1/chat/completions"
GX10_MODEL = "RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic"

GERMAN = {
    "CarOK voorraadtelling 2021-04",
    "ATZ.-.Design.und.Aerodynamik.bei.Nutzfahrzeugen",
    "Handbuch Entwicklungs- und Erziehungspsychologie",
    "Grundlagen Fahrzeug- und Motorentechnik",
    "ATZ.Elektronik.-.Effizientere.Software.Entwicklung.GERMAN.RETAiL.eBOOk-PDFWriters",
}


def _gen_judge(query, passages, gold):
    ans = _call_vllm(GX10_URL, GX10_MODEL, _gen_messages(query, passages), max_tokens=400)
    if ans is None:
        return None, None
    jt = _call_vllm(GX10_URL, GX10_MODEL, _judge_messages(query, gold, ans), max_tokens=200)
    return ans, _parse_score(jt)


def run() -> None:
    key = os.environ["MLX_API_KEY"]
    rr = get_reranker("omlx")
    done = set()
    if OUT.exists():
        for line in OUT.open(encoding="utf-8"):
            try:
                d = json.loads(line)
                done.add((d["arm"], d["qid"]))
            except Exception:
                pass
    rows = [json.loads(line) for line in WORK.open(encoding="utf-8") if line.strip()]
    n = 0
    with OUT.open("a", encoding="utf-8") as fh:
        for r in rows:
            gold_id = r["gold_chunk_id"]
            gold = r.get("gold_content") or ""
            doc = r.get("doc_dir")
            for q in r.get("queries", []):
                qt, qid = q.get("query_text"), q.get("query_id")
                if not qt:
                    continue
                for arm in ("dense_top5", "dense_top10", "hybrid_top10"):
                    if (arm, qid) in done:
                        continue
                    try:
                        if arm.startswith("dense"):
                            hits = retrieve_reranked(
                                qt,
                                collection=DENSE,
                                top_k_retrieve=50,
                                top_n_return=(5 if arm == "dense_top5" else 10),
                                embed_provider="omlx",
                                embed_model=EM,
                                embed_api_key=key,
                                reranker=rr,
                            )
                        else:
                            hits = retrieve_hybrid_reranked(
                                qt,
                                dense_collection=DENSE,
                                sparse_collection=SPARSE,
                                bm25_index_path=IDX,
                                top_k_retrieve=50,
                                top_n_fuse=50,
                                top_n_return=10,
                                rrf_weights=(1.0, 0.25),
                                embed_provider="omlx",
                                embed_model=EM,
                                embed_api_key=key,
                                reranker=rr,
                            )
                    except Exception as e:  # noqa: BLE001
                        print(f"! retrieve {arm} {qid}: {e}", file=sys.stderr)
                        continue
                    ids = [(h.get("payload") or {}).get("chunk_id") for h in hits]
                    passages = [(h.get("payload") or {}).get("content") or "" for h in hits]
                    ans, score = _gen_judge(qt, passages, gold)
                    fh.write(
                        json.dumps(
                            {
                                "arm": arm,
                                "qid": qid,
                                "doc": doc,
                                "gold_retrieved": gold_id in ids,
                                "score": score,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    fh.flush()
                    n += 1
                    if n % 40 == 0:
                        print(f"  {n} (arm/query) done", flush=True)
    report()


def report() -> None:
    recs = [json.loads(line) for line in OUT.open(encoding="utf-8") if line.strip()]
    by_arm = defaultdict(list)
    for r in recs:
        if isinstance(r.get("score"), int):
            by_arm[r["arm"]].append(r)
    lines = ["# End-to-end RAG answer quality (LOCAL GX10 14B; A vs B delta is the signal)\n"]
    lines.append(f"{'arm':<14} | {'n':>4} | {'overall correct':>15} | {'german correct':>14}")
    lines.append("|".join(["-" * 15, "-" * 6, "-" * 17, "-" * 16]))
    for arm in ("dense_top5", "dense_top10", "hybrid_top10"):
        rs = by_arm.get(arm, [])
        if not rs:
            continue
        ger = [r for r in rs if r["doc"] in GERMAN]

        def pct(a):
            return 100 * sum(1 for r in a if r["score"] == 2) / len(a) if a else 0.0

        lines.append(f"{arm:<14} | {len(rs):>4} | {pct(rs):>14.1f}% | {pct(ger):>13.1f}%")
    out = "\n".join(lines)
    REPORT.write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    report() if "--report-only" in sys.argv else run()
