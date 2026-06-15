#!/usr/bin/env python3
"""End-to-end RAG ANSWER-quality eval (measure the answer, not the chunk).

The synthetic soak judged the retrieved CHUNK. This judges the ANSWER a RAG would
actually produce: it takes the soak's already-retrieved top-5 passages, generates an
answer from them (the generation step a real RAG does), and grades the ANSWER for
correctness against the gold passage. This measures what R@1 cannot: whether the LLM
answers correctly even when the exact gold chunk was not top-1 (doc recall is 92% and
the model reads all five passages).

Reuses the soak's Dashscope client + the existing work.jsonl (queries + retrieval +
gold). Resumable: appends per query to answers.jsonl; re-run skips done query_ids.

Usage:
  python scripts/rag_answer_eval.py [--limit N] [--report-only]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_soak import _call_dashscope  # noqa: E402  (reuse the OpenAI-compat client)

MODEL = "qwen-max"
WORK = Path("output/v3_soak_code/work.jsonl")
OUT = Path("output/v3_soak_code/answers.jsonl")
REPORT = Path("output/v3_soak_code/answer_report.md")

GEN_SYS = (
    "You are a retrieval-augmented assistant. Answer the question using ONLY the "
    "information in the numbered passages. If the passages do not contain the answer, "
    "reply exactly: NOT IN CONTEXT. Be concise and specific."
)
JUDGE_SYS = "You grade a retrieval-augmented assistant's answer for CORRECTNESS. Output ONLY JSON."


def _gen_messages(query: str, passages: list[str]) -> list[dict]:
    ctx = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    return [
        {"role": "system", "content": GEN_SYS},
        {"role": "user", "content": f"Passages:\n{ctx}\n\nQuestion: {query}\nAnswer:"},
    ]


def _judge_messages(query: str, gold: str, answer: str) -> list[dict]:
    return [
        {"role": "system", "content": JUDGE_SYS},
        {
            "role": "user",
            "content": (
                f"Question: {query}\n\n"
                f"Reference passage (the ground truth the question was written from):\n{gold[:1500]}\n\n"
                f"Assistant answer:\n{answer}\n\n"
                "Grade CORRECTNESS: does the answer correctly and specifically answer the "
                "question, consistent with the reference?\n"
                "2 = fully correct; 1 = partially correct or incomplete; 0 = incorrect, "
                "hallucinated, or 'NOT IN CONTEXT' when the answer was actually available.\n"
                'Respond with JSON only: {"score": 0|1|2, "why": "<short>"}'
            ),
        },
    ]


def _parse_score(text: str | None):
    if not text:
        return None
    m = re.search(r'"score"\s*:\s*([0-2])', text)
    return int(m.group(1)) if m else None


def _iter_queries(rows):
    for r in rows:
        for q in r.get("queries", []):
            tk = q.get("retrieval", {}).get("top_k") or []
            if tk and q.get("query_text"):
                yield r, q, tk


def run(limit: int | None) -> None:
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not key:
        print("ERROR: DASHSCOPE_API_KEY not set", file=sys.stderr)
        raise SystemExit(2)
    done = set()
    if OUT.exists():
        for line in OUT.open(encoding="utf-8"):
            try:
                done.add(json.loads(line)["query_id"])
            except Exception:
                pass
    rows = [json.loads(line) for line in WORK.open(encoding="utf-8") if line.strip()]
    todo = [(r, q, tk) for r, q, tk in _iter_queries(rows) if q.get("query_id") not in done]
    if limit:
        todo = todo[:limit]
    print(f"queries to do: {len(todo)} ({len(done)} already done)", flush=True)
    with OUT.open("a", encoding="utf-8") as fh:
        for i, (r, q, tk) in enumerate(todo, 1):
            query = q["query_text"]
            gold = r.get("gold_content") or ""
            gold_id = r.get("gold_chunk_id")
            ids = [e.get("chunk_id") for e in tk]
            passages = [(e.get("content") or "").strip() for e in tk]
            ans = _call_dashscope(key, MODEL, _gen_messages(query, passages), max_tokens=400)
            score = None
            if ans is not None:
                jt = _call_dashscope(key, MODEL, _judge_messages(query, gold, ans), max_tokens=200)
                score = _parse_score(jt)
            fh.write(
                json.dumps(
                    {
                        "query_id": q.get("query_id"),
                        "doc_dir": r.get("doc_dir"),
                        "gold_top1": bool(ids) and ids[0] == gold_id,
                        "gold_in_top5": gold_id in ids[:5],
                        "answer": ans,
                        "score": score,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fh.flush()
            if i % 20 == 0:
                print(f"  {i}/{len(todo)} done", flush=True)
    print(f"done -> {OUT}", flush=True)


def report() -> None:
    recs = [json.loads(line) for line in OUT.open(encoding="utf-8") if line.strip()]
    scored = [r for r in recs if isinstance(r.get("score"), int)]
    n = len(scored)
    if not n:
        print("no scored records yet")
        return

    def rate(rs):
        if not rs:
            return None, 0
        return sum(1 for r in rs if r["score"] == 2) / len(rs), len(rs)

    full, _ = rate(scored)
    correct_or_partial = sum(1 for r in scored if r["score"] >= 1) / n
    # answer correctness conditioned on retrieval outcome
    t1 = [r for r in scored if r["gold_top1"]]
    t5 = [r for r in scored if r["gold_in_top5"] and not r["gold_top1"]]
    miss = [r for r in scored if not r["gold_in_top5"]]
    per_doc = defaultdict(list)
    for r in scored:
        per_doc[r["doc_dir"]].append(r["score"])

    lines = []
    lines.append("# End-to-end RAG answer-quality eval\n")
    lines.append(f"- Scored answers: **{n}** (judge: qwen-max, generator: qwen-max).")
    lines.append(f"- **Answer fully correct (score 2): {100*full:.1f}%**")
    lines.append(f"- Answer correct-or-partial (score >=1): {100*correct_or_partial:.1f}%\n")
    lines.append("## Answer correctness by retrieval outcome (the key question)\n")
    lines.append("| retrieval of the exact gold chunk | n | answer fully correct |")
    lines.append("|---|--:|--:|")
    for label, rs in (
        ("gold was top-1", t1),
        ("gold in top-5 (not top-1)", t5),
        ("gold NOT in top-5", miss),
    ):
        rr, c = rate(rs)
        lines.append(
            f"| {label} | {c} | {100*rr:.1f}% |" if rr is not None else f"| {label} | 0 | - |"
        )
    lines.append("\n> If 'gold NOT in top-5' still answers correctly often, the LLM is")
    lines.append(
        "> rescuing right-document/wrong-chunk cases (doc recall 92%), and R@1 understates the RAG.\n"
    )
    lines.append("## Per-document answer correctness\n")
    lines.append("| Doc | n | fully correct |")
    lines.append("|---|--:|--:|")
    for doc in sorted(
        per_doc, key=lambda d: sum(1 for s in per_doc[d] if s == 2) / len(per_doc[d])
    ):
        s = per_doc[doc]
        lines.append(f"| {doc} | {len(s)} | {100*sum(1 for x in s if x==2)/len(s):.1f}% |")
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nreport -> {REPORT}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if not args.report_only:
        run(args.limit)
    report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
