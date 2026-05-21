#!/usr/bin/env python3
"""v2.12 Phase 1 soak comparator — pick the reranker that wins.

Reads two synthetic-soak work files (output/soak/<run>/work.jsonl)
and reports per-axis deltas + recommends a winner based on the
plan's floor + stretch targets.

Used by the v2.12 Phase 1 shootout to compare the cloud Dashscope
gte-rerank run against the local omlx ModernBERT run. Either run can
also be compared against the v2.11.0 baseline by passing the v2.11
work file as side A.

The winner is recommended; the final decision rule comes from
PLAN_V2.12 §"Acceptance Gate":

  Floor   Recall@1 chunk >= 55%
          Recall@5 chunk >= 85%
          Recall@5 doc   >= 95%
          Relevance      >= 75%
          Faithfulness   >= 70%
          Format         >= 96%   (post-Phase-0 carry-forward)

  Stretch Recall@1 chunk >= 70%
          Recall@5 chunk >= 90%
          Recall@5 doc   >= 97%
          Relevance      >= 85%
          Faithfulness   >= 80%
          Format         >= 98%

Output: a markdown table + a recommended winner (the higher-scoring
on the embedder-attributable axes: R@1, R@5 chunk, Relevance,
Faithfulness, with Format as tiebreaker if both pass the floor).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_metrics(work_path: Path) -> dict:
    """Aggregate per-axis metrics from a soak work file."""
    rows = []
    for line in open(work_path):
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

    n_queries = 0
    r1_chunk = 0
    r5_chunk = 0
    r5_doc = 0
    judged = 0
    rel = fmt = fai = 0
    rel_max = fmt_max = fai_max = 0
    per_doc: dict[str, dict] = {}

    for row in rows:
        doc = row["doc_dir"]
        s = per_doc.setdefault(doc, {
            "queries": 0, "r1": 0, "r5_chunk": 0, "r5_doc": 0,
            "rel": 0, "fmt": 0, "fai": 0, "judged": 0,
        })
        gold_chunk = row["gold_chunk_id"]
        gold_doc = row["gold_doc_id"]
        for q in row.get("queries", []) or []:
            n_queries += 1
            s["queries"] += 1
            top = (q.get("retrieval") or {}).get("top_k") or []
            top_ids = [r["chunk_id"] for r in top]
            top_docs = [r.get("doc_id") for r in top]
            if top_ids and top_ids[0] == gold_chunk:
                r1_chunk += 1
                s["r1"] += 1
            if gold_chunk in top_ids[:5]:
                r5_chunk += 1
                s["r5_chunk"] += 1
            if gold_doc in top_docs[:5]:
                r5_doc += 1
                s["r5_doc"] += 1
            jud = q.get("judgment") or {}
            if jud.get("relevance") is not None:
                judged += 1
                s["judged"] += 1
                rel += int(jud["relevance"]);     rel_max += 2; s["rel"] += int(jud["relevance"])
                fmt += int(jud["format"]);        fmt_max += 2; s["fmt"] += int(jud["format"])
                fai += int(jud["faithfulness"]);  fai_max += 2; s["fai"] += int(jud["faithfulness"])

    return {
        "n_rows": len(rows),
        "n_queries": n_queries,
        "judged": judged,
        "r1_chunk_pct": (r1_chunk / n_queries * 100) if n_queries else 0.0,
        "r5_chunk_pct": (r5_chunk / n_queries * 100) if n_queries else 0.0,
        "r5_doc_pct":   (r5_doc / n_queries * 100) if n_queries else 0.0,
        "relevance_pct":     (rel / rel_max * 100) if rel_max else 0.0,
        "format_pct":        (fmt / fmt_max * 100) if fmt_max else 0.0,
        "faithfulness_pct":  (fai / fai_max * 100) if fai_max else 0.0,
        "per_doc": per_doc,
    }


FLOORS = {
    "r1_chunk_pct":     55.0,
    "r5_chunk_pct":     85.0,
    "r5_doc_pct":       95.0,
    "relevance_pct":    75.0,
    "format_pct":       96.0,
    "faithfulness_pct": 70.0,
}

STRETCH = {
    "r1_chunk_pct":     70.0,
    "r5_chunk_pct":     90.0,
    "r5_doc_pct":       97.0,
    "relevance_pct":    85.0,
    "format_pct":       98.0,
    "faithfulness_pct": 80.0,
}

LABEL = {
    "r1_chunk_pct":     "Recall@1 chunk",
    "r5_chunk_pct":     "Recall@5 chunk",
    "r5_doc_pct":       "Recall@5 doc",
    "relevance_pct":    "Relevance",
    "format_pct":       "Format",
    "faithfulness_pct": "Faithfulness",
}


def floor_status(metrics: dict, axis: str) -> str:
    """Return a status marker for a single axis."""
    v = metrics[axis]
    if v >= STRETCH[axis]:
        return "stretch"
    if v >= FLOORS[axis]:
        return "floor"
    return "miss"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a-work", required=True,
                        help="Work file for side A (e.g. cloud-rerank soak).")
    parser.add_argument("--a-name", default="A",
                        help="Display name for side A (e.g. 'cloud gte-rerank').")
    parser.add_argument("--b-work", required=True,
                        help="Work file for side B (e.g. omlx-rerank soak).")
    parser.add_argument("--b-name", default="B",
                        help="Display name for side B (e.g. 'omlx ModernBERT').")
    parser.add_argument("--baseline-work", default=None,
                        help="Optional: v2.11.0 baseline work file for a third "
                             "delta column.")
    parser.add_argument("--baseline-name", default="baseline",
                        help="Display name for the baseline column.")
    args = parser.parse_args()

    a = load_metrics(Path(args.a_work))
    b = load_metrics(Path(args.b_work))
    baseline = load_metrics(Path(args.baseline_work)) if args.baseline_work else None

    # Sanity checks.
    if a["n_queries"] != b["n_queries"]:
        print(f"WARNING: query counts differ — {args.a_name}={a['n_queries']} "
              f"vs {args.b_name}={b['n_queries']}; deltas may not be apples-to-apples",
              file=sys.stderr)

    print()
    print(f"=== v2.12 Phase 1 reranker shootout ===")
    print(f"  {args.a_name:30s}: {a['n_queries']:>4} queries, {a['judged']} judged")
    print(f"  {args.b_name:30s}: {b['n_queries']:>4} queries, {b['judged']} judged")
    if baseline:
        print(f"  {args.baseline_name:30s}: {baseline['n_queries']:>4} queries, {baseline['judged']} judged")
    print()

    print(f"| {'Axis':18} | {args.a_name:>20} | {args.b_name:>20} | "
          + (f"{args.baseline_name:>20} | Winner |" if baseline else "Winner |"))
    print(f"|{'-'*20}|{'-'*22}:|{'-'*22}:|"
          + (f"{'-'*22}:|{'-'*8}|" if baseline else f"{'-'*8}|"))

    a_wins = 0
    b_wins = 0
    for axis in ["r1_chunk_pct", "r5_chunk_pct", "r5_doc_pct",
                 "relevance_pct", "format_pct", "faithfulness_pct"]:
        a_val = a[axis]
        b_val = b[axis]
        a_status = floor_status(a, axis)
        b_status = floor_status(b, axis)
        if abs(a_val - b_val) < 0.5:
            winner = "tie"
        elif a_val > b_val:
            winner = args.a_name
            a_wins += 1
        else:
            winner = args.b_name
            b_wins += 1
        baseline_str = ""
        if baseline:
            baseline_val = baseline[axis]
            baseline_str = f" {baseline_val:>5.1f}% |"
        floor_target = FLOORS[axis]
        floor_marker = "FLOOR" if a_val >= floor_target or b_val >= floor_target else "MISS"
        print(f"| {LABEL[axis]:18} | {a_val:>5.1f}% ({a_status:>7s}) | "
              f"{b_val:>5.1f}% ({b_status:>7s}) |{baseline_str} {winner:>8s} |")

    print()
    print(f"Axis-level wins (excluding ties):")
    print(f"  {args.a_name}: {a_wins}")
    print(f"  {args.b_name}: {b_wins}")
    print()

    # Recommended winner: higher wins on embedder-attributable axes
    # (R@1, R@5 chunk, Relevance, Faithfulness). Format is a tie-breaker.
    embedder_axes = ["r1_chunk_pct", "r5_chunk_pct", "relevance_pct", "faithfulness_pct"]
    a_embedder_wins = sum(1 for ax in embedder_axes if a[ax] - b[ax] > 0.5)
    b_embedder_wins = sum(1 for ax in embedder_axes if b[ax] - a[ax] > 0.5)
    print(f"Embedder-attributable axis wins (R@1, R@5 chunk, Relevance, Faithfulness):")
    print(f"  {args.a_name}: {a_embedder_wins}/{len(embedder_axes)}")
    print(f"  {args.b_name}: {b_embedder_wins}/{len(embedder_axes)}")
    print()

    if a_embedder_wins > b_embedder_wins:
        recommendation = args.a_name
    elif b_embedder_wins > a_embedder_wins:
        recommendation = args.b_name
    else:
        # Tie on embedder axes; prefer the one with better Format.
        if a["format_pct"] > b["format_pct"]:
            recommendation = args.a_name
        else:
            recommendation = args.b_name

    print(f"RECOMMENDED v2.12 PHASE 1 RERANKER: {recommendation}")

    # Phase trigger check.
    winner_metrics = a if recommendation == args.a_name else b
    phase_2_needed = winner_metrics["r5_chunk_pct"] < FLOORS["r5_chunk_pct"]
    phase_3_needed = (
        winner_metrics["r1_chunk_pct"] < FLOORS["r1_chunk_pct"]
        or winner_metrics["faithfulness_pct"] < FLOORS["faithfulness_pct"]
    )
    print()
    print(f"Phase 2 (hybrid retrieval) trigger: Recall@5 chunk {winner_metrics['r5_chunk_pct']:.1f}% "
          f"{'<' if phase_2_needed else '>='} 85% -> "
          f"{'TRIGGERED' if phase_2_needed else 'skipped'}")
    print(f"Phase 3 (HyDE) trigger: Recall@1 {winner_metrics['r1_chunk_pct']:.1f}% "
          f"{'<' if winner_metrics['r1_chunk_pct'] < FLOORS['r1_chunk_pct'] else '>='} 55% "
          f"OR Faithfulness {winner_metrics['faithfulness_pct']:.1f}% "
          f"{'<' if winner_metrics['faithfulness_pct'] < FLOORS['faithfulness_pct'] else '>='} 70% -> "
          f"{'TRIGGERED' if phase_3_needed else 'skipped'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
