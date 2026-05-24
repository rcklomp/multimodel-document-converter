#!/usr/bin/env python3
"""v2.15 Phase 1 acceptance evaluator — compound gate per PLAN_V2.15.md.

Reads two judged work.jsonl files (HyDE-off baseline + HyDE-on
arm) and evaluates the v0.9 compound acceptance gate:

  1. Aggregate R@1 lift >= +6pp
  2. Per-doc directional consistency (subgroup-aware):
     - ATZ_Elektronik (German subgroup) delta >= +10pp
     - >=3/4 code-dense docs (Python_Cookbook, IRJET, Hybrid_electric_vehicles,
       Greenhouse_Design) show positive R@1 delta
  3. Format axis: no regression (delta >= -1pp)
  4. Faithfulness axis: delta >= -1pp (small noise budget)

All four gates must pass. Aggregate-only or per-doc-only wins are
rejected as insufficient signal (Round-4 + Round-7 acceptance
schema).

ALSO: falsification rule (Round-4 Finding 3): if per-doc R@1
delta is NULL (delta <= 0) on >=3 of the 5 target docs, HyDE
bridging is closed as a DEAD LEVER via DECISIONS.md entry rather
than carried forward. Reports this disposition explicitly.

Usage:
  python scripts/evaluate_phase1_acceptance.py \\
    --off output/soak/v2.15_p1_narrow_hyde_off/work.jsonl \\
    --on  output/soak/v2.15_p1_narrow_hyde_on/work.jsonl \\
    --output docs/SOAK_<date>_v2.15_p1_narrow_hyde_AB.md
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

GERMAN_DOC = "ATZ_Elektronik_German"
CODE_DENSE_DOCS = (
    "Python_Cookbook",
    "IRJET_Modeling_of_Solar_PV",
    "Hybrid_electric_vehicles",
    "Greenhouse_Design",
)
ALL_DOCS = (GERMAN_DOC,) + CODE_DENSE_DOCS


def _load(path: Path) -> list[dict]:
    rows = []
    for line in path.open("r", encoding="utf-8"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _per_query(rows: list[dict]) -> list[dict]:
    """Flatten to one record per query with judgment + gold + retrieval."""
    out = []
    for r in rows:
        for q in (r.get("queries") or []):
            if not q.get("retrieval") or not q.get("judgment"):
                continue
            retrieved = (q["retrieval"].get("top_k") or [])
            top1 = retrieved[0] if retrieved else None
            judgment = q.get("judgment") or {}
            out.append({
                "doc_dir": r["doc_dir"],
                "gold_chunk_id": r.get("gold_chunk_id"),
                "gold_doc_id": r.get("gold_doc_id"),
                "query": q["query_text"],
                "top1_chunk_id": (top1 or {}).get("chunk_id"),
                "top1_doc_id": (top1 or {}).get("doc_id"),
                "relevance": judgment.get("relevance"),
                "format": judgment.get("format"),
                "faithfulness": judgment.get("faithfulness"),
            })
    return out


def _r1_chunk_hit(q: dict) -> bool:
    """Recall@1 by chunk: top-1 chunk_id matches gold_chunk_id."""
    return q["top1_chunk_id"] is not None and q["top1_chunk_id"] == q["gold_chunk_id"]


def _axis_avg(qs: list[dict], axis: str) -> float:
    """Average of an axis score (relevance/format/faithfulness), 0-2 scale → 0-100%."""
    scores = [q[axis] for q in qs if q.get(axis) is not None]
    if not scores:
        return 0.0
    # Convert 0-2 ordinal → percent (a 2 = 100, 1 = 50, 0 = 0).
    return sum(scores) * 50.0 / len(scores)


def _r1_rate(qs: list[dict]) -> tuple[float, int, int]:
    n = len(qs)
    if n == 0:
        return 0.0, 0, 0
    hits = sum(1 for q in qs if _r1_chunk_hit(q))
    return hits * 100.0 / n, hits, n


def evaluate(off: list[dict], on: list[dict]) -> dict:
    """Compute per-doc metrics + apply the compound acceptance gate."""
    by_doc: dict[str, dict] = {}
    for doc in ALL_DOCS:
        off_qs = [q for q in off if q["doc_dir"] == doc]
        on_qs = [q for q in on if q["doc_dir"] == doc]
        r1_off, hits_off, n_off = _r1_rate(off_qs)
        r1_on, hits_on, n_on = _r1_rate(on_qs)
        by_doc[doc] = {
            "n_off": n_off,
            "n_on": n_on,
            "r1_off_pct": r1_off,
            "r1_on_pct": r1_on,
            "r1_delta_pp": r1_on - r1_off,
            "fmt_off_pct": _axis_avg(off_qs, "format"),
            "fmt_on_pct": _axis_avg(on_qs, "format"),
            "faith_off_pct": _axis_avg(off_qs, "faithfulness"),
            "faith_on_pct": _axis_avg(on_qs, "faithfulness"),
            "rel_off_pct": _axis_avg(off_qs, "relevance"),
            "rel_on_pct": _axis_avg(on_qs, "relevance"),
        }
    # Aggregate
    r1_off_agg, hits_off_agg, n_off_agg = _r1_rate(off)
    r1_on_agg, hits_on_agg, n_on_agg = _r1_rate(on)
    agg = {
        "n_off": n_off_agg,
        "n_on": n_on_agg,
        "r1_off_pct": r1_off_agg,
        "r1_on_pct": r1_on_agg,
        "r1_delta_pp": r1_on_agg - r1_off_agg,
        "fmt_off_pct": _axis_avg(off, "format"),
        "fmt_on_pct": _axis_avg(on, "format"),
        "faith_off_pct": _axis_avg(off, "faithfulness"),
        "faith_on_pct": _axis_avg(on, "faithfulness"),
        "rel_off_pct": _axis_avg(off, "relevance"),
        "rel_on_pct": _axis_avg(on, "relevance"),
    }

    # Compound acceptance gate
    gate_agg = agg["r1_delta_pp"] >= 6.0
    gate_german = by_doc[GERMAN_DOC]["r1_delta_pp"] >= 10.0
    code_dense_positives = sum(
        1 for d in CODE_DENSE_DOCS if by_doc[d]["r1_delta_pp"] > 0
    )
    gate_code_dense = code_dense_positives >= 3
    gate_format = (agg["fmt_on_pct"] - agg["fmt_off_pct"]) >= -1.0
    gate_faith = (agg["faith_on_pct"] - agg["faith_off_pct"]) >= -1.0
    overall_pass = gate_agg and gate_german and gate_code_dense and gate_format and gate_faith

    # Falsification rule (Round-4 Finding 3):
    # if per-doc delta is NULL (<=0) on >=3 of 5 docs, close as dead lever.
    null_count = sum(1 for d in ALL_DOCS if by_doc[d]["r1_delta_pp"] <= 0)
    dead_lever = null_count >= 3

    return {
        "by_doc": by_doc,
        "aggregate": agg,
        "gates": {
            "aggregate_r1_lift_ge_6pp": gate_agg,
            "german_r1_delta_ge_10pp": gate_german,
            "code_dense_3_of_4_positive": gate_code_dense,
            "code_dense_positives": code_dense_positives,
            "format_no_regression": gate_format,
            "faith_no_regression_beyond_-1pp": gate_faith,
        },
        "overall_pass": overall_pass,
        "falsification_null_count": null_count,
        "dead_lever_triggered": dead_lever,
    }


def render(result: dict, off_path: Path, on_path: Path) -> str:
    today = datetime.date.today().isoformat()
    lines = [
        "# v2.15 Phase 1 — Targeted HyDE Bridging A/B Soak Report",
        "",
        f"> Date: {today}",
        f"> Off-arm work file: `{off_path}`",
        f"> On-arm work file:  `{on_path}`",
        f"> Configuration: HyDE-off baseline vs HyDE-on with auto-intent",
        f">                (intent classifier + vLLM HyDE provider via GX10 FP8-14B)",
        "",
        "## 1. Per-document R@1 delta",
        "",
        "| Document | n (off/on) | R@1 off | R@1 on | Δ pp | Format off | Format on | Faith off | Faith on |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for doc in ALL_DOCS:
        d = result["by_doc"][doc]
        lines.append(
            f"| {doc} | {d['n_off']}/{d['n_on']} | "
            f"{d['r1_off_pct']:.1f}% | {d['r1_on_pct']:.1f}% | "
            f"{d['r1_delta_pp']:+.1f} | "
            f"{d['fmt_off_pct']:.1f}% | {d['fmt_on_pct']:.1f}% | "
            f"{d['faith_off_pct']:.1f}% | {d['faith_on_pct']:.1f}% |"
        )
    a = result["aggregate"]
    lines.append(
        f"| **AGGREGATE** | **{a['n_off']}/{a['n_on']}** | "
        f"**{a['r1_off_pct']:.1f}%** | **{a['r1_on_pct']:.1f}%** | "
        f"**{a['r1_delta_pp']:+.1f}** | "
        f"**{a['fmt_off_pct']:.1f}%** | **{a['fmt_on_pct']:.1f}%** | "
        f"**{a['faith_off_pct']:.1f}%** | **{a['faith_on_pct']:.1f}%** |"
    )
    lines.append("")
    lines.append("## 2. Compound acceptance gate")
    lines.append("")
    g = result["gates"]
    fire = lambda b: "✓ PASS" if b else "✗ FAIL"  # noqa: E731
    lines.append(f"- Aggregate R@1 lift ≥ +6pp: **{fire(g['aggregate_r1_lift_ge_6pp'])}** "
                 f"(measured {a['r1_delta_pp']:+.1f}pp)")
    lines.append(f"- German subgroup R@1 delta ≥ +10pp (ATZ_Elektronik_German): **{fire(g['german_r1_delta_ge_10pp'])}** "
                 f"(measured {result['by_doc'][GERMAN_DOC]['r1_delta_pp']:+.1f}pp on n={result['by_doc'][GERMAN_DOC]['n_on']})")
    lines.append(f"- ≥3/4 code-dense docs positive R@1 delta: **{fire(g['code_dense_3_of_4_positive'])}** "
                 f"({g['code_dense_positives']}/4 positive)")
    lines.append(f"- Format axis no regression: **{fire(g['format_no_regression'])}** "
                 f"(Δ {a['fmt_on_pct'] - a['fmt_off_pct']:+.1f}pp)")
    lines.append(f"- Faithfulness Δ ≥ -1pp: **{fire(g['faith_no_regression_beyond_-1pp'])}** "
                 f"(Δ {a['faith_on_pct'] - a['faith_off_pct']:+.1f}pp)")
    lines.append("")
    lines.append(f"### Overall: **{'✓ ACCEPT' if result['overall_pass'] else '✗ REJECT'}**")
    lines.append("")
    lines.append("## 3. Falsification check (Round-4 Finding 3)")
    lines.append("")
    lines.append(f"Per-doc null count (delta ≤ 0): **{result['falsification_null_count']}/{len(ALL_DOCS)}**")
    if result["dead_lever_triggered"]:
        lines.append("")
        lines.append("**DEAD LEVER TRIGGER FIRED** — ≥3 of 5 target docs show null/negative R@1 "
                     "delta. Per the falsification rule, HyDE bridging is closed as a dead lever "
                     "rather than carried forward to v2.16. Recommended action: add a DECISIONS.md "
                     '"HyDE bridging dead-lever; not carried to v2.16" entry.')
    else:
        lines.append("")
        lines.append("Dead-lever trigger NOT fired. HyDE bridging shows directional positive signal.")
    lines.append("")
    lines.append("## 4. Disposition")
    lines.append("")
    if result["overall_pass"]:
        lines.append("**SHIP** — compound acceptance gate cleared. Promote `auto_intent_hyde=True` "
                     "to production default for hybrid retrieval (or recommend opt-in via "
                     "DECISIONS.md if user prefers conservative rollout).")
    elif result["dead_lever_triggered"]:
        lines.append("**CLOSE AS DEAD LEVER** — falsification rule fired. HyDE bridging infra "
                     "(v2.14 P2 commit `156dfa7`) stays in the code tree as opt-in but is NOT "
                     "promoted; DECISIONS.md gets a closure entry.")
    else:
        lines.append("**DEFER WITH EVIDENCE** — partial pass / partial fail. Specific gates failed:")
        for k, v in g.items():
            if k in ("code_dense_positives",):
                continue
            if not v:
                lines.append(f"  - {k}")
        lines.append("Consider iterating the HyDE system prompt for failing subgroup, or "
                     "documenting partial-success in DECISIONS.md and carrying to v2.16.")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--off", type=Path, required=True,
                   help="HyDE-off baseline work.jsonl (judged)")
    p.add_argument("--on", type=Path, required=True,
                   help="HyDE-on test arm work.jsonl (judged)")
    p.add_argument("--output", type=Path, default=None,
                   help='Report markdown output; defaults to '
                        '"docs/SOAK_<today>_v2.15_p1_narrow_hyde_AB.md"')
    args = p.parse_args()

    off = _per_query(_load(args.off))
    on = _per_query(_load(args.on))
    if not off or not on:
        print(f"ERROR: empty input — off={len(off)}, on={len(on)}", file=sys.stderr)
        return 2
    result = evaluate(off, on)
    report = render(result, args.off, args.on)

    if args.output is None:
        today = datetime.date.today().isoformat()
        args.output = Path(f"docs/SOAK_{today}_v2.15_p1_narrow_hyde_AB.md")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Overall: {'ACCEPT' if result['overall_pass'] else 'REJECT'}; "
          f"dead-lever={result['dead_lever_triggered']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
