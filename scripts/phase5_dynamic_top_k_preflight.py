"""v2.16 Phase 5 pre-flight — analytical dynamic-top-k SHIP/KILL gate.

Per PLAN_V2.16.md §3 Phase 5 disposition gate. Applies the proposed
truncation logic to Phase 1's BASELINE rerank outputs and emits a
binary SHIP-default-on / KILL verdict.

Algorithm (mirrors the proposed Phase 5 production code):

  logits = [r.rerank_score for r in reranked]
  if len(logits) < 2:    return reranked   # no truncation
  gaps = [logits[i] - logits[i+1] for i in range(len(logits)-1)]
  mean_gap = sum(gaps) / len(gaps)
  for i, gap in enumerate(gaps):
      if gap > drop_off_threshold * mean_gap and gap > min_absolute_gap:
          return reranked[: max(min_top_n, i + 1)]
  return reranked

Defaults: drop_off_threshold=2.5, min_absolute_gap=0.05, min_top_n=1.

SHIP-default-on gate (ALL THREE must hold):
  (a) ≥20% of queries `would_truncate`.
  (b) PASS-retention ≥ 0.97 across full fixture set.
  (c) No HIGH-class fixture pass rate falls more than 2pp below
      its static baseline.

ANY leg fails → KILL permanently. No opt-in middle ground per
PLAN_V2.16.md §3 Phase 5.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from mmrag_v2.retrieval.pipeline import retrieve_hybrid_reranked  # noqa: E402

import run_personal_validation as rpv  # noqa: E402


DROP_OFF_THRESHOLD = 2.5
MIN_ABSOLUTE_GAP = 0.05
MIN_TOP_N = 1
TOP_N_BASELINE = 5


def simulate_truncate(
    rerank_scores: list[float],
    *,
    drop_off_threshold: float = DROP_OFF_THRESHOLD,
    min_absolute_gap: float = MIN_ABSOLUTE_GAP,
    min_top_n: int = MIN_TOP_N,
) -> int:
    """Return the truncated top-N size for a sorted-desc rerank_score list.
    Returns `len(rerank_scores)` if no truncation applies."""
    if len(rerank_scores) < 2:
        return len(rerank_scores)
    gaps = [
        rerank_scores[i] - rerank_scores[i + 1]
        for i in range(len(rerank_scores) - 1)
    ]
    mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
    for i, gap in enumerate(gaps):
        if gap > drop_off_threshold * mean_gap and gap > min_absolute_gap:
            return max(min_top_n, i + 1)
    return len(rerank_scores)


@dataclass
class QueryProbe:
    query_id: str
    class_name: str
    importance: str
    static_pass: bool
    rerank_scores: list[float] = field(default_factory=list)
    truncated_n: int = TOP_N_BASELINE
    would_truncate: bool = False
    dynamic_pass: bool = False


def evaluate_one_query(
    query: dict,
    class_name: str,
    importance: str,
) -> QueryProbe:
    qid = query.get("id", "?")
    qtext = query.get("query_text", "")
    expected = query.get("expected") or {}
    require_gold = bool(expected.get("top_5_gold_doc"))
    format_constraint = expected.get("format_constraint")
    regex_patterns = expected.get("expected_anchor_regexes") or []

    chunks = retrieve_hybrid_reranked(qtext, top_n_return=TOP_N_BASELINE)
    probe = QueryProbe(
        query_id=qid,
        class_name=class_name,
        importance=importance,
        static_pass=False,
    )
    if not chunks:
        return probe

    probe.rerank_scores = [float(c.get("rerank_score", 0.0)) for c in chunks]
    probe.truncated_n = simulate_truncate(probe.rerank_scores)
    probe.would_truncate = probe.truncated_n < len(chunks)

    # Compute static PASS (top-5 baseline).
    top_5_basenames = [
        rpv._doc_id_from_payload(c.get("payload") or {})
        for c in chunks[:TOP_N_BASELINE]
    ]
    top_1_payload = chunks[0].get("payload") or {}
    static_pass = True
    if require_gold and class_name not in top_5_basenames:
        static_pass = False
    if static_pass:
        fp = rpv._check_format(format_constraint, top_1_payload)
        if fp is False:
            static_pass = False
    if static_pass and regex_patterns:
        m = rpv._check_regex(regex_patterns, rpv._content_of(top_1_payload))
        if m is None:
            static_pass = False
    probe.static_pass = static_pass

    # Compute dynamic PASS (truncated top-N).
    # When `min_top_n=1` truncates aggressively, top-1 is unchanged and
    # so per-query PASS hinges on the same top-1 + the gold-in-top-N
    # subset. Truncation can only DROP PASS (never elevate); recompute.
    dynamic_top_basenames = top_5_basenames[: probe.truncated_n]
    dynamic_pass = True
    if require_gold and class_name not in dynamic_top_basenames:
        dynamic_pass = False
    if dynamic_pass:
        fp = rpv._check_format(format_constraint, top_1_payload)
        if fp is False:
            dynamic_pass = False
    if dynamic_pass and regex_patterns:
        m = rpv._check_regex(regex_patterns, rpv._content_of(top_1_payload))
        if m is None:
            dynamic_pass = False
    probe.dynamic_pass = dynamic_pass
    return probe


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fixtures-dir",
        type=Path,
        default=_REPO_ROOT / "tests/fixtures/personal_validation_queries",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "docs/PHASE5_PREFLIGHT_2026-05-25.md",
    )
    args = ap.parse_args()

    # Build doc_id map for top-5 resolution (same as run_personal_validation).
    rpv._DOC_ID_MAP = rpv._build_doc_id_to_basename_map(_REPO_ROOT / "output")

    fixtures = sorted(args.fixtures_dir.glob("*.json"))
    if not fixtures:
        print(f"No fixtures found in {args.fixtures_dir}", file=sys.stderr)
        return 2

    all_probes: list[QueryProbe] = []
    per_class_static: dict[str, list[bool]] = {}
    per_class_dynamic: dict[str, list[bool]] = {}
    per_class_importance: dict[str, str] = {}

    for fx in fixtures:
        d = rpv.load_fixture(fx)
        class_name = d["class"]
        importance = d["personal_importance"]
        per_class_importance[class_name] = importance
        per_class_static[class_name] = []
        per_class_dynamic[class_name] = []
        for q in d["queries"]:
            probe = evaluate_one_query(q, class_name, importance)
            all_probes.append(probe)
            per_class_static[class_name].append(probe.static_pass)
            per_class_dynamic[class_name].append(probe.dynamic_pass)

    # Compute gate conditions.
    n_total = len(all_probes)
    n_would_truncate = sum(1 for p in all_probes if p.would_truncate)
    truncate_rate = (n_would_truncate / n_total) if n_total else 0.0
    n_static_pass = sum(1 for p in all_probes if p.static_pass)
    n_dynamic_pass = sum(1 for p in all_probes if p.dynamic_pass)
    if n_static_pass == 0:
        retention = None
    else:
        retention = n_dynamic_pass / n_static_pass

    leg_a = truncate_rate >= 0.20
    leg_b = retention is not None and retention >= 0.97
    leg_c = True
    leg_c_detail: list[str] = []
    for cls, importance in per_class_importance.items():
        if importance != "HIGH":
            continue
        static_rate = (
            sum(per_class_static[cls]) / len(per_class_static[cls])
            if per_class_static[cls]
            else 0.0
        )
        dyn_rate = (
            sum(per_class_dynamic[cls]) / len(per_class_dynamic[cls])
            if per_class_dynamic[cls]
            else 0.0
        )
        delta_pp = (dyn_rate - static_rate) * 100.0
        leg_c_detail.append(
            f"{cls}: static={static_rate*100:.1f}% dyn={dyn_rate*100:.1f}% Δ={delta_pp:+.1f}pp"
        )
        if delta_pp < -2.0:
            leg_c = False

    ship = leg_a and leg_b and leg_c

    # Render verdict report.
    lines = [
        "# v2.16 Phase 5 — Dynamic Top-K Pre-Flight Verdict",
        "",
        "> Generated: 2026-05-25",
        f"> Fixture set: {len(fixtures)} class file(s), {n_total} queries.",
        f"> drop_off_threshold={DROP_OFF_THRESHOLD}, "
        f"min_absolute_gap={MIN_ABSOLUTE_GAP}, "
        f"min_top_n={MIN_TOP_N}, baseline top_n_return={TOP_N_BASELINE}.",
        "",
        "## Verdict",
        "",
    ]
    if ship:
        lines.append("**SHIP default-on.** All three gate legs satisfied.")
    else:
        lines.append("**KILL permanently.** At least one gate leg failed.")
    lines.extend([
        "",
        "## Gate evaluation",
        "",
        "| Leg | Condition | Result | Detail |",
        "|---|---|---|---|",
        f"| (a) | ≥20% of queries `would_truncate` | "
        f"{'PASS' if leg_a else 'FAIL'} | "
        f"{n_would_truncate}/{n_total} = {truncate_rate*100:.1f}% |",
        f"| (b) | PASS-retention ≥ 0.97 | "
        f"{'PASS' if leg_b else 'FAIL'} | "
        f"static_pass={n_static_pass}, dyn_pass={n_dynamic_pass}, "
        f"retention={retention if retention is not None else 'undefined (static=0)'} |",
        f"| (c) | No HIGH class drops >2pp | "
        f"{'PASS' if leg_c else 'FAIL'} | "
        f"{'; '.join(leg_c_detail)} |",
        "",
        "## Per-query truncation samples",
        "",
        "| query_id | class | top-N scores | trunc_n | would_truncate | static_pass | dynamic_pass |",
        "|---|---|---|---|---|---|---|",
    ])
    for p in all_probes:
        scores = ", ".join(f"{s:.3f}" for s in p.rerank_scores)
        lines.append(
            f"| {p.query_id} | {p.class_name} | [{scores}] | "
            f"{p.truncated_n} | "
            f"{'✓' if p.would_truncate else '·'} | "
            f"{'✓' if p.static_pass else '·'} | "
            f"{'✓' if p.dynamic_pass else '·'} |"
        )
    lines.extend([
        "",
        "## Disposition",
        "",
    ])
    if ship:
        lines.append(
            "Implement the dynamic top-k branch in `retrieve_hybrid_reranked`. "
            "Default `dynamic_top_k=True`. Diagnostic disable via "
            "`--no-dynamic-top-k` on `synthetic_soak.py`. Add ≥3 bridge "
            "tests in `tests/test_retrieval_pipeline.py` per PLAN_V2.16.md "
            "§3 Phase 5 step 3."
        )
    else:
        legs_failed = []
        if not leg_a:
            legs_failed.append("(a) truncate-rate below 20%")
        if not leg_b:
            legs_failed.append("(b) PASS-retention undefined or below 0.97")
        if not leg_c:
            legs_failed.append("(c) HIGH class regressed >2pp")
        lines.append(
            "Per PLAN_V2.16.md §3 Phase 5, KILL is permanent — no opt-in "
            "middle ground. DECISIONS.md entry: "
            "\"v2.16 Phase 5 KILL — pre-flight evidence shows dynamic "
            "top-k has no measurable upside on the corpus. "
            f"Failed legs: {', '.join(legs_failed)}.\""
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    rel = args.output.relative_to(_REPO_ROOT) if _REPO_ROOT in args.output.parents else args.output
    print(f"Wrote {rel}")
    print(f"Phase 5 verdict: {'SHIP' if ship else 'KILL'}")
    print(f"  leg (a) truncate-rate {truncate_rate*100:.1f}% >= 20%: {'PASS' if leg_a else 'FAIL'}")
    print(
        f"  leg (b) retention {'undefined (static=0)' if retention is None else f'{retention:.3f}'} "
        f">= 0.97: {'PASS' if leg_b else 'FAIL'}"
    )
    print(f"  leg (c) HIGH no >2pp drop: {'PASS' if leg_c else 'FAIL'}")
    return 0 if ship else 1


if __name__ == "__main__":
    raise SystemExit(main())
