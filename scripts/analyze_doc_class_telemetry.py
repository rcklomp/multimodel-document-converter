#!/usr/bin/env python3
"""v2.15 Phase 3 [F] — document-class telemetry analyzer.

Reads the rolling document-class-hits log (default:
`output/telemetry/document_class_hits.jsonl`), applies the v2.15
promotion / closure / middle-band rules from
`DECISIONS.md` "v2.15 Documented-Limitation Telemetry Threshold",
and emits a markdown report at
`docs/TELEMETRY_REPORT_<YYYY-MM-DD>.md`.

Per-class output (one section per class registered in
`src/mmrag_v2/retrieval/documented_limitations.py`):

  ## CarOK_voorraadtelling
  - added_cycle: v2.15  (current: v2.16 → grace_period_elapsed: True)
  - severe_defect_tag: True
  - 30-day hit-rate: 7.2% (37 / 514 qualified queries)
  - 60-day hit-rate: 6.8% (62 / 911 qualified queries)
  - open_user_issues: 0
  - consecutive_middle_cycles: 0
  - PROMOTION TRIGGER (standard arm: >=5% AND pain-signal): FIRED
  - PROMOTION TRIGGER (defect-override arm: defect-tag AND >=1%): FIRED
  - CLOSURE TRIGGER (<1% AND 0 issues AND no defect-tag AND grace elapsed): NOT FIRED
  - MIDDLE-BAND ESCALATION (>=3 consecutive cycles): NOT FIRED
  - v2.X disposition: Option A treatment (extraction-lane investment)

The cycle-open checklist line item is:
  "Run scripts/analyze_doc_class_telemetry.py;
   copy trigger-fired booleans into opening plan's Carry-Forwards table"

Usage:
  python scripts/analyze_doc_class_telemetry.py \\
    --current-cycle v2.16 \\
    --telemetry-log output/telemetry/document_class_hits.jsonl \\
    --user-issues-doc docs/USER_ISSUES.md \\
    --previous-report docs/TELEMETRY_REPORT_2026-08-15.md \\
    --output docs/TELEMETRY_REPORT_<today>.md

  All inputs have sensible defaults; running with no args is fine
  for the standard cycle-open workflow.

Audit trail: ships as Phase 3 deliverable per Round-4 Finding 1
(was the HIGH that revealed v0.4's "trigger with no reader" gap).
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
import time
from pathlib import Path

# Repo path setup so we can import the documented-limitation registry.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mmrag_v2.retrieval.documented_limitations import (  # noqa: E402
    CLOSURE_THRESHOLD_PCT,
    DEFECT_OVERRIDE_THRESHOLD_PCT,
    DOCUMENTED_LIMITATION_CLASSES,
    MIDDLE_BAND_PERSISTENCE_CYCLES,
    NEW_CLASS_GRACE_CYCLES,
    PROMOTION_THRESHOLD_PCT,
    cycles_since,
    personal_importance,
)

DEFAULT_TELEMETRY_LOG = _REPO_ROOT / "output/telemetry/document_class_hits.jsonl"
DEFAULT_USER_ISSUES = _REPO_ROOT / "docs/USER_ISSUES.md"


def _load_telemetry(path: Path) -> list[dict]:
    """Read the JSONL log; skip malformed lines silently (one bad
    line shouldn't crash the analyzer)."""
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _qualified_in_window(rows: list[dict], now: float, window_days: int) -> list[dict]:
    """Return rows within the last `window_days` that have a
    `rerank_top_5_non_empty == True` flag. These are the
    'qualified queries' that form the denominator for hit-rate
    computation."""
    cutoff = now - (window_days * 86400)
    return [
        r for r in rows
        if r.get("timestamp", 0) >= cutoff
        and r.get("rerank_top_5_non_empty") is True
    ]


def _class_hit_count(qualified: list[dict], class_name: str) -> int:
    """Count rows whose `document_class_hits` includes `class_name`."""
    return sum(
        1 for r in qualified
        if class_name in (r.get("document_class_hits") or [])
    )


def _hit_rate_pct(qualified: list[dict], class_name: str) -> tuple[float, int, int]:
    """Return (rate_pct, hits, denominator) for `class_name` against
    the qualified set. Returns 0.0 rate if denominator is 0."""
    denom = len(qualified)
    if denom == 0:
        return 0.0, 0, 0
    hits = _class_hit_count(qualified, class_name)
    return (hits * 100.0 / denom), hits, denom


# USER_ISSUES.md is an append-only markdown table with rows like:
#   | 2026-08-15 | CarOK_voorraadtelling | "..." | "..." | "..." |
# We tally entries whose doc_class column matches the class name AND
# whose date is on/after the prior-cycle-tag date (or all-time if no
# prior report provided).
#
# Cells are pipe-delimited; first two cells are date + doc_class.
_USER_ISSUE_ROW_RE = re.compile(
    r"^\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*([^|]+?)\s*\|"
)


def _load_user_issues(
    path: Path, since_date: str | None = None,
) -> dict[str, int]:
    """Parse USER_ISSUES.md; return per-class issue count.

    If `since_date` (ISO YYYY-MM-DD) is provided, only count issues
    on/after that date. Otherwise count all-time entries.
    """
    if not path.exists():
        return {}
    counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            m = _USER_ISSUE_ROW_RE.match(line)
            if not m:
                continue
            date_str, class_str = m.group(1), m.group(2).strip()
            if since_date and date_str < since_date:
                continue
            counts[class_str] = counts.get(class_str, 0) + 1
    return counts


def _consecutive_middle_cycles(
    previous_report: Path | None, class_name: str,
) -> int:
    """Look up `consecutive_middle_cycles` for `class_name` from a
    previous TELEMETRY_REPORT_*.md file. Returns 0 if no prior
    report or class not found (first time analyzing).

    Parses lines like:
      - consecutive_middle_cycles: 2
    appearing under the class section header
      ## <class_name>
    """
    if previous_report is None or not previous_report.exists():
        return 0
    text = previous_report.read_text(encoding="utf-8")
    # Find the section for this class
    header = f"## {class_name}"
    if header not in text:
        return 0
    section_start = text.index(header)
    next_header = text.find("\n## ", section_start + 1)
    section = text[section_start:next_header] if next_header > 0 else text[section_start:]
    m = re.search(r"^- consecutive_middle_cycles:\s*(\d+)", section, re.MULTILINE)
    return int(m.group(1)) if m else 0


def _classify(
    class_meta: dict,
    rate30_pct: float,
    rate60_pct: float,
    open_issues: int,
    prior_middle_streak: int,
    current_cycle: str,
) -> dict:
    """Apply all v2.15 promotion / closure / middle-band rules.
    Returns the trigger-fired booleans and the cycle disposition.
    """
    severe = bool(class_meta.get("severe_defect_tag"))
    added = class_meta.get("added_cycle", "v2.15")
    importance = class_meta.get("personal_importance", "MED")
    grace_n = cycles_since(added, current_cycle)
    # v2.16 Phase 1 overlay: LOW reduces grace from 2 to 1.
    effective_grace = 1 if importance == "LOW" else NEW_CLASS_GRACE_CYCLES
    grace_elapsed = grace_n >= effective_grace

    # Promotion arms (R5F1 standard + R6F1 defect-override) — telemetry-only.
    standard_arm = (
        rate30_pct >= PROMOTION_THRESHOLD_PCT
        and (severe or open_issues >= 1)
    )
    defect_arm = (
        severe and rate30_pct >= DEFECT_OVERRIDE_THRESHOLD_PCT
    )
    telemetry_promotion_fired = standard_arm or defect_arm

    # v2.16 Phase 1 overlay: HIGH forces Option A regardless of telemetry.
    importance_override_fired = importance == "HIGH"
    promotion_fired = telemetry_promotion_fired or importance_override_fired

    # Closure: <1% AND no issues AND no defect-tag AND grace elapsed
    closure_fired = (
        rate60_pct < CLOSURE_THRESHOLD_PCT
        and open_issues == 0
        and not severe
        and grace_elapsed
    )

    # Middle-band aging: streak increment if in middle band this cycle
    in_middle_band_now = (
        CLOSURE_THRESHOLD_PCT <= rate60_pct < PROMOTION_THRESHOLD_PCT
    )
    new_streak = prior_middle_streak + 1 if in_middle_band_now else 0
    escalation_fired = new_streak >= MIDDLE_BAND_PERSISTENCE_CYCLES

    # Disposition (priority: HIGH-override > telemetry promotion > closure > escalation > defer).
    # HIGH-override carries a distinct label so the reader can see overlay
    # vs telemetry origin.
    if importance_override_fired and not telemetry_promotion_fired:
        disposition = "Option A treatment (HIGH personal_importance override; telemetry quiet)"
    elif promotion_fired:
        disposition = "Option A treatment (extraction-lane investment)"
    elif closure_fired:
        disposition = "Option E closure (documented-limitation)"
    elif escalation_fired:
        disposition = (
            f"REQUIRES EXPLICIT A/E ADJUDICATION "
            f"(persisted in middle band for {new_streak} cycles)"
        )
    else:
        disposition = "Defer to next cycle (continue telemetry)"

    return {
        "severe_defect_tag": severe,
        "added_cycle": added,
        "personal_importance": importance,
        "grace_period_elapsed": grace_elapsed,
        "grace_cycles_since_add": grace_n,
        "effective_grace_cycles": effective_grace,
        "open_user_issues": open_issues,
        "consecutive_middle_cycles": new_streak,
        "standard_arm_fired": standard_arm,
        "defect_arm_fired": defect_arm,
        "telemetry_promotion_fired": telemetry_promotion_fired,
        "importance_override_fired": importance_override_fired,
        "promotion_fired": promotion_fired,
        "closure_fired": closure_fired,
        "escalation_fired": escalation_fired,
        "disposition": disposition,
    }


def _render_class_section(
    class_meta: dict,
    rate30_pct: float, hits30: int, denom30: int,
    rate60_pct: float, hits60: int, denom60: int,
    result: dict,
) -> str:
    name = class_meta["name"]
    grace_note = (
        f"current: {class_meta.get('added_cycle', '?')} + "
        f"{result['grace_cycles_since_add']} cycles → "
        f"grace_period_elapsed: {result['grace_period_elapsed']}"
    )
    fire = lambda b: "FIRED" if b else "NOT FIRED"  # noqa: E731
    return (
        f"## {name}\n"
        f"- added_cycle: {class_meta.get('added_cycle', '?')}  ({grace_note})\n"
        f"- personal_importance: {result['personal_importance']}\n"
        f"- severe_defect_tag: {result['severe_defect_tag']}\n"
        f"- 30-day hit-rate: {rate30_pct:.1f}% ({hits30} / {denom30} qualified queries)\n"
        f"- 60-day hit-rate: {rate60_pct:.1f}% ({hits60} / {denom60} qualified queries)\n"
        f"- open_user_issues: {result['open_user_issues']}\n"
        f"- consecutive_middle_cycles: {result['consecutive_middle_cycles']}\n"
        f"- PROMOTION TRIGGER (standard arm: >={PROMOTION_THRESHOLD_PCT}% AND pain-signal): "
        f"{fire(result['standard_arm_fired'])}\n"
        f"- PROMOTION TRIGGER (defect-override arm: defect-tag AND >={DEFECT_OVERRIDE_THRESHOLD_PCT}%): "
        f"{fire(result['defect_arm_fired'])}\n"
        f"- IMPORTANCE OVERRIDE (HIGH forces Option A regardless of telemetry): "
        f"{fire(result['importance_override_fired'])}\n"
        f"- CLOSURE TRIGGER (<{CLOSURE_THRESHOLD_PCT}% AND 0 issues AND no defect-tag AND grace elapsed): "
        f"{fire(result['closure_fired'])}\n"
        f"- MIDDLE-BAND ESCALATION (>={MIDDLE_BAND_PERSISTENCE_CYCLES} consecutive cycles): "
        f"{fire(result['escalation_fired'])}\n"
        f"- {class_meta.get('added_cycle', 'v2.X')[:2]}.X disposition: {result['disposition']}\n"
        f"- defect_summary: {class_meta.get('defect_summary', 'n/a')}\n"
    )


def analyze(
    telemetry_log: Path,
    user_issues_doc: Path,
    previous_report: Path | None,
    current_cycle: str,
    now: float | None = None,
) -> tuple[str, dict]:
    """Run the full analysis. Returns (markdown_report, summary_dict)."""
    if now is None:
        now = time.time()
    rows = _load_telemetry(telemetry_log)
    qualified_30 = _qualified_in_window(rows, now, 30)
    qualified_60 = _qualified_in_window(rows, now, 60)

    # User issues: count entries since the last cycle tag date if
    # previous report provided; otherwise count all-time
    since_date: str | None = None
    if previous_report is not None and previous_report.exists():
        # Try to extract a "Generated:" or filename date for the since_date
        fn_match = re.search(r"(\d{4}-\d{2}-\d{2})", previous_report.name)
        if fn_match:
            since_date = fn_match.group(1)
    issue_counts = _load_user_issues(user_issues_doc, since_date=since_date)

    today_iso = datetime.datetime.utcfromtimestamp(now).strftime("%Y-%m-%d")
    sections: list[str] = []
    summary: dict = {
        "current_cycle": current_cycle,
        "generated_at": today_iso,
        "total_rows_in_log": len(rows),
        "qualified_queries_30d": len(qualified_30),
        "qualified_queries_60d": len(qualified_60),
        "user_issues_since_date": since_date,
        "classes": {},
    }
    for class_meta in DOCUMENTED_LIMITATION_CLASSES:
        name = class_meta["name"]
        rate30, hits30, denom30 = _hit_rate_pct(qualified_30, name)
        rate60, hits60, denom60 = _hit_rate_pct(qualified_60, name)
        open_issues = issue_counts.get(name, 0)
        prior_streak = _consecutive_middle_cycles(previous_report, name)
        result = _classify(
            class_meta=class_meta,
            rate30_pct=rate30,
            rate60_pct=rate60,
            open_issues=open_issues,
            prior_middle_streak=prior_streak,
            current_cycle=current_cycle,
        )
        sections.append(_render_class_section(
            class_meta, rate30, hits30, denom30, rate60, hits60, denom60, result,
        ))
        summary["classes"][name] = {
            "rate_30d_pct": rate30,
            "rate_60d_pct": rate60,
            "open_user_issues": open_issues,
            "promotion_fired": result["promotion_fired"],
            "closure_fired": result["closure_fired"],
            "escalation_fired": result["escalation_fired"],
            "disposition": result["disposition"],
        }

    header = (
        f"# v2.15+ Documented-Limitation Telemetry Report\n\n"
        f"> Generated: {today_iso}\n"
        f"> Current cycle: {current_cycle}\n"
        f"> Telemetry log: `{telemetry_log.relative_to(_REPO_ROOT) if telemetry_log.is_absolute() and _REPO_ROOT in telemetry_log.parents else telemetry_log}`\n"
        f"> Total rows in log: {len(rows)}\n"
        f"> Qualified queries (30d window, `rerank_top_5_non_empty=True`): {len(qualified_30)}\n"
        f"> Qualified queries (60d window): {len(qualified_60)}\n"
        f"> User issues counted since: {since_date or '(all-time)'}\n\n"
        f"Per-class disposition follows. Disposition priority is:\n"
        f"promotion > closure > escalation > defer-to-next-cycle.\n\n"
        f"---\n\n"
    )
    return header + "\n".join(sections), summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--current-cycle", default="v2.15",
                   help='Cycle label like "v2.15" or "v2.16"')
    p.add_argument("--telemetry-log", type=Path,
                   default=DEFAULT_TELEMETRY_LOG,
                   help="Path to document_class_hits.jsonl rolling log")
    p.add_argument("--user-issues-doc", type=Path,
                   default=DEFAULT_USER_ISSUES,
                   help="Path to USER_ISSUES.md")
    p.add_argument("--previous-report", type=Path, default=None,
                   help="Path to prior TELEMETRY_REPORT_*.md (for "
                        "consecutive_middle_cycles streak lookup)")
    p.add_argument("--output", type=Path, default=None,
                   help='Output report path; defaults to '
                        '"docs/TELEMETRY_REPORT_<today>.md"')
    args = p.parse_args()

    report, summary = analyze(
        telemetry_log=args.telemetry_log,
        user_issues_doc=args.user_issues_doc,
        previous_report=args.previous_report,
        current_cycle=args.current_cycle,
    )

    if args.output is None:
        today = datetime.date.today().isoformat()
        args.output = _REPO_ROOT / f"docs/TELEMETRY_REPORT_{today}.md"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")

    print(f"Wrote {args.output.relative_to(_REPO_ROOT) if _REPO_ROOT in args.output.parents else args.output}")
    print(f"Summary:")
    for name, cls in summary["classes"].items():
        flags = []
        if cls["promotion_fired"]:
            flags.append("PROMOTE")
        if cls["closure_fired"]:
            flags.append("CLOSE")
        if cls["escalation_fired"]:
            flags.append("ESCALATE")
        if not flags:
            flags.append("defer")
        print(f"  {name}: {'+'.join(flags)} (30d={cls['rate_30d_pct']:.1f}%, "
              f"60d={cls['rate_60d_pct']:.1f}%, issues={cls['open_user_issues']})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
