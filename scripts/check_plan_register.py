#!/usr/bin/env python3
"""Structural check of docs/PLAN_QUALITY_REMEDIATION_V1_REGISTER.md against its plan.

Runs in a clean clone (reads only committed files). Fails non-zero if:
- an id appears twice, or a row lacks exactly one disposition from the closed vocabulary;
- a DECIDE/BLOCKED row has no answer-by date or no owner;
- a DECIDE row references a decision id that is absent from the plan's decision table;
- a FIX row references a work package that is absent from the plan;
- the counts line disagrees with the rows.

The completeness of the id set against the four 2026-09-29 audit registers was verified when the
register was generated (the audit registers are local, gitignored evidence); this script guards
the committed register's internal integrity afterwards.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTER = ROOT / "docs" / "PLAN_QUALITY_REMEDIATION_V1_REGISTER.md"
PLAN = ROOT / "docs" / "PLAN_QUALITY_REMEDIATION_V1.md"
VERBS = {"FIX", "DECIDE", "CLOSE", "BLOCKED"}
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def main() -> int:
    plan = PLAN.read_text(encoding="utf-8")
    # A decision counts only when its row states a decision (not an "(unused id ...)" placeholder);
    # a work package counts only when the plan DEFINES it as a bold heading, not when it is merely
    # mentioned (a row pointing at a removed package would otherwise pass).
    decisions = set(re.findall(r"^\| (D-\d+) \| (?!\(unused)", plan, flags=re.M))
    wp_ids = set(re.findall(r"\*\*(WP-[A-Za-z0-9.]+)", plan))
    problems: list[str] = []
    seen: dict[str, int] = {}
    counts: dict[str, int] = {}
    rows = 0
    for line in REGISTER.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\| ((?:INV|G0|DC|AP)-\d+) \|", line)
        if not m:
            continue
        rows += 1
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 9:
            problems.append(f"{m.group(1)}: expected 9 cells, got {len(cells)}")
            continue
        iid, _sev, _cls, _title, verb, ref, owner, answer_by, _note = cells
        seen[iid] = seen.get(iid, 0) + 1
        counts[verb] = counts.get(verb, 0) + 1
        if verb not in VERBS:
            problems.append(f"{iid}: disposition {verb!r} not in {sorted(VERBS)}")
        if verb in ("DECIDE", "BLOCKED"):
            if not DATE.match(answer_by):
                problems.append(f"{iid}: {verb} without an answer-by date")
            if owner != "owner":
                problems.append(f"{iid}: {verb} without owner=owner")
        if verb == "DECIDE" and ref not in decisions:
            problems.append(f"{iid}: decision {ref!r} not in the plan's decision table")
        if verb == "FIX":
            refs = [r.strip() for r in ref.split("/")]
            for r in refs:
                # "WP-G1/G3" style: the tail piece lacks the prefix
                cand = r if r.startswith("WP-") else "WP-" + r
                if cand not in wp_ids:
                    problems.append(f"{iid}: work package {cand!r} not in the plan")
    for iid, n in seen.items():
        if n != 1:
            problems.append(f"{iid}: appears {n} times")
    counts_line = re.search(
        r"^Counts: (.*); total ids=(\d+)\.", REGISTER.read_text(encoding="utf-8"), flags=re.M
    )
    if not counts_line:
        problems.append("counts line missing")
    else:
        claimed = dict(kv.split("=") for kv in counts_line.group(1).split(", "))
        if {k: int(v) for k, v in claimed.items()} != counts or int(counts_line.group(2)) != rows:
            problems.append(
                f"counts line {counts_line.group(0)!r} disagrees with rows {counts} total={rows}"
            )
    if problems:
        print("REGISTER_FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print(f"REGISTER_OK rows={rows} counts={counts} decisions={len(decisions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
