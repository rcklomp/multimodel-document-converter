"""v2.15 Option F — documented-limitation document classes.

Single source of truth for Phase 3 telemetry. Read by:
  - `scripts/analyze_doc_class_telemetry.py` (per-class hit-rate
    computation against the rolling log; applies promotion / closure
    / middle-band / grace-period rules)
  - `synthetic_soak.py` (per-query telemetry write path; compares
    each query's reranked top-5 doc_ids against `class_names()`)
  - `docs/CYCLE_OPEN_CHECKLIST.md` (cycle-open process reads the
    analyzer report; trigger-fired classes are required-decision
    items for that cycle's plan)

Per DECISIONS.md "v2.15 Documented-Limitation Telemetry Threshold"
the per-class transition rules are:

  PROMOTION (F → A):
    (hit_rate_30d >= PROMOTION_THRESHOLD_PCT
     AND (severe_defect_tag OR open_user_issues >= 1))
    OR
    (severe_defect_tag AND hit_rate_30d >= DEFECT_OVERRIDE_THRESHOLD_PCT)

  CLOSURE (F → E):
    hit_rate_60d < CLOSURE_THRESHOLD_PCT
    AND open_user_issues == 0
    AND severe_defect_tag == False
    AND (current_cycle_n - added_cycle_n) >= NEW_CLASS_GRACE_CYCLES

  MIDDLE BAND (F → F):
    CLOSURE_THRESHOLD_PCT <= hit_rate_60d < PROMOTION_THRESHOLD_PCT

  MIDDLE-BAND ESCALATION (F → explicit A/E adjudication):
    consecutive_middle_cycles >= MIDDLE_BAND_PERSISTENCE_CYCLES

Audit history: threshold constants and rule schema co-evolved across
Round-2 Finding 1 (define rule), Round-4 Finding 2 (closure arm),
Round-5 Findings 1 + 3 (pain-signal + middle-band aging), Round-6
Finding 1 (defect-override arm + closure defect-tag clause), Round-7
Finding 3 (new-class grace period). See PLAN_V2.15.md Appendix A.
"""
from __future__ import annotations

from typing import Optional

# Thresholds — keep in sync with DECISIONS.md
# "v2.15 Documented-Limitation Telemetry Threshold" entry.
PROMOTION_THRESHOLD_PCT = 5
"""Standard-arm corpus-frequency floor for F→A promotion (30d window)."""

CLOSURE_THRESHOLD_PCT = 1
"""F→E closure floor (60d window)."""

DEFECT_OVERRIDE_THRESHOLD_PCT = 1
"""Defect-override-arm corpus-frequency floor for F→A promotion
(Round-6 Finding 1). Triggers when severe_defect_tag=True and
hit-rate >= this floor — closes the suppression death spiral that
otherwise blocks promotion for known-defective classes whose users
abandon queries."""

MIDDLE_BAND_PERSISTENCE_CYCLES = 3
"""Number of consecutive middle-band cycles before forced explicit
A/E adjudication (Round-5 Finding 3)."""

NEW_CLASS_GRACE_CYCLES = 2
"""New-class grace period before auto-closure is eligible
(Round-7 Finding 3). Closes the silent-decay-of-new-class failure
mode where a v2.16+ addition with severe defects but no manual tag
could be auto-closed before any human review."""


# Documented-limitation registry. Entries on v2.15 entry are
# CarOK_voorraadtelling and Fluent_Python; both have severe extraction
# defects documented in prior-cycle quality snapshots.
#
# Field schema:
#   name              : doc directory identifier (matches CANONICAL_34
#                       in synthetic_soak.py and doc_id payloads in
#                       Qdrant)
#   severe_defect_tag : True if class has documented extraction defects
#                       from prior cycle quality snapshots. Defaults
#                       False for newly-added classes; gain True via
#                       explicit DECISIONS.md entry after defect
#                       diagnosis.
#   added_cycle       : cycle the class was added to this list (string
#                       like "v2.15"). Used by `analyze_doc_class_telemetry.py`
#                       to apply the 2-cycle grace period from Round-7
#                       Finding 3.
#   defect_summary    : one-line rationale for severe_defect_tag=True;
#                       points at the canonical evidence doc.
DOCUMENTED_LIMITATION_CLASSES: list[dict] = [
    {
        "name": "CarOK_voorraadtelling",
        "severe_defect_tag": True,
        "added_cycle": "v2.15",
        "defect_summary": (
            "v2.14 P1 mini-soak Format -26.9pp regression; "
            "VLM tables + flat-prose duplicates coexist post "
            "force_table_vlm; retrieval picks prose 29/30 times. "
            "See QUALITY_SNAPSHOT_2026-05-23_v2.14_after.md §1 "
            "Phase 1 PARTIAL row."
        ),
    },
    {
        "name": "Fluent_Python",
        "severe_defect_tag": True,
        "added_cycle": "v2.15",
        "defect_summary": (
            "Docling extraction-layer prose+code intermixing at page "
            "boundaries; truncated CODE chunks (e.g. p326 ends mid-"
            "statement at '    return'). HybridChunker post-merge "
            "tested and reverted (fires 0x in production). See "
            "PROJECT_STATUS.md v2.14 Phase 6 PARTIAL row."
        ),
    },
]


def class_names() -> list[str]:
    """List of documented-limitation class names. Used by the soak
    harness telemetry write path to compute hits per query."""
    return [c["name"] for c in DOCUMENTED_LIMITATION_CLASSES]


def get_class(name: str) -> Optional[dict]:
    """Return the config entry for `name`, or None if not registered."""
    for c in DOCUMENTED_LIMITATION_CLASSES:
        if c["name"] == name:
            return c
    return None


def cycles_since(added_cycle: str, current_cycle: str) -> int:
    """Compute integer cycle-distance between two cycle labels.

    Used by `analyze_doc_class_telemetry.py` to apply the
    NEW_CLASS_GRACE_CYCLES rule. Cycle labels are like "v2.15",
    "v2.16", "v2.17" — minor-version increments.

    Returns 0 if `current_cycle == added_cycle`; negative if
    `current_cycle < added_cycle` (caller error; should not happen);
    positive integer otherwise.
    """
    def _minor(label: str) -> int:
        # Accept "v2.15", "2.15", "v2.15.0", "2.15.0" → 15
        s = label.lstrip("v")
        parts = s.split(".")
        if len(parts) < 2:
            raise ValueError(f"unrecognized cycle label: {label!r}")
        return int(parts[1])

    return _minor(current_cycle) - _minor(added_cycle)
