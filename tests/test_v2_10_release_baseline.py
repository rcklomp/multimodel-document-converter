"""PLAN_V2.10 Phase 8 — v2.10.0-rc1 release-baseline regression pins.

These tests freeze the v2.10 release-baseline so any future drift is
visible in the diff. Evidence is the tracked
`docs/QUALITY_SNAPSHOT_2026-05-16_v2.10_after.md` snapshot (not
ignored `data/` or `output/` artifacts — `AGENT-EVIDENCE-01`).

Three pins:

1. Engine version is `2.10.0-rc1`; schema version is `2.7.0`
   (chunk-shape contract unchanged since v2.7).
2. The v2.10.0-rc1 AFTER snapshot exists, names the correct
   predecessor (v2.9.0-rc1), and reports the corpus headline of
   34 PASS / 0 WARN / 0 FAIL with the 16 QA_PASS + 18
   QA_PASS_WITH_ADVISORIES breakdown.
3. The advisory-allowance frozenset is unchanged versus v2.9.0-rc1
   (no new code added in Phase 8 per the doc-budget rule).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

V2_10_AFTER_SNAPSHOT = (
    REPO_ROOT / "docs" / "QUALITY_SNAPSHOT_2026-05-16_v2.10_after.md"
)
V2_9_AFTER_SNAPSHOT = (
    REPO_ROOT / "docs" / "QUALITY_SNAPSHOT_2026-05-11_v2.9.0-rc1_after.md"
)


def test_engine_and_schema_version_pinned() -> None:
    from mmrag_v2.version import __engine_version__, __schema_version__

    assert __engine_version__ == "2.10.0-rc1"
    assert __schema_version__ == "2.7.0"


def test_v2_10_after_snapshot_exists() -> None:
    assert V2_10_AFTER_SNAPSHOT.exists(), (
        f"v2.10 AFTER snapshot missing: {V2_10_AFTER_SNAPSHOT}"
    )
    text = V2_10_AFTER_SNAPSHOT.read_text(encoding="utf-8")
    # The snapshot must name its predecessor for delta reproducibility
    # (parallel to the v2.9.0-rc1 snapshot's predecessor reference).
    assert "QUALITY_SNAPSHOT_2026-05-11_v2.9.0-rc1_after.md" in text


def test_v2_10_after_snapshot_corpus_headline() -> None:
    text = V2_10_AFTER_SNAPSHOT.read_text(encoding="utf-8")
    # The headline table row "Total PASS-class | 26 | 34 | +8".
    pass_row = re.search(
        r"\|\s*\*\*Total PASS-class\*\*\s*\|\s*26\s*\|\s*\*\*34\*\*\s*\|\s*\*\*\+8\*\*\s*\|",
        text,
    )
    assert pass_row, "v2.10 AFTER headline must show Total PASS-class 26 → 34 (+8)"
    fail_row = re.search(
        r"\|\s*`QA_FAIL`\s*\|\s*8\s*\|\s*\*\*0\*\*\s*\|\s*\*\*−8\*\*\s*\|",
        text,
    )
    assert fail_row, "v2.10 AFTER headline must show QA_FAIL 8 → 0 (−8)"


def test_v2_10_pass_breakdown_pin() -> None:
    """Pin the exact PASS distribution: 16 QA_PASS + 18 QA_PASS_WITH_ADVISORIES.

    Phase 8 closure brought the eight v2.9.0-rc1 signed-deferral
    FAILs to PASS-class. The breakdown is part of the release
    contract; if the corpus shape ever drifts (e.g. an advisory promotes
    or demotes), this test fails so the change is reviewed.
    """
    text = V2_10_AFTER_SNAPSHOT.read_text(encoding="utf-8")
    qa_pass_row = re.search(
        r"\|\s*`QA_PASS`\s*\|\s*12\s*\|\s*\*\*16\*\*\s*\|\s*\+4\s*\|",
        text,
    )
    assert qa_pass_row, "v2.10 AFTER must show QA_PASS 12 → 16 (+4)"
    advisory_row = re.search(
        r"\|\s*`QA_PASS_WITH_ADVISORIES`\s*\|\s*14\s*\|\s*\*\*18\*\*\s*\|\s*\+4\s*\|",
        text,
    )
    assert advisory_row, (
        "v2.10 AFTER must show QA_PASS_WITH_ADVISORIES 14 → 18 (+4)"
    )


def test_advisory_allowance_unchanged_vs_v2_9_rc1() -> None:
    """Phase 8 must NOT add new advisory codes — the doc-budget rule
    (`AGENT-DOCS-01`) and `docs/QUALITY_GATES.md` 4-step procedure
    forbid it without explicit user sign-off."""
    from qa_full_conversion import _ALLOWED_ADVISORY_WARN_CODES  # noqa: E402

    # Same set the v2.9.0-rc1 cycle landed; pinned here so a Phase 8 or
    # later silent expansion would fail this test in addition to
    # tests/test_qa_advisory_promotion.py::test_allowed_advisory_codes_match_expected_set.
    assert _ALLOWED_ADVISORY_WARN_CODES == frozenset({
        "ASSET_TINY",
        "PAGE_COUNT_UNKNOWN",
        "SCRIPT_ADVISORY_FAIL",
        "VISION_HARD_FALLBACK_RATE",
        "MISSING_CHAPTERS",
    })


def test_v2_10_predecessor_snapshot_kept() -> None:
    """Per `docs/QUALITY_SNAPSHOT_2026-05-16_v2.10_after.md` headline,
    the v2.9.0-rc1 AFTER snapshot must remain tracked so the v2.10
    delta column is reproducible."""
    assert V2_9_AFTER_SNAPSHOT.exists(), (
        f"v2.9.0-rc1 predecessor snapshot must remain tracked: {V2_9_AFTER_SNAPSHOT}"
    )


def test_search_qdrant_api_key_is_env_var() -> None:
    """v2.10 Phase 8 housekeeping: the hard-coded Dashscope literal
    in `scripts/search_qdrant.py` has been replaced with
    `os.environ.get("DASHSCOPE_API_KEY", ...)`. The script must not
    re-introduce any `sk-...` literal."""
    text = (REPO_ROOT / "scripts" / "search_qdrant.py").read_text(encoding="utf-8")
    assert 'os.environ.get("DASHSCOPE_API_KEY"' in text, (
        "search_qdrant.py must read DASHSCOPE_API_KEY from the env"
    )
    # No `sk-` literal in production code.
    assert not re.search(r'"sk-[A-Za-z0-9]{20,}"', text), (
        "search_qdrant.py must not contain any sk-... API-key literal"
    )
