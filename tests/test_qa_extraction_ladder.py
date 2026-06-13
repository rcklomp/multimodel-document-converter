"""WS1b — extraction-ladder verdict signal (PLAN_FIDELITY_ORACLE_FIRST_V1 Section 3').

`scripts/qa_full_conversion.py:_extraction_ladder_issues` promotes the fail-closed
ladder provenance stamps from display-only into a QA verdict signal:

- a code-bearing doc with ANY laddered page -> FAIL `EXTRACTION_DEGRADED_CODE`
  (tier-2 docling / tier-3 PyMuPDF are vacuous on code; laddered code = data loss);
- a non-code doc -> ladder-served fraction ABOVE the 2% Phase-4 bound is a real
  WARN, at/below the bound a documented advisory `EXTRACTION_LADDER_SERVED`;
- legacy outputs (no stamps) and healthy runs (degraded == 0) raise nothing.

These tests pin the contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from qa_full_conversion import (  # noqa: E402
    _ALLOWED_ADVISORY_WARN_CODES,
    _extraction_ladder_issues,
    _warn_is_documented_advisory,
)


def _meta(engine="mineru_qwen_hybrid", degraded=0, total=20, fallback="docling_fast"):
    return {
        "extraction_engine": engine,
        "extraction_degraded_pages": degraded,
        "extraction_fallback": fallback,
        "total_pages": total,
    }


def _text_chunk():
    return {"modality": "text", "content": "hello world"}


def _code_chunk():
    return {"modality": "code", "content": "def f():\n    return 1"}


def test_healthy_run_no_issue() -> None:
    """degraded == 0: the primary served every page -> nothing raised."""
    assert _extraction_ladder_issues(_meta(degraded=0), [_code_chunk()]) == []


def test_legacy_output_no_stamps_no_issue() -> None:
    """No extraction_engine stamp (legacy output) -> never judged."""
    meta = {"total_pages": 20}  # no extraction_* keys
    assert _extraction_ladder_issues(meta, [_code_chunk()]) == []


def test_laddered_code_doc_hard_fails() -> None:
    """ANY laddered page on a code-bearing doc is a hard FAIL (data loss)."""
    issues = _extraction_ladder_issues(_meta(degraded=1, total=20), [_code_chunk(), _text_chunk()])
    assert len(issues) == 1
    assert issues[0].severity == "FAIL"
    assert issues[0].code == "EXTRACTION_DEGRADED_CODE"
    # never reclassifiable as an advisory
    assert _warn_is_documented_advisory(issues[0], []) is False


def test_laddered_noncode_within_bound_is_advisory() -> None:
    """Non-code doc, ladder fraction <= 2% -> documented advisory WARN."""
    # 1/100 = 1% <= 2%
    issues = _extraction_ladder_issues(_meta(degraded=1, total=100), [_text_chunk()])
    assert len(issues) == 1
    assert issues[0].severity == "WARN"
    assert issues[0].code == "EXTRACTION_LADDER_SERVED"
    assert "within the 2% Phase-4 bound" in issues[0].message
    assert _warn_is_documented_advisory(issues[0], []) is True


def test_laddered_noncode_above_bound_is_real_warn() -> None:
    """Non-code doc, ladder fraction > 2% -> real WARN (blocks QA_PASS)."""
    # 5/20 = 25% > 2%
    issues = _extraction_ladder_issues(_meta(degraded=5, total=20), [_text_chunk()])
    assert len(issues) == 1
    assert issues[0].severity == "WARN"
    assert issues[0].code == "EXTRACTION_LADDER_SERVED"
    assert "exceeds the 2% Phase-4 bound" in issues[0].message
    assert _warn_is_documented_advisory(issues[0], []) is False


def test_code_check_precedes_bound() -> None:
    """A code doc fails even when the ladder fraction is tiny (well within 2%)."""
    issues = _extraction_ladder_issues(_meta(degraded=1, total=1000), [_code_chunk()])
    assert issues[0].code == "EXTRACTION_DEGRADED_CODE"
    assert issues[0].severity == "FAIL"


def test_advisory_code_is_registered() -> None:
    """The advisory code must be in the allowed set (doc/test contract)."""
    assert "EXTRACTION_LADDER_SERVED" in _ALLOWED_ADVISORY_WARN_CODES
    # the hard-fail code must NOT be an allowed advisory
    assert "EXTRACTION_DEGRADED_CODE" not in _ALLOWED_ADVISORY_WARN_CODES
