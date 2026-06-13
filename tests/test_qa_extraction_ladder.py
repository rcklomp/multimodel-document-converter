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
    _content_emptiness_issues,
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


# ---------------------------------------------------------------------------
# WS1a — content-emptiness visibility on no-source-pdf runs
# ---------------------------------------------------------------------------


def _chunk_on_page(page: int, modality="text", content="x"):
    return {"modality": modality, "content": content, "metadata": {"page_number": page}}


def test_emptiness_advisory_fires_without_source_when_orphan_pages_high() -> None:
    """No --source-pdf, >15% pages produced no chunk -> advisory WARN."""
    # total 10 pages, only pages 1-2 present -> 8/10 = 80% orphan
    chunks = [_chunk_on_page(1), _chunk_on_page(2)]
    issues = _content_emptiness_issues(_meta(degraded=0, total=10), chunks, False)
    assert len(issues) == 1
    assert issues[0].severity == "WARN"
    assert issues[0].code == "CONTENT_EMPTY_PAGES_UNVERIFIED"
    assert _warn_is_documented_advisory(issues[0], []) is True  # always advisory


def test_emptiness_inert_with_source_pdf() -> None:
    """With --source-pdf, MISSING_PAGES is the authority; this is inert."""
    chunks = [_chunk_on_page(1)]
    assert _content_emptiness_issues(_meta(total=10), chunks, True) == []


def test_emptiness_quiet_below_bound() -> None:
    """A couple of blank dividers (<=15%) raise nothing."""
    chunks = [_chunk_on_page(p) for p in range(1, 10)]  # 9/10 present -> 10% orphan
    assert _content_emptiness_issues(_meta(total=10), chunks, False) == []


def test_emptiness_quiet_on_legacy_output() -> None:
    """No extraction stamps (legacy) -> not judged."""
    chunks = [_chunk_on_page(1)]
    assert _content_emptiness_issues({"total_pages": 10}, chunks, False) == []


def test_emptiness_never_fails() -> None:
    """Even 100% orphan pages is WARN (advisory), never FAIL - blank-vs-lost
    is unverifiable without the source."""
    issues = _content_emptiness_issues(_meta(total=10), [], False)
    assert issues and all(i.severity == "WARN" for i in issues)
