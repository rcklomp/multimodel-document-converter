"""Phase G — advisory-warning allowance / QA_PASS_WITH_ADVISORIES
(`docs/PLAN_V2.9.md` §3 Phase G, `docs/DECISIONS.md` Retrieval-Value Test,
`docs/QUALITY_GATES.md` "Advisory Warning Classes").

`scripts/qa_full_conversion.py:_warn_is_documented_advisory` classifies
each WARN-level Issue as either:
- documented advisory (allowed, does not block QA_PASS_WITH_ADVISORIES), or
- real warning (still blocks ship; final_status = QA_WARN).

Allowed advisory codes:
- `ASSET_TINY` (unconditional)
- `PAGE_COUNT_UNKNOWN` (unconditional; EPUB lane)
- `SCRIPT_ADVISORY_FAIL` (unconditional; qa_semantic_fidelity exits 0)
- `VISION_HARD_FALLBACK_RATE` (conditional: every hard_fallback chunk
  must carry the F4 sentinel `complex_asset_short_response_after_retry`)

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
    _F4_HARD_FALLBACK_SENTINEL,
    Issue,
    _warn_is_documented_advisory,
)


def _f4_hf_chunk(error: str = _F4_HARD_FALLBACK_SENTINEL):
    return {
        "modality": "image",
        "metadata": {
            "vision_status": "hard_fallback",
            "vision_error": error,
        },
    }


def _ok_image_chunk():
    return {
        "modality": "image",
        "metadata": {
            "vision_status": "complete",
            "visual_description": "Real description.",
        },
    }


# ---------------------------------------------------------------------------
# Unconditional allowed advisories
# ---------------------------------------------------------------------------


def test_asset_tiny_is_documented_advisory() -> None:
    issue = Issue("WARN", "ASSET_TINY", "1 asset under 1KB.")
    assert _warn_is_documented_advisory(issue, []) is True


def test_page_count_unknown_is_documented_advisory() -> None:
    issue = Issue("WARN", "PAGE_COUNT_UNKNOWN", "EPUB has no page count.")
    assert _warn_is_documented_advisory(issue, []) is True


def test_script_advisory_fail_is_documented_advisory() -> None:
    issue = Issue("WARN", "SCRIPT_ADVISORY_FAIL", "advisory only.")
    assert _warn_is_documented_advisory(issue, []) is True


def test_unknown_warn_code_is_NOT_advisory() -> None:
    issue = Issue("WARN", "SOME_UNDOCUMENTED_FUTURE_CODE", "?")
    assert _warn_is_documented_advisory(issue, []) is False


def test_FAIL_severity_never_advisory() -> None:
    issue = Issue("FAIL", "ASSET_TINY", "even if code matches.")
    assert _warn_is_documented_advisory(issue, []) is False


# ---------------------------------------------------------------------------
# Conditional VISION_HARD_FALLBACK_RATE
# ---------------------------------------------------------------------------


def test_vision_hf_rate_with_all_F4_sentinels_is_advisory() -> None:
    chunks = [_f4_hf_chunk(), _f4_hf_chunk(), _f4_hf_chunk(), _ok_image_chunk()]
    issue = Issue(
        "WARN", "VISION_HARD_FALLBACK_RATE",
        "3/4 image chunks are hard_fallback (75 %; limit 5 %).",
    )
    assert _warn_is_documented_advisory(issue, chunks) is True


def test_vision_hf_rate_with_non_F4_sentinel_is_NOT_advisory() -> None:
    chunks = [
        _f4_hf_chunk(),
        _f4_hf_chunk(error="asset not found: assets/missing.png"),
        _ok_image_chunk(),
    ]
    issue = Issue(
        "WARN", "VISION_HARD_FALLBACK_RATE",
        "2/3 image chunks are hard_fallback.",
    )
    assert _warn_is_documented_advisory(issue, chunks) is False


def test_vision_hf_rate_with_no_hard_fallback_chunks_is_NOT_advisory() -> None:
    """Defensive: if VISION_HARD_FALLBACK_RATE WARN was raised but the
    corpus has zero hard_fallback chunks, something is inconsistent —
    do not treat as advisory."""
    chunks = [_ok_image_chunk(), _ok_image_chunk()]
    issue = Issue("WARN", "VISION_HARD_FALLBACK_RATE", "?")
    assert _warn_is_documented_advisory(issue, chunks) is False


def test_vision_hf_rate_with_missing_vision_error_is_NOT_advisory() -> None:
    """A hard_fallback chunk without vision_error metadata cannot prove
    F4-sentinel coverage; reject as advisory."""
    chunks = [{"modality": "image", "metadata": {"vision_status": "hard_fallback"}}]
    issue = Issue("WARN", "VISION_HARD_FALLBACK_RATE", "?")
    assert _warn_is_documented_advisory(issue, chunks) is False


# ---------------------------------------------------------------------------
# Allowed-code set is intentionally small
# ---------------------------------------------------------------------------


def test_allowed_advisory_codes_match_expected_set() -> None:
    """Pin the set so changes to QUALITY_GATES.md governance are
    visible in diffs (matching the SCAN0013 GATE_PASS variant
    discipline)."""
    assert _ALLOWED_ADVISORY_WARN_CODES == frozenset({
        "ASSET_TINY",
        "PAGE_COUNT_UNKNOWN",
        "SCRIPT_ADVISORY_FAIL",
        "VISION_HARD_FALLBACK_RATE",
    })


def test_f4_sentinel_constant_pinned() -> None:
    """The F4 sentinel must match `scripts/enrich_image_chunks_v29.py:
    SHORT_RESPONSE_SENTINEL`."""
    assert _F4_HARD_FALLBACK_SENTINEL == "complex_asset_short_response_after_retry"
