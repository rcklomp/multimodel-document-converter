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
- `MISSING_CHAPTERS` (conditional: edge-only low-content EPUB spine items)
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
    EpubChapterInfo,
    _F4_HARD_FALLBACK_SENTINEL,
    Issue,
    _epub_chapter_coverage_issues,
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


def _chapter(index: int, name: str, text_len: int, sample: str = "") -> EpubChapterInfo:
    return EpubChapterInfo(
        index=index,
        name=name,
        text_len=text_len,
        text_sample=sample,
    )


def _text_chunk_on_epub_page(page_number: int):
    return {
        "modality": "text",
        "metadata": {
            "page_number": page_number,
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


def test_missing_epub_leading_structural_chapters_are_documented_advisory(
    monkeypatch,
) -> None:
    """Leading low-content title/colophon spine items can be stripped by
    Docling's HTML parser without representing real body-content loss.
    """
    chapters = [
        _chapter(1, "titlepage.xhtml", 0),
        _chapter(2, "OEBPS/c9.xhtml", 536, "Colofon Copyright: Example"),
        _chapter(3, "OEBPS/chapter1.xhtml", 4000, "Chapter 1 real content"),
        _chapter(4, "OEBPS/chapter2.xhtml", 4000, "Chapter 2 real content"),
    ]
    monkeypatch.setattr(
        "qa_full_conversion._epub_spine_chapters",
        lambda _path: chapters,
    )

    issues = _epub_chapter_coverage_issues(
        Path("book.epub"),
        [_text_chunk_on_epub_page(3000), _text_chunk_on_epub_page(4000)],
        allow_missing=False,
    )

    assert len(issues) == 1
    assert issues[0].severity == "WARN"
    assert issues[0].code == "MISSING_CHAPTERS"
    assert _warn_is_documented_advisory(issues[0], []) is True


def test_missing_epub_internal_chapter_is_FAIL(monkeypatch) -> None:
    """A missing internal chapter is not a structural edge artifact; it is
    real content-loss risk even if the missing item is short.
    """
    chapters = [
        _chapter(1, "OEBPS/chapter1.xhtml", 4000, "Chapter 1 real content"),
        _chapter(2, "OEBPS/titlepage.xhtml", 0),
        _chapter(3, "OEBPS/chapter3.xhtml", 4000, "Chapter 3 real content"),
    ]
    monkeypatch.setattr(
        "qa_full_conversion._epub_spine_chapters",
        lambda _path: chapters,
    )

    issues = _epub_chapter_coverage_issues(
        Path("book.epub"),
        [_text_chunk_on_epub_page(1000), _text_chunk_on_epub_page(3000)],
        allow_missing=False,
    )

    assert len(issues) == 1
    assert issues[0].severity == "FAIL"
    assert issues[0].code == "MISSING_CHAPTERS"
    assert _warn_is_documented_advisory(issues[0], []) is False


def test_missing_epub_content_bearing_edge_chapter_is_FAIL(monkeypatch) -> None:
    """The edge carve-out is for low-value structural stubs only. A real
    first chapter that disappears must remain a hard failure.
    """
    chapters = [
        _chapter(1, "OEBPS/introduction.xhtml", 2500, "Introduction with real prose"),
        _chapter(2, "OEBPS/chapter1.xhtml", 4000, "Chapter 1 real content"),
        _chapter(3, "OEBPS/chapter2.xhtml", 4000, "Chapter 2 real content"),
    ]
    monkeypatch.setattr(
        "qa_full_conversion._epub_spine_chapters",
        lambda _path: chapters,
    )

    issues = _epub_chapter_coverage_issues(
        Path("book.epub"),
        [_text_chunk_on_epub_page(2000), _text_chunk_on_epub_page(3000)],
        allow_missing=False,
    )

    assert len(issues) == 1
    assert issues[0].severity == "FAIL"
    assert issues[0].code == "MISSING_CHAPTERS"


def test_missing_epub_short_opaque_edge_chapter_is_FAIL(monkeypatch) -> None:
    """Short is not enough. An edge chapter must have structural
    name/text evidence before MISSING_CHAPTERS can be advisory.
    """
    chapters = [
        _chapter(1, "OEBPS/c1.xhtml", 400, "A concise but real opening essay."),
        _chapter(2, "OEBPS/chapter1.xhtml", 4000, "Chapter 1 real content"),
    ]
    monkeypatch.setattr(
        "qa_full_conversion._epub_spine_chapters",
        lambda _path: chapters,
    )

    issues = _epub_chapter_coverage_issues(
        Path("book.epub"),
        [_text_chunk_on_epub_page(2000)],
        allow_missing=False,
    )

    assert len(issues) == 1
    assert issues[0].severity == "FAIL"
    assert issues[0].code == "MISSING_CHAPTERS"


def test_missing_chapters_warn_without_edge_structural_message_is_NOT_advisory() -> None:
    """Defensive: the allow-list code alone is not enough. The WARN must
    be the narrow edge-structural case emitted by the EPUB coverage gate.
    """
    issue = Issue("WARN", "MISSING_CHAPTERS", "internal chapter 7 missing")
    assert _warn_is_documented_advisory(issue, []) is False


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
    discipline). MISSING_CHAPTERS was added in v2.10 Phase 7
    (KI_EPUB_EXTRACTION_LANE_REWRITE), but the advisory path is pinned
    separately above: only edge low-content structural spine items are
    allowed to remain WARN.
    """
    assert _ALLOWED_ADVISORY_WARN_CODES == frozenset({
        "ASSET_TINY",
        "PAGE_COUNT_UNKNOWN",
        "SCRIPT_ADVISORY_FAIL",
        "VISION_HARD_FALLBACK_RATE",
        "MISSING_CHAPTERS",
    })


def test_f4_sentinel_constant_pinned() -> None:
    """The F4 sentinel must match `scripts/enrich_image_chunks_v29.py:
    SHORT_RESPONSE_SENTINEL`."""
    assert _F4_HARD_FALLBACK_SENTINEL == "complex_asset_short_response_after_retry"
