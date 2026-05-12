"""Phase B2 — "intentionally left blank" boilerplate page classification
(`docs/PLAN_V2.9.md` §3 Phase B2).

`scripts/qa_full_conversion.py:_is_intentionally_blank_text` recognizes
the publisher-template chapter-divider boilerplate ("This page
intentionally left blank", optionally duplicated by a backing layer)
as blank-equivalent. Pages whose only text content is this boilerplate
now count toward `MISSING_PAGES_BLANK` (informational) rather than
`MISSING_PAGES` (hard fail).

Affected docs (Phase A diagnostic): Greenhouse_Design p3, p11, p23.

The detector is restrictive on purpose: it must NOT fire on pages
where the boilerplate is one detail amidst real content (e.g., a
preface that mentions blank pages in a discussion). The length cap
(≤120 chars) is the structural guard for this.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from qa_full_conversion import _is_intentionally_blank_text  # noqa: E402


# Real Greenhouse_Design p3 content (verified 2026-05-11 via fitz)
GREENHOUSE_P3_TEXT = (
    "This page intentionally left blank\n"
    "This page intentionally left blank"
)


def test_greenhouse_p3_p11_p23_shape_classified_blank() -> None:
    assert _is_intentionally_blank_text(GREENHOUSE_P3_TEXT) is True


def test_single_line_boilerplate_classified_blank() -> None:
    assert _is_intentionally_blank_text("This page intentionally left blank") is True


def test_uppercase_boilerplate_classified_blank() -> None:
    assert _is_intentionally_blank_text("THIS PAGE INTENTIONALLY LEFT BLANK") is True


def test_with_is_variant_classified_blank() -> None:
    # Some publishers add "is": "This page is intentionally left blank"
    assert _is_intentionally_blank_text("This page is intentionally left blank") is True


def test_trailing_whitespace_classified_blank() -> None:
    assert _is_intentionally_blank_text("  This page intentionally left blank  \n\n") is True


def test_real_content_with_blank_mention_NOT_classified_blank() -> None:
    # Negative-control: a preface that DISCUSSES blank pages must not
    # be classified as blank. Length > 120 chars trips the structural
    # guard.
    real_preface = (
        "In this edition, we have intentionally placed several blank pages "
        "between chapters to allow notes. The reader will find the phrase "
        "\"This page intentionally left blank\" on those leaves to indicate "
        "that no content has been omitted."
    )
    assert _is_intentionally_blank_text(real_preface) is False


def test_empty_text_NOT_classified_blank() -> None:
    # Truly empty pages are handled by the other branch of
    # _read_blank_pages_in_source; this detector only fires on
    # boilerplate text.
    assert _is_intentionally_blank_text("") is False
    assert _is_intentionally_blank_text("   \n  \t") is False


def test_unrelated_short_text_NOT_classified_blank() -> None:
    # A short chapter heading must not be classified as blank.
    assert _is_intentionally_blank_text("Chapter 5 — Greenhouse Control") is False
    assert _is_intentionally_blank_text("Vision and Image Generation 16") is False


def test_boilerplate_with_extra_short_garbage_still_classified_blank() -> None:
    # Publishers sometimes add a page-number marker. As long as the
    # total length stays under the structural cap, classify as blank.
    assert _is_intentionally_blank_text(
        "iii\nThis page intentionally left blank"
    ) is True


def test_long_text_with_boilerplate_NOT_classified_blank() -> None:
    # Boundary: a page with the boilerplate but also a paragraph of
    # real content exceeds the 120-char cap and is NOT blank.
    text = (
        "This page intentionally left blank. "
        "However, please note that the appendix on page 247 contains "
        "supplementary material relevant to chapters 3 and 4."
    )
    assert _is_intentionally_blank_text(text) is False
