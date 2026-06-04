"""V3 within-page duplicate-long-text dedup (crucible fix #2, 2026-06-04).

Dense pages can make the VLM loop and re-emit the same paragraph/heading
several times (Combat Aircraft: a heading 10x on one page), tripping QA's
DUPLICATE_LONG_TEXT / within_page_text_dupe_excess gate. The chunker drops
later within-page exact (whitespace-normalized) duplicate TEXT chunks above the
gate's 120-char threshold, keeping the first. IMAGE/TABLE/CODE/FORM, short
text, and cross-page repeats (headers/footers) are untouched.

Deterministic/offline: synthetic UIR chunks, no VLM.
"""

from __future__ import annotations

from mmrag_v2.chunking.uir_chunker import _DEDUP_MIN_CHARS, _dedupe_within_page_text
from mmrag_v2.schema.ingestion_schema import Modality
from mmrag_v2.universal.intermediate import (
    ConfidenceBreakdown,
    CoordinateFrame,
    Locator,
    LocatorType,
    UIRChunk,
)

LONG = "The Xian Aircraft Corporation's next KJ-600 naval AEW aircraft entered service " + (
    "with the PLAN. " * 6
)  # > 120 chars
assert len(LONG) >= _DEDUP_MIN_CHARS


def _txt(content: str, page: int, modality: Modality = Modality.TEXT) -> UIRChunk:
    return UIRChunk(
        modality=modality,
        content=content,
        locator=Locator(
            type=LocatorType.BBOX,
            bbox=[10, 10, 990, 50],
            page_number=page,
            coordinate_frame=CoordinateFrame.PDF_PAGE_PORTRAIT,
        ),
        confidence=ConfidenceBreakdown(),
        extraction_method="vlm_native",
        extraction_engine_version="qwen3-vl-8b",
    )


def _contents(chunks):
    return [c.content for c in chunks]


def test_exact_long_duplicates_on_page_are_collapsed():
    chunks = [_txt(LONG, 1), _txt(LONG, 1), _txt(LONG, 1), _txt("distinct tail.", 1)]
    out = _dedupe_within_page_text(chunks)
    assert _contents(out) == [LONG, "distinct tail."]  # first kept, 2 dropped


def test_whitespace_variant_duplicates_collapse():
    spaced = LONG.replace(" ", "  ").replace("service", "service\n")
    out = _dedupe_within_page_text([_txt(LONG, 1), _txt(spaced, 1)])
    assert len(out) == 1  # normalized-equal -> deduped


def test_short_duplicates_are_kept():
    short = "KJ-600 AEW"  # < 120 chars
    out = _dedupe_within_page_text([_txt(short, 1), _txt(short, 1)])
    assert len(out) == 2


def test_cross_page_duplicates_are_kept():
    out = _dedupe_within_page_text([_txt(LONG, 1), _txt(LONG, 2)])
    assert len(out) == 2  # header/footer repeated across pages is legitimate


def test_image_and_table_duplicates_are_never_deduped():
    img = _txt(LONG, 1, modality=Modality.IMAGE)
    tbl = _txt(LONG, 1, modality=Modality.TABLE)
    out = _dedupe_within_page_text([img, _txt(LONG, 1), tbl, _txt(LONG, 1)])
    # both visual chunks survive; the 2nd TEXT copy is dropped (1 TEXT kept).
    assert sum(1 for c in out if c.modality is Modality.IMAGE) == 1
    assert sum(1 for c in out if c.modality is Modality.TABLE) == 1
    assert sum(1 for c in out if c.modality is Modality.TEXT) == 1


def test_order_is_preserved():
    a, b = LONG, LONG.replace("Xian", "Shenyang")  # two distinct long texts
    out = _dedupe_within_page_text([_txt(a, 1), _txt(b, 1), _txt(a, 1)])
    assert _contents(out) == [a, b]
