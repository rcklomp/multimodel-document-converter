"""Phase 2 tests for the post-Docling drop-cap promoter.

Literary drop caps (large display-font glyph at the start of a chapter
paragraph) are emitted by Docling as a separate TextItem because the
glyph uses a different font from the body. The reading-order pass
inevitably places the standalone glyph adjacent to its target paragraph
but without merging them; the chunk text reads `"r. and Mrs. Dursley...
nonsense. M"` instead of `"Mr. and Mrs. Dursley... nonsense."`.

The promoter runs ahead of the y-sort so the final sort sees the merged
item as a single y-anchored paragraph.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import List

import pytest

from mmrag_v2.engines.docling_postprocess import (
    _heal_inline_trailing_dropcap,
    apply_dropcap_promotion,
    apply_postprocessors,
)
from mmrag_v2.engines.pdf_plan import PdfConversionPlan


class _FakeRef:
    def __init__(self, item):
        self._item = item

    def resolve(self, _doc):
        return self._item


class _FakeBody:
    def __init__(self, refs):
        self.children = list(refs)


class _FakeDoc:
    def __init__(self, items):
        self.body = _FakeBody([_FakeRef(it) for it in items])


def _bbox(t: float, b: float, l: float = 50.0, origin: str = "TOPLEFT") -> SimpleNamespace:
    return SimpleNamespace(
        l=l,
        t=t,
        r=l + 100.0,
        b=b,
        coord_origin=SimpleNamespace(name=origin),
    )


def _text_item(
    text: str,
    page: int,
    t: float,
    b: float,
    l: float = 50.0,
    origin: str = "TOPLEFT",
):
    prov = [SimpleNamespace(page_no=page, bbox=_bbox(t, b, l, origin))]
    return SimpleNamespace(text=text, orig=text, prov=prov)


def _texts(doc: _FakeDoc) -> List[str]:
    return [ref.resolve(doc).text for ref in doc.body.children]


def _dropcap_plan() -> PdfConversionPlan:
    return PdfConversionPlan(reading_order_strategy="y_sort_with_dropcap")


def test_dropcap_M_prepended_to_paragraph():
    """HARRY page 13: standalone "M" + lowercase paragraph -> merged 'Mr...'."""
    cap = _text_item("M", page=13, t=281.8, b=367.2)
    para = _text_item(
        "r. and Mrs. Dursley, of number four, Privet Drive, were proud to say "
        "that they were perfectly normal, thank you very much.",
        page=13,
        t=266.2,
        b=576.6,
    )
    doc = _FakeDoc([cap, para])
    apply_dropcap_promotion(doc, _dropcap_plan())
    texts = _texts(doc)
    assert len(texts) == 1
    assert texts[0].startswith("Mr. and Mrs. Dursley")


def test_dropcap_handles_emit_order_paragraph_then_cap():
    """Docling can emit body before drop cap; promoter looks both directions."""
    para = _text_item("r. and Mrs. Dursley...", page=13, t=266.2, b=576.6)
    cap = _text_item("M", page=13, t=281.8, b=367.2)
    doc = _FakeDoc([para, cap])
    apply_dropcap_promotion(doc, _dropcap_plan())
    texts = _texts(doc)
    assert texts == ["Mr. and Mrs. Dursley..."]


def test_dropcap_not_promoted_when_next_paragraph_starts_uppercase():
    """A standalone "M" with an uppercase-starting neighbor is not a drop cap."""
    cap = _text_item("M", page=13, t=281.8, b=367.2)
    para = _text_item(
        "Mr. Dursley was the director of a firm called Grunnings",
        page=13,
        t=266.2,
        b=576.6,
    )
    doc = _FakeDoc([cap, para])
    apply_dropcap_promotion(doc, _dropcap_plan())
    texts = _texts(doc)
    assert texts == ["M", "Mr. Dursley was the director of a firm called Grunnings"]


def test_dropcap_not_promoted_without_y_overlap():
    """Drop cap whose y-band sits outside the neighbor's lines is not merged."""
    cap = _text_item("M", page=13, t=50.0, b=80.0)
    para = _text_item("r. and Mrs. Dursley...", page=13, t=200.0, b=400.0)
    doc = _FakeDoc([cap, para])
    apply_dropcap_promotion(doc, _dropcap_plan())
    texts = _texts(doc)
    assert texts == ["M", "r. and Mrs. Dursley..."]


def test_dropcap_not_promoted_across_page_boundaries():
    """Drop cap on page 12 must not absorb the lowercase opener of page 13."""
    cap = _text_item("M", page=12, t=281.8, b=367.2)
    para = _text_item("r. and Mrs. Dursley...", page=13, t=266.2, b=576.6)
    doc = _FakeDoc([cap, para])
    apply_dropcap_promotion(doc, _dropcap_plan())
    texts = _texts(doc)
    assert texts == ["M", "r. and Mrs. Dursley..."]


def test_dropcap_promoter_idempotent():
    """Running the promoter twice produces the same result."""
    cap = _text_item("M", page=13, t=281.8, b=367.2)
    para = _text_item("r. and Mrs. Dursley...", page=13, t=266.2, b=576.6)
    doc = _FakeDoc([cap, para])
    plan = _dropcap_plan()
    apply_dropcap_promotion(doc, plan)
    apply_dropcap_promotion(doc, plan)
    assert _texts(doc) == ["Mr. and Mrs. Dursley..."]


def test_dropcap_disabled_under_y_sort_only_strategy():
    """`y_sort` (without dropcap) leaves stray drop caps alone."""
    cap = _text_item("M", page=13, t=281.8, b=367.2)
    para = _text_item("r. and Mrs. Dursley...", page=13, t=266.2, b=576.6)
    doc = _FakeDoc([cap, para])
    apply_dropcap_promotion(doc, PdfConversionPlan(reading_order_strategy="y_sort"))
    assert _texts(doc) == ["M", "r. and Mrs. Dursley..."]


def test_dropcap_disabled_by_default():
    """Default plan (`docling_native`) leaves stray drop caps alone."""
    cap = _text_item("M", page=13, t=281.8, b=367.2)
    para = _text_item("r. and Mrs. Dursley...", page=13, t=266.2, b=576.6)
    doc = _FakeDoc([cap, para])
    apply_dropcap_promotion(doc, PdfConversionPlan())
    assert _texts(doc) == ["M", "r. and Mrs. Dursley..."]


def test_dropcap_skips_multi_character_text():
    """A two-character item is not a drop cap candidate."""
    cap = _text_item("Mr", page=13, t=281.8, b=367.2)
    para = _text_item("rs. Dursley...", page=13, t=266.2, b=576.6)
    doc = _FakeDoc([cap, para])
    apply_dropcap_promotion(doc, _dropcap_plan())
    assert _texts(doc) == ["Mr", "rs. Dursley..."]


def test_inline_trailing_dropcap_is_healed():
    """The actual HARRY page-13 pattern: drop cap glued to the END of the body.

    Docling 2.86 emits 'r. and Mrs. Dursley...nonsense. M' as a single
    TextItem with the drop-cap M trailing the paragraph. The heal moves
    the M to the front: 'Mr. and Mrs. Dursley...nonsense.'.
    """
    item = _text_item(
        "r. and Mrs. Dursley, of number four, were proud to say nonsense. M",
        page=13, t=266.2, b=576.6,
    )
    assert _heal_inline_trailing_dropcap(item) is True
    assert item.text.startswith("Mr. and Mrs. Dursley")
    assert not item.text.rstrip().endswith(" M")


def test_inline_trailing_dropcap_idempotent():
    item = _text_item(
        "r. and Mrs. Dursley were proud to say nonsense. M",
        page=13, t=266.2, b=576.6,
    )
    _heal_inline_trailing_dropcap(item)
    snapshot = item.text
    assert _heal_inline_trailing_dropcap(item) is False
    assert item.text == snapshot


def test_inline_trailing_dropcap_skips_paragraphs_starting_uppercase():
    item = _text_item(
        "Mr. Dursley was the director of a firm called Grunnings. A",
        page=13, t=266.2, b=576.6,
    )
    assert _heal_inline_trailing_dropcap(item) is False
    assert item.text == "Mr. Dursley was the director of a firm called Grunnings. A"


def test_inline_trailing_dropcap_skips_no_trailing_letter():
    item = _text_item(
        "r. and Mrs. Dursley, normal text without a trailing glyph.",
        page=13, t=266.2, b=576.6,
    )
    assert _heal_inline_trailing_dropcap(item) is False


def test_dropcap_pipeline_heals_inline_trailing_glyph_in_full_pipeline():
    """End-to-end: apply_postprocessors heals an inline trailing M."""
    item = _text_item(
        "r. and Mrs. Dursley, perfectly normal nonsense. M",
        page=13, t=266.2, b=576.6,
    )
    doc = _FakeDoc([item])
    apply_postprocessors(doc, _dropcap_plan())
    assert _texts(doc)[0].startswith("Mr. and Mrs. Dursley")


def test_dropcap_pipeline_runs_dropcap_before_ysort():
    """End-to-end: promoter runs first, then y-sort sees a single anchored item."""
    para_higher = _text_item("r. and Mrs. Dursley were proud", page=13, t=266.2, b=576.6)
    cap = _text_item("M", page=13, t=281.8, b=367.2)
    para_lower = _text_item(
        "Mr. Dursley was the director of a firm called Grunnings",
        page=13,
        t=600.0,
        b=900.0,
    )
    doc = _FakeDoc([para_higher, cap, para_lower])
    apply_postprocessors(doc, _dropcap_plan())
    texts = _texts(doc)
    # Promoter merges M into the lowercase paragraph; y-sort puts it first
    # because its t (266.2) is lower than the second paragraph's t (600.0).
    assert texts == [
        "Mr. and Mrs. Dursley were proud",
        "Mr. Dursley was the director of a firm called Grunnings",
    ]
