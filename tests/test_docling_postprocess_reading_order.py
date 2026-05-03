"""Phase 1 tests for the post-Docling reading-order y-sort.

The body-text reading order on native-digital pages must follow the PDF's
y-coordinates (top-to-bottom = decreasing `t` in BOTTOMLEFT origin), with
ties broken by ascending `l`. Items lacking a prov bbox keep their original
Docling order, and pages remain ascending so a page-14 item never reorders
ahead of a page-13 item.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import List

import pytest

from mmrag_v2.engines.docling_postprocess import apply_reading_order_sort
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
        self._items = items
        self.body = _FakeBody([_FakeRef(it) for it in items])


def _bbox(t: float, l: float, origin: str = "BOTTOMLEFT") -> SimpleNamespace:
    return SimpleNamespace(
        l=l,
        t=t,
        r=l + 100.0,
        b=t - 30.0,
        coord_origin=SimpleNamespace(name=origin),
    )


def _item(text: str, page: int, t: float, l: float = 50.0, origin: str = "BOTTOMLEFT"):
    prov = [SimpleNamespace(page_no=page, bbox=_bbox(t, l, origin))]
    return SimpleNamespace(text=text, prov=prov)


def _no_bbox_item(text: str, page: int):
    prov = [SimpleNamespace(page_no=page, bbox=None)]
    return SimpleNamespace(text=text, prov=prov)


def _orphan_item(text: str):
    return SimpleNamespace(text=text, prov=[])


def _texts(doc: _FakeDoc) -> List[str]:
    return [ref.resolve(doc).text for ref in doc.body.children]


def _y_sort_plan() -> PdfConversionPlan:
    return PdfConversionPlan(reading_order_strategy="y_sort")


def test_y_sort_orders_three_paragraphs_top_to_bottom():
    """Mirrors HARRY page 13: doc order [354, 123, 255] -> y-sorted [354, 255, 123]."""
    doc = _FakeDoc(
        [
            _item("para_top", page=13, t=354.0),
            _item("para_bot", page=13, t=123.0),
            _item("para_mid", page=13, t=255.0),
        ]
    )
    apply_reading_order_sort(doc, _y_sort_plan())
    assert _texts(doc) == ["para_top", "para_mid", "para_bot"]


def test_y_sort_topleft_origin_inverts_sign():
    """TOPLEFT origin inverts: smaller `t` is higher on the page."""
    doc = _FakeDoc(
        [
            _item("para_mid", page=1, t=255.0, origin="TOPLEFT"),
            _item("para_top", page=1, t=100.0, origin="TOPLEFT"),
            _item("para_bot", page=1, t=400.0, origin="TOPLEFT"),
        ]
    )
    apply_reading_order_sort(doc, _y_sort_plan())
    assert _texts(doc) == ["para_top", "para_mid", "para_bot"]


def test_y_sort_ties_broken_by_ascending_left():
    """Same `t` -> sorted by ascending `l` (column reading order)."""
    doc = _FakeDoc(
        [
            _item("right", page=1, t=300.0, l=400.0),
            _item("left", page=1, t=300.0, l=50.0),
        ]
    )
    apply_reading_order_sort(doc, _y_sort_plan())
    assert _texts(doc) == ["left", "right"]


def test_y_sort_skips_items_without_bbox():
    """Items missing a prov bbox keep their original document order."""
    doc = _FakeDoc(
        [
            _item("with_bbox_low", page=1, t=200.0),
            _no_bbox_item("no_bbox_a", page=1),
            _no_bbox_item("no_bbox_b", page=1),
            _item("with_bbox_high", page=1, t=500.0),
        ]
    )
    apply_reading_order_sort(doc, _y_sort_plan())
    # Bbox items go first in y-sort order; no-bbox items follow in their own order.
    assert _texts(doc) == [
        "with_bbox_high",
        "with_bbox_low",
        "no_bbox_a",
        "no_bbox_b",
    ]


def test_y_sort_does_not_break_multipage_order():
    """Items on page 14 never reorder ahead of page 13 items."""
    doc = _FakeDoc(
        [
            _item("p14_top", page=14, t=600.0),
            _item("p13_bot", page=13, t=100.0),
            _item("p13_top", page=13, t=600.0),
            _item("p14_bot", page=14, t=100.0),
        ]
    )
    apply_reading_order_sort(doc, _y_sort_plan())
    texts = _texts(doc)
    assert texts == ["p13_top", "p13_bot", "p14_top", "p14_bot"]


def test_y_sort_disabled_when_strategy_is_docling_native():
    """Default strategy is docling_native: post-pass is a no-op."""
    doc = _FakeDoc(
        [
            _item("first_in_docling_order", page=1, t=100.0),
            _item("second_in_docling_order", page=1, t=600.0),
        ]
    )
    apply_reading_order_sort(doc, PdfConversionPlan())
    assert _texts(doc) == ["first_in_docling_order", "second_in_docling_order"]


def test_y_sort_with_dropcap_strategy_also_applies_y_sort():
    """`y_sort_with_dropcap` enables the y-sort stage too."""
    doc = _FakeDoc(
        [
            _item("bot", page=1, t=100.0),
            _item("top", page=1, t=500.0),
        ]
    )
    apply_reading_order_sort(
        doc, PdfConversionPlan(reading_order_strategy="y_sort_with_dropcap")
    )
    assert _texts(doc) == ["top", "bot"]


def test_y_sort_handles_orphan_item_without_provenance():
    """Items with empty prov stay at the end (no page bucket)."""
    doc = _FakeDoc(
        [
            _orphan_item("orphan_first"),
            _item("page_1_top", page=1, t=600.0),
            _item("page_1_bot", page=1, t=100.0),
            _orphan_item("orphan_last"),
        ]
    )
    apply_reading_order_sort(doc, _y_sort_plan())
    texts = _texts(doc)
    assert texts == ["page_1_top", "page_1_bot", "orphan_first", "orphan_last"]


def test_y_sort_returns_same_doc_for_chaining():
    """Apply returns the input doc reference for fluent composition."""
    doc = _FakeDoc([_item("a", page=1, t=100.0)])
    result = apply_reading_order_sort(doc, _y_sort_plan())
    assert result is doc


def test_invalid_reading_order_strategy_raises_value_error():
    with pytest.raises(ValueError):
        PdfConversionPlan(reading_order_strategy="bogus")  # type: ignore[arg-type]


def test_plan_default_is_docling_native():
    assert PdfConversionPlan().reading_order_strategy == "docling_native"


def test_adapter_convert_invokes_postprocessors(monkeypatch):
    """Bridge: DoclingPdfAdapter.convert runs the postprocess pipeline on the doc."""
    from mmrag_v2.engines import docling_adapter as adapter_module
    from mmrag_v2.engines.docling_adapter import DoclingPdfAdapter

    captured = {}

    def fake_apply(doc, plan):
        captured["doc"] = doc
        captured["plan"] = plan
        return doc

    monkeypatch.setattr(adapter_module, "apply_postprocessors", fake_apply)

    fake_doc = SimpleNamespace(body=_FakeBody([]))
    fake_result = SimpleNamespace(document=fake_doc)
    fake_converter = SimpleNamespace(convert=lambda _path: fake_result)

    monkeypatch.setattr(
        DoclingPdfAdapter, "get_converter", lambda self: fake_converter
    )

    plan = PdfConversionPlan(reading_order_strategy="y_sort")
    adapter = DoclingPdfAdapter(plan)
    result = adapter.convert("dummy.pdf")

    assert result is fake_result
    assert captured["doc"] is fake_doc
    assert captured["plan"] is plan
