"""PLAN_QUALITY_REMEDIATION_V1 WP-F1: Docling boxes are BOTTOMLEFT-origin and must be converted.

Defect: ``_normalize_bbox`` read l/t/r/b raw, ignoring ``coord_origin``. Docling's PDF provenance is
BOTTOMLEFT (y grows upward), so every box came out mirrored vertically: on IRJET page 2 the running
header (true y 55-91 of 1000) was emitted at 907-949 and the left column at 394-879 instead of 120-605.
Downstream that puts the running header in the BOTTOM furniture band and defeats spatial rules.
"""

from types import SimpleNamespace

from docling_core.types.doc import BoundingBox, CoordOrigin

from mmrag_v3.engines.docling_fast import _normalize_bbox

PAGE_W, PAGE_H = 595.0, 842.0


def test_bottomleft_box_is_flipped_to_top_left():
    # a header near the TOP of an A4 page: t=750 / b=714 in bottom-left coordinates
    bb = BoundingBox(l=57, t=750, r=112, b=714, coord_origin=CoordOrigin.BOTTOMLEFT)
    x0, y0, x1, y1 = _normalize_bbox(bb, PAGE_W, PAGE_H)
    assert (y0, y1) == (109, 152)  # the header sits at the top, not at 847-890
    assert (x0, x1) == (95, 188)


def test_topleft_box_is_left_unchanged():
    bb = BoundingBox(l=57, t=92, r=112, b=128, coord_origin=CoordOrigin.TOPLEFT)
    assert _normalize_bbox(bb, PAGE_W, PAGE_H) == [95, 109, 188, 152]


def test_duck_typed_box_without_an_origin_is_used_as_is():
    bb = SimpleNamespace(l=57, t=92, r=112, b=128)
    assert _normalize_bbox(bb, PAGE_W, PAGE_H) == [95, 109, 188, 152]


def test_flipped_and_unflipped_agree_for_the_same_physical_box():
    physical_top = BoundingBox(l=100, t=92, r=300, b=128, coord_origin=CoordOrigin.TOPLEFT)
    physical_bottomleft = BoundingBox(
        l=100, t=750, r=300, b=714, coord_origin=CoordOrigin.BOTTOMLEFT
    )
    assert _normalize_bbox(physical_top, PAGE_W, PAGE_H) == _normalize_bbox(
        physical_bottomleft, PAGE_W, PAGE_H
    )


def test_a_box_that_cannot_be_converted_falls_back_instead_of_being_dropped():
    class Odd:
        coord_origin = CoordOrigin.BOTTOMLEFT
        l = 57.0  # noqa: E741 - Docling's own attribute names
        t, r, b = 92.0, 112.0, 128.0

        def to_top_left_origin(self, page_height):
            raise ValueError("no page height")

    assert _normalize_bbox(Odd(), PAGE_W, PAGE_H) is not None
