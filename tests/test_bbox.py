"""v2.16 Phase 4 — bbox_iou unit tests.

5 cases per PLAN_V2.16.md §3 Phase 4 step 1: identical, disjoint, 50%
overlap, contained, degenerate (zero-area).
"""
from mmrag_v2.utils.bbox import bbox_iou


def test_identical_bboxes_yield_iou_1():
    assert bbox_iou([100, 100, 200, 200], [100, 100, 200, 200]) == 1.0


def test_disjoint_bboxes_yield_iou_0():
    # Horizontally separated boxes — no overlap.
    assert bbox_iou([0, 0, 100, 100], [200, 200, 300, 300]) == 0.0


def test_half_overlap_horizontal_yields_iou_one_third():
    # Two 100x100 boxes overlapping by 50% on the x axis:
    #   A: x ∈ [0, 100], B: x ∈ [50, 150], both y ∈ [0, 100].
    # intersection = 50 * 100 = 5000
    # union        = 100*100 + 100*100 - 5000 = 15000
    # IoU          = 5000 / 15000 ≈ 0.333
    assert 0.32 < bbox_iou([0, 0, 100, 100], [50, 0, 150, 100]) < 0.34


def test_contained_bbox_yields_iou_area_ratio():
    # Inner 100x100 fully inside outer 200x200:
    # intersection = 100*100 = 10000
    # union        = 200*200 = 40000 (outer area; inner is subset)
    # IoU          = 10000 / 40000 = 0.25
    assert bbox_iou([0, 0, 200, 200], [50, 50, 150, 150]) == 0.25


def test_degenerate_zero_area_yields_iou_0():
    # Zero-width bbox (x0 == x1).
    assert bbox_iou([100, 100, 100, 200], [100, 100, 200, 200]) == 0.0
    # Zero-height bbox.
    assert bbox_iou([100, 100, 200, 100], [100, 100, 200, 200]) == 0.0
    # Both inputs degenerate.
    assert bbox_iou([100, 100, 100, 100], [100, 100, 100, 100]) == 0.0


def test_invalid_inputs_yield_iou_0():
    # None / wrong length / inverted coordinates — all return 0.0 (defensive).
    assert bbox_iou(None, [0, 0, 100, 100]) == 0.0
    assert bbox_iou([0, 0, 100, 100], None) == 0.0
    assert bbox_iou([0, 0, 100], [0, 0, 100, 100]) == 0.0
    # Inverted (x1 < x0) — treated as degenerate.
    assert bbox_iou([200, 0, 100, 100], [0, 0, 100, 100]) == 0.0


def test_high_overlap_below_dedup_threshold():
    """Edge case for the v2.16 Phase 4 dedup at IoU=0.85: bboxes overlapping
    by ~0.86 must not collapse to <0.85 due to float math."""
    # Construct boxes whose IoU is ~0.86: 100x100 boxes with 95% overlap area.
    # A: [0, 0, 100, 100] area 10000
    # B: [5, 0, 100, 100] area 9500
    # intersection: 95 * 100 = 9500; union: 10000 + 9500 - 9500 = 10000
    # IoU = 9500/10000 = 0.95
    iou = bbox_iou([0, 0, 100, 100], [5, 0, 100, 100])
    assert 0.94 < iou < 0.96
