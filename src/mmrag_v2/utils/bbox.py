"""v2.16 Phase 4 — bounding-box geometry helpers.

Currently exposes `bbox_iou` only (Intersection-over-Union on normalized
integer `[0, 1000]` bboxes per AGENT-SPATIAL-20 invariant). Used by the
VLM-table dedup pass in `processor.py` to suppress flat-prose chunks that
spatially overlap a VLM-extracted table on the same page.
"""
from __future__ import annotations

from typing import Sequence

BBox = Sequence[int]  # [x0, y0, x1, y1] in [0, 1000] integer space


def bbox_iou(a: BBox, b: BBox) -> float:
    """Return Intersection-over-Union for two bboxes.

    Bbox convention: `[x0, y0, x1, y1]` integer coordinates in `[0, 1000]`
    space (AGENT-SPATIAL-20 invariant). The y-axis orientation is irrelevant
    because IoU is reflection-symmetric.

    Returns:
      - `0.0` for degenerate / empty / invalid inputs (None, wrong length,
        zero-area boxes, no overlap).
      - A float in `[0.0, 1.0]` for valid overlapping inputs.
    """
    if a is None or b is None:
        return 0.0
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    if ax1 <= ax0 or ay1 <= ay0 or bx1 <= bx0 or by1 <= by0:
        return 0.0

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0

    intersection = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union
