"""
Coordinate Normalization - Spatial Precision Utilities for MM-Converter-V2
===========================================================================
ENGINE_USE: Docling v2.66.0 (Native Layout Analysis)

This module provides utilities for normalizing bounding box coordinates
from various coordinate systems to the canonical **0-1000 INTEGER scale**.

REQ Compliance:
- REQ-COORD-01: All bounding boxes MUST be normalized to 0-1000 integer scale
- REQ-COORD-02: Strict validation prevents coordinate overflow
- REQ-COORD-03: Coordinates represent permille of page dimensions

SRS Section 6.2: Spatial Data
"All spatial coordinates MUST be normalized to a 0-1000 integer scale
representing permille of page dimensions."

Author: Claude 4.5 Opus (Architect)
Date: 2025-12-28
Updated: 2025-12-30 (Integer 0-1000 Scale)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Standard PDF page dimensions (Letter size in points)
DEFAULT_PAGE_WIDTH: float = 612.0
DEFAULT_PAGE_HEIGHT: float = 792.0

# Normalized coordinate scale (0-1000 integers)
SCALE_FACTOR: int = 1000
MIN_COORD: int = 0
MAX_COORD: int = 1000


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================


def validate_bbox_strict(
    bbox: List[int],
    context: str = "unknown",
) -> bool:
    """
    Strictly validate that a bounding box is properly normalized (0-1000 int).

    REQ-COORD-02: This validation is STRICT - no tolerance for errors.

    Args:
        bbox: Bounding box as [l, t, r, b] integers (0-1000)
        context: Context string for error messages

    Returns:
        True if valid, raises ValueError if not

    Raises:
        ValueError: If bbox is not properly normalized
    """
    if bbox is None:
        raise ValueError(f"[{context}] bbox is None")

    if len(bbox) != 4:
        raise ValueError(f"[{context}] bbox must have 4 values, got {len(bbox)}")

    l, t, r, b = bbox

    # Check all values are integers
    for i, (name, val) in enumerate([("l", l), ("t", t), ("r", r), ("b", b)]):
        if not isinstance(val, int):
            raise ValueError(
                f"[{context}] bbox[{i}] ({name}={val}) must be int, got {type(val).__name__}"
            )
        if not (MIN_COORD <= val <= MAX_COORD):
            raise ValueError(
                f"[{context}] bbox[{i}] ({name}={val}) out of normalized range "
                f"[{MIN_COORD}, {MAX_COORD}]"
            )

    # Check valid dimensions
    if r <= l:
        raise ValueError(f"[{context}] Invalid bbox: right ({r}) must be > left ({l})")
    if b <= t:
        raise ValueError(f"[{context}] Invalid bbox: bottom ({b}) must be > top ({t})")

    return True


def is_normalized(bbox: List[int]) -> bool:
    """
    Check if a bounding box is already normalized (all values in 0-1000 int range).

    Args:
        bbox: Bounding box as [l, t, r, b]

    Returns:
        True if all coordinates are integers in 0-1000 range
    """
    if bbox is None or len(bbox) != 4:
        return False

    return all(isinstance(coord, int) and MIN_COORD <= coord <= MAX_COORD for coord in bbox)


def _is_float_normalized(bbox: Sequence[Union[int, float]]) -> bool:
    """
    Check if bbox appears to be in 0.0-1.0 float format.

    Helper to detect incoming float-normalized coordinates.
    """
    if bbox is None or len(bbox) != 4:
        return False
    return all(isinstance(c, (int, float)) and 0.0 <= float(c) <= 1.0 for c in bbox)


# ============================================================================
# NORMALIZATION FUNCTIONS
# ============================================================================


def normalize_bbox(
    bbox: Sequence[Union[int, float]],
    page_width: float = DEFAULT_PAGE_WIDTH,
    page_height: float = DEFAULT_PAGE_HEIGHT,
    context: str = "unknown",
) -> List[int]:
    """
    Normalize a bounding box from absolute coordinates to 0-1000 integer scale.

    REQ-COORD-01: Converts absolute pixel/point coordinates to normalized
    integer values on a 0-1000 scale.

    Args:
        bbox: Bounding box as [x0, y0, x1, y1] in absolute coordinates
        page_width: Page width in the same units as bbox
        page_height: Page height in the same units as bbox
        context: Context string for logging/errors

    Returns:
        Normalized bbox as [l, t, r, b] integers in 0-1000 range

    Raises:
        ValueError: If normalization results in invalid coordinates
    """
    if bbox is None:
        raise ValueError(f"[{context}] Cannot normalize None bbox")

    if len(bbox) != 4:
        raise ValueError(f"[{context}] bbox must have 4 values, got {len(bbox)}")

    x0, y0, x1, y1 = bbox

    # Ensure width/height are positive
    if page_width <= 0:
        logger.warning(f"[{context}] Invalid page_width={page_width}, using default")
        page_width = DEFAULT_PAGE_WIDTH
    if page_height <= 0:
        logger.warning(f"[{context}] Invalid page_height={page_height}, using default")
        page_height = DEFAULT_PAGE_HEIGHT

    # Normalize to 0-1000 integer range
    l = int(round((x0 / page_width) * SCALE_FACTOR))
    t = int(round((y0 / page_height) * SCALE_FACTOR))
    r = int(round((x1 / page_width) * SCALE_FACTOR))
    b = int(round((y1 / page_height) * SCALE_FACTOR))

    # Clamp to valid range
    l = max(MIN_COORD, min(MAX_COORD, l))
    t = max(MIN_COORD, min(MAX_COORD, t))
    r = max(MIN_COORD, min(MAX_COORD, r))
    b = max(MIN_COORD, min(MAX_COORD, b))

    # Ensure proper ordering
    if r < l:
        l, r = r, l
    if b < t:
        t, b = b, t

    # Ensure minimum dimensions (at least 1 unit)
    if r <= l:
        r = l + 1
    if b <= t:
        b = t + 1

    # Final clamp after adjustments
    r = min(MAX_COORD, r)
    b = min(MAX_COORD, b)

    normalized = [l, t, r, b]

    logger.debug(
        f"[{context}] Normalized bbox: "
        f"[{x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}] → "
        f"[{l}, {t}, {r}, {b}]"
    )

    return normalized


def ensure_normalized(
    bbox: Sequence[Union[int, float]],
    page_width: float = DEFAULT_PAGE_WIDTH,
    page_height: float = DEFAULT_PAGE_HEIGHT,
    context: str = "unknown",
) -> List[int]:
    """
    Ensure a bounding box is normalized to 0-1000 integer scale.

    This function handles multiple input formats:
    1. Already normalized integers (0-1000) - validates and returns
    2. Float normalized (0.0-1.0) - converts to 0-1000 integers
    3. Absolute coordinates - normalizes using page dimensions

    REQ-COORD-01: Guarantees output is integers in 0-1000 range.

    Args:
        bbox: Bounding box (may be normalized or absolute)
        page_width: Page width for normalization
        page_height: Page height for normalization
        context: Context string for logging

    Returns:
        Normalized bbox as [l, t, r, b] integers in 0-1000 range

    Raises:
        ValueError: If bbox cannot be normalized to valid range
    """
    if bbox is None:
        raise ValueError(f"[{context}] Cannot ensure_normalized on None bbox")

    if len(bbox) != 4:
        raise ValueError(f"[{context}] bbox must have 4 values, got {len(bbox)}")

    # Case 1: Already normalized integers (0-1000)
    # Check if all values are integers and in range
    if all(isinstance(c, int) and MIN_COORD <= c <= MAX_COORD for c in bbox):
        l, t, r, b = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        # Validate ordering
        if r <= l or b <= t:
            if r < l:
                l, r = r, l
            if b < t:
                t, b = b, t
        result: List[int] = [l, t, r, b]
        logger.debug(f"[{context}] bbox already normalized (int): {result}")
        return result

    # Case 2: Float normalized (0.0-1.0) - convert to integer scale
    if _is_float_normalized(bbox):
        l = int(round(bbox[0] * SCALE_FACTOR))
        t = int(round(bbox[1] * SCALE_FACTOR))
        r = int(round(bbox[2] * SCALE_FACTOR))
        b = int(round(bbox[3] * SCALE_FACTOR))

        # Ensure proper ordering
        if r < l:
            l, r = r, l
        if b < t:
            t, b = b, t

        # Ensure minimum dimensions
        if r <= l:
            r = l + 1
        if b <= t:
            b = t + 1

        normalized = [l, t, r, b]
        logger.debug(f"[{context}] Converted float→int: {bbox} → {normalized}")
        return normalized

    # Case 3: Absolute coordinates - full normalization
    return normalize_bbox(bbox, page_width, page_height, context)


def denormalize_bbox(
    bbox: List[int],
    page_width: float = DEFAULT_PAGE_WIDTH,
    page_height: float = DEFAULT_PAGE_HEIGHT,
) -> List[float]:
    """
    Convert normalized bbox (0-1000 int) back to absolute coordinates.

    Useful for cropping operations that need pixel coordinates.

    Args:
        bbox: Normalized bbox as [l, t, r, b] integers (0-1000)
        page_width: Target page width
        page_height: Target page height

    Returns:
        Absolute bbox as [x0, y0, x1, y1] floats
    """
    if bbox is None or len(bbox) != 4:
        raise ValueError("Invalid normalized bbox")

    l, t, r, b = bbox

    return [
        (l / SCALE_FACTOR) * page_width,
        (t / SCALE_FACTOR) * page_height,
        (r / SCALE_FACTOR) * page_width,
        (b / SCALE_FACTOR) * page_height,
    ]


def scale_bbox(
    bbox: List[int],
    scale_factor: float,
) -> List[int]:
    """
    Scale a normalized bbox by a factor (for high-DPI operations).

    Note: This operates on the integer scale and returns integers.

    Args:
        bbox: Normalized bbox as [l, t, r, b] integers (0-1000)
        scale_factor: Scale factor to apply

    Returns:
        Scaled bbox (still in 0-1000 range, clamped)
    """
    if bbox is None or len(bbox) != 4:
        raise ValueError("Invalid bbox for scaling")

    result = [max(MIN_COORD, min(MAX_COORD, int(round(coord * scale_factor)))) for coord in bbox]

    return result


def bbox_iou(
    bbox1: Sequence[Union[int, float]],
    bbox2: Sequence[Union[int, float]],
) -> float:
    """
    Calculate Intersection over Union (IoU) between two bboxes.

    Args:
        bbox1: First bbox as [l, t, r, b] (normalized 0-1000 or 0.0-1.0)
        bbox2: Second bbox as [l, t, r, b] (normalized 0-1000 or 0.0-1.0)

    Returns:
        IoU value between 0.0 and 1.0 (float for precision)
    """
    if bbox1 is None or bbox2 is None:
        return 0.0
    if len(bbox1) != 4 or len(bbox2) != 4:
        return 0.0

    # Convert to float for consistent comparison (handles both int and float inputs)
    l1, t1, r1, b1 = float(bbox1[0]), float(bbox1[1]), float(bbox1[2]), float(bbox1[3])
    l2, t2, r2, b2 = float(bbox2[0]), float(bbox2[1]), float(bbox2[2]), float(bbox2[3])

    # Intersection
    inter_l = max(l1, l2)
    inter_t = max(t1, t2)
    inter_r = min(r1, r2)
    inter_b = min(b1, b2)

    if inter_r <= inter_l or inter_b <= inter_t:
        return 0.0

    inter_area = (inter_r - inter_l) * (inter_b - inter_t)

    # Union
    area1 = (r1 - l1) * (b1 - t1)
    area2 = (r2 - l2) * (b2 - t2)
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return float(inter_area) / float(union_area)


def bbox_overlap_ratio(
    bbox1: List[int],
    bbox2: List[int],
) -> Tuple[float, float]:
    """
    Calculate how much of each bbox overlaps with the other.

    Args:
        bbox1: First bbox as [l, t, r, b] (0-1000 integers)
        bbox2: Second bbox as [l, t, r, b] (0-1000 integers)

    Returns:
        Tuple of (overlap_of_bbox1, overlap_of_bbox2) as floats
        Each value is the percentage of that bbox covered by intersection
    """
    if bbox1 is None or bbox2 is None:
        return 0.0, 0.0
    if len(bbox1) != 4 or len(bbox2) != 4:
        return 0.0, 0.0

    l1, t1, r1, b1 = bbox1
    l2, t2, r2, b2 = bbox2

    # Intersection
    inter_l = max(l1, l2)
    inter_t = max(t1, t2)
    inter_r = min(r1, r2)
    inter_b = min(b1, b2)

    if inter_r <= inter_l or inter_b <= inter_t:
        return 0.0, 0.0

    inter_area = (inter_r - inter_l) * (inter_b - inter_t)

    area1 = (r1 - l1) * (b1 - t1)
    area2 = (r2 - l2) * (b2 - t2)

    overlap1 = float(inter_area) / float(area1) if area1 > 0 else 0.0
    overlap2 = float(inter_area) / float(area2) if area2 > 0 else 0.0

    return overlap1, overlap2


def to_float_normalized(bbox: List[int]) -> List[float]:
    """
    Convert 0-1000 integer bbox to 0.0-1.0 float format.

    Useful for compatibility with systems expecting float coordinates.

    Args:
        bbox: Normalized bbox as [l, t, r, b] integers (0-1000)

    Returns:
        Float bbox as [l, t, r, b] floats (0.0-1.0)
    """
    if bbox is None or len(bbox) != 4:
        raise ValueError("Invalid bbox for conversion")

    return [coord / SCALE_FACTOR for coord in bbox]


def from_float_normalized(bbox: List[float]) -> List[int]:
    """
    Convert 0.0-1.0 float bbox to 0-1000 integer format.

    Args:
        bbox: Float bbox as [l, t, r, b] floats (0.0-1.0)

    Returns:
        Integer bbox as [l, t, r, b] integers (0-1000)
    """
    if bbox is None or len(bbox) != 4:
        raise ValueError("Invalid bbox for conversion")

    return [int(round(coord * SCALE_FACTOR)) for coord in bbox]
