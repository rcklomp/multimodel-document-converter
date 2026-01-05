"""
Advanced Spatial Propagator - Universal Bbox Normalization for ALL Modalities
==============================================================================
ENGINE_USE: Docling v2.66.0 (Native Layout Analysis)

PROBLEM SOLVED:
95% of chunks had `spatial: null` because only image/table modalities
propagated bbox data. This module ensures ALL elements (text, image, table)
receive normalized 0-1000 spatial coordinates from Docling provenance.

REQ Compliance:
- REQ-COORD-01: ALL bounding boxes normalized to 0-1000 integer scale
- REQ-SPATIAL-01: Text chunks MUST inherit spatial from Docling provenance
- REQ-SPATIAL-02: No chunk shall have spatial:null if Docling provides bbox

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│ Docling Provenance                                               │
│   └─ prov.bbox (raw PDF coords)                                  │
│        ↓                                                         │
│   SpatialPropagator.extract_and_normalize()                      │
│        ↓                                                         │
│   [l, t, r, b] integers 0-1000                                   │
│        ↓                                                         │
│   ALL modality chunks (text, image, table, shadow)               │
└─────────────────────────────────────────────────────────────────┘

Author: Claude (Senior Architect)
Date: 2026-01-02
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .coordinate_normalization import ensure_normalized, normalize_bbox

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Default PDF page dimensions (Letter size in points)
DEFAULT_PAGE_WIDTH: float = 612.0
DEFAULT_PAGE_HEIGHT: float = 792.0

# Minimum bbox area to be considered valid (avoid point/line bboxes)
MIN_BBOX_AREA_NORMALIZED: int = 100  # 0.01% of page (10x10 in 0-1000 scale)


# ============================================================================
# SPATIAL EXTRACTION RESULT
# ============================================================================


@dataclass
class SpatialExtractionResult:
    """Result of spatial extraction from Docling element."""

    bbox_normalized: Optional[List[int]] = None
    page_width: Optional[float] = None
    page_height: Optional[float] = None
    raw_bbox: Optional[List[float]] = None
    extraction_source: str = "none"

    @property
    def is_valid(self) -> bool:
        """Check if extraction produced valid spatial data."""
        if self.bbox_normalized is None:
            return False
        # Check minimum area
        l, t, r, b = self.bbox_normalized
        area = (r - l) * (b - t)
        return area >= MIN_BBOX_AREA_NORMALIZED

    @property
    def area_ratio(self) -> float:
        """Calculate bbox area as ratio of page (0.0 - 1.0)."""
        if self.bbox_normalized is None:
            return 0.0
        l, t, r, b = self.bbox_normalized
        return ((r - l) * (b - t)) / (1000 * 1000)


# ============================================================================
# SPATIAL PROPAGATOR
# ============================================================================


class SpatialPropagator:
    """
    Universal spatial data extractor and normalizer.

    Extracts bbox from ANY Docling element and normalizes to 0-1000 scale.
    Solves the "95% spatial:null" problem by propagating to ALL modalities.

    Usage:
        propagator = SpatialPropagator()
        result = propagator.extract_and_normalize(element, page_dims)
        if result.is_valid:
            spatial_metadata = SpatialMetadata(bbox=result.bbox_normalized)
    """

    def __init__(self) -> None:
        """Initialize SpatialPropagator."""
        self._extraction_count = 0
        self._success_count = 0
        self._failure_count = 0

    def extract_and_normalize(
        self,
        element: Any,
        page_dims: Optional[Tuple[float, float]] = None,
        context: str = "unknown",
    ) -> SpatialExtractionResult:
        """
        Extract spatial data from Docling element and normalize to 0-1000.

        Tries multiple extraction paths:
        1. element.prov[0].bbox (primary provenance)
        2. element.prov.bbox (single provenance)
        3. element.bbox (direct bbox attribute)
        4. element.bounding_box (alternative naming)

        Args:
            element: Docling document element
            page_dims: (width, height) in PDF points, or None for defaults
            context: Context string for logging

        Returns:
            SpatialExtractionResult with normalized bbox or None
        """
        self._extraction_count += 1

        # Default page dimensions
        page_width = page_dims[0] if page_dims else DEFAULT_PAGE_WIDTH
        page_height = page_dims[1] if page_dims else DEFAULT_PAGE_HEIGHT

        result = SpatialExtractionResult(
            page_width=page_width,
            page_height=page_height,
        )

        # Try extraction paths in order of preference
        raw_bbox = self._try_extract_raw_bbox(element)

        if raw_bbox is None:
            self._failure_count += 1
            logger.debug(f"[{context}] No bbox found in element provenance")
            return result

        result.raw_bbox = raw_bbox
        result.extraction_source = "provenance"

        # Normalize to 0-1000 integer scale
        try:
            normalized = ensure_normalized(
                raw_bbox,
                page_width=page_width,
                page_height=page_height,
                context=context,
            )

            # Validate dimensions (r > l, b > t)
            # NOTE: Some Docling elements have zero-area bboxes (points/lines)
            # This is expected behavior - log at DEBUG level, not WARNING
            if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
                logger.debug(
                    f"[{context}] Zero-area bbox (expected for some text elements): "
                    f"raw={raw_bbox} → normalized={normalized}"
                )
                self._failure_count += 1
                return result

            result.bbox_normalized = normalized
            self._success_count += 1

            logger.debug(
                f"[{context}] Normalized bbox: {raw_bbox} → {normalized} "
                f"(area={result.area_ratio:.2%})"
            )

        except Exception as e:
            logger.warning(f"[{context}] Bbox normalization failed: {e}")
            self._failure_count += 1

        return result

    def _try_extract_raw_bbox(self, element: Any) -> Optional[List[float]]:
        """
        Try multiple extraction paths for raw bbox.

        Returns:
            Raw bbox as [x0, y0, x1, y1] or None
        """
        # Path 1: element.prov[0].bbox (list provenance)
        if hasattr(element, "prov") and element.prov:
            prov = element.prov[0] if isinstance(element.prov, list) else element.prov

            if hasattr(prov, "bbox") and prov.bbox:
                bbox_obj = prov.bbox
                return self._parse_bbox_object(bbox_obj)

        # Path 2: element.bbox (direct attribute)
        if hasattr(element, "bbox") and element.bbox:
            return self._parse_bbox_object(element.bbox)

        # Path 3: element.bounding_box (alternative naming)
        if hasattr(element, "bounding_box") and element.bounding_box:
            return self._parse_bbox_object(element.bounding_box)

        return None

    def _parse_bbox_object(self, bbox_obj: Any) -> Optional[List[float]]:
        """
        Parse bbox object to [x0, y0, x1, y1] list.

        Handles multiple Docling bbox formats:
        - Object with l, t, r, b attributes
        - Object with as_tuple() method
        - List/tuple [x0, y0, x1, y1]
        - Object with x0, y0, x1, y1 attributes
        """
        try:
            # Format 1: l, t, r, b attributes
            if hasattr(bbox_obj, "l") and hasattr(bbox_obj, "t"):
                x0 = min(float(bbox_obj.l), float(bbox_obj.r))
                x1 = max(float(bbox_obj.l), float(bbox_obj.r))
                y0 = min(float(bbox_obj.t), float(bbox_obj.b))
                y1 = max(float(bbox_obj.t), float(bbox_obj.b))
                if x1 > x0 and y1 > y0:
                    return [x0, y0, x1, y1]

            # Format 2: as_tuple() method
            if hasattr(bbox_obj, "as_tuple"):
                raw = bbox_obj.as_tuple()
                x0 = min(float(raw[0]), float(raw[2]))
                x1 = max(float(raw[0]), float(raw[2]))
                y0 = min(float(raw[1]), float(raw[3]))
                y1 = max(float(raw[1]), float(raw[3]))
                if x1 > x0 and y1 > y0:
                    return [x0, y0, x1, y1]

            # Format 3: List/tuple
            if isinstance(bbox_obj, (list, tuple)) and len(bbox_obj) >= 4:
                x0 = min(float(bbox_obj[0]), float(bbox_obj[2]))
                x1 = max(float(bbox_obj[0]), float(bbox_obj[2]))
                y0 = min(float(bbox_obj[1]), float(bbox_obj[3]))
                y1 = max(float(bbox_obj[1]), float(bbox_obj[3]))
                if x1 > x0 and y1 > y0:
                    return [x0, y0, x1, y1]

            # Format 4: x0, y0, x1, y1 attributes
            if hasattr(bbox_obj, "x0") and hasattr(bbox_obj, "y0"):
                x0 = min(float(bbox_obj.x0), float(bbox_obj.x1))
                x1 = max(float(bbox_obj.x0), float(bbox_obj.x1))
                y0 = min(float(bbox_obj.y0), float(bbox_obj.y1))
                y1 = max(float(bbox_obj.y0), float(bbox_obj.y1))
                if x1 > x0 and y1 > y0:
                    return [x0, y0, x1, y1]

        except (TypeError, ValueError, AttributeError) as e:
            logger.debug(f"Failed to parse bbox object: {e}")

        return None

    def get_stats(self) -> Dict[str, int]:
        """Get extraction statistics."""
        return {
            "total_extractions": self._extraction_count,
            "successful": self._success_count,
            "failed": self._failure_count,
            "success_rate": round(self._success_count / max(1, self._extraction_count) * 100, 1),
        }


# ============================================================================
# FACTORY FUNCTION
# ============================================================================


def create_spatial_propagator() -> SpatialPropagator:
    """Create a SpatialPropagator instance."""
    return SpatialPropagator()
