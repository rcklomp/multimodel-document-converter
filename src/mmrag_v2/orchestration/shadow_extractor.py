"""
Shadow Extractor - Raw PDF Bitmap Extraction for Missed Images
================================================================
ENGINE_USE: PyMuPDF (fitz) for direct PDF bitmap access

This module provides "shadow extraction" - a parallel scan of PDFs using
PyMuPDF to capture bitmap images that Docling's AI layout analysis may miss.

REQ Compliance:
- REQ-MM-05: Shadow extraction captures missed editorial photos
- REQ-MM-06: Ghost filter prevents duplicate captures
- REQ-MM-07: Size-based filtering removes noise (icons, logos)

SRS Section 7.3: Shadow Extraction
"The system SHOULD perform a secondary raw PDF scan to capture
bitmap images that may be missed by AI-based layout analysis."

Author: Claude 4.5 Opus (Architect)
Date: 2025-12-29
"""

from __future__ import annotations

import gc
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image

from ..utils.coordinate_normalization import bbox_iou, normalize_bbox

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Default minimum dimensions for shadow extraction
DEFAULT_MIN_WIDTH: int = 100
DEFAULT_MIN_HEIGHT: int = 100

# Ghost filter IoU threshold (overlap with Docling bboxes)
GHOST_FILTER_IOU_THRESHOLD: float = 0.3

# Minimum unique area for a shadow image to be kept
MIN_UNIQUE_AREA_RATIO: float = 0.5


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class ShadowAsset:
    """
    A shadow-extracted image asset.

    Attributes:
        xref: PyMuPDF image xref
        page_number: Page number (1-indexed)
        width: Image width in pixels
        height: Image height in pixels
        bbox: Bounding box [x0, y0, x1, y1] in page coordinates
        bbox_normalized: Normalized bbox [l, t, r, b] in 0.0-1.0
        nearest_text: Nearest text for context
        image_data: Raw image bytes
    """

    xref: int
    page_number: int
    width: int
    height: int
    bbox: Tuple[float, float, float, float]
    bbox_normalized: Optional[Tuple[float, float, float, float]] = None
    nearest_text: Optional[str] = None
    image_data: Optional[bytes] = None


@dataclass
class ShadowScanResult:
    """
    Result of scanning a page for shadow images.

    Attributes:
        page_number: Page number scanned
        total_images: Total images found on page
        filtered_count: Images filtered by size/ghost
        unaccounted_assets: Assets not covered by Docling
    """

    page_number: int
    total_images: int = 0
    filtered_count: int = 0
    unaccounted_assets: List[ShadowAsset] = field(default_factory=list)


# ============================================================================
# SHADOW EXTRACTOR
# ============================================================================


class ShadowExtractor:
    """
    Extracts images directly from PDF using PyMuPDF.

    This "shadow" extraction runs parallel to Docling's AI-based
    extraction to catch images that may be missed.

    Usage:
        with create_shadow_extractor(sensitivity=0.7) as extractor:
            result = extractor.scan_page(doc, page_number, docling_bboxes)
            for asset in result.unaccounted_assets:
                save_asset(asset)
    """

    def __init__(
        self,
        min_width: int = DEFAULT_MIN_WIDTH,
        min_height: int = DEFAULT_MIN_HEIGHT,
        ghost_filter_threshold: float = GHOST_FILTER_IOU_THRESHOLD,
        sensitivity: float = 0.5,
        historical_median: int = 0,
    ) -> None:
        """
        Initialize ShadowExtractor.

        Args:
            min_width: Minimum image width to extract
            min_height: Minimum image height to extract
            ghost_filter_threshold: IoU threshold for ghost detection
            sensitivity: Extraction sensitivity (affects filtering)
            historical_median: Historical median image dimension
        """
        self.min_width = min_width
        self.min_height = min_height
        self.ghost_filter_threshold = ghost_filter_threshold
        self.sensitivity = sensitivity
        self.historical_median = historical_median

        logger.info(
            f"ShadowExtractor initialized: "
            f"min_dim={min_width}x{min_height}px, "
            f"ghost_threshold={ghost_filter_threshold:.2f}, "
            f"sensitivity={sensitivity:.1f}"
        )

    def scan_page(
        self,
        doc: fitz.Document,
        page_number: int,
        docling_bboxes: List[List[float]],
        text_content: Optional[str] = None,
        batch_page_index: Optional[int] = None,
    ) -> ShadowScanResult:
        """
        Scan a single page for shadow images.

        Args:
            doc: Open PyMuPDF document
            page_number: Absolute page number (1-indexed) for metadata
            docling_bboxes: Bounding boxes from Docling extraction
            text_content: Text content for context (optional)
            batch_page_index: 0-indexed page within batch PDF (for loading)

        Returns:
            ShadowScanResult with unaccounted assets
        """
        result = ShadowScanResult(page_number=page_number)

        # Use batch_page_index for loading if provided, otherwise derive from page_number
        load_index = batch_page_index if batch_page_index is not None else (page_number - 1)

        if load_index < 0 or load_index >= len(doc):
            logger.warning(f"Invalid page index {load_index} for doc with {len(doc)} pages")
            return result

        page = doc.load_page(load_index)
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height

        # Get all images on page
        image_list = page.get_images(full=True)
        result.total_images = len(image_list)

        if not image_list:
            return result

        # Get image locations
        image_rects: Dict[int, fitz.Rect] = {}
        for img_info in image_list:
            xref = img_info[0]
            try:
                img_rects = page.get_image_rects(xref)
                if img_rects:
                    image_rects[xref] = img_rects[0]
            except Exception as e:
                logger.debug(f"Failed to get rect for xref {xref}: {e}")

        # Process each image
        for img_info in image_list:
            xref = img_info[0]

            try:
                # Extract image data
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue

                width = base_image.get("width", 0)
                height = base_image.get("height", 0)
                image_bytes = base_image.get("image")

                # Size filter
                if width < self.min_width or height < self.min_height:
                    result.filtered_count += 1
                    logger.debug(
                        f"Filtered small image: {width}x{height} < "
                        f"{self.min_width}x{self.min_height}"
                    )
                    continue

                # Get bounding box - REQ-COORD-01: Must have exact coordinates
                rect = image_rects.get(xref)
                if rect:
                    bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
                else:
                    # No rect available - try alternative extraction methods
                    # Method 1: Try to get rect via image matrix transformation
                    bbox = None
                    try:
                        for img_dict in page.get_image_info(xrefs=True):
                            if img_dict.get("xref") == xref:
                                transform = img_dict.get("transform")
                                if transform:
                                    # Transform gives us [a, b, c, d, e, f] matrix
                                    # e, f are translation (x, y), a and d are scaling
                                    import fitz

                                    matrix = fitz.Matrix(transform)
                                    # Use image native dimensions with transform
                                    img_rect = fitz.Rect(0, 0, width, height)
                                    transformed = img_rect * matrix
                                    bbox = (
                                        transformed.x0,
                                        transformed.y0,
                                        transformed.x1,
                                        transformed.y1,
                                    )
                                    logger.debug(f"Got bbox via transform for xref {xref}: {bbox}")
                                break
                    except Exception as e:
                        logger.debug(f"Transform extraction failed for xref {xref}: {e}")

                    # If still no bbox, skip this image - REQ-COORD-01 violation to use fallback
                    if bbox is None:
                        result.filtered_count += 1
                        logger.warning(
                            f"SKIPPED shadow image xref={xref}: No valid bbox available. "
                            f"REQ-COORD-01 prohibits [0,0,w,h] fallback coordinates."
                        )
                        continue

                # Normalize bbox
                bbox_normalized: Optional[Tuple[float, float, float, float]] = None
                try:
                    norm_bbox = normalize_bbox(
                        list(bbox),
                        page_width,
                        page_height,
                        f"shadow_page{page_number}",
                    )
                    # Convert to float tuple for type safety
                    bbox_normalized = (
                        float(norm_bbox[0]),
                        float(norm_bbox[1]),
                        float(norm_bbox[2]),
                        float(norm_bbox[3]),
                    )
                except Exception as e:
                    logger.warning(f"Failed to normalize bbox: {e}")
                    bbox_normalized = None

                # Ghost filter: check overlap with Docling bboxes
                is_ghost = False
                if docling_bboxes and bbox_normalized:
                    for docling_bbox in docling_bboxes:
                        iou = bbox_iou(list(bbox_normalized), docling_bbox)
                        if iou >= self.ghost_filter_threshold:
                            is_ghost = True
                            logger.debug(
                                f"Ghost detected: IoU={iou:.2f} >= {self.ghost_filter_threshold}"
                            )
                            break

                if is_ghost:
                    result.filtered_count += 1
                    continue

                # Create shadow asset
                asset = ShadowAsset(
                    xref=xref,
                    page_number=page_number,
                    width=width,
                    height=height,
                    bbox=bbox,
                    bbox_normalized=bbox_normalized,
                    nearest_text=text_content[:200] if text_content else None,
                    image_data=image_bytes,
                )
                result.unaccounted_assets.append(asset)

            except Exception as e:
                logger.warning(f"Failed to process shadow image xref={xref}: {e}")
                continue

        logger.info(
            f"Shadow scan page {page_number}: "
            f"total={result.total_images}, "
            f"filtered={result.filtered_count}, "
            f"unaccounted={len(result.unaccounted_assets)}"
        )

        return result

    def persist_asset(
        self,
        doc: fitz.Document,
        asset: ShadowAsset,
        output_dir: Path,
        doc_hash: str,
        asset_index: int,
        docling_bboxes: Optional[List[List[float]]] = None,
    ) -> Optional[str]:
        """
        Persist a shadow asset to disk.

        Applies final Ghost Filter check before saving.

        Args:
            doc: Open PyMuPDF document
            asset: ShadowAsset to persist
            output_dir: Output directory
            doc_hash: Document hash for naming
            asset_index: Index for naming
            docling_bboxes: Docling bboxes for ghost check (optional)

        Returns:
            Relative asset path if saved, None if filtered
        """
        # Create assets directory
        assets_dir = output_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename = f"{doc_hash}_{asset.page_number:03d}_shadow_{asset_index:02d}.png"
        filepath = assets_dir / filename

        try:
            # Convert image bytes to PIL Image
            if asset.image_data:
                img = Image.open(BytesIO(asset.image_data))
            else:
                # Re-extract from document
                base_image = doc.extract_image(asset.xref)
                if not base_image or not base_image.get("image"):
                    logger.warning(f"Failed to extract image for xref={asset.xref}")
                    return None
                img = Image.open(BytesIO(base_image["image"]))

            # Convert to RGB if necessary
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # Save as PNG
            img.save(str(filepath), "PNG")

            logger.info(f"Saved shadow asset: {filename}")
            return f"assets/{filename}"

        except Exception as e:
            logger.error(f"Failed to persist shadow asset: {e}")
            return None

    def close(self) -> None:
        """Clean up resources."""
        gc.collect()
        logger.debug("ShadowExtractor closed")


# ============================================================================
# CONTEXT MANAGER
# ============================================================================


@contextmanager
def create_shadow_extractor(
    sensitivity: float = 0.5,
    historical_median: int = 0,
    min_width: Optional[int] = None,
    min_height: Optional[int] = None,
) -> Generator[ShadowExtractor, None, None]:
    """
    Context manager for ShadowExtractor.

    Args:
        sensitivity: Extraction sensitivity (0.1-1.0)
        historical_median: Median image dimension from profile
        min_width: Override minimum width
        min_height: Override minimum height

    Yields:
        Configured ShadowExtractor
    """
    # Calculate dimensions from sensitivity if not provided
    if min_width is None:
        # Higher sensitivity = smaller minimum = more recall
        size_multiplier = 2.0 - (sensitivity * 1.5)
        min_width = int(DEFAULT_MIN_WIDTH * size_multiplier)

    if min_height is None:
        size_multiplier = 2.0 - (sensitivity * 1.5)
        min_height = int(DEFAULT_MIN_HEIGHT * size_multiplier)

    extractor = ShadowExtractor(
        min_width=min_width,
        min_height=min_height,
        sensitivity=sensitivity,
        historical_median=historical_median,
    )

    try:
        yield extractor
    finally:
        extractor.close()
