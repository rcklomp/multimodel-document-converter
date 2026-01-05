"""
Smart Config Provider - Document Analysis and Profile Detection
================================================================
ENGINE_USE: PyMuPDF (fitz) for fast document analysis

This module provides intelligent document profiling to determine
optimal extraction strategies based on document characteristics.

REQ Compliance:
- REQ-SMART-01: Auto-detect document type (magazine, academic, report)
- REQ-SMART-02: Calculate image density and median dimensions
- REQ-SMART-03: Provide profile-based extraction recommendations

SRS Section 9: Smart Configuration
"The system SHOULD analyze document characteristics to automatically
configure extraction parameters for optimal results."

Author: Claude 4.5 Opus (Architect)
Date: 2025-12-29
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Thresholds for document classification
MAGAZINE_IMAGE_DENSITY_THRESHOLD: float = 0.3  # 30% of pages have images
ACADEMIC_TEXT_RATIO_THRESHOLD: float = 0.8  # 80% text-heavy pages

# Sampling parameters
MAX_SAMPLE_PAGES: int = 20  # Sample first N pages for analysis
MIN_SAMPLE_PAGES: int = 5


# ============================================================================
# ENUMS
# ============================================================================


class DocumentType(str, Enum):
    """Document type classification."""

    MAGAZINE = "magazine"  # High image density, editorial photos
    ACADEMIC = "academic"  # Text-heavy, figures/diagrams
    REPORT = "report"  # Mixed content, charts/tables
    PRESENTATION = "presentation"  # Slide-based, high visual content
    UNKNOWN = "unknown"


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class ImageInfo:
    """Information about a detected image."""

    page_number: int
    width: int
    height: int
    area: int
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class DocumentProfile:
    """
    Profile of document characteristics.

    Attributes:
        document_type: Classified document type
        total_pages: Total number of pages
        pages_analyzed: Number of pages sampled
        image_count: Total images found in sample
        image_density: Images per page ratio
        median_image_width: Median image width in pixels
        median_image_height: Median image height in pixels
        has_text: Whether document has extractable text
        avg_text_per_page: Average text length per page
        images_per_page: Dict tracking asset counts per page
        median_assets_per_page: Median asset count across all pages
    """

    document_type: DocumentType = DocumentType.UNKNOWN
    total_pages: int = 0
    pages_analyzed: int = 0
    image_count: int = 0
    image_density: float = 0.0
    median_image_width: int = 0
    median_image_height: int = 0
    has_text: bool = True
    avg_text_per_page: float = 0.0
    images: List[ImageInfo] = field(default_factory=list)
    images_per_page: Dict[int, int] = field(default_factory=dict)
    median_assets_per_page: float = 0.0

    def is_image_heavy(self) -> bool:
        """Check if document is image-heavy."""
        return self.image_density >= MAGAZINE_IMAGE_DENSITY_THRESHOLD

    def is_text_heavy(self) -> bool:
        """Check if document is text-heavy."""
        return self.avg_text_per_page > 1000 and self.image_density < 0.2


# ============================================================================
# SMART CONFIG PROVIDER
# ============================================================================


class SmartConfigProvider:
    """
    Analyzes documents to determine optimal extraction configuration.

    Uses PyMuPDF for fast analysis without full Docling processing.
    Samples first N pages to build a document profile.

    Usage:
        provider = SmartConfigProvider()
        profile = provider.analyze("document.pdf")
        print(f"Document type: {profile.document_type}")
    """

    def __init__(
        self,
        max_sample_pages: int = MAX_SAMPLE_PAGES,
    ) -> None:
        """
        Initialize SmartConfigProvider.

        Args:
            max_sample_pages: Maximum pages to sample for analysis
        """
        self.max_sample_pages = max_sample_pages
        logger.info(f"SmartConfigProvider initialized: sample={max_sample_pages} pages")

    def analyze(self, pdf_path: Path | str) -> DocumentProfile:
        """
        Analyze a PDF document and return its profile.

        Args:
            pdf_path: Path to PDF file

        Returns:
            DocumentProfile with document characteristics
        """
        pdf_path = Path(pdf_path).resolve()

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Analyzing document: {pdf_path.name}")

        doc: Optional[fitz.Document] = None
        try:
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)

            # Determine sample range
            pages_to_analyze = min(self.max_sample_pages, total_pages)

            # Collect statistics
            all_images: List[ImageInfo] = []
            total_text_length = 0
            pages_with_text = 0

            for page_num in range(pages_to_analyze):
                page = doc.load_page(page_num)

                # Extract text
                text = page.get_text()
                if text.strip():
                    pages_with_text += 1
                    total_text_length += len(text)

                # Extract image info
                image_list = page.get_images(full=True)
                for img_info in image_list:
                    xref = img_info[0]
                    try:
                        # Get image dimensions
                        base_image = doc.extract_image(xref)
                        if base_image:
                            width = base_image.get("width", 0)
                            height = base_image.get("height", 0)

                            if width > 0 and height > 0:
                                all_images.append(
                                    ImageInfo(
                                        page_number=page_num + 1,
                                        width=width,
                                        height=height,
                                        area=width * height,
                                        x0=0,
                                        y0=0,
                                        x1=width,
                                        y1=height,
                                    )
                                )
                    except Exception as e:
                        logger.debug(f"Failed to extract image info: {e}")

            # Calculate statistics
            image_count = len(all_images)
            image_density = image_count / pages_to_analyze if pages_to_analyze > 0 else 0
            avg_text = total_text_length / pages_to_analyze if pages_to_analyze > 0 else 0

            # Calculate median dimensions
            median_width = 0
            median_height = 0
            if all_images:
                widths = sorted([img.width for img in all_images])
                heights = sorted([img.height for img in all_images])
                mid = len(widths) // 2
                median_width = widths[mid]
                median_height = heights[mid]

            # Classify document type
            doc_type = self._classify_document(
                image_density=image_density,
                avg_text=avg_text,
                image_count=image_count,
            )

            profile = DocumentProfile(
                document_type=doc_type,
                total_pages=total_pages,
                pages_analyzed=pages_to_analyze,
                image_count=image_count,
                image_density=image_density,
                median_image_width=median_width,
                median_image_height=median_height,
                has_text=pages_with_text > 0,
                avg_text_per_page=avg_text,
                images=all_images,
            )

            logger.info(
                f"Document profile: type={doc_type.value}, "
                f"pages={total_pages}, images={image_count}, "
                f"density={image_density:.2f}, median_dim={median_width}x{median_height}"
            )

            return profile

        finally:
            if doc is not None:
                doc.close()

    def _classify_document(
        self,
        image_density: float,
        avg_text: float,
        image_count: int,
    ) -> DocumentType:
        """
        Classify document based on characteristics.

        Args:
            image_density: Images per page ratio
            avg_text: Average text per page
            image_count: Total images found

        Returns:
            DocumentType classification
        """
        # High image density with moderate text = Magazine
        if image_density >= 0.5 and avg_text > 500:
            return DocumentType.MAGAZINE

        # Very high image density = Presentation
        if image_density >= 1.0:
            return DocumentType.PRESENTATION

        # High text, low images = Academic
        if avg_text > 2000 and image_density < 0.2:
            return DocumentType.ACADEMIC

        # Moderate everything = Report
        if image_count > 0:
            return DocumentType.REPORT

        # Text-only
        if avg_text > 500 and image_count == 0:
            return DocumentType.ACADEMIC

        return DocumentType.UNKNOWN

    def get_quick_stats(self, pdf_path: Path | str) -> Tuple[int, int]:
        """
        Get quick page and image count without full analysis.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Tuple of (page_count, approximate_image_count)
        """
        pdf_path = Path(pdf_path).resolve()
        doc: Optional[fitz.Document] = None

        try:
            doc = fitz.open(str(pdf_path))
            page_count = len(doc)

            # Sample first 5 pages for image estimate
            sample_size = min(5, page_count)
            image_count = 0

            for i in range(sample_size):
                page = doc.load_page(i)
                image_count += len(page.get_images(full=True))

            # Extrapolate
            estimated_images = int(image_count * (page_count / sample_size))

            return page_count, estimated_images

        finally:
            if doc is not None:
                doc.close()
