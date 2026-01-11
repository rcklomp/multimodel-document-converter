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
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import fitz  # PyMuPDF

if TYPE_CHECKING:
    from .document_diagnostic import DiagnosticReport

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
    """Document type classification.

    V16 FIX (2026-01-10): Added LITERATURE and TECHNICAL types
    to properly distinguish scanned books from magazines and
    technical manuals from generic reports.
    """

    MAGAZINE = "magazine"  # High image density, editorial photos
    ACADEMIC = "academic"  # Text-heavy, figures/diagrams
    REPORT = "report"  # Mixed content, charts/tables
    PRESENTATION = "presentation"  # Slide-based, high visual content
    LITERATURE = "literature"  # V16: Scanned books, novels, fiction (long-form narrative)
    TECHNICAL = "technical"  # V16: Technical manuals, handbooks, documentation
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

    def analyze(
        self,
        pdf_path: Path | str,
        diagnostic_report: Optional["DiagnosticReport"] = None,
    ) -> DocumentProfile:
        """
        Analyze a PDF document and return its profile.

        Args:
            pdf_path: Path to PDF file
            diagnostic_report: Optional diagnostic report from DocumentDiagnosticEngine

        Returns:
            DocumentProfile with document characteristics
        """
        return self._analyze_internal(pdf_path, diagnostic_report)

    def _analyze_internal(
        self,
        pdf_path: Path | str,
        diagnostic_report: Optional["DiagnosticReport"] = None,
    ) -> DocumentProfile:
        """Internal analyze method with diagnostic support."""
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
                text_result = page.get_text()
                text = str(text_result) if text_result else ""
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

            # Classify document type - NOW WITH DIAGNOSTIC CONTEXT!
            doc_type = self._classify_document(
                image_density=image_density,
                avg_text=avg_text,
                image_count=image_count,
                total_pages=total_pages,
                diagnostic_report=diagnostic_report,
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
        total_pages: int,
        diagnostic_report: Optional["DiagnosticReport"] = None,
    ) -> DocumentType:
        """
        Classify document based on characteristics.

        ARCHITECTURE: TEXT DENSITY IS KING (BUT SMART FOR SCANS & BOOKS)
        ==================================================================
        Academic papers have 3000-5000+ chars/page even with many figures/diagrams.
        Magazines have 500-1500 chars/page with editorial photos.
        TEXT DENSITY must override image density for academic detection.

        FIX (2025-01-10): Harry Potter misclassification
        -------------------------------------------------
        For scanned books (low native text but high page count + editorial domain),
        trust diagnostics over text density to avoid "presentation" trap.

        Args:
            image_density: Images per page ratio
            avg_text: Average text per page
            image_count: Total images found
            total_pages: Total number of pages in document
            diagnostic_report: Optional diagnostic context from DocumentDiagnosticEngine

        Returns:
            DocumentType classification
        """
        # ================================================================
        # PRIORITY RULE 0: PAGE COUNT BIAS FOR LITERATURE/BOOKS
        # ================================================================
        # Documents with > 50 pages are NEVER presentations or magazines.
        # This catches novels, textbooks, long-form fiction, etc.
        if total_pages > 50:
            # For scanned books with low native text, trust domain hints
            if diagnostic_report:
                domain = diagnostic_report.confidence_profile.detected_domain.value
                is_scan = diagnostic_report.physical_check.is_likely_scan

                # If it's a scan with "editorial" domain (book/fiction) and many pages
                # → Force to ACADEMIC (which represents "long-form text")
                if is_scan and domain == "editorial":
                    logger.info(
                        f"[SMART-DETECT] ACADEMIC (LITERATURE): {total_pages} pages + "
                        f"editorial domain + scan → long-form book/fiction"
                    )
                    return DocumentType.ACADEMIC

                # If domain is already academic, keep it
                if domain == "academic":
                    logger.info(f"[SMART-DETECT] ACADEMIC: {total_pages} pages + academic domain")
                    return DocumentType.ACADEMIC

            # Fallback: many pages + reasonable text = academic/report
            if avg_text > 100:  # Even low OCR text indicates content
                logger.info(
                    f"[SMART-DETECT] ACADEMIC (LONG-FORM): {total_pages} pages, "
                    f"forcing long-form classification (not presentation)"
                )
                return DocumentType.ACADEMIC

        # ================================================================
        # PRIORITY RULE 1: EDITORIAL DOMAIN = NEVER PRESENTATION
        # ================================================================
        # If diagnostics say "editorial" (books, magazines, articles),
        # trust that over image density calculations
        if diagnostic_report:
            domain = diagnostic_report.confidence_profile.detected_domain.value
            is_scan = diagnostic_report.physical_check.is_likely_scan

            if domain == "editorial":
                # Editorial content with scan = likely magazine or book
                if is_scan and avg_text < 100:
                    # Low OCR text but editorial → likely magazine or scanned book
                    if total_pages > 20:
                        logger.info(
                            f"[SMART-DETECT] ACADEMIC (SCANNED BOOK): Editorial domain + "
                            f"{total_pages} pages + scan"
                        )
                        return DocumentType.ACADEMIC
                    else:
                        logger.info(f"[SMART-DETECT] MAGAZINE: Editorial domain + scan + few pages")
                        return DocumentType.MAGAZINE
                elif not is_scan and image_density > 0.5:
                    # Digital editorial with images = magazine
                    logger.info(f"[SMART-DETECT] MAGAZINE: Editorial domain + high image density")
                    return DocumentType.MAGAZINE

            # If diagnostic says academic, trust it (even if low text due to scan)
            if domain == "academic" and is_scan:
                logger.info(
                    f"[SMART-DETECT] ACADEMIC: Academic domain + scan "
                    f"(trusting domain over low text density {avg_text:.0f})"
                )
                return DocumentType.ACADEMIC

            # V2.6 FIX: If diagnostic says ACADEMIC, trust it for DIGITAL docs too!
            # This catches academic papers with diagrams/figures that have moderate image density
            if domain == "academic" and not is_scan:
                logger.info(
                    f"[SMART-DETECT] ACADEMIC: Academic domain detected "
                    f"(text={avg_text:.0f}, density={image_density:.2f})"
                )
                return DocumentType.ACADEMIC

        # ================================================================
        # RULE 1: HIGH TEXT DENSITY = ACADEMIC (overrides all else)
        # Academic papers: 3500+ chars/page (even with many figures)
        # This catches: research papers, technical whitepapers, dissertations
        if avg_text > 3500:
            logger.info(
                f"[SMART-DETECT] ACADEMIC: High text density {avg_text:.0f} chars/page "
                f"(threshold: 3500+, image_density: {image_density:.2f})"
            )
            return DocumentType.ACADEMIC

        # RULE 2: MODERATE-HIGH TEXT WITH FIGURES = ACADEMIC
        # Papers with lots of diagrams still have 2500+ chars/page
        if avg_text > 2500 and image_density < 0.8:
            logger.info(
                f"[SMART-DETECT] ACADEMIC: Moderate text {avg_text:.0f} chars/page "
                f"with controlled image density {image_density:.2f}"
            )
            return DocumentType.ACADEMIC

        # RULE 3: VERY HIGH IMAGE DENSITY = PRESENTATION
        # Slides have >1 image per page
        if image_density >= 1.0:
            logger.info(f"[SMART-DETECT] PRESENTATION: Very high image density {image_density:.2f}")
            return DocumentType.PRESENTATION

        # RULE 4: HIGH IMAGE DENSITY + LOW TEXT = MAGAZINE
        # Magazines: 500-1500 chars/page with many editorial photos
        # NOTE: avg_text < 2500 prevents academic papers from being misclassified
        if image_density >= 0.5 and avg_text > 500 and avg_text < 2500:
            logger.info(
                f"[SMART-DETECT] MAGAZINE: High image density {image_density:.2f} "
                f"with low-moderate text {avg_text:.0f} chars/page"
            )
            return DocumentType.MAGAZINE

        # RULE 5: MODERATE EVERYTHING = REPORT
        # Business reports: 1500-2500 chars/page with charts/tables
        if image_count > 0 and avg_text > 1000:
            logger.info(
                f"[SMART-DETECT] REPORT: Moderate content "
                f"({avg_text:.0f} chars/page, {image_count} images)"
            )
            return DocumentType.REPORT

        # RULE 6: TEXT-ONLY = ACADEMIC
        # Pure text documents (books, essays)
        if avg_text > 500 and image_count == 0:
            logger.info(f"[SMART-DETECT] ACADEMIC: Text-only document {avg_text:.0f} chars/page")
            return DocumentType.ACADEMIC

        logger.warning(
            f"[SMART-DETECT] UNKNOWN: Could not classify "
            f"(text={avg_text:.0f}, density={image_density:.2f}, images={image_count})"
        )
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
