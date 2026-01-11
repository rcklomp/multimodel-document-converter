"""
Universal Intermediate Representation (UIR) Data Structures
============================================================

This module defines the core data structures for the format-agnostic
document processing pipeline. All format engines convert their input
to these structures, enabling quality-based routing.

Key Structures:
    - Element: A single document element (text block, image, or table)
    - UniversalPage: A single page with classification and elements
    - UniversalDocument: Complete document with all pages

Design Principles:
    1. Format Agnostic: Same structure regardless of source (PDF, ePub, HTML)
    2. Quality Embedded: Confidence scores enable intelligent routing
    3. OCR Ready: raw_image fields allow deferred OCR processing
    4. Coordinate Normalized: All bboxes in 0-1000 range (SRS REQ-COORD-01)

SRS Compliance:
    - REQ-COORD-01: Coordinates normalized to 1000x1000 canvas
    - REQ-MM-01: 10px padding on element bboxes (applied during extraction)

Author: Claude (Architect)
Date: January 2026
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================


class ElementType(Enum):
    """
    Type of document element.

    TEXT: Text content (paragraphs, headings, lists)
    IMAGE: Visual content (figures, photos, diagrams)
    TABLE: Tabular data (tables, spreadsheets)
    """

    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"


class PageClassification(Enum):
    """
    Quality classification of a page.

    DIGITAL: Native text extraction works (confidence >= 0.7)
    SCANNED: Requires OCR processing (confidence < 0.5)
    HYBRID: Mixed content, some text extractable (0.5 <= confidence < 0.7)
    """

    DIGITAL = "digital"
    SCANNED = "scanned"
    HYBRID = "hybrid"


class ExtractionMethod(Enum):
    """
    Method used to extract content.

    NATIVE: Direct extraction from document structure
    OCR_TESSERACT: Tesseract OCR (Layer 1)
    OCR_DOCLING: Docling internal OCR (Layer 2)
    OCR_DOCTR: Doctr deep learning OCR (Layer 3)
    VLM: Vision Language Model description
    FALLBACK: Fallback method when primary fails
    """

    NATIVE = "native"
    OCR_TESSERACT = "ocr_tesseract"
    OCR_DOCLING = "ocr_docling"
    OCR_DOCTR = "ocr_doctr"
    VLM = "vlm"
    FALLBACK = "fallback"


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class BoundingBox:
    """
    Normalized bounding box (0-1000 coordinate system).

    SRS REQ-COORD-01: All coordinates MUST be integers in [0, 1000] range.

    Attributes:
        x_min: Left edge (0-1000)
        y_min: Top edge (0-1000)
        x_max: Right edge (0-1000)
        y_max: Bottom edge (0-1000)
    """

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def __post_init__(self) -> None:
        """Validate coordinates are in valid range."""
        for coord, name in [
            (self.x_min, "x_min"),
            (self.y_min, "y_min"),
            (self.x_max, "x_max"),
            (self.y_max, "y_max"),
        ]:
            if not isinstance(coord, int):
                raise TypeError(f"{name} must be int, got {type(coord).__name__}")
            if coord < 0 or coord > 1000:
                raise ValueError(f"{name}={coord} out of range [0, 1000]")

        if self.x_max <= self.x_min:
            raise ValueError(f"x_max ({self.x_max}) must be > x_min ({self.x_min})")
        if self.y_max <= self.y_min:
            raise ValueError(f"y_max ({self.y_max}) must be > y_min ({self.y_min})")

    def to_list(self) -> List[int]:
        """Convert to [x_min, y_min, x_max, y_max] list."""
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    @classmethod
    def from_list(cls, bbox: List[int]) -> "BoundingBox":
        """Create from [x_min, y_min, x_max, y_max] list."""
        if len(bbox) != 4:
            raise ValueError(f"bbox must have 4 elements, got {len(bbox)}")
        return cls(x_min=bbox[0], y_min=bbox[1], x_max=bbox[2], y_max=bbox[3])

    @classmethod
    def from_raw(
        cls,
        raw_bbox: List[float],
        page_width: float,
        page_height: float,
        apply_padding: bool = True,
        padding_px: int = 10,
    ) -> "BoundingBox":
        """
        Create from raw pixel coordinates with normalization.

        Args:
            raw_bbox: [x_min, y_min, x_max, y_max] in pixels
            page_width: Page width in pixels
            page_height: Page height in pixels
            apply_padding: Whether to apply REQ-MM-01 padding
            padding_px: Padding in pixels (default 10)

        Returns:
            Normalized BoundingBox (0-1000 scale)
        """
        x_min, y_min, x_max, y_max = raw_bbox

        # Apply padding (REQ-MM-01)
        if apply_padding:
            x_min = max(0.0, x_min - padding_px)
            y_min = max(0.0, y_min - padding_px)
            x_max = min(page_width, x_max + padding_px)
            y_max = min(page_height, y_max + padding_px)

        # Normalize to 0-1000 (REQ-COORD-01)
        norm_x_min = int(round((x_min / page_width) * 1000))
        norm_y_min = int(round((y_min / page_height) * 1000))
        norm_x_max = int(round((x_max / page_width) * 1000))
        norm_y_max = int(round((y_max / page_height) * 1000))

        # Clamp to valid range
        norm_x_min = max(0, min(1000, norm_x_min))
        norm_y_min = max(0, min(1000, norm_y_min))
        norm_x_max = max(0, min(1000, norm_x_max))
        norm_y_max = max(0, min(1000, norm_y_max))

        # Ensure valid box (min < max)
        if norm_x_max <= norm_x_min:
            norm_x_max = norm_x_min + 1
        if norm_y_max <= norm_y_min:
            norm_y_max = norm_y_min + 1

        return cls(
            x_min=norm_x_min,
            y_min=norm_y_min,
            x_max=norm_x_max,
            y_max=norm_y_max,
        )

    @property
    def width(self) -> int:
        """Width in normalized units."""
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        """Height in normalized units."""
        return self.y_max - self.y_min

    @property
    def area(self) -> int:
        """Area in normalized units squared."""
        return self.width * self.height

    @property
    def area_ratio(self) -> float:
        """Ratio of this box area to full page (0.0 - 1.0)."""
        return self.area / 1_000_000  # 1000 * 1000 = 1,000,000


@dataclass
class Element:
    """
    A single document element (text block, image, or table).

    This is the fundamental unit of content extraction. Elements are
    classified by type and carry quality metadata for routing decisions.

    Attributes:
        type: Element type (TEXT, IMAGE, TABLE)
        content: Extracted text content (empty for IMAGE before VLM)
        bbox: Normalized bounding box (0-1000 coordinates)
        confidence: Extraction confidence (0.0-1.0)
        raw_image: Cropped image data for OCR/VLM fallback
        extraction_method: How content was extracted
        element_index: Index within page (for ordering)
        source_label: Original label from source format (e.g., "paragraph", "figure")
        metadata: Additional format-specific metadata
    """

    type: ElementType
    content: str
    bbox: Optional[BoundingBox]
    confidence: float
    raw_image: Optional[np.ndarray] = None
    extraction_method: ExtractionMethod = ExtractionMethod.NATIVE
    element_index: int = 0
    source_label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate element data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence {self.confidence} out of range [0.0, 1.0]")

    @property
    def needs_ocr(self) -> bool:
        """Whether this element needs OCR processing."""
        return self.type == ElementType.TEXT and self.confidence < 0.7

    @property
    def needs_vlm(self) -> bool:
        """Whether this element needs VLM description."""
        return self.type == ElementType.IMAGE

    @property
    def has_image_data(self) -> bool:
        """Whether raw image data is available."""
        return self.raw_image is not None and self.raw_image.size > 0

    def get_bbox_list(self) -> Optional[List[int]]:
        """Get bbox as list or None."""
        return self.bbox.to_list() if self.bbox else None


@dataclass
class UniversalPage:
    """
    A single page from any document format.

    Pages contain elements and carry quality classification for routing.
    The classification is determined by analyzing text extraction confidence.

    Attributes:
        page_number: 1-indexed page number
        elements: List of elements on this page
        classification: Quality classification (DIGITAL, SCANNED, HYBRID)
        dimensions: (width, height) in pixels of original page
        raw_image: Full page render for fallback processing
        text_density: Ratio of text area to page area (0.0-1.0)
        avg_confidence: Average confidence of text elements
    """

    page_number: int
    elements: List[Element]
    classification: PageClassification
    dimensions: Tuple[int, int]
    raw_image: Optional[np.ndarray] = None
    text_density: float = 0.0
    avg_confidence: float = 0.0

    @classmethod
    def classify_page(
        cls,
        text_char_count: int,
        threshold_digital: int = 100,
        threshold_scanned: int = 20,
    ) -> PageClassification:
        """
        Classify page based on text character count.

        Args:
            text_char_count: Number of extractable characters
            threshold_digital: Above this = DIGITAL (default 100)
            threshold_scanned: Below this = SCANNED (default 20)

        Returns:
            PageClassification enum value
        """
        if text_char_count >= threshold_digital:
            return PageClassification.DIGITAL
        elif text_char_count <= threshold_scanned:
            return PageClassification.SCANNED
        else:
            return PageClassification.HYBRID

    @property
    def width(self) -> int:
        """Page width in pixels."""
        return self.dimensions[0]

    @property
    def height(self) -> int:
        """Page height in pixels."""
        return self.dimensions[1]

    @property
    def text_elements(self) -> List[Element]:
        """Get all TEXT elements."""
        return [e for e in self.elements if e.type == ElementType.TEXT]

    @property
    def image_elements(self) -> List[Element]:
        """Get all IMAGE elements."""
        return [e for e in self.elements if e.type == ElementType.IMAGE]

    @property
    def table_elements(self) -> List[Element]:
        """Get all TABLE elements."""
        return [e for e in self.elements if e.type == ElementType.TABLE]

    @property
    def is_scanned(self) -> bool:
        """Whether page is classified as scanned."""
        return self.classification == PageClassification.SCANNED

    @property
    def is_digital(self) -> bool:
        """Whether page is classified as digital."""
        return self.classification == PageClassification.DIGITAL

    def compute_avg_confidence(self) -> float:
        """Compute average confidence of text elements."""
        text_elements = self.text_elements
        if not text_elements:
            return 0.0
        return sum(e.confidence for e in text_elements) / len(text_elements)


@dataclass
class DocumentMetadata:
    """
    Metadata about a document.

    Attributes:
        title: Document title (if extractable)
        author: Document author (if extractable)
        creation_date: Document creation date
        modification_date: Last modification date
        page_count: Total number of pages
        file_size_bytes: File size in bytes
        has_text_layer: Whether document has extractable text
        has_images: Whether document contains images
        language: Detected or declared language
        extra: Additional format-specific metadata
    """

    title: Optional[str] = None
    author: Optional[str] = None
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    page_count: int = 0
    file_size_bytes: int = 0
    has_text_layer: bool = True
    has_images: bool = False
    language: str = "en"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UniversalDocument:
    """
    Format-agnostic document representation.

    This is the primary output of all format engines. It contains
    all pages with their elements, ready for quality-based processing.

    Attributes:
        doc_id: Unique document identifier (MD5 hash of file)
        source_file: Original filename
        file_type: Source format (pdf, epub, html, docx, etc.)
        pages: List of UniversalPage objects
        metadata: Document metadata
        total_pages: Total page count
        created_at: When this UIR was created
    """

    doc_id: str
    source_file: str
    file_type: str
    pages: List[UniversalPage] = field(default_factory=list)
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    total_pages: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def compute_doc_id(cls, file_path: Union[str, Path]) -> str:
        """
        Compute MD5 hash of file for unique identification.

        Args:
            file_path: Path to document file

        Returns:
            12-character hex string (first 12 chars of MD5)
        """
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:12]

    @property
    def scanned_page_count(self) -> int:
        """Number of pages classified as scanned."""
        return sum(1 for p in self.pages if p.is_scanned)

    @property
    def digital_page_count(self) -> int:
        """Number of pages classified as digital."""
        return sum(1 for p in self.pages if p.is_digital)

    @property
    def scanned_ratio(self) -> float:
        """Ratio of scanned pages to total pages."""
        if not self.pages:
            return 0.0
        return self.scanned_page_count / len(self.pages)

    @property
    def is_predominantly_scanned(self) -> bool:
        """Whether document is >50% scanned pages."""
        return self.scanned_ratio > 0.5

    @property
    def all_elements(self) -> List[Element]:
        """Get all elements from all pages (flattened)."""
        elements: List[Element] = []
        for page in self.pages:
            elements.extend(page.elements)
        return elements

    @property
    def total_text_elements(self) -> int:
        """Total count of TEXT elements."""
        return sum(len(p.text_elements) for p in self.pages)

    @property
    def total_image_elements(self) -> int:
        """Total count of IMAGE elements."""
        return sum(len(p.image_elements) for p in self.pages)

    @property
    def total_table_elements(self) -> int:
        """Total count of TABLE elements."""
        return sum(len(p.table_elements) for p in self.pages)

    def get_page(self, page_number: int) -> Optional[UniversalPage]:
        """Get page by 1-indexed page number."""
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None

    def summary(self) -> str:
        """Generate human-readable summary of document."""
        return (
            f"UniversalDocument(doc_id={self.doc_id}, "
            f"source={self.source_file}, "
            f"type={self.file_type}, "
            f"pages={len(self.pages)}, "
            f"digital={self.digital_page_count}, "
            f"scanned={self.scanned_page_count}, "
            f"text_elements={self.total_text_elements}, "
            f"image_elements={self.total_image_elements}, "
            f"table_elements={self.total_table_elements})"
        )


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================


def create_element(
    element_type: ElementType,
    content: str,
    bbox: Optional[List[int]] = None,
    confidence: float = 0.9,
    raw_image: Optional[np.ndarray] = None,
    extraction_method: ExtractionMethod = ExtractionMethod.NATIVE,
    element_index: int = 0,
    source_label: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Element:
    """
    Factory function to create an Element.

    Args:
        element_type: Type of element (TEXT, IMAGE, TABLE)
        content: Extracted text content
        bbox: Optional [x_min, y_min, x_max, y_max] normalized coordinates
        confidence: Extraction confidence (0.0-1.0)
        raw_image: Optional image data for fallback processing
        extraction_method: How content was extracted
        element_index: Index within page
        source_label: Original label from source format
        metadata: Additional metadata

    Returns:
        Element instance
    """
    bbox_obj = BoundingBox.from_list(bbox) if bbox else None

    return Element(
        type=element_type,
        content=content,
        bbox=bbox_obj,
        confidence=confidence,
        raw_image=raw_image,
        extraction_method=extraction_method,
        element_index=element_index,
        source_label=source_label,
        metadata=metadata or {},
    )


def create_page(
    page_number: int,
    elements: List[Element],
    dimensions: Tuple[int, int],
    classification: Optional[PageClassification] = None,
    raw_image: Optional[np.ndarray] = None,
) -> UniversalPage:
    """
    Factory function to create a UniversalPage.

    Args:
        page_number: 1-indexed page number
        elements: List of elements on this page
        dimensions: (width, height) in pixels
        classification: Optional override for classification
        raw_image: Optional full page image

    Returns:
        UniversalPage instance
    """
    # Auto-classify if not provided
    if classification is None:
        text_content = " ".join(e.content for e in elements if e.type == ElementType.TEXT)
        classification = UniversalPage.classify_page(len(text_content))

    page = UniversalPage(
        page_number=page_number,
        elements=elements,
        classification=classification,
        dimensions=dimensions,
        raw_image=raw_image,
    )

    # Compute derived values
    page.avg_confidence = page.compute_avg_confidence()

    return page


def create_document(
    file_path: Union[str, Path],
    file_type: str,
    pages: List[UniversalPage],
    metadata: Optional[DocumentMetadata] = None,
) -> UniversalDocument:
    """
    Factory function to create a UniversalDocument.

    Args:
        file_path: Path to source file
        file_type: Format type (pdf, epub, html, etc.)
        pages: List of UniversalPage objects
        metadata: Optional document metadata

    Returns:
        UniversalDocument instance
    """
    file_path = Path(file_path)

    doc = UniversalDocument(
        doc_id=UniversalDocument.compute_doc_id(file_path),
        source_file=file_path.name,
        file_type=file_type,
        pages=pages,
        metadata=metadata or DocumentMetadata(),
        total_pages=len(pages),
    )

    logger.info(f"Created UIR: {doc.summary()}")

    return doc
