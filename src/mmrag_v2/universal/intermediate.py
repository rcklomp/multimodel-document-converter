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

    Frozen at 3 values per Charter §7.1: this is the legacy extraction
    vocabulary, being replaced by the 5-value ``Modality`` (which adds
    CODE/FORM). The widening lives at the ElementType->Modality boundary
    (the VLM adapter smuggles code/form through as TEXT, then the chunker
    promotes to Modality.CODE/FORM), NOT in this enum. Guarded by
    tests/test_v3_uir_contract.py::test_modality_distinct_from_elementtype.
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


# ============================================================================
# V3.0 UIR CONTRACT (additive — see docs/ARCHITECTURE_V3_DRAFT_0.5.md §3.2)
# ============================================================================
#
# These types are introduced for the v3.0 UIR refactor (Phase A). They live
# alongside the v2.X types above (Element, ElementType, UniversalPage,
# UniversalDocument) which remain the v2.16 production types. The Charter
# §7.1 type-vocabulary reconciliation specifies a one-way migration with
# no two-way shims — ElementType is replaced by Modality at the end of
# Phase A, not during the foundation session.
#
# Imports remain at module level (already present above): dataclass, field,
# Enum, Optional, Tuple, Dict, Any, Set, List.
# Add: Literal for ExtractionWarning.severity; uuid for continuation groups.
# These are stdlib-only additions; no new third-party dependency.

import uuid as _uuid  # noqa: E402  (intentionally after the v2.X block)
from typing import Literal, Set  # noqa: E402,F811  (Set may already be unused above)


class Modality(str, Enum):
    """V3.0 UIR modality. Charter §3.2.

    Replaces today's ElementType (TEXT/IMAGE/TABLE) with five values
    after Phase A migration (Charter §7.1). Inherits from `str` so the
    same enum is the single source of truth for the Pydantic
    `IngestionChunk.modality` field as well — eliminating the v2.16
    duplicate `schema.ingestion_schema.Modality` enum (C14 one-direction
    reconciliation: Modality replaces ElementType; ChunkType derives
    from Modality).
    """

    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CODE = "code"
    FORM = "form"


class LocatorType(Enum):
    """How to locate this element in its source document. Charter §3.2."""

    BBOX = "bbox"  # Fixed-layout: PDF, scanned images
    FLOW_OFFSET = "flow_offset"  # Reflowable: EPUB, HTML
    DOM_PATH = "dom_path"  # Structured: HTML, DOCX


class CoordinateFrame(Enum):
    """The frame `bbox` values are expressed in. Charter §3.2.

    [0,1000] is a normalization, not a frame — the frame says what was
    normalized.
    """

    PDF_PAGE_PORTRAIT = "pdf_page_portrait"
    PDF_PAGE_LANDSCAPE = "pdf_page_landscape"
    PDF_PAGE_ROTATED = "pdf_page_rotated"  # Page rotation handled at extraction
    IMAGE_NATIVE = "image_native"  # Scanned image native pixel frame
    EPUB_VIEWPORT = "epub_viewport"  # Reflowable; bbox at default viewport
    UNKNOWN = "unknown"


class StructuralFlag(Enum):
    """Closed vocabulary of structural flags. Charter §3.2.

    Open-ended `Dict[str, bool]` is forbidden — the semantic-identity
    gate ('flags additive') requires a canonical registry to diff against.
    Additions to this enum require an ADR per Charter §3.2.
    """

    PARTIAL_CODE_IN_BLOCK = "partial_code_in_block"  # v2.16 in-block oversized
    PARTIAL_CODE_CROSS_PAGE = "partial_code_cross_page"  # v3.0 new (activates inert)
    PARTIAL_TABLE_CROSS_PAGE = "partial_table_cross_page"  # v3.1+ deferred (reserved)
    CROSS_PAGE_SPLIT = "cross_page_split"
    ORPHAN_LABEL = "orphan_label"
    PICTURE_CLASSIFICATION_LABEL = "picture_classification_label"
    DROP_CAP_REPAIRED = "drop_cap_repaired"
    READING_ORDER_REPAIRED = "reading_order_repaired"
    OCR_FORCED = "ocr_forced"
    OCR_FALLBACK = "ocr_fallback"
    BBOX_IOU_DEDUPED = "bbox_iou_deduped"
    TRAILING_PREPOSITION_HEALED = "trailing_preposition_healed"


@dataclass
class ExtractionWarning:
    """Structured signal from the extraction engine. Charter §3.2.

    Replaces today's DOM-coupled inspection of Docling internals. The
    chunker / sanitizer / quality gate consult these instead of reaching
    back into engine-specific objects.
    """

    code: str  # e.g. "DOCLING_OCR_LOW_CONFIDENCE", "PAGE_ROTATION_CORRECTED"
    severity: Literal["info", "warn", "error"]
    message: str
    source_element_id: Optional[str] = None


@dataclass
class Locator:
    """Source-document location, format-appropriate. Charter §3.2."""

    type: LocatorType
    # For BBOX type:
    bbox: Optional[List[int]] = None  # [x1, y1, x2, y2] in [0, 1000]
    page_number: Optional[int] = None
    coordinate_frame: CoordinateFrame = CoordinateFrame.UNKNOWN
    # For FLOW_OFFSET / DOM_PATH types:
    path: Optional[str] = None  # CFI, DOM XPath, or character offset range

    def __post_init__(self) -> None:
        if self.type == LocatorType.BBOX:
            if self.bbox is None:
                raise ValueError("Locator(type=BBOX) requires bbox")
            if len(self.bbox) != 4:
                raise ValueError(f"Locator.bbox must be 4 ints, got len={len(self.bbox)}")
            for i, coord in enumerate(self.bbox):
                if not isinstance(coord, int):
                    raise TypeError(f"Locator.bbox[{i}]={coord!r} must be int (REQ-COORD-01)")
                if not (0 <= coord <= 1000):
                    raise ValueError(
                        f"Locator.bbox[{i}]={coord} out of range [0,1000] " "(REQ-COORD-01)"
                    )
        elif self.type in (LocatorType.FLOW_OFFSET, LocatorType.DOM_PATH):
            if self.path is None:
                raise ValueError(f"Locator(type={self.type.value}) requires path")


@dataclass
class ConfidenceBreakdown:
    """Per-source confidence scores. Charter §3.2.

    Single-sentinel convention: a field that is None means 'not measured
    for this chunk'. Whether a field is *applicable* is recorded in
    `applicable` (Set[str]) to avoid the two-sentinel encoding (None vs
    -1.0) that Draft 0.3 had. A field that is in `applicable` but has
    value None = 'applicable but unavailable' — request a re-extraction
    or treat as missing data.
    """

    layout_confidence: Optional[float] = None
    text_extraction_confidence: Optional[float] = None
    ocr_confidence: Optional[float] = None
    classification_confidence: Optional[float] = None
    applicable: Set[str] = field(default_factory=set)

    _VALID_FIELDS = frozenset(
        {
            "layout_confidence",
            "text_extraction_confidence",
            "ocr_confidence",
            "classification_confidence",
        }
    )

    def __post_init__(self) -> None:
        for name in self.applicable:
            if name not in self._VALID_FIELDS:
                raise ValueError(
                    f"ConfidenceBreakdown.applicable contains unknown field "
                    f"name {name!r}; valid: {sorted(self._VALID_FIELDS)}"
                )
        for name in self._VALID_FIELDS:
            value = getattr(self, name)
            if value is not None and not (0.0 <= value <= 1.0):
                raise ValueError(f"ConfidenceBreakdown.{name}={value} out of range [0,1]")


@dataclass
class UIRChunk:
    """V3.0 UIR chunk. Charter §3.2.

    Emitted by ElementProcessor; consumed by chunker + sanitizer. The
    provenance contract for sanitization (content / content_original /
    content_sanitized / sanitization_status) is documented in the
    Charter §3.2 docstring; the same semantics apply here.

    Note: the foundation session adds this type to the codebase. The
    v3.0 chunker and ElementProcessor rewrite that consumes it lands
    in Phase A tasks A2-A3.
    """

    modality: Modality
    content: str  # Always authoritative
    locator: Locator
    confidence: ConfidenceBreakdown
    extraction_method: str  # "docling_direct" | "ocr_tesseract" | ...
    extraction_engine_version: str  # e.g., "docling-2.86.0"
    extraction_warnings: List[ExtractionWarning] = field(default_factory=list)
    structural_flags: Set[StructuralFlag] = field(default_factory=set)
    source_element_ids: List[str] = field(default_factory=list)
    asset_ref: Optional[str] = None  # Path to extracted image/asset (IMAGE)
    lang: Optional[str] = None  # ISO 639-1 language code
    reading_order: Optional[int] = None  # Monotonic logical position

    # Provenance: the VLM's original element type when it did not map cleanly
    # onto the 3-value ElementType vocabulary - 'code'/'form' promoted to
    # Modality.CODE/FORM, or a non-standard type degraded to TEXT. None for
    # natively-typed extraction. Surfaced into
    # IngestionChunk.metadata.original_vlm_type by from_uir so smuggled and
    # degraded chunks stay auditable rather than silently indistinguishable
    # from prose. Additive-optional: does NOT bump uir_version (3.0 is still
    # the current contract; the exporter roundtrips this field symmetrically).
    original_vlm_type: Optional[str] = None

    # Provenance fields (Charter §3.2):
    content_original: Optional[str] = None  # Pre-sanitization raw extraction
    content_sanitized: Optional[str] = None  # LLM output when attempted
    sanitizer_model_id: Optional[str] = None
    sanitizer_prompt_version: Optional[str] = None  # Git-hash of prompt template
    sanitization_status: str = "not_applied"
    # Allowed sanitization_status values:
    #   "not_applied" | "accepted" | "rejected:<guard_name>"
    #     | "skipped:endpoint_unreachable" | "skipped:<reason>"

    # Hierarchical context:
    parent_element_id: Optional[str] = None  # Table cell → parent table
    parent_heading: Optional[str] = None  # Nearest ancestor heading text
    # Full TOC breadcrumb (e.g. [doc, Part, Chapter, Section, "Page N"]).
    # Populated by the UIR-native heading-assignment pass in uir_chunker
    # when a PyMuPDF TOC is threaded in (PLAN_V3.1 P2). Empty when no TOC
    # covers the chunk's page; consumed by IngestionChunk.from_uir.
    breadcrumb_path: List[str] = field(default_factory=list)

    # Cross-page sibling grouping (Charter §3.2, NEW in 0.5):
    #   Shared UUID across chunks that are halves/parts of the same logical
    #   element split across pages (cross-page code block, cross-page table).
    #   Required for the adjacency-fetch mechanism to identify which chunk
    #   is the continuation, not just that some continuation exists. None
    #   when the chunk is not part of a split.
    continuation_group_id: Optional[str] = None

    # UIR internal contract version (Charter §3.2, NEW in 0.5):
    #   Distinct from schema_version (which stamps the output JSONL).
    #   uir_version governs the internal extraction→chunking contract so
    #   the ElementProcessor knows which UIR generation it is consuming.
    #   Bumped on additive field changes to UIRChunk/UniversalDocument/
    #   UniversalPage.
    uir_version: str = "3.0"

    _VALID_SANITIZATION_STATUS_PREFIXES = (
        "not_applied",
        "accepted",
        "rejected:",
        "skipped:",
    )

    def __post_init__(self) -> None:
        if not any(
            self.sanitization_status == prefix or self.sanitization_status.startswith(prefix)
            for prefix in self._VALID_SANITIZATION_STATUS_PREFIXES
        ):
            raise ValueError(
                f"UIRChunk.sanitization_status={self.sanitization_status!r} "
                f"must be one of: not_applied, accepted, rejected:*, "
                f"skipped:*"
            )

    @staticmethod
    def new_continuation_group_id() -> str:
        """Generate a fresh continuation_group_id (UUIDv4).

        Use this when emitting the first chunk of a cross-page split; pass
        the same id to all sibling chunks in the group.
        """
        return str(_uuid.uuid4())
