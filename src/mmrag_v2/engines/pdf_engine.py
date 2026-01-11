"""
PDF Engine - Convert PDFs to Universal Intermediate Representation
===================================================================

This engine converts PDF documents to UIR using Docling v2.66.0 for
layout analysis and element extraction. It handles both digital and
scanned PDFs with proper quality classification.

Key Features:
    - Per-page classification (digital vs scanned)
    - Element-level confidence scoring
    - Layout-aware region detection
    - Support for batch processing

SRS Compliance:
    - REQ-PDF-01: De-columnization via Docling
    - REQ-PDF-02: Ad detection (delegated to processor)
    - REQ-PDF-03: Hybrid OCR via confidence thresholds
    - REQ-PDF-04: High-fidelity rendering (scale 2.0)
    - REQ-PDF-05: Memory hygiene

Author: Claude (Architect)
Date: January 2026
"""

from __future__ import annotations

import gc
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

# Disable PIL decompression bomb check for large DPI renders
Image.MAX_IMAGE_PIXELS = None

from .base import BaseBinaryEngine
from ..universal.intermediate import (
    BoundingBox,
    DocumentMetadata,
    Element,
    ElementType,
    ExtractionMethod,
    PageClassification,
    UniversalDocument,
    UniversalPage,
    create_document,
    create_element,
    create_page,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Thresholds for page classification
DIGITAL_TEXT_THRESHOLD = 100  # chars: above this = digital
SCANNED_TEXT_THRESHOLD = 20  # chars: below this = scanned

# Docling label to ElementType mapping
LABEL_TYPE_MAP: Dict[str, ElementType] = {
    "text": ElementType.TEXT,
    "paragraph": ElementType.TEXT,
    "section_header": ElementType.TEXT,
    "title": ElementType.TEXT,
    "list_item": ElementType.TEXT,
    "caption": ElementType.TEXT,
    "footnote": ElementType.TEXT,
    "picture": ElementType.IMAGE,
    "figure": ElementType.IMAGE,
    "image": ElementType.IMAGE,
    "background": ElementType.IMAGE,
    "table": ElementType.TABLE,
}

# Minimum image size to extract (filter noise)
MIN_IMAGE_WIDTH = 50
MIN_IMAGE_HEIGHT = 50


# ============================================================================
# PDF ENGINE
# ============================================================================


class PDFEngine(BaseBinaryEngine):
    """
    PDF extraction engine using Docling v2.66.0.

    Converts PDF documents to Universal Intermediate Representation with:
    - Per-page quality classification (digital vs scanned)
    - Element extraction with confidence scores
    - Raw image data for OCR fallback

    Usage:
        engine = PDFEngine()
        uir = engine.convert("document.pdf")

        for page in uir.pages:
            print(f"Page {page.page_number}: {page.classification}")
            for element in page.elements:
                if element.needs_ocr:
                    # Run OCR on element.raw_image
                    pass

    Attributes:
        render_dpi: DPI for page rendering (default 300)
        extract_images: Whether to extract element images
    """

    def __init__(
        self,
        render_dpi: int = 300,
        extract_images: bool = True,
        enable_ocr: bool = True,
    ) -> None:
        """
        Initialize PDF engine.

        Args:
            render_dpi: DPI for rendering pages (higher = better OCR)
            extract_images: Whether to extract element images for OCR
            enable_ocr: Whether to enable OCR in Docling
        """
        super().__init__(
            name="PDFEngine",
            version="3.0.0",
            magic_bytes=b"%PDF",
        )
        self.render_dpi = render_dpi
        self.extract_images = extract_images
        self.enable_ocr = enable_ocr

        self._converter = None
        self._docling_available = False

    @property
    def supported_extensions(self) -> List[str]:
        """List of supported file extensions."""
        return [".pdf", ".PDF"]

    def detect(self, file_path: Path) -> bool:
        """Check if file is a valid PDF."""
        file_path = Path(file_path)

        if file_path.suffix.lower() != ".pdf":
            return False

        return self.check_magic_bytes(file_path)

    def _do_initialize(self) -> None:
        """Initialize Docling converter."""
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                EasyOcrOptions,
                PdfPipelineOptions,
            )
            from docling.document_converter import DocumentConverter, PdfFormatOption

            # Configure pipeline options (REQ-PDF-04)
            pipeline_options = PdfPipelineOptions()
            pipeline_options.images_scale = 2.0  # High-fidelity rendering
            pipeline_options.generate_page_images = True  # For classification
            pipeline_options.generate_picture_images = True  # Extract figures
            pipeline_options.generate_table_images = True  # Extract tables

            if self.enable_ocr:
                pipeline_options.do_ocr = True
                pipeline_options.ocr_options = EasyOcrOptions()
            else:
                pipeline_options.do_ocr = False

            self._converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
            )

            self._docling_available = True
            logger.info("ENGINE_USE: Docling v2.66.0 initialized for PDF processing")

        except ImportError as e:
            logger.warning(f"Docling not available: {e}. Using PyMuPDF fallback.")
            self._docling_available = False

    def convert(self, file_path: Path) -> UniversalDocument:
        """
        Convert PDF to Universal Intermediate Representation.

        Args:
            file_path: Path to PDF file

        Returns:
            UniversalDocument with extracted content
        """
        file_path = Path(file_path)
        self.validate_file(file_path)
        self.initialize()

        logger.info(f"Converting PDF to UIR: {file_path.name}")

        if self._docling_available and self._converter:
            return self._convert_with_docling(file_path)
        else:
            return self._convert_with_pymupdf(file_path)

    def _convert_with_docling(self, file_path: Path) -> UniversalDocument:
        """Convert using Docling engine."""
        logger.info(f"Using Docling for: {file_path.name}")

        # Run Docling conversion
        result = self._converter.convert(str(file_path))
        doc = result.document

        # Get page information
        page_dims: Dict[int, Tuple[float, float]] = {}
        page_images: Dict[int, Image.Image] = {}

        if hasattr(doc, "pages") and doc.pages:
            pages_iter = doc.pages.values() if isinstance(doc.pages, dict) else doc.pages
            for page in pages_iter:
                pg_no = getattr(page, "page_no", 1) or 1
                width = getattr(page, "width", 612.0) or 612.0
                height = getattr(page, "height", 792.0) or 792.0
                page_dims[pg_no] = (width, height)

                # Get page image if available
                if hasattr(page, "image") and page.image:
                    if hasattr(page.image, "pil_image") and page.image.pil_image:
                        page_images[pg_no] = page.image.pil_image
                    elif isinstance(page.image, Image.Image):
                        page_images[pg_no] = page.image

        # Group elements by page
        elements_by_page: Dict[int, List[Element]] = {}
        text_by_page: Dict[int, str] = {}

        for item_tuple in doc.iterate_items():
            element, _ = item_tuple
            page_no, bbox, elem = self._process_docling_element(element, page_dims, page_images)

            if elem is not None:
                if page_no not in elements_by_page:
                    elements_by_page[page_no] = []
                elements_by_page[page_no].append(elem)

                # Track text for classification
                if elem.type == ElementType.TEXT and elem.content:
                    if page_no not in text_by_page:
                        text_by_page[page_no] = ""
                    text_by_page[page_no] += elem.content + " "

        # Create UniversalPages
        pages: List[UniversalPage] = []

        # Use PyMuPDF to get total page count and render images
        with fitz.open(str(file_path)) as pdf_doc:
            total_pages = len(pdf_doc)

            for page_no in range(1, total_pages + 1):
                elements = elements_by_page.get(page_no, [])
                text_content = text_by_page.get(page_no, "")
                dims = page_dims.get(page_no, (612, 792))

                # Classify page
                classification = UniversalPage.classify_page(
                    len(text_content),
                    threshold_digital=DIGITAL_TEXT_THRESHOLD,
                    threshold_scanned=SCANNED_TEXT_THRESHOLD,
                )

                # Render page image for scanned pages
                raw_image = None
                if classification != PageClassification.DIGITAL and self.extract_images:
                    raw_image = self._render_page(pdf_doc, page_no - 1)

                    # If no elements found and page is scanned, create full-page text element
                    if not elements and raw_image is not None:
                        elements = [self._create_fullpage_text_element(page_no, dims, raw_image)]

                page = create_page(
                    page_number=page_no,
                    elements=elements,
                    dimensions=(int(dims[0]), int(dims[1])),
                    classification=classification,
                    raw_image=raw_image,
                )
                pages.append(page)

        # Create document metadata
        metadata = self._extract_metadata(file_path)

        # Create and return UniversalDocument
        return create_document(
            file_path=file_path,
            file_type="pdf",
            pages=pages,
            metadata=metadata,
        )

    def _process_docling_element(
        self,
        element: Any,
        page_dims: Dict[int, Tuple[float, float]],
        page_images: Dict[int, Image.Image],
    ) -> Tuple[int, Optional[BoundingBox], Optional[Element]]:
        """
        Process a single Docling element.

        Returns:
            (page_number, bbox, Element) or (page_number, None, None) if filtered
        """
        # Get element label
        label_obj = getattr(element, "label", None)
        if label_obj is not None:
            label = str(label_obj.value) if hasattr(label_obj, "value") else str(label_obj)
        else:
            label = "text"

        label_lower = label.lower()

        # Map to ElementType
        element_type = LABEL_TYPE_MAP.get(label_lower, ElementType.TEXT)

        # Get text content
        text = getattr(element, "text", "") or ""

        # Get page number and bbox
        page_no = 1
        bbox = None

        if hasattr(element, "prov") and element.prov:
            prov = element.prov[0] if isinstance(element.prov, list) else element.prov

            if hasattr(prov, "page_no") and prov.page_no is not None:
                page_no = prov.page_no
            elif hasattr(prov, "page") and prov.page is not None:
                page_no = prov.page

            if hasattr(prov, "bbox") and prov.bbox:
                bbox_obj = prov.bbox
                page_w, page_h = page_dims.get(page_no, (612.0, 792.0))

                if hasattr(bbox_obj, "l"):
                    raw_bbox = [
                        float(bbox_obj.l),
                        float(bbox_obj.t),
                        float(bbox_obj.r),
                        float(bbox_obj.b),
                    ]
                    # Ensure valid bbox
                    if raw_bbox[2] > raw_bbox[0] and raw_bbox[3] > raw_bbox[1]:
                        bbox = BoundingBox.from_raw(raw_bbox, page_w, page_h, apply_padding=True)

        # Determine confidence based on text quality
        confidence = 0.9  # Default high confidence for digital
        if element_type == ElementType.TEXT:
            if not text or len(text.strip()) < 5:
                confidence = 0.3  # Low confidence for sparse text
            elif len(text) < 20:
                confidence = 0.6  # Medium confidence

        # Extract raw image for IMAGE elements
        raw_image = None
        if element_type == ElementType.IMAGE and self.extract_images:
            raw_image = self._extract_element_image(element, page_images, page_no)

            # Filter small images
            if raw_image is not None:
                h, w = raw_image.shape[:2]
                if w < MIN_IMAGE_WIDTH or h < MIN_IMAGE_HEIGHT:
                    logger.debug(f"Filtered small image: {w}x{h}")
                    return (page_no, None, None)

        # Create Element
        elem = Element(
            type=element_type,
            content=text,
            bbox=bbox,
            confidence=confidence,
            raw_image=raw_image,
            extraction_method=ExtractionMethod.NATIVE,
            source_label=label,
        )

        return (page_no, bbox, elem)

    def _extract_element_image(
        self,
        element: Any,
        page_images: Dict[int, Image.Image],
        page_no: int,
    ) -> Optional[np.ndarray]:
        """Extract image data from element."""
        try:
            # Try direct image access
            if hasattr(element, "image") and element.image:
                img_data = element.image
                if hasattr(img_data, "pil_image") and img_data.pil_image is not None:
                    return np.array(img_data.pil_image.convert("RGB"))
                elif isinstance(img_data, bytes):
                    pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")
                    return np.array(pil_img)
                elif isinstance(img_data, Image.Image):
                    return np.array(img_data.convert("RGB"))

            # Fallback: crop from page image (not implemented here for efficiency)
            return None

        except Exception as e:
            logger.warning(f"Failed to extract element image: {e}")
            return None

    def _create_fullpage_text_element(
        self,
        page_no: int,
        dims: Tuple[float, float],
        raw_image: np.ndarray,
    ) -> Element:
        """Create a full-page TEXT element for scanned pages."""
        return Element(
            type=ElementType.TEXT,
            content="",  # Will be filled by OCR
            bbox=BoundingBox(x_min=0, y_min=0, x_max=1000, y_max=1000),
            confidence=0.1,  # Low confidence = needs OCR
            raw_image=raw_image,
            extraction_method=ExtractionMethod.NATIVE,
            source_label="scanned_page",
            metadata={"requires_ocr": True, "page_number": page_no},
        )

    def _convert_with_pymupdf(self, file_path: Path) -> UniversalDocument:
        """
        Fallback conversion using PyMuPDF only.

        Used when Docling is not available. Provides basic text extraction
        but no layout analysis.
        """
        logger.info(f"Using PyMuPDF fallback for: {file_path.name}")

        pages: List[UniversalPage] = []

        with fitz.open(str(file_path)) as pdf_doc:
            for page_idx in range(len(pdf_doc)):
                page = pdf_doc.load_page(page_idx)
                page_no = page_idx + 1

                # Extract text
                text = page.get_text("text") or ""

                # Classify page
                classification = UniversalPage.classify_page(
                    len(text),
                    threshold_digital=DIGITAL_TEXT_THRESHOLD,
                    threshold_scanned=SCANNED_TEXT_THRESHOLD,
                )

                # Create elements
                elements: List[Element] = []

                if text.strip():
                    # Create text element
                    elem = Element(
                        type=ElementType.TEXT,
                        content=text.strip(),
                        bbox=BoundingBox(x_min=0, y_min=0, x_max=1000, y_max=1000),
                        confidence=0.8 if classification == PageClassification.DIGITAL else 0.4,
                        extraction_method=ExtractionMethod.NATIVE,
                        source_label="pymupdf_text",
                    )
                    elements.append(elem)

                # Render page for scanned content
                raw_image = None
                if classification != PageClassification.DIGITAL and self.extract_images:
                    raw_image = self._render_page(pdf_doc, page_idx)

                    if not elements:
                        elements.append(
                            self._create_fullpage_text_element(
                                page_no, (page.rect.width, page.rect.height), raw_image
                            )
                        )

                # Extract images
                if self.extract_images:
                    for img in page.get_images():
                        try:
                            img_elem = self._extract_pymupdf_image(pdf_doc, img, page_no)
                            if img_elem:
                                elements.append(img_elem)
                        except Exception as e:
                            logger.warning(f"Failed to extract image: {e}")

                # Create page
                page_obj = create_page(
                    page_number=page_no,
                    elements=elements,
                    dimensions=(int(page.rect.width), int(page.rect.height)),
                    classification=classification,
                    raw_image=raw_image,
                )
                pages.append(page_obj)

                # Memory cleanup
                gc.collect()

        # Create metadata
        metadata = self._extract_metadata(file_path)

        return create_document(
            file_path=file_path,
            file_type="pdf",
            pages=pages,
            metadata=metadata,
        )

    def _render_page(
        self,
        pdf_doc: fitz.Document,
        page_idx: int,
    ) -> np.ndarray:
        """Render a page to numpy array at configured DPI."""
        page = pdf_doc.load_page(page_idx)

        zoom = self.render_dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Convert to PIL then numpy
        img_data = pix.tobytes("ppm")
        pil_image = Image.open(io.BytesIO(img_data)).convert("RGB")

        return np.array(pil_image)

    def _extract_pymupdf_image(
        self,
        pdf_doc: fitz.Document,
        img_info: tuple,
        page_no: int,
    ) -> Optional[Element]:
        """Extract an image using PyMuPDF."""
        try:
            xref = img_info[0]
            base_image = pdf_doc.extract_image(xref)

            if not base_image:
                return None

            img_bytes = base_image["image"]
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # Check size
            w, h = pil_img.size
            if w < MIN_IMAGE_WIDTH or h < MIN_IMAGE_HEIGHT:
                return None

            return Element(
                type=ElementType.IMAGE,
                content="",  # Will be filled by VLM
                bbox=None,  # PyMuPDF doesn't provide bbox easily
                confidence=0.9,
                raw_image=np.array(pil_img),
                extraction_method=ExtractionMethod.NATIVE,
                source_label="pymupdf_image",
                metadata={"xref": xref, "page_number": page_no},
            )

        except Exception as e:
            logger.debug(f"Image extraction failed: {e}")
            return None

    def _extract_metadata(self, file_path: Path) -> DocumentMetadata:
        """Extract document metadata."""
        metadata = DocumentMetadata()

        try:
            with fitz.open(str(file_path)) as pdf_doc:
                meta = pdf_doc.metadata

                if meta:
                    metadata.title = meta.get("title")
                    metadata.author = meta.get("author")
                    metadata.page_count = len(pdf_doc)

                metadata.file_size_bytes = file_path.stat().st_size
                metadata.has_text_layer = any(
                    pdf_doc.load_page(i).get_text() for i in range(min(5, len(pdf_doc)))
                )

        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")

        return metadata
