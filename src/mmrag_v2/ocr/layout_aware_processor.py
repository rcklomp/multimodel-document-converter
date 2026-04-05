"""
Layout-Aware OCR Processor (Phase 1B)
=====================================

Integrates Docling layout detection with Enhanced OCR cascade to produce
clean TEXT and IMAGE chunks from scanned documents.

Architecture:
    1. Docling detects TEXT/IMAGE/TABLE regions
    2. TEXT regions → Enhanced OCR Engine (3-layer cascade)
    3. IMAGE regions → VLM descriptions (existing pipeline)
    4. TABLE regions → Specialized table OCR

This produces output identical to digital PDFs:
    - modality: "text" chunks with verbatim content
    - modality: "image" chunks with VLM descriptions + asset_ref

Usage:
    processor = LayoutAwareOCRProcessor(
        ocr_confidence_threshold=0.7,
        enable_doctr=True
    )

    chunks = processor.process_page(page_image, page_num=21, doc_id="abc123")

    for chunk in chunks:
        if chunk.modality == "text":
            print(f"TEXT: {chunk.content[:100]}...")
        elif chunk.modality == "image":
            print(f"IMAGE: {chunk.visual_description}")

Author: Claude (Architect)
Date: January 3, 2026
"""

import hashlib
import logging
import gc
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Any

import numpy as np
from PIL import Image as PILImage

# REQ-RENDER: Disable PIL decompression bomb check for large DPI renders
# Combat Airplanes at 300 DPI produces images > 115M pixels
PILImage.MAX_IMAGE_PIXELS = None

from .enhanced_ocr_engine import EnhancedOCREngine, OCRResult, OCRLayer
from .image_preprocessor import ImagePreprocessor
from ..utils.coordinate_normalization import normalize_bbox

logger = logging.getLogger(__name__)


@dataclass
class Region:
    """Layout region detected by Docling."""

    type: str  # "text", "image", "table", "picture"
    bbox: List[int]  # [x_min, y_min, x_max, y_max] in pixels
    confidence: float
    text: Optional[str] = None  # For text regions from Docling

    @property
    def area(self) -> int:
        """Calculate region area in pixels."""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


@dataclass
class ProcessedChunk:
    """
    Processed chunk ready for ingestion.

    V3.0.0: Added semantic_context fields per REQ-MM-03.
    """

    chunk_id: str
    modality: str  # "text", "image", "table"
    content: str
    bbox: List[int]  # Normalized 0-1000 per REQ-COORD-01
    page_number: int
    extraction_method: str
    ocr_confidence: Optional[float] = None
    ocr_layer: Optional[str] = None
    visual_description: Optional[str] = None
    asset_ref: Optional[dict] = None
    # REQ-MM-03: Contextual anchoring with prev/next text snippets
    prev_text_snippet: Optional[str] = None
    next_text_snippet: Optional[str] = None


class LayoutAwareOCRProcessor:
    """
    Combines Docling layout detection with Enhanced OCR cascade.

    This processor solves the "OCR garbage" problem by:
    1. Detecting which regions are TEXT vs IMAGE
    2. Running OCR only on TEXT regions
    3. Running VLM only on IMAGE regions
    4. Producing clean modality-separated output

    Args:
        ocr_confidence_threshold: Minimum OCR confidence (0.0-1.0)
        enable_doctr: Whether to enable Doctr Layer 3 (required for scans)
        output_dir: Directory for saving extracted assets
        vlm_manager: Optional VLM manager for image descriptions

    Example:
        >>> processor = LayoutAwareOCRProcessor(
        ...     ocr_confidence_threshold=0.7,
        ...     enable_doctr=True,
        ...     output_dir=Path("./output/assets")
        ... )
        >>> chunks = processor.process_page(page_image, page_num=1, doc_id="abc")
        >>> print(f"Extracted {len(chunks)} chunks")
    """

    def __init__(
        self,
        ocr_confidence_threshold: float = 0.7,
        enable_doctr: bool = True,
        output_dir: Optional[Path] = None,
        vlm_manager: Any = None,
    ):
        self.ocr_threshold = ocr_confidence_threshold
        self.output_dir = Path(output_dir) if output_dir else Path("./output/assets")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize OCR engine
        self.ocr_engine = EnhancedOCREngine(
            confidence_threshold=ocr_confidence_threshold,
            enable_doctr=enable_doctr,
        )

        # VLM manager for page transcription (scanned docs)
        self.vlm_manager = vlm_manager

        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor()

        # VLM manager for image descriptions (optional)
        self.vlm_manager = vlm_manager

        # V3.0.0: Text buffer for semantic context (REQ-MM-03)
        self._text_buffer: List[str] = []

        logger.info(
            f"[LAYOUT-OCR] Initialized with OCR threshold={ocr_confidence_threshold:.2f}, "
            f"Doctr={'enabled' if enable_doctr else 'disabled'}"
        )

    def process_page(
        self,
        page_image: np.ndarray,
        page_number: int,
        doc_id: str,
        docling_elements: Optional[List] = None,
        render_dpi: int = 150,
    ) -> List[ProcessedChunk]:
        """
        Process a scanned page through layout-aware OCR pipeline.

        V3.0.0: Added render_dpi parameter for coordinate conversion.

        Args:
            page_image: RGB numpy array of full page
            page_number: Page number (1-indexed)
            doc_id: Document identifier hash
            docling_elements: Optional pre-extracted Docling elements
            render_dpi: DPI used to render the page image (for coordinate conversion)

        Returns:
            List of ProcessedChunk objects (TEXT, IMAGE, TABLE)

        Raises:
            ValueError: If page_image is invalid
        """
        if page_image is None or page_image.size == 0:
            raise ValueError("Invalid page_image: empty or None")

        logger.info(
            f"[LAYOUT-OCR] Processing page {page_number} ({page_image.shape}) at {render_dpi} DPI"
        )

        # Step 1: Detect layout regions using Docling
        if docling_elements:
            regions = self._convert_docling_elements(docling_elements, page_image.shape, render_dpi)
        else:
            regions = self._detect_regions_fallback(page_image, page_number)

        text_count = sum(1 for r in regions if r.type == "text")
        image_count = sum(1 for r in regions if r.type in ("image", "picture"))
        table_count = sum(1 for r in regions if r.type == "table")

        logger.info(
            f"[LAYOUT-OCR] Detected {len(regions)} regions: "
            f"{text_count} text, {image_count} image, {table_count} table"
        )

        # Step 1a: VLM page transcription for scanned pages.
        # VLMs read text FAR better than OCR on degraded scans because they
        # understand language and context. "Jamu la Frizgi" becomes "J.B. Wood".
        # Use VLM transcription as primary text source when available.
        if self.vlm_manager and text_count > 0 and image_count <= 2:
            vlm_text = self._vlm_transcribe_page(page_image, page_number)
            if vlm_text and len(vlm_text.strip()) > 50:
                logger.info(
                    f"[LAYOUT-OCR] Page {page_number}: VLM transcription "
                    f"({len(vlm_text)} chars) — using instead of OCR"
                )
                page_h, page_w = page_image.shape[:2]
                chunks = []
                # Create text chunk from VLM transcription
                chunk_id = self._generate_chunk_id(doc_id, page_number, "text", 0)
                chunks.append(ProcessedChunk(
                    chunk_id=chunk_id,
                    modality="text",
                    content=vlm_text.strip(),
                    bbox=self._normalize_bbox([0, 0, page_w, page_h], page_w, page_h),
                    page_number=page_number,
                    extraction_method="vlm_transcription",
                    ocr_confidence=0.95,
                    ocr_layer="vlm",
                ))
                # Still process image regions for visual descriptions
                for i, region in enumerate(regions):
                    if region.type in ("image", "picture"):
                        img_chunk = self._process_image_region(
                            page_image, region, page_number, doc_id, i
                        )
                        if img_chunk:
                            img_chunk.bbox = self._normalize_bbox(
                                img_chunk.bbox, page_w, page_h
                            )
                            chunks.append(img_chunk)
                return chunks

        # Step 1b: Full-page Tesseract baseline for scanned pages.
        # Region-by-region OCR loses context on degraded scans with multi-column
        # layouts. A single Tesseract pass with auto page segmentation (PSM 3)
        # handles columns correctly and produces a quality baseline we can compare
        # region OCR against.
        _fullpage_text: Optional[str] = None
        try:
            import pytesseract
            from PIL import Image as _PILImage
            _pil = _PILImage.fromarray(page_image)
            _fullpage_text = pytesseract.image_to_string(_pil)
            if _fullpage_text:
                logger.debug(
                    f"[LAYOUT-OCR] Full-page Tesseract baseline: {len(_fullpage_text)} chars"
                )
        except Exception as _e:
            logger.debug(f"[LAYOUT-OCR] Full-page Tesseract baseline unavailable: {_e}")

        # Make full-page baseline available to region processors
        self._fullpage_baseline = _fullpage_text

        # Optimisation: if all regions are text (no images/tables) and the full-page
        # Tesseract produced good output, use it directly instead of running
        # region-by-region OCR. Full-page Tesseract handles multi-column layouts
        # far better than cropping individual regions on degraded scans.
        if (
            _fullpage_text
            and len(_fullpage_text.strip()) > 50
            and image_count == 0
            and table_count == 0
            and text_count > 0
        ):
            logger.info(
                f"[LAYOUT-OCR] Page {page_number}: text-only page — using full-page "
                f"Tesseract ({len(_fullpage_text)} chars) instead of {text_count} region crops"
            )
            page_h, page_w = page_image.shape[:2]
            chunk_id = self._generate_chunk_id(doc_id, page_number, "text", 0)
            full_chunk = ProcessedChunk(
                chunk_id=chunk_id,
                modality="text",
                content=_fullpage_text.strip(),
                bbox=self._normalize_bbox([0, 0, page_w, page_h], page_w, page_h),
                page_number=page_number,
                extraction_method="tesseract_fullpage",
                ocr_confidence=0.7,
                ocr_layer="tesseract",
            )
            return [full_chunk]

        # Step 2: Process each region based on type
        # V3.0.0: Track text chunks for semantic context and pending IMAGE chunks
        chunks = []
        pending_image_chunks: List[ProcessedChunk] = []
        page_h, page_w = page_image.shape[:2]

        for i, region in enumerate(regions):
            try:
                chunk = None

                if region.type == "text":
                    chunk = self._process_text_region(page_image, region, page_number, doc_id, i)
                    if chunk:
                        # Normalize bbox to 0-1000 scale (REQ-COORD-01)
                        chunk.bbox = self._normalize_bbox(chunk.bbox, page_w, page_h)

                        # Fill prev_text from buffer
                        if self._text_buffer:
                            chunk.prev_text_snippet = " ".join(self._text_buffer[-3:])[-300:]

                        # Flush pending IMAGE chunks with this text as next_text
                        for pending in pending_image_chunks:
                            pending.next_text_snippet = chunk.content[:300]
                            chunks.append(pending)
                        pending_image_chunks.clear()

                        # Add to output and update buffer
                        chunks.append(chunk)
                        self._text_buffer.append(chunk.content)
                        if len(self._text_buffer) > 10:
                            self._text_buffer.pop(0)

                elif region.type in ("image", "picture"):
                    chunk = self._process_image_region(page_image, region, page_number, doc_id, i)
                    if chunk:
                        # Normalize bbox to 0-1000 scale (REQ-COORD-01)
                        chunk.bbox = self._normalize_bbox(chunk.bbox, page_w, page_h)

                        # Fill prev_text from buffer
                        if self._text_buffer:
                            chunk.prev_text_snippet = " ".join(self._text_buffer[-3:])[-300:]

                        # Hold IMAGE chunk until next TEXT arrives (pending context queue)
                        pending_image_chunks.append(chunk)
                        logger.debug(f"[PENDING-CONTEXT] Holding IMAGE chunk for next_text_snippet")

                elif region.type == "table":
                    chunk = self._process_table_region(page_image, region, page_number, doc_id, i)
                    if chunk:
                        # Normalize bbox to 0-1000 scale (REQ-COORD-01)
                        chunk.bbox = self._normalize_bbox(chunk.bbox, page_w, page_h)
                        chunks.append(chunk)
                else:
                    logger.warning(f"[LAYOUT-OCR] Unknown region type: {region.type}")
                    continue

            except Exception as e:
                logger.error(f"[LAYOUT-OCR] Failed to process {region.type} region {i}: {e}")
                continue

        # Flush remaining pending IMAGE chunks (no following text)
        for pending in pending_image_chunks:
            logger.debug(f"[PENDING-CONTEXT] Flushing IMAGE chunk without next_text")
            chunks.append(pending)

        logger.info(f"[LAYOUT-OCR] Page {page_number}: Generated {len(chunks)} chunks")
        return chunks

    def _normalize_bbox(
        self,
        bbox_pixels: List[int],
        page_width_px: int,
        page_height_px: int,
    ) -> List[int]:
        """
        Normalize pixel bbox to 0-1000 integer scale per REQ-COORD-01.

        Args:
            bbox_pixels: [x_min, y_min, x_max, y_max] in pixels
            page_width_px: Page width in pixels
            page_height_px: Page height in pixels

        Returns:
            Normalized [x_min, y_min, x_max, y_max] in 0-1000 range
        """
        if page_width_px == 0 or page_height_px == 0:
            return [0, 0, 1000, 1000]

        x_min = int((bbox_pixels[0] / page_width_px) * 1000)
        y_min = int((bbox_pixels[1] / page_height_px) * 1000)
        x_max = int((bbox_pixels[2] / page_width_px) * 1000)
        y_max = int((bbox_pixels[3] / page_height_px) * 1000)

        # Clamp to 0-1000 range
        x_min = max(0, min(x_min, 1000))
        y_min = max(0, min(y_min, 1000))
        x_max = max(0, min(x_max, 1000))
        y_max = max(0, min(y_max, 1000))

        return [x_min, y_min, x_max, y_max]

    def _convert_docling_elements(
        self,
        elements: List,
        page_shape: Tuple[int, int, int],
        render_dpi: int = 150,
    ) -> List[Region]:
        """
        Convert Docling document elements to Region objects.

        V3.0.0: Added render_dpi parameter for coordinate conversion.

        Args:
            elements: List of Docling elements for this page
            page_shape: (height, width, channels) of page image
            render_dpi: DPI used to render the page image (default 150)

        Returns:
            List of Region objects
        """
        regions = []
        page_height, page_width = page_shape[:2]

        for elem in elements:
            try:
                # Get element type
                label_obj = getattr(elem, "label", None)
                if label_obj is not None:
                    label = str(label_obj.value) if hasattr(label_obj, "value") else str(label_obj)
                else:
                    label = "text"

                # Map Docling labels to our types
                type_map = {
                    "text": "text",
                    "paragraph": "text",
                    "section_header": "text",
                    "title": "text",
                    "list_item": "text",
                    "caption": "text",
                    "picture": "image",
                    "figure": "image",
                    "table": "table",
                }
                region_type = type_map.get(label.lower(), "text")

                # Get bounding box from Docling
                # Docling provides coordinates in PDF points (72 DPI) with BOTTOM-LEFT origin
                # We must scale AND flip Y-axis to match rendered image (TOP-LEFT origin)
                bbox = [0, 0, page_width, page_height]  # Default to full page
                if hasattr(elem, "prov") and elem.prov:
                    prov = elem.prov[0] if isinstance(elem.prov, list) else elem.prov
                    if hasattr(prov, "bbox") and prov.bbox:
                        bbox_obj = prov.bbox
                        if hasattr(bbox_obj, "l"):
                            # CRITICAL FIX: Docling bbox uses PDF coordinate system
                            # 1. Scale from PDF points (72 DPI) to rendered pixels
                            # 2. Flip Y-axis: PDF origin is BOTTOM-LEFT, numpy is TOP-LEFT
                            scale_factor = render_dpi / 72.0

                            # Get PDF page height to calculate Y-flip
                            # PDF height in points = page_height / scale_factor
                            pdf_height = page_height / scale_factor

                            # Scale and flip Y coordinates
                            # X coords are simple: just scale
                            x_left = int(bbox_obj.l * scale_factor)
                            x_right = int(bbox_obj.r * scale_factor)

                            # Y coords need flip: numpy_y = pdf_height - pdf_y
                            # In PDF: .t (top) has HIGHER y-value than .b (bottom)
                            # In numpy: top has LOWER y-value than bottom
                            # Therefore: numpy_top comes from PDF .t, numpy_bottom from PDF .b
                            y_top = int((pdf_height - bbox_obj.t) * scale_factor)
                            y_bottom = int((pdf_height - bbox_obj.b) * scale_factor)

                            # Safety check: ensure valid bbox
                            if y_bottom <= y_top:
                                logger.warning(
                                    f"[BBOX-ERROR] Invalid Y coords after flip: "
                                    f"PDF t={bbox_obj.t:.1f} b={bbox_obj.b:.1f} → "
                                    f"numpy top={y_top} bottom={y_bottom}"
                                )
                                # Skip this element
                                continue

                            bbox = [x_left, y_top, x_right, y_bottom]

                # Get text if available
                text = str(getattr(elem, "text", "")) if hasattr(elem, "text") else None

                # Calculate real confidence based on extracted text quality
                confidence = self._calculate_text_confidence(text, region_type)

                regions.append(
                    Region(
                        type=region_type,
                        bbox=bbox,
                        confidence=confidence,
                        text=text,
                    )
                )

            except Exception as e:
                logger.warning(f"[LAYOUT-OCR] Failed to convert Docling element: {e}")
                continue

        return regions

    def _detect_regions_fallback(
        self,
        page_image: np.ndarray,
        page_number: int,
    ) -> List[Region]:
        """
        Fallback region detection when Docling elements aren't available.

        Uses contour analysis to detect image regions vs text regions.
        This allows extraction of standalone images from scanned pages.

        Args:
            page_image: RGB numpy array
            page_number: Page number for logging

        Returns:
            List of Region objects
        """
        logger.info("[LAYOUT-OCR] Using enhanced fallback region detection with contour analysis.")

        h, w = page_image.shape[:2]
        regions: List[Region] = []

        try:
            import cv2  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            logger.warning(
                f"[LAYOUT-OCR] cv2 unavailable; fallback region detection degraded ({e}). "
                "Treating full page as TEXT."
            )
            return [
                Region(
                    type="text",
                    bbox=[0, 0, w, h],
                    confidence=0.5,
                    text=None,
                )
            ]

        # Convert to grayscale for analysis
        gray = cv2.cvtColor(page_image, cv2.COLOR_RGB2GRAY)

        # Detect potential image regions using edge detection
        # Images typically have sharp edges while text has fine detail
        edges = cv2.Canny(gray, 50, 150)

        # Dilate to connect nearby edges (creates blobs for image regions)
        kernel = np.ones((20, 20), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Find contours (potential image regions)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Minimum size for image region (at least 5% of page area)
        min_area = (h * w) * 0.05
        # Maximum size for image region (at most 80% of page - not full page)
        max_area = (h * w) * 0.80

        image_regions_found = 0

        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch

            # Check if this could be an image region
            if min_area < area < max_area:
                # Check aspect ratio (images usually not too extreme)
                aspect = max(cw, ch) / min(cw, ch) if min(cw, ch) > 0 else 100

                if aspect < 5:  # Not too elongated (like a line of text)
                    # Check if region has enough edge density (images have clear edges)
                    region_edges = edges[y : y + ch, x : x + cw]
                    edge_density = np.sum(region_edges > 0) / area if area > 0 else 0

                    # Images typically have edge density between 5% and 40%
                    if 0.03 < edge_density < 0.50:
                        regions.append(
                            Region(
                                type="image",
                                bbox=[x, y, x + cw, y + ch],
                                confidence=0.7,
                                text=None,
                            )
                        )
                        image_regions_found += 1
                        logger.debug(
                            f"[LAYOUT-OCR] Found image region: {x},{y} {cw}x{ch} "
                            f"(area={area:.0f}, aspect={aspect:.2f}, edges={edge_density:.2%})"
                        )

        # Always add full page as text region (OCR will extract whatever text is there)
        # This ensures we don't miss any text content
        regions.append(
            Region(
                type="text",
                bbox=[0, 0, w, h],
                confidence=0.8,
                text=None,  # Will be OCR'd
            )
        )

        logger.info(
            f"[LAYOUT-OCR] Fallback detection: {image_regions_found} image regions + 1 text region"
        )

        return regions

    def _process_text_region(
        self,
        page_image: np.ndarray,
        region: Region,
        page_number: int,
        doc_id: str,
        region_idx: int,
    ) -> Optional[ProcessedChunk]:
        """
        Process a text region through OCR cascade.

        Args:
            page_image: Full page image
            region: Text region
            page_number: Page number
            doc_id: Document ID
            region_idx: Region index on page

        Returns:
            ProcessedChunk with modality="text"
        """
        # Crop text region from page with padding
        x1, y1, x2, y2 = region.bbox
        h, w = page_image.shape[:2]

        # Apply 10px padding (REQ-MM-01) with bounds checking
        padding = 10
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        text_crop = page_image[y1:y2, x1:x2]

        if text_crop.size == 0:
            logger.warning(f"[LAYOUT-OCR] Empty text crop at {region.bbox}")
            return None

        # Prepare Docling result if text already extracted
        docling_result = None
        if region.text and len(region.text.strip()) > 0:
            if self._is_flat_code_like_text(region.text):
                logger.debug(
                    f"[LAYOUT-OCR] Text region {region_idx}: "
                    "bypassing Docling fast-path for flattened code-like text"
                )
            else:
                docling_result = OCRResult(
                    text=region.text,
                    confidence=region.confidence,
                    layer_used=OCRLayer.DOCLING,
                    processing_time_ms=0,
                )

        # Run OCR cascade
        ocr_result = self.ocr_engine.process_page(
            page_image=text_crop,
            docling_result=docling_result,
        )

        # Generate chunk ID
        chunk_id = self._generate_chunk_id(doc_id, page_number, "text", region_idx)

        logger.debug(
            f"[LAYOUT-OCR] Text region {region_idx}: "
            f"{ocr_result.layer_used.value} (conf={ocr_result.confidence:.2f}), "
            f"{len(ocr_result.text)} chars"
        )

        # Quality check: if region OCR produced garbled text but the full-page
        # Tesseract baseline has cleaner content for this region, prefer the baseline.
        region_text = ocr_result.text
        if (
            self._fullpage_baseline
            and region_text
            and len(region_text) >= 20
        ):
            region_text = self._improve_from_fullpage_baseline(
                region_text, self._fullpage_baseline
            )

        return ProcessedChunk(
            chunk_id=chunk_id,
            modality="text",
            content=region_text,
            bbox=region.bbox,
            page_number=page_number,
            extraction_method="layout_aware_ocr",
            ocr_confidence=ocr_result.confidence,
            ocr_layer=ocr_result.layer_used.value,
        )

    def _is_flat_code_like_text(self, text: Optional[str]) -> bool:
        """
        Detect flattened code-like Docling text that should bypass Layer-1 acceptance.

        We only bypass when text is single-line and has strong code signals, so prose
        keeps fast Docling behavior.
        """
        if not text:
            return False
        t = text.strip()
        if not t or "\n" in t or len(t) < 24:
            return False
        if ">>>" in t or "..." in t:
            return True
        has_keywords = re.search(
            r"\b(def|class|import|from|return|yield|lambda|try|except|finally|with|for|while|if|elif|else)\b",
            t,
        )
        has_symbols = re.search(r"[{}()\[\];=:@]", t)
        looks_signature = re.search(r"\b(class|def)\s+[A-Za-z_]\w*\s*[:(]", t)
        return bool((has_keywords and has_symbols) or looks_signature)

    def _process_image_region(
        self,
        page_image: np.ndarray,
        region: Region,
        page_number: int,
        doc_id: str,
        region_idx: int,
    ) -> Optional[ProcessedChunk]:
        """
        Process an image region through VLM pipeline.

        Args:
            page_image: Full page image
            region: Image region
            page_number: Page number
            doc_id: Document ID
            region_idx: Region index on page

        Returns:
            ProcessedChunk with modality="image"
        """
        # Crop image region with padding
        x1, y1, x2, y2 = region.bbox
        h, w = page_image.shape[:2]

        padding = 10
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        image_crop = page_image[y1:y2, x1:x2]

        if image_crop.size == 0:
            logger.warning(f"[LAYOUT-OCR] Empty image crop at {region.bbox}")
            return None

        # If the crop is too tight (ink touches edges), expand slightly to avoid
        # cutting off diagram lines / labels. This is common when bboxes are a
        # bit under-estimated on digital manuals.
        try:
            from ..utils.image_trim import edge_ink_fractions

            # Iteratively expand up to a small cap.
            step = 20
            max_extra = 80
            expanded = 0
            cur_x1, cur_y1, cur_x2, cur_y2 = x1, y1, x2, y2
            while expanded < max_extra:
                pil_probe = PILImage.fromarray(page_image[cur_y1:cur_y2, cur_x1:cur_x2])
                fr = edge_ink_fractions(pil_probe)
                if max(fr.values()) <= 0.02:
                    break
                if fr["left"] > 0.02:
                    cur_x1 = max(0, cur_x1 - step)
                if fr["right"] > 0.02:
                    cur_x2 = min(w, cur_x2 + step)
                if fr["top"] > 0.02:
                    cur_y1 = max(0, cur_y1 - step)
                if fr["bottom"] > 0.02:
                    cur_y2 = min(h, cur_y2 + step)
                expanded += step
            if (cur_x1, cur_y1, cur_x2, cur_y2) != (x1, y1, x2, y2):
                logger.debug(
                    f"[LAYOUT-OCR][CROP-EXPAND] Page {page_number} region {region_idx}: "
                    f"expanded crop ({x1},{y1},{x2},{y2}) -> ({cur_x1},{cur_y1},{cur_x2},{cur_y2})"
                )
                x1, y1, x2, y2 = cur_x1, cur_y1, cur_x2, cur_y2
                image_crop = page_image[y1:y2, x1:x2]
        except Exception as expand_err:
            logger.debug(f"[LAYOUT-OCR][CROP-EXPAND] Skipped due to error: {expand_err}")

        # Save asset to disk
        asset_filename = f"{doc_id}_{page_number:03d}_figure_{region_idx:02d}.png"
        asset_path = self.output_dir / asset_filename

        # Prefer PIL for PNG writes (keeps cv2 optional).
        # Also trim large white margins from page-render crops to produce cleaner assets
        # (common in digital PDFs with vector diagrams rendered to raster).
        pil_image = PILImage.fromarray(image_crop)
        try:
            from ..utils.image_trim import trim_white_margins

            trim_res = trim_white_margins(pil_image)
            if trim_res.trimmed:
                pil_image = trim_res.image
                logger.debug(
                    f"[LAYOUT-OCR][TRIM] Page {page_number} region {region_idx}: "
                    f"trimmed white margins (bbox={trim_res.bbox}, size={pil_image.size})"
                )
        except Exception as trim_err:
            # Trimming is a quality enhancement; never fail extraction due to trimming errors.
            logger.debug(f"[LAYOUT-OCR][TRIM] Skipped due to error: {trim_err}")

        pil_image.save(str(asset_path))

        # Get VLM description if manager available
        visual_description = ""
        if self.vlm_manager:
            try:
                # Use the same PIL image we saved (post-trim) for VLM enrichment.

                # Check for different VLM manager method signatures
                if hasattr(self.vlm_manager, "enrich_image"):
                    # Use the standard enrich_image method
                    from ..state.context_state import create_context_state

                    temp_state = create_context_state(
                        doc_id=doc_id,
                        source_file=f"page_{page_number}",
                    )
                    temp_state.update_page(page_number)

                    visual_description = self.vlm_manager.enrich_image(
                        image=pil_image,
                        state=temp_state,
                        page_number=page_number,
                        anchor_text=None,
                    )
                elif hasattr(self.vlm_manager, "describe_image"):
                    visual_description = self.vlm_manager.describe_image(
                        pil_image,
                        context=f"Page {page_number} image region",
                    )
                else:
                    visual_description = f"Image region from page {page_number}"
            except Exception as e:
                logger.error(f"[LAYOUT-OCR] VLM failed: {e}")
                visual_description = f"Image region from page {page_number} ({image_crop.shape[1]}x{image_crop.shape[0]}px)"
        else:
            visual_description = f"Image region from page {page_number} ({image_crop.shape[1]}x{image_crop.shape[0]}px)"

        # Generate chunk ID
        chunk_id = self._generate_chunk_id(doc_id, page_number, "image", region_idx)

        logger.debug(
            f"[LAYOUT-OCR] Image region {region_idx}: "
            f"{image_crop.shape}, saved to {asset_filename}"
        )

        return ProcessedChunk(
            chunk_id=chunk_id,
            modality="image",
            content=visual_description,
            bbox=region.bbox,
            page_number=page_number,
            extraction_method="layout_aware_vlm",
            visual_description=visual_description,
            asset_ref={
                "file_path": f"assets/{asset_filename}",
                "mime_type": "image/png",
                "width_px": pil_image.size[0],
                "height_px": pil_image.size[1],
            },
        )

    def _process_table_region(
        self,
        page_image: np.ndarray,
        region: Region,
        page_number: int,
        doc_id: str,
        region_idx: int,
    ) -> Optional[ProcessedChunk]:
        """
        Process a table region.

        CRITICAL FIX: QA-CHECK-05 requires tables to have asset_ref.
        This method now saves the table crop to disk and includes asset_ref.

        Args:
            page_image: Full page image
            region: Table region
            page_number: Page number
            doc_id: Document ID
            region_idx: Region index on page

        Returns:
            ProcessedChunk with modality="table" including asset_ref
        """
        # Crop table region with padding (REQ-MM-01: 10px padding)
        x1, y1, x2, y2 = region.bbox
        h, w = page_image.shape[:2]

        padding = 10
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        table_crop = page_image[y1:y2, x1:x2]

        if table_crop.size == 0:
            logger.warning(f"[LAYOUT-OCR] Empty table crop at {region.bbox}")
            return None

        # CRITICAL FIX: Save table asset to disk (QA-CHECK-05 compliance)
        # REQ-MM-02: Asset naming [DocHash]_[Page]_[Type]_[Index].png
        asset_filename = f"{doc_id}_{page_number:03d}_table_{region_idx:02d}.png"
        asset_path = self.output_dir / asset_filename

        # Prefer PIL for PNG writes (keeps cv2 optional).
        try:
            table_pil = PILImage.fromarray(table_crop)
            # Apply the same conservative trim used for figures: many Docling bboxes
            # include large white margins around tables in digital PDFs.
            try:
                from ..utils.image_trim import trim_white_margins

                trim_res = trim_white_margins(table_pil)
                if trim_res.trimmed:
                    table_pil = trim_res.image
                    logger.debug(
                        f"[LAYOUT-OCR][TRIM] Page {page_number} table {region_idx}: "
                        f"trimmed white margins (bbox={trim_res.bbox}, size={table_pil.size})"
                    )
            except Exception as trim_err:
                logger.debug(f"[LAYOUT-OCR][TRIM] Skipped (table) due to error: {trim_err}")

            table_pil.save(str(asset_path))
            logger.debug(f"[LAYOUT-OCR] Saved table asset: {asset_filename}")
        except Exception as e:
            logger.error(f"[LAYOUT-OCR] Failed to save table asset {asset_filename}: {e}")
            # Continue without asset_ref - validation will catch this

        # Run OCR on table to extract text content
        # Use the raw numpy crop for OCR (keeps existing behavior). Trimming is for
        # asset cleanliness and doesn't affect the extraction bbox.
        ocr_result = self.ocr_engine.process_page(table_crop)

        # Generate chunk ID
        chunk_id = self._generate_chunk_id(doc_id, page_number, "table", region_idx)

        logger.debug(
            f"[LAYOUT-OCR] Table region {region_idx}: "
            f"{ocr_result.layer_used.value} (conf={ocr_result.confidence:.2f}), "
            f"saved to {asset_filename}"
        )

        # Convert raw OCR text to markdown table format.
        # Without this, table content is dumped as a single garbled string.
        table_text = self._ocr_text_to_markdown_table(ocr_result.text)

        # Return chunk with asset_ref (QA-CHECK-05 requirement)
        return ProcessedChunk(
            chunk_id=chunk_id,
            modality="table",
            content=table_text,
            bbox=region.bbox,
            page_number=page_number,
            extraction_method="layout_aware_ocr",
            ocr_confidence=ocr_result.confidence,
            ocr_layer=ocr_result.layer_used.value,
            asset_ref={
                "file_path": f"assets/{asset_filename}",
                "mime_type": "image/png",
                # Report the saved asset dimensions (post-trim if applied).
                "width_px": int(table_pil.size[0]) if "table_pil" in locals() else table_crop.shape[1],
                "height_px": int(table_pil.size[1]) if "table_pil" in locals() else table_crop.shape[0],
            },
        )

    def _vlm_transcribe_page(self, page_image: np.ndarray, page_number: int) -> Optional[str]:
        """Use the VLM to transcribe text from a scanned page image.

        VLMs understand language and context, producing far superior text
        extraction on degraded scans compared to traditional OCR.
        Bypasses the normal describe_image 2000-token cap — page transcription
        needs 4000+ tokens for dense text pages.
        """
        try:
            import base64
            import io
            import json
            import urllib.request
            from PIL import Image as _PILImage

            pil_img = _PILImage.fromarray(page_image)

            # Get provider config
            provider = self.vlm_manager._provider
            model = getattr(provider, "model", None) or getattr(provider, "_model", "")
            base_url = getattr(provider, "base_url", None) or getattr(provider, "_base_url", "")
            api_key = getattr(provider, "api_key", None) or getattr(provider, "_api_key", "")

            if not base_url or not api_key:
                # Fallback to provider.describe_image if we can't get credentials
                prompt = (
                    "Transcribe ALL text on this scanned page exactly as written. "
                    "Preserve paragraph structure. Output ONLY the transcribed text."
                )
                response = provider.describe_image(pil_img, prompt)
                return response.strip() if response and len(response.strip()) > 20 else None

            # Encode image
            buf = io.BytesIO()
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            pil_img.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()

            # Direct API call with high max_tokens
            data = json.dumps({
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": (
                            "Transcribe ALL text on this scanned page exactly as written. "
                            "Preserve paragraph structure and line breaks between sections. "
                            "Output ONLY the transcribed text, nothing else. "
                            "Do not describe images or diagrams — only transcribe printed text."
                        )}
                    ]
                }],
                "max_tokens": 4096,
                "temperature": 0.0,
            }).encode()

            url = base_url.rstrip("/")
            if not url.endswith("/chat/completions"):
                url += "/chat/completions"

            req = urllib.request.Request(url, data=data)
            req.add_header("Authorization", f"Bearer {api_key}")
            req.add_header("Content-Type", "application/json")
            resp = urllib.request.urlopen(req, timeout=120)
            result = json.loads(resp.read())
            text = result["choices"][0]["message"]["content"]

            if not text or len(text.strip()) < 20:
                return None

            # Detect truncation: if response ends mid-word or mid-sentence,
            # request a continuation. VLM token limits can cut off dense pages.
            import re as _re
            full_text = text.strip()
            max_continuations = 3
            for _ in range(max_continuations):
                ends_clean = bool(_re.search(r"[.!?:;\"')\]}\d]\s*$", full_text.rstrip()))
                if ends_clean:
                    break
                # Request continuation
                try:
                    cont_data = json.dumps({
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                                    {"type": "text", "text": (
                                        "Transcribe ALL text on this scanned page exactly as written. "
                                        "Output ONLY the transcribed text."
                                    )}
                                ]
                            },
                            {
                                "role": "assistant",
                                "content": full_text[-200:]  # Last 200 chars as context
                            },
                            {
                                "role": "user",
                                "content": "Continue transcribing from where you left off. Output ONLY the remaining text."
                            }
                        ],
                        "max_tokens": 2048,
                        "temperature": 0.0,
                    }).encode()
                    cont_req = urllib.request.Request(url, data=cont_data)
                    cont_req.add_header("Authorization", f"Bearer {api_key}")
                    cont_req.add_header("Content-Type", "application/json")
                    cont_resp = urllib.request.urlopen(cont_req, timeout=120)
                    cont_result = json.loads(cont_resp.read())
                    continuation = cont_result["choices"][0]["message"]["content"].strip()
                    if continuation and len(continuation) > 10:
                        full_text = full_text.rstrip() + " " + continuation
                        logger.debug(
                            f"[VLM-TRANSCRIBE] Page {page_number}: "
                            f"continuation added (+{len(continuation)} chars)"
                        )
                    else:
                        break
                except Exception:
                    break

            logger.debug(
                f"[VLM-TRANSCRIBE] Page {page_number}: "
                f"{len(full_text)} chars transcribed"
            )
            return full_text.strip()

        except Exception as e:
            logger.warning(f"[VLM-TRANSCRIBE] Page {page_number} failed: {e}")

        return None

    @staticmethod
    def _ocr_text_to_markdown_table(text: str) -> str:
        """Convert raw OCR text from a table region to markdown table format.

        Detects tab-separated or multi-space-aligned columns and converts to
        pipe-separated markdown. Falls back to a single-column table to
        preserve content for RAG retrieval.
        """
        import re

        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        if not lines:
            return text or ""

        # Already has pipe separators → treat as markdown
        if sum(1 for ln in lines if "|" in ln) >= 2:
            return "\n".join(lines)

        # Detect columns from tabs or multi-space alignment
        parsed_rows: list[list[str]] = []
        for ln in lines:
            cols = [c.strip() for c in re.split(r"\t+|\s{2,}", ln) if c.strip()]
            if len(cols) >= 2:
                parsed_rows.append(cols)

        if parsed_rows:
            max_cols = max(len(r) for r in parsed_rows)
            header = parsed_rows[0] + [""] * (max_cols - len(parsed_rows[0]))
            md = [
                "| " + " | ".join(header) + " |",
                "| " + " | ".join(["---"] * max_cols) + " |",
            ]
            for row in parsed_rows[1:]:
                padded = row + [""] * (max_cols - len(row))
                md.append("| " + " | ".join(padded) + " |")
            return "\n".join(md)

        # Single-column fallback: wrap every line in a table row
        md = ["| Content |", "| --- |"]
        for ln in lines:
            md.append("| " + ln.replace("|", "\\|") + " |")
        return "\n".join(md)

    @staticmethod
    def _improve_from_fullpage_baseline(region_text: str, fullpage_text: str) -> str:
        """Replace garbled region OCR with cleaner full-page Tesseract text.

        The full-page Tesseract pass (PSM 3, auto page segmentation) handles
        multi-column layouts correctly. If the region OCR has low word overlap
        with the full-page text, the region result is likely garbled.

        Strategy: find the best-matching segment in the full-page text for
        this region's content using word overlap. If the overlap is poor but
        a good match exists in the full-page text, substitute it.
        """
        import re

        region_words = set(re.findall(r"[a-zA-Z]{3,}", region_text.lower()))
        if len(region_words) < 3:
            return region_text  # Too short to evaluate

        # Split full-page text into sentences for matching
        fp_sentences = re.split(r"(?<=[.!?])\s+", fullpage_text)

        # Find full-page sentences that share words with this region
        best_overlap = 0.0
        for sent in fp_sentences:
            sent_words = set(re.findall(r"[a-zA-Z]{3,}", sent.lower()))
            if not sent_words:
                continue
            overlap = len(region_words & sent_words) / len(region_words)
            best_overlap = max(best_overlap, overlap)

        # If region text has good overlap with full-page (>50%), it's fine
        if best_overlap > 0.5:
            return region_text

        # Region text is garbled — try to find the matching full-page segment.
        # Use the first and last recognizable words as anchors.
        region_word_list = re.findall(r"[a-zA-Z]{4,}", region_text)
        if len(region_word_list) < 2:
            return region_text

        # Search for the first anchor word in the full-page text
        first_anchor = region_word_list[0].lower()
        fp_lower = fullpage_text.lower()
        anchor_pos = fp_lower.find(first_anchor)
        if anchor_pos < 0:
            return region_text  # Can't find anchor, keep region text

        # Extract a segment of similar length from the full-page text
        target_len = len(region_text)
        segment = fullpage_text[anchor_pos:anchor_pos + int(target_len * 1.5)]

        # Verify the segment is actually better (more real words)
        seg_words = set(re.findall(r"[a-zA-Z]{3,}", segment.lower()))
        seg_dict_score = sum(1 for w in seg_words if len(w) >= 4) / max(len(seg_words), 1)
        reg_dict_score = sum(1 for w in region_words if len(w) >= 4) / max(len(region_words), 1)

        if seg_dict_score > reg_dict_score:
            logger.info(
                f"[LAYOUT-OCR] Replaced garbled region OCR ({len(region_text)} chars, "
                f"overlap={best_overlap:.2f}) with full-page baseline segment"
            )
            return segment.strip()

        return region_text

    def _calculate_text_confidence(self, text: Optional[str], region_type: str) -> float:
        """
        Calculate confidence score based on extracted text quality.

        This heuristic prevents the hardcoded 0.9 bypass and ensures
        the OCR cascade is properly triggered for low-quality extractions.

        Args:
            text: Extracted text from Docling
            region_type: Type of region (text, image, table)

        Returns:
            Confidence score 0.0-1.0
        """
        # IMAGE regions get standard confidence (no text expected)
        if region_type in ("image", "picture"):
            return 0.85

        # Empty or None text = very low confidence (trigger OCR)
        if not text or len(text.strip()) == 0:
            return 0.1  # Force OCR cascade

        text_clean = text.strip()
        text_len = len(text_clean)

        # Very short text (< 5 chars) = suspicious, likely noise
        if text_len < 5:
            return 0.3

        # Check for garbage characters (common in bad OCR)
        garbage_chars = sum(1 for c in text if c in "@#$%^&*()_+=[]{}|\\<>~`")
        garbage_ratio = garbage_chars / text_len if text_len > 0 else 1.0

        if garbage_ratio > 0.3:  # >30% garbage = bad extraction
            return 0.2
        elif garbage_ratio > 0.15:  # >15% garbage = mediocre
            return 0.5

        # Check for excessive whitespace (indicates layout issues)
        space_count = text.count(" ") + text.count("\n") + text.count("\t")
        space_ratio = space_count / text_len if text_len > 0 else 1.0

        if space_ratio > 0.5:  # >50% whitespace = suspicious
            return 0.6

        # Check alphanumeric ratio (real text should have letters/numbers)
        alphanumeric = sum(1 for c in text if c.isalnum())
        alpha_ratio = alphanumeric / text_len if text_len > 0 else 0.0

        if alpha_ratio < 0.3:  # <30% alphanumeric = likely garbage
            return 0.4
        elif alpha_ratio < 0.5:  # <50% alphanumeric = mediocre
            return 0.6

        # Check for repeated characters (OCR artifacts)
        max_repeat = 1
        current_repeat = 1
        prev_char = None
        for c in text:
            if c == prev_char:
                current_repeat += 1
                max_repeat = max(max_repeat, current_repeat)
            else:
                current_repeat = 1
            prev_char = c

        if max_repeat > 10:  # >10 repeated chars = OCR failure
            return 0.3
        elif max_repeat > 5:  # >5 repeated chars = suspicious
            return 0.6

        # Text length based scoring
        # Very long text (>500 chars) with good quality = high confidence
        if text_len > 500:
            return 0.95
        elif text_len > 200:
            return 0.85
        elif text_len > 50:
            return 0.75
        else:
            return 0.65  # Short but valid text

    def _generate_chunk_id(
        self,
        doc_id: str,
        page_number: int,
        modality: str,
        region_idx: int,
    ) -> str:
        """Generate unique chunk ID."""
        parts = f"{doc_id}_{page_number}_{modality}_{region_idx}"
        hash_suffix = hashlib.md5(parts.encode()).hexdigest()[:8]
        return f"{doc_id}_{page_number:03d}_{modality}_{hash_suffix}"

    def cleanup(self) -> None:
        """
        Release OCR/layout runtime references for long-running jobs.
        """
        try:
            if getattr(self, "ocr_engine", None) is not None:
                cleanup = getattr(self.ocr_engine, "cleanup", None)
                if callable(cleanup):
                    cleanup()
        except Exception as e:
            logger.debug(f"[LAYOUT-OCR] ocr_engine cleanup skipped: {e}")
        finally:
            self.ocr_engine = None
            self.vlm_manager = None
            self._text_buffer.clear()
            gc.collect()
