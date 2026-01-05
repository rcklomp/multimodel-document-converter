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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Any

import cv2
import numpy as np
from PIL import Image as PILImage

# REQ-RENDER: Disable PIL decompression bomb check for large DPI renders
# Combat Airplanes at 300 DPI produces images > 115M pixels
PILImage.MAX_IMAGE_PIXELS = None

from .enhanced_ocr_engine import EnhancedOCREngine, OCRResult, OCRLayer
from .image_preprocessor import ImagePreprocessor

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
    """Processed chunk ready for ingestion."""

    chunk_id: str
    modality: str  # "text", "image", "table"
    content: str
    bbox: List[int]
    page_number: int
    extraction_method: str
    ocr_confidence: Optional[float] = None
    ocr_layer: Optional[str] = None
    visual_description: Optional[str] = None
    asset_ref: Optional[dict] = None


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

        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor()

        # VLM manager for image descriptions (optional)
        self.vlm_manager = vlm_manager

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
    ) -> List[ProcessedChunk]:
        """
        Process a scanned page through layout-aware OCR pipeline.

        Args:
            page_image: RGB numpy array of full page
            page_number: Page number (1-indexed)
            doc_id: Document identifier hash
            docling_elements: Optional pre-extracted Docling elements

        Returns:
            List of ProcessedChunk objects (TEXT, IMAGE, TABLE)

        Raises:
            ValueError: If page_image is invalid
        """
        if page_image is None or page_image.size == 0:
            raise ValueError("Invalid page_image: empty or None")

        logger.info(f"[LAYOUT-OCR] Processing page {page_number} ({page_image.shape})")

        # Step 1: Detect layout regions using Docling
        if docling_elements:
            regions = self._convert_docling_elements(docling_elements, page_image.shape)
        else:
            regions = self._detect_regions_fallback(page_image, page_number)

        text_count = sum(1 for r in regions if r.type == "text")
        image_count = sum(1 for r in regions if r.type in ("image", "picture"))
        table_count = sum(1 for r in regions if r.type == "table")

        logger.info(
            f"[LAYOUT-OCR] Detected {len(regions)} regions: "
            f"{text_count} text, {image_count} image, {table_count} table"
        )

        # Step 2: Process each region based on type
        chunks = []

        for i, region in enumerate(regions):
            try:
                chunk = None

                if region.type == "text":
                    chunk = self._process_text_region(page_image, region, page_number, doc_id, i)
                elif region.type in ("image", "picture"):
                    chunk = self._process_image_region(page_image, region, page_number, doc_id, i)
                elif region.type == "table":
                    chunk = self._process_table_region(page_image, region, page_number, doc_id, i)
                else:
                    logger.warning(f"[LAYOUT-OCR] Unknown region type: {region.type}")
                    continue

                if chunk:
                    chunks.append(chunk)

            except Exception as e:
                logger.error(f"[LAYOUT-OCR] Failed to process {region.type} region {i}: {e}")
                continue

        logger.info(f"[LAYOUT-OCR] Page {page_number}: Generated {len(chunks)} chunks")
        return chunks

    def _convert_docling_elements(
        self,
        elements: List,
        page_shape: Tuple[int, int, int],
    ) -> List[Region]:
        """
        Convert Docling document elements to Region objects.

        Args:
            elements: List of Docling elements for this page
            page_shape: (height, width, channels) of page image

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

                # Get bounding box
                bbox = [0, 0, page_width, page_height]  # Default to full page
                if hasattr(elem, "prov") and elem.prov:
                    prov = elem.prov[0] if isinstance(elem.prov, list) else elem.prov
                    if hasattr(prov, "bbox") and prov.bbox:
                        bbox_obj = prov.bbox
                        if hasattr(bbox_obj, "l"):
                            # Convert normalized coords to pixels
                            bbox = [
                                int(bbox_obj.l * page_width),
                                int(bbox_obj.t * page_height),
                                int(bbox_obj.r * page_width),
                                int(bbox_obj.b * page_height),
                            ]

                # Get text if available
                text = str(getattr(elem, "text", "")) if hasattr(elem, "text") else None

                regions.append(
                    Region(
                        type=region_type,
                        bbox=bbox,
                        confidence=0.9,  # Docling doesn't provide confidence per element
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

        return ProcessedChunk(
            chunk_id=chunk_id,
            modality="text",
            content=ocr_result.text,
            bbox=region.bbox,
            page_number=page_number,
            extraction_method="layout_aware_ocr",
            ocr_confidence=ocr_result.confidence,
            ocr_layer=ocr_result.layer_used.value,
        )

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

        # Save asset to disk
        asset_filename = f"{doc_id}_{page_number:03d}_figure_{region_idx:02d}.png"
        asset_path = self.output_dir / asset_filename

        # Convert RGB to BGR for cv2.imwrite
        cv2.imwrite(str(asset_path), cv2.cvtColor(image_crop, cv2.COLOR_RGB2BGR))

        # Get VLM description if manager available
        visual_description = ""
        if self.vlm_manager:
            try:
                # Convert numpy array to PIL Image for VisionManager
                from PIL import Image as PILImageConvert

                pil_image = PILImageConvert.fromarray(image_crop)

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
                "width_px": image_crop.shape[1],
                "height_px": image_crop.shape[0],
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

        For now, treat tables as text regions and run OCR.
        Future: Use specialized table structure detection.

        Args:
            page_image: Full page image
            region: Table region
            page_number: Page number
            doc_id: Document ID
            region_idx: Region index on page

        Returns:
            ProcessedChunk with modality="table"
        """
        # Crop table region with padding
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

        # Run OCR on table
        ocr_result = self.ocr_engine.process_page(table_crop)

        # Generate chunk ID
        chunk_id = self._generate_chunk_id(doc_id, page_number, "table", region_idx)

        logger.debug(
            f"[LAYOUT-OCR] Table region {region_idx}: "
            f"{ocr_result.layer_used.value} (conf={ocr_result.confidence:.2f})"
        )

        return ProcessedChunk(
            chunk_id=chunk_id,
            modality="table",
            content=ocr_result.text,
            bbox=region.bbox,
            page_number=page_number,
            extraction_method="layout_aware_ocr",
            ocr_confidence=ocr_result.confidence,
            ocr_layer=ocr_result.layer_used.value,
        )

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
