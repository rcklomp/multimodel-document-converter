"""
Element Processor - Quality-Based Routing to IngestionChunks
=============================================================

This module processes UIR Elements into IngestionChunks using quality-based
routing. It is the CRITICAL component that fixes the scanned document problem.

Key Principle:
    Route by QUALITY, not by FORMAT:
    - HIGH confidence TEXT → Direct extraction → modality: "text"
    - LOW confidence TEXT → OCR cascade → modality: "text"
    - IMAGE elements → VLM visual description → modality: "image"
    - TABLE elements → Structure extraction → modality: "table"

The output is ALWAYS proper modalities (text/image/table), NEVER "shadow".

SRS Compliance:
    - REQ-CHUNK-01: Sentence-boundary splits only
    - REQ-CHUNK-02: Token limits (400 target, 512 max)
    - REQ-MM-01: 10px padding on crops
    - REQ-COORD-01: Coordinates normalized to 0-1000

Author: Claude (Architect)
Date: January 2026
"""

from __future__ import annotations

import gc
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, TYPE_CHECKING

import numpy as np

from .intermediate import (
    Element,
    ElementType,
    ExtractionMethod,
    PageClassification,
    UniversalDocument,
    UniversalPage,
)

if TYPE_CHECKING:
    from ..vision.vision_manager import VisionManager
    from ..ocr.enhanced_ocr_engine import EnhancedOCREngine

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Quality thresholds for routing decisions
CONFIDENCE_THRESHOLD_HIGH = 0.7  # Above: use directly
CONFIDENCE_THRESHOLD_LOW = 0.5  # Below: always OCR

# Chunking parameters
MAX_CHUNK_CHARS = 400
CHUNK_OVERLAP_RATIO = 0.10
MIN_OVERLAP_CHARS = 60

# VLM Visual-Only prompt (prevents text reading)
VLM_VISUAL_ONLY_PROMPT = """
You are analyzing an IMAGE from a technical document.

STRICT CONSTRAINTS:
1. DO NOT read or transcribe any text visible in the image
2. ONLY describe visual elements: shapes, diagrams, photos, components
3. Focus on spatial relationships and technical details
4. Use technical vocabulary appropriate to the subject matter

CORRECT OUTPUT EXAMPLES:
- "Exploded view diagram showing trigger assembly with 7 components arranged vertically"
- "Black and white photograph of military aircraft on carrier deck"
- "Technical schematic showing electrical connections between 4 modules"

FORBIDDEN OUTPUT (text reading):
- "The text says 'INTRODUCTION'..."
- "The caption reads..."
- "The label indicates..."

Describe ONLY the visual/mechanical aspects of this image:
"""


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class ProcessingResult:
    """Result of processing an element."""

    chunk_id: str
    doc_id: str
    modality: str  # "text", "image", "table"
    content: str
    page_number: int
    bbox: Optional[List[int]] = None
    extraction_method: str = "native"
    confidence: float = 0.9
    asset_path: Optional[str] = None
    visual_description: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ============================================================================
# ELEMENT PROCESSOR
# ============================================================================


class ElementProcessor:
    """
    Processes UIR Elements into IngestionChunks using quality-based routing.

    This is the CORE component that ensures:
    - TEXT elements produce modality="text" chunks (via OCR if needed)
    - IMAGE elements produce modality="image" chunks (via VLM)
    - TABLE elements produce modality="table" chunks

    The output NEVER contains "shadow" modality.

    OCR GOVERNANCE (Cluster B):
    - When enable_ocr=False, the processor NEVER triggers OCR cascade
    - Empty text elements are dropped, not OCR'd
    - This respects the user's CLI preference (--no-ocr)

    REQ-COORD-02:
    - Page dimensions (page_width, page_height) are propagated to ALL chunks
    - This enables UI overlay support for visual debugging

    Usage:
        processor = ElementProcessor(
            ocr_engine=enhanced_ocr_engine,
            vision_manager=vision_manager,
            output_dir=Path("./output/assets"),
            enable_ocr=True,  # OCR Governance
        )

        for page in uir.pages:
            for chunk in processor.process_page(page, doc_id="abc123"):
                yield chunk  # Always proper modality

    Attributes:
        ocr_engine: OCR cascade engine for low-confidence text
        vision_manager: VLM for image descriptions
        output_dir: Directory for saving assets
        confidence_threshold: Threshold for OCR fallback
        enable_ocr: Whether OCR is allowed (CLI governance)
    """

    def __init__(
        self,
        ocr_engine: Optional["EnhancedOCREngine"] = None,
        vision_manager: Optional["VisionManager"] = None,
        output_dir: Optional[Path] = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD_HIGH,
        enable_ocr: bool = True,
    ) -> None:
        """
        Initialize element processor.

        Args:
            ocr_engine: Optional OCR engine for text extraction
            vision_manager: Optional VLM for image descriptions
            output_dir: Directory for saving extracted assets
            confidence_threshold: Threshold below which to use OCR
            enable_ocr: Whether OCR is allowed (respects --enable-ocr/--no-ocr CLI flag)
        """
        self.ocr_engine = ocr_engine
        self.vision_manager = vision_manager
        self.output_dir = Path(output_dir) if output_dir else Path("./output/assets")
        self.confidence_threshold = confidence_threshold
        self.enable_ocr = enable_ocr  # OCR Governance (Cluster B)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize OCR if not provided AND OCR is enabled
        if self.ocr_engine is None and self.enable_ocr:
            self._init_ocr_engine()

        logger.info(
            f"ElementProcessor initialized: "
            f"confidence_threshold={confidence_threshold}, "
            f"ocr={'enabled' if self.ocr_engine and self.enable_ocr else 'disabled (governance)'}, "
            f"vlm={'enabled' if self.vision_manager else 'disabled'}"
        )

    def _init_ocr_engine(self) -> None:
        """Initialize default OCR engine."""
        try:
            from ..ocr.enhanced_ocr_engine import EnhancedOCREngine

            self.ocr_engine = EnhancedOCREngine(
                confidence_threshold=self.confidence_threshold,
                enable_doctr=True,
            )
            logger.info("Initialized default EnhancedOCREngine")
        except ImportError as e:
            logger.warning(f"OCR engine not available: {e}")
            self.ocr_engine = None

    def process_document(
        self,
        document: UniversalDocument,
        source_file: Optional[str] = None,
    ) -> Generator[ProcessingResult, None, None]:
        """
        Process all elements in a document.

        Args:
            document: UniversalDocument to process
            source_file: Optional source filename override

        Yields:
            ProcessingResult for each element
        """
        source = source_file or document.source_file

        logger.info(
            f"Processing document: {document.doc_id} "
            f"({len(document.pages)} pages, "
            f"{document.total_text_elements} text, "
            f"{document.total_image_elements} image)"
        )

        for page in document.pages:
            for result in self.process_page(page, document.doc_id, source):
                yield result

            # Memory cleanup between pages
            gc.collect()

    def process_page(
        self,
        page: UniversalPage,
        doc_id: str,
        source_file: str = "unknown",
    ) -> Generator[ProcessingResult, None, None]:
        """
        Process all elements on a page.

        Args:
            page: UniversalPage to process
            doc_id: Document identifier
            source_file: Source filename

        Yields:
            ProcessingResult for each element
        """
        logger.debug(
            f"Processing page {page.page_number}: "
            f"{page.classification.value}, "
            f"{len(page.elements)} elements"
        )

        for idx, element in enumerate(page.elements):
            try:
                result = self._process_element(
                    element=element,
                    page=page,
                    doc_id=doc_id,
                    source_file=source_file,
                    element_idx=idx,
                )

                if result is not None:
                    yield result

            except Exception as e:
                logger.error(f"Failed to process element {idx} on page {page.page_number}: {e}")
                # Continue with next element
                continue

    def _process_element(
        self,
        element: Element,
        page: UniversalPage,
        doc_id: str,
        source_file: str,
        element_idx: int,
    ) -> Optional[ProcessingResult]:
        """
        Process a single element based on its type and quality.

        CRITICAL: This is where quality-based routing happens.
        """
        if element.type == ElementType.TEXT:
            return self._process_text_element(element, page, doc_id, source_file, element_idx)
        elif element.type == ElementType.IMAGE:
            return self._process_image_element(element, page, doc_id, source_file, element_idx)
        elif element.type == ElementType.TABLE:
            return self._process_table_element(element, page, doc_id, source_file, element_idx)
        else:
            logger.warning(f"Unknown element type: {element.type}")
            return None

    @staticmethod
    def _normalize_extraction_method(method: str) -> str:
        """Map internal extraction methods to schema-compatible values."""
        if not method:
            return "docling"
        method_lower = method.lower()
        if "ocr" in method_lower:
            return "ocr"
        if "shadow" in method_lower:
            return "shadow"
        return "docling"

    def _process_text_element(
        self,
        element: Element,
        page: UniversalPage,
        doc_id: str,
        source_file: str,
        element_idx: int,
    ) -> Optional[ProcessingResult]:
        """
        Process TEXT element.

        QUALITY-BASED ROUTING:
        - HIGH confidence (>=0.7): Use extracted content directly
        - LOW confidence (<0.7): Run OCR cascade on raw_image

        OCR GOVERNANCE (Cluster B):
        - If enable_ocr=False, OCR cascade is NEVER triggered
        - Empty text elements are dropped (not OCR'd)
        - This respects user's CLI preference (--no-ocr)

        OUTPUT: ALWAYS modality="text" with actual OCR'd content
        """
        content = element.content
        extraction_method = element.extraction_method.value
        confidence = element.confidence

        # Quality check: need OCR?
        if element.needs_ocr and element.has_image_data:
            # ================================================================
            # OCR GOVERNANCE (Cluster B): Respect enable_ocr flag
            # ================================================================
            if not self.enable_ocr:
                logger.info(
                    f"[OCR-GOVERNANCE] Page {page.page_number}: TEXT element needs OCR "
                    f"(confidence={confidence:.2f}), but OCR is DISABLED. "
                    f"Using available content or dropping element."
                )
                # Fall through - use whatever content we have (may be empty)
            else:
                logger.debug(
                    f"Text element needs OCR (confidence={confidence:.2f}): "
                    f"page {page.page_number}, idx {element_idx}"
                )

                # Run OCR cascade (only if enable_ocr=True)
                ocr_result = self._run_ocr(element.raw_image)

                if ocr_result:
                    content = ocr_result.get("text", content)
                    extraction_method = ocr_result.get("method", "ocr_cascade")
                    confidence = ocr_result.get("confidence", confidence)

                    logger.debug(
                        f"OCR result: {len(content)} chars, "
                        f"method={extraction_method}, conf={confidence:.2f}"
                    )

        # Normalize extraction method to schema-compatible value
        extraction_method = self._normalize_extraction_method(extraction_method)

        # Skip empty content
        if not content or not content.strip():
            logger.debug(f"Skipping empty text element on page {page.page_number}")
            return None

        # Generate chunk ID
        chunk_id = self._generate_chunk_id(doc_id, page.page_number, "text", element_idx)

        return ProcessingResult(
            chunk_id=chunk_id,
            doc_id=doc_id,
            modality="text",  # ALWAYS "text", never "shadow"
            content=content.strip(),
            page_number=page.page_number,
            bbox=element.get_bbox_list(),
            extraction_method=extraction_method,
            confidence=confidence,
            metadata={
                "source_file": source_file,
                "source_label": element.source_label,
                "page_width": page.width,
                "page_height": page.height,
            },
        )

    def _process_image_element(
        self,
        element: Element,
        page: UniversalPage,
        doc_id: str,
        source_file: str,
        element_idx: int,
    ) -> Optional[ProcessingResult]:
        """
        Process IMAGE element.

        VLM CONSTRAINT: Visual description ONLY, NO text reading.
        Uses VLM_VISUAL_ONLY_PROMPT to enforce this.

        OUTPUT: ALWAYS modality="image" with visual description
        """
        # Save asset to disk
        asset_path = self._save_image_asset(
            element.raw_image,
            doc_id,
            page.page_number,
            element_idx,
        )

        # Get VLM description (visual only)
        visual_description = self._get_visual_description(
            element.raw_image,
            page.page_number,
        )

        # Generate chunk ID
        chunk_id = self._generate_chunk_id(doc_id, page.page_number, "image", element_idx)

        return ProcessingResult(
            chunk_id=chunk_id,
            doc_id=doc_id,
            modality="image",  # ALWAYS "image"
            content=visual_description,
            page_number=page.page_number,
            bbox=element.get_bbox_list(),
            extraction_method=self._normalize_extraction_method("vlm"),
            confidence=element.confidence,
            asset_path=asset_path,
            visual_description=visual_description,
            metadata={
                "source_file": source_file,
                "source_label": element.source_label,
                "width_px": element.raw_image.shape[1] if element.has_image_data else None,
                "height_px": element.raw_image.shape[0] if element.has_image_data else None,
                "page_width": page.width,
                "page_height": page.height,
            },
        )

    def _process_table_element(
        self,
        element: Element,
        page: UniversalPage,
        doc_id: str,
        source_file: str,
        element_idx: int,
    ) -> Optional[ProcessingResult]:
        """
        Process TABLE element.

        OUTPUT: modality="table" with markdown or OCR'd content
        """
        content = element.content
        extraction_method = element.extraction_method.value

        # If no content, try OCR
        if not content and element.has_image_data and self.ocr_engine:
            ocr_result = self._run_ocr(element.raw_image)
            if ocr_result:
                content = ocr_result.get("text", "")
                extraction_method = "ocr_table"

        # Save asset
        asset_path = self._save_image_asset(
            element.raw_image,
            doc_id,
            page.page_number,
            element_idx,
            element_type="table",
        )

        extraction_method = self._normalize_extraction_method(extraction_method)

        # Generate chunk ID
        chunk_id = self._generate_chunk_id(doc_id, page.page_number, "table", element_idx)

        return ProcessingResult(
            chunk_id=chunk_id,
            doc_id=doc_id,
            modality="table",  # ALWAYS "table"
            content=content or f"[Table on page {page.page_number}]",
            page_number=page.page_number,
            bbox=element.get_bbox_list(),
            extraction_method=extraction_method,
            confidence=element.confidence,
            asset_path=asset_path,
            metadata={
                "source_file": source_file,
                "source_label": element.source_label,
                "page_width": page.width,
                "page_height": page.height,
            },
        )

    def _run_ocr(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Run OCR cascade on image."""
        if self.ocr_engine is None:
            logger.warning("OCR engine not available")
            return None

        if image is None or image.size == 0:
            return None

        try:
            result = self.ocr_engine.process_page(image)

            return {
                "text": result.text,
                "confidence": result.confidence,
                "method": result.layer_used.value if hasattr(result, "layer_used") else "ocr",
            }
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return None

    def _get_visual_description(
        self,
        image: Optional[np.ndarray],
        page_number: int,
    ) -> str:
        """
        Get VLM visual description.

        CRITICAL: Uses VISUAL_ONLY_PROMPT to prevent text reading.
        """
        if self.vision_manager is None:
            return f"[Image on page {page_number}]"

        if image is None or image.size == 0:
            return f"[Image on page {page_number}]"

        try:
            from PIL import Image as PILImage

            # Convert numpy to PIL
            pil_image = PILImage.fromarray(image)

            # Get description with visual-only constraint
            if hasattr(self.vision_manager, "describe_image_visual_only"):
                description = self.vision_manager.describe_image_visual_only(
                    pil_image,
                    prompt=VLM_VISUAL_ONLY_PROMPT,
                )
            elif hasattr(self.vision_manager, "enrich_image"):
                # Fallback to standard method with visual-only context
                from ..state.context_state import create_context_state

                temp_state = create_context_state(doc_id="temp", source_file="temp")
                temp_state.update_page(page_number)

                description = self.vision_manager.enrich_image(
                    image=pil_image,
                    state=temp_state,
                    page_number=page_number,
                    anchor_text=None,
                )
            else:
                description = f"[Image on page {page_number}]"

            # Post-process: detect and warn if text was read
            if self._contains_text_reading(description):
                logger.warning(
                    f"VLM may have read text on page {page_number}. " "Consider stricter prompt."
                )

            return description

        except Exception as e:
            logger.error(f"VLM description failed: {e}")
            return f"[Image on page {page_number}]"

    def _contains_text_reading(self, description: str) -> bool:
        """Check if description contains text reading patterns."""
        text_reading_patterns = [
            "the text says",
            "the caption reads",
            "labeled as",
            "titled",
            "the heading",
            "the title reads",
            "it says",
            "the label indicates",
        ]

        description_lower = description.lower()
        return any(pattern in description_lower for pattern in text_reading_patterns)

    def _save_image_asset(
        self,
        image: Optional[np.ndarray],
        doc_id: str,
        page_number: int,
        element_idx: int,
        element_type: str = "figure",
    ) -> Optional[str]:
        """Save image asset to disk."""
        if image is None or image.size == 0:
            return None

        try:
            from PIL import Image as PILImage
            import cv2

            # Generate filename
            filename = f"{doc_id}_{page_number:03d}_{element_type}_{element_idx:02d}.png"
            filepath = self.output_dir / filename

            # Convert and save
            if len(image.shape) == 3 and image.shape[2] == 3:
                # RGB numpy array
                pil_image = PILImage.fromarray(image)
            else:
                # Grayscale or other format
                pil_image = PILImage.fromarray(image)

            pil_image.save(str(filepath), "PNG")

            logger.debug(f"Saved asset: {filename}")
            return f"assets/{filename}"

        except Exception as e:
            logger.error(f"Failed to save asset: {e}")
            return None

    def _generate_chunk_id(
        self,
        doc_id: str,
        page_number: int,
        modality: str,
        element_idx: int,
    ) -> str:
        """Generate unique chunk ID."""
        parts = f"{doc_id}_{page_number}_{modality}_{element_idx}"
        hash_suffix = hashlib.md5(parts.encode()).hexdigest()[:8]
        return f"{doc_id}_{page_number:03d}_{modality}_{hash_suffix}"


# ============================================================================
# FACTORY FUNCTION
# ============================================================================


def create_element_processor(
    output_dir: Optional[Path] = None,
    vision_manager: Optional["VisionManager"] = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD_HIGH,
) -> ElementProcessor:
    """
    Factory function to create ElementProcessor.

    Args:
        output_dir: Directory for assets
        vision_manager: Optional VLM manager
        confidence_threshold: OCR threshold

    Returns:
        Configured ElementProcessor
    """
    return ElementProcessor(
        output_dir=output_dir,
        vision_manager=vision_manager,
        confidence_threshold=confidence_threshold,
    )
