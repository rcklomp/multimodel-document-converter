"""
V2DocumentProcessor - Docling-Native Document Processing Engine
================================================================
ENGINE_USE: Docling v2.66.0 (Native Layout Analysis)

This module implements the V2.0 compliant document processor using Docling's
native layout analysis engine. It processes PDF, EPUB, HTML, DOCX documents
and produces validated ingestion.jsonl output per SRS Section 6.

REQ Compliance:
- REQ-PDF-01: Rich structure extraction via Docling
- REQ-PDF-02: Image/figure extraction with 10px padding (REQ-MM-01)
- REQ-PDF-03: Hybrid OCR fallback via Tesseract
- REQ-STATE: Hierarchical breadcrumb tracking via ContextStateV2
- REQ-MM-02: Asset naming [DocHash]_[Page]_[Type]_[Index].png
- REQ-CHUNK-03: VLM descriptions truncated to 400 chars

SRS Section 4: PDF Processing Pipeline
"The system MUST use a Docling-native processing pipeline for structured
document extraction with layout analysis."

Author: Claude 4.5 Opus (Architect)
Date: 2025-12-28
Updated: 2025-12-29 (Vision Enrichment Layer)
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import fitz  # PyMuPDF for page rendering (OCR cascade)
import numpy as np
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from PIL import Image

# V3.0.0: Import OCR cascade for scanned text regions
from .ocr.enhanced_ocr_engine import EnhancedOCREngine, OCRResult, OCRLayer

# Import ExtractionStrategy type for type hints (avoid circular import)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .orchestration.strategy_orchestrator import ExtractionStrategy
    from .orchestration.strategy_profiles import ProfileParameters

from .schema.ingestion_schema import (
    FileType,
    HierarchyMetadata,
    IngestionChunk,
    SemanticContext,
    create_image_chunk,
    create_table_chunk,
    create_text_chunk,
    get_ocr_confidence_level,
    calculate_hierarchy_level,
    COORD_SCALE,
)
from .state.context_state import ContextStateV2, create_context_state, is_valid_heading
from .state.magazine_section_detector import (
    MagazineSectionDetector,
    create_section_detector,
)
from .utils.advanced_spatial_propagator import (
    SpatialPropagator,
    create_spatial_propagator,
)
from .utils.coordinate_normalization import (
    ensure_normalized,
    validate_bbox_strict,
    denormalize_bbox,
)
from .utils.image_hash_registry import (
    ImageHashRegistry,
    create_image_hash_registry,
    DuplicateInfo,
)
from .vision.vision_manager import VisionManager, create_vision_manager

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# REQ-MM-01: 10px padding on all image crops
CROP_PADDING_PX: int = 10

# REQ-MM-02: Asset naming pattern
ASSET_PATTERN: str = "{doc_hash}_{page:03d}_{element_type}_{index:02d}.png"

# REQ-MM-03: Context snippet length
CONTEXT_SNIPPET_LENGTH: int = 300

# Image Quality Filter: Minimum size to avoid noise (e.g., USPS ID statements, logos)
# 50x50 pixels at 2.0x scale = 100x100 in original coordinates
MIN_IMAGE_WIDTH_PX: int = 50
MIN_IMAGE_HEIGHT_PX: int = 50

# REQ-CHUNK: Chunking parameters (SRS v2.1: 400 characters for child chunks)
MAX_CHUNK_CHARS: int = 400
CHUNK_OVERLAP_RATIO: float = 0.10  # 10% overlap
MIN_OVERLAP_CHARS: int = 60

# Sentence ending pattern for smart chunking
SENTENCE_END_PATTERN = re.compile(r"[.!?]\s+")

# REQ-META-02: Content denoising pattern (OCR artifacts)
NOISE_PATTERN = re.compile(r"^[· \-\.]+$")

# SRS Rule 4: Ad detection keywords (case-insensitive)
AD_KEYWORDS = [
    "subscription",
    "subscribe",
    "buy now",
    "purchase",
    "shop now",
    "click here",
    "limited time",
    "special offer",
    "discount",
    "save now",
    "order today",
    "call now",
    "visit our",
    "www.",
    "http://",
    "https://",
    "facebook.com",
    "twitter.com",
    "instagram.com",
    "youtube.com",
    "linkedin.com",
    "mental health",
    "finances",
    "reach out",
    "price",
    "special price",
    "sale",
    "deal",
    "offer ends",
    "hurry",
    "act now",
]


# ============================================================================
# V2DocumentProcessor
# ============================================================================


class V2DocumentProcessor:
    """
    V2.0 compliant document processor using Docling-native layout analysis.

    This processor:
    1. Accepts PDF, EPUB, HTML, DOCX files
    2. Extracts text, images, tables with structural hierarchy
    3. Maintains ContextStateV2 for breadcrumb tracking
    4. Enriches images with VLM descriptions (optional)
    5. Outputs validated IngestionChunk objects

    Usage:
        processor = V2DocumentProcessor(output_dir="./output", vision_provider="ollama")
        chunks = processor.process_document("document.pdf")
        for chunk in chunks:
            # chunk is validated IngestionChunk

    Batch Processing Mode:
        When used with BatchProcessor, accepts additional parameters for:
        - External VisionManager (shared cache across batches)
        - Initial ContextStateV2 (breadcrumb continuity)
        - Page offset (correct page numbering for batches)
        - Doc hash override (use original document hash)
        - Source file override (use original filename)
    """

    def __init__(
        self,
        output_dir: str = "./output",
        enable_ocr: bool = True,
        ocr_engine: str = "tesseract",
        max_pages: Optional[int] = None,
        vision_provider: str = "none",
        vision_api_key: Optional[str] = None,
        vision_cache_dir: Optional[Path] = None,
        vision_base_url: Optional[str] = None,
        # Batch processing parameters
        external_vision_manager: Optional["VisionManager"] = None,
        initial_state: Optional[ContextStateV2] = None,
        page_offset: int = 0,
        doc_hash_override: Optional[str] = None,
        source_file_override: Optional[str] = None,
        # Smart orchestration parameters
        extraction_strategy: Optional["ExtractionStrategy"] = None,
    ) -> None:
        """
        Initialize V2DocumentProcessor.

        Args:
            output_dir: Directory for extracted assets
            enable_ocr: Whether to enable OCR for scanned pages
            ocr_engine: OCR engine ("tesseract" or "easyocr")
            max_pages: Maximum number of pages to process (None = all pages)
            vision_provider: VLM provider ("ollama", "openai", "anthropic", "none")
            vision_api_key: API key for cloud vision providers
            vision_cache_dir: Directory for vision cache (None = use output_dir)
            vision_base_url: Custom API base URL for OpenAI-compatible APIs (LM Studio)
            external_vision_manager: Pre-configured VisionManager (for batch processing)
            initial_state: Initial ContextStateV2 (for breadcrumb continuity)
            page_offset: Offset to add to page numbers (for batch processing)
            doc_hash_override: Override document hash (for batch processing)
            source_file_override: Override source filename (for batch processing)
            extraction_strategy: Dynamic extraction strategy from StrategyOrchestrator
        """
        # Store extraction strategy for dynamic thresholds
        self._extraction_strategy = extraction_strategy

        # V3.0.0: Profile parameters for OCR configuration (shadow-first mode REMOVED)
        self._profile_params: Optional["ProfileParameters"] = None

        # Use strategy's min dimensions if provided, else defaults
        if extraction_strategy:
            self._min_image_width = extraction_strategy.min_image_width
            self._min_image_height = extraction_strategy.min_image_height
            self._extract_backgrounds = extraction_strategy.extract_backgrounds
            logger.info(
                f"Using extraction strategy: min_dim={self._min_image_width}x{self._min_image_height}px, "
                f"backgrounds={'enabled' if self._extract_backgrounds else 'disabled'}"
            )
        else:
            self._min_image_width = MIN_IMAGE_WIDTH_PX
            self._min_image_height = MIN_IMAGE_HEIGHT_PX
            self._extract_backgrounds = True  # Default to extracting backgrounds

        # Store batch processing parameters
        self._external_vision_manager = external_vision_manager
        self._initial_state = initial_state
        self._page_offset = page_offset
        self._doc_hash_override = doc_hash_override
        self._source_file_override = source_file_override
        self._final_state: Optional[ContextStateV2] = None

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir = self.output_dir / "assets"
        self.assets_dir.mkdir(parents=True, exist_ok=True)

        self.enable_ocr = enable_ocr
        self.ocr_engine = ocr_engine
        self.max_pages = max_pages

        # Initialize Docling converter
        self._converter = self._create_converter()

        # Initialize Vision Manager (use external if provided for batch processing)
        if external_vision_manager is not None:
            self._vision_manager: Optional[VisionManager] = external_vision_manager
            logger.info("Using external VisionManager (batch processing mode)")
        elif vision_provider and vision_provider.lower() != "none":
            cache_dir = vision_cache_dir if vision_cache_dir else self.output_dir
            try:
                self._vision_manager = create_vision_manager(
                    provider=vision_provider,
                    api_key=vision_api_key,
                    cache_dir=cache_dir,
                    base_url=vision_base_url,
                )
                logger.info(f"Vision enrichment enabled: {vision_provider}")
            except Exception as e:
                logger.warning(f"Failed to initialize vision provider: {e}")
                self._vision_manager = None
        else:
            self._vision_manager = None

        # ================================================================
        # GEMINI AUDIT FIX: Initialize enhancement modules
        # ================================================================
        # FIX #2: SpatialPropagator - Universal bbox for ALL modalities
        self._spatial_propagator = create_spatial_propagator()

        # FIX #4: MagazineSectionDetector - Enriched breadcrumbs
        self._section_detector = create_section_detector()

        logger.info(
            f"ENGINE_USE: Docling v2.66.0 | V2DocumentProcessor initialized | "
            f"OCR: {ocr_engine if enable_ocr else 'disabled'} | "
            f"Vision: {vision_provider} | "
            f"Max pages: {max_pages if max_pages else 'ALL'} | "
            f"Enhancements: SpatialPropagator, MagazineSectionDetector"
        )

    def _create_converter(self) -> DocumentConverter:
        """
        Create Docling DocumentConverter with V2.0 compliant options.

        SRS REQ-PDF-04: High-fidelity rendering (min scale 2.0)

        OCR is explicitly disabled via do_ocr=False when enable_ocr is False
        to prevent EasyOCR import warnings.

        Increased figure detection: generate_background_images=True to catch
        editorial photos that may be labeled as background elements.
        """
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = 2.0  # High-Fidelity Rendering
        pipeline_options.generate_page_images = False  # No full-page renders
        pipeline_options.generate_picture_images = True  # Extract figures
        pipeline_options.generate_table_images = True  # Extract tables

        # REQ: ZERO-LATENCY PARAMETER ENFORCEMENT
        # Note: Page limiting is enforced at the batch splitting stage,
        # not at the Docling converter level
        if self.max_pages is not None and self.max_pages > 0:
            logger.info(
                f"[CORE] Page limit set to: {self.max_pages}. Will process only first {self.max_pages} pages."
            )

        # Explicitly disable OCR at the pipeline level to prevent EasyOCR warnings
        if self.enable_ocr:
            pipeline_options.do_ocr = True
            try:
                if self.ocr_engine == "easyocr":
                    pipeline_options.ocr_options = EasyOcrOptions()
                else:
                    pipeline_options.ocr_options = EasyOcrOptions()
            except ImportError:
                logger.warning("OCR library not installed, disabling OCR")
                pipeline_options.do_ocr = False
                self.enable_ocr = False
        else:
            # Explicitly disable OCR to prevent EasyOCR import warnings
            pipeline_options.do_ocr = False
            logger.info("OCR explicitly disabled via pipeline_options.do_ocr=False")

        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

        logger.info("ENGINE_USE: Using Docling v2.66.0 with REQ-PDF-04 compliance")
        return converter

    def _compute_doc_hash(self, file_path: Path) -> str:
        """Compute MD5 hash of document for unique identification."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:12]

    def _get_file_type(self, file_path: Path) -> FileType:
        """Determine FileType from extension."""
        ext = file_path.suffix.lower()
        type_map = {
            ".pdf": FileType.PDF,
            ".epub": FileType.EPUB,
            ".html": FileType.HTML,
            ".htm": FileType.HTML,
            ".docx": FileType.DOCX,
            ".pptx": FileType.PPTX,
            ".xlsx": FileType.XLSX,
        }
        return type_map.get(ext, FileType.PDF)

    def _is_advertisement(self, text: str) -> bool:
        """Detect if text is likely an advertisement (SRS Rule 4)."""
        if not text or len(text) < 10:
            return False

        text_lower = text.lower()
        keyword_count = sum(1 for keyword in AD_KEYWORDS if keyword in text_lower)

        if keyword_count >= 2 and len(text) < 200:
            return True
        if keyword_count >= 3:
            return True

        return False

    def _is_noise_content(self, text: str) -> bool:
        """
        REQ-META-02: Detect if content is layout noise/OCR artifacts.

        Filters: "· ·", "---", "...", "-", etc.
        """
        if not text or not text.strip():
            return True

        text = text.strip()

        # Pattern match: only dots, dashes, spaces, bullets
        if NOISE_PATTERN.match(text):
            logger.debug(f"[DENOISE] Rejected content (noise pattern): '{text}'")
            return True

        # Use same validation as heading
        if not is_valid_heading(text):
            logger.debug(f"[DENOISE] Rejected content (invalid): '{text}'")
            return True

        return False

    def _extract_heading_level(self, label: str) -> Optional[int]:
        """Extract heading level from Docling label."""
        label_lower = label.lower()
        if "title" in label_lower:
            return 1
        elif "section" in label_lower or "heading" in label_lower:
            for i in range(1, 7):
                if str(i) in label_lower or f"level{i}" in label_lower:
                    return i
            return 2
        return None

    def _apply_padding(
        self,
        bbox: List[float],
        page_width: float = 612.0,
        page_height: float = 792.0,
    ) -> List[float]:
        """Apply REQ-MM-01 10px padding to bounding box."""
        x0, y0, x1, y1 = bbox
        x0 = max(0.0, x0 - CROP_PADDING_PX)
        y0 = max(0.0, y0 - CROP_PADDING_PX)
        x1 = min(page_width, x1 + CROP_PADDING_PX)
        y1 = min(page_height, y1 + CROP_PADDING_PX)
        return [x0, y0, x1, y1]

    def _check_image_size(
        self,
        image: Optional[Image.Image],
        min_width: int = MIN_IMAGE_WIDTH_PX,
        min_height: int = MIN_IMAGE_HEIGHT_PX,
    ) -> bool:
        """
        Check if image meets minimum size requirements.

        Filters out noise like USPS ID statements and tiny logos.
        Images smaller than 50x50 pixels are considered noise.

        Args:
            image: PIL Image to check
            min_width: Minimum width in pixels
            min_height: Minimum height in pixels

        Returns:
            True if image size is acceptable, False if it's noise
        """
        if image is None:
            return False

        width, height = image.size

        if width < min_width or height < min_height:
            logger.debug(
                f"Filtered image (too small): {width}x{height} " f"(min {min_width}x{min_height})"
            )
            return False

        logger.debug(f"Image size OK: {width}x{height}")
        return True

    def _extract_raw_image(
        self,
        element: Any,
        bbox_normalized: Optional[List[int]],
        page_images: Dict[int, Image.Image],
        page_no: int,
        page_dims: Tuple[float, float],
    ) -> Optional[Image.Image]:
        """
        Extract raw image from element WITHOUT saving to disk.

        This enables size-checking BEFORE saving to prevent orphan assets.

        COORDINATE TRANSFORMATION (REQ-COORD-02):
        ==========================================
        The bbox parameter is normalized 0-1000 (per REQ-COORD-01).
        For cropping, we must convert to absolute pixels:

        1. Denormalize: bbox_normalized (0-1000) → PDF points
           pixel_x = (normalized_x / 1000) * page_width
        2. Scale: PDF points → rendered pixels (dynamic scale)
           rendered_x = pixel_x * (page_img.width / page_width)

        MATHEMATICAL VERIFICATION:
        ==========================
        Given: bbox_normalized = [100, 200, 300, 400] on 612x792 page
        Step 1: Denormalize to PDF points
                x0 = (100/1000) * 612 = 61.2
                y0 = (200/1000) * 792 = 158.4
                x1 = (300/1000) * 612 = 183.6
                y1 = (400/1000) * 792 = 316.8
        Step 2: Scale to rendered pixels (2.0x)
                crop = (122, 316, 367, 633)

        Args:
            element: Docling element (may have .image attribute)
            bbox_normalized: Bounding box in 0-1000 normalized scale (or None)
            page_images: Dict of page number → rendered PIL Image
            page_no: Batch page number (for page_images lookup)
            page_dims: (page_width, page_height) in PDF points

        Returns:
            PIL Image if extracted successfully, None otherwise
        """
        extracted_image: Optional[Image.Image] = None

        try:
            # PRIORITY 1: Crop from rendered page image using bbox
            # This preserves consistent 10px padding from _apply_padding()
            if bbox_normalized and page_no in page_images:
                page_img = page_images[page_no]
                page_width, page_height = page_dims

                # ============================================================
                # COORDINATE TRANSFORMATION: Normalized → Absolute Pixels
                # ============================================================
                # Step 1: Denormalize bbox from 0-1000 to PDF points
                abs_bbox = denormalize_bbox(bbox_normalized, page_width, page_height)
                # abs_bbox is now [x0_pts, y0_pts, x1_pts, y1_pts] in PDF points

                # Step 2: Scale PDF points to rendered pixels (dynamic)
                # Derive scale from rendered image size vs PDF points
                if page_width <= 0 or page_height <= 0:
                    logger.warning(
                        f"Invalid page_dims for scaling: {page_dims}. "
                        f"Falling back to 1.0 scale."
                    )
                    scale_x = 1.0
                    scale_y = 1.0
                else:
                    scale_x = page_img.size[0] / page_width
                    scale_y = page_img.size[1] / page_height

                crop_box = (
                    int(abs_bbox[0] * scale_x),
                    int(abs_bbox[1] * scale_y),
                    int(abs_bbox[2] * scale_x),
                    int(abs_bbox[3] * scale_y),
                )

                # Validate crop coordinates against page image bounds
                img_width, img_height = page_img.size
                x0 = max(0, min(crop_box[0], img_width - 1))
                y0 = max(0, min(crop_box[1], img_height - 1))
                x1 = max(x0 + 1, min(crop_box[2], img_width))
                y1 = max(y0 + 1, min(crop_box[3], img_height))

                # Validate minimum crop size
                if x1 <= x0 or y1 <= y0:
                    logger.warning(
                        f"Invalid crop coordinates after denormalization: "
                        f"bbox_normalized={bbox_normalized} → crop=({x0},{y0},{x1},{y1})"
                    )
                    return None

                extracted_image = page_img.crop((x0, y0, x1, y1))

                logger.debug(
                    f"Extracted image via crop: "
                    f"bbox_normalized={bbox_normalized} → "
                    f"abs_bbox={[f'{v:.1f}' for v in abs_bbox]} → "
                    f"crop=({x0},{y0},{x1},{y1}) → "
                    f"size={extracted_image.size}"
                )
                return extracted_image

            # PRIORITY 2: Try to get image from element directly (Docling native)
            # NOTE: If no page image is available, we cannot apply consistent padding.
            if bbox_normalized and page_no not in page_images:
                logger.warning(
                    f"Page image missing for page {page_no}; "
                    f"falling back to element.image without padding."
                )
            if hasattr(element, "image") and element.image:
                img_data = element.image
                if hasattr(img_data, "pil_image") and img_data.pil_image is not None:
                    extracted_image = img_data.pil_image
                    logger.debug(f"Extracted image from element.image (native)")
                    return extracted_image
                elif isinstance(img_data, bytes):
                    extracted_image = Image.open(BytesIO(img_data))
                    logger.debug(f"Extracted image from element.image (bytes)")
                    return extracted_image
                elif isinstance(img_data, Image.Image):
                    extracted_image = img_data
                    logger.debug(f"Extracted image from element.image (PIL)")
                    return extracted_image

            logger.debug(f"No image source available for element on page {page_no}")
            return None

        except Exception as e:
            logger.error(f"Failed to extract raw image on page {page_no}: {e}")
            return None

    def _save_asset(
        self,
        element: Any,
        asset_path: str,
        bbox_normalized: Optional[List[int]],
        page_images: Dict[int, Image.Image],
        page_no: int,
        page_dims: Tuple[float, float] = (612.0, 792.0),
    ) -> Optional[Image.Image]:
        """
        Save image/table asset to disk with proper coordinate transformation.

        CRITICAL FIX (Cluster A - Visual Integrity):
        =============================================
        The bbox parameter is in NORMALIZED 0-1000 scale (per REQ-COORD-01).
        This method now properly denormalizes to PDF points, then scales to
        rendered pixels for accurate cropping.

        REQ-MM-01: 10px padding is applied BEFORE normalization in _apply_padding()

        Args:
            element: Docling element with potential .image attribute
            asset_path: Relative path for saving (e.g., "assets/xxx_001_figure_01.png")
            bbox_normalized: Bounding box in 0-1000 normalized scale
            page_images: Dict of page number → rendered PIL Image (at 2.0x scale)
            page_no: Batch page number (key for page_images)
            page_dims: (page_width, page_height) in PDF points (default: Letter size)

        Returns:
            PIL Image if successfully saved (for VLM enrichment), None otherwise
        """
        full_path = self.assets_dir / Path(asset_path).name

        try:
            # Extract raw image using proper coordinate transformation
            saved_image = self._extract_raw_image(
                element=element,
                bbox_normalized=bbox_normalized,
                page_images=page_images,
                page_no=page_no,
                page_dims=page_dims,
            )

            if saved_image is not None:
                saved_image.save(str(full_path), "PNG")
                logger.debug(
                    f"Saved asset: {full_path} ({saved_image.size[0]}x{saved_image.size[1]})"
                )
                return saved_image

            logger.warning(f"Could not save asset: {asset_path} - no image data")
            return None

        except Exception as e:
            logger.error(f"Failed to save asset {asset_path}: {e}")
            return None

    def _enrich_image_with_vlm(
        self,
        image: Optional[Image.Image],
        state: ContextStateV2,
        page_no: int,
        anchor_text: Optional[str] = None,
        profile_params: Optional["ProfileParameters"] = None,
    ) -> str:
        """
        Enrich image with VLM description if available.

        REQ-OCR-01: If profile_params.enable_ocr_hints is True, uses the
        OCR-hint-aware enrichment method for brand name detection.

        Falls back to breadcrumb-based description if VLM unavailable.

        Args:
            image: PIL Image to describe
            state: Context state with breadcrumbs
            page_no: Current page number
            anchor_text: Surrounding text for context
            profile_params: Profile parameters (optional, enables OCR hints)
        """
        # If no vision manager or no image, use fallback
        if not self._vision_manager or image is None:
            return self._generate_fallback_description(state, page_no, anchor_text)

        try:
            # REQ-OCR-01: Use OCR-hint-aware enrichment when profile enables it
            if profile_params and profile_params.enable_ocr_hints:
                logger.info(
                    f"[HYBRID-VLM] Using OCR-hint enrichment for page {page_no} "
                    f"(profile: enable_ocr_hints=True)"
                )
                description = self._vision_manager.enrich_image_with_ocr_hints(
                    image=image,
                    state=state,
                    page_number=page_no,
                    anchor_text=anchor_text,
                    profile_params=profile_params,
                )
            else:
                # Standard VLM enrichment (no OCR hints)
                description = self._vision_manager.enrich_image(
                    image=image,
                    state=state,
                    page_number=page_no,
                    anchor_text=anchor_text,
                )
            return description
        except Exception as e:
            logger.warning(f"VLM enrichment failed: {e}")
            return self._generate_fallback_description(state, page_no, anchor_text)

    def _generate_fallback_description(
        self,
        state: ContextStateV2,
        page_no: int,
        anchor_text: Optional[str] = None,
    ) -> str:
        """Generate fallback description using breadcrumbs and anchor text."""
        parts = [f"[Figure on page {page_no}]"]

        breadcrumbs = state.get_breadcrumb_path()
        if breadcrumbs:
            path = " > ".join(breadcrumbs)
            parts.append(f"Context: {path}")

        if anchor_text and len(anchor_text) > 20:
            ctx = anchor_text[:100] + "..." if len(anchor_text) > 100 else anchor_text
            parts.append(f"Surrounded by: {ctx}")

        description = " | ".join(parts)
        return description[:MAX_CHUNK_CHARS]

    def _chunk_text_with_overlap(
        self,
        text: str,
        max_chars: int = MAX_CHUNK_CHARS,
        overlap_ratio: float = CHUNK_OVERLAP_RATIO,
    ) -> List[str]:
        """Split text into chunks with sentence-aware boundaries and overlap."""
        if len(text) <= max_chars:
            return [text]

        chunks = []
        overlap_chars = max(MIN_OVERLAP_CHARS, int(max_chars * overlap_ratio))

        sentences = SENTENCE_END_PATTERN.split(text)
        sentences_with_endings = []
        pos = 0
        for sent in sentences:
            if sent.strip():
                start = text.find(sent, pos)
                if start >= 0:
                    end = start + len(sent)
                    if end < len(text) and text[end] in ".!?":
                        end += 1
                    sentences_with_endings.append(text[start:end].strip())
                    pos = end

        if not sentences_with_endings:
            return self._chunk_by_words(text, max_chars, overlap_chars)

        current_chunk = ""
        for sentence in sentences_with_endings:
            if len(current_chunk) + len(sentence) + 1 > max_chars and current_chunk:
                chunks.append(current_chunk.strip())
                overlap_text = self._get_overlap_text(current_chunk, overlap_chars)
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk = current_chunk + " " + sentence if current_chunk else sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]

    def _chunk_by_words(
        self,
        text: str,
        max_chars: int,
        overlap_chars: int,
    ) -> List[str]:
        """Fallback word-boundary aware chunking."""
        words = text.split()
        chunks = []
        current_chunk = ""

        for word in words:
            if len(current_chunk) + len(word) + 1 > max_chars and current_chunk:
                chunks.append(current_chunk.strip())
                overlap_text = self._get_overlap_text(current_chunk, overlap_chars)
                current_chunk = overlap_text + " " + word
            else:
                current_chunk = current_chunk + " " + word if current_chunk else word

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]

    def _get_overlap_text(self, text: str, overlap_chars: int) -> str:
        """Get the last N characters of text at word boundary."""
        if len(text) <= overlap_chars:
            return text

        overlap_start = len(text) - overlap_chars
        search_limit = overlap_start + overlap_chars

        while overlap_start < len(text) and overlap_start < search_limit:
            if text[overlap_start] in " \n\t":
                break
            overlap_start += 1

        if overlap_start >= search_limit:
            overlap_start = len(text) - overlap_chars

        return text[overlap_start:].strip()

    def set_profile_params(self, profile_params: "ProfileParameters") -> None:
        """
        V3.0.0: Set profile parameters for OCR configuration.

        Shadow-first mode is REMOVED per ARCHITECTURE.md V3.0.0.
        Profile params are now used only for OCR hint configuration.

        Args:
            profile_params: Profile parameters from profile
        """
        self._profile_params = profile_params

        if profile_params.enable_ocr_hints:
            logger.info(
                f"[OCR-HINTS] Enabled for profile. "
                f"DPI: {profile_params.render_dpi}, "
                f"min_confidence: {profile_params.ocr_min_confidence}"
            )

    def get_vision_stats(self) -> Dict[str, Any]:
        """Get vision manager statistics."""
        if self._vision_manager:
            return self._vision_manager.get_stats()
        return {}

    def get_final_state(self) -> ContextStateV2:
        """
        Get the final ContextStateV2 after processing.

        Used by BatchProcessor to maintain breadcrumb continuity across batches.
        REQ-STATE: Breadcrumb path must be preserved at batch boundaries.

        Returns:
            Final ContextStateV2 with breadcrumbs from end of processing
        """
        if self._final_state is not None:
            return self._final_state.get_state_copy()

        # Return initial state if processing hasn't started
        if self._initial_state is not None:
            return self._initial_state.get_state_copy()

        # Return empty state as fallback
        return create_context_state()

    def process_document(
        self,
        input_path: str,
    ) -> Generator[IngestionChunk, None, None]:
        """
        Process a document and yield validated IngestionChunk objects.

        Args:
            input_path: Path to document file

        Yields:
            IngestionChunk objects validated against schema

        Note:
            In batch processing mode, uses overrides for doc_hash, source_file,
            page_offset, and inherits initial_state for breadcrumb continuity.
        """
        file_path = Path(input_path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        logger.info(f"ENGINE_USE: Docling v2.66.0 | Processing: {file_path.name}")

        # Use overrides for batch processing, otherwise compute from file
        doc_hash = self._doc_hash_override or self._compute_doc_hash(file_path)
        file_type = self._get_file_type(file_path)
        source_file = self._source_file_override or file_path.name

        # Use initial state if provided (batch processing), otherwise create new
        if self._initial_state is not None:
            state = self._initial_state.get_state_copy()
            logger.info(
                f"Inheriting state from previous batch: "
                f"breadcrumbs={state.get_breadcrumb_path()}"
            )
        else:
            state = create_context_state(doc_id=doc_hash, source_file=source_file)

        # Log page offset for batch processing
        if self._page_offset > 0:
            logger.info(f"Batch processing mode: page_offset={self._page_offset}")

        print("⏳ Starting Docling layout analysis...", flush=True)
        logger.info("Starting Docling document conversion...")

        import time as _time

        _start = _time.perf_counter()
        result = self._converter.convert(str(file_path))
        _elapsed = _time.perf_counter() - _start

        print(f"✓ Docling conversion complete in {_elapsed:.1f}s", flush=True)
        logger.info(f"Docling conversion completed in {_elapsed:.1f}s")

        element_indices: Dict[str, int] = {"figure": 0, "table": 0}
        text_buffer: List[str] = []

        page_images: Dict[int, Image.Image] = {}
        doc = result.document
        if hasattr(doc, "pages") and doc.pages:
            for page in doc.pages.values() if isinstance(doc.pages, dict) else doc.pages:
                if hasattr(page, "image") and page.image:
                    pg_no = getattr(page, "page_no", 1) or 1
                    if hasattr(page.image, "pil_image") and page.image.pil_image:
                        page_images[pg_no] = page.image.pil_image
                    elif isinstance(page.image, Image.Image):
                        page_images[pg_no] = page.image

        page_dims: Dict[int, Tuple[float, float]] = {}
        if hasattr(doc, "pages") and doc.pages:
            for page in doc.pages.values() if isinstance(doc.pages, dict) else doc.pages:
                pg_no = getattr(page, "page_no", 1) or 1
                width = getattr(page, "width", 612.0) or 612.0
                height = getattr(page, "height", 792.0) or 792.0
                page_dims[pg_no] = (width, height)

        # V3.0.0: Pending context queue for IMAGE next_text_snippet
        # IMAGE chunks are held until next TEXT chunk arrives, then their
        # next_text_snippet is filled with the subsequent text content.
        pending_image_chunks: List[IngestionChunk] = []

        for item_tuple in doc.iterate_items():
            element, _ = item_tuple

            for chunk in self._process_element_v2(
                element=element,
                state=state,
                doc_hash=doc_hash,
                source_file=source_file,
                file_type=file_type,
                element_indices=element_indices,
                text_buffer=text_buffer,
                page_images=page_images,
                page_dims=page_dims,
                page_offset=self._page_offset,
            ):
                # V3.0.0: Deferred yield for IMAGE chunks (pending context)
                from .schema.ingestion_schema import Modality

                if chunk.modality == Modality.IMAGE:
                    # Hold IMAGE chunk until next TEXT arrives
                    pending_image_chunks.append(chunk)
                    logger.debug(f"[PENDING-CONTEXT] Holding IMAGE chunk for next_text_snippet")
                elif chunk.modality == Modality.TEXT:
                    # Flush all pending IMAGE chunks with this text as next_text_snippet
                    if pending_image_chunks:
                        next_snippet = chunk.content[:CONTEXT_SNIPPET_LENGTH]
                        for pending in pending_image_chunks:
                            # Update semantic context with next text
                            if pending.semantic_context is None:
                                pending.semantic_context = SemanticContext(
                                    next_text_snippet=next_snippet
                                )
                            else:
                                pending.semantic_context.next_text_snippet = next_snippet
                            logger.debug(
                                f"[PENDING-CONTEXT] Filled IMAGE next_text: '{next_snippet[:50]}...'"
                            )
                            yield pending
                        pending_image_chunks.clear()
                    # Now yield the TEXT chunk
                    yield chunk
                else:
                    # TABLE or other modality - yield directly
                    yield chunk

        # Flush any remaining pending IMAGE chunks (no following text)
        for pending in pending_image_chunks:
            logger.debug(f"[PENDING-CONTEXT] Flushing IMAGE chunk without next_text")
            yield pending

        # Store final state for batch processing (REQ-STATE: breadcrumb continuity)
        self._final_state = state.get_state_copy()

        # Only flush cache if we own the vision manager (not external)
        if self._vision_manager and self._external_vision_manager is None:
            self._vision_manager.flush_cache()

        assets_saved = len(list(self.assets_dir.glob("*.png")))
        logger.info(
            f"ENGINE_USE: Docling v2.66.0 | Completed: {file_path.name} | "
            f"Doc ID: {doc_hash} | Assets saved: {assets_saved}"
        )

    def _process_element_v2(
        self,
        element: Any,
        state: ContextStateV2,
        doc_hash: str,
        source_file: str,
        file_type: FileType,
        element_indices: Dict[str, int],
        text_buffer: List[str],
        page_images: Dict[int, Image.Image],
        page_dims: Dict[int, Tuple[float, float]],
        page_offset: int = 0,
    ) -> Generator[IngestionChunk, None, None]:
        """
        Process a single document element with VLM enrichment.

        Args:
            page_offset: Offset to add to page numbers (for batch processing).
                        If batch starts at page 11, offset=10, so batch page 1 → actual page 11.
        """
        label_obj = getattr(element, "label", None)
        if label_obj is not None:
            label = str(label_obj.value) if hasattr(label_obj, "value") else str(label_obj)
        else:
            label = "text"

        text = getattr(element, "text", "") or ""

        # ========================================================================
        # PAGE NUMBER EXTRACTION - PROVENANCE LOCKED (REQ-PAGE-01)
        # ========================================================================
        # RULE: Page number MUST come from Docling provenance, NOT counters.
        # The prov[0].page_no is the absolute truth from the PDF.
        batch_page_no = 1
        docling_prov_page = None  # For validation
        bbox = None

        if hasattr(element, "prov") and element.prov:
            prov = element.prov[0] if isinstance(element.prov, list) else element.prov

            # PROVENANCE-BASED PAGE NUMBER (MANDATORY)
            if hasattr(prov, "page_no") and prov.page_no is not None:
                docling_prov_page = prov.page_no
                batch_page_no = prov.page_no
                logger.debug(
                    f"[PAGE-PROV] Element '{label[:20]}' → Docling prov.page_no={docling_prov_page}"
                )
            else:
                # Fallback: try page attribute
                if hasattr(prov, "page") and prov.page is not None:
                    docling_prov_page = prov.page
                    batch_page_no = prov.page
                    logger.debug(
                        f"[PAGE-PROV] Element '{label[:20]}' → Docling prov.page={docling_prov_page}"
                    )
                else:
                    logger.warning(
                        f"[PAGE-PROV-MISSING] Element '{label[:20]}' has NO page_no in provenance! "
                        f"Defaulting to page 1. prov={prov}"
                    )

            if hasattr(prov, "bbox") and prov.bbox:
                bbox_obj = prov.bbox
                if hasattr(bbox_obj, "l"):
                    x0, x1 = min(bbox_obj.l, bbox_obj.r), max(bbox_obj.l, bbox_obj.r)
                    y0, y1 = min(bbox_obj.t, bbox_obj.b), max(bbox_obj.t, bbox_obj.b)
                    if x1 > x0 and y1 > y0:
                        bbox = [x0, y0, x1, y1]
                elif hasattr(bbox_obj, "as_tuple"):
                    raw = bbox_obj.as_tuple()
                    x0, x1 = min(raw[0], raw[2]), max(raw[0], raw[2])
                    y0, y1 = min(raw[1], raw[3]), max(raw[1], raw[3])
                    if x1 > x0 and y1 > y0:
                        bbox = [x0, y0, x1, y1]

        # Apply page offset for batch processing
        # batch_page_no is 1-indexed within the batch PDF
        # page_offset is the starting page - 1 (e.g., batch 2 starts at page 11, offset=10)
        # So: actual_page = batch_page_no + page_offset = 1 + 10 = 11 ✓
        page_no = batch_page_no + page_offset

        if bbox:
            # DPI-CHECK: Fetch page dimensions per page (some magazines vary)
            page_w, page_h = page_dims.get(batch_page_no, (612.0, 792.0))
            bbox = self._apply_padding(bbox, page_w, page_h)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                bbox = None
            else:
                # REQ-COORD-01: NORMALIZE - Let ValueError PROPAGATE (no silent skip)
                bbox = ensure_normalized(bbox, page_w, page_h, f"element_{label}")

        state.update_page(page_no)

        heading_level = self._extract_heading_level(label)
        if heading_level and text.strip():
            if not state.breadcrumbs and heading_level > 1:
                heading_level = 1
            state.update_on_heading(text.strip(), heading_level)

        # REQ-HIER-03: breadcrumb_path MUST contain at minimum [source_filename, "Page X"]
        breadcrumbs = state.get_breadcrumb_path()
        source_name = Path(source_file).stem if source_file else "Document"
        page_marker = f"Page {page_no}"

        if not breadcrumbs:
            # No hierarchy detected - use minimum fallback
            breadcrumbs = [source_name, page_marker]
        elif len(breadcrumbs) == 1:
            # Only document name - add page marker
            breadcrumbs = [breadcrumbs[0], page_marker]
        elif page_marker not in " ".join(breadcrumbs):
            # Ensure page marker is present for navigation
            breadcrumbs = [breadcrumbs[0], page_marker] + breadcrumbs[1:]

        hierarchy = HierarchyMetadata(
            parent_heading=state.get_parent_heading(),
            breadcrumb_path=breadcrumbs,
            level=len(breadcrumbs) if breadcrumbs else None,
        )

        label_lower = label.lower()

        # Check for image/figure labels: picture, figure, image, background
        # Background images are included to capture editorial photos that may be in background layer
        is_image_label = (
            "picture" in label_lower
            or "figure" in label_lower
            or "image" in label_lower
            or "background" in label_lower  # Include background images (editorial photos)
        )

        if is_image_label:
            element_indices["figure"] += 1
            asset_name = ASSET_PATTERN.format(
                doc_hash=doc_hash,
                page=page_no,
                element_type="figure",
                index=element_indices["figure"],
            )
            asset_path = f"assets/{asset_name}"

            # ================================================================
            # PAGE VALIDATION: Filename MUST match metadata page number
            # ================================================================
            # Extract page from filename (e.g., "abc123_001_figure_01.png" → 1)
            filename_page = int(asset_name.split("_")[1])
            if filename_page != page_no:
                error_msg = (
                    f"[METADATA-VALIDATION-ERROR] Page mismatch! "
                    f"Filename says page {filename_page}, "
                    f"metadata says page {page_no}. "
                    f"Docling prov={docling_prov_page}, offset={page_offset}. "
                    f"STOPPING TO PREVENT CORRUPT DATA."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            # ================================================================
            # CLUSTER A FIX: DEFERRED SAVING TO PREVENT ORPHAN ASSETS
            # ================================================================
            # Extract raw image FIRST (without saving to disk)
            # Check size BEFORE saving to prevent orphan PNGs
            # This fixes the "Orphan Asset" issue where images are saved
            # before size filtering, leaving orphan files when rejected.
            # ================================================================

            # Get page dimensions for coordinate transformation
            img_page_w, img_page_h = page_dims.get(batch_page_no, (612.0, 792.0))

            # STEP 1: Extract raw image WITHOUT saving to disk
            raw_image = self._extract_raw_image(
                element=element,
                bbox_normalized=bbox,  # Already normalized 0-1000
                page_images=page_images,
                page_no=batch_page_no,
                page_dims=(img_page_w, img_page_h),
            )

            # STEP 2: Size check BEFORE saving (prevents orphan assets)
            # Note: _check_image_size returns False for None images
            if raw_image is None or not self._check_image_size(
                raw_image,
                min_width=self._min_image_width,
                min_height=self._min_image_height,
            ):
                logger.debug(
                    f"[ORPHAN-PREVENTION] Skipping image on page {page_no}: "
                    f"{'no image extracted' if raw_image is None else 'too small'} "
                    f"(threshold: {self._min_image_width}x{self._min_image_height}px). "
                    f"No file saved to disk."
                )
                return  # Skip this element - NO file written to disk

            # At this point, raw_image is guaranteed non-None (passed size check)
            # STEP 3: Only save to disk if size check passed
            print(
                f"    📸 Asset: {asset_name} | "
                f"Page={page_no} | "
                f"DoclingProv={docling_prov_page}",
                flush=True,
            )

            full_path = self.assets_dir / Path(asset_path).name
            try:
                raw_image.save(str(full_path), "PNG")
                logger.debug(
                    f"[DEFERRED-SAVE] Saved asset after size check: {full_path} "
                    f"({raw_image.size[0]}x{raw_image.size[1]})"
                )
                saved_image = raw_image
            except Exception as e:
                logger.error(f"Failed to save asset {asset_path}: {e}")
                return  # Skip this element if save fails

            # Context for VLM
            prev_text = " ".join(text_buffer[-3:]) if text_buffer else None
            if prev_text:
                prev_text = prev_text[-CONTEXT_SNIPPET_LENGTH:]

            # Enrich with VLM or fallback (only for high-value images)
            # REQ-OCR-01: Pass profile_params to enable OCR hints for scanned profiles
            content = self._enrich_image_with_vlm(
                image=saved_image,
                state=state,
                page_no=page_no,
                anchor_text=prev_text,
                profile_params=self._profile_params,  # Enables OCR hints if set
            )

            # Visual description for provenance
            # RULE A & B: Enforce non-null visual descriptions
            if self._vision_manager and content and len(content) >= 20:
                visual_description = content
            else:
                # RULE A: Trigger Summary Fallback if no valid VLM description
                if not self._vision_manager:
                    # This is expected when --vision-provider none is used
                    # Only log at debug level, not warning
                    logger.debug("VLM disabled: using fallback description")
                    visual_description = content if content else f"[Figure on page {page_no}]"
                elif not content or len(content) < 20:
                    logger.warning(
                        f"VLM description too short ({len(content) if content else 0} chars)"
                    )
                    fallback = content if content else "No description available"
                    visual_description = f"VLM_FALLBACK: {fallback}"
                else:
                    visual_description = content

            # Ensure visual_description is never None
            if not visual_description:
                visual_description = (
                    "ERROR: Description generation failed - [Manual Review Required]"
                )

            # REQ-COORD-01: bbox is REQUIRED for image modality
            # Provide fallback full-page bbox if not available
            image_bbox: List[int] = bbox if bbox is not None else [0, 0, COORD_SCALE, COORD_SCALE]

            # REQ-COORD-02: Get page dimensions for UI overlay support
            img_page_w, img_page_h = page_dims.get(batch_page_no, (612.0, 792.0))

            yield create_image_chunk(
                doc_id=doc_hash,
                content=content,
                source_file=source_file,
                file_type=file_type,
                page_number=page_no,
                asset_path=asset_path,
                bbox=image_bbox,
                hierarchy=hierarchy,
                prev_text=prev_text,
                visual_description=visual_description,
                page_width=int(img_page_w),
                page_height=int(img_page_h),
            )

        elif "table" in label_lower:
            element_indices["table"] += 1
            asset_name = ASSET_PATTERN.format(
                doc_hash=doc_hash,
                page=page_no,
                element_type="table",
                index=element_indices["table"],
            )
            asset_path = f"assets/{asset_name}"

            # Get page dimensions for proper coordinate transformation
            tbl_page_w, tbl_page_h = page_dims.get(batch_page_no, (612.0, 792.0))

            # Save table asset with proper coordinate transformation
            saved_table_image = self._save_asset(
                element,
                asset_path,
                bbox,
                page_images,
                batch_page_no,
                page_dims=(tbl_page_w, tbl_page_h),
            )
            asset_path = asset_path if saved_table_image is not None else None

            table_content = text if text else f"[Table on page {page_no}]"

            # REQ-COORD-01: bbox is REQUIRED for table modality
            # Provide fallback full-page bbox if not available
            table_bbox: List[int] = bbox if bbox is not None else [0, 0, COORD_SCALE, COORD_SCALE]

            # REQ-COORD-02: Get page dimensions for UI overlay support
            tbl_page_w, tbl_page_h = page_dims.get(batch_page_no, (612.0, 792.0))

            yield create_table_chunk(
                doc_id=doc_hash,
                content=table_content,
                source_file=source_file,
                file_type=file_type,
                page_number=page_no,
                bbox=table_bbox,
                hierarchy=hierarchy,
                asset_path=asset_path,
                page_width=int(tbl_page_w),
                page_height=int(tbl_page_h),
            )

        # ========================================================================
        # V3.0.0: TEXT ELEMENT PROCESSING WITH OCR CASCADE
        # ========================================================================
        # Per DECISION_OCR_CASCADE.md and ARCHITECTURE.md:
        # - TEXT regions with content → use directly
        # - TEXT regions with EMPTY content (scanned) → OCR cascade
        # - "text", "paragraph", "section_header", "list_item" labels are TEXT
        # ========================================================================
        is_text_label = (
            "text" in label_lower
            or "paragraph" in label_lower
            or "section" in label_lower
            or "list" in label_lower
            or "caption" in label_lower
            or label_lower == ""  # Default label is text
        )

        # Handle TEXT elements (with or without content)
        if is_text_label and not is_image_label and "table" not in label_lower:
            # Get prev_text for semantic context BEFORE processing
            prev_text_context = " ".join(text_buffer[-3:]) if text_buffer else None
            if prev_text_context:
                prev_text_context = prev_text_context[-CONTEXT_SNIPPET_LENGTH:]

            # ================================================================
            # V3.0.0 OCR CASCADE: If text is EMPTY, trigger OCR on the region
            # Per DECISION_OCR_CASCADE.md: Scanned pages produce empty text
            # ================================================================
            final_text = text.strip()

            if not final_text and bbox is not None:
                # ================================================================
                # OCR GOVERNANCE (Cluster B): Respect CLI --enable-ocr/--disable-ocr
                # ================================================================
                # If enable_ocr is False, we NEVER trigger OCR - the element
                # remains empty or is dropped. This respects user's CLI preference.
                # ================================================================
                if not self.enable_ocr:
                    logger.info(
                        f"[OCR-GOVERNANCE] Page {page_no}: Empty TEXT region detected, "
                        f"but OCR is DISABLED via CLI (--no-ocr). Skipping element."
                    )
                    final_text = ""
                    return  # Drop the empty element - no OCR allowed

                # OCR is enabled - proceed with cascade
                logger.info(
                    f"[OCR-CASCADE] Page {page_no}: TEXT element has empty content, "
                    f"triggering OCR cascade on bbox"
                )
                print(
                    f"    🔬 [OCR] Page {page_no}: Empty TEXT region detected, running OCR...",
                    flush=True,
                )

                # Render the region for OCR using PyMuPDF
                # This requires the original PDF path which we need to get
                try:
                    # Initialize OCR engine if not done
                    if not hasattr(self, "_ocr_engine"):
                        self._ocr_engine = EnhancedOCREngine(
                            confidence_threshold=0.7,
                            enable_tesseract=True,
                            enable_doctr=True,
                        )

                    # Get page image for OCR
                    if batch_page_no in page_images:
                        page_img = page_images[batch_page_no]
                        # Convert PIL to numpy for OCR
                        page_np = np.array(page_img.convert("RGB"))

                        # Denormalize bbox for cropping (0-1000 → pixels)
                        page_w, page_h = page_dims.get(batch_page_no, (612.0, 792.0))
                        scale = 2.0  # Docling render scale

                        # bbox is normalized 0-1000, convert to pixels
                        x0 = int((bbox[0] / COORD_SCALE) * page_w * scale)
                        y0 = int((bbox[1] / COORD_SCALE) * page_h * scale)
                        x1 = int((bbox[2] / COORD_SCALE) * page_w * scale)
                        y1 = int((bbox[3] / COORD_SCALE) * page_h * scale)

                        # Ensure valid crop coordinates
                        h, w = page_np.shape[:2]
                        x0 = max(0, min(x0, w - 1))
                        y0 = max(0, min(y0, h - 1))
                        x1 = max(x0 + 1, min(x1, w))
                        y1 = max(y0 + 1, min(y1, h))

                        # Crop region
                        region_crop = page_np[y0:y1, x0:x1]

                        if region_crop.size > 0:
                            # Run OCR cascade
                            ocr_result = self._ocr_engine.process_page(region_crop)

                            if ocr_result.text and len(ocr_result.text.strip()) > 5:
                                final_text = ocr_result.text.strip()
                                logger.info(
                                    f"[OCR-CASCADE] Page {page_no}: OCR extracted "
                                    f"{len(final_text)} chars via {ocr_result.layer_used.value} "
                                    f"(confidence={ocr_result.confidence:.2f})"
                                )
                                print(
                                    f"    ✓ [OCR] Extracted {len(final_text)} chars "
                                    f"({ocr_result.layer_used.value}, conf={ocr_result.confidence:.2f})",
                                    flush=True,
                                )
                            else:
                                logger.warning(
                                    f"[OCR-CASCADE] Page {page_no}: OCR returned no usable text"
                                )
                                print(
                                    f"    ⚠️ [OCR] No text extracted from region",
                                    flush=True,
                                )
                    else:
                        logger.warning(
                            f"[OCR-CASCADE] Page {page_no}: No page image available for OCR"
                        )
                except Exception as ocr_err:
                    logger.error(f"[OCR-CASCADE] Page {page_no}: OCR failed: {ocr_err}")
                    print(f"    ❌ [OCR] Error: {ocr_err}", flush=True)

            # Skip if still no text after OCR attempt
            if not final_text:
                logger.debug(f"[TEXT] Page {page_no}: Skipping empty text element after OCR")
                return

            # REQ-META-02: Filter noise content BEFORE advertisement check
            if self._is_noise_content(final_text):
                logger.debug(f"[DENOISE] Filtered noise on page {page_no}: '{final_text[:50]}'")
                return

            if self._is_advertisement(final_text):
                logger.debug(f"Filtered advertisement on page {page_no}")
                return

            # ================================================================
            # GEMINI AUDIT FIX #4: Magazine Section Detection
            # ================================================================
            # Detect magazine section headers and update breadcrumbs
            is_first_text_on_page = state.current_page != page_no or not text_buffer
            section_result = self._section_detector.analyze(
                text=final_text,
                page_number=page_no,
                is_first_on_page=is_first_text_on_page,
            )
            if section_result.is_section and section_result.section_name:
                # Update breadcrumbs with detected section
                state.update_on_heading(
                    section_result.section_name,
                    section_result.suggested_level,
                )
                self._section_detector.register_detected_section(
                    section_result.section_name, page_no
                )
                logger.debug(
                    f"[SECTION-DETECT] Page {page_no}: "
                    f"'{section_result.section_name}' ({section_result.detection_method})"
                )
                # Update hierarchy with new breadcrumbs
                hierarchy = HierarchyMetadata(
                    parent_heading=state.get_parent_heading(),
                    breadcrumb_path=state.get_breadcrumb_path(),
                    level=state.current_header_level if state.current_header_level > 0 else None,
                )

            # Update text buffer for semantic context
            text_buffer.append(final_text)
            if len(text_buffer) > 10:
                text_buffer.pop(0)

            # ================================================================
            # GEMINI AUDIT FIX #2: Spatial Propagation for TEXT chunks
            # ================================================================
            # Extract and normalize bbox for TEXT modality
            page_w, page_h = page_dims.get(batch_page_no, (612.0, 792.0))
            spatial_result = self._spatial_propagator.extract_and_normalize(
                element=element,
                page_dims=(page_w, page_h),
                context=f"text_page{page_no}",
            )
            text_bbox = spatial_result.bbox_normalized if spatial_result.is_valid else None

            text_chunks = self._chunk_text_with_overlap(final_text)

            for i, chunk_text in enumerate(text_chunks):
                if not self._is_noise_content(chunk_text) and not self._is_advertisement(
                    chunk_text
                ):
                    # ================================================================
                    # V3.0.0: SEMANTIC CONTEXT - prev_text_snippet, next_text_snippet
                    # ================================================================
                    # prev_text: from text_buffer (text BEFORE this chunk)
                    # next_text: next chunk in list (if available)
                    next_text_context = None
                    if i + 1 < len(text_chunks):
                        next_text_context = text_chunks[i + 1][:CONTEXT_SNIPPET_LENGTH]

                    yield create_text_chunk(
                        doc_id=doc_hash,
                        content=chunk_text,
                        source_file=source_file,
                        file_type=file_type,
                        page_number=page_no,
                        hierarchy=hierarchy,
                        bbox=text_bbox,  # FIX #2: Propagate spatial data
                        prev_text=prev_text_context,
                        next_text=next_text_context,
                    )

    def process_to_jsonl(
        self,
        file_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """Process document and write to JSONL file."""
        import json

        if output_path is None:
            final_output_path = self.output_dir / "ingestion.jsonl"
        else:
            final_output_path = Path(output_path)

        chunk_count = 0
        with open(final_output_path, "w", encoding="utf-8") as f:
            for chunk in self.process_document(file_path):
                chunk_dict = chunk.model_dump(mode="json")

                # REQ-COORD-01: ASSERTION - Crash if unnormalized coordinates
                # Per SRS Section 6.2: bbox MUST be integers in range [0, 1000]
                spatial = chunk_dict.get("metadata", {}).get("spatial")
                if spatial and spatial.get("bbox"):
                    bbox = spatial["bbox"]
                    assert all(isinstance(c, int) and 0 <= c <= 1000 for c in bbox), (
                        f"REQ-COORD-01 VIOLATION: Invalid bbox {bbox} in chunk "
                        f"{chunk_dict['chunk_id'][:16]}. Must be integers 0-1000."
                    )

                json_line = json.dumps(chunk_dict, ensure_ascii=False)
                f.write(json_line + "\n")
                chunk_count += 1

        logger.info(
            f"ENGINE_USE: Docling v2.66.0 | " f"Written {chunk_count} chunks to {final_output_path}"
        )

        return str(final_output_path)

    def process_to_jsonl_atomic(
        self,
        file_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Process document and write to JSONL file with ATOMIC WRITES.

        V3.0.0: Per task requirements - opens in append mode and flushes
        after each chunk to prevent data loss on connection errors.

        Args:
            file_path: Path to input document
            output_path: Optional output path (defaults to output_dir/ingestion.jsonl)

        Returns:
            Path to output JSONL file
        """
        import json

        if output_path is None:
            final_output_path = self.output_dir / "ingestion.jsonl"
        else:
            final_output_path = Path(output_path)

        # Clear file if exists (we're starting fresh)
        if final_output_path.exists():
            final_output_path.unlink()

        chunk_count = 0

        # Open in APPEND mode for atomic writes
        for chunk in self.process_document(file_path):
            chunk_dict = chunk.model_dump(mode="json")

            # REQ-COORD-01: ASSERTION - Crash if unnormalized coordinates
            spatial = chunk_dict.get("metadata", {}).get("spatial")
            if spatial and spatial.get("bbox"):
                bbox = spatial["bbox"]
                assert all(isinstance(c, int) and 0 <= c <= 1000 for c in bbox), (
                    f"REQ-COORD-01 VIOLATION: Invalid bbox {bbox} in chunk "
                    f"{chunk_dict['chunk_id'][:16]}. Must be integers 0-1000."
                )

            # ATOMIC WRITE: Open, write, flush, close for each chunk
            with open(final_output_path, "a", encoding="utf-8") as f:
                json_line = json.dumps(chunk_dict, ensure_ascii=False)
                f.write(json_line + "\n")
                f.flush()  # Force write to disk immediately

            chunk_count += 1

            # Log progress every 10 chunks
            if chunk_count % 10 == 0:
                logger.debug(f"[ATOMIC-WRITE] {chunk_count} chunks written to {final_output_path}")

        logger.info(
            f"ENGINE_USE: Docling v2.66.0 | "
            f"Written {chunk_count} chunks ATOMICALLY to {final_output_path}"
        )

        return str(final_output_path)


# ============================================================================
# FACTORY FUNCTION
# ============================================================================


def create_processor(
    output_dir: str = "./output",
    enable_ocr: bool = True,
    ocr_engine: str = "tesseract",
    vision_provider: str = "none",
    vision_api_key: Optional[str] = None,
) -> V2DocumentProcessor:
    """Factory function to create V2DocumentProcessor."""
    return V2DocumentProcessor(
        output_dir=output_dir,
        enable_ocr=enable_ocr,
        ocr_engine=ocr_engine,
        vision_provider=vision_provider,
        vision_api_key=vision_api_key,
    )
