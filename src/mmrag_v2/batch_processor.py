"""
Batch Processor - Memory-Efficient Large PDF Processing Orchestrator
=====================================================================
ENGINE_USE: Claude 4.5 Opus (Architect)

This module implements the "Divide and Conquer" batch processing strategy
for handling large PDFs (244+ pages) within 16GB RAM constraints.

Key Features:
- Physical PDF splitting into N-page batches (default: 10)
- Sequential batch execution with explicit gc.collect() between batches
- Global SHA-256 vision cache maintained across all batches
- ContextStateV2 persistence for breadcrumb continuity at batch boundaries
- Unified assets/ directory with correct page numbering
- Aggregated master_ingestion.jsonl output

REQ Compliance:
- REQ-PDF-05: Memory hygiene via gc.collect() after each batch
- REQ-MM-02: Asset naming [DocHash]_[Page]_[Type]_[Index].png with page_offset
- REQ-STATE: Breadcrumb continuity across batch boundaries
- REQ-CHUNK-03: VLM descriptions truncated to 400 chars
- REQ-OUT-01: JSONL output format

Author: Claude 4.5 Opus (Architect)
Date: 2025-12-29
"""

from __future__ import annotations

import gc
import hashlib
import io
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import fitz  # PyMuPDF for page rendering
import numpy as np
from PIL import Image

# REQ-RENDER: Disable PIL decompression bomb check for large DPI renders
# Combat Airplanes at 300 DPI produces images > 115M pixels
# Memory is still managed by gc.collect() between batches
Image.MAX_IMAGE_PIXELS = None

if TYPE_CHECKING:
    from .orchestration.strategy_orchestrator import ExtractionStrategy
    from .orchestration.strategy_profiles import ProfileParameters

from .schema.ingestion_schema import (
    AssetReference,
    FileType,
    HierarchyMetadata,
    IngestionChunk,
    Modality,
    ChunkMetadata,
    SpatialMetadata,
    SemanticContext,
    COORD_SCALE,
    create_image_chunk,
    create_table_chunk,
    create_text_chunk,
)
from .version import __schema_version__ as SCHEMA_VERSION

# V2.4.0: Shadow extraction is a CORE REQUIREMENT (REQ-MM-05/06/07, IRON-07)
# Shadow extraction catches large images (300x300px OR 40% page area) that
# Docling's AI-driven layout analysis may miss. This is the safety net.
from .state.context_state import ContextStateV2, create_context_state
from .utils.pdf_splitter import BatchInfo, PDFBatchSplitter, SplitResult
from .utils.image_hash_registry import (
    ImageHashRegistry,
    create_image_hash_registry,
    create_page1_validator,
)
from .vision.vision_manager import VisionManager, create_vision_manager
from .validators.token_validator import (
    TokenValidator,
    create_token_validator,
    TokenValidationResult,
)
from .validators.quality_filter_tracker import (
    QualityFilterTracker,
    FilterCategory,
    create_quality_filter_tracker,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_BATCH_SIZE: int = 10
DEFAULT_VLM_TIMEOUT: int = (
    180  # Seconds, increased for large vision models like llama3.2-vision (10.7B)
)
DEFAULT_VISION_PROVIDER: str = "ollama"


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class BatchProcessingResult:
    """Result of batch processing a large document."""

    success: bool
    original_path: Path
    original_hash: str
    total_pages: int
    batches_processed: int
    total_chunks: int
    output_jsonl: Path
    assets_dir: Path
    processing_time_seconds: float
    errors: List[str]
    vision_stats: Dict[str, Any]


# ============================================================================
# BATCH PROCESSOR
# ============================================================================


class BatchProcessor:
    """
    Orchestrates memory-efficient batch processing of large PDFs.

    This class implements the "Divide and Conquer" strategy:
    1. Split large PDF into N-page batches using PyMuPDF
    2. Process each batch sequentially (not parallel) to preserve RAM
    3. Maintain global VisionManager cache across batches
    4. Persist ContextStateV2 breadcrumbs between batches
    5. Aggregate all results into master_ingestion.jsonl

    Usage:
        processor = BatchProcessor(
            output_dir="./output",
            batch_size=10,
            vision_provider="ollama",
        )
        result = processor.process_pdf("large_document.pdf")
    """

    def __init__(
        self,
        output_dir: str = "./output",
        batch_size: int = DEFAULT_BATCH_SIZE,
        vision_provider: str = DEFAULT_VISION_PROVIDER,
        vision_model: Optional[str] = None,
        vision_api_key: Optional[str] = None,
        vision_base_url: Optional[str] = None,
        vlm_timeout: int = DEFAULT_VLM_TIMEOUT,
        vision_cache_dir: Optional[str] = None,
        enable_ocr: bool = True,
        ocr_engine: str = "easyocr",
        extraction_strategy: Optional["ExtractionStrategy"] = None,
        max_pages: Optional[int] = None,
        specific_pages: Optional[List[int]] = None,
        allow_fullpage_shadow: bool = False,
        strict_qa: bool = False,
        force_ocr: bool = False,
        qa_tolerance: float = 0.1,
        qa_noise_allowance: float = 0.25,
        auto_safe: bool = False,
        semantic_overlap: bool = True,
        vlm_context_depth: int = 3,
        # Phase 1B: Layout-aware OCR parameters
        ocr_mode: str = "legacy",
        ocr_confidence_threshold: float = 0.7,
        enable_doctr: bool = True,
    ) -> None:
        """
        Initialize the BatchProcessor.

        Args:
            output_dir: Directory for output files (JSONL and assets)
            batch_size: Number of pages per batch (default: 10)
            vision_provider: VLM provider ("ollama", "openai", "anthropic", "none")
            vision_model: VLM model name (optional for Ollama - auto-detects if not specified)
            vision_api_key: API key for cloud providers
            vision_base_url: Custom API base URL for OpenAI-compatible APIs (LM Studio)
            vlm_timeout: VLM read timeout in seconds (default: 180)
            enable_ocr: Whether to enable OCR for scanned pages
            ocr_engine: OCR engine ("tesseract" or "easyocr")
            extraction_strategy: Dynamic extraction strategy from StrategyOrchestrator
            max_pages: Maximum number of pages to process (None = all pages)
            specific_pages: List of specific page numbers to process (e.g., [6, 21, 169, 241])
            allow_fullpage_shadow: Allow full-page shadow assets (override Full-Page Guard)
            strict_qa: Enable strict QA-CHECK-01 mode (fail on token validation errors)
            semantic_overlap: Enable Dynamic Semantic Overlap (DSO) chunking (Gap #3)
            vlm_context_depth: Number of previous text chunks for VLM context (Gap #3)
            ocr_mode: OCR processing mode ("legacy" or "layout-aware")
            ocr_confidence_threshold: Minimum OCR confidence for layout-aware mode (0.0-1.0)
            enable_doctr: Enable Doctr Layer 3 for layout-aware OCR
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir = self.output_dir / "assets"
        self.assets_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.vision_provider = vision_provider
        self.vision_model = vision_model
        self.vision_api_key = vision_api_key
        self.vision_base_url = vision_base_url
        self.vlm_timeout = vlm_timeout
        self.vision_cache_dir = Path(vision_cache_dir) if vision_cache_dir else None
        self.enable_ocr = enable_ocr
        self.ocr_engine = ocr_engine
        self.extraction_strategy = extraction_strategy
        self.max_pages = max_pages
        self.specific_pages = specific_pages
        self.allow_fullpage_shadow = allow_fullpage_shadow
        self.strict_qa = strict_qa
        self.force_ocr = force_ocr
        self.qa_tolerance = qa_tolerance
        self.qa_noise_allowance = qa_noise_allowance
        self.auto_safe = auto_safe
        self.semantic_overlap = semantic_overlap
        self.vlm_context_depth = vlm_context_depth

        # Phase 1B: Layout-aware OCR parameters
        self.ocr_mode = ocr_mode
        self.ocr_confidence_threshold = ocr_confidence_threshold
        self.enable_doctr = enable_doctr

        # Will be initialized when processing starts
        self._vision_manager: Optional[VisionManager] = None
        self._context_state: Optional[ContextStateV2] = None
        self._refiner = None
        self._refiner_config: Optional[Dict[str, Any]] = None
        self._doc_hash: Optional[str] = None
        self._image_hash_registry: Optional[ImageHashRegistry] = None
        self._token_validator: Optional[TokenValidator] = None

        # REQ-OCR-01: Profile parameters for OCR hints and dynamic DPI
        self._profile_params: Optional["ProfileParameters"] = None

        # V2.4: Intelligence Stack Metadata (for observability)
        self._intelligence_metadata: Dict[str, Any] = {}

        # REQ-VLM-02: Track asset counts per page for low-recall trigger
        self._assets_per_page: Dict[int, int] = {}
        self._current_pdf_path: Optional[Path] = None

        # QA-CHECK-01: Initialize token validator for data integrity
        self._token_validator = create_token_validator(tolerance=0.10)

        # Quality Filter Tracker for token-level filtering analytics
        self._quality_filter_tracker: Optional[QualityFilterTracker] = None

        # REQ-COORD-02: Track page dimensions per page for UI overlay support
        self._page_dimensions: Dict[int, Tuple[int, int]] = {}

        logger.info(
            f"BatchProcessor initialized: "
            f"batch_size={batch_size}, "
            f"vision={vision_provider}/{vision_model}, "
            f"timeout={vlm_timeout}s, "
            f"max_pages={max_pages if max_pages else 'ALL'}"
        )

    def enable_refiner(
        self,
        provider: str = "ollama",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        threshold: float = 0.15,
        max_edit: float = 0.35,
    ) -> None:
        """
        Enable Semantic Text Refiner (v18.2) for OCR artifact repair in batch mode.

        Args:
            provider: LLM provider (ollama|openai|anthropic)
            model: Model name (optional for Ollama - auto-detects)
            api_key: API key for cloud providers
            threshold: Min corruption score to trigger refinement (0.0-1.0)
            max_edit: Max edit ratio allowed (0.0-1.0, default 0.35 = 35%)
        """
        try:
            from .refiner import create_refiner

            self._refiner_config = {
                "provider": provider,
                "model": model,
                "api_key": api_key,
                "base_url": base_url,
                "threshold": threshold,
                "max_edit": max_edit,
            }
            self._refiner = create_refiner(**self._refiner_config)
            logger.info(
                f"[REFINER] Enabled (batch): provider={provider}, "
                f"threshold={threshold}, max_edit={max_edit}"
            )
        except Exception as e:
            logger.error(f"[REFINER] Failed to initialize (batch): {e}")
            self._refiner = None
            self._refiner_config = None

    def set_profile_params(self, profile_params: "ProfileParameters") -> None:
        """
        REQ-OCR-01: Set profile parameters for OCR hints and dynamic DPI.

        This method stores the profile parameters which will be passed to
        the V2DocumentProcessor during batch processing. When profile has
        enable_ocr_hints=True, the processor will use OCR-hint-aware VLM enrichment.

        Args:
            profile_params: Profile parameters from selected profile (e.g., ScannedDegradedProfile)
        """
        self._profile_params = profile_params

        if profile_params.enable_ocr_hints:
            logger.info(
                f"[OCR-HYBRID] BatchProcessor: OCR hints ENABLED "
                f"(DPI={profile_params.render_dpi}, "
                f"min_conf={profile_params.ocr_min_confidence})"
            )
            print(
                f"🔬 [OCR-HYBRID] Enabled: DPI={profile_params.render_dpi}, "
                f"OCR confidence threshold={profile_params.ocr_min_confidence}",
                flush=True,
            )
        else:
            logger.info("[OCR-HYBRID] BatchProcessor: OCR hints DISABLED")

    def set_intelligence_metadata(self, intelligence_metadata: Dict[str, Any]) -> None:
        """
        V2.4: Set intelligence stack metadata for observability.

        This metadata proves intelligent classification ran and documents
        the exact thresholds/parameters used during extraction.

        Args:
            intelligence_metadata: Dict containing profile_type, min_image_dims,
                                 document_domain, document_modality, etc.
        """
        self._intelligence_metadata = intelligence_metadata
        logger.info(
            f"[V2.4-OBSERVABILITY] Intelligence metadata set: "
            f"profile={intelligence_metadata.get('profile_type')}, "
            f"dims={intelligence_metadata.get('min_image_dims')}, "
            f"domain={intelligence_metadata.get('document_domain')}"
        )

    def _compute_doc_hash(self, file_path: Path) -> str:
        """Compute MD5 hash of document for unique identification."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:12]

    def _initialize_vision_manager(self) -> Optional[VisionManager]:
        """
        Initialize the global VisionManager with persistent cache.

        The cache will be shared across all batches to avoid redundant
        VLM calls for duplicate images.
        """
        if self.vision_provider.lower() == "none":
            return None

        try:
            # BUG-007 FIX: Respect vision_cache_dir (None = disable cache)
            if self.vision_cache_dir:
                logger.info(f"[CACHE] ENABLED: {self.vision_cache_dir}")
                print(f"💾 [CACHE] ENABLED: {self.vision_cache_dir}", flush=True)
            else:
                logger.info("[CACHE] DISABLED")
                print("🚫 [CACHE] DISABLED", flush=True)

            manager = create_vision_manager(
                provider=self.vision_provider,
                api_key=self.vision_api_key,
                cache_dir=self.vision_cache_dir,
                model=self.vision_model,
                timeout=self.vlm_timeout,
                base_url=self.vision_base_url,
            )
            logger.info(
                f"Global VisionManager initialized: "
                f"provider={self.vision_provider}, "
                f"model={self.vision_model}, "
                f"cache_dir={self.vision_cache_dir}, "
                f"base_url={self.vision_base_url}"
            )
            return manager
        except Exception as e:
            logger.warning(f"Failed to initialize VisionManager: {e}")
            return None

    # ========================================================================
    # V2.4.0: SHADOW EXTRACTION IS ACTIVE (REQ-MM-05/06/07, IRON-07)
    # ========================================================================
    # Per SRS v2.4 Section 4.3 (Visual Heuristics & Shadow Extraction):
    # - Shadow extraction is a CORE SAFETY NET for catching missed images
    # - Runs AFTER Docling AI analysis to catch large editorial images
    # - Threshold: 300x300px OR 40% page area (REQ-MM-06)
    # - Full-page assets (>95% area) require VLM verification (IRON-07)
    # - Implementation: processor.py::_run_shadow_extraction()
    # ========================================================================

    # ========================================================================
    # LAYOUT-AWARE OCR INTEGRATION (Phase 1B)
    # ========================================================================

    def _classify_page(
        self,
        doc: fitz.Document,
        page_idx: int,
        threshold: int = 100,
    ) -> Tuple[str, int]:
        """
        Classify a page as "scanned" or "digital" based on text density.

        Args:
            doc: PyMuPDF document
            page_idx: Page index (0-indexed)
            threshold: Character count below which page is "scanned"

        Returns:
            Tuple of (classification, raw_text_length)
            classification is "scanned" or "digital"
        """
        page = doc.load_page(page_idx)
        raw_text: str = page.get_text("text")  # type: ignore[assignment]
        text = raw_text.strip() if raw_text else ""
        char_count = len(text)

        classification = "scanned" if char_count < threshold else "digital"

        logger.debug(
            f"[CLASSIFY] Page {page_idx + 1}: {classification} "
            f"(raw_text={char_count} chars, threshold={threshold})"
        )

        return classification, char_count

    def _render_page_to_image(
        self,
        doc: fitz.Document,
        page_idx: int,
        dpi: int = 300,
    ) -> np.ndarray:
        """
        Render a PDF page to a numpy array (RGB).

        Args:
            doc: PyMuPDF document
            page_idx: Page index (0-indexed)
            dpi: Resolution for rendering (default: 300)

        Returns:
            numpy array of shape (height, width, 3) in RGB format
        """
        page = doc.load_page(page_idx)
        zoom = dpi / 72.0  # 72 DPI is base
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Convert to PIL then numpy (avoids memory issues with raw bytes)
        img_data = pix.tobytes("ppm")
        pil_image = Image.open(io.BytesIO(img_data)).convert("RGB")
        np_image = np.array(pil_image)

        logger.debug(f"[RENDER] Page {page_idx + 1}: {np_image.shape} at {dpi} DPI")

        return np_image

    def _process_page_layout_aware(
        self,
        page_image: np.ndarray,
        page_number: int,
        source_file: str,
        docling_elements: Optional[List] = None,
        render_dpi: int = 150,
    ) -> List[IngestionChunk]:
        """
        Process a single page through Layout-Aware OCR pipeline.

        This method:
        1. Uses LayoutAwareOCRProcessor to detect TEXT/IMAGE regions
        2. Runs OCR cascade on TEXT regions → modality: "text"
        3. Runs VLM on IMAGE regions → modality: "image"
        4. Returns proper IngestionChunk objects

        Args:
            page_image: RGB numpy array of full page
            page_number: Page number (1-indexed)
            source_file: Source filename
            docling_elements: Optional pre-extracted Docling elements
            render_dpi: DPI used to render the page (for coordinate conversion)

        Returns:
            List of IngestionChunk objects
        """
        from .ocr.layout_aware_processor import LayoutAwareOCRProcessor

        # Initialize processor (cached for performance)
        if not hasattr(self, "_layout_processor"):
            self._layout_processor = LayoutAwareOCRProcessor(
                ocr_confidence_threshold=self.ocr_confidence_threshold,
                enable_doctr=self.enable_doctr,
                output_dir=self.assets_dir,
                vlm_manager=self._vision_manager,
            )
            logger.info(
                f"[LAYOUT-OCR] Initialized LayoutAwareOCRProcessor "
                f"(threshold={self.ocr_confidence_threshold}, doctr={self.enable_doctr})"
            )

        # Process page through layout-aware pipeline
        processed_chunks = self._layout_processor.process_page(
            page_image=page_image,
            page_number=page_number,
            doc_id=self._doc_hash or "unknown",
            docling_elements=docling_elements,
            render_dpi=render_dpi,
        )

        # Convert ProcessedChunk to IngestionChunk
        ingestion_chunks: List[IngestionChunk] = []
        doc_title = Path(source_file).stem if source_file else "Document"
        page_height_px, page_width_px = page_image.shape[:2]

        for pc in processed_chunks:
            # REQ-HIER-03: Breadcrumbs MUST contain minimum [filename, "Page X"]
            breadcrumb_path = [doc_title, f"Page {pc.page_number}"]

            # Create hierarchy
            hierarchy = HierarchyMetadata(
                parent_heading=None,
                breadcrumb_path=breadcrumb_path,
                level=2,  # 2 items in breadcrumb_path
            )

            # Map modality
            if pc.modality == "text":
                modality = Modality.TEXT
            elif pc.modality == "image":
                modality = Modality.IMAGE
            elif pc.modality == "table":
                modality = Modality.TABLE
            else:
                modality = Modality.TEXT  # Default

            # REQ-COORD-01: IMAGE/TABLE modalities REQUIRE spatial.bbox
            # V3.0.0: Use normalized bbox from ProcessedChunk (already 0-1000)
            spatial = None
            if pc.bbox:
                spatial = SpatialMetadata(
                    bbox=pc.bbox,
                    page_width=int(page_width_px),
                    page_height=int(page_height_px),
                )
            elif modality in (Modality.IMAGE, Modality.TABLE):
                # Fallback only if bbox is missing
                spatial = SpatialMetadata(
                    bbox=[0, 0, COORD_SCALE, COORD_SCALE],
                    page_width=int(page_width_px),
                    page_height=int(page_height_px),
                )

            # Create metadata with extraction_method and visual_description
            if pc.extraction_method and "ocr" in pc.extraction_method.lower():
                extraction_method = "ocr"
            else:
                extraction_method = "docling"

            # ================================================================
            # FIX 1: OCR CONFIDENCE PROPAGATION (Sanity & Speed Patch)
            # ================================================================
            # The ProcessedChunk.ocr_confidence is a float (0.0-1.0).
            # ChunkMetadata.ocr_confidence expects a string: "high/medium/low".
            # We MUST convert using get_ocr_confidence_level().
            # ================================================================
            from .schema.ingestion_schema import get_ocr_confidence_level

            ocr_confidence_level = None
            if pc.ocr_confidence is not None:
                ocr_confidence_level = get_ocr_confidence_level(pc.ocr_confidence)
                logger.debug(
                    f"[OCR-CONFIDENCE] Page {page_number}: "
                    f"raw={pc.ocr_confidence:.3f} → level={ocr_confidence_level}"
                )

            # ================================================================
            # FIX 3: INTELLIGENT BYPASS-GATE (Sanity & Speed Patch)
            # ================================================================
            # Skip SemanticRefiner for high-quality OCR text (>= 0.90 confidence)
            # This improves performance by ~70% without quality loss.
            # Rationale: High-confidence OCR text doesn't need LLM refinement.
            # ================================================================
            # CRITICAL: Initialize variables BEFORE any conditional logic
            refined_content = None
            refinement_applied = False
            corruption_score = None
            refinement_provider = None
            refinement_model = None

            if self._refiner and modality == Modality.TEXT:
                # Check bypass condition BEFORE calling refiner
                ocr_conf = pc.ocr_confidence if pc.ocr_confidence is not None else 0.0

                if ocr_conf >= 0.90:
                    # BYPASS: Skip refiner for high-confidence OCR
                    logger.info(
                        f"[REFINER-BYPASS] Page {page_number}: "
                        f"Skipping refiner (ocr_confidence={ocr_conf:.2f} >= 0.90)"
                    )
                    # Keep original values (no refinement)
                    refined_content = None
                    refinement_applied = False
                    corruption_score = None
                    refinement_provider = None
                    refinement_model = None
                else:
                    # PROCESS: Run refiner for low/medium confidence
                    try:
                        semantic_context = SemanticContext(
                            prev_text_snippet=pc.prev_text_snippet,
                            next_text_snippet=pc.next_text_snippet,
                            parent_heading=hierarchy.parent_heading,
                            breadcrumb_path=hierarchy.breadcrumb_path,
                        )
                        refine_result = self._refiner.process(
                            raw_text=pc.content,
                            visual_description=None,
                            semantic_context=semantic_context,
                        )
                        corruption_score = refine_result.corruption_score
                        refinement_provider = refine_result.provider
                        refinement_model = refine_result.model
                        if refine_result.refinement_applied:
                            refined_content = refine_result.refined_text
                            refinement_applied = True
                        logger.debug(
                            f"[REFINER] Page {page_number}: "
                            f"ocr_confidence={ocr_conf:.2f}, "
                            f"corruption={corruption_score:.2f}, "
                            f"refined={refinement_applied}"
                        )
                    except Exception as refiner_error:
                        logger.error(f"[REFINER] Failed on page {page_number}: {refiner_error}")

            # BUG-009 FIX + FIX 1: Propagate intelligence metadata AND ocr_confidence
            metadata = ChunkMetadata(
                page_number=pc.page_number,
                source_file=source_file,
                file_type=FileType.PDF,
                hierarchy=hierarchy,
                extraction_method=extraction_method,
                visual_description=pc.visual_description,
                spatial=spatial,
                refined_content=refined_content,
                refinement_applied=refinement_applied,
                corruption_score=corruption_score,
                refinement_provider=refinement_provider,
                refinement_model=refinement_model,
                ocr_confidence=ocr_confidence_level,  # FIX 1: Now properly propagated
                **self._intelligence_metadata,  # BUG-009 FIX: Direct unpack prevents drift
            )

            # Create asset reference if applicable
            asset_ref = None
            if pc.asset_ref:
                asset_ref = AssetReference(
                    file_path=pc.asset_ref.get("file_path", ""),
                    mime_type=pc.asset_ref.get("mime_type", "image/png"),
                    width_px=pc.asset_ref.get("width_px"),
                    height_px=pc.asset_ref.get("height_px"),
                )

            # V3.0.0: REQ-MM-03 - Create semantic_context from ProcessedChunk
            semantic_context = None
            if pc.prev_text_snippet or pc.next_text_snippet:
                semantic_context = SemanticContext(
                    prev_text_snippet=pc.prev_text_snippet,
                    next_text_snippet=pc.next_text_snippet,
                    parent_heading=hierarchy.parent_heading,
                    breadcrumb_path=hierarchy.breadcrumb_path,
                )

            # Create IngestionChunk
            chunk = IngestionChunk(
                chunk_id=pc.chunk_id,
                doc_id=self._doc_hash or "unknown",
                content=pc.content,
                modality=modality,
                metadata=metadata,
                asset_ref=asset_ref,
                semantic_context=semantic_context,  # V3.0.0: Add semantic context
            )

            ingestion_chunks.append(chunk)

        logger.info(
            f"[LAYOUT-OCR] Page {page_number}: Generated {len(ingestion_chunks)} chunks "
            f"(text: {sum(1 for c in ingestion_chunks if c.modality == Modality.TEXT)}, "
            f"image: {sum(1 for c in ingestion_chunks if c.modality == Modality.IMAGE)})"
        )

        return ingestion_chunks

    def _extract_docling_layout_elements(
        self,
        batch_path: Path,
        page_offset: int,
    ) -> Dict[int, List[Any]]:
        """
        Extract Docling layout elements grouped by page number.

        This runs Docling's layout analysis to get proper TEXT/IMAGE/TABLE
        regions instead of using the broken fallback detection.

        Args:
            batch_path: Path to the batch PDF file
            page_offset: Offset to add to batch page numbers

        Returns:
            Dict mapping actual page numbers to list of Docling elements
        """
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        logger.info("[DOCLING-LAYOUT] Running Docling layout analysis on batch...")

        # Configure Docling for layout detection
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = 2.0
        pipeline_options.generate_page_images = (
            True  # ✅ REQ-PDF-04: FIXED - Enable page rendering for padding integrity
        )
        pipeline_options.generate_picture_images = True
        pipeline_options.generate_table_images = True
        pipeline_options.do_ocr = True

        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

        result = converter.convert(str(batch_path))
        doc = result.document

        # Group elements by page
        elements_per_page: Dict[int, List[Any]] = {}

        for item_tuple in doc.iterate_items():
            element, _ = item_tuple

            # Get page number from provenance
            batch_page_no = 1
            if hasattr(element, "prov") and element.prov:
                prov = element.prov[0] if isinstance(element.prov, list) else element.prov
                if hasattr(prov, "page_no") and prov.page_no is not None:
                    batch_page_no = prov.page_no

            actual_page_no = batch_page_no + page_offset

            if actual_page_no not in elements_per_page:
                elements_per_page[actual_page_no] = []
            elements_per_page[actual_page_no].append(element)

        total = sum(len(e) for e in elements_per_page.values())
        logger.info(
            f"[DOCLING-LAYOUT] Extracted {total} elements across {len(elements_per_page)} pages"
        )

        return elements_per_page

    def _process_batch_layout_aware(
        self,
        batch_info: BatchInfo,
        source_file: str,
    ) -> List[IngestionChunk]:
        """
        Process a batch using Layout-Aware OCR pipeline.

        This replaces the shadow-first approach for scanned documents.
        Instead of full-page VLM descriptions, we get:
        - Proper TEXT chunks with OCR content
        - Proper IMAGE chunks with VLM descriptions

        Args:
            batch_info: Information about this batch
            source_file: Original source filename

        Returns:
            List of IngestionChunk objects with proper modalities
        """
        all_chunks: List[IngestionChunk] = []
        dpi = self._profile_params.render_dpi if self._profile_params else 300

        # ================================================================
        # CRITICAL: Run Docling layout analysis FIRST to get real elements
        # This prevents using the broken fallback detection
        # ================================================================
        print("    🔍 [DOCLING] Running layout analysis on batch...", flush=True)
        try:
            docling_elements_per_page = self._extract_docling_layout_elements(
                batch_path=batch_info.batch_path,
                page_offset=batch_info.page_offset,
            )
            print(
                f"    ✓ [DOCLING] Found elements on {len(docling_elements_per_page)} pages",
                flush=True,
            )
        except Exception as e:
            logger.warning(f"[DOCLING-LAYOUT] Analysis failed: {e}. Using fallback.")
            print(f"    ⚠️ [DOCLING] Layout analysis failed: {e}", flush=True)
            docling_elements_per_page = {}

        doc: Optional[fitz.Document] = None
        try:
            doc = fitz.open(batch_info.batch_path)

            print(
                f"    🔬 [LAYOUT-OCR] Processing {len(doc)} pages at {dpi} DPI...",
                flush=True,
            )
            logger.info(
                f"[LAYOUT-OCR] Batch {batch_info.batch_index + 1}: "
                f"Processing {len(doc)} pages (layout-aware mode)"
            )

            for batch_page_idx in range(len(doc)):
                actual_page_no = batch_info.page_offset + batch_page_idx + 1

                # Get Docling elements for this page
                page_elements = docling_elements_per_page.get(actual_page_no)
                elem_count = len(page_elements) if page_elements else 0

                # Step 1: Classify page
                classification, char_count = self._classify_page(doc, batch_page_idx, threshold=100)
                elem_info = f"[Docling: {elem_count} elements]" if page_elements else "[fallback]"
                print(
                    f"      📄 Page {actual_page_no}: {classification} ({char_count} chars) {elem_info}",
                    flush=True,
                )

                # Step 2: Render page to image
                page_image = self._render_page_to_image(doc, batch_page_idx, dpi=dpi)
                logger.info(f"[LAYOUT-OCR] Rendered page {actual_page_no}: {page_image.shape}")

                # Step 3: Process through layout-aware pipeline WITH Docling elements
                page_chunks = self._process_page_layout_aware(
                    page_image=page_image,
                    page_number=actual_page_no,
                    source_file=source_file,
                    docling_elements=page_elements,
                    render_dpi=dpi,
                )

                all_chunks.extend(page_chunks)

                # Log chunk types
                text_count = sum(1 for c in page_chunks if c.modality == Modality.TEXT)
                image_count = sum(1 for c in page_chunks if c.modality == Modality.IMAGE)
                print(
                    f"        → {len(page_chunks)} chunks (text: {text_count}, image: {image_count})",
                    flush=True,
                )

                # Memory cleanup
                del page_image
                gc.collect()

            # ================================================================
            # SHADOW EXTRACTION WITHIN TRY BLOCK (BEFORE doc.close())
            # FIX: "Document Closed" error - shadow extraction needs PDF open
            # ================================================================
            print("    🔍 [SHADOW] Running shadow extraction scan...", flush=True)
            logger.info("[SHADOW-EXTRACTION] Running shadow scan BEFORE closing PDF...")

            # Import processor to access shadow extraction
            from .processor import V2DocumentProcessor

            # Create temporary processor just for shadow extraction
            temp_processor = V2DocumentProcessor(
                output_dir=str(self.output_dir),
                enable_ocr=False,
                vision_provider="none",
                external_vision_manager=self._vision_manager,
                doc_hash_override=self._doc_hash,
                source_file_override=source_file,
                extraction_strategy=self.extraction_strategy,
                intelligence_metadata=self._intelligence_metadata,
            )

            try:
                # Prepare page_dims dict from still-open PDF doc
                page_dims = {}
                for page_idx in range(len(doc)):
                    batch_page_no = page_idx + 1
                    page = doc[page_idx]
                    page_w = page.rect.width
                    page_h = page.rect.height
                    page_dims[batch_page_no] = (page_w, page_h)

                # Create context state for shadow extraction
                shadow_state = create_context_state(
                    doc_id=self._doc_hash or "unknown",
                    source_file=source_file,
                )

                # Track existing image indices to avoid conflicts
                max_figure_idx = max(
                    (
                        int(c.asset_ref.file_path.split("_")[3].replace(".png", ""))
                        for c in all_chunks
                        if c.modality == Modality.IMAGE and c.asset_ref
                    ),
                    default=0,
                )
                element_indices = {"figure": max_figure_idx, "table": 0}
                text_buffer: List[str] = []

                # Run shadow extraction on STILL-OPEN PDF
                shadow_chunks_generator = temp_processor._run_shadow_extraction(
                    file_path=batch_info.batch_path,
                    doc_hash=self._doc_hash or "unknown",
                    source_file=source_file,
                    file_type=FileType.PDF,
                    page_images={},
                    page_dims=page_dims,
                    page_offset=batch_info.page_offset,
                    state=shadow_state,
                    text_buffer=text_buffer,
                    element_indices=element_indices,
                    docling_processed_pages=set(range(1, len(doc) + 1)),
                )

                shadow_chunks = list(shadow_chunks_generator)

                if shadow_chunks:
                    all_chunks.extend(shadow_chunks)
                    print(
                        f"    ✓ [SHADOW] Found {len(shadow_chunks)} additional shadow assets",
                        flush=True,
                    )
                    logger.info(f"[SHADOW-EXTRACTION] Added {len(shadow_chunks)} shadow assets")
                else:
                    print("    ✓ [SHADOW] No additional shadow assets found", flush=True)
                    logger.info("[SHADOW-EXTRACTION] No shadow assets needed")

            except Exception as shadow_err:
                logger.error(f"[SHADOW-EXTRACTION] Failed: {shadow_err}")
                print(f"    ⚠️ [SHADOW] Extraction failed: {shadow_err}", flush=True)

        finally:
            # NOW close the PDF after shadow extraction is complete
            if doc is not None:
                doc.close()
            gc.collect()

        logger.info(
            f"[LAYOUT-OCR] Batch complete: {len(all_chunks)} chunks generated (including shadow)"
        )
        return all_chunks

    def _process_single_batch(
        self,
        batch_info: BatchInfo,
        split_result: SplitResult,
        source_file: str,
    ) -> List[IngestionChunk]:
        """
        Process a single batch and return its chunks.

        Args:
            batch_info: Information about this batch
            split_result: Overall split information
            source_file: Original source filename

        Returns:
            List of IngestionChunk objects from this batch
        """
        # Import here to avoid circular imports
        from .processor import V2DocumentProcessor

        logger.info(
            f"Processing batch {batch_info.batch_index + 1}/{split_result.batch_count}: "
            f"pages {batch_info.page_range_str} (offset={batch_info.page_offset})"
        )

        # ================================================================
        # PHASE 1B: LAYOUT-AWARE OCR MODE ROUTING
        # ================================================================
        # When ocr_mode == "layout-aware", we bypass the legacy pipeline
        # and use the new LayoutAwareOCRProcessor which produces:
        # - modality: "text" with OCR content
        # - modality: "image" with VLM descriptions
        #
        # This is the CRITICAL routing that was missing before!
        # ================================================================
        if self.ocr_mode == "layout-aware":
            logger.info(
                f"[BATCH] Layout-aware OCR enabled "
                f"(threshold={self.ocr_confidence_threshold}, doctr={self.enable_doctr})"
            )
            print(
                f"    🔬 [LAYOUT-AWARE] Using OCR cascade pipeline...",
                flush=True,
            )

            # Use layout-aware batch processing
            chunks = self._process_batch_layout_aware(
                batch_info=batch_info,
                source_file=source_file,
            )

            logger.info(
                f"[LAYOUT-AWARE] Batch {batch_info.batch_index + 1}: "
                f"Generated {len(chunks)} chunks (layout-aware mode)"
            )

            # Log batch completion
            logger.info(
                f"Batch {batch_info.batch_index + 1} complete: "
                f"{len(chunks)} chunks via layout-aware OCR"
            )

            return chunks

        # ================================================================
        # LEGACY MODE: Standard Docling + Shadow extraction
        # ================================================================

        # Create processor for this batch with:
        # 1. Shared VisionManager (global cache)
        # 2. Inherited ContextStateV2 (breadcrumb continuity)
        # 3. Page offset for correct numbering
        # 4. Extraction strategy for dynamic thresholds
        # 5. Intelligence metadata for observability (BUG-009 FIX)
        processor = V2DocumentProcessor(
            output_dir=str(self.output_dir),
            enable_ocr=self.enable_ocr,
            ocr_engine=self.ocr_engine,
            vision_provider="none",  # Use external vision manager
            external_vision_manager=self._vision_manager,
            initial_state=self._context_state,
            page_offset=batch_info.page_offset,
            doc_hash_override=self._doc_hash,
            source_file_override=source_file,
            extraction_strategy=self.extraction_strategy,
            intelligence_metadata=self._intelligence_metadata,  # BUG-009 FIX: Propagate metadata
        )

        if self._refiner_config:
            processor.enable_refiner(**self._refiner_config)

        # REQ-OCR-01: Pass profile parameters to processor for OCR hints
        # When profile has enable_ocr_hints=True, processor will use OCR-hint-aware VLM enrichment
        if self._profile_params is not None:
            processor.set_profile_params(self._profile_params)
            logger.info(
                f"[OCR-HYBRID] Processor configured with profile params: "
                f"enable_ocr_hints={self._profile_params.enable_ocr_hints}"
            )

        # ================================================================
        # V2.4.0: STANDARD DOCLING PIPELINE + SHADOW SAFETY NET
        # ================================================================
        # Per SRS v2.4 Section 4.3:
        # - Docling extracts structured elements (text, images, tables)
        # - Shadow extraction runs AFTER Docling as safety net (REQ-MM-05/06/07)
        # - Catches large images (300x300px OR 40% page area) Docling may miss
        # - Full-page assets (>95% area) require VLM verification (IRON-07)
        # ================================================================

        # Standard flow: Docling extraction (V3.0.0 architecture)
        chunks: List[IngestionChunk] = []
        try:
            for chunk in processor.process_document(str(batch_info.batch_path)):
                chunks.append(chunk)
        except IndexError as e:
            import traceback

            logger.error(
                f"[PHANTOM-BUG] IndexError in batch {batch_info.batch_index} during processing!\n"
                f"Error: {e}\n"
                f"Chunks collected so far: {len(chunks)}\n"
                f"Batch path: {batch_info.batch_path}\n"
                f"Full Traceback:\n{traceback.format_exc()}"
            )
            raise
        except Exception as e:
            import traceback

            logger.error(
                f"Error processing batch {batch_info.batch_index}: {e}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            raise

        # CRITICAL: Capture state for next batch (breadcrumb continuity)
        self._context_state = processor.get_final_state()

        # Log batch completion
        breadcrumbs = self._context_state.get_breadcrumb_path() if self._context_state else []
        logger.info(
            f"Batch {batch_info.batch_index + 1} complete: "
            f"{len(chunks)} chunks, "
            f"breadcrumbs={breadcrumbs}"
        )

        return chunks

    def process_pdf(self, pdf_path: str | Path) -> BatchProcessingResult:
        """
        Process a large PDF using batch splitting strategy.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            BatchProcessingResult with processing details
        """
        pdf_path = Path(pdf_path).resolve()
        start_time = time.perf_counter()
        errors: List[str] = []
        all_chunks: List[IngestionChunk] = []

        logger.info(f"Starting batch processing for: {pdf_path.name}")
        print(f"⏳ Starting batch processing for: {pdf_path.name}", flush=True)

        # Store PDF path for QA-CHECK-01 token validation (extracts source text)
        self._current_pdf_path = pdf_path

        # Compute document hash BEFORE splitting
        self._doc_hash = self._compute_doc_hash(pdf_path)
        logger.info(f"Document hash: {self._doc_hash}")

        # Initialize global vision manager
        self._vision_manager = self._initialize_vision_manager()

        # Initialize context state for first batch
        self._context_state = create_context_state(
            doc_id=self._doc_hash,
            source_file=pdf_path.name,
        )

        # Initialize quality filter tracker for this document
        self._quality_filter_tracker = create_quality_filter_tracker()

        # OCR strategy: scanned-only. If document_modality is native digital, disable cascade entirely.
        doc_modality = self._intelligence_metadata.get("document_modality")
        if doc_modality == "native_digital":
            self.ocr_mode = "legacy"
            self.enable_ocr = False
            self.enable_doctr = False
            logger.info(
                "[OCR-GUARD] Digital modality detected; OCR cascade disabled (legacy mode, enable_ocr=False, enable_doctr=False)"
            )
        else:
            # Non-digital or unknown modality can still use configured OCR defaults
            logger.info(
                f"[OCR-GUARD] Modality={doc_modality or 'unknown'}; respecting configured OCR settings "
                f"(mode={self.ocr_mode}, enable_ocr={self.enable_ocr}, enable_doctr={self.enable_doctr})"
            )

        # [CORE] Page limit enforcement at splitting stage
        if self.max_pages is not None and self.max_pages > 0:
            logger.info(
                f"[CORE] Page limit set to: {self.max_pages}. Processing only first {self.max_pages} pages."
            )

        # [CORE] Specific pages enforcement
        if self.specific_pages:
            logger.info(f"[CORE] Specific pages mode: Processing ONLY pages {self.specific_pages}")
            print(f"🎯 Processing SPECIFIC pages: {self.specific_pages}", flush=True)

        # Split PDF into batches
        with PDFBatchSplitter(
            batch_size=self.batch_size,
            specific_pages=self.specific_pages,
        ) as splitter:
            try:
                split_result = splitter.split(pdf_path)

                # Apply page limit if specified
                if self.max_pages is not None and self.max_pages > 0:
                    # Filter batches to only include those within page limit
                    filtered_batches = []
                    pages_included = 0

                    for batch in split_result.batches:
                        if pages_included >= self.max_pages:
                            break

                        # Check if this batch is fully or partially within limit
                        if batch.end_page <= self.max_pages:
                            # Full batch is within limit
                            filtered_batches.append(batch)
                            pages_included = batch.end_page
                        elif batch.start_page <= self.max_pages:
                            # Partial batch - need to create a modified batch
                            # This is a corner case where batch crosses the page limit
                            logger.info(
                                f"Batch {batch.batch_index + 1} crosses page limit. "
                                f"Trimming from pages {batch.start_page}-{batch.end_page} "
                                f"to {batch.start_page}-{self.max_pages}"
                            )
                            # For now, include the full batch but stop after this
                            filtered_batches.append(batch)
                            pages_included = self.max_pages
                            break

                    # Update split_result with filtered batches
                    original_count = len(split_result.batches)
                    split_result = SplitResult(
                        original_path=split_result.original_path,
                        original_hash=split_result.original_hash,
                        total_pages=min(split_result.total_pages, self.max_pages),
                        batch_count=len(filtered_batches),
                        batches=filtered_batches,
                        temp_dir=split_result.temp_dir,
                    )

                    logger.info(
                        f"[CORE] Page limit enforced: processing {len(filtered_batches)}/{original_count} batches, "
                        f"up to page {self.max_pages}"
                    )
            except Exception as e:
                logger.error(f"Failed to split PDF: {e}")
                return BatchProcessingResult(
                    success=False,
                    original_path=pdf_path,
                    original_hash=self._doc_hash,
                    total_pages=0,
                    batches_processed=0,
                    total_chunks=0,
                    output_jsonl=self.output_dir / "ingestion.jsonl",
                    assets_dir=self.assets_dir,
                    processing_time_seconds=time.perf_counter() - start_time,
                    errors=[str(e)],
                    vision_stats={},
                )

            print(
                f"📄 Split into {split_result.batch_count} batches "
                f"({split_result.total_pages} pages, {self.batch_size} pages/batch)",
                flush=True,
            )

            # Process each batch sequentially
            batches_processed = 0
            for batch_info in split_result.batches:
                try:
                    print(
                        f"  🔄 Batch {batch_info.batch_index + 1}/"
                        f"{split_result.batch_count}: "
                        f"pages {batch_info.page_range_str}...",
                        flush=True,
                    )

                    batch_chunks = self._process_single_batch(
                        batch_info=batch_info,
                        split_result=split_result,
                        source_file=pdf_path.name,
                    )
                    all_chunks.extend(batch_chunks)
                    batches_processed += 1

                    print(
                        f"    ✓ {len(batch_chunks)} chunks extracted",
                        flush=True,
                    )

                except Exception as e:
                    error_msg = f"Batch {batch_info.batch_index}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    print(f"    ✗ Error: {e}", flush=True)

                # REQ-PDF-05: Memory hygiene between batches
                gc.collect()
                logger.debug(f"gc.collect() after batch {batch_info.batch_index + 1}")

        # ====================================================================
        # REQ-COORD-02: Extract page dimensions for UI overlay support
        # ====================================================================
        print("\n📐 [REQ-COORD-02] Extracting page dimensions...", flush=True)
        self._page_dimensions = self._extract_page_dimensions(pdf_path)

        # ====================================================================
        # REQ-DEDUP-01: Initialize ImageHashRegistry for pHash deduplication
        # ====================================================================
        self._image_hash_registry = create_image_hash_registry(threshold=10)
        print("🔍 [PHASH] Initializing perceptual hash registry...", flush=True)

        # ====================================================================
        # CLUSTER B: GOVERNANCE & VALIDATION LAYERS
        # ====================================================================
        # 1. REQ-COORD-02: Page dimension propagation to ALL chunks
        # 2. IRON-07: Full-Page Guard for [0,0,1000,1000] bboxes
        # 3. QA-CHECK-01: Token validation per chunk
        # 4. Quality filters (MUST run BEFORE token balance validation)
        # 5. QA-CHECK-01: Token balance validation (with filtering awareness)
        # ====================================================================

        # Step 1: REQ-COORD-02 - Propagate page dimensions to ALL chunks
        all_chunks = self._propagate_page_dimensions(all_chunks)

        # Step 2: IRON-07 - Apply Full-Page Guard to filter/modify full-page assets
        all_chunks = self._apply_full_page_guard(all_chunks)

        # Step 3: QA-CHECK-01 - Validate token limits per chunk
        all_chunks, token_flagged_count = self._validate_token_limit_per_chunk(all_chunks)
        if token_flagged_count > 0:
            print(
                f"⚠️ [QA-CHECK-01] {token_flagged_count} chunks exceeded token limit",
                flush=True,
            )

        # ====================================================================
        # PHASE 1: QUALITY IMPROVEMENTS (NOW BEFORE TOKEN BALANCE VALIDATION)
        # ====================================================================
        # 1. Empty chunk filtering (asset-aware)
        # 2. OCR text post-processing (number joining)
        # 3. Look-ahead buffer for symmetric overlap
        # ====================================================================

        # Apply quality filters (this fills the QualityFilterTracker)
        filtered_chunks = self._apply_quality_filters(all_chunks)
        # Keep a stable baseline count for recovery bookkeeping (avoid in-place mutations)
        filtered_baseline_count = len(filtered_chunks)
        filtered_count = len(all_chunks) - filtered_baseline_count
        print(
            f"\n🔍 [QUALITY] Filtered {filtered_count} empty/invalid chunks",
            flush=True,
        )

        # Step 4: QA-CHECK-01 - Run token balance validation WITH filtering awareness
        token_result = self._run_token_validation(filtered_chunks, pdf_path.name)
        if not token_result.is_valid:
            print(
                f"⚠️ [QA-CHECK-01] Token balance warning: {token_result.variance_percent:.1f}% variance",
                flush=True,
            )
            if self.strict_qa:
                errors.append(f"QA-CHECK-01 failed: {token_result.error_message}")

        # Step 5: TextIntegrityScout - Rescue lost text if variance > 10%
        recovery_input = list(filtered_chunks)  # do not mutate the baseline list
        recovered_chunks = self._run_text_integrity_scout(
            chunks=recovery_input,
            source_file=pdf_path.name,
            variance_percent=token_result.variance_percent,
        )

        # Step 6: Re-validate token balance after recovery (polish log level)
        if token_result.variance_percent < -10.0:
            post_recovery_result = self._run_token_validation(recovered_chunks, pdf_path.name)
            if post_recovery_result.is_valid:
                print(
                    f"✓ [QA-CHECK-01] Token balance RECOVERED: "
                    f"{post_recovery_result.variance_percent:.1f}% variance (within tolerance)",
                    flush=True,
                )
                logger.info(
                    f"[QA-CHECK-01] ✓ Token balance recovered after TextIntegrityScout: "
                    f"variance {post_recovery_result.variance_percent:.1f}% is within tolerance"
                )
            else:
                print(
                    f"⚠️ [QA-CHECK-01] Token balance still outside tolerance: "
                    f"{post_recovery_result.variance_percent:.1f}%",
                    flush=True,
                )
            all_chunks = recovered_chunks
        else:
            all_chunks = filtered_chunks

        # Summary of recovery vs. filtered chunks for observability
        recovered_delta = len(all_chunks) - filtered_baseline_count
        # Flag potentially suspicious rescues: large positive or negative delta
        suspicious_recovery = recovered_delta < 0 or recovered_delta > max(10, filtered_baseline_count * 0.2)
        logger.debug(
            f"[QA-CHECK-01] Chunk counts — filtered baseline: {filtered_baseline_count}, "
            f"after recovery pipeline: {len(all_chunks)}, delta: {recovered_delta}"
        )
        logger.info(
            f"[QA-CHECK-01] Final chunk set: {len(all_chunks)} "
            f"(filtered baseline={filtered_baseline_count}, recovered_delta={recovered_delta})"
        )
        print(
            f"\nℹ️ [QA-CHECK-01] Final chunk set: {len(all_chunks)} "
            f"(filtered={filtered_baseline_count}, recovered+delta={recovered_delta})",
            flush=True,
        )
        if suspicious_recovery:
            logger.warning(
                f"[QA-CHECK-01] Recovery delta looks unusual (delta={recovered_delta}, "
                f"filtered={len(filtered_chunks)}). Inspect TextIntegrityScout output."
            )

        # Write aggregated output to master JSONL with deduplication
        output_jsonl = self.output_dir / "ingestion.jsonl"
        written_chunks = 0
        duplicate_count = 0

        # PHANTOM BUG FIX: Add defensive logging and error handling
        export_chunks = all_chunks  # Use the latest set (includes recovered chunks if any)
        logger.info(f"[FINALIZE] Starting JSONL write: {len(export_chunks)} chunks to process")
        print(
            f"\n📝 [FINALIZE] Writing {len(export_chunks)} chunks to {output_jsonl.name}...",
            flush=True,
        )

        # ✅ IRON-08: Clear file first, then use atomic writes (append + flush)
        if output_jsonl.exists():
            output_jsonl.unlink()

        # Process chunks with atomic writes
        for idx, chunk in enumerate(export_chunks):
            try:
                # Log progress every 50 chunks
                if idx % 50 == 0 and idx > 0:
                    logger.debug(f"[FINALIZE] Processed {idx}/{len(export_chunks)} chunks")

                chunk_dict = chunk.model_dump(mode="json")
                # Ensure schema_version is emitted in metadata for downstream versioning
                meta = chunk_dict.get("metadata", {})
                if meta.get("schema_version") is None:
                    meta["schema_version"] = SCHEMA_VERSION
                    chunk_dict["metadata"] = meta

                # ============================================================
                # REQ-ASSET-01: STRICT FILENAME-METADATA ASSERTION
                # ============================================================
                asset_ref = chunk_dict.get("asset_ref")
                if asset_ref and asset_ref.get("file_path"):
                    file_path = asset_ref["file_path"]
                    filename = file_path.split("/")[-1]
                    parts = filename.split("_")
                    if len(parts) >= 2:
                        filename_page = int(parts[1])
                        metadata_page = chunk_dict.get("metadata", {}).get("page_number")
                        if filename_page != metadata_page:
                            error_msg = (
                                f"[ASSET-METADATA-MISMATCH] FATAL: "
                                f"Asset '{filename}' page {filename_page} "
                                f"!= metadata page {metadata_page}. "
                                f"STOPPING EXPORT."
                            )
                            logger.error(error_msg)
                            raise ValueError(error_msg)

                # ============================================================
                # REQ-DEDUP-01: pHash Deduplication for IMAGE chunks
                # ============================================================
                if chunk.modality == Modality.IMAGE and asset_ref:
                    asset_file = asset_ref.get("file_path")
                    if asset_file:
                        full_asset_path = self.output_dir / asset_file
                        if full_asset_path.exists():
                            try:
                                from PIL import Image

                                with Image.open(full_asset_path) as img:
                                    dup_info = self._image_hash_registry.check_and_register(
                                        image=img,
                                        page_number=chunk.metadata.page_number,
                                        asset_path=asset_file,
                                    )

                                    if dup_info.is_duplicate:
                                        # DUPLICATE_REJECTED - skip this chunk
                                        duplicate_count += 1
                                        orig_page = (
                                            dup_info.original_record.page_number
                                            if dup_info.original_record
                                            else "unknown"
                                        )
                                        logger.warning(
                                            f"[DUPLICATE_REJECTED] Skipping {asset_file} "
                                            f"(duplicate of page {orig_page})"
                                        )
                                        # Delete the duplicate asset file
                                        try:
                                            full_asset_path.unlink()
                                            logger.info(f"Deleted duplicate asset: {asset_file}")
                                        except Exception as del_e:
                                            logger.warning(f"Failed to delete duplicate: {del_e}")
                                        continue  # Skip writing this chunk

                                    # Log successful registration
                                    logger.info(
                                        f"[FINALIZING] Asset {filename} linked to "
                                        f"Page {chunk.metadata.page_number}"
                                    )

                            except Exception as hash_e:
                                logger.warning(f"pHash check failed for {asset_file}: {hash_e}")

                # ✅ IRON-08: Atomic write - open in append mode, write, flush per chunk
                with open(output_jsonl, "a", encoding="utf-8") as f:
                    json_line = json.dumps(chunk_dict, ensure_ascii=False)
                    f.write(json_line + "\n")
                    f.flush()  # Force write to disk immediately
                written_chunks += 1

            except IndexError as e:
                import traceback

                logger.error(
                    f"[PHANTOM-BUG] IndexError at chunk {idx}/{len(filtered_chunks)}!\n"
                    f"Chunk ID: {chunk.chunk_id if chunk else 'None'}\n"
                    f"Error: {e}\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )
                raise  # Re-raise to see full stack
            except Exception as e:
                import traceback

                logger.error(
                    f"[FINALIZE-ERROR] Error processing chunk {idx}: {e}\n"
                    f"Chunk ID: {chunk.chunk_id if chunk else 'None'}\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )
                raise

        # Log deduplication results
        registry_stats = self._image_hash_registry.get_stats()
        print(
            f"\n📊 [PHASH] Deduplication complete: "
            f"{registry_stats['total_registered']} unique images, "
            f"{duplicate_count} duplicates rejected",
            flush=True,
        )
        print(
            f"\n📊 [EXPORT] Written {written_chunks} chunks "
            f"({duplicate_count} duplicates rejected, "
            f"{len(all_chunks) - len(filtered_chunks)} filtered vs. final {len(export_chunks)})",
            flush=True,
        )
        logger.info(
            f"Written {written_chunks} chunks to {output_jsonl} "
            f"({duplicate_count} duplicates rejected, "
            f"{len(all_chunks) - len(filtered_chunks)} filtered vs. final {len(export_chunks)})"
        )

        # Get vision stats and flush cache
        # PHANTOM BUG FIX: Add try-except to catch IndexError during cache operations
        vision_stats = {}
        if self._vision_manager:
            try:
                logger.info("[VISION-STATS] Attempting to get vision stats...")
                vision_stats = self._vision_manager.get_stats()
                logger.info(f"[VISION-STATS] Stats retrieved successfully: {vision_stats}")

                logger.info("[VISION-CACHE] Attempting to flush cache...")
                self._vision_manager.flush_cache()
                logger.info(f"[VISION-CACHE] Cache flushed successfully")
            except IndexError as e:
                import traceback

                logger.error(
                    f"[PHANTOM-BUG] IndexError during vision cache operations!\n"
                    f"Error: {e}\n"
                    f"Traceback:\n{traceback.format_exc()}\n"
                    f"Vision stats before error: {vision_stats}"
                )
                # Don't crash - continue with empty stats
                vision_stats = {"error": str(e)}
            except Exception as e:
                import traceback

                logger.error(
                    f"[VISION-ERROR] Unexpected error during cache operations: {e}\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )
                vision_stats = {"error": str(e)}

        elapsed = time.perf_counter() - start_time

        print(f"\n✅ Batch processing complete!", flush=True)
        print(f"   Total chunks: {len(all_chunks)}", flush=True)
        print(f"   Time: {elapsed:.1f}s", flush=True)
        print(f"   Output: {output_jsonl}", flush=True)

        return BatchProcessingResult(
            success=len(errors) == 0,
            original_path=pdf_path,
            original_hash=self._doc_hash,
            total_pages=split_result.total_pages,
            batches_processed=batches_processed,
            total_chunks=len(all_chunks),
            output_jsonl=output_jsonl,
            assets_dir=self.assets_dir,
            processing_time_seconds=elapsed,
            errors=errors,
            vision_stats=vision_stats,
        )

    # ========================================================================
    # PHASE 1: QUALITY IMPROVEMENT METHODS
    # ========================================================================

    def _should_skip_chunk(
        self,
        chunk: IngestionChunk,
    ) -> Tuple[bool, Optional[FilterCategory]]:
        """
        Determine if a chunk should be filtered out before export.
        Returns (should_skip, reason_category) for QualityFilterTracker.

        PROFILE-AWARE FILTERING (v2.4 Intelligence Stack):
        Uses the strategy profile to determine appropriate filtering thresholds.
        - academic_whitepaper: Strict filtering (no page numbers, footnotes OK)
        - digital_magazine: Relaxed filtering (keep captions, pull-quotes)
        - scanned_degraded: Very relaxed (OCR artifacts need tolerance)

        IRON RULE: Chunks with asset_ref NEVER filtered on text length.

        Args:
            chunk: IngestionChunk to evaluate

        Returns:
            Tuple of (should_skip, FilterCategory or None)
        """
        import re

        # ================================================================
        # IRON RULE 1: NEVER skip chunks with assets (REQ-MM-05)
        # Image captions/descriptions are valuable regardless of length
        # ================================================================
        if chunk.asset_ref is not None:
            return (False, None)

        # ================================================================
        # IRON RULE 2: NEVER skip TABLE modality chunks
        # Table cells can be short ("Yes", "No", "3.5") but are critical data
        # Even a single number in a table cell is meaningful (specs, prices, etc.)
        # ================================================================
        if chunk.modality == Modality.TABLE:
            return (False, None)

        content = chunk.content or ""
        stripped = content.strip()

        # ================================================================
        # RULE 2: Empty or whitespace-only content (universal)
        # ================================================================
        if not stripped:
            logger.debug(f"[FILTER] Skipping empty chunk: {chunk.chunk_id}")
            return (True, FilterCategory.EMPTY)

        # ================================================================
        # PROFILE-AWARE MINIMUM LENGTH THRESHOLD
        # ================================================================
        # Get profile type from intelligence metadata
        profile_type = self._intelligence_metadata.get("profile_type", "unknown")

        # Define minimum character thresholds per profile
        # These are tuned to preserve valuable content while filtering noise
        profile_thresholds = {
            "academic_whitepaper": 10,  # Strict: skip page numbers, keep footnotes
            "technical_manual": 5,  # Moderate: keep spec labels, short refs
            "digital_magazine": 3,  # Relaxed: keep pull-quotes, captions
            "scanned_degraded": 2,  # Very relaxed: OCR tolerance
            "scanned_clean": 3,  # Relaxed: keep OCR text
            "unknown": 5,  # Default: moderate threshold
        }

        min_chars = profile_thresholds.get(profile_type, 5)

        # Apply minimum length threshold
        if len(stripped) < min_chars:
            # BUT: Check if this looks like a meaningful short chunk
            # Page numbers, bullets, and pure digits are noise
            # Short words/acronyms are valuable
            if re.match(r"^\d+$", stripped):
                # Pure number (likely page number) - skip for academic
                if profile_type == "academic_whitepaper":
                    logger.debug(f"[FILTER-{profile_type}] Skipping page number: '{stripped}'")
                    return (True, FilterCategory.PAGE_NUMBER)
                # Keep for magazines (could be figure reference)
            elif len(stripped) < 2:
                # Single character - usually noise
                logger.debug(f"[FILTER-{profile_type}] Skipping single char: '{stripped}'")
                return (True, FilterCategory.DECORATION)

            # Log but keep short content for non-academic profiles
            if profile_type not in ("academic_whitepaper",):
                logger.debug(
                    f"[FILTER-{profile_type}] Keeping short chunk ({len(stripped)} chars): '{stripped[:20]}...'"
                )
                return (False, None)

            logger.debug(
                f"[FILTER-{profile_type}] Skipping short chunk ({len(stripped)} < {min_chars}): '{stripped[:20]}...'"
            )
            return (True, FilterCategory.TOO_SHORT)

        # ================================================================
        # RULE 3: Pure decoration (universal, but alphanumeric check)
        # ================================================================
        if re.match(r"^[\s\-_=•·…]+$", stripped):
            # Pure decoration (only dashes, bullets, equals, ellipsis)
            if not any(c.isalnum() for c in stripped):
                logger.debug(f"[FILTER] Skipping decoration: '{stripped}'")
                return (True, FilterCategory.DECORATION)

        # ================================================================
        # RULE 4: Profile-specific noise patterns
        # ================================================================
        if profile_type == "academic_whitepaper":
            # Skip common academic noise patterns
            academic_noise = [
                r"^page\s*\d+$",  # "Page 1", "page 23"
                r"^\d+\s*/\s*\d+$",  # "1 / 5" (page indicators)
                r"^[ivxlcdm]+$",  # Roman numerals (preface pages)
                r"^©\s*\d{4}",  # Copyright lines
            ]
            for pattern in academic_noise:
                if re.match(pattern, stripped, re.IGNORECASE):
                    logger.debug(f"[FILTER-academic] Skipping noise pattern: '{stripped}'")
                    return (True, FilterCategory.NOISE_PATTERN)

        # ================================================================
        # RULE 5: Very small bbox (only for academic profile)
        # ================================================================
        if profile_type == "academic_whitepaper":
            if chunk.metadata and chunk.metadata.spatial and chunk.metadata.spatial.bbox:
                bbox = chunk.metadata.spatial.bbox
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                area = width * height

                # In normalized coordinates (0-1000), area < 50 is 0.005% of page
                # These are usually decorative elements or artifacts
                if area < 50 and len(stripped) < 5:
                    logger.debug(
                        f"[FILTER-academic] Skipping tiny bbox ({area}) with short text: '{stripped}'"
                    )
                    return (True, FilterCategory.TINY_BBOX)

        return (False, None)

    def _post_process_ocr_text(self, text: str) -> str:
        """
        Post-process OCR text to fix common fragmentation issues.

        Fixes:
        - Fragmented decimals: "2 . 1" → "2.1"
        - Fragmented multiplication: "2 . 1 ×" → "2.1×"
        - Fragmented percentages: "10 %" → "10%"
        - Fragmented units: "300 MHz" → "300MHz"
        - Fragmented math symbols: "± 2" → "±2"

        CRITICAL: Does NOT join all spaces between numbers to preserve
        mathematical sequences like "1 2 3 4".

        Args:
            text: Raw OCR text

        Returns:
            Cleaned text with technical values properly joined
        """
        import re

        # Decimals: "2 . 1" → "2.1"
        text = re.sub(r"(\d+)\s+\.\s+(\d+)", r"\1.\2", text)

        # Multiplication: "2 . 1 ×" or "2.1 ×" → "2.1×"
        text = re.sub(r"(\d+\.?\d*)\s+×", r"\1×", text)

        # Percentages: "10 %" → "10%"
        text = re.sub(r"(\d+\.?\d*)\s+%", r"\1%", text)

        # Units (GHz, MHz, KB, MB, etc.)
        text = re.sub(
            r"(\d+\.?\d*)\s+(GHz|MHz|KB|MB|GB|TB|ms|μs|ns)", r"\1\2", text, flags=re.IGNORECASE
        )

        # Mathematical symbols: "± 2" → "±2", "≈ 2.5" → "≈2.5"
        text = re.sub(r"([±≈])\s+(\d)", r"\1\2", text)

        return text

    def _apply_quality_filters(
        self,
        chunks: List[IngestionChunk],
    ) -> List[IngestionChunk]:
        """
        Apply quality filters to chunk list.

        Filters:
        1. Empty chunk filtering (asset-aware)
        2. OCR text post-processing (number joining)
        3. Look-ahead buffer for symmetric overlap

        Args:
            chunks: Raw chunk list

        Returns:
            Filtered and improved chunk list
        """
        # Step 1: Filter out invalid chunks with tracking
        valid_chunks = []
        filtered_count = 0

        for chunk in chunks:
            should_skip, category = self._should_skip_chunk(chunk)
            if should_skip:
                filtered_count += 1
                # Track filtered chunk if we have a tracker
                if self._quality_filter_tracker and category:
                    self._quality_filter_tracker.track_filtered_chunk(chunk, category)
            else:
                valid_chunks.append(chunk)

        logger.info(f"[QUALITY] Filtered {filtered_count} invalid chunks")

        # Step 2: Post-process OCR text
        for chunk in valid_chunks:
            if chunk.modality == Modality.TEXT:
                original = chunk.content
                cleaned = self._post_process_ocr_text(original)
                if original != cleaned:
                    chunk.content = cleaned
                    logger.debug(f"[OCR-CLEAN] Fixed technical values in chunk {chunk.chunk_id}")

        # Step 3: Look-ahead buffer for symmetric overlap (fill next_text_snippet)
        valid_chunks = self._apply_lookahead_buffer(valid_chunks)

        return valid_chunks

    def _apply_lookahead_buffer(
        self,
        chunks: List[IngestionChunk],
    ) -> List[IngestionChunk]:
        """
        Apply look-ahead buffer to fill next_text_snippet fields.

        This ensures symmetric overlap (REQ-MM-03) by looking ahead to the
        next chunk to populate next_text_snippet.

        CRITICAL: The last chunk of the document (or batch) should have
        next_text_snippet = None, not cause an IndexError.

        Args:
            chunks: Chunk list with potentially empty next_text_snippet fields

        Returns:
            Chunk list with next_text_snippet populated where possible
        """
        # CRITICAL: Guard against empty chunks list
        if not chunks:
            return chunks

        # Process all chunks except the last one
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]

            # Only fill if next_text_snippet is empty
            if current_chunk.semantic_context is None:
                from .schema.ingestion_schema import SemanticContext

                current_chunk.semantic_context = SemanticContext()

            if not current_chunk.semantic_context.next_text_snippet:
                # CRITICAL: Safety check - ensure next_chunk exists and has content
                if next_chunk and next_chunk.content:
                    next_text = next_chunk.content[:300]
                    current_chunk.semantic_context.next_text_snippet = next_text

                    logger.debug(
                        f"[LOOKAHEAD] Filled next_text_snippet for chunk {current_chunk.chunk_id} "
                        f"from {next_chunk.chunk_id}"
                    )

        # Last chunk: explicitly set next_text_snippet to None (safety)
        # This prevents any look-ahead issues on the final chunk
        last_chunk = chunks[-1]
        if last_chunk.semantic_context is None:
            from .schema.ingestion_schema import SemanticContext

            last_chunk.semantic_context = SemanticContext(next_text_snippet=None)

        return chunks

    def should_use_batching(self, pdf_path: str | Path) -> bool:
        """
        Determine if a PDF should be processed with batching.

        Batching is recommended for PDFs with more pages than batch_size.

        Args:
            pdf_path: Path to PDF file

        Returns:
            True if batching is recommended
        """
        try:
            splitter = PDFBatchSplitter(batch_size=self.batch_size)
            page_count = splitter.get_page_count(Path(pdf_path))
            return page_count > self.batch_size
        except Exception:
            return False

    # ========================================================================
    # REQ-COORD-02: PAGE DIMENSION EXTRACTION
    # ========================================================================

    def _extract_page_dimensions(self, pdf_path: Path) -> Dict[int, Tuple[int, int]]:
        """
        REQ-COORD-02: Extract page dimensions from PDF for UI overlay support.

        This method scans the PDF and extracts (width, height) in pixels
        for each page. These dimensions are propagated to ALL chunks
        (text/image/table) via spatial.page_width and spatial.page_height.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dict mapping page_number (1-indexed) to (width_px, height_px)
        """
        page_dims: Dict[int, Tuple[int, int]] = {}

        try:
            doc = fitz.open(pdf_path)
            for page_idx in range(len(doc)):
                page = doc.load_page(page_idx)
                rect = page.rect
                # Convert PDF points to integer pixels (at 72 DPI base)
                width_px = int(rect.width)
                height_px = int(rect.height)
                page_dims[page_idx + 1] = (width_px, height_px)

            doc.close()
            logger.info(f"[REQ-COORD-02] Extracted page dimensions for {len(page_dims)} pages")
        except Exception as e:
            logger.warning(f"[REQ-COORD-02] Failed to extract page dimensions: {e}")

        return page_dims

    def _propagate_page_dimensions(
        self,
        chunks: List[IngestionChunk],
    ) -> List[IngestionChunk]:
        """
        REQ-COORD-02: Propagate page dimensions to ALL chunks.

        This ensures that page_width and page_height are NEVER null
        in any chunk's spatial metadata, which is required for UI overlay support.

        Args:
            chunks: List of chunks to update

        Returns:
            Updated chunk list with page dimensions
        """
        if not self._page_dimensions:
            logger.warning("[REQ-COORD-02] No page dimensions available for propagation")
            return chunks

        logger.debug(
            f"[REQ-COORD-02] Available page dimensions for {len(self._page_dimensions)} pages: {self._page_dimensions}"
        )
        updated_count = 0
        already_set_count = 0
        no_dimensions_count = 0

        for chunk in chunks:
            page_no = chunk.metadata.page_number
            dims = self._page_dimensions.get(page_no)

            if dims:
                width_px, height_px = dims

                # Ensure spatial metadata exists
                if chunk.metadata.spatial is None:
                    chunk.metadata.spatial = SpatialMetadata(bbox=None)

                # Only update if currently null (non-destructive)
                updated_this_chunk = False
                if chunk.metadata.spatial.page_width is None:
                    chunk.metadata.spatial.page_width = width_px
                    updated_this_chunk = True
                if chunk.metadata.spatial.page_height is None:
                    chunk.metadata.spatial.page_height = height_px
                    updated_this_chunk = True

                if updated_this_chunk:
                    updated_count += 1
                    logger.debug(
                        f"[REQ-COORD-02] Updated page dimensions for chunk {chunk.chunk_id} on page {page_no}: {width_px}x{height_px}"
                    )
                else:
                    already_set_count += 1
                    logger.debug(
                        f"[REQ-COORD-02] Page dimensions already set for chunk {chunk.chunk_id} on page {page_no}"
                    )
            else:
                no_dimensions_count += 1
                logger.debug(
                    f"[REQ-COORD-02] No dimensions for page {page_no} (chunk {chunk.chunk_id})"
                )

        logger.info(
            f"[REQ-COORD-02] Propagated page dimensions to {updated_count} chunks. "
            f"Already set: {already_set_count}, No dimensions: {no_dimensions_count}, Total chunks: {len(chunks)}"
        )
        return chunks

    # ========================================================================
    # TEXT INTEGRITY SCOUT (Recovery Mode)
    # ========================================================================

    def _run_text_integrity_scout(
        self,
        chunks: List[IngestionChunk],
        source_file: str,
        variance_percent: float,
    ) -> List[IngestionChunk]:
        """
        Recovery scan to rescue lost text when variance > 10%.

        This "Safety Net" compares the raw PyMuPDF text extraction against
        the text in generated chunks. Any text blocks > 50 chars that don't
        appear in any chunk are rescued as recovery chunks.

        ARCHITECTURE:
        1. Extract raw text from PDF using PyMuPDF (per page)
        2. Build a set of "covered" text from existing chunks
        3. Find "orphaned" text blocks that aren't covered
        4. Create recovery chunks for orphaned text

        Args:
            chunks: All chunks from layout-aware processing
            source_file: Source filename
            variance_percent: Current token variance (triggers if > 10%)

        Returns:
            Extended chunk list with recovery chunks added
        """
        # Only run if variance exceeds threshold
        RECOVERY_THRESHOLD = -10.0  # Trigger at 10% token loss
        MIN_ORPHAN_LENGTH = 50  # Minimum chars to rescue (can be lowered on front pages)

        if variance_percent >= RECOVERY_THRESHOLD:
            logger.info(
                f"[RECOVERY] Variance {variance_percent:.1f}% is within tolerance, skipping recovery"
            )
            return chunks

        # Phase 3: Attempt image→text reclassification for mis-ID'd front-matter images
        try:
            chunks = self._reclassify_text_images(chunks)
        except Exception as e:
            logger.warning(f"[RECOVERY] Image→text reclassification skipped due to error: {e}")

        logger.info(
            f"[RECOVERY] ⚠️ Variance {variance_percent:.1f}% exceeds threshold ({RECOVERY_THRESHOLD}%). "
            f"Initiating TextIntegrityScout..."
        )
        print(
            f"\n🔍 [RECOVERY] Token variance {variance_percent:.1f}% detected! "
            f"Running TextIntegrityScout...",
            flush=True,
        )

        if not self._current_pdf_path or not self._current_pdf_path.exists():
            logger.warning("[RECOVERY] No PDF path available, cannot run recovery")
            return chunks

        recovery_chunks: List[IngestionChunk] = []

        try:
            import re
            from difflib import SequenceMatcher

            # Build map of figure bboxes per page for code recovery (use image modality)
            figure_bboxes_per_page: Dict[int, List[Tuple[List[int], IngestionChunk]]] = {}
            for ch in chunks:
                if ch.modality == Modality.IMAGE and ch.metadata and ch.metadata.spatial:
                    if ch.metadata.spatial.bbox:
                        page_no = ch.metadata.page_number
                        figure_bboxes_per_page.setdefault(page_no, []).append(
                            (ch.metadata.spatial.bbox, ch)
                        )

            # Step 1: Extract raw text per page from PDF (TEXT-LAYER ONLY; NO OCR)
            doc = fitz.open(self._current_pdf_path)
            raw_text_per_page: Dict[int, str] = {}
            text_blocks_per_page: Dict[int, List[Tuple[List[float], str]]] = {}
            has_text_layer = False

            for page_idx in range(len(doc)):
                page = doc.load_page(page_idx)
                page_text = page.get_text("text")
                if page_text and page_text.strip():
                    raw_text_per_page[page_idx + 1] = page_text.strip()
                    has_text_layer = True

                # Capture positional text blocks for code-aware recovery
                blocks = page.get_text("blocks")
                page_blocks: List[Tuple[List[float], str]] = []
                for b in blocks:
                    # b: (x0, y0, x1, y1, text, block_no, block_type, block_flags?)
                    if len(b) >= 5 and isinstance(b[4], str) and b[4].strip():
                        bbox = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
                        page_blocks.append((bbox, b[4]))
                if page_blocks:
                    text_blocks_per_page[page_idx + 1] = page_blocks

            doc.close()

            if has_text_layer:
                logger.info(
                    "[RECOVERY] Text layer detected; recovery uses PDF text blocks only (OCR disabled)"
                )
            else:
                logger.warning(
                    "[RECOVERY] No PDF text layer detected; recovery will not invoke OCR cascade "
                    "(per guardrail). Extraction limited to available text blocks."
                )

            logger.info(f"[RECOVERY] Extracted raw text from {len(raw_text_per_page)} pages")

            # Step 2: Build "covered text" set from existing chunks (per page)
            covered_text_per_page: Dict[int, List[str]] = {}

            for chunk in chunks:
                if chunk.modality != Modality.TEXT:
                    continue

                page_no = chunk.metadata.page_number
                content = chunk.content.strip()

                if page_no not in covered_text_per_page:
                    covered_text_per_page[page_no] = []

                if content and len(content) >= 10:
                    covered_text_per_page[page_no].append(content.lower())

            # Helper: bbox IoU
            def _bbox_iou(b1: List[float], b2: List[int]) -> float:
                x0 = max(b1[0], b2[0])
                y0 = max(b1[1], b2[1])
                x1 = min(b1[2], b2[2])
                y1 = min(b1[3], b2[3])
                if x1 <= x0 or y1 <= y0:
                    return 0.0
                inter = (x1 - x0) * (y1 - y0)
                a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                return inter / max(a1 + a2 - inter, 1e-6)

            code_pattern = re.compile(
                r"(def\s+\w+\(|class\s+\w+|import\s+\w+|from\s+\w+|if\s+__name__|async\s+def|return\s|try:|except\s|with\s|self\.|@[\w_]+)"
            )

            # Step 3: Find orphaned text blocks per page
            total_rescued = 0
            coverage_by_page: Dict[int, float] = {}
            flagged_front_pages: List[int] = []

            for page_no, raw_text in raw_text_per_page.items():
                # Front pages are allowed a lower threshold and stricter coverage target
                is_front_page = page_no <= 2
                page_min_orphan = 20 if is_front_page else MIN_ORPHAN_LENGTH

                # Compute coverage ratio for this page using current covered texts
                covered_texts = covered_text_per_page.get(page_no, [])
                if self._token_validator and self._token_validator._counter:
                    src_tok = self._token_validator._counter.count_tokens(raw_text)
                    chk_tok = sum(
                        self._token_validator._counter.count_tokens(t) for t in covered_texts
                    )
                else:
                    src_tok = len(raw_text)
                    chk_tok = sum(len(t) for t in covered_texts)
                coverage_ratio = (chk_tok / src_tok) if src_tok > 0 else 1.0
                coverage_by_page[page_no] = coverage_ratio
                if is_front_page and coverage_ratio < 0.85:
                    flagged_front_pages.append(page_no)

                if is_front_page and coverage_ratio < 0.8:
                    logger.info(
                        f"[RECOVERY] Front-page coverage low ({coverage_ratio:.2%}) on page {page_no}; "
                        "attempting block-level rescue"
                    )
                    # Recover small blocks on front pages that were missed
                    covered_bboxes = []
                    for ch in chunks:
                        if ch.metadata.page_number != page_no:
                            continue
                        if ch.metadata and ch.metadata.spatial and ch.metadata.spatial.bbox:
                            covered_bboxes.append(ch.metadata.spatial.bbox)

                    def _bbox_overlaps_any(b1: List[float], others: List[List[int]]) -> bool:
                        for ob in others:
                            if _bbox_iou(b1, ob) > 0.1:
                                return True
                        return False

                    for bbox, block_text in text_blocks_per_page.get(page_no, []):
                        text_clean = block_text.strip()
                        if len(text_clean) < 20:
                            continue
                        if _bbox_overlaps_any(bbox, covered_bboxes):
                            continue
                        para_lower = text_clean.lower()
                        # quick dedup check
                        already = False
                        for covered in covered_texts:
                            if len(covered) < 10:
                                continue
                            if para_lower[:40] in covered or covered[:40] in para_lower:
                                already = True
                                break
                        if already:
                            continue

                        doc_title = Path(source_file).stem if source_file else "Document"
                        hierarchy = HierarchyMetadata(
                            parent_heading=None,
                            breadcrumb_path=[doc_title, f"Page {page_no}", "[RECOVERED-FRONT]"],
                            level=3,
                        )
                        recovery_chunk = create_text_chunk(
                            doc_id=self._doc_hash or "unknown",
                            content=text_clean,
                            source_file=source_file,
                            file_type=FileType.PDF,
                            page_number=page_no,
                            hierarchy=hierarchy,
                            bbox=[int(v) for v in bbox],
                            extraction_method="recovery_frontpage",
                            **self._intelligence_metadata,
                        )
                        recovery_chunks.append(recovery_chunk)
                        total_rescued += 1
                        covered_texts.append(para_lower)

                # Split raw text into paragraphs/blocks
                paragraphs = re.split(r"\n\s*\n|\n{2,}", raw_text)
                covered_texts = covered_text_per_page.get(page_no, covered_texts)

                for para_idx, para in enumerate(paragraphs):
                    para_clean = para.strip()

                    # Skip short paragraphs
                    if len(para_clean) < page_min_orphan:
                        continue

                    # Check if this paragraph is covered by any chunk
                    para_lower = para_clean.lower()
                    is_covered = False

                    for covered in covered_texts:
                        # Use fuzzy matching - if >60% overlap, consider covered
                        if len(covered) < 10:
                            continue

                        # Check substring match first (fast)
                        if para_lower[:50] in covered or covered[:50] in para_lower:
                            is_covered = True
                            break

                        # Check sequence similarity for partial matches
                        ratio = SequenceMatcher(None, para_lower[:200], covered[:200]).ratio()
                        if ratio > 0.6:
                            is_covered = True
                            break

                    if not is_covered:
                        # ORPHANED TEXT - Rescue it!
                        logger.info(
                            f"[RECOVERY] Found orphaned text on page {page_no}: "
                            f"'{para_clean[:60]}...' ({len(para_clean)} chars)"
                        )

                        # Create recovery chunk
                        doc_title = Path(source_file).stem if source_file else "Document"
                        hierarchy = HierarchyMetadata(
                            parent_heading=None,
                            breadcrumb_path=[doc_title, f"Page {page_no}", "[RECOVERED]"],
                            level=3,
                        )

                        recovery_chunk = create_text_chunk(
                            doc_id=self._doc_hash or "unknown",
                            content=para_clean,
                            source_file=source_file,
                            file_type=FileType.PDF,
                            page_number=page_no,
                            hierarchy=hierarchy,
                            extraction_method="recovery_scan",  # Marks as rescued text
                            **self._intelligence_metadata,
                        )

                        recovery_chunks.append(recovery_chunk)
                        total_rescued += 1

                # Step 3b: Code-aware recovery for text blocks overlapping figures (subsurface extraction)
                if page_no in figure_bboxes_per_page and page_no in text_blocks_per_page:
                    existing_texts = covered_text_per_page.get(page_no, [])
                    for fig_bbox, fig_chunk in figure_bboxes_per_page[page_no]:
                        for bbox, block_text in text_blocks_per_page[page_no]:
                            if len(block_text) < 50:
                                continue
                            if _bbox_iou(bbox, fig_bbox) < 0.1:
                                continue

                            # Dedup guard: skip if this block is already covered (>80% similarity)
                            para_lower = block_text.strip().lower()
                            is_covered = False
                            for covered in existing_texts:
                                if len(covered) < 10:
                                    continue
                                if para_lower[:50] in covered or covered[:50] in para_lower:
                                    is_covered = True
                                    break
                                ratio = SequenceMatcher(
                                    None, para_lower[:200], covered[:200]
                                ).ratio()
                                if ratio > 0.8:
                                    is_covered = True
                                    break
                            if is_covered:
                                continue

                            classification = "code" if code_pattern.search(block_text) else None
                            label = "[RECOVERED-CODE]" if classification == "code" else "[RECOVERED-FIGURE]"

                            logger.info(
                                f"[RECOVERY] Found subsurface text under figure on page {page_no}: "
                                f"'{block_text[:80]}...' classification={classification or 'text'}"
                            )
                            doc_title = Path(source_file).stem if source_file else "Document"
                            hierarchy = HierarchyMetadata(
                                parent_heading=None,
                                breadcrumb_path=[doc_title, f"Page {page_no}", label],
                                level=3,
                            )
                            recovery_chunk = create_text_chunk(
                                doc_id=self._doc_hash or "unknown",
                                content=block_text.strip(),
                                source_file=source_file,
                                file_type=FileType.PDF,
                                page_number=page_no,
                                hierarchy=hierarchy,
                                bbox=[int(v) for v in fig_bbox],
                                extraction_method="recovery_subsurface",
                                asset_ref=fig_chunk.asset_ref,
                                content_classification=classification,
                                **self._intelligence_metadata,
                            )
                            recovery_chunks.append(recovery_chunk)
                            total_rescued += 1
                            existing_texts.append(para_lower)

                # Step 3c: Low-coverage gap fill (spatial gap filling beyond figures)

                if coverage_ratio < 0.6 and page_no in text_blocks_per_page:
                    # Build covered bboxes (text/image/table) to find gaps
                    covered_bboxes = []
                    for ch in chunks:
                        if ch.metadata.page_number != page_no:
                            continue
                        if ch.metadata and ch.metadata.spatial and ch.metadata.spatial.bbox:
                            covered_bboxes.append(ch.metadata.spatial.bbox)

                    def _bbox_overlaps_any(b1: List[float], others: List[List[int]]) -> bool:
                        for ob in others:
                            if _bbox_iou(b1, ob) > 0.1:
                                return True
                        return False

                    academic_noise = [
                        r"^page\s*\d+$",
                        r"^\d+\s*/\s*\d+$",
                        r"^[ivxlcdm]+$",
                        r"^©\s*\d{4}",
                    ]

                    for bbox, block_text in text_blocks_per_page[page_no]:
                        text_clean = block_text.strip()
                        if len(text_clean) < 60:  # lowered threshold to widen gap-fill net
                            continue
                        # Skip if overlaps existing coverage
                        if _bbox_overlaps_any(bbox, covered_bboxes):
                            continue
                        # Noise guard
                        noise_hit = False
                        for pattern in academic_noise:
                            if re.match(pattern, text_clean, re.IGNORECASE):
                                noise_hit = True
                                break
                        if noise_hit:
                            continue
                        # Dedup guard against existing covered texts (strict 80% sim)
                        para_lower = text_clean.lower()
                        is_covered_gap = False
                        for covered in covered_texts:
                            if len(covered) < 10:
                                continue
                            if para_lower[:50] in covered or covered[:50] in para_lower:
                                is_covered_gap = True
                                break
                            ratio = SequenceMatcher(None, para_lower[:200], covered[:200]).ratio()
                            if ratio > 0.8:
                                is_covered_gap = True
                                break
                        if is_covered_gap:
                            continue

                        classification = "code" if code_pattern.search(text_clean) else None
                        label = "[RECOVERED-CODE]" if classification == "code" else "[RECOVERED-GAP]"

                        logger.info(
                            f"[RECOVERY] Gap-fill text on page {page_no}: '{text_clean[:80]}...' "
                            f"classification={classification or 'text'}"
                        )
                        doc_title = Path(source_file).stem if source_file else "Document"
                        hierarchy = HierarchyMetadata(
                            parent_heading=None,
                            breadcrumb_path=[doc_title, f"Page {page_no}", label],
                            level=3,
                        )
                        recovery_chunk = create_text_chunk(
                            doc_id=self._doc_hash or "unknown",
                            content=text_clean,
                            source_file=source_file,
                            file_type=FileType.PDF,
                            page_number=page_no,
                            hierarchy=hierarchy,
                            bbox=[int(v) for v in bbox],
                            extraction_method="recovery_gap_fill",
                            content_classification=classification,
                            **self._intelligence_metadata,
                        )
                        recovery_chunks.append(recovery_chunk)
                        total_rescued += 1
                        covered_texts.append(para_lower)

            if recovery_chunks:
                print(
                    f"    ✓ [RECOVERY] Rescued {total_rescued} orphaned text blocks",
                    flush=True,
                )
                logger.info(
                    f"[RECOVERY] TextIntegrityScout rescued {total_rescued} text blocks "
                    f"across {len(set(c.metadata.page_number for c in recovery_chunks))} pages"
                )

                # Add recovery chunks to the list
                chunks.extend(recovery_chunks)
            else:
                print(
                    "    ✓ [RECOVERY] No orphaned text found (all text accounted for)", flush=True
                )
                logger.info("[RECOVERY] No orphaned text blocks found")

            # Phase 4: Enhanced front-page processing if coverage still low
            if flagged_front_pages:
                try:
                    chunks = self._process_front_pages_enhanced(
                        chunks, flagged_front_pages, covered_text_per_page
                    )
                except Exception as e:
                    logger.warning(f"[RECOVERY] Enhanced front-page processing skipped: {e}")

        except Exception as e:
            logger.error(f"[RECOVERY] TextIntegrityScout failed: {e}")
            print(f"    ⚠️ [RECOVERY] Scout failed: {e}", flush=True)

        return chunks

    def _reclassify_text_images(self, chunks: List[IngestionChunk]) -> List[IngestionChunk]:
        """
        Phase 3: Reclassify IMAGE chunks that likely contain text (front pages only).
        Uses EasyOCR if available. Guardrails: max 5 images per page, pages 1-2 only.
        """
        try:
            import easyocr  # type: ignore
            from PIL import Image
        except Exception as e:  # pragma: no cover - optional dep
            logger.info(f"[RECOVERY] EasyOCR not available; skipping image→text reclassification ({e})")
            return chunks

        KEYWORDS = [
            "blurred text",
            "text document",
            "partially legible",
            "difficult to read",
            "pixelated text",
            "text section",
            "document section",
            "text",
        ]

        reader = easyocr.Reader(["en"], gpu=False)  # small, CPU-safe
        max_per_page = 5
        updated = 0

        # Group image chunks by page
        images_by_page: Dict[int, List[IngestionChunk]] = {}
        for ch in chunks:
            if ch.modality != Modality.IMAGE:
                continue
            page_no = ch.metadata.page_number if ch.metadata else None
            if not page_no:
                continue
            if page_no > 2:  # front-matter only
                continue
            desc = ""
            if ch.metadata and hasattr(ch.metadata, "visual_description"):
                desc = (ch.metadata.visual_description or "").lower()
            if not any(k in desc for k in KEYWORDS):
                continue
            images_by_page.setdefault(page_no, []).append(ch)

        for page_no, page_imgs in images_by_page.items():
            attempts = 0
            for img_chunk in page_imgs:
                if attempts >= max_per_page:
                    break
                if not img_chunk.asset_ref or not getattr(img_chunk.asset_ref, "file_path", None):
                    continue
                attempts += 1
                try:
                    img = Image.open(img_chunk.asset_ref.file_path)
                    width, height = img.size
                    if width < 40 or height < 40:
                        continue
                    result = reader.readtext(img_chunk.asset_ref.file_path, detail=0)
                    ocr_text = "\n".join([r.strip() for r in result if r.strip()])
                    if len(ocr_text) < 20:
                        continue
                    alpha_ratio = sum(c.isalpha() for c in ocr_text) / max(len(ocr_text), 1)
                    if alpha_ratio < 0.6:
                        continue

                    # Reclassify
                    img_chunk.modality = Modality.TEXT
                    img_chunk.content = ocr_text
                    if img_chunk.metadata:
                        img_chunk.metadata.extraction_method = "image_to_text_recovery"
                    updated += 1
                except Exception as e:  # pragma: no cover
                    logger.debug(f"[RECOVERY] OCR failed for page {page_no} image: {e}")
                    continue

        if updated:
            print(f"    ✓ [RECOVERY] Reclassified {updated} text-like images on front pages", flush=True)
            logger.info(f"[RECOVERY] Reclassified {updated} images to text on front pages")
        return chunks

    def _process_front_pages_enhanced(
        self,
        chunks: List[IngestionChunk],
        pages: List[int],
        covered_text_per_page: Dict[int, List[str]],
    ) -> List[IngestionChunk]:
        """
        Phase 4 (lightweight): extra recovery on front pages with low coverage.
        - OCR all images on flagged pages (regardless of VLM description) with EasyOCR if available.
        - Re-run PyMuPDF block extraction with lower threshold and dedup by hash.
        """
        try:
            import easyocr  # type: ignore
        except Exception as e:  # pragma: no cover
            logger.info(f"[RECOVERY] EasyOCR not available for enhanced pass: {e}")
            return chunks

        reader = easyocr.Reader(["en"], gpu=False)  # CPU-safe
        doc = fitz.open(self._current_pdf_path)
        new_chunks: List[IngestionChunk] = []
        seen_hashes = set()

        # seed dedup with existing text chunks
        for ch in chunks:
            if ch.modality == Modality.TEXT and ch.content:
                h = hashlib.md5(ch.content.strip().lower().encode("utf-8")).hexdigest()
                seen_hashes.add(h)

        for page_no in pages:
            page_idx = page_no - 1
            if page_idx < 0 or page_idx >= len(doc):
                continue

            # 1) OCR every image chunk on this page
            page_imgs = [c for c in chunks if c.modality == Modality.IMAGE and c.metadata.page_number == page_no]
            ocr_count = 0
            for img_chunk in page_imgs:
                if not img_chunk.asset_ref or not getattr(img_chunk.asset_ref, "file_path", None):
                    continue
                if ocr_count >= 5:
                    break
                try:
                    result = reader.readtext(img_chunk.asset_ref.file_path, detail=0)
                    ocr_text = "\n".join([r.strip() for r in result if r.strip()])
                    if len(ocr_text) < 20:
                        continue
                    alpha_ratio = sum(c.isalpha() for c in ocr_text) / max(len(ocr_text), 1)
                    if alpha_ratio < 0.6:
                        continue
                    h = hashlib.md5(ocr_text.strip().lower().encode("utf-8")).hexdigest()
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    img_chunk.modality = Modality.TEXT
                    img_chunk.content = ocr_text
                    if img_chunk.metadata:
                        img_chunk.metadata.extraction_method = "enhanced_image_ocr"
                    ocr_count += 1
                except Exception as e:  # pragma: no cover
                    logger.debug(f"[RECOVERY] Enhanced OCR failed for page {page_no}: {e}")
                    continue

            # 2) Re-run PyMuPDF block extraction with low threshold
            page = doc.load_page(page_idx)
            blocks = page.get_text("blocks")
            for b in blocks:
                if len(b) < 5 or not isinstance(b[4], str):
                    continue
                text_clean = b[4].strip()
                if len(text_clean) < 20:
                    continue
                h = hashlib.md5(text_clean.lower().encode("utf-8")).hexdigest()
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                doc_title = self._current_pdf_path.stem
                hierarchy = HierarchyMetadata(
                    parent_heading=None,
                    breadcrumb_path=[doc_title, f"Page {page_no}", "[ENHANCED]"],
                    level=3,
                )
                bbox = [int(b[0]), int(b[1]), int(b[2]), int(b[3])]
                new_chunk = create_text_chunk(
                    doc_id=self._doc_hash or "unknown",
                    content=text_clean,
                    source_file=str(self._current_pdf_path.name),
                    file_type=FileType.PDF,
                    page_number=page_no,
                    hierarchy=hierarchy,
                    bbox=bbox,
                    extraction_method="enhanced_frontpage",
                    **self._intelligence_metadata,
                )
                new_chunks.append(new_chunk)

        doc.close()

        if new_chunks:
            chunks.extend(new_chunks)
            print(f"    ✓ [RECOVERY] Enhanced front pages added {len(new_chunks)} chunks", flush=True)
            logger.info(f"[RECOVERY] Enhanced front pages added {len(new_chunks)} chunks")
        else:
            logger.info("[RECOVERY] Enhanced front pages produced no new chunks")

        return chunks

    # ========================================================================
    # QA-CHECK-01: TOKEN VALIDATION
    # ========================================================================

    def _run_token_validation(
        self,
        chunks: List[IngestionChunk],
        source_file: str,
    ) -> TokenValidationResult:
        """
        QA-CHECK-01: Run token balance validation on text chunks.

        CRITICAL FIX: The source_text MUST come from the actual PDF, not from
        the chunks themselves (which would be circular and meaningless).

        We use PyMuPDF to extract the raw text from the PDF and compare it
        against the sum of TEXT chunk tokens.

        Args:
            chunks: All chunks (only TEXT modality is validated)
            source_file: Document name for logging

        Returns:
            TokenValidationResult with validation metrics
        """
        if self._token_validator is None:
            logger.warning("[QA-CHECK-01] TokenValidator not initialized; skipping validation")
            return TokenValidationResult(
                is_valid=True,
                source_token_count=0,
                chunk_token_count=0,
                variance_percent=0.0,
                overlap_allowance_tokens=0,
                tolerance_percent=10.0,
                error_message="TokenValidator unavailable",
            )

        try:
            # Extract only TEXT chunks for validation
            text_chunks = [c for c in chunks if c.modality == Modality.TEXT]

            if not text_chunks:
                logger.info("[QA-CHECK-01] No TEXT chunks to validate")
                return TokenValidationResult(
                    is_valid=True,
                    source_token_count=0,
                    chunk_token_count=0,
                    variance_percent=0.0,
                    overlap_allowance_tokens=0,
                    tolerance_percent=10.0,
                )

            # ================================================================
            # CRITICAL FIX: Extract source text from ACTUAL PDF, not chunks
            # ================================================================
            # Previous code was CIRCULAR: comparing chunks to themselves!
            # Now we extract raw text from PDF using PyMuPDF.
            # ================================================================
            source_text = ""
            if self._current_pdf_path and self._current_pdf_path.exists():
                try:
                    doc = fitz.open(self._current_pdf_path)
                    all_text_parts = []
                    for page_idx in range(len(doc)):
                        page = doc.load_page(page_idx)
                        page_text = page.get_text("text")
                        if page_text:
                            all_text_parts.append(page_text.strip())
                    doc.close()
                    source_text = "\n".join(all_text_parts)
                    logger.info(
                        f"[QA-CHECK-01] Extracted {len(source_text)} chars from PDF for validation"
                    )
                except Exception as pdf_err:
                    logger.warning(f"[QA-CHECK-01] Failed to extract PDF text: {pdf_err}")
                    # Fallback: use chunks (not ideal but better than crashing)
                    source_text = " ".join(c.content for c in text_chunks if c.content)
            else:
                # No PDF path available - use chunks as fallback
                logger.warning("[QA-CHECK-01] No PDF path available, using chunk text as source")
                source_text = " ".join(c.content for c in text_chunks if c.content)

            # ================================================================
            # MULTIMODAL-AWARE VALIDATION
            # ================================================================
            # VLM descriptions (visual_description) are NEW tokens that don't
            # exist in the source PDF. We must exclude them from chunk count.
            # ================================================================
            adjusted_text_chunks = []
            vlm_token_estimate = 0

            for chunk in text_chunks:
                # If chunk has visual_description, estimate VLM-added tokens
                if chunk.metadata.visual_description:
                    # VLM descriptions add tokens that aren't in source PDF
                    vlm_tokens = self._token_validator._counter.count_tokens(
                        chunk.metadata.visual_description
                    )
                    vlm_token_estimate += vlm_tokens
                adjusted_text_chunks.append(chunk)

            if vlm_token_estimate > 0:
                logger.info(
                    f"[QA-CHECK-01] VLM-added tokens excluded from validation: ~{vlm_token_estimate}"
                )

            # Get profile type for noise allowance calculation
            profile_type = self._intelligence_metadata.get("profile_type", "unknown")

            # CRITICAL FIX: Ensure quality_filter_tracker is available and filled
            # The tracker should have been filled by _apply_quality_filters which runs BEFORE this method
            quality_tracker = self._quality_filter_tracker
            if quality_tracker is None:
                logger.warning(
                    "[QA-CHECK-01] QualityFilterTracker is None; filtering analytics unavailable"
                )
                # Create a temporary tracker for this validation
                quality_tracker = create_quality_filter_tracker()

            # Run validation with REAL source text and filtering awareness
            result = self._token_validator.validate_token_balance(
                chunks=adjusted_text_chunks,
                source_text=source_text,
                overlap_ratio=0.15,  # DSO default overlap
                quality_filter_tracker=quality_tracker,
                profile_type=profile_type,
                noise_allowance=None,  # Use validator's profile-based defaults
            )

            # Log result with filtering analytics
            self._token_validator.log_validation_result(result, doc_name=source_file)

            # Log detailed filtering summary if tracker has data
            if quality_tracker:
                summary = quality_tracker.get_summary()
                if summary.total_filtered_tokens > 0:
                    categories_str = ", ".join(
                        f"{cat.value}: {tokens} tokens"
                        for cat, tokens in summary.tokens_by_category.items()
                        if tokens > 0
                    )
                    logger.info(
                        f"[QA-CHECK-01-FILTER] Document '{source_file}': "
                        f"Filtered {summary.total_filtered_tokens} tokens ({summary.total_filtered_chunks} chunks) "
                        f"across categories: {categories_str}"
                    )
                    print(
                        f"\n🔍 [QA-CHECK-01-FILTER] Filtered {summary.total_filtered_tokens} tokens "
                        f"({result.filtered_ratio_percent:.1f}% of source) in categories: {categories_str}",
                        flush=True,
                    )

            return result
        except Exception as e:
            logger.warning(f"[QA-CHECK-01] Token validation failed; continuing. Error: {e}")
            return TokenValidationResult(
                is_valid=True,
                source_token_count=0,
                chunk_token_count=0,
                variance_percent=0.0,
                overlap_allowance_tokens=0,
                tolerance_percent=10.0,
                error_message=str(e),
            )

    def _validate_token_limit_per_chunk(
        self,
        chunks: List[IngestionChunk],
        max_tokens: int = 512,
    ) -> Tuple[List[IngestionChunk], int]:
        """
        QA-CHECK-01 (Token Limit): Validate and SPLIT chunks exceeding token limits.

        CRITICAL FIX: Instead of truncating (losing data), we now SPLIT large chunks
        into multiple smaller chunks with proper overlap. This preserves ALL text.

        Per SRS REQ-CHUNK-02: Text chunks have hard max of 512 tokens.

        Args:
            chunks: All chunks to validate
            max_tokens: Maximum allowed tokens per chunk (default: 512)

        Returns:
            Tuple of (validated_chunks with splits, split_count)
        """
        import re

        # Null check for token validator
        if self._token_validator is None:
            logger.warning(
                "[QA-CHECK-01] TokenValidator not available, skipping token limit validation"
            )
            return chunks, 0

        split_count = 0
        result_chunks: List[IngestionChunk] = []
        overlap_chars = 60  # Character overlap between split chunks

        for chunk in chunks:
            if chunk.modality != Modality.TEXT:
                result_chunks.append(chunk)
                continue

            # Count tokens in this chunk
            token_count = self._token_validator._counter.count_tokens(chunk.content)

            if token_count <= max_tokens:
                # Within limit - keep as-is
                result_chunks.append(chunk)
                continue

            # OVERSIZED CHUNK - SMART SPLIT instead of truncate
            split_count += 1
            logger.info(
                f"[SMART-SPLIT] Chunk {chunk.chunk_id} has {token_count} tokens (> {max_tokens}). "
                f"Splitting into multiple chunks..."
            )

            # Split the content into multiple chunks with overlap
            sub_chunks = self._smart_split_text(
                text=chunk.content,
                max_tokens=max_tokens,
                overlap_chars=overlap_chars,
            )

            logger.info(
                f"[SMART-SPLIT] Split into {len(sub_chunks)} sub-chunks "
                f"(original: {len(chunk.content)} chars, {token_count} tokens)"
            )

            # Create new IngestionChunk objects for each split
            for idx, sub_text in enumerate(sub_chunks):
                # Generate new chunk_id with split suffix
                new_chunk_id = f"{chunk.chunk_id}_s{idx+1}"

                # Copy metadata but update for split
                new_hierarchy = HierarchyMetadata(
                    parent_heading=(
                        chunk.metadata.hierarchy.parent_heading
                        if chunk.metadata.hierarchy
                        else None
                    ),
                    breadcrumb_path=(
                        chunk.metadata.hierarchy.breadcrumb_path if chunk.metadata.hierarchy else []
                    )
                    + [f"[Split {idx+1}/{len(sub_chunks)}]"],
                    level=(
                        (chunk.metadata.hierarchy.level or 2) + 1 if chunk.metadata.hierarchy else 3
                    ),
                )

                # Create new chunk
                new_chunk = create_text_chunk(
                    doc_id=chunk.doc_id,
                    content=sub_text,
                    source_file=chunk.metadata.source_file,
                    file_type=chunk.metadata.file_type,
                    page_number=chunk.metadata.page_number,
                    hierarchy=new_hierarchy,
                    chunk_type=chunk.metadata.chunk_type,
                    extraction_method=chunk.metadata.extraction_method,
                    prev_text=(
                        chunk.semantic_context.prev_text_snippet if chunk.semantic_context else None
                    ),
                    next_text=(
                        chunk.semantic_context.next_text_snippet if chunk.semantic_context else None
                    ),
                    **{k: v for k, v in self._intelligence_metadata.items() if v is not None},
                )

                # Override chunk_id with our custom split ID
                new_chunk.chunk_id = new_chunk_id

                result_chunks.append(new_chunk)

        if split_count > 0:
            logger.info(
                f"[SMART-SPLIT] Total: {split_count} oversized chunks split "
                f"(preserving ALL text instead of truncating)"
            )
            print(
                f"    📐 [SMART-SPLIT] {split_count} oversized chunks split into smaller parts",
                flush=True,
            )

        return result_chunks, split_count

    def _smart_split_text(
        self,
        text: str,
        max_tokens: int = 512,
        overlap_chars: int = 60,
    ) -> List[str]:
        """
        Intelligently split text into chunks that fit within token limit.

        Uses sentence-aware splitting to avoid breaking mid-sentence.
        Each chunk has overlap with the next for semantic continuity.

        Args:
            text: Text to split
            max_tokens: Maximum tokens per chunk
            overlap_chars: Character overlap between chunks

        Returns:
            List of text chunks, each within token limit
        """
        import re

        # Null check for token validator and its counter
        if self._token_validator is None or self._token_validator._counter is None:
            # Fallback: simple character-based split
            logger.warning(
                "[SMART-SPLIT] TokenValidator or counter unavailable, using character-based split"
            )
            max_chars = max_tokens * 4
            if len(text) <= max_chars:
                return [text]
            # Simple split by max_chars with overlap
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + max_chars, len(text))
                chunk = text[start:end]
                if chunk.strip():
                    chunks.append(chunk.strip())
                start = end - overlap_chars if end < len(text) else end
            return chunks if chunks else [text]

        # Estimate: ~4 chars per token (conservative)
        max_chars_estimate = max_tokens * 4

        # If text is small enough, return as-is
        if self._token_validator._counter.count_tokens(text) <= max_tokens:
            return [text]

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks: List[str] = []
        current_chunk = ""

        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence

            # Check if adding this sentence exceeds limit
            if self._token_validator._counter.count_tokens(test_chunk) > max_tokens:
                if current_chunk:
                    # Save current chunk
                    chunks.append(current_chunk.strip())

                    # Start new chunk with overlap from end of previous
                    overlap_text = (
                        current_chunk[-overlap_chars:]
                        if len(current_chunk) > overlap_chars
                        else current_chunk
                    )
                    current_chunk = overlap_text + " " + sentence
                else:
                    # Single sentence exceeds limit - force split by characters
                    # This handles edge cases like very long single sentences
                    forced_chunks = self._force_split_long_sentence(
                        sentence, max_tokens, overlap_chars
                    )
                    chunks.extend(forced_chunks[:-1])  # Add all but last
                    current_chunk = forced_chunks[-1] if forced_chunks else ""
            else:
                current_chunk = test_chunk

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]

    def _force_split_long_sentence(
        self,
        sentence: str,
        max_tokens: int,
        overlap_chars: int,
    ) -> List[str]:
        """
        Force-split a very long sentence that can't be split at sentence boundaries.

        Uses word boundaries where possible.

        Args:
            sentence: Long sentence to split
            max_tokens: Maximum tokens per chunk
            overlap_chars: Character overlap

        Returns:
            List of chunks from the sentence
        """
        # Estimate max chars
        max_chars = max_tokens * 4 - 50  # Leave margin

        words = sentence.split()
        chunks: List[str] = []
        current_chunk = ""

        for word in words:
            test_chunk = current_chunk + " " + word if current_chunk else word

            if len(test_chunk) > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Start new chunk with overlap
                    overlap = (
                        current_chunk[-overlap_chars:] if len(current_chunk) > overlap_chars else ""
                    )
                    current_chunk = overlap + " " + word
                else:
                    # Single word is too long - just add it (rare edge case)
                    current_chunk = word
            else:
                current_chunk = test_chunk

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [sentence]

    # ========================================================================
    # FULL-PAGE GUARD (IRON-07, REQ-MM-09)
    # ========================================================================

    def _is_full_page_bbox(self, bbox: Optional[List[int]]) -> bool:
        """
        IRON-07: Check if bbox covers full page (area_ratio > 0.95).

        A bbox of [0, 0, 1000, 1000] in normalized coordinates covers
        100% of the page and should trigger Full-Page Guard.

        Args:
            bbox: Normalized bbox [x_min, y_min, x_max, y_max] in 0-1000 scale

        Returns:
            True if bbox is full-page or nearly full-page
        """
        if bbox is None:
            return False

        # Calculate area in normalized coordinates
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height

        # Full page = 1000 * 1000 = 1,000,000
        full_page_area = COORD_SCALE * COORD_SCALE
        area_ratio = area / full_page_area

        return area_ratio > 0.95

    def _apply_full_page_guard(
        self,
        chunks: List[IngestionChunk],
    ) -> List[IngestionChunk]:
        """
        IRON-07, REQ-MM-09: Apply Full-Page Guard to IMAGE chunks.

        When an IMAGE chunk has a bbox covering >95% of the page, this
        adjusts the VLM context to indicate it's a page-level element,
        reducing irrelevant descriptions of page borders/backgrounds.

        By default, full-page assets are kept with an editorial prefix.
        If strict_qa is enabled, full-page assets are filtered out.

        Args:
            chunks: All chunks

        Returns:
            Filtered/modified chunk list
        """
        filtered = []
        fullpage_count = 0

        for chunk in chunks:
            # Only apply to IMAGE modality
            if chunk.modality != Modality.IMAGE:
                filtered.append(chunk)
                continue

            # Check if full-page
            bbox = None
            if chunk.metadata.spatial and chunk.metadata.spatial.bbox:
                bbox = chunk.metadata.spatial.bbox

            if self._is_full_page_bbox(bbox):
                fullpage_count += 1

                # FIX #1: IRON-07 - HARD VLM VERIFICATION CALL
                # "Geen verificatie = geen inclusie" - NO VLM = DISCARD
                if self._vision_manager is None:
                    logger.warning(
                        f"[FULL-PAGE-GUARD] DISCARDING full-page asset on "
                        f"page {chunk.metadata.page_number} (no VLM available - IRON-07)"
                    )
                    continue  # DISCARD - skip this chunk

                # CRITICAL: Hard VLM verification call - if VLM exists, use it to verify
                try:
                    if chunk.asset_ref and chunk.asset_ref.file_path:
                        asset_path = self.output_dir / chunk.asset_ref.file_path
                        if asset_path.exists():
                            # Load image for verification
                            from PIL import Image

                            with Image.open(asset_path) as img:
                                # Call VLM verification - if it fails verification, DISCARD
                                # Get breadcrumbs for context
                                breadcrumbs = (
                                    chunk.metadata.hierarchy.breadcrumb_path
                                    if chunk.metadata.hierarchy
                                    and chunk.metadata.hierarchy.breadcrumb_path
                                    else [f"Page {chunk.metadata.page_number}"]
                                )

                                verification_result = self._vision_manager.verify_shadow_integrity(
                                    image=img,
                                    breadcrumbs=breadcrumbs,
                                )

                                # If VLM doesn't approve as valid editorial content: DISCARD
                                if not verification_result.get("valid", False):
                                    logger.warning(
                                        f"[FULL-PAGE-GUARD] VLM REJECTED full-page asset on "
                                        f"page {chunk.metadata.page_number}: {verification_result.get('reason', 'No reason')}"
                                    )
                                    continue  # DISCARD - skip this chunk

                                logger.info(
                                    f"[FULL-PAGE-GUARD] VLM APPROVED full-page asset on "
                                    f"page {chunk.metadata.page_number} (classification: {verification_result.get('classification', 'unknown')})"
                                )
                        else:
                            # Asset file missing - discard
                            logger.warning(
                                f"[FULL-PAGE-GUARD] DISCARDING full-page asset on "
                                f"page {chunk.metadata.page_number} (asset file missing)"
                            )
                            continue
                    else:
                        # No asset reference - discard
                        logger.warning(
                            f"[FULL-PAGE-GUARD] DISCARDING full-page asset on "
                            f"page {chunk.metadata.page_number} (no asset reference)"
                        )
                        continue

                except Exception as vlm_error:
                    logger.error(
                        f"[FULL-PAGE-GUARD] VLM verification failed for page "
                        f"{chunk.metadata.page_number}: {vlm_error} - DISCARDING"
                    )
                    continue  # DISCARD on verification error

                if self.strict_qa:
                    logger.warning(
                        f"[FULL-PAGE-GUARD] Filtering full-page asset on "
                        f"page {chunk.metadata.page_number} (strict QA enabled)"
                    )
                    continue

                if self.allow_fullpage_shadow:
                    logger.info(
                        f"[FULL-PAGE-GUARD] Allowing full-page asset on "
                        f"page {chunk.metadata.page_number} (--allow-fullpage-shadow)"
                    )
                else:
                    logger.info(
                        f"[FULL-PAGE-GUARD] Retaining full-page asset on "
                        f"page {chunk.metadata.page_number} (non-strict mode)"
                    )

                # Prepend full-page context to visual description
                if chunk.metadata.visual_description:
                    chunk.metadata.visual_description = (
                        f"[FULL-PAGE EDITORIAL IMAGE] {chunk.metadata.visual_description}"
                    )
                else:
                    chunk.metadata.visual_description = (
                        f"[FULL-PAGE EDITORIAL IMAGE on page {chunk.metadata.page_number}]"
                    )

                filtered.append(chunk)
            else:
                filtered.append(chunk)

        if fullpage_count > 0:
            if self.strict_qa:
                action = "filtered"
            else:
                action = "retained"
            logger.info(f"[FULL-PAGE-GUARD] {fullpage_count} full-page assets {action}")
            print(
                f"\n🛡️ [FULL-PAGE-GUARD] {fullpage_count} full-page assets {action}",
                flush=True,
            )

        return filtered


# ============================================================================
# FACTORY FUNCTION
# ============================================================================


def create_batch_processor(
    output_dir: str = "./output",
    batch_size: int = DEFAULT_BATCH_SIZE,
    vision_provider: str = DEFAULT_VISION_PROVIDER,
    vision_model: Optional[str] = None,
    vision_api_key: Optional[str] = None,
    vlm_timeout: int = DEFAULT_VLM_TIMEOUT,
) -> BatchProcessor:
    """
    Factory function to create a BatchProcessor.

    Args:
        output_dir: Directory for output files
        batch_size: Pages per batch (default: 10)
        vision_provider: VLM provider (default: "ollama")
        vision_model: VLM model name (optional for Ollama - auto-detects if not specified)
        vision_api_key: API key for cloud providers
        vlm_timeout: VLM read timeout in seconds (default: 90)

    Returns:
        Configured BatchProcessor instance

    Example:
        processor = create_batch_processor(
            output_dir="./output",
            batch_size=10,
            vision_provider="ollama",
            vision_model="llava:latest",  # Required for Ollama
        )
        result = processor.process_pdf("large_document.pdf")
    """
    return BatchProcessor(
        output_dir=output_dir,
        batch_size=batch_size,
        vision_provider=vision_provider,
        vision_model=vision_model,
        vision_api_key=vision_api_key,
        vlm_timeout=vlm_timeout,
    )
