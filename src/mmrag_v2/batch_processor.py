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
    ChunkType,
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
from .utils.coordinate_normalization import ensure_normalized
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
DEFAULT_EXPORT_WRITE_BATCH_SIZE: int = 25


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
        semantic_overlap_ratio: float = 0.15,
        # Phase 1B: Layout-aware OCR parameters
        ocr_mode: str = "legacy",
        ocr_confidence_threshold: float = 0.7,
        enable_doctr: bool = True,
        force_table_vlm: bool = False,
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
            force_table_vlm: Force table image -> VLM markdown path (fallback to OCR/docling if needed)
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
        self._semantic_overlap_ratio = semantic_overlap_ratio

        # Phase 1B: Layout-aware OCR parameters
        self.ocr_mode = ocr_mode
        self.ocr_confidence_threshold = ocr_confidence_threshold
        self.enable_doctr = enable_doctr
        self.force_table_vlm = force_table_vlm

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
        self._layout_processor = None

        # V2.4: Intelligence Stack Metadata (for observability)
        self._intelligence_metadata: Dict[str, Any] = {}

        # REQ-VLM-02: Track asset counts per page for low-recall trigger
        self._assets_per_page: Dict[int, int] = {}
        self._current_pdf_path: Optional[Path] = None

        # QA-CHECK-01: Initialize token validator for data integrity
        self._token_validator = create_token_validator(tolerance=qa_tolerance)

        # Quality Filter Tracker for token-level filtering analytics
        self._quality_filter_tracker: Optional[QualityFilterTracker] = None

        # REQ-COORD-02: Track page dimensions per page for UI overlay support
        self._page_dimensions: Dict[int, Tuple[int, int]] = {}

        # Track which original page numbers were actually processed. This is critical
        # when using --pages (max-pages) or specific pages, so QA validation/recovery
        # doesn't scan the entire PDF and trigger massive false recoveries.
        self._processed_pages: Optional[set[int]] = None

        # MEMORY FIX: Cache heavy objects across batches instead of re-creating per batch.
        # Docling DocumentConverter loads ~500MB+ of ML models (LayoutPredictor,
        # TableFormer, OCR) into MPS memory. Re-creating it 18× for a 54-page doc
        # causes catastrophic memory growth.
        self._docling_converter = None  # Cached Docling DocumentConverter
        self._shadow_processor = None   # Cached V2DocumentProcessor for shadow extraction

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
        # Adjust QA tolerance per profile (digital_magazine is allowed higher variance)
        profile = intelligence_metadata.get("profile_type")
        if profile == "digital_magazine":
            self._token_validator = create_token_validator(tolerance=0.18)
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

    def _sanitize_chunk_for_export(self, chunk: IngestionChunk) -> Dict[str, Any]:
        """
        Build a JSON-safe chunk dict with defensive bbox normalization.

        This prevents export crashes from malformed spatial metadata while keeping
        REQ-COORD invariants for emitted output.
        """
        self._enrich_asset_ref_from_disk(chunk)
        chunk_dict = chunk.model_dump(mode="json")

        # Ensure schema_version is emitted in metadata for downstream versioning
        meta = chunk_dict.get("metadata", {})
        if meta.get("schema_version") is None:
            meta["schema_version"] = SCHEMA_VERSION
            chunk_dict["metadata"] = meta

        spatial = meta.get("spatial")
        if isinstance(spatial, dict):
            bbox = spatial.get("bbox")
            if bbox is not None:
                page_w = spatial.get("page_width") or 612
                page_h = spatial.get("page_height") or 792
                context = f"chunk_id={chunk_dict.get('chunk_id', 'unknown')}"
                try:
                    spatial["bbox"] = ensure_normalized(
                        bbox=bbox,
                        page_width=float(page_w),
                        page_height=float(page_h),
                        context=context,
                    )
                except Exception as bbox_err:
                    # Keep pipeline alive: fallback for visual chunks, drop bbox for text.
                    logger.warning(
                        f"[FINALIZE] Invalid bbox normalized via fallback ({context}): {bbox_err}"
                    )
                    modality = str(chunk_dict.get("modality", "")).lower()
                    if modality in ("image", "table"):
                        spatial["bbox"] = [0, 0, COORD_SCALE, COORD_SCALE]
                    else:
                        spatial["bbox"] = None
            chunk_dict["metadata"]["spatial"] = spatial

        return chunk_dict

    def _enrich_asset_ref_from_disk(self, chunk: IngestionChunk) -> None:
        """Populate missing asset metadata (width/height/file size) from saved file."""
        asset_ref = getattr(chunk, "asset_ref", None)
        if not asset_ref or not asset_ref.file_path:
            return

        if (
            asset_ref.width_px is not None
            and asset_ref.height_px is not None
            and asset_ref.file_size_bytes is not None
        ):
            return

        asset_path = self.output_dir / asset_ref.file_path
        if not asset_path.exists() or not asset_path.is_file():
            return

        try:
            if asset_ref.file_size_bytes is None:
                asset_ref.file_size_bytes = int(asset_path.stat().st_size)
        except Exception:
            pass

        if asset_ref.width_px is None or asset_ref.height_px is None:
            try:
                with Image.open(asset_path) as img:
                    w, h = img.size
                if asset_ref.width_px is None:
                    asset_ref.width_px = int(w)
                if asset_ref.height_px is None:
                    asset_ref.height_px = int(h)
            except Exception:
                pass

    def _classify_text_content(self, text: str) -> str:
        """Deterministic classification for text chunk metadata."""
        lowered = (text or "").lower()
        ad_keywords = (
            "buy now",
            "special offer",
            "discount",
            "order now",
            "limited time",
            "subscribe",
        )
        if any(tok in lowered for tok in ad_keywords):
            return "advertisement"

        technical_keywords = (
            "api",
            "schema",
            "algorithm",
            "function",
            "class",
            "module",
            "pipeline",
            "configuration",
            "implementation",
            "model",
        )
        if sum(1 for tok in technical_keywords if tok in lowered) >= 2:
            return "technical"
        return "editorial"

    def _classify_recovery_text_content(self, text: str) -> str:
        """
        Deterministic non-null classification for recovery-generated TEXT chunks.

        Recovery chunks must always carry content_classification so downstream
        retrieval filters do not need special null handling.
        """
        return "code" if self._looks_like_code_text(text or "") else self._classify_text_content(text or "")

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
        if self._layout_processor is None:
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

            # Classify code-like OCR text (programming manuals/books).
            chunk_type = None
            content_classification = self._classify_text_content(pc.content or "") if modality == Modality.TEXT else None
            if modality == Modality.TEXT and self._looks_like_code_text(pc.content or ""):
                chunk_type = ChunkType.CODE
                content_classification = "code"

            # BUG-009 FIX + FIX 1: Propagate intelligence metadata AND ocr_confidence
            metadata = ChunkMetadata(
                page_number=pc.page_number,
                source_file=source_file,
                file_type=FileType.PDF,
                chunk_type=chunk_type,
                hierarchy=hierarchy,
                extraction_method=extraction_method,
                visual_description=pc.visual_description,
                spatial=spatial,
                content_classification=content_classification,
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
                    file_size_bytes=pc.asset_ref.get("file_size_bytes"),
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

        # MEMORY FIX: Reuse cached DocumentConverter across batches.
        # Creating a new converter per batch loads ~500MB+ of ML models
        # (LayoutPredictor, TableFormer, OCR) into MPS memory each time.
        if self._docling_converter is None:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.images_scale = 2.0
            pipeline_options.generate_page_images = (
                True  # ✅ REQ-PDF-04: FIXED - Enable page rendering for padding integrity
            )
            # MEMORY FIX: Disable picture/table image generation.
            # We crop regions ourselves in layout_aware_processor using the
            # PyMuPDF-rendered page image.  Docling's copies are never used
            # and waste ~5-15 MB per batch that is never freed.
            pipeline_options.generate_picture_images = False
            pipeline_options.generate_table_images = False
            pipeline_options.do_ocr = True

            self._docling_converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
            )
            logger.info("[DOCLING-LAYOUT] Created and cached DocumentConverter (models loaded once)")
        else:
            logger.info("[DOCLING-LAYOUT] Reusing cached DocumentConverter")

        result = self._docling_converter.convert(str(batch_path))
        docling_doc = result.document

        # Group elements by page — convert to LIGHTWEIGHT dicts immediately
        # so the heavy Docling Document (with page images) can be freed.
        from types import SimpleNamespace

        elements_per_page: Dict[int, List[Any]] = {}

        for item_tuple in docling_doc.iterate_items():
            element, _ = item_tuple

            # Get page number from provenance
            batch_page_no = 1
            prov_bbox = None
            if hasattr(element, "prov") and element.prov:
                prov = element.prov[0] if isinstance(element.prov, list) else element.prov
                if hasattr(prov, "page_no") and prov.page_no is not None:
                    batch_page_no = prov.page_no
                # Extract bbox data as lightweight copy
                if hasattr(prov, "bbox") and prov.bbox:
                    bbox_obj = prov.bbox
                    if hasattr(bbox_obj, "l"):
                        prov_bbox = SimpleNamespace(
                            l=bbox_obj.l, r=bbox_obj.r,
                            t=bbox_obj.t, b=bbox_obj.b,
                        )

            actual_page_no = batch_page_no + page_offset

            # Build lightweight element that matches the interface
            # expected by layout_aware_processor._convert_docling_elements
            label_obj = getattr(element, "label", None)
            label_ns = None
            if label_obj is not None:
                label_ns = SimpleNamespace(
                    value=str(label_obj.value) if hasattr(label_obj, "value") else str(label_obj)
                )

            text_val = str(getattr(element, "text", "")) if hasattr(element, "text") else ""

            light_elem = SimpleNamespace(
                label=label_ns,
                prov=[SimpleNamespace(bbox=prov_bbox, page_no=batch_page_no)] if prov_bbox else [],
                text=text_val,
            )

            if actual_page_no not in elements_per_page:
                elements_per_page[actual_page_no] = []
            elements_per_page[actual_page_no].append(light_elem)

        total = sum(len(e) for e in elements_per_page.values())
        logger.info(
            f"[DOCLING-LAYOUT] Extracted {total} elements across {len(elements_per_page)} pages"
        )

        # MEMORY FIX: Explicitly break references to Docling's heavy document
        # object which holds rendered page images (~10 MB per page at 2x scale).
        # The lightweight elements above have no back-references to docling_doc.
        del docling_doc
        del result
        gc.collect()

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

                # For digital technical-manual pages, OCR can under-read dense TOC/index
                # layouts. When extracted text is much shorter than the native text layer,
                # replace OCR text chunks with a clean PDF-text fallback chunk.
                profile_type = str(self._intelligence_metadata.get("profile_type", "unknown"))
                if classification == "digital" and profile_type == "technical_manual":
                    try:
                        page_obj = doc.load_page(batch_page_idx)
                        native_text = (page_obj.get_text("text") or "").strip()
                        if len(native_text) >= 120:
                            extracted_text_chars = sum(
                                len(c.content or "") for c in page_chunks if c.modality == Modality.TEXT
                            )
                            native_chars = len(native_text)
                            if extracted_text_chars < int(native_chars * 0.80):
                                text_fallback_chunks = [c for c in page_chunks if c.modality != Modality.TEXT]
                                doc_title = Path(source_file).stem if source_file else "Document"
                                fallback_parts = self._split_nearest_paragraph_breaks(
                                    text=native_text,
                                    max_chars=1200,
                                    overlap_chars=120,
                                )
                                fallback_parts = [p for p in fallback_parts if p and p.strip()]
                                if not fallback_parts:
                                    fallback_parts = [native_text]

                                part_total = len(fallback_parts)
                                for idx, part_text in enumerate(fallback_parts):
                                    breadcrumb = [
                                        doc_title,
                                        f"Page {actual_page_no}",
                                        "[DIGITAL-TEXT-FALLBACK]",
                                    ]
                                    if part_total > 1:
                                        breadcrumb.append(f"[Part {idx + 1}/{part_total}]")

                                    hierarchy = HierarchyMetadata(
                                        parent_heading=None,
                                        breadcrumb_path=breadcrumb,
                                        level=len(breadcrumb),
                                    )
                                    fallback_chunk = create_text_chunk(
                                        doc_id=self._doc_hash or "unknown",
                                        content=part_text,
                                        source_file=source_file,
                                        file_type=FileType.PDF,
                                        page_number=actual_page_no,
                                        hierarchy=hierarchy,
                                        extraction_method="digital_text_layer_fallback",
                                        **self._intelligence_metadata,
                                    )
                                    text_fallback_chunks.append(fallback_chunk)

                                page_chunks = text_fallback_chunks
                                logger.info(
                                    f"[DIGITAL-TEXT-FALLBACK] Page {actual_page_no}: replaced "
                                    f"OCR text chunks ({extracted_text_chars} chars) with "
                                    f"{part_total} native text chunk(s) ({native_chars} chars)"
                                )
                    except Exception as fallback_err:
                        logger.debug(
                            f"[DIGITAL-TEXT-FALLBACK] Page {actual_page_no}: skipped due to error: {fallback_err}"
                        )

                # Nuclear fallback for scanned_degraded code pages:
                # stitch fragmented OCR code lines, then optionally re-transcribe via VLM.
                if self._is_scanned_degraded_profile():
                    page_chunks = self._nuclear_code_fix(
                        page_chunks=page_chunks,
                        page_image=page_image,
                        page_number=actual_page_no,
                    )

                # REQ-STRUCT-01: Flat Code OCR Rescue — active for ALL profiles when
                # has_flat_text_corruption=True (broken PDF generator stripped newlines).
                if self._intelligence_metadata.get("has_flat_text_corruption"):
                    page_chunks = self._flat_code_ocr_rescue(
                        page_chunks=page_chunks,
                        pdf_doc=doc,
                        pdf_page_idx=batch_page_idx,
                        page_number=actual_page_no,
                    )

                all_chunks.extend(page_chunks)

                # Log chunk types
                text_count = sum(1 for c in page_chunks if c.modality == Modality.TEXT)
                image_count = sum(1 for c in page_chunks if c.modality == Modality.IMAGE)
                print(
                    f"        → {len(page_chunks)} chunks (text: {text_count}, image: {image_count})",
                    flush=True,
                )

                # MEMORY FIX: Release page image AND consumed Docling elements
                # to avoid accumulating stale references across the page loop.
                del page_image
                if actual_page_no in docling_elements_per_page:
                    del docling_elements_per_page[actual_page_no]
                gc.collect()

            # ================================================================
            # SHADOW EXTRACTION WITHIN TRY BLOCK (BEFORE doc.close())
            # FIX: "Document Closed" error - shadow extraction needs PDF open
            # ================================================================
            print("    🔍 [SHADOW] Running shadow extraction scan...", flush=True)
            logger.info("[SHADOW-EXTRACTION] Running shadow scan BEFORE closing PDF...")

            # Import processor to access shadow extraction
            from .processor import V2DocumentProcessor

            # MEMORY FIX: Reuse cached V2DocumentProcessor for shadow extraction.
            # Creating a new processor per batch re-initializes Docling internals.
            if self._shadow_processor is None:
                self._shadow_processor = V2DocumentProcessor(
                    output_dir=str(self.output_dir),
                    enable_ocr=False,
                    vision_provider="none",
                    external_vision_manager=self._vision_manager,
                    doc_hash_override=self._doc_hash,
                    source_file_override=source_file,
                    extraction_strategy=self.extraction_strategy,
                    intelligence_metadata=self._intelligence_metadata,
                    force_table_vlm=self.force_table_vlm,
                )
                logger.info("[SHADOW-EXTRACTION] Created and cached shadow processor")
            else:
                # Update mutable state that may change between batches
                self._shadow_processor._doc_hash_override = self._doc_hash
                self._shadow_processor._source_file_override = source_file
                logger.info("[SHADOW-EXTRACTION] Reusing cached shadow processor")
            temp_processor = self._shadow_processor

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

                shadow_count = 0
                for shadow_chunk in shadow_chunks_generator:
                    all_chunks.append(shadow_chunk)
                    shadow_count += 1

                if shadow_count > 0:
                    print(
                        f"    ✓ [SHADOW] Found {shadow_count} additional shadow assets",
                        flush=True,
                    )
                    logger.info(f"[SHADOW-EXTRACTION] Added {shadow_count} shadow assets")
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

    def _is_scanned_degraded_profile(self) -> bool:
        profile = str(self._intelligence_metadata.get("profile_type", "") or "").strip().lower()
        return profile == "scanned_degraded"

    def _nuclear_code_fix(
        self,
        page_chunks: List[IngestionChunk],
        page_image: np.ndarray,
        page_number: int,
    ) -> List[IngestionChunk]:
        """
        Stitch fragmented OCR code chunks and optionally transcribe stitched blocks via VLM.

        This is intentionally restricted to scanned_degraded profile pages.
        """
        if not page_chunks or self._vision_manager is None:
            return page_chunks

        import re

        page_h, page_w = page_image.shape[:2] if page_image is not None else (0, 0)
        if page_h <= 0 or page_w <= 0:
            return page_chunks

        pil_page: Optional[Image.Image] = None
        max_vlm_calls_per_page = 3
        vlm_calls = 0
        merged_groups = 0
        transcribed_groups = 0
        merge_serial = 0

        def _bbox_of(ch: IngestionChunk) -> Optional[List[int]]:
            try:
                bb = ch.metadata.spatial.bbox if ch.metadata and ch.metadata.spatial else None
                if bb and len(bb) == 4:
                    return [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])]
            except Exception:
                return None
            return None

        def _union_bbox(a: Optional[List[int]], b: Optional[List[int]]) -> Optional[List[int]]:
            if a is None:
                return b
            if b is None:
                return a
            return [
                min(a[0], b[0]),
                min(a[1], b[1]),
                max(a[2], b[2]),
                max(a[3], b[3]),
            ]

        def _to_pixels(bbox_norm: Optional[List[int]], padding: int = 10) -> Optional[Tuple[int, int, int, int]]:
            if bbox_norm is None:
                return None
            x0 = int((bbox_norm[0] / COORD_SCALE) * page_w) - padding
            y0 = int((bbox_norm[1] / COORD_SCALE) * page_h) - padding
            x1 = int((bbox_norm[2] / COORD_SCALE) * page_w) + padding
            y1 = int((bbox_norm[3] / COORD_SCALE) * page_h) + padding

            x0 = max(0, min(page_w - 1, x0))
            y0 = max(0, min(page_h - 1, y0))
            x1 = max(1, min(page_w, x1))
            y1 = max(1, min(page_h, y1))
            if x1 - x0 < 16 or y1 - y0 < 16:
                return None
            return (x0, y0, x1, y1)

        def _is_code_candidate(ch: IngestionChunk) -> bool:
            if ch.modality != Modality.TEXT:
                return False
            txt = (ch.content or "").strip("\n")
            if not txt.strip():
                return False
            line_text = txt.strip()
            if re.fullmatch(
                r"(?:from\s+[A-Za-z_][\w\.]*\s+import\s+[A-Za-z_][\w\.,\s]*|import\s+[A-Za-z_][\w\.]*(?:\s*,\s*[A-Za-z_][\w\.]*)*(?:\s+as\s+[A-Za-z_][\w]*)?)",
                line_text,
            ):
                return False

            words = re.findall(r"[A-Za-z]{2,}", txt)
            if (
                len(words) >= 12
                and txt[:1].isupper()
                and any(p in txt for p in (".", ";", "!", "?"))
                and not re.search(r"[{}()\[\]=]", txt)
            ):
                return False

            try:
                if (
                    ch.metadata.chunk_type == ChunkType.CODE
                    or ch.metadata.content_classification == "code"
                ):
                    return True
            except Exception:
                pass

            if txt.startswith(("    ", "\t")):
                return True

            if re.search(
                r"(?m)^\s*(def|class|if\s+.*:|elif\s+.*:|else:|for\s+.*:|while\s+.*:|try:|except\b.*:|finally:|return\b|with\s+.*:|yield\b|raise\b|pass\b|break\b|continue\b)\s*$",
                txt,
            ):
                return True
            if re.search(r"(?m)^\s*from\s+[A-Za-z_][\w\.]*\s+import\b", txt):
                return True
            if re.search(
                r"(?m)^\s*import\s+[A-Za-z_][\w\.]*(\s*,\s*[A-Za-z_][\w\.]*)*(\s+as\s+[A-Za-z_][\w]*)?\s*$",
                txt,
            ):
                return True
            if re.search(r"(?m)^\s*[A-Za-z_][\w\.]*(?:\[[^\]]+\])?\s*=\s*.+$", txt):
                return True
            if re.search(r"(?m)^\s*(?:>>>|\.\.\.)\s*", txt):
                return True
            if re.search(r"(?m)^\s*#\s*\S+", txt):
                return True
            if self._looks_like_code_text(txt) and re.search(r"[=\(\)\[\]\{\}:#]", txt):
                return True
            if (
                re.search(r"[=\(\)\[\]\{\}:#]", txt)
                and len(txt) <= 140
                and len(re.findall(r"[A-Za-z]{2,}", txt)) <= 16
            ):
                return True
            return False

        def _looks_like_code_continuation(ch: IngestionChunk) -> bool:
            if ch.modality != Modality.TEXT:
                return False
            txt = (ch.content or "").strip("\n")
            if not txt.strip():
                return False
            if _is_code_candidate(ch):
                return True
            if re.search(
                r"(?m)^\s*(print|assert|del|return|yield|pass|break|continue|raise)\b",
                txt,
            ):
                return True
            if re.search(
                r"(?m)^\s*(if|elif|else|for|while|try|except|finally|with)\b.*:?\s*$",
                txt,
            ):
                return True
            if re.search(r"(?m)^\s*[A-Za-z_][\w\.]*\s*\(.*\)\s*$", txt) and len(txt) <= 160:
                return True
            if re.search(r"(?m)^\s*[A-Za-z_][\w\.]*(?:\[[^\]]+\])?\s*=\s*.+$", txt):
                return True
            if re.search(r"(?m)^\s*(?:>>>|\.\.\.)\s*", txt):
                return True
            if re.search(r"[=\(\)\[\]\{\}:#]", txt) and len(txt) <= 120:
                return True
            return False

        def _is_suspicious_code_noise(text: str) -> bool:
            if not text:
                return False
            t = text
            if re.search(r"[A-Za-z]{2,}\d+[A-Za-z]{2,}", t):
                return True
            if re.search(r"[A-Za-z]{2,}[;:,][A-Za-z]{2,}", t):
                return True
            if re.search(r"\b(?:initia|intere;|retum|ciass|moduie)\b", t.lower()):
                return True
            return False

        def _has_code_signals(text: str) -> bool:
            if not text:
                return False
            return (
                re.search(
                    r"(?m)^\s*(def|class|if\s+.*:|elif\s+.*:|else:|for\s+.*:|while\s+.*:|try:|except\b.*:|return\b|with\s+.*:)\s*$",
                    text,
                )
                is not None
                or re.search(r"(?m)^\s*from\s+[A-Za-z_][\w\.]*\s+import\b", text) is not None
                or re.search(
                    r"(?m)^\s*import\s+[A-Za-z_][\w\.]*(\s*,\s*[A-Za-z_][\w\.]*)*(\s+as\s+[A-Za-z_][\w]*)?\s*$",
                    text,
                )
                is not None
                or re.search(r"(?m)^\s*[A-Za-z_][\w\.]*\s*=\s*.+$", text) is not None
                or re.search(r"[=\(\)\[\]\{\}:#]", text) is not None
                or re.search(r"(?m)^\s*>>>", text) is not None
            )

        def _can_attach(prev_ch: IngestionChunk, cur_ch: IngestionChunk) -> bool:
            prev_bbox = _bbox_of(prev_ch)
            cur_bbox = _bbox_of(cur_ch)
            if prev_bbox is None or cur_bbox is None:
                return False

            v_gap = cur_bbox[1] - prev_bbox[3]
            prev_h = max(1, prev_bbox[3] - prev_bbox[1])
            cur_h = max(1, cur_bbox[3] - cur_bbox[1])
            max_v_gap = max(140, int(max(prev_h, cur_h) * 4))
            if v_gap < -10 or v_gap > max_v_gap:
                return False

            prev_w = max(1, prev_bbox[2] - prev_bbox[0])
            cur_w = max(1, cur_bbox[2] - cur_bbox[0])
            overlap = max(0, min(prev_bbox[2], cur_bbox[2]) - max(prev_bbox[0], cur_bbox[0]))
            overlap_ratio = overlap / float(max(1, min(prev_w, cur_w)))

            if overlap_ratio >= 0.05:
                return True

            left_shift = cur_bbox[0] - prev_bbox[0]
            if -70 <= left_shift <= 180 and cur_bbox[0] <= prev_bbox[2] + 180:
                return True

            return False

        def _should_transcribe(merged_text: str, group_size: int) -> bool:
            if vlm_calls >= max_vlm_calls_per_page:
                return False
            text = (merged_text or "").strip()
            if not text:
                return False
            line_count = max(1, text.count("\n") + 1)
            has_leading_indent = any(
                ln.startswith(("    ", "\t")) for ln in text.splitlines() if ln.strip()
            )
            if _is_suspicious_code_noise(text):
                return True
            if _has_code_signals(text) and len(text) >= 30 and group_size >= 2:
                return True
            if (
                group_size == 1
                and line_count == 1
                and _has_code_signals(text)
                and (text.rstrip().endswith(":") or text.lstrip().startswith(("def ", "class ", ">>>", "...")))
            ):
                return True
            if group_size <= 2 and line_count <= 3 and not has_leading_indent and _has_code_signals(text):
                return True
            if line_count >= 4 and not has_leading_indent and _has_code_signals(text):
                return True
            return False

        def _transcribe_if_needed(
            base: IngestionChunk,
            merged_text: str,
            merged_bbox: Optional[List[int]],
            group_size: int,
        ) -> None:
            nonlocal vlm_calls, transcribed_groups, pil_page

            if not _should_transcribe(merged_text, group_size):
                return

            crop_px = _to_pixels(merged_bbox, padding=12)
            if crop_px is None:
                return

            x0, y0, x1, y1 = crop_px
            if group_size == 1 and merged_text.strip().endswith(":"):
                expand_down = max(80, int(page_h * 0.05))
                y1 = min(page_h, y1 + expand_down)
            if group_size == 1 and len(merged_text.strip()) < 40:
                expand_down = max(60, int(page_h * 0.03))
                y1 = min(page_h, y1 + expand_down)
            crop_px = (x0, y0, x1, y1)

            if pil_page is None:
                pil_page = Image.fromarray(page_image)
            crop = pil_page.crop(crop_px)
            transcribed = self._vision_manager.transcribe_code_block(
                crop,
                page_number=page_number,
            )
            if transcribed:
                base.content = transcribed
                base.metadata.extraction_method = "vlm_code_transcribed"
                vlm_calls += 1
                transcribed_groups += 1

        def _flush_buffer(buffer: List[IngestionChunk], output: List[IngestionChunk]) -> None:
            nonlocal merged_groups, merge_serial

            if not buffer:
                return
            if len(buffer) == 1:
                base = buffer[0].model_copy(deep=True)
                if base.modality == Modality.TEXT:
                    if base.metadata.chunk_type != ChunkType.CODE:
                        base.metadata.chunk_type = ChunkType.CODE
                    base.metadata.content_classification = "code"
                    if base.metadata.extraction_method == "ocr":
                        base.metadata.extraction_method = "ocr_code_stitched"
                single_bbox = _bbox_of(base)
                _transcribe_if_needed(
                    base=base,
                    merged_text=base.content or "",
                    merged_bbox=single_bbox,
                    group_size=1,
                )
                output.append(base)
                buffer.clear()
                return

            merge_serial += 1
            merged_groups += 1

            base = buffer[0].model_copy(deep=True)
            merged_bbox: Optional[List[int]] = None
            merged_lines: List[str] = []
            for entry in buffer:
                merged_bbox = _union_bbox(merged_bbox, _bbox_of(entry))
                line = (entry.content or "").strip("\n")
                if line:
                    merged_lines.append(line)

            merged_text = "\n".join(merged_lines).strip("\n")
            base.content = merged_text
            base.chunk_id = f"{base.chunk_id}_nc{merge_serial}"

            if base.metadata.spatial is None:
                base.metadata.spatial = SpatialMetadata(bbox=None)
            base.metadata.spatial.bbox = merged_bbox
            base.metadata.chunk_type = ChunkType.CODE
            base.metadata.content_classification = "code"
            base.metadata.extraction_method = "ocr_code_stitched"

            if len(buffer) > 1:
                if base.semantic_context is None:
                    base.semantic_context = SemanticContext()
                first_ctx = buffer[0].semantic_context
                last_ctx = buffer[-1].semantic_context
                if first_ctx and first_ctx.prev_text_snippet:
                    base.semantic_context.prev_text_snippet = first_ctx.prev_text_snippet
                if last_ctx and last_ctx.next_text_snippet:
                    base.semantic_context.next_text_snippet = last_ctx.next_text_snippet

            _transcribe_if_needed(
                base=base,
                merged_text=merged_text,
                merged_bbox=merged_bbox,
                group_size=len(buffer),
            )

            output.append(base)
            buffer.clear()

        merged_chunks: List[IngestionChunk] = []
        code_buffer: List[IngestionChunk] = []

        for chunk in page_chunks:
            if not _is_code_candidate(chunk):
                if code_buffer and _can_attach(code_buffer[-1], chunk) and _looks_like_code_continuation(chunk):
                    code_buffer.append(chunk)
                    continue
                _flush_buffer(code_buffer, merged_chunks)
                merged_chunks.append(chunk)
                continue

            if code_buffer and _can_attach(code_buffer[-1], chunk):
                code_buffer.append(chunk)
            else:
                _flush_buffer(code_buffer, merged_chunks)
                code_buffer.append(chunk)

        _flush_buffer(code_buffer, merged_chunks)

        if merged_groups > 0 or transcribed_groups > 0:
            logger.info(
                f"[NUCLEAR-CODE] Page {page_number}: stitched_groups={merged_groups} "
                f"vlm_transcribed={transcribed_groups} "
                f"calls={vlm_calls}/{max_vlm_calls_per_page}"
            )

        return merged_chunks

    def _flat_code_ocr_rescue(
        self,
        page_chunks: List[IngestionChunk],
        pdf_doc: "fitz.Document",
        pdf_page_idx: int,
        page_number: int,
    ) -> List[IngestionChunk]:
        """
        REQ-STRUCT-01 / Flat Code OCR Rescue (v2.5.0).

        Runs for any profile when has_flat_text_corruption=True in intelligence_metadata.
        Detects CODE chunks whose content has no embedded newlines and is longer than
        120 characters (signature of a broken PDF generator that stripped all \\n from
        the code stream), renders the page crop via PyMuPDF at 150 DPI, runs Tesseract
        on the crop, and replaces the chunk content if the OCR result is better structured
        (contains >= 2 newlines).

        Unlike _nuclear_code_fix (which stitches fragmented OCR lines), this method
        re-reads already-extracted CODE chunks via OCR to restore their internal structure.
        It does NOT require scanned_degraded profile.

        Args:
            page_chunks: Chunks produced for this page.
            pdf_doc: Open fitz.Document (must stay open during this call).
            pdf_page_idx: 0-based page index within pdf_doc.
            page_number: 1-based page number for logging.

        Returns:
            Updated chunk list with flat code chunks re-extracted where possible.
        """
        import re

        try:
            import pytesseract
            from PIL import Image as PILImage
        except ImportError:
            logger.debug("[FLAT-CODE-RESCUE] pytesseract not available; skipping.")
            return page_chunks

        # Quick pass: any candidates at all?
        def _is_flat_code_candidate(ch: IngestionChunk) -> bool:
            if ch.modality != Modality.TEXT:
                return False
            try:
                is_code = (
                    ch.metadata.chunk_type == ChunkType.CODE
                    or ch.metadata.content_classification == "code"
                )
            except Exception:
                is_code = False
            if not is_code:
                return False
            txt = ch.content or ""
            return len(txt) > 120 and "\n" not in txt

        candidates = [ch for ch in page_chunks if _is_flat_code_candidate(ch)]
        if not candidates:
            return page_chunks

        # Render the page at 150 DPI for good OCR quality
        try:
            fitz_page = pdf_doc.load_page(pdf_page_idx)
            mat = fitz.Matrix(150 / 72, 150 / 72)
            pixmap = fitz_page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
            page_h_px = pixmap.height
            page_w_px = pixmap.width
            page_pil = PILImage.frombytes("L", [page_w_px, page_h_px], pixmap.samples)
        except Exception as exc:
            logger.debug(f"[FLAT-CODE-RESCUE] Page {page_number}: could not render page: {exc}")
            return page_chunks

        rescued = 0
        for chunk in candidates:
            try:
                bbox = chunk.metadata.spatial.bbox if chunk.metadata and chunk.metadata.spatial else None
                if not bbox or len(bbox) != 4:
                    continue

                # Denormalise bbox (stored in COORD_SCALE=1000 units) to pixel coords
                padding = 12
                x0 = max(0, int((bbox[0] / COORD_SCALE) * page_w_px) - padding)
                y0 = max(0, int((bbox[1] / COORD_SCALE) * page_h_px) - padding)
                x1 = min(page_w_px, int((bbox[2] / COORD_SCALE) * page_w_px) + padding)
                y1 = min(page_h_px, int((bbox[3] / COORD_SCALE) * page_h_px) + padding)

                if x1 - x0 < 20 or y1 - y0 < 20:
                    continue

                crop = page_pil.crop((x0, y0, x1, y1))
                ocr_text = pytesseract.image_to_string(
                    crop, lang="eng", config="--psm 6 -c preserve_interword_spaces=1"
                )
                ocr_text = ocr_text.strip()

                if not ocr_text or ocr_text.count("\n") < 2:
                    # OCR didn't recover meaningful structure — keep original
                    continue

                # Accept OCR result only if it's longer or better-structured than original
                original = chunk.content or ""
                if len(ocr_text) < len(original) * 0.3:
                    # OCR result is suspiciously short vs. original — skip
                    continue

                chunk.content = ocr_text
                chunk.metadata.extraction_method = "flat_code_ocr_rescue"
                rescued += 1

            except Exception as exc:
                logger.debug(f"[FLAT-CODE-RESCUE] Page {page_number}: chunk rescue failed: {exc}")

        if rescued > 0:
            logger.info(
                f"[FLAT-CODE-RESCUE] Page {page_number}: rescued {rescued}/{len(candidates)} flat code chunks via Tesseract."
            )
        else:
            logger.debug(
                f"[FLAT-CODE-RESCUE] Page {page_number}: {len(candidates)} flat candidates found but none improved by OCR."
            )

        return page_chunks

    def _flat_code_rescue_legacy_pass(
        self,
        chunks: List[IngestionChunk],
        batch_path: "Path",
        page_offset: int,
    ) -> List[IngestionChunk]:
        """
        REQ-STRUCT-01 / Flat Code OCR Rescue — legacy-path post-pass (v2.5.0).

        Wraps _flat_code_ocr_rescue() for the standard Docling (legacy) processing path.
        Groups flat code chunks by page, opens the batch PDF read-only, and dispatches
        the per-page rescue.  This fires regardless of enable_ocr / ocr_mode settings
        because it is a targeted per-chunk correction, not a full OCR run.

        Args:
            chunks: All chunks produced for this batch.
            batch_path: Path to the split batch PDF file.
            page_offset: 0-based page offset for this batch in the original document.

        Returns:
            Updated chunk list with flat code chunks re-extracted where possible.
        """
        try:
            import pytesseract  # noqa: F401 — abort early if not available
        except ImportError:
            logger.debug("[FLAT-CODE-RESCUE-LEGACY] pytesseract not available; skipping.")
            return chunks

        # Collect page numbers that have at least one flat code candidate.
        def _is_flat_code(ch: IngestionChunk) -> bool:
            if ch.modality != Modality.TEXT:
                return False
            try:
                is_code = (
                    ch.metadata.chunk_type == ChunkType.CODE
                    or ch.metadata.content_classification == "code"
                )
            except Exception:
                is_code = False
            if not is_code:
                return False
            txt = ch.content or ""
            return len(txt) > 120 and "\n" not in txt

        # Group chunks by page_number (1-based in original doc).
        pages_with_flat: set = set()
        for ch in chunks:
            if _is_flat_code(ch):
                try:
                    pn = ch.metadata.page_number
                    if pn is not None:
                        pages_with_flat.add(int(pn))
                except Exception:
                    pass

        if not pages_with_flat:
            return chunks

        doc: Optional[fitz.Document] = None
        try:
            doc = fitz.open(str(batch_path))

            for page_no in sorted(pages_with_flat):
                # 0-based index within the batch PDF
                batch_page_idx = page_no - page_offset - 1
                if batch_page_idx < 0 or batch_page_idx >= len(doc):
                    continue

                # Isolate chunks for this page, run rescue, put them back.
                page_chunks = [ch for ch in chunks if (
                    ch.metadata is not None
                    and getattr(ch.metadata, "page_number", None) == page_no
                )]
                if not page_chunks:
                    continue

                rescued = self._flat_code_ocr_rescue(
                    page_chunks=page_chunks,
                    pdf_doc=doc,
                    pdf_page_idx=batch_page_idx,
                    page_number=page_no,
                )
                # The rescue mutates chunks in-place via chunk.content assignment,
                # so the original `chunks` list is already updated.

        except Exception as exc:
            logger.debug(f"[FLAT-CODE-RESCUE-LEGACY] Post-pass failed (non-fatal): {exc}")
        finally:
            if doc is not None:
                try:
                    doc.close()
                except Exception:
                    pass

        return chunks

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
        if self.ocr_mode == "layout-aware" and self.enable_ocr:
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
        elif self.ocr_mode == "layout-aware" and not self.enable_ocr:
            logger.info(
                "[BATCH] OCR mode is layout-aware, but OCR is disabled. "
                "Falling back to legacy pipeline."
            )

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
            force_table_vlm=self.force_table_vlm,
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
        finally:
            # CRITICAL: capture state before cleanup so breadcrumb continuity
            # survives per-batch teardown.
            try:
                self._context_state = processor.get_final_state()
            except Exception as state_err:
                logger.debug(
                    f"[BATCH-CLEANUP] Unable to capture final state for batch "
                    f"{batch_info.batch_index + 1}: {state_err}"
                )

            # Explicit processor cleanup per batch prevents converter/model buildup
            # across long runs (e.g. 54-page manuals).
            try:
                processor.cleanup()
            except Exception as cleanup_err:
                logger.debug(f"[BATCH-CLEANUP] Processor cleanup skipped: {cleanup_err}")
            try:
                del processor
            except Exception:
                pass
            gc.collect()

        # REQ-STRUCT-01: Flat Code OCR Rescue — legacy-path post-pass.
        # The layout-aware path runs rescue inline (per page). For the legacy Docling path
        # (digital PDFs, OCR disabled) we do a single targeted pass over all chunks here,
        # opening the batch PDF read-only just for page rendering.
        # This fires regardless of enable_ocr/ocr_mode because it is a per-chunk correction,
        # not a full OCR run.
        if self._intelligence_metadata.get("has_flat_text_corruption") and chunks:
            chunks = self._flat_code_rescue_legacy_pass(
                chunks=chunks,
                batch_path=batch_info.batch_path,
                page_offset=batch_info.page_offset,
            )

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
        prefinal_vision_stats: Dict[str, Any] = {}

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

        # OCR strategy: respect user flags, but auto-disable for digital-like PDFs
        # (native_digital or image_heavy) unless --force-ocr is explicitly set.
        #
        # Per AGENTS.md: "Combat Aircraft" / text-in-graphics recovery is known debt.
        # We keep the pipeline simple and stable here.
        doc_modality = self._intelligence_metadata.get("document_modality")
        profile_type = (self._intelligence_metadata.get("profile_type") or "").lower()
        is_digital_like = doc_modality in ("native_digital", "image_heavy")

        # REQ-STRUCT-02: Override OCR guard when encoding corruption is detected.
        # Even if the document looks digital (native_digital), if the text layer is
        # encoding-garbage (CIDFont / broken char map), we MUST force full OCR.
        has_encoding_corruption = bool(self._intelligence_metadata.get("has_encoding_corruption"))
        if has_encoding_corruption and is_digital_like and not self.force_ocr:
            self.force_ocr = True
            logger.warning(
                f"[OCR-GUARD] ENCODING CORRUPTION detected on digital-like PDF "
                f"(modality={doc_modality}); overriding force_ocr=True to bypass corrupt text layer."
            )

        # Default policy (AGENTS.md): avoid OCR cascade on digital-like PDFs unless user explicitly forces it.
        # This applies to all profiles, including technical_manual.
        if is_digital_like and not self.force_ocr:
            self.ocr_mode = "legacy"
            self.enable_ocr = False
            self.enable_doctr = False
            logger.info(
                f"[OCR-GUARD] Digital-like modality={doc_modality} "
                "(force_ocr=False); "
                "OCR cascade disabled (legacy mode, enable_ocr=False, enable_doctr=False)"
            )
        elif is_digital_like and self.force_ocr:
            # User explicitly wants OCR on digital PDF - respect the flag for recovery phases
            logger.info(
                f"[OCR-GUARD] Digital-like modality={doc_modality} BUT force_ocr=True; preserving OCR settings "
                f"(mode={self.ocr_mode}, enable_ocr={self.enable_ocr}, enable_doctr={self.enable_doctr}) "
                f"for recovery phase compatibility"
            )
        else:
            # Non-digital or unknown modality can still use configured OCR defaults
            logger.info(
                f"[OCR-GUARD] Modality={doc_modality or 'unknown'}; respecting configured OCR settings "
                f"(mode={self.ocr_mode}, enable_ocr={self.enable_ocr}, enable_doctr={self.enable_doctr})"
            )

        # Hard governance: if OCR is disabled for this run, force legacy routing.
        if not self.enable_ocr:
            if self.force_ocr:
                logger.warning(
                    "[OCR-GOVERNANCE] force_ocr=True ignored because enable_ocr=False; "
                    "forcing force_ocr=False for this run"
                )
                self.force_ocr = False
            if self.ocr_mode != "legacy":
                logger.info(
                    f"[OCR-GOVERNANCE] OCR disabled; overriding ocr_mode={self.ocr_mode} -> legacy"
                )
            self.ocr_mode = "legacy"
            self.enable_doctr = False

        # Guardrail: when OCR is disabled for this run, OCR-hint injection must
        # be disabled as well (it uses EasyOCR runtime and can cause late OOM).
        if (not self.enable_ocr) and self._profile_params and self._profile_params.enable_ocr_hints:
            try:
                from dataclasses import replace

                self._profile_params = replace(self._profile_params, enable_ocr_hints=False)
            except Exception:
                self._profile_params.enable_ocr_hints = False
            logger.info(
                "[OCR-HINT-GUARD] Disabled profile OCR hints because OCR is disabled "
                "(enable_ocr=False)"
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
                            # Create a trimmed batch PDF that contains ONLY the required pages.
                            # This keeps processing faithful to --pages/max_pages and prevents
                            # downstream QA/recovery from seeing "extra" processed content.
                            try:
                                start_0 = batch.start_page - 1
                                end_0 = self.max_pages - 1
                                trimmed_name = (
                                    f"batch_{batch.batch_index:03d}_p"
                                    f"{batch.start_page}-{self.max_pages}_trim.pdf"
                                )
                                trimmed_path = split_result.temp_dir / trimmed_name

                                src_doc = fitz.open(str(pdf_path))
                                out_doc = fitz.open()
                                try:
                                    out_doc.insert_pdf(src_doc, from_page=start_0, to_page=end_0)
                                    out_doc.save(str(trimmed_path))
                                finally:
                                    out_doc.close()
                                    src_doc.close()

                                trimmed_batch = BatchInfo(
                                    batch_index=batch.batch_index,
                                    batch_path=trimmed_path,
                                    start_page=batch.start_page,
                                    end_page=self.max_pages,
                                    page_count=self.max_pages - batch.start_page + 1,
                                    page_offset=start_0,
                                )
                                filtered_batches.append(trimmed_batch)
                                pages_included = self.max_pages
                                break
                            except Exception as trim_err:
                                logger.warning(
                                    f"[CORE] Failed to trim batch {batch.batch_index + 1} "
                                    f"to page limit; falling back to full batch. Error: {trim_err}"
                                )
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

            # Track processed page numbers (for QA validation / recovery scans).
            processed_pages: set[int] = set()
            for b in split_result.batches:
                processed_pages.update(range(b.start_page, b.end_page + 1))
            self._processed_pages = processed_pages if processed_pages else None

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
                # MEMORY FIX: Release MPS/CUDA tensor caches in addition to Python GC.
                # gc.collect() alone does NOT free Apple MPS memory held by PyTorch.
                self._release_torch_runtime_memory()
                self._log_memory_checkpoint(f"after batch {batch_info.batch_index + 1}/{split_result.batch_count}")
                logger.debug(f"gc.collect() + MPS cache clear after batch {batch_info.batch_index + 1}")

        # Batch extraction is complete; release heavy extraction runtimes before
        # validation/recovery stages so OCR recovery does not overlap Docling models.
        self._release_extraction_runtime_models("[MEMORY] post-batch extraction release")

        # ====================================================================
        # REQ-COORD-02: Extract page dimensions for UI overlay support
        # ====================================================================
        print("\n📐 [REQ-COORD-02] Extracting page dimensions...", flush=True)
        self._page_dimensions = self._extract_page_dimensions(pdf_path)

        # ====================================================================
        # REQ-DEDUP-01: Initialize ImageHashRegistry for pHash deduplication
        # ====================================================================
        if profile_type == "technical_manual":
            self._image_hash_registry = None
            print(
                "🔍 [PHASH] Disabled for technical_manual profile (stability/performance).",
                flush=True,
            )
        else:
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

        # Final hygiene pass for technical manuals AFTER recovery (recovery may re-introduce
        # TOC artifacts like control chars / embedded page-number lines).
        if profile_type == "technical_manual":
            # IMPORTANT: unload heavy extraction runtimes before final hygiene.
            # This avoids Docling+EasyOCR overlap in late-stage processing where
            # macOS can issue hard Killed:9 despite low apparent Python RSS.
            self._release_extraction_runtime_models("[TECHMANUAL-FINAL] preflight")
            # Release vision-side runtime state (OCR hint engines + cache payloads)
            # before final hygiene to reduce end-of-run memory pressure.
            prefinal_vision_stats = self._release_vision_runtime_models(
                "[TECHMANUAL-FINAL] preflight vision release"
            )
            self._log_memory_checkpoint("[TECHMANUAL-FINAL] preflight")
            all_chunks = self._sanitize_technical_manual_final(all_chunks)
        all_chunks = self._apply_oversize_breaker(all_chunks, max_chars=1500)
        all_chunks = self._apply_table_recovery_highlander_dedup(all_chunks)

        # Write aggregated output to master JSONL with deduplication
        output_jsonl = self.output_dir / "ingestion.jsonl"
        written_chunks = 0
        duplicate_count = 0
        export_error_count = 0

        # PHANTOM BUG FIX: Add defensive logging and error handling
        export_chunks = all_chunks  # Use the latest set (includes recovered chunks if any)
        logger.info(f"[FINALIZE] Starting JSONL write: {len(export_chunks)} chunks to process")
        print(
            f"\n📝 [FINALIZE] Writing {len(export_chunks)} chunks to {output_jsonl.name}...",
            flush=True,
        )

        # ✅ IRON-08: Clear file first, then stream write in small batches.
        if output_jsonl.exists():
            output_jsonl.unlink()

        with open(output_jsonl, "a", encoding="utf-8") as f:
            write_buffer: List[str] = []

            # Process chunks with streaming writes
            for idx, chunk in enumerate(export_chunks):
                try:
                    # Log progress every 50 chunks
                    if idx % 50 == 0 and idx > 0:
                        logger.debug(f"[FINALIZE] Processed {idx}/{len(export_chunks)} chunks")

                    chunk_dict = self._sanitize_chunk_for_export(chunk)

                    # Safety net: final JSONL-level hygiene for technical manuals.
                    # This catches rare cases where recovery/splitting re-introduces digit-only lines
                    # after earlier passes (page headers, running footers).
                    try:
                        if (
                            chunk_dict.get("modality") == "text"
                            and chunk_dict.get("metadata", {}).get("profile_type")
                            == "technical_manual"
                            and chunk_dict.get("metadata", {}).get("chunk_type") != ChunkType.CODE
                        ):
                            c = chunk_dict.get("content") or ""
                            c2 = self._strip_control_chars(c)
                            c2 = self._remove_standalone_page_number_lines(c2)
                            c2 = self._remove_all_digit_only_lines(c2)
                            c2 = self._fix_linebreak_hyphenation(c2)
                            if c2 != c:
                                chunk_dict["content"] = c2
                    except Exception:
                        pass

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
                                    f"[ASSET-METADATA-MISMATCH] "
                                    f"Asset '{filename}' page {filename_page} "
                                    f"!= metadata page {metadata_page}. Skipping chunk."
                                )
                                logger.error(error_msg)
                                export_error_count += 1
                                errors.append(error_msg)
                                continue

                    # ============================================================
                    # REQ-DEDUP-01: pHash Deduplication for IMAGE chunks
                    # ============================================================
                    if chunk.modality == Modality.IMAGE and asset_ref and self._image_hash_registry:
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

                    json_line = json.dumps(chunk_dict, ensure_ascii=False)
                    write_buffer.append(json_line)
                    written_chunks += 1

                    if len(write_buffer) >= DEFAULT_EXPORT_WRITE_BATCH_SIZE:
                        f.write("\n".join(write_buffer) + "\n")
                        f.flush()
                        write_buffer.clear()

                except Exception as e:
                    import traceback

                    export_error_count += 1
                    logger.error(
                        f"[FINALIZE-ERROR] Error processing chunk {idx}: {e}\n"
                        f"Chunk ID: {chunk.chunk_id if chunk else 'None'}\n"
                        f"Traceback:\n{traceback.format_exc()}"
                    )
                    errors.append(f"Finalize chunk {idx} failed: {e}")
                    continue

            if write_buffer:
                f.write("\n".join(write_buffer) + "\n")
                f.flush()

        # Log deduplication results
        if self._image_hash_registry:
            registry_stats = self._image_hash_registry.get_stats()
            print(
                f"\n📊 [PHASH] Deduplication complete: "
                f"{registry_stats['total_registered']} unique images, "
                f"{duplicate_count} duplicates rejected",
                flush=True,
            )
        else:
            print(
                "\n📊 [PHASH] Deduplication disabled for this profile.",
                flush=True,
            )
        print(
            f"\n📊 [EXPORT] Written {written_chunks} chunks "
            f"({duplicate_count} duplicates rejected, "
            f"{filtered_count} filtered, {export_error_count} export errors, "
            f"final attempted {len(export_chunks)})",
            flush=True,
        )
        logger.info(
            f"Written {written_chunks} chunks to {output_jsonl} "
            f"({duplicate_count} duplicates rejected, "
            f"{filtered_count} filtered, {export_error_count} export errors, "
            f"final attempted {len(export_chunks)})"
        )

        # Get vision stats and flush cache
        # PHANTOM BUG FIX: Add try-except to catch IndexError during cache operations
        vision_stats = dict(prefinal_vision_stats) if prefinal_vision_stats else {}
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
        print(f"   Total chunks written: {written_chunks}", flush=True)
        print(f"   Time: {elapsed:.1f}s", flush=True)
        print(f"   Output: {output_jsonl}", flush=True)

        return BatchProcessingResult(
            success=len(errors) == 0,
            original_path=pdf_path,
            original_hash=self._doc_hash,
            total_pages=split_result.total_pages,
            batches_processed=batches_processed,
            total_chunks=written_chunks,
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

        # ================================================================
        # IRON RULE 3: NEVER skip CODE chunks (programming books/manuals)
        # Code snippets can be short but still high-signal for RAG.
        # ================================================================
        try:
            if (
                chunk.metadata
                and (
                    chunk.metadata.content_classification == "code"
                    or chunk.metadata.chunk_type == ChunkType.CODE
                )
            ):
                return (False, None)
        except Exception:
            # Be conservative: if metadata is unexpected, fall through to standard rules.
            pass

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

    # ========================================================================
    # TECHNICAL MANUAL TEXT HYGIENE (Post-pass)
    # ========================================================================

    def _strip_control_chars(self, text: str) -> str:
        """
        Remove C0 control characters (except \\n and \\t) that commonly appear in TOC/Index
        pages (e.g., 0x08 backspace artifacts).
        """
        if not text:
            return text
        out_chars: List[str] = []
        for ch in text:
            o = ord(ch)
            if ch in ("\n", "\t"):
                out_chars.append(ch)
                continue
            # Drop C0 controls and DEL.
            if o < 32 or o == 127:
                continue
            out_chars.append(ch)
        return "".join(out_chars)

    def _remove_standalone_page_number_lines(self, text: str) -> str:
        """
        Remove standalone page number lines that get embedded in extracted text.

        Example:
            "Discussing naive RAG issues\\n171\\nLet's discuss..." -> removes the "171" line.
        """
        if not text:
            return text
        import re

        lines = text.splitlines()
        if len(lines) < 2:
            return text

        # Count digit-only lines. If we see multiple digit-only lines inside a text
        # chunk, it is almost always a TOC/Index page-number artifact.
        digit_only = [i for i, ln in enumerate(lines) if re.fullmatch(r"\s*\d{1,4}\s*", ln)]

        cleaned: List[str] = []
        for i, ln in enumerate(lines):
            s = ln.strip()
            if re.fullmatch(r"\d{1,4}", s):
                # Aggressive mode: multiple digit-only lines in the same chunk.
                if len(digit_only) >= 2:
                    continue

                # Conservative mode: only drop when adjacent to "real" text.
                prev = lines[i - 1].strip() if i > 0 else ""
                nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
                prev_is_text = len(prev) >= 6 and not re.fullmatch(r"\d{1,4}", prev)
                nxt_is_text = len(nxt) >= 6 and not re.fullmatch(r"\d{1,4}", nxt)
                if prev_is_text or nxt_is_text:
                    continue
            cleaned.append(ln)

        return "\n".join(cleaned).strip("\n")

    def _fix_linebreak_hyphenation(self, text: str) -> str:
        """Fix hyphenation across line breaks: 'multi-\\nstep' -> 'multi-step'."""
        if not text:
            return text
        import re

        return re.sub(r"([A-Za-z0-9])-\s*\n\s*([a-z])", r"\1-\2", text)

    def _remove_infix_list_numbering(self, text: str) -> str:
        """
        Remove list markers that were injected mid-sentence by OCR/layout ordering.

        Example:
            "... this set from 2. Brownells ..." -> "... this set from Brownells ..."

        IRON-09 COMPLIANCE: Pure regex pattern matching - NO hardcoded word lists.
        Only removes infix numbers that appear between lowercase words.
        Guardrail: do NOT touch valid section/prose continuations like
        "chapter 3. Note" (capitalized continuation).
        """
        if not text:
            return text
        import re

        # Pattern: lowercase_word + number marker + lowercase continuation.
        # Using lowercase continuation avoids false positives on valid prose
        # such as "chapter 3. Note".
        pattern = re.compile(
            r"(?P<prev>\b[a-z][a-z'\-]{0,15})\s+"  # lowercase word (1-15 chars to avoid matching "section")
            r"(?P<num>(?:[1-9]|[12]\d|3\d|40))\.\s+"  # number 1-40 followed by period
            r"(?P<next>[a-z][A-Za-z'\-]*)"  # lowercase continuation only
        )

        def repl(match: "re.Match[str]") -> str:
            prev = match.group("prev")
            nxt = match.group("next")
            # Join without the number - this is a mid-sentence list artifact
            return f"{prev} {nxt}"

        return pattern.sub(repl, text)

    def _remove_all_digit_only_lines(self, text: str) -> str:
        """Remove any line that is just a 1-4 digit number (technical_manual hygiene)."""
        if not text:
            return text
        import re

        return re.sub(r"(?m)^\s*\d{1,4}\s*$\n?", "", text).strip("\n")

    def _reflow_flat_code(self, text: str) -> str:
        """
        Best-effort reflow for code chunks that have lost newlines (common in PDF text extraction).

        We cannot perfectly reconstruct formatting without layout info, but inserting newlines
        around strong syntactic markers makes retrieval and copy/paste far more usable.
        """
        if not text:
            return text
        if "\n" in text:
            return text
        import re

        t = text
        # Reflow flattened REPL transcripts.
        t = re.sub(r"\s+(>>>|\.\.\.)\s+", r"\n\1 ", t)

        # If this is a Python def/class signature that's been flattened, split once after the header.
        if re.match(r"^\s*(async def|def|class)\b", t):
            t = re.sub(r"\)\s*:\s*", "):\n", t, count=1)

        # Newline before starters when preceded by non-newline whitespace.
        t = re.sub(
            r"(?<!\n)\s+(async def|def|class|import|from|return|yield|try|except|finally|with|if|elif|else|for|while)\b",
            r"\n\1",
            t,
        )
        # Split before assignments (helps with flattened Python/JS pseudo-code).
        # This may also split keyword arguments (dim=-1), which is acceptable for retrieval.
        t = re.sub(r"(?<!\n)\s+([A-Za-z_][A-Za-z0-9_]*\s*=)", r"\n\1", t)
        # Split after ':' when followed by an identifier (common in if/for/while headers).
        t = re.sub(r":\s+([A-Za-z_])", r":\n\1", t)
        # Newline after semicolons and braces.
        t = re.sub(r";\s*", ";\n", t)
        t = re.sub(r"\{\s*", "{\n", t)
        t = re.sub(r"\}\s*", "}\n", t)
        # Collapse excessive spaces but keep indentation minimal (no indentation info available).
        t = re.sub(r"[ \t]{2,}", " ", t)
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        if not lines:
            return ""

        # Heuristic indentation reconstruction for flattened Python-like blocks.
        # Only applies to non-REPL chunks with no existing indentation.
        has_repl = any(ln.startswith((">>>", "...")) for ln in lines)
        has_indent = any(ln.startswith(("    ", "\t")) for ln in lines)
        if not has_repl and not has_indent:
            opener = re.compile(
                r"^(async\s+def|def|class|if|elif|else|for|while|try|except|finally|with)\b.*:\s*$"
            )
            dedent_before = re.compile(r"^(elif|else|except|finally)\b")
            dedent_after = re.compile(r"^(return|yield|raise|break|continue|pass)\b")

            rebuilt: List[str] = []
            indent_level = 0
            for ln in lines:
                if dedent_before.match(ln):
                    indent_level = max(0, indent_level - 1)

                if indent_level > 0:
                    rebuilt.append(("    " * min(indent_level, 3)) + ln)
                else:
                    rebuilt.append(ln)

                if opener.match(ln):
                    indent_level += 1
                elif dedent_after.match(ln):
                    indent_level = max(0, indent_level - 1)
            lines = rebuilt

        # If we still have flat top-level code (imports/assignments/calls), emit as REPL lines.
        has_repl = any(ln.startswith((">>>", "...")) for ln in lines)
        has_indent = any(ln.startswith(("    ", "\t")) for ln in lines)
        if not has_repl and not has_indent and lines:
            stmt_like = re.compile(
                r"^(%pip\s+|!?[A-Za-z_][\w\.]*\s*=|from\s+[A-Za-z_][\w\.]*\s+import\b|import\s+[A-Za-z_][\w\.,\s]*$|[A-Za-z_][\w\.]*\s*\()"
            )
            codey_lines = sum(
                1
                for ln in lines
                if stmt_like.search(ln) is not None or re.search(r"[()\[\]{}=]", ln) is not None
            )
            if codey_lines >= max(1, int(len(lines) * 0.6)):
                lines = [f">>> {ln}" for ln in lines]

        return "\n".join(lines).strip("\n")

    def _preserve_or_reflow_code_text(self, text: str) -> str:
        """
        Preserve multiline code exactly; best-effort reflow only for flattened one-line code.
        """
        import re

        t = (text or "").strip("\n")
        if not t:
            return t
        if "\n" in t:
            lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
            if not lines:
                return t
            has_repl = any(ln.startswith((">>>", "...")) for ln in lines)
            has_indent = any(ln.startswith(("    ", "\t")) for ln in lines)
            if has_repl or has_indent:
                return t
            stmt_like = re.compile(
                r"^(%pip\s+|!?[A-Za-z_][\w\.]*\s*=|from\s+[A-Za-z_][\w\.]*\s+import\b|import\s+[A-Za-z_][\w\.,\s]*|[A-Za-z_][\w\.]*\s*\()"
            )
            codey_lines = sum(
                1
                for ln in lines
                if stmt_like.search(ln) is not None or re.search(r"[()\[\]{}=]", ln) is not None
            )
            if codey_lines >= max(1, int(len(lines) * 0.6)):
                return "\n".join(f">>> {ln}" for ln in lines)
            return t
        if self._looks_like_code_text(t):
            return self._reflow_flat_code(t)
        return t

    def _is_toc_or_index_text(self, text: str) -> bool:
        """
        Heuristic detector for TOC/Index style text blocks.

        We use this in technical_manual hygiene and recovery suppression to avoid
        polluting the main corpus with backmatter/frontmatter noise.
        """
        if not text:
            return False
        import re

        t = self._strip_control_chars(text)
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        if len(lines) < 6:
            return False

        head = "\n".join(lines[:3]).lower()
        if "table of contents" in head or head.startswith("contents") or head.startswith("index"):
            return True

        leader = re.compile(r"\.{2,}\s*\d{1,4}\s*$")
        ends_num = re.compile(r".{6,}\s\d{1,4}\s*$")
        digit_only = re.compile(r"^\d{1,4}$")
        index_refs = re.compile(r"\b\d{1,4}(,\s*\d{1,4}){1,}\b")

        leader_n = sum(1 for ln in lines if leader.search(ln))
        ends_n = sum(1 for ln in lines if ends_num.search(ln))
        digit_n = sum(1 for ln in lines if digit_only.fullmatch(ln))
        idxref_n = sum(1 for ln in lines if index_refs.search(ln))

        # Score & ratio gate.
        signal = leader_n * 2 + ends_n + idxref_n * 2 + digit_n
        ratio = (leader_n + ends_n + idxref_n + digit_n) / max(len(lines), 1)
        return (signal >= 8 and ratio >= 0.35) or (leader_n >= 3 and ratio >= 0.25)

    def _demote_toc_index_chunk(self, ch: IngestionChunk) -> None:
        """Demote TOC/Index chunks to reduce retrieval noise (do not delete by default)."""
        try:
            if ch.metadata.chunk_type not in (ChunkType.HEADING, ChunkType.LIST_ITEM):
                ch.metadata.chunk_type = ChunkType.LIST_ITEM
            ch.metadata.search_priority = "low"
        except Exception:
            pass

    def _maybe_demote_false_code_chunk(self, ch: IngestionChunk) -> None:
        """
        Docling sometimes classifies monospaced callouts as CODE even when they are prose
        (e.g., "after the >>> line ..."). Demote obvious prose back to PARAGRAPH.
        """
        try:
            if ch.modality != Modality.TEXT:
                return
            is_marked_code = (
                ch.metadata.chunk_type == ChunkType.CODE
                or ch.metadata.content_classification == "code"
            )
            if not is_marked_code:
                return
            txt = (ch.content or "").strip()
            if not txt:
                return
            import re

            def _demote() -> None:
                ch.metadata.chunk_type = ChunkType.PARAGRAPH
                if ch.metadata.content_classification == "code":
                    ch.metadata.content_classification = None

            lines = [ln.rstrip() for ln in txt.splitlines() if ln.strip()]
            if not lines:
                return
            has_repl = any(re.search(r"^\s*(>>>|\.\.\.)\s", ln) for ln in lines)
            has_indent = any(ln.startswith(("    ", "\t")) for ln in lines)

            # Keep true REPL/code signals.
            if has_repl:
                return

            # Scanned-degraded OCR often emits tiny orphan code fragments
            # (e.g., one-line def/class/import) without body indentation.
            # Demote these to prose so quality gates measure meaningful code blocks.
            if self._is_scanned_degraded_profile() and len(lines) <= 2 and not has_indent:
                _demote()
                return

            # "import X : explanation..." is typically prose explaining imports, not runnable code.
            explanatory_import = re.compile(r"^\s*(import|from)\b.+\s:\s+[A-Z]")
            explanatory_import_lines = sum(1 for ln in lines if explanatory_import.search(ln))

            def is_code_line(ln: str) -> bool:
                if explanatory_import.search(ln):
                    return False
                if re.search(r"^\s*(def|class|return|yield|async\s+def|await|if|elif|else|for|while|try|except|with)\b", ln):
                    return True
                if re.search(r"^\s*(from\s+[A-Za-z_][\w\.]*\s+import|import\s+[A-Za-z_][\w\.]*)\b", ln):
                    return True
                if ln.startswith(("    ", "\t")):
                    return True
                if re.search(r"[{}[\]();=]{2,}", ln):
                    return True
                return False

            code_lines = sum(1 for ln in lines if is_code_line(ln))
            prose_lines = sum(
                1
                for ln in lines
                if (
                    len(re.findall(r"[A-Za-z]{2,}", ln)) >= 6
                    and re.search(r"[:.;!?]$", ln.strip()) is not None
                )
                or (
                    not is_code_line(ln)
                    and len(re.findall(r"[A-Za-z]{2,}", ln)) >= 8
                    and ln[:1].isupper()
                )
            )

            if code_lines == 0:
                _demote()
                return
            # Single-line narrative sentences that happen to contain "import"/"from".
            if len(lines) == 1 and code_lines <= 1:
                line = lines[0].strip()
                # Standalone import lines create noisy one-line code chunks in prose-heavy pages.
                if re.fullmatch(
                    r"(?:from\s+[A-Za-z_][\w\.]*\s+import\s+[A-Za-z_][\w\.,\s]*|import\s+[A-Za-z_][\w\.]*(?:\s*,\s*[A-Za-z_][\w\.]*)*(?:\s+as\s+[A-Za-z_][\w]*)?)",
                    line,
                ):
                    _demote()
                    return
                if (
                    len(re.findall(r"[A-Za-z]{2,}", line)) >= 10
                    and line[:1].isupper()
                    and any(p in line for p in (".", ";", "!", "?"))
                    and not re.search(r"[{}()\[\]=]", line)
                ):
                    _demote()
                    return
            if explanatory_import_lines >= max(1, len(lines) // 2) and code_lines <= 1:
                _demote()
                return
            if code_lines <= 1 and prose_lines >= 2:
                _demote()
                return
            if code_lines <= 1 and prose_lines >= 1 and len(lines) <= 3:
                _demote()
                return

            lower = txt.lower()
            # Prose-y signals: long sentences, common verbs, few code symbols.
            word_count = len(re.findall(r"[A-Za-z]{2,}", txt))
            prose_markers = ("essentially", "allows you", "consists", "for example", "just after", "in this")
            if word_count >= 25 and any(m in lower for m in prose_markers):
                if not re.search(r"[{};]", txt) and txt.count("=") <= 1:
                    _demote()
        except Exception:
            return

    def _is_manual_label_text(self, text: str) -> bool:
        """
        Detect short field/header labels often used in technical manuals.

        Examples:
        - "Reassembly Tips:"
        - "Origin: United States" (still short label-like line)
        - "A Note on Reassembly"
        """
        import re

        s = (text or "").strip()
        if not s:
            return False
        if len(s) > 60:
            return False
        if "\n" in s:
            return False
        if re.fullmatch(r"\d{1,4}", s):
            return False
        if not re.fullmatch(r"[A-Z][A-Za-z0-9/&()' .,-]{1,59}:?", s):
            return False

        # Treat field-value lines as complete records, not attachable labels.
        # Example: "Origin: United States" should remain standalone.
        if ":" in s and not s.endswith(":"):
            return False

        if ":" not in s:
            words = [w for w in re.split(r"\s+", s) if w]
            if len(words) > 6:
                return False
            if any(w.endswith((".", "?", "!")) for w in words):
                return False
        return True

    def _apply_spatial_refiner(self, chunks: List[IngestionChunk]) -> List[IngestionChunk]:
        """Unified Spatial Refiner - no DPI, no heading branches, geometry only."""
        MAX_MERGED_CHARS = 8000

        def bbox(ch: IngestionChunk) -> List[int]:
            if ch.metadata and ch.metadata.spatial and ch.metadata.spatial.bbox:
                return ch.metadata.spatial.bbox
            return [0, 0, 0, 0]

        def is_code_chunk(ch: IngestionChunk) -> bool:
            try:
                return (
                    ch.metadata.chunk_type == ChunkType.CODE
                    or ch.metadata.content_classification == "code"
                )
            except Exception:
                return False

        text_chunks = [c for c in chunks if c.modality == Modality.TEXT]
        non_text_chunks = [c for c in chunks if c.modality != Modality.TEXT]
        if not text_chunks:
            return chunks

        ordered = sorted(
            text_chunks,
            key=lambda c: (
                c.metadata.page_number if c.metadata else 0,
                bbox(c)[1],
                bbox(c)[0],
            ),
        )
        refined: List[IngestionChunk] = []
        current = ordered[0]

        for nxt in ordered[1:]:
            cur_page = current.metadata.page_number if current.metadata else -1
            nxt_page = nxt.metadata.page_number if nxt.metadata else -2
            if cur_page != nxt_page:
                refined.append(current)
                current = nxt
                continue

            # Never merge code and prose chunks; this destroys code fidelity.
            if is_code_chunk(current) != is_code_chunk(nxt):
                refined.append(current)
                current = nxt
                continue

            box_a = bbox(current)
            box_b = bbox(nxt)
            v_gap = box_b[1] - box_a[3]
            h_overlap = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
            min_width = max(1, min(box_a[2] - box_a[0], box_b[2] - box_b[0]))

            if 0 <= v_gap <= 20 and (h_overlap / float(min_width)) > 0.4:
                cur_text = (current.content or "").rstrip()
                nxt_text = (nxt.content or "").lstrip()
                projected_chars = len(cur_text) + len(nxt_text) + 1
                if projected_chars > MAX_MERGED_CHARS:
                    refined.append(current)
                    current = nxt
                    continue

                current.content = f"{cur_text}\n{nxt_text}".strip()
                if current.metadata and current.metadata.spatial:
                    current.metadata.spatial.bbox = [
                        min(box_a[0], box_b[0]),
                        min(box_a[1], box_b[1]),
                        max(box_a[2], box_b[2]),
                        max(box_a[3], box_b[3]),
                    ]
            else:
                refined.append(current)
                current = nxt

        refined.append(current)
        all_chunks = refined + non_text_chunks
        all_chunks.sort(
            key=lambda c: (
                c.metadata.page_number if c.metadata else 0,
                bbox(c)[1],
                bbox(c)[0],
            )
        )
        return all_chunks

    def _apply_vertical_proximity_merger(self, chunks: List[IngestionChunk]) -> List[IngestionChunk]:
        return self._apply_spatial_refiner(chunks)

    def _apply_vertical_proximity_merger_pagewise(
        self, chunks: List[IngestionChunk], gc_every_pages: int = 20
    ) -> List[IngestionChunk]:
        """
        Run vertical-proximity merging page-by-page to bound peak memory.

        This keeps the same 20-unit merge contract while avoiding large
        all-document merge passes on long manuals.
        """
        page_buckets: Dict[int, List[IngestionChunk]] = {}
        passthrough: List[IngestionChunk] = []

        for ch in chunks:
            if ch.modality != Modality.TEXT:
                passthrough.append(ch)
                continue
            page_no = int(ch.metadata.page_number or 0) if ch.metadata else 0
            page_buckets.setdefault(page_no, []).append(ch)

        merged_text: List[IngestionChunk] = []
        for idx, page_no in enumerate(sorted(page_buckets.keys())):
            page_chunks = page_buckets[page_no]
            if len(page_chunks) <= 1:
                merged_text.extend(page_chunks)
            else:
                merged_text.extend(self._apply_vertical_proximity_merger(page_chunks))

            if gc_every_pages > 0 and (idx + 1) % gc_every_pages == 0:
                gc.collect()

        all_chunks = merged_text + passthrough

        def _sort_key(ch: IngestionChunk) -> Tuple[int, int, int]:
            page_no = int(ch.metadata.page_number or 0) if ch.metadata else 0
            bbox = None
            if ch.metadata and ch.metadata.spatial and ch.metadata.spatial.bbox:
                bbox = ch.metadata.spatial.bbox
            x0 = int(bbox[0]) if bbox and len(bbox) >= 4 else 0
            y0 = int(bbox[1]) if bbox and len(bbox) >= 4 else 0
            return (page_no, y0, x0)

        all_chunks.sort(key=_sort_key)
        return all_chunks

    def _merge_micro_text_chunks(
        self, chunks: List[IngestionChunk], max_chars: int = 30
    ) -> List[IngestionChunk]:
        """
        Attach tiny non-label text fragments to neighboring text chunks.

        This reduces standalone micro-chunk noise that hurts retrieval quality.
        """

        def is_code_chunk(ch: IngestionChunk) -> bool:
            try:
                return (
                    ch.metadata.chunk_type == ChunkType.CODE
                    or ch.metadata.content_classification == "code"
                )
            except Exception:
                return False

        def page_of(ch: IngestionChunk) -> int:
            return int(ch.metadata.page_number or 0) if ch.metadata else 0

        def sort_key(ch: IngestionChunk) -> Tuple[int, int, int]:
            page_no = page_of(ch)
            bbox = None
            if ch.metadata and ch.metadata.spatial and ch.metadata.spatial.bbox:
                bbox = ch.metadata.spatial.bbox
            x0 = int(bbox[0]) if bbox and len(bbox) >= 4 else 0
            y0 = int(bbox[1]) if bbox and len(bbox) >= 4 else 0
            return (page_no, y0, x0)

        ordered = sorted(chunks, key=sort_key)
        page_has_body: Dict[int, bool] = {}
        for ch in ordered:
            if ch.modality != Modality.TEXT or is_code_chunk(ch):
                continue
            txt = (ch.content or "").strip()
            if len(txt) >= 20 and not self._is_manual_label_text(txt):
                page_has_body[page_of(ch)] = True

        out: List[IngestionChunk] = []
        i = 0

        while i < len(ordered):
            cur = ordered[i]
            if (
                cur.modality != Modality.TEXT
                or is_code_chunk(cur)
                or not (cur.content or "").strip()
            ):
                out.append(cur)
                i += 1
                continue

            cur_text = (cur.content or "").strip()
            cur_page = page_of(cur)
            cur_is_label = self._is_manual_label_text(cur_text)

            # Rule 1: Glue heading/label text onto a following code block.
            if cur_is_label and i + 1 < len(ordered):
                nxt = ordered[i + 1]
                if (
                    nxt.modality == Modality.TEXT
                    and is_code_chunk(nxt)
                    and page_of(nxt) == cur_page
                    and (nxt.content or "").strip()
                ):
                    nxt.content = f"{cur_text}\n{(nxt.content or '').lstrip()}".strip()
                    i += 1
                    continue

            # Rule 1b: Merge dense short list-item runs (TOC-style lines) to prevent
            # over-fragmentation and false orphan labels.
            try:
                cur_is_list_item = cur.metadata.chunk_type == ChunkType.LIST_ITEM
            except Exception:
                cur_is_list_item = False
            if cur_is_list_item and len(cur_text) <= 90:
                run_end = i + 1
                run_parts = [cur_text]
                while run_end < len(ordered):
                    cand = ordered[run_end]
                    if cand.modality != Modality.TEXT or is_code_chunk(cand):
                        break
                    if page_of(cand) != cur_page:
                        break
                    try:
                        cand_is_list_item = cand.metadata.chunk_type == ChunkType.LIST_ITEM
                    except Exception:
                        cand_is_list_item = False
                    cand_text = (cand.content or "").strip()
                    if not cand_is_list_item or not cand_text or len(cand_text) > 90:
                        break
                    run_parts.append(cand_text)
                    run_end += 1
                    if len(run_parts) >= 8:
                        break

                if len(run_parts) >= 3:
                    cur.content = "\n".join(run_parts).strip()
                    out.append(cur)
                    i = run_end
                    continue

            # Rule 1c: If a short label follows body text on the same page, absorb it
            # into that body chunk so it doesn't become an orphan.
            if cur_is_label and len(cur_text) <= 60 and out:
                prev = out[-1]
                if (
                    prev.modality == Modality.TEXT
                    and not is_code_chunk(prev)
                    and page_of(prev) == cur_page
                    and (prev.content or "").strip()
                ):
                    prev.content = f"{(prev.content or '').rstrip()}\n{cur_text}".strip()
                    i += 1
                    continue

            # Rule 1d: Drop standalone label-only pages with no body content.
            # This removes repeated running headers/blank-page captions.
            if cur_is_label and not page_has_body.get(cur_page, False):
                i += 1
                continue

            is_micro = len(cur_text) < max_chars and not self._is_manual_label_text(cur_text)
            if not is_micro:
                out.append(cur)
                i += 1
                continue

            attached = False

            # Prefer attaching to the following text chunk on the same page.
            if i + 1 < len(ordered):
                nxt = ordered[i + 1]
                if (
                    nxt.modality == Modality.TEXT
                    and not is_code_chunk(nxt)
                    and page_of(nxt) == cur_page
                    and (nxt.content or "").strip()
                ):
                    nxt.content = f"{cur_text} {(nxt.content or '').lstrip()}".strip()
                    attached = True

            if attached:
                i += 1
                continue

            # Otherwise append to the previous text chunk on the same page.
            if out:
                prev = out[-1]
                if (
                    prev.modality == Modality.TEXT
                    and not is_code_chunk(prev)
                    and page_of(prev) == cur_page
                    and (prev.content or "").strip()
                ):
                    prev.content = f"{(prev.content or '').rstrip()} {cur_text}".strip()
                    i += 1
                    continue

            out.append(cur)
            i += 1

        return out

    def _apply_technical_manual_hygiene(self, chunks: List[IngestionChunk]) -> List[IngestionChunk]:
        """
        Technical-manual-specific cleanup to improve RAG quality:
        - Strip control chars
        - Remove embedded page-number lines
        - Fix hyphenation across line breaks
        - Best-effort reflow of flattened code chunks
        - Join obviously broken chunk boundaries (mid-word / mid-sentence)
        - Apply vertical proximity merger (UIR layout-aware)
        """
        import re

        def is_code_chunk(ch: IngestionChunk) -> bool:
            try:
                return (
                    ch.metadata.chunk_type == ChunkType.CODE
                    or ch.metadata.content_classification == "code"
                )
            except Exception:
                return False

        # Step 0: Apply vertical proximity merger (UIR layout-aware merging)
        chunks = self._apply_vertical_proximity_merger(chunks)

        # Step A: per-chunk sanitation
        infix_fixed_chunks = 0
        for ch in chunks:
            if ch.modality != Modality.TEXT:
                continue
            txt = ch.content or ""

            # Reclassify missed code blocks (Docling sometimes emits code as paragraph with flattened whitespace).
            try:
                if (not is_code_chunk(ch)) and self._looks_like_code_text(txt):
                    ch.metadata.chunk_type = ChunkType.CODE
                    ch.metadata.content_classification = "code"
            except Exception:
                pass

            # Also demote obvious false positives.
            self._maybe_demote_false_code_chunk(ch)

            if not is_code_chunk(ch):
                txt2 = self._strip_control_chars(txt)
                txt2 = self._remove_standalone_page_number_lines(txt2)
                # Ensure no digit-only lines survive (TOC/index and running headers).
                txt2 = self._remove_all_digit_only_lines(txt2)
                txt2 = self._fix_linebreak_hyphenation(txt2)
                txt3 = self._remove_infix_list_numbering(txt2)
                if txt3 != txt2:
                    infix_fixed_chunks += 1
                txt2 = txt3
            else:
                # Indentation shield: preserve code blocks as-is.
                txt2 = self._preserve_or_reflow_code_text(txt)

            if txt2 != txt:
                ch.content = txt2

            # Keep refined_content aligned with the same hygiene rules so downstream
            # consumers do not reintroduce refiner artifacts.
            try:
                rc = ch.metadata.refined_content
            except Exception:
                rc = None
            if isinstance(rc, str) and rc:
                if not is_code_chunk(ch):
                    rc2 = self._strip_control_chars(rc)
                    rc2 = self._remove_standalone_page_number_lines(rc2)
                    rc2 = self._remove_all_digit_only_lines(rc2)
                    rc2 = self._fix_linebreak_hyphenation(rc2)
                    rc2 = self._remove_infix_list_numbering(rc2)
                else:
                    # Indentation shield: preserve code blocks as-is.
                    rc2 = self._preserve_or_reflow_code_text(rc)
                if rc2 != rc:
                    ch.metadata.refined_content = rc2

            # Demote TOC/Index noise so it doesn't dominate retrieval.
            if self._is_toc_or_index_text(ch.content or ""):
                self._demote_toc_index_chunk(ch)

        if infix_fixed_chunks:
            logger.info(
                f"[TECHMANUAL-HYGIENE] Removed infix list numbering in {infix_fixed_chunks} chunks"
            )

        # Step B: join broken chunk boundaries (conservative)
        joined: List[IngestionChunk] = []
        i = 0

        end_punct = re.compile(r"[\\.!\\?\\:\\;\\\"\\'\\)\\]\\}]\\s*$")
        begins_lower = re.compile(r"^[a-z]")
        begins_word = re.compile(r"^[A-Za-z]")
        label_like = re.compile(r"^[A-Za-z][A-Za-z0-9\\-\\s]{0,50}$")

        while i < len(chunks):
            cur = chunks[i]
            if (
                cur.modality != Modality.TEXT
                or is_code_chunk(cur)
                or not cur.content
                or i == len(chunks) - 1
            ):
                joined.append(cur)
                i += 1
                continue

            nxt = chunks[i + 1]
            if (
                nxt.modality != Modality.TEXT
                or not nxt.content
                or cur.metadata.page_number != nxt.metadata.page_number
            ):
                joined.append(cur)
                i += 1
                continue

            cur_s = cur.content.rstrip()
            nxt_s = nxt.content.lstrip()
            cur_is_label = self._is_manual_label_text(cur_s)

            # Rule 1: heading/label followed by code should stay together.
            # Keep CODE classification by folding the heading into the next chunk.
            if cur_is_label and is_code_chunk(nxt):
                nxt.content = f"{cur_s}\n{nxt_s}".strip()
                i += 1
                continue

            if is_code_chunk(nxt):
                joined.append(cur)
                i += 1
                continue

            # Heuristic 0: glue short label/headings onto their following paragraph.
            # This reduces retrieval noise from standalone "Summary", "Further reading", etc.
            if (
                len(cur_s) <= 40
                and len(nxt_s) >= 80
                and (label_like.fullmatch(cur_s.strip()) is not None or cur_is_label)
                and not begins_lower.search(nxt_s)
            ):
                sep = ": " if not cur_s.endswith((".", ":", "?", "!", ";")) else " "
                cur.content = (cur_s + sep + nxt_s).strip()
                i += 2
                joined.append(cur)
                continue

            # Heuristic 0b: compact short heading/name runs that Docling often emits
            # as many tiny chunks in front matter and cookbook-like layouts.
            if cur_is_label and len(nxt_s) <= 120:
                sep = ": " if not cur_s.endswith((".", ":", "?", "!", ";")) else " "
                nxt.content = (cur_s + sep + nxt_s).strip()
                i += 1
                continue

            # Heuristic 1: mid-sentence split (current lacks terminal punctuation; next starts lowercase).
            should_join = (not end_punct.search(cur_s)) and bool(begins_lower.search(nxt_s))

            # Heuristic 2: mid-word split ("... thou" + "sand ...").
            if not should_join:
                last_token = cur_s.split()[-1] if cur_s.split() else ""
                first_token = nxt_s.split()[0] if nxt_s.split() else ""
                if (
                    last_token
                    and first_token
                    and len(last_token) <= 3
                    and begins_word.search(first_token or "") is not None
                    and cur_s and cur_s[-1].isalpha()
                    and first_token[0].isalpha()
                ):
                    should_join = True

            if not should_join:
                joined.append(cur)
                i += 1
                continue

            # Join.
            # If it looks like a mid-word join, don't add a space.
            join_with_space = True
            if cur_s and nxt_s and cur_s[-1].isalpha() and nxt_s[0].isalpha():
                # short last token -> more likely mid-word
                last_token = cur_s.split()[-1] if cur_s.split() else ""
                if len(last_token) <= 3:
                    join_with_space = False

            cur.content = (cur_s + (" " if join_with_space else "") + nxt_s).strip()
            # Prefer the next snippet from the chunk we are absorbing.
            try:
                if (
                    cur.semantic_context
                    and nxt.semantic_context
                    and nxt.semantic_context.next_text_snippet
                ):
                    cur.semantic_context.next_text_snippet = nxt.semantic_context.next_text_snippet
            except Exception:
                pass

            # Drop nxt.
            i += 2
            joined.append(cur)

        # Step C: run spatial merger again after textual boundary repair.
        return self._apply_vertical_proximity_merger(joined)

    def _sanitize_technical_manual_final(self, chunks: List[IngestionChunk]) -> List[IngestionChunk]:
        """
        Final technical-manual sanitation pass applied AFTER the recovery pipeline.

        The recovery pipeline can introduce raw text blocks containing control chars or
        embedded page numbers (especially from TOC/Index). We sanitize again here, but
        avoid any cross-chunk merging to keep recovery bookkeeping stable.
        
        MEMORY OPTIMIZATION: Processes chunks in batches to prevent OOM during final phase.
        """
        import re
        import gc

        # MEMORY FIX: Process in batches to avoid holding all processed chunks in memory
        # This prevents OOM when EasyOCR and other heavy operations are also active
        BATCH_SIZE = 50  # Process 50 chunks at a time
        
        logger.info(f"[TECHMANUAL-FINAL] Running final hygiene pass on {len(chunks)} chunks (batch size: {BATCH_SIZE})")
        
        # Force garbage collection before starting final hygiene
        gc.collect()
        self._log_memory_checkpoint("[TECHMANUAL-FINAL] start")

        page_num_fixed_chunks = 0
        infix_fixed_chunks = 0

        def is_code_chunk(ch: IngestionChunk) -> bool:
            try:
                return (
                    ch.metadata.chunk_type == ChunkType.CODE
                    or ch.metadata.content_classification == "code"
                )
            except Exception:
                return False

        # In-place compaction to avoid duplicating the full chunk list in memory.
        # `read_idx` scans original positions, `write_idx` stores kept items.
        read_total = len(chunks)
        write_idx = 0

        # Process chunks in batches to reduce peak memory pressure.
        total_batches = (read_total + BATCH_SIZE - 1) // BATCH_SIZE
        for batch_idx in range(total_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, read_total)

            logger.debug(
                f"[TECHMANUAL-FINAL] Processing batch {batch_idx + 1}/{total_batches} "
                f"(chunks {start_idx}-{end_idx})"
            )

            self._log_memory_checkpoint(
                f"[TECHMANUAL-FINAL] batch {batch_idx + 1}/{total_batches} start"
            )

            for read_idx in range(start_idx, end_idx):
                ch = chunks[read_idx]

                if ch.modality != Modality.TEXT:
                    chunks[write_idx] = ch
                    write_idx += 1
                    continue

                txt = ch.content or ""
                had_digit_line = bool(re.search(r"(?m)^\s*\d{1,4}\s*$", txt))
                if is_code_chunk(ch):
                    # Indentation shield: preserve code blocks as-is.
                    txt2 = self._preserve_or_reflow_code_text(txt)
                else:
                    txt2 = self._strip_control_chars(txt)
                    txt2 = self._remove_standalone_page_number_lines(txt2)
                    txt2 = self._remove_all_digit_only_lines(txt2)
                    txt2 = self._fix_linebreak_hyphenation(txt2)
                    txt3 = self._remove_infix_list_numbering(txt2)
                    if txt3 != txt2:
                        infix_fixed_chunks += 1
                    txt2 = txt3
                if txt2 != txt:
                    ch.content = txt2
                    if had_digit_line and not re.search(r"(?m)^\s*\d{1,4}\s*$", txt2):
                        page_num_fixed_chunks += 1

                # Apply the same final hygiene rules to refined_content, if present.
                try:
                    rc = ch.metadata.refined_content
                except Exception:
                    rc = None
                if isinstance(rc, str) and rc:
                    if is_code_chunk(ch):
                        # Indentation shield: preserve code blocks as-is.
                        rc2 = self._preserve_or_reflow_code_text(rc)
                    else:
                        rc2 = self._strip_control_chars(rc)
                        rc2 = self._remove_standalone_page_number_lines(rc2)
                        rc2 = self._remove_all_digit_only_lines(rc2)
                        rc2 = self._fix_linebreak_hyphenation(rc2)
                        rc2 = self._remove_infix_list_numbering(rc2)
                    if rc2 != rc:
                        ch.metadata.refined_content = rc2

                # Re-check code false positives (Docling can mark monospaced prose as CODE).
                self._maybe_demote_false_code_chunk(ch)

                # Demote TOC/Index chunks instead of dropping them entirely.
                # Keeping low-priority TOC text helps token parity and recall.
                if self._is_toc_or_index_text(ch.content or ""):
                    self._demote_toc_index_chunk(ch)

                chunks[write_idx] = ch
                write_idx += 1

            self._log_memory_checkpoint(
                f"[TECHMANUAL-FINAL] batch {batch_idx + 1}/{total_batches} end"
            )
            if batch_idx < total_batches - 1:
                gc.collect()
                self._log_memory_checkpoint(
                    f"[TECHMANUAL-FINAL] batch {batch_idx + 1}/{total_batches} post-gc"
                )

        # Trim dropped items after in-place compaction.
        if write_idx < len(chunks):
            del chunks[write_idx:]
        self._log_memory_checkpoint("[TECHMANUAL-FINAL] after compaction")

        if page_num_fixed_chunks:
            logger.info(
                f"[TECHMANUAL-FINAL] Removed digit-only lines in {page_num_fixed_chunks} chunks"
            )
        if infix_fixed_chunks:
            logger.info(
                f"[TECHMANUAL-FINAL] Removed infix list numbering in {infix_fixed_chunks} chunks"
            )
        # Final spatial consolidation after recovery/splitting.
        gc.collect()
        self._log_memory_checkpoint("[TECHMANUAL-FINAL] before final spatial merge")
        # Stability guard: run merge page-wise on large final sets to keep
        # memory bounded while preserving final label/paragraph consolidation.
        if len(chunks) > 250:
            logger.warning(
                f"[TECHMANUAL-FINAL] Using page-wise final spatial merge for large chunk set "
                f"({len(chunks)} chunks) to preserve runtime stability"
            )
            merged = self._apply_vertical_proximity_merger_pagewise(chunks)
            return self._merge_micro_text_chunks(merged, max_chars=30)

        merged = self._apply_vertical_proximity_merger(chunks)
        return self._merge_micro_text_chunks(merged, max_chars=30)

    def _split_nearest_paragraph_breaks(
        self,
        text: str,
        max_chars: int = 1500,
        overlap_chars: int = 120,
    ) -> List[str]:
        """
        OversizeBreaker split policy:
        - Prefer the nearest paragraph break (\\n\\n) around max_chars.
        - Fallback to nearest single newline, then hard split.
        """
        if not text or len(text) <= max_chars:
            return [text]

        # Code-aware shield: keep line boundaries intact for multiline code-ish text.
        # This avoids fragmenting indented blocks into retrieval-hostile snippets.
        if "\n" in text and self._looks_like_code_text(text):
            return self._split_preserve_line_boundaries(text=text, max_chars=max_chars)

        pieces: List[str] = []
        remaining = text.strip()
        max_iters = max(32, (len(remaining) // max(1, max_chars - overlap_chars)) * 4)
        iters = 0

        while remaining:
            iters += 1
            if iters > max_iters:
                logger.warning(
                    f"[OVERSIZE-BREAKER] Split loop guard triggered after {iters} iterations; "
                    "falling back to hard split for remaining text"
                )
                hard = remaining[:max_chars].strip()
                if hard:
                    pieces.append(hard)
                remaining = remaining[max_chars:].lstrip()
                continue

            if len(remaining) <= max_chars:
                pieces.append(remaining.strip())
                break

            target = max_chars
            # Bound forward searches to a local window around the split target.
            # Unbounded `.find(..., target)` on very large chunks creates
            # quadratic behavior in late-stage splitting.
            search_end = min(len(remaining), target + max_chars)
            p_before = remaining.rfind("\n\n", 0, target + 1)
            p_after = remaining.find("\n\n", target, search_end)

            split_idx: Optional[int] = None
            delimiter_len = 2

            candidates: List[Tuple[int, int]] = []
            if p_before > 0:
                candidates.append((abs(target - p_before), p_before))
            if p_after > 0:
                candidates.append((abs(p_after - target), p_after))

            if candidates:
                candidates.sort(key=lambda x: x[0])
                split_idx = candidates[0][1]
            else:
                # fallback: nearest single newline
                n_before = remaining.rfind("\n", 0, target + 1)
                n_after = remaining.find("\n", target, search_end)
                nl_candidates: List[Tuple[int, int]] = []
                if n_before > 0:
                    nl_candidates.append((abs(target - n_before), n_before))
                if n_after > 0:
                    nl_candidates.append((abs(n_after - target), n_after))
                if nl_candidates:
                    nl_candidates.sort(key=lambda x: x[0])
                    split_idx = nl_candidates[0][1]
                    delimiter_len = 1
                else:
                    # Sentence-aware hard-cap fallback:
                    # prefer the last sentence boundary before target before raw hard split.
                    sentence_marks = []
                    for marker in (". ", ".\n", "? ", "?\n", "! ", "!\n"):
                        pos = remaining.rfind(marker, 0, target + 1)
                        if pos > 0:
                            sentence_marks.append(pos + 1)  # include punctuation
                    for marker in (".", "?", "!"):
                        pos = remaining.rfind(marker, 0, target + 1)
                        if pos > 0:
                            sentence_marks.append(pos + 1)
                    split_idx = max(sentence_marks) if sentence_marks else target
                    if split_idx < (max_chars // 2):
                        split_idx = target
                    delimiter_len = 0

            if split_idx is None or split_idx <= 0:
                split_idx = target
                delimiter_len = 0

            # Guard: avoid tiny early splits that can create near-zero progress loops.
            if split_idx < (max_chars // 2):
                split_idx = target
                delimiter_len = 0

            # Hard cap: OversizeBreaker must never emit a head segment above max_chars.
            # A nearest paragraph/newline break after the target is useful for semantics,
            # but we cannot violate the configured chunk ceiling.
            if split_idx > max_chars:
                split_idx = max_chars
                delimiter_len = 0

            head = remaining[:split_idx].strip()
            if not head:
                head = remaining[:target].strip()
                split_idx = target
                delimiter_len = 0

            pieces.append(head)

            tail = remaining[max(0, len(head) - overlap_chars) : split_idx]
            if tail and "\n" in tail:
                tail = tail[tail.find("\n") + 1 :].lstrip()
            next_start = split_idx + delimiter_len
            candidate = (tail + ("\n\n" if tail else "") + remaining[next_start:].lstrip("\n")).strip()

            # Enforce monotonic progress in all cases.
            if len(candidate) >= len(remaining):
                candidate = remaining[next_start:].lstrip("\n").strip()
            if len(candidate) >= len(remaining):
                # Last-resort hard advancement to avoid infinite loops.
                hard_next = min(len(remaining), max_chars)
                candidate = remaining[hard_next:].lstrip("\n").strip()

            remaining = candidate

        return [p for p in pieces if p.strip()]

    def _split_preserve_line_boundaries(
        self,
        text: str,
        max_chars: int = 1500,
    ) -> List[str]:
        """
        Split multiline code-ish text by complete lines only.

        Never split in the middle of an indented/code line.
        """
        lines = text.splitlines()
        if not lines:
            return [text]

        parts: List[str] = []
        current: List[str] = []
        current_len = 0

        for line in lines:
            # Force split very long single lines with hard caps only.
            # Do not apply sentence tokenization to code paths.
            if len(line) > max_chars:
                if current:
                    parts.append("\n".join(current).strip())
                    current = []
                    current_len = 0

                rem = line
                while rem:
                    if len(rem) <= max_chars:
                        parts.append(rem.strip())
                        rem = ""
                        continue
                    split_idx = max_chars
                    parts.append(rem[:split_idx].rstrip())
                    rem = rem[split_idx:]
                continue

            add_len = len(line) + (1 if current else 0)
            if current and current_len + add_len > max_chars:
                parts.append("\n".join(current).strip())
                current = [line]
                current_len = len(line)
            else:
                current.append(line)
                current_len += add_len

        if current:
            parts.append("\n".join(current).strip())

        return [p for p in parts if p]

    def _apply_oversize_breaker(
        self,
        chunks: List[IngestionChunk],
        max_chars: int = 1500,
    ) -> List[IngestionChunk]:
        """
        Apply OversizeBreaker to TEXT chunks.

        - prose: split at nearest paragraph/newline/sentence boundaries.
        - code: split strictly on line boundaries.
        """
        split_count = 0
        out: List[IngestionChunk] = []

        for ch in chunks:
            if ch.modality != Modality.TEXT or not ch.content:
                out.append(ch)
                continue

            is_code = False
            try:
                is_code = (
                    ch.metadata.chunk_type == ChunkType.CODE
                    or ch.metadata.content_classification == "code"
                )
            except Exception:
                is_code = False

            if len(ch.content) <= max_chars:
                out.append(ch)
                continue

            if is_code:
                parts = self._split_preserve_line_boundaries(
                    text=ch.content,
                    max_chars=max_chars,
                )
            else:
                parts = self._split_nearest_paragraph_breaks(
                    text=ch.content,
                    max_chars=max_chars,
                    overlap_chars=120,
                )
            if len(parts) <= 1:
                out.append(ch)
                continue

            split_count += 1
            for idx, sub in enumerate(parts):
                if is_code:
                    sub_is_code = self._looks_like_code_text(sub)
                    sub_chunk_type = ChunkType.CODE if sub_is_code else ChunkType.PARAGRAPH
                    sub_content_classification = "code" if sub_is_code else None
                else:
                    sub_chunk_type = ch.metadata.chunk_type
                    sub_content_classification = getattr(ch.metadata, "content_classification", None)

                if idx == 0:
                    ch.content = sub
                    ch.metadata.chunk_type = sub_chunk_type
                    ch.metadata.content_classification = sub_content_classification
                    out.append(ch)
                    continue
                try:
                    new_h = HierarchyMetadata(
                        parent_heading=(
                            ch.metadata.hierarchy.parent_heading if ch.metadata.hierarchy else None
                        ),
                        breadcrumb_path=(
                            (ch.metadata.hierarchy.breadcrumb_path if ch.metadata.hierarchy else [])
                            + [f"[Oversize Split {idx+1}/{len(parts)}]"]
                        ),
                        level=(
                            (ch.metadata.hierarchy.level or 2) + 1
                            if ch.metadata and ch.metadata.hierarchy
                            else 3
                        ),
                    )
                    new_chunk = create_text_chunk(
                        doc_id=ch.doc_id,
                        content=sub,
                        source_file=ch.metadata.source_file,
                        file_type=ch.metadata.file_type,
                        page_number=ch.metadata.page_number,
                        hierarchy=new_h,
                        chunk_type=sub_chunk_type,
                        bbox=(ch.metadata.spatial.bbox if ch.metadata.spatial else None),
                        extraction_method=ch.metadata.extraction_method,
                        prev_text=(ch.semantic_context.prev_text_snippet if ch.semantic_context else None),
                        next_text=(ch.semantic_context.next_text_snippet if ch.semantic_context else None),
                        content_classification=sub_content_classification,
                        **{k: v for k, v in self._intelligence_metadata.items() if v is not None},
                    )
                    new_chunk.chunk_id = f"{ch.chunk_id}_o{idx+1}"
                    if ch.metadata.spatial:
                        if new_chunk.metadata.spatial is None:
                            new_chunk.metadata.spatial = SpatialMetadata(bbox=None)
                        new_chunk.metadata.spatial.page_width = ch.metadata.spatial.page_width
                        new_chunk.metadata.spatial.page_height = ch.metadata.spatial.page_height
                    out.append(new_chunk)
                except Exception:
                    out.append(ch)
                    break

        if split_count:
            logger.info(f"[OVERSIZE-BREAKER] Split {split_count} oversized chunks (> {max_chars} chars)")

        return out

    def _looks_like_code_text(self, text: str) -> bool:
        """Heuristic code detector for technical manuals (used in layout-aware OCR mode)."""
        import re

        if not text:
            return False

        t = text.strip("\n")
        if "```" in t:
            return True

        lines = [ln for ln in t.splitlines() if ln.strip()]
        if len(lines) < 2:
            # Only treat REPL prompts as code when they appear as actual prompts,
            # not when they're merely mentioned in prose (e.g., "after the >>> line").
            if re.search(r"(?m)^\s*>>>\s", t) or re.search(r"(?m)^\s*\.\.\.\s", t):
                return True
            if re.search(r"^\s*(import|from)\b.+\s:\s+[A-Z]", t):
                return False
            if re.search(r"^\s*(def|class)\b", t):
                return True
            # Avoid false positives on English "from ..."; require Python import syntax.
            if re.search(r"^\s*from\s+[A-Za-z_][\w\.]*\s+import\b", t):
                return True
            if re.search(
                r"^\s*import\s+[A-Za-z_][\w\.]*(\s*,\s*[A-Za-z_][\w\.]*)*(\s+as\s+[A-Za-z_][\w]*)?\s*$",
                t,
            ):
                return True
            # REQ-STRUCT-01 / _looks_like_code_text flat-code extension:
            # A broken PDF generator (e.g. Kimothi 2025) strips all \n from code blocks,
            # producing one very long flat string. The ^ anchors above cannot match
            # keywords that appear mid-string. Detect flat code by looking for multiple
            # Python keyword occurrences anywhere in a long single-line string.
            if len(t) > 80 and "\n" not in t:
                kw_hits = len(re.findall(
                    r"\b(import|from|def|class|return|raise|yield|elif|else|except|with|pass|break|continue)\b",
                    t,
                ))
                if kw_hits >= 2 and re.search(r"[=\(\)\[\]\{\}:.,]", t):
                    return True
            return False

        indented = sum(1 for ln in lines if ln.startswith(("    ", "\t")))
        if indented / max(len(lines), 1) >= 0.3:
            return True

        if any(ln.lstrip().startswith((">>>", "...")) for ln in lines):
            return True

        explanatory_import_lines = sum(
            1 for ln in lines if re.search(r"^\s*(import|from)\b.+\s:\s+[A-Z]", ln)
        )
        if explanatory_import_lines >= max(1, len(lines) // 2):
            return False

        if re.search(r"^\s*(def|class|return|yield)\b", t, flags=re.MULTILINE):
            return True

        if re.search(r"(?m)^\s*from\s+[A-Za-z_][\w\.]*\s+import\b", t):
            return True

        if re.search(
            r"(?m)^\s*import\s+[A-Za-z_][\w\.]*(\s*,\s*[A-Za-z_][\w\.]*)*(\s+as\s+[A-Za-z_][\w]*)?\s*$",
            t,
        ):
            return True

        return False

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
                is_marked_code = bool(
                    chunk.metadata
                    and (
                        chunk.metadata.content_classification == "code"
                        or chunk.metadata.chunk_type == ChunkType.CODE
                    )
                )
                # Avoid mutating code syntax/indentation.
                if not is_marked_code:
                    original = chunk.content
                    cleaned = self._post_process_ocr_text(original)
                    if original != cleaned:
                        chunk.content = cleaned
                        logger.debug(f"[OCR-CLEAN] Fixed technical values in chunk {chunk.chunk_id}")

                # Always run false-code demotion, including scanned_degraded profile.
                self._maybe_demote_false_code_chunk(chunk)

        # Step 3: Profile-specific text hygiene (technical manuals are sensitive to
        # embedded page numbers, control chars, hyphenation, and broken chunk joins).
        profile_type = str(self._intelligence_metadata.get("profile_type", "unknown"))
        if profile_type == "technical_manual":
            before = len(valid_chunks)
            valid_chunks = self._apply_technical_manual_hygiene(valid_chunks)
            after = len(valid_chunks)
            if after != before:
                logger.info(
                    f"[TECHMANUAL-HYGIENE] Joined chunks: {before} -> {after} (delta={after-before})"
                )

        # Step 4: Look-ahead buffer for symmetric overlap (fill next_text_snippet)
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
        doc: Optional[fitz.Document] = None

        try:
            doc = fitz.open(pdf_path)
            for page_idx in range(len(doc)):
                page_no = page_idx + 1
                if self._processed_pages is not None and page_no not in self._processed_pages:
                    continue
                page = doc.load_page(page_idx)
                rect = page.rect
                # Convert PDF points to integer pixels (at 72 DPI base)
                width_px = int(rect.width)
                height_px = int(rect.height)
                page_dims[page_no] = (width_px, height_px)

            logger.info(f"[REQ-COORD-02] Extracted page dimensions for {len(page_dims)} pages")
        except Exception as e:
            logger.warning(f"[REQ-COORD-02] Failed to extract page dimensions: {e}")
        finally:
            if doc is not None:
                try:
                    doc.close()
                except Exception:
                    pass

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
        # Only run if variance exceeds threshold.
        #
        # Profile-aware tweak:
        # - technical_manual conversions should prioritize recall (code/books);
        #   trigger recovery sooner to avoid stopping just above the threshold
        #   (e.g., -9.8% would otherwise skip recovery entirely).
        profile_type = str(self._intelligence_metadata.get("profile_type", "unknown"))
        RECOVERY_THRESHOLD = -8.0 if profile_type == "technical_manual" else -10.0
        MIN_ORPHAN_LENGTH = 50  # Minimum chars to rescue (can be lowered on front pages)
        MAX_TOC_LINE_RESCUES = 8
        MAX_TOTAL_RECOVERY_CHUNKS = 48 if profile_type == "technical_manual" else 200
        ocr_recovery_allowed = bool(self.enable_ocr or self.force_ocr)

        if variance_percent >= RECOVERY_THRESHOLD:
            logger.info(
                f"[RECOVERY] Variance {variance_percent:.1f}% is within tolerance, skipping recovery"
            )
            return chunks

        if not ocr_recovery_allowed:
            logger.info(
                "[RECOVERY] OCR-assisted image recovery disabled "
                "(enable_ocr=False and force_ocr=False); using text-layer recovery only"
            )
        else:
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
            doc: Optional[fitz.Document] = None
            raw_text_per_page: Dict[int, str] = {}
            text_blocks_per_page: Dict[int, List[Tuple[List[float], str]]] = {}
            page_size_per_page: Dict[int, Tuple[float, float]] = {}
            has_text_layer = False

            try:
                doc = fitz.open(self._current_pdf_path)
                for page_idx in range(len(doc)):
                    page_no = page_idx + 1
                    if self._processed_pages is not None and page_no not in self._processed_pages:
                        continue

                    page = doc.load_page(page_idx)
                    page_size_per_page[page_no] = (float(page.rect.width), float(page.rect.height))
                    page_text = page.get_text("text")
                    if page_text and page_text.strip():
                        raw_text_per_page[page_no] = page_text.strip()
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
                        text_blocks_per_page[page_no] = page_blocks
            finally:
                if doc is not None:
                    try:
                        doc.close()
                    except Exception:
                        pass

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
            primary_text_chars_per_page: Dict[int, int] = {}

            for chunk in chunks:
                if chunk.modality != Modality.TEXT:
                    continue

                page_no = chunk.metadata.page_number
                content = chunk.content.strip()
                extraction_method = str(getattr(chunk.metadata, "extraction_method", "") or "").lower()

                if page_no not in covered_text_per_page:
                    covered_text_per_page[page_no] = []

                if content and len(content) >= 10:
                    covered_text_per_page[page_no].append(content.lower())
                    if not extraction_method.startswith("recovery_"):
                        primary_text_chars_per_page[page_no] = (
                            primary_text_chars_per_page.get(page_no, 0) + len(content)
                        )

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

            def _normalize_bbox_pdf_points(page_no: int, bbox: List[float]) -> List[int]:
                """
                Convert PyMuPDF block bboxes (PDF points) to normalized [0,1000] ints (REQ-COORD-01).
                """
                w_h = page_size_per_page.get(page_no)
                if not w_h:
                    return [0, 0, COORD_SCALE, COORD_SCALE]
                page_w, page_h = w_h
                if page_w <= 0 or page_h <= 0:
                    return [0, 0, COORD_SCALE, COORD_SCALE]

                x0 = int(round((bbox[0] / page_w) * COORD_SCALE))
                y0 = int(round((bbox[1] / page_h) * COORD_SCALE))
                x1 = int(round((bbox[2] / page_w) * COORD_SCALE))
                y1 = int(round((bbox[3] / page_h) * COORD_SCALE))

                # Clamp to [0, 1000]
                x0 = max(0, min(COORD_SCALE, x0))
                y0 = max(0, min(COORD_SCALE, y0))
                x1 = max(0, min(COORD_SCALE, x1))
                y1 = max(0, min(COORD_SCALE, y1))

                # Ensure bbox is well-formed
                if x1 <= x0 or y1 <= y0:
                    return [0, 0, COORD_SCALE, COORD_SCALE]
                return [x0, y0, x1, y1]

            def _clean_recovery_text(s: str, is_code: bool = False) -> str:
                raw = s or ""
                if is_code:
                    # Indentation shield for recovery path as well.
                    return self._preserve_or_reflow_code_text(raw)
                s2 = self._strip_control_chars(raw)
                s2 = self._remove_standalone_page_number_lines(s2)
                s2 = self._remove_all_digit_only_lines(s2)
                s2 = self._fix_linebreak_hyphenation(s2)
                return s2.strip()

            def _apply_toc_recovery_policy(
                chunk: IngestionChunk,
                toc_like_page: bool,
            ) -> None:
                if not toc_like_page:
                    return
                try:
                    # Keep recovered TOC/index text for recall, but make it low-priority.
                    self._demote_toc_index_chunk(chunk)
                    chunk.metadata.search_priority = "low"
                except Exception:
                    pass

            def _has_recovery_capacity() -> bool:
                return len(recovery_chunks) < MAX_TOTAL_RECOVERY_CHUNKS

            # Step 3: Find orphaned text blocks per page
            total_rescued = 0
            coverage_by_page: Dict[int, float] = {}
            flagged_front_pages: List[int] = []

            for page_no, raw_text in raw_text_per_page.items():
                if not _has_recovery_capacity():
                    logger.info(
                        f"[RECOVERY] Recovery cap reached ({MAX_TOTAL_RECOVERY_CHUNKS} chunks); "
                        "stopping additional rescue."
                    )
                    break

                # Recovery is intended for pages where primary extraction is effectively blank.
                # If we already extracted enough native text on this page, skip noisy rescue passes.
                primary_chars = primary_text_chars_per_page.get(page_no, 0)
                if primary_chars >= 50:
                    logger.debug(
                        f"[RECOVERY] Skipping page {page_no}: primary extraction already has "
                        f"{primary_chars} chars"
                    )
                    continue

                # Front pages are allowed a lower threshold and stricter coverage target
                is_front_page = page_no <= 2
                page_min_orphan = 20 if is_front_page else MIN_ORPHAN_LENGTH

                toc_like_page = False
                try:
                    profile_type = self._intelligence_metadata.get("profile_type", "unknown")
                    if profile_type == "technical_manual" and self._is_toc_or_index_text(raw_text):
                        toc_like_page = True
                        logger.info(
                            f"[RECOVERY] TOC/Index-like page {page_no}: "
                            "allowing recovery with low-priority demotion"
                        )
                except Exception:
                    pass

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
                        if not _has_recovery_capacity():
                            break
                        text_clean = _clean_recovery_text(block_text.strip())
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
                            bbox=_normalize_bbox_pdf_points(page_no, bbox),
                            extraction_method="recovery_frontpage",
                            content_classification=self._classify_recovery_text_content(text_clean),
                            **self._intelligence_metadata,
                        )
                        _apply_toc_recovery_policy(recovery_chunk, toc_like_page)
                        recovery_chunks.append(recovery_chunk)
                        total_rescued += 1
                        covered_texts.append(para_lower)

                # Clean page-level text once (prevents TOC/page-number artifacts from entering recovery chunks).
                raw_text = _clean_recovery_text(raw_text)

                # Split raw text into paragraphs/blocks
                if toc_like_page:
                    # TOC/index pages often become one giant paragraph; recover bounded line units instead.
                    paragraphs = [ln for ln in (raw_text or "").splitlines() if ln.strip()]
                else:
                    paragraphs = re.split(r"\n\s*\n|\n{2,}", raw_text)
                covered_texts = covered_text_per_page.get(page_no, covered_texts)
                toc_rescued = 0

                for para_idx, para in enumerate(paragraphs):
                    if not _has_recovery_capacity():
                        break
                    para_clean = _clean_recovery_text(para.strip())

                    # Skip short paragraphs
                    if len(para_clean) < page_min_orphan:
                        continue

                    if toc_like_page and toc_rescued >= MAX_TOC_LINE_RESCUES:
                        break

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
                            content_classification=self._classify_recovery_text_content(para_clean),
                            **self._intelligence_metadata,
                        )
                        _apply_toc_recovery_policy(recovery_chunk, toc_like_page)

                        recovery_chunks.append(recovery_chunk)
                        total_rescued += 1
                        if toc_like_page:
                            toc_rescued += 1

                # Step 3b: Code-aware recovery for text blocks overlapping figures (subsurface extraction)
                if (not toc_like_page) and page_no in figure_bboxes_per_page and page_no in text_blocks_per_page:
                    existing_texts = covered_text_per_page.get(page_no, [])
                    for fig_bbox, fig_chunk in figure_bboxes_per_page[page_no]:
                        if not _has_recovery_capacity():
                            break
                        for bbox, block_text in text_blocks_per_page[page_no]:
                            if not _has_recovery_capacity():
                                break
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
                                ratio = SequenceMatcher(None, para_lower[:200], covered[:200]).ratio()
                                if ratio > 0.8:
                                    is_covered = True
                                    break
                            if is_covered:
                                continue

                            classification = self._classify_recovery_text_content(block_text)
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
                            cleaned_block = _clean_recovery_text(
                                block_text.strip(), is_code=(classification == "code")
                            )
                            if len(cleaned_block) < 20:
                                continue
                            recovery_chunk = create_text_chunk(
                                doc_id=self._doc_hash or "unknown",
                                content=cleaned_block,
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
                            _apply_toc_recovery_policy(recovery_chunk, toc_like_page)
                            recovery_chunks.append(recovery_chunk)
                            total_rescued += 1
                            existing_texts.append(para_lower)

                # Step 3c: Low-coverage gap fill (spatial gap filling beyond figures)

                if (not toc_like_page) and coverage_ratio < 0.6 and page_no in text_blocks_per_page:
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
                        if not _has_recovery_capacity():
                            break
                        text_clean = _clean_recovery_text(block_text.strip())
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

                        classification = self._classify_recovery_text_content(text_clean)
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
                            bbox=_normalize_bbox_pdf_points(page_no, bbox),
                            extraction_method="recovery_gap_fill",
                            content_classification=classification,
                            **self._intelligence_metadata,
                        )
                        _apply_toc_recovery_policy(recovery_chunk, toc_like_page)
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
                if ocr_recovery_allowed:
                    try:
                        chunks = self._process_front_pages_enhanced(
                            chunks, flagged_front_pages, covered_text_per_page
                        )
                    except Exception as e:
                        logger.warning(f"[RECOVERY] Enhanced front-page processing skipped: {e}")
                else:
                    logger.info(
                        "[RECOVERY] Skipping enhanced front-page OCR pass "
                        "(enable_ocr=False and force_ocr=False)"
                    )

        except Exception as e:
            logger.error(f"[RECOVERY] TextIntegrityScout failed: {e}")
            print(f"    ⚠️ [RECOVERY] Scout failed: {e}", flush=True)

        return chunks

    def _apply_table_recovery_highlander_dedup(
        self,
        chunks: List[IngestionChunk],
    ) -> List[IngestionChunk]:
        """
        Drop recovery text chunks that duplicate forced VLM table chunks on the same page.

        Rule set ("Highlander"):
        1. Identify table chunks extracted via `vlm_table_markdown_forced`.
        2. For recovery chunks (`recovery_gap_fill` / `recovery_scan`) on the same page:
           - If both bboxes exist, drop recovery chunk when intersection area
             covers >50% of the recovery bbox.
           - If either bbox is missing, fallback to token-overlap and drop when
             >30% of recovery unique tokens are present in the VLM table text.
        """
        table_method = "vlm_table_markdown_forced"
        zombie_methods = {"recovery_gap_fill", "recovery_scan"}

        tables_by_page: Dict[int, List[IngestionChunk]] = {}
        for chunk in chunks:
            try:
                method = str(getattr(chunk.metadata, "extraction_method", "") or "").lower()
                if chunk.modality == Modality.TABLE and method == table_method:
                    page_no = int(getattr(chunk.metadata, "page_number", 0) or 0)
                    if page_no > 0:
                        tables_by_page.setdefault(page_no, []).append(chunk)
            except Exception:
                continue

        if not tables_by_page:
            return chunks

        def _safe_bbox(ch: IngestionChunk) -> Optional[List[int]]:
            try:
                spatial = getattr(ch.metadata, "spatial", None)
                bbox = getattr(spatial, "bbox", None)
                if not bbox or len(bbox) != 4:
                    return None
                x0, y0, x1, y1 = [int(v) for v in bbox]
                if x1 <= x0 or y1 <= y0:
                    return None
                return [x0, y0, x1, y1]
            except Exception:
                return None

        def _intersection_ratio_of_first(b1: List[int], b2: List[int]) -> float:
            x0 = max(b1[0], b2[0])
            y0 = max(b1[1], b2[1])
            x1 = min(b1[2], b2[2])
            y1 = min(b1[3], b2[3])
            if x1 <= x0 or y1 <= y0:
                return 0.0
            inter = float((x1 - x0) * (y1 - y0))
            area1 = float((b1[2] - b1[0]) * (b1[3] - b1[1]))
            if area1 <= 0:
                return 0.0
            return inter / area1

        def _unique_tokens(text: str) -> set[str]:
            import re

            tokens = re.findall(r"[A-Za-z0-9]{3,}", (text or "").lower())
            return set(tokens)

        def _token_overlap_ratio(recovery_text: str, table_text: str) -> float:
            rec_tokens = _unique_tokens(recovery_text)
            if not rec_tokens:
                return 0.0
            tbl_tokens = _unique_tokens(table_text)
            if not tbl_tokens:
                return 0.0
            return len(rec_tokens & tbl_tokens) / max(1, len(rec_tokens))

        kept: List[IngestionChunk] = []
        dropped_total = 0
        dropped_spatial = 0
        dropped_text = 0

        for chunk in chunks:
            try:
                page_no = int(getattr(chunk.metadata, "page_number", 0) or 0)
                method = str(getattr(chunk.metadata, "extraction_method", "") or "").lower()
            except Exception:
                kept.append(chunk)
                continue

            if (
                chunk.modality != Modality.TEXT
                or method not in zombie_methods
                or page_no not in tables_by_page
            ):
                kept.append(chunk)
                continue

            should_drop = False
            drop_reason = ""
            recovery_bbox = _safe_bbox(chunk)

            for table_chunk in tables_by_page[page_no]:
                table_bbox = _safe_bbox(table_chunk)
                if recovery_bbox is not None and table_bbox is not None:
                    overlap_ratio = _intersection_ratio_of_first(recovery_bbox, table_bbox)
                    if overlap_ratio > 0.50:
                        should_drop = True
                        dropped_spatial += 1
                        drop_reason = f"spatial_overlap={overlap_ratio:.2f}"
                        break
                else:
                    token_ratio = _token_overlap_ratio(chunk.content, table_chunk.content)
                    if token_ratio > 0.30:
                        should_drop = True
                        dropped_text += 1
                        drop_reason = f"text_overlap={token_ratio:.2f}"
                        break

            if should_drop:
                dropped_total += 1
                logger.info(
                    f"[HIGHLANDER] Dropping recovery chunk {chunk.chunk_id} "
                    f"(page={page_no}, method={method}, reason={drop_reason})"
                )
                continue

            kept.append(chunk)

        if dropped_total > 0:
            logger.info(
                f"[HIGHLANDER] Dedup complete: dropped {dropped_total} recovery duplicates "
                f"(spatial={dropped_spatial}, text={dropped_text})"
            )
            print(
                f"🗡️ [HIGHLANDER] Dropped {dropped_total} recovery duplicates "
                f"(spatial={dropped_spatial}, text={dropped_text})",
                flush=True,
            )

        return kept

    def _release_torch_runtime_memory(self) -> None:
        """
        Best-effort release of OCR runtime memory after EasyOCR phases.

        EasyOCR pulls in PyTorch internals; clearing caches here reduces
        peak-memory overlap with the final technical-manual hygiene pass.
        """
        try:
            import torch  # type: ignore

            try:
                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            try:
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()  # type: ignore[attr-defined]
            except Exception:
                pass
        except Exception:
            pass

        gc.collect()

    def _release_extraction_runtime_models(self, label: str = "[MEMORY] extraction release") -> None:
        """
        Release heavy extraction/runtime objects no longer needed after batch extraction.

        This proactively frees model-backed objects (Docling converter + shadow processor)
        before recovery/finalization phases to avoid late hard-kill OOM conditions.
        """
        released_any = False

        # Shadow processor can retain runtime/model state; give it a chance to cleanup.
        if self._shadow_processor is not None:
            try:
                cleanup = getattr(self._shadow_processor, "cleanup", None)
                if callable(cleanup):
                    cleanup()
            except Exception as e:
                logger.debug(f"[MEMORY] shadow processor cleanup skipped: {e}")
            finally:
                self._shadow_processor = None
                released_any = True

        # Docling converter holds heavy ML models in memory.
        if self._docling_converter is not None:
            try:
                for method_name in ("cleanup", "close", "shutdown"):
                    method = getattr(self._docling_converter, method_name, None)
                    if callable(method):
                        method()
            except Exception as e:
                logger.debug(f"[MEMORY] docling converter cleanup skipped: {e}")
            finally:
                self._docling_converter = None
                released_any = True

        # Layout-aware OCR processor can retain OCR runtime models.
        layout_processor = getattr(self, "_layout_processor", None)
        if layout_processor is not None:
            try:
                for method_name in ("cleanup", "close", "shutdown"):
                    method = getattr(layout_processor, method_name, None)
                    if callable(method):
                        method()
            except Exception as e:
                logger.debug(f"[MEMORY] layout processor cleanup skipped: {e}")
            finally:
                self._layout_processor = None
                released_any = True

        # Always clear OCR-runtime caches, even when no object needed explicit teardown.
        self._release_torch_runtime_memory()

        if released_any:
            self._log_memory_checkpoint(label)

    def _release_vision_runtime_models(
        self, label: str = "[MEMORY] vision release"
    ) -> Dict[str, Any]:
        """
        Release vision-side runtime state before finalize-heavy phases.

        Returns:
            Best-effort vision stats snapshot captured before release.
        """
        stats: Dict[str, Any] = {}
        if self._vision_manager is None:
            return stats

        try:
            stats = self._vision_manager.get_stats()
        except Exception as e:
            logger.debug(f"[MEMORY] vision stats snapshot skipped: {e}")

        try:
            self._vision_manager.flush_cache()
        except Exception as e:
            logger.debug(f"[MEMORY] vision cache flush skipped: {e}")
        finally:
            # Drop reference so cache payloads can be reclaimed before finalize.
            self._vision_manager = None

        self._release_torch_runtime_memory()
        self._log_memory_checkpoint(label)
        return stats

    def _get_process_rss_mb(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Return (current_rss_mb, peak_rss_mb) when available.

        - current_rss_mb: best-effort via psutil (optional dependency)
        - peak_rss_mb: ru_maxrss via stdlib resource
        """
        current_rss_mb: Optional[float] = None
        peak_rss_mb: Optional[float] = None

        # Current RSS (optional; psutil may not be installed in all environments).
        try:
            import os
            import psutil  # type: ignore

            proc = psutil.Process(os.getpid())
            current_rss_mb = proc.memory_info().rss / (1024.0 * 1024.0)
        except Exception:
            pass

        # Peak RSS (stdlib; available on macOS/Linux).
        try:
            import resource

            raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            # macOS reports bytes; Linux reports KB.
            if raw > 10_000_000:
                peak_rss_mb = raw / (1024.0 * 1024.0)
            else:
                peak_rss_mb = raw / 1024.0
        except Exception:
            pass

        return current_rss_mb, peak_rss_mb

    def _log_memory_checkpoint(self, label: str) -> None:
        """Emit a standardized memory checkpoint log line."""
        current_rss_mb, peak_rss_mb = self._get_process_rss_mb()
        cur_str = f"{current_rss_mb:.1f}MB" if current_rss_mb is not None else "n/a"
        peak_str = f"{peak_rss_mb:.1f}MB" if peak_rss_mb is not None else "n/a"
        logger.info(f"[MEMORY] {label}: rss={cur_str}, peak_rss={peak_str}")

    def _reclassify_text_images(self, chunks: List[IngestionChunk]) -> List[IngestionChunk]:
        """
        Phase 3: Reclassify IMAGE chunks that likely contain text (front pages only).
        Uses EasyOCR if available. Guardrails: max 5 images per page, pages 1-2 only.
        """
        import gc
        
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

        # EARLY-EXIT GUARD: Check if there are any images that might need OCR
        # before loading EasyOCR models into memory
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
        
        # Early exit if no candidates found
        if not images_by_page:
            logger.debug("[RECOVERY] No text-like image candidates found; skipping EasyOCR load")
            return chunks
        
        # Log memory before loading EasyOCR
        logger.info(f"[MEMORY] Loading EasyOCR for {sum(len(v) for v in images_by_page.values())} candidate images...")
        self._log_memory_checkpoint("[RECOVERY] image->text before EasyOCR load")

        reader = None
        max_per_page = 5
        updated = 0

        def _resolve_asset_path(file_path: str) -> Path:
            """
            Resolve an asset_ref.file_path to an absolute path.

            asset_ref.file_path is typically stored as a document-relative path like
            'assets/<doc>_<page>_figure_XX.png'. Recovery helpers must never pass
            that relative string into OCR/vision libraries because they may resolve
            relative to the current working directory (causing silent misses).
            """
            p = Path(file_path)
            if p.is_absolute():
                return p
            return self.output_dir / p

        try:
            reader = easyocr.Reader(["en"], gpu=False)  # small, CPU-safe
            gc.collect()  # MEMORY FIX: Force GC after EasyOCR model load
            self._log_memory_checkpoint("[RECOVERY] image->text after EasyOCR load")

            for page_no, page_imgs in images_by_page.items():
                attempts = 0
                for img_chunk in page_imgs:
                    if attempts >= max_per_page:
                        break
                    if not img_chunk.asset_ref or not getattr(img_chunk.asset_ref, "file_path", None):
                        continue
                    attempts += 1
                    try:
                        asset_path = _resolve_asset_path(img_chunk.asset_ref.file_path)
                        if not asset_path.exists():
                            continue
                        with Image.open(asset_path) as img:
                            width, height = img.size
                        if width < 40 or height < 40:
                            continue
                        result = reader.readtext(str(asset_path), detail=0)
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
                            img_chunk.metadata.content_classification = self._classify_recovery_text_content(ocr_text)
                        updated += 1
                    except Exception as e:  # pragma: no cover
                        logger.debug(f"[RECOVERY] OCR failed for page {page_no} image: {e}")
                        continue
        finally:
            if reader is not None:
                try:
                    del reader
                except Exception:
                    pass
            self._release_torch_runtime_memory()
            self._log_memory_checkpoint("[RECOVERY] image->text after EasyOCR release")

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
        import gc
        
        try:
            import easyocr  # type: ignore
        except Exception as e:  # pragma: no cover
            logger.info(f"[RECOVERY] EasyOCR not available for enhanced pass: {e}")
            return chunks
        
        # EARLY-EXIT GUARD: Check if there are any images on the flagged pages
        # before loading EasyOCR models into memory
        images_on_flagged_pages = []
        for ch in chunks:
            if ch.modality == Modality.IMAGE and ch.metadata and ch.metadata.page_number in pages:
                images_on_flagged_pages.append(ch)
        
        # Early exit if no images on flagged pages
        if not images_on_flagged_pages:
            logger.debug("[RECOVERY] No images on flagged front pages; skipping EasyOCR load")
            return chunks
        
        # Log memory before loading EasyOCR
        logger.info(f"[MEMORY] Loading EasyOCR for enhanced front-page recovery on {len(pages)} pages...")
        self._log_memory_checkpoint("[RECOVERY] enhanced frontpage before EasyOCR load")

        reader = None
        doc = None
        new_chunks: List[IngestionChunk] = []
        seen_hashes = set()

        def _resolve_asset_path(file_path: str) -> Path:
            p = Path(file_path)
            if p.is_absolute():
                return p
            return self.output_dir / p

        try:
            reader = easyocr.Reader(["en"], gpu=False)  # CPU-safe
            gc.collect()  # MEMORY FIX: Force GC after EasyOCR model load
            self._log_memory_checkpoint("[RECOVERY] enhanced frontpage after EasyOCR load")

            doc = fitz.open(self._current_pdf_path)

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
                page_imgs = [
                    c for c in chunks if c.modality == Modality.IMAGE and c.metadata.page_number == page_no
                ]
                ocr_count = 0
                for img_chunk in page_imgs:
                    if not img_chunk.asset_ref or not getattr(img_chunk.asset_ref, "file_path", None):
                        continue
                    if ocr_count >= 5:
                        break
                    try:
                        asset_path = _resolve_asset_path(img_chunk.asset_ref.file_path)
                        if not asset_path.exists():
                            continue
                        result = reader.readtext(str(asset_path), detail=0)
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
                            img_chunk.metadata.content_classification = self._classify_recovery_text_content(ocr_text)
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
                    # Normalize PyMuPDF block bbox (PDF points) to REQ-COORD-01 scale.
                    page_w = float(page.rect.width)
                    page_h = float(page.rect.height)
                    if page_w > 0 and page_h > 0:
                        x0 = int(round((float(b[0]) / page_w) * COORD_SCALE))
                        y0 = int(round((float(b[1]) / page_h) * COORD_SCALE))
                        x1 = int(round((float(b[2]) / page_w) * COORD_SCALE))
                        y1 = int(round((float(b[3]) / page_h) * COORD_SCALE))
                        bbox = [
                            max(0, min(COORD_SCALE, x0)),
                            max(0, min(COORD_SCALE, y0)),
                            max(0, min(COORD_SCALE, x1)),
                            max(0, min(COORD_SCALE, y1)),
                        ]
                    else:
                        bbox = [0, 0, COORD_SCALE, COORD_SCALE]
                    new_chunk = create_text_chunk(
                        doc_id=self._doc_hash or "unknown",
                        content=text_clean,
                        source_file=str(self._current_pdf_path.name),
                        file_type=FileType.PDF,
                        page_number=page_no,
                        hierarchy=hierarchy,
                        bbox=bbox,
                        extraction_method="enhanced_frontpage",
                        content_classification=self._classify_recovery_text_content(text_clean),
                        **self._intelligence_metadata,
                    )
                    new_chunks.append(new_chunk)
        finally:
            if doc is not None:
                try:
                    doc.close()
                except Exception:
                    pass
            if reader is not None:
                try:
                    del reader
                except Exception:
                    pass
            self._release_torch_runtime_memory()
            self._log_memory_checkpoint("[RECOVERY] enhanced frontpage after EasyOCR release")

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

        Uses PyMuPDF to extract raw text from PDF as source of truth for triggering
        recovery mechanisms. This ensures we catch missing content that Docling
        may have filtered out or missed.

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
            # Use PyMuPDF raw text extraction as source for recovery triggering
            # ================================================================
            # PyMuPDF sees the raw PDF text layer which includes content that
            # Docling may filter out (ads, headers, etc.) or miss entirely.
            # This higher baseline ensures recovery mechanisms trigger when
            # content is missing from Docling's extraction.
            # ================================================================
            source_text = ""
            if self._current_pdf_path and self._current_pdf_path.exists():
                try:
                    doc: Optional[fitz.Document] = None
                    all_text_parts = []
                    try:
                        doc = fitz.open(self._current_pdf_path)
                        for page_idx in range(len(doc)):
                            page_no = page_idx + 1
                            if self._processed_pages is not None and page_no not in self._processed_pages:
                                continue
                            page = doc.load_page(page_idx)
                            page_text = page.get_text("text")
                            if page_text:
                                all_text_parts.append(page_text.strip())
                    finally:
                        if doc is not None:
                            try:
                                doc.close()
                            except Exception:
                                pass
                    source_text = "\n".join(all_text_parts)
                    logger.info(
                        f"[QA-CHECK-01] Extracted {len(source_text)} chars from PDF for validation "
                        f"(pages={len(self._processed_pages) if self._processed_pages else 'ALL'})"
                    )
                except Exception as pdf_err:
                    logger.warning(f"[QA-CHECK-01] Failed to extract PDF text: {pdf_err}")
                    source_text = " ".join(c.content for c in text_chunks if c.content)
            else:
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
                overlap_ratio=self._semantic_overlap_ratio,  # DSO overlap (adaptive-capable)
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
            is_code_chunk = False
            try:
                is_code_chunk = (
                    chunk.metadata.content_classification == "code"
                    or chunk.metadata.chunk_type == ChunkType.CODE
                )
            except Exception:
                is_code_chunk = False

            if is_code_chunk:
                sub_chunks = self._smart_split_code(
                    text=chunk.content,
                    max_tokens=max_tokens,
                    overlap_lines=5,
                )
            else:
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
                    bbox=(chunk.metadata.spatial.bbox if chunk.metadata.spatial else None),
                    extraction_method=chunk.metadata.extraction_method,
                    prev_text=(
                        chunk.semantic_context.prev_text_snippet if chunk.semantic_context else None
                    ),
                    next_text=(
                        chunk.semantic_context.next_text_snippet if chunk.semantic_context else None
                    ),
                    content_classification=getattr(chunk.metadata, "content_classification", None),
                    **{k: v for k, v in self._intelligence_metadata.items() if v is not None},
                )

                # Override chunk_id with our custom split ID
                new_chunk.chunk_id = new_chunk_id

                # Preserve page dimension metadata if it existed on the original chunk.
                if chunk.metadata.spatial:
                    if new_chunk.metadata.spatial is None:
                        new_chunk.metadata.spatial = SpatialMetadata(bbox=None)
                    if chunk.metadata.spatial.page_width is not None:
                        new_chunk.metadata.spatial.page_width = chunk.metadata.spatial.page_width
                    if chunk.metadata.spatial.page_height is not None:
                        new_chunk.metadata.spatial.page_height = chunk.metadata.spatial.page_height

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

    def _smart_split_code(
        self,
        text: str,
        max_tokens: int = 512,
        overlap_lines: int = 5,
    ) -> List[str]:
        """
        Split code-like text on line boundaries to preserve scope/indentation.

        Token-aware using the configured TokenValidator counter.
        """
        if self._token_validator is None or self._token_validator._counter is None:
            # Fallback: line-based split by approximate character budget.
            max_chars = max_tokens * 4
            lines = text.splitlines()
            chunks: List[str] = []
            cur: List[str] = []
            cur_len = 0
            for ln in lines:
                add = len(ln) + (1 if cur else 0)
                if cur and cur_len + add > max_chars:
                    chunks.append("\n".join(cur).strip())
                    cur = cur[-overlap_lines:] if overlap_lines > 0 else []
                    cur_len = sum(len(x) + 1 for x in cur)
                cur.append(ln)
                cur_len += add
            if cur:
                chunks.append("\n".join(cur).strip())
            return [c for c in chunks if c.strip()] or [text]

        lines = text.splitlines()
        if not lines:
            return [text]

        chunks: List[str] = []
        cur: List[str] = []
        for ln in lines:
            candidate = "\n".join(cur + [ln]) if cur else ln
            if cur and self._token_validator._counter.count_tokens(candidate) > max_tokens:
                chunks.append("\n".join(cur).strip())
                cur = cur[-overlap_lines:] if overlap_lines > 0 else []
            cur.append(ln)
        if cur:
            chunks.append("\n".join(cur).strip())

        return [c for c in chunks if c.strip()] or [text]

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
                        "[FULL-PAGE EDITORIAL IMAGE]"
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

    def process_to_jsonl(self, file_path: str) -> str:
        """
        Compatibility wrapper for CLI integration.

        The CLI expects a process_to_jsonl method. For BatchProcessor,
        this maps to process_pdf.
        """
        result = self.process_pdf(file_path)
        if not result.success:
            error_msg = "; ".join(result.errors)
            raise RuntimeError(f"Batch processing failed for {file_path}: {error_msg}")
        return str(result.output_jsonl)

    def process_to_jsonl_atomic(self, file_path: str) -> str:
        """Alias for process_to_jsonl (BatchProcessor is already atomic)."""
        return self.process_to_jsonl(file_path)

    def cleanup(self) -> None:
        """
        Best-effort resource cleanup for graceful shutdown paths.

        This is safe to call multiple times and helps reduce leaked worker
        resources when a run exits with errors.
        """
        try:
            if self._vision_manager:
                try:
                    self._vision_manager.flush_cache()
                except Exception as e:
                    logger.debug(f"[CLEANUP] vision cache flush failed: {e}")

            if self._image_hash_registry:
                clear_fn = getattr(self._image_hash_registry, "clear", None)
                if callable(clear_fn):
                    try:
                        clear_fn()
                    except Exception as e:
                        logger.debug(f"[CLEANUP] image hash registry clear failed: {e}")

            if self._refiner:
                for method_name in ("shutdown", "close"):
                    method = getattr(self._refiner, method_name, None)
                    if callable(method):
                        try:
                            method()
                        except Exception as e:
                            logger.debug(f"[CLEANUP] refiner.{method_name} failed: {e}")

            # Drop large references to help GC reclaim memory quickly.
            self._image_hash_registry = None
            self._context_state = None
            self._vision_manager = None

            # Release cached extraction runtimes (Docling converter, shadow processor)
            # and clear torch caches.
            self._release_extraction_runtime_models("[CLEANUP] extraction runtime release")
            logger.debug("[CLEANUP] BatchProcessor cleanup complete")
        except Exception as e:
            logger.debug(f"[CLEANUP] BatchProcessor cleanup skipped due to error: {e}")


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
    force_table_vlm: bool = False,
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
        force_table_vlm: Force table image -> VLM markdown path (fallback to OCR/docling if needed)

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
        force_table_vlm=force_table_vlm,
    )
