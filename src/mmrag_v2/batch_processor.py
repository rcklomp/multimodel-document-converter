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

# V3.0.0: Shadow extraction is REMOVED per ARCHITECTURE.md
# All extraction goes through UIR → ElementProcessor
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
        enable_ocr: bool = True,
        ocr_engine: str = "easyocr",
        extraction_strategy: Optional["ExtractionStrategy"] = None,
        max_pages: Optional[int] = None,
        specific_pages: Optional[List[int]] = None,
        allow_fullpage_shadow: bool = False,
        strict_qa: bool = False,
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
        self.enable_ocr = enable_ocr
        self.ocr_engine = ocr_engine
        self.extraction_strategy = extraction_strategy
        self.max_pages = max_pages
        self.specific_pages = specific_pages
        self.allow_fullpage_shadow = allow_fullpage_shadow
        self.strict_qa = strict_qa
        self.semantic_overlap = semantic_overlap
        self.vlm_context_depth = vlm_context_depth

        # Phase 1B: Layout-aware OCR parameters
        self.ocr_mode = ocr_mode
        self.ocr_confidence_threshold = ocr_confidence_threshold
        self.enable_doctr = enable_doctr

        # Will be initialized when processing starts
        self._vision_manager: Optional[VisionManager] = None
        self._context_state: Optional[ContextStateV2] = None
        self._doc_hash: Optional[str] = None
        self._image_hash_registry: Optional[ImageHashRegistry] = None
        self._token_validator: Optional[TokenValidator] = None

        # REQ-OCR-01: Profile parameters for OCR hints and dynamic DPI
        self._profile_params: Optional["ProfileParameters"] = None

        # REQ-VLM-02: Track asset counts per page for low-recall trigger
        self._assets_per_page: Dict[int, int] = {}
        self._current_pdf_path: Optional[Path] = None

        # QA-CHECK-01: Initialize token validator for data integrity
        self._token_validator = create_token_validator(tolerance=0.10)

        # REQ-COORD-02: Track page dimensions per page for UI overlay support
        self._page_dimensions: Dict[int, Tuple[int, int]] = {}

        logger.info(
            f"BatchProcessor initialized: "
            f"batch_size={batch_size}, "
            f"vision={vision_provider}/{vision_model}, "
            f"timeout={vlm_timeout}s, "
            f"max_pages={max_pages if max_pages else 'ALL'}"
        )

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
            manager = create_vision_manager(
                provider=self.vision_provider,
                api_key=self.vision_api_key,
                cache_dir=self.output_dir,
                model=self.vision_model,
                timeout=self.vlm_timeout,
                base_url=self.vision_base_url,
            )
            logger.info(
                f"Global VisionManager initialized: "
                f"provider={self.vision_provider}, "
                f"model={self.vision_model}, "
                f"cache_dir={self.output_dir}, "
                f"base_url={self.vision_base_url}"
            )
            return manager
        except Exception as e:
            logger.warning(f"Failed to initialize VisionManager: {e}")
            return None

    # ========================================================================
    # V3.0.0: SHADOW EXTRACTION REMOVED
    # ========================================================================
    # Per ARCHITECTURE.md V3.0.0:
    # - Shadow extraction is REMOVED from the pipeline
    # - All extraction goes through UIR → ElementProcessor
    # - TEXT regions → OCR cascade → modality: "text"
    # - IMAGE regions → VLM visual description → modality: "image"
    # - Errors are logged via ElementProcessor, NO fallback to shadow
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
            metadata = ChunkMetadata(
                page_number=pc.page_number,
                source_file=source_file,
                file_type=FileType.PDF,
                hierarchy=hierarchy,
                extraction_method=extraction_method,
                visual_description=pc.visual_description,
                spatial=spatial,
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
        pipeline_options.generate_page_images = False
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

        finally:
            if doc is not None:
                doc.close()
            gc.collect()

        logger.info(f"[LAYOUT-OCR] Batch complete: {len(all_chunks)} chunks generated")
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
        )

        # REQ-OCR-01: Pass profile parameters to processor for OCR hints
        # When profile has enable_ocr_hints=True, processor will use OCR-hint-aware VLM enrichment
        if self._profile_params is not None:
            processor.set_profile_params(self._profile_params)
            logger.info(
                f"[OCR-HYBRID] Processor configured with profile params: "
                f"enable_ocr_hints={self._profile_params.enable_ocr_hints}"
            )

        # ================================================================
        # V3.0.0: STANDARD DOCLING PIPELINE (Shadow extraction REMOVED)
        # ================================================================
        # Per ARCHITECTURE.md V3.0.0:
        # - All extraction goes through Docling → UIR → ElementProcessor
        # - TEXT regions → OCR cascade → modality: "text"
        # - IMAGE regions → VLM visual description → modality: "image"
        # - NO fallback to shadow extraction - errors are logged
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
        # 4. Quality filters (existing)
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

        # Step 4: QA-CHECK-01 - Run token balance validation
        token_result = self._run_token_validation(all_chunks, pdf_path.name)
        if not token_result.is_valid:
            print(
                f"⚠️ [QA-CHECK-01] Token balance warning: {token_result.variance_percent:.1f}% variance",
                flush=True,
            )
            if self.strict_qa:
                errors.append(f"QA-CHECK-01 failed: {token_result.error_message}")

        # ====================================================================
        # PHASE 1: QUALITY IMPROVEMENTS
        # ====================================================================
        # 1. Empty chunk filtering (asset-aware)
        # 2. OCR text post-processing (number joining)
        # 3. Look-ahead buffer for symmetric overlap
        # ====================================================================

        # Apply quality filters
        filtered_chunks = self._apply_quality_filters(all_chunks)
        print(
            f"\n🔍 [QUALITY] Filtered {len(all_chunks) - len(filtered_chunks)} empty/invalid chunks",
            flush=True,
        )

        # Write aggregated output to master JSONL with deduplication
        output_jsonl = self.output_dir / "ingestion.jsonl"
        written_chunks = 0
        duplicate_count = 0

        # PHANTOM BUG FIX: Add defensive logging and error handling
        logger.info(f"[FINALIZE] Starting JSONL write: {len(filtered_chunks)} chunks to process")
        print(
            f"\n📝 [FINALIZE] Writing {len(filtered_chunks)} chunks to {output_jsonl.name}...",
            flush=True,
        )

        with open(output_jsonl, "w", encoding="utf-8") as f:
            for idx, chunk in enumerate(filtered_chunks):
                try:
                    # Log progress every 50 chunks
                    if idx % 50 == 0 and idx > 0:
                        logger.debug(f"[FINALIZE] Processed {idx}/{len(filtered_chunks)} chunks")

                    chunk_dict = chunk.model_dump(mode="json")

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
                                                logger.info(
                                                    f"Deleted duplicate asset: {asset_file}"
                                                )
                                            except Exception as del_e:
                                                logger.warning(
                                                    f"Failed to delete duplicate: {del_e}"
                                                )
                                            continue  # Skip writing this chunk

                                        # Log successful registration
                                        logger.info(
                                            f"[FINALIZING] Asset {filename} linked to "
                                            f"Page {chunk.metadata.page_number}"
                                        )

                                except Exception as hash_e:
                                    logger.warning(f"pHash check failed for {asset_file}: {hash_e}")

                    json_line = json.dumps(chunk_dict, ensure_ascii=False)
                    f.write(json_line + "\n")
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
            f"{len(all_chunks) - len(filtered_chunks)} filtered)",
            flush=True,
        )
        logger.info(
            f"Written {written_chunks} chunks to {output_jsonl} "
            f"({duplicate_count} duplicates rejected, "
            f"{len(all_chunks) - len(filtered_chunks)} filtered)"
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
    ) -> bool:
        """
        Determine if a chunk should be filtered out before export.

        CRITICAL RULE: Chunks with asset_ref (images/tables) must NEVER be
        filtered, even if content is empty. The asset itself contains information.

        Args:
            chunk: IngestionChunk to evaluate

        Returns:
            True if chunk should be skipped, False if it should be kept
        """
        # RULE 1: NEVER skip chunks with assets (REQ-MM-05)
        if chunk.asset_ref is not None:
            return False

        # RULE 2: Empty or whitespace-only content (and no asset)
        content = chunk.content or ""
        if not content or not content.strip():
            logger.debug(f"[FILTER] Skipping empty chunk: {chunk.chunk_id}")
            return True

        # RULE 3: Too short (< 3 chars - likely artifacts)
        if len(content.strip()) < 3:
            logger.debug(f"[FILTER] Skipping short chunk ({len(content)} chars): {chunk.chunk_id}")
            return True

        # RULE 4: Only special characters (decorations)
        import re

        if re.match(r"^[\s\-_=•]+$", content):
            logger.debug(f"[FILTER] Skipping decoration chunk: {chunk.chunk_id}")
            return True

        # RULE 5: Suspicious bbox (very small area) - only for non-asset chunks
        if chunk.metadata and chunk.metadata.spatial and chunk.metadata.spatial.bbox:
            bbox = chunk.metadata.spatial.bbox
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height

            # In normalized coordinates (0-1000), area < 100 is 0.01% of page
            if area < 100:
                logger.debug(f"[FILTER] Skipping tiny bbox chunk (area={area}): {chunk.chunk_id}")
                return True

        return False

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
        # Step 1: Filter out invalid chunks
        valid_chunks = [chunk for chunk in chunks if not self._should_skip_chunk(chunk)]
        logger.info(f"[QUALITY] Filtered {len(chunks) - len(valid_chunks)} invalid chunks")

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

        updated_count = 0
        for chunk in chunks:
            page_no = chunk.metadata.page_number
            dims = self._page_dimensions.get(page_no)

            if dims:
                width_px, height_px = dims

                # Ensure spatial metadata exists
                if chunk.metadata.spatial is None:
                    chunk.metadata.spatial = SpatialMetadata(bbox=None)

                # Only update if currently null (non-destructive)
                if chunk.metadata.spatial.page_width is None:
                    chunk.metadata.spatial.page_width = width_px
                    updated_count += 1
                if chunk.metadata.spatial.page_height is None:
                    chunk.metadata.spatial.page_height = height_px

        logger.info(f"[REQ-COORD-02] Propagated page dimensions to {updated_count} chunks")
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

        This validates that the sum of TEXT chunk tokens approximately matches
        the expected token count, accounting for DSO overlap.

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

            # Build source text from all TEXT chunks (without overlap)
            # NOTE: This is an approximation - true source is the original document
            source_text = " ".join(c.content for c in text_chunks if c.content)

            # Run validation
            result = self._token_validator.validate_token_balance(
                chunks=text_chunks,
                source_text=source_text,
                overlap_ratio=0.15,  # DSO default overlap
            )

            # Log result
            self._token_validator.log_validation_result(result, doc_name=source_file)

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
        QA-CHECK-01 (Token Limit): Validate individual chunk token limits.

        Per SRS REQ-CHUNK-02: Text chunks have hard max of 512 tokens.
        Chunks exceeding this limit are flagged and optionally truncated.

        Args:
            chunks: All chunks to validate
            max_tokens: Maximum allowed tokens per chunk (default: 512)

        Returns:
            Tuple of (validated_chunks, flagged_count)
        """
        flagged_count = 0

        for chunk in chunks:
            if chunk.modality != Modality.TEXT:
                continue

            # Count tokens in this chunk
            token_count = self._token_validator._counter.count_tokens(chunk.content)

            if token_count > max_tokens:
                flagged_count += 1
                logger.warning(
                    f"[QA-CHECK-01] Chunk {chunk.chunk_id} exceeds token limit: "
                    f"{token_count} > {max_tokens}. Content will be truncated."
                )

                # Truncate content to fit within limit
                # This is a simple character-based truncation; a more sophisticated
                # approach would use sentence boundaries
                if self.strict_qa:
                    raise ValueError(
                        f"[QA-CHECK-01 STRICT] Chunk {chunk.chunk_id} exceeds "
                        f"token limit ({token_count} > {max_tokens})"
                    )

                # Approximate truncation: 4 chars per token (rough estimate)
                max_chars = max_tokens * 4
                chunk.content = chunk.content[:max_chars] + "..."

        if flagged_count > 0:
            logger.warning(
                f"[QA-CHECK-01] {flagged_count} chunks exceeded token limit "
                f"(strict_qa={'ENABLED' if self.strict_qa else 'DISABLED'})"
            )

        return chunks, flagged_count

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
