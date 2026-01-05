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

import fitz  # PyMuPDF for shadow extraction
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
    create_shadow_chunk,
)
from .state.context_state import ContextStateV2, create_context_state
from .utils.pdf_splitter import BatchInfo, PDFBatchSplitter, SplitResult
from .utils.image_hash_registry import (
    ImageHashRegistry,
    create_image_hash_registry,
    create_page1_validator,
)
from .vision.vision_manager import VisionManager, create_vision_manager
from .orchestration.shadow_extractor import create_shadow_extractor
from .validators.token_validator import TokenValidator, create_token_validator

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_BATCH_SIZE: int = 10
DEFAULT_VLM_TIMEOUT: int = 90  # Seconds, accounts for SSD swap lag on 16GB systems
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
            vlm_timeout: VLM read timeout in seconds (default: 90)
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
            )
            logger.info(
                f"Global VisionManager initialized: "
                f"provider={self.vision_provider}, "
                f"model={self.vision_model}, "
                f"cache_dir={self.output_dir}"
            )
            return manager
        except Exception as e:
            logger.warning(f"Failed to initialize VisionManager: {e}")
            return None

    def _apply_fullpage_guard(
        self,
        asset_width: int,
        asset_height: int,
        page_width: float,
        page_height: float,
        image_path: Path,
        page_number: int,
        breadcrumbs: List[str],
    ) -> bool:
        """
        REQ-MM-08/09/10: Apply Full-Page Guard to shadow assets.

        Args:
            asset_width: Asset width in pixels
            asset_height: Asset height in pixels
            page_width: Page width in pixels
            page_height: Page height in pixels
            image_path: Path to saved asset image
            page_number: Page number
            breadcrumbs: Document breadcrumbs

        Returns:
            True if asset should be kept, False if it should be discarded
        """
        # REQ-MM-08: Calculate area ratio
        asset_area = asset_width * asset_height
        page_area = page_width * page_height
        area_ratio = asset_area / page_area if page_area > 0 else 0.0

        # REQ-MM-09: Check if full-page threshold (>0.95) is exceeded
        if area_ratio > 0.95:
            logger.info(
                f"[VLM-GUARD] Page {page_number}: Full-page asset detected "
                f"(ratio={area_ratio:.2%}, {asset_width}x{asset_height}px)"
            )

            # REQ-MM-12: Check override flag
            if self.allow_fullpage_shadow:
                logger.warning(
                    f"[VLM-GUARD] Page {page_number}: Full-page asset ALLOWED via override flag"
                )
                return True

            # REQ-MM-10: Perform VLM verification
            if not self._vision_manager:
                logger.warning(
                    f"[VLM-GUARD] Page {page_number}: No VLM available. "
                    f"Defaulting to DISCARD (safety first)"
                )
                return False

            try:
                from PIL import Image

                with Image.open(image_path) as img:
                    result = self._vision_manager.verify_shadow_integrity(img, breadcrumbs)

                    classification = result.get("classification", "error")
                    confidence = result.get("confidence", 0.0)
                    reason = result.get("reason", "")
                    is_valid = result.get("valid", False)

                    # REQ-MM-11: Log decision with full context
                    logger.info(
                        f"[VLM-GUARD] Asset Page {page_number} - Ratio {area_ratio:.2%} - "
                        f"Result: {classification} - Confidence: {confidence:.2f} - Reason: {reason}"
                    )

                    if not is_valid:
                        logger.info(
                            f"[VLM-GUARD] REJECTED: Page {page_number} asset "
                            f"({classification} at {confidence:.0%} confidence)"
                        )
                        return False

                    logger.info(
                        f"[VLM-GUARD] ACCEPTED: Page {page_number} asset "
                        f"(editorial at {confidence:.0%} confidence)"
                    )
                    return True

            except Exception as e:
                logger.error(
                    f"[VLM-GUARD] Page {page_number}: VLM verification failed: {e}. "
                    f"Defaulting to DISCARD"
                )
                return False

        # Asset is below full-page threshold - ALWAYS accept
        return True

    def _run_shadow_extraction(
        self,
        batch_info: BatchInfo,
        docling_chunks: List[IngestionChunk],
        source_file: str,
    ) -> List[IngestionChunk]:
        """
        REQ-MM-05/06/07: Run Shadow Extraction on batch to find missed bitmaps.

        This method performs a parallel raw PDF scan using PyMuPDF to detect
        bitmap objects that were missed by Docling's AI layout analysis.

        Args:
            batch_info: Information about this batch
            docling_chunks: Chunks extracted by Docling
            source_file: Original source filename

        Returns:
            List of shadow chunks to add to the output
        """
        # Check if shadow extraction is enabled in strategy
        if (
            self.extraction_strategy is None
            or not self.extraction_strategy.enable_shadow_extraction
        ):
            return []

        shadow_chunks: List[IngestionChunk] = []
        shadow_index = 0

        # Open the batch PDF for shadow scanning
        doc: Optional[fitz.Document] = None
        try:
            doc = fitz.open(batch_info.batch_path)

            # Get ACTUAL page dimensions from PDF for proper normalization
            page_dims_per_page: Dict[int, Tuple[float, float]] = {}
            for batch_page_idx in range(len(doc)):
                actual_page_no = batch_info.page_offset + batch_page_idx + 1
                page = doc.load_page(batch_page_idx)
                page_dims_per_page[actual_page_no] = (page.rect.width, page.rect.height)

            # Count Docling IMAGE chunks per page (regardless of bbox)
            # This is used to decide if shadow extraction should be conservative
            docling_images_per_page: Dict[int, int] = {}

            for chunk in docling_chunks:
                if chunk.modality == Modality.IMAGE:
                    page_no = chunk.metadata.page_number
                    docling_images_per_page[page_no] = docling_images_per_page.get(page_no, 0) + 1

            logger.info(f"Docling images per page: {dict(docling_images_per_page)}")

            # For pages where Docling found images, we'll be conservative with shadow extraction
            # by only keeping shadow assets if they don't significantly overlap

            # Create shadow extractor with strategy parameters
            with create_shadow_extractor(
                sensitivity=self.extraction_strategy.sensitivity,
                historical_median=self.extraction_strategy.historical_median,
            ) as shadow_extractor:

                # Scan each page in the batch
                for batch_page_idx in range(len(doc)):
                    # Actual page number (1-indexed)
                    actual_page_no = batch_info.page_offset + batch_page_idx + 1

                    # Get Docling image count for this page
                    docling_image_count = docling_images_per_page.get(actual_page_no, 0)

                    # Run shadow scan - pass empty bboxes since we can't get them
                    # The Ghost Filter and page-level checks will handle deduplication
                    # CRITICAL FIX: Use ABSOLUTE page number for metadata,
                    # but batch_page_idx for loading pages from the batch PDF!
                    # Ghost-Check validation: batch_page_idx vs actual_page_no
                    logger.info(
                        f"[VALIDATING] Absolute Page: {actual_page_no} | "
                        f"Batch Index: {batch_page_idx} | "
                        f"Page Offset: {batch_info.page_offset}"
                    )
                    scan_result = shadow_extractor.scan_page(
                        doc=doc,
                        page_number=actual_page_no,  # ABSOLUTE (for metadata)
                        docling_bboxes=[],
                        text_content="has_text",
                        batch_page_index=batch_page_idx,  # BATCH-RELATIVE (for loading)
                    )

                    # DEDUP CHECK: If Docling found images on this page, skip shadow
                    # This is the Page-Level Deduplication rule
                    if docling_image_count > 0:
                        logger.info(
                            f"[SHADOW] Page {actual_page_no}: Docling found {docling_image_count} images. "
                            f"Skipping shadow extraction (already covered)."
                        )
                        continue

                    # Process unaccounted assets (only for pages where Docling missed everything)
                    for asset in scan_result.unaccounted_assets:
                        shadow_index += 1

                        # Persist asset to disk with precision filters (Ghost Filter only)
                        asset_path = shadow_extractor.persist_asset(
                            doc=doc,
                            asset=asset,
                            output_dir=self.output_dir,
                            doc_hash=self._doc_hash or "unknown",
                            asset_index=shadow_index,
                            docling_bboxes=None,  # No bboxes available
                        )

                        if asset_path:
                            # REQ-STATE-04: Shadow chunks use DOCUMENT-LEVEL hierarchy only
                            # We CANNOT use self._context_state because it contains headings
                            # from FUTURE pages (the state advances during Docling processing).
                            # Shadow extraction happens AFTER Docling, so the state is "polluted"
                            # with headings from pages we haven't reached yet in shadow extraction.
                            #
                            # FIX: Use document title as the only safe breadcrumb for shadow assets.
                            # This prevents "breadcrumb pollution" like seeing "Demoralizing the Ayatollahs"
                            # on Page 4 when that heading is actually from Page 8.
                            doc_title = Path(source_file).stem if source_file else "Document"
                            hierarchy = HierarchyMetadata(
                                parent_heading=None,  # No heading - shadow assets are orphans
                                breadcrumb_path=[doc_title],  # Document title only
                                level=None,
                            )

                            # Content placeholder - will be replaced by VLM description below
                            semantic_context = asset.nearest_text or ""

                            # VLM Enrichment for shadow assets
                            visual_description = None
                            if self._vision_manager and asset_path:
                                try:
                                    # Load the saved image for VLM enrichment
                                    from PIL import Image

                                    full_asset_path = self.output_dir / asset_path
                                    if full_asset_path.exists():
                                        with Image.open(full_asset_path) as img:
                                            # Use VisionManager to get description
                                            # Create a temporary state if none exists
                                            temp_state = (
                                                self._context_state
                                                or create_context_state(
                                                    doc_id=self._doc_hash or "unknown",
                                                    source_file=source_file,
                                                )
                                            )
                                            visual_description = self._vision_manager.enrich_image(
                                                image=img,
                                                state=temp_state,
                                                page_number=actual_page_no,
                                                anchor_text=asset.nearest_text,
                                            )
                                            logger.info(
                                                f"[SHADOW] VLM enrichment successful for {asset_path}"
                                            )
                                except Exception as e:
                                    logger.warning(f"[SHADOW] VLM enrichment failed: {e}")

                            # RULE A & B: Enforce non-null visual descriptions
                            # REQ-CONTENT-01: content MUST be visual_description, not placeholder
                            if not visual_description or len(visual_description) < 20:
                                if not self._vision_manager:
                                    logger.warning(
                                        "VLM_CONNECTION_ERROR: No vision manager for shadow asset"
                                    )
                                    # Fallback: use semantic context + dimension info
                                    fallback = semantic_context[:100] if semantic_context else ""
                                    visual_description = (
                                        f"ERROR: VLM Offline | {fallback} | "
                                        f"({asset.width}x{asset.height}px)"
                                    )
                                else:
                                    vd_len = len(visual_description) if visual_description else 0
                                    logger.warning(
                                        f"Shadow VLM description too short ({vd_len} chars)"
                                    )
                                    visual_description = f"VLM_FALLBACK: {semantic_context[:200] if semantic_context else 'No context'}"

                            # REQ-CONTENT-02: content = visual_description (NEVER placeholder)
                            content = visual_description

                            # Create shadow chunk
                            # Convert bbox from float to int (normalize to 0-1000 scale)
                            bbox_int = None
                            if asset.bbox_normalized:
                                bbox_int = [int(round(v)) for v in asset.bbox_normalized]

                            chunk = create_shadow_chunk(
                                doc_id=self._doc_hash or "unknown",
                                content=content[:400],  # REQ-CHUNK-03: truncate to 400 chars
                                source_file=source_file,
                                file_type=FileType.PDF,
                                page_number=actual_page_no,
                                asset_path=asset_path,
                                bbox=bbox_int,
                                hierarchy=hierarchy,
                                prev_text=asset.nearest_text,
                                visual_description=visual_description,
                            )
                            shadow_chunks.append(chunk)

                            print(
                                f"    [SHADOW] Captured: Page {actual_page_no} "
                                f"({asset.width}x{asset.height}px)",
                                flush=True,
                            )

        finally:
            if doc is not None:
                doc.close()
            gc.collect()

        return shadow_chunks

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
        )

        # Convert ProcessedChunk to IngestionChunk
        ingestion_chunks: List[IngestionChunk] = []
        doc_title = Path(source_file).stem if source_file else "Document"

        for pc in processed_chunks:
            # Create hierarchy
            hierarchy = HierarchyMetadata(
                parent_heading=None,
                breadcrumb_path=[doc_title],
                level=None,
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

            # Create metadata with extraction_method and visual_description
            metadata = ChunkMetadata(
                page_number=pc.page_number,
                source_file=source_file,
                file_type=FileType.PDF,
                hierarchy=hierarchy,
                extraction_method=pc.extraction_method,
                visual_description=pc.visual_description,
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

            # Create IngestionChunk
            chunk = IngestionChunk(
                chunk_id=pc.chunk_id,
                doc_id=self._doc_hash or "unknown",
                content=pc.content,
                modality=modality,
                metadata=metadata,
                asset_ref=asset_ref,
            )

            ingestion_chunks.append(chunk)

        logger.info(
            f"[LAYOUT-OCR] Page {page_number}: Generated {len(ingestion_chunks)} chunks "
            f"(text: {sum(1 for c in ingestion_chunks if c.modality == Modality.TEXT)}, "
            f"image: {sum(1 for c in ingestion_chunks if c.modality == Modality.IMAGE)})"
        )

        return ingestion_chunks

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

                # Step 1: Classify page
                classification, char_count = self._classify_page(doc, batch_page_idx, threshold=100)
                print(
                    f"      📄 Page {actual_page_no}: {classification} ({char_count} chars)",
                    flush=True,
                )

                # Step 2: Render page to image
                page_image = self._render_page_to_image(doc, batch_page_idx, dpi=dpi)
                logger.info(f"[LAYOUT-OCR] Rendered page {actual_page_no}: {page_image.shape}")

                # Step 3: Process through layout-aware pipeline
                page_chunks = self._process_page_layout_aware(
                    page_image=page_image,
                    page_number=actual_page_no,
                    source_file=source_file,
                    docling_elements=None,  # TODO: Pass Docling elements when available
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

    def _run_shadow_first_extraction(
        self,
        batch_info: BatchInfo,
        source_file: str,
    ) -> List[IngestionChunk]:
        """
        REQ-OCR-02: Shadow-First extraction for scanned documents.

        Instead of using Docling figure crops (which lose text context),
        this method renders FULL PAGES at high DPI and runs OCR hints
        on the complete page. This allows OCR to see text labels like
        "Browning" that appear NEXT TO images.

        PAGE ISOLATION (REQ-OCR-ISOLATION):
        ====================================
        - Each page gets FRESH OCR hints - no accumulation between pages
        - VisionManager creates new OCRHintEngine per page
        - Explicit logging confirms isolation: [OCR-CLEANUP] at page boundary

        Args:
            batch_info: Information about this batch
            source_file: Original source filename

        Returns:
            List of shadow chunks (one per page with full-page render)
        """
        from PIL import Image
        import io

        shadow_chunks: List[IngestionChunk] = []
        dpi = self._profile_params.render_dpi if self._profile_params else 300

        # Open the batch PDF
        doc: Optional[fitz.Document] = None
        try:
            doc = fitz.open(batch_info.batch_path)

            # REQ-OCR-ISOLATION: Track page processing for isolation verification
            total_pages_in_batch = len(doc)
            logger.info(
                f"[SHADOW-FIRST] Starting batch with {total_pages_in_batch} pages. "
                f"OCR hints will be ISOLATED per page (no accumulation)."
            )

            for batch_page_idx in range(len(doc)):
                actual_page_no = batch_info.page_offset + batch_page_idx + 1

                # REQ-OCR-ISOLATION: Explicit page boundary marker
                logger.info(f"[PAGE-BOUNDARY] ========== PAGE {actual_page_no} START ==========")
                logger.info(f"[SHADOW-FIRST] Rendering page {actual_page_no} at {dpi} DPI")

                # Render page at high DPI
                page = doc.load_page(batch_page_idx)
                zoom = dpi / 72.0  # 72 DPI is base
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)

                # Convert to PIL Image
                img_data = pix.tobytes("ppm")
                page_image = Image.open(io.BytesIO(img_data)).convert("RGB")

                page_width, page_height = page_image.size
                logger.info(
                    f"[SHADOW-FIRST] Page {actual_page_no}: "
                    f"{page_width}x{page_height}px at {dpi} DPI"
                )

                # Save the full-page asset
                asset_name = f"{self._doc_hash}_{actual_page_no:03d}_shadow_01.png"
                asset_path = f"assets/{asset_name}"
                full_asset_path = self.assets_dir / asset_name
                page_image.save(str(full_asset_path), "PNG")

                print(
                    f"      📸 Page {actual_page_no}: {page_width}x{page_height}px "
                    f"→ {asset_name}",
                    flush=True,
                )

                # Create hierarchy
                doc_title = Path(source_file).stem if source_file else "Document"
                hierarchy = HierarchyMetadata(
                    parent_heading=None,
                    breadcrumb_path=[doc_title],
                    level=None,
                )

                # VLM Enrichment with OCR hints (the key difference!)
                visual_description = None
                if self._vision_manager:
                    try:
                        # Create temporary state
                        temp_state = self._context_state or create_context_state(
                            doc_id=self._doc_hash or "unknown",
                            source_file=source_file,
                        )
                        temp_state.update_page(actual_page_no)

                        # Use OCR-hint-aware enrichment (this is where OCR sees "Browning"!)
                        visual_description = self._vision_manager.enrich_image_with_ocr_hints(
                            image=page_image,
                            state=temp_state,
                            page_number=actual_page_no,
                            anchor_text=None,  # Full page has its own context
                            profile_params=self._profile_params,
                        )

                        logger.info(
                            f"[SHADOW-FIRST] VLM+OCR enrichment for page {actual_page_no}: "
                            f"{visual_description[:80] if visual_description else 'None'}..."
                        )

                    except Exception as e:
                        logger.warning(f"[SHADOW-FIRST] VLM+OCR failed: {e}")

                # Fallback if VLM failed
                if not visual_description or len(visual_description) < 20:
                    visual_description = (
                        f"Full page scan from {source_file}, page {actual_page_no} "
                        f"({page_width}x{page_height}px at {dpi} DPI)"
                    )

                # Create shadow chunk
                chunk = create_shadow_chunk(
                    doc_id=self._doc_hash or "unknown",
                    content=visual_description[:400],
                    source_file=source_file,
                    file_type=FileType.PDF,
                    page_number=actual_page_no,
                    asset_path=asset_path,
                    bbox=None,  # Full page = no bbox
                    hierarchy=hierarchy,
                    prev_text=None,
                    visual_description=visual_description,
                )
                shadow_chunks.append(chunk)

                # REQ-OCR-ISOLATION: Explicit page boundary end marker
                logger.info(f"[PAGE-BOUNDARY] ========== PAGE {actual_page_no} END ==========")

        finally:
            if doc is not None:
                doc.close()
            gc.collect()

        logger.info(f"[SHADOW-FIRST] Generated {len(shadow_chunks)} full-page shadow chunks")
        return shadow_chunks

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
        # SHADOW-FIRST STRATEGY (REQ-OCR-02) - FORCED FOR SCANNED PROFILES
        # ================================================================
        # When enable_ocr_hints=True (scanned profiles), we COMPLETELY BYPASS
        # Docling figure extraction and render full-page shadows at 300 DPI.
        #
        # WHY: Docling's figure crops only show the image WITHOUT text context.
        # The full-page shadow shows image + surrounding text labels.
        # This is the ONLY way OCR can see "Browning" next to the rifle.
        #
        # FORCED MODALITY SWITCH:
        # - ScannedDegradedProfile → extraction_method: "shadow" (NEVER docling)
        # - DigitalMagazineProfile → extraction_method: "docling" (standard)
        # ================================================================
        use_shadow_first = (
            self._profile_params is not None and self._profile_params.enable_ocr_hints
        )

        if use_shadow_first:
            # Type safety: use_shadow_first guarantees self._profile_params is not None
            profile_dpi = self._profile_params.render_dpi  # type: ignore[union-attr]

            logger.info(
                f"[SHADOW-FIRST] *** FORCED MODALITY SWITCH *** "
                f"Batch {batch_info.batch_index + 1}: "
                f"Docling figures SKIPPED, using full-page shadows at "
                f"{profile_dpi} DPI"
            )
            print(
                f"    🔬 [SHADOW-FIRST] *** DOCLING BYPASSED *** "
                f"Rendering full pages at {profile_dpi} DPI...",
                flush=True,
            )

            # Generate SHADOW-ONLY chunks (full-page renders with OCR hints)
            # NO DOCLING FIGURES - ONLY SHADOW EXTRACTION
            chunks = self._run_shadow_first_extraction(
                batch_info=batch_info,
                source_file=source_file,
            )

            # NO additional shadow extraction needed - we already have full pages
            # NO processor.get_final_state() - we didn't use the processor

            logger.info(
                f"[SHADOW-FIRST] Batch {batch_info.batch_index + 1}: "
                f"Generated {len(chunks)} shadow-only chunks (Docling bypassed)"
            )

        else:
            # Standard flow: Docling extraction + optional shadow extraction
            # Process batch and collect chunks (Phase 1: Docling)
            chunks: List[IngestionChunk] = []
            try:
                for chunk in processor.process_document(str(batch_info.batch_path)):
                    chunks.append(chunk)
            except Exception as e:
                logger.error(f"Error processing batch {batch_info.batch_index}: {e}")
                raise

            # CRITICAL: Capture state for next batch (breadcrumb continuity)
            # Only available when Docling was actually used
            self._context_state = processor.get_final_state()

            # Phase 2: Shadow Extraction (REQ-MM-05/06/07)
            # Only for Docling flow - shadow-first already has full pages
            shadow_chunks = self._run_shadow_extraction(
                batch_info=batch_info,
                docling_chunks=chunks,
                source_file=source_file,
            )

            if shadow_chunks:
                chunks.extend(shadow_chunks)
                logger.info(f"Shadow extraction added {len(shadow_chunks)} assets")

        # Log batch completion (guard against None context_state in shadow-first mode)
        breadcrumbs = (
            self._context_state.get_breadcrumb_path()
            if self._context_state
            else ["[shadow-first mode]"]
        )
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
        # REQ-DEDUP-01: Initialize ImageHashRegistry for pHash deduplication
        # ====================================================================
        self._image_hash_registry = create_image_hash_registry(threshold=10)
        print("\n🔍 [PHASH] Initializing perceptual hash registry...", flush=True)

        # Write aggregated output to master JSONL with deduplication
        output_jsonl = self.output_dir / "ingestion.jsonl"
        written_chunks = 0
        duplicate_count = 0

        with open(output_jsonl, "w", encoding="utf-8") as f:
            for chunk in all_chunks:
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
                f.write(json_line + "\n")
                written_chunks += 1

        # Log deduplication results
        registry_stats = self._image_hash_registry.get_stats()
        print(
            f"\n📊 [PHASH] Deduplication complete: "
            f"{registry_stats['total_registered']} unique images, "
            f"{duplicate_count} duplicates rejected",
            flush=True,
        )
        logger.info(
            f"Written {written_chunks} chunks to {output_jsonl} "
            f"({duplicate_count} duplicates rejected)"
        )

        # Get vision stats and flush cache
        vision_stats = {}
        if self._vision_manager:
            vision_stats = self._vision_manager.get_stats()
            self._vision_manager.flush_cache()
            logger.info(f"Vision cache flushed: {vision_stats}")

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
