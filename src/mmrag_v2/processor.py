"""
V2DocumentProcessor - Layout-Native Document Processing Engine
================================================================
ENGINE_USE: extraction engine v2.86.0 (Native Layout Analysis)

This module implements the V2.0 compliant document processor using the engine's
native layout analysis engine. It processes PDF, EPUB, HTML, DOCX documents
and produces validated ingestion.jsonl output per SRS Section 6.

REQ Compliance:
- REQ-PDF-01: Rich structure extraction via the engine
- REQ-PDF-02: Image/figure extraction with 10px padding (REQ-MM-01)
- REQ-PDF-03: Hybrid OCR fallback via Tesseract
- REQ-STATE: Hierarchical breadcrumb tracking via ContextStateV2
- REQ-MM-02: Asset naming [DocHash]_[Page]_[Type]_[Index].png
- REQ-CHUNK-03: VLM descriptions truncated to 400 chars

SRS Section 4: PDF Processing Pipeline
"The system MUST use a layout-native processing pipeline for structured
document extraction with layout analysis."

Author: Claude 4.5 Opus (Architect)
Date: 2025-12-28
Updated: 2025-12-29 (Vision Enrichment Layer)
"""

from __future__ import annotations

import math
import hashlib
import io
import json as _json
import logging
import re
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import fitz  # PyMuPDF for page rendering (OCR cascade)
import numpy as np
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
    IngestionMetadata,
    Modality,
    SemanticContext,
    SpatialMetadata,
    ChunkType,
    _generate_chunk_id,
    create_image_chunk,
    create_table_chunk,
    create_text_chunk,
    filter_chunk_factory_metadata,
    get_ocr_confidence_level,
    calculate_hierarchy_level,
    COORD_SCALE,
)
from .engines.pdf_extraction import PdfExtractionAdapter
from .engines.pdf_plan import PdfConversionPlan, build_pdf_conversion_plan
# V3.0 UIR contract (Charter §3.2): chunker call sites consume UIR types
# only — the underlying extraction-engine layout classes never reach
# this module. The `engines/pdf_extraction.py` module owns the engine
# boundary and produces (UIRChunk, ChunkType) pairs for the chunker to
# emit IngestionChunks against via `IngestionChunk.from_uir`.
from .universal.intermediate import (
    ConfidenceBreakdown as UIRConfidenceBreakdown,
    CoordinateFrame as UIRCoordinateFrame,
    Locator as UIRLocator,
    LocatorType as UIRLocatorType,
    UIRChunk,
)
from .version import __schema_version__ as SCHEMA_VERSION
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


# Charter §3.2 (ARCHITECTURE_V3_DRAFT_0.5.md): the chunker is engine-
# agnostic. Every byte of extraction-engine coupling (page-item iteration,
# label/prov inspection, HybridChunker invocation, UIRChunk → UIR
# conversion, dense-index / back-index detection) lives in the engine
# module `engines/pdf_extraction.py`. This module imports the engine's
# UIR-producing helpers under neutral names and never touches the
# underlying extraction-engine types directly.
# EPUB synthetic-pagination markers (non-extraction-engine; `_epub_to_html`
# prepends `<p>__MMRAG_EPUB_CH_NNNN__</p>` to each spine chapter so
# `_apply_epub_synthetic_pagination` can derive a per-chapter
# synthetic page_number).
_EPUB_CHAPTER_MARKER_PREFIX = "__MMRAG_EPUB_CH_"
_EPUB_CHAPTER_MARKER_RE = re.compile(r"__MMRAG_EPUB_CH_(\d+)__")


from .engines.pdf_extraction import (
    CROSS_PAGE_CONTINUED_MARKER,
    EXTRACTION_ENGINE_VERSION,
    EXTRACTION_METHOD_NATIVE,
    EXTRACTION_METHOD_TABLE_MARKDOWN,
    EXTRACTION_METHOD_TABLE_MARKDOWN_FALLBACK,
    looks_like_subtitle_continuation as _looks_like_subtitle_continuation_engine,  # re-export under legacy test alias
    sanitize_toc_index_text as _sanitize_toc_index_text,
    split_doc_chunk_text_by_page as _split_doc_chunk_text_by_page,
    classify_dense_index_pages as _classify_dense_index_pages,
    classify_dense_back_index_pages_by_source as _classify_dense_back_index_pages_by_source,
    dense_page_to_uir_chunk as _dense_page_to_uir_chunk,
    doc_chunk_first_label as _doc_chunk_first_label,
    doc_chunk_to_uir_chunks as _engine_chunk_to_uir,
    doc_chunk_validated_headings as _doc_chunk_validated_headings,
    extract_pdf_page_lines as _extract_pdf_page_lines,
    invoke_text_chunker as _invoke_text_chunker,
    item_label as _item_label,
    item_page_no as _item_page_no,
    item_prov_list as _item_prov_list,
    item_text as _item_text,
    looks_like_subtitle_continuation as _looks_like_subtitle_continuation,
    section_header_page_to_uir_chunk as _section_header_page_to_uir_chunk,
    text_items_for_page as _text_items_for_page,
    union_item_bboxes_for_uir as _union_item_bboxes_for_uir,
)




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

# Technical manuals (especially programming books) benefit from larger chunks,
# but overly large chunks hurt retrieval and increase mid-sentence cut risk.
# Token-limit enforcement (512 tokens) is handled later in BatchProcessor.
TECHNICAL_MANUAL_MAX_CHUNK_CHARS: int = 1200

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

TECHNICAL_KEYWORDS = [
    "algorithm",
    "architecture",
    "api",
    "class",
    "function",
    "method",
    "parameter",
    "configuration",
    "schema",
    "pipeline",
    "module",
    "implementation",
    "dataset",
    "model",
    "token",
]


# ============================================================================
# V2DocumentProcessor
# ============================================================================


class V2DocumentProcessor:
    """
    V2.0 compliant document processor using layout-native layout analysis.

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
        # V2.4 Intelligence Stack Metadata (Observability)
        intelligence_metadata: Optional[Dict[str, Any]] = None,
        conversion_plan: Optional[PdfConversionPlan] = None,
        # Force table serialization through VLM route (when available)
        force_table_vlm: bool = False,
        # Shared engine converter for batch processing (avoids model reload)
        external_converter: Optional[Any] = None,
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
            conversion_plan: Shared PDF conversion policy
            force_table_vlm: Force table image -> VLM markdown path (fallback to OCR/engine markdown if needed)
        """
        # Store extraction strategy for dynamic thresholds
        self._extraction_strategy = extraction_strategy

        # V3.0.0: Profile parameters for OCR configuration (shadow-first mode REMOVED)
        self._profile_params: Optional["ProfileParameters"] = None

        # V2.4: Store factory-safe intelligence metadata for chunk observability.
        # Structural/document-level flags are kept on the processor, but must
        # never be unpacked into create_*_chunk() kwargs.
        self._conversion_plan = conversion_plan
        if conversion_plan is not None:
            self.has_flat_text_corruption = conversion_plan.has_flat_text_corruption
            self.has_encoding_corruption = conversion_plan.has_encoding_corruption
            self.geometry_error_rate = conversion_plan.geometry_error_rate
            self._doc_total_pages = conversion_plan.total_pages
            self._doc_image_density = conversion_plan.image_density
            self._doc_avg_text_per_page = conversion_plan.avg_text_per_page
            self.needs_code_enrichment = conversion_plan.needs_code_enrichment
            self._extraction_route = conversion_plan.extraction_route
            self._hybrid_chunker_enabled = conversion_plan.hybrid_chunker_enabled
            self._max_chunker_input_chars = conversion_plan.max_chunker_input_chars
            self._max_chunker_per_element_chars = conversion_plan.max_chunker_per_element_chars
            self._allow_page_level_visuals = conversion_plan.allow_page_level_visuals
            self._drop_blank_assets = conversion_plan.drop_blank_assets
            self._quarantine_corrupted_chunks = conversion_plan.quarantine_corrupted_chunks
            self._suppress_layout_label_text = conversion_plan.suppress_layout_label_text
            self._intelligence_metadata = conversion_plan.chunk_factory_metadata()
        else:
            _raw_meta: Dict[str, Any] = dict(intelligence_metadata or {})
            self.has_flat_text_corruption = bool(
                _raw_meta.pop("has_flat_text_corruption", False)
            )
            self.has_encoding_corruption = bool(
                _raw_meta.pop("has_encoding_corruption", False)
            )
            self.geometry_error_rate = float(_raw_meta.pop("geometry_error_rate", 0.0))
            self._doc_total_pages = _raw_meta.pop("total_pages", None)
            self._doc_image_density = _raw_meta.pop("image_density", None)
            self._doc_avg_text_per_page = _raw_meta.pop("avg_text_per_page", None)
            # Workstream B: pop needs_code_enrichment before storing — it is used
            # only to configure the engine converter, not as a chunk field.
            self.needs_code_enrichment = bool(_raw_meta.pop("needs_code_enrichment", False))
            self._extraction_route = "native_digital"
            self._max_chunker_input_chars = 500_000
            self._max_chunker_per_element_chars = 100_000
            self._drop_blank_assets = True
            self._quarantine_corrupted_chunks = True
            self._intelligence_metadata = filter_chunk_factory_metadata(_raw_meta)
        if self._intelligence_metadata:
            logger.info(
                f"[OBSERVABILITY] Intelligence metadata loaded: "
                f"profile_type={self._intelligence_metadata.get('profile_type')}, "
                f"min_dims={self._intelligence_metadata.get('min_image_dims')}, "
                f"needs_code_enrichment={self.needs_code_enrichment}, "
                f"flat_corrupt={self.has_flat_text_corruption}, "
                f"encoding_corrupt={self.has_encoding_corruption}"
            )

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
        self._force_table_vlm = force_table_vlm
        self._final_state: Optional[ContextStateV2] = None

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir = self.output_dir / "assets"
        self.assets_dir.mkdir(parents=True, exist_ok=True)

        self.enable_ocr = enable_ocr
        self.ocr_engine = ocr_engine
        self.max_pages = max_pages

        # Initialize engine converter — reuse external converter if provided
        # to avoid reloading the picture classification model per batch.
        adapter_plan = conversion_plan or build_pdf_conversion_plan(
            enable_ocr=enable_ocr,
            ocr_engine=ocr_engine,
            force_table_vlm=force_table_vlm,
            needs_code_enrichment=self.needs_code_enrichment,
            has_encoding_corruption=self.has_encoding_corruption,
            has_flat_text_corruption=self.has_flat_text_corruption,
            geometry_error_rate=self.geometry_error_rate,
            total_pages=self._doc_total_pages or 0,
            image_density=self._doc_image_density or 0.0,
            avg_text_per_page=self._doc_avg_text_per_page or 0.0,
            **self._intelligence_metadata,
        )
        self._adapter = PdfExtractionAdapter(adapter_plan)
        if external_converter is not None:
            self._converter = external_converter
            logger.info("[PROCESSOR] Reusing shared engine converter (batch mode)")
        else:
            self._converter = self._adapter.get_converter()

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

        # Pass document domain to VisionManager for domain-aware prompt selection
        if self._vision_manager and self._intelligence_metadata:
            self._vision_manager.document_domain = self._intelligence_metadata.get(
                "document_domain", ""
            )

        # Initialize Semantic Text Refiner (v18.2) - disabled by default
        self._refiner = None

        # v2.9 Phase 1: per-document monotonic chunk counter feeding the
        # ``position`` argument of ``create_*_chunk`` factories so two chunks
        # with byte-identical (page, modality, content) get distinct chunk_ids.
        # Reset by callers at the start of each document (process_document /
        # process_to_jsonl_atomic). When this V2DocumentProcessor is used as a
        # shadow extractor by BatchProcessor, position uniqueness within the
        # shadow scope is sufficient — BatchProcessor owns its own counter for
        # the canonical chunks.
        self._chunk_position: int = 0

        # ================================================================
        # GEMINI AUDIT FIX: Initialize enhancement modules
        # ================================================================
        # FIX #2: SpatialPropagator - Universal bbox for ALL modalities
        self._spatial_propagator = create_spatial_propagator()

        # FIX #4: MagazineSectionDetector - Enriched breadcrumbs
        self._section_detector = create_section_detector()

        logger.info(
            f"ENGINE_USE: extraction engine v2.86.0 | V2DocumentProcessor initialized | "
            f"OCR: {ocr_engine if enable_ocr else 'disabled'} | "
            f"Vision: {vision_provider} | "
            f"Max pages: {max_pages if max_pages else 'ALL'} | "
            f"Enhancements: SpatialPropagator, MagazineSectionDetector"
        )

    def _extract_page_no_from_element(self, element: Any) -> int:
        """Best-effort page number extraction from element provenance."""
        if hasattr(element, "prov") and element.prov:
            prov = element.prov[0] if isinstance(element.prov, list) else element.prov
            if hasattr(prov, "page_no") and prov.page_no is not None:
                return int(prov.page_no)
            if hasattr(prov, "page") and prov.page is not None:
                return int(prov.page)
        return 1

    def _extract_bbox_from_element(self, element: Any) -> Optional[Tuple[float, float, float, float]]:
        """Best-effort bbox extraction as (x0, y0, x1, y1)."""
        if hasattr(element, "prov") and element.prov:
            prov = element.prov[0] if isinstance(element.prov, list) else element.prov
            bbox_obj = getattr(prov, "bbox", None)
            if bbox_obj is None:
                return None

            if hasattr(bbox_obj, "l"):
                x0, x1 = min(float(bbox_obj.l), float(bbox_obj.r)), max(
                    float(bbox_obj.l), float(bbox_obj.r)
                )
                y0, y1 = min(float(bbox_obj.t), float(bbox_obj.b)), max(
                    float(bbox_obj.t), float(bbox_obj.b)
                )
                return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None

            if hasattr(bbox_obj, "as_tuple"):
                raw = bbox_obj.as_tuple()
                if raw and len(raw) >= 4:
                    x0, x1 = min(float(raw[0]), float(raw[2])), max(float(raw[0]), float(raw[2]))
                    y0, y1 = min(float(raw[1]), float(raw[3])), max(float(raw[1]), float(raw[3]))
                    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None

        return None

    def _order_items_for_page(self, records: List[Dict[str, Any]], page_width: float) -> List[Dict[str, Any]]:
        """
        Order page records in reading order.

        Strategy:
        - Detect two-column layouts using bbox x-center clustering.
        - Sort within column by y_min then x_min.
        - Keep elements without bbox at the end in original order.
        """
        with_bbox = [r for r in records if r["bbox"] is not None]
        if not with_bbox:
            return sorted(records, key=lambda r: r["orig_index"])

        two_column = False
        split_x = None

        # Only try split when we have enough evidence.
        if len(with_bbox) >= 2:
            x_centers = sorted((r["bbox"][0] + r["bbox"][2]) / 2.0 for r in with_bbox)
            mid = len(x_centers) // 2
            if mid > 0:
                split_x = (x_centers[mid - 1] + x_centers[mid]) / 2.0
                left = [r for r in with_bbox if ((r["bbox"][0] + r["bbox"][2]) / 2.0) <= split_x]
                right = [r for r in with_bbox if ((r["bbox"][0] + r["bbox"][2]) / 2.0) > split_x]

                if len(left) >= 1 and len(right) >= 1:
                    left_max = max(r["bbox"][2] for r in left)
                    right_min = min(r["bbox"][0] for r in right)
                    gap = right_min - left_max
                    inferred_width = max(
                        max(r["bbox"][2] for r in with_bbox) - min(r["bbox"][0] for r in with_bbox),
                        1.0,
                    )
                    width_ref = page_width if page_width > 1 else inferred_width
                    two_column = gap > width_ref * 0.04

        def _sort_key(rec: Dict[str, Any]) -> Tuple[int, float, float, int]:
            bbox = rec["bbox"]
            if bbox is None:
                return (99, math.inf, math.inf, rec["orig_index"])

            x0, y0, x1, _ = bbox
            if two_column and split_x is not None:
                x_center = (x0 + x1) / 2.0
                col = 0 if x_center <= split_x else 1
            else:
                col = 0

            return (col, y0, x0, rec["orig_index"])

        return sorted(records, key=_sort_key)

    def _get_ordered_doc_items(self, doc: Any) -> List[Tuple[Any, Any]]:
        """Return doc items ordered by page and reading order within page."""
        raw_items = list(doc.iterate_items())
        if not raw_items:
            return []

        page_widths: Dict[int, float] = {}
        if hasattr(doc, "pages") and doc.pages:
            pages_iter = doc.pages.values() if isinstance(doc.pages, dict) else doc.pages
            for page in pages_iter:
                pg_no = getattr(page, "page_no", 1) or 1
                width = float(getattr(page, "width", 0.0) or 0.0)
                page_widths[int(pg_no)] = width

        by_page: Dict[int, List[Dict[str, Any]]] = {}
        for orig_index, item_tuple in enumerate(raw_items):
            element, _ = item_tuple
            page_no = self._extract_page_no_from_element(element)
            bbox = self._extract_bbox_from_element(element)
            by_page.setdefault(page_no, []).append(
                {
                    "item": item_tuple,
                    "orig_index": orig_index,
                    "bbox": bbox,
                }
            )

        ordered_items: List[Tuple[Any, Any]] = []
        for page_no in sorted(by_page.keys()):
            page_records = by_page[page_no]
            page_width = page_widths.get(page_no, 0.0)
            ordered_page_records = self._order_items_for_page(page_records, page_width)
            ordered_items.extend(r["item"] for r in ordered_page_records)

        return ordered_items

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

    def _epub_to_html(self, epub_path: Path) -> Optional[Path]:
        """Extract EPUB spine content into a single HTML file with
        chapter-boundary markers.

        v2.10 Phase 7 (`KI_EPUB_EXTRACTION_LANE_REWRITE`): walks
        ``book.spine`` (the canonical reading order) instead of
        ``get_items_of_type`` (which returns manifest order). Prepends a
        marker paragraph ``__MMRAG_EPUB_CH_NNNN__`` to each non-empty
        chapter so the post-extraction pass (``_apply_epub_synthetic_pagination``)
        can derive a per-chunk synthetic ``page_number``. Stashes the
        chapter count on ``self._epub_chapter_count`` and the spine
        chapter names on ``self._epub_spine_chapter_names`` for the
        post-process pass and diagnostics.

        Returns:
            Path to a temporary HTML file, or None on failure.
        """
        import tempfile

        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup
        except ImportError as e:
            logger.warning(f"[EPUB] Missing dependency for EPUB: {e}")
            return None

        try:
            book = epub.read_epub(str(epub_path), options={"ignore_ncx": True})
            parts: list[str] = []
            chapter_names: list[str] = []
            chapter_count = 0

            for sp_entry in book.spine:
                idref = sp_entry[0] if isinstance(sp_entry, tuple) else sp_entry
                item = book.get_item_with_id(idref)
                if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
                    continue
                soup = BeautifulSoup(item.get_content(), "html.parser")
                body = soup.find("body")
                if body is None:
                    continue
                for tag in body.find_all(["svg", "script", "style"]):
                    tag.decompose()
                inner = body.decode_contents()
                if not inner.strip():
                    continue
                chapter_count += 1
                chapter_names.append(item.get_name() or f"chapter_{chapter_count}")
                marker_para = f"<p>{_EPUB_CHAPTER_MARKER_PREFIX}{chapter_count:04d}__</p>"
                parts.append(marker_para + "\n" + inner)

            if not parts:
                self._epub_chapter_count = 0
                self._epub_spine_chapter_names = []
                return None

            html = (
                "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                f"<title>{epub_path.stem}</title></head><body>\n"
                + "\n<hr/>\n".join(parts)
                + "\n</body></html>"
            )

            tmp = tempfile.NamedTemporaryFile(
                suffix=".html", prefix="epub_", delete=False, mode="w", encoding="utf-8"
            )
            tmp.write(html)
            tmp.close()
            self._epub_chapter_count = chapter_count
            self._epub_spine_chapter_names = chapter_names
            logger.info(
                f"[EPUB] Extracted {chapter_count} spine chapters, {len(html)} bytes; "
                f"chapter markers embedded for synthetic pagination"
            )
            return Path(tmp.name)
        except Exception as exc:
            logger.warning(f"[EPUB] Failed to convert EPUB to HTML: {exc}")
            return None

    def _apply_epub_synthetic_pagination(
        self,
        chunks: List[IngestionChunk],
        chapter_count: int,
    ) -> Generator[IngestionChunk, None, None]:
        """v2.10 Phase 7 (`KI_EPUB_EXTRACTION_LANE_REWRITE`): rewrite EPUB
        chunks with deterministic per-chapter synthetic ``page_number`` and
        the documented full-page bbox sentinel ``[0, 0, 1000, 1000]``.

        EPUB has no native pagination; ``_epub_to_html`` inserts a
        per-chapter marker (``__MMRAG_EPUB_CH_NNNN__``) at the head of each
        spine chapter. This pass walks chunks in emission order, switches
        the running chapter index when a marker appears in chunk content,
        strips the marker, and assigns:

            page_number = chapter_1based * 1000 + position_in_chapter // 5
            bbox = [0, 0, 1000, 1000]
            extraction_method = "epub_html"

        Pre-marker fallback: the HTML parser sometimes drops the
        first one or more marker paragraphs (e.g. KI EPUB's titlepage +
        colophon are stripped before the first surviving marker reaches
        the chunk stream). Chunks that arrive before any marker are
        buffered and back-attributed to chapter ``(first_marker - 1)``
        once the first marker is seen — that's the chapter directly
        preceding the first surviving marker in spine order. When NO
        marker survives (catastrophic), the buffer flushes to chapter 1.

        Chunks that become empty after marker removal are dropped (those
        are the marker-only paragraphs the engine emits as standalone chunks).
        ``chunk_id`` is regenerated with the new ``page_number`` and a
        monotonic global position so the v2.9 chunk_id position-component
        contract (no duplicate chunk_ids within a document) holds across
        the rewrite.
        """
        marker_re = _EPUB_CHAPTER_MARKER_RE
        full_page_bbox = [0, 0, COORD_SCALE, COORD_SCALE]

        def _rewrite_chunk(
            chunk: IngestionChunk,
            cleaned: str,
            new_page: int,
            global_pos: int,
        ) -> IngestionChunk:
            chunk.content = cleaned
            meta = chunk.metadata
            if meta is not None:
                meta.page_number = new_page
                meta.extraction_method = "epub_html"
                if meta.refined_content:
                    refined_cleaned = marker_re.sub("", meta.refined_content).strip()
                    meta.refined_content = refined_cleaned or cleaned
                else:
                    meta.refined_content = cleaned
                meta.spatial = SpatialMetadata(
                    bbox=list(full_page_bbox),
                    page_width=COORD_SCALE,
                    page_height=COORD_SCALE,
                )
            modality_str = (
                chunk.modality.value if hasattr(chunk.modality, "value") else "text"
            )
            chunk.chunk_id = _generate_chunk_id(
                chunk.doc_id,
                cleaned,
                new_page,
                modality_str,
                global_pos,
            )
            return chunk

        pre_marker_buffer: List[Tuple[IngestionChunk, str]] = []
        current_chapter: Optional[int] = None
        chapter_position = 0
        global_position = 0
        emitted = 0
        # Per-synthetic-page seen set: drop byte-equal content within the
        # same synthetic page so qa_universal_invariants'
        # ``within_page_text_dupe_excess`` check does not fire on
        # producer-side short-text repeats ("Voorbeeldprompt:", "of",
        # etc. emitted twice within a 5-chunk window).
        seen_per_page: Dict[int, set[str]] = {}
        dedup_skipped = 0

        def _emit_for_chapter(
            chunk: IngestionChunk,
            cleaned: str,
            chapter: int,
            chapter_pos: int,
        ) -> Tuple[Optional[IngestionChunk], int]:
            nonlocal global_position, emitted, dedup_skipped
            new_page = chapter * 1000 + (chapter_pos // 5)
            page_seen = seen_per_page.setdefault(new_page, set())
            if cleaned in page_seen:
                dedup_skipped += 1
                return None, chapter_pos
            page_seen.add(cleaned)
            global_position += 1
            emitted += 1
            return (
                _rewrite_chunk(chunk, cleaned, new_page, global_position),
                chapter_pos + 1,
            )

        for chunk in chunks:
            text = chunk.content or ""
            markers = list(marker_re.finditer(text))
            cleaned = marker_re.sub("", text).strip() if markers else text.strip()
            if not cleaned and not markers:
                continue
            if markers:
                first_marker_idx = int(markers[0].group(1))
                if current_chapter is None:
                    # First marker we have seen. Back-attribute the
                    # pre-marker buffer to the chapter preceding this
                    # marker (e.g. KI: marker 4 is first; buffered
                    # chunks 0..32 belong to chapter 3 = cW.xhtml TOC).
                    flush_chapter = max(1, first_marker_idx - 1)
                    flush_pos = 0
                    for buf_chunk, buf_cleaned in pre_marker_buffer:
                        emitted_chunk, flush_pos = _emit_for_chapter(
                            buf_chunk, buf_cleaned, flush_chapter, flush_pos
                        )
                        if emitted_chunk is not None:
                            yield emitted_chunk
                    pre_marker_buffer.clear()
                    current_chapter = first_marker_idx
                    chapter_position = 0
                elif first_marker_idx != current_chapter:
                    current_chapter = first_marker_idx
                    chapter_position = 0
            if not cleaned:
                # Marker-only chunk; chapter advanced above, drop the chunk.
                continue
            if current_chapter is None:
                # No marker seen yet; defer chapter assignment.
                pre_marker_buffer.append((chunk, cleaned))
                continue
            emitted_chunk, chapter_position = _emit_for_chapter(
                chunk, cleaned, current_chapter, chapter_position
            )
            if emitted_chunk is not None:
                yield emitted_chunk

        # Catastrophic: no marker ever seen → flush buffer to chapter 1.
        if pre_marker_buffer and current_chapter is None:
            flush_pos = 0
            for buf_chunk, buf_cleaned in pre_marker_buffer:
                emitted_chunk, flush_pos = _emit_for_chapter(
                    buf_chunk, buf_cleaned, 1, flush_pos
                )
                if emitted_chunk is not None:
                    yield emitted_chunk
            pre_marker_buffer.clear()

        logger.info(
            f"[EPUB] Synthetic pagination applied: chapters={chapter_count} "
            f"emitted_chunks={emitted} dropped_chunks={len(chunks) - emitted} "
            f"per_page_dedup_skipped={dedup_skipped}"
        )

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

        # Code blocks can look "non-heading-like" (symbols/indentation). Never treat
        # code-like content as noise; chunking should preserve it for technical_manual RAG.
        if self._looks_like_code(text):
            return False

        # Body text must not be validated as a heading. The heading validator
        # intentionally rejects long/multi-sentence strings; using it here
        # drops legitimate element-by-element fallback text from TOC/index
        # pages when HybridChunker times out on dense engine cells.
        if len(text) < 2:
            logger.debug(f"[DENOISE] Rejected content (too short): '{text}'")
            return True

        if re.fullmatch(r"(?:page\s*)?\d+", text, flags=re.IGNORECASE):
            logger.debug(f"[DENOISE] Rejected content (page number): '{text}'")
            return True

        if not any(c.isalpha() for c in text):
            logger.debug(f"[DENOISE] Rejected content (no alphabetic text): '{text}'")
            return True

        return False

    def _looks_like_code(self, text: str) -> bool:
        """
        Heuristic code detector for programming books/manuals.

        The extraction engine + PyMuPDF often preserve indentation/newlines for code blocks; we prefer
        to keep those blocks intact (and avoid sentence splitting / denoising).
        """
        if not text:
            return False

        t = text.strip("\n")
        if "```" in t:
            return True

        lines = [ln for ln in t.splitlines() if ln.strip()]
        if len(lines) < 2:
            # Single-line code (common in OCR layers / inline examples).
            if ">>>" in t or t.lstrip().startswith("..."):
                return True
            if re.search(r"^\s*(def|class|import|from)\b", t):
                return True
            # Avoid false positives on English "from ..."; require Python import syntax.
            if re.search(r"\bfrom\s+[A-Za-z_][\w\.]*\s+import\b", t):
                return True
            if re.search(r"\bimport\s+[A-Za-z_][\w\.]*(\s*,\s*[A-Za-z_][\w\.]*)*", t):
                return True
            return False

        indented = sum(1 for ln in lines if ln.startswith(("    ", "\t")))
        repl_prompt = sum(
            1 for ln in lines if ln.lstrip().startswith((">>>", "..."))
        )
        if indented / max(len(lines), 1) >= 0.3:
            return True
        if repl_prompt >= 1:
            return True

        # Language-agnostic + Python-heavy signals
        code_kw = re.search(
            r"^\s*(def|class|import|from|return|yield|async\s+def|await|try|except|with|for|while|if|elif|else)\b",
            t,
            flags=re.MULTILINE,
        )
        if code_kw:
            return True

        # Symbol density: code tends to include many operators/brackets.
        symbols = sum(c in "{}[]()<>:=/*+-#.,;\\|" for c in t)
        alnum = sum(c.isalnum() for c in t)
        if symbols >= 20 and symbols > alnum * 0.10:
            return True

        return False

    def _extract_heading_level(self, label: str) -> Optional[int]:
        """Extract heading level from engine label."""
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
            element: engine element (may have .image attribute)
            bbox_normalized: Bounding box in 0-1000 normalized scale (or None)
            page_images: Dict of page number → rendered PIL Image
            page_no: Batch page number (for page_images lookup)
            page_dims: (page_width, page_height) in PDF points

        Returns:
            PIL Image if extracted successfully, None otherwise
        """
        extracted_image: Optional[Image.Image] = None

        try:
            # PRIORITY 1: Use the engine-extracted image when available.
            #
            # This yields "clean" assets (not page crops) and avoids the common failure mode
            # where bbox crops cut off chart axes / labels / code text in technical books.
            if hasattr(element, "image") and element.image:
                img_data = element.image
                if hasattr(img_data, "pil_image") and img_data.pil_image is not None:
                    extracted_image = img_data.pil_image
                    logger.debug("Extracted image from element.image (native)")
                elif isinstance(img_data, bytes):
                    extracted_image = Image.open(BytesIO(img_data))
                    logger.debug("Extracted image from element.image (bytes)")
                elif isinstance(img_data, Image.Image):
                    extracted_image = img_data
                    logger.debug("Extracted image from element.image (PIL)")

                if extracted_image is not None:
                    # REQ-MM-01: ensure a small safety padding on the asset itself.
                    # (BBox padding is applied separately to metadata.)
                    try:
                        from PIL import ImageOps

                        extracted_image = ImageOps.expand(extracted_image, border=10, fill="white")
                    except Exception:
                        pass
                    return extracted_image

            # PRIORITY 2: Crop from rendered page image using bbox (fallback).
            # This preserves consistent padding when native extraction isn't available.
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
                if page_width <= 0 or page_height <= 0:
                    logger.warning(
                        f"Invalid page_dims for scaling: {page_dims}. Falling back to 1.0 scale."
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

                img_width, img_height = page_img.size
                x0 = max(0, min(crop_box[0], img_width - 1))
                y0 = max(0, min(crop_box[1], img_height - 1))
                x1 = max(x0 + 1, min(crop_box[2], img_width))
                y1 = max(y0 + 1, min(crop_box[3], img_height))

                if x1 <= x0 or y1 <= y0:
                    logger.warning(
                        "Invalid crop coordinates after denormalization: "
                        f"bbox_normalized={bbox_normalized} → crop=({x0},{y0},{x1},{y1})"
                    )
                    return None

                crop_tuple = (x0, y0, x1, y1)
                try:
                    from .utils.image_trim import expand_crop_box_if_clipped

                    expanded_crop = expand_crop_box_if_clipped(page_img, crop_tuple)
                    if expanded_crop != crop_tuple:
                        logger.debug(
                            f"[CROP-EXPAND] Page {page_no}: expanded crop to avoid clipping "
                            f"(from={crop_tuple} to={expanded_crop})"
                        )
                        crop_tuple = expanded_crop
                except Exception as expand_err:
                    logger.debug(f"[CROP-EXPAND] Skipped due to error: {expand_err}")

                extracted_image = page_img.crop(crop_tuple)

                logger.debug(
                    f"Extracted image via crop: bbox_normalized={bbox_normalized} → "
                    f"abs_bbox={[f'{v:.1f}' for v in abs_bbox]} → "
                    f"crop=({crop_tuple[0]},{crop_tuple[1]},{crop_tuple[2]},{crop_tuple[3]}) → "
                    f"size={extracted_image.size}"
                )
                return extracted_image

            # PRIORITY 3: If bbox exists but we can't crop, fail fast (invariant).
            # NOTE: If no page image is available, we cannot apply consistent padding.
            if bbox_normalized and page_no not in page_images:
                # ✅ IRON-06: FAIL FAST on missing page buffer
                error_msg = (
                    f"[IRON-06 VIOLATION] Page image buffer missing for page {page_no}. "
                    f"Cannot apply REQ-MM-01 10px padding. Processing HALTED. "
                    f"This should NOT happen with generate_page_images=True. "
                    f"Check Engine configuration and batch_processor page_images dict."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)

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
            element: engine element with potential .image attribute
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
                # Quality enhancement: trim large near-white margins from page-render crops.
                # This is especially common in digital PDFs where vector diagrams are rasterized
                # into cropped page images with lots of whitespace.
                try:
                    from .utils.image_trim import trim_white_margins

                    trim_res = trim_white_margins(saved_image)
                    if trim_res.trimmed:
                        saved_image = trim_res.image
                        logger.debug(
                            f"[ASSET-TRIM] Page {page_no}: trimmed margins for {Path(asset_path).name} "
                            f"(bbox={trim_res.bbox}, size={saved_image.size})"
                        )
                except Exception as trim_err:
                    # Trimming is best-effort only.
                    logger.debug(f"[ASSET-TRIM] Skipped due to error: {trim_err}")

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
            if self.enable_ocr and profile_params and profile_params.enable_ocr_hints:
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

    @staticmethod
    def _is_table_placeholder_text(text: Optional[str], page_no: int) -> bool:
        """Detect placeholder table content that is unusable for retrieval."""
        if not text:
            return True
        t = text.strip()
        if not t:
            return True
        if re.match(r"^\[table on page \d+\]$", t, flags=re.IGNORECASE):
            return True
        if re.match(r"^\[table extraction unavailable on page \d+\]$", t, flags=re.IGNORECASE):
            return True
        if t.lower() in {"[table]", "table", f"[table on page {page_no}]"}:
            return True
        return False

    @staticmethod
    def _is_markdown_table(text: str) -> bool:
        """Check whether text is a structured markdown table."""
        t = (text or "").strip()
        if not t:
            return False
        lines = [ln for ln in t.splitlines() if ln.strip()]
        if len(lines) < 2:
            return False
        if "|" not in lines[0]:
            return False
        return any(re.search(r"\|\s*-{2,}", ln) for ln in lines[1:3])

    def _is_valid_vlm_table_markdown(self, markdown: str, page_no: int) -> bool:
        """Post-hoc quality gate for VLM table markdown before acceptance."""
        text = (markdown or "").strip()
        if not text or not self._is_markdown_table(text):
            return False

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) < 3:
            return False

        try:
            row_cols = [max(0, ln.count("|") - 1) for ln in lines]
            if min(row_cols) < 2:
                return False
            if max(row_cols) - min(row_cols) > 2:
                logger.warning(
                    f"[TABLE-QUALITY] Page {page_no}: inconsistent VLM markdown columns "
                    f"(min={min(row_cols)}, max={max(row_cols)})"
                )
                return False
        except Exception:
            return False

        # Reject prose-like cells: a strong hallucination signal for table extraction.
        sentence_punct = re.compile(r"[.!?]")
        for ln in lines:
            if "|" not in ln:
                continue
            cells = [c.strip().strip("`*_ ") for c in ln.split("|")[1:-1]]
            for cell in cells:
                if len(cell) > 60 and len(sentence_punct.findall(cell)) >= 2:
                    logger.warning(
                        f"[TABLE-QUALITY] Page {page_no}: rejected prose-like cell in VLM table"
                    )
                    return False

        lowered = text.lower()
        suspicious = [
            "lorem ipsum",
            "placeholder",
            "example row",
            "sample data",
            "generated content",
            "the matrix",
            "movie",
            "findwindowexa",
            "was passing",
            "htrm",
            "pulse length",
        ]
        if any(tok in lowered for tok in suspicious):
            logger.warning(f"[TABLE-QUALITY] Page {page_no}: rejected suspicious VLM table content")
            return False

        return True

    def _is_unstructured_table_text(self, text: str, page_no: int) -> bool:
        """
        Detect table "text soup" that should trigger image-based VLM/OCR fallback.

        Criteria are intentionally conservative: if it's already markdown, keep it.
        """
        t = (text or "").strip()
        if self._is_table_placeholder_text(t, page_no):
            return True
        if self._is_markdown_table(t):
            return False
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        if len(lines) < 2:
            return True
        # Number-heavy line lists usually indicate degraded TOC-like dumps.
        numbery = sum(1 for ln in lines if re.match(r"^\d+(\.\d+)?\b", ln))
        shortish = sum(1 for ln in lines if len(ln) <= 120)
        if numbery >= max(2, len(lines) // 3):
            return True
        if shortish >= max(3, int(len(lines) * 0.8)):
            return True
        return False

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove markdown code fences from model output."""
        cleaned = (text or "").strip()
        m = re.match(r"^```(?:markdown|md|text)?\s*([\s\S]*?)\s*```$", cleaned, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return cleaned

    def _table_text_to_markdown(self, text: str) -> str:
        """
        Convert table-like text to markdown.

        If the text already appears to be markdown, return it as-is.
        Otherwise fallback to a one-column markdown table to preserve content.
        """
        raw = self._strip_markdown_fences(text)
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if not lines:
            return ""

        # Already markdown-like table.
        if sum(1 for ln in lines if "|" in ln) >= 2:
            return "\n".join(lines)

        # Try simple columnization for TSV / multi-space aligned rows.
        parsed_rows: List[List[str]] = []
        for ln in lines:
            cols = [c.strip() for c in re.split(r"\t+|\s{2,}", ln) if c.strip()]
            if len(cols) >= 2:
                parsed_rows.append(cols)

        if parsed_rows:
            max_cols = max(len(r) for r in parsed_rows)
            header = parsed_rows[0] + [""] * (max_cols - len(parsed_rows[0]))
            md_lines = [
                "| " + " | ".join(header) + " |",
                "| " + " | ".join(["---"] * max_cols) + " |",
            ]
            for row in parsed_rows[1:]:
                row_padded = row + [""] * (max_cols - len(row))
                md_lines.append("| " + " | ".join(row_padded) + " |")
            return "\n".join(md_lines)

        # Single-column fallback: still serializes all table text for retrieval.
        md_lines = ["| Table Content |", "| --- |"]
        for ln in lines:
            md_lines.append("| " + ln.replace("|", "\\|") + " |")
        return "\n".join(md_lines)

    def _extract_table_text(self, element: Any, page_no: int, doc: Any = None) -> str:
        """
        Best-effort extraction of richer table text from engine table elements.

        Some engine table elements expose markdown/structured export methods while
        `element.text` may contain only placeholders. This helper probes common
        export APIs and falls back safely.
        """
        method_names = (
            "export_to_markdown",
            "to_markdown",
            "as_markdown",
            "to_md",
        )

        for method_name in method_names:
            method = getattr(element, method_name, None)
            if not callable(method):
                continue
            try:
                # New engine API requires doc argument
                if doc is not None:
                    value = method(doc=doc)
                else:
                    value = method()
            except TypeError:
                # Fallback for older API without doc parameter
                try:
                    value = method()
                except Exception:
                    continue
            except Exception as e:
                logger.debug(
                    f"[TABLE-DOC-LING] Page {page_no}: {method_name} failed: {e}"
                )
                continue

            text = (value or "").strip() if isinstance(value, str) else ""
            if text and not self._is_table_placeholder_text(text, page_no):
                logger.info(
                    f"[TABLE-DOC-LING] Page {page_no}: recovered table text via {method_name}"
                )
                return text

        return ""

    def _extract_table_markdown_with_ocr(
        self,
        table_image: Optional[Image.Image],
        page_no: int,
    ) -> str:
        """OCR-based fallback for table chunks when engine table text is missing."""
        if table_image is None:
            return ""
        if not self.enable_ocr:
            return ""

        try:
            ocr_engine = getattr(self, "_ocr_engine", None)
            if ocr_engine is None:
                ocr_engine = EnhancedOCREngine(
                    confidence_threshold=0.4,
                    enable_tesseract=True,
                    enable_doctr=True,
                )
                self._ocr_engine = ocr_engine

            np_img = np.array(table_image.convert("RGB"))
            ocr_result = ocr_engine.process_page(np_img)
            ocr_text = (ocr_result.text or "").strip()
            if len(ocr_text) < 10:
                return ""

            markdown = self._table_text_to_markdown(ocr_text)
            if markdown:
                logger.info(
                    f"[TABLE-FALLBACK] Page {page_no}: OCR recovered table text "
                    f"({len(ocr_text)} chars, layer={ocr_result.layer_used.value})"
                )
            return markdown
        except Exception as e:
            logger.debug(f"[TABLE-FALLBACK] Page {page_no}: OCR table fallback failed: {e}")
            return ""

    def _extract_table_markdown_with_vlm(
        self,
        table_image: Optional[Image.Image],
        page_no: int,
    ) -> str:
        """VLM fallback for table markdown extraction when OCR/engine markdown are insufficient."""
        if table_image is None or self._vision_manager is None:
            return ""

        try:
            if hasattr(self._vision_manager, "extract_table_markdown"):
                markdown = self._vision_manager.extract_table_markdown(
                    image=table_image,
                    page_number=page_no,
                )
            else:
                markdown = ""
            markdown = self._strip_markdown_fences(markdown or "")
            if not markdown:
                return ""
            lowered = markdown.strip().lower()
            # Accept markdown tables that contain some [unreadable] cells; reject only
            # fully-empty sentinel responses.
            if lowered in {"unreadable", "[unreadable]", "n/a", "none", "unable to read"}:
                return ""
            if not self._is_valid_vlm_table_markdown(markdown, page_no):
                logger.warning(
                    f"[TABLE-QUALITY] Page {page_no}: VLM table markdown failed quality gate, "
                    "falling back to OCR/engine markdown"
                )
                return ""
            logger.info(
                f"[TABLE-FALLBACK] Page {page_no}: VLM recovered table markdown "
                f"({len(markdown)} chars)"
            )
            return markdown
        except Exception as e:
            logger.debug(f"[TABLE-FALLBACK] Page {page_no}: VLM table fallback failed: {e}")
            return ""

    @staticmethod
    def _shadow_image_meets_threshold(
        img_width: float,
        img_height: float,
        area_ratio: float,
        *,
        page_needs_coverage: bool,
    ) -> bool:
        """Return True if a candidate shadow image meets the
        SHADOW-EXTRACTION emission threshold.

        Standard threshold (REQ-MM-06): 300x300 OR area_ratio >= 40 %.

        Page-coverage relaxation (PLAN_V2.10 Phase 3): when the page
        has NO prior chunks, the size floor drops to 200x200 so mid-
        size chapter-intro diagrams still emit and avoid landing as
        MISSING_PAGES at the strict gate. The area floor stays at
        40 % for that lane — thin banners and decorative dividers
        remain filtered out.
        """
        if page_needs_coverage:
            meets_size_threshold = img_width >= 200 and img_height >= 200
        else:
            meets_size_threshold = img_width >= 300 and img_height >= 300
        meets_area_threshold = area_ratio >= 0.40
        return meets_size_threshold or meets_area_threshold

    def _run_shadow_extraction(
        self,
        file_path: Path,
        doc_hash: str,
        source_file: str,
        file_type: FileType,
        page_images: Dict[int, Image.Image],
        page_dims: Dict[int, Tuple[float, float]],
        page_offset: int,
        state: ContextStateV2,
        text_buffer: List[str],
        element_indices: Dict[str, int],
        engine_processed_pages: set,
        pages_with_prior_chunks: Optional[set] = None,
    ) -> Generator[IngestionChunk, None, None]:
        """
        FIX 1: SHADOW EXTRACTION (REQ-MM-05/06/07)

        Scan PDF for bitmaps/images that the engine may have missed.
        This is the "safety net" that catches large editorial images.

        THRESHOLD (SRS v2.4 STRICT):
        - 300x300 pixels OR
        - 40% of page area OR GREATER

        PLAN_V2.10 Phase 3 (2026-05-12): page-coverage-conditional
        relaxation. When `pages_with_prior_chunks` is supplied and the
        page is NOT in that set (Engine and the main pipeline produced
        nothing for it), the size threshold drops to 200x200 so mid-
        sized chapter-intro diagrams (Python_Distilled pp 686/688/913:
        453x258-290) still emit one image chunk and avoid landing as
        MISSING_PAGES at the strict gate. The area floor stays at 40%
        for that lane too — too-narrow icons and thin banners remain
        filtered.

        This method runs AFTER engine processing and extracts any
        visual elements that meet the threshold but weren't extracted
        by the AI-driven layout analysis.

        Args:
            file_path: Path to original PDF file
            doc_hash: Document hash for asset naming
            source_file: Source filename
            file_type: FileType enum
            page_images: Rendered page images (from the engine)
            page_dims: Page dimensions dict
            page_offset: Page offset for batch processing
            state: Context state for breadcrumbs
            text_buffer: Text buffer for prev_text context
            element_indices: Element counters (figure, table)
            engine_processed_pages: Set of pages engine-processed

        Yields:
            IngestionChunk objects for shadow-extracted assets
        """
        if not file_path.exists() or file_path.suffix.lower() != ".pdf":
            logger.info("[SHADOW-EXTRACTION] Skipping non-PDF or missing file")
            return

        logger.info(f"[SHADOW-EXTRACTION] Scanning {file_path.name} for missed visual assets...")

        pdf_doc = None
        try:
            pdf_doc = fitz.open(str(file_path))
            shadow_count = 0

            for page_idx in range(len(pdf_doc)):
                batch_page_no = page_idx + 1  # 1-indexed
                actual_page_no = batch_page_no + page_offset

                # Only process pages that engine-processed
                if batch_page_no not in engine_processed_pages:
                    continue

                page = pdf_doc[page_idx]
                page_w = page.rect.width
                page_h = page.rect.height
                page_area = page_w * page_h

                # Extract image list from PDF
                image_list = page.get_images(full=True)

                if not image_list:
                    continue

                logger.debug(
                    f"[SHADOW-EXTRACTION] Page {actual_page_no}: "
                    f"Found {len(image_list)} raw images in PDF stream"
                )

                for img_idx, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]

                        # Get image bbox on page
                        img_rects = page.get_image_rects(xref)
                        if not img_rects:
                            continue

                        # Use first occurrence bbox
                        img_rect = img_rects[0]
                        img_width = img_rect.width
                        img_height = img_rect.height
                        img_area = img_width * img_height
                        area_ratio = img_area / page_area if page_area > 0 else 0

                        # ================================================================
                        # REQ-MM-06: THRESHOLD CHECK - 300x300px OR 40% page area
                        # PLAN_V2.10 Phase 3: when the page has NO prior chunks,
                        # relax the size floor to 200x200 so mid-size
                        # chapter-intro diagrams still emit. The area gate is
                        # NOT relaxed — thin banners and decorative dividers
                        # stay filtered out.
                        # ================================================================
                        _page_needs_coverage = (
                            pages_with_prior_chunks is not None
                            and actual_page_no not in pages_with_prior_chunks
                        )
                        if not self._shadow_image_meets_threshold(
                            img_width=img_width,
                            img_height=img_height,
                            area_ratio=area_ratio,
                            page_needs_coverage=_page_needs_coverage,
                        ):
                            logger.debug(
                                f"[SHADOW-EXTRACTION] Page {actual_page_no}, img {img_idx}: "
                                f"Below threshold ({img_width:.0f}x{img_height:.0f}, {area_ratio:.1%})"
                            )
                            continue
                        if _page_needs_coverage and not (
                            img_width >= 300 and img_height >= 300
                        ) and area_ratio < 0.40:
                            logger.info(
                                f"[SHADOW-EXTRACTION][PAGE-COVERAGE] Page {actual_page_no}: "
                                f"Emitting mid-size image ({img_width:.0f}x{img_height:.0f}, "
                                f"{area_ratio:.1%}) under page-coverage threshold; page has "
                                f"no prior chunks (PLAN_V2.10 Phase 3)."
                            )

                        logger.info(
                            f"[SHADOW-EXTRACTION] Page {actual_page_no}: "
                            f"Found large image ({img_width:.0f}x{img_height:.0f}, {area_ratio:.1%})"
                        )

                        # Extract image data
                        base_image = pdf_doc.extract_image(xref)
                        if not base_image:
                            continue

                        image_bytes = base_image["image"]
                        image_ext = base_image.get("ext", "png")

                        # Convert to PIL Image
                        pil_image = Image.open(io.BytesIO(image_bytes))

                        # Apply size check (same as regular images)
                        if not self._check_image_size(
                            pil_image,
                            min_width=self._min_image_width,
                            min_height=self._min_image_height,
                        ):
                            logger.debug(
                                f"[SHADOW-EXTRACTION] Page {actual_page_no}: "
                                f"Image too small after extraction"
                            )
                            continue

                        # ================================================================
                        # FIX 2: VLM GUARD (IRON-07) - Same as regular images
                        # ================================================================
                        if area_ratio > 0.95:
                            logger.info(
                                f"[SHADOW-EXTRACTION][FULL-PAGE-GUARD] Page {actual_page_no}: "
                                f"Shadow asset covers {area_ratio:.1%} of page. Checking VLM..."
                            )

                            # v2.9: defer instead of discard. The
                            # post-conversion enrichment script handles
                            # placeholder image chunks with vision_status
                            # ='pending'. Discarding here was erasing
                            # full-page-image pages from magazine corpora
                            # (e.g. Combat p4 lost its 9-image spread).
                            if not self._vision_manager:
                                logger.info(
                                    f"[SHADOW-EXTRACTION][FULL-PAGE-GUARD] Page {actual_page_no}: "
                                    f"DEFERRING full-page shadow asset (no conversion-time VLM; "
                                    f"will enrich post-conversion)"
                                )
                                print(
                                    f"    ⏸ [SHADOW-VLM-GUARD] Deferred full-page shadow asset "
                                    f"on page {actual_page_no}",
                                    flush=True,
                                )
                                # Fall through — emit the chunk as a
                                # placeholder; the enrichment script picks
                                # it up via vision_status="pending".

                        # Increment figure counter for shadow assets
                        element_indices["figure"] += 1
                        asset_name = ASSET_PATTERN.format(
                            doc_hash=doc_hash,
                            page=actual_page_no,
                            element_type="figure",
                            index=element_indices["figure"],
                        )
                        asset_path = f"assets/{asset_name}"
                        full_path = self.assets_dir / Path(asset_path).name

                        # Save asset to disk
                        pil_image.save(str(full_path), "PNG")
                        logger.info(f"[SHADOW-EXTRACTION] Saved shadow asset: {asset_name}")
                        print(
                            f"    🔍 [SHADOW] {asset_name} | "
                            f"Page={actual_page_no} | "
                            f"Size={img_width:.0f}x{img_height:.0f} ({area_ratio:.1%})",
                            flush=True,
                        )
                        shadow_count += 1

                        # Get context for VLM
                        prev_text = " ".join(text_buffer[-3:]) if text_buffer else None
                        if prev_text:
                            prev_text = prev_text[-CONTEXT_SNIPPET_LENGTH:]

                        # Enrich with VLM
                        content = self._enrich_image_with_vlm(
                            image=pil_image,
                            state=state,
                            page_no=actual_page_no,
                            anchor_text=prev_text,
                            profile_params=self._profile_params,
                        )

                        # Visual description
                        if self._vision_manager and content and len(content) >= 20:
                            visual_description = content
                        else:
                            visual_description = (
                                content if content else f"[Shadow image on page {actual_page_no}]"
                            )

                        # Normalize bbox
                        bbox_list = [
                            img_rect.x0,
                            img_rect.y0,
                            img_rect.x1,
                            img_rect.y1,
                        ]
                        bbox_padded = self._apply_padding(bbox_list, page_w, page_h)
                        bbox_normalized = ensure_normalized(
                            bbox_padded,
                            page_w,
                            page_h,
                            f"shadow_page{actual_page_no}_img{img_idx}",
                        )

                        # Get hierarchy
                        breadcrumbs = state.get_breadcrumb_path()
                        source_name = Path(source_file).stem if source_file else "Document"
                        page_marker = f"Page {actual_page_no}"

                        if not breadcrumbs:
                            breadcrumbs = [source_name, page_marker]
                        elif len(breadcrumbs) == 1:
                            breadcrumbs = [breadcrumbs[0], page_marker]
                        elif page_marker not in " ".join(breadcrumbs):
                            breadcrumbs = [breadcrumbs[0], page_marker] + breadcrumbs[1:]

                        # Charter §3.2 Phase A: shadow-extraction image emit
                        # via UIR. IMAGE modality requires bbox + asset_ref;
                        # visual_description is a v2.16 metadata field set
                        # post-construction (no UIR field yet).
                        _shadow_uir = UIRChunk(
                            modality=Modality.IMAGE,
                            content=content,
                            locator=UIRLocator(
                                type=UIRLocatorType.BBOX,
                                bbox=list(bbox_normalized),
                                page_number=actual_page_no,
                                coordinate_frame=UIRCoordinateFrame.PDF_PAGE_PORTRAIT,
                            ),
                            confidence=UIRConfidenceBreakdown(),
                            extraction_method="shadow",
                            extraction_engine_version="pymupdf-shadow",
                            asset_ref=asset_path,
                            parent_heading=state.get_parent_heading(),
                        )
                        _shadow_chunk = IngestionChunk.from_uir(
                            _shadow_uir,
                            doc_id=doc_hash,
                            source_file=source_file,
                            file_type=file_type,
                            position=self._next_chunk_position(),
                            page_width=int(page_w),
                            page_height=int(page_h),
                            prev_text=prev_text,
                            breadcrumb_path=breadcrumbs,
                            **self._intelligence_metadata,
                        )
                        _shadow_chunk.metadata.visual_description = visual_description
                        # v2.16 literal level on this branch:
                        _shadow_chunk.metadata.hierarchy.level = (
                            len(breadcrumbs) if breadcrumbs else None
                        )
                        yield _shadow_chunk

                    except Exception as img_err:
                        logger.error(
                            f"[SHADOW-EXTRACTION] Page {actual_page_no}, img {img_idx}: "
                            f"Failed to process: {img_err}"
                        )
                        continue

            if shadow_count > 0:
                logger.info(
                    f"[SHADOW-EXTRACTION] Extracted {shadow_count} shadow assets "
                    f"from {file_path.name}"
                )
            else:
                logger.info(
                    f"[SHADOW-EXTRACTION] No additional shadow assets found in {file_path.name}"
                )

        except Exception as e:
            logger.error(f"[SHADOW-EXTRACTION] Failed to scan {file_path.name}: {e}")
        finally:
            if pdf_doc is not None:
                try:
                    pdf_doc.close()
                except Exception:
                    pass

    def _chunk_text_with_overlap(
        self,
        text: str,
        max_chars: int = MAX_CHUNK_CHARS,
        overlap_ratio: float = CHUNK_OVERLAP_RATIO,
    ) -> List[Tuple[str, ChunkType, bool]]:
        """
        Split text into chunks with overlap.

        For programming/technical manuals, prefer code-aware chunking (avoid splitting
        indented blocks mid-scope). For other documents, default to sentence-aware.

        Returns `(text, ChunkType, partial_code)` triples. `partial_code=True`
        means the chunker had to sever a code unit mid-body because the unit
        exceeded the size safe-max (v2.14 Phase 6). Always `False` for prose.
        """
        if not text:
            return []

        max_chars, overlap_ratio = self._effective_chunk_params(max_chars, overlap_ratio)
        overlap_chars = max(MIN_OVERLAP_CHARS, int(max_chars * overlap_ratio))

        # If the text contains code blocks, chunk by block boundaries (code/prose),
        # otherwise keep the simpler sentence-aware chunker.
        if self._looks_like_code(text):
            return self._chunk_mixed_text_and_code(
                text=text,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )

        if len(text) <= max_chars:
            return [(text, ChunkType.PARAGRAPH, False)]

        return [
            (c, ChunkType.PARAGRAPH, False)
            for c in self._chunk_by_sentences(text, max_chars, overlap_chars)
        ]

    def _effective_chunk_params(
        self, max_chars: int, overlap_ratio: float
    ) -> Tuple[int, float]:
        """Profile-aware chunk sizing (keeps default small chunks for non-technical docs)."""
        profile_type = (self._intelligence_metadata.get("profile_type") or "").lower()
        if profile_type == "technical_manual":
            # Avoid over-fragmenting code/prose in programming books.
            return max(max_chars, TECHNICAL_MANUAL_MAX_CHUNK_CHARS), overlap_ratio
        return max_chars, overlap_ratio

    def _chunk_by_sentences(self, text: str, max_chars: int, overlap_chars: int) -> List[str]:
        """Sentence-aware chunking (non-code)."""
        if len(text) <= max_chars:
            return [text]

        sentences = SENTENCE_END_PATTERN.split(text)
        sentences_with_endings: List[str] = []
        pos = 0
        for sent in sentences:
            if not sent.strip():
                continue
            start = text.find(sent, pos)
            if start < 0:
                continue
            end = start + len(sent)
            if end < len(text) and text[end] in ".!?":
                end += 1
            sentences_with_endings.append(text[start:end].strip())
            pos = end

        if not sentences_with_endings:
            return self._chunk_by_words(text, max_chars, overlap_chars)

        chunks: List[str] = []
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

    def _split_blocks_preserve_newlines(self, text: str) -> List[str]:
        """
        Split into blocks separated by blank lines while preserving internal newlines.
        """
        blocks: List[str] = []
        current: List[str] = []
        for ln in text.splitlines():
            if ln.strip() == "":
                if current:
                    blocks.append("\n".join(current).strip("\n"))
                    current = []
                continue
            current.append(ln)
        if current:
            blocks.append("\n".join(current).strip("\n"))
        return [b for b in blocks if b.strip()]

    @staticmethod
    def _code_unit_ranges(lines: List[str]) -> List[Tuple[int, int]]:
        """Identify (start, end-exclusive) ranges of atomic code units.

        Each unit is either:
        - A fenced region (from a ``` opener line through its closer, inclusive)
        - A contiguous run of non-blank lines (between blank lines or fence
          boundaries)

        Blank lines are unit separators and not themselves part of any unit.
        v2.14 Phase 6: used by `_chunk_code_by_lines` to avoid severing a
        code unit mid-body when end-of-unit fits within the size safe-max.
        """
        n = len(lines)
        units: List[Tuple[int, int]] = []
        i = 0
        while i < n:
            while i < n and lines[i].strip() == "":
                i += 1
            if i >= n:
                break
            if lines[i].lstrip().startswith("```"):
                start = i
                i += 1
                while i < n and not lines[i].lstrip().startswith("```"):
                    i += 1
                if i < n:
                    i += 1
                units.append((start, i))
            else:
                start = i
                while i < n and lines[i].strip() != "":
                    i += 1
                units.append((start, i))
        return units

    def _chunk_code_by_lines(
        self, code: str, max_chars: int, overlap_lines: int = 4
    ) -> List[Tuple[str, bool]]:
        """Chunk code by line boundaries, returning (text, partial_code) pairs.

        v2.14 Phase 6: when a chunk would otherwise close mid-unit (fenced
        block or contiguous indented run), extend up to `safe_max` (1.5×
        max_chars) to reach end-of-unit. If end-of-unit still exceeds
        safe_max, accept a line-boundary split and flag every chunk that
        sits inside that oversized unit with `partial_code=True`. Otherwise
        `partial_code=False`. Small line overlap preserves context across
        non-partial boundaries.
        """
        lines = code.splitlines()
        if not lines:
            return []

        safe_max = int(max_chars * 1.5)
        units = self._code_unit_ranges(lines)
        unit_start_of: Dict[int, int] = {}
        unit_end_of: Dict[int, int] = {}
        unit_len_from: Dict[int, int] = {}
        for start, end in units:
            running = 0
            for idx in range(end - 1, start - 1, -1):
                running += len(lines[idx]) + (1 if idx + 1 < end else 0)
                unit_start_of[idx] = start
                unit_end_of[idx] = end
                unit_len_from[idx] = running

        chunks: List[Tuple[str, bool]] = []
        i = 0
        while i < len(lines):
            # A chunk that STARTS mid-unit is a continuation of an oversized
            # unit and must be flagged partial regardless of where it closes.
            chunk_starts_mid_unit = (
                i in unit_start_of and i != unit_start_of[i]
            )
            current: List[str] = []
            cur_len = 0
            closes_mid_unit = False
            while i < len(lines):
                ln = lines[i]
                add_len = len(ln) + (1 if current else 0)
                if current and cur_len + add_len > max_chars:
                    unit_end = unit_end_of.get(i)
                    if unit_end is not None:
                        remaining_in_unit = unit_len_from.get(i, 0)
                        projected = cur_len + 1 + remaining_in_unit
                        if projected <= safe_max:
                            while i < unit_end:
                                ln_e = lines[i]
                                current.append(ln_e)
                                cur_len += len(ln_e) + (1 if len(current) > 1 else 0)
                                i += 1
                        else:
                            closes_mid_unit = True
                    break
                current.append(ln)
                cur_len += add_len
                i += 1
            if current:
                partial = chunk_starts_mid_unit or closes_mid_unit
                chunks.append(("\n".join(current).strip("\n"), partial))
            if i < len(lines):
                if closes_mid_unit:
                    # Continuation begins exactly where we stopped; no overlap
                    # (would duplicate code lines from the previous chunk).
                    continue
                prev_i = i
                i = max(i - overlap_lines, 0)
                # Prevent infinite loop on extremely long single lines: if the
                # overlap would re-emit the same chunk verbatim, advance past it.
                if chunks and chunks[-1][0] == "\n".join(lines[i : i + len(current)]).strip("\n"):
                    i = prev_i + max(1, overlap_lines)
        return [(t, p) for t, p in chunks if t.strip()]

    def _chunk_mixed_text_and_code(
        self, text: str, max_chars: int, overlap_chars: int
    ) -> List[Tuple[str, ChunkType, bool]]:
        """
        Chunk mixed prose/code by block boundaries.

        - Prose blocks: sentence-aware, then size-limited with overlap.
        - Code blocks: never sentence-split; only split by line boundaries
          if needed. v2.14 Phase 6: the third tuple element carries the
          `partial_code` flag from `_chunk_code_by_lines` (True iff the
          code unit had to be split mid-body because it exceeded safe_max).
        """
        blocks = self._split_blocks_preserve_newlines(text)
        if not blocks:
            return [(text, ChunkType.PARAGRAPH, False)]

        typed_blocks: List[Tuple[str, ChunkType]] = []
        for b in blocks:
            b_type = ChunkType.CODE if self._looks_like_code(b) else ChunkType.PARAGRAPH
            typed_blocks.append((b, b_type))

        # Merge adjacent blocks of the same type.
        merged: List[Tuple[str, ChunkType]] = []
        for b, t in typed_blocks:
            if not merged or merged[-1][1] != t:
                merged.append((b, t))
            else:
                merged[-1] = (merged[-1][0] + "\n\n" + b, t)

        out: List[Tuple[str, ChunkType, bool]] = []
        for block_text, block_type in merged:
            if block_type == ChunkType.CODE:
                if len(block_text) <= max_chars:
                    out.append((block_text, ChunkType.CODE, False))
                else:
                    for c, partial in self._chunk_code_by_lines(
                        block_text, max_chars=max_chars
                    ):
                        out.append((c, ChunkType.CODE, partial))
                continue

            # Prose: chunk within the block, then merge adjacent chunks up to max_chars.
            prose_chunks = self._chunk_by_sentences(block_text, max_chars, overlap_chars)
            for c in prose_chunks:
                out.append((c, ChunkType.PARAGRAPH, False))

        return [(t, k, p) for (t, k, p) in out if t and t.strip()]

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
        # Guardrail: when OCR is disabled by CLI/governance, OCR hint extraction
        # must also be disabled to avoid loading EasyOCR runtimes.
        if (not self.enable_ocr) and profile_params.enable_ocr_hints:
            from dataclasses import replace

            self._profile_params = replace(profile_params, enable_ocr_hints=False)
            logger.info(
                "[OCR-HINTS] Disabled for this processor because OCR is off "
                "(enable_ocr=False)"
            )
        else:
            self._profile_params = profile_params

        if self._profile_params.enable_ocr_hints:
            logger.info(
                f"[OCR-HINTS] Enabled for profile. "
                f"DPI: {self._profile_params.render_dpi}, "
                f"min_confidence: {self._profile_params.ocr_min_confidence}"
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
        Enable Semantic Text Refiner (v18.2) for OCR artifact repair.

        Args:
            provider: LLM provider (ollama|openai|anthropic)
            model: Model name (optional for Ollama - auto-detects)
            api_key: API key for cloud providers
            threshold: Min corruption score to trigger refinement (0.0-1.0)
            max_edit: Max edit ratio (0.0-1.0, default 0.35 = 35%)
        """
        try:
            from .refiner import create_refiner

            self._refiner = create_refiner(
                provider=provider,
                model=model,
                api_key=api_key,
                base_url=base_url,
                threshold=threshold,
                max_edit=max_edit,
            )
            logger.info(
                f"[REFINER] Enabled: provider={provider}, "
                f"threshold={threshold}, max_edit={max_edit}"
            )
        except Exception as e:
            logger.error(f"[REFINER] Failed to initialize: {e}")
            self._refiner = None

    def get_vision_stats(self) -> Dict[str, Any]:
        """Get vision manager statistics."""
        if self._vision_manager:
            return self._vision_manager.get_stats()
        return {}

    def cleanup(self) -> None:
        """
        Best-effort resource cleanup for graceful shutdown paths.

        Safe to call multiple times.
        """
        try:
            if getattr(self, "_converter", None) is not None:
                try:
                    for method_name in ("cleanup", "close", "shutdown"):
                        method = getattr(self._converter, method_name, None)
                        if callable(method):
                            method()
                except Exception as e:
                    logger.debug(f"[CLEANUP] converter cleanup skipped: {e}")
                finally:
                    self._converter = None

            if self._vision_manager and self._external_vision_manager is None:
                try:
                    self._vision_manager.flush_cache()
                except Exception as e:
                    logger.debug(f"[CLEANUP] vision cache flush failed: {e}")

            ocr_engine = getattr(self, "_ocr_engine", None)
            if ocr_engine is not None:
                try:
                    for method_name in ("cleanup", "close", "shutdown"):
                        method = getattr(ocr_engine, method_name, None)
                        if callable(method):
                            method()
                except Exception as e:
                    logger.debug(f"[CLEANUP] ocr_engine cleanup skipped: {e}")
                finally:
                    self._ocr_engine = None

            if self._refiner:
                for method_name in ("shutdown", "close"):
                    method = getattr(self._refiner, method_name, None)
                    if callable(method):
                        try:
                            method()
                        except Exception as e:
                            logger.debug(f"[CLEANUP] refiner.{method_name} failed: {e}")

            self._refiner = None
            if self._external_vision_manager is None:
                self._vision_manager = None
            logger.debug("[CLEANUP] V2DocumentProcessor cleanup complete")
        except Exception as e:
            logger.debug(f"[CLEANUP] V2DocumentProcessor cleanup skipped due to error: {e}")

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

    def _next_chunk_position(self) -> int:
        """Allocate the next per-document chunk position (v2.9 Phase 1)."""
        pos = self._chunk_position
        self._chunk_position = pos + 1
        return pos

    def process_document(
        self,
        input_path: str,
    ) -> Generator[IngestionChunk, None, None]:
        """
        Process a document and yield validated IngestionChunk objects.

        v2.10 Phase 7: when the source is an EPUB, the inner generator's
        chunks are buffered and rewritten by
        ``_apply_epub_synthetic_pagination`` so EPUB chunks emit a
        deterministic synthetic ``page_number`` (per chapter + position)
        and the documented full-page bbox sentinel. PDF and other formats
        stream through unchanged.
        """
        src_path = Path(input_path)
        is_epub_source = src_path.suffix.lower() == ".epub"
        inner = self._process_document_inner(input_path)
        if is_epub_source:
            buffered = list(inner)
            chapter_count = getattr(self, "_epub_chapter_count", 0) or 0
            yield from self._apply_epub_synthetic_pagination(buffered, chapter_count)
        else:
            yield from inner

    def _process_document_inner(
        self,
        input_path: str,
    ) -> Generator[IngestionChunk, None, None]:
        """
        Inner per-format extraction generator.

        Args:
            input_path: Path to document file

        Yields:
            IngestionChunk objects validated against schema

        Note:
            In batch processing mode, uses overrides for doc_hash, source_file,
            page_offset, and inherits initial_state for breadcrumb continuity.
            EPUB post-processing (synthetic pagination) is applied by the
            public ``process_document`` wrapper, not here.
        """
        file_path = Path(input_path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        logger.info(f"ENGINE_USE: extraction engine v2.86.0 | Processing: {file_path.name}")

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

        # EPUB pre-processing: extract XHTML chapters to a temporary HTML file.
        # extraction engine 2.86.0 supports HTML but not EPUB natively.
        _tmp_epub_html: Optional[Path] = None
        if file_path.suffix.lower() == ".epub":
            _tmp_epub_html = self._epub_to_html(file_path)
            if _tmp_epub_html:
                logger.info(f"[EPUB] Converted to temporary HTML: {_tmp_epub_html}")
                file_path = _tmp_epub_html
            else:
                raise ValueError(f"Failed to extract HTML from EPUB: {input_path}")

        print("⏳ Starting engine layout analysis...", flush=True)
        logger.info("Starting extraction-engine conversion...")

        import time as _time

        _start = _time.perf_counter()
        # Route through the adapter so post-extraction sanity stages (reading-order
        # y-sort, drop-cap promotion) run when the plan opts in. Calling the
        # raw self._converter would bypass them and re-create the page-13
        # swap and dislocated drop-cap fixed by PLAN_DOCLING_POSTPROCESSOR.md.
        # The adapter is unconditionally constructed in __init__ (see ~L367),
        # so the previous fallback `else: self._converter.convert(...)` was
        # dead code AND a static-guard violation per PLAN_V2.8 §2. Removed.
        result = self._adapter.convert(str(file_path))
        _elapsed = _time.perf_counter() - _start

        print(f"✓ extraction conversion complete in {_elapsed:.1f}s", flush=True)
        logger.info(f"extraction conversion completed in {_elapsed:.1f}s")

        # Reset cross-call state for the dense-index per-element table skip set.
        self._dense_index_pages_skip_table = set()

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

        # ================================================================
        # HYBRID CHUNKER PATH (Phase 1 of migration)
        # ================================================================
        # Use the engine's HybridChunker for text elements (sentence-aware
        # splitting, proper heading hierarchy). Process IMAGE/TABLE
        # elements through the existing element-by-element pipeline.
        #
        # PATHOLOGICAL INPUT GUARD: If total text exceeds the safe
        # threshold, skip HybridChunker to avoid CPU-bound hangs from
        # the sentence-transformer tokenizer on massive inputs.
        # ================================================================
        _use_hybrid = True
        _hybrid_text_chunks: List[IngestionChunk] = []

        # Extraction route check: hybrid_chunker_enabled flag controls whether
        # HybridChunker is used. Scanned documents set this to False because
        # HybridChunker assumes clean digital text and OCR output is noisy.
        _use_hybrid = getattr(self, "_hybrid_chunker_enabled", True)
        if not _use_hybrid:
            logger.info(
                "[EXTRACTION-ROUTE] hybrid_chunker_enabled=False: using element-by-element "
                "chunking (HybridChunker disabled for scan-origin documents)"
            )

        # Guard: estimate total document text size before chunking, and
        # detect any single pathological element (e.g. an index page that
        # the engine extracts as one mega-element) that would feed a
        # multi-million-token sequence to the sentence-transformer tokenizer.
        if _use_hybrid:
            _max_chars = getattr(self, "_max_chunker_input_chars", 500_000)
            _max_per_elem = getattr(self, "_max_chunker_per_element_chars", 100_000)
            _total_text_chars = 0
            _max_elem_chars = 0
            try:
                for _item_tuple in self._get_ordered_doc_items(doc):
                    _elem, _ = _item_tuple
                    _elem_text = getattr(_elem, "text", None)
                    if _elem_text:
                        _elem_len = len(_elem_text)
                        _total_text_chars += _elem_len
                        if _elem_len > _max_elem_chars:
                            _max_elem_chars = _elem_len
                    if _total_text_chars > _max_chars or _max_elem_chars > _max_per_elem:
                        break
            except Exception:
                pass

            if _total_text_chars > _max_chars:
                _use_hybrid = False
                logger.warning(
                    f"[HYBRID-CHUNKER-GUARD] Document text ({_total_text_chars:,} chars) "
                    f"exceeds safe threshold ({_max_chars:,}). Falling back to "
                    f"element-by-element chunking to avoid CPU hang."
                )
            elif _max_elem_chars > _max_per_elem:
                _use_hybrid = False
                logger.warning(
                    f"[HYBRID-CHUNKER-GUARD] Single element ({_max_elem_chars:,} chars) "
                    f"exceeds per-element threshold ({_max_per_elem:,}). Falling back "
                    f"to element-by-element chunking to avoid pathological tokenization."
                )

        if _use_hybrid:
            try:
                _hybrid_text_chunks = self._process_text_with_hybrid_chunker(
                    doc=doc,
                    doc_hash=doc_hash,
                    source_file=source_file,
                    file_type=file_type,
                    page_dims=page_dims,
                    page_offset=self._page_offset,
                    pdf_path=file_path if file_type == FileType.PDF else None,
                )
                logger.info(
                    f"[HYBRID-CHUNKER] Produced {len(_hybrid_text_chunks)} text chunks"
                )

                # Phase 2: Apply refiner to HybridChunker text chunks.
                # When encoding corruption is detected, force-refine ALL chunks
                # (the text layer has /C211 etc. that need LLM reconstruction).
                # This is "Heal-over" — keep HybridChunker's structure, fix the text.
                _encoding_corrupt = self.has_encoding_corruption
                if self._refiner and _encoding_corrupt:
                    try:
                        self._refiner.config.min_refine_threshold = 0.0
                        logger.info("[HYBRID-CHUNKER] Encoding corruption → refiner threshold=0.0")
                    except Exception:
                        pass

                if self._refiner and _hybrid_text_chunks:
                    _refined_count = 0
                    for _hc in _hybrid_text_chunks:
                        if _hc.metadata.chunk_type == ChunkType.CODE:
                            continue  # Never refine code
                        try:
                            _ref_result = self._refiner.process(
                                raw_text=_hc.content,
                                visual_description=None,
                                semantic_context=_hc.semantic_context,
                            )
                            if _ref_result.refinement_applied and _ref_result.refined_text != _hc.content:
                                _hc.metadata.refined_content = _ref_result.refined_text
                                _hc.metadata.refinement_applied = True
                                _hc.metadata.corruption_score = _ref_result.corruption_score
                                _refined_count += 1
                            else:
                                _hc.metadata.refined_content = _hc.content
                        except Exception:
                            _hc.metadata.refined_content = _hc.content
                    logger.info(f"[HYBRID-CHUNKER] Refined {_refined_count} text chunks")

                # Contextualize AFTER refining: prepend heading context to the
                # (possibly refined) text. This ensures the embedded text has
                # both clean content and heading context.
                for _hc in _hybrid_text_chunks:
                    dc_ref = getattr(_hc, "_dc_ref", None)
                    if dc_ref:
                        try:
                            contextualized = chunker.contextualize(dc_ref)
                            # Prepend heading context if contextualize added something
                            if len(contextualized) > len(_hc.metadata.refined_content or ""):
                                _hc.metadata.refined_content = contextualized.strip()
                        except Exception:
                            pass
                        delattr(_hc, "_dc_ref")

            except Exception as _hc_err:
                logger.warning(
                    f"[HYBRID-CHUNKER] Failed, falling back to element-by-element: {_hc_err}"
                )
                _use_hybrid = False
                _hybrid_text_chunks = []

        # V3.0.0: Pending context queue for IMAGE and TABLE next_text_snippet
        pending_visual_chunks: List[IngestionChunk] = []
        # PLAN_V2.10 Phase 3: track pages with any chunk so SHADOW-EXTRACTION
        # can use a page-coverage-aware threshold for pages the engine left
        # entirely empty (Python_Distilled pp 686/688/913: mid-size diagrams
        # that fall just below the standard 300x300 / 40% threshold but the
        # rendered page is non-blank, so the strict gate flags it as
        # MISSING_PAGES). The set is keyed on the absolute page number (i.e.
        # batch_page_no + page_offset) so it lines up with the shadow path.
        pages_with_prior_chunks: set[int] = set()
        for hc in _hybrid_text_chunks:
            _hc_pg = (
                hc.metadata.page_number
                if hc.metadata and hc.metadata.page_number
                else None
            )
            if _hc_pg:
                pages_with_prior_chunks.add(int(_hc_pg))

        for item_tuple in self._get_ordered_doc_items(doc):
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
                skip_text=_use_hybrid,  # Skip text if HybridChunker handled it
            ):
                # V3.0.0: Deferred yield for IMAGE and TABLE chunks (pending context)
                from .schema.ingestion_schema import Modality

                _chunk_pg = (
                    chunk.metadata.page_number
                    if chunk.metadata and chunk.metadata.page_number
                    else None
                )
                if _chunk_pg:
                    pages_with_prior_chunks.add(int(_chunk_pg))

                if chunk.modality in (Modality.IMAGE, Modality.TABLE):
                    # Hold IMAGE/TABLE chunk until next TEXT arrives
                    pending_visual_chunks.append(chunk)
                    logger.debug(
                        f"[PENDING-CONTEXT] Holding {chunk.modality.value} chunk for next_text_snippet"
                    )
                elif chunk.modality == Modality.TEXT:
                    # Flush all pending IMAGE/TABLE chunks with this text as next_text_snippet
                    if pending_visual_chunks:
                        next_snippet = chunk.content[:CONTEXT_SNIPPET_LENGTH]
                        for pending in pending_visual_chunks:
                            # Update semantic context with next text
                            if pending.semantic_context is None:
                                pending.semantic_context = SemanticContext(
                                    next_text_snippet=next_snippet
                                )
                            else:
                                pending.semantic_context.next_text_snippet = next_snippet
                            logger.debug(
                                f"[PENDING-CONTEXT] Filled {pending.modality.value} next_text: '{next_snippet[:50]}...'"
                            )
                            yield pending
                        pending_visual_chunks.clear()
                    # Now yield the TEXT chunk
                    yield chunk
                else:
                    # Unknown modality - yield directly
                    yield chunk

        # ================================================================
        # FIX 1: SHADOW EXTRACTION (REQ-MM-05/06/07)
        # Shadow scan runs AFTER engine extraction to catch missed large images
        # ================================================================
        logger.info(
            "[SHADOW-EXTRACTION] Running post-extraction shadow scan for missed visual assets..."
        )
        for shadow_chunk in self._run_shadow_extraction(
            file_path=Path(input_path),
            doc_hash=doc_hash,
            source_file=source_file,
            file_type=file_type,
            page_images=page_images,
            page_dims=page_dims,
            page_offset=self._page_offset,
            state=state,
            text_buffer=text_buffer,
            element_indices=element_indices,
            engine_processed_pages=set(page_images.keys()),
            pages_with_prior_chunks=pages_with_prior_chunks,
        ):
            # Shadow chunks may also be IMAGE/TABLE, handle pending context
            from .schema.ingestion_schema import Modality

            if shadow_chunk.modality in (Modality.IMAGE, Modality.TABLE):
                pending_visual_chunks.append(shadow_chunk)
            else:
                yield shadow_chunk

        # Yield HybridChunker text chunks (if used)
        if _use_hybrid and _hybrid_text_chunks:
            for hc_chunk in _hybrid_text_chunks:
                yield hc_chunk

        # Flush any remaining pending IMAGE/TABLE chunks (no following text)
        for pending in pending_visual_chunks:
            logger.debug(
                f"[PENDING-CONTEXT] Flushing {pending.modality.value} chunk without next_text"
            )
            yield pending

        # Store final state for batch processing (REQ-STATE: breadcrumb continuity).
        # Phase 4 Step 4: also persist `_last_hybrid_heading` so the next batch's
        # V2DocumentProcessor (created fresh) can resume heading carry-forward.
        state.last_hybrid_heading = getattr(self, "_last_hybrid_heading", None)
        self._final_state = state.get_state_copy()

        # Only flush cache if we own the vision manager (not external)
        if self._vision_manager and self._external_vision_manager is None:
            self._vision_manager.flush_cache()

        assets_saved = len(list(self.assets_dir.glob("*.png")))
        logger.info(
            f"ENGINE_USE: extraction engine v2.86.0 | Completed: {file_path.name} | "
            f"Doc ID: {doc_hash} | Assets saved: {assets_saved}"
        )

        # Clean up temporary EPUB-to-HTML file
        if _tmp_epub_html is not None:
            try:
                _tmp_epub_html.unlink(missing_ok=True)
            except Exception:
                pass

    def _process_text_with_hybrid_chunker(
        self,
        doc: Any,
        doc_hash: str,
        source_file: str,
        file_type: "FileType",
        page_dims: Dict[int, Tuple[float, float]],
        page_offset: int = 0,
        pdf_path: Optional[Path] = None,
    ) -> List["IngestionChunk"]:
        """Process text elements via the engine's sentence-aware chunker.

        Charter §3.2: HybridChunker invocation lives behind the engine's
        `_invoke_text_chunker` entry point. This method consumes the raw
        chunker output, converts each segment through `_engine_chunk_to_uir`,
        and emits IngestionChunks via `IngestionChunk.from_uir`. The engine
        is the only place that touches the underlying extraction-engine
        chunker class.
        """
        # Phase 4 Step 4: seed `_last_hybrid_heading` from the prior batch's
        # state so heading carry-forward survives V2DocumentProcessor
        # re-instantiation between batches. Without this, batches that
        # start before the next section_header lose all heading attribution
        # (Firearms: 84 pages with zero parent_heading clustered at batch
        # boundaries).
        if not getattr(self, "_last_hybrid_heading", None):
            seeded = (
                self._initial_state.last_hybrid_heading
                if self._initial_state is not None
                else None
            )
            if seeded:
                self._last_hybrid_heading = seeded
                logger.info(
                    "[HYBRID-CHUNKER] Seeded last_hybrid_heading from prior batch state: %r",
                    seeded,
                )

        # Engine boundary: run the sentence-aware text chunker + dense-
        # index detection in one call. The engine returns the raw chunker
        # output (`doc_chunks`) for iteration here, plus the
        # dense_index_pages / source_back_index_pages sets the per-element
        # emitter consults to skip table emission.
        try:
            invocation = _invoke_text_chunker(
                doc=doc,
                page_offset=page_offset,
                page_dims=page_dims,
                pdf_path=pdf_path,
                suppress_layout_label_text=getattr(
                    self, "_suppress_layout_label_text", False
                ),
                timeout_seconds=120,
            )
        except TimeoutError:
            raise  # Let caller set _use_hybrid=False via its except block

        dense_index_pages = invocation.dense_index_pages
        # Stash dense_index_pages so the per-element table/image emitter
        # downstream can skip emission on these pages. Use GLOBAL page
        # numbers because `_process_element_v2` checks against
        # `batch_page_no + page_offset`.
        self._dense_index_pages_skip_table = {
            page + page_offset for page in dense_index_pages
        }
        doc_chunks = invocation.doc_chunks

        # v2.9: keep DOCUMENT_INDEX (TOC) elements. Previously these
        # were filtered out as boilerplate, but TOC pages contain real
        # navigation content (chapter titles, section refs, page nums)
        # that the page-coverage gate legitimately expects to see. The
        # garbled-TOC filter at finalization (`_TOC_MARKER` regex) still
        # drops chunks where the engine's cell-marker noise leaks through;
        # legitimate TOC pages survive that filter and are emitted.

        chunks: List[IngestionChunk] = []
        # v2.9: per-page content dedup. the engine's HybridChunker can yield
        # the same doc_chunk text 60+ times for cell-rich tables (e.g. the
        # squadron-roster table on Combat Aircraft p66 produced 73 byte-
        # equal text dupes). The downstream chunk_id position counter
        # gives each emission a distinct id, but the content is identical
        # — wasted index space + skewed retrieval scores. Drop dupes
        # here at the producer rather than papering over them at the
        # writer.
        _seen_per_page: Dict[int, set[str]] = {}
        _hybrid_dedup_skipped = 0

        # Charter §3.2 INPUT boundary: each chunker output segment is
        # converted to UIR by `_engine_chunk_to_uir`. Inside the loop body
        # only UIR types (UIRChunk, Locator) and v2.16 schema types
        # (IngestionChunk, ChunkType) are touched. The validated-heading
        # list per segment also comes from the engine
        # (`_doc_chunk_validated_headings`) so no extraction-engine
        # metadata fields are accessed directly here.
        for i, dc in enumerate(doc_chunks):
            _carryover_heading = getattr(self, "_last_hybrid_heading", None)
            uir_pairs = _engine_chunk_to_uir(
                dc=dc,
                page_offset=page_offset,
                page_dims=page_dims,
                doc=doc,
                last_hybrid_heading=_carryover_heading,
            )
            if not uir_pairs:
                continue

            is_cross_page = len(uir_pairs) > 1
            if is_cross_page:
                logger.info(
                    f"[HYBRID-CHUNKER] Cross-page chunk split across "
                    f"{sorted({u.locator.page_number for u, _ in uir_pairs})}: "
                    f"emitting per-page text"
                )

            # Heading carry-forward state lives at the caller. The engine
            # carries no heading info on cross-page splits (v2.16 used the
            # doc-name root only); single-page emissions inherit
            # self._last_hybrid_heading when no explicit heading is present.
            headings: List[str] = _doc_chunk_validated_headings(dc)
            if headings and not is_cross_page:
                self._last_hybrid_heading = headings[-1]

            _clean_name = (
                Path(source_file).stem.replace("_", " ") if source_file else "Document"
            )

            for _uir, _chunk_type in uir_pairs:
                _page_no = _uir.locator.page_number
                _page_w, _page_h = page_dims.get(_page_no, (612.0, 792.0))

                # Dedup at the page level (UIR content, not UIRChunk text).
                _page_seen = _seen_per_page.setdefault(_page_no, set())
                if _uir.content in _page_seen:
                    _hybrid_dedup_skipped += 1
                    continue
                _page_seen.add(_uir.content)

                if is_cross_page:
                    # v2.16 literal: cross-page split breadcrumb is doc-name
                    # + "Page N" only. No headings — heading inheritance
                    # would attribute the wrong heading to fragments.
                    _breadcrumb = [_clean_name, f"Page {_page_no}"]
                    _parent_heading = None
                    _level = 2
                else:
                    _bc = [_clean_name]
                    if headings:
                        _bc.extend(headings)
                    elif (
                        hasattr(self, "_last_hybrid_heading")
                        and self._last_hybrid_heading
                    ):
                        _bc.append(self._last_hybrid_heading)
                    _bc.append(f"Page {_page_no}")
                    _breadcrumb = _bc
                    _parent_heading = (
                        headings[-1] if headings else self._last_hybrid_heading
                    )
                    _level = min(len(_breadcrumb), 5)

                # Stamp parent_heading onto the UIR for from_uir to consume.
                _uir_for_emit = UIRChunk(
                    modality=_uir.modality,
                    content=_uir.content,
                    locator=_uir.locator,
                    confidence=_uir.confidence,
                    extraction_method=_uir.extraction_method,
                    extraction_engine_version=_uir.extraction_engine_version,
                    structural_flags=set(_uir.structural_flags),
                    parent_heading=_parent_heading,
                )

                _prev_snippet = (
                    chunks[-1].content[-CONTEXT_SNIPPET_LENGTH:] if chunks else None
                )

                chunk = IngestionChunk.from_uir(
                    _uir_for_emit,
                    doc_id=doc_hash,
                    source_file=source_file,
                    file_type=file_type,
                    position=self._next_chunk_position(),
                    page_width=int(_page_w),
                    page_height=int(_page_h),
                    chunk_type=_chunk_type,
                    prev_text=_prev_snippet if not is_cross_page else None,
                    breadcrumb_path=_breadcrumb,
                    **self._intelligence_metadata,
                )
                chunk.metadata.refined_content = _uir.content
                chunk.metadata.hierarchy.level = _level
                # Post-refiner contextualization needs the UIRChunk reference
                # (engine-side state); kept on single-page emissions only,
                # matching v2.16 (cross-page splits did not preserve _dc_ref).
                if not is_cross_page:
                    chunk._dc_ref = dc  # type: ignore[attr-defined]
                chunks.append(chunk)

        if _hybrid_dedup_skipped:
            logger.info(
                f"[HYBRID-CHUNKER] Dedup-skipped {_hybrid_dedup_skipped} byte-equal "
                f"text emissions (engine cell-rich-table cluster)"
            )

        if dense_index_pages:
            chunks.extend(
                self._emit_dense_index_page_chunks(
                    doc=doc,
                    dense_pages=dense_index_pages,
                    doc_hash=doc_hash,
                    source_file=source_file,
                    file_type=file_type,
                    page_dims=page_dims,
                    page_offset=page_offset,
                    pdf_path=pdf_path,
                    source_text_only_pages=source_back_index_pages,
                )
            )
            chunks.sort(key=lambda ch: ch.metadata.page_number if ch.metadata else 0)

        # Plan v2.9 Phase B3 (2026-05-11): section-header-only pages.
        # HybridChunker treats `section_header` engine items as heading
        # metadata to be propagated into the breadcrumbs of subsequent
        # text chunks. On pages whose ONLY content is one or more
        # section_header items (chapter divider / part-opener pages,
        # e.g. Devlin p170 "II — Building Intelligent Foundations"),
        # no text chunk is emitted at all and the strict gate reports
        # MISSING_PAGES even though the heading is high-retrieval-value
        # signal (it answers queries like "where does Part II start?").
        # Per the Retrieval-Value Test (`docs/DECISIONS.md`), emit one
        # text chunk for each such page using the concatenated heading
        # text as content. Tagged `hybrid_chunker_section_header_page`
        # so the chunks are traceable and so the B1 corruption-quarantine
        # exemption pattern (extraction_method-scoped) can extend here if
        # ever needed.
        section_header_only_chunks = self._emit_section_header_only_page_chunks(
            doc=doc,
            existing_chunks=chunks,
            dense_index_pages=dense_index_pages,
            doc_hash=doc_hash,
            source_file=source_file,
            file_type=file_type,
            page_dims=page_dims,
            page_offset=page_offset,
        )
        if section_header_only_chunks:
            chunks.extend(section_header_only_chunks)
            chunks.sort(key=lambda ch: ch.metadata.page_number if ch.metadata else 0)

        # Fill next_text_snippet (skip chunks that don't have a
        # semantic_context — e.g. v2.9 hybrid_chunker_pagesplit emits
        # don't carry one because they lack the dc.contextualize
        # input).
        for i, ch in enumerate(chunks):
            ctx = ch.semantic_context
            if ctx is None:
                continue
            ctx.prev_text_snippet = (
                chunks[i - 1].content[-CONTEXT_SNIPPET_LENGTH:] if i > 0 else None
            )
            ctx.next_text_snippet = (
                chunks[i + 1].content[:CONTEXT_SNIPPET_LENGTH]
                if i < len(chunks) - 1
                else None
            )
        return chunks

    def _emit_dense_index_page_chunks(
        self,
        doc: Any,
        dense_pages: set[int],
        doc_hash: str,
        source_file: str,
        file_type: "FileType",
        page_dims: Dict[int, Tuple[float, float]],
        page_offset: int = 0,
        pdf_path: Optional[Path] = None,
        source_text_only_pages: Optional[set[int]] = None,
    ) -> List["IngestionChunk"]:
        """Emit IngestionChunks for dense TOC/index pages (UIR-native).

        Charter §3.2 Phase A step 2: the extraction boundary is encapsulated
        in `_dense_page_to_uir_chunk` (one call per page). Everything after
        the bridge call operates on UIR types — chunk-splitting, breadcrumb
        synthesis, and emission via `IngestionChunk.from_uir`.

        Behavior parity with v2.16:
          * `extraction_method` carried verbatim from the bridge
            (`hybrid_chunker_pageskip` or `hybrid_chunker_pageskip_source_pdf`).
          * `chunk_type=ChunkType.LIST_ITEM` on every emitted chunk.
          * `parent_heading` = "Contents" when first line starts with
            "contents", else "Index".
          * `refined_content` mirrors `content` (v2.16 invariant).
          * `search_priority` = "low".
        """
        chunks: List[IngestionChunk] = []
        source_text_only_pages = source_text_only_pages or set()

        clean_name = Path(source_file).stem.replace("_", " ") if source_file else "Document"
        intelligence_md = self._intelligence_metadata
        for raw_page in sorted(dense_pages):
            out_page = raw_page + page_offset
            page_w, page_h = page_dims.get(out_page, page_dims.get(raw_page, (612.0, 792.0)))

            # --- Engine boundary: convert page slice to UIR (one call) ---
            page_uir = _dense_page_to_uir_chunk(
                doc=doc,
                raw_page=raw_page,
                source_text_only_pages=source_text_only_pages,
                pdf_path=pdf_path,
                page_w=page_w,
                page_h=page_h,
            )
            if page_uir is None:
                continue

            # --- UIR-native logic from here on ---
            first_line = page_uir.content.splitlines()[0].strip().lower() if page_uir.content else ""
            title = "Contents" if "contents" in first_line else "Index"
            parts = self._split_dense_index_text(page_uir.content, max_chars=6000)

            for idx, part in enumerate(parts):
                breadcrumb = [clean_name, title, f"Page {out_page}"]
                if len(parts) > 1:
                    breadcrumb.append(f"Part {idx + 1}")

                # Each part is a child UIRChunk derived from the page UIR.
                # locator.page_number carries the page_offset already applied;
                # locator.bbox is inherited from the page-level union.
                part_uir = UIRChunk(
                    modality=Modality.TEXT,
                    content=part,
                    locator=UIRLocator(
                        type=UIRLocatorType.BBOX,
                        bbox=list(page_uir.locator.bbox) if page_uir.locator.bbox else [0, 0, COORD_SCALE, COORD_SCALE],
                        page_number=out_page,
                        coordinate_frame=UIRCoordinateFrame.PDF_PAGE_PORTRAIT,
                    ),
                    confidence=page_uir.confidence,
                    extraction_method=page_uir.extraction_method,
                    extraction_engine_version=page_uir.extraction_engine_version,
                    parent_heading=title,
                )

                chunk = IngestionChunk.from_uir(
                    part_uir,
                    doc_id=doc_hash,
                    source_file=source_file,
                    file_type=file_type,
                    position=self._next_chunk_position(),
                    page_width=int(page_w),
                    page_height=int(page_h),
                    chunk_type=ChunkType.LIST_ITEM,
                    breadcrumb_path=breadcrumb,
                    **intelligence_md,
                )

                # v2.16 invariants that from_uir doesn't auto-populate:
                chunk.metadata.refined_content = part
                chunk.metadata.search_priority = "low"
                chunks.append(chunk)
        return chunks

    def _emit_section_header_only_page_chunks(
        self,
        doc: Any,
        existing_chunks: List["IngestionChunk"],
        dense_index_pages: set[int],
        doc_hash: str,
        source_file: str,
        file_type: "FileType",
        page_dims: Dict[int, Tuple[float, float]],
        page_offset: int = 0,
    ) -> List["IngestionChunk"]:
        """Plan v2.9 Phase B3: emit one chunk for each page whose only
        engine items are `section_header` (a chapter / part divider
        page).

        Skips pages already covered by another chunk (the hybrid_chunker
        main pass or the dense-index emitter). Skips pages with any
        non-heading text items (those go through the normal hybrid_chunker
        path and any chunk drop on them is a separate defect to be
        diagnosed, not papered over here). Skips pages in
        `dense_index_pages` (already handled by
        `_emit_dense_index_page_chunks`).

        See `docs/DECISIONS.md` "Retrieval-Value Test" for the principle:
        a section_header on a chapter-divider page is high-retrieval-value
        signal (answers "where does Chapter / Part X start?"), so we
        emit the chunk rather than mark the page blank-equivalent.
        """
        # Charter §3.2 Phase A step 3: the extraction boundary is encapsulated
        # in `_section_header_page_to_uir_chunk` (one call per candidate
        # page). The method body operates on UIR-typed inputs after the
        # bridge call and emits IngestionChunks via IngestionChunk.from_uir.
        if not hasattr(doc, "iterate_items"):
            return []

        # Compute pages already covered.
        covered_pages: set[int] = set()
        for ch in existing_chunks:
            md = getattr(ch, "metadata", None)
            page_no = getattr(md, "page_number", None)
            if page_no is not None:
                covered_pages.add(int(page_no))

        # Group engine items per (raw) page number. This is the engine
        # boundary scan — the per-page item list it produces is the input
        # to the UIR bridge below.
        items_per_page: Dict[int, List[Any]] = {}
        for item, _level in doc.iterate_items():
            raw_page = _item_page_no(item)
            if raw_page is None:
                continue
            if not _item_text(item):
                continue
            items_per_page.setdefault(raw_page, []).append(item)

        clean_name = (
            Path(source_file).stem.replace("_", " ") if source_file else "Document"
        )

        intelligence_md = self._intelligence_metadata
        emitted: List[IngestionChunk] = []
        for raw_page in sorted(items_per_page.keys()):
            out_page = raw_page + page_offset
            if out_page in covered_pages:
                continue
            if raw_page in dense_index_pages:
                # Already handled by the dense-index emitter; if a
                # leaked dense-index doc-chunk caused the gap, that's
                # a separate defect.
                continue

            page_w, page_h = page_dims.get(
                out_page, page_dims.get(raw_page, (612.0, 792.0))
            )

            # --- Engine boundary: convert page slice to UIR (returns
            # None when the page is mixed-label or has no heading text) ---
            page_uir = _section_header_page_to_uir_chunk(
                items=items_per_page[raw_page],
                page_number=out_page,
                page_w=page_w,
                page_h=page_h,
            )
            if page_uir is None:
                continue

            # --- UIR-native emission ---
            primary_heading = page_uir.parent_heading or ""
            breadcrumb = [clean_name, primary_heading, f"Page {out_page}"]

            chunk = IngestionChunk.from_uir(
                page_uir,
                doc_id=doc_hash,
                source_file=source_file,
                file_type=file_type,
                position=self._next_chunk_position(),
                page_width=int(page_w),
                page_height=int(page_h),
                chunk_type=ChunkType.HEADING,
                breadcrumb_path=breadcrumb,
                **intelligence_md,
            )

            # v2.16 invariants that from_uir doesn't auto-populate:
            chunk.metadata.refined_content = page_uir.content
            chunk.metadata.search_priority = "high"
            # v2.16 dictated hierarchy.level=2 explicitly for section-header
            # page chunks (chapter / part dividers), independent of the
            # 3-element breadcrumb. Preserve the literal v2.16 value.
            chunk.metadata.hierarchy.level = 2
            emitted.append(chunk)

        if emitted:
            logger.info(
                "[HYBRID-CHUNKER] Emitted %d section-header-only page chunk(s) "
                "for pages: %s",
                len(emitted),
                sorted(int(ch.metadata.page_number) for ch in emitted),
            )
        return emitted

    def _union_item_bboxes(
        self,
        items: List[Any],
        page_w: float,
        page_h: float,
    ) -> List[int]:
        left, top, right, bottom = COORD_SCALE, COORD_SCALE, 0, 0
        have_bbox = False
        for item in items:
            prov = getattr(item, "prov", None)
            if not prov:
                continue
            first = prov[0] if isinstance(prov, list) else prov
            bbox = getattr(first, "bbox", None)
            if not bbox:
                continue
            x0 = int(float(getattr(bbox, "l", 0)) / page_w * COORD_SCALE)
            y0 = int(float(getattr(bbox, "t", 0)) / page_h * COORD_SCALE)
            x1 = int(float(getattr(bbox, "r", page_w)) / page_w * COORD_SCALE)
            y1 = int(float(getattr(bbox, "b", page_h)) / page_h * COORD_SCALE)
            left = min(left, max(0, min(COORD_SCALE, x0)))
            top = min(top, max(0, min(COORD_SCALE, min(y0, y1))))
            right = max(right, max(0, min(COORD_SCALE, x1)))
            bottom = max(bottom, max(0, min(COORD_SCALE, max(y0, y1))))
            have_bbox = True
        if have_bbox and right > left and bottom > top:
            return [left, top, right, bottom]
        return [0, 0, COORD_SCALE, COORD_SCALE]

    def _split_dense_index_text(self, text: str, max_chars: int) -> List[str]:
        if len(text) <= max_chars:
            return [text]
        lines = [line for line in text.splitlines() if line.strip()]
        parts: List[str] = []
        current: List[str] = []
        current_len = 0
        for line in lines:
            extra = len(line) + (1 if current else 0)
            if current and current_len + extra > max_chars:
                parts.append("\n".join(current).strip())
                current = [line]
                current_len = len(line)
            else:
                current.append(line)
                current_len += extra
        if current:
            tail = "\n".join(current).strip()
            if parts and len(tail) < 200:
                parts[-1] = f"{parts[-1]}\n{tail}".strip()
            else:
                parts.append(tail)
        return parts or [text]

    def _generate_chunk_id(self, doc_hash: str, page_no: int, index: int) -> str:
        """Generate a deterministic chunk ID."""
        import hashlib
        raw = f"{doc_hash}_{page_no:03d}_{index:04d}"
        h = hashlib.md5(raw.encode()).hexdigest()[:16]
        return f"{doc_hash}_{page_no:03d}_text_{h}"

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
        skip_text: bool = False,
    ) -> Generator[IngestionChunk, None, None]:
        """
        Process a single document element with VLM enrichment.

        Args:
            page_offset: Offset to add to page numbers (for batch processing).
                        If batch starts at page 11, offset=10, so batch page 1 → actual page 11.
            skip_text: If True, skip text elements (handled by HybridChunker).
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
        # RULE: Page number MUST come from engine provenance, NOT counters.
        # The prov[0].page_no is the absolute truth from the PDF.
        batch_page_no = 1
        provenance_page = None  # For validation
        bbox = None

        if hasattr(element, "prov") and element.prov:
            prov = element.prov[0] if isinstance(element.prov, list) else element.prov

            # PROVENANCE-BASED PAGE NUMBER (MANDATORY)
            if hasattr(prov, "page_no") and prov.page_no is not None:
                provenance_page = prov.page_no
                batch_page_no = prov.page_no
                logger.debug(
                    f"[PAGE-PROV] Element '{label[:20]}' → engine prov.page_no={provenance_page}"
                )
            else:
                # Fallback: try page attribute
                if hasattr(prov, "page") and prov.page is not None:
                    provenance_page = prov.page
                    batch_page_no = prov.page
                    logger.debug(
                        f"[PAGE-PROV] Element '{label[:20]}' → engine prov.page={provenance_page}"
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
                    f"engine prov={provenance_page}, offset={page_offset}. "
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

            # ================================================================
            # FIX 2: VLM GUARD (IRON-07 / REQ-MM-11)
            # Full-page assets (>95% page area) MUST be discarded if no VLM
            # ================================================================
            img_page_w, img_page_h = page_dims.get(batch_page_no, (612.0, 792.0))

            # Calculate area ratio (REQ-MM-08)
            if bbox:
                # Denormalize bbox to PDF points for area calculation
                from .utils.coordinate_normalization import denormalize_bbox

                abs_bbox = denormalize_bbox(bbox, img_page_w, img_page_h)
                asset_width = abs_bbox[2] - abs_bbox[0]
                asset_height = abs_bbox[3] - abs_bbox[1]
                page_area = img_page_w * img_page_h
                asset_area = asset_width * asset_height
                area_ratio = asset_area / page_area if page_area > 0 else 0
            else:
                # No bbox - assume full page
                area_ratio = 1.0

            # REQ-MM-09/10/11: Full-Page Guard validation
            if area_ratio > 0.95:
                logger.info(
                    f"[FULL-PAGE-GUARD] Page {page_no}: Asset covers {area_ratio:.1%} of page "
                    f"(threshold: 95%). Checking VLM availability..."
                )

                # v2.9 architecture: conversion runs with --vision-provider
                # none; image enrichment is a separate post-step driven by
                # ``scripts/enrich_image_chunks_v29.py``. Discarding full-page
                # assets when the conversion-time VLM is absent erases entire
                # pages from the corpus (Combat p4 lost all 9 image chunks
                # this way in the v2.9 broad reconversion). Defer the
                # editorial-vs-UI verification to the enrichment step
                # instead: emit the chunk with vision_status="pending" and
                # let the post-conversion VLM decide.
                if not self._vision_manager or self._vision_manager is None:
                    logger.info(
                        f"[FULL-PAGE-GUARD] Page {page_no}: DEFERRING full-page asset "
                        f"(area_ratio={area_ratio:.1%}). Will enrich post-conversion. "
                        f"Asset: {asset_name}"
                    )
                    print(
                        f"    ⏸ [VLM-GUARD] Deferred full-page asset on page {page_no} "
                        f"(awaiting post-conversion enrichment)",
                        flush=True,
                    )
                    # Fall through — the chunk is still emitted. The
                    # placeholder visual_description is the existing
                    # ``[Figure on page N]`` sentinel, picked up by the
                    # enrichment script's placeholder-detection.
                else:
                    # IRON-07: VLM VERIFICATION - Actually verify the asset is editorial
                    logger.info(
                        f"[FULL-PAGE-GUARD] Page {page_no}: Running VLM verification for full-page asset..."
                    )
                    try:
                        # Call VLM with verification prompt
                        verification_prompt = (
                            "Is this image editorial content (photo, illustration, infographic, diagram) "
                            "or is it a UI element/page scan/navigation element? "
                            "Answer 'EDITORIAL' if it contains meaningful visual content, "
                            "or 'UI' if it's just a page background, navigation, or UI element."
                        )

                        verification_result = self._vision_manager.enrich_image(
                            image=raw_image,
                            state=state,
                            page_number=page_no,
                            anchor_text=verification_prompt,
                        )

                        # Check VLM response
                        if verification_result and "UI" in verification_result.upper():
                            logger.warning(
                                f"[FULL-PAGE-GUARD] Page {page_no}: VLM identified as UI element: '{verification_result}'. "
                                f"DISCARDING asset: {asset_name}"
                            )
                            print(
                                f"    ⛔ [VLM-VERIFICATION] Discarded full-page asset on page {page_no} "
                                f"(VLM: UI element)",
                                flush=True,
                            )
                            return  # DISCARD - VLM says it's UI
                        else:
                            logger.info(
                                f"[FULL-PAGE-GUARD] Page {page_no}: VLM verified as editorial: '{verification_result}'. "
                                f"KEEPING asset: {asset_name}"
                            )
                            print(
                                f"    ✅ [VLM-VERIFICATION] Verified full-page editorial asset on page {page_no}",
                                flush=True,
                            )

                    except Exception as vlm_err:
                        logger.error(
                            f"[FULL-PAGE-GUARD] Page {page_no}: VLM verification failed: {vlm_err}. "
                            f"DISCARDING asset for safety: {asset_name}"
                        )
                        print(
                            f"    ⛔ [VLM-VERIFICATION] VLM error on page {page_no}, discarding for safety",
                            flush=True,
                        )
                        return  # DISCARD - VLM failed, err on side of caution

            # STEP 3: Only save to disk if size check passed AND VLM guard passed
            print(
                f"    📸 Asset: {asset_name} | "
                f"Page={page_no} | "
                f"EngineProv={provenance_page}",
                flush=True,
            )

            full_path = self.assets_dir / Path(asset_path).name
            try:
                # Same trim as _save_asset: clean up whitespace-heavy page crops for better downstream RAG.
                try:
                    from .utils.image_trim import trim_white_margins

                    trim_res = trim_white_margins(raw_image)
                    if trim_res.trimmed:
                        raw_image = trim_res.image
                        logger.debug(
                            f"[ASSET-TRIM] Page {page_no}: trimmed margins for {Path(asset_path).name} "
                            f"(bbox={trim_res.bbox}, size={raw_image.size})"
                        )
                except Exception as trim_err:
                    logger.debug(f"[ASSET-TRIM] Skipped due to error: {trim_err}")

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

            # Charter §3.2 Phase A: per-element image emit via UIR.
            _img_uir = UIRChunk(
                modality=Modality.IMAGE,
                content=content,
                locator=UIRLocator(
                    type=UIRLocatorType.BBOX,
                    bbox=list(image_bbox),
                    page_number=page_no,
                    coordinate_frame=UIRCoordinateFrame.PDF_PAGE_PORTRAIT,
                ),
                confidence=UIRConfidenceBreakdown(),
                extraction_method=EXTRACTION_METHOD_NATIVE,
                extraction_engine_version=EXTRACTION_ENGINE_VERSION,
                asset_ref=asset_path,
                parent_heading=hierarchy.parent_heading if hierarchy else None,
            )
            _img_chunk = IngestionChunk.from_uir(
                _img_uir,
                doc_id=doc_hash,
                source_file=source_file,
                file_type=file_type,
                position=self._next_chunk_position(),
                page_width=int(img_page_w),
                page_height=int(img_page_h),
                prev_text=prev_text,
                breadcrumb_path=(
                    list(hierarchy.breadcrumb_path) if hierarchy else []
                ),
                **self._intelligence_metadata,
            )
            _img_chunk.metadata.visual_description = visual_description
            if hierarchy and hierarchy.level is not None:
                _img_chunk.metadata.hierarchy.level = hierarchy.level
            yield _img_chunk

        elif "table" in label_lower:
            # Phase 4 Step 2: skip table emission on dense back-index pages
            # routed through `_emit_dense_index_page_chunks` — the engine's table
            # extraction on these pages produces page-corrupted output (e.g.
            # Adedeji p301: 9-chunk repetition loop).
            if page_no in getattr(self, "_dense_index_pages_skip_table", set()):
                return
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

            # Last-resort crop if save path failed but we still have page buffers.
            if saved_table_image is None and bbox is not None:
                try:
                    recovered = self._extract_raw_image(
                        element=element,
                        bbox_normalized=bbox,
                        page_images=page_images,
                        page_no=batch_page_no,
                        page_dims=(tbl_page_w, tbl_page_h),
                    )
                    if recovered is not None:
                        full_path = self.assets_dir / Path(asset_name).name
                        recovered.save(str(full_path), "PNG")
                        saved_table_image = recovered
                        asset_path = f"assets/{Path(asset_name).name}"
                        logger.info(
                            f"[TABLE-FALLBACK] Page {page_no}: recovered table asset via raw crop"
                        )
                except Exception as asset_err:
                    logger.debug(
                        f"[TABLE-FALLBACK] Page {page_no}: failed raw-crop table asset recovery: {asset_err}"
                    )

            table_content = (text or "").strip()
            if self._is_table_placeholder_text(table_content, page_no):
                recovered_table_text = self._extract_table_text(element, page_no)
                if recovered_table_text:
                    table_content = recovered_table_text

            table_extraction_method = EXTRACTION_METHOD_NATIVE
            engine_table_markdown = ""
            if table_content and not self._is_table_placeholder_text(table_content, page_no):
                engine_table_markdown = self._table_text_to_markdown(table_content)
                if self._is_markdown_table(engine_table_markdown):
                    table_content = engine_table_markdown
                    table_extraction_method = EXTRACTION_METHOD_TABLE_MARKDOWN

            # Guard against placeholder or text-soup table output:
            # Prefer VLM table serialization, then OCR, then engine-table-markdown fallback.
            vlm_attempted = False
            # v2.14 Phase 1 (2026-05-23): `force_table_vlm` truly forces, even when
            # the resolved profile sets `vlm_table_enabled=False` (e.g. technical_manual
            # disables it to prevent hallucination on schematics — but for form-class
            # docs misclassified as technical_manual, the VLM is the ONLY path that
            # produces structurally-clean tables). The earlier "force requested but
            # disabled by profile" branch silently ignored the user's flag; that was
            # a semantic bug. Profile's vlm_table_enabled still governs the AUTOMATIC
            # path (no flag) — only an explicit `--force-table-vlm` overrides it.
            profile_vlm_table_enabled = (
                self._profile_params.vlm_table_enabled if self._profile_params is not None else True
            )
            vlm_table_enabled = profile_vlm_table_enabled or self._force_table_vlm

            # Emergency guardrail: if profile disables table-VLM but we have no usable
            # table text and OCR is unavailable, allow one strict VLM attempt as a
            # last resort to avoid guaranteed table_image_only collapse.
            emergency_vlm_allowed = (
                (not vlm_table_enabled)
                and (not self.enable_ocr)
                and self._is_table_placeholder_text(table_content, page_no)
                and not engine_table_markdown
                and (saved_table_image is not None)
            )

            if self._force_table_vlm and not profile_vlm_table_enabled:
                logger.info(
                    f"[TABLE-VLM] Page {page_no}: forced VLM table serialization overrides "
                    "profile's vlm_table_enabled=False (semantic fix 2026-05-23)."
                )

            if self._force_table_vlm and vlm_table_enabled:
                vlm_attempted = True
                vlm_markdown = self._extract_table_markdown_with_vlm(saved_table_image, page_no)
                if vlm_markdown:
                    table_content = (
                        vlm_markdown
                        if self._is_markdown_table(vlm_markdown)
                        else self._table_text_to_markdown(vlm_markdown)
                    )
                    table_extraction_method = "vlm_table_markdown_forced"
                else:
                    logger.warning(
                        f"[TABLE-VLM] Page {page_no}: forced VLM serialization returned empty, "
                        "falling back to OCR/engine markdown path."
                    )

            if self._is_unstructured_table_text(table_content, page_no):
                if not vlm_attempted and (vlm_table_enabled or emergency_vlm_allowed):
                    vlm_markdown = self._extract_table_markdown_with_vlm(saved_table_image, page_no)
                else:
                    vlm_markdown = ""
                if vlm_markdown:
                    table_content = (
                        vlm_markdown
                        if self._is_markdown_table(vlm_markdown)
                        else self._table_text_to_markdown(vlm_markdown)
                    )
                    table_extraction_method = (
                        "vlm_table_markdown_emergency"
                        if emergency_vlm_allowed and not vlm_table_enabled
                        else "vlm_table_markdown"
                    )
                else:
                    ocr_markdown = self._extract_table_markdown_with_ocr(saved_table_image, page_no)
                    if ocr_markdown:
                        table_content = (
                            ocr_markdown
                            if self._is_markdown_table(ocr_markdown)
                            else self._table_text_to_markdown(ocr_markdown)
                        )
                        table_extraction_method = "ocr_table_markdown"
                    elif engine_table_markdown:
                        table_content = engine_table_markdown
                        table_extraction_method = EXTRACTION_METHOD_TABLE_MARKDOWN_FALLBACK

            if self._is_table_placeholder_text(table_content, page_no):
                table_content = f"[Table extraction unavailable on page {page_no}]"
                table_extraction_method = "table_image_only"
                logger.warning(
                    f"[TABLE-FALLBACK] Page {page_no}: table text unavailable after all fallbacks"
                )

            # REQ-COORD-01: bbox is REQUIRED for table modality
            # Provide fallback full-page bbox if not available
            table_bbox: List[int] = bbox if bbox is not None else [0, 0, COORD_SCALE, COORD_SCALE]

            # REQ-COORD-02: Get page dimensions for UI overlay support
            tbl_page_w, tbl_page_h = page_dims.get(batch_page_no, (612.0, 792.0))

            # ================================================================
            # FIX 3: TABLE CONTEXT PARITY (REQ-MM-03) - UNIVERSAL TREATMENT
            # Tables MUST get EXACT same prev/next treatment as images
            # Use pending queue logic for next_text_snippet population
            # ================================================================
            prev_text_table = " ".join(text_buffer[-3:]) if text_buffer else None
            if prev_text_table:
                prev_text_table = prev_text_table[-CONTEXT_SNIPPET_LENGTH:]

            # Create table or text chunk depending on whether an asset image is available.
            # Non-PDF formats (HTML, EPUB, DOCX) cannot render table images, so tables
            # are emitted as TEXT with chunk_type=table to satisfy QA-CHECK-05.
            # Charter §3.2 Phase A: table emission via UIR.
            # When asset_path is None (non-PDF formats: HTML/EPUB/DOCX cannot
            # render table images), emit as TEXT modality with the same UIR
            # locator + extraction method. With asset, emit as TABLE.
            _tbl_modality = Modality.TABLE if asset_path is not None else Modality.TEXT
            _tbl_chunk_type = None if asset_path is not None else ChunkType.PARAGRAPH
            _tbl_uir = UIRChunk(
                modality=_tbl_modality,
                content=table_content,
                locator=UIRLocator(
                    type=UIRLocatorType.BBOX,
                    bbox=list(table_bbox),
                    page_number=page_no,
                    coordinate_frame=UIRCoordinateFrame.PDF_PAGE_PORTRAIT,
                ),
                confidence=UIRConfidenceBreakdown(),
                extraction_method=table_extraction_method,
                extraction_engine_version=EXTRACTION_ENGINE_VERSION,
                asset_ref=asset_path,
                parent_heading=hierarchy.parent_heading if hierarchy else None,
            )
            table_chunk = IngestionChunk.from_uir(
                _tbl_uir,
                doc_id=doc_hash,
                source_file=source_file,
                file_type=file_type,
                position=self._next_chunk_position(),
                page_width=int(tbl_page_w),
                page_height=int(tbl_page_h),
                chunk_type=_tbl_chunk_type,
                prev_text=prev_text_table,
                breadcrumb_path=(
                    list(hierarchy.breadcrumb_path) if hierarchy else []
                ),
                **self._intelligence_metadata,
            )
            if hierarchy and hierarchy.level is not None:
                table_chunk.metadata.hierarchy.level = hierarchy.level

            # REQ-MM-03: TABLE CONTEXT PARITY — `from_uir` already created the
            # SemanticContext (prev_text_table was passed in); next_text_snippet
            # stays None for the pending-queue to fill on the following TEXT.
            yield table_chunk

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
            or "code" in label_lower  # Programming books: the engine emits code blocks as \"code\"
            or "footnote" in label_lower  # Common in manuals/books; should be treated as TEXT
            or "formula" in label_lower  # Math/formulas are text-like (often missed otherwise)
            or "equation" in label_lower  # Alias for formula-like regions
            or "index" in label_lower  # e.g., "document_index" pages
            or label_lower == ""  # Default label is text
        )

        # Handle TEXT elements (with or without content)
        # If HybridChunker handles text, skip text element processing here.
        if skip_text and is_text_label and not is_image_label and "table" not in label_lower:
            # Still update state for heading tracking (needed by image/table context)
            heading_level = self._extract_heading_level(label)
            if heading_level and text.strip():
                state.update_on_heading(text.strip(), heading_level)
            return

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
                        scale = 2.0  # engine render scale

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

            for i, (chunk_text, chunk_type, chunk_partial_code) in enumerate(text_chunks):
                if not self._is_noise_content(chunk_text) and not self._is_advertisement(chunk_text):
                    # ================================================================
                    # V3.0.0: SEMANTIC CONTEXT - prev_text_snippet, next_text_snippet
                    # ================================================================
                    # prev_text: from text_buffer (text BEFORE this chunk)
                    # next_text: next chunk in list (if available)
                    next_text_context = None
                    if i + 1 < len(text_chunks):
                        next_text_context = text_chunks[i + 1][0][:CONTEXT_SNIPPET_LENGTH]

                    # Default refined_content to raw text so downstream always
                    # has a non-null value (even when refiner is disabled or skips).
                    refined_content = chunk_text
                    refinement_applied = False
                    corruption_score = None
                    refinement_provider = None
                    refinement_model = None

                    # Never run the refiner on code blocks (avoid mutating syntax/indentation).
                    if self._refiner and chunk_type != ChunkType.CODE:
                        try:
                            semantic_context = SemanticContext(
                                prev_text_snippet=prev_text_context,
                                next_text_snippet=next_text_context,
                                parent_heading=hierarchy.parent_heading,
                                breadcrumb_path=hierarchy.breadcrumb_path,
                            )
                            refine_result = self._refiner.process(
                                raw_text=chunk_text,
                                visual_description=None,
                                semantic_context=semantic_context,
                            )
                            corruption_score = refine_result.corruption_score
                            refinement_provider = refine_result.provider
                            refinement_model = refine_result.model
                            if refine_result.refinement_applied:
                                refined_content = refine_result.refined_text
                                refinement_applied = True
                        except Exception as refiner_error:
                            logger.error(f"[REFINER] Failed on page {page_no}: {refiner_error}")

                    # Charter §3.2 Phase A: per-element text emission via UIR.
                    # `partial_code=True` for the in-block CODE oversize case is
                    # routed via StructuralFlag.PARTIAL_CODE_IN_BLOCK so from_uir
                    # maps it back to metadata.partial_code=True.
                    from .universal.intermediate import StructuralFlag as _SF_TEXT
                    _txt_flags: set = set()
                    if chunk_type == ChunkType.CODE and chunk_partial_code:
                        _txt_flags.add(_SF_TEXT.PARTIAL_CODE_IN_BLOCK)
                    # text_bbox may be None on this branch (chunked code path);
                    # FLOW_OFFSET locator when bbox absent.
                    if text_bbox:
                        _txt_locator = UIRLocator(
                            type=UIRLocatorType.BBOX,
                            bbox=list(text_bbox),
                            page_number=page_no,
                            coordinate_frame=UIRCoordinateFrame.PDF_PAGE_PORTRAIT,
                        )
                    else:
                        _txt_locator = UIRLocator(
                            type=UIRLocatorType.FLOW_OFFSET,
                            page_number=page_no,
                            coordinate_frame=UIRCoordinateFrame.UNKNOWN,
                            path=f"page:{page_no}:element:{i}",
                        )
                    _txt_uir = UIRChunk(
                        modality=Modality.TEXT,
                        content=chunk_text,
                        locator=_txt_locator,
                        confidence=UIRConfidenceBreakdown(),
                        extraction_method=EXTRACTION_METHOD_NATIVE,
                        extraction_engine_version=EXTRACTION_ENGINE_VERSION,
                        structural_flags=_txt_flags,
                        parent_heading=hierarchy.parent_heading if hierarchy else None,
                    )
                    _txt_chunk = IngestionChunk.from_uir(
                        _txt_uir,
                        doc_id=doc_hash,
                        source_file=source_file,
                        file_type=file_type,
                        position=self._next_chunk_position(),
                        page_width=int(page_w) if text_bbox and page_w else None,
                        page_height=int(page_h) if text_bbox and page_h else None,
                        chunk_type=chunk_type,
                        prev_text=prev_text_context,
                        next_text=next_text_context,
                        breadcrumb_path=(
                            list(hierarchy.breadcrumb_path) if hierarchy else []
                        ),
                        **self._intelligence_metadata,
                    )
                    # v2.16 invariants that from_uir doesn't auto-populate:
                    _txt_chunk.metadata.refined_content = refined_content
                    _txt_chunk.metadata.refinement_applied = refinement_applied
                    _txt_chunk.metadata.corruption_score = corruption_score
                    _txt_chunk.metadata.refinement_provider = refinement_provider
                    _txt_chunk.metadata.refinement_model = refinement_model
                    _txt_chunk.metadata.content_classification = (
                        self._classify_text_content(chunk_text, chunk_type)
                    )
                    if hierarchy and hierarchy.level is not None:
                        _txt_chunk.metadata.hierarchy.level = hierarchy.level
                    yield _txt_chunk

    def _classify_text_content(self, text: str, chunk_type: ChunkType) -> str:
        """Deterministic metadata classification for emitted text chunks."""
        if chunk_type == ChunkType.CODE or self._looks_like_code(text):
            return "code"
        if self._is_advertisement(text):
            return "advertisement"

        lowered = (text or "").lower()
        technical_hits = sum(1 for kw in TECHNICAL_KEYWORDS if kw in lowered)
        if technical_hits >= 2:
            return "technical"
        return "editorial"

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

        # Compute doc_id from file MD5 (12-char hex) for metadata record.
        _fp = Path(file_path)
        _hasher = hashlib.md5()
        with open(_fp, "rb") as _fh:
            for _blk in iter(lambda: _fh.read(8192), b""):
                _hasher.update(_blk)
        _doc_id = _hasher.hexdigest()[:12]

        chunk_count = 0
        with open(final_output_path, "w", encoding="utf-8") as f:
            # Write document-level metadata record as the FIRST line.
            _meta_rec = IngestionMetadata(
                schema_version=SCHEMA_VERSION,
                doc_id=_doc_id,
                source_file=_fp.name,
                ingestion_timestamp=datetime.now(timezone.utc).isoformat(),
            )
            f.write(json.dumps(_meta_rec.model_dump(mode="json"), ensure_ascii=False) + "\n")

            for chunk in self.process_document(file_path):
                chunk_dict = chunk.model_dump(mode="json")
                # Ensure schema_version is emitted in metadata for downstream versioning
                meta = chunk_dict.get("metadata", {})
                if meta.get("schema_version") is None:
                    meta["schema_version"] = SCHEMA_VERSION
                    chunk_dict["metadata"] = meta

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
            f"ENGINE_USE: extraction engine v2.86.0 | " f"Written {chunk_count} chunks to {final_output_path}"
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

        # Compute doc_id from file MD5 (12-char hex) for metadata record.
        _fp = Path(file_path)
        _hasher = hashlib.md5()
        with open(_fp, "rb") as _fh:
            for _blk in iter(lambda: _fh.read(8192), b""):
                _hasher.update(_blk)
        _doc_id = _hasher.hexdigest()[:12]

        # Write document-level metadata record as the FIRST line (atomic write).
        _meta_rec = IngestionMetadata(
            schema_version=SCHEMA_VERSION,
            doc_id=_doc_id,
            source_file=_fp.name,
            ingestion_timestamp=datetime.now(timezone.utc).isoformat(),
        )
        with open(final_output_path, "a", encoding="utf-8") as _mf:
            _mf.write(json.dumps(_meta_rec.model_dump(mode="json"), ensure_ascii=False) + "\n")
            _mf.flush()

        chunk_count = 0

        # Open in APPEND mode for atomic writes
        for chunk in self.process_document(file_path):
            chunk_dict = chunk.model_dump(mode="json")

            # Ensure schema_version is emitted in metadata for downstream versioning
            meta = chunk_dict.get("metadata", {})
            if meta.get("schema_version") is None:
                meta["schema_version"] = SCHEMA_VERSION
                chunk_dict["metadata"] = meta

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
            f"ENGINE_USE: extraction engine v2.86.0 | "
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
