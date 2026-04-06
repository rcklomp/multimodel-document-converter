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
    IngestionMetadata,
    SemanticContext,
    ChunkType,
    create_image_chunk,
    create_table_chunk,
    create_text_chunk,
    get_ocr_confidence_level,
    calculate_hierarchy_level,
    COORD_SCALE,
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
        # V2.4 Intelligence Stack Metadata (Observability)
        intelligence_metadata: Optional[Dict[str, Any]] = None,
        # Force table serialization through VLM route (when available)
        force_table_vlm: bool = False,
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
            force_table_vlm: Force table image -> VLM markdown path (fallback to OCR/docling if needed)
        """
        # Store extraction strategy for dynamic thresholds
        self._extraction_strategy = extraction_strategy

        # V3.0.0: Profile parameters for OCR configuration (shadow-first mode REMOVED)
        self._profile_params: Optional["ProfileParameters"] = None

        # V2.4: Store intelligence metadata for JSONL observability
        self._intelligence_metadata: Dict[str, Any] = intelligence_metadata or {}
        if self._intelligence_metadata:
            logger.info(
                f"[OBSERVABILITY] Intelligence metadata loaded: "
                f"profile_type={self._intelligence_metadata.get('profile_type')}, "
                f"min_dims={self._intelligence_metadata.get('min_image_dims')}"
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

        # Pass document domain to VisionManager for domain-aware prompt selection
        if self._vision_manager and self._intelligence_metadata:
            self._vision_manager.document_domain = self._intelligence_metadata.get(
                "document_domain", ""
            )

        # Initialize Semantic Text Refiner (v18.2) - disabled by default
        self._refiner = None

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
        pipeline_options.generate_page_images = (
            True  # ✅ REQ-PDF-04: Enable page rendering for padding
        )
        pipeline_options.generate_picture_images = True  # Extract figures
        pipeline_options.generate_table_images = True  # Extract tables
        if hasattr(pipeline_options, "sort_by_reading_order"):
            pipeline_options.sort_by_reading_order = True
            logger.info("Docling reading-order sort enabled: sort_by_reading_order=True")
        # Prefer structured table extraction so table content is emitted as text/markdown,
        # not as image-only placeholders.
        if hasattr(pipeline_options, "do_table_structure"):
            pipeline_options.do_table_structure = True
        if hasattr(pipeline_options, "do_cell_matching"):
            pipeline_options.do_cell_matching = True

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

    @staticmethod
    def _epub_to_html(epub_path: Path) -> Optional[Path]:
        """Extract XHTML content from an EPUB file into a temporary HTML file.

        EPUB files are ZIP archives containing XHTML chapters. Docling 2.66.0
        doesn't support EPUB natively, but it does support HTML. This method
        extracts all chapter content into a single HTML file that Docling can
        process.

        Returns:
            Path to a temporary HTML file, or None on failure.
        """
        import tempfile
        import zipfile
        from xml.etree import ElementTree as ET

        try:
            parts: list[str] = []
            with zipfile.ZipFile(epub_path, "r") as zf:
                # Find the OPF file via container.xml
                spine_items: list[str] = []
                try:
                    container = ET.fromstring(zf.read("META-INF/container.xml"))
                    ns = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
                    rootfile = container.find(".//c:rootfile", ns)
                    opf_path = rootfile.attrib["full-path"] if rootfile is not None else None
                except Exception:
                    opf_path = None

                if opf_path:
                    try:
                        opf = ET.fromstring(zf.read(opf_path))
                        opf_ns = {"opf": "http://www.idpf.org/2007/opf"}
                        opf_dir = str(Path(opf_path).parent)
                        manifest = {}
                        for item in opf.findall(".//opf:manifest/opf:item", opf_ns):
                            manifest[item.attrib.get("id", "")] = item.attrib.get("href", "")
                        for itemref in opf.findall(".//opf:spine/opf:itemref", opf_ns):
                            idref = itemref.attrib.get("idref", "")
                            href = manifest.get(idref, "")
                            if href:
                                full = f"{opf_dir}/{href}" if opf_dir != "." else href
                                spine_items.append(full)
                    except Exception:
                        pass

                # Fallback: grab all .xhtml/.html files alphabetically
                if not spine_items:
                    spine_items = sorted(
                        n for n in zf.namelist()
                        if n.endswith((".xhtml", ".html", ".htm"))
                        and "META-INF" not in n
                    )

                for item_path in spine_items:
                    try:
                        raw = zf.read(item_path).decode("utf-8", errors="replace")
                        # Strip XML declaration and doctype for concatenation
                        raw = raw.split("<body", 1)[-1] if "<body" in raw.lower() else raw
                        if raw.startswith((" ", "\n", ">")):
                            raw = "<body" + raw  # restore tag
                        parts.append(raw)
                    except Exception:
                        continue

            if not parts:
                return None

            html = (
                "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                f"<title>{epub_path.stem}</title></head>\n"
                + "\n".join(parts)
                + "\n</html>"
            )

            tmp = tempfile.NamedTemporaryFile(
                suffix=".html", prefix="epub_", delete=False, mode="w", encoding="utf-8"
            )
            tmp.write(html)
            tmp.close()
            return Path(tmp.name)
        except Exception as exc:
            logger.warning(f"[EPUB] Failed to convert EPUB to HTML: {exc}")
            return None

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

        # Use same validation as heading
        if not is_valid_heading(text):
            logger.debug(f"[DENOISE] Rejected content (invalid): '{text}'")
            return True

        return False

    def _looks_like_code(self, text: str) -> bool:
        """
        Heuristic code detector for programming books/manuals.

        Docling/PyMuPDF often preserve indentation/newlines for code blocks; we prefer
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
            # PRIORITY 1: Use Docling's native extracted image when available.
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
                    f"Check Docling configuration and batch_processor page_images dict."
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

    def _extract_docling_table_text(self, element: Any, page_no: int) -> str:
        """
        Best-effort extraction of richer table text from Docling table elements.

        Some Docling table elements expose markdown/structured export methods while
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
                value = method()
            except TypeError:
                # Some implementations may require keyword args; skip silently.
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
        """OCR-based fallback for table chunks when Docling table text is missing."""
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
        """VLM fallback for table markdown extraction when OCR/docling are insufficient."""
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
                    "falling back to OCR/docling"
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
        docling_processed_pages: set,
    ) -> Generator[IngestionChunk, None, None]:
        """
        FIX 1: SHADOW EXTRACTION (REQ-MM-05/06/07)

        Scan PDF for bitmaps/images that Docling may have missed.
        This is the "safety net" that catches large editorial images.

        THRESHOLD (SRS v2.4 STRICT):
        - 300x300 pixels OR
        - 40% of page area OR GREATER

        This method runs AFTER Docling processing and extracts any
        visual elements that meet the threshold but weren't extracted
        by Docling's AI-driven layout analysis.

        Args:
            file_path: Path to original PDF file
            doc_hash: Document hash for asset naming
            source_file: Source filename
            file_type: FileType enum
            page_images: Rendered page images (from Docling)
            page_dims: Page dimensions dict
            page_offset: Page offset for batch processing
            state: Context state for breadcrumbs
            text_buffer: Text buffer for prev_text context
            element_indices: Element counters (figure, table)
            docling_processed_pages: Set of pages Docling processed

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

                # Only process pages that Docling processed
                if batch_page_no not in docling_processed_pages:
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
                        # ================================================================
                        meets_size_threshold = img_width >= 300 and img_height >= 300
                        meets_area_threshold = area_ratio >= 0.40

                        if not (meets_size_threshold or meets_area_threshold):
                            logger.debug(
                                f"[SHADOW-EXTRACTION] Page {actual_page_no}, img {img_idx}: "
                                f"Below threshold ({img_width:.0f}x{img_height:.0f}, {area_ratio:.1%})"
                            )
                            continue

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

                            if not self._vision_manager:
                                logger.warning(
                                    f"[SHADOW-EXTRACTION][FULL-PAGE-GUARD] Page {actual_page_no}: "
                                    f"DISCARDING full-page shadow asset (no VLM for verification)"
                                )
                                print(
                                    f"    ⛔ [SHADOW-VLM-GUARD] Discarded full-page shadow asset "
                                    f"on page {actual_page_no}",
                                    flush=True,
                                )
                                continue  # DISCARD

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

                        hierarchy = HierarchyMetadata(
                            parent_heading=state.get_parent_heading(),
                            breadcrumb_path=breadcrumbs,
                            level=len(breadcrumbs) if breadcrumbs else None,
                        )

                        # Create IMAGE chunk with extraction_method="shadow"
                        yield create_image_chunk(
                            doc_id=doc_hash,
                            content=content,
                            source_file=source_file,
                            file_type=file_type,
                            page_number=actual_page_no,
                            asset_path=asset_path,
                            bbox=bbox_normalized,
                            hierarchy=hierarchy,
                            prev_text=prev_text,
                            visual_description=visual_description,
                            page_width=int(page_w),
                            page_height=int(page_h),
                            extraction_method="shadow",  # REQ-MM-07: Mark as shadow
                            **self._intelligence_metadata,
                        )

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
    ) -> List[Tuple[str, ChunkType]]:
        """
        Split text into chunks with overlap.

        For programming/technical manuals, prefer code-aware chunking (avoid splitting
        indented blocks mid-scope). For other documents, default to sentence-aware.
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
            return [(text, ChunkType.PARAGRAPH)]

        return [(c, ChunkType.PARAGRAPH) for c in self._chunk_by_sentences(text, max_chars, overlap_chars)]

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

    def _chunk_code_by_lines(
        self, code: str, max_chars: int, overlap_lines: int = 4
    ) -> List[str]:
        """
        Chunk code by line boundaries, with a small line overlap for context.
        """
        lines = code.splitlines()
        if not lines:
            return []

        chunks: List[str] = []
        i = 0
        while i < len(lines):
            current: List[str] = []
            cur_len = 0
            while i < len(lines):
                ln = lines[i]
                add_len = len(ln) + (1 if current else 0)
                if current and cur_len + add_len > max_chars:
                    break
                current.append(ln)
                cur_len += add_len
                i += 1
            if current:
                chunks.append("\n".join(current).strip("\n"))
            if i < len(lines):
                i = max(i - overlap_lines, 0)
                # Prevent infinite loop on extremely long single lines.
                if chunks and chunks[-1] == "\n".join(lines[i : i + len(current)]).strip("\n"):
                    i += max(1, overlap_lines)
        return [c for c in chunks if c.strip()]

    def _chunk_mixed_text_and_code(
        self, text: str, max_chars: int, overlap_chars: int
    ) -> List[Tuple[str, ChunkType]]:
        """
        Chunk mixed prose/code by block boundaries.

        - Prose blocks: sentence-aware, then size-limited with overlap.
        - Code blocks: never sentence-split; only split by line boundaries if needed.
        """
        blocks = self._split_blocks_preserve_newlines(text)
        if not blocks:
            return [(text, ChunkType.PARAGRAPH)]

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

        out: List[Tuple[str, ChunkType]] = []
        for block_text, block_type in merged:
            if block_type == ChunkType.CODE:
                if len(block_text) <= max_chars:
                    out.append((block_text, ChunkType.CODE))
                else:
                    for c in self._chunk_code_by_lines(block_text, max_chars=max_chars):
                        out.append((c, ChunkType.CODE))
                continue

            # Prose: chunk within the block, then merge adjacent chunks up to max_chars.
            prose_chunks = self._chunk_by_sentences(block_text, max_chars, overlap_chars)
            for c in prose_chunks:
                out.append((c, ChunkType.PARAGRAPH))

        return [(t, k) for (t, k) in out if t and t.strip()]

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

        # EPUB pre-processing: extract XHTML chapters to a temporary HTML file.
        # Docling 2.66.0 supports HTML but not EPUB natively.
        _tmp_epub_html: Optional[Path] = None
        if file_path.suffix.lower() == ".epub":
            _tmp_epub_html = self._epub_to_html(file_path)
            if _tmp_epub_html:
                logger.info(f"[EPUB] Converted to temporary HTML: {_tmp_epub_html}")
                file_path = _tmp_epub_html
            else:
                raise ValueError(f"Failed to extract HTML from EPUB: {input_path}")

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

        # ================================================================
        # HYBRID CHUNKER PATH (Phase 1 of migration)
        # ================================================================
        # Use Docling's HybridChunker for text elements (sentence-aware
        # splitting, proper heading hierarchy). Process IMAGE/TABLE
        # elements through the existing element-by-element pipeline.
        # ================================================================
        _use_hybrid = True
        _hybrid_text_chunks: List[IngestionChunk] = []
        if _use_hybrid:
            try:
                _hybrid_text_chunks = self._process_text_with_hybrid_chunker(
                    doc=doc,
                    doc_hash=doc_hash,
                    source_file=source_file,
                    file_type=file_type,
                    page_dims=page_dims,
                )
                logger.info(
                    f"[HYBRID-CHUNKER] Produced {len(_hybrid_text_chunks)} text chunks"
                )

                # Phase 2: Apply refiner to HybridChunker text chunks
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

            except Exception as _hc_err:
                logger.warning(
                    f"[HYBRID-CHUNKER] Failed, falling back to element-by-element: {_hc_err}"
                )
                _use_hybrid = False
                _hybrid_text_chunks = []

        # V3.0.0: Pending context queue for IMAGE and TABLE next_text_snippet
        pending_visual_chunks: List[IngestionChunk] = []

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
        # Shadow scan runs AFTER Docling to catch missed large images
        # ================================================================
        logger.info(
            "[SHADOW-EXTRACTION] Running post-Docling shadow scan for missed visual assets..."
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
            docling_processed_pages=set(page_images.keys()),
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
    ) -> List["IngestionChunk"]:
        """Process text elements using Docling's HybridChunker.

        Replaces custom element-by-element text chunking with Docling's
        sentence-boundary-aware chunker. Produces IngestionChunks with
        proper heading hierarchy from the document structure.
        """
        from docling_core.transforms.chunker import HybridChunker

        chunker = HybridChunker(
            tokenizer="sentence-transformers/all-MiniLM-L6-v2",
            max_tokens=350,  # ~1400 chars — keeps chunks under the 1500-char oversize gate
        )
        doc_chunks = list(chunker.chunk(doc))

        chunks: List[IngestionChunk] = []
        for i, dc in enumerate(doc_chunks):
            text = dc.text
            if not text or not text.strip():
                continue

            # Extract page number and bbox from doc_items
            page_no = 1
            bbox: Optional[List[int]] = None
            page_w, page_h = 612.0, 792.0

            if dc.meta and dc.meta.doc_items:
                first_item = dc.meta.doc_items[0]
                if hasattr(first_item, "prov") and first_item.prov:
                    prov = first_item.prov[0] if isinstance(first_item.prov, list) else first_item.prov
                    page_no = getattr(prov, "page_no", 1) or 1
                    prov_bbox = getattr(prov, "bbox", None)
                    if prov_bbox:
                        # Convert Docling bbox to normalized [0, 1000] coords
                        pw, ph = page_dims.get(page_no, (612.0, 792.0))
                        page_w, page_h = pw, ph
                        x0 = int(float(getattr(prov_bbox, "l", 0)) / pw * COORD_SCALE)
                        y0 = int(float(getattr(prov_bbox, "t", 0)) / ph * COORD_SCALE)
                        x1 = int(float(getattr(prov_bbox, "r", pw)) / pw * COORD_SCALE)
                        y1 = int(float(getattr(prov_bbox, "b", ph)) / ph * COORD_SCALE)
                        bbox = [
                            max(0, min(COORD_SCALE, x0)),
                            max(0, min(COORD_SCALE, min(y0, y1))),
                            max(0, min(COORD_SCALE, x1)),
                            max(0, min(COORD_SCALE, max(y0, y1))),
                        ]

            # Extract headings from HybridChunker metadata, filtered through
            # our heading validator (rejects credit lines, copyright, TOC fill)
            from .state.context_state import is_valid_heading
            headings = []
            if dc.meta and dc.meta.headings:
                for h in dc.meta.headings:
                    h_text = h if isinstance(h, str) else getattr(h, "text", str(h))
                    if is_valid_heading(h_text):
                        headings.append(h_text)

            # Build hierarchy
            breadcrumb = [source_file]
            if headings:
                breadcrumb.extend(headings)
            breadcrumb.append(f"Page {page_no}")

            parent_heading = headings[-1] if headings else None

            hierarchy = HierarchyMetadata(
                parent_heading=parent_heading,
                breadcrumb_path=breadcrumb,
                level=min(len(breadcrumb), 5),
            )

            # Determine chunk type
            chunk_type = ChunkType.PARAGRAPH
            label = ""
            if dc.meta and dc.meta.doc_items:
                label = getattr(dc.meta.doc_items[0], "label", "")
            if label == "code":
                chunk_type = ChunkType.CODE
            elif label == "list_item":
                chunk_type = ChunkType.LIST_ITEM
            elif "heading" in label or "title" in label:
                chunk_type = ChunkType.HEADING

            # Create chunk ID
            chunk_id = self._generate_chunk_id(doc_hash, page_no, i)

            # Semantic context (prev/next snippets)
            prev_snippet = chunks[-1].content[-CONTEXT_SNIPPET_LENGTH:] if chunks else None
            semantic_context = SemanticContext(
                prev_text_snippet=prev_snippet,
                next_text_snippet=None,  # filled by lookahead later
                parent_heading=parent_heading,
                breadcrumb_path=breadcrumb,
            )

            chunk = create_text_chunk(
                doc_id=doc_hash,
                content=text.strip(),
                source_file=source_file,
                file_type=file_type,
                page_number=page_no,
                bbox=bbox or [0, 0, COORD_SCALE, COORD_SCALE],
                hierarchy=hierarchy,
                chunk_type=chunk_type,
                page_width=int(page_w),
                page_height=int(page_h),
                extraction_method="hybrid_chunker",
                **self._intelligence_metadata,
            )
            chunk.semantic_context = semantic_context
            # Set refined_content to content (will be updated by refiner if enabled)
            chunk.metadata.refined_content = text.strip()

            chunks.append(chunk)

        # Fill next_text_snippet
        for i in range(len(chunks) - 1):
            chunks[i].semantic_context.next_text_snippet = chunks[i + 1].content[:CONTEXT_SNIPPET_LENGTH]

        return chunks

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

                # IRON-07: If no VLM, DISCARD the asset
                if not self._vision_manager or self._vision_manager is None:
                    logger.warning(
                        f"[FULL-PAGE-GUARD] Page {page_no}: DISCARDING full-page asset "
                        f"(area_ratio={area_ratio:.1%}). No VLM available for verification. "
                        f"Asset: {asset_name}"
                    )
                    print(
                        f"    ⛔ [VLM-GUARD] Discarded full-page asset on page {page_no} "
                        f"(no VLM verification available)",
                        flush=True,
                    )
                    return  # DISCARD - do not save, do not yield chunk

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
                f"DoclingProv={docling_prov_page}",
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
                # V2.4: Intelligence Stack Metadata
                **self._intelligence_metadata,
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
                recovered_table_text = self._extract_docling_table_text(element, page_no)
                if recovered_table_text:
                    table_content = recovered_table_text

            table_extraction_method = "docling"
            docling_markdown = ""
            if table_content and not self._is_table_placeholder_text(table_content, page_no):
                docling_markdown = self._table_text_to_markdown(table_content)
                if self._is_markdown_table(docling_markdown):
                    table_content = docling_markdown
                    table_extraction_method = "docling_table_markdown"

            # Guard against placeholder or text-soup table output:
            # Prefer VLM table serialization, then OCR, then docling markdown fallback.
            vlm_attempted = False
            vlm_table_enabled = (
                self._profile_params.vlm_table_enabled if self._profile_params is not None else True
            )

            # Emergency guardrail: if profile disables table-VLM but we have no usable
            # table text and OCR is unavailable, allow one strict VLM attempt as a
            # last resort to avoid guaranteed table_image_only collapse.
            emergency_vlm_allowed = (
                (not vlm_table_enabled)
                and (not self.enable_ocr)
                and self._is_table_placeholder_text(table_content, page_no)
                and not docling_markdown
                and (saved_table_image is not None)
            )

            if self._force_table_vlm and not vlm_table_enabled:
                logger.info(
                    f"[TABLE-VLM] Page {page_no}: forced VLM table serialization requested but "
                    "disabled by profile. Falling back to OCR/docling path."
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
                        "falling back to OCR/docling path."
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
                    elif docling_markdown:
                        table_content = docling_markdown
                        table_extraction_method = "docling_table_markdown_fallback"

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
            if asset_path is not None:
                table_chunk = create_table_chunk(
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
                    extraction_method=table_extraction_method,
                    # V2.4: Intelligence Stack Metadata
                    **self._intelligence_metadata,
                )
            else:
                # No rendered image available (non-PDF) — emit as TEXT chunk
                table_chunk = create_text_chunk(
                    doc_id=doc_hash,
                    content=table_content,
                    source_file=source_file,
                    file_type=file_type,
                    page_number=page_no,
                    bbox=table_bbox,
                    hierarchy=hierarchy,
                    chunk_type=ChunkType.PARAGRAPH,
                    page_width=int(tbl_page_w),
                    page_height=int(tbl_page_h),
                    extraction_method=table_extraction_method,
                    # V2.4: Intelligence Stack Metadata
                    **self._intelligence_metadata,
                )

            # REQ-MM-03: TABLE CONTEXT PARITY - Initialize semantic context
            if table_chunk.semantic_context is None:
                table_chunk.semantic_context = SemanticContext(
                    prev_text_snippet=prev_text_table,
                    next_text_snippet=None,  # Will be filled by pending queue
                    parent_heading=hierarchy.parent_heading,
                    breadcrumb_path=hierarchy.breadcrumb_path,
                )
            else:
                table_chunk.semantic_context.prev_text_snippet = prev_text_table

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
            or "code" in label_lower  # Programming books: Docling emits code blocks as \"code\"
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

            for i, (chunk_text, chunk_type) in enumerate(text_chunks):
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

                    yield create_text_chunk(
                        doc_id=doc_hash,
                        content=chunk_text,
                        source_file=source_file,
                        file_type=file_type,
                        page_number=page_no,
                        hierarchy=hierarchy,
                        chunk_type=chunk_type,
                        bbox=text_bbox,
                        page_width=int(page_w) if text_bbox and page_w else None,
                        page_height=int(page_h) if text_bbox and page_h else None,
                        prev_text=prev_text_context,
                        next_text=next_text_context,
                        refined_content=refined_content,
                        refinement_applied=refinement_applied,
                        corruption_score=corruption_score,
                        refinement_provider=refinement_provider,
                        refinement_model=refinement_model,
                        content_classification=self._classify_text_content(
                            chunk_text,
                            chunk_type,
                        ),
                        # V2.4: Intelligence Stack Metadata
                        **self._intelligence_metadata,
                    )

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
