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
import re
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

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
    IngestionMetadata,
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
from .engines.pdf_plan import PdfConversionPlan, build_pdf_conversion_plan
# V3.0 UIR contract (Charter §3.2 Phase A step 5): reconciliation paths
# emit IngestionChunks via IngestionChunk.from_uir(UIRChunk(...)). Aliases
# avoid clash with the Pydantic v2.16 names imported above.
from .universal.intermediate import (
    ConfidenceBreakdown as UIRConfidenceBreakdown,
    CoordinateFrame as UIRCoordinateFrame,
    Locator as UIRLocator,
    LocatorType as UIRLocatorType,
    UIRChunk,
)
from .vision.vision_manager import VisionManager, create_vision_manager
from .vision.vision_prompts import validate_vlm_response
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
# HELPERS
# ============================================================================

# Keywords that, when two or more appear on a single long line inside a fenced
# code block, indicate the line is a squished (newline-stripped) code body.
_FENCED_FLAT_KEYWORDS: tuple = (
    "def ", "class ", "return ", "self.", "import ", "raise ", "if ", "for ",
)


_CODE_EVIDENCE_KEYWORDS: tuple = (
    "def ", "class ", "import ", "from ", "return ", "yield ",
)
# Minimum fence markers in the page sample to qualify as code-heavy.
_CODE_FENCE_THRESHOLD: int = 5
# Minimum weighted code-line ratio in the page sample.
_CODE_RATIO_THRESHOLD: float = 0.10

# --- PLAN_F1 4.1: text_native_code page signal -----------------------------
# Font-INDEPENDENT born-digital code-page signal. RECALIBRATED after the WP-4
# spike (PLAN_F1 1.1 / report): the original keyword-START-fraction + leading-
# whitespace-depth channels were calibrated on SYNTHETIC fixtures and fired on 0
# of 39 real Jungjun code pages and ~0 Chaubal code pages, because (a) real PDFs
# encode indentation as x-position or non-breaking spaces that ``get_text`` does
# NOT surface as regular leading whitespace, and (b) the def/class/import-only
# keyword set misses body-heavy code (loops, calls, assignments). The robust
# discriminator is the CODE-LINE RATIO: the fraction of non-blank lines that look
# like code statements (keyword headers, assignments, or call expressions). This
# fires on real code pages (Chaubal/Jungjun ~0.5-0.8) and stays ~0 on the frozen
# Workstream B negatives (prose, magazines, incidental shell, poetry, nested
# lists), which carry indentation but not code syntax. Fenced code still qualifies
# via the existing threshold. A real text layer (>= _TEXT_NATIVE_MIN_CHARS) is a
# precondition: the signal gates the Mechanism-B text-layer patch, meaningless on
# raster pages. The over-trigger contract is the Workstream B negative set.
_TEXT_NATIVE_MIN_CHARS: int = 100
_TEXT_NATIVE_CODE_RATIO_MIN: float = 0.40

_TN_KEYWORDS: tuple = (
    "def ", "class ", "import ", "from ", "return", "yield", "for ", "while ",
    "if ", "elif ", "else", "try", "except", "finally", "with ", "async ",
    "await ", "raise ", "print(", "assert ", "lambda ", "@",
)
_TN_ASSIGN = re.compile(r"[^=!<>]=[^=]")       # a plain/augmented assignment, not ==/!=/<=/>=
_TN_CALL = re.compile(r"[A-Za-z_]\w*\(")        # function/method call


def _tn_is_code_line(stripped: str) -> bool:
    """True if a stripped line looks like a code statement (not prose/list/kv)."""
    if any(stripped.startswith(k) for k in _TN_KEYWORDS):
        return True
    if _TN_ASSIGN.search(stripped):
        return True
    if _TN_CALL.search(stripped):
        return True
    if stripped.endswith(":") and "(" in stripped:  # suite header
        return True
    return False


# --- PLAN_F1 Phase 1 residual-defect fixes (user J1, 2026-06-12) -----------
# Three real extraction defects surfaced by the Jungjun oracle (all ORTHOGONAL to
# indentation): (b) smart quotes used as code delimiters, (c) single/double/f-strings
# hard-wrapped across printed-source lines (illegal in Python -> parse fail), and
# (a) code blocks cut mid-docstring across a chunk boundary. Fixes are conservative
# and apply to CODE chunks only.
_SMART_QUOTES = {
    "“": '"', "”": '"', "″": '"',   # " " ”  -> "
    "‘": "'", "’": "'", "′": "'",   # ' ' ′  -> '
}
_SMART_QUOTE_TABLE = str.maketrans(_SMART_QUOTES)


def _normalize_code_quotes(text: str) -> str:
    """(b) Replace typographic/smart quotes with ASCII quotes (code chunks only)."""
    return (text or "").translate(_SMART_QUOTE_TABLE)


def _scan_code_line(line: str, triple: "Optional[str]") -> "Tuple[Optional[str], bool]":
    """Scan one line starting inside triple-string ``triple`` (or None).

    Returns ``(end_triple, nontriple_string_open)``: the triple-delimiter still
    open at line end (carried to the next line - legal multiline docstring), and
    whether a SINGLE/DOUBLE/f-string was left open at line end (an illegal
    hard-wrap that must be rejoined). ``#`` comments outside strings end scanning.
    """
    i, n = 0, len(line)
    s = triple  # active string delimiter (triple, or single/double, or None)
    while i < n:
        if s in ('"""', "'''"):
            if line[i:i + 3] == s:
                s = None
                i += 3
                continue
            i += 1
            continue
        if s in ("'", '"'):
            if line[i] == "\\":
                i += 2
                continue
            if line[i] == s:
                s = None
                i += 1
                continue
            i += 1
            continue
        # not in a string
        c = line[i]
        if c == "#":
            break
        if line[i:i + 3] in ('"""', "'''"):
            s = line[i:i + 3]
            i += 3
            continue
        if c in ("'", '"'):
            s = c
            i += 1
            continue
        i += 1
    if s in ('"""', "'''"):
        return s, False           # still inside a (legal) triple-quoted block
    if s in ("'", '"'):
        return None, True         # non-triple string left open -> wrapped line
    return None, False


def _rejoin_wrapped_code_lines(text: str) -> str:
    """(c) Conservatively rejoin lines where a NON-triple string is left open.

    A bare ``"abc`` / ``'abc`` at a line end is always a Python syntax error - it
    is a printed-source hard-wrap. Join it with following line(s) (word-wrap
    convention: single space) until the string closes. Triple-quoted docstrings
    that legally span lines are NOT touched. Open brackets alone are legal
    multiline and are left as-is (conservative: only rejoin what breaks parse).
    """
    raw = (text or "").split("\n")
    out: "List[str]" = []
    triple: "Optional[str]" = None
    i = 0
    while i < len(raw):
        line = raw[i]
        while True:
            end_triple, nontriple_open = _scan_code_line(line, triple)
            if nontriple_open and i + 1 < len(raw):
                line = line + " " + raw[i + 1].lstrip()
                i += 1
                continue
            break
        triple = end_triple
        out.append(line)
        i += 1
    return "\n".join(out)


def _code_bracket_depth(line: str) -> int:
    """Net unclosed ([{ depth at end of a line, ignoring strings and # comments."""
    d = 0
    s = None
    i = 0
    n = len(line)
    while i < n:
        if s:
            if s in ("'", '"') and line[i] == "\\":
                i += 2
                continue
            if s in ('"""', "'''") and line[i:i + 3] == s:
                s = None
                i += 3
                continue
            if s in ("'", '"') and line[i] == s:
                s = None
            i += 1
            continue
        c = line[i]
        if c == "#":
            break
        if line[i:i + 3] in ('"""', "'''"):
            s = line[i:i + 3]
            i += 3
            continue
        if c in ("'", '"'):
            s = c
        elif c in "([{":
            d += 1
        elif c in ")]}":
            d = max(0, d - 1)
        i += 1
    return d


def _rejoin_open_brackets(text: str) -> str:
    """Collapse open-bracket line continuations (no separator: code wraps split
    mid-token, e.g. ``request.tool`` + ``s``). REPAIR-ONLY use: collapsing legal
    multi-line calls is non-conservative, so only apply when the chunk does not
    already parse and keep the result only if it then parses (see _repair_code_content).
    """
    raw = text.split("\n")
    out: "List[str]" = []
    i = 0
    while i < len(raw):
        line = raw[i]
        while _code_bracket_depth(line) > 0 and i + 1 < len(raw):
            line = line + raw[i + 1].lstrip()
            i += 1
        out.append(line)
        i += 1
    return "\n".join(out)


def _repair_code_content(text: str) -> str:
    """PLAN_F1 J1 (b)+(c): normalize smart quotes, rejoin open-string hard-wraps,
    and (only if still unparseable) rejoin open-bracket wraps - keeping the bracket
    rejoin solely when it makes the chunk parse, so a parseable chunk is never
    degraded. No-op on already-clean code (idempotent)."""
    fixed = _rejoin_wrapped_code_lines(_normalize_code_quotes(text or ""))
    try:
        import ast as _ast
        _ast.parse(fixed)
        return fixed
    except (SyntaxError, ValueError):
        pass
    candidate = _rejoin_open_brackets(fixed)
    try:
        import ast as _ast
        _ast.parse(candidate)
        return candidate
    except (SyntaxError, ValueError):
        return fixed


def _leaves_docstring_open(text: str) -> bool:
    """True if ``text`` ends inside an unterminated triple-quoted string (a code
    block cut mid-docstring across a chunk boundary - PLAN_F1 J1 (a))."""
    triple = None
    for line in (text or "").split("\n"):
        triple, _ = _scan_code_line(line, triple)
    return triple in ('"""', "'''")


def _score_text_native_code(page_text: str) -> "Tuple[bool, dict]":
    """Decide whether a page is born-digital code from its text-layer text alone.

    Returns ``(is_text_native_code, channels)``. Font-independent by construction
    (P1 is font-blind) and indentation-encoding-independent (uses code SYNTAX, not
    leading whitespace, which ``get_text`` strips for x-positioned indentation).
    """
    text = (page_text or "").replace("\xa0", " ")  # normalize nbsp -> space
    lines = [ln for ln in text.splitlines() if ln.strip()]
    n = len(lines)
    channels = {
        "chars": len(text.strip()),
        "lines": n,
        "code_ratio": 0.0,
        "kw_starts": 0,
        "fence": 0,
        "depths": 0,
    }
    if n == 0 or channels["chars"] < _TEXT_NATIVE_MIN_CHARS:
        return False, channels

    code_lines = sum(1 for ln in lines if _tn_is_code_line(ln.strip()))
    fence = sum(1 for ln in lines if ln.lstrip().startswith(("```", "~~~")))
    kw = sum(1 for ln in lines if any(ln.lstrip().startswith(k) for k in _CODE_EVIDENCE_KEYWORDS))
    depths = len({len(ln) - len(ln.lstrip(" \t")) for ln in lines if len(ln) - len(ln.lstrip(" \t")) > 0})

    channels["code_ratio"] = round(code_lines / n, 3)
    channels["kw_starts"] = kw
    channels["fence"] = fence
    channels["depths"] = depths

    dense_code = channels["code_ratio"] >= _TEXT_NATIVE_CODE_RATIO_MIN
    fenced_code = fence >= _CODE_FENCE_THRESHOLD
    return bool(dense_code or fenced_code), channels


def _select_code_evidence_sample_indices(total_pages: int) -> "List[int]":
    """Return page indices for cheap code-evidence sampling.

    Programming books often introduce dense code after front matter but before
    evenly spaced samples would land on it.  Sample the early body more densely,
    then add a light spread across the whole document for later-code controls.
    """
    if total_pages <= 0:
        return []

    indices: set = set()

    early_window = min(total_pages, 120)
    early_step = max(1, early_window // 30)
    for idx in range(0, early_window, early_step):
        indices.add(idx)

    spread_size = min(15, total_pages)
    spread_step = max(1, total_pages // spread_size)
    for idx in range(0, total_pages, spread_step):
        indices.add(idx)

    return sorted(i for i in indices if 0 <= i < total_pages)


def _score_code_evidence(
    page_texts: "List[str]",
) -> "Tuple[bool, str, float, int, int]":
    """Score code evidence from sampled page-text strings.

    Counts three lightweight signals without running any ML model:

    * ``fence_count``    — lines starting with ``` or ~~~
    * ``keyword_lines`` — lines containing Python/code keywords
    * ``indented_lines`` — indented lines counted only after code context exists

    A ratio is formed as ``(keyword_lines + contextual_indented_lines) / total_lines``.
    Fence markers are a separate threshold, not ratio ballast; otherwise two short
    shell-command blocks in a prose-heavy manual can look code-heavy.

    Thresholds (module constants, overridable in tests):

    * ``fence_count >= _CODE_FENCE_THRESHOLD`` — substantial fenced code
    * ``code_ratio >= _CODE_RATIO_THRESHOLD`` with code context — code-dense text

    Returns:
        (needs_enrichment, reason, code_ratio, fence_count, keyword_line_count)
    """
    fence_count = 0
    keyword_lines = 0
    indented_lines = 0
    total_lines = 0

    for text in page_texts:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            total_lines += 1
            if stripped.startswith("```") or stripped.startswith("~~~"):
                fence_count += 1
            elif any(stripped.startswith(kw) for kw in _CODE_EVIDENCE_KEYWORDS):
                # startswith avoids false positives from English prose, e.g.
                # "range from 15°C" → "from" mid-sentence should not count.
                keyword_lines += 1
            elif line.startswith("    ") or line.startswith("\t"):
                indented_lines += 1

    if total_lines == 0:
        return False, "no text content in sample", 0.0, 0, 0

    has_code_context = fence_count > 0 or keyword_lines >= 2
    contextual_indented = indented_lines if has_code_context else 0
    code_ratio = (keyword_lines + contextual_indented) / total_lines

    if fence_count >= _CODE_FENCE_THRESHOLD:
        reason = (
            f"fence_count={fence_count} (>= {_CODE_FENCE_THRESHOLD}); "
            f"code_ratio={code_ratio:.3f}"
        )
        return True, reason, code_ratio, fence_count, keyword_lines

    if has_code_context and code_ratio >= _CODE_RATIO_THRESHOLD:
        reason = (
            f"code_ratio={code_ratio:.3f} (>= {_CODE_RATIO_THRESHOLD}); "
            f"fences={fence_count} keywords={keyword_lines} indented={contextual_indented}"
        )
        return True, reason, code_ratio, fence_count, keyword_lines

    reason = (
        f"below threshold: fence={fence_count}(<{_CODE_FENCE_THRESHOLD}) "
        f"code_ratio={code_ratio:.3f}(<{_CODE_RATIO_THRESHOLD})"
    )
    return False, reason, code_ratio, fence_count, keyword_lines


def _decide_code_evidence_from_pages(
    page_texts: "List[str]",
) -> "Tuple[bool, str, float, int, int]":
    """Decide document-level code enrichment from sampled page texts.

    A whole-document aggregate can dilute dense code pages with prose pages.
    Count strong sampled pages first; fall back to the aggregate for documents
    where code is distributed across many pages.
    """
    strong_pages = 0
    max_score = 0.0
    total_fences = 0
    total_keywords = 0

    for text in page_texts:
        page_needs, _, page_score, page_fences, page_keywords = _score_code_evidence([text])
        if page_needs:
            strong_pages += 1
        max_score = max(max_score, page_score)
        total_fences += page_fences
        total_keywords += page_keywords

    if strong_pages >= 2:
        return (
            True,
            f"strong_code_pages={strong_pages}; max_page_score={max_score:.3f}",
            max_score,
            total_fences,
            total_keywords,
        )

    aggregate_needs, aggregate_reason, aggregate_score, aggregate_fences, aggregate_keywords = (
        _score_code_evidence(page_texts)
    )
    return aggregate_needs, aggregate_reason, aggregate_score, aggregate_fences, aggregate_keywords


def decide_code_enrichment_for_pdf(
    pdf_path: "Path",
    config: "Optional[Any]" = None,
) -> "Tuple[bool, str, float]":
    """Return the code-enrichment decision for a PDF without mutating a processor.

    Used by both BatchProcessor and the direct V2DocumentProcessor CLI path so
    the decision lane does not diverge by entry point.
    """
    if config is None:
        return False, "disabled: code_enrichment config not registered", 0.0
    if not getattr(config, "enabled", False):
        return False, "disabled by code_enrichment.enabled=false in config", 0.0

    import fitz  # type: ignore[import]

    doc = fitz.open(str(pdf_path))
    try:
        total_pages = len(doc)
        sample_indices = _select_code_evidence_sample_indices(total_pages)

        page_texts: List[str] = []
        for idx in sample_indices:
            try:
                page_texts.append(doc.load_page(idx).get_text("text"))
            except Exception:
                pass
    finally:
        doc.close()

    needs, reason, score, _fence_count, _keyword_lines = _decide_code_evidence_from_pages(
        page_texts
    )
    return needs, reason, score


def _has_fenced_flat_code(txt: str) -> bool:
    """Return True if *txt* contains a fenced code block with a flat body line.

    PROVISIONAL FALLBACK — used only when Docling native code enrichment
    (`do_code_enrichment=True`) is unavailable or still returns flat code.
    Prefer Docling CodeItem/CodeFormulaV2 output when available.

    Catches code chunks produced when a PDF generator strips internal newlines
    from code blocks, leaving multiple statements on one line inside backtick
    fences, e.g.::

        ```python
        class CreditCard(PaymentBase): def process_payment(self): print(msg)
        ```

    A body line qualifies when it is inside a fence, exceeds 120 characters,
    and matches at least two Python-code keyword patterns.
    """
    in_fence = False
    for line in txt.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence and len(stripped) > 120:
            hits = sum(1 for kw in _FENCED_FLAT_KEYWORDS if kw in stripped)
            if hits >= 2:
                return True
    return False


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
        # Cluster B: last active heading carried across batch boundaries so a
        # chapter heading propagates into a later batch whose chunks have no
        # heading of their own (reset per document in process_pdf).
        self._carry_heading: Optional[str] = None
        self._carry_breadcrumb: Optional[List[str]] = None

        # REQ-OCR-01: Profile parameters for OCR hints and dynamic DPI
        self._profile_params: Optional["ProfileParameters"] = None
        self._layout_processor = None

        # V2.4: Intelligence Stack Metadata (for observability)
        self._intelligence_metadata: Dict[str, Any] = {}
        self._conversion_plan: Optional[PdfConversionPlan] = None

        # v2.5.0: Structural pathology flags (REQ-STRUCT-01/02/03)
        # Stored as dedicated instance vars so they are NOT unpacked into
        # chunk creation functions via **self._intelligence_metadata.
        self.has_flat_text_corruption: bool = False
        self.has_encoding_corruption: bool = False
        self.geometry_error_rate: float = 0.0
        self._doc_total_pages: Optional[int] = None
        self._doc_image_density: Optional[float] = None
        self._doc_avg_text_per_page: Optional[float] = None

        # Workstream B: code enrichment decision flag and telemetry.
        # Set by _decide_code_enrichment() in process_pdf() from a cheap
        # fitz pre-pass; NOT inferred from has_encoding_corruption alone.
        self.needs_code_enrichment: bool = False
        self._code_enrichment_reason: str = ""
        self._code_enrichment_score: float = 0.0
        # CodeEnrichmentConfig from app config; None = enrichment disabled.
        self._code_enrichment_config: "Optional[Any]" = None

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

        # v2.9 Phase 1: per-document monotonic chunk counter feeding the
        # ``position`` argument of ``create_*_chunk`` factories so two chunks
        # with byte-identical (page, modality, content) get distinct chunk_ids.
        # Reset at the top of process_pdf().
        self._chunk_position: int = 0

        logger.info(
            f"BatchProcessor initialized: "
            f"batch_size={batch_size}, "
            f"vision={vision_provider}/{vision_model}, "
            f"timeout={vlm_timeout}s, "
            f"max_pages={max_pages if max_pages else 'ALL'}"
        )

    def _next_chunk_position(self) -> int:
        """Allocate the next per-document chunk position (v2.9 Phase 1)."""
        pos = self._chunk_position
        self._chunk_position = pos + 1
        return pos

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

    def set_conversion_plan(self, plan: PdfConversionPlan) -> None:
        """Set the shared PDF conversion plan for this batch run."""
        self._conversion_plan = plan
        self._docling_converter = None

        self.has_flat_text_corruption = plan.has_flat_text_corruption
        self.has_encoding_corruption = plan.has_encoding_corruption
        self.geometry_error_rate = plan.geometry_error_rate
        self._doc_total_pages = plan.total_pages
        self._doc_image_density = plan.image_density
        self._doc_avg_text_per_page = plan.avg_text_per_page
        self.needs_code_enrichment = plan.needs_code_enrichment
        self._code_enrichment_reason = plan.code_enrichment_reason
        self._code_enrichment_score = plan.code_enrichment_score
        self._extraction_route = plan.extraction_route
        self._hybrid_chunker_enabled = plan.hybrid_chunker_enabled
        self._max_chunker_input_chars = plan.max_chunker_input_chars
        self._max_chunker_per_element_chars = plan.max_chunker_per_element_chars
        self._allow_page_level_visuals = plan.allow_page_level_visuals
        self._drop_blank_assets = plan.drop_blank_assets
        self._quarantine_corrupted_chunks = plan.quarantine_corrupted_chunks
        self._suppress_layout_label_text = plan.suppress_layout_label_text
        self._intelligence_metadata = plan.chunk_factory_metadata()

        # v2.13 Phase 2: when the plan requests force_full_page_ocr (scanned
        # profiles), route through the Docling-direct (legacy) OCR path so
        # the flag actually reaches Docling. The layout-aware OCR mode runs
        # its own per-region OCR via LayoutAwareOCRProcessor / EnhancedOCREngine
        # and does NOT consult the Docling adapter's PdfPipelineOptions.
        # For scanned multi-column docs (Earthship), layout-aware misjudges
        # column boundaries and produces interleaved text; full-page OCR via
        # Docling fixes it. The Phase-6 _promote_ocr_section_headers fallback
        # path preserves heading attribution in legacy mode.
        if (
            getattr(plan, "force_full_page_ocr", False)
            and getattr(self, "ocr_mode", "legacy") == "layout-aware"
            and getattr(self, "enable_ocr", True)
        ):
            logger.info(
                "[OCR-GOVERNANCE] plan.force_full_page_ocr=True; "
                "overriding ocr_mode 'layout-aware' -> 'legacy' so "
                "Docling's force_full_page_ocr setting is honored. "
                "Heading attribution falls back to _promote_ocr_section_headers."
            )
            self.ocr_mode = "legacy"

        logger.info(
            f"[PDF-PLAN] Conversion plan set: profile={plan.profile_type}, "
            f"modality={plan.document_modality}, ocr={plan.do_ocr}, "
            f"code_enrich={plan.needs_code_enrichment}, "
            f"encoding_corrupt={plan.has_encoding_corruption}, "
            f"route={plan.extraction_route}, "
            f"reading_order={plan.reading_order_strategy}, "
            f"force_full_page_ocr={getattr(plan, 'force_full_page_ocr', False)}, "
            f"effective_ocr_mode={getattr(self, 'ocr_mode', 'legacy')}"
        )

    def _build_legacy_conversion_plan(self) -> PdfConversionPlan:
        """Build a plan from legacy BatchProcessor state for compatibility."""
        return build_pdf_conversion_plan(
            enable_ocr=self.enable_ocr,
            ocr_engine=self.ocr_engine,
            force_table_vlm=self.force_table_vlm,
            needs_code_enrichment=self.needs_code_enrichment,
            code_enrichment_reason=self._code_enrichment_reason,
            code_enrichment_score=self._code_enrichment_score,
            has_encoding_corruption=self.has_encoding_corruption,
            has_flat_text_corruption=self.has_flat_text_corruption,
            geometry_error_rate=self.geometry_error_rate,
            total_pages=self._doc_total_pages or 0,
            image_density=self._doc_image_density or 0.0,
            avg_text_per_page=self._doc_avg_text_per_page or 0.0,
            **self._intelligence_metadata,
        )

    def _ensure_conversion_plan(self) -> PdfConversionPlan:
        """Ensure a conversion plan exists for profile/intelligence metadata.

        V3.0: batch_processor no longer constructs a Docling adapter — extraction
        is delegated to the engine-agnostic mmrag_v3 router. The plan is retained
        only as the carrier of profile/policy metadata for chunk emission.
        """
        if self._conversion_plan is None:
            self.set_conversion_plan(self._build_legacy_conversion_plan())
        return self._conversion_plan

    def enable_code_enrichment(self, config: "Any") -> None:
        """Store CodeEnrichmentConfig; gates _decide_code_enrichment in process_pdf.

        If config.enabled is False, code enrichment is skipped even when evidence
        suggests a code-heavy document.  This is the primary production guard against
        accidental CPU-bound CodeFormulaV2 inference.
        """
        self._code_enrichment_config = config
        logger.info(
            f"[CODE-ENRICH] Config registered: enabled={config.enabled} "
            f"model={getattr(config, 'model', '?')}"
        )

    def _decide_code_enrichment(self, pdf_path: "Path") -> None:
        """Run a cheap fitz pre-pass to decide if Docling code enrichment should fire.

        Samples early-body pages plus a spread across the document, scores
        code-evidence indicators, and sets ``self.needs_code_enrichment``.  The
        decision is written to logs with counts so evidence stays separate from
        the fallback Tesseract rescue path.

        Guards:
        - If ``_code_enrichment_config`` is missing or disabled, skips immediately.
        - Never infers from ``has_encoding_corruption`` alone (see DECISIONS.md).
        """
        try:
            needs, reason, score = decide_code_enrichment_for_pdf(
                pdf_path, self._code_enrichment_config
            )
            self.needs_code_enrichment = needs
            self._code_enrichment_reason = reason
            self._code_enrichment_score = score

            log_fn = logger.info if needs else logger.debug
            log_fn(
                f"[CODE-ENRICH-DECISION] needs={needs} | {reason}"
            )

        except Exception as exc:
            logger.debug(
                f"[CODE-ENRICH-DECISION] pre-pass failed (non-fatal): {exc}"
            )
            self.needs_code_enrichment = False
            self._code_enrichment_reason = f"pre-pass error: {exc}"

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

        # Pre-sanitize: strip null bytes from content BEFORE Pydantic serialization.
        # Pydantic's model_dump can crash with "source code string cannot contain
        # null bytes" if \x00 is present in any string field.
        _ctrl_table = {c: None for c in range(32) if c not in (10, 9)} | {127: None}
        if hasattr(chunk, "content") and chunk.content and "\x00" in chunk.content:
            chunk.content = chunk.content.translate(_ctrl_table)

        chunk_dict = chunk.model_dump(mode="json")

        # Strip remaining control characters (U+0001-U+001F except \n \t, plus U+007F)
        # from serialized content. Catches chars that survived from encoding-corrupted
        # PDF text layers.
        content = chunk_dict.get("content")
        if content and isinstance(content, str):
            cleaned = content.translate(_ctrl_table)
            if cleaned != content:
                chunk_dict["content"] = cleaned

        # Sanitize heading hierarchy: reject long headings that are clearly
        # misclassified paragraphs/tables from Docling's layout model.
        hierarchy = chunk_dict.get("metadata", {}).get("hierarchy")
        if hierarchy:
            ph = hierarchy.get("parent_heading")
            if ph and (len(ph) > 80 or ph.count(". ") > 1):
                hierarchy["parent_heading"] = None
            # Also clean breadcrumb entries
            bp = hierarchy.get("breadcrumb_path", [])
            hierarchy["breadcrumb_path"] = [
                b for b in bp if not isinstance(b, str) or len(b) <= 80
            ]

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

    @staticmethod
    def _phash_carve_out_should_preserve_duplicate(
        page_number: Optional[int],
        image_only_pages: set,
        pages_with_exported_image: set,
    ) -> bool:
        """PLAN_V2.10 Phase 3: should a duplicate IMAGE chunk on the
        given page be PRESERVED (rather than dropped by the pHash
        registry) because the page would otherwise have no surviving
        content?

        The carve-out fires when ALL hold:
          - `page_number` is known
          - the page has no surviving TEXT/TABLE chunks
            (i.e. `page_number in image_only_pages`)
          - the page has not yet exported any IMAGE chunk in this
            run — INCLUDING unique images, not just preserved
            duplicates. Once an image (unique or preserved) is
            written for the page, subsequent near-duplicates on
            the same page still drop because the page is covered.

        Returns True ⇒ preserve. The caller is then responsible for
        adding `page_number` to `pages_with_exported_image` so a
        second near-duplicate on the same page does not also
        preserve.
        """
        if page_number is None:
            return False
        if page_number not in image_only_pages:
            return False
        if page_number in pages_with_exported_image:
            return False
        return True

    @staticmethod
    def _is_blank_asset(path: Path) -> bool:
        """Check if an image asset is blank (all white/all black/empty).

        Returns True if the image has low variance (std < 10) and mean
        is near 0 or 255, indicating a blank/empty rendered asset.

        Plan v2.9 Phase E (2026-05-11): the std threshold was relaxed
        from < 5 to < 10 to catch Combat_Aircraft_August_2025 p27
        `figure_36` (mean=253, std=7.4) which has a faint watermark or
        compression noise above the prior std<5 cap. Validated on 10
        random Combat assets: all real-content assets have std ≥ 49,
        so the wider band has zero FP on the audited sample.
        """
        try:
            from PIL import Image
            import numpy as np

            with Image.open(path) as img:
                arr = np.array(img.convert("L"))  # Grayscale
                std = float(arr.std())
                mean = float(arr.mean())
                if std < 10.0 and (mean > 250.0 or mean < 5.0):
                    logger.debug(
                        f"[BLANK-ASSET] {path.name}: mean={mean:.1f}, std={std:.1f} → blank"
                    )
                    return True
        except Exception as e:
            logger.debug(f"[BLANK-ASSET] Could not check {path}: {e}")
        return False

    def _quarantine_corrupted_text_chunks(
        self, chunks: List[IngestionChunk]
    ) -> List[IngestionChunk]:
        """Remove text chunks that still contain encoding artifacts.

        Called after patch_corrupted_chunks(); any remaining corrupted text
        must not reach final JSONL.  Non-TEXT chunks are never quarantined.

        PLAN_V2.10 Phase 2 (2026-05-12): switched from
        `has_encoding_artifacts` (single-match) to `is_irreparably_corrupt`
        (artifact-ratio + OCR-failure runs). Single-match was over-firing
        on Fluent Python's encodings chapter where pp125/126/136 contain
        Python REPL output explaining UTF-8 byte literals like
        ``bytearray(b'caf\\xc3\\xa9')`` — a chunk with one or two
        ``\\xHH`` literal escapes in ~1800 chars of legitimate prose
        registered as "corruption" and was being quarantined post-scout.
        The ratio-based detector keeps the original corpus contract intact
        (Combat p66 squadron-roster /C211 + em-dash runs still drop;
        CRONIN-style /uniFB01 + replacement-char clusters still drop)
        while sparing low-density legitimate escapes.
        """
        if not getattr(self, "_quarantine_corrupted_chunks", True):
            return chunks
        from .validators.corruption_interceptor import is_irreparably_corrupt

        clean = []
        quarantined = 0
        for c in chunks:
            if not (c.modality == Modality.TEXT and c.content):
                clean.append(c)
                continue
            extraction_method = getattr(
                getattr(c, "metadata", None), "extraction_method", None
            )
            # Phase 1 dense-index router output is structured TOC/index text
            # extracted via the page-skip grid traversal. Source fonts in
            # these tables often lack ToUnicode mappings for dotted-leader
            # `.` glyphs, leaving U+FFFD (`�`) replacement characters in the
            # content. has_encoding_artifacts() correctly flags those as
            # corruption signatures, but quarantining the chunk discards the
            # entire TOC entry. The Phase 1 router is the trusted producer
            # for these chunks; exempt them. (Plan v2.9 B1, 2026-05-11.)
            if extraction_method and extraction_method.startswith(
                "hybrid_chunker_pageskip"
            ):
                clean.append(c)
                continue
            if is_irreparably_corrupt(c.content):
                quarantined += 1
            else:
                clean.append(c)
        if quarantined:
            logger.warning(
                f"[CORRUPTION-QUARANTINE] Dropped {quarantined} text chunks "
                f"with unrepairable encoding artifacts"
            )
        return clean

    def _filter_blank_assets(
        self, chunks: List[IngestionChunk]
    ) -> List[IngestionChunk]:
        """Drop/promote chunks whose image asset is blank.

        IMAGE chunks with blank assets are dropped entirely.
        TABLE chunks with usable markdown content are promoted to TEXT
        modality (IMAGE/TABLE require asset_ref per schema contract).
        """
        if not getattr(self, "_drop_blank_assets", True):
            return chunks
        surviving: List[IngestionChunk] = []
        dropped = 0
        promoted = 0
        for c in chunks:
            asset_ref = getattr(c, "asset_ref", None)
            asset_path = getattr(asset_ref, "file_path", None) if asset_ref else None
            if asset_path and c.modality in (Modality.IMAGE, Modality.TABLE):
                full_path = self.output_dir / asset_path
                if full_path.exists() and self._is_blank_asset(full_path):
                    dropped += 1
                    logger.warning(f"[BLANK-ASSET] Dropping blank asset: {asset_path}")
                    try:
                        full_path.unlink()
                    except Exception:
                        pass
                    # Promote TABLE with markdown to TEXT so content survives.
                    if (
                        c.modality == Modality.TABLE
                        and c.content
                        and len(c.content.strip()) > 20
                    ):
                        c.modality = Modality.TEXT
                        c.asset_ref = None
                        if c.metadata:
                            c.metadata.chunk_type = ChunkType.PARAGRAPH
                        surviving.append(c)
                        promoted += 1
                        logger.info(
                            "[BLANK-ASSET] Table promoted to TEXT "
                            "(markdown content preserved)"
                        )
                    # IMAGE or tiny TABLE — drop entirely.
                    continue
            surviving.append(c)
        if dropped:
            logger.info(
                f"[FINALIZE] Dropped {dropped} blank assets "
                f"({promoted} tables promoted to TEXT, "
                f"{dropped - promoted} chunks removed)"
            )
        return surviving

    def _filter_tiny_icon_images(
        self, chunks: List[IngestionChunk]
    ) -> List[IngestionChunk]:
        """Drop icon/glyph-class IMAGE chunks (sub-content regions).

        The V3 path emits every detected image region; the geometric crop (B1)
        can isolate a tiny embedded raster - a page icon, bullet glyph, small
        logo, or inline mark - that is not retrievable content and only adds
        IMAGE_NO_VLM / ASSET_TINY noise. An image is icon-class when its
        RENDERED asset is small in BOTH dimensions AND has a tiny file. The bbox
        is NOT used: a large (often hallucinated) VLM bbox can resolve to a
        23x23px geometric crop, so the rendered asset is the only reliable size
        signal. Triple-AND is conservative - a small real figure survives via
        one larger dimension or a detailed (>=1.5KB) file. Assets render at
        DEFAULT_CROP_ZOOM=2.0, so <96px == <48pt in source space.

        A page-coverage guard never drops the only content on a page, so an
        image-only page cannot become MISSING_PAGES.
        """
        if not getattr(self, "_drop_tiny_icon_images", True):
            return chunks
        from PIL import Image

        MAX_ICON_PX = 96
        MAX_ICON_BYTES = 1500

        def _page(c: IngestionChunk) -> Optional[int]:
            return c.metadata.page_number if c.metadata else None

        icon_ids: set[int] = set()
        info: Dict[int, Tuple[str, int, int, int, Path]] = {}
        for c in chunks:
            if c.modality != Modality.IMAGE:
                continue
            asset_ref = getattr(c, "asset_ref", None)
            asset_path = getattr(asset_ref, "file_path", None) if asset_ref else None
            if not asset_path:
                continue
            full = self.output_dir / asset_path
            if not full.exists():
                continue
            try:
                sz = full.stat().st_size
                with Image.open(full) as im:
                    w, h = im.size
            except Exception:
                continue
            if w < MAX_ICON_PX and h < MAX_ICON_PX and sz < MAX_ICON_BYTES:
                icon_ids.add(id(c))
                info[id(c)] = (asset_path, w, h, sz, full)

        if not icon_ids:
            return chunks

        # Page-coverage guard: keep an icon-class image when it is the only
        # content on its page (do not manufacture a MISSING_PAGES failure).
        pages_with_other = {
            _page(c)
            for c in chunks
            if id(c) not in icon_ids and _page(c) is not None
        }

        surviving: List[IngestionChunk] = []
        dropped = 0
        for c in chunks:
            if id(c) in icon_ids and _page(c) in pages_with_other:
                asset_path, w, h, sz, full = info[id(c)]
                dropped += 1
                logger.info(
                    f"[TINY-ICON] Dropping icon-class image {asset_path} "
                    f"({w}x{h}px, {sz}B)"
                )
                try:
                    full.unlink()
                except Exception:
                    pass
                continue
            surviving.append(c)
        if dropped:
            logger.info(
                f"[FINALIZE] Dropped {dropped} icon-class image chunk(s) "
                f"(sub-content regions)"
            )
        return surviving

    def _promote_or_drop_empty_tables(
        self, chunks: List[IngestionChunk]
    ) -> List[IngestionChunk]:
        """Drop empty-content TABLE chunks; promote the page's-only-content case.

        A TABLE carries its value in the markdown grid; an empty-content table
        (DOCLING_FAST/offline layout miss) is a corrupt placeholder that fails
        the table-format gate. Online extraction always yields markdown, so this
        only culls genuine empties.

        Page-coverage guard (composes with _filter_tiny_icon_images, which runs
        earlier in the export chain): never drop the only surviving chunk on a
        page. An empty TABLE still carries its rendered crop (asset_ref), so when
        it is the page's only content, PROMOTE it to IMAGE rather than drop -
        preserving page coverage (no MISSING_PAGES) without leaving a corrupt
        empty TABLE (no TABLE_CORRUPTION). pages_with_other is computed on the
        CURRENT (post-tiny-icon) list so the two filters cannot together orphan a
        page. IMAGE chunks are never dropped here (multimodal).
        """

        def _has_text(c: IngestionChunk) -> bool:
            return bool(c.content and c.content.strip())

        empty_ids = {
            id(c)
            for c in chunks
            if c.modality == Modality.TABLE and not _has_text(c)
        }
        if not empty_ids:
            return chunks

        pages_with_other = {
            c.metadata.page_number
            for c in chunks
            if id(c) not in empty_ids and c.metadata and c.metadata.page_number
        }
        kept: List[IngestionChunk] = []
        dropped = promoted = 0
        for c in chunks:
            if id(c) in empty_ids:
                pg = c.metadata.page_number if c.metadata else None
                if pg in pages_with_other:
                    dropped += 1
                    continue
                if getattr(c, "asset_ref", None):
                    c.modality = Modality.IMAGE  # keep the crop, page stays covered
                    if c.metadata:
                        c.metadata.chunk_type = None
                    promoted += 1
                    kept.append(c)
                else:
                    dropped += 1  # no asset to fall back to
                continue
            kept.append(c)
        if dropped or promoted:
            logger.info(
                f"[FINALIZE] empty-content TABLE: dropped {dropped}, promoted "
                f"{promoted} to IMAGE (page-coverage guard)"
            )
        return kept

    # Running-furniture filter (PLAN_GATE_QUALITY_V1 F1). Folio/masthead detector.
    # DELIBERATELY NARROW TLD set (magazine-folio domains): this rule fires on a
    # SINGLE occurrence behind the spatial+length gate, so a broad set (.gov/.edu/
    # .net) would drop a legitimate bottom-margin citation ("Source: data.census.
    # gov") as furniture (review Finding 3). Other-TLD running headers are still
    # caught by the cross-page repetition path. NOT the same set as F3
    # (context_state.is_valid_heading), which is broader + position-keyed because
    # it has no spatial signal - the two intentionally differ.
    _FURNITURE_MASTHEAD_RE = re.compile(r"https?://|www\.|\.(?:com|aero|org)\b", re.I)
    _FURNITURE_MAX_CHARS = 70
    _FURNITURE_TOP_BAND = 80  # bbox y1 < 80 -> top ~8% of the [0,1000] page
    _FURNITURE_BOTTOM_BAND = 920  # bbox y0 > 920 -> bottom ~8%
    _FURNITURE_MIN_REPEATS = 3

    def _filter_running_furniture(
        self, chunks: List[IngestionChunk]
    ) -> List[IngestionChunk]:
        """Drop running-header / footer / folio furniture (PLAN_GATE_QUALITY_V1 F1).

        Page furniture (running headers, page folios, mastheads) is junk for
        retrieval, not content, yet it passes the structural gates. Detection is
        SPATIAL-FIRST (bbox Y-position is decisive - verified on the crucible: the
        CombatAircraft folio sits at y=960-975) plus cross-page repetition: a
        short TEXT chunk in the top/bottom page margin whose DIGIT-NORMALIZED text
        repeats across >= 3 pages (so the page number varies but the masthead/
        chapter line is constant), OR that matches a masthead/URL folio pattern.

        The repetition rule protects real headings (a section heading does not
        repeat 3+ times) and the spatial band protects content (a real heading
        sits in the content area, not the top/bottom 8% margin). A page-coverage
        guard never drops the only surviving chunk on a page.
        """
        if not getattr(self, "_drop_running_furniture", True):
            return chunks

        def _bbox(c: IngestionChunk):
            sp = c.metadata.spatial if c.metadata else None
            return sp.bbox if sp and sp.bbox else None

        def _normz(text: str) -> str:
            return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text.strip())).lower()

        # Pass 1: collect band-positioned short chunks + digit-normalized repeats.
        # id(chunk) is a safe identity key here: every chunk is held alive in
        # `chunks` for this method's lifetime, so no GC/id-reuse window exists
        # (review #5). Do not lift these sets across a generator boundary.
        norm_pages: Dict[str, set] = {}
        band: Dict[int, str] = {}
        for c in chunks:
            if c.modality != Modality.TEXT:
                continue
            content = (c.content or "").strip()
            bb = _bbox(c)
            if not content or len(content) > self._FURNITURE_MAX_CHARS:
                continue
            if not bb or len(bb) != 4:
                continue
            y0, y1 = bb[1], bb[3]
            if not (y0 > self._FURNITURE_BOTTOM_BAND or y1 < self._FURNITURE_TOP_BAND):
                continue
            nz = _normz(content)
            pg = c.metadata.page_number if c.metadata else None
            norm_pages.setdefault(nz, set()).add(pg)
            band[id(c)] = nz

        furniture_ids = {
            cid
            for cid, nz in band.items()
            if len(norm_pages[nz]) >= self._FURNITURE_MIN_REPEATS
        }
        # masthead/URL folios fire even on a single (OCR-garbled) occurrence.
        for c in chunks:
            if id(c) in band and self._FURNITURE_MASTHEAD_RE.search((c.content or "")):
                furniture_ids.add(id(c))
        if not furniture_ids:
            return chunks

        pages_with_other = {
            c.metadata.page_number
            for c in chunks
            if id(c) not in furniture_ids and c.metadata and c.metadata.page_number
        }
        kept: List[IngestionChunk] = []
        dropped = 0
        for c in chunks:
            pg = c.metadata.page_number if c.metadata else None
            if id(c) in furniture_ids and pg in pages_with_other:
                dropped += 1
                continue
            kept.append(c)
        if dropped:
            logger.info(
                f"[FINALIZE] Dropped {dropped} running-furniture chunk(s) "
                f"(folio/header/masthead; F1)"
            )
        return kept

    _CROSS_PAGE_DUPE_MIN_LEN = 20
    _CROSS_PAGE_DUPE_MIN_REPEATS = 3

    def _dedup_cross_page_repeats(
        self, chunks: List[IngestionChunk]
    ) -> List[IngestionChunk]:
        """Collapse exact TEXT content repeated across PAGE boundaries (F6).

        Captions/headers that F1 misses because they sit in the CONTENT area (not
        the margin) get repeated verbatim across pages (AIOS "(a) Normalized
        throughput..." x5), as do VLM degenerate-repetition loops (the CarOK
        class). Keep the FIRST occurrence in reading order; drop later exact
        duplicates. Requires >= 3 distinct-page occurrences and >= 20 chars so a
        short recurring label is not collapsed.

        TEXT ONLY: TABLE/FORM legitimately repeat their column-header row on every
        page (the multi-page-table trap), and IMAGE descriptions can legitimately
        be similar across distinct figures - both are excluded. A page-coverage
        guard never drops the only surviving chunk on a page.
        """
        norm = lambda t: re.sub(r"\s+", " ", (t or "").strip()).lower()  # noqa: E731
        page_occ: Dict[str, set] = {}
        for c in chunks:
            if c.modality != Modality.TEXT:
                continue
            content = (c.content or "").strip()
            if len(content) < self._CROSS_PAGE_DUPE_MIN_LEN:
                continue
            page_occ.setdefault(norm(content), set()).add(
                c.metadata.page_number if c.metadata else None
            )
        repeated = {
            n
            for n, pgs in page_occ.items()
            if len(pgs) >= self._CROSS_PAGE_DUPE_MIN_REPEATS
        }
        if not repeated:
            return chunks

        # Which chunks are drop CANDIDATES (a repeated content, not its first
        # occurrence). First occurrence is always kept. id(chunk) is a safe
        # identity key (chunks held alive in `chunks` for this scope; review #5).
        seen: set = set()
        candidate_ids: set = set()
        for c in chunks:
            if c.modality != Modality.TEXT:
                continue
            content = (c.content or "").strip()
            n = norm(content)
            if n not in repeated:
                continue
            if n in seen:
                candidate_ids.add(id(c))  # a later duplicate
            else:
                seen.add(n)
        if not candidate_ids:
            return chunks
        pages_with_other = {
            c.metadata.page_number
            for c in chunks
            if id(c) not in candidate_ids and c.metadata and c.metadata.page_number
        }
        kept: List[IngestionChunk] = []
        dropped = 0
        for c in chunks:
            pg = c.metadata.page_number if c.metadata else None
            if id(c) in candidate_ids and pg in pages_with_other:
                dropped += 1
                continue
            kept.append(c)
        if dropped:
            logger.info(
                f"[FINALIZE] Dropped {dropped} cross-page duplicate TEXT chunk(s) "
                f"(captions/headers/VLM-loop; F6)"
            )
        return kept

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

    def _is_scanned_degraded_profile(self) -> bool:
        profile = str(self._intelligence_metadata.get("profile_type", "") or "").strip().lower()
        return profile == "scanned_degraded"

    def _render_visual_assets(
        self,
        uir_chunks: List[Any],
        batch_path: Path,
        page_offset: int,
    ) -> None:
        """Render IMAGE/TABLE region crops so vision-native chunks get asset_ref.

        Thin wrapper over the shared ``materialize_visual_assets`` helper -
        the single source of truth shared with scripts/v3_batch_ingest.py so
        the batch and soak crop paths cannot diverge. The vision-native engine
        describes image/table regions but emits no binary asset; the helper
        renders each region crop from the batch PDF page (bbox is
        [0,COORD_SCALE] normalized in the page-portrait frame), saves a PNG
        under assets/, and sets the relative asset_ref so the chunk satisfies
        QA-CHECK-05. Falls back to a full-page render when a chunk has no usable
        bbox. Page numbers passed in are batch-local; the saved filename uses
        the absolute page (local + page_offset).
        """
        from .universal.asset_materializer import materialize_visual_assets

        # Asset rendering is cosmetic enrichment (region crops for IMAGE/TABLE
        # chunks). It must NEVER abort the batch: a render failure here would
        # bubble to the per-batch handler in process_pdf and discard the entire
        # batch's already-extracted text chunks, forcing the recovery net to
        # rebuild them heading-less (observed on Kimothi: a MuPDF PNG encode
        # crash discarded 151 extracted elements). Fail open - keep the text.
        try:
            materialize_visual_assets(
                uir_chunks,
                batch_path,
                self.assets_dir,
                doc_hash=self._doc_hash or "doc",
                page_offset=page_offset,
            )
        except Exception as exc:
            logger.warning(
                "[V3-ASSET] visual asset rendering failed for %s; continuing "
                "with extracted text chunks (some asset_refs may be missing): %s",
                batch_path.name,
                exc,
            )

    def _toc_for_batch(self, page_offset: int) -> Optional[Dict[Any, Any]]:
        """Project the document-wide PyMuPDF TOC into a batch's LOCAL page space.

        ``_extract_toc_headings`` keys breadcrumbs by ABSOLUTE document page
        plus a ``"__heading_map__"`` (title -> breadcrumb) entry. The
        UIR-native chunker (PLAN_V3.1 P2) sees batch-LOCAL page numbers
        (local = absolute - page_offset), so shift the integer page keys here
        and pass the heading_map through unchanged. Returns ``None`` when no
        TOC was extracted (born-digital docs with no bookmarks).
        """
        toc = getattr(self, "_toc_headings", None) or {}
        if not toc:
            return None
        local: Dict[Any, Any] = {}
        for key, value in toc.items():
            if isinstance(key, int):
                local_page = key - page_offset
                if local_page >= 1:
                    local[local_page] = value
            else:
                # "__heading_map__" (title -> breadcrumb) is page-independent.
                local[key] = value
        return local or None

    # Fail-closed ladder tier severity (PLAN_EXTRACTION_FIDELITY_V1 Section 5.4):
    # None (primary served) < docling_fast < pymupdf_terminal. Aggregating the
    # MOST-severe tier across batches answers "did any page need the ladder?".
    _FALLBACK_SEVERITY = {None: 0, "docling_fast": 1, "pymupdf_terminal": 2}

    def _accumulate_extraction_provenance(self, universal_doc) -> None:
        """Fold one batch's extraction provenance into the doc-level summary.

        Reads the ``extraction_*`` stamps that ``mmrag_v3.extract`` records on
        ``universal_doc.metadata.extra`` (served engine, fail-closed fallback
        tier, degraded/recovered page counts) and aggregates them across the
        document's batches: the first engine seen, the most-severe ladder tier
        engaged anywhere, and summed degraded/recovered page counts. ADVISORY
        observability for the Section 5.4 consumers; never gates anything.
        """
        extra = getattr(getattr(universal_doc, "metadata", None), "extra", None) or {}
        acc = self._extraction_provenance
        engine = extra.get("extraction_engine")
        if engine and acc["engine"] is None:
            acc["engine"] = engine
        fallback = extra.get("extraction_fallback")
        sev = self._FALLBACK_SEVERITY.get(fallback, 1 if fallback else 0)
        cur_sev = self._FALLBACK_SEVERITY.get(acc["fallback"], 1 if acc["fallback"] else 0)
        if sev > cur_sev:
            acc["fallback"] = fallback
        acc["degraded"] += int(extra.get("extraction_degraded_pages") or 0)
        acc["recovered"] += int(extra.get("extraction_recovered_pages") or 0)

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
        # ================================================================
        # V3.0 native extraction — Phase A Step 5 input-boundary decoupling.
        # ================================================================
        # The batch PDF is extracted to a UniversalDocument by the engine-
        # agnostic mmrag_v3 router (HybridEngine cost-optimizer by default;
        # USE_DOCLING_FAST / USE_VLM_ENGINE force a single engine), then
        # chunked by the UIR-native chunker. No DoclingDocument crosses this
        # boundary — batch_processor.py is now pure engine-agnostic
        # orchestration (splitting, dedup, filtering, JSONL export).
        # ================================================================
        from mmrag_v3.processor import extract as v3_extract
        from .chunking.uir_chunker import chunk_universal_document

        logger.info(
            f"Processing batch {batch_info.batch_index + 1}/{split_result.batch_count}: "
            f"pages {batch_info.page_range_str} (offset={batch_info.page_offset})"
        )

        profile_type = self._intelligence_metadata.get("profile_type") or None

        universal_doc = v3_extract(str(batch_info.batch_path))
        self._accumulate_extraction_provenance(universal_doc)

        # PLAN_V3.1 P2: thread the PyMuPDF TOC (extracted document-wide in
        # process_pdf, keyed by ABSOLUTE page) into the UIR-native chunker as
        # plain data, shifted into this batch's LOCAL page space (the chunker
        # sees batch-local page numbers; the absolute-page projection happens
        # below). Drives cross-page heading carry-forward + breadcrumb_path.
        local_toc = self._toc_for_batch(batch_info.page_offset)
        uir_chunks = chunk_universal_document(
            universal_doc,
            profile_type=profile_type,
            toc_headings=local_toc,
            # Cluster B (2026-06-07): heading assignment runs per batch, so seed
            # it with the last active heading from the previous batch. Without
            # this, a batch whose chapter title appears only as a glued running
            # header (HarryPotter ch.1, batch 3) starts with no heading context
            # and every chunk goes null. A real in-page heading/TOC leaf on this
            # batch still overrides the carry.
            carry_in_heading=self._carry_heading,
            carry_in_breadcrumb=self._carry_breadcrumb,
        )
        # Capture carry-out: the last text chunk that received a heading becomes
        # the seed for the next batch.
        for _uir in reversed(uir_chunks):
            if _uir.modality == Modality.TEXT and _uir.parent_heading:
                self._carry_heading = _uir.parent_heading
                self._carry_breadcrumb = list(_uir.breadcrumb_path or [])
                break

        # Vision-native extraction describes image/table regions but emits no
        # binary asset. Render the region crops here (batch_processor owns the
        # assets dir) so IMAGE/TABLE chunks carry a real asset_ref and satisfy
        # QA-CHECK-05. Must run BEFORE the page-offset projection below — the
        # crop is rendered from the batch-local page.
        self._render_visual_assets(
            uir_chunks, batch_info.batch_path, batch_info.page_offset
        )

        # Fail-open guard: a render OR encode failure inside the materializer
        # (degenerate clip get_pixmap raise; full-page fallback itself failing)
        # leaves an IMAGE/TABLE chunk with no asset_ref. That chunk would raise
        # QA-CHECK-05 in from_uir below and the per-batch handler would discard
        # this batch's extracted TEXT (the Kimothi-class loss via a different
        # MuPDF entry point). Drop the un-renderable visual chunk instead -
        # losing one crop beats losing the page's text.
        _pre_render = len(uir_chunks)
        uir_chunks = [
            uir
            for uir in uir_chunks
            if uir.modality not in (Modality.IMAGE, Modality.TABLE)
            or getattr(uir, "asset_ref", None)
        ]
        if len(uir_chunks) != _pre_render:
            logger.warning(
                "[V3-ASSET] dropped %d IMAGE/TABLE chunk(s) with no rendered "
                "asset (QA-CHECK-05) to keep the batch's text",
                _pre_render - len(uir_chunks),
            )

        chunks: List[IngestionChunk] = []
        for uir in uir_chunks:
            # Project batch-local page numbers into absolute document pages.
            if uir.locator is not None and uir.locator.page_number is not None:
                uir.locator.page_number = uir.locator.page_number + batch_info.page_offset
                # Re-leaf the TOC breadcrumb's "Page N" tail to the absolute
                # page so the breadcrumb agrees with the projected locator
                # (PLAN_V3.1 P2; the heading pass ran in batch-local space).
                if batch_info.page_offset and uir.breadcrumb_path:
                    uir.breadcrumb_path = [
                        f"Page {uir.locator.page_number}"
                        if isinstance(b, str) and b.startswith("Page ")
                        else b
                        for b in uir.breadcrumb_path
                    ]
            chunks.append(
                IngestionChunk.from_uir(
                    uir,
                    doc_id=self._doc_hash or "unknown",
                    source_file=source_file,
                    file_type=FileType.PDF,
                    position=self._next_chunk_position(),
                    profile_type=profile_type,
                    document_domain=self._intelligence_metadata.get("document_domain"),
                    document_modality=self._intelligence_metadata.get("document_modality"),
                )
            )

        logger.info(
            f"Batch {batch_info.batch_index + 1} complete: {len(chunks)} chunks "
            f"via V3 UIR-native extraction ({universal_doc.total_pages} pages)"
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

        # v2.9 Phase 1: reset per-document chunk position counter so chunk_id
        # collisions cannot accumulate across documents in batch CLI runs.
        self._chunk_position = 0

        # Cluster B: reset cross-batch heading carry so one document's last
        # heading never bleeds into the next document in a batch CLI run.
        self._carry_heading = None
        self._carry_breadcrumb = None

        # PLAN_EXTRACTION_FIDELITY_V1 Section 5.4: aggregate the per-batch
        # extraction provenance (served engine + fail-closed ladder outcome,
        # stamped on each UniversalDocument by mmrag_v3.extract) into a
        # doc-level summary written onto the IngestionMetadata header. ADVISORY
        # observability only; never affects gate semantics.
        self._extraction_provenance = {
            "engine": None,
            "fallback": None,
            "degraded": 0,
            "recovered": 0,
        }

        # Workstream B: legacy callers still get the cheap pre-pass here.
        # Canonical CLI paths pass a PdfConversionPlan with this decision already made.
        if self._conversion_plan is None:
            self._decide_code_enrichment(pdf_path)
            self._ensure_conversion_plan()

        # Extract TOC from PyMuPDF BEFORE splitting — gives us the correct
        # heading hierarchy for every page, independent of batch boundaries.
        self._toc_headings = self._extract_toc_headings(pdf_path)

        # Initialize global vision manager
        self._vision_manager = self._initialize_vision_manager()
        if self._vision_manager:
            self._vision_manager.document_domain = self._intelligence_metadata.get(
                "document_domain", ""
            )

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

        # For scanned/degraded documents, lower the refiner threshold to 0.0
        # so ALL OCR text gets refined. OCR on degraded scans is never trustworthy —
        # even "clean-looking" text like "Jamu la Frizgi" (should be "J.B. Wood")
        # scores corruption=0.0 because it looks like valid text to pattern matchers.
        if self._refiner and doc_modality in ("scanned_degraded", "scanned_clean"):
            try:
                self._refiner.config.min_refine_threshold = 0.0
                logger.info(
                    f"[REFINER] Lowered threshold to 0.0 for {doc_modality} — "
                    "all OCR text will be refined"
                )
            except Exception:
                pass

        # REQ-STRUCT-02: Override OCR guard when encoding corruption is detected.
        # Even if the document looks digital (native_digital), if the text layer is
        # encoding-garbage (CIDFont / broken char map), we MUST force full OCR.
        # This auto-enables enable_ocr too — the governance layer must not override
        # a pathology-driven decision.
        if self.has_encoding_corruption and is_digital_like:
            if not self.force_ocr:
                self.force_ocr = True
            if not self.enable_ocr:
                self.enable_ocr = True
            logger.warning(
                f"[OCR-GUARD] ENCODING CORRUPTION detected on digital-like PDF "
                f"(modality={doc_modality}); overriding force_ocr=True, enable_ocr=True "
                f"to bypass corrupt text layer."
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
        # Phase 2 (PLAN_V2.10.md): also trigger on per-batch localized shortfall
        # so the scout fires on large documents where doc-level variance averages
        # out (e.g. Fluent_Python: 770 pages, 6 missing pages, doc-variance ~0%).
        per_batch_force_run = self._per_batch_shortfall_fires(
            chunks=filtered_chunks,
            batches=split_result.batches,
            pdf_path=pdf_path,
        )
        recovery_input = list(filtered_chunks)  # do not mutate the baseline list
        recovered_chunks = self._run_text_integrity_scout(
            chunks=recovery_input,
            source_file=pdf_path.name,
            variance_percent=token_result.variance_percent,
            force_run=per_batch_force_run,
        )

        # Step 6: Re-validate token balance after recovery (polish log level)
        scout_produced_chunks = len(recovered_chunks) > len(filtered_chunks)
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
        elif scout_produced_chunks:
            # Per-batch trigger fired and the scout rescued real text on
            # individual pages — keep those recovery chunks even though
            # doc-level variance was within tolerance.
            extra = len(recovered_chunks) - len(filtered_chunks)
            print(
                f"✓ [QA-CHECK-01] Per-batch scout recovered {extra} chunk(s) "
                f"on localized-shortfall batches (doc-variance "
                f"{token_result.variance_percent:.1f}% within tolerance)",
                flush=True,
            )
            logger.info(
                f"[QA-CHECK-01] Per-batch TextIntegrityScout added {extra} recovery chunk(s); "
                f"doc-level variance {token_result.variance_percent:.1f}% remained within tolerance"
            )
            all_chunks = recovered_chunks
        else:
            all_chunks = filtered_chunks

        # Recovery chunks are appended AFTER _apply_quality_filters (which hosts the
        # Step 3a1a infix step-number repair), so they have not been through it.
        # Re-apply here so rescued text gets the same corrected paragraph
        # boundaries before the semchunk re-split / dedup passes see it. Idempotent:
        # already-repaired primary chunks have a ``\n`` separator (not ``[ \t]+``)
        # and cannot re-match. (PLAN_V3.1 follow-up: closes the recovery-path gap
        # left by re-homing the repair into _apply_quality_filters.)
        all_chunks = self._repair_infix_step_numbers(all_chunks)

        # Re-split oversize recovery chunks using semchunk (sentence-aware).
        # Recovery creates large text blobs; our custom splitter cuts mid-sentence.
        try:
            import semchunk as _sc
            _token_counter = len  # Use char count as proxy — 1200 chars ≈ 300 tokens
            _resplit = 0
            _resplit_out: List[IngestionChunk] = []
            for ch in all_chunks:
                em = ch.metadata.extraction_method if ch.metadata else ""
                if (
                    ch.modality == Modality.TEXT
                    and em in (
                        "recovery_scan",
                        "recovery_gap_fill",
                    )
                    and ch.content
                    and len(ch.content) > 1200
                ):
                    parts = _sc.chunk(ch.content, chunk_size=1200, token_counter=_token_counter, memoize=False)
                    if len(parts) > 1:
                        for j, part in enumerate(parts):
                            if not part.strip():
                                continue
                            clone = ch.model_copy(deep=True)
                            clone.content = part.strip()
                            clone.chunk_id = f"{ch.chunk_id}_s{j}"
                            if clone.metadata and clone.metadata.refined_content:
                                clone.metadata.refined_content = part.strip()
                            _resplit_out.append(clone)
                        _resplit += 1
                    else:
                        _resplit_out.append(ch)
                else:
                    _resplit_out.append(ch)
            if _resplit:
                all_chunks = _resplit_out
                logger.info(f"[SEMCHUNK] Re-split {_resplit} oversize recovery chunks")
        except ImportError:
            pass

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
        all_chunks = self._normalize_chunk_text(all_chunks)  # PUA + whitespace normalization
        all_chunks = self._filter_no_visual_images(all_chunks)
        all_chunks = self._filter_repetition_garbage(all_chunks)
        all_chunks = self._apply_table_recovery_highlander_dedup(all_chunks)
        # Drop recovery text chunks that duplicate the primary VLM extraction on
        # the same page (the scout re-pulls VLM-captured code from the flush-left
        # PDF text layer, polluting output + the R3 code-indentation metric).
        all_chunks = self._apply_recovery_vs_primary_dedup(all_chunks)

        # v2.16 Phase 4: VLM-table IoU dedup — suppress flat-prose text chunks
        # that spatially overlap a VLM-extracted table on the same page above
        # `dedup_vlm_table_iou_threshold`. Closes the v2.14 P1 CarOK regression
        # where VLM tables coexisted with flat-prose duplicates and retrieval
        # picked the prose chunk 29/30 times.
        all_chunks = self._apply_vlm_table_iou_dedup(all_chunks)

        # PLAN_V3.1 P4 (2026-06-01): the spatial final-boundary-repair bridge
        # (_apply_final_boundary_repairs + _merge_hungry_operators +
        # _strip_trailing_headings + _apply_vision_aided_front_matter_detection)
        # is DEPRECATED for the VLM-native path: the P4 probe proved
        # spatial/geometric merging overrides the VLM's semantic chunk boundaries,
        # over-merging distinct concepts into oversized blobs and diluting
        # retrieval vectors. The bridge was a geometric fix for an OCR/Docling
        # problem (sentences physically split across bboxes) that VLM-native
        # extraction does not have. Those 5 methods were DELETED 2026-06-06 (R3
        # review follow-up; dead since P4). See DECISIONS.md "Spatial proximity
        # boundary-repair bridge DEPRECATED for VLM-native" + "Orphaned
        # boundary-repair bridge deleted; infix step-number repair re-homed".
        # NOTE: _merge_mid_sentence_chunks remains live via _apply_quality_filters,
        # _apply_spatial_refiner remains live via _sanitize_technical_manual_final,
        # and _repair_infix_step_numbers (a content repair collaterally cut with the
        # bridge) was re-homed into _apply_quality_filters Step 3a1a.
        # (AGENT-SPATIAL-20) - those are separate paths, not this bridge.

        # Selective OCR patching for encoding-corrupted chunks.
        # Keeps HybridChunker structure, replaces only the corrupted text spans.
        if self.has_encoding_corruption:
            try:
                from .validators.corruption_interceptor import patch_corrupted_chunks
                all_chunks = patch_corrupted_chunks(all_chunks, self._current_pdf_path)
            except Exception as _ci_err:
                logger.warning(f"[CORRUPTION-INTERCEPTOR] Failed: {_ci_err}")

        # Quarantine unrepairable corrupted chunks.
        all_chunks = self._quarantine_corrupted_text_chunks(all_chunks)

        all_chunks = self._sanitize_toc_cell_markers(all_chunks)

        # Final oversize breaker — catches chunks created or enlarged by
        # recovery, merging, or other post-processing passes.
        all_chunks = self._apply_oversize_breaker(all_chunks, max_chars=1500)

        # Deduplicate repeated paragraphs WITHIN individual chunks.
        # VLM transcription on cover pages can repeat the same text 3-4x
        # when it reads title, spine, and bleed-through as separate instances.
        all_chunks = self._dedup_intra_chunk_repeats(all_chunks)

        # v2.9 cross-chunk dedup: drop byte-equal text/table content on
        # the same page. The corruption interceptor's per-bbox OCR
        # frequently runs Tesseract on overlapping regions of a single
        # corrupted table (e.g. Combat Aircraft p66 squadron roster),
        # producing identical OCR output for ~19 source chunks. Each
        # patched chunk then survives the oversize-breaker as a 4-part
        # split, leaving the canonical output with ~74 byte-equal text
        # rows per real OCR result. First-wins dedup at this stage
        # collapses those redundant rows without changing extraction
        # decisions upstream.
        _seen: Dict[Tuple[int, str, str], int] = {}
        _deduped: List[IngestionChunk] = []
        _cross_chunk_dropped = 0
        for _ch in all_chunks:
            if _ch.modality not in (Modality.TEXT, Modality.TABLE):
                _deduped.append(_ch)
                continue
            _content = (_ch.content or "").strip()
            if not _content:
                _deduped.append(_ch)
                continue
            _page = (
                _ch.metadata.page_number
                if _ch.metadata and _ch.metadata.page_number
                else 0
            )
            _key = (int(_page), _ch.modality.value, _content)
            if _key in _seen:
                _cross_chunk_dropped += 1
                continue
            _seen[_key] = 1
            _deduped.append(_ch)
        if _cross_chunk_dropped:
            logger.info(
                f"[CROSS-CHUNK-DEDUP] Dropped {_cross_chunk_dropped} byte-equal "
                f"text/table chunks on same page (post-corruption-patch + "
                f"post-oversize-breaker)"
            )
        all_chunks = _deduped

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
            # ============================================================
            # PRE-METADATA FILTERS: run all chunk-level filters BEFORE
            # writing IngestionMetadata so chunk_count is accurate.
            # ============================================================

            # Filter redundant FULL-PAGE EDITORIAL images on pages that already have text.
            # Scanned documents produce both OCR text chunks AND a full-page image with
            # VLM description for the same page. The image description is redundant.
            pages_with_text = {
                c.metadata.page_number for c in export_chunks
                if c.modality == Modality.TEXT and c.metadata and c.metadata.page_number
            }
            _pre_filter = len(export_chunks)
            export_chunks = [
                c for c in export_chunks
                if not (
                    c.modality == Modality.IMAGE
                    and c.content
                    and "[FULL-PAGE EDITORIAL" in c.content
                    and c.metadata
                    and c.metadata.page_number in pages_with_text
                )
            ]
            _editorial_filtered = _pre_filter - len(export_chunks)
            if _editorial_filtered:
                logger.info(
                    f"[FINALIZE] Filtered {_editorial_filtered} redundant FULL-PAGE EDITORIAL images"
                )

            # Drop/promote blank image/table assets.
            export_chunks = self._filter_blank_assets(export_chunks)

            # Drop icon/glyph-class image regions (sub-content tiny rasters that
            # only add retrieval noise + IMAGE_NO_VLM/ASSET_TINY advisories).
            export_chunks = self._filter_tiny_icon_images(export_chunks)

            # Re-apply oversize breaker: TABLE→TEXT promotion may create
            # text chunks exceeding the 1500-char gate.
            export_chunks = self._apply_oversize_breaker(export_chunks, max_chars=1500)

            # Export-level hygiene for technical manuals must run before
            # metadata so chunks zeroed by page-number/control-char cleanup
            # are dropped at the same boundary as every other empty text row.
            export_chunks = self._apply_technical_manual_export_sanitizer(export_chunks)

            # Phase 4 Step 3: drop chunks whose content matches strict-gate
            # corruption patterns (Combat p66 magazine-font garble; mirrors the
            # qa_full_conversion.py LOCALIZED_CORRUPTION detector).
            export_chunks = self._drop_corrupted_chunks_before_metadata(export_chunks)

            # Final guard: drop any text chunk that ended up empty after
            # all upstream sanitisers/dedup/breaker passes. The strict gate's
            # empty_text_chunks invariant (UNIVERSAL_FAIL) cannot tolerate
            # even one such chunk; this catches every upstream path at once.
            export_chunks = self._drop_empty_text_chunks_before_metadata(export_chunks)

            # Drop/repair empty-content TABLE chunks (no markdown to retrieve),
            # with a page-coverage guard so they cannot manufacture MISSING_PAGES.
            export_chunks = self._promote_or_drop_empty_tables(export_chunks)

            # Drop running-header/footer/folio furniture (retrieval noise that
            # passes the structural gates). PLAN_GATE_QUALITY_V1 F1. Runs last in
            # the hygiene sequence so its page-coverage guard sees the final set.
            export_chunks = self._filter_running_furniture(export_chunks)

            # Collapse exact TEXT repeated across page boundaries (captions/
            # headers F1 missed in the content area; VLM repetition loops).
            # PLAN_GATE_QUALITY_V1 F6. TEXT only (TABLE/FORM repeat headers).
            export_chunks = self._dedup_cross_page_repeats(export_chunks)

            # V3.0 Phase A Step 5: canonical heading propagation + vision-aided
            # front-matter detection are STRIPPED — the UIR chunker carries
            # parent_heading directly; breadcrumb reconciliation is deferred to
            # the LLM-sanitization layer per V3_EXECUTION_MANDATE.md §3.

            # v2.9 Phase 1 (followup): final-stage dedup by chunk_id. The
            # per-document position counter eliminates the bulk of v2.8's
            # within-file collisions, but the legacy hybrid_chunker path
            # in V2DocumentProcessor occasionally yields the same chunk
            # twice (Docling-side chunker emitting duplicate DocChunks for
            # the same text element on certain pages). When that happens
            # the duplicates are byte-equal, so first-wins is safe.
            _seen_chunk_ids: set[str] = set()
            _deduped: List[IngestionChunk] = []
            for _ch in export_chunks:
                _cid = getattr(_ch, "chunk_id", None)
                if not _cid:
                    _deduped.append(_ch)
                    continue
                if _cid in _seen_chunk_ids:
                    continue
                _seen_chunk_ids.add(_cid)
                _deduped.append(_ch)
            _dropped = len(export_chunks) - len(_deduped)
            if _dropped:
                logger.info(
                    f"[FINALIZE] chunk_id dedup: dropped {_dropped} byte-equal "
                    f"duplicate chunks (v2.9 Phase 1 follow-up)"
                )
            export_chunks = _deduped

            # ============================================================
            # PLAN_V2.10 Phase 3 — `B4B_FULL_DOC_PICTURE_DEDUP`
            #
            # Build the set of pages whose only content reaching the
            # exporter is IMAGE chunks. The cross-page pHash dedup further
            # below is a publisher-artwork fix (Earthship Vol 1, Python
            # Distilled): hand-drawn illustrations across consecutive
            # pages share pHash signatures within Hamming ≤ 10. Without
            # this guard, the registry rejects every same-style figure
            # except the first one it sees, orphaning whole image-only
            # pages and reporting them as MISSING_PAGES at strict-gate
            # time. The guard preserves AT MOST one image per such page
            # — subsequent near-duplicates on the same page still drop
            # because the page already has surviving content.
            # ============================================================
            _phash_image_only_pages: set[int] = set()
            try:
                _pages_with_non_image = {
                    int(c.metadata.page_number)
                    for c in export_chunks
                    if c.modality in (Modality.TEXT, Modality.TABLE)
                    and c.metadata
                    and c.metadata.page_number
                }
                _pages_with_image_chunk = {
                    int(c.metadata.page_number)
                    for c in export_chunks
                    if c.modality == Modality.IMAGE
                    and c.metadata
                    and c.metadata.page_number
                }
                _phash_image_only_pages = _pages_with_image_chunk - _pages_with_non_image
            except Exception:
                _phash_image_only_pages = set()
            # Tracks pages that have already written at least one IMAGE
            # chunk during this export loop — INCLUDING unique images
            # (not just preserved near-duplicates). The carve-out below
            # only fires when an image-only page would otherwise emit
            # zero IMAGE chunks; once any image has been written for a
            # page, subsequent near-duplicates on that same page still
            # drop because the page is already covered.
            _phash_pages_with_exported_image: set[int] = set()

            # ============================================================
            # METADATA RECORD: write AFTER all filtering so chunk_count
            # reflects the actual exported chunk set.
            # ============================================================
            intel = self._intelligence_metadata or {}
            # Derive is_scan from document_modality (key set by cli.py intelligence stack).
            _modality = intel.get("document_modality") or ""
            _is_scan: Optional[bool] = _modality.startswith("scanned") if _modality else None
            # Compute provenance hashes
            import hashlib as _hl
            _src_hash = None
            if self._current_pdf_path and self._current_pdf_path.exists():
                _h = _hl.sha256()
                with open(self._current_pdf_path, "rb") as _sf:
                    for _blk in iter(lambda: _sf.read(8192), b""):
                        _h.update(_blk)
                _src_hash = _h.hexdigest()

            meta_record = IngestionMetadata(
                schema_version=SCHEMA_VERSION,
                doc_id=export_chunks[0].doc_id if export_chunks else "",
                source_file=Path(self._current_pdf_path).name if self._current_pdf_path else "",
                profile_type=intel.get("profile_type"),
                document_type=intel.get("document_modality"),
                domain=intel.get("document_domain"),
                is_scan=_is_scan,
                total_pages=self._doc_total_pages,
                image_density=self._doc_image_density,
                avg_text_per_page=self._calculate_actual_avg_text(export_chunks),
                has_flat_text_corruption=self.has_flat_text_corruption,
                has_encoding_corruption=self.has_encoding_corruption,
                chunk_count=len(export_chunks),
                ingestion_timestamp=datetime.now(timezone.utc).isoformat(),
                pipeline_version=SCHEMA_VERSION,
                source_file_hash=_src_hash,
                # PLAN_EXTRACTION_FIDELITY_V1 Section 5.4: doc-level extraction
                # provenance aggregated across batches (advisory observability).
                extraction_engine=self._extraction_provenance.get("engine"),
                extraction_fallback=self._extraction_provenance.get("fallback"),
                extraction_degraded_pages=self._extraction_provenance.get("degraded"),
                extraction_recovered_pages=self._extraction_provenance.get("recovered"),
            )
            f.write(json.dumps(meta_record.model_dump(mode="json"), ensure_ascii=False) + "\n")

            write_buffer: List[str] = []

            # Process chunks with streaming writes
            for idx, chunk in enumerate(export_chunks):
                try:
                    # Log progress every 50 chunks
                    if idx % 50 == 0 and idx > 0:
                        logger.debug(f"[FINALIZE] Processed {idx}/{len(export_chunks)} chunks")

                    chunk_dict = self._sanitize_chunk_for_export(chunk)

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

                                        _pg = (
                                            int(chunk.metadata.page_number)
                                            if chunk.metadata and chunk.metadata.page_number
                                            else None
                                        )
                                        if dup_info.is_duplicate:
                                            orig_page = (
                                                dup_info.original_record.page_number
                                                if dup_info.original_record
                                                else "unknown"
                                            )
                                            # PLAN_V2.10 Phase 3 carve-out:
                                            # if this duplicate sits on an
                                            # image-only page that has not
                                            # yet exported any IMAGE, keep
                                            # it — rejecting it would orphan
                                            # the page (MISSING_PAGES). The
                                            # decision is pinned by the pure
                                            # `_phash_carve_out_should_preserve_duplicate`
                                            # helper above so the contract is
                                            # exercised by the same code path
                                            # the tests assert.
                                            if self._phash_carve_out_should_preserve_duplicate(
                                                page_number=_pg,
                                                image_only_pages=_phash_image_only_pages,
                                                pages_with_exported_image=_phash_pages_with_exported_image,
                                            ):
                                                _phash_pages_with_exported_image.add(_pg)
                                                logger.info(
                                                    f"[PHASH-PAGE-COVERAGE] Preserving near-duplicate "
                                                    f"{asset_file} on image-only page {_pg} "
                                                    f"(would otherwise orphan the page; "
                                                    f"matches asset on page {orig_page})"
                                                )
                                                # fall through to write this chunk
                                            else:
                                                # DUPLICATE_REJECTED - skip this chunk
                                                duplicate_count += 1
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
                                        else:
                                            # Log successful registration; record
                                            # the page so any later near-duplicate
                                            # on the same image-only page still
                                            # drops via the carve-out's "already
                                            # covered" branch.
                                            if _pg is not None:
                                                _phash_pages_with_exported_image.add(_pg)
                                            logger.info(
                                                f"[FINALIZING] Asset {filename} linked to "
                                                f"Page {chunk.metadata.page_number}"
                                            )

                                except Exception as hash_e:
                                    logger.warning(f"pHash check failed for {asset_file}: {hash_e}")

                    # Content-Type Classification: demote boilerplate to low priority.
                    if chunk_dict.get("modality") == "text":
                        import re as _re_bp
                        _content = chunk_dict.get("content", "")
                        _BOILERPLATE_RE = _re_bp.compile(
                            r"\bISBN\b|\bISSN\b|Library of Congress|All rights reserved"
                            r"|Printed in .{2,20}$|First .{2,20} edition"
                            r"|©\s*\d{4}",
                            _re_bp.IGNORECASE,
                        )
                        _boilerplate_hits = len(_BOILERPLATE_RE.findall(_content))
                        if _boilerplate_hits >= 2:
                            meta = chunk_dict.get("metadata", {})
                            meta["search_priority"] = "low"
                            chunk_dict["metadata"] = meta

                    # Safety net: ensure refined_content is never null for text chunks.
                    # Some code paths (list_items, ads, edge cases) may skip the default.
                    if chunk_dict.get("modality") == "text":
                        meta = chunk_dict.get("metadata", {})
                        if meta.get("refined_content") is None:
                            meta["refined_content"] = chunk_dict.get("content", "")

                    # Set VLM enrichment status for image chunks
                    if chunk_dict.get("modality") == "image":
                        meta = chunk_dict.get("metadata", {})
                        vd = meta.get("visual_description") or chunk_dict.get("content") or ""
                        if not vd or vd.startswith("[Figure on page") or vd.startswith("[VLM_FAILED"):
                            if self.vision_provider and self.vision_provider != "none":
                                meta["vision_status"] = "failed"
                                if vd.startswith("[VLM_FAILED"):
                                    meta["vision_error"] = vd
                            else:
                                # No vision provider: terminal, documented no-VLM
                                # state (NOT awaiting a VLM). This is a multimodal
                                # converter - retain the image as an ID-only
                                # fallback (asset filename) so the asset still
                                # ships; the strict gate treats no_vlm as advisory,
                                # not a VISION_PENDING failure.
                                _ar = chunk_dict.get("asset_ref") or {}
                                _fn = (_ar.get("file_path") or "").split("/")[-1]
                                meta["vision_status"] = "no_vlm"
                                meta["vision_provider_used"] = "none"
                                meta["vision_error"] = (
                                    "no vision provider configured (--vision-provider none)"
                                )
                                if not meta.get("visual_description"):
                                    meta["visual_description"] = (
                                        f"[image: {_fn}]" if _fn else "[image: no VLM description]"
                                    )
                        elif "extraction unavailable" in vd.lower():
                            meta["vision_status"] = "pending"
                        else:
                            meta["vision_status"] = "done"
                            meta["vision_provider_used"] = self.vision_provider
                            # Source Sanctity: validate final description
                            vr = validate_vlm_response(vd)
                            if not vr.is_valid:
                                meta["vision_validation_issues"] = vr.issues
                        chunk_dict["metadata"] = meta

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

        # IngestionMetadata is required to be the first JSONL record, but final
        # image deduplication happens while chunk lines are being streamed. Patch
        # the first record after export so chunk_count reflects emitted chunks,
        # not pre-dedup candidates.
        try:
            with open(output_jsonl, "r", encoding="utf-8") as _rf:
                _lines = _rf.readlines()
            if _lines:
                _first = json.loads(_lines[0])
                if _first.get("object_type") == "ingestion_metadata":
                    _first["chunk_count"] = written_chunks
                    _tmp = output_jsonl.with_suffix(output_jsonl.suffix + ".tmp")
                    with open(_tmp, "w", encoding="utf-8") as _wf:
                        _wf.write(json.dumps(_first, ensure_ascii=False) + "\n")
                        _wf.writelines(_lines[1:])
                    _tmp.replace(output_jsonl)
        except Exception as e:
            logger.warning(f"[FINALIZE] Failed to reconcile metadata chunk_count: {e}")

        # Clean up orphan assets: files saved to disk during extraction but
        # not referenced in the final JSONL (e.g., Docling images skipped in
        # favor of PyMuPDF extraction for digital PDFs).
        assets_dir = self.output_dir / "assets"
        if assets_dir.exists():
            referenced_files = set()
            with open(output_jsonl, "r", encoding="utf-8") as _rf:
                for _rl in _rf:
                    _robj = json.loads(_rl)
                    _fp = (_robj.get("asset_ref") or {}).get("file_path", "")
                    if _fp:
                        referenced_files.add(Path(_fp).name)
            orphans_removed = 0
            for asset_file in assets_dir.iterdir():
                if asset_file.suffix == ".png" and asset_file.name not in referenced_files:
                    asset_file.unlink()
                    orphans_removed += 1
            if orphans_removed:
                logger.info(f"[CLEANUP] Removed {orphans_removed} orphan asset files")

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

        # Multimodal notice: this is a multimodal converter. If it ran without a
        # vision provider on an image-bearing document, the images shipped as
        # ID-only fallbacks (no descriptions) - of little retrieval value. Warn
        # loudly so the user knows the run was, for the image content, a waste of
        # time/resources and should be re-run with a VLM.
        # no_vlm is stamped onto the export dict during serialization, not the
        # chunk object - so derive the count from the same trigger condition
        # (no vision provider) over the image chunks actually written.
        _vp = (self.vision_provider or "none").strip().lower()
        _no_vlm_imgs = (
            sum(1 for c in export_chunks if c.modality == Modality.IMAGE)
            if _vp == "none"
            else 0
        )
        if _no_vlm_imgs:
            _share = _no_vlm_imgs / max(written_chunks, 1)
            warn = (
                f"[MULTIMODAL] {_no_vlm_imgs} image chunk(s) "
                f"({_share:.0%} of output) shipped as ID-only fallbacks WITHOUT "
                f"descriptions because no vision provider was configured "
                f"(--vision-provider none). For an image-bearing document this "
                f"conversion has limited multimodal value - re-run with a VLM."
            )
            logger.warning(warn)
            print(f"\n⚠️  {warn}\n", flush=True)

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
            "scanned": 3,  # Relaxed: keep OCR text
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

    # PUA → readable-unicode mapping (Wingdings/Symbol font characters extracted by docling).
    # Keys are Private Use Area codepoints (U+E000–U+F8FF); values are their visual equivalents.
    _PUA_MAP: ClassVar[dict] = {
        "\uf02d": "\u2022",  # Wingdings dash/bullet → •
        "\uf0b7": "\u2022",  # Wingdings solid bullet → •
        "\uf0e0": "\u2192",  # Wingdings right arrow → →
        "\uf070": "\u25c4",  # Wingdings left triangle → ◄
        "\uf071": "\u25ba",  # Wingdings right triangle → ►
        "\uf074": "\u25b2",  # Wingdings up triangle → ▲
        "\uf075": "\u25bc",  # Wingdings down triangle → ▼
    }

    @staticmethod
    def _normalize_pua_chars(text: str) -> str:
        """Replace Private Use Area (PUA) Unicode codepoints with readable equivalents.

        Docling converts Wingdings/Symbol font characters to PUA codepoints that are
        meaningless to embedding models. Known mappings are replaced; unknown PUA
        codepoints become a single space (safe fallback).
        """
        if not text:
            return text
        result: List[str] = []
        for ch in text:
            code = ord(ch)
            if 0xE000 <= code <= 0xF8FF:
                result.append(BatchProcessor._PUA_MAP.get(ch, " "))
            else:
                result.append(ch)
        return "".join(result)

    @staticmethod
    def _collapse_spaced_heading(text: str) -> str:
        """Collapse decorative spaced-out headings like 'C H A P T E R  O N E' → 'CHAPTER ONE'.

        Scanned books use letter-spaced headings for decoration. Docling extracts
        them verbatim, producing unreadable breadcrumbs and chunk text.
        """
        import re as _re

        _WORDS = (
            "CHAPTER", "PART", "BOOK", "SECTION", "APPENDIX", "EPILOGUE", "PROLOGUE",
            "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE",
            "TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN", "SIXTEEN",
            "SEVENTEEN", "EIGHTEEN", "NINETEEN", "TWENTY", "THIRTY",
            "THE", "AND", "OF",
        )

        def _collapse(m: "re.Match[str]") -> str:
            spaced = m.group(0)
            # Split on double-space word boundaries first, then collapse each word
            parts = _re.split(r" {2,}", spaced)
            words = []
            for part in parts:
                word = part.replace(" ", "")  # "C H A P T E R" → "CHAPTER"
                words.append(word)
            collapsed = " ".join(words)  # "CHAPTER ONE"
            # If word splitting didn't produce known words, try dictionary split
            if len(words) == 1 and len(words[0]) > 7:
                blob = words[0]
                result_parts = []
                while blob:
                    matched = False
                    for w in sorted(_WORDS, key=len, reverse=True):
                        if blob.startswith(w):
                            result_parts.append(w)
                            blob = blob[len(w):]
                            matched = True
                            break
                    if not matched:
                        result_parts.append(blob)
                        break
                collapsed = " ".join(result_parts)
            return collapsed

        # Match sequences of 3+ uppercase letters separated by 1-2 spaces
        return _re.sub(r"(?:[A-Z]\s{1,2}){3,}[A-Z]", _collapse, text)

    def _normalize_chunk_text(
        self, chunks: List["IngestionChunk"]
    ) -> List["IngestionChunk"]:
        """Post-processing pass: PUA normalization + whitespace collapsing for text chunks.

        - PUA chars: applied to ALL text chunks regardless of chunk_type.
        - Double-space collapsing: applied only to non-code chunks.
        - Spaced-heading collapse: 'C H A P T E R  O N E' → 'CHAPTER ONE'.
        """
        import re as _re

        for ch in chunks:
            if ch.modality != Modality.TEXT or not ch.content:
                continue
            text = self._normalize_pua_chars(ch.content)
            is_code = ch.metadata.chunk_type == ChunkType.CODE if ch.metadata else False
            if not is_code:
                text = _re.sub(r"[^\S\n]{2,}", " ", text)
                text = self._collapse_spaced_heading(text)
                # De-hyphenate line-broken words: "man-\nage" → "manage"
                text = _re.sub(r"(\w)-\n\s*(\w)", r"\1\2", text)
                # Strip orphan trailing hyphens where the continuation is lost
                text = _re.sub(r"(\w)-\s*$", r"\1", text)
                # Split uppercase-to-titlecase merges: "TORetrieval" → "TO Retrieval"
                text = _re.sub(r"([A-Z]{2,})([A-Z][a-z]{2,})", r"\1 \2", text)
            ch.content = text

            # Also fix breadcrumbs containing spaced headings
            if ch.metadata and ch.metadata.hierarchy and ch.metadata.hierarchy.breadcrumb_path:
                ch.metadata.hierarchy.breadcrumb_path = [
                    self._collapse_spaced_heading(b) for b in ch.metadata.hierarchy.breadcrumb_path
                ]
                if ch.metadata.hierarchy.parent_heading:
                    ch.metadata.hierarchy.parent_heading = self._collapse_spaced_heading(
                        ch.metadata.hierarchy.parent_heading
                    )
        return chunks

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

        # Pass 1: soft hyphen (U+00AD) at line break — join WITHOUT hyphen.
        text = re.sub(r"([A-Za-z0-9])\xad\s*\n\s*([a-z])", r"\1\2", text)
        # Pass 2: hard hyphen at line break — join WITH hyphen preserved.
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

    def _sanitize_technical_manual_export_content(self, text: str) -> Optional[str]:
        """Return export-safe technical-manual text, or None if hygiene zeroes it."""
        cleaned = self._strip_control_chars(text or "")
        cleaned = self._remove_standalone_page_number_lines(cleaned)
        cleaned = self._remove_all_digit_only_lines(cleaned)
        cleaned = self._fix_linebreak_hyphenation(cleaned)
        if not cleaned.strip():
            return None
        return cleaned

    def _apply_technical_manual_export_sanitizer(
        self,
        chunks: List[IngestionChunk],
    ) -> List[IngestionChunk]:
        """
        Apply the final technical-manual export sanitizer before metadata is written.

        The JSONL writer used to zero content in the serialized dict and rely on a
        late write-loop skip. Returning None here makes the zeroed-content condition
        explicit and lets the finalize boundary account for dropped chunks.
        """
        out: List[IngestionChunk] = []
        dropped = 0
        for ch in chunks:
            if (
                ch.modality != Modality.TEXT
                or not ch.metadata
                or ch.metadata.profile_type != "technical_manual"
                or ch.metadata.chunk_type == ChunkType.CODE
            ):
                out.append(ch)
                continue

            original = ch.content or ""
            sanitized = self._sanitize_technical_manual_export_content(original)
            if sanitized is None:
                dropped += 1
                logger.info(
                    "[FINALIZE] tech-manual sanitiser drop: "
                    f"chunk_id={ch.chunk_id} page={ch.metadata.page_number}"
                )
                continue
            if sanitized != original:
                ch.content = sanitized
            out.append(ch)

        if dropped:
            logger.info(
                f"[FINALIZE] Dropped {dropped} technical_manual text chunk(s) "
                f"zeroed by export sanitiser before metadata write"
            )
        return out

    # Phase 4 Step 3: shared corruption patterns matching the strict-gate
    # detector in scripts/qa_full_conversion.py. The set is intentionally
    # narrow — magazine-PDF font corruption produces these very specific
    # artifacts; sane prose never matches.
    _CHUNK_CORRUPTION_PATTERNS = (
        re.compile(r"[—]{6,}"),         # 6+ em-dashes in a row
        re.compile(r"[™]{2,}"),         # repeated trademark symbols
        re.compile(r"[CS]{10,}"),            # 10+ Cs/Ss in a row
        re.compile(r"\b[BSQ][0-9]th"),       # garbled ordinals (B5th, S3th, ...)
        re.compile(r"\bFe35\b"),
        re.compile(r"\bF1SC\b"),
        re.compile(r"\bNCOCOC\b"),
    )

    @classmethod
    def _is_corrupted_chunk_content(cls, content: str) -> bool:
        """Detect known unrecoverable extraction-time corruption shapes.

        Mirrors the strict-gate `LOCALIZED_CORRUPTION` detector in
        scripts/qa_full_conversion.py so chunks that would fail the gate
        are dropped at the finalize boundary instead of leaking into the
        JSONL. Replacement-char ratio gate matches the gate's 0.005 threshold.

        Plan v2.9 Phase E refinement (2026-05-11): a structural
        gibberish-density check for large tables. A table chunk with
        > 30 K characters AND fewer than 10 four-letter-or-longer
        English-shape words per 1 K characters is by definition
        gibberish (mostly broken-glyph noise around sparse real
        tokens). Catches the Combat_Aircraft_August_2025 p66 squadron
        roster after the 2026-05-11 reconvert produced a different
        gibberish signature (no em-dashes / CS runs) than the
        original Phase 4 Step 3 patterns target. Corpus probe: 0
        false positives across 18 large table chunks (the other 17
        legitimate tables have word_density 20-54 w/k chars).
        """
        if not content:
            return False
        n = len(content)
        if content.count("�") / max(1, n) > 0.005:
            return True
        if any(p.search(content) for p in cls._CHUNK_CORRUPTION_PATTERNS):
            return True
        # Structural gibberish check (length + word-density)
        if n > 30000:
            import re as _re
            word_density = (
                len(_re.findall(r"\b[A-Za-z]{4,}\b", content)) / (n / 1000.0)
            )
            if word_density < 10.0:
                return True
        return False

    def _drop_corrupted_chunks_before_metadata(
        self,
        chunks: List[IngestionChunk],
    ) -> List[IngestionChunk]:
        """Drop text/table chunks whose content matches strict-gate corruption patterns.

        Combat Aircraft p66 is the canonical example: the source PDF has
        font/encoding corruption that no extraction logic can recover, so
        Docling's table extractor produces a chunk like
        ``'le See Se ee ee ee ... US AirForce Academy,Colorado—“C—si‘“—;s‘“‘“...'``.
        Phase 4 Step 3 quarantines such chunks at the finalize boundary
        instead of letting them ship into the corpus.
        """
        filtered: List[IngestionChunk] = []
        dropped: List[IngestionChunk] = []
        for chunk in chunks:
            if chunk.modality not in (Modality.TEXT, Modality.TABLE):
                filtered.append(chunk)
                continue
            extraction_method = getattr(
                getattr(chunk, "metadata", None), "extraction_method", None
            )
            # Phase 1 dense-index router output (TOC/index chunks with
            # publisher-template U+FFFD replacement-char artifacts in
            # dotted-leader regions) must survive this finalize-stage
            # quarantine for the same reason it must survive
            # `_quarantine_corrupted_text_chunks`. (Plan v2.9 B1, 2026-05-11.)
            if extraction_method and extraction_method.startswith(
                "hybrid_chunker_pageskip"
            ):
                filtered.append(chunk)
                continue
            if self._is_corrupted_chunk_content(chunk.content or ""):
                dropped.append(chunk)
                continue
            filtered.append(chunk)
        if dropped:
            page_counts: Dict[int, int] = {}
            for c in dropped:
                page = getattr(c.metadata, "page_number", None)
                if page is not None:
                    page_counts[page] = page_counts.get(page, 0) + 1
            page_summary = ", ".join(
                f"p{page}={count}" for page, count in sorted(page_counts.items())
            )
            logger.warning(
                "[FINALIZE] Dropped %d corrupted chunk(s) before JSONL write "
                "(strict-gate LOCALIZED_CORRUPTION quarantine; pages: %s)",
                len(dropped),
                page_summary or "n/a",
            )
        return filtered

    def _drop_empty_text_chunks_before_metadata(
        self,
        chunks: List[IngestionChunk],
    ) -> List[IngestionChunk]:
        """Canonical finalize boundary: no empty text chunk reaches metadata/write."""
        filtered = [
            c for c in chunks
            if c.modality != Modality.TEXT or (c.content and c.content.strip())
        ]
        dropped = len(chunks) - len(filtered)
        if dropped:
            logger.info(
                f"[FINALIZE] Dropped {dropped} empty text chunk(s) "
                f"before JSONL write (UNIVERSAL_FAIL safety net)"
            )
        return filtered

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
                r"^(%pip\s+|!?[A-Za-z_][\w\.]*\s*=|from\s+[A-Za-z_][\w\.]*\s+import\b"
                r"|from\s+[A-Za-z_][\w\.]*\s*$"  # bare 'from X' split off from 'from X import Y'
                r"|import\s+[A-Za-z_][\w\.,\s]*$|[A-Za-z_][\w\.]*\s*\()"
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

        Key principle: NEVER strip leading whitespace from code that already has
        line breaks. Indentation is structural meaning in Python.
        """
        import re

        t = (text or "").strip("\n")
        if not t:
            return t
        if "\n" in t:
            # Preserve original lines — do NOT strip leading whitespace.
            # Only rstrip to remove trailing spaces/tabs.
            raw_lines = [ln.rstrip() for ln in t.splitlines()]
            lines = [ln for ln in raw_lines if ln.strip()]
            if not lines:
                return t
            has_repl = any(ln.lstrip().startswith((">>>", "...")) for ln in lines)
            has_indent = any(ln.startswith(("    ", "\t")) for ln in lines)
            if has_repl or has_indent:
                # Code already has structure — preserve exactly
                return "\n".join(raw_lines).strip("\n")
            # Code has newlines but no indentation — check if it looks like code
            stripped_lines = [ln.strip() for ln in lines]
            stmt_like = re.compile(
                r"^(%pip\s+|!?[A-Za-z_][\w\.]*\s*=|from\s+[A-Za-z_][\w\.]*\s+import\b|import\s+[A-Za-z_][\w\.,\s]*|[A-Za-z_][\w\.]*\s*\()"
            )
            codey_lines = sum(
                1
                for ln in stripped_lines
                if stmt_like.search(ln) is not None or re.search(r"[()\[\]{}=]", ln) is not None
            )
            if codey_lines >= max(1, int(len(stripped_lines) * 0.6)):
                return "\n".join(f">>> {ln}" for ln in stripped_lines)
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

    def _sanitize_toc_cell_markers(
        self, chunks: List[IngestionChunk]
    ) -> List[IngestionChunk]:
        """Strip Docling internal cell markers without dropping TOC/index text."""
        import re

        marker = re.compile(r",\s*\d+\s*=")

        def has_marker_noise(content: str) -> bool:
            return len(marker.findall(content or "")) >= 3

        sanitized: List[IngestionChunk] = []
        changed = 0
        for ch in chunks:
            if not ch.content or not ch.content.strip():
                # Preserve - this sanitizer only strips TOC cell markers from
                # TEXT. IMAGE/TABLE chunks carry no text content and must not be
                # dropped here (that silently deleted every image on image-only
                # pages -> MISSING_PAGES); empty TEXT chunks are removed at the
                # canonical boundary by _drop_empty_text_chunks_before_metadata.
                sanitized.append(ch)
                continue
            if ch.modality == Modality.TEXT and has_marker_noise(ch.content):
                cleaned = marker.sub(" ", ch.content)
                cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
                cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
                if not cleaned:
                    continue
                ch.content = cleaned
                try:
                    if ch.metadata and ch.metadata.refined_content:
                        refined = marker.sub(" ", ch.metadata.refined_content)
                        refined = re.sub(r"[ \t]{2,}", " ", refined)
                        refined = re.sub(r"\n{3,}", "\n\n", refined).strip()
                        ch.metadata.refined_content = refined or cleaned
                except Exception:
                    pass
                self._demote_toc_index_chunk(ch)
                changed += 1
            sanitized.append(ch)

        if changed:
            logger.info(
                "[TOC-SANITIZE] Stripped Docling cell markers from %d TOC/index chunk(s)",
                changed,
            )
        return sanitized

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
                    ch.metadata.content_classification = self._classify_text_content(ch.content or "")

            lines = [ln.rstrip() for ln in txt.splitlines() if ln.strip()]
            if not lines:
                return
            has_repl = any(re.search(r"^\s*(>>>|\.\.\.)\s", ln) for ln in lines)
            has_indent = any(ln.startswith(("    ", "\t")) for ln in lines)

            # Keep true REPL/code signals — but guard against book-formatting abuse
            # where ">>>" is used as a cross-reference or prose lead-in marker rather
            # than a Python REPL prompt (e.g. ">>> Input and Output 9.1 ...",
            # ">>> The round() function implements ...").
            if has_repl:
                import keyword as _kw
                repl_lines_content = [
                    ln.lstrip()[4:]  # strip leading ">>> "
                    for ln in lines
                    if ln.lstrip().startswith(">>> ")
                ]
                # Classify each >>> line: is the content after the prompt likely
                # real Python?  A digit start (arithmetic REPL), special char, or
                # a lowercase non-keyword token (variable / function name) are
                # reliable signals.  Uppercase starts are book prose.
                any_real_repl_start = any(
                    rc[:1].isdigit()
                    or rc[:1] in "_'\"-+~"
                    or (
                        rc
                        and rc[:1].islower()
                        and not _kw.iskeyword(
                            rc.split()[0].rstrip(":([,=") if rc.split() else ""
                        )
                    )
                    for rc in repl_lines_content
                    if rc
                )
                if any_real_repl_start or self._looks_like_code_text(txt):
                    return
                # >>> markers present but content is prose/index (not real REPL).
                # is_code_line() can still trigger on incidental brackets in prose
                # (e.g. "round()" in ">>> The round() function implements …"),
                # so demote directly rather than relying on the scoring fallback.
                _demote()
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
                # Structural keywords that form valid statements without ':' on the same line.
                if re.search(r"^\s*(def|class|return|yield|async\s+def|await)\b", ln):
                    return True
                # Flow-control keywords require ':' on the same line to distinguish from prose.
                # "if the control is desired" has no ':' → prose, not code.
                # "if x > 0:" has ':' → code line.
                if re.search(r"^\s*(if|elif|else|for|while|try|except|with)\b", ln) and ":" in ln:
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
                    not is_code_line(ln)
                    and len(re.findall(r"[A-Za-z]{2,}", ln)) >= 6
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

        v2.10 Phase 4: cross-page-split fallback marker chunks
        (`extraction_method == "hybrid_chunker_pagesplit_fallback"`,
        content `[CROSS_PAGE_CONTINUED]`) are *sentinel* outputs of the
        cross-page split emergency path. They must stay standalone — if
        we merge them into a neighbor, the marker string contaminates
        retrievable prose (Python_Distilled p472 saw the marker
        appended onto the "Set Operations" table chunk before this
        guard was added).
        """

        def is_pagesplit_fallback(ch: IngestionChunk) -> bool:
            method = getattr(getattr(ch, "metadata", None), "extraction_method", "") or ""
            return method == "hybrid_chunker_pagesplit_fallback"

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
                or is_pagesplit_fallback(cur)
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
                    and not is_pagesplit_fallback(nxt)
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
                    and not is_pagesplit_fallback(prev)
                    and page_of(prev) == cur_page
                    and (prev.content or "").strip()
                ):
                    prev.content = f"{(prev.content or '').rstrip()} {cur_text}".strip()
                    i += 1
                    continue

            out.append(cur)
            i += 1

        return out

    # Common English word suffixes that can appear at the START of a chunk when a
    # two-column PDF splits a hyphenated word across a column boundary (Packt books).
    # Only suffixes in this whitelist trigger cross-chunk rejoining (conservative).
    _WORD_FRAGMENT_SUFFIXES: frozenset = frozenset({
        "ment", "ments", "ness", "tion", "tions", "sion", "sions",
        "ing", "ings", "ated", "ates", "ized", "izes",
        "ance", "ences", "ence", "ances", "ible", "able",
        "ward", "wards", "wise", "ure", "ures", "ive", "ives",
        "ary", "ory", "ery", "nge", "uce", "rames", "ted", "ent",
        "ers", "ler", "ding", "king", "ling", "ring", "sing", "ting",
        "ched", "shed", "ies", "ied",
    })

    def _rejoin_leading_word_fragments(
        self,
        chunks: List[IngestionChunk],
    ) -> List[IngestionChunk]:
        """
        Rejoin cross-chunk word fragments caused by Packt two-column hyphenation.

        When Docling extracts two-column PDFs, a word split at a column break
        sometimes lands in two separate chunks:
          chunk[i]   ends: "... introd"    (no terminal punctuation, ends lowercase)
          chunk[i+1] starts: "uce to..."   (≤6 lowercase chars in whitelist)

        Guards (ALL must pass):
        1. chunk[i]   is TEXT, non-code, ends with a lowercase letter (no terminal punct).
        2. chunk[i+1] is TEXT, non-code, on the same page as chunk[i].
        3. Leading token of chunk[i+1] is ≤6 lowercase chars AND in suffix whitelist.
        4. chunk[i+1] has more content beyond the leading fragment (≥2 tokens total).
        """
        import re

        def _is_code(ch: IngestionChunk) -> bool:
            try:
                return (
                    ch.metadata.chunk_type == ChunkType.CODE
                    or ch.metadata.content_classification == "code"
                )
            except Exception:
                return False

        _ends_lower = re.compile(r"[a-z]$")
        result: List[IngestionChunk] = []
        rejoined = 0
        i = 0

        while i < len(chunks):
            cur = chunks[i]
            if (
                i == len(chunks) - 1
                or cur.modality != Modality.TEXT
                or _is_code(cur)
                or not cur.content
            ):
                result.append(cur)
                i += 1
                continue

            nxt = chunks[i + 1]
            if (
                nxt.modality != Modality.TEXT
                or _is_code(nxt)
                or not nxt.content
            ):
                result.append(cur)
                i += 1
                continue

            try:
                same_page = cur.metadata.page_number == nxt.metadata.page_number
            except Exception:
                same_page = False

            if not same_page or not _ends_lower.search(cur.content.rstrip()):
                result.append(cur)
                i += 1
                continue

            nxt_tokens = nxt.content.lstrip().split()
            if len(nxt_tokens) < 2:
                result.append(cur)
                i += 1
                continue

            leading_alpha = nxt_tokens[0].rstrip(".,;:!?")
            if not (
                1 <= len(leading_alpha) <= 6
                and leading_alpha.islower()
                and leading_alpha in self._WORD_FRAGMENT_SUFFIXES
            ):
                result.append(cur)
                i += 1
                continue

            # All guards passed: append fragment directly — no space, continues the word.
            cur.content = cur.content.rstrip() + nxt.content.lstrip()
            try:
                if (
                    cur.semantic_context
                    and nxt.semantic_context
                    and nxt.semantic_context.next_text_snippet
                ):
                    cur.semantic_context.next_text_snippet = (
                        nxt.semantic_context.next_text_snippet
                    )
            except Exception:
                pass
            rejoined += 1
            i += 2
            result.append(cur)

        if rejoined:
            logger.info(
                f"[FRAGMENT-REJOIN] Rejoined {rejoined} cross-chunk word fragments"
            )
        return result

    def _remove_subset_chunks(self, chunks: List[IngestionChunk]) -> List[IngestionChunk]:
        """Remove text chunks that are subsets of another chunk on the same page.

        When VLM transcription and Docling both produce text for the same page,
        one may be a subset of the other. Keep only the longer version.
        """
        import re as _re

        # Group text chunks by page
        page_chunks: dict[int, list[int]] = {}
        for i, ch in enumerate(chunks):
            if ch.modality == Modality.TEXT and ch.metadata and ch.metadata.page_number:
                page_chunks.setdefault(ch.metadata.page_number, []).append(i)

        drop_indices: set[int] = set()
        for pg, indices in page_chunks.items():
            if len(indices) < 2:
                continue
            for a_idx in indices:
                if a_idx in drop_indices:
                    continue
                a_words = set(_re.findall(r"[a-zA-Z]{3,}", chunks[a_idx].content.lower()))
                for b_idx in indices:
                    if b_idx <= a_idx or b_idx in drop_indices:
                        continue
                    b_words = set(_re.findall(r"[a-zA-Z]{3,}", chunks[b_idx].content.lower()))
                    if not a_words or not b_words:
                        continue
                    # If shorter is >80% contained in longer, drop the shorter
                    shorter, longer = (a_words, b_words) if len(a_words) < len(b_words) else (b_words, a_words)
                    overlap = len(shorter & longer) / len(shorter)
                    if overlap > 0.80:
                        drop_idx = a_idx if len(a_words) < len(b_words) else b_idx
                        drop_indices.add(drop_idx)

        if drop_indices:
            logger.info(f"[SUBSET-DEDUP] Removed {len(drop_indices)} subset text chunks")

        return [ch for i, ch in enumerate(chunks) if i not in drop_indices]

    def _infer_headings_from_text(self, chunks: List[IngestionChunk]) -> List[IngestionChunk]:
        """Infer section headings from text content for scanned documents.

        Scanned documents have flat hierarchy because Docling can't detect
        headings from pixel-based extraction. This method looks for heading
        patterns in the text and assigns them as parent_heading/breadcrumb.
        """
        import re as _re

        current_heading: Optional[str] = None
        updated = 0

        for ch in chunks:
            if ch.modality != Modality.TEXT or not ch.content:
                continue

            # Look for heading patterns at the start of a chunk:
            # - All-caps line (e.g., "MAUSER 1898", "TOOLS", "INTRODUCTION")
            # - Title-case with model number (e.g., "British SMLE No. 1, MKIII")
            first_line = ch.content.strip().split("\n")[0].strip()

            is_heading = False
            if (
                len(first_line) > 3
                and len(first_line) < 60
                and first_line == first_line.upper()
                and any(c.isalpha() for c in first_line)
                and not first_line.startswith(("THE ", "A ", "IN ", "ON ", "TO "))
            ):
                # All-caps heading: "MAUSER 1898", "TOOLS", "DISASSEMBLY"
                is_heading = True
            elif _re.match(r"^[A-Z][a-z].*(?:No\.|Model|Mk|Type|Mark)\s", first_line):
                # Title-case with model identifier
                is_heading = True

            if is_heading:
                current_heading = first_line

            if current_heading and ch.metadata and ch.metadata.hierarchy:
                old_heading = ch.metadata.hierarchy.parent_heading
                if not old_heading or old_heading == "Firearms" or old_heading == ch.metadata.source_file:
                    ch.metadata.hierarchy.parent_heading = current_heading
                    # Update breadcrumb
                    bp = ch.metadata.hierarchy.breadcrumb_path
                    if bp and len(bp) >= 2:
                        ch.metadata.hierarchy.breadcrumb_path = [bp[0], current_heading, bp[-1]]
                    updated += 1

        if updated:
            logger.info(f"[HEADING-INFER] Assigned inferred headings to {updated} chunks")

        return chunks

    def _extract_embedded_images(
        self,
        batch_path: Path,
        page_offset: int,
        source_file: str,
    ) -> List[IngestionChunk]:
        """Extract embedded image objects from a digital PDF using PyMuPDF.

        For native_digital and image_heavy documents, the PDF contains discrete
        embedded image objects (photos, diagrams, charts) with exact coordinates.
        This is more accurate than Docling's layout model which guesses image
        regions and often captures surrounding text.

        Filters:
        - Minimum size: 100x100 pixels (skip icons, bullets, decorations)
        - Maximum area ratio: skip full-page background images (>90% of page)
        """
        import fitz
        from .schema.ingestion_schema import (
            create_image_chunk, FileType, HierarchyMetadata, COORD_SCALE,
        )

        chunks: List[IngestionChunk] = []
        doc_hash = self._doc_hash or "unknown"
        intel = self._intelligence_metadata or {}

        try:
            pdf = fitz.open(str(batch_path))
        except Exception as e:
            logger.warning(f"[PYMUPDF-IMAGES] Could not open {batch_path}: {e}")
            return chunks

        fig_index = 0

        for page_idx in range(len(pdf)):
            page = pdf.load_page(page_idx)
            actual_page = page_idx + 1 + page_offset
            page_w, page_h = page.rect.width, page.rect.height
            page_area = page_w * page_h

            for img_info in page.get_images(full=True):
                xref = img_info[0]
                img_w, img_h = img_info[2], img_info[3]

                # Skip small images (icons, bullets, logos)
                if img_w < 100 or img_h < 100:
                    continue

                # Extract the image
                try:
                    pix = fitz.Pixmap(pdf, xref)
                    # Convert CMYK to RGB
                    if pix.n > 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    elif pix.n == 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                except Exception as e:
                    logger.debug(f"[PYMUPDF-IMAGES] Could not extract xref {xref}: {e}")
                    continue

                # Skip solid-color placeholder images (all black, all white)
                samples = pix.samples
                if len(set(samples[:1000])) <= 1:  # Check first 1000 bytes
                    pix = None
                    continue

                # Skip background/container images. A background image
                # has other images placed ON TOP of it (overlapping
                # children). This reliably detects page backgrounds
                # without arbitrary size thresholds.
                all_rects = {}
                for other_img in page.get_images(full=True):
                    other_xref = other_img[0]
                    ow, oh = other_img[2], other_img[3]
                    if ow < 100 or oh < 100:
                        continue
                    other_rects = page.get_image_rects(other_xref)
                    if other_rects:
                        all_rects[other_xref] = other_rects[0]

                my_rect = all_rects.get(xref)
                is_background = False
                if my_rect:
                    for other_xref, other_rect in all_rects.items():
                        if other_xref == xref:
                            continue
                        # Check if other image is INSIDE this one
                        if (other_rect.x0 >= my_rect.x0 - 5
                            and other_rect.y0 >= my_rect.y0 - 5
                            and other_rect.x1 <= my_rect.x1 + 5
                            and other_rect.y1 <= my_rect.y1 + 5
                            and other_rect.width * other_rect.height
                                < my_rect.width * my_rect.height * 0.9):
                            is_background = True
                            break

                if is_background:
                    logger.debug(
                        f"[PYMUPDF-IMAGES] Skipping full-page image on pg {actual_page} "
                        f"({pix.width}x{pix.height})"
                    )
                    continue

                # Save to disk
                fig_index += 1
                asset_name = f"{doc_hash}_{actual_page:03d}_photo_{fig_index:02d}.png"
                asset_path = f"assets/{asset_name}"
                full_path = self.output_dir / "assets" / asset_name
                full_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    pix.save(str(full_path))
                except Exception as e:
                    logger.debug(f"[PYMUPDF-IMAGES] Could not save {asset_name}: {e}")
                    continue
                finally:
                    pix = None

                # Get image position on page for bbox
                img_rects = page.get_image_rects(xref)
                if img_rects:
                    rect = img_rects[0]
                    bbox = [
                        int(rect.x0 / page_w * COORD_SCALE),
                        int(rect.y0 / page_h * COORD_SCALE),
                        int(rect.x1 / page_w * COORD_SCALE),
                        int(rect.y1 / page_h * COORD_SCALE),
                    ]
                    bbox = [max(0, min(COORD_SCALE, v)) for v in bbox]
                else:
                    bbox = [0, 0, COORD_SCALE, COORD_SCALE]

                # Charter §3.2 Phase A step 5 site 2 (pymupdf image emit):
                # build UIRChunk with IMAGE modality + BBOX locator, emit via
                # from_uir, then post-construction enrich AssetReference with
                # width/height (from_uir only sets file_path + mime_type).
                _image_uir = UIRChunk(
                    modality=Modality.IMAGE,
                    content=f"[Embedded image on page {actual_page}]",
                    locator=UIRLocator(
                        type=UIRLocatorType.BBOX,
                        bbox=bbox,
                        page_number=actual_page,
                        coordinate_frame=UIRCoordinateFrame.PDF_PAGE_PORTRAIT,
                    ),
                    confidence=UIRConfidenceBreakdown(),
                    extraction_method="pymupdf",
                    extraction_engine_version="pymupdf",
                    asset_ref=asset_path,
                )
                chunk = IngestionChunk.from_uir(
                    _image_uir,
                    doc_id=doc_hash,
                    source_file=source_file,
                    file_type=FileType.PDF,
                    position=self._next_chunk_position(),
                    page_width=int(page_w),
                    page_height=int(page_h),
                    profile_type=intel.get("profile_type"),
                    profile_sensitivity=intel.get("profile_sensitivity"),
                    min_image_dims=intel.get("min_image_dims"),
                    confidence_threshold=intel.get("confidence_threshold"),
                    document_domain=intel.get("document_domain"),
                    document_modality=intel.get("document_modality"),
                )
                # v2.16 invariant: image AssetReference carries pixel dimensions.
                if chunk.asset_ref is not None:
                    chunk.asset_ref.width_px = img_w
                    chunk.asset_ref.height_px = img_h
                chunks.append(chunk)

                logger.debug(
                    f"[PYMUPDF-IMAGES] pg{actual_page}: {asset_name} "
                    f"({img_w}x{img_h})"
                )

        pdf.close()
        return chunks

    @staticmethod
    def _extract_toc_headings(pdf_path: Path) -> Dict[int, List[str]]:
        """Extract TOC data from the PDF.

        Returns two structures:
        - page_map: page → breadcrumb at end of page [Part, Chapter, Section]
        - heading_map: heading_title → full breadcrumb at the point where it appears

        The heading_map is key: it maps each heading to its breadcrumb context,
        so we can look up the correct hierarchy for a HybridChunker heading
        regardless of which page the chunk is on.
        """
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            toc = doc.get_toc()  # [(level, title, page), ...]
            doc.close()

            if not toc:
                doc2 = fitz.open(str(pdf_path))
                result = BatchProcessor._extract_toc_from_content(doc2)
                doc2.close()
                return result

            entries = [
                (page, level, title.strip())
                for level, title, page in toc
                if title.strip() and len(title.strip()) <= 80
            ]
            if not entries:
                return {}

            # Build TWO maps:
            # 1. page_map: page → breadcrumb (state at END of page, for chunks with no heading)
            # 2. heading_map: normalized_title → breadcrumb (state WHEN that heading appears)

            heading_map: Dict[str, List[str]] = {}  # norm_title → breadcrumb
            page_map: Dict[int, List[str]] = {}

            active: Dict[int, str] = {}
            max_page = max(p for p, _, _ in entries) + 50

            for entry_page, entry_level, entry_title in entries:
                active[entry_level] = entry_title
                for deeper in list(active.keys()):
                    if deeper > entry_level:
                        del active[deeper]
                # Snapshot the breadcrumb at the moment this heading appears
                bc = [active[lvl] for lvl in sorted(active.keys())]
                import re as _re_tn
                norm = _re_tn.sub(r"\s+", " ", entry_title.replace("\xa0", " ")).strip()
                heading_map[norm] = bc

            # Build page map (state at end of each page)
            active = {}
            for pg in range(1, max_page + 1):
                for entry_page, entry_level, entry_title in entries:
                    if entry_page > pg:
                        break
                    active[entry_level] = entry_title
                    for deeper in list(active.keys()):
                        if deeper > entry_level:
                            del active[deeper]
                if active:
                    page_map[pg] = [active[lvl] for lvl in sorted(active.keys())]

            logger.info(
                f"[TOC] Extracted {len(entries)} TOC entries, "
                f"{len(heading_map)} unique headings"
            )

            # Store both maps — page_map as the return value, heading_map as attribute
            # We'll access heading_map via a class attribute set after this call
            page_map["__heading_map__"] = heading_map  # type: ignore[assignment]
            return page_map

        except Exception as e:
            logger.debug(f"[TOC] Could not extract TOC: {e}")
            return {}

    @staticmethod
    def _extract_toc_from_content(doc) -> Dict[int, List[str]]:
        """Extract article titles from magazine TOC pages when no PDF bookmarks exist.

        Scans pages 1-15 for TOC-like content: lines matching NUMBER TITLE
        or TITLE NUMBER. Finds the page with the highest density of matches
        (the actual TOC page) and uses only entries from that page.

        Returns the same page_map + heading_map format as _extract_toc_headings.
        Returns empty dict if confidence is too low (< 50% of entries valid).
        """
        import re

        total_pages = len(doc)
        if total_pages < 5:
            return {}

        # Scan pages 1-15 for TOC candidates
        page_candidates: Dict[int, list] = {}  # page_idx → list of (ref_pg, title)
        for pg_idx in range(min(15, total_pages)):
            text = doc.load_page(pg_idx).get_text("text")
            entries = []
            for line in text.split("\n"):
                line = line.strip()
                if not line or len(line) < 5:
                    continue
                # Pattern: NUMBER TITLE (e.g., "28 Dancing with the Devils")
                m = re.match(r"^(\d{1,3})\s+(.{5,60})$", line)
                if m:
                    ref_pg = int(m.group(1))
                    title = m.group(2).strip()
                    # Page ref must be within document and after this page
                    if pg_idx + 1 < ref_pg <= total_pages:
                        entries.append((ref_pg, title))
            if entries:
                page_candidates[pg_idx + 1] = entries

        if not page_candidates:
            logger.info("[TOC-CONTENT] No TOC candidates found on pages 1-15")
            return {}

        # Find the page with the most entries — that's the actual TOC page
        toc_page = max(page_candidates, key=lambda p: len(page_candidates[p]))
        entries = page_candidates[toc_page]

        # Confidence check: need at least 3 entries
        if len(entries) < 3:
            logger.info(
                f"[TOC-CONTENT] Only {len(entries)} entries on page {toc_page} — "
                f"below confidence threshold, falling back to document-level breadcrumbs"
            )
            return {}

        logger.info(
            f"[TOC-CONTENT] Found {len(entries)} article entries on page {toc_page}"
        )

        # Sort entries by page number to build article ranges
        entries.sort(key=lambda e: e[0])

        # Build page_map: each page maps to its article title (flat, L1 only)
        # Article A runs from its start page to the page before article B starts
        heading_map: Dict[str, List[str]] = {}
        page_map: Dict[int, List[str]] = {}

        for i, (start_pg, title) in enumerate(entries):
            end_pg = entries[i + 1][0] - 1 if i + 1 < len(entries) else total_pages
            norm_title = re.sub(r"\s+", " ", title.replace("\xa0", " ")).strip()
            heading_map[norm_title] = [title]  # flat hierarchy for magazines
            for pg in range(start_pg, end_pg + 1):
                page_map[pg] = [title]

        logger.info(
            f"[TOC-CONTENT] Built article ranges for {len(entries)} articles, "
            f"covering pages {entries[0][0]}-{entries[-1][0]}"
        )

        page_map["__heading_map__"] = heading_map  # type: ignore[assignment]
        return page_map

    def _attribute_ocr_chunk_heading(
        self,
        pc: Any,
    ) -> Optional[str]:
        """Phase 6 (PLAN_V2.10) — ordered per-chunk heading attribution
        on the OCR/element-by-element lane.

        Called once per ``ProcessedChunk`` in the order returned by
        :meth:`LayoutAwareOCRProcessor.process_page`. If the chunk
        carries ``is_heading=True`` (Docling labelled the source element
        ``section_header`` or ``title``), the chunk's content is pushed
        into ``self._context_state`` via ``update_on_heading`` (which
        validates through ``is_valid_heading`` and silently rejects
        garbage). The current effective heading is then read back via
        :meth:`ContextStateV2.get_section_heading`, giving:

          - body chunks BEFORE the first heading on a page inherit the
            previously-active heading (or ``None`` if none exists);
          - body chunks AFTER a heading on the same page inherit that
            heading;
          - multiple headings on one page switch attribution at the
            right ordinal position;
          - heading chunks attribute to themselves (the legitimate
            "chunk that defines the heading" case noted in the plan);
          - garbage section_header text (Phase 5 / Phase 6 audit
            shapes) does not displace a prior valid heading because
            ``update_on_heading`` rejects it before pushing.

        The page-level fallback ``_promote_ocr_section_headers`` is
        used ONLY when no chunk in the stream carried
        ``is_heading=True`` (VLM-fullpage / Tesseract-fullpage paths
        emit a single synthesized chunk per page).

        Multi-page-doc gate. The push targets inter-page heading
        propagation; on a single-page document there is no cross-page
        propagation to do, and pushing a Docling-tagged section_header
        on a form/invoice (canonical 0013-shape: layout-prominent
        invoice total or field-label-value line) flips the downstream
        form-detection heuristic in ``scripts/qa_conversion_audit.py``.
        State READS still happen so prior-state from earlier pages
        (none, on a single-page doc) remains accessible.
        """
        state = self._context_state
        if state is None:
            return None
        if (
            getattr(pc, "is_heading", False)
            and self._doc_total_pages
            and self._doc_total_pages > 1
        ):
            content = getattr(pc, "content", None)
            if content and content.strip():
                state.update_on_heading(content.strip(), level=1)
        return state.get_section_heading()

    def _should_prescan_ocr_headings(
        self,
        docling_elements: Optional[Iterable[Any]],
    ) -> bool:
        """Phase 6 (PLAN_V2.10) — multi-page-doc gate for OCR-lane
        heading state pushes.

        The push targets inter-page heading propagation; on a single-
        page document there is no cross-page propagation to do, and
        pushing a Docling-tagged ``section_header`` on a form/invoice
        (canonical 0013-shape: layout-prominent invoice total or field-
        label-value line) flips the downstream form-detection heuristic
        in ``scripts/qa_conversion_audit.py``
        (``form := scanned + total_pages ≤ 5 + heading_coverage <
        0.10``).
        """
        if not docling_elements:
            return False
        if self._context_state is None:
            return False
        if not self._doc_total_pages or self._doc_total_pages <= 1:
            return False
        return True

    def _promote_ocr_section_headers(
        self,
        docling_elements: Optional[Iterable[Any]],
    ) -> None:
        """Phase 6 (PLAN_V2.10) — VLM/Tesseract-fullpage fallback push.

        Ordered per-chunk heading attribution in
        :meth:`_process_page_layout_aware` is the canonical OCR-lane
        path; it walks ``processed_chunks`` and pushes
        ``section_header`` / ``title`` items into ``self._context_state``
        at the exact ordinal position they appear within the page (via
        ``ProcessedChunk.is_heading``). On the rare paths where
        ``LayoutAwareOCRProcessor.process_page`` collapses an entire
        page into a single synthesized chunk (VLM full-page
        transcription; full-page Tesseract baseline on text-only pages),
        the per-chunk walk sees no ``is_heading=True`` markers and the
        page's Docling-recognised section_headers would be silently
        dropped from state — breaking cross-page propagation. This
        helper is invoked only on those fallback paths to re-establish
        state.

        Validation is centralised: ``update_on_heading`` re-uses the
        ``is_valid_heading`` validator in ``state.context_state``. No
        parallel OCR-lane validator. This is the OCR lane's analogue of
        the HybridChunker lane's :meth:`_propagate_headings`; the two
        lanes are independent and do not share a propagation site (the
        HybridChunker lane's call site in ``process_pdf`` stays at
        exactly one occurrence).
        """
        state = self._context_state
        if not docling_elements or state is None:
            return
        if not self._doc_total_pages or self._doc_total_pages <= 1:
            return
        for elem in docling_elements:
            label_obj = getattr(elem, "label", None)
            if label_obj is None:
                continue
            label_val = (
                str(label_obj.value)
                if hasattr(label_obj, "value")
                else str(label_obj)
            )
            if label_val not in ("section_header", "title"):
                continue
            heading_text = str(getattr(elem, "text", "")).strip()
            if not heading_text:
                continue
            # update_on_heading() validates via is_valid_heading and silently
            # rejects garbage (repeated tokens, code/JSON shapes, page-number
            # patterns, etc.), so a single call site is enough.
            state.update_on_heading(heading_text, level=1)

    def _propagate_headings(self, chunks: List[IngestionChunk]) -> List[IngestionChunk]:
        """Assign heading hierarchy from the PDF's table of contents.

        The TOC provides the authoritative document structure: Part > Chapter >
        Section > Subsection. For each chunk, the breadcrumb is built from the
        TOC based on page number. The parent_heading is the deepest (most
        specific) level.

        When HybridChunker already assigned a heading that appears in the TOC
        hierarchy for that page, it's kept as the parent_heading (it's
        position-accurate for page-boundary cases).

        Falls back to forward-propagation when no TOC is available.
        """
        from .state.context_state import is_valid_heading

        _GENERIC_CARRY_HEADINGS = {"start", "front matter"}
        _rejected_heading_chunk_ids: set[int] = set()

        def _norm_heading(s: str) -> str:
            import re as _re_norm
            return _re_norm.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()

        def _is_informative_heading(heading: Optional[str]) -> bool:
            if not heading or not is_valid_heading(heading):
                return False
            return _norm_heading(heading).lower() not in _GENERIC_CARRY_HEADINGS

        def _sanitize_existing_heading(ch: IngestionChunk) -> Optional[str]:
            heading = ch.metadata.hierarchy.parent_heading
            if not heading:
                return None
            if _is_informative_heading(heading):
                return heading
            # Invalid headings are removed rather than allowed to seed carry
            # state. Generic labels may remain on their own chunk but do not
            # propagate to neighboring HybridChunker chunks.
            if not is_valid_heading(heading) or _norm_heading(heading).lower() == "start":
                _rejected_heading_chunk_ids.add(id(ch))
                ch.metadata.hierarchy.parent_heading = None
                bp = ch.metadata.hierarchy.breadcrumb_path or []
                if len(bp) >= 2:
                    ch.metadata.hierarchy.breadcrumb_path = [bp[0], bp[-1]]
                    ch.metadata.hierarchy.level = min(len(ch.metadata.hierarchy.breadcrumb_path), 5)
                return None
            return heading

        def _apply_forward_heading(
            ch: IngestionChunk,
            heading: str,
        ) -> None:
            ch.metadata.hierarchy.parent_heading = heading
            bp = ch.metadata.hierarchy.breadcrumb_path
            if bp and len(bp) >= 2:
                ch.metadata.hierarchy.breadcrumb_path = [bp[0], heading, bp[-1]]
                ch.metadata.hierarchy.level = min(len(ch.metadata.hierarchy.breadcrumb_path), 5)

        def _is_hybrid_text_without_heading(ch: IngestionChunk) -> bool:
            if id(ch) in _rejected_heading_chunk_ids:
                return False
            if ch.modality != Modality.TEXT or not ch.metadata or not ch.metadata.hierarchy:
                return False
            if ch.metadata.hierarchy.parent_heading:
                return False
            method = ch.metadata.extraction_method or ""
            return method.startswith("hybrid_chunker")

        def _page_of(ch: IngestionChunk) -> int:
            try:
                return int(ch.metadata.page_number or 0)
            except Exception:
                return 0

        def _fill_remaining_hybrid_page_context() -> int:
            """Fill unordered HybridChunker page-split siblings.

            Some generated technical manuals expose pages where an explicit
            HybridChunker heading chunk exists, but some page-split siblings
            occur earlier in the in-memory list. A single forward scan misses
            those siblings. Page-scoped carry uses the explicit heading found
            anywhere on that page, then carries that context to later pages
            that have only page-split continuations.
            """
            by_page: Dict[int, List[IngestionChunk]] = {}
            for item in chunks:
                if item.modality != Modality.TEXT or not item.metadata or not item.metadata.hierarchy:
                    continue
                page = _page_of(item)
                if page > 0:
                    by_page.setdefault(page, []).append(item)

            filled = 0
            carry: Optional[str] = None
            for page in sorted(by_page):
                page_chunks = by_page[page]
                page_heading: Optional[str] = None
                for item in page_chunks:
                    heading = _sanitize_existing_heading(item)
                    if _is_informative_heading(heading):
                        page_heading = heading
                        break

                effective_heading = page_heading or carry
                if effective_heading:
                    for item in page_chunks:
                        if _is_hybrid_text_without_heading(item):
                            _apply_forward_heading(item, effective_heading)
                            filled += 1

                if _is_informative_heading(page_heading):
                    carry = page_heading

            return filled

        toc = getattr(self, "_toc_headings", {}) or {}
        assigned = 0

        if toc:
            def _norm(s: str) -> str:
                import re as _re_norm
                return _re_norm.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()

            # heading_map: norm_title → breadcrumb at that heading's definition point
            heading_map: Dict[str, List[str]] = toc.get("__heading_map__", {})  # type: ignore[assignment]
            fallback_assigned = 0
            last_heading: Optional[str] = None

            for ch in chunks:
                if not ch.metadata or not ch.metadata.hierarchy:
                    continue

                pg = ch.metadata.page_number
                toc_breadcrumb = toc.get(pg)
                if not toc_breadcrumb:
                    # Partial TOCs are common in generated PDFs. A truthy but
                    # page-incomplete TOC must not disable the HybridChunker
                    # forward-propagation fallback for pages outside the TOC map.
                    if ch.modality != Modality.TEXT:
                        continue
                    ph = _sanitize_existing_heading(ch)
                    if _is_informative_heading(ph):
                        last_heading = ph
                    elif last_heading and _is_informative_heading(last_heading):
                        _apply_forward_heading(ch, last_heading)
                        assigned += 1
                        fallback_assigned += 1
                    continue

                existing_bp = ch.metadata.hierarchy.breadcrumb_path
                doc_name = existing_bp[0] if existing_bp else "Document"
                current_heading = _sanitize_existing_heading(ch)

                if current_heading:
                    current_norm = _norm(current_heading)
                    heading_bc = heading_map.get(current_norm)
                    page_bc_norm = [_norm(b) for b in toc_breadcrumb]

                    if heading_bc:
                        heading_bc_norm = [_norm(b) for b in heading_bc]
                        # The HybridChunker heading is only valid if it's a
                        # child of the current page's TOC section. If its
                        # parent chain belongs to a different section, it's
                        # a stale heading from a previous batch boundary.
                        is_child = all(
                            hb in page_bc_norm
                            for hb in heading_bc_norm
                            if hb != current_norm
                        )
                        if is_child:
                            # Valid: heading is within the page's section
                            parent = current_heading
                            new_bp = [doc_name] + heading_bc + [f"Page {pg}"]
                        else:
                            # Stale: heading belongs to a different section
                            parent = toc_breadcrumb[-1]
                            new_bp = [doc_name] + toc_breadcrumb + [f"Page {pg}"]
                    elif current_norm in page_bc_norm:
                        # Heading not in heading_map but matches a page TOC entry
                        parent = current_heading
                        new_bp = [doc_name] + toc_breadcrumb + [f"Page {pg}"]
                    else:
                        # Heading not in TOC at all. Use the TOC page heading
                        # instead — the HybridChunker heading is likely OCR
                        # noise from stylized magazine text or a misclassified
                        # paragraph. The TOC article title is more reliable.
                        parent = toc_breadcrumb[-1]
                        new_bp = [doc_name] + toc_breadcrumb + [f"Page {pg}"]
                else:
                    parent = toc_breadcrumb[-1]
                    new_bp = [doc_name] + toc_breadcrumb + [f"Page {pg}"]

                if not _is_informative_heading(parent):
                    continue

                ch.metadata.hierarchy.parent_heading = parent
                ch.metadata.hierarchy.breadcrumb_path = new_bp
                ch.metadata.hierarchy.level = min(len(new_bp), 5)
                if ch.modality == Modality.TEXT:
                    last_heading = parent
                assigned += 1

            if fallback_assigned:
                logger.info(
                    "[HEADING-PROPAGATE] Assigned headings to %s chunks via "
                    "forward-propagation fallback outside TOC coverage",
                    fallback_assigned,
                )

        else:
            # No TOC — forward-propagate last known heading
            last_heading: Optional[str] = None
            for ch in chunks:
                if ch.modality != Modality.TEXT or not ch.metadata or not ch.metadata.hierarchy:
                    continue
                ph = _sanitize_existing_heading(ch)
                if _is_informative_heading(ph):
                    last_heading = ph
                elif last_heading and _is_informative_heading(last_heading):
                    _apply_forward_heading(ch, last_heading)
                    assigned += 1

        page_context_assigned = _fill_remaining_hybrid_page_context()
        if page_context_assigned:
            assigned += page_context_assigned
            logger.info(
                "[HEADING-PROPAGATE] Assigned headings to %s unordered "
                "HybridChunker page-split chunks via page context",
                page_context_assigned,
            )

        if assigned:
            source = "TOC hierarchy" if toc else "forward-propagation"
            logger.info(f"[HEADING-PROPAGATE] Assigned headings to {assigned} chunks via {source}")

        return chunks

    def _calculate_actual_avg_text(self, chunks: List["IngestionChunk"]) -> float:
        """Calculate avg text chars per page from actual extracted chunks."""
        page_chars: dict[int, int] = {}
        for ch in chunks:
            if ch.modality != Modality.TEXT or not ch.content or not ch.metadata:
                continue
            pg = ch.metadata.page_number or 0
            if pg > 0:
                page_chars[pg] = page_chars.get(pg, 0) + len(ch.content)
        if not page_chars:
            return self._doc_avg_text_per_page or 0.0
        return sum(page_chars.values()) / len(page_chars)

    # Phase 6 (PLAN_V2.10) audit follow-up — embedded step-number
    # repair. Matches the canonical Firearms-shape OCR artifact where
    # a numbered-instruction-step marker ("2.", "12.", …) is mashed
    # mid-sentence into the trailing word of the preceding paragraph
    # ("release the trigger 12. forsemgvaupwaros").
    #
    # Detection behaviorally mirrors the audit detector
    # ``scripts/qa_conversion_audit.py::_INFIX_RE`` AFTER its
    # newline / stop-word post-filters, not byte-for-byte at the raw
    # regex level. Specifically:
    #
    #   - The audit uses ``\s+`` between prev and num and then drops
    #     any hit whose ``between`` substring contains a newline. The
    #     production regex collapses both steps into a single
    #     ``[ \t]+`` (non-newline whitespace), which is equivalent
    #     after the audit's post-filter.
    #   - The audit's left-context check excludes prev preceded by
    #     ``\n``, ``\r``, ``". "``, ``": "``, ``"; "``, ``"! "``, or
    #     ``"? "``. The production regex reproduces those exclusions
    #     as zero-width lookbehinds.
    #   - The audit's short-word and stop-word filters are reproduced
    #     in ``_INFIX_STOP_PREV`` / ``_INFIX_STOP_NEXT`` and the
    #     ``len(...) <= 1`` checks inside ``_replace_one``.
    #
    # The audit-detector parity test in
    # ``tests/test_infix_step_number_repair.py`` re-applies the audit
    # detector to repaired chunk content and asserts the count drops
    # to zero, pinning the behavioural equivalence directly.
    _INFIX_STEP_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?<!\n)(?<!\r)(?<!\. )(?<!: )(?<!; )(?<!! )(?<!\? )"
        r"\b([a-z][a-z'\-]{0,24})[ \t]+"
        r"((?:[1-9]|[12]\d|3\d|40))\.(\s+)"
        r"([a-z][A-Za-z'\-]*)"
    )
    _INFIX_STOP_PREV: ClassVar[frozenset] = frozenset(
        {"bis", "to", "from", "through", "vom", "von", "and", "or"}
    )
    _INFIX_STOP_NEXT: ClassVar[frozenset] = frozenset(
        {"bis", "to", "through"}
    )

    def _repair_infix_step_numbers(
        self,
        chunks: List[IngestionChunk],
    ) -> List[IngestionChunk]:
        """Repair embedded numbered-step markers mid-sentence
        (Firearms-shape canonical OCR artifact).

        OCR layout failures on multi-column numbered-instruction-list
        manuals can mash an instruction-step number into the trailing
        word of the preceding paragraph
        (e.g. ``"release the trigger 12. forsemgvaupwaros"``) instead
        of the canonical paragraph break before ``12.``. The repair
        inserts a newline between the preceding word and the step
        number so the chunk content reflects the source manual's
        paragraph structure. Content is preserved character-for-
        character except for the one inserted ``\\n`` per hit.

        Detection behaviorally mirrors the audit detector
        ``scripts/qa_conversion_audit.py::_INFIX_RE`` AFTER its
        newline / stop-word post-filters — the production regex
        collapses the audit's ``\\s+`` + ``"\\n" in between``
        post-filter into a single ``[ \\t]+`` capture group on the
        prev→num side, and reproduces the audit's left-context
        exclusion and short-word / stop-word filters explicitly. A
        chunk that no longer triggers the audit's ``infix_artifacts``
        counter cannot trigger this production repair either; the
        ``test_production_repair_matches_audit_detector_parity`` pin
        in ``tests/test_infix_step_number_repair.py`` re-applies the
        audit detector to repaired content and asserts the count
        drops to zero.
        """
        if not chunks:
            return chunks

        repaired_hits = 0
        repaired_chunks = 0

        def _replace_one(match: "re.Match[str]") -> str:
            nonlocal repaired_hits
            prev_word = match.group(1)
            num = match.group(2)
            sep_after_period = match.group(3)
            next_word = match.group(4)
            if len(prev_word) <= 1 or len(next_word) <= 1:
                return match.group(0)
            if prev_word.lower() in self._INFIX_STOP_PREV:
                return match.group(0)
            if next_word.lower() in self._INFIX_STOP_NEXT:
                return match.group(0)
            repaired_hits += 1
            # Preserve the original whitespace after the period so that
            # existing paragraph breaks (``\n\n``) are not collapsed
            # into a single space.
            return f"{prev_word}\n{num}.{sep_after_period}{next_word}"

        for ch in chunks:
            if ch.modality != Modality.TEXT or not ch.content:
                continue
            new_content = self._INFIX_STEP_PATTERN.sub(_replace_one, ch.content)
            if new_content != ch.content:
                ch.content = new_content
                repaired_chunks += 1

        if repaired_hits:
            logger.info(
                f"[INFIX-REPAIR] Repaired {repaired_hits} embedded step-number "
                f"boundaries across {repaired_chunks} chunks"
            )
        return chunks

    def _dedup_intra_chunk_repeats(self, chunks: List[IngestionChunk]) -> List[IngestionChunk]:
        """Remove repeated paragraphs within a single chunk.

        VLM transcription on cover/title pages can read the same text 3-4x
        (title on cover, spine, back cover bleed-through). The result is one
        chunk with the entire cover text repeated four times.

        Fix: split chunk into paragraphs, keep only unique ones in order.
        """
        fixed = 0
        for ch in chunks:
            if ch.modality != Modality.TEXT or not ch.content:
                continue
            # Split into lines and detect repeated blocks.
            # VLM can repeat title text as individual lines within one paragraph.
            lines = ch.content.split("\n")
            if len(lines) < 4:
                continue
            # Detect the repeat boundary: find longest prefix of content
            # that, when repeated, reconstructs most of the content.
            # This handles VLM reading cover text 2-4x.
            content_stripped = ch.content.strip()
            found_repeat = False
            # Loop: cut repeated prefixes until clean (4x→2x→1x)
            changed = True
            while changed and len(content_stripped) > 60:
                changed = False
                for frac in (2, 3, 4):
                    prefix_len = len(content_stripped) // frac
                    if prefix_len < 30:
                        continue
                    prefix = content_stripped[:prefix_len].rstrip()
                    rest = content_stripped[prefix_len:].lstrip()
                    if rest.startswith(prefix[:min(len(prefix), 40)]):
                        logger.info(
                            f"[INTRA-DEDUP] Prefix repeat on pg "
                            f"{ch.metadata.page_number}: {len(content_stripped)}→{len(prefix)} chars"
                        )
                        content_stripped = prefix
                        found_repeat = True
                        changed = True
                        break
            if found_repeat:
                ch.content = content_stripped
                # Apply same dedup to refined_content
                if ch.metadata and ch.metadata.refined_content:
                    rc = ch.metadata.refined_content.strip()
                    rc_changed = True
                    while rc_changed and len(rc) > 60:
                        rc_changed = False
                        for frac in (2, 3, 4):
                            rc_prefix_len = len(rc) // frac
                            if rc_prefix_len < 30:
                                continue
                            rc_prefix = rc[:rc_prefix_len].rstrip()
                            rc_rest = rc[rc_prefix_len:].lstrip()
                            if rc_rest.startswith(rc_prefix[:min(len(rc_prefix), 40)]):
                                rc = rc_prefix
                                rc_changed = True
                                break
                    ch.metadata.refined_content = rc
                fixed += 1

            if not found_repeat:
                # Fallback: line-level dedup
                seen_keys: set[str] = set()
                kept_lines: list[str] = []
                for line in lines:
                    key = line.strip().lower()
                    if not key:
                        kept_lines.append(line)
                        continue
                    if key in seen_keys and len(key) > 5:
                        continue
                    seen_keys.add(key)
                    kept_lines.append(line)
                deduped = "\n".join(kept_lines).strip()
                if deduped != ch.content.strip():
                    ch.content = deduped
                if ch.metadata and ch.metadata.refined_content:
                    rc_lines = ch.metadata.refined_content.split("\n")
                    rc_seen_keys: set[str] = set()
                    rc_deduped: list[str] = []
                    for line in rc_lines:
                        key = line.strip().lower()
                        if not key:
                            rc_deduped.append(line)
                            continue
                        if key in rc_seen_keys and len(key) > 5:
                            continue
                        rc_seen_keys.add(key)
                        rc_deduped.append(line)
                    ch.metadata.refined_content = "\n".join(rc_deduped).strip()
                fixed += 1
        if fixed:
            logger.info(f"[INTRA-DEDUP] Removed repeated paragraphs in {fixed} chunks")
        return chunks

    def _merge_mid_sentence_chunks(self, chunks: List[IngestionChunk]) -> List[IngestionChunk]:
        """Merge consecutive text chunks split mid-sentence.

        Layout-aware OCR creates one chunk per detected region. When a sentence
        spans two regions, each chunk gets half — producing fragments like
        "...court cases in" followed by "which the gun was". Merging these
        produces coherent text for embedding and retrieval.
        """
        import re as _re

        merged: list[IngestionChunk] = []
        merge_count = 0

        # Build index: for each text chunk, find the NEXT text chunk (skipping images)
        text_indices = [i for i, ch in enumerate(chunks) if ch.modality == Modality.TEXT and ch.content]
        merge_targets: set[int] = set()  # indices of chunks consumed by merge

        for ti in range(len(text_indices) - 1):
            cur_idx = text_indices[ti]
            nxt_idx = text_indices[ti + 1]

            if cur_idx in merge_targets or nxt_idx in merge_targets:
                continue

            cur = chunks[cur_idx]
            nxt = chunks[nxt_idx]

            # v2.9: skip merge when either side is a page-coverage
            # split copy (hybrid_chunker_pagesplit). Those are
            # intentional duplicates of cross-page chunks; merging
            # them would reattribute the latter page's content to the
            # former and break page coverage.
            cur_method = cur.metadata.extraction_method if cur.metadata else ""
            nxt_method = nxt.metadata.extraction_method if nxt.metadata else ""
            if "pagesplit" in (cur_method or "") or "pagesplit" in (nxt_method or ""):
                continue

            # v2.9: skip cross-page merges. This filter exists to
            # rejoin sentences split across regions on the SAME page
            # by layout-aware OCR. Cross-page mid-sentence breaks
            # ("...rather a" → "lot; Harry didn't feel brave...") are
            # legitimate page boundaries — merging them assigns the
            # latter page's content to the former and erases its
            # page attribution (HARRY p131, p141 lost this way).
            cur_page = (
                cur.metadata.page_number if cur.metadata else None
            )
            nxt_page = (
                nxt.metadata.page_number if nxt.metadata else None
            )
            if cur_page is not None and nxt_page is not None and int(cur_page) != int(nxt_page):
                continue

            cur_text = cur.content.rstrip()
            nxt_text = nxt.content.lstrip()
            ends_mid = not _re.search(r"[.!?:;\"')\]}\d]\s*$", cur_text)
            starts_lower = bool(nxt_text) and nxt_text[0].islower()

            if ends_mid and starts_lower:
                cur.content = cur_text + " " + nxt_text
                if cur.metadata.refined_content and nxt.metadata.refined_content:
                    cur.metadata.refined_content = (
                        cur.metadata.refined_content.rstrip() + " " +
                        nxt.metadata.refined_content.lstrip()
                    )
                merge_targets.add(nxt_idx)
                merge_count += 1
                logger.info(
                    f"[MID-SENTENCE-MERGE] Merged pg {cur.metadata.page_number}→"
                    f"{nxt.metadata.page_number}: ...{cur_text[-20:]} + {nxt_text[:20]}..."
                )

        # Build output: keep all non-merged chunks in order
        merged = [ch for i, ch in enumerate(chunks) if i not in merge_targets]

        if merge_count:
            logger.info(f"[MID-SENTENCE-MERGE] Merged {merge_count} mid-sentence chunk boundaries")

        return merged

    def _remove_near_duplicate_chunks(self, chunks: List[IngestionChunk]) -> List[IngestionChunk]:
        """Remove text chunks that are near-duplicates of earlier chunks.

        Technical manuals repeat short instructions across sections (e.g.,
        "Remove the trigger housing downward" appears for every gun model).
        These pollute RAG retrieval with identical results.

        Uses word-set overlap: if >85% of words in a shorter chunk appear
        in an earlier chunk, drop the shorter one.

        v2.9: page-scoped — same-content chunks on DIFFERENT pages are
        kept. The hybrid_chunker_pagesplit emit deliberately copies a
        cross-page DocChunk to each source page (HARRY p28+p29 chapter-
        intro narrative). The near-dup filter must not collapse those
        copies; otherwise the page-coverage invariant breaks. Real
        manual-style cross-section repetition still gets caught when
        it happens on the same page (rare) or via the more aggressive
        post-corruption cross-chunk dedup at finalization.
        """
        import re as _re

        seen_per_page: Dict[int, list[set[str]]] = {}
        kept: list[IngestionChunk] = []
        dropped = 0

        for ch in chunks:
            if ch.modality != Modality.TEXT or not ch.content:
                kept.append(ch)
                continue

            words = set(_re.findall(r"[a-zA-Z]{3,}", ch.content.lower()))
            page = (
                int(ch.metadata.page_number)
                if ch.metadata and ch.metadata.page_number
                else 0
            )
            page_seen = seen_per_page.setdefault(page, [])
            if len(words) < 5:
                # Very short chunks — keep (headings, labels)
                kept.append(ch)
                page_seen.append(words)
                continue

            is_dup = False
            for seen in page_seen:
                if not seen:
                    continue
                overlap = len(words & seen) / len(words)
                if overlap > 0.95:
                    is_dup = True
                    break

            if is_dup:
                dropped += 1
                logger.debug(
                    f"[NEAR-DEDUP] Dropped near-duplicate chunk on p{page} "
                    f"({len(ch.content)} chars, {overlap:.0%} overlap)"
                )
            else:
                kept.append(ch)
                page_seen.append(words)

        if dropped:
            logger.info(f"[NEAR-DEDUP] Removed {dropped} near-duplicate text chunks (page-scoped)")

        return kept

    def _deduplicate_chunk_overlap(self, chunks: List[IngestionChunk]) -> List[IngestionChunk]:
        """Remove duplicated text at chunk boundaries.

        DSO adds overlap so consecutive chunks share ~1 sentence. This helps
        context continuity during extraction but pollutes vector search with
        near-identical results. The context is already preserved in
        prev_text_snippet/next_text_snippet, so the content duplication is
        unnecessary for RAG.

        v2.10 (`DOCLING_DUPLICATE_DOC_CHUNK_OVERLAP_TRIM` resolution):
        the comparison is now **page-scoped**. Docling 2.86 occasionally
        emits two byte-identical DocChunks for the same code block on
        adjacent pages — e.g. Python_Cookbook DocChunk #335 prov=[396]
        and DocChunk #336 prov=[397] with identical `dc.text`. The
        cross-page overlap-trim was stripping chunk[N+1]'s entire
        content (because chunk[N+1].content equals chunk[N].content's
        tail in full), dropping that page from coverage. DSO's intended
        overlap target was always *same-page* sentence trimming;
        page-scoping the prev/cur pair preserves the intent while
        eliminating cross-page page-loss.
        """
        trimmed = 0
        last_text_idx = -1
        for i in range(len(chunks)):
            cur = chunks[i]
            if cur.modality != Modality.TEXT or not cur.content:
                continue
            if last_text_idx < 0:
                last_text_idx = i
                continue
            prev = chunks[last_text_idx]
            if not prev.content:
                continue

            cur_page = (
                int(cur.metadata.page_number)
                if cur.metadata and cur.metadata.page_number
                else None
            )
            prev_page = (
                int(prev.metadata.page_number)
                if prev.metadata and prev.metadata.page_number
                else None
            )
            if cur_page is not None and prev_page is not None and cur_page != prev_page:
                # Cross-page consecutive chunks: skip overlap trim.
                # Their content may coincidentally match (Docling
                # duplicate-DocChunks shape) but trimming would drop
                # page coverage.
                last_text_idx = i
                continue

            # Find longest exact overlap: tail of prev == head of cur.
            # Search from longest to shortest (greedy) for efficiency.
            max_check = min(len(prev.content), len(cur.content), 300)
            overlap_len = 0
            for length in range(max_check, 9, -1):
                if prev.content[-length:] == cur.content[:length]:
                    overlap_len = length
                    break

            if overlap_len > 0:
                cur.content = cur.content[overlap_len:].lstrip()
                # Also trim refined_content
                if cur.metadata and cur.metadata.refined_content and prev.metadata and prev.metadata.refined_content:
                    prev_rc = prev.metadata.refined_content
                    rc = cur.metadata.refined_content
                    rc_overlap = 0
                    for length in range(min(len(prev_rc), len(rc), 300), 9, -1):
                        if prev_rc[-length:] == rc[:length]:
                            rc_overlap = length
                            break
                    if rc_overlap > 0:
                        cur.metadata.refined_content = rc[rc_overlap:].lstrip()
                trimmed += 1

            last_text_idx = i

        if trimmed:
            logger.info(f"[DEDUP-OVERLAP] Trimmed content overlap from {trimmed} chunk boundaries")

        return chunks

    def _repair_cross_chunk_hyphenation(self, chunks: List[IngestionChunk]) -> List[IngestionChunk]:
        """Rejoin words split by hyphens across chunk boundaries.

        When a chunk ends with "man-" and the next starts with "age...", the
        word "manage" is broken across two chunks. Neither chunk embeds well
        for a query about "management".

        Fix: append the continuation fragment to the current chunk's last word,
        and remove it from the next chunk's start.
        """
        import re as _re

        repaired = 0
        for i in range(len(chunks) - 1):
            cur = chunks[i]
            nxt = chunks[i + 1]
            if cur.modality != Modality.TEXT or nxt.modality != Modality.TEXT:
                continue
            if not cur.content or not nxt.content:
                continue

            # Does current chunk end with a hyphenated word break?
            match = _re.search(r"([a-zA-Z]{2,})-\s*$", cur.content)
            if not match:
                continue

            # Does next chunk start with a lowercase continuation?
            next_match = _re.match(r"^([a-z]{2,})\b", nxt.content.lstrip())
            if not next_match:
                continue

            word_start = match.group(1)
            word_end = next_match.group(1)
            full_word = word_start + word_end

            # Repair: replace trailing "word-" with "word" + continuation
            cur.content = _re.sub(r"([a-zA-Z]{2,})-\s*$", full_word, cur.content)
            # Remove the continuation fragment from next chunk start
            nxt.content = _re.sub(r"^" + _re.escape(word_end) + r"\b\s*", "", nxt.content.lstrip())

            # Also fix refined_content if present
            if cur.metadata and cur.metadata.refined_content:
                cur.metadata.refined_content = _re.sub(
                    r"([a-zA-Z]{2,})-\s*$", full_word, cur.metadata.refined_content
                )
            if nxt.metadata and nxt.metadata.refined_content:
                nxt.metadata.refined_content = _re.sub(
                    r"^" + _re.escape(word_end) + r"\b\s*", "", nxt.metadata.refined_content.lstrip()
                )

            repaired += 1
            logger.debug(f"[DEHYPHEN] Rejoined '{word_start}-' + '{word_end}' → '{full_word}'")

        if repaired:
            logger.info(f"[DEHYPHEN] Repaired {repaired} cross-chunk hyphenations")

        return chunks

    def _apply_code_hygiene(self, chunks: List[IngestionChunk]) -> List[IngestionChunk]:
        """
        Detect, reclassify, and reflow code chunks for ALL profiles.

        Extracted from _apply_technical_manual_hygiene so that academic papers,
        magazines, and other profiles also get proper code handling. Without this,
        Docling-extracted code blocks in non-technical-manual documents remain as
        flat single-line paragraph chunks.

        Operations:
        1. Reclassify missed code (paragraph chunks containing code)
        2. Demote false positives (prose misidentified as code)
        3. Reflow flat code (restore newlines in concatenated code)
        """
        def is_code_chunk(ch: IngestionChunk) -> bool:
            try:
                return (
                    ch.metadata.chunk_type == ChunkType.CODE
                    or ch.metadata.content_classification == "code"
                )
            except Exception:
                return False

        reclassified = 0
        reflowed = 0
        for ch in chunks:
            if ch.modality != Modality.TEXT:
                continue
            txt = ch.content or ""
            if not txt.strip():
                continue

            # Reclassify missed code blocks
            try:
                if (not is_code_chunk(ch)) and self._looks_like_code_text(txt):
                    ch.metadata.chunk_type = ChunkType.CODE
                    ch.metadata.content_classification = "code"
                    reclassified += 1
            except Exception:
                pass

            # Demote false positives
            self._maybe_demote_false_code_chunk(ch)

            # Reflow flat code and track repair metadata
            if is_code_chunk(ch):
                reflowed_txt = self._preserve_or_reflow_code_text(txt)
                if reflowed_txt != txt:
                    ch.content = reflowed_txt
                    reflowed += 1
                    try:
                        ch.metadata.code_repair_applied = True
                    except Exception:
                        pass
                # Track indentation fidelity and parse status on code chunks
                try:
                    code_lines = [ln for ln in (ch.content or "").splitlines() if ln.strip()]
                    has_indent = any(ln.startswith(("    ", "\t")) for ln in code_lines)
                    # REPL code uses >>> markers instead of indentation — count as structured
                    has_repl = any(ln.lstrip().startswith(">>> ") for ln in code_lines)
                    ch.metadata.is_code = True
                    ch.metadata.indentation_fidelity = 1.0 if (has_indent or has_repl) else 0.0
                    # Try parsing as Python — structural validity check
                    import ast as _ast
                    try:
                        _ast.parse(ch.content or "")
                        ch.metadata.code_parse_ok = True
                    except (SyntaxError, ValueError):
                        ch.metadata.code_parse_ok = False
                except Exception:
                    pass

        # Code indentation recovery — two strategies:
        # 1. Pure code chunks: PyMuPDF x-coordinate recovery from bbox
        # 2. Mixed prose+code chunks: reflow fenced code blocks (```...```)
        recovered = 0
        fence_recovered = 0
        if self._current_pdf_path and self._current_pdf_path.exists():
            # PLAN_F1 WP-2 (Mechanism B): on a born-digital text_native_code page
            # the PDF text layer is the AUTHORITATIVE source for code indentation,
            # so EVERY code chunk on such a page is re-served from the text-layer
            # clip — both lanes, not only flat chunks. Precompute the page signal
            # once (single PDF open) over the pages that actually carry code chunks.
            code_pages = {
                ch.metadata.page_number
                for ch in chunks
                if (
                    ch.modality in (Modality.TEXT, Modality.CODE)
                    and is_code_chunk(ch)
                    and ch.metadata
                    and ch.metadata.page_number
                )
            }
            text_native_pages: dict = {}
            if code_pages:
                try:
                    import fitz as _fitz

                    _doc = _fitz.open(str(self._current_pdf_path))
                    for _pno in code_pages:
                        if 1 <= _pno <= len(_doc):
                            _native, _ = _score_text_native_code(_doc[_pno - 1].get_text())
                            text_native_pages[_pno] = _native
                    _doc.close()
                except Exception as _e:
                    logger.debug(f"[CODE-INDENT] text_native_code precompute failed: {_e}")

            for ch in chunks:
                # Admit promoted Modality.CODE chunks, not only code smuggled as
                # TEXT. V3 promotes code to Modality.CODE, which the old TEXT-only
                # gate skipped, leaving this recovery dead on the entire promoted
                # population (PLAN_F1 Phase 0(b) modality seam).
                if ch.modality not in (Modality.TEXT, Modality.CODE) or not is_code_chunk(ch):
                    continue
                try:
                    page_num = ch.metadata.page_number if ch.metadata else None
                    is_text_native = bool(text_native_pages.get(page_num, False))

                    fidelity = getattr(ch.metadata, "indentation_fidelity", None)
                    if fidelity is None:
                        # The hygiene loop above stamps indentation_fidelity only
                        # on TEXT chunks; derive the same flat/indented signal for
                        # CODE chunks so already-indented code is not re-extracted.
                        _code_lines = [ln for ln in (ch.content or "").splitlines() if ln.strip()]
                        _has_indent = any(ln.startswith(("    ", "\t")) for ln in _code_lines)
                        _has_repl = any(ln.lstrip().startswith(">>> ") for ln in _code_lines)
                        fidelity = 1.0 if (_has_indent or _has_repl) else 0.0
                    # WP-2 rule (documented, supersedes the c95950b skip on
                    # text-native pages): the text layer wins on a text_native_code
                    # page, so re-serve even already-indented chunks (the engine's
                    # indentation may be a lossy raster round-trip). OFF text-native
                    # pages, retain c95950b: only attempt flat chunks.
                    if not is_text_native and fidelity is not None and fidelity > 0:
                        continue  # Already has indentation; not a text-native page

                    # Strategy 1: PyMuPDF recovery for pure code chunks
                    if self._recover_code_indentation_from_pdf(ch):
                        recovered += 1
                        continue

                    # Strategy 2: Fence-aware reflow for mixed prose+code
                    if self._recover_fenced_code_blocks(ch):
                        fence_recovered += 1
                except Exception as e:
                    logger.debug(f"[CODE-INDENT] Recovery failed for {ch.chunk_id}: {e}")

        # PLAN_F1 J1 (b)+(c): repair the residual non-indentation defects in code
        # chunks - smart quotes, hard-wrapped open strings, and (repair-only) open
        # brackets. Idempotent / no-op on already-clean code.
        repaired_code = 0
        for ch in chunks:
            if ch.modality in (Modality.TEXT, Modality.CODE) and is_code_chunk(ch):
                new_content = _repair_code_content(ch.content or "")
                if new_content != ch.content:
                    ch.content = new_content
                    repaired_code += 1

        # PLAN_F1 J1 (a): heal code blocks cut mid-docstring across a chunk
        # boundary - merge a code chunk that ends inside an unterminated
        # triple-quoted string into the next adjacent (same/next page) code chunk.
        merged_docstrings = 0
        if any(_leaves_docstring_open(c.content or "") for c in chunks
               if c.modality in (Modality.TEXT, Modality.CODE) and is_code_chunk(c)):
            healed: List[IngestionChunk] = []
            i = 0
            while i < len(chunks):
                ch = chunks[i]
                if (
                    ch.modality in (Modality.TEXT, Modality.CODE)
                    and is_code_chunk(ch)
                    and _leaves_docstring_open(ch.content or "")
                    and i + 1 < len(chunks)
                ):
                    nxt = chunks[i + 1]
                    cur_pg = ch.metadata.page_number if ch.metadata else None
                    nxt_pg = nxt.metadata.page_number if nxt.metadata else None
                    adjacent = (
                        cur_pg is not None and nxt_pg is not None and 0 <= (nxt_pg - cur_pg) <= 1
                    )
                    if (
                        nxt.modality in (Modality.TEXT, Modality.CODE)
                        and is_code_chunk(nxt)
                        and adjacent
                    ):
                        ch.content = (ch.content or "") + "\n" + _repair_code_content(nxt.content or "")
                        merged_docstrings += 1
                        i += 2  # absorbed nxt
                        healed.append(ch)
                        continue
                healed.append(ch)
                i += 1
            chunks = healed

        if reclassified or reflowed or recovered or fence_recovered or repaired_code or merged_docstrings:
            logger.info(
                f"[CODE-HYGIENE] Reclassified {reclassified} chunks as code, "
                f"reflowed {reflowed} flat code, "
                f"recovered indentation for {recovered} chunks (PyMuPDF), "
                f"{fence_recovered} chunks (fence reflow), "
                f"repaired {repaired_code} code chunks (quotes/wraps), "
                f"merged {merged_docstrings} split-docstring code chunks"
            )

        return chunks

    def _recover_code_indentation_from_pdf(self, chunk: IngestionChunk) -> bool:
        """
        Re-extract code text from the PDF using PyMuPDF character x-coordinates
        to recover original indentation that Docling's text layer lost.

        Only applies to code chunks that are currently flat (indentation_fidelity=0).
        Returns True if indentation was recovered.
        """
        import fitz

        content = chunk.content or ""
        if not content.strip():
            return False

        # Need page number and bbox to locate the code in the PDF
        page_num = getattr(chunk.metadata, "page_number", None)
        spatial = getattr(chunk.metadata, "spatial", None)
        if not page_num or not spatial:
            return False
        bbox = getattr(spatial, "bbox", None)
        if not bbox or len(bbox) < 4:
            return False

        try:
            doc = fitz.open(str(self._current_pdf_path))
            if page_num > len(doc):
                doc.close()
                return False
            page = doc[page_num - 1]
            pw, ph = page.rect.width, page.rect.height

            # Convert normalized [0,1000] bbox to PDF points
            x0 = bbox[0] / 1000.0 * pw
            y0 = bbox[1] / 1000.0 * ph
            x1 = bbox[2] / 1000.0 * pw
            y1 = bbox[3] / 1000.0 * ph
            clip = fitz.Rect(x0, y0, x1, y1)

            # Extract text with character position data
            blocks = page.get_text("dict", clip=clip, flags=fitz.TEXT_PRESERVE_WHITESPACE)
            doc.close()

            # Build lines from character spans, tracking x-positions
            raw_lines: list = []  # [(y_center, x_start, text), ...]
            for block in blocks.get("blocks", []):
                if block.get("type") != 0:  # text block only
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    y_center = (line["bbox"][1] + line["bbox"][3]) / 2
                    x_start = spans[0]["bbox"][0]
                    text = "".join(s.get("text", "") for s in spans)
                    if text.strip():
                        # Strip leading whitespace too (PLAN_F1 WP-2 double-indent
                        # guard): x_start already encodes the indent, so leading
                        # space glyphs in the span text would be added TWICE.
                        raw_lines.append((y_center, x_start, text.strip()))

            if len(raw_lines) < 2:
                return False

            # Sort by y position (top to bottom)
            raw_lines.sort(key=lambda t: t[0])

            # Find the leftmost x-position (base indentation)
            min_x = min(x for _, x, _ in raw_lines)

            # Estimate character width from the most common font
            # Use median span width / span char count as approximation
            char_width = 7.0  # default monospace at ~10pt
            all_widths = []
            for block in blocks.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        t = span.get("text", "")
                        w = span["bbox"][2] - span["bbox"][0]
                        if len(t) > 2 and w > 0:
                            all_widths.append(w / len(t))
            if all_widths:
                all_widths.sort()
                char_width = all_widths[len(all_widths) // 2]  # median

            # Reconstruct lines with indentation
            indented_lines = []
            for _, x_start, text in raw_lines:
                indent_chars = max(0, round((x_start - min_x) / char_width))
                indented_lines.append(" " * indent_chars + text)

            # Check if we actually recovered indentation
            result = "\n".join(indented_lines)
            has_indent = any(ln.startswith(("    ", "   ")) for ln in indented_lines if ln.strip())
            if not has_indent:
                return False

            # Only replace content if we found indentation and the result is reasonable
            chunk.content = result
            chunk.metadata.indentation_fidelity = 1.0
            chunk.metadata.code_repair_applied = True

            # Re-check parse status
            import ast as _ast
            try:
                _ast.parse(result)
                chunk.metadata.code_parse_ok = True
            except (SyntaxError, ValueError):
                chunk.metadata.code_parse_ok = False

            return True

        except Exception:
            return False

    def _recover_fenced_code_blocks(self, chunk: IngestionChunk) -> bool:
        """
        Recover indentation inside markdown code fences (```...```) within mixed
        prose+code chunks.

        HybridChunker often merges code fences with surrounding prose into a single
        chunk. The code inside the fences is flat (all on one line) because the PDF
        text layer concatenated the REPL lines. This method:

        1. Finds ``` fence boundaries in the content
        2. Reflowes each fenced block using REPL markers (>>>, ...)
        3. Uses PyMuPDF x-coordinate extraction if bbox is available
        4. Reassembles the chunk with reflowed code

        Returns True if any code block was improved.
        """
        import re

        content = chunk.content or ""
        if "```" not in content:
            return False

        # Split content into segments: prose and fenced code blocks
        parts = re.split(r"(```[^\n]*\n?)", content)
        if len(parts) < 3:
            return False

        improved = False
        result_parts = []
        in_fence = False

        for part in parts:
            if part.startswith("```"):
                in_fence = not in_fence
                result_parts.append(part)
                continue

            if in_fence and part.strip():
                # This is code inside a fence — try to reflow
                original = part
                reflowed = self._reflow_fenced_code(part)
                if reflowed != original:
                    result_parts.append(reflowed)
                    improved = True
                else:
                    result_parts.append(part)
            else:
                result_parts.append(part)

        if improved:
            chunk.content = "".join(result_parts)
            chunk.metadata.code_repair_applied = True
            # Re-check indentation fidelity
            code_lines = [ln for ln in chunk.content.splitlines() if ln.strip()]
            has_indent = any(ln.startswith(("    ", "\t")) for ln in code_lines)
            chunk.metadata.indentation_fidelity = 1.0 if has_indent else 0.0
            # Re-check parse status
            import ast as _ast
            try:
                _ast.parse(chunk.content)
                chunk.metadata.code_parse_ok = True
            except (SyntaxError, ValueError):
                chunk.metadata.code_parse_ok = False
            return True
        return False

    def _reflow_fenced_code(self, code_text: str) -> str:
        """
        Reflow a flat code block that appears inside markdown fences.

        Handles two common patterns:
        1. REPL: ">>> stmt1 >>> stmt2 ... continuation" → split on >>> markers
        2. Script: "def foo(): return bar" → split on Python keywords

        Also separates command output that's concatenated on the same line as
        the REPL prompt (e.g. ">>> v1 + v2 Vector(4, 5)" → separate lines).
        """
        import re

        t = code_text.strip()
        if not t:
            return code_text

        # If already multiline with indentation, preserve exactly
        if "\n" in t:
            lines = t.splitlines()
            if any(ln.startswith(("    ", "\t")) for ln in lines):
                return code_text

        # REPL reflow: split on >>> and ... markers
        if ">>>" in t:
            # First: split inline >>> markers onto separate lines
            t = re.sub(r"(?<!\n)\s*>>>\s*", "\n>>> ", t)
            t = re.sub(r"(?<!\n)\s*\.\.\.\s*", "\n... ", t)

            lines = [ln for ln in t.splitlines() if ln.strip()]
            if not lines:
                return code_text

            # Second: separate command output from prompt lines.
            # ">>> expr result_text" → ">>> expr\nresult_text"
            # Heuristic: if a >>> line contains a complete expression followed
            # by what looks like output (starts with [, (, {, digit, quote, or
            # capitalized class name), split there.
            expanded = []
            for ln in lines:
                stripped = ln.lstrip()
                if stripped.startswith(">>> ") or stripped.startswith("... "):
                    marker = stripped[:4]
                    body = stripped[4:]
                    # Try to find where the command ends and output begins
                    # Look for balanced expression followed by output
                    split_pos = self._find_repl_output_boundary(body)
                    if split_pos and split_pos < len(body) - 1:
                        cmd = body[:split_pos].rstrip()
                        output = body[split_pos:].lstrip()
                        expanded.append(f"{marker}{cmd}")
                        expanded.append(output)
                    else:
                        expanded.append(stripped)
                else:
                    expanded.append(stripped)

            return "\n".join(expanded) + "\n"

        # Script reflow: use _reflow_flat_code for non-REPL code
        if "\n" not in t:
            return self._reflow_flat_code(t) + "\n"

        return code_text

    @staticmethod
    def _find_repl_output_boundary(body: str) -> Optional[int]:
        """
        Find where a REPL command ends and its output begins on the same line.

        E.g. "v1 + v2 Vector(4, 5)" → split at position of "Vector"
             "deck[:3] [Card(..." → split at position of "["
             "abs(v) 5.0" → split at position of "5.0"

        Returns the split position, or None if no boundary found.
        """
        import re

        if not body or len(body) < 5:
            return None

        # Pattern: after a closing bracket/paren or end-of-identifier,
        # followed by space then output (starts with [, (, upper, digit, ', ")
        m = re.search(
            r"([\]\)'\"])\s+([\[(\d'\"A-Z])",
            body,
        )
        if m:
            return m.start(2)

        # Pattern: simple expression result — "identifier number"
        m = re.search(r"\)\s+(\d)", body)
        if m:
            return m.start(1)

        return None

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

        # Step A.5: Rejoin cross-chunk word fragments from two-column column-break hyphenation.
        chunks = self._rejoin_leading_word_fragments(chunks)

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

            # Three-level fallback chain: \n\n → \n → sentence mark.
            # Each level checks its own minimum-size threshold (max_chars // 5).
            # When the best candidate at a level is too early (< threshold), that
            # level sets split_idx = None and the next level is tried.  This avoids
            # the previous bug where a \n\n at position ~120 blocked the \n at
            # position ~1100 and the sentence mark at ~1450 from ever being reached,
            # causing a hard mid-word cut at exactly max_chars.
            # Candidates beyond max_chars are excluded: they would be clamped by the
            # hard-cap below, producing a mid-word cut identical to a raw hard split.

            # Level 1: nearest paragraph break (\n\n)
            candidates: List[Tuple[int, int]] = []
            if p_before > 0:
                candidates.append((abs(target - p_before), p_before))
            if 0 < p_after <= max_chars:
                candidates.append((abs(p_after - target), p_after))
            if candidates:
                candidates.sort(key=lambda x: x[0])
                best_para = candidates[0][1]
                if best_para >= (max_chars // 5):
                    split_idx = best_para
                # else: too early — fall through to single-newline level

            # Level 2: nearest single newline (only reached when \n\n level failed)
            if split_idx is None:
                delimiter_len = 1
                n_before = remaining.rfind("\n", 0, target + 1)
                n_after = remaining.find("\n", target, search_end)
                nl_candidates: List[Tuple[int, int]] = []
                if n_before > 0:
                    nl_candidates.append((abs(target - n_before), n_before))
                if 0 < n_after <= max_chars:
                    nl_candidates.append((abs(n_after - target), n_after))
                if nl_candidates:
                    nl_candidates.sort(key=lambda x: x[0])
                    best_nl = nl_candidates[0][1]
                    if best_nl >= (max_chars // 5):
                        split_idx = best_nl
                    # else: too early — fall through to sentence-mark level

            # Level 3: sentence-aware fallback (only reached when both \n levels failed)
            if split_idx is None:
                delimiter_len = 0
                sentence_marks = []
                for marker in (". ", ".\n", "? ", "?\n", "! ", "!\n"):
                    pos = remaining.rfind(marker, 0, target + 1)
                    if pos > 0:
                        sentence_marks.append(pos + 1)  # include punctuation
                for marker in (".", "?", "!"):
                    pos = remaining.rfind(marker, 0, target + 1)
                    if pos > 0:
                        sentence_marks.append(pos + 1)
                best_sent = max(sentence_marks) if sentence_marks else target
                if best_sent >= (max_chars // 5):
                    split_idx = best_sent
                else:
                    split_idx = target  # genuine hard split: no usable break found

            if split_idx is None or split_idx <= 0:
                split_idx = target
                delimiter_len = 0

            # Guard: avoid trivially small splits that cause near-zero progress loops.
            # Threshold is max_chars // 5 (300 for max_chars=1500) rather than the old
            # max_chars // 2 (750) which was discarding valid sentence/paragraph breaks
            # found at positions like 720, producing unnecessary hard mid-sentence cuts.
            if split_idx < (max_chars // 5):
                split_idx = target
                delimiter_len = 0

            # Hard cap: OversizeBreaker must never emit a head segment above max_chars.
            # A nearest paragraph/newline break after the target is useful for semantics,
            # but we cannot violate the configured chunk ceiling.
            if split_idx > max_chars:
                split_idx = max_chars
                delimiter_len = 0

            # Word-boundary snap: if the split point is inside a word (both the character
            # just before split_idx and the one at split_idx are alphabetic), back up to
            # the last space in the valid range to avoid mid-word cuts like "prog|rammed".
            # This fires whether the hard cap landed at exactly max_chars OR a level-3
            # sentence-mark fallback set split_idx to target mid-word.
            if (
                0 < split_idx <= len(remaining)
                and split_idx - 1 < len(remaining)
                and remaining[split_idx - 1 : split_idx].isalpha()
                and split_idx < len(remaining)
                and remaining[split_idx : split_idx + 1].isalpha()
            ):
                last_space = remaining.rfind(" ", max_chars // 5, split_idx)
                if last_space > max_chars // 5:
                    split_idx = last_space
                    delimiter_len = 1  # consume the space

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

    def _enhance_magazine_images(self, chunks: List[IngestionChunk]) -> int:
        """
        Re-render magazine image regions at high DPI from the original PDF.

        Magazine PDFs store composite page layouts (text+photos baked together).
        Docling's layout model detects image regions but the extracted images
        often include text overlays and layout artifacts. Re-rendering from
        PyMuPDF at the detected bbox gives cleaner crops.

        Only applies to image chunks with asset_ref (saved PNG files).
        Skips chunks where the existing image is already high quality.

        Returns the number of enhanced images.
        """
        import fitz

        if not self._current_pdf_path or not self._current_pdf_path.exists():
            return 0

        enhanced = 0
        render_dpi = 200  # Good quality for magazine photos without excessive file sizes

        try:
            doc = fitz.open(str(self._current_pdf_path))
        except Exception as e:
            logger.debug(f"[MAGAZINE-IMAGE] Could not open PDF: {e}")
            return 0

        try:
            for chunk in chunks:
                if chunk.modality != Modality.IMAGE:
                    continue
                asset_ref = getattr(chunk, "asset_ref", None)
                if not asset_ref or not asset_ref.file_path:
                    continue
                spatial = getattr(chunk.metadata, "spatial", None)
                if not spatial or not spatial.bbox:
                    continue
                page_num = getattr(chunk.metadata, "page_number", None)
                if not page_num or page_num > len(doc):
                    continue

                bbox = spatial.bbox
                # Skip full-page images (layout artifacts, not photos)
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area > 800000:  # > 80% of [0,1000]² area
                    continue
                # Skip tiny images (icons, decorative elements)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < 100 or h < 100:
                    continue

                try:
                    page = doc[page_num - 1]
                    pw, ph = page.rect.width, page.rect.height

                    # Convert normalized [0,1000] bbox to PDF points
                    x0 = bbox[0] / 1000.0 * pw
                    y0 = bbox[1] / 1000.0 * ph
                    x1 = bbox[2] / 1000.0 * pw
                    y1 = bbox[3] / 1000.0 * ph
                    clip = fitz.Rect(x0, y0, x1, y1)

                    # Render at high DPI
                    zoom = render_dpi / 72.0
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)

                    if pix.width < 50 or pix.height < 50:
                        continue

                    # Save rendered crop, replacing the Docling-extracted image
                    asset_path = self.output_dir / asset_ref.file_path
                    pix.save(str(asset_path))

                    # Update asset metadata
                    asset_ref.width_px = pix.width
                    asset_ref.height_px = pix.height
                    asset_ref.file_size_bytes = asset_path.stat().st_size
                    chunk.metadata.extraction_method = "rendered_region_crop"
                    enhanced += 1

                except Exception as e:
                    logger.debug(
                        f"[MAGAZINE-IMAGE] Failed to enhance pg{page_num} "
                        f"bbox={bbox}: {e}"
                    )
        finally:
            doc.close()

        return enhanced

    def _filter_no_visual_images(
        self,
        chunks: List[IngestionChunk],
    ) -> List[IngestionChunk]:
        """
        Remove image chunks whose VLM description explicitly states there is no
        distinct visual content (i.e., the shadow extractor captured a text-only
        region and the VLM correctly identified it as non-visual).

        The sentinel phrase is "no distinct non-text visuals" (case-insensitive).
        Images with richer "Dense typographic" descriptions that describe actual
        content (chat interfaces, comparison panels, etc.) are kept.

        These zero-value image chunks add retrieval noise — a query about any topic
        would weakly match the phrase "no distinct non-text visuals" — without
        providing information that isn't already in the co-located text chunks.
        """
        _SENTINEL = "no distinct non-text visuals"
        before = len(chunks)
        out: List[IngestionChunk] = []
        removed = 0
        for ch in chunks:
            if ch.modality == Modality.IMAGE:
                vd = (ch.metadata.visual_description or "").lower()
                if _SENTINEL in vd:
                    logger.info(
                        f"[VISUAL-FILTER] Dropping zero-value image chunk "
                        f"p{ch.metadata.page_number} '{ch.metadata.visual_description[:60]}'"
                    )
                    removed += 1
                    continue
            out.append(ch)
        if removed:
            logger.info(
                f"[VISUAL-FILTER] Removed {removed} no-visual-content image chunks "
                f"({before} → {len(out)})"
            )
        return out

    def _filter_repetition_garbage(
        self,
        chunks: List[IngestionChunk],
    ) -> List[IngestionChunk]:
        """
        Remove TEXT chunks that are token-repetition garbage from broken UI/callout
        extraction — e.g. "down: down: down: down:" or "today: today: today:".

        A short token (2-15 chars) repeated 5+ times with only separator characters
        (': ', ', ', '; ', ' ') between repetitions is classified as garbage.
        CODE and TABLE chunks are never touched.
        """
        import re

        _SEP = r"(?::\s+|,\s+|;\s+|\s+)"
        _TOKEN = r"[A-Za-z0-9'\-]{2,15}"
        _REPETITION_RE = re.compile(
            r"^(?P<tok>" + _TOKEN + r")" + _SEP +
            r"(?P=tok)" + _SEP + r"(?P=tok)" + _SEP +
            r"(?P=tok)" + _SEP + r"(?P=tok)",
            re.IGNORECASE,
        )

        before = len(chunks)
        out: List[IngestionChunk] = []
        removed = 0

        for ch in chunks:
            if ch.modality != Modality.TEXT:
                out.append(ch)
                continue
            try:
                if (
                    ch.metadata.chunk_type == ChunkType.CODE
                    or ch.metadata.content_classification == "code"
                ):
                    out.append(ch)
                    continue
            except Exception:
                pass
            txt = (ch.content or "").strip()
            if txt and _REPETITION_RE.search(txt):
                logger.info(
                    f"[REPETITION-FILTER] Dropping garbage-repetition chunk "
                    f"p{ch.metadata.page_number} '{txt[:60]}'"
                )
                removed += 1
                continue
            out.append(ch)

        if removed:
            logger.info(
                f"[REPETITION-FILTER] Removed {removed} token-repetition garbage chunks "
                f"({before} → {len(out)})"
            )
        return out

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
                    overlap_chars=0,  # no overlap — prevents duplicate sentence content
                )
            # Drop empty/whitespace-only parts so a trailing-newline edge case
            # does not emit an empty "Part N/N" chunk (UNIVERSAL_FAIL trigger).
            parts = [p for p in parts if p and p.strip()]
            if len(parts) <= 1:
                out.append(ch)
                continue

            split_count += 1
            for idx, sub in enumerate(parts):
                if is_code:
                    sub_is_code = self._looks_like_code_text(sub)
                    sub_chunk_type = ChunkType.CODE if sub_is_code else ChunkType.PARAGRAPH
                    sub_content_classification = "code" if sub_is_code else self._classify_text_content(sub)
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
                    # Charter §3.2 Phase A step 5 site 3 (oversize-split):
                    # build a UIRChunk that inherits the original chunk's
                    # locator + extraction_method, then emit via from_uir.
                    # The new chunk_id is overridden post-construction to
                    # the v2.16-stable "_oN" suffix so the join key against
                    # any retrieval-side stash stays preserved.
                    _orig_bbox = (
                        ch.metadata.spatial.bbox
                        if ch.metadata.spatial and ch.metadata.spatial.bbox
                        else None
                    )
                    _orig_pw = (
                        ch.metadata.spatial.page_width
                        if ch.metadata.spatial
                        else None
                    )
                    _orig_ph = (
                        ch.metadata.spatial.page_height
                        if ch.metadata.spatial
                        else None
                    )
                    _orig_breadcrumb = (
                        list(ch.metadata.hierarchy.breadcrumb_path)
                        if ch.metadata.hierarchy
                        else []
                    )
                    _new_breadcrumb = _orig_breadcrumb + [
                        f"[Oversize Split {idx+1}/{len(parts)}]"
                    ]
                    _new_level = (
                        (ch.metadata.hierarchy.level or 2) + 1
                        if ch.metadata and ch.metadata.hierarchy
                        else 3
                    )
                    if _orig_bbox:
                        _ov_locator = UIRLocator(
                            type=UIRLocatorType.BBOX,
                            bbox=list(_orig_bbox),
                            page_number=ch.metadata.page_number,
                            coordinate_frame=UIRCoordinateFrame.PDF_PAGE_PORTRAIT,
                        )
                    else:
                        _ov_locator = UIRLocator(
                            type=UIRLocatorType.FLOW_OFFSET,
                            page_number=ch.metadata.page_number,
                            coordinate_frame=UIRCoordinateFrame.UNKNOWN,
                            path=f"page:{ch.metadata.page_number}:oversize:{idx+1}",
                        )
                    _ov_uir = UIRChunk(
                        modality=Modality.TEXT,
                        content=sub,
                        locator=_ov_locator,
                        confidence=UIRConfidenceBreakdown(),
                        extraction_method=ch.metadata.extraction_method,
                        extraction_engine_version="docling-2.86.0",
                        parent_heading=(
                            ch.metadata.hierarchy.parent_heading
                            if ch.metadata.hierarchy
                            else None
                        ),
                    )
                    new_chunk = IngestionChunk.from_uir(
                        _ov_uir,
                        doc_id=ch.doc_id,
                        source_file=ch.metadata.source_file,
                        file_type=ch.metadata.file_type,
                        position=self._next_chunk_position(),
                        page_width=_orig_pw,
                        page_height=_orig_ph,
                        chunk_type=sub_chunk_type,
                        prev_text=(
                            ch.semantic_context.prev_text_snippet
                            if ch.semantic_context
                            else None
                        ),
                        next_text=(
                            ch.semantic_context.next_text_snippet
                            if ch.semantic_context
                            else None
                        ),
                        breadcrumb_path=_new_breadcrumb,
                        **{k: v for k, v in self._intelligence_metadata.items() if v is not None},
                    )
                    new_chunk.metadata.content_classification = sub_content_classification
                    new_chunk.metadata.hierarchy.level = _new_level
                    # v2.16 stable chunk_id suffix preserved (retrieval-side
                    # adjacency depends on this).
                    new_chunk.chunk_id = f"{ch.chunk_id}_o{idx+1}"
                    out.append(new_chunk)
                except Exception:
                    out.append(ch)
                    break

        if split_count:
            logger.info(f"[OVERSIZE-BREAKER] Split {split_count} oversized chunks (> {max_chars} chars)")

        return out

    def _looks_like_code_text(self, text: str) -> bool:
        """Determine whether *text* looks like source code rather than prose.

        Strategy (in order of precedence):

        1. Fenced-code marker → True immediately.
        2. AST parse — if the text is valid Python with a non-empty body it is
           definitively code.  A file consisting only of comments produces an
           empty body and is NOT counted as code here.
        3. REPL session — strip ``>>> ``/``... `` prefixes and re-parse.
           Falls back to assignment (``=``) check or compound-keyword
           verification via ``ast.parse``.  Index entries that use ``>>>`` as a
           cross-reference symbol never contain ``=`` and their stripped content
           does not parse as Python, so they fall through as non-code.
        4. Incomplete definition guard — any line starting with ``def``,
           ``class``, or ``async def`` is code even when the body is absent.
        5. Shebang guard — ``#!`` on the first line is code.
        6. Structural scoring — no regex; just counts of indentation, per-line
           keyword usage (each keyword verified by a mini ``ast.parse``), and
           bracket/assignment density.  Prose sentences accumulate a negative
           offset.  Score ÷ line_count ≥ 2.0 → code.
        7. Flat-code detection — a single long line (> 80 chars, no newlines)
           with ≥ 2 Python keywords and at least one code operator is treated
           as flattened code from a broken PDF generator.
        """
        import ast
        import keyword
        import re as _re

        if not text:
            return False

        t = text.strip("\n")
        if "```" in t:
            return True

        lines = [ln for ln in t.splitlines() if ln.strip()]
        if not lines:
            return False

        # ── 1. AST parse (ground truth for complete blocks) ──────────────────
        # An empty body means the text was only comments — not code on its own.
        try:
            tree = ast.parse(t)
            if tree.body:
                return True
        except (SyntaxError, ValueError):
            # ValueError: "source code string cannot contain null bytes"
            # from encoding-corrupted PDF text layers
            pass

        # ── 2. REPL session ───────────────────────────────────────────────────
        if any(ln.lstrip().startswith(">>> ") for ln in lines):
            extracted = [
                ln.lstrip()[4:]
                for ln in lines
                if ln.lstrip().startswith((">>> ", "... "))
            ]
            if extracted:
                try:
                    tree = ast.parse("\n".join(extracted))
                    # Only return True when the parse result contains real code
                    # constructs.  A bare tuple like (MISSING_VALUE, 42-43) is
                    # valid Python but is almost certainly a book-index entry,
                    # not a REPL expression — it has no calls, assignments, or
                    # control-flow nodes.
                    _CODE_NODES = (
                        ast.Call, ast.Assign, ast.AugAssign, ast.AnnAssign,
                        ast.For, ast.While, ast.If, ast.With, ast.Try,
                        ast.Return, ast.Yield, ast.YieldFrom,
                        ast.Import, ast.ImportFrom,
                        ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                        ast.Subscript, ast.Attribute,
                    )
                    if tree.body and any(isinstance(n, _CODE_NODES) for n in ast.walk(tree)):
                        return True
                except (SyntaxError, ValueError):
                    pass
                # Extracted text didn't parse cleanly.
                # An assignment operator is a reliable signal for real REPL code;
                # book-index entries never use "=".
                if any("=" in e for e in extracted):
                    return True
                # Compound keyword (for/while/with/if/try) used as real Python.
                # Verify by wrapping in a dummy body — index entries starting
                # with an English keyword word (e.g. "with, 284") won't parse.
                for e in extracted:
                    first = e.split()[0].rstrip(":([,") if e.split() else ""
                    if keyword.iskeyword(first):
                        try:
                            ast.parse(e + "\n    pass")
                            return True
                        except (SyntaxError, ValueError):
                            pass
            # REPL markers present but content looks like cross-references.
            return False

        # ── 3. Incomplete definition guard ───────────────────────────────────
        if any(ln.lstrip().startswith(("def ", "class ", "async def ")) for ln in lines):
            return True

        # ── 4. Shebang guard ─────────────────────────────────────────────────
        if lines[0].lstrip().startswith("#!"):
            return True

        # ── 5. Structural scoring ─────────────────────────────────────────────
        score = 0.0
        for ln in lines:
            stripped = ln.strip()
            tokens = stripped.split()
            if not tokens:
                continue

            # Indentation is the strongest single code signal.
            if ln.startswith(("    ", "\t")):
                score += 3.0
                continue

            first = tokens[0].rstrip(":([,")

            # import / from — verify it's Python syntax, not English "from ...".
            if first in ("import", "from"):
                try:
                    ast.parse(stripped)
                    score += 4.0
                except (SyntaxError, ValueError):
                    score += 0.5  # prose: "from this perspective..."
                continue

            # Other Python keywords at line start — verify per-line syntax.
            if keyword.iskeyword(first):
                try:
                    ast.parse(stripped)
                    score += 3.0
                except (SyntaxError, ValueError):
                    try:
                        ast.parse(stripped + "\n    pass")
                        score += 3.0
                    except (SyntaxError, ValueError):
                        score += 0.5  # keyword used as an English word
                continue

            # Bracket / assignment density (capped per line).
            syms = (stripped.count("(") + stripped.count("=")
                    + stripped.count("[") + stripped.count("{"))
            score += min(syms * 0.8, 2.5)

            # Prose counter-signal: sentence-shaped line.
            if stripped[0].isupper() and len(tokens) >= 7 and stripped[-1] in ".?!":
                score -= 2.0

        if score / len(lines) >= 2.0:
            # Guard: pure indentation (OCR column-layout artifact) is not
            # sufficient on its own.  Require at least one unambiguous code
            # token — assignment, subscript/dict literal, function call, or a
            # Python keyword in a syntactically valid position.
            _code_anchor = _re.compile(
                r"[=\[{]"                          # assignment / subscript / dict
                r"|\w\("                           # function call: word(
                r"|^\s*(?:import\s"                # import statement
                r"|from\s+[\w.]+\s+import"         # from X import Y
                r"|def |class "                    # definition
                r"|return\b|yield\b|raise\b|pass\b"  # returns / control
                r"|elif\b|else:|try:|except\b|finally:)",  # block starters
                _re.MULTILINE,
            )
            if _code_anchor.search(t):
                return True
            # No genuine code token found — likely OCR-indented prose.

        # ── 6. Flat-code detection (PDF strips all newlines from code blocks) ──
        if len(lines) == 1:
            kw_count = sum(1 for w in t.split() if keyword.iskeyword(w))
            if len(t) > 80 and kw_count >= 2 and any(c in t for c in "=([{"):
                return True
            # Flat import chain: starts with 'import'/'from', with multiple import
            # occurrences or one import keyword + an operator.  Catches multi-statement
            # import blocks that PDF extraction flattened onto a single line.
            # Threshold lowered to 60 chars to catch import+assignment concatenations.
            if len(t) > 60:
                split_words = t.split()
                if split_words and split_words[0] in ("import", "from"):
                    import_kw_count = len(_re.findall(r'\bimport\s+[A-Za-z_]', t))
                    if import_kw_count >= 2:
                        return True
                    if import_kw_count >= 1 and any(c in t for c in "=([{"):
                        return True

        # ── 7. Compact single-line if/else control flow ───────────────────────
        # PDFs sometimes flatten "if cond:\n    a()\nelse:\n    b()" onto one line.
        # Require: `if <cond>:` (colon after condition), `else:` (colon after else),
        # AND at least one bracket/call token — rules out English "if … else" prose.
        if (
            len(lines) == 1
            and _re.search(r"\bif\b[^:]+:", t)
            and _re.search(r"\belse\s*:", t)
            and any(c in t for c in "=([{")
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

        # Step 3a: Code hygiene for ALL profiles — detect, reclassify, and reflow
        # flat code chunks. Without this, academic papers with code snippets produce
        # unreadable single-line output (e.g., "class Foo: def __init__(self):...").
        valid_chunks = self._apply_code_hygiene(valid_chunks)

        # Step 3a1a: Repair embedded step-number boundaries (a content-level fix:
        # insert a newline between jammed numbered steps so downstream merge/dedup
        # see corrected paragraph boundaries). Re-homed here (PLAN_V3.1 follow-up
        # 2026-06-06) from the deleted spatial boundary-repair bridge, where it was
        # collaterally disabled when P4 cut the bridge; it is a content repair in
        # the same family as Step 3a2 hyphenation, NOT spatial merging.
        valid_chunks = self._repair_infix_step_numbers(valid_chunks)

        # Step 3a1b: Merge mid-sentence chunk boundaries.
        # Layout-aware OCR creates one chunk per region. When a sentence spans
        # two regions, each chunk gets half. Merge consecutive TEXT chunks where
        # the first ends without sentence punctuation and the second starts lowercase.
        valid_chunks = self._merge_mid_sentence_chunks(valid_chunks)

        # Step 3a2: Cross-chunk hyphenation repair — rejoin words split at chunk boundaries.
        # "man-" at end of chunk + "age" at start of next → "manage" + "".
        valid_chunks = self._repair_cross_chunk_hyphenation(valid_chunks)

        # Step 3a2b: Remove same-page text subset chunks.
        # VLM transcription + Docling fallback can produce two text chunks for
        # the same page where one is a subset of the other. Keep the longer one.
        valid_chunks = self._remove_subset_chunks(valid_chunks)

        # Step 3a3: Remove near-duplicate chunks (>85% word overlap).
        # Firearms-type manuals repeat instructions across gun models
        # ("Remove the trigger housing downward" × 6). These pollute RAG top-k.
        valid_chunks = self._remove_near_duplicate_chunks(valid_chunks)

        # Step 3a4: Remove content overlap between consecutive chunks.
        # DSO adds ~55 chars of overlap at chunk boundaries for context continuity,
        # but this duplicates sentences in the vector index. The context is already
        # preserved in prev_text_snippet/next_text_snippet — no need to duplicate
        # it in the actual content.
        valid_chunks = self._deduplicate_chunk_overlap(valid_chunks)

        # Step 3a5: Magazine image enhancement — re-render image regions at high DPI
        # for digital_magazine profile. Docling's layout model extracts oversized regions
        # that include text. Re-rendering from PyMuPDF gives cleaner photo crops.
        profile_type = str(self._intelligence_metadata.get("profile_type", "unknown"))
        if profile_type == "digital_magazine" and self._current_pdf_path:
            enhanced = self._enhance_magazine_images(valid_chunks)
            if enhanced:
                logger.info(f"[MAGAZINE-IMAGE] Enhanced {enhanced} image assets via rendered-region crop")

        # Step 3b: Profile-specific text hygiene (technical manuals are sensitive to
        # embedded page numbers, control chars, hyphenation, and broken chunk joins).
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

    def _per_batch_shortfall_fires(
        self,
        chunks: List[IngestionChunk],
        batches: List["BatchInfo"],
        pdf_path: Path,
    ) -> bool:
        """Phase 2 (PLAN_V2.10.md) — Per-batch TextIntegrityScout trigger.

        Returns True when at least one processed batch's page range shows
        localized text-coverage shortfall, even if doc-level QA-CHECK-01
        variance is within tolerance. Universal page-shape rule (no
        filename- or profile-specific logic). Thresholds live in
        `mmrag_v2.validators.text_integrity_scout_trigger` and are
        defended by `scripts/probe_phase2_scout_threshold.py`.
        """
        try:
            from .validators.text_integrity_scout_trigger import any_batch_fires
        except Exception as imp_err:  # pragma: no cover
            logger.warning(
                f"[RECOVERY] Per-batch trigger module unavailable; falling back to doc-level gate only: {imp_err}"
            )
            return False

        if not batches:
            return False
        if not pdf_path or not Path(pdf_path).exists():
            return False

        # Per-page emitted TEXT-chunk char counts.
        chunk_chars_per_page: Dict[int, int] = {}
        for ch in chunks:
            if ch.modality != Modality.TEXT:
                continue
            content = ch.content or ""
            page_no = getattr(ch.metadata, "page_number", None)
            if not isinstance(page_no, int) or page_no <= 0:
                continue
            chunk_chars_per_page[page_no] = chunk_chars_per_page.get(page_no, 0) + len(content)

        # Per-page source text char counts via PyMuPDF (text layer only).
        source_chars_per_page: Dict[int, int] = {}
        doc: Optional[fitz.Document] = None
        try:
            try:
                doc = fitz.open(str(pdf_path))
            except Exception as open_err:
                logger.warning(
                    f"[RECOVERY] Per-batch trigger: failed to open PDF for source text extraction: {open_err}"
                )
                return False
            for page_idx in range(len(doc)):
                page_no = page_idx + 1
                if self._processed_pages is not None and page_no not in self._processed_pages:
                    continue
                try:
                    txt = doc.load_page(page_idx).get_text("text") or ""
                except Exception:
                    txt = ""
                source_chars_per_page[page_no] = len(txt.strip())
        finally:
            if doc is not None:
                try:
                    doc.close()
                except Exception:
                    pass

        batch_triples = [
            (b.batch_index, int(b.start_page), int(b.end_page)) for b in batches
        ]
        fired, shapes = any_batch_fires(
            batches=batch_triples,
            source_chars_per_page=source_chars_per_page,
            chunk_chars_per_page=chunk_chars_per_page,
        )
        if fired:
            firing = [s for s in shapes if s.fires()]
            details = "; ".join(
                f"batch {s.batch_index + 1} pp{s.start_page}-{s.end_page} "
                f"var={s.variance_ratio:.0%} missing={list(s.missing_pages)}"
                for s in firing
            )
            logger.info(
                f"[RECOVERY] Per-batch shortfall trigger fired on {len(firing)}/{len(shapes)} batch(es): {details}"
            )
        else:
            logger.debug(
                f"[RECOVERY] Per-batch shortfall trigger did not fire across {len(shapes)} batch(es)"
            )
        return fired

    def _run_text_integrity_scout(
        self,
        chunks: List[IngestionChunk],
        source_file: str,
        variance_percent: float,
        force_run: bool = False,
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
            force_run: Phase 2 (PLAN_V2.10.md) — when True, bypass the
                doc-level variance gate. The per-batch shortfall trigger
                ORs into this flag so the scout fires on localized
                shortfalls inside large documents whose doc-level
                variance lands within tolerance.

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

        if not force_run and variance_percent >= RECOVERY_THRESHOLD:
            logger.info(
                f"[RECOVERY] Variance {variance_percent:.1f}% is within tolerance, skipping recovery"
            )
            return chunks
        if force_run and variance_percent >= RECOVERY_THRESHOLD:
            logger.info(
                f"[RECOVERY] Doc variance {variance_percent:.1f}% within tolerance but "
                "per-batch shortfall trigger fired; running scout on localized batches."
            )

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
                        _pg_wh = page_size_per_page.get(page_no)
                        # Charter §3.2 Phase A step 5 site 4 (recovery_frontpage):
                        # build UIRChunk with BBOX locator (front-page rescue
                        # always has the block's source bbox normalized to
                        # [0,1000]), then emit via from_uir.
                        _rec_bbox = _normalize_bbox_pdf_points(page_no, bbox)
                        _rec_uir = UIRChunk(
                            modality=Modality.TEXT,
                            content=text_clean,
                            locator=UIRLocator(
                                type=UIRLocatorType.BBOX,
                                bbox=list(_rec_bbox),
                                page_number=page_no,
                                coordinate_frame=UIRCoordinateFrame.PDF_PAGE_PORTRAIT,
                            ),
                            confidence=UIRConfidenceBreakdown(),
                            extraction_method="recovery_frontpage",
                            extraction_engine_version="pymupdf-recovery",
                        )
                        recovery_chunk = IngestionChunk.from_uir(
                            _rec_uir,
                            doc_id=self._doc_hash or "unknown",
                            source_file=source_file,
                            file_type=FileType.PDF,
                            position=self._next_chunk_position(),
                            page_width=int(_pg_wh[0]) if _pg_wh and _pg_wh[0] > 0 else None,
                            page_height=int(_pg_wh[1]) if _pg_wh and _pg_wh[1] > 0 else None,
                            breadcrumb_path=[doc_title, f"Page {page_no}", "[RECOVERED-FRONT]"],
                            **self._intelligence_metadata,
                        )
                        recovery_chunk.metadata.content_classification = (
                            self._classify_recovery_text_content(text_clean)
                        )
                        # v2.16 literal: hierarchy.level=3 explicit on the
                        # original. Auto-compute from len(breadcrumb)=3 also
                        # gives 3, but set explicitly for parity.
                        recovery_chunk.metadata.hierarchy.level = 3
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
                        # Charter §3.2 Phase A step 5 site 5 (recovery_scan):
                        # no bbox available (paragraph-level orphan; bbox is
                        # block-level only for some recovery paths). FLOW_OFFSET
                        # locator preserves the v2.16 shape (SpatialMetadata
                        # absent on this branch).
                        doc_title = Path(source_file).stem if source_file else "Document"
                        _scan_uir = UIRChunk(
                            modality=Modality.TEXT,
                            content=para_clean,
                            locator=UIRLocator(
                                type=UIRLocatorType.FLOW_OFFSET,
                                page_number=page_no,
                                coordinate_frame=UIRCoordinateFrame.UNKNOWN,
                                path=f"page:{page_no}:recovery_scan:{para_idx}",
                            ),
                            confidence=UIRConfidenceBreakdown(),
                            extraction_method="recovery_scan",
                            extraction_engine_version="pymupdf-recovery",
                        )
                        recovery_chunk = IngestionChunk.from_uir(
                            _scan_uir,
                            doc_id=self._doc_hash or "unknown",
                            source_file=source_file,
                            file_type=FileType.PDF,
                            position=self._next_chunk_position(),
                            breadcrumb_path=[doc_title, f"Page {page_no}", "[RECOVERED]"],
                            **self._intelligence_metadata,
                        )
                        recovery_chunk.metadata.content_classification = (
                            self._classify_recovery_text_content(para_clean)
                        )
                        recovery_chunk.metadata.hierarchy.level = 3
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
                            cleaned_block = _clean_recovery_text(
                                block_text.strip(), is_code=(classification == "code")
                            )
                            if len(cleaned_block) < 20:
                                continue
                            _pg_wh2 = page_size_per_page.get(page_no)
                            # Charter §3.2 Phase A step 5 site 6 (recovery_subsurface):
                            # text under figure; carries the figure's bbox + the
                            # figure's asset_ref (so retrieval can link rescued
                            # text back to the visual context it was extracted
                            # from). The asset_ref on UIRChunk is the path str —
                            # post-construction we copy the parent fig_chunk's
                            # full AssetReference (mime_type, width_px, height_px,
                            # file_size_bytes) onto the recovery chunk.
                            _sub_bbox = [int(v) for v in fig_bbox]
                            _sub_asset_path = (
                                fig_chunk.asset_ref.file_path
                                if fig_chunk.asset_ref
                                else None
                            )
                            _sub_uir = UIRChunk(
                                modality=Modality.TEXT,
                                content=cleaned_block,
                                locator=UIRLocator(
                                    type=UIRLocatorType.BBOX,
                                    bbox=_sub_bbox,
                                    page_number=page_no,
                                    coordinate_frame=UIRCoordinateFrame.PDF_PAGE_PORTRAIT,
                                ),
                                confidence=UIRConfidenceBreakdown(),
                                extraction_method="recovery_subsurface",
                                extraction_engine_version="pymupdf-recovery",
                                asset_ref=_sub_asset_path,
                            )
                            recovery_chunk = IngestionChunk.from_uir(
                                _sub_uir,
                                doc_id=self._doc_hash or "unknown",
                                source_file=source_file,
                                file_type=FileType.PDF,
                                position=self._next_chunk_position(),
                                page_width=int(_pg_wh2[0]) if _pg_wh2 and _pg_wh2[0] > 0 else None,
                                page_height=int(_pg_wh2[1]) if _pg_wh2 and _pg_wh2[1] > 0 else None,
                                breadcrumb_path=[doc_title, f"Page {page_no}", label],
                                **self._intelligence_metadata,
                            )
                            # v2.16 invariant: subsurface chunks inherit the
                            # parent figure's full AssetReference (not just
                            # file_path).
                            if recovery_chunk.asset_ref is not None and fig_chunk.asset_ref is not None:
                                recovery_chunk.asset_ref = fig_chunk.asset_ref
                            recovery_chunk.metadata.content_classification = classification
                            recovery_chunk.metadata.hierarchy.level = 3
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
                        _pg_wh3 = page_size_per_page.get(page_no)
                        # Charter §3.2 Phase A step 5 site 7 (recovery_gap_fill):
                        # spatial-gap rescue beyond figures. Bbox via the
                        # pdf-points→normalized helper; BBOX locator.
                        _gap_bbox = _normalize_bbox_pdf_points(page_no, bbox)
                        _gap_uir = UIRChunk(
                            modality=Modality.TEXT,
                            content=text_clean,
                            locator=UIRLocator(
                                type=UIRLocatorType.BBOX,
                                bbox=list(_gap_bbox),
                                page_number=page_no,
                                coordinate_frame=UIRCoordinateFrame.PDF_PAGE_PORTRAIT,
                            ),
                            confidence=UIRConfidenceBreakdown(),
                            extraction_method="recovery_gap_fill",
                            extraction_engine_version="pymupdf-recovery",
                        )
                        recovery_chunk = IngestionChunk.from_uir(
                            _gap_uir,
                            doc_id=self._doc_hash or "unknown",
                            source_file=source_file,
                            file_type=FileType.PDF,
                            position=self._next_chunk_position(),
                            page_width=int(_pg_wh3[0]) if _pg_wh3 and _pg_wh3[0] > 0 else None,
                            page_height=int(_pg_wh3[1]) if _pg_wh3 and _pg_wh3[1] > 0 else None,
                            breadcrumb_path=[doc_title, f"Page {page_no}", label],
                            **self._intelligence_metadata,
                        )
                        recovery_chunk.metadata.content_classification = classification
                        recovery_chunk.metadata.hierarchy.level = 3
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

    def _apply_recovery_vs_primary_dedup(
        self,
        chunks: List[IngestionChunk],
    ) -> List[IngestionChunk]:
        """Drop recovery text chunks that re-extract content the primary (VLM)
        extraction already captured on the same page.

        The TextIntegrityScout pulls whole-page text from the PDF text layer to
        rescue genuinely-missing text. On code/dense pages the VLM already
        extracted the content cleanly (with indentation), but the scout re-adds
        it from the flush-left text layer, producing mangled duplicates — output
        bloat plus R3 code-indentation pollution (the same code appears once
        clean as ``modality=code`` and again flush-left as recovery text). The
        Highlander pass above handles recovery-vs-TABLE; this handles
        recovery-vs-TEXT/CODE.

        A recovery chunk is dropped only when >=85% of its unique tokens are
        already present in the primary chunks on its page. A recovery chunk on a
        page with NO primary chunk (a page the VLM dropped entirely) is always
        kept — that is the scout's legitimate purpose — and a recovery chunk with
        substantial genuinely-new content stays below the threshold and survives.
        The 0.85 floor was set from measured data: every spurious AIOS recovery
        duplicate scored 0.92-1.00 against the primary, while genuinely-missing
        prose (introducing new content words) scores far lower.

        DUAL-LAYER NOTE: this drops the duplicates at PRODUCTION time so they
        never reach the JSONL. The R3 gate metric defends the same fact
        independently at AUDIT time (``scripts/_code_quality._duplicates_primary``)
        so files already on disk, or any other producer's output, are still
        scored correctly. Two layers, one domain fact — keep them consistent. The
        two use DIFFERENT algorithms by design (this: token-overlap; audit:
        substring-window), so they need not agree on every edge case; the shared
        contract (both must catch the canonical recovery-duplicate, and keep a
        VLM-dropped-page rescue) is enforced by
        ``tests/test_dual_layer_recovery_dedup.py`` (PR #4 Finding 4 follow-up).
        """
        overlap_floor = 0.85

        recovery_methods = {
            "recovery_gap_fill",
            "recovery_scan",
            "recovery_subsurface",
            "recovery_frontpage",
        }

        def _tokens(text: str) -> set:
            return set(re.findall(r"[A-Za-z0-9]{3,}", (text or "").lower()))

        primary_tokens_by_page: Dict[int, set] = {}
        for chunk in chunks:
            method = str(getattr(chunk.metadata, "extraction_method", "") or "").lower()
            if method in recovery_methods:
                continue
            if chunk.modality in (Modality.TEXT, Modality.CODE) and chunk.content:
                page_no = int(getattr(chunk.metadata, "page_number", 0) or 0)
                if page_no > 0:
                    primary_tokens_by_page.setdefault(page_no, set()).update(_tokens(chunk.content))

        if not primary_tokens_by_page:
            return chunks

        kept: List[IngestionChunk] = []
        dropped = 0
        for chunk in chunks:
            method = str(getattr(chunk.metadata, "extraction_method", "") or "").lower()
            page_no = int(getattr(chunk.metadata, "page_number", 0) or 0)
            if (
                chunk.modality == Modality.TEXT
                and method in recovery_methods
                and page_no in primary_tokens_by_page
                and chunk.content
            ):
                rec = _tokens(chunk.content)
                if rec:
                    overlap = len(rec & primary_tokens_by_page[page_no]) / len(rec)
                    if overlap >= overlap_floor:
                        dropped += 1
                        logger.info(
                            f"[RECOVERY-DEDUP] Drop recovery chunk {chunk.chunk_id} "
                            f"(page={page_no}, method={method}, primary_overlap={overlap:.2f})"
                        )
                        continue
            kept.append(chunk)

        if dropped:
            print(
                f"\n🧹 [RECOVERY-DEDUP] Dropped {dropped} recovery chunk(s) duplicating "
                f"primary VLM extraction",
                flush=True,
            )
            logger.info(
                f"[RECOVERY-DEDUP] Dropped {dropped} recovery text chunk(s) already "
                f"present in the primary extraction"
            )
        return kept

    def _apply_vlm_table_iou_dedup(
        self,
        chunks: List[IngestionChunk],
    ) -> List[IngestionChunk]:
        """v2.16 Phase 4 — drop text chunks that spatially overlap a VLM-
        extracted table on the same page above
        `plan.dedup_vlm_table_iou_threshold`.

        Targets the v2.14 P1 CarOK regression: when force_table_vlm produces
        clean markdown tables, Docling's text-extraction pass simultaneously
        emits the SAME content as flat prose. Retrieval picks the prose chunk
        29/30 times. IoU>0.85 suppression is the spec'd close (PLAN_V2.16.md
        §3 Phase 4). At 0.0 threshold the pass is disabled (the
        `_apply_table_recovery_highlander_dedup` pass above handles the
        narrower recovery_* case).
        """
        plan = self._conversion_plan
        threshold = float(getattr(plan, "dedup_vlm_table_iou_threshold", 0.85))
        if threshold <= 0.0:
            return chunks

        # Accept both forced and emergency VLM methods — the regression mode
        # appears whenever a VLM markdown table is emitted, regardless of why.
        vlm_methods = {
            "vlm_table_markdown_forced",
            "vlm_table_markdown",
            "vlm_table_markdown_emergency",
            "vlm_table",
        }

        from .utils.bbox import bbox_iou

        vlm_tables_by_page: Dict[int, List[IngestionChunk]] = {}
        for chunk in chunks:
            try:
                method = str(
                    getattr(chunk.metadata, "extraction_method", "") or ""
                ).lower()
                if chunk.modality == Modality.TABLE and method in vlm_methods:
                    page_no = int(
                        getattr(chunk.metadata, "page_number", 0) or 0
                    )
                    if page_no > 0:
                        vlm_tables_by_page.setdefault(page_no, []).append(chunk)
            except Exception:
                continue

        if not vlm_tables_by_page:
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

        kept: List[IngestionChunk] = []
        dropped_total = 0
        per_page_drops: Dict[int, int] = {}

        for chunk in chunks:
            try:
                page_no = int(getattr(chunk.metadata, "page_number", 0) or 0)
            except Exception:
                kept.append(chunk)
                continue

            if chunk.modality != Modality.TEXT or page_no not in vlm_tables_by_page:
                kept.append(chunk)
                continue

            text_bbox = _safe_bbox(chunk)
            if text_bbox is None:
                kept.append(chunk)
                continue

            should_drop = False
            drop_iou = 0.0
            for vlm_table in vlm_tables_by_page[page_no]:
                tbl_bbox = _safe_bbox(vlm_table)
                if tbl_bbox is None:
                    continue
                iou = bbox_iou(text_bbox, tbl_bbox)
                if iou > threshold:
                    should_drop = True
                    drop_iou = iou
                    break

            if should_drop:
                dropped_total += 1
                per_page_drops[page_no] = per_page_drops.get(page_no, 0) + 1
                logger.info(
                    f"[VLM-TABLE-DEDUP] Dropping text chunk {chunk.chunk_id} "
                    f"(page={page_no}, iou={drop_iou:.3f}, threshold={threshold:.2f})"
                )
                continue
            kept.append(chunk)

        if dropped_total > 0:
            logger.info(
                f"[VLM-TABLE-DEDUP] Suppressed {dropped_total} text chunks via "
                f"IoU>{threshold:.2f} (per-page: {per_page_drops})"
            )
            print(
                f"🔁 [VLM-TABLE-DEDUP] Suppressed {dropped_total} text chunks "
                f"overlapping VLM tables (IoU>{threshold:.2f})",
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
                            # TEXT chunks must have a chunk_type; image chunks start with None.
                            if img_chunk.metadata.chunk_type is None:
                                img_chunk.metadata.chunk_type = ChunkType.PARAGRAPH
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
                            if img_chunk.metadata.chunk_type is None:
                                img_chunk.metadata.chunk_type = ChunkType.PARAGRAPH
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
                    # Charter §3.2 Phase A step 5 site 8 (enhanced_frontpage):
                    # PyMuPDF block recovery on enhanced front pages.
                    _enh_uir = UIRChunk(
                        modality=Modality.TEXT,
                        content=text_clean,
                        locator=UIRLocator(
                            type=UIRLocatorType.BBOX,
                            bbox=list(bbox),
                            page_number=page_no,
                            coordinate_frame=UIRCoordinateFrame.PDF_PAGE_PORTRAIT,
                        ),
                        confidence=UIRConfidenceBreakdown(),
                        extraction_method="enhanced_frontpage",
                        extraction_engine_version="pymupdf-recovery",
                    )
                    new_chunk = IngestionChunk.from_uir(
                        _enh_uir,
                        doc_id=self._doc_hash or "unknown",
                        source_file=str(self._current_pdf_path.name),
                        file_type=FileType.PDF,
                        position=self._next_chunk_position(),
                        page_width=int(page_w) if page_w > 0 else None,
                        page_height=int(page_h) if page_h > 0 else None,
                        breadcrumb_path=[doc_title, f"Page {page_no}", "[ENHANCED]"],
                        **self._intelligence_metadata,
                    )
                    new_chunk.metadata.content_classification = (
                        self._classify_recovery_text_content(text_clean)
                    )
                    new_chunk.metadata.hierarchy.level = 3
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
            # IMAGE-BBOX-AWARE SOURCE TEXT EXTRACTION
            # ================================================================
            # PyMuPDF sees ALL text in the PDF text layer, including labels
            # and data embedded in charts/graphs/figures. Docling correctly
            # classifies chart regions as IMAGE chunks. If we count that text
            # as "expected" source text, the variance is inflated because the
            # content IS preserved — just as image chunks, not text chunks.
            #
            # Fix: extract PyMuPDF text blocks WITH positions, then exclude
            # blocks whose area overlaps >50% with an IMAGE chunk bbox.
            # This gives an accurate baseline of text that SHOULD be in
            # TEXT chunks, not text that's legitimately in IMAGE chunks.
            # ================================================================
            source_text = ""
            if self._current_pdf_path and self._current_pdf_path.exists():
                try:
                    doc: Optional[fitz.Document] = None
                    all_text_parts = []

                    # Build per-page IMAGE bbox lookup from ALL chunks (not just text)
                    image_bboxes_by_page: Dict[int, list] = {}
                    for c in chunks:
                        if (
                            c.modality == Modality.IMAGE
                            and c.metadata
                            and c.metadata.spatial
                            and c.metadata.spatial.bbox
                        ):
                            pg = c.metadata.page_number or 0
                            image_bboxes_by_page.setdefault(pg, []).append(
                                c.metadata.spatial.bbox  # [x0, y0, x1, y1] in 0-1000 coords
                            )

                    excluded_chars = 0
                    try:
                        doc = fitz.open(self._current_pdf_path)
                        for page_idx in range(len(doc)):
                            page_no = page_idx + 1
                            if self._processed_pages is not None and page_no not in self._processed_pages:
                                continue
                            page = doc.load_page(page_idx)

                            # Filter IMAGE bboxes to only SMALL chart/graph regions.
                            # Large images (>35% of page area) are editorial photos
                            # where text IS legitimately placed around/within them.
                            # Charts/graphs (<35%) contain embedded text labels that
                            # PyMuPDF sees but Docling correctly treats as image content.
                            _PAGE_AREA = 1_000_000  # 1000×1000 normalized page
                            _MAX_CHART_FRACTION = 0.35
                            chart_bboxes = [
                                bbox for bbox in image_bboxes_by_page.get(page_no, [])
                                if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < _PAGE_AREA * _MAX_CHART_FRACTION
                            ]
                            if not chart_bboxes:
                                # No small chart-sized IMAGE chunks — use all text
                                page_text = page.get_text("text")
                                if page_text:
                                    all_text_parts.append(page_text.strip())
                            else:
                                # Has chart-sized IMAGE chunks — filter text blocks by bbox overlap
                                pw = page.rect.width or 1.0
                                ph = page.rect.height or 1.0
                                blocks = page.get_text("blocks")
                                for block in blocks:
                                    if block[6] != 0:  # block_type 0 = text
                                        continue
                                    block_text = (block[4] or "").strip()
                                    if not block_text:
                                        continue
                                    # Normalize block bbox to 0-1000 coordinate space
                                    bx0 = int(block[0] / pw * 1000)
                                    by0 = int(block[1] / ph * 1000)
                                    bx1 = int(block[2] / pw * 1000)
                                    by1 = int(block[3] / ph * 1000)
                                    block_area = max(1, (bx1 - bx0) * (by1 - by0))

                                    in_chart = False
                                    for ibbox in chart_bboxes:
                                        ix0, iy0, ix1, iy1 = ibbox
                                        ox0 = max(bx0, ix0)
                                        oy0 = max(by0, iy0)
                                        ox1 = min(bx1, ix1)
                                        oy1 = min(by1, iy1)
                                        if ox0 < ox1 and oy0 < oy1:
                                            overlap_area = (ox1 - ox0) * (oy1 - oy0)
                                            if overlap_area / block_area > 0.70:
                                                in_chart = True
                                                break

                                    if in_chart:
                                        excluded_chars += len(block_text)
                                    else:
                                        all_text_parts.append(block_text)
                    finally:
                        if doc is not None:
                            try:
                                doc.close()
                            except Exception:
                                pass
                    source_text = "\n".join(all_text_parts)
                    logger.info(
                        f"[QA-CHECK-01] Extracted {len(source_text)} chars from PDF for validation "
                        f"(excluded {excluded_chars} chars in IMAGE regions, "
                        f"pages={len(self._processed_pages) if self._processed_pages else 'ALL'})"
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

                # Charter §3.2 Phase A step 5 site 9 (smart-split): build
                # a UIRChunk that inherits the parent's locator + extraction
                # method, then emit via from_uir. The v2.16 chunk_id suffix
                # `_sN` is overridden post-construction.
                _orig_bbox = (
                    chunk.metadata.spatial.bbox
                    if chunk.metadata.spatial and chunk.metadata.spatial.bbox
                    else None
                )
                _orig_pw = (
                    chunk.metadata.spatial.page_width
                    if chunk.metadata.spatial
                    else None
                )
                _orig_ph = (
                    chunk.metadata.spatial.page_height
                    if chunk.metadata.spatial
                    else None
                )
                _orig_breadcrumb = (
                    list(chunk.metadata.hierarchy.breadcrumb_path)
                    if chunk.metadata.hierarchy
                    else []
                )
                _new_level = (
                    (chunk.metadata.hierarchy.level or 2) + 1
                    if chunk.metadata.hierarchy
                    else 3
                )
                if _orig_bbox:
                    _sp_locator = UIRLocator(
                        type=UIRLocatorType.BBOX,
                        bbox=list(_orig_bbox),
                        page_number=chunk.metadata.page_number,
                        coordinate_frame=UIRCoordinateFrame.PDF_PAGE_PORTRAIT,
                    )
                else:
                    _sp_locator = UIRLocator(
                        type=UIRLocatorType.FLOW_OFFSET,
                        page_number=chunk.metadata.page_number,
                        coordinate_frame=UIRCoordinateFrame.UNKNOWN,
                        path=f"page:{chunk.metadata.page_number}:smartsplit:{idx+1}",
                    )
                _sp_uir = UIRChunk(
                    modality=Modality.TEXT,
                    content=sub_text,
                    locator=_sp_locator,
                    confidence=UIRConfidenceBreakdown(),
                    extraction_method=chunk.metadata.extraction_method,
                    extraction_engine_version="docling-2.86.0",
                    parent_heading=(
                        chunk.metadata.hierarchy.parent_heading
                        if chunk.metadata.hierarchy
                        else None
                    ),
                )
                new_chunk = IngestionChunk.from_uir(
                    _sp_uir,
                    doc_id=chunk.doc_id,
                    source_file=chunk.metadata.source_file,
                    file_type=chunk.metadata.file_type,
                    position=self._next_chunk_position(),
                    page_width=_orig_pw,
                    page_height=_orig_ph,
                    chunk_type=(chunk.metadata.chunk_type or ChunkType.PARAGRAPH),
                    prev_text=(
                        chunk.semantic_context.prev_text_snippet
                        if chunk.semantic_context
                        else None
                    ),
                    next_text=(
                        chunk.semantic_context.next_text_snippet
                        if chunk.semantic_context
                        else None
                    ),
                    breadcrumb_path=(
                        _orig_breadcrumb + [f"[Split {idx+1}/{len(sub_chunks)}]"]
                    ),
                    **{k: v for k, v in self._intelligence_metadata.items() if v is not None},
                )
                new_chunk.metadata.content_classification = getattr(
                    chunk.metadata, "content_classification", None
                )
                new_chunk.metadata.hierarchy.level = _new_level

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

                # v2.9: defer instead of discard. The conversion-time
                # path is intentionally run with --vision-provider none
                # in the v2.9 architecture; VLM-side verification + real
                # description happens in the post-conversion enrichment
                # script (`scripts/enrich_image_chunks_v29.py`). Hard
                # discard at this site previously erased every page that
                # only had full-page imagery (Combat p4 lost its 9-image
                # spread). Mark the chunk as pending and let it through.
                if self._vision_manager is None:
                    logger.info(
                        f"[FULL-PAGE-GUARD] DEFERRING full-page asset on "
                        f"page {chunk.metadata.page_number} (no conversion-time "
                        f"VLM; will enrich post-conversion)"
                    )
                    if chunk.metadata:
                        chunk.metadata.vision_status = "pending"
                    filtered.append(chunk)
                    continue

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
