"""Cost-optimizer page router — the V3 Phase C default engine.

Per-page pre-flight via PyMuPDF picks the right tool per page:

* Pages with images, tables, or non-trivial vector drawings are
  visually complex → routed to the ``VlmNativeEngine`` (vision-native
  UIR JSON extraction).
* Pages dense with monospace text (source-code blocks) carry no object
  signal but would lose their indentation under Docling → also routed
  to the ``VlmNativeEngine``.
* Pure-prose pages → routed to the ``DoclingFastEngine`` (CPU,
  OCR disabled) for sub-second extraction.

Outputs a single unified ``UniversalDocument`` with mixed-provenance
pages. Records the routing decisions on ``last_routing_decisions``
for observability.

This module deliberately does NOT import Docling itself — the Docling
boundary lives entirely inside ``docling_fast.py``. The router is
auditable as engine-agnostic glue.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF — pre-flight only; pure layout introspection

from mmrag_v2.universal.intermediate import (
    DocumentMetadata,
    UniversalDocument,
    UniversalPage,
    create_document,
)

from .docling_fast import DoclingFastEngine
from .mineru_native import MineruNativeEngine, extract_page_mineru
from .vlm_native import VlmNativeEngine, extract_page_vlm
from .vlm_provider import VlmInfraError

logger = logging.getLogger(__name__)


# A page with *only* a handful of vector drawings (page borders, header
# rules, footer lines) does not justify a VLM call. The router treats
# drawings as a visual signal only once their count crosses this
# threshold. Override via ``VLM_DRAWINGS_THRESHOLD`` env.
DEFAULT_DRAWINGS_THRESHOLD = 10


def _drawings_threshold() -> int:
    raw = os.environ.get("VLM_DRAWINGS_THRESHOLD", "").strip()
    if not raw:
        return DEFAULT_DRAWINGS_THRESHOLD
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_DRAWINGS_THRESHOLD


# Code blocks are the one visually-complex content class that carries NO
# object signal: a pure-text page of source code has zero images, tables,
# or drawings, so the object-only pre-flight would route it to Docling,
# which strips leading whitespace and destroys code indentation. Code is
# almost universally typeset in a monospace font, which prose is not, so
# the monospace-character ratio is a precise, object-independent code
# signal. Empirically calibrated across three doc profiles (code/prose
# mix, code-heavy book, pure-prose novel): code pages land >= 0.28, prose
# pages <= 0.014, with an empty margin between; a page with a single inline
# monospace token (a URL or a variable name in running prose) stays well
# below threshold. Token list excludes foundry-family prefixes like
# "nimbus" that also name proportional prose faces (Nimbus Roman/Sans).
MONO_FONT_TOKENS = (
    "mono",  # ubuntu mono, roboto mono, dejavu sans mono, pt mono, ...
    "nimbusmon",  # URW Nimbus Mono (mono family lacking the 'mono' substring)
    "courier",
    "consol",  # consolas / consola
    "menlo",
    "monaco",
    "inconsolata",
    "typewriter",
    "fira code",
    "firacode",
    "source code",
    "sourcecode",
    "jetbrains",
    "lucida console",
    "andale",
)

DEFAULT_MONO_RATIO_THRESHOLD = 0.10


def _mono_ratio_threshold() -> float:
    raw = os.environ.get("VLM_MONO_RATIO_THRESHOLD", "").strip()
    if not raw:
        return DEFAULT_MONO_RATIO_THRESHOLD
    try:
        val = float(raw)
        return val if val >= 0 else DEFAULT_MONO_RATIO_THRESHOLD
    except ValueError:
        return DEFAULT_MONO_RATIO_THRESHOLD


def page_mono_char_ratio(page: "fitz.Page") -> float:
    """Fraction of glyphs on the page set in a monospace font.

    Char-weighted (not span-weighted) so a single long mono run counts more than
    many short proportional spans. Returns 0.0 on empty or unreadable pages. The
    object-independent code signal shared by ``HybridEngine`` and
    ``MineruQwenHybridEngine``.
    """
    try:
        text_dict = page.get_text("dict")
    except Exception:  # pragma: no cover - defensive: PyMuPDF API drift
        return 0.0
    total = 0
    mono = 0
    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                n = len(span.get("text", ""))
                total += n
                font = span.get("font", "").lower()
                if any(token in font for token in MONO_FONT_TOKENS):
                    mono += n
    return mono / total if total else 0.0


# A real code BLOCK is a contiguous run of monospace-dominant lines. Page-average
# mono ratio (above) can fall below the routing threshold when a genuine code
# block is diluted by surrounding prose (Fluent Python p111: a nested for/if
# block at page-average 0.096), so MinerU-1.2B gets the page and can COLLAPSE the
# block. This run-based signal recovers exactly those pages while ignoring
# scattered inline monospace (method-name lists, a URL) which never forms a run.
DEFAULT_CODE_BLOCK_MIN_LINES = 4
DEFAULT_CODE_BLOCK_LINE_MONO = 0.6


def page_has_code_block(
    page: "fitz.Page",
    min_lines: int = DEFAULT_CODE_BLOCK_MIN_LINES,
    line_mono_floor: float = DEFAULT_CODE_BLOCK_LINE_MONO,
) -> bool:
    """True if the page has a contiguous run of >= ``min_lines`` lines that are
    each predominantly (>= ``line_mono_floor``) monospace — a code block that the
    page-average mono ratio can miss when diluted by prose."""
    try:
        text_dict = page.get_text("dict")
    except Exception:  # pragma: no cover - defensive: PyMuPDF API drift
        return False
    run = 0
    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            tot = sum(len(s.get("text", "")) for s in spans)
            if tot == 0:
                continue
            mono = sum(
                len(s.get("text", ""))
                for s in spans
                if any(token in s.get("font", "").lower() for token in MONO_FONT_TOKENS)
            )
            if mono / tot >= line_mono_floor:
                run += 1
                if run >= min_lines:
                    return True
            else:
                run = 0
    return False


def page_has_table(page: "fitz.Page") -> bool:
    """True if PyMuPDF detects a table on the page. Used to keep table-bearing
    pages on MinerU (its strength) even when a code block is also present — Qwen
    empties dense tables, so a code block is never traded for a table."""
    try:
        return len(page.find_tables().tables) > 0
    except Exception:  # pragma: no cover - defensive: PyMuPDF API drift
        return False


class HybridEngine:
    """Cost-optimizer router fulfilling ``extract(str) -> UniversalDocument``."""

    def __init__(
        self,
        vlm_engine: Optional[VlmNativeEngine] = None,
        docling_engine: Optional[DoclingFastEngine] = None,
        drawings_threshold: Optional[int] = None,
        mono_ratio_threshold: Optional[float] = None,
    ) -> None:
        self.vlm_engine = vlm_engine or VlmNativeEngine()
        self.docling_engine = docling_engine or DoclingFastEngine()
        self.drawings_threshold = (
            drawings_threshold if drawings_threshold is not None else _drawings_threshold()
        )
        self.mono_ratio_threshold = (
            mono_ratio_threshold if mono_ratio_threshold is not None else _mono_ratio_threshold()
        )
        # Populated by extract(); list of (page_number, "vlm"|"docling", reason).
        self.last_routing_decisions: List[Tuple[int, str, str]] = []

    # ------------------------------------------------------------------
    # Pre-flight
    # ------------------------------------------------------------------

    def _classify_page(self, page: "fitz.Page") -> Tuple[str, str]:
        """Return ``(engine_choice, reason)`` for a single page."""
        n_images = len(page.get_images())
        try:
            n_tables = len(page.find_tables().tables)
        except Exception:  # pragma: no cover - defensive: PyMuPDF API drift
            n_tables = 0
        n_drawings = len(page.get_drawings())

        if n_tables > 0:
            return "vlm", f"tables={n_tables}"
        if n_images > 0:
            return "vlm", f"images={n_images}"
        if n_drawings > self.drawings_threshold:
            return "vlm", f"drawings={n_drawings}>{self.drawings_threshold}"
        # Object-independent code signal: a pure-text page dense with
        # monospace characters is a code block that Docling would mangle.
        mono_ratio = self._mono_char_ratio(page)
        if mono_ratio >= self.mono_ratio_threshold:
            return "vlm", (f"code (mono_ratio={mono_ratio:.2f}>={self.mono_ratio_threshold})")
        return "docling", (
            f"prose (images=0, tables=0, drawings={n_drawings}"
            f"<={self.drawings_threshold}, mono_ratio={mono_ratio:.2f}"
            f"<{self.mono_ratio_threshold})"
        )

    def _mono_char_ratio(self, page: "fitz.Page") -> float:
        return page_mono_char_ratio(page)

    # ------------------------------------------------------------------
    # Public contract
    # ------------------------------------------------------------------

    def extract(self, file_path: str) -> UniversalDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        doc = fitz.open(str(path))
        try:
            decisions: List[Tuple[int, str, str]] = []
            vlm_page_indices: List[int] = []
            docling_page_indices: List[int] = []
            for i in range(doc.page_count):
                page_number = i + 1
                choice, reason = self._classify_page(doc[i])
                decisions.append((page_number, choice, reason))
                if choice == "vlm":
                    vlm_page_indices.append(i)
                else:
                    docling_page_indices.append(i)
            self.last_routing_decisions = decisions

            pages_by_number: Dict[int, UniversalPage] = {}

            # VLM subset runs FIRST. Docling's TableFormer pulls in
            # torch + multiprocessing workers; doing those first leaves
            # the process in a state where subsequent outbound HTTP
            # requests to the omlx-server are dropped mid-stream
            # ("Response ended prematurely"). Issuing the network calls
            # against a clean process and only then loading Docling
            # avoids that interaction.
            fallback_indices: List[int] = []
            if vlm_page_indices:
                for i in vlm_page_indices:
                    page_number = i + 1
                    try:
                        pages_by_number[page_number] = extract_page_vlm(
                            self.vlm_engine, doc[i], page_number
                        )
                    except VlmInfraError:
                        # CIRCUIT BREAKER. An infrastructure/transport failure
                        # (node offline, connection refused, connect/read
                        # timeout, gateway 502/503/504) is NOT a per-page
                        # quality problem. Demoting to Docling here would
                        # silently fabricate a CPU-extracted page that
                        # masquerades as a VLM baseline and corrupt the whole
                        # run. Propagate so the batch HALTS instead of
                        # degrading. Semantic failures still fall back below.
                        logger.error(
                            "VLM INFRASTRUCTURE FAILURE on page %d - circuit "
                            "breaker tripped; halting (NO docling fallback for "
                            "transport/endpoint outages)",
                            page_number,
                        )
                        raise
                    except Exception as exc:
                        # Single-page *semantic* VLM failures (empty content,
                        # malformed JSON, non-retryable 4xx) must not kill the
                        # whole document. Fall back to Docling for this
                        # page and record the demotion in the decision log.
                        logger.warning(
                            "VLM failed on page %d (%s); falling back to docling",
                            page_number,
                            exc,
                        )
                        fallback_indices.append(i)
                        for idx, (pn, _, _) in enumerate(decisions):
                            if pn == page_number:
                                decisions[idx] = (
                                    pn,
                                    "docling_fallback",
                                    f"vlm_failed: {type(exc).__name__}: {exc}",
                                )
                                break
                self.last_routing_decisions = decisions

            # Docling subset — covers planned-prose pages plus any VLM
            # fallbacks that demoted to Docling above.
            all_docling_indices = sorted(set(docling_page_indices + fallback_indices))
            if all_docling_indices:
                docling_doc = self.docling_engine.extract(str(path))
                wanted = {i + 1 for i in all_docling_indices}
                for page in docling_doc.pages:
                    if page.page_number in wanted:
                        pages_by_number[page.page_number] = page
        finally:
            doc.close()

        ordered_pages = [pages_by_number[n] for n in sorted(pages_by_number.keys())]
        metadata = DocumentMetadata(
            page_count=len(ordered_pages),
            file_size_bytes=path.stat().st_size,
            has_text_layer=any(p.text_elements for p in ordered_pages),
            has_images=any(p.image_elements for p in ordered_pages),
        )
        return create_document(
            file_path=path,
            file_type="pdf",
            pages=ordered_pages,
            metadata=metadata,
        )


class MineruQwenHybridEngine:
    """Per-page hybrid: Qwen VLM for code-dense pages, MinerU for everything else.

    MinerU2.5 is the strong default (tables, layout, scanned OCR) but its 1.2B
    recognizer mangles DENSE code indentation — measured R3 fidelity 0.44 on AIOS
    vs Qwen3-VL-8B's 1.00 on the same listings (live F5 validation, 2026-06-06).
    This engine routes ONLY code-dense pages (monospace-char ratio >= threshold)
    to the Qwen VLM and keeps every other page on MinerU, so a code-heavy academic
    doc gets clean, indentation-preserving code AND MinerU-quality tables/prose.

    Page classification is object-independent (monospace ratio), the same signal
    HybridEngine uses; AIOS measured a clean split (non-code pages <= 0.02, code
    pages 0.19-0.98), so the shared 0.10 floor separates them with wide margin.
    Transport/endpoint failures on the Qwen subset trip the circuit breaker
    (raise, no silent MinerU fallback); single-page SEMANTIC VLM failures demote
    that one page to MinerU and are recorded in the decision log.
    """

    def __init__(
        self,
        mineru_engine: Optional[MineruNativeEngine] = None,
        vlm_engine: Optional[VlmNativeEngine] = None,
        mono_ratio_threshold: Optional[float] = None,
    ) -> None:
        self.mineru_engine = mineru_engine or MineruNativeEngine()
        self.vlm_engine = vlm_engine or VlmNativeEngine()
        self.mono_ratio_threshold = (
            mono_ratio_threshold if mono_ratio_threshold is not None else _mono_ratio_threshold()
        )
        # Populated by extract(); (page_number, "qwen_code"|"mineru"|..., reason).
        self.last_routing_decisions: List[Tuple[int, str, str]] = []

    def extract(self, file_path: str) -> UniversalDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        doc = fitz.open(str(path))
        try:
            decisions: List[Tuple[int, str, str]] = []
            code_indices: List[int] = []
            mineru_indices: List[int] = []
            for i in range(doc.page_count):
                ratio = page_mono_char_ratio(doc[i])
                if ratio >= self.mono_ratio_threshold:
                    code_indices.append(i)
                    decisions.append((i + 1, "qwen_code", f"mono_ratio={ratio:.2f}"))
                elif page_has_code_block(doc[i]) and not page_has_table(doc[i]):
                    # Sub-threshold page-average but a real contiguous code block
                    # diluted by prose (a nested suite MinerU-1.2B can collapse).
                    # Route to Qwen to preserve indentation. Table-guarded: a page
                    # with a table stays on MinerU (Qwen empties dense tables; the
                    # block's R3 risk is caught by the gate metric's collapsed-
                    # suite detection).
                    code_indices.append(i)
                    decisions.append(
                        (i + 1, "qwen_code_block", f"mono_ratio={ratio:.2f} code_block")
                    )
                else:
                    mineru_indices.append(i)
                    decisions.append((i + 1, "mineru", f"mono_ratio={ratio:.2f}"))
            self.last_routing_decisions = decisions

            pages_by_number: Dict[int, UniversalPage] = {}

            # Qwen subset runs FIRST against a clean process (same ordering
            # rationale as HybridEngine: heavy native runtimes loaded later must
            # not poison outbound HTTP). A semantic failure demotes that page to
            # MinerU; a transport failure trips the circuit breaker.
            fallback_indices: List[int] = []
            if code_indices:
                for i in code_indices:
                    page_number = i + 1
                    try:
                        pages_by_number[page_number] = extract_page_vlm(
                            self.vlm_engine, doc[i], page_number
                        )
                    except VlmInfraError:
                        logger.error(
                            "VLM INFRASTRUCTURE FAILURE on code page %d - circuit "
                            "breaker tripped; halting (no MinerU fallback for "
                            "transport/endpoint outages)",
                            page_number,
                        )
                        raise
                    except Exception as exc:
                        logger.warning(
                            "Qwen failed on code page %d (%s); falling back to MinerU",
                            page_number,
                            exc,
                        )
                        fallback_indices.append(i)
                        for idx, (pn, _, _) in enumerate(decisions):
                            if pn == page_number:
                                decisions[idx] = (
                                    pn,
                                    "mineru_fallback",
                                    f"qwen_failed: {type(exc).__name__}: {exc}",
                                )
                                break
                self.last_routing_decisions = decisions

            # MinerU subset — planned-prose/table pages plus any Qwen fallbacks.
            for i in sorted(set(mineru_indices + fallback_indices)):
                page_number = i + 1
                pages_by_number[page_number] = extract_page_mineru(
                    self.mineru_engine, doc[i], page_number
                )
        finally:
            doc.close()

        ordered_pages = [pages_by_number[n] for n in sorted(pages_by_number.keys())]
        metadata = DocumentMetadata(
            page_count=len(ordered_pages),
            file_size_bytes=path.stat().st_size,
            has_text_layer=any(p.text_elements for p in ordered_pages),
            has_images=any(p.image_elements for p in ordered_pages),
        )
        return create_document(
            file_path=path,
            file_type="pdf",
            pages=ordered_pages,
            metadata=metadata,
        )


def extract(file_path: str) -> UniversalDocument:
    """Module-level convenience: ``extract(file_path) -> UniversalDocument``."""
    return HybridEngine().extract(file_path)
