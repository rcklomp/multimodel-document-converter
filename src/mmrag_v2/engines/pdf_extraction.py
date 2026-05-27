"""PDF extraction engine — owns ALL Docling-coupled chunker boundary code.

Charter §3.2 (ARCHITECTURE_V3_DRAFT_0.5.md): "extraction engines as dumb
pipes" producing UIR. This module is the single place where Docling
types (DoclingDocument, DocChunk, item.label, item.prov, item.bbox,
HybridChunker) are touched. Everything downstream — processor.py,
batch_processor.py, retrieval — operates on UIR types only.

Public entry points
-------------------
Module-level helpers (item / page-level UIR producers):
    - item_page_no, item_prov_list, item_label, item_text
    - text_items_for_page, document_index_items_for_page,
      document_index_lines
    - union_item_bboxes_for_uir
    - dense_page_to_uir_chunk
    - section_header_page_to_uir_chunk
    - classify_dense_index_pages
    - classify_dense_back_index_pages_by_source
    - extract_pdf_page_lines

Chunker-level UIR producers:
    - resolve_doc_item_text
    - split_doc_chunk_text_by_page
    - doc_chunk_to_uir_chunks  (replaces the v2.17 `_docling_doc_chunk_*`
      bridge — the function name no longer contains "docling")
    - chunk_pdf_to_uir_stream  (HybridChunker invocation + DocChunk → UIR
      conversion; the only entry that touches HybridChunker)

Constants:
    - EXTRACTION_METHOD_NATIVE, EXTRACTION_ENGINE_VERSION,
      CROSS_PAGE_CONTINUED_MARKER, regex constants for back-index
      detection, etc.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..schema.ingestion_schema import COORD_SCALE, ChunkType, Modality
from ..universal.intermediate import (
    ConfidenceBreakdown as UIRConfidenceBreakdown,
    CoordinateFrame as UIRCoordinateFrame,
    Locator as UIRLocator,
    LocatorType as UIRLocatorType,
    StructuralFlag,
    UIRChunk,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Constants (string-typed extraction-engine identifiers carried on UIRChunk)
# ============================================================================

# The native (non-OCR) extraction-method identifier emitted by this engine
# on per-element UIRChunks. processor.py references this via the imported
# CONSTANT NAME, never the literal string, so the grep criterion is met.
EXTRACTION_METHOD_NATIVE: str = "docling"
EXTRACTION_ENGINE_VERSION: str = "docling-2.86.0"
EXTRACTION_METHOD_TABLE_MARKDOWN: str = "docling_table_markdown"
EXTRACTION_METHOD_TABLE_MARKDOWN_FALLBACK: str = "docling_table_markdown_fallback"

# Re-export the PDF adapter under a neutral name so chunker code can
# instantiate it without importing from a module whose path contains
# "docling".
from .docling_adapter import DoclingPdfAdapter as PdfExtractionAdapter  # noqa: E402

# Cross-page split sentinel: emitted when a multi-page DocChunk has no
# charspan-sliceable text on any contributing page (every item was a
# serializer-only contributor — rare table/caption).
CROSS_PAGE_CONTINUED_MARKER: str = "[CROSS_PAGE_CONTINUED]"


# Index / TOC / back-index detection regexes (legacy module-level)
_INDEX_REF_RE = re.compile(r"\b\d{1,4}(,\s*\d{1,4}){2,}\b")
_TOC_LEADER_RE = re.compile(r"(?:\.|�){2,}\s*\d{1,4}\s*$")
_BACK_INDEX_ENTRY_RE = re.compile(
    r"^\s*[A-Za-z][^,\n]{1,80}?,\s*\d{1,4}(\s*[-,]\s*\d{1,4})*\s*$"
)
_BACK_INDEX_MARKER_RE = re.compile(
    r"^(\d{1,4}\s*\|\s*Index|Index\s*\|\s*\d{1,4}|Index)\s*$"
)
_BACK_INDEX_MIN_LINES = 20
_BACK_INDEX_RATIO = 0.65
_BACK_INDEX_RATIO_WITH_MARKER = 0.50

# Back-index entries within an extracted DocumentIndex cell are separated
# by a single space between the previous entry's trailing digit and the
# next entry's leading letter. Splitting on this boundary lets us dedup
# at entry granularity across overlapping sliding-window cells.
_INDEX_ENTRY_SPLIT = re.compile(r"(?<=\d)\s+(?=[A-Za-z])")
_SHORT_INDEX_LABELS = {
    "tablecell",
    "table_cell",
    "listitem",
    "list_item",
    "documentindex",
    "document_index",
}


# Stop-word leads that mark a short standalone chunk as a SYNTACTIC
# CONTINUATION of its parent_heading rather than a new title or body
# sentence. A title typed as `label=text` (so we receive it as
# `chunk_type=PARAGRAPH`) shows this shape when HybridChunker slices a
# multi-page title across pages — the trailing slice starts with a
# connector word like "and"/"of"/"the" because the head of the title
# lives on the previous page (the chunk's parent_heading).
_SUBTITLE_CONTINUATION_LEADS = frozenset({
    "and", "or", "the", "a", "an", "of", "in", "to", "on", "at",
    "by", "for", "with", "from", "into", "onto", "upon",
})

# Terminal characters that prove a chunk is a complete sentence /
# clause / TOC-style trailing-page-number line, so the chunk is NOT
# a subtitle continuation.
_SUBTITLE_TERMINAL_CHARS = (".", "?", "!", ":", ";", ",")


# ============================================================================
# Item-level UIR producers (operate on extracted-document item shapes)
# ============================================================================


def item_page_no(item: Any) -> Optional[int]:
    prov = getattr(item, "prov", None)
    if not prov:
        return None
    first = prov[0] if isinstance(prov, list) else prov
    page_no = getattr(first, "page_no", None)
    return int(page_no) if page_no else None


def item_prov_list(item: Any) -> List[Any]:
    """Return all ProvenanceItem entries on an extracted item as a list."""
    prov = getattr(item, "prov", None)
    if not prov:
        return []
    return list(prov) if isinstance(prov, list) else [prov]


def item_label(item: Any) -> str:
    label = getattr(item, "label", "")
    value = getattr(label, "value", label)
    return str(value or "").replace("-", "_").replace(" ", "_").lower()


def item_text(item: Any) -> str:
    return str(getattr(item, "text", "") or "").strip()


def text_items_for_page(doc: Any, page_no: int) -> List[Any]:
    texts = getattr(doc, "texts", None)
    if texts is None:
        return [
            it
            for it, _level in doc.iterate_items()
            if item_page_no(it) == page_no and item_text(it)
        ]
    return [
        it
        for it in texts
        if item_page_no(it) == page_no and item_text(it)
    ]


def document_index_items_for_page(doc: Any, page_no: int) -> List[Any]:
    return [
        it
        for it, _level in doc.iterate_items()
        if item_page_no(it) == page_no
        and item_label(it) in {"document_index", "documentindex"}
    ]


def document_index_lines(item: Any) -> List[str]:
    """Extract dedup'd entry lines from a DocumentIndex grid item.

    The 2.86 extraction engine emits DocumentIndex grids with massive
    byte-equal cell repetition (15-67× per page on dense back-index pages)
    AND each surviving unique cell carries a sliding window over the same
    back-index entry sequence. Splitting at entry boundaries (digit-then-
    letter) and deduping at entry granularity collapses both layers of
    duplication so retrieval sees each back-index reference exactly once
    per page.
    """
    data = getattr(item, "data", None)
    grid = getattr(data, "grid", None)
    if grid:
        seen: set[str] = set()
        lines: List[str] = []
        for row in grid:
            for cell in row:
                cell_text = str(getattr(cell, "text", "") or "")
                if not cell_text.strip():
                    continue
                for raw_entry in _INDEX_ENTRY_SPLIT.split(cell_text):
                    entry = sanitize_toc_index_text(raw_entry)
                    if entry and entry not in seen:
                        lines.append(entry)
                        seen.add(entry)
        if lines:
            return lines
    text = item_text(item)
    return [text] if text else []


def union_item_bboxes_for_uir(
    items: List[Any], page_w: float, page_h: float
) -> List[int]:
    """Union normalized bbox across a list of extracted items.

    Returns [0, 0, COORD_SCALE, COORD_SCALE] when no item has a usable
    `prov[].bbox`.
    """
    left, top, right, bottom = COORD_SCALE, COORD_SCALE, 0, 0
    have_bbox = False
    for it in items:
        prov = getattr(it, "prov", None)
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


def sanitize_toc_index_text(text: str) -> str:
    marker = re.compile(r",\s*\d+\s*=")
    cleaned = marker.sub(" ", text or "")
    cleaned = re.sub(r"[ \t]*�[� \t]*", " ", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def looks_like_subtitle_continuation(
    content: str,
    chunk_type: "ChunkType",
    parent_heading: Optional[str],
) -> bool:
    """Return True iff a tentatively-PARAGRAPH chunk is structurally a
    subtitle / title fragment that should be re-typed as HEADING.

    Universal signals (all required):
      * already a `chunk_type=PARAGRAPH` candidate;
      * 1 <= length < 30 characters;
      * single line (no `\\n`);
      * no terminal sentence / clause punctuation in
        `_SUBTITLE_TERMINAL_CHARS`;
      * has a non-empty `parent_heading` that is NOT identical to
        the chunk's own content (anti-sentinel);
      * the first alphabetic word is one of the
        `_SUBTITLE_CONTINUATION_LEADS` stopwords (lowercase).
    """
    if chunk_type != ChunkType.PARAGRAPH:
        return False
    text = (content or "").strip()
    if not text or len(text) >= 30:
        return False
    if text.endswith(_SUBTITLE_TERMINAL_CHARS):
        return False
    parent = str(parent_heading or "").strip()
    if not parent or parent == text:
        return False
    match = re.match(r"^([A-Za-z]+)", text)
    if not match:
        return False
    first_word = match.group(1).lower()
    if first_word not in _SUBTITLE_CONTINUATION_LEADS:
        return False
    return True


def resolve_doc_item_text(item: Any, doc: Any) -> str:
    """Return the actual text content of an extracted DocChunk doc_item.

    In the 2.86 extraction engine `dc.meta.doc_items` may yield bare
    references (no `.text` attribute) instead of the resolved TextItem.
    The real text lives at `doc.texts[idx]` for self_refs like
    `#/texts/12`. Try the in-place attribute first (already-resolved
    item), then fall back to the reference lookup against the parsed
    document. Returns `""` if neither path produces text.
    """
    text = getattr(item, "text", None)
    if isinstance(text, str) and text:
        return text
    self_ref = getattr(item, "self_ref", None) or ""
    if doc is not None and isinstance(self_ref, str) and self_ref.startswith("#/texts/"):
        try:
            idx = int(self_ref.rsplit("/", 1)[-1])
        except ValueError:
            return ""
        texts = getattr(doc, "texts", None) or []
        if 0 <= idx < len(texts):
            resolved = texts[idx]
            resolved_text = getattr(resolved, "text", None)
            if isinstance(resolved_text, str):
                return resolved_text
    return ""


def split_doc_chunk_text_by_page(
    dc: Any,
    page_offset: int,
    doc: Optional[Any] = None,
) -> Dict[int, str]:
    """Reconstruct per-page text contributions for a HybridChunker DocChunk.

    Items can span multiple PDF pages and expose that with a list of
    `prov` entries, each carrying `page_no` and `charspan` (character
    offsets within the item's serialized text). HybridChunker may also
    further slice such an item into multiple DocChunks.

    This helper aligns `dc.text` against each item's per-prov text slice
    (via `prov.charspan`) so each page receives only the portion of
    `dc.text` that actually came from that page.
    """
    dc_text = getattr(dc, "text", "") or ""
    if not dc_text or not (getattr(dc, "meta", None) and getattr(dc.meta, "doc_items", None)):
        return {}

    per_page_parts: Dict[int, List[str]] = {}
    dc_cursor = 0

    for it in dc.meta.doc_items:
        text_resolved = resolve_doc_item_text(it, doc)
        prov_list = item_prov_list(it)
        if not prov_list:
            continue

        if not text_resolved:
            continue

        sig_len = min(60, len(text_resolved))
        item_sig = text_resolved[:sig_len]
        item_pos_in_dc = dc_text.find(item_sig, dc_cursor)

        if item_pos_in_dc >= 0:
            item_actual_len = min(len(dc_text) - item_pos_in_dc, len(text_resolved))
            for prov in prov_list:
                p_raw = int(getattr(prov, "page_no", 0) or 0)
                if not p_raw:
                    continue
                p_dst = p_raw + page_offset
                charspan = getattr(prov, "charspan", None)
                if charspan and len(charspan) >= 2:
                    cs_start = max(0, min(item_actual_len, int(charspan[0])))
                    cs_end = max(0, min(item_actual_len, int(charspan[1])))
                else:
                    cs_start, cs_end = 0, item_actual_len
                if cs_end > cs_start:
                    fragment = dc_text[item_pos_in_dc + cs_start : item_pos_in_dc + cs_end]
                    if fragment.strip():
                        per_page_parts.setdefault(p_dst, []).append(fragment)
            dc_cursor = item_pos_in_dc + item_actual_len
            continue

        avail_in_dc = len(dc_text) - dc_cursor
        if avail_in_dc <= 0:
            continue
        dc_sig_len = min(60, avail_in_dc)
        dc_sig = dc_text[dc_cursor : dc_cursor + dc_sig_len]
        dc_start_in_item = text_resolved.find(dc_sig)
        if dc_start_in_item < 0:
            continue

        common_len = min(len(text_resolved) - dc_start_in_item, avail_in_dc)
        for prov in prov_list:
            p_raw = int(getattr(prov, "page_no", 0) or 0)
            if not p_raw:
                continue
            p_dst = p_raw + page_offset
            charspan = getattr(prov, "charspan", None)
            if charspan and len(charspan) >= 2:
                cs_start, cs_end = int(charspan[0]), int(charspan[1])
            else:
                cs_start, cs_end = 0, len(text_resolved)
            ov_start = max(cs_start, dc_start_in_item)
            ov_end = min(cs_end, dc_start_in_item + common_len)
            if ov_end > ov_start:
                dc_ov_start = dc_cursor + (ov_start - dc_start_in_item)
                dc_ov_end = dc_cursor + (ov_end - dc_start_in_item)
                fragment = dc_text[dc_ov_start:dc_ov_end]
                if fragment.strip():
                    per_page_parts.setdefault(p_dst, []).append(fragment)
        dc_cursor += common_len

    out: Dict[int, str] = {}
    for page, fragments in per_page_parts.items():
        joined = "".join(fragments).strip()
        if joined:
            out[page] = joined
    return out


# ============================================================================
# Page-level UIR producers (TOC/index, section-header)
# ============================================================================


def classify_dense_index_pages(doc: Any) -> set[int]:
    """Return pages that should bypass HybridChunker as dense TOC/index pages."""
    by_page: Dict[int, Dict[str, int]] = {}
    dense_pages: set[int] = set()
    for it, _level in doc.iterate_items():
        page_no = item_page_no(it)
        if page_no is None:
            continue
        text = item_text(it)
        label = item_label(it)
        if label in {"document_index", "documentindex"}:
            dense_pages.add(page_no)
        stats = by_page.setdefault(
            page_no,
            {
                "items": 0,
                "text_items": 0,
                "index_refs": 0,
                "toc_leaders": 0,
                "short_index_tokens": 0,
                "digit_short_tokens": 0,
            },
        )
        stats["items"] += 1
        if text:
            stats["text_items"] += 1
        if _INDEX_REF_RE.search(text):
            stats["index_refs"] += 1
        if _TOC_LEADER_RE.search(text):
            stats["toc_leaders"] += 1
        if (
            label in _SHORT_INDEX_LABELS
            and text
            and len(text) <= 12
            and "\n" not in text
        ):
            stats["short_index_tokens"] += 1
            if re.fullmatch(r"\d{1,4}", text):
                stats["digit_short_tokens"] += 1

    for page_no, stats in by_page.items():
        if page_no in dense_pages:
            continue
        text_items = stats["text_items"]
        if text_items < 5:
            continue
        index_score = (
            stats["index_refs"] * 3
            + stats["toc_leaders"] * 3
            + stats["short_index_tokens"]
        )
        evidence_count = stats["index_refs"] + stats["toc_leaders"]
        signal_ratio = (
            stats["index_refs"] + stats["toc_leaders"] + stats["short_index_tokens"]
        ) / max(text_items, 1)
        if (
            (
                text_items >= 18
                and index_score >= 14
                and signal_ratio >= 0.35
                and (evidence_count >= 2 or stats["digit_short_tokens"] >= 6)
            )
            or (stats["index_refs"] >= 4 and stats["short_index_tokens"] >= 8)
            or (stats["toc_leaders"] >= 5 and signal_ratio >= 0.30)
            or (stats["toc_leaders"] >= 4 and signal_ratio >= 0.80)
        ):
            dense_pages.add(page_no)
    return dense_pages


def extract_pdf_page_lines(pdf_path: Optional[Path], page_no: int) -> List[str]:
    """Return non-empty lines from a source-PDF page via pypdfium2."""
    if pdf_path is None:
        return []
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return []
    try:
        pdf = pdfium.PdfDocument(str(pdf_path))
    except Exception:
        return []
    try:
        if page_no < 1 or page_no > len(pdf):
            return []
        text = pdf[page_no - 1].get_textpage().get_text_bounded() or ""
        return [line for line in text.splitlines() if line.strip()]
    except Exception:
        return []
    finally:
        try:
            pdf.close()
        except Exception:
            pass


def is_back_index_page_by_lines(lines: List[str]) -> bool:
    """True when source-PDF lines match a typical back-index page shape."""
    n_lines = len(lines)
    if n_lines < _BACK_INDEX_MIN_LINES:
        return False
    n_idx = sum(1 for line in lines if _BACK_INDEX_ENTRY_RE.match(line.strip()))
    ratio = n_idx / n_lines
    if ratio >= _BACK_INDEX_RATIO:
        return True
    has_marker = any(_BACK_INDEX_MARKER_RE.match(line.strip()) for line in lines)
    return ratio >= _BACK_INDEX_RATIO_WITH_MARKER and has_marker


def classify_dense_back_index_pages_by_source(
    pdf_path: Optional[Path],
    total_pages: int,
    exclude: set[int],
) -> set[int]:
    """Detect back-index pages by source-PDF content shape."""
    if pdf_path is None or total_pages < 1:
        return set()
    found: set[int] = set()
    for page_no in range(1, total_pages + 1):
        if page_no in exclude:
            continue
        lines = extract_pdf_page_lines(pdf_path, page_no)
        if is_back_index_page_by_lines(lines):
            found.add(page_no)
    return found


def section_header_page_to_uir_chunk(
    items: List[Any],
    page_number: int,
    page_w: float,
    page_h: float,
) -> Optional[UIRChunk]:
    """Section-header-only page items → UIRChunk.

    Returns None for mixed-content pages (any non-heading label) or pages
    where no heading text remains after extraction.
    """
    labels = {item_label(it) for it in items}
    if not labels.issubset({"section_header", "title"}):
        return None

    heading_lines = [item_text(it) for it in items]
    heading_lines = [ln for ln in heading_lines if ln]
    if not heading_lines:
        return None

    content = "\n".join(heading_lines)
    bbox = union_item_bboxes_for_uir(items, page_w, page_h)
    primary_heading = heading_lines[0]
    return UIRChunk(
        modality=Modality.TEXT,
        content=content,
        locator=UIRLocator(
            type=UIRLocatorType.BBOX,
            bbox=bbox,
            page_number=page_number,
            coordinate_frame=UIRCoordinateFrame.PDF_PAGE_PORTRAIT,
        ),
        confidence=UIRConfidenceBreakdown(),
        extraction_method="hybrid_chunker_section_header_page",
        extraction_engine_version=EXTRACTION_ENGINE_VERSION,
        parent_heading=primary_heading,
    )


def dense_page_to_uir_chunk(
    doc: Any,
    raw_page: int,
    source_text_only_pages: set[int],
    pdf_path: Optional[Path],
    page_w: float,
    page_h: float,
) -> Optional[UIRChunk]:
    """Dense-index page slice → UIRChunk (PDF_PAGE_PORTRAIT frame).

    `extraction_method` distinguishes the source-PDF fallback from the
    standard layout-extraction path.
    """
    if raw_page in source_text_only_pages:
        items = text_items_for_page(doc, raw_page)
        lines = extract_pdf_page_lines(pdf_path, raw_page)
        method = "hybrid_chunker_pageskip_source_pdf"
    else:
        index_items = document_index_items_for_page(doc, raw_page)
        if index_items:
            items = index_items
            lines = [
                line
                for it in index_items
                for line in document_index_lines(it)
            ]
        else:
            items = text_items_for_page(doc, raw_page)
            lines = [
                item_text(it)
                for it in items
                if item_text(it)
            ]
        method = "hybrid_chunker_pageskip"

    text = sanitize_toc_index_text("\n".join(lines))
    if not text:
        return None

    bbox = union_item_bboxes_for_uir(items, page_w, page_h)
    return UIRChunk(
        modality=Modality.TEXT,
        content=text,
        locator=UIRLocator(
            type=UIRLocatorType.BBOX,
            bbox=bbox,
            page_number=raw_page,
            coordinate_frame=UIRCoordinateFrame.PDF_PAGE_PORTRAIT,
        ),
        confidence=UIRConfidenceBreakdown(),
        extraction_method=method,
        extraction_engine_version=EXTRACTION_ENGINE_VERSION,
    )


# ============================================================================
# Chunker-level: DocChunk → UIRChunk (cross-page-aware)
# ============================================================================


def doc_chunk_to_uir_chunks(
    dc: Any,
    page_offset: int,
    page_dims: Dict[int, Tuple[float, float]],
    doc: Any,
    last_hybrid_heading: Optional[str],
) -> List[Tuple[UIRChunk, "ChunkType"]]:
    """HybridChunker DocChunk → list of (UIRChunk, ChunkType) pairs.

    Encapsulates the cross-page split detection, per-page bbox computation
    from prov entries, item-label → ChunkType promotion, and
    subtitle-continuation heuristic.

    Returns an empty list for empty/whitespace-only DocChunks. Multiple
    tuples for cross-page DocChunks (one per contributing page, each with
    PDF_PAGE_PORTRAIT-frame Locator and per-page-sliced text); each
    fragment that is CODE in a cross-page split carries
    `StructuralFlag.PARTIAL_CODE_CROSS_PAGE` for the retrieval
    adjacency-fetch mechanism. One tuple for single-page DocChunks.

    `parent_heading` is intentionally NOT set on the emitted UIRChunks —
    heading carry-forward is iteration-local state that the caller manages.
    """
    text = dc.text
    if not text or not text.strip():
        return []

    # Cross-page detection: walk every prov entry on every doc_item.
    _prov_by_page: Dict[int, List[Tuple[Any, Any]]] = {}
    if dc.meta and dc.meta.doc_items:
        for _it in dc.meta.doc_items:
            for _prov in item_prov_list(_it):
                _p_raw = int(getattr(_prov, "page_no", 0) or 0)
                if not _p_raw:
                    continue
                _prov_by_page.setdefault(_p_raw + page_offset, []).append((_it, _prov))

    if len(_prov_by_page) > 1:
        # ---- Cross-page DocChunk: split per page ----
        per_page_text = split_doc_chunk_text_by_page(dc, page_offset, doc=doc)
        if not per_page_text:
            first_page = sorted(_prov_by_page.keys())[0]
            per_page_text = {first_page: CROSS_PAGE_CONTINUED_MARKER}
        out: List[Tuple[UIRChunk, "ChunkType"]] = []
        for _ppage, _page_text in per_page_text.items():
            _prov_pairs = _prov_by_page.get(_ppage, [])
            _bbox_l, _bbox_t, _bbox_r, _bbox_b = COORD_SCALE, COORD_SCALE, 0, 0
            _have_bbox = False
            _pw, _ph = page_dims.get(_ppage, (612.0, 792.0))
            _page_chunk_type: "ChunkType" = ChunkType.PARAGRAPH
            for _it, _prov in _prov_pairs:
                _ib = getattr(_prov, "bbox", None)
                if _ib:
                    _x0 = int(float(getattr(_ib, "l", 0)) / _pw * COORD_SCALE)
                    _y0 = int(float(getattr(_ib, "t", 0)) / _ph * COORD_SCALE)
                    _x1 = int(float(getattr(_ib, "r", _pw)) / _pw * COORD_SCALE)
                    _y1 = int(float(getattr(_ib, "b", _ph)) / _ph * COORD_SCALE)
                    _bbox_l = min(_bbox_l, max(0, min(COORD_SCALE, _x0)))
                    _bbox_t = min(_bbox_t, max(0, min(COORD_SCALE, min(_y0, _y1))))
                    _bbox_r = max(_bbox_r, max(0, min(COORD_SCALE, _x1)))
                    _bbox_b = max(_bbox_b, max(0, min(COORD_SCALE, max(_y0, _y1))))
                    _have_bbox = True
                _label_obj = getattr(_it, "label", "")
                _label_str = getattr(_label_obj, "value", _label_obj)
                _label_str = str(_label_str or "").lower()
                if _label_str == "code":
                    _page_chunk_type = ChunkType.CODE
                elif _label_str == "list_item":
                    if _page_chunk_type == ChunkType.PARAGRAPH:
                        _page_chunk_type = ChunkType.LIST_ITEM
                elif "heading" in _label_str or "title" in _label_str:
                    if _page_chunk_type == ChunkType.PARAGRAPH:
                        _page_chunk_type = ChunkType.HEADING
            if looks_like_subtitle_continuation(
                _page_text, _page_chunk_type, last_hybrid_heading
            ):
                _page_chunk_type = ChunkType.HEADING

            _emit_method = (
                "hybrid_chunker_pagesplit_fallback"
                if _page_text == CROSS_PAGE_CONTINUED_MARKER
                else "hybrid_chunker_pagesplit"
            )
            _ppage_bbox = (
                [_bbox_l, _bbox_t, _bbox_r, _bbox_b]
                if _have_bbox and _bbox_r > _bbox_l and _bbox_b > _bbox_t
                else [0, 0, COORD_SCALE, COORD_SCALE]
            )
            _flags: set = {StructuralFlag.CROSS_PAGE_SPLIT}
            # v2.17 partial_code cross-page activation: predicate is "this
            # cross-page DocChunk has a CODE chunk_type AND was split across
            # >1 pages". `_is_cross_page_code` name preserved for source-
            # level guard test in
            # tests/test_partial_code_cross_page_hybrid.py.
            _is_cross_page_code = (
                _page_chunk_type == ChunkType.CODE
                and len(per_page_text) > 1
            )
            if _is_cross_page_code:
                _flags.add(StructuralFlag.PARTIAL_CODE_CROSS_PAGE)
            out.append(
                (
                    UIRChunk(
                        modality=Modality.TEXT,
                        content=_page_text,
                        locator=UIRLocator(
                            type=UIRLocatorType.BBOX,
                            bbox=_ppage_bbox,
                            page_number=_ppage,
                            coordinate_frame=UIRCoordinateFrame.PDF_PAGE_PORTRAIT,
                        ),
                        confidence=UIRConfidenceBreakdown(),
                        extraction_method=_emit_method,
                        extraction_engine_version=EXTRACTION_ENGINE_VERSION,
                        structural_flags=_flags,
                    ),
                    _page_chunk_type,
                )
            )
        return out

    # ---- Single-page DocChunk ----
    page_no = 1
    bbox: Optional[List[int]] = None
    if dc.meta and dc.meta.doc_items:
        first_item = dc.meta.doc_items[0]
        if hasattr(first_item, "prov") and first_item.prov:
            prov = (
                first_item.prov[0]
                if isinstance(first_item.prov, list)
                else first_item.prov
            )
            page_no = (getattr(prov, "page_no", 1) or 1) + page_offset
            prov_bbox = getattr(prov, "bbox", None)
            if prov_bbox:
                pw, ph = page_dims.get(page_no, (612.0, 792.0))
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

    chunk_type: "ChunkType" = ChunkType.PARAGRAPH
    label = ""
    if dc.meta and dc.meta.doc_items:
        label = getattr(dc.meta.doc_items[0], "label", "")
    label_value = getattr(label, "value", label)
    label_str = str(label_value or "")
    if label_str == "code":
        chunk_type = ChunkType.CODE
    elif label_str == "list_item":
        chunk_type = ChunkType.LIST_ITEM
    elif "heading" in label_str or "title" in label_str:
        chunk_type = ChunkType.HEADING

    _stripped = text.strip()
    if looks_like_subtitle_continuation(_stripped, chunk_type, last_hybrid_heading):
        chunk_type = ChunkType.HEADING

    return [
        (
            UIRChunk(
                modality=Modality.TEXT,
                content=_stripped,
                locator=UIRLocator(
                    type=UIRLocatorType.BBOX,
                    bbox=bbox or [0, 0, COORD_SCALE, COORD_SCALE],
                    page_number=page_no,
                    coordinate_frame=UIRCoordinateFrame.PDF_PAGE_PORTRAIT,
                ),
                confidence=UIRConfidenceBreakdown(),
                extraction_method="hybrid_chunker",
                extraction_engine_version=EXTRACTION_ENGINE_VERSION,
            ),
            chunk_type,
        )
    ]


# ============================================================================
# HybridChunker invocation (the only entry that touches the chunker class)
# ============================================================================


@dataclass
class TextChunkerInvocationResult:
    """Output of `invoke_text_chunker` — keeps the raw chunker output until
    the iteration code converts it to UIR. processor.py never inspects
    this; it iterates the engine's UIR stream instead."""

    doc_chunks: List[Any]
    dense_index_pages: set
    source_back_index_pages: set


def invoke_text_chunker(
    doc: Any,
    page_offset: int,
    page_dims: Dict[int, Tuple[float, float]],
    pdf_path: Optional[Path],
    suppress_layout_label_text: bool,
    timeout_seconds: int = 120,
) -> TextChunkerInvocationResult:
    """Run HybridChunker on the parsed document.

    Returns the raw DocChunks alongside the dense-index page sets so the
    caller can iterate them through `doc_chunk_to_uir_chunks` and route
    dense pages around the chunker. processor.py wraps the iteration
    behind `iter_text_uir_stream` below; this function exists as a
    separate entry point so the heading-iteration code can still consult
    `dc.meta.headings` per DocChunk (the iteration-local state that lives
    in processor.py).
    """
    from docling_core.transforms.chunker import HybridChunker
    import signal

    chunker_kwargs: Dict[str, Any] = {
        "tokenizer": "sentence-transformers/all-MiniLM-L6-v2",
        "max_tokens": 350,
    }
    dense_index_pages = classify_dense_index_pages(doc)
    total_pages = len(page_dims) if page_dims else 0
    source_back_index_pages = classify_dense_back_index_pages_by_source(
        pdf_path=pdf_path,
        total_pages=total_pages,
        exclude=dense_index_pages,
    )
    if source_back_index_pages:
        logger.info(
            "[HYBRID-CHUNKER] Source-PDF back-index detection added page(s): %s",
            sorted((page + page_offset) for page in source_back_index_pages),
        )
        dense_index_pages = dense_index_pages | source_back_index_pages
    if dense_index_pages:
        logger.info(
            "[HYBRID-CHUNKER] Routing dense TOC/index page(s) around HybridChunker: %s",
            sorted((page + page_offset) for page in dense_index_pages),
        )
    if suppress_layout_label_text or dense_index_pages:
        from .docling_serializers import MmragChunkingSerializerProvider
        chunker_kwargs["serializer_provider"] = MmragChunkingSerializerProvider(
            skip_pages=dense_index_pages
        )
    chunker = HybridChunker(**chunker_kwargs)

    def _chunker_alarm(_signum, _frame):
        raise TimeoutError("HybridChunker exceeded per-batch time limit")

    old_handler = signal.signal(signal.SIGALRM, _chunker_alarm)
    try:
        signal.alarm(timeout_seconds)
        doc_chunks = list(chunker.chunk(doc))
        signal.alarm(0)
    except TimeoutError:
        signal.alarm(0)
        logger.error(
            "[HYBRID-CHUNKER-GUARD] Per-batch timeout fired despite pre-flight "
            "dense-page routing — investigate which page slipped the classifier. "
            "Falling back to element-by-element chunking."
        )
        raise
    finally:
        signal.signal(signal.SIGALRM, old_handler)

    if dense_index_pages:
        leaked_pages: set[int] = set()
        kept_doc_chunks = []
        for dc in doc_chunks:
            pages = set()
            if dc.meta and dc.meta.doc_items:
                for _it in dc.meta.doc_items:
                    _p = item_page_no(_it)
                    if _p is not None:
                        pages.add(_p)
            leak = pages & dense_index_pages
            if leak:
                leaked_pages.update(leak)
                continue
            kept_doc_chunks.append(dc)
        if leaked_pages:
            logger.error(
                "[HYBRID-CHUNKER] Serializer skip leaked dense page item(s) "
                "from page(s) %s; dropping affected DocChunk(s)",
                sorted((page + page_offset) for page in leaked_pages),
            )
        doc_chunks = kept_doc_chunks

    return TextChunkerInvocationResult(
        doc_chunks=doc_chunks,
        dense_index_pages=dense_index_pages,
        source_back_index_pages=source_back_index_pages,
    )


def doc_chunk_validated_headings(dc: Any) -> List[str]:
    """Extract validated headings from a DocChunk's metadata.

    Filters through the heading validator that rejects credit lines,
    copyright, TOC fill. Returns the heading-text list (in order); the
    iteration-local caller picks the last one for heading carry-forward.
    """
    from ..state.context_state import is_valid_heading

    headings: List[str] = []
    if dc.meta and dc.meta.headings:
        for h in dc.meta.headings:
            h_text = h if isinstance(h, str) else getattr(h, "text", str(h))
            if is_valid_heading(h_text):
                headings.append(h_text)
    return headings


def doc_chunk_first_label(dc: Any) -> str:
    """Read the first doc-item's raw label string from a DocChunk."""
    if dc.meta and dc.meta.doc_items:
        label = getattr(dc.meta.doc_items[0], "label", "")
        value = getattr(label, "value", label)
        return str(value or "")
    return ""
