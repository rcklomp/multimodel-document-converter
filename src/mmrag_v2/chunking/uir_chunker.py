"""UIR-Native Chunker — operates STRICTLY on UniversalDocument (UIR).

Charter §3.2 (ARCHITECTURE_V3_DRAFT_0.5.md): "The chunker operates
solely on UniversalDocument/UIRChunk, entirely detached from Docling's
DoclingDocument layout classes."

This module NEVER imports Docling types (DoclingDocument, DocChunk,
HybridChunker). It iterates over UniversalDocument.pages and their
Element lists to produce UIRChunks ready for ingestion.

Design:
  - TEXT elements → sentence-boundary-aware grouping → UIRChunk(modality=TEXT)
  - IMAGE elements → UIRChunk(modality=IMAGE) with asset_ref
  - TABLE elements → UIRChunk(modality=TABLE)
  - Heading/code/list-item detection from Element.source_label + content heuristics
  - Cross-page continuity via continuation_group_id
  - BBoxes preserved from Element.bbox (already [0,1000] normalized)
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

from ..universal.intermediate import (
    BoundingBox,
    ConfidenceBreakdown,
    CoordinateFrame,
    Element,
    ElementType,
    ExtractionWarning,
    Locator,
    LocatorType,
    Modality,
    StructuralFlag,
    UIRChunk,
    UniversalDocument,
    UniversalPage,
)
from ..universal.table_markdown import ensure_table_separator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Token budget per chunk (roughly 1 token ≈ 4 chars for English)
DEFAULT_MAX_CHARS: int = 1400  # ~350 tokens
MIN_CHUNK_CHARS: int = 20
EXTRACTION_METHOD_UIR: str = "uir_native_chunker"
EXTRACTION_ENGINE_VERSION_DEFAULT: str = "mmrag-v3.0-uir"

# Source labels that indicate heading elements
_HEADING_LABELS: Set[str] = {
    "section_header",
    "section-header",
    "sectionheader",
    "title",
    "heading",
    "h1",
    "h2",
    "h3",
    "h4",
    "subtitle",
}

# Source labels that indicate code elements
_CODE_LABELS: Set[str] = {
    "code",
    "code_block",
    "codeblock",
    "pre",
    "listing",
    "programlisting",
}

# Source labels that indicate list items
_LIST_LABELS: Set[str] = {
    "list_item",
    "listitem",
    "list-item",
    "bullet",
    "enum",
    "item",
}

# Source labels that indicate table elements
_TABLE_LABELS: Set[str] = {
    "table",
    "table_cell",
    "tablecell",
    "tabular",
    "grid",
}

# Terminal sentence boundary characters
_SENTENCE_ENDS = {".", "!", "?", ":", ";", ")", '"', "\u201d"}

# Heading continuation leads (lowercase)
_HEADING_CONTINUATION_LEADS = frozenset(
    {
        "and",
        "or",
        "the",
        "a",
        "an",
        "of",
        "in",
        "to",
        "on",
        "at",
        "by",
        "for",
        "with",
        "from",
        "into",
        "onto",
        "upon",
    }
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def chunk_universal_document(
    universal_doc: UniversalDocument,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = MIN_CHUNK_CHARS,
    extraction_engine_version: str = EXTRACTION_ENGINE_VERSION_DEFAULT,
    profile_type: Optional[str] = None,
    toc_headings: Optional[Dict[Any, Any]] = None,
) -> List[UIRChunk]:
    """Chunk a UniversalDocument into UIRChunks — pure UIR-native.

    Iterates strictly over UniversalDocument.pages and their Element
    lists. NEVER imports or accepts Docling types (DoclingDocument,
    DocChunk, HybridChunker).

    Args:
        universal_doc: Fully populated UIR document.
        max_chars: Soft character budget per chunk.
        min_chars: Minimum chunk content length before merging.
        extraction_engine_version: Stamped on every emitted UIRChunk.
        profile_type: Optional profile hint (reserved for future tuning).
        toc_headings: Plain-data TOC map from
            ``BatchProcessor._extract_toc_headings`` (PyMuPDF bookmarks):
            ``{page_number: [breadcrumb...]}`` plus a ``"__heading_map__"``
            key mapping normalized heading title → breadcrumb. Threaded in
            as plain data (no Docling); drives cross-page heading
            carry-forward + breadcrumb_path (PLAN_V3.1 P2). ``None`` for
            docs with no bookmarks (heading assignment then relies only on
            in-page heading elements + carry-forward).

    Returns:
        List of UIRChunk objects ready for ingestion.
    """
    chunks: List[UIRChunk] = []
    reading_order: int = 0

    doc_title: Optional[str] = None
    meta = getattr(universal_doc, "metadata", None)
    if meta is not None:
        doc_title = getattr(meta, "title", None)

    for page in universal_doc.pages:
        page_chunks = _chunk_page(
            page,
            doc_id=universal_doc.doc_id,
            max_chars=max_chars,
            min_chars=min_chars,
            extraction_engine_version=extraction_engine_version,
            start_reading_order=reading_order,
        )
        chunks.extend(page_chunks)
        reading_order += len(page_chunks)

    # Crucible fix #2: drop within-page duplicate long-text chunks. Dense pages
    # can make the VLM loop and re-emit the same paragraph/heading several times
    # (Combat Aircraft: a heading 10x on one page) - QA's DUPLICATE_LONG_TEXT /
    # within_page_text_dupe_excess gate. No document legitimately repeats a
    # 120+ char paragraph verbatim on one page, so the first occurrence is kept
    # and later exact (whitespace-normalized) copies are dropped. Mirrors the
    # gate's modality (TEXT only), normalization, and threshold.
    chunks = _dedupe_within_page_text(chunks)

    # PLAN_V3.1 P2: UIR-native heading assignment. Per text chunk, in
    # reading order, set parent_heading by precedence:
    #   1. nearest preceding in-page heading element (already on the chunk);
    #   2. carry-forward of the last active heading from earlier pages;
    #   3. TOC entry whose page covers the chunk;
    # and build breadcrumb_path from the TOC hierarchy.
    _assign_headings(chunks, toc_headings, doc_title=doc_title)

    return chunks


# ---------------------------------------------------------------------------
# Heading assignment (TOC-driven, UIR-native — PLAN_V3.1 P2)
# ---------------------------------------------------------------------------


def _normalize_heading(text: str) -> str:
    """Collapse whitespace + NBSP so TOC titles match in-page headings."""
    return re.sub(r"\s+", " ", text.replace(" ", " ")).strip()


def _assign_headings(
    chunks: List[UIRChunk],
    toc_headings: Optional[Dict[Any, Any]],
    *,
    doc_title: Optional[str] = None,
) -> None:
    """Assign parent_heading + breadcrumb_path to text chunks in place.

    Precedence per text chunk (in reading order):
      1. nearest preceding in-page heading element — already stamped on the
         chunk by the per-page chunker (``parent_heading``);
      2. carry-forward — the last active heading from an earlier page when
         the chunk has none of its own;
      3. TOC fallback — the breadcrumb whose page range covers this chunk's
         page (from PyMuPDF bookmarks), used when neither 1 nor 2 applies.

    breadcrumb_path is built from the TOC hierarchy: when a chunk's heading
    is a known TOC title, its TOC breadcrumb is used; otherwise the page's
    TOC breadcrumb is used. The document title (when available) is prepended
    and a ``"Page N"`` leaf appended, matching the v2.16 breadcrumb shape.

    No Docling, no batch_processor methods — pure UIR data.
    """
    # Central heading validator (shared with the OCR-lane producer); reused
    # here so the chunker's HEURISTIC in-page headings honor the same
    # garbage-rejection contract. NOTE: PyMuPDF TOC bookmark titles are
    # authoritative document structure and are NOT re-validated through
    # is_valid_heading — that validator is tuned to reject OCR noise and
    # numbered body-step prose, which would wrongly drop legitimate numbered
    # subsection titles like "1.1.1 ...". A heuristic heading that EXACTLY
    # matches a TOC title is therefore trusted.
    from ..state.context_state import is_valid_heading

    toc = toc_headings or {}
    heading_map: Dict[str, List[str]] = toc.get("__heading_map__", {}) if toc else {}
    # page_map: page_number -> breadcrumb at end of that page.
    page_map: Dict[int, List[str]] = (
        {k: v for k, v in toc.items() if isinstance(k, int)} if toc else {}
    )

    doc_name = _normalize_heading(doc_title) if doc_title else "Document"

    def _is_toc_title(h: Optional[str]) -> bool:
        return bool(h) and _normalize_heading(h) in heading_map  # type: ignore[arg-type]

    def _trusted_heading(h: Optional[str]) -> bool:
        # Trust a chunker heading if it is a known TOC title (authoritative)
        # or passes the central garbage-rejection validator.
        if not h:
            return False
        return _is_toc_title(h) or is_valid_heading(h)

    def _page_of(ch: UIRChunk) -> int:
        loc = ch.locator
        if loc is not None and loc.page_number is not None:
            return int(loc.page_number)
        return 0

    def _build_breadcrumb(toc_bc: List[str], page: int) -> List[str]:
        body = [_normalize_heading(b) for b in toc_bc if b]
        crumb = [doc_name] + body
        if page > 0:
            crumb.append(f"Page {page}")
        # Cap depth at 5 (REQ-HIER-04) keeping doc root + deepest context.
        if len(crumb) > 5:
            crumb = [crumb[0]] + crumb[-4:]
        return crumb

    last_heading: Optional[str] = None
    last_breadcrumb: List[str] = []

    for ch in chunks:
        if ch.modality != Modality.TEXT:
            continue
        page = _page_of(ch)
        toc_bc = page_map.get(page)

        own_heading = ch.parent_heading if _trusted_heading(ch.parent_heading) else None

        if own_heading is not None:
            # (1) Nearest preceding in-page heading element. Build its
            # breadcrumb from the TOC heading_map when the title is known,
            # else from the page breadcrumb, else a minimal doc+heading+page.
            norm = _normalize_heading(own_heading)
            heading_bc = heading_map.get(norm)
            if heading_bc:
                breadcrumb = _build_breadcrumb(heading_bc, page)
            elif toc_bc:
                breadcrumb = _build_breadcrumb(toc_bc, page)
            else:
                breadcrumb = _build_breadcrumb([own_heading], page)
            ch.parent_heading = _normalize_heading(own_heading)
            ch.breadcrumb_path = breadcrumb
            last_heading = ch.parent_heading
            last_breadcrumb = breadcrumb
            continue

        # (3) TOC fallback takes precedence over a stale carry when the TOC
        # has a more specific section for this exact page. The TOC leaf is
        # authoritative document structure, so it is used as-is.
        if toc_bc:
            parent = _normalize_heading(toc_bc[-1])
            if parent:
                breadcrumb = _build_breadcrumb(toc_bc, page)
                ch.parent_heading = parent
                ch.breadcrumb_path = breadcrumb
                last_heading = parent
                last_breadcrumb = breadcrumb
                continue

        # (2) Carry-forward the last active heading from an earlier page.
        if last_heading:
            ch.parent_heading = last_heading
            # Re-leaf the carried breadcrumb to this chunk's page.
            if last_breadcrumb:
                body = [b for b in last_breadcrumb if not b.startswith("Page ")]
                crumb = list(body)
                if page > 0:
                    crumb.append(f"Page {page}")
                ch.breadcrumb_path = crumb
            continue
        # No heading available for this chunk — leave it null (honest).


# ---------------------------------------------------------------------------
# Per-page chunking
# ---------------------------------------------------------------------------


def _chunk_page(
    page: UniversalPage,
    *,
    doc_id: str,
    max_chars: int,
    min_chars: int,
    extraction_engine_version: str,
    start_reading_order: int = 0,
) -> List[UIRChunk]:
    """Chunk a single UniversalPage into UIRChunks.

    Groups consecutive TEXT elements by reading order, then splits
    at sentence boundaries. IMAGE and TABLE elements each become
    their own UIRChunk.
    """
    chunks: List[UIRChunk] = []
    reading_order = start_reading_order

    # Separate non-TEXT elements from the TEXT grouping stream
    text_buffer: List[Element] = []
    page_w: float = float(page.dimensions[0]) if page.dimensions[0] else 612.0
    page_h: float = float(page.dimensions[1]) if page.dimensions[1] else 792.0

    def _flush_text_buffer(finalize_page: bool = False) -> None:
        nonlocal reading_order
        if not text_buffer:
            return

        # Group consecutive text elements into chunks
        text_chunks = _partition_text_elements(
            text_buffer,
            page_number=page.page_number,
            page_w=page_w,
            page_h=page_h,
            max_chars=max_chars,
            min_chars=min_chars,
        )

        for tc_text, tc_label, tc_bbox, tc_heading, tc_metadata in text_chunks:
            modality = Modality.TEXT
            structural_flags: Set[StructuralFlag] = set()
            chunk_type_label = tc_label.lower().replace("-", "_").replace(" ", "_")

            # Classify chunk type from label + content heuristics
            if chunk_type_label in _HEADING_LABELS or _looks_like_heading(tc_text, tc_label):
                chunk_type_label = "section_header"
            elif chunk_type_label in _CODE_LABELS or _looks_like_code(tc_text, tc_label):
                chunk_type_label = "code"
            elif chunk_type_label in _LIST_LABELS:
                chunk_type_label = "list_item"

            # Detect subtitle continuations
            if _looks_like_subtitle_continuation(tc_text, chunk_type_label, tc_heading):
                chunk_type_label = "section_header"

            confidence = _element_confidence(text_buffer[0])

            chunks.append(
                UIRChunk(
                    modality=modality,
                    content=tc_text.strip(),
                    locator=Locator(
                        type=LocatorType.BBOX,
                        bbox=tc_bbox,
                        page_number=page.page_number,
                        coordinate_frame=CoordinateFrame.PDF_PAGE_PORTRAIT,
                        page_width=int(page_w),
                        page_height=int(page_h),
                    ),
                    confidence=confidence,
                    extraction_method=EXTRACTION_METHOD_UIR,
                    extraction_engine_version=extraction_engine_version,
                    structural_flags=structural_flags,
                    parent_heading=tc_heading,
                    reading_order=reading_order,
                    original_vlm_type=tc_metadata.get("original_vlm_type"),
                )
            )
            reading_order += 1

        text_buffer.clear()

    for element in page.elements:
        # Charter §7.1: ElementType is the 3-value legacy vocabulary. The VLM
        # adapter smuggles 'code'/'form' through as ElementType.TEXT and tags
        # the original signal here; promote it to Modality.CODE/FORM at this
        # ElementType->Modality boundary (where the 5-value widening belongs).
        promoted = (element.metadata or {}).get("promoted_modality")
        if element.type == ElementType.IMAGE:
            # Flush pending text before emitting image
            _flush_text_buffer()
            chunks.append(
                _image_element_to_uirchunk(
                    element,
                    page,
                    extraction_engine_version,
                    reading_order,
                )
            )
            reading_order += 1
        elif element.type == ElementType.TABLE:
            _flush_text_buffer()
            chunks.append(
                _table_element_to_uirchunk(
                    element,
                    page,
                    extraction_engine_version,
                    reading_order,
                )
            )
            reading_order += 1
        elif promoted == "code":
            _flush_text_buffer()
            chunks.append(
                _code_element_to_uirchunk(
                    element,
                    page,
                    extraction_engine_version,
                    reading_order,
                )
            )
            reading_order += 1
        elif promoted == "form":
            _flush_text_buffer()
            chunks.append(
                _form_element_to_uirchunk(
                    element,
                    page,
                    extraction_engine_version,
                    reading_order,
                )
            )
            reading_order += 1
        else:
            # TEXT element — buffer for grouping
            text_buffer.append(element)

    # Flush any remaining text
    _flush_text_buffer()

    return chunks


# ---------------------------------------------------------------------------
# Element → UIRChunk converters (IMAGE, TABLE, CODE, FORM)
# ---------------------------------------------------------------------------


# Whitespace-normalized TEXT duplicates >= this length also clear
# scripts/qa_full_conversion.py::_duplicate_text_issues (its default
# --min-duplicate-chars), catching whitespace-variant long repeats on top of
# the exact-match pass below.
_DEDUP_MIN_CHARS = 120


def _dedupe_within_page_text(chunks: List[UIRChunk]) -> List[UIRChunk]:
    """Drop later within-page duplicate TEXT chunks (keep the first occurrence).

    The VLM loops on dense pages and re-emits the same paragraph/heading. Two
    QA gates forbid the result, at different strictness:
      * qa_universal_invariants: ANY within-page byte-equal (``content.strip()``)
        text dupe is a hard fail - no length floor;
      * qa_full_conversion: whitespace-normalized dupes >= 120 chars.
    Dedup at the union: drop on an exact ``content.strip()`` match (any length)
    OR a whitespace-normalized match >= ``_DEDUP_MIN_CHARS``. TEXT only;
    IMAGE/TABLE/CODE/FORM, empty text, and cross-page repeats (headers/footers)
    are untouched; order preserved.
    """
    seen_exact: Dict[int, Set[str]] = {}
    seen_norm: Dict[int, Set[str]] = {}
    out: List[UIRChunk] = []
    for c in chunks:
        if c.modality is not Modality.TEXT:
            out.append(c)
            continue
        stripped = (c.content or "").strip()
        page = c.locator.page_number if c.locator else None
        if page is None or not stripped:
            out.append(c)
            continue
        exact = seen_exact.setdefault(page, set())
        norm = re.sub(r"\s+", " ", stripped)
        norms = seen_norm.setdefault(page, set())
        if stripped in exact or (len(norm) >= _DEDUP_MIN_CHARS and norm in norms):
            continue  # within-page duplicate of an earlier chunk
        exact.add(stripped)
        if len(norm) >= _DEDUP_MIN_CHARS:
            norms.add(norm)
        out.append(c)
    return out


def _page_dims_px(page: UniversalPage) -> Tuple[int, int]:
    """Page (width, height) in pixels, with the standard Letter fallback.

    Carried onto each Locator so IngestionChunk.from_uir can populate
    spatial.page_width/page_height (a null page dim is a hard TABLE QA failure).
    """
    w = int(page.dimensions[0]) if page.dimensions and page.dimensions[0] else 612
    h = int(page.dimensions[1]) if page.dimensions and page.dimensions[1] else 792
    return w, h


def _image_element_to_uirchunk(
    element: Element,
    page: UniversalPage,
    extraction_engine_version: str,
    reading_order: int,
) -> UIRChunk:
    """Convert an IMAGE Element to a UIRChunk."""
    bbox = _element_bbox_list(element)
    asset_ref: Optional[str] = None
    # Preserve asset path if stored in metadata
    if "asset_path" in element.metadata:
        asset_ref = str(element.metadata["asset_path"])

    pw, ph = _page_dims_px(page)
    return UIRChunk(
        modality=Modality.IMAGE,
        content=element.content or "",
        locator=Locator(
            type=LocatorType.BBOX,
            bbox=bbox,
            page_number=page.page_number,
            coordinate_frame=CoordinateFrame.PDF_PAGE_PORTRAIT,
            page_width=pw,
            page_height=ph,
        ),
        confidence=ConfidenceBreakdown(),
        extraction_method=EXTRACTION_METHOD_UIR,
        extraction_engine_version=extraction_engine_version,
        asset_ref=asset_ref,
        reading_order=reading_order,
    )


def _table_element_to_uirchunk(
    element: Element,
    page: UniversalPage,
    extraction_engine_version: str,
    reading_order: int,
) -> UIRChunk:
    """Convert a TABLE Element to a UIRChunk.

    Engine-agnostic separator repair: MinerU and Qwen both occasionally emit a
    pipe table WITHOUT the Markdown ``|---|`` separator row (FluentPython p17),
    which fails the table-format gate. ``ensure_table_separator`` injects it and
    is a no-op on already-valid grids and non-pipe (HTML/prose) content.
    """
    bbox = _element_bbox_list(element)
    pw, ph = _page_dims_px(page)
    return UIRChunk(
        modality=Modality.TABLE,
        content=ensure_table_separator(element.content or ""),
        locator=Locator(
            type=LocatorType.BBOX,
            bbox=bbox,
            page_number=page.page_number,
            coordinate_frame=CoordinateFrame.PDF_PAGE_PORTRAIT,
            page_width=pw,
            page_height=ph,
        ),
        confidence=ConfidenceBreakdown(),
        extraction_method=EXTRACTION_METHOD_UIR,
        extraction_engine_version=extraction_engine_version,
        reading_order=reading_order,
    )


def _code_element_to_uirchunk(
    element: Element,
    page: UniversalPage,
    extraction_engine_version: str,
    reading_order: int,
) -> UIRChunk:
    """Convert a CODE Element to a UIRChunk.

    Content is preserved VERBATIM (no strip, no whitespace normalization) so
    code indentation survives - the whole reason vision-native handles code
    instead of letting it fall back to Docling.
    """
    bbox = _element_bbox_list(element)
    pw, ph = _page_dims_px(page)
    return UIRChunk(
        modality=Modality.CODE,
        content=element.content or "",
        locator=Locator(
            type=LocatorType.BBOX,
            bbox=bbox,
            page_number=page.page_number,
            coordinate_frame=CoordinateFrame.PDF_PAGE_PORTRAIT,
            page_width=pw,
            page_height=ph,
        ),
        confidence=ConfidenceBreakdown(),
        extraction_method=EXTRACTION_METHOD_UIR,
        extraction_engine_version=extraction_engine_version,
        reading_order=reading_order,
        original_vlm_type=(element.metadata or {}).get("original_vlm_type"),
    )


def _form_element_to_uirchunk(
    element: Element,
    page: UniversalPage,
    extraction_engine_version: str,
    reading_order: int,
) -> UIRChunk:
    """Convert a FORM Element to a UIRChunk (key-value / structured layout)."""
    bbox = _element_bbox_list(element)
    pw, ph = _page_dims_px(page)
    return UIRChunk(
        modality=Modality.FORM,
        content=element.content or "",
        locator=Locator(
            type=LocatorType.BBOX,
            bbox=bbox,
            page_number=page.page_number,
            coordinate_frame=CoordinateFrame.PDF_PAGE_PORTRAIT,
            page_width=pw,
            page_height=ph,
        ),
        confidence=ConfidenceBreakdown(),
        extraction_method=EXTRACTION_METHOD_UIR,
        extraction_engine_version=extraction_engine_version,
        reading_order=reading_order,
        original_vlm_type=(element.metadata or {}).get("original_vlm_type"),
    )


# ---------------------------------------------------------------------------
# Text buffer → chunk partitioner
# ---------------------------------------------------------------------------


def _partition_text_elements(
    elements: List[Element],
    page_number: int,
    page_w: float,
    page_h: float,
    max_chars: int,
    min_chars: int,
) -> List[Tuple[str, str, List[int], Optional[str], Dict[str, Any]]]:
    """Partition consecutive TEXT elements into chunk-sized groups.

    Returns list of (content, label, bbox, parent_heading, metadata) tuples.
    Each tuple represents one candidate chunk.
    """
    if not elements:
        return []

    # Determine the dominant label for this group
    labels = [e.source_label for e in elements if e.source_label]
    dominant_label = _most_common(labels) if labels else "text"

    # Find the most-recent heading element before this text group
    # (heading is typically a separate element with section_header label)
    parent_heading: Optional[str] = None
    for e in elements:
        if e.source_label.lower().replace("-", "_").replace(" ", "_") in _HEADING_LABELS:
            if e.content.strip():
                parent_heading = e.content.strip()

    # Aggregate any original_vlm_type provenance markers carried by the
    # constituent elements (a degraded-unknown VLM type smuggled in as TEXT).
    # Distinct values are comma-joined so a merged text group containing more
    # than one degraded type stays auditable. Empty for ordinary prose.
    vlm_types = sorted({t for e in elements if (t := (e.metadata or {}).get("original_vlm_type"))})
    group_metadata: Dict[str, Any] = {"original_vlm_type": ",".join(vlm_types)} if vlm_types else {}

    # Build combined text
    full_text = _join_element_contents(elements)
    current_bbox = _elements_union_bbox(elements, page_w, page_h)

    # If under budget, return as single chunk
    if len(full_text) <= max_chars:
        return [(full_text, dominant_label, current_bbox, parent_heading, group_metadata)]

    # Split at sentence boundaries
    parts = _split_at_sentence_boundaries(full_text, max_chars, min_chars)
    result: List[Tuple[str, str, List[int], Optional[str], Dict[str, Any]]] = []
    for part in parts:
        result.append((part, dominant_label, current_bbox, parent_heading, group_metadata))

    return result


# ---------------------------------------------------------------------------
# Sentence-boundary splitting
# ---------------------------------------------------------------------------


def _split_at_sentence_boundaries(
    text: str,
    max_chars: int,
    min_chars: int,
) -> List[str]:
    """Split text at sentence boundaries, respecting the char budget.

    Prefers splitting at `. `, `! `, `? `; falls back to newline splits
    when no sentence boundary is found within the budget.
    """
    if len(text) <= max_chars:
        return [text]

    parts: List[str] = []
    cursor = 0
    while cursor < len(text):
        remaining = text[cursor:]
        if len(remaining) <= max_chars:
            parts.append(remaining)
            break

        # Look for sentence boundary within [max_chars * 0.5, max_chars]
        search_start = cursor + max(0, max_chars // 2)
        search_end = cursor + max_chars
        best_split = -1

        for i in range(search_end - 1, search_start - 1, -1):
            if i >= len(text):
                continue
            ch = text[i]
            if ch in _SENTENCE_ENDS and i + 1 < len(text) and text[i + 1] in {" ", "\n"}:
                best_split = i + 1  # include the period, split after the space
                break

        if best_split < 0:
            # Fallback: split at newline
            nl_idx = text.find("\n", search_start)
            if 0 < nl_idx < search_end:
                best_split = nl_idx + 1  # include the newline
            # Final fallback: split at max_chars on a word boundary
            else:
                for i in range(search_end - 1, search_start - 1, -1):
                    if text[i] == " ":
                        best_split = i + 1
                        break
                if best_split < 0:
                    best_split = search_end

        chunk = text[cursor:best_split].strip()
        if chunk:
            parts.append(chunk)
        cursor = best_split

    # Merge undersized tail
    if len(parts) >= 2 and len(parts[-1]) < min_chars:
        parts[-2] = f"{parts[-2]} {parts[-1]}".strip()
        parts.pop()

    return parts or [text]


# ---------------------------------------------------------------------------
# Element content joining
# ---------------------------------------------------------------------------


def _join_element_contents(elements: List[Element]) -> str:
    """Join adjacent TEXT elements with the appropriate separator.

    Consecutive elements from the same block use space; elements
    separated by explicit block boundaries use newline.
    """
    if not elements:
        return ""
    parts: List[str] = []
    for e in elements:
        content = e.content.strip() if e.content else ""
        if content:
            parts.append(content)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# BBox helpers
# ---------------------------------------------------------------------------


def _element_bbox_list(element: Element) -> List[int]:
    """Get element bbox as [x1, y1, x2, y2] list."""
    if element.bbox:
        return element.bbox.to_list()
    return [0, 0, 1000, 1000]


def _elements_union_bbox(
    elements: List[Element],
    page_w: float,
    page_h: float,
) -> List[int]:
    """Union bbox across a list of Elements."""
    left, top, right, bottom = 1000, 1000, 0, 0
    have = False
    for e in elements:
        if e.bbox:
            left = min(left, e.bbox.x_min)
            top = min(top, e.bbox.y_min)
            right = max(right, e.bbox.x_max)
            bottom = max(bottom, e.bbox.y_max)
            have = True
    if have and right > left and bottom > top:
        return [left, top, right, bottom]
    return [0, 0, 1000, 1000]


# ---------------------------------------------------------------------------
# Content classification heuristics
# ---------------------------------------------------------------------------


def _looks_like_heading(text: str, label: str) -> bool:
    """Return True if text content suggests a heading."""
    text = text.strip()
    if not text:
        return False
    # Short, no terminal punctuation, Title Case or ALL CAPS
    if len(text) > 120:
        return False
    if any(text.rstrip().endswith(ch) for ch in (";", ":")):
        return False
    # Chapter/part/section markers
    if re.match(r"^(Chapter|Part|Section|Appendix|Unit|Module)\s+\d+", text):
        return True
    # Short all-caps
    if len(text) < 80 and text.isupper():
        return True
    return False


def _looks_like_code(text: str, label: str) -> bool:
    """Return True if text content suggests code."""
    text = text.strip()
    if not text:
        return False
    # Code indicators: indentation, keywords, braces
    lines = text.splitlines()
    if len(lines) < 3:
        return False
    code_keywords = {
        "def ",
        "class ",
        "import ",
        "from ",
        "return ",
        "if __",
        "def __",
        "// ",
        "package ",
        "func ",
        "public ",
        "private ",
        "void ",
        "int ",
        "var ",
        "const ",
        "let ",
        "async ",
        "await ",
    }
    leading_ws = sum(1 for ln in lines if ln.startswith((" ", "\t")))
    keyword_hits = sum(1 for ln in lines if any(kw in ln for kw in code_keywords))
    if keyword_hits >= 2:
        return True
    if leading_ws >= len(lines) * 0.6 and keyword_hits >= 1:
        return True
    return False


def _looks_like_subtitle_continuation(
    content: str,
    label: str,
    parent_heading: Optional[str],
) -> bool:
    """Detect subtitle continuations (cross-page title fragments)."""
    if "section_header" in label or "heading" in label:
        return False
    text = content.strip()
    if not text or len(text) >= 30:
        return False
    if text.endswith((".", "?", "!", ":", ";", ",")):
        return False
    if not parent_heading:
        return False
    match = re.match(r"^([A-Za-z]+)", text)
    if not match:
        return False
    first_word = match.group(1).lower()
    return first_word in _HEADING_CONTINUATION_LEADS


def _element_confidence(element: Element) -> ConfidenceBreakdown:
    """Build ConfidenceBreakdown from a single Element's confidence."""
    c = element.confidence
    return ConfidenceBreakdown(
        text_extraction_confidence=c,
        applicable={"text_extraction_confidence"},
    )


def _most_common(items: List[str]) -> str:
    """Return the most common item in a list."""
    if not items:
        return "text"
    from collections import Counter

    return Counter(items).most_common(1)[0][0]
