"""MinerU2.5 vision-native extraction engine.

Implements the V3 extraction contract::

    extract(file_path: str) -> UniversalDocument

MinerU2.5 is a two-stage document VLM (global layout detector ->
per-region recognition; `opendatalab/MinerU2.5-*-1.2B`). Its
``two_step_extract`` returns a flat, reading-ordered element list, each
element::

    {"type": <str>, "bbox": [x0, y0, x1, y1], "content": <str>,
     "angle": <int>, "merge_prev": <bool>}

where ``bbox`` is NORMALIZED to [0, 1] floats in the rendered page frame
(top-left origin) and ``type`` is MinerU's rich 13-value vocabulary
(text/title/header/footer/page_number/image/image_caption/table/
table_caption/table_footnote/list/code/aside_text).

This module is the MinerU -> UIR boundary. It contains:

* the PURE converter (this file's first half) — element-JSON + page
  dimensions -> ``UniversalDocument``, with NO model / network dependency,
  so it is fully offline-testable; and
* ``MineruNativeEngine`` (second half) — renders pages with PyMuPDF and
  drives a MinerU2.5 server over HTTP via the light, lazy-imported
  ``mineru_vl_utils`` http-client, keeping the model (and the heavy mlx /
  vllm / transformers stack) in an isolated server, out of the mmrag-v2 env.

Charter §7.1: ElementType stays the 3-value legacy vocabulary
(TEXT/IMAGE/TABLE). MinerU's ``code`` signal is SMUGGLED through as TEXT
here and PROMOTED to Modality.CODE in the chunker (the same boundary the
VLM adapter uses); the original MinerU type is preserved as ``source_label``.

The UIR contract types are imported from ``mmrag_v2.universal.intermediate``
— the canonical, format-agnostic UIR definition (NOT a v2.x extraction
module). No Docling, no v2.x post-processing.
"""

from __future__ import annotations

import logging
import os
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF — pure renderer; no extraction semantics used

from mmrag_v2.universal.intermediate import (
    DocumentMetadata,
    Element,
    ElementType,
    ExtractionMethod,
    UniversalDocument,
    UniversalPage,
    create_document,
    create_element,
    create_page,
)

# Single-source the degenerate-repeat collapse (a unit repeated >= 8x in a
# row) from the VLM adapter so both vision-native engines share one detector.
from .vlm_native import _collapse_degenerate_repeats

logger = logging.getLogger(__name__)


# MinerU2.5's 13-value type vocabulary mapped onto the frozen 3-value
# ElementType (Charter §7.1). Only ``table`` and ``image`` leave the TEXT
# lane; ``code`` is smuggled as TEXT + promoted to Modality.CODE in the
# chunker; every other MinerU type (title/header/footer/page_number/
# image_caption/table_caption/table_footnote/list/aside_text/...) is prose.
_TABLE_TYPES = frozenset({"table"})
_IMAGE_TYPES = frozenset({"image"})
_CODE_TYPES = frozenset({"code"})

# MinerU does not emit a per-element confidence; use the same default the VLM
# adapter applies to elements with no confidence field.
_DEFAULT_CONFIDENCE = 0.9


class _TableHTMLParser(HTMLParser):
    """Collect ``<tr>``/``<td>``/``<th>`` cells from a MinerU HTML table.

    MinerU2.5's recognition stage emits tables as HTML (``<table><tr><td>``),
    but the pipeline contract (R2) and the QA gates require Markdown grids.
    Tolerant by design — MinerU HTML is well-formed, but a stray tag must not
    raise. colspan fills blank trailing cells (Markdown has no cell merging);
    rowspan collapses to the row it appears on (Markdown cannot span rows).
    """

    def __init__(self) -> None:
        super().__init__()
        self.rows: List[List[str]] = []
        self._row: Optional[List[str]] = None
        self._cell: Optional[List[str]] = None
        self._colspan = 1

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
            self._colspan = 1
            for key, value in attrs:
                if key == "colspan" and value:
                    try:
                        self._colspan = max(1, int(value))
                    except ValueError:
                        self._colspan = 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            text = unescape("".join(self._cell))
            text = " ".join(text.split())  # collapse internal newlines/runs
            text = text.replace("|", r"\|")  # pipes would break the grid
            self._row.append(text)
            self._row.extend([""] * (self._colspan - 1))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def _looks_like_html_table(content: str) -> bool:
    lowered = content.lower()
    return "<table" in lowered or "<tr" in lowered


def _html_table_to_markdown(content: str) -> Optional[str]:
    """Transcode a MinerU HTML table into a Markdown grid (None if not parseable).

    First row becomes the header, followed by a ``| --- |`` separator and the
    body rows. Ragged rows are right-padded to the widest row so the grid is
    rectangular.
    """
    parser = _TableHTMLParser()
    try:
        parser.feed(content)
        parser.close()
    except Exception:  # pragma: no cover - defensive: malformed HTML
        return None
    rows = [r for r in parser.rows if r]
    if not rows:
        return None
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines.extend("| " + " | ".join(r) + " |" for r in rows[1:])
    return "\n".join(lines)


def _mineru_bbox_to_uir(bbox: Sequence[float]) -> List[int]:
    """Project a MinerU normalized [0, 1] bbox into the UIR [0, 1000] frame.

    Deterministic adapter math. MinerU emits ``[x_min, y_min, x_max, y_max]``
    as floats in the rendered page frame (top-left origin — the same
    orientation as UIR), so the projection is a direct scale by 1000. Clamps
    to [0, 1000] and guarantees ``x_max > x_min`` / ``y_max > y_min`` (the
    BoundingBox invariants), tolerating swapped pairs.
    """
    if len(bbox) != 4:
        raise ValueError(f"MinerU bbox must have 4 elements, got {len(bbox)}")
    x_min, y_min, x_max, y_max = (float(c) for c in bbox)
    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if y_max < y_min:
        y_min, y_max = y_max, y_min

    def _norm(value: float) -> int:
        return max(0, min(1000, int(round(value * 1000))))

    x0, y0, x1, y1 = _norm(x_min), _norm(y_min), _norm(x_max), _norm(y_max)
    if x1 <= x0:
        x1 = min(1000, x0 + 1)
        if x1 == x0:
            x0 = max(0, x1 - 1)
    if y1 <= y0:
        y1 = min(1000, y0 + 1)
        if y1 == y0:
            y0 = max(0, y1 - 1)
    return [x0, y0, x1, y1]


def _mineru_element_metadata(
    promoted_modality: Optional[str],
    original_vlm_type: Optional[str],
    merge_prev: bool,
) -> Optional[Dict[str, Any]]:
    """Build the element metadata bag for a converted MinerU element.

    ``promoted_modality`` drives the chunker's Modality.CODE promotion;
    ``original_vlm_type`` is the provenance marker surfaced on the final
    chunk; ``merge_prev`` is MinerU's hint that this element continues the
    previous one (preserved for a future reading-order merge pass — inert
    today). Returns None when none apply, so plain prose carries no metadata.
    """
    md: Dict[str, Any] = {}
    if promoted_modality:
        md["promoted_modality"] = promoted_modality
    if original_vlm_type:
        md["original_vlm_type"] = original_vlm_type
    if merge_prev:
        md["merge_prev"] = True
    return md or None


def _mineru_element_to_element(raw: Dict[str, Any], index: int) -> Element:
    """Convert one MinerU element dict into a UIR ``Element``."""
    mtype = str(raw.get("type") or "text").strip().lower()
    promoted_modality: Optional[str] = None
    original_vlm_type: Optional[str] = None
    if mtype in _TABLE_TYPES:
        element_type = ElementType.TABLE
    elif mtype in _IMAGE_TYPES:
        element_type = ElementType.IMAGE
    elif mtype in _CODE_TYPES:
        # Charter §7.1 smuggle: code rides as TEXT, promoted to Modality.CODE
        # by the chunker.
        element_type = ElementType.TEXT
        promoted_modality = "code"
        original_vlm_type = "code"
    else:
        element_type = ElementType.TEXT

    content = str(raw.get("content") or "")
    # MinerU emits tables as HTML; the pipeline contract (R2) + QA gates
    # require Markdown grids. Transcode HTML tables; keep the original on a
    # parse miss so content is never lost.
    if element_type is ElementType.TABLE and _looks_like_html_table(content):
        markdown = _html_table_to_markdown(content)
        if markdown:
            content = markdown
    # Collapse degenerate generation loops on everything but code (which must
    # stay verbatim). MinerU has built-in anti-repetition; this is defense in
    # depth shared with the VLM adapter.
    if mtype not in _CODE_TYPES:
        content = _collapse_degenerate_repeats(content)

    raw_bbox = raw.get("bbox")
    normalized_bbox = _mineru_bbox_to_uir(raw_bbox) if raw_bbox is not None else None
    merge_prev = bool(raw.get("merge_prev"))

    return create_element(
        element_type=element_type,
        content=content,
        bbox=normalized_bbox,
        confidence=_DEFAULT_CONFIDENCE,
        extraction_method=ExtractionMethod.VLM,
        element_index=index,
        # Preserve MinerU's fine-grained type for downstream provenance
        # (heading/caption/footer/list/... semantics survive into chunking).
        source_label=mtype,
        metadata=_mineru_element_metadata(promoted_modality, original_vlm_type, merge_prev),
    )


def _is_text_lane(mtype: str) -> bool:
    """True for MinerU types that ride the TEXT lane (not table/image/code)."""
    return mtype not in _TABLE_TYPES and mtype not in _IMAGE_TYPES and mtype not in _CODE_TYPES


def _merge_continuations(elements_raw: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fold MinerU ``merge_prev`` text continuations into their predecessor.

    MinerU sets ``merge_prev`` on a fragment that continues the previous
    element (e.g. a paragraph broken across a column/line). The chunker joins
    adjacent TEXT elements with a newline, which would insert a spurious break
    mid-paragraph; merging the continuation here (content joined with a space,
    bbox unioned) makes the downstream newline land only at real element
    boundaries. Only TEXT-lane -> TEXT-lane merges (never into/from a table,
    image, or code block). Input dicts are not mutated.
    """
    merged: List[Dict[str, Any]] = []
    for raw in elements_raw:
        mtype = str(raw.get("type") or "text").strip().lower()
        if raw.get("merge_prev") and merged and _is_text_lane(mtype):
            prev = merged[-1]
            if _is_text_lane(str(prev.get("type") or "text").strip().lower()):
                prev_content = str(prev.get("content") or "").strip()
                cur_content = str(raw.get("content") or "").strip()
                prev["content"] = " ".join(p for p in (prev_content, cur_content) if p)
                pb, cb = prev.get("bbox"), raw.get("bbox")
                if pb and cb and len(pb) == 4 and len(cb) == 4:
                    prev["bbox"] = [
                        min(pb[0], cb[0]),
                        min(pb[1], cb[1]),
                        max(pb[2], cb[2]),
                        max(pb[3], cb[3]),
                    ]
                elif cb and not pb:
                    prev["bbox"] = list(cb)
                continue
        merged.append(dict(raw))
    return merged


def mineru_page_to_universal_page(
    elements_raw: Sequence[Dict[str, Any]],
    page_number: int,
    dimensions: Tuple[int, int],
) -> UniversalPage:
    """Convert one page's MinerU element list into a ``UniversalPage``.

    ``dimensions`` is the rendered page's pixel ``(width, height)`` — carried
    onto the page so the chunker can stamp ``page_width``/``page_height`` on
    each Locator (a null page dim is a hard TABLE-structural QA failure).
    Classification is left to the factory's text-volume auto-classifier;
    MinerU does not emit a page-level digital/scanned label. ``merge_prev``
    text continuations are folded in first.
    """
    elements_raw = _merge_continuations(elements_raw)
    elements = [_mineru_element_to_element(raw, i) for i, raw in enumerate(elements_raw)]
    return create_page(
        page_number=int(page_number),
        elements=elements,
        dimensions=dimensions,
        classification=None,
    )


def mineru_to_universal_document(
    pages: Sequence[Tuple[int, Sequence[Dict[str, Any]], Tuple[int, int]]],
    file_path: str,
    *,
    file_type: str = "pdf",
) -> UniversalDocument:
    """Assemble a ``UniversalDocument`` from per-page MinerU element lists.

    ``pages`` is a sequence of ``(page_number, elements_raw, (width, height))``
    tuples in document order. ``file_path`` must point at the real source file
    (its bytes are MD5-hashed for ``doc_id``).
    """
    universal_pages = [
        mineru_page_to_universal_page(elements_raw, page_number, dimensions)
        for page_number, elements_raw, dimensions in pages
    ]
    path = Path(file_path)
    metadata = DocumentMetadata(
        page_count=len(universal_pages),
        file_size_bytes=path.stat().st_size if path.exists() else 0,
        has_text_layer=any(p.text_elements for p in universal_pages),
        has_images=any(p.image_elements for p in universal_pages),
    )
    return create_document(
        file_path=path,
        file_type=file_type,
        pages=universal_pages,
        metadata=metadata,
    )


# ============================================================================
# EXTRACTION ENGINE
# ============================================================================

PAGE_RENDER_DPI = 200
DEFAULT_MINERU_MODEL = "MinerU2.5-2509-1.2B"


class MineruConfigError(RuntimeError):
    """Raised when the MinerU engine is asked to run with no endpoint configured."""


class MineruNativeEngine:
    """MinerU2.5 vision-native UIR extraction engine.

    Fulfils the V3 contract ``extract(file_path: str) -> UniversalDocument``.
    Each page is rendered with PyMuPDF and sent to a MinerU2.5 server (the
    model lives in an ISOLATED env — mlx on the M5, or vLLM on the GX10 —
    NEVER in the mmrag-v2 env). The two-stage layout->recognition
    orchestration is driven by ``mineru_vl_utils`` over the OpenAI-compatible
    HTTP endpoint; that client is a LIGHT pure-python dependency (httpx/
    pillow/pydantic) and is LAZY-imported so this module stays importable
    (and the converter above stays dependency-free) where MinerU is not
    installed.

    Parameters:
        provider/client: an object exposing ``two_step_extract(image) -> list``
            (a ``mineru_vl_utils.MinerUClient``). When omitted, one is built
            from env at first use. Injectable for offline testing.
        render_dpi: PDF page render resolution.
        server_url: MinerU endpoint base (default ``MINERU_ENDPOINT`` env).
        model_name: served model id (default ``MINERU_MODEL`` env).
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        *,
        render_dpi: int = PAGE_RENDER_DPI,
        server_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self._client = client
        self.render_dpi = render_dpi
        self._server_url = (
            server_url if server_url is not None else os.environ.get("MINERU_ENDPOINT", "")
        )
        self._model_name = model_name or os.environ.get("MINERU_MODEL", DEFAULT_MINERU_MODEL)

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> Any:
        if not self._server_url:
            raise MineruConfigError(
                "MinerU endpoint not configured: set MINERU_ENDPOINT (or pass "
                "server_url=) to the OpenAI-compatible MinerU2.5 server."
            )
        # Lazy import: keeps mineru_native importable (and the converter
        # dependency-free) without mineru_vl_utils installed.
        from mineru_vl_utils import MinerUClient

        headers = None
        api_key = os.environ.get("MINERU_API_KEY", "").strip()
        if api_key:
            headers = {"Authorization": f"Bearer {api_key}"}
        return MinerUClient(
            backend="http-client",
            server_url=self._server_url,
            server_headers=headers,
            model_name=self._model_name,
            skip_model_name_checking=True,
        )

    def extract(self, file_path: str) -> UniversalDocument:
        """Render each page, run MinerU two-step extraction, assemble UIR."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        doc = fitz.open(str(path))
        try:
            pages: List[Tuple[int, Sequence[Dict[str, Any]], Tuple[int, int]]] = []
            for page_index in range(doc.page_count):
                image, pixel_w, pixel_h = self._render_page(doc[page_index])
                elements = self.client.two_step_extract(image)
                pages.append((page_index + 1, list(elements), (pixel_w, pixel_h)))
        finally:
            doc.close()

        return mineru_to_universal_document(pages, str(path))

    def _render_page(self, page: "fitz.Page") -> "Tuple[Any, int, int]":
        """Render a page to a PIL RGB image (MinerU's input contract)."""
        from PIL import Image

        zoom = self.render_dpi / 72.0
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        return image, pixmap.width, pixmap.height
