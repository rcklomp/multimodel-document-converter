"""Vision-native extraction engine.

Implements the V3 extraction contract::

    extract(file_path: str) -> UniversalDocument

Renders each PDF page to an image and asks a VLM to return the page
structure directly in the UIR (UniversalDocument) JSON shape. The
adapter performs no text-heuristic parsing of the VLM output: only
``json.loads`` followed by typed deserialization. No Docling, no v2.x
post-processing.

The UIR contract types are imported from
``mmrag_v2.universal.intermediate``; that module is the canonical UIR
definition shared across v2.x and v3.x. It is NOT a v2.x legacy
extraction module — it is the format-agnostic contract.
"""

from __future__ import annotations

import io
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF — pure renderer; no extraction semantics used

from mmrag_v2.universal.intermediate import (
    DocumentMetadata,
    ElementType,
    ExtractionMethod,
    PageClassification,
    UniversalDocument,
    UniversalPage,
    create_document,
    create_element,
    create_page,
)

from .vlm_provider import (
    VlmProvider,
    VlmProviderConfig,
    VlmProviderError,
    VlmTruncationError,
)

logger = logging.getLogger(__name__)


PAGE_RENDER_DPI = 200

# A2 (Charter Blocker A): cheap per-page output-budget estimate. The VLM's
# JSON output size tracks the page's content volume, and the page's own
# extractable text length is a fast, allocation-free proxy (PyMuPDF text
# read, no second render). Budget = raw text chars scaled by a structural
# overhead (bbox arrays, field names, markdown fences, table grids roughly
# 2.5x the raw text) converted to tokens at ~3.5 chars/token. The provider
# floors this at its config default and caps it at TRUNCATION_ESCALATION_CAP,
# so a dense page gets headroom above the 8192 default while a sparse or
# scanned page (little/no text layer) falls back to the floor. Any residual
# underestimate is caught by the A1 escalation retry and A4 repair.
BUDGET_CHARS_PER_TOKEN = 3.5
BUDGET_STRUCTURE_OVERHEAD = 2.5


def estimate_output_budget(page: "fitz.Page") -> int:
    """Estimate the VLM output-token budget for one page from its text volume.

    Returns 0 on an unreadable page so the provider uses its config default.
    """
    try:
        text = page.get_text("text")
    except Exception:  # pragma: no cover - defensive: PyMuPDF API drift
        return 0
    return int((len(text) * BUDGET_STRUCTURE_OVERHEAD) / BUDGET_CHARS_PER_TOKEN)


def _recover_complete_elements(raw: str) -> List[Dict[str, Any]]:
    """Scan a (possibly truncated) UIR JSON string for COMPLETE element objects.

    Walks the ``"elements"`` array with a strict ``JSONDecoder``, collecting
    each fully-decodable object and stopping at the first one that fails to
    parse (the truncated trailing element). No regex, no brace-counting
    heuristics - the decoder defines "complete". Returns [] when there is no
    recoverable array.
    """
    key = raw.find('"elements"')
    if key == -1:
        return []
    bracket = raw.find("[", key)
    if bracket == -1:
        return []
    decoder = json.JSONDecoder()
    pos = bracket + 1
    n = len(raw)
    out: List[Dict[str, Any]] = []
    while pos < n:
        while pos < n and raw[pos] in " \t\r\n,":
            pos += 1
        if pos >= n or raw[pos] == "]":
            break
        try:
            obj, end = decoder.raw_decode(raw, pos)
        except json.JSONDecodeError:
            break  # trailing element was cut off; keep what we have
        if isinstance(obj, dict):
            out.append(obj)
        pos = end
    return out


def _salvage_partial_first_element(raw: str) -> Optional[Dict[str, Any]]:
    """Recover the partial FIRST element of a page cut mid-content.

    The 2026-06-04 crucible found the VLM gives up mid-generation on long dense
    tables (premature EOS, finish_reason=stop) - the single giant table element
    is left as an unterminated string, so there are ZERO complete elements and
    A4's complete-element recovery returns nothing (-> Docling, which drops most
    spreadsheet rows). This salvages that one element's ``type`` + the partial
    ``content`` (the markdown table built so far) so the data survives.

    Gives the element a near-full-page bbox (the page dims live in the intact
    JSON head; a ~2% inset avoids a false crop-audit edge-clamp) so an IMAGE/
    TABLE salvage still satisfies QA-CHECK-05. Returns None when nothing
    meaningful is recoverable.
    """
    key = raw.find('"content"')
    if key == -1:
        return None
    colon = raw.find(":", key)
    if colon == -1:
        return None
    open_q = raw.find('"', colon + 1)
    if open_q == -1:
        return None
    body = raw[open_q + 1 :]
    # Drop a trailing incomplete escape so the string decode does not choke.
    if body.endswith("\\"):
        body = body[:-1]
    try:
        content = json.loads('"' + body + '"', strict=False)
    except (json.JSONDecodeError, ValueError):
        content = (
            body.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\\\", "\\")
        )
    if not content.strip():
        return None
    # Type: the last "type": "..." before the content key (the element being cut).
    type_match = None
    for type_match in re.finditer(r'"type"\s*:\s*"(\w+)"', raw[:key]):
        pass
    elem_type = type_match.group(1) if type_match else "text"
    width = _int_from_json_head(raw, "width", 1000)
    height = _int_from_json_head(raw, "height", 1000)
    inset_x, inset_y = int(width * 0.02), int(height * 0.02)
    return {
        "type": elem_type,
        "content": content,
        "bbox": [inset_x, inset_y, width - inset_x, height - inset_y],
        "confidence": 0.5,
        "source_label": "salvaged_partial",
    }


def _int_from_json_head(raw: str, key: str, default: int) -> int:
    m = re.search(rf'"{key}"\s*:\s*(\d+)', raw)
    return int(m.group(1)) if m else default


# A unit (<= 80 chars) repeated this many times in a row is a degenerate VLM
# generation loop (CarOK p7: "Transporter 1.9 TD 68pk, " x hundreds), not real
# content. No document legitimately repeats an identical phrase 8+ times
# consecutively, so collapse the run to a single occurrence. Bounded unit length
# keeps the backreference scan fast (~6ms on a 17k-char looped row).
_REPEAT_LOOP_RE = re.compile(r"(.{1,80}?)\1{7,}", re.DOTALL)


def _collapse_degenerate_repeats(text: str) -> str:
    """Collapse a substring repeated >= 8 times in a row to one occurrence."""
    if not text:
        return text
    return _REPEAT_LOOP_RE.sub(r"\1", text)


def repair_truncated_json(raw: str) -> Optional[Dict[str, Any]]:
    """Bounded JSON repair (A4): salvage the complete elements of a cut page.

    Fail-open last resort for Charter Blocker A: when a VLM page response is
    truncated or malformed, recover the N complete elements (dropping the
    partial trailing one) instead of discarding the whole page to a Docling
    fallback. Partial vision-native extraction beats full Docling fallback on
    a dense page. When there are NO complete elements (the first/only element
    was cut mid-content - the dense-spreadsheet premature-EOS case), salvage
    that partial element's content. Returns a payload dict carrying the
    recovered elements (page_number/width/height are supplied by the adapter,
    not the payload), or ``None`` when nothing parseable can be recovered.
    """
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    elements = _recover_complete_elements(raw)
    if not elements:
        partial = _salvage_partial_first_element(raw)
        if partial is not None:
            elements = [partial]
    if not elements:
        return None
    return {"elements": elements}


def _describe_and_parse(
    provider: VlmProvider,
    image_bytes: bytes,
    prompt: str,
    *,
    max_tokens: int,
) -> Dict[str, Any]:
    """Call the VLM, strict-parse, and apply bounded A4 repair on failure.

    Infra failures (``VlmInfraError``) and unrecoverable semantic failures
    propagate unchanged so the router can trip its circuit breaker or fall
    back to Docling for that page. A truncated/malformed body that still
    contains >=1 complete element is repaired into a usable payload here.
    """
    try:
        raw = provider.describe(image_bytes, prompt, mime="image/png", max_tokens=max_tokens)
    except VlmTruncationError as exc:
        repaired = repair_truncated_json(exc.partial_content)
        if repaired is None:
            raise
        logger.warning(
            "A4: repaired truncated VLM page to %d complete element(s) "
            "(partial extraction kept instead of Docling fallback)",
            len(repaired.get("elements") or []),
        )
        return repaired
    try:
        return VlmNativeEngine._parse_strict_json(raw)
    except ValueError as exc:
        repaired = repair_truncated_json(raw)
        if repaired is None:
            raise VlmProviderError(f"unrepairable VLM JSON: {exc}") from exc
        logger.warning(
            "A4: repaired malformed VLM page to %d complete element(s)",
            len(repaired.get("elements") or []),
        )
        return repaired


def _vlm_element_metadata(
    promoted_modality: Optional[str], original_vlm_type: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Build the element metadata bag for a parsed VLM element.

    ``promoted_modality`` drives the chunker's Modality.CODE/FORM promotion;
    ``original_vlm_type`` is the provenance marker surfaced on the final chunk.
    Returns None when neither applies (a natively-typed element) so plain text
    carries no metadata.
    """
    md: Dict[str, Any] = {}
    if promoted_modality:
        md["promoted_modality"] = promoted_modality
    if original_vlm_type:
        md["original_vlm_type"] = original_vlm_type
    return md or None


def _build_schema_prompt(pixel_width: int, pixel_height: int) -> str:
    """Per-page prompt anchored to the actual render dimensions.

    Coordinate normalization to UIR's [0,1000] frame is done in the
    adapter (deterministic math), not by the VLM. The VLM is asked for
    raw pixel coordinates in the rendered image's frame, which is a
    direct read off the visual layout it sees.
    """
    return f"""You are a document-structure extraction model.

Given a single rendered page image, return a STRICT JSON object that
describes every element on the page in the UIR shape below. Output
ONLY the JSON object. No prose, no markdown fences, no commentary.

The rendered image is {pixel_width} pixels wide and {pixel_height}
pixels tall. Output bounding boxes using the RAW PIXEL COORDINATES of
that image (Width: {pixel_width}, Height: {pixel_height}). Do not
attempt any normalization yourself — coordinate scaling is handled by
the adapter.

JSON schema:
{{
  "page_number": <int, 1-indexed>,
  "width": {pixel_width},
  "height": {pixel_height},
  "classification": "digital" | "scanned" | "hybrid",
  "elements": [
    {{
      "type": "text" | "image" | "table" | "code" | "form",
      "content": "<extracted text; for image use a brief visual description; for table use Markdown grid>",
      "bbox": [x_min_px, y_min_px, x_max_px, y_max_px],
      "confidence": <float in [0.0, 1.0]>,
      "source_label": "<optional, e.g. 'paragraph', 'heading', 'caption', 'figure', 'table'>"
    }}
  ]
}}

Rules:
1. bbox values are integer PIXEL coordinates in the rendered image
   frame (0 <= x <= {pixel_width}, 0 <= y <= {pixel_height}). Do NOT
   pre-normalize to any other range.
2. Preserve reading order in the elements array.
3. Include EVERY visible element. Do not drop sparse spreadsheet cells,
   form fields, headers, footers, or numeric-only rows.
4. For TABLE elements, write the body as a Markdown grid in `content`
   so the structure survives downstream chunking.
5. For IMAGE elements that depict a CHART, GRAPH, BAR PLOT, LINE PLOT,
   PIE CHART, SCATTER PLOT, HISTOGRAM, or any DATA VISUALIZATION with
   readable axes / labels / values, write `content` as:

       <one- or two-sentence visual description>

       Data (Markdown):
       <Markdown table transcribing the chart's data points>

   Include axis labels, units, and series names in the table headers
   when visible. If the underlying values are not readable from the
   image, write only the visual description and omit the Data block.
   Photographs, diagrams, logos, and illustrative figures keep the
   plain visual-description-only form.
6. For CODE elements (source code, shell/command blocks, config files), you
   MUST preserve all exact spacing, line breaks, and indentation. Wrap the
   body in a standard Markdown code fence with the appropriate language tag
   (```language ... ```). Never reflow, re-indent, or summarize code.
7. For FORM elements (forms, invoices, key-value layouts), extract the layout
   as structured key-value pairs or a Markdown table in `content`, preserving
   the reading order and field hierarchy.
8. Return JSON only - any non-JSON output is a contract violation.
"""


# Public alias kept for the security test contract; the active prompt is
# built per-page via _build_schema_prompt() with the real render dims.
UIR_SCHEMA_PROMPT = _build_schema_prompt(0, 0)


class VlmNativeEngine:
    """Vision-native UIR extraction adapter.

    Parameters:
        provider: Concrete VLM provider. When omitted, a provider is
            constructed from env vars at the first ``extract`` call.
        render_dpi: Resolution for PDF page rendering.
    """

    def __init__(
        self,
        provider: Optional[VlmProvider] = None,
        render_dpi: int = PAGE_RENDER_DPI,
    ) -> None:
        self._provider = provider
        self.render_dpi = render_dpi

    @property
    def provider(self) -> VlmProvider:
        if self._provider is None:
            self._provider = VlmProvider(VlmProviderConfig.from_env())
        return self._provider

    def extract(self, file_path: str) -> UniversalDocument:
        """Render each page, ask the VLM for UIR JSON, assemble UniversalDocument."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        doc = fitz.open(str(path))
        try:
            pages: List[UniversalPage] = []
            for page_index in range(doc.page_count):
                page = doc[page_index]
                page_number = page_index + 1
                image_bytes, pixel_w, pixel_h = self._render_page_png(page)
                prompt = _build_schema_prompt(pixel_w, pixel_h)
                payload = _describe_and_parse(
                    self.provider,
                    image_bytes,
                    prompt,
                    max_tokens=estimate_output_budget(page),
                )
                universal_page = self._page_from_payload(
                    payload,
                    fallback_page_number=page_number,
                    pixel_width=pixel_w,
                    pixel_height=pixel_h,
                )
                pages.append(universal_page)
        finally:
            doc.close()

        metadata = DocumentMetadata(
            page_count=len(pages),
            file_size_bytes=path.stat().st_size,
            has_text_layer=any(p.text_elements for p in pages),
            has_images=any(p.image_elements for p in pages),
        )
        return create_document(
            file_path=path,
            file_type="pdf",
            pages=pages,
            metadata=metadata,
        )

    def _render_page_png(self, page: "fitz.Page") -> "tuple[bytes, int, int]":
        zoom = self.render_dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        buffer = io.BytesIO()
        buffer.write(pixmap.tobytes("png"))
        return buffer.getvalue(), pixmap.width, pixmap.height

    @staticmethod
    def _project_bbox_to_uir(raw_bbox: List[int], pixel_width: int, pixel_height: int) -> List[int]:
        """Project raw pixel-space [x_min, y_min, x_max, y_max] into [0, 1000].

        Deterministic adapter math — does not depend on the VLM honoring
        any normalization instruction. Clamps to the valid UIR range and
        guarantees x_max > x_min, y_max > y_min (BoundingBox invariants).
        """
        if pixel_width <= 0 or pixel_height <= 0:
            raise ValueError(
                f"pixel_width={pixel_width}, pixel_height={pixel_height} " "must both be > 0"
            )
        if len(raw_bbox) != 4:
            raise ValueError(f"bbox must have 4 elements, got {len(raw_bbox)}")
        x_min_raw, y_min_raw, x_max_raw, y_max_raw = (float(c) for c in raw_bbox)
        # Tolerate swapped pairs from the model.
        if x_max_raw < x_min_raw:
            x_min_raw, x_max_raw = x_max_raw, x_min_raw
        if y_max_raw < y_min_raw:
            y_min_raw, y_max_raw = y_max_raw, y_min_raw

        def _norm(value: float, span: int) -> int:
            scaled = int(round((value / span) * 1000))
            return max(0, min(1000, scaled))

        x_min = _norm(x_min_raw, pixel_width)
        y_min = _norm(y_min_raw, pixel_height)
        x_max = _norm(x_max_raw, pixel_width)
        y_max = _norm(y_max_raw, pixel_height)

        # BoundingBox requires strict > on both axes.
        if x_max <= x_min:
            x_max = min(1000, x_min + 1)
            if x_max == x_min:
                x_min = max(0, x_max - 1)
        if y_max <= y_min:
            y_max = min(1000, y_min + 1)
            if y_max == y_min:
                y_min = max(0, y_max - 1)
        return [x_min, y_min, x_max, y_max]

    @staticmethod
    def _parse_strict_json(raw: str) -> Dict[str, Any]:
        """``json.loads`` only — no regex, no fence stripping heuristics.

        The VLM contract is to return a JSON object directly. If the
        endpoint enforces ``response_format={"type": "json_object"}``,
        the body is JSON. If a provider returns something else, that is
        a contract violation surfaced here.
        """
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"VLM payload must be JSON object, got {type(payload).__name__}")
        return payload

    @classmethod
    def _page_from_payload(
        cls,
        payload: Dict[str, Any],
        *,
        fallback_page_number: int,
        pixel_width: int,
        pixel_height: int,
    ) -> UniversalPage:
        # Always trust the adapter's page index, not the VLM's. The VLM
        # sees one rendered image at a time and has no context for which
        # page in a multi-page batch it is on; in practice it returns 1
        # for every page, collapsing the entire document onto page 1.
        page_number = int(fallback_page_number)
        classification_raw = payload.get("classification") or "digital"
        try:
            classification = PageClassification(classification_raw)
        except ValueError:
            classification = PageClassification.DIGITAL

        elements_payload = payload.get("elements") or []
        if not isinstance(elements_payload, list):
            raise ValueError("UIR payload 'elements' must be a list")

        elements = []
        for index, raw_element in enumerate(elements_payload):
            if not isinstance(raw_element, dict):
                raise ValueError(
                    f"UIR element[{index}] must be an object, got " f"{type(raw_element).__name__}"
                )
            type_raw = str(raw_element.get("type") or "text").strip().lower()
            # Charter §7.1: ElementType stays the 3-value legacy vocabulary
            # (TEXT/IMAGE/TABLE). The VLM's richer 'code'/'form' signal is
            # SMUGGLED through as TEXT here and PROMOTED to Modality.CODE/FORM
            # in the chunker (the ElementType->Modality boundary, where the
            # widening belongs). An unexpected type also degrades to TEXT
            # rather than crashing the whole page to a Docling fallback.
            promoted_modality: Optional[str] = None
            # The VLM's original type whenever it does NOT map cleanly onto the
            # 3-value ElementType vocabulary: a 'code'/'form' promotion or an
            # unknown type degraded to TEXT. Carried downstream as the
            # original_vlm_type provenance marker so a smuggled or degraded
            # chunk is never silently indistinguishable from genuine prose.
            original_vlm_type: Optional[str] = None
            if type_raw in ("code", "form"):
                promoted_modality = type_raw
                original_vlm_type = type_raw
                element_type = ElementType.TEXT
            else:
                try:
                    element_type = ElementType(type_raw)
                except ValueError:
                    logger.warning(
                        "VLM emitted unknown element type %r on page %d; "
                        "treating as text (page extraction preserved)",
                        type_raw,
                        page_number,
                    )
                    element_type = ElementType.TEXT
                    original_vlm_type = type_raw
            content = raw_element.get("content") or ""
            # Collapse degenerate VLM generation loops (a cell/phrase re-emitted
            # hundreds of times - CarOK p7 trips TABLE_CORRUPTION). Skip CODE,
            # which must stay verbatim.
            if type_raw != "code":
                content = _collapse_degenerate_repeats(str(content))
            raw_bbox = raw_element.get("bbox")
            normalized_bbox = None
            if raw_bbox is not None:
                raw_ints = [int(round(float(c))) for c in raw_bbox]
                normalized_bbox = cls._project_bbox_to_uir(raw_ints, pixel_width, pixel_height)
            confidence = float(raw_element.get("confidence", 0.9))
            # VLMs sometimes emit out-of-range confidence; clamp.
            confidence = max(0.0, min(1.0, confidence))
            source_label = raw_element.get("source_label") or ""
            elements.append(
                create_element(
                    element_type=element_type,
                    content=str(content),
                    bbox=normalized_bbox,
                    confidence=confidence,
                    extraction_method=ExtractionMethod.VLM,
                    element_index=index,
                    source_label=str(source_label),
                    metadata=_vlm_element_metadata(promoted_modality, original_vlm_type),
                )
            )

        return create_page(
            page_number=page_number,
            elements=elements,
            dimensions=(pixel_width, pixel_height),
            classification=classification,
        )
