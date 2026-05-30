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

from .vlm_provider import VlmProvider, VlmProviderConfig

logger = logging.getLogger(__name__)


PAGE_RENDER_DPI = 200


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
      "type": "text" | "image" | "table",
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
6. Return JSON only — any non-JSON output is a contract violation.
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
                raw_json = self.provider.describe(
                    image_bytes,
                    prompt,
                    mime="image/png",
                )
                payload = self._parse_strict_json(raw_json)
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
    def _project_bbox_to_uir(
        raw_bbox: List[int], pixel_width: int, pixel_height: int
    ) -> List[int]:
        """Project raw pixel-space [x_min, y_min, x_max, y_max] into [0, 1000].

        Deterministic adapter math — does not depend on the VLM honoring
        any normalization instruction. Clamps to the valid UIR range and
        guarantees x_max > x_min, y_max > y_min (BoundingBox invariants).
        """
        if pixel_width <= 0 or pixel_height <= 0:
            raise ValueError(
                f"pixel_width={pixel_width}, pixel_height={pixel_height} "
                "must both be > 0"
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
            raise ValueError(
                f"VLM payload must be JSON object, got {type(payload).__name__}"
            )
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
                    f"UIR element[{index}] must be an object, got "
                    f"{type(raw_element).__name__}"
                )
            type_raw = raw_element.get("type") or "text"
            element_type = ElementType(type_raw)
            content = raw_element.get("content") or ""
            raw_bbox = raw_element.get("bbox")
            normalized_bbox = None
            if raw_bbox is not None:
                raw_ints = [int(round(float(c))) for c in raw_bbox]
                normalized_bbox = cls._project_bbox_to_uir(
                    raw_ints, pixel_width, pixel_height
                )
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
                )
            )

        return create_page(
            page_number=page_number,
            elements=elements,
            dimensions=(pixel_width, pixel_height),
            classification=classification,
        )
