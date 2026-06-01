"""Cost-optimizer page router — the V3 Phase C default engine.

Per-page pre-flight via PyMuPDF picks the right tool per page:

* Pages with images, tables, or non-trivial vector drawings are
  visually complex → routed to the ``VlmNativeEngine`` (vision-native
  UIR JSON extraction).
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

import io
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
from .vlm_native import VlmNativeEngine, _build_schema_prompt
from .vlm_provider import VlmInfraError, VlmProvider, VlmProviderConfig

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


class HybridEngine:
    """Cost-optimizer router fulfilling ``extract(str) -> UniversalDocument``."""

    def __init__(
        self,
        vlm_engine: Optional[VlmNativeEngine] = None,
        docling_engine: Optional[DoclingFastEngine] = None,
        drawings_threshold: Optional[int] = None,
    ) -> None:
        self.vlm_engine = vlm_engine or VlmNativeEngine()
        self.docling_engine = docling_engine or DoclingFastEngine()
        self.drawings_threshold = (
            drawings_threshold
            if drawings_threshold is not None
            else _drawings_threshold()
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
        return "docling", (
            f"prose (images=0, tables=0, drawings={n_drawings}"
            f"<={self.drawings_threshold})"
        )

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
                provider = self._provider_from_engine(self.vlm_engine)
                for i in vlm_page_indices:
                    page_number = i + 1
                    try:
                        image_bytes, pw, ph = self._render_page_png(doc[i])
                        prompt = _build_schema_prompt(pw, ph)
                        raw_json = provider.describe(
                            image_bytes, prompt, mime="image/png"
                        )
                        payload = VlmNativeEngine._parse_strict_json(raw_json)
                        universal_page = VlmNativeEngine._page_from_payload(
                            payload,
                            fallback_page_number=page_number,
                            pixel_width=pw,
                            pixel_height=ph,
                        )
                        pages_by_number[page_number] = universal_page
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

        ordered_pages = [
            pages_by_number[n] for n in sorted(pages_by_number.keys())
        ]
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _provider_from_engine(engine: VlmNativeEngine) -> VlmProvider:
        """Return the underlying provider (lazy-construct if needed)."""
        if engine._provider is None:
            engine._provider = VlmProvider(VlmProviderConfig.from_env())
        return engine._provider

    @staticmethod
    def _render_page_png(page: "fitz.Page") -> Tuple[bytes, int, int]:
        zoom = VlmNativeEngine().render_dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        buffer = io.BytesIO()
        buffer.write(pixmap.tobytes("png"))
        return buffer.getvalue(), pixmap.width, pixmap.height


def extract(file_path: str) -> UniversalDocument:
    """Module-level convenience: ``extract(file_path) -> UniversalDocument``."""
    return HybridEngine().extract(file_path)
