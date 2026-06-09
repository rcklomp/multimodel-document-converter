"""V3 Phase C processor — engine-agnostic UIR producer.

Exposes a single ``extract`` function that returns a
``mmrag_v2.universal.intermediate.UniversalDocument``. Several engines
plug in via the same contract: MinerU2.5 (the chosen default extractor),
the vision-native VLM engine, fast-Docling, and the legacy HybridEngine
cost-optimizer.

Routing is controlled by ``USE_MINERU_ENGINE`` / ``USE_VLM_ENGINE`` /
``USE_DOCLING_FAST`` / ``USE_HYBRID_ENGINE`` (and, for the default,
whether ``MINERU_ENDPOINT`` is configured) so any batch driver can flip
the engine without code edits. See ``extract`` for the precedence.

This file lives in the Phase C namespace and intentionally does NOT
import from ``v3_execution_root``; the execution-sandbox subprocess
loads it by absolute file path to sidestep the ``mmrag_v3`` namespace
collision between the two trees.
"""

from __future__ import annotations

import logging
import os
from typing import Union

from mmrag_v2.universal.intermediate import (
    BoundingBox,
    DocumentMetadata,
    Element,
    ElementType,
    ExtractionMethod,
    PageClassification,
    UniversalDocument,
    UniversalPage,
)

from .engines.docling_fast import DoclingFastEngine
from .engines.mineru_native import MineruNativeEngine
from .engines.router import HybridEngine, MineruQwenHybridEngine
from .engines.vlm_native import VlmNativeEngine

logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip() in {"1", "true", "TRUE", "yes"}


def is_mineru_route_enabled() -> bool:
    """Return True when the MinerU2.5 engine is force-selected via env."""
    return _env_flag("USE_MINERU_ENGINE")


def is_vlm_route_enabled() -> bool:
    """Return True when the VLM-native engine is force-selected via env."""
    return _env_flag("USE_VLM_ENGINE")


def is_docling_fast_route_enabled() -> bool:
    """Return True when the fast-Docling engine is force-selected via env."""
    return _env_flag("USE_DOCLING_FAST")


def is_hybrid_route_enabled() -> bool:
    """Return True when the legacy HybridEngine is force-selected via env."""
    return _env_flag("USE_HYBRID_ENGINE")


def is_mineru_qwen_hybrid_route_enabled() -> bool:
    """Return True when the MinerU+Qwen-for-code hybrid is force-selected."""
    return _env_flag("USE_MINERU_QWEN_HYBRID")


def _default_route_is_mineru() -> bool:
    """The default route is MinerU2.5 whenever a MinerU server is configured.

    MinerU2.5 is the chosen extractor (PLAN_VLM_EVAL §10-13). Making it the
    default is gated on ``MINERU_ENDPOINT`` being set so that standing up a
    MinerU server is the single, explicit act that opts a deployment into it;
    setups with no MinerU endpoint keep the legacy ``HybridEngine`` default
    and are never hard-broken by this flip.
    """
    return bool(os.environ.get("MINERU_ENDPOINT", "").strip())


# --- Fail-closed fallback (2026-06-09) --------------------------------------
# Reliability must NOT depend on a remote multi-model GPU server staying healthy.
# When the selected engine raises (server 500s, JSON-contract mismatch, model
# load failure), or returns a page where it detected text regions but extracted
# NO text (the M5 mlx MinerU2.5 content-step failure: TEXT elements present,
# combined text empty), that page is recovered from the offline DoclingFastEngine,
# which has no network/model-server dependency. Healthy extractions pay nothing;
# a degraded server degrades GRACEFULLY and LOUDLY (logged + stamped on
# ``doc.metadata.extra``) instead of silently emitting empty chunks.
_FALLBACK_ENGINE_NAME = "docling_fast"


def _select_engine() -> "tuple[str, object]":
    """Resolve (engine_name, engine) by the env precedence (first match wins)."""
    if is_mineru_route_enabled():
        return "mineru", MineruNativeEngine()
    if is_vlm_route_enabled():
        return "vlm_native", VlmNativeEngine()
    if is_docling_fast_route_enabled():
        return "docling_fast", DoclingFastEngine()
    if is_hybrid_route_enabled():
        return "hybrid", HybridEngine()
    if is_mineru_qwen_hybrid_route_enabled():
        return "mineru_qwen_hybrid", MineruQwenHybridEngine()
    if _default_route_is_mineru():
        return "mineru_qwen_hybrid", MineruQwenHybridEngine()
    return "hybrid", HybridEngine()


def _page_is_degenerate(page) -> bool:
    """True when the engine found TEXT regions but extracted no content at all.

    Defense-in-depth signature (TEXT elements present, combined content empty). A
    genuinely image-only page has no TEXT elements and is NOT degenerate, so the
    VLM/MinerU lane is never second-guessed on real figures. NOTE: ``Element``
    stores text in ``.content`` (there is no ``.text`` attribute). An over-fire is
    safe: the page is only replaced when docling does strictly better.
    """
    has_text_region = any(e.type == ElementType.TEXT for e in page.elements)
    if not has_text_region:
        return False
    content_chars = sum(len((e.content or "").strip()) for e in page.elements)
    return content_chars == 0


def _stamp(doc: UniversalDocument, *, engine: str, fallback=None,
           degraded: int = 0, recovered: int = 0, reason: str = "") -> UniversalDocument:
    """Record the served lane + fallback outcome on ``doc.metadata.extra``."""
    extra = doc.metadata.extra
    extra["extraction_engine"] = engine
    extra["extraction_fallback"] = fallback
    extra["extraction_degraded_pages"] = degraded
    extra["extraction_recovered_pages"] = recovered
    if reason:
        extra["extraction_fallback_reason"] = reason
    return doc


# --- Terminal tier: PyMuPDF native text -------------------------------------
# The last rung of the ladder. No model, no network, no GPU server — it reads the
# PDF's own text layer. The ONLY way it fails is an unreadable file, which is a
# true input error (raised loudly), never a silent degrade. A page with no text
# layer (genuine scan/blank) yields an empty-but-present page: we do NOT fabricate
# text. Guarantee: no page that HAS extractable text is ever zeroed by an
# engine/server/network failure.
_TERMINAL_ENGINE_NAME = "pymupdf_terminal"


def _pymupdf_page(fdoc, page_index: int, page_number: int) -> "tuple[UniversalPage, bool]":
    """Build a UniversalPage from one PDF page's native text. Returns (page, has_text)."""
    fpage = fdoc[page_index]
    rect = fpage.rect
    dims = (int(rect.width) or 1000, int(rect.height) or 1000)
    text = (fpage.get_text() or "").strip()
    if not text:
        return UniversalPage(page_number=page_number, elements=[],
                             classification=PageClassification.SCANNED, dimensions=dims), False
    element = Element(
        type=ElementType.TEXT, content=text,
        bbox=BoundingBox(x_min=0, y_min=0, x_max=1000, y_max=1000),
        confidence=1.0, extraction_method=ExtractionMethod.NATIVE,
        element_index=0, source_label="pymupdf_text",
    )
    return UniversalPage(page_number=page_number, elements=[element],
                         classification=PageClassification.DIGITAL, dimensions=dims), True


def _terminal_doc(path: str) -> UniversalDocument:
    """Whole-doc terminal extraction (used when the primary AND docling both fail)."""
    import fitz  # lazy: PyMuPDF, already a dependency of the MinerU engine

    fdoc = fitz.open(path)
    try:
        pages = [_pymupdf_page(fdoc, i, i + 1)[0] for i in range(fdoc.page_count)]
    finally:
        fdoc.close()
    return UniversalDocument(
        doc_id=UniversalDocument.compute_doc_id(path), source_file=str(path),
        file_type="pdf", pages=pages, metadata=DocumentMetadata(), total_pages=len(pages),
    )


def _terminal_recover(doc: UniversalDocument, path: str, indices: list) -> int:
    """Swap native-text pages in for `indices` that have a text layer. Returns count."""
    import fitz

    recovered = 0
    fdoc = fitz.open(path)
    try:
        for i in indices:
            page, has_text = _pymupdf_page(fdoc, doc.pages[i].page_number - 1, doc.pages[i].page_number)
            if has_text:
                doc.pages[i] = page
                recovered += 1
    finally:
        fdoc.close()
    return recovered


def extract(file_path: Union[str, "os.PathLike[str]"]) -> UniversalDocument:
    """Run the Phase C pipeline and return a v2-UIR document, FAIL-CLOSED.

    Routing precedence (first match wins):
        * ``USE_MINERU_ENGINE=1``        → ``MineruNativeEngine`` (pure MinerU).
        * ``USE_VLM_ENGINE=1``           → ``VlmNativeEngine``.
        * ``USE_DOCLING_FAST=1``         → ``DoclingFastEngine``.
        * ``USE_HYBRID_ENGINE=1``        → legacy ``HybridEngine`` (explicit).
        * ``USE_MINERU_QWEN_HYBRID=1``   → ``MineruQwenHybridEngine`` (explicit).
        * default                        → ``MineruQwenHybridEngine`` when
          ``MINERU_ENDPOINT`` is configured, else legacy ``HybridEngine``.

    The default route is the MinerU+Qwen-for-code hybrid (code-dense pages to Qwen,
    the rest to MinerU). Whatever the route, extraction is FAIL-CLOSED through a
    three-tier ladder; each tier only serves pages the tier above could not:

        tier 1  selected engine (best quality, may fail/degrade)
        tier 2  offline ``DoclingFastEngine`` (no network; may itself fail)
        tier 3  PyMuPDF native text layer (no model/network; only an unreadable
                file defeats it — a true input error, raised loudly)

    Guarantee: no page that HAS extractable text is ever zeroed by an engine,
    server, or network failure. A genuinely text-less page (scan/blank) stays
    empty — we do not fabricate. The served engine, fallback tier, and degraded/
    recovered page counts are stamped on ``doc.metadata.extra`` and logged.
    """
    path = str(file_path)
    engine_name, engine = _select_engine()

    # 1) Whole-engine failure (server 500s, JSON mismatch, model load fail).
    try:
        doc = engine.extract(path)
    except Exception as exc:  # noqa: BLE001 — fail-closed on ANY engine failure
        if engine_name == _FALLBACK_ENGINE_NAME:
            raise  # nothing more reliable to fall back to
        logger.warning(
            "extract: %s raised %s; falling back to %s for %s",
            engine_name, type(exc).__name__, _FALLBACK_ENGINE_NAME, path,
        )
        try:
            doc = DoclingFastEngine().extract(path)
            fallback = _FALLBACK_ENGINE_NAME
        except Exception as dexc:  # noqa: BLE001 — docling can fail too; drop to terminal
            logger.warning(
                "extract: %s also failed (%s); using PyMuPDF terminal for %s",
                _FALLBACK_ENGINE_NAME, type(dexc).__name__, path,
            )
            doc = _terminal_doc(path)
            fallback = _TERMINAL_ENGINE_NAME
        return _stamp(
            doc, engine=engine_name, fallback=fallback,
            degraded=len(doc.pages),
            recovered=sum(1 for p in doc.pages if not _page_is_degenerate(p)),
            reason=f"{type(exc).__name__}: {exc}",
        )

    if engine_name == _FALLBACK_ENGINE_NAME:
        return _stamp(doc, engine=engine_name)

    # 2) Per-page degradation (e.g. MinerU empty content-step on some pages).
    degraded = [i for i, p in enumerate(doc.pages) if _page_is_degenerate(p)]
    if not degraded:
        return _stamp(doc, engine=engine_name)

    logger.warning(
        "extract: %s produced %d/%d degraded page(s) for %s; recovering from %s",
        engine_name, len(degraded), len(doc.pages), path, _FALLBACK_ENGINE_NAME,
    )
    recovered = 0
    try:
        fb_doc = DoclingFastEngine().extract(path)
        fb_by_page = {p.page_number: p for p in fb_doc.pages}
        for i in degraded:
            fb_page = fb_by_page.get(doc.pages[i].page_number)
            if fb_page is not None and not _page_is_degenerate(fb_page):
                doc.pages[i] = fb_page
                recovered += 1
    except Exception as exc:  # noqa: BLE001 — keep primary if docling itself fails
        logger.warning(
            "extract: docling fallback itself failed (%s); trying terminal for %s",
            type(exc).__name__, path,
        )
    # 3) Terminal tier: any page neither primary nor docling could serve falls back
    #    to the PDF's native text layer (no model, no network — cannot lose text).
    still = [i for i in degraded if _page_is_degenerate(doc.pages[i])]
    if still:
        try:
            recovered += _terminal_recover(doc, path, still)
        except Exception as exc:  # noqa: BLE001 — terminal only fails on unreadable file
            logger.warning("extract: PyMuPDF terminal failed (%s) for %s", type(exc).__name__, path)
    return _stamp(
        doc, engine=engine_name, fallback=_FALLBACK_ENGINE_NAME,
        degraded=len(degraded), recovered=recovered,
    )
