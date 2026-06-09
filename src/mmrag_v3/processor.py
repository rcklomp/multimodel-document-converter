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

from mmrag_v2.universal.intermediate import ElementType, UniversalDocument

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
    the rest to MinerU). Whatever the route, extraction is FAIL-CLOSED: an engine
    that raises, or returns a page that found text regions but no text, has that
    page recovered from the offline ``DoclingFastEngine`` — keeping the primary's
    good pages. The pipeline never silently emits empty chunks because a remote GPU
    server misbehaved. The served engine, fallback, and degraded/recovered page
    counts are stamped on ``doc.metadata.extra`` and logged.
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
        doc = DoclingFastEngine().extract(path)
        return _stamp(
            doc, engine=engine_name, fallback=_FALLBACK_ENGINE_NAME,
            degraded=len(doc.pages), recovered=len(doc.pages),
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
            "extract: docling fallback itself failed (%s); keeping primary for %s",
            type(exc).__name__, path,
        )
    return _stamp(
        doc, engine=engine_name, fallback=_FALLBACK_ENGINE_NAME,
        degraded=len(degraded), recovered=recovered,
    )
