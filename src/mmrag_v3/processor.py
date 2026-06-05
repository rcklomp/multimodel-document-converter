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

import os
from typing import Union

from mmrag_v2.universal.intermediate import UniversalDocument

from .engines.docling_fast import DoclingFastEngine
from .engines.mineru_native import MineruNativeEngine
from .engines.router import HybridEngine, MineruQwenHybridEngine
from .engines.vlm_native import VlmNativeEngine


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


def extract(file_path: Union[str, "os.PathLike[str]"]) -> UniversalDocument:
    """Run the Phase C pipeline and return a v2-UIR document.

    Routing precedence (first match wins):
        * ``USE_MINERU_ENGINE=1``        → all pages through ``MineruNativeEngine``
          (pure MinerU, no per-page Qwen — the escape hatch).
        * ``USE_VLM_ENGINE=1``           → all pages through ``VlmNativeEngine``
        * ``USE_DOCLING_FAST=1``         → all pages through ``DoclingFastEngine``
        * ``USE_HYBRID_ENGINE=1``        → legacy ``HybridEngine`` (explicit)
        * ``USE_MINERU_QWEN_HYBRID=1``   → ``MineruQwenHybridEngine`` (explicit)
        * default                        → ``MineruQwenHybridEngine`` when
          ``MINERU_ENDPOINT`` is configured, else legacy ``HybridEngine``.

    The default route is the MinerU+Qwen-for-code hybrid: MinerU mangles dense
    code indentation (R3 0.44 on AIOS) while Qwen extracts it cleanly (1.00, live
    F5 validation 2026-06-06), so code-dense pages go to Qwen and everything else
    stays on MinerU. A doc with no code routes every page to MinerU — identical
    to the prior pure-MinerU default. ``USE_MINERU_ENGINE=1`` forces pure MinerU.
    """
    if is_mineru_route_enabled():
        return MineruNativeEngine().extract(str(file_path))
    if is_vlm_route_enabled():
        return VlmNativeEngine().extract(str(file_path))
    if is_docling_fast_route_enabled():
        return DoclingFastEngine().extract(str(file_path))
    if is_hybrid_route_enabled():
        return HybridEngine().extract(str(file_path))
    if is_mineru_qwen_hybrid_route_enabled():
        return MineruQwenHybridEngine().extract(str(file_path))
    if _default_route_is_mineru():
        return MineruQwenHybridEngine().extract(str(file_path))
    return HybridEngine().extract(str(file_path))
