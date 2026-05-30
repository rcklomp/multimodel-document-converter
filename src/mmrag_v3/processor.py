"""V3 Phase C processor — engine-agnostic UIR producer.

Exposes a single ``extract`` function that returns a
``mmrag_v2.universal.intermediate.UniversalDocument``. Today the only
route is the vision-native VLM engine; future routes (e.g., direct
PDF parsing, EPUB engines) plug in via the same contract.

Routing is controlled by the ``USE_VLM_ENGINE`` environment variable
so the Identity Gate (and any other batch driver) can flip the engine
without code edits.

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
from .engines.router import HybridEngine
from .engines.vlm_native import VlmNativeEngine


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip() in {"1", "true", "TRUE", "yes"}


def is_vlm_route_enabled() -> bool:
    """Return True when the VLM-native engine is force-selected via env."""
    return _env_flag("USE_VLM_ENGINE")


def is_docling_fast_route_enabled() -> bool:
    """Return True when the fast-Docling engine is force-selected via env."""
    return _env_flag("USE_DOCLING_FAST")


def extract(file_path: Union[str, "os.PathLike[str]"]) -> UniversalDocument:
    """Run the Phase C pipeline and return a v2-UIR document.

    Routing precedence:
        * ``USE_VLM_ENGINE=1``      → all pages through ``VlmNativeEngine``
        * ``USE_DOCLING_FAST=1``    → all pages through ``DoclingFastEngine``
        * default                   → ``HybridEngine`` (cost-optimizer router)
    """
    if is_vlm_route_enabled():
        return VlmNativeEngine().extract(str(file_path))
    if is_docling_fast_route_enabled():
        return DoclingFastEngine().extract(str(file_path))
    return HybridEngine().extract(str(file_path))
