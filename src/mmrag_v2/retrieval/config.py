"""v2.12 retrieval — backend selection factory.

The production retrieval path picks a reranker via:

  1. Explicit `name=` arg to `get_reranker()`, OR
  2. `RERANKER_BACKEND` env var, OR
  3. The compile-time default below.

Supported names:

  "dashscope"  — Dashscope intl `gte-rerank` (cloud)
  "omlx"       — local `gte-reranker-modernbert-base-mlx` (omlx-server)
  "null"       — no-op pass-through (for tests / fallback)

The compile-time default is set after the Phase 1 soak picks the
winner; until then it's `None` so the env var or explicit arg
determines the choice (avoids silently picking a backend before
the soak data exists).
"""
from __future__ import annotations

import os

from mmrag_v2.retrieval.reranker import (
    DashscopeReranker,
    LocalOmlxReranker,
    Reranker,
    RerankerError,
    _NullReranker,
)


# Production reranker chosen by the v2.12 Phase 1 shootout (2026-05-21).
# v2.12 Phase 2 then promoted *hybrid* (BM25 + dense + RRF + rerank)
# as the production retrieval flow — same reranker, wider candidate
# set. See `retrieve_hybrid_reranked()` in pipeline.py.
#
# Phase 1 (dense + rerank only) -> Phase 2 (hybrid + rerank):
#   Recall@1 chunk    35.5% (v2.11) → 61.8% (P1) → 67.8% (P2)
#   Recall@5 chunk    66.8% → 81.3% → 90.2%   <-- STRETCH target met
#   Recall@5 doc      91.7% → 95.2% → 98.6%   <-- STRETCH target met
#   Relevance         59.3% → 78.3% → 82.1%
#   Faithfulness      50.6% → 69.4% → 72.6%   <-- floor target met
#   Format            89.8% → 89.0% → 88.4%
# See docs/DECISIONS.md "v2.12 Phase 1/2 Outcome".
_COMPILE_DEFAULT: str = "omlx"


def get_reranker(
    name: str | None = None,
    *,
    api_key: str | None = None,
    model: str | None = None,
    url: str | None = None,
) -> Reranker:
    """Construct a reranker by backend name.

    Precedence: `name` arg > `RERANKER_BACKEND` env var > compile-time
    default. Raises ValueError if no backend is resolvable.

    Per-backend kwargs (`api_key`, `model`, `url`) flow through to the
    constructor; if omitted, each backend reads its own env vars
    (`DASHSCOPE_API_KEY` or `MLX_API_KEY`) and uses its DEFAULT_MODEL /
    DEFAULT_URL.
    """
    backend = (
        name
        or os.environ.get("RERANKER_BACKEND")
        or _COMPILE_DEFAULT
    )
    if not backend:
        raise ValueError(
            "No reranker backend resolved. Pass `name=` arg, set "
            "RERANKER_BACKEND env var, or wait for the Phase 1 soak "
            "to set the compile-time default in mmrag_v2.retrieval.config."
        )
    backend = backend.lower()
    if backend == "dashscope":
        return DashscopeReranker(api_key=api_key, model=model, url=url)
    if backend == "omlx":
        return LocalOmlxReranker(api_key=api_key, model=model, url=url)
    if backend == "null":
        return _NullReranker()
    raise ValueError(
        f"Unknown reranker backend: {backend!r}. "
        f"Supported: 'dashscope', 'omlx', 'null'."
    )
