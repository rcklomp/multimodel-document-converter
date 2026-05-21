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


# Set to "dashscope" or "omlx" after the Phase 1 soak resolves the
# bake-off. Until then, env var or explicit arg is required.
_COMPILE_DEFAULT: str | None = None


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
