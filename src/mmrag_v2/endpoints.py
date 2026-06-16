"""Single source of truth for every model endpoint (OpenAI-compatible).

Each ROLE maps to an `Endpoint` (base_url, model, api_key, box), resolved
from environment variables with the current verified deployment values as
defaults. Change a box's address or the model it serves in ONE place (or
via env) instead of editing scattered client code.

Design intent (2026-06-16): the models stay DISTRIBUTED across their own
machines - this module does NOT merge them onto one server. It only unifies
how the application *addresses* them, and standardizes on the OpenAI wire
format. Optimal placement across the available hardware:

    role    box   default address              what / why
    ----    ----  -------------------------     ---------------------------
    chat    GX10  10.0.10.239:8000/v1           Qwen2.5-14B-FP8 generation/
                                                HyDE/judge (Blackwell throughput)
    embed   Mini  10.0.10.246:8000/v1           Qwen3-Embedding-8B (always-on
                                                serving core, co-located w/ Qdrant)
    rerank  Mini  10.0.10.246:8000/v1           ModernBERT reranker (same)
    vlm     M5    10.0.10.235:8000/v1           Qwen3-VL-8B extraction
                                                (INGESTION ONLY - M5 is mobile,
                                                never in the live query path)
    mineru  GX10  10.0.10.239:8001              MinerU2.5 batch extraction primary

The live query path depends only on the always-on boxes (Mini Qdrant+embed+
rerank, GX10 generation); nothing live depends on the mobile M5.

Env override convention (per role X in {CHAT, EMBED, RERANK, VLM, MINERU}):
    MMRAG_<X>_BASE_URL   OpenAI-style base, e.g. http://host:port/v1
    MMRAG_<X>_MODEL      served model id
    MMRAG_<X>_API_KEY    bearer token (most local servers need none)
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# Default deployment (verified 2026-06-16). Tuple: (base_url, model, box).
_DEFAULTS: dict[str, tuple[str, str, str]] = {
    "chat": (
        "http://10.0.10.239:8000/v1",
        "RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic",
        "GX10",
    ),
    "embed": (
        "http://10.0.10.246:8000/v1",
        "Qwen3-Embedding-8B-mxfp8",
        "Mini",
    ),
    "rerank": (
        "http://10.0.10.246:8000/v1",
        "gte-reranker-modernbert-base-mlx",
        "Mini",
    ),
    "vlm": (
        "http://10.0.10.235:8000/v1",
        "mlx-community/Qwen3-VL-8B-Instruct-8bit",
        "M5",
    ),
    "mineru": (
        "http://10.0.10.239:8001",
        "MinerU2.5-2509-1.2B",
        "GX10",
    ),
}

ROLES = tuple(_DEFAULTS.keys())


@dataclass(frozen=True)
class Endpoint:
    """One OpenAI-compatible model endpoint."""

    role: str
    base_url: str
    model: str
    api_key: str = ""
    box: str = ""

    def url(self, path: str) -> str:
        """Full URL for an OpenAI route, e.g. url("/chat/completions").

        Joins onto `base_url` without doubling slashes.
        """
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

    @property
    def chat_url(self) -> str:
        return self.url("chat/completions")

    @property
    def embeddings_url(self) -> str:
        return self.url("embeddings")

    @property
    def rerank_url(self) -> str:
        return self.url("rerank")


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v.strip() if v and v.strip() else default


def endpoint(role: str) -> Endpoint:
    """Resolve an endpoint by role, applying env overrides over the defaults.

    Roles: chat, embed, rerank, vlm, mineru. Raises ValueError otherwise.
    The api_key default falls back to MLX_API_KEY for the Mini-hosted
    local-MLX roles (embed, rerank) so existing env keeps working.
    """
    role = role.lower()
    if role not in _DEFAULTS:
        raise ValueError(
            f"unknown endpoint role: {role!r} (known: {', '.join(ROLES)})"
        )
    base_default, model_default, box = _DEFAULTS[role]
    prefix = f"MMRAG_{role.upper()}_"
    key_default = os.environ.get("MLX_API_KEY", "") if role in ("embed", "rerank") else ""
    return Endpoint(
        role=role,
        base_url=_env(prefix + "BASE_URL", base_default),
        model=_env(prefix + "MODEL", model_default),
        api_key=_env(prefix + "API_KEY", key_default),
        box=box,
    )


def all_endpoints() -> list[Endpoint]:
    """Every configured endpoint, for health checks / introspection."""
    return [endpoint(r) for r in ROLES]


def openai_client(role: str):
    """An `openai.OpenAI` client bound to the role's base_url + key.

    The OpenAI SDK requires a non-empty api_key even for keyless local
    servers, so a placeholder is substituted when none is configured.
    """
    from openai import OpenAI

    ep = endpoint(role)
    return OpenAI(base_url=ep.base_url, api_key=ep.api_key or "not-needed")
