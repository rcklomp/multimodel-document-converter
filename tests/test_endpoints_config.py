"""Tests for the central endpoint config (single source of truth).

Pins the resolution contract: defaults match the verified deployment,
env overrides win, URL joining is clean, and unknown roles raise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmrag_v2.endpoints import (  # noqa: E402
    ROLES,
    all_endpoints,
    endpoint,
)


def test_all_five_roles_present():
    assert set(ROLES) == {"chat", "embed", "rerank", "vlm", "mineru"}
    assert len(all_endpoints()) == 5


def test_default_deployment_values(monkeypatch):
    # Clear any env overrides so we test the shipped defaults.
    for role in ROLES:
        for suffix in ("BASE_URL", "MODEL", "API_KEY"):
            monkeypatch.delenv(f"MMRAG_{role.upper()}_{suffix}", raising=False)
    monkeypatch.delenv("MLX_API_KEY", raising=False)

    chat = endpoint("chat")
    assert chat.base_url == "http://10.0.10.239:8000/v1"
    assert chat.model == "RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic"
    assert chat.box == "GX10"

    embed = endpoint("embed")
    assert embed.base_url == "http://10.0.10.246:8000/v1"
    assert embed.box == "Mini"

    assert endpoint("vlm").box == "M5"
    assert endpoint("mineru").base_url == "http://10.0.10.239:8001"


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("MMRAG_CHAT_BASE_URL", "http://example:9000/v1")
    monkeypatch.setenv("MMRAG_CHAT_MODEL", "my-model")
    monkeypatch.setenv("MMRAG_CHAT_API_KEY", "sk-xyz")
    ep = endpoint("chat")
    assert ep.base_url == "http://example:9000/v1"
    assert ep.model == "my-model"
    assert ep.api_key == "sk-xyz"


def test_mlx_api_key_fallback_for_local_roles(monkeypatch):
    monkeypatch.delenv("MMRAG_EMBED_API_KEY", raising=False)
    monkeypatch.delenv("MMRAG_RERANK_API_KEY", raising=False)
    monkeypatch.setenv("MLX_API_KEY", "omlx-secret")
    assert endpoint("embed").api_key == "omlx-secret"
    assert endpoint("rerank").api_key == "omlx-secret"
    # Non-local roles do NOT inherit MLX_API_KEY.
    monkeypatch.delenv("MMRAG_CHAT_API_KEY", raising=False)
    assert endpoint("chat").api_key == ""


def test_url_joining_clean():
    ep = endpoint("chat")
    assert ep.chat_url == "http://10.0.10.239:8000/v1/chat/completions"
    assert endpoint("embed").embeddings_url == "http://10.0.10.246:8000/v1/embeddings"
    assert endpoint("rerank").rerank_url == "http://10.0.10.246:8000/v1/rerank"


def test_unknown_role_raises():
    with pytest.raises(ValueError, match="unknown endpoint role"):
        endpoint("bogus")
