"""v2.12 Phase 3 — HyDE module unit tests (mock-driven).

Pin the HyDE generation interface + fallback semantics without
making live Dashscope calls. Live integration is exercised by the
Phase 3 soak.
"""
from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmrag_v2.retrieval.hyde import (
    HydeError,
    generate_hypothetical_answer,
    generate_with_fallback,
)


def _fake_dashscope_response(content: str) -> BytesIO:
    body = {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 80, "total_tokens": 130},
    }
    return BytesIO(json.dumps(body).encode("utf-8"))


def test_generate_hypothetical_answer_returns_content(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")

    fake_resp = _fake_dashscope_response("MCP standardizes how applications provide context to LLMs.")
    with patch("urllib.request.urlopen", return_value=fake_resp):
        out = generate_hypothetical_answer("What is the Model Context Protocol?")
    assert "MCP" in out
    assert "LLMs" in out


def test_generate_hypothetical_answer_requires_api_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(HydeError, match="api_key"):
        generate_hypothetical_answer("test query")


def test_generate_hypothetical_answer_empty_choices_raises(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    fake_resp = BytesIO(b'{"choices": []}')
    with patch("urllib.request.urlopen", return_value=fake_resp):
        with pytest.raises(HydeError, match="no .choices."):
            generate_hypothetical_answer("test query")


def test_generate_hypothetical_answer_empty_content_raises(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    fake_resp = BytesIO(b'{"choices": [{"message": {"content": ""}}]}')
    with patch("urllib.request.urlopen", return_value=fake_resp):
        with pytest.raises(HydeError, match="empty"):
            generate_hypothetical_answer("test query")


def test_generate_with_fallback_returns_literal_on_error(monkeypatch):
    """The fallback variant must never raise — it returns the literal
    query string if HyDE fails."""
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    query = "the literal query text"
    result = generate_with_fallback(query)
    assert result == query


def test_generate_with_fallback_returns_hypothesis_on_success(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    fake_resp = _fake_dashscope_response("Some plausible answer.")
    with patch("urllib.request.urlopen", return_value=fake_resp):
        result = generate_with_fallback("test query")
    assert result == "Some plausible answer."


def test_generate_with_fallback_handles_network_error(monkeypatch):
    """On a network error after retries, fallback returns the literal query."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    import urllib.error
    # Mock urlopen to raise URLError every call. The retries (default 3)
    # will all fail; generate_with_fallback returns literal.
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("simulated")):
        # Patch time.sleep to make retries instant (default backoff = 2^attempt)
        with patch("mmrag_v2.retrieval.hyde.time.sleep"):
            result = generate_with_fallback("the literal query")
    assert result == "the literal query"
