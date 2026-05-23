"""HyDE module unit tests (mock-driven).

Pin the HyDE generation interface + fallback semantics without
making live network calls. Live integration is exercised by soak
runs.

Coverage:
- v2.12 Phase 3: cloud Dashscope `qwen-max` provider (default).
- v2.14 Phase 4a: local vLLM provider (`provider="vllm"`). Default
  endpoint is the GX10 at `http://10.0.10.239:8000` serving
  `Qwen/Qwen3.6-27B-FP8` with native MTP=3 (as of 2026-05-23).
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


# ── v2.14 Phase 4a: local vllm HyDE provider ─────────────────────────────────


def test_generate_hypothetical_answer_vllm_provider_uses_local_url(monkeypatch):
    """provider='vllm' must POST to the VLLM_DEFAULT_URL, not the
    Dashscope endpoint, and must NOT require DASHSCOPE_API_KEY."""
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)

    captured = {}

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = req.data
        # Authorization header should be absent when no key is set.
        captured["has_auth"] = req.has_header("Authorization")
        return _fake_dashscope_response("Hypothetical answer from local vllm.")

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        out = generate_hypothetical_answer("test query", provider="vllm")

    assert out == "Hypothetical answer from local vllm."
    assert "10.0.10.239" in captured["url"]
    assert "/v1/chat/completions" in captured["url"]
    assert captured["has_auth"] is False
    body = json.loads(captured["body"].decode("utf-8"))
    assert body["model"] == "RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic"


def test_generate_hypothetical_answer_vllm_provider_honors_overrides(monkeypatch):
    """Explicit url + model overrides win over the vllm defaults."""
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    captured = {}

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = req.data
        return _fake_dashscope_response("override hypothesis")

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        out = generate_hypothetical_answer(
            "test query",
            provider="vllm",
            url="http://10.0.0.99:1234/v1/chat/completions",
            model="my-custom-model",
        )

    assert out == "override hypothesis"
    assert captured["url"] == "http://10.0.0.99:1234/v1/chat/completions"
    body = json.loads(captured["body"].decode("utf-8"))
    assert body["model"] == "my-custom-model"


def test_generate_hypothetical_answer_vllm_uses_bearer_when_api_key_set(monkeypatch):
    """If VLLM_API_KEY env var or explicit api_key is set, the call
    must include Authorization: Bearer <key>. This is for gated
    deployments where vLLM is behind an auth proxy."""
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("VLLM_API_KEY", "test-bearer-token-123")

    captured = {}

    def _fake_urlopen(req, timeout=None):
        captured["auth"] = req.get_header("Authorization")
        return _fake_dashscope_response("auth ok")

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        generate_hypothetical_answer("test query", provider="vllm")

    assert captured["auth"] == "Bearer test-bearer-token-123"


def test_generate_hypothetical_answer_dashscope_provider_still_default(monkeypatch):
    """Default provider remains dashscope — no behavior change for existing callers."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")

    captured = {}

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        return _fake_dashscope_response("dashscope hypothesis")

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        # No provider= arg → must default to dashscope
        out = generate_hypothetical_answer("test query")

    assert out == "dashscope hypothesis"
    assert "dashscope" in captured["url"]


def test_generate_hypothetical_answer_vllm_disables_thinking_mode(monkeypatch):
    """vllm payload must include `chat_template_kwargs.enable_thinking=False`.

    The GX10 endpoint runs Qwen3 with `--reasoning-parser qwen3`, which
    defaults to thinking mode. Thinking routes the answer into
    `message.reasoning` and starves `message.content` of token budget,
    so without this flag the parser sees empty content and HydeError fires.
    See 2026-05-23 Phase 0 re-cal debug — 510/518 parse failures from
    this exact cause.
    """
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)

    captured = {}

    def _fake_urlopen(req, timeout=None):
        captured["body"] = req.data
        return _fake_dashscope_response("ok")

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        generate_hypothetical_answer("test query", provider="vllm")

    body = json.loads(captured["body"].decode("utf-8"))
    assert body.get("chat_template_kwargs") == {"enable_thinking": False}


def test_generate_hypothetical_answer_dashscope_omits_thinking_kwargs(monkeypatch):
    """Dashscope (cloud qwen-max) is NOT Qwen3 thinking-capable and must
    not receive the chat_template_kwargs extension — keep the dashscope
    payload identical to its pre-Phase-4a shape."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")

    captured = {}

    def _fake_urlopen(req, timeout=None):
        captured["body"] = req.data
        return _fake_dashscope_response("ok")

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        generate_hypothetical_answer("test query")

    body = json.loads(captured["body"].decode("utf-8"))
    assert "chat_template_kwargs" not in body


def test_generate_with_fallback_vllm_provider_falls_back_on_error(monkeypatch):
    """vllm-provider fallback returns the literal query when BOTH the
    local vllm AND the Dashscope qwen3-max cloud fallback fail (no API
    key here means the cloud-fallback retry raises immediately)."""
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    import urllib.error
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("simulated")):
        with patch("mmrag_v2.retrieval.hyde.time.sleep"):
            result = generate_with_fallback("the literal query", provider="vllm")
    assert result == "the literal query"


# ── Phase 4 Resilience: qwen3-max cloud fallback on vllm primary failure ─


def test_generate_with_fallback_vllm_failure_falls_through_to_qwen3_max(monkeypatch):
    """When the local vllm endpoint fails AND DASHSCOPE_API_KEY is set,
    generate_with_fallback must retry against Dashscope qwen3-max and
    return the cloud response (Phase 4 Resilience scope, designated
    2026-05-23)."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    import urllib.error

    cloud_calls: list[dict] = []

    def _fake_urlopen(req, timeout=None):
        # First call (local vllm) raises; subsequent calls (cloud retry)
        # return a successful Dashscope-shape response.
        if "10.0.10.239" in req.full_url:
            raise urllib.error.URLError("simulated local outage")
        body = json.loads(req.data.decode("utf-8"))
        cloud_calls.append({"url": req.full_url, "model": body.get("model")})
        return _fake_dashscope_response("qwen3-max fallback hypothesis")

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        with patch("mmrag_v2.retrieval.hyde.time.sleep"):
            result = generate_with_fallback("test query", provider="vllm")

    assert result == "qwen3-max fallback hypothesis"
    assert len(cloud_calls) == 1, "cloud retry must be a single call, no retries-of-retries"
    assert cloud_calls[0]["model"] == "qwen3-max"
    assert "dashscope" in cloud_calls[0]["url"]


def test_generate_with_fallback_vllm_opt_out_skips_cloud_fallback(monkeypatch):
    """Passing cloud_fallback_model=None preserves the pre-Phase-4-Resilience
    behavior: vllm failure → literal query immediately, no Dashscope retry."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")  # would succeed if tried
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    import urllib.error

    def _fake_urlopen(req, timeout=None):
        if "10.0.10.239" in req.full_url:
            raise urllib.error.URLError("simulated local outage")
        raise AssertionError("cloud must NOT be called when cloud_fallback_model=None")

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        with patch("mmrag_v2.retrieval.hyde.time.sleep"):
            result = generate_with_fallback(
                "literal", provider="vllm", cloud_fallback_model=None,
            )
    assert result == "literal"


def test_generate_with_fallback_dashscope_primary_does_not_retry_cloud(monkeypatch):
    """When the primary provider is dashscope (not vllm), a failure
    must NOT trigger the qwen3-max fallback — same-provider retry
    doesn't help when dashscope itself is unreachable. Degrades
    straight to literal query."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    import urllib.error

    call_count = {"n": 0}

    def _fake_urlopen(req, timeout=None):
        call_count["n"] += 1
        raise urllib.error.URLError("simulated dashscope outage")

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        with patch("mmrag_v2.retrieval.hyde.time.sleep"):
            result = generate_with_fallback("literal")  # default provider=dashscope
    assert result == "literal"
    # Only the primary attempts (default retries=3) — no extra cloud-fallback layer
    assert call_count["n"] == 3, (
        "dashscope primary failure must not trigger a second dashscope call "
        f"via the cloud-fallback path; got {call_count['n']} calls (expected 3 retries)"
    )
