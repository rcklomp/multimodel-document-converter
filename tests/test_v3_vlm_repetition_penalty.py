"""V3 VLM repetition penalty (crucible fix #2, part 2, 2026-06-04).

The crucible found the VLM loops on dense pages (re-emitting the same paragraph
until the token cap). A mild sampling repetition penalty curbs that at the
source. It is sent only when > 1.0, defaulted endpoint-aware, and folded into
the fail-open 400-strip so a backend that rejects it degrades gracefully.

Offline/deterministic: requests.post mocked.
"""

from __future__ import annotations

from types import SimpleNamespace

import requests

from mmrag_v3.engines.vlm_provider import (
    STRUCTURED_OFF,
    VlmProvider,
    VlmProviderConfig,
)


def _resp(status, body=None, text=""):
    return SimpleNamespace(
        status_code=status,
        text=text,
        json=lambda: body if body is not None else {},
    )


def _ok():
    return _resp(200, body={"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]})


def _provider(rp, *, structured=STRUCTURED_OFF):
    return VlmProvider(
        VlmProviderConfig(
            endpoint="http://node.invalid/v1",
            model="m",
            api_key=None,
            timeout_seconds=1.0,
            max_retries=3,
            retry_backoff_seconds=0.0,
            structured_output_mode=structured,
            repetition_penalty=rp,
        )
    )


def test_penalty_sent_when_above_one(monkeypatch):
    seen = []
    monkeypatch.setattr(requests, "post", lambda url, **k: (seen.append(k["json"]), _ok())[1])
    _provider(1.1).describe(b"img", "p")
    assert seen[0]["repetition_penalty"] == 1.1


def test_penalty_not_sent_when_none_or_le_one(monkeypatch):
    seen = []
    monkeypatch.setattr(requests, "post", lambda url, **k: (seen.append(dict(k["json"])), _ok())[1])
    _provider(None).describe(b"img", "p")
    _provider(1.0).describe(b"img", "p")
    assert "repetition_penalty" not in seen[0]
    assert "repetition_penalty" not in seen[1]


def test_400_strips_penalty_only_request(monkeypatch):
    """A 400 strips repetition_penalty and retries even with structured OFF."""
    payloads = []
    calls = {"n": 0}

    def fake_post(url, json=None, headers=None, timeout=None):
        payloads.append(dict(json))
        calls["n"] += 1
        if calls["n"] == 1:
            return _resp(400, text="repetition_penalty not supported")
        return _ok()

    monkeypatch.setattr(requests, "post", fake_post)
    result = _provider(1.1).describe(b"img", "p")
    assert result == "{}"
    assert payloads[0]["repetition_penalty"] == 1.1
    assert "repetition_penalty" not in payloads[1]


def test_from_env_defaults(monkeypatch):
    monkeypatch.delenv("VLM_NATIVE_REPETITION_PENALTY", raising=False)
    monkeypatch.setenv("VLM_NATIVE_ENDPOINT", "http://macbook-pro-m5.lan:8000/v1")
    monkeypatch.setenv("VLM_NATIVE_MODEL", "Qwen3-VL")
    assert VlmProviderConfig.from_env().repetition_penalty == 1.1  # self-hosted default

    monkeypatch.setenv("VLM_NATIVE_ENDPOINT", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("VLM_NATIVE_API_KEY", "k")
    monkeypatch.setenv("VLM_NATIVE_MODEL", "qwen/qwen3-vl-8b-instruct")
    assert VlmProviderConfig.from_env().repetition_penalty is None  # cloud off


def test_from_env_override(monkeypatch):
    monkeypatch.setenv("VLM_NATIVE_ENDPOINT", "http://macbook-pro-m5.lan:8000/v1")
    monkeypatch.setenv("VLM_NATIVE_MODEL", "Qwen3-VL")
    monkeypatch.setenv("VLM_NATIVE_REPETITION_PENALTY", "1.25")
    assert VlmProviderConfig.from_env().repetition_penalty == 1.25
    monkeypatch.setenv("VLM_NATIVE_REPETITION_PENALTY", "off")
    assert VlmProviderConfig.from_env().repetition_penalty is None
    monkeypatch.setenv("VLM_NATIVE_REPETITION_PENALTY", "1.0")  # not > 1.0 -> disabled
    assert VlmProviderConfig.from_env().repetition_penalty is None
