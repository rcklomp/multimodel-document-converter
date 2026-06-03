"""V3 guided/constrained JSON-decode contract (Charter Blocker A / A3).

The structural fix for the malformation class: upgrade the provider from the
weak ``response_format={"type":"json_object"}`` hint to a full ``json_schema``
constrained decode of the UIR element schema (mlx-vlm / vLLM both expose this).
A constrained decode guarantees a well-formed JSON object, eliminating the
malformation half of Blocker A; it PAIRS with A1+A4 because a constrained
decode can still hit the token cap (valid-but-incomplete).

Fail-open requirement also locked here: a backend that does not support the
structured-output field returns 400; the provider strips the field and retries
with the prompt-only contract rather than hard-failing the page.

Fully offline/deterministic: ``requests.post`` is mocked. Bodies are DENSE
multi-element pages (the real workload).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import requests

from mmrag_v3.engines.vlm_provider import (
    STRUCTURED_GUIDED_JSON,
    STRUCTURED_JSON_OBJECT,
    STRUCTURED_JSON_SCHEMA,
    STRUCTURED_OFF,
    UIR_PAGE_JSON_SCHEMA,
    VlmProvider,
    VlmProviderConfig,
)


def _dense_body() -> str:
    elements = [
        {
            "type": "text",
            "content": f"Paragraph {i}: " + ("dense body text " * 8).strip(),
            "bbox": [80, 100 + i * 20, 1500, 118 + i * 20],
            "confidence": 0.95,
        }
        for i in range(30)
    ]
    return json.dumps(
        {
            "page_number": 1,
            "width": 1654,
            "height": 2339,
            "classification": "digital",
            "elements": elements,
        }
    )


def _resp(status_code, body=None, text=""):
    return SimpleNamespace(
        status_code=status_code,
        text=text,
        json=lambda: body if body is not None else (_ for _ in ()).throw(ValueError("no body")),
    )


def _ok(content):
    return _resp(
        200, body={"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
    )


def _provider(mode: str, *, send_rf: bool = True) -> VlmProvider:
    return VlmProvider(
        VlmProviderConfig(
            endpoint="http://node.invalid/v1",
            model="m",
            api_key=None,
            timeout_seconds=1.0,
            max_retries=3,
            retry_backoff_seconds=0.0,
            structured_output_mode=mode,
            send_response_format=send_rf,
        )
    )


def test_json_schema_mode_sends_constrained_decode(monkeypatch):
    payloads = []

    def fake_post(url, json=None, headers=None, timeout=None):
        payloads.append(json)
        return _ok(_dense_body())

    monkeypatch.setattr(requests, "post", fake_post)
    _provider(STRUCTURED_JSON_SCHEMA).describe(b"img", "p")

    rf = payloads[0]["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"] is UIR_PAGE_JSON_SCHEMA
    # The schema constrains to an object whose elements are typed.
    assert rf["json_schema"]["schema"]["required"] == ["elements"]


def test_guided_json_mode_sends_vllm_field(monkeypatch):
    payloads = []
    monkeypatch.setattr(
        requests, "post", lambda url, **k: (payloads.append(k["json"]), _ok(_dense_body()))[1]
    )
    _provider(STRUCTURED_GUIDED_JSON).describe(b"img", "p")
    assert payloads[0]["guided_json"] is UIR_PAGE_JSON_SCHEMA
    assert "response_format" not in payloads[0]


def test_json_object_mode_is_legacy_hint(monkeypatch):
    payloads = []
    monkeypatch.setattr(
        requests, "post", lambda url, **k: (payloads.append(k["json"]), _ok(_dense_body()))[1]
    )
    _provider(STRUCTURED_JSON_OBJECT).describe(b"img", "p")
    assert payloads[0]["response_format"] == {"type": "json_object"}


def test_off_mode_sends_no_structured_field(monkeypatch):
    payloads = []
    monkeypatch.setattr(
        requests, "post", lambda url, **k: (payloads.append(k["json"]), _ok(_dense_body()))[1]
    )
    _provider(STRUCTURED_OFF).describe(b"img", "p")
    assert "response_format" not in payloads[0]
    assert "guided_json" not in payloads[0]


def test_400_on_structured_strips_and_retries(monkeypatch):
    """A 400 on the schema field -> strip it and retry prompt-only (fail-open)."""
    payloads = []
    calls = {"n": 0}

    def fake_post(url, json=None, headers=None, timeout=None):
        # Snapshot: the provider mutates the same payload dict across retries.
        payloads.append(dict(json))
        calls["n"] += 1
        if calls["n"] == 1:
            return _resp(400, text="response_format json_schema not supported")
        return _ok(_dense_body())

    monkeypatch.setattr(requests, "post", fake_post)
    result = _provider(STRUCTURED_JSON_SCHEMA).describe(b"img", "p")

    assert json.loads(result)["elements"]  # succeeded on the retry
    # First attempt carried the schema; the retry stripped it.
    assert "response_format" in payloads[0]
    assert "response_format" not in payloads[1]
    assert "guided_json" not in payloads[1]


def test_from_env_defaults_self_hosted_to_json_schema(monkeypatch):
    monkeypatch.setenv("VLM_NATIVE_ENDPOINT", "http://macbook-pro-m5.lan:8000/v1")
    monkeypatch.setenv("VLM_NATIVE_MODEL", "Qwen3-VL")
    monkeypatch.delenv("VLM_NATIVE_STRUCTURED_OUTPUT", raising=False)
    cfg = VlmProviderConfig.from_env()
    assert cfg.structured_output_mode == STRUCTURED_JSON_SCHEMA


def test_from_env_defaults_openrouter_to_json_object(monkeypatch):
    monkeypatch.setenv("VLM_NATIVE_ENDPOINT", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("VLM_NATIVE_MODEL", "qwen/qwen3-vl-8b-instruct")
    monkeypatch.setenv("VLM_NATIVE_API_KEY", "k")
    monkeypatch.delenv("VLM_NATIVE_STRUCTURED_OUTPUT", raising=False)
    cfg = VlmProviderConfig.from_env()
    assert cfg.structured_output_mode == STRUCTURED_JSON_OBJECT


def test_from_env_explicit_override_wins(monkeypatch):
    monkeypatch.setenv("VLM_NATIVE_ENDPOINT", "http://macbook-pro-m5.lan:8000/v1")
    monkeypatch.setenv("VLM_NATIVE_MODEL", "Qwen3-VL")
    monkeypatch.setenv("VLM_NATIVE_STRUCTURED_OUTPUT", "off")
    cfg = VlmProviderConfig.from_env()
    assert cfg.structured_output_mode == STRUCTURED_OFF
