"""V3 adaptive output-budget contract (Charter Blocker A / A2, 2026-06-03).

The dense-page soak truncated at the old 4096 default. A2 raises the floor to
8192 and scales the per-page output budget by a cheap text-density estimate so
a dense magazine/manual page gets headroom while a sparse page stays at the
floor - all bounded by the OOM-safety cap shared with the A1 escalation.

Three layers are locked, all offline/deterministic (no network, no VLM):
  1. estimate_output_budget scales with page text volume (dense >> sparse).
  2. VlmProvider.describe floors at the config default and caps at
     TRUNCATION_ESCALATION_CAP for any per-call override.
  3. The VLM engine wires a DENSE page's scaled budget through to the provider.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import fitz
import requests

from mmrag_v3.engines.vlm_native import VlmNativeEngine, estimate_output_budget
from mmrag_v3.engines.vlm_provider import (
    TRUNCATION_ESCALATION_CAP,
    VlmProvider,
    VlmProviderConfig,
)


class _FakePage:
    """Minimal stand-in exposing only the get_text the estimator reads."""

    def __init__(self, text: str, *, raises: bool = False) -> None:
        self._text = text
        self._raises = raises

    def get_text(self, kind: str) -> str:
        if self._raises:
            raise RuntimeError("unreadable page")
        return self._text


# --------------------------------------------------------------------------
# Layer 1 - estimator scales with density
# --------------------------------------------------------------------------


def test_estimate_scales_with_density():
    sparse = estimate_output_budget(_FakePage("a short caption"))
    dense = estimate_output_budget(_FakePage("dense body text " * 1500))  # ~24k chars
    # A genuinely dense page must estimate far above the 8192 floor so it
    # earns headroom; a sparse page estimates near zero (-> floored downstream).
    assert dense > 8192
    assert sparse < 1000
    assert dense > sparse * 10


def test_estimate_unreadable_page_returns_zero():
    assert estimate_output_budget(_FakePage("", raises=True)) == 0
    assert estimate_output_budget(_FakePage("")) == 0


# --------------------------------------------------------------------------
# Layer 2 - provider floors and caps the per-call override
# --------------------------------------------------------------------------


def _provider() -> VlmProvider:
    return VlmProvider(
        VlmProviderConfig(
            endpoint="http://node.invalid/v1",
            model="m",
            api_key=None,
            timeout_seconds=1.0,
            max_retries=1,
            retry_backoff_seconds=0.0,
            max_completion_tokens=8192,
        )
    )


def _stop_resp(captured, kwargs):
    captured["max_tokens"] = kwargs["json"]["max_tokens"]
    return SimpleNamespace(
        status_code=200,
        text="",
        json=lambda: {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]},
    )


def test_provider_floors_below_default(monkeypatch):
    captured = {}
    monkeypatch.setattr(requests, "post", lambda url, **k: _stop_resp(captured, k))
    _provider().describe(b"img", "p", max_tokens=2000)
    assert captured["max_tokens"] == 8192  # floored to the config default


def test_provider_passes_scaled_budget(monkeypatch):
    captured = {}
    monkeypatch.setattr(requests, "post", lambda url, **k: _stop_resp(captured, k))
    _provider().describe(b"img", "p", max_tokens=12000)
    assert captured["max_tokens"] == 12000


def test_provider_caps_at_oom_ceiling(monkeypatch):
    captured = {}
    monkeypatch.setattr(requests, "post", lambda url, **k: _stop_resp(captured, k))
    _provider().describe(b"img", "p", max_tokens=10_000_000)
    assert captured["max_tokens"] == TRUNCATION_ESCALATION_CAP


def test_provider_none_uses_default(monkeypatch):
    captured = {}
    monkeypatch.setattr(requests, "post", lambda url, **k: _stop_resp(captured, k))
    _provider().describe(b"img", "p")
    assert captured["max_tokens"] == 8192


# --------------------------------------------------------------------------
# Layer 3 - real wiring: a dense page's budget reaches the provider
# --------------------------------------------------------------------------


def _dense_pdf(tmp_path):
    pdf = tmp_path / "dense.pdf"
    doc = fitz.open()
    page = doc.new_page(width=2000, height=3000)
    page.insert_textbox(
        fitz.Rect(10, 10, 1990, 2990),
        ("dense body text " * 1200),
        fontsize=7,
    )
    doc.save(str(pdf))
    doc.close()
    return pdf


class _RecordingProvider:
    """Fake provider that records the max_tokens it is handed."""

    def __init__(self):
        self.seen_max_tokens = []

    def describe(self, image_bytes, prompt, *, mime="image/png", max_tokens=None):
        self.seen_max_tokens.append(max_tokens)
        return json.dumps(
            {
                "page_number": 1,
                "width": 100,
                "height": 100,
                "classification": "digital",
                "elements": [
                    {
                        "type": "text",
                        "content": "ok",
                        "bbox": [1, 1, 50, 50],
                        "confidence": 0.9,
                    }
                ],
            }
        )


def test_dense_page_budget_reaches_provider(tmp_path):
    prov = _RecordingProvider()
    engine = VlmNativeEngine(provider=prov)
    engine.extract(str(_dense_pdf(tmp_path)))
    assert prov.seen_max_tokens, "provider was never called"
    # The dense page must hand the provider a scaled budget above the floor,
    # proving the estimator -> describe(max_tokens=...) wiring is live.
    assert prov.seen_max_tokens[0] > 8192
