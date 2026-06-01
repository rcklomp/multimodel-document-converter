"""V3 VLM circuit-breaker contract (PLAN_V3.1, 2026-06-01).

Executable guard for the fail-fast rule added after the crucible soak:
an *infrastructure* VLM failure (node offline, connection refused,
connect/read timeout, gateway 502/503/504) must HALT the run, never
silently demote the page to the Docling CPU path. A *semantic* VLM
failure (empty content, malformed JSON, non-retryable 4xx) must still
fall back per-page so one bad page does not kill a document.

Two layers are locked:
  1. Provider classification: VlmProvider.describe raises VlmInfraError
     for transport/gateway faults and the semantic base otherwise.
  2. Router bridge: HybridEngine.extract propagates VlmInfraError WITHOUT
     touching the Docling engine (the circuit breaker), but routes a
     semantic VlmProviderError into the per-page Docling fallback.

Fully offline/deterministic: no real network, no real VLM, no Docling.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import requests

from mmrag_v2.universal.intermediate import create_page
from mmrag_v3.engines.router import HybridEngine
from mmrag_v3.engines.vlm_provider import (
    VlmInfraError,
    VlmProvider,
    VlmProviderConfig,
    VlmProviderError,
)

# --------------------------------------------------------------------------
# Layer 1 - provider classifies infra vs semantic terminal failures
# --------------------------------------------------------------------------


def _provider() -> VlmProvider:
    return VlmProvider(
        VlmProviderConfig(
            endpoint="http://node.invalid/v1",
            model="test-model",
            api_key=None,
            timeout_seconds=1.0,
            max_retries=2,
            retry_backoff_seconds=0.0,
        )
    )


class _Resp:
    def __init__(self, status_code: int, body=None, text: str = ""):
        self.status_code = status_code
        self._body = body
        self.text = text

    def json(self):
        if self._body is None:
            raise ValueError("no json body")
        return self._body


@pytest.mark.parametrize(
    "exc",
    [
        requests.exceptions.ConnectTimeout("connect timed out"),
        requests.exceptions.ReadTimeout("read timed out"),
        requests.exceptions.ConnectionError("connection refused"),
    ],
)
def test_transport_faults_raise_infra(monkeypatch, exc):
    def boom(*a, **k):
        raise exc

    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(VlmInfraError):
        _provider().describe(b"img", "prompt")


@pytest.mark.parametrize("status", [502, 503, 504, 408])
def test_gateway_statuses_raise_infra(monkeypatch, status):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(status, text="down"))
    with pytest.raises(VlmInfraError):
        _provider().describe(b"img", "prompt")


def test_rate_limit_is_semantic_not_infra(monkeypatch):
    # 429 is a quota/app condition on a *live* node, not an outage.
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(429, text="slow down"))
    with pytest.raises(VlmProviderError) as ei:
        _provider().describe(b"img", "prompt")
    assert not isinstance(ei.value, VlmInfraError)


def test_non_retryable_4xx_is_semantic_not_infra(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(400, text="bad request"))
    with pytest.raises(VlmProviderError) as ei:
        _provider().describe(b"img", "prompt")
    assert not isinstance(ei.value, VlmInfraError)


def test_malformed_200_is_semantic_not_infra(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(200, body={"unexpected": "shape"}))
    with pytest.raises(VlmProviderError) as ei:
        _provider().describe(b"img", "prompt")
    assert not isinstance(ei.value, VlmInfraError)


# --------------------------------------------------------------------------
# Layer 2 - router bridge: infra propagates (no fallback); semantic falls back
# --------------------------------------------------------------------------


def _one_page_pdf(tmp_path):
    import fitz

    pdf = tmp_path / "one.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf))
    doc.close()
    return pdf


def _force_vlm_route(monkeypatch):
    """Route every page to the VLM branch and stub the renderer."""
    monkeypatch.setattr(HybridEngine, "_classify_page", lambda self, page: ("vlm", "forced"))
    monkeypatch.setattr(HybridEngine, "_render_page_png", staticmethod(lambda page: (b"x", 10, 10)))


def test_router_infra_error_trips_breaker_without_docling(tmp_path, monkeypatch):
    """VlmInfraError propagates out of extract() and Docling is never touched."""
    _force_vlm_route(monkeypatch)

    class _InfraProvider:
        def describe(self, *a, **k):
            raise VlmInfraError("node offline")

    class _DoclingMustNotRun:
        def extract(self, *a, **k):
            raise AssertionError(
                "docling fallback invoked for an INFRA error - circuit breaker "
                "is not wired; an offline node would masquerade as CPU baselines"
            )

    engine = HybridEngine(
        vlm_engine=SimpleNamespace(_provider=_InfraProvider()),
        docling_engine=_DoclingMustNotRun(),
    )

    with pytest.raises(VlmInfraError):
        engine.extract(str(_one_page_pdf(tmp_path)))


def test_router_semantic_error_falls_back_to_docling(tmp_path, monkeypatch):
    """A semantic VlmProviderError still demotes the page to Docling."""
    _force_vlm_route(monkeypatch)

    class _SemanticProvider:
        def describe(self, *a, **k):
            raise VlmProviderError("malformed JSON")

    fallback_page = create_page(page_number=1, elements=[], dimensions=(612, 792))

    called = {"docling": False}

    class _FakeDocling:
        def extract(self, *a, **k):
            called["docling"] = True
            return SimpleNamespace(pages=[fallback_page])

    engine = HybridEngine(
        vlm_engine=SimpleNamespace(_provider=_SemanticProvider()),
        docling_engine=_FakeDocling(),
    )

    doc = engine.extract(str(_one_page_pdf(tmp_path)))

    assert called["docling"], "semantic VLM failure should fall back to Docling"
    assert any(c == "docling_fallback" for _, c, _ in engine.last_routing_decisions)
    assert [p.page_number for p in doc.pages] == [1]
