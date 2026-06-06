"""MineruQwenHybridEngine — per-page MinerU+Qwen-for-code routing.

MinerU mangles dense code indentation (R3 0.44 on AIOS); Qwen extracts it
cleanly (1.00, live F5 validation). This engine routes code-dense pages
(monospace ratio >= threshold) to the Qwen VLM and everything else to MinerU.
These offline tests pin the routing + merge with a real 2-page PDF (one Courier
code page, one Helvetica prose page) and injected engine mocks — no VLM server,
no MinerU server.
"""

from __future__ import annotations

import fitz

from mmrag_v2.universal.intermediate import (
    ElementType,
    ExtractionMethod,
    create_element,
    create_page,
)
from mmrag_v3.engines import vlm_native as V
from mmrag_v3.engines.router import MineruQwenHybridEngine
from mmrag_v3.engines.vlm_native import VlmNativeEngine


def _make_pdf(tmp_path):
    doc = fitz.open()
    code_page = doc.new_page()
    code_page.insert_text(
        (72, 72),
        "\n".join(["class A:", "    def f(self):", "        return self.x"] * 6),
        fontname="courier",
        fontsize=9,
    )
    prose_page = doc.new_page()
    prose_page.insert_text(
        (72, 72),
        "The quick brown fox jumps over the lazy dog. " * 20,
        fontname="helv",
        fontsize=11,
    )
    path = tmp_path / "doc.pdf"
    doc.save(str(path))
    doc.close()
    return str(path)


class _FakeMinerUClient:
    def __init__(self):
        self.calls = 0

    def two_step_extract(self, image):
        self.calls += 1
        return [{"type": "text", "bbox": [0.1, 0.1, 0.9, 0.2], "content": "FROM_MINERU"}]


class _FakeMinerUEngine:
    def __init__(self):
        self.client = _FakeMinerUClient()
        self.render_calls = []

    def _render_page(self, page):
        self.render_calls.append(page.number)
        return (b"img", 100, 100)


def _fake_page_from_payload(payload, fallback_page_number, pixel_width, pixel_height):
    el = create_element(
        element_type=ElementType.TEXT,
        content="FROM_QWEN",
        bbox=[10, 10, 90, 20],
        confidence=0.9,
        extraction_method=ExtractionMethod.VLM,
        element_index=0,
    )
    return create_page(
        page_number=fallback_page_number,
        elements=[el],
        dimensions=(pixel_width, pixel_height),
        classification=None,
    )


def _install_vlm_mocks(monkeypatch):
    vlm = VlmNativeEngine.__new__(VlmNativeEngine)
    vlm._provider = object()  # extract_page_vlm uses this provider without network
    monkeypatch.setattr(
        V,
        "_describe_and_parse",
        lambda *a, **k: {"elements": [{"type": "text", "content": "FROM_QWEN"}]},
    )
    monkeypatch.setattr(
        VlmNativeEngine, "_page_from_payload", staticmethod(_fake_page_from_payload)
    )
    return vlm


def _page_text(page):
    return " ".join(e.content for e in page.elements)


def test_code_page_routes_to_qwen_prose_to_mineru(tmp_path, monkeypatch):
    path = _make_pdf(tmp_path)
    mineru = _FakeMinerUEngine()
    vlm = _install_vlm_mocks(monkeypatch)
    eng = MineruQwenHybridEngine(mineru_engine=mineru, vlm_engine=vlm)

    ud = eng.extract(path)

    routes = {pn: choice for pn, choice, _ in eng.last_routing_decisions}
    assert routes[1] == "qwen_code"  # Courier/code page
    assert routes[2] == "mineru"  # Helvetica/prose page

    by_page = {p.page_number: _page_text(p) for p in ud.pages}
    assert "FROM_QWEN" in by_page[1]
    assert "FROM_MINERU" in by_page[2]
    # MinerU was invoked for exactly the one prose page (not the code page).
    assert mineru.client.calls == 1


def test_high_threshold_sends_everything_to_mineru(tmp_path, monkeypatch):
    # With the threshold above 1.0, even a pure-code page stays on MinerU — the
    # pure-MinerU behavior, proving the routing is threshold-driven.
    path = _make_pdf(tmp_path)
    mineru = _FakeMinerUEngine()
    vlm = _install_vlm_mocks(monkeypatch)
    eng = MineruQwenHybridEngine(mineru_engine=mineru, vlm_engine=vlm, mono_ratio_threshold=1.5)

    ud = eng.extract(path)

    assert all(choice == "mineru" for _, choice, _ in eng.last_routing_decisions)
    assert mineru.client.calls == 2
    assert all("FROM_MINERU" in _page_text(p) for p in ud.pages)


def test_semantic_vlm_failure_falls_back_to_mineru(tmp_path, monkeypatch):
    # A single-page SEMANTIC VLM failure must demote that page to MinerU, not
    # kill the document (transport failures still trip the breaker — covered by
    # the VlmInfraError path).
    path = _make_pdf(tmp_path)
    mineru = _FakeMinerUEngine()
    vlm = _install_vlm_mocks(monkeypatch)

    def _boom(*a, **k):
        raise ValueError("malformed VLM JSON")

    monkeypatch.setattr(V, "_describe_and_parse", _boom)
    eng = MineruQwenHybridEngine(mineru_engine=mineru, vlm_engine=vlm)

    ud = eng.extract(path)

    routes = {pn: choice for pn, choice, _ in eng.last_routing_decisions}
    assert routes[1] == "mineru_fallback"  # code page demoted after Qwen failure
    assert mineru.client.calls == 2  # both pages ended up on MinerU
    assert all("FROM_MINERU" in _page_text(p) for p in ud.pages)
