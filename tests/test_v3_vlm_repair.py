"""V3 bounded JSON-repair contract (Charter Blocker A / A4, 2026-06-03).

Fail-open last resort for the dense-page failure mode: when a VLM page
response is truncated or malformed, recover the N COMPLETE elements (dropping
the partial trailing one) instead of discarding the whole page to a Docling
fallback. Partial vision-native extraction beats full Docling fallback on a
dense page.

Every payload here is a DENSE multi-element page cut mid-array (the real
failure mode); a pass on a 1-element body would prove nothing. Fully
offline/deterministic: no network, no VLM, no Docling.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import fitz
import pytest

from mmrag_v3.engines.router import HybridEngine
from mmrag_v3.engines.vlm_native import (
    VlmNativeEngine,
    repair_truncated_json,
)
from mmrag_v3.engines.vlm_provider import VlmProviderError, VlmTruncationError

_HEAD = (
    '{"page_number": 1, "width": 1654, "height": 2339, '
    '"classification": "digital", "elements": ['
)


def _complete_elements(n: int = 40) -> list:
    return [
        json.dumps(
            {
                "type": "text",
                "content": f"Paragraph {i}: " + ("dense body text " * 10).strip(),
                "bbox": [80, 100 + i * 30, 1500, 128 + i * 30],
                "confidence": 0.95,
                "source_label": "paragraph",
            }
        )
        for i in range(n)
    ]


def _dense_truncated() -> str:
    partial = '{"type": "text", "content": "Paragraph 40: cut off mid-sent'
    return _HEAD + ", ".join(_complete_elements()) + ", " + partial


def _dense_full() -> str:
    return _HEAD + ", ".join(_complete_elements()) + "]}"


# --------------------------------------------------------------------------
# repair_truncated_json
# --------------------------------------------------------------------------


def test_repair_recovers_all_complete_elements():
    payload = repair_truncated_json(_dense_truncated())
    assert payload is not None
    # 40 complete elements survive; the partial 41st is dropped.
    assert len(payload["elements"]) == 40
    assert payload["elements"][0]["content"].startswith("Paragraph 0:")
    assert payload["elements"][-1]["content"].startswith("Paragraph 39:")


def test_repair_passthrough_on_valid_json():
    full = _dense_full()
    payload = repair_truncated_json(full)
    assert payload == json.loads(full)


def test_repair_returns_none_when_nothing_recoverable():
    assert repair_truncated_json("not json at all, no elements here") is None
    # A JSON array (not object) with no recoverable element objects.
    assert repair_truncated_json('{"elements": [') is None


def test_repair_handles_single_complete_then_cut():
    one = (
        _HEAD
        + '{"type": "text", "content": "kept", "bbox": [1,1,2,2]}, '
        + '{"type": "text", "content": "lost mid'
    )
    payload = repair_truncated_json(one)
    assert payload is not None
    assert len(payload["elements"]) == 1
    assert payload["elements"][0]["content"] == "kept"


# --------------------------------------------------------------------------
# Engine integration: truncation -> repaired page, NOT discarded
# --------------------------------------------------------------------------


def _one_page_pdf(tmp_path):
    pdf = tmp_path / "one.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf))
    doc.close()
    return pdf


class _TruncatingProvider:
    """Raises VlmTruncationError carrying a dense partial body (post-A1)."""

    def describe(self, image_bytes, prompt, *, mime="image/png", max_tokens=None):
        raise VlmTruncationError(
            "truncated", partial_content=_dense_truncated(), finish_reason="length"
        )


def test_vlm_engine_repairs_truncated_page(tmp_path):
    engine = VlmNativeEngine(provider=_TruncatingProvider())
    doc = engine.extract(str(_one_page_pdf(tmp_path)))
    assert len(doc.pages) == 1
    # The page is salvaged with its complete elements, not emptied/dropped.
    assert len(doc.pages[0].elements) == 40


def test_router_repairs_truncation_without_docling_fallback(tmp_path, monkeypatch):
    """A truncated VLM page is repaired in-place; Docling is never invoked."""
    monkeypatch.setattr(HybridEngine, "_classify_page", lambda self, page: ("vlm", "forced"))
    monkeypatch.setattr(
        HybridEngine, "_render_page_png", staticmethod(lambda page: (b"x", 100, 100))
    )

    class _DoclingMustNotRun:
        def extract(self, *a, **k):
            raise AssertionError("Docling fallback invoked for a REPAIRABLE truncation")

    engine = HybridEngine(
        vlm_engine=SimpleNamespace(_provider=_TruncatingProvider()),
        docling_engine=_DoclingMustNotRun(),
    )
    doc = engine.extract(str(_one_page_pdf(tmp_path)))
    assert len(doc.pages[0].elements) == 40
    # The page stays a VLM page (no docling_fallback in the decision log).
    assert all(c != "docling_fallback" for _, c, _ in engine.last_routing_decisions)


class _UnrecoverableProvider:
    def describe(self, image_bytes, prompt, *, mime="image/png", max_tokens=None):
        raise VlmTruncationError("truncated", partial_content="garbage{", finish_reason="length")


def test_unrecoverable_truncation_still_raises(tmp_path):
    """When repair recovers nothing, the typed failure still propagates."""
    engine = VlmNativeEngine(provider=_UnrecoverableProvider())
    with pytest.raises(VlmProviderError):
        engine.extract(str(_one_page_pdf(tmp_path)))
