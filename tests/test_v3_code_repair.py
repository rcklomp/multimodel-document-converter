"""Conversion-time code-fidelity repair (mmrag_v3.processor._repair_degraded_code).

PLAN_EXTRACTION_FIDELITY_V1 §5.4 consumer 1: a page whose code was extracted with
destroyed indentation (the R3 gate's own signal) gets ONE bounded VLM re-extraction
and the STRICTLY-better page is kept. Profile-independent and per-page, so it fixes
prose-dominant code books (Devlin classifies digital_literature, not technical_manual)
that doc-level routing misses. These tests pin the contract with a stubbed VLM lane
(no network): flag -> attempt -> keep-better, never trade text/tables for an
empty/worse page, and a no-op when the served engine is already the VLM.
"""
from __future__ import annotations

import pytest

from mmrag_v2.universal.intermediate import (
    DocumentMetadata,
    Element,
    ElementType,
    PageClassification,
    UniversalDocument,
    UniversalPage,
)
from mmrag_v3 import processor
from mmrag_v3.engines.vlm_provider import VlmInfraError

# Code blocks the R3 detector judges: a flattened suite (fails indentation_ok) and a
# correctly-nested one (passes).
_FLAT = "for n in xs:\nif n:\nfound = 1\nreturn found"
_NESTED = "for n in xs:\n    if n:\n        found = 1\n    return found"


def _el(content: str, etype: ElementType = ElementType.TEXT) -> Element:
    return Element(type=etype, content=content, bbox=None, confidence=1.0)


def _page(page_number: int, elements: list[Element]) -> UniversalPage:
    return UniversalPage(
        page_number=page_number, elements=elements,
        classification=PageClassification.DIGITAL, dimensions=(1000, 1000),
    )


def _doc(pages: list[UniversalPage], engine: str = "mineru_qwen_hybrid") -> UniversalDocument:
    d = UniversalDocument(
        doc_id="d", source_file="d.pdf", file_type="pdf",
        pages=pages, metadata=DocumentMetadata(), total_pages=len(pages),
    )
    d.metadata.extra["extraction_engine"] = engine
    return d


class _FakeFitzDoc:
    def __init__(self, n: int) -> None:
        self.page_count = n

    def __getitem__(self, i):
        return f"fitz_page_{i}"

    def close(self):  # noqa: D401
        pass


@pytest.fixture
def stub_fitz(monkeypatch):
    """Make ``fitz.open`` (imported inside the function) return a fake doc and the
    VLM engine a cheap stub (the per-page extractor is patched per test)."""
    import fitz

    monkeypatch.setattr(fitz, "open", lambda _p: _FakeFitzDoc(10))
    monkeypatch.setattr(processor, "VlmNativeEngine", lambda: object())


def _patch_vlm(monkeypatch, by_page):
    """``extract_page_vlm(vlm, fitz_page, page_number)`` -> by_page[page_number]."""
    def fake(_vlm, _fp, page_number):
        out = by_page[page_number]
        if isinstance(out, Exception):
            raise out
        return out
    monkeypatch.setattr(processor, "extract_page_vlm", fake)


def test_flagged_page_repaired_when_vlm_strictly_better(stub_fitz, monkeypatch):
    doc = _doc([_page(1, [_el(_FLAT)])])
    _patch_vlm(monkeypatch, {1: _page(1, [_el(_NESTED)])})
    out = processor._repair_degraded_code(doc, "d.pdf")
    assert out.metadata.extra["extraction_quality_risk_pages"] == 1
    assert out.metadata.extra["extraction_code_repaired_pages"] == 1
    assert out.pages[0].elements[0].content == _NESTED  # swapped to the good page


def test_no_swap_when_vlm_not_better(stub_fitz, monkeypatch):
    doc = _doc([_page(1, [_el(_FLAT)])])
    _patch_vlm(monkeypatch, {1: _page(1, [_el(_FLAT)])})  # VLM equally flat
    out = processor._repair_degraded_code(doc, "d.pdf")
    assert out.metadata.extra["extraction_code_repaired_pages"] == 0
    assert out.pages[0].elements[0].content == _FLAT  # primary kept


def test_no_swap_when_vlm_empty_keeps_text(stub_fitz, monkeypatch):
    doc = _doc([_page(1, [_el(_FLAT)])])
    _patch_vlm(monkeypatch, {1: _page(1, [])})  # VLM produced nothing
    out = processor._repair_degraded_code(doc, "d.pdf")
    assert out.metadata.extra["extraction_code_repaired_pages"] == 0
    assert out.pages[0].elements[0].content == _FLAT  # never trade text for empty


def test_table_guard_refuses_swap_that_drops_a_table(stub_fitz, monkeypatch):
    # Primary page: flattened code + a MinerU table. VLM repairs the code but drops
    # the table -> must NOT swap (do not trade a table for code).
    doc = _doc([_page(1, [_el(_FLAT), _el("col|col", ElementType.TABLE)])])
    _patch_vlm(monkeypatch, {1: _page(1, [_el(_NESTED)])})  # better code, no table
    out = processor._repair_degraded_code(doc, "d.pdf")
    assert out.metadata.extra["extraction_code_repaired_pages"] == 0
    assert any(e.type == ElementType.TABLE for e in out.pages[0].elements)


def test_table_guard_allows_swap_when_table_preserved(stub_fitz, monkeypatch):
    doc = _doc([_page(1, [_el(_FLAT), _el("col|col", ElementType.TABLE)])])
    _patch_vlm(monkeypatch, {1: _page(1, [_el(_NESTED), _el("col|col", ElementType.TABLE)])})
    out = processor._repair_degraded_code(doc, "d.pdf")
    assert out.metadata.extra["extraction_code_repaired_pages"] == 1


def test_no_swap_when_vlm_drops_surrounding_prose(stub_fitz, monkeypatch):
    # DEFECT 1 (adversarial): VLM fixes the code but sheds the page's prose. The
    # content-preservation guard must refuse the swap rather than lose paragraphs.
    prose = "This is a long paragraph of explanatory prose. " * 40  # ~1900 chars
    doc = _doc([_page(1, [_el(_FLAT), _el(prose)])])
    _patch_vlm(monkeypatch, {1: _page(1, [_el(_NESTED)])})  # better code, prose gone
    out = processor._repair_degraded_code(doc, "d.pdf")
    assert out.metadata.extra["extraction_code_repaired_pages"] == 0
    assert any((e.content or "").startswith("This is a long") for e in out.pages[0].elements)


def test_no_swap_when_vlm_empties_table_cells(stub_fitz, monkeypatch):
    # DEFECT 2 (adversarial): same table COUNT but gutted cells must not pass.
    doc = _doc([_page(1, [_el(_FLAT), _el("r1c1|r1c2|r1c3|r2c1|r2c2|r2c3", ElementType.TABLE)])])
    _patch_vlm(monkeypatch, {1: _page(1, [_el(_NESTED), _el("", ElementType.TABLE)])})
    out = processor._repair_degraded_code(doc, "d.pdf")
    assert out.metadata.extra["extraction_code_repaired_pages"] == 0
    assert any(
        e.type == ElementType.TABLE and (e.content or "").strip()
        for e in out.pages[0].elements
    )


def test_clean_code_page_not_flagged(stub_fitz, monkeypatch):
    called = []
    monkeypatch.setattr(processor, "extract_page_vlm",
                        lambda *a, **k: called.append(1))
    doc = _doc([_page(1, [_el(_NESTED)])])
    out = processor._repair_degraded_code(doc, "d.pdf")
    assert out.metadata.extra["extraction_quality_risk_pages"] == 0
    assert called == []  # no VLM call on a clean page


def test_vlm_native_engine_is_noop(stub_fitz, monkeypatch):
    called = []
    monkeypatch.setattr(processor, "extract_page_vlm",
                        lambda *a, **k: called.append(1))
    doc = _doc([_page(1, [_el(_FLAT)])], engine="vlm_native")
    out = processor._repair_degraded_code(doc, "d.pdf")
    # already the specialist: do not re-run, do not even stamp a risk count
    assert "extraction_quality_risk_pages" not in out.metadata.extra
    assert called == []


def test_infra_failure_aborts_repair_keeps_primary(stub_fitz, monkeypatch):
    doc = _doc([_page(1, [_el(_FLAT)]), _page(2, [_el(_FLAT)])])
    _patch_vlm(monkeypatch, {1: VlmInfraError("node offline"), 2: _page(2, [_el(_NESTED)])})
    out = processor._repair_degraded_code(doc, "d.pdf")
    # circuit-breaker: aborts on the first transport failure; nothing repaired
    assert out.metadata.extra["extraction_code_repaired_pages"] == 0
    assert out.pages[0].elements[0].content == _FLAT
    assert out.pages[1].elements[0].content == _FLAT
