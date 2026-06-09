"""Fail-closed extraction fallback (mmrag_v3.processor.extract, 2026-06-09).

Reliability must not depend on a remote multi-model GPU server staying healthy.
The M5 mlx MinerU2.5 server throws intermittent `broadcast_shapes` 500s when its
two-step extract batches different-sized block images; without a safety net those
pages produced ZERO chunks silently. These tests pin the contract: an engine that
RAISES, or returns a page that found text regions but extracted no content, is
recovered from the offline DoclingFastEngine, keeping the primary's good pages,
and the served lane + fallback outcome are stamped on doc.metadata.extra.
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


def _page(page_number: int, elements: list[Element]) -> UniversalPage:
    return UniversalPage(
        page_number=page_number,
        elements=elements,
        classification=PageClassification.DIGITAL,
        dimensions=(1000, 1000),
    )


def _text(content: str) -> Element:
    return Element(type=ElementType.TEXT, content=content, bbox=None, confidence=1.0)


def _image() -> Element:
    return Element(type=ElementType.IMAGE, content="", bbox=None, confidence=1.0)


def _doc(pages: list[UniversalPage], source: str = "d.pdf") -> UniversalDocument:
    return UniversalDocument(
        doc_id="d", source_file=source, file_type="pdf",
        pages=pages, metadata=DocumentMetadata(), total_pages=len(pages),
    )


class _Engine:
    """Stub engine: return a fixed doc, or raise a fixed exception."""

    def __init__(self, doc=None, exc=None, counter=None, label=""):
        self._doc, self._exc, self._counter, self._label = doc, exc, counter, label

    def extract(self, _path):
        if self._counter is not None:
            self._counter.append(self._label)
        if self._exc is not None:
            raise self._exc
        return self._doc


def _route(monkeypatch, *, primary_env, primary, docling):
    """Force `primary_env`, patch primary + DoclingFastEngine to stubs."""
    for var in ("USE_MINERU_ENGINE", "USE_VLM_ENGINE", "USE_DOCLING_FAST",
                "USE_HYBRID_ENGINE", "USE_MINERU_QWEN_HYBRID"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("MINERU_ENDPOINT", raising=False)
    monkeypatch.setenv(primary_env, "1")
    name = {"USE_MINERU_ENGINE": "MineruNativeEngine", "USE_VLM_ENGINE": "VlmNativeEngine",
            "USE_DOCLING_FAST": "DoclingFastEngine"}[primary_env]
    monkeypatch.setattr(processor, name, lambda: primary)
    if name != "DoclingFastEngine":
        monkeypatch.setattr(processor, "DoclingFastEngine", lambda: docling)


def test_engine_exception_falls_back_to_docling(monkeypatch):
    good = _doc([_page(1, [_text("recovered body text")])])
    primary = _Engine(exc=RuntimeError("ServerError 500 broadcast_shapes"))
    docling = _Engine(doc=good)
    _route(monkeypatch, primary_env="USE_MINERU_ENGINE", primary=primary, docling=docling)

    out = processor.extract("/d.pdf")
    assert out is good
    assert out.metadata.extra["extraction_engine"] == "mineru"
    assert out.metadata.extra["extraction_fallback"] == "docling_fast"
    assert "broadcast_shapes" in out.metadata.extra["extraction_fallback_reason"]


def test_degenerate_page_recovered_from_docling(monkeypatch):
    # primary: page 1 good, page 2 degenerate (TEXT region, empty content).
    primary_doc = _doc([_page(1, [_text("good page one")]), _page(2, [_text("")])])
    docling_doc = _doc([_page(1, [_text("DOCLING one")]), _page(2, [_text("DOCLING recovered two")])])
    primary = _Engine(doc=primary_doc)
    docling = _Engine(doc=docling_doc)
    _route(monkeypatch, primary_env="USE_MINERU_ENGINE", primary=primary, docling=docling)

    out = processor.extract("/d.pdf")
    assert out.pages[0].elements[0].content == "good page one"  # kept primary's good page
    assert out.pages[1].elements[0].content == "DOCLING recovered two"  # swapped degenerate page
    assert out.metadata.extra["extraction_engine"] == "mineru"
    assert out.metadata.extra["extraction_fallback"] == "docling_fast"
    assert out.metadata.extra["extraction_degraded_pages"] == 1
    assert out.metadata.extra["extraction_recovered_pages"] == 1


def test_healthy_doc_pays_nothing(monkeypatch):
    calls: list[str] = []
    good = _doc([_page(1, [_text("all good"), _image()])])
    primary = _Engine(doc=good, counter=calls, label="primary")
    docling = _Engine(doc=good, counter=calls, label="docling")
    _route(monkeypatch, primary_env="USE_MINERU_ENGINE", primary=primary, docling=docling)

    out = processor.extract("/d.pdf")
    assert calls == ["primary"]  # docling never instantiated/called on a healthy doc
    assert out.metadata.extra["extraction_engine"] == "mineru"
    assert out.metadata.extra["extraction_fallback"] is None


def test_image_only_page_is_not_degenerate(monkeypatch):
    calls: list[str] = []
    img_doc = _doc([_page(1, [_image()])])  # no TEXT element -> legit visual page
    primary = _Engine(doc=img_doc, counter=calls, label="primary")
    docling = _Engine(doc=img_doc, counter=calls, label="docling")
    _route(monkeypatch, primary_env="USE_VLM_ENGINE", primary=primary, docling=docling)

    out = processor.extract("/d.pdf")
    assert calls == ["primary"]  # not flagged degenerate, no docling fallback
    assert out.metadata.extra["extraction_fallback"] is None


def test_degenerate_but_docling_also_empty_keeps_primary(monkeypatch):
    primary_doc = _doc([_page(1, [_text("")])])
    docling_doc = _doc([_page(1, [_text("")])])  # docling no better
    primary = _Engine(doc=primary_doc)
    docling = _Engine(doc=docling_doc)
    _route(monkeypatch, primary_env="USE_MINERU_ENGINE", primary=primary, docling=docling)

    out = processor.extract("/d.pdf")
    assert out.pages[0] is primary_doc.pages[0]  # kept primary (docling no improvement)
    assert out.metadata.extra["extraction_recovered_pages"] == 0
    assert out.metadata.extra["extraction_degraded_pages"] == 1


def test_docling_primary_exception_reraises(monkeypatch):
    # USE_DOCLING_FAST primary that raises has nothing more reliable to fall back to.
    primary = _Engine(exc=RuntimeError("docling boom"))
    _route(monkeypatch, primary_env="USE_DOCLING_FAST", primary=primary, docling=primary)
    with pytest.raises(RuntimeError, match="docling boom"):
        processor.extract("/d.pdf")


# --- Terminal tier: PyMuPDF native text (the floor that cannot lose text) ----
def _make_pdf(path, page_texts):
    import fitz

    fdoc = fitz.open()
    for text in page_texts:
        page = fdoc.new_page()
        if text:
            page.insert_text((72, 72), text)
    fdoc.save(str(path))
    fdoc.close()


def test_docling_also_fails_uses_pymupdf_terminal(tmp_path, monkeypatch):
    pdf = tmp_path / "t.pdf"
    _make_pdf(pdf, ["terminal recovered this page"])
    primary = _Engine(exc=RuntimeError("ServerError 500 broadcast_shapes"))
    docling = _Engine(exc=RuntimeError("docling boom"))  # tier 2 ALSO fails
    _route(monkeypatch, primary_env="USE_MINERU_ENGINE", primary=primary, docling=docling)

    out = processor.extract(str(pdf))
    assert "terminal recovered this page" in out.pages[0].elements[0].content
    assert out.metadata.extra["extraction_engine"] == "mineru"
    assert out.metadata.extra["extraction_fallback"] == "pymupdf_terminal"


def test_terminal_recovers_per_page_text_layer(tmp_path, monkeypatch):
    pdf = tmp_path / "t.pdf"
    _make_pdf(pdf, ["page one native text"])
    primary = _Engine(doc=_doc([_page(1, [_text("")])]))   # degenerate page
    docling = _Engine(doc=_doc([_page(1, [_text("")])]))   # docling no better
    _route(monkeypatch, primary_env="USE_MINERU_ENGINE", primary=primary, docling=docling)

    out = processor.extract(str(pdf))
    assert "page one native text" in out.pages[0].elements[0].content
    assert out.metadata.extra["extraction_recovered_pages"] == 1


def test_terminal_does_not_fabricate_on_blank_page(tmp_path, monkeypatch):
    pdf = tmp_path / "blank.pdf"
    _make_pdf(pdf, [""])  # genuinely text-less page
    primary = _Engine(exc=RuntimeError("boom"))
    docling = _Engine(exc=RuntimeError("boom2"))
    _route(monkeypatch, primary_env="USE_MINERU_ENGINE", primary=primary, docling=docling)

    out = processor.extract(str(pdf))
    assert out.metadata.extra["extraction_fallback"] == "pymupdf_terminal"
    assert out.pages[0].elements == []  # empty, NOT fabricated
