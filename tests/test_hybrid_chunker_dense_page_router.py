from __future__ import annotations

import logging
import os
import re
import time
from types import SimpleNamespace

import pytest

from docling_core.transforms.chunker import HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import ChunkingSerializerProvider
from docling_core.types.doc.document import (
    BoundingBox,
    CoordOrigin,
    DoclingDocument,
    ProvenanceItem,
)
from docling_core.types.doc.labels import DocItemLabel

from mmrag_v2.engines.docling_serializers import MmragChunkingSerializerProvider
from mmrag_v2.processor import _classify_dense_index_pages


def _prov(page_no: int):
    return SimpleNamespace(
        page_no=page_no,
        bbox=SimpleNamespace(l=10.0, t=10.0, r=100.0, b=20.0),
    )


def _item(page_no: int, text: str, label: str = "list_item"):
    return SimpleNamespace(text=text, label=label, prov=[_prov(page_no)])


class _FixtureDoc:
    def __init__(self, items):
        self._items = items

    def iterate_items(self):
        for item in self._items:
            yield item, 0


def _dense_ayeva_like_page(page_no: int):
    items = []
    for idx in range(12):
        items.append(_item(page_no, f"term{idx}", "list_item"))
        items.append(_item(page_no, f"{idx + 10}, {idx + 30}, {idx + 70}", "document_index"))
    return items


def test_classify_dense_index_pages_flags_synthetic_unit_shape_only():
    doc = _FixtureDoc(
        _dense_ayeva_like_page(285)
        + _dense_ayeva_like_page(286)
        + [
            _item(
                5,
                "Chapter 1 introduces the problem domain with prose and no index references.",
                "paragraph",
            )
            for _ in range(10)
        ]
        + [
            _item(
                66,
                "Squadron roster table row with names, roles, and ordinary numeric values",
                "table",
            )
            for _ in range(24)
        ]
    )

    assert _classify_dense_index_pages(doc) == {285, 286}


def test_classify_dense_index_pages_rejects_chapter_intro_and_code_page():
    hao_intro = [
        _item(5, "A chapter introduction with normal front matter prose.", "paragraph")
        for _ in range(14)
    ]
    ayeva_code = [
        _item(42, f"def example_{idx}(value): return value + {idx}", "code")
        for idx in range(20)
    ]
    doc = _FixtureDoc(hao_intro + ayeva_code)

    assert _classify_dense_index_pages(doc) == set()


def test_classify_dense_index_pages_trusts_document_index_label_without_text():
    doc = _FixtureDoc(
        [
            _item(285, "", "document_index"),
            _item(286, "", "DOCUMENT_INDEX"),
            _item(5, "Normal chapter prose.", "paragraph"),
        ]
    )

    assert _classify_dense_index_pages(doc) == {285, 286}


def test_document_index_lines_dedups_per_entry_across_grid_cells():
    # Real Docling 2.86 DocumentIndex grids contain massive cell-level
    # duplication PLUS sliding-window overlap between unique cells, with
    # entries separated by single space (digit-then-letter boundary), not
    # newlines. The helper must dedup at entry granularity in both axes.
    from mmrag_v2.processor import _docling_document_index_lines

    # Two cells that are byte-equal duplicates of the same sliding window.
    win_a = SimpleNamespace(text="Mock Object pattern  235 behavior verification  235 implementing  237, 238")
    win_a_dup = SimpleNamespace(text="Mock Object pattern  235 behavior verification  235 implementing  237, 238")
    # Cell that overlaps win_a but adds one new entry.
    win_b = SimpleNamespace(text="behavior verification  235 implementing  237, 238 isolation  235")
    # Alphabet header with no digit -- must survive the split.
    header = SimpleNamespace(text="M")
    item = SimpleNamespace(
        data=SimpleNamespace(grid=[[header, win_a], [win_a_dup, win_b]]),
        text="",
    )

    lines = _docling_document_index_lines(item)

    # Each unique entry exactly once, in document order.
    # _sanitize_toc_index_text collapses the double-space term/page
    # delimiter to a single space, so entries are stored normalized.
    assert lines == [
        "M",
        "Mock Object pattern 235",
        "behavior verification 235",
        "implementing 237, 238",
        "isolation 235",
    ]


def _docling_prov(page_no: int):
    return ProvenanceItem(
        page_no=page_no,
        bbox=BoundingBox(
            l=10.0,
            t=10.0,
            r=200.0,
            b=40.0,
            coord_origin=CoordOrigin.TOPLEFT,
        ),
        charspan=(0, 1),
    )


def _make_docling_doc() -> DoclingDocument:
    doc = DoclingDocument(name="dense-router")
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="Normal paragraph with enough words to make a chunk.",
        prov=_docling_prov(1),
    )
    doc.add_text(
        label=DocItemLabel.PARAGRAPH,
        text="alpha 1, 2, 3, 4",
        prov=_docling_prov(2),
    )
    return doc


def _chunk_pages(chunks):
    pages = set()
    for chunk in chunks:
        for item in chunk.meta.doc_items:
            pages.add(item.prov[0].page_no)
    return pages


def test_serializer_skip_pages_drops_dense_page_items_from_chunker_input():
    doc = _make_docling_doc()
    chunker = HybridChunker(
        tokenizer="sentence-transformers/all-MiniLM-L6-v2",
        serializer_provider=MmragChunkingSerializerProvider(skip_pages={2}),
    )

    chunks = list(chunker.chunk(doc))

    assert chunks
    assert 2 not in _chunk_pages(chunks)


def test_serializer_provider_without_skip_pages_matches_default_output():
    doc = _make_docling_doc()
    default_text = ChunkingSerializerProvider().get_serializer(doc).serialize().text
    mmrag_text = MmragChunkingSerializerProvider(skip_pages=set()).get_serializer(doc).serialize().text

    assert mmrag_text == default_text


def test_ayeva_dense_index_batch_completes_under_30s_when_enabled(tmp_path, caplog):
    if os.environ.get("RUN_DENSE_ROUTER_PERF") != "1":
        pytest.skip("set RUN_DENSE_ROUTER_PERF=1 to run the Ayeva batch performance smoke")

    import fitz

    from mmrag_v2.engines.docling_adapter import DoclingPdfAdapter
    from mmrag_v2.engines.pdf_plan import build_pdf_conversion_plan
    from mmrag_v2.processor import V2DocumentProcessor
    from mmrag_v2.schema.ingestion_schema import FileType

    src = (
        "data/technical_manual/"
        "Ayeva K. Mastering Python Design Patterns...essential Python patterns...3ed 2024.pdf"
    )
    src_doc = fitz.open(src)
    sliced = fitz.open()
    try:
        sliced.insert_pdf(src_doc, from_page=280, to_page=289)
        batch_path = tmp_path / "ayeva_281_290.pdf"
        sliced.save(batch_path, garbage=4, deflate=True)
    finally:
        sliced.close()
        src_doc.close()

    plan = build_pdf_conversion_plan(
        enable_ocr=False,
        needs_code_enrichment=False,
        has_encoding_corruption=False,
        has_flat_text_corruption=False,
        document_modality="technical_manual",
        profile_type="technical_manual",
    )
    result = DoclingPdfAdapter(plan).convert(batch_path)
    doc = result.document
    processor = V2DocumentProcessor(output_dir=str(tmp_path), vision_provider="none")

    caplog.set_level(logging.ERROR, logger="mmrag_v2.processor")
    started = time.monotonic()
    chunks = processor._process_text_with_hybrid_chunker(
        doc=doc,
        doc_hash="ayeva_perf",
        source_file="ayeva.pdf",
        file_type=FileType.PDF,
        page_dims={page: (612.0, 792.0) for page in range(1, 11)},
        page_offset=280,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 30
    assert chunks
    assert "Per-batch timeout fired" not in caplog.text


def test_ayeva_real_docling_document_index_pages_are_flagged_when_enabled(tmp_path):
    if os.environ.get("RUN_DENSE_ROUTER_PERF") != "1":
        pytest.skip("set RUN_DENSE_ROUTER_PERF=1 to run real Ayeva Docling coverage")

    doc = _load_ayeva_index_slice(tmp_path)

    assert _classify_dense_index_pages(doc) == {1, 2, 3, 4, 5, 6}


def test_ayeva_real_docling_pageskip_emits_real_index_text_when_enabled(tmp_path):
    if os.environ.get("RUN_DENSE_ROUTER_PERF") != "1":
        pytest.skip("set RUN_DENSE_ROUTER_PERF=1 to run real Ayeva Docling coverage")

    from mmrag_v2.processor import V2DocumentProcessor
    from mmrag_v2.schema.ingestion_schema import FileType

    doc = _load_ayeva_index_slice(tmp_path)
    processor = V2DocumentProcessor(output_dir=str(tmp_path), vision_provider="none")
    chunks = processor._emit_dense_index_page_chunks(
        doc=doc,
        dense_pages={1, 2, 3, 4, 5, 6},
        doc_hash="ayeva_emit",
        source_file="ayeva.pdf",
        file_type=FileType.PDF,
        page_dims={page: (612.0, 792.0) for page in range(1, 7)},
        page_offset=284,
    )

    by_page = {chunk.metadata.page_number: chunk for chunk in chunks}
    assert set(by_page) == {285, 286, 287, 288, 289, 290}
    for chunk in by_page.values():
        assert chunk.metadata.extraction_method == "hybrid_chunker_pageskip"
        assert len(chunk.content) > 200
        assert re.search(r"\b[a-z]+\s+\d{1,4}\b", chunk.content, re.I)


def _load_ayeva_index_slice(tmp_path):
    import fitz

    from mmrag_v2.engines.docling_adapter import DoclingPdfAdapter
    from mmrag_v2.engines.pdf_plan import build_pdf_conversion_plan

    src = (
        "data/technical_manual/"
        "Ayeva K. Mastering Python Design Patterns...essential Python patterns...3ed 2024.pdf"
    )
    src_doc = fitz.open(src)
    sliced = fitz.open()
    try:
        sliced.insert_pdf(src_doc, from_page=284, to_page=289)
        batch_path = tmp_path / "ayeva_285_290.pdf"
        sliced.save(batch_path, garbage=4, deflate=True)
    finally:
        sliced.close()
        src_doc.close()

    plan = build_pdf_conversion_plan(
        enable_ocr=False,
        needs_code_enrichment=False,
        has_encoding_corruption=False,
        has_flat_text_corruption=False,
        document_modality="technical_manual",
        profile_type="technical_manual",
    )
    return DoclingPdfAdapter(plan).convert(batch_path).document
