"""Phase B3 — section-header-only page chunk emission
(`docs/PLAN_V2.9.md` §3 Phase B3, `docs/DECISIONS.md` Retrieval-Value Test).

`V2DocumentProcessor._emit_section_header_only_page_chunks` (in
`src/mmrag_v2/processor.py`) emits one text chunk for each page whose
ONLY Docling items are `section_header` (chapter-divider / part-opener
pages). HybridChunker treats section_header items as heading metadata
to be propagated into the breadcrumbs of subsequent body chunks; on
heading-only pages this means no chunk is emitted and the strict gate
reports MISSING_PAGES.

Per the Retrieval-Value Test, chapter dividers are high-retrieval-value
signal (they answer "where does Chapter / Part X start?"), so the fix
emits a chunk rather than marking the page blank-equivalent.

Verified end-to-end on Devlin_LLM_Agents p170
("II - Building Intelligent Foundations"), 2026-05-11.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import pytest

from mmrag_v2.processor import V2DocumentProcessor
from mmrag_v2.schema.ingestion_schema import (
    ChunkType,
    FileType,
    IngestionChunk,
    create_text_chunk,
)


def _mk_label(value: str) -> SimpleNamespace:
    return SimpleNamespace(value=value)


def _mk_prov(page_no: int) -> SimpleNamespace:
    return SimpleNamespace(page_no=page_no, bbox=None)


def _mk_item(label: str, text: str, page_no: int) -> SimpleNamespace:
    return SimpleNamespace(
        label=_mk_label(label),
        text=text,
        prov=[_mk_prov(page_no)],
    )


class _FakeDoc:
    """Minimal Docling-document stand-in for the per-item iteration
    used by the emitter."""

    def __init__(self, items: List[SimpleNamespace]):
        self._items = items

    def iterate_items(self):
        return [(it, 1) for it in self._items]


def _emitter() -> V2DocumentProcessor:
    p = V2DocumentProcessor.__new__(V2DocumentProcessor)
    p._chunk_position = 0
    p._intelligence_metadata = {}
    return p


def _existing_text_chunk(page: int) -> IngestionChunk:
    return create_text_chunk(
        content=f"body text on page {page}",
        doc_id="fake_doc_id",
        source_file="Fake.pdf",
        file_type=FileType.PDF,
        page_number=page,
        extraction_method="hybrid_chunker",
        position=page,  # ensure unique chunk_id
    )


# ---------------------------------------------------------------------------
# Positive cases — page produces a chunk when it should
# ---------------------------------------------------------------------------


def test_single_section_header_page_emits_chunk() -> None:
    """The canonical Devlin p170 case: one section_header on its own
    page produces one chunk."""
    doc = _FakeDoc([
        _mk_item("section_header", "II - Building Intelligent Foundations", page_no=170),
    ])
    emitter = _emitter()
    result = emitter._emit_section_header_only_page_chunks(
        doc=doc,
        existing_chunks=[_existing_text_chunk(169), _existing_text_chunk(171)],
        dense_index_pages=set(),
        doc_hash="fakehash01",
        source_file="Devlin.pdf",
        file_type=FileType.PDF,
        page_dims={170: (612.0, 792.0)},
        page_offset=0,
    )
    assert len(result) == 1
    chunk = result[0]
    assert chunk.metadata.page_number == 170
    assert chunk.metadata.extraction_method == "hybrid_chunker_section_header_page"
    assert chunk.content == "II - Building Intelligent Foundations"
    assert chunk.metadata.hierarchy.parent_heading == "II - Building Intelligent Foundations"
    assert chunk.metadata.search_priority == "high"
    assert chunk.metadata.chunk_type == ChunkType.HEADING


def test_multiple_section_headers_on_one_page_emit_one_chunk_with_concatenated_text() -> None:
    """Title page with multiple stacked headings (Nagasubramanian p2
    shape) emits ONE chunk with newline-joined content. Per the
    Retrieval-Value Test, this is high-value because it captures the
    full title / subtitle / author signal."""
    doc = _FakeDoc([
        _mk_item("section_header", "Agentic AI for Engineers", page_no=2),
        _mk_item("section_header", "Architecting Goal-Driven Systems", page_no=2),
        _mk_item("section_header", "Dhivya Nagasubramanian", page_no=2),
    ])
    emitter = _emitter()
    result = emitter._emit_section_header_only_page_chunks(
        doc=doc,
        existing_chunks=[_existing_text_chunk(1), _existing_text_chunk(3)],
        dense_index_pages=set(),
        doc_hash="fakehash02",
        source_file="Nagasubramanian.pdf",
        file_type=FileType.PDF,
        page_dims={2: (612.0, 792.0)},
        page_offset=0,
    )
    assert len(result) == 1
    chunk = result[0]
    assert "Agentic AI for Engineers" in chunk.content
    assert "Architecting Goal-Driven Systems" in chunk.content
    assert "Dhivya Nagasubramanian" in chunk.content
    assert chunk.metadata.page_number == 2


def test_title_label_treated_as_heading() -> None:
    """Docling sometimes labels the top-of-title item as `title`
    rather than `section_header`. Both should be treated as heading
    content."""
    doc = _FakeDoc([
        _mk_item("title", "The MCP Standard", page_no=2),
        _mk_item("section_header", "A Developer's Guide to Building Universal AI Tools", page_no=2),
    ])
    emitter = _emitter()
    result = emitter._emit_section_header_only_page_chunks(
        doc=doc,
        existing_chunks=[_existing_text_chunk(1)],
        dense_index_pages=set(),
        doc_hash="fakehash03",
        source_file="Sekar.pdf",
        file_type=FileType.PDF,
        page_dims={2: (612.0, 792.0)},
        page_offset=0,
    )
    assert len(result) == 1
    assert "The MCP Standard" in result[0].content


# ---------------------------------------------------------------------------
# Negative cases — page does NOT produce a chunk (out of scope or already covered)
# ---------------------------------------------------------------------------


def test_page_already_covered_does_not_double_emit() -> None:
    """If the page already has a chunk from the normal hybrid_chunker
    path, the emitter must NOT add a duplicate."""
    doc = _FakeDoc([
        _mk_item("section_header", "Chapter 5: Foo", page_no=42),
    ])
    emitter = _emitter()
    result = emitter._emit_section_header_only_page_chunks(
        doc=doc,
        existing_chunks=[_existing_text_chunk(42)],  # p42 already covered
        dense_index_pages=set(),
        doc_hash="fakehash04",
        source_file="Foo.pdf",
        file_type=FileType.PDF,
        page_dims={42: (612.0, 792.0)},
        page_offset=0,
    )
    assert result == []


def test_page_with_mixed_text_and_header_NOT_emitted() -> None:
    """Pages with both section_header AND body text items go through
    the normal hybrid_chunker path. If a chunk drop happens there, it
    is a separate cross-page-split defect (Phase B3 Step 3) and must
    NOT be papered over by this emitter."""
    doc = _FakeDoc([
        _mk_item("section_header", "Chapter 3", page_no=50),
        _mk_item("text", "This chapter introduces the concept of ...", page_no=50),
    ])
    emitter = _emitter()
    result = emitter._emit_section_header_only_page_chunks(
        doc=doc,
        existing_chunks=[],
        dense_index_pages=set(),
        doc_hash="fakehash05",
        source_file="Mixed.pdf",
        file_type=FileType.PDF,
        page_dims={50: (612.0, 792.0)},
        page_offset=0,
    )
    assert result == []


def test_page_with_only_text_items_NOT_emitted() -> None:
    """Pages with only body text items are out of scope (handled by
    normal hybrid_chunker or by the dedication / URL-list diagnostic
    in B3 Step 3)."""
    doc = _FakeDoc([
        _mk_item("text", "I would like to thank my parents for their love and support.", page_no=4),
        _mk_item("text", "- Kamon Ayeva", page_no=4),
    ])
    emitter = _emitter()
    result = emitter._emit_section_header_only_page_chunks(
        doc=doc,
        existing_chunks=[],
        dense_index_pages=set(),
        doc_hash="fakehash06",
        source_file="Ayeva.pdf",
        file_type=FileType.PDF,
        page_dims={4: (612.0, 792.0)},
        page_offset=0,
    )
    assert result == []


def test_dense_index_page_NOT_emitted_again() -> None:
    """Pages already routed to `_emit_dense_index_page_chunks` must
    not get a second pass here."""
    doc = _FakeDoc([
        _mk_item("section_header", "Table of Contents", page_no=5),
    ])
    emitter = _emitter()
    result = emitter._emit_section_header_only_page_chunks(
        doc=doc,
        existing_chunks=[],
        dense_index_pages={5},
        doc_hash="fakehash07",
        source_file="Cronin.pdf",
        file_type=FileType.PDF,
        page_dims={5: (612.0, 792.0)},
        page_offset=0,
    )
    assert result == []


def test_page_with_empty_section_header_text_NOT_emitted() -> None:
    """Defensive: a section_header with empty text must not produce
    a chunk."""
    doc = _FakeDoc([
        _mk_item("section_header", "", page_no=99),
    ])
    emitter = _emitter()
    result = emitter._emit_section_header_only_page_chunks(
        doc=doc,
        existing_chunks=[],
        dense_index_pages=set(),
        doc_hash="fakehash08",
        source_file="Empty.pdf",
        file_type=FileType.PDF,
        page_dims={99: (612.0, 792.0)},
        page_offset=0,
    )
    assert result == []


def test_doc_without_iterate_items_returns_empty() -> None:
    """Defensive: a doc object that doesn't expose iterate_items()
    must not crash."""
    doc = SimpleNamespace()  # no iterate_items attribute
    emitter = _emitter()
    result = emitter._emit_section_header_only_page_chunks(
        doc=doc,
        existing_chunks=[],
        dense_index_pages=set(),
        doc_hash="fakehash09",
        source_file="Foo.pdf",
        file_type=FileType.PDF,
        page_dims={},
        page_offset=0,
    )
    assert result == []


# ---------------------------------------------------------------------------
# Page-offset handling (batch processing)
# ---------------------------------------------------------------------------


def test_page_offset_applied_to_output_page_number() -> None:
    """Doc-local raw_page=5 in a batch that starts at page 100 should
    emit a chunk with metadata.page_number=105."""
    doc = _FakeDoc([
        _mk_item("section_header", "Chapter Title", page_no=5),
    ])
    emitter = _emitter()
    result = emitter._emit_section_header_only_page_chunks(
        doc=doc,
        existing_chunks=[],
        dense_index_pages=set(),
        doc_hash="fakehash10",
        source_file="Batched.pdf",
        file_type=FileType.PDF,
        page_dims={105: (612.0, 792.0)},
        page_offset=100,
    )
    assert len(result) == 1
    assert result[0].metadata.page_number == 105
