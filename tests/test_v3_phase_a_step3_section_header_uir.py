"""V3.0 Phase A step 3 bridge tests: section-header-only page Docling -> UIR.

Charter: docs/ARCHITECTURE_V3_DRAFT_0.5.md §3.2 Phase A step 3
(rewrite `_emit_section_header_only_page_chunks` for UIR-native).

These tests pin:
  - `_section_header_page_to_uir_chunk` returns a UIRChunk for pages
    where ALL items are section_header/title labels, and None for
    mixed-content pages.
  - The rewritten emit method preserves v2.16 invariants:
    chunk_type=HEADING, extraction_method=hybrid_chunker_section_header_page,
    parent_heading=first heading line, breadcrumb=[doc, heading, Page N],
    hierarchy.level=2 (literal v2.16 value, not auto-computed),
    search_priority="high", refined_content==content.
  - Pages already covered by existing chunks are skipped.
  - Dense-index pages are skipped (already handled by step-2 emitter).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List

import pytest

from mmrag_v2.processor import (
    _section_header_page_to_uir_chunk,
    V2DocumentProcessor,
)
from mmrag_v2.schema.ingestion_schema import (
    ChunkType,
    FileType,
    IngestionChunk,
    Modality,
)
from mmrag_v2.universal.intermediate import (
    CoordinateFrame,
    LocatorType,
    UIRChunk,
)


# --------------------------------------------------------------------------
# DoclingDocument shape mocks
# --------------------------------------------------------------------------


def _bbox(l=10, t=10, r=600, b=200):
    return SimpleNamespace(l=l, t=t, r=r, b=b)


def _heading_item(text: str, page_no: int = 1, label: str = "section_header"):
    return SimpleNamespace(
        text=text,
        prov=[SimpleNamespace(page_no=page_no, bbox=_bbox())],
        label=SimpleNamespace(value=label),
    )


def _text_item(text: str, page_no: int = 1, label: str = "text"):
    return SimpleNamespace(
        text=text,
        prov=[SimpleNamespace(page_no=page_no, bbox=_bbox())],
        label=SimpleNamespace(value=label),
    )


def _make_doc(items_per_page: dict) -> Any:
    flat_items = []
    for page_no, items in items_per_page.items():
        # items is List[(text, label)]
        for text, label in items:
            if label in ("section_header", "title"):
                flat_items.append(_heading_item(text, page_no=page_no, label=label))
            else:
                flat_items.append(_text_item(text, page_no=page_no, label=label))

    def iterate_items():
        for it in flat_items:
            yield it, 0

    return SimpleNamespace(iterate_items=iterate_items)


# --------------------------------------------------------------------------
# _section_header_page_to_uir_chunk
# --------------------------------------------------------------------------


class TestBridge:
    def test_all_section_header_returns_uir(self):
        items = [
            _heading_item("Chapter 5", label="section_header"),
            _heading_item("Title page", label="title"),
        ]
        uir = _section_header_page_to_uir_chunk(
            items=items,
            page_number=12,
            page_w=600.0,
            page_h=800.0,
        )
        assert uir is not None
        assert uir.modality == Modality.TEXT
        assert uir.content == "Chapter 5\nTitle page"
        assert uir.parent_heading == "Chapter 5"
        assert uir.locator.type == LocatorType.BBOX
        assert uir.locator.page_number == 12
        assert uir.locator.coordinate_frame == CoordinateFrame.PDF_PAGE_PORTRAIT
        assert uir.extraction_method == "hybrid_chunker_section_header_page"

    def test_mixed_content_returns_none(self):
        items = [
            _heading_item("Chapter 5", label="section_header"),
            _text_item("This is body text", label="text"),
        ]
        uir = _section_header_page_to_uir_chunk(
            items=items,
            page_number=12,
            page_w=600.0,
            page_h=800.0,
        )
        # Mixed content → out of scope per v2.16 contract.
        assert uir is None

    def test_no_heading_text_returns_none(self):
        # Heading items with empty text.
        items = [_heading_item("", label="section_header")]
        uir = _section_header_page_to_uir_chunk(
            items=items,
            page_number=12,
            page_w=600.0,
            page_h=800.0,
        )
        assert uir is None


# --------------------------------------------------------------------------
# _emit_section_header_only_page_chunks
# --------------------------------------------------------------------------


class TestEmit:
    def _make_processor(self, tmp_path) -> V2DocumentProcessor:
        return V2DocumentProcessor(
            output_dir=str(tmp_path),
            vision_provider="none",
        )

    def test_emit_basic_section_header_page(self, tmp_path):
        doc = _make_doc({
            5: [("Chapter 5: The Hunt", "section_header")],
        })
        processor = self._make_processor(tmp_path)
        chunks = processor._emit_section_header_only_page_chunks(
            doc=doc,
            existing_chunks=[],
            dense_index_pages=set(),
            doc_hash="testdoc12345",
            source_file="manual.pdf",
            file_type=FileType.PDF,
            page_dims={5: (600.0, 800.0)},
        )
        assert len(chunks) == 1
        ch = chunks[0]
        assert ch.modality == Modality.TEXT
        assert ch.metadata.chunk_type == ChunkType.HEADING
        assert ch.metadata.extraction_method == "hybrid_chunker_section_header_page"
        assert ch.metadata.hierarchy.parent_heading == "Chapter 5: The Hunt"
        assert ch.metadata.hierarchy.breadcrumb_path == [
            "manual", "Chapter 5: The Hunt", "Page 5"
        ]
        # v2.16 literal: section-header chunks carry level=2
        assert ch.metadata.hierarchy.level == 2
        assert ch.metadata.search_priority == "high"
        assert ch.metadata.refined_content == ch.content
        assert ch.content == "Chapter 5: The Hunt"

    def test_mixed_content_page_skipped(self, tmp_path):
        doc = _make_doc({
            5: [
                ("Chapter 5", "section_header"),
                ("Body paragraph here", "text"),
            ],
        })
        processor = self._make_processor(tmp_path)
        chunks = processor._emit_section_header_only_page_chunks(
            doc=doc,
            existing_chunks=[],
            dense_index_pages=set(),
            doc_hash="testdoc12345",
            source_file="manual.pdf",
            file_type=FileType.PDF,
            page_dims={5: (600.0, 800.0)},
        )
        assert chunks == []

    def test_covered_page_skipped(self, tmp_path):
        doc = _make_doc({
            5: [("Chapter 5", "section_header")],
        })
        # Synthesize an existing chunk on page 5 to mark it covered.
        from mmrag_v2.schema.ingestion_schema import create_text_chunk
        existing = create_text_chunk(
            doc_id="testdoc12345",
            content="Already covered",
            source_file="manual.pdf",
            file_type=FileType.PDF,
            page_number=5,
            bbox=[10, 20, 990, 900],
        )
        processor = self._make_processor(tmp_path)
        chunks = processor._emit_section_header_only_page_chunks(
            doc=doc,
            existing_chunks=[existing],
            dense_index_pages=set(),
            doc_hash="testdoc12345",
            source_file="manual.pdf",
            file_type=FileType.PDF,
            page_dims={5: (600.0, 800.0)},
        )
        assert chunks == []

    def test_dense_index_page_skipped(self, tmp_path):
        doc = _make_doc({
            5: [("Section Headers Index", "section_header")],
        })
        processor = self._make_processor(tmp_path)
        chunks = processor._emit_section_header_only_page_chunks(
            doc=doc,
            existing_chunks=[],
            dense_index_pages={5},  # already handled by step-2 emitter
            doc_hash="testdoc12345",
            source_file="manual.pdf",
            file_type=FileType.PDF,
            page_dims={5: (600.0, 800.0)},
        )
        assert chunks == []

    def test_page_offset_applied(self, tmp_path):
        doc = _make_doc({
            1: [("Chapter 12", "section_header")],
        })
        processor = self._make_processor(tmp_path)
        chunks = processor._emit_section_header_only_page_chunks(
            doc=doc,
            existing_chunks=[],
            dense_index_pages=set(),
            doc_hash="testdoc12345",
            source_file="manual.pdf",
            file_type=FileType.PDF,
            page_dims={150: (600.0, 800.0)},
            page_offset=149,
        )
        assert len(chunks) == 1
        assert chunks[0].metadata.page_number == 150
        assert "Page 150" in chunks[0].metadata.hierarchy.breadcrumb_path

    def test_multiple_header_pages(self, tmp_path):
        doc = _make_doc({
            5: [("Chapter 5", "section_header")],
            7: [("Chapter 6", "title")],
            9: [("Chapter 7", "section_header"), ("Subtitle", "title")],
        })
        processor = self._make_processor(tmp_path)
        chunks = processor._emit_section_header_only_page_chunks(
            doc=doc,
            existing_chunks=[],
            dense_index_pages=set(),
            doc_hash="testdoc12345",
            source_file="manual.pdf",
            file_type=FileType.PDF,
            page_dims={5: (600.0, 800.0), 7: (600.0, 800.0), 9: (600.0, 800.0)},
        )
        assert len(chunks) == 3
        assert {c.metadata.page_number for c in chunks} == {5, 7, 9}
        # Page-9 chunk has two heading lines joined:
        page_9 = next(c for c in chunks if c.metadata.page_number == 9)
        assert page_9.content == "Chapter 7\nSubtitle"
        assert page_9.metadata.hierarchy.parent_heading == "Chapter 7"
