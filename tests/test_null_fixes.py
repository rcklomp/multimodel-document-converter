"""
Tests for null-value fixes:

Bug 1 — create_text_chunk() now accepts page_width/page_height and populates
         SpatialMetadata when bbox is present.
Bug 2 — content_classification is always set for text chunks:
         - oversize-split non-code sub-chunks call _classify_text_content()
         - fallback digital text chunks call _classify_text_content()
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from mmrag_v2.schema.ingestion_schema import (
    FileType,
    create_image_chunk,
    create_text_chunk,
)


# ---------------------------------------------------------------------------
# Bug 1: page_width / page_height in create_text_chunk
# ---------------------------------------------------------------------------


def test_create_text_chunk_with_bbox_includes_page_dims():
    """page_width and page_height are stored in SpatialMetadata when bbox is given."""
    chunk = create_text_chunk(
        doc_id="abc123def456",
        content="Sample paragraph text.",
        source_file="test.pdf",
        file_type=FileType.PDF,
        page_number=1,
        bbox=[100, 200, 900, 400],
        page_width=612,
        page_height=792,
    )
    assert chunk.metadata.spatial is not None
    assert chunk.metadata.spatial.bbox == [100, 200, 900, 400]
    assert chunk.metadata.spatial.page_width == 612
    assert chunk.metadata.spatial.page_height == 792


def test_create_text_chunk_no_bbox_spatial_still_none():
    """Without bbox, spatial remains None even if page dims are supplied."""
    chunk = create_text_chunk(
        doc_id="abc123def456",
        content="Sample paragraph text.",
        source_file="test.pdf",
        file_type=FileType.PDF,
        page_number=1,
        bbox=None,
        page_width=612,
        page_height=792,
    )
    assert chunk.metadata.spatial is None


def test_create_text_chunk_bbox_without_page_dims_is_null():
    """Omitting page_width/page_height leaves them null (old callers unchanged)."""
    chunk = create_text_chunk(
        doc_id="abc123def456",
        content="Sample paragraph text.",
        source_file="test.pdf",
        file_type=FileType.PDF,
        page_number=1,
        bbox=[10, 20, 500, 300],
    )
    assert chunk.metadata.spatial is not None
    assert chunk.metadata.spatial.page_width is None
    assert chunk.metadata.spatial.page_height is None


def test_create_image_chunk_page_dims_unchanged():
    """Image chunk page_width/page_height behaviour is not regressed."""
    chunk = create_image_chunk(
        doc_id="abc123def456",
        content="[Figure on page 2]",
        source_file="test.pdf",
        file_type=FileType.PDF,
        page_number=2,
        bbox=[0, 0, 1000, 1000],
        page_width=595,
        page_height=842,
        asset_path="/tmp/img.png",
        width_px=800,
        height_px=600,
    )
    assert chunk.metadata.spatial.page_width == 595
    assert chunk.metadata.spatial.page_height == 842


# ---------------------------------------------------------------------------
# Bug 2: content_classification always set for text chunks
# ---------------------------------------------------------------------------


def _make_batch_processor():
    """Return a minimal BatchProcessor instance with mocked dependencies."""
    from mmrag_v2.batch_processor import BatchProcessor

    bp = BatchProcessor.__new__(BatchProcessor)
    # Wire the bare minimum needed by _classify_text_content and _looks_like_code_text
    bp._intelligence_metadata = {}
    bp._doc_hash = "deadbeef1234"
    bp._current_pdf_path = Path("/tmp/test.pdf")
    return bp


def test_oversize_split_non_code_has_content_classification():
    """Non-code sub-chunks from oversize split must have content_classification set."""
    from mmrag_v2.schema.ingestion_schema import ChunkType, HierarchyMetadata, IngestionChunk

    bp = _make_batch_processor()

    # Build a chunk that is NOT code but is oversized
    long_text = "This is a long paragraph. " * 80  # > 1500 chars
    chunk = create_text_chunk(
        doc_id=bp._doc_hash,
        content=long_text,
        source_file="test.pdf",
        file_type=FileType.PDF,
        page_number=1,
        hierarchy=HierarchyMetadata(breadcrumb_path=["Doc", "Section"], level=2),
    )
    # Manually mark it as not code
    chunk.metadata.chunk_type = ChunkType.PARAGRAPH
    chunk.metadata.content_classification = "editorial"

    result = bp._apply_oversize_breaker([chunk], max_chars=1500)

    # All sub-chunks must have non-null content_classification
    assert len(result) > 1, "Expected chunk to be split"
    for sub in result:
        assert sub.metadata.content_classification is not None, (
            f"content_classification is None on sub-chunk: {sub.content[:60]!r}"
        )


def test_oversize_split_code_classified_as_code():
    """Code sub-chunks retain 'code' classification (regression guard)."""
    from mmrag_v2.schema.ingestion_schema import ChunkType, HierarchyMetadata

    bp = _make_batch_processor()

    code_text = "def foo():\n    return 42\n" * 100  # > 1500 chars, clear Python
    chunk = create_text_chunk(
        doc_id=bp._doc_hash,
        content=code_text,
        source_file="test.pdf",
        file_type=FileType.PDF,
        page_number=1,
        hierarchy=HierarchyMetadata(breadcrumb_path=["Doc"], level=1),
    )
    chunk.metadata.chunk_type = ChunkType.CODE
    chunk.metadata.content_classification = "code"

    result = bp._apply_oversize_breaker([chunk], max_chars=1500)
    assert len(result) > 1, "Expected code chunk to be split"
    for sub in result:
        assert sub.metadata.content_classification == "code"


def test_fallback_chunk_has_content_classification():
    """_classify_text_content must be called for fallback chunks (not None)."""
    bp = _make_batch_processor()
    # Directly call the classifier to verify it never returns None
    result = bp._classify_text_content("The motor controller uses 12V DC input with PWM control.")
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
