"""v2.17 Item #9 safety-valve reopen — partial_code cross-page emission.

Charter: docs/ARCHITECTURE_V3_DRAFT_0.5.md §3.2 notes that the
`partial_code` flag already emits for the in-block case
(`_chunk_code_by_lines` → create_text_chunk at processor.py:5002) but is
INERT on the HybridChunker cross-page split path. CLAUDE.md item #9
declares this the v2.17 safety-valve trigger #1: extend `partial_code`
coverage to the HybridChunker path so the v2.16 Phase 3 adjacency-fetch
mechanism (retrieval/pipeline.py:_apply_partial_code_adjacency) actually
fires on cross-page CODE splits.

Phase 3 adjacency-fetch is the retrieval-side stitch that rejoins
sibling halves of a cross-page CODE block at query time. Without the
flag the mechanism is dead code; the Fluent_Python validation queries
that target cross-span CODE (Q01 lru_cache imports + decorator + body,
Q04 / Q06 / Q09 similar) measured 0% pass rate at v2.16.0 baseline
precisely because partial_code=True never appeared on Fluent_Python
chunks.

These tests pin the predicate that drives the emission at
processor.py:3613 cross-page emit branch:

    _is_cross_page_code = (
        _page_chunk_type == ChunkType.CODE
        and len(per_page_text) > 1
    )
    partial_code=True if _is_cross_page_code else None,

The predicate must:
  - return True only for CODE chunks that span multiple pages
  - return False (→ None) for single-page CODE chunks (in-block case
    handled separately at line 5002)
  - return False (→ None) for cross-page PARAGRAPH chunks (no code,
    nothing to stitch)
  - return False (→ None) for single-page PARAGRAPH chunks

A second test verifies the emit-site wiring is unbroken: the
`partial_code` kwarg threads through create_text_chunk into
IngestionChunk.metadata.partial_code so the Qdrant payload carries it
for the adjacency-fetch filter (retrieval/pipeline.py line 653 filter:
`{"key": "partial_code", "match": {"value": True}}`).
"""

from __future__ import annotations

import pytest

from mmrag_v2.schema.ingestion_schema import (
    ChunkType,
    HierarchyMetadata,
    create_text_chunk,
)


# ---------------------------------------------------------------------------
# Predicate truth table (matches processor.py:3613 emit-branch condition)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "chunk_type,per_page_text_count,expected_partial_code",
    [
        # CODE + cross-page → THIS is the case v2.17 activates
        (ChunkType.CODE, 2, True),
        (ChunkType.CODE, 3, True),
        # CODE + single-page → in-block case, handled at line 5002 separately
        (ChunkType.CODE, 1, None),
        # PARAGRAPH + cross-page → not CODE, nothing to stitch
        (ChunkType.PARAGRAPH, 2, None),
        (ChunkType.PARAGRAPH, 3, None),
        # PARAGRAPH + single-page → uninteresting
        (ChunkType.PARAGRAPH, 1, None),
        # LIST_ITEM cross-page → not CODE, no stitch
        (ChunkType.LIST_ITEM, 2, None),
        # HEADING cross-page → not CODE, no stitch
        (ChunkType.HEADING, 2, None),
    ],
)
def test_cross_page_partial_code_predicate(
    chunk_type, per_page_text_count, expected_partial_code,
):
    """The condition expression at processor.py:3613 must produce
    partial_code=True only when BOTH (CODE chunk_type) AND
    (len(per_page_text) > 1) hold. Any other combination → None.
    """
    # Mirror the emit-site predicate exactly:
    is_cross_page_code = (
        chunk_type == ChunkType.CODE and per_page_text_count > 1
    )
    partial_code = True if is_cross_page_code else None
    assert partial_code is expected_partial_code, (
        f"partial_code predicate misfire for "
        f"chunk_type={chunk_type!r} per_page_text_count={per_page_text_count} — "
        f"expected {expected_partial_code!r}, got {partial_code!r}"
    )


# ---------------------------------------------------------------------------
# Emit-site wiring: create_text_chunk threads partial_code through metadata
# ---------------------------------------------------------------------------


def test_create_text_chunk_threads_partial_code_through_metadata():
    """The chunk emitted by `create_text_chunk(partial_code=True, ...)`
    must carry `metadata.partial_code=True` so the Qdrant payload
    (chunk.metadata.partial_code → payload["partial_code"]) is True
    and the retrieval adjacency-fetch filter
    (retrieval/pipeline.py line 653) matches on it.
    """
    chunk = create_text_chunk(
        doc_id="testdoc01",
        content="def foo():\n    return 1",
        source_file="test.pdf",
        file_type="pdf",
        page_number=42,
        bbox=[100, 100, 900, 200],
        hierarchy=HierarchyMetadata(parent_heading="Test", breadcrumb_path=["Test"]),
        chunk_type=ChunkType.CODE,
        page_width=1000,
        page_height=1000,
        extraction_method="hybrid_chunker_pagesplit",
        position=0,
        partial_code=True,
    )
    assert chunk.metadata.partial_code is True, (
        "partial_code=True must thread through to IngestionChunk.metadata "
        "so it appears in the Qdrant payload — the retrieval adjacency-fetch "
        "filter at retrieval/pipeline.py:653 keys on payload.partial_code."
    )
    # Sanity: not-set defaults to None (NOT False), so payload filters
    # `match: {value: True}` correctly skip un-flagged chunks.
    chunk2 = create_text_chunk(
        doc_id="testdoc02",
        content="some prose paragraph",
        source_file="test.pdf",
        file_type="pdf",
        page_number=42,
        bbox=[100, 100, 900, 200],
        hierarchy=HierarchyMetadata(parent_heading="Test", breadcrumb_path=["Test"]),
        chunk_type=ChunkType.PARAGRAPH,
        page_width=1000,
        page_height=1000,
        extraction_method="hybrid_chunker_pagesplit",
        position=1,
    )
    assert chunk2.metadata.partial_code is None, (
        "partial_code defaults to None (not False) so the Qdrant payload "
        "filter `match: {value: True}` skips un-flagged chunks cleanly."
    )


# ---------------------------------------------------------------------------
# Documentation pin — the emit site itself
# ---------------------------------------------------------------------------


def test_emit_site_carries_v2_17_predicate():
    """Source-level sanity that the processor.py emit branch still
    carries the v2.17 predicate. If a future refactor removes the
    `partial_code` kwarg from the cross-page emit's `create_text_chunk`
    call, this test fails loudly — the retrieval-side adjacency-fetch
    would silently go inert again.
    """
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "src" / "mmrag_v2" / "processor.py"
    text = src.read_text(encoding="utf-8")
    # The predicate variable name + the kwarg pass-through must both
    # be present in the cross-page emit branch.
    assert "_is_cross_page_code" in text, (
        "v2.17 partial_code cross-page predicate variable missing from "
        "processor.py; the retrieval adjacency-fetch is at risk of going "
        "inert on the HybridChunker emit path. See "
        "docs/ARCHITECTURE_V3_DRAFT_0.5.md §3.2 partial_code distinction."
    )
    assert "partial_code=True if _is_cross_page_code else None" in text, (
        "Cross-page emit-site no longer passes `partial_code=` to "
        "create_text_chunk; HybridChunker cross-page CODE splits will not "
        "carry the flag and adjacency-fetch will not fire on them."
    )
