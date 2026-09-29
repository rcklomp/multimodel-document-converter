"""PLAN_QUALITY_REMEDIATION_V1 WP-B3: split parts keep the PARENT's breadcrumb and level.

Defect: the oversize breaker (and the sibling token-limit splitter) appended a positional
"[Oversize Split n/m]" / "[Split n/m]" leaf to breadcrumb_path and bumped the level by one, keeping
REQ-HIER-04 (level == depth) true. breadcrumb_path is embedded into the vectors (to_embedding_text and
the ingest prefix), so every split chunk embedded a synthetic node as if it were a section. Adjacency
between the parts is carried by the "_o<n>" chunk_id suffix, which is unchanged.
No consumer of the marker text exists (grep over src, scripts, tests).
"""

from mmrag_v2.batch_processor import BatchProcessor
from mmrag_v2.schema.ingestion_schema import (
    ChunkType,
    FileType,
    HierarchyMetadata,
    create_text_chunk,
)


def _bp(tmp_path):
    return BatchProcessor(
        output_dir=str(tmp_path), batch_size=3, vision_provider="none", enable_ocr=False
    )


def _chunk(breadcrumb, level):
    c = create_text_chunk(
        doc_id="d",
        content="This is a long paragraph. " * 120,
        source_file="test.pdf",
        file_type=FileType.PDF,
        page_number=1,
        hierarchy=HierarchyMetadata(breadcrumb_path=breadcrumb, level=level),
    )
    c.metadata.chunk_type = ChunkType.PARAGRAPH
    c.metadata.content_classification = "editorial"
    return c


def test_oversize_parts_keep_the_parent_breadcrumb_and_level(tmp_path):
    parent = _chunk(["Doc", "B. PSO applied to MPPT", "Page 5"], 3)
    parts = _bp(tmp_path)._apply_oversize_breaker([parent], max_chars=1500)
    assert len(parts) >= 2
    for p in parts:
        assert p.metadata.hierarchy.breadcrumb_path == ["Doc", "B. PSO applied to MPPT", "Page 5"]
        assert p.metadata.hierarchy.level == 3
        assert "Oversize Split" not in " ".join(p.metadata.hierarchy.breadcrumb_path)


def test_oversize_parts_keep_the_adjacency_suffix_in_their_chunk_ids(tmp_path):
    parent = _chunk(["Doc", "Section", "Page 1"], 3)
    parent_id = parent.chunk_id
    parts = _bp(tmp_path)._apply_oversize_breaker([parent], max_chars=1500)
    # the first part keeps the parent's id, later parts get the "_o<n>" adjacency suffix
    assert parts[0].chunk_id == parent_id
    assert [p.chunk_id for p in parts[1:]] == [
        f"{parent_id}_o{i + 2}" for i in range(len(parts) - 1)
    ]


def test_token_limit_split_parts_keep_the_parent_breadcrumb_and_level(tmp_path):
    parent = _chunk(["Doc", "Section", "Page 1"], 3)
    parent.content = "word " * 4000
    parts = _bp(tmp_path)._validate_token_limit_per_chunk([parent], max_tokens=200)[0]
    assert len(parts) >= 2
    for p in parts:
        assert p.metadata.hierarchy.breadcrumb_path == ["Doc", "Section", "Page 1"]
        assert p.metadata.hierarchy.level == 3
