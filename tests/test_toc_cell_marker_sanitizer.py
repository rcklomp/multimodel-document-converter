from mmrag_v2.batch_processor import BatchProcessor
from mmrag_v2.schema.ingestion_schema import (
    ChunkType,
    FileType,
    HierarchyMetadata,
    create_text_chunk,
)


def _chunk(content: str):
    return create_text_chunk(
        doc_id="doc",
        content=content,
        source_file="manual.pdf",
        file_type=FileType.PDF,
        page_number=6,
        hierarchy=HierarchyMetadata(
            parent_heading="Contents",
            breadcrumb_path=["manual", "Contents", "Page 6"],
            level=3,
        ),
        chunk_type=ChunkType.PARAGRAPH,
        position=0,
    )


def test_toc_cell_marker_sanitizer_keeps_chunk_and_strips_markers() -> None:
    processor = BatchProcessor(output_dir="/tmp/mmrag-test", vision_provider="none")
    chunk = _chunk(
        "brief contents, 1 = Part 1 Foundations 1, 2 = Chapter 1 3, "
        "3 = Chapter 2 17, 4 = Part 2 Creating RAG systems 31"
    )

    sanitized = processor._sanitize_toc_cell_markers([chunk])

    assert len(sanitized) == 1
    assert ", 1 =" not in sanitized[0].content
    assert "Part 1 Foundations" in sanitized[0].content
    assert sanitized[0].metadata.chunk_type == ChunkType.LIST_ITEM
    assert sanitized[0].metadata.search_priority == "low"


def test_toc_cell_marker_sanitizer_still_drops_empty_chunks() -> None:
    processor = BatchProcessor(output_dir="/tmp/mmrag-test", vision_provider="none")
    assert processor._sanitize_toc_cell_markers([_chunk("   ")]) == []
