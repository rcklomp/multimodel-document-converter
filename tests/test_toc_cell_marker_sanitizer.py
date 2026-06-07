from mmrag_v2.batch_processor import BatchProcessor
from mmrag_v2.schema.ingestion_schema import (
    ChunkType,
    FileType,
    HierarchyMetadata,
    Modality,
    create_image_chunk,
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


def test_toc_cell_marker_sanitizer_preserves_empty_text_chunk() -> None:
    # The sanitizer's sole job is stripping TOC markers from TEXT; it must NOT
    # drop chunks (that silently deleted empty-content IMAGE chunks ->
    # MISSING_PAGES). Empty TEXT is removed at the canonical boundary by
    # _drop_empty_text_chunks_before_metadata, not here.
    processor = BatchProcessor(output_dir="/tmp/mmrag-test", vision_provider="none")
    assert len(processor._sanitize_toc_cell_markers([_chunk("   ")])) == 1


def test_toc_cell_marker_sanitizer_preserves_image_chunk() -> None:
    # Regression (Cluster D, 2026-06-06): an IMAGE chunk carries no text
    # content; the sanitizer must keep it so image-only pages are not orphaned.
    processor = BatchProcessor(output_dir="/tmp/mmrag-test", vision_provider="none")
    img = create_image_chunk(
        doc_id="doc",
        content="",  # offline/MinerU image: no visual description
        source_file="photos.pdf",
        file_type=FileType.PDF,
        page_number=6,
        asset_path="assets/doc_0006_image_000.png",
        bbox=[100, 100, 900, 900],
        position=0,
    )
    out = processor._sanitize_toc_cell_markers([img])
    assert len(out) == 1
    assert out[0].modality == Modality.IMAGE


def test_toc_cell_marker_sanitizer_does_not_demote_plain_chunks() -> None:
    processor = BatchProcessor(output_dir="/tmp/mmrag-test", vision_provider="none")
    chunk = _chunk("This ordinary paragraph has no Docling table cell markers.")

    sanitized = processor._sanitize_toc_cell_markers([chunk])

    assert len(sanitized) == 1
    assert sanitized[0].content == "This ordinary paragraph has no Docling table cell markers."
    assert sanitized[0].metadata.chunk_type == ChunkType.PARAGRAPH
