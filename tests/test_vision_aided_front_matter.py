import inspect

from mmrag_v2.batch_processor import BatchProcessor
from mmrag_v2.schema.ingestion_schema import (
    ChunkType,
    FileType,
    HierarchyMetadata,
    create_image_chunk,
    create_text_chunk,
)


def _processor(tmp_path):
    return BatchProcessor(
        output_dir=str(tmp_path),
        batch_size=3,
        vision_provider="none",
        enable_ocr=False,
    )


def _text(content: str, page: int, heading: str | None):
    chunk = create_text_chunk(
        doc_id="doc123",
        content=content,
        source_file="book.pdf",
        file_type=FileType.PDF,
        page_number=page,
        hierarchy=HierarchyMetadata(
            parent_heading=heading,
            breadcrumb_path=["book", heading or f"Page {page}", f"Page {page}"],
            level=3,
        ),
        chunk_type=ChunkType.PARAGRAPH,
    )
    chunk.metadata.hierarchy.parent_heading = heading
    return chunk


def _image(
    page: int,
    visual_description: str = "Publisher logo and cover ornament.",
    extraction_method: str = "docling",
):
    return create_image_chunk(
        doc_id="doc123",
        content="",
        source_file="book.pdf",
        file_type=FileType.PDF,
        page_number=page,
        asset_path=f"assets/page_{page}.png",
        bbox=[0, 0, 500, 500],
        visual_description=visual_description,
        extraction_method=extraction_method,
    )


def test_pre_chapter_visual_page_demotes_author_heading_to_front_matter(tmp_path):
    processor = _processor(tmp_path)
    author = _text("J. K. Rowling", page=1, heading="J. K. Rowling")
    chapter = _text("The story begins.", page=5, heading="Chapter 1")

    result = processor._apply_vision_aided_front_matter_detection(
        [_image(1), author, chapter]
    )

    assert result[1].metadata.hierarchy.parent_heading == "Front Matter"
    assert result[1].metadata.hierarchy.breadcrumb_path == [
        "book",
        "Front Matter",
        "Page 1",
    ]
    assert result[2].metadata.hierarchy.parent_heading == "Chapter 1"


def test_shadow_visual_page_uses_same_front_matter_rule(tmp_path):
    processor = _processor(tmp_path)
    publisher = _text("Acme Publishing", page=2, heading="Acme Publishing")
    chapter = _text("Body text.", page=4, heading="Chapter Two")

    result = processor._apply_vision_aided_front_matter_detection(
        [_image(2, extraction_method="shadow"), publisher, chapter]
    )

    assert result[1].metadata.hierarchy.parent_heading == "Front Matter"


def test_text_only_front_page_does_not_demote(tmp_path):
    processor = _processor(tmp_path)
    author = _text("J. K. Rowling", page=1, heading="J. K. Rowling")
    chapter = _text("The story begins.", page=5, heading="Chapter 1")

    result = processor._apply_vision_aided_front_matter_detection([author, chapter])

    assert result[0].metadata.hierarchy.parent_heading == "J. K. Rowling"


def test_numbered_section_heading_is_not_demoted(tmp_path):
    processor = _processor(tmp_path)
    section = _text("Method details.", page=2, heading="1.1 System Overview")
    chapter = _text("Body text.", page=5, heading="Chapter 1")

    result = processor._apply_vision_aided_front_matter_detection([_image(2), section, chapter])

    assert result[1].metadata.hierarchy.parent_heading == "1.1 System Overview"


def test_post_chapter_visual_page_is_not_demoted(tmp_path):
    processor = _processor(tmp_path)
    chapter = _text("Body text.", page=3, heading="Chapter 1")
    illustrated_body = _text("Diagram discussion.", page=4, heading="J. K. Rowling")

    result = processor._apply_vision_aided_front_matter_detection(
        [chapter, _image(4, visual_description="An illustration in the chapter."), illustrated_body]
    )

    assert result[2].metadata.hierarchy.parent_heading == "J. K. Rowling"


def test_no_chapter_boundary_requires_explicit_front_matter_visual_signal(tmp_path):
    processor = _processor(tmp_path)
    body = _text("Diagram discussion.", page=7, heading="Overview")

    result = processor._apply_vision_aided_front_matter_detection(
        [_image(7, visual_description="A technical wiring diagram."), body]
    )

    assert result[1].metadata.hierarchy.parent_heading == "Overview"


def test_legacy_vision_gate_wrapper_routes_to_front_matter_detector(tmp_path):
    processor = _processor(tmp_path)
    author = _text("J. K. Rowling", page=1, heading="J. K. Rowling")
    chapter = _text("The story begins.", page=5, heading="Chapter 1")

    result = processor._vision_gate_headings([_image(1), author, chapter])

    assert result[1].metadata.hierarchy.parent_heading == "Front Matter"


def test_process_pdf_routes_front_matter_after_all_heading_assignment_paths():
    source = inspect.getsource(BatchProcessor.process_pdf)

    assert "_apply_vision_aided_front_matter_detection(all_chunks)" in source
    assert source.index("_infer_headings_from_text(all_chunks)") < source.index(
        "_apply_vision_aided_front_matter_detection(all_chunks)"
    )
    assert source.index("_propagate_headings(all_chunks)") < source.index(
        "_apply_vision_aided_front_matter_detection(all_chunks)"
    )
