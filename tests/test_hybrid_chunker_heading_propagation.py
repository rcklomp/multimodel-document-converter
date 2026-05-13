from __future__ import annotations

from mmrag_v2.batch_processor import BatchProcessor
from mmrag_v2.schema.ingestion_schema import (
    ChunkType,
    FileType,
    HierarchyMetadata,
    create_text_chunk,
)
from mmrag_v2.state.context_state import is_valid_heading


def _processor(tmp_path):
    return BatchProcessor(
        output_dir=str(tmp_path),
        batch_size=10,
        vision_provider="none",
        enable_ocr=False,
    )


def _text(content: str, page: int, heading: str | None, position: int = 0):
    breadcrumb = ["devlin"]
    if heading:
        breadcrumb.append(heading)
    breadcrumb.append(f"Page {page}")
    return create_text_chunk(
        doc_id="devlin",
        content=content,
        source_file="devlin.pdf",
        file_type=FileType.PDF,
        page_number=page,
        hierarchy=HierarchyMetadata(
            parent_heading=heading,
            breadcrumb_path=breadcrumb,
            level=len(breadcrumb),
        ),
        chunk_type=ChunkType.PARAGRAPH,
        extraction_method="hybrid_chunker_pagesplit" if heading is None else "hybrid_chunker",
        position=position,
    )


def test_devlin_shape_batch_boundary_carries_forward(tmp_path) -> None:
    """Partial TOC coverage must not disable HybridChunker fallback propagation."""
    processor = _processor(tmp_path)
    processor._toc_headings = {
        1: ["Start"],
        "__heading_map__": {"Start": ["Start"]},
    }
    chunks = [
        _text("Section body before batch boundary.", page=10, heading="Retrieval Basics", position=1),
        _text("Body continues at the next batch start.", page=11, heading=None, position=2),
    ]

    result = processor._propagate_headings(chunks)

    assert result[1].metadata.hierarchy.parent_heading == "Retrieval Basics"
    assert result[1].metadata.hierarchy.breadcrumb_path == [
        "devlin",
        "Retrieval Basics",
        "Page 11",
    ]


def test_propagation_does_not_overwrite_explicit_heading(tmp_path) -> None:
    processor = _processor(tmp_path)
    processor._toc_headings = {
        1: ["Start"],
        "__heading_map__": {"Start": ["Start"]},
    }
    chunks = [
        _text("Prior section body.", page=10, heading="Retrieval Basics", position=1),
        _text("New section body.", page=11, heading="Knowledge Graphs", position=2),
        _text("Continuation under explicit new section.", page=11, heading=None, position=3),
    ]

    result = processor._propagate_headings(chunks)

    assert result[1].metadata.hierarchy.parent_heading == "Knowledge Graphs"
    assert result[2].metadata.hierarchy.parent_heading == "Knowledge Graphs"


def test_propagation_unit_test_from_b429cb5_still_passes(tmp_path) -> None:
    """The original no-TOC forward propagation contract remains intact."""
    processor = _processor(tmp_path)
    processor._toc_headings = {}
    chunks = [
        _text("Known section body.", page=1, heading="Chapter 3: Implementation", position=1),
        _text("Same section without explicit HybridChunker metadata.", page=2, heading=None, position=2),
    ]

    result = processor._propagate_headings(chunks)

    assert result[1].metadata.hierarchy.parent_heading == "Chapter 3: Implementation"
    assert result[1].metadata.hierarchy.breadcrumb_path == [
        "devlin",
        "Chapter 3: Implementation",
        "Page 2",
    ]


def test_unordered_pagesplit_sibling_uses_same_page_heading(tmp_path) -> None:
    processor = _processor(tmp_path)
    processor._toc_headings = {
        1: ["Start"],
        "__heading_map__": {"Start": ["Start"]},
    }
    unordered_pagesplit = _text(
        "Same-page continuation emitted before the explicit heading.",
        page=81,
        heading=None,
        position=1,
    )
    explicit_heading = _text(
        "The page's explicit HybridChunker heading body.",
        page=81,
        heading="The Myth of the Omniscient Model",
        position=2,
    )

    result = processor._propagate_headings([unordered_pagesplit, explicit_heading])

    assert result[0].metadata.hierarchy.parent_heading == "The Myth of the Omniscient Model"


def test_garbage_repeated_heading_does_not_seed_carry_forward(tmp_path) -> None:
    processor = _processor(tmp_path)
    processor._toc_headings = {}
    chunks = [
        _text("Type Type TypeTypeTypeType", page=283, heading="Type Type TypeTypeTypeType", position=1),
        _text("A body paragraph after the corrupted divider.", page=284, heading=None, position=2),
    ]

    result = processor._propagate_headings(chunks)

    assert is_valid_heading("Type Type TypeTypeTypeType") is False
    assert result[0].metadata.hierarchy.parent_heading is None
    assert result[1].metadata.hierarchy.parent_heading is None


def test_code_shape_heading_does_not_seed_carry_forward(tmp_path) -> None:
    processor = _processor(tmp_path)
    processor._toc_headings = {}
    chunks = [
        _text("GRAPH DATA: {", page=120, heading="GRAPH DATA: {", position=1),
        _text("Narrative text after a code payload.", page=121, heading=None, position=2),
    ]

    result = processor._propagate_headings(chunks)

    assert is_valid_heading("GRAPH DATA: {") is False
    assert result[0].metadata.hierarchy.parent_heading is None
    assert result[1].metadata.hierarchy.parent_heading is None


def test_real_heading_wins_over_garbage_heading_on_same_page(tmp_path) -> None:
    processor = _processor(tmp_path)
    processor._toc_headings = {
        1: ["Start"],
        "__heading_map__": {"Start": ["Start"]},
    }
    unordered_pagesplit = _text(
        "Continuation emitted before page heading.",
        page=286,
        heading=None,
        position=1,
    )
    garbage_heading = _text(
        "Type Type TypeTypeTypeType",
        page=286,
        heading="Type Type TypeTypeTypeType",
        position=2,
    )
    explicit_heading = _text(
        "Pipeline steps:",
        page=286,
        heading="Pipeline steps:",
        position=3,
    )

    result = processor._propagate_headings([unordered_pagesplit, garbage_heading, explicit_heading])

    assert result[0].metadata.hierarchy.parent_heading == "Pipeline steps:"
    assert result[1].metadata.hierarchy.parent_heading is None
    assert result[2].metadata.hierarchy.parent_heading == "Pipeline steps:"
