import inspect

from mmrag_v2.batch_processor import BatchProcessor
from mmrag_v2.schema.ingestion_schema import (
    ChunkType,
    FileType,
    HierarchyMetadata,
    create_text_chunk,
)


def _make_processor(tmp_path):
    return BatchProcessor(
        output_dir=str(tmp_path),
        batch_size=3,
        vision_provider="none",
        enable_ocr=False,
    )


def _chunk(
    content: str,
    page_number: int = 1,
    chunk_type: ChunkType = ChunkType.PARAGRAPH,
    refined_content: str | None = None,
):
    return create_text_chunk(
        doc_id="doc123",
        content=content,
        source_file="book.pdf",
        file_type=FileType.PDF,
        page_number=page_number,
        hierarchy=HierarchyMetadata(
            parent_heading=None,
            breadcrumb_path=["book", f"Page {page_number}"],
            level=2,
        ),
        chunk_type=chunk_type,
        refined_content=refined_content,
    )


def test_orphan_preposition_moves_to_capitalized_next_chunk(tmp_path):
    processor = _make_processor(tmp_path)
    current = _chunk("Cover illustration\nBY", refined_content="Cover illustration\nBY")
    nxt = _chunk("Mary GrandPre", refined_content="Mary GrandPre")

    merged = processor._merge_hungry_operators([current, nxt])

    assert len(merged) == 2
    assert merged[0].content == "Cover illustration"
    assert merged[1].content == "BY Mary GrandPre"
    assert merged[0].metadata.refined_content == "Cover illustration"
    assert merged[1].metadata.refined_content == "BY Mary GrandPre"


def test_punctuated_orphan_preposition_is_stitched_without_punctuation(tmp_path):
    processor = _make_processor(tmp_path)
    current = _chunk("Translated\nBY,")
    nxt = _chunk("Jane Doe")

    merged = processor._merge_hungry_operators([current, nxt])

    assert merged[0].content == "Translated"
    assert merged[1].content == "BY Jane Doe"


def test_preposition_only_chunk_is_removed_after_stitching(tmp_path):
    processor = _make_processor(tmp_path)
    current = _chunk("BY")
    nxt = _chunk("Mary GrandPre")

    merged = processor._merge_hungry_operators([current, nxt])

    assert len(merged) == 1
    assert merged[0].content == "BY Mary GrandPre"


def test_mid_sentence_preposition_does_not_move(tmp_path):
    processor = _make_processor(tmp_path)
    current = _chunk("Temperature setpoints range from")
    nxt = _chunk("15C to 35C depending on crop.")

    merged = processor._merge_hungry_operators([current, nxt])

    assert merged[0].content == "Temperature setpoints range from"
    assert merged[1].content == "15C to 35C depending on crop."


def test_noun_like_boundary_tokens_do_not_move(tmp_path):
    processor = _make_processor(tmp_path)
    current = _chunk("Warranty status\nEND")
    nxt = _chunk("Mary GrandPre")

    merged = processor._merge_hungry_operators([current, nxt])

    assert merged[0].content == "Warranty status\nEND"
    assert merged[1].content == "Mary GrandPre"


def test_orphan_preposition_requires_capitalized_next_chunk(tmp_path):
    processor = _make_processor(tmp_path)
    current = _chunk("Cover illustration\nBY")
    nxt = _chunk("the design team")

    merged = processor._merge_hungry_operators([current, nxt])

    assert merged[0].content == "Cover illustration\nBY"
    assert merged[1].content == "the design team"


def test_orphan_preposition_does_not_cross_distant_pages(tmp_path):
    processor = _make_processor(tmp_path)
    current = _chunk("Cover illustration\nBY", page_number=1)
    nxt = _chunk("Mary GrandPre", page_number=4)

    merged = processor._merge_hungry_operators([current, nxt])

    assert merged[0].content == "Cover illustration\nBY"
    assert merged[1].content == "Mary GrandPre"


def test_final_boundary_repair_pipeline_invokes_semantic_stitching_in_order(tmp_path, monkeypatch):
    processor = _make_processor(tmp_path)
    calls: list[str] = []

    def _record(name):
        def _inner(chunks):
            calls.append(name)
            return chunks

        return _inner

    monkeypatch.setattr(processor, "_strip_trailing_headings", _record("strip"))
    monkeypatch.setattr(processor, "_merge_hungry_operators", _record("hungry"))
    monkeypatch.setattr(processor, "_merge_mid_sentence_chunks", _record("mid_sentence"))
    monkeypatch.setattr(processor, "_deduplicate_chunk_overlap", _record("dedup"))

    processor._apply_final_boundary_repairs([_chunk("Cover illustration\nBY"), _chunk("Mary")])

    assert calls == ["strip", "hungry", "mid_sentence", "dedup"]


def test_process_pdf_routes_through_final_boundary_repair_bridge():
    source = inspect.getsource(BatchProcessor.process_pdf)

    assert "_apply_final_boundary_repairs(all_chunks)" in source
    assert source.index("_apply_final_boundary_repairs(all_chunks)") < source.index(
        "_apply_vision_aided_front_matter_detection(all_chunks)"
    )
