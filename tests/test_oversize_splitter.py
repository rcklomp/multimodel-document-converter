import fitz

from mmrag_v2.batch_processor import BatchProcessor
from mmrag_v2.schema.ingestion_schema import ChunkType, FileType, HierarchyMetadata, create_text_chunk


def _make_batch_processor(tmp_path):
    return BatchProcessor(
        output_dir=str(tmp_path),
        batch_size=3,
        vision_provider="none",
        enable_ocr=False,
    )


def _make_text_chunk(
    content: str,
    page_number: int = 1,
    chunk_type: ChunkType = ChunkType.PARAGRAPH,
    content_classification=None,
):
    return create_text_chunk(
        doc_id="doc123",
        content=content,
        source_file="manual.pdf",
        file_type=FileType.PDF,
        page_number=page_number,
        hierarchy=HierarchyMetadata(
            parent_heading=None,
            breadcrumb_path=["manual", f"Page {page_number}"],
            level=2,
        ),
        chunk_type=chunk_type,
        content_classification=content_classification,
    )


def test_oversize_splitter_progress_with_early_paragraph_break(tmp_path):
    processor = _make_batch_processor(tmp_path)

    # Pathological shape: paragraph break appears very early and then never again.
    # The splitter must still make steady progress and terminate quickly.
    text = "H\n\n" + ("technicalmanual " * 600)

    parts = processor._split_nearest_paragraph_breaks(
        text=text,
        max_chars=500,
        overlap_chars=120,
    )

    assert parts
    assert all(p.strip() for p in parts)
    assert len(parts) < 40
    assert max(len(p) for p in parts) <= 500


def test_oversize_splitter_hard_caps_when_nearest_break_is_after_target(tmp_path):
    processor = _make_batch_processor(tmp_path)

    # Break before target is farther than break after target.
    # Splitter should still enforce max_chars strictly.
    text = ("A" * 420) + "\n\n" + ("B" * 103) + "\n\n" + ("C" * 800)

    parts = processor._split_nearest_paragraph_breaks(
        text=text,
        max_chars=500,
        overlap_chars=120,
    )

    assert parts
    assert max(len(p) for p in parts) <= 500


def test_micro_chunks_attach_to_neighbor_text(tmp_path):
    processor = _make_batch_processor(tmp_path)

    micro = _make_text_chunk("abc")
    body = _make_text_chunk(
        "This paragraph is intentionally long enough to remain a normal body chunk."
    )

    merged = processor._merge_micro_text_chunks([micro, body], max_chars=30)

    assert len(merged) == 1
    assert merged[0].content.startswith("abc ")


def test_oversize_splitter_prefers_sentence_boundary_before_hard_split(tmp_path):
    processor = _make_batch_processor(tmp_path)

    # No paragraph/newline breaks available: fallback should split on sentence
    # boundary before doing a raw hard cut.
    text = ("This sentence is intentionally medium length. " * 80).strip()
    parts = processor._split_nearest_paragraph_breaks(
        text=text,
        max_chars=220,
        overlap_chars=40,
    )

    assert parts
    assert max(len(p) for p in parts) <= 220
    assert parts[0].endswith((".", "!", "?"))


def test_merge_micro_chunks_glues_label_to_code(tmp_path):
    processor = _make_batch_processor(tmp_path)

    heading = _make_text_chunk("Recipe 1:", page_number=2)
    code = _make_text_chunk(
        "def build():\n    return 1",
        page_number=2,
        chunk_type=ChunkType.CODE,
        content_classification="code",
    )

    merged = processor._merge_micro_text_chunks([heading, code], max_chars=30)

    assert len(merged) == 1
    assert merged[0].metadata.chunk_type == ChunkType.CODE
    assert merged[0].content.startswith("Recipe 1:\n")


def test_merge_micro_chunks_merges_short_list_item_runs(tmp_path):
    processor = _make_batch_processor(tmp_path)

    c1 = _make_text_chunk("Installing dependencies", page_number=3, chunk_type=ChunkType.LIST_ITEM)
    c2 = _make_text_chunk("Configuring the runtime", page_number=3, chunk_type=ChunkType.LIST_ITEM)
    c3 = _make_text_chunk("Running your first script", page_number=3, chunk_type=ChunkType.LIST_ITEM)

    merged = processor._merge_micro_text_chunks([c1, c2, c3], max_chars=30)

    assert len(merged) == 1
    assert merged[0].content.count("\n") >= 2


def test_recovery_classifier_is_always_non_null(tmp_path):
    processor = _make_batch_processor(tmp_path)

    assert processor._classify_recovery_text_content("Simple editorial sentence.") in {
        "editorial",
        "technical",
        "advertisement",
    }
    assert processor._classify_recovery_text_content("def hello():\n    return 1") == "code"


def test_text_integrity_scout_recovery_chunks_always_have_classification(tmp_path):
    processor = _make_batch_processor(tmp_path)

    pdf_path = tmp_path / "recovery_source.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "This technical recovery paragraph contains API schema module pipeline implementation details "
        "so it should be rescued and classified during recovery.",
    )
    doc.save(str(pdf_path))
    doc.close()

    processor._current_pdf_path = pdf_path
    processor._doc_hash = "doc123hash"

    recovered = processor._run_text_integrity_scout(
        chunks=[],
        source_file=pdf_path.name,
        variance_percent=-25.0,
    )

    recovery_text_chunks = [
        c
        for c in recovered
        if c.modality.name == "TEXT"
        and c.metadata
        and isinstance(c.metadata.extraction_method, str)
        and c.metadata.extraction_method.startswith("recovery_")
    ]

    assert recovery_text_chunks, "Expected recovery to produce at least one recovery text chunk"
    assert all(c.metadata.content_classification is not None for c in recovery_text_chunks)
