"""PLAN_F1 WP-A: chunker-level code-block contiguity.

The chunker emits one chunk per code Element in document order. When the
extractor splits a single logical code block into several code Elements with a
figure/table/prose Element interleaved between them, the fragments each fail
``ast.parse`` (the F1 oracle's dominant residual: 15/26 Chaubal fails were a
code block split across an interleaved non-code chunk). ``_coalesce_code_blocks``
merges code Elements that form ONE logical block into a single chunk and defers
the interleaved non-code Element to AFTER the block.

Contracts pinned here:
  - split-then-interleaved code blocks merge into one parseable CODE chunk;
  - the interleaved figure/table is emitted as its own chunk AFTER the block;
  - two complete, standalone code blocks are NOT merged (no over-coalescing);
  - merged bbox is an integer [0,1000] union (AGENT-SPATIAL / COORD_SCALE);
  - a single-code-element page is unchanged (no-op).

Fully offline/deterministic: no VLM, no network.
"""

from __future__ import annotations

import ast

import fitz

from mmrag_v2.chunking.uir_chunker import (
    _code_block_continues,
    _coalesce_code_blocks,
    _strip_code_fence,
    chunk_universal_document,
)
from mmrag_v2.schema.ingestion_schema import Modality
from mmrag_v2.universal.intermediate import DocumentMetadata, create_document
from mmrag_v3.engines.vlm_native import VlmNativeEngine

W, H = 1000, 1000


def _page_from_vlm(elements):
    payload = {
        "page_number": 1,
        "width": W,
        "height": H,
        "classification": "digital",
        "elements": elements,
    }
    return VlmNativeEngine._page_from_payload(
        payload, fallback_page_number=1, pixel_width=W, pixel_height=H
    )


def _doc(tmp_path, page):
    pdf = tmp_path / "t.pdf"
    d = fitz.open()
    d.new_page(width=612, height=792)
    d.save(str(pdf))
    d.close()
    return create_document(
        file_path=pdf,
        file_type="pdf",
        pages=[page],
        metadata=DocumentMetadata(
            page_count=1, file_size_bytes=1, has_text_layer=True, has_images=False
        ),
    )


def _el(type_, content, bbox=(10, 10, 500, 200)):
    return {"type": type_, "content": content, "bbox": list(bbox), "confidence": 0.95}


def _defence(content: str) -> str:
    return _strip_code_fence(content)


# --- the continuation predicate ---------------------------------------------


def test_continuation_predicate_prev_open_suite():
    # prev ends on a suite header (':'); next is the indented body.
    assert _code_block_continues("def f(x):\n    for y in x:", "        g(y)") is True


def test_continuation_predicate_next_midbody_indent():
    # prev is a fragment, next starts indented (unexpected-indent fragment).
    assert _code_block_continues("class A:\n    x = 1", "    def m(self):\n        pass") is True


def test_continuation_predicate_unterminated_docstring():
    assert (
        _code_block_continues('def f():\n    """start of doc', 'more doc\n    """\n    return 1')
        is True
    )


def test_continuation_predicate_unclosed_brackets():
    assert _code_block_continues("result = foo(\n    a,", "    b,\n)") is True


def test_two_complete_blocks_not_continued():
    # Both blocks complete; next starts a fresh statement at column 0.
    assert _code_block_continues("def a():\n    return 1", "def b():\n    return 2") is False


# --- coalescing across an interleaved non-code element -----------------------


def test_coalesce_code_split_by_table_into_one_parseable_chunk(tmp_path):
    seg1 = "def process(items):\n    for x in items:"  # ends open (':')
    seg2 = "        handle(x)\n    return len(items)"  # indented continuation
    page = _page_from_vlm(
        [
            _el("code", seg1, bbox=(10, 10, 500, 100)),
            _el("table", "| a | b |\n|---|---|\n| 1 | 2 |", bbox=(10, 110, 500, 200)),
            _el("code", seg2, bbox=(10, 210, 500, 300)),
        ]
    )
    chunks = chunk_universal_document(_doc(tmp_path, page))

    code_chunks = [c for c in chunks if c.modality == Modality.CODE]
    table_chunks = [c for c in chunks if c.modality == Modality.TABLE]
    assert len(code_chunks) == 1, "the two code fragments must merge into ONE chunk"
    assert len(table_chunks) == 1

    # The merged body parses; neither fragment parses alone.
    merged = _defence(code_chunks[0].content)
    ast.parse(merged)
    import pytest

    with pytest.raises(SyntaxError):
        ast.parse(seg1)
    with pytest.raises((SyntaxError, IndentationError)):
        ast.parse(seg2)

    # The interleaved table is emitted AFTER the merged code block.
    assert code_chunks[0].reading_order < table_chunks[0].reading_order


def test_coalesce_code_split_by_prose(tmp_path):
    seg1 = "def f(x):\n    for y in x:"
    seg2 = "        z = y + 1\n        print(z)"
    page = _page_from_vlm(
        [
            _el("code", seg1, bbox=(10, 10, 500, 100)),
            _el(
                "text",
                "This loop processes each item in turn and prints it.",
                bbox=(10, 110, 500, 160),
            ),
            _el("code", seg2, bbox=(10, 170, 500, 260)),
        ]
    )
    chunks = chunk_universal_document(_doc(tmp_path, page))
    code_chunks = [c for c in chunks if c.modality == Modality.CODE]
    assert len(code_chunks) == 1
    ast.parse(_defence(code_chunks[0].content))


def test_adjacent_split_code_merges(tmp_path):
    # No interleaving: two adjacent code Elements forming one block still merge.
    seg1 = "for i in range(10):"
    seg2 = "    total += i\n    log(total)"
    page = _page_from_vlm(
        [
            _el("code", seg1, bbox=(10, 10, 500, 60)),
            _el("code", seg2, bbox=(10, 70, 500, 160)),
        ]
    )
    chunks = chunk_universal_document(_doc(tmp_path, page))
    code_chunks = [c for c in chunks if c.modality == Modality.CODE]
    assert len(code_chunks) == 1
    body = _defence(code_chunks[0].content)
    # parseable inside a function wrapper (top-level augmented assign needs a name)
    ast.parse("total = 0\n" + body)


def test_two_standalone_blocks_not_coalesced(tmp_path):
    page = _page_from_vlm(
        [
            _el("code", "def a():\n    return 1", bbox=(10, 10, 500, 100)),
            _el(
                "text",
                "Some explanation between two unrelated functions.",
                bbox=(10, 110, 500, 160),
            ),
            _el("code", "def b():\n    return 2", bbox=(10, 170, 500, 260)),
        ]
    )
    chunks = chunk_universal_document(_doc(tmp_path, page))
    code_chunks = [c for c in chunks if c.modality == Modality.CODE]
    assert len(code_chunks) == 2, "standalone blocks must NOT be merged"
    # Document order preserved: prose stays between the two code chunks.
    text_chunks = [c for c in chunks if c.modality == Modality.TEXT]
    assert text_chunks
    assert (
        code_chunks[0].reading_order < text_chunks[0].reading_order < code_chunks[1].reading_order
    )


def test_single_code_element_is_noop(tmp_path):
    page = _page_from_vlm([_el("code", "x = 1\ny = 2\nprint(x + y)")])
    elems = _coalesce_code_blocks(page.elements)
    assert len(elems) == 1
    assert "code_block_coalesced" not in (elems[0].metadata or {})


def test_merged_bbox_is_integer_union(tmp_path):
    page = _page_from_vlm(
        [
            _el("code", "with open(p) as f:", bbox=(20, 30, 400, 120)),
            _el("code", "    data = f.read()", bbox=(20, 130, 460, 220)),
        ]
    )
    merged = _coalesce_code_blocks(page.elements)
    assert len(merged) == 1
    bbox = merged[0].bbox.to_list()
    assert bbox == [min(20, 20), min(30, 130), max(400, 460), max(120, 220)]
    assert all(isinstance(v, int) and 0 <= v <= 1000 for v in bbox)


def test_strip_code_fence_idempotent_and_lang_tolerant():
    assert _strip_code_fence("```python\nx = 1\n```") == "x = 1"
    assert _strip_code_fence("```\nx = 1\n```") == "x = 1"
    # Unfenced content is returned unchanged.
    assert _strip_code_fence("x = 1\ny = 2") == "x = 1\ny = 2"
