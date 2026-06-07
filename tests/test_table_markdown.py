"""Engine-agnostic Markdown-table separator repair (Cluster C, 2026-06-06).

MinerU and Qwen both occasionally emit a pipe-delimited table WITHOUT the
Markdown ``|---|`` separator row (FluentPython p17), which fails the
table-format gate while sibling tables that include it pass. The repair lives
at the engine-agnostic chunker chokepoint so every engine is covered.

Fully offline/deterministic: synthetic UIR documents, no engine, no endpoint.
"""

from __future__ import annotations

from mmrag_v2.chunking.uir_chunker import chunk_universal_document
from mmrag_v2.universal.intermediate import (
    ElementType,
    Modality,
    PageClassification,
    UniversalDocument,
    UniversalPage,
    create_element,
)
from mmrag_v2.universal.table_markdown import (
    ensure_table_separator,
    normalize_pipe_table,
)


def test_pipe_table_without_separator_gets_one():
    """A separator-less pipe table (FluentPython p17) gains a |---| row."""
    content = (
        "| s.__add__(s2) | list | array | s + s2 |\n"
        "| s.append(e) | list | array | Append one element |\n"
        "| s.clear() | list | array | Delete all items |"
    )
    md = normalize_pipe_table(content)
    lines = md.splitlines()
    assert lines[0] == "| s.__add__(s2) | list | array | s + s2 |"
    assert lines[1] == "| --- | --- | --- | --- |"  # the separator the gate requires
    assert lines[2] == "| s.append(e) | list | array | Append one element |"


def test_pipe_table_with_separator_left_untouched():
    """An already-valid pipe table is not re-normalized (no double separator)."""
    assert normalize_pipe_table("| a | b |\n|---|---|\n| 1 | 2 |") is None


def test_pipe_table_normalizer_ignores_non_pipe_content():
    """Plain prose / HTML in a table element is never re-shaped (no content loss)."""
    assert normalize_pipe_table("just prose, no pipes here") is None
    assert normalize_pipe_table("<table><tr><td>x</td></tr></table>") is None


def test_ragged_pipe_rows_padded_rectangular():
    grid = normalize_pipe_table("| a | b | c |\n| x |")
    assert grid.splitlines()[-1] == "| x |  |  |"


def test_ensure_table_separator_is_noop_on_valid_and_empty():
    valid = "| a | b |\n| --- | --- |\n| 1 | 2 |"
    assert ensure_table_separator(valid) == valid
    assert ensure_table_separator("") == ""
    assert ensure_table_separator("prose") == "prose"


def _table_doc(content: str) -> UniversalDocument:
    page = UniversalPage(
        page_number=1,
        elements=[
            create_element(ElementType.TABLE, content, bbox=[60, 60, 900, 360], element_index=0)
        ],
        classification=PageClassification.DIGITAL,
        dimensions=(612, 792),
    )
    return UniversalDocument(doc_id="tbl", source_file="t.pdf", file_type="pdf", pages=[page])


def test_chunker_repairs_separatorless_table_end_to_end():
    """A TABLE element with separator-less pipe content is fixed in chunking -
    covers BOTH engines, since every table passes through this chokepoint."""
    doc = _table_doc("| x | y |\n| 1 | 2 |\n| 3 | 4 |")
    tables = [c for c in chunk_universal_document(doc) if c.modality is Modality.TABLE]
    assert len(tables) == 1
    assert "| --- | --- |" in tables[0].content


def test_chunker_leaves_valid_table_untouched_end_to_end():
    valid = "| x | y |\n| --- | --- |\n| 1 | 2 |"
    doc = _table_doc(valid)
    tables = [c for c in chunk_universal_document(doc) if c.modality is Modality.TABLE]
    assert tables[0].content == valid
