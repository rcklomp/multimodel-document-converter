"""Empty-content TABLE handling with a page-coverage guard (review fix #1/#3).

An empty-content TABLE (no markdown) is a corrupt placeholder that fails the
table-format gate, so it is dropped - EXCEPT when it is the only surviving
content on its page, where dropping would manufacture MISSING_PAGES. There it is
promoted to IMAGE (it carries the rendered table crop) so the page stays covered
without a corrupt empty table. Composes safely with the tiny-icon filter.

Fully offline/deterministic: synthetic chunks, no extraction.
"""

from __future__ import annotations

from pathlib import Path

from mmrag_v2.batch_processor import BatchProcessor
from mmrag_v2.schema.ingestion_schema import (
    FileType,
    Modality,
    create_table_chunk,
    create_text_chunk,
)


def _bp(tmp_path: Path) -> BatchProcessor:
    return BatchProcessor(output_dir=str(tmp_path), vision_provider="none")


def _empty_table(page: int, *, asset: bool = True):
    return create_table_chunk(
        doc_id="doc",
        content="",  # empty -> no markdown
        source_file="d.pdf",
        file_type=FileType.PDF,
        page_number=page,
        bbox=[100, 100, 900, 900],
        asset_path=f"assets/doc_{page:04d}_table_000.png" if asset else None,
        position=0,
    )


def _text(page: int):
    return create_text_chunk(
        doc_id="doc",
        content="Real body text on the page.",
        source_file="d.pdf",
        file_type=FileType.PDF,
        page_number=page,
        position=0,
    )


def test_empty_table_dropped_when_page_has_other_content(tmp_path):
    out = _bp(tmp_path)._promote_or_drop_empty_tables([_text(1), _empty_table(1)])
    assert [c.modality for c in out] == [Modality.TEXT]  # table dropped


def test_empty_table_only_on_page_is_promoted_not_dropped(tmp_path):
    # Page 5 holds ONLY the empty table -> promote to IMAGE (keep coverage).
    out = _bp(tmp_path)._promote_or_drop_empty_tables([_text(1), _empty_table(5)])
    pages = {c.metadata.page_number: c.modality for c in out}
    assert pages == {1: Modality.TEXT, 5: Modality.IMAGE}
    assert len(out) == 2  # nothing orphaned


def test_nonempty_table_untouched(tmp_path):
    t = create_table_chunk(
        doc_id="doc",
        content="| a | b |\n| --- | --- |\n| 1 | 2 |",
        source_file="d.pdf",
        file_type=FileType.PDF,
        page_number=1,
        bbox=[100, 100, 900, 900],
        asset_path="assets/t.png",
        position=0,
    )
    out = _bp(tmp_path)._promote_or_drop_empty_tables([t])
    assert len(out) == 1 and out[0].modality == Modality.TABLE
