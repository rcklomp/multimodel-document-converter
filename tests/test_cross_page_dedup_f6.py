"""Cross-page duplicate dedup (PLAN_GATE_QUALITY_V1 F6).

Captions/headers F1 misses in the content area get repeated verbatim across
pages (AIOS "(a) Normalized throughput..." x5); VLM loops do too. F6 keeps the
first occurrence and drops later exact duplicates - TEXT ONLY (a multi-page TABLE
legitimately repeats its column-header row, the review trap), >= 3 distinct
pages, >= 20 chars, page-coverage guarded.

Offline/deterministic: synthetic chunks.
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

CAP = "(a) Normalized throughput. Higher is better."  # 44 chars


def _txt(content: str, page: int):
    return create_text_chunk(
        doc_id="d", content=content, source_file="d.pdf", file_type=FileType.PDF,
        page_number=page, bbox=[80, 300, 900, 360], position=0,
    )


def _tbl(content: str, page: int):
    return create_table_chunk(
        doc_id="d", content=content, source_file="d.pdf", file_type=FileType.PDF,
        page_number=page, bbox=[80, 300, 900, 600], asset_path=f"a/t{page}.png", position=0,
    )


def _bp(tmp_path: Path) -> BatchProcessor:
    return BatchProcessor(output_dir=str(tmp_path), vision_provider="none")


def _dedup(tmp_path, chunks):
    return _bp(tmp_path)._dedup_cross_page_repeats(chunks)


def test_cross_page_caption_deduped_keep_first(tmp_path):
    chunks = [_txt("Body on page %d." % p, p) for p in (9, 32, 33, 34, 35)]
    chunks += [_txt(CAP, p) for p in (9, 32, 33, 34, 35)]  # 5 copies across pages
    out = _dedup(tmp_path, chunks)
    assert sum(1 for c in out if c.content == CAP) == 1  # only the first survives


def test_two_page_repeat_not_deduped(tmp_path):
    # Only 2 pages -> below the >=3 threshold -> not a confident duplicate.
    chunks = [_txt("Body.", 1), _txt(CAP, 1), _txt(CAP, 2)]
    out = _dedup(tmp_path, chunks)
    assert sum(1 for c in out if c.content == CAP) == 2


def test_table_headers_repeating_across_pages_are_kept(tmp_path):
    # The review trap: a multi-page table repeats its column-header row on every
    # page. F6 must NOT touch TABLE chunks.
    header = "| Qtr | Revenue | Margin |\n| --- | --- | --- |\n| Q1 | 10 | 2 |"
    chunks = [_tbl(header, p) for p in (1, 2, 3)]
    out = _dedup(tmp_path, chunks)
    assert sum(1 for c in out if c.modality == Modality.TABLE) == 3


def test_page_coverage_guard_keeps_only_chunk_on_page(tmp_path):
    # Page 35 holds ONLY a duplicate caption -> keep it (do not orphan the page).
    chunks = [_txt("Body on page 9.", 9), _txt(CAP, 9), _txt(CAP, 32), _txt(CAP, 35)]
    # page 32 also caption-only; both 32 and 35 are caption-only here.
    out = _dedup(tmp_path, chunks)
    pages_present = {c.metadata.page_number for c in out}
    assert {9, 32, 35} <= pages_present  # no page orphaned


def test_short_recurring_label_not_deduped(tmp_path):
    chunks = [_txt("Note:", p) for p in (1, 2, 3, 4)]  # < 20 chars
    out = _dedup(tmp_path, chunks)
    assert sum(1 for c in out if c.content == "Note:") == 4
