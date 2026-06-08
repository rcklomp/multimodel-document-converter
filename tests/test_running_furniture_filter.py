"""Running-header/footer/folio furniture filter (PLAN_GATE_QUALITY_V1 F1).

Page furniture (running headers, folios, mastheads) is retrieval noise that
passes the structural gates. Detection is spatial-first (bbox Y-position in the
top/bottom page margin) plus cross-page repetition or a masthead/URL pattern.
The repetition rule protects real headings; the spatial band protects content;
a page-coverage guard never orphans a page.

Fully offline/deterministic: synthetic chunks with bbox in the [0,1000] frame.
"""

from __future__ import annotations

from pathlib import Path

from mmrag_v2.batch_processor import BatchProcessor
from mmrag_v2.schema.ingestion_schema import (
    FileType,
    Modality,
    create_text_chunk,
)

BOTTOM = [40, 960, 220, 978]  # bottom margin (folio position, verified on crucible)
TOP = [80, 20, 900, 70]  # top margin (running-header position)
BODY = [80, 300, 900, 600]  # content area


def _txt(content: str, page: int, bbox):
    return create_text_chunk(
        doc_id="doc",
        content=content,
        source_file="d.pdf",
        file_type=FileType.PDF,
        page_number=page,
        bbox=bbox,
        position=0,
    )


def _bp(tmp_path: Path) -> BatchProcessor:
    return BatchProcessor(output_dir=str(tmp_path), vision_provider="none")


def _filter(tmp_path, chunks):
    return _bp(tmp_path)._filter_running_furniture(chunks)


def test_masthead_folio_dropped(tmp_path):
    # CombatAircraft folio: bottom margin + masthead -> drop (even one occurrence).
    chunks = [
        _txt("Body text on the page, real content.", 22, BODY),
        _txt("22\nAugust 2025 // www.Key.Aero", 22, BOTTOM),
    ]
    out = _filter(tmp_path, chunks)
    assert all("Key.Aero" not in (c.content or "") for c in out)
    assert any("Body text" in (c.content or "") for c in out)


def test_repeating_running_header_dropped(tmp_path):
    # A title running-header that repeats across >=3 pages (digit-normalized) -> drop.
    chunks = []
    for pg in (1, 2, 3, 4):
        chunks.append(_txt("Body content on page.", pg, BODY))
        chunks.append(_txt(f"{pg} | Chapter 2: An Array of Sequences", pg, BOTTOM))
    out = _filter(tmp_path, chunks)
    assert not any("Chapter 2" in (c.content or "") for c in out)
    assert sum(1 for c in out if "Body content" in (c.content or "")) == 4


def test_real_heading_in_content_area_kept(tmp_path):
    # A short heading-like line in the CONTENT area (not the margin) and not
    # repeating must NOT be dropped - it is real content, not furniture.
    chunks = [
        _txt("Introduction", 1, BODY),
        _txt("Body paragraph under the heading.", 1, BODY),
    ]
    out = _filter(tmp_path, chunks)
    assert any(c.content == "Introduction" for c in out)


def test_non_repeating_margin_line_without_masthead_kept(tmp_path):
    # A single short margin line that neither repeats nor has a masthead is kept
    # (conservative: we do not drop on spatial position alone).
    chunks = [
        _txt("A one-off marginal note.", 1, TOP),
        _txt("Body.", 1, BODY),
    ]
    out = _filter(tmp_path, chunks)
    assert any("one-off marginal" in (c.content or "") for c in out)


def test_page_coverage_guard_keeps_furniture_only_page(tmp_path):
    # If a folio is the ONLY chunk on its page, keep it (do not orphan the page).
    chunks = [
        _txt("Body on page 1.", 1, BODY),
        _txt("www.Key.Aero // August 2025\n9", 9, BOTTOM),  # page 9: folio only
    ]
    out = _filter(tmp_path, chunks)
    assert any(c.metadata.page_number == 9 for c in out)
