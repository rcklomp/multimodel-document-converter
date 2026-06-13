"""Thin-strip image filter (WS2b, PLAN_FIDELITY_ORACLE_FIRST_V1 Section 3').

MinerU/hybrid sometimes emits a table-header/row band (e.g. 720x28, aspect 26) as
an IMAGE region; the table content is already a TABLE chunk, so the strip is a
redundant crop that fails the strict IMAGE gate (`qa_conversion_audit.py` flags
rendered aspect > 25 as `thin_strips`, a hard FAIL - Adedeji). `_filter_thin_strip_images`
culls EXACTLY what the gate flags, behind a page-coverage guard.

Fully offline/deterministic: synthetic PNG assets + synthetic chunks. The predicate
matches the gate (aspect > 25) so a fix here clears the gate there.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from mmrag_v2.batch_processor import BatchProcessor
from mmrag_v2.schema.ingestion_schema import (
    FileType,
    Modality,
    create_image_chunk,
    create_text_chunk,
)


def _write_png(path: Path, w: int, h: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), (200, 200, 200)).save(path)


def _img_chunk(fname: str, page: int):
    return create_image_chunk(
        doc_id="doc",
        content="",
        source_file="d.pdf",
        file_type=FileType.PDF,
        page_number=page,
        asset_path=f"assets/{fname}",
        bbox=[100, 100, 900, 900],  # bbox deliberately large + ignored
        position=0,
    )


def _text_chunk(page: int):
    return create_text_chunk(
        doc_id="doc",
        content="Body text so the page has non-image content.",
        source_file="d.pdf",
        file_type=FileType.PDF,
        page_number=page,
        position=0,
    )


def _bp(tmp_path: Path) -> BatchProcessor:
    return BatchProcessor(output_dir=str(tmp_path), vision_provider="none")


def test_thin_strip_on_content_page_is_dropped(tmp_path):
    # The Adedeji case: 720x28 table-header band, aspect ~26 > 25.
    _write_png(tmp_path / "assets/row_table_000.png", 720, 28)
    chunks = [_text_chunk(1), _img_chunk("row_table_000.png", 1)]
    out = _bp(tmp_path)._filter_thin_strip_images(chunks)
    assert all(c.modality != Modality.IMAGE for c in out)
    assert not (tmp_path / "assets/row_table_000.png").exists()  # asset removed too


def test_vertical_thin_strip_is_dropped(tmp_path):
    # Tall narrow strip (28x800, aspect ~29) is also a thin strip.
    _write_png(tmp_path / "assets/vstrip.png", 28, 800)
    chunks = [_text_chunk(1), _img_chunk("vstrip.png", 1)]
    out = _bp(tmp_path)._filter_thin_strip_images(chunks)
    assert all(c.modality != Modality.IMAGE for c in out)


def test_normal_figure_survives(tmp_path):
    _write_png(tmp_path / "assets/fig.png", 400, 300)  # aspect 1.3
    chunks = [_text_chunk(1), _img_chunk("fig.png", 1)]
    out = _bp(tmp_path)._filter_thin_strip_images(chunks)
    assert any(c.modality == Modality.IMAGE for c in out)
    assert (tmp_path / "assets/fig.png").exists()


def test_wide_but_not_strip_figure_survives(tmp_path):
    # A wide banner (800x200, aspect 4) is NOT a thin strip; kept.
    _write_png(tmp_path / "assets/banner.png", 800, 200)
    chunks = [_text_chunk(1), _img_chunk("banner.png", 1)]
    out = _bp(tmp_path)._filter_thin_strip_images(chunks)
    assert any(c.modality == Modality.IMAGE for c in out)


def test_boundary_aspect_25_is_kept(tmp_path):
    # Gate predicate is aspect > 25 (strict). 500x20 = aspect 25.0 -> kept.
    _write_png(tmp_path / "assets/edge.png", 500, 20)
    chunks = [_text_chunk(1), _img_chunk("edge.png", 1)]
    out = _bp(tmp_path)._filter_thin_strip_images(chunks)
    assert any(c.modality == Modality.IMAGE for c in out)


def test_page_coverage_guard_keeps_strip_only_page(tmp_path):
    # A thin strip that is the ONLY content on its page is kept (no MISSING_PAGES).
    _write_png(tmp_path / "assets/lonely_strip.png", 720, 28)
    chunks = [_text_chunk(1), _img_chunk("lonely_strip.png", 7)]  # page 7 = strip only
    out = _bp(tmp_path)._filter_thin_strip_images(chunks)
    assert any(c.modality == Modality.IMAGE for c in out)
    assert (tmp_path / "assets/lonely_strip.png").exists()
