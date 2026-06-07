"""Tiny icon-class image filter (Cluster D follow-up, 2026-06-07).

The V3 path emits every detected image region; the geometric crop can isolate a
tiny embedded raster (page icon, bullet glyph, small logo) that is not
retrievable content and only adds IMAGE_NO_VLM/ASSET_TINY noise. The filter
drops an IMAGE chunk only when its RENDERED asset is small in BOTH dimensions
AND has a tiny file (triple-AND) - small real figures survive via one larger
dimension or a detailed file. A page-coverage guard never orphans a page.

Fully offline/deterministic: synthetic PNG assets + synthetic chunks.
"""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image

from mmrag_v2.batch_processor import BatchProcessor
from mmrag_v2.schema.ingestion_schema import (
    FileType,
    Modality,
    create_image_chunk,
    create_text_chunk,
)


def _write_png(path: Path, w: int, h: int, *, noisy: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if noisy:
        # Random bytes -> incompressible -> file comfortably exceeds 1.5 KB.
        img = Image.frombytes("RGB", (w, h), os.urandom(w * h * 3))
    else:
        img = Image.new("RGB", (w, h), (200, 200, 200))
    img.save(path)


def _img_chunk(out: Path, fname: str, page: int) -> "object":
    return create_image_chunk(
        doc_id="doc",
        content="",
        source_file="d.pdf",
        file_type=FileType.PDF,
        page_number=page,
        asset_path=f"assets/{fname}",
        bbox=[100, 100, 900, 900],  # bbox is deliberately large + ignored
        position=0,
    )


def _text_chunk(page: int) -> "object":
    return create_text_chunk(
        doc_id="doc",
        content="Body text on the page so the page has non-image content.",
        source_file="d.pdf",
        file_type=FileType.PDF,
        page_number=page,
        position=0,
    )


def _bp(tmp_path: Path) -> BatchProcessor:
    return BatchProcessor(output_dir=str(tmp_path), vision_provider="none")


def test_icon_class_image_on_content_page_is_dropped(tmp_path):
    _write_png(tmp_path / "assets/icon.png", 30, 30, noisy=False)  # 30x30, tiny file
    chunks = [_text_chunk(1), _img_chunk(tmp_path, "icon.png", 1)]
    out = _bp(tmp_path)._filter_tiny_icon_images(chunks)
    assert all(c.modality != Modality.IMAGE for c in out)
    assert not (tmp_path / "assets/icon.png").exists()  # asset removed too


def test_real_figure_survives_via_large_dimension(tmp_path):
    _write_png(tmp_path / "assets/fig.png", 200, 60, noisy=False)  # width 200 >= 96
    chunks = [_text_chunk(1), _img_chunk(tmp_path, "fig.png", 1)]
    out = _bp(tmp_path)._filter_tiny_icon_images(chunks)
    assert any(c.modality == Modality.IMAGE for c in out)


def test_small_but_detailed_figure_survives_via_file_size(tmp_path):
    # Both dims < 96 but a detailed (>=1.5KB) file => real content, kept.
    _write_png(tmp_path / "assets/small.png", 67, 68, noisy=True)
    assert (tmp_path / "assets/small.png").stat().st_size >= 1500
    chunks = [_text_chunk(1), _img_chunk(tmp_path, "small.png", 1)]
    out = _bp(tmp_path)._filter_tiny_icon_images(chunks)
    assert any(c.modality == Modality.IMAGE for c in out)


def test_page_coverage_guard_keeps_icon_only_page(tmp_path):
    # An icon-class image that is the ONLY content on its page is kept so the
    # page is not orphaned into MISSING_PAGES.
    _write_png(tmp_path / "assets/lonely.png", 30, 30, noisy=False)
    chunks = [_text_chunk(1), _img_chunk(tmp_path, "lonely.png", 7)]  # page 7 = icon only
    out = _bp(tmp_path)._filter_tiny_icon_images(chunks)
    assert any(c.modality == Modality.IMAGE for c in out)
    assert (tmp_path / "assets/lonely.png").exists()
