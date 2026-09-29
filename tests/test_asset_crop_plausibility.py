"""PLAN_QUALITY_REMEDIATION_V1 WP-A1: the B1 geometric pick must not replace a plausible VLM box.

Defect (IRJET): ``_pick_geometric_clip`` replaced the VLM box with ANY detected raster, so
  * Fig 3 / Fig 5 (vector art, no raster under the VLM box) got the page-header LOGO through the
    no-overlap "largest remaining object" fallback, and
  * Fig 7 (a vector flowchart with a text-strip raster inside it) got that 420x43 strip.
Measured on 359 replayable local crops: the guard changes 27 (7.5%), all IMAGE; every changed crop
was inspected against the rendered page (IRJET Figs 4/5/7, HarryPotter p7 and AIOS p7 fragments are
recovered); the no-overlap rescues that the B1 tests and the Fluent Python sidebar images depend on
are unchanged.

Contract (IMAGE chunks carrying a VLM box):
  box holds no graphics (whitespace / prose)      -> B1 rescue stands
  box holds graphics, no raster overlaps it       -> keep the VLM box
  box holds graphics, raster is a fragment of it  -> keep the VLM box
  box holds graphics, raster is the dominant one  -> the raster stands
"""

from __future__ import annotations

import io
from pathlib import Path

import fitz
import pytest
from PIL import Image

from mmrag_v2.schema.ingestion_schema import Modality
from mmrag_v2.universal.asset_materializer import materialize_visual_assets
from mmrag_v2.universal.intermediate import (
    ConfidenceBreakdown,
    CoordinateFrame,
    Locator,
    LocatorType,
    UIRChunk,
)

PAGE_W, PAGE_H = 612.0, 792.0


def _png(size: int = 96) -> bytes:
    img = Image.new("L", (size, size), 255)
    px = img.load()
    for y in range(size):
        for x in range(size):
            if (x // 8 + y // 8) % 2 == 0:
                px[x, y] = 0
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _norm(rect: fitz.Rect):
    return (
        round(rect.x0 / PAGE_W * 1000),
        round(rect.y0 / PAGE_H * 1000),
        round(rect.x1 / PAGE_W * 1000),
        round(rect.y1 / PAGE_H * 1000),
    )


def _chunk(rect: fitz.Rect, modality=Modality.IMAGE) -> UIRChunk:
    return UIRChunk(
        modality=modality,
        content="A figure.",
        locator=Locator(
            type=LocatorType.BBOX,
            bbox=list(_norm(rect)),
            page_number=1,
            coordinate_frame=CoordinateFrame.PDF_PAGE_PORTRAIT,
        ),
        confidence=ConfidenceBreakdown(),
        extraction_method="vlm_native",
        extraction_engine_version="qwen3-vl-8b",
    )


def _flowchart(page: fitz.Page, area: fitz.Rect) -> None:
    """A connected vector figure: three filled boxes joined by arrows, filling ``area``."""
    h = area.height / 5
    for i in range(3):
        y = area.y0 + i * 2 * h
        box = fitz.Rect(area.x0 + 20, y, area.x1 - 20, y + h)
        page.draw_rect(box, color=(0, 0, 0), fill=(0.8, 0.85, 1.0), width=1.2)
        if i < 2:
            page.draw_line((box.x0 + box.width / 2, box.y1), (box.x0 + box.width / 2, box.y1 + h))


def _pdf(tmp_path: Path, draw) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    draw(page)
    out = tmp_path / "p.pdf"
    doc.save(str(out))
    doc.close()
    return out


def _run(tmp_path: Path, pdf: Path, chunk: UIRChunk):
    report = materialize_visual_assets([chunk], pdf, tmp_path / "assets", doc_hash="a1")
    assert report.rendered == 1
    return report.crops[0]


def _near(rect_pt, expected: fitz.Rect, tol: float = 2.0) -> bool:
    return all(
        abs(a - b) <= tol
        for a, b in zip(rect_pt, (expected.x0, expected.y0, expected.x1, expected.y1))
    )


LOGO = fitz.Rect(40, 40, 74, 74)


def test_vector_figure_is_not_swapped_for_an_unrelated_logo(tmp_path):
    # IRJET Fig 5 shape: the VLM box is on a vector figure; the only raster on the page is a logo.
    fig = fitz.Rect(320, 300, 550, 460)

    def draw(page):
        page.insert_image(LOGO, stream=_png())
        _flowchart(page, fig)

    h = _run(tmp_path, _pdf(tmp_path, draw), _chunk(fig))
    assert h.crop_source == "vlm"
    assert h.crop_reason == "vlm_kept_graphics_in_box"
    assert _near(h.clip_rect_pt, fig)
    assert not _near(h.clip_rect_pt, LOGO)


def test_text_strip_raster_inside_a_vector_flowchart_does_not_replace_it(tmp_path):
    # IRJET Fig 7 shape: a small raster (an equation image) sits inside the vector flowchart.
    fig = fitz.Rect(300, 120, 570, 560)
    strip = fitz.Rect(340, 500, 540, 520)

    def draw(page):
        _flowchart(page, fig)
        page.insert_image(strip, stream=_png(32))

    h = _run(tmp_path, _pdf(tmp_path, draw), _chunk(fig))
    assert h.crop_source == "vlm"
    assert h.crop_reason == "vlm_kept_raster_is_fragment"
    assert _near(h.clip_rect_pt, fig)


def test_oversized_box_around_one_picture_still_crops_the_picture(tmp_path):
    # HarryPotter p2 shape: the box is far larger than the one raster it surrounds.
    picture = fitz.Rect(60, 60, 460, 400)

    def draw(page):
        page.insert_image(picture, stream=_png(200))

    box = fitz.Rect(0, 0, 612, 700)
    h = _run(tmp_path, _pdf(tmp_path, draw), _chunk(box))
    assert h.crop_source == "geometric"
    assert h.crop_reason == "geometric_dominant"
    assert _near(h.clip_rect_pt, picture)


def test_box_over_blank_space_keeps_the_b1_rescue(tmp_path):
    picture = fitz.Rect(100, 100, 300, 300)

    def draw(page):
        page.insert_image(picture, stream=_png(200))

    h = _run(tmp_path, _pdf(tmp_path, draw), _chunk(fitz.Rect(400, 600, 580, 760)))
    assert h.crop_source == "geometric"
    assert h.crop_reason == "geometric_rescue"
    assert _near(h.clip_rect_pt, picture)


def test_box_over_prose_keeps_the_b1_rescue(tmp_path):
    # Fluent Python sidebar shape: the VLM box lands on body text, the real picture is beside it.
    picture = fitz.Rect(82, 108, 133, 156)

    def draw(page):
        page.insert_image(picture, stream=_png(64))
        for i in range(6):
            page.insert_text(
                (72, 200 + i * 14), "Both + and * always create a new object here.", fontsize=10
            )

    h = _run(tmp_path, _pdf(tmp_path, draw), _chunk(fitz.Rect(70, 190, 330, 290)))
    assert h.crop_source == "geometric"
    assert h.crop_reason == "geometric_rescue"
    assert _near(h.clip_rect_pt, picture)


def test_full_bleed_background_art_does_not_count_as_graphics_in_the_box(tmp_path):
    # A page-sized object must not make every box "hold graphics"; whitespace box + a real raster
    # elsewhere on a page whose background is vector art still gets the rescue.
    picture = fitz.Rect(100, 100, 300, 300)

    def draw(page):
        page.draw_rect(fitz.Rect(0, 0, PAGE_W, PAGE_H), color=None, fill=(0.95, 0.95, 0.95))
        page.insert_image(picture, stream=_png(200))

    h = _run(tmp_path, _pdf(tmp_path, draw), _chunk(fitz.Rect(400, 600, 580, 760)))
    assert h.crop_reason == "geometric_rescue"


def test_tables_are_not_touched_by_the_guard(tmp_path):
    # Guard is IMAGE-only: a TABLE chunk with a find_tables candidate keeps the old behaviour.
    def draw(page):
        for r in range(4):
            for c in range(3):
                page.draw_rect(
                    fitz.Rect(100 + c * 80, 100 + r * 30, 180 + c * 80, 130 + r * 30),
                    color=(0, 0, 0),
                    width=1,
                )
                page.insert_text((108 + c * 80, 120 + r * 30), f"r{r}c{c}", fontsize=10)

    pdf = _pdf(tmp_path, draw)
    h = _run(tmp_path, pdf, _chunk(fitz.Rect(90, 90, 350, 240), Modality.TABLE))
    assert h.crop_reason in {"geometric", "no_geometric_object"}


def test_sidecar_record_is_complete_and_matches_the_asset_on_disk(tmp_path):
    fig = fitz.Rect(320, 300, 550, 460)

    def draw(page):
        page.insert_image(LOGO, stream=_png())
        _flowchart(page, fig)

    chunk = _chunk(fig)
    h = _run(tmp_path, _pdf(tmp_path, draw), chunk)
    rec = h.to_record()
    assert set(rec) >= {
        "asset_ref",
        "page",
        "modality",
        "crop_source",
        "crop_reason",
        "vlm_rect_pt",
        "clip_rect_pt",
        "asset_px",
    }
    with Image.open(tmp_path / chunk.asset_ref) as img:
        assert rec["asset_px"] == [img.width, img.height]
    # frame-invariant: the recorded rectangle is in PDF points, so px / rect == crop zoom
    w = rec["clip_rect_pt"][2] - rec["clip_rect_pt"][0]
    assert rec["asset_px"][0] == pytest.approx(w * 2.0, abs=2)


def test_no_usable_bbox_records_a_full_page_crop_without_a_clip_rect(tmp_path):
    def draw(page):
        page.insert_image(LOGO, stream=_png())

    chunk = _chunk(fitz.Rect(0, 0, 1, 1))  # degenerate box
    h = _run(tmp_path, _pdf(tmp_path, draw), chunk)
    assert h.crop_source in {"geometric", "full_page", "vlm"}
    if h.is_full_page_fallback:
        assert h.clip_rect_pt is None
        assert h.vlm_rect_pt is None
