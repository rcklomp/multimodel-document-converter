"""V3 deterministic-bbox cropping contract (Charter Blocker B / B1, 2026-06-03).

Blocker B: VLM bbox crop drift 40-50% - hallucinated coordinates crop garbage
(whitespace) on forms/scans/tables. B1's structural fix: when the page yields a
detectable object (embedded image / found table), crop from that GEOMETRIC bbox
and use the VLM only for the description. Trust VLM coordinates only when no
deterministic source exists.

These tests pin the rescue: a chunk whose VLM bbox points at blank whitespace
still produces a content-bearing crop because B1 cropped the real object, not
the hallucinated coordinates. Fully offline/deterministic: synthetic fitz PDFs
with a real embedded raster, synthetic UIR chunks, no VLM.
"""

from __future__ import annotations

import io
from pathlib import Path

import fitz

from mmrag_v2.schema.ingestion_schema import Modality
from mmrag_v2.universal.asset_materializer import materialize_visual_assets
from mmrag_v2.universal.intermediate import (
    ConfidenceBreakdown,
    CoordinateFrame,
    Locator,
    LocatorType,
    UIRChunk,
)


def _checker_png(size: int = 96, square: int = 8) -> bytes:
    """A deterministic high-variance checkerboard (never 'blank')."""
    from PIL import Image

    img = Image.new("L", (size, size), 255)
    px = img.load()
    for y in range(size):
        for x in range(size):
            if (x // square + y // square) % 2 == 0:
                px[x, y] = 0
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _pdf_with_image(tmp_path: Path, img_rect: fitz.Rect) -> Path:
    """A white page with one real embedded raster image at ``img_rect``."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(img_rect, stream=_checker_png())
    out = tmp_path / "img.pdf"
    doc.save(str(out))
    doc.close()
    return out


def _image_uir(bbox, *, page: int = 1) -> UIRChunk:
    return UIRChunk(
        modality=Modality.IMAGE,
        content="A figure.",
        locator=Locator(
            type=LocatorType.BBOX,
            bbox=list(bbox),
            page_number=page,
            coordinate_frame=CoordinateFrame.PDF_PAGE_PORTRAIT,
        ),
        confidence=ConfidenceBreakdown(),
        extraction_method="vlm_native",
        extraction_engine_version="qwen3-vl-8b",
    )


def test_b1_rescues_garbage_vlm_bbox_via_geometric_object(tmp_path):
    """A garbage VLM bbox over whitespace still yields a content-bearing crop."""
    img_rect = fitz.Rect(100, 100, 300, 300)  # real image, upper-left
    pdf = _pdf_with_image(tmp_path, img_rect)
    # The VLM bbox (normalized [0,1000]) points at the blank lower-right
    # quadrant - hallucinated coordinates that pre-B1 cropped pure whitespace.
    chunk = _image_uir(bbox=(720, 800, 960, 980))

    report = materialize_visual_assets([chunk], pdf, tmp_path / "assets", doc_hash="b1")

    assert report.rendered == 1
    health = report.crops[0]
    # Cropped from the detected object, not the hallucinated coordinates.
    assert health.crop_source == "geometric"
    # The rescue: the crop is the real (high-variance) image, NOT blank.
    assert health.is_low_information is False
    assert health.is_edge_clamped is False
    assert health.std_luminance > 10.0
    assert (tmp_path / chunk.asset_ref).exists()


def test_b1_uses_vlm_bbox_when_no_geometric_object(tmp_path):
    """With no detectable object, the VLM bbox is still used (back-compat)."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.draw_rect(fitz.Rect(90, 100, 320, 280), color=(0, 0, 0), fill=(0, 0, 0))
    pdf = tmp_path / "noimg.pdf"
    doc.save(str(pdf))
    doc.close()

    # Bbox over the drawn block (a vector drawing is NOT a get_image_info object).
    chunk = _image_uir(bbox=(120, 110, 560, 380))
    report = materialize_visual_assets([chunk], pdf, tmp_path / "assets", doc_hash="novlm")

    assert report.crops[0].crop_source == "vlm"


def test_b1_distributes_two_chunks_across_two_objects(tmp_path):
    """Two image chunks on a 2-image page consume distinct objects, not one twice."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(fitz.Rect(60, 60, 240, 240), stream=_checker_png())
    page.insert_image(fitz.Rect(360, 400, 540, 580), stream=_checker_png())
    pdf = tmp_path / "two.pdf"
    doc.save(str(pdf))
    doc.close()

    # Both VLM bboxes garbage (blank strip across the middle); B1 still assigns
    # each chunk a real object and neither crop is blank.
    chunks = [_image_uir(bbox=(400, 480, 600, 520)), _image_uir(bbox=(400, 490, 600, 530))]
    report = materialize_visual_assets(chunks, pdf, tmp_path / "assets", doc_hash="two")

    assert report.rendered == 2
    assert all(c.crop_source == "geometric" for c in report.crops)
    assert all(c.is_low_information is False for c in report.crops)
    # Distinct assets on disk.
    assert len({c.asset_ref for c in report.crops}) == 2
