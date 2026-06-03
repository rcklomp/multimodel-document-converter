"""V3 crop-audit re-extraction trigger (Charter Blocker B / B2, 2026-06-03).

Pre-B2, crop-audit only WARNED: a drift-flagged crop (blank whitespace from a
hallucinated VLM bbox) was still PERSISTED as the asset. B2 makes the audit a
re-extraction trigger (fail-open): a flagged VLM crop is re-rendered to the
full page before persisting, so a garbage crop is never written to disk.

The detection fingerprints are PRESERVED (the audit still records that the VLM
crop drifted, for telemetry + the doc-level gate); what changes is the bytes on
disk and a new ``reextracted`` flag. Geometric crops (B1) are trusted and never
re-triggered.

Fully offline/deterministic: synthetic fitz PDFs, synthetic UIR chunks, no VLM.
"""

from __future__ import annotations

import io
from pathlib import Path

import fitz

from mmrag_v2.schema.ingestion_schema import Modality
from mmrag_v2.universal.asset_materializer import (
    _luminance_from_png,
    materialize_visual_assets,
)
from mmrag_v2.universal.intermediate import (
    ConfidenceBreakdown,
    CoordinateFrame,
    Locator,
    LocatorType,
    UIRChunk,
)

# Blank quadrant (hallucinated coords landing on whitespace) and the page's
# content block, in the [0,1000] frame.
BLANK_BBOX = (550, 600, 950, 900)
CONTENT_BBOX = (120, 110, 560, 380)


def _pdf_with_block(tmp_path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.draw_rect(fitz.Rect(90, 100, 320, 280), color=(0, 0, 0), fill=(0, 0, 0))
    out = tmp_path / "block.pdf"
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


def _disk_std(path: Path) -> float:
    return _luminance_from_png(path.read_bytes())[1]


def test_blank_vlm_crop_is_reextracted_not_persisted(tmp_path):
    """A blank VLM crop is detected AND the persisted file becomes the full page."""
    pdf = _pdf_with_block(tmp_path)
    chunk = _image_uir(bbox=BLANK_BBOX)

    report = materialize_visual_assets([chunk], pdf, tmp_path / "assets", doc_hash="b2")

    health = report.crops[0]
    # Detection preserved: the audit still records the drift (telemetry + gate).
    assert health.is_low_information is True
    assert health.reextracted is True
    # But the PERSISTED asset is no longer the blank crop - it is the full page,
    # which carries the content block (high variance).
    on_disk = tmp_path / chunk.asset_ref
    assert on_disk.exists()
    assert _disk_std(on_disk) > 10.0


def test_clean_crop_is_not_reextracted(tmp_path):
    """A content-bearing crop is persisted as-is (no full-page fallback)."""
    pdf = _pdf_with_block(tmp_path)
    chunk = _image_uir(bbox=CONTENT_BBOX)

    report = materialize_visual_assets([chunk], pdf, tmp_path / "assets", doc_hash="b2")

    health = report.crops[0]
    assert health.is_low_information is False
    assert health.reextracted is False


def test_reextraction_surfaces_in_meta_json(tmp_path):
    pdf = _pdf_with_block(tmp_path)
    chunk = _image_uir(bbox=BLANK_BBOX)

    report = materialize_visual_assets([chunk], pdf, tmp_path / "assets", doc_hash="b2")
    suspects = report.to_dict()["suspect_assets"]

    assert suspects and suspects[0]["reextracted"] is True


def test_geometric_crop_is_never_reextracted(tmp_path):
    """A trusted geometric crop (B1) is not subject to the B2 re-render."""

    def _checker_png(size=96, sq=8):
        from PIL import Image

        img = Image.new("L", (size, size), 255)
        px = img.load()
        for y in range(size):
            for x in range(size):
                if (x // sq + y // sq) % 2 == 0:
                    px[x, y] = 0
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(fitz.Rect(100, 100, 300, 300), stream=_checker_png())
    pdf = tmp_path / "img.pdf"
    doc.save(str(pdf))
    doc.close()

    # Garbage VLM bbox, but B1 crops the real object -> geometric, never re-run.
    chunk = _image_uir(bbox=(720, 800, 960, 980))
    report = materialize_visual_assets([chunk], pdf, tmp_path / "assets", doc_hash="b2")

    health = report.crops[0]
    assert health.crop_source == "geometric"
    assert health.reextracted is False
