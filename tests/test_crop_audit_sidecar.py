"""PLAN_QUALITY_REMEDIATION_V1 WP-A1 (sidecar + call-site bridge).

The crop audit report used to be discarded at the batch_processor call site, so a wrong-object crop
(the header logo standing in for Fig 3) left no trace anywhere. ``crop_audit.json`` is the
frame-invariant record (VLM box and rendered rectangle in PDF points, asset pixels, reason) that the
figure-fidelity checks read.

Bridge test: the guard and the sidecar are proven through the real ``process_pdf`` call path with a
stubbed extractor, not only through ``materialize_visual_assets`` in isolation.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import fitz
from PIL import Image

PAGE_W, PAGE_H = 595, 842
FIG = fitz.Rect(320, 300, 550, 460)
LOGO = fitz.Rect(40, 40, 74, 74)


def _checker(size: int = 96) -> bytes:
    img = Image.new("L", (size, size), 255)
    px = img.load()
    for y in range(size):
        for x in range(size):
            if (x // 8 + y // 8) % 2 == 0:
                px[x, y] = 0
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _pdf_with_vector_figure_and_logo(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_image(LOGO, stream=_checker())
    page.insert_text((72, 100), "3. Results and discussion", fontsize=18)
    h = FIG.height / 5
    for i in range(3):
        y = FIG.y0 + i * 2 * h
        box = fitz.Rect(FIG.x0 + 20, y, FIG.x1 - 20, y + h)
        page.draw_rect(box, color=(0, 0, 0), fill=(0.8, 0.85, 1.0), width=1.2)
        if i < 2:
            page.draw_line((box.x0 + box.width / 2, box.y1), (box.x0 + box.width / 2, box.y1 + h))
    doc.save(str(path))
    doc.close()


def _uir(with_figure: bool):
    from mmrag_v2.universal.intermediate import (
        BoundingBox,
        Element,
        ElementType,
        ExtractionMethod,
        PageClassification,
        UniversalDocument,
        UniversalPage,
    )

    def el(i, etype, content, bbox, label):
        return Element(
            type=etype,
            content=content,
            bbox=BoundingBox(*bbox),
            confidence=0.9,
            extraction_method=ExtractionMethod.VLM,
            element_index=i,
            source_label=label,
        )

    elements = [
        el(0, ElementType.TEXT, "3. Results and discussion", (100, 100, 900, 130), "heading"),
        el(
            1,
            ElementType.TEXT,
            "A body paragraph long enough to be a chunk of text on this page. " * 3,
            (100, 140, 900, 250),
            "paragraph",
        ),
    ]
    if with_figure:
        bb = (
            round(FIG.x0 / PAGE_W * 1000),
            round(FIG.y0 / PAGE_H * 1000),
            round(FIG.x1 / PAGE_W * 1000),
            round(FIG.y1 / PAGE_H * 1000),
        )
        elements.append(
            el(2, ElementType.IMAGE, "A three-step flowchart of the proposed method.", bb, "figure")
        )
    return UniversalDocument(
        doc_id="x",
        source_file="s.pdf",
        file_type="pdf",
        pages=[
            UniversalPage(
                page_number=1,
                elements=elements,
                classification=PageClassification.DIGITAL,
                dimensions=(1132, 1600),
            )
        ],
        total_pages=1,
    )


def _convert(tmp_path, monkeypatch, with_figure: bool):
    import mmrag_v3.processor as v3p
    from mmrag_v2.batch_processor import BatchProcessor

    pdf = tmp_path / "s.pdf"
    _pdf_with_vector_figure_and_logo(pdf)
    monkeypatch.setattr(v3p, "extract", lambda path: _uir(with_figure))
    out = tmp_path / "out"
    BatchProcessor(output_dir=str(out), batch_size=10, vision_provider="none").process_pdf(pdf)
    rows = [
        json.loads(ln) for ln in (out / "ingestion.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    return out, rows


def test_figure_crop_is_the_vector_figure_not_the_logo_and_the_sidecar_says_so(
    tmp_path, monkeypatch
):
    out, rows = _convert(tmp_path, monkeypatch, with_figure=True)
    images = [r for r in rows if r.get("modality") == "image"]
    assert len(images) == 1
    ref = images[0]["asset_ref"]["file_path"]
    with Image.open(out / ref) as img:
        # the figure is 230x160 pt at zoom 2; the logo would be 68x68 px
        assert img.width > 300 and img.height > 200

    audit = json.loads((out / "crop_audit.json").read_text(encoding="utf-8"))
    assert len(audit["crops"]) == 1
    rec = audit["crops"][0]
    assert rec["crop_source"] == "vlm" and rec["crop_reason"] == "vlm_kept_graphics_in_box"
    assert rec["asset_px"] == [img.width, img.height]
    assert rec["asset_ref"].endswith(Path(ref).name)
    assert abs(rec["clip_rect_pt"][0] - FIG.x0) < 3 and abs(rec["clip_rect_pt"][3] - FIG.y1) < 3


def test_a_text_only_document_writes_no_crop_audit_sidecar(tmp_path, monkeypatch):
    out, rows = _convert(tmp_path, monkeypatch, with_figure=False)
    assert not [r for r in rows if r.get("modality") == "image"]
    assert not (out / "crop_audit.json").exists()
