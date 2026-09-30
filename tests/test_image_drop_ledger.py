"""PLAN_QUALITY_REMEDIATION_V1 WP-A2b: every IMAGE chunk the export chain drops is itemized.

Defect (IRJET): "Dropped 7 icon-class image chunk(s)" was the only trace of figures that the source
shows; there was no per-document account of which described images vanished at which filter, and a
drop site nobody logged left no trace at all. The invariant asserted here is

    IMAGE chunks in == IMAGE chunks written + itemized drops

through the real ``process_pdf`` path, so a site that is not wrapped shows as ``unaccounted``.
"""

from __future__ import annotations

import io
import json
import logging
from types import SimpleNamespace

import fitz
from PIL import Image

from mmrag_v2.validators.image_drop_ledger import ImageDropLedger

PAGE_W, PAGE_H = 595, 842


def ch(cid, modality="image", page=1, asset=None):
    return SimpleNamespace(
        chunk_id=cid,
        modality=modality,
        metadata=SimpleNamespace(page_number=page),
        asset_ref=SimpleNamespace(file_path=f"assets/{asset}") if asset else None,
    )


# ------------------------------------------------------------------ ledger unit behaviour
def test_track_itemizes_only_the_image_chunks_a_stage_removed():
    before = [ch("t1", "text"), ch("i1", asset="a_0001_image_000.png"), ch("i2", asset="b.png")]
    after = [before[0], before[2]]
    ledger = ImageDropLedger()
    assert ledger.track("tiny_icon", before, after) is after
    assert [(d.reason, d.chunk_id, d.asset) for d in ledger.drops] == [
        ("tiny_icon", "i1", "a_0001_image_000.png")
    ]


def test_a_removed_text_chunk_is_not_an_image_drop():
    ledger = ImageDropLedger()
    ledger.track("quality_filter", [ch("t1", "text"), ch("i1")], [ch("i1")])
    assert ledger.drops == []


def test_a_replaced_chunk_object_with_the_same_id_is_kept():
    ledger = ImageDropLedger()
    ledger.track("blank_asset", [ch("i1")], [ch("i1")])
    assert ledger.drops == []


def test_chunk_id_dedup_of_two_equal_ids_is_itemized_once():
    ledger = ImageDropLedger()
    ledger.track("chunk_id_duplicate", [ch("i1"), ch("i1")], [ch("i1")])
    assert [d.reason for d in ledger.drops] == ["chunk_id_duplicate"]


def test_unaccounted_exposes_a_drop_site_nobody_wrapped():
    ledger = ImageDropLedger()
    ledger.begin([ch("i1"), ch("i2"), ch("i3")])
    ledger.begin([ch("x")])  # first call wins
    ledger.record("phash_duplicate", ch("i2", asset="dup.png"))
    assert ledger.image_in == 3
    assert ledger.unaccounted(image_out=1) == 1  # i3 vanished without an itemized reason
    lines = ledger.summary_lines(image_out=1)
    assert "unaccounted=1" in lines[0] and "phash_duplicate: 1 -> dup.png" in lines[1]


def test_summary_is_empty_when_nothing_was_dropped_and_everything_is_accounted():
    ledger = ImageDropLedger()
    ledger.begin([ch("i1")])
    assert ledger.summary_lines(image_out=1) == []


def test_summary_truncates_long_asset_lists():
    ledger = ImageDropLedger()
    ledger.begin([ch(f"i{n}") for n in range(20)])
    for n in range(20):
        ledger.record("tiny_icon", ch(f"i{n}", asset=f"a{n}.png"))
    assert "(+8 more)" in ledger.summary_lines(image_out=0)[1]


# ------------------------------------------------------------------ bridge: the real process_pdf path
def _checker(w, h):
    img = Image.new("L", (w, h), 255)
    px = img.load()
    for y in range(h):
        for x in range(w):
            if (x // 4 + y // 4) % 2 == 0:
                px[x, y] = 0
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


FIG = fitz.Rect(320, 300, 550, 460)
ICON = fitz.Rect(60, 200, 80, 220)
STRIP = fitz.Rect(60, 600, 460, 610)


def _pdf(path):
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_text((72, 100), "3. Results and discussion", fontsize=18)
    h = FIG.height / 5
    for i in range(3):
        y = FIG.y0 + i * 2 * h
        box = fitz.Rect(FIG.x0 + 20, y, FIG.x1 - 20, y + h)
        page.draw_rect(box, color=(0, 0, 0), fill=(0.8, 0.85, 1.0), width=1.2)
    page.insert_image(ICON, stream=_checker(20, 20))
    page.insert_image(STRIP, stream=_checker(400, 10))
    doc.save(str(path))
    doc.close()


def _norm(r):
    return (
        round(r.x0 / PAGE_W * 1000),
        round(r.y0 / PAGE_H * 1000),
        round(r.x1 / PAGE_W * 1000),
        round(r.y1 / PAGE_H * 1000),
    )


def _uir():
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
        el(2, ElementType.IMAGE, "A three-step flowchart of the method.", _norm(FIG), "figure"),
        el(3, ElementType.IMAGE, "A small decorative icon.", _norm(ICON), "figure"),
        el(4, ElementType.IMAGE, "A thin table-row band.", _norm(STRIP), "figure"),
    ]
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


def test_process_pdf_itemizes_every_dropped_image_and_the_invariant_holds(
    tmp_path, monkeypatch, caplog
):
    import mmrag_v3.processor as v3p
    from mmrag_v2.batch_processor import BatchProcessor

    pdf = tmp_path / "s.pdf"
    _pdf(pdf)
    monkeypatch.setattr(v3p, "extract", lambda path: _uir())
    out = tmp_path / "out"
    bp = BatchProcessor(output_dir=str(out), batch_size=10, vision_provider="none")
    with caplog.at_level(logging.WARNING, logger="mmrag_v2.batch_processor"):
        bp.process_pdf(pdf)

    rows = [json.loads(ln) for ln in (out / "ingestion.jsonl").read_text("utf-8").splitlines()]
    written = [r for r in rows if r.get("modality") == "image"]
    ledger = bp._image_drops
    assert ledger.image_in == 3
    assert len(written) == 1
    assert sorted(d.reason for d in ledger.drops) == ["thin_strip", "tiny_icon"]
    assert ledger.unaccounted(len(written)) == 0  # in == written + itemized
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "[IMAGE-DROPS]" in text and "tiny_icon: 1" in text and "thin_strip: 1" in text
    assert "unaccounted=0" in text


def test_a_document_that_drops_nothing_logs_no_image_drop_summary(tmp_path, monkeypatch, caplog):
    import mmrag_v3.processor as v3p
    from mmrag_v2.batch_processor import BatchProcessor

    pdf = tmp_path / "s.pdf"
    _pdf(pdf)
    uir = _uir()
    uir.pages[0].elements = uir.pages[0].elements[:3]  # text + the figure only
    monkeypatch.setattr(v3p, "extract", lambda path: uir)
    out = tmp_path / "out"
    bp = BatchProcessor(output_dir=str(out), batch_size=10, vision_provider="none")
    with caplog.at_level(logging.WARNING, logger="mmrag_v2.batch_processor"):
        bp.process_pdf(pdf)
    assert not any("[IMAGE-DROPS]" in r.getMessage() for r in caplog.records)
    assert bp._image_drops.unaccounted(1) == 0
