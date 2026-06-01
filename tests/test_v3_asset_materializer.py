"""V3 vision-native asset materialization + schema conformance + crop-audit.

Two contracts:

1. (2026-06-01) The extraction-layer fix for the crucible-soak schema breach: a
   VLM-native IMAGE/TABLE UIRChunk carries a description but no binary asset, so
   it fails QA-CHECK-05 (IMAGE/TABLE require asset_ref *and* spatial.bbox) at
   ``IngestionChunk.from_uir``. ``materialize_visual_assets`` renders the bbox
   region crop to disk and sets a valid ``asset_ref`` so the chunk conforms.
   The full VLM text stays authoritative in ``content``; ``visual_description``
   is safely fit to the 400-char cap (the cap is NOT raised).

2. (PLAN_V3.1 pre-Crucible hardening) Crop-audit: clamping a hallucinated VLM
   bbox keeps the process alive but can emit a *garbage* PNG that still passes
   QA-CHECK-05. The materializer scores each crop for three drift fingerprints
   (full-page fallback, edge-clamp, low-information/blank) and gates the document
   with QA_WARN_CROP_DRIFT past a threshold. These tests pin every signal and
   the doc-level gate.

Fully offline/deterministic: synthetic fitz PDFs, synthetic UIR chunks, no VLM.
"""

from __future__ import annotations

from pathlib import Path

import fitz

from mmrag_v2.schema.ingestion_schema import FileType, IngestionChunk, Modality
from mmrag_v2.universal.asset_materializer import (
    CROP_AUDIT_PASS,
    QA_WARN_CROP_DRIFT,
    materialize_visual_assets,
)
from mmrag_v2.universal.intermediate import (
    ConfidenceBreakdown,
    CoordinateFrame,
    Locator,
    LocatorType,
    UIRChunk,
)

DOC_ID = "abc123abc123"

# bboxes are [0, COORD_SCALE=1000] in the page-portrait frame.
CONTENT_BBOX = (120, 110, 560, 380)  # covers the black block -> high variance
BLANK_BBOX = (550, 600, 950, 900)  # pure-white quadrant -> low information
EDGE_BBOX = (0, 0, 1000, 600)  # touches the frame -> edge-clamp fingerprint
CENTER_BBOX = (300, 300, 700, 700)  # interior, content-bearing -> clean


def _pdf(tmp_path: Path, pages: int = 1, w: int = 612, h: int = 792) -> Path:
    doc = fitz.open()
    for _ in range(pages):
        doc.new_page(width=w, height=h)
    out = tmp_path / "doc.pdf"
    doc.save(str(out))
    doc.close()
    return out


def _pdf_with_block(tmp_path: Path, w: int = 612, h: int = 792) -> Path:
    """A white page with one solid-black rectangle in the upper-left quadrant.

    A crop overlapping the rectangle has high luminance variance (not blank);
    a crop over the clean quadrant is near-white (blank/low-information).
    """
    doc = fitz.open()
    page = doc.new_page(width=w, height=h)
    page.draw_rect(fitz.Rect(90, 100, 320, 280), color=(0, 0, 0), fill=(0, 0, 0))
    out = tmp_path / "block.pdf"
    doc.save(str(out))
    doc.close()
    return out


def _visual_uir(
    modality: Modality,
    *,
    content: str = "A figure.",
    page: int = 1,
    bbox=(50, 100, 950, 700),
    asset_ref=None,
) -> UIRChunk:
    return UIRChunk(
        modality=modality,
        content=content,
        locator=Locator(
            type=LocatorType.BBOX,
            bbox=list(bbox),
            page_number=page,
            coordinate_frame=CoordinateFrame.PDF_PAGE_PORTRAIT,
        ),
        confidence=ConfidenceBreakdown(),
        extraction_method="vlm_native",
        extraction_engine_version="qwen3-vl-8b",
        asset_ref=asset_ref,
    )


def _text_uir(content: str = "Body text.", page: int = 1) -> UIRChunk:
    return UIRChunk(
        modality=Modality.TEXT,
        content=content,
        locator=Locator(
            type=LocatorType.BBOX,
            bbox=[100, 200, 800, 250],
            page_number=page,
            coordinate_frame=CoordinateFrame.PDF_PAGE_PORTRAIT,
        ),
        confidence=ConfidenceBreakdown(),
        extraction_method="vlm_native",
        extraction_engine_version="qwen3-vl-8b",
    )


def _from_uir(uir: UIRChunk, position: int = 0) -> IngestionChunk:
    return IngestionChunk.from_uir(
        uir,
        doc_id=DOC_ID,
        source_file="doc.pdf",
        file_type=FileType.PDF,
        position=position,
    )


# --------------------------------------------------------------------------
# Contract 1 - materialization + schema conformance
# --------------------------------------------------------------------------


def test_image_chunk_materializes_and_conforms(tmp_path):
    pdf = _pdf(tmp_path)
    chunk = _visual_uir(Modality.IMAGE)

    report = materialize_visual_assets([chunk], pdf, tmp_path / "assets", doc_hash="deadbeef")

    assert report.rendered == 1
    assert chunk.asset_ref == "assets/deadbeef_0001_image_000.png"
    on_disk = tmp_path / chunk.asset_ref
    assert on_disk.exists() and on_disk.stat().st_size > 0

    # The chunk that previously failed QA-CHECK-05 now constructs cleanly.
    ic = _from_uir(chunk)
    assert ic.modality == Modality.IMAGE
    assert ic.asset_ref is not None
    assert ic.asset_ref.file_path == "assets/deadbeef_0001_image_000.png"
    assert ic.metadata.spatial.bbox == [50, 100, 950, 700]


def test_table_chunk_materializes_and_conforms(tmp_path):
    pdf = _pdf(tmp_path)
    chunk = _visual_uir(Modality.TABLE, content="| a | b |\n|---|---|\n| 1 | 2 |")

    report = materialize_visual_assets([chunk], pdf, tmp_path / "assets", doc_hash="d")

    assert report.rendered == 1
    assert chunk.asset_ref == "assets/d_0001_table_000.png"
    assert (tmp_path / chunk.asset_ref).exists()

    ic = _from_uir(chunk)
    assert ic.modality == Modality.TABLE
    assert ic.asset_ref is not None
    assert ic.metadata.search_priority == "medium"


def test_existing_asset_ref_is_not_overwritten(tmp_path):
    pdf = _pdf(tmp_path)
    chunk = _visual_uir(Modality.IMAGE, asset_ref="assets/keep_me.png")

    report = materialize_visual_assets([chunk], pdf, tmp_path / "assets", doc_hash="d")

    assert report.rendered == 0
    assert chunk.asset_ref == "assets/keep_me.png"


def test_text_chunks_are_not_materialized(tmp_path):
    pdf = _pdf(tmp_path)
    chunk = _text_uir()

    report = materialize_visual_assets([chunk], pdf, tmp_path / "assets", doc_hash="d")

    assert report.rendered == 0
    assert chunk.asset_ref is None
    assert _from_uir(chunk).modality == Modality.TEXT


def test_long_description_truncates_but_content_is_preserved(tmp_path):
    pdf = _pdf(tmp_path)
    long_desc = "A densely annotated schematic. " * 30  # ~900 chars, > 400 cap
    assert len(long_desc) > 400
    chunk = _visual_uir(Modality.IMAGE, content=long_desc)
    materialize_visual_assets([chunk], pdf, tmp_path / "assets", doc_hash="d")

    ic = _from_uir(chunk)

    assert ic.content == long_desc  # Directive 1.2: full text in content
    vd = ic.metadata.visual_description  # Directive 1.3: fit to cap
    assert vd is not None
    assert len(vd) <= 400
    assert vd.endswith("...")
    assert long_desc.startswith(vd[:-3])


def test_short_description_is_not_truncated(tmp_path):
    pdf = _pdf(tmp_path)
    chunk = _visual_uir(Modality.IMAGE, content="A small inline icon.")
    materialize_visual_assets([chunk], pdf, tmp_path / "assets", doc_hash="d")

    assert _from_uir(chunk).metadata.visual_description == "A small inline icon."


def test_bridge_mixed_page_conforms_end_to_end(tmp_path):
    """A mixed TEXT+IMAGE+TABLE page emits zero QA-CHECK-05 violations once
    assets are materialized."""
    pdf = _pdf_with_block(tmp_path)
    chunks = [
        _text_uir(),
        _visual_uir(Modality.IMAGE, content="A chart.", bbox=CONTENT_BBOX),
        _visual_uir(Modality.TABLE, content="| x |\n|---|\n| 1 |", bbox=CONTENT_BBOX),
    ]

    report = materialize_visual_assets(chunks, pdf, tmp_path / "assets", doc_hash="mix")
    assert report.rendered == 2  # text skipped; image + table rendered

    ics = [_from_uir(u, position=i) for i, u in enumerate(chunks)]
    assert [c.modality for c in ics] == [Modality.TEXT, Modality.IMAGE, Modality.TABLE]
    for c in ics:
        if c.modality in (Modality.IMAGE, Modality.TABLE):
            assert c.asset_ref is not None
            assert (tmp_path / c.asset_ref.file_path).exists()


# --------------------------------------------------------------------------
# Contract 2 - crop-audit health signals
# --------------------------------------------------------------------------


def test_signal_full_page_fallback_on_degenerate_bbox(tmp_path):
    pdf = _pdf(tmp_path)
    # Sub-2pt region: cannot clip -> full-page fallback.
    chunk = _visual_uir(Modality.IMAGE, bbox=(0, 0, 1, 1))

    report = materialize_visual_assets([chunk], pdf, tmp_path / "assets", doc_hash="d")

    assert report.rendered == 1
    health = report.crops[0]
    assert health.is_full_page_fallback is True
    # A degenerate bbox is reported as fallback, not double-counted as edge.
    assert health.is_edge_clamped is False
    assert (tmp_path / chunk.asset_ref).exists()  # still a valid asset


def test_signal_edge_clamp_on_boundary_bbox(tmp_path):
    pdf = _pdf_with_block(tmp_path)
    edge = _visual_uir(Modality.IMAGE, bbox=EDGE_BBOX)
    center = _visual_uir(Modality.IMAGE, bbox=CENTER_BBOX)

    report = materialize_visual_assets([edge, center], pdf, tmp_path / "assets", doc_hash="d")

    by_ref = {c.asset_ref: c for c in report.crops}
    assert by_ref[edge.asset_ref].is_edge_clamped is True
    assert by_ref[edge.asset_ref].is_full_page_fallback is False
    assert by_ref[center.asset_ref].is_edge_clamped is False


def test_signal_low_information_on_blank_crop(tmp_path):
    pdf = _pdf_with_block(tmp_path)
    blank = _visual_uir(Modality.IMAGE, bbox=BLANK_BBOX)
    content = _visual_uir(Modality.IMAGE, bbox=CONTENT_BBOX)

    report = materialize_visual_assets([blank, content], pdf, tmp_path / "assets", doc_hash="d")

    by_ref = {c.asset_ref: c for c in report.crops}
    assert by_ref[blank.asset_ref].is_low_information is True
    assert by_ref[content.asset_ref].is_low_information is False
    # The content crop's variance is well above the blank std threshold.
    assert by_ref[content.asset_ref].std_luminance > 10.0


# --------------------------------------------------------------------------
# Contract 2 - document-level drift gate
# --------------------------------------------------------------------------


def test_gate_passes_when_crops_are_clean(tmp_path):
    pdf = _pdf_with_block(tmp_path)
    chunks = [_visual_uir(Modality.IMAGE, bbox=CONTENT_BBOX) for _ in range(4)]

    report = materialize_visual_assets(chunks, pdf, tmp_path / "assets", doc_hash="clean")

    assert report.rendered == 4
    assert report.drift_flagged == 0
    assert report.exceeds_threshold is False
    assert report.gate_status == CROP_AUDIT_PASS


def test_gate_warns_when_drift_exceeds_threshold(tmp_path):
    pdf = _pdf_with_block(tmp_path)
    # 1 blank out of 4 = 25% drift > 15% default threshold.
    chunks = [
        _visual_uir(Modality.IMAGE, bbox=CONTENT_BBOX),
        _visual_uir(Modality.IMAGE, bbox=CONTENT_BBOX),
        _visual_uir(Modality.IMAGE, bbox=CONTENT_BBOX),
        _visual_uir(Modality.IMAGE, bbox=BLANK_BBOX),
    ]

    report = materialize_visual_assets(chunks, pdf, tmp_path / "assets", doc_hash="drift")

    assert report.rendered == 4
    assert report.drift_flagged == 1
    assert report.drift_rate == 0.25
    assert report.exceeds_threshold is True
    assert report.gate_status == QA_WARN_CROP_DRIFT
    # The report serializes the suspect crop for meta.json triage.
    suspects = report.to_dict()["suspect_assets"]
    assert any(s["is_low_information"] for s in suspects)


def test_gate_to_dict_shape_is_meta_json_ready(tmp_path):
    pdf = _pdf_with_block(tmp_path)
    chunks = [_visual_uir(Modality.IMAGE, bbox=CONTENT_BBOX)]

    payload = materialize_visual_assets(chunks, pdf, tmp_path / "assets", doc_hash="d").to_dict()

    for key in (
        "gate_status",
        "rendered",
        "drift_flagged",
        "drift_rate",
        "warn_threshold",
        "full_page_fallbacks",
        "edge_clamped",
        "low_information",
        "suspect_assets",
    ):
        assert key in payload
