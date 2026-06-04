"""V3 page-dimension propagation (crucible fix #1, 2026-06-04).

The bounded crucible re-run found every V3 chunk carried a bbox but
``spatial.page_width/page_height = None`` - a hard TABLE structural QA failure
on CarOK + Form, advisory elsewhere. The UIR page has ``dimensions``; they were
dropped. Fix: the chunker stamps page dims onto each Locator and
``IngestionChunk.from_uir`` reads them, so EVERY caller (batch path + soak
harness) gets correct dims without threading them.

Deterministic/offline: synthetic UIR, no VLM.
"""

from __future__ import annotations

import fitz

from mmrag_v2.chunking.uir_chunker import chunk_universal_document
from mmrag_v2.schema.ingestion_schema import FileType, IngestionChunk
from mmrag_v2.universal.intermediate import (
    DocumentMetadata,
    ElementType,
    ExtractionMethod,
    PageClassification,
    create_document,
    create_element,
    create_page,
)

PW, PH = 1654, 2339


def _real_pdf(tmp_path):
    pdf = tmp_path / "x.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf))
    doc.close()
    return pdf


def _doc(tmp_path) -> "object":
    text = create_element(
        element_type=ElementType.TEXT,
        content="Body text on the page.",
        bbox=[100, 100, 900, 150],
        extraction_method=ExtractionMethod.VLM,
        element_index=0,
    )
    table = create_element(
        element_type=ElementType.TABLE,
        content="| a | b |\n|---|---|\n| 1 | 2 |",
        bbox=[100, 200, 900, 400],
        extraction_method=ExtractionMethod.VLM,
        element_index=1,
    )
    page = create_page(
        page_number=1,
        elements=[text, table],
        dimensions=(PW, PH),
        classification=PageClassification.DIGITAL,
    )
    return create_document(
        file_path=_real_pdf(tmp_path),
        file_type="pdf",
        pages=[page],
        metadata=DocumentMetadata(
            page_count=1, file_size_bytes=1, has_text_layer=True, has_images=False
        ),
    )


def test_chunker_stamps_page_dims_on_locator(tmp_path):
    chunks = chunk_universal_document(_doc(tmp_path))
    assert chunks, "chunker produced no chunks"
    for c in chunks:
        assert c.locator.page_width == PW
        assert c.locator.page_height == PH


def test_from_uir_surfaces_page_dims_including_table(tmp_path):
    chunks = chunk_universal_document(_doc(tmp_path))
    for pos, uir in enumerate(chunks):
        # IMAGE/TABLE need an asset_ref for QA-CHECK-05 (set post-materialization
        # in production); simulate it so from_uir constructs.
        if uir.modality.value in ("image", "table"):
            uir.asset_ref = "assets/x.png"
        ic = IngestionChunk.from_uir(
            uir,
            doc_id="abc123abc123",
            source_file="x.pdf",
            file_type=FileType.PDF,
            position=pos,
        )
        assert ic.metadata.spatial is not None
        assert ic.metadata.spatial.page_width == PW
        assert ic.metadata.spatial.page_height == PH


def test_explicit_param_still_overrides_locator(tmp_path):
    chunks = chunk_universal_document(_doc(tmp_path))
    uir = chunks[0]
    ic = IngestionChunk.from_uir(
        uir,
        doc_id="abc123abc123",
        source_file="x.pdf",
        file_type=FileType.PDF,
        position=0,
        page_width=999,
        page_height=888,
    )
    assert ic.metadata.spatial.page_width == 999
    assert ic.metadata.spatial.page_height == 888
