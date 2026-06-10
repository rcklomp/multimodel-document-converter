"""V3 VLM adapter: CODE/FORM smuggle-and-promote (Charter-respecting, 2026-06-01).

Qwen3-VL emits ``"type": "code"`` / ``"type": "form"``, but ``ElementType`` is
frozen at the 3-value legacy vocabulary (Charter §7.1: it is being replaced by
``Modality``, not widened). So the adapter SMUGGLES code/form through the
intermediate layer as ``ElementType.TEXT`` (tagging the original signal on the
element), and the chunker PROMOTES it to ``Modality.CODE`` / ``Modality.FORM``
at the ElementType->Modality boundary - where the 5-value widening belongs.

This pins the full path: a raw VLM JSON ``{'type':'code'}`` parses (no enum
crash), traverses the intermediate layer as ``ElementType.TEXT``, and resolves
as a final UIR chunk with ``Modality.CODE`` that constructs a valid
``IngestionChunk`` via ``from_uir``. Same for form. Also: ``ElementType`` stays
3-value, an unknown type degrades to TEXT (page preserved), and the prompt
advertises the new types. Fully offline/deterministic: no VLM, no network.
"""

from __future__ import annotations

import fitz

from mmrag_v2.chunking.uir_chunker import chunk_universal_document
from mmrag_v2.schema.ingestion_schema import ChunkType, FileType, IngestionChunk, Modality
from mmrag_v2.universal.intermediate import DocumentMetadata, ElementType, create_document
from mmrag_v3.engines.vlm_native import VlmNativeEngine, _build_schema_prompt

W, H = 1000, 1000


def _page_from_vlm(elements):
    payload = {
        "page_number": 1,
        "width": W,
        "height": H,
        "classification": "digital",
        "elements": elements,
    }
    return VlmNativeEngine._page_from_payload(
        payload, fallback_page_number=1, pixel_width=W, pixel_height=H
    )


def _doc(tmp_path, page):
    pdf = tmp_path / "t.pdf"
    d = fitz.open()
    d.new_page(width=612, height=792)
    d.save(str(pdf))
    d.close()
    return create_document(
        file_path=pdf,
        file_type="pdf",
        pages=[page],
        metadata=DocumentMetadata(
            page_count=1, file_size_bytes=1, has_text_layer=True, has_images=False
        ),
    )


def _el(type_, content, bbox=(10, 10, 500, 200)):
    return {"type": type_, "content": content, "bbox": list(bbox), "confidence": 0.95}


def test_code_smuggles_as_text_promotes_to_code_modality(tmp_path):
    code = "def f(x):\n    if x:\n        return 1\n    return 0"
    page = _page_from_vlm([_el("code", code)])

    # Intermediate layer: ElementType stays TEXT (the smuggle), but the original
    # signal is preserved on the element for the chunker to promote.
    assert page.elements[0].type == ElementType.TEXT
    assert page.elements[0].metadata.get("promoted_modality") == "code"
    # Provenance marker captured alongside the routing flag.
    assert page.elements[0].metadata.get("original_vlm_type") == "code"

    # Chunk layer: promoted to Modality.CODE, indentation preserved verbatim.
    chunks = chunk_universal_document(_doc(tmp_path, page))
    code_chunks = [c for c in chunks if c.modality == Modality.CODE]
    assert len(code_chunks) == 1
    assert "        return 1" in code_chunks[0].content
    assert code_chunks[0].original_vlm_type == "code"

    ic = IngestionChunk.from_uir(
        code_chunks[0], doc_id="abc123abc123", source_file="t.pdf", file_type=FileType.PDF
    )
    assert ic.modality == Modality.CODE
    assert ic.metadata.chunk_type == ChunkType.CODE
    # F4 resolved contract (eeffcff, user-adjudicated 2026-06-10): code MUST be
    # fenced at the chunker chokepoint; indentation preserved verbatim inside.
    assert ic.content.lstrip().startswith("```")
    assert code in ic.content
    # Provenance survives serialization into the output metadata.
    assert ic.metadata.original_vlm_type == "code"


def test_form_smuggles_as_text_promotes_to_form_modality(tmp_path):
    form = "Name: John Doe\nDate: 2026-01-01\n| Field | Value |\n|---|---|\n| Total | 42 |"
    page = _page_from_vlm([_el("form", form)])
    assert page.elements[0].type == ElementType.TEXT
    assert page.elements[0].metadata.get("promoted_modality") == "form"
    assert page.elements[0].metadata.get("original_vlm_type") == "form"

    chunks = chunk_universal_document(_doc(tmp_path, page))
    form_chunks = [c for c in chunks if c.modality == Modality.FORM]
    assert len(form_chunks) == 1
    assert form_chunks[0].original_vlm_type == "form"

    # FORM needs no asset_ref -> from_uir must not raise QA-CHECK-05.
    ic = IngestionChunk.from_uir(
        form_chunks[0], doc_id="abc123abc123", source_file="t.pdf", file_type=FileType.PDF
    )
    assert ic.modality == Modality.FORM
    assert ic.metadata.original_vlm_type == "form"


def test_elementtype_stays_three_values():
    # The whole point of Option 1: the legacy enum is NOT widened.
    assert {e.name for e in ElementType} == {"TEXT", "IMAGE", "TABLE"}
    assert "CODE" not in {e.name for e in ElementType}
    assert "FORM" not in {e.name for e in ElementType}


def test_unknown_type_degrades_to_text_without_dropping_page(tmp_path):
    # 'formula' is unknown -> degrades to plain TEXT (no promotion). The sibling
    # code element on the same page still smuggles+promotes; the page survives.
    page = _page_from_vlm(
        [
            _el("formula", "E = mc^2", bbox=(10, 10, 300, 60)),
            _el("code", "print('ok')", bbox=(10, 80, 300, 140)),
        ]
    )
    assert len(page.elements) == 2
    assert page.elements[0].type == ElementType.TEXT
    assert page.elements[0].metadata.get("promoted_modality") is None  # plain text
    # The degraded type is NOT silently lost: provenance marker is recorded.
    assert page.elements[0].metadata.get("original_vlm_type") == "formula"
    assert page.elements[1].type == ElementType.TEXT
    assert page.elements[1].metadata.get("promoted_modality") == "code"
    assert page.elements[1].metadata.get("original_vlm_type") == "code"

    chunks = chunk_universal_document(_doc(tmp_path, page))
    assert any(c.modality == Modality.CODE for c in chunks)


def test_degraded_unknown_marker_survives_chunking_and_serialization(tmp_path):
    # A pure degraded-unknown page (no sibling code) must still carry the
    # 'formula' marker through the text-grouping path onto the final chunk and
    # into the serialized IngestionChunk metadata - the previously-silent gap.
    page = _page_from_vlm([_el("formula", "E = mc^2 is a famous equation. " * 4)])
    assert page.elements[0].metadata.get("original_vlm_type") == "formula"

    chunks = chunk_universal_document(_doc(tmp_path, page))
    text_chunks = [c for c in chunks if c.modality == Modality.TEXT]
    assert text_chunks, "degraded element must still produce a text chunk"
    assert text_chunks[0].original_vlm_type == "formula"

    ic = IngestionChunk.from_uir(
        text_chunks[0], doc_id="abc123abc123", source_file="t.pdf", file_type=FileType.PDF
    )
    assert ic.modality == Modality.TEXT
    assert ic.metadata.original_vlm_type == "formula"


def test_native_text_has_no_provenance_marker(tmp_path):
    # Ordinary prose (VLM type 'text') must NOT get a spurious marker.
    page = _page_from_vlm([_el("text", "Just some ordinary prose. " * 4)])
    assert page.elements[0].metadata is None or (
        page.elements[0].metadata.get("original_vlm_type") is None
    )
    chunks = chunk_universal_document(_doc(tmp_path, page))
    text_chunks = [c for c in chunks if c.modality == Modality.TEXT]
    assert text_chunks
    assert text_chunks[0].original_vlm_type is None


def test_prompt_advertises_code_and_form_with_instructions():
    p = _build_schema_prompt(800, 1000)
    assert '"code"' in p and '"form"' in p
    assert "indentation" in p.lower()  # code instruction
    assert "key-value" in p.lower() or "reading order" in p.lower()  # form instruction
