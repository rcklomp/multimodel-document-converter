"""PLAN_QUALITY_REMEDIATION_V1 WP-0.5: opt-in UIR dump for deterministic chunker A/B.

The replay fidelity test is the point: chunking the reloaded document must give exactly what
chunking the original gives, otherwise a chunker A/B on a dump would be meaningless. The bridge
test drives the real ``_process_single_batch`` so the call site (which carries the TOC and the
carry-in heading across the object boundary) is proven, not just the two ends.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mmrag_v2.chunking.uir_chunker import chunk_universal_document
from mmrag_v2.universal.intermediate import (
    BoundingBox,
    DocumentMetadata,
    Element,
    ElementType,
    ExtractionMethod,
    PageClassification,
    UniversalDocument,
    UniversalPage,
)
from mmrag_v2.universal.serialization import dump_uir, load_uir, uir_from_dict, uir_to_dict


def _doc() -> UniversalDocument:
    def el(t, content, bbox, label="", idx=0, meta=None):
        return Element(
            type=t,
            content=content,
            bbox=BoundingBox(*bbox) if bbox else None,
            confidence=0.9,
            extraction_method=ExtractionMethod.VLM,
            element_index=idx,
            source_label=label,
            metadata=meta or {},
        )

    page1 = UniversalPage(
        page_number=1,
        elements=[
            el(ElementType.TEXT, "1. INTRODUCTION", (50, 50, 400, 80), "heading", 0),
            el(
                ElementType.TEXT,
                "First paragraph of the introduction. " * 3,
                (50, 90, 900, 200),
                "paragraph",
                1,
            ),
            el(
                ElementType.IMAGE,
                "A diagram of the circuit.",
                (100, 220, 600, 500),
                "figure",
                2,
                {"promoted_modality": None},
            ),
            el(
                ElementType.TABLE,
                "| a | b |\n| --- | --- |\n| 1 | 2 |",
                (100, 520, 600, 700),
                "table",
                3,
            ),
            el(
                ElementType.TEXT,
                "def f(x):\n    return x",
                (50, 720, 900, 800),
                "code",
                4,
                {"promoted_modality": "code", "tags": ("x", "y")},
            ),
        ],
        classification=PageClassification.DIGITAL,
        dimensions=(1132, 1600),
        text_density=0.4,
        avg_confidence=0.9,
    )
    return UniversalDocument(
        doc_id="abc123",
        source_file="sample.pdf",
        file_type="pdf",
        pages=[page1],
        metadata=DocumentMetadata(
            title="T",
            page_count=1,
            creation_date=datetime(2026, 1, 2, 3, 4, 5),
            extra={"extraction_engine": "hybrid"},
        ),
        total_pages=1,
        created_at=datetime(2026, 9, 29, 12, 0, 0),
    )


def test_round_trip_is_exact_for_json_typed_metadata():
    doc = _doc()
    # tuples are the one documented lossy case: give the fixture a list so equality is exact
    doc.pages[0].elements[4].metadata["tags"] = ["x", "y"]
    assert uir_from_dict(uir_to_dict(doc)) == doc


def test_numpy_image_is_not_preserved_but_flagged_and_odd_metadata_is_coerced():
    doc = _doc()
    e = doc.pages[0].elements[1]
    e.raw_image = np.zeros((4, 4), dtype=np.uint8)
    e.metadata["score"] = np.float32(0.5)
    e.metadata["seen"] = {2, 1}
    d = uir_to_dict(doc)
    assert d["pages"][0]["elements"][1]["has_raw_image"] is True
    back = uir_from_dict(d)
    assert back.pages[0].elements[1].raw_image is None
    assert back.pages[0].elements[1].metadata["score"] == 0.5
    assert back.pages[0].elements[1].metadata["seen"] == [1, 2]


def test_unknown_schema_is_rejected():
    d = uir_to_dict(_doc())
    d["schema"] = 99
    with pytest.raises(ValueError):
        uir_from_dict(d)


def test_replay_fidelity_chunking_the_reloaded_document_matches_the_original(tmp_path):
    doc = _doc()
    inputs = {
        "toc_headings": {
            1: ["Document", "Chapter 1"],
            "__heading_map__": {"1. INTRODUCTION": ["Document", "1. INTRODUCTION"]},
        },
        "carry_in_heading": None,
    }
    path = dump_uir(doc, tmp_path / "d.uir.json", inputs)
    reloaded, reloaded_inputs = load_uir(path)
    assert reloaded_inputs["toc_headings"][1] == ["Document", "Chapter 1"]  # int page key restored

    def shape(chunks):
        return [
            (
                c.modality,
                c.content,
                c.parent_heading,
                list(c.bbox or []) if hasattr(c, "bbox") else None,
            )
            for c in chunks
        ]

    a = chunk_universal_document(doc, toc_headings=inputs["toc_headings"])
    b = chunk_universal_document(reloaded, toc_headings=reloaded_inputs["toc_headings"])
    assert shape(a) == shape(b)


def _bp(tmp_path):
    from mmrag_v2.batch_processor import BatchProcessor

    bp = BatchProcessor.__new__(BatchProcessor)
    bp._doc_hash = "hash9"
    bp._carry_heading = "Carried"
    bp._carry_breadcrumb = ["Document", "Carried"]
    return bp


def test_hook_writes_nothing_when_the_env_is_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("MMRAG_DUMP_UIR", raising=False)
    _bp(tmp_path)._maybe_dump_uir(_doc(), SimpleNamespace(batch_index=0), {})
    assert list(tmp_path.iterdir()) == []


def test_hook_writes_a_named_dump_when_the_env_is_set(tmp_path, monkeypatch):
    monkeypatch.setenv("MMRAG_DUMP_UIR", str(tmp_path / "dumps"))
    _bp(tmp_path)._maybe_dump_uir(_doc(), SimpleNamespace(batch_index=2), {"batch_index": 2})
    out = tmp_path / "dumps" / "hash9_b002.uir.json"
    assert out.exists()
    doc, inputs = load_uir(out)
    assert doc.doc_id == "abc123" and inputs["batch_index"] == 2


def test_hook_never_raises_even_when_the_dump_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("MMRAG_DUMP_UIR", str(tmp_path / "dumps"))
    import mmrag_v2.universal.serialization as ser

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(ser, "dump_uir", boom)
    _bp(tmp_path)._maybe_dump_uir(_doc(), SimpleNamespace(batch_index=0), {})  # must not raise


def test_call_site_bridge_passes_toc_and_carry_into_the_dump(tmp_path, monkeypatch):
    """The chunker's OTHER inputs must cross the boundary into the dump (bridge test)."""
    import mmrag_v3.processor as v3p
    from mmrag_v2.batch_processor import BatchProcessor

    captured = {}
    bp = BatchProcessor.__new__(BatchProcessor)
    bp._doc_hash = "h"
    bp._carry_heading = "Prev Chapter"
    bp._carry_breadcrumb = ["Document", "Prev Chapter"]
    bp._intelligence_metadata = {}
    bp._toc_headings = {5: ["Document", "Ch"], "__heading_map__": {}}
    bp._accumulate_extraction_provenance = lambda d: None
    bp._render_visual_assets = lambda *a, **k: None
    bp._next_chunk_position = lambda: 0
    monkeypatch.setattr(v3p, "extract", lambda path: _doc())
    monkeypatch.setattr(bp, "_maybe_dump_uir", lambda doc, info, inputs: captured.update(inputs))
    batch_info = SimpleNamespace(
        batch_index=1, page_offset=4, batch_path=Path("x.pdf"), page_range_str="5-9"
    )
    split_result = SimpleNamespace(batch_count=3)
    try:
        bp._process_single_batch(batch_info, split_result, "sample.pdf")
    except (
        Exception
    ):  # noqa: BLE001 - downstream from_uir needs the real processor state; the capture happens first
        pass
    assert captured["carry_in_heading"] == "Prev Chapter"
    assert captured["toc_headings"] == {
        1: ["Document", "Ch"],
        "__heading_map__": {},
    }  # shifted to batch-local page 1
    assert captured["batch_index"] == 1 and captured["page_offset"] == 4
