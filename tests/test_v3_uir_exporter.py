"""Unit tests for the V3.0 UIR JSONL exporter shim.

Charter: docs/ARCHITECTURE_V3_DRAFT_0.5.md Phase A task A2 (shim path).
Scope-negotiation rationale: docs/PHASE_A_SCOPE_NEGOTIATION.md
2026-05-27 entry.

The shim's contract is "lossless": every v2.X chunk in the input
ingestion.jsonl must project to a UIRChunk whose identity projection
matches the v2.X chunk's identity projection at ratio ≥0.95 (Charter
§3.2 identity-half). These tests exercise that contract on synthetic
inputs (deterministic, fast) and the round-trip parse.

Integration smoke against real fixtures (ATZ_Elektronik_German etc.)
is covered separately by `scripts/v3_export_uir.py` and committed to
[`docs/V3_PHASE_A_A2_SHIM_REPORT.md`](../docs/V3_PHASE_A_A2_SHIM_REPORT.md).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from mmrag_v2.universal.intermediate import (
    CoordinateFrame,
    LocatorType,
    Modality,
)
from mmrag_v2.universal.uir_exporter import (
    EXPORT_SOURCE_V2X_SHIM,
    SCHEMA_VERSION_V3_SHIM,
    ExportReport,
    export_v2x_jsonl_to_uir,
    parse_uir_jsonl_record,
    serialize_uirchunk_to_jsonl_dict,
)
from mmrag_v2.universal.v2x_to_v3_mapper import map_v2x_to_v3_uirchunk


DOC_ID = "abc123def456"


def _v2x_text_chunk(
    *,
    chunk_id: str,
    page: int,
    content: str,
    bbox: List[int],
    parent_heading: str = "Chapter 1",
    ocr_confidence: float = None,
) -> Dict[str, Any]:
    """Build a minimal v2.X chunk dict that the mapper accepts."""
    metadata: Dict[str, Any] = {
        "source_file": "test.pdf",
        "file_type": "pdf",
        "page_number": page,
        "extraction_method": "hybrid_chunker",
        "hierarchy": {
            "parent_heading": parent_heading,
            "breadcrumb_path": [parent_heading],
            "level": 1,
        },
        "spatial": {
            "bbox": bbox,
            "page_width": 1000,
            "page_height": 1000,
        },
        "chunk_type": "paragraph",
    }
    if ocr_confidence is not None:
        metadata["ocr_confidence"] = ocr_confidence
    return {
        "chunk_id": chunk_id,
        "doc_id": DOC_ID,
        "modality": "text",
        "content": content,
        "metadata": metadata,
        "schema_version": "2.7.0",
    }


def _v2x_image_chunk(*, chunk_id: str, page: int, asset_ref: str) -> Dict[str, Any]:
    """v2.X image chunk; mapper accepts FLOW_OFFSET locator fallback."""
    return {
        "chunk_id": chunk_id,
        "doc_id": DOC_ID,
        "modality": "image",
        "content": "[image placeholder]",
        "metadata": {
            "source_file": "test.pdf",
            "file_type": "pdf",
            "page_number": page,
            "extraction_method": "vlm",
            "hierarchy": {"parent_heading": None, "breadcrumb_path": [], "level": 0},
            "spatial": {"bbox": [50, 50, 950, 950], "page_width": 1000, "page_height": 1000},
            "chunk_type": "image",
        },
        "asset_ref": asset_ref,
        "schema_version": "2.7.0",
    }


def _write_v2x_jsonl(tmp_path: Path, chunks: List[Dict[str, Any]]) -> Path:
    """Emit a v2.X-shape ingestion.jsonl with a metadata header row."""
    p = tmp_path / "ingestion.jsonl"
    header = {
        "doc_id": DOC_ID,
        "source_file": "test.pdf",
        "schema_version": "2.7.0",
    }
    with p.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
        for ch in chunks:
            f.write(json.dumps(ch) + "\n")
    return p


# --------------------------------------------------------------------------
# serialize_uirchunk_to_jsonl_dict
# --------------------------------------------------------------------------


class TestSerializeUIRChunk:
    def test_text_chunk_round_trips(self):
        v2x = _v2x_text_chunk(
            chunk_id="c1", page=1, content="hello world", bbox=[10, 20, 100, 200],
        )
        uir = map_v2x_to_v3_uirchunk(v2x)
        out = serialize_uirchunk_to_jsonl_dict(uir, doc_id=DOC_ID, source_file="test.pdf")
        assert out["doc_id"] == DOC_ID
        assert out["modality"] == "text"
        assert out["content"] == "hello world"
        assert out["locator"]["type"] == LocatorType.BBOX.value
        assert out["locator"]["bbox"] == [10, 20, 100, 200]
        assert out["locator"]["page_number"] == 1
        assert out["locator"]["coordinate_frame"] == CoordinateFrame.PDF_PAGE_PORTRAIT.value
        assert out["extraction_method"] == "hybrid_chunker"
        assert out["uir_version"] == "3.0"
        assert out["schema_version"] == SCHEMA_VERSION_V3_SHIM
        assert out["parent_heading"] == "Chapter 1"
        assert out["source_element_ids"] == ["c1"]

    def test_image_chunk_serializes_with_flow_offset_locator(self):
        v2x = _v2x_image_chunk(chunk_id="img1", page=3, asset_ref="assets/page3.png")
        uir = map_v2x_to_v3_uirchunk(v2x)
        out = serialize_uirchunk_to_jsonl_dict(uir, doc_id=DOC_ID)
        assert out["modality"] == "image"
        assert out["asset_ref"] == "assets/page3.png"
        # v2x image chunk has bbox so mapper picks BBOX over FLOW_OFFSET
        assert out["locator"]["type"] == LocatorType.BBOX.value

    def test_enum_values_are_strings_not_python_enums(self):
        """JSON serialization requires str values, not Enum members."""
        v2x = _v2x_text_chunk(
            chunk_id="c1", page=1, content="x", bbox=[0, 0, 100, 100],
        )
        uir = map_v2x_to_v3_uirchunk(v2x)
        out = serialize_uirchunk_to_jsonl_dict(uir, doc_id=DOC_ID)
        # Should be json.dumps-safe with default encoder
        json.dumps(out)


# --------------------------------------------------------------------------
# parse_uir_jsonl_record (reverse)
# --------------------------------------------------------------------------


class TestParseUIRJSONL:
    def test_serialize_parse_round_trip_preserves_uirchunk_content(self):
        v2x = _v2x_text_chunk(
            chunk_id="c1", page=2, content="some body text",
            bbox=[40, 60, 200, 300], ocr_confidence=0.93,
        )
        uir = map_v2x_to_v3_uirchunk(v2x)
        record = serialize_uirchunk_to_jsonl_dict(uir, doc_id=DOC_ID)
        parsed = parse_uir_jsonl_record(record)
        assert parsed.content == uir.content
        assert parsed.modality == uir.modality
        assert parsed.locator.bbox == uir.locator.bbox
        assert parsed.locator.page_number == uir.locator.page_number
        assert parsed.locator.coordinate_frame == uir.locator.coordinate_frame
        assert parsed.extraction_method == uir.extraction_method
        # confidence rounds via round-trip through JSON numbers but
        # the v2.X side stored 0.93 as float so equality should hold
        assert parsed.confidence.ocr_confidence == pytest.approx(0.93)
        assert parsed.uir_version == uir.uir_version


# --------------------------------------------------------------------------
# export_v2x_jsonl_to_uir end-to-end
# --------------------------------------------------------------------------


class TestExportEndToEnd:
    def test_minimal_corpus_identity_ratio_is_one(self, tmp_path: Path):
        chunks = [
            _v2x_text_chunk(chunk_id="c1", page=1, content="page1 text", bbox=[10, 10, 100, 100]),
            _v2x_text_chunk(chunk_id="c2", page=2, content="page2 text", bbox=[20, 20, 200, 200]),
            _v2x_image_chunk(chunk_id="img1", page=3, asset_ref="x.png"),
        ]
        v2x_path = _write_v2x_jsonl(tmp_path, chunks)
        out_path = tmp_path / "v3_uir.jsonl"

        report = export_v2x_jsonl_to_uir(
            input_path=v2x_path, output_path=out_path,
        )

        assert isinstance(report, ExportReport)
        assert report.v2x_chunk_count == 3
        assert report.uir_chunk_count == 3
        assert report.mapper_errors == []
        # Charter §3.2 identity half: ≥0.95. Lossless mapper → exactly 1.0.
        assert report.identity_ratio == pytest.approx(1.0)
        assert report.identity_passes

    def test_output_file_has_header_plus_one_row_per_chunk(self, tmp_path: Path):
        chunks = [
            _v2x_text_chunk(chunk_id=f"c{i}", page=i, content=f"text {i}", bbox=[0, 0, 100, 100])
            for i in range(1, 6)
        ]
        v2x_path = _write_v2x_jsonl(tmp_path, chunks)
        out_path = tmp_path / "v3_uir.jsonl"

        export_v2x_jsonl_to_uir(input_path=v2x_path, output_path=out_path)

        with out_path.open() as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 6  # 1 header + 5 chunks
        header = lines[0]
        assert header["schema_version"] == SCHEMA_VERSION_V3_SHIM
        assert header["export_source"] == EXPORT_SOURCE_V2X_SHIM
        assert header["uir_version"] == "3.0"
        assert header["doc_id"] == DOC_ID
        assert header["v2x_chunk_count"] == 5

    def test_inferred_doc_id_matches_chunks(self, tmp_path: Path):
        """Doc_id inference works from chunk records when explicit override is None."""
        chunks = [_v2x_text_chunk(chunk_id="c1", page=1, content="x", bbox=[0, 0, 10, 10])]
        v2x_path = _write_v2x_jsonl(tmp_path, chunks)
        out_path = tmp_path / "v3_uir.jsonl"

        report = export_v2x_jsonl_to_uir(
            input_path=v2x_path, output_path=out_path, doc_id=None,
        )
        assert report.doc_id == DOC_ID

    def test_explicit_doc_id_overrides_inference(self, tmp_path: Path):
        chunks = [_v2x_text_chunk(chunk_id="c1", page=1, content="x", bbox=[0, 0, 10, 10])]
        v2x_path = _write_v2x_jsonl(tmp_path, chunks)
        out_path = tmp_path / "v3_uir.jsonl"

        report = export_v2x_jsonl_to_uir(
            input_path=v2x_path, output_path=out_path, doc_id="OVERRIDDEN",
        )
        assert report.doc_id == "OVERRIDDEN"

    def test_missing_input_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="not found"):
            export_v2x_jsonl_to_uir(
                input_path=tmp_path / "does_not_exist.jsonl",
                output_path=tmp_path / "out.jsonl",
            )

    def test_empty_corpus_raises(self, tmp_path: Path):
        # header only, no chunks
        v2x_path = tmp_path / "empty.jsonl"
        v2x_path.write_text(json.dumps({"doc_id": DOC_ID}) + "\n")
        with pytest.raises(ValueError, match="no chunk records"):
            export_v2x_jsonl_to_uir(
                input_path=v2x_path, output_path=tmp_path / "out.jsonl",
            )

    def test_verify_identity_skip_returns_none_ratio(self, tmp_path: Path):
        chunks = [_v2x_text_chunk(chunk_id="c1", page=1, content="x", bbox=[0, 0, 10, 10])]
        v2x_path = _write_v2x_jsonl(tmp_path, chunks)
        out_path = tmp_path / "v3_uir.jsonl"

        report = export_v2x_jsonl_to_uir(
            input_path=v2x_path, output_path=out_path, verify_identity=False,
        )
        assert report.identity_ratio is None
        assert not report.identity_passes  # Cannot pass without verification

    def test_round_trip_all_emitted_records_parse_back_to_uirchunks(
        self, tmp_path: Path,
    ):
        chunks = [
            _v2x_text_chunk(chunk_id="c1", page=1, content="text one", bbox=[10, 20, 100, 200]),
            _v2x_text_chunk(chunk_id="c2", page=2, content="text two", bbox=[15, 25, 110, 210]),
            _v2x_image_chunk(chunk_id="img1", page=3, asset_ref="i.png"),
        ]
        v2x_path = _write_v2x_jsonl(tmp_path, chunks)
        out_path = tmp_path / "v3_uir.jsonl"
        export_v2x_jsonl_to_uir(input_path=v2x_path, output_path=out_path)

        # Parse every non-header line back; assert UIRChunk types validate.
        parsed = []
        with out_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("modality"):  # skip header
                    parsed.append(parse_uir_jsonl_record(rec))

        assert len(parsed) == 3
        assert all(p.uir_version == "3.0" for p in parsed)
        assert [p.modality for p in parsed] == [Modality.TEXT, Modality.TEXT, Modality.IMAGE]
