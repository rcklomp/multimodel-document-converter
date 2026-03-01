"""
Tests for v2.6 Gap A/B/C changes:

Gap A — IngestionMetadata model and JSONL first-line guarantee
Gap B — VLM failure sentinel differentiation ([VLM_FAILED: ...])
Gap C — QA script metadata-record filter
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from mmrag_v2.schema.ingestion_schema import IngestionMetadata


# ---------------------------------------------------------------------------
# Gap A: IngestionMetadata model
# ---------------------------------------------------------------------------


def test_ingestion_metadata_object_type():
    """object_type must be the literal 'ingestion_metadata'."""
    m = IngestionMetadata(
        schema_version="2.6.0",
        doc_id="abc123def456",
        source_file="test.pdf",
    )
    assert m.object_type == "ingestion_metadata"


def test_ingestion_metadata_serialization_keys():
    """Serialized dict must contain all required keys at root level."""
    m = IngestionMetadata(
        schema_version="2.6.0",
        doc_id="abc123def456",
        source_file="test.pdf",
        profile_type="technical_manual",
        domain="technical",
        is_scan=False,
        total_pages=20,
        chunk_count=350,
        ingestion_timestamp="2026-03-01T00:00:00+00:00",
    )
    d = m.model_dump(mode="json")
    assert d["object_type"] == "ingestion_metadata"
    assert d["schema_version"] == "2.6.0"
    assert d["doc_id"] == "abc123def456"
    assert d["source_file"] == "test.pdf"
    assert d["profile_type"] == "technical_manual"
    assert d["domain"] == "technical"
    assert d["is_scan"] is False
    assert d["total_pages"] == 20
    assert d["chunk_count"] == 350


def test_ingestion_metadata_optional_fields_default_none():
    """Optional fields that are not set must serialize as null/None."""
    m = IngestionMetadata(
        schema_version="2.6.0",
        doc_id="abc123def456",
        source_file="test.pdf",
    )
    d = m.model_dump(mode="json")
    for optional_key in (
        "profile_type",
        "document_type",
        "domain",
        "is_scan",
        "total_pages",
        "image_density",
        "avg_text_per_page",
        "has_flat_text_corruption",
        "has_encoding_corruption",
        "chunk_count",
    ):
        assert optional_key in d, f"Missing key: {optional_key}"
        assert d[optional_key] is None, f"{optional_key} should be None when unset"


def test_ingestion_metadata_json_round_trip():
    """JSON serialization must produce a valid JSON line that round-trips cleanly."""
    m = IngestionMetadata(
        schema_version="2.6.0",
        doc_id="deadbeef1234",
        source_file="my document.pdf",
        profile_type="technical_manual",
        has_flat_text_corruption=True,
        has_encoding_corruption=False,
        chunk_count=42,
        ingestion_timestamp="2026-03-01T12:34:56+00:00",
    )
    json_line = json.dumps(m.model_dump(mode="json"), ensure_ascii=False)
    decoded = json.loads(json_line)
    assert decoded["object_type"] == "ingestion_metadata"
    assert decoded["doc_id"] == "deadbeef1234"
    assert decoded["has_flat_text_corruption"] is True
    assert decoded["has_encoding_corruption"] is False


# ---------------------------------------------------------------------------
# Gap A: JSONL first-line guarantee (synthetic JSONL test)
# ---------------------------------------------------------------------------


def _make_synthetic_jsonl(tmp_path: Path) -> Path:
    """Write a synthetic JSONL that mimics v2.6+ output."""
    out = tmp_path / "ingestion.jsonl"
    meta = IngestionMetadata(
        schema_version="2.6.0",
        doc_id="deadbeef1234",
        source_file="test.pdf",
        profile_type="technical_manual",
        chunk_count=2,
        ingestion_timestamp="2026-03-01T00:00:00+00:00",
    )
    chunk1 = {
        "object_type": None,
        "chunk_id": "chunk_001",
        "doc_id": "deadbeef1234",
        "modality": "text",
        "content": "Introduction paragraph.",
        "metadata": {"page_number": 1},
    }
    chunk2 = {
        "object_type": None,
        "chunk_id": "chunk_002",
        "doc_id": "deadbeef1234",
        "modality": "text",
        "content": "Second paragraph.",
        "metadata": {"page_number": 1},
    }
    with out.open("w", encoding="utf-8") as f:
        f.write(json.dumps(meta.model_dump(mode="json")) + "\n")
        f.write(json.dumps(chunk1) + "\n")
        f.write(json.dumps(chunk2) + "\n")
    return out


def test_first_line_is_metadata_record(tmp_path):
    """First line of JSONL must be the ingestion_metadata record."""
    jsonl = _make_synthetic_jsonl(tmp_path)
    with jsonl.open("r", encoding="utf-8") as f:
        first = json.loads(f.readline())
    assert first.get("object_type") == "ingestion_metadata"
    assert first.get("profile_type") == "technical_manual"


def test_metadata_record_filter_excludes_from_chunk_count(tmp_path):
    """After filtering, metadata record must not appear in the chunk list."""
    jsonl = _make_synthetic_jsonl(tmp_path)
    rows = []
    with jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    # 3 lines total (1 metadata + 2 chunks)
    assert len(rows) == 3

    # Apply the QA script filter
    chunks = [r for r in rows if r.get("object_type") != "ingestion_metadata"]
    assert len(chunks) == 2
    for c in chunks:
        assert c.get("object_type") != "ingestion_metadata"


# ---------------------------------------------------------------------------
# Gap B: VLM failure sentinel
# ---------------------------------------------------------------------------


def test_vlm_failed_sentinel_is_placeholder():
    """[VLM_FAILED: ...] sentinel must be treated as a placeholder by QA."""
    # Import the QA helper (it's a standalone script, import path via sys.path)
    script_dir = Path(__file__).parent.parent / "scripts"
    sys.path.insert(0, str(script_dir))
    try:
        from qa_semantic_fidelity import is_placeholder_image_or_table
    finally:
        sys.path.pop(0)

    assert is_placeholder_image_or_table("[VLM_FAILED: call error]") is True
    assert is_placeholder_image_or_table("[VLM_FAILED: response invalid]") is True
    assert is_placeholder_image_or_table("[VLM_FAILED: parse error]") is True


def test_no_vlm_placeholder_still_detected():
    """No-VLM path placeholder [Figure on page N] must still be detected."""
    script_dir = Path(__file__).parent.parent / "scripts"
    sys.path.insert(0, str(script_dir))
    try:
        from qa_semantic_fidelity import is_placeholder_image_or_table
    finally:
        sys.path.pop(0)

    assert is_placeholder_image_or_table("[Figure on page 3]") is True
    assert is_placeholder_image_or_table("[Image on page 7]") is True


def test_real_vlm_description_not_placeholder():
    """Genuine VLM descriptions must NOT be flagged as placeholders."""
    script_dir = Path(__file__).parent.parent / "scripts"
    sys.path.insert(0, str(script_dir))
    try:
        from qa_semantic_fidelity import is_placeholder_image_or_table
    finally:
        sys.path.pop(0)

    assert is_placeholder_image_or_table(
        "A photograph showing a solar panel array mounted on a rooftop with clear blue sky in the background."
    ) is False
    assert is_placeholder_image_or_table(
        "Circuit schematic showing transistor amplifier stage with input and output coupling capacitors."
    ) is False


def test_old_sentinel_not_present_in_vision_manager():
    """The legacy 'Unverified visual element.' string must not appear in vision_manager.py."""
    vm_path = Path(__file__).parent.parent / "src" / "mmrag_v2" / "vision" / "vision_manager.py"
    text = vm_path.read_text(encoding="utf-8")
    assert "Unverified visual element." not in text, (
        "Legacy sentinel found in vision_manager.py — should be [VLM_FAILED: ...] variants"
    )
