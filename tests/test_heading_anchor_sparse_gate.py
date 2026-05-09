"""Phase 4 Step 4 Path A — anchor-sparse HEADING gate regression tests.

Pins the profile-scoped relaxation: `scanned` and `digital_magazine`
documents with `unique_headings / text_chunks <= 0.05` get a
HEADING coverage floor of 0.70 instead of the default 0.80. Strict
gate stays unchanged for `technical_manual`, `academic_whitepaper`,
and other non-sparse profiles.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_AUDIT = _REPO_ROOT / "scripts" / "qa_conversion_audit.py"


def _build_doc(
    profile_type: str,
    unique_headings: int,
    text_chunks_with_heading: int,
    text_chunks_without_heading: int,
) -> list[dict]:
    """Build a synthetic document header + text chunks for the audit script."""
    total = text_chunks_with_heading + text_chunks_without_heading
    header = {
        "object_type": "ingestion_metadata",
        "schema_version": "2.7.0",
        "doc_id": "deadbeef1234",
        "source_file": "synthetic.pdf",
        "profile_type": profile_type,
        "document_type": "synthetic",
        "domain": "test",
        "is_scan": profile_type == "scanned",
        "total_pages": 100,
        "image_density": 0.0,
        "avg_text_per_page": 200.0,
        "has_flat_text_corruption": False,
        "has_encoding_corruption": False,
        "chunk_count": total,
        "ingestion_timestamp": "2026-05-09T22:00:00Z",
        "pipeline_version": "2.9.0-dev",
        "source_file_hash": "deadbeef",
        "config_hash": "deadbeef",
    }
    chunks: list[dict] = [header]
    # Spread `unique_headings` distinct heading values across the
    # heading-attributed chunks. Each chunk's parent_heading is one of
    # h0…h(unique_headings-1).
    for i in range(text_chunks_with_heading):
        h_idx = i % max(unique_headings, 1)
        chunks.append({
            "chunk_id": f"deadbeef1234_001_text_{i:04x}",
            "doc_id": "deadbeef1234",
            "modality": "text",
            "content": f"paragraph {i} body content",
            "metadata": {
                "source_file": "synthetic.pdf",
                "file_type": "pdf",
                "page_number": 1 + (i // 4),
                "chunk_type": "paragraph",
                "hierarchy": {
                    "parent_heading": f"Section {h_idx}",
                    "breadcrumb_path": ["Document", f"Section {h_idx}"],
                    "level": 2,
                },
                "spatial": None,
                "extraction_method": "synthetic",
                "schema_version": "2.7.0",
            },
            "asset_ref": None,
            "semantic_context": None,
            "schema_version": "2.7.0",
            "contextualized_text": None,
            "visual_description": None,
        })
    for i in range(text_chunks_without_heading):
        chunks.append({
            "chunk_id": f"deadbeef1234_002_text_{i:04x}",
            "doc_id": "deadbeef1234",
            "modality": "text",
            "content": f"orphan paragraph {i} body content",
            "metadata": {
                "source_file": "synthetic.pdf",
                "file_type": "pdf",
                "page_number": 50 + (i // 4),
                "chunk_type": "paragraph",
                "hierarchy": {
                    "parent_heading": None,
                    "breadcrumb_path": ["Document"],
                    "level": 1,
                },
                "spatial": None,
                "extraction_method": "synthetic",
                "schema_version": "2.7.0",
            },
            "asset_ref": None,
            "semantic_context": None,
            "schema_version": "2.7.0",
            "contextualized_text": None,
            "visual_description": None,
        })
    return chunks


def _run_audit(jsonl: Path) -> tuple[str, int]:
    result = subprocess.run(
        [sys.executable, str(_AUDIT), str(jsonl)],
        capture_output=True, text=True,
    )
    return result.stdout, result.returncode


def _write_jsonl(tmp_path: Path, records: list[dict]) -> Path:
    fp = tmp_path / "ingestion.jsonl"
    fp.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )
    return fp


# ---------------------------------------------------------------------------
# Sparse-eligible profiles: HEADING floor relaxes to 0.70
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("profile_type", ["scanned", "digital_magazine"])
def test_anchor_sparse_eligible_profile_passes_at_72_percent(
    profile_type: str, tmp_path: Path
) -> None:
    """Firearms-shape: 31 headings / 1094 chunks = 0.028 sparseness, 72 % coverage."""
    chunks = _build_doc(
        profile_type=profile_type,
        unique_headings=31,
        text_chunks_with_heading=790,
        text_chunks_without_heading=304,  # 1094 total, 72 % coverage
    )
    fp = _write_jsonl(tmp_path, chunks)
    stdout, _ = _run_audit(fp)
    assert "anchor-sparse, floor=0.70" in stdout, stdout
    # Fragment of the HEADING block — must be PASS not FAIL
    heading_block = stdout.split("HEADING:", 1)[1].splitlines()[0]
    assert "PASS" in heading_block, f"HEADING line: {heading_block!r}"


# ---------------------------------------------------------------------------
# Non-sparse profile: strict 0.80 floor enforced
# ---------------------------------------------------------------------------


def test_technical_manual_keeps_strict_80_percent_floor(tmp_path: Path) -> None:
    chunks = _build_doc(
        profile_type="technical_manual",
        unique_headings=31,  # would be sparse (0.028) but profile keeps strict gate
        text_chunks_with_heading=790,
        text_chunks_without_heading=304,
    )
    fp = _write_jsonl(tmp_path, chunks)
    stdout, _ = _run_audit(fp)
    assert "anchor-sparse" not in stdout
    heading_block = stdout.split("HEADING:", 1)[1].splitlines()[0]
    assert "FAIL" in heading_block, f"HEADING line: {heading_block!r}"


# ---------------------------------------------------------------------------
# Sparse-eligible profile but heading-rich: strict 0.80 still applies (>0.05 sparse_ratio)
# ---------------------------------------------------------------------------


def test_scanned_profile_with_dense_headings_keeps_strict_floor(tmp_path: Path) -> None:
    """A scanned doc with many distinct headings is NOT anchor-sparse."""
    # 200 unique headings / 1000 chunks = 0.20 — far above the 0.05 sparseness gate
    chunks = _build_doc(
        profile_type="scanned",
        unique_headings=200,
        text_chunks_with_heading=750,  # 75% coverage — would pass the 0.70 floor
        text_chunks_without_heading=250,
    )
    fp = _write_jsonl(tmp_path, chunks)
    stdout, _ = _run_audit(fp)
    assert "anchor-sparse" not in stdout, stdout
    heading_block = stdout.split("HEADING:", 1)[1].splitlines()[0]
    # 75 % is below the strict 0.80 floor
    assert "FAIL" in heading_block, f"HEADING line: {heading_block!r}"


# ---------------------------------------------------------------------------
# Sparse-eligible profile and sparse but coverage below 0.70: still fails
# ---------------------------------------------------------------------------


def test_anchor_sparse_below_70_percent_still_fails(tmp_path: Path) -> None:
    chunks = _build_doc(
        profile_type="scanned",
        unique_headings=20,  # sparse
        text_chunks_with_heading=600,  # 60 % coverage — below the 0.70 floor
        text_chunks_without_heading=400,
    )
    fp = _write_jsonl(tmp_path, chunks)
    stdout, _ = _run_audit(fp)
    assert "anchor-sparse, floor=0.70" in stdout
    heading_block = stdout.split("HEADING:", 1)[1].splitlines()[0]
    assert "FAIL" in heading_block, f"HEADING line: {heading_block!r}"
