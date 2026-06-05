"""Tabular-document HEADING-gate skip in qa_conversion_audit.py.

A table-dominant born-digital document (data spreadsheet, parts/price export,
inventory count) has no chapter hierarchy by nature — like a form — but can run
longer than the 5-page `short_document` bound, so it used to fall into `book`
and fail the >= 0.80 HEADING gate spuriously (canonical example:
`data/data_spreadsheet/CarOK voorraadtelling 2021-04.pdf`, 0/37 coverage ->
HEADING FAIL, every chunk correctly extracted).

These tests pin the gate-correctness fix AND its non-leak guard: the skip
requires a TABLE-chunk share a prose document never reaches, so a prose book
with missed headings still FAILS. See docs/DECISIONS.md "Tabular-Document
HEADING-Gate Skip".
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT = REPO_ROOT / "scripts" / "qa_conversion_audit.py"
_VER = "2.7.0"


def _meta(total_pages: int, profile: str = "technical_manual") -> dict:
    return {
        "object_type": "ingestion_metadata",
        "schema_version": _VER,
        "pipeline_version": _VER,
        "source_file": "doc.pdf",
        "source_file_hash": "deadbeefcafe",
        "profile_type": profile,
        "total_pages": total_pages,
    }


def _text(i: int, content: str, heading: str | None) -> dict:
    return {
        "chunk_id": f"t_{i:04d}",
        "doc_id": "doc123456789",
        "modality": "text",
        "content": content,
        "schema_version": _VER,
        "metadata": {
            "source_file": "doc.pdf",
            "page_number": (i % 12) + 1,
            "chunk_type": "paragraph",
            "hierarchy": {"parent_heading": heading, "breadcrumb_path": [], "level": 1},
        },
    }


def _table(i: int) -> dict:
    return {
        "chunk_id": f"tb_{i:04d}",
        "doc_id": "doc123456789",
        "modality": "table",
        "content": "| aant | merk | prijs |\n| --- | --- | --- |\n| 1 | Castrol | 6,55 |",
        "schema_version": _VER,
        "metadata": {
            "source_file": "doc.pdf",
            "page_number": (i % 12) + 1,
            "chunk_type": "table",
            "hierarchy": {"parent_heading": None, "breadcrumb_path": [], "level": 0},
        },
    }


def _write(tmp_path: Path, meta: dict, chunks: list) -> Path:
    out = tmp_path / "ingestion.jsonl"
    out.write_text("\n".join([json.dumps(meta)] + [json.dumps(c) for c in chunks]) + "\n")
    return out


def _run_audit(jsonl: Path) -> str:
    res = subprocess.run([sys.executable, str(AUDIT), str(jsonl)], capture_output=True, text=True)
    return res.stdout


# Realistic data rows from the CarOK voorraadtelling shape (price + product).
_DATA_ROWS = [
    "6,55 Castrol Edge FST 0W-30 Motorolie, 1 liter",
    "1,00 Remblokkenset vooras Peugeot 205, 309, 405, Renault 5, 19, 21",
    "7,55 TwinPower Turbo Longlife-01 5W-30, 1 liter",
    "3,99 Interieurfilter Volkswagen Golf, Passat, Audi A4",
]


def test_long_table_dominant_doc_skips_heading_gate(tmp_path):
    """12-page data spreadsheet: many tables, 0 headings -> HEADING SKIP [tabular]."""
    chunks = [_table(i) for i in range(12)]
    chunks += [_text(i, _DATA_ROWS[i % len(_DATA_ROWS)], heading=None) for i in range(20)]
    out = _run_audit(_write(tmp_path, _meta(12), chunks))
    assert "SKIP [tabular_document" in out, out
    assert "AUDIT_FAIL (HEADING" not in out, out


def test_prose_book_with_missing_headings_still_fails(tmp_path):
    """NON-LEAK GUARD: a long prose doc with 0 headings and NO tables must NOT be
    exempted — the HEADING gate must still FAIL (catches a real heading regression)."""
    prose = (
        "The history of the automobile begins in the late nineteenth century, when "
        "inventors across Europe converged on the internal combustion engine as a "
        "practical means of propulsion. Over the following decades the design matured."
    )
    chunks = [_text(i, prose, heading=None) for i in range(30)]  # 0 tables -> table_share 0
    out = _run_audit(_write(tmp_path, _meta(12), chunks))
    assert "SKIP [tabular_document" not in out, out
    assert "AUDIT_FAIL (HEADING" in out, out


def test_table_heavy_doc_with_headings_is_not_tabular(tmp_path):
    """A doc that HAS headings (>= 0.10 coverage) is never tabular even with many
    tables — the HEADING gate is evaluated, not skipped."""
    chunks = [_table(i) for i in range(12)]
    # All text chunks carry a real heading -> coverage 1.0, well above 0.10.
    chunks += [
        _text(i, _DATA_ROWS[i % len(_DATA_ROWS)], heading="Section 3 Lubricants") for i in range(20)
    ]
    out = _run_audit(_write(tmp_path, _meta(12), chunks))
    assert "SKIP [tabular_document" not in out, out


def test_sparse_tables_below_share_threshold_not_tabular(tmp_path):
    """A mostly-prose doc with a couple of incidental tables (< 0.20 share) and 0
    headings is NOT tabular — the gate still applies (guards the threshold)."""
    chunks = [_table(i) for i in range(2)]  # only 2 tables
    chunks += [
        _text(i, _DATA_ROWS[i % len(_DATA_ROWS)], heading=None) for i in range(40)
    ]  # share 2/42
    out = _run_audit(_write(tmp_path, _meta(12), chunks))
    assert "SKIP [tabular_document" not in out, out
    assert "AUDIT_FAIL (HEADING" in out, out
