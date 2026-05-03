"""Form / invoice acceptance lane in the smoke gate evaluator.

Per PLAN_V2.8 §5 (Workstream SCAN0013): scanned business forms / invoices /
receipts are first-class RAG content, not "unsupported". The smoke gate
evaluator (`scripts/evaluate_technical_manual_gates.py`) must detect a
short scanned document with no real heading hierarchy as a `form` and
skip the prose-calibrated `micro_non_label_ratio` check. Pre-fix, SCAN0013
(a 1-page German invoice) failed the smoke matrix with
`micro_non_label_ratio=0.294 > 0.22`; the failure was a probe mismatch,
not a quality regression — every chunk was correctly extracted.

Both the smoke gate AND the per-doc audit (`scripts/qa_conversion_audit.py`)
share this detection logic; the test exercises the smoke evaluator end-to-end
because that is what `bash scripts/smoke_multiprofile.sh` actually runs.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = REPO_ROOT / "scripts" / "evaluate_technical_manual_gates.py"


def _form_jsonl(tmp_path: Path) -> Path:
    """Write a synthetic SCAN0013-shape JSONL: short scanned doc, no headings."""
    out = tmp_path / "ingestion.jsonl"
    meta = {
        "object_type": "ingestion_metadata",
        "schema_version": "2.7.0",
        "profile_type": "scanned",
        "total_pages": 1,
    }
    fields = [
        "Level Automotive Farshad Khalkhali Stolberger Str 69-71 52068 Aachen",
        "NL-3971 MN Driebergen",
        "Sehr geehrte Damen und Herren;",
        "gemäß Ihrer Bestellung berechnen wir Ihnen folgenden Auftrag:",
        "Zahlungsmethode Versandart",
        "Mit freundlichen Grüßen",
        "Bankverbindung:",
        "IBAN: DE12 3456 7890",
        "BIC: ABCDEF12",
        "Rechnungsnummer: 0013",
        "Datum: 18.12.2012",
        "Betrag: EUR 250,00",
    ]

    def _chunk(i: int, text: str) -> dict:
        return {
            "chunk_id": f"form_{i:03d}",
            "doc_id": "deadbeefcafe",
            "modality": "text",
            "content": text,
            "schema_version": "2.7.0",
            "metadata": {
                "source_file": "0013_140302111325_001.pdf",
                "file_type": "pdf",
                "page_number": 1,
                "extraction_method": "ocr",
                "search_priority": "high",
                "document_modality": "scanned_degraded",
                "chunk_type": "paragraph",
                # No real parent_heading — invoice has no section hierarchy.
                # Auto-generated breadcrumb is fine; the gate must not count it.
                "hierarchy": {
                    "parent_heading": None,
                    "breadcrumb_path": ["0013_140302111325_001", "Page 1"],
                    "level": 1,
                },
            },
        }

    lines = [json.dumps(meta)]
    lines.extend(json.dumps(_chunk(i, f)) for i, f in enumerate(fields))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _run_gate(jsonl: Path) -> str:
    result = subprocess.run(
        [sys.executable, str(EVALUATOR), str(jsonl), "--doc-class", "auto"],
        capture_output=True, text=True,
    )
    return result.stdout


def test_form_short_scanned_doc_passes_via_form_lane(tmp_path):
    """SCAN0013-shape input must GATE_PASS via the `[form]` acceptance lane."""
    out = _run_gate(_form_jsonl(tmp_path))
    assert "document_type=form" in out, out
    assert "GATE_PASS [form" in out, out


def test_form_lane_is_skipped_for_long_documents(tmp_path):
    """A 50-page scanned doc must NOT be treated as a form even with no headings."""
    out_path = tmp_path / "ingestion.jsonl"
    meta = {
        "object_type": "ingestion_metadata",
        "schema_version": "2.7.0",
        "profile_type": "scanned",
        "total_pages": 50,
    }
    chunks = [
        {
            "chunk_id": f"c_{i:04d}",
            "doc_id": "longdoc12345",
            "modality": "text",
            "content": "tiny",  # micro chunk, will trip micro_non_label
            "schema_version": "2.7.0",
            "metadata": {
                "source_file": "long.pdf",
                "file_type": "pdf",
                "page_number": (i % 50) + 1,
                "extraction_method": "ocr",
                "search_priority": "high",
                "document_modality": "scanned_degraded",
                "chunk_type": "paragraph",
                "hierarchy": {"parent_heading": None, "breadcrumb_path": [], "level": 0},
            },
        }
        for i in range(20)
    ]
    out_path.write_text("\n".join([json.dumps(meta)] + [json.dumps(c) for c in chunks]) + "\n")

    out = _run_gate(out_path)
    # Long doc with no headings is not a form (page count > 5)
    assert "document_type=form" not in out
    # ...so the micro_non_label gate fires normally.
    assert "GATE_FAIL" in out and "micro_non_label_ratio" in out


def test_form_lane_is_skipped_for_digital_documents(tmp_path):
    """A 1-page DIGITAL doc must not be auto-classified as a form.

    Forms require `doc_class == scanned`. A digital one-pager (e.g. a slide
    or a generated PDF) shouldn't get the relaxed gate — it could mask a
    genuine extraction problem.
    """
    out_path = tmp_path / "ingestion.jsonl"
    meta = {
        "object_type": "ingestion_metadata",
        "schema_version": "2.7.0",
        "profile_type": "academic_whitepaper",
        "total_pages": 1,
    }
    chunks = [
        {
            "chunk_id": f"c_{i:04d}",
            "doc_id": "digonepg1234",
            "modality": "text",
            "content": "x",
            "schema_version": "2.7.0",
            "metadata": {
                "source_file": "tiny.pdf",
                "file_type": "pdf",
                "page_number": 1,
                "extraction_method": "docling",
                "search_priority": "high",
                "document_modality": "native_digital",
                "chunk_type": "paragraph",
                "hierarchy": {"parent_heading": None, "breadcrumb_path": [], "level": 0},
            },
        }
        for i in range(20)
    ]
    out_path.write_text("\n".join([json.dumps(meta)] + [json.dumps(c) for c in chunks]) + "\n")

    out = _run_gate(out_path)
    assert "document_type=form" not in out
