"""R3 code-indentation gate behavior in qa_conversion_audit.py (Policy B).

Drives the audit script via subprocess on synthetic JSONL (same pattern as
tests/test_tabular_audit_gate.py) to pin the redesigned gate end-to-end:

  - mangled judgeable code ABOVE the density floor -> AUDIT_FAIL (CODE)
  - equations mislabelled as code -> excluded, CODE never fails (the false-fail
    class)
  - well-indented code -> CODE PASS
  - a couple of mangled snippets BELOW the density floor -> CODE WARN, not FAIL
    (a prose doc is not discarded over incidental code; F3 / Policy B)

These are executable requirements for docs/PLAN_R3_CODE_GATE_REDESIGN.md and
docs/DECISIONS.md "R3 Code-Indentation Gate Redesign". See also the unit-level
contracts in tests/test_code_quality_metric.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT = REPO_ROOT / "scripts" / "qa_conversion_audit.py"
_VER = "2.7.0"

_PROSE = (
    "The history of the automobile begins in the late nineteenth century, when "
    "inventors across Europe converged on the internal combustion engine as a "
    "practical means of propulsion. Over the following decades the design matured."
)

# Multi-line Python suite whose body indentation was stripped (every line at
# column 0): judgeable (block opener) and indentation-degraded -> FAIL.
_MANGLED = "class Worker:\ndef __init__(self, q):\nself.q = q\ndef run(self):\nself.go()"
# Correctly nested suite -> PASS.
_GOOD = (
    "class Worker:\n    def __init__(self, q):\n        self.q = q\n"
    "    def run(self):\n        self.go()"
)
# Equation an extractor VLM mislabels as code -> no code structure -> excluded.
_EQUATION = "V_oc = a + b × DOD + (c + d × DOD)T"


def _meta(total_pages: int = 12, profile: str = "academic_whitepaper") -> dict:
    return {
        "object_type": "ingestion_metadata",
        "schema_version": _VER,
        "pipeline_version": _VER,
        "source_file": "doc.pdf",
        "source_file_hash": "deadbeefcafe",
        "profile_type": profile,
        "total_pages": total_pages,
    }


def _text(i: int) -> dict:
    return {
        "chunk_id": f"t_{i:04d}",
        "doc_id": "doc123456789",
        "modality": "text",
        "content": _PROSE,
        "schema_version": _VER,
        "metadata": {
            "source_file": "doc.pdf",
            "page_number": (i % 12) + 1,
            "chunk_type": "paragraph",
            # Carry a heading so the HEADING gate passes and CODE is isolated.
            "hierarchy": {
                "parent_heading": "Section 1 Introduction",
                "breadcrumb_path": ["Doc", "Section 1"],
                "level": 1,
            },
        },
    }


def _code(i: int, content: str) -> dict:
    return {
        "chunk_id": f"c_{i:04d}",
        "doc_id": "doc123456789",
        "modality": "code",
        "content": content,
        "schema_version": _VER,
        "metadata": {
            "source_file": "doc.pdf",
            "page_number": (i % 12) + 1,
            "chunk_type": "code",
            "hierarchy": {
                "parent_heading": "Section 1 Introduction",
                "breadcrumb_path": ["Doc", "Section 1"],
                "level": 1,
            },
        },
    }


def _write(tmp_path: Path, meta: dict, chunks: list) -> Path:
    out = tmp_path / "ingestion.jsonl"
    out.write_text("\n".join([json.dumps(meta)] + [json.dumps(c) for c in chunks]) + "\n")
    return out


def _run_audit(jsonl: Path) -> str:
    res = subprocess.run([sys.executable, str(AUDIT), str(jsonl)], capture_output=True, text=True)
    return res.stdout


def test_mangled_code_above_density_hard_fails(tmp_path):
    # 8 mangled judgeable blocks + 20 prose -> density 8/28 = 0.29 >> floor.
    chunks = [_code(i, _MANGLED) for i in range(8)] + [_text(i) for i in range(20)]
    out = _run_audit(_write(tmp_path, _meta(), chunks))
    assert "CODE:        FAIL" in out, out
    assert "AUDIT_FAIL (CODE" in out, out
    assert "indentation_fidelity: 0.00" in out, out


def test_equations_mislabelled_as_code_do_not_fail(tmp_path):
    # The false-fail class: 8 equation chunks typed as code -> excluded.
    chunks = [_code(i, _EQUATION) for i in range(8)] + [_text(i) for i in range(20)]
    out = _run_audit(_write(tmp_path, _meta(), chunks))
    assert "math_excluded: 8" in out, out
    assert "CODE:        PASS" in out, out
    assert "(CODE" not in out, out


def test_well_indented_code_passes(tmp_path):
    chunks = [_code(i, _GOOD) for i in range(8)] + [_text(i) for i in range(20)]
    out = _run_audit(_write(tmp_path, _meta(), chunks))
    assert "CODE:        PASS" in out, out
    assert "indentation_fidelity: 1.00" in out, out
    assert "(CODE" not in out, out


def test_incidental_mangled_code_below_density_warns(tmp_path):
    # 3 mangled judgeable blocks (>= min_judgeable) in a 93-chunk prose doc ->
    # density 3/93 = 0.032 < 0.04 floor -> WARN + per-chunk flag, NOT a hard fail.
    chunks = [_code(i, _MANGLED) for i in range(3)] + [_text(i) for i in range(90)]
    out = _run_audit(_write(tmp_path, _meta(), chunks))
    assert "CODE:        WARN" in out, out
    assert "AUDIT_FAIL (CODE" not in out, out
    assert "degraded code indentation (advisory)" in out, out
