#!/usr/bin/env python3
"""Seeded-fault sensitivity suite (PLAN_EXTRACTION_FIDELITY_V1 Section 7.3).

Entry-gate INSTRUMENT VALIDATION for Phase 1. The failure class this plan most
fears is CONTENT OMISSION (dropped diagram labels, flattened table cells, stripped
code indentation, reordered columns). Before either selection instrument may decide
anything, we must know whether it actually MOVES when such a fault is injected -
an instrument that does not move is BLIND for that class and any verdict resting on
it there is QUALITATIVE, not measured.

This harness injects each named fault into representative content and runs it
through the instruments that EXIST tonight, producing a BLINDNESS REPORT
(fault class x instrument -> MOVED / BLIND / NOT-RUN, with the numeric delta).

Instruments
-----------
  text_edit_distance   OmniDocBench text-metric kernel: the scorer's exact
                       ``clean_string`` normalization + normalized Levenshtein.
                       The normalization deletes ALL whitespace, so this is the
                       canonical way to expose its indentation-blindness. (The
                       real scorer adds block matching on top; the normalization
                       finding is identical either way - clean_string runs
                       per-block.) Library-light, in-process (Section 7.3).
  table_teds           OmniDocBench's real ``TEDS.evaluate(pred, true)`` on HTML
                       tables, run in the ``omnidocbench`` conda env. Structure
                       metric for the two table faults. NOT-RUN if the env/repo
                       is absent.
  gate_quality_signals The PLAN_GATE_QUALITY_V1 advisory signals that are BUILT
                       (qa_semantic_fidelity: running-furniture, cross-page
                       dupes, insane-headings). These are junk-PRESENCE detectors
                       and CANNOT, by construction, see content ABSENCE - the
                       suite proves it rather than assuming it. Image signals
                       (non-visual / blank) are N/A to text/table faults.

Usage
-----
  python scripts/seeded_fault_sensitivity.py report [--json OUT] [--md OUT]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HOME = Path.home()
ODB_REPO = HOME / "omnidocbench-eval" / "OmniDocBench"
ODB_PY = HOME / "miniforge3/envs/omnidocbench/bin/python"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

# MOVED iff the instrument's delta exceeds this; below it the instrument did not
# react to the injected fault and is recorded BLIND for that class.
_TEXT_ED_MOVE_EPS = 0.005
_TEDS_MOVE_EPS = 0.005


# --------------------------------------------------------------------------- #
# Representative fixtures (instrument-validation inputs, NOT real extractions -
# kept deterministic so the blindness verdicts are stable regression fixtures).
# --------------------------------------------------------------------------- #
CODE_BLOCK = (
    "def discharge(safety):\n"
    "    if safety:\n"
    "        return None\n"
    "    for cell in cells:\n"
    "        cell.fire()\n"
)

LABELED_CAPTION = (
    "Figure 3. Battery management system wiring.\n"
    "BMS-04 connects to the J17 header via the orange CAN-H lead.\n"
)
SMALL_LABEL = "BMS-04"

HTML_TABLE = (
    "<table>"
    "<tr><td>Cell</td><td>Voltage</td><td>Temp</td></tr>"
    "<tr><td>A1</td><td>3.7</td><td>24</td></tr>"
    "<tr><td>A2</td><td>3.6</td><td>25</td></tr>"
    "</table>"
)


# --------------------------------------------------------------------------- #
# Fault generators (deterministic)
# --------------------------------------------------------------------------- #
def strip_code_indentation(text: str) -> str:
    """Remove leading whitespace from every line (the R3 fidelity-loss class)."""
    return "\n".join(line.lstrip() for line in text.splitlines()) + (
        "\n" if text.endswith("\n") else ""
    )


def drop_small_label(text: str, label: str) -> str:
    """Delete one small label token (the dropped wiring-diagram-label class)."""
    return text.replace(label, "", 1)


def flatten_table_to_prose(html_table: str) -> str:
    """Collapse an HTML table to space-joined prose - all structure lost."""
    cells = re.findall(r"<td>(.*?)</td>", html_table)
    return " ".join(cells)


def reorder_two_columns(html_table: str) -> str:
    """Swap the first two columns of every row (column-reorder class)."""
    rows = re.findall(r"<tr>(.*?)</tr>", html_table)
    out = []
    for row in rows:
        cells = re.findall(r"<td>(.*?)</td>", row)
        if len(cells) >= 2:
            cells[0], cells[1] = cells[1], cells[0]
        out.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return "<table>" + "".join(out) + "</table>"


# --------------------------------------------------------------------------- #
# Instrument 1: OmniDocBench text-metric kernel (exact normalization)
# --------------------------------------------------------------------------- #
def _clean_string(s: str) -> str:
    """Replicates OmniDocBench text_postprocess.clean_string (the text-metric
    normalization). KEY: it deletes ALL whitespace, so indentation loss is
    invisible to the scored text edit distance. Kept in sync with the scorer."""
    s = str(s or "")
    for a, b in [
        ("\\t", ""), ("\\n", ""), ("\t", ""), ("\n", ""), ("/t", ""),
        ("/n", ""), (" ", ""), ("✓", "✔"), ("√", "✔"),
        ("-", "—"), ("∼", "～"), ("Ø", "∅"),
    ]:
        s = s.replace(a, b)
    s = re.sub(r"_{4,}", "____", s)
    s = re.sub(r" {4,}", "    ", s)
    return s


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def text_edit_distance(original: str, mutated: str) -> float:
    """Scorer-normalized text edit distance between original and mutated."""
    co, cm = _clean_string(original), _clean_string(mutated)
    return _levenshtein(cm, co) / max(len(cm), len(co), 1)


# --------------------------------------------------------------------------- #
# Instrument 2: OmniDocBench TEDS (real, omnidocbench env)
# --------------------------------------------------------------------------- #
def table_teds(pred_html: str, true_html: str) -> float | None:
    """Real TEDS.evaluate via the omnidocbench env. None if unavailable."""
    if not ODB_PY.exists() or not ODB_REPO.exists():
        return None
    # TEDS.evaluate requires a `body/table` document structure (it xpaths
    # `body/table`); wrap bare <table> fragments. A non-table mutation (flattened
    # prose) has no body/table and correctly scores 0.0 (structure destroyed).
    def _wrap(h: str) -> str:
        return f"<html><body>{h}</body></html>"

    code = (
        "import sys; sys.path.insert(0, 'src')\n"
        "from metrics.table_metric import TEDS\n"
        "import json\n"
        f"pred = {_wrap(pred_html)!r}\n"
        f"true = {_wrap(true_html)!r}\n"
        "print(json.dumps(TEDS().evaluate(pred, true)))\n"
    )
    try:
        proc = subprocess.run(
            [str(ODB_PY), "-c", code], cwd=str(ODB_REPO),
            capture_output=True, text=True, timeout=120,
        )
    except Exception:  # noqa: BLE001 - probe; any failure -> NOT-RUN
        return None
    if proc.returncode != 0:
        return None
    try:
        return float(json.loads(proc.stdout.strip().splitlines()[-1]))
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Instrument 3: PLAN_GATE_QUALITY_V1 advisory signals (the BUILT ones)
# --------------------------------------------------------------------------- #
def gate_quality_signal_vector(text: str) -> dict:
    """Run the BUILT junk-presence signals on a single-page text representation.

    Returns the signal counts. These detect junk PRESENCE (furniture, dupes,
    garbled headings) and are structurally blind to content ABSENCE - this
    function makes that measurable rather than asserted. Image signals
    (non-visual / blank) need an asset on disk and are N/A here.
    """
    from qa_semantic_fidelity import (
        count_cross_page_dupes,
        count_insane_headings,
        count_running_furniture,
    )

    chunk = {"content": text, "modality": "text", "metadata": {"page_number": 1}}
    texts = [chunk]
    return {
        "running_furniture": count_running_furniture(texts),
        "cross_page_dupes": count_cross_page_dupes(texts),
        "insane_headings": count_insane_headings(texts),
    }


# --------------------------------------------------------------------------- #
# Blindness matrix
# --------------------------------------------------------------------------- #
def _verdict(delta: float | None, eps: float) -> str:
    if delta is None:
        return "NOT-RUN"
    return "MOVED" if delta > eps else "BLIND"


def build_blindness_report() -> dict:
    """Run every fault x applicable instrument and return the matrix."""
    faults = []

    # 1) strip code indentation -- text-ED + gate signals (TEDS N/A: not a table)
    orig, mut = CODE_BLOCK, strip_code_indentation(CODE_BLOCK)
    faults.append(_row("strip_code_indentation", orig, mut, table=False))

    # 2) drop a small label -- text-ED + gate signals
    orig, mut = LABELED_CAPTION, drop_small_label(LABELED_CAPTION, SMALL_LABEL)
    faults.append(_row("drop_small_label", orig, mut, table=False))

    # 3) flatten a table to prose -- text-ED + TEDS + gate signals
    orig, mut = HTML_TABLE, flatten_table_to_prose(HTML_TABLE)
    faults.append(_row("flatten_table_to_prose", orig, mut, table=True))

    # 4) reorder two columns -- text-ED + TEDS + gate signals
    orig, mut = HTML_TABLE, reorder_two_columns(HTML_TABLE)
    faults.append(_row("reorder_two_columns", orig, mut, table=True))

    return {"faults": faults}


def _row(name: str, original: str, mutated: str, *, table: bool) -> dict:
    ted = text_edit_distance(original, mutated)
    teds = table_teds(mutated, original) if table else None
    # TEDS=1.0 means identical structure; the "delta" that must exceed eps is
    # (1 - TEDS) so a structure change reads as MOVED.
    teds_delta = (1.0 - teds) if teds is not None else None
    sig_o = gate_quality_signal_vector(original)
    sig_m = gate_quality_signal_vector(mutated)
    sig_moved = sig_o != sig_m
    return {
        "fault": name,
        "text_edit_distance": {
            "delta": round(ted, 6),
            "verdict": _verdict(ted, _TEXT_ED_MOVE_EPS),
        },
        "table_teds": {
            "teds": (round(teds, 6) if teds is not None else None),
            "delta": (round(teds_delta, 6) if teds_delta is not None else None),
            "verdict": ("N/A" if not table else _verdict(teds_delta, _TEDS_MOVE_EPS)),
        },
        "gate_quality_signals": {
            "original": sig_o,
            "mutated": sig_m,
            "verdict": "MOVED" if sig_moved else "BLIND",
        },
    }


def render_markdown(report: dict) -> str:
    lines = [
        "## Seeded-fault BLINDNESS REPORT (Section 7.3)",
        "",
        "fault class x instrument -> MOVED (reacts) / BLIND (does not) / NOT-RUN / N/A.",
        "An instrument BLIND to a fault class CANNOT decide it; a Phase 1/2 verdict",
        "resting on it there is QUALITATIVE (Section 7.3).",
        "",
        "| fault class | text-ED (OmniDocBench) | table-TEDS | gate junk-presence signals |",
        "|---|---|---|---|",
    ]
    for f in report["faults"]:
        ted = f["text_edit_distance"]
        teds = f["table_teds"]
        sig = f["gate_quality_signals"]
        ted_cell = f"{ted['verdict']} (d={ted['delta']})"
        if teds["verdict"] == "N/A":
            teds_cell = "N/A"
        elif teds["verdict"] == "NOT-RUN":
            teds_cell = "NOT-RUN (env absent)"
        else:
            teds_cell = f"{teds['verdict']} (TEDS={teds['teds']})"
        sig_cell = sig["verdict"]
        lines.append(f"| {f['fault']} | {ted_cell} | {teds_cell} | {sig_cell} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("report")
    p.add_argument("--json", help="write the raw blindness matrix JSON here")
    p.add_argument("--md", help="write the rendered markdown table here")
    args = ap.parse_args(argv)

    report = build_blindness_report()
    md = render_markdown(report)
    print(md)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\n-> {args.json}")
    if args.md:
        Path(args.md).write_text(md)
        print(f"-> {args.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
