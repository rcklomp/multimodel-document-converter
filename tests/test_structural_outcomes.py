"""PLAN_QUALITY_REMEDIATION_V1 WP-0.2: structural regression-guard metrics.

Each metric has a fixture that FIRES on the defect and stays QUIET on a clean shape
(AGENT-GATE-PROGRESSION). The "fix-induced fault" tests record what each metric can and
cannot see when a *fix* misbehaves; a metric that cannot see a fault is declared
KNOWN-BLIND here and must never be re-labelled as covering it (the source-anchored
acceptance in scripts/qa_gold_anchor_smoke.py is what catches those).

Fixtures are synthetic (no copyrighted text) and shaped like the IRJET conversion that
passed every hard gate while carrying these defects.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mmrag_v2.validators import structural_outcomes as so

REPO = Path(__file__).resolve().parents[1]


def row(cid, modality="text", page=1, content="", heading=None, nxt=None):
    return {
        "chunk_id": cid,
        "modality": modality,
        "content": content,
        "metadata": {
            "page_number": page,
            "hierarchy": {"parent_heading": heading, "breadcrumb_path": [], "level": None},
        },
        "semantic_context": {"prev_text_snippet": None, "next_text_snippet": nxt},
    }


def chain(rows):
    """Fill next_text_snippet the way the pipeline does (successor content prefix)."""
    for i, r in enumerate(rows):
        r["semantic_context"]["next_text_snippet"] = (
            rows[i + 1]["content"][:300] if i + 1 < len(rows) else None
        )
    return rows


# ------------------------------------------------------------------ snippets
def test_snippets_quiet_on_consistent_chain():
    rows = chain(
        [
            row("a", content="alpha beta gamma"),
            row("b", content="delta epsilon"),
            row("c", content="zeta"),
        ]
    )
    rep = so.snippet_consistency(rows)
    assert rep.orphans == [] and rep.coverage_gaps == []


def test_snippets_fire_on_stale_neighbour_after_a_dropped_chunk():
    # b's snippet still describes a figure that was filtered out after the lookahead ran.
    rows = chain(
        [
            row("a", content="alpha beta"),
            row("img", modality="image", content="A line graph of power"),
            row("c", content="gamma delta"),
        ]
    )
    rows.pop(1)  # the image chunk is dropped after snippets were computed
    rep = so.snippet_consistency(rows)
    assert rep.orphans == ["a"]


def test_snippets_stray_snippet_on_last_chunk_is_an_orphan():
    rows = chain([row("a", content="alpha"), row("b", content="beta")])
    rows[-1]["semantic_context"]["next_text_snippet"] = "leftover text"
    assert so.snippet_consistency(rows).orphans == ["b"]


def test_snippet_coverage_gap_catches_nulling_every_snippet():
    # KNOWN fix-induced fault: a "refresh" that nulls all snippets scores 0 orphans.
    rows = chain([row("a", content="alpha"), row("b", content="beta"), row("c", content="gamma")])
    for r in rows:
        r["semantic_context"]["next_text_snippet"] = None
    rep = so.snippet_consistency(rows)
    assert rep.orphans == []
    assert rep.coverage_gaps == ["a", "b"]  # the last chunk has no successor: no gap


def test_snippets_ignore_whitespace_and_control_characters():
    rows = [row("a", content="alpha"), row("b", content="one  two\x0bthree")]
    rows[0]["semantic_context"]["next_text_snippet"] = "one two three"
    assert so.snippet_consistency(rows).orphans == []


# ------------------------------------------------------------------ headings
def test_heading_inside_body_fires_when_a_heading_is_a_tail_line():
    rows = [
        row(
            "a",
            heading="III. METHOD",
            content="Intro paragraph text.\nIII. METHOD\nBody after the heading.",
        ),
        row("b", heading="III. METHOD", content="More body."),
    ]
    assert so.heading_inside_body(rows) == ["a"]


def test_heading_inside_body_quiet_when_heading_leads_its_chunk():
    rows = [
        row("a", heading="III. METHOD", content="III. METHOD\nBody paragraph."),
        row("b", heading="III. METHOD", content="More body."),
    ]
    assert so.heading_inside_body(rows) == []


def test_heading_inside_body_is_known_blind_to_nulled_parents():
    # KNOWN-BLIND: a fix that nulls every parent_heading shrinks the reference set to nothing.
    rows = [row("a", heading=None, content="Intro.\nIII. METHOD\nBody.")]
    assert so.heading_inside_body(rows) == []


# ------------------------------------------------------------------ furniture
HDR = "International Journal of Widget Research e-ISSN {n}\nVolume 4 Issue 2 www.example.org"


def test_furniture_lines_fire_on_header_glued_into_body_on_three_pages():
    rows = [
        row(
            f"c{p}",
            page=p,
            content=HDR.format(n=p) + f"\nBody text unique to page {p} of the paper.",
        )
        for p in (1, 2, 3)
    ]
    flagged = so.furniture_line_chunks(rows)
    assert {cid for cid, _ in flagged} == {"c1", "c2", "c3"}


def test_furniture_lines_quiet_when_a_line_repeats_on_only_two_pages():
    rows = [row(f"c{p}", page=p, content=HDR.format(n=p) + f"\nBody unique {p}.") for p in (1, 2)]
    assert so.furniture_line_chunks(rows) == []


def test_furniture_lines_ignore_short_lines_and_tables():
    rows = [row(f"c{p}", page=p, content=f"Fig {p}\nIntro") for p in (1, 2, 3, 4)]
    rows += [row("t", modality="table", page=1, content="| a | b |\n| c | d |")]
    assert so.furniture_line_chunks(rows) == []


def test_furniture_lines_are_known_blind_to_over_deleted_body_lines():
    # KNOWN-BLIND: the metric cannot tell furniture from a legitimately repeated body line.
    # Deleting such a line drives the metric to zero exactly as deleting true furniture does;
    # only the source-anchored / token-balance checks can see the loss.
    legit = "Table of the measured values follows below in this section."
    topics = {
        1: "resistance",
        2: "capacitance",
        3: "inductance",
    }  # words, not digits: digits are normalized away
    before = [
        row(
            f"c{p}", page=p, content=legit + f"\nA distinct discussion of {topics[p]} in this part."
        )
        for p in (1, 2, 3)
    ]
    after = [
        row(f"c{p}", page=p, content=f"A distinct discussion of {topics[p]} in this part.")
        for p in (1, 2, 3)
    ]
    assert len(so.furniture_line_chunks(before)) == 3
    assert so.furniture_line_chunks(after) == []


# ------------------------------------------------------------------ references
def test_references_quiet_on_ordered_newline_entries():
    body = "\n".join(
        f"[{i}] A. Author, Title {i}, Journal, vol. {i}, no. 2, pp. 1-9, 2010." for i in range(1, 6)
    )
    rows = [row("r", heading="REFERENCES", content=body)]
    rep = so.reference_entry_integrity(rows)
    assert (rep.order_breaks, rep.split_entries) == (0, 0)


def test_references_fire_on_inline_entries_out_of_order_and_cut():
    # Right-column entries emitted as ONE inline element before the left-column ones,
    # and an entry cut at "vol. 4, no." (real IRJET shape; no newline before the labels).
    c1 = "[10] A. Author, Paper ten, Journal, 2011. [11] B. Writer, Paper eleven, Journal, vol. 4, no."
    c2 = "8, August 2013. [12] C. Person, Paper twelve, Journal, 2012."
    c3 = "[1] D. Lead, Paper one, Journal, 2001. [2] E. Second, Paper two, Journal, 2002."
    rows = [
        row("r1", heading="REFERENCES", content=c1),
        row("r2", heading="REFERENCES", content=c2),
        row("r3", heading="REFERENCES", content=c3),
    ]
    rep = so.reference_entry_integrity(rows)
    assert rep.order_breaks >= 1
    assert rep.split_entries >= 1


def test_references_ignore_body_citations_outside_reference_sections():
    rows = [
        row(
            "b",
            heading="INTRODUCTION",
            content="Reference [10] presented a model. Reference [11] extended it. See also [12] for details.",
        )
    ]
    rep = so.reference_entry_integrity(rows)
    assert rep.reference_rows == 1  # three labels make it reference-like...
    assert (rep.order_breaks, rep.split_entries) == (
        0,
        0,
    )  # ...but no label follows a sentence end + capital


def test_references_known_blind_to_dropped_entries():
    # KNOWN-BLIND: a fix that silently drops an entry leaves the remaining labels ordered.
    body = "[1] A. Author, Title, 2001.\n[3] C. Person, Title, 2003."
    rep = so.reference_entry_integrity([row("r", heading="References", content=body)])
    assert (rep.order_breaks, rep.split_entries) == (0, 0)


# ------------------------------------------------------------------ figures
def test_figure_deficit_fires_when_captions_outnumber_images():
    rows = [
        row("t1", page=3, content="Fig. 3 Interfacing model\nFig. 6 I-V characteristic curve"),
        row("i1", modality="image", page=3, content="A schematic."),
    ]
    assert so.figure_deficit_pages(rows) == [(3, [3, 6], 1)]


def test_figure_deficit_quiet_when_every_caption_has_an_image():
    rows = [
        row("t1", page=2, content="Fig. 1 Equivalent circuit\nFig. 2 Array model"),
        row("i1", modality="image", page=2, content="x"),
        row("i2", modality="image", page=2, content="y"),
    ]
    assert so.figure_deficit_pages(rows) == []


def test_figure_deficit_ignores_body_sentences_and_counts_panels_once():
    rows = [
        row(
            "t",
            page=4,
            content="Fig. 4 shows the simulation of the array.\nFig. 10(a) Array output\nFig. 10(b) Boost output",
        ),
        row("i", modality="image", page=4, content="z"),
    ]
    assert so.figure_deficit_pages(rows) == []


def test_figure_deficit_counts_a_caption_folded_into_the_image_content():
    rows = [row("i", modality="image", page=5, content="Fig. 7 Flow chart for the algorithm")]
    assert so.figure_deficit_pages(rows) == []


# ------------------------------------------------------------------ aggregate + host script
def _write_jsonl(path, rows):
    meta = {"object_type": "ingestion_metadata", "doc_id": "d"}
    path.write_text("\n".join(json.dumps(r) for r in [meta] + rows) + "\n", encoding="utf-8")


def test_compute_all_summary_lines_shape():
    rows = chain([row("a", content="alpha"), row("b", content="beta")])
    lines = list(so.compute_all(rows).summary_lines())
    assert lines[0].startswith("orphan_snippet_chunks=0 snippet_coverage_gap=0 snippets_present=1")
    assert any(ln.startswith("figure_deficit_pages=0") for ln in lines)


def test_host_script_prints_guards_and_keeps_exit_and_verdict(tmp_path):
    rows = chain(
        [row("a", page=1, content="alpha text here"), row("b", page=1, content="beta text here")]
    )
    jl = tmp_path / "ingestion.jsonl"
    _write_jsonl(jl, rows)
    res = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "qa_semantic_fidelity.py"), str(jl)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert res.returncode == 0
    assert "orphan_snippet_chunks=0" in res.stdout
    assert "SEMANTIC_PASS" in res.stdout  # report-only: the new guards never flip the verdict
    assert "Traceback" not in res.stdout + res.stderr


def test_host_script_survives_metric_failure(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "qa_sem_wp02", REPO / "scripts" / "qa_semantic_fidelity.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class Boom:
        @staticmethod
        def compute_all(rows):
            raise RuntimeError("unexpected row shape")

    monkeypatch.setattr(mod, "structural_outcomes", Boom)
    jl = tmp_path / "ingestion.jsonl"
    _write_jsonl(jl, [row("a", content="alpha")])
    monkeypatch.setattr(sys, "argv", ["qa_semantic_fidelity.py", str(jl)])
    assert mod.main() == 0


# ------------------------------------------------------------------ INV-062: older advisory metrics had no fixtures
def _load_qa():
    spec = importlib.util.spec_from_file_location(
        "qa_sem_inv062", REPO / "scripts" / "qa_semantic_fidelity.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bbox_row(cid, page, content, y0, y1):
    r = row(cid, page=page, content=content)
    r["metadata"]["spatial"] = {"bbox": [50, y0, 900, y1]}
    return r


def test_count_running_furniture_fires_on_repeating_margin_folio_and_is_quiet_on_content():
    qa = _load_qa()
    bad = [_bbox_row(f"f{p}", p, f"Month {p} 2025 // www.example.org", 960, 975) for p in (1, 2, 3)]
    assert qa.count_running_furniture(bad) == 3
    good = [
        _bbox_row("h", 1, "Chapter heading", 300, 340),
        _bbox_row("g", 2, "Another heading", 300, 340),
    ]
    assert qa.count_running_furniture(good) == 0


def test_count_cross_page_dupes_fires_on_caption_repeated_across_pages_and_is_quiet_when_unique():
    qa = _load_qa()
    bad = [
        row(f"d{p}", page=p, content="(a) Normalized throughput. Higher is better.")
        for p in (1, 2, 3, 4)
    ]
    assert qa.count_cross_page_dupes(bad) == 3
    good = [
        row(f"u{p}", page=p, content=f"Unique paragraph number {p} with enough characters.")
        for p in (1, 2, 3)
    ]
    assert qa.count_cross_page_dupes(good) == 0


def test_code_fence_consistency_metric_via_host_script(tmp_path):
    def code(cid, content):
        r = row(cid, modality="code", page=1, content=content)
        return r

    for content_set, expect in (
        (
            [code("k1", "```python\nprint(1)\n```"), code("k2", "```python\nprint(2)\n```")],
            "code_fence_consistency=1.0000",
        ),
        (
            [code("k1", "```python\nprint(1)\n```"), code("k2", "print(2)")],
            "code_fence_consistency=0.5000",
        ),
    ):
        jl = tmp_path / "ingestion.jsonl"
        _write_jsonl(jl, content_set)
        res = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "qa_semantic_fidelity.py"), str(jl)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert expect in res.stdout


# ------------------------------------------------------------------ real output (skipped when the gitignored output is absent)
REAL = REPO / "output" / "cloud_probe_irjet" / "ingestion.jsonl"


@pytest.mark.skipif(not REAL.exists(), reason="gitignored local output not present")
def test_real_defective_irjet_output_is_seen_by_the_guards():
    rows = [json.loads(ln) for ln in REAL.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rows = [r for r in rows if r.get("object_type") != "ingestion_metadata"]
    out = so.compute_all(rows)
    assert len(out.snippets.orphans) >= 6
    assert len(out.heading_inside_body) >= 6
    assert len(out.furniture_lines) >= 9
    assert out.references.order_breaks >= 1
    assert [p for p, _, _ in out.figure_deficits] == [3, 4, 6]
