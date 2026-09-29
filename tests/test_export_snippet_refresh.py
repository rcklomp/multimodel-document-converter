"""PLAN_QUALITY_REMEDIATION_V1 WP-A2: neighbour snippets are correct on the FINAL export.

Defect: ``_apply_lookahead_buffer`` copies the successor's content[:300] before the export-chain
filters run, so every chunk removed afterwards (tiny-icon drop, pHash duplicate rejected INSIDE the
write loop, asset-mismatch skip) leaves its predecessor pointing at text that is not its neighbour
(IRJET: the description of a dropped figure survived as the next_text_snippet of the paragraph
before it; those descriptions also carried the only trace of the missing figures).

The repair runs in the post-export pass, i.e. on the final list, after every in-loop drop.
Fix-induced-fault guard: a "refresh" that nulls everything is caught by the coverage-gap metric,
and a consistent snippet stays byte-identical.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from mmrag_v2.batch_processor import BatchProcessor
from mmrag_v2.validators.structural_outcomes import snippet_consistency

REPO = Path(__file__).resolve().parents[1]
refresh = BatchProcessor._refresh_stale_next_snippets


def row(cid, content, nxt="__unset__", modality="text"):
    return {
        "chunk_id": cid,
        "modality": modality,
        "content": content,
        "semantic_context": {
            "prev_text_snippet": None,
            "next_text_snippet": None if nxt == "__unset__" else nxt,
        },
    }


def test_stale_snippet_after_a_dropped_chunk_is_replaced_by_the_real_successor():
    # a's snippet still describes the dropped figure that sat between a and b
    rows = [
        row("a", "alpha paragraph", nxt="A line graph of module power"),
        row("b", "beta paragraph text", nxt=None),
    ]
    assert refresh(rows) == 1
    assert rows[0]["semantic_context"]["next_text_snippet"] == "beta paragraph text"


def test_consistent_snippets_are_left_untouched_and_last_snippet_is_cleared():
    rows = [
        row("a", "alpha", nxt="beta text here"),
        row("b", "beta text here", nxt="gamma"),
        row("c", "gamma", nxt="leftover"),
    ]
    assert refresh(rows) == 1
    assert rows[0]["semantic_context"]["next_text_snippet"] == "beta text here"
    assert rows[1]["semantic_context"]["next_text_snippet"] == "gamma"
    assert rows[2]["semantic_context"]["next_text_snippet"] is None


def test_missing_snippet_with_a_successor_is_filled_with_the_lookahead_rule():
    rows = [row("a", "alpha", nxt=None), row("b", "x" * 500)]
    refresh(rows)
    assert rows[0]["semantic_context"]["next_text_snippet"] == "x" * 300


def test_whitespace_and_control_differences_are_not_treated_as_stale():
    rows = [row("a", "alpha", nxt="one two\x0bthree"), row("b", "one  twothree tail")]
    assert refresh(rows) == 0


def test_prev_snippet_is_never_touched_and_rows_without_semantic_context_are_skipped():
    rows = [
        row("a", "alpha", nxt="stale"),
        {"chunk_id": "z", "modality": "text", "content": "zzz"},
        row("b", "beta"),
    ]
    rows[0]["semantic_context"]["prev_text_snippet"] = "keep me"
    refresh(rows)
    assert rows[0]["semantic_context"]["prev_text_snippet"] == "keep me"
    assert "semantic_context" not in rows[1]


def test_guard_metric_reads_zero_orphans_and_zero_gaps_after_the_refresh():
    rows = [
        row("a", "alpha para", nxt="dropped figure text"),
        row("b", "beta para", nxt="also stale"),
        row("c", "gamma para", nxt="stale tail"),
    ]
    assert snippet_consistency(rows).orphans  # the defect is visible before
    refresh(rows)
    rep = snippet_consistency(rows)
    assert rep.orphans == [] and rep.coverage_gaps == []


def _bp():
    return BatchProcessor.__new__(BatchProcessor)


def test_post_export_pass_patches_header_count_and_repairs_snippets_but_keeps_clean_lines(tmp_path):
    header = {"object_type": "ingestion_metadata", "chunk_count": 99, "doc_id": "d"}
    clean = row("a", "alpha", nxt="beta text")
    stale = row("b", "beta text", nxt="dropped image description")
    last = row("c", "gamma")
    path = tmp_path / "ingestion.jsonl"
    path.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in (header, clean, stale, last)) + "\n",
        encoding="utf-8",
    )
    original_lines = path.read_text(encoding="utf-8").splitlines()
    _bp()._patch_export_file(path, written_chunks=3)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["chunk_count"] == 3
    assert lines[1] == original_lines[1]  # consistent row: byte-identical
    assert json.loads(lines[2])["semantic_context"]["next_text_snippet"] == "gamma"
    assert lines[3] == original_lines[3]


def test_post_export_pass_survives_a_corrupt_line_and_still_patches_the_header(tmp_path):
    header = {"object_type": "ingestion_metadata", "chunk_count": 1}
    path = tmp_path / "ingestion.jsonl"
    path.write_text(
        json.dumps(header) + "\n" + json.dumps(row("a", "alpha")) + "\n{not json\n",
        encoding="utf-8",
    )
    _bp()._patch_export_file(path, written_chunks=2)
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["chunk_count"] == 2


REAL = REPO / "output" / "cloud_probe_irjet" / "ingestion.jsonl"


@pytest.mark.skipif(not REAL.exists(), reason="gitignored local output not present")
def test_real_defective_output_has_orphans_before_and_none_after(tmp_path):
    copy = tmp_path / "ingestion.jsonl"
    shutil.copy(REAL, copy)

    def rows_of(path):
        rs = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return [r for r in rs if r.get("object_type") != "ingestion_metadata"]

    before = snippet_consistency(rows_of(copy))
    assert len(before.orphans) >= 6
    _bp()._patch_export_file(copy, written_chunks=len(rows_of(copy)))
    after = snippet_consistency(rows_of(copy))
    assert after.orphans == [] and after.coverage_gaps == []
