"""Seeded-fault sensitivity suite — instrument-validation regression net.

PLAN_EXTRACTION_FIDELITY_V1 Section 7.3 (entry gate for Phase 1's internal axis
and Phase 2's lane decisions). These tests pin the BLINDNESS verdicts the harness
measures so a later change to a scorer normalization or a gate signal cannot
silently re-open a blind spot. The headline contract: the OmniDocBench text metric
strips ALL whitespace and is therefore BLIND to code-indentation loss — the exact
failure class this plan most fears, invisible to the instrument that would judge it.

In-process instruments (text edit distance, gate junk-presence signals) are
asserted unconditionally. The real TEDS instrument needs the ``omnidocbench`` conda
env and is asserted only when available (skipped otherwise, never faked).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

_spec = importlib.util.spec_from_file_location(
    "seeded_fault_sensitivity", SCRIPTS / "seeded_fault_sensitivity.py"
)
sfs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sfs)


# --------------------------------------------------------------------------- #
# Instrument 1: OmniDocBench text edit distance (the headline blindness)
# --------------------------------------------------------------------------- #
def test_text_ed_is_blind_to_code_indentation_strip():
    # clean_string deletes all whitespace -> stripping indentation is invisible.
    # This is the corner the whole plan turns on: the metric that would judge a
    # code-fidelity regression cannot see one.
    mutated = sfs.strip_code_indentation(sfs.CODE_BLOCK)
    assert mutated != sfs.CODE_BLOCK  # the fault really changed the bytes
    delta = sfs.text_edit_distance(sfs.CODE_BLOCK, mutated)
    assert delta == 0.0  # BLIND


def test_text_ed_moves_on_dropped_label():
    mutated = sfs.drop_small_label(sfs.LABELED_CAPTION, sfs.SMALL_LABEL)
    delta = sfs.text_edit_distance(sfs.LABELED_CAPTION, mutated)
    assert delta > sfs._TEXT_ED_MOVE_EPS  # MOVED — text removal is visible


def test_text_ed_moves_on_table_flatten_and_reorder():
    flat = sfs.flatten_table_to_prose(sfs.HTML_TABLE)
    reordered = sfs.reorder_two_columns(sfs.HTML_TABLE)
    assert sfs.text_edit_distance(sfs.HTML_TABLE, flat) > sfs._TEXT_ED_MOVE_EPS
    assert sfs.text_edit_distance(sfs.HTML_TABLE, reordered) > sfs._TEXT_ED_MOVE_EPS


# --------------------------------------------------------------------------- #
# Instrument 3: gate junk-presence signals are BLIND to every omission fault
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "original, mutated",
    [
        (sfs.CODE_BLOCK, sfs.strip_code_indentation(sfs.CODE_BLOCK)),
        (sfs.LABELED_CAPTION, sfs.drop_small_label(sfs.LABELED_CAPTION, sfs.SMALL_LABEL)),
        (sfs.HTML_TABLE, sfs.flatten_table_to_prose(sfs.HTML_TABLE)),
        (sfs.HTML_TABLE, sfs.reorder_two_columns(sfs.HTML_TABLE)),
    ],
)
def test_gate_quality_signals_are_blind_to_content_omission(original, mutated):
    # The built signals detect junk PRESENCE (furniture, dupes, garbled
    # headings); none can see content ABSENCE. This is the I7 risk made
    # measurable: a lossy engine's output scores identically to the faithful one.
    assert sfs.gate_quality_signal_vector(original) == sfs.gate_quality_signal_vector(mutated)


# --------------------------------------------------------------------------- #
# Instrument 2: real OmniDocBench TEDS (only when the env is present)
# --------------------------------------------------------------------------- #
def test_teds_sanity_and_table_fault_movement():
    identical = sfs.table_teds(sfs.HTML_TABLE, sfs.HTML_TABLE)
    if identical is None:
        pytest.skip("omnidocbench env/repo unavailable — TEDS instrument NOT-RUN")
    assert identical == pytest.approx(1.0)  # a table vs itself must score 1.0
    flat = sfs.table_teds(sfs.flatten_table_to_prose(sfs.HTML_TABLE), sfs.HTML_TABLE)
    reordered = sfs.table_teds(sfs.reorder_two_columns(sfs.HTML_TABLE), sfs.HTML_TABLE)
    assert flat == pytest.approx(0.0)  # structure destroyed
    assert reordered < 1.0 - sfs._TEDS_MOVE_EPS  # column swap is visible to TEDS


# --------------------------------------------------------------------------- #
# Full blindness matrix matches the committed fixture (regression guard)
# --------------------------------------------------------------------------- #
def test_blindness_report_matches_committed_verdicts():
    report = sfs.build_blindness_report()
    by_fault = {f["fault"]: f for f in report["faults"]}
    # text-ED verdicts
    assert by_fault["strip_code_indentation"]["text_edit_distance"]["verdict"] == "BLIND"
    assert by_fault["drop_small_label"]["text_edit_distance"]["verdict"] == "MOVED"
    assert by_fault["flatten_table_to_prose"]["text_edit_distance"]["verdict"] == "MOVED"
    assert by_fault["reorder_two_columns"]["text_edit_distance"]["verdict"] == "MOVED"
    # every fault is BLIND to the junk-presence gate signals
    for f in report["faults"]:
        assert f["gate_quality_signals"]["verdict"] == "BLIND"
    # non-table faults mark TEDS N/A
    assert by_fault["strip_code_indentation"]["table_teds"]["verdict"] == "N/A"
    assert by_fault["drop_small_label"]["table_teds"]["verdict"] == "N/A"
