"""Heading sanity (PLAN_GATE_QUALITY_V1 F3).

The layout model labels furniture/garble as a "title" and it passes the
structural HEADING gate. F3 tightens is_valid_heading (the chunker-side fix) to
reject URL/email/bare-domain mastheads and CJK-Latin mixed garble, and adds a
heading_sanity_ratio advisory metric (the gate-side regression net) on the
resulting parent_heading strings. Real headings (numbered, all-caps, question,
German/Dutch) must still pass.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from mmrag_v2.state.context_state import is_valid_heading

# Audit garbage that previously passed and must now be rejected.
REJECT = [
    "会",  # a single hallucinated CJK glyph (also too short)
    "合DANCGING-WIITH",  # CJK-Latin mixed garble
    "SCAN THE QR CODE TO ORDER DIRECT FROM OUR SHOP shop.keypubliking.com/casubs",
    "Order from our online shop... shop.keypublishing.com/a400matlas",
    "editor@combataircraftjournal.com",
    "www.Key.Aero",
]

# Real headings across the corpus that must keep passing (regression guard).
KEEP = [
    "THE BOY WHO LIVED",
    "Chapter 2: An Array of Sequences",
    "5.1 Notation and Preliminaries",
    "14. Linking to Memory and Context",
    "Introduction",
    "What Is an AI Agent?",
    "KREATIVE AKTFOTOGRAFIE",
    "Durchgangige Prozesskette von der Spezifikation bis zum Test",
    # Tech headings: domain-shaped tokens with NO path must NOT be rejected
    # (review #2 - the bare-domain rule requires a path now).
    "ASP.NET Core",
    "asp.net Core",
    "Node.js Internals",
]


def test_masthead_with_path_still_rejected():
    # The audit mastheads carry a path/www/@ and must still be rejected, so the
    # #2 path-requirement did not open a hole.
    for h in [
        "SCAN THE QR CODE TO ORDER ... shop.keypubliking.com/casubs",
        "Order from our online shop... shop.keypublishing.com/a400matlas",
        "www.Key.Aero",
        "editor@combataircraftjournal.com",
    ]:
        assert not is_valid_heading(h), f"should reject: {h!r}"


def test_is_valid_heading_rejects_garbage():
    for h in REJECT:
        assert not is_valid_heading(h), f"should reject: {h!r}"


def test_is_valid_heading_keeps_real_headings():
    for h in KEEP:
        assert is_valid_heading(h), f"should keep: {h!r}"


def _load_qa_sem():
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "qa_semantic_fidelity", repo / "scripts" / "qa_semantic_fidelity.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qa_semantic_fidelity"] = mod
    spec.loader.exec_module(mod)
    return mod


def _chunk(heading):
    return {"modality": "text", "metadata": {"hierarchy": {"parent_heading": heading}}}


def test_heading_sanity_metric_counts_garbage():
    qa = _load_qa_sem()
    rows = [
        _chunk("THE BOY WHO LIVED"),  # ok
        _chunk("shop.keypubliking.com/casubs"),  # masthead url (with path, #2)
        _chunk("合DANCGING-WIITH"),  # cjk garble
        _chunk("36 | Chapter 2"),  # folio-shaped
        _chunk("Introduction"),  # ok
        _chunk("ASP.NET Core"),  # tech heading, must NOT be counted (#2)
    ]
    assert qa.count_insane_headings(rows) == 3


def test_heading_sanity_metric_quiet_on_clean_headings():
    qa = _load_qa_sem()
    rows = [_chunk(h) for h in KEEP]
    assert qa.count_insane_headings(rows) == 0
