"""V3 degenerate-repetition collapse (crucible fix #3b, 2026-06-04).

The 8B VLM loops on dense repetitive content - the crucible's CarOK p7 table
had a cell with "Transporter 1.9 TD 68pk, " repeated ~hundreds of times (one
13,955-char row) tripping TABLE_CORRUPTION. Chunk-level dedup can't see it (one
chunk) and the mild repetition penalty didn't stop it. ``_collapse_degenerate_
repeats`` collapses any <=80-char unit repeated >= 8 times in a row to a single
occurrence; CODE is excluded at the call site (must stay verbatim).

Offline/deterministic.
"""

from __future__ import annotations

import time

from mmrag_v3.engines.vlm_native import _collapse_degenerate_repeats


def test_collapses_long_loop():
    looped = "| Mapco | Oliefilter Audi, " + "Transporter 1.9 TD 68pk, " * 400
    out = _collapse_degenerate_repeats(looped)
    assert out.count("Transporter 1.9 TD 68pk,") == 1
    assert len(out) < 200
    assert out.startswith("| Mapco | Oliefilter Audi, ")


def test_keeps_below_threshold():
    # 3 and 7 consecutive repeats are kept (threshold is 8).
    assert _collapse_degenerate_repeats("a, a, a") == "a, a, a"
    seven = "x" * 7
    assert _collapse_degenerate_repeats(seven) == seven


def test_collapses_single_char_runaway():
    out = _collapse_degenerate_repeats("value" + "." * 5000)
    assert out == "value."


def test_distinct_content_untouched():
    text = "Castrol Edge 5W-30, Castrol Edge 0W-40, Gulf 10W-40, Shell Helix 5W-30"
    assert _collapse_degenerate_repeats(text) == text


def test_empty_and_short():
    assert _collapse_degenerate_repeats("") == ""
    assert _collapse_degenerate_repeats("hello") == "hello"


def test_fast_on_large_loop():
    big = "ABCDEFGHIJ " * 20000  # 220k chars, one unit looped
    t0 = time.time()
    out = _collapse_degenerate_repeats(big)
    assert (time.time() - t0) < 1.0  # no catastrophic backtracking
    assert out == "ABCDEFGHIJ "
