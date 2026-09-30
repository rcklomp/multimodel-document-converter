"""Fullwidth code token-corruption repair (WS2a, PLAN_FIDELITY_ORACLE_FIRST_V1 Section 3').

A VLM emits fullwidth ASCII-variant punctuation/digits in place of their ASCII code
counterparts - the `[:，2]` fullwidth-comma class of Chaubal engine token corruption.
`_normalize_code_fullwidth` maps the UNAMBIGUOUS fullwidth punctuation/digits back to
ASCII (correct-by-construction, not a guess), chained into `_repair_code_content`.

Frozen fixture: the exact corrupted tokens. Also pins the DEFERRED boundary - fullwidth
LETTERS (possible legit string content) and ambiguous LaTeX (`\\equiv`) are NOT touched.
"""

from __future__ import annotations

from mmrag_v2.batch_processor import _normalize_code_fullwidth, _repair_code_content


def test_fullwidth_comma_in_slice_repaired():
    # Chaubal: numpy slice `arr[:,2]` corrupted with a fullwidth comma U+FF0C.
    corrupt = "arr[:，2]"
    assert _normalize_code_fullwidth(corrupt) == "arr[:,2]"


def test_fullwidth_punctuation_set_repaired():
    # colon, parens, semicolon, brackets, operators -> ASCII
    corrupt = "def f（x）：\n    return x；"
    fixed = _normalize_code_fullwidth(corrupt)
    assert fixed == "def f(x):\n    return x;"


def test_fullwidth_digits_repaired():
    assert _normalize_code_fullwidth("x = １２３") == "x = 123"


def test_repair_makes_corrupted_code_parse():
    # End-to-end via _repair_code_content: fullwidth-corrupted code becomes valid.
    corrupt = "for i in range（10）：\n    print（i）"
    fixed = _repair_code_content(corrupt)
    import ast

    ast.parse(fixed)  # must not raise


def test_idempotent_on_clean_code():
    clean = "def g(a, b):\n    return a + b\n"
    assert _normalize_code_fullwidth(clean) == clean
    assert _repair_code_content(clean) == clean


# --- deferred boundary: do NOT over-scrub --------------------------------------


def test_fullwidth_letters_preserved():
    # Fullwidth LETTERS can be legitimate string content; a guess would be wrong.
    # 'Ａ' = U+FF21, 'ｚ' = U+FF5A -> intentionally untouched.
    s = 'label = "Ａｚ"'
    assert _normalize_code_fullwidth(s) == s


def test_latex_equiv_left_untouched():
    # Ambiguous (`=` vs `==`); DEFERRED, so it must pass through unchanged.
    s = "x \\(\\equiv\\) y"
    assert _normalize_code_fullwidth(s) == s
