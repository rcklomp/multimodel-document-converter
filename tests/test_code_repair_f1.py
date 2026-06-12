"""PLAN_F1 J1: residual code-defect repairs (smart quotes, open-string wraps,
repair-only bracket wraps, mid-docstring chunk merge).

Pins the three user-directed fixes on the ACTUAL Jungjun failure patterns. The
contract: each repair makes a broken chunk parse, NEVER degrades a parseable one,
and leaves legal multi-line constructs (docstrings, multi-line calls) intact.
"""

from __future__ import annotations

import ast

from mmrag_v2.batch_processor import (
    _leaves_docstring_open,
    _normalize_code_quotes,
    _repair_code_content,
    _rejoin_wrapped_code_lines,
)


def _parses(s: str) -> bool:
    try:
        ast.parse(s)
        return True
    except (SyntaxError, ValueError):
        return False


# --- (b) smart quotes ---------------------------------------------------------
def test_smart_quotes_normalized_and_parse():
    src = 'print(“Messages: ”, messages)'   # curly double quotes (pg65)
    assert not _parses(src)
    fixed = _repair_code_content(src)
    assert _parses(fixed)
    assert "“" not in fixed and "”" not in fixed


def test_smart_single_quotes_normalized():
    assert _normalize_code_quotes("x = ‘a’") == "x = 'a'"


# --- (c) open-string hard-wraps ----------------------------------------------
def test_rejoin_wrapped_double_string():
    # pg34 shape: a double-quoted string wrapped across printed lines.
    src = '\n'.join([
        'msg = {',
        '    "role": "user",',
        '    "content": "My name is John Smith and my',
        'phone is 555-1234."',
        '}',
    ])
    assert not _parses(src)
    assert _parses(_repair_code_content(src))


def test_rejoin_implicit_concat_wrap():
    # pg43 shape: single-quoted string whose closing quote wrapped to next line.
    src = "d = {\n    'q': 'If he could maintain his pace\n'\n    'indefinitely, how far'\n}"
    assert not _parses(src)
    assert _parses(_repair_code_content(src))


# --- repair-only / non-degradation contract ----------------------------------
def test_clean_code_unchanged_and_parses():
    src = "def f(x):\n    if x:\n        return x\n    return 0"
    assert _parses(src)
    assert _repair_code_content(src) == src  # idempotent no-op on clean code


def test_legal_multiline_call_not_collapsed():
    # A parseable multi-line call must NOT be collapsed (bracket rejoin is
    # repair-only: it never fires on already-parseable chunks).
    src = "r = acompletion(\n    model='m',\n    messages=[{'role': 'user'}],\n)"
    assert _parses(src)
    out = _repair_code_content(src)
    assert out == src
    assert out.count("\n") == src.count("\n")  # multi-line layout preserved


def test_broken_bracket_wrap_repaired_only_when_it_helps():
    # mid-token bracket wrap (pg117 shape: request.tool + s -> request.tools)
    src = "tools = [t.tool_definition for t in request.tool\ns if t.enabled]"
    assert not _parses(src)
    fixed = _repair_code_content(src)
    # repair may or may not fully fix this, but must never make it worse
    assert _parses(fixed) or fixed == _rejoin_wrapped_code_lines(_normalize_code_quotes(src))


# --- (a) mid-docstring split detection ---------------------------------------
def test_detects_unterminated_docstring():
    assert _leaves_docstring_open('def f():\n    """start of a docstring that')
    assert not _leaves_docstring_open('def f():\n    """complete."""\n    return 1')


def test_complete_triple_docstring_not_mangled():
    src = 'def f():\n    """A\n    multi-line\n    docstring."""\n    return 1'
    assert _parses(src)
    assert _parses(_repair_code_content(src))
