"""Contract tests for the shared R3 code-quality metric (scripts/_code_quality).

These pin the redesign in docs/PLAN_R3_CODE_GATE_REDESIGN.md:
  - the right population (modality=code + legacy text-code),
  - positive code-ID excluding equations/LaTeX (the false-fail class),
  - judge-only-judgeable (flat/REPL exempt; flattened blocks FAIL),
  - Policy B verdict (warn vs density-gated hard-fail).

They are executable requirements, not fixture-fits. AIOS-class mangled Python
must FAIL the metric; equations must never enter the code population.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import _code_quality as cq  # noqa: E402

# --- positive code-ID ------------------------------------------------------


def test_python_code_has_structure():
    assert cq.has_code_structure("def f(x):\n    return x")
    assert cq.has_code_structure("class A(B):\n    pass")
    assert cq.has_code_structure("import os")
    assert cq.has_code_structure(">>> 1 + 1\n2")


@pytest.mark.parametrize(
    "math",
    [
        "Pb+PbO₂+2H₂SO₄ ↔ 2PbSO₄+2H₂O",
        "V_oc = a + b × DOD + (c + d × DOD)T",
        "V_{fc} = E_{cell}",
        r"\frac{a}{b} = \ln(x)",
        "Anode:\nH₂ → 2H⁺ + 2e⁻",
    ],
)
def test_equations_have_no_code_structure(math):
    # The false-fail class: extractor VLMs mislabel these as code. Positive
    # code-ID must drop them so they never enter the indentation metric.
    assert not cq.has_code_structure(math)


# --- judgeable classification ---------------------------------------------


def test_flat_single_statement_not_judgeable():
    assert not cq.is_judgeable("import threading")
    assert not cq.is_judgeable("x = compute(y)")


def test_repl_not_judgeable():
    assert not cq.is_judgeable(">>> def f(x):\n...     return x\n")


def test_multiline_suite_is_judgeable():
    assert cq.is_judgeable("def f(x):\n    return x + 1")
    assert cq.is_judgeable("class A:\n    def m(self):\n        return 1")


def test_brace_block_is_judgeable():
    assert cq.is_judgeable("int main() {\n    return 0;\n}")


def test_collapsed_nested_suite_is_judgeable_and_degraded():
    # The WORST extraction outcome: a nested suite flattened onto ONE line so the
    # ``:`` is no longer at end-of-line. Measured live: MinerU-1.2B rendered
    # Fluent Python p111's
    #     found = 0
    #     for n in needles:
    #         if n in haystack:
    #             found += 1
    # as a single jammed line. This must be judgeable (so it is scored) AND fail
    # indentation_ok (the nesting was destroyed) — not slip through as "flat".
    collapsed = "found = 0\nfor n in needles: if n in haystack: found += 1"
    assert cq.is_judgeable(collapsed)
    assert not cq.indentation_ok(collapsed)
    # The exact on-disk shape (fenced + MinerU math-artifacted operators).
    p111 = "```\nfound \\(= 0\\)\nfor n in needles: if n in haystack: found \\(+ = 1\\)\n```"
    assert cq.is_judgeable(p111)
    assert not cq.indentation_ok(p111)


def test_single_line_collapsed_suite_is_judgeable_and_degraded():
    # The WHOLE chunk is one fully-collapsed line (no surrounding lines). The
    # line-count guard must NOT drop it as "flat": its nesting was destroyed, so
    # it must still be scored AND fail indentation_ok.
    one_line = "def f(x): if x: return x"
    assert cq.is_judgeable(one_line)
    assert not cq.indentation_ok(one_line)
    # And it must drive a doc hard-fail when non-incidental.
    rows = [_code_chunk(f"c{i}", "for i in xs: if i in ys: a += i") for i in range(3)]
    m = cq.code_quality(rows)
    assert m.n_judgeable == 3 and m.n_judgeable_fail == 3
    assert cq.gate_verdict(m) == "fail"


@pytest.mark.parametrize(
    "legit",
    [
        "x = compute()\nif x: return y",  # single one-line compound stmt
        "data = load()\nfor x in y: print(x)",  # single one-line for
        "result = [x for x in items if x > 0]\nprint(result)",  # comprehension
        "f = lambda x: x + 1\ng = lambda y: y * 2",  # lambda (one expr colon)
    ],
)
def test_collapsed_detector_no_false_positive(legit):
    # A LEGITIMATE one-line compound statement (a single suite colon),
    # comprehension, or lambda is NOT a collapsed nested block — it must not be
    # flagged judgeable-via-collapse. The detector requires TWO block-keyword
    # suite colons on one line (flattened nesting) precisely to avoid this.
    assert not cq.is_judgeable(legit)


def test_collapsed_blocks_drive_doc_hardfail():
    # When collapsed nested blocks are non-incidental (>= min_judgeable, above the
    # density floor) the document hard-fails — the blind spot previously let them
    # pass invisibly. Three collapsed chunks, nothing else.
    rows = [_code_chunk(f"c{i}", "a = 0\nfor i in xs: if i in ys: a += i") for i in range(3)]
    m = cq.code_quality(rows)
    assert m.n_judgeable == 3 and m.n_judgeable_fail == 3
    assert cq.gate_verdict(m) == "fail"


# --- indentation_ok --------------------------------------------------------


def test_nested_block_indentation_ok():
    assert cq.indentation_ok("def f(x):\n    return x + 1")


def test_flattened_block_indentation_fails():
    # AIOS shape: a multi-line class whose body indentation was stripped, so
    # every line sits at column 0 (one depth) — the mangling R3 must catch.
    flat = "class Scheduler:\ndef __init__(self, llm):\nself.llm = llm"
    assert not cq.indentation_ok(flat)


# --- population + aggregate metric ----------------------------------------


def _code_chunk(cid, content, modality="code"):
    c = {"chunk_id": cid, "modality": modality, "content": content}
    if modality == "text":
        c["metadata"] = {"chunk_type": "code"}
    return c


def test_population_covers_both_seams():
    chunks = [
        _code_chunk("a", "def f():\n    return 1", modality="code"),
        _code_chunk("b", "def g():\n    return 2", modality="text"),
        {"chunk_id": "c", "modality": "text", "content": "prose", "metadata": {}},
    ]
    r = cq.code_quality(chunks)
    assert r.n_population == 2
    assert r.n_judgeable == 2


def test_equations_excluded_from_population_metric():
    chunks = [
        _code_chunk("e1", "V_oc = a + b × DOD"),
        _code_chunk("e2", "Pb+PbO₂ ↔ 2PbSO₄"),
    ]
    r = cq.code_quality(chunks)
    assert r.n_population == 2
    assert r.n_struct == 0
    assert r.n_math_excluded == 2
    assert r.n_judgeable == 0
    assert r.indentation_fidelity == 1.0  # nothing judgeable -> no false fail


def test_mangled_python_fails_fidelity():
    good = "def a():\n    return 1"
    flat1 = "class X:\ndef __init__(self):\nself.a = 1"
    flat2 = "class Y:\ndef run(self):\nself.go()"
    chunks = [
        _code_chunk("g", good),
        _code_chunk("f1", flat1),
        _code_chunk("f2", flat2),
    ]
    r = cq.code_quality(chunks)
    assert r.n_judgeable == 3
    assert r.n_judgeable_fail == 2
    assert r.indentation_fidelity == pytest.approx(1 / 3)
    assert set(r.degraded_ids) == {"f1", "f2"}


# --- duplicate text-as-code exclusion --------------------------------------


def test_duplicate_text_code_excluded_from_metric():
    # The production pipeline can emit a listing twice: a clean modality=code
    # chunk + a flush-left modality=text duplicate stamped content_classification
    # =code. The duplicate is output redundancy, not an independent indentation
    # failure — the code WAS extracted with indentation. It must be excluded so
    # it does not drag down fidelity. (Observed live on AIOS via Qwen.)
    clean_src = (
        "class Scheduler:\n"
        "    def __init__(self, llm, memory_manager, storage_manager):\n"
        "        self.llm = llm\n"
        "        self.memory_manager = memory_manager\n"
        "    def stop(self):\n"
        "        self.active = False\n"
        "        return None"
    )
    clean = _code_chunk("c", clean_src, modality="code")
    # Flush-left duplicate with a leaked page header prefixing it.
    dup = {
        "chunk_id": "d",
        "modality": "text",
        "content": "AIOS: LLM Agent Operating System\n"
        + "\n".join(ln.lstrip() for ln in clean_src.splitlines()),
        "metadata": {"chunk_type": "paragraph", "content_classification": "code"},
    }
    r = cq.code_quality([clean, dup])
    assert r.n_duplicate_excluded == 1
    assert r.n_judgeable == 1  # only the clean code chunk is scored
    assert r.indentation_fidelity == 1.0


def test_text_only_mangled_code_with_no_clean_twin_still_fails():
    # ANTI-WEAKENING GUARD: a flush-left code block with NO clean modality=code
    # twin is a genuine extraction failure and must still be counted/failed.
    frag = {
        "chunk_id": "f",
        "modality": "text",
        "content": "class Z:\ndef go(self):\nself.run()",
        "metadata": {"chunk_type": "code"},
    }
    r = cq.code_quality([frag])
    assert r.n_duplicate_excluded == 0
    assert r.n_judgeable == 1
    assert r.n_judgeable_fail == 1
    assert r.indentation_fidelity == 0.0


# --- Policy B verdict ------------------------------------------------------


def test_verdict_pass_when_too_few_judgeable():
    r = cq.CodeQuality(n_judgeable=2, n_judgeable_fail=2, total_chunks=100)
    assert cq.gate_verdict(r) == "pass"


def test_verdict_pass_when_fidelity_ok():
    r = cq.CodeQuality(n_judgeable=10, n_judgeable_fail=0, total_chunks=100)
    assert cq.gate_verdict(r) == "pass"


def test_verdict_warn_below_density_floor():
    # 3 mangled judgeable blocks in a 300-chunk prose doc: incidental -> warn.
    r = cq.CodeQuality(n_judgeable=3, n_judgeable_fail=3, total_chunks=300)
    assert r.judgeable_density < cq.DEFAULT_HARDFAIL_DENSITY
    assert cq.gate_verdict(r) == "warn"


def test_verdict_hardfail_above_density_floor():
    # AIOS-class: 9 judgeable, 5 broken, 163 chunks -> density 0.055 -> fail.
    r = cq.CodeQuality(n_judgeable=9, n_judgeable_fail=5, total_chunks=163)
    assert r.judgeable_density >= cq.DEFAULT_HARDFAIL_DENSITY
    assert r.indentation_fidelity < cq.DEFAULT_INDENT_FIDELITY_FLOOR
    assert cq.gate_verdict(r) == "fail"
