"""Shared code-quality metric for the R3 code-indentation gate.

Single source of truth imported by ``qa_conversion_audit.py`` (hard gate) and
``qa_semantic_fidelity.py`` (advisory). It replaces the dead metric that scored
only ``modality=="text"`` chunks and was blind to the V3 pipeline's
``modality=="code"`` promotion. See ``docs/PLAN_R3_CODE_GATE_REDESIGN.md``.

Design (resolved in that plan):

* **Right population** — code is collected from ``modality=="code"`` chunks (the
  V3 promotion target) AND legacy ``modality=="text"`` chunks tagged
  ``chunk_type``/``content_classification == "code"``. Both, because the seam is
  in both producers.
* **Positive code-ID** — a chunk counts as *code* only if it shows code
  structure (keywords OR code punctuation OR a ``>>>`` REPL prompt). This
  POSITIVELY identifies code, so equations/formulas that an extractor VLM
  mislabels as ``code`` (verified: they carry ``original_vlm_type: code``) are
  excluded rather than heuristically un-excluded. A prior "exclude unicode math"
  heuristic was falsified (it missed LaTeX); positive ID is the robust direction.
* **Judge only what is judgeable** — indentation fidelity is scored ONLY on
  multi-line code that syntactically REQUIRES nesting (a Python ``:`` suite
  header or a brace block). Flat/single-statement code has no nesting to assess
  (exempt); REPL transcripts are intentionally flush-left (exempt). Genuine
  free-form pseudocode (no ``:`` suite) is not judgeable and falls out for free.
  This abstraction subsumes "language/style-aware" judging without a brittle
  language classifier, and (validated against the corpus) still FAILS genuinely
  mangled Python such as AIOS at 0.44 — "ignore flat code" alone does not excuse
  it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

# --- Positive code-ID ------------------------------------------------------

# Strong + common code keywords. Equations/chemistry/LaTeX contain none of these
# (validated: Hybrid-EV's 15 equation chunks yield 0 structural matches), so they
# are excluded from the code population.
_CODE_KEYWORDS = re.compile(
    r"\b(?:def|class|import|from|return|yield|lambda|self|async|await|try|"
    r"except|finally|with|elif|raise|assert|print|for|while|if|else|None|"
    r"True|False)\b"
)
# Code punctuation that does not occur in prose math (single ``=`` and bare
# parens-with-content are deliberately excluded so ``V_oc = a + b`` does not
# match).
_CODE_PUNCT = re.compile(r"\(\)|\)\s*:|->|=>|==|!=|<=|>=|:=|\+=|-=|::|\bself\.")
_REPL = re.compile(r"(?m)>>>|^\s*\.\.\.\s")

# Python-style block opener: a suite header line ending in ``:``.
_BLOCK_OPENER = re.compile(
    r"(?m)^\s*(?:async\s+)?(?:def|class|if|elif|else|for|while|try|except|"
    r"finally|with)\b.*:\s*$"
)

# R3 floor (unchanged from the legacy gate — the population/method is what is
# fixed, not the threshold).
DEFAULT_INDENT_FIDELITY_FLOOR = 0.90
# Meaningfulness floor: do not gate on a statistically tiny judgeable population
# (mirrors the existing ">= 3 code chunks" guards in both gate scripts).
DEFAULT_MIN_JUDGEABLE = 3
# Policy B density floor (docs/DECISIONS.md "R3 Code-Indentation Gate Redesign").
# A document whose judgeable-code share reaches this fraction has non-incidental
# code; a sub-floor failure is advisory (per-chunk flag + WARN) so a prose
# document's good content is not discarded over one or two mangled snippets (F3).
DEFAULT_HARDFAIL_DENSITY = 0.04


def has_code_structure(text: str) -> bool:
    """Positive code-ID: keywords OR code punctuation OR REPL prompt."""
    s = text or ""
    if not s.strip():
        return False
    return bool(_CODE_KEYWORDS.search(s) or _CODE_PUNCT.search(s) or _REPL.search(s))


def is_repl(text: str) -> bool:
    return bool(_REPL.search(text or ""))


def is_judgeable(text: str) -> bool:
    """True for multi-line code that syntactically requires nesting.

    REPL transcripts are excluded (flush-left by design). Flat/single-statement
    code and free-form pseudocode (no ``:`` suite, no brace block) are excluded
    because they have no nesting to assess.
    """
    s = text or ""
    if is_repl(s):
        return False
    lines = [ln for ln in s.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    if _BLOCK_OPENER.search(s):
        return True
    return "{" in s and "}" in s


def indentation_ok(text: str) -> bool:
    """For a judgeable chunk: is real nesting depth present?

    A correctly-extracted suite has at least two distinct indent depths with a
    non-zero level (the body is indented under its opener). A block flattened
    onto a single indent level (e.g. ``class Scheduler: def __init__(...) ...``)
    collapses to one depth and fails.
    """
    s = text or ""
    lines = [ln for ln in s.splitlines() if ln.strip()]
    depths = {len(ln) - len(ln.lstrip(" \t")) for ln in lines}
    return len(depths) > 1 and max(depths) > 0


def is_code_population(chunk: Dict[str, Any]) -> bool:
    """True if the chunk is in the code population (either producer seam)."""
    if chunk.get("modality") == "code":
        return True
    if chunk.get("modality") == "text":
        md = chunk.get("metadata") or {}
        ct = str(md.get("chunk_type") or "").lower()
        cc = str(md.get("content_classification") or "").lower()
        return ct == "code" or cc == "code"
    return False


@dataclass
class CodeQuality:
    n_population: int = 0  # code-typed chunks (either seam)
    n_struct: int = 0  # of those, with positive code structure
    n_math_excluded: int = 0  # code-typed but no code structure (equations)
    n_repl: int = 0  # REPL transcripts (auto-pass, exempt)
    n_flat_exempt: int = 0  # structural but not judgeable (no nesting)
    n_judgeable: int = 0
    n_judgeable_fail: int = 0
    degraded_ids: List[str] = field(default_factory=list)
    total_chunks: int = 0

    @property
    def indentation_fidelity(self) -> float:
        if self.n_judgeable == 0:
            return 1.0
        return (self.n_judgeable - self.n_judgeable_fail) / self.n_judgeable

    @property
    def judgeable_density(self) -> float:
        if self.total_chunks == 0:
            return 0.0
        return self.n_judgeable / self.total_chunks


def code_quality(chunks: Sequence[Dict[str, Any]]) -> CodeQuality:
    """Compute the R3 code-indentation metric over a chunk list.

    ``chunks`` is the content chunks (the ``ingestion_metadata`` record must
    already be filtered out by the caller).
    """
    cq = CodeQuality(total_chunks=len(chunks))
    for ch in chunks:
        if not is_code_population(ch):
            continue
        cq.n_population += 1
        content = ch.get("content") or ""
        if not has_code_structure(content):
            cq.n_math_excluded += 1
            continue
        cq.n_struct += 1
        if is_repl(content):
            cq.n_repl += 1
            continue
        if not is_judgeable(content):
            cq.n_flat_exempt += 1
            continue
        cq.n_judgeable += 1
        if not indentation_ok(content):
            cq.n_judgeable_fail += 1
            cid = str(ch.get("chunk_id") or "")
            if cid and len(cq.degraded_ids) < 20:
                cq.degraded_ids.append(cid)
    return cq


def gate_verdict(
    cq: CodeQuality,
    fidelity_floor: float = DEFAULT_INDENT_FIDELITY_FLOOR,
    min_judgeable: int = DEFAULT_MIN_JUDGEABLE,
    hardfail_density: float = DEFAULT_HARDFAIL_DENSITY,
) -> str:
    """Policy B verdict for the R3 code-indentation gate.

    Returns one of:
        "pass" — fidelity at/above floor, or too few judgeable chunks to gate.
        "warn" — fidelity below floor but judgeable-code density below the
                 hard-fail floor (incidental code → per-chunk flag + WARN).
        "fail" — fidelity below floor AND judgeable-code density at/above the
                 hard-fail floor (non-incidental broken code → hard-fail).

    The metric (fidelity) is reported identically in every case; only the
    document-level escalation differs. This is strictly more enforcement than
    the legacy dead gate, not a relaxation. See the DECISIONS.md entry.
    """
    if cq.n_judgeable < min_judgeable:
        return "pass"
    if cq.indentation_fidelity >= fidelity_floor:
        return "pass"
    if cq.judgeable_density >= hardfail_density:
        return "fail"
    return "warn"
