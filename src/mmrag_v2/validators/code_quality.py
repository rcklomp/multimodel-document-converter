"""Shared code-quality metric for the R3 code-indentation gate.

Single source of truth for the R3 detector. Imported by the audit-time gate
(``qa_conversion_audit.py`` hard gate, ``qa_semantic_fidelity.py`` advisory, via
the ``scripts/_code_quality.py`` back-compat shim) AND by the conversion-time
specialist re-extraction (``mmrag_v3.processor`` PLAN §5.4 consumer 1), so the
flag that gates a doc is the SAME detector that triggers its repair. It replaces
the dead metric that scored only ``modality=="text"`` chunks and was blind to the
V3 pipeline's ``modality=="code"`` promotion. See
``docs/PLAN_R3_CODE_GATE_REDESIGN.md``.

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

# Collapsed nested suite: a multi-line block flattened ONTO ONE LINE so the
# suite ``:`` is no longer at end-of-line and ``_BLOCK_OPENER`` misses it — the
# WORST extraction outcome (e.g. MinerU-1.2B on a sub-threshold code page renders
# ``for n in needles:\n    if n in haystack:\n        found += 1`` as the single
# line ``for n in needles: if n in haystack: found += 1``). Detected by TWO
# block-keyword suite colons on one line (flattened nesting). Requires two so a
# legitimate one-line compound statement (``if x: return y``) is NOT flagged;
# dict literals / comprehensions / lambda carry no block-keyword suite colon.
_SUITE_COLON = r"(?:def|class|if|elif|for|while|try|except|with)\b[^:\n]*:"
_COLLAPSED_NESTED = re.compile(r"(?m)^\s*(?:async\s+)?" + _SUITE_COLON + r".*?" + _SUITE_COLON)

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
# Accept-with-remark floor (docs/DECISIONS.md "R3 Accept-With-Remark Band",
# user-signed 2026-06-17). For code-bearing docs, indentation fidelity in
# [ACCEPT_FLOOR, INDENT_FIDELITY_FLOOR) is ACCEPTED but flagged with a remark
# (advisory WARN, non-blocking) instead of hard-failing: a code book degrades
# gracefully and the realised-quality drop is tracked for targeted follow-up.
# Below this floor the original density-gated hard-fail still applies.
DEFAULT_ACCEPT_FIDELITY_FLOOR = 0.65


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
    because they have no nesting to assess. A nested suite FLATTENED onto one
    line (``_COLLAPSED_NESTED``) IS judgeable — it is mangled code whose nesting
    was destroyed, so it must be scored (and will fail ``indentation_ok``), not
    slip through as "flat".
    """
    s = text or ""
    if is_repl(s):
        return False
    # A collapsed nested suite is judgeable regardless of line count: flattening
    # destroyed its nesting, so a block mangled onto a SINGLE line
    # (``def f(x): if x: return x``) must still be scored — it cannot pass
    # ``indentation_ok``. Checked before the line-count guard, which would
    # otherwise drop it as "flat". The regex requires TWO block-keyword suite
    # colons, so a legitimate one-line compound stmt is not flagged here.
    if _COLLAPSED_NESTED.search(s):
        return True
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


def _norm_ws(text: str) -> str:
    """Whitespace-stripped content key for duplicate detection."""
    return re.sub(r"\s+", "", text or "")


def _duplicates_primary(norm: str, primary: List[str], window: int = 40) -> bool:
    """True if ``norm`` (a whitespace-stripped text-as-code chunk) duplicates a
    cleanly-extracted ``modality=code`` chunk.

    The production pipeline can emit a code listing TWICE — once as a clean,
    fully-indented ``modality=code`` chunk (the VLM's code element) and again as
    redundant flush-left ``modality=text`` fragments that a downstream text
    classifier stamps ``content_classification=code``. Such a fragment is NOT an
    independent code-indentation failure (the code WAS extracted with
    indentation, in the code chunk) — it is output redundancy, a separate
    concern. Detect it by a long contiguous overlap (tolerant of a leaked page
    header prefixing the fragment).

    DUAL-LAYER NOTE: the production pipeline also drops these at extraction time
    (``BatchProcessor._apply_recovery_vs_primary_dedup``). This is the AUDIT-time
    backstop so files already on disk, or any other producer's output, are scored
    correctly regardless. Two layers, one domain fact — keep them consistent. The
    two use DIFFERENT algorithms by design (this: substring-window; production:
    token-overlap), so they are not required to agree on every edge case; the
    shared contract (both must catch the canonical recovery-duplicate) is enforced
    by ``tests/test_dual_layer_recovery_dedup.py`` (PR #4 Finding 4 follow-up).
    """
    if not norm:
        return False
    if len(norm) <= window:
        return any(norm in p for p in primary)
    step = max(1, (len(norm) - window) // 4)
    for i in range(0, len(norm) - window + 1, step):
        w = norm[i : i + window]
        if any(w in p for p in primary):
            return True
    return False


@dataclass
class CodeQuality:
    n_population: int = 0  # code-typed chunks (either seam)
    n_struct: int = 0  # of those, with positive code structure
    n_math_excluded: int = 0  # code-typed but no code structure (equations)
    n_repl: int = 0  # REPL transcripts (auto-pass, exempt)
    n_flat_exempt: int = 0  # structural but not judgeable (no nesting)
    n_duplicate_excluded: int = 0  # text-as-code that duplicates clean code
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
    # Clean primary code (the VLM's authoritative code elements), normalized, so
    # redundant text-as-code fragments duplicating it are not scored as failures.
    primary = [_norm_ws(ch.get("content") or "") for ch in chunks if ch.get("modality") == "code"]
    primary = [p for p in primary if p]
    for ch in chunks:
        if not is_code_population(ch):
            continue
        content = ch.get("content") or ""
        # A modality=text chunk that duplicates a cleanly-extracted code chunk is
        # output redundancy, not an independent code-indentation failure.
        if ch.get("modality") == "text" and _duplicates_primary(_norm_ws(content), primary):
            cq.n_duplicate_excluded += 1
            continue
        cq.n_population += 1
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
    accept_floor: float = DEFAULT_ACCEPT_FIDELITY_FLOOR,
) -> str:
    """Policy B verdict for the R3 code-indentation gate.

    Returns one of:
        "pass" — fidelity at/above the clean floor, or too few judgeable chunks.
        "warn" — accepted but flagged. Two cases: (a) the accept-with-remark band,
                 fidelity in [accept_floor, fidelity_floor) (user-signed 2026-06-17:
                 a code book degrades gracefully rather than hard-failing, and the
                 quality drop is tracked for targeted follow-up); or (b) fidelity
                 below accept_floor but judgeable-code density below the hard-fail
                 floor (incidental code).
        "fail" — fidelity below accept_floor AND judgeable-code density at/above the
                 hard-fail floor (non-incidental, genuinely broken code).

    The metric (fidelity) is reported identically in every case; only the
    document-level escalation differs. See the DECISIONS.md entries "R3
    Code-Indentation Gate Redesign" and "R3 Accept-With-Remark Band".
    """
    if cq.n_judgeable < min_judgeable:
        return "pass"
    if cq.indentation_fidelity >= fidelity_floor:
        return "pass"
    # Accept-with-remark band: code present, fidelity below the clean floor but at or
    # above the accept floor — accepted (non-blocking WARN), drop tracked.
    if cq.indentation_fidelity >= accept_floor:
        return "warn"
    # Below the accept floor: genuinely broken. Hard-fail only when non-incidental.
    if cq.judgeable_density >= hardfail_density:
        return "fail"
    return "warn"
