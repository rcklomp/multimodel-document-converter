#!/usr/bin/env python3
"""Code-repo-diff fidelity oracle (PLAN_FIDELITY_ORACLE_FIRST_V1).

Measures extraction fidelity of code chunks against an AUTHORITATIVE author
repository (ground truth), not a parse-presence proxy.

Why this exists: the shipping code oracle (`scripts/f1_oracle.py`) scores
`ast.parse` success. Parsing is BLIND to the failure class this project has
chased for weeks - a code line whose CONTENT is correct but whose leading
indentation was stripped by the extractor still parses fine as a flat
statement, and text edit-distance whitespace-normalizes it away too. So both
incumbent signals can report a de-indented (semantically broken) program as
high fidelity. This oracle diffs each extracted code line against the real
source and isolates exactly that:

  verbatim_fidelity   = chunk code lines present in ground truth WITH their
                        leading indentation
  deindented_fidelity = chunk code lines present in ground truth after BOTH
                        sides are left-stripped
  indentation_gap     = deindented - verbatim
                        (lines whose content is right but indentation is wrong
                        = the silent fidelity killer ast.parse cannot see)
  content_loss        = 1 - deindented
                        (lines absent even ignoring indentation: OCR/engine
                        corruption, or simply not from this repo)

REPL/console transcript chunks (>>> / ... prompts + their output) are NOT in
.py source files, so they are bucketed separately and EXCLUDED from the
source-diff denominator (counting them as "content loss" would be dishonest).

Deterministic, library-light (stdlib only). Reads JSONL produced by the
shipping CLI. NOT a gate edit; a standalone measurement instrument.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))
import _code_quality as cq  # noqa: E402

_MIN_SIG_LEN = 4  # stripped length below which a line is too trivial to attribute

# Book callout markers appended to code lines, e.g. "...  # 1", "...  # <3>", or the
# unicode circled forms O'Reilly/Packt use in print ("...  # ①"). The repo source
# has none of these, so they are book-vs-repo divergence, not extraction error.
# Normalizing them is the honest move for an ABSOLUTE fidelity number.
_CIRCLED = "①-⑳❵-❿⓪⓵-⓾➀-➓"
_CALLOUT_RE = re.compile(
    r"\s*#\s*<?\d{1,3}>?\s*$"            # "# 3" / "# <3>"
    r"|\s*#?\s*[" + _CIRCLED + r"]+\s*$"  # "# ①" / trailing "①" (no #)
)


def _norm_callout(line: str, strip: bool) -> str:
    return _CALLOUT_RE.sub("", line) if strip else line


# True-corruption refinement: a code chunk can carry NON-code lines (interleaved
# prose, page furniture like "12 | Chapter 1: ..."). Those are never in a .py repo,
# so counting them as "content loss" overstates corruption. ``_is_code_line`` keeps
# only plausibly-code lines so the refined denominator isolates token corruption
# (a code line absent even de-indented = real OCR/engine corruption or edition diff,
# not prose). Conservative: keep anything with code punctuation/keywords; drop
# multi-word natural-language sentences and page furniture.
_FURNITURE_RE = re.compile(
    r"^\s*\d{1,4}\s*[|│｜]"               # "12 | ..." running header/footer
    r"|^(?:chapter|figure|table|example|listing|part|section)\b",
    re.IGNORECASE,
)
_CODE_SIGNAL_RE = re.compile(
    r"[=(){}\[\]:;]|->|=>|::|\bself\.|\.\w+\("
    r"|^\s*(?:def|class|import|from|return|if|for|while|with|try|except|finally|"
    r"elif|else|raise|yield|assert|lambda|async|await|print|pass|break|continue)\b"
)


def _is_code_line(stripped: str) -> bool:
    s = stripped
    if not s:
        return False
    if _FURNITURE_RE.match(s):
        return False
    if _CODE_SIGNAL_RE.search(s):
        return True
    # No code signal: treat a multi-word sentence as prose (exclude).
    alpha_words = [w for w in s.split() if w.replace("'", "").isalpha()]
    if len(alpha_words) >= 5:
        return False
    if len(alpha_words) >= 3 and s[-1:] in ".!?":
        return False
    return True


def _iter_chunks(jsonl: Path):
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("modality") is None and d.get("schema_version"):
            continue  # ingestion_metadata header
        yield d


def _strip_fences(content: str) -> list[str]:
    out = []
    for ln in (content or "").splitlines():
        if ln.lstrip().startswith("```"):
            continue
        out.append(ln.rstrip("\n"))
    return out


def build_ground_truth(repo: Path, strip_callouts: bool = False) -> tuple[set[str], set[str], int]:
    """Index all .py source lines. Returns (verbatim_set, deindented_set, n_files)."""
    verbatim: set[str] = set()
    deindent: set[str] = set()
    n = 0
    for py in repo.rglob("*.py"):
        try:
            text = py.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        n += 1
        for raw in text.splitlines():
            line = _norm_callout(raw.rstrip(), strip_callouts)
            s = line.strip()
            if len(s) < _MIN_SIG_LEN:
                continue
            verbatim.add(line)
            deindent.add(s)
    return verbatim, deindent, n


def score(
    jsonl: Path, verbatim: set[str], deindent: set[str], strip_callouts: bool = False
) -> dict[str, Any]:
    sig = 0
    v_hit = 0
    d_hit = 0
    repl_chunks = 0
    code_chunks = 0
    # Refined (code-shaped lines only) — isolates TRUE token corruption.
    csig = 0
    cv_hit = 0
    cd_hit = 0
    noncode_lines = 0
    examples_indent_loss: list[str] = []
    examples_content_loss: list[str] = []  # code-shaped only (true corruption)

    for ch in _iter_chunks(jsonl):
        if not cq.is_code_population(ch):
            continue
        code_chunks += 1
        content = ch.get("content") or ""
        if cq.is_repl(content):
            repl_chunks += 1
            continue
        for raw in _strip_fences(content):
            line = _norm_callout(raw.rstrip(), strip_callouts)
            s = line.strip()
            if len(s) < _MIN_SIG_LEN:
                continue
            sig += 1
            v = line in verbatim
            d = s in deindent
            if v:
                v_hit += 1
            if d:
                d_hit += 1
            if d and not v and len(examples_indent_loss) < 25:
                examples_indent_loss.append(line)
            # Refined pass: only score plausibly-code lines.
            if _is_code_line(s):
                csig += 1
                if v:
                    cv_hit += 1
                if d:
                    cd_hit += 1
                if not d and len(examples_content_loss) < 25:
                    examples_content_loss.append(line)
            else:
                noncode_lines += 1

    vf = v_hit / sig if sig else 0.0
    df = d_hit / sig if sig else 0.0
    cvf = cv_hit / csig if csig else 0.0
    cdf = cd_hit / csig if csig else 0.0
    return {
        "code_chunks": code_chunks,
        "repl_chunks_excluded": repl_chunks,
        "significant_lines": sig,
        "verbatim_fidelity": vf,
        "deindented_fidelity": df,
        "indentation_gap": df - vf,
        "content_loss": 1.0 - df,
        # refined (code-shaped lines only)
        "codeline_lines": csig,
        "noncode_lines_excluded": noncode_lines,
        "codeline_verbatim": cvf,
        "codeline_deindented": cdf,
        "codeline_indentation_gap": cdf - cvf,
        "codeline_corruption": 1.0 - cdf,
        "_indent_loss_examples": examples_indent_loss,
        "_content_loss_examples": examples_content_loss,
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", required=True, help="extraction ingestion.jsonl")
    ap.add_argument("--repo", required=True, help="authoritative author code repo (cloned)")
    ap.add_argument("--label", default=None, help="display label")
    ap.add_argument("--examples", action="store_true", help="print sample mismatched lines")
    ap.add_argument(
        "--strip-callouts",
        action="store_true",
        help="normalize trailing publisher callout markers (# 1 / # <3>) on both "
        "sides before matching - for an honest ABSOLUTE number (does not affect "
        "the engine-comparative ranking)",
    )
    args = ap.parse_args(argv)

    repo = Path(args.repo)
    jsonl = Path(args.jsonl)
    verbatim, deindent, n_files = build_ground_truth(repo, args.strip_callouts)
    r = score(jsonl, verbatim, deindent, args.strip_callouts)

    label = args.label or jsonl.parent.name
    print(f"=== CODE-REPO ORACLE: {label} ===")
    print(f"  ground truth: {n_files} .py files, {len(verbatim)} distinct source lines")
    print(f"  code chunks: {r['code_chunks']}  (REPL excluded: {r['repl_chunks_excluded']})")
    print(f"  significant code lines diffed: {r['significant_lines']}")
    print(f"  verbatim_fidelity   (content+indent): {r['verbatim_fidelity']:.3f}")
    print(f"  deindented_fidelity (content only):   {r['deindented_fidelity']:.3f}")
    print(f"  indentation_gap (silent killer):      {r['indentation_gap']:.3f}")
    print(f"  content_loss (corruption/off-repo):   {r['content_loss']:.3f}")
    print(f"  --- refined: code-shaped lines only (non-code excluded: {r['noncode_lines_excluded']}) ---")
    print(f"  codeline_verbatim   (content+indent): {r['codeline_verbatim']:.3f}")
    print(f"  codeline_deindented (content only):   {r['codeline_deindented']:.3f}")
    print(f"  codeline_indentation_gap:             {r['codeline_indentation_gap']:.3f}")
    print(f"  codeline_corruption (TRUE token loss):{r['codeline_corruption']:.3f}  "
          f"(over {r['codeline_lines']} code lines)")
    if args.examples:
        print("  --- indentation-loss lines (content right, indent wrong) ---")
        for ln in r["_indent_loss_examples"]:
            print(f"    |{ln}")
        print("  --- TRUE corruption candidates (code-shaped, absent even de-indented) ---")
        for ln in r["_content_loss_examples"]:
            print(f"    |{ln}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
