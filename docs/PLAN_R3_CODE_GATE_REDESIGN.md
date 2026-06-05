# PLAN — Redesign the R3 Code-Quality Gate for the V3 / MinerU Pipeline

Status: SIGNED + IMPLEMENTED (2026-06-05). Branch `v3.1-extraction-hardening`.
Author: architecture session following the brief in the session prompt.

**Decisions taken (user-signed 2026-06-05):** §6 policy = **Option B**
(per-chunk flag + density-gated hard-fail). Extraction (Thread 2) = **deferred to
separate sign-off**. Threads 1 (metric) + the policy are SHIPPED:
`scripts/_code_quality.py`, gate wiring in `qa_conversion_audit.py` (hard) and
`qa_semantic_fidelity.py` (advisory), `tests/test_code_quality_metric.py` +
`tests/test_code_indentation_audit_gate.py`, and the `DECISIONS.md` entry "R3
Code-Indentation Gate Redesign". The optional upstream equation guard in
`mineru_native.py` (demote VLM-mislabelled math out of the code lane) is also
SHIPPED (e4196bf). Thread 2 PROPER — the Qwen-for-code extraction route
(AIOS 0.33 -> ~1.00, F5) — remains open for separate sign-off.

This plan supersedes the "Proposed fix (needs sign-off)" stub in
`docs/PLAN_VLM_EVAL.md` §16.2. It is the deliberate requirement-change record the
anti-weakening rule demands before any code touches a gate.

---

## 0. TL;DR

The code-indentation quality gate (R3) is **dead** — it scores the wrong chunk
population and, because of that same blindness, mis-routes documents so the hard
gate never activates. I verified this end to end: on AIOS the audit prints
`indentation_fidelity: 0.00` and still only WARNs, while the 18 real
`modality=code` chunks (the actual mangled code bodies) are invisible to it.

The fix is **not** "include `modality=code` chunks" (the §16.2 stub) and **not**
a threshold tweak. It is three coupled changes, in order:

1. **METRIC** — replace the dead metric with an honest one: a single shared
   module that (a) measures the real code population (`modality=code` + legacy
   text-code), (b) positively identifies *code* (so equations/formulas are
   excluded, not heuristically un-excluded), and (c) judges only what is
   *judgeable* (multi-line blocks that syntactically require nesting), exempting
   flat/single-statement code and REPL transcripts. Validated against the corpus
   with every verdict flip justified (§5).
2. **EXTRACTION** — raise the achievable quality ceiling for code-dense pages by
   routing them to the Qwen lane (F5: Qwen extracted the same AIOS code at 1.00
   vs MinerU's 0.22), MinerU everywhere else. Measured by the new metric.
3. **GATING POLICY** — decide the proportionate response (per-chunk flag +
   document-level verdict) once §1+§2 reveal the achievable ceiling. Written as a
   `DECISIONS.md` entry, not slipped in.

Threads 2 and 3 depend on Thread 1: the metric is the measuring instrument.

---

## 1. Verified diagnosis (re-measured this session)

All measurements reproduced with
`/Users/ronald/miniforge3/envs/mmrag-v2/bin/python` over the on-disk corpus.

### F1 — The gate is DEAD, via a *dual* defect (confirmed, and worse than stated)

The brief frames F1 as a "modality seam". It is actually **two** compounding
defects in `scripts/qa_conversion_audit.py` (the hard gate) — and a third script
(`qa_semantic_fidelity.py`) that is advisory-only and therefore never blocked
anything regardless:

1. **Wrong population.** The code metric counts only chunks with `modality ==
   "text"` whose `chunk_type`/`content_classification == "code"`
   (`_is_code_chunk`, the `if modality != "text": continue` at line 390). The V3
   pipeline promotes real code to `modality == "code"` — so the actual code
   bodies never enter the metric.
2. **Blind content-type routing.** `_classify_content_type` (line 588) computes
   `code_ratio = code_chunks / text_chunks` using that same text-only count. With
   the real code excluded, the ratio collapses below `0.15`, the document is
   classed `mixed_prose` or `non_code`, and the code gate is set to `warn` or
   `None` — so the *hard* gate is unreachable even on a genuine code failure.

Measured on `output/mineru_scale_aios2` (AIOS, post the a47f0c4 fencing fix):

```
modality counts: {'text': 138, 'table': 7, 'code': 18}
modality=text code (what gate sees): 9     <- metric population
modality=code (real code):           18     <- invisible to gate
```

Live audit verdict on the same file:

```
Content: mixed_prose
CODE:  WARN [mixed_prose]
  code_chunks: 9   indentation_fidelity: 0.00
  ⚠ indent_fidelity=0.00 (<0.9)
AUDIT_FAIL (HEADING)        <- fails on HEADING, NOT code
```

The gate reads **0.00 indentation fidelity and only warns.** R3 (a HARD pipeline
contract) is silently unenforced. `qa_semantic_fidelity.py` has the identical
text-only blindness *and* `main()` returns 0 unconditionally — the orchestrator
downgrades it to `SCRIPT_ADVISORY_FAIL` (an allowed advisory). So neither script
can hard-fail on code today.

Corpus scale: 353 `modality=code` chunks exist across `output/**`; the
`modality=text` code population the metric scores is ~0 on most documents.

### F2 — No reliable signal separates "code book" from "paper with code" (confirmed)

Re-measured count-ratio and char-ratio for FluentPython (real code book) vs AIOS
(paper). They overlap on every axis; AIOS even has higher code-*char* density on
some page ranges. `profile_type` is noisy (AIOS is stamped
`academic_whitepaper`; FluentPython has been seen mis-stamped). **There is no
clean automatic "is this a code book" signal — so the redesign must not depend on
making that classification.** (Resolution: it doesn't; see §3.)

### F3 — Code is *always* minority content by character (confirmed)

Even FluentPython is 6-19% code by character. No document exists where mangled
code makes the *whole* document worthless. This is the core argument for the
policy thread: a whole-document hard-fail on a minority content type discards
good prose, tables, and figures.

### F4 — AIOS's failure is GENUINE, real (mislabelled) Python — not loose pseudocode

The prior session's §16 repeatedly calls AIOS "pseudocode". **That label is
wrong and it matters.** The extracted chunks are real Python from the paper's
listings: `class SysCall(Thread):`, `def __init__(self, ...)`,
`threading.Event()`, `@abstractmethod`. MinerU2.5-1.2B (the smallest variant)
recognizes them imperfectly: `self.` read as `self(`/`self/`, `created_time` as
`createed_time`, and one `class Scheduler` class flattened onto a single line.
Measured `modality=code` indentation fidelity after fencing: **0.22** (simple
rule) / **0.44** (judgeable-only rule). Both are far below 0.90 — the failure is
real and a correct metric MUST fail it. Calling it "pseudocode" to grant it loose
rules would hide a real defect (explicitly forbidden by the brief).

### F5 — The Qwen lane extracted the SAME AIOS code at 1.00 (confirmed)

`output/crucible_full_run/.../AIOS_academic` (Qwen path) vs
`output/mineru_scale_aios2` (MinerU path), same source listings:

| chunk | MinerU2.5-1.2B | Qwen3-VL-8B |
|---|---|---|
| `class SysCall(Thread)` | `self(agent_name`, `self/response`, `self.set.pid` | `self.agent_name`, `self.response`, `self.set_pid()` (clean) |
| `class LLMCore(ABC)` | partial indent, `""` docstrings | full indent, `"""` docstrings |
| `class Scheduler` | flattened to 3 jammed lines | properly nested |

Judgeable-fidelity: MinerU **0.44**, Qwen **1.00**. A real, no-new-model
extraction fix exists.

### F6 — A positive code-ID metric works and excludes math (confirmed + improved)

Positive structure = code keywords OR code punctuation OR `>>>` REPL prompt.
Measured `modality=code` indentation fidelity with this metric:

| doc | modality=code | struct-code | LaTeX/unicode math | indent fidelity |
|---|---:|---:|---:|---:|
| AIOS-MinerU (bad) | 18 | 18 | 4 | **0.22** FAIL |
| AIOS-Qwen (good) | 24 | 24 | 1 | **1.00** PASS |
| FluentPython (subset) | 22 | 22 | 0 | **0.95** PASS |
| FluentPython (p60-84) | 30 | 30 | 0 | **0.97** PASS |
| Hybrid-EV (equations) | 15 | **0** | 11 | n/a (excluded) |

Hybrid-EV's 15 equation chunks (`Pb+PbO₂+2H₂SO₄ ↔ ...`, `V_oc = a + b × DOD`,
LaTeX) carry `original_vlm_type: code` — **the extractor VLM itself mislabels
equations as code** (verified in metadata; not the chunker's doing). Positive
code-ID drops all 15 from the code population, removing the false-fail class
*regardless of which extractor produced the file*. A "exclude unicode math"
heuristic was previously falsified (it missed LaTeX); positive code-ID is the
robust direction.

---

## 2. Reframing the three orthogonal questions

The current gate conflates: (1) how much code, (2) is code the doc's purpose,
(3) is the code well-extracted. The redesign's key realization:

> **Once the metric is honest, question (2) stops mattering.** A document with
> well-extracted code passes the metric whether it is a textbook or a paper. A
> document with mangled code fails it the same way. The only thing question (2)
> ever decided was hard-vs-warn — and F2 proves it cannot be answered reliably.
> So we stop trying to answer it and gate on the *measured quality of the
> judgeable code that is actually present*.

This dissolves F2 (no classification needed) and sets up F3 (policy granularity).

---

## 3. Resolved architecture questions

**Q: Language/style-aware, or "judge only what's judgeable"?**
RESOLVED: **judge only what's judgeable** — the cleaner abstraction, and it
*subsumes* language-awareness without a brittle language classifier.

- A chunk is *judgeable* iff it is multi-line AND contains a syntactic
  block-opener (a Python `:`-terminated suite header, or a brace block `{...}`),
  and is not a pure REPL transcript.
- Judgeable chunks must show real nesting (>1 distinct indent depth, max depth
  >0). This is what catches AIOS's flattened `class Scheduler`.
- Flat / single-statement code (no block-opener, or one logical line) has no
  nesting to assess -> **exempt** (PASS contribution removed from denominator).
- REPL transcripts (`>>>`/`...`) are auto-pass (doctest lines are intentionally
  flush-left).
- **Pseudocode falls out for free.** Genuine free-form pseudocode (numbered
  steps, `FOR..END`, no `:` suite) is not judgeable -> exempt -> the brief's
  "loose rule" with no special code path. (I searched the entire corpus: zero
  pseudocode-style code chunks exist, so this is defensive, not load-bearing.)
- **AIOS is NOT rescued by this.** Measured: judgeable-only still scores AIOS at
  **0.44** (9 judgeable blocks, ~4 pass), a clear FAIL. Its blocks are real
  multi-line Python suites that *require* indentation — exactly the judgeable
  class. Verified against the brief's warning that "ignore flat code" alone must
  not excuse AIOS.

**Q: Should code quality EVER hard-fail a whole doc (given F3)?**
This is the §6 policy DECISION — surfaced to the user, not chosen here. My
recommendation: per-chunk flag + a document-level verdict that is **hard-fail
only above a code-density floor**, advisory below it. Rationale in §6.

**Q: Unify the two divergent gate scripts?**
RESOLVED: extract one shared module `scripts/_code_quality.py` (single source of
truth for *the metric*). Both scripts import it; each keeps its own gate wiring
(`qa_conversion_audit.py` = hard, `qa_semantic_fidelity.py` = advisory). This
removes the divergence without a wholesale merge and keeps diffs surgical —
`qa_conversion_audit.py` has pre-existing ruff/black drift and must not be
mass-reformatted (same discipline the prior session used for
`tests/test_v3_security.py`).

**Q: Fix equation-as-code upstream in `mineru_native.py`?**
RESOLVED: **metric-level exclusion is primary and required; upstream typing is
optional, secondary hardening.** The gate must be correct for files *already on
disk* and for *any* extractor's output — and F6 shows the VLM mislabels
equations as code at the source, so the gate can never trust the type. Positive
code-ID at the metric is therefore mandatory. Separately, MinerU emits a distinct
`interline_equation` type that the converter currently sends to TEXT (good);
*VLM-mislabelled* equations are the leak, and they ride the `code` type, so an
upstream "is this content actually math, retype to TEXT" guard in
`mineru_native._mineru_element_to_element` is a clean MinerU-only improvement but
does not remove the need for the metric guard. Proposed as an optional Thread-2
sub-item, not a substitute.

---

## 4. Proposed design

### Thread 1 — METRIC (do first)

New module `scripts/_code_quality.py`:

```
code_population(rows)      -> chunks where modality=="code"
                              OR (modality=="text" AND chunk_type/cc=="code")
has_code_structure(text)   -> keywords | code-punct | REPL   (positive code-ID)
classify(text)             -> "math" | "repl" | "flat" | "judgeable"
indentation_ok(text)       -> real nesting depth present (for judgeable only)
code_quality(rows)         -> dataclass: n_code, n_struct, n_judgeable,
                              n_math_excluded, fidelity (judgeable pass / judgeable,
                              =1.0 when no judgeable chunks)
```

Wiring:
- `qa_conversion_audit.py`: replace the inline `_is_code_chunk` counting and the
  `code_with_indent` loop with `code_quality(...)`. Feed the corrected code count
  into `_classify_content_type` so routing is no longer blind. **The CODE gate
  becomes: fidelity over judgeable code, gated per §6 policy.** Surgical edits
  only; no reformat of untouched lines.
- `qa_semantic_fidelity.py`: same metric via the shared module; remains advisory.
- Report `n_math_excluded` and `n_judgeable` in both outputs so an operator can
  see *why* a verdict landed (e.g. "0 judgeable code chunks -> code gate n/a").

### Thread 2 — EXTRACTION (do second, measured by Thread 1)

Primary candidate (F5): a **MinerU-default + Qwen-for-code per-page route**. The
`router.py` monospace-ratio code signal (threshold 0.10) already identifies
code-dense pages; today the MinerU default bypasses the per-page router (every
page -> MinerU). Proposal: when the MinerU engine is active, route pages whose
`mono_ratio >= threshold` to the Qwen `VlmNativeEngine` and the rest to MinerU,
then merge. Acceptance = the Thread-1 metric on a re-run AIOS rises to PASS while
golden/soak docs stay PASS. Requires the M5 Qwen server co-available with the
GX10 MinerU server.

Optional sub-item: upstream equation guard in `mineru_native.py` (retype
math-only `code` content to TEXT) — MinerU-only hardening, validated by the same
metric (Hybrid-EV must keep 0 false fails; nothing else regresses).

Fallback if the hybrid route is not pursued now: accept MinerU's code ceiling and
rely on the §6 per-chunk flag to mark AIOS-class code for later re-extraction.
This is why Thread 2 can be deferred without leaving R3 dead — Thread 1 + Thread
3 already make the gate honest and live.

### Thread 3 — GATING POLICY (do last; §6)

---

## 5. Validation plan (every verdict flip justified)

The metric change is validated against the on-disk corpus before commit. Expected
verdict table (to be reproduced by a committed test fixture, mirroring
`tests/test_tabular_audit_gate.py`'s subprocess-on-synthetic-JSONL pattern):

| doc | today | after metric | flip justified by |
|---|---|---|---|
| AIOS-MinerU | code WARN @0.00 (dead) | code **FAIL** @0.44 | F4 — real mangled Python, must fail |
| AIOS-Qwen | n/a | code **PASS** @1.00 | F5 — clean extraction |
| FluentPython (×3) | non_code/ignored | code **PASS** @0.95-0.97 | real code, well extracted |
| Hybrid-EV | code chunks counted | code **n/a** (0 judgeable) | F6 — equations excluded, no false fail |
| golden 6/6, soak 7/7 | PASS | **PASS** (unchanged) | no code, or code already clean |

Plus deterministic unit tests for the shared module: positive code-ID on Python /
exclusion of unicode+LaTeX math / flat-exempt / REPL-auto-pass / flattened-block
FAIL. Each is a contract, not a fixture-fit.

Gate discipline per change: `pytest tests/ -q`, `tests/test_repo_integrity.py`,
ruff+black on changed files, `bash scripts/smoke_production.sh` ->
`SMOKE_PRODUCTION_PASS`. One atomic commit per surgical change, push origin
(Gitea) only.

---

## 6. POLICY DECISION (requires user sign-off — DECISIONS.md entry)

F3 establishes code is always minority content. The question: when judgeable code
fidelity is low, what is the proportionate response?

- **Option A — per-chunk flag + doc-level advisory.** Mark each mangled judgeable
  code chunk with a `structural_flag` in `IngestionChunk` metadata; the document
  WARNs (visible, routable for re-extraction) but does not hard-fail. Preserves
  the good prose/tables/figures (F3). Risk: a genuinely code-purpose document
  with pervasively broken code would only WARN.
- **Option B (recommended) — per-chunk flag + density-gated hard-fail.** Same
  per-chunk flag, plus a document-level **hard-fail when judgeable-code density
  is above a floor AND fidelity is below threshold**; advisory below the floor.
  Reserves the hard verdict for documents where code is load-bearing, without
  needing the impossible "is it a code book" classification (the density floor is
  a measured fraction, not a content-type guess).
- **Option C — keep whole-doc hard-fail unconditionally.** Simplest; contradicts
  F3 (discards good content over minority mangled code).

**Anti-weakening note:** moving from the *current* dead hard-fail to any of these
is **strictly more enforcement** — the gate fires today on nothing. This is not a
relaxation; it is reviving a dead contract at the correct granularity. Whichever
option is chosen is recorded in `DECISIONS.md` with this rationale.

---

## 7. Implementation order (per-change loop, each its own commit)

1. Design doc (this file) + user sign-off on §3 abstraction and §6 policy.
2. `scripts/_code_quality.py` + unit tests (no gate behavior change yet — pure
   module + tests).
3. Wire `qa_conversion_audit.py` to the module; fix content-type routing; apply
   §6 policy. Add the corpus-verdict subprocess test.
4. Wire `qa_semantic_fidelity.py` to the module (advisory parity).
5. `DECISIONS.md` entry for the §6 policy; update `docs/QUALITY_GATES.md` R3
   description and `PLAN_VLM_EVAL.md` §16.2 (close the stub).
6. (Thread 2, separate sign-off) MinerU+Qwen code routing; optional upstream
   equation guard. Validated by the metric on a live AIOS re-run.

---

## 8. What this plan explicitly does NOT do

- Does not raise/lower the `0.15` `code_heavy` threshold to pass AIOS (F2: no
  threshold separates AIOS from a code book — and the redesign makes the
  classification irrelevant anyway).
- Does not label AIOS "pseudocode" to grant loose rules (F4: real mangled
  Python).
- Does not fix only one script (F1: the seam is in both; shared module fixes
  both).
- Does not mass-reformat `qa_conversion_audit.py` (pre-existing ruff/black drift;
  surgical edits only).
- Does not skip user review before the Thread-2 (extraction) implementation.
