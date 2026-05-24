# Audit prompt — `docs/PLAN_V2.15.md` Draft v0.2

Paste the block below into a fresh top-tier model session (GPT-5.x,
Opus 4.7, Gemini 3.x reasoning, etc.). Attach `docs/PLAN_V2.15.md`
as a file; the auditor needs the whole document, not an excerpt.

---

## Prompt

You are auditing the v2.15 execution plan for **MM-Converter-V2**, a
PDF→JSONL multimodal RAG ingestion + retrieval pipeline. The plan is
attached as `PLAN_V2.15.md` (currently Draft v0.2, dated 2026-05-24).
The author is the project's primary maintainer plus an AI coding
agent; together they've shipped v2.10 → v2.14 over the past two
weeks and are now scoping v2.15.

This is the **second** audit pass. The first was done by Gemini and
its findings are already incorporated in v0.2 (see the header
"three changes drove the revision" block + Appendix A archaeology).
**Do not re-flag any of the five Gemini items**:

1. Option A pdfplumber budget was bumped 2–3d → 5–7d with explicit
   UIR-schema mapping sub-task (Section 3 Phase 2)
2. Phase 4 Approach 1 (regex chunk-splitting) was explicitly
   rejected; only Approach 2 (Docling config tuning) viable, with a
   hard abort gate if not feasible (Section 3 Phase 4)
3. Phase 5c paraphrase fusion got a strict latency budget
   (<1500ms p50 / <3000ms p99) (Section 3 Phase 5c)
4. Phase 1/5 isolation rule added — mini-soaks for HyDE bridging
   and retrieval tuning may not overlap in the same run
   (Sections 3 + 5)
5. New Phase 3 [F] document-class query telemetry — required
   mechanism for Option F to mean anything (Section 3 Phase 3)

Your job is to find what the **first audit missed**, not to validate
its work or pad with restated concerns. If you have nothing of
substance to add in a given area, say so explicitly — "audited,
nothing to flag" is a valuable signal in this context.

### Background context the auditor needs

- **Project state**: v2.14.0 was tagged + pushed to origin
  2026-05-23 (commit `36482e0`, sha `122a62e`). v2.14.x patch
  range added Phase 2 (intent classifier — FALSIFIED), Phase 3
  (rollback collection drop), and v2.14.1 (GX10 endpoint swap to
  `RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic`, Phase 0 verdict
  rel 82.2 / **format 90.7 TRUSTWORTHY** / faith 76.6). n-gram
  speculative decoding was tested and rejected post-swap (6.3%
  acceptance).
- **Hard memories** (load-bearing rules the agent will follow):
  - `no-gx10-model-swap-reflex` — never propose a model swap as
    the reflex response to a perf disappointment; offline eval
    first via `scripts/calibrate_local_judge_vs_qwen_max.py`
  - `gx10-deployment-guardrails` — 5-point hard checklist before
    any GX10 vLLM swap (aarch64 image, port collision check,
    unified-memory sizing, pre-quantized FP8, Phase 0 re-cal)
  - `fix-extraction-not-judge` — when weak-query rationales cite
    truncation/whitespace/garbled OCR, fix the extraction layer;
    never shift judge prompts to mask defects
  - `contract-violation-mode` — never weaken a gate to ship a
    clean close; defer with sign-off instead
- **Production retrieval stack (unchanged from v2.13.0, will stay
  unchanged through v2.15)**:
  ```
  query → omlx Qwen3-Embedding-8B-mxfp8 dense + BM25 sparse
        → RRF fusion (k=60), top-25 candidates per leg
        → local ModernBERT rerank
        → top-5 return
  ```
- **Cost cap**: $25/cycle (Dashscope API spend). Cycles have been
  coming in well under this ($1–5 typical).

### Audit lenses (specific questions to answer)

Address each of these explicitly, even if briefly. Don't reorganize;
the user wants to compare your answers to Gemini's section-by-section.

1. **Is the Option A/E/F framing a real trichotomy or a false one?**
   Concretely: is there a fourth path the plan doesn't consider, or
   are A/E/F actually overlapping points on a continuum that the
   plan presents as discrete? If the user picks F today, what
   commits them to A or E in v2.16 vs. perpetually rolling F
   forward?

2. **Phase 3 [F] telemetry: sufficient or theatrical?**
   The plan claims telemetry will produce v2.16-actionable evidence.
   Will it? Specifically: who reads the rolling log, against what
   denominator, and what hit-rate triggers Option A treatment? If
   that threshold isn't defined now, what's the failure mode where
   v2.16 inherits a dataset with no decision rule attached?

3. **Phase 1 (HyDE bridging) re-targeting**:
   v2.14 Phase 2 falsified the *broad-query* lift hypothesis.
   v2.15 Phase 1 retries on a narrower 5-doc mini-soak (ATZ_Elektronik,
   Python_Cookbook, IRJET, Hybrid_electric_vehicles, Greenhouse_Design).
   What evidence is there that narrowing the fixture WILL change
   the result, vs. just resampling the same noise floor? Is the
   ≥8pp lift acceptance bar achievable given that the broad soak
   found null? What stops this from being "Phase 2 again but with
   smaller n"?

4. **Phase 2 [A] / Phase 4 [A] interaction**:
   Both are gated on Option A. If A is chosen, Phase 4's "1-day
   spike for Approach 2 viability" runs in parallel with Phase 2's
   "5–7 days of pdfplumber + UIR-schema mapping". The plan says
   they can run as parallel branches. Is that engineering realism,
   or are they actually fighting for the same engineer's attention?
   How should the user think about effective serial time vs.
   advertised parallel time?

5. **Phase 4 hard abort gate**:
   The plan kills Approach 1 (regex) and demands Approach 2
   (Docling config) or full abort. Defensible. But: **what's the
   real cost of leaving Fluent_Python's truncated-code defect
   unfixed**, if Approach 2 fails the viability spike? Is the
   abort path equivalent to "Fluent_Python stays a documented-
   limitation forever," and if so, is that compatible with the
   plan's stated goal of "fix at the right layer"?

6. **Phase 5c latency budget**:
   Gemini set <1500ms p50 / <3000ms p99 for paraphrase fusion on
   the FP8-14B. Single-stream judge calls are ~2.0s and HyDE
   generation is ~9s for ~600-token outputs. The plan proposes
   either batched-`n=5` sampling or `asyncio.gather` parallelism
   to clear the budget. **Will either actually work** given the
   measured single-stream throughput of ~15 tok/s on this
   endpoint? If batched-n produces siblings that are too
   correlated, what's the empirical test, and what's the fallback
   if both paths fail?

7. **Phase 6 [U] calibration freshness**:
   Window expires 2026-06-22. v2.15 close-out date isn't fixed.
   What's the failure mode if the cycle closes 2026-06-23 and a
   re-cal hasn't been scheduled? The plan says "re-run if >30
   days OR model change" but doesn't specify when in the cycle
   the check fires. Is this a process gap?

8. **Definition of Done bar**:
   The DoD requires Option F's Phase 3 telemetry to ship IF F is
   chosen. But the strategic decision is itself listed as a DoD
   item ("Strategic decision recorded in DECISIONS.md with
   evidence"). What happens if the user defers the strategic
   decision past v2.15 close-out? Does v2.15 ship without a
   chosen path, with Phase 1 + 6 + N as a "Phase F-prep" cycle
   in name only?

9. **Carry-forwards** (Section 4): items 1.1 (same-page prose/VLM
   dedup), 6.1 (Docling prose+code disambig), and the 3a/3c/3e
   v2.11 carry-forwards. Are any of these badly triaged? In
   particular: 3c (UIR refactor PAUSED for user signoff) has
   carried since v2.11. At what point does a perpetually-paused
   item become technical debt that the plan should address vs.
   continuing to defer?

10. **Cost discipline**: the plan estimates $1–10 worst-case
    across all three Options. v2.14 came in at ~$1.20. Are these
    estimates honest given a realistic Option-A pdfplumber soak
    that may iterate 2–3 times before hitting acceptance? Is
    there a hidden cost dimension (LAN bandwidth, GX10 wall-
    clock, dev time) the plan undercounts?

### What NOT to do

- Don't restate Gemini's findings as if new.
- Don't suggest features that aren't problems ("you should also
  add X monitoring") — only flag concrete failure modes.
- Don't suggest reorganizing the plan structure for its own sake.
- Don't write "the plan should consider..." without naming the
  concrete consequence of not considering it.
- Don't pad with sycophantic affirmation ("the plan is well-
  structured and..."). Get straight to substantive findings.
- Don't refuse to engage with the strategic A/E/F decision because
  "the user should decide." You're being asked for a second-
  opinion stance; give one with reasoning, and label it as
  recommendation not requirement.
- Don't critique decisions that are already locked-in load-
  bearing rules (the no-swap-reflex memory, the byte-identical
  retrieval stack, the $25/cycle cap). Take those as constraints.

### Output format

Use the exact structure below. One entry per finding. No prose
introduction; start at "Finding 1".

```
## Finding N — [one-sentence title]

- **Severity**: HIGH | MED | LOW
- **Plan section**: [e.g., "Section 3 Phase 2, paragraph 3"]
- **Issue**: [one sentence, no padding]
- **Concrete failure mode**: [one sentence — what breaks, when, and how the user notices]
- **Recommended fix**: [one sentence — actionable, specific enough that the agent can implement it without further clarification]
- **Confidence**: HIGH | MED | LOW [your confidence that this finding is real and not noise]
```

After all findings, add:

```
## Audit lenses with nothing to flag

[Bulleted list of the numbered audit lenses above where you
genuinely have nothing of substance to add. Be explicit — this is
information, not a failure to engage.]

## Overall stance

[2-3 sentences max. Recommend Option A, E, or F with reasoning.
State whether the plan is shippable as Draft v0.2 or needs a v0.3
revision before the cycle can proceed.]
```

End of prompt.

---

## Notes for the user (not part of the prompt)

- The audit assumes the auditor has not seen this conversation; the
  background block is self-contained.
- Audit lens #4 (Phase 2+4 parallel-vs-serial) is the most likely
  place to surface real findings the Gemini pass missed — both
  involve engineering judgment about effective time, not just
  technical correctness.
- If the auditor's overall stance disagrees with Gemini's (Option F
  recommendation), that disagreement is itself useful signal —
  worth reading carefully rather than averaging.
- If the auditor's findings are mostly LOW/MED severity with no
  HIGH, that suggests Draft v0.2 is close to executable. If a HIGH
  surfaces, a Draft v0.3 revision before any phase begins is the
  right move.
