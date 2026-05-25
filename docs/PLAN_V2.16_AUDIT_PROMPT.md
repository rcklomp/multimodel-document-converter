# Pre-execution sanity read — `docs/PLAN_V2.16.md` (Convergence Cycle, v0.10, audit cleared)

Paste the block below into a capable model session (GPT-5.x, Opus 4.7,
Gemini 3.x reasoning, etc.). Attach `docs/PLAN_V2.16.md` as a file.

This is a **pre-execution sanity check**, not an audit round. The plan
has already cleared the formal audit cadence (8 external rounds + 1
self-audit; stopping rule fired at Round 8 per v2.15 §9). It is
declared Ready to Execute. This prompt exists as an escape hatch: if
the plan has a blind spot that nine rounds of structured audit missed,
a fresh read without the audit-round framing might surface it.

**Do not produce audit-round output.** Do not enumerate findings. Do not
assign severity labels. Do not recommend more validation or process
steps. Most uses of this prompt should return "nothing to flag" — that
is the expected answer for a plan that cleared nine audit rounds.

---

## Prompt

You are reading the **convergence-cycle execution plan** for v2.16 of
MM-Converter-V2, a solo-dev PDF→JSONL multimodal RAG pipeline. v2.16
is intended as the final v2.X release: feature-complete, then
indefinite bug-fix-only patch lane. The plan is a SHIP/KILL/OUT-OF-SCOPE
disposition of 20+ carry-forward items accumulated across v2.11→v2.15.

Nine prior audit rounds have been completed. The plan is at Draft v0.10
and declared Ready to Execute. This read is to catch anything nine
rounds missed — a narrow net, deliberately.

### What you need to know

- **Production stack**: omlx Qwen3-Embedding-8B-mxfp8 dense + BM25 sparse
  → RRF fusion → ModernBERT rerank → top-5. Six unconditional phases
  (0-4), two conditional (5-6), one default-KILL (7), one close-out (N).
- **Budget**: 12 working-day cap, $25/cycle cloud spend cap.
- **Constraint**: convergence cycle forbids soft state. Every item gets
  SHIP, KILL, or OUT-OF-SCOPE. No deferral, no "we'll see."
- **v2.17 exists only as a narrow safety valve** with four enumerated
  triggers (§9); any outcome outside those four folds into v2.16.
- **Post-tag**: only v2.16.x bug-fix patches; new features = v3.0
  re-charter. v2.16.x vs v3.0 boundary defined in §10.

### What to read for

You are looking for things that would cause a **concrete production
failure** if the plan is executed as written. That means:

- A phase acceptance bar that is structurally impossible to pass (not
  just ambitious — actually unreachable given the method described).
- A disposition that contradicts another part of the plan (e.g., a
  SHIP phase depends on output from a KILL'd item).
- A v2.17 trigger that would fire on the *normal* execution path,
  making "feature-complete" an unfalsifiable claim.
- A gap in the DoD (§2) that, if unmet, would still allow the tag
  to ship (a genuine gate that doesn't gate).
- A KILL rationale that is factually wrong about the current state
  of the codebase (not "could be better argued" — factually incorrect).

What you are NOT looking for:
- Rationales you'd write differently
- Thresholds you'd tune differently
- Features you'd add or remove
- Things that "might" be a problem in some hypothetical scenario
- "More validation would be nice" observations

### Questions to answer (concisely)

If the answer to all three is "no," say so and stop. That is the
expected outcome.

1. **Is there anything in this plan that, if executed exactly as
   written, will produce a verifiably wrong result?** A phase will fail
   its own acceptance bar by construction; a dependency is circular; a
   gate can't fire. Concrete, not speculative.

2. **Is the v2.17 safety valve (§9) structurally equivalent to "we'll
   ship v2.16 and fix the rest in v2.17"?** If yes, the convergence-cycle
   frame is dishonest. The four triggers must be genuine safety-valve
   conditions, not predictable outcomes of normal execution.

3. **After reading the full plan, is there a §2 Definition of Done item
   that could pass its stated check while the underlying problem remains
   unsolved?** A gate with a hole — the check passes, the user still has
   the problem the gate was supposed to prevent.

### What NOT to do

- Don't produce a finding list. If you have something, describe it in
  prose under the relevant question.
- Don't suggest "more validation," "more testing," "add a checkpoint,"
  or "defer to v3.0."
- Don't critique KILL rationales for being insufficiently evidence-backed
  — nine rounds already stress-tested those.
- Don't re-dispose items. The disposition matrix is locked.
- Don't audit the audit process. The stopping rule fired; that's settled.

The correct length of this response is a short paragraph answering each
of the three questions. If all three answers are "no, nothing to flag,"
the entire response should be under 200 words.

---

## Notes for the user

- This replaces the stale Round 5 audit prompt that described v0.6 state
  and drove an unending iteration loop through its 14-lens structure.
- The old prompt's structural problems: required the auditor to know what
  9 prior rounds caught (impossible), produced output volume through its
  lens count, and made findings the default mode. This replacement
  inverts all three.
- The plan has cleared audit. This prompt is a fire extinguisher, not a
  fire drill — use it once before executing, expect nothing, and if
  nothing comes back, start Phase 0.
- If this prompt surfaces a substantive issue, fix it in the plan
  directly, record the change in a DECISIONS.md entry, re-run the prompt
  once more, then execute. Do not re-enter the audit-round cycle.