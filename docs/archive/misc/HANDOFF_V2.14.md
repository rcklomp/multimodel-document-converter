# v2.14 Cycle Handoff Prompt

> **Purpose:** paste the body below into the opening message of a fresh
> Claude Code session to hand off the v2.14 cycle. Designed to maximize
> autonomous execution while gating the actions that have specific
> recorded failure modes.
>
> **Last updated:** 2026-05-23 (post GX10 27B-MTP swap).
> **Keep this file in sync** with PLAN_V2.14.md when phase scope shifts
> or when new feedback memories land that constrain operation.

---

You're picking up the v2.14 cycle of MM-Converter-V2. The cycle is in
progress, the GX10 local LLM endpoint is live, and Phase 0 re-cal is
the next concrete action. Your job is to execute the rest of v2.14 to
ship state autonomously, using the gates below to know when to stop.

## Read first, in this order

1. `CLAUDE.md` — invariants + commands.
2. `docs/PROJECT_STATUS.md` — current state. v2.14 IN PROGRESS.
3. `docs/PLAN_V2.14.md` — authoritative scope (currently Draft v0.5).
   Read all of it; do not skim section 2 (Phases) or the GX10
   deployment guardrails under Phase 4.
4. `MEMORY.md` and follow the links — especially:
   - `feedback_gx10_deployment_guardrails` (5-point hard checklist)
   - `feedback_fix_extraction_not_judge`
   - `feedback_contract_violation_mode`
   - `feedback_qa_policy` / `feedback_bridge_tests` / `feedback_libraries_first`
   - `feedback_doc_sanitization_completeness`
   - `project_v2_14_gx10_27b_mtp_swap` (current GX10 endpoint state)

## Where you are right now (2026-05-23)

- v2.13.0 SHIPPED; v2.14 active.
- GX10 vLLM endpoint LIVE at `http://10.0.10.239:8000` serving
  `Qwen/Qwen3.6-27B-FP8` with native MTP=3 (~32 tok/s). `hyde.py`
  default already points at it. Phase 4a + Phase 5 shipped.
- Phase 0 calibration against the original 14B is SUPERSEDED.
- Plan + docs + memory are consistent and freshly swept; trust them.

## Immediate next step — Phase 0 re-cal (your entry action)

Before anything else, verify the endpoint is healthy:

```bash
curl -sf http://10.0.10.239:8000/v1/models >/dev/null \
  || echo "GX10 down — see PLAN_V2.14.md Phase 4 Resilience"
```

Then run, expecting ~80 min, $0:

```bash
python scripts/calibrate_local_judge_vs_qwen_max.py \
  --work-path output/soak/v2.13_p1_omlx/work.jsonl \
  --report-path docs/CALIBRATION_2026-05-23_v2.14_p0_local_judge_qwen36_27b_mtp.md
```

When it finishes:

1. Read the per-axis verdicts.
2. Update `memory/project_v2_14_p0_calibration.md` SUPERSEDED block to
   cross-link the new report.
3. Update `docs/PLAN_V2.14.md` "Phase outcomes" Phase 0 row.
4. Derive the new Phase 4 PERMITTED/FORBIDDEN list using the
   Disposition thresholds in `PLAN_V2.14.md` §"Phase 0".
5. Proceed per `PLAN_V2.14.md` §"Phase ordering rationale" — Phase 6
   and Phase 1 can run in parallel.

## You may do these without asking

- Read/edit code, run the test suite, run mini-soaks ≤ $5 each.
- Re-extract documents and re-ingest into existing collections.
- Edit any doc EXCEPT new entries in Layer-0 contracts (`AGENTS.md`,
  `docs/QUALITY_GATES.md`, `docs/DECISIONS.md`, `docs/ARCHITECTURE.md`)
  — appending new dated entries is fine; rewriting existing contract
  text is not.
- Update `PLAN_V2.14.md`, `PROJECT_STATUS.md`, snapshots, and memory
  files when state changes. Apply doc-sanitization-completeness:
  sweep WHOLE files, not just the section you came for.
- Use Plan mode liberally for any phase whose method has
  architectural ambiguity (Phase 1 VLM fallback, Phase 6 chunking
  policy specifically benefit). Use TodoWrite to surface multi-step
  work. Use the Explore/Plan subagents and parallel tool calls
  freely. Run long soaks as background processes.

## Stop and ask before doing any of these

1. Swapping the GX10 endpoint (any model/container change). Walk
   the 5-point checklist in `feedback_gx10_deployment_guardrails`,
   write a deployment note, pause for sign-off.
2. Phase 3 (dropping `mmrag_v2_8__qwen3_dashscope` or `mmrag_v2_8`)
   — even after 2026-06-19, require explicit "no regression, drop it".
   Draft v0.5 requires a 90-day cold-storage snapshot before deletion.
3. Phase 1 VLM-fallback choice (Docling TSR alone vs TSR + VLM) once
   you have evidence Docling alone doesn't resolve CarOK.
4. UIR refactor (3c) — PAUSED for user signoff.
5. Final v2.14.0 tag creation.
6. ANY git push to GitHub/Gitea remotes.
7. Cumulative cloud spend approaching $20 (cap is $25/cycle).
8. Any retrieval-stack architecture change (RRF weights, top_k,
   sparse/dense balance, reranker swap) — these belong to a later
   cycle, not v2.14 (`PLAN_V2.14.md` §"Non-goals").

## Hard rules (do not violate, even under pressure)

- DO NOT weaken test assertions, gates, or quality thresholds to
  make a failing run pass. Fix the defect or defer with sign-off.
  (`feedback_contract_violation_mode`)
- DO NOT shift the judge prompt to mask extraction defects. When
  weak-query rationales cite truncation/whitespace/garbled OCR, fix
  the extraction layer. (`feedback_fix_extraction_not_judge`)
- DO NOT use the local LLM judge for RRF weight / top_k / sparse-
  dense / rerank-hyperparameter tuning or for cycle-close go/no-go
  on Relevance + Faithfulness axes. Cloud `qwen-max` owns those
  decisions. (`PLAN_V2.14.md` §"FORBIDDEN local-judge uses")
- Run the v2.13 retrieval-regression fingerprint before declaring
  any retrieval-touching phase shipped.
- Apply doc-sanitization-completeness when you change anything that
  affects the headline / "current" sections of any doc.

## Operating cadence

- After each phase ships: update `PLAN_V2.14.md` "Phase outcomes" +
  `PROJECT_STATUS.md` phase table + relevant memory + commit (one
  phase per commit when possible; do NOT push).
- Surface a one-sentence status update at each phase boundary.
- Run `pytest` before considering any code-side phase done.
- Keep commits small, descriptive, and focused.

## Stop and report (don't improvise)

- Phase 0 re-cal verdict puts any axis below 70% (NOT USABLE).
- Soak result with >10pp swing from expected baseline.
- Any "workaround" emerges as the fix rather than the root cause.
- A hard rule is in tension with making progress.
- Anything that surprises you, including the things you'd describe
  in a commit message as "for now" or "temporary".

## Definition of Done for v2.14

`PLAN_V2.14.md` §"Phase N — Cycle close-out" + Draft v0.5 audit
item #6. The v2.14.0 tag goes to staging only — never push.
