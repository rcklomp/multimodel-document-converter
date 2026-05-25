# MMRAG V2 Documentation Index

Three-layer documentation structure so a new coding session can load project state without reading every file.

## Read order for new sessions

1. `docs/PROJECT_STATUS.md` — current task state, active models/endpoints, current quality baseline, next work.
2. `docs/QUALITY_SNAPSHOT_2026-05-25_v2.16_after.md` — **current canonical baseline**. **v2.16.0 CONVERGENCE RELEASE — FEATURE-COMPLETE FOR v2.X.** Phase 1 personal_importance overlay + Phase 3 partial_code adjacency mechanism (shipped + inert on current corpus) + Phase 4 VLM-table IoU dedup + Phase 0 corpus expansion (34→41 docs). Phases 2/5/6/7 KILL'd per `docs/DECISIONS.md` "v2.16 …" entries. Post-tag: only bug-fix patches (v2.16.x); new features = re-charter as v3.0.
3. `docs/CYCLE_OPEN_CHECKLIST.md` — cycle-open process (telemetry analyzer + USER_ISSUES review + Docling watcher + cal freshness + §5 personal_importance review + cycle_slip.log spec). Read at every v2.X cycle open.
4. `docs/PLAN_V2.16.md` — **closed cycle plan** (final v2.X tag). Convergence cycle SHIPPED 2026-05-25. 8 external audit rounds + 1 self-audit (v2.15 §9 stopping rule fired at Round 8; §8a Q1 answered ≥85% uniform). Full v0.10 audit archaeology preserved at `docs/archive/plans/PLAN_V2.16_0.10.md`.
5. `AGENTS.md` — hard project invariants + architecture constraints + source-of-truth rules.
6. `docs/AGENT_GOVERNANCE.md` — evidence, completion, review, documentation-budget rules.
7. `docs/DECISIONS.md` + `docs/QUALITY_GATES.md` — design decisions log + acceptance thresholds.

Read `docs/ARCHITECTURE.md` when changing core pipeline behavior. When a metric appears in both a layer-1 status doc and a dated snapshot, **the latest snapshot is canonical** (per `docs/AGENT_GOVERNANCE.md` Canonicality Rule). Current task state lives in `docs/PROJECT_STATUS.md` — do not bury it in chat history.

## Live working set (top-level `docs/`)

### Layer 0: Contracts (stable rules, rarely change)

- `AGENTS.md` (repo root)
- `CLAUDE.md` (repo root)
- `docs/AGENT_GOVERNANCE.md`
- `docs/DECISIONS.md`
- `docs/QUALITY_GATES.md`
- `docs/ARCHITECTURE.md`

### Layer 1: Current state

- `docs/PROJECT_STATUS.md`
- `docs/QUALITY_SNAPSHOT_2026-05-25_v2.16_after.md` — current canonical baseline (v2.16.0 convergence release).
- `docs/QUALITY_SNAPSHOT_2026-05-24_v2.15_after.md` — predecessor canonical (v2.15.0); retained top-level for delta reproducibility.
- `docs/USER_ISSUES.md` — active issues backlog.
- `docs/CYCLE_OPEN_CHECKLIST.md` — cycle-open process.

### Layer 2: Execution

- `docs/PLAN_V2.16.md` — closed cycle plan (final v2.X tag).
- `docs/archive/PLAN_V2.16_AUDIT_PROMPT.md` — audit-prompt template (archived at v2.16.0 close).
- `docs/DIAGNOSTIC_2026-05-25_v2.16_p2_omlx_deficit_root_cause.md` — Phase 2 verdict (multi-factor → Phase 6 KILL).
- `docs/PHASE5_PREFLIGHT_2026-05-25.md` — Phase 5 pre-flight verdict (KILL).
- `docs/VALIDATION_REPORT_2026-05-25_v2.15.0_baseline.md` — Phase 1 baseline (delta anchor for Phase 3/4 lift measurement).
- `docs/TESTING.md` — test conventions.
- `docs/CONVERSION_PROFILES.md` — per-profile conversion rules.

## Historical reference (`docs/archive/`)

- `docs/archive/plans/` — PLAN_V2.10 through PLAN_V2.15 execution histories (all CLOSED + PUSHED with tags on origin); PLAN_V2.15_AUDIT_PROMPT; PLAN_V2.16_0.10.md (current cycle's 70+ finding audit archaeology).
- `docs/archive/snapshots/` — predecessor quality snapshots v2.8 through v2.14 (current canonical baseline v2.16 stays top-level; v2.15 retained top-level for delta reproducibility).
- `docs/archive/calibrations/` — v2.14 Phase 0 calibration runs. Operative verdict per v2.14 close-out: FP8-14B (`RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic`) — rel 82.2% / format 90.7% TRUSTWORTHY / faith 76.6%.
- `docs/archive/diagnostics/` — v2.9–v2.10 phase diagnostic reports (heading reclassification, missing pages, cross-page splits, OCR).
- `docs/archive/soaks/` — v2.14/v2.15 soak reports (intent-HyDE FALSIFIED; v2.15 narrow-HyDE AB).
- `docs/archive/misc/` — HANDOFF_V2.14, JUDGE_EVAL openrouter shortlist, TELEMETRY_REPORT 2026-05-24.
- `docs/archive/quality_snapshots/` — v2.8-era raw audit outputs.

Sibling-to-sibling links inside archived files still resolve (everything lives under the same archive root). Live-doc references to archived files use the explicit `docs/archive/<subdir>/<file>` path.

## Update rules

- If a hard invariant changes, update `AGENTS.md` and record the rationale in `docs/DECISIONS.md`.
- If a completion claim or evidence claim changes, apply `docs/AGENT_GOVERNANCE.md`.
- If adding documentation, obey the documentation budget in `docs/AGENT_GOVERNANCE.md`.
- If the quality baseline changes, create or update a dated `docs/QUALITY_SNAPSHOT_*.md`. The prior baseline moves to `docs/archive/snapshots/`.
- If task status or recommended next step changes, update `docs/PROJECT_STATUS.md`.
- If the active cycle plan supersedes (next v2.X), the prior plan moves to `docs/archive/plans/` at Phase N close-out.
