# MMRAG V2 Documentation Index

Three-layer documentation structure so a new coding session can load project state without reading every file.

## Read order for new sessions

1. `docs/PROJECT_STATUS.md` — current task state, active models/endpoints, current quality baseline, next work.
2. `docs/QUALITY_SNAPSHOT_2026-05-24_v2.15_after.md` — **current canonical baseline**. v2.15.0 SHIPPED + PUSHED 2026-05-24 under Option F (tag on origin + GitHub at commit `fff67d9`). Phase 3 [F] telemetry suite + Phase 1 narrow-fixture sampler on top of unchanged v2.14.x state; NO retrieval-stack changes vs v2.13.0/v2.14.0. Phase 1 HyDE bridging executed end-to-end post-tag + CLOSED as DEAD LEVER.
3. `docs/CYCLE_OPEN_CHECKLIST.md` — cycle-open process (analyzer run + USER_ISSUES review + Docling watcher + calibration freshness + cycle_slip.log spec). Read at every v2.X cycle open.
4. `docs/PLAN_V2.16.md` — **active cycle plan**. Convergence cycle; final v2.X tag. Draft v0.10 → Ready to Execute as of 2026-05-25 (8 external audit rounds + 1 self-audit; v2.15 §9 stopping rule fired at Round 8; §8a Q1 answered ≥85% uniform). Full v0.10 audit archaeology preserved at `docs/archive/plans/PLAN_V2.16_0.10.md`. Cycle opens on next commit.
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
- `docs/QUALITY_SNAPSHOT_2026-05-24_v2.15_after.md` — current canonical baseline.
- `docs/USER_ISSUES.md` — active issues backlog.
- `docs/CYCLE_OPEN_CHECKLIST.md` — cycle-open process.

### Layer 2: Execution

- `docs/PLAN_V2.16.md` — active cycle plan.
- `docs/PLAN_V2.16_AUDIT_PROMPT.md` — audit-prompt template for v2.16 review cycles.
- `docs/TESTING.md` — test conventions.
- `docs/CONVERSION_PROFILES.md` — per-profile conversion rules.

## Historical reference (`docs/archive/`)

- `docs/archive/plans/` — PLAN_V2.10 through PLAN_V2.15 execution histories (all CLOSED + PUSHED with tags on origin); PLAN_V2.15_AUDIT_PROMPT; PLAN_V2.16_0.10.md (current cycle's 70+ finding audit archaeology).
- `docs/archive/snapshots/` — predecessor quality snapshots v2.8 through v2.14 (current canonical baseline v2.15 stays top-level).
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
