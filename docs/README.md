# MMRAG V2 Documentation Index

Three-layer documentation structure so a new coding session can load project state without reading every file.

## Read order for new sessions

1. `docs/PROJECT_STATUS.md` — current task state, active models/endpoints, current quality baseline, next work.
2. `docs/QUALITY_SNAPSHOT_2026-05-25_v2.16_after.md` — **current canonical baseline**. **v2.16.0 CONVERGENCE RELEASE — FEATURE-COMPLETE FOR v2.X. SHIPPED + PUSHED 2026-05-25** at commit `15d1349` (tag `53726ec`) on origin (Gitea) + GitHub. Phase 1 personal_importance overlay + Phase 3 partial_code adjacency mechanism (shipped + inert on current corpus) + Phase 4 VLM-table IoU dedup + Phase 0 corpus expansion (34→38 docs honest reduction). Phases 2/5/6/7 KILL'd per `docs/DECISIONS.md` "v2.16 …" entries. Post-tag: only bug-fix patches (v2.16.x); new features = re-charter as v3.0.
3. `docs/CYCLE_OPEN_CHECKLIST.md` — cycle-open process (telemetry analyzer + USER_ISSUES review + Docling watcher + cal freshness + §5 personal_importance review + cycle_slip.log spec). Read at every v2.X cycle open.
4. `docs/archive/plans/PLAN_V2.16.md` — closed cycle plan (final v2.X tag). Convergence cycle SHIPPED 2026-05-25. 8 external audit rounds + 1 self-audit (v2.15 §9 stopping rule fired at Round 8). Full v0.10 audit archaeology preserved at `docs/archive/plans/PLAN_V2.16_0.10.md`.
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
- `docs/QUALITY_SNAPSHOT_2026-05-25_v2.16_after.md` — current canonical baseline (v2.16.0 convergence release, SHIPPED + PUSHED).
- `docs/USER_ISSUES.md` — active issues backlog.
- `docs/CYCLE_OPEN_CHECKLIST.md` — cycle-open process.

### Layer 2: Execution

Post-v2.16.0: only bug-fix patches (v2.16.x) and re-charter to v3.0
remain. There is no active cycle plan in top-level `docs/` until
v2.17 fires (safety valve) or v3.0 charters.

- `docs/TESTING.md` — test conventions.
- `docs/CONVERSION_PROFILES.md` — per-profile conversion rules.

v2.16 closed cycle artifacts (plan, audit prompt, Phase 2/5
diagnostics, Phase 0 inventory, validation reports, Form_0013
smoke-fail diagnostic) live under `docs/archive/` — see below.

## Historical reference (`docs/archive/`)

- `docs/archive/plans/` — PLAN_V2.10 through PLAN_V2.16 execution histories (all CLOSED + PUSHED with tags on origin + GitHub); PLAN_V2.15_AUDIT_PROMPT + PLAN_V2.16_AUDIT_PROMPT; PLAN_V2.16_0.10.md (v2.16 70+ finding audit archaeology).
- `docs/archive/snapshots/` — predecessor quality snapshots v2.8 through v2.15 (current canonical baseline v2.16 stays top-level).
- `docs/archive/calibrations/` — v2.14 Phase 0 calibration runs. Operative verdict per v2.14 close-out: FP8-14B (`RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic`) — rel 82.2% / format 90.7% TRUSTWORTHY / faith 76.6%.
- `docs/archive/diagnostics/` — v2.9–v2.10 phase diagnostic reports + v2.16 Phase 2 omlx-deficit + v2.16 Phase 5 pre-flight verdicts.
- `docs/archive/soaks/` — v2.14/v2.15 soak reports (intent-HyDE FALSIFIED; v2.15 narrow-HyDE AB).
- `docs/archive/misc/` — HANDOFF_V2.14, JUDGE_EVAL openrouter shortlist, telemetry reports 2026-05-24 + 2026-05-25, v2.16 corpus-expansion inventory, v2.16 validation reports, v2.16 Form_0013 smoke-fail diagnostic.
- `docs/archive/quality_snapshots/` — v2.8-era raw audit outputs.

Sibling-to-sibling links inside archived files still resolve (everything lives under the same archive root). Live-doc references to archived files use the explicit `docs/archive/<subdir>/<file>` path.

## Update rules

- If a hard invariant changes, update `AGENTS.md` and record the rationale in `docs/DECISIONS.md`.
- If a completion claim or evidence claim changes, apply `docs/AGENT_GOVERNANCE.md`.
- If adding documentation, obey the documentation budget in `docs/AGENT_GOVERNANCE.md`.
- If the quality baseline changes, create or update a dated `docs/QUALITY_SNAPSHOT_*.md`. The prior baseline moves to `docs/archive/snapshots/`.
- If task status or recommended next step changes, update `docs/PROJECT_STATUS.md`.
- If the active cycle plan supersedes (next v2.X), the prior plan moves to `docs/archive/plans/` at Phase N close-out.
