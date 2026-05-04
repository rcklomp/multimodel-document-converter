# MMRAG V2 Documentation Index

This folder uses a three-layer documentation structure so a new coding session can load the project state without reading every file.

## Read Order For New Sessions

1. `docs/PROJECT_STATUS.md`
   - current project state
   - active models/endpoints, without secrets
   - current quality baseline
   - immediate next work

2. `docs/PROGRESS_CHECKLIST.md`
   - executable task checklist
   - per-workstream status
   - commands to reproduce checks
   - handoff protocol

3. `AGENTS.md`
   - hard project invariants
   - architecture constraints
   - source-of-truth rules for agents

4. `docs/AGENT_GOVERNANCE.md`
   - evidence, completion, review, and documentation-budget rules

5. `docs/DECISIONS.md` and `docs/QUALITY_GATES.md`
   - design decisions and acceptance thresholds
   - quality gates used to decide pass/fail

Read `docs/ARCHITECTURE.md` when changing core pipeline behavior. Read the SRS only as historical v2.5 context; if it conflicts with `AGENTS.md`, `docs/DECISIONS.md`, or `docs/ARCHITECTURE.md`, the current docs win. **Current canonical baseline is `docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md`** — when a metric appears in both a layer-1 status doc and a dated snapshot, the latest snapshot is canonical (per `docs/AGENT_GOVERNANCE.md` Canonicality Rule).

## Layer Model

### Layer 0: Contracts

Stable rules. These should change rarely.

- `AGENTS.md`
- `CLAUDE.md`
- `docs/AGENT_GOVERNANCE.md`
- `docs/DECISIONS.md`
- `docs/QUALITY_GATES.md`
- `docs/ARCHITECTURE.md`
- `docs/archive/SRS_Multimodal_Ingestion_V2.5.md`
  - Stale v2.5 reference; not the current source of truth for v2.8 behavior. Use `AGENTS.md`, `docs/DECISIONS.md`, `docs/ARCHITECTURE.md` instead.

### Layer 1: Current State

Compact project status. These files should be updated whenever a session changes direction, baseline, or known quality state.

- `docs/PROJECT_STATUS.md`
- `docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md` — **current canonical baseline** (v2.8.0 SHIPPED)
- `docs/QUALITY_SNAPSHOT_2026-05-03.md` — v2.8 BEFORE baseline (kept frozen so the AFTER snapshot's delta column is reproducible; not a current-state document)
- `docs/archive/quality_snapshots/QUALITY_SNAPSHOT_2026-05-01.md` — superseded; banner-annotated. Specifically: Ayeva 0.93 reading is from the older probe under different flags; v2.8 fresh re-conversion reads 0.83 FAIL.
- `docs/archive/quality_snapshots/QUALITY_SNAPSHOT_2026-04-30.md` — superseded; banner-annotated.
- `docs/archive/quality_snapshots/QUALITY_SNAPSHOT_2026-04-29.md` — superseded; banner-annotated.

### Layer 2: Work Logs And Execution

Operational checklists, commands, and historical notes.

- `docs/PROGRESS_CHECKLIST.md`
- `docs/TESTING.md`
- `docs/CONVERSION_PROFILES.md`
- `docs/PLAN_V2.9_DRAFT_PROMPT.md` — prompt for drafting `docs/PLAN_V2.9.md` (next execution cycle)
- `docs/archive/PLAN_V2.8_PRODUCTION_GAPS.md` — **SHIPPED 2026-05-04** (annotated tag `v2.8.0` on `645ab2b`); retained for historical context
- `docs/archive/PLAN_DOCLING_POSTPROCESSOR.md` — shipped 2026-05-03; folded into v2.8 (post-Docling sanity pass + `digital_literature` profile)
- `docs/archive/` — completed plans (`PLAN_V2.7_DOCUMENT_UNDERSTANDING.md`, `PLAN_HYBRID_CHUNKER_MIGRATION.md`) and historical notes

## Update Rules

- If a hard invariant changes, update `AGENTS.md` and record the rationale in `docs/DECISIONS.md`.
- If a completion claim or evidence claim changes, apply `docs/AGENT_GOVERNANCE.md`.
- If adding documentation, obey the documentation budget in `docs/AGENT_GOVERNANCE.md`.
- If the quality baseline changes, create or update a dated `docs/QUALITY_SNAPSHOT_*.md`.
- If task status changes, update `docs/PROGRESS_CHECKLIST.md`.
- If the recommended next step changes, update `docs/PROJECT_STATUS.md`.
- Do not bury current-state information in chat history only.
