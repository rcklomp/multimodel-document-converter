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

4. `docs/DECISIONS.md` and `docs/QUALITY_GATES.md`
   - design decisions and acceptance thresholds
   - quality gates used to decide pass/fail

Read `docs/ARCHITECTURE.md` only when changing core pipeline behavior. Read the SRS only when a contract is ambiguous.

## Layer Model

### Layer 0: Contracts

Stable rules. These should change rarely.

- `AGENTS.md`
- `CLAUDE.md`
- `docs/DECISIONS.md`
- `docs/QUALITY_GATES.md`
- `docs/ARCHITECTURE.md`
- `docs/SRS_Multimodal_Ingestion_V2.5.md`

### Layer 1: Current State

Compact project status. These files should be updated whenever a session changes direction, baseline, or known quality state.

- `docs/PROJECT_STATUS.md`
- `docs/QUALITY_SNAPSHOT_2026-04-29.md`

### Layer 2: Work Logs And Execution

Operational checklists, commands, and historical notes.

- `docs/PROGRESS_CHECKLIST.md`
- `docs/TESTING.md`
- `docs/CONVERSION_PROFILES.md`
- `docs/PLAN_V2.7_DOCUMENT_UNDERSTANDING.md`
- `docs/archive/`

## Update Rules

- If a hard invariant changes, update `AGENTS.md` and record the rationale in `docs/DECISIONS.md`.
- If the quality baseline changes, create or update a dated `docs/QUALITY_SNAPSHOT_*.md`.
- If task status changes, update `docs/PROGRESS_CHECKLIST.md`.
- If the recommended next step changes, update `docs/PROJECT_STATUS.md`.
- Do not bury current-state information in chat history only.

