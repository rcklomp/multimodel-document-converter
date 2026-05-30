# MMRAG Documentation Index

Minimal docs index. The authoritative ordering is in `CLAUDE.md`'s
"Read First" list — this file is the per-doc summary.

## Read order for new sessions

1. `docs/PROJECT_STATUS.md` — current task state, active models/endpoints, next work.
2. `AGENTS.md` (repo root) — technical invariants + UIR contract + classification rules.
3. `docs/V3_EXECUTION_MANDATE.md` — single-source governance for V3. Supersedes any conflict in other docs.
4. `docs/ARCHITECTURE_V3_DRAFT_0.5.md` — V3.0 target architecture (canonical).
5. `docs/ARCHITECTURE.md` — v2.X pipeline architecture (production baseline being evolved).
6. `docs/DECISIONS.md` + `docs/QUALITY_GATES.md` — decisions log + acceptance thresholds.
7. `docs/TESTING.md` — test conventions.

Current task state lives in `docs/PROJECT_STATUS.md` — do not bury it in chat
history. The only definition of done lives in `docs/V3_EXECUTION_MANDATE.md`.

## Live working set (top-level `docs/`)

### Layer 0: Contracts (stable rules)

- `AGENTS.md` (repo root)
- `CLAUDE.md` (repo root)
- `docs/V3_EXECUTION_MANDATE.md`
- `docs/DECISIONS.md`
- `docs/QUALITY_GATES.md`
- `docs/ARCHITECTURE.md`
- `docs/ARCHITECTURE_V3_DRAFT_0.5.md`

### Layer 1: Current state

- `docs/PROJECT_STATUS.md`

### Layer 2: Execution

- `docs/TESTING.md`

## Historical reference

All v2.X plans, snapshots, calibrations, diagnostics, soaks, telemetry
reports, and the v2.16 canonical baseline are quarantined in
`docs/.archive/`. **`.aiignore` blocks agent reads on that subtree.** Do
not reference archived paths from active docs.

## Update rules

- If a hard invariant changes, update `AGENTS.md` and record the rationale in `docs/DECISIONS.md`.
- If a completion claim or evidence claim changes, apply `docs/V3_EXECUTION_MANDATE.md`.
- Do not create new governance docs. `docs/V3_EXECUTION_MANDATE.md` is the only one.
- If task status or recommended next step changes, update `docs/PROJECT_STATUS.md`.
