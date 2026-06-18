# MMRAG Documentation Index

Minimal docs index. The authoritative ordering is in `CLAUDE.md`'s
"Read First" list — this file is the per-doc summary.

## Read order for new sessions

1. `docs/PROJECT_STATUS.md` — current task state, active models/endpoints, next work.
2. `AGENTS.md` (repo root) — technical invariants + UIR contract + classification rules.
3. `docs/V3_EXECUTION_MANDATE.md` — single-source governance for V3. Supersedes any conflict in other docs.
4. `docs/ARCHITECTURE_V3_DRAFT_0.5.md` — V3.0 target architecture (canonical *aspirational target*, NOT as-built; the as-built reality is the V3.1 charter at 4a — read the charter for current behavior) (F9).
4a. `docs/ARCHITECTURE_V3.1_CHARTER.md` — V3.1 as-built + roadmap (current reality; status-tagged). Read alongside the 0.5 target.
5. `docs/ARCHITECTURE.md` — v2.X pipeline architecture (production baseline being evolved).
6. `docs/DECISIONS.md` + `docs/QUALITY_GATES.md` — decisions log + acceptance thresholds. `DECISIONS.md` opens with a **"Settled Precedents"** anti-circle index (load-bearing entries + measured-and-rejected dead ends).
7. `docs/TESTING.md` — test conventions.
8. `docs/paper/FINDINGS_DIGEST.md` — **anti-circle cold-start index** (`## SETTLED` / `## DEAD ENDS` / `## OPEN`). Read this before proposing any fix/plan (per `AGENTS.md` `AGENT-PRECEDENT-01`); the full `docs/paper/FINDINGS_LOG.md` is the detailed archive behind it. G7-enforced structure.

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
- `docs/ARCHITECTURE_V3.1_CHARTER.md` (as-built + roadmap)

### Anti-circle onboarding (required reading, not a conflict-winner)

- `docs/paper/FINDINGS_DIGEST.md` — cold-start SETTLED / DEAD ENDS / OPEN index.
  This is **not** a Layer-0 contract that wins conflicts (the MANDATE remains the
  conflict-resolution authority); it is the required-present knowledge that
  prevents re-litigating settled decisions and re-proposing measured-and-rejected
  approaches. G2-enforced as tracked; G7 enforces its three-section structure.
  The full `docs/paper/FINDINGS_LOG.md` is the append-only detailed archive.

### Layer 1: Current state

- `docs/PROJECT_STATUS.md`

### Layer 2: Execution

- `docs/TESTING.md`
- `docs/PLAN_GATE_QUALITY_V1.md` (next iteration): close the proxy-vs-outcome gap in the gate suite (spatial-first advisory metrics, fix-and-guard, `AGENT-GATE-PROGRESSION`).
- `docs/PLAN_OMNIDOCBENCH_EVAL.md` (next iteration): ground-truth fidelity benchmark (the fidelity axis paired with the gate-quality retrieval-value axis).

## Historical reference

All v2.X plans, snapshots, calibrations, diagnostics, soaks, telemetry
reports, and the v2.16 canonical baseline are quarantined in
`docs/.archive/`. **`.aiignore` blocks agent reads on that subtree.** Do
not reference archived paths from active docs.

## Update rules

- If a hard invariant changes, update `AGENTS.md` and record the rationale in `docs/DECISIONS.md`.
- If a completion claim or evidence claim changes, apply `docs/V3_EXECUTION_MANDATE.md`.
- Do not proliferate overlapping *contract* docs. The governance set is the Layer-0 list above; `docs/V3_EXECUTION_MANDATE.md` is the conflict-resolution authority within it, not the only governance file. Plans (`PLAN_*`), audits (`AUDIT_*`), and execution docs are not governance docs and may be added.
- If task status or recommended next step changes, update `docs/PROJECT_STATUS.md`.
- **If an approach is measured-and-rejected OR a decision is settled**, update
  `docs/paper/FINDINGS_DIGEST.md` (`## DEAD ENDS` / `## SETTLED`) and add the
  load-bearing entry to the "Settled Precedents" index at the top of
  `docs/DECISIONS.md`. This is how the project stops agents from re-deriving
  settled work (per `AGENT-PRECEDENT-01`). Keep the digest's three-section
  structure intact (G7 enforces it).
