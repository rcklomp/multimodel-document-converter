# MMRAG V2 Documentation Index

This folder uses a three-layer documentation structure so a new coding session can load the project state without reading every file.

## Read Order For New Sessions

1. `docs/PROJECT_STATUS.md`
   - current project state
   - active models/endpoints, without secrets
   - current quality baseline
   - immediate next work and per-phase status

2. `docs/QUALITY_SNAPSHOT_2026-05-20_v2.11_soak_qwen3.md`
   - current canonical baseline (v2.11 Phase 1 soak — production
     embedder Dashscope `text-embedding-v4`; Recall@1 35.5%,
     Recall@5 chunk 66.8%, Format 89.8%)

2b. `docs/QUALITY_SNAPSHOT_2026-05-16_v2.10_after.md`
   - v2.10 strict-gate baseline (corpus 34/34 PASS; unchanged in
     v2.11 because the swap is retrieval-side only)

2c. `docs/QUALITY_SNAPSHOT_2026-05-16_v2.10_soak.md`
   - v2.10 soak (Format 98.3%, Recall@1 2.1%); the baseline that the
     v2.11 Phase 1 numbers compare against

3. `docs/PLAN_V2.11.md`
   - v2.11 plan (Draft v1.0); Phase 1 swap SHIPPED locally
     (`c2a461c`); Phase 2 CI + Phase 3 carry-forward dispositions
     SHIPPED. v2.11.0 tag staged for user push.

3b. `docs/PLAN_V2.10.md`
   - v2.10 execution history — Phases 1-8 SHIPPED 2026-05-16
   - tag `v2.10.0` on commit `db6527c` pushed to GitHub

4. `AGENTS.md`
   - hard project invariants
   - architecture constraints
   - source-of-truth rules for agents

5. `docs/AGENT_GOVERNANCE.md`
   - evidence, completion, review, and documentation-budget rules

6. `docs/DECISIONS.md` and `docs/QUALITY_GATES.md`
   - design decisions and acceptance thresholds
   - quality gates used to decide pass/fail

Read `docs/ARCHITECTURE.md` when changing core pipeline behavior. **Current canonical baseline is `docs/QUALITY_SNAPSHOT_2026-05-20_v2.11_soak_qwen3.md`** — v2.11 Phase 1 challenger soak with the post-swap production numbers. Production embedder is Dashscope `text-embedding-v4` against `mmrag_v2_8__qwen3_dashscope`; legacy `mmrag_v2_8` (Ollama llava) retained through 2026-06-19 as the 30-day rollback baseline. Strict-gate state from v2.10 Phase 8 (`docs/QUALITY_SNAPSHOT_2026-05-16_v2.10_after.md`) is unchanged because the v2.11.0 swap touches retrieval-side only, not extraction / chunking / validation. The v2.11.0 release pin for the soak Format judge axis is **≥85%** (89.8% actual; coverage-reveal of pre-existing OCR/scan format imperfections in scanned/form docs that the v2.10 baseline never retrieved — see `docs/DECISIONS.md` "v2.11.0 Embedder Swap Executed — Format Gate Downgrade"); v2.11.x recovery target ≥95%, v2.12 reverts to ≥96% after two consecutive recovery soaks. When a metric appears in both a layer-1 status doc and a dated snapshot, the latest snapshot is canonical (per `docs/AGENT_GOVERNANCE.md` Canonicality Rule). The archived task logs are archaeology only — current task state lives in `docs/PROJECT_STATUS.md`.

## Layer Model

### Layer 0: Contracts

Stable rules. These should change rarely.

- `AGENTS.md`
- `CLAUDE.md`
- `docs/AGENT_GOVERNANCE.md`
- `docs/DECISIONS.md`
- `docs/QUALITY_GATES.md`
- `docs/ARCHITECTURE.md`

### Layer 1: Current State

Compact project status. These files should be updated whenever a session changes direction, baseline, or known quality state.

- `docs/PROJECT_STATUS.md`
- `docs/QUALITY_SNAPSHOT_2026-05-20_v2.11_soak_qwen3.md` — **current canonical baseline** (v2.11 Phase 1 challenger soak; production embedder Dashscope `text-embedding-v4`)
- `docs/QUALITY_SNAPSHOT_2026-05-16_v2.10_after.md` — v2.10 corpus strict-gate baseline (34/34 PASS); unchanged in v2.11 (retrieval-side swap only)
- `docs/QUALITY_SNAPSHOT_2026-05-16_v2.10_soak.md` — v2.10 soak (Format 98.3%, Recall@1 2.1%); baseline for v2.11 Phase 1 delta column
- `docs/QUALITY_SNAPSHOT_2026-05-11_v2.9.0-rc1_after.md` — v2.9.0-rc1 ship state (kept for v2.10 delta column)
- `docs/QUALITY_SNAPSHOT_2026-05-11_v2.9_strict_gate_full_corpus.md` — v2.9.0-rc1 strict-gate full-corpus reading
- `docs/QUALITY_SNAPSHOT_2026-05-09_v2.9_phase4_after.md` — v2.9 Phase 4 closure (historical; SUPERSEDED banner)
- `docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md` — v2.8.0 SHIPPED reference baseline

### Layer 2: Work Logs And Execution

Operational commands and historical notes.

- `docs/PLAN_V2.11.md` — v2.11 plan Draft v1.0; Phase 1 swap shipped locally; tag staged for user push
- `docs/PLAN_V2.10.md` — v2.10 execution history; Phases 1-8 SHIPPED 2026-05-16; tag `v2.10.0` on commit `db6527c` pushed to GitHub
- `docs/TESTING.md`
- `docs/CONVERSION_PROFILES.md`
- `docs/PHASE_A_MISSING_PAGES_DIAGNOSTIC.md` — v2.9 Phase A diagnostic notes (historical)
- `docs/PHASE_B3_CROSS_PAGE_SPLIT_DIAGNOSTIC.md` — v2.9 Phase B3 diagnostic notes (historical)
- `docs/PHASE_5_DEVLIN_HEADING_DIAGNOSTIC.md` — v2.10 Phase 5 diagnostic notes (historical)
- `docs/PHASE_6_FIREARMS_OCR_HEADING_DIAGNOSTIC.md` — v2.10 Phase 6 diagnostic notes (historical)

## Update Rules

- If a hard invariant changes, update `AGENTS.md` and record the rationale in `docs/DECISIONS.md`.
- If a completion claim or evidence claim changes, apply `docs/AGENT_GOVERNANCE.md`.
- If adding documentation, obey the documentation budget in `docs/AGENT_GOVERNANCE.md`.
- If the quality baseline changes, create or update a dated `docs/QUALITY_SNAPSHOT_*.md`.
- If task status or recommended next step changes, update `docs/PROJECT_STATUS.md`.
- Do not bury current-state information in chat history only.
