# MMRAG V2 Documentation Index

This folder uses a three-layer documentation structure so a new coding session can load the project state without reading every file.

## Read Order For New Sessions

1. `docs/PROJECT_STATUS.md`
   - current project state
   - active models/endpoints, without secrets
   - current quality baseline
   - immediate next work and per-phase status

2. `docs/QUALITY_SNAPSHOT_2026-05-23_v2.14_after.md`
   - **current canonical baseline** (v2.14.0 PUSHED 2026-05-23 —
     local-LLM accelerator stack on top of unchanged v2.13.0
     retrieval. Operative Phase 0 verdict is FP8-14B per §8 post-
     ship addendum: rel 82.2% / format 90.7% TRUSTWORTHY / faith
     76.6%. Phase 1 + Phase 6 PARTIAL, carried to v2.15.)

2b. `docs/QUALITY_SNAPSHOT_2026-05-22_v2.13_after.md`
   - v2.13.0 predecessor baseline (local omlx embedder swap + OCR
     auto-routing; apples-to-apples 6/6-axis omlx win vs dashscope
     on same fixture)

2c. `docs/QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_omlx_vs_dashscope.md`
   - v2.13 Phase 1 SWAP evidence (canonical comparison report)

2d. `docs/QUALITY_SNAPSHOT_2026-05-21_v2.12_after.md`
   - v2.12.0 predecessor baseline (retrieval stack — hybrid +
     ModernBERT rerank; Recall@1 67.8%, Recall@5 chunk 90.2%
     STRETCH on v2.11 baseline fixture)

2e. `docs/QUALITY_SNAPSHOT_2026-05-20_v2.11_soak_qwen3.md`
   - v2.11.0 baseline soak; the 518-query × 259-chunk fixture every
     v2.12 phase soak ran against

2f. `docs/QUALITY_SNAPSHOT_2026-05-16_v2.10_after.md`
   - v2.10 strict-gate baseline (corpus 34/34 PASS; unchanged in
     v2.11 + v2.12 + v2.13 + v2.14 — all four cycles changed only
     the retrieval side / OCR routing / local-LLM accelerators)

3. `docs/PLAN_V2.15.md`
   - **active cycle plan** — Draft v0.2 2026-05-24. Strategic
     Option A/E/F fork pending user decision; recommended default
     Option F (telemetry-augmented hybrid).

3b. `docs/PLAN_V2.14.md`
   - v2.14 execution history — CLOSED 2026-05-23; tag `v2.14.0`
     PUSHED to origin at commit `36482e0` (sha `122a62e`).
     Close-out header + Draft v0.5 archaeology.

3c. `docs/PLAN_V2.13.md`
   - v2.13 execution history — CLOSED 2026-05-22; Phase 1 SWAP
     (local Qwen3-Embedding-8B) + Phase 2 (OCR auto-routing) both
     shipped; tag `v2.13.0` PUSHED to origin (commit `021ef05`).

3d. `docs/PLAN_V2.12.md`
   - v2.12 execution history — CLOSED 2026-05-21; tag `v2.12.0` on
     commit `5a2ce18` public on GitHub + Gitea.

3e. `docs/PLAN_V2.11.md`
   - v2.11 execution history — CLOSED 2026-05-20; tag `v2.11.0` on
     commit `c2a461c`.

3f. `docs/PLAN_V2.10.md`
   - v2.10 execution history — CLOSED 2026-05-16; tag `v2.10.0` on
     commit `db6527c`.

4. `AGENTS.md`
   - hard project invariants
   - architecture constraints
   - source-of-truth rules for agents

5. `docs/AGENT_GOVERNANCE.md`
   - evidence, completion, review, and documentation-budget rules

6. `docs/DECISIONS.md` and `docs/QUALITY_GATES.md`
   - design decisions and acceptance thresholds
   - quality gates used to decide pass/fail

Read `docs/ARCHITECTURE.md` when changing core pipeline behavior. **Current canonical baseline is `docs/QUALITY_SNAPSHOT_2026-05-23_v2.14_after.md`** — v2.14.0 ship state (local-LLM accelerator stack on top of unchanged v2.13.0 retrieval), with §8 post-ship addendum covering the v2.14.x patch range (Phase 2 + Phase 3 + v2.14.1 GX10 swap to FP8-14B + n-gram spec rejection). Production embedder is local `Qwen3-Embedding-8B-mxfp8` via omlx-server (`10.0.10.246:8000`) against `mmrag_v2_8__qwen3_local` (4096-dim, 31,371 pts); the dashscope rollback baseline `mmrag_v2_8__qwen3_dashscope` was **dropped 2026-05-23 PM** (v2.14 Phase 3 under user "full send" override), with 90-day cold-storage snapshot retained on `multimodal-doc-converter_qdrant_snapshots` through 2026-08-21. Production local LLM is `RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic` at `http://10.0.10.239:8000/v1/chat/completions` (Asus Ascent GX10; Blackwell-native FP8; bare config — n-gram spec REJECTED). Strict-gate state from v2.10 Phase 8 (`docs/QUALITY_SNAPSHOT_2026-05-16_v2.10_after.md`) is unchanged because v2.11 / v2.12 / v2.13 / v2.14 all touched retrieval-side / OCR-routing / observability only, not extraction / chunking / validation. The v2.11.0 Format-gate downgrade to ≥85% (see `docs/DECISIONS.md` "v2.11.0 Embedder Swap Executed — Format Gate Downgrade") was structurally addressed in v2.13 via OCR auto-routing (Earthship +6.2pp) + the omlx embedder Format lift (+3.7pp on same-fixture comparison); the remaining CarOK Format penalty is now tracked as a v2.15-cycle decision (Option A/E/F) per `docs/PLAN_V2.15.md`. When a metric appears in both a layer-1 status doc and a dated snapshot, the latest snapshot is canonical (per `docs/AGENT_GOVERNANCE.md` Canonicality Rule). The archived task logs are archaeology only — current task state lives in `docs/PROJECT_STATUS.md`.

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
- `docs/QUALITY_SNAPSHOT_2026-05-23_v2.14_after.md` — **current canonical baseline** (v2.14.0 PUSHED — local-LLM accelerator stack on top of unchanged v2.13.0 retrieval; FP8-14B Phase 0 verdict TRUSTWORTHY on Format per §8 post-ship addendum)
- `docs/QUALITY_SNAPSHOT_2026-05-22_v2.13_after.md` — v2.13.0 predecessor baseline (local omlx Qwen3-Embedding-8B + OCR auto-routing; 6/6-axis apples-to-apples win)
- `docs/QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_omlx_vs_dashscope.md` — v2.13 Phase 1 SWAP evidence (canonical comparison report)
- `docs/QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_omlx.md` — v2.13 P1 omlx per-doc + weakest queries
- `docs/QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_dashscope_baseline.md` — v2.13 P1 dashscope per-doc + weakest queries
- `docs/QUALITY_SNAPSHOT_2026-05-21_v2.12_after.md` — v2.12.0 predecessor baseline (retrieval stack hybrid+rerank)
- `docs/QUALITY_SNAPSHOT_2026-05-20_v2.11_soak_qwen3.md` — v2.11.0 baseline soak; the 518-query fixture every v2.12 phase ran against
- `docs/QUALITY_SNAPSHOT_2026-05-16_v2.10_after.md` — v2.10 corpus strict-gate baseline (34/34 PASS); unchanged in v2.11 + v2.12 + v2.13 + v2.14
- `docs/QUALITY_SNAPSHOT_2026-05-16_v2.10_soak.md` — v2.10 soak (Format 98.3%, Recall@1 2.1%); baseline for v2.11 Phase 1 delta column
- `docs/QUALITY_SNAPSHOT_2026-05-11_v2.9.0-rc1_after.md` — v2.9.0-rc1 ship state (kept for v2.10 delta column)
- `docs/QUALITY_SNAPSHOT_2026-05-11_v2.9_strict_gate_full_corpus.md` — v2.9.0-rc1 strict-gate full-corpus reading
- `docs/QUALITY_SNAPSHOT_2026-05-09_v2.9_phase4_after.md` — v2.9 Phase 4 closure (historical; SUPERSEDED banner)
- `docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md` — v2.8.0 SHIPPED reference baseline

### Layer 2: Work Logs And Execution

Operational commands and historical notes.

- `docs/PLAN_V2.15.md` — **active cycle plan** (Draft v0.2 2026-05-24; Option A/E/F fork pending user decision)
- `docs/PLAN_V2.14.md` — v2.14 execution history; CLOSED 2026-05-23; tag `v2.14.0` PUSHED to origin (commit `36482e0`, sha `122a62e`); close-out header + Draft v0.5 archaeology
- `docs/PLAN_V2.13.md` — v2.13 execution history; Phase 1 + Phase 2 SHIPPED 2026-05-22; tag `v2.13.0` PUSHED to origin (commit `021ef05`)
- `docs/PLAN_V2.12.md` — v2.12 execution history; Phases 0-3 SHIPPED 2026-05-21; tag `v2.12.0` on commit `5a2ce18` public on both remotes
- `docs/PLAN_V2.11.md` — v2.11 execution history; Phase 1 swap SHIPPED 2026-05-20; tag `v2.11.0` on commit `c2a461c` public
- `docs/PLAN_V2.10.md` — v2.10 execution history; Phases 1-8 SHIPPED 2026-05-16; tag `v2.10.0` on commit `db6527c` public
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
