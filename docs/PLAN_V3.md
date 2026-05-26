# V3.0 Execution Plan — Implementation Tracking

**Status:** ACTIVE (autonomous foundation work started 2026-05-26)
**Charter:** [`docs/ARCHITECTURE_V3_DRAFT_0.5.md`](ARCHITECTURE_V3_DRAFT_0.5.md)
**Predecessor:** v2.16.0 (shipped 2026-05-25, tag `15d1349`)

This document tracks V3.0 implementation against the Charter (Draft 0.5).
The Charter is authoritative; this plan is execution state only.

## V3.0 Phases

Per Charter §4:

| Phase | Cycle | Nominal days | Status |
|---|---|---|---|
| **Pre-A** | C pre-spike + C-spike before Phase A code | 2h + 2-3d | NOT STARTED (harness scaffolding only — see §Foundation below) |
| **A** | UIR Foundation | 24d (2× nominal per R17) | FOUNDATION-ONLY (additive scaffolding 2026-05-26) |
| **B** | LLM Sanitization | 18-22d | NOT STARTED |
| **C** | Visual Retrieval (ColPali) | 12d | NOT STARTED |
| **D** | Modality-Aware Gates | 12d | NOT STARTED |

## Foundation work (2026-05-26 autonomous session)

**Scope:** purely additive scaffolding. No modifications to v2.16 production
code paths. v2.16 must remain runnable + reversible by `git revert` of the
foundation commits.

Tasks attempted in this session (∼9 hour autonomous block):

1. **Docs scaffolding** — this plan + Phase A administrative templates
   ([`PHASE_A_INTENTIONAL_DELTAS.md`](PHASE_A_INTENTIONAL_DELTAS.md),
   [`PHASE_A_SCOPE_NEGOTIATION.md`](PHASE_A_SCOPE_NEGOTIATION.md),
   [`PHASE_A_SKIP_AUDIT.md`](PHASE_A_SKIP_AUDIT.md)).
2. **V3 UIR contract types** — additive to
   [`src/mmrag_v2/universal/intermediate.py`](../src/mmrag_v2/universal/intermediate.py):
   `Modality`, `StructuralFlag`, `LocatorType`, `CoordinateFrame`,
   `ExtractionWarning`, `Locator`, `ConfidenceBreakdown`, `UIRChunk`.
3. **`ConversionPlan` parent class** — new file
   [`src/mmrag_v2/universal/conversion_plan.py`](../src/mmrag_v2/universal/conversion_plan.py)
   with `render_dpi` validation `[72, 600]` per Charter §3.2.
4. **Sanitization package** scaffolding — new
   [`src/mmrag_v2/sanitization/`](../src/mmrag_v2/sanitization/)
   subpackage per Charter §5.1:
   - `orchestrator.py` — mode flag handling stub
   - `llm_sanitizer.py` — GX10 client + cache stub
   - `guards/` — 8 guard stubs (one file per guard)
   - `golden_set.py`, `prompts.py`, `graceful_degradation.py`
5. **omlx package** scaffolding per §7.7 tenancy policy:
   - `omlx/scheduler.py`, `omlx/coresidency_monitor.py`
6. **A8 skipped-tests audit** — classify each currently-skipped test
   (still-skip / re-enable-now / re-enable-post-A) per Charter Phase A task A8.
7. **C pre-spike harness** — scripted skeleton for the 2-hour falsification
   test per Charter §4.2 step 1.

## NOT done in foundation session

- A0 per-doc spike on `ATZ_Elektronik_German` (requires conversion runs).
- A1-A5 actual refactor work (mapper.py, processor.py HybridChunker
  call site, batch_processor.py reconciliation paths).
- A7 σ-baseline soak runs (~6h soak time).
- A9 holdout document acquisition (external sourcing).
- C pre-spike execution itself (requires `omlx` ColPali deployment).
- All of Phase B / C / D.

## Reversibility

All foundation-session commits are isolated under:
- `docs/PLAN_V3.md`, `docs/PHASE_A_*.md` (new files)
- `src/mmrag_v2/universal/conversion_plan.py` (new file)
- `src/mmrag_v2/universal/intermediate.py` (additive sections only —
  `class Modality`, `class StructuralFlag`, etc., do NOT modify
  existing `class Element`, `class UniversalPage`, `class ElementType`,
  etc., which remain the v2.16 production types).
- `src/mmrag_v2/sanitization/` (new package, no `__init__` import
  from production code paths).
- `src/mmrag_v2/omlx/` (new package, no production imports).
- `tests/test_v3_uir_contract.py`, `tests/test_v3_conversion_plan.py`,
  `tests/test_v3_sanitization_scaffold.py` (new test files).
- `scripts/v3_c_prespike.py` (new script).

`git revert <foundation_commit_hashes>` restores v2.16 state without
touching anything else.

## v2.16 production guarantees during V3 foundation

- All 1162 existing tests still pass (run after each commit).
- `mmrag-v2 process` / `mmrag-v2 batch` CLI behavior unchanged.
- Qdrant collections unchanged.
- Retrieval pipeline (`retrieve_hybrid_reranked`) unchanged.
- Schema version stays at v2.7.0 (the 3.0.0 bump happens at A6,
  long after this foundation work).

## Next executable steps after foundation

In rough priority order (the Charter governs):

1. **Charter §4.2 Step 1**: execute the 2-hour C pre-spike using the
   harness from this session against a workstation off-the-shelf ColPali
   inference (HF Spaces or local) and the gold + 3 distractor pages
   from `ATZ_Elektronik_German`.
2. **A0**: per-doc spike on `ATZ_Elektronik_German` using the V3 UIR
   types — does the semantic-identity gate pass on this doc alone
   with both halves (identity ≥95% + explained-delta ≤5%)?
3. Decide scope-negotiation outcome per Charter §Phase A protocol:
   full UIR refactor vs UIR-shim fallback.
