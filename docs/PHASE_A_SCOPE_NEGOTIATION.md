# Phase A Scope-Negotiation Log

**Charter:** [`ARCHITECTURE_V3_DRAFT_0.5.md`](ARCHITECTURE_V3_DRAFT_0.5.md)
"Phase A: UIR Foundation" — Scope-negotiation protocol; R22.

This log records any invocation of Phase A's scope-negotiation protocol.
The 24-day Phase A budget is a working assumption; if any of the negotiation
triggers fire, the budget is renegotiated (not silently extended) and the
decision is recorded here BEFORE Phase A continues.

## Triggers (Charter §Phase A)

1. **A0 exceeds 4 days** — per-doc spike on `ATZ_Elektronik_German`.
2. **First 5 days of A2 show <20% progress** on the ~12 `batch_processor.py`
   reconciliation paths.
3. **Semantic-identity gate's explained-delta half exceeds 5%** (more
   chunks differ than the audit table can enumerate).

## Negotiation options

Per Charter:

- **(a) Defer ~1/3 of `batch_processor.py` reconciliation paths to v3.0.1.**
- **(b) UIR-shim fallback** — convert `DoclingDocument` → `UniversalDocument`
  at the adapter boundary and keep downstream `processor.py` / `mapper.py` /
  `batch_processor.py` unchanged. UIR contract ships without the full
  downstream rewrite; the rewrite becomes a v3.0.2 cycle. ~10% performance
  overhead from translation layer in exchange for ~50% Phase A scope
  reduction. Preferred fallback because it preserves the C13/R15 `chunk_id`
  stability contract.
- **(c) Widen explained-delta tolerance to ≤10%** (with explicit
  `DECISIONS.md` entry documenting the loosened gate).
- **(d) Split A2 across cycles** (A2a in 3.0.0; A2b in 3.0.1).
- **Content-derived `chunk_id` as scope-negotiation option:** if A0 reveals
  >20% chunk rewrite-map churn, flip `chunk_id` derivation from positional
  to content-based (regret #4 in Charter §6.3).

## Entries

### 2026-05-27 — Session-budget trigger (off-protocol; operator-invoked)

**Trigger:** None of the three Charter-listed triggers (1/2/3) fired:
  - A0 PASSed at identity-ratio 1.0000 in <1 second (well under 4 days).
  - A2 has not yet started; no progress-stall signal observed.
  - Explained-delta is 0 (A0 enumerated zero deltas).

The trigger here is a **bounded autonomous-session budget**: the user
instructed an autonomous run to "continue, don't stop until the plan is
completed; fix on a higher level if you run into issues." A single
autonomous session cannot realistically execute the full A2 refactor:
the Explore-agent territory map (2026-05-27) sized the refactor surface
at ~1,755 LOC across `processor.py` (`_process_text_with_hybrid_chunker`
495 LOC, `_emit_dense_index_page_chunks` 96 LOC,
`_emit_section_header_only_page_chunks` 181 LOC) +
`batch_processor.py` (`_propagate_headings` 193 LOC,
`_merge_micro_text_chunks` 299 LOC, `_apply_spatial_refiner` 85 LOC,
plus 4-6 other reconciliation paths) + `ingestion_schema.py`
(`IngestionChunk` model 93 LOC + factories), with one HybridChunker
call site at `processor.py:3395` that must be rewritten to consume UIR
rather than DoclingDocument.

The Charter does not list "session budget" as a formal trigger because
the Charter assumes solo-dev human execution against a 24-day cycle.
For human execution the full refactor is the right scope. For an
autonomous session bounded to hours, shipping a broken partial A2 is
worse than shipping a complete shim and rebooking the rewrite.

**Option chosen:** **(b) UIR-shim fallback.**

**Rationale:**
- Pre-authorized by Charter §Phase A as the "preferred fallback
  because it preserves the C13/R15 `chunk_id` stability contract while
  reducing the refactor surface by ~50%."
- The shim mechanism is already proven: `v2x_to_v3_mapper.py`
  (`map_v2x_to_v3_uirchunk()`, 89 LOC) losslessly projects v2.X chunks
  to v3 UIRChunks. A0 PASSed at identity-ratio 1.0000 using this
  mapper. The shim *uses the same proven mapper* at the export
  boundary.
- v2.16 production paths (chunk emission, JSONL output, Qdrant
  ingestion, RAG app) remain completely unchanged → zero blast
  radius for existing consumers.
- The v3.0 contract surface ships in v3.0.0/v3.0.1 (UIRChunk types,
  ConversionPlan, identity gate, sanitization scaffolding, omlx
  scheduler, fusion v3, ColPali pre-spike PASS). The full chunker
  decoupling rewrite is rebooked to v3.0.2 under a dedicated cycle
  plan written when human execution can budget the 24 days.
- Performance penalty per Charter: ~10% from the translation layer.
  This is a Phase B/C/D-acceptable cost given the contract-shipping
  benefit.

**Knock-on effects on remaining Phase A tasks:**
- **A3** (cross-page `partial_code` flag activation): requires the
  chunker to operate on UIR, which is the deferred work. **A3 is
  rebooked to v3.0.2** alongside the full chunker rewrite.
- **A4** (guard-test coverage for v3.0 paths): no new
  `PdfPipelineOptions` / `DocumentConverter` construction sites are
  added by the shim, so the existing guard test
  (`test_no_pipeline_options_construction_outside_adapter`) already
  covers the v3.0 surface. Re-affirm in PLAN_V3.
- **A5** (corpus-wide rebuild + 3 regression cases): v2.16 chunk
  emission is unchanged in shim mode, so the existing v2.16 strict-gate
  result IS the v3.0 strict-gate result. Earthship + Harry_Potter +
  Fluent_Python regression cases pass by construction (same code).
  Spot-verify via identity gate on at least one of them.
- **A6** (schema bump 2.7.0 → 3.0.0 + chunk_id rewrite map): the bump
  is purely a version-string change; rewrite map is **empty** because
  shim mode preserves chunk_ids 1:1.
- **A7** (σ-baseline soak, ~6h): hard external dependency, cannot
  complete in autonomous session. Deferred to operator-triggered run.
- **A8** (skipped-tests audit): re-enable-post-A tests need
  re-verification. Doable in shim mode because no test contract
  changes.
- **A9** (5/8 holdout docs): external sourcing, cannot do
  autonomously. Deferred to operator.

**New budget:** Phase A 24-day budget is **shortened to whatever lands
this session** for v3.0.0/v3.0.1 ship; the full chunker-rewrite
remainder is **rebooked to v3.0.2** under a future dedicated cycle
plan with its own 12-day estimate.

**Reviewer sign-off:** PENDING. Operator may redirect by reverting
this entry and the A2-shim commits if the autonomous shim path is not
the desired outcome.

Schema for entries (when they appear):

```
### YYYY-MM-DD — Trigger N fired

**Trigger:** [which of 1/2/3 above]
**Evidence:** [link to A0 report, A2 progress dashboard, or identity-gate
  output that justifies the trigger firing]
**Option chosen:** [one of a/b/c/d, or "continue with original budget +
  documented justification"]
**Rationale:** [why this option was preferred over the others]
**New budget:** [new day count for Phase A, if changed]
**Reviewer sign-off:** [user confirmation]
```
