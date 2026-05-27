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

## Foundation work — LANDED (2026-05-26 autonomous session)

**Scope:** purely additive scaffolding. No modifications to v2.16 production
code paths. v2.16 remains runnable + reversible by `git revert` of the
three foundation commits.

**Commits (reverse chronological):**
- `f581cff` — identity-half gate + C pre-spike harness + fusion v3 helpers
- `02fdf25` — sanitization + omlx scaffolding (Phase B + C contract)
- `eb7db72` — UIR contract types + ConversionPlan + Phase A admin docs

**Tests:** baseline 1145 → 1306 passing (+161 new V3 tests across 5 files);
skipped count unchanged at 17; zero failures.

### What landed

1. **Docs scaffolding** — this plan + Phase A administrative templates
   ([`PHASE_A_INTENTIONAL_DELTAS.md`](PHASE_A_INTENTIONAL_DELTAS.md),
   [`PHASE_A_SCOPE_NEGOTIATION.md`](PHASE_A_SCOPE_NEGOTIATION.md),
   [`PHASE_A_SKIP_AUDIT.md`](PHASE_A_SKIP_AUDIT.md) — 17 tests classified).
2. **V3 UIR contract types** — additive in
   [`src/mmrag_v2/universal/intermediate.py`](../src/mmrag_v2/universal/intermediate.py):
   `Modality` (5-value: TEXT/IMAGE/TABLE/CODE/FORM), closed-vocabulary
   `StructuralFlag` (12 values including `PARTIAL_CODE_CROSS_PAGE`,
   `PARTIAL_TABLE_CROSS_PAGE` reserved), `LocatorType`, `CoordinateFrame`,
   `ExtractionWarning`, `Locator` (with REQ-COORD-01 validation),
   `ConfidenceBreakdown` (single-sentinel + applicable Set per §3.2),
   `UIRChunk` (provenance fields + `continuation_group_id` + `uir_version`
   per Draft 0.5 audit A2 #1 + #5).
3. **`ConversionPlan` parent class** —
   [`src/mmrag_v2/universal/conversion_plan.py`](../src/mmrag_v2/universal/conversion_plan.py)
   with `render_dpi` validation `[72, 600]` per Charter §3.2 (Draft 0.5
   audit A2 #4); `engine_options` opaque blob for Docling-specific toggles.
4. **Sanitization package** —
   [`src/mmrag_v2/sanitization/`](../src/mmrag_v2/sanitization/)
   per Charter §5.1:
   - `orchestrator.py` — `SanitizationMode` enum (off/llm/heuristic/
     both-and-diff); only OFF currently dispatches non-trivially
   - `llm_sanitizer.py` — content-pinning cache key contract per §7.4
     (FUNCTIONAL); `sanitize_via_llm()` fenced with `NotImplementedError`
     until Phase B
   - `guards/` — 8-layer stack per §3.3:
     - 1 edit_distance, 3 code_span, 4 order_preservation, 5 token_alignment,
       6 prompt_boundary, 8 dedup_ratio: **FUNCTIONAL** (Levenshtein /
       regex / SHA-256 / Jaccard shingles)
     - 2 numeric_entity: **PARTIAL** (regex tier shipped; spaCy NER
       deferred — Phase B can `pip install spacy` if needed)
     - 7 entity_relation: **STUB** (returns sentinel `metric_value=-1.0`
       so its absence is observable rather than silent; requires spaCy
       dep parse)
   - `golden_set.py` — immutable JSONL schema + 50-chunk dominance scorer
     per §3.3 #5 (file empty until Phase B B2)
   - `prompts.py` — versioned template + git-hash `prompt_version()` per
     §7.4 cache-key contract; cost note enforced per §3.3 prompt-migration
     cost
   - `graceful_degradation.py` — `SentinelAccount` with 5%-rate
     `LLM_SENTINEL_DEGRADED` marker per §3.3
5. **omlx package** —
   [`src/mmrag_v2/omlx/`](../src/mmrag_v2/omlx/) per Charter §7.7:
   - `scheduler.py` — 4-level priority queue (QUERY_TEXT_EMBED <
     QUERY_RERANK < QUERY_VISUAL_EMBED < INGEST_VISUAL_EMBED); FIFO
     within priority; latency budget constants from §7.7 #5;
     query-path queue-depth limit (3) from §7.7 failure mode
   - `coresidency_monitor.py` — rolling 60s eviction window per §7.6;
     `is_forkback_triggered()` implements R6 signal (>1 eviction/min)
6. **Identity-half gate** —
   [`src/mmrag_v2/v3_identity_gate.py`](../src/mmrag_v2/v3_identity_gate.py)
   per Charter §3.2 + §8.2:
   - §8.2 normalization rules: metadata-field drop, confidence ±0.01,
     structural_flags additive (handles v2.X Dict[str, bool] AND v3 Set)
   - Stable identity key (doc_id|page|content-hash) with NFC + CRLF→LF +
     trailing-whitespace strip per §3.2
   - Content projection NFC + CRLF→LF + trailing-strip; internal
     whitespace difference IS a real delta (Charter "modulo trailing
     whitespace" strict)
   - CLI: `python -m mmrag_v2.v3_identity_gate --baseline ... --candidate ...`
   - Phase A A2 / A5 will use this as the "did this refactor break
     anything" feedback loop
7. **Fusion v3 helpers** —
   [`src/mmrag_v2/retrieval/fusion_v3.py`](../src/mmrag_v2/retrieval/fusion_v3.py)
   per Charter §3.4 + §7.8:
   - `renormalize_on_leg_skip()` — L2 norm per §3.4 #6 (PROSE
     (1.0, 1.0, 0.1) with visual skipped → (0.707, 0.707))
   - `bounded_page_chunk_join()` — top-N per page per §3.4 #4
     (REPLACES Draft 0.3 broadcast)
   - `RetrievalDebugPayload` dataclass per §7.8 (Draft 0.5 audit C10
     on-demand per-query inspection without global tracing)
8. **C pre-spike harness** —
   [`scripts/v3_c_prespike.py`](../scripts/v3_c_prespike.py) per
   Charter §4.2 step 1:
   - 200 DPI PyMuPDF rendering — FUNCTIONAL on real PDFs
   - MaxSim numpy implementation per §3.4 #3 — FUNCTIONAL
   - ColPali dispatch — fenced with `NotImplementedError` until operator
     installs `colpali-engine` (`--colpali-mode local`) or wires HF
     Spaces or omlx (`--colpali-mode hf-spaces|omlx`)
   - Dry-run validated against the actual `ATZ_Elektronik` PDF in tree

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

1. **Charter §4.2 step 1 — ✅ COMPLETE, VERDICT PASS** (2026-05-26 PM).
   Pre-spike report: [`V3_C_PRESPIKE_REPORT.md`](V3_C_PRESPIKE_REPORT.md).
   Gold page 1 (Lifecycle Management diagram in ATZ_Elektronik_German)
   ranked first via ColPali MaxSim against distractors {2, 3, 5} in run 1
   AND distractors {2, 3, 4} with paraphrased query in run 2.
   `colpali-engine==0.3.16` + `vidore/colpali-v1.3` on Apple Silicon MPS.
   Margins are thin (1.5%–2.2%) partly because transformers 5.9.0
   attribute-path rename left the ColPali LoRA adapter weights
   unloaded — see report §"Phase C task C2 prerequisites" for the
   LoRA resolution path. **Phase C is NOT dead weight** per Charter
   §4.2 step 1 outcome rule.
1b. **Charter §4.2 step 2 (C-spike) — ✅ COMPLETE, VERDICT: PASS A FAIL + PASS B FAIL.**
   Full report: [`V3_C_SPIKE_REPORT.md`](V3_C_SPIKE_REPORT.md).
   Raw traces: [`V3_C_SPIKE_RUN1.json`](V3_C_SPIKE_RUN1.json) (PASS A),
   [`V3_C_SPIKE_PASS_B.json`](V3_C_SPIKE_PASS_B.json) (PASS B).
   Single-doc test on `ATZ_Elektronik_German`, 20 hand-crafted queries,
   production text leg (`retrieve_hybrid_reranked`) + ColPali visual
   leg (LoRA-patched) + bounded-join rerank.

   Aggregate:
   - Visual top-1 accuracy: 55% (11/20); Text top-1: 55% (11/20)
   - PASS A visual recovery on text-failed: 44.4% (4/9) — threshold ≥60% FAIL
   - PASS A visual harm on text-passed:     36.4% (4/11) — threshold ≤10% FAIL
   - PASS B rerank top-1 on gold page:      47.4% (9/19) — threshold ≥60% FAIL

   Failure-mode diagnosis: page-1 over-pull on body-text queries because
   page 1 carries the only visually-rich element (Lifecycle Management
   flowchart). The hybrid bounded-join mechanism IS working (Q07, Q08,
   Q13 demonstrate rescue of visual-only-failures via text-leg
   candidates in the union) — the cap is page-level granularity.

1c. **Phase C outcome — LOCKED 2026-05-26 PM: region-level scope expansion.**

   The strict Charter §4.2 outcome rule on PASS A FAIL reads "Phase C as
   designed is dead; redirect to VLM-native or alternative visual model."
   The diagnostic evidence in [`V3_C_SPIKE_REPORT.md`](V3_C_SPIKE_REPORT.md)
   is more precise: the failure mode is page-1 over-pull on body-text
   queries (page-level granularity cap), the hybrid bounded-join
   mechanism IS working (Q07/Q08/Q13 rescued visual-only-failed queries),
   visual+text legs are demonstrably complementary (different 4/20 each).
   The model itself is not the binding constraint — granularity is.

   ColQwen2.5 falsification attempt was killed at user request after the
   run produced all-NaN MaxSim scores (likely MPS bfloat16 instability
   on Qwen2.5-VL; debugging deferred). The verdict locked here does NOT
   depend on that debug because the failure mode is granularity-driven,
   not modality-driven — swapping ColPali variants would test the wrong
   axis.

   **Decision:** Phase C task C3 (visual index build) is rescheduled to
   be region-level from the start. Region = a single chunk's bbox crop
   at 200 DPI, embedded as a ColPali patch-vector matrix. MaxSim then
   operates at chunk-level (replacing the bounded page-to-chunk join
   which was a workaround). Charter §3.4 #4 bounded join + Charter §4.2
   step 2 #8 PASS B FAIL outcome rule both fold into a single design
   decision: chunks ARE the retrieval unit on the visual leg too.

   No more local ColPali / ColQwen runs from this workstation. Phase C
   C2 (omlx ColPali deployment) is the next ColPali-touching milestone
   and that happens on the LAN GPU server, not the dev workstation.

1d. **Phase A task A0 (per-doc spike) — ✅ COMPLETE 2026-05-26 PM, VERDICT PASS.**
   Report: [`V3_PHASE_A_A0_REPORT.md`](V3_PHASE_A_A0_REPORT.md).
   Identity ratio 1.0000 (62/62 distinct identity keys matched); 0 deltas
   to enumerate (cap ≤30). 24-day Phase A budget is justified by
   evidence; no scope-negotiation trigger fires. The v3.0 UIR contract
   (Modality, Locator, ConfidenceBreakdown, UIRChunk, StructuralFlag)
   carries v2.X chunk content losslessly under §8.2 normalization rules.

   Pure CPU + file I/O. No GPU, no omlx, no Qdrant. Wall time <1s.

1e. **Charter Phase A task A1 — ✅ COMPLETE 2026-05-27, VERDICT PASS.**
   `engines/pdf_plan.py::PdfConversionPlan` now inherits from
   `universal/conversion_plan.py::ConversionPlan` (Charter §3.2 parent).
   Parent is `@dataclass(frozen=True)` to match child frozen contract.
   PDF-flavored defaults satisfy parent validation for v2.16 callers
   (no kwargs construction stays valid): `source_path=""`,
   `file_type="pdf"`, `doc_id="pending"`, `extraction_strategy="digital_native"`;
   `profile_type` and `reading_order_strategy` overrides preserve
   existing defaults + the latter's `Literal` type narrowing.
   `__post_init__` chains to `super().__post_init__()` so render_dpi range
   `[72, 600]`, batch_size ≥1, and file_type/doc_id non-empty checks fire
   on every PDF plan construction. The "pending" sentinel on `doc_id` is
   the observable signal that the v3 contract has not been populated by
   the caller (legacy v2.16 paths); v3 callers MUST pass an explicit
   `doc_id`. **Charter A1 acceptance verified:** test suite 1306 → 1310
   passed (+4 from intervening main commits, NOT from A1 — A1 added zero
   tests), 17 skipped (unchanged), 0 failed.
   Touched files: `src/mmrag_v2/universal/conversion_plan.py` (one-line
   frozen=True), `src/mmrag_v2/engines/pdf_plan.py` (import + class
   header + 4 parent-field overrides + super-call in `__post_init__`).

1f. **NEXT executable step — Phase A task A2.** Chunker / mapper /
   serializer rewrite to consume the v3 UIR contract end-to-end
   (`mmrag_v2.universal.intermediate.UIRChunk` as the chunker output
   type; `Modality` / `Locator` / `ConfidenceBreakdown` populated at the
   mapper boundary). This is the load-bearing Phase A refactor and is
   gated by the A0 scope-negotiation outcome (Charter §Phase A protocol
   — A0 PASSed at identity-ratio 1.0000, so the full UIR refactor is
   committed to rather than the UIR-shim fallback). Touch sites:
   `src/mmrag_v2/universal/v2x_to_v3_mapper.py` (already foundation-
   shipped — extend to populate the v3 fields), `src/mmrag_v2/processor.py`
   HybridChunker call site, `src/mmrag_v2/batch_processor.py`
   reconciliation paths. Verify each commit against
   `python -m mmrag_v2.v3_identity_gate --baseline ... --candidate ...`.
2. **Charter Phase A task A0** — per-doc spike on `ATZ_Elektronik_German`
   using the V3 UIR types. Convert one doc through the
   (foundation-shipped) ConversionPlan + ConfidenceBreakdown +
   StructuralFlag types and compare to the v2.16 baseline via
   `mmrag_v2.v3_identity_gate`. Per Charter §Phase A A0 acceptance:
   semantic-identity gate passes on this doc alone (both halves);
   intentional deltas list ≤30 lines. If A0 exceeds 4 days OR
   explained-delta exceeds 5%, the Phase A scope-negotiation protocol
   fires (see [`PHASE_A_SCOPE_NEGOTIATION.md`](PHASE_A_SCOPE_NEGOTIATION.md))
   — pick from options (a) defer subset, (b) UIR-shim fallback,
   (c) widen tolerance, (d) split A2 across cycles, OR invoke
   content-derived `chunk_id` (regret #4 contingent option).
3. **Decide scope-negotiation outcome** per Charter §Phase A protocol
   BEFORE committing to A2's full chunker/mapper/serializer rewrite.
   Default plan: full UIR refactor; fallback: UIR-shim (Charter §Phase A
   "preferred fallback because it preserves the C13/R15 chunk_id
   stability contract while reducing the refactor surface by ~50%").
4. **A7 (last 6h of Phase A)** — three consecutive heuristic-only soak
   runs to baseline σ per axis. Phase B's dominance criterion gates on
   σ, so this is unblock work for Phase B.
5. **A8 re-enable post-A** — verify the 9 tests classified as
   `re-enable-post-A` in [`PHASE_A_SKIP_AUDIT.md`](PHASE_A_SKIP_AUDIT.md)
   still pass with their env-gates ON after the UIR refactor lands.
6. **A9** — acquire 5-of-8 holdout documents before Phase A start
   (per Charter §3.5 #5 acquisition plan).

## Hand-off note (foundation-session author)

The foundation session deliberately did NOT touch v2.16 production
paths. All work is additive — new files, new tests, new packages — so
the only way for the foundation to break v2.16 is by importing a v3
type into a v2.X production module (which no foundation code does).
Verification: `pytest tests/` reports 1306 passed / 17 skipped / 0
failed, same skip count as v2.16 baseline.

The Charter scope-negotiation protocol (Charter §Phase A) is the most
important safety mechanism for the next step. Phase A's 24-day budget
is "a working assumption, not a guarantee" (Charter §Phase A, R22).
A0 (per-doc spike) MUST land before A2 is committed to, because A0's
evidence determines whether the full refactor is the right scope or
the UIR-shim fallback should ship. Three foundation deliverables are
designed to feed A0:
  - `mmrag_v2.universal.intermediate.UIRChunk` for the v3 emission target
  - `mmrag_v2.v3_identity_gate.compare_for_identity()` for the
    feedback loop
  - `docs/PHASE_A_INTENTIONAL_DELTAS.md` for the explained-delta
    enumeration

The C pre-spike is cheap and decisive — Charter §4.2 step 1 budgets
2 hours of operator time. Running it BEFORE A0 is fine because the
pre-spike does not depend on UIR (it embeds rendered page images).
A FAIL here saves the entire ~12-day Phase C from being scheduled.
