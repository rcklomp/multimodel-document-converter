# V3 Deferred Tests (skip registry)

This file is the **registry of unconditionally-skipped `V3_DEFERRED` tests**.
`tests/test_repo_integrity.py` G6 enforces that every such skip appears here, so
deferred coverage cannot rot off the books. Per `docs/V3_EXECUTION_MANDATE.md`
§3, each entry must be DISPOSITIONED - restored, deleted-by-decision, or deferred
with an owner + un-defer trigger. "Permanent" deferral is not a valid state.

## History correction (2026-05-31)

- The old premise "deferred until the Phase B LLM-sanitization layer subsumes
  these heuristics" is **retired**: that layer's hypothesis was **falsified**
  (Identity-Gate residual delta proved to be upstream of the V3 boundary, not
  closable by any text-only post-processor). Deferring to it = deferring to
  something that does not work.
- Earlier revisions of this file listed ~90 tests inherited from the retired
  `v3_execution_root` sandbox. That was misleading: the large majority either
  RUN AND PASS today (the retrieval / HyDE / BM25 / Qdrant / refiner /
  token-validator / telemetry subsystems are NOT deferred - their tests are
  green) or were deleted. The authoritative deferred surface is the small set
  below. Listing passing tests as "deferred" is what made the deferral picture
  look like fiction; it has been removed.

## The actual deferred surface (6 unconditionally-skipped modules)

Each is module-skipped with `pytestmark = pytest.mark.skip(reason="V3_DEFERRED -
...")`. Disposition column follows MANDATE §3.

- `tests/test_ocr_path_heading_propagation.py` - pins heading propagation across
  the OCR/extraction path. **Disposition: RESTORE** (owner: PLAN_V3.1 P2;
  un-defer trigger: the UIR-native TOC heading pass lands and qa HEADING gate
  passes). NOT sanitization-dependent - heading hierarchy is TOC-driven.
- `tests/test_cross_chunk_semantic_stitching.py` - pins the mid-sentence /
  trailing-preposition merge. **Disposition: ADOPT-or-restore** - the underlying
  `_merge_mid_sentence_chunks` still executes on the V3 path (PROJECT_STATUS
  "Phase B Technical Debt" #2); PLAN_V3.1 P3 decides adopt (un-skip) vs remove.
- `tests/test_docling_postprocess_ocr_gating.py` - pins the post-Docling OCR
  heading-override heuristic. **Disposition: DEFERRED** (owner: PLAN_V3.1 P3
  restore-or-delete; trigger: P3 disposition).
- `tests/test_docling_postprocess_profile_integration.py` - pins the post-Docling
  y-sort + drop-cap heuristics. **Disposition: DEFERRED** (PLAN_V3.1 P3).
- `tests/test_vision_aided_front_matter.py` - pins VLM-aided front-matter pickup.
  **Disposition: DEFERRED** (PLAN_V3.1 P3; candidate to fold into the VLM-native
  engine rather than restore as a separate heuristic).
- `tests/test_pdf_conversion_plan.py` - pins the v2.16 `PdfConversionPlan`
  profile-specific policy fields; the V3 engine uses a fixed policy.
  **Disposition: DELETE-by-decision candidate** (PLAN_V3.1 P3 - record the
  removed contract in `docs/DECISIONS.md`, then drop the test).

## Identity Gate

The Identity-Gate script (`scripts/run_identity_gate.py`, NOT YET BUILT) is no
longer a Definition-of-Done step. MANDATE §2 RETIRED the "< 5% delta vs v2.16"
criterion (impossible by design once the V3 chunker changes shape) in favor of an
explained-delta review (identity-half >= 95% + explained-delta <= 5%) plus the
production-CLI lane smoke. If a scripted identity check is wanted later, build it
against VLM-native output with calibrated tolerances, not v2.16 HybridChunker
output.
