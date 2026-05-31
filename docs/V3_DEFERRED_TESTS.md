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

## The actual deferred surface (5 unconditionally-skipped modules)

Each is module-skipped with `pytestmark = pytest.mark.skip(reason="V3_DEFERRED -
...")`. Disposition column follows MANDATE §3.

- `tests/test_ocr_path_heading_propagation.py` - **RESTORED 2026-05-31** (no
  longer skipped, no longer deferred). The un-defer trigger fired: PLAN_V3.1 P2
  landed the UIR-native TOC heading pass (`uir_chunker._assign_headings`, fed
  the PyMuPDF TOC threaded from `batch_processor._toc_for_batch`) and the qa
  HEADING gate now PASSES (academic doc with bookmarks: 68% -> 100% coverage).
  The module's 68 behavioral contracts (is_valid_heading garbage rejection +
  ContextStateV2 carry/attribution) are green, plus 7 new contracts pinning
  `_assign_headings` directly. Two prior structural pins on the DELETED OCR/
  layout-lane production wiring (`callers == 1`, `_propagate_headings(` once in
  process_pdf) were DELETED-by-decision - see `docs/DECISIONS.md` "OCR-lane
  production-wiring pins retired (PLAN_V3.1 P2)".
- `tests/test_cross_chunk_semantic_stitching.py` - pins the mid-sentence /
  trailing-preposition merge. **Disposition: ADOPT-or-restore** - the underlying
  `_merge_mid_sentence_chunks` still executes on the V3 path (PROJECT_STATUS
  "Phase B Technical Debt" #2); PLAN_V3.1 P3 decides adopt (un-skip) vs remove.
- `tests/test_vision_aided_front_matter.py` - pins VLM-aided front-matter pickup.
  **Disposition: DEFERRED** (PLAN_V3.1 P3; candidate to fold into the VLM-native
  engine rather than restore as a separate heuristic).

### Legacy V2 Docling-lane cluster (3 modules) - DELETE-by-decision on lane retirement

These three guard the legacy non-batch `V2DocumentProcessor` / `DoclingPdfAdapter`
/ `PdfConversionPlan` path, NOT the V3 chunker. Per `docs/DECISIONS.md` "Legacy
V2DocumentProcessor / Docling lane — retirement PLANNED (PLAN_V3.1 P3,
2026-05-31)" the lane is slated for retirement, blocked only on V3 gaining
non-PDF (EPUB/HTML/DOCX/PPTX/XLSX) extraction. **Disposition: DELETE-by-decision
WITH the guarded code when the legacy lane is cut** (its own future phase); they
are deliberately NOT adopted (the lane is on a retirement path) and NOT deleted
yet (the code still ships for non-PDF + `--batch-size 0`). Trigger: legacy-lane
retirement phase.

- `tests/test_docling_postprocess_ocr_gating.py` - post-Docling OCR
  heading-override + bitmap-threshold gating on the adapter.
- `tests/test_docling_postprocess_profile_integration.py` - post-Docling
  y-sort + drop-cap profile wiring through `PdfConversionPlan`.
- `tests/test_pdf_conversion_plan.py` (62 tests) - the `PdfConversionPlan` +
  `DoclingPdfAdapter` policy contract for the legacy lane.

## Identity Gate

The Identity-Gate script (`scripts/run_identity_gate.py`, NOT YET BUILT) is no
longer a Definition-of-Done step. MANDATE §2 RETIRED the "< 5% delta vs v2.16"
criterion (impossible by design once the V3 chunker changes shape) in favor of an
explained-delta review (identity-half >= 95% + explained-delta <= 5%) plus the
production-CLI lane smoke. If a scripted identity check is wanted later, build it
against VLM-native output with calibrated tolerances, not v2.16 HybridChunker
output.
