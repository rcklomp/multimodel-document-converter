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
  closable by any text-only post-processor).
- Earlier revisions listed ~90 tests inherited from the retired
  `v3_execution_root` sandbox. The large majority either RUN AND PASS today
  (retrieval / HyDE / BM25 / Qdrant / refiner / token-validator / telemetry are
  NOT deferred) or were deleted. The authoritative deferred surface is the small
  set below.

## Restored (no longer deferred - kept here as audit trail)

- `tests/test_ocr_path_heading_propagation.py` - **RESTORED 2026-05-31** (P2).
  The un-defer trigger fired: the UIR-native TOC heading pass landed
  (`uir_chunker._assign_headings`) and the qa HEADING gate now PASSES. 76 green
  incl. 7 new `_assign_headings` contracts. Two prior structural pins on DELETED
  OCR/layout-lane wiring were DELETED-by-decision (see `docs/DECISIONS.md`
  "OCR-lane production-wiring pins retired (PLAN_V3.1 P2)").
- `tests/test_cross_chunk_semantic_stitching.py` - **RESTORED 2026-06-01** (P3).
  Un-skipping it surfaced a Phase A REGRESSION: `_apply_final_boundary_repairs`
  (the bridge running `_merge_hungry_operators` + `_strip_trailing_headings` +
  `_merge_mid_sentence_chunks` + dedup) was DEFINED but never called by
  `process_pdf`, so those repairs ran on no document. Disposition: RE-WIRE (the
  user's call) - the bridge + the sibling `_apply_vision_aided_front_matter_detection`
  were restored into `process_pdf` finalize (after `_apply_quality_filters`,
  before the export sanitizer). All 9 tests green; full suite 1313/111/0, no
  regressions. See `docs/DECISIONS.md` "Phase A orphaned the final-boundary-repair
  bridge - RE-WIRED (PLAN_V3.1 P3)".

## The actual deferred surface (4 unconditionally-skipped modules)

Each is module-skipped with `pytestmark = pytest.mark.skip(reason="V3_DEFERRED -
...")`. Disposition column follows MANDATE §3.

- `tests/test_vision_aided_front_matter.py` - pins the VLM-aided front-matter
  demotion. **Disposition: ADOPT-pending** - NOTE: `_apply_vision_aided_front_matter_detection`
  was RE-WIRED into `process_pdf` on 2026-06-01 (it was the sibling of the
  cross_chunk bridge). This module should now be un-skippable; it stays listed
  until its tests are run + fixed to the current call shape (next P3 step).

### Legacy V2 Docling-lane cluster (3 modules) - DELETE-by-decision on lane retirement

These three guard the legacy non-batch `V2DocumentProcessor` / `DoclingPdfAdapter`
/ `PdfConversionPlan` path, NOT the V3 chunker. Per `docs/DECISIONS.md` "Legacy
V2DocumentProcessor / Docling lane - retirement PLANNED (PLAN_V3.1 P3,
2026-05-31)" the lane is slated for retirement, blocked only on V3 gaining
non-PDF (EPUB/HTML/DOCX/PPTX/XLSX) extraction. **Disposition: DELETE-by-decision
WITH the guarded code when the legacy lane is cut** (its own future phase, P6);
NOT adopted (the lane is on a retirement path) and NOT deleted yet (the code
still ships for non-PDF + `--batch-size 0`). Trigger: legacy-lane retirement.

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
