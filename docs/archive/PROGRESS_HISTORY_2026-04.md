# Progress History — April 2026

Archived from `docs/PROGRESS_CHECKLIST.md` on 2026-05-01 to keep the active
checklist within the 350-line budget set by `docs/AGENT_GOVERNANCE.md`.

This file holds completed sub-tasks and historical implementation notes for
Workstreams A (VLM Source Sanctity) and B (Code Block Fidelity) prior to the
Milestone 1 closure. Active TODOs and acceptance signals remain in
`docs/PROGRESS_CHECKLIST.md`.

---

## Workstream A — Completed Implementation Sub-tasks

- Strengthened generic visual-only prompt.
- Added text-reading validator patterns.
- Added corrective retry prompt.
- Added sanitizer for salvageable text-reading responses.
- Added prompt-aware vision cache.
- Added prompt harness: `scripts/eval_vlm_image_prompt.py`.
- Added detector/sanitizer tests: `tests/test_vlm_text_detection.py`.
- Verified compile check for modified VLM files and harness.
- Added blind non-magazine image set to the prompt harness.
  - Manifest tracked at `tests/fixtures/blind_set_manifest.json` (16 images,
    7 non-magazine docs). Asset paths point at `output/` (gitignored,
    runtime-regenerated). chunk_id includes asset stem (collision fix).
- Machine-readable VLM quality summary: `scripts/vlm_quality_summary.py`
  reports `clean` and `sanitized` separately (fix 2026-04-29).
- Structured VLM issue metadata in output JSONL via `vision_validation_issues`
  field in `ChunkMetadata` for `vision_status=done` chunks.

### Harness evidence (2026-04-29)

- PCWorld: `output/PCWorld_eval_qwen3vl_latest.jsonl` (126 images, qwen3-vl-plus).
  Text-reading 36.5% → 22.2%; hard fallback 37.3% → 21.4%; truly clean 61.1%; sanitized 17.5%.
- Combat Aircraft: `output/Combat_Aircraft_eval_qwen3vl_latest.jsonl`
  (18 images). 77.8% clean, 5.6% hard fallback, 3 timeout errors. Zero
  Combat-style hallucinations.
- Blind set: `output/blind_set_eval_qwen3vl.jsonl` (16 images). 75% truly
  clean, 12.5% sanitized, 12.5% hard fallback.

---

## Workstream B — Completed Architecture Sub-tasks

- Backed out broad CodeFormulaV2 trigger from `has_encoding_corruption` alone.
- Added cheap code-evidence decision pass using `CodeItem` count, code-chunk
  ratio, sampled code-candidate regions.
- Emit/use explicit `needs_code_enrichment=True` metadata with reason/counts.
- Preserved structural pathology flags when passing metadata from
  `BatchProcessor` to `V2DocumentProcessor`.
- Refactored duplicated PDF extraction policy into shared `PdfConversionPlan`
  + Docling adapter. `batch_processor.py`, `processor.py`, and
  `engines/pdf_engine.py` no longer construct Docling options independently.
  Evidence: `src/mmrag_v2/engines/pdf_plan.py`,
  `src/mmrag_v2/engines/docling_adapter.py`, `tests/test_pdf_conversion_plan.py`.
- Canonical PDF flow: diagnostics/config → `PdfConversionPlan` → Docling
  adapter → `UniversalDocument` → `ElementProcessor` → chunks.
- Bridge tests for every plan boundary: CLI process → plan, CLI batch → plan,
  batch → processor, processor → adapter, PDFEngine → adapter, adapter →
  Docling options. Focused plan/bridge/UIR run `73 passed`; full unit suite
  `412 passed, 1 skipped` at completion (2026-04-30).
- Static guard tests reject new `PdfPipelineOptions(` / `DocumentConverter(`
  construction outside the shared adapter:
  `tests/test_pdf_conversion_plan.py::test_no_pipeline_options_construction_outside_adapter`,
  `::test_no_production_docling_imports_outside_adapter`.
- Workstream B negative tests preserved as contracts: incidental shell
  commands, sparse fenced snippets, non-code magazines, and encoding
  corruption alone must not trigger CodeFormulaV2.

### Investigation notes (2026-04-29)

- Docling 2.86.0 exposes `do_code_enrichment`, `code_formula_options`, and
  `CodeFormulaVlmOptions`.
- Targeted Ayeva probe: CodeFormulaV2 restored multiline/indented code for
  21/22 items in a 10-page sample, but local CPU runtime is not viable
  (~18 min / 10 pages).
- Decision recorded in `docs/DECISIONS.md` — Selective Code Enrichment Lane.

### Completed next-step items

- Inspected failing code chunks in Ayeva and Chaubal.
- Compared extraction/chunking behavior against Fluent Python.
- Identified loss path (Docling extraction vs chunking vs detection vs
  refinement vs JSONL export).
- Ran targeted Docling-native `do_code_enrichment=True` probe on Ayeva.
- Confirmed Docling emits `CodeItem` for Ayeva and CodeFormulaV2 restores
  multiline/indented output in the targeted probe.
- Fixed trigger design: no broad `has_encoding_corruption` activation;
  added explicit `needs_code_enrichment` lane.
- Reviewed Workstream B test edits for AGENT-TEST-01 compliance.

---

## Milestone 1 — Test-Count Snapshots

- Initial focused-suite count: `80 passed`.
- Initial full-unit-suite count: `461 passed, 1 skipped`.
- Pre-RAG-Guide-fix smoke baseline: `output/smoke_multiprofile_20260430_225539/`,
  10/10 GATE_PASS + UNIVERSAL_PASS.

Closure counts (2026-05-01) live in
`docs/QUALITY_SNAPSHOT_2026-05-01.md`.
