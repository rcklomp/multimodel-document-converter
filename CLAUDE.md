# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Read First
1. `docs/PROJECT_STATUS.md`
2. `AGENTS.md`
3. `docs/README.md`
4. `docs/AGENT_GOVERNANCE.md`
5. `docs/DECISIONS.md`
6. `docs/TESTING.md`
7. `docs/QUALITY_GATES.md`
8. `docs/ARCHITECTURE.md`
9. `docs/PLAN_V2.9.md` (active plan)
10. `docs/archive/PROGRESS_CHECKLIST.md` (historical execution log; archived 2026-05-07 — current task state lives in `PROJECT_STATUS.md`)
11. `docs/archive/SRS_Multimodal_Ingestion_V2.5.md`

Use the three-layer docs model:
- Layer 0 contracts: invariants, governance, decisions, architecture, quality gates.
- Layer 1 current state: project status and quality snapshots.
- Layer 2 execution: tests, active plan, and archive (per-task history lives in `docs/archive/PROGRESS_CHECKLIST.md`).

## Engineering Principles

- **Think before coding.** State assumptions explicitly. If multiple interpretations exist, present them — don't pick silently. If something is unclear, stop and ask.
- **Simplicity first.** Minimum code that solves the problem. No speculative features, abstractions for single-use code, or error handling for impossible scenarios. If 200 lines could be 50, rewrite it.
- **Surgical changes.** Touch only what you must. Don't "improve" adjacent code, comments, or formatting. Match existing style. Remove only imports/variables/functions that YOUR changes made unused. Every changed line should trace directly to the request.
- **Libraries first, custom code last.** Before writing filters, heuristics, or workarounds, check whether the library (Docling, ebooklib, etc.) already has a configuration option that solves the problem. The v2.4 script is a valid reference for what Docling can do natively.
- **Keep configurations in sync.** Shared `PdfConversionPlan` + `DoclingPdfAdapter` is the single source of Docling option/converter construction (`src/mmrag_v2/engines/pdf_plan.py`, `engines/docling_adapter.py`; complete 2026-04-30). Static guard tests (`tests/test_pdf_conversion_plan.py::test_no_pipeline_options_construction_outside_adapter`, `::test_no_production_docling_imports_outside_adapter`) reject any new `PdfPipelineOptions(` / `DocumentConverter(` construction outside the adapter — treat any duplication as a regression. Every PDF extraction change must flow through the plan and include bridge tests proving the decision crosses object boundaries.
- **Verify before converting.** Run the test suite and a single-document smoke test before starting batch conversions. Confirm schema version, chunk counts, and gate results on a real output before burning VLM credits.
- **Goal-driven execution.** Transform tasks into verifiable goals with success criteria. For multi-step tasks, state a brief plan with verification checks at each step.

## Workstream B Code Enrichment Guardrail

- Use Docling-native `CodeItem` / CodeFormulaV2 before custom code-repair heuristics.
- Do not enable `do_code_enrichment` from `has_encoding_corruption` alone; encoding corruption is not code evidence and includes magazine/text-corruption workstreams.
- Add/use an explicit `needs_code_enrichment` decision based on cheap code evidence: `CodeItem` count, code chunk ratio, or sampled code-candidate regions.
- Prefer CodeFormulaV2 inference on stronger local-network hardware. Cloud is acceptable when data policy and cost allow. **Custom client-local MLX/transformers** setups are diagnostic/fallback only. **Docling-native CodeFormulaV2 on CPU** (the model bundled with `docling==2.86.0`) is acceptable for one-off batch reconversion (~27 sec/page on Apple Silicon target, forced to CPU by Docling because MPS is unsupported by this model). See `docs/DECISIONS.md` "Selective Code Enrichment Lane → Amendment 2026-05-03".
- If Docling only supports document-level code enrichment, enable it only after the code-evidence pass. If region-level remote inference exists, send only `CodeItem`/code-candidate crops.
- Keep fallback regex/Tesseract repairs clearly marked and do not let them mask whether Docling-native/remote enrichment worked.
- Workstream B negative tests are contracts: incidental shell commands, sparse fenced snippets, non-code magazines, and encoding corruption alone must not trigger CodeFormulaV2. Do not loosen these assertions or rewrite fixtures to match a broad heuristic. If one fails, fix the heuristic or stop.
- v2.7 §5 (shared PDF extraction plan + adapter refactor) and `docs/archive/PLAN_DOCLING_POSTPROCESSOR.md` (post-Docling sanity pass — y-sort, drop-cap heal, label-leak filter, OCR gating; new `digital_literature` profile) are both **shipped** as of 2026-05-03. If a new design plan is needed for next-phase work, draft it as `docs/PLAN_V2.8_*.md` rather than adding parallel sections to either existing plan.
- Canonical target flow is diagnostics/config -> `PdfConversionPlan` -> Docling adapter -> `UniversalDocument` -> `ElementProcessor` -> chunks. Do not expand direct Docling-item-to-chunk mapping.

## Test Contract Integrity

- Negative tests, regression tests, and acceptance fixtures are executable requirements.
- Do not remove, weaken, or reframe assertions to make the current implementation pass.
- Do not rewrite fixtures to avoid the behavior under test.
- If a test expectation appears wrong, stop and document the proposed requirement change before editing the test.

## Project Invariants
- Python is locked to 3.10 (`pyproject.toml`: `>=3.10,<3.11`).
- Runtime target is Apple Silicon; prefer Torch MPS when available.
- `docling` is exact-pinned to `2.86.0` (upgraded from 2.66.0 — enables picture/code enrichment features used by current plans).
- Keep PDF batch size at `<=10` pages.
- Use the `ProfileClassifier` in `orchestration/profile_classifier.py` for automatic routing; do not replace it with the V2.4.2 `DocumentClassifier` approach. Profile overrides (`--profile-override`) are for debugging only, never for production acceptance runs.
- Spatial metadata `bbox` must be emitted as integer `[0,1000]` coordinates.
- AGENT-SPATIAL-20: keep the single 20-unit vertical threshold behavior (no profile-specific branching for that rule).
- Acceptance is not complete unless `GATE_PASS` + `UNIVERSAL_PASS` are reported across all document categories in the multi-profile smoke test, and at least one per-category blind-test document is included.
- QA-CHECK-01 tolerance target is `0.10` for all profiles (no waivers).

## Setup
```bash
conda env create -f environment.yml
conda activate mmrag-v2
pip install -e .
pip install -e ".[dev]"
```

## Core Commands
```bash
mmrag-v2 version
mmrag-v2 check
mmrag-v2 process data/<category>/<file>.pdf --output-dir output/<run_name> --vision-provider none
mmrag-v2 process data/<category>/<file>.pdf --batch-size 10 --output-dir output/<run_name>
mmrag-v2 process data/<category>/<file>.pdf --profile-override <profile> --output-dir output/<run_name>
mmrag-v2 batch data/<category> --pattern "*.pdf" --output-dir output/<run_name> --vision-provider none
```

## Tests and Lint
```bash
pytest tests/ -v
pytest tests/test_token_validator.py -v
pytest tests/test_token_validator.py::test_simple_text_exact_match -v -s
ruff check src tests
black --check src tests
mypy src/mmrag_v2
```

## Acceptance Gate
```bash
# Multi-profile smoke test (cross-category baseline — run first):
bash scripts/smoke_multiprofile.sh
# Look for GATE_PASS + UNIVERSAL_PASS in every row of the summary table.

# Technical-manual deep acceptance (4 docs × 20 pages):
bash scripts/acceptance_technical_manual.sh
python scripts/evaluate_technical_manual_gates.py output/<run_name>/ingestion.jsonl --doc-class auto

# Canonical full strict-gate on any single output (use --source-pdf when available
# so blank source pages do not count as MISSING_PAGES failures):
python scripts/qa_full_conversion.py output/<run_name>/ingestion.jsonl \
  --source-pdf data/<category>/<file>.pdf

# Lighter universal invariant check (no blank-page awareness — advisory only):
python scripts/qa_universal_invariants.py output/<run_name>/ingestion.jsonl
```
Look for explicit `GATE_PASS` / `GATE_FAIL` and `UNIVERSAL_PASS` / `UNIVERSAL_FAIL` in output, and `QA_PASS` / `QA_WARN` / `QA_FAIL` from `qa_full_conversion.py`. The strict-gate command is `qa_full_conversion.py --source-pdf` (per Phase 4 Step 1, 2026-05-09); the no-flag form reports phantom MISSING_PAGES failures on docs with blank-source pages.

## Runtime Architecture
- CLI entry: `src/mmrag_v2/cli.py` (`process`, `batch`, `version`, `check`).
- `process` for PDFs runs: `DocumentDiagnosticEngine` -> `SmartConfigProvider` -> `ProfileManager`/`ProfileClassifier` -> `StrategyOrchestrator`.
- PDF + `--batch-size > 0` uses `BatchProcessor.process_pdf(...)`.
- Non-batch or non-PDF uses `V2DocumentProcessor.process_to_jsonl_atomic(...)`.
- `batch` command loops files; for PDF files it runs the same intelligence stack for parity, then uses `BatchProcessor`.
- `src/mmrag_v2/batch_processor.py` is the primary PDF orchestrator (splitting, OCR governance, filtering, token validation/recovery, dedupe, JSONL export).
- `src/mmrag_v2/processor.py` currently maps Docling elements to text/image/table chunks and runs shadow extraction; this is legacy behavior to shrink behind the UIR path during the PDF extraction refactor.
- Profile intelligence modules: `orchestration/document_diagnostic.py`, `orchestration/profile_classifier.py`, `orchestration/strategy_profiles.py`, `orchestration/strategy_orchestrator.py`.
- Canonical chunk schema: `src/mmrag_v2/schema/ingestion_schema.py`.
- Schema/version stamping uses `src/mmrag_v2/version.py`.
- QA-CHECK-01 token balance logic: `src/mmrag_v2/validators/token_validator.py`.
- Filtering analytics: `src/mmrag_v2/validators/quality_filter_tracker.py`.
- UIR abstractions live under `src/mmrag_v2/universal/` and engines under `src/mmrag_v2/engines/`.
