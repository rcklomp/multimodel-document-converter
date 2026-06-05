# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Read First (live working set — top-level `docs/` only)

1. `docs/PROJECT_STATUS.md` — current task state.
2. `AGENTS.md` — agent-protocol contract.
3. `docs/ARCHITECTURE_V3_DRAFT_0.5.md` — **V3.0 target architecture (canonical)**.
3a. `docs/ARCHITECTURE_V3.1_CHARTER.md` — **V3.1 as-built + roadmap** (current reality, status-tagged SHIPPED/PARTIAL/PROPOSED; read alongside the 0.5 target).
4. `docs/README.md` — docs index + three-layer model overview.
5. `docs/V3_EXECUTION_MANDATE.md` — conflict-resolution authority for V3 work (governance set is the Layer-0 list in `docs/README.md`; the mandate wins where it conflicts).
6. `docs/DECISIONS.md` — locked decisions log.
7. `docs/TESTING.md` — test conventions.
8. `docs/QUALITY_GATES.md` — gate definitions.
9. `docs/ARCHITECTURE.md` — v2.X pipeline architecture (production baseline being evolved).
10. **Committed-Truth convention** — repo-integrity guards in `tests/test_repo_integrity.py` (its docstring documents G1–G6 + author conventions); the contract is AGENTS.md `AGENT-INTEGRITY-01` (assert outcomes, not proxies).

All v2.14–v2.16 history, telemetry, calibration reports, and legacy
quality snapshots are quarantined in `docs/.archive/` and blocked by
`.aiignore`. Agents MUST NOT read or reference them.

Use the three-layer docs model:
- Layer 0 contracts: invariants, governance, decisions, architecture, quality gates.
- Layer 1 current state: PROJECT_STATUS.md, ARCHITECTURE_V3_DRAFT_0.5.md.
- Layer 2 execution: active plan docs; legacy v2.X history quarantined in `docs/.archive/`.

## Engineering Principles

- **Think before coding.** State assumptions explicitly. If multiple interpretations exist, present them — don't pick silently. If something is unclear, stop and ask.
- **Simplicity first.** Minimum code that solves the problem. No speculative features, abstractions for single-use code, or error handling for impossible scenarios. If 200 lines could be 50, rewrite it.
- **Surgical changes.** Touch only what you must. Don't "improve" adjacent code, comments, or formatting. Match existing style. Remove only imports/variables/functions that YOUR changes made unused. Every changed line should trace directly to the request.
- **Libraries first, custom code last.** Before writing filters, heuristics, or workarounds, check whether the library (Docling, ebooklib, etc.) already has a configuration option that solves the problem. The v2.4 script is a valid reference for what Docling can do natively.
- **Keep configurations in sync.** *Legacy v2 Docling path:* shared `PdfConversionPlan` + `DoclingPdfAdapter` is the single source of Docling option/converter construction (`src/mmrag_v2/engines/pdf_plan.py`, `engines/docling_adapter.py`). Its static guard tests (`tests/test_pdf_conversion_plan.py`) are currently **deferred** (`@pytest.mark.skip`, `V3_DEFERRED`; see `docs/PROJECT_STATUS.md`) because the V3 path no longer constructs Docling there. *V3 path (current default for `BatchProcessor.process_pdf`):* `batch_processor.py` is engine-agnostic and constructs **no** Docling — extraction is delegated to `mmrag_v3.extract()`, where Docling is confined to `src/mmrag_v3/engines/docling_fast.py` (the sole V3 Docling boundary, AST-guarded by `tests/test_v3_security.py`). Do not add Docling construction to `batch_processor.py`.
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
- Canonical flow — *legacy v2 path:* diagnostics/config -> `PdfConversionPlan` -> Docling adapter -> `UniversalDocument` -> `ElementProcessor` -> chunks. *V3 path (current default for `BatchProcessor.process_pdf`):* `mmrag_v3.extract()` (HybridEngine) -> `UniversalDocument` -> `chunk_universal_document()` -> `IngestionChunk.from_uir()`. Do not expand direct Docling-item-to-chunk mapping in either path.

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
# Production-CLI anti-rot smoke (MANDATORY pre-merge gate for any change to the
# extraction path: batch_processor.py, chunking/uir_chunker.py, src/mmrag_v3/**,
# IngestionChunk.from_uir). Offline/CI by default (USE_DOCLING_FAST, no VLM):
bash scripts/smoke_production.sh
# Full M5-VLM check (opt-in): SMOKE_FULL=1 + VLM_NATIVE_* env. Look for the
# per-lane table and the final SMOKE_PRODUCTION_PASS line (exit 0).

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

`scripts/smoke_production.sh` (PLAN_V3.1 Phase 5) is the mandatory pre-merge gate for any change touching the V3 extraction path; it must print `SMOKE_PRODUCTION_PASS` (exit 0) in offline mode before merge. It runs one doc per routing lane through the shipping CLI and asserts batch integrity, IMAGE/TABLE asset_ref + on-disk asset (QA-CHECK-05), V3-path routing (`extraction_method=uir_native_chunker` offline), and `QA_PASS`/`QA_PASS_WITH_ADVISORIES`.

## Runtime Architecture
- CLI entry: `src/mmrag_v2/cli.py` (`process`, `batch`, `version`, `check`).
- `process` for PDFs runs: `DocumentDiagnosticEngine` -> `SmartConfigProvider` -> `ProfileManager`/`ProfileClassifier` -> `StrategyOrchestrator`.
- PDF + `--batch-size > 0` uses `BatchProcessor.process_pdf(...)`.
- Non-batch or non-PDF uses `V2DocumentProcessor.process_to_jsonl_atomic(...)`.
- `batch` command loops files; for PDF files it runs the same intelligence stack for parity, then uses `BatchProcessor`.
- `src/mmrag_v2/batch_processor.py` is the primary PDF orchestrator (splitting, filtering, token validation/recovery, dedupe, JSONL export). As of Phase A Step 5 (`813b9ba`) it **delegates extraction to `mmrag_v3.extract()`** (→ `UniversalDocument` → `chunk_universal_document` → `from_uir`); the legacy OCR/layout lanes were deleted and it has **zero docling imports**.
- `src/mmrag_v2/processor.py` maps Docling elements to text/image/table chunks and runs shadow extraction — **legacy; bypassed by the V3 `batch_processor` path**, still used by the non-batch `V2DocumentProcessor` lane (line above). Its chunker INPUT boundary still consumes `DoclingDocument` (documented Phase A residual).
- Profile intelligence modules: `orchestration/document_diagnostic.py`, `orchestration/profile_classifier.py`, `orchestration/strategy_profiles.py`, `orchestration/strategy_orchestrator.py`.
- Canonical chunk schema: `src/mmrag_v2/schema/ingestion_schema.py`.
- Schema/version stamping uses `src/mmrag_v2/version.py`.
- QA-CHECK-01 token balance logic: `src/mmrag_v2/validators/token_validator.py`.
- Filtering analytics: `src/mmrag_v2/validators/quality_filter_tracker.py`.
- UIR abstractions live under `src/mmrag_v2/universal/` and engines under `src/mmrag_v2/engines/`.
- **V3 Phase C — vision-native extraction (2026-05-29):** new namespace at `src/mmrag_v3/`. Entry `src/mmrag_v3/processor.py`. **Default route (2026-06-06): `MineruQwenHybridEngine`** when `MINERU_ENDPOINT` is set — code-dense pages (monospace ratio >= 0.10) → Qwen VLM, every other page → MinerU2.5 (neither engine alone passes a code-heavy doc: MinerU mangles dense code to R3 0.44, Qwen empties dense tables to 50%; the hybrid gets both, AIOS `QA_PASS` — see `docs/DECISIONS.md` "MinerU+Qwen-for-code hybrid"). Pure MinerU via `USE_MINERU_ENGINE=1`; the legacy `HybridEngine` (Docling+VLM) is the no-`MINERU_ENDPOINT` fallback. Engines: `engines/vlm_native.py`, `engines/vlm_provider.py`, `engines/docling_fast.py` (sole `docling` import boundary in V3), `engines/router.py`. AST firewall at `tests/test_v3_security.py` (13 tests). Default VLM provider is OpenRouter `qwen/qwen3-vl-8b-instruct`; override via `VLM_NATIVE_ENDPOINT` / `VLM_NATIVE_MODEL` / `VLM_NATIVE_API_KEY`. Engine route forced via `USE_VLM_ENGINE=1` or `USE_DOCLING_FAST=1`. Rebaseline utility at `scripts/rebaseline_v3.py`. The V3 Phase A sandbox (`v3_execution_root/`) was **removed 2026-05-30** — it was a duplicate `mmrag_v3` namespace and NOT a production dependency (`src/mmrag_v3/processor.py` never imported it). Durable artifacts were salvaged: `docs/V3_DEFERRED_TESTS.md` (active contract), `docs/paper/archive_extracts/v3_mandate/` + `…/sanitization_prompts/` (reference). Full backup tarball: `~/mmrag_v3_execution_root_backup_2026-05-30.tar.gz`. NOTE: the V3 baseline/soak scripts (`scripts/v3_batch_ingest.py`, `rebaseline_v3.py`) imported the sandbox chunker and need repointing to `src/mmrag_v2/chunking/uir_chunker.py` before they run again. See `docs/DECISIONS.md` "v3.0 Phase C — Vision-Native Extraction" and `docs/PROJECT_STATUS.md` "Phase C" section.
