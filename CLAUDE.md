# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Read First
1. `docs/PROJECT_STATUS.md`
2. `docs/PROGRESS_CHECKLIST.md`
3. `AGENTS.md`
4. `docs/README.md`
5. `docs/DECISIONS.md`
6. `docs/TESTING.md`
7. `docs/QUALITY_GATES.md`
8. `docs/ARCHITECTURE.md`
9. `docs/SRS_Multimodal_Ingestion_V2.5.md`

Use the three-layer docs model:
- Layer 0 contracts: invariants, decisions, architecture, quality gates.
- Layer 1 current state: project status and quality snapshots.
- Layer 2 execution: progress checklist, tests, plans, and archive.

## Engineering Principles

- **Think before coding.** State assumptions explicitly. If multiple interpretations exist, present them — don't pick silently. If something is unclear, stop and ask.
- **Simplicity first.** Minimum code that solves the problem. No speculative features, abstractions for single-use code, or error handling for impossible scenarios. If 200 lines could be 50, rewrite it.
- **Surgical changes.** Touch only what you must. Don't "improve" adjacent code, comments, or formatting. Match existing style. Remove only imports/variables/functions that YOUR changes made unused. Every changed line should trace directly to the request.
- **Libraries first, custom code last.** Before writing filters, heuristics, or workarounds, check whether the library (Docling, ebooklib, etc.) already has a configuration option that solves the problem. The v2.4 script is a valid reference for what Docling can do natively.
- **Keep configurations in sync.** `batch_processor.py` and `processor.py` each create their own `PdfPipelineOptions` independently. When changing Docling settings in one, check the other. There is no shared factory (known debt).
- **Verify before converting.** Run the test suite and a single-document smoke test before starting batch conversions. Confirm schema version, chunk counts, and gate results on a real output before burning VLM credits.
- **Goal-driven execution.** Transform tasks into verifiable goals with success criteria. For multi-step tasks, state a brief plan with verification checks at each step.

## Project Invariants
- Python is locked to 3.10 (`pyproject.toml`: `>=3.10,<3.11`).
- Runtime target is Apple Silicon; prefer Torch MPS when available.
- `docling` minimum is `2.86.0` (upgraded from 2.66.0 — enables font metadata for heading classification).
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

# Universal invariant check on any single output:
python scripts/qa_universal_invariants.py output/<run_name>/ingestion.jsonl
```
Look for explicit `GATE_PASS` / `GATE_FAIL` and `UNIVERSAL_PASS` / `UNIVERSAL_FAIL` in output.

## Runtime Architecture
- CLI entry: `src/mmrag_v2/cli.py` (`process`, `batch`, `version`, `check`).
- `process` for PDFs runs: `DocumentDiagnosticEngine` -> `SmartConfigProvider` -> `ProfileManager`/`ProfileClassifier` -> `StrategyOrchestrator`.
- PDF + `--batch-size > 0` uses `BatchProcessor.process_pdf(...)`.
- Non-batch or non-PDF uses `V2DocumentProcessor.process_to_jsonl_atomic(...)`.
- `batch` command loops files; for PDF files it runs the same intelligence stack for parity, then uses `BatchProcessor`.
- `src/mmrag_v2/batch_processor.py` is the primary PDF orchestrator (splitting, OCR governance, filtering, token validation/recovery, dedupe, JSONL export).
- `src/mmrag_v2/processor.py` maps Docling elements to text/image/table chunks and runs shadow extraction.
- Profile intelligence modules: `orchestration/document_diagnostic.py`, `orchestration/profile_classifier.py`, `orchestration/strategy_profiles.py`, `orchestration/strategy_orchestrator.py`.
- Canonical chunk schema: `src/mmrag_v2/schema/ingestion_schema.py`.
- Schema/version stamping uses `src/mmrag_v2/version.py`.
- QA-CHECK-01 token balance logic: `src/mmrag_v2/validators/token_validator.py`.
- Filtering analytics: `src/mmrag_v2/validators/quality_filter_tracker.py`.
- UIR abstractions live under `src/mmrag_v2/universal/` and engines under `src/mmrag_v2/engines/`.
