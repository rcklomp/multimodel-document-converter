# Testing Guide (v2.11.0)

**Version:** v2.11.0 (Phase 1 swap staged locally on `c2a461c`; tag pending user push)
**Validation Policy:** Required for every test command
**DashScope API key:** read from the `DASHSCOPE_API_KEY` env var by `scripts/ingest_to_qdrant.py`, `scripts/search_qdrant.py`, `scripts/retrieval_regression.py`, and `scripts/synthetic_soak.py`. As of v2.11.0 the production embedder is Dashscope `text-embedding-v4` — the key is required for any ingestion, search, or retrieval-regression run. The reranker call in `search_qdrant.py` degrades to vector-rank truncation when the key is unset; the embedding call hard-fails (returns 2). Test-suite skip-gates handle the unset case for CI (the production retrieval-regression test skips cleanly).

## Environment
- Runner: `conda run -n mmrag-v2`
- Default VLM: `none` (set explicit provider when needed)

## Validation (Required)

For every test command in this file, validation is mandatory:

1. Capture terminal output for the command and check for explicit success markers.
2. A command is considered **Pass** only if at least one of these is true:
   - Output contains `[OK]`, or
   - Process returns `exit code 0`, or
   - Test summary explicitly reports zero failures (for example, `X passed, 0 failed`).
3. A command is considered **Fail** if output includes any of: `stderr`, `FAIL`, `ERROR`, traceback, or non-zero exit code.
4. On fail, stop and report the failing command with the relevant terminal excerpt before attempting any additional fixes.

## Intelligence Stack Parity (Data-Agnostic)

Representative test set (all present in `data/`):
- `data/academic_journal/AIOS LLM Agent Operating System.pdf` — digital, high text density
- `data/digital_literature/HarryPotter_and_the_Sorcerers_Stone.pdf` — born-digital novel; routes to `digital_literature` profile and exercises the post-Docling sanity pass (y-sort, drop-cap heal, label-leak filter, OCR gating). See `docs/archive/PLAN_DOCLING_POSTPROCESSOR.md`.
- `data/technical_manual/Firearms.pdf` — scanned technical manual
- `data/digital_magazine/PCWorld_July_2025_USA.pdf` — high image density, digital

### Multi-Profile Smoke Test (primary acceptance gate)
```bash
conda run -n mmrag-v2 bash scripts/smoke_multiprofile.sh
```
Expected: `GATE_PASS` + `UNIVERSAL_PASS` in every row of the summary table.
The smoke runner defaults to cache-only model resolution and
`TORCH_COMPILE_DISABLE=1` for deterministic Apple-Silicon runs; override
those environment variables only when intentionally testing online model
resolution or `torch.compile`.

### Single-Conversion Full QA
```bash
conda run -n mmrag-v2 python scripts/qa_full_conversion.py \
  output/<document>/ingestion.jsonl \
  --source-pdf "data/<category>/<source>.pdf" \
  --allow-warnings
```

Expected: `QA_PASS` or `QA_PASS_WITH_ADVISORIES`. `QA_FAIL` means the
conversion is not production-clean. The wrapper runs the existing QA scripts
and adds deterministic checks for missing pages/chapters, duplicate long text,
per-page outliers, localized corruption, image-description quality, asset
health, and table corruption. Documented advisory codes are governed by
`docs/QUALITY_GATES.md`.

### TOC / Index Page-Loss Contract (env-gated)
Phase 1 v2.9 uses generated page-window probes to lock the TOC/index page-loss
contract without making the default test suite run Docling conversions.

First generate or preserve these outputs:
- `output/probe_kimothi_toc_contract_codex/ingestion.jsonl`
- `output/probe_kimothi_toc_contract_codex_rerun/ingestion.jsonl`
- `output/probe_hao_toc_contract_codex/ingestion.jsonl`
- `output/probe_python_cookbook_toc_contract_codex/ingestion.jsonl`

Then run:
```bash
RUN_TOC_PAGE_CONTRACT=1 conda run -n mmrag-v2 python -m pytest \
  tests/test_toc_index_page_contract.py -q
```

Expected: `4 passed`. If the probe outputs were cleaned from `output/`, the
test fails with an explicit missing-output assertion. Without
`RUN_TOC_PAGE_CONTRACT=1`, the same file is skipped by default.

### Process vs Batch (Parity)
```bash
PDF="data/academic_journal/AIOS LLM Agent Operating System.pdf"
conda run -n mmrag-v2 mmrag-v2 process "$PDF" --output-dir output/test_process --vision-provider none
conda run -n mmrag-v2 mmrag-v2 batch data/academic_journal --pattern "AIOS*.pdf" --output-dir output/test_batch --vision-provider none
```

Expected:
- Diagnostics banner + Profile banner + Extraction Strategy banner appear in both paths.
- Same profile type and strategy parameters.
- No cross-modality fallback (scanned -> digital or vice versa).

### Filename Independence (Parity Regression)
```bash
PDF="data/academic_journal/AIOS LLM Agent Operating System.pdf"
rm -rf /tmp/parity_name_test /tmp/parity_name_out
mkdir -p /tmp/parity_name_test
cp "$PDF" /tmp/parity_name_test/doc1.pdf

conda run -n mmrag-v2 mmrag-v2 process "$PDF" --output-dir /tmp/parity_name_out/process --vision-provider none
conda run -n mmrag-v2 mmrag-v2 batch /tmp/parity_name_test --pattern "*.pdf" --output-dir /tmp/parity_name_out/batch --vision-provider none

jq -r '.metadata.profile_type' /tmp/parity_name_out/process/ingestion.jsonl | sort | uniq -c
jq -r '.metadata.profile_type' /tmp/parity_name_out/batch/doc1/ingestion.jsonl | sort | uniq -c
```

Expected:
- Identical `profile_type` counts regardless of filename.

## Manual JSONL Spot-Checks
```bash
jq -r '.metadata.profile_type' /tmp/test-output/ingestion.jsonl | sort | uniq -c
jq -r 'select(.metadata.profile_type == null or \
              .metadata.min_image_dims == null or \
              .metadata.confidence_threshold == null or \
              .metadata.document_domain == null or \
              .metadata.document_modality == null or \
              .metadata.profile_sensitivity == null) | \
       .chunk_id' /tmp/test-output/ingestion.jsonl
```
