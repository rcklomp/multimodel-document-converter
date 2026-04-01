# Testing Guide (v2.6.0-dev)

**Version:** v2.6.0-dev  
**Validation Policy:** Required for every test command

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

## BUG-009: Metadata Propagation (Verified)

### Test 1: `process` (no batching)
```bash
rm -rf /tmp/test1-output
conda run -n mmrag-v2 \
  mmrag-v2 process \
  "data/academic_journal/AIOS LLM Agent Operating System.pdf" \
  --output-dir /tmp/test1-output \
  --vision-provider none
conda run -n mmrag-v2 \
  python tests/test_bug009_metadata_propagation.py /tmp/test1-output/ingestion.jsonl
```

### Test 2: `process --batch-size`
```bash
rm -rf /tmp/test2-output
conda run -n mmrag-v2 \
  mmrag-v2 process \
  "data/academic_journal/AIOS LLM Agent Operating System.pdf" \
  --output-dir /tmp/test2-output \
  --batch-size 3 \
  --vision-provider none
conda run -n mmrag-v2 \
  python tests/test_bug009_metadata_propagation.py /tmp/test2-output/ingestion.jsonl
```

### Test 3: `batch` (multi-file)
```bash
rm -rf /tmp/test3-input /tmp/test3-output
mkdir -p /tmp/test3-input
cp "data/academic_journal/AIOS LLM Agent Operating System.pdf" /tmp/test3-input/doc1.pdf
cp "data/academic_journal/A_comprehensive_review_on_hybrid_electri.pdf" /tmp/test3-input/doc2.pdf

conda run -n mmrag-v2 \
  mmrag-v2 batch /tmp/test3-input \
  --output-dir /tmp/test3-output \
  --vision-provider none

conda run -n mmrag-v2 \
  python tests/test_bug009_metadata_propagation.py /tmp/test3-output/doc1/ingestion.jsonl
conda run -n mmrag-v2 \
  python tests/test_bug009_metadata_propagation.py /tmp/test3-output/doc2/ingestion.jsonl
```

### Test 4: Layout-aware OCR (scanned document)
```bash
rm -rf /tmp/test4-output
conda run -n mmrag-v2 \
  mmrag-v2 process \
  "data/scanned_literature/HarryPotter_and_the_Sorcerers_Stone.pdf" \
  --output-dir /tmp/test4-output \
  --batch-size 3 \
  --ocr-mode layout-aware \
  --enable-ocr \
  --ocr-confidence-threshold 0.7 \
  --vision-provider none

conda run -n mmrag-v2 \
  python tests/test_bug009_metadata_propagation.py /tmp/test4-output/ingestion.jsonl
```

## Intelligence Stack Parity (Data-Agnostic)

Representative test set (all present in `data/`):
- `data/academic_journal/AIOS LLM Agent Operating System.pdf` — digital, high text density
- `data/scanned_literature/HarryPotter_and_the_Sorcerers_Stone.pdf` — scanned long-form
- `data/technical_manual/Firearms.pdf` — scanned technical manual
- `data/digital_magazine/PCWorld_July_2025_USA.pdf` — high image density, digital

### Multi-Profile Smoke Test (primary acceptance gate)
```bash
conda run -n mmrag-v2 bash scripts/smoke_multiprofile.sh
```
Expected: `GATE_PASS` + `UNIVERSAL_PASS` in every row of the summary table.

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
