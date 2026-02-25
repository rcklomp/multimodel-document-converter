# Testing Guide (v2.4.1-stable)

**Version:** v2.4.1-stable  
**Validation Policy:** Required for every test command

## Environment
- Runner: `conda run -p ./env`
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
conda run -p ./env \
  mmrag-v2 process \
  "data/raw/Hybrid_electric_vehicles_and_their_challenges.pdf" \
  --output-dir /tmp/test1-output \
  --vision-provider none
conda run -p ./env \
  python tests/test_bug009_metadata_propagation.py /tmp/test1-output/ingestion.jsonl
```

### Test 2: `process --batch-size`
```bash
rm -rf /tmp/test2-output
conda run -p ./env \
  mmrag-v2 process \
  "data/raw/Hybrid_electric_vehicles_and_their_challenges.pdf" \
  --output-dir /tmp/test2-output \
  --batch-size 3 \
  --vision-provider none
conda run -p ./env \
  python tests/test_bug009_metadata_propagation.py /tmp/test2-output/ingestion.jsonl
```

### Test 3: `batch` (multi-file)
```bash
rm -rf /tmp/test3-input /tmp/test3-output
mkdir -p /tmp/test3-input
cp "data/raw/Hybrid_electric_vehicles_and_their_challenges.pdf" /tmp/test3-input/doc1.pdf
cp "data/raw/IRJET_Modeling_of_Solar_PV_system_under.pdf" /tmp/test3-input/doc2.pdf

conda run -p ./env \
  mmrag-v2 batch /tmp/test3-input \
  --output-dir /tmp/test3-output \
  --vision-provider none

conda run -p ./env \
  python tests/test_bug009_metadata_propagation.py /tmp/test3-output/doc1/ingestion.jsonl
conda run -p ./env \
  python tests/test_bug009_metadata_propagation.py /tmp/test3-output/doc2/ingestion.jsonl
```

### Test 4: Layout-aware OCR (full run)
```bash
rm -rf /tmp/test4-output
conda run -p ./env \
  mmrag-v2 process \
  "data/raw/HarryPotter_and_the_Sorcerers_Stone.pdf" \
  --output-dir /tmp/test4-output \
  --batch-size 3 \
  --ocr-mode layout-aware \
  --enable-ocr \
  --ocr-confidence-threshold 0.7 \
  --vision-provider none

conda run -p ./env \
  python tests/test_bug009_metadata_propagation.py /tmp/test4-output/ingestion.jsonl
```

## Intelligence Stack Parity (Data-Agnostic)

Prepare a small, representative set:
- Academic/technical whitepaper (digital, high text density, small diagrams)
- Scanned literature (long-form, low OCR text, small illustrations)
- Technical manual (technical domain, scanned, low text density)
- Magazine/editorial (high image density; digital + scanned if possible)
- At least one non-PDF file for guard behavior

### Process vs Batch (Parity)
```bash
conda run -p ./env mmrag-v2 process <FILE> --output-dir output/test_process
conda run -p ./env mmrag-v2 batch <DIR> --pattern "<GLOB>" --output-dir output/test_batch
```

Expected:
- Diagnostics banner + Profile banner + Extraction Strategy banner appear in both paths.
- Same profile type and strategy parameters.
- No cross-modality fallback (scanned -> digital or vice versa).

### Filename Independence (Parity Regression)
```bash
rm -rf /tmp/parity_name_test /tmp/parity_name_out
mkdir -p /tmp/parity_name_test
cp "<SOURCE_PDF>" /tmp/parity_name_test/doc1.pdf

conda run -p ./env mmrag-v2 process "<SOURCE_PDF>" --output-dir /tmp/parity_name_out/process
conda run -p ./env mmrag-v2 batch /tmp/parity_name_test --pattern "*.pdf" --output-dir /tmp/parity_name_out/batch

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
