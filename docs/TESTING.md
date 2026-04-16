# Testing Guide (v2.7.0)

**Version:** v2.7.0  
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

## Intelligence Stack Parity (Data-Agnostic)

Representative test set (all present in `data/`):
- `data/academic_journal/AIOS LLM Agent Operating System.pdf` — digital, high text density
- `data/scanned/HarryPotter_and_the_Sorcerers_Stone.pdf` — scanned long-form
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
