# Bug Report: Process vs Batch Profile Mismatch (Academic/Editorial PDFs)

## Purpose
This report is designed to seed a Claude fix session. It provides context, history, evidence, and guardrails to fix the parity bug without regressions.

---

## Summary
**Bug:** `process` and `batch` classify the same academic/editorial PDFs into different `profile_type` values.  
**Impact:** Intelligence Stack parity is broken; batch uses a different extraction strategy than process for the same input, which can change min dimensions, background extraction, and downstream asset recall.

---

## Affected Area
- `mmrag-v2 process` (single file)
- `mmrag-v2 batch` (per-file path)
- Intelligence Stack: diagnostics → smart profile → classifier → strategy

---

## Observed Behavior (Evidence)
**Run ID:** `20260116_050543`  
**Output root:** `/tmp/mmrag_full_tests/20260116_050543`

### Case 1: Academic PDF (Hybrid electric vehicles)
- `process` profile:
  ```bash
  jq -r '.metadata.profile_type' /tmp/mmrag_full_tests/20260116_050543/process_baseline/ingestion.jsonl | sort | uniq -c
  # 456 academic_whitepaper
  ```
- `batch` profile:
  ```bash
  jq -r '.metadata.profile_type' /tmp/mmrag_full_tests/20260116_050543/batch_out/doc1/ingestion.jsonl | sort | uniq -c
  # 455 digital_magazine
  ```

### Case 2: Editorial PDF (IRJET)
- `process` profile:
  ```bash
  jq -r '.metadata.profile_type' /tmp/mmrag_full_tests/20260116_050543/sensitivity_low/ingestion.jsonl | sort | uniq -c
  # 166 academic_whitepaper
  ```
- `batch` profile:
  ```bash
  jq -r '.metadata.profile_type' /tmp/mmrag_full_tests/20260116_050543/batch_out/doc2/ingestion.jsonl | sort | uniq -c
  # 164 digital_magazine
  ```

**Expected:** Process and batch must yield the **same profile_type** for the same input.

---

## Reproduction (Minimal)
```bash
# process
PYTHONPATH=src conda run -p /Users/ronald/Projects/MM-Converter-V2/env \
  python -m mmrag_v2.cli process \
  "data/raw/Hybrid_electric_vehicles_and_their_challenges.pdf" \
  --output-dir /tmp/repro_process \
  --vision-provider none

# batch
mkdir -p /tmp/repro_batch_input
cp "data/raw/Hybrid_electric_vehicles_and_their_challenges.pdf" /tmp/repro_batch_input/doc1.pdf

PYTHONPATH=src conda run -p /Users/ronald/Projects/MM-Converter-V2/env \
  python -m mmrag_v2.cli batch /tmp/repro_batch_input \
  --output-dir /tmp/repro_batch \
  --vision-provider none

# compare
jq -r '.metadata.profile_type' /tmp/repro_process/ingestion.jsonl | sort | uniq -c
jq -r '.metadata.profile_type' /tmp/repro_batch/doc1/ingestion.jsonl | sort | uniq -c
```

---

## Context / History
- The project has an “Intelligence Stack” to keep `process` and `batch` parity.
- Batch previously bypassed the stack; later fixed to run the same pipeline.
- Observability metadata plumbing was fixed (BUG-009) and verified.
- This parity mismatch persists only for some document types (academic/editorial).

---

## Suspected Cause (Hypotheses)
These are **starting points** for investigation, not fixes:
1. **Batch path may be using different diagnostic inputs** (file path naming or metadata) that alter the classifier features.
2. **Batch path might be invoking a different profile selection branch** (e.g., missing/overridden `profile_type` mapping).
3. **Document heuristics may differ due to per-file batching** (page subset, different sample pages, or missing doc title).

---

## Constraints (Do Not Break)
- No changes to extraction thresholds, OCR logic, or VLM prompts unless required by the parity fix.
- Preserve existing observability metadata fields (profile_type, min_image_dims, etc.).
- No new regressions in technical/scanned parity (Firearms and HarryPotter already match).

---

## Required Output for the Fix
- `process` and `batch` must produce **identical profile_type** for the same input.
- Evidence must be provided via JSONL checks (profile_type counts).
- Changes must be minimal and localized to the classification/parity wiring.

---

## Suggested Verification
```bash
# Academic
jq -r '.metadata.profile_type' /tmp/repro_process/ingestion.jsonl | sort | uniq -c
jq -r '.metadata.profile_type' /tmp/repro_batch/doc1/ingestion.jsonl | sort | uniq -c

# Editorial
jq -r '.metadata.profile_type' /tmp/mmrag_full_tests/20260116_050543/sensitivity_low/ingestion.jsonl | sort | uniq -c
jq -r '.metadata.profile_type' /tmp/mmrag_full_tests/20260116_050543/batch_out/doc2/ingestion.jsonl | sort | uniq -c
```

---

## Notes for Claude
- The goal is **strict parity** between `process` and `batch`.
- If parity holds for scanned and technical documents but fails for academic/editorial, the fix should target the **decision inputs**, not the classifier logic itself.
- Start by tracing the exact pipeline and feature inputs used in each path.
