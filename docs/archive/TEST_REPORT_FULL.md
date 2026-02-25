# Test Report (Comprehensive)

**Run ID:** 20260116_050543  
**Output Root:** `/tmp/mmrag_full_tests/20260116_050543`  
**Runner:** `conda run -p /Users/ronald/Projects/MM-Converter-V2/env`  
**Endpoint:** `http://192.168.10.11:1234/v1`

## Summary
- **Completed:** Full Phase A + Phase B + Phase D + additional parity runs (Firearms/HarryPotter).
- **Primary Passes:** Core PDF paths, OCR routing variants, caching flags, and metadata propagation.
- **Notable Failures:** Anthropic provider missing API key; non-PDF formats (EPUB/HTML/DOCX/XLSX) exit non‑zero due to format/asset_ref validation; classification parity mismatch between process and batch for academic/magazine PDFs; refiner runs log missing OpenAI API key.

## Phase A — CLI Surface Coverage
**Status:** Completed
- A1/A2/A3: process/batch/batch-size ✅
- A4–A9: sensitivity, strict‑qa, semantic overlap, vlm context depth, fullpage shadow, cache flags ✅
- A10–A13: OCR engine + OCR mode + confidence threshold + Doctr on/off ✅
- A14: refiner runs completed but refiner API key missing ⚠️
- A15: pages slicing ✅

## Phase B — Vision Provider Matrix
- **OpenAI (LM Studio)**: ✅ Completed (`vision_openai/ingestion.jsonl` exists)
- **Ollama**: ✅ Completed (`vision_ollama/ingestion.jsonl` exists)
- **Anthropic**: ❌ Failed — API key missing
  - Error: `anthropic requires an API key. Use --api-key or set the appropriate environment variable.`

## Phase C — Parity (Additional Runs)
**Firearms (technical_manual):**
- process: `technical_manual`
- batch: `technical_manual`
- parity: ✅

**HarryPotter (scanned_literature):**
- process: `scanned_literature`
- batch: `scanned_literature`
- parity: ✅

**Hybrid/IRJET (academic/editorial):**
- process (Hybrid): `academic_whitepaper`
- batch (doc1=Hybrid): `digital_magazine` ❌
- process (IRJET): `academic_whitepaper`
- batch (doc2=IRJET): `digital_magazine` ❌

**Finding:** Classification parity mismatch between process and batch for the academic/editorial PDFs.

## Phase D — Non-PDF Formats
All commands executed, but several exited non‑zero due to validation errors:

- **EPUB** ❌
  - Error: `File format not allowed` (format not recognized by Docling)
- **HTML** ❌
  - Error: `QA-CHECK-05 VIOLATION: modality=table requires asset_ref` (table asset missing)
- **DOCX** ❌
  - Error: `QA-CHECK-05 VIOLATION: modality=table requires asset_ref` (table asset missing)
- **PPTX** ✅
  - Completed successfully; assets written
- **XLSX** ❌
  - Error: `QA-CHECK-05 VIOLATION: modality=table requires asset_ref` (table asset missing)

## Metadata Propagation Checks
- **Process Firearms:** ✅ 2103 chunks, all metadata non‑null
- **Batch Firearms:** ✅ 2104 chunks, all metadata non‑null
- **Process HarryPotter:** ✅ 3234 chunks, all metadata non‑null
- **Batch HarryPotter:** ✅ 3234 chunks, all metadata non‑null

## Refiner Runs
- **refiner_default / refiner_maxedit:** outputs produced, but logs show repeated
  `ContextualRefiner OpenAI API key missing` — refiner likely skipped.

## Known Warnings Observed
- QA‑CHECK‑01 token variance warnings (non‑blocking unless strict‑qa).
- `ocrmac` NoneType iterable errors in layout‑aware OCR; fallback OCR completed.
- `imagehash not installed` fallback hashing.

---

## Pass/Fail Matrix (High Level)

| Area | Status | Notes |
|------|--------|------|
| Core PDF processing | ✅ | process/batch/batch-size all completed |
| OCR routing (legacy/auto/layout-aware) | ✅ | completed; fallback used in some runs |
| Refiner integration | ⚠️ | missing API key; outputs produced but refiner not executed |
| Vision providers | ⚠️ | OpenAI/Ollama ok; Anthropic failed (API key missing) |
| Metadata propagation | ✅ | validated across key PDF runs |
| Parity (academic/editorial) | ❌ | profile_type mismatch process vs batch |
| Parity (technical/scanned) | ✅ | profile_type matches |
| Non‑PDF formats | ❌ | EPUB format not allowed; HTML/DOCX/XLSX fail QA‑CHECK‑05 |

## Evidence Files
- Results log: `/tmp/mmrag_full_tests/20260116_050543/run.log`
- Summary exits: `/tmp/mmrag_full_tests/20260116_050543/results.txt`
- Outputs: `/tmp/mmrag_full_tests/20260116_050543/*/ingestion.jsonl`

