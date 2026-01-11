# Comprehensive Analysis: MM-Converter-V2 vs SRS v2.3

## Executive Summary

After thorough analysis of the SRS v2.3 requirements, the SCANNED_DOCUMENT_IMPROVEMENT_PLAN, the ARCHITECTURE.md design, the source code, and the actual output in `output/Firearms_10/ingestion.jsonl`, I've identified **17 issues** across 5 categories that need to be addressed.

---

## 🔴 CRITICAL ISSUES (Must Fix)

### 1. Schema Version Mismatch
- **SRS Requirement:** `schema_version: "2.3.0"`
- **Actual Output:** `schema_version: "2.0.0"`
- **Location:** `src/mmrag_v2/schema/ingestion_schema.py` line 30

### 2. Missing `spatial.bbox` for Image/Table Modalities (REQ-COORD-01)
- **SRS Section 6.3:** `metadata.spatial.bbox` is **REQUIRED** when modality is `image` or `table`
- **Actual Output:** `"spatial": null` for ALL image chunks
- **Impact:** CRITICAL - Violates mandatory field requirement
- **Example from output:**
  ```json
  {"modality": "image", "metadata": {"spatial": null, ...}}
  ```

### 3. Missing `semantic_context` Top-Level Field (REQ-MM-03)
- **SRS Section 6.1:** Requires top-level `semantic_context` with `prev_text_snippet` and `next_text_snippet`
- **Current Schema:** Uses `metadata.prev_text` instead
- **Impact:** Schema non-compliance

### 4. Hierarchy Level Not Populated (REQ-HIER-04)
- **SRS Requirement:** `breadcrumb_path` depth MUST match `hierarchy.level`
- **Actual Output:** `"level": null` for all chunks while `breadcrumb_path: ["Firearms"]` (depth=1)
- **Should be:** `level: 1` for this case

### 5. Breadcrumb Path Too Shallow (REQ-HIER-03)
- **SRS Section 3.4:** For periodicals, minimum Levels 1-3; fallback: `[source_filename, page_number]`
- **Actual Output:** Only `["Firearms"]` - no page structure
- **Should include:** At minimum page-level context

---

## 🟠 HIGH PRIORITY ISSUES

### 6. OCR Quality Problems (REQ-PATH-06)
Evidence of OCR errors in output:
- Page 1: `"Jamu la Frizgi"` (garbage OCR)
- Page 3: `"The Goo Digest"` should be `"The Gun Digest"`
- Page 10: `"samaring"` should be `"non-marring"`
- **Missing:** `ocr_confidence` is `null` for all chunks - should flag low confidence

### 7. VLM Reading Text (Violates VLM_VISUAL_ONLY_PROMPT)
Some VLM descriptions contain text reading:
```json
"content": "An advertisement featuring a vintage Mauser rifle, as indicated by the text in the document context which refers to \"Mauser\""
```
- **Issue:** VLM is reading document text, violating VISUAL_ONLY constraint
- **Impact:** Pollutes embeddings with duplicate/hallucinated text

### 8. Missing Full-Page Guard Implementation (IRON-07, REQ-MM-08 through REQ-MM-12)
- **SRS Section 4.4:** Assets with `area_ratio > 0.95` require VLM verification
- **Current Code:** `layout_aware_processor.py` has fallback that creates full-page text region but no Full-Page Guard validation
- **Impact:** Could allow prohibited full-page captures

### 9. Missing `extraction_method` Values (REQ-SCHEMA)
- **SRS Section 6.1:** Allowed values: `"docling|shadow|ocr"`
- **Actual Output:** Uses `"layout_aware_ocr"` and `"layout_aware_vlm"` - not in spec
- **Should be:** Map to standard values or update SRS

---

## 🟡 MEDIUM PRIORITY ISSUES

### 10. Architecture Not Fully Implemented
- **ARCHITECTURE.md** describes Universal Intermediate Representation (UIR) pipeline
- **Current Implementation:** `processor.py` still uses V2DocumentProcessor with older approach
- **Gap:** `universal/` modules exist but aren't fully integrated in main flow

### 11. Shadow Modality Still in Schema
- **SRS v2.3:** Does not define "shadow" as valid modality
- **ARCHITECTURE.md:** States "output NEVER contains shadow modality"
- **Current Schema:** Still has `Modality.SHADOW` enum value

### 12. OCR Cascade Not Using Doctr (REQ-PATH-04)
- **SCANNED_DOCUMENT_IMPROVEMENT_PLAN:** Doctr Layer 3 marked as REQUIRED
- **Current Implementation:** `enhanced_ocr_engine.py` has Doctr but may not be enabled by default
- **Validation Test 2:** Tesseract alone is INSUFFICIENT for vintage scans

### 13. Missing `file_size_bytes` in Asset Reference
- **SRS Section 6.1:** `asset_ref.file_size_bytes` is defined
- **Actual Output:** `"file_size_bytes": null` for all assets

### 14. Content Classification Missing (REQ-VLM)
- **SRS Section 6.1:** `content_classification: "editorial|technical|advertisement"`
- **Actual Output:** Field not populated

---

## 🟢 LOW PRIORITY ISSUES

### 15. Logging Version Check (REQ-ERR-03)
- **SRS Requirement:** On startup, log `"Using Docling v2.66.0"`
- **Current:** Logs `ENGINE_USE: Docling v2.66.0` but should verify actual version

### 16. Asset Naming Pattern Inconsistency
- **SRS REQ-MM-02:** `[DocHash]_[PageNum]_[Type]_[Index].png`
- **Current:** Uses `_{page:03d}_` (3 digits) but SRS example shows 3 digits
- **Minor:** Actually compliant, but page 1 shows as `001` which is correct

### 17. Test Coverage Gaps
- No validation test for REQ-COORD-01 through REQ-COORD-05
- No test for Full-Page Guard (IRON-07)
- Missing integration test for complete schema compliance

---

## Proposed Improvement Plan

### Phase 1: Schema Compliance (Priority: CRITICAL)
1. Update `SCHEMA_VERSION` to `"2.3.0"`
2. Add `semantic_context` top-level field to schema
3. Ensure `spatial.bbox` is populated for all image/table chunks
4. Implement hierarchy level calculation from breadcrumb depth
5. Remove or deprecate `Modality.SHADOW`

### Phase 2: OCR Quality (Priority: HIGH)
1. Enable Doctr Layer 3 by default for scanned documents
2. Implement OCR confidence scoring and populate `ocr_confidence`
3. Add low-confidence flagging (REQ-PATH-06: <70% = "low")
4. Improve image preprocessing for vintage scans

### Phase 3: VLM Enhancement (Priority: HIGH)
1. Enforce stricter VISUAL_ONLY_PROMPT with post-processing validation
2. Implement text-reading detection and retry mechanism
3. Add Full-Page Guard VLM verification

### Phase 4: Architecture Alignment (Priority: MEDIUM)
1. Complete UIR integration from `universal/` modules
2. Implement quality-based routing as designed in ARCHITECTURE.md
3. Add confidence normalization across all extraction methods

### Phase 5: Testing & Validation (Priority: MEDIUM)
1. Add schema compliance validation test
2. Add coordinate system validation test (REQ-COORD-01)
3. Add Full-Page Guard test
4. Add OCR cascade performance benchmark

---

## Recommended Immediate Actions

1. **Schema Fix** (1 hour): Update `ingestion_schema.py` to v2.3.0 compliance
2. **Bbox Population** (2 hours): Ensure all image/table chunks have normalized bbox
3. **OCR Confidence** (2 hours): Track and report OCR confidence scores
4. **Hierarchy Level** (1 hour): Auto-calculate level from breadcrumb depth
