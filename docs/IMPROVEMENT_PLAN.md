# MM-Converter-V2 Improvement Plan

## Analysis Date: 2026-01-08
## SRS Version: 2.3.0

---

## Executive Summary

After comprehensive analysis of the SRS v2.3.0 requirements against the actual implementation and output in `output/Firearms_10/ingestion.jsonl`, 17 issues were identified across 5 categories. This document tracks the status of all fixes.

---

## ✅ COMPLETED FIXES

### Phase 1: Schema Compliance (CRITICAL)

| Issue | Status | Description |
|-------|--------|-------------|
| Schema Version | ✅ FIXED | Updated from "2.0.0" to "2.3.0" |
| SemanticContext | ✅ FIXED | Added top-level `semantic_context` field per REQ-MM-03 |
| bbox Required | ✅ FIXED | Image/Table chunks now REQUIRE `spatial.bbox` with validation |
| Hierarchy Level | ✅ FIXED | Auto-calculated from `breadcrumb_path` depth (REQ-HIER-04) |
| SHADOW Modality | ✅ FIXED | Deprecated with warnings, use IMAGE with `extraction_method="shadow"` |
| OCR Confidence | ✅ FIXED | Added `get_ocr_confidence_level()` for high/medium/low strings |
| Fallback bbox | ✅ FIXED | Images/Tables without bbox now get [0, 0, 1000, 1000] fallback |

### Files Modified:
- `src/mmrag_v2/schema/ingestion_schema.py` - Complete v2.3.0 compliance
- `src/mmrag_v2/processor.py` - Updated to use new schema

---

## 🔶 REMAINING ISSUES (To Be Implemented)

### Phase 2: OCR Quality Improvements

| Issue | Priority | Description |
|-------|----------|-------------|
| REQ-PATH-04 | HIGH | Enable Doctr Layer 3 by default for scanned documents |
| REQ-PATH-06 | HIGH | Populate `ocr_confidence` field with actual Tesseract/Doctr scores |
| OCR Cascade | MEDIUM | Improve cascade for vintage magazine text ("The Goo Digest" → "The Gun Digest") |
| Image Preprocessing | MEDIUM | Add deskew, binarization for degraded scans |

### Phase 3: VLM Enhancement

| Issue | Priority | Description |
|-------|----------|-------------|
| VLM_VISUAL_ONLY | HIGH | Enforce stricter visual-only descriptions (no text reading) |
| Text Detection | HIGH | Post-process VLM output to filter text-based content |
| Full-Page Guard | HIGH | Implement REQ-MM-08 through REQ-MM-12 (area_ratio > 0.95 check) |
| Retry Logic | MEDIUM | Add retry mechanism when VLM reads text instead of describing visuals |

### Phase 4: Architecture Alignment

| Issue | Priority | Description |
|-------|----------|-------------|
| UIR Integration | MEDIUM | Complete Universal Intermediate Representation pipeline |
| Quality Router | MEDIUM | Implement quality-based element routing per ARCHITECTURE.md |
| Confidence Normalization | LOW | Normalize confidence scores across extraction methods |

### Phase 5: Testing & Validation

| Issue | Priority | Description |
|-------|----------|-------------|
| Schema Validation Test | HIGH | Add test to validate all output against SRS v2.3.0 schema |
| REQ-COORD Tests | HIGH | Add tests for coordinate normalization (REQ-COORD-01 through 05) |
| Full-Page Guard Test | MEDIUM | Add test for IRON-07 full-page rejection |
| OCR Benchmark | MEDIUM | Add benchmark for OCR cascade performance |

---

## Schema Changes Detail

### New Fields Added (v2.3.0):

```python
# IngestionChunk
semantic_context: Optional[SemanticContext]  # REQ-MM-03

# SemanticContext (new class)
prev_text_snippet: Optional[str]  # max 300 chars
next_text_snippet: Optional[str]  # max 300 chars

# ChunkMetadata
content_classification: Optional[str]  # editorial|technical|advertisement
ocr_confidence: Optional[str]  # high|medium|low

# BoundingBox
area_ratio()  # For Full-Page Guard
is_full_page()  # REQ-MM-09
```

### Breaking Changes:

1. **`create_image_chunk(bbox=...)`** - bbox is now REQUIRED (was Optional)
2. **`create_table_chunk(bbox=...)`** - bbox is now REQUIRED (was Optional)
3. **`Modality.SHADOW`** - DEPRECATED, use IMAGE with extraction_method="shadow"

### Deprecation Warnings:

```python
# Using SHADOW modality will emit:
DeprecationWarning: SHADOW modality is DEPRECATED per ARCHITECTURE.md.
Use IMAGE modality with extraction_method='shadow' instead.
SHADOW will be removed in v3.0.0

# Using create_shadow_chunk will emit:
DeprecationWarning: create_shadow_chunk is DEPRECATED.
Use create_image_chunk with extraction_method='shadow' instead.
```

---

## Validation Commands

After running conversion, validate output with:

```bash
# Check schema version
jq -r '.schema_version' output/Firearms_10/ingestion.jsonl | head -1
# Expected: 2.3.0

# Check all image chunks have bbox
jq -r 'select(.modality=="image") | .metadata.spatial.bbox' output/Firearms_10/ingestion.jsonl

# Check hierarchy levels are set
jq -r '.metadata.hierarchy | "\(.breadcrumb_path | length) -> \(.level)"' output/Firearms_10/ingestion.jsonl

# Check semantic_context is present for images
jq -r 'select(.modality=="image") | .semantic_context' output/Firearms_10/ingestion.jsonl
```

---

## Next Steps

1. **Re-run Firearms_10 conversion** with updated code to verify schema compliance
2. **Implement Phase 2** OCR quality improvements for scanned documents
3. **Add validation tests** for schema compliance
4. **Review VLM prompts** to enforce visual-only descriptions

---

## References

- `SRS_Multimodal_Ingestion_V2.3.md` - Requirements specification
- `SCANNED_DOCUMENT_IMPROVEMENT_PLAN.md` - OCR cascade improvements
- `ARCHITECTURE.md` - System architecture design
