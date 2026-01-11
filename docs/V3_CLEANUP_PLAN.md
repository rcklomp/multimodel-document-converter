# V3.0.0 Architecture Cleanup Plan

**Date:** January 2026  
**Status:** CRITICAL - Must implement properly, no more band-aids

---

## 1. The Problem

The codebase is polluted with "shadow-first" legacy patterns that violate ARCHITECTURE.md:

### Violations Found:
| File | Violation | Lines |
|------|-----------|-------|
| `batch_processor.py` | `create_shadow_chunk`, `_run_shadow_extraction`, `_run_shadow_first_extraction` | ~200 lines |
| `processor.py` | `_shadow_first_mode`, `create_shadow_splitter` | ~50 lines |
| `schema/ingestion_schema.py` | `Modality.SHADOW`, `create_shadow_chunk` | ~30 lines |
| `mapper.py` | Fallback bbox `[0, 0, 1000, 1000]` | ~10 lines |
| `cli.py` | `allow_fullpage_shadow`, `LEGACY = "legacy"` | ~20 lines |
| `chunking/shadow_content_splitter.py` | Entire file is legacy | 200+ lines |
| `orchestration/shadow_extractor.py` | Entire file is legacy | 400+ lines |

### Total Legacy Code: ~900 lines that need to be removed/refactored

---

## 2. V3.0.0 Architecture Requirements

From ARCHITECTURE.md Section 4.1 Decision Tree:
```
OUTPUT: modality="text" | "image" | "table"
        (NEVER "shadow")
```

### The Correct Flow:
1. **PDFEngine** → Extract elements with REAL bbox from Docling
2. **QualityClassifier** → Classify by confidence (not "scanned vs digital")
3. **ElementProcessor** → Route by quality:
   - TEXT + confidence ≥ 0.7 → Direct extraction
   - TEXT + confidence < 0.7 → OCR cascade
   - IMAGE → VLM visual description (VISUAL_ONLY_PROMPT)
   - TABLE → Structure extraction
4. **Output** → `modality: "text"` | `"image"` | `"table"` (NEVER shadow)

---

## 3. Required Actions

### Phase 1: Schema Cleanup
- [ ] Remove `Modality.SHADOW` from enum (keep deprecation warning only for v2.x compatibility)
- [ ] Remove `create_shadow_chunk()` function entirely
- [ ] Enforce bbox from layout engine, NOT fallback `[0,0,1000,1000]`
- [ ] Add top-level `semantic_context` field per REQ-MM-03

### Phase 2: Remove Shadow Extraction
- [ ] Delete `_run_shadow_extraction()` from `batch_processor.py`
- [ ] Delete `_run_shadow_first_extraction()` from `batch_processor.py`
- [ ] Delete `allow_fullpage_shadow` CLI option
- [ ] Delete `enable_shadow_extraction` from strategy profiles
- [ ] Delete `verify_shadow_integrity()` from VisionManager
- [ ] Archive `orchestration/shadow_extractor.py` (move to `_legacy/`)
- [ ] Archive `chunking/shadow_content_splitter.py` (move to `_legacy/`)

### Phase 3: Implement Proper UIR Pipeline
- [ ] `PDFEngine.convert()` MUST return `UniversalDocument` with REAL bbox
- [ ] `ElementProcessor` MUST route TEXT/IMAGE/TABLE by quality
- [ ] OCR cascade (Tesseract → Doctr) for low-confidence TEXT only
- [ ] VLM with `VISUAL_ONLY_PROMPT` for IMAGE only
- [ ] NO fallback to shadow extraction - fail with error instead

### Phase 4: Fix Coordinate Extraction
- [ ] `_extract_provenance()` in mapper.py MUST get REAL bbox from Docling prov
- [ ] If bbox unavailable → **LOG ERROR and SKIP element** (not fallback!)
- [ ] Investigate why Docling isn't providing bbox (API bug? wrong config?)

### Phase 5: Test Validation
- [ ] Run on Firearms.pdf with --strict mode
- [ ] Verify ALL image chunks have REAL bbox (not `[0,0,1000,1000]`)
- [ ] Verify NO "shadow" modality in output
- [ ] Verify TEXT chunks have OCR content (not VLM summaries)

---

## 4. Files to Archive/Delete

### Move to `_legacy/` directory:
```
src/mmrag_v2/_legacy/
├── shadow_extractor.py     (from orchestration/)
├── shadow_content_splitter.py  (from chunking/)
├── legacy_processor.py     (shadow-first code from batch_processor.py)
```

### Remove entirely:
- All `create_shadow_chunk()` calls
- All `Modality.SHADOW` references
- All `_shadow_first_mode` logic
- All `allow_fullpage_shadow` options

---

## 5. Breaking Changes

Users running with `--legacy-mode` flag will get an error:
```
ERROR: Legacy shadow-first mode is no longer supported.
The V3.0.0 architecture uses proper layout analysis.
See ARCHITECTURE.md for details.
```

---

## 6. Success Criteria

After cleanup, `ingestion.jsonl` for Firearms.pdf MUST:

1. **Modality**: Only `text`, `image`, `table` - NEVER `shadow`
2. **BBox**: Real coordinates from Docling, NOT `[0,0,1000,1000]`
3. **Text Content**: OCR-extracted text, NOT "The image shows a page..."
4. **Visual Descriptions**: For IMAGE only, visual content description
5. **Extraction Method**: `docling` or `ocr`, NOT `shadow`

---

## 7. Estimated Effort

| Phase | Effort | Risk |
|-------|--------|------|
| Schema Cleanup | 1 hour | Low |
| Remove Shadow | 2 hours | Medium |
| UIR Pipeline | 4 hours | High |
| Fix Coordinates | 2 hours | High |
| Testing | 2 hours | Low |

**Total: ~11 hours of focused work**

---

## 8. DO NOT:

- ❌ Add more fallback code
- ❌ Use `[0,0,1000,1000]` as bbox
- ❌ Call `create_shadow_chunk()`
- ❌ Allow "shadow" modality to pass validation
- ❌ Use VLM to describe text content

## DO:

- ✅ Get REAL bbox from Docling layout analysis
- ✅ Fail loudly if bbox unavailable (investigate and fix root cause)
- ✅ Route TEXT regions through OCR cascade
- ✅ Route IMAGE regions through VLM with VISUAL_ONLY_PROMPT
- ✅ Produce only `text`, `image`, `table` modalities

---

**END OF CLEANUP PLAN**
