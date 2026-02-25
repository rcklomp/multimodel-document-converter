# Bug Fix Report: Process vs Batch Profile Mismatch

## Executive Summary

**Status:** ✅ **FIX APPLIED + VERIFIED**  
**Fix Applied:** Content-first domain detection + fail-fast metadata wiring  
**Remaining Issue:** None observed for parity mismatch

## Bug Reproduction Results (Pre-Fix)

### Process Command
```bash
jq -r '.metadata.profile_type' /tmp/repro_test_parity/process_output/ingestion.jsonl | sort | uniq -c
# Output: 456 academic_whitepaper
```

### Batch Command  
```bash
jq -r '.metadata.profile_type' /tmp/repro_test_parity/batch_output/doc1/ingestion.jsonl | sort | uniq -c
# Output: 455 digital_magazine
```

**Result:** ❌ **PARITY MISMATCH CONFIRMED** (academic_whitepaper vs digital_magazine)

---

## Post-Fix Verification Results

### Process Command (original filename)
```bash
jq -r '.metadata.profile_type' /tmp/mmrag_parity_fix_verify/20260116_225454/process/ingestion.jsonl | sort | uniq -c
# Output: 456 academic_whitepaper
```

### Batch Command (renamed to doc1.pdf)
```bash
jq -r '.metadata.profile_type' /tmp/mmrag_parity_fix_verify/20260116_225454/batch/doc1/ingestion.jsonl | sort | uniq -c
# Output: 456 academic_whitepaper
```

**Result:** ✅ **PARITY MATCH CONFIRMED** (academic_whitepaper vs academic_whitepaper)

---

## Root Cause Analysis

### Initial Hypothesis (INCORRECT)
The bug report suggested that the batch command had a defensive check that caused `intelligence_metadata` to become empty, leading to a fallback profile.

### Actual Root Cause (CONFIRMED)
The parity mismatch is caused by **filename-dependent domain detection** in `DocumentDiagnosticEngine`:

**Process Command (original filename):**
```
[DOMAIN-DETECT] Scores for 'Hybrid_electric_vehicles_and_their_challenges.pdf': 
  academic=7, editorial=0, technical=0, commercial=0
[DOMAIN-DETECT] → ACADEMIC (score=7)
```

**Batch Command (renamed to doc1.pdf):**
```
[DOMAIN-DETECT] Scores for 'doc1.pdf': 
  academic=0, editorial=0, technical=0, commercial=0
[DOMAIN-DETECT] → EDITORIAL (content-based: high images)
```

### Why This Causes Profile Mismatch

1. **Original filename** "Hybrid_electric_vehicles_and_their_challenges.pdf" contains keywords like "electric", "vehicles", "challenges" → scores 7 academic points
2. **Batch renamed file** "doc1.pdf" contains no keywords → scores 0, falls back to content-based detection → classified as "editorial"
3. Different domains feed into ProfileClassifier:
   - `domain=academic` → `academic_whitepaper` (score 0.750)
   - `domain=editorial` → `digital_magazine` (score 0.650)

---

## Fix Applied (Complete)

### Files Modified
- `src/mmrag_v2/orchestration/document_diagnostic.py` - Content-first domain detection (90% content / 10% filename)
- `src/mmrag_v2/cli.py` - Fail-fast validation for intelligence metadata

### Change Summary (CLI Guard)
**Before (Defensive Check):**
```python
if selected_profile and profile_params and diagnostic_report:
    intelligence_metadata = {...}
else:
    intelligence_metadata = {}  # Silent fallback
```

**After (Fail-Fast):**
```python
if not (selected_profile and profile_params and diagnostic_report):
    logger.error("[PARITY-BUG] Intelligence Stack returned None values...")
    raise ValueError("Intelligence Stack failed...")

intelligence_metadata = {...}  # Always set, or fail loudly
```

### What This Fixes
- ✅ Ensures both `process` and `batch` use identical metadata propagation logic
- ✅ Prevents silent failures that could mask bugs
- ✅ Removes filename dependence for domain detection (parity restored)
- ✅ Adds logging for parity verification

---

## Implemented Fix Strategy

### Content-First Classification (Implemented)
**Rationale:**
- Guarantees parity regardless of filename
- More robust for production pipelines
- Content features are more reliable than filenames

**Implementation:**
```python
def detect_domain(filename: str, content_features: dict) -> str:
    # Remove filename scoring entirely
    # Use ONLY content features: text_density, image_density, formatting, etc.
    
    if content_features["avg_text_per_page"] > 3000:
        if content_features["has_citations"] or content_features["has_abstracts"]:
            return "academic"
    
    if content_features["image_density"] > 0.5:
        return "editorial"
    
    # etc...
```

**Note:** The implemented approach keeps filename as a weak hint (10%) rather than removing it entirely.

---

## Verification Test Cases

### Test 1: Same File, Different Names ✅ CRITICAL
```bash
# Should produce IDENTICAL profile_type
cp source.pdf /tmp/test1.pdf
cp source.pdf /tmp/academic_paper_2025.pdf

mmrag-v2 process /tmp/test1.pdf --output-dir /tmp/out1 --vision-provider none
mmrag-v2 process /tmp/academic_paper_2025.pdf --output-dir /tmp/out2 --vision-provider none

# Compare
jq -r '.metadata.profile_type' /tmp/out1/ingestion.jsonl | sort | uniq -c
jq -r '.metadata.profile_type' /tmp/out2/ingestion.jsonl | sort | uniq -c
```

### Test 2: Process vs Batch Parity ✅ CRITICAL  
```bash
# Should produce IDENTICAL profile_type
mmrag-v2 process source.pdf --output-dir /tmp/process --vision-provider none
mmrag-v2 batch /tmp/batch_input --output-dir /tmp/batch --vision-provider none

jq -r '.metadata.profile_type' /tmp/process/ingestion.jsonl | sort | uniq -c
jq -r '.metadata.profile_type' /tmp/batch/source/ingestion.jsonl | sort | uniq -c
```

### Test 3: All Document Types
- ✅ Academic PDF (Hybrid vehicles)
- ✅ Editorial PDF (IRJET)  
- ✅ Scanned technical (Firearms)
- ✅ Scanned literature (HarryPotter)

---

## Implementation Priority (Current Status)

### Phase 1: IMMEDIATE (Completed)
- [x] Remove defensive check in batch command
- [x] Add fail-fast validation
- [x] Add parity logging

### Phase 2: HIGH PRIORITY (Completed)
- [x] Reduce filename dependency from domain detection
- [x] Enhance content-based classification features

### Phase 3: VALIDATION (Completed)
- [x] Run parity verification for renamed batch files
- [x] Verify parity across representative document types
- [x] Update CHANGELOG.md

---

## Testing Evidence

### Logs Comparison

**Process (domain=academic):**
```
[CLASSIFIER] Features: text=6472ch/pg, imgs=0.31/pg, median=298px, pages=16, scan=False, domain=academic
[CLASSIFIER] academic_whitepaper: 0.750 (confidence: 0.56) [✓ VALID]
[PROFILE] Classifier selected: academic_whitepaper
```

**Batch (domain=editorial):**
```
[CLASSIFIER] Features: text=6472ch/pg, imgs=0.31/pg, median=298px, pages=16, scan=False, domain=editorial
[CLASSIFIER] digital_magazine: 0.650 (confidence: 0.50) [✓ VALID]
[PROFILE] Classifier selected: digital_magazine
```

**Note:** All features IDENTICAL except `domain` → Different classification result

---

## Conclusion

The parity bug has a **two-layer cause**, now fully resolved:

1. **Metadata Propagation** (✅ FIXED): Batch command now fails fast instead of silently falling back
2. **Domain Detection** (✅ FIXED): Content-first scoring prevents filename renaming from flipping domain

**Next Steps:**
1. Add a targeted unit test for `_estimate_content_domain()` to prevent regression
2. Add a parity test case to `docs/TESTING.md` (renamed filename case)

---

**Report Generated:** 2026-01-16 22:33 CET  
**Author:** Cline (AI Assistant)  
**Status:** Fix Applied and Verified
