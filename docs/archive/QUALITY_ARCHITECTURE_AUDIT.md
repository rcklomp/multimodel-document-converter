# QUALITY ARCHITECTURE AUDIT: MM-Converter-V2
**Version:** v2.4.1-stable  
**Date:** 2026-01-17  
**Status:** CRITICAL - QUALITY REGRESSION IDENTIFIED  
**Compliance:** SRS v2.4 Deviations Documented

---

## EXECUTIVE SUMMARY

This audit identifies **7 violations + 1 verification required** against SRS v2.4 that have degraded output quality. The architecture drifted from the specification through incremental changes that individually seemed benign but collectively violated the "quality-first" mandate. **All deviations reduce fidelity without corresponding gains.**

### Impact Assessment
- **IRON-06 Violated:** Silent failures on missing image buffers (data loss)
- **IRON-07 Violated:** Full-page assets not VLM-verified (false positives)
- **IRON-08 Violated:** Non-atomic writes (integrity risk)
- **REQ-MM-03 Violated:** Tables missing semantic context (retrieval quality)
- **REQ-MM-03 Verification:** Image next_text_snippet requires QA validation
- **REQ-MM-05/06/07 Violated:** Shadow extraction removed (loss of safety net for missed editorial images)
- **REQ-PDF-04 Violated:** Page rendering disabled breaks padding consistency
- **REQ-SENS Violated:** Formula drift causes behavior mismatch with SRS

**Recommendation:** Immediate rollback/fix required for production readiness.

---

## ROOT CAUSE ANALYSIS

### Primary Failure Mode: **Incremental Decay Without Quality Gates**

The codebase exhibits classic "architectural drift" where small, well-intentioned changes accumulated into systemic quality regression. Each change passed code review in isolation but collectively violated the SRS contract.

### Contributing Factors

1. **Missing Quality Acceptance Tests:** No automated QA checks for IRON rules
2. **Incomplete Code Reviews:** Focus on "does it work" vs "does it meet spec"
3. **Documentation Lag:** ARCHITECTURE.md conflicts with SRS v2.4
4. **No Regression Detection:** Changes merged without before/after quality metrics

---

## EVIDENCE-BASED FAILURES

### 1. IRON-06 VIOLATION: Silent Failure on Missing Image Buffers
**Location:** `src/mmrag_v2/processor.py:594-616` (`_extract_raw_image`)

**SRS Requirement:**
> IRON-06: If a document reports visual elements but the image buffer is null, processing MUST HALT and trigger a configuration audit. Silent failures are unacceptable.

**Current Behavior:**
```python
# Line 594-616 in processor.py
if bbox_normalized and page_no not in page_images:
    logger.warning(  # ❌ SHOULD BE ERROR + HALT
        f"Page image missing for page {page_no}; "
        f"falling back to element.image without padding."
    )
if hasattr(element, "image") and element.image:
    # Falls back silently - IRON-06 violation
```

**Impact:**
- Missing 10px padding (REQ-MM-01) when fallback triggers
- No asset extraction failure → silent data loss
- Logs show "Page image missing" warnings in normal runs (should be ZERO)

**Root Cause:**
Changed from hard error to soft warning to "improve robustness," but this violates fail-fast principle. Better to fail loudly than deliver corrupted data.

---

### 2. IRON-08 VIOLATION: Non-Atomic Writes
**Location:** `src/mmrag_v2/cli.py:894` and `cli.py:1166`

**SRS Requirement:**
> IRON-08: The ingestion engine MUST use atomic write operations (append + flush) to ensure data integrity during network or process interruptions.

**Current Behavior:**
```python
# Line 894 in cli.py (process command)
output_path = proc.process_to_jsonl(str(input_file))  # ❌ Uses buffered writes

# Line 1166 in cli.py (batch command)
processor.process_to_jsonl(str(file_path))  # ❌ Same issue
```

**Correct Implementation Exists But Unused:**
```python
# processor.py:1680 - Atomic write method EXISTS
def process_to_jsonl_atomic(self, file_path: str, ...) -> str:
    with open(final_output_path, "a", encoding="utf-8") as f:
        json_line = json.dumps(chunk_dict, ensure_ascii=False)
        f.write(json_line + "\n")
        f.flush()  # ✅ Force write to disk immediately
```

**Impact:**
- Data loss on network interruptions (SSH, cloud mounts)
- Incomplete JSONL files without error indication
- Silent corruption during OOM crashes

**Root Cause:**
`process_to_jsonl_atomic` was implemented but never wired into CLI. Both commands call the non-atomic version.

---

### 3. REQ-MM-05/06/07 REMOVED: Shadow Extraction Deleted
**Location:** `src/mmrag_v2/batch_processor.py:386-395`

**SRS Requirement:**
> REQ-MM-05: Shadow Extraction: Parallel to AI analysis, perform raw physical scan of PDF stream (via PyMuPDF) to identify embedded bitmap/image objects.
> 
> REQ-MM-06: Conflict Resolution: If bitmap detected with dimensions ≥300×300px (or ≥40% page area) lacking corresponding Docling Figure/Picture block, force-extract as "Shadow Asset."

**Current Code:**
```python
# Line 386-395 in batch_processor.py
# ========================================================================
# V3.0.0: SHADOW EXTRACTION REMOVED
# ========================================================================
# Per ARCHITECTURE.md V3.0.0:
# - Shadow extraction is REMOVED from the pipeline
# - All extraction goes through UIR → ElementProcessor
```

**Impact:**
- **Critical:** Large editorial background images missed by Docling are now lost
- Magazine layouts with text-over-photo lose the photo entirely
- Quality regression: fewer assets extracted than v2.3

**Root Cause:**
ARCHITECTURE.md V3.0.0 claimed to "simplify" by removing shadow extraction, but SRS v2.4 still mandates it. Documentation conflict led to code removal.

**Evidence of Need:**
- SRS Section 4.3 explicitly requires shadow extraction as safety net
- Docling layout analyzer can miss background-layer images (known limitation)
- Combat Aircraft test case shows missing full-page editorial photos

---

### 4. IRON-07 / REQ-MM-10/11 MISSING: Full-Page Guard VLM Verification
**Location:** 
- `src/mmrag_v2/batch_processor.py:1816-1874` (`_apply_full_page_guard`)
- `src/mmrag_v2/processor.py:1323-1326` (full-page fallback bbox)

**SRS Requirement:**
> IRON-07: Shadow-extracted assets with area_ratio > 0.95 require VLM verification before inclusion.
> 
> REQ-MM-10: VLM Verification: Full-Page Guard assets MUST be verified by VLM to confirm editorial nature.

**Current Behavior in BatchProcessor:**
```python
# batch_processor.py:1816-1874 - _apply_full_page_guard
def _apply_full_page_guard(self, chunks: List[IngestionChunk]) -> List[IngestionChunk]:
    """
    IRON-07, REQ-MM-09: Apply Full-Page Guard to IMAGE chunks.
    """
    for chunk in chunks:
        if self._is_full_page_bbox(bbox):
            # ❌ NO VLM CALL HERE - just prefixes description
            chunk.metadata.visual_description = (
                f"[FULL-PAGE EDITORIAL IMAGE] {chunk.metadata.visual_description}"
            )
```

**Additional Gap in V2DocumentProcessor:**
```python
# processor.py:1323-1326
# REQ-COORD-01: bbox is REQUIRED for image modality
# Provide fallback full-page bbox if not available
image_bbox: List[int] = bbox if bbox is not None else [0, 0, COORD_SCALE, COORD_SCALE]
```

When `bbox is None`, processor creates a full-page bbox `[0, 0, 1000, 1000]` as fallback. This bypasses Full-Page Guard entirely because:
1. Guard only runs in BatchProcessor, not V2DocumentProcessor
2. Fallback bbox is treated as "valid coordinate" not "needs verification"
3. Creates IRON-03 violation risk (full-page image exports)

**What's Missing:**
- No call to `VisionManager.verify_shadow_integrity()` (exists but unused)
- No UI/navigation filtering (REQ-MM-11)
- Prefixing description ≠ verification
- V2DocumentProcessor fallback bbox bypasses all guards

**Impact:**
- Full-page UI screenshots pass through as "editorial content"
- Navigation bars, page scans included as assets
- False positives reduce retrieval precision
- V2DocumentProcessor has NO full-page protection at all

**Root Cause:**
`verify_shadow_integrity` was implemented in `vision_manager.py:1552+` but never wired into either guard location. The guards currently only do bbox checking + text prefixing.

---

### 5. Full-Page Detection Logic
**Location:** `src/mmrag_v2/batch_processor.py:1789-1814` (`_is_full_page_bbox`)

**Current Implementation:**
```python
def _is_full_page_bbox(self, bbox: Optional[List[int]]) -> bool:
    """Check if bbox covers full page (area_ratio > 0.95)."""
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    area = width * height
    full_page_area = COORD_SCALE * COORD_SCALE  # 1,000,000
    area_ratio = area / full_page_area
    return area_ratio > 0.95  # ✅ Correct
```

**Status:** ✅ **CORRECT** - Detection logic works, but verification missing (see #4)

---

### 6. REQ-MM-03 STATUS: next_text_snippet for Assets (VERIFY WITH QA)
**Location:** `src/mmrag_v2/processor.py:1000-1053` (pending context queue)

**SRS Requirement:**
> REQ-MM-03: Contextual Anchoring: JSONL entry MUST contain semantic_context with prev_text_snippet (300 chars) and next_text_snippet (300 chars).

**Current Implementation:**
```python
# processor.py:1000-1053 - Pending context queue
# V3.0.0: Pending context queue for IMAGE next_text_snippet
pending_image_chunks: List[IngestionChunk] = []

for element in doc.iterate_items():
    for chunk in self._process_element_v2(...):
        if chunk.modality == Modality.IMAGE:
            # Hold IMAGE chunk until next TEXT arrives
            pending_image_chunks.append(chunk)
        elif chunk.modality == Modality.TEXT:
            # Flush pending images with this text as next_text_snippet
            if pending_image_chunks:
                next_snippet = chunk.content[:CONTEXT_SNIPPET_LENGTH]
                for pending in pending_image_chunks:
                    pending.semantic_context.next_text_snippet = next_snippet
                    yield pending
                pending_image_chunks.clear()
            yield chunk
```

**Status:**
- ✅ **Implementation EXISTS in V2DocumentProcessor** - Pending context queue (lines 1000-1053) fills `next_text_snippet`
- ⚠️ **REQUIRES PATH-SPECIFIC VERIFICATION** - QA audit needed to confirm it works across ALL processing paths:
  - **Main processor path** (process_document): Uses pending queue ✅
  - **Batch mode** (BatchProcessor → V2DocumentProcessor): Needs verification
  - **Mapper paths** (`src/mmrag_v2/mapper.py`): Needs verification
  - **Layout-aware OCR path** (batch_processor.py:1816+): Needs verification
- Edge case: Last IMAGE on page has no next_text (acceptable per SRS "when text exists")

**Action Required:**
Run QA-CHECK **per processing path** to verify `semantic_context.next_text_snippet` is populated for IMAGE chunks:
1. Single-file processing: `mmrag-v2 process test.pdf`
2. Batch processing: `mmrag-v2 batch ./docs`
3. Layout-aware OCR: `mmrag-v2 process scan.pdf --ocr-mode layout-aware`

If any path fails validation, investigate why the pending context queue isn't triggered in that specific code path.

**Root Cause (if verification fails):**
Queue implementation exists but may have edge cases or not be consistently triggered.

---

### 7. REQ-MM-03 VIOLATION: Tables Missing Semantic Context
**Location:** `src/mmrag_v2/schema/ingestion_schema.py:732-807` (`create_table_chunk`)

**Current Signature:**
```python
def create_table_chunk(
    doc_id: str,
    content: str,
    # ...
    # ❌ NO prev_text/next_text parameters at all
) -> IngestionChunk:
```

**Compare to create_image_chunk:**
```python
def create_image_chunk(
    # ...
    prev_text: Optional[str] = None,  # ✅ Has these
    next_text: Optional[str] = None,
```

**Impact:**
- Tables have NO semantic_context field populated
- Symmetry violation: images get context, tables don't
- Retrieval quality: table chunks lack contextual grounding

**Root Cause:**
Schema design oversight. `create_table_chunk` was never updated to accept `prev_text`/`next_text` parameters like `create_image_chunk` did.

---

### 8. REQ-PDF-04 CRITICAL: generate_page_images=False Breaks Padding
**Location:** 
- `src/mmrag_v2/processor.py:320-339` (PdfPipelineOptions)
- `src/mmrag_v2/engines/pdf_engine.py:164-177` (if used)

**SRS Requirement:**
> REQ-PDF-04: Mandatory Rendering: DocumentConverter MUST be initialized with PdfPipelineOptions(do_extract_images=True). Minimum render scale: 2.0.

**Current Code:**
```python
# processor.py:320-339
pipeline_options = PdfPipelineOptions()
pipeline_options.images_scale = 2.0  # ✅
pipeline_options.generate_page_images = False  # ❌ CRITICAL ISSUE
pipeline_options.generate_picture_images = True  # ✅
pipeline_options.generate_table_images = True  # ✅
# ❌ do_extract_images NOT SET (relies on Docling default)
```

**Critical Issue:**
`generate_page_images = False` means Docling does NOT render full page images into the `page_images` dictionary. This directly causes:

1. **IRON-06 violations**: "Page image missing" warnings (lines 596-599) trigger constantly
2. **REQ-MM-01 violations**: 10px padding cannot be applied without page image buffer
3. **Fallback to element.image**: Extracts assets without consistent padding (quality degradation)

**Impact:**
- **HIGH** - This is the ROOT CAUSE of "Page image missing" warnings in normal runs
- Asset extraction quality degraded (inconsistent padding)
- Links directly to IRON-06 violation (#1 in this audit)

**Root Cause:**
Set to `False` to optimize memory/speed, but this breaks the padding guarantee and forces unreliable fallback paths. The SRS mandates page rendering for quality.

**Additional Issue - pdf_engine.py:**
```python
# pdf_engine.py:164-177
pipeline_options.generate_page_images = True  # ✅ Correct here
# But still missing explicit do_extract_images=True per SRS naming
```

If pdf_engine.py is used instead of processor.py, it works correctly. But processor.py is the primary path and has the critical bug.

---

### 9. REQ-SENS DRIFT: Fallback Strategy Formula
**Location:** `src/mmrag_v2/orchestration/strategy_orchestrator.py:225-233`

**SRS Requirement:**
> REQ-SENS-01: Sensitivity scales minimum dimension threshold linearly:  
> `min_dimension = 400px - (sensitivity * 300px)`

**Current Code:**
```python
# Line 225-233 in strategy_orchestrator.py
size_multiplier = 2.0 - (sensitivity * 1.5)
min_width = int(DEFAULT_MIN_WIDTH * size_multiplier)
# Where DEFAULT_MIN_WIDTH = 50px
```

**Calculation:**
- At sensitivity=0.5: `size_multiplier = 2.0 - 0.75 = 1.25` → `50 * 1.25 = 62.5px`
- SRS formula: `400 - (0.5 * 300) = 250px`

**Discrepancy:** 62.5px vs 250px (4x difference!)

**Impact:**
- **MODERATE** - More images extracted than SRS specifies
- Can increase noise for high-fidelity documents
- Strategy parameters don't match documented behavior

**Root Cause:**
Formula rewritten during "smart orchestration" refactor without updating SRS. Code uses multiplicative scaling, SRS uses linear subtraction.

---

## ARCHITECTURAL DRIFT TIMELINE

```
SRS v2.4 (Gold Standard)
    ↓
V2.0 Implementation (compliant)
    ↓
V2.3 Refactor (shadow extraction added)
    ↓
V3.0.0 "Simplification" ← DRIFT BEGINS
    - Shadow extraction removed
    - Atomic writes not wired
    - VLM verification skipped
    ↓
Current State (Quality Degraded)
```

---

## FIX PLAN: MINIMAL HIGH-IMPACT PATCHES

### Priority 1: Data Integrity (IRON-06, IRON-08)

#### Fix 1A: IRON-06 & REQ-PDF-04 - Enable Page Rendering + Fail Fast
**File:** `src/mmrag_v2/processor.py`  
**Lines:** 320-339 (configuration), 607-608 (error handling)

**Change 1 - ROOT CAUSE FIX (line 338):**
```python
# BEFORE:
pipeline_options.generate_page_images = False  # ❌ Breaks padding

# AFTER:
pipeline_options.generate_page_images = True  # ✅ Enable page rendering for padding
```

**Change 2 - FAIL FAST (line 607-608):**
```python
# BEFORE (line 607-608):
if bbox_normalized and page_no not in page_images:
    logger.warning(  # ❌ Soft warning
        f"Page image missing for page {page_no}; "
        f"falling back to element.image without padding."
    )

# AFTER:
if bbox_normalized and page_no not in page_images:
    error_msg = (
        f"[IRON-06 VIOLATION] Page image buffer missing for page {page_no}. "
        f"Cannot apply REQ-MM-01 10px padding. Processing HALTED. "
        f"This should NOT happen with generate_page_images=True. "
        f"Check Docling configuration and batch_processor page_images dict."
    )
    logger.error(error_msg)
    raise RuntimeError(error_msg)  # ✅ Fail fast
```

**Rationale:**
Fixing `generate_page_images=True` eliminates the root cause (missing page buffers). The fail-fast logic then acts as a safety net for any edge cases. This is a two-line fix that resolves both IRON-06 and REQ-PDF-04 issues.

**Acceptance:** 
- Zero "Page image missing" warnings in normal PDF runs
- All images get 10px padding consistently

---

#### Fix 1B: IRON-08 - Use Atomic Writes in CLI
**File:** `src/mmrag_v2/cli.py`  
**Lines:** 894, 1166

**Change 1 (line 894 in process command):**
```python
# BEFORE:
output_path = proc.process_to_jsonl(str(input_file))

# AFTER:
output_path = proc.process_to_jsonl_atomic(str(input_file))  # ✅ Atomic
```

**Change 2 (line 1166 in batch command):**
```python
# BEFORE:
processor.process_to_jsonl(str(file_path))

# AFTER:
# Use V2DocumentProcessor with atomic writes for non-PDF files
proc = V2DocumentProcessor(
    output_dir=str(doc_output_dir),
    enable_ocr=enable_ocr,
    ocr_engine=ocr_engine.value,
    # ... other params
)
output_path = proc.process_to_jsonl_atomic(str(file_path))  # ✅ Atomic
```

**Additional Issue - BatchProcessor JSONL Export:**

**Location:** `src/mmrag_v2/batch_processor.py:1190-1272`

**Current Behavior:**
```python
# Lines 1190-1272 - Non-atomic batch export
with open(output_jsonl, "w", encoding="utf-8") as f:
    for chunk in filtered_chunks:
        json_line = json.dumps(chunk_dict, ensure_ascii=False)
        f.write(json_line + "\n")
        # ❌ NO flush() per chunk - buffered writes
```

**Fix Required:**
```python
# AFTER - Atomic writes per chunk:
with open(output_jsonl, "a", encoding="utf-8") as f:  # Append mode
    for chunk in filtered_chunks:
        json_line = json.dumps(chunk_dict, ensure_ascii=False)
        f.write(json_line + "\n")
        f.flush()  # ✅ Force write to disk per chunk
```

**Note:** BatchProcessor is NOT currently atomic. It opens the file once in write mode and buffers all chunks. This violates IRON-08 and needs the same atomic write pattern as `process_to_jsonl_atomic()`. This is the **core JSONL export path** for batch processing, not ancillary logging.

**Acceptance:** Network interruption test shows no partial JSONL writes.

---

### Priority 2: Missing Assets (REQ-MM-05/06/07, IRON-07)

#### Fix 2A: Restore Shadow Extraction (Conditional)
**File:** `src/mmrag_v2/batch_processor.py`  
**Location:** After line 395 (where removal comment is)

**Strategy:** Don't revert V3.0.0 wholesale. Instead, add shadow extraction as **fallback only** when Docling misses assets.

**Implementation:**
```python
# Add after _process_single_batch logic
def _shadow_extraction_fallback(
    self,
    batch_path: Path,
    docling_chunks: List[IngestionChunk],
    page_offset: int,
) -> List[IngestionChunk]:
    """
    REQ-MM-05/06: Shadow extraction as safety net for missed assets.
    Only runs if Docling's image count is suspiciously low.
    """
    # Count Docling-extracted images per page
    images_per_page = defaultdict(int)
    for chunk in docling_chunks:
        if chunk.modality == Modality.IMAGE:
            images_per_page[chunk.metadata.page_number] += 1
    
    # Shadow scan for pages with zero images
    shadow_chunks = []
    doc = fitz.open(batch_path)
    for batch_page_idx in range(len(doc)):
        actual_page = batch_page_idx + page_offset + 1
        if images_per_page[actual_page] == 0:
            # Run PyMuPDF physical scan
            page = doc.load_page(batch_page_idx)
            image_list = page.get_images(full=True)
            if image_list:
                # REQ-MM-06 thresholds: dimensions >=300x300 or >=40% page area
                logger.warning(
                    f"[SHADOW-FALLBACK] Page {actual_page}: Docling found 0 images "
                    f"but PyMuPDF detected {len(image_list)} bitmaps. Extracting."
                )
                # Extract shadow assets with proper metadata
                # ... (implementation)
    
    doc.close()
    return shadow_chunks
```

**Acceptance:** No regression in image count vs v2.3 baseline.

---

#### Fix 2B: Wire Full-Page Guard VLM Verification (Two Locations)
**File 1:** `src/mmrag_v2/batch_processor.py`  
**Line:** 1850-1870 (inside `_apply_full_page_guard`)

**File 2:** `src/mmrag_v2/processor.py`  
**Line:** 1323-1326 (fallback bbox creation)

**Change 1 - BatchProcessor Guard:**
```python
# BEFORE (line 1850+):
if self._is_full_page_bbox(bbox):
    chunk.metadata.visual_description = (
        f"[FULL-PAGE EDITORIAL IMAGE] {chunk.metadata.visual_description}"
    )
    filtered.append(chunk)

# AFTER:
if self._is_full_page_bbox(bbox):
    # IRON-07: VLM verification required for full-page assets
    if self._vision_manager and chunk.asset_ref:
        asset_path = self.output_dir / chunk.asset_ref.file_path
        if asset_path.exists():
            try:
                from PIL import Image
                with Image.open(asset_path) as img:
                    verification = self._vision_manager.verify_shadow_integrity(
                        image=img,
                        page_number=chunk.metadata.page_number,
                    )
                    if verification.is_editorial:
                        # ✅ Verified editorial - keep with prefix
                        chunk.metadata.visual_description = (
                            f"[FULL-PAGE EDITORIAL IMAGE] {chunk.metadata.visual_description}"
                        )
                        filtered.append(chunk)
                        logger.info(
                            f"[FULL-PAGE-GUARD] Page {chunk.metadata.page_number}: "
                            f"VLM verified as editorial"
                        )
                    else:
                        # ❌ UI/Navigation - reject
                        logger.warning(
                            f"[FULL-PAGE-GUARD] Page {chunk.metadata.page_number}: "
                            f"VLM rejected as UI element - {verification.reject_reason}"
                        )
                        # Don't append - effectively filters out
            except Exception as e:
                logger.error(f"[FULL-PAGE-GUARD] VLM verification failed: {e}")
                # On error, keep asset but flag it
                chunk.metadata.visual_description = (
                    f"[FULL-PAGE UNVERIFIED] {chunk.metadata.visual_description}"
                )
                filtered.append(chunk)
    else:
        # No VLM available - use safe default (keep with warning)
        chunk.metadata.visual_description = (
            f"[FULL-PAGE UNVERIFIED] {chunk.metadata.visual_description}"
        )
        filtered.append(chunk)
```

**Change 2 - V2DocumentProcessor Fallback:**
```python
# BEFORE (processor.py:1323-1326):
# REQ-COORD-01: bbox is REQUIRED for image modality
# Provide fallback full-page bbox if not available
image_bbox: List[int] = bbox if bbox is not None else [0, 0, COORD_SCALE, COORD_SCALE]

# AFTER:
# REQ-COORD-01: bbox is REQUIRED for image modality
if bbox is None:
    # IRON-03: Full-page fallback bbox requires explicit flag or error
    if not self.allow_fullpage_fallback:  # Add this flag to __init__
        raise ValueError(
            f"[IRON-03 VIOLATION] Image element on page {page_no} has no bbox. "
            f"Cannot create full-page fallback [0,0,1000,1000] without explicit flag. "
            f"Use --allow-fullpage-fallback if this is intentional."
        )
    logger.warning(
        f"[FULL-PAGE-FALLBACK] Page {page_no}: Using full-page bbox "
        f"[0,0,1000,1000] for image without coordinates"
    )
    image_bbox = [0, 0, COORD_SCALE, COORD_SCALE]
else:
    image_bbox = bbox
```

**Acceptance:** 
- Full-page UI elements rejected with logged reason in BatchProcessor
- V2DocumentProcessor fails fast on missing bbox unless explicitly allowed

---

### Priority 3: Semantic Context (REQ-MM-03)

#### Fix 3A: Verify next_text for Images (QA Validation)
**File:** `src/mmrag_v2/processor.py`  
**Lines:** 1000-1053 (pending context queue)

**Status:** ✅ **IMPLEMENTATION EXISTS in Main Path** - Code change needed only if verification fails

**Current Implementation:**
The pending context queue (V3.0.0, processor.py:1000-1053) correctly fills `next_text_snippet` for IMAGE chunks in the main processor path by holding them until the next TEXT chunk arrives, then backfilling the context.

**Path-Specific Verification Required:**
```bash
# Test 1: Main processor path
mmrag-v2 process test.pdf --output-dir test1/
jq 'select(.modality == "image") | .semantic_context.next_text_snippet' test1/ingestion.jsonl | grep -c null
# Expected: 0 (or very low - only last images)

# Test 2: Batch processing path
mmrag-v2 process large.pdf --batch-size 10 --output-dir test2/
jq 'select(.modality == "image") | .semantic_context.next_text_snippet' test2/ingestion.jsonl | grep -c null
# Expected: 0 (or very low)

# Test 3: Layout-aware OCR path
mmrag-v2 process scan.pdf --ocr-mode layout-aware --output-dir test3/
jq 'select(.modality == "image") | .semantic_context.next_text_snippet' test3/ingestion.jsonl | grep -c null
# Expected: 0 (or very low)
```

**If ALL paths pass:** Remove this from violation list - it's fully handled.

**If ANY path fails:** The pending context queue isn't working in that path. Likely causes:
- **Batch mode:** Queue not persisted across batch boundaries
- **Layout-aware OCR:** Different chunking logic bypasses queue
- **Mapper paths:** Alternative processing doesn't use pending queue

Investigate the failing path's code flow and add queue logic if missing.

---

#### Fix 3B: Add Semantic Context to Tables
**File:** `src/mmrag_v2/schema/ingestion_schema.py`  
**Line:** 807 (`create_table_chunk` signature)

**Change:**
```python
# BEFORE (line 807):
def create_table_chunk(
    doc_id: str,
    content: str,
    source_file: str,
    file_type: FileType,
    page_number: int,
    bbox: List[int],
    hierarchy: Optional[HierarchyMetadata] = None,
    asset_path: Optional[str] = None,
    # ... other params
) -> IngestionChunk:

# AFTER:
def create_table_chunk(
    doc_id: str,
    content: str,
    source_file: str,
    file_type: FileType,
    page_number: int,
    bbox: List[int],
    hierarchy: Optional[HierarchyMetadata] = None,
    asset_path: Optional[str] = None,
    # ✅ ADD THESE:
    prev_text: Optional[str] = None,
    next_text: Optional[str] = None,
    # ... other params
) -> IngestionChunk:
    # ... existing code ...
    
    # ✅ ADD BEFORE RETURN:
    # Build semantic context if provided
    semantic_context = None
    if prev_text or next_text:
        semantic_context = SemanticContext(
            prev_text_snippet=prev_text[:300] if prev_text else None,
            next_text_snippet=next_text[:300] if next_text else None,
            parent_heading=hierarchy.parent_heading if hierarchy else None,
            breadcrumb_path=hierarchy.breadcrumb_path if hierarchy else None,
        )
    
    return IngestionChunk(
        # ... existing fields ...
        semantic_context=semantic_context,  # ✅ Add this
    )
```

**Also update ALL call sites to create_table_chunk:**

**File 1:** `src/mmrag_v2/processor.py`
- Locate calls to `create_table_chunk()`
- Add `prev_text=prev_text_context` and `next_text=next_text_context` parameters

**File 2:** `src/mmrag_v2/mapper.py:524-536`
```python
# BEFORE (mapper.py:524-536):
chunk = create_table_chunk(
    doc_id=doc_id,
    content=table_content,
    source_file=source_file,
    file_type=file_type,
    page_number=page_number,
    bbox=bbox_normalized,
    hierarchy=hierarchy,
    asset_path=asset_path,
    # ❌ NO prev_text/next_text parameters
)

# AFTER:
chunk = create_table_chunk(
    doc_id=doc_id,
    content=table_content,
    source_file=source_file,
    file_type=file_type,
    page_number=page_number,
    bbox=bbox_normalized,
    hierarchy=hierarchy,
    asset_path=asset_path,
    prev_text=prev_text_context,  # ✅ Add context
    next_text=next_text_context,  # ✅ Add context
)
```

**Note:** The mapper.py path is critical - it creates table chunks independently and must be updated alongside the schema change.

**Acceptance:** QA-CHECK-03 validates non-null prev/next_text for tables across ALL creation paths (processor.py + mapper.py).

---

### Priority 4: Behavioral Drift

#### Fix 4A: REQ-PDF-04 Clarification
**File:** `docs/SRS_Multimodal_Ingestion_V2.4.md`  
**Line:** REQ-PDF-04 section

**Change:**
```markdown
<!-- BEFORE: -->
**REQ-PDF-04**: Mandatory Rendering: DocumentConverter MUST be initialized with 
PdfPipelineOptions(do_extract_images=True).

<!-- AFTER: -->
**REQ-PDF-04**: Mandatory Rendering: DocumentConverter MUST be initialized with 
PdfPipelineOptions configured for image extraction. In Docling v2.66.0, this requires:
- `generate_page_images=True` (CRITICAL: enables page rendering for padding consistency per REQ-MM-01)
- `generate_picture_images=True` (extracts figures)
- `generate_table_images=True` (extracts tables)
- `images_scale >= 2.0` (minimum render scale)

*Note: Parameter naming changed in Docling v2.66.0. The critical quality requirement is 
`generate_page_images=True` to ensure consistent 10px padding on all extracted assets.*
```

---

#### Fix 4B: REQ-SENS Formula Correction
**File:** `src/mmrag_v2/orchestration/strategy_orchestrator.py`  
**Line:** 225-233

**Options:**
1. **Change Code to Match SRS** (recommended for spec compliance)
2. **Change SRS to Match Code** (if current formula is proven better)

**Recommended: Option 1 (Match SRS)**
```python
# BEFORE (line 229-231):
size_multiplier = 2.0 - (sensitivity * 1.5)
min_width = int(DEFAULT_MIN_WIDTH * size_multiplier)
min_height = int(DEFAULT_MIN_HEIGHT * size_multiplier)

# AFTER (SRS formula):
# REQ-SENS-01: min_dimension = 400px - (sensitivity * 300px)
base_dimension = 400
sensitivity_range = 300
min_width = int(base_dimension - (sensitivity * sensitivity_range))
min_height = int(base_dimension - (sensitivity * sensitivity_range))

# Result at sensitivity=0.5: 400 - 150 = 250px ✅
```

**Acceptance:** Min dimensions match SRS formula output.

---

## ACCEPTANCE CRITERIA (MEASURABLE)

### AC-1: IRON Rule Compliance
- [ ] Zero "Page image missing" warnings in normal PDF runs
- [ ] JSONL writes survive network interruption (test: kill during write)
- [ ] Full-page assets have VLM verification logged
- [ ] Padding applied OR processing fails (no silent fallback)

### AC-2: Asset Integrity
- [ ] Image count ≥ v2.3 baseline (no regression)
- [ ] Full-page UI elements rejected with logged reason
- [ ] Shadow extraction triggers only on low-recall pages

### AC-3: Semantic Context
- [ ] IMAGE chunks: `semantic_context.next_text_snippet` is non-null except for trailing images without subsequent text
- [ ] TABLE chunks: `semantic_context.prev_text_snippet != null` (when text exists)
- [ ] QA-CHECK-03 validates context completeness

### AC-4: Formula Parity
- [ ] Token variance within 10% tolerance (QA-CHECK-01)
- [ ] Missing assets count = 0 (QA-CHECK-02)
- [ ] Breadcrumb depth matches hierarchy.level (QA-CHECK-03)
- [ ] All bboxes 0-1000 integers (QA-CHECK-04)
- [ ] No IMAGE chunks without asset_ref (QA-CHECK-05)

---

## TEST COMMANDS

### Before/After Quality Metrics
```bash
# Baseline (current state)
mmrag-v2 process test_doc.pdf --output-dir baseline/

# After fixes
mmrag-v2 process test_doc.pdf --output-dir fixed/

# Compare (use jq if no compare script exists)
jq -r '.modality' baseline/ingestion.jsonl | sort | uniq -c
jq -r '.modality' fixed/ingestion.jsonl | sort | uniq -c

# Metrics to compare:
# - Total chunks
# - IMAGE chunks count
# - Token variance (QA-CHECK-01)
# - Missing semantic_context count
# - Full-page assets count
```

### Specific Validation Tests
```bash
# Test 1: IRON-06 (missing page image should halt)
# Manually corrupt page_images dict, verify RuntimeError

# Test 2: IRON-08 (atomic writes)
# Kill process during JSONL write, verify no corruption

# Test 3: Full-Page Guard VLM
grep "FULL-PAGE-GUARD" output/logs.txt | grep "VLM verified"

# Test 4: Semantic Context
jq '.semantic_context | select(.next_text_snippet == null)' output/ingestion.jsonl | wc -l
# Should be 0 for IMAGE/TABLE chunks

# Test 5: Formula Parity
# Run with sensitivity=0.5, verify min_dims=250px in logs
```

---

## RISK MITIGATION

### Risk 1: Fix Breaks Existing Behavior
**Mitigation:** 
- Feature flag for fixes (e.g., `--strict-iron-rules`)
- Gradual rollout: Fix 1A → 1B → 2A → etc.
- Regression test suite on v2.3 test corpus

### Risk 2: Performance Degradation
**Mitigation:**
- Shadow extraction only on low-recall pages (not every page)
- VLM verification cached (existing mechanism)
- Benchmark before/after with `validate_processing_time.py`

### Risk 3: False Positives in Full-Page Guard
**Mitigation:**
- Whitelist flag: `--allow-fullpage-shadow` (already exists)
- VLM rejection reason logged for manual review
- Confidence threshold tunable via profile params

---

## FINAL AUDIT SUMMARY

**7 Confirmed Violations:**
1. **IRON-06** (processor.py:607-608): Silent failures on missing image buffers
2. **IRON-08** (cli.py:894, 1166; batch_processor.py:1190-1272): Non-atomic writes
3. **REQ-MM-05/06/07** (batch_processor.py:386-395): Shadow extraction removed
4. **IRON-07** (batch_processor.py:1816-1874; processor.py:1323-1326): Full-page guard VLM verification missing
5. **REQ-MM-03 (Tables)** (ingestion_schema.py:732-807; mapper.py:524-536): Missing semantic context
6. **REQ-PDF-04** (processor.py:320-339): `generate_page_images=False` breaks padding
7. **REQ-SENS** (strategy_orchestrator.py:225-233): Formula drift

**1 Verification Required:**
8. **REQ-MM-03 (Images)** (processor.py:1000-1053): Implementation exists via pending context queue; requires path-specific QA validation

---

## CONCLUSION

This audit identified **7 violations + 1 verification required** against SRS v2.4, all of which reduce output quality without corresponding benefits. The root cause is architectural drift through incremental changes lacking quality gates.

**All fixes restore SRS compliance.** No changes weaken existing safeguards. Implementation time: **2-3 days** for a senior engineer. Note: Fix 1B (BatchProcessor atomic writes) requires refactoring the JSONL export loop at lines 1190-1272, not just a method swap.

**Quality-loss proof:** Each fix restores a mandatory SRS requirement that was violated. Removing these fixes would re-introduce known data loss, integrity issues, and retrieval quality degradation.

---

**Next Steps:**
1. Approve fix plan
2. Implement fixes in priority order
3. Run acceptance tests
4. Update ARCHITECTURE.md to match SRS v2.4
5. Add automated QA checks to CI/CD

---

**Prepared By:** Architecture Audit Team  
**Reviewed By:** [Pending]  
**Approved By:** [Pending]  
**Date:** 2026-01-17
