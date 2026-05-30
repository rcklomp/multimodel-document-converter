# Scanned Document Quality Improvement Plan
## Layout-First Hybrid Extraction Architecture

**Document Version:** 2.1  
**Date:** January 3, 2026  
**Status:** ✅ VALIDATED - Ready for Implementation  
**Authors:** Claude (Layout Architecture), Gemini (OCR Cascade)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Critical Assessment](#critical-assessment)
3. [Current Architecture Analysis](#current-architecture-analysis)
4. [Problem Statement](#problem-statement)
5. [Phase 1: Enhanced OCR Pipeline (Gemini)](#phase-1-enhanced-ocr-pipeline-gemini)
6. [Phase 1 Concerns (Claude)](#phase-1-concerns-claude)
7. [Proposed Phase 1.5: Layout-Aware OCR](#proposed-phase-15-layout-aware-ocr)
8. [Phase 2: Full Layout-First Hybrid](#phase-2-full-layout-first-hybrid)
9. [Implementation Details](#implementation-details)
10. [Technology Recommendations](#technology-recommendations)
11. [Risk Assessment](#risk-assessment)
12. [Testing Strategy](#testing-strategy)
13. [Success Criteria](#success-criteria)
14. [Open Questions](#open-questions)

---

## ✅ WEEK 0 VALIDATION RESULTS (January 3, 2026)

### Test Results Summary

| Test | Question | Result | Decision |
|------|----------|--------|----------|
| **Test 1: Docling Layout** | Does Docling detect regions on scans? | ✅ **PASS** | Use Docling native |
| **Test 2: Tesseract Baseline** | Is Tesseract sufficient? | ❌ **FAIL** | Doctr Layer 3 REQUIRED |
| **Test 3: Processing Time** | Is <30s/page achievable? | ⚠️ **MARGINAL** | 30-60s/page, needs optimization |

### Confirmed Technology Stack

Based on validation tests, the **final technology decisions** are:

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Layout Detection** | Docling (native) | Test 1 passed - works on scans ✅ |
| **OCR Layer 1** | Docling | Existing, fast for digital |
| **OCR Layer 2** | Tesseract 5.x + Preprocessing | Standard fallback |
| **OCR Layer 3** | Doctr (REQUIRED) | Test 2 showed Tesseract insufficient |
| **VLM** | Existing (Ollama/OpenAI) | No change needed |

### Performance Budget

- **Target:** <30s/page (optimal) / <60s acceptable
- **Current:** 30-60s/page (marginal)
- **Action:** Accept for initial release, optimize in Phase 2

---

## Executive Summary

The current MM-Converter-V2 pipeline produces fundamentally different output for scanned documents compared to digital PDFs:

| Document Type | Output Quality | Confidence | Usable for RAG? |
|--------------|----------------|------------|-----------------|
| Digital PDF | Verbatim text + VLM image descriptions | 0.6-0.8 | ✅ Yes |
| Scanned PDF | VLM summaries only | 0.15-0.6 | ❌ No |

This document proposes a **two-phase improvement plan**:
- **Phase 1** (Gemini): OCR Cascade to improve confidence from 0.15-0.6 → 0.75-0.85
- **Phase 1.5** (Claude): Layout-aware OCR to separate text from image regions
- **Phase 2**: Full Layout-First Hybrid with proper TEXT/IMAGE chunk separation

---

## Critical Assessment

### Current Confidence Scores
```
Digital PDF:  0.6-0.8 confidence  ✅ Acceptable
Scanned PDF:  0.15-0.6 confidence ❌ UNUSABLE for production
```

**0.15-0.6 confidence means concretely:**
- 40-85% of text is corrupt or missing
- Specifications are missed ("Barrel length: 22 inches" → "Barrel ength 2 nches")
- Tables are unreadable
- OCR hallucinations ("III" → "111", "O" → "0")

For a **firearms catalog** where exact specifications are crucial (caliber, dimensions, years), this is **disqualifying**.

### Target: 95% Accuracy for V1
- **Volume:** 1000+ documents × 100-300 pages
- **Domain:** Technical catalogs (firearms, aviation)
- **Era:** Vintage scans (1900s-1950s)

---

## Current Architecture Analysis

### Digital Document Flow (Working Correctly)
```
PDF → Docling Layout Analysis → Text Elements → TEXT chunks (verbatim)
                              → Image Elements → IMAGE chunks (VLM description)
                              → Table Elements → TABLE chunks
```

### Scanned Document Flow (Current - Problematic)
```
PDF → Page Render (300 DPI) → OCR Hints → VLM → "A poster showing..." 
                                                       ↓
                                              content = VLM SUMMARY 😞
```

### Current Output for Scanned Documents

```json
{
  "chunk_id": "29f7c8bb7680_021_...",
  "content": "A poster featuring a semi-automatic rifle model from the Browning company...",
  "modality": "shadow",
  "metadata": {
    "page_number": 21,
    "visual_description": "A poster featuring a semi-automatic rifle..."
  }
}
```

**Problem:** The actual text "BROWNING BAR AUTOMATIC RIFLE - Caliber: .30-06 - Magazine: 4 rounds" is **never captured**.

---

## Problem Statement

### Core Issue
For scanned documents, the VLM is used as an **OCR replacement** rather than as an **image analyzer**:

1. **Loss of Verbatim Text**: Actual document text is never extracted
2. **RAG Retrieval Failure**: Searching for "magazine capacity 4 rounds" fails
3. **Hallucination Risk**: VLM may invent or misremember details
4. **Inconsistent Output**: Scanned vs digital documents produce incompatible formats

### Business Impact
- RAG systems cannot reliably retrieve information from scanned sources
- Users cannot search for specific technical specifications
- Document fidelity is compromised for archival purposes

---

## Phase 1: Enhanced OCR Pipeline (Gemini)

### Goal
Improve scanned PDF confidence from 0.15-0.6 → **0.75-0.85** through a layered OCR strategy.

### Architecture: 3-Layer OCR Cascade

```
┌─────────────────────────────────────────────────────────────┐
│                    Layer 1: Docling                         │
│  • Existing pipeline (stays primary for digital PDFs)       │
│  • Fastest route (10-20s per page)                          │
│  • Confidence threshold: >0.7 = accept                      │
└────────────────┬────────────────────────────────────────────┘
                 │ IF confidence < 0.7
                 ▼
┌─────────────────────────────────────────────────────────────┐
│              Layer 2: Tesseract 5.x + Preprocessing         │
│  • Image enhancement (deskew, denoise, contrast)            │
│  • PSM 3 (fully automatic page segmentation)                │
│  • Per-word confidence available                            │
│  • Confidence threshold: >0.7 = accept                      │
└────────────────┬────────────────────────────────────────────┘
                 │ IF confidence still < 0.7
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                Layer 3: Doctr (Fallback)                    │
│  • State-of-the-art transformers (db_resnet50 + CRNN)      │
│  • Best for degraded scans                                  │
│  • Slowest but most accurate (30-40s per page)             │
│  • FINAL PASS - accept all results                         │
└─────────────────────────────────────────────────────────────┘
```

### Module Structure

```
src/mmrag_v2/ocr/
├── __init__.py
├── enhanced_ocr_engine.py      # 3-layer cascade
├── image_preprocessor.py       # Deskew, denoise, contrast
├── tesseract_wrapper.py        # Tesseract 5.x integration
└── doctr_wrapper.py            # Doctr fallback (optional)
```

### Key Classes

```python
class OCRLayer(Enum):
    DOCLING = "docling"
    TESSERACT = "tesseract"
    DOCTR = "doctr"

@dataclass
class OCRResult:
    text: str
    confidence: float
    layer_used: OCRLayer
    word_confidences: Optional[list[float]] = None
    processing_time_ms: int = 0

class EnhancedOCREngine:
    """Cascade OCR engine that auto-escalates on low confidence."""
    
    def process_page(
        self,
        page_image: np.ndarray,
        docling_result: Optional[OCRResult] = None,
    ) -> OCRResult:
        """Cascade through layers until acceptable confidence."""
        
        # Layer 1: Use existing Docling result if available
        if docling_result and docling_result.confidence >= 0.7:
            return docling_result
        
        # Layer 2: Tesseract + preprocessing
        tesseract_result = self._run_tesseract(page_image)
        if tesseract_result.confidence >= 0.7:
            return tesseract_result
        
        # Layer 3: Doctr (last resort)
        if self.enable_doctr:
            return self._run_doctr(page_image)
        
        # Fallback: Return best available
        return max([docling_result, tesseract_result], key=lambda r: r.confidence)
```

### Image Preprocessing Pipeline

```python
class ImagePreprocessor:
    """Enhancement pipeline for degraded scans."""
    
    def enhance_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Pipeline:
        1. Grayscale conversion
        2. Deskewing (straighten rotated scans)
        3. Noise removal (salt-and-pepper)
        4. Adaptive thresholding (better than binary for old scans)
        5. Contrast enhancement
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        deskewed = self._deskew(gray)
        denoised = cv2.fastNlMeansDenoising(deskewed, h=10)
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11, C=2
        )
        return self._enhance_contrast(binary)
```

### Integration Point

**File:** `batch_processor.py`

```python
# After Docling processing, check confidence per page
if self.enable_enhanced_ocr:
    ocr_engine = EnhancedOCREngine(confidence_threshold=0.7)
    
    for page_num, page_data in enumerate(batch_result.pages):
        if page_data.ocr_confidence < 0.7:
            logger.info(f"[OCR-ENHANCE] Page {page_num}: confidence {page_data.ocr_confidence:.2f} < 0.7")
            
            page_image = self._render_page_to_image(batch_pdf_path, page_num)
            enhanced_result = ocr_engine.process_page(page_image)
            
            page_data.text = enhanced_result.text
            page_data.ocr_confidence = enhanced_result.confidence
            page_data.ocr_method = enhanced_result.layer_used.value
```

---

## Phase 1 Concerns (Claude)

### ⚠️ CRITICAL ISSUE: Full-Page OCR Without Layout Detection

Gemini's Phase 1 plan OCRs the **entire page** without distinguishing between text regions and image regions. This causes:

#### Problem 1: OCR Garbage from Image Regions
When OCR runs on a photograph of a rifle, it produces garbage:
```
Actual image: [Photo of Browning BAR rifle]
OCR output:   "BRWNNG" "lII" "___" "|||"
```
This garbage goes into the `content` field alongside real text.

#### Problem 2: No Modality Separation
The output is still just one blob of OCR text - not separated TEXT and IMAGE chunks like digital documents produce.

**Digital document output:**
```json
{"modality": "text", "content": "BROWNING BAR AUTOMATIC RIFLE..."}
{"modality": "image", "content": "Profile photo of rifle...", "asset_ref": {...}}
```

**Gemini Phase 1 output:**
```json
{"modality": "shadow", "content": "BROWNING BAR AUTOMATIC RIFLE... BRWNNG lII ___ ..."}
```

#### Problem 3: RAG Pollution
The OCR garbage from image regions pollutes the vector embeddings, causing retrieval failures.

### My Recommendation

**Don't skip Phase 1** - the OCR cascade is valuable. But **add Phase 1.5** before Phase 2:

```
Phase 1:   OCR Cascade (improve raw confidence)
Phase 1.5: Layout Detection (separate text from images) ← ADD THIS
Phase 2:   Full Layout-First Hybrid (TEXT + IMAGE chunks)
```

---

## Proposed Phase 1.5: Layout-Aware OCR

### Goal
After OCR cascade improves confidence, use **layout detection** to:
1. Identify text regions vs image regions
2. OCR only text regions
3. Create IMAGE chunks for image regions (with VLM descriptions)

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              PHASE 1.5: LAYOUT-AWARE OCR                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Page Image (from Phase 1 OCR Cascade)               │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  LAYOUT DETECTION (Docling native or LayoutParser)   │   │
│  │  • Identifies: text blocks, images, tables           │   │
│  │  • Provides: bounding boxes for each region          │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       │                                      │
│          ┌────────────┼────────────┐                        │
│          ▼            ▼            ▼                        │
│    ┌───────────┐ ┌───────────┐ ┌───────────┐               │
│    │   TEXT    │ │   IMAGE   │ │   TABLE   │               │
│    │  REGIONS  │ │  REGIONS  │ │  REGIONS  │               │
│    └─────┬─────┘ └─────┬─────┘ └─────┬─────┘               │
│          │             │             │                      │
│          ▼             ▼             ▼                      │
│    ┌───────────┐ ┌───────────┐ ┌───────────┐               │
│    │ OCR only  │ │  VLM only │ │ Table OCR │               │
│    │ on TEXT   │ │ on IMAGE  │ │ or VLM    │               │
│    └─────┬─────┘ └─────┬─────┘ └─────┬─────┘               │
│          │             │             │                      │
│          ▼             ▼             ▼                      │
│    ┌───────────┐ ┌───────────┐ ┌───────────┐               │
│    │ modality: │ │ modality: │ │ modality: │               │
│    │   text    │ │   image   │ │   table   │               │
│    └───────────┘ └───────────┘ └───────────┘               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Why This Matters

| Approach | OCR Garbage Problem | Modality Separation | RAG Quality |
|----------|--------------------|--------------------|-------------|
| Phase 1 only | ❌ Images get OCR'd | ❌ Still "shadow" | ⚠️ Polluted |
| Phase 1 + 1.5 | ✅ OCR only on text | ✅ TEXT + IMAGE | ✅ Clean |

### Implementation Sketch

```python
class LayoutAwareOCRProcessor:
    """Combines OCR cascade with layout detection."""
    
    def process_page(self, page_image: np.ndarray) -> List[Chunk]:
        # Step 1: Layout detection
        regions = self.layout_detector.detect(page_image)
        
        chunks = []
        
        for region in regions:
            if region.type == "text":
                # OCR on text region only
                ocr_result = self.ocr_engine.process_region(
                    page_image, region.bbox
                )
                chunks.append(TextChunk(
                    content=ocr_result.text,
                    confidence=ocr_result.confidence,
                    bbox=region.bbox
                ))
                
            elif region.type == "image":
                # VLM on image region
                image_crop = self._crop_region(page_image, region.bbox)
                description = self.vlm.describe(image_crop)
                chunks.append(ImageChunk(
                    content=description,
                    asset_ref=self._save_asset(image_crop),
                    bbox=region.bbox
                ))
                
            elif region.type == "table":
                # Table-specific handling
                table_result = self._process_table(page_image, region.bbox)
                chunks.append(TableChunk(
                    content=table_result.text,
                    bbox=region.bbox
                ))
        
        return chunks
```

---

## Phase 2: Full Layout-First Hybrid

This is the complete architecture described in the original plan, building on Phase 1 and 1.5:

### Expected Output for Scanned Page 21

```json
// TEXT chunk (OCR extracted - verbatim content)
{
  "chunk_id": "29f7c8bb7680_021_text_01",
  "content": "BROWNING BAR AUTOMATIC RIFLE\n\nThe Browning BAR is a gas-operated semi-automatic rifle manufactured by Fabrique Nationale (FN) in Belgium.\n\nSpecifications:\n- Caliber: .30-06 Springfield\n- Barrel Length: 22\" or 24\"\n- Magazine Capacity: 4 rounds\n- Weight: 7.5 lbs",
  "modality": "text",
  "metadata": {
    "page_number": 21,
    "extraction_source": "ocr",
    "ocr_confidence": 0.89,
    "ocr_layer": "tesseract"
  }
}

// IMAGE chunk (VLM described - visual only)
{
  "chunk_id": "29f7c8bb7680_021_figure_01",
  "content": "Profile photograph of Browning BAR rifle showing walnut Monte Carlo stock, blued steel barrel and receiver, with detachable box magazine visible.",
  "modality": "image",
  "metadata": {
    "page_number": 21,
    "extraction_source": "vlm",
    "visual_description": "Profile photograph of Browning BAR rifle..."
  },
  "asset_ref": {
    "file_path": "assets/29f7c8bb7680_021_figure_01.png"
  }
}
```

---

## Implementation Details

### Dependencies Update

**File:** `pyproject.toml`

```toml
[project.dependencies]
# ... existing dependencies ...

# OCR Enhancement (Phase 1)
pytesseract = ">=0.3.10"
opencv-python = ">=4.8.0"

[project.optional-dependencies]
ocr-enhanced = [
    "python-doctr>=0.7.0",
    "tensorflow>=2.13.0",  # Required for doctr on Apple Silicon
]
```

**Installation:**
```bash
# Base (with Tesseract)
pip install -e .
brew install tesseract  # macOS

# With Doctr fallback
pip install -e ".[ocr-enhanced]"
```

### File Structure

```
src/mmrag_v2/ocr/
├── __init__.py
├── enhanced_ocr_engine.py      # Phase 1: OCR cascade
├── image_preprocessor.py       # Phase 1: Deskew, denoise
├── tesseract_wrapper.py        # Phase 1: Tesseract integration
├── doctr_wrapper.py            # Phase 1: Doctr fallback
└── layout_aware_processor.py   # Phase 1.5: Layout + OCR

src/mmrag_v2/layout/
├── __init__.py
├── region_detector.py          # Phase 1.5: Layout detection
├── region_types.py             # Phase 1.5: Region classes
└── reading_order.py            # Phase 2: Multi-column handling
```

---

## Technology Recommendations

### OCR Engine Comparison

| Engine | Accuracy | Speed | Degraded Scans | Recommendation |
|--------|----------|-------|----------------|----------------|
| **Docling** | Good | Fast | Poor | ✅ Layer 1 (digital) |
| **Tesseract 5.x** | Good | Fast | Medium | ✅ Layer 2 (with preprocessing) |
| **Doctr** | Excellent | Slow | Excellent | ✅ Layer 3 (fallback) |
| **EasyOCR** | Good | Medium | Medium | ⚠️ Already in use |

### Layout Detection Options

| Technology | Pros | Cons | Recommendation |
|------------|------|------|----------------|
| **Docling (native)** | Already integrated | May miss some regions | ✅ Try first |
| **LayoutParser** | Excellent accuracy | Additional dependency | ✅ Backup option |
| **Doctr layout** | Combined with OCR | Less flexible | ⚠️ Consider |

### Recommended Stack

1. **OCR Layer 1:** Docling (existing)
2. **OCR Layer 2:** Tesseract 5.x + OpenCV preprocessing
3. **OCR Layer 3:** Doctr (optional, for degraded scans)
4. **Layout:** Docling native → LayoutParser fallback
5. **VLM:** Existing Ollama/OpenAI/Anthropic integration

---

## Risk Assessment

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Doctr too slow | Medium | Medium | Make optional (`--enable-doctr` flag) |
| Tesseract hallucinations on tables | High | Medium | Use Docling's table extractor as primary |
| Memory explosion on 300-page PDFs | High | Low | OCR enhancement runs per-page, `gc.collect()` preserved |
| Layout detection errors | Medium | Medium | Fallback to full-page OCR with warning |
| Breaking changes | High | Low | Feature flag for gradual rollout |

### Fallback Strategy

```python
def process_with_fallback(page_image: Image) -> List[Chunk]:
    """Process with intelligent fallback."""
    try:
        # Primary: Layout-aware OCR
        regions = layout_detector.detect(page_image)
        return process_regions(regions)
    except LayoutDetectionError:
        # Fallback: Full-page OCR (Phase 1 only)
        logger.warning("Layout detection failed, using full-page OCR")
        ocr_result = ocr_engine.process_page(page_image)
        return [create_text_chunk(ocr_result.text)]
```

---

## Testing Strategy

### Test 1: Confidence Baseline Analysis

**Goal:** Measure current confidence scores across corpus

```python
# Run BEFORE implementation to establish baseline
def analyze_confidence_distribution(pdf_dir: Path):
    results = {"digital": [], "scanned": []}
    
    for pdf_path in pdf_dir.glob("*.pdf"):
        result = processor.process_pdf(pdf_path)
        avg_confidence = result.avg_ocr_confidence
        
        category = "digital" if avg_confidence > 0.6 else "scanned"
        results[category].append({
            "file": pdf_path.name,
            "confidence": avg_confidence
        })
    
    return results
```

### Test 2: OCR Engine Comparison

**Goal:** Compare Docling vs Tesseract vs Doctr on 10 sample pages

```python
def compare_ocr_engines(pdf_path: Path, sample_pages: list[int]):
    results = []
    
    for page_num in sample_pages:
        page_image = render_page(pdf_path, page_num)
        
        tess_result = tesseract_engine.process(page_image)
        doctr_result = doctr_engine.process(page_image)
        
        results.append({
            "page": page_num,
            "tesseract_conf": tess_result.confidence,
            "doctr_conf": doctr_result.confidence,
        })
    
    return pd.DataFrame(results)
```

### Test 3: Layout Detection Accuracy

**Goal:** Verify text/image region separation

```python
def test_layout_detection(pdf_path: Path, page_num: int):
    page_image = render_page(pdf_path, page_num)
    regions = layout_detector.detect(page_image)
    
    # Visualize regions
    visualize_regions(page_image, regions)
    
    # Manual verification
    print(f"Detected {len(regions)} regions:")
    for r in regions:
        print(f"  {r.type}: {r.bbox} (conf: {r.confidence:.2f})")
```

---

## Success Criteria

### Phase 1 Success

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| Scanned PDF Confidence | 0.15-0.6 | 0.75-0.85 | Confidence baseline test |
| Processing Time | ~15s/page | <30s/page | Accept 2x slowdown for 3x accuracy |
| OCR Hallucinations | Unknown | <2% | Manual review of 50 random chunks |

### Phase 1.5 Success

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| Region Detection Accuracy | N/A | >90% | Manual review of 20 sample pages |
| Modality Separation | 0% (all shadow) | 100% | Check for text/image chunks |
| Image OCR Garbage | High | Near zero | Review image region OCR attempts |

### Phase 2 Success

| Metric | Target | Measurement |
|--------|--------|-------------|
| RAG Query Success | "magazine capacity 4 rounds" returns correct page | Manual test |
| Output Parity | Scanned output matches digital structure | Schema validation |
| Text Extraction Accuracy | >90% character accuracy | Levenshtein distance |

---

## Open Questions

### Questions for Consensus

1. **Phase 1.5 Timing:**
   - Should we implement Phase 1 and 1.5 together, or Phase 1 first as standalone?
   - My recommendation: Together, to avoid the garbage-in-output problem.

2. **Doctr Dependency:**
   - Doctr requires TensorFlow. Is this acceptable for the project?
   - Alternative: Keep as optional `--enable-doctr` flag.

3. **Layout Detection Technology:**
   - Use Docling's native layout (already integrated) or add LayoutParser?
   - Recommendation: Try Docling first, it may already work.

4. **Tesseract vs EasyOCR:**
   - Replace EasyOCR with Tesseract, or keep both?
   - Recommendation: Tesseract for Phase 1 cascade (better confidence scores), keep EasyOCR for hints.

5. **Performance Budget:**
   - Acceptable processing time per page: 30s? 60s?
   - Recommendation: Accept 30s for initial release, optimize later.

---

## Implementation Timeline (Revised)

### WEEK 0: Validation Tests (BEFORE ANY CODING)
**Goal:** Test 3 critical assumptions before writing any code.

Run these tests to determine:
1. Does Docling layout work on scans? (determines layout tech choice)
2. What's Tesseract baseline on vintage scans? (determines if Doctr needed)
3. What's realistic processing time? (determines optimization needs)

See `tests/validate_*.py` scripts.

### Week 1: Phase 1A - Pure OCR Engine (Isolated)
**Goal:** Improve OCR confidence without layout dependency

- Day 1-2: Implement `EnhancedOCREngine` as standalone module
- Day 3: Implement `ImagePreprocessor` (deskew, denoise, contrast)
- Day 4: Unit tests on **hand-cropped** text regions
- Day 5: Validate confidence improvement: 0.15-0.6 → 0.75-0.85

**Success Criteria:** OCR cascade works on pure text crops (no mixed content).

### Week 2: Phase 1B - Layout Detection Integration
**Goal:** Add layout awareness to route regions to OCR or VLM

- Day 1-2: Implement layout detector (LayoutParser or Docling based on Week 0 tests)
- Day 3: Integrate `EnhancedOCREngine` with layout router
- Day 4: Test on 10 full scanned pages
- Day 5: Validate TEXT/IMAGE chunk separation

**Success Criteria:** Pages produce both `modality: text` and `modality: image` chunks.

### Week 3: Full Integration + Tuning
**Goal:** Integrate into main pipeline with feature flag

- Day 1-2: Add to `BatchProcessor` with `--ocr-mode=layout-aware` flag
- Day 3: A/B test: legacy shadow vs new layout-aware
- Day 4: Tune preprocessing parameters for vintage catalogs
- Day 5: Documentation + final validation

---

## Immediate Next Steps

### For You (Ronald):
1. Run baseline confidence test on 10-20 sample PDFs
2. Confirm Tesseract installation: `tesseract --version` (should be 5.x)
3. Decide on Phase 1 vs Phase 1+1.5 approach

### For Me (Claude):
Once you share baseline data:
1. Generate complete `enhanced_ocr_engine.py` code
2. Tune preprocessing parameters for 1900s-era catalogs
3. Test layout detection with Docling native

---

## References

- [Docling Documentation](https://github.com/DS4SD/docling)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Doctr Documentation](https://github.com/mindee/doctr)
- [LayoutParser Paper](https://arxiv.org/abs/2103.15348)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Current SRS Document](./SRS_Multimodal_Ingestion_V2.3.md)

---

*This document is a living proposal. Version 2.0 integrates Gemini's OCR Cascade (Phase 1) with Claude's Layout-First approach (Phase 1.5). Awaiting consensus before implementation.*
