# 🏗️ Universal Multi-Format RAG Pipeline Architecture

**Version:** v2.4.1-stable  
**Date:** January 2026  
**Status:** ACTIVE BASELINE  
**Policy Update (v2.4.1-stable):** Native-digital PDFs bypass the OCR cascade (Docling text layer + recovery only). Layout-aware OCR (Tesseract/Doctr) is reserved for scanned/unknown modalities. Gap-fill recovery on academic whitepapers uses a 60-character minimum block to fill low-coverage pages with strict deduplication and noise filters.

---

**Versioning Note:** `schema_version` is sourced from `src/mmrag_v2/version.py` (`__schema_version__`) and emitted as the top-level `schema_version` field for each chunk. The export layer may also mirror it in `metadata.schema_version` for downstream compatibility.

**Scope Note:** This document is branch-canonical for `MM-Converter-V2.4.1`; file paths and module inventory reflect the current branch. Optional engines such as `src/mmrag_v2/engines/epub_engine.py` and `src/mmrag_v2/engines/html_engine.py` may exist in sibling repos, but are not required in this branch.

## 1. Executive Summary

This document describes the current MM-Converter-V2.4.1-stable architecture for robust multimodal ingestion across supported formats.

### Problem Statement

The legacy system produces VLM summaries for scanned documents instead of OCR-extracted text:

```
BROKEN OUTPUT:
{"modality": "shadow", "content": "The image shows a page titled..."}

CORRECT OUTPUT:
{"modality": "text", "content": "INTRODUCTION\n\nThis manual covers..."}
{"modality": "image", "content": "Exploded view diagram of trigger assembly"}
```

### Solution

A **Universal Intermediate Representation (UIR)** that decouples format-specific extraction from quality-based processing, ensuring:

1. TEXT regions → OCR cascade → `modality: "text"`
2. IMAGE regions → VLM visual description → `modality: "image"`
3. TABLE regions → Structure extraction → `modality: "table"`

---

## 2. System Architecture

### 2.1 High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     UNIVERSAL DOCUMENT PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT FILES                                                                │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │   PDF   │  │  ePub   │  │  HTML   │  │  DOCX   │  │  PPTX   │            │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘            │
│       │            │            │            │            │                 │
│       ▼            ▼            ▼            ▼            ▼                 │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                    FORMAT ROUTER                            │            │
│  │  router.detect(path) → appropriate FormatEngine             │            │
│  │  Uses: magic bytes, extension, content sampling             │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                               │                                             │
│       ┌───────────────────────┼───────────────────────┐                     │
│       ▼                       ▼                       ▼                     │
│  ┌──────────┐           ┌──────────┐           ┌──────────┐                 │
│  │PDFEngine │           │EpubEngine│           │HTMLEngine│                 │
│  │ (Docling)│           │(EbookLib)│           │(Trafil.) │                 │
│  └────┬─────┘           └────┬─────┘           └────┬─────┘                 │
│       │                      │                      │                       │
│       └──────────────────────┼──────────────────────┘                       │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │           UNIVERSAL INTERMEDIATE REPRESENTATION             │            │
│  │                                                             │            │
│  │  UniversalDocument                                          │            │
│  │    ├── doc_id: str (first 12 chars of MD5 hash)             │            │
│  │    ├── source_file: str                                     │            │
│  │    ├── file_type: FileType                                  │            │
│  │    └── pages: List[UniversalPage]                           │            │
│  │          ├── page_number: int                               │            │
│  │          ├── classification: "digital" | "scanned"          │            │
│  │          ├── dimensions: (width, height)                    │            │
│  │          └── elements: List[Element]                        │            │
│  │                ├── type: TEXT | IMAGE | TABLE               │            │
│  │                ├── content: str (extracted text/empty)      │            │
│  │                ├── bbox: [x1, y1, x2, y2] (0-1000)          │            │
│  │                ├── confidence: float (0.0-1.0)              │            │
│  │                └── raw_image: Optional[np.ndarray]          │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │              QUALITY-BASED ELEMENT PROCESSOR                │            │
│  │                                                             │            │
│  │  for element in page.elements:                              │            │
│  │                                                             │            │
│  │    if element.type == TEXT:                                 │            │
│  │      if element.confidence >= 0.7:                          │            │
│  │        → Use extracted content directly                     │            │
│  │      else:                                                  │            │
│  │        → OCR Cascade (Tesseract → Doctr)                    │            │
│  │      OUTPUT: modality="text"                                │            │
│  │                                                             │            │
│  │    elif element.type == IMAGE:                              │            │
│  │      → VLM visual description (NO text reading)             │            │
│  │      OUTPUT: modality="image"                               │            │
│  │                                                             │            │
│  │    elif element.type == TABLE:                              │            │
│  │      → Structure-preserving extraction                      │            │
│  │      OUTPUT: modality="table"                               │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                    OUTPUT GENERATOR                         │            │
│  │                                                             │            │
│  │  ingestion.jsonl + assets/                                  │            │
│  │  - Every chunk has proper modality (text/image/table)       │            │
│  │  - Coordinates normalized to 0-1000 (REQ-COORD-01)          │            │
│  │  - Assets named per REQ-MM-02                               │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Responsibilities

| Component | Responsibility | Input | Output | Implementation File |
|-----------|---------------|-------|--------|---------------------|
| **FormatRouter** | Detect file format, route to engine | File path | FormatEngine | `src/mmrag_v2/universal/router.py` |
| **PDFEngine** | Extract from PDF via Docling | PDF file | UniversalDocument | `src/mmrag_v2/engines/pdf_engine.py` |
| **EpubEngine** | Extract from ePub via EbookLib | ePub file | UniversalDocument | `src/mmrag_v2/engines/epub_engine.py` (optional module, not present in this branch) |
| **HTMLEngine** | Extract from HTML via Trafilatura | HTML file | UniversalDocument | `src/mmrag_v2/engines/html_engine.py` (optional module, not present in this branch) |
| **QualityClassifier** | Assess page/element quality | UniversalPage | Classification | `src/mmrag_v2/universal/quality_classifier.py` |
| **ConfidenceNormalizer** | Normalize cross-format confidence to universal 0.0-1.0 scale | Format-specific confidence metrics | Normalized score + quality tier/OCR trigger decisions | `src/mmrag_v2/universal/quality_classifier.py` |
| **ElementProcessor** | Route elements by quality | Element | IngestionChunk | `src/mmrag_v2/universal/element_processor.py` |
| **EnhancedOCREngine** | 3-layer OCR cascade | Image region | OCR text | `src/mmrag_v2/ocr/enhanced_ocr_engine.py` |
| **VisionManager** | Visual description | Image | Description | `src/mmrag_v2/vision/vision_manager.py` |
| **SmartConfigProvider** | Profile document characteristics before extraction | PDF path | DocumentProfile | `src/mmrag_v2/orchestration/smart_config.py` |
| **DocumentDiagnosticEngine** | Pre-flight modality/domain diagnostics | PDF path | DiagnosticReport | `src/mmrag_v2/orchestration/document_diagnostic.py` |
| **ProfileClassifier** | Select profile using weighted feature scoring | DocumentProfile + DiagnosticReport | ProfileType | `src/mmrag_v2/orchestration/profile_classifier.py` |
| **StrategyOrchestrator** | Convert profile into extraction strategy parameters | DocumentProfile + ProfileParameters | ExtractionStrategy | `src/mmrag_v2/orchestration/strategy_orchestrator.py` |
| **MagazineSectionDetector** | Detect section headers for breadcrumb enrichment | Text element + page context | SectionDetectionResult | `src/mmrag_v2/state/magazine_section_detector.py` |
| **SpatialPropagator** | Propagate/normalize spatial metadata to 0-1000 | Docling element + page dimensions | SpatialExtractionResult | `src/mmrag_v2/utils/advanced_spatial_propagator.py` |
| **QualityFilterTracker** | Track filtered tokens/chunks for QA accounting | Filtered chunks | QualityFilterSummary | `src/mmrag_v2/validators/quality_filter_tracker.py` |

---

## 3. Universal Intermediate Representation (UIR)

### 3.1 Core Data Structures

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

class ElementType(Enum):
    """Type of document element."""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"

class PageClassification(Enum):
    """Quality classification of a page."""
    DIGITAL = "digital"      # Native text extraction works
    SCANNED = "scanned"      # Requires OCR
    HYBRID = "hybrid"        # Mixed content

@dataclass
class Element:
    """A single document element (text block, image, or table)."""
    type: ElementType
    content: str                           # Extracted text or empty
    bbox: List[int]                        # [x1, y1, x2, y2] normalized 0-1000
    confidence: float                      # Extraction confidence 0.0-1.0
    raw_image: Optional[np.ndarray] = None # For OCR fallback
    extraction_method: str = "unknown"
    element_index: int = 0

@dataclass
class UniversalPage:
    """A single page from any document format."""
    page_number: int
    elements: List[Element]
    classification: PageClassification
    dimensions: Tuple[int, int]            # (width, height) in pixels
    raw_image: Optional[np.ndarray] = None # Full page render for fallback

@dataclass
class UniversalDocument:
    """Format-agnostic document representation."""
    doc_id: str                            # first 12 chars of MD5 hash of source file
    source_file: str                       # Original filename
    file_type: str                         # pdf, epub, html, docx, etc.
    pages: List[UniversalPage] = field(default_factory=list)
    total_pages: int = 0
```

### 3.2 Design Principles

1. **Format Agnostic**: All engines output the same structure
2. **Quality Embedded**: Confidence scores enable intelligent routing
3. **OCR Ready**: `raw_image` fields allow deferred OCR processing
4. **Coordinate Normalized**: All bboxes in 0-1000 range (SRS REQ-COORD-01)

---

## 4. Quality-Based Routing

### 4.1 Decision Tree

```
                    ┌─────────────────┐
                    │    Element      │
                    │   from UIR      │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  element.type?  │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────▼────┐         ┌────▼────┐         ┌────▼────┐
    │  TEXT   │         │  IMAGE  │         │  TABLE  │
    └────┬────┘         └────┬────┘         └────┬────┘
         │                   │                   │
    ┌────▼────────────┐      │              ┌────▼────┐
    │ confidence≥0.7? │      │              │  OCR +  │
    └────┬────────────┘      │              │ Struct  │
         │                   │              └────┬────┘
    ┌────┴────┐              │                   │
    │Yes      │No            │                   │
    ▼         ▼              ▼                   ▼
┌───────┐ ┌───────┐    ┌───────────┐       ┌────────┐
│Direct │ │  OCR  │    │VLM Visual │       │Markdown│
│Extract│ │Cascade│    │   Only    │       │ Table  │
└───┬───┘ └───┬───┘    └─────┬─────┘       └───┬────┘
    │         │              │                 │
    ▼         ▼              ▼                 ▼
┌────────────────────────────────────────────────────┐
│            modality: "text" | "image" | "table"    │
│                    (NEVER "shadow")                │
└────────────────────────────────────────────────────┘
```

### 4.2 Confidence Thresholds

| Threshold | Meaning | Action |
|-----------|---------|--------|
| ≥ 0.85 | EXCELLENT quality | Direct text extraction |
| 0.7 - 0.85 | GOOD quality | Direct extraction with validation |
| 0.5 - 0.7 | FAIR quality | OCR cascade with comparison |
| < 0.5 | POOR quality | Full OCR cascade |

### 4.3 Confidence Normalization

**Problem**: Confidence scores from different extraction engines are NOT directly comparable:
- Docling: Measures font embedding and text layer quality (0.0-1.0)
- Tesseract: Measures OCR character confidence (0-100)
- Trafilatura: Measures content extraction quality (0-100)
- EbookLib: Binary success/failure with text length heuristic

**Solution**: `ConfidenceNormalizer` maps all scores to a universal 0.0-1.0 scale.
Implementation module: `src/mmrag_v2/universal/quality_classifier.py` (exported via `src/mmrag_v2/universal/__init__.py`).
Runtime placement: UIR quality layer; consumed by `ElementConfidenceCalculator` and `PageQualityClassifier` in the same module.

```python
from mmrag_v2.universal import ConfidenceNormalizer

# PDF confidence (Docling score + adjustments)
pdf_conf = ConfidenceNormalizer.normalize_pdf(
    docling_score=0.85,
    text_length=500,
    page_image_ratio=0.3,
)  # → 0.85

# ePub confidence (text length based)
epub_conf = ConfidenceNormalizer.normalize_epub(
    text_length=1500,
    has_toc=True,
    chapter_count=12,
)  # → 0.95

# HTML confidence (Trafilatura 0-100 → 0.0-1.0)
html_conf = ConfidenceNormalizer.normalize_html(
    trafilatura_score=75,
    has_article_tag=True,
)  # → 0.80

# Office formats
docx_conf = ConfidenceNormalizer.normalize_docx(text_length=2000)  # → 0.9
pptx_conf = ConfidenceNormalizer.normalize_pptx(slide_count=20, text_length=4000)  # → 0.85

# OCR engines
tesseract_conf = ConfidenceNormalizer.normalize_ocr_tesseract(tesseract_conf=85)  # → 0.85
doctr_conf = ConfidenceNormalizer.normalize_ocr_doctr(doctr_conf=0.9)  # → 0.95
```

**Normalization Heuristics**:

| Format | Native Metric | Normalization Strategy |
|--------|---------------|------------------------|
| PDF (Docling) | 0.0-1.0 | Use as-is, adjust for image ratio |
| ePub | Text length | `>2000 chars = 0.9`, `>500 = 0.8`, etc. |
| HTML (Trafilatura) | 0-100 | Divide by 100, boost for semantic tags |
| DOCX | Text length | Generally reliable, `>100 chars = 0.85` |
| PPTX | Text per slide | `>200/slide = 0.85`, boost for notes |
| Tesseract | 0-100 | Divide by 100, penalize short results |
| Doctr | 0.0-1.0 | Use as-is, slight boost for non-empty |

### 4.4 OCR Cascade Priority

**IMPORTANT: Docling ALWAYS runs first for layout analysis.**

The OCR cascade is invoked ONLY when Docling's initial extraction has low confidence:

```
Layer 1: Docling Result (already extracted during layout analysis)
    ↓ if confidence < threshold
Layer 2: Tesseract 5.x + Preprocessing (fast fallback)
    ↓ if confidence < threshold  
Layer 3: Doctr (transformer-based, final fallback)
```

**Context:** The pinned Docling release is the primary layout analyzer, not just an OCR engine. It performs both layout detection AND text extraction. The cascade starts by evaluating Docling's existing result, then falls back to Tesseract/Doctr only if needed.

**Performance Benefit:** For digital PDFs (90% of documents), Docling extracts text directly from embedded fonts at ~100ms/page. Running Tesseract first would waste ~2500ms/page on unnecessary OCR.

---

## 5. Format Engines

### 5.1 Engine Contract

```python
from abc import ABC, abstractmethod
from pathlib import Path

class FormatEngine(ABC):
    """Abstract base class for format-specific extraction engines."""
    
    @abstractmethod
    def detect(self, file_path: Path) -> bool:
        """Check if this engine can handle the file."""
        pass
    
    @abstractmethod
    def convert(self, file_path: Path) -> UniversalDocument:
        """Convert file to Universal Intermediate Representation."""
        pass
    
    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """List of supported file extensions."""
        pass
```

### 5.2 PDF Engine (Priority: HIGH)

```python
class PDFEngine(FormatEngine):
    """
    PDF extraction using the pinned Docling release.
    
    Key Features:
    - Per-page classification (digital vs scanned)
    - Element-level confidence scoring
    - Layout-aware region detection
    """
    
    def convert(self, file_path: Path) -> UniversalDocument:
        # 1. Run Docling conversion
        # 2. For each page:
        #    - Classify as digital/scanned based on text density
        #    - Extract elements with bboxes
        #    - Store raw page image for OCR fallback
        # 3. Return UniversalDocument
```

### 5.2.1 PDF Layout Detection Strategy

**Primary Method: Docling Native Layout Model**
```python
docling_result = converter.convert(pdf_path)
for page in docling_result.pages:
    for element in page.elements:
        if element.label in ["paragraph", "heading", "text"]:
            yield Element(type=ElementType.TEXT, ...)
        elif element.label in ["figure", "picture", "image"]:
            yield Element(type=ElementType.IMAGE, ...)
```

**Fallback for Scanned Pages (Docling returns 0 elements):**
```python
if len(page.elements) == 0:
    # Full page is scanned - treat as single TEXT element
    page_image = render_page(pdf_path, page_num)
    yield Element(
        type=ElementType.TEXT,
        content="",  # Empty, will be OCR'd
        bbox=[0, 0, 1000, 1000],
        confidence=0.1,  # Low confidence triggers OCR
        raw_image=page_image
    )
```

### 5.3 ePub Engine (Priority: MEDIUM)

```python
class EpubEngine(FormatEngine):
    """
    ePub extraction using EbookLib + BeautifulSoup.
    
    Key Features:
    - Spine order processing (REQ-EPUB-01)
    - Image path resolution (REQ-EPUB-03)
    - HTML heading → hierarchy mapping
    """
```

### 5.4 HTML Engine (Priority: MEDIUM)

```python
class HTMLEngine(FormatEngine):
    """
    HTML extraction using Trafilatura.
    
    Key Features:
    - Editorial content extraction (REQ-HTML-01)
    - Navigation/footer removal
    - Heading hierarchy mapping (REQ-HTML-02)
    """
```

---

## 6. Integration Points

### 6.1 Core Components in Use

| Module | Status | Integration |
|--------|--------|-------------|
| `src/mmrag_v2/ocr/enhanced_ocr_engine.py` | ✅ Keep | Called by ElementProcessor for low-confidence TEXT |
| `src/mmrag_v2/ocr/image_preprocessor.py` | ✅ Keep | Called before OCR cascade |
| `src/mmrag_v2/vision/vision_manager.py` | ✅ Keep | Called by ElementProcessor for IMAGE elements |
| `src/mmrag_v2/schema/ingestion_schema.py` | ✅ Keep | Output schema unchanged |
| `src/mmrag_v2/state/context_state.py` | ✅ Keep | Breadcrumb tracking unchanged |
| `src/mmrag_v2/state/magazine_section_detector.py` | ✅ Keep | Magazine breadcrumb enrichment during text processing |
| `src/mmrag_v2/validators/quality_filter_tracker.py` | ✅ Keep | Feeds filtered-token analytics into token validation |

### 6.2 Pipeline-Critical Components

| Module | Change |
|--------|--------|
| `src/mmrag_v2/batch_processor.py` | Route through FormatRouter → UIR → ElementProcessor |
| `src/mmrag_v2/ocr/layout_aware_processor.py` | Proper layout detection (not full-page fallback) |
| `src/mmrag_v2/vision/vision_prompts.py` | Defines VISUAL_ONLY_PROMPT policy and response validation helpers |
| `src/mmrag_v2/refiner.py` | Optional post-OCR text repair layer with edit-budget guardrails |

### 6.3 UIR Components

| Module | Purpose |
|--------|---------|
| `src/mmrag_v2/universal/intermediate.py` | UIR data structures |
| `src/mmrag_v2/universal/router.py` | Format detection and engine routing |
| `src/mmrag_v2/universal/quality_classifier.py` | Quality classifiers and `ConfidenceNormalizer` for cross-format confidence normalization |
| `src/mmrag_v2/universal/element_processor.py` | Quality-based element processing |
| `src/mmrag_v2/engines/base.py` | FormatEngine ABC |
| `src/mmrag_v2/engines/pdf_engine.py` | PDF → UIR conversion |
| `src/mmrag_v2/engines/epub_engine.py` | Optional ePub engine plugin (not present in this branch) |
| `src/mmrag_v2/engines/html_engine.py` | Optional HTML engine plugin (not present in this branch) |
| `src/mmrag_v2/processor.py` | Current non-PDF extraction path (epub/html/docx/pptx/xlsx) |

### 6.4 Orchestration & Profile Intelligence

| Module | Purpose |
|--------|---------|
| `src/mmrag_v2/orchestration/smart_config.py` | Fast pre-analysis (text/image density, median image sizes) to build DocumentProfile |
| `src/mmrag_v2/orchestration/document_diagnostic.py` | Pre-flight modality/domain diagnostics and confidence profiling |
| `src/mmrag_v2/orchestration/profile_classifier.py` | Weighted profile selection (e.g., academic_whitepaper, digital_magazine) |
| `src/mmrag_v2/orchestration/strategy_profiles.py` | Profile parameter sets and adaptive profile behavior |
| `src/mmrag_v2/orchestration/strategy_orchestrator.py` | Converts selected profile into concrete extraction strategy knobs |

### 6.5 Utility & QA Extensions

| Module | Purpose |
|--------|---------|
| `src/mmrag_v2/utils/advanced_spatial_propagator.py` | Extracts Docling provenance bbox and normalizes for all modalities |
| `src/mmrag_v2/utils/image_trim.py` | Margin trim and anti-clipping crop expansion utilities for visual assets |
| `src/mmrag_v2/utils/image_quality.py` | Lightweight blur metrics used for CLI/QA diagnostics |
| `src/mmrag_v2/validators/quality_filter_tracker.py` | Tracks token impact of quality filters by category |

### 6.6 Adjacent Modules (Not on Default Ingestion Path)

| Module | Purpose |
|--------|---------|
| `src/mmrag_v2/adapters/vision_providers.py` | Alternate pluggable VLM provider abstraction layer |
| `src/mmrag_v2/rag/advanced_pipeline.py` | Downstream multimodal retrieval/indexing pipeline over `ingestion.jsonl` |

---

## 7. VLM Prompt Engineering

### 7.1 The Problem

Current VLM prompts allow the model to read text in images, producing:
```
"The image shows a page titled 'INTRODUCTION' with text describing..."
```

### 7.2 The Solution: Visual-Only Prompt

```python
VLM_VISUAL_ONLY_PROMPT = """
You are analyzing an IMAGE from a technical document.

STRICT CONSTRAINTS:
1. DO NOT read or transcribe any text visible in the image
2. ONLY describe visual elements: shapes, diagrams, photos, components
3. Focus on spatial relationships and technical details
4. Use technical vocabulary appropriate to the subject matter

CORRECT OUTPUT EXAMPLES:
- "Exploded view diagram showing trigger assembly with 7 components arranged vertically"
- "Black and white photograph of military aircraft on carrier deck"
- "Technical schematic showing electrical connections between 4 modules"

FORBIDDEN OUTPUT (text reading):
- "The text says 'INTRODUCTION'..."
- "The caption reads..."
- "The label indicates..."

Describe ONLY the visual/mechanical aspects of this image:
"""
```

### 7.3 Enforcement

The `ElementProcessor`:
1. Send IMAGE elements to VLM with VISUAL_ONLY_PROMPT
2. Post-process response to detect/reject text transcription
3. Retry with stricter prompt if text detected

---

## 8. Memory Management

### 8.1 Batch Processing Strategy

```
┌─────────────────────────────────────────────────────────┐
│                  LARGE DOCUMENT (244 pages)             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │
│  │Batch 1  │  │Batch 2  │  │Batch 3  │  │ ...     │     │
│  │ p1-10   │  │ p11-20  │  │ p21-30  │  │         │     │
│  └────┬────┘  └────┬────┘  └────┬────┘  └─────────┘     │
│       │            │            │                       │
│       ▼            ▼            ▼                       │
│  ┌──────────────────────────────────────────────┐       │
│  │  Process batch → gc.collect() → Next batch   │       │
│  │  (REQ-PDF-05: Memory hygiene)                │       │
│  └──────────────────────────────────────────────┘       │
│                                                         │
│  Memory Budget: < 8GB (16GB system - 8GB for OS/other)  │
│  Batch Size: 10 pages (configurable)                    │
│  GC: Explicit gc.collect() between batches              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 8.1.1 Domain Detection Parity

To guarantee `process` and `batch` parity, domain detection is **content-first**:
- Content features (text density, image coverage, table presence) carry the primary weight.
- Filename keywords are a weak hint only, so renames do not flip `profile_type`.
- This prevents batch-renamed files (e.g., `doc1.pdf`) from diverging in classification.

### 8.2 Image Buffer Management

- Page images stored only for current batch
- OCR results cached per-page, cleared after processing
- VLM cache persists across batches (hash-based deduplication)

---

## 9. Error Handling

### 9.1 Per-Element Resilience

```python
def process_element(element: Element) -> Optional[IngestionChunk]:
    try:
        if element.type == ElementType.TEXT:
            return process_text_element(element)
        elif element.type == ElementType.IMAGE:
            return process_image_element(element)
        elif element.type == ElementType.TABLE:
            return process_table_element(element)
    except Exception as e:
        logger.error(f"Element processing failed: {e}")
        return create_fallback_chunk(element)  # Never crash
```

### 9.2 Per-File Resilience (REQ-ERR-01)

```python
def process_batch(files: List[Path]) -> List[BatchResult]:
    results = []
    for file in files:
        try:
            result = process_file(file)
            results.append(result)
        except Exception as e:
            logger.error(f"File {file.name} failed: {e}")
            # Continue with next file - single corrupt file doesn't crash batch
    return results
```

---

## 10. Testing Strategy

### 10.1 Test Pyramid

```
                    ┌─────────────────┐
                    │  E2E Tests      │
                    │  (5 documents)  │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │      Integration Tests      │
              │  (UIR → Processor → Output) │
              └──────────────┬──────────────┘
                             │
       ┌─────────────────────┴─────────────────────┐
       │              Unit Tests                   │
       │  (Each engine, OCR cascade, VLM routing)  │
       └───────────────────────────────────────────┘
```

### 10.2 Required Test Cases

| Test | Input | Expected Output |
|------|-------|-----------------|
| Digital PDF | Clean digital PDF | TEXT chunks with high confidence |
| Scanned PDF | Firearms.pdf | TEXT chunks from OCR (not VLM summaries) |
| Image extraction | PDF with figures | IMAGE chunks with visual descriptions |
| Modality check | Any input | No "shadow" modality in output |
| Quality routing | Mixed document | High-quality → direct, low → OCR |
| Memory test | 244-page PDF | Peak memory < 8GB |

### 10.3 Regression Tests

```python
def test_digital_pdf_still_works():
    """Ensure digital PDFs aren't broken by new architecture."""
    result = process_file("test_data/digital_clean.pdf")
    assert all(chunk.modality in ["text", "image", "table"] for chunk in result)
    assert any(chunk.modality == "text" for chunk in result)
    
def test_scanned_pdf_produces_text():
    """The core fix: scanned PDFs produce TEXT, not VLM summaries."""
    result = process_file("test_data/Firearms.pdf")
    text_chunks = [c for c in result if c.modality == "text"]
    assert len(text_chunks) > 0
    # Check actual OCR text, not VLM description
    assert not any("The image shows" in c.content for c in text_chunks)
```

---

## 11. Compatibility Notes

### 11.1 Backward Compatibility

- Old CLI flags continue to work
- Output schema unchanged

---

## 12. Appendix: File Structure (Current Branch)

This snapshot reflects files currently present under `src/mmrag_v2/` in this branch.

```
src/mmrag_v2/
├── __init__.py
├── batch_processor.py              # Batch orchestration pipeline
├── cli.py                          # Command-line entrypoints
├── mapper.py                       # DoclingDocument -> IngestionChunk mapper
├── processor.py                    # Core document processor
├── refiner.py                      # OCR text refinement layer (optional)
├── version.py                      # Central schema/engine version constants
│
├── adapters/                       # Pluggable provider adapters
│   ├── __init__.py
│   └── vision_providers.py         # Alternate VLM provider interfaces
│
├── chunking/                       # Text chunking helpers
│   └── semantic_overlap_manager.py
│
├── universal/                      # Universal Intermediate Layer
│   ├── __init__.py
│   ├── intermediate.py             # Core UIR data structures
│   ├── router.py                   # Format detection and routing
│   ├── quality_classifier.py       # Page/element quality + ConfidenceNormalizer
│   └── element_processor.py        # Quality-based element processing
│
├── engines/                        # Format-specific engines
│   ├── __init__.py
│   ├── base.py                     # FormatEngine ABC
│   └── pdf_engine.py               # PDF → UIR (Docling-based)
│
├── ocr/                            # Enhanced OCR components
│   ├── __init__.py
│   ├── enhanced_ocr_engine.py      # 3-layer cascade
│   ├── image_preprocessor.py       # Deskew, denoise, contrast
│   └── layout_aware_processor.py   # Layout-aware OCR integration
│
├── vision/                         # VLM integration
│   ├── __init__.py
│   ├── vision_manager.py           # Provider abstraction + inference
│   ├── vision_prompts.py           # VISUAL_ONLY_PROMPT policy
│   └── ocr_hint_engine.py          # OCR assistance utilities
│
├── schema/                         # Output schema
│   ├── __init__.py
│   └── ingestion_schema.py         # Ingestion chunk contracts
│
├── state/                          # Context tracking
│   ├── __init__.py
│   ├── context_state.py            # Breadcrumb/context state
│   └── magazine_section_detector.py # Magazine section header detection
│
├── orchestration/                  # Strategy + profile intelligence
│   ├── __init__.py
│   ├── document_diagnostic.py      # Pre-flight diagnostics
│   ├── profile_classifier.py       # Multi-dimensional profile selection
│   ├── smart_config.py             # Fast document profiling
│   ├── strategy_orchestrator.py    # Strategy synthesis from profile
│   └── strategy_profiles.py        # Profile parameter definitions
│
├── rag/                            # Downstream multimodal retrieval
│   ├── __init__.py
│   └── advanced_pipeline.py
│
├── utils/                          # Utilities
│   ├── __init__.py
│   ├── advanced_spatial_propagator.py
│   ├── coordinate_normalization.py
│   ├── image_hash_registry.py
│   ├── image_quality.py
│   ├── image_trim.py
│   └── pdf_splitter.py
│
└── validators/                     # QA checks
    ├── __init__.py
    ├── quality_filter_tracker.py
    └── token_validator.py
```

---

**END OF ARCHITECTURE DOCUMENT**
