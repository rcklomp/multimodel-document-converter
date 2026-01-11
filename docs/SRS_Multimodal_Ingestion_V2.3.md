# SOFTWARE REQUIREMENTS SPECIFICATION: Multimodal RAG Ingestion Engine (v2.3)

**Version:** 2.3.0 (PRODUCTION SPEC)
**Target Agent:** Cline (Python 3.10)
**Output:** JSONL Canonical Schema + Asset Directory
**Platform:** Apple Silicon (ARM64 Native)
**Status:** PRODUCTION READY
**Supersedes:** SRS v2.2.1

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 2.3.0 | 2026-01-02 | System | Surya-OCR verwijderd, coördinatensysteem gedocumenteerd, Shadow Extraction aangescherpt, hiërarchiestandaard toegevoegd |
| 2.2.1 | 2025-12-xx | System | Initial PDF spec |

---

## 1. PROJECT DEFINITION & SCOPE

The application is a high-fidelity ETL (Extract, Transform, Load) pipeline designed to convert a heterogeneous corpus of documents into an ingestion-optimized format for Multimodal RAG systems.

### 1.1 Critical Design Mandates (The "Iron Rules")

These rules are **INVIOLABLE**. Any implementation that violates these rules is considered a critical defect.

| Rule ID | Rule | Rationale |
|---------|------|-----------|
| **IRON-01** | **Atomicity:** Tables and Figures are atomic semantic units. They MUST NOT be split across chunks. | Prevents semantic fragmentation in RAG retrieval |
| **IRON-02** | **State Persistence:** The parser MUST maintain a hierarchical state (`ContextState`) that persists across internal file boundaries (e.g., ePub chapters). | Prevents context loss and "orphan" chunks |
| **IRON-03** | **Granularity Prohibition:** Full-page image exports (i.e., 'flattened' screenshots containing both text and images) are **STRICTLY PROHIBITED**. | Maintains RAG-cleanliness; text must remain parseable |
| **IRON-04** | **Denoising:** Non-editorial content (Ads, Navigation, Mastheads) MUST be identified and discarded. | Reduces noise in retrieval results |
| **IRON-05** | **Disk-First Persistence:** Data for each document MUST be written to disk immediately after conversion. Keeping multiple documents in memory is STRICTLY PROHIBITED. | Prevents OOM on 16GB systems |
| **IRON-06** | **Fail-Safe Asset Extraction:** If a document reports visual elements but the image buffer is null, processing MUST HALT and trigger a configuration audit. Silent failures are unacceptable. | Guarantees asset integrity |
| **IRON-07** | **Full-Page Guard (NEW):** Shadow-extracted assets with `area_ratio > 0.95` (covering >95% of page area) require VLM verification before inclusion. Assets confirmed as UI/navigation elements MUST be discarded. | Prevents accidental full-page captures via shadow extraction |

**Clarification on IRON-03 vs Shadow Extraction:**
A raw, full-page background photo extracted via the Shadow Extraction layer (Section 4.3) is classified as an **Atomic Asset**, NOT a prohibited export, PROVIDED:
1. The asset is a genuine editorial image (photo, illustration)
2. Text overlaying the image remains in its parseable markdown format
3. The asset passes the Full-Page Guard validation (IRON-07)

---

## 2. INPUT FORMAT SPECIFICATIONS

The engine routes input files to specific processing pipelines based on MIME type.

### 2.1 PDF (Portable Document Format)

* **Pipeline:** `Docling v2.66.0` (Layout Analysis + Structure Extraction via IBM LayoutModels)
* **Note:** Standalone Surya-OCR is **DEPRECATED** and MUST NOT be used. Docling v2.66.0 includes all required layout analysis capabilities.

| Requirement ID | Requirement | Priority |
|----------------|-------------|----------|
| **REQ-PDF-01** | **De-columnization:** Detect multi-column layouts. Reading order must follow vertical column flow, not horizontal line scan. | MUST |
| **REQ-PDF-02** | **Ad Detection:** Blocks identified as "Advertisement" by layout model or keyword/link-density analysis MUST be excluded. | MUST |
| **REQ-PDF-03** | **Hybrid OCR:** If text extraction confidence < 90% (scanned PDF), trigger OCR on specific bounding boxes via Docling's internal OCR or Tesseract fallback. | SHOULD |
| **REQ-PDF-04** | **Mandatory Rendering:** DocumentConverter MUST be initialized with `PdfPipelineOptions(do_extract_images=True)`. Minimum render scale: 2.0. | MUST |
| **REQ-PDF-05** | **Memory Hygiene:** Between document cycles, explicitly trigger `gc.collect()` to prevent RAM saturation on 16GB systems. | MUST |

### 2.2 EPUB (Electronic Publication)

* **Pipeline:** `EbookLib` + `BeautifulSoup4`

| Requirement ID | Requirement | Priority |
|----------------|-------------|----------|
| **REQ-EPUB-01** | Process content strictly in `content.opf` spine order. | MUST |
| **REQ-EPUB-02** | Remove all internal filenames, CSS classes, and hidden HTML comments before extraction. | MUST |
| **REQ-EPUB-03** | Resolve relative image paths in `src` attributes to absolute paths within the ePub container. | MUST |

### 2.3 HTML (Web Content)

* **Pipeline:** `Trafilatura` (Primary) + `BeautifulSoup4` (Fallback)

| Requirement ID | Requirement | Priority |
|----------------|-------------|----------|
| **REQ-HTML-01** | Extract ONLY `<body>` editorial content. Discard `<nav>`, `<footer>`, `<aside>`, `<script>` tags. | MUST |
| **REQ-HTML-02** | Map `<h1>` through `<h6>` tags directly to `ContextState` hierarchy. | MUST |

### 2.4 Microsoft Office (DOCX, PPTX, XLSX)

* **Pipeline:** `Docling` (Primary)

| Requirement ID | Requirement | Priority |
|----------------|-------------|----------|
| **REQ-DOCX-01** | Map XML Styles (`Heading 1`...) to metadata hierarchy. | MUST |
| **REQ-PPTX-01** | Treat Slides as Pages. Slide Title = `Heading 1`. Group overlapping shapes into single `asset` image. | MUST |
| **REQ-XLSX-01** | Convert active worksheets to **GitHub Flavored Markdown** tables. Prune empty rows/columns. | MUST |

---

## 3. SYSTEM ARCHITECTURE: THE STATE MACHINE

To prevent "Context Bleeding" (e.g., Preface inheriting Chapter 10's title), the system implements a persistent State Machine.

### 3.1 The ContextState Object

```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ContextState:
    """Persistent state object maintained during document parsing."""
    current_page: int = 0
    breadcrumbs: List[str] = field(default_factory=list)
    current_header_level: int = 0
    last_processed_header: Optional[str] = None
    document_type: str = "generic"  # "generic" | "periodical" | "technical"
```

### 3.2 The Sensitivity Dial: Dynamic Autonomy Scale

The `--sensitivity` parameter (range: 0.1–1.0, default: 0.5) controls the balance between AI-driven layout analysis (Docling) and deterministic heuristic extraction (PyMuPDF shadow scan).

| Sensitivity | Heuristic Aggression | Docling Trust | Use Case |
|-------------|---------------------|---------------|----------|
| **0.1** (Strict) | None (Docling only) | 100% | Technical documents with clean layouts |
| **0.5** (Balanced) | Medium (Bitmaps ≥50% page area) | 50% | General purpose. **RECOMMENDED DEFAULT** |
| **1.0** (Max Recall) | Maximum (All bitmaps ≥100px) | 0% | Visually-dense editorial content |

**Implementation Requirements:**

| Requirement ID | Requirement |
|----------------|-------------|
| **REQ-SENS-01** | Sensitivity scales minimum dimension threshold linearly: `min_dimension = 400px - (sensitivity * 300px)` |
| **REQ-SENS-02** | At sensitivity ≥0.7, enable background layer extraction for full-page editorial images with text overlays. |
| **REQ-SENS-03** | Editorial documents (image ratio ≥40%, avg size ≥300px) receive automatic -50px threshold reduction. |

### 3.3 State Transition Logic

| Requirement ID | Requirement |
|----------------|-------------|
| **REQ-STATE-01** | `breadcrumbs` list UPDATES ONLY when a new header is explicitly detected. |
| **REQ-STATE-02** | If new header is `Level N`, it replaces existing `Level N` and removes all deeper levels (N+1, N+2, etc.). |
| **REQ-STATE-03** | Every generated chunk MUST inherit a **deep copy** of the current `ContextState`. |

### 3.4 Hierarchy Standard for Periodical Publications (NEW)

For periodical publications (magazines, journals, newsletters), the following hierarchy depth standard MUST be enforced:

```
Level 1: Publication Title (e.g., "Combat Aircraft Journal")
Level 2: Edition/Issue (e.g., "August 2025", "Vol. 42 No. 8")
Level 3: Section (e.g., "IN THE NEWS", "Features", "Tech Focus")
Level 4: Article Title (e.g., "Operation Rising Lion")
Level 5: Article Subsection (e.g., "First Wave", "Second Stage")
```

| Requirement ID | Requirement |
|----------------|-------------|
| **REQ-HIER-01** | Documents classified as `periodical` MUST populate at minimum Levels 1-3 in breadcrumbs. |
| **REQ-HIER-02** | Article boundaries (Level 4) SHOULD be detected via page breaks, horizontal rules, or large heading style changes. |
| **REQ-HIER-03** | If hierarchy cannot be determined, breadcrumbs MUST contain at minimum: `[source_filename, page_number]`. |
| **REQ-HIER-04** | Breadcrumb depth MUST match `hierarchy.level` value in output schema (e.g., Level 3 = 3 items in breadcrumb_path). |

---

## 4. MULTIMODAL ASSET EXTRACTION (REQ-MM)

### 4.1 Visual Assets (Images/Charts)

| Requirement ID | Requirement |
|----------------|-------------|
| **REQ-MM-01** | **Element Cropping:** Detect bounding box, apply **10px padding**, crop and save as PNG. |
| **REQ-MM-02** | **Naming Standard:** `[DocHash]_[PageNum]_[Type]_[Index].png` (e.g., `a1b2c3d4_005_figure_01.png`) |
| **REQ-MM-03** | **Contextual Anchoring:** JSONL entry MUST contain `text_before` (300 chars) and `text_after` (300 chars). |

### 4.2 Tabular Data

| Requirement ID | Requirement |
|----------------|-------------|
| **REQ-MM-04** | Tables MUST be converted to **Markdown** (default) or **HTML Table** (if cells merged/complex). |
| **REQ-MM-04b** | Tables are NEVER flattened to unstructured text. |

### 4.3 Visual Heuristics & Shadow Extraction

**Context:** Docling's AI-driven layout analyzer can miss large editorial images when they function as background layers with text overlays. This section mandates a deterministic safety net.

| Requirement ID | Requirement |
|----------------|-------------|
| **REQ-MM-05** | **Shadow Extraction:** Parallel to AI analysis, perform raw physical scan of PDF stream (via PyMuPDF) to identify embedded bitmap/image objects. |
| **REQ-MM-06** | **Conflict Resolution:** If bitmap detected with dimensions ≥300×300px (or ≥40% page area) lacking corresponding Docling `Figure`/`Picture` block, force-extract as "Shadow Asset." |
| **REQ-MM-07** | **Asset Pairing:** Shadow Assets MUST link to nearest `TextBlock` on current page. Metadata MUST include `extraction_method: "shadow"`. |

### 4.4 Full-Page Guard (NEW)

This section implements IRON-07 to prevent accidental full-page captures.

| Requirement ID | Requirement |
|----------------|-------------|
| **REQ-MM-08** | **Area Ratio Calculation:** For every shadow asset, calculate `area_ratio = (asset_width × asset_height) / (page_width × page_height)`. |
| **REQ-MM-09** | **Full-Page Threshold:** If `area_ratio > 0.95`, the asset triggers Full-Page Guard validation. |
| **REQ-MM-10** | **VLM Verification:** Full-Page Guard assets MUST be verified by VLM to confirm editorial nature. VLM prompt: "Is this image editorial content (photo, illustration, infographic) or is it a UI element/page scan/navigation?" |
| **REQ-MM-11** | **Rejection Criteria:** Assets identified as UI elements, full-page scans, or navigation MUST be discarded. Log rejection reason. |
| **REQ-MM-12** | **Override Flag:** CLI flag `--allow-fullpage-shadow` bypasses Full-Page Guard for known editorial-heavy documents. Use with caution. |

---

## 5. VISION & VLM REQUIREMENTS (REQ-VLM)

### 5.1 Vision Provider Integration

| Requirement ID | Requirement |
|----------------|-------------|
| **REQ-VLM-01** | Support multiple VLM providers: **Local:** Ollama, **Cloud:** OpenAI (gpt-4o-mini), Anthropic (claude-3-5-haiku), **Fallback:** Breadcrumb-based context (no VLM). |

### 5.2 Low-Recall Trigger & Full-Page Preview

| Requirement ID | Requirement |
|----------------|-------------|
| **REQ-VLM-02** | For `editorial` documents, trigger Full-Page Preview VLM analysis if: (a) detected assets < page-median, OR (b) total assets on page = ZERO. |
| **REQ-VLM-03** | Maintain global `vision_cache.json` to prevent redundant VLM calls for identical image hashes. |
| **REQ-VLM-04** | VLM calls have configurable timeout (default: 90s). On timeout/failure, fall back to breadcrumb-based context. |

---

## 6. CANONICAL OUTPUT SCHEMA (JSONL)

Every line in `ingestion.jsonl` MUST validate against this schema. **No other output formats are permitted.**

### 6.1 Schema Definition

```json
{
  "chunk_id": "string (UUID_v4 or composite hash)",
  "doc_id": "string (12-char hex from file SHA256)",
  "modality": "text | image | table",
  "content": "string (actual text, markdown, or VLM description)",
  "metadata": {
    "source_file": "string (original filename)",
    "file_type": "string (pdf|epub|html|docx|pptx|xlsx)",
    "page_number": "integer (1-indexed)",
    "chunk_type": "string|null (paragraph|heading|list|caption|null)",
    "hierarchy": {
      "parent_heading": "string|null",
      "breadcrumb_path": ["string"],
      "level": "integer|null (1-5)"
    },
    "spatial": {
      "bbox": "[int, int, int, int] (REQUIRED for images, see 6.2)",
      "page_width": "integer|null",
      "page_height": "integer|null"
    },
    "extraction_method": "string (docling|shadow|ocr)",
    "content_classification": "string|null (editorial|technical|advertisement)",
    "ocr_confidence": "string|null (high|medium|low)",
    "created_at": "string (ISO 8601 timestamp)"
  },
  "asset_ref": {
    "file_path": "string (relative path to asset)",
    "mime_type": "string (image/png)",
    "width_px": "integer|null",
    "height_px": "integer|null",
    "file_size_bytes": "integer|null"
  },
  "semantic_context": {
    "prev_text_snippet": "string|null (max 300 chars)",
    "next_text_snippet": "string|null (max 300 chars)"
  },
  "schema_version": "string (2.3.0)"
}
```

### 6.2 Coordinate System Specification (REQ-COORD-01) — MANDATORY

**This section defines the ONLY permitted coordinate system for bounding boxes.**

| Requirement ID | Requirement |
|----------------|-------------|
| **REQ-COORD-01** | **Normalization Base:** ALL bounding boxes MUST be normalized to a **1000×1000 integer canvas**. |
| **REQ-COORD-02** | **Data Type:** `bbox` values MUST be `List[int]` with exactly 4 elements. |
| **REQ-COORD-03** | **Format:** `[x_min, y_min, x_max, y_max]` where all values are integers in range `[0, 1000]`. |
| **REQ-COORD-04** | **PROHIBITED Formats:** Floats (0.0-1.0), raw pixel values, percentages, or any other coordinate system. |
| **REQ-COORD-05** | **Conversion Formula:** `normalized_coord = int((raw_coord / page_dimension) * 1000)` |

**Examples:**

```python
# CORRECT ✓
"bbox": [100, 200, 500, 600]   # Integer values 0-1000
"bbox": [0, 0, 1000, 1000]     # Full page (requires Full-Page Guard)

# INCORRECT ✗
"bbox": [0.1, 0.2, 0.5, 0.6]   # Floats PROHIBITED
"bbox": [72, 144, 360, 432]    # Raw pixels PROHIBITED
"bbox": ["10%", "20%", "50%", "60%"]  # Strings PROHIBITED
```

**Validation Rule:** Any chunk with `modality: "image"` or `modality: "table"` MUST have a valid `spatial.bbox` field conforming to REQ-COORD-01 through REQ-COORD-05.

### 6.3 Required vs Optional Fields

| Field | Required | Condition |
|-------|----------|-----------|
| `chunk_id` | **REQUIRED** | Always |
| `doc_id` | **REQUIRED** | Always |
| `modality` | **REQUIRED** | Always |
| `content` | **REQUIRED** | Always (non-empty string) |
| `metadata.source_file` | **REQUIRED** | Always |
| `metadata.file_type` | **REQUIRED** | Always |
| `metadata.page_number` | **REQUIRED** | Always |
| `metadata.extraction_method` | **REQUIRED** | Always |
| `metadata.created_at` | **REQUIRED** | Always |
| `metadata.spatial.bbox` | **REQUIRED** | When `modality` is `image` or `table` |
| `metadata.hierarchy.breadcrumb_path` | **REQUIRED** | Always (may be empty list) |
| `asset_ref` | **REQUIRED** | When `modality` is `image` or `table` |
| `asset_ref.file_path` | **REQUIRED** | When `asset_ref` is present |
| `schema_version` | **REQUIRED** | Always |
| All other fields | OPTIONAL | May be null |

---

## 7. CHUNKING STRATEGY & LOGIC (REQ-CHUNK)

### 7.1 Semantic Partitioning

| Requirement ID | Requirement |
|----------------|-------------|
| **REQ-CHUNK-01** | Splits MUST ONLY occur at sentence delimiters (`.`, `!`, `?`, `\n`). Mid-sentence splits are fatal errors. |
| **REQ-CHUNK-02** | **Token Limits:** Text: target 400, hard max 512. Atomic (Tables/Figures): max 1024 (split with header repetition if exceeded). |

### 7.2 Dynamic Semantic Overlap (DSO)

| Requirement ID | Requirement |
|----------------|-------------|
| **REQ-CHUNK-03** | **Trigger:** `--semantic-overlap` flag. **Model:** `sentence-transformers/all-MiniLM-L6-v2` (v2.2.2). |
| **REQ-CHUNK-04** | **Algorithm:** Extract last 3 sentences of Chunk A, first 3 of Chunk B. Compute cosine similarity. If `sim > 0.85`: `overlap = base_overlap * 1.5`. |
| **REQ-CHUNK-05** | **Constraint:** Overlap < 25% of total chunk size. |

### 7.3 Output Format

| Requirement ID | Requirement |
|----------------|-------------|
| **REQ-OUT-01** | Primary output MUST be `.jsonl` (JSON Lines), not single `.json` array. Supports streaming and memory efficiency. |

---

## 8. SCANNED & HYBRID DOCUMENT HANDLING (REQ-PATH)

### 8.1 Detection & Pathing

| Requirement ID | Requirement |
|----------------|-------------|
| **REQ-PATH-01** | **Entropy Check:** Sample pages 1, 25%, 50%, 75%, 100% of document on initialization. |
| **REQ-PATH-02** | **Classification:** Path A (Native): text-to-image ratio >0.6. Path B (Scanned): ratio <0.2. Path C (Hybrid): ratio 0.2–0.6. |

### 8.2 Scanned Document Pipeline (Path B)

| Requirement ID | Requirement |
|----------------|-------------|
| **REQ-PATH-03** | Render each page at minimum 300 DPI using PyMuPDF. |
| **REQ-PATH-04** | OCR Engine Priority: (1) Docling internal OCR, (2) Tesseract 5.x, (3) EasyOCR, (4) OCRmac (Apple Silicon). |
| **REQ-PATH-05** | After OCR, segment page into logical blocks (paragraphs, lists, tables) using bounding box data. |
| **REQ-PATH-06** | OCR results with confidence <70% MUST be flagged with `metadata.ocr_confidence: "low"`. |

### 8.3 Hybrid Document Strategy (Path C)

| Requirement ID | Requirement |
|----------------|-------------|
| **REQ-PATH-07** | **Per-Page Routing:** >100 chars parseable → Docling. <50 chars → OCR. 50-100 chars → Both, merge results. |
| **REQ-PATH-08** | Breadcrumb state (Section 3.1) MUST persist across path transitions. |

---

## 9. QUALITY ASSURANCE & LOGGING

### 9.1 Automated Checks (Post-Processing)

| Check ID | Validation Rule |
|----------|-----------------|
| **QA-CHECK-01** | Verify `sum(chunk_tokens) ~= total_document_tokens` (tolerance 10%). |
| **QA-CHECK-02** | Verify every `asset_ref.file_path` exists on disk. |
| **QA-CHECK-03** | Verify `breadcrumb_path` depth matches `hierarchy.level` value. |
| **QA-CHECK-04** | Verify all `bbox` values conform to REQ-COORD-01 (integers 0-1000). |
| **QA-CHECK-05** | Verify no chunks have `modality: "image"` with missing `asset_ref`. |

### 9.2 Engineering Imperatives

**Dependency Pinning (EXACT VERSIONS):**

```toml
# pyproject.toml - AUTHORITATIVE SOURCE
docling==2.66.0                 # EXACT - Core layout engine with IBM models
docling-core>=2.0.0             # Minimum version
sentence-transformers>=3.0.0    # Chunking embeddings
pymupdf>=1.24.0                 # PDF rendering & shadow extraction
pytesseract>=0.3.10             # OCR fallback
numpy>=1.24.4,<2.0.0            # NumPy 1.x shield (Docling/PyTorch compatibility)
```

**DEPRECATED Dependencies (DO NOT USE):**
- ❌ `surya-ocr` — Replaced by Docling v2.66.0 internal models
- ❌ `paddleocr` — Not compatible with Apple Silicon
- ❌ `numpy>=2.0.0` — Breaks Docling/PyTorch compatibility

**Error Handling:**

| Requirement ID | Requirement |
|----------------|-------------|
| **REQ-ERR-01** | Use `try-except` block per file. Single corrupt file MUST NOT crash batch. |
| **REQ-ERR-02** | Log all errors to `ingestion_errors.log` with timestamp, filename, and stack trace. |
| **REQ-ERR-03** | On startup, log: `"Using Docling v2.66.0"` to confirm correct engine. |

**Platform Optimization:**

| Requirement ID | Requirement |
|----------------|-------------|
| **REQ-PLAT-01** | Application MUST be optimized for Apple Silicon (ARM64). |
| **REQ-PLAT-02** | PyTorch MUST use MPS backend. Verify with `torch.backends.mps.is_available() == True`. |
| **REQ-PLAT-03** | Local Conda environment (`./env`) MUST be used. No `.venv` or system Python. |

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Shadow Asset** | A visual element detected via raw PDF stream scan (PyMuPDF) that was missed by Docling's AI analysis |
| **Full-Page Guard** | Validation mechanism preventing accidental capture of full-page UI/navigation elements |
| **Iron Rules** | Inviolable design mandates (Section 1.1) |
| **Breadcrumbs** | Hierarchical path representing chunk's position in document structure |
| **Area Ratio** | Ratio of asset area to total page area, used in Full-Page Guard |

---

## Appendix B: Change Log from v2.2.1

| Section | Change Type | Description |
|---------|-------------|-------------|
| 1.1 | ADDED | IRON-07 (Full-Page Guard) |
| 2.1 | MODIFIED | Removed Surya-OCR reference, clarified Docling-only pipeline |
| 3.4 | ADDED | Hierarchy Standard for Periodical Publications |
| 4.4 | ADDED | Full-Page Guard implementation (REQ-MM-08 through REQ-MM-12) |
| 6.2 | ADDED | Coordinate System Specification (REQ-COORD-01 through REQ-COORD-05) |
| 6.3 | ADDED | Required vs Optional Fields table |
| 9.1 | ADDED | QA-CHECK-04, QA-CHECK-05 |
| 9.2 | MODIFIED | Removed `surya-ocr>=0.4.0` from dependencies, added deprecation notice |
| ALL | MODIFIED | Corrected section numbering (removed duplicates) |
| ALL | MODIFIED | Converted requirements to table format for clarity |

---

**END OF DOCUMENT**
