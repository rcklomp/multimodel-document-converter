# SOFTWARE REQUIREMENTS SPECIFICATION: Multimodal RAG Ingestion Engine (v2.0)

**Version:** 2.0.0 (FINAL)
**Target Agent:** Cline (Python 3.10)
**Output:** JSONL Canonical Schema + Asset Directory
**Platform:** Apple Sillicon machines

---

## 1. PROJECT DEFINITION & SCOPE

The application is a high-fidelity ETL (Extract, Transform, Load) pipeline designed to convert a heterogeneous corpus of documents into an ingestion-optimized format for Multimodal RAG systems.

### 1.1 Critical Design Mandates (The "Iron Rules")

1. **Atomicity:** Tables and Figures are atomic semantic units. They MUST NOT be split across chunks.
2. **State Persistence:** The parser MUST maintain a hierarchical state (`ContextState`) that persists across internal file boundaries (e.g., ePub chapters) to prevent context loss.
3. **Granularity:** Full-page image exports are **STRICTLY PROHIBITED**. Only individual, cropped visual elements (figures, charts, photos) may be saved.
4. **Denoising:** Non-editorial content (Ads, Navigation, Mastheads) MUST be identified and discarded.
5. **Disk-First Persistence:** Processing MUST follow a "Stream-to-Disk" pattern. Data for each document (JSONL + Assets) MUST be written to the final storage directory immediately after that document's conversion is complete. Keeping multiple documents in memory before saving is STRICTLY PROHIBITED.
6. **Fail-Safe Asset Extraction:** If a document reports visual elements (figures/tables) but the image buffer is null, the process MUST stop and trigger a configuration audit. Silent failures for asset extraction are unacceptable.

---

## 2. INPUT FORMAT SPECIFICATIONS

The engine must route input files to specific processing pipelines based on MIME type.

### 2.1 PDF (Portable Document Format)

* **Pipeline:** `Surya` (Layout Analysis) + `Docling` (Structure).
* **REQ-PDF-01 (De-columnization):** The engine must detect multi-column layouts. Text reading order must follow the vertical column flow, not the horizontal line scan.
* **REQ-PDF-02 (Ad Detection):** Blocks identified as "Advertisement" by the layout model or via keyword density/link-density analysis MUST be excluded from the text stream.
* **REQ-PDF-03 (Hybrid OCR):** If text extraction confidence < 90% (scanned PDF), trigger `PaddleOCR` or `Tesseract` on specific bounding boxes.
* **REQ-PDF-04 (Mandatory Rendering):** The DocumentConverter MUST be initialized with PdfPipelineOptions(do_extract_images=True). High-fidelity rendering (min scale 2.0) is mandatory to ensure pixel data is captured in the doc.pictures objects.
* **REQ-PDF-05 (Memory Hygiene):** Between document processing cycles, the engine MUST explicitly trigger Python's garbage collector (gc.collect()) to prevent RAM saturation on 16GB systems.

### 2.2 EPUB (Electronic Publication)

* **Pipeline:** `EbookLib` + `BeautifulSoup4`.
* **REQ-EPUB-01 (Spine Traversal):** Process content strictly in the order defined in the `content.opf` spine.
* **REQ-EPUB-02 (Artifact Stripping):** Remove all internal filenames (e.g., `OEBPS/part1.xhtml`), CSS classes, and hidden HTML comments before extraction.
* **REQ-EPUB-03 (Image Resolution):** Resolve relative image paths in `src` attributes to the absolute path within the ePub container for extraction.

### 2.3 HTML (Web Content)

* **Pipeline:** `Trafilatura` (Primary) + `BeautifulSoup4` (Fallback).
* **REQ-HTML-01 (Main Content Extraction):** Extract ONLY the `<body>` editorial content. Discard `<nav>`, `<footer>`, `<aside>`, and `<script>` tags.
* **REQ-HTML-02 (DOM Hierarchy):** Map `<h1>` through `<h6>` tags directly to the `ContextState` hierarchy.

### 2.4 Microsoft Office (DOCX, PPTX, XLSX)

* **Pipeline:** `Docling` (Primary).
* **REQ-DOCX-01:** Map XML Styles (`Heading 1`...) to metadata hierarchy.
* **REQ-PPTX-01:** Treat Slides as Pages. Slide Title = `Heading 1`. Group overlapping shapes into a single `asset` image.
* **REQ-XLSX-01:** Convert active worksheets to **GitHub Flavored Markdown** tables. Prune empty rows/columns to minimize token usage.

---

## 3. SYSTEM ARCHITECTURE: THE STATE MACHINE

To prevent "Context Bleeding" (e.g., Preface getting Chapter 10's title), the system MUST implement a persistent State Machine.

### 3.1 The ContextState Object

The application must maintain a global object during parsing:

```python
class ContextState:
    current_page: int
    breadcrumbs: List[str]  # e.g., ["Chapter 1", "Section 1.2"]
    current_header_level: int
    last_processed_header: str

```

### 3.2 State Transition Logic

* **REQ-STATE-01:** The `breadcrumbs` list UPDATE ONLY when a new header is explicitly detected.
* **REQ-STATE-02:** If a new header is `Level 2`, it replaces the existing `Level 2` in the breadcrumbs and removes all deeper levels (3, 4, etc.).
* **REQ-STATE-03:** Every generated chunk (text or image) MUST strictly inherit a deep copy of the current `ContextState`.

---

## 4. MULTIMODAL ASSET EXTRACTION (REQ-MM)

### 4.1 Visual Assets (Images/Charts)

* **REQ-MM-01 (Element Cropping):**
* Detect Bounding Box (`bbox`) of the visual element.
* Apply 10px padding.
* Crop and save as PNG.

* **REQ-MM-02 (Naming Standard):**
* Format: `[DocHash]_[PageNum]_[Type]_[Index].png`
* Example: `a1b2c3d4_005_figure_01.png`

* **REQ-MM-03 (Contextual Anchoring):**
* The JSONL entry for an image MUST contain `text_before` (300 chars) and `text_after` (300 chars) from the surrounding text flow.

### 4.2 Tabular Data

* **REQ-MM-04 (Structure Preservation):**
* Tables MUST generally be converted to **Markdown**.
* **Exception:** If cells are merged or structure is complex, use **HTML Table** string representation.
* **Strict Rule:** Tables are never flattened to unstructured text.

---

## 5. CHUNKING STRATEGY & LOGIC (REQ-CHUNK)

### 5.1 Semantic Partitioning

* **REQ-CHUNK-01 (Boundary Integrity):** Splits MUST ONLY occur at sentence delimiters (`.`, `!`, `?`, `\n`). Mid-sentence splits are fatal errors.
* **REQ-CHUNK-02 (Token Limits):**
  * **Text Chunks:** Target 400 tokens. **Hard Max 512 tokens**.
  * **Atomic Chunks (Tables/Figures):** Allowed to extend to **1024 tokens** to preserve integrity. If >1024, split table with Header Repetition.

### 5.2 Dynamic Semantic Overlap (DSO)

* **REQ-CHUNK-03 (Implementation):**
  * **Trigger:** `--semantic-overlap` flag.
  * **Model:** `sentence-transformers/all-MiniLM-L6-v2` (Version 2.2.2).

  * **Algorithm:**

1. Extract last 3 sentences of Chunk A.
2. Extract first 3 sentences of Chunk B.
3. Compute Cosine Similarity.
4. IF `sim > 0.85` THEN `overlap = base_overlap * 1.5` ELSE `overlap = base_overlap`.

* **Constraint:** Overlap < 25% of total chunk size.

### 5.3 Primary Output Format

* **REQ-OUT-01 (JSONL Format):** The primary text output MUST be in .jsonl (JSON Lines) format, not a single .json array, to support high-scale streaming and memory efficiency.

---

## 6. CANONICAL OUTPUT SCHEMA (JSONL)

Every line in `ingestion.jsonl` MUST validate against this schema. **No other output formats are permitted.**

```json
{
  "chunk_id": "UUID_v4",
  "doc_id": "File_Hash_MD5",
  "modality": "text | image | table",
  "content": "The actual text or markdown content.",
  "metadata": {
    "source_file": "manual.pdf",
    "file_type": "pdf",
    "page_number": 42,
    "hierarchy": {
      "parent_heading": "3.2 Installation",
      "breadcrumb_path": ["Chapter 3", "3.2 Installation"],
      "level": 2
    },
    "spatial": {
      "bbox": [100, 200, 500, 600] 
    },
    "content_classification": "editorial" 
  },
  "asset_ref": {
    "file_path": "assets/manual_p42_fig1.png",
    "visual_description": "Optional AI description"
  },
  "semantic_context": {
    "prev_text_snippet": "As shown in the diagram...",
    "next_text_snippet": "...connect the cable."
  }
}

```

---

## 7. QUALITY ASSURANCE & LOGGING

### 7.1 Automated Checks (Post-Processing)

* **QA-CHECK-01:** Verify `sum(chunk_tokens) ~= total_document_tokens` (tolerance 10%).
* **QA-CHECK-02:** Verify every `asset_ref.file_path` exists on disk.
* **QA-CHECK-03:** Verify `breadcrumb_path` depth is consistent (e.g., Level 3 must have 3 items).

### 7.2 Engineering Imperatives

* **Dependency Pinning:** `docling>=2.66.0`, `surya-ocr>=0.4.0`, `sentence-transformers>=3.0.0`.
* **Error Handling:** Use a `try-catch` block per file. A single corrupt file MUST NOT crash the batch. Log errors to `ingestion_errors.log`.
* **Platform Optimisation:** The application must be optimised for the platform is runs on (i.e. Apple Sillicon (ARM))
