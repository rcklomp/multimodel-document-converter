# MM-RAG Converter V2

Convert PDF, EPUB, HTML, and Office documents into structured JSONL datasets for Multimodal RAG systems.

The converter extracts text, images, and tables from complex documents while preserving spatial layout, document hierarchy, and semantic context. It handles everything from born-digital magazines to degraded scanned manuals.

**Version 2.13.0 SHIPPED** (annotated tag staged 2026-05-22 for user push to GitHub + Gitea) | predecessors `v2.12.0` (`5a2ce18`) → `v2.11.0` (`c2a461c`) → `v2.10.0` (`db6527c`) → `v2.9.0-rc1` (`3e06d1b`) → `v2.8.0` | Python 3.10 | Apple Silicon native | Docling 2.86.0 | Schema 2.7.0

> **v2.13.0 — local embedder release.** v2.13 closes two parallel
> workstreams on top of v2.12.0's retrieval stack: Phase 1 swaps
> the production embedder from cloud Dashscope `text-embedding-v4`
> to local `Qwen3-Embedding-8B-mxfp8` via omlx-server (6/6-axis
> apples-to-apples win on same fixture); Phase 2 fixes OCR
> auto-routing so scanned-profile docs honour Docling's
> `force_full_page_ocr` flag (Earthship Format +6.2pp).
>
> | Axis | v2.12.0 (cloud) | **v2.13.0 (local omlx)** | Δ |
> |---|---:|---:|---|
> | Recall@1 chunk | 55.0% | **57.5%** | **+2.5pp** |
> | Recall@5 chunk | 72.6% | **78.0%** | **+5.4pp** |
> | Recall@5 doc | 93.1% | **95.2%** | +2.1pp |
> | Relevance (judge) | 74.1% | **74.6%** | +0.5pp |
> | **Format (judge)** | 89.2% | **92.9%** | **+3.7pp** |
> | Faithfulness (judge) | 65.9% | **66.9%** | +1.0pp |
>
> Numbers from the v2.13 P1 apples-to-apples fixture (same chunks,
> queries, judge, and retrieval stack — only the embedder differs).
> Not directly comparable to v2.12.0's 67.8% R@1 canonical number
> (that was on the v2.11 baseline fixture); the v2.13 P1 fixture is
> a fresh post-v2.13-P2 sample, harder for both providers, but omlx
> wins on every axis vs the same-fixture dashscope read.
>
> **Production retrieval stack (v2.13.0):**
>
> ```
> query
>   ├─ dense  : omlx Qwen3-Embedding-8B-mxfp8 → mmrag_v2_8__qwen3_local (4096-dim)
>   └─ sparse : BM25 → mmrag_v2_8__bm25_sparse
>   → RRF fusion (k=60, equal weights), top-25 per leg
>   → rerank (local gte-reranker-modernbert-base-mlx via omlx-server)
>   → top-5 return
>
> End-to-end p99 latency: ~1.6 s (cloud→LAN embed swap saves ~400 ms)
> Per-query cost: $0 (no cloud calls on retrieval path)
> External dependencies: omlx-server LAN only (privacy + offline-capable)
> ```
>
> **Phase contributions:**
>
> - **Phase 1**: local embedder swap. Qwen3-Embedding-8B-mxfp8 wins
>   6/6 axes vs cloud text-embedding-v4 on same fixture; production
>   default flipped. Dashscope retained as 30-day rollback baseline
>   through 2026-06-19.
> - **Phase 2**: OCR auto-routing. `force_full_page_ocr=True` is now
>   honored end-to-end (was previously bypassed by layout-aware OCR
>   path). Earthship +6.2pp Format on partial soak; CarOK documented
>   as judge-calibration limitation for v2.14.
>
> Test suite: **1033 passed**, 16 skipped, 0 failed. v2.10 strict-gate
> corpus state unchanged at **34 PASS / 0 WARN / 0 FAIL** (v2.11,
> v2.12, v2.13 all changed only the retrieval side and OCR routing;
> extraction/chunking/validation contracts untouched).
>
> AFTER snapshot:
> [`docs/archive/snapshots/QUALITY_SNAPSHOT_2026-05-22_v2.13_after.md`](docs/archive/snapshots/QUALITY_SNAPSHOT_2026-05-22_v2.13_after.md).
> Phase 1 SWAP evidence:
> [`docs/archive/snapshots/QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_omlx_vs_dashscope.md`](docs/archive/snapshots/QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_omlx_vs_dashscope.md).
> Plan: [`docs/archive/plans/PLAN_V2.13.md`](docs/archive/plans/PLAN_V2.13.md) (CLOSED 2026-05-22).
>
> **Cumulative v2.10 → v2.13 trajectory on Recall@1 chunk** (note: v2.13
> bar is on the harder v2.13 P1 fixture; v2.10-v2.12 are on the v2.11
> baseline fixture):
>
> ```
> v2.10  ====                                          2.1%   (llava embedder, dense only)
> v2.11  ==============                               35.5%   (dashscope, dense only)
> v2.12  ==============================               67.8%   (hybrid + rerank, dashscope)
> v2.13  ===========================                  57.5%*  (hybrid + rerank, omlx local — *new fixture)
>         0%       20%       40%       60%       80%
> ```
>
> **Recommendation:** use v2.13.0 as the production retrieval stack.
> The system now runs the full retrieval path with no cloud
> dependency on the embedding side. CarOK Format penalty and a few
> domain regressions (German content, code-heavy Python_Cookbook)
> are documented v2.14 carry-forwards.

---

## Quick Start

```bash
# Install
conda env create -f environment.yml
conda activate mmrag-v2
pip install -e .

# Convert a PDF
mmrag-v2 process document.pdf --output-dir ./output

# Convert with VLM image descriptions (LM Studio example)
mmrag-v2 process document.pdf \
  --vision-provider openai \
  --vision-model your-vision-model \
  --vision-base-url http://localhost:1234/v1 \
  --api-key lm-studio \
  --output-dir ./output

# Convert an EPUB or HTML file
mmrag-v2 process book.epub --output-dir ./output
mmrag-v2 process article.html --output-dir ./output
```

### Output

```
output/
├── ingestion.jsonl     # One JSON object per line (text, image, or table chunk)
├── assets/             # Extracted images as PNG files
│   ├── a1b2c3d4_001_figure_01.png
│   └── ...
└── .vision_cache.json  # Cached VLM descriptions (avoids re-processing)
```

---

## How It Works

The pipeline has three stages:

### 1. Document Analysis

Before extraction begins, the converter analyzes the document to determine the best processing strategy:

- **Structural diagnosis**: Detects scanned pages, encoding corruption, and flat-text corruption
- **Profile classification**: Selects one of 5 processing profiles based on text density, image ratio, page count, and content domain
- **OCR decision**: Automatically enables OCR when scanned content is detected

### 2. Extraction

The extraction engine (Docling) processes each page to identify text paragraphs, images, tables, headings, code blocks, and list items. For scanned documents, a 3-layer OCR cascade runs:

```
Layer 1: Docling layout-aware OCR  →  already extracted during layout analysis
                                    ↓
Layer 2: Tesseract 5 + preprocessing →  confidence > threshold → accept
                                    ↓
Layer 3: DocTR                       →  final pass (accept all)
```

A Vision Language Model (VLM) generates descriptions for extracted images, enabling image search through text queries.

### 3. Post-Processing & Quality Assurance

After extraction, the pipeline applies:

- **Code detection and reflow**: Identifies code blocks misclassified as paragraphs and restores formatting
- **Oversize breaking**: Splits chunks exceeding 1500 characters at sentence boundaries
- **Token validation (QA-CHECK-01)**: Verifies that extracted text accounts for the document's content within 10% tolerance
- **Deduplication**: Perceptual hashing (pHash) removes duplicate images
- **Coordinate normalization**: All bounding boxes mapped to a [0, 1000] integer grid

---

## Processing Profiles

The converter automatically selects a profile based on document characteristics:

| Profile | When Selected | Key Settings |
|---------|--------------|--------------|
| `digital_magazine` | Born-digital editorial content (magazines, illustrated books) | Sensitivity 0.5, min image 100px, DPI 150 |
| `academic_whitepaper` | High text density + academic/technical domain | Sensitivity 0.6, min image 30px, DPI 150 |
| `scanned` | Standard quality scanned documents | Sensitivity 0.7, min image 30px, DPI 200, OCR enabled |
| `scanned_degraded` | Low quality or degraded scans | Sensitivity 0.8, min image 30px, DPI 300, aggressive OCR |
| `technical_manual` | Technical manuals, coding books, handbooks | Sensitivity 0.8, min image 30px, DPI 300, batch size 3 |

Content domains detected: `academic`, `editorial`, `technical`, `literature`, `commercial`, `unknown`.

---

## Supported Formats

| Format | Support | Notes |
|--------|---------|-------|
| PDF | Full | Batched processing, OCR cascade, VLM enrichment |
| HTML/HTM | Full | Direct Docling processing |
| EPUB | Full | Spine-ordered chapter extraction with synthetic per-chapter pagination (v2.10 §EPUB lane) |
| DOCX | Full | Direct Docling processing |
| PPTX | Full | Direct Docling processing |
| XLSX | Full | Direct Docling processing |

Batched processing (splitting into N-page batches for memory efficiency) is PDF-only. Other formats use single-pass processing.

**EPUB note (v2.10 Phase 7):** EPUB has no native pagination, so chunks emit a synthetic `page_number = chapter_1based * 1000 + position_in_chapter // 5` and the documented full-page bbox sentinel `[0, 0, 1000, 1000]`. A chunk on page `13029` means "chapter 13, ~6th synthetic page" — not "page 13029 of the source book". Cite chapter index (`page_number // 1000`) for human-facing references. See [`docs/CONVERSION_PROFILES.md`](docs/CONVERSION_PROFILES.md) §EPUB Lane for details.

---

## CLI Reference

### `mmrag-v2 process`

Convert a single document.

```bash
mmrag-v2 process [OPTIONS] INPUT_FILE
```

**Common options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir`, `-o` | `./output` | Output directory |
| `--batch-size`, `-b` | `10` | Pages per batch (PDF only; 0 = no batching) |
| `--pages` | all | Page limit (`20`) or specific pages (`6,21,169`) |
| `--vision-provider`, `-v` | `none` | VLM: `openai`, `ollama`, `anthropic`, `none` |
| `--vision-model` | auto | Model name for VLM |
| `--vision-base-url` | none | OpenAI-compatible endpoint URL |
| `--api-key`, `-k` | env var | API key for VLM/cloud providers |
| `--sensitivity`, `-s` | profile-based | Image extraction sensitivity (0.1-1.0) |
| `--strict-qa` | off | Fail processing on QA violations |

**OCR options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--enable-ocr/--no-ocr` | auto | OCR auto-enables for scanned documents |
| `--ocr-engine` | `easyocr` | Engine: `tesseract`, `easyocr`, `doctr` |
| `--ocr-mode` | `auto` | `auto`, `legacy`, or `layout-aware` |
| `--force-ocr` | off | Force OCR even on digital PDFs |

**Refiner options (LLM-based OCR cleanup):**

| Option | Default | Description |
|--------|---------|-------------|
| `--enable-refiner` | off | Enable semantic text refinement |
| `--refiner-provider` | `ollama` | `ollama`, `openai`, `anthropic` |
| `--refiner-model` | auto | Model for refinement |
| `--refiner-base-url` | none | OpenAI-compatible endpoint |

### `mmrag-v2 batch`

Convert all matching files in a directory.

```bash
mmrag-v2 batch INPUT_DIR --pattern "*.pdf" --output-dir ./output
```

### `mmrag-v2 version` / `mmrag-v2 check`

Show version info or verify system dependencies.

### Examples

```bash
# Digital magazine with VLM descriptions
mmrag-v2 process magazine.pdf -b 10 \
  -v openai --vision-model llava-1.6 \
  --vision-base-url http://localhost:1234/v1 --api-key lm-studio

# Scanned technical manual (OCR auto-enabled)
mmrag-v2 process manual_scan.pdf -b 3

# Academic paper, strict QA
mmrag-v2 process paper.pdf -b 10 --strict-qa

# First 20 pages only
mmrag-v2 process large_book.pdf --pages 20 -b 10

# Specific pages
mmrag-v2 process reference.pdf --pages 6,21,169,241

# Batch convert a folder
mmrag-v2 batch ./documents -p "*.pdf" -v none -o ./converted
```

---

## Output Schema

The first line of `ingestion.jsonl` is a metadata record:

```json
{
  "object_type": "ingestion_metadata",
  "schema_version": "2.7.0",
  "doc_id": "a1b2c3d4e5f6",
  "source_file": "document.pdf",
  "profile_type": "digital_magazine",
  "domain": "editorial",
  "is_scan": false,
  "total_pages": 108,
  "chunk_count": 276
}
```

All subsequent lines are content chunks. Three modalities:

### Text Chunk

```json
{
  "chunk_id": "a1b2c3d4_042_8f3a2b1c",
  "doc_id": "a1b2c3d4e5f6",
  "modality": "text",
  "content": "The converter extracts text while preserving document hierarchy...",
  "chunk_type": "paragraph",
  "metadata": {
    "page_number": 42,
    "chunk_type": "paragraph",
    "hierarchy": {
      "parent_heading": "Processing Pipeline",
      "breadcrumb_path": ["Document Title", "Page 42", "Processing Pipeline"],
      "level": 3
    },
    "spatial": {
      "bbox": [50, 100, 950, 400],
      "page_width": 612,
      "page_height": 792
    },
    "extraction_method": "docling"
  },
  "semantic_context": {
    "prev_text_snippet": "...end of previous chunk for overlap.",
    "next_text_snippet": "Start of next chunk for context..."
  },
  "schema_version": "2.7.0"
}
```

### Image Chunk

```json
{
  "chunk_id": "a1b2c3d4_042_figure_01",
  "modality": "image",
  "content": "Cutaway diagram showing internal mechanism with labeled components.",
  "visual_description": "Cutaway diagram showing internal mechanism with labeled components.",
  "asset_ref": {
    "file_path": "assets/a1b2c3d4_042_figure_01.png",
    "mime_type": "image/png",
    "width_px": 1600,
    "height_px": 800
  }
}
```

### Table Chunk

```json
{
  "chunk_id": "a1b2c3d4_042_table_01",
  "modality": "table",
  "content": "| Component | Role |\n| --- | --- |\n| Converter | Document to chunks |\n| Vector Store | Similarity search |",
  "asset_ref": {
    "file_path": "assets/a1b2c3d4_042_table_01.png",
    "mime_type": "image/png"
  }
}
```

### Coordinate System

All `bbox` values are integers in the range [0, 1000], representing a normalized page grid. To convert to pixel coordinates:

```python
# bbox = [100, 200, 900, 600], page_width = 612, page_height = 792
dpi_scale = 150 / 72  # PDF points to pixels at 150 DPI
render_width = page_width * dpi_scale

x0_px = (bbox[0] / 1000) * render_width
y0_px = (bbox[1] / 1000) * render_height
```

---

## Installation

### Requirements

- macOS with Apple Silicon (M1/M2/M3/M4) or Linux
- Python 3.10 (not 3.11+)
- Conda (Miniconda or Anaconda)
- 16 GB RAM minimum

### Setup

```bash
git clone git@github.com:rcklomp/multimodel-document-converter.git
cd multimodel-document-converter

conda env create -f environment.yml
conda activate mmrag-v2
pip install -e .

# Verify
mmrag-v2 version
mmrag-v2 check
```

### Vision Provider Setup

The VLM is optional but recommended. Without it, images get placeholder descriptions.

**LM Studio (recommended for local use):**
```bash
# Download from https://lmstudio.ai, load a vision model, start the server
mmrag-v2 process doc.pdf -v openai --vision-base-url http://localhost:1234/v1 --api-key lm-studio
```

**Ollama:**
```bash
brew install ollama && ollama pull llava:latest && ollama serve
mmrag-v2 process doc.pdf -v ollama
```

**Cloud (OpenAI / Anthropic):**
```bash
export OPENAI_API_KEY="sk-..."
mmrag-v2 process doc.pdf -v openai
```

---

## Development

```bash
# Run tests
make test

# Lint and format
make lint
make fmt

# Type checking
make typecheck

# All checks (lint + typecheck + tests)
make check

# Multi-profile smoke test (10 pages each, no VLM)
make smoke

# Technical manual acceptance test (4 docs x 20 pages)
make acceptance
```

### Project Structure

```
src/mmrag_v2/
├── cli.py                          # CLI entry point (process, batch, version, check)
├── batch_processor.py              # PDF batch orchestrator
├── processor.py                    # Core Docling-based document conversion
├── version.py                      # Single source of truth for version
├── schema/ingestion_schema.py      # Pydantic models for JSONL output
├── orchestration/
│   ├── document_diagnostic.py      # Pre-flight structural analysis
│   ├── profile_classifier.py       # Multi-dimensional profile selection
│   ├── strategy_profiles.py        # Profile parameter definitions
│   └── strategy_orchestrator.py    # Dynamic extraction configuration
├── vision/
│   ├── vision_manager.py           # Multi-provider VLM abstraction
│   └── vision_prompts.py           # Diagram/photo/generic prompt templates
├── ocr/
│   ├── enhanced_ocr_engine.py      # 3-layer OCR cascade
│   └── layout_aware_processor.py   # Layout-based region detection + OCR
├── validators/
│   ├── token_validator.py          # QA-CHECK-01 token balance
│   └── quality_filter_tracker.py   # Filtering analytics
└── state/context_state.py          # Breadcrumb hierarchy state machine
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No text from scanned PDF | OCR auto-enables for scans since v2.6. For older behavior, add `--enable-ocr` |
| Memory error on large PDF | Reduce `--batch-size` (try 3 or 5) |
| Slow VLM processing | VLM processes each image individually. Reduce pages with `--pages 20` or use `--vision-provider none` for text-only |
| No images extracted | Increase `--sensitivity` (try 0.8) |
| `ModuleNotFoundError: tiktoken` | `pip install tiktoken` |
| Ollama connection refused | Run `ollama serve` in another terminal |

---

## License

See LICENSE file for details.
