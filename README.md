# 🚀 Multimodal RAG Corpus Converter V2.4.1

**Enterprise-grade document conversion for Advanced Multimodal RAG systems.**

Transform complex, visually-rich documents (magazines, technical manuals, reports) into production-ready RAG datasets with **zero information loss**, **intelligent overlap**, and **military-grade data integrity**.

[![SRS Compliance](https://img.shields.io/badge/SRS-v2.4.1--stable-blue.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10-green.svg)]()
[![Engine](https://img.shields.io/badge/Engine-Docling%20v2.66.0-orange.svg)]()
[![Platform](https://img.shields.io/badge/Platform-Apple%20Silicon%20Native-silver.svg)]()

---

## 📋 Table of Contents

- [Why This Exists](#-why-this-exists)
- [Key Features](#-key-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [CLI Reference](#-cli-reference)
- [Semantic Text Refiner (v18.2)](#-semantic-text-refiner-v182)
- [Architecture](#-architecture)
- [Processing Pipeline](#-processing-pipeline)
- [Smart Vision Orchestration](#-smart-vision-orchestration)
- [Quality Assurance](#-quality-assurance)
- [Output Schema](#-output-schema)
- [Integration Examples](#-integration-examples)
- [Layout-Aware OCR](#-layout-aware-ocr-phase-1b)
- [Advanced Configuration](#-advanced-configuration)
- [Troubleshooting](#-troubleshooting)
- [API Reference](#-api-reference)

---

## 🎯 Why This Exists

**The Problem:** Converting PDFs to RAG-ready datasets sounds simple—until you encounter:
- Full-page editorial photos that standard parsers miss
- Advertisements polluting your vector database
- Context loss when text chunks lose their document hierarchy
- "Garbage in, garbage out" syndrome from naive chunking
- Memory explosions on 244-page magazines

**The Solution:** This converter implements a **production-grade ETL pipeline** that:

✅ **Never loses information** (Token validation with 10% tolerance)  
✅ **Maintains semantic context** (Dynamic Semantic Overlap between chunks)  
✅ **Extracts missed assets** (Shadow Extraction for AI-blind images)  
✅ **Preserves hierarchy** (Breadcrumb state machine across batch boundaries)  
✅ **Runs on 16GB RAM** (Memory-efficient batch processing with gc.collect())  

---

## ✨ Key Features

### 🧠 Intelligent Processing

| Feature | Description |
|---------|-------------|
| **Multi-Dimensional Profile Classifier** | Automatic document classification based on physical checks, text density, image ratio, and detected domain/era. Selects optimal processing strategy. |
| **Dynamic Semantic Overlap (DSO)** | AI-powered overlap calculation between chunks using sentence embeddings. High-similarity boundaries get 1.5x overlap to preserve context. |
| **Smart Vision Orchestration** | Automatic document classification with adaptive extraction thresholds via `--sensitivity` dial (0.1-1.0). |
| **Auto OCR Mode Detection** | Automatically selects `legacy` or `layout-aware` OCR mode based on document diagnostics. |
| **Token Post-Validation (QA-CHECK-01)** | Tiktoken-based verification that chunk tokens match source tokens ±10%. Catches data loss before it reaches your vector DB. |

### 🎨 Multimodal Excellence

| Feature | Description |
|---------|-------------|
| **High-Fidelity Rendering** | 2.0x scale, 10px padding on all asset crops. No blurry extractions. |
| **VLM Integration** | Ollama (local), OpenAI (gpt-4o-mini), LM Studio (OpenAI-compatible), Anthropic (claude-3-5-haiku) for image descriptions. |
| **Perceptual Deduplication** | pHash-based duplicate detection prevents the same image appearing multiple times. |
| **Asset Integrity** | Every `asset_ref.file_path` verified to exist on disk. Orphan chunks are impossible. |
| **Coordinate System** | REQ-COORD-02: `page_width` and `page_height` in spatial metadata for UI overlay support. |

### 🏗️ Production Architecture

| Feature | Description |
|---------|-------------|
| **Batch Processing** | Large PDFs split into N-page batches. 244 pages? No problem. |
| **Memory Hygiene** | `gc.collect()` between batches. Runs stable on 16GB M-series Macs. |
| **Breadcrumb Continuity** | Document hierarchy preserved across batch boundaries. No "orphan" chunks. |
| **Strict Mode** | `--strict-qa` flag fails processing if token validation detects data loss. |

---

## 🚀 Quick Start

### Process Your First Document

```bash
# Basic processing with local Ollama VLM
mmrag-v2 process document.pdf --output-dir ./output

# Large document with batch processing
mmrag-v2 process large_magazine.pdf --batch-size 10 --vision-provider ollama

# Maximum recall for photo-heavy documents
mmrag-v2 process photo_book.pdf --sensitivity 0.9 --batch-size 10

# Strict mode (fails on data loss)
mmrag-v2 process critical_document.pdf --batch-size 10 --strict-qa

# Using LM Studio (OpenAI-compatible API)
mmrag-v2 process document.pdf --vision-provider openai --api-key lm-studio
```

### Output Structure

```
output/
├── ingestion.jsonl          # RAG-ready dataset (1 chunk per line)
├── assets/                  # High-resolution PNG extractions
│   ├── a1b2c3d4_001_figure_01.png
│   ├── a1b2c3d4_002_table_01.png
│   └── ...
└── vision_cache.json        # VLM description cache (prevents redundant calls)
```

---

## 📦 Installation

### Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3) - Native ARM64
- **Python 3.10** (not 3.11+ due to Docling compatibility)
- **Conda** (Miniconda or Anaconda)
- **16GB RAM** minimum (32GB recommended for large documents)

### Step 1: Clone and Setup Environment

```bash
# Clone repository
git clone https://github.com/your-org/MM-Converter-V2.git
cd MM-Converter-V2

# Create conda environment from spec
conda env create -f environment.yml

# Activate environment
conda activate ./env

# Install package in development mode
pip install -e .
```

### Step 2: Verify Installation

```bash
# Check CLI is available
mmrag-v2 version

# Verify Docling engine
python -c "import docling; print(f'Docling {docling.__version__}')"

# Check vision providers
mmrag-v2 check
```

### Step 3: Setup Vision Provider

#### Option A: Local Ollama (Free, Private)

```bash
# Install Ollama
brew install ollama

# Pull vision model
ollama pull llava:latest

# Start server (keep running)
ollama serve
```

#### Option B: LM Studio (OpenAI-Compatible, Recommended for v17.1+)

```bash
# 1. Download LM Studio from https://lmstudio.ai
# 2. Load a vision model (e.g., llama-3.2-vision)
# 3. Start the local server (default: http://localhost:1234)

# Set base URL environment variable
export OPENAI_BASE_URL="http://localhost:1234/v1"

# Use with the openai provider
mmrag-v2 process document.pdf --vision-provider openai --api-key lm-studio
```

#### Option C: Cloud Providers

```bash
# OpenAI
export OPENAI_API_KEY="sk-your-key"

# Anthropic
export ANTHROPIC_API_KEY="your-key"
```

---

## 💻 CLI Reference

### Main Command: `process`

```bash
mmrag-v2 process [OPTIONS] INPUT_FILE
```

#### Core Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir`, `-o` | `./output` | Directory for JSONL and assets |
| `--batch-size`, `-b` | `10` | Pages per batch (0=disable batching) |
| `--pages` | `None` | **Specific pages** (comma-separated: `6,21,169,241`) or **max count** (single number: `10`) |
| `--vision-provider`, `-v` | `ollama` | VLM provider: `ollama`, `openai`, `anthropic`, `haiku`, `none` |
| `--vision-model`, `-m` | Auto-detect | Model name (e.g., `llava:latest`, `gpt-4o-mini`) |
| `--vision-base-url` | `None` | Base URL for OpenAI-compatible endpoints (e.g., `http://localhost:1234/v1`) |
| `--api-key`, `-k` | Env var | API key for cloud providers |
| `--vlm-timeout` | `180` | VLM call timeout in seconds (default: 180) |
| `--enable-cache/--no-cache` | `True` | Enable/disable vision cache for repeated images |

#### Quality Control Options

| Option | Default | Description |
|--------|---------|-------------|
| `--sensitivity`, `-s` | `0.5` | Image extraction sensitivity (0.1-1.0) |
| `--strict-qa` | `False` | Fail on token validation errors |
| `--semantic-overlap` | `True` | Enable Dynamic Semantic Overlap (DSO) chunking |
| `--vlm-context-depth` | `3` | Previous text chunks for VLM semantic anchoring |
| `--allow-fullpage-shadow` | `False` | Override Full-Page Guard |

#### OCR Options

| Option | Default | Description |
|--------|---------|-------------|
| `--enable-ocr/--no-ocr` | `False` | Enable OCR for scanned pages |
| `--ocr-engine` | `easyocr` | Engine: `tesseract`, `easyocr`, or `doctr` |
| `--ocr-mode` | `auto` | OCR mode: `auto` (smart detection), `legacy`, or `layout-aware` |
| `--ocr-confidence-threshold` | `0.7` | Minimum OCR confidence for layout-aware mode (0.0-1.0) |
| `--enable-doctr/--no-doctr` | `True` | Enable Doctr Layer 3 for layout-aware OCR |

### Examples

```bash
# Editorial magazine with high recall
mmrag-v2 process magazine.pdf \
  --sensitivity 0.8 \
  --batch-size 10 \
  --vision-provider ollama \
  --output-dir ./magazine_output

# Technical manual with strict validation
mmrag-v2 process manual.pdf \
  --sensitivity 0.3 \
  --batch-size 20 \
  --strict-qa \
  --output-dir ./manual_output

# Scanned document with layout-aware OCR
mmrag-v2 process scanned.pdf \
  --ocr-mode layout-aware \
  --enable-ocr \
  --batch-size 10 \
  --output-dir ./scanned_output

# LM Studio with OpenAI-compatible API
mmrag-v2 process document.pdf \
  --vision-provider openai \
  --api-key lm-studio \
  --batch-size 10

# Process SPECIFIC pages only (targeted extraction)
mmrag-v2 process firearms.pdf --pages 6,21,169,241 --no-cache

# Process first 20 pages only (max count mode)
mmrag-v2 process large.pdf --pages 20 --batch-size 10
```

### Utility Commands

```bash
# Check vision provider availability
mmrag-v2 check

# Show version and engine info
mmrag-v2 version

# Batch process directory
mmrag-v2 batch ./documents --pattern "*.pdf" --vision-provider ollama
```

### Batch Processing

```bash
mmrag-v2 batch [OPTIONS] INPUT_DIR
```

Batch options (includes refiner passthrough):

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir`, `-o` | `./output` | Directory for JSONL and assets |
| `--pattern`, `-p` | `*.pdf` | Glob pattern for files |
| `--vision-provider`, `-v` | `ollama` | VLM provider for images |
| `--vision-base-url` | `None` | Base URL for OpenAI-compatible endpoints |
| `--api-key`, `-k` | Env var | API key for cloud providers |
| `--vlm-timeout` | `180` | VLM call timeout in seconds |
| `--enable-cache/--no-cache` | `True` | Enable/disable vision cache for repeated images |
| `--enable-ocr/--no-ocr` | `False` | Enable OCR for scanned documents |
| `--ocr-mode` | `auto` | OCR mode: `auto`, `legacy`, or `layout-aware` |
| `--enable-refiner/--no-refiner` | `False` | Enable Semantic Text Refiner |
| `--refiner-provider` | `ollama` | Refiner provider (`ollama|openai|anthropic`) |
| `--refiner-model` | Auto-detect | Refiner model name |
| `--refiner-base-url` | `None` | OpenAI-compatible base URL (LM Studio, LocalAI, vLLM) |

Example:

```bash
mmrag-v2 batch ./documents --enable-refiner \
  --refiner-provider openai \
  --refiner-model llama-joycaption-beta-one-hf-llava-mmproj \
  --refiner-base-url http://localhost:1234/v1 \
  --api-key "lm-studio"
```

---

## 🧠 Semantic Text Refiner (v18.2)

The Semantic Text Refiner runs after OCR to fix obvious OCR artifacts while preserving technical integrity. The original `content` is never overwritten; accepted refinements are stored in `metadata.refined_content`.

### Safety Guarantees

- **Provenance lock:** Original text preserved; refined text is opt-in.
- **Protected tokens:** Part numbers, ECU codes, SN/ID tokens, URLs, and dates are immutable.
- **Edit budget:** Levenshtein ratio guardrail prevents over-editing.
- **Vision anchors:** Visual descriptions are treated as ground truth for entity names.
- **Fail-safe:** On errors/timeouts, the refiner bypasses and returns original text.

### Provider Matrix

1) **Local Ollama (default)**
```bash
mmrag-v2 process file.pdf --enable-refiner
```

2) **LM Studio / Local OpenAI-compatible**
```bash
mmrag-v2 process file.pdf --enable-refiner \
  --refiner-provider openai \
  --refiner-model llama-joycaption-beta-one-hf-llava-mmproj \
  --refiner-base-url http://localhost:1234/v1 \
  --api-key "lm-studio"
```

3) **OpenAI / Anthropic Cloud**
```bash
# OpenAI
mmrag-v2 process file.pdf --enable-refiner \
  --refiner-provider openai \
  --refiner-model gpt-4o-mini \
  --api-key $OPENAI_API_KEY

# Anthropic
mmrag-v2 process file.pdf --enable-refiner \
  --refiner-provider anthropic \
  --refiner-model claude-3-5-haiku-20241022 \
  --api-key $ANTHROPIC_API_KEY
```

### Refiner Flags

- `--enable-refiner/--no-refiner`
- `--refiner-provider`
- `--refiner-model`
- `--refiner-base-url`
- `--api-key`

---

## 🏛️ Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                     MM-Converter-V2 Pipeline                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌───────────────────┐    ┌───────────────────┐  │
│  │ PDF/EPUB │───▶│ DocumentDiagnostic│───▶│Profile Classifier │  │
│  │ DOCX/HTML│    │ (Physical Checks) │    │(Multi-Dimensional)│  │
│  └──────────┘    └───────────────────┘    └───────────────────┘  │
│                           │                        │             │
│                           ▼                        ▼             │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                   BatchProcessor                         │    │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐  │    │
│  │  │ PDF Split  │─▶│ Docling    │─▶│ OCR Cascade        │  │    │
│  │  │ (N pages)  │  │ (Layout AI)│  │ (Auto/Layout-Aware)│  │    │
│  │  └────────────┘  └────────────┘  └────────────────────┘  │    │
│  │         │               │                    │           │    │
│  │         ▼               ▼                    ▼           │    │
│  │  ┌────────────────────────────────────────────────────┐  │    │
│  │  │              Context State Machine                 │  │    │
│  │  │    (Breadcrumbs, Hierarchy, Page Tracking)         │  │    │
│  │  └────────────────────────────────────────────────────┘  │    │
│  │         │                                                │    │
│  │         ▼                                                │    │
│  │  ┌────────────────────────────────────────────────────┐  │    │
│  │  │              VLM Enrichment Layer                  │  │    │
│  │  │    (Ollama/OpenAI/LM Studio/Anthropic + Cache)     │  │    │
│  │  └────────────────────────────────────────────────────┘  │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                │                                 │
│                                ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                 Quality Assurance Layer                  │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │    │
│  │  │ Token        │  │ Asset        │  │ Coordinate     │  │    │
│  │  │ Validator    │  │ Validator    │  │ Validator      │  │    │
│  │  │ (QA-CHECK-01)│  │ (QA-CHECK-02)│  │ (REQ-COORD)    │  │    │
│  │  └──────────────┘  └──────────────┘  └────────────────┘  │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                │                                 │
│                                ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                    Output Layer                          │    │
│  │         ingestion.jsonl + assets/ directory              │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `BatchProcessor` | `src/mmrag_v2/batch_processor.py` | Orchestrates large PDF processing with memory management |
| `V2DocumentProcessor` | `src/mmrag_v2/processor.py` | Core Docling-based document conversion |
| `DocumentDiagnosticEngine` | `src/mmrag_v2/orchestration/document_diagnostic.py` | Pre-flight document analysis |
| `ProfileClassifier` | `src/mmrag_v2/orchestration/profile_classifier.py` | Multi-dimensional profile selection |
| `SmartConfigProvider` | `src/mmrag_v2/orchestration/smart_config.py` | Document profiling and classification |
| `StrategyOrchestrator` | `src/mmrag_v2/orchestration/strategy_orchestrator.py` | Dynamic threshold computation |
| `VisionManager` | `src/mmrag_v2/vision/vision_manager.py` | Multi-provider VLM abstraction |
| `TokenValidator` | `src/mmrag_v2/validators/token_validator.py` | QA-CHECK-01 data integrity guard |
| `ContextStateV2` | `src/mmrag_v2/state/context_state.py` | Breadcrumb state machine |
| `EnhancedOCREngine` | `src/mmrag_v2/ocr/enhanced_ocr_engine.py` | 3-layer OCR cascade |

---

## 📄 Output Schema

### ingestion.jsonl Format

Each line is a valid JSON object:

```json
{
  "chunk_id": "a1b2c3d4_042_8f3a2b1c9d4e5f6a",
  "doc_id": "a1b2c3d4e5f6",
  "modality": "text",
  "content": "The F-35 Lightning II represents a quantum leap in stealth technology...",
  "metadata": {
    "source_file": "combat_aircraft_aug2025.pdf",
    "file_type": "pdf",
    "page_number": 42,
    "chunk_type": "paragraph",
    "hierarchy": {
      "parent_heading": "Stealth Technology",
      "breadcrumb_path": ["Combat Aircraft", "August 2025", "Features", "F-35 Deep Dive"],
      "level": 4
    },
    "spatial": {
      "bbox": [50, 100, 950, 800],
      "page_width": 612,
      "page_height": 792
    },
    "extraction_method": "docling",
    "created_at": "2025-12-30T14:23:45.123456+00:00"
  },
  "schema_version": "2.0.0"
}
```

### Image Chunk Example

```json
{
  "chunk_id": "a1b2c3d4_042_figure_01",
  "doc_id": "a1b2c3d4e5f6",
  "modality": "image",
  "content": "Cutaway diagram showing the F-35's internal weapons bay with AIM-120 AMRAAM missiles.",
  "metadata": {
    "source_file": "combat_aircraft_aug2025.pdf",
    "file_type": "pdf",
    "page_number": 42,
    "hierarchy": {
      "breadcrumb_path": ["Combat Aircraft", "August 2025", "Features", "F-35 Deep Dive"],
      "level": 4
    },
    "spatial": {
      "bbox": [100, 200, 900, 600],
      "page_width": 612,
      "page_height": 792
    },
    "extraction_method": "docling",
    "visual_description": "Cutaway diagram showing the F-35's internal weapons bay..."
  },
  "asset_ref": {
    "file_path": "assets/a1b2c3d4_042_figure_01.png",
    "mime_type": "image/png",
    "width_px": 1600,
    "height_px": 800
  },
  "schema_version": "2.0.0"
}
```

**Versioning Note:** `metadata.schema_version` is always injected at export time from the central `version.py` (`__schema_version__`). JSONL output will carry the current schema version (e.g., `2.4.1-stable`) even if upstream parsers omit it.

### Spatial Metadata (REQ-COORD-02)

The `spatial` object contains normalized coordinates for UI overlay support:

| Field | Type | Description |
|-------|------|-------------|
| `bbox` | `[int, int, int, int]` | Normalized coordinates [x0, y0, x1, y1] in range 0-1000 |
| `page_width` | `int` | Original page width in PDF points (e.g., 612 for US Letter) |
| `page_height` | `int` | Original page height in PDF points (e.g., 792 for US Letter) |

**Converting bbox to pixels for UI overlay:**
```python
# Given: bbox = [100, 200, 900, 600], page_width = 612, page_height = 792
# Scale factor for rendering at 150 DPI
dpi_scale = 150 / 72  # PDF points are 72 DPI
render_width = page_width * dpi_scale  # 1275 pixels
render_height = page_height * dpi_scale  # 1650 pixels

# Convert normalized bbox to pixel coordinates
x0 = (bbox[0] / 1000) * render_width   # 127.5
y0 = (bbox[1] / 1000) * render_height  # 330.0
x1 = (bbox[2] / 1000) * render_width   # 1147.5
y1 = (bbox[3] / 1000) * render_height  # 990.0
```

### Modality Types

| Modality | Description | Has asset_ref |
|----------|-------------|---------------|
| `text` | Paragraphs, headings, lists | No |
| `image` | Figures, photos, diagrams | Yes |
| `table` | Markdown tables | Optional |

---

## 🎯 Smart Vision Orchestration

### Document Diagnostic Layer

Before processing begins, the system runs a pre-flight diagnostic:

```
━━━━━ DOCUMENT DIAGNOSTICS ━━━━━
Modality: digital
File Size: 45.2 MB
Avg Text/Page: 1250 chars
Is Likely Scan: No
Confidence: 0.92
Era: modern
Domain: editorial
Strategy: editorial_balanced
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Multi-Dimensional Profile Selection

The ProfileClassifier analyzes multiple dimensions to select the optimal processing profile:

| Profile | Use Case | VLM Freedom | Scan Hints |
|---------|----------|-------------|------------|
| `EditorialProfile` | Modern digital magazines | `discovery` | No |
| `TechnicalProfile` | Technical manuals, reports | `strict` | No |
| `ScannedModernProfile` | Clean scans (2000s+) | `guided` | Yes |
| `ScannedDegradedProfile` | Vintage scans (pre-1970) | `guided` | Yes |
| `AcademicWhitepaperProfile` | Academic papers, whitepapers | `strict` | No |

### Strategy Banner

When processing begins, you'll see the strategy configuration:

```
🎯 SMART VISION ORCHESTRATION
   📊 Document Profile: EDITORIAL
   📐 Image-to-Text Ratio: 0.72
   🎚️  Sensitivity Dial: 0.7
   📏 Min Image Dimension: 140x140px
   🖼️  Background Extraction: ENABLED
```

---

## ✅ Quality Assurance

### QA-CHECK-01: Token Balance Validation

Ensures no text is lost during chunking:

```
Source Tokens: 10,000 (from denoised text)
Chunk Tokens:  11,500 (includes DSO overlap)
Overlap Allow: 1,500 (15% estimate)
─────────────────────────────────
Effective:     10,000 (11,500 - 1,500)
Variance:      0.0% ✓ (within 10% tolerance)
```

### QA-CHECK-02: Asset Verification

Every `asset_ref.file_path` is verified to exist on disk before JSONL export.

### QA-CHECK-03: Hierarchy Integrity

`breadcrumb_path` depth must match `hierarchy.level` value.

### REQ-COORD: Coordinate Validation

- All `bbox` values must be integers in range [0, 1000]
- `page_width` and `page_height` must be populated for IMAGE and TABLE chunks
- Digital PDFs: OCR cascade is bypassed (Docling text layer + TextIntegrityScout only). Layout-aware OCR runs only for scanned/unknown modalities.
- Gap-fill recovery: Academic whitepapers use a 60-character minimum block size on low-coverage pages, with noise filters and strict deduplication to avoid duplicates/noise.

---

## 🔬 Layout-Aware OCR (Phase 1B)

### Overview

For scanned documents, the standard pipeline produces VLM summaries instead of verbatim text. The **Layout-Aware OCR** module uses a 3-layer confidence-based cascade:

```
Layer 1: Docling (existing)  →  confidence > 0.7 → accept
                             ↓
Layer 2: Tesseract 5.x       →  confidence > 0.7 → accept
                             ↓
Layer 3: Doctr (transformer) →  FINAL PASS (accept all)
```

### Auto-Detection (Default)

When `--ocr-mode auto` (default), the system automatically selects:
- **Digital documents** → `legacy` mode (shadow extraction)
- **Scanned documents** → `layout-aware` mode (3-layer OCR cascade)

```bash
# Auto-detection (recommended)
mmrag-v2 process document.pdf --batch-size 10

# Force layout-aware for vintage documents
mmrag-v2 process vintage_catalog.pdf \
  --ocr-mode layout-aware \
  --ocr-confidence-threshold 0.7 \
  --enable-doctr \
  --batch-size 10
```

---

## ⚙️ Advanced Configuration

### Environment Variables

```bash
# Vision providers
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="..."

# Ollama (if not on localhost)
export OLLAMA_HOST="http://your-server:11434"

# OpenAI-compatible APIs (LM Studio, LocalAI, vLLM)
export OPENAI_BASE_URL="http://localhost:1234/v1"
```

### Python API

```python
from mmrag_v2.batch_processor import BatchProcessor

processor = BatchProcessor(
    output_dir="./output",
    batch_size=10,
    vision_provider="openai",  # or "ollama", "anthropic", "none"
    vision_api_key="your-key",
    strict_qa=True,
    semantic_overlap=True,
)

result = processor.process_pdf("document.pdf")
print(f"Chunks: {result.total_chunks}")
print(f"Output: {result.output_jsonl}")
```

---

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: tiktoken` | `pip install tiktoken` |
| `Ollama connection refused` | Run `ollama serve` in another terminal |
| `Memory error on large PDF` | Reduce `--batch-size` (try 5) |
| `Token validation failed` | Check if ads/headers were incorrectly included |
| `No images extracted` | Increase `--sensitivity` (try 0.8) |
| `page_width/page_height is null` | Update to latest version (fixed in v2.4) |

### Debug Mode

```bash
# Verbose logging
mmrag-v2 process document.pdf --verbose

# Process single batch for testing
mmrag-v2 process document.pdf --pages 10 --batch-size 10
```

---

## 📊 Performance Benchmarks

| Document | Pages | Time | Chunks | Assets | RAM Peak |
|----------|-------|------|--------|--------|----------|
| Combat Aircraft (Aug 2025) | 244 | 8m 32s | 1,847 | 244 | 4.2 GB |
| Technical Manual | 120 | 3m 15s | 892 | 45 | 2.1 GB |
| Photo Book | 80 | 5m 48s | 423 | 156 | 3.8 GB |

*Benchmarks on M2 Max 32GB with Ollama llava:latest*

---

## 🛡️ SRS Compliance

This implementation is fully compliant with **SRS Multimodal Ingestion V2.4**:

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| IRON-01: Atomic tables/figures | ✅ | Never split across chunks |
| IRON-07: Full-Page Guard | ✅ | Area ratio > 0.95 rejected |
| QA-CHECK-01: Token validation | ✅ | TokenValidator with 10% tolerance |
| REQ-COORD-01: Normalized bbox | ✅ | Integer range [0, 1000] |
| REQ-COORD-02: Page dimensions | ✅ | page_width, page_height populated |
| REQ-CHUNK-03: DSO overlap | ✅ | SemanticOverlapManager |

---

## 🧪 Running Tests

### Test Infrastructure Overview

The test suite is organized into two categories:

1. **pytest Unit Tests** (in `tests/`) - Automated tests for CI/CD
2. **CLI Test Scripts** (in `scripts/`) - Manual testing scripts for specific features

### Running pytest Tests

```bash
# Activate conda environment
conda activate ./env

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_token_validator.py -v

# Run tests with coverage
pytest tests/ --cov=mmrag_v2 --cov-report=html

# Run only fast tests (skip slow integration tests)
pytest tests/ -v -m "not slow"
```

### Available pytest Test Suites

| Test File | Purpose | Fast/Slow |
|-----------|---------|-----------|
| `test_token_validator.py` | QA-CHECK-01 token balance validation | Fast |
| `test_strategy_profiles.py` | Profile selection and parameter isolation | Fast |
| `test_vlm_text_detection.py` | VLM text transcription detection | Fast |
| `test_domain_detection_parity.py` | Domain detection accuracy | Fast |
| `test_full_page_guard.py` | Full-page asset validation (IRON-07) | Fast |
| `test_semantic_overlap.py` | Dynamic Semantic Overlap (DSO) | Fast |
| `test_universal_pipeline.py` | End-to-end pipeline integration | Slow |
| `test_magazine_layout.py` | Magazine-specific layout processing | Slow |

### CLI Test Scripts (Manual Execution)

These scripts are NOT run by pytest and require manual execution with specific PDF files:

```bash
# Layout-Aware OCR testing
python scripts/test_layout_aware_ocr.py \
    --pdf data/raw/vintage_catalog.pdf \
    --page 21 \
    --output ./test_results

# Benchmark OCR cascade sequences
python tests/benchmark_ocr_cascade_sequences.py

# Quick Docling integration test
python tests/quick_docling_test.py
```

### Test Execution Best Practices

**Before Running Tests:**

1. Ensure conda environment is activated: `conda activate ./env`
2. Verify dependencies: `pip install -e .`
3. Check that test data exists in `data/raw/` (if required by test)

**PDF-Only Test Runs:**

To test only PDF processing without VLM/OCR overhead:

```bash
# Process PDF without vision provider
mmrag-v2 process document.pdf --vision-provider none --no-ocr

# Run tests with minimal dependencies
pytest tests/test_token_validator.py tests/test_strategy_profiles.py -v
```

**Debugging Test Failures:**

```bash
# Run single test with verbose output
pytest tests/test_token_validator.py::test_simple_text_exact_match -v -s

# Run with debugger on failure
pytest tests/test_strategy_profiles.py --pdb

# Show local variables on failure
pytest tests/ -v -l
```

### Continuous Integration

The test suite is designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Tests
  run: |
    conda activate ./env
    pytest tests/ -v --cov=mmrag_v2
```

### Test Coverage Goals

| Component | Coverage Target | Current |
|-----------|----------------|---------|
| Core Processing | 80%+ | ✅ 85% |
| Validators | 90%+ | ✅ 92% |
| Vision/VLM | 70%+ | ✅ 75% |
| OCR Engines | 70%+ | ✅ 73% |

---

## 📄 License

Internal Project - SRS v2.4 Compliant

---

**Built with ❤️ for Advanced RAG Systems**
