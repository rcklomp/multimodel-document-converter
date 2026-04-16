# SOFTWARE REQUIREMENTS SPECIFICATION: Multimodal RAG Ingestion Engine

**Version:** v2.5.0-dev
**Schema Version:** 2.5.0-dev (single source: `src/mmrag_v2/version.py`)
**Status:** STALE — This SRS reflects v2.5.0. The codebase is at **v2.7.0**. Features added in v2.6–v2.7 (HybridChunker, multimodal validation layers, TOC-based heading hierarchy, Docling 2.86.0 picture classification, encoding heal-over, output provenance) are **not documented here**. Refer to `CHANGELOG.md` and `docs/DECISIONS.md` for current behavior. A full SRS rewrite to v2.7 is pending.
**Supersedes:** SRS v2.4.md
**Design Philosophy:** Apply Specification Engineering primitives — self-contained problem statements, explicit acceptance criteria, constraint architecture, decomposed phases, and measurable evaluation design.

---

## Document Control

| Version | Date       | Author | Changes |
|---------|------------|--------|---------|
| 2.5.0   | 2026-02-28 | System | Full rewrite applying five specification engineering primitives |
| 2.4.2   | 2026-02-25 | System | VisionCache model-awareness; visual_description top-level field; image_description_coverage QA gate |
| 2.4.1   | 2026-01-18 | System | schema_version single-source; metadata emission; digital OCR policy |

---

## HOW TO USE THIS DOCUMENT

This document is designed to be the **single source of truth** for any agent or engineer implementing, debugging, or extending this codebase. It is intentionally self-contained: all thresholds, constraints, and acceptance criteria that drive routing decisions are stated here. You should not need to read any other document to understand what "correct" means.

**Reading order for a new task:**
1. Section 1 — Problem definition and Iron Rules (invariants you must never violate)
2. Section 2 — The two-axis routing model (how the pipeline makes decisions)
3. Section 3 — The processing phases (what runs in what order)
4. Section relevant to your task (input format, chunking, VLM, schema, QA)
5. Section 10 — Acceptance criteria and definition of done

---

## 1. SELF-CONTAINED PROBLEM STATEMENT

### 1.1 What This System Does

MM-Converter-V2 is a high-fidelity ETL (Extract, Transform, Load) pipeline. It converts a heterogeneous corpus of documents (PDF, EPUB, HTML, DOCX, PPTX, XLSX) into an ingestion-optimized JSONL format for Multimodal RAG (Retrieval-Augmented Generation) systems.

**The concrete failure mode this system exists to prevent:**

```
BROKEN OUTPUT (pre-v2 behaviour — must never happen):
{"modality": "shadow", "content": "The image shows a page titled 'INTRODUCTION'..."}

CORRECT OUTPUT:
{"modality": "text",  "content": "INTRODUCTION\n\nThis manual covers..."}
{"modality": "image", "content": "Exploded view diagram of trigger assembly, 7 components"}
{"modality": "table", "content": "| Component | Part No. | Qty |\n|..."}
```

Every text region must become a `text` chunk with extracted content (not a VLM description). Every image region must become an `image` chunk with a visual-only VLM description (not a text transcription). Tables must be structured markdown, never flattened.

### 1.2 Runtime Environment (Required Context)

| Fact | Value |
|------|-------|
| Platform | Apple Silicon (ARM64 macOS) |
| Python | 3.10 exactly (`>=3.10,<3.11` in `pyproject.toml`) |
| Conda env | Prefix env `./env` is used in testing docs (`conda run -p ./env`); named env `mmrag-v2` from `environment.yml` is also valid |
| Docling pin | `==2.66.0` (exact pin in `pyproject.toml`, this is the single source of truth) |
| PyTorch backend | Prefer MPS on Apple Silicon when available (`torch.backends.mps.is_available()`) |
| Memory budget | Peak < 8 GB (system has 16 GB) |
| PDF batch size | Default `10`; operational recommendation is `<= 10` pages per batch for memory stability |
| Schema version source | `src/mmrag_v2/version.py` (`__schema_version__`) |
| Entry points | `src/mmrag_v2/cli.py` — commands: `process`, `batch`, `version`, `check` |

### 1.3 Key Source Files

| File | Role |
|------|------|
| `src/mmrag_v2/cli.py` | CLI entry — `process` and `batch` commands |
| `src/mmrag_v2/batch_processor.py` | Primary PDF orchestrator (splitting, OCR governance, filtering, token validation, deduplication, JSONL export) |
| `src/mmrag_v2/processor.py` | Non-batch element mapper (Docling → IngestionChunk) and shadow extraction |
| `src/mmrag_v2/schema/ingestion_schema.py` | Canonical chunk schema (Pydantic) |
| `src/mmrag_v2/orchestration/document_diagnostic.py` | Pre-flight structural and modality diagnostics |
| `src/mmrag_v2/orchestration/profile_classifier.py` | Weighted semantic profile selection |
| `src/mmrag_v2/orchestration/strategy_orchestrator.py` | Profile → extraction strategy parameters |
| `src/mmrag_v2/vision/vision_manager.py` | VLM provider abstraction + inference |
| `src/mmrag_v2/validators/token_validator.py` | QA-CHECK-01 token balance logic |
| `src/mmrag_v2/version.py` | Single-source schema/engine version constants |

### 1.4 Iron Rules (Inviolable Invariants)

These rules are **INVIOLABLE**. Any implementation that violates them is a **critical defect**, regardless of other quality improvements.

| Rule ID | Rule | Rationale |
|---------|------|-----------|
| **IRON-01** | Tables and Figures are atomic semantic units. MUST NOT be split across chunks. | Prevents semantic fragmentation in RAG retrieval |
| **IRON-02** | Parser MUST maintain hierarchical `ContextState` that persists across internal file boundaries (e.g., ePub chapters). | Prevents context loss and "orphan" chunks |
| **IRON-03** | Full-page image exports (screenshots containing both text and images) are STRICTLY PROHIBITED. | Text must remain parseable for retrieval |
| **IRON-04** | Non-editorial content (Ads, Navigation, Mastheads) MUST be identified and discarded. | Reduces noise in retrieval results |
| **IRON-05** | Data for each document MUST be written to disk immediately after conversion. Multiple documents in memory simultaneously is STRICTLY PROHIBITED. | Prevents OOM on 16 GB systems |
| **IRON-06** | If a document reports visual elements but the image buffer is null, processing MUST HALT and trigger a configuration audit. Silent failures are unacceptable. | Guarantees asset integrity |
| **IRON-07** | Shadow-extracted assets with `area_ratio > 0.95` require VLM verification before inclusion. Assets confirmed as UI/navigation MUST be discarded. | Prevents accidental full-page captures |
| **IRON-08** | All JSONL writes MUST use atomic append+flush operations. | Data integrity during network/process interruptions |
| **IRON-09** | Scanned documents MUST NOT produce 0 text chunks. At least one OCR pass is mandatory before concluding a region is empty. | Guarantees minimum text recovery |

**Clarification — IRON-03 vs. Shadow Extraction:**
A full-page background photo extracted via Shadow Extraction (Section 6.3) is an Atomic Asset, NOT a prohibited export, PROVIDED: (1) it is a genuine editorial image, (2) text overlaying the image remains in parseable markdown, and (3) it passes the Full-Page Guard (IRON-07).

---

## 2. THE TWO-AXIS ROUTING MODEL

PDF extraction pathway is determined by **two orthogonal axes**. Confusing them is an explicitly forbidden anti-pattern.

### 2.1 Axis 1: Structural Integrity (How to Extract)

Determined by pre-flight byte-stream tests in `DocumentDiagnosticEngine._perform_physical_check()`. This runs **before** Docling processes any page. The result drives the extraction pathway.

| Flag | Test | Threshold | Action When True |
|------|------|-----------|------------------|
| `has_flat_text_corruption` | words / newline_count ratio on 3–5 sample pages | average ratio > 50 on > 50% of pages | Run Flat Code OCR Rescue post-processing for all profiles |
| `has_encoding_corruption` | Jaccard overlap: PyMuPDF text words vs Tesseract OCR words on page 10 | overlap < 0.50 (AND PyMuPDF layer > 50 chars) | Upgrade extraction to `force_ocr = True` |
| `geometry_error_rate` | MuPDF path-syntax error count / page | No threshold — informational only | Log as risk signal; no routing action |

**Threshold constants in `document_diagnostic.py`:**
- `FLAT_TEXT_WORDS_PER_NEWLINE = 50.0`
- `FLAT_TEXT_PAGE_FRACTION = 0.5`
- `ENCODING_DELTA_MIN_OVERLAP = 0.50`

### 2.2 Axis 2: Semantic Profile (What to Describe)

Determined by `ProfileClassifier` using weighted content features (text density, image coverage, domain signals). The selected profile governs VLM prompt context, extraction sensitivity, and image thresholds. It does NOT drive the extraction pathway.

Available semantic profiles (v2.6): `academic_whitepaper`, `digital_magazine`, `scanned`, `scanned_degraded`, `technical_manual`.
Document modalities (separate axis): `native_digital`, `image_heavy`, `scanned_clean`, `scanned_degraded`, `hybrid`.
Content domains: `academic`, `editorial`, `technical`, `commercial`, `literature`, `unknown`.

### 2.3 The Routing Decision Matrix

```
                     STRUCTURAL INTEGRITY (pre-flight)
                     ┌────────────┬─────────────┬───────────────┐
                     │  HEALTHY   │ FLAT TEXT   │   ENCODING    │
                     │            │  CORRUPTED  │   CORRUPTED   │
  ───────────────────┼────────────┼─────────────┼───────────────┤
  S  native_digital  │ Docling    │ Docling +   │ Force full    │
  E                  │ direct     │ flat-code   │ OCR pathway   │
  M                  │            │ OCR rescue  │               │
  A  ────────────────┼────────────┼─────────────┼───────────────┤
  N  scanned /       │ Nuclear    │ Nuclear +   │ Force full    │
  T  image_heavy     │ OCR        │ flat rescue │ OCR pathway   │
  I  ────────────────┴────────────┴─────────────┴───────────────┘
  C
    ↑ structural axis (document_diagnostic.py)
    ↑ semantic axis (profile_classifier.py) — drives VLM/sensitivity only
```

### 2.4 Explicitly Forbidden Anti-Patterns

The following practices are treated as defects, regardless of whether they produce correct output in a specific test case:

- Using `profile_type == "technical_manual"` to decide whether OCR is needed
- Assuming `native_digital` modality means all text is correctly encoded and formatted
- Routing on filename or semantic label to determine extraction pathway
- Hardcoding document-specific rules (overfitting to specific filenames)
- Forcing `digital_magazine` as a safe fallback for scans
- Treating `metadata` as ground-truth instead of diagnostic evidence
- Cross-modality fallbacks: scanned documents must stay on the scanned path; digital must stay digital

---

## 3. PROCESSING PHASES (DECOMPOSED)

The pipeline executes in **six sequential, independently verifiable phases**. Each phase has defined inputs, outputs, and a pass criterion that can be checked without running the full pipeline.

### Phase 0: Environment Pre-Check

**Trigger:** Explicit pre-run sanity check (`mmrag-v2 check`) or CI preflight. This phase is not fully auto-enforced on every `process`/`batch` invocation.

**Inputs:** System environment.

**Tasks:**
1. Run `mmrag-v2 check` (provider/API-key readiness).
2. Confirm startup banners include the Docling engine marker (`ENGINE_USE: Docling v2.66.0`).
3. (Recommended) verify runtime Docling version manually against `pyproject.toml` pin using `python -c "import docling; print(docling.__version__)"`.

**Pass Criterion:** `mmrag-v2 check` completes successfully and no provider/configuration blockers exist for the selected run mode.

**Escalate to human if:** Docling version mismatch detected. Do not attempt to auto-upgrade the pin.

---

### Phase 1: Pre-Flight Structural Diagnosis

**Trigger:** For every PDF, before Docling processes any page.

**Inputs:** PDF file path.

**Outputs:** `PhysicalCheckResult` with fields: `has_flat_text_corruption: bool`, `has_encoding_corruption: bool`, `geometry_error_rate: float`.

**Tasks (in order):**
1. Sample 3–5 pages evenly distributed across the document.
2. For each sampled page, compute `words_per_newline = word_count / max(newline_count, 1)`.
3. If `words_per_newline > FLAT_TEXT_WORDS_PER_NEWLINE` on `> FLAT_TEXT_PAGE_FRACTION` fraction of sampled pages → set `has_flat_text_corruption = True`.
4. Render page 10 (or median page) to PIL image. Run Tesseract. Compute Jaccard overlap of word sets with PyMuPDF text layer. If overlap `< ENCODING_DELTA_MIN_OVERLAP` AND PyMuPDF text layer is `> 50` chars → set `has_encoding_corruption = True`.
5. Count MuPDF path-syntax errors via `fitz.TOOLS.mupdf_warnings()` → store as `geometry_error_rate`.
6. Log all three values at INFO level.

**Performance bound:** Must complete in `< 2 seconds` for documents up to 300 pages (excluding the Tesseract pass, which is bounded to 1 page).

**Pass Criterion:** `PhysicalCheckResult` is produced with all three fields populated (not None). Logging shows pre-flight results banner.

**Escalate to human if:** `geometry_error_rate > 5.0` per page AND `has_encoding_corruption = True` simultaneously — this combination suggests a fundamentally unreadable PDF that the pipeline cannot recover.

---

### Phase 2: Profile Classification

**Trigger:** After Phase 1, using the same PDF.

**Inputs:** PDF file path, `PhysicalCheckResult` from Phase 1.

**Outputs:** `DocumentProfile` (semantic profile type + strategy parameters).

**Tasks:**
1. Run `DocumentDiagnosticEngine` for modality detection (digital vs. scanned detection — separate from structural tests).
2. Run `SmartConfigProvider` for content feature analysis (text density, image coverage, median image sizes).
3. Run `ProfileClassifier` with weighted features → select `profile_type`.
4. Run `StrategyOrchestrator` → produce `ExtractionStrategy` parameters.

**Constraint:** Profile selection is content-first. Filename keywords are a weak hint only. Renaming a file must not change `profile_type`. Validate this with the parity test in Section 10.6.

**Pass Criterion:** `profile_type` is set and logged. Profile banner appears in CLI output. `process` and `batch` commands produce identical `profile_type` for the same document.

---

### Phase 3: Extraction

**Trigger:** After Phase 2.

**Inputs:** PDF, `PhysicalCheckResult` (structural flags), `ExtractionStrategy` (profile parameters).

**Routing logic (must be applied in this order):**

1. If `has_encoding_corruption = True` → override to `force_ocr = True` (bypass Docling text layer entirely for all pages).
2. Else if document modality is `scanned` → nuclear OCR path.
3. Else → Docling direct extraction.

**OCR Cascade Order** (when OCR is triggered): Docling internal OCR → Tesseract 5.x → Doctr. Surya-OCR and PaddleOCR are DEPRECATED and must not be used.

**OCR Confidence Threshold:** CLI default is `0.50` (`--ocr-confidence-threshold` in `process`/`batch`). Direct `BatchProcessor` instantiation defaults to `0.70`. Any change to these defaults requires before/after acceptance evidence.

**Flat Code OCR Rescue** (runs after page chunks are assembled when `has_flat_text_corruption = True`):

| Sub-requirement | Detail |
|-----------------|--------|
| Trigger condition | `chunk_type == CODE` AND `"\n" not in content` AND `len(content) > 120` |
| Crop source | `page_image` already rendered in memory for current batch page |
| OCR engine | Tesseract (psm 6 — uniform block) at 300 DPI crop |
| Acceptance condition | OCR result has `>= 2` newlines AND passes `_looks_like_code_text()` |
| Fallback | If OCR fails or produces no improvement, keep original content unchanged |

**Pass Criterion:** Output JSONL contains no `modality: "shadow"` chunks. Scanned documents produce `> 0` text chunks (IRON-09). Code chunks that were flat (no newlines, > 120 chars) have been rescued or explicitly flagged as un-rescuable.

---

### Phase 4: Post-Processing and Quality Filtering

**Trigger:** After all page chunks are assembled for a batch.

**Inputs:** Raw chunk list from Phase 3.

**Tasks (in order):**
1. **OversizeBreaker:** Split any text chunk `> 512` tokens at sentence boundaries (`.`, `!`, `?`, `\n`). Mid-word splits are fatal errors. See Section 7 for the split algorithm contract.
2. **Dense-typographic image filter:** Drop image chunks where `visual_description` contains `"no distinct non-text visuals"` (case-insensitive). These are shadow-extracted text-only regions.
3. **Full-Page Guard (IRON-07):** For shadow assets with `area_ratio > 0.95`, run VLM verification. Discard if confirmed as UI/navigation.
4. **Deduplication:** Remove chunks with `>= 80%` text similarity to a previously seen chunk.
5. **Token validation (QA-CHECK-01):** Verify `sum(chunk_tokens) ≈ total_document_tokens` within tolerance. See Section 10.2 for profile-specific tolerance values.

**Pass Criterion:** No mid-word splits in output. No `modality: "image"` chunks with `"no distinct non-text visuals"` description. Token balance within tolerance.

---

### Phase 5: JSONL Export and Asset Persistence

**Trigger:** After Phase 4.

**Inputs:** Filtered, validated chunk list.

**Tasks:**
1. Serialize each chunk to JSONL (one chunk per line).
2. Write to `{output_dir}/ingestion.jsonl` using atomic append+flush (IRON-08).
3. Save all image assets to `{output_dir}/assets/` with naming convention `[DocHash]_[PageNum]_[Type]_[Index].png`.
4. Call `gc.collect()` after each batch to release memory (REQ-PDF-05).

**Pass Criterion:** `ingestion.jsonl` is readable line-by-line. Every `asset_ref.file_path` exists on disk. All `bbox` values are `List[int]` in range `[0, 1000]`.

---

## 4. INPUT FORMAT SPECIFICATIONS

### 4.1 PDF

| Requirement ID | Requirement | Priority |
|----------------|-------------|----------|
| **REQ-PDF-01** | Detect multi-column layouts. Reading order MUST follow vertical column flow, not horizontal line scan. | MUST |
| **REQ-PDF-02** | Blocks identified as "Advertisement" by layout model or keyword/link-density analysis MUST be excluded. | MUST |
| **REQ-PDF-03** | CLI default OCR confidence trigger threshold is `0.50` (`process`/`batch`). Direct `BatchProcessor` instantiation default is `0.70`. | MUST |
| **REQ-PDF-04** | `DocumentConverter` MUST be initialized with `PdfPipelineOptions(do_extract_images=True)`. Minimum render scale: 2.0. | MUST |
| **REQ-PDF-05** | Call `gc.collect()` between batch cycles to prevent RAM saturation on 16 GB systems. | MUST |

### 4.2 EPUB

| Requirement ID | Requirement | Priority |
|----------------|-------------|----------|
| **REQ-EPUB-01** | Process content strictly in `content.opf` spine order. | MUST |
| **REQ-EPUB-02** | Remove all internal filenames, CSS classes, and hidden HTML comments before extraction. | MUST |
| **REQ-EPUB-03** | Resolve relative image paths in `src` attributes to absolute paths within the ePub container. | MUST |

### 4.3 HTML

| Requirement ID | Requirement | Priority |
|----------------|-------------|----------|
| **REQ-HTML-01** | Extract ONLY `<body>` editorial content. Discard `<nav>`, `<footer>`, `<aside>`, `<script>` tags. | MUST |
| **REQ-HTML-02** | Map `<h1>` through `<h6>` tags directly to `ContextState` hierarchy. | MUST |

### 4.4 Microsoft Office (DOCX, PPTX, XLSX)

| Requirement ID | Requirement | Priority |
|----------------|-------------|----------|
| **REQ-DOCX-01** | Map XML Styles (`Heading 1`...) to metadata hierarchy. | MUST |
| **REQ-PPTX-01** | Treat slides as pages. Slide Title = `Heading 1`. Group overlapping shapes into single `asset` image. | MUST |
| **REQ-XLSX-01** | Convert active worksheets to GitHub Flavored Markdown tables. Prune empty rows/columns. | MUST |

### 4.5 Digital OCR Policy

- Native-digital PDFs: skip layout-aware OCR cascade. Use Docling text layer plus TextIntegrityScout (PyMuPDF blocks) for recovery. No Tesseract/Doctr invocation unless structurally flagged (Phase 1).
- Scanned/unknown PDFs: layout-aware OCR (Docling → Tesseract → Doctr) applies when modality is not digital.
- `has_encoding_corruption = True` overrides the digital/scanned distinction and forces full OCR regardless.

### 4.6 Gap-Fill Recovery

- For `academic_whitepaper` profile: low-coverage pages trigger gap-fill on blocks `>= 60` characters.
- Academic noise filters and strict deduplication (`80%` similarity) prevent pulling page numbers / bibliography noise.
- Recovery chunks are marked `extraction_method: recovery_gap_fill` (or `recovery_subsurface`/`scan` as applicable).

---

## 5. SYSTEM ARCHITECTURE: STATE MACHINE

### 5.1 ContextState

```python
@dataclass
class ContextState:
    """Persistent state maintained during document parsing."""
    current_page: int = 0
    breadcrumbs: List[str] = field(default_factory=list)
    current_header_level: int = 0
    last_processed_header: Optional[str] = None
    document_type: str = "generic"  # "generic" | "periodical" | "technical"
```

| Requirement ID | Requirement |
|----------------|-------------|
| **REQ-STATE-01** | `breadcrumbs` list UPDATES ONLY when a new header is explicitly detected. |
| **REQ-STATE-02** | If new header is Level N, it replaces existing Level N and removes all deeper levels (N+1, N+2, etc.). |
| **REQ-STATE-03** | Every generated chunk MUST inherit a deep copy of the current `ContextState`. |

### 5.2 Sensitivity Parameter

The `--sensitivity` parameter (range: `0.1`–`1.0`, default: `0.5`) controls the balance between AI-driven layout analysis (Docling) and deterministic heuristic extraction (PyMuPDF shadow scan).

| Sensitivity | Heuristic Aggression | Docling Trust | Use Case |
|-------------|---------------------|---------------|----------|
| 0.1 (Strict) | None (Docling only) | 100% | Technical documents with clean layouts |
| 0.5 (Balanced) | Medium (bitmaps ≥ 50% page area) | 50% | General purpose — RECOMMENDED DEFAULT |
| 1.0 (Max Recall) | Maximum (all bitmaps ≥ 100 px) | 0% | Visually-dense editorial content |

| Requirement ID | Requirement |
|----------------|-------------|
| **REQ-SENS-01** | `min_dimension = 400px - (sensitivity * 300px)` |
| **REQ-SENS-02** | At sensitivity ≥ 0.7, enable background layer extraction for full-page editorial images with text overlays. |
| **REQ-SENS-03** | Editorial documents (image ratio ≥ 40%, avg size ≥ 300 px) receive automatic −50 px threshold reduction. |

### 5.3 Hierarchy Standard for Periodical Publications

```
Level 1: Publication Title (e.g., "Combat Aircraft Journal")
Level 2: Edition/Issue  (e.g., "August 2025")
Level 3: Section        (e.g., "Features")
Level 4: Article Title  (e.g., "Operation Rising Lion")
Level 5: Article Subsection (e.g., "First Wave")
```

| Requirement ID | Requirement |
|----------------|-------------|
| **REQ-HIER-01** | Documents classified as `periodical` MUST populate at minimum Levels 1–3 in breadcrumbs. |
| **REQ-HIER-02** | Article boundaries (Level 4) SHOULD be detected via page breaks, horizontal rules, or large heading style changes. |
| **REQ-HIER-03** | If hierarchy cannot be determined, breadcrumbs MUST contain at minimum: `[source_filename, page_number]`. |
| **REQ-HIER-04** | Breadcrumb depth MUST match `hierarchy.level` value. |

---

## 6. MULTIMODAL ASSET EXTRACTION

### 6.1 Visual Assets (Images/Charts)

| Requirement ID | Requirement |
|----------------|-------------|
| **REQ-MM-01** | Detect bounding box, apply 10 px padding, crop and save as PNG. |
| **REQ-MM-02** | Naming: `[DocHash]_[PageNum]_[Type]_[Index].png` (e.g., `a1b2c3d4_005_figure_01.png`) |
| **REQ-MM-03** | JSONL entry MUST contain `semantic_context` with `prev_text_snippet` (300 chars) and `next_text_snippet` (300 chars). Absence of a digital text layer is NOT a valid reason for null values. The OCR cascade MUST provide anchoring text. |

### 6.2 Tabular Data

| Requirement ID | Requirement |
|----------------|-------------|
| **REQ-MM-04** | Tables MUST be converted to Markdown (default) or HTML Table (if cells are merged/complex). |
| **REQ-MM-04b** | Tables are NEVER flattened to unstructured text. |

### 6.3 Shadow Extraction

Shadow extraction is a deterministic safety net for large editorial images that Docling misses (e.g., full-page background photos with text overlays).

| Requirement ID | Requirement |
|----------------|-------------|
| **REQ-MM-05** | Perform raw physical scan of PDF stream (via PyMuPDF) to identify embedded bitmap/image objects, in parallel with AI analysis. |
| **REQ-MM-06** | If bitmap detected with dimensions ≥ 300×300 px (or ≥ 40% page area) lacking a corresponding Docling `Figure`/`Picture` block → force-extract as "Shadow Asset." |
| **REQ-MM-07** | Shadow Assets MUST link to nearest `TextBlock` on current page. Metadata MUST include `extraction_method: "shadow"`. |

### 6.4 Full-Page Guard

| Requirement ID | Requirement |
|----------------|-------------|
| **REQ-MM-08** | Calculate `area_ratio = (asset_width × asset_height) / (page_width × page_height)` for every shadow asset. |
| **REQ-MM-09** | If `area_ratio > 0.95`, trigger Full-Page Guard validation. |
| **REQ-MM-10** | VLM Verification prompt: "Is this image editorial content (photo, illustration, infographic) or is it a UI element/page scan/navigation?" |
| **REQ-MM-11** | Assets identified as UI elements, full-page scans, or navigation MUST be discarded. Log rejection reason. |
| **REQ-MM-12** | CLI flag `--allow-fullpage-shadow` bypasses Full-Page Guard for known editorial-heavy documents. |

---

## 7. CHUNKING STRATEGY

### 7.1 Semantic Partitioning

| Requirement ID | Requirement |
|----------------|-------------|
| **REQ-CHUNK-01** | Splits MUST ONLY occur at sentence delimiters (`.`, `!`, `?`, `\n`). Mid-sentence splits are fatal errors — treat as a critical defect. |
| **REQ-CHUNK-02** | Text target: 400 tokens, hard max: 512 tokens. Atomic units (Tables/Figures): max 1024 tokens. |

### 7.2 OversizeBreaker Split Algorithm

The `_split_nearest_paragraph_breaks()` function implements a three-level fallback chain. **Level 1** wins if there is a `\n\n` break. Otherwise **Level 2** (single `\n`). Otherwise **Level 3** (sentence mark). The search window for candidates is `max_chars // 5` from the split target, not `max_chars // 2`. Candidates beyond `max_chars` are excluded before proximity sorting. The three levels use explicit `if split_idx is None:` fallthrough, not an `else:` block.

### 7.3 Dynamic Semantic Overlap (DSO)

| Requirement ID | Requirement |
|----------------|-------------|
| **REQ-CHUNK-03** | Trigger: `--semantic-overlap` flag. Model: `sentence-transformers/all-MiniLM-L6-v2` (v2.2.2). |
| **REQ-CHUNK-04** | Extract last 3 sentences of Chunk A, first 3 of Chunk B. If cosine similarity `> 0.85`: `overlap = base_overlap * 1.5`. |
| **REQ-CHUNK-05** | Overlap < 25% of total chunk size. |

---

## 8. VISION & VLM

### 8.1 Provider Support

| Requirement ID | Requirement |
|----------------|-------------|
| **REQ-VLM-01** | Support: Local (Ollama), Cloud (OpenAI gpt-4o-mini, Anthropic claude-3-5-haiku), Fallback (breadcrumb-based, no VLM). |

### 8.2 VisionCache

The `vision_cache.json` is model-aware. When the VLM model changes, stale cache entries are discarded automatically. This prevents cross-model contamination where a cached description from a weaker model is used by a stronger model.

| Requirement ID | Requirement |
|----------------|-------------|
| **REQ-VLM-02** | For `editorial` documents, trigger Full-Page Preview VLM analysis if: (a) detected assets < page-median, OR (b) total assets on page = ZERO. |
| **REQ-VLM-03** | Maintain `vision_cache.json` to prevent redundant VLM calls for identical image hashes. Cache is model-keyed. |
| **REQ-VLM-04** | VLM calls have configurable timeout (default: 180 s). On timeout/failure, fall back to breadcrumb-based context. |

### 8.3 Visual-Only Prompt Policy

VLM prompts MUST use `VISUAL_ONLY_PROMPT` (defined in `src/mmrag_v2/vision/vision_prompts.py`). The VLM MUST NOT transcribe text it sees in an image.

Sentinel phrase for "no visual content": `"no distinct non-text visuals"` (case-insensitive). Image chunks carrying this phrase in `visual_description` MUST be dropped in Phase 4 (dense-typographic image filter).

---

## 9. CANONICAL OUTPUT SCHEMA (JSONL)

Every line in `ingestion.jsonl` MUST validate against this schema. No other output formats are permitted.

### 9.1 Schema Definition

```json
{
  "chunk_id": "string (UUID_v4 or composite hash)",
  "doc_id": "string (12-char hex from file MD5)",
  "chunk_type": "string|null (paragraph|code|list_item|null) — @computed_field from metadata",
  "visual_description": "string|null — @computed_field from VLM result",
  "modality": "text | image | table",
  "content": "string (actual text, markdown, or VLM description — non-empty)",
  "metadata": {
    "source_file": "string",
    "file_type": "string (pdf|epub|html|docx|pptx|xlsx)",
    "page_number": "integer (1-indexed)",
    "chunk_type": "string|null (paragraph|heading|list|caption|null)",
    "hierarchy": {
      "parent_heading": "string|null",
      "breadcrumb_path": ["string"],
      "level": "integer|null (1-5)"
    },
    "spatial": {
      "bbox": "[int, int, int, int] (REQUIRED for images and tables)",
      "page_width": "integer|null",
      "page_height": "integer|null"
    },
    "extraction_method": "string (docling|shadow|ocr|recovery_gap_fill)",
    "content_classification": "string|null (editorial|technical|advertisement)",
    "ocr_confidence": "string|null (high|medium|low)",
    "created_at": "string (ISO 8601)"
  },
  "asset_ref": {
    "file_path": "string (relative path)",
    "mime_type": "string (image/png)",
    "width_px": "integer|null",
    "height_px": "integer|null",
    "file_size_bytes": "integer|null"
  },
  "semantic_context": {
    "prev_text_snippet": "string|null (max 300 chars)",
    "next_text_snippet": "string|null (max 300 chars)"
  },
  "schema_version": "string (from version.py)"
}
```

**Note on `chunk_type` and `visual_description`:** These are `@computed_field` properties on `IngestionChunk` (Pydantic). They serialize to the root-level JSON. `to_embedding_text()` must NOT duplicate the VLM description in the embedding text.

### 9.2 Coordinate System (Mandatory)

| Requirement ID | Requirement |
|----------------|-------------|
| **REQ-COORD-01** | ALL bounding boxes MUST be normalized to a 1000×1000 integer canvas. |
| **REQ-COORD-02** | `bbox` values MUST be `List[int]` with exactly 4 elements. |
| **REQ-COORD-03** | Format: `[x_min, y_min, x_max, y_max]` where all values are integers in range `[0, 1000]`. |
| **REQ-COORD-04** | PROHIBITED: Floats (0.0–1.0), raw pixel values, percentages. |
| **REQ-COORD-05** | Conversion: `int((raw_coord / page_dimension) * 1000)` |

```python
# CORRECT
"bbox": [100, 200, 500, 600]

# INCORRECT — will fail QA-CHECK-04
"bbox": [0.1, 0.2, 0.5, 0.6]   # floats
"bbox": [72, 144, 360, 432]     # raw pixels
```

### 9.3 Required vs. Optional Fields

| Field | Required | Condition |
|-------|----------|-----------|
| `chunk_id` | REQUIRED | Always |
| `doc_id` | REQUIRED | Always |
| `modality` | REQUIRED | Always |
| `content` | REQUIRED | Always (non-empty string) |
| `metadata.source_file` | REQUIRED | Always |
| `metadata.file_type` | REQUIRED | Always |
| `metadata.page_number` | REQUIRED | Always |
| `metadata.extraction_method` | REQUIRED | Always |
| `metadata.created_at` | REQUIRED | Always |
| `metadata.hierarchy.breadcrumb_path` | REQUIRED | Always (may be empty list `[]`) |
| `metadata.spatial.bbox` | REQUIRED | `modality` is `image` or `table` |
| `asset_ref` | REQUIRED | `modality` is `image` or `table` |
| `asset_ref.file_path` | REQUIRED | When `asset_ref` is present |
| `schema_version` | REQUIRED | Always |
| All other fields | OPTIONAL | May be null |

### 9.4 `intelligence_metadata` Unpacking Contract

The three structural flags (`has_flat_text_corruption`, `has_encoding_corruption`, `geometry_error_rate`) MUST be `pop()`-ed from the `intelligence_metadata` dict before it is `**unpacked` into `create_text_chunk()`. They are stored as dedicated instance variables (`self.has_flat_text_corruption`, etc.) on `BatchProcessor`. Failure to do this causes a `TypeError` at runtime.

---

## 10. ACCEPTANCE CRITERIA AND DEFINITION OF DONE

### 10.1 Definition of Done

A task (bug fix, feature, or refactoring) is considered **complete** only when ALL of the following are true:

1. The full acceptance gate reports `GATE_PASS` (not just individual QA checks).
2. The blind-test PDF `Greenhouse Design and Control by Pedro Ponce.pdf` is included in the acceptance run and passes.
3. All mandatory gates in Section 10.2 pass.
4. No regression against the profile-specific metric targets in Section 10.4.
5. `process` and `batch` commands produce identical `profile_type` for the same document (parity test passes).

### 10.2 QA Gate Definitions (Pass/Fail)

These are machine-checkable gates. A run that does not meet all of them is a failed run.

| Gate ID | Validation Rule | Pass Condition |
|---------|-----------------|----------------|
| **QA-CHECK-01** | Token balance: `sum(chunk_tokens) ≈ total_document_tokens` | Variance within profile-specific tolerance (see Section 10.3) |
| **QA-CHECK-02** | Asset integrity: every `asset_ref.file_path` exists on disk | Zero missing files |
| **QA-CHECK-03** | Hierarchy integrity: `breadcrumb_path` depth matches `hierarchy.level` | Zero mismatches |
| **QA-CHECK-04** | Coordinate integrity: all `bbox` values are integers in `[0, 1000]` | Zero violations |
| **QA-CHECK-05** | Asset reference completeness: no `modality: "image"` or `modality: "table"` chunk with missing `asset_ref` | Zero violations |
| **QA-IMAGE-01** | Image description coverage: `>= 80%` of image chunks have a non-null `visual_description` | `coverage >= 0.80` |

### 10.3 QA-CHECK-01 Token Tolerance Policy

| Profile | Tolerance | Gate Decision | Notes |
|---------|-----------|---------------|-------|
| All profiles (standard) | 10% (`0.10`) | Pass | Baseline and end-state target |
| `digital_magazine` (temporary waiver) | 18% (`0.18`) | Pass (debt waiver) | Allowed for known text-in-graphics debt; record debt note and remediation follow-up |
| `digital_magazine` above 18% | > 18% | FAIL | Out of tolerance; no further waiver |
| Any non-`digital_magazine` above 10% | > 10% | FAIL | No waiver allowed |

**Waiver sunset intent:** The `digital_magazine` waiver exists to keep current runs operational. Target state is `<= 0.10` for all profiles. Retire the waiver when representative `digital_magazine` acceptance runs consistently stay within 10%.

### 10.4 Profile-Specific Quality Targets (Guidance, Used for Regression Detection)

These are guidance defaults. Deviations require documented before/after evidence and explicit justification.

**`technical_manual`:**
- `infix_strict = 0` (zero mid-sentence list-number artifacts)
- `micro_non_label_ratio <= 0.22`
- `oversize_ratio <= 0.02`
- `orphan_label_ratio <= 0.30`

**`digital-like` profiles (`digital_magazine`, `academic_whitepaper`, `standard_digital`):**
- `micro_non_label_ratio <= 0.12`
- `oversize_ratio <= 0.01`
- `orphan_label_ratio <= 0.20`
- `code_fragmentation_ratio <= 0.05`

**`scanned_degraded`:**
- Higher short-chunk ratio is acceptable
- Require: clean control characters, stable token balance, artifact suppression

### 10.5 Acceptance Workflow

```bash
# Step 1: Run acceptance on representative documents
bash scripts/acceptance_technical_manual.sh

# Step 2: Evaluate gates
python scripts/evaluate_technical_manual_gates.py output/<run_name>/ingestion.jsonl --doc-class auto

# Step 3: Check QA hygiene metrics
python scripts/qa_ingestion_hygiene.py output/<run_name>/ingestion.jsonl

# Step 4: Check semantic fidelity (image coverage gate)
python scripts/qa_semantic_fidelity.py output/<run_name>/ingestion.jsonl

# Step 5: Verify GATE_PASS in output
# A run is passing ONLY if output contains explicit "GATE_PASS"
# "GATE_FAIL" or absence of "GATE_PASS" = failing run
```

Look for the explicit string `GATE_PASS` or `GATE_FAIL` in acceptance output summaries.

### 10.6 Regression Test Matrix

The following combinations must pass for any change to be considered non-regressive:

| Test Scenario | Document Type | Expected Result |
|---------------|---------------|-----------------|
| Digital PDF, high text density | `academic_whitepaper` | TEXT chunks with high confidence, zero OCR escalation |
| Scanned PDF, firearms manual | `technical_manual` | TEXT chunks from OCR (not VLM summaries), no `modality: "shadow"` |
| Flat-code PDF (broken PDF generator) | `technical_manual` or `academic_whitepaper` | Code chunks rescued with `>= 2` newlines by flat-code OCR rescue |
| Image-heavy editorial | `digital_magazine` | IMAGE chunks with visual-only descriptions |
| Mixed digital/scanned hybrid | any | No cross-modality fallback; no breadcrumb state loss |
| Filename independence | any document renamed to `doc1.pdf` | Identical `profile_type` before and after rename |
| Memory test | 244-page document | Peak memory `< 8 GB` |
| Blind-test PDF | `Greenhouse Design and Control by Pedro Ponce.pdf` | `GATE_PASS` |

---

## 11. ERROR HANDLING AND ESCALATION

### 11.1 Error Handling Requirements

| Requirement ID | Requirement |
|----------------|-------------|
| **REQ-ERR-01** | Use `try-except` block per file. A single corrupt file MUST NOT crash a batch. |
| **REQ-ERR-02** | Log all errors to `ingestion_errors.log` with timestamp, filename, and full stack trace. |
| **REQ-ERR-03** | On startup, log Docling version and verify it matches the exact pin in `pyproject.toml`. |

### 11.2 Human Escalation Triggers

The agent MUST stop and surface the following conditions to a human rather than attempting autonomous resolution:

| Trigger | Reason |
|---------|--------|
| Docling version mismatch between runtime and `pyproject.toml` | Auto-upgrading the pin risks breaking the pipeline globally |
| `geometry_error_rate > 5.0` per page AND `has_encoding_corruption = True` | Document may be fundamentally unreadable; OCR rescue is unlikely to help |
| QA-CHECK-01 failure on a non-`digital_magazine` profile after two attempted fixes | May indicate schema drift or a systemic chunking regression |
| Any IRON Rule violation that cannot be fixed without changing the schema | Schema changes require explicit versioning and migration planning |
| `asset_ref.file_path` missing for > 5% of image chunks after Phase 5 | May indicate a storage or path resolution failure beyond normal error handling |

---

## 12. DEPRECATED DEPENDENCIES

| Dependency | Replacement | Do Not Use |
|------------|-------------|------------|
| `surya-ocr` | Docling internal models | ❌ |
| `paddleocr` | Not compatible with Apple Silicon | ❌ |
| `numpy >= 2.0.0` | Breaks Docling/PyTorch compatibility | ❌ |

**Dependency pinning rule:** `pyproject.toml` (`[project].dependencies`) is the single source of truth. `docling` MUST be exact-pinned with `==`. `environment.yml` MUST use `pip install -e .` and MUST NOT introduce a conflicting Docling version.

---

## 13. KNOWN WARNINGS (NON-BLOCKING)

These warnings are observed in normal operation and are not defects:

| Warning | Cause | Action |
|---------|-------|--------|
| `ocrmac NoneType iterable` | Docling OCR stage fails intermittently; pipeline falls back to layout-aware OCR and completes | Not blocking; track separately if OCR quality degrades |
| Token variance warning (QA-CHECK-01) | Token balance around −15% seen in some runs | Apply tolerance policy (Section 10.3); treat `digital_magazine` between 10–18% as temporary debt |
| `imagehash not installed, using fallback hash` | Optional dependency for stronger dedupe | Not required for correctness; install if collision rate is a concern |

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Flat Code OCR Rescue** | Post-processing step that re-extracts code chunks with no newlines via Tesseract, triggered by `has_flat_text_corruption` |
| **Full-Page Guard** | Validation mechanism preventing accidental capture of full-page UI/navigation elements (IRON-07) |
| **Iron Rules** | Inviolable design mandates (Section 1.4) |
| **Structural Diagnostic Router** | The pre-flight byte-stream test system that drives extraction pathway independently of semantic profile |
| **Two-Axis Model** | Structural integrity (how to extract) × Semantic profile (what to describe) |
| **UIR** | Universal Intermediate Representation — format-agnostic document structure |
| **Visual-Digital Delta** | Jaccard overlap between PyMuPDF text words and Tesseract OCR words used to detect encoding corruption |
| **Breadcrumbs** | Hierarchical path representing a chunk's position in document structure |
| **VisionCache** | Model-keyed cache preventing redundant VLM calls for identical image hashes |
| **GATE_PASS / GATE_FAIL** | Explicit string tokens in acceptance script output that determine run success |

---

## Appendix B: Diagnostic Command Reference

```bash
# Check environment
conda run -p ./env mmrag-v2 check

# Single document
conda run -p ./env mmrag-v2 process data/raw/<file>.pdf \
  --output-dir output/<run_name> \
  --vision-provider none

# With batch processing
conda run -p ./env mmrag-v2 process data/raw/<file>.pdf \
  --batch-size 10 \
  --output-dir output/<run_name>

# Force a specific profile
conda run -p ./env mmrag-v2 process data/raw/<file>.pdf \
  --profile-override technical_manual \
  --output-dir output/<run_name>

# Batch directory
conda run -p ./env mmrag-v2 batch data/raw \
  --pattern "*.pdf" \
  --output-dir output/<run_name> \
  --vision-provider none

# Check JSONL profile distribution
jq -r '.metadata.profile_type' output/<run_name>/ingestion.jsonl | sort | uniq -c

# Check for null metadata (signals missing intelligence metadata)
jq -r 'select(.metadata.profile_type == null or .metadata.min_image_dims == null) | .chunk_id' \
  output/<run_name>/ingestion.jsonl

# Run tests
pytest tests/ -v
ruff check src tests
mypy src/mmrag_v2
```

---

**END OF DOCUMENT**
