# Changelog: MM-Converter-V2

All notable changes to this project will be documented in this file. This project adheres to the functional requirements defined in **SRS v2.4.1**.

> **Versioning note:** Historical entries before the `v2.4.x` line used an internal `v18.x` milestone scheme during rapid iteration and test/fix cycles. Only stable or decision-worthy checkpoints were recorded, so intermediate builds are intentionally omitted. From `v2.4` onward, entries follow the current public semantic line.

## [v2.7.0] - 2026-04-16

### Added
- **4 multimodal validation layers** replacing heuristic-loop patching:
  1. **CorruptionInterceptor:** Per-bbox OCR patching for `/C211`-class encoding
     artifacts. Renders only the corrupted chunk's bbox at 300 DPI, runs Tesseract,
     replaces text if OCR is cleaner. Preserves HybridChunker structure.
  2. **POS Boundary Logic:** Merges trailing prepositions (`BY`, `FOR`, `OF`, `WITH`,
     `von`, `für`, `van`, `voor`, `par`, `pour`) into next chunk when it starts with
     a proper noun. Multilingual. Same-page guard prevents cross-page false merges.
  3. **Vision-Gated Hierarchy:** When a page has cover/logo/illustration images
     (detected via VLM description), demotes non-chapter headings to "Front Matter".
     Uses multimodal signals, not text patterns.
  4. **Content-Type Classification:** Chunks with 2+ boilerplate markers (ISBN, ©,
     "All rights reserved", "Printed in") get `search_priority` downgraded to `low`.
- **PyMuPDF image extraction for digital PDFs (I10):** Route image extraction by
  document classification — `native_digital` uses `page.get_images()` for clean
  embedded photo objects directly from the PDF stream. Scanned docs remain on
  Docling layout model. Combat Aircraft: 336 → 109 images.
- **Docling picture classification:** Enabled `DocumentFigureClassifier-v2.5`
  (new in Docling 2.86.0). Deny filter rejects `full_page_image` and
  `page_thumbnail` layout artifacts. Disabled for scanned docs (classifier hangs
  on large books with hundreds of image regions).
- **Magazine TOC extraction from page content (I1):** When PDF has no bookmarks,
  scans pages 1–15 for `NUMBER TITLE` patterns and builds article page ranges.
  Both Combat Aircraft and PCWorld now AUDIT_PASS with 98–100% heading coverage.
- **TOC-based heading hierarchy:** PyMuPDF `get_toc()` extracts bookmarks before
  batching; assigns correct breadcrumb hierarchy (Book > Part > Chapter > Section).
  HybridChunker headings validated against TOC — stale headings from batch
  boundaries detected and replaced.
- **Output provenance (I7):** `pipeline_version`, `source_file_hash` (SHA-256),
  and `config_hash` added to `IngestionMetadata`. `qa_conversion_audit.py` now
  warns on version drift.
- **Heading quality checks (I8):** `qa_conversion_audit.py` detects suspicious
  headings misclassified by Docling ("This chapter covers", "Listing X.Y",
  "Figure X.Y", "(continued)"). Heading fragmentation metric added as advisory.
- **Heading validation:** `is_valid_heading()` rejects >80 chars, multi-sentence,
  captions/listings. `_sanitize_chunk_for_export()` is the final heading gate
  before JSONL write.
- **`search_qdrant.py` enhancements:** `--stats`, `--page`, `--json` modes and
  partial collection name matching.

### Fixed
- **Encoding corruption: heal-over, not fail-over.** Instead of disabling
  HybridChunker for encoding-corrupted docs (losing structural metadata), the
  refiner now cleans ALL chunks at `threshold=0.0`, preserving heading hierarchy
  and table structures while fixing glyph placeholders.
- **POS merger same-page guard:** Only merge prepositions on same or adjacent pages.
  Prevents cross-page false merges (e.g. "BY ILLUSTRATIONS BY Mary GrandPré").
- **Missing `re` import in boilerplate classifier:** `NameError` silently dropped
  ALL text chunks (Harry Potter: 414 → 0).
- **Caption/marker heading rejection (I9):** Reject "Listing X.Y", "Figure X.Y",
  "Table X.Y", "Example X-Y.", "(continued)" from heading classification.
- **VLM timeout for all providers (I3):** `_post_with_deadline()` moved to
  module-level; applied to OpenAI, Ollama, and Anthropic providers. Hard
  thread-based deadline prevents infinite hangs.
- **Docling config sync:** `batch_processor.py` aligned with `processor.py`:
  `generate_picture_images=True`, `generate_table_images=False`,
  `do_cell_matching=False`, `TableFormerMode.ACCURATE`.
- **CLI `--no-refiner` override:** Config file no longer overrides explicit flag.
- **PyMuPDF image filtering:** Area threshold tuned, solid-color placeholders
  skipped, full-page backgrounds rejected, orphan asset cleanup after JSONL
  finalization.
- **Image extraction routing:** ALL extraction routed back to Docling after
  PyMuPDF proved unreliable for magazines (composite layouts) and academic papers
  (vector figures). PyMuPDF method retained for future use.
- **`NameError` fixes:** `doc` not defined in `_process_element_v2`;
  `_doc_mod` undefined in `_extract_embedded_images` — caused silent 0-chunk
  and 0-asset failures.

### Changed
- **Docling upgrade 2.66.0 → 2.86.0:** Enables `PdfTextCell.font_name` for
  font-based heading classification and `docling-hierarchical-pdf` compatibility.
- Removed dead code: `_filter_blank_images` (0 callers), redundant 150-char
  heading downgrade in `processor.py` (I4, I5).
- Gate script: skip infix counting for code chunks, skip orphan ratio when
  label count < 5.
- Version bumped to `2.7.0` in `version.py` and `pyproject.toml`.

---

## [v2.6.0] - 2026-04-07

### Added
- **Profile consolidation (7→5):** Merged `scanned_clean`, `scanned_literature`,
  `scanned_magazine` into single `scanned` profile. Kept `digital_magazine`,
  `academic_whitepaper`, `technical_manual`, `scanned_degraded` (all have dedicated
  batch_processor behavior).
- **Literature domain detection:** New `ContentDomain.LITERATURE` with dialogue-ratio
  heuristic. Correctly identifies novels/fiction (Harry Potter: `domain=literature`).
- **Multi-format support:** HTML, EPUB (auto-extract to HTML), DOCX, PPTX, XLSX via
  Docling. Batching remains PDF-only.
- **VLM diagram auto-detection:** Light heuristic classifies images as diagram vs
  photograph for prompt selection. Domain-aware: literature bypasses diagram prompt.
- **Auto-OCR for scanned documents:** OCR automatically enabled when diagnostics
  detect a scan. Previously required explicit `--enable-ocr` flag.
- **Code hygiene for all profiles:** Code detection, reclassification, and reflow
  now runs for all profiles (was technical_manual only). Fixes flat code in academic
  papers (AIOS: `code_flat_ratio` 1.0 → 0.04).
- **Decorative heading normalization:** Spaced-out headings like
  "C H A P T E R  O N E" collapsed to "CHAPTER ONE" in content and breadcrumbs.
- **OCR table markdown formatting:** Layout-aware OCR table regions now produce
  pipe-separated markdown instead of raw garbled text.
- **Makefile:** `make test`, `make lint`, `make smoke`, `make acceptance`, etc.
- **Gitea CI:** `.gitea/workflows/ci.yml` — lint + test on push/PR.
- **HybridChunker integration:** Replaced custom 30-pass text chunking pipeline
  with Docling's built-in HybridChunker (sentence-aware, structure-aware).
  Text chunks now get proper heading hierarchy from document structure.
- **VLM page transcription:** Scanned documents use VLM to transcribe text
  directly from page images instead of OCR. Produces clean text on degraded scans.
- **Config file:** `~/.mmrag-v2.yml` provides VLM + refiner defaults.
  No more typing 6 flags per conversion.
- **Qdrant tools:** `ingest_to_qdrant.py` (ingestion), `search_qdrant.py`
  (semantic search with reranking), `validate_qdrant.py` (RAG readiness check).
- **Refiner degraded scan patterns:** Detects merged words, symbol noise,
  accent artifacts. Threshold 0.0 for scanned_degraded (all OCR text refined).

### Fixed
- **Token variance waiver retired:** IMAGE-bbox-aware source text extraction excludes
  chart/graph label text from PyMuPDF baseline. PCWorld -20.2% → -8.9%. All profiles
  use standard 10% tolerance (18% digital_magazine waiver removed).
- **Harry Potter classification:** `technical_manual` → `digital_magazine` via
  decorative image density guard (>5.0 images/page penalizes technical_manual).
- **Gate thresholds:** Infix "and"/"or" conjunction guard; orphan label 0.20 → 0.25
  for digital docs (accommodates literature chapter headings).

### Changed
- 11 non-pytest scripts moved from `tests/` to `tests/manual/`.
- Data directories: `scanned_clean/`, `scanned_magazine/`, `standard_digital/` removed;
  `scanned_literature/` renamed to `scanned/`.

## [v2.5.0] - 2026-03-01 (stable)

### Added
- **Structural Diagnostic Router** (`document_diagnostic.py`): Three hardware-level
  pathology tests run on 3–5 sample pages before any extraction begins:
  - **Test 1 — Line-Break Health:** Measures words-per-newline ratio. A ratio > 50
    consistently across pages means the PDF generator stripped all newlines from the
    character stream, corrupting code blocks and structured text.
    Flags: `has_flat_text_corruption = True` → triggers Flat Code OCR Rescue.
  - **Test 2 — Visual-Digital Delta:** Renders one page as image, runs Tesseract,
    compares word-set overlap with the PyMuPDF text layer. Overlap < 50% means the
    digital text layer is encoding-garbage (CIDFont, broken character maps) and
    cannot be trusted at all.
    Flags: `has_encoding_corruption = True` → forces full OCR pathway.
  - **Test 3 — Geometry Error Rate:** Captures MuPDF path-syntax error count per page.
    Signals a broken PDF compiler (cosmetic for extraction, but useful for risk
    logging and downstream triage).
    Adds: `geometry_error_rate` to `PhysicalCheckResult`.
- **Flat Code OCR Rescue** (`batch_processor.py`): Post-processing step for all
  profiles (not just `scanned_degraded`) that detects CODE chunks with suspiciously
  flat content (no `\n`, length > 120 chars), renders the page crop via PyMuPDF, runs
  Tesseract on the crop, and replaces the content if the OCR result is better
  structured. Directly fixes the Kimothi-class broken-PDF-generator problem.
- **`_looks_like_code_text` flat-code extension**: Single-line branch now searches
  for Python keywords anywhere in a long flat string (not just at `^` anchors),
  catching code that had its newlines stripped by a broken PDF generator.

### Fixed
- **`intelligence_metadata` structural flags TypeError:** `has_flat_text_corruption`,
  `has_encoding_corruption`, and `geometry_error_rate` were placed inside the
  `intelligence_metadata` dict in `cli.py`, which is later `**`-unpacked into
  `create_text_chunk()` / `create_image_chunk()` / `create_table_chunk()`. Those
  functions don't accept those keyword arguments, causing a `TypeError` at runtime.
  Fixed in `BatchProcessor.set_intelligence_metadata()`: the three keys are now
  `pop()`-ed into dedicated instance variables (`self.has_flat_text_corruption` etc.)
  before the dict is stored, keeping the dict clean for `**`-unpacking.
- **`chunk_type` invisible to downstream tools:** `IngestionChunk` now exposes
  `chunk_type` as a Pydantic `@computed_field`, surfacing `metadata.chunk_type` at
  the root level of every serialised text chunk. Tools reading
  `chunk["chunk_type"]` no longer get `None`. Image and table chunks return `null`.
- **OversizeBreaker mid-word hard cuts (three-layer bug):** `_split_nearest_paragraph_breaks()`
  was producing 73 mid-word 1500-char hard cuts per run due to three independent defects:
  1. `max_chars // 2` guard discarded valid sentence breaks near the target —
     lowered to `max_chars // 5`.
  2. `p_after` / `n_after` positions beyond `max_chars` could win on proximity and
     then get hard-capped to 1500 — excluded when `> max_chars`.
  3. `if candidates: … else:` structure blocked the `\n` / sentence-mark fallthrough
     whenever a `\n\n` existed below the threshold — restructured to an explicit
     `if split_idx is None:` fallthrough chain (Level 1: `\n\n` → Level 2: `\n` →
     Level 3: sentence mark). Result: 73 mid-word hard cuts reduced to 0 (2
     remaining are genuine sentence-boundary splits at `.`).
- **Dense-typographic zero-value image chunks:** A new `_filter_no_visual_images()`
  post-processing pass (wired after `_apply_oversize_breaker`) drops image chunks
  whose `visual_description` contains the sentinel phrase
  `"no distinct non-text visuals"`. These are shadow-extracted text-only regions
  where the VLM fallback (after two `text_reading_detected` rejections in
  `vision_manager.py`) returns this phrase — they add no visual signal to the RAG
  index.

### Changed
- `PhysicalCheckResult` extended with `has_flat_text_corruption: bool`,
  `has_encoding_corruption: bool`, `geometry_error_rate: float`.
- OCR pathway decision in `batch_processor.py` now reads `has_encoding_corruption`
  in addition to the existing `is_likely_scan` flag.

---

## [v2.4.2] - 2026-02-25 stable

### Fixed
- **VLM cache silent reuse (critical):** `VisionCache` is now model-aware. At load
  time it reads the `_model` key embedded in the cache JSON. If the configured model
  differs from the cached model, the stale entries are discarded and fresh VLM calls
  are made. An INFO-level log message explains the decision so the user always knows
  whether VLM is being called or served from cache.
- **`visual_description` invisible to downstream tools:** `IngestionChunk` now exposes
  `visual_description` as a Pydantic `@computed_field`, surfacing `metadata.visual_description`
  at the root level of every serialised image chunk. Tools reading
  `chunk["visual_description"]` no longer get `None`.
- **Embedding text duplication:** `to_embedding_text()` no longer appends
  `[Visual: …]` after `content` for image chunks. Since `content` for image chunks
  already IS the VLM description, appending it a second time was inflating embedding
  vectors with no benefit.

### Added
- **`image_description_coverage` QA gate** (`qa_semantic_fidelity.py`): Explicit
  metric counting image chunks that carry a non-null visual description. Gate fails
  if coverage < 80 %, making VLM description regressions automatically detectable.

### Changed
- Version bumped `2.4.1-stable` → `2.4.2-stable` in `version.py` and `pyproject.toml`.
- Docs: historical planning artefacts moved from `docs/` root to `docs/archive/`.

---

## [v2.4.1] - 2026-01-18 stable

### Changed
- **No more wasted time:** Digital PDFs are detected and the OCR cascade is skipped, speeding up runs without sacrificing quality.
- **The safety net:** TextIntegrityScout now hunts for hidden code/tables and rescued dozens of high-value blocks in stress tests that other parsers missed.
- **Smart accounting:** Token balance checks won’t raise alarms when variance is within the profile’s noise allowance—academic papers get a green light when the gap is just expected noise.
- **Strict versioning:** Every chunk now carries a schema version from a single source of truth, so downstream systems always know which logic produced the data.

## [v2.4] - 2026-01-16

### Fixed
- **Parity bug (process vs batch profile mismatch):** domain detection now prioritizes content features with a low-weight filename hint, preventing filename-renaming from flipping `profile_type`.
- **Batch parity diagnostics:** added combined content/filename score logging in domain detection for traceability.
- **Fail-fast intelligence metadata wiring:** batch path raises if the intelligence stack returns `None` values instead of silently falling back.

## [v18.2] - 2026-01-15 (internal milestone)

### Added
- **Semantic Text Refiner:** Introduced the post-OCR refinement layer with staged processing (diagnostic triage -> contextual refinement -> integrity validation).
- **Refiner CLI controls:** Added operational flags for enabling/tuning refiner providers and endpoints (`--enable-refiner`, `--refiner-provider`, `--refiner-model`, `--refiner-base-url`).

### Changed
- **Non-destructive refinement storage:** Refined output is recorded in `metadata.refined_content`, while original `content` remains preserved.
- **Integrity guardrails:** Protected-token handling and edit-budget validation were enforced to prevent aggressive LLM rewrites.

## [v18.1.1] - 2026-01-13

### Added
- **Cluster B Governance**: Activated `QA-CHECK-01` (Token Validation) to prevent downstream RAG failures caused by over-length text chunks.
- **Full-Page Guard**: Implemented intelligent labeling for page-spanning elements (`[0,0,1000,1000]`) to reduce visual noise in VLM descriptions.
- **Strict OCR Governance**: The `--enable-ocr` flag is now strictly enforced across the entire extraction cascade, including fallback scenarios.

### Fixed
- Resolved coordinate mismatch between JSONL metadata and physical asset crops.
- Eliminated "null leakage" in spatial metadata for text and table chunks.

### Changed
- **Bbox/Crop Paradox Fix**: Complete overhaul of the coordinate transformation chain (Denormalization -> Scaling -> Cropping) for resolution-independent asset extraction.
- **Dynamic Scaling**: Automatic detection of render resolution (DPI) to prevent "crop drift" across diverse PDF sources.
- **Metadata Integrity**: `page_width` and `page_height` are now "sticky" and attached to every chunk at creation time (Resolves REQ-COORD-02).
- **Deferred Saving**: Images are now written to disk only after validation, effectively eliminating "orphan" PNG files.

## [v18.1] - 2026-01-11

### Added

* **JoyCaption VLM Integration:** Full implementation of `llama-joycaption-beta-one` via OpenAI-compatible API (LM Studio) for high-fidelity visual descriptions.
* **Asynchronous Batch Processing:** Decoupled the VLM inference from the text extraction pipeline. The processor now fills a queue, increasing throughput for text-heavy documents by 3x.
* **VLM Contextual Awareness:** Implementation of a 3-page sliding window for text-context injection into image prompts, significantly improving entity recognition (e.g., identifying the "USS Abraham Lincoln" via nearby captions).

### Fixed

* **REQ-COORD-02 (Spatial Anchor Fix):** Resolved the critical bug where `page_width` and `page_height` were returned as `null`. All assets now include correct physical page dimensions (612x792 for standard PDF points).
* **Metadata Sanitization:** Added post-processing filters to remove LLM internal monologues (e.g., `<think>` tags) from the final JSONL output.
* **Path Normalization:** Improved handling of relative asset paths, ensuring the `ingestion.jsonl` remains portable across different environments.

### Changed

* **Strategy Orchestrator Tuning:** Refined the `High-Fidelity` strategy with a balanced `Sensitivity: 0.5` setting, optimized for complex magazine layouts (validated against *Combat Aircraft*).
