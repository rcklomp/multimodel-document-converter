# Changelog: MM-Converter-V2

All notable changes to this project will be documented in this file. This project adheres to the functional requirements defined in **SRS v2.4.1**.

> **Versioning note:** Historical entries before the `v2.4.x` line used an internal `v18.x` milestone scheme during rapid iteration and test/fix cycles. Only stable or decision-worthy checkpoints were recorded, so intermediate builds are intentionally omitted. From `v2.4` onward, entries follow the current public semantic line.

## [v2.5.0] - TBD (in development)

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
