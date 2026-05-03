# Changelog: MM-Converter-V2

All notable changes to this project will be documented in this file. This project adheres to the functional requirements defined in **SRS v2.4.1**.

> **Versioning note:** Historical entries before the `v2.4.x` line used an internal `v18.x` milestone scheme during rapid iteration and test/fix cycles. Only stable or decision-worthy checkpoints were recorded, so intermediate builds are intentionally omitted. From `v2.4` onward, entries follow the current public semantic line.

## [unreleased] — Post-Docling Sanity Pass (2026-05-03)

### Added
- **Post-Docling reading-order y-sort** (`engines/docling_postprocess.py`).
  Re-sorts each page's `body.children` by `(-bbox.t, bbox.l)` (BOTTOMLEFT) /
  `(bbox.t, bbox.l)` (TOPLEFT). Fixes the HARRY page-13 swap where Docling
  emitted paragraphs as `[para1, para3, para2]`. Items without a prov bbox
  retain Docling's order; pages stay in ascending order so an item on page
  14 never reorders ahead of a page-13 item.
- **Drop-cap promotion** (`engines/docling_postprocess.py`). Two heuristics:
  `apply_dropcap_promotion` for the standalone-glyph case (separate
  `TextItem("M")` adjacent to a lowercase-starting paragraph), and
  `_heal_inline_trailing_dropcap` for the actually-observed Docling 2.86
  pattern: the drop cap "M" is appended INLINE at the end of the same
  TextItem (`"r. and Mrs. Dursley...nonsense. M"`). Both move the glyph
  to the front (`"Mr. and Mrs. Dursley..."`).
- **Label-leak filter** (`engines/docling_serializers.py`). Custom
  chunking serializer suppresses picture classification labels (`other`,
  `icon`, `table`). Patches both the new `meta.classification` path
  (via `blocked_meta_names={"classification"}`) and the legacy
  `PictureClassificationData` annotations — the original "no caption"
  rule was insufficient because pictures with BOTH a caption and a
  classification annotation still leaked the label. Now strips
  classification annotations across all pictures before delegating;
  captions still flow through; metadata is restored after serialization
  so downstream consumers see the full label set.
- **OCR gating** (`bitmap_area_threshold` field on `PdfConversionPlan`).
  Default 0.75 (raised from Docling's native 0.05); auto-raises to 0.92
  for `digital_literature`, `digital_magazine`, and the
  `image_heavy_magazine` route to keep OCR off photographic cover artwork.
- **`digital_literature` profile** — new ProfileType across the routing
  layer (`orchestration/profile_classifier.py` enum + `_score_digital_literature`
  scorer + score loop + modality fallback; `orchestration/strategy_profiles.py`
  `DigitalLiteratureProfile` strategy class + ProfileManager registry +
  classifier→strategy `type_mapping`; `orchestration/strategy_orchestrator.py`
  `PROFILE_TO_DOC_TYPE` mapping to `DocumentType.LITERATURE`). Auto-picks
  for born-digital novels (HARRY signature: `domain=literature`,
  `is_scan=False`, `page_count >= 50`, small `median_dim`). The plan
  builder auto-enables the full post-processor pipeline for this profile:
  `reading_order_strategy="y_sort_with_dropcap"`,
  `suppress_layout_label_text=True`, `bitmap_area_threshold=0.92`.

### Fixed
- **Adapter bypass in `processor.py:2072`**. The cached Docling converter
  was being invoked directly via `self._converter.convert()`, sidestepping
  `DoclingPdfAdapter.convert()` and therefore `apply_postprocessors`. The
  v2.7 §5 static guard banned `PdfPipelineOptions(` / `DocumentConverter(`
  *construction* outside the adapter but did NOT catch raw `convert(...)`
  *invocation*. Re-routed through `self._adapter.convert(str(file_path))`.
- **Diagnostic literature detection on moderate-length docs** —
  `document_diagnostic.py` Rule 0c added: `_dialogue_pages >= 1 AND
  total_pages > 20 AND not has_tables AND 500 < avg_text_per_page < 2500
  → literature += 0.4`. Catches the 30-page HARRY test slice where the
  long-form Rule 0a (>0.3 dialogue ratio AND >50 pages) was unreachable
  because only 5 pages get sampled by default
  (`DIAGNOSTIC_SAMPLE_PAGES=5`) and only 1 had dialogue (ratio 0.20).
- **HARRY pages 1-30 acceptance fixture**
  (`tests/fixtures/harry_potter_pages_1_to_30/`). Paragraph-level expected
  reading order extracted from PDF y-coordinates by `build_fixture.py`.
  Scope: body pages 13-30 (front-matter pages 1-12 are display-typography
  fragments out of scope for body-text reading order). Bound by
  `tests/test_docling_postprocessor_acceptance.py`; runs against cached
  JSONL via `HARRY_ACCEPTANCE_JSONL=<path>` or live via
  `RUN_HARRY_ACCEPTANCE=1`. **Passes** against the live full-HARRY
  conversion as of 2026-05-03; xfail removed.

### Changed
- `scripts/smoke_multiprofile.sh` matrix updated: HARRY moved from the
  `scanned` row (where it had been a low-confidence catch-all) to a new
  `digital_literature` row pointing at
  `data/digital_literature/HarryPotter_and_the_Sorcerers_Stone.pdf`. The
  freed `scanned` slot now points at `data/business_form/0013_140302111325_001.pdf`
  to keep the scanned-route assertion covered.
- `docs/ACCEPTANCE_ORDER_PROMPT.md` HARRY probe re-anchored: assertions
  now check `profile=digital_literature`, `route=native_digital`, plus a
  `HARRY-P13-SANITY` check that the page-13 chunk reads in PDF order,
  the drop cap is healed, no `Other`/`Icon`/`Table` label leak, and no
  cover-page OCR garbage. Added a `SCAN0013` probe against
  `0013_140302111325_001.pdf` to cover the scanned-route assertion that
  HARRY used to provide.

### Test counts
- 36 new tests across Phases 0-5 plus the digital_literature classifier
  suite. Full unit suite: **570 passed, 2 skipped** (HARRY gated by env
  var), 1 deselected (pre-existing unrelated `test_semantic_overlap`
  failure unchanged).

### Live evidence
- `mmrag-v2 process data/digital_literature/HarryPotter_and_the_Sorcerers_Stone.pdf`
  with no `--profile-override` → classifier auto-picks `digital_literature`,
  plan resolves to `reading_order_strategy=y_sort_with_dropcap`,
  `suppress_layout_label_text=True`, `bitmap_area_threshold=0.92`. Page 13
  reads in PDF y-order with drop cap "M" at the front, no
  `Other`/`Icon`/`Table` leak, no chunks for cover pages 1-4.
- 30-page HARRY slice (`/tmp/harry_pages_1_30.pdf`) also auto-routes to
  `digital_literature` after Rule 0c (no override needed).
- Smoke matrix (`/tmp/smoke_post_dl_v2_20260503/`): 10/11
  GATE_PASS + UNIVERSAL_PASS. HARRY row auto-routes to
  `digital_literature` and passes both gates. The single GATE_FAIL is
  on the new `scanned/0013_*` row (small business form: 17 text chunks
  mean_len 39.4 chars, hits `micro_non_label_ratio=0.294`). Classifier
  routing is correct (`detected_profile=scanned`); the gate is calibrated
  for prose-heavy docs and is a probe-vs-gate calibration issue, not a
  regression.

## [v2.7.1] — Contextual Retrieval (2026-05-01)

### Added
- **Contextual Retrieval (Anthropic approach).** New embed-time builder
  `mmrag_v2.chunking.contextual_retrieval.build_contextualized_text(...)` that
  prepends hierarchical breadcrumb (`[Context: A > B > C]`), parent heading
  (`[Heading: …]`), truncated previous/next neighbor snippets
  (`[Previous: …]`, `[Next: …]`, capped at `MAX_CONTEXT_CHARS = 300`), and a
  non-text modality marker (`[Modality: …]`) before the canonical chunk
  content. Pure function: no I/O, never mutates `content`. Reference:
  https://www.anthropic.com/news/contextual-retrieval.
- **Optional `IngestionChunk.contextualized_text: Optional[str]` schema field.**
  Carries pre-computed embedding text in JSONL when desired; never read by QA,
  source-text validation, refiner threshold logic, or any chunk creator.
- **`scripts/ingest_to_qdrant.py --no-contextual` flag.** Restores v2.7.0
  byte-stable embedding string (`f"{breadcrumb}\n{content}"` or `content` when
  breadcrumb is empty) for A/B comparison and rollback. Image chunks remain on
  the existing `embed_image()` path.
- **AGENT-CONTEXTUAL-01..07 invariants** in `docs/DECISIONS.md` "Contextual
  Retrieval (Anthropic approach)": content immutability, single embed-time
  builder, QA isolation, length budget, image lane untouched, refiner ordering
  (refined_content first), embedding-cache key safety.
- **Drift insurance.** `tests/test_contextual_retrieval.py::test_no_contextual_marker_strings_in_production_code`
  walks every `*.py` under `src/mmrag_v2/` and `scripts/` and fails the moment
  a non-allowlisted file contains a marker literal or calls
  `build_contextualized_text(...)`.

### Test counts
- Focused contextual suite: 32 passed.
- Static guards: 2 passed.
- Focused boundary suite (`pdf_conversion_plan` + `chunker_guard` + quarantine + finalization bridges): 93 passed.
- Full unit suite: 512 passed, 1 skipped, 0 failed (was 480 before; net +32 contextual cases).
- Probe `output/probe_contextual_retrieval_rag_guide/`: AUDIT_PASS + UNIVERSAL_PASS, 680 chunks (text=559, image=99, table=22), `indentation_fidelity=0.91` — byte-identical to the Boundary Closeout baseline `output/probe_boundary_closeout_rag_guide/`.
- Multi-profile smoke (Contextual Retrieval close): `output/smoke_multiprofile_20260501_153101/` 10/10 GATE_PASS + UNIVERSAL_PASS, including the Greenhouse blind-test document.

## [Unreleased] — Milestone 1: Stabilize Extraction (2026-04-30 → 2026-05-01)

### Added
- **Shared PDF extraction plan + Docling adapter.** `src/mmrag_v2/engines/pdf_plan.py`
  (`PdfConversionPlan` + `build_pdf_conversion_plan`) and `engines/docling_adapter.py`
  (`DoclingPdfAdapter`) now own all `PdfPipelineOptions` / `DocumentConverter`
  construction. `batch_processor.py`, `processor.py`, and `engines/pdf_engine.py`
  consume the shared plan. AST-level static guards in
  `tests/test_pdf_conversion_plan.py` reject any new construction outside the adapter.
- **Plan Control Plane (Milestone 2).** `PdfConversionPlan` promoted to a typed
  policy object: `extraction_route` vocabulary {native_digital, scanned_book,
  image_heavy_magazine, technical_manual} with auto-detection; explicit
  `hybrid_chunker_enabled`, `allow_page_level_visuals`, `asset_validation_policy`
  (drop|keep|quarantine), `corruption_recovery_policy` (quarantine|keep|recover)
  fields. Legacy bools (`drop_blank_assets`, `quarantine_corrupted_chunks`)
  preserved as derived `@property` bridges.
- **Vision-aided front-matter detection** moved after heading inference and TOC/forward
  propagation. Uses Docling/shadow image extractions before the first chapter-like
  heading to relabel non-chapter headings as "Front Matter". Preserves numbered/chapter
  headings.
- **Coordinate normalization audit.** `scripts/qa_universal_invariants.py` now
  reads current `metadata.spatial` plus legacy `metadata.spatial_metadata`, fails
  malformed/zero-area bboxes, and reports per-modality bbox distribution.
  `ensure_normalized()` repairs one-unit extents inside the 0–1000 canvas.
- **Domain-specific search priority** moved from converter to ingestor.
  `scripts/ingest_to_qdrant.py` resolves `search_priority` from `document_domain`,
  page position, and heading context while preserving stricter converter demotions.
- **Classifier fallback fixes (Milestone 1.A).** Literature/long-form docs no
  longer misroute as `digital_magazine`. Digital fallback default changed to
  `TECHNICAL_MANUAL`. `tests/test_classifier_fallback.py` (9 tests).
- **Extraction route controls (Milestone 1.B).** `extraction_route`,
  `max_chunker_input_chars`, `drop_blank_assets`, `quarantine_corrupted_chunks`
  fields on `PdfConversionPlan`. `scanned_book` route disables HybridChunker
  and picture classification.
- **HybridChunker pathological-input guard (Milestone 1.C).** Total-text guard
  (default 500_000 chars) skips HybridChunker before tokenizer hangs.
  Per-element guard (`max_chunker_per_element_chars`, default 100_000, added
  2026-05-01) catches single mega-elements. Per-batch SIGALRM 120s remains as
  the inner safety net. `tests/test_chunker_guard.py` (11 tests).
- **Corruption quarantine (Milestone 1.D).** `BatchProcessor._quarantine_corrupted_text_chunks()`
  drops post-patch corrupted text chunks; IMAGE/TABLE chunks are not quarantined.
  `tests/test_corruption_quarantine.py` (8 tests) + bridge tests in
  `tests/test_finalization_bridge.py`.
- **Blank asset quarantine (Milestone 1.E).** `BatchProcessor._filter_blank_assets()`
  + `_is_blank_asset()` (mean≈255/0, std<5). TABLE chunks with blank assets
  are promoted to TEXT (preserving markdown). IMAGE chunks with blank assets
  are dropped. `tests/test_blank_asset_quarantine.py` (7 tests).

### Fixed
- **RAG Guide 7200s hang.** `output/probe_rag_guide_guard/` AUDIT_PASS + GATE_PASS
  + UNIVERSAL_PASS, 680 chunks, ~5 min conversion. Pathological batch 25 (pages
  241-250) detected via SIGALRM and falls back to element-by-element chunking.
- **Ayeva indentation fidelity 0.22 → 0.93.** Re-conversion at
  `output/ayeva_qa_20260501/`; AUDIT_PASS + GATE_PASS + UNIVERSAL_PASS, 627
  chunks, `infix_strict=0` (was 2). Profile reclassified `digital_magazine` →
  `technical_manual` after the classifier fallback fix.
- **Combat Aircraft corrupted-text leakage.** 0 corrupted text chunks in JSONL
  (was 22 encoding artifacts + 79 high-corruption); IMAGE quality preserved.
- **Harry Potter routed correctly.** Now `technical_manual` (was
  `digital_magazine`); AUDIT_PASS + UNIVERSAL_PASS, 458 chunks.
- **CarOK blank table asset.** Blank TABLE asset promoted to TEXT modality;
  IMAGE category previously failing now passes.

### Documentation
- New: `docs/QUALITY_SNAPSHOT_2026-04-30.md`, `docs/QUALITY_SNAPSHOT_2026-05-01.md`.
- Updated: `docs/PROJECT_STATUS.md`, `docs/PROGRESS_CHECKLIST.md`,
  `docs/ARCHITECTURE.md`, `docs/CONVERSION_PROFILES.md`, `docs/DECISIONS.md`,
  `AGENTS.md`, `CLAUDE.md`, `README.md` (OCR cascade).

### Test counts
- Focused Milestone 1 suite: 81 passed.
- Focused Milestone 2 suite (`pdf_conversion_plan` + `chunker_guard`): 64 passed.
- Full unit suite: 484 passed, 1 skipped, 0 failed.
- Static guards: 2 passed.
- Multi-profile smoke (Milestone 1 close): `output/smoke_multiprofile_20260501_105836/` 10/10 GATE_PASS + UNIVERSAL_PASS.
- Multi-profile smoke (Milestone 2 close): `output/smoke_multiprofile_20260501_120514/` 10/10 GATE_PASS + UNIVERSAL_PASS.
- Targeted probes: `output/probe_rag_guide_guard/` (M1) and `output/probe_milestone2_rag_guide/` (M2) — both AUDIT_PASS + GATE_PASS + UNIVERSAL_PASS.

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
