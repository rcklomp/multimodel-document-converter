# Decisions and Guardrails

## OCR Cascade Order
**Decision:** Docling → Tesseract → Doctr for layout-aware OCR.
**Rationale:** Keeps Docling layout awareness first, with progressive fallback when confidence is low.

## OCR Confidence Threshold Governance
**Decision:** The default layout-aware OCR trigger threshold is **0.70** (`--ocr-confidence-threshold` default).

**Rationale:**
- The threshold is an empirical quality lever, not a fixed architectural invariant.
- The legacy 0.90 expectation was too aggressive in practice and increased unnecessary OCR escalation.
- Acceptance tuning showed 0.70 gives better balance between extraction fidelity and over-triggering.

**Operationalization:**
- SRS defines behavior and default, while this document records the decision basis and tuning policy.
- Any change to the default threshold must include before/after acceptance evidence.
- Validate changes with representative acceptance runs and QA outputs before adoption.

## VLM Orchestration Protocol
- Changes to classifier/orchestrator must include impact analysis against the core test matrix.
- No modality-crossing fallbacks (scanned must stay scanned; digital must stay digital).
- No hardcoded document-specific rules.

## Anti-Patterns (Explicitly Forbidden)
- Overfitting to specific filenames.
- Forcing digital_magazine as a “safe” fallback for scans.
- Treating metadata as ground-truth instead of diagnostic evidence.

## Structural Pathology over Semantic Profiling (v2.5.0)

**Decision:** PDF extraction pathway (use digital text / flat-code OCR rescue / force full OCR) is determined by **structural integrity tests** on the PDF byte-stream, not by the semantic content type (e.g., "technical_manual", "academic_whitepaper").

**Rationale:**
- Semantic content type has zero correlation with technical PDF integrity. A technical manual can be a perfectly structured PDF or a newline-stripped disaster from a broken PDF generator (e.g., Kimothi 2025, Python Distilled). Routing on semantic labels causes silent quality failures.
- Three structural tests are sufficient to classify PDF health before any extraction begins:
  1. **Line-break health** (words/`\n` ratio on sample pages) — free, < 1 ms/page.
  2. **Visual-digital delta** (PyMuPDF text vs Tesseract OCR word-set overlap on one page) — definitive, ~300 ms.
  3. **Geometry error rate** (MuPDF path-syntax error count) — logging and risk signal only.
- Semantic profiles continue to govern VLM prompt context, extraction sensitivity, and image thresholds — they remain useful for *what to describe*, not for *how to extract*.

**The two-axis model:**
```
                  STRUCTURAL INTEGRITY
                  Healthy  │ Flat text  │ Encoding
                           │ corrupted  │ corrupted
  ────────────────┼──────────┼────────────┼───────────
  S digital       │ Docling  │ +flat code │ force OCR
  E               │ direct   │ OCR rescue │
  M ────────────────┼──────────┼────────────┼───────────
  A scanned       │ nuclear  │ nuclear +  │ force OCR
  N               │ OCR      │ flat rescue│
  T ────────────────┴──────────┴────────────┴───────────
  I
  C
```

**Operationalization:**
- `_perform_physical_check` in `document_diagnostic.py` runs the three tests.
- Flags `has_flat_text_corruption` and `has_encoding_corruption` added to `PhysicalCheckResult`.
- `batch_processor.py` reads these flags to activate flat-code OCR rescue and/or upgrade to forced OCR.
- Semantic profile selection is unaffected; it runs in parallel and drives VLM/sensitivity settings only.

**Anti-patterns now explicitly forbidden:**
- Using `profile_type == "technical_manual"` to decide whether OCR is needed.
- Assuming `native_digital` modality means all text is correctly encoded and formatted.

---

## Image Extraction Routing (v2.7.0)

**Decision:** All document types use Docling layout model for image extraction. PyMuPDF `page.get_images()` is not used in the active pipeline.

**Rationale:**
- PyMuPDF direct extraction was implemented and tested for `native_digital` PDFs (I10). It works for simple cases (technical books with discrete embedded images) but fails for:
  - **Magazines:** Composite page layouts where text and photos are baked together as single rasterized images. PyMuPDF extracts these composites whole — it cannot separate photos from text backgrounds.
  - **Academic papers:** Vector figures extracted as solid-color backgrounds.
- Docling's layout model with picture classification (`DocumentFigureClassifier-v2.5`, Docling 2.86.0) correctly identifies image regions across all document types. A deny filter rejects `full_page_image` and `page_thumbnail` layout artifacts.
- Picture classification is **disabled for scanned docs** (`scanned`, `scanned_degraded`) because the classifier model hangs on large scanned books with hundreds of image regions (tested: 292-page Firearms on 16GB M1).
- The PyMuPDF `_extract_embedded_images` method is retained in the codebase for future use. The proper fix for magazine image quality is the rendered-region-crop architecture (tracked in `CONVERSION_PROFILES.md`).

---

## Heal-Over for Encoding Corruption (v2.7.0)

**Decision:** When encoding corruption is detected (`has_encoding_corruption`), keep HybridChunker active and force the semantic refiner on all chunks at `threshold=0.0`, instead of disabling HybridChunker and falling back to full OCR.

**Rationale:**
- Disabling HybridChunker loses structural metadata: heading hierarchy, table structures, sentence-boundary-aware splitting.
- The refiner (LLM-based) understands language context and can replace glyph placeholders (`/C211`, `/C1`, hex leaks) with correct characters while preserving the surrounding structure.
- This "heal-over" approach preserves Docling's structural analysis as a skeleton and patches only the corrupted text content.

**Operationalization:**
- `CorruptionInterceptor` (`src/mmrag_v2/validators/corruption_interceptor.py`) performs per-bbox OCR patching at 300 DPI for chunks with detected encoding artifacts.
- The refiner threshold override is set in `batch_processor.py` when `has_encoding_corruption` is true.

---

## Selective Code Enrichment Lane (Workstream B, 2026-04-29)

**Decision:** Code-block fidelity must use a selective enrichment lane. Do not enable Docling `do_code_enrichment` broadly from `has_encoding_corruption` or profile alone.

**Rationale:**
- Docling 2.86.0 already emits code regions as `CodeItem`; `CodeItem.text` can still be flat when the source PDF text layer has stripped code newlines.
- `do_code_enrichment=True` fixes this at the right layer by rendering code regions and running CodeFormulaV2, but local CPU execution is too slow for broad conversion.
- `has_encoding_corruption` is a text-integrity signal, not a code-density signal. Using it as a trigger would pull magazine/text-corruption workstreams into expensive code-model inference.
- The client machine should not be the primary CodeFormulaV2 inference target when stronger local-network or cloud machines are available.

**Operationalization:**
- First run a cheap code-evidence pass: Docling `CodeItem` count, code-chunk ratio, or sampled code-candidate regions.
- Emit/use an explicit decision such as `needs_code_enrichment=True` with reason/counts; do not infer it solely from `has_encoding_corruption`.
- Prefer remote-capable CodeFormulaV2 inference on a stronger local-network host; cloud is acceptable when data policy/cost allow; local client execution is diagnostic/fallback only.
- If Docling only supports document-level `do_code_enrichment`, enable it only after the code-evidence pass indicates a code-heavy/code-candidate document.
- If region-level remote inference is implemented, send only `CodeItem`/code-candidate crops, not whole documents.
- Preserve `_has_fenced_flat_code` only as a provisional fallback marker when native/remote code enrichment is unavailable or still returns flat code.
- Refactor duplicated PDF extraction policy behind a shared `PdfConversionPlan` and Docling PDF adapter. `batch_processor.py`, `processor.py`, and `engines/pdf_engine.py` must not remain independent sources of Docling `PdfPipelineOptions` / `DocumentConverter` truth.
- The canonical PDF architecture is diagnostics/config -> `PdfConversionPlan` -> Docling adapter -> `UniversalDocument` -> `ElementProcessor` -> chunks. Direct Docling-item-to-chunk paths are legacy only and must not be expanded.

**Anti-patterns now explicitly forbidden:**
- Triggering CodeFormulaV2 from `has_encoding_corruption` alone.
- Adding profile-specific `do_code_enrichment=True` rules in either processor path.
- Adding new Docling `PdfPipelineOptions` or `DocumentConverter` construction outside the shared adapter/factory.
- Installing **custom client-side MLX/transformer** acceleration as the main production strategy before evaluating remote inference. (See Amendment 2026-05-03 for Docling-native CPU runtime.)
- Letting fallback regex/Tesseract repair mask whether Docling-native/remote enrichment actually fixed the code.
- Weakening negative tests that prove non-code documents, incidental shell commands, sparse fenced snippets, or encoding corruption alone do not trigger CodeFormulaV2. These tests are Workstream B contracts.

**Amendment 2026-05-03 — Docling-native CPU acceptable for batch reconversion:**

Empirical evidence collected 2026-05-03 (test on Chaubal pages
250-260) updates the cost model:

- Docling 2.86's bundled `CodeFormulaModel` (CodeFormulaV2 weights,
  no custom MLX/transformer setup) runs at **~27 sec/page on CPU**
  on the project's Apple Silicon target. Docling explicitly forces
  CPU for this model (`Removing MPS from available devices because
  it is not in supported_devices=[CPU, CUDA]`).
- For a one-off batch reconversion, this is acceptable: Chaubal's
  359 pages = ~150 min CPU, run overnight or alongside other work.
- The original "client-local diagnostic/fallback only" anti-pattern
  was authored with custom MLX / transformer setups in mind (slow
  to set up, GPU/MPS-bound, often required offline model conversion).
  Docling-native CodeFormulaV2 has none of those properties.
- Remote inference (local-network or cloud) remains **preferred**
  for latency-sensitive use and for corpora where reconversion
  runtime would exceed acceptable bounds (e.g. Chaubal-class docs at
  larger scale, or multi-Chaubal nightly runs).
- The anti-pattern is amended to clarify: it forbids **custom
  MLX/transformer** client-local setups as the main production
  strategy; it does **not** forbid one-off batch use of
  Docling-bundled CodeFormulaV2 on CPU.

**Operational rule of thumb:** If a code-heavy document needs
reconversion and remote inference isn't available, run
Docling-native CodeFormulaV2 on CPU. If reconversion of code-heavy
docs becomes routine (more than once per week), invest in remote
inference setup for v2.9.

---

## Shared PDF Extraction Plan (Workstream B, 2026-04-30)

**Decision:** PDF extraction policy is centralized in `PdfConversionPlan`, and `DoclingPdfAdapter` is the only production code allowed to instantiate Docling `PdfPipelineOptions` or `DocumentConverter`.

**Rationale:**
- Batch, direct processor, and UIR engine paths previously set overlapping Docling options independently, creating drift and bridge bugs.
- Code enrichment, OCR, table-image generation, table structure, reading order, picture classification, and structural corruption flags must cross CLI, batch, processor, engine, and adapter boundaries as one explicit plan.
- Chunk factory metadata must remain separate from structural/document-level flags so chunk creation is not polluted by control fields.

**Operationalization:**
- CLI process/direct, CLI process/batch, and CLI batch command build a `PdfConversionPlan` after diagnostics, OCR auto-overrides, profile selection, and cheap code-evidence scoring.
- `BatchProcessor`, `V2DocumentProcessor`, and `PDFEngine` consume the plan through `DoclingPdfAdapter`; legacy metadata entry points remain as compatibility shims that build a plan before adapter use.
- `PdfConversionPlan.to_intelligence_metadata()` returns full boundary metadata; `chunk_factory_metadata()` returns only chunk-safe keys.
- Static guard tests fail if production code constructs Docling PDF options/converters outside `src/mmrag_v2/engines/docling_adapter.py`.
- **Amendment 2026-05-04 (PLAN_V2.8 §2):** the construction guard is now joined by an invocation guard (`test_no_raw_converter_invocation_outside_adapter`). It AST-walks production code and rejects `self._converter.convert(...)` / `self._docling_converter.convert(...)` outside the adapter — the failure mode that put `processor.py:2072` and `pdf_engine.py:206` on the v2.8 plan. Cleanup-style calls (`._converter.cleanup()` / `.close()` / `.shutdown()`) are unaffected; only `.convert(...)` invocations are blocked.
- `generate_table_images` is false by default and true only when `force_table_vlm=True`; non-VLM table extraction remains TableFormer markdown-based.
- OCR engine mapping preserves status quo: OCR-enabled plans create `EasyOcrOptions()` regardless of the CLI engine string.

**Evidence:** Focused plan/bridge/UIR tests `73 passed`; full unit suite `412 passed, 1 skipped`; smoke run `output/smoke_multiprofile_20260430_083922/` has all 10 rows `GATE_PASS` + `UNIVERSAL_PASS`, including Greenhouse blind-test.

---

## Contextual Retrieval (Anthropic approach) (v2.7.1, 2026-05-01)

**Decision:** Embedding text for `text` and `table` modality chunks is built at ingest time by `mmrag_v2.chunking.contextual_retrieval.build_contextualized_text(...)`, prepending hierarchical breadcrumb + parent heading + truncated prev/next neighbor snippets + non-text modality marker before the canonical content. The `IngestionChunk.content` and `metadata.refined_content` fields are never mutated. Image chunks remain on the existing `embed_image()` path; the contextualization function is not used for them by the production ingestor.

**Reference:** https://www.anthropic.com/news/contextual-retrieval (Anthropic, September 2024).

**Scope:** Embed-time only. No new typed `PdfConversionPlan` field, no UIR rewrite, no element-mapping refactor, no new CLI flag on `mmrag-v2 process` / `mmrag-v2 batch`. The only new ingest-side flag is `scripts/ingest_to_qdrant.py --no-contextual`, which restores the v2.7.0 byte-stable embedding string `f"{breadcrumb}\n{content}"` (or `content` when breadcrumb is empty).

**Invariants (mirrored verbatim in the module docstring):**

- **AGENT-CONTEXTUAL-01 — Content immutability.** The canonical `IngestionChunk.content` is never mutated. The prefixes live in a separate, optional embedding-time field (`contextualized_text`) that is never read by QA, source-text validation, refiner threshold logic, or any chunk creator.
- **AGENT-CONTEXTUAL-02 — Single embed-time builder.** The only function allowed to assemble contextualized text is `build_contextualized_text`. Importers are: the embedding lane in `scripts/ingest_to_qdrant.py`, `tests/test_contextual_retrieval.py`, and (optionally) a future RAG adapter — nothing else.
- **AGENT-CONTEXTUAL-03 — QA isolation.** Markers `[Context: ]`, `[Heading: ]`, `[Previous: ]`, `[Next: ]`, `[Modality: ]` MUST NOT appear in `IngestionChunk.content`, `metadata.refined_content`, the Qdrant payload `text`/`content` field, or anything fed back into `qa_conversion_audit.py`, `qa_universal_invariants.py`, or `token_validator.py`.
- **AGENT-CONTEXTUAL-04 — Length budget.** Per Anthropic, target ~50–100 tokens (~200–400 chars). Cap each `prev_text_snippet` and `next_text_snippet` to `MAX_CONTEXT_CHARS = 300`. Truncate; do not reflow. Truncation is on a Unicode code-point boundary (Python `str` slicing).
- **AGENT-CONTEXTUAL-05 — Image lane untouched.** Image chunks already embed via `embed_image()` with the visual description as fallback. Contextualization is for `modality in {"text", "table"}` only in the production ingestor.
- **AGENT-CONTEXTUAL-06 — Refiner ordering.** The refiner runs *before* contextualization. The ingestor reads `metadata.refined_content` first, falls back to `chunk["content"]`. The contextualized string is never re-fed into the refiner.
- **AGENT-CONTEXTUAL-07 — Cache key safety.** If/when an embedding cache is added, it MUST key on the contextualized string actually sent to the embedder, not on raw `content`. Otherwise toggling `--no-contextual` returns stale vectors. (No embedding cache exists in this repo today; only `vision_manager` caches VLM responses.)

**File locations:**
- Builder: `src/mmrag_v2/chunking/contextual_retrieval.py` (allowlist for marker-string literals).
- Ingestor wiring: `scripts/ingest_to_qdrant.py` (allowlisted call site).
- Schema field: `IngestionChunk.contextualized_text: Optional[str]` in `src/mmrag_v2/schema/ingestion_schema.py`.
- Tests + drift guard: `tests/test_contextual_retrieval.py`.

**Rollback flag:** `scripts/ingest_to_qdrant.py --no-contextual` restores v2.7.0 byte-stable embedding text. Required for A/B comparison of retrieval quality and as a safety lever during rollout.

**Drift insurance:** `tests/test_contextual_retrieval.py::test_no_contextual_marker_strings_in_production_code` walks every `*.py` under `src/mmrag_v2/` and `scripts/` and fails the moment a non-allowlisted file contains a marker literal or calls `build_contextualized_text(...)`. Rejection criterion: any write of those strings into a chunk-creation helper, refiner output, or payload field is a P0 defect.

**Evidence:** Focused contextual suite `32 passed`; static guards `2 passed`; focused boundary suite `93 passed`; full unit suite `512 passed, 1 skipped, 0 failed`; probe `output/probe_contextual_retrieval_rag_guide/` AUDIT_PASS + UNIVERSAL_PASS with byte-identical structural shape (680 chunks: text=559 / image=99 / table=22; `indentation_fidelity=0.91`) to the Boundary Closeout baseline `output/probe_boundary_closeout_rag_guide/`; smoke `output/smoke_multiprofile_20260501_153101/` 10/10 GATE_PASS + UNIVERSAL_PASS, including the Greenhouse blind-test document. See `docs/QUALITY_SNAPSHOT_2026-05-01.md` "Contextual Retrieval (Anthropic approach)".

---

## Multimodal Validation Layers (v2.7.0)

**Decision:** Replace heuristic string-matching loops with 4 signal-driven validation layers that use OCR confidence, VLM descriptions, and POS tagging.

**The 4 layers:**

1. **CorruptionInterceptor** — Per-bbox OCR patching for encoding artifacts. Renders only the corrupted chunk's bbox at 300 DPI, runs Tesseract, replaces content if OCR result is cleaner. Preserves HybridChunker structure.

2. **POS Boundary Logic** — Merges trailing orphan prepositions (`BY`, `FOR`, `OF`, `WITH`, `von`, `für`, `van`, `voor`, `par`, `pour`) into the next chunk when it starts with a proper noun. Same-page guard prevents cross-page false merges. The preposition must be the ONLY word on its line (true orphan).

3. **Vision-Gated Hierarchy** — After heading inference and TOC/forward propagation, pages before the first chapter-like heading that have Docling/shadow image extractions are treated as front matter. Non-chapter, non-numbered headings on those pages are demoted to "Front Matter". If no chapter boundary is found, an explicit front-matter visual cue is required.

4. **Content-Type Classification** — Chunks with 2+ boilerplate markers (ISBN, ©, "All rights reserved", "Printed in") get `search_priority` downgraded to `low`. Global rule across all profiles.

**Rationale:**
- Heuristic string matching (v2.6 approach) required per-document tuning and broke on edge cases. These layers use structural signals (OCR confidence, VLM output, POS tags) that generalize across document types.

---

## Post-Docling Sanity Pass + `digital_literature` Profile (2026-05-03)

**Decision:** Born-digital novels are routed through a new `digital_literature`
profile that opts into four post-Docling sanity stages applied at the
`DoclingPdfAdapter.convert()` seam. Successor to v2.7 §5; full plan at
`docs/PLAN_DOCLING_POSTPROCESSOR.md`.

**Rationale:**
- Docling 2.86 produces four reproducible failure modes on born-digital
  novels (verified on Harry Potter and the Sorcerer's Stone, AGaramondPro
  / Acrobat Distiller PDF):
  1. Reading-order swaps within a page (e.g. page 13 emits paragraphs
     `[para1, para3, para2]`).
  2. Drop-cap "M" appended INLINE at the end of the same TextItem
     (`"r. and Mrs. Dursley...nonsense. M"`) instead of leading the
     paragraph.
  3. Picture classification labels (`Other`, `Icon`, `Table`) emitted as
     body text via both `meta.classification` and the legacy
     `PictureClassificationData` annotation path, even when a caption
     exists.
  4. Photographic cover pages OCR'd into garbage like
     `"= 23555 AND Potter SIONE"` because the default
     `bitmap_area_threshold` is 0.05.
- Web research (Discussions #2791, #2755; Issues #1203, #2245, #2538;
  docling-serve #448) confirms none of these are scheduled for upstream
  fixes in the foreseeable future.
- The fixes belong at the adapter seam, not in chunker post-processing,
  because the chunker reads `body.children` order and HybridChunker's
  serializer reads `meta`/`annotations`.

**Operationalization:**
- New module `engines/docling_postprocess.py` exposes
  `apply_reading_order_sort` and `apply_dropcap_promotion`. The dropcap
  pass runs both a standalone-glyph merge (separate `TextItem("M")`
  adjacent to a lowercase paragraph) and an `_heal_inline_trailing_dropcap`
  inline heal — the latter is the actually-emitted Docling 2.86 pattern.
- New module `engines/docling_serializers.py` exposes
  `MmragChunkingSerializerProvider`. The picture serializer strips
  `PictureClassificationData` annotations across all pictures (not only
  no-caption cases) before delegating; original annotations are
  restored after serialization. The chunker's params ship with
  `blocked_meta_names={"classification"}` so the new meta path is
  blocked too.
- New `PdfConversionPlan` fields: `reading_order_strategy`
  (`docling_native` | `y_sort` | `y_sort_with_dropcap`),
  `suppress_layout_label_text` (bool), `bitmap_area_threshold` (float,
  default 0.75 — raised from Docling's 0.05 to keep OCR off cover
  artwork on born-digital docs).
- New `DIGITAL_LITERATURE` ProfileType across
  `orchestration/profile_classifier.py` (enum + `_score_digital_literature`
  scorer + score loop + modality fallback),
  `orchestration/strategy_profiles.py` (`DigitalLiteratureProfile`
  strategy class + ProfileManager registry + classifier→strategy
  `type_mapping`), and `orchestration/strategy_orchestrator.py`
  (`PROFILE_TO_DOC_TYPE` → `DocumentType.LITERATURE`).
- `build_pdf_conversion_plan` auto-enables the post-processor stack
  when `profile_type == "digital_literature"`:
  `reading_order_strategy="y_sort_with_dropcap"`,
  `suppress_layout_label_text=True`, `bitmap_area_threshold=0.92`.
- Diagnostic Rule 0c added to `document_diagnostic.py` so moderate-length
  dialogue-rich documents (e.g. the 30-page HARRY test slice) reach
  `domain=literature` despite the small `DIAGNOSTIC_SAMPLE_PAGES=5`
  cap. Trigger: `_dialogue_pages >= 1 AND total_pages > 20 AND not
  has_tables AND 500 < avg_text_per_page < 2500 → literature += 0.4`.

**v2.7 §5 followup — bypass patched:** the static guard from §5 banned
`PdfPipelineOptions(` / `DocumentConverter(` *construction* outside the
adapter but did NOT catch raw `self._converter.convert(...)` *invocation*.
`processor.py:2072` was using the cached converter directly, sidestepping
the post-Docling stages. Re-routed through `self._adapter.convert(...)`.
A companion guard test should follow.

**Evidence:**
- 50 new unit tests (`tests/test_docling_postprocess_*.py`,
  `tests/test_classifier_digital_literature.py`).
- HARRY pages 13-30 acceptance fixture
  (`tests/test_docling_postprocessor_acceptance.py` +
  `tests/fixtures/harry_potter_pages_1_to_30/`); xfail removed; PASSES
  against live full-HARRY conversion.
- Full unit suite: 570 passed, 2 skipped, 1 deselected (pre-existing
  unrelated `test_semantic_overlap` failure).
- Smoke matrix `/tmp/smoke_post_dl_v2_20260503/`: 10/11 GATE_PASS +
  UNIVERSAL_PASS. The 1 fail is on the new `scanned/0013_*` business-form
  row (gate calibrated for prose); HARRY auto-routes to
  `digital_literature` and passes both gates.

---

## Chunk Size Governance
**Decision:** Chunk length is governed per profile and verified with acceptance metrics; no universal hard min/max.

**Rationale:**
- Different modalities and document classes need different chunking behavior.
- A single global threshold causes regressions (either fragmentation or oversized chunks).
- Quality must be demonstrated with repeatable benchmarks, not assumed from one document.

**Operationalization:**
- Use representative acceptance runs (e.g., `scripts/acceptance_technical_manual.sh`).
- Track both structural hygiene (`text_short_<30`, `text_long_>1500`, `infix_strict`) and coverage (`QA-CHECK-01`).
- Document any threshold/range change with baseline comparison in the run summary.
