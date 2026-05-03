# Plan: v2.7 — Document Understanding Layer

**Status (May 2026):** Features 1–6 are all **shipped**. The shared `PdfConversionPlan` / `DoclingPdfAdapter` foundation from §5 is the seam that the post-Docling sanity pass (`docs/PLAN_DOCLING_POSTPROCESSOR.md`, shipped 2026-05-03) builds on. The 4 multimodal validation layers (bottom of doc) are realised through Features 1–4 plus `validators/corruption_interceptor.py`. This plan is retained as architectural rationale — see `CHANGELOG.md`, `docs/DECISIONS.md`, and `docs/PROGRESS_CHECKLIST.md` for current state.

## Context

v2.6 delivered HybridChunker integration, VLM transcription for scanned docs,
multi-format support, and 10/10 smoke test pass. The remaining issues are
Docling extraction limitations that can't be fixed in post-processing without
overfitting. v2.7 moves from text-based heuristics to layout-aware structural
mapping.

Source: Gemini Pro audit recommendations, April 2026.

## Features

### 1. Cross-Chunk Semantic Stitching

**Status:** Pre-existing implementation hardened and unit-validated. Batch
finalization already called `BatchProcessor._merge_hungry_operators(...)`;
2026-04-30 work fixed punctuation and empty-chunk edge cases, added the
`_apply_final_boundary_repairs(...)` bridge, and added regression coverage in
`tests/test_cross_chunk_semantic_stitching.py`.

**Problem:** Orphan prepositions at chunk boundaries ("BY" at end of credits
chunk, separated from "Mary GrandPré" in the next chunk).

**Approach:** When a chunk ends with a semantically "hungry" word (preposition:
BY, FOR, OF, WITH, FROM) and the next chunk starts with a proper noun or title,
pull the preposition into the next chunk.

**Why it's not overfitting:** Prepositions are structurally incomplete without
their object. "VOID" and "END" are self-contained. This is a syntactic rule,
not a character-count heuristic.

**Implementation:** Post-processing pass after mid-sentence merger. Check
last word of each chunk against a preposition list. Check first word of next
chunk for capitalization (proper noun signal). Language-agnostic — prepositions
exist in all languages.

### 2. Vision-Aided Front Matter Detection

**Status:** `complete`.
2026-04-30 work moved the detector after all heading assignment paths, broadened
the visual signal to Docling/shadow image extractions before the first
chapter-like heading, preserved numbered/chapter headings, and added focused
negative + bridge coverage in `tests/test_vision_aided_front_matter.py`.
Completion evidence: focused tests `41 passed`, full unit suite
`356 passed, 1 skipped`, and fresh smoke run
`output/smoke_multiprofile_20260430_frontmatter_complete/` with all 10 rows
non-empty (`min_chunks=8`), all rows `GATE_PASS` + `UNIVERSAL_PASS`, no
conversion-error log matches, and the Greenhouse blind-test document included.

**Problem:** Docling misidentifies author names and publisher names as headings
on front matter pages ("J. K. Rowling" as parent_heading).

**Approach:** When a potential heading is on a page that:
- Has `shadow` or `docling` figure extractions (cover art, logos)
- Is before the first "Chapter" heading
- Has no numbered section pattern

Default the parent heading to "Front Matter" instead of using the detected
text as a heading.

**Why it's not overfitting:** Every book has front matter (title, copyright,
dedication, TOC) before content starts. Detecting the boundary between front
matter and content is a structural pattern.

**Implementation:** After heading inference, scan for the first chapter-like
heading. Everything before it with non-chapter headings gets re-labeled as
"Front Matter".

### 3. Domain-Specific Search Priority

**Problem:** `search_priority: "high"` on all text chunks including copyright
notices, printer information, ISBN numbers.

**Approach:** Move priority logic from the converter to the ingestor. Use
`document_domain` to apply domain-specific rules:
- Literature: pages before TOC → "low"
- Academic: references section → "medium"
- Technical: index/appendix → "medium"

**Why it's not overfitting:** Priority is a RAG retrieval concern, not a
conversion concern. The converter shouldn't decide what's important — the
ingestor/RAG system should, based on document structure metadata.

**Implementation:** In `ingest_to_qdrant.py`, add a priority field to the
Qdrant payload based on page position, heading context, and domain.

### 4. Coordinate Normalization Audit

**Status:** `complete`.
2026-04-30 work hardened `qa_universal_invariants.py` into a final bbox audit:
it now reads current `metadata.spatial` and legacy `metadata.spatial_metadata`,
rejects malformed/zero-area bboxes, and reports bbox distribution statistics
per modality. The audit exposed zero-extent producer output in smoke; the
shared coordinate normalizer now repairs one-unit extents inside the 0-1000
canvas before export.

Completion evidence: `tests/test_coordinate_normalization_audit.py` covers
negative malformed bbox cases, current-schema JSONL bridge flow, legacy key
compatibility, CLI output, and zero-extent normalizer regressions. Full unit
suite: `381 passed, 1 skipped`. Smoke run:
`output/smoke_multiprofile_20260430_075420/` with all 10 rows
`GATE_PASS` + `UNIVERSAL_PASS`, including the Greenhouse blind-test document.

**Problem:** Potential inconsistency between text and image bounding boxes
reported by Gemini (later confirmed to both use [0,1000] — may be a
non-issue, but worth auditing).

**Approach:** Add a final validation pass that checks all bbox values are
in [0,1000] and flag any that aren't. Already partially done by
QA-CHECK-04 / REQ-COORD-01.

**Implementation:** Enhance `qa_universal_invariants.py` to report bbox
distribution statistics per modality.

### 5. Shared PDF Extraction Plan and Adapter Refactor

**Status:** `complete` (2026-04-30 foundation; 2026-05-03 bypass patch).
Shared `PdfConversionPlan` and `DoclingPdfAdapter` are implemented;
bridge/static guard tests and multi-profile smoke evidence exist. See
`docs/DECISIONS.md` and `docs/QUALITY_SNAPSHOT_2026-04-30.md`.

**2026-05-03 followup — latent bypass patched.** While shipping the
Post-Docling Sanity Pass (see `docs/PLAN_DOCLING_POSTPROCESSOR.md`), a
gap was found in the success criterion *"Single Docling options
authority"*: the static guard `test_no_pipeline_options_construction_outside_adapter`
banned `PdfPipelineOptions(` and `DocumentConverter(` *construction*
outside the adapter, but did not catch raw `self._converter.convert(...)`
*invocation* on the cached converter. `processor.py:2072` was using the
cached object directly, silently bypassing any post-Docling stage gated
on the plan. Re-routed through `self._adapter.convert(...)` which
delegates to the cached converter and runs `apply_postprocessors`. A
companion guard test should follow.

**Problem:** `batch_processor.py`, `processor.py`, and
`engines/pdf_engine.py` all construct Docling PDF converters /
`PdfPipelineOptions`. This duplicates extraction policy, causes drift between
batch, direct, and UIR paths, and already produced Workstream B bugs where the
producer and consumer were correct independently but the call-site bridge
dropped the decision.

**Researched design pattern:** Use a small Ports-and-Adapters boundary around
PDF extraction, with a Pipes-and-Filters pipeline inside it:
- Application pipeline stages: diagnose -> plan -> extract -> map -> enrich ->
  validate -> export.
- Primary adapters: CLI, batch command, tests.
- Secondary adapters: Docling PDF adapter, OCR adapter, VLM/refiner adapter,
  optional code-recovery adapter.
- Canonical PDF flow: diagnostics/config -> `PdfConversionPlan` ->
  `DoclingPdfAdapter` -> `UniversalDocument` -> `ElementProcessor` -> chunks.
- `BatchProcessor` owns batching/orchestration/export only.
- `V2DocumentProcessor` is transitional legacy glue only. New PDF work must
  not add Docling-item-to-chunk behavior there; it should move mapping behind
  UIR/ElementProcessor or delete the legacy path when parity is proven.
- One shared PDF extraction adapter owns Docling converter creation and
  `PdfPipelineOptions`.

**Docling contract:** Docling code enrichment is optional and defaults off.
`PdfPipelineOptions.do_code_enrichment` enables specialized processing for
`CodeItem`; Docling's enrichment docs state that code understanding processes
`CodeItem` and sets `code_language`. Installed Docling 2.86.0 confirms
`StandardPdfPipeline` inserts a `CodeFormulaVlmModel` enrichment stage that
filters `CodeItem` / formula items, sends item images to a VLM engine, and
overwrites `item.text`. Therefore CodeFormulaV2 is a code-recovery adapter,
not a default extraction mode.

**Approach:**
- Add `src/mmrag_v2/engines/pdf_plan.py`.
  - `PdfConversionPlan` is a dataclass and the single PDF extraction policy
    object.
  - `build_pdf_conversion_plan(...)` is the only function allowed to create a
    plan from diagnostics, profile, config, and cheap code evidence.
  - The plan contains structural flags, OCR policy, image/table options,
    `needs_code_enrichment`, code-evidence reason/counts, remote-enrichment
    settings, refiner threshold override, and config hashes.
- Add `src/mmrag_v2/engines/docling_adapter.py`.
  - `DoclingPdfAdapter` / `PdfPipelineOptionsFactory` is the only code allowed
    to instantiate Docling PDF converters or `PdfPipelineOptions`.
  - The adapter returns `UniversalDocument` and preserves Docling provenance,
    `CodeItem` labels, bboxes, page dimensions, and item images needed by
    downstream recovery.
- `batch_processor.py`, `processor.py`, and `engines/pdf_engine.py` must
  consume the same `PdfConversionPlan`; none may independently infer Docling
  options.
- Preserve native `CodeItem.text` when the extracted code is already
  structurally sound.
- Decide code preservation/recovery before chunking, refinement, or fallback
  text repair can flatten/merge code layout.
- Run code recovery only for broken `CodeItem` / code-candidate regions after
  cheap evidence proves code-heavy/code-candidate content.
- Prefer region-level recovery when available. If Docling only exposes
  document-level `do_code_enrichment`, enable it only after
  `needs_code_enrichment=True`; never from profile or encoding corruption
  alone.
- Remote code-recovery contract:
  - Uses only `[code_enrichment]` config and `code_enrichment.api_key`.
  - Never falls back to `vlm.api_key`, `refiner.api_key`, or CLI `--api-key`.
  - Carries explicit `enabled`, `provider`, `base_url`, `model`, `timeout`,
    and `concurrency`.
  - If using Docling-native remote services, construct `ApiVlmOptions` /
    `CodeFormulaVlmOptions` in the adapter with `enable_remote_services=True`.
  - If using a repo-local region-crop client, send only `CodeItem` /
    code-candidate crops and return text plus language; do not send whole
    documents.
  - Keep client-local CodeFormulaV2 as diagnostic/fallback. Preferred
    inference target is a stronger local-network host; cloud is optional if
    policy allows.

**Implementation phases:**
1. Add `PdfConversionPlan` and tests proving Ayeva/Chaubal/Fluent/Combat
   decisions without running full conversions.
2. Move all Docling `PdfPipelineOptions` / `DocumentConverter` construction
   into one factory/adapter with golden tests for batch, direct, and UIR paths.
3. Replace duplicated converter creation in `batch_processor.py`,
   `processor.py`, and `engines/pdf_engine.py` with calls to the shared
   adapter.
4. Add bridge tests at each object boundary: CLI process -> plan, CLI batch ->
   plan, batch -> processor, processor -> adapter, PDFEngine -> adapter,
   adapter -> Docling options.
5. Add a static guard test that fails if `PdfPipelineOptions(` or
   `DocumentConverter(` appear outside the adapter/factory and explicitly
   allowed tests.
6. Only then run targeted page/region probes and acceptance smoke tests.

**Hard constraints:**
- Do not add profile-specific `do_code_enrichment=True`.
- Do not let `has_encoding_corruption` trigger CodeFormulaV2 by itself.
- Do not weaken negative tests to fit the implementation.
- Do not add any new Docling `PdfPipelineOptions` or `DocumentConverter`
  construction site outside `docling_adapter.py`.
- Do not leave PDF extraction bypassing UIR as the final architecture.
- Do not add new governance docs for this refactor; update this plan,
  `DECISIONS.md`, and the checklist only when scope or evidence changes.

### 5b. Post-Docling Sanity Pass (successor to §5)

**Status:** `complete` (2026-05-03). Owned by
`docs/PLAN_DOCLING_POSTPROCESSOR.md`; summarised here so the v2.7 arc is
self-contained.

Four Docling 2.86 failure modes that surfaced on born-digital novels
(HARRY Potter as canonical fixture) are fixed at the
`DoclingPdfAdapter.convert()` seam:

1. **Reading-order y-sort** — items per page sorted by `(-bbox.t, bbox.l)`.
2. **Drop-cap promotion** — both standalone-glyph merge AND inline
   trailing-glyph heal. The latter is what Docling 2.86 actually emits:
   the drop cap "M" is *glued to the END* of the same TextItem
   (`"r. and Mrs. Dursley...nonsense. M"`), not as a separate item.
   The original plan assumed a separate item; reality required the
   complementary inline heal.
3. **Label-leak filter** — both `meta.classification` (via
   `blocked_meta_names`) and legacy `PictureClassificationData`
   annotations (via custom picture serializer that strips them even
   when a caption is present — the original "no caption" rule was
   insufficient).
4. **OCR gating** — `bitmap_area_threshold` raised from Docling's 0.05
   default to 0.75, auto-bumped to 0.92 for `digital_literature` /
   `digital_magazine` / `image_heavy_magazine`.

**Routing (the trigger):** new `DIGITAL_LITERATURE` profile across
`orchestration/profile_classifier.py`, `orchestration/strategy_profiles.py`
(`DigitalLiteratureProfile` class + ProfileManager registry +
classifier→strategy `type_mapping`), and
`orchestration/strategy_orchestrator.py` (`PROFILE_TO_DOC_TYPE` mapping).
The classifier auto-picks `digital_literature` for native-digital novels.

**Hard constraint surfaced:** `processor.py:2072` was bypassing the
adapter (see §5 followup above). A future static guard should ban raw
`self._converter.convert(...)` calls outside the adapter.

### 6. Contextual Retrieval (Anthropic approach)

**Status:** `complete` (2026-05-01). Builder `mmrag_v2.chunking.contextual_retrieval.build_contextualized_text` prepends breadcrumb + heading + truncated prev/next + non-text modality marker before canonical content; `IngestionChunk.contextualized_text` schema field added (optional, never read by QA); `scripts/ingest_to_qdrant.py` wires text+table modalities through the builder with a `--no-contextual` byte-stable rollback. AGENT-CONTEXTUAL-01..07 invariants in `docs/DECISIONS.md`. Drift guard `tests/test_contextual_retrieval.py::test_no_contextual_marker_strings_in_production_code` fails the day a marker leaks into production code outside the allowlist. See `docs/QUALITY_SNAPSHOT_2026-05-01.md` "Contextual Retrieval (Anthropic approach)".

**Problem:** Chunks lose context when embedded in isolation.

**Approach:** Per Anthropic's research, prepend 50-100 tokens of
chunk-specific context before embedding. HybridChunker's `contextualize()`
does this partially — v2.7 should ensure the contextualized text is what
gets embedded, not overwritten by the refiner.

**Implementation:** Fix the refiner/contextualize ordering (partially done
in v2.6), ensure `ingest_to_qdrant.py` uses contextualized text.

## Success Criteria For Next-Phase Refactor

| Gate | Required Result |
|------|-----------------|
| Single Docling options authority | No `PdfPipelineOptions(` or `DocumentConverter(` construction outside `src/mmrag_v2/engines/docling_adapter.py` and explicit tests. |
| UIR boundary | PDF extraction emits `UniversalDocument` before OCR/VLM/refiner processing; no new direct Docling-item-to-chunk path is added. |
| Plan ownership | `PdfConversionPlan` is built only by `build_pdf_conversion_plan(...)`; CLI process, CLI batch, BatchProcessor, V2DocumentProcessor legacy glue, and PDFEngine all consume it. |
| Structural flags | `has_encoding_corruption`, `has_flat_text_corruption`, and related diagnostics survive every boundary needed for OCR/refiner/code decisions. |
| Code-enrichment guard | Ayeva/Chaubal/Fluent/Combat decision tests prove code-heavy docs can enable selective recovery and non-code/encoding-only docs do not. |
| Remote key isolation | `code_enrichment.api_key` is independent; no fallback to VLM/refiner/API keys. |
| Regression suite | Unit tests pass; static guard tests pass; targeted Ayeva/Chaubal/Fluent probes pass before broad conversions. |
| Acceptance smoke | `scripts/smoke_multiprofile.sh` reports `GATE_PASS` + `UNIVERSAL_PASS` for all rows. |

## Execution Order

1. Add the plan dataclass/builder and pure decision tests.
2. Add the adapter/factory and static guard test.
3. Move `engines/pdf_engine.py` first so the UIR path becomes canonical.
4. Move direct `processor.py` PDF converter use behind the adapter.
5. Move `batch_processor.py` converter use behind the adapter without changing
   batching/export behavior.
6. Delete or quarantine dead duplicated policy after parity tests pass.
7. Run targeted Workstream B probes, then the full unit suite, then
   `smoke_multiprofile.sh`.

## References

- Gemini Pro audit, April 7, 2026
- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [Docling Pipeline Options](https://docling-project.github.io/docling/reference/pipeline_options/)
- [Docling Enrichment Features](https://docling-project.github.io/docling/usage/enrichments/)
- [Docling Code & Formula Example](https://docling-project.github.io/docling/examples/code_formula_granite_docling/)
- [Microsoft Azure Architecture Center: Pipes and Filters](https://learn.microsoft.com/en-us/azure/architecture/patterns/pipes-and-filters)
- [Alistair Cockburn: Hexagonal Architecture / Ports and Adapters](https://alistair.cockburn.us/hexagonal-architecture)
- [Docling Issue #287: Heading Hierarchy](https://github.com/docling-project/docling/issues/287)

## Updated Architecture (from Gemini Pro, April 7 2026)

### Principle: Stop deleting artifacts. Start validating multimodally.

The v2.6 approach of string-length rules and hardcoded skips is overfitting.
The v2.7 approach uses OCR confidence, VLM descriptions, and POS tagging
as validation signals — not just text pattern matching.

### 1. CorruptionInterceptor (replaces threshold-based refiner bypass)
- Per-bbox OCR patching: when a chunk contains /C211 or fails dictionary
  token ratio, re-extract ONLY that bbox via OCR
- Keep HybridChunker structure (headings, tables, hierarchy)
- Replace only the corrupted text spans

### 2. POS Boundary Logic (replaces character-count orphan stripping)
- Check if trailing word is a "Hungry Operator" (BY, FOR, OF, WITH)
- If next chunk starts with Proper Noun → merge operator into next chunk
- "END", "VOID", "N/A" are Nouns → stay where they are
- Language-agnostic: prepositions are structurally incomplete without objects

### 3. Vision-Gated Hierarchy (replaces quote/credit pattern filters)
- When heading detected on high-image-density page, check VLM description
- If description contains "Cover", "Logo", "Sketch" → demote heading to
  "Front Matter" via MetadataRefiner pass
- Uses multimodal signals, not text heuristics

### 4. Content-Type Classification (replaces hardcoded search_priority)
- Lightweight classifier or regex-weighted score per chunk
- High density of legal/boilerplate tokens → search_priority: "low"
- Works globally for all books and papers, not document-specific
