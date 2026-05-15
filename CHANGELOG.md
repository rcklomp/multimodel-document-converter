# Changelog: MM-Converter-V2

All notable changes to this project will be documented in this file. Current behavior is governed by `AGENTS.md`, `docs/DECISIONS.md`, `docs/QUALITY_GATES.md`, and `docs/ARCHITECTURE.md`; archived SRS files are historical references.

> **Versioning note:** Historical entries before the `v2.4.x` line used an internal `v18.x` milestone scheme during rapid iteration and test/fix cycles. Only stable or decision-worthy checkpoints were recorded, so intermediate builds are intentionally omitted. From `v2.4` onward, entries follow the current public semantic line.

## [v2.10-dev] — unreleased

PLAN_V2.10 Phases 1–7 closed locally. All seven named root-cause
classes are `validated-local`; the v2.10 production tag remains gated
on Phase 8 full-corpus strict-gate re-verification, Qdrant rebuild,
AFTER snapshot, and release tagging (see `docs/PLAN_V2.10.md`).

### v2.10 Phase 7 — `KI_EPUB_EXTRACTION_LANE_REWRITE` (KI EPUB) (2026-05-15, `validated-local`)

- **Spine-order EPUB extraction.** `_epub_to_html` now walks
  `book.spine` and injects `__MMRAG_EPUB_CH_NNNN__` chapter markers in
  canonical reading order before the HTML is handed to Docling.
- **Synthetic EPUB pagination.** `_apply_epub_synthetic_pagination`
  rewrites EPUB chunks with virtual page numbers
  `chapter_1based * 1000 + position_in_chapter // 5`, the documented
  EPUB bbox sentinel `[0,0,1000,1000]`,
  `extraction_method="epub_html"`, regenerated unique chunk IDs, and
  per-synthetic-page dedup.
- **EPUB-aware strict gate.** `scripts/qa_full_conversion.py` detects
  `.epub` source paths, enumerates spine chapters via `ebooklib`, and
  replaces PDF page coverage with chapter coverage. `MISSING_CHAPTERS`
  is a documented advisory only for contiguous leading/trailing
  low-content structural spine items; internal or content-bearing
  missing chapters remain failures.
- **Tests and evidence.** `tests/test_epub_extraction_lane.py` adds
  EPUB-lane coverage and `tests/test_qa_advisory_promotion.py` pins the
  advisory code. KI EPUB strict gate:
  `QA_PASS_WITH_ADVISORIES: failures=0 warnings=1`; ChatGPT EPUB
  regression control: `QA_PASS_WITH_ADVISORIES: failures=0 warnings=1`.
  Full pytest: **966 passed, 14 skipped, 0 failed**. Smoke:
  **11/11 GATE_PASS + 11/11 UNIVERSAL_PASS**.
- **Smoke runner stability.** `scripts/smoke_multiprofile.sh` now
  defaults to offline model resolution and `TORCH_COMPILE_DISABLE=1`
  for deterministic Apple-Silicon smoke runs; callers can override the
  environment variables when intentionally exercising online model
  resolution or `torch.compile`.

### v2.10 Phase 6 — `OCR_PATH_HEADING_PROPAGATION` (Firearms) (2026-05-15, `validated-local`)

- **Ordered OCR-lane heading attribution.** `Region.is_heading` and
  `ProcessedChunk.is_heading` carry Docling section-header/title labels
  through the layout-aware OCR lane. `BatchProcessor` attributes each
  OCR chunk by ordered push+read against `ContextStateV2`, while the
  older page-level promotion path remains only as the VLM/Tesseract
  full-page fallback.
- **Central heading validation.** `ContextStateV2.get_section_heading()`
  skips the level-0 document title breadcrumb; `is_valid_heading`
  rejects terminal-period sentence-shape headings and numbered-prefix
  body-case strings while preserving real numbered/question/exclamation
  headings.
- **Audit-fix text repair.** `BatchProcessor._repair_infix_step_numbers`
  repairs OCR/multi-column numbered-step infix artifacts and
  behaviorally mirrors the audit detector after its newline and
  stop-word post-filters. Parity is pinned by
  `tests/test_infix_step_number_repair.py`.
- **Evidence.** Firearms strict gate:
  `QA_PASS: failures=0 warnings=0`; HEADING coverage 1091/1094
  (0.997); TEXT `infix_artifacts` 148 → 0; targeted enrichment cleared
  264 pending shadow chunks. Earthship scanned-class regression:
  HEADING 549/549 unchanged. Phase 6 added 93 regression cases; full
  pytest after Phase 6 was **953 passed, 14 skipped, 0 failed**.

### v2.10 Phase 5 — `HYBRID_CHUNKER_HEADING_PROPAGATION` (Devlin) (2026-05-13, `validated-local`)

- **Single propagation site at export boundary.** `_propagate_headings(all_chunks)`
  removed from the mid-pipeline call site; replaced with one canonical
  `_propagate_headings(export_chunks)` after all filters/splitters. Pins
  the "only one site" contract via source introspection in
  `tests/test_vision_aided_front_matter.py`.
- **Heading validator tightened.** `is_valid_heading` in
  `src/mmrag_v2/state/context_state.py` now rejects: (a) code/JSON prefixes
  (`def`, `class`, `return`, `import`, `>>>`, `{`, `[`, ` ``` `); (b) repeated-token
  compact-alpha (`'Type Type TypeTypeTypeType'`); (c) repeated-word density
  (≥50% of a 3+-word heading, ≥3 occurrences). No parallel
  `_valid_export_heading` exists — the validator is the single source.
- **Generic-bucket carry block.** `_GENERIC_CARRY_HEADINGS = {"start", "front matter"}`
  blocks Docling-generic labels from seeding forward carry. `"Start"`
  stripped from its bearing chunk; `"Front Matter"` remains but does not propagate.
- **No `chunk_dict` mutation at write time.** Heading edits happen on the
  `IngestionChunk` model before serialization.
- **Tests.** `tests/test_hybrid_chunker_heading_propagation.py` adds 7 cases
  (4 positive including Devlin-shape batch boundary, 3 negative including
  garbage/code rejection and real-heading-wins-over-garbage).
- **Evidence.** Devlin strict gate: `HEADING PASS coverage 783/790 (99%)`,
  null_headings=7 (legitimate front-matter pages 3–7). Garbage strings
  each have 0 chunks attributed. Full pytest: **860 passed, 14 skipped,
  0 failed**. Smoke: **11/11 GATE_PASS + 11/11 UNIVERSAL_PASS**.
  Committed in `f3d8478`.

### v2.10 Phase 4 — `CROSS_PAGE_SPLIT_PAGE_ATTRIBUTION` (Cookbook + Distilled) (2026-05-13, `validated-local`)

- **Per-page text split via `prov.charspan` slicing.** New helper
  `_split_doc_chunk_text_by_page` walks each item's `prov` list, slices
  the item's full text by charspan per page, and intersects with `dc.text`.
  Bare `DocItem` references are dereferenced via `self_ref` against
  `doc.texts`. Replaces the v2.9 same-merged-text broadcast that dropped
  page attribution.
- **`[CROSS_PAGE_CONTINUED]` marker reserved for total failure.** Partial
  reconstruction emits only real chunks (no markers); total failure emits
  one marker at the *earliest* contributing page (not one per page).
- **BatchProcessor page-scoped overlap-trim.** `_deduplicate_chunk_overlap`
  now only trims same-page consecutive chunks; cross-page skips the trim
  regardless of content identity. Closes Cookbook p397
  `DOCLING_DUPLICATE_DOC_CHUNK_OVERLAP_TRIM`.
- **Micro-merge skips fallback markers.** `_merge_micro_text_chunks` skips
  chunks with `extraction_method == "hybrid_chunker_pagesplit_fallback"`
  on both sides. Prevents marker text concatenation onto neighbor prose.
- **Subtitle-continuation promotion.** `_looks_like_subtitle_continuation`
  promotes short single-line PARAGRAPH chunks to `ChunkType.HEADING` when
  structural signals match (length<30, no terminal punctuation, non-empty
  parent_heading distinct from content, first word is English connector).
  Closes HarryPotter `LITERATURE_MICRO_GATE_TUNE_AFTER_CROSS_PAGE_FIX`.
- **Audit-side heading/title exemption.** Three gate scripts now treat
  `chunk_type ∈ {code, heading, title}` as non-paragraph structural
  content exempt from `micro_non_label` counter. Thresholds unchanged.
- **Tests.** `tests/test_cross_page_split_page_attribution.py` adds 27 cases
  (p209 URL attribution, p208 content regression, three-page merge,
  marker reservation, bare-DocItem dereferencing, page_offset handling,
  overlap-trim page scoping, subtitle-continuation promotion, audit
  micro_non_label exemption).
- **Evidence.** Cookbook: cross-page-split MISSING_PAGES 4→0 (63/128/365/397
  closed); markers 62→0; `micro_non_label_ratio=0.002`. Distilled:
  cross-page MISSING_PAGES=0; markers=2 (genuine fallback); `micro_non_label_ratio=0.008`.
  Smoke: **11/11 GATE_PASS + 11/11 UNIVERSAL_PASS**. Full pytest:
  **853 passed, 14 skipped, 0 failed**. Committed in `8effdfd`.

### v2.10 Phase 3 — `B4B_FULL_DOC_PICTURE_DEDUP` (2026-05-12, `validated-local`)

- **pHash dedup page-coverage carve-out.** At
  `BatchProcessor`'s finalize-time export loop
  ([batch_processor.py:3364](src/mmrag_v2/batch_processor.py#L3364)),
  the cross-page pHash registry (Hamming threshold 10) was
  rejecting Earthship's hand-drawn cross-section illustrations on
  consecutive pages as near-duplicates, leaving image-only pages
  with 0 chunks and a strict-gate `MISSING_PAGES` failure
  (Earthship p109; extraction_method `docling` — the Docling chunk
  is preserved, not synthesized). The export loop pre-computes
  `_phash_image_only_pages` and maintains
  `_phash_pages_with_exported_image` (updated on BOTH unique-image
  emission AND preserved-duplicate emission); a duplicate IMAGE is
  preserved only when the page is image-only AND no IMAGE has yet
  been exported for it (logged
  `[PHASH-PAGE-COVERAGE] Preserving`). Decision factored as the
  pure static helper
  `BatchProcessor._phash_carve_out_should_preserve_duplicate` so
  production and tests cannot drift apart.
- **SHADOW-EXTRACTION page-coverage-aware threshold.** At
  [processor.py:_run_shadow_extraction](src/mmrag_v2/processor.py),
  the historical `300×300 OR area ≥ 40 %` gate was rejecting
  Python_Distilled's chapter-intro diagrams (453×258–290 points,
  24–27 % page area) — these pages then had 0 chunks of any
  modality and registered `MISSING_PAGES` at the strict gate. The
  new static helper `V2DocumentProcessor._shadow_image_meets_threshold`
  keeps the standard threshold for pages with prior chunks and
  drops the size floor to 200×200 (area floor unchanged at 40 %)
  for pages with no prior chunks. `process_document` now tracks
  `pages_with_prior_chunks` (TEXT yielded + pending IMAGE/TABLE +
  `_hybrid_text_chunks`) and threads it into the shadow path.
- **Tests.**
  `tests/test_phash_dedup_page_coverage.py` adds 9 regression
  tests pinning both contracts: the pHash carve-out's cases
  (image-only-page preserved; text-bearing page drops as before;
  second duplicate on the same image-only page drops;
  unique-then-duplicate on the same image-only page drops;
  direct pin on the pure decision helper covering the
  page-already-covered, text-bearing, and missing-page-number
  branches), the SHADOW threshold's three cases (mid-size
  relaxes; tiny icons still filtered; full-page editorial image
  passes both lanes), and a guard that the image-only-page set
  is computed from `export_chunks` rather than survivors. The
  last two tests were added after the 2026-05-12 post-close audit
  that flagged a bookkeeping gap: the carve-out previously
  tracked "page had preserved duplicate" instead of "page has
  any exported IMAGE", which could let a duplicate slip through
  if a unique image had been emitted earlier on the same
  image-only page.
- **Evidence.** Earthship strict gate
  (`python scripts/qa_full_conversion.py
  output/Earthship_Vol1/ingestion.jsonl --source-pdf
  "data/technical_manual/Earthship_Vol1_How to build your own.pdf"`)
  reports `QA_PASS: failures=0 warnings=0`
  (baseline was `QA_FAIL: failures=1 — MISSING_PAGES: 109`).
  Python_Distilled strict gate reports
  `QA_PASS_WITH_ADVISORIES: failures=0 warnings=1` (advisory is
  pre-existing `ASSET_TINY`; baseline was `QA_FAIL: failures=1 —
  MISSING_PAGES: 6, 686, 688, 913`). Full pytest: **826 passed,
  14 skipped, 0 failed** (was 817 at end of Phase 2; +9 new
  tests including the 2 post-close-audit additions). Smoke
  multiprofile remains **11/11 GATE_PASS + 11/11 UNIVERSAL_PASS**.

### v2.10 Phase 2 — `TEXT_INTEGRITY_SCOUT_FULL_DOC_SENSITIVITY` (2026-05-12, `validated-local`)

- **Per-batch TextIntegrityScout trigger.** New pure module
  `src/mmrag_v2/validators/text_integrity_scout_trigger.py` exposes
  `any_batch_fires(...)` over universal page-shape rules
  (per-batch variance ≥ 30 % OR ≥ 2 pages where source ≥ 100 chars
  and emitted text < 50 chars, with a 500-char batch source floor).
  `BatchProcessor._per_batch_shortfall_fires(...)` computes per-page
  source-text length once via PyMuPDF and feeds the helper using
  `split_result.batches` as the partition.
  `_run_text_integrity_scout(force_run=...)` now accepts the verdict
  and bypasses the doc-level variance gate when localized shortfall
  is detected; the post-scout merge logic keeps recovery output
  whenever the scout produced extra chunks (not only when doc-level
  variance was already bad). Closes Fluent_Python pp 8 / 10 / 11
  MISSING_PAGES on the 770-page full-doc path.
- **Quarantine parallel-site fix.** During Phase 2 validation,
  diagnostic snapshots localized a residual MISSING_PAGES regression
  (Fluent pp 125 / 126 / 136) to `_quarantine_corrupted_text_chunks`.
  The detector used the single-pattern-match `has_encoding_artifacts`,
  whereas the parallel patcher `patch_corrupted_chunks` is gated on
  the doc-level `has_encoding_corruption` flag — a parallel-site
  audit gap. Detector swapped to the ratio-based
  `is_irreparably_corrupt` so isolated ``\xHH`` literal escapes
  inside legitimate prose (Python REPL output describing UTF-8 byte
  literals in the encodings chapter) are preserved while the
  existing Combat p66 / Cronin TOC contract holds.
- **Corpus probe.** `scripts/probe_phase2_scout_threshold.py`
  simulates per-batch shape across every
  `output/<doc>/ingestion.jsonl` and records firing batches in
  `output/probe_phase2_scout_threshold/probe_summary.json`. 8 docs
  fire at the chosen threshold; for the seven non-Fluent cases the
  scout's existing per-page primary_chars ≥ 50 guard means no
  spurious recovery chunks are added.
- **Tests.** `tests/test_text_integrity_scout_per_batch_trigger.py`
  adds 10 regression tests pinning the trigger contract and the
  quarantine low-ratio-hex-escape behavior. Existing quarantine
  contract suites
  (`tests/test_corruption_quarantine_toc_exemption.py`,
  `tests/test_finalization_bridge.py::TestCorruptionQuarantineBridge`,
  `tests/test_export_integrity.py::test_full_pipeline_writes_valid_jsonl`)
  all remain green.
- **Evidence.** Fluent strict gate
  (`python scripts/qa_full_conversion.py
  output/Fluent_Python/ingestion.jsonl --source-pdf
  "data/technical_manual/Fluent Python Luciano Ramalho 2015.pdf"`)
  reports `QA_PASS_WITH_ADVISORIES: failures=0 warnings=1`
  (advisory is the pre-existing `ASSET_TINY` 7-asset finding,
  unrelated to Phase 2). Full pytest: **817 passed, 14 skipped,
  0 failed** (was 807 before Phase 2; the 10 new tests are the
  ones above).

## [v2.9.0-rc1] — 2026-05-12 (strict-gate close, RC tagged)

Tag `v2.9.0-rc1` created on commit `3e06d1b` (`main`, pushed to
GitHub). This is the v2.9 ship state; no intermediate final
`v2.9.0` tag is planned. The 8 signed deferrals carry forward as
v2.10 production-tag blockers (see `docs/DECISIONS.md`
"v2.9.0-rc1 Signed Deferrals").

### Closing state

- Strict gate (`scripts/qa_full_conversion.py --source-pdf --allow-warnings`):
  **26 PASS / 0 WARN / 8 FAIL** across 34 canonical docs (12
  `QA_PASS` + 14 `QA_PASS_WITH_ADVISORIES`; all 8 FAILs signed v2.10
  deferrals).
- Test suite: **806 passed, 14 skipped, 0 failed** (was 736 at
  Phase 4 close, +70 net new regression tests).
- Qdrant `mmrag_v2_8`: rebuilt to **30,461 points**
  (status=green, 4096-dim llava, 10h15m wall time).
- BEFORE state: `docs/QUALITY_SNAPSHOT_2026-05-11_v2.9_strict_gate_full_corpus.md`
  (9 PASS / 8 WARN / 17 FAIL — first full-corpus run).
- AFTER state: `docs/QUALITY_SNAPSHOT_2026-05-11_v2.9.0-rc1_after.md`.

### Phases shipped this cycle (per docs/PLAN_V2.9.md)

- **B1**: U+FFFD universal-collapse sanitizer at chunk creation +
  two-site BatchProcessor exemption for `hybrid_chunker_pageskip*`
  output. Closes Cronin / Nagasubramanian / Sekar / Chaubal TOC
  page MISSING_PAGES (62 → 0 in those docs).
- **B2**: `_is_intentionally_blank_text` recognizes "This page
  intentionally left blank" boilerplate as `MISSING_PAGES_BLANK`.
- **B3 Step 2**: `_emit_section_header_only_page_chunks` emits a
  chunk for chapter-divider / part-opener pages whose only Docling
  items are `section_header`. Closes Devlin p170,
  Nagasubramanian p2, Sekar p2/p159/p228/p247.
- **B4.a**: Render-based blank-equivalent classification (mean>245
  AND std<20 AND text<200) + zero-text image-only placeholder
  variant (text=0 AND images>=1 AND mean>250). Closes
  Python_Distilled (697 → 4 missing), Devlin p2/p264, Chaubal p4.
- **Phase D**: Tiny-bbox iconography lane (bbox<1% → `simple`
  complexity). Closes Hybrid_EV "Logo icon." short-description
  flag.
- **Phase E**: Combat blank-asset filter widened (std<5 → std<10)
  + word-density gibberish-table detector (len>30K AND
  density<10 w/k). Closes Combat figure_36 p27 + p66 squadron
  roster table.
- **Phase G**: `QA_PASS_WITH_ADVISORIES` allowed PASS variant in
  `scripts/qa_full_conversion.py`. Advisory codes: `ASSET_TINY`,
  `PAGE_COUNT_UNKNOWN`, `SCRIPT_ADVISORY_FAIL`, and F4-conditional
  `VISION_HARD_FALLBACK_RATE`. Promotes 13 docs from WARN to PASS
  variant.
- **Phase H**: Targeted re-enrichment of B1 / B3 reconverted docs
  (170 chunks) + Combat re-enrichment (348 chunks). Closes
  SEMANTIC_FAIL image_placeholder_ratio on Cronin (→ pure
  QA_PASS), Nagasubramanian, Sekar, Chaubal.
- **Phase I**: `mmrag_v2_8` Qdrant drop-and-recreate from 34
  canonical post-recovery JSONLs.

### Governance

- New: `docs/DECISIONS.md` "Retrieval-Value Test" principle.
  Features that do not improve retrieval / embedding quality /
  factual query answering are omitted (blank-equivalent), not
  backfilled.
- New: `docs/QUALITY_GATES.md` "Advisory Warning Classes" section
  documenting the PASS variant.
- New: `docs/DECISIONS.md` "v2.9.0-rc1 Signed Deferrals" listing
  the 8 v2.10 carry-forward items with rationale and acceptance
  baseline.

### Anti-overfit accounting (no Path A repeats)

Every threshold and lane added this cycle has:
- A documented user-query-shape rationale per the Retrieval-Value
  Test.
- Corpus-wide false-positive validation (Phase B4.a: 0 FP on 15
  real-content sample; Phase D iconography lane: 216/4031 bbox<1%
  classes, all 14 short-description cases are legitimately
  icon-class; Phase E gibberish-density: 0 FP on 18 large tables;
  Phase G F4 condition: 100 % F4 coverage on the 5 affected
  docs).
- A negative-control regression test that asserts the rule does
  NOT fire on a clearly-different shape (Combat p66 still drops
  via the existing patterns; legitimate Preface acknowledgments
  not flagged as "intentionally blank"; non-icon assets keep the
  `complex` classification).

### Carry-forward to v2.10 (8 signed deferrals)

| # | Doc | Class |
|---|---|---|
| 1 | Firearms | `OCR_PATH_HEADING_PROPAGATION` (signed 2026-05-10) |
| 2 | KI_En_ChatGPT_Praktische_Gids | `KI_EPUB_EXTRACTION_LANE_REWRITE` (signed 2026-05-10) |
| 3 | Devlin_LLM_Agents | `HYBRID_CHUNKER_HEADING_PROPAGATION` |
| 4 | Python_Cookbook | `CROSS_PAGE_SPLIT_PAGE_ATTRIBUTION` |
| 5 | Python_Distilled | `CROSS_PAGE_SPLIT_PAGE_ATTRIBUTION` (partial) + `B4B_FULL_DOC_PICTURE_DEDUP` (partial) |
| 6 | Fluent_Python | `TEXT_INTEGRITY_SCOUT_FULL_DOC_SENSITIVITY` |
| 7 | Chaubal_PyTorch_Projects | `TEXT_LABEL_TOC_DENSE_INDEX_ROUTER_MISS` |
| 8 | Earthship_Vol1 | `B4B_FULL_DOC_PICTURE_DEDUP` |

Engine version on `main` reads **2.9.0-rc1**. Schema version stays
**2.7.0** (no chunk-shape change). The next production iteration is
v2.10; the 8 carry-forward deferrals must close there under the
unchanged strict gate.

## [Unreleased] — v2.9 development arc (superseded by v2.9.0-rc1 above)

> **2026-05-06 retraction.** A `v2.9.0` annotated tag was created
> on 2026-05-05 against a 32/34 PASS reading from the
> `qa_conversion_audit.py`-only gate, then deleted on 2026-05-06
> after a user-driven QA review showed the loose gate had missed
> several real defects (HARRY chapter-intro pages silently merged
> into adjacent pages; Combat Aircraft p4 lost its full-page imagery;
> Combat p66 emitted 73 byte-equal copies of a corrupted-table chunk;
> the Phase 5b enrichment script never updated the canonical
> ``content`` field). The original `[2.9.0]` entry below was
> marked `[Unreleased]` until the corpus cleared the strict
> four-gate acceptance bar (``scripts/qa_full_conversion.py``;
> documented in ``docs/TESTING.md``). That close happened
> 2026-05-12 with the `v2.9.0-rc1` tag (see above section).

### Added
- **Phase 1 (TOC/index page-loss closure) — committed 2026-05-07 in `df91061`**
  (`tests/test_hybrid_chunker_dense_page_router.py`,
  `tests/test_toc_index_page_contract.py` env-gated): dense-index
  page router via Docling's `document_index` label fast path +
  `MmragChunkingSerializerProvider(skip_pages=...)` so HybridChunker
  never tokenizes those pages; dedicated grid-traversal emitter with
  two-layer dedup (byte-equal cell collapse + entry-boundary regex
  split `(?<=\d)\s+(?=[A-Za-z])`) producing
  `extraction_method="hybrid_chunker_pageskip"`. Three layered
  empty-text-chunk safety nets prevent the strict-gate
  `empty_text_chunks` invariant from ever tripping
  (`_apply_oversize_breaker` filters empty parts; finalize stage
  drops empty text chunks before `IngestionMetadata` write;
  JSONL-write loop skips text rows whose content was zeroed by the
  technical-manual line-stripper sanitiser). 120 s SIGALRM kept as
  a load-bearing safety net, downgraded to ERROR with
  should-not-fire wording. Validation: full Kimothi (258 pages)
  reports `AUDIT_PASS / UNIVERSAL_PASS / HYGIENE_PASS`; Ayeva
  back-index probe reports per-page chars 76–105 % of source PDF
  text (closes prior −30 % token variance). Test suite **628 passed,
  14 skipped, 0 failed**. Static `recovery_page_coverage` guard
  passes; SIGALRM did not fire on any tested document.
- **Phase 1 chunk_id collision fix**
  (`tests/test_chunk_id_collision_v29.py`, 4 new tests): the schema
  factory `_generate_chunk_id` now hashes a per-document monotonic
  `position` argument so two chunks with byte-identical
  `(doc_id, page, modality, content)` get distinct chunk_ids.
  Production paths (BatchProcessor, V2DocumentProcessor, Mapper)
  each maintain a per-document counter and thread it through every
  factory call. Fixes the v2.8 surface where 427 within-file
  duplicate chunk_ids silently overwrote each other on Qdrant
  upsert (uuid5 from chunk_id).
- **Phase 2 refiner smart-routing**
  (`tests/test_cli_refiner_smart_routing.py`, 5 new tests): pure
  helper `cli._decide_enable_refiner` factors the decision into a
  single rule — explicit `--no-refiner` always wins; explicit
  `--enable-refiner` always wins; otherwise the config default
  (`cfg.refiner.enabled=true`) only fires when the diagnostic
  reports `has_encoding_corruption=True`. Fixes the v2.8 HARRY
  symptom (clean prose hammered qwen-plus per chunk). Parallel-site
  fix in `cli.batch_process`: the existing reference to
  `_refiner_explicitly_disabled` was a NameError-waiting-to-fire,
  now defined alongside `_refiner_explicitly_enabled`.
- **Phase 3 Rule 0/0c code-evidence guard**
  (`tests/test_classifier_rule_0c_tightening.py`, 6 new tests):
  `document_diagnostic._estimate_content_domain` counts
  "code-evidence pages" (line-starting Python keywords or fenced
  blocks) and suppresses BOTH the +0.4 weak-dialogue lane and the
  +0.8 full-novel lane when ≥2 sampled pages show code. Fixes the
  Ayeva misroute to `digital_literature` (Python f-strings + string
  literals were being counted as dialogue). HARRY still routes to
  `digital_literature` because novels show zero code keyword starts.
- **Phase 4 Firearms HARD REJECT**
  (`tests/test_classifier_firearms_route.py`, 4 new + 1 env-gated
  test): `profile_classifier._score_technical_manual` HARD-REJECTs
  long-form scanned docs (`is_scan=True AND image_density>=1.0 AND
  page_count>100`). Firearms and Earthship flip from
  `technical_manual` back to `scanned`, restoring HEADING coverage.
  AGENT-SPATIAL-20 unchanged — this is a profile-classifier scorer
  adjustment, not a per-profile spatial-threshold branch.
- **Phase 5b targeted VLM enrichment script**
  (`scripts/enrich_image_chunks_v29.py`): post-conversion image-only
  enrichment via cloud `qwen3-vl-plus`. Atomic tmp-file write-back;
  `--dry-run` for cost preview; resumable on crash. Hard-fallback
  policy via the existing `VisionManager` Source Sanctity harness.
  Local lane deferred to v2.10 per `docs/PLAN_V2.9.md` §Phase 5
  decision e.
- **Phase 5b acceptance test**
  (`tests/test_v29_image_enrichment_acceptance.py`, env-gated by
  `RUN_V29_VLM_ACCEPTANCE=1`): pins `vision_status≠pending`, no
  placeholder `visual_description`, `vision_provider_used=qwen3-vl-plus`
  across the canonical corpus.
- **Quality snapshot**: `docs/QUALITY_SNAPSHOT_2026-05-04_v2.9_after.md`.
- **Plan**: `docs/PLAN_V2.9.md` (superseded
  `docs/archive/PLAN_V2.8_PRODUCTION_GAPS.md`).

### Changed
- **`src/mmrag_v2/schema/ingestion_schema.py`**: factory functions
  `create_text_chunk` / `create_image_chunk` / `create_table_chunk`
  accept new `position: int = 0` parameter; `_generate_chunk_id`
  signature extended to take `position` (hashed into the seed).
- **`src/mmrag_v2/cli.py`**: removed eager config-default refiner
  enable at `cli.py:686`; replaced with `_decide_enable_refiner`
  helper invoked at the diagnostic-metadata block. Parallel-site
  fix in `batch_process`.
- **`src/mmrag_v2/orchestration/document_diagnostic.py`**: Rule 0
  and Rule 0c gated on `_code_evidence_pages < 2`.
- **`src/mmrag_v2/orchestration/profile_classifier.py`**:
  `_score_technical_manual` HARD-REJECTs Firearms-class signatures.
- **`scripts/convert_books.sh`**: `--no-refiner` dropped (Phase 2
  smart-routing makes it unnecessary). Wraps `python -m mmrag_v2.cli`
  in `conda run -n mmrag-v2` so the script works without the env
  pre-activated.
- **`src/mmrag_v2/version.py`**: `__engine_version__` 2.8.0 → 2.9.0;
  schema-version note clarifies that chunk_id values differ for
  affected chunks even though the field shape is unchanged.

### Fixed
- **chunk_id collisions across the canonical corpus.** v2.8 had 427
  within-file dupes (KI_En_ChatGPT 279, Devlin 76, Fluent 15, …);
  v2.9 broad reconversion target: 0.
- **Refiner per-chunk hammering on clean-prose docs.** HARRY
  conversion now runs without `--no-refiner` and produces
  `refinement_applied=0` text chunks (config default = on, but
  diagnostic says clean → smart-routing skips refiner).
- **Ayeva CODE FAIL.** v2.8 fresh produced `indentation_fidelity=0.83`
  (under 0.85 gate) because Ayeva routed to `digital_literature`,
  suppressing CodeFormulaV2. v2.9 routes Ayeva to `technical_manual`,
  CodeFormulaV2 engages.
- **Firearms HEADING FAIL.** v2.8 fresh produced HEADING coverage
  78% (under 80% gate) because Firearms routed to `technical_manual`,
  whose stricter heading-inheritance left 178/815 chunks
  orphan-headed. v2.9 routes Firearms back to `scanned`.
- **`mmrag_v2_8` placeholder image points.** v2.8 ingest left
  ~5,500 image points with `vision_status="pending"` and placeholder
  `visual_description`. v2.9 Phase 5b enriches every image with a
  real cloud-VLM description; image-side RAG retrieval restored.

### Removed
- `--no-refiner` flag from `scripts/convert_books.sh` (Phase 2 fix
  obviates the workaround).

### Notes
- Local NuMarkdown-8B-Thinking-mlx-8bits VLM lane stays deferred to
  v2.10 while the off-network endpoint is unreachable. v2.9
  enrichment script does NOT branch on local availability.
- Remote CodeFormulaV2 inference (`RemoteCodeFormulaOptions` /
  `ApiCodeFormulaOptions`) still not exposed by Docling 2.86 — v2.10
  followup if code-heavy reconversion frequency exceeds 1/week.

## [2.8.0] — 2026-05-04 — PLAN_V2.8 Production Gaps Closed

Engine version bumps to **2.8.0**. Schema version stays **2.7.0** (no
chunk-shape change; all v2.8 work is behavioral / pipeline-level).
Tagged as `v2.8.0` (annotated tag, commit `9726b43`).

### Added
- **Form acceptance class for invoices / short scanned docs** (Phase 5a).
  `scripts/evaluate_technical_manual_gates.py` now reads `total_pages`
  from ingestion metadata + counts only real `parent_heading` entries
  (auto-generated `[doc_id, "Page N"]` breadcrumbs do not count) →
  detects `document_type=form` for short scanned docs, skips the
  prose-calibrated `micro_non_label_ratio` and label-orphan checks,
  emits `GATE_PASS [form: ...]`. `scripts/qa_conversion_audit.py`
  emits `FORM_AUDIT_PASS` instead of the dismissive
  `UNSUPPORTED_HIERARCHICAL_RAG`. New form acceptance class documented
  in `docs/QUALITY_GATES.md`. SCAN0013 row in the smoke matrix flips
  from `GATE_FAIL micro_non_label_ratio=0.294` to `GATE_PASS [form]`.
  3 contract tests in `tests/test_form_audit_gate.py`.
- **Adapter-invocation static guard** (Phase 2 — v2.7 §5 followup).
  `tests/test_pdf_conversion_plan.py::test_no_raw_converter_invocation_outside_adapter`
  AST-walks every production `*.py` and rejects any
  `self._converter.convert(...)` / `self._docling_converter.convert(...)`
  outside the adapter. Promotes the v2.7 §5 rule from "construction
  guarded" to "construction + invocation guarded". 4 new guard tests
  (1 real-code + 3 synthetic positive/negative).
- **Cross-cutting Parallel-Site Audit principle** baked into
  `docs/archive/PLAN_V2.8_PRODUCTION_GAPS.md` §2b. Every production-code
  change must walk parallel call sites before designing the fix.
- **Phase 4 named contract tests** for the CodeFormulaV2 enable
  decision (`tests/test_code_enrichment_decision.py`):
  Chaubal positive, Fluent positive (non-regression control),
  Combat negative (encoding corruption alone must not trigger).
- **Phase 1 keyword-separator regression tests**
  (`tests/test_oversize_pua_fixes.py`): 4 new tests pinning the
  `\x01` → `; ` (prose) / `" "` (code) replacement contract.
- **Phase 3 ornament-glyph contract tests**
  (`tests/test_corruption_quarantine.py`): 3 tests pinning the
  Combat page-66 detection contract + Arabic/CJK passthrough.
- **Qdrant ingest collision-free point IDs**: deterministic UUID5
  derived from `chunk_id`. 6 regression tests in
  `tests/test_qdrant_point_id_collision.py`. Fixes the
  multi-doc-per-collection ingest workflow (was overwriting all
  docs with the largest single doc's chunks).
- **Overnight pipeline scaffolding**: `scripts/v28_overnight_pipeline.sh`
  (convert → audit → snapshot → commit) + `scripts/v28_build_after_snapshot.py`
  (parses qa_audit SUMMARY, splits canonical from legacy probes,
  emits delta column vs prior snapshot).
- **Quality snapshots**:
  `docs/QUALITY_SNAPSHOT_2026-05-03.md` (Phase 0 BEFORE) and
  `docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md` (Phase 5c AFTER
  with empirical Phase outcomes + known limitations).

### Changed
- **Schema validator `_strip_c0_controls`** (Phase 1, Workstream F).
  In `src/mmrag_v2/schema/ingestion_schema.py`, replaces runs of
  C0/DEL controls with `; ` (prose context) or `" "` (code context)
  instead of dropping them outright. Preserves the keyword-separator
  structure that PDFs occasionally use `\x01` for. Existing test
  `test_text_chunk_strips_c0_controls_but_preserves_newline_tab`
  updated to reflect the new replace-vs-strip semantics.
- **`src/mmrag_v2/processor.py`**: removed the dead else-branch at
  line 2079 (`else: result = self._converter.convert(...)`). The
  adapter is unconditionally constructed in `__init__` so the
  fallback was both dead code AND a guard violation.
- **`src/mmrag_v2/engines/pdf_engine.py`**: routed
  `_convert_with_docling` through the adapter so post-Docling
  sanity stages run (parallel-site fix per PLAN_V2.8 §2b).
- **`scripts/convert_books.sh`**: extended manifest from 24 to 34
  entries — every PDF/EPUB in `data/` now in scope, including the
  business forms and data spreadsheet (first-class per the form
  acceptance class). Added clear comment block on `--no-refiner
  --vision-provider none --no-cache` flag rationale (apples-to-apples
  vs Phase 0 baseline, plus refiner-config-default bug bypass).
- **`scripts/qa_conversion_audit.py`**: TEXT verdict now honors
  `is_form` consistently — fixes a contradictory
  `TEXT: PASS / FORM_AUDIT_FAIL (TEXT)` output where the form bypass
  worked at the label-print site but missed the fails-tracking site.
- Stale `data/scanned/HarryPotter_*.pdf` paths fixed in 3 sites
  (test fixture, classifier docstring, convert_books.sh) after the
  2026-05-03 `digital_literature` data move.

### Fixed
- **Workstream F (0x01 keyword separator)** — `A_comprehensive_review_on_hybrid_electri`
  fresh re-conversion: AUDIT_PASS, `ctrl_chunks=0` (was 1, total 4).
  Keyword chunk now reads `"Hybrid electric vehicle; Hybrid energy
  storage system; Architecture; ..."` — semantically intact.
- **Workstream C (Combat ornament-glyph)** — `Combat_Aircraft_August_2025`
  fresh re-conversion: AUDIT_PASS, `encoding_artifacts=0`,
  `high_corruption=0` (was 48, 79). Empirical no-op outcome — the
  shipped `CorruptionInterceptor` + quarantine + 2026-05-03
  post-Docling sanity stages already heal Combat on a fresh run.
- **Workstream B (Chaubal CodeFormulaV2)** — `Chaubal_PyTorch_Projects`
  fresh re-conversion: AUDIT_PASS, `indentation_fidelity=0.96`
  (was 0.54; target ≥0.85). CodeFormulaV2 engaged via the existing
  cheap-evidence trigger + Docling adapter `do_code_enrichment=True`
  plumbing.
- **HARRY page-30 reading-order acceptance test** — locator was
  matching chapter-heading text inside a downstream image-chunk's
  auto-injected breadcrumb leak (`"[Figure on page N] | Context: ... > THE VANISHING GLASS"`)
  and reporting a false swap. Fixed `_locate` to (a) only consider
  text-modality chunks, (b) recognize `parent_heading` matches at
  offset −1. Live HARRY pages-1-30 acceptance now passes.
- **Qdrant ingest point_id collision** — `scripts/ingest_to_qdrant.py:439`
  used `point_id = i + 1` (sequential per-file). Multi-doc-per-collection
  ingest overwrote 32 of 34 docs with the largest doc's chunks
  (only 1,690 / 22,587 chunks survived in the first ingest attempt).
  Fixed via deterministic UUID5 from chunk_id; second ingest landed
  22,137 / 22,160 unique chunks (100% of unique embeddable chunks).

### Pre-flight evidence (committed in 5b0e13d → 9726b43)
- `pytest tests/ -q`: **596 passed, 2 skipped, 0 failed.**
- `bash scripts/smoke_multiprofile.sh`: **11/11 GATE_PASS + 11/11 UNIVERSAL_PASS.**
- HARRY pages-1-30 live acceptance: **PASS.**
- `mmrag_v2_8` Qdrant collection: **22,137 points**, status green,
  768-dim vectors, `on_disk` storage. Sits side-by-side with the
  17 healthy `*_v2` per-doc collections; nothing was overwritten in
  user-owned data.

### Known limitations (v2.9 scope, documented in tag message)
1. `Ayeva_Python_Patterns` CODE FAIL (`indentation_fidelity=0.83` vs
   `0.85` gate). Profile misclassified as `digital_literature`
   (rule 0c misfire on a code-heavy book), suppressing the
   `needs_code_enrichment` cheap-evidence trigger. Net 0.22 → 0.83
   is a massive lift even without enrichment.
2. `Firearms` HEADING coverage 78% (gate ≥80%). Profile changed
   `scanned` → `technical_manual` between baselines; stricter
   heading-inheritance leaves 178 chunks orphan-headed. Same
   content fidelity, just less hierarchy annotation.
3. Within-file `chunk_id` collisions (427 across the corpus,
   largest contributor `KI_En_ChatGPT` 279). Schema generator
   collapses identical content (boilerplate footers, repeated
   page numbers) to the same ID. v2.9 fix: include the chunk's
   `i+1` position in the hash seed.
4. Refiner smart-routing (`cli.py:686`) enables refiner from
   `~/.mmrag-v2.yml` `refiner.enabled=true` regardless of
   `has_encoding_corruption`. v2.8 broad reconversion bypassed
   this with `--no-refiner` for apples-to-apples vs Phase 0
   baseline. v2.9 fix: gate the config-default enable on the
   diagnostic just like the explicit auto-override at `cli.py:1101`.
5. `mmrag_v2_8` collection has placeholder image descriptions
   (`vision_status: pending`) because `--vision-provider none` was
   used in the broad reconversion. v2.9 fix: targeted image-only
   enrichment script that reads existing JSONLs, runs VLM per
   image, updates `visual_description`/`refined_content`, and
   re-ingests just the image points.

## [Folded into 2.8.0] — Post-Docling Sanity Pass (2026-05-03)

Originally landed as `[unreleased]` and is now bundled into the **v2.8.0**
release. The full Added/Changed/Fixed breakdown is in the consolidated
**[2.8.0]** section above (look for the `Post-Docling reading-order y-sort`,
drop-cap promotion, label-leak filter, and OCR gating bullets). The earlier
verbatim duplicate has been removed to keep the changelog single-source.
Per-feature historical context is preserved in commit `3bdbe0f` (post-Docling
sanity pass + `digital_literature` profile) and in
`docs/archive/PLAN_DOCLING_POSTPROCESSOR.md` (the predecessor plan, marked shipped).

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

## [Folded into 2.8.0] — Milestone 1: Stabilize Extraction (2026-04-30 → 2026-05-01)

Originally landed as `[Unreleased]` and is now bundled into the **v2.8.0**
release. The full Added/Changed/Fixed breakdown is in the consolidated
**[2.8.0]** section above (look for the shared PDF extraction plan,
Plan Control Plane typed policy fields, refactor boundary closeout, and
Milestone 1 chunker-guard bullets). The earlier verbatim duplicate has been
removed to keep the changelog single-source. Per-feature historical context
is preserved in `docs/archive/quality_snapshots/QUALITY_SNAPSHOT_2026-05-01.md` (banner-annotated as
superseded by the 2026-05-04 baseline; selected metrics like Ayeva 0.93 do
NOT reflect the v2.8 fresh re-conversion).

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
