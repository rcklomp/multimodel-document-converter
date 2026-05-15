# Project Status

Last updated: 2026-05-13

Purpose: fast orientation for a new coding session. Read this before deeper project docs.

## Current Objective

**`v2.9.0-rc1` SHIPPED 2026-05-12** — tag `v2.9.0-rc1` on commit
`3e06d1b`, pushed to GitHub. `v2.9.0-rc1` is the v2.9 ship state;
**no intermediate `v2.9.0` final tag is planned**. The 8 signed
deferrals from the close-out carry forward as **v2.10 production-tag
blockers**. See `docs/DECISIONS.md` "v2.9.0-rc1 Signed Deferrals
(2026-05-11 close-out)" for the named list.

**Strict-gate state at RC1:** 26 PASS / 0 WARN / 8 FAIL out of 34
(12 `QA_PASS` + 14 `QA_PASS_WITH_ADVISORIES`). All 8 FAILs are signed
v2.10 deferrals. Test suite: **806 passed, 14 skipped, 0 failed**
(Phases 1-4 reconverts add +47 tests → 853 total).

**Qdrant `mmrag_v2_8`:** 30,461 points, status green (25,691 text +
4,379 image + 391 table). Rebuilt 2026-05-12 from the 34 canonical
post-recovery JSONLs via local Ollama llava (4096-dim, 10h15m wall
time). Sole collection in the local Qdrant — 16 sister `*_v2`
per-doc collections dropped during post-RC1 sanitization.

**Active canonical baseline:**
[`docs/QUALITY_SNAPSHOT_2026-05-11_v2.9.0-rc1_after.md`](QUALITY_SNAPSHOT_2026-05-11_v2.9.0-rc1_after.md)
(revised 2026-05-12 for vision-status precision, Qdrant collection
cleanup, Devlin Phase H catch-up, search-tool default re-tune
housekeeping, and Option-1 v2.9.0-final terminology cleanup; see
§10 Revision log).

**Next cycle:** v2.10. Plan at
[`docs/PLAN_V2.10.md`](PLAN_V2.10.md) (authored from
`PLAN_V2.10_DRAFT_PROMPT.md`); Phases 1-6 validated-local
(Phase 6 closed 2026-05-15 after the audit-fix iteration). Firearms
full strict gate returns
`QA_PASS: failures=0 warnings=0` with HEADING coverage 0.997,
smoke 11/11 GATE_PASS + 11/11 UNIVERSAL_PASS, Earthship_Vol1
full-doc regression HEADING coverage 100 % → 100 % (no drop),
**953 pytest passing**. The audit-fix iteration also closed the
pre-existing TEXT `infix_artifacts: 148 → 0` via the new
`_repair_infix_step_numbers` chunk-content repair and the VLM
`image_placeholder_ratio: 0.2424 → 0.0000` via targeted enrichment
of the 264 pending shadow chunks. Phase 7
(`KI_EPUB_EXTRACTION_LANE_REWRITE`) is the next implementation
target.

---

## v2.9 Execution History (archaeology only)

The phase-by-phase narrative that produced the RC1 close-out is
preserved below. **Authoritative ship state is in §Current Objective
above**; the close-out snapshot's "Phases shipped in this cycle" is
the canonical summary.

### v2.9 origins (Phase 0)

A v2.9.0 tag was created on 2026-05-05 against a 32/34 AUDIT_PASS
reading from `scripts/qa_conversion_audit.py` alone, then deleted on
2026-05-06 after a user-driven review surfaced multiple defects that
the single-script gate did not catch (HARRY chapter-intro pages
silently merged into adjacent pages; Combat p4 lost full-page
imagery; Combat p66 emitted 73 byte-equal corrupted-table copies;
Phase 5b enrichment never updated the canonical ``content`` field).

The v2.9 cycle landed real bug fixes on `main` (see "v2.9 in-flight
fixes" below) and adopted a stricter four-gate acceptance via
``scripts/qa_full_conversion.py`` (see ``docs/TESTING.md``). Phase 4
closed on 2026-05-10 with explicit user sign-off to defer two defects
to v2.10: Firearms `OCR_PATH_HEADING_PROPAGATION` and KI EPUB
`KI_EPUB_EXTRACTION_LANE_REWRITE`. Phase 5a (broad reconversion, 34/34
fresh JSONLs) and Phase 5b (cloud VLM enrichment, 4,269 complete /
113 hard_fallback / 0 pending) completed before the RC1 close.

**2026-05-11 update.** The first full-corpus strict-gate run
(`scripts/qa_full_conversion.py --source-pdf --allow-warnings`)
reported **9 PASS / 8 WARN / 17 FAIL out of 34**. The previous v2.9
plan was archived and replaced with a new recovery plan organized
around nine work-streams (Phase A diagnostic → I tag).

**`v2.9.0-rc1` ship state achieved 2026-05-11; tag landed 2026-05-12.**
AFTER snapshot at `docs/QUALITY_SNAPSHOT_2026-05-11_v2.9.0-rc1_after.md`.
Per `docs/DECISIONS.md` "v2.9.0-rc1 Signed Deferrals (2026-05-11
close-out)", the RC ships with 8 signed v2.10 deferrals (Firearms,
KI EPUB, Devlin, Python_Cookbook, Python_Distilled, Fluent_Python,
Chaubal, Earthship) covering all 8 remaining QA_FAIL rows. Strict
gate state: **26 PASS / 0 WARN / 8 FAIL** (12 `QA_PASS` + 14
`QA_PASS_WITH_ADVISORIES`).

### v2.9 in-flight fixes (committed)

- **chunk_id collision fix** — per-document monotonic ``position``
  hashed into the chunk-id seed; v2.8's 427 within-file dupes
  collapse to zero on a fresh broad reconversion.
- **Refiner smart-routing** — the config-default refiner only
  auto-enables when the diagnostic engine reports
  ``has_encoding_corruption=True``.
- **Cross-page DocChunk page-coverage split** — Docling's
  HybridChunker emits multi-page chunks; the chunker now emits one
  IngestionChunk per source page so chapter-intro pages aren't lost.
- **CorruptionInterceptor extended to TABLE modality** — Combat p66's
  squadron-roster table is now subject to the same patch+quarantine
  path as text.
- **FULL-PAGE-GUARD defers full-page assets** — three sites changed
  from discard → defer with ``vision_status='pending'``.
- **Phase 5b enrichment script writes canonical ``chunk.content``** —
  not just ``metadata.visual_description``.
- **``scripts/qa_full_conversion.py`` strict gate** — bundles
  audit + universal + hygiene + semantic_fidelity plus deterministic
  page-coverage / dup-excess / corruption / image-quality checks.
  Documented in ``docs/TESTING.md`` as the v2.9 acceptance bar.

---

## Active Baseline

- **`docs/QUALITY_SNAPSHOT_2026-05-11_v2.9.0-rc1_after.md`** (current
  v2.9 ship state — 26/34 PASS / 0 WARN / 8 FAIL; tag `v2.9.0-rc1`
  on commit `3e06d1b`).
- **`docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md`** (v2.8.0 SHIPPED
  reference baseline).
- `docs/QUALITY_SNAPSHOT_2026-05-09_v2.9_phase4_after.md` (historical;
  superseded by 2026-05-11 BEFORE snapshot).

`tests/test_docling_postprocessor_acceptance.py` (HARRY pages 13-30
reading-order fixture) is the binding regression test.

## Active Model/Endpoint State

Do not print or commit API keys.

Current local VLM setting:

- provider: OpenAI-compatible
- model: `NuMarkdown-8B-Thinking-mlx-8bits`
- base URL: `http://10.0.10.246:8000/v1`

Cloud comparison tested:

- provider: OpenAI-compatible DashScope endpoint
- model: `qwen3-vl-plus`

Observed behavior:

- local NuMarkdown is faster in the PCWorld harness after retry flow but needs many hard fallbacks
- Qwen3-VL-Plus gives richer visual descriptions and fewer hard fallbacks
- both models still read visible text, so model-agnostic enforcement is required

## Current Quality Summary

Source of truth: [`docs/QUALITY_SNAPSHOT_2026-05-11_v2.9.0-rc1_after.md`](QUALITY_SNAPSHOT_2026-05-11_v2.9.0-rc1_after.md)
(RC1 AFTER, revised 2026-05-12).

**Strict-gate (`scripts/qa_full_conversion.py --source-pdf --allow-warnings`)
at RC1 close-out: 26 PASS / 0 WARN / 8 FAIL out of 34.** All 8 FAILs
are signed v2.10 deferrals per `docs/DECISIONS.md`. PASS breakdown:
12 `QA_PASS` + 14 `QA_PASS_WITH_ADVISORIES` (the documented PASS
variant introduced in Phase G; see `docs/QUALITY_GATES.md` "Advisory
Warning Classes").

**Image enrichment:** 4,379 image chunks, 4,257 `vision_status=complete`
(97.2 %), 122 `vision_status=hard_fallback` with F4 sentinel (2.8 %),
0 pending. Devlin Phase H catch-up ran 2026-05-12 (the original
Phase H on 2026-05-11 missed Devlin's 67 pending chunks).

**Test suite:** 853 passed, 14 skipped, 0 failed (was 806 at
v2.9.0-rc1; +47 net regression tests across Phases 2–5, including
27 cross-page-split tests in
`tests/test_cross_page_split_page_attribution.py` and 14 audit/micro-gate
tests).

### Open work — v2.10 production-tag blockers

Per `docs/DECISIONS.md` "v2.9.0-rc1 Signed Deferrals (2026-05-11
close-out)", the seven named root-cause classes below carry forward as
v2.10 production-tag blockers (8 deferral rows over 7 classes — items
4 and 5 each cover two docs). No intermediate `v2.9.0` final tag is
planned.

**Remaining open blockers (1 of 7 classes, plus residual scope on row 2):**

1. `KI_EPUB_EXTRACTION_LANE_REWRITE` — EPUB lane structural gaps
   (no pagination, no bbox, heavy dedup excess). User signed off
   v2.10 deferral on 2026-05-10.

**Phase 6 `validated-local` (2026-05-15).**

2. `OCR_PATH_HEADING_PROPAGATION` — Firearms.
   `Region.is_heading` / `ProcessedChunk.is_heading` carry Docling's
   structural `section_header` / `title` label through
   [`src/mmrag_v2/ocr/layout_aware_processor.py`](../src/mmrag_v2/ocr/layout_aware_processor.py).
   [`BatchProcessor._attribute_ocr_chunk_heading`](../src/mmrag_v2/batch_processor.py)
   walks the `ProcessedChunk` stream in order; heading-marked chunks
   push into `ContextStateV2` via `update_on_heading` BEFORE state is
   read for that chunk, so within-page ordering is preserved (body
   chunks before the first heading on a page inherit prior-page
   state or `None`; chunks after a same-page heading inherit it;
   multiple headings switch attribution at the right position).
   The new `ContextStateV2.get_section_heading()` skips the level-0
   doc-title initial breadcrumb. `_promote_ocr_section_headers`
   remains as a fallback for VLM-fullpage / Tesseract-fullpage paths
   that emit a single synthesized chunk per page.

   The central `is_valid_heading` validator was tightened by two
   universal rules: **terminal-period** sentence-shape (≥5 words
   ending in `.` → reject; `?` and `!` are accepted so real question
   / exclamation headings pass) and **numbered-prefix body-case**
   shape (numbered prefix + ≥2 lowercase content words → reject).
   All Phase 5 audit-named garbage rejections retained; all Phase 5
   Devlin real headings still pass.

   The **single-page push gate** (`self._doc_total_pages > 1`)
   applies on BOTH the per-chunk path
   (`_attribute_ocr_chunk_heading`) and the fallback path
   (`_promote_ocr_section_headers`) so the canonical
   `scanned/0013_140302111325_001` invoice form-detection contract
   holds.

   **Numeric evidence:** Firearms strict-gate HEADING audit
   `PASS coverage 1091/1094 (99.7%)` (was 0.72; the 3 NULL chunks are
   page-1 front-matter correctly unattributed under ordered design);
   top-5 all real chapter/section titles. 70 OCR-lane tests + 23
   infix-repair tests + 7 Phase 5 + 27 Phase 4 + 8 vision-aided pin
   all pass under bare pytest (**953 total, 14 skipped**). Smoke remains
   **11/11 GATE_PASS + 11/11 UNIVERSAL_PASS**. Earthship_Vol1
   full-doc regression: HEADING coverage 1.00 → 1.00 (no drop);
   `micro_non_label_ratio=0.033` (limit 0.12).

   **Audit-fix iteration (2026-05-15):** the two scope-orthogonal
   defects flagged by the first audit are now closed without
   weakening any threshold.

   - TEXT `infix_artifacts: 148 → 0`. The hits were real OCR
     artifacts (Firearms multi-column numbered-instruction-step
     layouts where the step number was mashed into the trailing
     word of the preceding paragraph). Closed by the new
     `BatchProcessor._repair_infix_step_numbers` chunk-content
     repair. Detection **behaviorally mirrors**
     `scripts/qa_conversion_audit.py::_INFIX_RE` **after the
     audit's newline / stop-word post-filters** (the production
     regex collapses the audit's ``\s+`` + ``"\n" in between``
     post-filter into a single ``[ \t]+`` on the prev→num side and
     reproduces the audit's left-context exclusion and
     short-word / stop-word filters explicitly). Parity is pinned
     by 23 cases in `tests/test_infix_step_number_repair.py`,
     including an audit-detector parity test that re-applies the
     audit to repaired content and asserts the count drops to 0.
     Universal heuristic, not Firearms-specific; cross-corpus
     verified to not regress Devlin / Cronin / Earthship / Greenhouse.

   - VLM `image_placeholder_ratio: 0.2424 → 0.0000`. Closed by
     targeted enrichment of the 264 ``vision_status="pending"``
     shadow full-page chunks via
     `scripts/enrich_firearms_pending_only.py` (delegates per-chunk
     work to the canonical
     `scripts/enrich_image_chunks_v29.py::_enrich_one` helper, so
     prompt / retry / hard-fallback semantics are unchanged). 264
     enriched, 0 hard_fallback.

   Firearms strict-gate verdict (post-audit-fix):
   `QA_PASS: failures=0 warnings=0`.

   Diagnostic doc:
   [`docs/PHASE_6_FIREARMS_OCR_HEADING_DIAGNOSTIC.md`](PHASE_6_FIREARMS_OCR_HEADING_DIAGNOSTIC.md).

**Phase 5 `validated-local` (2026-05-13; committed in `f3d8478` —
"feat(v2.10): Phase 5 — HYBRID_CHUNKER_HEADING_PROPAGATION (Devlin)
validated-local").**

3. `HYBRID_CHUNKER_HEADING_PROPAGATION` — Devlin. Producer-side fix
   in [`src/mmrag_v2/batch_processor.py`](../src/mmrag_v2/batch_processor.py)
   collapses HybridChunker heading propagation to ONE site
   (export-boundary `_propagate_headings(export_chunks)`). Validator
   centralized in
   [`src/mmrag_v2/state/context_state.py::is_valid_heading`](../src/mmrag_v2/state/context_state.py#L64)
   — tightened against repeated-token artefacts, code/JSON heading
   shapes, and bracket-prefixed code labels. Generic Docling buckets
   (`Start`, `Front Matter`) may remain on their owning chunk but are
   NOT permitted to seed forward carry-state via `_GENERIC_CARRY_HEADINGS`.
   Devlin's strict-gate HEADING audit: **`PASS coverage 783/790 (99%)`,
   null_headings=7** (legitimate front-matter pages before the first
   real section signal — Pages 3–7). 7 new tests in
   `tests/test_hybrid_chunker_heading_propagation.py`. The "only one
   propagation site" structural contract is pinned by
   `tests/test_vision_aided_front_matter.py` via source introspection.
   Smoke 11/11 GATE_PASS + 11/11 UNIVERSAL_PASS verified 2026-05-13.
   See `docs/PLAN_V2.10.md` §Phase 5.

**Phase 4 `validated-local` (2026-05-13; committed in `8effdfd` —
"feat(v2.10): land Phase 2/3/4 — TextIntegrityScout + B4B picture
dedup + cross-page-split"). All Phase 4 charter criteria are met:
Cookbook & Distilled page-loss closed, smoke 11/11 GATE_PASS, full
pytest green.**

4. `CROSS_PAGE_SPLIT_PAGE_ATTRIBUTION` — Python_Cookbook + Python_Distilled.
   Fix landed in [`src/mmrag_v2/processor.py`](../src/mmrag_v2/processor.py):
   per-page text reconstruction via `prov.charspan` slicing + bare
   `DocItem` reference dereferencing against `doc.texts` +
   ``_looks_like_subtitle_continuation`` helper that promotes
   short-title-continuation chunks to ``ChunkType.HEADING`` when
   the universal structural signature matches (short single-line
   text under a parent_heading, no terminal sentence punctuation,
   first word is a small English connector).
   Companion fixes in [`src/mmrag_v2/batch_processor.py`](../src/mmrag_v2/batch_processor.py):
   `_merge_micro_text_chunks` skips `hybrid_chunker_pagesplit_fallback`
   markers; `_deduplicate_chunk_overlap` is now page-scoped.
   Audit-side: ``chunk_type ∈ {heading, title}`` treated as non-paragraph
   structural content alongside ``code`` — exempt from the
   ``micro_non_label`` counter. Threshold values unchanged.
   **Status: all 4 plan-listed Cookbook pages closed** (63 / 128 / 365 / 397).
   Python_Distilled's cross-page-split MISSING_PAGES list is empty.
   See `docs/PLAN_V2.10.md` §Phase 4 for the full investigation log.

**Closed locally (`validated-local`; awaiting Phase 8 full-corpus re-verification):**

5. `B4B_FULL_DOC_PICTURE_DEDUP` — Earthship + Python_Distilled
   (3 image-only pages). **`validated-local` 2026-05-12.** Two-site fix:
   (a) pHash dedup page-coverage carve-out; (b) SHADOW-EXTRACTION
   page-coverage-aware threshold (200×200 floor for pages with no
   prior chunks). Both strict gates now report `QA_PASS` /
   `QA_PASS_WITH_ADVISORIES` after re-enrichment. See
   `docs/PLAN_V2.10.md` §Phase 3.
6. `TEXT_INTEGRITY_SCOUT_FULL_DOC_SENSITIVITY` — Fluent_Python.
   **`validated-local` 2026-05-12.** Per-batch trigger module +
   parallel-site fix on `_quarantine_corrupted_text_chunks`
   (ratio-based detector). Fluent strict gate reports
   `QA_PASS_WITH_ADVISORIES: failures=0 warnings=1`. See
   `docs/PLAN_V2.10.md` §Phase 2.
7. `TEXT_LABEL_TOC_DENSE_INDEX_ROUTER_MISS` — Chaubal p11.
   **`validated-local` 2026-05-12.** Dense-index classifier extended
   to recognize compact TOC tails with U+FFFD leader runs. Chaubal
   strict gate reports `QA_PASS` with `MISSING_PAGES=[]`. See
   `docs/PLAN_V2.10.md` §Phase 1.

v2.10 housekeeping (non-blocking; see snapshot §9):
- Devlin re-ingest into `mmrag_v2_8` so payload metadata matches the
  post-catch-up JSONL.
- Re-tune `scripts/search_qdrant.py` `--model` default and `MIN_SCORE`
  floor once v2.10 rebuilds the collection.
- Move the hard-coded Dashscope API key out of
  [`scripts/search_qdrant.py:40`](../scripts/search_qdrant.py#L40)
  into an env var and rotate the leaked key.

### Already-known followups (not v2.9 scope)

Carry forward to v2.10 planning:

- **Local NuMarkdown-8B-Thinking-mlx-8bits VLM lane.** Cloud-only
  for v2.9 enrichment; re-evaluate when network reachability returns.
- **Remote CodeFormulaV2 inference.** Docling 2.86 still doesn't
  expose `RemoteCodeFormulaOptions` / `ApiCodeFormulaOptions`.
- **HybridChunker per-item token guard.** Requires upstream Docling.
- **Broader UIR refactor** (canonical PdfConversionPlan →
  UniversalDocument → ElementProcessor → chunks per CLAUDE.md).
- **Magazine rendered-region-crop architecture** for composite layouts.

## Active Engineering Direction

Plan at [`docs/PLAN_V2.10.md`](PLAN_V2.10.md). Phases 1-6 are
`validated-local` (Phase 6 closed 2026-05-15 after the audit-fix
iteration). Firearms full strict gate returns
`QA_PASS: failures=0 warnings=0`. The next task is Phase 7
(`KI_EPUB_EXTRACTION_LANE_REWRITE`, KI EPUB).

PCWorld VLM evidence from the RC1 cycle remains valid: raw
text-reading detections 36.5 % → 22.2 %, zero measured Combat-style
hallucinations, blind-set 87.5 % final-valid. See
`tests/fixtures/blind_set_manifest.json`.

## Recently Completed

Reverse-chronological. Each entry's evidence files are tracked.

- **PLAN_V2.10 Phase 5 — `HYBRID_CHUNKER_HEADING_PROPAGATION`
  (Devlin):** `validated-local` (2026-05-13, `f3d8478`). Single
  propagation site at export boundary; heading validator tightened
  against garbage strings and code shapes. Devlin HEADING 99 %.
  7 new tests in `tests/test_hybrid_chunker_heading_propagation.py`.
  Smoke 11/11 GATE_PASS + UNIVERSAL_PASS. Full pytest 860 passed.
  See `docs/PLAN_V2.10.md` §Phase 5 and
  `docs/PHASE_5_DEVLIN_HEADING_DIAGNOSTIC.md`.

- **PLAN_V2.10 Phase 4 — `CROSS_PAGE_SPLIT_PAGE_ATTRIBUTION`
  (Cookbook + Distilled):** `validated-local` (2026-05-13, `8effdfd`).
  Per-page text split via `prov.charspan` + bare `DocItem`
  dereferencing + subtitle-continuation promotion + page-scoped
  overlap-trim + micro_non_label heading exemption. Cookbook 4/4
  missing pages closed; Distilled cross-page MISSING_PAGES=0.
  27 tests in `tests/test_cross_page_split_page_attribution.py`.
  Smoke 11/11 GATE_PASS. Full pytest 853 passed.

- **PLAN_V2.10 Phase 3 — `B4B_FULL_DOC_PICTURE_DEDUP`
  (Earthship + Distilled):** `validated-local` (2026-05-12). Two-site
  fix: pHash dedup page-coverage carve-out + SHADOW-EXTRACTION
  page-coverage-aware threshold (200×200 floor for pages with no
  prior chunks). Earthship `QA_PASS`; Distilled `QA_PASS_WITH_ADVISORIES`.
  9 tests in `tests/test_phash_dedup_page_coverage.py`.
  Full pytest 826 passed. See `docs/PLAN_V2.10.md` §Phase 3.

- **PLAN_V2.10 Phase 2 — `TEXT_INTEGRITY_SCOUT_FULL_DOC_SENSITIVITY`
  (Fluent_Python):** `validated-local` (2026-05-12). Per-batch
  trigger module + parallel-site quarantine fix (ratio-based
  detector). Fluent `QA_PASS_WITH_ADVISORIES`. 10 tests in
  `tests/test_text_integrity_scout_per_batch_trigger.py`.
  Full pytest 817 passed. See `docs/PLAN_V2.10.md` §Phase 2.

- **PLAN_V2.10 Phase 1 — `TEXT_LABEL_TOC_DENSE_INDEX_ROUTER_MISS`
  (Chaubal p11):** `validated-local` (2026-05-12). Compact U+FFFD
  TOC-tail router. Chaubal `QA_PASS`, `MISSING_PAGES=[]`.
  See `docs/PLAN_V2.10.md` §Phase 1.

- **PLAN_V2.9 Phase 1 (TOC/index page-loss closure):** `complete`
  (2026-05-07, commit `df91061`). Dense-index page router via Docling
  `document_index` label fast path. Test suite 628 passed.

- **PLAN_V2.8 (production gaps + broad reconversion + Qdrant
  re-ingestion):** SHIPPED 2026-05-04. 7 commits on main
  `5b0e13d → 645ab2b`. See `docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md`.

## Must-Respect Constraints

- Python 3.10 only.
- Batch size must stay at or below 10 pages.
- Do not use `--profile-override` for acceptance runs.
- Do not add filename-specific or document-specific quality rules.
- OCR handles text; VLMs describe visuals only.
- BBoxes must remain normalized integer `[0,1000]`.
- Acceptance requires `GATE_PASS` plus `UNIVERSAL_PASS` across the smoke matrix.
