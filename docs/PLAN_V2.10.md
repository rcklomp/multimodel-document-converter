# Plan: v2.10 — Close the v2.9.0-rc1 Signed Deferrals and Ship the Next Production Baseline

**Status:** Phases 1-7 `validated-local` (2026-05-15); Phase 8
`implemented` (2026-05-16, pending the AFTER snapshot commit that
moves it to `validated-local`).

Phase 8 closes the v2.10 release-prep workstream: corpus-wide
strict-gate re-verification on the 34-doc canonical corpus
(**34 PASS / 0 WARN / 0 FAIL** target), targeted re-enrichment of
the 443 image chunks that the Phase 4 / Phase 5 reconverts left
`vision_status="pending"` (Devlin 68, Python_Cookbook 26,
Python_Distilled 349), canonical-CLI reconvert + VLM enrichment of
Firearms (the pre-Phase-6 canonical JSONL was stale), Qdrant
`mmrag_v2_8` rebuild, the v2.10 release tag stage, and closure of
the three v2.9.0-rc1 housekeeping carry-forwards (`search_qdrant.py`
defaults, Dashscope API-key env-var migration, Devlin payload
refresh).

Phase 1-7 evidence stays valid:
Firearms strict gate `QA_PASS: failures=0 warnings=0` (HEADING
coverage 1091/1094 = 0.997, TEXT `infix_artifacts` 148 → 0); KI EPUB
strict gate `QA_PASS_WITH_ADVISORIES: failures=0 warnings=1`
(`MISSING_CHAPTERS` advisory for two contiguous leading low-content
structural spine items); ChatGPT EPUB regression control
`QA_PASS_WITH_ADVISORIES`. Full pytest is
**966 passed, 14 skipped, 0 failed** entering Phase 8. Smoke
multiprofile remains **11/11 GATE_PASS + 11/11 UNIVERSAL_PASS** with
the Apple-Silicon `TORCH_COMPILE_DISABLE=1` runtime guard. No QA
threshold weakened. No new advisory codes added.
**Owner:** ingestion pipeline
**Successor to:** `docs/PLAN_V2.9.md` (v2.9.0-rc1 shipped 2026-05-12, tag `v2.9.0-rc1` on commit `3e06d1b`)
**Related:** `docs/PROJECT_STATUS.md`, `docs/QUALITY_GATES.md`, `docs/DECISIONS.md`,
`docs/ARCHITECTURE.md`, `docs/AGENT_GOVERNANCE.md`, `AGENTS.md`, `CHANGELOG.md`

**BEFORE state:** `docs/QUALITY_SNAPSHOT_2026-05-11_v2.9.0-rc1_after.md`
(26 PASS / 0 WARN / 8 FAIL across the 34-doc canonical corpus; all
8 FAILs are signed v2.10 production-tag blockers).

**CURRENT working state (2026-05-16):** Phases 1-8 are
`validated-local`. Phase 8 closed after the Qdrant `mmrag_v2_8`
rebuild completed at 18:17:26 UTC: `status: green`,
`points_count: 30,454` (raw chunk count was 30,588; `ingest_to_qdrant.py`
filters ~0.44 %), `indexed_vectors_count: 30,192`. v2.9.0-rc1 →
v2.10.0-rc1 delta: −7 points. The v2.10.0-rc1 release tag is staged
but not pushed; the user controls the final `git tag` push.

- Phase 1 (`TEXT_LABEL_TOC_DENSE_INDEX_ROUTER_MISS`, Chaubal p11) —
  `output/Chaubal_PyTorch_Projects/ingestion.jsonl` was regenerated,
  re-enriched, and passed the canonical single-doc strict gate with
  `MISSING_PAGES=[]` and `QA_PASS`.
- Phase 2 (`TEXT_INTEGRITY_SCOUT_FULL_DOC_SENSITIVITY`, Fluent_Python) —
  per-batch trigger + quarantine parallel-site fix shipped; Fluent
  reconverted + re-enriched; strict gate reports
  `QA_PASS_WITH_ADVISORIES: failures=0 warnings=1`.
- Phase 3 (`B4B_FULL_DOC_PICTURE_DEDUP`, Earthship + Python_Distilled)
  — pHash dedup page-coverage carve-out + SHADOW-EXTRACTION
  page-coverage-aware threshold shipped. Both strict gates report
  `QA_PASS` / `QA_PASS_WITH_ADVISORIES`.
- Phase 4 (`CROSS_PAGE_SPLIT_PAGE_ATTRIBUTION`, Python_Cookbook +
  Python_Distilled) — per-page text split via `prov.charspan` slicing +
  bare DocItem dereferencing + `_looks_like_subtitle_continuation`
  promotion + page-scoped overlap-trim + audit micro_non_label heading
  exemption. Cookbook 4/4 missing pages closed; Distilled cross-page
  MISSING_PAGES=0. Smoke 11/11 GATE_PASS + UNIVERSAL_PASS. Full pytest
  853 passed. Committed in `8effdfd`.
- Phase 5 (`HYBRID_CHUNKER_HEADING_PROPAGATION`, Devlin) — single
  propagation site at export boundary; heading validator tightened
  against garbage/code shapes. Devlin HEADING 99%. Full pytest 860
  passed. Committed in `f3d8478`.
- Phase 6 (`OCR_PATH_HEADING_PROPAGATION`, Firearms) — ordered OCR-lane
  heading attribution plus audit-fix TEXT infix repair and targeted
  pending-image enrichment. Firearms strict gate `QA_PASS: failures=0
  warnings=0`; Earthship scanned-class heading regression unchanged.
- Phase 7 (`KI_EPUB_EXTRACTION_LANE_REWRITE`, KI EPUB) — EPUB spine
  markers, synthetic pagination/bbox, per-synthetic-page dedup, and
  EPUB-aware `MISSING_CHAPTERS` advisory. KI and ChatGPT EPUB strict
  gates both `QA_PASS_WITH_ADVISORIES`.

Full local test suite now reads **966 passed / 14 skipped / 0
failed** (plus the Phase 8 regression pins). Smoke multiprofile
remains **11/11 GATE_PASS + 11/11 UNIVERSAL_PASS**. The corpus-level
AFTER snapshot is authored at
`docs/QUALITY_SNAPSHOT_2026-05-16_v2.10_after.md`; the v2.10
release-tag command is staged but not executed.

**AFTER state:**
[`docs/QUALITY_SNAPSHOT_2026-05-16_v2.10_after.md`](QUALITY_SNAPSHOT_2026-05-16_v2.10_after.md)
(v2.10.0-rc1 AFTER — 34/34 PASS, 0 WARN, 0 FAIL).

---

## 1. Why this plan exists

### Thesis

**Close the 8 signed v2.9.0-rc1 deferrals under the unchanged
strict gate, ship the next v2.10 production baseline with 34/34
PASS-class coverage on `scripts/qa_full_conversion.py --source-pdf
--allow-warnings`, and establish the evidence base for any future
SRS/UIR refactor work.** No gate weakening, no filename-specific
logic, no blank backfill — each deferral closes via the named
root-cause class identified in
`docs/DECISIONS.md` "v2.9.0-rc1 Signed Deferrals (2026-05-11
close-out)".

### Where the previous cycle left off

`v2.9.0-rc1` shipped on commit `3e06d1b` with:

- Strict gate (`scripts/qa_full_conversion.py --source-pdf --allow-warnings`):
  **26 PASS / 0 WARN / 8 FAIL** across the 34 canonical docs
  (12 `QA_PASS` + 14 `QA_PASS_WITH_ADVISORIES`).
- Test suite: **806 passed, 14 skipped, 0 failed.**
- Qdrant `mmrag_v2_8`: **30,461 points**, status green, 4096-dim
  llava, rebuilt 2026-05-12 from the 34 post-recovery JSONLs.
- All 8 FAILs covered by named, signed v2.10 deferrals.

There is no intermediate final `v2.9.0` tag planned; `v2.9.0-rc1`
is the v2.9 ship state, and the 8 deferrals carry forward as **v2.10
production-tag blockers** under the unchanged strict gate
(`docs/DECISIONS.md` "Signed deferral list").

### The 8 signed deferrals

| # | Doc(s) | Class | Affected pages | Retrieval impact | v2.10 status | Root cause / current evidence |
|---|---|---|---:|---|---|---|
| 1 | Firearms | `OCR_PATH_HEADING_PROPAGATION` | ~300 (HEADING 72 % → 99.7 %) | Moderate | `validated-local` (2026-05-15). Full strict gate `QA_PASS: failures=0 warnings=0`. Audit-fix iteration also closed the pre-existing TEXT `infix_artifacts: 148 → 0` via the new chunk-content repair `_repair_infix_step_numbers` (universal heuristic that **behaviorally mirrors** the audit detector after its newline / stop-word post-filters; parity pinned by re-applying the audit detector to repaired content), and the VLM `image_placeholder_ratio: 0.2424 → 0.0000` via targeted enrichment of the 264 ``vision_status="pending"`` shadow chunks (`scripts/enrich_firearms_pending_only.py` delegating to the canonical `enrich_image_chunks_v29.py::_enrich_one` helper). | OCR/element-by-element path did not promote Docling `section_header` items into `ContextStateV2.hierarchy_stack`. Phase 6 fix: `Region.is_heading` / `ProcessedChunk.is_heading` carry Docling's structural label through `LayoutAwareOCRProcessor`; `BatchProcessor._attribute_ocr_chunk_heading` does ordered per-chunk push+read; `_promote_ocr_section_headers` is the fallback for VLM/Tesseract-fullpage paths. New `ContextStateV2.get_section_heading()` skips the level-0 doc-title breadcrumb. Validator tightenings (terminal-period sentence-shape on `.` only — `?` and `!` accepted; numbered-prefix body-case shape) centralised in `is_valid_heading`. Single-page push gate (`self._doc_total_pages > 1`) on BOTH call sites prevents form-shape regressions on the canonical 0013 invoice. HEADING coverage 1091/1094 (3 NULL = page-1 front-matter, correctly unattributed by ordered design), top-5 all real chapter/section titles; smoke 11/11 GATE_PASS + 11/11 UNIVERSAL_PASS; Earthship_Vol1 full-doc HEADING coverage unchanged at 1.00 (549/549). See [docs/PHASE_6_FIREARMS_OCR_HEADING_DIAGNOSTIC.md](PHASE_6_FIREARMS_OCR_HEADING_DIAGNOSTIC.md). |
| 2 | KI_En_ChatGPT_Praktische_Gids | `KI_EPUB_EXTRACTION_LANE_REWRITE` | full doc | Moderate | `validated-local` (Phase 7, 2026-05-15). KI EPUB strict `QA_PASS_WITH_ADVISORIES: failures=0 warnings=1` (advisory = `MISSING_CHAPTERS` for the two contiguous leading low-content structural spine items — titlepage + colophon — Docling's HTML parser strips before any chunk emerges). ChatGPT EPUB regression control also `QA_PASS_WITH_ADVISORIES: failures=0 warnings=1`. Smoke 11/11 GATE_PASS + 11/11 UNIVERSAL_PASS. | Fix lives in the existing EPUB lane: `_epub_to_html` walks `book.spine` and prepends a `<p>__MMRAG_EPUB_CH_NNNN__</p>` marker to each non-empty chapter; the post-Docling `_apply_epub_synthetic_pagination` pass scans chunks in emission order, rewrites `page_number = chapter_1based * 1000 + position_in_chapter // 5`, sets the EPUB sentinel `bbox=[0,0,1000,1000]` + `extraction_method="epub_html"`, regenerates `chunk_id` with the new page (preserving the v2.9 position-component uniqueness contract), and dedups byte-equal content within a 5-chunk synthetic page. Pre-marker buffer back-attributes chunks emitted before any marker survives to chapter `first_marker - 1` (handles KI's titlepage+colophon stripping by Docling). `qa_full_conversion.py` adds an EPUB-aware branch that enumerates spine chapters via `ebooklib` and replaces PDF page-coverage with `MISSING_CHAPTERS`; it is WARN/advisory only when every missing item is a contiguous leading/trailing low-content structural spine item. Internal or content-bearing missing chapters remain FAIL. |
| 3 | Devlin_LLM_Agents | `HYBRID_CHUNKER_HEADING_PROPAGATION` | ~250 (HEADING 72 %) | Moderate | `validated-local` (Phase 5, 2026-05-13, `f3d8478`) | Cross-batch heading carry-forward (`b429cb5`) verified in unit tests + small probe but does NOT move Devlin's metric. Root cause: Devlin's batches end mid-section without an end-of-section heading chunk. Phase 5 fix: single propagation site at export boundary + heading validator tightened against garbage/code shapes. Devlin HEADING 99%. |
| 4 | Python_Cookbook | `CROSS_PAGE_SPLIT_PAGE_ATTRIBUTION` | 4 pages | None — content present, wrong `page_number` | `validated-local` (Phase 4, 2026-05-13, `8effdfd`) | Cross-page split at [src/mmrag_v2/processor.py:2870-2929](../src/mmrag_v2/processor.py#L2870-L2929) emits same merged `text.strip()` to every contributing page. Phase 4 fix: per-page text split via `prov.charspan` slicing + bare DocItem dereferencing. All 4 Cookbook pages closed. |
| 5 | Python_Distilled | `CROSS_PAGE_SPLIT_PAGE_ATTRIBUTION` (3p) + `B4B_FULL_DOC_PICTURE_DEDUP` (3p) | 7 / 1411 | Mixed | `validated-local` (Phase 3 + Phase 4, 2026-05-13) | Both portions closed. Phase 3: image-only pages extracted under relaxed threshold. Phase 4: cross-page-split MISSING_PAGES=0. |
| 6 | Fluent_Python | `TEXT_INTEGRITY_SCOUT_FULL_DOC_SENSITIVITY` | 6 / 770 | None at retrieval | `validated-local` | Closed locally 2026-05-12 by adding a per-batch shortfall trigger (`src/mmrag_v2/validators/text_integrity_scout_trigger.py` + `BatchProcessor._per_batch_shortfall_fires`) that ORs into the scout's doc-level variance gate. Validation surfaced a parallel-site audit gap in `_quarantine_corrupted_text_chunks`; detector swapped to the existing ratio-based `is_irreparably_corrupt` so legitimate Python REPL byte-literal content (Fluent's encodings chapter) is preserved while Combat p66 / Cronin contracts stay green. Fresh Fluent reconvert + re-enrich reports `QA_PASS_WITH_ADVISORIES` (failures=0). |
| 7 | Chaubal_PyTorch_Projects | `TEXT_LABEL_TOC_DENSE_INDEX_ROUTER_MISS` | 1 / 800 | None | `validated-local` | Closed locally 2026-05-12 by extending the dense-index classifier to recognize compact TOC tails with U+FFFD leader runs from Docling `text` / `section_header` items. Fresh Chaubal reconvert + re-enrichment reports `QA_PASS`, `MISSING_PAGES=[]`; page 11 emits one `hybrid_chunker_pageskip` chunk. |
| 8 | Earthship_Vol1 | `B4B_FULL_DOC_PICTURE_DEDUP` | 1 / 287 | Marginal | `validated-local` | Closed locally 2026-05-12 with row 5/8 (Phase 3). The drop site was the cross-page pHash dedup at [batch_processor.py:3364](../src/mmrag_v2/batch_processor.py#L3364), not `_filter_blank_assets`. Earthship's hand-drawn cross-sections share pHash signatures within Hamming ≤ 10. The page-coverage carve-out preserves a near-duplicate only when no IMAGE has yet been exported for the image-only page; once any image (unique or preserved-duplicate) is written for that page, subsequent near-duplicates still drop. |

### What closing all phases achieves

Closing these 8 deferrals, in the order set in §3, produces a
34/34 PASS-class corpus under the unchanged strict gate. That is
the v2.10 production-tag bar. The work also yields:

- Three real bug fixes that benefit other corpora (Phase 1 router
  extension, Phase 4 cross-page-split repair, Phase 6 OCR-lane
  heading propagation) — each tightens a contract that today only
  appears in unit tests.
- A measured baseline for any future SRS/UIR refactor: every fix
  in this plan goes through the canonical
  `PdfConversionPlan → DoclingPdfAdapter → UniversalDocument →
  ElementProcessor → chunks` pipeline per CLAUDE.md, so the
  remaining direct-Docling-item-to-chunk paths in
  `processor.py` shrink rather than grow.

---

## 2. Goals & Non-Goals

### Goals (measurable from JSONL / strict-gate output / Qdrant counts)

1. **`scripts/qa_full_conversion.py --source-pdf --allow-warnings
   output/<doc>/ingestion.jsonl` reports `QA_PASS` (pure or
   `QA_PASS_WITH_ADVISORIES`) for all 34 canonical docs.** No
   `QA_FAIL` and no `QA_WARN` rows. Specifically each of the 8
   deferral rows in §1 flips from FAIL to a PASS-class status.
2. **AGENTS.md Completion Rules (`docs/AGENT_GOVERNANCE.md`)
   satisfied for v2.10 tag.** Every acceptance signal in §4 met;
   evidence is `tracked` or `snapshot`; known limitations
   documented; a fresh session can reproduce the claim without
   chat history.
3. **`pytest tests/ -q` reports the Phase 7 local baseline
   (`966 passed, 14 skipped, 0 failed`) or higher** with every new
   red→green test from Phases 1-7 added and green.
4. **`bash scripts/smoke_multiprofile.sh`: every row `GATE_PASS` +
   `UNIVERSAL_PASS`** (form variants per `docs/QUALITY_GATES.md`
   "Form / Invoice Acceptance Class" remain valid).
5. **Within-file `chunk_id` duplicates across the 34 canonical
   JSONLs remain 0** after every reconvert in this plan. The v2.9
   Phase 1 contract from `eae27e8` is non-negotiable.
6. **0 image points with `vision_status="pending"`** in any JSONL
   after each affected-doc re-enrichment and Phase 8 verification; `hard_fallback` rate
   <= 5 % per-doc except where the F4 sentinel
   (`complex_asset_short_response_after_retry`) exempts.
7. **`mmrag_v2_8` Qdrant collection is rebuilt** from the
   post-v2.10 JSONLs once Goals 1-6 are met, with payloads matching
   the reconverted JSONLs (closing the Devlin Phase H catch-up
   payload-staleness item from
   `docs/QUALITY_SNAPSHOT_2026-05-11_v2.9.0-rc1_after.md` §7).
8. **AGENTS.md, `docs/PROJECT_STATUS.md`, `docs/TESTING.md`,
   `CHANGELOG.md`, and `docs/README.md` all reference the v2.10
   ship state** without stale post-RC1 language ("`v2.9.0` final
   tag planned", "Phase 5c blocked", etc.). Phase 0 verifies this
   before any code change.

### Non-Goals (deferred beyond v2.10)

- **Local NuMarkdown-8B-Thinking-mlx-8bits VLM lane.**
  Off-network endpoint still unreachable as of 2026-05-12.
  Cloud-only v2.10 enrichment continues. Re-evaluate when network
  reachability returns.
- **Remote CodeFormulaV2 inference target
  (`RemoteCodeFormulaOptions` / `ApiCodeFormulaOptions`).** Not
  exposed by Docling 2.86. Trigger condition (>1 code-heavy
  reconversion / week, per `docs/DECISIONS.md` "Selective Code
  Enrichment Lane → Amendment 2026-05-03") is not met within the
  v2.10 cycle; revisit if v2.10's Phase 4 reconverts of code-heavy
  docs (Cookbook, Distilled, Fluent) require repeating beyond a
  one-off batch.
- **Broader UIR refactor.** Canonical target per `CLAUDE.md` /
  `docs/DECISIONS.md` "Shared PDF Extraction Plan". The Phase 1,
  Phase 4, and Phase 6 fixes each route through the existing
  adapter without expanding direct-Docling-item-to-chunk paths,
  but a full UIR rewrite of `processor.py` is NOT v2.10 scope.
- **HybridChunker per-item token guard.** Requires upstream
  Docling support; out of repo scope.
- **Magazine rendered-region-crop architecture** for composite
  layouts (`docs/CONVERSION_PROFILES.md`).
- **EPUB lane redesign beyond Phase 7.** Phase 7 closes
  `KI_EPUB_EXTRACTION_LANE_REWRITE` for the named doc. A broader
  EPUB engine extraction redesign (e.g. an `EpubEngine` parallel
  to `PDFEngine` in `engines/`) stays out of v2.10 scope; the
  Phase 7 fix lands in the existing
  `processor._epub_to_html` path.

## 2b. Parallel-Site Audit (cross-cutting principle — unchanged)

Permanent project requirement since v2.8 §2b, re-asserted by every
phase below. The 2026-05-03 lesson stands: a single-site fix is
suspect until parallel call sites are audited. The 2026-05-11
B1 postscript in `docs/PHASE_A_MISSING_PAGES_DIAGNOSTIC.md` §8
expanded the audit perimeter to include the strict-gate scripts
themselves (`scripts/qa_full_conversion.py`, `scripts/qa_conversion_audit.py`,
`scripts/qa_universal_invariants.py`, `scripts/qa_semantic_fidelity.py`).

Every phase below answers these four questions BEFORE writing any
code:

1. Does the issue ALREADY have a fix elsewhere in the pipeline
   that the failing data simply hasn't been re-run through?
2. Does an existing fix have too narrow a gate (e.g., a label
   match on one Docling label when the bug surface uses another)?
3. Are there parallel boundaries that need the same change?
   - CLI `process` vs CLI `batch`
   - `BatchProcessor` vs `V2DocumentProcessor` vs the Mapper
   - `engines/pdf_engine.py` vs `engines/docling_adapter.py`
   - HybridChunker path vs OCR/element-by-element path
   - Producer site vs the four strict-gate validator scripts
4. Is there an upstream library config (Docling 2.86 option,
   ebooklib option, PyMuPDF API) that already addresses the issue
   without custom code? "Libraries first, custom code last."

A phase MUST NOT design a fix until its parallel-site audit
section is populated with concrete answers (not "N/A" or
"presumed none").

## 2c. Architectural constraints (carried from v2.9 §2d)

These constraints bind the implementation of the phases below.

1. **Page-coverage fixes land in the extraction policy layer.**
   If a deferral's root cause traces to chunks being attributed
   to the wrong source page (Phase 4, Phase 5 partial,
   Phase 7 partial), the fix lives in `processor.py` /
   `PdfConversionPlan` / `DoclingPdfAdapter` where the
   page-index mapping originates. The fix MUST NOT be a post-hoc
   patch in the JSONL writer or in `BatchProcessor`'s emission
   loop, and MUST NOT emit `extraction_method="recovery_page_coverage"`
   (banned by v2.9 Phase 1).
2. **No filename-specific or doc-specific logic.** Per
   AGENT-VAL-01. Every threshold or rule added must be a universal
   geometric or content-shape rule with at least one positive and
   one negative corpus-validated regression test (see the 2026-05-09
   Path A overfit `5e58e6e` → `cbd7fb4` precedent in
   `docs/DECISIONS.md` "No gate weakening to make a failing run
   pass").
3. **Test contracts are executable requirements.** Per
   AGENT-TEST-01 / CLAUDE.md "Test Contract Integrity": no
   negative or regression assertion may be weakened, rewritten,
   or reframed to make current code pass. If a fixture appears to
   require change, stop and document the proposed requirement
   change first.
4. **No gate weakening.** Per `docs/DECISIONS.md` "No gate
   weakening to make a failing run pass". Each deferral closes
   by fixing the underlying defect OR by extending an
   already-documented advisory class (`docs/QUALITY_GATES.md`
   "Advisory Warning Classes") via the formal "add a new
   advisory code" procedure in that document. Profile-scoped
   threshold tuning to make a single doc pass is forbidden.

## 2d. Cost-aware ordering

Re-enrichment is paid in cloud-VLM tokens (`qwen3-vl-plus` on
DashScope). The v2.9.0-rc1 RC1 cycle consumed ~4,500 calls.
v2.10's re-enrichment budget depends on which fixes require
reconvert (chunking/serialization changes invalidate image
chunk_ids — re-enrich needed) versus post-processing only
(existing enriched chunks survive). Each phase below names which
class it falls into.

`scripts/enrich_image_chunks_v29.py` is idempotent on
`vision_status="complete"` chunks (verified empirically during
the Phase 5b race incident). Running it across the full 34 doc
list after a partial reconvert costs only the new chunks.

---

## 3. Phases

The 8 signed deferrals decompose into 7 unique root-cause classes
(Python_Distilled covers both #4 and #8's classes). Phases 1-7
each close one class; Phase 0 sets the documentation baseline,
Phase 8 re-verifies the strict gate corpus-wide and tags the
release. Ordering is smallest-cost / lowest-blast-radius first.
As of this draft revision, Phases 1–7 are `validated-local`.
Phase 6 closed 2026-05-15 after the audit-fix iteration: Firearms
full strict gate returns `QA_PASS: failures=0 warnings=0`; the TEXT
`infix_artifacts` defect (pre-existing real OCR/multi-column-layout
artifact, 148 hits in BEFORE) was closed by the new
`BatchProcessor._repair_infix_step_numbers` chunk-content repair
whose detection **behaviorally mirrors the audit detector after its
newline / stop-word post-filters**; parity is pinned by the
audit-detector parity test in
`tests/test_infix_step_number_repair.py`; and the VLM
image-placeholder defect was closed by targeted enrichment of the 264
pending shadow chunks via cloud `qwen3-vl-plus`. Phase 7
(`KI_EPUB_EXTRACTION_LANE_REWRITE`) closed the EPUB lane with
spine-order chapter markers, synthetic pagination/bbox, scoped dedup,
and the documented conditional `MISSING_CHAPTERS` advisory. Phase 8
closes the v2.10 release-prep workstream (corpus-wide strict-gate
re-verification at **34 PASS / 0 WARN / 0 FAIL**, Qdrant `mmrag_v2_8`
rebuild, AFTER snapshot, and the staged v2.10.0-rc1 release tag).

| Phase | Scope | Docs affected | Reconvert? | Re-enrich? | Status |
|---|---|---|---|---|---|
| 0 | Documentation Hygiene (control-doc verification) | — | no | no | `pending` (absorbed by v2.10 doc updates) |
| 1 | `TEXT_LABEL_TOC_DENSE_INDEX_ROUTER_MISS` (router extension for `text` / `section_header`-labeled compact TOC with U+FFFD leaders) | Chaubal_PyTorch_Projects p11 | Chaubal (1 doc, done locally) | done locally (55 image chunks; 80 VLM calls incl. retries) | `validated-local` |
| 2 | `TEXT_INTEGRITY_SCOUT_FULL_DOC_SENSITIVITY` (per-batch trigger threshold) + parallel-site fix on corruption quarantine | Fluent_Python | Fluent (1 doc, done locally) | done locally (83 image chunks; 78 complete + 5 hard_fallback retries) | `validated-local` |
| 3 | `B4B_FULL_DOC_PICTURE_DEDUP` (full-doc image-only-page chunk drop) + SHADOW-EXTRACTION page-coverage threshold | Earthship_Vol1, Python_Distilled | Earthship + Python_Distilled (2 docs, done locally) | done locally (468 + 349 image chunks) | `validated-local` |
| 4 | `CROSS_PAGE_SPLIT_PAGE_ATTRIBUTION` (per-page text split with correct `page_number`) | Python_Cookbook, Python_Distilled (3 pages overlap with Phase 3) | Cookbook + Distilled (2 docs; Distilled overlaps Phase 3) | yes (image chunks on both) | `validated-local` (2026-05-13, `8effdfd`) |
| 5 | `HYBRID_CHUNKER_HEADING_PROPAGATION` (Devlin-shape batch-boundary investigation + fix) | Devlin_LLM_Agents | Devlin (1 doc) | yes | `validated-local` (2026-05-13, `f3d8478`) |
| 6 | `OCR_PATH_HEADING_PROPAGATION` (ordered per-chunk attribution via `Region.is_heading` / `ProcessedChunk.is_heading`; central `is_valid_heading` tightened for terminal-period sentence-shape `.`-only + numbered-prefix body-case; single-page push gate on BOTH per-chunk and fallback paths; audit-fix `BatchProcessor._repair_infix_step_numbers` repairs OCR-mashed numbered-step boundaries and behaviorally mirrors the audit detector after its newline / stop-word post-filters) | Firearms (primary) + Earthship_Vol1 (regression) | Firearms reconverted; Earthship_Vol1 reconverted for regression confirmation | targeted enrichment of 264 ``vision_status="pending"`` shadow chunks via cloud `qwen3-vl-plus` | `validated-local` (2026-05-15). Firearms `QA_PASS: failures=0 warnings=0`. |
| 7 | `KI_EPUB_EXTRACTION_LANE_REWRITE` (pagination + bbox + dedup in EPUB lane) | KI_En_ChatGPT_Praktische_Gids; ChatGPT_Praktijk_handboek (regression control) | KI EPUB (1 doc); ChatGPT EPUB validated as control | yes (KI image chunks) | `validated-local` (2026-05-15). KI strict `QA_PASS_WITH_ADVISORIES`. ChatGPT regression `QA_PASS_WITH_ADVISORIES`. Smoke 11/11 GATE_PASS + 11/11 UNIVERSAL_PASS. |
| 8 | Strict-Gate Re-Verification + v2.10 Release Prep (Qdrant rebuild, AFTER snapshot, tag) | all 34 | no | no | `pending` |

---

### Phase 0 — Documentation Hygiene

**What:** Verify that every control document reflects the
post-RC1 sanitization reality (Option 1 release model:
`v2.9.0-rc1` IS the v2.9 ship state; no intermediate `v2.9.0`
final tag is planned; the 8 deferrals carry forward as v2.10
production-tag blockers). Any stale reference is fixed here
before the next implementation phase begins.

The recent commits `91ca4e1` ("complete the post-RC1 sanitization
— Option 1 terminology + headline fix") and `e60f70f` (version
alignment) did most of this sweep, but the hygiene checks below
verify it holds across every control document the next session
will read.

**Parallel-site audit (do this FIRST):**

| Control document | What to check | Risk if stale |
|---|---|---|
| `AGENTS.md` §5 ("CURRENT STATE & DIRECTIVES") | Engine version says `v2.9.0-rc1`; v2.10 planning scope explicit; 8 deferrals listed as v2.10 production-tag blockers; no language that implies a future final `v2.9.0` tag. | A new session enters v2.10 work thinking v2.9 still has open execution phases. |
| `docs/PROJECT_STATUS.md` headline (lines 1-40) | Banner says `v2.9.0-rc1` shipped; explicit "no intermediate final `v2.9.0` tag planned"; the 8 deferrals framed as v2.10 production-tag blockers. | Banner contradicts §"Open work — v2.10 production-tag blockers" — new sessions disagree with themselves. |
| `docs/README.md` "Read Order For New Sessions" | Step 2 points at `docs/PLAN_V2.10.md`; `docs/PLAN_V2.10_DRAFT_PROMPT.md` is historical only. | New sessions land on the active v2.10 plan instead of obsolete draft-planning guidance. |
| `CLAUDE.md` "Read First" list | Same as README — v2.10 plan / draft prompt is the active execution doc; PLAN_V2.9.md framed as history. | Same risk; CLAUDE.md is read on every Claude Code session start. |
| `docs/TESTING.md` version header + strict-gate command examples | Header reads `v2.9.0-rc1` (or `v2.10.x` once Phase 8 lands); the Single-Conversion Full QA block uses `--source-pdf --allow-warnings` per `docs/QUALITY_GATES.md` "Canonical Single-Doc Strict Gate". | Test instructions disagree with the canonical gate, producing phantom MISSING_PAGES failures. |
| `CHANGELOG.md` `[v2.9.0-rc1]` | Section is the v2.9 close-out; does not promise a final `v2.9.0` tag; carry-forward to v2.10 list is the 8 named deferrals. | Changelog implies a tag that doesn't exist. |
| `docs/archive/PROGRESS_CHECKLIST.md` | Explicitly archival; the file's banner names archive date 2026-05-07. | If anyone re-activates it, current task state forks between two files. |
| `docs/PLAN_V2.9.md` header | Marked as "v2.9 execution history through the rc1 scope cut" / "RETAINED for execution history". | Confuses the next session into editing v2.9 execution to drive v2.10 work. |

**Approach:**

1. Run the parallel-site audit above as read-only diff: for each
   document, capture the relevant header / lines and confirm
   compliance.
2. Patch only the documents that have residual stale references.
   Each patch is a surgical edit (no large rewrites); commit per
   document or as one consolidated "docs: complete post-RC1 hygiene
   sweep" commit.
3. Run `pytest tests/ -q` after the hygiene sweep to confirm no
   doc-string-walking test (e.g., the test_contextual_retrieval
   drift guard at `tests/test_contextual_retrieval.py`) is
   broken by the edits.

**Tests (red→green):** No new test required. The existing
`tests/test_contextual_retrieval.py::test_no_contextual_marker_strings_in_production_code`
walks `src/mmrag_v2/` and `scripts/` and would already fail on
any improper docstring marker drift. Hygiene patches are
documentation-only; no behavior change.

**Done when:**

- Each row of the audit table reads "compliant" with no
  hand-waving.
- `pytest tests/ -q` shows the v2.9.0-rc1 baseline count (806
  passed, 14 skipped, 0 failed) — no regressions from doc edits.
- Phase 0 commit(s) merged on `main`.

**Risk:** Very low. Documentation-only changes.

**Estimated effort:** 1-2 h.

**Cost class:** No reconvert, no re-enrich.

---

### Phase 1 — `TEXT_LABEL_TOC_DENSE_INDEX_ROUTER_MISS` (Chaubal p11)

**Status:** `validated-local` (2026-05-12). Code site:
[src/mmrag_v2/processor.py:97](../src/mmrag_v2/processor.py#L97)
and [src/mmrag_v2/processor.py:251](../src/mmrag_v2/processor.py#L251).

**What:** Chaubal_PyTorch_Projects p11 is a TOC page with
dotted-leader entries. Docling labels most items as `text` and the
last item as `section_header`, not `document_index`, so the Phase 1 v2.9 dense-index router
([src/mmrag_v2/processor.py:200 `_classify_dense_index_pages`](../src/mmrag_v2/processor.py#L200))
does not route the page around HybridChunker. HybridChunker then
either fragments the TOC content into many short chunks (lost in
hygiene filters) or merges it into the surrounding pages, leaving
p11 as MISSING_PAGES under the strict gate.

Actual probe result (2026-05-12): p11 is a compact TOC tail with
five Docling items and U+FFFD leader runs, e.g. `Summary���� 335`
and `Index���� 337`. The pre-existing router's `text_items < 8`
floor and `_TOC_LEADER_RE = r"\.{2,}\s*\d{1,4}$"` missed it.

The local fix extends the router to recognize U+FFFD leader runs
and accepts compact pages where at least 4 items, and at least
80 % of text items, are TOC-leader entries. This remains a
content-shape rule; it does not use filename, document id, or page
number.

**Parallel-site audit (do this FIRST):**

1. **Existing fix elsewhere?** The Phase 1 router
   ([processor.py:200](../src/mmrag_v2/processor.py#L200)) and
   the secondary `_classify_dense_index_pages` fallback walker
   ([processor.py:355](../src/mmrag_v2/processor.py#L355)) are
   the only two routing decisions. The B1 universal U+FFFD
   sanitizer (`_collapse_replacement_chars` in
   `mmrag_v2.schema.ingestion_schema`) is content cleanup, not
   routing.
2. **Too-narrow gate?** Yes — the router matched Docling
   `document_index` labels and ASCII dot leaders, then required at
   least 8 text items for unlabeled pages. Chaubal p11 is a
   5-item TOC tail with U+FFFD leaders.
3. **Parallel boundaries?** The router is consumed by
   `MmragChunkingSerializerProvider` (engines/docling_serializers.py)
   via `skip_pages=`. There's only one entry point from the
   processor. CLI `process` and CLI `batch` both flow through
   `processor._classify_dense_index_pages` once converted.
   `V2DocumentProcessor` and `BatchProcessor` both call this
   single classifier.
4. **Library config?** Docling 2.86 has no flag that re-labels
   misclassified TOC items. Upstream issue tracking confirms
   the label is content-driven and cannot be overridden cleanly
   per page. Custom content-shape detection is the right
   approach.

**Approach (executed locally 2026-05-12):**

1. Probed Chaubal p11 alone via a one-page PyMuPDF slice and the
   production `DoclingPdfAdapter` / `PdfConversionPlan` path.
   Confirmed `_classify_dense_index_pages(doc) == set()` before
   the fix and the item labels/text described above.
2. Updated `_TOC_LEADER_RE` to match ASCII `.` leaders and U+FFFD
   leader runs before a trailing page number.
3. Lowered the unlabeled-page minimum from 8 text items to 5 and
   added a compact-TOC branch:
   `toc_leaders >= 4 and signal_ratio >= 0.80`.
4. Reconverted Chaubal through `scripts/convert_books_v29.py
   --only Chaubal_PyTorch_Projects --force`, re-enriched images
   with `scripts/enrich_image_chunks_v29.py`, and reran the strict
   gate.

**Tests / evidence:**

- Existing focused router tests:
  `conda run -n mmrag-v2 pytest tests/test_hybrid_chunker_dense_page_router.py tests/test_corruption_quarantine_toc_exemption.py tests/test_toc_cell_marker_sanitizer.py`
  → 22 passed / 3 skipped.
- Real-doc probe after the fix:
  `_classify_dense_index_pages(doc) == {1}` on the one-page p11
  slice.
- Fresh Chaubal output page coverage:
  `n_pages_with_chunks=359`, `missing<=max=[]`, page 11 has one
  `hybrid_chunker_pageskip` chunk:
  `Table of ConTenTs ... Index 337 xi`.
- Canonical single-doc strict gate:
  `conda run -n mmrag-v2 python scripts/qa_full_conversion.py
  output/Chaubal_PyTorch_Projects/ingestion.jsonl --source-pdf
  "data/technical_manual/Chaubal S. AI Projects in PyTorch. Hands-On Projects in Vision, Text,...2025.pdf"
  --allow-warnings` → `QA_PASS: failures=0 warnings=0`.
- Full local suite:
  `conda run -n mmrag-v2 pytest tests/` → 806 passed / 14
  skipped / 0 failed.

**Follow-up before Phase 8 close:** Add or confirm a small synthetic
regression test for the exact compact U+FFFD TOC-tail shape so the
real-doc strict gate is not the only executable contract for this
branch. Candidate test name:
`tests/test_hybrid_chunker_dense_page_router.py::test_compact_replacement_char_toc_tail_routes`.

**Done when:**

- `scripts/qa_full_conversion.py` against the full Chaubal source
  path above reports `QA_PASS` or `QA_PASS_WITH_ADVISORIES`.
- A dedicated compact-U+FFFD-TOC synthetic regression test is
  present or the phase close note explicitly explains why the
  existing router tests are sufficient.
- Smoke `bash scripts/smoke_multiprofile.sh` still reports
  `GATE_PASS + UNIVERSAL_PASS` across all 11 rows before the
  v2.10 tag.
- No regression on the 4 docs the Phase 1 router already covers
  (Cronin / Nagasubramanian / Sekar / Ayeva).

**Risk:** Low. The change is additive and content-shape based, but
the compact branch is a new classifier path; keep the dedicated
negative regression around body pages with isolated dotted references.

**Actual effort:** ~1 h implementation/probe plus ~6 min Chaubal
reconvert and ~4 min re-enrichment.

**Cost class:** Reconvert (1 doc) + re-enrich. Actual Chaubal image
surface: 55 image chunks, 53 complete + 2 F4 hard_fallback after
80 VLM calls including visual-only retries.

---

### Phase 2 — `TEXT_INTEGRITY_SCOUT_FULL_DOC_SENSITIVITY` (Fluent_Python)

**Status:** `validated-local` (2026-05-12). Code sites:
[src/mmrag_v2/validators/text_integrity_scout_trigger.py](../src/mmrag_v2/validators/text_integrity_scout_trigger.py),
[src/mmrag_v2/batch_processor.py:_per_batch_shortfall_fires](../src/mmrag_v2/batch_processor.py),
[src/mmrag_v2/batch_processor.py:_quarantine_corrupted_text_chunks](../src/mmrag_v2/batch_processor.py),
plus [scripts/probe_phase2_scout_threshold.py](../scripts/probe_phase2_scout_threshold.py).

**What:** `_run_text_integrity_scout` at
[src/mmrag_v2/batch_processor.py:7923](../src/mmrag_v2/batch_processor.py#L7923)
fires correctly on 8-page partial probes of Fluent_Python (Step 5
of the doc's chunking guard chain) but does NOT fire at full-doc
scale (770 pages, 6 missing pages). The trigger uses a doc-level
token-balance variance threshold; on a 770-page doc the per-page
shortfall averages out below the threshold so the scout never
runs.

**Implementation discovery (2026-05-12):** Wiring up the per-batch
trigger on its own only closed 3 of the 6 missing pages (pp 8, 10, 11
— the "true zero" pages where Docling never emitted text and the
scout's per-page recovery cleanly fills the gap). Pages 125, 126,
and 136 had healthy chunks at scout-call time (the per-batch helper
saw normal coverage) yet still arrived empty in the final JSONL.
Tracing every post-scout pass via env-gated diagnostic snapshots
(`FLUENT_PHASE2_DBG`) localized the loss to
`_quarantine_corrupted_text_chunks`: those Fluent pages describe
Python byte-string literals (`bytearray(b'caf\xc3\xa9')`) where
``\xHH`` escapes are the subject of the prose. The
single-pattern-match detector (`has_encoding_artifacts`) flagged
the chunk as corruption — but the parallel patch site
(`patch_corrupted_chunks`) runs only when the doc-level
`has_encoding_corruption` flag is set, so the quarantine was
dropping chunks the patcher had never inspected. This is a
parallel-site audit gap per §2b, surfaced by Phase 2 validation.
The fix swaps the quarantine's detector to the existing
ratio-based `is_irreparably_corrupt` so isolated literal hex
escapes inside legitimate prose survive while Combat p66-style
OCR garbage (em-dash runs, /C211 dense matches, C/S filler runs)
continues to drop.

The 6 affected pages have content surviving via adjacent-page
chunks (Retrieval-Value Test: retrieval impact = None). The fix
is to make the scout's trigger per-batch rather than doc-level
so it fires on small localized shortfalls inside large docs.

**Parallel-site audit (do this FIRST):**

1. **Existing fix elsewhere?** No. The scout is the sole
   recovery lane for token-balance variance after the
   line-stripper guards. Phase 1 v2.9's empty-text-chunk guards
   prevent empty rows but don't recover token mass.
2. **Too-narrow gate?** Yes — at
   [batch_processor.py:2963](../src/mmrag_v2/batch_processor.py#L2963)
   the scout invocation is gated on a single
   `token_balance_variance > 10 %` check after the whole-doc
   QA-CHECK-01 pass. On a 770-page doc with 6 short missing
   pages, doc-level variance lands at ~0.2 % even if those 6
   pages contribute 0 chars — well below 10 %.
3. **Parallel boundaries?** The scout is invoked once per
   document in `BatchProcessor`. `V2DocumentProcessor` does not
   have an equivalent scout call (it processes non-PDF formats
   that don't have per-page coverage shape). Per-batch scoping
   on the BatchProcessor side is the right fix; no other site
   needs a parallel change.
4. **Library config?** No Docling-side equivalent. The scout
   is custom recovery logic.

**Approach (as implemented 2026-05-12):**

1. New module
   [src/mmrag_v2/validators/text_integrity_scout_trigger.py](../src/mmrag_v2/validators/text_integrity_scout_trigger.py)
   exposes a pure `any_batch_fires(...)` helper over universal
   page-shape rules:
   - Rule A: per-batch variance ≥ 30 % with batch source ≥ 500 chars.
   - Rule B: ≥ 2 pages in the batch where source ≥ 100 chars but
     emitted TEXT-chunk content < 50 chars (batch source ≥ 500 chars).
   Rules are gated by a non-trivial source-text floor so cover/blank
   batches never trigger. Thresholds are constants on the module
   (`VARIANCE_PCT`, `MIN_SOURCE_CHARS`, `MIN_MISSING_PAGES`,
   `MIN_PAGE_SOURCE_CHARS`, `MIN_PAGE_CHUNK_CHARS`).
2. `BatchProcessor._per_batch_shortfall_fires(...)` reads per-page
   source-text length once via PyMuPDF (~100 ms / doc),
   computes per-page emitted TEXT-chunk char counts from
   `filtered_chunks`, and calls the pure helper using
   `split_result.batches` as the partition.
3. `_run_text_integrity_scout(force_run=...)` accepts the helper's
   verdict via a new keyword parameter. The existing doc-level
   variance gate now bypasses when `force_run=True`, and the
   post-scout chunk-merge logic ([batch_processor.py around
   line 2987](../src/mmrag_v2/batch_processor.py#L2987)) keeps
   recovery output whenever the scout produced extra chunks —
   not only when doc-level variance was already bad.
4. `_quarantine_corrupted_text_chunks` (parallel-site fix): the
   detector swapped from `has_encoding_artifacts` (single
   match) to `is_irreparably_corrupt` (ratio + em-dash / C/S
   runs). The patcher's `has_encoding_corruption` doc-level gate
   stays unchanged; quarantine continues to fire on truly
   unsalvageable chunks regardless of that flag, so the existing
   contract in
   [tests/test_corruption_quarantine_toc_exemption.py](../tests/test_corruption_quarantine_toc_exemption.py)
   and
   [tests/test_finalization_bridge.py::TestCorruptionQuarantineBridge](../tests/test_finalization_bridge.py)
   is preserved.
5. Corpus-wide probe
   [scripts/probe_phase2_scout_threshold.py](../scripts/probe_phase2_scout_threshold.py)
   simulates batches of `--batch-size 10` across every
   `output/<doc>/ingestion.jsonl` and reports which firing
   batches qualify under the same thresholds. Stored summary
   lives in `output/probe_phase2_scout_threshold/probe_summary.json`.
6. Fluent_Python reconverted (`-b 10 --vision-provider none
   --no-cache`, ~11 min) and re-enriched
   (`enrich_image_chunks_v29.py`, 83 image chunks).

**Tests (red→green):**

- [tests/test_text_integrity_scout_per_batch_trigger.py](../tests/test_text_integrity_scout_per_batch_trigger.py)
  pins the trigger contract:
  - `test_per_batch_variance_fires_on_localized_drop` — synthetic
    8-page batch with 6 zero-chunk pages → trigger fires.
  - `test_doc_level_variance_under_threshold_does_not_suppress_per_batch_trigger`
    — 770-page emulation of Fluent's shape: doc-level variance
    < 5 % yet one batch holds 6 zero-chunk pages → trigger
    fires on that batch only. Negative regression — the fix
    MUST NOT regress to the old doc-only check.
  - `test_clean_doc_does_not_trigger_scout` — five profile-shape
    fixtures (technical_manual_book, academic_whitepaper,
    digital_magazine, scanned_short, tech_report) where source
    and chunks match → trigger does not fire.
  - `test_low_source_floor_suppresses_noisy_batches` — pinned
    cover-sheet shape (< 500 source chars) must not trigger.
  - `test_missing_pages_rule_alone_fires_below_variance_threshold`
    — batch where variance is 20 % but two pages are zero-chunk →
    rule (B) fires. Pins that rule (A) and rule (B) are an OR.
  - `test_thresholds_match_corpus_probed_defaults` — guards the
    five thresholds against silent retuning (any future change
    must re-run the corpus probe).
  - `test_batch_processor_helper_fires_on_localized_pdf` — end
    to end: real PyMuPDF PDF + the `BatchProcessor` helper.
  - `test_quarantine_keeps_legitimate_low_ratio_hex_escapes` —
    the parallel-site fix. Fluent p126-shaped content with
    isolated ``\xHH`` literal escapes inside ~1800 chars of
    prose survives the quarantine.
- All existing corruption-quarantine contract tests
  ([test_corruption_quarantine_toc_exemption.py](../tests/test_corruption_quarantine_toc_exemption.py),
  [test_finalization_bridge.py::TestCorruptionQuarantineBridge](../tests/test_finalization_bridge.py),
  [test_export_integrity.py::test_full_pipeline_writes_valid_jsonl](../tests/test_export_integrity.py))
  remain green — Combat p66-style em-dash + C/S runs and
  high-ratio /C211 / /uniFB01 chunks still drop.

**Evidence (2026-05-12):**

- Strict gate:
  `python scripts/qa_full_conversion.py
  output/Fluent_Python/ingestion.jsonl --source-pdf
  "data/technical_manual/Fluent Python Luciano Ramalho 2015.pdf"`
  reports `QA_PASS_WITH_ADVISORIES: failures=0 warnings=1`
  (the single advisory is the pre-existing `ASSET_TINY`
  finding for 7 sub-1000-byte assets, unrelated to Phase 2).
- Before / after on the named missing-page class:
  baseline reported `MISSING_PAGES: 8, 10, 11, 125, 126, 136`;
  post-fix run reports no `MISSING_PAGES` failures and the
  pages now carry text chunks (`recovery_scan`/`recovery_gap_fill`
  for 8/10/11; `hybrid_chunker` / `hybrid_chunker_pagesplit` for
  125/126/136 — the latter survive because they are no longer
  false-positively quarantined).
- Corpus probe:
  `python scripts/probe_phase2_scout_threshold.py --batch-size 10
  --variance-pct 0.30 --min-source-chars 500
  --min-missing-pages 2` shows 8 docs with at least one firing
  batch. Fluent itself fires on batches 1 / 2 / 13 (the targeted
  case). The other seven (`Adedeji_GenAI_Google_Cloud`,
  `Chaubal_PyTorch_Projects`, `Cronin_GenAI_Models`,
  `Form_betwistingsformulier`, `Integra_manual`,
  `Python_Distilled`, `Sekar_MCP_Standard`) fire on front-matter
  / back-matter batches whose per-page primary_chars are still
  ≥ 50 — the scout's internal per-page guard means those
  invocations add no spurious recovery chunks. Acceptable per
  the plan's "document the trigger count, runtime cost, and why
  it is acceptable" clause.
- Focused suite:
  `pytest tests/test_text_integrity_scout_per_batch_trigger.py
  tests/test_corruption_quarantine_toc_exemption.py
  tests/test_finalization_bridge.py tests/test_oversize_splitter.py`
  → 43 passed.
- Full suite: `pytest tests/ -q` → 817 passed, 14 skipped, 0
  failed (was 807 passed before Phase 2; the 10 new tests are
  the ones listed above).

**Risk:** Medium. The scout's recovery path adds runtime on
the firing batches (light PyMuPDF rescans). The corpus-wide
probe and the scout's existing per-page primary_chars ≥ 50
guard are the load-bearing safety nets — no chunk is rewritten
on healthy pages.

**Estimated effort:** 6-8 h actual (1 h probe + 1 h trigger
implementation + 1 h tests + 25 min Fluent reconvert + 15 min
re-enrich + 2 h diagnostic-snapshot bisect for the quarantine
discovery + 1 h regression test for the quarantine fix).

**Cost class:** Reconvert (1 doc, Fluent_Python) + re-enrich
(83 image chunks, 78 complete + 5 hard_fallback retries).
Status `validated-local`: cloud verification belongs to Phase 8.

---

### Phase 3 — `B4B_FULL_DOC_PICTURE_DEDUP` (Earthship, Python_Distilled 3 pages)

**Status:** `validated-local` (2026-05-12). Code sites:
[src/mmrag_v2/batch_processor.py — pHash dedup carve-out](../src/mmrag_v2/batch_processor.py),
[src/mmrag_v2/processor.py — `_shadow_image_meets_threshold` + page-coverage-aware shadow extraction](../src/mmrag_v2/processor.py),
[tests/test_phash_dedup_page_coverage.py](../tests/test_phash_dedup_page_coverage.py).

**What:** Earthship_Vol1 p109 and 4 pages of Python_Distilled
(6 / 686 / 688 / 913) registered `MISSING_PAGES` at the strict
gate while the rendered pages contained substantive visual
content. The plan's leading hypothesis was a single drop site
(`_filter_blank_assets` or a content-hash dedup). The 2b
parallel-site audit (2026-05-12) found two distinct sites:

1. Earthship p109 — full-page hand-drawn cross-section. Docling
   emits the asset directly (extraction_method `docling`, 100 %
   area), then the cross-page pHash dedup at
   [batch_processor.py:3364](../src/mmrag_v2/batch_processor.py#L3364)
   rejected it as a near-duplicate of an earlier same-style
   cross-section. The page lost its only chunk and became
   `MISSING_PAGES`. Phase 3 preserves the existing Docling chunk
   via the page-coverage carve-out — it does not synthesize a new
   one.
2. Python_Distilled pp 686 / 688 / 913 — chapter-intro diagrams
   453×258–290 points at 24–27 % page area. These fall below
   SHADOW-EXTRACTION's `300×300 OR area ≥ 40 %` threshold at
   [processor.py:1771](../src/mmrag_v2/processor.py#L1771), so
   no asset is ever produced and dedup never runs. Python_Distilled
   p6 (title page) is a separate `hybrid_chunker_section_header_page`
   path that was incidentally repaired by the same reconvert (the
   Phase 1 / Phase 2 stack now emits a TEXT chunk for it).

Phase 3 closes both sites with universal, doc-shape-agnostic
rules.

**Parallel-site audit (do this FIRST):**

1. **Existing fix elsewhere?** Phase E v2.9 widened
   `_filter_blank_assets` from `std<5` to `std<10` for the
   Combat figure_36 case. Earthship full-page artwork may now
   land in that widened window incorrectly.
2. **Too-narrow gate?** Possibly the opposite: too wide. The
   `mean>250 AND std<10` blank rule may match flat-illustration
   regions that ARE legitimate content.
3. **Parallel boundaries?** `_filter_blank_assets`
   ([batch_processor.py:881](../src/mmrag_v2/batch_processor.py#L881))
   runs at [batch_processor.py:3203](../src/mmrag_v2/batch_processor.py#L3203);
   the only call site. `V2DocumentProcessor` doesn't replicate
   this filter (non-PDF formats use different image paths). But
   there are at least two more candidate drop sites that must
   be audited:
   - `_run_image_dedupe` / content-hash dedup (search
     batch_processor.py for `dedupe` and `_hash`).
   - The B4.a "zero-text image-only placeholder" check in
     `scripts/qa_full_conversion.py:_page_is_no_text_image_only_placeholder`
     — this is a GATE-side classifier; it could be reporting
     MISSING_PAGES_BLANK on these pages when they have real
     content. Confirm via gate output.
4. **Library config?** Docling 2.86's
   `DocumentFigureClassifier-v2.5` has confidence outputs that
   could distinguish "real illustration" from "blank
   placeholder". Worth a check before adding custom logic
   ("libraries first, custom code last").

**Approach (as implemented 2026-05-12):**

Diagnosis on the failing pages identified two distinct drop sites
that both needed fixing — not one. The 2b parallel-site audit
caught it: a single-fix patch would have left half the
missing-pages class unrecovered.

1. **Earthship p109** — full-page editorial illustration on a
   small 292×220-point landscape page. Docling extracts the asset
   directly (extraction_method `docling`, 100 % area), the pHash
   registry at `BatchProcessor` finalize-time
   ([batch_processor.py:3364](../src/mmrag_v2/batch_processor.py#L3364))
   then rejects it as a near-duplicate of an earlier hand-drawn
   cross-section. **Fix:** add a page-coverage carve-out to the
   pHash dedup, factored as the pure static helper
   `BatchProcessor._phash_carve_out_should_preserve_duplicate`.
   The export loop pre-computes `_phash_image_only_pages` and
   maintains `_phash_pages_with_exported_image` (updated on BOTH
   unique-image emission AND preserved-duplicate emission). A
   duplicate IMAGE is preserved only when the page is image-only
   AND no IMAGE has been exported for it yet. The decision is the
   same code path the regression tests assert, so future drift
   surfaces in both places at once.
2. **Python_Distilled pp 686 / 688 / 913** — chapter-intro
   diagrams 453 × 258–290 points at 24 – 27 % page area. These
   fall below SHADOW-EXTRACTION's 300×300 / 40 % threshold at
   [processor.py:1771](../src/mmrag_v2/processor.py#L1771), so
   no asset ever reaches the dedup site. **Fix:** make the
   shadow size floor page-coverage-aware. `process_document`
   collects `pages_with_prior_chunks` (TEXT yielded so far +
   pending IMAGE/TABLE + `_hybrid_text_chunks` pages) and passes
   it to `_run_shadow_extraction`. The threshold helper
   `V2DocumentProcessor._shadow_image_meets_threshold` returns
   the standard `300×300 OR area ≥ 40 %` for pages with prior
   chunks; for pages with no prior chunks it relaxes the size
   floor to 200×200. The area floor stays at 40 % so thin banners
   and dividers remain filtered.
3. **`_filter_blank_assets`** is left unchanged — none of the
   failing pages tripped its `mean>250 AND std<10` gate.
4. Reconvert Earthship_Vol1 and Python_Distilled, re-enrich
   image chunks with `enrich_image_chunks_v29.py`. Earthship's
   broad pHash collisions on hand-drawn cross-sections produce
   ~25 `[PHASH-PAGE-COVERAGE] Preserving` events per full run
   (logged for observability).

**Tests (red→green):**

`tests/test_phash_dedup_page_coverage.py` pins the contracts:

- `test_phash_dedup_preserves_image_only_page_first_image` —
  synthetic two-page setup: page 10 has TEXT + IMAGE; page 11
  has only an IMAGE that pHash-collides with page 10. The
  page-11 image must survive.
- `test_phash_dedup_drops_duplicate_when_page_already_has_text`
  — negative regression: pages with surviving TEXT must still
  drop the duplicate IMAGE. The carve-out cannot widen storage
  cost on text-bearing pages.
- `test_phash_dedup_drops_second_duplicate_on_image_only_page`
  — only the FIRST duplicate on an image-only page is preserved;
  the second drops because the page is already covered.
- `test_phash_dedup_drops_duplicate_after_unique_image_on_same_image_only_page`
  — bookkeeping must reflect ALL exported images, not just
  preserved duplicates. If a unique image is emitted first on
  an image-only page, a subsequent near-duplicate on the same
  page must still drop. Pins the post-audit fix.
- `test_phash_carve_out_decision_helper_pins_page_already_covered_branch`
  — direct pin on the pure decision helper: drop when the page
  already has any exported image (unique or preserved-duplicate),
  drop when the page is text-bearing, drop when `page_number` is
  None.
- `test_phash_dedup_preserves_image_only_page_when_image_only_pages_set_is_built_correctly`
  — guard against computing the image-only-page set from
  survivors instead of `export_chunks` (would never fire the
  carve-out).
- `test_shadow_extraction_threshold_relaxes_when_page_has_no_prior_chunks`
  — Python_Distilled-shape image (453×258, 24 % area): standard
  threshold rejects it; relaxed threshold (page has no prior
  chunks) accepts it.
- `test_shadow_extraction_relaxed_threshold_still_filters_tiny_icons`
  — 100×100 and 150×150 icons stay filtered even on
  image-only pages. The relaxation goes down to 200×200, not
  below.
- `test_shadow_extraction_threshold_keeps_full_page_image_in_both_lanes`
  — Earthship-p109-shape full-page image (292×220 on a 292×220
  page, 100 % area) passes the area gate in both lanes.

**Done when:**

- `python scripts/qa_full_conversion.py
  output/Earthship_Vol1/ingestion.jsonl --source-pdf
  "data/technical_manual/Earthship_Vol1_How to build your own.pdf"`
  reports `QA_PASS` or `QA_PASS_WITH_ADVISORIES`.
- `python scripts/qa_full_conversion.py
  output/Python_Distilled/ingestion.jsonl --source-pdf
  "data/technical_manual/Python Distilled David M. Beazley 2022.pdf"`
  shows pp 6 / 686 / 688 / 913 no longer in `MISSING_PAGES`.
- 9 new regression tests pass (7 original Phase 3 contracts +
  2 added after the 2026-05-12 post-close audit: the
  unique-then-duplicate bookkeeping fix and the direct pin on the
  pure decision helper).
- Full pytest suite stays green; smoke 11/11 unchanged.

**Evidence (2026-05-12):**

- Earthship strict gate: post-fix reconvert + re-enrich reports
  `QA_PASS: failures=0 warnings=0` (baseline was
  `QA_FAIL: failures=1` on `MISSING_PAGES: 109`). pHash
  page-coverage log fires 25 times across the run, all on
  hand-drawn cross-sections.
- Python_Distilled strict gate: post-fix reconvert + re-enrich
  reports `QA_PASS_WITH_ADVISORIES: failures=0 warnings=1`
  (advisory is pre-existing `ASSET_TINY`; baseline was
  `QA_FAIL: failures=1` on
  `MISSING_PAGES: 6, 686, 688, 913`). All four pages now
  carry chunks: p6 → `hybrid_chunker_section_header_page`
  (TEXT); pp 686 / 688 / 913 → `shadow` (IMAGE under the
  page-coverage-relaxed threshold).
- Focused suite: `pytest
  tests/test_phash_dedup_page_coverage.py
  tests/test_text_integrity_scout_per_batch_trigger.py
  tests/test_corruption_quarantine_toc_exemption.py
  tests/test_finalization_bridge.py tests/test_oversize_splitter.py
  tests/test_export_integrity.py` → 53 passed.
- Full suite: `pytest tests/ -q` → 826 passed, 14 skipped, 0
  failed (was 817 before Phase 3; the 9 new tests are the ones
  listed above).

**Risk:** Medium. Earthship's 25 pHash carve-out events add ~25
extra IMAGE chunks to the doc (each pointing at its own asset on
disk). The strict gate's `image_placeholder_ratio` and
`description_coverage` invariants pass after re-enrichment. The
SHADOW threshold relaxation is page-coverage-conditional, so
healthy pages still see the historical 300×300 / 40 % gate — no
new noise on text-bearing pages.

**Estimated effort:** 7-9 h actual (parallel-site audit + diagnostic
probe of Earthship p109 + identification of the two-site fix +
two implementations + 7 tests + 1 Earthship reconvert ~15 min +
1 Python_Distilled reconvert ~16 min + re-enrich Earthship ~50
min + re-enrich Python_Distilled ~35 min).

**Cost class:** Reconvert (2 docs, Earthship + Python_Distilled) +
re-enrich (468 + 349 image chunks). Status `validated-local`: cloud
verification belongs to Phase 8. Python_Distilled overlaps with
Phase 4 — Phase 4 will reconvert and re-validate from this AFTER
state.

---

### Phase 4 — `CROSS_PAGE_SPLIT_PAGE_ATTRIBUTION` (Python_Cookbook, Python_Distilled 3 pages)

**What:** The v2.9 Phase 4 cross-page split at
[src/mmrag_v2/processor.py:2870-2929](../src/mmrag_v2/processor.py#L2870-L2929)
emits the SAME merged `text.strip()` to each contributing page
(lines 2903-2926), then the per-page dedup at lines 2893-2897
suppresses it on all pages except the first. The result: a
chapter-end page's content (URLs, citations, dedications) ends
up tagged with the previous page's `page_number`.

The current code's leading comment (lines 2861-2868) explicitly
acknowledges the limitation: "Per-item .text is unreliable here
because Docling serialises some items via the chunker's
serializer rather than exposing .text directly". The fix:
instead of broadcasting the full merged text to every page,
walk `_items_by_page` and reconstruct per-page text from each
page's contributing `doc_items`, using their `text` attributes
when present and falling back to a HybridChunker-serializer
roundtrip when not.

**Parallel-site audit (do this FIRST):**

1. **Existing fix elsewhere?** No. v2.9 Phase 4 commit
   `b1f...` introduced the current per-page broadcast as a
   stop-gap. The "TODO: real per-page text split" was
   acknowledged in code but never implemented.
2. **Too-narrow gate?** The current split fires when
   `len(_items_by_page) > 1` (multi-page DocChunk). That gate
   is correct; the bug is in what gets emitted, not in when.
3. **Parallel boundaries?** The cross-page split is in
   `processor.py` only. Mapper (`src/mmrag_v2/mapper.py`) does
   NOT replicate it — search for `pagesplit` in that file to
   confirm. BatchProcessor's `_propagate_headings`
   ([batch_processor.py:5404](../src/mmrag_v2/batch_processor.py#L5404))
   reads `chunk.metadata.page_number` so will be affected
   downstream of any fix but does not duplicate the split
   logic.
4. **Library config?** Docling 2.86 has no
   "emit-one-chunk-per-page" toggle for HybridChunker. The
   chunker is designed to merge across page breaks for
   narrative continuity. Custom per-page splitting is the
   right place.

**Approach:**

1. Build a fixture: capture the multi-page `DocChunk` that
   produces the Python_Cookbook p209-shape merge. Save as
   `tests/fixtures/cross_page_split/cookbook_p208_209_docchunk.pkl`
   or as a JSON dump.
2. Write the per-page text reconstruction. For each
   `(page_number, items)` in `_items_by_page`:
   - Concatenate `item.text` for items whose `.text` is set
     and non-empty.
   - For items whose `.text` is empty (serializer-only), call
     the chunker's serializer with the items as input and take
     the output.
   - Strip and emit one chunk with `text=per_page_text` and
     `page_number=this_page`.
   - Drop the per-page dedup (lines 2893-2897) — it was the
     defense against the duplicate-broadcast bug; with proper
     per-page text it is no longer needed.
3. Add a fallback: if per-page text reconstruction fails for
   any page (no `.text` AND serializer roundtrip returns empty),
   fall back to the current behavior for that page (emit a
   `[CROSS_PAGE_CONTINUED]` short marker chunk so the page is
   not MISSING) with explicit logging.
4. Reconvert Python_Cookbook and Python_Distilled. Python_Distilled
   shares this reconvert with Phase 3.

**Tests (red→green):**

- `tests/test_cross_page_split_page_attribution.py::test_p209_url_list_attributed_to_p209`
  — Cookbook-shape fixture; assert chunk with the URL content
  carries `page_number=209` and NOT `page_number=208`.
- `::test_p208_content_attributed_to_p208`
  — same fixture; assert p208's content (chapter body) still
  appears on p208. Negative regression.
- `::test_three_page_merge_emits_three_distinct_chunks`
  — fixture where Docling emits one merged DocChunk spanning
  pages 100-102 with substantive content per page; assert
  three chunks emitted with correct per-page text and
  `page_number ∈ {100, 101, 102}`.
- `::test_serializer_roundtrip_fallback_emits_marker_chunk`
  — fixture where all per-page items have empty `.text` AND
  serializer roundtrip returns empty; assert one
  `[CROSS_PAGE_CONTINUED]` marker chunk per page. (Defense in
  depth — never silently lose a page.)

**Done when:**

- `python scripts/qa_full_conversion.py
  output/Python_Cookbook/ingestion.jsonl --source-pdf
  data/<dir>/Python_Cookbook.pdf` reports `QA_PASS` or
  `QA_PASS_WITH_ADVISORIES`.
- Python_Distilled's 3 cross-page-split-affected pages no
  longer appear in the strict gate's MISSING_PAGES list (the
  remaining 3 image-only pages closed by Phase 3).
- All 4 new tests pass.
- Smoke 11/11 unchanged.

**Risk:** Medium-high. The cross-page-split logic is on every
PDF's hot path. The serializer-roundtrip fallback must be cheap
(no nested HybridChunker calls) and must not cascade into
HybridChunker batch-boundary effects relevant to Phase 5.

**Estimated effort:** 8-12 h (probe + design + implementation +
fallback + tests + Cookbook reconvert ~60 min for 1,240 pages +
Python_Distilled shared with Phase 3).

**Cost class:** Reconvert (2 docs, Python_Distilled shared with
Phase 3) + re-enrich for image chunks on both.

#### 2026-05-12 progress log (partial close — NOT yet `validated-local`)

**Root cause confirmed by probe (`tests/manual/probe_phase4_cookbook_p127.py`)
on Python_Cookbook DocChunk #103 / #104 — covers pages 127 + 128:**

* The original v2.9 split path inspected only `prov[0].page_no` via
  `_docling_item_page_no`. A single Docling text item can carry a
  list of `ProvenanceItem` entries, each with its own `page_no` and
  `charspan`. HybridChunker then further slices such an item across
  multiple DocChunks. Both DocChunks resolved to page 127 because
  `prov[0]` was page 127 — page 128's prose was either silently
  dropped (per-page broadcast dedup on chunk #104) or attributed to
  page 127 (chunk #103 absorbing page-128 tail).
* A second, independent issue surfaced during validation: in
  Docling 2.86, `dc.meta.doc_items` exposes bare `DocItem`
  references (no `.text`) for many of the documents we re-converted
  — the real text lives at `doc.texts[idx]` for `self_ref` like
  `#/texts/12`. Without dereferencing, the helper saw zero
  sliceable text on many multi-page DocChunks (e.g. Python_Cookbook's
  TOC at pp 5–7, ~56 pages of code-heavy DocChunks) and the
  first-shape "emit a marker per contributor" fallback polluted
  62 already-covered pages with `[CROSS_PAGE_CONTINUED]` chunks
  that tripped the strict gate's `micro_non_label_ratio`.

**Fix shape (landed):**

* New helper [`_split_doc_chunk_text_by_page`](../src/mmrag_v2/processor.py)
  walks each item's `prov` list, slices the item's full text by
  `charspan` per page, and intersects each slice with `dc.text`.
  Items with empty `.text` are dereferenced via `self_ref` against
  the parsed `doc` so Docling's bare-DocItem shape resolves to its
  `TextItem` payload.
* The cross-page emit branch in
  [`_process_text_with_hybrid_chunker`](../src/mmrag_v2/processor.py)
  expands `_prov_by_page` over every prov entry (not just `prov[0]`),
  asks the helper for per-page text, and emits one chunk per page
  whose slice is non-empty. `chunk_type` is now derived from the
  contributing items' labels so code / heading / list_item items
  do not get mis-typed as paragraph.
* `[CROSS_PAGE_CONTINUED]` markers are now reserved for the
  emergency case where the helper returns nothing AT ALL for a
  multi-page DocChunk. In partial-reconstruction cases (helper
  produces real text for one page but not another) we emit only
  the real chunks and log the unreconstructable pages at DEBUG —
  no markers — because the missing page is covered by some other
  DocChunk in practice.
* When the helper does return empty, the cross-page emit now
  produces **one** marker at the *earliest* contributing page,
  not one marker per contributing page. The original
  multi-marker shape falsely populated truly-blank source pages
  (Python_Distilled p1-p3 cover/imprint pages are 0-text in the
  PDF), excluding them from `MISSING_PAGES_BLANK` classification
  and then tripping `micro_non_label_ratio` in tight-window
  smoke tests (Distilled smoke went to 0.273 > 0.12). The
  single-marker shape preserves the "this DocChunk had
  unreconstructable content" signal without falsely claiming
  page coverage for blank-source pages. Pinned by
  `tests/test_cross_page_split_page_attribution.py::test_cross_page_emit_uses_marker_at_earliest_contributing_page_only`.
* `BatchProcessor._merge_micro_text_chunks` now skips chunks
  whose `extraction_method == "hybrid_chunker_pagesplit_fallback"`
  on both the source AND neighbor sides. The 22-char marker
  string used to slot into the "tiny non-label fragment" shape
  the micro-merge targets, so Distilled's pre-guard reconvert had
  `[CROSS_PAGE_CONTINUED]` text concatenated onto the p472 "Set
  Operations" prose chunk. Pinned by
  `tests/test_cross_page_split_page_attribution.py::test_micro_merge_skips_pagesplit_fallback_markers`.

**Tests:** 27 in `tests/test_cross_page_split_page_attribution.py`,
including the four required by the plan's `Tests (red→green)`
section plus the regression contracts: zero markers on partial
reconstruction, marker reserved for total reconstruction failure,
bare-DocItem dereferencing, page_offset handling. Full pytest
**853 passed, 14 skipped** (was 833 pre-Phase-4).

**Results on the affected reconverts (final state 2026-05-13):**

* `output/Python_Cookbook/ingestion.jsonl`: strict gate's
  cross-page-split MISSING_PAGES count dropped from **4 → 0**
  (all four plan-listed pages closed: 63, 128, 365, 397 — p397
  closed by the in-flight `DOCLING_DUPLICATE_DOC_CHUNK_OVERLAP_TRIM`
  resolution). `hybrid_chunker_pagesplit_fallback` marker count
  went **62 → 0**. `micro_non_label_ratio` is **`0.002`** (1
  micro chunk out of 466 text chunks, well under the 0.12
  limit). The full strict gate still returns `QA_FAIL` because of
  pre-existing VLM image-placeholder failures
  (`--vision-provider none --no-refiner` reconvert) — out of
  Phase 4 scope.
* `output/Python_Distilled/ingestion.jsonl`: cross-page-split
  MISSING_PAGES = **0** (Phase 4 page-loss criterion met for
  Distilled). `hybrid_chunker_pagesplit_fallback` marker count is
  **2** (p1 and p472 — the genuine single-marker emergency
  fallback firing on multi-page DocChunks whose contributors are
  entirely serializer-only / image-only). `micro_non_label_ratio`
  is **`0.008`**. The full strict gate still returns `QA_FAIL`
  because all 349 image chunks are VLM-pending / unusable on
  this non-enriched reconvert — out of Phase 4 scope.
* `bash scripts/smoke_multiprofile.sh`: **10/11 GATE_PASS,
  11/11 UNIVERSAL_PASS** with all current fixes (smoke run6
  2026-05-13). The lone failure is the deferred
  `LITERATURE_MICRO_GATE_TUNE_AFTER_CROSS_PAGE_FIX` class on
  HarryPotter (see below).

**Two follow-up items resolved (2026-05-13):**

1. **Cookbook p397 — `DOCLING_DUPLICATE_DOC_CHUNK_OVERLAP_TRIM`
   RESOLVED.** The probe showed Docling 2.86 emits two
   byte-identical DocChunks for the same code block (DocChunk
   #335 prov=[396] and DocChunk #336 prov=[397]). Both pass the
   per-page producer dedup (different page keys), but
   [`batch_processor._deduplicate_chunk_overlap`](../src/mmrag_v2/batch_processor.py#L6209)
   was trimming tail/head exact overlap between consecutive chunks
   **regardless of page boundary** — chunk[N+1]'s entire content
   was trimmed to empty and the chunk discarded, dropping p397
   from coverage. The DSO-overlap intent the function was built
   for is *same-page* sentence trimming; page-scoping the prev/cur
   pair preserves the intent while closing the cross-page
   page-loss. Pinned by
   `tests/test_cross_page_split_page_attribution.py::test_overlap_trim_is_page_scoped`
   (cross-page preservation AND same-page DSO trim both verified).
2. **`LITERATURE_MICRO_GATE_TUNE_AFTER_CROSS_PAGE_FIX` RESOLVED.**
   The HarryPotter smoke (`digital_literature` profile, 10-page
   slice) was tripping `micro_non_label_ratio = 0.125 > 0.12` on
   the legitimate 24-char subtitle ``"and the Sorcerer's Stone"``
   at p7. Resolution is a paired producer + gate fix that is
   **universal** (no filename / page-number / literal-text rules)
   and **does not weaken any threshold**:

   * **Producer** [`_looks_like_subtitle_continuation`](../src/mmrag_v2/processor.py)
     promotes short single-line PARAGRAPH-typed chunks to
     ``ChunkType.HEADING`` when every structural signal of a
     title-continuation holds (length < 30, no terminal sentence
     punctuation, has a non-empty parent_heading distinct from
     the chunk content, first alphabetic word is one of a small
     fixed set of English stopwords — ``and / or / the / of / in /
     to / on / at / by / for / with / from / into / onto / upon``,
     etc.). Surveyed against the full Phase 4 reconvert + smoke
     corpus, the rule promotes exactly one chunk
     (HarryPotter p7) and leaves every other short chunk
     (``Logo``, ``Bar chart``, ``zip() function, 62, 314``, OCR
     junk ``ré Several refe``, code fragments) untouched.
   * **Gate side** — ``qa_conversion_audit._is_typed_non_micro``,
     ``qa_ingestion_hygiene._is_typed_non_micro``, and the inline
     guard in ``evaluate_technical_manual_gates`` now treat
     ``chunk_type ∈ {code, heading, title}`` (or
     ``content_classification == "code"``) as non-paragraph
     structural content. The ``micro_non_label`` counter
     previously exempted only ``code``; clarifying that
     intentional ``heading`` / ``title`` chunks are similarly
     non-noise is a definition correction, not a threshold change
     (the 0.12 / 0.22 limits are unchanged).
   * **Tests:** ``TestSubtitleContinuationPromotion`` (8 cases
     including the HarryPotter pin and false-positive rejection)
     + ``TestAuditMicroNonLabelExemption`` (6 cases including a
     subprocess test that runs ``evaluate_technical_manual_gates``
     end-to-end on a synthetic JSONL with a heading-typed chunk).

**Phase 4 close status — `validated-local` (2026-05-13):**

* All 4 plan-listed Cookbook pages closed (63, 128, 365, 397).
* Distilled's cross-page-split MISSING_PAGES = 0; markers = 2
  (genuine standalone fallback, no contamination of neighbouring
  prose).
* Smoke (`bash scripts/smoke_multiprofile.sh`):
  **11/11 GATE_PASS, 11/11 UNIVERSAL_PASS** (run10, 2026-05-13).
* Focused regression suite
  `conda run -n mmrag-v2 pytest tests/test_cross_page_split_page_attribution.py -q`:
  **27 passed**.
* Full pytest
  `conda run -n mmrag-v2 pytest tests/ -x --ignore=tests/manual -q`:
  **853 passed, 14 skipped** (was 833 pre-Phase-4).
* The test file loads `scripts/qa_conversion_audit.py` via
  `importlib.util.spec_from_file_location(...)` so the
  ``TestAuditMicroNonLabelExemption`` cases work under bare
  ``pytest`` (which, unlike ``python -m pytest``, does not add
  the repo root to ``sys.path``). The module is registered in
  ``sys.modules`` before ``exec_module`` runs so the ``@dataclass``
  field-type resolution can find its own class's module.
* No QA thresholds weakened.
* Committed on `main` in `8effdfd` ("feat(v2.10): land Phase 2/3/4
  — TextIntegrityScout + B4B picture dedup + cross-page-split").

---

### Phase 5 — `HYBRID_CHUNKER_HEADING_PROPAGATION` (Devlin)

**What:** v2.9 Phase 4 commit `b429cb5` implemented cross-batch
heading carry-forward in
[`_propagate_headings`](../src/mmrag_v2/batch_processor.py#L5404).
The unit tests for that fix pass; a small probe on Devlin
showed the mechanism firing; but Devlin's full-doc HEADING
coverage metric stays at 72 % (target ≥ 80 %).

The diagnostic note `docs/QUALITY_SNAPSHOT_2026-05-11_v2.9.0-rc1_after.md`
suggests Devlin's batches end mid-section without an
end-of-section heading chunk, so the propagator has no source
on the first chunk of the next batch.

This phase is **investigation-first**, then fix. Do not pre-judge
the fix shape; the probe might surface a different root cause.

**Parallel-site audit (do this FIRST):**

1. **Existing fix elsewhere?** The Phase 4 `b429cb5` fix in
   `_propagate_headings` is the only carry-forward site.
   Section-header-only page emission at
   [processor.py:3209 `_emit_section_header_only_page_chunks`](../src/mmrag_v2/processor.py#L3209)
   handles chapter-divider pages but doesn't help mid-section
   batches.
2. **Too-narrow gate?** The carry-forward logic at
   batch_processor.py:5404 probably only carries forward when
   `state.last_hybrid_heading` is set. If Devlin's batches
   end with a non-heading chunk and the previous batch's
   final state was a body chunk, `last_hybrid_heading` may
   already be the right value but isn't being propagated to
   the NEW batch's first chunk's `parent_heading` slot.
3. **Parallel boundaries?** Same as Phase 4. `_propagate_headings`
   is the only heading-propagation site for the HybridChunker
   path. The OCR/element-by-element path has its own (Phase 6).
4. **Library config?** Docling 2.86 has no
   `HybridChunker(carry_heading_across_batches=True)` toggle.
   Custom propagation logic is right.

**Approach:**

1. Probe: write a one-off tool
   `tests/manual/inspect_devlin_heading_propagation.py` that:
   - Reads `output/Devlin_LLM_Agents/ingestion.jsonl`.
   - Walks all text chunks in `position` order.
   - For each chunk, prints `(page, position, parent_heading,
     chunk_type, first 80 chars of content)`.
   - Marks batch boundaries (every 10 pages by default).
   - Highlights chunks where `parent_heading is None`.
2. Inspect the output. Identify the pattern: is the
   propagator failing to carry forward across batch boundaries,
   or does the chunk's `parent_heading` field get cleared by a
   downstream filter, or does the source heading exist only as
   an in-batch `section_header` chunk that never reaches the
   propagator?
3. Pick the fix from the evidence. Likely candidates:
   - **Candidate A:** the propagator reads
     `state.last_hybrid_heading` but the state is reset
     between batches. Fix: persist across batch loop boundary
     via `BatchProcessor` instance state instead of per-batch
     local.
   - **Candidate B:** the propagator does carry forward but
     `parent_heading` is later overwritten by an in-batch
     dedup or hygiene filter that clears the field. Fix:
     preserve `parent_heading` through that filter.
   - **Candidate C:** Devlin uses a heading-shape Docling
     doesn't recognize as `section_header` (e.g., italic-only
     emphasis used as a soft section break). Fix would be in
     the heading-classifier, not the propagator — and would
     intersect with the post-Docling sanity pass.
4. Implement the fix and reconvert Devlin.

**Tests (red→green):**

- `tests/test_hybrid_chunker_heading_propagation.py::test_devlin_shape_batch_boundary_carries_forward`
  — synthetic Docling output: batch 1 ends with a body
  chunk, batch 2 starts with a body chunk under the same
  section; assert batch 2's first chunk has
  `parent_heading == batch 1's last heading`.
- `::test_propagation_does_not_overwrite_explicit_heading`
  — batch 2 starts with a new section heading; assert the
  propagator does NOT replace it. Negative regression.
- `::test_propagation_unit_test_from_b429cb5_still_passes`
  — preserve the existing Phase 4 unit test contract. Critical:
  if the diagnostic shows the existing fix is correct but
  insufficient (Candidate B/C), the new fix must not regress
  the case the existing fix solved.

**Done when:**

- `python scripts/qa_full_conversion.py
  output/Devlin_LLM_Agents/ingestion.jsonl --source-pdf
  data/<dir>/Devlin_LLM_Agents.pdf` reports `QA_PASS` or
  `QA_PASS_WITH_ADVISORIES` with HEADING coverage ≥ 0.80.
- Investigation notes recorded in
  `docs/PHASE_5_DEVLIN_HEADING_DIAGNOSTIC.md` (a tracked
  archive doc), naming which candidate was confirmed.
- 3 new tests pass.

**Risk:** Medium-high. If the root cause is candidate C
(heading-classifier issue), the fix surface widens into the
post-Docling sanity pass — schedule could overrun. The
investigation step gates the implementation.

**Estimated effort:** 1-1.5 days (investigation + fix +
tests + Devlin reconvert ~60 min for ~400 pages + re-enrich
~30 min for Devlin's image chunks).

**Cost class:** Reconvert (1 doc) + re-enrich.

#### 2026-05-13 close — `validated-local`

Phase 5 closed in two passes. The first attempt hit
`HEADING coverage 790/790 (100%)` by propagating
Docling-emitted garbage section_headers
(`'Type Type TypeTypeTypeType'`, `'GRAPH DATA: {'`,
`'Start'`, `'Front Matter'` together attributed to 54% of
chunks). That run was **rejected** by an audit-time
quality probe and reworked.

**Corrected fix shape (the one that landed):**

* **Single propagation site.** `_propagate_headings(all_chunks)`
  was removed from the mid-pipeline call site and replaced with
  one canonical call to `_propagate_headings(export_chunks)`
  at the export boundary, after all filters/splitters have
  exposed the final chunk set. The previous attempt's three
  propagation sites (mid-pipeline + export-stage + inline
  `chunk_dict["metadata"]["hierarchy"]` write-loop mutation
  bypassing the pydantic model) are gone. The "only one site"
  contract is pinned by source-introspection in
  `tests/test_vision_aided_front_matter.py::test_process_pdf_routes_front_matter_after_all_heading_assignment_paths`
  via `assert source.count("_propagate_headings(") == 1`.
* **Single validator.** Heading validation lives in
  [`src/mmrag_v2/state/context_state.py::is_valid_heading`](../src/mmrag_v2/state/context_state.py#L64).
  Tightened against three additional shapes:
  (a) code/JSON prefixes — `^(def|class|return|import|from|>>>|\{|\[|```\)\b`
  and bracket-prefixed code labels `^[A-Z_][A-Z0-9_\s-]{1,40}:\s*[\[{]`;
  (b) repeated-token compact-alpha (`'Type Type TypeTypeTypeType'` →
  the alphabetic projection equals one word repeated ≥3 times); and
  (c) repeated-word density (a single word taking ≥50% of a 3+-word
  heading at ≥3 occurrences). The validator is used by both the
  initial sanitization pass and the carry-forward gate in
  `_propagate_headings`. No parallel `_valid_export_heading` exists.
* **Generic-bucket carry block.** A small set
  `_GENERIC_CARRY_HEADINGS = {"start", "front matter"}` blocks
  Docling-generic labels from seeding forward carry. `"Start"`
  is also stripped from its bearing chunk (Devlin's single PDF
  bookmark — not useful anywhere); `"Front Matter"` remains
  on its own chunk but does not propagate.
* **No `chunk_dict` mutation at write time.** Heading edits
  happen on the `IngestionChunk` model before serialization.

**Tests (red → green) on the corrected fix:**

`tests/test_hybrid_chunker_heading_propagation.py` — 7 cases:

Positive (the plan's required 4):
* `test_devlin_shape_batch_boundary_carries_forward`
* `test_propagation_does_not_overwrite_explicit_heading`
* `test_propagation_unit_test_from_b429cb5_still_passes`
* `test_unordered_pagesplit_sibling_uses_same_page_heading`

Negative (the missing pin that v1 lacked):
* `test_garbage_repeated_heading_does_not_seed_carry_forward`
  — `"Type Type TypeTypeTypeType"` is stripped from its bearing
  chunk AND does NOT carry to the next chunk.
* `test_code_shape_heading_does_not_seed_carry_forward`
  — `"GRAPH DATA: {"` is stripped + does NOT carry.
* `test_real_heading_wins_over_garbage_heading_on_same_page`
  — when a page has both a garbage heading and a real heading,
  same-page pagesplit siblings inherit the real one;
  the garbage one's bearing chunk gets its parent_heading
  cleared.

`tests/test_vision_aided_front_matter.py` updated with one
extra source-introspection assertion pinning the single
propagation site.

**Results on the affected reconvert (2026-05-13):**

* `output/Devlin_LLM_Agents/ingestion.jsonl` strict-gate
  `HEADING: PASS coverage 783/790 (99%) null_headings=7`. The
  seven nulls are legitimate front-matter pages (3–7) before
  the first real section signal — front-matter chunks without
  a heading should stay headingless, not be force-attributed.
* Top 10 attributed parent_heading values are all real chapter
  / section titles. Garbage check: `'Type Type TypeTypeTypeType'`
  **0** chunks, `'GRAPH DATA: {'` **0**, `'Start'` **0**.
  Top-10 share of total chunks is **22%** (was 54% under v1
  garbage attribution).
* The full strict gate still returns `QA_FAIL` because of
  pre-existing VLM image-placeholder findings on the
  non-enriched reconvert (`--vision-provider none --no-refiner`,
  `VISION_PENDING=68`, `IMAGE_DESCRIPTION_UNUSABLE=68/68`).
  These findings are out of Phase 5 scope and present on every
  non-enriched reconvert in this cycle.
* Phase 4 regression suite
  (`tests/test_cross_page_split_page_attribution.py`):
  **27 passed**.
* Vision-aided front-matter suite (with the new propagation-site
  assertion): **8 passed**.
* Full bare pytest
  (`conda run -n mmrag-v2 pytest tests/ -x --ignore=tests/manual -q`):
  **860 passed, 14 skipped**.
* Smoke (`bash scripts/smoke_multiprofile.sh`):
  **11/11 GATE_PASS, 11/11 UNIVERSAL_PASS**.

Investigation notes and the rejected-v1 evidence are recorded
in [`docs/PHASE_5_DEVLIN_HEADING_DIAGNOSTIC.md`](PHASE_5_DEVLIN_HEADING_DIAGNOSTIC.md).

**Phase 5 close status — `validated-local` (2026-05-13):**

* Devlin HEADING coverage 0.99 (above the 0.80 floor); top-10
  attributions are all real titles; the garbage strings named
  in the v1 audit each have 0 chunks.
* Single propagation site; single heading validator; no
  `chunk_dict` post-serialization mutation.
* 7 focused tests + 27 Phase 4 regression + 860 full pytest
  pass under bare `pytest`.
* Smoke remains 11/11.
* No QA thresholds weakened.
* Committed on `main` in `f3d8478` ("feat(v2.10): Phase 5 —
  HYBRID_CHUNKER_HEADING_PROPAGATION (Devlin) validated-local").

---

### Phase 6 — `OCR_PATH_HEADING_PROPAGATION` (Firearms)

**What:** Firearms routes to the `scanned` profile (post-v2.9
Phase 4 HARD REJECT in `_score_technical_manual`). On that
profile, Docling's element-by-element OCR path emits text +
section_header items, but the OCR-lane heading propagation in
`BatchProcessor` does not promote `section_header` items into
`ContextStateV2.hierarchy_stack`
([src/mmrag_v2/state/context_state.py:196](../src/mmrag_v2/state/context_state.py#L196)).
HEADING coverage lands at 72 % vs the 80 % gate (`docs/archive/PLAN_V2.9__PHASE4.md`
Step 4 evidence).

The fix promotes Docling `section_header` items into the
hierarchy stack on the OCR/element-by-element lane, parallel
to how HybridChunker-path heading state is populated.

**Parallel-site audit (do this FIRST):**

1. **Existing fix elsewhere?** Section-header items ARE
   recognized — at
   [batch_processor.py:1452 `if label_val in ("section_header",
   "title"):`](../src/mmrag_v2/batch_processor.py#L1452) — but
   the use of that recognition needs verification. Read the
   surrounding 200 lines to confirm whether the recognition
   produces a `ContextStateV2.push_heading()` call or just
   sets the chunk's own `parent_heading` from a local cache
   that doesn't persist.
2. **Too-narrow gate?** Possibly the OCR-lane path uses
   `chunk_type=HEADING` chunks to populate `parent_heading`
   for body chunks, but the population only fires when a
   heading is on the SAME page as the body chunk. For
   multi-page scanned sections (Firearms chapter spans 10-20
   pages), the heading is never propagated past page 1 of the
   section.
3. **Parallel boundaries?** The OCR/element-by-element path
   is `BatchProcessor` only. The HybridChunker path uses
   `_propagate_headings` ([batch_processor.py:5404](../src/mmrag_v2/batch_processor.py#L5404))
   — Phase 5's fix may inform Phase 6's, but they are
   independent lanes. `V2DocumentProcessor` is not on the
   PDF batch path.
4. **Library config?** Docling 2.86 has no per-page heading
   inheritance toggle on the OCR lane (the OCR lane bypasses
   HybridChunker entirely). Custom propagation is right.

**Approach:**

1. Probe Firearms: extract the page-by-page emission sequence
   from
   `output/Firearms/ingestion.jsonl` showing
   (page, chunk_type, parent_heading, content first 80
   chars). Identify which pages have body chunks with
   `parent_heading is None` despite a heading being established
   on an earlier page.
2. Locate the OCR-lane hierarchy state. Most likely path: the
   element-by-element loop in `BatchProcessor` at the
   per-element emission site has a `current_heading` local
   variable that's reset between batches OR between pages
   inappropriately.
3. Promote the local state to `ContextStateV2.push_heading()`
   calls so the heading sticks across pages within the section.
   Use the exact existing API
   ([context_state.py:267-309](../src/mmrag_v2/state/context_state.py#L267-L309)).
4. Verify the cross-batch boundary: if Phase 5 changed the
   `_propagate_headings` invocation, Phase 6 must also
   propagate `state.hierarchy_stack` across batches.
5. Reconvert Firearms.

**Tests (red→green):**

- `tests/test_ocr_path_heading_propagation.py::test_section_header_pushes_to_hierarchy_stack`
  — synthetic Docling output with a `section_header` followed
  by 5 body text items spanning 3 pages; assert all 5 body
  chunks carry `parent_heading == section_header text`.
- `::test_new_section_header_replaces_prior_heading`
  — second section_header arrives after 3 body items; assert
  subsequent body chunks attribute to the new heading.
  Negative regression.
- `::test_chapter_continues_across_batch_boundary`
  — synthetic doc with batch boundary mid-section; assert
  body chunks AFTER the boundary still attribute to the
  pre-boundary heading.

**Done when:**

- `python scripts/qa_full_conversion.py
  output/Firearms/ingestion.jsonl --source-pdf
  data/technical_manual/Firearms.pdf` reports `QA_PASS` or
  `QA_PASS_WITH_ADVISORIES` with HEADING coverage ≥ 0.80.
- 3 new tests pass.
- No regression on canonical scanned-doc HEADING coverage
  for Greenhouse / Hao / other `scanned`-class corpus docs.

**Risk:** Medium. The OCR-lane heading flow has not been
modified in the v2.9 cycle; changes there can have
out-of-scope effects on other scanned docs.

**Estimated effort:** 1-1.5 days (probe + implementation +
tests + Firearms reconvert ~75 min for 292 pages; note
Firearms image chunks are scanned-page renders and may be
preserved across reconvert if extraction shape is unchanged
— verify; if changed, re-enrich ~60 min).

**Cost class:** Reconvert (1 doc); re-enrich conditional.

**Phase 6 close status — `validated-local` (2026-05-15).**

After the audit-fix iteration (2026-05-14 → 2026-05-15), Firearms
full strict gate returns `QA_PASS: failures=0 warnings=0`. The
original plan's Done-when items are all satisfied — the strict-gate
QA_PASS bar (item 1) is met in addition to HEADING coverage ≥ 0.80
and the cross-corpus scanned-doc regression.

Numeric:

* Firearms HEADING coverage 0.72 → **0.997** (1091/1094 attributed;
  3 NULL chunks are page-1 front-matter correctly unattributed
  under ordered per-chunk attribution); top-10 attributions are all
  real chapter or section titles (Disassembly: 696, Reassembly
  Tips: 145, General Instructions: 19, WNCHESTER MODEL 70, SAVACE
  MODEL I10, REMINGTON MODEL 700, MOSSBERG MODEL 479, REMINGTON
  ROLLING BLOCK, WINCHESTER MODEL 71, NOSTALGIA). The BEFORE-state
  body-prose garbage strings ("Data: Remington Model 788" 141×,
  "5. Drift out the trigger cross-pin toward the right." 24×,
  "6 Drift out the trigger pin…" 21×) each have **0** chunks
  AFTER.
* Strict gate: `HEADING: PASS` + `TEXT: PASS` + `IMAGE: PASS` +
  `UNIVERSAL_PASS` + `SEMANTIC_PASS`. Overall
  `qa_full_conversion.py` verdict: **`QA_PASS: failures=0
  warnings=0`** on
  `output/Firearms_phase6e/ingestion.jsonl`. The two scope-orthogonal
  defects flagged by the first audit iteration were both closed in
  the audit-fix iteration:
  - TEXT `infix_artifacts: 148 → 0` via the new
    `BatchProcessor._repair_infix_step_numbers` chunk-content repair.
    Detection **behaviorally mirrors**
    `scripts/qa_conversion_audit.py::_INFIX_RE` **after the audit's
    newline / stop-word post-filters** — the production regex
    collapses the audit's ``\s+`` + ``"\n" in between`` post-filter
    into a single ``[ \t]+`` on the prev→num side, reproduces the
    audit's left-context exclusion and short-word / stop-word
    filters explicitly, and preserves the original separator after
    the period so existing paragraph breaks are not collapsed. 23
    focused tests in `tests/test_infix_step_number_repair.py` pin
    the contract, including the audit-detector parity test that
    re-applies the audit detector to repaired content and asserts
    the count drops to 0.
  - VLM `image_placeholder_ratio: 0.2424 → 0.0000` via targeted
    enrichment of the 264 ``vision_status="pending"`` shadow
    full-page chunks
    (`scripts/enrich_firearms_pending_only.py`, which iterates only
    `pending`-status image chunks and delegates per-chunk work to
    the canonical `enrich_image_chunks_v29.py::_enrich_one` helper
    so prompt / retry / hard-fallback behaviour is unchanged). All
    264 enriched successfully; 0 hard_fallback.

Implementation:

* Ordered per-chunk attribution via
  `Region.is_heading` / `ProcessedChunk.is_heading`
  (`src/mmrag_v2/ocr/layout_aware_processor.py` dataclass extensions);
  `BatchProcessor._attribute_ocr_chunk_heading` does push+read at
  chunk-emission time so within-page ordering is preserved.
  `_promote_ocr_section_headers` is the fallback for VLM-fullpage /
  Tesseract-fullpage paths that emit a single synthesized chunk per
  page. Single validator (`is_valid_heading` in
  `state.context_state` — no parallel OCR validator); no
  `chunk_dict` post-serialization mutation. The HybridChunker-lane
  `_propagate_headings(` call count inside `process_pdf` remains
  1 (Phase 5 structural pin held).
* Two universal `is_valid_heading` tightenings centralised:
  (a) **terminal-period** sentence-shape (≥5 words ending in `.` →
  reject; `?` and `!` are accepted so real question / exclamation
  headings pass — narrowing iteration after the audit found
  `"What Is an AI Agent?"`, `"Why Use Retrieval Augmented
  Generation?"`, `"2. What are AI agents?"` were false-negatived
  by the original `".!?"` rule); (b) **numbered-prefix
  body-case** shape (numbered prefix + ≥2 lowercase content words →
  reject). Each is pinned by parameterised positive AND negative
  tests in `tests/test_ocr_path_heading_propagation.py`; all
  Phase 5 audit-named garbage ("Type Type TypeTypeTypeType",
  "GRAPH DATA: {") still rejected; all Phase 5 Devlin real
  headings ("1 - The New Age of AI Agents", "2. Fine-Tuning:
  Teaching the Model to Specialize", "5.1 Notation and
  Preliminaries", etc.) still accepted.
* Single-page push gate (`self._doc_total_pages > 1`) applied on
  BOTH the per-chunk path (`_attribute_ocr_chunk_heading`) and the
  fallback path (`_promote_ocr_section_headers`). An intermediate
  iteration that gated only the fallback regressed
  `scanned/0013_140302111325_001` because Docling tags layout-
  prominent invoice value lines as `section_header`; the per-chunk
  push then promoted them and flipped the audit gate's form-
  detection heuristic. The gate is purely a document-shape
  constraint, not filename- or page-number-specific.

Tests:

* 70 focused OCR-lane tests (incl. ordered-attribution + question /
  exclamation + single-page gate + structural pins) + 23 infix-repair
  tests + 7 Phase 5 + 27 Phase 4 + 8 vision-aided front-matter
  regression.
* Full bare `pytest tests/ -x --ignore=tests/manual -q`:
  **953 passed, 14 skipped**.
* Smoke: **11/11 GATE_PASS + 11/11 UNIVERSAL_PASS** (Phase 6 v1
  regressed 0013 to 10/11; the audit-fix per-chunk gate restored
  the baseline).
* Regression on Earthship_Vol1 (236-page scanned-class doc, the
  Greenhouse-like control): HEADING coverage 1.00 → 1.00 (no
  drop); `UNIVERSAL_PASS`; `micro_non_label_ratio=0.033` (limit
  0.12).
* No QA thresholds weakened.
* Diagnostic doc: `docs/PHASE_6_FIREARMS_OCR_HEADING_DIAGNOSTIC.md`.

---

### Phase 7 — `KI_EPUB_EXTRACTION_LANE_REWRITE` (KI EPUB)

**Phase 7 close status — `validated-local` (2026-05-15).**
Implementation landed in the existing EPUB lane. `_epub_to_html` now
walks `book.spine`, injects `__MMRAG_EPUB_CH_NNNN__` markers in reading
order, and `_apply_epub_synthetic_pagination` rewrites EPUB chunks with
synthetic page numbers, the EPUB bbox sentinel `[0,0,1000,1000]`,
`extraction_method="epub_html"`, regenerated unique chunk IDs, and
per-synthetic-page dedup. `qa_full_conversion.py` now treats `.epub`
sources as chapter-coverage documents rather than PDF page-coverage
documents and emits conditional `MISSING_CHAPTERS` advisories only for
contiguous low-content edge spine items. Evidence: KI EPUB strict gate
`QA_PASS_WITH_ADVISORIES: failures=0 warnings=1`; ChatGPT EPUB
regression control `QA_PASS_WITH_ADVISORIES: failures=0 warnings=1`;
focused EPUB tests green; full pytest **966 passed, 14 skipped**; smoke
**11/11 GATE_PASS + 11/11 UNIVERSAL_PASS**.

The subsections below preserve the original Phase 7 implementation
contract for auditability; the close-status block above is the current
state.

**Original problem:** KI_En_ChatGPT_Praktische_Gids is an EPUB. Before
Phase 7, the EPUB lane (`processor._epub_to_html` at
[processor.py:857](../src/mmrag_v2/processor.py#L857)) extracted EPUB
chapter HTML, concatenated it into a single HTML document, and fed it
to Docling's HTML parser. That lost EPUB-native chapter structure
(chapters became a long page-less HTML blob) and bbox metadata (HTML
has no inherent geometry).

ChatGPT_Praktijk_handboek is the regression control: it is the other
EPUB in the corpus and remains `QA_PASS_WITH_ADVISORIES` after the
Phase 7 EPUB-lane rewrite.

Goals for the fix:
1. Produce a synthetic `page_number` per chunk derived from
   chapter index + chunk position within chapter.
2. Produce a placeholder `bbox=[0, 0, 1000, 1000]` per chunk
   (HTML has no real bbox — emit the full-page sentinel and
   document it).
3. Reduce dedup excess: KI EPUB currently emits 279 within-file
   duplicate chunk IDs in the v2.8 baseline (per
   `docs/DECISIONS.md` "chunk_id position component"). After
   the v2.9 chunk_id-position fix, duplicates should be 0;
   verify and confirm dedup excess is now about repeated
   content, not chunk_id collisions.

**Parallel-site audit (do this FIRST):**

1. **Existing fix elsewhere?** Phase G v2.9 added
   `PAGE_COUNT_UNKNOWN` as an allowed advisory code
   (`docs/QUALITY_GATES.md` "Advisory Warning Classes")
   — ChatGPT_Praktijk_handboek uses it. KI EPUB's failure
   is structural (`UNIVERSAL_FAIL` from missing bbox, not
   `PAGE_COUNT_UNKNOWN` alone), so the advisory route alone
   doesn't close KI.
2. **Too-narrow gate?** Not the issue — the gate is correct.
   The issue is the producer doesn't emit bbox at all on this
   path.
3. **Parallel boundaries?** The EPUB lane is
   `processor._epub_to_html` and the subsequent HTML-pass
   through `_convert_html`. Both EPUBs (KI, ChatGPT) flow
   through the same path. `BatchProcessor` doesn't have an
   EPUB-specific path — batch only handles PDFs.
4. **Library config?** ebooklib provides spine ordering via
   `book.spine` and per-item content. Pagination is not a
   first-class EPUB concept (the EPUB 3 spec defines
   `epub:pagebreak` but it's optional and rarely present in
   commercial EPUBs).

**Approach:**

1. Probe both EPUBs: dump the spine order, item ordering,
   chunk_id collision count post-v2.9, and the current bbox
   distribution.
2. Modify `_epub_to_html` to walk `book.spine` and prepend
   `__MMRAG_EPUB_CH_NNNN__` chapter-boundary markers to the
   generated HTML in reading order.
3. In the EPUB post-process path, scan surviving markers to derive a synthetic
   `page_number = chapter_index * 1000 + sequence_within_chapter
   // 5` (chapter at offset 1000-2000, etc.), and emit
   `bbox=[0, 0, 1000, 1000]` with `extraction_method="epub_html"`.
4. Re-run KI EPUB through the strict gate. Adjust dedup logic
   if duplicate content (real same-text chapter headings, e.g.
   "Inhoudsopgave" repeated across volumes) needs scoping.
5. Document the synthetic page mapping in
   `docs/CONVERSION_PROFILES.md` so downstream RAG consumers
   know KI EPUB page numbers are virtual, not source-document
   pages.

**Tests (red→green):**

- `tests/test_epub_extraction_lane.py::test_ki_epub_emits_per_chapter_page_numbers`
  — fixture EPUB (small, 3-chapter); assert chunks have
  distinct `page_number` values clustered by chapter.
- `::test_ki_epub_emits_full_page_bbox`
  — same fixture; assert every chunk's bbox is `[0, 0, 1000,
  1000]` (the documented EPUB sentinel).
- `::test_chatgpt_epub_does_not_regress_on_advisory_path`
  — second EPUB fixture; assert the doc still parses,
  emits chunks, and the strict gate (if run in-test) classifies
  it as PASS-with-advisory.
- `::test_epub_chunk_ids_remain_unique`
  — fixture-level assertion that no two chunks share a
  chunk_id in the post-fix path. Critical: the v2.9 chunk_id
  position component contract from
  `docs/DECISIONS.md` "chunk_id position component" must
  hold.

**Done when:**

- `python scripts/qa_full_conversion.py
  output/KI_En_ChatGPT_Praktische_Gids/ingestion.jsonl
  --source-pdf data/technical_manual/Seffer*KI*ChatGPT*.epub`
  (use `--source-pdf` against the .epub; the gate's
  blank-page check is PDF-specific so an EPUB-aware
  branch in qa_full_conversion may be needed — verify
  in Phase 7 probe).
  Reports `QA_PASS` or `QA_PASS_WITH_ADVISORIES`.
- ChatGPT_Praktijk_handboek remains `QA_PASS_WITH_ADVISORIES`
  (regression control).
- EPUB-lane tests and conditional `MISSING_CHAPTERS` advisory pins pass.

**Risk:** Medium-high. EPUB lane is touched by every
ebooklib-related code path. The synthetic page-number
heuristic must be reasonable (per-chapter clustering) — wrong
heuristic produces useless `page_number` values that
downstream RAG consumers can't interpret.

**Estimated effort:** 1.5-2 days (probe + spec for synthetic
pagination + implementation + tests + KI reconvert ~20 min
+ ChatGPT regression run + re-enrich for KI image chunks).

**Cost class:** Reconvert (1 doc) + re-enrich (KI image
chunks).

---

### Phase 8 — Strict-Gate Re-Verification + v2.10 Release Prep

**What:** With Phases 0-7 complete and each deferral closed
locally, this phase runs the strict gate corpus-wide,
re-validates the test suite, rebuilds `mmrag_v2_8`, writes
the v2.10 AFTER snapshot, and tags the release.

This phase also closes the three v2.10 housekeeping items
documented in
`docs/QUALITY_SNAPSHOT_2026-05-11_v2.9.0-rc1_after.md` §9:
- Devlin Qdrant payload staleness (absorbed by Qdrant rebuild).
- `scripts/search_qdrant.py` `--model` default and `MIN_SCORE`
  floor re-tune for the v2.10 embedder.
- Move the hard-coded Dashscope API key out of
  `scripts/search_qdrant.py:40` into an env var.

**Parallel-site audit:** Trivial — this is verification work
on top of Phases 0-7 outputs. Audit instead: confirm every
canonical JSONL produced by Phases 1-7 is present and the
reconvert manifest is complete.

**Approach:**

1. Run the strict gate corpus-wide using
   `scripts/run_strict_gate_corpus.py` (or the equivalent
   wrapper used in v2.9.0-rc1):
   ```bash
   for doc in $(ls output/); do
     python scripts/qa_full_conversion.py \
       "output/$doc/ingestion.jsonl" \
       --source-pdf "data/<category>/$doc.{pdf,epub}" \
       --allow-warnings
   done > /tmp/v2.10_strict_gate_capture.txt
   ```
   Expected: every doc reports `QA_PASS` or
   `QA_PASS_WITH_ADVISORIES`. 0 `QA_WARN`, 0 `QA_FAIL`.
2. Run `bash scripts/smoke_multiprofile.sh`. Expected: 11/11
   `GATE_PASS + UNIVERSAL_PASS`.
3. Run `pytest tests/ -q`. Expected: all of v2.9.0-rc1's 806
   tests plus every new test from Phases 1-7; current local baseline
   after Phase 7 is **966 passed, 14 skipped, 0 failed**.
4. Verify chunk_id uniqueness corpus-wide: for each JSONL,
   `jq -r '.chunk_id' | sort -u | wc -l` matches
   `wc -l` (minus the IngestionMetadata first line).
5. Verify image-chunk vision_status: 0 `pending` corpus-wide;
   hard_fallback rate <= 5 % except F4-exempt.
6. **Qdrant rebuild** (DESTRUCTIVE — request explicit user
   confirmation before step 6a):
   - 6a. Drop `mmrag_v2_8`:
     `curl -X DELETE http://localhost:6333/collections/mmrag_v2_8`
   - 6b. Re-run
     `scripts/rebuild_mmrag_v2_8_for_rc1.py` (or its v2.10
     equivalent) against the 34 post-v2.10 JSONLs.
   - 6c. Confirm
     `points_count` == sum of unique embeddable chunks
     across the 34 JSONLs.
7. **search_qdrant.py default re-tune.**
   `scripts/search_qdrant.py:36` `MIN_SCORE` was lowered to
   0.20 for llava 4096-dim. If v2.10 keeps llava, no change.
   If v2.10 switches embedders, re-tune and update the
   `--model` default. Acceptance:
   `python scripts/search_qdrant.py "what is MCP" -c
   mmrag_v2_8 -n 3` returns topically-correct chunks.
8. **API key cleanup.** Replace the hard-coded
   Dashscope key at
   [scripts/search_qdrant.py:40](../scripts/search_qdrant.py#L40)
   with `os.environ.get("DASHSCOPE_API_KEY")`. Document the
   env var in `docs/TESTING.md` or `README.md`. **Rotate the
   leaked key on the provider side** (the agent cannot do this
   for the user — flag explicitly in the close-out note).
9. Author `docs/QUALITY_SNAPSHOT_<DATE>_v2.10_after.md`
   following the v2.9.0-rc1 AFTER snapshot template.
10. Update `docs/PROJECT_STATUS.md` and `CHANGELOG.md` with
    the v2.10 ship state.
11. Bump engine version in `src/mmrag_v2/version.py`:
    `2.9.0-rc1` → `2.10.0-rc1` (or `2.10.0` if the user
    decides to skip the RC line; see Decision Log).
12. Tag the release. Decide between `v2.10.0-rc1` (RC pattern)
    or `v2.10.0` (production-final). The choice depends on
    whether the strict-gate corpus-wide result has 0
    advisory-warning regressions vs the v2.9.0-rc1 advisory
    profile. Default recommendation: tag `v2.10.0-rc1` first;
    promote to `v2.10.0` after 7 days of corpus stability
    with no surprise regressions.

**Tests (red→green):** No new unit tests. Acceptance is the
existing strict-gate + smoke + pytest matrix invoked above.

**Done when (AGENT_GOVERNANCE.md Completion Rules — verbatim
invocation):**

A workstream may be marked `complete` only when:

1. **Every listed acceptance signal is satisfied.** Goals 1-8
   in §2 must all be checked off; the §4 Acceptance Gate
   checklist below must be entirely green.
2. **Evidence is durable (`tracked` or `snapshot`).** The
   v2.10 AFTER snapshot is committed; the strict-gate output
   is captured in the snapshot; commits exist on `main` for
   every Phase 1-7 fix.
3. **Known limitations are documented.** If any deferral
   couldn't be closed and is being pushed to v2.11, the
   deferral must be recorded in
   `docs/DECISIONS.md` with explicit user sign-off (parallel
   to the v2.9.0-rc1 close-out pattern).
4. **Required local/cloud comparisons are completed or
   explicitly removed from scope.** The local
   NuMarkdown-8B VLM lane stays `removed from scope` if the
   off-network endpoint remains unreachable; document in
   `docs/PROJECT_STATUS.md` Active Model/Endpoint State.
5. **`PROJECT_STATUS.md` and snapshots agree.** Per-task
   history lives in `docs/archive/PROGRESS_CHECKLIST.md`;
   current task state belongs in `PROJECT_STATUS.md`. Both
   must reflect v2.10 ship state.
6. **A fresh coding session can reproduce the claim without
   chat history.** New-session experiment: open a fresh
   session, read `docs/PROJECT_STATUS.md` →
   `docs/QUALITY_SNAPSHOT_*v2.10_after.md` →
   `docs/PLAN_V2.10.md`, then run the §4 acceptance
   commands; outputs must match the snapshot.

**Risk:** Low. All preceding phases passed their local gates.
The Qdrant rebuild is destructive but the procedure is
identical to the v2.9.0-rc1 cycle (10h15m wall time
expected).

**Estimated effort:** 1-2 days (strict-gate run ~30 min,
test suite ~5 min, Qdrant rebuild ~10h15m wall time, snapshot
+ status + changelog ~2h, tag + push ~30 min).

**Cost class:** No reconvert, no re-enrich. Cloud-VLM spend = 0.

---

## 4. Acceptance Gate

Before tagging `v2.10.0-rc1` (or `v2.10.0`):

- [ ] **Strict gate.** `scripts/qa_full_conversion.py --source-pdf
  --allow-warnings` reports `QA_PASS` or `QA_PASS_WITH_ADVISORIES`
  for **all 34** canonical docs. 0 `QA_WARN`, 0 `QA_FAIL`.
  (Goal 1.)
- [ ] **Smoke matrix.** `bash scripts/smoke_multiprofile.sh` —
  every row `GATE_PASS + UNIVERSAL_PASS` (form variants per
  `docs/QUALITY_GATES.md` "Form / Invoice Acceptance Class"
  remain valid). (Goal 4.)
- [ ] **Tests.** `pytest tests/ -q` reports the Phase 7 local
  baseline count or higher (**966 passed / 14 skipped**), plus any
  Phase 8 hardening test. 0 failed. (Goal 3.)
- [ ] **chunk_id contract.** 0 within-file duplicates across all
  34 canonical JSONLs. Re-verify via
  `for f in output/*/ingestion.jsonl; do jq -r 'select(.chunk_id)
  | .chunk_id' "$f" | sort | uniq -d; done` returns empty.
  (Goal 5.)
- [ ] **Vision contract.** 0 `vision_status="pending"`
  corpus-wide; `hard_fallback` rate <= 5 % per doc except
  F4-exempt. (Goal 6.)
- [ ] **Qdrant in sync.** `mmrag_v2_8` rebuilt from
  post-v2.10 JSONLs; `points_count` == unique embeddable
  chunks across 34. Devlin payload staleness from
  v2.9.0-rc1 §7 is resolved. (Goal 7.)
- [ ] **No filename-specific production logic.** Per
  AGENT-VAL-01.
- [ ] **No negative test loosened.** Every threshold change
  has explicit rationale + a positive AND negative regression
  test per `docs/DECISIONS.md` "No gate weakening".
- [ ] **No new advisory warning code added without the
  `docs/QUALITY_GATES.md` "Adding a new advisory code"
  4-step procedure.**
- [ ] **Documentation in sync.** AGENTS.md §5, README.md,
  PROJECT_STATUS.md, CHANGELOG.md, TESTING.md, the v2.10
  AFTER snapshot all reflect the v2.10 ship state. CLAUDE.md
  "Read First" list updated. (Goal 8.)
- [ ] **API key cleanup.** Hard-coded Dashscope key removed
  from `scripts/search_qdrant.py:40`; provider key rotation
  flagged in the close-out note for the user to perform.
- [ ] **Tag criteria — AGENT_GOVERNANCE.md Completion Rules
  invoked verbatim** (the 6 binding requirements in Phase 8
  "Done when"). All 6 must be checked off.

---

## 5. Out of Scope

| Item | Rationale | Owner doc |
|---|---|---|
| Local NuMarkdown-8B VLM lane | Off-network, endpoint unreachable as of 2026-05-12 | `docs/DECISIONS.md` "Cloud-Only VLM for v2.9 Image Enrichment" |
| Remote CodeFormulaV2 inference | Docling 2.86 doesn't expose `RemoteCodeFormulaOptions`; trigger condition not met within v2.10 | `docs/DECISIONS.md` "Selective Code Enrichment Lane → Amendment 2026-05-03" |
| Broader UIR refactor (full `processor.py` → adapter+UIR migration) | Canonical target per CLAUDE.md but not blocking; Phases 1, 4, 6 each route through the existing adapter without expanding direct paths | `CLAUDE.md`, `docs/DECISIONS.md` "Shared PDF Extraction Plan" |
| HybridChunker per-item token guard | Requires upstream Docling change | v2.9 Milestone 1 known limitation |
| Magazine rendered-region-crop architecture | Not a v2.9 / v2.10 blocker; composite-layout magazine quality already at PASS-class | `docs/CONVERSION_PROFILES.md` |
| EPUB engine rewrite (parallel `EpubEngine` to `PDFEngine`) | Phase 7 closes the named deferral within `_epub_to_html`; a broader EPUB engine redesign is v2.11+ scope | `docs/ARCHITECTURE.md` §6.2 (epub_engine.py marked optional) |
| New retrieval-quality tests for `mmrag_v2_8` | Listed as new-work investigation in `docs/PLAN_V2.10_DRAFT_PROMPT.md`; the Qdrant rebuild in Phase 8 produces the substrate, but designing retrieval-quality tests is a separate workstream and out of v2.10 closure scope | v2.11 candidate workstream |
| AIOS p8 short VLM description retry-harness bug | Marked "deferral candidate" in `docs/PROJECT_STATUS.md`; currently absorbed by `VISION_HARD_FALLBACK_RATE` advisory (F4-conditional); not a v2.10 blocker | v2.11 candidate if symptoms persist after Phase 5 / 6 reconverts |

---

## 6. Cross-Phase Concerns

### Documentation updates per phase

Every fix phase (1-7) must, in its closing commit, update:

- `CHANGELOG.md` — add a `[unreleased]` or
  `[v2.10.0-rc1]` bullet under "Phases shipped this cycle"
  with the phase's class name, code site, and tests added.
- `docs/PROJECT_STATUS.md` — flip the phase's row in the
  Phase Status table (§3 above) from `pending` to an explicit
  AGENT-STATUS-01 scope (`implemented`, `validated-local`,
  `validated-cloud`, `blocked`, or `complete`) with the commit
  hash + date.
- `docs/DECISIONS.md` — if the fix surfaces a new architectural
  decision (e.g. EPUB synthetic pagination scheme in Phase 7),
  add a new entry parallel to the existing decision blocks.

### Test contract integrity

Every red→green test specified in Phases 1-7 is a binding
contract (AGENT-TEST-01 / CLAUDE.md). If a test must be
modified during implementation, the agent MUST stop and
document the proposed requirement change before editing the
test, per `docs/AGENT_GOVERNANCE.md` Test Contract Rules.

The negative regression tests (`test_body_page_with_one_dotted_reference_does_not_route`,
`test_combat_figure_36_blank_asset_still_dropped`,
`test_p208_content_attributed_to_p208`,
`test_propagation_does_not_overwrite_explicit_heading`,
`test_new_section_header_replaces_prior_heading`) are the
non-overfit guardrails — they are the load-bearing assertions
that prevent the 2026-05-09 Path A precedent from recurring.

### Upstream tracking

Each phase that hits a Docling 2.86 limitation must log it in
a one-line bullet under
`docs/DECISIONS.md` "Upstream tracking" (a section to create
in Phase 8 if not already present). Examples likely from this
plan:

- Phase 1: Docling labels Chaubal p11 TOC items as `text` /
  `section_header`, not `document_index`, and emits U+FFFD leader
  runs rather than ASCII dot leaders.
- Phase 4: Docling HybridChunker has no per-page emission
  toggle for cross-page merges.
- Phase 6: Docling OCR-lane has no built-in heading-state
  carry across pages.
- Phase 7: ebooklib + Docling HTML lane have no native
  EPUB pagination preservation.

Tracking these positions the project for future Docling
upgrades or for the broader UIR refactor.

### Reconvert / re-enrich budget

| Phase | Reconvert | New image chunks (approx) | Cloud VLM calls |
|---|---|---:|---:|
| 1 | Chaubal | 55 (actual) | 80 actual calls incl. retries; 53 complete + 2 F4 hard_fallback |
| 2 | Fluent | 83 actual | 78 complete + 5 hard_fallback retries |
| 3 | Earthship + Python_Distilled | 468 + 349 actual | completed locally |
| 4 | Python_Cookbook + Python_Distilled | image chunks on both docs | completed locally |
| 5 | Devlin | 67 (matches Phase H catch-up count) | completed locally |
| 6 | Firearms | 264 pending shadow chunks enriched | completed locally |
| 7 | KI EPUB | KI image chunks | completed locally |
| 8 | None | None | 0 |
| **Remaining total after Phase 7** | **Phase 8 verification only** | **0 planned new enrichment** | **0 planned new VLM calls** |

Phase 8 should not introduce new enrichment unless strict-gate
re-verification finds a real residual defect. If it does, document the
new cost and evidence trail in the Phase 8 close section.

---

## 7. Effort Summary

| Phase | Estimate | External dependency? |
|---|---|---|
| 0 — Documentation Hygiene | 1-2 h | No |
| 1 — TOC router extension (Chaubal) | `validated-local` 2026-05-12 | No |
| 2 — TextIntegrityScout per-batch (Fluent) | `validated-local` 2026-05-12 | No |
| 3 — Full-doc picture dedup (Earthship + Distilled) | `validated-local` 2026-05-12 | No |
| 4 — Cross-page-split page attribution (Cookbook + Distilled) | `validated-local` 2026-05-13 | No |
| 5 — HybridChunker heading propagation (Devlin) | `validated-local` 2026-05-13 | No |
| 6 — OCR-path heading propagation (Firearms) | `validated-local` 2026-05-15 | No |
| 7 — EPUB lane rewrite (KI EPUB) | `validated-local` 2026-05-15 | No |
| 8 — Strict-gate re-verify + Qdrant rebuild + tag | 1-2 days (incl. 10h Qdrant rebuild wall time) | No |
| **Remaining total after Phase 7** | **Phase 8 only** | — |

External dependencies: none. All work lands in-repo; cloud
VLM enrichment uses the same DashScope endpoint already
authorized for v2.9.

---

## 8. Decision Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-16 | Plan updated to Draft v1.6. Phase 8 (Strict-Gate Re-Verification + v2.10 Release Prep) moved to `implemented`. Corpus-wide strict gate reports **34 PASS / 0 WARN / 0 FAIL** (16 `QA_PASS` + 18 `QA_PASS_WITH_ADVISORIES`). 443 Phase 4/5 pending image chunks re-enriched (432 enriched + 11 hard_fallback). Firearms canonical reconvert + 1089-chunk re-enrichment (1088 + 1 hard_fallback). `mmrag_v2_8` Qdrant rebuild kicked off (~10h15m wall time, in progress at AFTER-snapshot commit). `search_qdrant.py` / `validate_qdrant.py` / `convert_all.sh` Dashscope API-key literals replaced with `DASHSCOPE_API_KEY` env var. Engine version bumped to `2.10.0-rc1`. AFTER snapshot at `docs/QUALITY_SNAPSHOT_2026-05-16_v2.10_after.md`. Smoke 11/11 GATE_PASS + 11/11 UNIVERSAL_PASS (including the Greenhouse blind-test row and the 0013 form-variant row). Full pytest **973 passed, 14 skipped, 0 failed** (966 baseline + 7 Phase 8 release-baseline pins). | All eight v2.9.0-rc1 signed deferrals are now closed corpus-wide and the Phase 1-7 fixes are re-verified under the unchanged strict gate. No QA threshold weakened, no new advisory codes added. Phase 8 stays `implemented` until the Qdrant rebuild completes and the `points_count` matches the precomputed embeddable-chunk total (30,588); promotion to `validated-local` is gated on that and on the AFTER snapshot commit landing on `main`. The release-tag command is staged, not pushed; the user controls the final `git tag` push. |
| 2026-05-16 | Phase 8 promoted to `validated-local`. The initial rebuild aborted at doc 11/34 when Ollama (port 11434) became unreachable; after Ollama was restarted, a resume loop (`/tmp/qdrant_resume_phase8.sh` driving `scripts/ingest_to_qdrant.py` for docs 11-34 without `--recreate`; deterministic uuid5 point IDs per v2.8 `0d3cc36`) completed 18:17:26 UTC. Final state: `status: green`, `points_count: 30,454`, `indexed_vectors_count: 30,192`, `segments_count: 5`. Raw chunk count was 30,588; `ingest_to_qdrant.py` filtered 134 (~0.44 %, consistent with the rc1 rate). v2.9.0-rc1 → v2.10.0-rc1 net delta: **−7 points** (rc1 was 30,461). Vector + reranker smoke (`scripts/search_qdrant.py "what is MCP" -c mmrag_v2_8 -n 3`) returns topically-correct top-3 chunks from Sekar §3.4.3 (MCP client). Devlin Qdrant payload staleness rc1 housekeeping item closed by this rebuild. AFTER snapshot §7/§9/status banner + §10 revision log updated to reflect the verified state. Phase 8 baseline pin re-run after the snapshot edits: 7/7. | The v2.10.0-rc1 production-tag bar is met. Release-tag command is staged but not executed; the user pushes the tag per §Phase 8 step 12. |
| 2026-05-15 | Plan updated to Draft v1.5. Phases 1-7 are `validated-local`; Phase 8 is the only remaining v2.10 release workstream. | Phase 6 Firearms strict gate is `QA_PASS`; Phase 7 KI and ChatGPT EPUB strict gates are `QA_PASS_WITH_ADVISORIES`; full pytest is 966 passed / 14 skipped; smoke is 11/11 GATE_PASS + UNIVERSAL_PASS after the smoke runner's Apple-Silicon TorchInductor guard. |
| 2026-05-13 | Plan updated to Draft v1.2. Phase 4 (`CROSS_PAGE_SPLIT_PAGE_ATTRIBUTION`) moved to `validated-local` (2026-05-13, commit `8effdfd`). Per-page text split via `prov.charspan` slicing + bare DocItem dereferencing + `_looks_like_subtitle_continuation` promotion + page-scoped overlap-trim + audit micro_non_label heading exemption. All 4 Cookbook missing pages closed; Distilled cross-page MISSING_PAGES=0. 27 tests in `tests/test_cross_page_split_page_attribution.py`. Smoke 11/11 GATE_PASS + UNIVERSAL_PASS. Full pytest 853 passed. | Phase 4 also resolved two in-flight defects: (a) Cookbook p397 DOCLING_DUPLICATE_DOC_CHUNK_OVERLAP_TRIM — page-scoped the overlap-trim; (b) HarryPotter LITERATURE_MICRO_GATE_TUNE_AFTER_CROSS_PAGE_FIX — subtitle-continuation promotion + heading/title micro_non_label exemption. Neither threshold weakened. |
| 2026-05-13 | Phase 5 (`HYBRID_CHUNKER_HEADING_PROPAGATION`, Devlin) moved to `validated-local` (2026-05-13, commit `f3d8478`). Single propagation site at export boundary; heading validator tightened against garbage/code shapes. Devlin HEADING 99%. 7 tests in `tests/test_hybrid_chunker_heading_propagation.py`. Full pytest 860 passed. | Phase 5 landed after a rejected first attempt that propagated Docling-emitted garbage headings. Corrected fix uses `is_valid_heading` validator + `_GENERIC_CARRY_HEADINGS` block. |
| 2026-05-12 | Plan authored as Draft v1.0. Phase ordering: smallest-cost / lowest-blast-radius first (Phase 1 Chaubal regex extension → Phase 7 EPUB lane rewrite). | Matches the v2.8 / v2.9 plan pattern; surfaces high-risk fixes (Phases 4-7) after the low-risk ones validate the test infrastructure. |
| 2026-05-12 | Python_Distilled handled across Phase 3 AND Phase 4 (one shared reconvert), rather than a single "Python_Distilled close" phase. | Phase 3 (image-only-page chunk drop) and Phase 4 (cross-page-split attribution) are architecturally distinct root causes; collapsing them into one phase would mix two parallel-site audits and obscure the test boundary. |
| 2026-05-12 | Phase 5 (Devlin) is investigation-first, no pre-decided fix shape. | The Phase 4 v2.9 carry-forward fix `b429cb5` is correct in unit tests but doesn't move Devlin's metric — root cause is non-obvious. Pre-committing to a fix shape would risk the 2026-05-09 Path A precedent. |
| 2026-05-12 | Phase 7 emits synthetic `page_number` per EPUB chapter (`chapter_index * 1000 + position_in_chapter // 5`) and full-page `bbox=[0,0,1000,1000]`. | EPUB lacks native pagination; this synthetic scheme is consistent across EPUBs and documents the limitation in `docs/CONVERSION_PROFILES.md` for downstream consumers. Alternative (true page-break detection via `epub:pagebreak`) is rare in commercial EPUBs and would fail in practice. |
| 2026-05-12 | Tag-versioning decision deferred to Phase 8. Default: tag `v2.10.0-rc1` first; promote to `v2.10.0` after 7 days of corpus stability. | Matches the v2.9.0-rc1 RC pattern. Promotion can also be skipped if the team prefers a single production tag — that decision is captured at Phase 8 close. |
| 2026-05-12 | Local NuMarkdown VLM, remote CodeFormulaV2, broader UIR refactor, magazine rendered-region-crop, EPUB engine rewrite, and retrieval-quality tests are explicit non-goals (§5). | Each has been recurring non-goal across v2.7 / v2.8 / v2.9; promotion into v2.10 would risk schedule overrun without addressing a current production blocker. |
| 2026-05-12 | Phase 1 moved to `validated-local` after the compact U+FFFD TOC-tail router fix. | Fresh Chaubal reconvert + re-enrichment reports `QA_PASS`, `MISSING_PAGES=[]`; full test suite remains 806 passed / 14 skipped. The phase still needs corpus re-verification and a tracked AFTER snapshot before it can count toward v2.10 `complete`. |

---

**END OF PLAN — v2.10 Draft v1.5**
