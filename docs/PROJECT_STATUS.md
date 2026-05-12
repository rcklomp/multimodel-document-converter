# Project Status

Last updated: 2026-05-12

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
v2.10 deferrals. Test suite: **806 passed, 14 skipped, 0 failed**.

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

**Next cycle:** v2.10. Planning seed at
[`docs/PLAN_V2.10_DRAFT_PROMPT.md`](PLAN_V2.10_DRAFT_PROMPT.md);
authoring `docs/PLAN_V2.10.md` is the next-session task.

---

## v2.9 Execution History (through 2026-05-12)

The phase-by-phase narrative that produced the RC1 close-out is
preserved below as execution history. **Authoritative ship state is
in §Current Objective above**; the close-out snapshot §2 "Phases
shipped in this cycle" is the canonical summary. The blow-by-blow
below is kept for archaeology only.

---

### v2.9 origins (Phase 0)

A v2.9.0 tag was created on
2026-05-05 against a 32/34 AUDIT_PASS reading from
`scripts/qa_conversion_audit.py` alone, then deleted on 2026-05-06
after a user-driven review surfaced multiple defects that the
single-script gate did not catch (HARRY chapter-intro pages
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
reported **9 PASS / 8 WARN / 17 FAIL out of 34**. Only 2 of the 17
FAILs are the signed Phase 4 deferrals; the other 15 are
previously-unmeasured failure classes. The previous v2.9 plan
(`docs/archive/PLAN_V2.9_2026-05-06_strict_gate_recovery.md`) has
been archived and replaced with a new
[`docs/PLAN_V2.9.md`](PLAN_V2.9.md) recovery plan organized around
nine work-streams (Phase A diagnostic → I tag). Phase A is complete
(`docs/PHASE_A_MISSING_PAGES_DIAGNOSTIC.md`): the MISSING_PAGES
surface decomposes into four distinct sub-classes (B1 TOC
quarantine over-fire, B2 "intentionally left blank" disclaimer,
B3 short-text page filter, B4 image-only page chunk-drop). Phase B
sub-classes proceed in smallest-cost-first order.

**Phase B1 closed (2026-05-11).** Three-layer fix: universal
U+FFFD collapse at chunk-creation chokepoint
(`_collapse_replacement_chars` in
`mmrag_v2.schema.ingestion_schema`); producer-site sanitizer
retained in `_sanitize_toc_index_text` (cleans text before
dense-index splitting); two-site BatchProcessor exemption for
`hybrid_chunker_pageskip*` chunks as defense-in-depth for
non-U+FFFD signatures. The sanitizer was widened from TOC-only
to corpus-wide on 2026-05-11 after architectural review flagged
the narrower scope as borderline overfit. Reconverts of Cronin,
Nagasubramanian, Sekar, Chaubal: 62 missing pages → 7 (89 %
reduction; full TOC class closed). All four now report
`AUDIT_PASS` + `UNIVERSAL_PASS` + `LOCALIZED_CORRUPTION = 0` with
HEADING ≥ 99 %. Residual missing pages (Nagasubramanian p2,
Sekar p2/p159/p228/p247, Chaubal p4/p11) are sub-class C/A and
fold into B3/B4 scope. 13 regression tests in
`tests/test_corruption_quarantine_toc_exemption.py` + 1 updated
in `tests/test_finalization_bridge.py`. Test suite: **750 passed,
14 skipped, 0 failed** (was 736 at Phase 4 close, +14 net new).

**Phase B2 closed (2026-05-11).** Gate-side fix:
`scripts/qa_full_conversion.py:_is_intentionally_blank_text`
recognizes the "This page intentionally left blank" boilerplate
(with a 120-char structural length cap to reject false positives).
Greenhouse_Design: p3/p11/p23 now classified as
MISSING_PAGES_BLANK (INFO/acceptable). 10 regression tests in
`tests/test_qa_intentionally_blank_pages.py`. Test suite: 760
passed, 14 skipped, 0 failed (+10 net new vs B1 close).

**Phase B4.a closed (2026-05-11).** Two-rule gate-side detector in
`scripts/qa_full_conversion.py`: `_page_render_is_near_blank` (mean>245
AND std<20 AND text<200) plus `_page_is_no_text_image_only_placeholder`
(text_len==0 AND images>=1 AND mean>250). Catches publisher-template
placeholder pages with 0 false positives on 15-page real-content
sample. **Python_Distilled: 697 → 4 missing (694 reclassified as
MISSING_PAGES_BLANK INFO)**, Devlin p2/p264 closed, Chaubal p4
closed. 17 regression tests in `tests/test_qa_near_blank_render.py`.
Test suite: **787 passed**, 14 skipped, 0 failed.

**Phase H closed Cronin (2026-05-11).** Targeted re-enrichment of
4 docs (Cronin / Nagasubramanian / Sekar / Chaubal — the B1+B3
reconvert outputs with pending image chunks). 170 image chunks ×
~3 s/call ≈ 8 min. 157 enriched + 13 hard_fallback (7.6 %; F4
sentinel applies). Result: **Cronin → QA_PASS** (full strict gate
green); Nagasubramanian / Sekar → QA_WARN (was FAIL); Chaubal stays
QA_FAIL on its 1 remaining missing page (p11, sub-class A/B4
residual).

**Phase D iconography lane closed (2026-05-11).**
`src/mmrag_v2/vision/asset_complexity.py` adds a `tiny bbox`
iconography lane (bbox <1 % of page → `simple` complexity
regardless of file size). Hybrid_electric_vehicles "Logo icon." now
correctly classifies as simple-asset-acceptable description, flipping
the doc from QA_FAIL → QA_WARN. Corpus probe: 216/4031 image chunks
have bbox<1 %; all 14 with short descriptions are legitimate
icon-class. AIOS p8 "Bar chart.0 to 1.0." remains FAIL (5 % bbox,
genuinely complex; retry-harness issue under investigation).
2 regression tests in `tests/test_asset_complexity.py`.

**Phase G advisory-warning allowance closed (2026-05-11).**
`scripts/qa_full_conversion.py` now emits
`QA_PASS_WITH_ADVISORIES` (a documented PASS variant, parallel to
the SCAN0013 `GATE_PASS [form: ...]` pattern) when all WARN issues
are in the documented advisory set: `ASSET_TINY`,
`PAGE_COUNT_UNKNOWN`, `SCRIPT_ADVISORY_FAIL` (unconditional);
`VISION_HARD_FALLBACK_RATE` (conditional: every hard_fallback must
carry the F4 sentinel `complex_asset_short_response_after_retry`).
Corpus probe confirms 100 % F4 coverage on the 5 docs with that
WARN (Jungjun, Kimothi, Bourne, Nagasubramanian, Sekar — total
25 hard_fallback chunks, all F4). 11 regression tests in
`tests/test_qa_advisory_promotion.py`. Allowance documented in
`docs/QUALITY_GATES.md` "Advisory Warning Classes".

**Cumulative strict-gate state after B1+B2+B3+B4.a+D+G+H
(2026-05-11):**

| State | PASS | WARN | FAIL | Tests |
|---|---:|---:|---:|---:|
| 2026-05-11 BEFORE (first full-corpus run) | 9 | 8 | 17 | 736 |
| After all current Phase B/D/G/H work | **24** | **0** | **10** | **804** |
| Net delta | **+15** | **−8** | **−7** | **+68** |

PASS=24 breaks down as 11 pure `QA_PASS` + 13 `QA_PASS_WITH_ADVISORIES`.
Goal 1 (v2.9.0-rc1) requires 32/34 PASS-class; **8 more docs need
closure** to reach the RC target. The 10 remaining FAILs:

| Class | Docs | Path |
|---|---|---|
| Signed v2.10 deferral | Firearms, KI_En | Out of scope ✓ |
| In-flight Phase E reconvert | Combat | Producer-side blank-asset filter widened to std<10; reconvert running |
| Phase D residual | AIOS p8 | Retry-harness bug suspected; deferral candidate |
| Phase C | Devlin | HEADING 72% (HybridChunker-path); deferral candidate parallel to Firearms |
| B3.b cross-page-split (deferred) | Python_Cookbook (4 pages) | v2.10 sign-off needed |
| B3.b + B4.b mixed | Python_Distilled (4 pages: 1 absorbed, 3 image-only) | v2.10 sign-off needed |
| New: TextIntegrityScout sensitivity (deferred) | Fluent_Python (6 pages) | v2.10 sign-off needed |
| Chaubal p11 + Earthship p109 | 2 pages real image content | B4.b small surface fix |

**Phase B3 closed with 3 signed v2.10 deferrals (2026-05-11).**
Three sub-classes:
- **B3.a (closed)**: `_emit_section_header_only_page_chunks` in
  `src/mmrag_v2/processor.py` emits a chunk for pages whose only
  Docling items are `section_header` / `title` (chapter dividers,
  title pages). Reconverts: Devlin p170 ✓; Nagasubramanian p2 ✓;
  Sekar p2/p159/p228/p247 ✓. **MISSING_PAGES=[] now on
  Nagasubramanian and Sekar; AUDIT_PASS + UNIVERSAL_PASS on
  both.**
- **B3.b (deferred to v2.10 as
  `CROSS_PAGE_SPLIT_PAGE_ATTRIBUTION`)**: Bourne p209 URL/citation
  list and Ayeva p4 dedication are absorbed into adjacent-page
  chunks via the v2.9 cross-page-split with wrong `page_number`
  attribution. Content present in corpus (just on the wrong
  page); no band-aid backfill (would conflict with Phase 1's
  `recovery_page_coverage` ban). Diagnostic:
  `docs/PHASE_B3_CROSS_PAGE_SPLIT_DIAGNOSTIC.md`.
- **B3.c (deferred to v2.10 as
  `LOW_RETRIEVAL_VALUE_PAGE_TAXONOMY`)**: Greenhouse p2 (book-title
  page) and Ayeva p4 (dedication, alternate classification).
  Corpus-wide FP test on a candidate dedication regex matched
  Greenhouse p25 — a legitimate Preface acknowledgment. Detector
  cannot be made tight enough without overfit risk; the
  Retrieval-Value Test (`docs/DECISIONS.md`) signed deferral
  pattern applies.

Added `docs/DECISIONS.md` "Retrieval-Value Test" governance:
features that do not improve retrieval are omitted rather than
backfilled. 10 new B3 regression tests in
`tests/test_section_header_only_page_emit.py`. Test suite:
**770 passed**, 14 skipped, 0 failed (+10 net new vs B2 close;
+34 cumulative vs Phase 4 close baseline of 736).

**`v2.9.0-rc1` ship state achieved 2026-05-11; tag landed 2026-05-12.**
AFTER snapshot at `docs/QUALITY_SNAPSHOT_2026-05-11_v2.9.0-rc1_after.md`.
Per `docs/DECISIONS.md` "v2.9.0-rc1 Signed Deferrals (2026-05-11
close-out)", the RC ships with 8 signed v2.10 deferrals (Firearms,
KI EPUB, Devlin, Python_Cookbook, Python_Distilled, Fluent_Python,
Chaubal, Earthship) covering all 8 remaining QA_FAIL rows. No
intermediate `v2.9.0` final tag is planned; the 8 deferrals carry
forward as v2.10 production-tag blockers. Strict gate state:
**26 PASS / 0 WARN / 8 FAIL** (12 `QA_PASS` + 14
`QA_PASS_WITH_ADVISORIES`).

Phases shipped in this cycle: B1 (TOC U+FFFD sanitizer + exemption),
B2 (intentionally-left-blank), B3 Step 2 (section-header-only page
emission), B4.a (render-based + zero-text image-only blank
detection), Phase D iconography lane, Phase E (Combat blank-asset +
gibberish-table), Phase G (QA_PASS_WITH_ADVISORIES variant), Phase H
(targeted re-enrichment).

**Phase I Qdrant rebuild complete (2026-05-12).**
`mmrag_v2_8` dropped (was 22,446 v2.8 points) and recreated from the
34 canonical post-recovery JSONLs via
`scripts/rebuild_mmrag_v2_8_for_rc1.py`. Final state: status=green,
`points_count=30,461` (exact match to source: 25,691 text + 4,379
image + 391 table), `indexed_vectors_count=30,213`, vector dim 4096
(llava). Wall time 10h15m on local Ollama. The 16 sister `*_v2`
per-doc collections from earlier experiments were dropped during
post-RC1 sanitization; `mmrag_v2_8` is now the sole collection.

**`v2.9.0-rc1` tag landed 2026-05-12** on commit `3e06d1b`, pushed to
GitHub. Post-tag hygiene commits (`51f4b37` sanitization, `e60f70f`
version bump, `10e94d1` + `4fa871c` + `665a76c` snapshot/search-tool
follow-ups) are on `main`.

### v2.9 in-flight fixes (committed)

- **Phase 2 closed (2026-05-08, verification-only)** — re-verified the
  four shipped v2.9 fixes under the strict gate after Phase 1 outputs.
  All five contract checks green: chunk_id uniqueness across 5,749
  chunks (0 dupes); HARRY refiner-suppressed (`refinement_applied=false`
  on all 651 text chunks) AND HARRY postprocessor acceptance fixture
  (2 passed, not skipped, with `HARRY_ACCEPTANCE_JSONL=...`); Combat
  refiner-engaged (109 refined chunks, 0 edit-ratio spam, smart-route
  fired); Ayeva CodeFormulaV2 lane (`code_indentation_fidelity=0.9693`,
  `[CODE-ENRICH] Enabled Docling code enrichment` per batch); Firearms
  route-flip verified (`profile_type=scanned/scanned_degraded`).
  Smoke baseline 11/11 `GATE_PASS + UNIVERSAL_PASS`. Phase 1
  invariants hold across every conversion (0 SIGALRM, 0
  `recovery_page_coverage`, 0 empty text chunks). See
  `docs/QUALITY_SNAPSHOT_2026-05-08_v2.9_phase2_after.md` and the
  Config Provenance block recorded there.

  *Carried forward to Phase 4* (split decision documented in the
  Phase 2 snapshot): Firearms HEADING coverage 72 % (target ≥ 0.80)
  and chunk-count drift 1690 → 2183 (+29 %). The Phase 4 commit
  `3fbce7a` route-flip mechanism succeeded, but the OCR/shadow lane
  on the new profile emits more chunks per page than the v2.8
  baseline; this is fix work for Phase 4 alongside Combat p66 /
  Adedeji p301 / KI EPUB.
- **Phase 1 closed (2026-05-07, commit `df91061`)** — dense-index page
  router + empty-chunk safety net. `_classify_dense_index_pages`
  trusts Docling's `document_index` label and routes those pages
  around HybridChunker via `MmragChunkingSerializerProvider(skip_pages=...)`;
  a dedicated grid-traversal emitter with two-layer dedup (byte-equal
  cell collapse + entry-boundary regex split) produces
  `extraction_method="hybrid_chunker_pageskip"` chunks with no
  `recovery_page_coverage` and no SIGALRM fires. Three layered guards
  drop empty text chunks (oversize-breaker, finalize stage,
  JSONL-write loop) so the strict-gate `empty_text_chunks` invariant
  holds under full-document conversion. Full Kimothi (258 pages)
  reports `AUDIT_PASS / UNIVERSAL_PASS / HYGIENE_PASS`; Ayeva back-
  index probe reports per-page chars 76–105 % of source PDF text
  layer (closes the prior −30 % token-variance regression). Test
  suite: **628 passed, 14 skipped**.
- **chunk_id collision fix** — per-document monotonic ``position``
  hashed into the chunk-id seed; v2.8's 427 within-file dupes
  collapse to zero on a fresh broad reconversion.
- **Refiner smart-routing** — the config-default refiner only
  auto-enables when the diagnostic engine reports
  ``has_encoding_corruption=True``. ``--no-refiner`` no longer needed
  in the Phase 5 conversion runner.
- **Cross-page DocChunk page-coverage split** — Docling's
  HybridChunker emits multi-page chunks; the chunker now emits one
  IngestionChunk per source page so chapter-intro pages aren't lost.
- **Mid-sentence merger and near-duplicate filter are now
  page-scoped** — they no longer reattribute one page's content to
  another or drop legitimate cross-page split copies.
- **CorruptionInterceptor extended to TABLE modality** — Combat p66's
  squadron-roster table is now subject to the same patch+quarantine
  path as text. Long em-dash and ``CS`` filler runs are detected
  alongside CIDFont/uniXXXX/replacement-char patterns.
- **FULL-PAGE-GUARD defers full-page assets when no
  conversion-time VLM** — three sites changed from discard → defer
  with ``vision_status='pending'``. Combat p4 (and similar pages
  whose only content is a full-page image) now produces a chunk that
  Phase 5b enrichment can fill in.
- **Phase 5b enrichment script writes canonical ``chunk.content``** —
  not just ``metadata.visual_description``. Earlier v2.9 attempts
  reported ``image_placeholder_ratio=1.0`` because the canonical
  field was never updated.
- **``scripts/qa_full_conversion.py`` strict gate** — bundles
  audit + universal + hygiene + semantic_fidelity plus deterministic
  page-coverage / dup-excess / corruption / image-quality checks.
  Documented in ``docs/TESTING.md`` as the v2.9 acceptance bar.

### Open work for v2.9

- **~~MISSING_PAGES on TOC / index pages~~** — closed by Phase 1
  (commit `df91061`, 2026-05-07). Broad-doc re-verification across
  the 18 originally-failing docs is folded into Phase 5.
- **~~Phase 2: re-verify shipped fixes under strict gate~~** — closed
  2026-05-08 (verification only). See
  `docs/QUALITY_SNAPSHOT_2026-05-08_v2.9_phase2_after.md`.
- **Phase 3 Steps 1-3 implemented (2026-05-09).** Source Sanctity
  validator hardened across 3 commits surfacing 13 leak classes on
  real qwen3-vl-plus output (`c23d3f6`, `a879e85`, `f224aad`); 604
  image chunks across 3 docs (Hao 252, Adedeji 128, PCWorld 224)
  enriched cleanly at 0 % hard-fallback rate. Asset-complexity
  classifier shipped in `src/mmrag_v2/vision/asset_complexity.py`
  with 12 unit tests. Gate calibration shipped on both
  `qa_full_conversion.py` (`_is_blankish_visual_description`) and
  `qa_semantic_fidelity.py` (`is_placeholder_image_or_table`) with
  the F4 hard-fallback exemption and complexity-aware short-
  description rule, plus 15 regression tests
  (`tests/test_qa_image_gate_calibration.py`). Step 1 baseline doc
  at `docs/QUALITY_SNAPSHOT_2026-05-09_v2.9_phase3_vlm_baseline.md`.
  pytest 645 → 672 passing (+27). **A1 follow-up implemented
  2026-05-09:** `detect_text_reading()` now delegates through a
  module-scope pattern table with `_detect_first_match()` diagnostics
  for the Step 4 retry harness; behavior diff is identical across the
  13 documented leak fixtures and the 604-description Phase 3 corpus,
  benchmark delta +1.3 %, pytest `678 passed, 14 skipped`.
- **Phase 3 Step 4 retry harness shipped (2026-05-09, `649c952`).**
  VLM detail-retry on short-on-complex chunks. 5 of 10 documented
  targets resolved by retry (Hao p35-1, p139, p355, p364; Adedeji
  p227); 5 hard_fallback with `complex_asset_short_response_after_retry`
  sentinel (F4-exempt). Strict gate `IMAGE_DESCRIPTION_UNUSABLE = 0`
  across all 3 enriched JSONLs. Snapshot updated at
  `docs/QUALITY_SNAPSHOT_2026-05-09_v2.9_phase3_vlm_baseline.md`
  §9. Phase 3 closed.
- **Phase 4: localized strict-gate hard failures — closed with signed
  deferrals (2026-05-10).** Plan and closure
  evidence: `docs/archive/PLAN_V2.9__PHASE4.md` and
  `docs/QUALITY_SNAPSHOT_2026-05-09_v2.9_phase4_after.md`.
  - Step 1 — `qa_full_conversion.py --source-pdf` documented as
    canonical strict-gate command (`611805d`).
  - Step 2 — Adedeji p298-316 source-PDF back-index fallback
    (`afbbaa6`); 0 TABLE_CORRUPTION, 0 LOCALIZED_CORRUPTION at
    full 320-page scale.
  - Step 3 — Combat p66 chunk-level corruption filter at finalize
    boundary (`6719460`); 0 false positives across 3,709 chunks.
  - Step 4 (fix) — cross-batch heading carry-forward (`b429cb5`);
    correct in unit tests + small probe; doesn't apply to Firearms
    (OCR-routed). The Path A profile-scoped HEADING gate relaxation
    (`5e58e6e`) was overfit threshold tuning and was reverted in
    `cbd7fb4`. Firearms HEADING continues to FAIL the strict
    `>= 0.80` gate. **DEFERRED to v2.10 as
    `OCR_PATH_HEADING_PROPAGATION` — user sign-off recorded
    2026-05-10.**
  - Step 5 — Adedeji `code_indentation_fidelity` 0.886 → 0.9032
    (cascade win from Step 2; no separate fix needed).
  - Step 6 — KI EPUB structural failures (no pagination, no bbox,
    heavy dedup excess). **DEFERRED to v2.10 as
    `KI_EPUB_EXTRACTION_LANE_REWRITE` — user sign-off recorded
    2026-05-10.**
  - Tests: 736 passed, 14 skipped (was 685 at Phase 3 close,
    +51 net new across Phase 4).
- **Signed Phase 4 deferrals (allowed only for `v2.9.0-rc1`, still
  block final `v2.9.0`).**
  1. `OCR_PATH_HEADING_PROPAGATION` — Firearms HEADING coverage
     0.722 vs 0.80 floor. Defect: OCR/element-by-element path
     doesn't promote Docling section_header items into
     `ContextStateV2.hierarchy_stack` (probe data in
     `docs/archive/PLAN_V2.9__PHASE4.md` Step 4). User signed off
     v2.10 deferral on 2026-05-10.
  2. `KI_EPUB_EXTRACTION_LANE_REWRITE` — EPUB lane structural
     gaps (acceptance baseline in `docs/archive/PLAN_V2.9__PHASE4.md`
     Step 6). User signed off v2.10 deferral on 2026-05-10.
- **Phase 5a/5b broad corpus run (2026-05-10/11) — validated-local.**
  A Python-native MPS-safe runner (`scripts/convert_books_v29.py`)
  replaced the legacy shell runner for Phase 5. Fresh reconversion now
  exists for **34/34** canonical corpus docs; the long-tail documents
  (`Bourne_RAG_2024`, `Raieli_AI_Agents`, `Hao_ML_Platform`,
  `Greenhouse_Design`) completed under the 10,800-second timeout.
  The canonical corpus has **30,356 chunks / 30,356 unique chunk_ids**.
  Phase 5b cloud `qwen3-vl-plus` enrichment processed 4,382 image
  chunks; the env-gated acceptance test passed:
  `RUN_V29_VLM_ACCEPTANCE=1 pytest tests/test_v29_image_enrichment_acceptance.py -q`
  → `1 passed`.
- **Phase 5c Qdrant refresh — blocked.** `localhost:6333` is not
  reachable and Docker socket access is denied by the sandbox. The
  required escalation to inspect/start the Qdrant container was
  rejected by the platform usage limit, so `mmrag_v2_8` has **not**
  been dropped or recreated. Qdrant remains the next blocker before
  the RC AFTER snapshot and `v2.9.0-rc1` tag.
- **Phase 5 runner update (2026-05-10).** Use
  `conda run -n mmrag-v2 python scripts/convert_books_v29.py`.
  `scripts/convert_books.sh` is disabled for Phase 5 because nested
  shell execution makes torch report `mps=False` on Apple Silicon,
  causing Docling to fall back to the stalled CPU path. The runner now
  kills the full conversion process group on timeout so blocked-doc
  retries fail cleanly, and it uses per-target lock files to prevent
  two conversions from writing the same output directory concurrently.

## Active Baseline

- **`docs/QUALITY_SNAPSHOT_2026-05-11_v2.9.0-rc1_after.md`** (current
  v2.9 ship state — 26/34 PASS / 0 WARN / 8 FAIL; tag `v2.9.0-rc1`
  on commit `3e06d1b`. No intermediate `v2.9.0` final tag is planned;
  the 8 signed deferrals carry forward as v2.10 production-tag
  blockers).
- **`docs/QUALITY_SNAPSHOT_2026-05-11_v2.9_strict_gate_full_corpus.md`**
  (BEFORE state for the RC1 cycle — 9 PASS / 8 WARN / 17 FAIL on the
  first full-corpus strict-gate run).
- **`docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md`** (v2.8.0 SHIPPED
  reference baseline; 30/34 canonical PASS under the v2.8-era
  audit-only gate).
- `docs/QUALITY_SNAPSHOT_2026-05-09_v2.9_phase4_after.md` (Phase 4
  close state with 2 signed deferrals; full-corpus strict gate not
  yet run at that point; superseded by 2026-05-11 BEFORE).
- `docs/archive/quality_snapshots/v2.9_in_progress/` —
  v2.9-in-progress snapshots from 2026-05-06 / 05-08 / 05-09 phase3 /
  05-10 phase5 attempt (kept for execution-history archaeology;
  not current-state docs).
- `docs/archive/quality_snapshots/QUALITY_SNAPSHOT_2026-05-03.md`
  (v2.8 Phase 0 BEFORE state, archived 2026-05-12).

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

**Test suite:** 806 passed, 14 skipped, 0 failed (was 736 at Phase 4
close; +70 net regression tests across the RC1 cycle phases).

### Open work — v2.10 production-tag blockers

Per `docs/DECISIONS.md` "v2.9.0-rc1 Signed Deferrals (2026-05-11
close-out)", the 8 deferrals listed below carry forward as v2.10
production-tag blockers. No intermediate `v2.9.0` final tag is planned.

1. `OCR_PATH_HEADING_PROPAGATION` — Firearms.
2. `KI_EPUB_EXTRACTION_LANE_REWRITE` — KI EPUB.
3. `HYBRID_CHUNKER_HEADING_PROPAGATION` — Devlin.
4. `CROSS_PAGE_SPLIT_PAGE_ATTRIBUTION` — Python_Cookbook + part of Python_Distilled.
5. `B4B_FULL_DOC_PICTURE_DEDUP` — Earthship + part of Python_Distilled.
6. `TEXT_INTEGRITY_SCOUT_FULL_DOC_SENSITIVITY` — Fluent_Python.
7. `TEXT_LABEL_TOC_DENSE_INDEX_ROUTER_MISS` — Chaubal p11.
8. (No 8th class — items 4 and 5 each cover two docs; the deferral
   list has 8 *rows* across 7 named classes.)

v2.10 housekeeping (non-blocking; see snapshot §9):
- Devlin re-ingest into `mmrag_v2_8` so payload metadata matches the
  post-catch-up JSONL (vectors already correct).
- Re-tune `scripts/search_qdrant.py` `--model` default and `MIN_SCORE`
  floor once v2.10 rebuilds the collection with a (possibly different)
  embedder.
- Move the hard-coded Dashscope API key out of
  [`scripts/search_qdrant.py:40`](../scripts/search_qdrant.py#L40)
  into an env var and rotate the leaked key on the provider side.

### Already-known followups (not v2.9 scope)

Carry forward to v2.10 planning:

- **Local NuMarkdown-8B-Thinking-mlx-8bits VLM lane.** Cloud-only
  for v2.9 enrichment; re-evaluate when network reachability returns.
- **Remote CodeFormulaV2 inference.** Docling 2.86 still doesn't
  expose `RemoteCodeFormulaOptions` / `ApiCodeFormulaOptions`. v2.9
  uses client-local CPU inference (~27 s/page on Apple Silicon).
- **HybridChunker per-item token guard.** Requires upstream Docling.
- **Broader UIR refactor** (canonical PdfConversionPlan →
  UniversalDocument → ElementProcessor → chunks per CLAUDE.md).
- **Magazine rendered-region-crop architecture** for composite layouts.

## Active Engineering Direction

`v2.9.0-rc1` has shipped. The active engineering task is **drafting
`docs/PLAN_V2.10.md`** from the seed prompt at
[`docs/PLAN_V2.10_DRAFT_PROMPT.md`](PLAN_V2.10_DRAFT_PROMPT.md). The
plan must structure the close of the 8 deferrals (above), the v2.10
housekeeping items, and the carry-forward followups, with explicit
acceptance baselines per item and a strict-gate target state.

PCWorld VLM evidence from the RC1 cycle remains valid: raw
text-reading detections 36.5 % → 22.2 %, zero measured Combat-style
hallucinations, blind-set 87.5 % final-valid. See
`tests/fixtures/blind_set_manifest.json`.

## Recently Completed

Reverse-chronological. Each entry's evidence files are tracked. The `[Folded into 2.8.0]` items still appear in chronological CHANGELOG entries but the consolidated v2.8 closure is the canonical artifact.

- **PLAN_V2.9 Phase 1 (TOC/index page-loss closure):** `complete` (2026-05-07, commit `df91061`). Dense-index page router via Docling `document_index` label fast path + `MmragChunkingSerializerProvider(skip_pages=...)`; dedicated grid-traversal emitter with two-layer dedup; three layered empty-text-chunk guards (oversize-breaker, finalize stage, JSONL-write loop). Full Kimothi (258 pages) reports `AUDIT_PASS / UNIVERSAL_PASS / HYGIENE_PASS`; Ayeva back-index probe per-page chars 76–105 % of source PDF text (closes prior −30 % token variance). Test suite **628 passed, 14 skipped, 0 failed**. Static `recovery_page_coverage` guard passes. SIGALRM did not fire on any tested document.
- **PLAN_V2.8 (production gaps + broad reconversion + Qdrant re-ingestion):** SHIPPED 2026-05-04. 7 commits on main `5b0e13d → 645ab2b`. Test suite **596 passed, 2 skipped, 0 failed** (at v2.8 ship; current main is 628). Smoke **11/11 GATE_PASS + 11/11 UNIVERSAL_PASS**. Broad reconversion 34/34 PDF/EPUB exit=0. `mmrag_v2_8` Qdrant collection: 22,137 / 22,160 unique embeddable chunks. See `docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md` and `CHANGELOG.md` `[2.8.0]`.
- Post-Docling Sanity Pass + `digital_literature` profile (folded into v2.8): `complete` (2026-05-03, commits `3bdbe0f`, `2f51816`, `379a733`). Reading-order y-sort, drop-cap promotion, label-leak filter, OCR gating, `digital_literature` profile + scorer + strategy.
- Contextual Retrieval (Anthropic approach): `complete` (2026-05-01). Embed-time `build_contextualized_text(...)` with breadcrumb + heading + neighbor context, AGENT-CONTEXTUAL-01..07 invariants, AST-level drift guard, byte-stable `--no-contextual` rollback flag.
- Refactor Boundary Closeout: `complete` (2026-05-01). Removed `BatchProcessor.set_intelligence_metadata` deprecated `[V2.8-COMPAT]` API; added typed-policy round-trip drift insurance test.
- Milestone 2 — Plan Control Plane: `complete` (2026-05-01). `PdfConversionPlan` promoted to typed policy object.
- Milestone 1 — Stabilize Extraction: `complete` (2026-05-01). RAG Guide unblocked, per-element chunker guard. **⚠ Ayeva 0.93 reading from this milestone is from the older probe; v2.8 canonical reads 0.83 FAIL — see `docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md`.**
- Vision-Aided Front Matter / Domain Search Priority / Coordinate Audit: `complete` (2026-04-30). See `docs/archive/quality_snapshots/QUALITY_SNAPSHOT_2026-04-30.md` (banner-annotated as superseded for any specific metric drift; the architectural changes are still active).
- Dependency metadata: Docling exact-pinned to `2.86.0` in `pyproject.toml`; engine version bumped 2.7.0 → 2.8.0 in v2.8 release commit `645ab2b`.

## Immediate Next Work

`docs/PLAN_V2.9.md` is the active plan; the draft prompt that produced
it has been archived (`docs/archive/PLAN_V2.9_DRAFT_PROMPT.md`).

Phase status (per Plan §3 sequence):

| Phase | Scope | Status |
|---|---|---|
| Phase 0 | Establish strict-gate baseline | `complete` |
| Phase 1 | TOC/index page-loss closure | `complete` (2026-05-07, `df91061`) |
| Phase 2 | Re-verify shipped fixes under strict gate | `complete` (2026-05-08, verification only — see `QUALITY_SNAPSHOT_2026-05-08_v2.9_phase2_after.md`) |
| Phase 3 | Resolve `IMAGE_DESCRIPTION_UNUSABLE` | `complete` (2026-05-09, commits `649c952` + `51e897b`) |
| Phase 4 | Localized strict-gate hard failures (Combat p66, Adedeji p301, KI EPUB, Firearms HEADING + chunk-count drift carried from Phase 2) | `closed with signed deferrals` (2026-05-10) |
| Phase 5 | Broad reconversion + Qdrant refresh + RC AFTER snapshot | `blocked in 5c` — 34/34 fresh outputs and Phase 5b VLM acceptance pass; Qdrant service unavailable |

## Must-Respect Constraints

- Python 3.10 only.
- Batch size must stay at or below 10 pages.
- Do not use `--profile-override` for acceptance runs.
- Do not add filename-specific or document-specific quality rules.
- OCR handles text; VLMs describe visuals only.
- BBoxes must remain normalized integer `[0,1000]`.
- Acceptance requires `GATE_PASS` plus `UNIVERSAL_PASS` across the smoke matrix.
