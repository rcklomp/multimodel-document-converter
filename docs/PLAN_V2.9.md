# Plan: v2.9 — Strict-Gate Full-Corpus Recovery

**Status:** `v2.9.0-rc1` CLOSED 2026-05-12 (commit `3e06d1b`, tag `v2.9.0-rc1` local). Final `v2.9.0` production tag remains blocked by 8 signed v2.10 deferrals — see `docs/DECISIONS.md` "v2.9.0-rc1 Signed Deferrals (2026-05-11 close-out)". This plan is RETAINED as the execution history for the v2.9.0-rc1 cycle; v2.10 work will get a separate `docs/PLAN_V2.10.md`.
**Owner:** ingestion pipeline
**Predecessor:** `docs/archive/PLAN_V2.9_2026-05-06_strict_gate_recovery.md` (previous active plan, archived 2026-05-11 after first full-corpus strict-gate run revealed a wider failure surface than the previous plan anticipated)
**BEFORE state:** `docs/QUALITY_SNAPSHOT_2026-05-11_v2.9_strict_gate_full_corpus.md` (9 PASS / 8 WARN / 17 FAIL)
**AFTER state:** `docs/QUALITY_SNAPSHOT_2026-05-11_v2.9.0-rc1_after.md` (26 PASS / 0 WARN / 8 FAIL — all 8 signed v2.10 deferrals)
**Related:** `docs/PROJECT_STATUS.md`, `docs/QUALITY_GATES.md`, `docs/DECISIONS.md`, `docs/ARCHITECTURE.md`, `docs/AGENT_GOVERNANCE.md`, `AGENTS.md`, `CHANGELOG.md`

---

## 1. Why this plan exists

The previous v2.9 plan (`docs/archive/PLAN_V2.9_2026-05-06_strict_gate_recovery.md`) drove the corpus from the 2026-05-06 strict-gate baseline (5 PASS / 3 WARN / 26 FAIL) through Phase 1 (TOC/index page-loss closure), Phase 2 (shipped-fix re-verification on 6 docs), Phase 3 (image-description policy and retry harness), Phase 4 (localized hard failures with two signed v2.10 deferrals), and Phase 5a/5b (broad reconversion + enrichment of the canonical 34).

Phase 5a produced 34/34 fresh JSONLs (0 chunk_id duplicates corpus-wide). Phase 5b enrichment resolved 4,269 / 4,382 image chunks to `vision_status="complete"` (97.42 %); 113 (2.58 %) ended in `vision_status="hard_fallback"`. The Phase 5b run was scarred by a documented parallel-process race on `Combat_Aircraft_August_2025.ingestion.jsonl.v29tmp` (two concurrent enrichment processes wrote to the same tmp path); the race was contained, Combat was re-converted and re-enriched cleanly, and no other doc was affected.

The first full 34-doc strict-gate run under `scripts/qa_full_conversion.py --source-pdf --allow-warnings` then reported **9 PASS / 8 WARN / 17 FAIL**. Only 2 of the 17 FAILs are signed Phase 4 deferrals (Firearms `OCR_PATH_HEADING_PROPAGATION`, KI EPUB `KI_EPUB_EXTRACTION_LANE_REWRITE`). The other **15 unallowed rows** are failure classes that the Phase 4 close snapshot never measured at scale.

This plan exists to close those 15 unallowed rows under the unchanged strict gate. `v2.9.0-rc1` remains the next attainable tag and the final `v2.9.0` tag continues to require the signed Phase 4 deferrals to be repaired (out of v2.9 scope).

### What is preserved from the previous plan

- Phase 1 dense-index closure (commit `df91061`) — preserved. Its narrow-probe validation is still correct for the docs it tested; this plan extends coverage to the rest of the corpus.
- Phase 2 verification of chunk_id collision fix (`eae27e8`), refiner smart-routing (`b1b2f3f`), Ayeva profile route (`51f0884`), Firearms HARD REJECT (`3fbce7a`) — preserved. The corpus-wide chunk_id contract is verified: 0 within-file duplicates across all 34 canonical JSONLs.
- Phase 3 source-sanctity validator hardening (`c23d3f6`, `a879e85`, `f224aad`), asset-complexity classifier (`src/mmrag_v2/vision/asset_complexity.py`), gate calibration, F4 hard-fallback exemption — preserved. The complexity-aware short-description rule needs an extension (see §3 Phase D).
- Phase 4 Step 1 `--source-pdf` as canonical strict-gate command (`611805d`) — preserved.
- Phase 4 Step 2 Adedeji p298-316 back-index fallback (`afbbaa6`) — preserved.
- Phase 4 Step 3 Combat p66 chunk-level corruption filter (`6719460`) — preserved.
- Phase 4 Step 4 cross-batch heading carry-forward fix (`b429cb5`) — preserved; this plan investigates whether a second HybridChunker-path heading-propagation defect exists in Devlin (§3 Phase C).
- Phase 4 Step 5 Adedeji code_indentation_fidelity cascade (`a45b259`) — preserved.
- Signed v2.10 deferrals — preserved verbatim: Firearms `OCR_PATH_HEADING_PROPAGATION` and KI EPUB `KI_EPUB_EXTRACTION_LANE_REWRITE`.
- Phase 5a fresh JSONLs and Phase 5b enrichment — preserved as the working baseline. We do not throw out 4,269 cloud-VLM descriptions to start over.

### What this plan replaces

The previous plan's §3 phase narrative ("Phase 5 broad reconversion → enrichment → Qdrant → RC AFTER snapshot") is replaced by §3 below. The Phase 5 sequence cannot run as written because the corpus is not strict-gate clean. This plan inserts the failure-class repairs that have to happen before another Phase 5 attempt.

---

## 2. Goals & Non-Goals

### Goals (measurable from JSONL / strict-gate output / Qdrant counts)

1. `scripts/qa_full_conversion.py --source-pdf --allow-warnings output/<doc>/ingestion.jsonl` reports **`QA_PASS` for 32 of 34 canonical docs**. The only two allowed non-PASS rows are the unchanged signed Phase 4 deferrals: Firearms (HEADING < 0.80 from `OCR_PATH_HEADING_PROPAGATION`) and KI_En_ChatGPT_Praktische_Gids (`UNIVERSAL_FAIL` from `KI_EPUB_EXTRACTION_LANE_REWRITE`). `QA_WARN` is not a ship state for any other doc.
2. **MISSING_PAGES = 0** across the 10 currently-failing docs after this plan's reconversions. Visibly blank source pages must be classified by `--source-pdf` blank-page detection, not by manual flags.
3. **AUDIT_FAIL(HEADING) on Devlin_LLM_Agents resolves**. Either the HybridChunker-path carry-forward fix moves the metric for Devlin's specific shape, or Devlin's defect is shown to be a distinct second class (in which case the fix is in scope of this plan, not v2.10).
4. **AUDIT_FAIL(IMAGE) on Combat_Aircraft_August_2025 resolves**. The blank asset `figure_36 p27` (mean=253, std=7.2) is dropped at extraction time, not after the fact.
5. **IMAGE_DESCRIPTION_UNUSABLE on AIOS p8 and Hybrid_EV p1 resolves**. Either the existing Phase 3 retry harness is extended to cover these cases, or the gate's "useful description" rule is shown to be over-strict and is adjusted with explicit rationale and a regression test.
6. **SEMANTIC_FAIL hard FAIL on Greenhouse_Design resolves** (`code_indentation_fidelity >= 0.90`). The advisory cases (Kimothi 0.667, Raieli 0.884) remain WARN-class only because their `qa_semantic_fidelity.py` exit code does not propagate as a hard FAIL; this plan does not relax that contract.
7. **WARN classes on currently-WARN docs resolve to PASS** under the unchanged strict gate, OR the WARN class is promoted into an explicit allowed PASS variant in `docs/QUALITY_GATES.md` with rationale (per the previous plan's same provision). The candidates: `ASSET_TINY` (small-figure assets, 4 docs), `PAGE_COUNT_UNKNOWN` (EPUB), `VISION_HARD_FALLBACK_RATE > 5 %` (Jungjun 10.9 %, and the FAIL-class docs that also warn here).
8. **Within-file chunk_id duplicates across the 34 canonical JSONLs remain 0** after every reconvert in this plan. The corpus-wide invariant from `eae27e8` is non-negotiable.
9. **`pytest tests/ -q` reports 736 passed or higher with 0 failed** (current count from Phase 4 close), with any new test added by this plan included.
10. **`bash scripts/smoke_multiprofile.sh`: every row `GATE_PASS` + `UNIVERSAL_PASS`** (no waivers; form variants per `docs/QUALITY_GATES.md` "Form / Invoice Acceptance Class" remain valid).
11. **0 image points with `vision_status="pending"`** in any JSONL after the Phase H re-enrichment closure (see §3 Phase H). `vision_status="hard_fallback"` remains at or below 5 % per-doc except where the gate exempts (F4 sentinel).
12. **`mmrag_v2_8` Qdrant collection is dropped and recreated** from the post-recovery JSONLs once Goals 1-11 are met, with the embeddable chunk count matching the unique non-pending chunk count across the 34 canonical JSONLs.

### Non-Goals (deferred to v2.10 or later — unchanged from the previous plan)

- **Firearms OCR-path heading propagation** (`OCR_PATH_HEADING_PROPAGATION`). Signed deferral 2026-05-10.
- **KI EPUB extraction lane rewrite** (`KI_EPUB_EXTRACTION_LANE_REWRITE`). Signed deferral 2026-05-10.
- **Local NuMarkdown-8B-Thinking-mlx-8bits VLM lane**. Off-network, cloud-only for v2.9.
- **Remote CodeFormulaV2 inference target**. Docling 2.86 still does not expose `RemoteCodeFormulaOptions` / `ApiCodeFormulaOptions`. Trigger condition (>1 code-heavy reconversion/week) is not met; v2.10+.
- **Broader UIR refactor**. Canonical target per `CLAUDE.md` but not required for v2.9.
- **HybridChunker per-item token guard**. Requires upstream Docling.
- **EPUB lane redesign**. KI EPUB is the deferral; ChatGPT_Praktijk_handboek's `PAGE_COUNT_UNKNOWN` warn is the same lane's milder symptom and is in scope only via Goal 7's WARN-promotion route.

## 2b. Parallel-Site Audit (cross-cutting principle — unchanged)

Permanent project requirement since v2.8 §2b. Lesson learned 2026-05-03:
a single-site fix is suspect until parallel call sites in the
pipeline are audited. The four questions before each phase below:

1. Does the issue ALREADY have a fix elsewhere in the pipeline that
   the failing data simply hasn't been re-run through?
2. Does an existing fix have too narrow a gate (e.g., a label match
   on one Docling label when the bug surface uses another)?
3. Are there parallel boundaries that need the same change?
   (CLI `process` vs CLI `batch`; `BatchProcessor` vs
   `V2DocumentProcessor`; `engines/pdf_engine.py` vs
   `engines/docling_adapter.py`.)
4. Is there an upstream library config that already addresses the
   issue without custom code? ("Libraries first, custom code last".)

Each phase below answers all four before code is written.

## 2c. Cost-aware ordering (new for this plan)

Re-enrichment is paid in cloud-VLM tokens (`qwen3-vl-plus` on
DashScope). The Phase 5b run already consumed ~4,382 calls. To
avoid re-paying that:

- A fix that requires reconvert + re-enrich of N docs costs ≈ N × per-doc
  image-chunk count × ~3.5 s × DashScope rate. The enrichment script
  is idempotent: re-running on an already-complete JSONL skips
  every `vision_status="complete"` chunk (verified empirically in
  the Phase 5b race incident — the duplicate process correctly
  skipped all already-complete chunks).
- For fixes that change chunking/serialization, the post-fix JSONL
  has new chunk_ids and new image chunks need re-enrichment.
- For fixes that change only post-processing (e.g., dropping a
  blank asset, adjusting a gate threshold, post-hoc rewriting a
  visual_description), the existing enriched chunks survive.

Each phase below names which class it falls into and pays the
corresponding cost.

## 2d. Architectural constraints (review 2026-05-11)

Added after architectural review on 2026-05-11. These constraints
bind the implementation of the phases below:

1. **Phase A/B fix location.** If the MISSING_PAGES root cause is a
   page-numbering mismatch (off-by-N between JSONL `page_number`
   and source-PDF page index), the fix lands in `PdfConversionPlan`
   / `DoclingPdfAdapter` — i.e., at the extraction policy layer
   where the Docling page array is mapped into
   `UniversalDocument`. The fix must not be a post-hoc patch in
   the JSONL writer or in `BatchProcessor`'s emission loop.
   Reason: `AGENT-COORD-01` and the canonical pipeline order
   (`PdfConversionPlan` → adapter → `UniversalDocument` → chunks)
   require page-index authority to live in the extraction policy.
   A post-hoc rewrite would mask the same defect in any future
   non-PDF lane.

2. **Phase D pixel-area exemption (if used).** Any image-size
   exemption from the "useful description" rule must be a universal
   geometric rule that applies across every profile, not a
   profile-scoped or doc-scoped allowance. The rule must be
   expressible as a single threshold (e.g., `image_area < N
   pixels²`) that is independent of doc identity. The accompanying
   regression test must include at least one positive case (an
   AIOS/Hybrid_EV-shape small asset that is exempted) AND one
   negative case (a comparably-sized asset in a different profile
   that still requires a useful description). Reason: the
   2026-05-09 Phase 4 Step 4 Path A overfit (`5e58e6e` →
   `cbd7fb4`) is the cautionary precedent — gate thresholds
   reverse-engineered from one failing doc are forbidden.

3. **Phase G ASSET_TINY allowance.** If the < 1 KB asset warning
   is promoted to informational, the allowance is profile-scoped
   to `digital_literature` only (and any other profile where
   the chapter-divider-icon shape is documented). The promotion
   must ship with at least one regression test that confirms a
   sub-1KB asset in `technical_manual` / `digital_magazine` /
   `academic_journal` / any non-literature profile still produces
   the `ASSET_TINY` warning. Reason: `digital_literature` is the
   formal profile that captures the "small UI icons on chapter
   pages" shape; any other profile that emits a sub-1KB figure
   is signaling a different failure (e.g., a magazine ad icon, a
   journal logo extracted as a figure) and must continue to warn.

These three constraints are non-negotiable for `v2.9.0-rc1`. Any
proposed deviation requires a separate signed deferral parallel
to the Firearms / KI EPUB deferrals.

---

## 3. Phases

The current strict-gate failure surface decomposes into 9
work-streams. Phases A-G repair them. Phase H re-runs Phase 5b
on the reconverted subset and re-validates the full corpus. Phase I
runs the Qdrant rebuild and writes the RC AFTER snapshot.

| Phase | Scope | Docs affected | Reconvert? | Re-enrich? | Status |
|---|---|---|---|---|---|
| A | MISSING_PAGES root-cause diagnostic | (diagnostic; touches Python_Distilled and one front-cluster doc) | no | no | `complete` (2026-05-11, `docs/PHASE_A_MISSING_PAGES_DIAGNOSTIC.md`) |
| B1 | TOC corruption-quarantine over-fire | Cronin, Nagasubramanian, Sekar, Chaubal | yes (4 docs) | yes (those 4) | `complete` (2026-05-11): 62 → 7 missing pages across the 4 docs; AUDIT_PASS + UNIVERSAL_PASS on all four; LOCALIZED_CORRUPTION = 0; HEADING ≥ 99% |
| B2 | "Intentionally left blank" disclaimer pages | Greenhouse | no (gate-side fix) | no | `complete` (2026-05-11): p3/p11/p23 now classified as MISSING_PAGES_BLANK (INFO) by `_is_intentionally_blank_text` in qa_full_conversion.py; 10 regression tests |
| B3 | Section-header-only page emission + cross-page-split deferral + title/dedication deferral | Devlin p170, Nagasubramanian p2, Sekar p2/p159/p228/p247 (closed); Bourne p209, Ayeva p4, Greenhouse p2 (deferred to v2.10) | yes (Devlin/Nagasubramanian/Sekar) | yes (those 3) | `complete with 3 signed v2.10 deferrals` (2026-05-11): 8 pages closed; 3 deferred under Retrieval-Value Test + cross-page-split diagnostic |
| B4.a | Render-based blank-equivalent classification (gate-side) | Python_Distilled (615+79=694 of 697 closed), Devlin p2/p264, Chaubal p4, similar publisher-template placeholder pages corpus-wide | no (gate-side) | no | `complete` (2026-05-11): two-rule detector — `mean>245 AND std<20 AND text<200` plus stricter `text=0 AND images≥1 AND mean>250`; 17 regression tests; 0 FPs on 15-page real-content sample |
| B4.b | Real image-only content emission (image chunk for pages with substantive visual content) | Earthship p109, Python_Distilled p6/p686/p688/p913, Fluent ~6 pages, Python_Cookbook ~4 pages | depends on root cause | yes | `pending` (probe shows most remaining are cross-page-split absorption per B3.b, not true image-only-content; small real-content surface < ~5 pages) |
| C | Devlin HEADING coverage | Devlin_LLM_Agents | depends on root cause | depends | `pending` |
| D | IMAGE_DESCRIPTION_UNUSABLE policy | AIOS, Hybrid_EV | no | targeted retry only (2 chunks) | `pending` |
| E | Combat blank-asset extraction filter | Combat_Aircraft_August_2025 | no (post-hoc drop) or yes (if Docling-side filter) | no for post-hoc; yes if reconvert | `pending` |
| F | Greenhouse SEMANTIC_FAIL (code_indent 0.800) | Greenhouse_Design | rolled into Phase B reconvert | rolled into Phase B re-enrich | `pending` |
| G | WARN-class promotion / repair | HARRY, Form_betwistingsformulier, A_comprehensive, Hao, Jungjun, ChatGPT, Kimothi, Raieli | no (gate / post-process work) | no | `pending` |
| H | Targeted re-enrichment of Phase B / C outputs + full-corpus strict-gate re-validation | up to 12 docs re-enriched | — | yes (subset) | `pending` |
| I | Qdrant drop+recreate + RC AFTER snapshot + `v2.9.0-rc1` tag | all 34 | no | no | `pending` |

### Phase A — MISSING_PAGES root-cause diagnostic — `complete` 2026-05-11

> **Closure summary.** The MISSING_PAGES failure surface decomposes
> into four distinct sub-classes (A/B/C/D in the diagnostic note's
> terminology, mapped to Phase B sub-phases B4/B1/B3/B2 below in
> the plan execution order — smallest-cost-first). Full evidence
> in `docs/PHASE_A_MISSING_PAGES_DIAGNOSTIC.md`. Hypothesis 1
> (page-numbering mismatch) was ruled out via direct cross-check.
> The strict gate's blank-page detector is correct; no gate change
> is required. Phase B body below is updated to reflect the
> sub-class decomposition.

### Phase A — original diagnostic plan (historical)

**What:** The Phase 1 dense-index closure (`df91061`) was validated
on Kimothi + Ayeva back-index probes. The current snapshot shows
**Python_Distilled has 697 missing pages** out of ~880, which is
catastrophic and cannot be the same defect class as the
single-page Kimothi/Ayeva symptoms. Before any fix is written, this
phase determines which of the following is true:

1. **Page-numbering mismatch.** The JSONL's `page_number` metadata
   may be off-by-N relative to the source PDF's page index used by
   `qa_full_conversion.py`'s page-coverage check. If JSONL page 5
   actually corresponds to source-PDF page 12, the strict gate will
   correctly report 7 missing pages between them for every batch.
2. **Real chunk drop in a specific shape.** The chunks were
   produced but a downstream filter (e.g., empty-text-chunk guard,
   page-scoped dedup) dropped them silently.
3. **Conversion-time drop.** Docling never emitted chunks for those
   pages — they were filtered before HybridChunker received them.
4. **Profile-routing drop.** A profile-specific code path routes
   certain docs through a thinner lane that emits fewer chunks per
   page.

**Steps:**

1. Pick **Python_Distilled** as the worst case and **one
   front-cluster doc** (Cronin or Nagasubramanian, since they show
   the cleanest p2 + p5-p20 pattern) as the comparison case.
2. For each:
   - Dump `page_number` distribution from the JSONL:
     `python3 -c "import json, collections; ..."` (count chunks per
     page, find gaps, find duplicates).
   - Dump source-PDF visible-page count from
     `pdfplumber`/`fitz`: same page numbers, same range.
   - Cross-check: for 5 random missing pages reported by the strict
     gate, open the source PDF at that page index and confirm
     visible content (not blank, not separator).
   - If page numbers match: read 5 sample missing pages through
     `mmrag-v2 process ... --pages <range>` and observe whether
     Docling emits items, whether HybridChunker emits chunks, and
     whether BatchProcessor filters them.
3. Record the outcome as a 1-page evidence note (path:
   `docs/PHASE_A_MISSING_PAGES_DIAGNOSTIC.md`). The note must say
   exactly which of the 4 hypotheses is supported, with citations.

**Done when:** the diagnostic note exists, is checked into git, and
names the next phase (B) with the specific code lane to inspect.

**Risk:** Low. Read-only diagnostic, no production code changes,
no cloud cost.

**Estimated effort:** 1-2 h.

### Phase B — MISSING_PAGES fixes (4 sub-classes)

**What:** Implement the four sub-class fixes identified in Phase A
in priority order (smallest-cost-first). Each sub-phase is a
contained code change + a regression test fixture + an
acceptance-time reconvert. Sub-phases B2/B3/B4 share reconvert
budget where docs overlap. The reconvert at the end of Phase B
produces fresh JSONLs; the new image chunks are placeholder and
need Phase H re-enrichment.

#### B1 — TOC corruption-quarantine over-fire — `complete` 2026-05-11

> **Closure summary.** Three-layer fix: (1) universal U+FFFD
> collapse via `_collapse_replacement_chars` at the chunk-creation
> chokepoint (`mmrag_v2.schema.ingestion_schema.create_text_chunk`
> factory + `IngestionChunk.content` /
> `ChunkMetadata.refined_content` / `ChunkMetadata.visual_description`
> field-validators), (2) producer-site U+FFFD collapse in
> `_sanitize_toc_index_text` (still runs before chunk creation so
> downstream splitting sees clean text), and (3) two-site
> BatchProcessor exemption for `hybrid_chunker_pageskip*`
> (`_quarantine_corrupted_text_chunks` +
> `_drop_corrupted_chunks_before_metadata`) as defense-in-depth
> against non-U+FFFD corruption signatures in Phase 1 router
> output. The original narrow-scope sanitizer (TOC-only) was
> widened to corpus-wide on 2026-05-11 after architectural review
> flagged the implicit "TOC = cosmetic, body = real corruption"
> assertion as borderline overfit; U+FFFD is by definition an
> unrenderable glyph and cannot carry semantic content at any
> site. The BP quarantine remains the authority for CIDFont
> placeholders, em-dash runs, and C/S filler runs.
>
> The Phase A diagnostic predicted the surgical exemption would
> suffice. That prediction was incomplete: `qa_conversion_audit.py`
> and `qa_universal_invariants.py` have their own corruption
> detectors that do not honor the BatchProcessor exemption. The
> raw `�` runs in the dotted-leader regions tripped the
> strict gate's `LOCALIZED_CORRUPTION` and `UNIVERSAL_FAIL` checks
> even when the chunks survived the BatchProcessor. The sanitizer
> collapses `[\s]*�+[\s]*` runs to a single space at the producer
> site (`src/mmrag_v2/processor.py:_sanitize_toc_index_text`) so
> chunks land clean; the TOC's title + page number surface form
> ("About the Author xxix") survives intact and is retrievable.
>
> Reconvert results (4 docs):
>
> | Doc | Missing pages before | Missing pages after | Outcome |
> |---|---:|---:|---|
> | Cronin_GenAI_Models | 24 | **0** | All TOC missing closed; AUDIT_PASS + UNIVERSAL_PASS |
> | Nagasubramanian_Agentic_AI | 16 | **1** (p2) | TOC closed; remaining p2 is sub-class C (dedication) → B3 |
> | Sekar_MCP_Standard | 13 | **4** (p2, p159, p228, p247) | TOC closed; p2 is sub-class C, p159/p228/p247 likely sub-class A → B3/B4 |
> | Chaubal_PyTorch_Projects | 9 | **2** (p4, p11) | TOC closed; p4/p11 are sub-class C → B3 |
> | **Total** | **62** | **7** | 89 % reduction; 100 % of TOC class closed |
>
> Strict-gate dimensions on all four post-reconvert: `AUDIT_PASS`
> (HEADING ≥ 99 %), `UNIVERSAL_PASS` (irreparably_corrupt_chunks =
> 0), `LOCALIZED_CORRUPTION = 0`. Residual FAILs are `VISION_PENDING`
> + `IMAGE_DESCRIPTION_UNUSABLE` only — both Phase H scope.
>
> Tests: 11 cases in
> `tests/test_corruption_quarantine_toc_exemption.py` (TOC chunk
> preservation through both drop sites, Combat p66 body corruption
> still dropped, sanitizer collapses replacement-char runs, kill-
> switch behavior). Full suite: 742 passed, 14 skipped, 0 failed.
>
> **Operational note.** During the B1 reconvert, an orphaned runner
> process (PID 52998) from an earlier killed-but-incomplete kill
> attempt continued processing the same `--only` doc list,
> concurrently with the new sanitizer-fix runner (PID 56585). The
> orphan was identified via `ps aux | grep convert_books` and
> killed via process group (`kill -TERM -- -$PGID`). A stale
> conversion lock file
> (`output/.Sekar_MCP_Standard.convert.lock` containing PID 52998)
> blocked Sekar from converting and was manually removed. Lesson
> for B2/B3/B4 runners: always kill via process group, and check
> for stale `output/.*.convert.lock` files before launching a
> reconvert.
>
> Body retained for historical context (original B1 design plan
> below).

**Affected docs:** Cronin_GenAI_Models (24 missing front-TOC),
Nagasubramanian_Agentic_AI (16), Sekar_MCP_Standard (13),
Chaubal_PyTorch_Projects (9). ~62 pages total.

**Root cause (per Phase A diagnostic):** Phase 1's dense-index
router fires correctly and emits chunks for TOC pages. Phase 4
Step 3-era `_quarantine_corrupted_text_chunks`
(`src/mmrag_v2/batch_processor.py:829`) then drops them because
the chunks' content contains a regex match in
`has_encoding_artifacts` (`src/mmrag_v2/validators/corruption_interceptor.py:48`).
The match is most likely an em-dash run or a CIDFont placeholder
introduced by Docling's per-cell extraction of the TOC table on
this publisher template (2025-2026 GenAI books). The same regex
correctly drops Combat p66 corrupted-table chunks.

**Parallel-site audit:**

1. Existing fix? `patch_corrupted_chunks` at `batch_processor.py:3073`
   is supposed to repair corrupted content via OCR before quarantine
   runs. It does not repair the TOC-template signature; either the
   repair fails or the patched content still matches the regex.
2. Too-narrow gate? The quarantine's gate is "any TEXT chunk whose
   content matches `has_encoding_artifacts`". It does not exempt
   chunks whose `extraction_method` indicates they came from a
   known-safe lane (the Phase 1 dense-index router uses
   `extraction_method=hybrid_chunker_pageskip` /
   `hybrid_chunker_pageskip_source_pdf`).
3. Parallel boundaries: `_drop_corrupted_chunks_before_metadata`
   at `batch_processor.py:3192` is a separate finalize-stage drop
   path. Must verify it does not also drop TOC chunks.
4. Library config? None applicable; corruption-quarantine is a
   project construct.

**Fix design:** Empirically capture the exact substring match
that triggers the regex on Cronin TOC content (see B1 step 1
below). Two acceptable fix shapes, in order of preference:

- **Preferred (surgical):** Exempt chunks with
  `extraction_method in {"hybrid_chunker_pageskip",
  "hybrid_chunker_pageskip_source_pdf"}` from
  `_quarantine_corrupted_text_chunks`. Rationale: Phase 1's
  dense-index router emits these chunks specifically because the
  TOC page-skip path produces clean, structured text that the
  regular hybrid_chunker can't handle. Quarantining these chunks
  defeats the Phase 1 closure for any TOC that happens to share a
  visual signature with the corruption regex. The Phase 1 router
  is the quality gate; the corruption regex was designed for
  body-text corruption (Combat p66).
- **Alternative (deeper):** Extend `patch_corrupted_chunks` to
  recognize the publisher-template signature and rewrite it to
  clean text before quarantine runs. More correct but more
  invasive; defer unless the surgical fix has measurable false
  positives.

**Steps:**

1. Add temporary logging in `_quarantine_corrupted_text_chunks`
   that prints (chunk_id, extraction_method, first-200-chars) for
   each dropped chunk. Re-run Cronin p5-p10 probe. Confirm
   `extraction_method` is one of the Phase 1 router values.
   Identify the specific regex pattern triggering the drop. (No
   commit; this is local instrumentation.)
2. Remove the temporary logging. Write the surgical exemption in
   `_quarantine_corrupted_text_chunks`: skip TEXT chunks whose
   `extraction_method` starts with `hybrid_chunker_pageskip`.
3. Add a regression test at
   `tests/test_corruption_quarantine_toc_exemption.py` (new file)
   that:
   - asserts a `hybrid_chunker_pageskip`-tagged TOC chunk with
     embedded em-dash content is PRESERVED through quarantine;
   - asserts a body-text chunk (any non-pageskip extraction
     method) with the same em-dash content is DROPPED;
   - covers both `OCR_FAILURE_PATTERNS` and `CORRUPTION_PATTERNS`
     match shapes.
4. Run `pytest tests/ -q` — expect 736 passed + 1 new = 737.
5. Reconvert Cronin, Nagasubramanian, Sekar, Chaubal with
   `scripts/convert_books_v29.py --only
   Cronin_GenAI_Models,Nagasubramanian_Agentic_AI,Sekar_MCP_Standard,Chaubal_PyTorch_Projects
   --force --keep-going --append-log --timeout 7200`.
6. Validate each: 0 within-file chunk_id duplicates, 0 bad JSON,
   `MISSING_PAGES` reports only blank-source pages (or 0).

**Done when:** B1's 4 docs report `MISSING_PAGES=0` (non-blank).
Image chunks remain pending for Phase H.

**Risk:** Low. The exemption is one condition in one function.
The regression test ensures Combat p66 still passes.

**Estimated effort:** 1-2 h code + test. Reconvert wall time:
Cronin+Nagasubramanian+Sekar+Chaubal at observed Phase 5a rate ≈
50-90 minutes total (none are catastrophically large).

#### B2 — "Intentionally left blank" disclaimer pages — `complete` 2026-05-11

> **Closure summary.** Gate-side fix only: extended
> `scripts/qa_full_conversion.py:_read_blank_pages_in_source` to
> recognize "intentionally left blank" boilerplate pages as
> blank-equivalent (`_is_intentionally_blank_text`). The detector
> requires the literal four-word phrase plus a 120-char structural
> length cap to reject false positives (preface text that mentions
> blank pages). Producer pipeline unchanged — the boilerplate
> chunks were already dropped by short-content filtering, which is
> the correct producer behavior (retrieval over the boilerplate
> would pollute results). Greenhouse_Design strict-gate: 4
> MISSING_PAGES (p2, p3, p11, p23) → **3 MISSING_PAGES_BLANK (INFO,
> acceptable) on p3/p11/p23 + 1 MISSING_PAGES (FAIL on p2 only,
> sub-class C → B3 scope)**. 10 regression tests in
> `tests/test_qa_intentionally_blank_pages.py` (positive cases:
> doubled boilerplate, single line, uppercase, "is" variant, short
> garbage prefix; negative cases: preface discussing blank pages,
> empty text, chapter headings, boilerplate amid real content).
> Test suite: 760 passed, 14 skipped, 0 failed.
>
> Body retained for historical context.

**Affected docs:** Greenhouse_Design (3 pages: p3, p11, p23).

**Fix design:** Producer-side. Add a guard in `BatchProcessor` (or
the appropriate empty-text-chunk site identified during B3) that
detects the boilerplate phrase "intentionally left blank"
(case-insensitive substring on a stripped page-text region) and
emits one short text chunk with the boilerplate as content. This
keeps the chunk visible in retrieval and satisfies the strict
gate's page-coverage check.

**Steps:**

1. Locate the page-level filter site where currently-zero chunks
   are emitted. This is where the page text is examined and a
   "no chunks" decision is made.
2. Add a positive condition: if the stripped page text contains
   the case-insensitive substring "intentionally left blank",
   emit one text chunk with `content="[INTENTIONALLY_BLANK]"`
   (sentinel) or the verbatim boilerplate (decision: verbatim,
   preserves the doc's actual surface form for retrieval).
3. Regression test: a single page with boilerplate produces
   exactly 1 chunk. A normal page with no boilerplate still
   produces its normal chunks.
4. Reconvert Greenhouse (rolled into B3/B4's Greenhouse reconvert).

**Done when:** Greenhouse p3, p11, p23 each produce 1 chunk.

**Risk:** Low. Stand-alone boilerplate matcher.

**Estimated effort:** 1 h.

#### B3 — Section-header-only page emission + cross-page-split + title-page deferrals — `complete with 3 signed v2.10 deferrals` 2026-05-11

> **Closure summary.** The framing "short-text page filter overshoots"
> from the original B3 design was incorrect. Probing revealed three
> distinct sub-classes within the residual MISSING_PAGES surface:
>
> - **B3.a — Section-header-only pages** (closed by Step 2):
>   HybridChunker treats `section_header` Docling items as heading
>   metadata, never emitting them as text chunks. On pages whose
>   ONLY content is one or more `section_header` items (chapter
>   dividers, title pages, part openers) this produced 0 chunks.
>   Added `_emit_section_header_only_page_chunks` in
>   `src/mmrag_v2/processor.py` (extraction_method
>   `hybrid_chunker_section_header_page`, ChunkType.HEADING,
>   `search_priority="high"`). 10 regression tests in
>   `tests/test_section_header_only_page_emit.py` (positive: single
>   header, multi-header stacked, `title` label variant, page-offset
>   batches; negative: page already covered, mixed text+header,
>   text-only, dense-index, empty text, no iterate_items, page
>   offset).
>
>   Reconvert results (Devlin/Nagasubramanian/Sekar with the fix):
>
>   | Doc | Missing before | Missing after | Closed |
>   |---|---|---|---|
>   | Devlin_LLM_Agents | `[2, 170]` | `[2, 264]` | p170 ✓ |
>   | Nagasubramanian_Agentic_AI | `[2]` | `[]` | p2 ✓ |
>   | Sekar_MCP_Standard | `[2, 159, 228, 247]` | `[]` | All 4 ✓ |
>
>   **Sekar's p159/p228/p247 were unexpected catches** — the
>   section_header-only pattern caught more than just the
>   front-matter target. Verified each is genuinely section_header-only
>   in the source; emission is correct.
>
> - **B3.b — Cross-page-split page misattribution** (deferred to
>   v2.10 as `CROSS_PAGE_SPLIT_PAGE_ATTRIBUTION`):
>   Bourne p209's URL/citation list and Ayeva p4's dedication
>   text are absorbed into adjacent-page chunks
>   (tagged `page_number=208` / `page_number=3` respectively, with
>   `extraction_method=hybrid_chunker_pagesplit`). The v2.9 Phase 4
>   cross-page page-split fired but misattributed the content to
>   the earlier page. The strict gate's MISSING_PAGES is accurate
>   relative to per-page indexing, but the content is in the
>   corpus and discoverable via semantic search. Full diagnostic
>   in `docs/PHASE_B3_CROSS_PAGE_SPLIT_DIAGNOSTIC.md`.
>
>   No band-aid backfill was implemented because it would conflict
>   with the Phase 1 invariant ("no production code may contain or
>   emit `recovery_page_coverage`") and would duplicate content
>   already in the JSONL. The defect is upstream in the
>   cross-page-split logic.
>
> - **B3.c — Low-retrieval-value title / dedication pages**
>   (deferred to v2.10 as `LOW_RETRIEVAL_VALUE_PAGE_TAXONOMY`):
>   Greenhouse_Design p2 ("Greenhouse Design and Control" — book
>   title duplicate) and Ayeva_Python_Patterns p4 (dedication)
>   were proposed for a gate-side blank-equivalent extension under
>   the Retrieval-Value Test (`docs/DECISIONS.md`). Corpus-wide
>   false-positive validation (proposed dedication regex tested
>   against all 32 PDFs at length cap ≤200) matched Greenhouse p25
>   — a legitimate Preface acknowledgment with substantive
>   content. The detector cannot be made tight enough without
>   either narrowing to a different signal or accepting the FP
>   risk. Deferred rather than ship a fuzzy gate-side rule that
>   could drop a real Preface elsewhere. 2 pages, low total impact.
>
> Cumulative across B1 + B2 + B3 Step 2:
> 62 → 6 unaddressed-in-v2.9 missing pages (90 % reduction;
> 3 signed v2.10 deferrals on top of Firearms + KI EPUB).
> Test suite: **770 passed, 14 skipped, 0 failed** (was 760 at B2
> close, +10 net new from Step 2 section-header tests).
>
> Body retained for historical context.

**Affected docs:** Ayeva_Python_Patterns (p4 dedication, 74 chars),
Bourne_RAG_2024 (p209 URL list, 190 chars), Devlin_LLM_Agents
(p170 chapter heading, 37 chars), Greenhouse_Design (p2 title
page, 29 chars).

**Root cause (per Phase A diagnostic):** Phase 1's three layered
empty-text-chunk guards drop chunks below some minimum length
after normalization. The threshold catches legitimately-short
pages.

**Steps:**

1. Find each empty-text-chunk guard (Phase 1 closure summary
   names three: oversize-breaker, finalize stage, JSONL-write
   loop) in `src/mmrag_v2/batch_processor.py`. Identify which
   guard fires on the four B3 pages.
2. Lower the threshold OR exempt pages whose stripped text has at
   least one alphanumeric character + at least one whitespace
   (i.e., looks like a real sentence/title rather than empty
   whitespace). Avoid arbitrary threshold tuning; use a structural
   condition.
3. Regression tests: fixtures matching the four shapes — Ayeva
   p4 (dedication), Bourne p209 (URL list with section heading),
   Devlin p170 (chapter heading "II — Building Intelligent
   Foundations"), Greenhouse p2 (title page).
4. Reconvert Ayeva, Bourne, Devlin, Greenhouse (Devlin overlaps
   with B4 because Devlin also has sub-class A on p2; Greenhouse
   overlaps with B2 and B4).

**Done when:** the four B3 pages each produce ≥ 1 chunk.

**Risk:** Low-medium. Lowering the threshold risks reintroducing
chunks with whitespace-only content. The structural condition
(alpha + whitespace) is safer than a numeric threshold.

**Estimated effort:** 2-3 h.

#### B4 — Image-only page chunk-drop (largest by chunk count)

**Affected docs:** Python_Distilled (697 pages — catastrophic
case), Earthship_Vol1 (1 page), Devlin_LLM_Agents (1 page p2),
plus subsets of Fluent_Python, Python_Cookbook, Chaubal_PyTorch_Projects
to be confirmed during this sub-phase.

**Root cause (per Phase A diagnostic):** Source-PDF pages with
`text_len=0 blocks=0 images>=1` have a full-page image but no
text-layer. Docling's text-routed hybrid_chunker emits 0 chunks.
Shadow extraction fires on some pages (167 chunks corpus-wide
across Python_Distilled) but not consistently on all image-only
pages.

**Fix design (per §2d constraint #1):** Extraction policy layer.
Either `PdfConversionPlan` / `DoclingPdfAdapter` enables Docling
picture extraction unconditionally on every page, OR a
guaranteed-shadow-on-image-only-page condition is added in the
adapter / plan (NOT in the JSONL writer).

**Parallel-site audit:**

1. Existing fix? `_filter_blank_assets` at `batch_processor.py:859`
   already exists for the inverse case (drop blank asset chunks).
   There is no symmetric "guarantee an image-only page emits at
   least one image chunk" guard.
2. Too-narrow gate? Shadow extraction's activation condition
   needs inspection. It fires sometimes (167 chunks) but not
   universally on image-only pages.
3. Parallel boundaries: `engines/docling_adapter.py`,
   `engines/docling_serializers.py`, `engines/pdf_engine.py`, and
   the shadow extraction site in `batch_processor.py`. Inspect
   each before writing code.
4. Library config? Docling 2.86's `PdfPipelineOptions` exposes
   `do_picture_classification`. If picture classification +
   picture extraction is enabled, Docling should emit a
   `PictureItem` for full-page images even when text is empty.
   Verify on a Python_Distilled p887 single-page probe.

**Steps:**

1. Probe Python_Distilled p887 alone via `mmrag-v2 process
   --pages 887`. Capture: does Docling emit anything? Does shadow
   fire? What does the JSONL look like?
2. Decide between Docling-native (preferred per "libraries first")
   vs shadow-extraction fallback (less surgical but already in
   codebase).
3. Implement the fix at the extraction policy layer. Add a
   regression test fixture (an image-only page) that produces ≥ 1
   chunk.
4. Reconvert Python_Distilled + Earthship + Devlin + any other
   affected doc.

**Done when:** all docs with sub-class A pages report
`MISSING_PAGES=0` (non-blank).

**Risk:** Medium-high. Python_Distilled at 697 missing pages is
the canary; the fix must scale. Reconvert wall time is dominated
by Python_Distilled (1411 pages, expected ~3-4 h on MPS at
Phase 5a rates).

**Estimated effort:** 4-8 h diagnostic + fix; +5 h reconvert wall
time.

#### Phase B aggregate done-when

Phase B is complete when all four sub-phases (B1, B2, B3, B4)
have shipped their fix + regression test and the reconverts of
the affected 12 docs report `MISSING_PAGES=0` (non-blank).
Image chunks remain pending; Phase H handles re-enrichment.

Tests baseline: 736 at Phase 4 close → expect 736 + 4 (one
regression test per sub-phase) = 740 minimum.

Aggregate reconvert wall time: ~5-6 hours dominated by
Python_Distilled (1411 pages). The other 11 docs total ~2 hours
combined.

### Phase C — Devlin HEADING coverage

**What:** Devlin_LLM_Agents reports HEADING coverage 653/902 (72 %),
identical to Firearms' 72 %. Firearms is OCR-routed and the defect
is documented as `OCR_PATH_HEADING_PROPAGATION`. Devlin is
HybridChunker-routed and the Phase 4 carry-forward fix `b429cb5`
shipped — but the metric did not move for Devlin. Either the fix
does not apply to Devlin's specific shape, or Devlin has a second
HybridChunker-path heading-propagation defect.

**Parallel-site audit (4 questions):**

1. Existing fix? `b429cb5` exists. The question is whether it
   fires on Devlin's chunk shape.
2. Too-narrow gate? `b429cb5` updates `state.last_hybrid_heading`
   only when the prior batch ends with a heading-style chunk. If
   Devlin's batches end mid-section without a heading-style chunk,
   the carry-forward never activates.
3. Parallel boundaries: the `_process_element_v2` OCR-path
   heading handling (Firearms' deferral) lives in a different
   branch. If the defect is shared between paths,
   `OCR_PATH_HEADING_PROPAGATION` is misnamed; if it is genuinely
   distinct, Devlin gets a separate fix.
4. Library config? None applicable — heading state is a project
   construct, not a Docling primitive.

**Steps:**

1. Reproduce on a small page-window probe of Devlin
   (`mmrag-v2 process ... --pages 1-15` and 165-175): observe
   `parent_heading` on every chunk. Compare against what Docling's
   `dc.meta.headings` reports per page.
2. If the carry-forward fix never fires on the probe, fix it in
   the carry-forward decision. Add a test fixture that captures
   Devlin's batch-boundary shape.
3. If Docling's `dc.meta.headings` itself is empty on the missing
   chunks, this is the same defect as Firearms and gets the same
   deferral (with sign-off; do not silently broaden the deferral).

**Done when:** Devlin's HEADING coverage on full-doc reconvert is
≥ 0.80 OR a separate signed v2.10 deferral is recorded. The plan
expects the former; the latter requires explicit user sign-off
parallel to the Firearms deferral.

**Risk:** Medium. The fix may reveal that Phase 4's claim "the
HybridChunker-path heading carry-forward is fixed" was over-stated.

**Estimated effort:** 2-4 h diagnostic + fix; Devlin reconvert is
already in Phase B (Devlin also has MISSING_PAGES=2 — both fixes
land in the same reconvert).

### Phase D — IMAGE_DESCRIPTION_UNUSABLE policy

**What:** Two image chunks across the corpus are flagged
`IMAGE_DESCRIPTION_UNUSABLE`: AIOS p8 ("Bar chart.0 to 1.0.") and
Hybrid_EV p1 ("Logo icon."). Both have `vision_status="complete"`
(not `hard_fallback`), so the Phase 3 F4 hard-fallback exemption
does not apply. The gate's "useful description" rule rejects them
because the description is shorter than the per-asset minimum.

**Parallel-site audit (4 questions):**

1. Existing fix? Phase 3 Step 4 (`649c952`) added a retry harness
   for complex-asset-short-description cases via `_needs_detail_retry`
   in `scripts/enrich_image_chunks_v29.py`. That harness checks
   `_is_short_description` and the asset's `classify_asset_complexity`
   result. AIOS p8 is a bar chart (complex); the harness should
   have flagged it for retry.
2. Too-narrow gate? `_is_short_description` (
   `scripts/enrich_image_chunks_v29.py:114`) checks
   `len(stripped) < SHORT_DESC_THRESHOLD_CHARS and "layout" not
   in stripped.lower()`. "Bar chart.0 to 1.0." is short but does
   not contain "layout"; it should have been flagged. Check why
   the Phase 5b run did not retry it.
3. Parallel boundaries: the qa gate
   (`scripts/qa_full_conversion.py`'s `_is_blankish_visual_description`
   and `qa_semantic_fidelity.py`'s `is_placeholder_image_or_table`)
   both classify these descriptions. They should agree.
4. Library config? None applicable.

**Steps:**

1. Run the Phase 3 detail-retry path on the 2 chunks in isolation
   (one-off enrichment with `--detail-retry-only` if such a flag
   exists; otherwise a minimal script). Observe whether
   `_needs_detail_retry` returns True for them.
2. If it returns False on a clearly-needing-retry case, fix
   `_needs_detail_retry` (likely an asset-complexity-classifier
   threshold). Add a test fixture.
3. If it returns True but the retry response was still terse,
   accept that this is qwen3-vl-plus behavior on very small or
   low-content assets and propose a gate-side allowance backed by
   `docs/QUALITY_GATES.md` rationale (e.g., "image_area < N px²
   exempts from min-length rule").

**Done when:** AIOS and Hybrid_EV either upgrade to PASS, or the
gate accepts the description with explicit documented rationale.

**Risk:** Low. Two-chunk surface, well-scoped.

**Estimated effort:** 1-3 h. Re-enrichment cost: 2 chunks × 2-3
calls each (initial + retries) ≈ negligible.

### Phase E — Combat blank-asset extraction filter

**What:** Combat reports `AUDIT_FAIL(IMAGE)`: a single asset
`a4c2916a64c2_027_figure_36.png` is a near-blank image (mean=253,
std=7.2 on a 0-255 scale; this is effectively white pixels with
tiny noise). It was emitted as an image chunk and enriched (the
description is a generic blank-page transcription). The same
condition reproduced after Combat's fresh reconvert, so it is not
a race artifact.

**Parallel-site audit (4 questions):**

1. Existing fix? `docs/QUALITY_GATES.md` documents an image-quality
   check; the strict gate fires on this asset. The question is
   whether the producer side (engine/extraction) should drop
   blank assets before they reach the JSONL.
2. Too-narrow gate? `ASSET_TINY` checks file size < 1 KB.
   `BLANK_ASSET` would check pixel-statistics (mean > 240 AND std
   < 10, or similar). The current pipeline appears to have no
   producer-side blank check.
3. Parallel boundaries: `engines/docling_adapter.py` and
   `engines/docling_serializers.py` are the producer sites. If a
   filter is added, it must be at the asset-extraction site (where
   the PNG is written) not at the chunk-emission site (where the
   image is already serialized into the chunk).
4. Library config? Docling has no built-in blank-asset detection
   that I'm aware of; needs custom check.

**Steps:**

1. Confirm the blank asset is reproducible by re-running the
   Combat reconvert (already done — it is).
2. Add a producer-side blank-asset filter at the asset write site
   in `engines/docling_serializers.py`. Threshold: mean > 240
   AND std < 10 (these are the asset's measured values). Add a
   fixture-based test that asserts a blank PIL image is rejected
   and a normal image is kept.
3. Reconvert Combat only.
4. Re-validate via strict gate.

**Done when:** Combat strict gate reports `AUDIT_PASS(IMAGE)`.

**Risk:** Low-medium. The threshold needs to be conservative to
avoid dropping legitimate light-toned figures (e.g., line charts on
white backgrounds). The fixture must cover both rejection and
preservation cases.

**Estimated effort:** 2-3 h. Reconvert + re-enrich of Combat alone:
~80 min (per the Combat recovery timing already observed).

### Phase F — Greenhouse SEMANTIC_FAIL (code_indent 0.800)

**What:** Greenhouse_Design reports
`code_indentation_fidelity=0.800` against the floor of 0.90, as a
hard `SCRIPT_GATE_FAIL`. Kimothi (0.667) and Raieli (0.884) report
the same metric but only as `SCRIPT_ADVISORY_FAIL` (WARN-class).
The Greenhouse-vs-others routing through SCRIPT_GATE_FAIL needs
to be confirmed during this phase.

**Steps:**

1. Confirm the hard/advisory split in `scripts/qa_full_conversion.py`
   — find the code path that maps `qa_semantic_fidelity.py` exit
   to `[FAIL]` vs `[WARN]`. Document which condition trips for
   Greenhouse but not for Kimothi/Raieli.
2. If the hard FAIL is correct (Greenhouse legitimately has
   worse code-indent), Phase B's reconvert (which Greenhouse is
   already in) may carry a code-enrichment improvement. After
   Phase B re-runs Greenhouse, re-check this metric.
3. If the metric does not improve after reconvert, treat as a real
   code-indentation defect specific to Greenhouse's content type
   (an HVAC controls textbook with embedded MATLAB / Simulink
   blocks). Diagnose CodeFormulaV2 routing on those blocks.

**Done when:** Greenhouse `code_indentation_fidelity >= 0.90` OR
the hard-fail-routing is shown to be over-strict for non-software
docs and is adjusted with explicit rationale in
`docs/QUALITY_GATES.md`.

**Risk:** Medium. CodeFormulaV2 behavior on non-software content
is not well-characterized.

**Estimated effort:** 2-4 h, mostly diagnostic. No incremental
reconvert cost — rolled into Phase B.

### Phase G — WARN-class promotion / repair

**What:** Eight docs are at WARN level under the unchanged strict
gate. Per Goal 1, WARN is not a ship state. The candidates and
their classes:

| Doc | WARN class | Likely path |
|---|---|---|
| HarryPotter_and_the_Sorcerers_Stone | ASSET_TINY (2 figs < 1KB) | Either drop sub-1KB icons at extract, or document allowance for "icon" class in `QUALITY_GATES.md` |
| Form_betwistingsformulier | ASSET_TINY (1 fig) | Same as above |
| A_comprehensive_review_on_hybrid_electri | ASSET_TINY (1 fig) | Same as above |
| Hao_ML_Platform | ASSET_TINY (2 figs) | Same as above |
| Jungjun_AI_Agent | VISION_HARD_FALLBACK_RATE 5/46 (10.9%) | Detail-retry rerun OR per-doc allowance (Jungjun has many small UI screenshots — qwen3-vl-plus refuses to transcribe text-heavy UI per source-sanctity rule) |
| ChatGPT_Praktijk_handboek | PAGE_COUNT_UNKNOWN | EPUB lane has no source page count; the gate can either compute a virtual page count or accept "EPUB no-page" as a documented variant |
| Kimothi_RAG_Guide | SEMANTIC_FAIL advisory (code_indent=0.667) | Phase F decides whether to upgrade to hard fail (already advisory only — leave) |
| Raieli_AI_Agents | SEMANTIC_FAIL advisory (code_indent=0.884) | Same as Kimothi |

**Steps:**

1. **ASSET_TINY (HARRY, Form_betwist, A_comprehensive, Hao).**
   Decide producer-side vs gate-side. Producer-side: drop assets
   < 1 KB at extract time. Gate-side: per-doc allowance in
   `QUALITY_GATES.md` for "icon-class" content. The Phase 5b
   measurements (HARRY's two figures under 1 KB) look like real
   small UI icons. Recommend gate-side allowance: `ASSET_TINY`
   becomes informational, not a warning, when the asset is in
   HARRY-class digital_literature (small icons on chapter
   dividers). Make this explicit in `docs/QUALITY_GATES.md`.
2. **VISION_HARD_FALLBACK_RATE (Jungjun).** Re-run targeted
   detail-retry. Confirm whether the 5 hard_fallbacks all have
   the `complex_asset_short_response_after_retry` sentinel (the
   Phase 3 F4-exempt class). If they do, the gate's 5 % floor is
   too tight for doc-shapes dominated by text-heavy UI. Promote
   to per-doc-class threshold in `QUALITY_GATES.md`. No new VLM
   spend if the F4 sentinel is already on all 5.
3. **PAGE_COUNT_UNKNOWN (ChatGPT).** Add a virtual-page-count
   computation in the EPUB lane (count chunks that look like
   page breaks, or use a heuristic based on chunk text density).
   This is the same broader EPUB lane as KI EPUB but the lighter
   half — the warning is structural and can be promoted to PASS
   variant without a full lane rewrite.
4. **Kimothi, Raieli SEMANTIC_FAIL advisory.** Already WARN-only;
   if Phase F decides not to escalate to hard FAIL, these stay as
   documented advisories and are not blockers for Goal 1. Move
   into `QUALITY_GATES.md` as known advisory cases for the
   technical-manual code-heavy class.

**Done when:** every WARN doc is either PASS, or the WARN class
has a documented allowance in `docs/QUALITY_GATES.md` (per
the same provision in the previous plan's Goal 1).

**Risk:** Low. Most of this is gate / threshold work, not
production-code work.

**Estimated effort:** 2-4 h. No incremental VLM cost.

### Phase H — Targeted re-enrichment + full-corpus re-validation

**What:** Phase B reconverts produced fresh JSONLs with placeholder
image chunks. Combat (Phase E) reconverts add another. This phase
re-runs `scripts/enrich_image_chunks_v29.py` on those subsets and
re-runs the strict gate across all 34 to confirm Goal 1.

**Steps:**

1. List the docs whose JSONLs have `vision_status="pending"`
   chunks after Phases B and E. Confirm via dry-run:
   `python scripts/enrich_image_chunks_v29.py output/*/ingestion.jsonl --dry-run`.
2. Run targeted re-enrichment (only those docs) with the same
   `qwen3-vl-plus` settings used in Phase 5b. The enrichment
   script is idempotent on already-complete chunks; pass the full
   34 list and it will skip the unchanged ones at no API cost.
3. Run the full-corpus strict gate (the same wrapper as in
   `/tmp/run_strict_gate_v2.py`) — expect 32 PASS + 2 signed
   deferrals.
4. Spot-check 3 random reconverted JSONLs for: 0 within-file
   chunk_id duplicates, 0 within-file vision_status="pending",
   `hard_fallback` rate <= 5 % per doc (except F4-exempt).

**Done when:** Goal 1 is met (32 PASS + 2 signed deferrals).

**Risk:** Low. Same enrichment script that ran cleanly in Phase 5b.
The Combat-race incident's root cause (parallel invocation) does
not apply here because this plan does not spawn parallel runners.

**Estimated effort:** Cloud spend depends on how many new image
chunks Phase B's reconverts produce. Worst case = Python_Distilled
+ Python_Cookbook + Cronin + Nagasubramanian + Sekar + Fluent +
Chaubal + Greenhouse + Devlin + Bourne + Earthship + Ayeva ≈
207 + 26 + 4 + 76 + 35 + 83 + 55 + 323 + 67 + 29 + 442 + 24 ≈
1,371 image chunks. At ~3.5 s/call: ~80 min wall time. Combat
re-enrich (Phase E): 351 chunks ≈ 20 min. **Total cloud spend:
≈1,722 calls (~40 % of Phase 5b's 4,382).**

### Phase I — Qdrant rebuild + RC AFTER snapshot + `v2.9.0-rc1` tag

**What:** With Phases A-H complete and Goal 1 verified, drop
`mmrag_v2_8` and recreate it from the post-recovery JSONLs. Write
the RC AFTER snapshot. Tag `v2.9.0-rc1`.

**Steps:**

1. Confirm Qdrant container is up:
   ```bash
   curl -sS http://localhost:6333/collections/mmrag_v2_8/points/count \
     -X POST -H "Content-Type: application/json" -d '{"exact":true}'
   ```
   Expected pre-drop: `{"result":{"count":22137}, ...}` (v2.8
   shipped state).
2. **Drop the collection** (DESTRUCTIVE, single-call user
   confirmation required at this step):
   ```bash
   curl -X DELETE http://localhost:6333/collections/mmrag_v2_8
   ```
3. Run `python scripts/ingest_to_qdrant.py output/*/ingestion.jsonl --collection mmrag_v2_8`. Confirm point count == unique non-pending chunk count across the 34.
4. Spot-check via `python scripts/search_qdrant.py --stats` and a
   sample retrieval.
5. Write `docs/QUALITY_SNAPSHOT_<DATE>_v2.9.0-rc1_after.md` with
   the post-recovery strict-gate table, Qdrant counts, and
   provenance (commits, scripts, run timestamps).
6. Update `docs/PROJECT_STATUS.md` to record `v2.9.0-rc1` as the
   current shippable state with two signed deferrals.
7. Run the test suite one more time: `pytest tests/ -q` must show
   0 failures.
8. Tag `v2.9.0-rc1` with a release-style message naming the
   Phase 5 evidence and the two signed deferrals carrying forward
   to v2.10.

**Done when:** the tag exists on `main`, `mmrag_v2_8` is rebuilt,
and the RC AFTER snapshot is written. Final `v2.9.0` remains
blocked until the two signed deferrals are repaired in v2.10.

**Risk:** Low. All preceding phases passed Goal 1. The Qdrant
drop is destructive and gets explicit user confirmation at step 2.

**Estimated effort:** 1-2 h. Qdrant ingest depends on chunk count
(~25-30K embeddable chunks at the typical v2.8 ingest rate).

---

## 4. Acceptance gate (unchanged from the previous plan)

Before tagging `v2.9.0-rc1`:

- [ ] **Strict gate.** `scripts/qa_full_conversion.py --source-pdf --allow-warnings` reports `QA_PASS` for **32 of 34** docs. The only allowed non-PASS rows are Firearms (`OCR_PATH_HEADING_PROPAGATION`) and KI_En (`KI_EPUB_EXTRACTION_LANE_REWRITE`). (Goal 1.)
- [ ] **Smoke.** `bash scripts/smoke_multiprofile.sh` — every row `GATE_PASS` + `UNIVERSAL_PASS` (form variants per `docs/QUALITY_GATES.md` "Form / Invoice Acceptance Class" remain valid). (Goal 10.)
- [ ] **chunk_id contract.** 0 within-file duplicates across all 34 canonical JSONLs. (Goal 8.)
- [ ] **Vision contract.** 0 `vision_status="pending"` corpus-wide; `hard_fallback` rate <= 5 % per doc except where the F4 sentinel exempts. (Goal 11.)
- [ ] **Tests.** `pytest tests/ -q` reports 736 passed or higher with 0 failed. (Goal 9.)
- [ ] **No filename-specific production logic.** (Same as previous plan; preserved here.)
- [ ] **No negative test loosened.** Any threshold change in `docs/QUALITY_GATES.md` is documented with explicit rationale and added regression tests.
- [ ] **Documentation in sync.** `docs/PLAN_V2.9.md`, `docs/PROJECT_STATUS.md`, `docs/QUALITY_SNAPSHOT_*v2.9.0-rc1_after.md`, `docs/DECISIONS.md`, `CHANGELOG.md` all reflect the RC state.
- [ ] **Qdrant in sync.** `mmrag_v2_8` rebuilt from post-recovery JSONLs; point count matches unique non-pending chunk count across 34. (Goal 12.)

Before tagging final `v2.9.0`:

- [ ] All of the above.
- [ ] Firearms HEADING coverage >= 0.80 (`OCR_PATH_HEADING_PROPAGATION` fixed in v2.10 backlog).
- [ ] KI_En_ChatGPT_Praktische_Gids `UNIVERSAL_PASS` (`KI_EPUB_EXTRACTION_LANE_REWRITE` fixed in v2.10 backlog).

## 5. Provenance and accounting

- Active BEFORE state: `docs/QUALITY_SNAPSHOT_2026-05-11_v2.9_strict_gate_full_corpus.md`.
- Previous plan: `docs/archive/PLAN_V2.9_2026-05-06_strict_gate_recovery.md`.
- Phase 5 attempt evidence: `docs/QUALITY_SNAPSHOT_2026-05-10_v2.9_phase5_attempt.md`.
- Phase 4 close evidence: `docs/QUALITY_SNAPSHOT_2026-05-09_v2.9_phase4_after.md`.
- v2.8 SHIPPED baseline: `docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md`.
- Active Qdrant collection: `mmrag_v2_8` at v2.8 state (22,137 points). Will not be touched until Phase I.
- Cloud-VLM spend already incurred: ~4,382 calls (Phase 5b) + ~120 calls (Combat re-enrichment after race). Phase H adds ~1,722 calls (estimated, depends on Phase B reconvert outcomes).

## 6. Honest accounting

The previous plan implied that Phase 1 closed `MISSING_PAGES` for
the corpus. The Phase 1 closure commit message and snapshot were
honest about the scope ("Kimothi 258 pages reports
AUDIT_PASS/UNIVERSAL_PASS/HYGIENE_PASS; Ayeva back-index probe
per-page chars 76-105% of source PDF text"), but the inference
that this would generalize to the other 32 docs was not validated
until this plan's BEFORE snapshot.

This is not a regression. It is honesty catching up to scope. The
correct lesson: any closure phase that claims to close a failure
class corpus-wide must run the strict gate corpus-wide as part of
its done-criteria. Phase H of this plan codifies that for every
future failure-class closure.
