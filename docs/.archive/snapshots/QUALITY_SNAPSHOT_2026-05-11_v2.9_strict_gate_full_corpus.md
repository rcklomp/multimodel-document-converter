# Quality Snapshot 2026-05-11 — v2.9 Strict-Gate Full-Corpus AFTER Phase 5b

> **Status:** First full-corpus run of `scripts/qa_full_conversion.py
> --source-pdf --allow-warnings` against the v2.9 Phase 5a/b outputs.
> Result is **9 PASS / 8 WARN / 17 FAIL out of 34**. `v2.9.0-rc1` is
> NOT shippable in this state. Only **2 of the 17 FAILs are signed
> Phase 4 deferrals** (Firearms, KI EPUB). The other 15 are
> previously-unmeasured failure classes that the Phase 4 close
> snapshot did not enumerate.
>
> Predecessors: `docs/QUALITY_SNAPSHOT_2026-05-09_v2.9_phase4_after.md`
> (Phase 4 close, 6 docs validated under strict gate),
> `docs/QUALITY_SNAPSHOT_2026-05-10_v2.9_phase5_attempt.md` (Phase 5a
> 30/34 fresh outputs blocked by 4 conversion timeouts).

## 1. Why this snapshot exists

Phase 4 closed with two signed v2.10 deferrals (Firearms
`OCR_PATH_HEADING_PROPAGATION`, KI EPUB
`KI_EPUB_EXTRACTION_LANE_REWRITE`). Phase 5a then re-converted the
canonical 34 (34/34 fresh JSONLs, 0 chunk_id duplicates corpus-wide).
Phase 5b enrichment ran via `scripts/enrich_image_chunks_v29.py` and
resolved 4,269 / 4,382 image chunks to `vision_status="complete"`
(97.42 %); 113 (2.58 %) ended in `vision_status="hard_fallback"`
with sentinel reasons. Combat had to be re-converted + re-enriched
after a documented race-corruption incident; the recovered Combat
JSONL is structurally clean.

This snapshot is the first time the full strict gate has been run
across **all 34** canonical outputs. The Phase 4 AFTER snapshot
explicitly only validated 6 docs under the strict gate (Hao,
Adedeji, Combat, PCWorld, Firearms, KI EPUB). The 32/34 PASS goal
recorded in `docs/PLAN_V2.9.md` §Goals was an aspiration based on
the Phase 1 dense-index closure on Kimothi + Ayeva back-index probes,
not a verified Phase 4 close state.

## 2. Command and inputs

```bash
python scripts/qa_full_conversion.py output/<doc>/ingestion.jsonl \
  --source-pdf data/<category>/<file>.pdf \
  --allow-warnings
```

EPUBs (`ChatGPT_Praktijk_handboek`, `KI_En_ChatGPT_Praktische_Gids`)
were invoked without `--source-pdf`. All 34 invocations completed
without timeout. Per-doc captured outputs are at
`/tmp/strict_gate_capture/<doc>.txt` (temporary).

## 3. Per-document strict-gate result

| Doc | Status | Fails | Warns | Failure / warning detail |
|---|---|---:|---:|---|
| HarryPotter_and_the_Sorcerers_Stone | QA_WARN | 0 | 1 | ASSET_TINY (2 figures < 1KB) |
| Form_0013_invoice | QA_PASS | 0 | 0 | — |
| Form_betwistingsformulier | QA_WARN | 0 | 1 | ASSET_TINY (1 figure < 1KB) |
| CarOK_voorraadtelling | QA_PASS | 0 | 0 | — |
| **AIOS_LLM_Agent_Operating_System** | **QA_FAIL** | 1 | 0 | IMAGE_DESCRIPTION_UNUSABLE 1/10 (`p8` desc="Bar chart.0 to 1.0.") |
| A_comprehensive_review_on_hybrid_electri | QA_WARN | 0 | 1 | ASSET_TINY (1 figure < 1KB) |
| **Hybrid_electric_vehicles** | **QA_FAIL** | 1 | 1 | IMAGE_DESCRIPTION_UNUSABLE 1/12 (`p1` desc="Logo icon.") + ASSET_TINY |
| IRJET_Modeling_of_Solar_PV | QA_PASS | 0 | 0 | — |
| Recent_Trends_in_Transportation | QA_PASS | 0 | 0 | — |
| **Combat_Aircraft_August_2025** | **QA_FAIL** | 1 | 0 | AUDIT_FAIL(IMAGE): Blank asset `a4c2916a64c2_027_figure_36.png` (mean=253, std=7.2) |
| PCWorld_July_2025 | QA_PASS | 0 | 0 | — |
| ATZ_Elektronik_German | QA_PASS | 0 | 0 | — |
| Kimothi_RAG_Guide | QA_WARN | 0 | 3 | SEMANTIC_FAIL advisory (code_indentation_fidelity=0.667) |
| Integra_manual | QA_PASS | 0 | 0 | — |
| Jungjun_AI_Agent | QA_WARN | 0 | 1 | VISION_HARD_FALLBACK_RATE 5/46 (10.9 %; limit 5.0 %) |
| **Bourne_RAG_2024** | **QA_FAIL** | 1 | 1 | MISSING_PAGES=1 (p209) + VISION_HARD_FALLBACK_RATE |
| **Devlin_LLM_Agents** | **QA_FAIL** | 2 | 1 | MISSING_PAGES=2 (p2, p170) + AUDIT_FAIL(HEADING) coverage 653/902 (72 %) + VISION_HARD_FALLBACK_RATE 12/67 (17.9 %) |
| Raieli_AI_Agents | QA_WARN | 0 | 1 | SEMANTIC_FAIL advisory (code_indentation_fidelity=0.884) |
| Adedeji_GenAI_Google_Cloud | QA_PASS | 0 | 0 | — |
| **Cronin_GenAI_Models** | **QA_FAIL** | 1 | 0 | MISSING_PAGES=24 (p5-p20, p22-p29 cluster) |
| Hao_ML_Platform | QA_WARN | 0 | 1 | ASSET_TINY (2 figures < 1KB) |
| **Nagasubramanian_Agentic_AI** | **QA_FAIL** | 1 | 2 | MISSING_PAGES=16 (p2, p5-p19) |
| **Sekar_MCP_Standard** | **QA_FAIL** | 1 | 1 | MISSING_PAGES=13 (p2, p5-p13, p159, p228, p247) |
| **Python_Cookbook** | **QA_FAIL** | 1 | 0 | MISSING_PAGES=40 (sparse across body) |
| ArcGIS_Python_Cookbook | QA_PASS | 0 | 0 | — |
| **Fluent_Python** | **QA_FAIL** | 1 | 1 | MISSING_PAGES=13 (p5, p8, p10-p11, p27, p43, p125-p126, p136, p163, p243, p425, p609) |
| **Python_Distilled** | **QA_FAIL** | 1 | 1 | **MISSING_PAGES=697** (p5, p6, p17, p75, p541-p549, ... — most of the book) |
| **Ayeva_Python_Patterns** | **QA_FAIL** | 1 | 0 | MISSING_PAGES=1 (p4) |
| **Chaubal_PyTorch_Projects** | **QA_FAIL** | 1 | 0 | MISSING_PAGES=9 (p2, p4-p11) |
| **Earthship_Vol1** | **QA_FAIL** | 1 | 1 | MISSING_PAGES=1 (p109) + VISION_HARD_FALLBACK_RATE |
| **Firearms** ⚠ signed deferral | **QA_FAIL** | 1 | 0 | AUDIT_FAIL(TEXT, HEADING): HEADING coverage 790/1094 (72 %); page_coverage 292/292 clean |
| **Greenhouse_Design** | **QA_FAIL** | 1 | 1 | MISSING_PAGES=4 (p2, p3, p11, p23) + SEMANTIC_FAIL advisory (code_indentation_fidelity=0.800) |
| ChatGPT_Praktijk_handboek | QA_WARN | 0 | 1 | PAGE_COUNT_UNKNOWN (EPUB lane has no page count) |
| **KI_En_ChatGPT_Praktische_Gids** ⚠ signed deferral | **QA_FAIL** | 2 | 1 | UNIVERSAL_FAIL: within_page_text_dupe_excess=318; DUPLICATE_LONG_TEXT 24 groups; 4518/4518 text chunks have missing bbox |

## 4. Aggregate result

- **PASS: 9** — Form_0013, CarOK, IRJET, Recent_Trends, PCWorld, ATZ, Integra, Adedeji, ArcGIS.
- **WARN: 8** — HARRY, Form_betwistingsformulier, A_comprehensive, Kimothi, Jungjun, Raieli, Hao, ChatGPT.
- **FAIL: 17** — AIOS, Hybrid_EV, Combat, Bourne, Devlin, Cronin, Nagasubramanian, Sekar, Python_Cookbook, Fluent_Python, Python_Distilled, Ayeva, Chaubal, Earthship, Firearms¹, Greenhouse, KI_En¹.

¹ Signed Phase 4 deferrals; the other 15 FAILs are not authorized
under the v2.9.0-rc1 contract.

## 5. Failure-class taxonomy

### 5a. MISSING_PAGES (10 docs; the largest unaddressed class)

The Phase 1 dense-index closure on commit `df91061` was validated on
Kimothi + Ayeva back-index probes. It was not validated on the full
corpus. The full-corpus picture is:

| Doc | Missing pages | Pattern |
|---|---:|---|
| Python_Distilled | 697 | Catastrophic — most of body |
| Python_Cookbook | 40 | Scattered through body |
| Cronin_GenAI_Models | 24 | Continuous front cluster p5-p20 |
| Nagasubramanian_Agentic_AI | 16 | Continuous front cluster p2, p5-p19 |
| Sekar_MCP_Standard | 13 | Front cluster + back index |
| Fluent_Python | 13 | Scattered |
| Chaubal_PyTorch_Projects | 9 | Front cluster p2, p4-p11 |
| Greenhouse_Design | 4 | Sparse front (p2, p3, p11, p23) |
| Devlin_LLM_Agents | 2 | Front (p2) + p170 |
| Bourne_RAG_2024 | 1 | p209 (single body page) |
| Earthship_Vol1 | 1 | p109 (single body page) |
| Ayeva_Python_Patterns | 1 | p4 (single front page) |

Two shapes are visible in the data:

1. **Front-cluster shape** (p2 + p5-pXX continuous run) — Cronin,
   Nagasubramanian, Sekar, Chaubal, Greenhouse partially. These are
   not classic TOC/index pages; they sit in the front matter
   (preface / copyright / TOC / list-of-figures region) where
   pagination conventions shift between roman/arabic numerals or
   where Docling's `document_index` label classification may not
   apply.
2. **Sparse-body shape** (random single body pages) — Bourne p209,
   Earthship p109, Ayeva p4, Fluent/Python_Distilled/Python_Cookbook
   scattered, Devlin p170. Each is a single page that visibly
   contains content but never emits a chunk.

`Python_Distilled` at 697 missing is a separate phenomenon — it is
not the Phase 1 dense-index class. It almost certainly indicates a
page-numbering mismatch between the JSONL `page_number` field and the
source PDF page index (PDF declared pages = ~880, JSONL chunks
declare pages = ~183 visible). This must be diagnosed before any
single-page MISSING_PAGES fix is attempted, because the page-index
itself may be off-by-N.

### 5b. AUDIT_FAIL (HEADING coverage < 80 %) — 2 docs

| Doc | HEADING coverage | Notes |
|---|---:|---|
| Firearms | 790/1094 = 72 % | Signed deferral `OCR_PATH_HEADING_PROPAGATION` (Phase 4 Step 4); OCR-routed; documented v2.10 fix |
| Devlin_LLM_Agents | 653/902 = 72 % | **Not signed off.** Devlin is HybridChunker-routed, not OCR-routed; same metric but different defect class |

The Phase 4 fix `b429cb5` (heading carry-forward across batches) was
validated correct in unit tests + small probe on a HybridChunker-routed
doc, but Devlin shows the metric did not move in practice. Either the
fix doesn't apply to Devlin's specific shape, or there is a second
HybridChunker-path heading-propagation defect.

### 5c. IMAGE_DESCRIPTION_UNUSABLE (2 docs)

Single chunks per doc whose VLM description is too short / generic
to be useful, but `vision_status` is `complete` (not `hard_fallback`).
The Phase 3 F4 hard-fallback exemption and complexity-aware short-
description rule (`tests/test_qa_image_gate_calibration.py`) does not
exempt these because they are not flagged hard_fallback.

| Doc | Image | Description |
|---|---|---|
| AIOS | p8 chunk `07a1232cccf4_008_image_829a10c2` | "Bar chart.0 to 1.0." |
| Hybrid_EV | p1 chunk `2baf312fdd78_001_image_8d07b2d8` | "Logo icon." |

These are real (the bar chart needs more detail; the logo icon is
trivially unhelpful). The Phase 3 calibration may be missing a path
that flags qwen3-vl-plus's terse responses on complex assets for
detail-retry when the asset itself is complex but the description
is naive.

### 5d. AUDIT_FAIL (TEXT) — 1 doc

| Doc | Notes |
|---|---|
| Firearms | TEXT failure alongside HEADING; both signed-deferred under `OCR_PATH_HEADING_PROPAGATION` |

### 5e. AUDIT_FAIL (IMAGE) — 1 doc

| Doc | Evidence |
|---|---|
| Combat_Aircraft_August_2025 | Blank asset `a4c2916a64c2_027_figure_36.png` (mean=253, std=7.2) — a near-white image emitted as an image chunk |

This is a conversion-time defect (Docling extracted a nearly-blank
region as a figure asset), not an enrichment issue. After the
race-recovery reconvert, the same blank-asset condition occurred — so
it is reproducible from the source PDF, not a race artifact.

### 5f. SEMANTIC_FAIL (code_indentation_fidelity < 0.90) — 1 hard FAIL

| Doc | code_indent | Notes |
|---|---:|---|
| Greenhouse_Design | 0.800 | Hard FAIL (combined with MISSING_PAGES=4) |

The semantic-fidelity gate is `qa_semantic_fidelity.py`. For
Greenhouse it produces an advisory failure that propagates as a
hard FAIL via `SCRIPT_GATE_FAIL`. Kimothi (0.667) and Raieli (0.884)
also fail this metric but only as advisory `[WARN]`, not hard
`[FAIL]`. The Greenhouse-vs-Kimothi/Raieli difference appears to be
the document profile path through `SCRIPT_GATE_FAIL` vs
`SCRIPT_ADVISORY_FAIL` — to be confirmed during the v2.9 rework.

### 5g. UNIVERSAL_FAIL (1 doc, signed deferral)

| Doc | Notes |
|---|---|
| KI_En_ChatGPT_Praktische_Gids | `within_page_text_dupe_excess=318`; `DUPLICATE_LONG_TEXT` 24 groups; 4518/4518 text chunks have no bbox; no real pagination. Signed deferral `KI_EPUB_EXTRACTION_LANE_REWRITE` |

ChatGPT_Praktijk_handboek (the other EPUB) sits in WARN with only
`PAGE_COUNT_UNKNOWN` — the EPUB lane handles it less catastrophically
but still produces no real pagination. Both EPUBs are affected by
the same deferral.

### 5h. WARN classes (8 docs — non-PASS per v2.9 plan Goal 1)

| Class | Docs |
|---|---|
| ASSET_TINY (< 1 KB assets) | HARRY (2), Form_betwistingsformulier (1), A_comprehensive (1), Hao (2) |
| SEMANTIC_FAIL (advisory) | Kimothi (0.667), Raieli (0.884) |
| VISION_HARD_FALLBACK_RATE > 5 % | Jungjun (10.9 %), Devlin (already a FAIL), Earthship (warns alongside its FAIL), Bourne (warns alongside its FAIL), Nagasubramanian (warns alongside its FAIL), Fluent_Python (warns alongside its FAIL), Sekar (warns alongside its FAIL), Python_Distilled (warns alongside its FAIL) |
| PAGE_COUNT_UNKNOWN | ChatGPT (EPUB) |

`Hao` WARN class is ASSET_TINY only — its earlier Phase 4 PASS row in
the Phase 4 close snapshot used the `--source-pdf` blank-page-aware
form, which is the same form used here. Hao remained PASS for the
page-coverage check and only picked up the ASSET_TINY warning.

## 6. Phase 4 close vs. now — what actually changed

The Phase 4 AFTER snapshot (per `docs/QUALITY_SNAPSHOT_2026-05-09_v2.9_phase4_after.md`)
made 6 explicit strict-gate claims. Re-checked against this snapshot:

| Doc | Phase 4 claim | Now | Note |
|---|---|---|---|
| Hao | QA_PASS for blank-page check; ASSET_TINY warn | QA_WARN (ASSET_TINY) | Unchanged |
| Adedeji | QA_PASS | QA_PASS | Unchanged |
| Combat | QA_PASS for corruption check; pre-existing VLM_PENDING separate | QA_FAIL (blank asset p27 figure_36) | **New failure** surfaced by post-enrichment audit |
| PCWorld | QA_PASS | QA_PASS | Unchanged |
| Firearms | HEADING FAIL, DEFERRED | HEADING FAIL, DEFERRED | Unchanged (consistent with signed deferral) |
| KI EPUB | UNIVERSAL_FAIL, DEFERRED | UNIVERSAL_FAIL, DEFERRED | Unchanged (consistent with signed deferral) |

The 17 FAILs in this snapshot break down as:

- **0** regressions on Phase 4 close docs (excluding Combat — see
  next bullet).
- **1** new failure on a Phase 4 close doc: Combat blank asset
  `p27 figure_36`. This was not present in the Phase 4 close
  reading because that reading evaluated `Combat (p1-100, no VLM)`
  and reported corruption-check only. The blank-asset class was not
  explicitly checked.
- **2** signed deferrals (Firearms, KI EPUB) confirmed unchanged.
- **14** previously-unmeasured failure classes (MISSING_PAGES on 10
  docs minus 3 already in this list; AUDIT_FAIL HEADING Devlin;
  IMAGE_DESCRIPTION_UNUSABLE AIOS + Hybrid_EV; SEMANTIC_FAIL
  Greenhouse).

In short: **Phase 4 close was honest about what it validated, and
this snapshot is honest about what was never validated.** There is
no regression — only a much wider failure surface than Phase 4's
narrow strict-gate sample suggested.

## 7. Implications for v2.9.0-rc1

`v2.9.0-rc1` per `docs/PLAN_V2.9.md` §Goals 1 allows non-PASS rows
**only** for the two signed deferrals. With 15 unallowed FAIL/WARN
rows, the RC tag is blocked.

The Qdrant `mmrag_v2_8` collection has correctly not been touched.
The RC AFTER snapshot tag was correctly not generated. The cloud-VLM
enrichment is real and reusable; the JSONLs are structurally clean
(0 bad JSON, 0 chunk_id duplicates corpus-wide). The work to
reach `v2.9.0-rc1` is to close the 15 unallowed strict-gate failures
without paying for re-enrichment of the 4,269 complete image chunks.

## 8. Active artifacts

- `output/<doc>/ingestion.jsonl` × 34 — Phase 5a/b state (30,390
  total lines, 0 bad JSON across all 34 after Combat recovery).
- `output/_logs/phase5b_enrichment.log` — my Phase 5b run (PID 47624)
  partial, killed at 20:18 after detecting parallel-Codex race on
  Combat tmp file.
- `output/_logs/phase5_combat_reconvert.log` — first Combat reconvert
  attempt under my session (timed out at 1h before refiner could
  complete).
- `output/_convert_books.log` — full Phase 5a runner history including
  the Codex-session re-conversion of Combat at 02:51 that produced
  the clean Combat JSONL.

## 9. Next step

This snapshot is the BEFORE state for the rewrite of
`docs/PLAN_V2.9.md`. The new plan replaces the current
`docs/PLAN_V2.9.md` body (which still describes Phase 5 as the
forward path) with a recovery-phase plan organized around the 9
failure classes above, in priority order of severity and dependency.
The current `docs/PLAN_V2.9.md` will be archived to
`docs/archive/PLAN_V2.9_2026-05-06_strict_gate_recovery.md` before
the rewrite.
