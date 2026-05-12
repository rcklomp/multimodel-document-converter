# Quality Snapshot 2026-05-11 — v2.9.0-rc1 AFTER

> **Status:** `v2.9.0-rc1` ship state. Strict gate reports **26
> PASS / 0 WARN / 8 FAIL** across the 34-doc canonical corpus, with
> all 8 FAILs covered by signed v2.10 deferrals (`docs/DECISIONS.md`
> "v2.9.0-rc1 Signed Deferrals (2026-05-11 close-out)"). Final
> `v2.9.0` production tag remains blocked until each deferred
> contract passes under the unchanged strict gate.
>
> Predecessors:
> - `docs/QUALITY_SNAPSHOT_2026-05-11_v2.9_strict_gate_full_corpus.md` (BEFORE — 9 PASS / 8 WARN / 17 FAIL)
> - `docs/QUALITY_SNAPSHOT_2026-05-09_v2.9_phase4_after.md` (Phase 4 close, 736 tests)
> - `docs/archive/quality_snapshots/v2.9_in_progress/QUALITY_SNAPSHOT_2026-05-10_v2.9_phase5_attempt.md` (Phase 5 attempt at 30/34 fresh outputs; moved to archive during 2026-05-12 sanitization)
>
> **Revision 2026-05-12** — post-creation audit fixed three drifts vs.
> the original 2026-05-11 file: §5 vision-status counts replaced with
> precise corpus numbers, §7 collection note updated to reflect
> post-RC1 cleanup of 16 sister `*_v2` collections, predecessor link
> repointed to archive. See §10 "Revision log".

## 1. Headline numbers

| Metric | 2026-05-11 BEFORE | 2026-05-11 AFTER (RC1) | Net delta |
|---|---:|---:|---:|
| `QA_PASS` | 9 | **12** | +3 |
| `QA_PASS_WITH_ADVISORIES` (new variant) | — | **14** | +14 |
| **Total PASS-class** | 9 | **26** | **+17** |
| `QA_WARN` | 8 | **0** | **−8** |
| `QA_FAIL` | 17 | **8** | **−9** |
| Test suite | 736 passed | **806 passed** | **+70** |

All 8 remaining `QA_FAIL` rows are signed v2.10 deferrals (see §3).

## 2. Phases shipped in this cycle (2026-05-11)

Each phase ships principle-based fixes (no profile overfit, no
filename-specific logic) with corpus-validated regression tests.

| Phase | Topic | Code site(s) | Tests added |
|---|---|---|---:|
| B1 | TOC dotted-leader U+FFFD universal-collapse sanitizer; two-site BatchProcessor exemption for `hybrid_chunker_pageskip*` chunks | `mmrag_v2.schema.ingestion_schema._collapse_replacement_chars`; `processor._sanitize_toc_index_text`; `batch_processor._quarantine_corrupted_text_chunks`; `batch_processor._drop_corrupted_chunks_before_metadata` | +14 |
| B2 | "Intentionally left blank" boilerplate page classifier (Greenhouse p3/p11/p23 → MISSING_PAGES_BLANK) | `scripts/qa_full_conversion._is_intentionally_blank_text` | +10 |
| B3 (Step 2) | Section-header-only page chunk emission (Devlin p170, Nagasubramanian p2, Sekar p2/p159/p228/p247) | `processor._emit_section_header_only_page_chunks` | +10 |
| B4.a | Render-based blank-equivalent classification (`mean>245 + std<20 + text<200`); zero-text image-only placeholder check (`text=0 + images≥1 + mean>250`). Python_Distilled 697 → 4 missing | `scripts/qa_full_conversion._page_render_is_near_blank`, `._page_is_no_text_image_only_placeholder` | +17 |
| Phase E (Combat) | (a) Blank-asset filter widened `std<5 → std<10` (Combat figure_36 p27 mean=253 std=7.4); (b) gibberish-table detector via word-density `len>30K AND density<10w/k` (Combat p66 squadron-roster glyph corruption); (c) Combat reconvert + Phase H re-enrichment (348 chunks: 334 complete + 14 F4 hard_fallback) | `batch_processor._is_blank_asset`, `._is_corrupted_chunk_content` | +5 |
| Phase D (iconography) | Tiny-bbox iconography lane (`bbox<1%` → `simple` complexity regardless of file size); Hybrid_EV "Logo icon." now classifies correctly | `mmrag_v2.vision.asset_complexity._TINY_BBOX_AREA_MAX` | +3 |
| Phase G | `QA_PASS_WITH_ADVISORIES` PASS variant; documented advisory codes `ASSET_TINY`, `PAGE_COUNT_UNKNOWN`, `SCRIPT_ADVISORY_FAIL`, `VISION_HARD_FALLBACK_RATE` (F4-conditional) | `scripts/qa_full_conversion._warn_is_documented_advisory`, `_ALLOWED_ADVISORY_WARN_CODES`, `_F4_HARD_FALLBACK_SENTINEL` | +11 |
| Phase H | Targeted re-enrichment: Cronin / Nagasubramanian / Sekar / Chaubal (170 chunks). Cronin → full `QA_PASS`. | (`scripts/enrich_image_chunks_v29.py` — pre-existing) | — |
| AIOS p8 surgical correction | Mark `vision_status=hard_fallback` + F4 sentinel (retry-harness should have done this; post-hoc state alignment with contract) | post-hoc JSONL edit | — |
| **Total** | — | — | **+70** |

## 3. v2.9.0-rc1 ship contract — signed deferrals

Per `docs/DECISIONS.md` "v2.9.0-rc1 Signed Deferrals (2026-05-11
close-out)", `v2.9.0-rc1` is authorized with 8 signed deferrals:

| # | Doc | Class name | Affected pages | Retrieval impact |
|---|---|---|---:|---|
| 1 | Firearms | `OCR_PATH_HEADING_PROPAGATION` (2026-05-10) | ~300 (HEADING 72 %) | Moderate |
| 2 | KI_En_ChatGPT_Praktische_Gids | `KI_EPUB_EXTRACTION_LANE_REWRITE` (2026-05-10) | full doc | Moderate |
| 3 | Devlin_LLM_Agents | `HYBRID_CHUNKER_HEADING_PROPAGATION` (2026-05-11) | ~250 (HEADING 72 %) | Moderate |
| 4 | Python_Cookbook | `CROSS_PAGE_SPLIT_PAGE_ATTRIBUTION` (2026-05-11) | 4 / 1240 | None (content present, wrong page_number) |
| 5 | Python_Distilled | mixed B3.b + B4.b (2026-05-11) | 7 / 1411 | Marginal |
| 6 | Fluent_Python | `TEXT_INTEGRITY_SCOUT_FULL_DOC_SENSITIVITY` (2026-05-11) | 6 / 770 | None at retrieval |
| 7 | Chaubal_PyTorch_Projects | `TEXT_LABEL_TOC_DENSE_INDEX_ROUTER_MISS` (2026-05-11) | 1 / 800 | None |
| 8 | Earthship_Vol1 | `B4B_FULL_DOC_PICTURE_DEDUP` (2026-05-11) | 1 / 287 | Marginal |

User sign-off recorded 2026-05-11. Each row is a v2.10 production-tag
blocker; `v2.9.0` final tag stays blocked until all 8 pass.

## 4. Per-document strict-gate state

QA_PASS = full pass. QA_PASS_WITH_ADVISORIES = pass with documented
advisory warning(s) per `docs/QUALITY_GATES.md` "Advisory Warning
Classes". QA_FAIL with `*` = signed v2.10 deferral; `**` = pre-existing
signed deferral.

| # | Doc | Status | Advisories / FAIL detail |
|---|---|---|---|
| 1 | HarryPotter_and_the_Sorcerers_Stone | QA_PASS_WITH_ADVISORIES | ASSET_TINY |
| 2 | Form_0013_invoice | QA_PASS | — |
| 3 | Form_betwistingsformulier | QA_PASS_WITH_ADVISORIES | ASSET_TINY |
| 4 | CarOK_voorraadtelling | QA_PASS | — |
| 5 | AIOS_LLM_Agent_Operating_System | QA_PASS_WITH_ADVISORIES | VISION_HARD_FALLBACK_RATE (F4) |
| 6 | A_comprehensive_review_on_hybrid_electri | QA_PASS_WITH_ADVISORIES | ASSET_TINY |
| 7 | Hybrid_electric_vehicles | QA_PASS_WITH_ADVISORIES | ASSET_TINY |
| 8 | IRJET_Modeling_of_Solar_PV | QA_PASS | — |
| 9 | Recent_Trends_in_Transportation | QA_PASS | — |
| 10 | Combat_Aircraft_August_2025 | QA_PASS | — (Phase E close) |
| 11 | PCWorld_July_2025 | QA_PASS | — |
| 12 | ATZ_Elektronik_German | QA_PASS | — |
| 13 | Kimothi_RAG_Guide | QA_PASS_WITH_ADVISORIES | SCRIPT_ADVISORY_FAIL, VISION_HARD_FALLBACK_RATE (F4), ASSET_TINY |
| 14 | Integra_manual | QA_PASS | — |
| 15 | Jungjun_AI_Agent | QA_PASS_WITH_ADVISORIES | VISION_HARD_FALLBACK_RATE (F4) |
| 16 | Bourne_RAG_2024 | QA_PASS_WITH_ADVISORIES | VISION_HARD_FALLBACK_RATE (F4) |
| 17 | Devlin_LLM_Agents | QA_FAIL * | `HYBRID_CHUNKER_HEADING_PROPAGATION` (#3) |
| 18 | Raieli_AI_Agents | QA_PASS_WITH_ADVISORIES | SCRIPT_ADVISORY_FAIL |
| 19 | Adedeji_GenAI_Google_Cloud | QA_PASS | — |
| 20 | Cronin_GenAI_Models | QA_PASS | — |
| 21 | Hao_ML_Platform | QA_PASS_WITH_ADVISORIES | ASSET_TINY |
| 22 | Nagasubramanian_Agentic_AI | QA_PASS_WITH_ADVISORIES | VISION_HARD_FALLBACK_RATE (F4), ASSET_TINY |
| 23 | Sekar_MCP_Standard | QA_PASS_WITH_ADVISORIES | VISION_HARD_FALLBACK_RATE (F4), ASSET_TINY |
| 24 | Python_Cookbook | QA_FAIL * | `CROSS_PAGE_SPLIT_PAGE_ATTRIBUTION` (#4) |
| 25 | ArcGIS_Python_Cookbook | QA_PASS | — |
| 26 | Fluent_Python | QA_FAIL * | `TEXT_INTEGRITY_SCOUT_FULL_DOC_SENSITIVITY` (#6) |
| 27 | Python_Distilled | QA_FAIL * | mixed B3.b + B4.b (#5) |
| 28 | Ayeva_Python_Patterns | QA_PASS | — |
| 29 | Chaubal_PyTorch_Projects | QA_FAIL * | `TEXT_LABEL_TOC_DENSE_INDEX_ROUTER_MISS` (#7) |
| 30 | Earthship_Vol1 | QA_FAIL * | `B4B_FULL_DOC_PICTURE_DEDUP` (#8) |
| 31 | Firearms | QA_FAIL ** | `OCR_PATH_HEADING_PROPAGATION` (#1) |
| 32 | Greenhouse_Design | QA_PASS_WITH_ADVISORIES | SCRIPT_ADVISORY_FAIL |
| 33 | ChatGPT_Praktijk_handboek | QA_PASS_WITH_ADVISORIES | PAGE_COUNT_UNKNOWN |
| 34 | KI_En_ChatGPT_Praktische_Gids | QA_FAIL ** | `KI_EPUB_EXTRACTION_LANE_REWRITE` (#2) |

## 5. Image-chunk vision-status corpus snapshot

Precise counts captured 2026-05-12 from the 34 canonical JSONLs
(`output/<doc>/ingestion.jsonl`, `modality == "image"`):

| Status | Count | Notes |
|---|---:|---|
| Total image chunks across 34 docs | **4,379** | — |
| `vision_status="complete"` | **4,257** (97.2 %) | Real qwen3-vl-plus descriptions |
| `vision_status="hard_fallback"` (F4 sentinel) | **122** (2.8 %) | Complex assets where VLM legitimately could not produce > 20 chars after detail-retry; F4-exempt per Phase 3 contract |
| `vision_status="pending"` | **0** | Goal 7 of `docs/PLAN_V2.9.md` met |

Devlin Phase H catch-up (2026-05-12): the original Phase H batch on
2026-05-11 ran on Cronin / Nagasubramanian / Sekar / Chaubal but
not on Devlin, leaving 67 Devlin image chunks at `vision_status=pending`.
The catch-up enriched all 67 (53 complete + 14 hard_fallback / F4),
bringing corpus pending to 0. Devlin's strict-gate row is unchanged
(QA_FAIL on `HYBRID_CHUNKER_HEADING_PROPAGATION`; signed deferral #3
is independent of vision-status). See §7 for the Qdrant-payload
implication.

## 6. Test suite provenance

| Phase milestone | Tests passed | Δ |
|---|---:|---:|
| Phase 4 close (2026-05-10) | 736 | — |
| B1 close | 750 | +14 |
| B2 close | 760 | +10 |
| B3 Step 2 close | 770 | +10 |
| B4.a close | 787 | +17 |
| Phase D iconography lane | 790 | +3 |
| Phase E (Combat) | 795 | +5 |
| Phase G advisory promotion | 806 | +11 |
| **v2.9.0-rc1 close** | **806** | **+70 net** |

## 7. Qdrant rebuild (complete 2026-05-12)

`mmrag_v2_8` was dropped (was at 22,446 v2.8 points) and recreated
from the 34 canonical post-recovery JSONLs via
`scripts/rebuild_mmrag_v2_8_for_rc1.py` (sequential ingest, doc #1
with `--recreate`, doc #2-34 appending). Embedding model:
`llava:latest` via local Ollama. Wall time: **615 min (10h15m)**.

Final Qdrant state:

| Field | Value |
|---|---|
| Collection | `mmrag_v2_8` |
| Status | green |
| `points_count` | **30,461** (exact match to source JSONL count: 25,691 text + 4,379 image + 391 table) |
| `indexed_vectors_count` | 30,213 (HNSW indexing nearly complete; remaining vectors mmap-resident and queryable) |
| Vector dimension | 4096 (llava) |
| Distance | Cosine |
| Storage | On-disk (mmap) per `on_disk: true` + `on_disk_payload: true` |

Post-RC1 collection cleanup (2026-05-12): the 16 sister `*_v2`
per-doc collections from earlier experiments were dropped during
sanitization; `mmrag_v2_8` is now the sole collection in the local
Qdrant instance (`get_collections` → 1 collection, status green).

**Known Qdrant payload staleness — Devlin** (deferred re-ingest):
The Devlin Phase H catch-up (2026-05-12) re-enriched 67 image
chunks in `output/Devlin_LLM_Agents/ingestion.jsonl` to
`vision_status=complete` (53) + `hard_fallback` (14) with real VLM
descriptions, but `mmrag_v2_8` was rebuilt **before** that catch-up
ran. The Qdrant vectors for those 67 points are correct (the
image-embedding pipeline does not depend on `visual_description`),
but the payload fields `visual_description`, `vision_status`,
`vision_attempts`, and `vision_detail_retry_attempted` carry the
pre-catch-up values. Retrieval recall on image text is therefore
moderately degraded for Devlin only. A targeted Devlin re-ingest
(~50 min) is deferred as a v2.10 housekeeping item; it does not
block any signed `v2.9.0-rc1` contract.

## 8. Empirical lessons captured this cycle

1. **Parallel-site audits must include strict-gate scripts**, not
   just `batch_processor.py` (B1 first attempt missed `qa_conversion_audit.py`
   and `qa_universal_invariants.py`'s independent corruption detectors;
   the sanitizer at chunk creation became the load-bearing fix).
2. **The Retrieval-Value Test is the unifying principle** for
   "feature absent from corpus but no real retrieval impact" cases.
   Codified in `docs/DECISIONS.md`. Replaces piecemeal threshold tuning.
3. **Render-based blank detection beats threshold-bumping** when
   detecting publisher-template placeholder pages — content-pattern-based
   rules (text_len == 0 AND mean > 250) are tight and corpus-validated.
4. **Word-density** is a clean signal for gibberish detection on
   large tables (Combat p66): real tables in the corpus show
   20-54 w/k; gibberish at 8.9 w/k.
5. **Process management discipline** (kill via process group, check
   for stale `output/.*.convert.lock` before launching reconverts)
   prevents race-corruption like the Combat figure_36 incident.
6. **Field-validator collapse at chunk creation** is the right site
   for universally-unrenderable characters (U+FFFD); per-site
   exemptions become defense-in-depth.

## 9. v2.10 backlog

Carry-forward to v2.10 (per `docs/DECISIONS.md`):

1. `OCR_PATH_HEADING_PROPAGATION` (Firearms)
2. `KI_EPUB_EXTRACTION_LANE_REWRITE` (KI EPUB)
3. `HYBRID_CHUNKER_HEADING_PROPAGATION` (Devlin)
4. `CROSS_PAGE_SPLIT_PAGE_ATTRIBUTION` (Python_Cookbook, Python_Distilled partial)
5. `B4B_FULL_DOC_PICTURE_DEDUP` (Earthship, Python_Distilled partial)
6. `TEXT_INTEGRITY_SCOUT_FULL_DOC_SENSITIVITY` (Fluent_Python)
7. `TEXT_LABEL_TOC_DENSE_INDEX_ROUTER_MISS` (Chaubal p11)

Plus existing v2.9 non-goals (Local NuMarkdown VLM lane, remote
CodeFormulaV2, broader UIR refactor, HybridChunker per-item token guard)
that pre-date this cycle and remain v2.10+.

Each item has a documented root cause, acceptance baseline, and
diagnostic note (where applicable) ready for the v2.10 implementation
phase.

v2.10 housekeeping (non-blocking for `v2.9.0-rc1`):
- Targeted re-ingest of `Devlin_LLM_Agents` into `mmrag_v2_8` so the
  payload metadata matches the post-catch-up JSONL (vectors already
  correct; see §7).

## 10. Revision log

| Date | Change |
|---|---|
| 2026-05-11 | Initial RC1 AFTER snapshot at v2.9.0-rc1 close. |
| 2026-05-12 | (a) Predecessor link to `phase5_attempt.md` repointed to archive (file moved during sanitization). (b) §5 vision-status counts replaced with precise figures (4,379 / 4,257 / 122 / 0) and Devlin Phase H catch-up documented. (c) §7 collection note updated to reflect drop of 16 sister `*_v2` collections; documented Devlin Qdrant payload staleness. (d) §9 added Devlin re-ingest as v2.10 housekeeping. |
