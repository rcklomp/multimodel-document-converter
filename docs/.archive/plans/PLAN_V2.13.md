# Plan: v2.13 — Format Recovery + Local Embedder Swap

**Status:** **CLOSED + PUSHED 2026-05-22.** v2.13.0 SHIPPED —
annotated tag `v2.13.0` on origin (Gitea at 10.0.10.241) + GitHub
(rcklomp/multimodel-document-converter) at commit `021ef05`.
Phase 1 (local omlx embedder swap) and Phase 2 (OCR auto-routing)
both shipped.

**Outcome (apples-to-apples, same fixture, only embedder differs):**
omlx local `Qwen3-Embedding-8B-mxfp8` wins 6/6 axes vs cloud
`text-embedding-v4` (R@1 +2.5pp, R@5 chunk +5.4pp, R@5 doc +2.1pp,
Relevance +0.5pp, Format +3.7pp, Faithfulness +1.0pp). 3 of 6 with
meaningful margins; 3 within noise.

**Canonical AFTER snapshot:**
[`docs/QUALITY_SNAPSHOT_2026-05-22_v2.13_after.md`](QUALITY_SNAPSHOT_2026-05-22_v2.13_after.md).
**Phase 1 SWAP evidence:**
[`docs/QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_omlx_vs_dashscope.md`](QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_omlx_vs_dashscope.md).
**Decision record:** `docs/DECISIONS.md` "v2.13 Phase 1 Embedder Swap Executed — omlx Wins 6/6 Axes" (2026-05-22).

---

**(Original draft below; preserved as cycle archaeology.)**

**Status:** **Draft v0.1** (2026-05-22). Authored mid-cycle as the
plan-of-record after Phases 0 + 2 of the actual v2.13 work landed.
Phase 1 (local embedder swap) is still in flight. Plan promoted to
Draft v1.0 when the embedder soak decides swap/no-swap and the
v2.13.0 tag is staged.

**Predecessor:** [`docs/PLAN_V2.12.md`](PLAN_V2.12.md) — Draft v0.8,
CLOSED 2026-05-21 with `v2.12.0` annotated tag on commit `5a2ce18`
(`895a460`), public on both `github` (rcklomp/multimodel-document-converter)
and `origin` (Gitea at 10.0.10.241).
**Owner:** ingestion + retrieval pipeline.

---

## 1. Why this plan exists

### Thesis

v2.12 closed the retrieval-quality gap (Recall@5 chunk 66.8% →
90.2% STRETCH; Faithfulness 50.6% → 72.6%). Two known limitations
remain:

1. **Format axis** at 88.4% on the v2.12 soak vs ≥96% pin —
   concentrated in three scanned/form-class docs (Earthship,
   Firearms, CarOK). User wants Format ≥95% recovered.

2. **End-to-end latency** at ~2.05 s p99 — embedding stage
   contributes ~1.35 s of that (cloud round-trip). Local
   `Qwen3-Embedding-8B-mxfp8` benchmarked at ~180 ms p99 from the
   LAN (7× faster than cloud). If retrieval quality holds against
   v2.12, swapping cuts p99 to ~1.05 s and removes the only cloud
   dependency from the production retrieval path.

User direction (2026-05-22, post v2.12 ship): "both eventually" —
run Phase 2 (Format) and Phase 1 (embedder) in parallel; Phase 2 in
foreground while the Phase 1 rebuild chugs in the background.

### Where v2.12 ended

| Axis | v2.11.0 | **v2.12.0** | Floor | Stretch |
|---|---:|---:|---:|---:|
| Recall@1 chunk | 35.5% | **67.8%** | ≥55% ✓ | ≥70% (2pp gap) |
| Recall@5 chunk | 66.8% | **90.2%** | ≥85% ✓ | **≥90% ✓ STRETCH** |
| Recall@5 doc | 91.7% | **98.6%** | ≥95% ✓ | **≥97% ✓ STRETCH** |
| Relevance | 59.3% | **82.1%** | ≥75% ✓ | ≥85% |
| Faithfulness | 50.6% | **72.6%** | ≥70% ✓ | ≥80% |
| Format | 89.8% | 88.4% | **≥96% ✗** | ≥98% |

The only laggard is Format. The retrieval ceilings are nearly maxed
(R@5 doc 98.6% means the right doc is found 511/518 queries).

### What v2.13 starts from

- Production retrieval: cloud Dashscope `text-embedding-v4` embed →
  hybrid Qdrant top-25 (dense + BM25 sparse + RRF) → local
  ModernBERT rerank → top-5.
- BM25 index tracked at `tests/fixtures/bm25_index_v2_12.json`
  (will be updated to `bm25_index_v2_13.json` if any chunk re-extraction
  meaningfully changes the vocab).
- Same omlx-server (`http://10.0.10.246:8000`) hosts ModernBERT
  reranker AND `Qwen3-Embedding-8B-mxfp8` (registered + benchmarked
  during v2.12).
- 30-day v2.11 rollback contract ends 2026-06-19 (legacy `mmrag_v2_8`
  collection + `tests/test_retrieval_regression_v2_10.py` drop).

---

## 2. Goals

1. **Format ≥95% ex-CarOK** in the v2.13 corpus soak (see
   `docs/DECISIONS.md` "v2.13 Phase 2 CarOK" for the ex-CarOK
   rationale).
2. **Quality preserved or improved** vs v2.12.0 baseline on every
   non-Format axis. Floor-clearing of all v2.12 thresholds.
3. **Latency reduced** — if local Qwen3-Embedding-8B holds quality
   in soak, ship the swap. Target end-to-end p99 ≤ 1.5 s.
4. **No data leaves the LAN for production retrieval** if the
   embedder swap ships (currently only Dashscope embed crosses the
   network for retrieval; rerank is already local).

## 3. Phases

### Phase 0 — Carry-forward close-outs

| Item | State |
|---|---|
| 30-day rollback drop (2026-06-19) | Calendar trigger; not yet due |
| BM25 index file rename `bm25_index_v2_12.json` → `_v2_13.json` | Skipped — file content updated in place, no rename |
| Stale doc cleanup (`docs/PLAN_V2.13.md` reference in `.clinerules` etc.) | Done 2026-05-22 (this draft) |

### Phase 1 — Local embedder swap (Qwen3-Embedding-8B)

**What.** Replace cloud `text-embedding-v4` with local
`Qwen3-Embedding-8B-mxfp8` served by omlx-server, if quality holds.

**Pre-flight (done 2026-05-21):**
- Latency benchmark: ~180ms p99 single embed from the LAN.
- 4096-dim output (vs 1024-dim text-embedding-v4 — separate collection).
- Smoke ingest on Form_betwistingsformulier: 8 chunks in 4s, no errors.

**Implementation (done 2026-05-22):**
- `scripts/ingest_to_qdrant.py` gained `--provider omlx` + `embed_text_omlx()`.
- `scripts/rebuild_mmrag_v2_8_for_rc1.py` accepts omlx provider.
- Parallel collection `mmrag_v2_8__qwen3_local` rebuilding (in flight ~80%).

**Execution (in flight):**
- Background task `b3mzlgdwd` started 2026-05-22 00:23Z; doc 20+/34 at last
  check; expected total ~7-8h.
- After rebuild: full-corpus quality soak (518 queries) on the new collection
  vs v2.12.0 hybrid baseline.
- Cost estimate: ~$2-3 in qwen-max judge calls.

**Done when.**
- Soak metrics meet floors AND retrieval quality is within ±2pp of v2.12.0
  on every embedder-attributable axis (R@1, R@5 chunk, Relevance,
  Faithfulness, Recall@5 doc). Format is allowed to drift either direction
  (it's chunker-driven, not embedder-driven).
- If quality holds: flip `src/mmrag_v2/retrieval/config.py`
  `_COMPILE_DEFAULT` to point at omlx as the production embed provider;
  update `mmrag_v2.retrieval.pipeline.retrieve_hybrid_reranked` defaults.
- If quality regresses materially (>5pp loss on any embedder axis): stay
  on Dashscope; document the experiment outcome; the parallel collection
  remains for v2.14 experimentation.

**Risk.** Medium. The model is top-MTEB on multilingual but
production tuning sometimes diverges from benchmark performance. The
v2.12 production is well-validated (just shipped); regressing now
would be visible.

**Cost class.** Re-ingest: yes (~7-8h wall, $0 spend — local).
Soak: ~$2-3. No reconvert. No new code path; reuses Phase 1 module
from v2.12.

### Phase 2 — Format recovery (scanned profiles)

**What.** Earthship + Firearms had multi-column OCR damage in
v2.12 (62.5% and 68.8% Format respectively). The fix is at the
chunker level: enable Docling's `force_full_page_ocr=True` for
scanned profiles, which bypasses layout-model column-boundary
misjudgments.

**Implementation (done 2026-05-22, commits `b0dc7c6` + `cf3a909`):**

1. `PdfConversionPlan.force_full_page_ocr` field (default False;
   `True` automatically for scanned + scanned_degraded profiles).
2. `DoclingPdfAdapter` wires the flag into `EasyOcrOptions.force_full_page_ocr`.
3. **`BatchProcessor.set_conversion_plan` auto-overrides
   `ocr_mode "layout-aware" → "legacy"` when `plan.force_full_page_ocr
   = True`** so the flag actually reaches Docling's OCR (the layout-aware
   path uses its own EnhancedOCREngine and bypasses Docling). The Phase-6
   `_promote_ocr_section_headers` fallback preserves heading attribution
   in legacy mode.

**Re-extraction results (done 2026-05-22):**

| Doc | v2.12 chunks | v2.13 chunks | Δ text | Strict-gate |
|---|---:|---:|---:|---|
| Earthship_Vol1 | 1016 (548 text) | 1405 (946 text) | **+398 (+73%)** | QA_PASS ✓ |
| Firearms | 2183 (1094 text) | 2577 (1454 text) | **+360 (+33%)** | QA_PASS ✓ |

Both docs were re-extracted (Earthship 8.25 min, Firearms 23.7 min),
re-enriched via Dashscope `qwen3-vl-plus` (1571/1582 image chunks; 11
F4-sentinel hard fallbacks within the documented advisory class), and
re-ingested into the production dense + sparse collections (deleted
stale points first → fresh ingest with new chunk_ids).

**Partial soak (Earthship + Firearms only, 32 queries):**

| Doc | Format | Relevance | Faith | R@5 doc |
|---|---|---|---|---|
| Earthship | 62.5 → **68.8%** (+6.2pp ✓) | 71.9 → **75.0%** (+3.1pp ✓) | 65.6 (=) | 100% (=) |
| Firearms | 68.8 → 65.6% (−3.1pp) | 81.2 → 71.9% (−9.4pp) | 68.8 (=) | 100% (=) |

Earthship is a clear win. Firearms shows -3.1pp Format / -9.4pp Relevance
on a 16-query sample (1-2 queries differing = noise floor). Full-corpus
soak after Phase 1 will give the definitive picture.

**CarOK is a separate decision** — `docs/DECISIONS.md` "v2.13 Phase 2
CarOK Form-Class Format Penalty — Documented Limitation". Chunks are
correct; LLM judge penalizes form-shape data. Documented as known
limitation, carried forward to v2.14 (proper form-class soak judge
variant).

**Done when.**
- Both Earthship + Firearms strict-gate `QA_PASS` — DONE.
- Earthship Format ≥85% in full-corpus soak (intermediate target).
- v2.13 corpus aggregate Format ex-CarOK ≥95% in full-corpus soak.

**Risk.** Low for Earthship (decisive Phase-2 win). Medium for Firearms
(partial-soak noise; full soak decides).

**Cost class.** Reconvert: yes for 2 docs (already done — Earthship +
Firearms). Re-enrich: yes for the same 2 docs (already done; ~$16
Dashscope spend). Soak: bundled with Phase 1.

### Phase N — AFTER snapshot + v2.13.0 tag

**What.**

1. Full pytest + live retrieval-regression as final sanity (`scripts/retrieval_regression_v2_12.py`
   for the dense+rerank path; new fingerprint TBD for omlx-embed path if it ships).
2. Bump `__engine_version__` and `pyproject.toml` to `2.13.0`.
3. AFTER snapshot at `docs/QUALITY_SNAPSHOT_2026-05-22_v2.13_after.md` with
   per-axis deltas vs v2.12.0 (raw + ex-CarOK Format).
4. Layer-0/1 doc sweep (CLAUDE.md, AGENTS.md, PROJECT_STATUS.md, README.md,
   docs/README.md, ARCHITECTURE.md, CHANGELOG.md, .clinerules).
5. Stage v2.13.0 annotated tag — **NOT push by autonomous run**, user
   pushes/tags per bright-line constraint.

**Risk.** Low — Phase N is pure ceremony.
**Cost class.** Pure validation.

## 4. Acceptance Gate

Before staging v2.13.0:

1. Format ex-CarOK ≥95% in the v2.13 full-corpus soak.
2. Every embedder-attributable axis (R@1, R@5 chunk, R@5 doc,
   Relevance, Faithfulness) within ±2pp of v2.12.0 (no material
   regression).
3. Strict-gate corpus: 34 PASS / 0 WARN / 0 FAIL.
4. End-to-end p99 latency measured. Target ≤ 1.5 s if local embedder
   ships; ≤ 3.0 s otherwise (v2.12 baseline).
5. Full pytest green.
6. Both retrieval-regression fingerprints (v2.12 + new v2.13 if
   embedder swaps) committed.
7. `docs/DECISIONS.md` has decision rows for Phase 1 outcome (ship or
   no-ship local embedder) and Phase 2 outcome (Format recovery).
8. v2.13.0 tag command staged but not executed.

## 5. Out of Scope (this draft)

- Schema changes (chunk-shape unchanged since v2.7).
- Replacing Docling.
- Replacing ModernBERT reranker.
- CarOK chunk restructuring (carry-forward to v2.14).
- Form-class soak judge variant (carry-forward to v2.14).
- v2.11 carry-forwards 3a/3c/3e (VLM swap, UIR refactor, magazine
  rendered-region-crop) — still deferred.
- HyDE: ships opt-in in v2.12; v2.13 does not change that.

## 6. Decision log (this plan)

| Date | Change |
|---|---|
| 2026-05-22 | Draft v0.1 authored mid-cycle. Phases 0 + 2 (Format recovery) already executed before this plan was written — recovered in this doc + DECISIONS.md sections. Phase 1 (local embedder swap) in flight via background `mmrag_v2_8__qwen3_local` rebuild. Plan structure mirrors PLAN_V2.12: explicit floors, conditional triggers, soak-driven decisions, Phase N close-out. |

## 7. Open questions

1. **Local embedder swap outcome.** Pending Phase 1 soak. Default if quality
   holds: ship. Default if it regresses: stay on cloud + document outcome.
2. **Firearms Format regression in partial soak.** Within noise on 16
   queries but worth re-verifying on the 518-query full soak.
3. **BM25 index file naming.** Currently `bm25_index_v2_12.json`; could
   be renamed `_v2_13.json` after the Earthship/Firearms re-extraction
   updated the vocab. For simplicity, kept the v2.12 filename; the file
   itself is current.

---

**END OF DRAFT v0.1.** Promotion to Draft v1.0 happens when:
1. Phase 1 soak decides swap/no-swap.
2. Phase N close-out commit lands.
3. v2.13.0 annotated tag is staged for user push.
