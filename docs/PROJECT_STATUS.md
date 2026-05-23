# Project Status

Last updated: 2026-05-23

Purpose: fast orientation for a new coding session. Read this before deeper project docs.

## Current Objective

**v2.14 IN PROGRESS (started 2026-05-22, active 2026-05-23).** v2.13.0
SHIPPED 2026-05-22 with annotated tag `v2.13.0` staged for user push.
v2.14 opened the same day on top of the v2.13.0 retrieval stack;
plan + outcomes tracked in [`docs/PLAN_V2.14.md`](PLAN_V2.14.md)
(currently at Draft v0.5).

**v2.14 phases shipped:**
- **Phase 0 (judge calibration)** — three endpoints, all 2026-05-23:
  - 14B (morning, retired): rel 81.7% / format 90.2% TRUSTWORTHY / faith 76.1% (SUPERSEDED)
  - 27B-MTP (morning, retired): rel 82.0% / **format 70.7% RESTRICTED** / faith 78.8% — all RESTRICTED; bias direction flipped to strict-on-format (132 downgrades from qwen-max "2" to local "1"); motivated the afternoon swap (SUPERSEDED)
  - 30B-A3B-Instruct-2507 (afternoon, ACTIVE): re-cal PENDING — auto-runs once user confirms the new endpoint is live
  Report: `docs/CALIBRATION_2026-05-23_v2.14_p0_local_judge_qwen36_27b_mtp.md` (now SUPERSEDED); historical 14B at `docs/CALIBRATION_2026-05-22_v2.14_p0_local_judge.md`.
- **Phase 4a (local HyDE provider)** SHIPPED 2026-05-22 —
  `src/mmrag_v2/retrieval/hyde.py` gained `provider="vllm"` knob.
  Default vLLM model updated 2026-05-23 from 14B to `Qwen/Qwen3.6-27B-FP8`.
  Commit `0c5e818` (2026-05-23) added `chat_template_kwargs={enable_thinking: False}`
  to the vLLM payload after the Phase 0 debug revealed the 27B was
  silently dropping content via `--reasoning-parser qwen3` routing
  to `message.reasoning`. Two new bridge tests in `tests/test_hyde.py`.
  Live re-smoke: 670-char hypothesis in 8.8s (the earlier "392-char in 2s"
  smoke was the pre-thinking-discovery 14B run).
- **Phase 5 (soak disk-headroom precheck)** SHIPPED 2026-05-22.
- **Phase 6 (code-block chunking hygiene)** — **PARTIAL 2026-05-23.**
  Commit `d737147` ships block-extension policy + `partial_code` schema
  field + 3 new tests on the `_chunk_text_with_overlap` path (9/9 in
  `tests/test_code_chunking.py`; full suite 1036/1036). Fluent_Python
  re-extraction confirms the change is safe (chunk count -2.3%) but
  reveals all 547 code chunks in technical_manual docs come from
  Docling's `hybrid_chunker` / `hybrid_chunker_pagesplit`, NOT through
  the modified processor chunker. The committed change benefits the
  `scanned_book` path and adds universal observability; the actual
  Fluent_Python "truncated code" defect lives in HybridChunker and
  needs a separate design pass. **Sign-off needed** on direction —
  see PLAN_V2.14.md Phase 6 row.

**v2.14 phases pending / blocked**:
- Phase 1 (form/table extraction) — **evidence in hand 2026-05-23**:
  CarOK has 0 table chunks (Docling TSR didn't detect the inventory
  grid). Escalation to VLM fallback (3a carry-forward) needs user
  sign-off per the handoff "stop and ask" item #3.
- Phase 2 (targeted HyDE bridging) — depends on Phase 6 + Phase 1
  landing so per-doc deficits can be re-measured cleanly.
- Phase 3 (rollback drop) — time-gated 2026-06-19; needs explicit
  "no regression, drop it" sign-off + 90-day cold-storage snapshot
  per Draft v0.5.
- Phase 4b (local judge in soak) — **scope evaporated 2026-05-23**:
  the Draft v0.3 "Format-axis only" carve-out was predicated on the
  14B's 90.2% format TRUSTWORTHY verdict. 27B dropped format to 70.7%
  (RESTRICTED). Remaining viable: prompt A/B + tie-breaker harness.
- Phase 4c (gen-provider vllm) — safe; ready to wire.
- Phase 4d (tie-breaker) — code-only; ready to wire.
- Phase 4e (1500-query Format-only demo) — **DROPPED 2026-05-23**:
  predicate failed (Format not TRUSTWORTHY on 27B).
- Phase N (close-out + 2.14.0 tag).

---

**v2.13.0 SHIPPED 2026-05-22.** Annotated tag `v2.13.0` staged
locally for user push to GitHub + Gitea. Two parallel workstreams
closed this cycle on top of the v2.12.0 retrieval stack:

- **Phase 1 (local embedder swap) SHIPPED 2026-05-22** —
  `Qwen3-Embedding-8B-mxfp8` via omlx-server replaces cloud
  `text-embedding-v4` as the production embedder. Apples-to-apples
  shootout (same fixture, only embedder differs) won 6/6 axes
  (R@1 +2.5pp, R@5 chunk +5.4pp, R@5 doc +2.1pp, Relevance +0.5pp,
  Format +3.7pp, Faithfulness +1.0pp). Production dense collection
  flipped to `mmrag_v2_8__qwen3_local` (4096-dim, 31,371 pts).
  Dashscope collection retained as 30-day rollback baseline through
  2026-06-19.

- **Phase 2 (Format recovery) SHIPPED 2026-05-22** — commits
  `b0dc7c6` (v2.13 P1 infra: omlx provider + force_full_page_ocr
  scaffold) + `cf3a909` (batch_processor auto-routes scanned to
  legacy OCR when force_full_page_ocr is set) + `ef2925d` (Earthship
  + Firearms re-extraction outcome + CarOK form-class decision +
  BM25 index rebuild). Earthship: +6.2pp Format on partial soak;
  CarOK documented as judge-calibration limitation for v2.14.

**v2.13.0 ship gates:**
- Test suite: 1033 passed / 16 skipped / 0 failed
- v2.13 retrieval fingerprint: 20/20 PASS against live stack
- Engine 2.13.0, schema 2.7.0 (unchanged), `pyproject.toml` synced

Plan history: [`docs/PLAN_V2.13.md`](PLAN_V2.13.md) (CLOSED 2026-05-22).
Canonical baseline: [`docs/QUALITY_SNAPSHOT_2026-05-22_v2.13_after.md`](QUALITY_SNAPSHOT_2026-05-22_v2.13_after.md).
P1 evidence: [`docs/QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_omlx_vs_dashscope.md`](QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_omlx_vs_dashscope.md).

## v2.13.0 quality numbers — apples-to-apples vs v2.12.0 (same fixture, only embedder differs)

| Axis | v2.12.0 (cloud dashscope) | **v2.13.0 (local omlx)** | Δ (pp) |
|---|---:|---:|---:|
| Recall@1 chunk | 55.0% | **57.5%** | **+2.5** |
| Recall@5 chunk | 72.6% | **78.0%** | **+5.4** |
| Recall@5 doc | 93.1% | **95.2%** | +2.1 |
| Relevance | 74.1% | **74.6%** | +0.5 |
| **Format** | 89.2% | **92.9%** | **+3.7** |
| Faithfulness | 65.9% | **66.9%** | +1.0 |

omlx wins **6/6** axes; 3 with meaningful margins (R@1, R@5 chunk,
Format). Numbers are on the v2.13 P1 fresh fixture (post-v2.13-P2
ingestion); absolute values are not directly comparable to the
v2.12.0 6/6-axis canonical numbers below — those were on the v2.11
baseline fixture. The DELTA between providers (this table) is what
quantifies v2.13.0's Phase 1 contribution.

For predecessor reference: v2.12.0 vs v2.11.0 baseline reads were
R@1 67.8% / R@5c 90.2% STRETCH / R@5d 98.6% STRETCH / Relevance
82.1% / Format 88.4% / Faithfulness 72.6%.

## Production retrieval stack (v2.13.0)

```
query
  ├─ dense  : omlx Qwen3-Embedding-8B-mxfp8 → mmrag_v2_8__qwen3_local (4096-dim)
  └─ sparse : BM25 → mmrag_v2_8__bm25_sparse
  → RRF fusion (k=60, equal weights), top-25 candidates per leg
  → rerank (local gte-reranker-modernbert-base-mlx via omlx-server)
  → top-5 return

End-to-end p99 latency: ~1.6 s (estimated; cloud→LAN embed swap
saves ~400 ms over v2.12.0).
Per-query cost: $0 (no cloud calls on the retrieval path).
External dependencies: omlx-server only (LAN; privacy + offline-capable).
```

Tag tree on GitHub + Gitea:

```
v2.8.0       (2026-05-04, 645ab2b)
v2.9.0-rc1   (2026-05-12, 3e06d1b)  — v2.9 ship state, 8 deferrals
v2.10.0-rc1  (2026-05-16, 82c3639)  — all 8 closed corpus-wide
v2.10.0      (2026-05-16, db6527c)  — chunker baseline + soak evidence
v2.11.0      (2026-05-20, c2a461c)  — embedder swap (Dashscope text-embedding-v4)
v2.12.0      (2026-05-21, 5a2ce18)  — retrieval stack (hybrid + ModernBERT rerank)
v2.13.0      (PENDING, staged for user push) — local embedder swap (omlx Qwen3-Embedding-8B) + OCR auto-routing
```

**Active canonical baseline:**
[`docs/QUALITY_SNAPSHOT_2026-05-22_v2.13_after.md`](QUALITY_SNAPSHOT_2026-05-22_v2.13_after.md)
— v2.13.0 AFTER snapshot. Full numbers + per-phase contributions.

Predecessor canonical (kept for delta reproducibility):
[`docs/QUALITY_SNAPSHOT_2026-05-21_v2.12_after.md`](QUALITY_SNAPSHOT_2026-05-21_v2.12_after.md)
— v2.12.0 AFTER snapshot.

**v2.13 phase reports:**

- [`docs/QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_omlx_vs_dashscope.md`](QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_omlx_vs_dashscope.md) — **Phase 1 SWAP evidence** (apples-to-apples 6/6-axis win)
- [`docs/QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_omlx.md`](QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_omlx.md) — Phase 1 omlx per-doc + weakest queries
- [`docs/QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_dashscope_baseline.md`](QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_dashscope_baseline.md) — Phase 1 dashscope per-doc + weakest queries

**v2.12 phase reports (predecessor — kept for reference):**

- [`docs/QUALITY_SNAPSHOT_2026-05-21_v2.12_p1_cloud.md`](QUALITY_SNAPSHOT_2026-05-21_v2.12_p1_cloud.md) — Phase 1 cloud-rerank shootout
- [`docs/QUALITY_SNAPSHOT_2026-05-21_v2.12_p1_omlx.md`](QUALITY_SNAPSHOT_2026-05-21_v2.12_p1_omlx.md) — Phase 1 local-rerank shootout (winner)
- [`docs/QUALITY_SNAPSHOT_2026-05-21_v2.12_p2_hybrid.md`](QUALITY_SNAPSHOT_2026-05-21_v2.12_p2_hybrid.md) — Phase 2 hybrid+rerank
- [`docs/QUALITY_SNAPSHOT_2026-05-21_v2.12_p3_hyde.md`](QUALITY_SNAPSHOT_2026-05-21_v2.12_p3_hyde.md) — Phase 3 HyDE measurement (deltas in noise; opt-in only)

**Predecessor baselines (kept for delta reproducibility):**

- [`docs/QUALITY_SNAPSHOT_2026-05-20_v2.11_soak_qwen3.md`](QUALITY_SNAPSHOT_2026-05-20_v2.11_soak_qwen3.md)
  — v2.11.0 baseline. The 518-query × 259-chunk fixture every v2.12 soak ran against.
- [`docs/QUALITY_SNAPSHOT_2026-05-16_v2.10_after.md`](QUALITY_SNAPSHOT_2026-05-16_v2.10_after.md)
  — v2.10 corpus baseline (34/34 PASS strict gate — unchanged in v2.11 and v2.12 because those cycles touched retrieval-side only, not extraction/chunking/validation).

## v2.13 phase summary

| Phase | What | Outcome |
|---|---|---|
| 1 | Local embedder swap (omlx Qwen3-Embedding-8B-mxfp8 vs cloud text-embedding-v4) | **omlx wins 6/6 axes** on apples-to-apples soak; SWAP |
| 2 | OCR auto-routing (`force_full_page_ocr` honored end-to-end) | Earthship +6.2pp Format; Firearms within noise; CarOK documented as judge-calibration limitation |
| N | AFTER snapshot + version bump + docs + stage tag | (this commit cycle) |

## v2.12 phase summary (predecessor)

| Phase | What | Outcome |
|---|---|---|
| 0 | `content/refined_content` preference fix in `ingest_to_qdrant.py` | IRJET +15.6pp Format; Earthship + CarOK rolled to v2.13 |
| 1 | Cross-encoder reranker shootout (cloud `gte-rerank` vs local ModernBERT) | Local ModernBERT wins 4/4 embedder axes; production default flipped |
| 2 | Hybrid retrieval (BM25 + dense + RRF) | All embedder floors cleared; R@5 chunk + doc both hit stretch |
| 3 | HyDE measurement | All deltas in noise; ships opt-in only |

## Qdrant collections (current state)

| Collection | Vectors | Status | Role |
|---|---:|---|---|
| `mmrag_v2_8__qwen3_local` | 31,371 (4096-dim cosine, Qwen3-Embedding-8B) | green | **Production dense (v2.13.0)** |
| `mmrag_v2_8__bm25_sparse` | 26,381 (BM25 sparse) | green | **Production sparse** (v2.12+) |
| `mmrag_v2_8__qwen3_dashscope` | 31,371 (1024-dim cosine, text-embedding-v4) | green | **30-day rollback baseline** (drop after 2026-06-19 if unused) |
| `mmrag_v2_8` | 30,454 (4096-dim cosine, llava) | green | v2.10 legacy rollback (deletion candidate) |

## Active Model/Endpoint State

Do not print or commit API keys.

**Production text-retrieval embedder (v2.13.0):**

- provider: omlx-server (OpenAI-compatible local API)
- model: `Qwen3-Embedding-8B-mxfp8` (4096-dim, MLX FP8-quantized)
- endpoint: `http://10.0.10.246:8000/v1/embeddings`
- env var: `MLX_API_KEY` (required for retrieval through omlx)
- runtime: Apple Silicon (Mac Mini), ~80 ms LAN per query
- rollback (through 2026-06-19): provider Dashscope, model
  `text-embedding-v4` (1024-dim) against `mmrag_v2_8__qwen3_dashscope`;
  flip the embedder defaults in `src/mmrag_v2/retrieval/pipeline.py`
  back to `embed_provider="dashscope"` + `embed_model="text-embedding-v4"`
  if a regression surfaces — no re-ingestion needed since both
  collections are kept hot.

**Production cross-encoder reranker (NEW in v2.12):**

- provider: omlx-server (OpenAI-compatible local API)
- model: `gte-reranker-modernbert-base-mlx` (~150M params, MLX-quantized)
- endpoint: `http://10.0.10.246:8000/v1/rerank`
- env var: `MLX_API_KEY` (required for retrieval through `retrieve_hybrid_reranked()`)
- runtime: Apple Silicon, ~15 ms per (query, doc) pair, ~0.55 s p99 for K=25

**Synthetic soak judge:**

- provider: Dashscope, model `qwen-max` (used for both query generation and judging)
- v2.14 Phase 0/4 additions: local-LLM judge candidate at the GX10 vLLM endpoint below; ship-gate go/no-go on Relevance + Faithfulness axes stays on cloud `qwen-max` per the Phase 4 "leniency trap" guardrail in `PLAN_V2.14.md`

**Local LLM endpoint (v2.14 Phase 4a, ACTIVE model swap pending live verification 2026-05-23 afternoon):**

- provider: vLLM (OpenAI-compatible)
- model (code default updated 2026-05-23 afternoon): `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` (MoE — 30B total, ~3B active per pass; pure Instruct, no `<think>` mode)
- served-model aliases (per recipe): `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` (full) or `qwen3-30b-a3b` (short)
- endpoint: `http://10.0.10.239:8000/v1/chat/completions` (Asus Ascent GX10 = DGX Spark clone, Grace-Blackwell GB10, 128 GB unified memory)
- container: `vllm/vllm-openai:v0.20.0-aarch64-cu130-ubuntu2404` (proven aarch64 image, reused from 27B recipe)
- max context: 256K configured (32K typically sufficient for HyDE + judge)
- code default: `src/mmrag_v2/retrieval/hyde.py` `VLLM_DEFAULT_MODEL` (call with `provider="vllm"`)
- env var: `VLLM_API_KEY` (optional; endpoint runs unauthenticated by default)
- canonical recipe + path-of-pain notes: `memory/project_v2_14_gx10_30b_a3b_swap.md`
- guardrails before any future swap: `memory/feedback_gx10_deployment_guardrails.md` (5-point hard checklist)
- **Cloud fallback (Phase 4 Resilience, designated 2026-05-23):** Dashscope `qwen3-max` when GX10 is unavailable. Wiring is a pending change to `generate_with_fallback` in `hyde.py` (currently degrades to literal query).
- **Endpoint swap history (all 2026-05-23):**
  - morning: `Qwen2.5-14B-Instruct` → `Qwen/Qwen3.6-27B-FP8` + native MTP=3 (retired)
  - afternoon: `Qwen/Qwen3.6-27B-FP8` → `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` (active, awaiting `docker run` execution)
- **Phase 0 calibration history:**
  - 14B (2026-05-22): rel 81.7% / format 90.2% TRUSTWORTHY / faith 76.1% — SUPERSEDED
  - 27B-MTP (2026-05-23 morning): rel 82.0% / **format 70.7%** / faith 78.8% — all RESTRICTED, Format-axis collapsed vs 14B, motivated this afternoon's swap — SUPERSEDED
  - 30B-A3B-Instruct-2507 (2026-05-23 afternoon): PENDING — re-cal auto-runs once user confirms the new endpoint is live

**Production VLM (image enrichment — unchanged from v2.11):**

- preferred cloud: Dashscope `qwen3-vl-plus`
- local fallback: `NuMarkdown-8B-Thinking-mlx-8bits` on `http://10.0.10.246:8000/v1`

**Future candidates (v2.14 carry-forwards — alignment per PLAN_V2.14.md Draft v0.5):**

- Phase 1 — form/table layout extraction recovery for CarOK and other form-class docs (REDEFINED Draft v0.3 from earlier `format_form` judge axis proposal: fix extraction, not rubric; see DECISIONS.md "v2.13 Phase 2 CarOK")
- Phase 2 — targeted HyDE bridging for code + minority-language queries (REDEFINED Draft v0.3 from earlier per-doc embedder routing proposal: bridge at query time via local HyDE, no parallel embedder collections)
- Phase 6 — code-block chunking hygiene (NEW in Draft v0.3): fix mid-block truncation in Python_Cookbook / Fluent_Python / ArcGIS / Ayeva
- VLM swap (3a from v2.11) — promoted to Phase 1 VLM-assisted table parse fallback
- UIR refactor (3c, PAUSED for user signoff)
- Magazine rendered-region-crop (3e from v2.11) — deferred with soak-data rationale

## Current Quality Summary

Source of truth for v2.13.0:
[`docs/QUALITY_SNAPSHOT_2026-05-22_v2.13_after.md`](QUALITY_SNAPSHOT_2026-05-22_v2.13_after.md).
Strict-gate corpus state unchanged from v2.10: **34 PASS / 0 WARN /
0 FAIL** — extraction/chunking/validation are untouched by v2.11 +
v2.12 + v2.13 (all three cycles changed only the retrieval side
plus, in v2.13 P2, OCR-routing).

**Current local test suite: 1033 passed, 16 skipped, 0 failed**
after the v2.13 cycle (+1 over v2.12's 1032). v2.13 added no new
test files — the v2.13 P1 swap reused the existing mock-driven
`tests/test_retrieval_pipeline.py` (17 tests, updated to pin
explicit dashscope provider since the library defaults flipped to
omlx). v2.13 P2's OCR auto-routing is covered by the existing
`tests/test_pdf_conversion_plan.py` adapter-guard tests.

New v2.13 fingerprint:

- `tests/fixtures/retrieval_regression_v2_13_hybrid.json` —
  v2.13.0 production retrieval shape (omlx + Qwen3-Embedding-8B +
  mmrag_v2_8__qwen3_local + ModernBERT rerank). Capture/verify via
  `scripts/retrieval_regression_v2_13.py`. 20/20 PASS at ship.

## v2.13 commits (reverse chronological)

- `(staging)` (2026-05-22) — Phase N close-out: version bump
  2.12.0→2.13.0, AFTER snapshot, layer-0/1 docs sweep, v2.13.0
  annotated tag staged for user push.
- `4af0038` (2026-05-22) — Phase 1 SWAP decision committed
  (`docs/DECISIONS.md` + canonical comparison snapshot + omlx and
  dashscope per-doc soaks).
- `092abc3` (2026-05-22) — repo-wide doc sanity sweep (flip current
  anchor from v2.10/v2.11 → v2.12 SHIPPED + v2.13 in-progress).
- `ef2925d` (2026-05-22) — Phase 2 OCR auto-routing outcome +
  CarOK form-class decision + BM25 index rebuild.
- `cf3a909` (2026-05-22) — Phase 2 fix: batch_processor auto-routes
  scanned to legacy OCR when `force_full_page_ocr=True`.
- `b0dc7c6` (2026-05-22) — Phase 1 infra: omlx embed provider added
  to ingest, `force_full_page_ocr` field added to `PdfConversionPlan`.

## v2.12 commits (reverse chronological — predecessor)

- `5a2ce18` (2026-05-21) — v2.12.0 release commit (version 2.12.0).
- `181a5a1` (2026-05-21) — Phase 3 close-out: HyDE ships opt-in.
- `51ab67c` (2026-05-21) — Phase 2 close-out: hybrid is production default.
- `d7a0bfd` (2026-05-21) — Phase 2+3 infrastructure (sparse + RRF + HyDE).
- `65a5ba7` (2026-05-21) — Phase 1 close-out: local ModernBERT wins 4/4.
- `988fcaf` (2026-05-21) — Phase 1 retrieval module + tests + soak wiring.
- `0d731b1` (2026-05-21) — Phase 0: `content/refined_content` preference fix.

## Active Engineering Direction

**v2.14 is the active cycle** (started 2026-05-22; v2.13.0 shipped
the same day). Authoritative scope + ordering in
[`docs/PLAN_V2.14.md`](PLAN_V2.14.md) (Draft v0.5 as of 2026-05-23).
The summary table below reflects current Draft v0.5 framing; in case
of drift, the plan file wins.

## v2.14 Phase Status (mirrors PLAN_V2.14.md §"Phase outcomes")

| Phase | Topic | Status (2026-05-23) |
|---|---|---|
| 0 | Local-judge calibration | 14B SUPERSEDED. 27B-MTP SUPERSEDED 2026-05-23 afternoon (all axes RESTRICTED, format collapsed to 70.7%). **30B-A3B-Instruct-2507 re-cal PENDING** — auto-runs once user confirms the new endpoint is live. |
| 1 | Form/Table layout extraction recovery | PENDING — **evidence in hand**: CarOK has 0 table chunks (TSR didn't detect). VLM-fallback escalation needs user sign-off. |
| 2 | Targeted HyDE bridging for code + minority languages | PENDING — depends on Phase 6 + Phase 1 landing. |
| 3 | 30-day dashscope-rollback drop | PENDING — time-gated decision point 2026-06-19. |
| 4a | Local HyDE provider | SHIPPED 2026-05-22; default model 14B → 27B-MTP (morning 2026-05-23) → 30B-A3B-Instruct-2507 (afternoon 2026-05-23). Harness `chat_template_kwargs.enable_thinking=False` fix (commit `0c5e818`) kept across swaps as defensive no-op. Dashscope `qwen3-max` named as cloud fallback (wiring pending). |
| 4b | Local judge in soak (was Format-only) | **DOWNGRADED 2026-05-23** — Format axis no longer trustworthy on 27B. Remaining viable: prompt A/B + tie-breaker. |
| 4c | Local query gen | PENDING — safe; ready to wire `--gen-provider vllm`. |
| 4d | Tie-breaker harness | PENDING — code-only; ready to wire. |
| 4e | 1500-query Format-only demo soak | **DROPPED 2026-05-23** — predicate failed. |
| 5 | Soak disk-headroom precheck | SHIPPED 2026-05-22. |
| 6 | Code-block chunking hygiene | **PARTIAL** — commit `d737147` ships block-extension + `partial_code` schema field + 3 new tests on the `_chunk_text_with_overlap` path. Discovery 2026-05-23: production technical_manual chunks come from Docling's `hybrid_chunker`, NOT through that path. Committed change covers `scanned_book` + observability; Fluent_Python defect needs a HybridChunker-layer pass. **Sign-off needed** on direction. |
| N | Cycle close-out + 2.14.0 tag | PENDING — terminal. |

## Other Carry-Forwards

- **30-day dashscope-rollback drop (Phase 3).** Decision point
  2026-06-19: drop `mmrag_v2_8__qwen3_dashscope` (v2.13 baseline)
  if no v2.13 rollback fired during the window, and
  `mmrag_v2_8` (v2.10 legacy llava). Skip if regression reports
  arrive during the window. Draft v0.5 added a 90-day cold-storage
  snapshot policy before deletion.
- **v2.11/v2.12 carry-forwards still open:**
  - 3a (VLM swap) — **promoted to Phase 1 fallback** for VLM-assisted
    table parse on form-class docs.
  - 3c (UIR refactor) — still PAUSED for user signoff.
  - 3e (magazine rendered-region-crop) — deferred with soak-data
    rationale (image-axis perf is OK without it).
- HyDE stays opt-in by default unless a future use case warrants the
  +1 s latency (and now with Phase 4a, the +1 s is also $0/call for
  the local path).

## Must-Respect Constraints

- Python 3.10 only.
- Batch size must stay at or below 10 pages.
- Do not use `--profile-override` for acceptance runs.
- Do not add filename-specific or document-specific quality rules.
- OCR handles text; VLMs describe visuals only.
- BBoxes must remain normalized integer `[0,1000]`.
- Acceptance requires `GATE_PASS` plus `UNIVERSAL_PASS` across the smoke matrix.
- Production text-retrieval embedder is omlx/`Qwen3-Embedding-8B-mxfp8`
  against `mmrag_v2_8__qwen3_local` (v2.13.0 default). Production
  reranker is local `gte-reranker-modernbert-base-mlx` via omlx-server.
  Production retrieval flow is `mmrag_v2.retrieval.retrieve_hybrid_reranked()`.
- Dashscope text-embedding-v4 against `mmrag_v2_8__qwen3_dashscope`
  is the 30-day rollback baseline through 2026-06-19; not the default
  for any new code path. Ollama/llava lane is legacy; do not use as a
  comparison baseline.
- `MLX_API_KEY` env var is required for production retrieval (omlx
  embedder + omlx reranker). `DASHSCOPE_API_KEY` is required only
  for synthetic-soak judge + query generation and for the dashscope
  rollback path. Test-suite skip-gates handle the unset case for CI.
