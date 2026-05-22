# Project Status

Last updated: 2026-05-22

Purpose: fast orientation for a new coding session. Read this before deeper project docs.

## Current Objective

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

**Production VLM (image enrichment — unchanged from v2.11):**

- preferred cloud: Dashscope `qwen3-vl-plus`
- local fallback: `NuMarkdown-8B-Thinking-mlx-8bits` on `http://10.0.10.246:8000/v1`

**Future candidates (v2.14 carry-forwards):**

- form-class `format_form` judge axis (CarOK calibration limitation, see DECISIONS.md "v2.13 Phase 2 CarOK")
- language-aware embedder routing (ATZ_Elektronik German -12.5pp R@1 with omlx; v2.13 P1 per-doc breakdown)
- code-doc embedder choice (Python_Cookbook, IRJET, Hybrid_electric, Greenhouse regress 6-12pp R@1 with omlx)
- **local LLM integration on Asus Ascent GX10 (DGX Spark clone)** — proposed Qwen3.6-35B-A3B-FP8 as experimentation accelerator (judge / HyDE / query generation); hold until v2.14 scoping
- VLM swap (3a from v2.11), UIR refactor (3c, PAUSED)

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

v2.13.0 SHIPPED 2026-05-22. **Next ship state: v2.14** (scoped after
the 30-day rollback window closes 2026-06-19). Carry-forwards below.

## v2.14 Carry-Forwards

1. **Form-class `format_form` judge axis.** CarOK Format penalty is
   judge-calibration, not content (DECISIONS.md "v2.13 Phase 2
   CarOK"). v2.14 amends the soak protocol with a form-class-aware
   format rubric.
2. **Language-aware embedder routing.** ATZ_Elektronik German
   underperforms omlx by -12.5pp R@1. Consider per-doc embedder
   selection if regression deepens with future German content.
3. **Code-doc embedder choice.** Python_Cookbook, IRJET,
   Hybrid_electric, Greenhouse regress 6-12pp R@1 with omlx; offset
   by wins elsewhere but worth investigating a code-specialized
   embedder lane.
4. **Local LLM integration on Asus Ascent GX10.** Proposed
   `Qwen3.6-35B-A3B-FP8` as experimentation accelerator (judge,
   HyDE, query generation). Hold as future-use note until v2.14
   scoping; primary use: free local soaks for hyperparameter sweeps
   that v2.13 budget can't afford.
5. **30-day rollback drops on 2026-06-19.** Decision point: drop
   `mmrag_v2_8__qwen3_dashscope` (v2.12 baseline) if no v2.13
   rollback fired, and `mmrag_v2_8` (v2.10 legacy llava).
6. **v2.11/v2.12 carry-forwards still open:** 3a (VLM swap), 3c
   (UIR refactor, PAUSED), 3e (magazine rendered-region-crop). HyDE
   stays opt-in unless a future use case warrants the +1s latency.

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
