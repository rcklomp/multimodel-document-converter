# Project Status

Last updated: 2026-05-22

Purpose: fast orientation for a new coding session. Read this before deeper project docs.

## Current Objective

**v2.13 cycle IN PROGRESS (started 2026-05-22)** — `v2.12.0` SHIPPED
2026-05-21 (annotated tag `5a2ce18`, public on both GitHub and Gitea
at 10.0.10.241). v2.12.0 brought the retrieval-quality stack
(hybrid + ModernBERT rerank) — cumulative +32pp Recall@1 over
v2.11.0; two axes hit STRETCH. The only remaining laggard from v2.12
was Format (88.4% vs ≥96% pin), concentrated in three scanned/form
docs.

**v2.13 work in flight (autonomous):**

- **Phase 2 (Format recovery) SHIPPED 2026-05-22** — commits
  `b0dc7c6` (v2.13 P1 infra: omlx provider + force_full_page_ocr
  scaffold) + `cf3a909` (batch_processor auto-routes scanned to
  legacy OCR when force_full_page_ocr is set) + `ef2925d` (Earthship
  + Firearms re-extraction outcome + CarOK form-class decision +
  BM25 index rebuild).
  - Earthship: 1016 → 1405 chunks (+398 text); partial soak Format
    62.5 → 68.8% (+6.2pp). Strict-gate QA_PASS clean (was QA_PASS_WITH_ADVISORIES).
  - Firearms: 2183 → 2577 chunks (+360 text). Partial soak shows
    -3.1pp Format / -9.4pp Relevance within 16-query sample noise.
  - CarOK: documented as judge-calibration limitation, carry to v2.14.

- **Phase 1 (local Qwen3-Embedding-8B swap) IN FLIGHT** — background
  parallel-collection rebuild `mmrag_v2_8__qwen3_local` running on
  the omlx-server. Latency benchmarked at ~180ms p99 (vs cloud
  ~1.35s — 7× faster). If full-corpus soak holds quality vs v2.12.0
  baseline, swap ships; if it regresses, stay on cloud. Currently on
  doc ~20/34 of the 34-doc canonical rebuild.

Plan: [`docs/PLAN_V2.13.md`](PLAN_V2.13.md) (Draft v0.1).

## v2.12.0 quality numbers (vs v2.11.0 baseline)

| Axis | v2.11.0 baseline | **v2.12.0** | Δ (pp) | Floor | Stretch |
|---|---:|---:|---:|---:|---:|
| Recall@1 chunk | 35.5% | **67.8%** | +32.3 | ≥55% ✓ | ≥70% (2.2pp gap) |
| Recall@5 chunk | 66.8% | **90.2%** | +23.4 | ≥85% ✓ | **≥90% ✓ STRETCH** |
| Recall@5 doc | 91.7% | **98.6%** | +6.9 | ≥95% ✓ | **≥97% ✓ STRETCH** |
| Relevance | 59.3% | **82.1%** | +22.8 | ≥75% ✓ | ≥85% |
| Faithfulness | 50.6% | **72.6%** | +22.0 | ≥70% ✓ | ≥80% |
| **Format** | 89.8% | 88.4% | −1.4 | **≥96% ✗** | ≥98% |

Five of six axes pass their floors. Format remains below ≥96% pin
(chunk-level OCR/form-shape damage; retrieval-side work can't fix
it; v2.13 carry-forward).

## Production retrieval stack (v2.12.0)

```
query
  → embed (Dashscope text-embedding-v4)
  → dense Qdrant top-25 (mmrag_v2_8__qwen3_dashscope)
  + sparse Qdrant top-25 (mmrag_v2_8__bm25_sparse, BM25)
  → RRF fusion (k=60, equal weights)
  → top-25 candidates
  → rerank (local gte-reranker-modernbert-base-mlx via omlx-server)
  → top-5 return

End-to-end p99 latency: ~2.05 s (within 3.0 s budget).
Per-query cost: ~$0.001 (Dashscope embed only).
```

Tag tree on GitHub + Gitea:

```
v2.8.0       (2026-05-04, 645ab2b)
v2.9.0-rc1   (2026-05-12, 3e06d1b)  — v2.9 ship state, 8 deferrals
v2.10.0-rc1  (2026-05-16, 82c3639)  — all 8 closed corpus-wide
v2.10.0      (2026-05-16, db6527c)  — chunker baseline + soak evidence
v2.11.0      (2026-05-20, c2a461c)  — embedder swap (Dashscope text-embedding-v4)
v2.12.0      (2026-05-21, 5a2ce18)  — retrieval stack (hybrid + ModernBERT rerank)
v2.13.0      (PENDING, latest main) — Format recovery + local embedder candidate
```

**Active canonical baseline:**
[`docs/QUALITY_SNAPSHOT_2026-05-21_v2.12_after.md`](QUALITY_SNAPSHOT_2026-05-21_v2.12_after.md)
— v2.12.0 AFTER snapshot. Full numbers + per-phase contributions.

**Phase soak reports retained (one per phase):**

- [`docs/QUALITY_SNAPSHOT_2026-05-21_v2.12_p1_cloud.md`](QUALITY_SNAPSHOT_2026-05-21_v2.12_p1_cloud.md) — Phase 1 cloud-rerank shootout
- [`docs/QUALITY_SNAPSHOT_2026-05-21_v2.12_p1_omlx.md`](QUALITY_SNAPSHOT_2026-05-21_v2.12_p1_omlx.md) — Phase 1 local-rerank shootout (winner)
- [`docs/QUALITY_SNAPSHOT_2026-05-21_v2.12_p2_hybrid.md`](QUALITY_SNAPSHOT_2026-05-21_v2.12_p2_hybrid.md) — Phase 2 hybrid+rerank
- [`docs/QUALITY_SNAPSHOT_2026-05-21_v2.12_p3_hyde.md`](QUALITY_SNAPSHOT_2026-05-21_v2.12_p3_hyde.md) — Phase 3 HyDE measurement (deltas in noise; opt-in only)

**Predecessor baselines (kept for delta reproducibility):**

- [`docs/QUALITY_SNAPSHOT_2026-05-20_v2.11_soak_qwen3.md`](QUALITY_SNAPSHOT_2026-05-20_v2.11_soak_qwen3.md)
  — v2.11.0 baseline. The 518-query × 259-chunk fixture every v2.12 soak ran against.
- [`docs/QUALITY_SNAPSHOT_2026-05-16_v2.10_after.md`](QUALITY_SNAPSHOT_2026-05-16_v2.10_after.md)
  — v2.10 corpus baseline (34/34 PASS strict gate — unchanged in v2.11 and v2.12 because those cycles touched retrieval-side only, not extraction/chunking/validation).

## v2.12 phase summary

| Phase | What | Outcome |
|---|---|---|
| 0 | `content/refined_content` preference fix in `ingest_to_qdrant.py` | IRJET +15.6pp Format; Earthship + CarOK rolled to v2.13 |
| 1 | Cross-encoder reranker shootout (cloud `gte-rerank` vs local ModernBERT) | Local ModernBERT wins 4/4 embedder axes; production default flipped |
| 2 | Hybrid retrieval (BM25 + dense + RRF) | All embedder floors cleared; R@5 chunk + doc both hit stretch |
| 3 | HyDE measurement | All deltas in noise; ships opt-in only |
| 4 | NOT triggered | — |
| N | AFTER snapshot + version bump + docs + stage tag | (this commit cycle) |

## Qdrant collections (current state)

| Collection | Vectors | Status | Role |
|---|---:|---|---|
| `mmrag_v2_8__qwen3_dashscope` | 30,588 (1024-dim cosine) | green | **Production dense** |
| `mmrag_v2_8__bm25_sparse` | 25,623 (BM25 sparse) | green | **Production sparse** (new in v2.12) |
| `mmrag_v2_8` | 30,454 (4096-dim cosine, llava) | green | v2.10 rollback baseline (drop 2026-06-19) |

## Active Model/Endpoint State

Do not print or commit API keys.

**Production text-retrieval embedder:**

- provider: Dashscope (OpenAI-compatible)
- model: `text-embedding-v4` (1024-dim)
- endpoint: `https://dashscope-intl.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding`
- env var: `DASHSCOPE_API_KEY` (required for ingestion, retrieval, soak)

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

**Future candidates (v2.13 carry-forwards):**

- local embedder: `Qwen3-Embedding-8B-mxfp8` (registered on omlx-server, benchmarked at 7× faster than cloud — sub-1s end-to-end retrieval possible)
- VLM swap (3a from v2.11), UIR refactor (3c, PAUSED)

## Current Quality Summary

Source of truth for v2.12.0:
[`docs/QUALITY_SNAPSHOT_2026-05-21_v2.12_after.md`](QUALITY_SNAPSHOT_2026-05-21_v2.12_after.md).
Strict-gate corpus state unchanged from v2.10: **34 PASS / 0 WARN /
0 FAIL** — extraction/chunking/validation are untouched by v2.11 +
v2.12 (both cycles changed only the retrieval side).

**Current local test suite: 1032 passed, 15 skipped, 0 failed**
after the v2.12 cycle (+46 over the v2.11.0 baseline of 986). New
test files added in v2.12:

- `tests/test_retrieval_pipeline.py` — 17 mock-driven pipeline tests
- `tests/test_sparse_bm25.py` — 18 BM25 + RRF unit tests
- `tests/test_hyde.py` — 7 HyDE module tests
- `tests/test_retrieval_regression_v2_12.py` — production-pipeline fingerprint test
- `tests/test_ingest_content_preference.py` — 4 tests pinning the Phase 0 preference fix

## v2.12 commits (reverse chronological)

- `181a5a1` (2026-05-21) — Phase 3 close-out: HyDE ships opt-in.
- `51ab67c` (2026-05-21) — Phase 2 close-out: hybrid is production default.
- `d7a0bfd` (2026-05-21) — Phase 2+3 infrastructure (sparse + RRF + HyDE).
- `65a5ba7` (2026-05-21) — Phase 1 close-out: local ModernBERT wins 4/4.
- `988fcaf` (2026-05-21) — Phase 1 retrieval module + tests + soak wiring.
- `0d731b1` (2026-05-21) — Phase 0: `content/refined_content` preference fix.

The v2.12.0 AFTER-snapshot + version bumps + final docs sweep
commit is the next one (also unpushed until live verification).

## Active Engineering Direction

v2.12.0 is the next ship state. **Phase N work remaining (this
commit cycle):**

- bump `__engine_version__` + `pyproject.toml` to `2.12.0` (DONE)
- capture v2.12 production retrieval-regression fingerprint (DONE)
- write AFTER snapshot (DONE — you're reading the headline; full is
  `docs/QUALITY_SNAPSHOT_2026-05-21_v2.12_after.md`)
- update layer-0 docs (CLAUDE.md, AGENTS.md, ARCHITECTURE.md) for v2.12 (in progress)
- run full pytest + live retrieval-regression as final sanity check
- commit + push staging commit; user pushes the v2.12.0 annotated tag

## v2.13 Carry-Forwards

1. **Format gate recovery.** Format 88.4% in v2.12.0 vs ≥96% pin.
   Earthship multi-column OCR damage + CarOK form-shape penalty.
   Earthship re-OCR with layout-aware settings + CarOK form-class
   gate carve-out are the named work.
2. **30-day rollback drop on 2026-06-19.** Remove `mmrag_v2_8`
   legacy collection + `tests/test_retrieval_regression_v2_10.py`.
3. **Local Qwen3-Embedding-8B opportunity.** 7× faster embed than
   cloud (180ms vs 1.2s p50); end-to-end retrieval would be sub-1s.
   Worth a v2.13 Phase 0 (parallel collection rebuild + quality
   soak vs current cloud baseline).
4. **v2.11 carry-forwards 3a (VLM swap), 3c (UIR refactor, PAUSED), 3e (magazine rendered-region-crop).**
5. **HyDE opt-in remains opt-in.** Module ships with v2.12; revisit only if a future retrieval-quality regression or a new use case (e.g., weak-baseline domain expansion) makes the +1s latency worth it.

## Must-Respect Constraints

- Python 3.10 only.
- Batch size must stay at or below 10 pages.
- Do not use `--profile-override` for acceptance runs.
- Do not add filename-specific or document-specific quality rules.
- OCR handles text; VLMs describe visuals only.
- BBoxes must remain normalized integer `[0,1000]`.
- Acceptance requires `GATE_PASS` plus `UNIVERSAL_PASS` across the smoke matrix.
- Production text-retrieval embedder is dashscope/text-embedding-v4 against
  `mmrag_v2_8__qwen3_dashscope`. Production reranker is local
  `gte-reranker-modernbert-base-mlx` via omlx-server. Production retrieval flow is
  `mmrag_v2.retrieval.retrieve_hybrid_reranked()`.
- Ollama/llava lane is rollback-only through 2026-06-19; do not use as a
  comparison baseline beyond that date.
- `DASHSCOPE_API_KEY` + `MLX_API_KEY` env vars must be set for any
  production retrieval. Test-suite skip-gates handle the unset case for CI.
