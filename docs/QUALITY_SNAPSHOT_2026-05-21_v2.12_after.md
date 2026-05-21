# Quality Snapshot 2026-05-21 — v2.12 AFTER

> **Status:** v2.12.0 retrieval stack staged locally; annotated tag
> NOT yet pushed by the autonomous run. User pushes/tags after
> live-stack re-verification. Predecessor: `v2.11.0` (2026-05-20,
> `c2a461c`).
>
> Engine version: `2.12.0` (`src/mmrag_v2/version.py`,
> `pyproject.toml`). Schema version: `2.7.0` (unchanged since v2.7 —
> v2.12 added retrieval-side infrastructure only; the chunk-shape
> contract is untouched).

## 1. Headline result

v2.12 closes the absolute-quality gap the v2.11 soak revealed.
Cumulative lift over the v2.11.0 baseline:

| Axis | v2.11.0 | **v2.12.0** | Δ (pp) | Multiple | Floor | Stretch |
|---|---:|---:|---:|---:|---:|---:|
| Recall@1 chunk | 35.5% | **67.8%** | +32.3 | 1.9× | ≥55% ✓ | ≥70% (2.2pp gap) |
| Recall@5 chunk | 66.8% | **90.2%** | +23.4 | 1.4× | ≥85% ✓ | **≥90% ✓ STRETCH** |
| Recall@5 doc | 91.7% | **98.6%** | +6.9 | 1.1× | ≥95% ✓ | **≥97% ✓ STRETCH** |
| Relevance (judge) | 59.3% | **82.1%** | +22.8 | 1.4× | ≥75% ✓ | ≥85% (2.9pp gap) |
| Faithfulness (judge) | 50.6% | **72.6%** | +22.0 | 1.4× | ≥70% ✓ | ≥80% (7.4pp gap) |
| **Format (judge)** | 89.8% | 88.4% | **−1.4** | — | **≥96% ✗** | ≥98% |

**Five of six axes pass their floors. Two hit STRETCH (Recall@5
chunk and Recall@5 doc).** Format remains below the ≥96% pin —
identified during Phase 0 as chunk-level OCR / form-shape damage
that retrieval-side work can't address. Carry-forward to v2.13 as
named recovery work (Earthship re-OCR; CarOK form-shape decision).

The cumulative v2.10 → v2.11 → v2.12 trajectory on Recall@1 chunk:

```
v2.10  ====                                          2.1%   (llava embedder)
v2.11  ==============                               35.5%   (dashscope embedder)
v2.12  ==============================               67.8%   (hybrid + rerank)
        0%       20%       40%       60%       80%
```

## 2. Production retrieval stack (v2.12.0)

```
query
  ├─> embed (Dashscope text-embedding-v4, 1024-dim cloud)
  ├─> dense Qdrant top-25 (mmrag_v2_8__qwen3_dashscope, cosine)
  │
  └─> BM25 sparse encode (tests/fixtures/bm25_index_v2_12.json)
      └─> sparse Qdrant top-25 (mmrag_v2_8__bm25_sparse)
  │
  ├─> RRF fusion (k=60, equal weights, top-25)
  │
  └─> rerank (local gte-reranker-modernbert-base-mlx, omlx-server)
      └─> top-5 return
```

End-to-end p99 latency: **~2.05 s** from the user's network. Within
the 3.0 s soft budget. Stage breakdown:

| Stage | p99 |
|---|---:|
| Embed (Dashscope, cloud round-trip dominates) | 1.35 s |
| Qdrant dense top-25 | 0.05 s |
| Qdrant sparse top-25 | 0.05 s |
| RRF fusion (in-process) | ~instant |
| ModernBERT rerank top-25 → top-5 (Mac Mini LAN) | 0.55 s |

**Per-query cost.** ~$0.001 (Dashscope embed only — rerank is local
zero-cost). HyDE (opt-in via `use_hyde=True`) adds ~$0.001 and ~1 s
latency but did not meaningfully change quality in the Phase 3
measurement soak; see §"Phase 3 outcome" in `docs/DECISIONS.md`.

## 3. Phase-by-phase contributions

| Phase | Change | What it cost | What it bought |
|---|---|---|---|
| 0 | `content/refined_content` preference fix in `ingest_to_qdrant.py` | 1-day investigation + 22 min ingest | IRJET Format 71.9% → 87.5% (+15.6pp); rest of Format dip diagnosed as chunk-level OCR damage |
| 1 | Cross-encoder reranker (local ModernBERT) | 1 day + ~65 min soak (~$2-3) | R@1 chunk 35.5% → 61.8% (+26.3pp); Faith 50.6% → 69.4% |
| 2 | Hybrid (BM25 + dense + RRF) | ~30 min ingest + 33 min soak (~$2-3) | R@5 chunk 81.3% → 90.2% (+8.9pp); Faith 69.4% → 72.6% |
| 3 | HyDE — measured, opt-in only | 52 min soak (~$3-4) | All deltas within ±1pp (noise) |
| 4 | NOT triggered | — | — |
| N | Tag staging + AFTER snapshot | — | This document. |

Cumulative cycle spend: ~$8-10 in Dashscope (embed + judge + HyDE).
Comfortably under the $25 cap.

## 4. Qdrant state

| Collection | Built | Vectors | Status | Role |
|---|---|---:|---|---|
| `mmrag_v2_8__qwen3_dashscope` | 2026-05-19 (rebuild) + 2026-05-21 (Phase 0 patch on 3 docs) | 30,588 (1024-dim cosine, on_disk) | green | **Production dense** |
| `mmrag_v2_8__bm25_sparse` | 2026-05-21 | 25,623 (sparse, on_disk, no dense vector) | green | **Production sparse (BM25)** |
| `mmrag_v2_8` | 2026-05-16 | 30,454 (4096-dim cosine, llava) | green | v2.10 rollback baseline (drop 2026-06-19) |

The legacy `mmrag_v2_8` collection is kept untouched for the 30-day
rollback contract through 2026-06-19. After that date the production
retrieval will solely depend on the two qwen3-named collections.

## 5. Tests

- **1032 passed**, 15 skipped, 0 failed on the local stack
  (`pytest tests/ --ignore=tests/manual -q`).
- Two retrieval-regression contracts in tracked fixtures:
  - `tests/fixtures/retrieval_regression_v2_11_qwen3.json` — pins the
    v2.11.0 dense-only retrieval shape (kept for cross-version diff).
  - `tests/fixtures/retrieval_regression_v2_12_hybrid.json` — pins
    the v2.12.0 hybrid+rerank retrieval shape.
- Both regression tests skip cleanly when their backends are unreachable.
- Live-stack verification before push: run
  `pytest tests/ --ignore=tests/manual -q` and
  `python scripts/retrieval_regression_v2_12.py` (no flags = verify
  against the captured fingerprint).

## 6. New code in this cycle

```
src/mmrag_v2/retrieval/             new module — composable pipeline
  __init__.py                       public exports
  config.py                         get_reranker() factory + compile default
  pipeline.py                       retrieve_reranked() + retrieve_hybrid_reranked()
  reranker.py                       Reranker protocol + 3 implementations
  sparse.py                         BM25 + RRF (no external dep)
  hyde.py                           HyDE module (opt-in only)

scripts/
  build_bm25_index.py               build the 2MB tracked BM25 index
  ingest_bm25_sparse.py             ingest sparse side-collection (5.7s)
  measure_reranker_latency.py       reusable latency benchmark (pre-existing, extended)
  compare_reranker_quality.py       side-by-side reranker quality (pre-existing, extended)
  compare_soak_reports.py           pick winner from two soak reports
  retrieval_regression_v2_12.py     v2.12 production fingerprint capture/verify

tests/
  test_retrieval_pipeline.py        17 tests
  test_sparse_bm25.py               18 tests
  test_hyde.py                      7 tests
  test_retrieval_regression_v2_12.py  v2.12 regression test
  test_ingest_content_preference.py 4 tests (Phase 0)

tests/fixtures/
  bm25_index_v2_12.json             2 MB tracked index
  reranker_latency_*.json            ×4 benchmark data
  reranker_quality_*.json            ×2 quality comparisons
  retrieval_regression_v2_12_hybrid.json  v2.12 fingerprint

docs/
  PLAN_V2.12.md                     Draft v0.1 → v0.8
  DECISIONS.md                      4 new sections appended
  QUALITY_SNAPSHOT_2026-05-21_v2.12_p1_cloud.md    Phase 1 soak (cloud)
  QUALITY_SNAPSHOT_2026-05-21_v2.12_p1_omlx.md     Phase 1 soak (local)
  QUALITY_SNAPSHOT_2026-05-21_v2.12_p2_hybrid.md   Phase 2 soak (hybrid)
  QUALITY_SNAPSHOT_2026-05-21_v2.12_p3_hyde.md     Phase 3 measurement
  QUALITY_SNAPSHOT_2026-05-21_v2.12_after.md       This document
```

## 7. Known limitations carried to v2.13

| # | Item | What it is | What needs to happen |
|---|---|---|---|
| 1 | **Format gate (≥96%)** still misses (88.4%) | Chunk-level OCR damage in Earthship (multi-column scanned interleaving) + form-shape penalty on CarOK | Earthship re-OCR with layout-aware settings; CarOK form-class Format carve-out OR chunk restructuring |
| 2 | 30-day rollback drop on 2026-06-19 | Legacy `mmrag_v2_8` + `test_retrieval_regression_v2_10.py` | Remove both at drop date (or earlier with user sign-off) |
| 3 | Local Qwen3-Embedding-8B opportunity | Available + benchmarked at 7× faster than cloud embed; would deliver sub-1s p99 end-to-end retrieval | v2.13 Phase 0 candidate: build parallel collection with local embed, soak vs current cloud baseline, swap if quality holds |
| 4 | v2.11 carry-forwards 3a (VLM swap), 3c (UIR refactor, PAUSED), 3e (magazine rendered-region-crop) | Documented as v2.13+ work | Per-item revisit when v2.13 plan is authored |

## 8. Methodology — apples-to-apples comparison

All v2.12 Phase 1/2/3 soak runs used the EXACT SAME 518-query × 259-
chunk fixture as the v2.11 baseline:
`output/soak/v2.11_qwen3/work.jsonl`. The fixture was cloned for each
phase with `retrieval` and `judgment` fields stripped, then the only
variable was the retrieval pipeline. Same embedder (`text-embedding-v4`),
same Qdrant collection (`mmrag_v2_8__qwen3_dashscope`), same judge
(qwen-max temperature 0.0). This isolates the retrieval-side change
from any other source of variance.

Cumulative spend across all v2.12 soak runs: **~$8-10** in Dashscope
(embed + judge calls). Within the cycle's $25 cap.

## 9. Revision log

| Date | Change |
|---|---|
| 2026-05-21 | Initial v2.12 AFTER snapshot. Phase 0 + Phase 1 + Phase 2 + Phase 3 (opt-in) shipped; Phase 4 NOT triggered. v2.12.0 annotated tag staged but NOT pushed by the autonomous run — user pushes/tags after live-stack re-verification. |
