# Quality Snapshot 2026-05-22 — v2.13 P1 Apples-to-Apples: omlx vs Dashscope

> **Status:** v2.13 Phase 1 embedder shootout — final comparison report.
> **Decision:** SWAP. Local `Qwen3-Embedding-8B-mxfp8` via omlx-server becomes the
> v2.13.0 production embedder. Cloud `text-embedding-v4` retained as fallback
> through 2026-06-19 (30-day rollback window per v2.11.0 precedent).

## 1. Setup — apples-to-apples comparison

Both runs use **the same** sampled chunks, generated queries, retrieval stack,
and judge — only the embedder differs:

| Knob | Value |
|---|---|
| Fixture | `output/soak/v2.13_p1_*/work.jsonl` (259 chunks × 2 queries = 518) |
| Sample seed | 42 |
| Sample stratification | 8 queries per doc across 33 docs (Form_0013 has 0 eligible chunks) |
| Generator | Dashscope `qwen-max` (both runs share the same generated queries) |
| Judge | Dashscope `qwen-max` (both runs share the same judge model + prompts) |
| Retrieval | hybrid (dense + BM25 sparse + RRF) + ModernBERT rerank (local omlx) |
| Top-K retrieve | 25 |
| Top-N return | 5 |
| BM25 index | `tests/fixtures/bm25_index_v2_12.json` (post-v2.13 P2 rebuild; 26,407 text chunks) |
| Sparse collection | `mmrag_v2_8__bm25_sparse` |
| Reranker | local `gte-reranker-modernbert-base-mlx` via omlx-server (same for both) |

### What differs between the two runs

| Knob | omlx (challenger) | dashscope (baseline) |
|---|---|---|
| Embed provider | omlx-server (local LAN, `10.0.10.246:8000`) | Dashscope cloud |
| Embed model | `Qwen3-Embedding-8B-mxfp8` | `text-embedding-v4` |
| Dense collection | `mmrag_v2_8__qwen3_local` (4096-dim, 31,371 pts) | `mmrag_v2_8__qwen3_dashscope` (1024-dim, 31,371 pts) |
| Per-query cost (embed) | $0 | ~$0.0001 |
| Latency (embed) | ~80 ms LAN | ~250–500 ms WAN |

## 2. Headline metrics (apples-to-apples)

| Metric | omlx (local) | dashscope (cloud) | Δ |
|---|---:|---:|---:|
| **Recall@1 chunk** | **57.5%** (298/518) | 55.0% (285/518) | **+2.5 pp omlx** |
| **Recall@5 chunk** | **78.0%** (404/518) | 72.6% (376/518) | **+5.4 pp omlx** |
| **Recall@5 doc**   | **95.2%** (493/518) | 93.1% (482/518) | **+2.1 pp omlx** |
| Relevance score    | 74.6% (773/1036) | 74.1% (768/1036) | +0.5 pp |
| **Format score**   | **92.9%** (962/1036) | 89.2% (924/1036) | **+3.7 pp omlx** |
| Faithfulness score | 66.9% (693/1036) | 65.9% (683/1036) | +1.0 pp |

**omlx wins 6/6 axes.** Three of those are meaningful margins (R@1, R@5 chunk, Format);
three are within the noise floor (Relevance, Faithfulness, R@5 doc).

### Why this is not directly comparable to v2.12.0's R@1 = 67.8%

The v2.13 P1 fixture is a **fresh sample** drawn from post-v2.13-P2 ingestion (after
Earthship + Firearms were re-extracted with `force_full_page_ocr=True`). Different
gold chunk_ids → different fixture difficulty. The dashscope-on-the-new-fixture
read of 55.0% R@1 is the right anchor; the v2.12.0 number of 67.8% R@1 was on a
different fixture and is not the apples-to-apples comparison this report makes.

What *is* preserved across cycles: the **retrieval-stack composition** (hybrid + RRF
+ rerank) and the **judge model** (qwen-max). Cumulative production deltas over
v2.11.0 are tracked in the cycle-close ship reports, not here.

## 3. Per-doc win/loss

| Axis | omlx-win | tie | dashscope-win |
|---|---:|---:|---:|
| Recall@1 chunk | 13 | 7 | 12 |
| Recall@5 chunk | **17** | 9 | 6 |
| Format         | **15** | 12 | 5 |

Per-doc R@1 is a near-tie in counts (13-7-12), but aggregate wins because omlx's
wins are larger margins than its losses. Top-5 omlx wins on R@1:

| Doc | omlx | dash | Δ |
|---|---:|---:|---:|
| Cronin_GenAI_Models | 81.2% | 43.8% | **+37.4** |
| ChatGPT_Praktijk_handboek | 75.0% | 50.0% | **+25.0** |
| Adedeji_GenAI_Google_Cloud | 68.8% | 50.0% | +18.8 |
| Devlin_LLM_Agents | 75.0% | 56.2% | +18.8 |
| Kimothi_RAG_Guide | 75.0% | 56.2% | +18.8 |

Top-5 omlx losses on R@1 (carry-forward for v2.14 targeted improvements):

| Doc | omlx | dash | Δ | Notes |
|---|---:|---:|---:|---|
| ATZ_Elektronik_German | 62.5% | 75.0% | -12.5 | German-language content; Qwen3-Embedding-8B may underperform on Germanic minor langs |
| Greenhouse_Design | 50.0% | 62.5% | -12.5 | Domain-heavy technical (greenhouse engineering) |
| Hybrid_electric_vehicles | 81.2% | 93.8% | -12.6 | Automotive engineering; cloud has more recent training data |
| IRJET_Modeling_of_Solar_PV | 62.5% | 75.0% | -12.5 | Engineering paper |
| Python_Cookbook | 43.8% | 56.2% | -12.4 | Code-heavy; cloud's broader code training helps |

## 4. Decision — SWAP rationale

**Decision: SWAP** to omlx as v2.13.0 production embedder.

Justification (all axes positive):

1. **Quality** — omlx wins on all 6 axes; 3 with meaningful margins
2. **Cost** — embed cost drops to $0 (was ~$0.0001/query; cumulative over high-volume use this is non-trivial)
3. **Privacy** — corpus data never leaves the LAN; relevant for sensitive corpora
4. **Latency** — sub-100ms LAN embed vs ~250–500ms WAN; meaningful for interactive UIs
5. **Independence** — no Dashscope rate-limit or outage exposure on the embedding side (reranker is already local in v2.12)

Risks accepted:

- **German + minor language** content takes a measurable hit (ATZ_Elektronik German -12.5 R@1). Track for v2.14 — possibly add a per-doc language-aware embedder routing if regression deepens with future German content.
- **Some engineering / code-dense docs** (Python_Cookbook, IRJET, Hybrid_electric, Greenhouse) regress 6-12pp R@1. Acceptable given the offsetting wins elsewhere; revisit if the regression compounds across additional code-heavy corpora.

Rollback plan: the dashscope collection (`mmrag_v2_8__qwen3_dashscope`) is retained
unchanged through **2026-06-19** (30-day window). If a corpus-specific regression
surfaces in production use, the production embedder/collection knob in
`src/mmrag_v2/retrieval/config.py` flips back to dashscope without re-ingestion.

## 5. Side reports

- `docs/QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_omlx.md` — full omlx per-doc + weakest queries
- `docs/QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_dashscope_baseline.md` — full dashscope per-doc + weakest queries
- `output/soak/v2.13_p1_omlx/work.jsonl` — omlx fixture (518 queries, 518 retrievals, 518 judgments)
- `output/soak/v2.13_p1_dashscope_baseline/work.jsonl` — dashscope fixture (same 518 queries)

## 6. Methodology — apples-to-apples preservation

The key methodological choice: both runs share **identical** sample seeds, generated
query texts, judge model, judge prompts, retrieval stack, BM25 index, sparse
collection, and reranker. The work file was generated once via
`--stage sample --stage generate`, then forked to two directories. Each retrieve+judge
pass ran independently against its own dense collection but produced retrievals that
were judged by the same qwen-max judge with identical prompts.

This isolates the embedder as the single variable. The 6/6-axis omlx win is therefore
attributable to the embedder swap and not to any retrieval or judge artifact.

## 7. Cost ledger

| Stage | Calls | Provider | Approx cost |
|---|---:|---|---:|
| Sample | — | local | $0 |
| Generate (518 queries) | 259 chunks × 1 call | qwen-max | ~$0.20 |
| omlx retrieve (518 queries × embed) | 518 | omlx (LAN) | $0 |
| omlx rerank (518 × top-25) | 518 | omlx (LAN) | $0 |
| omlx judge (518) | 518 × 3 axes | qwen-max | ~$2.50 |
| dashscope retrieve (518 × embed) | 518 | dashscope | ~$0.05 |
| dashscope rerank (518 × top-25) | 518 | omlx (LAN) | $0 |
| dashscope judge (518) | 518 × 3 axes | qwen-max | ~$2.50 |
| **Total** | | | **~$5.25** |

Within the $25/cycle spend cap.
