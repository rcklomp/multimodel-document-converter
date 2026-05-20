# Plan: v2.12 — Close the absolute-quality gap: reranker → hybrid retrieval → HyDE

**Status:** **Draft v0.1** (2026-05-21). Authored after the v2.11.0
swap landed and was tagged (`c2a461c`, annotated `v2.11.0` on both
remotes 2026-05-20/21). v2.11 closed the *embedder* bottleneck with a
10× lift across every embedder-attributable axis; v2.12 closes the
*absolute-quality* gap that the v2.11 soak revealed: relative numbers
are great, but the system still misses the right passage in the top-5
about 1 query in 3 (Recall@5 chunk 66.8%). The user named this gap
explicitly during the v2.11.0 close-out — this plan exists to address
it, not to be agreed with after the fact.

**Predecessor:** [`docs/PLAN_V2.11.md`](PLAN_V2.11.md) — Draft v1.0,
Phase 1 swap executed 2026-05-20 on `c2a461c`, tag `v2.11.0` on
`c2a461c` public on both `github` and `origin` (Gitea at
`10.0.10.241`).
**Owner:** ingestion + retrieval pipeline.

---

## 1. Why this plan exists

### Thesis

After v2.11.0 the production stack retrieves the right *document*
91.7% of the time but the right *chunk* only 66.8% of the time at
top-5 and 35.5% at top-1. That's a textbook "right doc, wrong
passage" pattern. The retrieval candidate set is healthy (the doc-
level recall proves the embedder finds the relevant document); the
top-K ordering within candidates is the bottleneck. A cross-encoder
reranker is the canonical fix for this shape.

Beyond reranking, two additional levers raise *what gets into the
candidate set* in the first place: hybrid retrieval (BM25 + dense
fusion, recovers lexical/term-specific matches the dense embedder
misses) and HyDE (embed a hypothetical answer rather than the
question, recovers chunks whose surface form is answer-shaped). Each
is independent of the embedder and stackable with the others.

### Where v2.11 ended

| Axis | v2.10 baseline | v2.11.0 actual | v2.12 floor | v2.12 stretch |
|---|---:|---:|---:|---:|
| Recall@1 chunk | 2.1% | **35.5%** | ≥ 55% | ≥ 70% |
| Recall@5 chunk | 6.8% | **66.8%** | ≥ 85% | ≥ 90% |
| Recall@5 doc | 54.2% | **91.7%** | ≥ 95% | ≥ 97% |
| Relevance (judge) | 5.9% | **59.3%** | ≥ 75% | ≥ 85% |
| Faithfulness (judge) | 4.7% | **50.6%** | ≥ 70% | ≥ 80% |
| Format (judge) | 98.3% | **89.8%** | **≥ 96%** | ≥ 98% |

Floor rationale: every floor is the "good" tier in standard
production-RAG benchmarks. The Format floor reverts to v2.10's
original ≥96% (after two consecutive recovery soaks; v2.11.x
Format-recovery patch is Phase 0 here, see below).

### What v2.12 starts from

- Production embedder Dashscope `text-embedding-v4` against
  `mmrag_v2_8__qwen3_dashscope` (1024-dim, 30,588 points).
- Two regression-test lanes pinned (production + 30-day rollback);
  rollback lane drops 2026-06-19 (Phase 0).
- Measurement substrate is the same v2.11 soak harness
  (`scripts/retrieval_regression.py` + `scripts/synthetic_soak.py`).
  Every v2.12 phase re-uses the same 259-chunk × 518-query soak
  fixture so the deltas are apples-to-apples.
- Phase 1 → 2 → 3 → 4 ladder of escalation: each phase has an
  explicit floor; if the floor isn't cleared, the next phase
  triggers; if it is cleared, the next phase is *still allowed* but
  optional. v2.12.0 ships whichever subset clears the §"Acceptance
  Gate" thresholds at the lowest cumulative cost.

### Carry-forward register (Draft v0.1)

| # | Class | Source | v2.12 status | Notes |
|---|---|---|---|---|
| 1 | v2.11.x Format recovery | v2.11 Phase 1 soak Format 89.8% | **`in-scope`** | Phase 0 below; scanned/form chunk-content sanitization for `CarOK_voorraadtelling`, `Earthship_Vol1`, `IRJET_Modeling_of_Solar_PV` (top-3 offenders). Target ≥ 95% on next soak. |
| 2 | 30-day rollback drop | v2.11.0 close-out 2026-05-20 | **`in-scope`** | Phase 0 below; drop date 2026-06-19. Remove `mmrag_v2_8` collection + `tests/test_retrieval_regression_v2_10.py`. |
| 3 | **Reranker** (gte-rerank-v2) | v2.11 soak right-doc-wrong-chunk pattern (R@5 doc 91.7%, R@5 chunk 66.8%) | **`in-scope`** | Phase 1 below. Single biggest lever. Same Dashscope ecosystem as the production embedder + soak judge. |
| 4 | **Hybrid retrieval** (BM25 + dense + RRF) | Standard production-RAG technique; addresses lexical misses dense embeddings can't recover | **`in-scope`** | Phase 2 below. Conditional on Phase 1 not clearing R@5 chunk ≥ 85%. |
| 5 | **HyDE / query rewriting** | Standard technique; addresses answer-shaped-chunk vs question-shaped-query mismatch | **`in-scope`** | Phase 3 below. Conditional on Phase 1+2 not clearing R@1 ≥ 55%. |
| 6 | Per-doc-class chunking | v2.11 carry-forward (overlaps with 3d) | **`conditional (Phase 4)`** | Only if Phases 1-3 don't reach the floors. Heavy: full corpus reconversion. |
| 7 | 3a VLM swap (Qwen3-VL-8B or Dashscope qwen3-vl-plus) | v2.11 Phase 3 carry-forward | **`out-of-scope unless soak signals image-quality drag`** | Parallel to v2.12 retrieval work if it materializes; otherwise carry to v2.13. |
| 8 | 3c UIR refactor (`ConversionPlan` parent class) | v2.11 Phase 3c, PAUSED | **`still paused`** | User signoff required to unfreeze. v2.12 default: stay paused — the retrieval work is more impactful. |
| 9 | 3d HybridChunker per-item token guard | v2.11 Phase 3d, design recorded | **`out-of-scope (subsumed by Phase 4 if shipped)`** | If Phase 4 ships per-doc-class chunking, the per-item guard becomes part of the new chunker design rather than a separate flag. |
| 10 | 3e Magazine rendered-region-crop | v2.11 Phase 3e, deferred with data | **`out-of-scope`** | Same trigger rationale as v2.11: ceiling is retrieval, not chunk shape. Revisit only on a new magazine-class Format defect. |

---

## 2. Goals & Non-Goals

### Goals (measurable)

1. **Recall@1 chunk ≥ 55%** on the same 259-chunk × 518-query soak
   fixture used by v2.10 baseline + v2.11 Phase 1 (deltas are
   apples-to-apples).
2. **Recall@5 chunk ≥ 85%** on the same fixture.
3. **Recall@5 doc ≥ 95%** (small lift over v2.11.0 91.7%; mostly
   already clears the floor).
4. **Relevance ≥ 75%, Faithfulness ≥ 70%** (judge axes; same
   `qwen-max` judge as v2.10/v2.11).
5. **Format ≥ 96%** (revert to original pin; v2.11.x recovery
   landed in Phase 0).
6. **Strict-gate state unchanged at 34 PASS / 0 WARN / 0 FAIL**
   (extraction/chunking/validation untouched).
7. **Production retrieval p99 latency ≤ 1.5 s** end-to-end (a soft
   budget — reranker + HyDE add ~200-500 ms each; if total exceeds
   1.5 s, ship reranker-only and defer HyDE).
8. **Cost per soak ≤ $10** in Dashscope spend (reranker calls,
   judge calls, optional HyDE calls). Bounded for the v2.12 cycle.

### Non-Goals (deferred beyond v2.12 unless promoted)

1. Schema changes (chunk-shape contract stays at 2.7.0).
2. Replacing Docling (extraction unchanged).
3. Embedder bump (`text-embedding-v4` stays — v2.11.0 just paid the
   migration cost; diminishing returns on a fresh embedder swap).
4. Local-hosted reranker (Dashscope cloud is the v2.12 path; local
   MLX reranker is a v2.13 candidate iff data-policy or cost shifts).
5. Multi-query / fusion-of-rewrites beyond single-shot HyDE.
6. Cross-doc joins / multi-hop retrieval. Future-future.
7. Re-conversion of the corpus unless Phase 4 triggers (Phases 1-3
   reuse the existing 30,588-point collection).

---

## 2b. Cross-phase principles (carried from PLAN_V2.10 §2b / §2c / §2d)

- **Parallel-Site Audit.** Every change has a "where else does this
  knowledge live" sweep. Reranker integration is the highest-risk
  axis: `search_qdrant.py` already has a `--no-rerank` flag and a
  stub call site; v2.10 left it as "vector-rank truncation when key
  unset." Phase 1 must promote the reranker to a real call and
  thread it through the production retrieval path that production
  RAG consumers actually use.
- **Architectural constraints.** No changes to `ingest_to_qdrant.py`
  beyond optionally adding a sparse-vector field in Phase 2.
  Retrieval-side code lives under a new `src/mmrag_v2/retrieval/`
  module (creating it is the first surgical step in Phase 1).
- **Cost-aware ordering.** Phase 1 (reranker, no re-ingestion) →
  Phase 3 (HyDE, no re-ingestion) → Phase 2 (hybrid, ONE
  re-ingestion ≈ 5-7 h) → Phase 4 (per-doc-class chunking, full
  corpus reconversion, heavy). Phase 2 is in the middle of the
  cost-curve because it requires re-ingestion but not reconversion.
  Phase 3 is cheap per-call but expensive per-query at scale.

---

## 3. Phases

### Phase 0 — v2.11.x close-out + Format recovery  (housekeeping)

**What.** Two carry-forwards from v2.11 land before any v2.12
retrieval work starts:

1. **v2.11.x Format recovery.** Chunk-content sanitization for the
   three offending scanned/form documents. Each chunk's `content`
   field gets a pre-Qdrant cleanup pass that normalizes OCR-strip
   artefacts (stray Unicode replacement characters, broken column
   joins, isolated digit/letter fragments). The fix is in the JSONL
   files on disk, not in the chunker — it's a one-shot script that
   re-emits the three docs' JSONLs and re-ingests them into Qdrant.
   No reconversion needed.
2. **30-day rollback drop (2026-06-19).** Remove the legacy Ollama
   `llava` collection from Qdrant; remove
   `tests/test_retrieval_regression_v2_10.py`; remove
   `tests/fixtures/retrieval_regression_v2_10.json`; remove the
   `--provider ollama` lane from the production retrieval scripts
   (the option stays in code as opt-in for testing, but the rollback
   path is no longer a release contract).

**Tests.**
- `tests/test_format_recovery_v2_11x.py` — new fixture
  `tests/fixtures/format_recovery_targets.json` listing the three
  named docs + their pre-recovery Format scores; a regression test
  asserts the recovered JSONLs satisfy `qa_full_conversion.py
  --source-pdf --allow-warnings` and that re-running the v2.11 soak
  on those three docs reports Format ≥ 95%.
- Delete `tests/test_retrieval_regression_v2_10.py` on 2026-06-19.

**Acceptance.**
- v2.11 soak re-run on the three Phase-0 docs reports Format ≥ 95%
  (target stays below the v2.12 §1 ≥96% gate; Phase 0 is incremental
  recovery, not full recovery).
- Strict-gate state unchanged (still 34/0/0 corpus-wide).
- Test suite green.

**Risk.** Low. The fix targets JSONL content normalization only; no
chunker, no embedder, no schema changes.
**Cost class.** Reconvert: no. Re-enrich: no. Re-ingest: 3 docs only
(~5-10 min). Soak partial re-run: ~$0.50 in Dashscope spend.
**Effort.** 1-2 days.

---

### Phase 1 — Reranker  (the biggest single lever)

**What.** Add Dashscope `gte-rerank-v2` as a second-stage reranker
between Qdrant top-K retrieval and the final returned chunks. The
production retrieval flow becomes:

```
query → embed (text-embedding-v4) → Qdrant top-50 (cosine)
      → gte-rerank-v2 (query × chunk pairs) → reordered top-5/10
```

The reranker is a cross-encoder: it scores `(query, chunk)` pairs
*together*, capturing semantic interaction that a single-vector
embedder can't. For the v2.11 right-doc-wrong-chunk pattern, this
is the canonical fix.

**Architecture decisions.**

- **New module** `src/mmrag_v2/retrieval/` with three files at the
  start: `__init__.py`, `reranker.py` (Dashscope client), `pipeline.py`
  (composable retrieve → rerank → return).
- **No changes** to `scripts/ingest_to_qdrant.py` (retrieval-side
  only).
- **`scripts/search_qdrant.py` is the integration point.** Its
  existing `--no-rerank` flag flips meaning: today it's a no-op
  (rerank degrades to vector-rank truncation); after Phase 1,
  default behavior is rerank-on, and `--no-rerank` opts out.
- **Reranker API.** `POST` to
  `https://dashscope-intl.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank`
  with `model: "gte-rerank-v2"`. Same `DASHSCOPE_API_KEY` env var
  as the embedder. Cost: ~$0.001 per query × ~50 candidates per
  query → ~$0.05 per 1k queries. Soak run (518 queries): ~$0.03.
  Production: bounded by query traffic.
- **Latency budget.** Reranker adds ~200-400 ms per query (network
  round-trip + scoring). Within the §1.8 p99 ≤ 1.5 s soft budget.

**Approach.**

1. Build `src/mmrag_v2/retrieval/reranker.py` with
   `rerank_chunks(query, chunks, model="gte-rerank-v2") → list[dict]`.
   Returns chunks in reranker-order with a new `rerank_score` field.
   Network retries on 429/5xx mirror `embed_text_dashscope`.
2. Build `src/mmrag_v2/retrieval/pipeline.py` with
   `retrieve_reranked(query, collection, top_k_retrieve=50,
   top_k_return=5, qdrant_url, api_key) → list[dict]`. Composes
   embed → Qdrant search → reranker → top-k slicing.
3. Wire `scripts/search_qdrant.py` and `scripts/retrieval_regression.py`
   to use the new pipeline by default. Add a `--no-rerank` opt-out.
4. Extend `scripts/synthetic_soak.py` with a `--rerank` flag (default
   on for v2.12; default off keeps the v2.11 baseline reproducible).
5. Capture a **new fingerprint** at
   `tests/fixtures/retrieval_regression_v2_12_reranked.json` —
   reranked top-5 per query. Old fingerprint
   (`_v2_11_qwen3.json`) is kept as the v2.11.0 baseline.
6. Re-run the soak with `--rerank` against `mmrag_v2_8__qwen3_dashscope`.
   Compare deltas; record decision row in `docs/DECISIONS.md`.

**Tests (red → green).**

- `tests/test_retrieval_pipeline.py` — composable mock-driven tests
  for the pipeline: embed-mock returns vector, Qdrant-mock returns
  top-50 ordered, reranker-mock returns reordered, pipeline returns
  the reranker's order. Pins the integration shape.
- `tests/test_reranker_smoke.py` — live-skip-gated integration test
  against Dashscope (skips when `DASHSCOPE_API_KEY` unset).
- `tests/test_retrieval_regression_v2_12_reranked.py` — production
  retrieval-shape pin with rerank on, mirroring the structure of
  `test_retrieval_regression_v2_11.py`.
- `tests/test_retrieval_regression_v2_11.py` keeps passing
  (no-rerank lane stays valid as the unreranked fingerprint).

**Done when.**

- Soak with rerank-on reports **Recall@1 ≥ 55%, Recall@5 ≥ 75%
  (within Phase 1 alone — Phase 2 closes the rest), Faithfulness ≥
  70%, Relevance ≥ 75%**.
- `docs/DECISIONS.md` records a "v2.12 Phase 1 Reranker Outcome"
  row with the per-axis deltas vs v2.11.0.
- New fingerprint committed; both regression tests green.
- Strict-gate state unchanged.

**Risk.** Low-medium. The integration is bounded (one new module,
one API client, one config flag flipped). The biggest risk is
latency: if the reranker adds > 500 ms p99 the production retrieval
flow may need to drop `top_k_retrieve` from 50 to 25 (smaller pair
count → faster scoring → lower lift). The decision is data-driven
in the soak.
**Cost class.** No re-ingestion. No reconversion. Soak run: ~$0.05.
Production: query-traffic-bound.
**Effort.** 2-3 days.

**Why Phase 1 alone may not suffice for Recall@5 chunk ≥ 85%.**
Reranker improves *ordering within the retrieved candidate set*; it
does not change which chunks are in the set. If the right chunk
isn't in Qdrant's top-50 dense retrieval, no reranker can surface
it. Recall@5 doc is 91.7% in v2.11.0 — so for most queries the
right doc IS in top-5, and within-doc reranking can lift Recall@5
chunk. But for the ~8% of queries where the right doc isn't even in
top-5, the candidate set is wrong. Phase 2 addresses that.

---

### Phase 2 — Hybrid retrieval (BM25 + dense + RRF fusion)

**Conditional on Phase 1.** Triggered iff post-Phase-1 soak reports
Recall@5 chunk < 85%.

**What.** Augment Qdrant with sparse vectors (BM25-style term
frequencies) per chunk. At retrieval time, run two parallel
searches — dense (text-embedding-v4) and sparse (BM25) — then fuse
the result lists via Reciprocal Rank Fusion (RRF). The fused list
is the candidate set the Phase 1 reranker consumes.

Dense embeddings excel at semantic similarity but underperform on
exact-term recall (product names, version numbers, code
identifiers, rare technical terms). BM25 is the opposite. RRF
captures both without requiring score calibration between the two.

**Architecture decisions.**

- **Qdrant native sparse vectors.** Qdrant supports named sparse
  vectors on the same collection. Schema change: add a
  `bm25_sparse` named vector to `mmrag_v2_8__qwen3_dashscope`. This
  requires either (a) a re-ingest of the full collection with
  sparse + dense vectors per point (~5-7 h wall time, same shape as
  the v2.11.0 rebuild), or (b) a parallel side-collection
  `mmrag_v2_8__qwen3_dashscope__bm25_sparse` ingested independently
  and joined at query time. **Default (a):** single collection,
  cleaner.
- **BM25 corpus.** Pre-compute BM25 IDF over the 30,588 chunks
  using a standard `rank-bm25` implementation. Persist the vocab +
  IDF table as a tracked file (small — ~100 KB).
- **Fusion algorithm.** RRF with constant `k=60` (standard).
  `score(chunk) = 1/(k + rank_dense) + 1/(k + rank_sparse)`. Top-50
  by fused score becomes the candidate set for Phase 1 reranker.
- **No change to ingestion-side fields.** The sparse vector is
  derived from the existing `content` field; no new chunk metadata.

**Approach.**

1. Build `src/mmrag_v2/retrieval/sparse.py` — BM25 IDF computation,
   sparse-vector encoding for new queries.
2. Add `--with-sparse` to `scripts/ingest_to_qdrant.py`. Re-run
   ingest on the full 34-doc corpus with sparse vectors enabled.
   Expect ~5-7 h wall time (parallel to the v2.11.0 rebuild).
3. Extend `src/mmrag_v2/retrieval/pipeline.py` with a
   `retrieve_hybrid_reranked()` variant: dense + sparse → RRF →
   top-50 → reranker → top-5.
4. Capture a new fingerprint:
   `tests/fixtures/retrieval_regression_v2_12_hybrid.json`.
5. Re-run soak with hybrid + rerank. Compare deltas.

**Tests.**

- `tests/test_bm25_sparse_index.py` — IDF computation correctness
  on a synthetic corpus.
- `tests/test_rrf_fusion.py` — RRF math on small fixed-rank inputs.
- `tests/test_retrieval_pipeline_hybrid.py` — integration test
  with mock dense + sparse + reranker; pin the pipeline shape.
- `tests/test_retrieval_regression_v2_12_hybrid.py` — production
  fingerprint for the hybrid lane.

**Done when.**

- Soak with hybrid + rerank reports **Recall@5 chunk ≥ 85%, Recall@1
  ≥ 55%**.
- `docs/DECISIONS.md` records the Phase 2 outcome row.
- New fingerprint committed.

**Risk.** Medium. Schema migration (one new named vector) on the
production collection. Mitigation: do the re-ingest into a
parallel collection `mmrag_v2_8__qwen3_dashscope__hybrid` first,
verify the soak deltas, then swap collection defaults exactly like
v2.11 Phase 1 did. Two collections present during the transition.
**Cost class.** Reconvert: no. Re-enrich: no. Re-ingest: yes (~5-7
h, same as v2.11.0 rebuild). Soak run: ~$0.05.
**Effort.** 4-5 days (including the rebuild wall-time).

---

### Phase 3 — HyDE / query rewriting

**Conditional on Phase 1+2.** Triggered iff post-Phase-2 soak
reports Recall@1 < 55% **or** Faithfulness < 70%.

**What.** Before embedding the user's query, generate a
hypothetical answer to it via Dashscope `qwen-max` (the same model
used as soak judge — known good on this corpus). Embed *that
hypothetical answer* and search Qdrant with it. The intuition:
answers and question-answering chunks share vocabulary; questions
and chunks don't.

**Architecture decisions.**

- **Single-shot HyDE.** Generate one hypothetical answer per query,
  not 3-5 paraphrases. Lower cost, simpler pipeline.
- **Model: `qwen-max`** (already a judged-good model in the soak
  flow; reuse the same Dashscope client).
- **Temperature 0.3** for the answer generation (some diversity but
  not full hallucination).
- **Prompt:** "Write a 50-100 word direct answer to the following
  question, in the same language as the question. Do not hedge or
  refuse — write the answer as if you knew it confidently. If
  uncertain, write a plausible answer. {QUERY}"
- **Fallback:** if the LLM call fails (5xx, timeout, parse error),
  fall back to embedding the literal query. Don't fail the
  retrieval call.
- **Caching:** keep a small LRU of (query → hypothetical answer)
  pairs in memory; production may add Redis/disk later.

**Approach.**

1. Build `src/mmrag_v2/retrieval/hyde.py` with
   `generate_hypothetical_answer(query, api_key) → str`.
2. Extend `src/mmrag_v2/retrieval/pipeline.py` with
   `retrieve_hyde_hybrid_reranked()`: hyde → (dense+sparse RRF) →
   reranker → top-5.
3. Add `--hyde` flag (default off in v2.12.0; turn on in
   v2.12.1 if Phase 3 ships).
4. Soak run with hyde + hybrid + rerank.

**Tests.**

- `tests/test_hyde_smoke.py` — live-skip-gated.
- `tests/test_retrieval_pipeline_hyde.py` — mock-driven shape pin.
- `tests/test_retrieval_regression_v2_12_hyde.py` — fingerprint.

**Done when.**

- Soak with HyDE + hybrid + rerank reports **Recall@1 ≥ 55%,
  Faithfulness ≥ 70%, Relevance ≥ 75%**.
- Latency budget check: HyDE adds ~500-800 ms per query (qwen-max
  generation). If p99 total exceeds 1.5 s, ship HyDE behind an
  opt-in flag (default off) and document the latency trade-off.

**Risk.** Medium. The LLM-in-the-loop is a new failure mode (rate
limits, generation latency). Mitigation: the fallback to literal-
query embed on any error is the safety net.
**Cost class.** Reconvert: no. Re-ingest: no. Soak run: ~$1-2
(qwen-max generation calls are pricier than reranker calls).
Production: ~$0.001 per query.
**Effort.** 2-3 days.

---

### Phase 4 — Per-doc-class chunking  (conditional / stretch)

**Conditional on Phases 1-3.** Triggered iff post-Phase-1+2+3 soak
reports Recall@5 chunk < 85% **or** Faithfulness < 70%.

**What.** Today every document uses the same HybridChunker shape.
Phase 4 introduces profile-specific chunking strategies:

- `code_heavy` (Python_Cookbook, Fluent_Python, etc.) — chunks
  align to function / class / section boundaries via Docling's
  `CodeItem` hierarchy. Code blocks stay intact.
- `digital_magazine` (PCWorld, Combat) — chunks align to
  page-region boxes via the layout model. Each visually-distinct
  region becomes its own chunk; the v2.11 hub-collapse coverage
  reveal can be re-evaluated.
- `scanned` / `scanned_degraded` (Firearms, CarOK form,
  Earthship) — chunks align to OCR layout regions; the v2.11.x
  Format recovery work feeds this.
- `digital_literature` (HarryPotter, etc.) — chunks align to
  scene/chapter boundaries.

**Subsumes** v2.11 carry-forward 3d (HybridChunker per-item token
guard) — the per-item guard becomes part of the new chunker design.

**Approach (sketch only — full design deferred to Phase 4 kick-off).**

1. Audit which docs hit Recall@5 chunk < 70% after Phases 1-3 to
   prioritize the chunking work.
2. Build profile-specific chunker classes under
   `src/mmrag_v2/chunking/`. Each class is a small wrapper over
   Docling's chunker that overrides boundary selection.
3. Re-convert the corpus (full ~hours of wall time, depending on
   VLM load).
4. Re-ingest the re-converted corpus into a parallel collection.
5. Re-run soak; compare deltas.

**Risk.** High. Full reconversion = heavy wall time + Format-score
risk if a new chunker introduces regressions. Mitigation: each
profile-specific chunker has its own regression test against a
fixed doc's pre/post chunk counts and Format scores.
**Cost class.** Reconvert: yes (corpus-wide). Re-ingest: yes. Soak:
yes. Effort: 1-2 weeks.

**Default: don't trigger.** Phases 1-3 should clear the floors
without Phase 4 in most realistic outcomes. Phase 4 is the safety
valve, not the planned path.

---

### Phase N — Re-verification, AFTER snapshot, v2.12.0 tag

**What.**

1. Full pytest run against the live stack (Qdrant + Dashscope +
   optional Ollama for the rollback lane if still active).
2. All retrieval-regression tests green (v2.11 production lane stays
   passing; v2.12 reranked + optional hybrid + optional HyDE
   fingerprints all match).
3. Strict-gate corpus run reports 34/0/0.
4. AFTER snapshot authored at
   `docs/QUALITY_SNAPSHOT_<DATE>_v2.12_after.md` with per-axis
   deltas vs v2.11.0.
5. Bump `__engine_version__` and `pyproject.toml` to `2.12.0`.
6. Plan promoted to Draft v1.0.
7. Annotated tag `v2.12.0` staged but **NOT pushed** by autonomous
   run — user pushes/tags.

**Risk.** Low — Phase N is pure ceremony if Phases 0-3 cleared
their gates.
**Cost class.** Pure validation, no rebuild.
**Effort.** 1 day.

---

## 4. Acceptance Gate

Before the v2.12.0 tag is staged:

1. Recall@1 chunk ≥ 55% **and** Recall@5 chunk ≥ 85% **and**
   Recall@5 doc ≥ 95% in the post-Phase-3 (or earlier if floors
   met) soak.
2. Relevance ≥ 75% **and** Faithfulness ≥ 70%.
3. Format ≥ 96% (v2.11.x Phase 0 recovery must have landed).
4. Strict-gate corpus: 34/0/0 unchanged.
5. Full pytest green (live stack reachable for the integration tests).
6. All three retrieval-regression fingerprints
   (`_v2_11_qwen3.json`, `_v2_12_reranked.json`, optionally
   `_v2_12_hybrid.json` and `_v2_12_hyde.json`) committed and the
   matching tests green.
7. `docs/DECISIONS.md` has decision rows for each phase that
   shipped (1, 2, 3, optionally 4).
8. p99 production retrieval latency measured (sampled over 100
   queries) ≤ 1.5 s **OR** HyDE shipped opt-in with a documented
   latency trade-off.
9. v2.12 cycle Dashscope spend tracked; total ≤ $25 (matches v2.11
   cap).
10. v2.11.x Phase 0 recovery + 30-day rollback drop both completed.

---

## 5. Out of Scope (this draft)

- Local-hosted reranker (Mac Mini MLX) — v2.13 candidate iff
  cost/data-policy shifts.
- Multi-query rewriting beyond single-shot HyDE.
- Cross-encoder reranker fine-tuning on this corpus.
- Schema changes.
- VLM swap (3a carry-forward) unless the soak surfaces image-quality
  drag — current default: stays on the v2.11 baseline.
- UIR refactor (3c carry-forward) — still PAUSED for user signoff.
- Magazine rendered-region-crop (3e) — still deferred per v2.11
  rationale (ceiling is retrieval, not chunk shape).

---

## 6. Decision log (this plan)

| Date | Change |
|---|---|
| 2026-05-21 | Draft v0.1 authored after the v2.11.0 swap landed + was tagged. Order of phases driven by the v2.11 soak shape: Recall@5 doc 91.7% vs Recall@5 chunk 66.8% is a right-doc-wrong-chunk pattern → reranker is the highest-leverage first phase. Phase 2 (hybrid) and Phase 3 (HyDE) are conditional on Phase 1's floor outcomes. Phase 4 (per-doc-class chunking) is a stretch safety valve, not the planned path. Phase 0 closes the v2.11.x Format recovery + 30-day rollback drop before any retrieval work starts so the v2.12 measurements start from a clean Format baseline. |

---

## 7. Open questions

1. **Reranker latency budget.** Empirical p99 measurement needed
   before locking the `top_k_retrieve=50` choice. If reranker adds
   > 500 ms on 50 pairs, drop to 25.
2. **Phase 2 trigger condition.** Default-go vs. trigger-on-Phase-1-
   shortfall. Current draft says trigger-on-shortfall (the
   conservative path that minimizes re-ingestion cost). Reconsider
   if Phase 1 lands close-to-but-not-clearing the Recall@5 chunk
   floor — a single 5h re-ingest may be worth it for the headroom.
3. **HyDE production cost.** $0.001/query × N queries — at what
   production traffic does this stop being trivial? User decision.
4. **Phase 4 trigger.** Sharp threshold (Recall@5 < 80%?) or soft
   judgment call after Phases 1-3? Current draft says judgment
   call — Phase 4 is the heaviest work and shouldn't auto-trigger.

---

**END OF DRAFT v0.1.** Authored 2026-05-21, immediately after the
v2.11.0 swap + doc sweep + tag landed on both `github` and `origin`
(Gitea at `10.0.10.241`). Next checkpoint: user signoff on the
phase ordering and the conditional-go semantics. Promotion to Draft
v1.0 happens when Phase 0 lands (carry-forward close-out is the
first thing that actually executes).
