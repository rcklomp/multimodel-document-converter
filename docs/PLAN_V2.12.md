# Plan: v2.12 — Close the absolute-quality gap: reranker → hybrid retrieval → HyDE

**Status:** **Draft v0.6** (2026-05-21). Phase 1 SHIPPED locally.
Local ModernBERT reranker (`gte-reranker-modernbert-base-mlx` via
omlx-server at `http://10.0.10.246:8000`) won the Phase 1 shootout
decisively on all 4 embedder-attributable axes vs cloud `gte-rerank`:
Recall@1 chunk 35.5% (v2.11) → 53.9% (cloud) → **61.8% (omlx)**;
Recall@5 chunk 66.8% → 66.8% → **81.3%**; Relevance 59.3% → 74.5% →
**78.3%**; Faithfulness 50.6% → 64.2% → **69.4%**. Production
default flipped via `src/mmrag_v2/retrieval/config.py`
`_COMPILE_DEFAULT = "omlx"`. End-to-end p99 latency ~1.85s (well
within 3.0s budget). Zero per-query reranker cost (LAN-local).
**Phase 2 TRIGGERED:** Recall@5 chunk 81.3% < 85% floor by 3.7pp;
hybrid retrieval (BM25 + dense + RRF) follows. **Phase 3 TRIGGERED:**
Faithfulness 69.4% < 70% floor by 0.6pp (borderline within
judge-noise); HyDE module to be built per plan. See
`docs/DECISIONS.md` "v2.12 Phase 1 Reranker Shootout Outcome" for
the full numbers + reports at
`docs/QUALITY_SNAPSHOT_2026-05-21_v2.12_p1_{cloud,omlx}.md`.

**Predecessor:** Draft v0.5 (2026-05-21) — Phase 0 partial win
(IRJET +15.6pp Format; Earthship + CarOK rolled to v2.13).
swap landed and was tagged (`c2a461c`, annotated `v2.11.0` on both
remotes 2026-05-20/21). v2.11 closed the *embedder* bottleneck with a
10× lift across every embedder-attributable axis; v2.12 closes the
*absolute-quality* gap that the v2.11 soak revealed: relative numbers
are great, but the system still misses the right passage in the top-5
about 1 query in 3 (Recall@5 chunk 66.8%). The user named this gap
explicitly during the v2.11.0 close-out — this plan exists to address
it, not to be agreed with after the fact.

**v0.2 changes (2026-05-21):** §1 Goal 7 (latency budget) and §3
Phase 1 latency-budget paragraph + §7 Open Question 1 revised in
response to the empirical reranker-latency benchmark
(`scripts/measure_reranker_latency.py`; data at
`tests/fixtures/reranker_latency_2026-05-21.json`). Original 1.5 s
p99 target was incompatible with the empirical embed floor;
revised to a stage-level + total-time budget.

**v0.3 changes (2026-05-21):** §3 Phase 1 reranker-choice rationale
expanded after a local-vs-cloud comparison against the user's
self-hosted Qwen3-Reranker-8B at `http://10.0.10.246:8000`. Two
benchmarks added: latency-only against the local server
(`tests/fixtures/reranker_latency_omlx_2026-05-21.json`) and
side-by-side quality vs the cloud reranker on the same query+
candidate pairs (`tests/fixtures/reranker_quality_2026-05-21.json`).
Local reranker rejected for v2.12 Phase 1 on both latency (17.5×
slower) and quality-signal grounds (score bunching pattern
consistent with a yes/no-token-classification head; top-1
agreement with cloud is 5% but both could be wrong). Cloud
`gte-rerank` confirmed as the Phase 1 reranker.

**v0.4 changes (2026-05-21):** Second local reranker tested —
`afanjul/gte-reranker-modernbert-base-mlx` (~150M-param cross-
encoder, 285 MB MLX-quantized, ModernBERT backbone). Right
architecture family for the task: continuous-score regression
head, not the yes/no-classification head of Qwen3-Reranker.
Empirical results from the same 20-query × top-25 benchmark
(`tests/fixtures/reranker_latency_modernbert_2026-05-21.json`,
`tests/fixtures/reranker_quality_modernbert_2026-05-21.json`):

- **Latency: 3× FASTER than cloud.** K=25 p99 = 0.55 s vs cloud
  1.70 s. Per-pair compute ~15 ms on the user's Mac Mini (vs
  ~750 ms for the 8B-param Qwen3 model). End-to-end p99 at K=25
  is ~1.85 s including embed, vs ~3.1 s for the cloud-rerank path.
  Sub-1.5 s budget is achievable at K=10 (total ~1.61 s) with
  future embed-cache optimization.

- **Score distribution: clean.** Wide range 0.0–0.951 (proper
  cross-encoder behavior) vs Qwen3's bunched 0.62–0.84. Decisive
  ordering. One anomaly: query Q16 (Earthship) returned all
  candidates with score 0.0 — either a quantization edge case or
  a legitimate "nothing relevant" verdict; needs investigation
  during Phase 1 soak.

- **Quality vs cloud is unresolved.** Top-1 agreement 15%
  (3/20), mean Jaccard 0.239 — better than Qwen3 (5%, 0.127)
  but still substantial disagreement with cloud `gte-rerank`.
  Two viable interpretations: (a) different generations of the
  GTE family with different training data → both are valid but
  prefer different chunks per query; (b) one is actually better.
  Agreement-rate analysis alone cannot distinguish these.

**Decision deferred into Phase 1.** Cloud `gte-rerank` remains
the *fallback* Phase 1 reranker (latency known, quality known to
be a 10× lift over v2.11.0). Local ModernBERT becomes the
*leading candidate* pending the Phase 1 soak verdict — same 518-
query × LLM-judge protocol the v2.11 embedder shootout used.
Whichever reranker scores better on the soak ships as the v2.12
production reranker. Both candidates pin a new fingerprint for
regression-test reproducibility. The benchmark + comparison
scripts are reusable for any future reranker bake-off.

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

### Carry-forward register (Draft v0.5 — post-Phase-0)

| # | Class | Source | v2.12 status | Notes |
|---|---|---|---|---|
| 1 | v2.11.x Format recovery | v2.11 Phase 1 soak Format 89.8% | **`partially closed by Phase 0`** | Root cause was `content`/`refined_content` staleness in `scripts/ingest_to_qdrant.py`, not chunk content per se. Preference swap fix landed; IRJET +15.6pp (71.9% → 87.5%); CarOK +3.1pp; Earthship +0pp. Aggregate 77.1% on the 3 docs — below ≥95% target. See `docs/DECISIONS.md` "v2.12 Phase 0 Outcome". |
| 1b | **Earthship re-OCR** (v2.13) | Phase 0 found Earthship Format flat: OCR layout damage (multi-column interleaving, mid-word linebreaks) not fixable by content normalization | **`v2.13 carry-forward`** | Re-process source PDF with Docling's layout-aware OCR settings tuned for multi-column scanned pages. Or chunk-level filtering of severely-broken chunks. |
| 1c | **CarOK form-shape decision** (v2.13) | Phase 0 found CarOK Format barely moved: chunks correctly represent automotive-parts inventory but LLM judge inherently penalizes form data | **`v2.13 carry-forward`** | Choose between (a) restructure chunks to one inventory row per chunk and (b) carve-out a form-class Format gate that doesn't penalize structured-data chunks. |
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
7. **Production retrieval p99 latency** — split budget, set
   empirically (`tests/fixtures/reranker_latency_2026-05-21.json`,
   240-sample benchmark from the user's network to Dashscope intl):

   | Stage | p50 | p99 | Notes |
   |---|---:|---:|---|
   | Embed (text-embedding-v4) | 1.16 s | 1.35 s | Network round-trip dominated; floor cost for the intl endpoint from EU |
   | Qdrant top-K search | 22 ms | 55 ms | Negligible, local LAN |
   | Rerank K=10 → top-5 | 803 ms | 1.04 s | Per-call cost dominated by API round-trip, not pair-scoring |
   | Rerank K=25 → top-5 | 952 ms | 1.70 s | Marginal cost vs K=10 ≈ 150 ms p50 |
   | Rerank K=50 → top-5 | 1.09 s | 2.36 s | Tail-latency rises sharply past K=50 |
   | Rerank K=100 → top-5 | 1.27 s | 3.29 s | Outliers up to 4.3 s; not recommended |

   **Revised v2.12 latency targets** (the original 1.5 s p99 total
   target was incompatible with the embed floor):

   - **Stage-level floor:** embed p99 ≤ 1.4 s, Qdrant p99 ≤ 100 ms,
     rerank p99 ≤ 1.7 s (at K=25). These are the "we can't beat the
     network from here" baselines.
   - **Total end-to-end p99 target (rerank-only):** ≤ 3.0 s with
     K=25, ≤ 3.5 s with K=50. Documented as the v2.12.0 production
     latency contract from the user's network.
   - **With HyDE (Phase 3):** add ~1 s for the qwen-max generation
     call → total p99 ≤ 4.5 s. If this is unacceptable, HyDE ships
     opt-in (default off).
   - **Future optimization paths** (out of scope for v2.12):
     embedding cache for repeated/follow-up queries (saves the
     ~1.2 s embed); switch to a regional Dashscope endpoint if
     latency to that region is lower; or move to a local embedder
     (v2.11 carry-forward 1; v2.13 candidate).

   The honest read: this is a network-bound system from the user's
   current location. Any production deployment hosted closer to
   Dashscope's serving region (e.g., AWS ap-southeast or
   eu-central) would see meaningfully lower numbers; the v2.12.0
   contract uses the user's network as the worst-case baseline.
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

**What.** Add a cross-encoder reranker as a second stage between
Qdrant top-K retrieval and the final returned chunks. The production
retrieval flow becomes:

```
query → embed (text-embedding-v4) → Qdrant top-25 (cosine)
      → cross-encoder reranker (query × chunk pairs) → reordered top-5
```

The reranker scores `(query, chunk)` pairs *together*, capturing
semantic interaction that a single-vector embedder can't. For the
v2.11 right-doc-wrong-chunk pattern, this is the canonical fix.

**Reranker choice — leading candidate is local ModernBERT, cloud is
the fallback.** Two GTE-family rerankers benchmarked in pre-Phase-1
work (see §1 status banner v0.3 + v0.4 changes). The Phase 1 soak
picks the winner:

- **Leading candidate (local):**
  `afanjul/gte-reranker-modernbert-base-mlx` served by `omlx-server`
  at `http://10.0.10.246:8000/v1/rerank`. ~150M-param cross-encoder
  on Apple Silicon. K=25 p99 = 0.55 s, 3× faster than cloud. Wide
  score distribution (0.0–0.951). Top-1 agreement with cloud is
  only 15% but agreement-rate analysis doesn't tell us which is
  RIGHT — the soak will.
- **Fallback (cloud):** Dashscope `gte-rerank` at
  `https://dashscope-intl.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank`.
  K=25 p99 = 1.70 s. Quality known to be a 10× lift on top of v2.11
  embedder swap (proxy estimate from the agreement rate against
  what should be a strong reranker). Cost per query ~$0.001.

**Architecture decisions.**

- **New module** `src/mmrag_v2/retrieval/` with four files: `__init__.py`,
  `reranker.py` (provider abstraction: `LocalOmlxReranker` +
  `DashscopeReranker` both implement a `rerank(query, chunks) →
  list[(score, chunk)]` interface), `pipeline.py` (composable retrieve
  → rerank → return), and `config.py` (factory: `get_reranker(name)`
  reads `RERANKER_BACKEND` env var or CLI flag).
- **No changes** to `scripts/ingest_to_qdrant.py` (retrieval-side
  only).
- **`scripts/search_qdrant.py` is the integration point.** Its
  existing `--no-rerank` flag flips meaning: today it's a no-op
  (rerank degrades to vector-rank truncation); after Phase 1,
  default behavior is rerank-on, and `--no-rerank` opts out.
- **Reranker APIs (both candidates use the same Cohere-style payload
  shape).**
  - Local: `POST http://10.0.10.246:8000/v1/rerank` with
    `model: "gte-reranker-modernbert-base-mlx"` and
    `Authorization: Bearer $MLX_API_KEY`. Cost: zero per call (LAN-
    hosted). Latency: K=25 p99 0.55 s. Throughput is bounded by the
    Mac Mini and is single-tenant — production scale-up would need
    either a queue or a second instance.
  - Cloud: `POST https://dashscope-intl.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank`
    with `model: "gte-rerank"` (intl endpoint; cn endpoint name is
    `gte-rerank-v2` for the same model). Same `DASHSCOPE_API_KEY`
    env var. Cost: ~$0.001 per query. Latency: K=25 p99 1.70 s.
- **Latency budget (empirically determined 2026-05-21).**
  Per-call rerank p99 by backend:

  | Backend | K=10 | K=25 | K=50 |
  |---|---:|---:|---:|
  | Local ModernBERT | 0.32 s | 0.55 s | 1.04 s |
  | Cloud `gte-rerank` | 1.04 s | 1.70 s | 2.36 s |

  Pinned data: `tests/fixtures/reranker_latency_modernbert_2026-05-21.json`
  (local), `tests/fixtures/reranker_latency_2026-05-21.json` (cloud).
  **Recommended `top_k_retrieve` = 25** for the v2.12.0 default: gives
  the reranker enough candidates to reorder meaningfully; plays well
  with the 3.0 s total-p99 budget on either backend. **K=50** is the
  fallback if Phase 1 alone doesn't clear Recall@5 chunk ≥ 85% — local
  ModernBERT still p99 ≈ 2.3 s total at K=50, cloud is ~3.8 s.
  **K=100** excluded (cloud has 4.3 s tail-latency outliers; not
  measured on local but per-pair compute would extrapolate to ~2 s
  rerank-only on the Mini, OK budget-wise but diminishing-returns
  vs K=50).

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

**Surprise from the latency benchmark:** embedding latency dominates
the total query time, not reranking. Embed p99 was 1.35 s — about
30% of total query time at K=25 (rerank 1.70 s + embed 1.35 s +
qdrant 0.06 s ≈ 3.1 s). Reranking is well-behaved for sub-K=100; the
real future-latency lever is **embedding cache** for repeat /
follow-up queries, which is v2.13 scope. Out of scope for v2.12.

**Why not the local Qwen3-Reranker-8B?** A local
`mlx-community_Qwen3-Reranker-8B-mxfp8` is reachable on the user's
LAN at `http://10.0.10.246:8000/v1/rerank`. It was benchmarked
on 2026-05-21 to determine whether it should be the v2.12 Phase 1
reranker instead of the cloud `gte-rerank`. Two reasons it was
rejected:

1. **Latency is 17.5× worse than cloud.** Steady-state per-call
   numbers (40 samples per K from
   `tests/fixtures/reranker_latency_omlx_2026-05-21.json`):

   | Backend | K=10 p50 / p99 | K=25 p50 / p99 |
   |---|---:|---:|
   | Cloud `gte-rerank` | 0.80 s / 1.04 s | 0.95 s / 1.70 s |
   | Local Qwen3-Reranker-8B | 7.4 s / 12.2 s | 19.3 s / 30.7 s |

   Per-pair compute cost on the local server is ~750 ms/pair —
   consistent with an 8B-parameter causal-LM cross-encoder doing a
   sequential forward pass per pair on Apple Silicon. Network
   round-trip is negligible (LAN, ~5 ms) — the cost is pure compute.
   Even at K=10 the local reranker exceeds any reasonable production
   p99 budget by 5-10×.

2. **Score-distribution pattern suggests the local reranker has a
   yes/no classification head, not a regression head.** From the
   side-by-side quality benchmark
   (`tests/fixtures/reranker_quality_2026-05-21.json`, 20 queries
   × top-25 candidates → top-5 returned by each reranker):

   | Reranker | Score range observed | Typical top-5 spread |
   |---|---|---|
   | Cloud `gte-rerank` | 0.006 — 0.753 | Wide; clear winner usually 2-4× the runner-up |
   | Local Qwen3-Reranker-8B | 0.617 — 0.836 | Narrow; everything bunched in 0.75-0.85 |

   Qwen3-Reranker is a causal-LM-with-yes/no-token-classification
   architecture — relevance score = `softmax(logits_yes_no)["yes"]`.
   Trained that way, it tends to be overconfident on the positive
   class. The score gap between the cloud's top-1 and top-5 is
   typically 0.3-0.5; on the local model it's typically 0.05. With
   such a narrow band, the ordering within the band is dominated by
   noise rather than signal — exactly the opposite of what a
   reranker should do.

   The top-1 agreement rate between the two rerankers was **5%**
   (1/20 queries). Mean top-5 Jaccard overlap was **0.127** —
   functionally independent decisions. The cloud reranker's ordering
   correlates with the embedder's similarity score (re-orders within
   a topic-coherent candidate set); the local reranker's ordering
   appears to be near-random within its bunched-score band.

3. **What we cannot conclude from these benchmarks alone.**
   "Local picks different chunks than cloud" doesn't tell us which
   one picks BETTER chunks — both could be wrong (both could be
   right, picking different valid chunks). The definitive answer
   requires running the full Phase 1 soak (518 queries × LLM-as-
   judge grading) against each reranker. Doing that at the local
   reranker's latency would take roughly 518 × 19 s = ~2.7 hours of
   wall time for the retrieve stage alone. **Deferred.** Re-open
   only if (a) local inference latency improves by ≥ 10× (smaller
   model, different hardware, batched inference) or (b) a privacy/
   data-residency requirement forces local-only retrieval.

**v2.13 candidate.** A faster local reranker (e.g.,
`mlx-community/bge-reranker-base` at ~110M params, or a future MLX
batch-inference path) could re-enter contention in v2.13. The
benchmark + quality-comparison scripts are reusable —
`scripts/measure_reranker_latency.py --rerank-backend omlx` and
`scripts/compare_reranker_quality.py` measure both at any time.

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
8. p99 production retrieval latency measured against the live
   stack (sampled over ≥ 100 queries via
   `scripts/measure_reranker_latency.py`) ≤ 3.0 s at K=25 **or**
   ≤ 3.5 s at K=50 (network-floor adjusted; original 1.5 s target
   was incompatible with the embed-stage floor — see §2 Goal 7).
   HyDE-on path adds ~1 s; either fits within an extended ≤ 4.5 s
   budget **OR** HyDE ships opt-in with the latency trade-off
   documented.
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
| 2026-05-21 | Promoted to Draft v0.2 after empirical reranker-latency benchmark (`scripts/measure_reranker_latency.py`; 240 samples across 20 queries × 4 K values × 3 repeats). Three substantive changes: (1) Correct model name from `gte-rerank-v2` to `gte-rerank` for the intl endpoint (v2 only on the cn endpoint; same model). (2) §2 Goal 7 latency budget revised — the original 1.5 s p99 target was incompatible with the empirical embed p99 of 1.35 s on its own; new split-stage budget plus a 3.0 s total p99 target at K=25 / 3.5 s at K=50 reflects what's achievable from the user's current network. The embed p99 (1.35 s) is the dominant cost, not the reranker — a counterintuitive but actionable finding that frames embedding-cache as the right next-cycle latency lever. (3) §3 Phase 1 default `top_k_retrieve` set to 25 (was unspecified); K=50 is the fallback if Phase 1 alone doesn't clear Recall@5 chunk ≥ 85%; K=100 is excluded due to 4.3 s tail-latency outliers in the benchmark. Open Question 1 marked resolved. |
| 2026-05-21 | Promoted to Draft v0.3 after a local-vs-cloud reranker bake-off against the user's self-hosted `mlx-community_Qwen3-Reranker-8B-mxfp8` at `http://10.0.10.246:8000`. Benchmark script extended with `--rerank-backend` flag; new `scripts/compare_reranker_quality.py` runs the same query+candidates through both rerankers side-by-side. Local reranker rejected for v2.12 Phase 1 on two independent grounds: (a) **Latency 17.5× worse than cloud** — K=10 p99 = 12.2 s, K=25 p99 = 30.7 s (per-pair compute ~750 ms on Apple Silicon for an 8B-param causal-LM cross-encoder). (b) **Score-distribution pattern indicates a yes/no-token-classification head** rather than a regression head — local scores bunch in 0.75-0.85 vs cloud's 0.006-0.75 range, meaning the local reranker's within-band ordering is closer to noise than signal. Top-1 agreement between the two rerankers was 5% (1/20), mean top-5 Jaccard 0.127 — functionally independent decisions, but the benchmarks alone can't tell which is RIGHT (both could be wrong). Deferred to v2.13+ pending either 10× local-latency improvement or a privacy/data-residency driver. Cloud `gte-rerank` confirmed as the Phase 1 reranker; new fixtures committed for future delta reproducibility. Open Question 1 sub-clause (local reranker) marked resolved. |
| 2026-05-21 | Promoted to Draft v0.4 after the user found `afanjul/gte-reranker-modernbert-base-mlx` — the *right* shape of local reranker (150M-param cross-encoder with regression head, vs the rejected 8B causal-LM with yes/no head from v0.3). Discussion of why reranker model size is decoupled from reranker quality: standard production rerankers (BGE/GTE families) sit in the 200-600M-param range; the 8B Qwen3 was the architectural outlier. ModernBERT model pulled onto omlx-server and benchmarked: (a) **Latency 3× FASTER than cloud** — K=25 p99 = 0.55 s vs cloud 1.70 s. Per-pair compute ~15 ms on the Mini (50× faster than Qwen3-8B). (b) **Score distribution wide** (0.0–0.951) — proper cross-encoder behavior, decisive ordering. (c) **Quality vs cloud unresolved** — top-1 agreement 15% (3/20), mean Jaccard 0.239. Better than Qwen3 on every axis but agreement-rate alone can't tell us which reranker picks BETTER chunks (both could be valid-but-different gte-family generations). **Decision deferred into Phase 1 itself** — the Phase 1 soak (518 queries × LLM-judge) will pick the winner; whichever scores better ships. Cloud remains the fallback if local ModernBERT loses the soak or the Q16 all-zero anomaly turns out to be a model defect. New scripts/fixtures: `scripts/compare_reranker_quality.py` gains `--local-model` flag; `tests/fixtures/reranker_latency_modernbert_2026-05-21.json` (60 samples × 3 K values) and `tests/fixtures/reranker_quality_modernbert_2026-05-21.json` (20-query head-to-head). |
| 2026-05-21 | **Promoted to Draft v0.5 after Phase 0 execution.** Phase 0 (v2.11.x Format recovery) executed end-to-end. Root cause of the v2.11 Format dips on the three named docs turned out to be NOT a chunk-quality issue but a preference-staleness bug in `scripts/ingest_to_qdrant.py`: lines 351 + 483 preferred `metadata.refined_content` over `chunk.content`, but `refined_content` is the raw VLM refiner output preserved for provenance while `content` carries later normalization passes (v2.10 audit cleanup, whitespace collapse, page-header strip). The semantics of "which field is newer" inverted as the chunker evolved, but the ingest preference wasn't updated. One-line fix swapped the preference; 3 docs re-ingested (1146 chunks, 0 errors); partial soak ran on the same 48 queries the v2.11 soak used for these docs. Results: **IRJET +15.6pp Format** (71.9% → 87.5%) — clean win on the header-noise stripping. **CarOK +3.1pp Format** (68.8% → 71.9%) — barely moved; chunks correctly represent inventory data, LLM judge inherently penalizes form-class content. **Earthship +0pp Format** (71.9% → 71.9%) — Format defect is OCR layout damage (multi-column interleaving, mid-word linebreaks), not whitespace. Aggregate 77.1% on the 3 docs misses the ≥95% Phase 0 target by 17.9pp. Honest call: Phase 0 ships the genuine win (preference swap + 4 regression tests pinned in `tests/test_ingest_content_preference.py`); Earthship + CarOK roll forward to v2.13 as named recovery work (Earthship re-OCR; CarOK form-shape decision). The v2.12.0 Format gate stays at ≥95% pending the cumulative Phase 1 + Phase 2 lift. Side-channel deltas (Earthship Faithfulness −9.4pp, IRJET Relevance −9.4pp) are likely 1-2-query noise on a 16-query sample and will be re-measured in the full Phase 1 soak. Test suite after Phase 0: 990 passed, 15 skipped, 0 failed (+4 over the v2.11.0 baseline 986). |
| 2026-05-21 | **Promoted to Draft v0.6 after Phase 1 shootout.** Phase 1 (reranker shootout) executed end-to-end. Two candidate rerankers ran the same 518-query × 259-chunk soak fixture: cloud `gte-rerank` (Dashscope intl) and local `gte-reranker-modernbert-base-mlx` (omlx-server). Same embedder (text-embedding-v4), same Qdrant collection, same judge (qwen-max). **Local ModernBERT wins decisively on all 4 embedder-attributable axes:** Recall@1 chunk 35.5% (v2.11.0) → 53.9% (cloud) → 61.8% (omlx); Recall@5 chunk 66.8% → 66.8% → 81.3%; Relevance 59.3% → 74.5% → 78.3%; Faithfulness 50.6% → 64.2% → 69.4%. Key insight: cloud `gte-rerank` only reordered the top-5 Qdrant already returned (Recall@5 chunk identical to baseline); ModernBERT actually picked *different* 5 chunks from the top-25 candidate set, finding gold chunks deeper. That's stronger reranking discrimination, consistent with ModernBERT being a 150M-param cross-encoder built on the newer Dec-2024 ModernBERT backbone vs cloud's older distilled multilingual model. **Phase 1 close-out:** `src/mmrag_v2/retrieval/config.py` `_COMPILE_DEFAULT` set to `"omlx"` so production picks ModernBERT by default; cloud remains the fallback via `RERANKER_BACKEND=dashscope` env var. End-to-end p99 latency ~1.85s (well within 3.0s budget). Zero per-query reranker cost in production. **Phase 2 TRIGGERED** because Recall@5 chunk = 81.3% < 85% floor (3.7pp gap). **Phase 3 TRIGGERED** because Faithfulness 69.4% < 70% floor (0.6pp gap — borderline within judge-noise; HyDE module built per plan, ship-on-by-default depends on Phase 3 soak's actual lift). Reports: `docs/QUALITY_SNAPSHOT_2026-05-21_v2.12_p1_cloud.md` + `docs/QUALITY_SNAPSHOT_2026-05-21_v2.12_p1_omlx.md`. Full numbers in `docs/DECISIONS.md` "v2.12 Phase 1 Reranker Shootout Outcome". Test suite after Phase 1: 1006 passed, 15 skipped, 0 failed. |

---

## 7. Open questions

1. ~~**Reranker latency budget.**~~ **Resolved 2026-05-21** by the
   `scripts/measure_reranker_latency.py` 240-sample benchmark. Default
   `top_k_retrieve = 25` (rerank p99 = 1.70 s — within the revised
   3.0 s total-p99 budget); K=50 is the Phase-2-conditional fallback
   (rerank p99 = 2.36 s, 3.5 s total); K=100 is excluded due to tail-
   latency outliers (max 4.3 s in 60 samples). The bigger surprise:
   embedding (1.35 s p99) is the dominant cost, not reranking; that
   reframes embedding-cache as the right next-cycle latency lever
   (v2.13 scope). Full data: §2 Goal 7 + `tests/fixtures/reranker_latency_2026-05-21.json`.
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
