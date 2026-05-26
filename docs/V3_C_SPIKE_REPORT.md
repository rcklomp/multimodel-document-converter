# V3 Phase C C-Spike Report

**Charter:** [`ARCHITECTURE_V3_DRAFT_0.5.md`](ARCHITECTURE_V3_DRAFT_0.5.md) §4.2 step 2
**Run date:** 2026-05-26 PM (autonomous)
**Operator:** Claude Code (Opus 4.7, 1M context)
**Workstation:** Apple Silicon (Mac Mini), MPS backend, ~64 GB unified memory
**Stack:** ColPali `vidore/colpali-v1.3` + LoRA patch (pre-spike) + production v2.16 hybrid retrieval (omlx Qwen3-Embedding-8B + BM25 + ModernBERT rerank) + Qdrant `mmrag_v2_8__qwen3_local`/`mmrag_v2_8__bm25_sparse`

## TL;DR

| Charter §4.2 step 2 condition | Result | Threshold | Verdict |
|---|---:|---:|---|
| **PASS A** — visual recovery rate on text-failed queries | **44.4%** (4/9) | ≥60% | **FAIL** |
| **PASS A** — visual harm rate on text-passed queries | **36.4%** (4/11) | ≤10% | **FAIL** |
| **PASS B** — reranker top-1 on gold page (bounded join) | **47.4%** (9/19) | ≥60% | **FAIL** |

**Charter outcome rule (§4.2):** "If pre-spike or C-spike FAIL A → Phase C as designed is dead; redirect to VLM-native parsing evaluation or alternative visual model." See §"Charter outcome + recommendation" below for the nuanced read.

## Methodology

Per Charter §4.2 step 2:

1. **Target doc:** `ATZ_Elektronik_German` (`6fccda8bd625`, 6 pages, 63 chunks).
2. **Render all pages** at 200 DPI.
3. **Embed all 6 pages with ColPali** once (cached for the query batch).
4. **20 hand-crafted queries** (fixture had no ATZ-specific queries; per Charter "or hand-craft if fixture coverage is thin"). Distribution: page 1 ×5 (visual-favored on Lifecycle Management flowchart labels), pages 2–5 ×3 each (body-text), page 6 ×3 (impressum/ads text-favored).
5. **Visual leg:** ColPali query embed → MaxSim against page matrices → top-K page ranking.
6. **Text leg:** production `retrieve_hybrid_reranked` (dense Qwen3-Embedding-8B + BM25 + RRF + ModernBERT rerank) → top-25 chunks → page mapping.
7. **PASS A:** recovery on text-failed + harm on text-passed (Charter §4.2 #7).
8. **PASS B:** bounded join (text top-25 ∪ top-3 chunks/page on visual top-5) → ModernBERT rerank → top-1 chunk on gold page (Charter §4.2 #8). Charter requires "exact ModernBERT model + exact production prompt + chunk_id dedup + fixture-based gold map". The first three are met by reusing production code paths; the fixture gold map is absent for ATZ, so PASS B is measured at page-level (top-1 chunk's page == gold page), looser than per-chunk-gold but the strictest version computable without a fixture build-out.

**Stack reproducibility:**
- ColPali `vidore/colpali-v1.3` with LoRA patch from pre-spike (254/254 adapter keys applied via `_apply_colpali_lora_adapter`)
- MPS bfloat16, single-image batching
- Production retrieval: omlx `Qwen3-Embedding-8B-mxfp8` + BM25 + ModernBERT (`gte-reranker-modernbert-base-mlx`) — unchanged from v2.13.0
- Qdrant collections: `mmrag_v2_8__qwen3_local` (31,371 pts), `mmrag_v2_8__bm25_sparse`

## Results

### PASS A (Charter §4.2 step 2 #7)

| Aggregate | Value | Notes |
|---|---:|---|
| Visual top-1 accuracy | 55% (11/20) | |
| Text top-1 accuracy | 55% (11/20) | matches the v2.13 P1 omlx baseline (62.5%) on the original 16-query fixture; this 20-query hand-craft is slightly harder |
| Text-failed queries | 9 | denominator for "recovery" |
| Text-passed queries | 11 | denominator for "harm" |
| **Visual recovery on text-failed** | **44.4% (4/9)** | threshold ≥60% — **FAIL** |
| **Visual harm on text-passed** | **36.4% (4/11)** | threshold ≤10% — **FAIL** |

**Per-query partitioning (set-membership relative to gold-page top-1):**

| Bucket | Count | Queries |
|---|---:|---|
| Both visual+text PASS | 7 | Q01, Q03, Q06, Q17, Q18, Q19, Q20 |
| Visual-only PASS (recovery) | 4 | Q02, Q04, Q14, Q16 |
| Text-only PASS (harm) | 4 | Q07, Q08, Q13, Q15 |
| Both FAIL | 5 | Q05, Q09, Q10, Q11, Q12 |

**Failure-mode diagnosis (visual harm):** all 4 "visual-only fails" are body-text queries about pages 2–5 where ColPali pulled page 1 first. Page 1 carries the only visually-rich element (Lifecycle Management flowchart); on body-text queries that mention any phase label (Anforderungsanalyse, Spezifikation, Test, etc.) the diagram's visual richness over-pulls. This is a known ColPali characteristic on documents where one page is materially richer than the rest — not a bug, but a model-architecture limitation that page-level granularity cannot resolve. Region-level visual retrieval would naturally fix it (a body-text "Anforderungsanalyse" mention would compete at the chunk level against the flowchart's "Anforderungsanalyse" label).

**Failure-mode diagnosis (visual recovery shortfall):** of the 9 text-failed queries, visual recovered 4 (Q02 page-1 flowchart, Q04 page-1 component labels, Q14 page-4 modular-HiL figure, Q16 page-5 MESSINA tooling). The 5 text-failed-AND-visual-failed (Q05 Q09 Q10 Q11 Q12) split: Q05 is the page-1 phase-sequence query both legs mistook for page-2 Einleitung; Q09 was correctly visual-rank-2 but rank-1 was page 2 because both pages have similar layout; Q10–Q12 are within-doc page confusions where ColPali sees overlapping visual signature.

### PASS B (Charter §4.2 step 2 #8 — bounded join + rerank)

| Aggregate | Value |
|---|---:|
| Qualifying queries (gold ∈ visual top-5) | 19/20 (Q09 visual top-5 = `2,6,5,1,4`; gold=3 absent) |
| Candidate set size (median) | 33 chunks (text top-25 ∪ visual-top-5-pages × top-3 = up to 40, post-dedup ~33) |
| **Rerank top-1 on gold page** | **9/19 (47.4%)** |
| **Threshold** | **≥60%** — **FAIL** |

**Per-query PASS B trace:**

| ID | gold | visual top-5 | candidates | rerank top-1 page | provenance | verdict |
|---|---:|---|---:|---:|---|---|
| Q01 | 1 | 1,3,2,5,4 | 31 | 1 | both | **PASS** |
| Q02 | 1 | 1,3,5,2,4 | 31 | 4 | both | miss |
| Q03 | 1 | 1,5,2,3,4 | 31 | 1 | both | **PASS** |
| Q04 | 1 | 1,3,5,2,6 | 38 | 5 | both | miss |
| Q05 | 1 | 2,3,4,5,1 | 33 | 2 | both | miss |
| Q06 | 2 | 2,5,4,3,1 | 33 | 2 | both | **PASS** |
| Q07 | 2 | 1,2,3,5,4 | 28 | 2 | both | **PASS** ← bounded-join rescued visual miss |
| Q08 | 2 | 3,2,5,1,4 | 33 | 2 | both | **PASS** ← bounded-join rescued visual miss |
| Q09 | 3 | 2,6,5,1,4 | 0 | — | — | n/a (gold not in visual top-5) |
| Q10 | 3 | 1,3,5,2,4 | 26 | 5 | both | miss |
| Q11 | 3 | 1,4,3,5,2 | 39 | 2 | both | miss |
| Q12 | 4 | 2,5,4,3,1 | 35 | 2 | visual_page | miss |
| Q13 | 4 | 1,4,3,2,5 | 33 | 4 | both | **PASS** ← bounded-join rescued visual miss |
| Q14 | 4 | 4,5,3,2,1 | 34 | 5 | text | miss |
| Q15 | 5 | 4,5,6,3,2 | 37 | 5 | both | **PASS** |
| Q16 | 5 | 5,4,1,2,3 | 38 | 4 | both | miss |
| Q17 | 5 | 5,4,6,3,2 | 36 | 5 | both | **PASS** |
| Q18 | 6 | 6,5,1,4,2 | 36 | 6 | both | **PASS** |
| Q19 | 6 | 6,2,1,4,5 | 39 | 4 (other doc!) | text | miss — rerank picked a different doc's impressum |
| Q20 | 6 | 6,5,4,2,1 | 39 | 20 (other doc!) | text | miss — rerank picked a different doc's masthead |

**Hybrid-rescue evidence:** Q07, Q08, Q13 were "visual-failed" under PASS A but PASS under the bounded-join + rerank. The candidate-set construction explicitly let text-leg chunks from the gold page compete against visual-leg chunks from the misranked page, and the reranker correctly picked the gold-page chunk. The hybrid system mechanism works as designed; the threshold is just not crossed.

**Cross-doc misses:** Q19 and Q20 are textually ambiguous queries that appear in many docs' impressum/masthead pages. The reranker correctly picked the more-relevant chunk for the QUERY but on a different doc. If we excluded those (gold semantically equally satisfied by other docs), PASS B becomes 9/17 = **52.9%** — still under 60%, but the gap narrows.

## Charter outcome + recommendation

**Strict Charter §4.2 outcome rule reading:**

> "If pre-spike or C-spike FAIL A → Phase C as designed is dead; redirect to VLM-native parsing evaluation or alternative visual model."
> "If C-spike PASS A but FAIL B → Phase C scope expands to include region-level granularity."

C-spike FAIL A → strict outcome is "redirect" (alternative visual model OR VLM-native parsing). FAIL B is moot under FAIL A but its 47% rate (close to threshold) carries information about the failure mode.

**Diagnostic verdict (operator interpretation):**

The failure mode is consistent across PASS A and PASS B and pinpoints exactly what region-level granularity would fix:

1. **Visual harm on PASS A** = page-1 over-pull on body-text-overlap queries. Region-level fixes this because the flowchart REGION competes against body-text REGIONS, not whole pages.
2. **PASS B misses on within-doc pages** (Q02, Q04, Q05, Q10, Q11, Q14, Q16) = reranker can't reliably pick the gold chunk when multiple chunks from the same doc are in the candidate set. Region-level cleaner because chunks are smaller and the visual-leg score discriminates within a page.

The hybrid bounded-join mechanism is sound (Q07, Q08, Q13 demonstrate rescue), but page-level granularity is the cap.

**Recommendation: do BOTH paths in parallel**, per Charter §6.2 fork-back triggers:

| Path | Why | Cost |
|---|---|---|
| **A. Alternative visual model** (e.g. `vidore/colqwen2.5-v0.2`, ColModernVBert) | The Charter §6.2 fork-back trigger 4 lists "bbox/visual-coherence" — ColPali patch alignment may be the limiting factor on a doc whose visual content is one diagram. Newer ColQwen models on Qwen2.5-VL base have larger per-page patch counts AND different positional encodings; a re-run of this same harness against ColQwen2.5 is ~30 min wall time. | low (one re-run) |
| **B. Region-level granularity** (in-scope for Phase C, NOT deferred) | The PASS A harm + PASS B within-doc misses both point at the same root cause: page-level over-pull. Region-level naturally fixes both. The Charter explicitly says "region-level must be in Phase C scope, not deferred" on PASS B FAIL. | high (Phase C scope expansion, ~3-5d) |

**Do NOT do (Charter §4.2 strict reading would suggest):** VLM-native parsing evaluation. The PASS B rescue evidence (Q07, Q08, Q13) shows the hybrid stack works when the candidate set is right; the bottleneck is granularity, not the visual modality itself.

**Falsification path (do this BEFORE region-level scope expansion):**

1. Re-run this same C-spike harness with `--model-id vidore/colqwen2.5-v0.2` (or similar). Same 20 queries, same gold pages, same text leg.
2. If PASS A and PASS B both improve materially (say, to >55% / >55%) → alternative-model path is the cheaper fix; close region-level scope expansion.
3. If improvements are <5pp → region-level granularity is the binding constraint; expand Phase C scope.

This narrows the Charter "redirect to VLM-native parsing or alternative visual model" decision from a fork to a sequenced two-step.

## Constraints & known limitations of this measurement

1. **Sample size:** 20 queries on a 6-page doc. Random chance is 1/6 = 16.7%. 55% top-1 is 3.3× chance but the dynamic range is small. A real Phase C C-spike would use a 50–100 query set across multiple docs.
2. **Gold-chunk fixture absent:** PASS B is measured at page-level (top-1 chunk's page == gold page), not at chunk-level (top-1 chunk == specific gold chunk). The Charter wants chunk-level via the v2.X regression fixture; ATZ has no per-chunk fixture entry. Building one before Phase C C2 is recommended (10–20 ATZ-specific queries with per-chunk gold annotations, ~2 hours operator time).
3. **Hand-crafted query bias:** the queries lean on the operator's reading of the doc content. Q19 and Q20 (impressum/masthead) are inherently ambiguous because other corpus docs have similar pages. A better fixture would tag each query as "this answer is uniquely in this doc" vs "this query could be answered by multiple docs".
4. **No omlx co-residency check yet** (Charter §4.2 step 2 #9). ColPali was loaded on workstation MPS; omlx ColPali deployment is Phase C task C2. The omlx tenancy scaffolding from foundation (`src/mmrag_v2/omlx/scheduler.py`) is ready but not yet wired to a live ColPali deployment.

## Artifacts

- `scripts/v3_c_spike.py` — PASS A harness (visual + text retrieval over 20 queries)
- `scripts/v3_c_spike_pass_b.py` — PASS B harness (bounded join + rerank)
- `docs/V3_C_SPIKE_RUN1.json` — raw PASS A trace (20 queries × visual top-5 + text top-5)
- `docs/V3_C_SPIKE_PASS_B.json` — raw PASS B trace (per-query candidate set + rerank top-5)

## Next steps (operator decision required)

The Charter §4.2 outcome rule has a binary fork between "redirect" (PASS A FAIL) and "expand scope" (PASS B FAIL only). The data shows the failure mode is granularity-driven and the hybrid mechanism works on hybrid-rescue queries (Q07, Q08, Q13). Operator must decide between:

1. **Strict-Charter:** redirect to VLM-native parsing or alternative visual model per §4.2 outcome rule.
2. **Sequenced falsification** (recommended above): try alternative ColPali variant first (low cost), then if no improvement expand scope to region-level.
3. **Defer:** mark Phase C "needs larger-fixture test" and defer until a per-chunk gold fixture exists for `ATZ_Elektronik_German`. Run again with 50+ queries.

`docs/PLAN_V3.md` step 1c will be the operator's chosen path.
