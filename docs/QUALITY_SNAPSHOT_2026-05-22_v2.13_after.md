# Quality Snapshot 2026-05-22 — v2.13.0 AFTER (Cycle Close)

> **Status:** v2.13.0 SHIPPED 2026-05-22. Annotated tag `v2.13.0`
> staged locally; user pushes to GitHub + Gitea.
> **Engine:** `2.13.0` (`src/mmrag_v2/version.py`, `pyproject.toml`).
> **Schema:** `2.7.0` (unchanged from v2.7).
> **Predecessor:** `v2.12.0` (2026-05-21, `5a2ce18`).

## 1. What shipped in v2.13.0

v2.13.0 closes two parallel workstreams on top of the v2.12.0
retrieval stack:

### Phase 1 — Local Embedder Swap

`Qwen3-Embedding-8B-mxfp8` via omlx-server (`10.0.10.246:8000`)
replaces cloud `text-embedding-v4` as the production embedder. Same
hybrid+rerank retrieval shape as v2.12.0; only the embedder changes.

**Apples-to-apples win — 6/6 axes:**

| Metric | omlx (NEW prod) | dashscope (OLD prod) | Δ |
|---|---:|---:|---:|
| Recall@1 chunk | 57.5% | 55.0% | **+2.5 pp** |
| Recall@5 chunk | 78.0% | 72.6% | **+5.4 pp** |
| Recall@5 doc | 95.2% | 93.1% | +2.1 pp |
| Relevance | 74.6% | 74.1% | +0.5 pp |
| Format | 92.9% | 89.2% | **+3.7 pp** |
| Faithfulness | 66.9% | 65.9% | +1.0 pp |

3 of 6 are meaningful margins (R@1, R@5 chunk, Format); 3 within noise.
Full evidence: `docs/QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_omlx_vs_dashscope.md`.

### Phase 2 — OCR Auto-Routing (Earthship + Firearms Format Recovery)

`PdfConversionPlan` gained `force_full_page_ocr: bool` resolved to `True`
for `scanned*` profiles. `BatchProcessor.set_conversion_plan` now
auto-overrides `ocr_mode="layout-aware"` → `"legacy"` when
`plan.force_full_page_ocr=True` so Docling's flag actually reaches its
OCR engine.

Format recovery on the two target docs (16-query soak subset):

| Doc | Before P2 | After P2 | Δ |
|---|---:|---:|---:|
| Earthship_Vol1 | 56.3% | 62.5% | +6.2 pp |
| Firearms | within 16-query noise floor | — | partial |

Full evidence: `docs/DECISIONS.md` "v2.13 Phase 2 OCR Auto-Routing
Outcome" and "v2.13 Phase 2 CarOK Form-Class Format Penalty —
Documented Limitation".

## 2. Production stack (v2.13.0)

```
query
  ↓
  ├─ DENSE leg:  omlx Qwen3-Embedding-8B-mxfp8 → mmrag_v2_8__qwen3_local (4096-dim, 31,371 pts)
  ├─ SPARSE leg: BM25 tokenize → mmrag_v2_8__bm25_sparse (26,381 pts)
  ↓ RRF fuse (1.0, 1.0), top-25 each leg
candidates
  ↓
ModernBERT rerank (local omlx gte-reranker-modernbert-base-mlx)
  ↓
top-5 return
```

| Component | Provider | Model / Collection |
|---|---|---|
| Embedder | omlx-server LAN | `Qwen3-Embedding-8B-mxfp8` |
| Dense collection | Qdrant local | `mmrag_v2_8__qwen3_local` (4096-dim, 31,371 pts) |
| Sparse collection | Qdrant local | `mmrag_v2_8__bm25_sparse` (26,381 pts) |
| BM25 index | repo fixture | `tests/fixtures/bm25_index_v2_12.json` |
| Reranker | omlx-server LAN | `gte-reranker-modernbert-base-mlx` |
| HyDE | optional, OFF by default | qwen-max cloud (when enabled) |
| Strict-gate corpus | v2.10 Phase 8 (unchanged) | `docs/QUALITY_SNAPSHOT_2026-05-16_v2.10_after.md` |

**Rollback baseline through 2026-06-19:** `mmrag_v2_8__qwen3_dashscope`
(1024-dim, 31,371 pts) retained unchanged. Flip the embedder default
in `src/mmrag_v2/retrieval/pipeline.py` back to
`embed_provider="dashscope"` + `embed_model="text-embedding-v4"` +
`collection="mmrag_v2_8__qwen3_dashscope"` if a corpus-specific
regression surfaces in production use. After 2026-06-19 the
dashscope collection becomes a deletion candidate if no rollback
was triggered.

## 3. Quality gates — v2.13.0 results

### Test suite
```
pytest tests/ -q
1033 passed, 16 skipped, 0 failed   (run: 2026-05-22)
```

All five v2.13-relevant guards pass:
- `tests/test_v2_10_release_baseline.py::test_engine_and_schema_version_pinned` — engine 2.13.0 ✓
- `tests/test_retrieval_pipeline.py` — 17/17 (mock-driven composition + omlx provider) ✓
- `tests/test_pdf_conversion_plan.py` — adapter invocation guards intact ✓
- `tests/test_token_validator.py` — schema 2.7.0 token contract ✓
- All other domain tests (chunking, OCR, validators, etc.) — unchanged ✓

### Retrieval regression — v2.13 fingerprint
```
python scripts/retrieval_regression_v2_13.py
v2.13 hybrid regression: 20/20 PASS   (run: 2026-05-22)
```

Fingerprint pinned at
`tests/fixtures/retrieval_regression_v2_13_hybrid.json` (engine_version
2.13.0, embed_provider omlx, embed_model Qwen3-Embedding-8B-mxfp8,
dense_collection mmrag_v2_8__qwen3_local, reranker
gte-reranker-modernbert-base-mlx).

### Strict-gate corpus
Unchanged from v2.10 Phase 8 baseline (34/34 PASS-class). v2.11 and
v2.12 changed only the retrieval side, not extraction/chunking/validation.
v2.13.0 also keeps extraction/chunking/validation contracts identical;
only the embedder model + dense collection target swap. See
`docs/QUALITY_SNAPSHOT_2026-05-16_v2.10_after.md` for the 34/34
distribution.

### Format gate
**Format ex-CarOK ≥95% pin** — CarOK_voorraadtelling is a documented
form-class judge-calibration limitation (see DECISIONS.md "v2.13 Phase 2
CarOK Form-Class Format Penalty"). Aggregate Format on v2.13 P1 omlx
fixture: **92.9% (962/1036)**; ex-CarOK: ~95% (math in the P1 omlx
report). v2.14 carry-forward: judge-side `format_form` axis variant.

## 4. Cumulative deltas

### v2.13.0 vs v2.11.0 (full retrieval-stack lift, two cycles)

The v2.11.0 → v2.12.0 → v2.13.0 trajectory tracked from the
**v2.11.0 baseline fixture** (518 queries, used in v2.11 and v2.12
soaks). v2.13's P1 fixture is a different sample (post-P2 Earthship +
Firearms re-extraction), so absolute numbers are not directly
comparable. The v2.13 P1 apples-to-apples on the new fixture is what
quantifies v2.13's contribution; cumulative numbers below summarise the
two-cycle retrieval-stack lift on the v2.11 baseline shape:

| Metric | v2.11.0 (dense only) | v2.12.0 (hybrid+rerank, dashscope) | v2.13.0 (hybrid+rerank, omlx) |
|---|---:|---:|---:|
| Recall@1 chunk | 35.5% | 67.8% | ~70% (extrapolated +2.5pp omlx delta on v2.12 fixture) |
| Recall@5 chunk | 66.8% | 90.2% STRETCH | ~92% |
| Recall@5 doc | 91.7% | 98.6% STRETCH | ~99% |
| Relevance | 59.3% | 82.1% | ~83% |
| Format | 89.8% | 88.4% | ~92% |
| Faithfulness | 50.6% | 72.6% | ~73% |

The v2.13.0 extrapolations apply the P1 fixture omlx-over-dashscope
deltas to the v2.12.0 numbers. A re-soak on the v2.11 baseline fixture
with the omlx stack would tighten these numbers, but isn't required for
the ship gate (the P1 apples-to-apples win on a same-fixture basis is
the decision evidence).

### v2.13.0 vs v2.12.0 (Phase 1 SWAP, apples-to-apples)

See section 1 above. omlx wins 6/6 axes on the same fixture.

## 5. Costs and latency

| Item | v2.12.0 | v2.13.0 |
|---|---|---|
| Per-query embed cost | ~$0.0001 (Dashscope WAN) | **$0** (omlx LAN) |
| Per-query embed latency | ~250–500 ms WAN | **~80 ms LAN** |
| Per-query rerank | ~80 ms LAN (omlx) | ~80 ms LAN (omlx, unchanged) |
| End-to-end p99 (full retrieval) | ~2.05 s | **~1.6 s** (estimated; embed swap saves ~400 ms) |
| External dependencies | Dashscope (embed) + omlx (rerank) | **omlx only** (privacy + offline-capable) |

## 6. Cycle metadata

| Field | Value |
|---|---|
| Started | 2026-05-22 (Phase 1 + Phase 2 in parallel) |
| Closed | 2026-05-22 |
| Local commit | (pending; tag staged) |
| Tag | `v2.13.0` (annotated, staged for user push) |
| Push target | GitHub `rcklomp/multimodel-document-converter`, Gitea `ronald/MM-Converter-V2` |
| Plan history | `docs/PLAN_V2.13.md` |
| Phase outcomes | `docs/DECISIONS.md` (Phase 1 + Phase 2 entries 2026-05-22) |
| Soak cost | ~$5.25 (within $25/cycle cap) |
| Soak fixture | `output/soak/v2.13_p1_omlx/work.jsonl` (518 queries) + `_dashscope_baseline/` |

## 7. Carry-forward to v2.14

| Item | Source | Notes |
|---|---|---|
| Form-class `format_form` judge axis | v2.13 P2 CarOK decision | CarOK Format penalty is judge-calibration, not content; v2.14 amends soak protocol |
| Language-aware embedder routing | v2.13 P1 per-doc breakdown | German content (ATZ_Elektronik -12.5 R@1) underperforms with omlx |
| Code-doc embedder choice | v2.13 P1 per-doc breakdown | Python_Cookbook, IRJET, Hybrid_electric regress 6-12pp R@1 |
| Local LLM integration | Future plan | User acquired Asus Ascent GX10 (DGX Spark clone); proposed Qwen3.6-35B-A3B-FP8 for experimentation accelerator (judge, HyDE, query gen). Hold as future-use note until v2.14 scoping |
| 30-day dashscope rollback window | This snapshot | Decision point 2026-06-19: drop `mmrag_v2_8__qwen3_dashscope` if no rollback fired |

## 8. References

### Layer 0 (contracts)
- `AGENTS.md` (engine version block updated to 2.13.0)
- `CLAUDE.md` (Read First updated to v2.13.0 SHIPPED)
- `docs/DECISIONS.md` (Phase 1 + Phase 2 entries 2026-05-22)
- `docs/QUALITY_GATES.md`
- `docs/ARCHITECTURE.md`

### Layer 1 (current state)
- `docs/PROJECT_STATUS.md` (updated)
- **`docs/QUALITY_SNAPSHOT_2026-05-22_v2.13_after.md`** ← this file (current canonical baseline)
- `docs/QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_omlx_vs_dashscope.md` (decision evidence)
- `docs/QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_omlx.md` (omlx per-doc + weakest)
- `docs/QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_dashscope_baseline.md` (dashscope per-doc + weakest)
- `docs/QUALITY_SNAPSHOT_2026-05-21_v2.12_after.md` (predecessor)

### Layer 2 (execution)
- `docs/PLAN_V2.13.md` (CLOSED 2026-05-22)
- `docs/PLAN_V2.12.md` (CLOSED 2026-05-21)
- `tests/fixtures/retrieval_regression_v2_13_hybrid.json` (v2.13 fingerprint)
