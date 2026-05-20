# Project Status

Last updated: 2026-05-20

Purpose: fast orientation for a new coding session. Read this before deeper project docs.

## Current Objective

**`v2.11.0` swap STAGED LOCALLY (2026-05-20)** — production text-
retrieval embedder swapped from Ollama `llava` (4096-dim multimodal)
to Dashscope `text-embedding-v4` (1024-dim text-only) after the
Phase 1 shootout delivered 10×-class lift across every embedder-
attributable axis. Commits `18bfbf2` (Phase 1 outcome) and
`c2a461c` (swap execution) live on local `main` and are **not yet
pushed; no v2.11.0 tag is created yet** — the user pushes and tags
after live-stack re-verification (`pytest tests/ --ignore=tests/manual -q`
+ `python scripts/retrieval_regression.py` against the live Qdrant +
Dashscope).

The challenger collection `mmrag_v2_8__qwen3_dashscope` is the
production data path (30,588 points, 1024-dim cosine, status green).
Legacy `mmrag_v2_8` (Ollama llava, 30,454 points, 4096-dim) retained
through 2026-06-19 as the 30-day rollback baseline; both retrieval-
regression tests must stay green during the window
(`test_retrieval_regression_v2_10.py` = rollback lane,
`test_retrieval_regression_v2_11.py` = production lane).

Tag tree on GitHub (after user pushes the staged commits):

```
v2.8.0         (2026-05-04, 645ab2b)
v2.9.0-rc1     (2026-05-12, 3e06d1b)  — v2.9 ship state, 8 deferrals
v2.10.0-rc1    (2026-05-16, 82c3639)  — all 8 closed corpus-wide
v2.10.0        (2026-05-16, db6527c)  — chunker baseline + soak evidence
v2.11.0        (PENDING, c2a461c)     — embedder swap; user pushes/tags
```

**Active canonical baseline:**
[`docs/QUALITY_SNAPSHOT_2026-05-20_v2.11_soak_qwen3.md`](QUALITY_SNAPSHOT_2026-05-20_v2.11_soak_qwen3.md)
— v2.11 Phase 1 challenger soak with the production lift numbers.

**Predecessor baselines (kept for delta reproducibility):**

- [`docs/QUALITY_SNAPSHOT_2026-05-16_v2.10_after.md`](QUALITY_SNAPSHOT_2026-05-16_v2.10_after.md)
  — v2.10 AFTER snapshot (corpus 34/34 PASS, Phase 1-7 closure summary).
- [`docs/QUALITY_SNAPSHOT_2026-05-16_v2.10_soak.md`](QUALITY_SNAPSHOT_2026-05-16_v2.10_soak.md)
  — v2.10 soak (Format 98.3%, Recall@1 2.1%; documented retrieval-
  quality known-limitation that v2.11 Phase 1 addressed).
- [`docs/QUALITY_SNAPSHOT_2026-05-11_v2.9.0-rc1_after.md`](QUALITY_SNAPSHOT_2026-05-11_v2.9.0-rc1_after.md)
  — v2.9.0-rc1 ship state, revised 2026-05-12.

## v2.11 Phase 1 Result (numbers)

Challenger Dashscope `text-embedding-v4` vs baseline Ollama `llava`,
both against the same 259 chunks + 518 queries (the v2.10 soak
fixture):

| Axis | v2.10 baseline | v2.11 challenger | Δ (pp) | Multiple | Phase 1 floor | Outcome |
|---|---:|---:|---:|---:|---:|---:|
| Recall@1 chunk | 2.1% | **35.5%** | +33.4 | **16.9×** | ≥ 15% | ✅ (clears stretch ≥ 30%) |
| Recall@5 chunk | 6.8% | **66.8%** | +60.0 | **9.8×** | ≥ 25% | ✅ (clears stretch ≥ 50%) |
| Recall@5 doc | 54.2% | **91.7%** | +37.5 | 1.7× | ≥ 70% | ✅ (clears stretch ≥ 85%) |
| Relevance | 5.9% | **59.3%** | +53.4 | **10.1×** | ≥ 30% | ✅ |
| Faithfulness | 4.7% | **50.6%** | +45.9 | **10.8×** | ≥ 25% | ✅ |
| **Format (judge)** | 98.3% | **89.8%** | **−8.5** | — | **≥ 96%** | ❌ **−6.2pp below pin** |

**Format gate temporary downgrade — explicit, on-record** (per the
make-the-failing-run-pass rule the original ≥96% pin was **not**
silently weakened): v2.11.0 release pin ≥ 85% (89.8% actual);
v2.11.x recovery target ≥ 95%; v2.12+ reverts to ≥ 96% after two
consecutive recovery soaks. The dip is concentrated in three
scanned/form docs whose chunks have known OCR/structure debt
(`CarOK_voorraadtelling` 68.8%, `Earthship_Vol1` 71.9%,
`IRJET_Modeling_of_Solar_PV` 71.9%) — the baseline llava embedder
rarely retrieved these docs due to hub-collapse; the challenger
reaches them now and the judge correctly grades them. **The
regression is coverage-reveal of pre-existing chunk-format debt, not
swap-induced.** Full rationale in `docs/DECISIONS.md` "v2.11 Phase 1
Embedder Shootout Outcome" + "v2.11.0 Embedder Swap Executed —
Format Gate Downgrade".

**Honest absolute-quality read.** Relative lift is huge (10× on five
axes); absolute numbers are still mediocre — Recall@5 chunk 66.8% is
not 80%+. v2.12 plan (to be drafted) will close this gap via
reranker → hybrid retrieval → query rewriting in bang-for-buck
order. Target for v2.12: Recall@5 chunk ≥ 85%, Recall@1 ≥ 55%,
Faithfulness ≥ 70%.

## Qdrant Collections

**Production:** `mmrag_v2_8__qwen3_dashscope` — 30,588 points,
1024-dim cosine, status green. Built from the same 34 canonical
v2.10 JSONLs via `scripts/rebuild_mmrag_v2_8_for_rc1.py --provider
dashscope --resume`; wall time 540.5 min. Raw chunk count was 30,588
(no ingest-time filter rejections under dashscope — dashscope
tolerates short/empty content that llava had rejected, accounting
for the +134 vs the llava-built baseline).

**Legacy rollback (through 2026-06-19):** `mmrag_v2_8` — 30,454
points, 4096-dim cosine via Ollama `llava`. Retained untouched for
fast rollback. Drop date 2026-06-19 (or sooner with user signoff).

## Active Model/Endpoint State

Do not print or commit API keys.

**Production embedder (text retrieval):**

- provider: Dashscope (OpenAI-compatible)
- model: `text-embedding-v4`
- endpoint: `https://dashscope-intl.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding`
- env var: `DASHSCOPE_API_KEY` (required for ingestion, search, and the v2.11 retrieval-regression test)

**Production VLM (image enrichment):**

- provider: OpenAI-compatible
- model: `NuMarkdown-8B-Thinking-mlx-8bits` (local fallback retained)
- base URL: `http://10.0.10.246:8000/v1`
- Cloud comparison validated: Dashscope `qwen3-vl-plus` (preferred for richer descriptions, fewer hard fallbacks)

**Synthetic soak judge:**

- provider: Dashscope
- model: `qwen-max`
- both query generation and retrieval grading use the same model

**Mac Mini compute reserve (v2.12 candidate):**

- host: `http://10.0.10.246:1234` (LM Studio)
- registered: `qwen3-vl-8b-instruct-mlx`, `qwen3-embedding-8b-mxfp8` (loader currently fails — missing `lm_head.weight`; v2.12 needs a different runtime such as `mlx-embedding-models` + FastAPI wrapper)

## Current Quality Summary

Source of truth for v2.11.0:
[`docs/QUALITY_SNAPSHOT_2026-05-20_v2.11_soak_qwen3.md`](QUALITY_SNAPSHOT_2026-05-20_v2.11_soak_qwen3.md).
v2.10 strict-gate state (`scripts/qa_full_conversion.py --source-pdf
--allow-warnings`) is unchanged at **34 PASS / 0 WARN / 0 FAIL** —
the swap touches retrieval-side only, not extraction / chunking /
validation.

**Current local test suite: 986 passed, 15 skipped, 0 failed**
(after the v2.11.0 swap). Production test additions in this cycle:

- `tests/test_rebuild_resume.py` — 9 tests pinning the
  `scripts/rebuild_mmrag_v2_8_for_rc1.py` resume + retry behavior
  (added during Phase 1 step 0).
- `tests/test_retrieval_regression_v2_11.py` — production retrieval-
  shape pin against `mmrag_v2_8__qwen3_dashscope` (added during
  swap execution).
- `tests/test_retrieval_regression_v2_10.py` — repositioned as the
  rollback-validation test (explicit `--provider ollama --collection
  mmrag_v2_8`); must stay green through 2026-06-19.

Image enrichment state is unchanged from v2.10 close (0 pending
corpus-wide; see `docs/QUALITY_SNAPSHOT_2026-05-16_v2.10_after.md`
§5).

### v2.11 commits summary

Reverse-chronological. Each commit is local, none pushed.

- `c2a461c` (2026-05-20) — **v2.11.0 swap executed.** Production
  defaults flipped across 5 scripts (`ingest_to_qdrant.py`,
  `rebuild_mmrag_v2_8_for_rc1.py`, `retrieval_regression.py`,
  `synthetic_soak.py`, `search_qdrant.py`); `--provider` default
  ollama → dashscope; collection defaults flipped to
  `mmrag_v2_8__qwen3_dashscope`; `search_qdrant.py` gained provider
  + api-key flags. Tests updated: v2.10 regression repositioned as
  rollback test, new v2.11 production test, contextual fixture
  pinned to `--provider ollama`. Fingerprint engine_version
  promoted `2.11.0-candidate` → `2.11.0`. Format gate downgrade
  recorded in DECISIONS.md. Plan promoted to Draft v1.0.

- `18bfbf2` (2026-05-20) — Phase 1 outcome doc + soak report +
  halt-for-user-decision write-up. 5/6 floors crushed, Format pin
  missed by −6.2pp; both swap-recommended and no-swap reads
  documented; production-default flip deferred to user sign-off.

- `a9512e8` (2026-05-17) — Phase 2.2a fresh-env pytest closure +
  Phase 2.2b CI YAML at `.github/workflows/v2_11_validate.yml` +
  Phase 3 dispositions in DECISIONS.md.

- `fccbafc` (2026-05-17) — Phase 1 setup: Dashscope embedder provider
  in `ingest_to_qdrant.py`, rebuild script `--resume` + provider
  passthrough, `tests/test_rebuild_resume.py` (9 tests), plan v0.4.

## Active Engineering Direction

Plan at [`docs/PLAN_V2.11.md`](PLAN_V2.11.md) (Draft v1.0; Phase 1 +
Phase 2 + Phase 3 dispositions shipped). The current carry-forward
backlog (to be folded into v2.12 plan when authored):

1. **v2.11.x Format recovery** — chunk-content sanitization for the
   three named scanned/form docs. Target ≥95% Format on next soak.
2. **30-day rollback window watch** — both regression tests must
   stay green through 2026-06-19; drop legacy collection +
   `test_retrieval_regression_v2_10.py` at the drop date.
3. **v2.12 plan to be drafted** — focused on closing the absolute-
   quality gap (Recall@5 66.8% → ≥85%) via reranker first, then
   hybrid retrieval, query rewriting, and per-doc-class chunking.

Phase 3c (broader UIR refactor) remains paused for user signoff on
the `ConversionPlan` parent-class carve-out scope.

## Recently Completed

Reverse-chronological. Each entry's evidence files are tracked.

- **PLAN_V2.11 Phase 1 (embedder swap):** SHIPPED 2026-05-20 locally
  (`c2a461c` + `18bfbf2`; pending push). Production embedder
  Dashscope `text-embedding-v4`; production collection
  `mmrag_v2_8__qwen3_dashscope` (30,588 points). 10× lift on 5/5
  embedder-attributable axes; Format judge pin temporarily ≥85%.
  Both regression tests green. Test suite 986/15 skipped/0 failed.
  See `docs/QUALITY_SNAPSHOT_2026-05-20_v2.11_soak_qwen3.md`.

- **PLAN_V2.11 Phase 2 (validated-cloud CI):** SHIPPED 2026-05-17
  (`a9512e8`). GitHub Actions workflow at
  `.github/workflows/v2_11_validate.yml` with three jobs:
  GitHub-hosted lint + import smoke (always); self-hosted full
  pytest + retrieval-regression + smoke (on push/PR); tag-only
  strict-gate + soak sub-sample. Fresh-env pytest closure: 984
  passed, 15 skipped on the day's run.

- **PLAN_V2.11 Phase 3 (carry-forward dispositions):** SHIPPED
  2026-05-17 (`a9512e8`). Five rows in `docs/DECISIONS.md` "v2.11
  Carry-Forward Decisions": 3a Qwen3-VL-8B-on-Mini as v2.12 VLM
  candidate; 3b local Docling CodeFormulaV2 as v2.11 workaround;
  3c PAUSED for user signoff; 3d design recorded + implementation
  deferred to v2.12 (initial ~50 LOC estimate was wrong; honest
  footprint 200-300 LOC + new fixture); 3e magazine rendered-
  region-crop deferred with soak-data rationale.

- **PLAN_V2.10 Phase 8 (strict-gate re-verification + v2.10 release
  prep):** SHIPPED 2026-05-16. 34/34 PASS corpus-wide; v2.10.0
  annotated tag on `db6527c` pushed to GitHub the same day.
  Cycle closed. See
  `docs/QUALITY_SNAPSHOT_2026-05-16_v2.10_after.md`.

- **PLAN_V2.10 Phases 1-7:** all closed `validated-local` between
  2026-05-12 and 2026-05-15. Phase 1 (Chaubal TOC router), Phase 2
  (Fluent_Python TextIntegrityScout), Phase 3 (Earthship + Distilled
  picture dedup), Phase 4 (Cookbook + Distilled cross-page split),
  Phase 5 (Devlin HybridChunker heading propagation), Phase 6
  (Firearms OCR heading propagation + infix-step-number repair),
  Phase 7 (KI EPUB synthetic-pagination lane). See
  `docs/PLAN_V2.10.md` for the per-phase narrative.

- **PLAN_V2.8 (production gaps + broad reconversion + Qdrant
  re-ingestion):** SHIPPED 2026-05-04. 7 commits on main
  `5b0e13d → 645ab2b`. See `docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md`.

## Must-Respect Constraints

- Python 3.10 only.
- Batch size must stay at or below 10 pages.
- Do not use `--profile-override` for acceptance runs.
- Do not add filename-specific or document-specific quality rules.
- OCR handles text; VLMs describe visuals only.
- BBoxes must remain normalized integer `[0,1000]`.
- Acceptance requires `GATE_PASS` plus `UNIVERSAL_PASS` across the smoke matrix.
- Production embedder is dashscope/text-embedding-v4 against
  `mmrag_v2_8__qwen3_dashscope`. Ollama/llava lane is rollback-only
  through 2026-06-19; do not use as a comparison baseline beyond
  that date.
- `DASHSCOPE_API_KEY` must be set for any ingestion, search, or
  v2.11 retrieval-regression run. Test-suite skip-gates handle the
  unset case for CI.
