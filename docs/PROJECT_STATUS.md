# Project Status

Last updated: 2026-05-25

Purpose: fast orientation for a new coding session. Read this before deeper project docs.

## Current Objective

**v2.16.0 CONVERGENCE RELEASE — FEATURE-COMPLETE FOR v2.X.** Engine
2.15.0 → 2.16.0; v2.16.0 annotated tag PENDING user push to origin
+ GitHub at the close-out commit. Post-tag: only bug-fix patches
(v2.16.x) accepted; new features = re-charter as v3.0.

**Phases shipped (v2.16.0 scope):**
- **Phase 0** SHIPPED — corpus expansion (7 new PDFs from data/raw/
  ingested + classified + appended to Qdrant + BM25 rebuilt;
  CANONICAL_34 → CANONICAL_DOCS rename across 5 sites; anti-drift
  bridge test). 34 → 38 canonical docs (4 of 7 new PDFs PASSed strict gate; 3 honestly dropped per DECISIONS "v2.16 Phase 0 Strict-Gate Honest Reduction").
- **Phase 1** SHIPPED — personal_importance overlay (HIGH forces
  Option A; MED uses telemetry; LOW reduces grace). validation-
  query runner + fixtures for 2 HIGH classes (10 queries each).
  Baseline captured at v2.15.0 stack: 0/10 + 0/10 on the
  documented failure modes (Phase 3/4 targets).
- **Phase 3** SHIPPED — partial_code adjacency fetch mechanism in
  `retrieve_hybrid_reranked`. 8 bridge tests; v2.14 fingerprint
  20/20 PASS unchanged. **INERT on current corpus** (HybridChunker
  path doesn't set `partial_code=True` for academic_whitepaper /
  technical_manual profiles). Item #9 (Docling config hunt) reopens
  for v2.17 per §7 trigger #1 to extend partial_code coverage.
- **Phase 4** SHIPPED — VLM-table IoU dedup (bbox_iou utility +
  `dedup_vlm_table_iou_threshold` knob defaulting 0.85 +
  `_apply_vlm_table_iou_dedup` pre-final-boundary-repair pass).
  10 dedup + plan-knob bridge tests.
- **Phase N** — engine bump 2.15.0 → 2.16.0; AFTER snapshot at
  `docs/QUALITY_SNAPSHOT_2026-05-25_v2.16_after.md`; README
  feature-complete banner; DECISIONS.md 10 new entries.

**KILLed (with DECISIONS entries):**
- **Phase 2 / Phase 6** — omlx deficit diagnosed as multi-factor
  (5 deficit docs all -12.4 to -12.6pp uniform across heterogeneous
  classes contradicts a single-cause hypothesis); apples-to-apples
  re-test structurally blocked by dashscope-collection drop (v2.14
  P3). Compound trigger fails → Phase 6 KILL. 2nd dead lever (HyDE
  was the 1st). Item 4 (c) — full -12pp gap documented as accepted
  embedder limit.
- **Phase 5** — Dynamic top-k pre-flight KILL by gate leg (b)
  (PASS-retention undefined; v2.15.0 baseline static=0).
- **Phase 7** — Image re-read KILL by default (no user opt-in
  promotion of image-heavy validation fixture).

**Carry-Forward closures (8 KILL items in DECISIONS):** #9 (B1
Docling config hunt, conditional → reopens for v2.17 per §7), #10
(A2 HTML+summary split), #12 (B2 Code-Rescue heuristic), #13 (UIR
refactor), #14 (VLM swap), #15 (magazine crop), #21 (remote
CodeFormulaV2), #22 (HybridChunker per-item token guard).

**v3.0 OUT-OF-SCOPE:** #11 (ColPali / VisRAG visual retrieval).

Predecessor: **v2.15.0 SHIPPED + PUSHED 2026-05-24** under Option
F. Annotated tag `v2.15.0` on origin + GitHub at commit `fff67d9`.

**v2.15 phases shipped (Option F scope):**
- **Phase 3 [F]** SHIPPED — full telemetry suite (5 modules + 2 docs + soak hook + 29 tests; DECISIONS.md "v2.15 Documented-Limitation Telemetry Threshold" transitioned PRE-CYCLE PROPOSAL → ACTIVE RULE)
- **Phase 6 [U]** SHIPPED — calibration freshness check: FP8-14B cal fresh through 2026-06-22, T-72h pre-tag checkpoint armed
- **Phase 1 [U]** CLOSED as DEAD LEVER 2026-05-24 PM (post-v2.15.0 tag) — narrow A/B soak (n=224 across 5 docs) ran post-tag once `MLX_API_KEY` env was available; falsification rule fired (4/5 docs ZERO R@1 delta; +0.4pp aggregate within noise; German subgroup +0.0 on n=64). HyDE bridging closed per the v0.9 plan's explicit termination condition. Infra stays in tree as opt-in (production defaults unchanged). Report: `docs/archive/soaks/SOAK_2026-05-24_v2.15_p1_narrow_hyde_AB.md`. DECISIONS.md entry: "v2.15 Phase 1 HyDE Bridging — CLOSED as Dead Lever".
- **Phase N** — engine bump + AFTER snapshot + version-pin test update + **v2.15.0 tag PUSHED to origin + GitHub** (commit `fff67d9`)

**v2.16 cycle plan: Draft v0.8 in mid-audit (Round 7 pending).**
Convergence-cycle plan at `docs/PLAN_V2.16.md` (Round 1-6
dispositions in Appendix A). Round 6 (2026-05-25) accepted 4 HIGH
iteration-fallout findings from v0.7 propagation gaps + 1 MED +
2 LOW. None required disposition changes; all were
wording/cross-reference fixes. Structural-finding rate trajectory:
R1 0 → R2 1 → R3 5 → R4 1 + 1 disposition → R5 2 iter-induced
→ R6 4 iter-fallout (all from prior-round fixes not propagating).
Plan IS converging — every recent HIGH is "v0.X edit didn't
propagate to all sites," not "design is wrong." Cycle NOT yet
open for execution; opens when audit clears the stopping rule
(two consecutive 0-HIGH rounds).

**v2.15 phases skipped (Option F deferrals):**
- **Phase 2 [A]** pdfplumber lane — v2.16 contingent on Phase 3 telemetry evidence
- **Phase 4 [A]** Docling HybridChunker tuning — carry-fwd 6.1 trigger active (Docling ≥2.87 OR 90d)
- **Phase 5 [E]** retrieval-side investments — v2.16 contingent on F→E telemetry escalation

**Active canonical baseline:** [`docs/QUALITY_SNAPSHOT_2026-05-24_v2.15_after.md`](QUALITY_SNAPSHOT_2026-05-24_v2.15_after.md).
**Cycle history:** [`docs/archive/plans/PLAN_V2.15.md`](archive/plans/PLAN_V2.15.md) (CLOSED 2026-05-24; Draft v0.9 + 8-round audit archaeology in Appendix A).
**Cycle-open process:** [`docs/CYCLE_OPEN_CHECKLIST.md`](CYCLE_OPEN_CHECKLIST.md) (NEW in v2.15; load-bearing for v2.16+ telemetry analyzer run, Docling watcher, calibration freshness, cycle_slip.log).

---

**v2.14.0 CLOSED + PUSHED 2026-05-23** (engine 2.13.0→2.14.0;
annotated tag `v2.14.0` on origin at commit `36482e0`, sha
`122a62e`). v2.14 layered local-LLM accelerator infrastructure on
top of the v2.13.0 retrieval stack — **NO retrieval-stack changes**
vs v2.13.0 (omlx Qwen3-Embedding-8B-mxfp8 + BM25 + RRF +
ModernBERT rerank unchanged).

**v2.14.x patch range (post-tag, 2026-05-23 PM):**
- Phase 2 (intent classifier + targeted HyDE) — commit `156dfa7`. Infra opt-in; broad-query mini-soak FALSIFIED. v2.15 Phase 1 re-targets.
- Phase 3 (rollback drop) — commit `2527414`. User "full send" override of 2026-06-19 time gate; ~30 GB reclaimed; snapshots persisted.
- v2.14.1 GX10 endpoint swap — commit `53ffc73`. Retired Qwen3.6-27B-FP8 (format collapsed to 70.7%). Deployed `RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic` (Blackwell-native FP8). Phase 0 re-cal: rel 82.2% / **format 90.7% TRUSTWORTHY** / faith 76.6%.
- n-gram spec decoding REJECTED post-swap (6.3% acceptance; bare FP8-14B is production).

**v2.15 cycle CLOSED 2026-05-24 under Option F** (see headline
section above). Plan history: [`docs/archive/plans/PLAN_V2.15.md`](archive/plans/PLAN_V2.15.md)
(Draft v0.9 + 8-round audit archaeology in Appendix A).

**Predecessor v2.14.0 canonical baseline:** [`docs/archive/snapshots/QUALITY_SNAPSHOT_2026-05-23_v2.14_after.md`](archive/snapshots/QUALITY_SNAPSHOT_2026-05-23_v2.14_after.md) (ship state + §8 post-ship addendum 2026-05-24). The active canonical baseline is the v2.15 AFTER snapshot named in the headline section.
**Cycle history:** [`docs/archive/plans/PLAN_V2.14.md`](archive/plans/PLAN_V2.14.md) (CLOSED 2026-05-23; close-out header + Draft v0.5 archaeology).

**v2.14 final phase outcomes** (mirrors `QUALITY_SNAPSHOT_2026-05-23_v2.14_after.md` §1 + §8 addendum):

**Shipped (8 — 6 at tag, +2 in patch range):**
- **Phase 0 (judge calibration)** — operative verdict is the FP8-14B
  re-cal from 2026-05-23 PM (post v2.14.1 swap): rel 82.2% / **format
  90.7% TRUSTWORTHY** / faith 76.6%. Report: `docs/archive/calibrations/CALIBRATION_2026-05-23_v2.14_p0_local_judge_14b_fp8.md`. The 27B-MTP ship-state verdict (all axes RESTRICTED, format collapsed to 70.7%) is retained at `docs/archive/calibrations/CALIBRATION_2026-05-23_v2.14_p0_local_judge_qwen36_27b_mtp.md` as historical. Reclaiming format-axis TRUSTWORTHY re-opens the v2.14 Phase 4b carve-out for v2.15 sub-phases.
- **Phase 4a (local HyDE provider)** — `provider="vllm"` knob (5 tests); `chat_template_kwargs.enable_thinking=False` Qwen3 fix (commit `0c5e818`, 2 bridge tests).
- **Phase 4c (gen-provider vllm)** — `synthetic_soak.py --gen-provider vllm` (commit `1c201dd`).
- **Phase 4d (tie-breaker harness)** — `scripts/local_then_cloud_soak.py` two-tier judging (14 unit tests).
- **Phase 4-Resilience (qwen3-max cloud fallback)** — `generate_with_fallback` chains vllm → qwen3-max → literal (commit `1c201dd`, 3 bridge tests).
- **Phase 5 (disk precheck)** — `_check_disk_headroom()` aborts soak stages below 10 GB free.
- **Phase 2 (intent classifier + targeted HyDE)** — commit `156dfa7`, post-tag. Opt-in infra retained; broad-query lift FALSIFIED. v2.15 Phase 1 re-targets at narrower 5-doc mini-soak.
- **Phase 3 (rollback collection drop)** — commit `2527414`, post-tag under user "full send" override of 2026-06-19 time gate. ~30 GB reclaimed; cold-storage snapshots persisted.

**Partial (2 — carried to v2.15):**
- **Phase 1 (form/table extraction)** — code-side semantic-bug fix preserved (commit `e60a253`): `--force-table-vlm` now truly forces (was silently overridden by `technical_manual` profile's `vlm_table_enabled=False`); local NuMarkdown-8B VLM produces clean 5-col markdown tables on 5/12 CarOK pages, $0. **Mini-soak failed Phase 1 acceptance bar**: 30-query CarOK soak measured Format 45.0% (was 71.9% baseline; -26.9pp regression) because VLM tables coexist with flat-prose duplicates that win retrieval 29/30 times. **Production rolled back** to v2.13 baseline. **v2.15 Option A Phase 2** addresses via same-page prose-VLM dedup (if A chosen); v2.15 Option F treats CarOK as documented-limitation with telemetry.
- **Phase 6 (code-block chunking hygiene)** — block-extension + `partial_code` schema field shipped on `_chunk_text_with_overlap` (scanned_book) path (commit `d737147`; 9/9 tests). Fluent_Python truncated-code defect is upstream of chunking (Docling-extraction-layer prose+code intermixing at page boundaries); **v2.15 Option A Phase 4** addresses via Docling-config tuning (Approach 2 only — regex/heuristic fallback explicitly rejected per Gemini audit).

**Dropped (1):**
- **Phase 4e (1500-query Format-only demo)** — predicate failed on 27B-MTP; moot under FP8-14B Format TRUSTWORTHY (Format judging is now a no-cost local capability rather than a demonstration goal).

---

**v2.13.0 SHIPPED + PUSHED 2026-05-22.** Annotated tag `v2.13.0`
on origin + GitHub at commit `021ef05`. Two parallel workstreams
closed this cycle on top of the v2.12.0 retrieval stack:

- **Phase 1 (local embedder swap) SHIPPED 2026-05-22** —
  `Qwen3-Embedding-8B-mxfp8` via omlx-server replaces cloud
  `text-embedding-v4` as the production embedder. Apples-to-apples
  shootout (same fixture, only embedder differs) won 6/6 axes
  (R@1 +2.5pp, R@5 chunk +5.4pp, R@5 doc +2.1pp, Relevance +0.5pp,
  Format +3.7pp, Faithfulness +1.0pp). Production dense collection
  flipped to `mmrag_v2_8__qwen3_local` (4096-dim, 31,371 pts).
  Dashscope collection was retained as 30-day rollback baseline
  through 2026-06-19, then **dropped 2026-05-23 PM** under v2.14
  Phase 3 user "full send" override (cold-storage snapshot retained
  through 2026-08-21).

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

Plan history: [`docs/archive/plans/PLAN_V2.13.md`](archive/plans/PLAN_V2.13.md) (CLOSED 2026-05-22).
Canonical baseline: [`docs/archive/snapshots/QUALITY_SNAPSHOT_2026-05-22_v2.13_after.md`](archive/snapshots/QUALITY_SNAPSHOT_2026-05-22_v2.13_after.md).
P1 evidence: [`docs/archive/snapshots/QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_omlx_vs_dashscope.md`](archive/snapshots/QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_omlx_vs_dashscope.md).

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
v2.13.0      (2026-05-22, 021ef05)  — local embedder swap (omlx Qwen3-Embedding-8B) + OCR auto-routing
v2.14.0      (2026-05-23, 122a62e)  — local-LLM accelerator stack (HyDE/gen/tie-breaker on local vLLM + Qwen3 thinking-mode fix + disk precheck)
v2.14.x      (post-tag, untagged) — Phase 2 (`156dfa7`, intent classifier) + Phase 3 (`2527414`, rollback drop) + v2.14.1 (`53ffc73`, GX10 → FP8-14B)
```

**Active canonical baseline:**
[`docs/archive/snapshots/QUALITY_SNAPSHOT_2026-05-22_v2.13_after.md`](archive/snapshots/QUALITY_SNAPSHOT_2026-05-22_v2.13_after.md)
— v2.13.0 AFTER snapshot. Full numbers + per-phase contributions.

Predecessor canonical (kept for delta reproducibility):
[`docs/archive/snapshots/QUALITY_SNAPSHOT_2026-05-21_v2.12_after.md`](archive/snapshots/QUALITY_SNAPSHOT_2026-05-21_v2.12_after.md)
— v2.12.0 AFTER snapshot.

**v2.13 phase reports:**

- [`docs/archive/snapshots/QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_omlx_vs_dashscope.md`](archive/snapshots/QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_omlx_vs_dashscope.md) — **Phase 1 SWAP evidence** (apples-to-apples 6/6-axis win)
- [`docs/archive/snapshots/QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_omlx.md`](archive/snapshots/QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_omlx.md) — Phase 1 omlx per-doc + weakest queries
- [`docs/archive/snapshots/QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_dashscope_baseline.md`](archive/snapshots/QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_dashscope_baseline.md) — Phase 1 dashscope per-doc + weakest queries

**v2.12 phase reports (predecessor — kept for reference):**

- [`docs/archive/snapshots/QUALITY_SNAPSHOT_2026-05-21_v2.12_p1_cloud.md`](archive/snapshots/QUALITY_SNAPSHOT_2026-05-21_v2.12_p1_cloud.md) — Phase 1 cloud-rerank shootout
- [`docs/archive/snapshots/QUALITY_SNAPSHOT_2026-05-21_v2.12_p1_omlx.md`](archive/snapshots/QUALITY_SNAPSHOT_2026-05-21_v2.12_p1_omlx.md) — Phase 1 local-rerank shootout (winner)
- [`docs/archive/snapshots/QUALITY_SNAPSHOT_2026-05-21_v2.12_p2_hybrid.md`](archive/snapshots/QUALITY_SNAPSHOT_2026-05-21_v2.12_p2_hybrid.md) — Phase 2 hybrid+rerank
- [`docs/archive/snapshots/QUALITY_SNAPSHOT_2026-05-21_v2.12_p3_hyde.md`](archive/snapshots/QUALITY_SNAPSHOT_2026-05-21_v2.12_p3_hyde.md) — Phase 3 HyDE measurement (deltas in noise; opt-in only)

**Predecessor baselines (kept for delta reproducibility):**

- [`docs/archive/snapshots/QUALITY_SNAPSHOT_2026-05-20_v2.11_soak_qwen3.md`](archive/snapshots/QUALITY_SNAPSHOT_2026-05-20_v2.11_soak_qwen3.md)
  — v2.11.0 baseline. The 518-query × 259-chunk fixture every v2.12 soak ran against.
- [`docs/archive/snapshots/QUALITY_SNAPSHOT_2026-05-16_v2.10_after.md`](archive/snapshots/QUALITY_SNAPSHOT_2026-05-16_v2.10_after.md)
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
| `mmrag_v2_8__qwen3_local` | 31,371 (4096-dim cosine, Qwen3-Embedding-8B) | green | **Production dense (v2.13.0+)** |
| `mmrag_v2_8__bm25_sparse` | 26,396 (BM25 sparse) | green | **Production sparse** (v2.12+; last rebuilt 2026-05-23 in Phase 1 rollback) |
| `mmrag_v2_8__qwen3_dashscope_smoke` | small | green | legacy smoke fixture (out of scope) |

**Dropped 2026-05-23 (Phase 3, user "full send" override of 2026-06-19 time gate):**
- `mmrag_v2_8__qwen3_dashscope` (v2.13 dashscope rollback baseline, 31,371 pts, 219 MB) — snapshot retained
- `mmrag_v2_8` (v2.10 legacy llava rollback, 30,454 pts, 583 MB) — snapshot retained

**Cold-storage snapshots (90-day retention; persist past collection deletion):**

| Snapshot file | Size | Restore via |
|---|---|---|
| `mmrag_v2_8__qwen3_dashscope-4278644141892673-2026-05-23-17-40-32.snapshot` | 219 MB | Qdrant `POST /collections/{name}/snapshots/upload` or `recover` API |
| `mmrag_v2_8-4278644141892673-2026-05-23-17-40-34.snapshot` | 583 MB | same |

Stored at `/qdrant/snapshots/<collection>/` inside the Qdrant container, persisted on Docker volume `multimodal-doc-converter_qdrant_snapshots`. Delete after 2026-08-21 unless a rollback need surfaces.

## Active Model/Endpoint State

Do not print or commit API keys.

**Production text-retrieval embedder (v2.13.0; unchanged through v2.14):**

- provider: omlx-server (OpenAI-compatible local API)
- model: `Qwen3-Embedding-8B-mxfp8` (4096-dim, MLX FP8-quantized)
- endpoint: `http://10.0.10.246:8000/v1/embeddings`
- env var: `MLX_API_KEY` (required for retrieval through omlx)
- runtime: Apple Silicon (Mac Mini), ~80 ms LAN per query
- rollback: **dropped 2026-05-23 PM** via v2.14 Phase 3 (commit
  `2527414`, user "full send" override of original 2026-06-19 time
  gate). The dashscope rollback collection
  `mmrag_v2_8__qwen3_dashscope` no longer exists in Qdrant; cold-
  storage snapshot (219 MB) retained on
  `multimodal-doc-converter_qdrant_snapshots` through 2026-08-21
  for recovery if needed. If a regression surfaces past that date,
  re-ingest from source via the dashscope provider — no hot
  fallback collection.

**Production cross-encoder reranker (NEW in v2.12):**

- provider: omlx-server (OpenAI-compatible local API)
- model: `gte-reranker-modernbert-base-mlx` (~150M params, MLX-quantized)
- endpoint: `http://10.0.10.246:8000/v1/rerank`
- env var: `MLX_API_KEY` (required for retrieval through `retrieve_hybrid_reranked()`)
- runtime: Apple Silicon, ~15 ms per (query, doc) pair, ~0.55 s p99 for K=25

**Synthetic soak judge:**

- provider: Dashscope, model `qwen-max` (used for both query generation and judging)
- v2.14 Phase 0/4 additions: local-LLM judge candidate at the GX10 vLLM endpoint below; ship-gate go/no-go on Relevance + Faithfulness axes stays on cloud `qwen-max` per the Phase 4 "leniency trap" guardrail in `PLAN_V2.14.md`

**Local LLM endpoint (v2.14.1 ACTIVE, LIVE 2026-05-23 PM):**

- provider: vLLM (OpenAI-compatible)
- model: `RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic` (Neural Magic
  compressed-tensors FP8-dynamic; Blackwell tensor-core native)
- served-model aliases: `RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic` (full) or `qwen2.5-14b-fp8` (short)
- endpoint: `http://10.0.10.239:8000/v1/chat/completions` (Asus Ascent GX10 = DGX Spark clone, Grace-Blackwell GB10, 128 GB unified memory)
- container: `vllm/vllm-openai:v0.20.0-aarch64-cu130-ubuntu2404` (entrypoint = `[vllm serve]`; model as positional first arg, NO `serve` after image tag)
- max context: 32K configured
- code default: `src/mmrag_v2/retrieval/hyde.py` `VLLM_DEFAULT_MODEL` (call with `provider="vllm"`)
- env var: `VLLM_API_KEY` (optional; endpoint runs unauthenticated by default)
- canonical recipe: `memory/project_v2_14_gx10_14b_fp8_swap.md`
- guardrails before any future swap: `memory/feedback_gx10_deployment_guardrails.md` (5-point hard checklist) + `memory/feedback_no_gx10_model_swap_reflex.md` (offline-eval-first before any live swap)
- throughput characterization (bare config; n-gram spec REJECTED — see `memory/project_v2_14_ngram_spec_rejected.md`):
  - steady-state judge call: ≈2.0s / ≈15 tok/s
  - HyDE generation (≈600 tok): ≈41s / ≈14.6 tok/s
  - no spec-decoding path available (no same-vocab Qwen2.5 draft <7B; vocab=152064 only from 7B class up)
  - `--max-num-batched-tokens 8192` recommended in production recipe for chunked-prefill
- **Phase 0 FP8-14B calibration SHIPPED 2026-05-23 PM** (n=518, 0 parse
  failures). Per-axis verdict on the same 518-query fixture as the
  retired 14B-BF16 and 27B-MTP predecessors:
  - relevance: **82.2%** (RESTRICTED; +0.2pp vs 27B-MTP, +0.5pp vs 14B-BF16)
  - format: **90.7% ✓ TRUSTWORTHY** (+20.0pp vs 27B-MTP, +0.5pp vs 14B-BF16)
  - faithfulness: **76.6%** (RESTRICTED; -2.2pp vs 27B-MTP, +0.5pp vs 14B-BF16)
  FP8 quantization preserves the 14B's BF16 calibration profile and
  reclaims the format-axis TRUSTWORTHY verdict that the 27B-MTP had
  lost. Report:
  `docs/archive/calibrations/CALIBRATION_2026-05-23_v2.14_p0_local_judge_14b_fp8.md`.
  Phase 4b format-axis local judging carve-out is back on the table
  for v2.15 sub-phases (Phase 5a top-k tuning, Phase 5c paraphrase
  fusion if Option E chosen).
- **Predecessor endpoint history (all retired 2026-05-23):**
  - 27B-MTP (AM): rel 82.0 / format 70.7 / faith 78.8 — all RESTRICTED. Format collapse motivated this PM swap.
  - 14B-BF16 (2026-05-22): rel 81.7 / format 90.2 TRUSTWORTHY / faith 76.1 — close calibration profile to the new FP8-14B but Blackwell-suboptimal precision.

**Production VLM (image enrichment — unchanged from v2.11):**

- preferred cloud: Dashscope `qwen3-vl-plus`
- local fallback: `NuMarkdown-8B-Thinking-mlx-8bits` on `http://10.0.10.246:8000/v1`

**v2.16 cycle candidates (per PLAN_V2.16.md Draft v0.5, mid-audit):**

v2.15 closed under Option F; v2.16 is the convergence cycle that
disposes every open carry-forward to SHIP / KILL / OUT-OF-SCOPE
(v3.0). Disposition matrix (PLAN_V2.16.md §2) — current state:

- SHIP: Phase 0 corpus expansion, Phase 1 decision-mechanism
  overlay, Phase 2 omlx deficit diagnostic, Phase 3 partial_code
  adjacency, Phase 4 VLM-Table dedup, Phase 5 dynamic top-k
- CONDITIONAL: Phase 6 C1 query rewriting (gated on Phase 2);
  Phase 7 image re-read (OPT-IN, default KILL — gated on §8a Q3)
- KILL: 9 items (UIR refactor 3c; VLM swap 3a; magazine rendered-
  region-crop 3e; B2 code-rescue middleware; B1 Docling config
  hunt [conditional on Phase 3]; 3b remote CodeFormulaV2; 3d
  HybridChunker per-item guard; A2 HTML+summary split; v2.14 P1
  CarOK dedup absorbed into Phase 4)
- OUT-OF-SCOPE (v3.0): D1 ColPali visual retrieval
- KEEP active: telemetry collection, Phase 4-Resilience qwen3-max
  cloud fallback, Phase 6 calibration freshness check
- CLOSED already: HyDE bridging (dead lever), Phase 3 rollback
  collection drop

## Current Quality Summary

Source of truth for v2.15.0:
[`docs/QUALITY_SNAPSHOT_2026-05-24_v2.15_after.md`](QUALITY_SNAPSHOT_2026-05-24_v2.15_after.md).
Strict-gate corpus state unchanged from v2.10: **34 PASS / 0 WARN /
0 FAIL** — extraction/chunking/validation are untouched by v2.11 →
v2.15 (all five cycles changed only the retrieval side / OCR
routing / local-LLM accelerators / telemetry observability).

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

- `a77758b` (2026-05-22) — Phase N close-out: version bump
  2.12.0→2.13.0, AFTER snapshot, layer-0/1 docs sweep, v2.13.0
  annotated tag PUSHED to origin at commit `021ef05`.
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

**v2.15 CLOSED + PUSHED 2026-05-24 under Option F**; tag on
origin + GitHub at commit `fff67d9`. Authoritative scope +
execution outcomes in [`docs/archive/plans/PLAN_V2.15.md`](archive/plans/PLAN_V2.15.md)
(Draft v0.9 — 8-round audit archaeology in Appendix A). AFTER
snapshot at
[`docs/QUALITY_SNAPSHOT_2026-05-24_v2.15_after.md`](QUALITY_SNAPSHOT_2026-05-24_v2.15_after.md).

**v2.16 convergence-cycle plan at Draft v0.5 (mid-audit).**
[`docs/PLAN_V2.16.md`](PLAN_V2.16.md) frames v2.16 as the final
v2.X release: every open item gets SHIP / KILL / OUT-OF-SCOPE
(v3.0). Round 1-3 dispositions captured in Appendix A; Round 4
required (Round 3 returned 5 HIGH structural findings). Cycle
NOT yet open. When ready to open, read
[`docs/CYCLE_OPEN_CHECKLIST.md`](CYCLE_OPEN_CHECKLIST.md) FIRST —
it specifies the analyzer run + USER_ISSUES.md review + Docling
watcher + calibration freshness check that produce v2.16's
Carry-Forwards table inputs.

## v2.15 Phase Status — FINAL (mirrors PLAN_V2.15.md §3 close-out)

| Phase | Topic | Final status |
|---|---|---|
| Pre-cycle | dashscope-rollback drop | ✓ COMPLETED 2026-05-23 PM (commit `2527414`) |
| 1 [U/E] | Targeted HyDE bridging | **CLOSED as DEAD LEVER** 2026-05-24 PM (post-tag soak n=224 across 5 docs; falsification rule fired — 4/5 docs zero R@1 delta) |
| 2 [A] | pdfplumber lane | **SKIPPED** per Option F |
| 3 [F] | Document-class query telemetry | ✓ SHIPPED — full suite (5 modules + 2 docs + soak hook + 29 tests) |
| 4 [A] | Docling HybridChunker tuning | **SKIPPED** per Option F (carry-forward 6.1 trigger active for v2.16+) |
| 5 [E] | Retrieval-side investments | **SKIPPED** per Option F |
| 6 [U] | Calibration freshness check | ✓ SHIPPED — FP8-14B cal fresh through 2026-06-22 |
| N | Cycle close-out + 2.15.0 tag | ✓ PUSHED 2026-05-24 (commit `fff67d9`) |

## v2.14 Phase Status (HISTORICAL — see Final outcomes above)

This table reflects the final outcomes section above and is kept
inline as a quick anchor; the canonical source is
[`docs/archive/snapshots/QUALITY_SNAPSHOT_2026-05-23_v2.14_after.md`](archive/snapshots/QUALITY_SNAPSHOT_2026-05-23_v2.14_after.md)
§1 + §8 addendum.

| Phase | Topic | Final status |
|---|---|---|
| 0 | Local-judge calibration | FP8-14B operative (v2.14.1): rel 82.2 / **format 90.7 TRUSTWORTHY** / faith 76.6. 27B-MTP and 14B-BF16 predecessors retired. |
| 1 | Form/Table layout extraction recovery | **PARTIAL** — code WIN preserved; data ROLLED BACK; carried to v2.15 Option A Phase 2 |
| 2 | Targeted HyDE bridging | SHIPPED post-tag (`156dfa7`) opt-in; broad lift FALSIFIED; v2.15 Phase 1 re-targets |
| 3 | Rollback-collection drop | SHIPPED post-tag (`2527414`) under user override; ~30 GB reclaimed |
| 4a | Local HyDE provider | SHIPPED 2026-05-22; Qwen3 thinking-mode fix (`0c5e818`) |
| 4b | Local judge format-axis | RE-OPENED by v2.14.1 FP8-14B Format TRUSTWORTHY verdict; available for v2.15 |
| 4c | Local query gen | SHIPPED 2026-05-23 (`1c201dd`) |
| 4d | Tie-breaker harness | SHIPPED 2026-05-23 (`local_then_cloud_soak.py`, 14 tests) |
| 4e | 1500-query Format-only demo | DROPPED — moot under FP8-14B Format TRUSTWORTHY |
| 4-Resilience | qwen3-max cloud fallback for HyDE | SHIPPED 2026-05-23 (`1c001dd`, 3 bridge tests) |
| 5 | Soak disk-headroom precheck | SHIPPED 2026-05-22 |
| 6 | Code-block chunking hygiene | **PARTIAL** — scanned_book + observability shipped; Docling-layer defect carried to v2.15 Option A Phase 4 |
| N | Cycle close-out + 2.14.0 tag | ✓ PUSHED 2026-05-23 (commit `36482e0`, sha `122a62e`) |

## Other Carry-Forwards (post-v2.15 ship state)

- **3a (VLM swap)** — KEPT as v2.14 Phase 1 fallback infrastructure
  (force_table_vlm shipped). v2.16 Phase 4 ships the missing dedup
  piece (CarOK IoU>85% suppression). Per PLAN_V2.16.md disposition
  matrix, 3a/Item #14 will KILL at v2.16 close (current VLM works).
- **3c (UIR refactor)** — **PARKED WITH TRIGGERS** (user disposition
  2026-05-24 PM). Four reopen conditions in `docs/CYCLE_OPEN_CHECKLIST.md`
  §5. Per PLAN_V2.16.md disposition matrix, 3c/Item #13 will KILL
  at v2.16 close (triggers unrealistic for solo-dev PDF-only use).
- **3e (magazine rendered-region-crop)** — deferred. Per PLAN_V2.16.md
  disposition matrix, 3e/Item #15 will KILL at v2.16 close (no
  demand signal across v2.11→v2.15).
- **HyDE** — CLOSED as DEAD LEVER post-v2.15-tag (see headline
  section). Infra retained in tree as opt-in; production defaults
  unchanged.

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
- Dashscope text-embedding-v4 (formerly the 30-day rollback baseline)
  collection `mmrag_v2_8__qwen3_dashscope` was **dropped 2026-05-23 PM**
  (v2.14 Phase 3 under user override). Cold-storage snapshot retained
  on `multimodal-doc-converter_qdrant_snapshots` through 2026-08-21
  for recovery. Not the default for any code path. Ollama/llava lane
  is legacy; do not use as a comparison baseline.
- `MLX_API_KEY` env var is required for production retrieval (omlx
  embedder + omlx reranker). `DASHSCOPE_API_KEY` is required only
  for synthetic-soak judge + query generation and for the dashscope
  rollback path. Test-suite skip-gates handle the unset case for CI.
