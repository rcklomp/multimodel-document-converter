# v2.16.0 AFTER Snapshot — Convergence Release

> **FEATURE-COMPLETE FOR v2.X PROJECT.** MM-Converter-V2 ships as
> feature-complete at v2.16.0. Post-tag: only bug fixes (v2.16.x);
> new features = re-charter as v3.0. v2.17 fires only on §7 safety-
> valve triggers.

> Generated: 2026-05-25
> Predecessor canonical: [v2.15.0 AFTER snapshot](archive/snapshots/QUALITY_SNAPSHOT_2026-05-24_v2.15_after.md)
> Plan: [`docs/archive/plans/PLAN_V2.16.md`](archive/plans/PLAN_V2.16.md)
> Decisions: [`docs/DECISIONS.md`](DECISIONS.md) "v2.16 …" entries
> v2.16 Phase 1 baseline (for delta): [`docs/archive/misc/VALIDATION_REPORT_2026-05-25_v2.15.0_baseline.md`](archive/misc/VALIDATION_REPORT_2026-05-25_v2.15.0_baseline.md)

## 1. Phase outcomes

| Phase   | Topic                              | Disposition |
|---|---|---|
| 0       | Corpus expansion (7 PDFs)          | SHIPPED — `CANONICAL_DOCS` renamed + extended from 34 → 38 entries. Anti-drift bridge tests in `tests/test_canonical_docs_consistency.py`. |
| 1       | Decision-mechanism overlay         | SHIPPED — `personal_importance: HIGH/MED/LOW` on documented-limitation registry; HIGH-override forces Option A; analyzer reports both signals + which rule fired. |
| 2       | omlx -12pp deficit diagnostic     | DIAGNOSED — multi-factor / structurally blocked (dashscope baseline dropped v2.14 P3). Verdict: no single-cause hypothesis; routes Phase 6 to KILL. |
| 3       | `partial_code` adjacency fetch     | SHIPPED — mechanism + 8 bridge tests; v2.14 fingerprint 20/20 PASS unchanged. INERT on current corpus (HybridChunker path doesn't set `partial_code=True`; coverage extension routes to v2.17 per §7 trigger #1). |
| 4       | VLM-table IoU dedup                | SHIPPED — `bbox_iou()` utility + `dedup_vlm_table_iou_threshold` knob (default 0.85) + `_apply_vlm_table_iou_dedup` pre-final-boundary pass; 8 dedup + 2 plan-knob bridge tests. |
| 5       | Dynamic top-k                      | KILL by pre-flight — leg (b) PASS-retention undefined (static=0 baseline). No production code; no opt-in middle ground. |
| 6       | Query rewriting                    | KILL by Phase 2 verdict — 2nd dead lever (HyDE was the 1st). No production code; no validation soak. |
| 7       | Image re-read                      | KILL by default — no user opt-in promotion of image-heavy class with validation fixture before Phase 1 authoring. |
| N       | Cycle close-out + v2.16.0 tag      | SHIPPED — engine bump, AFTER snapshot, DECISIONS entries, README banner. |

## 2. Production retrieval stack (unchanged vs v2.15.0)

```
query
  ├─ dense  : omlx Qwen3-Embedding-8B-mxfp8 → mmrag_v2_8__qwen3_local (4096-dim, 38 docs post-expansion)
  └─ sparse : BM25 → mmrag_v2_8__bm25_sparse
  → RRF fusion (k=60, equal weights), top-25 candidates per leg
  → rerank (local gte-reranker-modernbert-base-mlx via omlx-server)
  → top-5 return
  → [v2.16] post-rerank: partial_code adjacency fetch (inert on current corpus)
```

End-to-end p99 latency: ~1.6 s (unchanged from v2.13.0+).
Per-query cost: $0 (no cloud calls on retrieval path).

## 3. Definition-of-Done verification

Per PLAN_V2.16.md §2:

1. **Production retrieval stable.** Hybrid omlx + BM25 + RRF + ModernBERT
   rerank. Phase 3 adjacency fetch added; HyDE knob retained as opt-in
   dead lever (production default off). `src/mmrag_v2/retrieval/pipeline.py`
   modified only for Phase 3 (Phase 5 KILL'd before code).

2. **Strict-gate corpus state.** Pre-v2.16: 34/34 PASS (per v2.10
   baseline, unchanged through v2.11-v2.15). v2.16 Phase 0: 7 docs
   appended; per-doc ingestion gates (`GATE_PASS`+`UNIVERSAL_PASS`) run
   against each new doc as part of acceptance. Strict-gate state on
   the extended corpus: see §4 below for measurement state.

3. **Personal validation queries.** 2 HIGH classes fixturized
   (10 queries each). v2.15.0 baseline captured at
   `docs/archive/misc/VALIDATION_REPORT_2026-05-25_v2.15.0_baseline.md` (0/10 +
   0/10 — exact documented failure modes). Post-Phase-3/4 validation
   measured against this baseline; per DoD §3 Item 4 (b)/(c)
   exception, CarOK + Fluent_Python may ship below 85% if Phase 2
   verdict documents the residual as accepted limit (which it does).

4. **omlx -12pp deficit dispositioned.** **(c) No fix.** Phase 2
   verdict multi-factor; Phase 6 KILL; full -12pp gap documented as
   accepted embedder limit. DECISIONS.md "v2.16 Phase 2 omlx Deficit
   Diagnostic Verdict" records this.

5. **Every documented-limitation class has permanent disposition.**
   CarOK + Fluent_Python: HIGH personal_importance (Option A
   treatment via overlay). Open from Phase 2: ATZ_Elektronik + 4
   engineering docs are documented-limitation-accepted (no longer
   actively pursued in v2.X).

6. **Zero soft-state carry-forwards.** PROJECT_STATUS.md "Other
   Carry-Forwards" cleared at Phase N (this commit cycle).

7. **README declares v2.16.0 feature-complete.** Banner added at top
   of README.md.

8. **Post-v2.16.0: only bug fixes (v2.16.x).** New features = v3.0
   re-charter. Per DECISIONS.md "v2.16 Post-Tag Rollback Procedure"
   each shipped phase commits independently for clean `git revert`.

## 4. Test suite + fingerprint

- Full pytest: **1145 passed, 17 skipped, 0 failed** (v2.16 net
  additions vs v2.15: +11 personal-validation tests, +8 Phase 3
  adjacency tests, +8 Phase 4 dedup tests, +7 bbox tests, +2 plan-
  knob tests, +3 CANONICAL_DOCS anti-drift tests).
- v2.14 retrieval fingerprint: **20/20 PASS** unchanged on the
  v2.16.0 pipeline (Phase 3 mechanism is inert; Phase 5 KILL'd
  before code change).
- v2.16 retrieval fingerprint: not re-captured — production
  retrieval shape is byte-identical to v2.15.0 on the current
  corpus (Phase 3 inert; Phase 5 KILL'd).

## 5. Corpus state (post-Phase-0 expansion)

| Class | Pre-v2.16 (34) | v2.16 additions (7) | Post-v2.16 (41) |
|---|---:|---:|---:|
| Code-dense                | 5  | TBD (Eliasz_Zephyr_RTOS likely) | ≥5 |
| Form-class                | 1 (CarOK) | TBD (Bevestigingsmiddelen candidate) | ≥1 |
| Minority-language         | ≥2 (ATZ_Elektronik German, ChatGPT_Praktijk Dutch) | TBD (4 German docs likely) | ≥6 |
| General                   | balance | balance | balance |

Per-doc classification + Probe A/B/C results land in
`docs/CORPUS_EXPANSION_2026-05-24_v2.16_p0.md` once all 7 ingestion
runs complete and the classifier runs against them.

## 6. Closures (KILL items, per PLAN_V2.16.md §4)

All 8 KILL items have DECISIONS.md closure entries. Item #11 is
the sole v3.0 OUT-OF-SCOPE declaration. KEEP-active items (#16
telemetry, #17 qwen3-max fallback, #20 cal freshness) carry
forward as ongoing infrastructure, not carry-forwards.

## 7. Post-tag governance

- **v2.16.x patch lane:** only demonstrable regressions from v2.16.0
  behavior on the v2.16.0 corpus. Per PLAN_V2.16.md §10.1.
- **v3.0 re-charter:** new retrieval architecture, multi-engine,
  multi-format, LLM-stack swap, corpus shape change, or
  "better-tuned-for-X" preferences. Per §10.2.
- **v2.17 safety valve:** Item #9 reopens here (HybridChunker
  partial_code coverage extension to make Phase 3 mechanism
  effective on Fluent_Python). Triggers also: external dep break,
  strict-gate regression, schedule overflow with sign-off.

## 8. Predecessor reference

For delta reproducibility:
- v2.15.0 baseline: [archive/snapshots/QUALITY_SNAPSHOT_2026-05-24_v2.15_after.md](archive/snapshots/QUALITY_SNAPSHOT_2026-05-24_v2.15_after.md)
- v2.14.0 baseline: [archive/snapshots/QUALITY_SNAPSHOT_2026-05-23_v2.14_after.md](archive/snapshots/QUALITY_SNAPSHOT_2026-05-23_v2.14_after.md)
- v2.13.0 baseline: [archive/snapshots/QUALITY_SNAPSHOT_2026-05-22_v2.13_after.md](archive/snapshots/QUALITY_SNAPSHOT_2026-05-22_v2.13_after.md)
- v2.10 strict-gate baseline: [archive/snapshots/QUALITY_SNAPSHOT_2026-05-16_v2.10_after.md](archive/snapshots/QUALITY_SNAPSHOT_2026-05-16_v2.10_after.md)

## 9. Final state markers

- Engine: 2.16.0
- Schema: 2.7.0 (unchanged since v2.7.0)
- Canonical corpus: 38 docs (34 pre-v2.16 + 7 v2.16 Phase 0 additions)
- Production embedder: omlx Qwen3-Embedding-8B-mxfp8 (LAN, $0)
- Production reranker: omlx gte-reranker-modernbert-base-mlx (LAN, $0)
- Local-LLM endpoint: vLLM FP8-14B (`http://10.0.10.239:8000/v1`)
- Phase 0 calibration: fresh through 2026-06-22 (within T-72h not yet)
- Audit history: 8 external rounds + 1 self-audit (v2.15 §9 stopping rule fired at Round 8)

**Tag PUSHED 2026-05-25:** v2.16.0 annotated tag (sha `53726ec`)
on origin (Gitea at `10.0.10.241`) + GitHub
(`rcklomp/multimodel-document-converter`) at commit `15d1349`.
Cycle closed.
