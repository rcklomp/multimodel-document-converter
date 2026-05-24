# v2.15.0 AFTER Snapshot — Option F (Telemetry-Augmented Hybrid)

> Date: 2026-05-24
> Engine: `__engine_version__ = "2.15.0"`
> Schema: `__schema_version__ = "2.7.0"` (unchanged from v2.7)
> Predecessor: v2.14.0 (2026-05-23, tag `122a62e` PUSHED to origin)
> Strategic path: **Option F** (telemetry-augmented hybrid) —
> user-explicit selection ahead of T-24h silent-default window.

## 1. What this cycle shipped

v2.15 executed under **Option F** of the §2 strategic fork. Active
phases: Phase 1 (HyDE bridging), Phase 3 (telemetry infrastructure),
Phase 6 (calibration freshness), Phase N (close-out). Skipped:
Phase 2 [A] (pdfplumber lane), Phase 4 [A] (Docling HybridChunker
tuning), Phase 5 [E] (retrieval-side investments).

**NO retrieval-stack changes** vs v2.14.0 (which had none vs
v2.13.0): production retrieval is byte-for-byte identical
(`pipeline.py` unmodified this cycle; the Phase 3 telemetry hook
sits in the soak-harness write path AFTER each retrieve call —
zero impact on candidate ordering or rerank scores).

### Phases SHIPPED

| Phase | What | Commit(s) |
|---|---|---|
| **6 [U]** (calibration freshness) | FP8-14B Phase 0 cal verified fresh through **2026-06-22** (no re-cal needed at cycle open). T-72h pre-tag checkpoint armed via `docs/CYCLE_OPEN_CHECKLIST.md` `cycle_slip.log` for future cycles. | (verification step only; no code change) |
| **3 [F]** (telemetry suite) | Full Option F instrumentation — 5 new modules + 2 new docs + soak-harness hook + 29 unit tests. See §2 below for component list. DECISIONS.md telemetry-threshold entry transitioned **PRE-CYCLE PROPOSAL → ACTIVE RULE** concurrent with the code landing. | `ca1fa18` |

### Phases SHIPPED (post-tag patch range, 2026-05-24 PM)

| Phase | What | Commit(s) |
|---|---|---|
| **1 [U]** (HyDE bridging) | **CLOSED as DEAD LEVER** post-tag. User authorized `MLX_API_KEY` sourcing from `.zshrc` 2026-05-24 PM; A/B soak ran end-to-end (n=224 queries across 5 docs; HyDE-off baseline vs HyDE-on auto-intent test). Falsification rule fired (4/5 docs ZERO R@1 delta; aggregate +0.4pp within noise; German subgroup +0.0 on n=64). DECISIONS.md entry "v2.15 Phase 1 HyDE Bridging — CLOSED as Dead Lever" records the closure; opt-in infra stays in tree (production defaults unchanged). Report: `docs/SOAK_2026-05-24_v2.15_p1_narrow_hyde_AB.md`. v2.14 retrieval fingerprint re-verified live: 20/20 PASS. | (post-tag, this commit) |

### Phases SKIPPED (Option F deferral)

- **Phase 2 [A]** (pdfplumber lane) — v2.16 contingent on Phase 3 telemetry hit-rate evidence per the ≥5% promotion-standard-arm rule.
- **Phase 4 [A]** (Docling HybridChunker config tuning) — carry-forward 6.1 re-evaluation trigger ("Docling minor ≥2.87 OR every 90 days, whichever first") registered in `docs/CYCLE_OPEN_CHECKLIST.md`.
- **Phase 5 [E]** (retrieval-side investments) — v2.16 contingent on F→E telemetry-escalation trigger (≥3 consecutive middle-band cycles).

## 2. Phase 3 [F] component inventory

| Component | Purpose | Tests |
|---|---|---|
| `src/mmrag_v2/retrieval/documented_limitations.py` | Single source of truth: thresholds (`PROMOTION_THRESHOLD_PCT=5`, `CLOSURE_THRESHOLD_PCT=1`, `DEFECT_OVERRIDE_THRESHOLD_PCT=1`, `MIDDLE_BAND_PERSISTENCE_CYCLES=3`, `NEW_CLASS_GRACE_CYCLES=2`) + entry classes (`CarOK_voorraadtelling` + `Fluent_Python`, both `severe_defect_tag=True` per prior-cycle defect history). | Covered indirectly via `test_doc_class_telemetry.py::test_registry_*` + `test_cycles_since_*` |
| `src/mmrag_v2/retrieval/telemetry.py` | `compute_document_class_hits` + `build_telemetry_record` helpers. Pipeline stays side-effect-free; soak harness owns the rolling-log write. | `test_compute_hits_*` (8 cases) + `test_build_record_*` (3 cases) |
| `scripts/analyze_doc_class_telemetry.py` | Cycle-open analyzer — reads rolling log, applies all 3 promotion arms + closure + middle-band-aging + grace-period rules, emits dated `docs/TELEMETRY_REPORT_<YYYY-MM-DD>.md` with explicit trigger-fired booleans per class. | `test_analyzer_*` (7 end-to-end cases covering each rule branch + empty-log edge) |
| `scripts/verify_phase2_teardown.py` | Phase N DoD gate (Round-6 Finding 5 + Round-8 Finding 6) — 4-assertion programmatic check for the Phase 2 Abort Teardown Mandate. Vacuously satisfied under Option F (early-exit on detecting Option-F selection in DECISIONS.md). | `test_verify_phase2_teardown.py` (9 cases: vacuous-pass under F, full-pass under hypothetical A, 4 independent FAIL paths, edge cases) |
| `scripts/synthetic_soak.py` | `_append_telemetry` hook added to both retrieve branches (hybrid + legacy); writes to `output/telemetry/document_class_hits.jsonl` (env-overridable via `MMRAG_TELEMETRY_LOG`). Best-effort writes — failures don't break the soak. | Indirectly covered via soak runs |
| `docs/USER_ISSUES.md` | Append-only markdown table; analyzer counts entries per `doc_class` for the `open_user_issues` signal in the standard promotion arm. | (data file; no tests) |
| `docs/CYCLE_OPEN_CHECKLIST.md` | Cycle-open process — analyzer run, USER_ISSUES.md review, Docling release-notes check (carry-fwd 6.1 trigger), calibration freshness, `cycle_slip.log` spec for T-72h slip handling. **Load-bearing artifact**: closes Round-4 Finding 1 (telemetry-without-reader) + Round-4 Finding 6 (Docling watcher) + Round-6 Finding 6 (chronological-race fix). | (process doc; no tests) |
| `scripts/sample_phase1_narrow_fixture.py` | Phase 1 prep — 5-doc stratified sampler per the v0.9 fixture spec. | (data-generation script; tested via end-to-end Phase 1 soak when executed) |

## 3. Production retrieval state (byte-for-byte identical to v2.14.0)

```
query
  ├─ dense  : omlx Qwen3-Embedding-8B-mxfp8 → mmrag_v2_8__qwen3_local (4096-dim)
  └─ sparse : BM25 → mmrag_v2_8__bm25_sparse
  → RRF fusion (k=60, equal weights), top-25 candidates per leg
  → rerank (local gte-reranker-modernbert-base-mlx via omlx-server)
  → top-5 return
```

End-to-end p99 latency: ~1.6 s (unchanged from v2.14.0).
Per-query cost: $0 on the retrieval path.

## 4. Active model/endpoint state (unchanged from v2.14.1)

- **Text embedder**: omlx Qwen3-Embedding-8B-mxfp8 @ `http://10.0.10.246:8000/v1/embeddings`
- **Reranker**: omlx gte-reranker-modernbert-base-mlx @ `http://10.0.10.246:8000/v1/rerank`
- **Local LLM (HyDE/gen/tie-breaker)**: vLLM `RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic` @ `http://10.0.10.239:8000/v1/chat/completions` (GX10 Blackwell-native FP8; bare config — n-gram spec REJECTED per `memory/project_v2_14_ngram_spec_rejected.md`)
- **Cloud judge (ship-gate)**: Dashscope `qwen-max`
- **Cloud HyDE fallback**: Dashscope `qwen3-max`
- **Local VLM**: omlx `NuMarkdown-8B-Thinking-mlx-8bits`

## 5. Validation evidence

### Test suite

- **1106 passed / 17 skipped / 0 failed** (up from v2.14.0's 1053). New: 29 tests across `tests/test_doc_class_telemetry.py` (20) + `tests/test_verify_phase2_teardown.py` (9), all green at promotion. Net +53 includes v2.14.x patch-range additions (intent classifier, etc.) on top of v2.14.0's 1053 baseline.
- Version-pin guard test (`test_v2_10_release_baseline.py::test_engine_and_schema_version_pinned`) updated from `"2.14.0"` to `"2.15.0"` to keep `pyproject.toml ↔ version.py` drift detection live.

### Retrieval fingerprint

- **v2.14 fingerprint** (`tests/fixtures/retrieval_regression_v2_14_hybrid.json`) is the canonical pin for v2.15. Production retrieval is byte-for-byte identical (pipeline.py untouched; Phase 3 hook is post-retrieve telemetry-only). **Live verification 2026-05-24 PM: 20/20 PASS** via `scripts/retrieval_regression_v2_14.py` (ran once `MLX_API_KEY` env was available).
- No v2.15 fingerprint captured — would be redundant against the v2.14 fingerprint given zero retrieval-stack delta.

### Strict-gate corpus state

**Unchanged from v2.10** (last touched at v2.10): 34/34 PASS.
v2.11+v2.12+v2.13+v2.14+v2.15 all touched retrieval-side /
observability only, not extraction/chunking/validation.

### Phase 6 calibration freshness

- FP8-14B Phase 0 calibration: **SHIPPED 2026-05-23 PM**
  (`docs/CALIBRATION_2026-05-23_v2.14_p0_local_judge_14b_fp8.md`)
- 30-day window expires **2026-06-22**
- Today (cycle open): 2026-05-24 → fresh; no re-cal needed at
  cycle open. T-72h pre-tag checkpoint armed for the close-out
  window (will fire automatically if tag date crosses
  2026-06-19 = expiration - 72h).

## 6. Cost summary

- **Total cloud spend this cycle**: **~$2.50** (post-tag Phase 1
  A/B soak — 224 query-gen calls + 448 judge calls via cloud
  qwen-max). Well under the $25 cap.
- **Local LLM utilization**: GX10 FP8-14B endpoint served the
  HyDE-on test arm's auto-intent HyDE generation (224 queries ×
  intent-matching rate ~50% = ~110 HyDE calls × ~9s each ≈ 16 min
  of GX10 inference time). $0 cost.

## 7. Open carry-forwards into v2.16

The cycle-open checklist (new in this cycle) is the load-bearing
artifact for ALL carry-forwards going forward. Run
`python scripts/analyze_doc_class_telemetry.py --current-cycle v2.16`
at v2.16 open and follow the checklist.

| # | Item | Status |
|---|---|---|
| 1 | Phase 1 HyDE bridging | **CLOSED as dead lever** 2026-05-24 PM (post-tag). Falsification rule fired (4/5 docs ZERO R@1 delta). DECISIONS.md "v2.15 Phase 1 HyDE Bridging — CLOSED as Dead Lever". Opt-in infra stays in tree; production defaults unchanged; **NOT a v2.16 carry-forward**. |
| 2 | Phase 3 telemetry data collection | Active as of v2.15.0 — every soak run via `synthetic_soak.py` writes to the rolling log. v2.16 analyzer reads the 30-day / 60-day windows for promotion / closure / middle-band rules. |
| 3 | Phase 2 [A] pdfplumber lane | Deferred indefinitely under Option F. Eligible for promotion in v2.16+ if `CarOK_voorraadtelling` hit-rate ≥5% (standard arm) OR severe-defect-tag override fires at ≥1%. |
| 4 | Phase 4 [A] Docling HybridChunker tuning | Deferred indefinitely under Option F. Carry-forward 6.1 re-evaluation trigger: Docling minor ≥2.87 OR every 90 days. Cycle-open checklist item #3. |
| 5 | Phase 5 [E] retrieval-side investments | Deferred indefinitely under Option F. Eligible for promotion in v2.16+ if F→E telemetry-escalation trigger fires (≥3 consecutive middle-band cycles on any class). |
| 6 | 3c (UIR refactor) | FORCE-CLOSED per Round-2 Finding 4 — needs explicit user re-charter as a fresh `docs/PLAN_*_UIR.md` proposal or permanent close in v2.16. v0.9 DoD blocks v2.15.0 tag without disposition; closed via this AFTER snapshot's recommendation: **CLOSE from carry-forwards** (zero forward motion across 5 cycles; user can re-open with a concrete trigger if needed). |

## 8. References

- Plan: [`docs/PLAN_V2.15.md`](PLAN_V2.15.md) (Draft v0.9; 8-round audit archaeology in Appendix A)
- Audit prompt (re-runnable for v2.16+): [`docs/PLAN_V2.15_AUDIT_PROMPT.md`](PLAN_V2.15_AUDIT_PROMPT.md)
- Predecessor: [`docs/QUALITY_SNAPSHOT_2026-05-23_v2.14_after.md`](QUALITY_SNAPSHOT_2026-05-23_v2.14_after.md) (v2.14.0 ship state + §8 post-ship addendum)
- Phase 0 calibration: [`docs/CALIBRATION_2026-05-23_v2.14_p0_local_judge_14b_fp8.md`](CALIBRATION_2026-05-23_v2.14_p0_local_judge_14b_fp8.md) (FP8-14B operative verdict)
- Telemetry threshold rules: [`docs/DECISIONS.md`](DECISIONS.md) §"v2.15 Documented-Limitation Telemetry Threshold (ACTIVE RULE)"
- Strategic Option F decision: [`docs/DECISIONS.md`](DECISIONS.md) §"v2.15 Strategic Path — Option F Selected"
- Project status: [`docs/PROJECT_STATUS.md`](PROJECT_STATUS.md)
