# v2.14.0 AFTER Snapshot — Local-LLM Accelerator Cycle

> Date: 2026-05-23
> Engine: `__engine_version__ = "2.14.0"`
> Schema: `__schema_version__ = "2.7.0"` (unchanged from v2.7)
> Predecessor: v2.13.0 (2026-05-22, annotated tag staged for user push)
> Status: tag **STAGED**, not pushed by autonomous run

## 1. What this cycle shipped

v2.14 layered **local-LLM accelerator infrastructure** on top of the
v2.13.0 retrieval stack. **NO retrieval-stack changes**: production
retrieval is byte-for-byte identical to v2.13.0 (omlx
Qwen3-Embedding-8B-mxfp8 → `mmrag_v2_8__qwen3_local` + BM25 sparse +
RRF + local ModernBERT rerank).

### Phases SHIPPED

| Phase | What | Commit(s) |
|---|---|---|
| **0** (calibration) | 27B-MTP Phase 0 verdict: all 3 axes RESTRICTED (rel 82.0% / format 70.7% / faith 78.8%). Bias direction flipped vs the retired 14B. PERMITTED uses contracted to query-gen + HyDE + tie-breaker harness; ship-gate judging stays on cloud `qwen-max`. | `57e80b0` |
| **4a** (HyDE) | `provider="vllm"` knob shipped 2026-05-22; harness Qwen3-thinking-mode payload fix (`chat_template_kwargs.enable_thinking=False`) 2026-05-23. Live re-smoke: 670-char hypothesis in 8.8s. | `0c5e818` |
| **4c** (gen-provider) | `synthetic_soak.py --gen-provider vllm` wires the local 27B for query generation at $0/query. Live smoke: 2.0s/query against the 27B. | `1c201dd` |
| **4d** (tie-breaker) | `scripts/local_then_cloud_soak.py` two-tier judging — local-vLLM judges all in-scope, cloud `qwen-max` re-judges contested only. Provenance tagged via `judgment.judge_source ∈ {local, cloud, local_fallback}`. 14 unit tests. | `0c3f0da` |
| **4-Resilience** | `hyde.generate_with_fallback` chains vllm → dashscope `qwen3-max` → literal query when primary is vllm. 3 new bridge tests. | `1c201dd` |
| **5** (disk precheck) | `_check_disk_headroom()` in `synthetic_soak.py` aborts retrieve/judge stages below 10 GB free. Override via `SOAK_DISK_HEADROOM_FLOOR_GB` env. | `b70d149` |

### Phases PARTIAL (code shipped; data-acceptance bar NOT met)

| Phase | What landed | What deferred | Commit(s) |
|---|---|---|---|
| **1** (form/table) | `--force-table-vlm` now truly forces (was silently overridden by `technical_manual` profile's `vlm_table_enabled=False`). Local NuMarkdown-8B VLM produces clean 5-col tables on 5/12 CarOK pages. | 30-query CarOK mini-soak measured Format **-26.9pp regression** (45.0% vs 71.9% baseline) — VLM tables coexist with flat-prose duplicates, retrieval picks the prose 29/30 times. Production data **rolled back** to v2.13 baseline. v2.15 needs same-page prose-VLM dedup at chunk-emission time. | `e60a253` (code), `56bf97f` (rollback evidence) |
| **6** (code chunking) | Block-extension policy + `partial_code` schema field on the `_chunk_text_with_overlap` (scanned_book) path. Improves the scanned lane + adds universal observability. | Fluent_Python's truncated-code defect is Docling-extraction-layer (prose+code intermixed at page boundaries); HybridChunker post-merge pass tested in isolation but doesn't fire in production (reverted this session). v2.15 needs upstream Docling-config or post-Docling text-normalization. | `d737147` |

### Phases NOT in scope this cycle

- **Phase 2** (targeted HyDE bridging for code + minority languages) — depended on Phase 6 landing cleanly to enable confounder-free per-doc deficit measurement; deferred.
- **Phase 3** (30-day dashscope-rollback drop) — time-gated to 2026-06-19 decision point; v2.14.1 candidate.

## 2. Production retrieval state (byte-for-byte identical to v2.13.0)

```
query
  ├─ dense  : omlx Qwen3-Embedding-8B-mxfp8 → mmrag_v2_8__qwen3_local (4096-dim, 31,371 pts)
  └─ sparse : BM25 → mmrag_v2_8__bm25_sparse (26,396 pts after Phase 1 rebuild — same content as v2.13)
  → RRF fusion (k=60, equal weights), top-25 candidates per leg
  → rerank (local gte-reranker-modernbert-base-mlx via omlx-server)
  → top-5 return
```

End-to-end p99 latency: **~1.6 s** (unchanged from v2.13.0).
Per-query cost: **$0** on the retrieval path.

## 3. Active model/endpoint state

- **Text embedder**: omlx Qwen3-Embedding-8B-mxfp8 @ `http://10.0.10.246:8000/v1/embeddings`
- **Reranker**: omlx gte-reranker-modernbert-base-mlx @ `http://10.0.10.246:8000/v1/rerank`
- **Local LLM (HyDE/gen/tie-breaker)**: vLLM `Qwen/Qwen3.6-27B-FP8` @ `http://10.0.10.239:8000/v1/chat/completions`
- **Cloud judge (ship-gate)**: Dashscope `qwen-max`
- **Cloud HyDE fallback** (Phase 4-Resilience): Dashscope `qwen3-max`
- **Local VLM** (auto-path + `--force-table-vlm`): omlx `NuMarkdown-8B-Thinking-mlx-8bits`

## 4. Validation evidence

### v2.14 retrieval fingerprint (canonical for v2.14+)

- **`tests/fixtures/retrieval_regression_v2_14_hybrid.json`** — 20-query fingerprint, captured 2026-05-23 post-rollback against live production stack. `scripts/retrieval_regression_v2_14.py` verifies. **20/20 PASS at ship.**
- v2.13 fingerprint (`retrieval_regression_v2_13_hybrid.json`) and `scripts/retrieval_regression_v2_13.py` retained for historical comparison. The v2.13 fingerprint failed 18/20 against the post-Phase-1-churn live stack — not from real retrieval regression but from HNSW graph rebuild + sparse collection drop+recreate cycles during the Phase 1 experiments. 17/20 of those v2.13-fingerprint failures still match on `doc_id` at top-1 (semantic continuity preserved); the 3 differing-doc cases are tie-breakers between topically-similar source books (RAG eval / RAG arch / Python iterator content).

### Test suite

- **1053 passed / 16 skipped / 0 failed** (up from v2.13.0's 1033). New additions:
  - `tests/test_hyde.py` — 17 tests (was 12; +3 Phase 4 Resilience + 2 Phase 0 thinking-mode bridge tests)
  - `tests/test_code_chunking.py` — 9 tests (was 6; +3 Phase 6 shape-specific cases)
  - `tests/test_local_then_cloud_soak.py` — 14 tests (new; Phase 4d harness coverage)

### Strict-gate corpus state

**Unchanged from v2.10** (last touched at v2.10): 34/34 PASS. v2.11+v2.12+v2.13+v2.14 all touched retrieval-side / observability only, not extraction/chunking/validation.

## 5. Cost summary

- **Total cloud spend this cycle**: ~$1.20 (Phase 0 re-cal qwen-max judging on 518 queries + Phase 1 mini-soak qwen-max judging on 30 queries + a handful of one-off probes). **Well under the $25/cycle cap.**
- **Phase 1 cloud-VLM escalation budget** (`qwen-vl-plus`) **not spent** — local NuMarkdown-8B handled the table extraction at $0 (though the data outcome regressed and was rolled back, the cloud budget was preserved).
- **Local LLM utilization**: ~2hr of GX10 vLLM time across Phase 0 re-cal + smoke tests + Phase 1 generation + Phase 4d smoke. $0.

## 6. Open carry-forwards into v2.15

1. **Phase 1.1 — Same-page VLM/prose dedup**: when `--force-table-vlm` produces a clean VLM table chunk for a page, suppress the parallel `hybrid_chunker` prose chunk emission for that same page. Today's mini-soak proved that dual-emission lets the prose win retrieval, dead-weighting the VLM tables. This is a `processor.py` chunk-emission-policy change, not a flag.
2. **Phase 6.1 — Docling prose+code disambiguation**: the Fluent_Python "truncated code" defect is upstream of chunking. Fluent_Python p326's CODE chunk ends mid-statement at `    return`, and the continuation `cls(memv)\n\`\`\`\n` is the first 11 chars of a PARAGRAPH chunk that explains the code in prose. Either: (a) configure Docling differently to preserve clean CodeItem boundaries, or (b) post-Docling text-normalization that re-inserts whitespace + closes fences around code transitions.
3. **Phase 3 — 30-day dashscope rollback drop**: time-gated to 2026-06-19. v2.14.1 candidate. Per Draft v0.5: needs 90-day cold-storage snapshot before deletion.
4. **Phase 2 — Targeted HyDE bridging**: still on the books. Re-evaluate after Phase 1.1 + Phase 6.1 produce confounder-free per-doc deficit measurements.

## 7. References

- Plan: [`docs/PLAN_V2.14.md`](PLAN_V2.14.md) (Draft v0.5)
- v2.13.0 AFTER baseline: [`docs/QUALITY_SNAPSHOT_2026-05-22_v2.13_after.md`](QUALITY_SNAPSHOT_2026-05-22_v2.13_after.md)
- Phase 0 calibration report: [`docs/CALIBRATION_2026-05-23_v2.14_p0_local_judge_qwen36_27b_mtp.md`](CALIBRATION_2026-05-23_v2.14_p0_local_judge_qwen36_27b_mtp.md)
- v2.14 fingerprint: `tests/fixtures/retrieval_regression_v2_14_hybrid.json` (20/20 PASS)
- Project status: [`docs/PROJECT_STATUS.md`](PROJECT_STATUS.md)
