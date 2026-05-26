# Phase A — Skipped-Tests Audit (A8)

**Charter:** [`ARCHITECTURE_V3_DRAFT_0.5.md`](ARCHITECTURE_V3_DRAFT_0.5.md) Phase A task A8
**Charter gate (§8.2):** each skipped test classified; re-enabled tests pass before Phase A merge.
**Baseline:** v2.16.0 commit `15d1349`; foundation session 2026-05-26.
**Total skipped:** 17 (matches Charter "17-skipped-tests audit (18 as of this draft per `pytest -v --co`)" — actual is 17; the audit baseline appears to have been captured pre-tag).

## Classification key

- `still-skip` — environment-gated test that legitimately depends on a
  resource (Ollama, Qdrant collection, omlx-server, env var). No action.
- `re-enable-now` — test can be re-enabled in foundation session by
  flipping the env-gate default to ON. Verify it passes.
- `re-enable-post-A` — test is UIR-dependent and must be re-enabled
  AFTER the Phase A UIR refactor lands. Phase A merge gate.

## Audit table

| # | Test | Skip reason | Classification | Rationale |
|---|---|---|---|---|
| 1 | `test_chunk_id_collision_v29.py:186` | `Set RUN_CORPUS_SCAN=1` | `still-skip` | Corpus-wide scan over `output/` JSONLs — opt-in by design; not UIR-dependent. |
| 2 | `test_classifier_firearms_route.py:185` | `Set RUN_FIREARMS_VERIFY=1 after Phase 5 v2.9 re-conversion of Firearms` | `still-skip` | Verification gate from v2.9 Phase 5 (long-closed); requires re-conversion fixture not in tree. Not UIR-related. |
| 3 | `test_docling_postprocessor_acceptance.py:235` | `Set HARRY_ACCEPTANCE_JSONL=<path> or RUN_HARRY_ACCEPTANCE=1` | `re-enable-post-A` | Third-party regression check (Charter §3.2): Harry Potter drop-cap promotion via UIR or its UIR-native equivalent. Must pass at A5. |
| 4 | `test_hybrid_chunker_dense_page_router.py:228` | `set RUN_DENSE_ROUTER_PERF=1 — Ayeva batch performance smoke` | `still-skip` | Performance smoke; opt-in. Not UIR-correctness. |
| 5 | `test_hybrid_chunker_dense_page_router.py:282` | `set RUN_DENSE_ROUTER_PERF=1 — real Ayeva Docling coverage` | `re-enable-post-A` | Real Docling chunker output on Ayeva — Phase A A3 must keep this behavior. Re-enable after A3. |
| 6 | `test_hybrid_chunker_dense_page_router.py:291` | `set RUN_DENSE_ROUTER_PERF=1 — real Ayeva Docling coverage` | `re-enable-post-A` | Same rationale as #5. |
| 7 | `test_retrieval_regression_v2_10.py:99` | `Ollama at http://localhost:11434 does not have the llava model` | `still-skip` | v2.10 legacy llava lane explicitly dropped 2026-05-23 (v2.14 Phase 3, commit `2527414`). Test is intentionally vestigial. Consider deletion in Phase A cleanup. |
| 8 | `test_retrieval_regression_v2_11.py:88` | `Qdrant collection mmrag_v2_8__qwen3_dashscope unreachable` | `still-skip` | Same — dashscope rollback collection dropped 2026-05-23. Vestigial; deletion candidate. |
| 9 | `test_retrieval_regression_v2_12.py:105` | `omlx-server unreachable or MLX_API_KEY not set` | `still-skip` | Runtime env gate. Passes when `MLX_API_KEY` is set and `:8000` is reachable. Production CI environment varies. |
| 10 | `test_semantic_overlap.py:176` | `Embedding model not available: assert 25 >= 40` | `still-skip` | Model-availability env gate (production embedder dimensions). |
| 11 | `test_toc_index_page_contract.py:17` (×6 cases) | `set RUN_TOC_PAGE_CONTRACT=1 to validate generated TOC probes` | `re-enable-post-A` | TOC page-contract probes touch HybridChunker's reconciliation paths in `batch_processor.py`. Phase A A2 rewires these paths to UIR; must verify they still pass at A5. |
| 17 | `test_v29_image_enrichment_acceptance.py:51` | `Set RUN_V29_VLM_ACCEPTANCE=1 after Phase 5b enrichment completes` | `still-skip` | v2.9 Phase 5b acceptance gate; v2.9 long-closed; vestigial. Deletion candidate. |

## Summary

| Classification | Count |
|---|---:|
| `still-skip` | 8 |
| `re-enable-now` | 0 |
| `re-enable-post-A` | 9 (tests #3, #5, #6, and the 6 cases under `test_toc_index_page_contract.py`) |
| **Total** | **17** |

## Action items

1. **Foundation session (now):** no changes — this audit is the deliverable
   for Charter A8 at foundation-session level.
2. **Post-A3 (chunker UIR rewire):** verify `test_hybrid_chunker_dense_page_router.py`
   real-Docling-coverage cases (#5, #6) and TOC page-contract probes
   (#11-#16) still pass when run with the requisite env vars. If any fail,
   they are Phase A regression blockers per Charter §8.2 "Phase A merge."
3. **Post-A5 (corpus rebuild):** run `RUN_HARRY_ACCEPTANCE=1` against the
   v3.0.0 Harry Potter output and verify drop-cap promotion behavior
   matches v2.16 or is documented in
   [`PHASE_A_INTENTIONAL_DELTAS.md`](PHASE_A_INTENTIONAL_DELTAS.md).
4. **Deletion candidates** (Phase A cleanup, not blocking):
   - `test_retrieval_regression_v2_10.py:99` — v2.10 llava lane dropped
   - `test_retrieval_regression_v2_11.py:88` — dashscope rollback collection dropped
   - `test_classifier_firearms_route.py:185` — v2.9 Phase 5 closed
   - `test_v29_image_enrichment_acceptance.py:51` — v2.9 Phase 5b closed
   Get user sign-off before deleting any of these; some may still serve
   as historical regression baselines.
