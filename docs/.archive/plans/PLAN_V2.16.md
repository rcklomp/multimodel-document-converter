# Plan: v2.16 — Convergence Cycle (Final v2.X Tag)

**Status:** Ready to execute. Audit cleared 2026-05-25 (v2.15 §9 stopping rule fired at Round 8; 8 external rounds + 1 self-audit, all dispositions in `docs/archive/plans/PLAN_V2.16_0.10.md`). User answered the one blocking question (§8a Q1) on 2026-05-25: **all validation pass-rate thresholds tighten to ≥85% uniformly**. All proposed defaults stand. Cycle opens on next commit.

**Predecessor:** `docs/archive/plans/PLAN_V2.15.md` — CLOSED 2026-05-24 under Option F. v2.15.0 tag pushed to origin + GitHub at commit `fff67d9`.

**Successor:** **v2.17 exists only as a safety valve** for unexpected issues. Four sharp triggers (§7); default outside the trigger set is "fold into v2.16, no defer." Post-v2.17 (if it ever fires): bug fixes only.

**Owner:** ingestion + retrieval + LLM-integration pipeline.

**Provenance:** v0.10 of this plan (130KB of audit-round archaeology + 70+ disposed findings across Rounds 0-8) preserved at `docs/archive/plans/PLAN_V2.16_0.10.md` for historical reference. This file is the execution plan: present-tense, no audit citations, no open questions.

---

## 1. Goal

MM-Converter-V2 ships as **feature-complete** at v2.16.0. Every open item from the v2.11 → v2.15 carry-forward history gets a binary disposition this cycle (SHIP, KILL, or OUT-OF-SCOPE-for-v3.0). Post-tag, only bug fixes (v2.16.x); new features require v3.0 re-charter.

---

## 2. Definition of Done

MM-Converter-V2 is feature-complete when ALL of the following hold:

1. **Production retrieval is stable.** Hybrid retrieval (omlx Qwen3-Embedding-8B-mxfp8 dense + BM25 sparse + RRF fusion + ModernBERT rerank) with the HyDE knob retained as switched-off dead-lever infra. `src/mmrag_v2/retrieval/pipeline.py` modified only for Phase 3 (`partial_code` adjacency fetch) and Phase 5 (dynamic top-k, if the Phase 1 pre-flight gate fires; default-on when shipped, not in production code at all when KILL'd).
2. **Strict-gate corpus state.** 34/34 PASS minimum (current baseline since v2.10), extended to whatever count Phase 0 corpus expansion produces. No regression.
3. **Personal validation queries.** Every documented-limitation class has a curated 10-20 query fixture; **per-class pass rate ≥85%** on the v2.16-shipped retrieval stack (uniform across HIGH + MED). **Exception aligned with Item 4 (b)/(c)**: a class whose Phase 6 outcome is (b) "partial fix + documented residual" or (c) "no fix; full -12pp gap documented as embedder limit" may ship below 85% — the residual gap recorded in DECISIONS.md is the authoritative ceiling for that class, and the per-class pass rate in this case is whatever the post-disposition stack delivers. The DoD check for such classes is: "fixture exists + pass rate measured + residual documented in DECISIONS.md," not the absolute ≥85% threshold. All other classes (HIGH + MED without an Item 4 (b)/(c) disposition) must clear ≥85%.
4. **omlx -12pp deficit dispositioned via Phase 2 verdict + Phase 6 outcome.** Three permanent outcomes, exactly one of which must be recorded in DECISIONS.md:
   - **(a) Full fix**: Phase 2 verdict positive + Phase 6 pre-flight ≥3pp lift + Phase 6 ships AND closes the full -12pp gap → DECISIONS.md records the closure ("Phase 6 closed the -12pp deficit; class now at parity").
   - **(b) Partial fix + documented residual**: Phase 2 verdict positive + Phase 6 pre-flight ≥3pp lift + Phase 6 ships with measurable but incomplete lift → DECISIONS.md records both the achieved lift and the residual gap as accepted limit (e.g., "Phase 6 closed 5pp; residual -7pp accepted as documented embedder limit on these classes — further closure is v3.0").
   - **(c) No fix**: Phase 2 verdict negative/multi-factor OR Phase 6 pre-flight <3pp lift → Phase 6 KILLs; DECISIONS.md records the full -12pp gap as documented embedder limit.

   "Dispositioned" not "fixed" — the convergence-cycle frame permits any residual gap as a valid SHIP outcome PROVIDED it is explicitly documented as accepted. The gate verifies disposition + documentation, not magnitude of fix.
5. **Every documented-limitation class has a permanent disposition.** SHIP'd a fix (Phase 3/4 close the file), KILL'd, or marked permanent-limitation-with-rationale. No class in a "we'll see" state.
6. **Zero soft-state carry-forwards.** `docs/PROJECT_STATUS.md` Other Carry-Forwards list either empty or holds only items with explicit v3.0-class trigger conditions.
7. **README declares v2.16.0 feature-complete** for the project's stated use case (solo-dev personal PDF→JSONL multimodal-RAG pipeline against a curated multilingual corpus including technical manuals, magazines, scanned books, forms, code-dense books).
8. **Post-v2.16.0**: only bug fixes (v2.16.x patches). New features = re-charter as v3.0. **v2.16.x vs v3.0 boundary**: a config-knob or threshold change is a v2.16.x patch ONLY if it fixes a demonstrable regression from v2.16.0 behavior on the v2.16.0 corpus. Changes motivated by new documents, new use cases, or "the threshold could be better tuned for X" are v3.0. Full re-charter conditions in §8.

---

## 3. SHIP phases

Five unconditional phases (0-4), two conditional (5, 6), one default-KILL (7), one close-out (N).

### Phase 0 — Corpus expansion + classification

**Goal:** ingest the 7 user-added PDFs in `data/raw/` into production indexes, classify them programmatically, surface near-boundary and probe-detected misclassifications, and rename the canonical-docs list. Gates Phase 2 (deficit-class diagnostic scope) and Phase 4 (form-class generality scope).

**Method:**

1. **Per-doc ingestion.** For each `data/raw/<doc>.pdf`:
   ```
   mmrag-v2 process data/raw/<doc>.pdf \
       --output-dir output/<doc_basename> \
       --batch-size 10
   ```
   Output lands at `output/<doc_basename>/ingestion.jsonl` — the canonical location all downstream consumers expect. No `--profile-override`; `ProfileClassifier` auto-routes per doc content. Source PDFs stay in `data/raw/`.

2. **Threshold pre-validation against known 34-doc corpus.** Before classifying the 7 unknowns, run the FULL classification stack (step 3 rules + step 4 probes) against the existing canonical corpus. Sanity targets:
   - `Fluent_Python`, `Python_Cookbook`, `Python_Distilled` must classify as code-dense via step 3 rules.
   - `HARRY`, `CarOK_voorraadtelling` must NOT classify as code-dense via step 3 rules.
   - `CarOK_voorraadtelling` must classify as form-class — **via either step 3 rules OR step 4 Probe A** (CarOK's current production extraction is pre-Phase-4 scanned/Tesseract with `table_chunks ≈ 0`, so the step-3 form-class ratio fails by construction; Probe A's `scanned + image_chunks>0 + table_chunks==0` re-extract path is the legitimate satisfaction route. Recording probe-based satisfaction in the sanity result is sufficient).

   If any expectation fails via BOTH paths, pause and recalibrate the threshold before classifying the unknowns. **Abort condition:** if no threshold setting satisfies all sanity targets via either step 3 OR step 4 simultaneously, classification rules are fundamentally broken (likely a dependency change). Halt; treat as a §7 trigger #2 condition. Output: one-line confirmation in the inventory report (state which path satisfied each target), or threshold-adjustment rationale.

3. **Programmatic classification rules.** Read each doc's `ingestion.jsonl` (never the source PDF — preserves bias discipline) and compute per-doc class assignments:
   - **Code-dense**: `code_chunks / total_text_chunks ≥ 0.30` OR (`profile == "technical_manual"` AND `diagnostic.has_code_evidence == True`).
   - **Form-class**: `table_chunks / total_chunks ≥ 0.40` AND `unique_table_template_patterns ≥ 3` (template pattern = `(column_count, row_count)` tuple).
   - **Minority-language**: `mean_non_ascii_ratio > 0.03` across ≥10 sampled text chunks OR ≥30% of sampled chunks return `"minority_language"` from `classify_intent()` in `src/mmrag_v2/retrieval/intent.py` (hit-rate derived per-doc by calling `classify_intent` on each sampled chunk-text).
   - **Other** → `general` (no special Phase 2/3/4 handling).

   Classes are not mutually exclusive; a doc can carry multiple flags.

4. **Probes for suspected misclassifications.** All three probes operate on pipeline outputs only (no source-PDF inspection).

   **Probe A — form-class misclassification** (image-based tables OCR'd to nothing):
   ```
   if profile in {"scanned", "scanned_degraded"}
      AND image_chunks > 0
      AND table_chunks == 0:
       run: mmrag-v2 process <doc> --force-table-vlm --pages 1-3 \
                    --output-dir output/_probe_<basename>
       if probe produces table chunks:
           override classification → form-class
           record rationale in inventory report
   ```

   **Probe B — minority-language with OCR-stripped diacritics** (signal-only, no auto-reclassification):
   ```
   if intent_classifier fires on ≥1 chunk
      AND total hit-rate < 0.30
      AND mean_non_ascii_ratio < 0.03:
       flag BORDERLINE_MINORITY_LANGUAGE in inventory report
   ```

   **Probe C — near-boundary classification flag** (signal-only):
   ```
   for each doc:
       if 0.25 <= code_chunks_ratio < 0.30: flag NEAR_BOUNDARY_CODE_DENSE
       if 0.35 <= table_chunks_ratio < 0.40: flag NEAR_BOUNDARY_FORM_CLASS
       if 0.025 <= mean_non_ascii_ratio <= 0.035: flag NEAR_BOUNDARY_MINORITY_LANGUAGE
   ```

   User reviews probe-flagged docs at Phase 0 acceptance and decides: accept the classification, re-extract with a profile override, or recalibrate the threshold (which re-runs step 2).

5. **Inventory report.** Write `docs/CORPUS_EXPANSION_2026-05-24_v2.16_p0.md` listing per doc: basename, auto-routed profile, chunk count (text/table/image/code), computed class flags, class-determining metrics (the actual numbers), probe flags, extraction warnings.

6. **Append to production indexes + rename canonical list.** Order matters — BM25 rebuild reads the canonical list at import time, so the rename + append MUST land before the rebuild fires. Sequence:

   **(6.1) Qdrant pre-mutation snapshot.** Capture `mmrag_v2_8__qwen3_local` via `qdrant snapshot create`. Stash snapshot ID + timestamp in the Phase 0 commit message. (Production indexes live outside git; snapshot is the dense-index revert anchor — see revert procedure at 6.5.)

   **(6.2) Atomic `CANONICAL_34` → `CANONICAL_DOCS` rename + append new basenames** across all 5 consumer sites. The constant name describes its semantic role (the canonical docs list), not its cardinality. Partial rename across sites would break the `_rebuild_mod.CANONICAL_34` imports in BM25 scripts — all five sites rename together in one commit:
     1. `scripts/synthetic_soak.py` (line 124): rename + append new basenames.
     2. `scripts/rebuild_mmrag_v2_8_for_rc1.py` (line 61): rename + append (canonical source imported by `build_bm25_index.py` + `ingest_bm25_sparse.py`).
     3. `tests/test_rebuild_resume.py` (lines 73-81): rename + update hard-coded length assertion to the new count.
     4. `scripts/build_bm25_index.py` (line 45): rename `CANONICAL_34 = _rebuild_mod.CANONICAL_34` → `CANONICAL_DOCS = _rebuild_mod.CANONICAL_DOCS` and all downstream uses.
     5. `scripts/ingest_bm25_sparse.py` (line 45): same rename + all downstream uses.

   Commit this rename + canonical-list update BEFORE 6.3. This commit is the git revert anchor.

   **(6.3) Append new chunks to dense.** Run existing ingest script against `mmrag_v2_8__qwen3_local`. Uses the renamed `CANONICAL_DOCS` to determine what to append.

   **(6.4) Rebuild BM25 sparse.** Run `scripts/build_bm25_index.py` then `scripts/ingest_bm25_sparse.py` against `mmrag_v2_8__bm25_sparse`. Both now import the renamed `CANONICAL_DOCS` with all 41 entries → BM25 parallel-maps the post-append dense.

   **(6.5) Anti-drift bridge test.** Add `tests/test_canonical_docs_consistency.py` asserting `set(synthetic_soak.CANONICAL_DOCS) == set(rebuild_mod.CANONICAL_DOCS)`. ~10 lines. Run as part of the Phase 0 acceptance test suite.

   **(6.6) Complete revert procedure if any of 6.2–6.4 fails** (dense + sparse must end up in parallel-mapping consistency for RRF fusion to work — dense snapshot alone is insufficient):
     1. Restore dense via `qdrant snapshot restore <snapshot_id>` → dense back to pre-mutation 34 docs.
     2. `git revert <canonical_docs_commit>` (from 6.2) → list back to 34 entries with `CANONICAL_34` name restored across all 5 sites.
     3. Re-run BM25 sparse rebuild (`scripts/build_bm25_index.py` + `scripts/ingest_bm25_sparse.py`) against the restored 34-doc list → sparse parallel-maps the restored dense.
     4. Verify: anti-drift bridge test passes against the restored state.

7. **Class composition feeds Phase 2/3/4 scoping:**
   - ≥3 minority-language docs → Phase 2 class-level diagnostic at scale; <3 → Phase 2 verdict labeled "inconclusive on class-level vs doc-specific."
   - ≥2 form-class docs → Phase 4 generality validates multi-doc; <2 → CarOK-only IS the final form of the Phase 4 test.
   - ≥2 code-dense docs → Phase 3 generality multi-doc; <2 → Fluent_Python-only IS the final Phase 3 test.

**Acceptance:**
- All 7 docs ingest cleanly. Any extraction failure is a Phase 0 FAIL pending explicit user resolution: (a) drop from v2.16 scope (user signs off + doc moves out of `data/raw/` + Phase 0 re-runs) or (b) fix extraction + re-ingest. Silent partial ingestion is forbidden.
- Threshold pre-validation passes against the 34-doc corpus OR the abort condition fires.
- Inventory report exists at the spec'd path with all per-doc fields populated.
- Probe-flagged docs have explicit user acceptance recorded in the inventory report.
- All 5 CANONICAL_DOCS sites renamed atomically; anti-drift bridge test passes.
- v2.10 strict-gate 34/34 PASS extended to the new count, still PASS.
- Qdrant snapshot ID recorded in the Phase 0 commit message.

**Cost:** ~1 day. $0 cloud spend.

**Risk + fallback:** if Probe A misclassifies (rare), the inventory report makes the reclassification visible at acceptance — user can revert. Qdrant snapshot is the revert path for index-level corruption.

---

### Phase 1 — Decision-mechanism overlay + validation fixtures

**Goal:** replace v2.15's telemetry-as-decision-mechanism with a curated personal-importance overlay (telemetry stays as second-class signal). Provides the validation-query mechanism Phases 3/4/5 depend on for acceptance measurement.

**Method:**

1. **Extend `src/mmrag_v2/retrieval/documented_limitations.py`:**
   - Add `personal_importance: Literal["HIGH", "MED", "LOW"]` field per registered class entry.
   - Existing CarOK + Fluent_Python entries get `personal_importance: "HIGH"`.
   - Default factory for new entries: `personal_importance: "MED"`.

2. **Update `scripts/analyze_doc_class_telemetry.py`** to incorporate personal_importance:
   - `HIGH` → forces Option A recommendation regardless of telemetry hit-rate (overrides defer / middle-band).
   - `MED` → existing telemetry rules apply (per DECISIONS.md "v2.15 Documented-Limitation Telemetry Threshold").
   - `LOW` → reduces `NEW_CLASS_GRACE_CYCLES` from 2 to 1 for auto-closure; still requires 0 issues + no defect-tag.
   - Report renders both signals + which rule fired.

3. **Create `tests/fixtures/personal_validation_queries/`** with one JSON file per HIGH or MED class. Schema:
   ```json
   {
     "class": "CarOK_voorraadtelling",
     "personal_importance": "HIGH",
     "target_pass_rate": 0.85,
     "queries": [
       {
         "id": "Q01_part_count_4567",
         "query_text": "What's the total count for part number 4567?",
         "expected": {
           "doc_contains_answer": true,
           "top_5_gold_doc": true,
           "format_constraint": "table_value",
           "gold_chunk_ids": ["CarOK_voorraadtelling__p6_t3_r12"],
           "expected_anchor_regexes": ["\\b4567\\b.*\\b\\d+\\b"]
         },
         "notes": "Page 6, column 3 of inventory table"
       }
     ]
   }
   ```

   **Per-query answer-correctness**: each query MUST provide `expected_anchor_regexes` (mandatory; the load-bearing signal). `gold_chunk_ids` is OPTIONAL and serves as a sanity-comparison signal in the validation report only — it does NOT gate PASS/FAIL. The regex is load-bearing because Phase 3 adjacency merges produce chunks with new content but the seed chunk's id (so chunk-id matching would systematically fail on merged-but-correct results), and because harvesting gold_chunk_ids from baseline outputs alone would bias the test toward what baseline already retrieves (queries baseline fails would have no gold or wrong-chunk gold). The regex is chunk-shape-independent: if the answer content appears in top-1 (whether original chunk, merged chunk, or Phase-4-dedupped chunk), the query PASSes.

   **Authoring rules**: 10-20 queries per class, manually written. ≥3 queries per class must test the class-specific failure mode (CarOK: multi-row table values; Fluent_Python: code-with-imports-and-usage spans; minority-language docs: cross-lingual semantic queries). `expected_anchor_regexes` is authored by inspecting the source content (or the chunk corpus directly) to identify text patterns that **uniquely match the correct answer** AND **reject wrong-chunk content from the same doc** (tight enough that a top-1 containing wrong-chunk prose from the right doc would not match). Use multiple regex patterns when an answer can appear in semantically equivalent forms. `gold_chunk_ids`, when authored, comes from a chunk-corpus search (which may use the baseline as one discovery tool but is not restricted to baseline top-25) — record the seed chunk that holds the answer pre-merge. **`target_pass_rate: 0.85` uniformly** (the DoD threshold).

4. **Create `scripts/run_personal_validation.py`:**
   - Reads all fixtures from `tests/fixtures/personal_validation_queries/`.
   - For each query: runs `retrieve_hybrid_reranked` with production defaults (no `--use-hyde` unless the validation target is HyDE).
   - **Per-query PASS rule** (ALL must hold):
     - (a) `top_5_gold_doc` — gold doc_id appears in retrieved top-5.
     - (b) `format_constraint` — table_value → top-1 modality is `table`; runnable_code → top-1 content parses via `ast.parse`.
     - (c) **Answer-correctness (MANDATORY)**: top-1 content (post Phase 3 adjacency merge if applicable) matches at least one `expected_anchor_regexes` pattern. The regex is the chunk-shape-independent answer-correctness signal — it does not care about chunk-id changes from Phase 3 merges or Phase 4 dedup. `gold_chunk_ids`, when provided in the fixture, is RECORDED in the validation report (helps diagnose "regex passed on a chunk the baseline didn't retrieve" cases — useful for cross-cycle comparison) but does NOT gate PASS.
   - Emits `docs/VALIDATION_REPORT_<YYYY-MM-DD>.md` with per-class pass rate vs target, per-query PASS/FAIL detail.
   - Returns nonzero exit code if any class drops below `target_pass_rate` (gate-able from CI / cycle-open).
   - $0 cloud spend (retrieval only). ~5-10 min wall-clock per full validation run.

5. **Update `docs/CYCLE_OPEN_CHECKLIST.md`**: add new line item between current §5 (UIR check — which Phase N removes at close-out) and §6 (cycle_slip review):
   > "Review/update `personal_importance` flags on documented-limitation classes per current personal workflow needs (2-minute review)."

6. **Establish Phase 1 BASELINE.** Run `run_personal_validation.py` once against the CURRENT v2.15.0 retrieval stack BEFORE any Phase 3/4/5 work. Capture as `docs/VALIDATION_REPORT_2026-05-24_v2.15.0_baseline.md`. This is the comparison anchor for Phase 3/4/5 acceptance bars (each phase's acceptance is delta vs this baseline, not absolute), AND the input to the Phase 5 pre-flight gate.

**Initial class assignments:**
- `CarOK_voorraadtelling`: HIGH.
- `Fluent_Python`: HIGH.
- Any class emerging from Phase 0 inventory: MED.

**Acceptance:**
- `personal_importance` field on every documented-limitation class entry.
- Analyzer report shows both signals + transparent override logic.
- Validation query fixtures exist for all HIGH + MED classes (≥10 queries each).
- `run_personal_validation.py` runs end-to-end + emits the dated report.
- Baseline run captured.
- DECISIONS.md "v2.16 Decision-Mechanism Overlay" entry recorded.
- ≥5 new unit tests in `tests/test_personal_validation.py` covering the runner + fixture-schema validation.

**Cost:** 0.5 day implementation + 0.5 day/class authoring (N=2 baseline → 1.5 days; N=5 if Phase 0 surfaces 3 MED classes → 3 days). $0 cloud spend.

**Risk + fallback:** if fixture authoring runs over, the cycle has no fallback — fixtures are load-bearing for Phases 3/4/5. Mitigation: prioritize HIGH classes; MED can ship with smaller fixtures (still ≥10 queries) if needed.

---

### Phase 2 — omlx -12pp deficit diagnostic

**Goal:** answer the v2.13 P1 open question definitively. Drives Phase 6 conditional ship.

**Method:**

1. **Hypothesis tests on the original 5 deficit docs** (ATZ_Elektronik_German, Python_Cookbook, IRJET_Modeling_of_Solar_PV, Hybrid_electric_vehicles, Greenhouse_Design):
   - H1 (truncation): retrieve at top-25; measure how many gold-chunk answers fall in ranks 6-25 vs top-5.
   - H2 (OOV/vocabulary): tokenize query + gold-chunk text with the omlx tokenizer; count out-of-vocab token overlap delta vs in-vocab queries.
   - H3 (cross-lingual): for non-English docs, measure recall delta when query and chunk are in different languages.
   - H4 (chunk length distribution): correlate chunk length with retrieval rank position for gold chunks.

2. **Class-level vs doc-specific test:**
   - If Phase 0 inventory identifies ≥3 docs in the same class as deficit docs (e.g., German tech, code-dense), re-run the omlx-vs-Dashscope shootout on those new docs. Deficit replicates → **class-level**. Doesn't replicate → original 5 were **doc-specific** quirks.
   - If Phase 0 has <3 same-class docs, run at whatever n is available; verdict explicitly labeled "inconclusive on class-level vs doc-specific."

3. **Phase 6 analytical pre-flight** (gating Phase 6 build cost). Only fires if Phase 2 verdict is H2 or H3 class-level:
   - Take 5-10 Phase 1 validation queries from the affected deficit class.
   - Manually author the query-rewrite variants (no production code yet).
   - Run them through `retrieve_hybrid_reranked` and RRF-fuse the candidate pools analytically.
   - Check R@1 / Hit@5 delta on the `gold_chunk_ids` fixtures.
   - **If ≥3pp R@1 lift on this 5-10-query sample → Phase 6 triggers; build production code.**
   - **If <3pp lift → Phase 6 KILLs before implementation** (DECISIONS.md entry: "v2.16 Phase 2 H2/H3 verdict positive but Phase 6 pre-flight insufficient lift; query rewriting closed as 2nd dead lever without build cost").

4. **Output:** `docs/DIAGNOSTIC_<date>_v2.16_p2_omlx_deficit_root_cause.md` with the verdict + binary outcome for Phase 6.

**Acceptance:**
- All 4 hypotheses tested with measurements on the original 5 deficit docs.
- Class-level vs doc-specific verdict (or explicit "inconclusive" label).
- Binary outcome for Phase 6: **YES** (H2/H3 class-level + pre-flight ≥3pp lift) → Phase 6 ships. **NO** (any leg fails: verdict not H2/H3+class-level, OR pre-flight <3pp lift, OR multi-factor verdict) → Phase 6 KILLs.
- Report shipped in `docs/`.

**Cost:** 1 day (incl. pre-flight if it fires). ~$0.50 cloud spend.

**Risk + fallback:** verdict may be "multiple factors interact." Per convergence discipline, multi-factor verdicts route to KILL (not deferral) — anything not crisply positive on the compound trigger kills Phase 6.

---

### Phase 3 — `partial_code`-aware retrieval (adjacency fetch)

**Goal:** the elegant fix for Fluent_Python (and any future cross-page code defect). Use the existing `partial_code=True` schema flag from v2.14 P6 to deterministically stitch adjacent chunks at retrieval time. Sidesteps Docling HybridChunker configuration entirely (Item #9 KILLs conditional on this phase passing).

**Method:**

1. **Schema verification spike** (30 min). Read `src/mmrag_v2/schema/ingestion_schema.py` + sample production `ingestion.jsonl` to verify `chunk_index` ordering semantics. Possible outcomes:
   - (a) `chunk_index` is monotonic in source-flow order across all modalities → use simple `chunk_index + 1` lookup.
   - (b) `chunk_index` is monotonic per-modality → use `(source_file, page_number, chunk_index)` tuple sort with text/code-modality filter on "next" lookup.
   - (c) `chunk_index` reflects emission order only → use `(source_file, page_number, char_offset_or_position_within_page)` as the canonical ordering key.

   Output documented in the Phase 3 commit message.

2. **In `retrieve_hybrid_reranked` (after rerank stage):**
   ```
   for each result chunk in top-N output:
       if chunk.payload.get("partial_code") is True:
           prev = lookup(direction=backward, filter={source_file, partial_code=True, modality in {text,code}})
           next = lookup(direction=forward,  filter={source_file, partial_code=True, modality in {text,code}})
           # Bounded: max 1 chunk backward + 1 chunk forward = up to 3-chunk merge
           if prev or next:
               merged = concat(prev?, current, next?)
               replace current with merged
               preserve rerank_score
               metadata.partial_code_resolved = True
               metadata.adjacency_source = [prev_id?, current_id, next_id?]
           else:
               metadata.partial_code_resolved = False  # sole partial_code chunk — rare
   ```
   - Both directions filtered to text/code modalities (skip tables, images).
   - Stop at the first non-`partial_code` neighbor in each direction (that's the boundary).
   - Larger windows risk pulling unrelated code; if a code unit truly spans >3 chunks, the v2.17 safety valve (Item #9 reopens) picks it up.

3. **Bridge tests in `tests/test_retrieval_pipeline.py`** (5 edge cases):
   - Leading chunk of partial_code sequence: prev is non-partial_code (boundary), next is partial_code → merge is `current + next`.
   - Middle chunk: both prev and next are partial_code → merge is `prev + current + next`.
   - Trailing chunk: prev is partial_code, next is non-partial_code → merge is `prev + current`.
   - Sole partial_code chunk: no eligible neighbor in either direction → pass-through + `partial_code_resolved=False`.
   - Non-text neighbor (table/image): skipped, treated as boundary.

   Plus: assert original `rerank_score` preserved (no inflation); assert merged-chunk content concatenation order is prev → current → next.

4. **Validation via Phase 1's Fluent_Python fixture:**
   - Run baseline (current `retrieve_hybrid_reranked` from v2.15.0) on Fluent_Python validation queries — capture per-query PASS/FAIL via `ast.parse` on top-1.
   - Run patched (with adjacency fetch) on same queries.
   - Compare; aggregate against acceptance threshold.

5. **No-regression check:** run `scripts/retrieval_regression_v2_14.py` end-to-end. 20/20 PASS unchanged. (Adjacency fetch only triggers on `partial_code=True`, which fingerprint queries don't currently trigger; this is a clean pass-through verification.)

**Acceptance:**
- Fluent_Python validation queries: **≥85% pass rate** (per DoD item 3) — top-1 returns a syntactically complete code block per `ast.parse` verification.
- **Generalization** (Phase-0-dependent): if Phase 0 inventory contains ≥2 code-dense docs, each must show ≥85% pass rate. If <2, Fluent_Python-only IS the final form; gap documented in the Phase 3 outcome entry.
- v2.14 retrieval fingerprint: 20/20 PASS unchanged.
- ≥4 new bridge tests in `tests/test_retrieval_pipeline.py`.

**Cost:** 1-2 days. $0 cloud spend.

**Risk + fallback:** if Fluent_Python pass rate falls short of 85%, route to v2.17 safety valve (Item #9 B1 Docling config hunt reopens). If pass rate is between baseline and 85% but adjacency fetch demonstrably helps individual queries, accept partial improvement + open v2.17 — but the Phase 3 acceptance bar is not lowered. **Rollback procedure**: `git revert <phase3_commit_sha>` (independent commit per §5 serial-order rule).

---

### Phase 4 — VLM-Table Dedup (IoU>85% suppression)

**Goal:** resurrect v2.14 P1's `--force-table-vlm` with the missing dedup piece. Surgical, $0 incremental cost, no new dependencies.

**Method:**

1. **`bbox_iou` utility** in new module `src/mmrag_v2/utils/bbox.py`:
   ```python
   def bbox_iou(a: BBox, b: BBox) -> float:
       """Standard Intersection-over-Union for normalized [0,1000] integer
       bbox tuples (AGENT-SPATIAL-20 invariant compliant). Returns 0.0 if
       either bbox is empty/invalid; returns float in [0.0, 1.0]."""
   ```
   With 5 unit tests covering: identical bboxes (IoU=1.0), disjoint (0.0), 50% overlap, contained, degenerate (zero-area).

2. **Dedup logic in chunk-emission flow** (`processor.py` — verify actual `ElementProcessor` location during implementation):
   ```
   for each page:
       group chunks by page_number
       vlm_tables = [c for c in chunks if c.extraction_method in {"vlm_table", "vlm_table_markdown"}]
       text_chunks = [c for c in chunks if c.modality == "text"]
       for text in text_chunks:
           for vt in vlm_tables:
               if bbox_iou(text.bbox, vt.bbox) > dedup_vlm_table_iou_threshold:
                   text._suppress = True
                   break
       filter out _suppress=True chunks before emission
       log suppression count per page to extraction stats
   ```

3. **Configuration knob** added to `PdfConversionPlan`:
   ```python
   dedup_vlm_table_iou_threshold: float = 0.85
   ```
   Flows through `PdfConversionPlan` → `DoclingPdfAdapter` → `ElementProcessor` per existing pattern. **Bridge test** in `tests/test_pdf_conversion_plan.py` proves the knob threads through.

4. **Bridge tests in `tests/test_processor.py`:**
   - IoU=0.95 → text chunk suppressed.
   - IoU=0.50 → text chunk NOT suppressed.
   - IoU=0.0 (disjoint) → text chunk NOT suppressed (negative case).
   - Page with no VLM tables → all text chunks pass through (no false positives).
   - Page with 2 VLM tables + 1 text chunk overlapping each at 0.90 → suppressed (any one overlap above threshold fires).

5. **Re-extract CarOK with `--force-table-vlm` + new dedup:**
   - Per-page chunk count comparison vs v2.13 baseline; expect VLM-table count = original prose count for pages that had both.
   - Sanity check: top-1 retrieval for a known multi-row CarOK query (from Phase 1 fixture) returns the VLM table, not the flat-prose chunk.

6. **Generality across Phase 0 form-class docs** (Phase-0-dependent — see acceptance below).

**Acceptance:**
- **CarOK Format axis ≥85%** measured on Dashscope qwen-max judge (apples-to-apples with v2.13 P1 baseline of 71.9% and v2.14 P1's regressed 45%). Honest-evaluation note: the documented CarOK judge-calibration limitation on form-class content means the 85% target may be hitting a judge ceiling; if Phase 4 lands at 78-84% Dashscope but Phase 1 CarOK validation queries (retrieval-only PASS/FAIL, judge-independent) clear ≥85%, that's a valid SHIP per the second bullet. Do NOT switch judges to clear the bar.
- **CarOK Phase 1 validation queries: ≥85% pass rate** (authoritative; retrieval-fixture-based, judge-independent).
- **Generality (Phase-0-dependent)**: if Phase 0 identifies ≥2 form-class docs, re-extract each with `--force-table-vlm` + new dedup; validate clean output via two **programmatic** gates per doc (no human inspection):
  1. `suppression_count_per_doc > 0` — proves dedup actually fired somewhere.
  2. **No two chunks on the same page have Jaccard token-overlap ≥ 0.5** — catches the below-threshold-IoU semantic-duplicate case where IoU ∈ [0.5, 0.85] still produces near-duplicate content the threshold lets through. The canonical instance: VLM extracts a table as markdown (`| Part | Count |\n| --- | --- |\n| 4567 | 12 |`) while Docling renders the same source region as prose (`Part Count 4567 12`) — high token-overlap, NOT byte-identical, so byte-equality would miss it. Jaccard on whitespace-tokenized lowercased content cleanly catches this: shared distinctive tokens (column headers + row values) drive Jaccard above 0.5; unrelated chunks that share only common stop-tokens stay well below. ~15 lines in `scripts/eval_phase4_generality.py` (tokenize → set → intersection/union per chunk pair per page).

  If <2 form-class docs, CarOK-only IS the final form of the test. No future-cycle watch.
- v2.10 strict-gate 34/34 PASS (extended for Phase 0 additions) unchanged.
- Bridge tests pass.

**Cost:** 2-3 days. ~$0.50 cloud spend (validation mini-soak).

**Risk + fallback:** IoU threshold (0.85) is a judgment call. Tunable during v2.16 execution (before tag). Post-tag tuning is v2.16.x ONLY if fixing a demonstrable v2.16.0-corpus regression; tuning for better performance is v3.0 (per §8).

---

### Phase 5 [CONDITIONAL] — Dynamic top-k from rerank logit drop-off

**Goal:** retrieval-time optimization that reduces LLM context size + hallucination risk on queries with sharp rerank drop-off — IF the corpus actually exhibits such drop-offs.

**Disposition gate** (binary, fires before any production code is written). Apply the proposed dynamic-top-k logic ANALYTICALLY to Phase 1's BASELINE rerank outputs (Phase 1 step 6 = v2.15.0 retrieval state):
- Compute `would_truncate` for each query under default params (`drop_off_threshold=2.5`, `min_absolute_gap=0.05`).
- Compute PASS-retention under simulated truncation using Phase 1's gold-anchor fixtures.

**Binary outcome:**
- **SHIP default-on** if ALL THREE hold:
  - ≥20% of Phase 1 validation queries `would_truncate` (meaningful context-reduction signal exists), AND
  - PASS-retention `≥ 0.97` across the full fixture set (truncation doesn't hurt answer-bearing retrieval), AND
  - No HIGH-class fixture's simulated pass rate falls **more than 2pp below its static baseline** (relative bound vs Phase 1 baseline, NOT against the fixture's authored `target_pass_rate` — `target_pass_rate` is the DoD-item-3 / Phase 1 reporting bar, not a Phase 5 gate condition).

  → Implement the code below; ship with `dynamic_top_k=True` as the default in v2.16.

- **KILL permanently** if ANY leg fails. DECISIONS.md entry: "v2.16 Phase 5 KILL — pre-flight evidence shows dynamic top-k has no measurable upside on the corpus." No opt-in middle ground — opt-in dead code is the failure mode for a feature-frozen product.

**Method (only if SHIP gate fires):**

1. **Drop-off detection** in `src/mmrag_v2/retrieval/pipeline.py` (`retrieve_hybrid_reranked`):
   ```python
   logits = [r.rerank_score for r in reranked]
   if len(logits) < 2:
       return reranked
   gaps = [logits[i] - logits[i+1] for i in range(len(logits)-1)]
   mean_gap = sum(gaps) / len(gaps)
   for i, gap in enumerate(gaps):
       if gap > drop_off_threshold * mean_gap and gap > min_absolute_gap:
           return reranked[: max(min_top_n, i + 1)]
   return reranked
   ```
   - `drop_off_threshold = 2.5` default (sharp).
   - `min_absolute_gap = 0.05` default (prevents truncation on uniformly-tiny logit deltas).
   - Bounded: `min_top_n = 1`, `max_top_n = 5` (existing `top_n_return`).

2. **CLI / API surface** (default-on):
   - New `retrieve_hybrid_reranked` parameter: `dynamic_top_k: bool = True`. Callers can pass `False` for diagnostic comparison.
   - New `synthetic_soak.py` flag: `--no-dynamic-top-k` (diagnostic disable).
   - README "Retrieval options" subsection documenting the parameter with one-sentence description + pointer to the Phase 5 validation report.

3. **Bridge tests** in `tests/test_retrieval_pipeline.py`:
   - Logits `[10.0, 9.5, 9.0, 5.0, 4.9]` → truncate at index 3 (gap 9.0→5.0 = 4.0 > 2.5 × 1.275).
   - Logits `[10.0, 9.5, 9.0, 8.5, 8.0]` → no truncation (flat).
   - Logits `[10.0, 10.0, 10.0, 10.0, 10.0]` → no truncation (identical).
   - Top-N length 1 → return as-is.
   - `dynamic_top_k=False` → bypass logic entirely; legacy behavior preserved.

4. **Validation against Phase 1 fixtures.** Run `run_personal_validation.py` twice — once with `dynamic_top_k=False` (baseline), once with `True`. Compare per-class pass rates + distribution of returned top-N sizes.

**Acceptance (only if SHIP gate fires):**
- Dynamic top-k produces variable top-N on Phase 1 validation queries; report the distribution.
- **PASS-retention bound**: pre-flight gate already verifies `PASS_rate_dynamic / PASS_rate_static ≥ 0.97` aggregate + no HIGH-class fixture's simulated rate more than 2pp below baseline. Acceptance re-verifies these against the IMPLEMENTED code on a fresh Phase 1 run (not just analytical pre-flight).
- ≥3 new bridge tests in `tests/test_retrieval_pipeline.py`.

**Cost:** 1 day (if SHIP gate fires). 0 days if KILL'd. $0 cloud spend.

**Risk + fallback:** **rollback procedure**: `git revert <phase5_commit_sha>` (independent commit per §5 serial-order rule). If KILL'd, no production code, no rollback risk.

---

### Phase 6 [CONDITIONAL] — C1 Query rewriting

**Triggers** (compound, both legs required):
1. Phase 2 verdict identifies vocabulary mismatch (H2) OR cross-lingual degradation (H3) AND class-level pattern (not doc-specific); AND
2. Phase 2's analytical pre-flight shows ≥3pp R@1 lift on the 5-10 manually-rewritten queries from the deficit class.

Either leg failing → **Phase 6 KILLs without implementation**. DECISIONS.md gets the closure entry ("Phase 2 verdict positive but pre-flight insufficient lift; query rewriting closed as 2nd dead lever — HyDE was the 1st").

**Method (only if both triggers fire):**

1. Use local FP8-14B (`RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic` @ `http://10.0.10.239:8000/v1/chat/completions`) to generate 2-3 rewritten queries per input.
2. Run retrieval against each rewrite; RRF-fuse the candidate pools.
3. Rerank fused candidates with ModernBERT.
4. Bridge tests in `tests/test_query_rewriting.py`: ≥3 happy-path + 2 edge cases (empty rewrite, rewrite identical to input).

**Acceptance (only if triggered):**
- ≥3pp R@1 lift on the deficit-class subset of Phase 1 validation queries (matching the pre-flight threshold).
- If the lift doesn't materialize after build, close as 2nd dead lever (HyDE was the 1st); no defer.

**Cost:** 2-3 days + ~$1 validation soak. 0 days if not triggered.

**Risk + fallback:** the pre-flight gate catches "hypothesis right but no measurable lift" before build cost — this is the primary risk mitigation. No production-code rollback needed if KILL'd.

---

### Phase 7 [DEFAULT KILL] — D2 Retrieval-time image re-read

**Disposition:** **KILL at Phase N** (default; §8a Q3 disposition). Phase 1 validation scope is CarOK + Fluent_Python — neither image-heavy. No fixture will surface ≥3 image-content gaps. Phase 7's structural prerequisite is absent.

**Reopen condition (user-only, before Phase 1 fixture authoring begins):** if user explicitly opts in via promoting an image-heavy class (e.g., `PCWorld_July_2025`, `Combat_Aircraft_August_2025`) to documented-limitation HIGH/MED status with 10-20 validation queries that specifically test image-content questions, Phase 7 becomes a real CONDITIONAL SHIP gated on the new fixture's failure pattern.

**If reopened, method:**
- `--enable-vision-reread` flag in retrieval/generation pipeline.
- If `IMAGE` or `TABLE` chunk in final top-5: load original image crop, pass to local VLM (NuMarkdown-8B) with user's query, use VLM output as expanded chunk context.
- Acceptance: image-gap validation queries ≥85% pass rate.
- Cost: 3-5 days + minimal local VLM compute.

**Default-path acceptance:** Phase 7 closes at Phase N with a DECISIONS.md entry citing absence of opt-in scope.

---

### Phase N — Cycle close-out + v2.16.0 tag (FINAL v2.X TAG)

**Method:**

1. Engine version bump `2.15.0` → `2.16.0`:
   - `src/mmrag_v2/version.py`
   - `pyproject.toml`
   - `tests/test_v2_10_release_baseline.py` version pin.

2. **v2.16 retrieval-regression fingerprint** — re-capture IF production retrieval shape changed:
   - Phase 3 adjacency fetch triggers only on `partial_code=True` (existing fingerprint queries don't trigger it; pass-through expected).
   - Phase 5 dynamic top-k: if shipped (default-on), the fingerprint MUST be re-captured to reflect the new production shape. If KILL'd, no fingerprint change.

3. **AFTER snapshot** at `docs/QUALITY_SNAPSHOT_<date>_v2.16_after.md` with prominent "FEATURE-COMPLETE FOR v2.X PROJECT" banner.

4. **DECISIONS.md entries** (all required):
   - "v2.16 Decision-Mechanism Overlay" (Phase 1).
   - "v2.16 Phase 2 omlx Deficit Diagnostic Verdict" with binary outcome for Phase 6.
   - "v2.16 Phase 3 partial_code Adjacency Fetch — SHIPPED".
   - "v2.16 Phase 4 VLM-Table Dedup — SHIPPED".
   - "v2.16 Phase 5 Dynamic Top-K — SHIPPED DEFAULT-ON" OR "v2.16 Phase 5 Dynamic Top-K — KILLed by pre-flight" (binary).
   - "v2.16 Phase 6 Query Rewriting" — one entry (SHIPPED OR CLOSED-as-2nd-dead-lever / KILLed).
   - "v2.16 Phase 7 Image Re-read" — one entry (SHIPPED if user opted in OR KILLed; no dead-lever framing).
   - **"v2.16 Carry-Forward Closures"** combined entry covering Items #9, #10, #12, #13, #14, #15, #21, #22 — one paragraph each per the §4 rationales.
   - **"v2.16 v3.0-Class Items Declared Out-of-Scope for v2.X"** entry covering Item #11 (ColPali / VisRAG).

5. **`docs/PROJECT_STATUS.md` Other Carry-Forwards list: empty** (automated check or visual confirmation).

6. **Dead-trigger process removal in `docs/CYCLE_OPEN_CHECKLIST.md`:**
   - **§5 (UIR refactor trigger review)**: REMOVE entirely. Replace with one line: "UIR refactor: KILL'd permanently per `docs/PLAN_V2.16.md` §4 Item #13 + `docs/DECISIONS.md` v2.16 Carry-Forward Closures. If multi-format need arises, v3.0 re-charter, not v2.X reopen."
   - **§3 carry-forward 6.1 (Docling 2.87 OR 90-day watcher)**: REMOVE. Item #9 KILL closes B1 Docling config hunt; if Phase 3 passes, the watcher is dead process; if Phase 3 fails, v2.17 picks up Item #9 and the watcher belongs to v2.17's checklist scope.

7. **README "feature-complete" banner** near the top of `README.md`:
   > **MM-Converter-V2 is feature-complete as of v2.16.0.** Production retrieval is stable; documented limitations are explicit; only bug-fix patches (v2.16.x) accepted post-tag. New features = re-charter as v3.0.

8. **Layer-0/1 docs sweep** per [[doc-sanitization-completeness]]: CLAUDE.md, AGENTS.md, PROJECT_STATUS.md, ARCHITECTURE.md, TESTING.md, DECISIONS.md, QUALITY_GATES.md — current canonical baseline → v2.16; stale "v2.15 SHIPPED" framings updated; READ FIRST list updated.

9. **Post-tag rollback procedure** documented in `docs/DECISIONS.md` "v2.16 Post-Tag Rollback Procedure":
   - Phase 3 (`partial_code` adjacency): `git revert <phase3_commit_sha>`.
   - Phase 5 (dynamic top-k, only if shipped): `git revert <phase5_commit_sha>`.
   - Both are independent commits on `pipeline.py` per §5 serial-order rule; reverts don't conflict.

10. **v2.16.0 annotated tag** with "FINAL v2.X release" message; pushed to origin + GitHub.

**Definition of Done — v2.16.0 ship gate** (ALL must hold; any failure routes to v2.17 per §7):
- §2 DoD items 1-8 all satisfied.
- All phases whose final disposition is SHIPPED passed their acceptance bars (per-phase: Phases 0-4 unconditional SHIP; Phase 5 SHIP-default-on if pre-flight fires else KILL; Phase 6 SHIP if Phase 2 compound trigger fires else KILL; Phase 7 SHIP only on user opt-in else KILL).
- All KILLed conditional/opt-in phases have DECISIONS.md closure entries AND no production code path.
- All §4 KILL items have DECISIONS.md closure entries.
- All OUT-OF-SCOPE items have DECISIONS.md declaration entries.
- `PROJECT_STATUS.md` Other Carry-Forwards: empty.
- Full pytest suite green.
- v2.14 retrieval fingerprint passes (or fresh v2.16 fingerprint captured if Phase 3/5 changed production retrieval).
- Strict-gate corpus state (now incl. Phase 0 additions) unchanged or improved.
- **Multi-profile smoke**: `bash scripts/smoke_multiprofile.sh` reports `GATE_PASS` + `UNIVERSAL_PASS` for every document category + ≥1 per-category blind-test document. (AGENT-VAL-01 invariant; failure here is a hard tag-block.)
- Phase 6 calibration `expiration_date > today` (T-72h pre-tag checkpoint per v2.15 carry-over).
- README updated with v2.16.0 feature-complete banner.

**Cost:** 0.5 day.

---

## 4. Permanent closures (KILLs)

For DECISIONS.md "v2.16 Carry-Forward Closures" entry. Each item gets a one-paragraph rationale; no future-cycle reopen path.

**Item #9 — B1 Docling HybridChunker config hunt: CLOSED (conditional).** v2.16 Phase 3 (`partial_code` adjacency fetch) resolves the Fluent_Python cross-page code defect deterministically at retrieval time. KILL fires ONLY when Phase 3 passes its full acceptance bar (Fluent_Python ≥85% AND — if Phase 0 surfaces ≥2 code-dense docs — each generalization doc ≥85%). If Phase 3 fails on any leg, Item #9 reopens to v2.17 safety valve (Docling-side investigation deferred under v2.15 Option F). If KILL fires, the Docling-side fight is no longer needed; future corpus-class chunking defects are v3.0-class architectural decisions, not v2.X carry-forwards. Original CYCLE_OPEN_CHECKLIST.md carry-forward 6.1 trigger ("Docling minor ≥2.87 OR every 90 days") REMOVED at Phase N close-out.

**Item #10 — A2 HTML+summary split (Unstructured pattern): CLOSED.** The work is a chunk-emission pattern change in the ingestion path (long-form chunk for embedding + short summary chunk for display) — a 2-5 day v2.X feature, not embedder-retraining-class. But zero demand signal across v2.11→v2.15: no validation query surfaces a summary/long-form mismatch; no strict-gate failure attributable to chunk shape. The current single-chunk pattern works for documented use cases. If summary-vs-content distinction ever becomes load-bearing, v3.0 re-charter.

**Item #12 — B2 Code-Rescue heuristic stitching middleware: CLOSED.** Heuristic regex-based stitching of truncated code blocks at retrieval time is the wrong layer — extraction-layer fixes (Phase 3 adjacency fetch via the `partial_code` flag) produce deterministic results; regex-stitching produces probabilistic results with maintenance debt (every code dialect grows its own special-case). Phase 3 also preserves provenance (which chunks merged together); a stitching middleware loses that. If Phase 3 proves insufficient, the answer is Docling-side configuration (Item #9 reopens to v2.17), not retrieval-time heuristics.

**Item #13 — UIR (Universal Intermediate Representation) refactor (3c): CLOSED.** v2.11 plan; PARKED 5 cycles. The four trigger conditions (3rd engine, cross-engine defect, ≥500 LOC test boilerplate, external integration request) are realistic only when multi-format scope expands beyond the current 2 engines (PDF via Docling + EPUB via `ebooklib`, per `docs/CONVERSION_PROFILES.md` §EPUB Lane). The 7 Phase 0 additions are PDFs — they route through the existing Docling-side scanned/scanned_degraded/technical_manual profiles without requiring a 3rd engine, so the "3rd engine added" trigger does not fire. The other three triggers (cross-engine defect surfacing in current PDF+EPUB usage, ≥500 LOC of cross-engine test boilerplate, external integration request) have not surfaced in 5 cycles and are not on any v2.X roadmap. If a 3rd engine (e.g., DOCX, HTML-native, archive formats) ever becomes load-bearing, that's a v3.0 architecture proposal — not v2.X reopen. CYCLE_OPEN_CHECKLIST.md §5 (trigger review) REMOVED at Phase N close-out.

**Item #14 — VLM swap to alternative model (3a from v2.11): CLOSED.** Current NuMarkdown-8B-Thinking-mlx-8bits produces clean output per v2.14 P1 evidence (5/12 CarOK pages clean output before dedup defect surfaced). The historical failure was dedup, not VLM quality — Phase 4 ships the fix. No swap needed.

**Item #15 — Magazine rendered-region-crop (3e from v2.11): CLOSED.** No demand signal surfaced through validation queries or strict-gate failures. Magazine content currently meets quality bars per existing soak data. Insufficient evidence to justify investment; if image-axis quality ever regresses on magazine content, v3.0.

**Item #21 — 3b Remote CodeFormulaV2 inference: CLOSED.** Local CPU CodeFormulaV2 in Docling 2.86.0 (~27 sec/page on Apple Silicon) is sufficient for one-off batch reconversion in this project's solo-dev workflow. Original trigger ("Docling 2.87+ exposes `RemoteCodeFormulaOptions`") never fired in 5 cycles. Remote inference is a v3.0 optimization.

**Item #22 — 3d HybridChunker per-item token guard: CLOSED.** v2.10 element-by-element fallback already handles pathological-input chunking. The per-item guard was opt-in/default-off in v2.11 design (no behavior change baseline) and never built (4 cycles, zero demand signal). Quality-optimization on an edge case; not load-bearing for feature-complete.

---

## 5. OUT-OF-SCOPE (v3.0)

**Item #11 — D1 ColPali / VisRAG visual retrieval: OUT-OF-SCOPE (v3.0).** Requires per-page visual embeddings, a separate vector store with different shape, and a re-rank stage operating on visual + text dual representations. Full vector-DB explosion + cold-start + architecture rewrite. v3.0 project, not a v2.X feature — re-charter if visual retrieval becomes load-bearing.

---

## 6. KEEP active (ongoing infrastructure, not carry-forwards)

These are not items being closed; they are existing infrastructure that stays through v2.16 and beyond.

**Item #16 — Telemetry collection (v2.15 P3).** Telemetry stays as a second-class signal for objective sanity check + future-deployment readiness; Phase 1's personal_importance overlay is the cycle-open decision authority. **Maintenance contract**: thresholds in `analyze_doc_class_telemetry.py` are FROZEN at v2.16.0 ship values per the feature-freeze — no post-tag tuning permitted (changing them = v3.0 per §8). Log retention: telemetry JSONL files in `logs/doc_class_telemetry/` rotate at 30-day age. If retention or threshold changes ever become necessary, that's a v2.17 trigger only if it surfaces as a Stop-the-Line condition (per §7 trigger #3); otherwise v3.0.

**Item #17 — Phase 4-Resilience qwen3-max cloud fallback (v2.14).** Unchanged production infrastructure.

**Item #20 — Phase 6 calibration freshness check (v2.15).** T-72h pre-tag checkpoint + cycle-open freshness review per CYCLE_OPEN_CHECKLIST.md. Stays through v2.16.

---

## 7. Phase ordering + serial-order rule

**Dependency graph:**
```
Phase 0 ──┬──> Phase 1 ──┬──> Phase 3 ──> Phase 5 (CONDITIONAL on Phase 1 pre-flight)
          │              ├──> Phase 4
          │              └──> Phase 7 (DEFAULT KILL; opt-in route exists)
          └──> Phase 2 ──> Phase 6 (CONDITIONAL on Phase 2 compound trigger)
                          │
                          └──> Phase N (ship gate)
```

**Phase 3 + Phase 5 serial-order rule.** Both modify `retrieve_hybrid_reranked` in `pipeline.py` (Phase 3: post-rerank adjacency fetch; Phase 5: post-rerank truncation). To prevent merge conflict on the same function body:
1. Phase 3 ships first (its commit lands on `main` before Phase 5 starts).
2. Phase 5 branches from post-Phase-3 `main`.
3. Each phase is an independent commit, enabling the §3 Phase N rollback procedure (`git revert <commit_sha>` per phase).

**Parallel-eligible:** Phase 1 (fixture authoring) and Phase 2 (diagnostic spike) can run in parallel after Phase 0. Phases 3 + 4 can run in parallel after Phase 1 (different code paths: Phase 3 = retrieval, Phase 4 = ingestion).

---

## 8. Budget

**Time cap: 12 working days (hard).** Overflow routes to v2.17 only with explicit user sign-off + `cycle_slip.log` entry (per §9 trigger #4); default is tag-block.

**Per-phase estimate** (median path; ranges in parens reflect Phase 1 fixture authoring N + conditional outcomes):

| Phase | Estimate | Notes |
|---|---|---|
| Phase 0 | 1d | Includes probes + pre-validation + Qdrant snapshot + CANONICAL_DOCS rename. |
| Phase 1 | 1.5-3d | Implementation + N × 0.5d fixture authoring; baseline run. |
| Phase 2 | 1d | Includes Phase 6 analytical pre-flight if triggered. |
| Phase 3 | 1-2d | Schema spike + adjacency code + bridge tests + validation. |
| Phase 4 | 2-3d | bbox_iou + dedup + bridge tests + re-extract + generality. |
| Phase 5 | 0-1d | 0d if pre-flight KILLs; 1d if ships. |
| Phase 6 | 0-3d | 0d if Phase 2 compound trigger fails; 2-3d if both legs fire. |
| Phase 7 | 0d default | 0d if default KILL; 3-5d if user opts in (Phase 1 fixture authoring extends). |
| Phase N | 0.5d | Tag, snapshot, DECISIONS, README, dead-trigger removal. |

**Serial-minimum (all conditionals fire, no parallel):** ~9-10 days.
**Serial-minimum (all conditionals KILL, parallel Phase 1+2 and Phase 3+4):** ~7-8 days.

**Cloud spend: $2-3 baseline, ~$6 worst case (Phases 2 + 6 both fire).** Hard cap: $25/cycle (Dashscope). Current cap is well above projection.

---

## 9. v2.17 safety valve (tight scope)

v2.17 exists ONLY for unexpected issues during v2.16 convergence execution. Four enumerated triggers; exhaustive. Default for anything outside these four = fold into v2.16, no defer.

1. **SHIP phase acceptance bar genuinely FAILS** and the fix is non-trivial (>2 dev days). Example: Phase 3 partial_code fetch produces ≤30% Fluent_Python pass rate (vs the ≥85% bar) AND investigation reveals a structural issue not addressable in-cycle. Acceptance-bar failure, not schedule overflow (those are separate; see #4).

2. **External dependency breaks** during convergence: Docling pin breaks; omlx-server protocol incompatibility; Qdrant schema change forces re-ingestion at scale.

3. **Strict-gate regression**: Phase 0/3/4/5 work causes 34/34 PASS to drop. Stop-the-line condition; fix in v2.17, do not ship v2.16 with regressed strict-gate.

4. **Convergence-cycle schedule overflow with explicit user sign-off**: at the 12-day hard cap, if a SHIP phase has not completed AND the user signs off in writing (`docs/cycle_slip.log` or DECISIONS.md entry) that the remaining work cannot be compressed inside the cap, the remaining work routes to v2.17. **Default behavior without sign-off: tag-block.** No silent overflow.

**Non-triggers** (these route to other resolution paths, NOT v2.17):
- Audit findings arriving late after v2.16.0 tag → either v2.16.x bug fix (if it's a regression) or v3.0 (if it's a tuning preference). Per §10.
- Phase 2 inconclusive verdict → routes to Phase 6 KILL per Phase 2 acceptance, not v2.17.
- Phase 1 pre-flight returning "ambiguous" → structurally precluded by the binary gate (three conjuncts, each bivalent).
- Post-tag corpus-level bug discovered later → v2.16.x patch lane (per §10).
- "Could be tuned better" preferences → v3.0 (per §10).

---

## 10. Post-convergence governance

After v2.16.0 ships, the cycle-plan workflow ends. Only the patch lane (v2.16.x) and the re-charter path (v3.0) remain.

### 10.1. v2.16.x patch lane

A change is a v2.16.x patch ONLY if it fixes a demonstrable regression from v2.16.0 behavior on the v2.16.0 corpus (something that worked at v2.16.0 ship now doesn't). Examples:
- Phase 4 IoU threshold tuned because a CarOK retrieval that PASSed at v2.16.0 now FAILs after a Docling minor update → v2.16.x.
- Phase 3 adjacency fetch produces wrong merge on a chunk that previously merged correctly → v2.16.x.
- Phase 5 truncation parameters tuned because the v2.16.0 PASS-retention bound is now violated on the same corpus → v2.16.x.

Changes motivated by new documents, new use cases, or "the threshold could be better tuned for X" are NOT v2.16.x — they are v3.0 re-charter. This rule applies uniformly to every tunable knob: Phase 4 IoU, Phase 5 truncation params, Item #16 telemetry thresholds, any future addition.

### 10.2. v3.0 re-charter conditions

v3.0 is appropriate when ANY of:
- New retrieval architecture proposed (visual retrieval, multi-modal embeddings, full Qdrant schema change).
- Multi-engine routing becomes necessary (Item #13 UIR refactor reopens for real).
- Multi-format support beyond PDF (HTML, EPUB, Office formats with native handlers).
- LLM-stack swap requiring re-calibration of all judges + Phase 6 fallback (GX10 endpoint replaced; new provider; etc.).
- Corpus expansion beyond curated personal use (e.g., multi-user, multi-tenant, hosted).
- "The threshold could be better tuned for X" desires that don't map to a v2.16.0-corpus regression.

### 10.3. v3.0 re-charter process

1. Draft `docs/PLAN_V3.0.md` — convergence-style discipline applies (SHIP / KILL / OUT-OF-SCOPE per item).
2. Audit cadence per v2.15 §9 (two consecutive 0-HIGH external rounds).
3. Cycle opens only after audit clears.
4. v2.16.x patch lane remains active in parallel during v3.0 development (v2.16.0 corpus regressions still patch to v2.16.x, not v3.0).

### 10.4. Opt-in post-convergence health monitoring

`scripts/run_personal_validation.py` exists and can be invoked at any post-tag interval the user chooses (manually, via cron, or on pipeline change). No required schedule — the script is infrastructure, not a process obligation. If invoked and any class drops below `target_pass_rate`, the script exits nonzero, surfacing the regression without a human verification loop.

Per [[no-human-verification-loops]], v2.16 does NOT prescribe a monitoring cadence. Active alerting infrastructure (cron-scheduled hash-verification, schema-drift watchers) is OUT-OF-SCOPE for v2.X — feature-frozen products rely on the opt-in script + the user noticing regressions during use.

---

## 11. Process notes

- Audit cleared 2026-05-25 (8 external rounds + 1 self-audit; v2.15 §9 stopping rule fired at Round 8). Full archaeology in `docs/archive/plans/PLAN_V2.16_0.10.md`.
- The convergence-cycle frame is itself reviewable — if mid-execution evidence surfaces that a §2 disposition is wrong, the matrix can be amended with a recorded rationale + ship-gate item update. But the bar is high; ad-hoc "let's keep this open" preferences do not override §2.
- Hard memories carry forward unchanged: [[no-gx10-model-swap-reflex]], [[gx10-deployment-guardrails]], [[fix-extraction-not-judge]], [[contract-violation-mode]], [[libraries-first]], [[no-human-verification-loops]], [[doc-sanitization-completeness]].
- Post-v2.16.0: the cycle-plan workflow ends. v2.17 only if a §9 trigger fires. After v2.17 (if it fires): bug fixes only as v2.17.x patches. No more v2.X plans.

---

## Pre-execution checklist

All cleared:
- [x] Audit stopping rule fired (2026-05-25 at Round 8).
- [x] §8a Q1 (validation pass-rate threshold) answered: **≥85% uniform**.
- [x] Q3 (Phase 7 scope) default accepted: Phase 7 KILLs at Phase N (no image-heavy validation queries authored).
- [x] §8b defaults applied: CarOK + Fluent_Python = HIGH; new classes = MED; README banner spec'd; inventory report path = `docs/CORPUS_EXPANSION_2026-05-24_v2.16_p0.md`; v2.17 trigger interpretation = exhaustive (4 triggers).
- [x] §4 disposition matrix locked: 8 KILL items + 1 OUT-OF-SCOPE + 3 KEEP-active + 2 already-CLOSED.

**The cycle opens on next commit.**

---

## Appendix — Provenance

Full audit-round archaeology (Rounds 0-8; 70+ findings disposed across 10 drafts) preserved at `docs/archive/plans/PLAN_V2.16_0.10.md`. This execution plan is the convergence of those rounds; do not edit it with parenthetical "REVISED in v0.X per Finding Y" annotations — if a post-tag amendment is necessary, supersede with `docs/PLAN_V2.16.1.md` (patch-lane plan) or `docs/PLAN_V3.0.md` (re-charter), not by mutating this file.
