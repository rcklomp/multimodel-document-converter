# Quality Snapshot 2026-05-16 — v2.10.0-rc1 AFTER

> **Status:** `v2.10.0-rc1` corpus baseline (Phase 8 close,
> `validated-local`; release tag staged, not pushed).
> Strict gate reports **34 PASS / 0 WARN / 0 FAIL** across the
> 34-doc canonical corpus, with all eight v2.9.0-rc1 signed
> deferrals closed locally by PLAN_V2.10 Phases 1-7 and
> re-verified corpus-wide in Phase 8. Qdrant `mmrag_v2_8` rebuilt
> 2026-05-16, `status: green`, `points_count: 30,454`,
> `indexed_vectors_count: 30,192`. No new advisory codes were
> added; no QA threshold was weakened.
>
> Predecessors:
> - `docs/QUALITY_SNAPSHOT_2026-05-11_v2.9.0-rc1_after.md` (v2.9.0-rc1 ship state — 26 PASS / 0 WARN / 8 FAIL)
> - `docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md` (v2.8.0 SHIPPED)

## 1. Headline numbers

| Metric | 2026-05-11 v2.9.0-rc1 AFTER | 2026-05-16 v2.10.0-rc1 AFTER | Net delta |
|---|---:|---:|---:|
| `QA_PASS` | 12 | **16** | +4 |
| `QA_PASS_WITH_ADVISORIES` | 14 | **18** | +4 |
| **Total PASS-class** | 26 | **34** | **+8** |
| `QA_WARN` | 0 | **0** | 0 |
| `QA_FAIL` | 8 | **0** | **−8** |
| Test suite | 806 passed | **966 passed + 7 Phase 8 pins, 14 skipped** | **+167** |

All 8 v2.9.0-rc1 signed deferrals are closed locally and corpus-wide
re-verified. 0 advisory codes were added to
`_ALLOWED_ADVISORY_WARN_CODES` between v2.9.0-rc1 and v2.10.0-rc1.

## 2. v2.10 phases shipped (cumulative)

Each phase ships a principle-based fix (no profile-overfit, no
filename-specific logic) with corpus-validated regression tests
and red→green test pins.

| Phase | Topic | Doc(s) | Code site(s) | Tests added |
|---|---|---|---|---:|
| 1 | `TEXT_LABEL_TOC_DENSE_INDEX_ROUTER_MISS` | Chaubal p11 | dense-index router extended to detect dotted-leader compact TOC tails even when Docling labels them `text` | +N |
| 2 | `TEXT_INTEGRITY_SCOUT_FULL_DOC_SENSITIVITY` | Fluent_Python | per-batch trigger module + parallel-site quarantine fix (ratio-based detector) | +10 |
| 3 | `B4B_FULL_DOC_PICTURE_DEDUP` | Earthship, Python_Distilled | pHash dedup page-coverage carve-out + SHADOW-EXTRACTION page-coverage-aware threshold (200×200 floor for pages with no prior chunks) | +9 |
| 4 | `CROSS_PAGE_SPLIT_PAGE_ATTRIBUTION` | Python_Cookbook, Python_Distilled | per-page text reconstruction via `prov.charspan` slicing + bare `DocItem` dereferencing + subtitle-continuation promotion + page-scoped overlap-trim + audit micro_non_label heading exemption | +27 |
| 5 | `HYBRID_CHUNKER_HEADING_PROPAGATION` | Devlin | single propagation site at export boundary; heading validator tightened against repeated-token artefacts, code/JSON heading shapes, and bracket-prefixed code labels; `_GENERIC_CARRY_HEADINGS` blocks `Start` / `Front Matter` from seeding forward carry-state | +7 |
| 6 | `OCR_PATH_HEADING_PROPAGATION` | Firearms | ordered OCR-lane heading attribution through `Region.is_heading` / `ProcessedChunk.is_heading`; central `ContextStateV2` validator tightened (terminal-period sentence-shape, numbered-prefix body-case shape); single-page push gate on both OCR heading paths; audit-fix infix step-number repair (`BatchProcessor._repair_infix_step_numbers`); targeted enrichment of 264 pending shadow chunks | +70 (OCR) +23 (infix) |
| 7 | `KI_EPUB_EXTRACTION_LANE_REWRITE` | KI EPUB, ChatGPT EPUB regression control | `_epub_to_html` spine walk + `__MMRAG_EPUB_CH_NNNN__` markers; `_apply_epub_synthetic_pagination` rewrites EPUB chunks with `chapter_1based * 1000 + position_in_chapter // 5`, `[0,0,1000,1000]` bbox, `extraction_method="epub_html"`, regenerated chunk_id, per-synthetic-page dedup; EPUB-aware strict-gate branch via `ebooklib`; `MISSING_CHAPTERS` advisory for contiguous leading/trailing low-content structural spine items | +8 |
| 8 | Strict-Gate Re-Verification + v2.10 Release Prep | corpus-wide | 3-doc image re-enrichment (Devlin/Cookbook/Distilled 443 chunks); Firearms canonical-CLI reconvert + re-enrichment; Qdrant `mmrag_v2_8` rebuild; `search_qdrant.py` API-key cleanup + env-var migration; v2.10 release tag staged | (regression tests only — corpus pin + version pin) |
| **Total** | — | — | — | **+~160** |

## 3. v2.10 release contract — closure of v2.9.0-rc1 deferrals

The eight v2.9.0-rc1 signed deferrals (per `docs/DECISIONS.md`
"v2.9.0-rc1 Signed Deferrals (2026-05-11 close-out)") are now all
`validated-local` and corpus-wide re-verified under the unchanged
strict gate:

| # | Doc | Class | v2.10 phase | v2.10 state |
|---|---|---|---:|---|
| 1 | Firearms | `OCR_PATH_HEADING_PROPAGATION` | Phase 6 | `QA_PASS: failures=0 warnings=0` |
| 2 | KI_En_ChatGPT_Praktische_Gids | `KI_EPUB_EXTRACTION_LANE_REWRITE` | Phase 7 | `QA_PASS_WITH_ADVISORIES: failures=0 warnings=1` (MISSING_CHAPTERS edge advisory) |
| 3 | Devlin_LLM_Agents | `HYBRID_CHUNKER_HEADING_PROPAGATION` | Phase 5 | `QA_PASS_WITH_ADVISORIES` (post Phase 5 + Phase 8 re-enrichment) |
| 4 | Python_Cookbook | `CROSS_PAGE_SPLIT_PAGE_ATTRIBUTION` | Phase 4 | `QA_PASS_WITH_ADVISORIES` (post Phase 4 + Phase 8 re-enrichment) |
| 5 | Python_Distilled | mixed B3.b + B4.b | Phase 3 + Phase 4 | `QA_PASS_WITH_ADVISORIES` (post Phases 3-4 + Phase 8 re-enrichment) |
| 6 | Fluent_Python | `TEXT_INTEGRITY_SCOUT_FULL_DOC_SENSITIVITY` | Phase 2 | `QA_PASS_WITH_ADVISORIES: failures=0 warnings=1` |
| 7 | Chaubal_PyTorch_Projects | `TEXT_LABEL_TOC_DENSE_INDEX_ROUTER_MISS` | Phase 1 | `QA_PASS` |
| 8 | Earthship_Vol1 | `B4B_FULL_DOC_PICTURE_DEDUP` | Phase 3 | `QA_PASS` |

No deferral was rolled forward to v2.11. No new signed deferrals were
created for v2.10.

## 4. Per-document strict-gate state

QA_PASS = full pass. QA_PASS_WITH_ADVISORIES = pass with documented
advisory warning(s) per `docs/QUALITY_GATES.md` "Advisory Warning
Classes". The 0013 form variant continues to use the
`GATE_PASS [form: ...]` lane per `docs/QUALITY_GATES.md` "Form /
Invoice Acceptance Class".

Source of truth for this table is the 2026-05-16 Phase 8 final run of
`scripts/run_strict_gate_corpus.py` (captured at
`/tmp/v2.10_strict_gate_final.txt`):

| # | Doc | v2.9.0-rc1 | v2.10.0-rc1 | Δ | Advisory / Notes |
|---|---|---|---|---|---|
| 1 | HarryPotter_and_the_Sorcerers_Stone | QA_PASS_WITH_ADVISORIES | QA_PASS_WITH_ADVISORIES | — | ASSET_TINY |
| 2 | Form_0013_invoice | QA_PASS | QA_PASS | — | form variant (`GATE_PASS [form: ...]` in smoke) |
| 3 | Form_betwistingsformulier | QA_PASS_WITH_ADVISORIES | QA_PASS_WITH_ADVISORIES | — | ASSET_TINY |
| 4 | CarOK_voorraadtelling | QA_PASS | QA_PASS | — | — |
| 5 | AIOS_LLM_Agent_Operating_System | QA_PASS_WITH_ADVISORIES | QA_PASS_WITH_ADVISORIES | — | VISION_HARD_FALLBACK_RATE (F4) |
| 6 | A_comprehensive_review_on_hybrid_electri | QA_PASS_WITH_ADVISORIES | QA_PASS_WITH_ADVISORIES | — | ASSET_TINY |
| 7 | Hybrid_electric_vehicles | QA_PASS_WITH_ADVISORIES | QA_PASS_WITH_ADVISORIES | — | ASSET_TINY |
| 8 | IRJET_Modeling_of_Solar_PV | QA_PASS | QA_PASS | — | — |
| 9 | Recent_Trends_in_Transportation | QA_PASS | QA_PASS | — | — |
| 10 | Combat_Aircraft_August_2025 | QA_PASS | QA_PASS | — | hf_rate 4.0 % (below 5 % WARN threshold) |
| 11 | PCWorld_July_2025 | QA_PASS | QA_PASS | — | — |
| 12 | ATZ_Elektronik_German | QA_PASS | QA_PASS | — | — |
| 13 | Kimothi_RAG_Guide | QA_PASS_WITH_ADVISORIES | QA_PASS_WITH_ADVISORIES | — | SCRIPT_ADVISORY_FAIL (code_indentation_fidelity 0.667), VISION_HARD_FALLBACK_RATE (F4) |
| 14 | Integra_manual | QA_PASS | QA_PASS | — | — |
| 15 | Jungjun_AI_Agent | QA_PASS_WITH_ADVISORIES | QA_PASS_WITH_ADVISORIES | — | VISION_HARD_FALLBACK_RATE (F4) |
| 16 | Bourne_RAG_2024 | QA_PASS_WITH_ADVISORIES | QA_PASS_WITH_ADVISORIES | — | VISION_HARD_FALLBACK_RATE (F4) |
| 17 | Devlin_LLM_Agents | QA_FAIL (#3) | **QA_PASS_WITH_ADVISORIES** | **+PASS** | VISION_HARD_FALLBACK_RATE (F4); Phase 5 + Phase 8 re-enrichment closed `HYBRID_CHUNKER_HEADING_PROPAGATION` deferral |
| 18 | Raieli_AI_Agents | QA_PASS_WITH_ADVISORIES | QA_PASS_WITH_ADVISORIES | — | SCRIPT_ADVISORY_FAIL (code_indentation_fidelity 0.884) |
| 19 | Adedeji_GenAI_Google_Cloud | QA_PASS | QA_PASS | — | — |
| 20 | Cronin_GenAI_Models | QA_PASS | QA_PASS | — | — |
| 21 | Hao_ML_Platform | QA_PASS_WITH_ADVISORIES | QA_PASS_WITH_ADVISORIES | — | ASSET_TINY |
| 22 | Nagasubramanian_Agentic_AI | QA_PASS_WITH_ADVISORIES | QA_PASS_WITH_ADVISORIES | — | VISION_HARD_FALLBACK_RATE (F4), ASSET_TINY |
| 23 | Sekar_MCP_Standard | QA_PASS_WITH_ADVISORIES | QA_PASS_WITH_ADVISORIES | — | VISION_HARD_FALLBACK_RATE (F4), ASSET_TINY |
| 24 | Python_Cookbook | QA_FAIL (#4) | **QA_PASS** | **+PASS** | Phase 4 + Phase 8 re-enrichment closed `CROSS_PAGE_SPLIT_PAGE_ATTRIBUTION` deferral |
| 25 | ArcGIS_Python_Cookbook | QA_PASS | QA_PASS | — | — |
| 26 | Fluent_Python | QA_FAIL (#6) | **QA_PASS_WITH_ADVISORIES** | **+PASS** | SCRIPT_ADVISORY_FAIL; Phase 2 closed `TEXT_INTEGRITY_SCOUT_FULL_DOC_SENSITIVITY` deferral |
| 27 | Python_Distilled | QA_FAIL (#5 mixed B3.b + B4.b) | **QA_PASS_WITH_ADVISORIES** | **+PASS** | VISION_HARD_FALLBACK_RATE; Phases 3 + 4 + Phase 8 re-enrichment closed the mixed deferral |
| 28 | Ayeva_Python_Patterns | QA_PASS | QA_PASS | — | — |
| 29 | Chaubal_PyTorch_Projects | QA_FAIL (#7) | **QA_PASS** | **+PASS** | Phase 1 closed `TEXT_LABEL_TOC_DENSE_INDEX_ROUTER_MISS` deferral |
| 30 | Earthship_Vol1 | QA_FAIL (#8) | **QA_PASS** | **+PASS** | Phase 3 closed `B4B_FULL_DOC_PICTURE_DEDUP` deferral; hf_rate 3.8 % (below 5 % WARN threshold) |
| 31 | Firearms | QA_FAIL (#1) | **QA_PASS** | **+PASS** | Phase 6 + Phase 8 canonical reconvert + re-enrichment closed `OCR_PATH_HEADING_PROPAGATION` deferral |
| 32 | Greenhouse_Design | QA_PASS_WITH_ADVISORIES | QA_PASS_WITH_ADVISORIES | — | SCRIPT_ADVISORY_FAIL (code_indentation_fidelity 0.800) — blind-test technical-manual document per AGENT-VAL-01 |
| 33 | ChatGPT_Praktijk_handboek | QA_PASS_WITH_ADVISORIES | QA_PASS_WITH_ADVISORIES | — | PAGE_COUNT_UNKNOWN (EPUB lane) |
| 34 | KI_En_ChatGPT_Praktische_Gids | QA_FAIL (#2) | **QA_PASS_WITH_ADVISORIES** | **+PASS** | MISSING_CHAPTERS (edge low-content structural spine items); Phase 7 closed `KI_EPUB_EXTRACTION_LANE_REWRITE` deferral |

**Eight v2.9.0-rc1 signed deferral FAILs → PASS:** rows 17 (Devlin),
24 (Cookbook), 26 (Fluent), 27 (Distilled), 29 (Chaubal), 30
(Earthship), 31 (Firearms), 34 (KI EPUB). All 26 prior PASS-class
rows retained their pass status; the rc1 Greenhouse blind-test row
(per AGENT-VAL-01) remained `QA_PASS_WITH_ADVISORIES`.

## 5. Image-chunk vision-status corpus snapshot

Precise counts captured 2026-05-16 from the 34 canonical JSONLs
(`output/<doc>/ingestion.jsonl`, `modality == "image"`):

| Status | Count | Notes |
|---|---:|---|
| Total image chunks across 34 docs | **4,548** | +169 vs v2.9.0-rc1 (post-Phase-6 Firearms reconvert image count is identical to rc1 at 1,089; the +169 delta comes from the Phase 4/5 reconverts of Devlin, Python_Cookbook, and Python_Distilled) |
| `vision_status="complete"` | **4,441** (97.6 %) | qwen3-vl-plus descriptions |
| `vision_status="hard_fallback"` | **107** (2.4 %) | 94 F4 sentinel (`complex_asset_short_response_after_retry`) + 13 `[VLM_FAILED: call error]` |
| `vision_status="pending"` | **0** | Goal 5 met |

Hard-fallback distribution per doc and the F4 sentinel-conditional
contract (`docs/QUALITY_GATES.md` "Advisory Warning Classes"):

- All docs with hard-fallback rate > 5 % raise the
  `VISION_HARD_FALLBACK_RATE` advisory and every hard_fallback chunk
  on those docs carries the F4 sentinel:
  - AIOS (10.0 %), Jungjun (10.9 %), Kimothi (7.1 %), Bourne (6.9 %),
    Nagasubramanian (10.5 %), Sekar (8.6 %), Devlin (14.7 % after
    Phase 5 reconvert + Phase 8 re-enrichment), Python_Distilled (3.2 %
    — below threshold, no advisory raised).
- Docs at hf_rate ≤ 5 % do not raise `VISION_HARD_FALLBACK_RATE` and
  the F4-conditional rule does not apply. Combat (4.0 %, 12 non-F4
  `[VLM_FAILED: call error]`) and Earthship (3.8 %, 1 non-F4
  `[VLM_FAILED: call error]`) fall in this band; both strict-gate
  rows remain `QA_PASS: failures=0 warnings=0`.

**No `vision_status="pending"` corpus-wide.** Phase 8 re-enrichment
of the 443 Devlin/Python_Cookbook/Python_Distilled chunks
(2026-05-16 02:43 → 03:06, 432 enriched + 11 F4 hard_fallback) plus
the Firearms reconvert + 1,089-chunk re-enrichment (2026-05-16
02:43 → 04:49, 1,088 enriched + 1 hard_fallback) closes the
post-Phase-4/5 pending residue noted in the initial Phase 8 probe.

## 6. Test suite provenance

| Phase milestone | Tests passed | Δ |
|---|---:|---:|
| `v2.9.0-rc1` close (2026-05-11) | 806 | — |
| Phase 2 close | 817 | +11 |
| Phase 3 close | 826 | +9 |
| Phase 4 close | 853 | +27 |
| Phase 5 close | 860 | +7 |
| Phase 6 close | 953 | +93 (OCR + infix-repair) |
| Phase 7 close | 966 | +13 (EPUB lane + advisory) |
| Phase 8 close (v2.10.0-rc1) | **973** | +7 |

## 7. Qdrant rebuild

Pre-rebuild state (2026-05-16 ≈ 04:55 local): `mmrag_v2_8` collection
absent (`/collections` returned empty list at HTTP 200 after the
operator started the Qdrant container; the destructive `curl -X
DELETE` step from `docs/PLAN_V2.10.md` §Phase 8 step 6a was therefore
a no-op).

User authorization recorded 2026-05-16 in the Phase 8 working
session: "Confirm: run the rebuild now" via the AskUserQuestion
follow-up after the auto-classifier blocked the initial launch.

Rebuild command (kicked off 2026-05-16 ≈ 04:55 local):

```bash
conda run -n mmrag-v2 --no-capture-output \
  python scripts/rebuild_mmrag_v2_8_for_rc1.py \
  | tee /tmp/qdrant_rebuild_v2_10.log
```

This iterates the 34-doc canonical list, calling
`scripts/ingest_to_qdrant.py` with `--collection mmrag_v2_8`
(`--recreate` on doc 1 only). Embedding model:
`llava:latest` via local Ollama (4096-dim, cosine, on-disk
mmap-resident). Expected wall time per the rc1 cycle: ~10h15m.

Raw chunk count across the 34 v2.10 JSONLs (computed during Phase 8
probe): **30,588** (25,649 text + 4,548 image + 391 table).
`scripts/ingest_to_qdrant.py` filters a small fraction of chunks
during ingest (empty content, missing asset refs, validator
rejections) at a consistent ~0.44 % corpus-wide; the filtered count
matches the rate observed across the rc1 rebuild.

**Final `points_count` after rebuild = 30,454**
(`indexed_vectors_count = 30,192`, `segments_count = 5`,
`status = green`). v2.9.0-rc1 final `points_count` was **30,461**;
the net delta versus rc1 is **−7** points. Decomposed at the
filtered (Qdrant-side) level rather than raw-JSONL level, the
−7 net is dominated by the Phase 4 cross-page split fix changing
per-doc text-chunk shapes (Cookbook / Distilled / Devlin / Fluent)
and the Phase 7 EPUB-lane rewrite (KI EPUB went 4,519 raw → 4,512
points; Phase 6 reconverts shifted Firearms / Adedeji / Cronin shapes
slightly). All chunk_ids remain unique corpus-wide (Phase 8 audit
pass), so the −7 is real shape change in the JSONLs, not duplicate
collapse.

**Verification (post-rebuild, 2026-05-16):**

```bash
curl -s http://localhost:6333/collections/mmrag_v2_8 \
  | jq '.result | {status, points_count, indexed_vectors_count, vectors_count}'
# → {"status":"green","points_count":30454,"indexed_vectors_count":30192,...}
```

Required: `status == "green"` and `points_count == 30454`. Devlin
payload staleness (v2.9.0-rc1 §7 housekeeping item) is absorbed by
this rebuild because the Devlin JSONL ingested is the Phase 5 +
Phase 8 re-enrichment output.

**Rebuild execution log (2026-05-16):**

- Initial rebuild launched 2026-05-16 ≈ 10:06 (`scripts/rebuild_mmrag_v2_8_for_rc1.py`)
  aborted at doc 11/34 (PCWorld) with `URLError: [Errno 61]
  Connection refused` when Ollama (port 11434) became unreachable
  mid-loop. The partial `mmrag_v2_8` collection at that point held
  1,999 points (docs 1-10 cleanly committed).
- Ollama was restarted out-of-band by the operator. A resume loop
  (`/tmp/qdrant_resume_phase8.sh`, log `/tmp/qdrant_resume_phase8.log`)
  invoked `scripts/ingest_to_qdrant.py` directly for docs 11-34
  without `--recreate`. Deterministic uuid5 point IDs (per the v2.8
  commit `0d3cc36` contract) made the per-doc retries idempotent;
  no document was re-embedded.
- Resume loop completed 2026-05-16 18:17:26 UTC (full duration
  09:18–18:17 ≈ 9h00m for docs 11-34 + the 0h46m initial run for
  docs 1-10 = ~9h45m total embedding wall time on the local
  16 GB M1 Mac).

**Rebuild status at AFTER-snapshot commit:** complete and
verified. Phase 8 advances to `validated-local`. Release-tag
command staged but not executed; user pushes the tag per
`docs/PLAN_V2.10.md` §Phase 8 step 12.

**Operational lesson captured for v2.11:** the rebuild script
should fail fast on Ollama unavailability (a connection probe
before doc 1) and should exit non-zero through the conda wrapper —
rather than letting the wrapper smuggle a misleading rc=0. Track
this as a v2.10 followup item in
[`docs/PROJECT_STATUS.md`](PROJECT_STATUS.md).

## 8. v2.10 lessons captured

1. **Reconverts must re-enrich.** Phases 4 and 5 reconverted
   Devlin / Python_Cookbook / Python_Distilled after the v2.9.0-rc1
   Phase H VLM enrichment, leaving 443 image chunks at
   `vision_status="pending"`. Phase 8 explicitly re-enriched these
   before the corpus strict-gate pass. Future reconvert-then-validate
   sequences must include the enrichment step in the workflow, not as
   a separate manual follow-up.
2. **Canonical JSONLs need explicit ownership.** The pre-Phase-8
   `output/Firearms/ingestion.jsonl` was the v2.9.0-rc1 enriched
   version; Phase 6 left its post-fix output in `output/Firearms_phase6e/`
   rather than overwriting canonical. Phase 8 reconverted canonical via
   the documented `mmrag-v2 process --vision-provider none` invocation
   and re-enriched.
3. **The strict-gate probe is the only authoritative state check.**
   Source-file `mtime` is not a sufficient proxy for "post-fix" state
   when the fix involves a reconvert. Phase 8 always ran the probe
   first.

## 9. v2.10 housekeeping closure

The three v2.9.0-rc1 carry-forward items from
`docs/QUALITY_SNAPSHOT_2026-05-11_v2.9.0-rc1_after.md` §9 are closed:

1. **Devlin Qdrant payload staleness** — **closed**. The Phase 8
   `mmrag_v2_8` rebuild completed 2026-05-16 18:17:26 UTC against the
   post-Phase-5 reconvert + Phase 8 re-enrichment Devlin JSONL. Final
   collection state: `status: green`, `points_count: 30,454`,
   `indexed_vectors_count: 30,192`. Vector + reranker smoke
   (`scripts/search_qdrant.py "what is MCP" -c mmrag_v2_8 -n 3`)
   returns topically-correct top-3 chunks (Sekar MCP §3.4.3 + Part III
   opener).
2. **`scripts/search_qdrant.py` `--model` default and `MIN_SCORE` floor
   re-tune** — v2.10 keeps the llava 4096-dim embedder in
   `scripts/ingest_to_qdrant.py`; the `--model llava` default and
   `MIN_SCORE = 0.20` floor remain calibrated for the rebuilt
   collection. No re-tune required.
3. **Hard-coded Dashscope API key** — **closed**. Replaced with
   `os.environ.get("DASHSCOPE_API_KEY", "")` in
   `scripts/search_qdrant.py` and `scripts/validate_qdrant.py`, plus
   `${DASHSCOPE_API_KEY:-}` in `scripts/convert_all.sh`. The rerank
   functions gracefully degrade to vector-rank truncation when the env
   var is unset. The leaked literal that still lived in git history
   was revoked by the user at Alibaba Cloud Model Studio on
   2026-05-16, so any historical commit containing the literal can no
   longer authenticate against the provider.

## 10. Revision log

| Date | Change |
|---|---|
| 2026-05-16 | Initial v2.10.0-rc1 AFTER snapshot at Phase 8 close (rebuild in progress at first author). |
| 2026-05-16 | §7 + §9 + status banner updated after the Phase 8 resume loop completed at 18:17:26 UTC. Final `points_count: 30,454` (raw chunk count was 30,588; `ingest_to_qdrant.py` filtered 134, ~0.44 %). v2.9.0-rc1 → v2.10.0-rc1 net delta: −7 points. Devlin staleness item closed by the rebuilt collection. Phase 8 advances to `validated-local`. |
| 2026-05-16 | §9 item 3 closed: the leaked Dashscope API-key literal that lived in git history was revoked by the user at Alibaba Cloud Model Studio. Any historical commit containing the literal can no longer authenticate. The last user-only v2.10 release-prep action is now done; only the `v2.10.0-rc1` annotated tag push remains. |
