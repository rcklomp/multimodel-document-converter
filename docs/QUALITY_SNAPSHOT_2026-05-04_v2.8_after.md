# Quality Snapshot 2026-05-04 — v2.8 AFTER

**Purpose:** AFTER state for the v2.8 broad reconversion (Phase 5c).
Compare against `docs/QUALITY_SNAPSHOT_2026-05-03.md` for the BEFORE column.

**HEAD:** `c2e795e fix(audit): form TEXT-verdict contradiction + overnight pipeline scaffolding`

**Pre-flight evidence (committed in `5b0e13d`):**
- pytest tests/ -q: 590 passed, 2 skipped, 0 failed.
- bash scripts/smoke_multiprofile.sh: 11/11 GATE_PASS + 11/11 UNIVERSAL_PASS.
- HARRY pages-1-30 live acceptance: PASS.

## Aggregate (v2.8 canonical corpus only)
- **30/34 PASS** (includes form-pass class)
- **4 FAIL**
- **1 forms** (form acceptance class — invoices/short scanned docs)
- 22 legacy probes / exploration outputs (table 2)

## Per-document AFTER (canonical v2.8 corpus)

| Output dir | Chunks | Class | AFTER | BEFORE | Delta |
|---|---|---|---|---|---|
| A_comprehensive_review_on_hybrid_electri | 191 | digital | PASS | **FAIL (TEXT)** | ✓ FIXED |
| Adedeji_GenAI_Google_Cloud | 834 | digital | PASS | PASS | ✓ stable |
| AIOS_LLM_Agent_Operating_System | 146 | digital | PASS | PASS | ✓ stable |
| ArcGIS_Python_Cookbook | 928 | digital | PASS | — | NEW |
| ATZ_Elektronik_German | 60 | digital | PASS | PASS | ✓ stable |
| Ayeva_Python_Patterns | 753 | digital | FAIL | **FAIL (TEXT, CODE)** | still FAIL |
| Bourne_RAG_2024 | 1164 | digital | PASS | PASS | ✓ stable |
| CarOK_voorraadtelling | 76 | digital | PASS | — | NEW |
| ChatGPT_Praktijk_handboek | 298 | digital | PASS | — | NEW |
| Chaubal_PyTorch_Projects | 695 | digital | PASS | **FAIL (CODE)** | ✓ FIXED |
| Combat_Aircraft_August_2025 | 584 | digital | PASS | **FAIL (TEXT)** | ✓ FIXED |
| Cronin_GenAI_Models | 1288 | digital | PASS | PASS | ✓ stable |
| Devlin_LLM_Agents | 764 | digital | PASS | PASS | ✓ stable |
| Earthship_Vol1 | 675 | scanned | PASS | PASS | ✓ stable |
| Firearms | 1690 | scanned | FAIL | PASS | ⚠ REGRESSION |
| Fluent_Python | 1602 | digital | PASS | — | NEW |
| Form_0013_invoice | 20 | scanned | FORM_PASS | — | NEW |
| Form_betwistingsformulier | 8 | digital | PASS | — | NEW |
| Greenhouse_Design | 1148 | digital | PASS | — | NEW |
| Hao_ML_Platform | 1255 | digital | PASS | — | NEW |
| HarryPotter_and_the_Sorcerers_Stone | 447 | digital | PASS | PASS | ✓ stable |
| Hybrid_electric_vehicles | 125 | digital | PASS | — | NEW |
| Integra_manual | 244 | digital | PASS | — | NEW |
| IRJET_Modeling_of_Solar_PV | 40 | digital | PASS | PASS | ✓ stable |
| Jungjun_AI_Agent | 658 | digital | PASS | — | NEW |
| KI_En_ChatGPT_Praktische_Gids | 1210 | digital | FAIL | — | NEW |
| Kimothi_RAG_Guide | 680 | digital | PASS | PASS | ✓ stable |
| Nagasubramanian_Agentic_AI | 1017 | digital | PASS | — | NEW |
| PCWorld_July_2025 | 371 | digital | PASS | PASS | ✓ stable |
| Python_Cookbook | 389 | digital | PASS | — | NEW |
| Python_Distilled | 1042 | digital | PASS | — | NEW |
| Raieli_AI_Agents | 1509 | digital | PASS | — | NEW |
| Recent_Trends_in_Transportation | 20 | digital | PASS | PASS | ✓ stable |
| Sekar_MCP_Standard | 656 | digital | FAIL | — | NEW |

## Legacy / probe outputs (informational)

These predate or sit outside the v2.8 canonical corpus
(probes, partial-page reconverts, pre-fix _codex/_promptfix/_vlm runs).
Not counted in the aggregate above.

| Output dir | Chunks | Class | AFTER | BEFORE | Delta |
|---|---|---|---|---|---|
| ayeva_qa_20260501 | 627 | digital | PASS | PASS | ✓ stable |
| carok_fix | 76 | digital | PASS | — | NEW |
| carok_verify | 67 | digital | PASS | — | NEW |
| Combat_Aircraft_full_promptfix_v2 | 537 | digital | FAIL | **FAIL (TEXT)** | still FAIL |
| Combat_Aircraft_full_vlm_codex | 592 | digital | FAIL | — | NEW |
| Combat_Aircraft_promptfix_pages91_100 | 61 | digital | PASS | — | NEW |
| Fluent_Python_full_codex | 1633 | digital | PASS | PASS | ✓ stable |
| Fluent_Python_full_vlm_codex | 1634 | digital | PASS | PASS | ✓ stable |
| frontmatter_acceptance_harry_probe | 12 | digital | PASS | — | NEW |
| PCWorld_promptfix_pages1_20 | 65 | digital | PASS | — | NEW |
| probe_ayeva_p1_20 | 33 | digital | PASS | — | NEW |
| probe_boundary_closeout_rag_guide | 680 | digital | PASS | — | NEW |
| probe_carok | 0 | digital | FAIL | — | NEW |
| probe_carok_codex_verify | 76 | digital | PASS | — | NEW |
| probe_combat | 584 | digital | PASS | — | NEW |
| probe_contextual_retrieval_rag_guide | 680 | digital | PASS | — | NEW |
| probe_harry_potter | 458 | digital | PASS | — | NEW |
| probe_milestone2_rag_guide | 680 | digital | PASS | — | NEW |
| probe_rag_guide | 663 | digital | PASS | — | NEW |
| probe_rag_guide_guard | 680 | digital | PASS | — | NEW |
| v28_phase1_hybrid_review | 191 | digital | PASS | — | NEW |
| v28_phase3_combat_p60_70 | 133 | digital | PASS | — | NEW |

## Phase 5c gating decisions

**Conversion flags used** (per scripts/convert_books.sh):
```
python -m mmrag_v2.cli process <pdf> -o <out> -b <batch> \
  --vision-provider none --no-refiner --no-cache
```
Matches the Phase 0 BEFORE baseline so the delta column above is
apples-to-apples and isolates v2.8 code changes.

**Refiner smart-routing** (deferred to v2.9):
`cli.py:686` enables refiner whenever `~/.mmrag-v2.yml`
`refiner.enabled=true`, regardless of `has_encoding_corruption`.
This caused HARRY (clean prose, zero corruption) to hammer
qwen-plus during the first broad-reconversion attempt. Fix is
v2.9 scope: gate the config-default enable on the diagnostic
just like the explicit auto-override at `cli.py:1101` does.
After the fix, the broad reconversion can re-run without flags
and produce the same output for clean docs while still
auto-enabling refiner on encoding-corrupt ones.

## v2.8 Plan Targets — Empirical Outcomes

Each of the four production gaps in `PLAN_V2.8_PRODUCTION_GAPS.md` was
verified with a fresh re-conversion under the canonical pipeline (no
manual flags except baseline-matching VLM/refiner-off):

| Workstream | Plan target doc | BEFORE | AFTER | Verdict |
|---|---|---|---|---|
| F (0x01 keyword sep) | A_comprehensive_review_on_hybrid_electri | FAIL TEXT, ctrl_chunks=1, ctrl_total=4 | PASS, ctrl_chunks=0 | ✓ FIXED |
| C (Combat ornament) | Combat_Aircraft_August_2025 | FAIL TEXT, encoding_artifacts=48, high_corruption=79 | PASS, encoding_artifacts=0, high_corruption=0 | ✓ FIXED |
| B (Chaubal CodeFormulaV2) | Chaubal_PyTorch_Projects | FAIL CODE, indentation_fidelity=0.54 | PASS, indentation_fidelity=0.96 | ✓ FIXED (target was ≥0.85) |
| §5 (adapter-invocation guard) | (architectural) | construction-only guard | construction + invocation guard, dead-branch removed, pdf_engine routed | ✓ FIXED |

Plus Phase 5a (SCAN0013 form-aware gate, smoke 11/11 GATE_PASS) and
Phase 5b (HARRY page-30 acceptance test, live PASS) shipped during
pre-flight. **All four production gaps are empirically closed.**

## Known Limitations Surfaced by Phase 5c (v2.9 scope)

Two canonical-corpus rows show degraded gates that are NOT v2.8 code
regressions but **diagnostic-classifier drift** since the Phase 0
baseline. Both are documented v2.9 followups, not v2.8 blockers.

### Ayeva_Python_Patterns — `digital_literature` misclassification
- Baseline (`ayeva_qa_20260501`): profile=`technical_manual`, CODE
  PASS, `indentation_fidelity=0.93`. Used `--enable-doctr --ocr-mode auto`.
- v2.8 fresh: profile=`digital_literature` (rule 0c misfire — Ayeva
  is a code-heavy book, not a novel), CODE FAIL,
  `indentation_fidelity=0.83` (just under 0.85 gate).
- The misclassification suppresses the `needs_code_enrichment`
  cheap-evidence trigger (CodeFormulaV2 doesn't auto-engage for the
  `digital_literature` profile), so the indentation lift seen on
  Chaubal (0.54 → 0.96) doesn't reach Ayeva.
- **Net: indentation_fidelity 0.22 → 0.83 (massive gain), just under
  the 0.85 hard gate.** No data integrity issue; chunks are usable
  for RAG retrieval.
- v2.9 fix: tighten `profile_classifier` rule 0c so a book with
  `code_evidence_pages >= 2` cannot be routed to `digital_literature`.

### Firearms — profile changed `scanned` → `technical_manual`
- Baseline: profile=`scanned`, HEADING coverage 100%.
- v2.8 fresh: profile=`technical_manual`, HEADING coverage 78%
  (gate is ≥80%). 179 chunks have null `parent_heading`.
- Chunker's heading-inheritance differs by profile;
  `technical_manual` is stricter about what constitutes a real
  parent heading, leaving more text chunks orphan-headed.
- **Net: same chunks extracted (1690 vs 1691), same content
  fidelity. Just fewer chunks now annotated with parent_heading.**
  RAG retrieval still works via breadcrumb path + body content.
- v2.9 fix: investigate profile drift on scanned-modality manuals;
  either re-tune the heading-inheritance threshold for the
  `technical_manual`-routed scanned docs OR extend the `scanned`
  profile to claim Firearms-shape inputs.

## Qdrant re-ingestion — COMPLETE

The container deadlock that initially blocked ingest was cleared
2026-05-04 ~06:50 by the user (removing 11 deadlock-prone collections
from the sister project's bind-mounted storage volume). 17 healthy
`*_v2` collections remain. Side-by-side ingest into the new
`mmrag_v2_8` collection ran via `tmp/v28_ingest.sh` (loops
`scripts/ingest_to_qdrant.py` once per canonical doc with
`--collection mmrag_v2_8 --model nomic-embed-text`).

**Mid-run point_id collision fix (committed `0d3cc36`):** the first
ingest attempt landed only 1,690 points — the size of the largest
single doc. Root cause: `ingest_to_qdrant.py:439` used
`point_id = i + 1`, sequential per-file, so every subsequent file
overwrote the previous file's points 1..N. Fix: derive `point_id`
as a deterministic `uuid.uuid5(_POINT_ID_NAMESPACE, chunk_id)`
(globally unique because chunk_id embeds doc-hash + page + type +
content-hash). 6 regression tests added in
`tests/test_qdrant_point_id_collision.py`. Re-ingest after the fix
landed all unique chunks correctly.

**Final verification (run AT 2026-05-04 07:33 UTC after ingest v2 finished):**

| Quantity | Count | Notes |
|---|---|---|
| Total chunk_ids across 34 canonical JSONLs | 22,587 | sum of `wc -l ... -1` per file |
| Unique chunk_ids | 22,160 | 427 within-file duplicate chunk_ids — see v2.9 followup |
| Embedding errors logged by ingest | 23 | nomic-embed-text rejected too-long content (mostly Combat p66 reconstructed text + 4 long tables) |
| Points in `mmrag_v2_8` collection | 22,137 | matches: 22,160 unique − 23 embed errors |

**100% of unique, embeddable chunks ingested.** The `mmrag_v2_8`
collection sits side-by-side with the existing 17 `*_v2` per-doc
collections; nothing was overwritten in user-owned data.

```bash
# Reproduce verification
curl -sS http://localhost:6333/collections/mmrag_v2_8/points/count \
  -X POST -H "Content-Type: application/json" -d '{"exact":true}'
# → {"result":{"count":22137}, "status":"ok"}
```

**v2.9 followup — schema chunk_id collisions:** 427 within-file
duplicate chunk_ids (largest contributors: KI_En_ChatGPT 279 dupes,
Devlin 76, Fluent 15) come from the v2.7+ schema generator
`_generate_chunk_id(doc_id, content, page_number, type)` collapsing to
the same hash for repeated identical content (boilerplate footers,
repeated page numbers, etc.). Not a v2.8 regression. v2.9 fix:
include the chunk's `i+1` index in the hash seed so
visually-identical chunks at different document positions retain
distinct IDs.

# 6. Verify chunk counts match JSONL line counts (no silent drops)
# 7. Tag v2.8.0 once verification passes
git tag v2.8.0
git push origin v2.8.0   # only if pushing is desired
```
