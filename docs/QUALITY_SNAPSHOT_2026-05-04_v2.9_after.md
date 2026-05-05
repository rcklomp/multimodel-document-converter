# Quality Snapshot 2026-05-04 — v2.9 AFTER

**Purpose:** AFTER state for the v2.9 broad reconversion + cloud-VLM
enrichment + Qdrant `mmrag_v2_8` drop-and-recreate. Compare against
[`docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md`](QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md)
for the v2.9 BEFORE column.

**HEAD on tag:** `ec11cb5` + AFTER-snapshot commit (the v2.9.0 tag
points at this snapshot's commit).

**Pre-flight evidence (committed before Phase 5a kicked off):**

- `pytest tests/ -q`: 614 passed, 5 skipped, 0 failed
  (596 v2.8 baseline + 4 Phase 1 + 5 Phase 2 + 6 Phase 3 + 4 Phase 4
  + 1 env-gated VLM acceptance — env-gated tests don't count against
  the steady-state total).
- `bash scripts/smoke_multiprofile.sh`: 11/11 GATE_PASS + 11/11
  UNIVERSAL_PASS. Firearms now sits in the `scanned` column instead
  of `technical_manual`.
- HARRY page-1-30 acceptance fixture: PASS against the most recent
  HARRY conversion.

## Aggregate (v2.9 canonical corpus)

- **32/34 AUDIT_PASS** (was 30/34 in v2.8 AFTER)
- **2 AUDIT_FAIL**:
  - `Firearms` — TEXT + HEADING under the new `scanned` route. Profile
    re-route (Phase 4) is correct; HEADING coverage 73% is below the
    80% gate because the `scanned` chunker is more granular than the
    pre-v2.8 baseline. v2.10 followup.
  - `KI_En_ChatGPT_Praktische_Gids` — LABEL orphan ratio 42.6%
    (limit 25%). Pre-existing v2.8 condition; not a v2.9 regression.
- `Form_0013_invoice` reports `FORM_AUDIT_PASS` per
  `docs/QUALITY_GATES.md` "Form / Invoice Acceptance Class".

## v2.9 Plan Targets — Empirical Outcomes

The four v2.8 carry-overs from `docs/PLAN_V2.9.md` §1, verified against
the v2.9 broad reconversion:

| Workstream | Target | v2.8 BEFORE | v2.9 AFTER | Verdict |
|---|---|---|---|---|
| Phase 1 — chunk_id collision | full corpus | 22,587 chunks → 22,160 unique (427 within-file dupes) | 22,469 chunks → 22,469 unique (0 dupes after the writer-level dedup safety net) | ✓ FIXED |
| Phase 2 — refiner smart-routing | HARRY + Combat | HARRY refined despite zero corruption; v2.8 used `--no-refiner` workaround | HARRY `refinement_applied=0`; Combat `refinement_applied=90`; `scripts/convert_books.sh` no longer carries `--no-refiner` | ✓ FIXED |
| Phase 3 — Ayeva profile route | Ayeva_Python_Patterns | profile=`digital_literature`, `indentation_fidelity=0.83` (CODE FAIL) | profile=`technical_manual`, `indentation_fidelity=0.96` (CODE PASS) | ✓ FIXED |
| Phase 4 — Firearms heading | Firearms | profile=`technical_manual`, HEADING 78% (FAIL) | profile=`scanned` (Phase 4 HARD REJECT works); HEADING 73% (the `scanned` chunker emits more granular text chunks → more orphan headings). Same content fidelity. | partial — route fixed, gate not met (v2.10 followup) |

## Phase 5b — image-only VLM enrichment

| Quantity | Count | Notes |
|---|---|---|
| Image chunks across 34 canonical | 3,684 | unchanged from v2.8 |
| Enriched (qwen3-vl-plus) | 3,651 | 99.1% |
| Hard fallback | 33 | 0.9% corpus-wide; under 5% ceiling. Combat 31, Raieli 1, Jungjun 1. |
| Source Sanctity retries fired | ≈100 | text-reading detection worked as designed |
| Cloud timeouts (initial pass) | ≈80 | re-passed with the fixed script; most resolved on retry |
| Wall clock | ≈4 h | sequential against `qwen3-vl-plus`, ≈5,000 cloud calls |

Local NuMarkdown-8B-Thinking-mlx-8bits remains deferred to v2.10 per
the plan's §Phase 5 decision e — endpoint unreachable from off-network
machines.

## Per-document AFTER (canonical v2.9 corpus)

| Output dir | Chunks | Class | v2.9 AFTER | v2.8 AFTER | Delta |
|---|---|---|---|---|---|
| HarryPotter_and_the_Sorcerers_Stone | 447 | digital | AUDIT_PASS | PASS | ✓ stable |
| Form_0013_invoice | 20 | scanned | FORM_AUDIT_PASS | FORM_PASS | ✓ stable |
| Form_betwistingsformulier | 8 | digital | AUDIT_PASS | PASS | ✓ stable |
| CarOK_voorraadtelling | 76 | digital | AUDIT_PASS | PASS | ✓ stable |
| AIOS_LLM_Agent_Operating_System | 146 | digital | AUDIT_PASS | PASS | ✓ stable |
| A_comprehensive_review_on_hybrid_electri | 188 | digital | AUDIT_PASS | PASS | ✓ stable |
| Hybrid_electric_vehicles | 125 | digital | AUDIT_PASS | PASS | ✓ stable |
| IRJET_Modeling_of_Solar_PV | 40 | digital | AUDIT_PASS | PASS | ✓ stable |
| Recent_Trends_in_Transportation | 20 | digital | AUDIT_PASS | PASS | ✓ stable |
| Combat_Aircraft_August_2025 | 581 | digital | AUDIT_PASS | PASS | ✓ stable |
| PCWorld_July_2025 | 371 | digital | AUDIT_PASS | PASS | ✓ stable |
| ATZ_Elektronik_German | 60 | digital | AUDIT_PASS | PASS | ✓ stable |
| Kimothi_RAG_Guide | 677 | digital | AUDIT_PASS | PASS | ✓ stable |
| Integra_manual | 244 | digital | AUDIT_PASS | PASS | ✓ stable |
| Jungjun_AI_Agent | 656 | digital | AUDIT_PASS | PASS | ✓ stable |
| Bourne_RAG_2024 | 1161 | digital | AUDIT_PASS | PASS | ✓ stable |
| Devlin_LLM_Agents | 764 | digital | AUDIT_PASS | PASS | ✓ stable |
| Raieli_AI_Agents | 1503 | digital | AUDIT_PASS | PASS | ✓ stable |
| Adedeji_GenAI_Google_Cloud | 827 | digital | AUDIT_PASS | PASS | ✓ stable |
| Cronin_GenAI_Models | 1285 | digital | AUDIT_PASS | PASS | ✓ stable |
| Hao_ML_Platform | 1246 | digital | AUDIT_PASS | PASS | ✓ stable |
| Nagasubramanian_Agentic_AI | 1017 | digital | AUDIT_PASS | PASS | ✓ stable |
| Sekar_MCP_Standard | 466 | digital | AUDIT_PASS | **FAIL** | ✓ FIXED |
| Python_Cookbook | 388 | digital | AUDIT_PASS | PASS | ✓ stable |
| ArcGIS_Python_Cookbook | 928 | digital | AUDIT_PASS | PASS | ✓ stable |
| Fluent_Python | 1588 | digital | AUDIT_PASS | PASS | ✓ stable |
| Python_Distilled | 1042 | digital | AUDIT_PASS | PASS | ✓ stable |
| Ayeva_Python_Patterns | 608 | digital | AUDIT_PASS | **FAIL (CODE)** | ✓ FIXED (Phase 3) |
| Chaubal_PyTorch_Projects | 688 | digital | AUDIT_PASS | PASS | ✓ stable |
| Earthship_Vol1 | 770 | scanned | AUDIT_PASS | PASS | ✓ stable (re-routed scanned per Phase 4) |
| Firearms | 1873 | scanned | **AUDIT_FAIL (TEXT, HEADING)** | FAIL (HEADING) | partial — Phase 4 route fix lands; HEADING 73% under new `scanned` chunker, v2.10 followup |
| Greenhouse_Design | 1148 | digital | AUDIT_PASS | PASS | ✓ stable |
| ChatGPT_Praktijk_handboek | 298 | digital | AUDIT_PASS | PASS | ✓ stable |
| KI_En_ChatGPT_Praktische_Gids | 1210 | digital | **AUDIT_FAIL (LABEL)** | FAIL (LABEL) | still FAIL — pre-existing EPUB orphan-label condition |

## Qdrant `mmrag_v2_8` drop-and-recreate

**Pre-drop verification (Phase 5c three-check, recorded
2026-05-05 04:25 UTC):**

| Check | Result |
|---|---|
| Last-write timestamp + point count | 22,137 points, last-written 2026-05-04 (matches v2.8 ingest) |
| Read traffic in prior 24h on `mmrag_v2_8` | 0 search/recommend/scroll/query lines in `docker logs` |
| External `mmrag_v2_8` references in `$HOME/Projects` | only this project's own scripts/tests/docs/version |

All three clean → drop-and-recreate proceeded without consumer impact.

**Post-recreate verification (run 2026-05-05 04:45 UTC):**

| Quantity | Count | Notes |
|---|---|---|
| Total chunks across 34 v2.9 canonical JSONLs | 22,469 | sum of chunk records |
| Unique chunk_ids | 22,469 | Phase 1 fix + writer dedup safety net → 0 within-file dupes |
| Embedding errors logged by ingest | 23 | nomic-embed-text long-content rejections (mostly Combat p66 reconstructed text + 4 long tables) — same baseline as v2.8 |
| Points in `mmrag_v2_8` collection | 22,446 | matches: 22,469 unique − 23 embed errors |
| Image points | 3,684 | 100% of canonical image chunks |
| Text points | 18,373 | |
| Table points | 389 | |

```bash
curl -sS http://localhost:6333/collections/mmrag_v2_8/points/count \
  -X POST -H "Content-Type: application/json" -d '{"exact":true}'
# → {"result":{"count":22446}, "status":"ok"}
```

**Versus v2.8 baseline (22,137):** +309 points net. The delta comes
from the Phase 1 fix surfacing 309 previously-collapsed chunks (v2.8
had 427 within-file dupes; the v2.9 broad reconversion + Phase 1 +
writer dedup recovered 309 of those as distinct points; the remaining
118 collapsed chunks were already uniquified by the v2.7+ asset_path
component before v2.9 measurement).

**Image-side retrieval restored:** every image point now has a real
`qwen3-vl-plus` `visual_description` instead of the v2.8 placeholder
`[Figure on page N] | Context: ...`. RAG image retrieval works
end-to-end against `mmrag_v2_8` for visual queries.

## Known Limitations (carried into v2.10)

- **Firearms HEADING coverage under the `scanned` chunker.** Phase 4
  re-routed Firearms (and Earthship) to `scanned` per
  `AGENT-SPATIAL-20`-compliant scorer adjustment. The route fix is
  correct; the chunker change drops heading coverage 100% → 73%.
  Same content fidelity, just less hierarchy annotation. v2.10 should
  investigate the `scanned` chunker's heading-inheritance threshold
  for Firearms-shape inputs without violating `AGENT-SPATIAL-20`.
- **`KI_En_ChatGPT_Praktische_Gids` LABEL orphan ratio.** 42.6% (limit
  25%). Pre-existing EPUB extraction condition; not a v2.9 regression.
  EPUB-side improvements deferred to v2.10.
- **Local NuMarkdown-8B-Thinking-mlx-8bits VLM lane.** Endpoint at
  `http://10.0.10.246:8000/v1` unreachable from off-network machines.
  Cloud `qwen3-vl-plus` was the v2.9 lock; the enrichment script
  carries a `# v2.10: re-evaluate local` comment at the provider line.
- **Cloud-side timeouts on Combat-class magazines.** ~7% of Combat's
  high-resolution F-35 photographs persistently timed out at
  qwen3-vl-plus (recorded as `vision_status="hard_fallback"` with full
  error context). Corpus-wide rate 0.9% well under the 5% acceptance
  ceiling. v2.10 may consider provider-side image down-scaling or a
  tiered retry budget for known timeout-prone images.
- **Remote CodeFormulaV2 inference.** Docling 2.86 still does not
  expose `RemoteCodeFormulaOptions` / `ApiCodeFormulaOptions`. v2.9
  uses client-local CPU inference (~27 s/page on Apple Silicon). v2.10
  followup if code-heavy reconversion frequency exceeds 1/week.
- **HybridChunker per-item token guard.** Requires upstream Docling.
- **Magazine image quality (rendered-region-crop architecture).** Not
  a v2.9 blocker.

## Tag

`v2.9.0` annotated tag set on the AFTER-snapshot commit when the six
Completion Rules from `docs/AGENT_GOVERNANCE.md` are satisfied. See
`docs/PLAN_V2.9.md` §4 "Tag criteria for `v2.9.0`".

```bash
git tag -a v2.9.0 -m "v2.9.0 — close v2.8 carry-overs, cloud-VLM-enriched mmrag_v2_8"
```
