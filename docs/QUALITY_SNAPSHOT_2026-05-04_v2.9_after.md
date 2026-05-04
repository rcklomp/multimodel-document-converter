# Quality Snapshot 2026-05-04 — v2.9 AFTER

> **Status:** template — populated as Phase 5 sub-phases complete.
> The numeric fields below are placeholders ({{TBD}}) until each
> sub-phase lands its evidence.

**Purpose:** AFTER state for the v2.9 broad reconversion + cloud-VLM
enrichment + Qdrant `mmrag_v2_8` drop-and-recreate. Compare against
[`docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md`](QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md)
for the v2.9 BEFORE column.

**HEAD on tag:** `{{TBD — set on tag commit}}`

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
- **{{TBD}}/34 PASS** (target: 34/34)
- **{{TBD}} FAIL** (target: 0)
- `Form_0013_invoice` reports `FORM_AUDIT_PASS` per
  `docs/QUALITY_GATES.md` "Form / Invoice Acceptance Class".

## v2.9 Plan Targets — Empirical Outcomes

The four v2.8 carry-overs from `docs/PLAN_V2.9.md` §1, verified against
the v2.9 broad reconversion under the canonical pipeline (no manual
flags except `--vision-provider none`; `--no-refiner` dropped per
Phase 2):

| Workstream | Plan target doc | v2.8 BEFORE | v2.9 AFTER | Verdict |
|---|---|---|---|---|
| Phase 1 — chunk_id collision | full corpus | 22,587 chunks → 22,160 unique (427 dupes) | {{TBD chunks → TBD unique}} (target: 0 dupes) | {{TBD}} |
| Phase 2 — refiner smart-routing | HARRY + Combat | HARRY refined despite zero corruption (workaround `--no-refiner`) | refiner OFF on HARRY, refiner ON on Combat (no flags) | {{TBD}} |
| Phase 3 — Ayeva profile route | Ayeva_Python_Patterns | profile=digital_literature, indentation_fidelity=0.83 (CODE FAIL) | profile=technical_manual, indentation_fidelity {{TBD}} | {{TBD}} |
| Phase 4 — Firearms heading | Firearms | profile=technical_manual, HEADING coverage 78% (FAIL) | profile=scanned, HEADING coverage {{TBD}} | {{TBD}} |

Plus the Phase 5b targeted image-only VLM enrichment via
`scripts/enrich_image_chunks_v29.py` against cloud `qwen3-vl-plus`:

- placeholder image chunks before enrichment: {{TBD ~5,200}}
- placeholder image chunks after enrichment: {{TBD 0}}
- `vision_status="hard_fallback"`: {{TBD}}
- per-image cost (DashScope qwen3-vl-plus): {{TBD record actual on completion}}

## Per-document AFTER (canonical v2.9 corpus)

> Filled in after Phase 5a audit. Columns mirror the v2.8 AFTER
> snapshot (Output dir / Chunks / Class / AFTER / v2.8 AFTER / Delta).

| Output dir | Chunks | Class | v2.9 AFTER | v2.8 AFTER | Delta |
|---|---|---|---|---|---|
| {{TBD per row}} | | | | | |

## Qdrant `mmrag_v2_8` drop-and-recreate

Pre-drop verification (recorded at the time of the destructive op):

- Last-write timestamp on collection: {{TBD}}
- Read traffic in prior 24h: {{TBD `docker logs --since 24h | grep mmrag_v2_8`}}
- External `mmrag_v2_8` references in `$HOME/Projects`: {{TBD grep result}}
- Decision: {{TBD drop-and-recreate vs. fallback to side-by-side `mmrag_v2_9`}}

Post-recreate verification:

| Quantity | Count | Notes |
|---|---|---|
| Total chunk_ids across 34 v2.9 canonical JSONLs | {{TBD}} | sum of chunk records |
| Unique chunk_ids | {{TBD == total}} | Phase 1 fix → 0 within-file dupes expected |
| Embedding errors logged by ingest | {{TBD}} | nomic-embed-text long-content rejections |
| Points in `mmrag_v2_8` collection | {{TBD}} | should equal unique − embed-errors |

```bash
curl -sS http://localhost:6333/collections/mmrag_v2_8/points/count \
  -X POST -H "Content-Type: application/json" -d '{"exact":true}'
# → {"result":{"count":{{TBD}}}, "status":"ok"}
```

Image-side retrieval spot-check (Source-Sanctity-safe prompts):

- HARRY wizard ornament: {{TBD hit on a digital_literature image point}}
- Combat F-35 photo: {{TBD hit on a magazine image point}}

## Known Limitations carried into v2.10

(Same `## Known Limitations` shape as v2.8 AFTER; populate when
Phase 5 closes.)

- Local NuMarkdown-8B-Thinking-mlx-8bits VLM lane: deferred to v2.10
  while the off-network endpoint is unreachable. Cloud `qwen3-vl-plus`
  is the v2.9 default.
- Remote CodeFormulaV2 inference: docling 2.86 does not yet expose
  `RemoteCodeFormulaOptions` / `ApiCodeFormulaOptions`. v2.9 still uses
  client-local CPU inference (~27 s/page on Apple Silicon target).
- Broader UIR refactor (canonical PdfConversionPlan → UniversalDocument
  → ElementProcessor → chunks): v2.10 scope per CLAUDE.md.
- Magazine image quality (rendered-region-crop architecture): not a
  v2.9 blocker.
- HybridChunker per-item token guard: requires upstream Docling work.

## Tag

`v2.9.0` annotated tag set on the AFTER-snapshot commit when the six
Completion Rules from `docs/AGENT_GOVERNANCE.md` are satisfied. See
`docs/PLAN_V2.9.md` §4 "Tag criteria for `v2.9.0`".

```bash
git tag -a v2.9.0 -m "v2.9.0 — close v2.8 carry-overs, cloud-VLM-enriched mmrag_v2_8"
```
