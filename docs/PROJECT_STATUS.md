# Project Status

Last updated: 2026-05-04

Purpose: fast orientation for a new coding session. Read this before deeper project docs.

## Current Objective

PLAN_V2.8 SHIPPED (2026-05-04). All four production gaps (Workstreams B, C, F, §5 adapter guard) empirically closed; broad reconversion of every PDF/EPUB in `data/` complete (34/34 conversions exit=0; 30/34 AUDIT_PASS); side-by-side Qdrant ingest into `mmrag_v2_8` collection underway. Two known limitations (Ayeva profile-misclass, Firearms heading regression) carried into v2.9 — neither is a v2.8 code regression; both are diagnostic-classifier drift surfacing as gate failures.

**Active execution: v2.9 planning.** Open items (top of list):
- Refiner smart-routing fix in `cli.py:686` so config-default refiner only auto-enables when `has_encoding_corruption=True`.
- ProfileClassifier rule 0c tightening (Ayeva misroute).
- Heading-inheritance threshold for `technical_manual` on scanned-modality input (Firearms).
- Local VLM swap (Workstream A blocker).

## Active Baseline

The current quality reference point is:

- **`docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md`** (current — v2.8 Phase 5c AFTER state: 30/34 canonical-corpus PASS including 1 FORM_PASS; all four PLAN_V2.8 production gaps empirically FIXED on the named target docs; Ayeva + Firearms documented as diagnostic-classifier drift v2.9 followups).
- `docs/QUALITY_SNAPSHOT_2026-05-03.md` (v2.8 Phase 0 BEFORE state: 30/37 outputs PASS — preserved as the before-column for the 2026-05-04 deltas).
- `docs/QUALITY_SNAPSHOT_2026-05-01.md` (Milestone 1 + 2 closure, RAG Guide unblock, Ayeva re-conversion, contextual retrieval)
- `docs/QUALITY_SNAPSHOT_2026-04-30.md` (Vision-Aided Front Matter, Shared PDF Plan, Coordinate Audit, Domain-Specific Search Priority completion evidence)
- `docs/QUALITY_SNAPSHOT_2026-04-29.md` (pre-Milestone-1 corpus baseline; rows for Ayeva and Harry Potter are now stale and superseded by the entries above)

Use the latest snapshot as the before-state for future comparisons. v2.8 commit chain on `main`: `5b0e13d` (Phases 0-5b code+tests) → `c2e795e` (audit-the-audit fix + overnight pipeline scaffolding) → `9e4b8f8` (raw AFTER snapshot) → `59994f9` (snapshot annotated with empirical Phase outcomes + known limitations + Qdrant resolution). `tests/test_docling_postprocessor_acceptance.py` (HARRY pages 13-30 reading-order fixture) passes live and is the binding regression test.

## Active Model/Endpoint State

Do not print or commit API keys.

Current local VLM setting:

- provider: OpenAI-compatible
- model: `NuMarkdown-8B-Thinking-mlx-8bits`
- base URL: `http://10.0.10.246:8000/v1`

Cloud comparison tested:

- provider: OpenAI-compatible DashScope endpoint
- model: `qwen3-vl-plus`

Observed behavior:

- local NuMarkdown is faster in the PCWorld harness after retry flow but needs many hard fallbacks
- Qwen3-VL-Plus gives richer visual descriptions and fewer hard fallbacks
- both models still read visible text, so model-agnostic enforcement is required

## Current Quality Summary

Source of truth: `docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md`. Aggregate: **30 of 34 canonical-corpus outputs PASS** under `scripts/qa_conversion_audit.py`. Smoke matrix: **11/11 GATE_PASS + 11/11 UNIVERSAL_PASS** (`scripts/smoke_multiprofile.sh`, run 2026-05-04, includes the new `digital_literature` slot for HARRY and the form-lane `scanned/0013_140302111325_001` row).

### Targets that v2.8 closed (read this section instead of stale "Known failures" lists below)

| Doc | BEFORE (Phase 0) | AFTER (Phase 5c, 2026-05-04) | Status |
|---|---|---|---|
| `A_comprehensive_review_on_hybrid_electri` | FAIL TEXT, ctrl_chunks=1, ctrl_total=4 (4× `\x01` keyword sep) | PASS, ctrl_chunks=0 | ✓ FIXED (Phase 1) |
| `Combat_Aircraft_August_2025` | FAIL TEXT, encoding_artifacts=48, high_corruption=79 | PASS, encoding_artifacts=0, high_corruption=0 | ✓ FIXED (Phase 3) |
| `Chaubal_PyTorch_Projects` | FAIL CODE, indentation_fidelity=0.54 | PASS, indentation_fidelity=0.96 | ✓ FIXED (Phase 4 — CodeFormulaV2 engaged) |
| Adapter-invocation static guard | construction-only (`processor.py:2072` bypass possible) | construction + invocation guard, dead-branch removed, pdf_engine routed through adapter | ✓ FIXED (Phase 2) |
| Form acceptance class for SCAN0013 | GATE_FAIL micro_non_label_ratio=0.294 | GATE_PASS [form: ...] / FORM_AUDIT_PASS | ✓ FIXED (Phase 5a) |

### Carry-overs to v2.9 (per `docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md` "Known Limitations")

- `Ayeva_Python_Patterns`: CODE FAIL, `indentation_fidelity=0.83` vs 0.85 gate. Profile mis-routed to `digital_literature` (rule 0c misfire on a code-heavy book); `needs_code_enrichment` cheap-evidence trigger doesn't fire for that profile, so CodeFormulaV2 never engaged. **Net 0.22 → 0.83 is a massive lift; just shy of the hard gate.** v2.9 fix in `profile_classifier.py` rule 0c.
- `Firearms`: HEADING coverage 78% vs 80% gate. Profile changed `scanned` → `technical_manual` between baselines, stricter heading-inheritance leaves 178 chunks orphan-headed. Same content fidelity, just less hierarchy annotation. v2.9 fix MUST respect `AGENT-SPATIAL-20`; preferred path is profile re-routing in `profile_classifier.py` (NOT a per-profile spatial threshold branch).
- Within-file `chunk_id` collisions: 427 across the v2.8 corpus (largest contributor `KI_En_ChatGPT_Praktische_Gids` 279). `_generate_chunk_id` collapses identical content to the same id. v2.9 fix: include chunk's i+1 position in the hash seed.
- Refiner smart-routing: `cli.py:686` config-default ignores `has_encoding_corruption`. v2.8 broad reconversion bypassed via `--no-refiner`; v2.9 fix: gate on diagnostic.
- VLM enrichment of `mmrag_v2_8` Qdrant collection: ~5,500 image chunks have placeholder descriptions because v2.8 used `--vision-provider none` for baseline matching. v2.9 fix: targeted image-only enrichment script + re-ingest just the image points (the v2.8 `0d3cc36` collision-free point_id makes upsert-in-place safe).

## Active Engineering Direction

v2.8.0 SHIPPED (annotated tag `645ab2b` on commit `645ab2b`, pushed to GitHub). v2.9 plan-writing prompt is at `docs/PLAN_V2.9_DRAFT_PROMPT.md`; the plan itself (`docs/PLAN_V2.9.md`) has not yet been drafted.

VLM Source Sanctity enforcement status: **validated-cloud / local-pending**. Cloud `qwen3-vl-plus` validated; local `NuMarkdown-8B-Thinking-mlx-8bits` at `http://10.0.10.246:8000/v1` not yet reachability-verified post-v2.8 (project memory). v2.9 Workstream A blocker.

PCWorld VLM evidence remains valid: raw text-reading detections 36.5% → 22.2%, zero measured Combat-style hallucinations, blind-set 87.5% final-valid. See `tests/fixtures/blind_set_manifest.json`.

## Recently Completed

Reverse-chronological. Each entry's evidence files are tracked. The `[Folded into 2.8.0]` items still appear in chronological CHANGELOG entries but the consolidated v2.8 closure is the canonical artifact.

- **PLAN_V2.8 (production gaps + broad reconversion + Qdrant re-ingestion):** SHIPPED 2026-05-04. 7 commits on main `5b0e13d → 645ab2b`. Test suite **596 passed, 2 skipped, 0 failed**. Smoke **11/11 GATE_PASS + 11/11 UNIVERSAL_PASS**. Broad reconversion 34/34 PDF/EPUB exit=0. `mmrag_v2_8` Qdrant collection: 22,137 / 22,160 unique embeddable chunks. See `docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md` and `CHANGELOG.md` `[2.8.0]`.
- Post-Docling Sanity Pass + `digital_literature` profile (folded into v2.8): `complete` (2026-05-03, commits `3bdbe0f`, `2f51816`, `379a733`). Reading-order y-sort, drop-cap promotion, label-leak filter, OCR gating, `digital_literature` profile + scorer + strategy.
- Contextual Retrieval (Anthropic approach): `complete` (2026-05-01). Embed-time `build_contextualized_text(...)` with breadcrumb + heading + neighbor context, AGENT-CONTEXTUAL-01..07 invariants, AST-level drift guard, byte-stable `--no-contextual` rollback flag.
- Refactor Boundary Closeout: `complete` (2026-05-01). Removed `BatchProcessor.set_intelligence_metadata` deprecated `[V2.8-COMPAT]` API; added typed-policy round-trip drift insurance test.
- Milestone 2 — Plan Control Plane: `complete` (2026-05-01). `PdfConversionPlan` promoted to typed policy object.
- Milestone 1 — Stabilize Extraction: `complete` (2026-05-01). RAG Guide unblocked, per-element chunker guard. **⚠ Ayeva 0.93 reading from this milestone is from the older probe; v2.8 canonical reads 0.83 FAIL — see `docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md`.**
- Vision-Aided Front Matter / Domain Search Priority / Coordinate Audit: `complete` (2026-04-30). See `docs/QUALITY_SNAPSHOT_2026-04-30.md` (banner-annotated as superseded for any specific metric drift; the architectural changes are still active).
- Dependency metadata: Docling exact-pinned to `2.86.0` in `pyproject.toml`; engine version bumped 2.7.0 → 2.8.0 in v2.8 release commit `645ab2b`.

## Immediate Next Work

Follow `docs/PLAN_V2.9_DRAFT_PROMPT.md` to draft `docs/PLAN_V2.9.md`, then execute its phases.

v2.9 priority sequence (per the prompt's §2):

1. VLM enrichment of `mmrag_v2_8` Qdrant collection (highest user impact).
2. Refiner smart-routing fix in `cli.py:686`.
3. ProfileClassifier rule 0c tightening (Ayeva → CodeFormulaV2 recovery).
4. Firearms heading regression (respect `AGENT-SPATIAL-20`; route fix preferred).
5. Within-file chunk_id collision fix in `_generate_chunk_id`.
6. Local VLM comparison (Workstream A — direct dependency for #1).
7. Remote CodeFormulaV2 inference target (only if code-heavy reconversions become routine).

## Must-Respect Constraints

- Python 3.10 only.
- Batch size must stay at or below 10 pages.
- Do not use `--profile-override` for acceptance runs.
- Do not add filename-specific or document-specific quality rules.
- OCR handles text; VLMs describe visuals only.
- BBoxes must remain normalized integer `[0,1000]`.
- Acceptance requires `GATE_PASS` plus `UNIVERSAL_PASS` across the smoke matrix.
