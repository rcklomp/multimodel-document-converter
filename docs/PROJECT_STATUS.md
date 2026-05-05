# Project Status

Last updated: 2026-05-05

Purpose: fast orientation for a new coding session. Read this before deeper project docs.

## Current Objective

**PLAN_V2.9 SHIPPED (2026-05-05).** Four v2.8 carry-overs closed
(chunk_id collision, refiner smart-routing, Ayeva misroute, Firearms
profile re-route) plus cloud-VLM enrichment of the `mmrag_v2_8`
Qdrant collection (3,651 of 3,684 images via `qwen3-vl-plus`; 33
hard_fallback at 0.9% corpus rate).

- 32/34 canonical AUDIT_PASS (was 30/34 in v2.8). Two remaining FAILs:
  Firearms (HEADING under new `scanned` chunker, v2.10 followup),
  KI_En_ChatGPT_Praktische_Gids (LABEL — pre-existing EPUB condition,
  not a v2.9 regression).
- Smoke matrix: 11/11 GATE_PASS + 11/11 UNIVERSAL_PASS.
- `mmrag_v2_8`: 22,446 unique points (was 22,137); image points
  carry real cloud-VLM descriptions; image-side RAG retrieval
  restored.

## Active Baseline

The current quality reference point is:

- **`docs/QUALITY_SNAPSHOT_2026-05-04_v2.9_after.md`** (current —
  v2.9 Phase 5d AFTER state: 32/34 canonical PASS; all four v2.8
  carry-overs empirically closed except the Firearms HEADING
  regression which is documented as v2.10 followup; cloud-VLM-enriched
  mmrag_v2_8).
- `docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md` (v2.8 Phase 5c
  AFTER — superseded; preserved as the v2.9 BEFORE column).
- `docs/QUALITY_SNAPSHOT_2026-05-03.md` (v2.8 Phase 0 BEFORE state: 30/37 outputs PASS — preserved as the before-column for the 2026-05-04 deltas).
- `docs/archive/quality_snapshots/QUALITY_SNAPSHOT_2026-05-01.md` (Milestone 1 + 2 closure, RAG Guide unblock, Ayeva re-conversion, contextual retrieval)
- `docs/archive/quality_snapshots/QUALITY_SNAPSHOT_2026-04-30.md` (Vision-Aided Front Matter, Shared PDF Plan, Coordinate Audit, Domain-Specific Search Priority completion evidence)
- `docs/archive/quality_snapshots/QUALITY_SNAPSHOT_2026-04-29.md` (pre-Milestone-1 corpus baseline; rows for Ayeva and Harry Potter are now stale and superseded by the entries above)

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

Source of truth: `docs/QUALITY_SNAPSHOT_2026-05-04_v2.9_after.md`.
Aggregate: **32 of 34 canonical-corpus outputs PASS** under
`scripts/qa_conversion_audit.py`. Smoke matrix: **11/11 GATE_PASS +
11/11 UNIVERSAL_PASS** (`scripts/smoke_multiprofile.sh`).
`mmrag_v2_8` Qdrant collection: 22,446 unique points, 3,684 image
points carrying real cloud-VLM descriptions.

### Targets that v2.9 closed

| Doc | v2.8 BEFORE | v2.9 AFTER | Status |
|---|---|---|---|
| `Ayeva_Python_Patterns` | FAIL CODE, `indentation_fidelity=0.83`, profile=`digital_literature` | PASS, `indentation_fidelity=0.96`, profile=`technical_manual` | ✓ FIXED (Phase 3) |
| `Sekar_MCP_Standard` | FAIL | PASS | ✓ FIXED (Phase 5a dedup + reconversion) |
| chunk_id collisions corpus-wide | 427 within-file dupes | 0 | ✓ FIXED (Phase 1) |
| Refiner smart-routing | HARRY hammered qwen-plus per chunk despite zero corruption; v2.8 used `--no-refiner` workaround | HARRY `refinement_applied=0`; Combat `refinement_applied=90`; `convert_books.sh` runs without `--no-refiner` | ✓ FIXED (Phase 2) |
| `mmrag_v2_8` placeholder image descriptions | ~5,500 images with `vision_status="pending"` | 3,651 enriched / 33 hard_fallback (0.9% corpus rate) via cloud `qwen3-vl-plus` | ✓ FIXED (Phase 5b/5c) |

### Carry-overs to v2.10

- **`Firearms` HEADING coverage under the new `scanned` chunker.**
  Phase 4 re-routed Firearms (and Earthship) to `scanned` per
  `AGENT-SPATIAL-20`-compliant scorer adjustment. The `scanned`
  chunker emits more granular text chunks → HEADING coverage 73%
  (under the 80% gate). Same content fidelity. v2.10 should investigate
  the chunker's heading-inheritance threshold for Firearms-shape
  inputs without violating `AGENT-SPATIAL-20`.
- **`KI_En_ChatGPT_Praktische_Gids` LABEL orphan ratio.** 42.6%
  (limit 25%). Pre-existing EPUB extraction condition; not a v2.9
  regression. EPUB-side improvements deferred.
- **Local NuMarkdown-8B-Thinking-mlx-8bits VLM lane.** Cloud-only
  enforced for v2.9 enrichment per the plan's §Phase 5 decision e;
  re-evaluate when network reachability returns.
- **Cloud-side timeouts on Combat-class magazines.** ~7% of Combat's
  high-resolution F-35 photographs persistently timed out. Recorded
  as `vision_status="hard_fallback"`; v2.10 may consider
  provider-side image down-scaling or a tiered retry budget.
- **Remote CodeFormulaV2 inference.** Docling 2.86 still does not
  expose `RemoteCodeFormulaOptions` / `ApiCodeFormulaOptions`. v2.9
  uses client-local CPU inference (~27 s/page on Apple Silicon).
- **HybridChunker per-item token guard.** Requires upstream Docling.

## Active Engineering Direction

v2.9.0 SHIPPED (annotated tag on the AFTER-snapshot commit). v2.10
scope is open; primary candidates are the Firearms HEADING fix,
local-VLM lane re-instatement when the endpoint is reachable, and
the broader UIR refactor (canonical PdfConversionPlan →
UniversalDocument → ElementProcessor → chunks per CLAUDE.md).

PCWorld VLM evidence remains valid: raw text-reading detections 36.5% → 22.2%, zero measured Combat-style hallucinations, blind-set 87.5% final-valid. See `tests/fixtures/blind_set_manifest.json`.

## Recently Completed

Reverse-chronological. Each entry's evidence files are tracked. The `[Folded into 2.8.0]` items still appear in chronological CHANGELOG entries but the consolidated v2.8 closure is the canonical artifact.

- **PLAN_V2.8 (production gaps + broad reconversion + Qdrant re-ingestion):** SHIPPED 2026-05-04. 7 commits on main `5b0e13d → 645ab2b`. Test suite **596 passed, 2 skipped, 0 failed**. Smoke **11/11 GATE_PASS + 11/11 UNIVERSAL_PASS**. Broad reconversion 34/34 PDF/EPUB exit=0. `mmrag_v2_8` Qdrant collection: 22,137 / 22,160 unique embeddable chunks. See `docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md` and `CHANGELOG.md` `[2.8.0]`.
- Post-Docling Sanity Pass + `digital_literature` profile (folded into v2.8): `complete` (2026-05-03, commits `3bdbe0f`, `2f51816`, `379a733`). Reading-order y-sort, drop-cap promotion, label-leak filter, OCR gating, `digital_literature` profile + scorer + strategy.
- Contextual Retrieval (Anthropic approach): `complete` (2026-05-01). Embed-time `build_contextualized_text(...)` with breadcrumb + heading + neighbor context, AGENT-CONTEXTUAL-01..07 invariants, AST-level drift guard, byte-stable `--no-contextual` rollback flag.
- Refactor Boundary Closeout: `complete` (2026-05-01). Removed `BatchProcessor.set_intelligence_metadata` deprecated `[V2.8-COMPAT]` API; added typed-policy round-trip drift insurance test.
- Milestone 2 — Plan Control Plane: `complete` (2026-05-01). `PdfConversionPlan` promoted to typed policy object.
- Milestone 1 — Stabilize Extraction: `complete` (2026-05-01). RAG Guide unblocked, per-element chunker guard. **⚠ Ayeva 0.93 reading from this milestone is from the older probe; v2.8 canonical reads 0.83 FAIL — see `docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md`.**
- Vision-Aided Front Matter / Domain Search Priority / Coordinate Audit: `complete` (2026-04-30). See `docs/archive/quality_snapshots/QUALITY_SNAPSHOT_2026-04-30.md` (banner-annotated as superseded for any specific metric drift; the architectural changes are still active).
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
