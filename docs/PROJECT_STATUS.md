# Project Status

Last updated: 2026-05-09

Purpose: fast orientation for a new coding session. Read this before deeper project docs.

## Current Objective

**v2.9 IN PROGRESS — NOT SHIPPED.** A v2.9.0 tag was created on
2026-05-05 against a 32/34 AUDIT_PASS reading from
`scripts/qa_conversion_audit.py` alone, then deleted on 2026-05-06
after a user-driven review surfaced multiple defects that the
single-script gate did not catch (HARRY chapter-intro pages
silently merged into adjacent pages; Combat p4 lost full-page
imagery; Combat p66 emitted 73 byte-equal corrupted-table copies;
Phase 5b enrichment never updated the canonical ``content`` field).

The v2.9 cycle has landed real bug fixes on `main` (see "v2.9 in-flight
fixes" below) and adopted a stricter four-gate acceptance via
``scripts/qa_full_conversion.py`` (see ``docs/TESTING.md``). Under
that gate the post-enrichment corpus reports
**5 PASS / 3 WARN / 26 FAIL out of 34**. v2.9 is not eligible for
shipping until the remaining failure modes are closed (see "Open
work" below).

### v2.9 in-flight fixes (committed)

- **Phase 2 closed (2026-05-08, verification-only)** — re-verified the
  four shipped v2.9 fixes under the strict gate after Phase 1 outputs.
  All five contract checks green: chunk_id uniqueness across 5,749
  chunks (0 dupes); HARRY refiner-suppressed (`refinement_applied=false`
  on all 651 text chunks) AND HARRY postprocessor acceptance fixture
  (2 passed, not skipped, with `HARRY_ACCEPTANCE_JSONL=...`); Combat
  refiner-engaged (109 refined chunks, 0 edit-ratio spam, smart-route
  fired); Ayeva CodeFormulaV2 lane (`code_indentation_fidelity=0.9693`,
  `[CODE-ENRICH] Enabled Docling code enrichment` per batch); Firearms
  route-flip verified (`profile_type=scanned/scanned_degraded`).
  Smoke baseline 11/11 `GATE_PASS + UNIVERSAL_PASS`. Phase 1
  invariants hold across every conversion (0 SIGALRM, 0
  `recovery_page_coverage`, 0 empty text chunks). See
  `docs/QUALITY_SNAPSHOT_2026-05-08_v2.9_phase2_after.md` and the
  Config Provenance block recorded there.

  *Carried forward to Phase 4* (split decision documented in the
  Phase 2 snapshot): Firearms HEADING coverage 72 % (target ≥ 0.80)
  and chunk-count drift 1690 → 2183 (+29 %). The Phase 4 commit
  `3fbce7a` route-flip mechanism succeeded, but the OCR/shadow lane
  on the new profile emits more chunks per page than the v2.8
  baseline; this is fix work for Phase 4 alongside Combat p66 /
  Adedeji p301 / KI EPUB.
- **Phase 1 closed (2026-05-07, commit `df91061`)** — dense-index page
  router + empty-chunk safety net. `_classify_dense_index_pages`
  trusts Docling's `document_index` label and routes those pages
  around HybridChunker via `MmragChunkingSerializerProvider(skip_pages=...)`;
  a dedicated grid-traversal emitter with two-layer dedup (byte-equal
  cell collapse + entry-boundary regex split) produces
  `extraction_method="hybrid_chunker_pageskip"` chunks with no
  `recovery_page_coverage` and no SIGALRM fires. Three layered guards
  drop empty text chunks (oversize-breaker, finalize stage,
  JSONL-write loop) so the strict-gate `empty_text_chunks` invariant
  holds under full-document conversion. Full Kimothi (258 pages)
  reports `AUDIT_PASS / UNIVERSAL_PASS / HYGIENE_PASS`; Ayeva back-
  index probe reports per-page chars 76–105 % of source PDF text
  layer (closes the prior −30 % token-variance regression). Test
  suite: **628 passed, 14 skipped**.
- **chunk_id collision fix** — per-document monotonic ``position``
  hashed into the chunk-id seed; v2.8's 427 within-file dupes
  collapse to zero on a fresh broad reconversion.
- **Refiner smart-routing** — the config-default refiner only
  auto-enables when the diagnostic engine reports
  ``has_encoding_corruption=True``. ``--no-refiner`` no longer needed
  in ``scripts/convert_books.sh``.
- **Cross-page DocChunk page-coverage split** — Docling's
  HybridChunker emits multi-page chunks; the chunker now emits one
  IngestionChunk per source page so chapter-intro pages aren't lost.
- **Mid-sentence merger and near-duplicate filter are now
  page-scoped** — they no longer reattribute one page's content to
  another or drop legitimate cross-page split copies.
- **CorruptionInterceptor extended to TABLE modality** — Combat p66's
  squadron-roster table is now subject to the same patch+quarantine
  path as text. Long em-dash and ``CS`` filler runs are detected
  alongside CIDFont/uniXXXX/replacement-char patterns.
- **FULL-PAGE-GUARD defers full-page assets when no
  conversion-time VLM** — three sites changed from discard → defer
  with ``vision_status='pending'``. Combat p4 (and similar pages
  whose only content is a full-page image) now produces a chunk that
  Phase 5b enrichment can fill in.
- **Phase 5b enrichment script writes canonical ``chunk.content``** —
  not just ``metadata.visual_description``. Earlier v2.9 attempts
  reported ``image_placeholder_ratio=1.0`` because the canonical
  field was never updated.
- **``scripts/qa_full_conversion.py`` strict gate** — bundles
  audit + universal + hygiene + semantic_fidelity plus deterministic
  page-coverage / dup-excess / corruption / image-quality checks.
  Documented in ``docs/TESTING.md`` as the v2.9 acceptance bar.

### Open work for v2.9

- **~~MISSING_PAGES on TOC / index pages~~** — closed by Phase 1
  (commit `df91061`, 2026-05-07). Broad-doc re-verification across
  the 18 originally-failing docs is folded into Phase 5.
- **~~Phase 2: re-verify shipped fixes under strict gate~~** — closed
  2026-05-08 (verification only). See
  `docs/QUALITY_SNAPSHOT_2026-05-08_v2.9_phase2_after.md`.
- **Phase 3 Steps 1-3 implemented (2026-05-09).** Source Sanctity
  validator hardened across 3 commits surfacing 13 leak classes on
  real qwen3-vl-plus output (`c23d3f6`, `a879e85`, `f224aad`); 604
  image chunks across 3 docs (Hao 252, Adedeji 128, PCWorld 224)
  enriched cleanly at 0 % hard-fallback rate. Asset-complexity
  classifier shipped in `src/mmrag_v2/vision/asset_complexity.py`
  with 12 unit tests. Gate calibration shipped on both
  `qa_full_conversion.py` (`_is_blankish_visual_description`) and
  `qa_semantic_fidelity.py` (`is_placeholder_image_or_table`) with
  the F4 hard-fallback exemption and complexity-aware short-
  description rule, plus 15 regression tests
  (`tests/test_qa_image_gate_calibration.py`). Step 1 baseline doc
  at `docs/QUALITY_SNAPSHOT_2026-05-09_v2.9_phase3_vlm_baseline.md`.
  pytest 645 → 672 passing (+27). **A1 follow-up implemented
  2026-05-09:** `detect_text_reading()` now delegates through a
  module-scope pattern table with `_detect_first_match()` diagnostics
  for the Step 4 retry harness; behavior diff is identical across the
  13 documented leak fixtures and the 604-description Phase 3 corpus,
  benchmark delta +1.3 %, pytest `678 passed, 14 skipped`.
- **Phase 3 Step 4 retry harness shipped (2026-05-09, `649c952`).**
  VLM detail-retry on short-on-complex chunks. 5 of 10 documented
  targets resolved by retry (Hao p35-1, p139, p355, p364; Adedeji
  p227); 5 hard_fallback with `complex_asset_short_response_after_retry`
  sentinel (F4-exempt). Strict gate `IMAGE_DESCRIPTION_UNUSABLE = 0`
  across all 3 enriched JSONLs. Snapshot updated at
  `docs/QUALITY_SNAPSHOT_2026-05-09_v2.9_phase3_vlm_baseline.md`
  §9. Phase 3 closed.
- **Phase 4: localized strict-gate hard failures — closed pending
  two sign-offs (2026-05-09 / 2026-05-10).** Plan and closure
  evidence: `docs/PLAN_V2.9__PHASE4.md` and
  `docs/QUALITY_SNAPSHOT_2026-05-09_v2.9_phase4_after.md`.
  - Step 1 — `qa_full_conversion.py --source-pdf` documented as
    canonical strict-gate command (`611805d`).
  - Step 2 — Adedeji p298-316 source-PDF back-index fallback
    (`afbbaa6`); 0 TABLE_CORRUPTION, 0 LOCALIZED_CORRUPTION at
    full 320-page scale.
  - Step 3 — Combat p66 chunk-level corruption filter at finalize
    boundary (`6719460`); 0 false positives across 3,709 chunks.
  - Step 4 (fix) — cross-batch heading carry-forward (`b429cb5`);
    correct in unit tests + small probe; doesn't apply to Firearms
    (OCR-routed). The Path A profile-scoped HEADING gate relaxation
    (`5e58e6e`) was overfit threshold tuning and was reverted in
    `cbd7fb4`. Firearms HEADING continues to FAIL the strict
    `>= 0.80` gate. **DEFERRED to v2.10 as
    `OCR_PATH_HEADING_PROPAGATION` — sign-off pending.**
  - Step 5 — Adedeji `code_indentation_fidelity` 0.886 → 0.9032
    (cascade win from Step 2; no separate fix needed).
  - Step 6 — KI EPUB structural failures (no pagination, no bbox,
    heavy dedup excess). **DEFERRED to v2.10 as
    `KI_EPUB_EXTRACTION_LANE_REWRITE` — sign-off pending.**
  - Tests: 736 passed, 14 skipped (was 685 at Phase 3 close,
    +51 net new across Phase 4).
- **Open Phase 4 sign-offs (block v2.9 tag).**
  1. `OCR_PATH_HEADING_PROPAGATION` — Firearms HEADING coverage
     0.722 vs 0.80 floor. Defect: OCR/element-by-element path
     doesn't promote Docling section_header items into
     `ContextStateV2.hierarchy_stack` (probe data in
     `PLAN_V2.9__PHASE4.md` Step 4).
  2. `KI_EPUB_EXTRACTION_LANE_REWRITE` — EPUB lane structural
     gaps (acceptance baseline in `PLAN_V2.9__PHASE4.md` Step 6).
- **Qdrant ``mmrag_v2_8`` re-ingest.** The collection currently
  contains v2.8.0 ingest data only; not refreshed for v2.9 because
  v2.9 isn't shippable yet (Phase 5 owns the broad reconversion +
  Qdrant migration once both Phase 4 sign-offs land).

## Active Baseline

- **`docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md`** (active baseline —
  v2.8.0 SHIPPED state; 30/34 canonical PASS under the v2.8-era
  audit-only gate. The "superseded" banner that was added during
  the v2.9 attempt has been removed since v2.9 has not shipped.)
- **`docs/QUALITY_SNAPSHOT_2026-05-06_v2.9_strict_gate.md`** (working
  snapshot — strict-gate state of the v2.9 in-progress corpus,
  documents PASS/WARN/FAIL per doc with specific failure codes).
- `docs/QUALITY_SNAPSHOT_2026-05-04_v2.9_after.md` REMOVED — that file
  asserted "32/34 PASS" against the loose gate and has been
  superseded by the strict-gate snapshot.
- `docs/QUALITY_SNAPSHOT_2026-05-03.md` (v2.8 Phase 0 BEFORE state).
- `docs/archive/quality_snapshots/...` historical snapshots (preserved).

`tests/test_docling_postprocessor_acceptance.py` (HARRY pages 13-30
reading-order fixture) is the binding regression test.

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

Source of truth: `docs/QUALITY_SNAPSHOT_2026-05-06_v2.9_strict_gate.md`.

**Strict-gate (`scripts/qa_full_conversion.py`) post-enrichment:
5 PASS / 3 WARN / 26 FAIL out of 34.**

The v2.8.0 SHIPPED state remains the active baseline:
30/34 canonical PASS under the older `qa_conversion_audit.py`-only
gate; smoke matrix 11/11; `mmrag_v2_8` Qdrant collection from the
v2.8 ingest. The v2.9 fixes on `main` are real but the corpus does
not yet meet the strict-gate ship criteria, and the v2.9.0 tag has
been removed.

### v2.9 progress vs ship gate

| Area | v2.8 baseline | v2.9 working state | Strict-gate PASS? |
|---|---|---|---|
| `Ayeva_Python_Patterns` | FAIL CODE, `indentation_fidelity=0.83`, profile=`digital_literature` | profile=`technical_manual`, `indentation_fidelity=0.96` | TOC MISSING_PAGES closed by Phase 1; CODE indentation pending Phase 2 (CodeFormulaV2 lane) |
| `Sekar_MCP_Standard` | FAIL | dedup + reconv lifted indentation | TOC MISSING_PAGES closed by Phase 1; pending broad re-verification |
| chunk_id collisions corpus-wide | 427 within-file dupes | 0 | (gate clean) |
| Refiner smart-routing | HARRY hammered qwen-plus | HARRY `refinement_applied=0`; Combat `refinement_applied=90` | (gate clean) |
| Image enrichment | ~5,500 placeholders | 4,329 enriched / 46 hard_fallback (1.0% corpus rate) | mostly clean; short descriptions still flag a few docs |
| Combat p66 corruption | 73 byte-equal dupes + 40k-char corrupted table | 1 corrupted table chunk; text dupes resolved | FAIL — LOCALIZED_CORRUPTION on the table |
| Combat p4 omission (full-page image only) | page lost | image chunk emitted, enriched | (gate clean) |
| HARRY chapter-intro page silent merge (p29, p54, …) | 13 pages lost | per-page split fixes coverage | WARN only (short VLM responses on a few terse images) |

### Open issues blocking the strict-gate ship

- ~~TOC/index page-drop in 18 docs~~ closed by Phase 1
  (commit `df91061`, 2026-05-07). Broad-doc re-verification deferred
  to Phase 5.
- Combat p66 table chunk still has corrupted typography (em-dash
  run just under threshold).
- Short VLM descriptions (<20 chars) flag a few docs as having
  unusable image descriptions.
- Adedeji p301 table corruption, Devlin / Earthship / Firearms
  doc-specific audit script failures, KI_En_ChatGPT EPUB LABEL
  pre-existing ratio.
- Qdrant `mmrag_v2_8` not refreshed for v2.9 — still v2.8 ingest.

### Already-known followups (not v2.9 scope)

- **Local NuMarkdown-8B-Thinking-mlx-8bits VLM lane.** Cloud-only
  for v2.9 enrichment; re-evaluate when network reachability returns.
- **Remote CodeFormulaV2 inference.** Docling 2.86 still doesn't
  expose `RemoteCodeFormulaOptions` / `ApiCodeFormulaOptions`. v2.9
  uses client-local CPU inference (~27 s/page on Apple Silicon).
- **HybridChunker per-item token guard.** Requires upstream Docling.

## Active Engineering Direction

v2.9 development continues. The 10+ root-cause fixes already on
`main` are kept, plus the Phase 1 dense-index router (commit
`df91061`) which closes the TOC/index page-drop class. Remaining
work to actually tag `v2.9.0`: complete Phase 2 (re-verify shipped
fixes under strict gate, especially Ayeva CodeFormulaV2 +
indentation), Phase 3 (short-VLM-description gate calibration),
Phase 4 (Combat p66 corruption + localized doc audits), then
Phase 5 broad reconversion + refresh `mmrag_v2_8` from the v2.9
corpus.

The broader UIR refactor (canonical PdfConversionPlan →
UniversalDocument → ElementProcessor → chunks per CLAUDE.md) remains
the longer-term direction.

PCWorld VLM evidence remains valid: raw text-reading detections 36.5% → 22.2%, zero measured Combat-style hallucinations, blind-set 87.5% final-valid. See `tests/fixtures/blind_set_manifest.json`.

## Recently Completed

Reverse-chronological. Each entry's evidence files are tracked. The `[Folded into 2.8.0]` items still appear in chronological CHANGELOG entries but the consolidated v2.8 closure is the canonical artifact.

- **PLAN_V2.9 Phase 1 (TOC/index page-loss closure):** `complete` (2026-05-07, commit `df91061`). Dense-index page router via Docling `document_index` label fast path + `MmragChunkingSerializerProvider(skip_pages=...)`; dedicated grid-traversal emitter with two-layer dedup; three layered empty-text-chunk guards (oversize-breaker, finalize stage, JSONL-write loop). Full Kimothi (258 pages) reports `AUDIT_PASS / UNIVERSAL_PASS / HYGIENE_PASS`; Ayeva back-index probe per-page chars 76–105 % of source PDF text (closes prior −30 % token variance). Test suite **628 passed, 14 skipped, 0 failed**. Static `recovery_page_coverage` guard passes. SIGALRM did not fire on any tested document.
- **PLAN_V2.8 (production gaps + broad reconversion + Qdrant re-ingestion):** SHIPPED 2026-05-04. 7 commits on main `5b0e13d → 645ab2b`. Test suite **596 passed, 2 skipped, 0 failed** (at v2.8 ship; current main is 628). Smoke **11/11 GATE_PASS + 11/11 UNIVERSAL_PASS**. Broad reconversion 34/34 PDF/EPUB exit=0. `mmrag_v2_8` Qdrant collection: 22,137 / 22,160 unique embeddable chunks. See `docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md` and `CHANGELOG.md` `[2.8.0]`.
- Post-Docling Sanity Pass + `digital_literature` profile (folded into v2.8): `complete` (2026-05-03, commits `3bdbe0f`, `2f51816`, `379a733`). Reading-order y-sort, drop-cap promotion, label-leak filter, OCR gating, `digital_literature` profile + scorer + strategy.
- Contextual Retrieval (Anthropic approach): `complete` (2026-05-01). Embed-time `build_contextualized_text(...)` with breadcrumb + heading + neighbor context, AGENT-CONTEXTUAL-01..07 invariants, AST-level drift guard, byte-stable `--no-contextual` rollback flag.
- Refactor Boundary Closeout: `complete` (2026-05-01). Removed `BatchProcessor.set_intelligence_metadata` deprecated `[V2.8-COMPAT]` API; added typed-policy round-trip drift insurance test.
- Milestone 2 — Plan Control Plane: `complete` (2026-05-01). `PdfConversionPlan` promoted to typed policy object.
- Milestone 1 — Stabilize Extraction: `complete` (2026-05-01). RAG Guide unblocked, per-element chunker guard. **⚠ Ayeva 0.93 reading from this milestone is from the older probe; v2.8 canonical reads 0.83 FAIL — see `docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md`.**
- Vision-Aided Front Matter / Domain Search Priority / Coordinate Audit: `complete` (2026-04-30). See `docs/archive/quality_snapshots/QUALITY_SNAPSHOT_2026-04-30.md` (banner-annotated as superseded for any specific metric drift; the architectural changes are still active).
- Dependency metadata: Docling exact-pinned to `2.86.0` in `pyproject.toml`; engine version bumped 2.7.0 → 2.8.0 in v2.8 release commit `645ab2b`.

## Immediate Next Work

`docs/PLAN_V2.9.md` is the active plan; the draft prompt that produced
it has been archived (`docs/archive/PLAN_V2.9_DRAFT_PROMPT.md`).

Phase status (per Plan §3 sequence):

| Phase | Scope | Status |
|---|---|---|
| Phase 0 | Establish strict-gate baseline | `complete` |
| Phase 1 | TOC/index page-loss closure | `complete` (2026-05-07, `df91061`) |
| Phase 2 | Re-verify shipped fixes under strict gate | `complete` (2026-05-08, verification only — see `QUALITY_SNAPSHOT_2026-05-08_v2.9_phase2_after.md`) |
| Phase 3 | Resolve `IMAGE_DESCRIPTION_UNUSABLE` | Steps 1-3 complete (2026-05-09); Steps 4-5 active — retry harness + end-to-end |
| Phase 4 | Localized strict-gate hard failures (Combat p66, Adedeji p301, KI EPUB, Firearms HEADING + chunk-count drift carried from Phase 2) | active |
| Phase 5 | Broad reconversion + Qdrant refresh + AFTER snapshot | blocked on Phases 3-4 |

## Must-Respect Constraints

- Python 3.10 only.
- Batch size must stay at or below 10 pages.
- Do not use `--profile-override` for acceptance runs.
- Do not add filename-specific or document-specific quality rules.
- OCR handles text; VLMs describe visuals only.
- BBoxes must remain normalized integer `[0,1000]`.
- Acceptance requires `GATE_PASS` plus `UNIVERSAL_PASS` across the smoke matrix.
