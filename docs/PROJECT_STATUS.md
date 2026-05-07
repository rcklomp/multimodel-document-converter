# Project Status

Last updated: 2026-05-06

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

- **MISSING_PAGES on TOC / index pages.** Affects ~18 of the 26
  QA_FAIL docs. Frontmatter pages 5–12 (TOC, contents, brief
  contents) and back-matter index pages produce zero output chunks.
  Removing the ``DocItemLabel.DOCUMENT_INDEX`` filter at chunker time
  partially helped; a deeper Docling- or pipeline-side filter is
  still dropping the rest. The next phase is the binding
  ``docs/PLAN_V2.9.md`` current Phase 1 contract: re-baseline after removal
  of final-stage recovery, trace one failing page through Docling raw
  → serializer → HybridChunker → final filters, then run targeted
  page-window probes before any broad reconversion or VLM spend.
  Unblocking this brings most of the failed docs to QA_PASS.
- **Combat Aircraft p66.** The squadron-roster table chunk still
  contains the corrupted typography that the OCR-failure regex
  doesn't quite match (em-dash run shorter than the 6-in-a-row
  threshold). Down from 5 corrupt chunks to 1.
- **Short VLM descriptions.** The strict gate's 20-char minimum
  flags terse-but-valid responses like ``"Venn diagram."`` Several
  docs hit single-digit miss counts; either lower the threshold or
  accept short descriptions when ``vision_status="complete"``.
- **Specific-doc audits.** Adedeji p301 table corruption,
  Devlin/Earthship/Firearms each have distinct audit-script
  failures that need case-by-case investigation. KI_En_ChatGPT
  EPUB has a pre-existing LABEL orphan ratio.
- **Qdrant ``mmrag_v2_8`` re-ingest.** The collection currently
  contains v2.8.0 ingest data only; not refreshed for v2.9 because
  v2.9 isn't shippable yet.

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
| `Ayeva_Python_Patterns` | FAIL CODE, `indentation_fidelity=0.83`, profile=`digital_literature` | profile=`technical_manual`, `indentation_fidelity=0.96` | FAIL — MISSING_PAGES on frontmatter (TOC) |
| `Sekar_MCP_Standard` | FAIL | dedup + reconv lifted indentation | FAIL — MISSING_PAGES |
| chunk_id collisions corpus-wide | 427 within-file dupes | 0 | (gate clean) |
| Refiner smart-routing | HARRY hammered qwen-plus | HARRY `refinement_applied=0`; Combat `refinement_applied=90` | (gate clean) |
| Image enrichment | ~5,500 placeholders | 4,329 enriched / 46 hard_fallback (1.0% corpus rate) | mostly clean; short descriptions still flag a few docs |
| Combat p66 corruption | 73 byte-equal dupes + 40k-char corrupted table | 1 corrupted table chunk; text dupes resolved | FAIL — LOCALIZED_CORRUPTION on the table |
| Combat p4 omission (full-page image only) | page lost | image chunk emitted, enriched | (gate clean) |
| HARRY chapter-intro page silent merge (p29, p54, …) | 13 pages lost | per-page split fixes coverage | WARN only (short VLM responses on a few terse images) |

### Open issues blocking the strict-gate ship

- TOC/index page-drop in 18 docs (frontmatter pages 5–12 and
  back-matter index pages produce zero chunks).
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
`main` are kept (chunk_id, refiner routing, cross-page split,
page-scoped dedup/merge, corruption interceptor extension, full-page
defer, enrichment content-field). Remaining work to actually tag
`v2.9.0`: close the TOC/index page-drop, the Combat p66 corruption,
the short-VLM-description gate calibration, and the localized doc
audits — then refresh `mmrag_v2_8` from the v2.9 corpus.

The broader UIR refactor (canonical PdfConversionPlan →
UniversalDocument → ElementProcessor → chunks per CLAUDE.md) remains
the longer-term direction.

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
