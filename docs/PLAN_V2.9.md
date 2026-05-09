# Plan: v2.9 — Strict-Gate Recovery and `mmrag_v2_8` Ship Contract

**Status:** Draft v2.0 (2026-05-06) — post-retraction recovery plan
**Owner:** ingestion pipeline
**Successor to:** `docs/archive/PLAN_V2.8_PRODUCTION_GAPS.md` (shipped 2026-05-04, tag `v2.8.0` on `645ab2b`)
**Related:** `docs/PROJECT_STATUS.md`,
`docs/QUALITY_GATES.md`, `docs/DECISIONS.md`, `docs/ARCHITECTURE.md`,
`docs/AGENT_GOVERNANCE.md`, `AGENTS.md`, `CHANGELOG.md`,
`docs/archive/PROGRESS_CHECKLIST.md` (historical task log; archived
2026-05-07 — read remaining `PROGRESS_CHECKLIST.md` references in
this plan as pointers to that archived file)

---

## 1. Why this plan exists

**v2.9 thesis (one sentence):** Recover from the retracted v2.9.0
attempt by closing the strict-gate failure classes on the 34-doc
canonical corpus, then refresh `mmrag_v2_8` only after
`scripts/qa_full_conversion.py` proves the corpus is shippable.

The original 2026-05-04 v2.9 plan was a forward roadmap for the
four v2.8 carry-overs (chunk_id collisions, refiner smart-routing,
Ayeva profile routing, Firearms heading regression) plus image
enrichment. Those implementation phases have now landed on `main`,
but the 2026-05-06 strict gate exposed that the corpus still is not
shippable. Treat the earlier phase bodies below as historical
implementation detail and re-verification context, not as fresh work
to start from the top.

### Current Baseline (v2.9 BEFORE)

The active v2.9 BEFORE state is
`docs/QUALITY_SNAPSHOT_2026-05-06_v2.9_strict_gate.md`, not the
2026-05-04 v2.8 AFTER snapshot.

- Strict gate: **5 PASS / 3 WARN / 26 FAIL out of 34**.
- Canonical gate: `scripts/qa_full_conversion.py`, which bundles
  audit + universal invariants + hygiene + semantic fidelity plus
  deterministic page-coverage, duplicate-excess, corruption, and
  image-quality checks.
- The earlier audit-only gate (`scripts/qa_conversion_audit.py`) is
  a sub-check only. It missed the defects that caused the v2.9.0 tag
  retraction and must not be used as the ship bar.
- `mmrag_v2_8` remains the v2.8.0 Qdrant collection at 22,137
  points. It has not been refreshed for v2.9 because v2.9 is not
  shippable.
- Phase 5b enrichment did run against the failed v2.9 corpus:
  **4,329 enriched / 46 hard_fallback**. Short but plausible VLM
  descriptions remain a gate-calibration issue.
- The v2.9.0 tag was created on 2026-05-05 and deleted on
  2026-05-06 after strict-gate review.

### Shipped Code vs Strict-Gate Residuals

| Original phase | Shipped evidence on `main` | Strict-gate residual |
|---|---|---|
| Phase 1 chunk_id collision fix | `eae27e8` shipped; later `ae8b891` added canonical-only corpus scan / dedup safety net | Re-verify 0 within-file duplicates after the next strict-gate reconversion |
| Phase 2 refiner smart-routing | `b1b2f3f` shipped; `scripts/convert_books.sh` no longer needs `--no-refiner` | Re-verify clean-prose byte stability and encoding-corrupt refiner activation under strict gate |
| Phase 3 Rule 0c tightening (Ayeva) | `51f0884` shipped | Ayeva still fails strict gate via `MISSING_PAGES`, not the original profile-route issue |
| Phase 4 Firearms HARD REJECT | `3fbce7a` shipped | Firearms still has `SCRIPT_GATE_FAIL` / image-description issues under strict gate |
| Phase 5 prep and enrichment | `538cf89`, `ae8b891`, `ec11cb5`; enrichment run produced 4,329 enriched / 46 hard_fallback | `MISSING_PAGES`, short VLM descriptions, localized corruption, and doc-specific audit failures still block ship |
| Failed ship attempt | `v2.9.0` tag created 2026-05-05, deleted 2026-05-06 | No tag until strict gate passes and governance evidence is durable |

### What closing all phases achieves

- **34/34 canonical PASS** across `scripts/qa_full_conversion.py`
  with source-PDF page awareness for every PDF and no manual flag
  workarounds.
- Every chunk has a globally-unique `chunk_id` (and therefore a
  globally-unique uuid5 `point_id`). 0 within-file collisions on the
  next broad reconversion.
- `mmrag_v2_8` repopulated from a clean drop-and-recreate, with every
  image point carrying either a real cloud-VLM `visual_description`
  with `vision_status="complete"` or a bounded, documented
  `hard_fallback` state.
- v2.10 inherits a corpus and a vector store with no behavioral debt
  attributable to v2.8.

## 2. Goals & Non-Goals

### Goals (measurable from JSONL / audit / Qdrant counts)

1. `scripts/qa_full_conversion.py output/<v29_run>/<doc>/ingestion.jsonl`
   reports `QA_PASS` or an explicitly allowed form/document-class
   pass variant for **all 34 canonical docs**. `QA_WARN` is not a
   ship state unless the warning class is first promoted into an
   explicit pass variant in `docs/QUALITY_GATES.md` with rationale.
2. `Ayeva_Python_Patterns` v2.9 fresh: `profile_type=technical_manual`
   AND `indentation_fidelity ≥ 0.85` AND CODE PASS.
3. `Firearms` v2.9 fresh: `HEADING coverage ≥ 0.80` AND content
   fidelity unchanged (chunk count within ±2% of v2.8 fresh's 1690).
4. `bash scripts/smoke_multiprofile.sh`: every row `GATE_PASS` +
   `UNIVERSAL_PASS` (no waivers; `GATE_PASS [form: ...]` /
   `FORM_AUDIT_PASS` count as variants only when `document_type=form`
   per `docs/QUALITY_GATES.md` "Form / Invoice Acceptance Class").
5. `pytest tests/ -q` reports the current committed test count or
   higher with **0 failed**. As of the strict-gate recovery cycle,
   the suite is already above the original 615-test planning target;
   do not use the stale 596/615 counts as a success signal.
6. Within-file `chunk_id` duplicates across the 34 canonical JSONLs:
   **0** (was 427 in v2.8 AFTER).
7. `mmrag_v2_8` Qdrant collection contains exactly the unique
   embeddable chunk count of the v2.9 broad reconversion, freshly
   recreated. **0 image points** with `vision_status="pending"`;
   `vision_status="hard_fallback"` remains at or below the
   documented threshold established in the v2.9 cloud pre-flight and
   every hard fallback has a recorded reason.
8. `scripts/convert_books.sh` runs **without** `--no-refiner` and
   produces byte-stable text output for clean-prose docs (HARRY) plus
   refiner-applied output for encoding-corrupt docs (Combat-class).

### Non-Goals (deferred to v2.10 or later — verbatim from prompt §2)

- **Local VLM comparison (Workstream A).** Local
  `NuMarkdown-8B-Thinking-mlx-8bits` at
  `http://10.0.10.246:8000/v1` is unreachable from off-network
  machines (per project memory, confirmed 2026-05-04). Cloud
  `qwen3-vl-plus` is the v2.9 default for all VLM use including
  Priority 1. Re-evaluate when network reachability returns;
  until then, v2.10+ scope.
- **Remote CodeFormulaV2 inference target (Workstream B followup).**
  *Trigger: code-heavy reconversion frequency exceeds 1/week per
  `docs/DECISIONS.md` "Selective Code Enrichment Lane → Amendment
  2026-05-03".* v2.8 accepted client-local CPU CodeFormulaV2 at
  ~27 sec/page for one-off batch. Docling 2.86 does NOT expose
  `RemoteCodeFormulaOptions` / `ApiCodeFormulaOptions` — only the
  inline `CodeFormulaModel` ships. v2.9 documents the trigger
  condition only; if still one-off after v2.9 close, push to v2.10.
- **Adapter-invocation static guard.** Shipped in v2.8 Phase 2
  (`tests/test_pdf_conversion_plan.py::test_no_raw_converter_invocation_outside_adapter`).
  Do NOT re-scope.
- **SCAN0013 form-aware gate.** Shipped in v2.8 Phase 5a. The smoke
  row reports `GATE_PASS [form: micro_non_label + label-orphan
  checks skipped]` / `FORM_AUDIT_PASS`. Documented in
  `docs/QUALITY_GATES.md`. Do NOT re-scope.
- **Qdrant ingest collision-free `point_id`.** Shipped in v2.8
  mid-Phase-5c (`fix(ingest): collision-free point_id`, commit
  `0d3cc36`). 6 regression tests in
  `tests/test_qdrant_point_id_collision.py`. Do NOT re-scope.
- **Broader UIR refactor.** Canonical target per CLAUDE.md but not
  required for v2.9; legacy direct Docling-item-to-chunk path
  remains acceptable as long as it doesn't expand.
- **HybridChunker per-item token guard.** Requires upstream Docling
  work.

## 2b. Parallel-Site Audit (cross-cutting principle)

**Permanent project requirement** since v2.8 (`docs/archive/PLAN_V2.8_PRODUCTION_GAPS.md` §2b).

**Lesson learned 2026-05-03:** A single-site fix is suspect until
parallel call sites in the pipeline are audited. The v2.7 §5 adapter
refactor shipped a *construction* guard but missed
`processor.py:2072`'s raw `self._converter.convert(...)`
*invocation* on a cached converter. Result: half a day where
post-Docling sanity stages were silently bypassed.

**Mandatory step for every production-code phase below:** before
designing a fix, walk the parallel call sites that touch the same
data. The four questions:

1. Does the issue ALREADY have a fix elsewhere in the pipeline that
   the failing data simply hasn't been re-run through? (Compare
   output timestamps to relevant commit dates.)
2. Does the existing fix have too narrow a gate (e.g. fires on
   `\x00` only when the bug surface is `\x01`-`\x1F`)?
3. Are there parallel boundaries (CLI `process` vs CLI `batch`;
   `BatchProcessor` vs `V2DocumentProcessor`; `engines/pdf_engine.py`)
   that need the same change?
4. Is there an upstream library config (Docling, EasyOCR, OcrMac)
   that already addresses the issue without custom code? (Per
   "Libraries first, custom code last" — CLAUDE.md.)

Each phase below has an explicit **Parallel-site audit** table.

## 3. Phases

Current-cycle phases are ordered to avoid another expensive broad
conversion until deterministic extraction loss is closed. The
2026-05-04 Phase 1-4 bodies remain below as shipped implementation
history; their code must be re-verified under the current strict
gate after Phase 1 produces clean outputs.

Current recovery sequence:

| Current phase | Purpose | Status scope |
|---|---|---|
| Phase 0 | Establish current strict-gate baseline from the 2026-05-06 snapshot and current working tree | `complete` |
| Phase 1 | TOC/index page-loss closure contract | `complete` (2026-05-07, commit `df91061`) |
| Phase 2 | Re-verify already-shipped v2.9 fixes under strict gate | `complete` (2026-05-08, verification only — see `docs/QUALITY_SNAPSHOT_2026-05-08_v2.9_phase2_after.md`) |
| Phase 3 | Resolve `IMAGE_DESCRIPTION_UNUSABLE` policy/model behavior — execution plan in `docs/PLAN_V2.9__STEP3.md` | `complete` (2026-05-09, commits `649c952` + `51e897b`) |
| Phase 4 | Resolve localized hard failures (Combat, Adedeji, Devlin, Earthship, Firearms, KI_En_ChatGPT) — Phase 2 added Firearms HEADING coverage and chunk-count drift to scope. Closure evidence in `docs/PLAN_V2.9__PHASE4.md` and `docs/QUALITY_SNAPSHOT_2026-05-09_v2.9_phase4_after.md`. | `closed pending two sign-offs` (2026-05-09 / 2026-05-10): Firearms HEADING (`OCR_PATH_HEADING_PROPAGATION`) and KI EPUB (`KI_EPUB_EXTRACTION_LANE_REWRITE`) both deferred to v2.10 |
| Phase 5 | Broad reconversion, enrichment, Qdrant drop/recreate, AFTER snapshot | blocked until both Phase 4 sign-offs land |

---

### Phase 0 — Establish current strict-gate baseline

**What:** Treat `docs/QUALITY_SNAPSHOT_2026-05-06_v2.9_strict_gate.md`
as the current v2.9 BEFORE and make sure the working tree and docs
all point at the same gate. This replaces the stale 2026-05-04
"v2.8 AFTER is v2.9 BEFORE" framing.

**Steps:**
1. `git status --short` — identify in-flight edits. For the current
   cycle, the first implementation commit should include the
   `batch_processor.py` TOC sanitizer/recovery-removal change plus
   `tests/test_toc_cell_marker_sanitizer.py`, extended by the Phase 1
   contract tests below.
2. Reproduce the strict-gate summary against canonical outputs:
   `scripts/qa_full_conversion.py` over all 34 `output/*/ingestion.jsonl`
   rows, with `--source-pdf` for PDFs where available. Expected
   baseline: **5 PASS / 3 WARN / 26 FAIL** unless a tracked snapshot
   supersedes it. If the result diverges, stop and either commit a
   v2.9 baseline-delta note before Phase 1 begins or refresh the
   strict-gate snapshot. Do not silently proceed on stale numbers.
3. `pytest tests/ -q` — record the current suite count, not the stale
   596/615 planning count. Expected: 0 failed.
4. `bash scripts/smoke_multiprofile.sh` — still required by
   `AGENT-VAL-01`, but it is no longer sufficient as the ship gate.
5. Verify Qdrant is still the unrefreshed v2.8 collection:
   ```bash
   curl -sS http://localhost:6333/collections/mmrag_v2_8/points/count \
     -X POST -H "Content-Type: application/json" -d '{"exact":true}'
   # expected: {"result":{"count":22137}, "status":"ok"}
   ```
   This confirms Qdrant has not been refreshed prematurely; it does
   not prove v2.9 quality.

**Done when:**
- `PROJECT_STATUS.md`, this plan, and the strict-gate snapshot agree
  that v2.9 is `in-progress`, not shipped.
- The current BEFORE is explicitly the 2026-05-06 strict-gate state.
- The first new code commit is scoped to the TOC/index page-loss
  contract, not another broad reconversion.

**Risk:** Low. Read-only verification.

**Estimated effort:** 30 min (commands run; Qdrant container must be
up). No engineering time beyond reconciling any doc drift.

---

### Historical Implementation Phases 1-4 — Shipped, Not Active

The original 2026-05-04 implementation plan for chunk_id collisions,
refiner smart-routing, Ayeva Rule 0c, and Firearms routing has shipped
on `main` (`eae27e8`, `b1b2f3f`, `51f0884`, `3fbce7a`, with Phase 5
prep in `538cf89`, `ae8b891`, `ec11cb5`). These sections are no
longer active implementation steps. Their contracts are re-verified in
current Phase 2 after current Phase 1 produces page-loss-clean outputs.

For detailed historical rationale, use the commit history and the
2026-05-04 decision-log entries in this plan. Do not restart work from
these historical phases.

---

### Phase 1 — TOC/index page-loss closure contract — `complete` 2026-05-07 (commit `df91061`)

> **Closure summary.** Dense-index router via Docling
> `document_index` label fast-path + `MmragChunkingSerializerProvider(skip_pages=...)`;
> dedicated grid-traversal emitter with two-layer dedup (byte-equal
> cell collapse + entry-boundary regex split) producing
> `extraction_method="hybrid_chunker_pageskip"`; three layered
> empty-text-chunk guards. Full Kimothi (258 pages) reports
> `AUDIT_PASS / UNIVERSAL_PASS / HYGIENE_PASS`; Ayeva back-index
> probe per-page chars 76–105 % of source PDF text (closes prior
> −30 % token variance). Test suite **628 passed, 14 skipped**.
> Static `recovery_page_coverage` guard passes; 0 SIGALRM fires
> on any tested document. The phase body below is retained as
> historical execution detail.

**What:** Close the strict-gate `MISSING_PAGES` class before any
new broad conversion, cloud VLM enrichment, Qdrant refresh, tag, or
v2.9 completion claim. This phase exists because the 2026-05-06
strict gate shows `MISSING_PAGES` dominates the remaining 26 FAILs,
and because the working-tree direction deliberately removes the
anti-pattern of final-stage `recovery_page_coverage` reconstruction.

**Scope:** Deterministic extraction/filtering only. Do not spend VLM
tokens here. Run targeted conversions with `--vision-provider none`
and, where valid for the diagnostic, `--no-refiner`.

**Non-negotiable contract:**

- A visible, non-blank TOC/index/content page must emit at least one
  real chunk.
- The pipeline must preserve or sanitize source chunks; it must not
  create synthetic final-stage text-layer recovery chunks.
- No production code may contain or emit `recovery_page_coverage`.
- Gate tuning for short VLM descriptions is out of scope until this
  page-loss class is closed.
- Combat p66 and other localized corruption failures remain separate
  hard failures after this phase.

#### Step 0 — Re-baseline after recovery removal

The first measurement must run on the current working tree after
`_recover_missing_text_layer_pages` has been removed. Previous
numbers may have been masked by that recovery path.

1. Convert Kimothi fresh:
   ```bash
   conda run -n mmrag-v2 python -m mmrag_v2.cli process \
     "data/technical_manual/A Simple Guide to Retrieval Augmented Generation Kimothi A. 2025.pdf" \
     --output-dir output/probe_kimothi_toc_contract \
     --vision-provider none \
     --batch-size 10 \
     --no-refiner \
     --no-ocr
   ```
2. Run strict QA with source-PDF page awareness:
   ```bash
   conda run -n mmrag-v2 python scripts/qa_full_conversion.py \
     output/probe_kimothi_toc_contract/ingestion.jsonl \
     --source-pdf "data/technical_manual/A Simple Guide to Retrieval Augmented Generation Kimothi A. 2025.pdf" \
     --no-require-image-descriptions \
     --allow-warnings
   ```
3. Record only the comparable signal for this phase:
   non-blank `MISSING_PAGES`, page set, chunk counts by page, and
   whether any chunk uses `extraction_method="recovery_page_coverage"`.

**Stop condition:** If the post-removal baseline is worse than the
last probe, diagnose the newly exposed drop first. Do not jump to a
single page fix.

#### Step 1 — Trace one known-failing page through the pipeline

Pick Hao p5 or p6 first, because the strict-gate snapshot already
indicates partial recovery behavior around Hao p5-p10. The unit of
work is not "fix page 250"; it is "find the stage where a visible
TOC/index page disappears."

Trace the page through:

| Stage | Question | Required evidence |
|---|---|---|
| Docling raw document | Does Docling expose text/items for the page? | page-scoped item count and labels |
| post-Docling serializer | Does `engines/docling_serializers.py` keep the page content? | page-scoped serialized item count |
| HybridChunker input | Is `DocItemLabel.DOCUMENT_INDEX` or another label still suppressed before chunking? | item labels by page before chunker |
| HybridChunker output | Does Docling emit DocChunks for the page? | DocChunks by page before MMRAG filtering |
| BatchProcessor final filters | Does final cleanup sanitize/keep the chunks? | IngestionChunks by page and extraction method |

The single most informative log is **HybridChunker DocChunks by page
before any MMRAG filtering**. If the page is absent there, debug
Docling/serializer/input labels. If present there, debug our
filtering/final emission.

Also do a two-minute OCR-route check for Hao p5-p7: confirm the
technical-manual route and OCR/image thresholds are not diverting a
decorated TOC page into a zero-text lane.

#### Step 2 — Cheap targeted page-window probes

Use concrete CLI page-range support; do not run full documents for
this diagnostic loop.

| Shape | Probe | Pages | Why |
|---|---|---:|---|
| front TOC + back index | Kimothi | `1-15,245-255` | Known front TOC and back index failure; current best progress signal |
| front TOC + large back index | Hao | `1-15,496-503` | Exercises both early contents and dense back-matter index |
| code/manual TOC | Python_Cookbook or Fluent_Python | `1-15` | Ensures code-book front matter is not a one-off |

Use:

```bash
conda run -n mmrag-v2 python -m mmrag_v2.cli process <PDF> \
  --pages <range> \
  --output-dir output/probe_<doc>_toc_contract \
  --vision-provider none \
  --batch-size 10 \
  --no-refiner
```

For each probe, require:

- all non-blank requested TOC/index pages have at least one chunk;
- no emitted chunk has `extraction_method="recovery_page_coverage"`;
- chunk counts are page-set deterministic across two Kimothi runs;
- no page-window probe creates new corruption, empty-text, or bbox
  invariant failures.

#### Step 3 — Strengthen tests before wider conversion

The existing unit-level sanitizer test
(`tests/test_toc_cell_marker_sanitizer.py`, currently the starting
point for the in-flight change) is useful but insufficient. Add or
extend tests so this phase has a durable contract:

- static guard in `tests/test_pdf_conversion_plan.py` or an adjacent
  static-guard test: no production file contains
  `recovery_page_coverage` or `PAGE-COVERAGE-RECOVERY`;
- acceptance fixture for at least one real TOC/index page window,
  asserting `pages_with_chunks` includes the expected visible pages;
- sanitizer negative case: ordinary chunks without Docling cell
  markers are not demoted/promoted as TOC/index chunks;
- if an E2E fixture is too expensive for the default test suite,
  make it explicitly env-gated and document the command in
  `docs/TESTING.md`.

#### Step 4 — Mini-matrix before full conversion

Do not use a 5-doc smoke as the only bridge to full reconversion.
Either:

- reconvert all 18 `MISSING_PAGES` docs with `--vision-provider none
  --no-refiner`, or
- cluster by failure shape and run at least two docs per shape
  (`front-only`, `back-only`, `front+back`), then require the unpicked
  docs' stale-output chunk counts to remain within ±2% of the
  re-baseline expectation before expanding.

Promotion criteria for Phase 5:

- strict-gate `MISSING_PAGES` count is 0 on the targeted/clustered
  reconversions, or every residual is a documented, reproducible,
  visibly blank/near-blank source page;
- no `recovery_page_coverage` string exists in production code or
  emitted JSONL;
- Kimothi page-set output is deterministic across two fresh runs;
- short-VLM-description and Combat p66 failures are the only major
  remaining classes, and both are explicitly deferred to their own
  phases;
- `pytest` targeted contract tests pass.

**Done when:** The above promotion criteria are met and the next
run is justified as a broad validation run, not another blind
diagnostic loop.

**Risk:** Medium. The likely bug may sit upstream of MMRAG filtering
inside Docling or its serialized input labels, so this phase may end
with a smaller custom preservation hook or an upstream-library
limitation note. It must not end with synthetic final-stage page
reconstruction.

**Path (b) escape hatch:** If the trace proves Docling drops visible
non-blank TOC/index pages before MMRAG receives recoverable chunks,
stop and get user sign-off before introducing any narrower recovery
path. Any approved recovery must be source-page-aware, limited to
visible non-blank pages, produce a distinct documented status, and
still satisfy the static guard that production code does not emit
`recovery_page_coverage`.

**Estimated effort:** 1-2 h for the trace and targeted probes;
additional time only if the disappearance is inside Docling rather
than our serializer/filter chain.

---

### Phase 2 — Re-verify shipped v2.9 fixes under strict gate — `complete` 2026-05-08

> **Closure summary.** Verification only — no production code edits.
> All five contracts green: chunk_id uniqueness (5,749 chunks / 0
> dupes), HARRY refiner-suppressed + acceptance fixture (2 passed,
> not skipped), Combat refiner-engaged (109 refined chunks, 0
> edit-ratio spam), Ayeva CodeFormulaV2 (`code_indentation_fidelity=0.9693`),
> Firearms profile route-flip verified. Smoke baseline 11/11
> `GATE_PASS + UNIVERSAL_PASS`. Phase 1 invariants hold across every
> conversion. Step 4 used a split acceptance: route-flip mechanism
> verified as Phase 2 contract; HEADING coverage and chunk-count
> drift carried forward to Phase 4. Full evidence in
> `docs/QUALITY_SNAPSHOT_2026-05-08_v2.9_phase2_after.md`.



**What:** After Phase 1 produces page-loss-clean outputs, re-verify
the fixes that already shipped on `main`. This phase is verification,
not re-implementation, unless strict-gate outputs expose a regression.

**Checks:**

- `chunk_id`: for every `output/<doc>/ingestion.jsonl`,
  `len(chunk_ids) - len(set(chunk_ids)) == 0`.
- Refiner smart-routing:
  `HARRY` has `refinement_applied == 0` for all chunks;
  `Combat_Aircraft_August_2025` has `refinement_applied > 0` on
  encoding-corrupt repair candidates; conversion logs do not show
  repeated edit-ratio rejection spam.
- Ayeva: `profile_type == "technical_manual"` AND CodeFormulaV2
  engages AND `indentation_fidelity >= 0.85`.
- Firearms: `profile_type in {"scanned", "scanned_degraded"}` AND
  HEADING coverage `>= 0.80` AND no `AGENT-SPATIAL-20` violation.
- HARRY: `profile_type == "digital_literature"` and
  `tests/test_docling_postprocessor_acceptance.py` passes against
  the new conversion.

**Done when:** The above checks pass under `scripts/qa_full_conversion.py`
outputs generated after Phase 1. Audit-only PASS is not sufficient.

---

### Phase 3 — Resolve `IMAGE_DESCRIPTION_UNUSABLE`

> **Execution plan: `docs/PLAN_V2.9__STEP3.md`** (split out 2026-05-08
> for review-isolation and history clarity). The body below is the
> master-plan summary; the sub-plan has the full step breakdown,
> pre-flights, acceptance signals, and review history.



**What:** Decide and implement the strict-gate handling for terse but
valid VLM responses such as `Venn diagram.` or `Line chart.`. This is
gate/model behavior, not extraction.

**Allowed paths:**

- **Path (a), gate calibration:** if `vision_status="complete"` and
  the image class is simple/non-text-heavy, treat short valid
  descriptions as `WARN`, not `FAIL`. Keep `FAIL` for complex,
  text-heavy, or placeholder-like responses.
- **Path (b), VLM retry:** if the asset is complex or text-heavy,
  re-prompt cloud VLM for a fuller visual description and keep the
  strict failure until the response improves.

**Acceptance:**

- 0 placeholder descriptions.
- 0 pending image chunks.
- `hard_fallback` rate is bounded by the documented v2.9 cloud
  threshold and every fallback has a reason.
- Short complete descriptions are classified consistently by asset
  complexity, not by a flat character count alone.

---

### Phase 4 — Localized strict-gate hard failures — `closed pending two sign-offs` (2026-05-09 / 2026-05-10)

**Closure summary:** Steps 1, 2, 3, 5 shipped; Steps 4 + 6 deferred to v2.10 as
`OCR_PATH_HEADING_PROPAGATION` (Firearms HEADING) and `KI_EPUB_EXTRACTION_LANE_REWRITE`.
Detailed closure plan + evidence in `docs/PLAN_V2.9__PHASE4.md` and
`docs/QUALITY_SNAPSHOT_2026-05-09_v2.9_phase4_after.md`. The original posture
table below is preserved for historical context; closure delta is documented
in the sub-plan.

**What:** Resolve the remaining non-page-loss hard failures after
Phases 1-3:

| Class | Docs | Phase 4 posture |
|---|---|---|
| Known table corruption | Combat Aircraft p66; Adedeji p301 | **Must close.** Known repair/quarantine shape; no ship with corrupted table/text emitted. |
| Likely page-loss overlap | Devlin, Earthship, Firearms script-gate failures; any `PAGE_CHUNK_OUTLIER` coupled to `MISSING_PAGES` | Re-evaluate after Phase 1. If still failing, fix under Phase 4 with a doc-specific investigation but no filename-specific production logic. |
| Pre-existing EPUB label issue | KI_En_ChatGPT EPUB label/universal failure | Deferrable only with explicit user sign-off and snapshot rationale, because the strict-gate snapshot marks it as pre-existing. |
| New residuals | Any remaining localized corruption surfaced by the current strict gate | Triage into one of the above classes before any tag decision. |

**Rules:**

- Do not loosen negative tests or strict-gate assertions to match
  current output.
- Prefer quarantine or upstream extraction fixes over emitting known
  corrupted table/text chunks.
- Keep document-specific investigation local to the failure, but do
  not add filename-specific production logic.

**Done when:** Combat p66 and Adedeji p301 are fixed or quarantined
and pass `qa_full_conversion.py`; page-loss-overlap failures are
re-tested after Phase 1 and either pass or have a concrete fix; the
only allowable residual is the pre-existing KI_En_ChatGPT EPUB label
class with explicit user sign-off and snapshot rationale.

---

### Phase 5 — Broad reconversion + Qdrant migration + VLM enrichment + AFTER snapshot

**What:** With current Phases 0–4 closed, run the broad reconversion that
verifies all shipped code fixes land on real corpus data, drop and
recreate the `mmrag_v2_8` Qdrant collection (historical Phase 1 chunk_id
migration callout), run the **Priority 1 VLM enrichment** of all
~5,500 image chunks via cloud `qwen3-vl-plus`, re-ingest the corpus
clean, and produce the v2.9 AFTER snapshot.

This phase is **not optional**. The Ayeva, Firearms, refiner-routing,
and chunk_id fixes are unverifiable without re-converting the corpus.
Per the prompt's mandatory shape, Phase 5 also runs the chunk_id
migration callout from historical Phase 1 (drop-and-recreate `mmrag_v2_8` to
absorb the new collision-free chunk_ids cleanly).

**VLM choice — locked to cloud `qwen3-vl-plus` only.** Local
`NuMarkdown-8B-Thinking-mlx-8bits` at `http://10.0.10.246:8000/v1`
is unreachable from off-network machines (project memory, confirmed
2026-05-04) and is deferred to v2.10 when network reachability
returns. The v2.9 enrichment script defaults to cloud and **does
NOT branch on local availability**. (May add a
`# v2.10: re-evaluate local` comment at the call site.)

#### Pre-flight checklist

- [ ] Phases 0-4 all merged; no broad-conversion tag or
      completion claim exists before Phase 5 finishes.
- [ ] Phase 1 closed: targeted/clustered TOC-index probes have
      `MISSING_PAGES=0` for non-blank pages, no emitted
      `recovery_page_coverage`, and deterministic Kimothi page sets.
- [ ] `pytest tests/ -q` reports the current committed suite count
      or higher, **0 failed**.
- [ ] `bash scripts/smoke_multiprofile.sh` reports **11/11 GATE_PASS
      + 11/11 UNIVERSAL_PASS** with no waivers (`AGENT-VAL-01`,
      CLAUDE.md "Project Invariants", `docs/QUALITY_GATES.md`).
      `GATE_PASS [form: ...]` for SCAN0013 is the only acceptable
      `[form]` variant; per `AGENT-VAL-01` no other doc may use the
      form lane as a workaround.
- [ ] Alibaba DashScope API key reachable; Source Sanctity
      pre-flight passes. Re-run the PCWorld + Combat-style
      hallucination probes from `tests/fixtures/blind_set_manifest.json`
      against the current cloud endpoint and **establish a v2.9 cloud
      baseline** for text-reading-hit rate, hard-fallback rate, and
      Combat-style hallucination count. The 2026-04-29 PCWorld cloud
      reference numbers (text-reading 22.2%, hard fallbacks 21.4%,
      Combat-style hallucinations = 0; per `docs/PROGRESS_CHECKLIST.md`
      Workstream A) are **reference points, not hard caps for v2.9**
      — the cloud endpoint or model behavior may have drifted since.
      Acceptance gate is "v2.9 pre-flight numbers ≤ 2026-04-29 cloud
      reference + 5 percentage-point cushion, AND Combat-style
      hallucinations strictly = 0." If the pre-flight breaches the
      cushion, **stop** and re-evaluate (regress on prompt? fall
      back to the prior model snapshot? defer Phase 5b to v2.10?)
      before burning cloud spend on the full ~5,500-image run.
- [ ] HARRY pages-1-30 acceptance fixture
      (`tests/test_docling_postprocessor_acceptance.py`) green
      against the most recent live HARRY conversion.

**Parallel-site audit (do this FIRST):**

| Site | File:line | Current behavior | Action |
|---|---|---|---|
| Conversion runner | `scripts/convert_books.sh` | 34 entries; v2.9 prep removed the `--no-refiner` workaround | Keep `--vision-provider none --no-cache`; Phase 5's targeted enrichment script handles VLM, not conversion-time VLM. |
| Image chunk pending status | per-doc `output/<doc>/ingestion.jsonl` | `vision_status="pending"`, `visual_description="[Figure on page N] | Context: <breadcrumb>"`, `refined_content=null` for image chunks | The enrichment script is the consumer; it must read these fields and write back `vision_status="complete"`, real `visual_description`, populated `refined_content`. |
| Ingestion script payload writer | `scripts/ingest_to_qdrant.py:316-317, 399-400` | Pulls `metadata.visual_description` or top-level `visual_description` for the payload `visual_description` field | Confirm the enrichment script writes BOTH locations consistently (or pick one — the schema canonical is `metadata.visual_description`). |
| `point_id` derivation | `scripts/ingest_to_qdrant.py:42, 453` | `uuid5(_POINT_ID_NAMESPACE, chunk_id)` | Historical Phase 1 changed chunk_id values for the 427 affected chunks; re-ingest produces new `point_id`s for those points. Drop-and-recreate cleans this up. |
| Existing `mmrag_v2_8` state | Qdrant 22,137 points | **No production retrieval state has been built up** (per project memory, the collection was just created 2026-05-04). | Drop-and-recreate is safe — no rollback of consumer state needed. Recommended option per the prompt §2 Priority 5 migration consideration. |
| 17 sister `*_v2` per-doc collections | Qdrant containers | Pre-existing user-owned data; out of v2.9 scope | DO NOT TOUCH. Drop only `mmrag_v2_8`. |
| Vision Source Sanctity validator | `src/mmrag_v2/vision/vision_prompts.py`, `src/mmrag_v2/vision/vision_manager.py` | Existing text-reading detection + sanitizer + retry harness | The enrichment script MUST call through the existing `VisionManager` so Source Sanctity rules apply (no text transcription, visual-only prompt, retry on detected text-reading). |
| Embedding rebuild scope | nomic-embed-text via `scripts/ingest_to_qdrant.py` | 23 embed errors logged in v2.8 ingest (mostly Combat p66 reconstructed text + 4 long tables) | Spot-check whether v2.9 changes (refiner routing, chunk_id) shift the embed-error count; document any new errors. |

#### Steps

**5a. Broad reconversion (after current Phases 1-4 pass).**

**Parallelism note:** `scripts/convert_books.sh` runs sequentially
by design — `AGENTS.md` §1.4 caps RAM at ~8 GB and batch size at
≤10 pages, and Docling 2.86's CodeFormulaV2 path is CPU-bound and
already saturates a single core. Running 2-3 docs concurrently could
breach the 8 GB ceiling on the code-heavy docs (Chaubal/Ayeva CPU
+ tensor allocations). v2.9 keeps the sequential posture; if Phase 5
runtime exceeds 2 days, propose a parallelism investigation as a
v2.10 followup with empirical RAM measurements rather than
parallelizing inside this cycle.

1. Run `bash scripts/convert_books.sh` with the existing v2.9 flag
   posture. `--no-refiner` was already removed in `538cf89`; verify
   it remains absent rather than re-doing that change.
2. Per-doc strict gate:
   `python scripts/qa_full_conversion.py output/<v29_run>/<doc>/ingestion.jsonl --source-pdf <matching-source-pdf-when-pdf>`.
   `qa_conversion_audit.py` may still be inspected, but only as a
   sub-check.
3. Targeted verification:
   - **Ayeva:** `profile_type=technical_manual`,
     `indentation_fidelity ≥ 0.85`, CODE PASS.
   - **Firearms:** `profile_type=scanned` (or `scanned_degraded`),
     HEADING coverage ≥ 0.80, chunk count within ±2% of 1690.
   - **HARRY:** `profile_type=digital_literature`, page-13 reading
     order intact (drop-cap heal, no label leak), 0 control chars.
   - **A_comprehensive_review:** `ctrl_chunks=0`.
   - **Combat:** `encoding_artifacts=0`, `high_corruption=0`.
   - **Chaubal:** `indentation_fidelity ≥ 0.85`.
   - **All 34:** within-file chunk_id collision count = 0.
4. If any target fails, **stop**. Do not proceed to 5b/c. Diagnose
   and fix in the corresponding earlier phase.

**5b. Targeted image-only VLM enrichment (Priority 1).**

1. Verify and run the existing `scripts/enrich_image_chunks_v29.py`
   against the post-Phase-1 regenerated corpus. The script shipped
   in `538cf89` and was used for the 2026-05-06 enrichment run
   (4,329 enriched / 46 hard_fallback); Phase 5b is not an authoring
   task unless verification exposes a defect. Required behavior:
   - Iterate `output/<v29_run>/<doc>/ingestion.jsonl`.
   - For each chunk where `modality == "image"` AND
     `vision_status in {"pending", "done"}` AND `visual_description`
     starts with the placeholder pattern `[Figure on page` (or
     `vision_provider_used == "none"`):
     - Resolve the asset path from `chunk["asset_ref"]["file_path"]`.
     - Call `VisionManager.describe_image(asset_path,
       provider="qwen3-vl-plus", prompt=VISUAL_ONLY_PROMPT)`.
     - On success: update `chunk["visual_description"]`,
       `chunk["metadata"]["visual_description"]`,
       `chunk["metadata"]["refined_content"]`,
       `chunk["metadata"]["vision_status"] = "complete"`,
       `chunk["metadata"]["vision_provider_used"] = "qwen3-vl-plus"`,
       `chunk["metadata"]["vision_attempts"]`, and reset
       `vision_error` / `vision_validation_issues` if previously set.
     - On Source-Sanctity rejection (text-reading detected): use
       the existing sanitizer + retry harness (per Workstream A).
     - On hard fallback: write `vision_status="hard_fallback"`,
       record the failure reason, do NOT inflate to "complete".
   - **Verify write-back is atomic-replace, never in-place line edit.**
     JSONL is append-friendly but not random-access-edit-friendly,
     and a crash mid-enrichment must not leave a half-written
     canonical output. Pattern:
       - Stream each input line to `ingestion.jsonl.v29tmp` in the
         same directory (same filesystem, so `os.replace` is atomic
         on POSIX).
       - On every Nth chunk (configurable, default 50), `fsync` the
         temp file so a crash loses ≤ N enrichments, not the whole
         file.
       - When the iteration completes successfully AND the temp
         file's line-count matches the original's line-count AND
         every image chunk's `vision_status` is `complete` or
         `hard_fallback`, `os.replace(tmp_path, original_path)`.
       - On any exception or count-mismatch, leave the temp file in
         place with a `.failed` suffix and exit non-zero. The
         original is untouched.
       - Confirm resume mode: on rerun, detect existing
         `ingestion.jsonl.v29tmp.failed` and continue from the last
         enriched chunk_id (skip already-`complete` entries).
   - Confirm the per-doc dry-run mode (`--dry-run`) reports the
     image-chunk count + estimated cost without making API calls,
     so cost can be verified before the multi-hour run.
   - **DO NOT branch on local-VLM availability.** Hardcode cloud.
     (Add `# v2.10: re-evaluate local NuMarkdown-8B endpoint`
     comment at the provider-selection line.)
2. Per-doc strict gate again:
   `python scripts/qa_full_conversion.py ...` — placeholder ratio for
   image chunks should drop to 0% for docs that completed enrichment,
   and `IMAGE_DESCRIPTION_UNUSABLE` must follow the Phase 3 policy.
3. Run `python scripts/vlm_quality_summary.py
   output/<v29_run>/<doc>/ingestion.jsonl --production` for at
   least the blind-set documents (Greenhouse, etc. — see
   `tests/fixtures/blind_set_manifest.json`).

**5c. Qdrant `mmrag_v2_8` drop-and-recreate.**

1. **Strict-gate proof before destructive action:** the post-Phase-5b
   corpus must have `QA_PASS` for all 34 canonical docs under
   `scripts/qa_full_conversion.py`, or every non-PASS must have
   explicit user sign-off recorded in the AFTER snapshot draft.
   If this is not true, abort 5c.
2. **Concrete consumer-absence verification** (do NOT rely on
   project memory alone before destroying data):
   - Inspect the collection metadata for last-write timestamp and
     point count to confirm it matches the v2.8 ingest snapshot
     (22,137 points; last write 2026-05-04):
     ```bash
     curl -sS http://localhost:6333/collections/mmrag_v2_8 | jq .
     curl -sS http://localhost:6333/collections/mmrag_v2_8/points/count \
       -X POST -H "Content-Type: application/json" -d '{"exact":true}'
     ```
   - Check the Qdrant container logs for any read traffic in the
     prior 24 h (any `GET /collections/mmrag_v2_8/points/search` or
     `/.../recommend` would indicate a consumer):
     ```bash
     docker logs --since 24h <qdrant-container-name> 2>&1 \
       | grep -i mmrag_v2_8 | grep -iE 'search|recommend|scroll|query'
     ```
     Expected: **zero** matches. Any match → stop, identify the
     consumer, get sign-off before dropping.
   - Confirm no `*.py` under the user's broader workspace
     (sibling RAG-adapter projects, scripts/) imports / queries
     `mmrag_v2_8`:
     ```bash
     grep -rn "mmrag_v2_8" "$HOME/Projects" 2>/dev/null \
       | grep -v "MM-Converter-V2.4.1/docs" \
       | grep -v "MM-Converter-V2.4.1/output" \
       | grep -vE "\\.(md|txt|jsonl|log|json):"
     ```
     Expected: only references inside this project's
     scripts/CHANGELOG/snapshots. Any external consumer hit →
     stop and notify.
   - **If any of the three checks is non-empty, abort 5c.**
     Re-evaluate: side-by-side ingest into `mmrag_v2_9` (creates
     a parallel collection; old consumers keep working; new
     consumers point at the new one) becomes the fallback.
3. Drop the collection (only after steps 1-2 pass):
   ```bash
   curl -X DELETE http://localhost:6333/collections/mmrag_v2_8
   ```
4. Re-create with the v2.8 schema (vector dim, distance metric —
   confirmed in `scripts/ingest_to_qdrant.py`).
5. Re-ingest the v2.9 corpus:
   ```bash
   bash tmp/v29_ingest.sh   # (loop scripts/ingest_to_qdrant.py once per canonical doc)
   ```
   Use `--collection mmrag_v2_8 --model nomic-embed-text` (same as
   v2.8). The new chunk_ids from historical Phase 1 produce new uuid5
   `point_id`s; the re-ingest is a clean populate.
6. Verify:
   ```bash
   curl -sS http://localhost:6333/collections/mmrag_v2_8/points/count \
     -X POST -H "Content-Type: application/json" -d '{"exact":true}'
   # expected: count == (sum of unique chunk_ids across 34 v2.9 JSONLs) − (embed errors)
   ```
7. Spot-check image retrieval — query with a Source-Sanctity-safe
   prompt for a known image (e.g. wizard ornament from HARRY,
   F-35 photo from Combat) and confirm a hit on the corresponding
   image point.

**5d. v2.9 AFTER snapshot.**

1. Create `docs/QUALITY_SNAPSHOT_<v29_ship_date>_v2.9_after.md`
   following the v2.8 AFTER template
   (`docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md`):
   - Per-document audit table (BEFORE / AFTER / Delta).
   - Smoke matrix table.
   - Current Phase 1-4 empirical outcomes (target docs and their before/after
     metrics).
   - Qdrant ingest evidence (chunk count, point count, embed errors).
   - Image enrichment evidence (placeholder ratio before/after,
     blind-set Source Sanctity numbers).
2. Update `docs/PROJECT_STATUS.md` "Active Baseline" pointer to the
   new snapshot.
3. Update `docs/PROGRESS_CHECKLIST.md` — flip Workstream E to
   `[x]` (placeholder image cleanup), update Workstream D Ayeva
   entry to `[x]`, update v2.9 followups list.
4. Update `CHANGELOG.md` `[2.9.0] — <ship_date>` entry: Added /
   Changed / Fixed for each phase.
5. Bump `__engine_version__` to `2.9.0` in `src/mmrag_v2/version.py`.
   Schema version stays `2.7.0` (no chunk-shape change — Phase 1
   changes the chunk_id *value*, not the schema field).
6. Tag `v2.9.0` only when the 6 binding requirements from
   `docs/AGENT_GOVERNANCE.md` "Completion Rules" are satisfied
   (see §4 below for the verbatim list).

**Tests (red→green) — not test code; this phase produces *empirical*
evidence:**

- The historical Phase 1-4 contract tests
  must all pass.
- The corpus-scan parametrized test from historical Phase 1
  (`test_full_corpus_no_within_file_chunk_id_collisions`) un-skips
  with `RUN_CORPUS_SCAN=1` against the v2.9 outputs and asserts 0
  collisions.
- The Firearms verify test from Phase 4
  (`test_firearms_heading_coverage_at_least_80pct_post_fix`)
  un-skips with `RUN_FIREARMS_VERIFY=1` and asserts ≥ 0.80.
- A new acceptance test
  `tests/test_v29_image_enrichment_acceptance.py` (env-gated
  `RUN_V29_VLM_ACCEPTANCE=1`) iterates all v2.9 image chunks and
  asserts: zero placeholder `visual_description`s, zero
  `vision_status="pending"` entries, all
  `vision_provider_used == "qwen3-vl-plus"` (or
  `"hard_fallback"` with a recorded reason — hard-fallback rate
  ≤ the v2.9 cloud baseline established in the Phase 5 pre-flight
  + 5 percentage-point cushion).

**Done when:**
- All 8 Goals from §2 are met empirically.
- v2.9 AFTER snapshot exists and `docs/PROJECT_STATUS.md` "Active
  Baseline" points at it.
- v2.8 AFTER snapshot
  (`docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md`) gets the
  `> ⚠ SUPERSEDED — historical reference only.` banner per the
  `docs/AGENT_GOVERNANCE.md` "Canonicality Rule (added 2026-05-04)".
- `mmrag_v2_8` Qdrant collection contains the v2.9 chunk_count;
  zero placeholder image points, zero pending image points, and
  bounded documented hard fallbacks.
- `v2.9.0` annotated tag on the AFTER-snapshot commit.

**Risk:** Medium. Three runtime cost components dominate; engineering
work is small. Mitigation: do not start 5b until 5a verifies all
target docs; do not start 5c until 5b's blind-set Source Sanctity
numbers match the v2.8 cloud baseline.

**Estimated effort:**
- Engineering: **~4 h** (enrichment script + ingest harness +
  snapshot drafting).
- Conversion runtime: **~1–2 days** (34 docs, of which Ayeva +
  Chaubal will burn ~2.5 hrs each on CPU CodeFormulaV2;
  remainder is faster).
- Cloud-VLM runtime: **~6–10 h** sequential at qwen3-vl-plus's
  observed throughput (~5,500 images × ~5 s/image including
  retry); parallelizable down to ~2 h with rate-limit caution.
- Cloud-VLM spend: **~5,500 × per-image cost** of qwen3-vl-plus
  (record actual on completion).

## 4. Acceptance Gate (whole plan)

The plan is "done" when:

```bash
# 1. Strict full-conversion gate — canonical ship bar
for doc in output/<v29_run>/*/ingestion.jsonl; do
  python scripts/qa_full_conversion.py "$doc" \
    --source-pdf <matching-source-pdf-when-pdf>
done
# expected: QA_PASS for all 34, except explicit documented
# form/document-class variants allowed by docs/QUALITY_GATES.md.
# qa_conversion_audit.py is a sub-check inside this gate, not the bar.

# 2. Smoke matrix — every row GATE_PASS + UNIVERSAL_PASS, no waivers
bash scripts/smoke_multiprofile.sh
# expected: 11/11 GATE_PASS + 11/11 UNIVERSAL_PASS.
# `GATE_PASS [form: ...]` for the SCAN0013 row is acceptable
# (per docs/QUALITY_GATES.md "Form / Invoice Acceptance Class")
# ONLY when document_type=form. Per AGENT-VAL-01 no other row
# may use the form lane as a workaround.

# 3. Full unit suite
pytest tests/ -q
# expected: current committed suite count or higher, 0 failed.
# Do not use the stale 596/615 planning numbers as evidence.

# 4. Recovery-path static guard
pytest tests/test_pdf_conversion_plan.py -q -k recovery_page_coverage
# expected: static guard passes; no production code contains or emits
# recovery_page_coverage / PAGE-COVERAGE-RECOVERY.

# 5. Per-doc audit — sub-check only; no FAIL row in the canonical 34
for doc in output/<v29_run>/*/ingestion.jsonl; do
  python scripts/qa_conversion_audit.py "$doc"
done
# expected: AUDIT_PASS (or FORM_AUDIT_PASS for the SCAN0013-class
# scanned forms) for all 34 canonical rows.

# 6. Universal invariants — sub-check only; zero hard fails on every output
for doc in output/<v29_run>/*/ingestion.jsonl; do
  python scripts/qa_universal_invariants.py "$doc"
done
# expected: UNIVERSAL_PASS on all 34.

# 7. Qdrant point-count verification
curl -sS http://localhost:6333/collections/mmrag_v2_8/points/count \
  -X POST -H "Content-Type: application/json" -d '{"exact":true}'
# expected: count == (unique chunk_ids across 34 v2.9 JSONLs) − (embed errors).

# 8. Image enrichment audit — zero placeholders, bounded fallbacks
python scripts/vlm_quality_summary.py output/<v29_run>/<doc>/ingestion.jsonl --production
# expected for every doc: placeholder_ratio = 0%, vision_provider_used=qwen3-vl-plus
# for every non-fallback image chunk; hard_fallback <= documented v2.9 threshold.

# 9. Tag
git tag v2.9.0
# only after the 6 Completion Rules below are all satisfied.
```

Human-readable quality checks per document:
- **Ayeva:** `profile_type=technical_manual`,
  `indentation_fidelity ≥ 0.85`, CODE PASS.
- **Firearms:** `profile_type=scanned` (or `scanned_degraded`),
  HEADING coverage ≥ 0.80.
- **HARRY:** `profile_type=digital_literature`, page-13 acceptance
  fixture passes against the v2.9 conversion, refiner did NOT fire
  (clean prose).
- **Combat:** `encoding_artifacts=0`, `high_corruption=0`,
  `placeholder_ratio=0%` (real qwen3-vl-plus visual descriptions on
  the F-35 photos), no firearm/bolt/exploded-view hallucinations.
- **A_comprehensive_review:** `ctrl_chunks=0`.
- **Chaubal:** `indentation_fidelity ≥ 0.85`.
- **34/34 canonical:** within-file chunk_id collision count = 0.

### Tag criteria for `v2.9.0` (from `docs/AGENT_GOVERNANCE.md` "Completion Rules" — verbatim)

A workstream may be marked `complete` only when:

1. Every listed acceptance signal is satisfied.
2. Evidence is durable (`tracked` or `snapshot`).
3. Known limitations are documented.
4. Required local/cloud comparisons are completed or explicitly
   removed from scope.
5. `PROJECT_STATUS.md`, `PROGRESS_CHECKLIST.md`, and snapshots
   agree.
6. A fresh coding session can reproduce the claim without chat
   history.

Apply each requirement explicitly to the v2.9.0 tag commit. These are
not satisfied until the evidence exists:

1. Will satisfy when all §2 Goals are met empirically (strict gate +
   Qdrant + smoke).
2. Will satisfy when the v2.9 AFTER snapshot is `tracked` in `docs/`; outputs are
   `local-run` with commands recorded in §3 / `convert_books.sh`.
3. Will satisfy when known limitations are documented in v2.9 AFTER snapshot under
   "Known Limitations" (the deferred-conditional remote
   CodeFormulaV2 trigger; the v2.10 local-VLM swap; any
   Phase-5b hard-fallback image rate above zero).
4. Will satisfy when local VLM comparison is **explicitly removed from v2.9 scope**
   (deferred to v2.10 — see §2 Non-Goals; project memory pin).
5. Will satisfy when `PROJECT_STATUS.md` "Active Baseline" points at the v2.9
   AFTER snapshot; `PROGRESS_CHECKLIST.md` flips updated; the
   v2.8 AFTER snapshot is banner-marked superseded per the
   Canonicality Rule.
6. Will satisfy when a fresh agent reading `PROJECT_STATUS.md` →
   `PROGRESS_CHECKLIST.md` → `AGENTS.md` →
   `docs/QUALITY_SNAPSHOT_<v29>_after.md` reproduces the v2.9
   claim from tracked files alone.

Failure on any of the six = no tag.

## 5. Out of Scope (deferred to v2.10 or later)

| Item | Why deferred | Owner doc |
|---|---|---|
| **Local VLM comparison (Workstream A)** | Local `NuMarkdown-8B-Thinking-mlx-8bits` at `http://10.0.10.246:8000/v1` is unreachable from off-network machines (project memory, confirmed 2026-05-04). v2.9 default is cloud `qwen3-vl-plus` only; the enrichment script does NOT branch on local availability. Re-evaluate when network reachability returns. | `docs/PROGRESS_CHECKLIST.md` Workstream A |
| **Remote CodeFormulaV2 inference target** | *Trigger: code-heavy reconversion frequency exceeds 1/week per `docs/DECISIONS.md` "Selective Code Enrichment Lane → Amendment 2026-05-03".* v2.8 + v2.9 accept client-local CPU CodeFormulaV2 at ~27 sec/page for one-off batch. Docling 2.86 does NOT expose `RemoteCodeFormulaOptions` / `ApiCodeFormulaOptions`; only the inline `CodeFormulaModel` ships. v2.9 documents the trigger only. If still one-off after v2.9 close, push to v2.10. Options when triggered: (a) custom adapter that intercepts `CodeItem`s post-Docling and POSTs to a remote VLM endpoint; (b) wait for upstream Docling. | `docs/DECISIONS.md` "Selective Code Enrichment Lane" |
| **Adapter-invocation static guard** | Shipped in v2.8 Phase 2 (`tests/test_pdf_conversion_plan.py::test_no_raw_converter_invocation_outside_adapter`). Do NOT re-scope. | v2.8 closure |
| **SCAN0013 form-aware gate** | Shipped in v2.8 Phase 5a. Smoke row `GATE_PASS [form: ...]` / `FORM_AUDIT_PASS`. Documented in `docs/QUALITY_GATES.md`. Do NOT re-scope. | v2.8 closure |
| **Qdrant ingest collision-free `point_id`** | Shipped in v2.8 commit `0d3cc36`. 6 regression tests in `tests/test_qdrant_point_id_collision.py`. Do NOT re-scope. | v2.8 closure |
| **Broader UIR refactor** (`PdfConversionPlan` → `UniversalDocument` → `ElementProcessor` flow) | Canonical target per CLAUDE.md but not required for v2.9; legacy direct-to-chunk path acceptable as long as it doesn't expand. | CLAUDE.md "Workstream B Code Enrichment Guardrail" |
| **HybridChunker per-item token guard** | Requires upstream Docling work. | Milestone 1 known limitation in `docs/PROGRESS_CHECKLIST.md` |
| **Magazine image quality (rendered-region-crop)** | Composite page layouts in magazines extract whole; the proper fix is a rendered-region-crop architecture. Not a v2.9 blocker. | `docs/CONVERSION_PROFILES.md` |
| **Automated baseline delta reporter** (`scripts/qa_delta_report.py`) | Useful tooling but not a blocker; manual diff against v2.8 AFTER suffices for v2.9. | `docs/PROGRESS_CHECKLIST.md` Baseline And Tracking |
| **New profile types** | None identified after `digital_literature`. | — |
| **New post-Docling stages** | The v2.8 four-stage pass (reading-order, drop-cap, label-leak, OCR gating) covers the observed Docling 2.86 failure modes. | `docs/archive/PLAN_DOCLING_POSTPROCESSOR.md` |

## 6. Cross-Phase Concerns

**Documentation updates** (one PR per phase, batched into the matching commit):

- `docs/PROGRESS_CHECKLIST.md` — flip `[ ]` items to `[x]` as each
  phase closes; record evidence path + test counts.
- `docs/PROJECT_STATUS.md` — refresh "Active Baseline" pointer when
  Phase 5 sub-step 5d lands the AFTER snapshot.
- `docs/DECISIONS.md` — make sure entries exist for historical
  Phase 1 chunk_id semantics, historical refiner smart-routing,
  historical Firearms route resolution, current Phase 1
  TOC/index page-loss closure, Phase 3 short-description policy, and
  any Phase 4 localized corruption decisions.
- `CHANGELOG.md` — `[2.9.0]` entry summarizing all phases at the
  end (mirrors v2.8 [2.8.0] entry style).
- v2.8 AFTER snapshot
  (`docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md`) gets the
  superseded banner per the Canonicality Rule.
- `AGENTS.md` "Priority TODOs" list updated post-v2.9 to drop the
  closed items; document any historical Firearms
  `AGENT-SPATIAL-20` amendment only if the historical path (b) was
  actually taken.

**Test contract integrity** (per CLAUDE.md, `AGENTS.md` AGENT-TEST-01,
`docs/AGENT_GOVERNANCE.md` "Test Contract Rules"):

- The 7 existing `tests/test_classifier_digital_literature.py` tests
  must remain green through every current phase.
- The 9 existing `tests/test_classifier_fallback.py` tests must
  remain green through every current phase.
- The 6 existing `tests/test_qdrant_point_id_collision.py` tests
  must remain green (the historical schema change must NOT
  break the Qdrant-layer `point_id` derivation contract).
- Workstream B negative tests
  (`tests/test_code_enrichment_decision.py`) are contracts: do not
  loosen.
- `tests/test_docling_postprocessor_acceptance.py` HARRY pages 13-30
  fixture passes through every phase.
- HARRY-class regression: HARRY MUST keep auto-routing to
  `digital_literature`; current Phase 2 re-verifies this after the
  already-shipped Rule 0c and Firearms classifier changes.

**Upstream tracking:**
- HybridChunker per-item token guard remains an upstream-Docling ask.
- Remote CodeFormulaV2 (`RemoteCodeFormulaOptions` /
  `ApiCodeFormulaOptions`) — track Docling release notes for
  appearance.
- Local NuMarkdown-8B endpoint reachability — re-check before
  v2.10 planning.

**Consumer-side warnings (historical Phase 1 chunk_id semantics):**
- `__schema_version__` stays at `2.7.0` because the chunk-shape
  contract is unchanged. **However, `chunk_id` *values* differ
  for the 427 within-file-collision chunks** between v2.8 outputs
  and v2.9 outputs. Downstream consumers that key on `chunk_id`
  for cross-version mapping (RAG adapters that cache vectors
  against chunk_id; sister projects' `*_v2` collections) MUST
  rebuild from v2.9 outputs after v2.9 ships — same-`schema_version`
  is NOT a stability guarantee for chunk_id this cycle.
- The decision-log entry recording this (Phase 1, decision (a))
  also lands in `docs/DECISIONS.md` so consumers can find it
  outside this plan.

**Architecture / classification scope clarity (Phase 4 Firearms):**
- `docs/DECISIONS.md` "Structural Pathology over Semantic
  Profiling (v2.5.0)" rules that **extraction pathway** (use
  digital text / OCR rescue / force OCR) is determined by
  structural integrity flags, not semantic profile. Phase 4 is
  about **profile classification routing** (`scanned` vs
  `technical_manual`), NOT extraction pathway. Firearms's
  structural-integrity flags (`is_scan=true`,
  `has_encoding_corruption=false`, `has_flat_text_corruption=false`)
  remain unchanged across v2.8 and v2.9; the extraction pathway
  must not flip. Phase 4 verifies this by checking that
  `extraction_method` per chunk and the auto-OCR auto-overrides
  in `cli.py:1093-1106` produce the same outputs as v2.8 fresh.

## 7. Effort Summary

| Phase | Engineering estimate | Runtime / external | External dependency? |
|---|---|---|---|
| Phase 0 — Current strict-gate baseline | 30 min | — | Qdrant container up |
| Phase 1 — TOC/index page-loss closure | 1-2 h trace + targeted probes; more if Docling-side | targeted `--pages` conversions | User sign-off only for Path (b) recovery |
| Phase 2 — Re-verify shipped fixes | 1-2 h analysis after Phase 1 outputs | targeted/full strict-gate checks | None |
| Phase 3 — Short VLM description resolution | 1-3 h depending on gate-vs-reprompt path | possible limited cloud retries | Alibaba DashScope if Path (b) |
| Phase 4 — Localized hard failures | 3-8 h depending on corruption/doc-specific failures | targeted page conversions | None |
| Phase 5 — Broad reconversion + drop/recreate + VLM enrichment + AFTER snapshot | ~4 h script + snapshot | ~1–2 days conversion runtime; ~6–10 h cloud-VLM runtime; ~2.5 h CPU per code-heavy doc | Alibaba DashScope API + spend |
| **Total current cycle** | **~8–18 h engineering** before broad run, depending on localized failures | **~2–3 days runtime** once Phase 5 is justified | Cloud VLM is the dominant external dependency; cost recorded on completion |

**RAM ceiling note (`AGENTS.md` §1.4 — ≤8 GB target):** Phase 5 is
the only phase that risks the ceiling. Code-heavy docs (Chaubal,
Ayeva) load CodeFormulaV2 weights on CPU + Docling layout model + a
~10-page batch render in memory simultaneously. v2.8's broad
reconversion did not hit OOM at sequential batch-size=10 (per Phase
5c evidence), but no formal RAM profile exists. Phase 5a should
record peak RSS via `/usr/bin/time -l` (macOS) or `time -v` (Linux)
on the first code-heavy doc; if peak exceeds ~7 GB, document it as
a v2.10 followup rather than retro-fitting parallelism into v2.9.

## 8. Decision Log

- **2026-05-06 v2.0 (decision g)** — v2.9.0 tag retracted.
  The tag created on 2026-05-05 was deleted on 2026-05-06 because
  the audit-only gate missed strict defects: page loss, image/content
  placeholder issues, localized corruption, and script-gate failures.
  Status vocabulary is `in-progress` / `implemented`, not `complete`.

- **2026-05-06 v2.0 (decision h)** — `scripts/qa_full_conversion.py`
  is the canonical v2.9 acceptance gate. `qa_conversion_audit.py`
  remains a sub-check only. The current BEFORE state is
  `docs/QUALITY_SNAPSHOT_2026-05-06_v2.9_strict_gate.md`
  (**5 PASS / 3 WARN / 26 FAIL**), not the 2026-05-04 v2.8 AFTER
  snapshot.

- **2026-05-06 v2.0 (decision i)** — final-stage
  `recovery_page_coverage` reconstruction is rejected as an
  anti-pattern. The current direction is to preserve/sanitize real
  TOC/index chunks before destructive cleanup. The first new code
  commit in this cycle should contain the in-flight
  `batch_processor.py` sanitizer/recovery-removal change plus its
  tests, then extend those tests with the Phase 1 static and
  acceptance guards.

- **2026-05-06 v2.0 (decision j)** — phase order revised around the
  strict-gate failure classes. Current Phase 1 closes TOC/index
  page loss; current Phase 2 re-verifies already-shipped fixes;
  current Phase 3 resolves short VLM descriptions; current Phase 4
  resolves localized hard failures; current Phase 5 is blocked until
  those classes pass strict gate.

- **2026-05-04 v1.0** — Plan ratified for execution. Scope locked
  to the four documented v2.8 carry-overs (Ayeva, Firearms,
  chunk_id, refiner) plus Priority 1 image-only VLM enrichment of
  the `mmrag_v2_8` Qdrant collection. v2.10 deferrals (local VLM,
  remote CodeFormulaV2, broader UIR refactor) explicitly recorded
  in §5. Cheapest-first phase order chosen so that the surgical
  schema/CLI fixes (Phases 1–2) land before the diagnostic
  investigations (Phases 3–4) and before the heavy Phase 5
  reconversion + cloud-VLM runtime.

- **2026-05-04 v1.0 (decision a)** — Phase 1 chunk_id generator
  includes per-document position component. Schema version stays
  `2.7.0` (chunk_id *value* changes, field shape doesn't).
  Migration absorbed via Phase 5c drop-and-recreate of `mmrag_v2_8`
  (no production retrieval state, per project memory). Alternative
  considered: keep old chunk_ids and accept ~427 stale points in
  `mmrag_v2_8`. Rejected because (i) the dupes silently overwrite
  each other, leaving `mmrag_v2_8` non-deterministic, and (ii) the
  drop-and-recreate is cheap given no consumer state.

- **2026-05-04 v1.0 (decision b)** — Phase 2 moves the config-default
  refiner-enable from CLI startup (`cli.py:686`) to the
  intelligence-metadata gate (`cli.py:1093-1106`), so the refiner
  only auto-enables when `has_encoding_corruption=True`. Explicit
  `--enable-refiner` and `--no-refiner` flags continue to win as
  before. Aligns with `docs/DECISIONS.md` "Heal-Over for Encoding
  Corruption".

- **2026-05-04 v1.0 (decision c)** — Phase 3 tightens
  `document_diagnostic.py:1457-1475` Rule 0c with a
  `code_evidence_pages < 2` guard. Threshold chosen empirically:
  Chaubal (≫2), Ayeva (target ≫2 once cheap-evidence runs); HARRY
  (0). Does NOT add document-specific or filename-specific logic —
  the new gate is a numeric threshold on a pre-existing diagnostic
  feature. Compliant with `AGENTS.md` AGENT-VAL-01 + DECISIONS.md
  anti-pattern "Overfitting to specific filenames".

- **2026-05-04 v1.0 (decision d)** — Phase 4 default resolution path
  is **(a) re-route Firearms via `profile_classifier.py` scorer
  adjustment**, NOT a per-profile spatial threshold branch.
  `AGENT-SPATIAL-20` is respected. Path (b)
  (`AGENT-SPATIAL-20` amendment) is gated on path (a) demonstrably
  failing AND user sign-off; do NOT auto-amend.

- **2026-05-04 v1.0 (decision e)** — Phase 5 VLM choice is
  **cloud `qwen3-vl-plus` only**. Local
  `NuMarkdown-8B-Thinking-mlx-8bits` at
  `http://10.0.10.246:8000/v1` is unreachable from off-network
  machines (project memory, confirmed 2026-05-04). The enrichment
  script does NOT branch on local availability. Per
  `docs/AGENT_GOVERNANCE.md` Completion Rule 4, the local
  comparison is **explicitly removed from v2.9 scope** — not
  pending.

- **2026-05-04 v1.0 (decision f)** — Qdrant migration strategy is
  **drop-and-recreate `mmrag_v2_8`**, not side-by-side. Rationale:
  no production retrieval state has been built up post-v2.8 ship
  (per project memory); the chunk_id-collision migration would
  otherwise leave ~427 orphan points pointing at indeterminate
  upsert winners. Drop-and-recreate gives a clean populate at
  zero rollback cost. The 17 sister `*_v2` per-doc collections are
  user-owned and out of scope.

---

**END OF PLAN_V2.9.md**
