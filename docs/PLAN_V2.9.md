# Plan: v2.9 — Close v2.8 Carry-Overs and Ship VLM-Enriched `mmrag_v2_8`

**Status:** Draft v1.0 (2026-05-04)
**Owner:** ingestion pipeline
**Successor to:** `docs/archive/PLAN_V2.8_PRODUCTION_GAPS.md` (shipped 2026-05-04, tag `v2.8.0` on `645ab2b`)
**Related:** `docs/PROJECT_STATUS.md`, `docs/PROGRESS_CHECKLIST.md`,
`docs/QUALITY_GATES.md`, `docs/DECISIONS.md`, `docs/AGENT_GOVERNANCE.md`,
`AGENTS.md`

---

## 1. Why this plan exists

**v2.9 thesis (one sentence):** Close the four v2.8 carry-overs (Ayeva
profile misroute, Firearms heading regression, schema chunk_id
collisions, refiner smart-routing) and ship the `mmrag_v2_8` Qdrant
collection with VLM-enriched image points so the 34-doc canonical
corpus reaches **34/34 PASS without manual flag workarounds** and
image-side RAG retrieval is restored end-to-end — leaving v2.10 free
for the SRS rewrite + UIR refactor.

### v2.8 Completion Recap (Active Baseline)

`docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md` is the v2.9 BEFORE
state:

- Engine `__engine_version__=2.8.0`, schema `__schema_version__=2.7.0`
  (de-aliased; chunk-shape unchanged since v2.7).
- Test suite: **596 passed, 2 skipped, 0 failed.**
- Smoke matrix: **11/11 GATE_PASS + 11/11 UNIVERSAL_PASS** (incl. the
  new `digital_literature` slot for HARRY and the form-lane row for
  `scanned/0013_140302111325_001`).
- Broad reconversion: 34/34 PDF/EPUB exit=0; **30/34 canonical PASS**
  (incl. 1 `FORM_PASS`).
- Qdrant `mmrag_v2_8`: **22,137 / 22,160** unique embeddable chunks
  ingested (collision-free `point_id` from commit `0d3cc36`).
- 7-commit v2.8 chain on `main`:
  `5b0e13d → c2e795e → 9e4b8f8 → 59994f9 → 2f94503 → 0d3cc36 → 9726b43 → 645ab2b`.

### Workstreams, symptoms, concrete patterns, last evidence

| Workstream | Symptom | Concrete pattern (verified 2026-05-04) | Last evidence |
|---|---|---|---|
| **E — VLM enrichment (`mmrag_v2_8`)** | ~5,500 image points have placeholder `visual_description="[Figure on page N] | Context: ..."`, `vision_status="pending"`, `refined_content=null`. Image-side RAG retrieval degraded — searching for "wizard ornament" cannot match. | v2.8 broad reconversion ran with `--vision-provider none --no-refiner --no-cache` for apples-to-apples baseline matching; nothing dispatched a VLM call. | 22,137 points in `mmrag_v2_8`; ~5,500 are images |
| **Refiner smart-routing** | HARRY (clean prose, zero encoding corruption) hammered qwen-plus per chunk during the v2.8 broad reconversion's first attempt; refinements rejected ("Edit ratio 53.16% exceeds budget"). | `cli.py:686` config-default enable fires whenever `~/.mmrag-v2.yml` `refiner.enabled=true`, regardless of `has_encoding_corruption`. The diagnostic-driven auto-override at `cli.py:1101-1102` is dead code under the config-default path. | v2.8 used `--no-refiner` to mask the bug; documented in `docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md` Phase 5c gating decisions |
| **Ayeva profile misroute (Workstream D)** | `Ayeva_Python_Patterns` v2.8 fresh routes to `digital_literature`, suppresses `needs_code_enrichment`, CodeFormulaV2 never engages. CODE FAIL at `indentation_fidelity=0.83` (just under the 0.85 hard gate). | Rule 0c at `document_diagnostic.py:1457` (`_dialogue_pages >= 1 AND total_pages > 20 AND not has_tables AND 500 < avg_text_per_page < 2500 → literature += 0.4`) misfires on a code-heavy book — Python code with quoted strings reads as dialogue. | `output/Ayeva_Python_Patterns/ingestion.jsonl` (v2.8 fresh, `digital_literature`, 0.83) vs `output/ayeva_qa_20260501/ingestion.jsonl` (probe, `technical_manual`, 0.93) |
| **Firearms heading regression** | Profile changed `scanned` → `technical_manual` between v2.8 baselines; chunker's stricter heading-inheritance leaves 178/815 chunks orphan-headed. HEADING coverage 100% → 78% (gate ≥80%). | Same content fidelity, just less hierarchy annotation. Naive fix (relax threshold per profile) directly violates `AGENT-SPATIAL-20`. Constraint conflict to resolve before designing the fix. | `output/Firearms/ingestion.jsonl` (v2.8 fresh) — see `docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md` "Known Limitations" |
| **Schema `chunk_id` collisions** | 22,587 chunks in v2.8 corpus collapse to 22,160 unique `chunk_id`s — **427 within-file duplicates** (largest contributor `KI_En_ChatGPT_Praktische_Gids` 279, then `Devlin_LLM_Agents` 76, `Fluent_Python` 15). Boilerplate footers / repeated page numbers / identical short labels collide. | `_generate_chunk_id` at `src/mmrag_v2/schema/ingestion_schema.py:715` hashes `f"{doc_id}:{page}:{modality}:{content}"` — does NOT include the chunk's per-document position. Two visually-identical chunks on the same page collapse to the same id; on Qdrant upsert (uuid5 from chunk_id, commit `0d3cc36`) the 427 duplicates silently overwrite each other. | `output/<canonical>/ingestion.jsonl` × 34; ingest evidence in v2.8 AFTER snapshot |

### What closing all phases achieves

- **34/34 canonical PASS** across `scripts/qa_conversion_audit.py`,
  with no `--no-refiner` workaround in `scripts/convert_books.sh`.
- Every chunk has a globally-unique `chunk_id` (and therefore a
  globally-unique uuid5 `point_id`). 0 within-file collisions on the
  next broad reconversion.
- `mmrag_v2_8` repopulated from a clean drop-and-recreate, with every
  ~5,500 image point carrying a real cloud-VLM `visual_description`
  + `vision_status="complete"`.
- v2.10 inherits a corpus and a vector store with no behavioral debt
  attributable to v2.8.

## 2. Goals & Non-Goals

### Goals (measurable from JSONL / audit / Qdrant counts)

1. `scripts/qa_conversion_audit.py output/<v29_run>/<doc>/ingestion.jsonl`
   reports `AUDIT_PASS` or `FORM_AUDIT_PASS` for **all 34 canonical
   docs**. No row remains in `FAIL` from the v2.8 AFTER snapshot.
2. `Ayeva_Python_Patterns` v2.9 fresh: `profile_type=technical_manual`
   AND `indentation_fidelity ≥ 0.85` AND CODE PASS.
3. `Firearms` v2.9 fresh: `HEADING coverage ≥ 0.80` AND content
   fidelity unchanged (chunk count within ±2% of v2.8 fresh's 1690).
4. `bash scripts/smoke_multiprofile.sh`: every row `GATE_PASS` +
   `UNIVERSAL_PASS` (no waivers; `GATE_PASS [form: ...]` /
   `FORM_AUDIT_PASS` count as variants only when `document_type=form`
   per `docs/QUALITY_GATES.md` "Form / Invoice Acceptance Class").
5. `pytest tests/ -q` ≥ **610 passed, 0 failed** (596 baseline + 14+
   new tests across phases 1-4).
6. Within-file `chunk_id` duplicates across the 34 canonical JSONLs:
   **0** (was 427 in v2.8 AFTER).
7. `mmrag_v2_8` Qdrant collection contains exactly the unique
   embeddable chunk count of the v2.9 broad reconversion, freshly
   recreated. **0 image points** with `vision_status="pending"`;
   `visual_description` non-placeholder for every image point.
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

Phases ordered cheapest-first so the snapshot baseline + the surgical
schema/CLI fixes land before the diagnostic investigation (Ayeva /
Firearms) and before the heavy Phase 5 reconversion + VLM enrichment
runtime.

---

### Phase 0 — Lock the v2.8 AFTER state as the v2.9 BEFORE

**What:** Re-run the v2.8 acceptance harness from a clean checkout
to confirm the entry state is reproducible **before any code change**.
Without this, "did v2.9 improve anything" becomes hand-waving — the
Phase 0 step in v2.8's plan was load-bearing for exactly the same
reason.

**Steps:**
1. `git status --short` — confirm tree is clean (no in-progress edits).
   Stash or commit any local changes first.
2. `pytest tests/ -q` — expect **596 passed, 2 skipped, 0 failed**.
   If different, capture the diff and update the v2.9 baseline note
   before proceeding.
3. `bash scripts/smoke_multiprofile.sh` — expect **11/11 GATE_PASS +
   11/11 UNIVERSAL_PASS** (incl. the form lane row for
   `scanned/0013_140302111325_001` and the `digital_literature` slot
   for HARRY).
4. Verify Qdrant baseline:
   ```bash
   curl -sS http://localhost:6333/collections/mmrag_v2_8/points/count \
     -X POST -H "Content-Type: application/json" -d '{"exact":true}'
   # expected: {"result":{"count":22137}, "status":"ok"}
   ```
5. Spot-check three representative outputs to confirm the v2.8
   AFTER metrics:
   - `output/A_comprehensive_review_on_hybrid_electri/ingestion.jsonl`
     → `ctrl_chunks=0`.
   - `output/Combat_Aircraft_August_2025/ingestion.jsonl`
     → `encoding_artifacts=0`, `high_corruption=0`.
   - `output/Chaubal_PyTorch_Projects/ingestion.jsonl`
     → `indentation_fidelity ≥ 0.85`.
6. Note the canonical baseline pointer in
   `docs/PROJECT_STATUS.md` "Active Baseline" (already
   `docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md`); if any of the
   above checks drift, **stop and reconcile** before opening any
   Phase 1+ commit.

**Done when:**
- All four reproducibility checks match the v2.8 AFTER snapshot
  numbers exactly (or any drift is committed as a v2.9 baseline
  delta note before phase work begins).
- `docs/PROJECT_STATUS.md` "Active Baseline" still references
  `docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md`.

**Risk:** Low. Read-only verification.

**Estimated effort:** 30 min (commands run; Qdrant container must be
up). No engineering time.

---

### Phase 1 — Schema: within-file `chunk_id` collision fix

**What:** `src/mmrag_v2/schema/ingestion_schema.py:715`
(`_generate_chunk_id(doc_id, content, page, modality)`) hashes
`f"{doc_id}:{page}:{modality}:{content}"`. Two chunks on the same page
with byte-identical `content` collapse to the same id. The 34-doc
v2.8 corpus has **427** within-file dupes (largest contributors:
`KI_En_ChatGPT_Praktische_Gids` 279, `Devlin_LLM_Agents` 76,
`Fluent_Python` 15) — typically boilerplate page footers, repeated
page numbers, identical short labels. The dupes silently overwrite
each other on Qdrant upsert (the v2.8 ingest landed `22,137` from
`22,160` unique − 23 embed errors; the 427 collisions are
indistinguishable in `mmrag_v2_8` from "successful upserts of a
single point").

**Parallel-site audit (do this FIRST):**

| Site | File:line | Current behavior | Action |
|---|---|---|---|
| Generator | `src/mmrag_v2/schema/ingestion_schema.py:715` | `_generate_chunk_id(doc_id, content, page, modality)` hashes 4 fields, no position component | **The fix site.** Add per-document chunk index to the hash seed. |
| Text factory caller | `ingestion_schema.py:782` | `_generate_chunk_id(doc_id, content, page_number, "text")` | Must thread the chunk's `i+1` position from the call stack |
| Image factory caller | `ingestion_schema.py:881` | `_generate_chunk_id(doc_id, f"image:{asset_path}", page_number, "image")` | Same; image dupes are rare (asset_path usually unique) but the contract should hold |
| Table factory caller | `ingestion_schema.py:970` | `_generate_chunk_id(doc_id, f"table:{content[:50]}", page_number, "table")` | Same; tables can collide when two short tables on a page share the same first 50 chars |
| Direct chunk creators (`mapper.py`) | `src/mmrag_v2/mapper.py` `create_*_chunk` | Each constructs an `IngestionChunk` then calls the factory | Audit whether the call sites already track a per-document position; if not, source one before the factory call |
| Schema version impact | `__schema_version__=2.7.0` (`src/mmrag_v2/version.py`) | Field shape unchanged | The `chunk_id` *value* changes for affected chunks but the field shape doesn't → schema version stays 2.7.0 (no bump) |
| Qdrant `point_id` impact | `scripts/ingest_to_qdrant.py:453` `uuid5(_POINT_ID_NAMESPACE, chunk_id)` | Deterministic uuid5 from chunk_id (v2.8 fix `0d3cc36`) | The 427 affected chunks get **new** uuid5s; old uuid5s become orphans on next ingest. Migration step is in Phase 5 (drop-and-recreate `mmrag_v2_8`). |
| Existing test coverage | `tests/test_qdrant_point_id_collision.py` (6 tests) | Pins the uuid5 mapping but assumes unique input chunk_ids | New test file at the schema layer is needed; do not weaken the Qdrant-layer tests |

**Approach:**

1. Extend `_generate_chunk_id` signature to accept a `position` int
   (default `0` for backward compatibility but require it for new
   call sites; mark the default-zero path deprecated in the
   docstring). Hash seed becomes
   `f"{doc_id}:{page}:{modality}:{position}:{content}"`.
2. Update the three factory functions
   (`text`, `image`, `table` at `ingestion_schema.py:782/881/970`)
   to thread a per-document `position` from their callers.
3. Audit `mapper.py` / `processor.py` / `batch_processor.py` chunk
   construction sites — each maintains a per-document chunk index;
   confirm it's plumbed through to the factory call.
4. Update existing fixtures and golden outputs only where the test's
   *contract* is unaffected (i.e. tests that compute chunk_id at
   runtime still pass; tests with hard-coded chunk_id strings need
   the new value computed once and committed).

**Tests (red→green) — 4 new in `tests/test_chunk_id_collision_v29.py`:**

- `test_within_file_dupe_content_yields_distinct_chunk_ids` — build
  two chunks with `(doc_id, page, modality, content)` identical and
  position differing by 1; assert `chunk_id` strings differ.
- `test_chunk_id_stable_under_position_zero_default` — same fixture
  with default `position=0`; assert chunk_id matches a pre-computed
  hex (deterministic; no drift across runs).
- `test_factory_threads_position_for_text_image_table` — call each
  factory twice on the same `(content, page)` with positions 0 and
  1; assert all six results have distinct chunk_ids.
- `test_full_corpus_no_within_file_chunk_id_collisions` —
  parameterized scan across `output/<canonical>/ingestion.jsonl`
  (env-gated `RUN_CORPUS_SCAN=1`); count
  `len(chunk_ids) - len(set(chunk_ids))` per file; assert 0. Until
  Phase 5 reconversion lands, the test is xfailed (or skipped) but
  the assertion contract is committed now.

**Done when:**
- 4 new tests green.
- `pytest tests/ -q` passes (no fixture-breakage from the signature
  change).
- Code-review confirmation that every direct factory call site
  threads `position` (i.e. the `position=0` default isn't being
  abused as an escape hatch in production paths).
- Phase 5 reconversion's collision-count audit reports 0 within-file
  dupes (verified at the end of Phase 5).

**Risk:** Low–Medium. Surgical at the generator, broad at the call
sites. Mitigation: the `position=0` default keeps existing tests and
external callers compiling; production paths must be migrated
explicitly.

**Estimated effort:** **1.5–2 h engineering** + 1 h regression-test
authoring.

---

### Phase 2 — CLI: refiner smart-routing fix (`cli.py:686`)

**What:** The CLI's config-default refiner-enable logic at
`src/mmrag_v2/cli.py:684-687`:

```python
import sys as _sys
_refiner_explicitly_disabled = "--no-refiner" in _sys.argv
if not enable_refiner and cfg.refiner.enabled and not _refiner_explicitly_disabled:
    enable_refiner = True
```

Fires whenever `~/.mmrag-v2.yml` has `refiner.enabled=true`,
**regardless of `has_encoding_corruption`**. The diagnostic-driven
auto-override at `cli.py:1101-1102` is dead code under the
config-default path because `enable_refiner` is already `True` by
the time it runs. Empirical: v2.8 broad reconversion's first
attempt left HARRY (clean prose, zero encoding corruption) hammering
qwen-plus per chunk with refinements rejected ("Edit ratio 53.16%
exceeds budget"). The remediation in v2.8 was the `--no-refiner`
flag — masking the bug, not fixing it.

**Parallel-site audit (do this FIRST):**

| Site | File:line | Current behavior | Action |
|---|---|---|---|
| Config-default enable (process) | `cli.py:683-695` | Eagerly sets `enable_refiner=True` from `cfg.refiner.enabled` before diagnostics run | **Fix site.** Defer the decision until after diagnostic detects `has_encoding_corruption`. |
| Diagnostic-driven auto-override (process) | `cli.py:1093-1106` | Sets `enable_ocr=True, force_ocr=True` and (gated on `not _refiner_explicitly_disabled`) `enable_refiner=True` when `intelligence_metadata.get("has_encoding_corruption")` | After Phase 2, this becomes the *primary* refiner-enable site under the config-default path. |
| `batch` command equivalent | `cli.py` `batch` command body (around `cli.py:1741`) | Need to confirm the same config-default → diagnostic-override sequence is wired identically; v2.8 plan §2b found the cheap-evidence trigger lived in TWO places (`cli.py:1112` AND `cli.py:1741`); check whether refiner-enable also has a parallel site | If so, fix both. |
| Refiner config consumer | `src/mmrag_v2/refiner.py` | Reads `enable_refiner` flag passed in by CLI; doesn't inspect `has_encoding_corruption` itself | Unchanged. Decision belongs at the CLI seam. |
| Refiner threshold-override on encoding corruption | `batch_processor.py` (per `docs/DECISIONS.md` "Heal-Over for Encoding Corruption") | `threshold=0.0` is set when `has_encoding_corruption` is true | Unchanged. The existing heal-over code path still fires once Phase 2 routes refiner-enable correctly. |
| `convert_books.sh` flag posture | `scripts/convert_books.sh` | Currently passes `--no-refiner` for apples-to-apples vs v2.8 baseline | After Phase 2 lands AND Phase 5 reconversion proves byte-stability for clean-prose docs, drop `--no-refiner`. |

**Approach:**

1. Move the config-default check OUT of `cli.py:683-695` (where
   `enable_refiner` is set before diagnostic runs) and INTO the
   intelligence-metadata block at `cli.py:1093-1106` (where
   `has_encoding_corruption` is known).
2. Replace the existing
   `if not enable_refiner and cfg.refiner.enabled and not _refiner_explicitly_disabled:`
   at line 686 with **only the model/provider/url/key default
   propagation** (lines 688-695) — leave `enable_refiner` at its
   CLI-supplied value.
3. In the diagnostic-override block at `cli.py:1101-1102`, expand
   the gate so the config-default still wins when corruption is
   detected:
   ```python
   _config_default_active = cfg.refiner.enabled and not _refiner_explicitly_disabled
   if intelligence_metadata.get("has_encoding_corruption") and _config_default_active:
       if not enable_refiner:
           enable_refiner = True
   ```
   (This is the integration point — explicit corruption signal
   gates the auto-enable; explicit `--enable-refiner` from the user
   bypasses the gate entirely; explicit `--no-refiner` always wins.)
4. **Mirror the same change in the `batch` command** if the
   parallel-site audit confirms the duplicate logic exists there.
5. After Phase 5 reconversion proves byte-stable text for clean-prose
   docs, drop `--no-refiner` from `scripts/convert_books.sh`.

**Tests (red→green) — 4 new in `tests/test_cli_refiner_smart_routing.py`:**

- `test_refiner_off_by_default_when_clean_prose` — fake
  `intelligence_metadata={"has_encoding_corruption": False}` +
  `cfg.refiner.enabled=True`; assert the resulting `enable_refiner`
  passed into `BatchProcessor` is `False`.
- `test_refiner_on_when_encoding_corruption_detected` — fake
  `intelligence_metadata={"has_encoding_corruption": True}` +
  `cfg.refiner.enabled=True`; assert `enable_refiner=True`.
- `test_refiner_explicit_no_refiner_always_wins` — `--no-refiner` in
  argv + corruption detected; assert `enable_refiner=False`.
- `test_refiner_explicit_enable_refiner_bypasses_diagnostic` —
  `--enable-refiner` in argv + clean prose; assert `enable_refiner=True`.

**Done when:**
- 4 new tests green.
- v2.9 broad reconversion (Phase 5) runs WITHOUT `--no-refiner` and
  produces byte-stable text for HARRY (clean prose) AND
  refiner-applied text for `Combat_Aircraft_August_2025` (encoding
  corruption — but Phase 5 will use whatever profile the v2.9
  pipeline routes to; spot-check).
- `scripts/convert_books.sh` no longer carries `--no-refiner` after
  Phase 5 verification.

**Risk:** Low. The fix is moving a single conditional and adding a
gate; the existing diagnostic-override path is already exercised by
v2.7-era tests for the OCR auto-override. **Constraint:** must not
violate `AGENT-VAL-01` (no document-specific or filename-specific
behavior). Check satisfied — the gate is on `has_encoding_corruption`,
a structural-integrity flag.

**Estimated effort:** **1–2 h engineering** + ~30 min per-doc
verification (run HARRY 10-page subset + Combat 10-page subset to
confirm the routing matrix).

---

### Phase 3 — Workstream D: ProfileClassifier rule 0c tightening (Ayeva)

**What:** `Ayeva_Python_Patterns` v2.8 fresh re-conversion routes to
`digital_literature` instead of `technical_manual`, suppressing the
`needs_code_enrichment` cheap-evidence trigger (CodeFormulaV2
doesn't auto-engage for the `digital_literature` profile). Empirical:

- BEFORE (probe `output/ayeva_qa_20260501/`, 2026-05-01,
  `--enable-doctr --ocr-mode auto`): `profile=technical_manual`,
  CODE PASS, `indentation_fidelity=0.93`.
- AFTER (v2.8 fresh `output/Ayeva_Python_Patterns/`, 2026-05-04):
  `profile=digital_literature`, CODE FAIL,
  `indentation_fidelity=0.83` (just under the 0.85 hard gate).

Rule 0c was added 2026-05-03 in commit `2f51816`
(`document_diagnostic.py:1457-1475`):

```
_dialogue_pages >= 1
AND total_pages > 20
AND not has_tables
AND 500 < avg_text_per_page < 2500
→ literature += 0.4
```

The hypothesis is that Python code blocks include strings with
quotation marks that the dialogue heuristic
(`document_diagnostic.py:1443-1446`, "≥4 double-quote chars on a
page → dialogue page") misreads. Ayeva is a code-heavy book, NOT a
novel.

**Parallel-site audit (do this FIRST):**

| Site | File:line | Current behavior | Action |
|---|---|---|---|
| Rule 0c | `src/mmrag_v2/orchestration/document_diagnostic.py:1457-1475` | `_dialogue_pages >= 1 AND total_pages > 20 AND not has_tables AND 500 < avg_text_per_page < 2500 → literature += 0.4` | **Fix site.** Add a code-density inverse-signal guard. |
| Rule 0 (full novel) | `document_diagnostic.py:1449-1455` | `_dialogue_ratio > 0.3 AND total_pages > 50 AND not has_tables → literature += 0.8` | Keep unchanged; the +0.8 path requires high dialogue ratio AND >50 pages (HARRY's signature). Ayeva is 359 pages so it could in principle hit the +0.8 path too — verify it doesn't (i.e. the *ratio*, not just count, of dialogue pages is low for Ayeva). |
| Profile scorer (digital_literature) | `src/mmrag_v2/orchestration/profile_classifier.py` `_score_digital_literature` | Reads `domain=literature` 0.50, `page_count ≥50` 0.20, etc. (per `docs/DECISIONS.md` "Post-Docling Sanity Pass + `digital_literature` Profile") | A code-density inverse-signal could ALSO live here ("HARD REJECT if `code_evidence_pages ≥ 2`") — but the upstream cleaner fix is at Rule 0c. |
| `needs_code_enrichment` decision | `cli.py:1112` (process) AND `cli.py:1741` (batch) calls `decide_code_enrichment_for_pdf(...)` | Current behavior triggers CodeFormulaV2 for Chaubal (technical_manual route) but NOT for Ayeva (digital_literature route) | Unchanged. Phase 3's classifier fix re-routes Ayeva → `technical_manual`, which makes `needs_code_enrichment` fire on cheap evidence as designed. |
| Existing classifier tests | `tests/test_classifier_digital_literature.py` (7 tests), `tests/test_classifier_fallback.py::test_harry_potter_like_literature` | Pins HARRY → `digital_literature`, code-heavy books → NOT `digital_literature` | DO NOT WEAKEN. v2.9 fix must keep all 7 + the negative case green. |
| Diagnostic test fixtures | `tests/fixtures/` for code-evidence | New fixture needed: an Ayeva-shaped feature vector (low dialogue ratio per page, HIGH code_evidence_pages) | Add to a new test file. |

**Approach:**

1. **Source the code-density signal.** `DocumentDiagnosticEngine`
   already counts `CodeItem`s and code-candidate regions for the
   `needs_code_enrichment` trigger. Promote the
   `code_evidence_pages` (or equivalent — exact field name to
   confirm during investigation) into the diagnostic features
   surface available to Rule 0c.
2. **Tighten Rule 0c** at `document_diagnostic.py:1457-1475`:
   ```
   _dialogue_pages >= 1
   AND total_pages > 20
   AND not has_tables
   AND 500 < avg_text_per_page < 2500
   AND code_evidence_pages < 2   # NEW
   → literature += 0.4
   ```
   *Why this threshold:* Chaubal's v2.8 fresh has CodeFormulaV2
   engaged on multiple pages → `code_evidence_pages >> 2`. HARRY
   has 0. The 2-page threshold is conservative enough not to flip
   Combat (a magazine with incidental code-shape decorations).
3. **Re-run the regression tests** for the `digital_literature`
   profile (7 tests) — they MUST still pass on HARRY and the other
   pinned positives. If any flips, that's a contract change → stop
   and document.
4. **Verify on the v2.8 fresh outputs** (no reconversion needed yet):
   load each canonical `intelligence_metadata` (first JSONL line),
   re-run the rule logic offline, confirm Ayeva flips to
   `technical_manual` and HARRY stays on `digital_literature`.

**Tests (red→green) — 6 new in `tests/test_classifier_rule_0c_tightening.py`:**

- `test_rule_0c_fires_for_harry_dialogue_low_code` — feature vector
  with HARRY's signature (high dialogue ratio low code evidence,
  29-page test slice); assert literature score includes the +0.4
  contribution.
- `test_rule_0c_suppressed_for_ayeva_code_heavy_book` — feature
  vector with Ayeva's signature (some quoted strings on dialogue
  pages BUT `code_evidence_pages >= 2`); assert literature score
  does NOT include +0.4.
- `test_rule_0c_suppressed_for_chaubal_pytorch_book` — Chaubal's
  feature vector; assert literature score does NOT include +0.4.
- `test_rule_0c_suppressed_for_fluent_python_book` — non-regression
  control; assert classifier still routes Fluent → `technical_manual`.
- `test_ayeva_routes_to_technical_manual_post_fix` — full classifier
  output on Ayeva's `intelligence_metadata`; assert
  `profile_type == "technical_manual"`.
- `test_harry_routes_to_digital_literature_post_fix` — non-regression
  control; assert HARRY still routes to `digital_literature`.

**Done when:**
- 6 new tests green.
- All 7 existing `tests/test_classifier_digital_literature.py` tests
  remain green.
- All 9 existing `tests/test_classifier_fallback.py` tests remain
  green.
- v2.9 fresh re-conversion of Ayeva (in Phase 5):
  `profile_type=technical_manual`, CodeFormulaV2 engages,
  `indentation_fidelity ≥ 0.85`, CODE PASS.
- HARRY v2.9 fresh re-conversion: still routes to
  `digital_literature`, page-13 reading-order acceptance test
  passes.

**Risk:** Medium. Classifier changes have the highest blast radius
of v2.9 phases — every routed doc could in principle flip. The
parallel-site audit is mandatory; the 6 named tests cover the main
contract surface. **AGENTS.md compliance:** no document-specific or
filename-specific logic added (the new gate is a numeric threshold
on a feature already used by `needs_code_enrichment`).

**Estimated effort:** **2–4 h engineering** (mostly investigation +
test authoring) + ~10 min Ayeva re-conversion verification.

---

### Phase 4 — Firearms heading regression (with `AGENT-SPATIAL-20` resolution)

**What:** `Firearms` v2.8 fresh re-conversion: profile changed
`scanned` → `technical_manual` between baselines, and the chunker's
heading-inheritance is stricter under `technical_manual`. HEADING
coverage **100% → 78%** (gate is ≥80%). 178 / 815 chunks now lack
`parent_heading`. Same content fidelity (chunk count 1690 vs 1691),
just less hierarchy annotation.

**Constraint conflict to resolve before designing the fix:**
`AGENT-SPATIAL-20` says "Refinement logic must rely on a single
20-unit vertical threshold. No profile-specific or heading-specific
branches allowed." (`AGENTS.md` §1.6) The naive fix — relaxing the
heading threshold for `technical_manual` on scanned-modality input —
**directly violates this**. Two acceptable resolutions, in
preference order:

- **(a) Re-route Firearms to the `scanned` profile.** Investigate
  why the classifier flipped the route; the fix lives in
  `profile_classifier.py`, not in any spatial threshold. Respects
  `AGENT-SPATIAL-20` unchanged.
- **(b) Propose an explicit `AGENT-SPATIAL-20` amendment** in
  `AGENTS.md` + a new entry in `docs/DECISIONS.md`, with empirical
  evidence that the single-threshold rule no longer serves the
  corpus. Requires user sign-off; do NOT auto-amend.

**Default: (a). Fall back to (b) only if (a) demonstrably fails.**

**Parallel-site audit (do this FIRST):**

| Site | File:line | Current behavior | Action |
|---|---|---|---|
| Firearms intelligence metadata | `output/Firearms/ingestion.jsonl:1` | v2.8 fresh: `profile_type=technical_manual`, `document_type=scanned_degraded`, `is_scan=true`, `total_pages=292`, `image_density=1.0` | The classifier saw `is_scan=true` AND `image_density=1.0` and STILL chose `technical_manual` over `scanned`. That's the routing bug to investigate. |
| Profile scorer (scanned) | `src/mmrag_v2/orchestration/profile_classifier.py` `_score_scanned` | Should boost on `is_scan=true` + scanned-shaped image density | Inspect the score for Firearms's feature vector; identify why `technical_manual` outscored it. |
| Profile scorer (technical_manual) | `_score_technical_manual` | Per Workstream D Milestone 1 (2026-04-30), this became the digital fallback for long-form non-magazine non-scanned docs | Confirm Firearms isn't tripping a fallback branch unintentionally. |
| Diagnostic baseline | `data/scanned/Firearms.pdf` (or wherever the file lives now) | Pre-v2.8 baseline routed to `scanned` cleanly | Compare the diagnostic features at the 2026-04-29 baseline run vs the v2.8 fresh run — what changed? |
| Heading inheritance threshold | `batch_processor.py` (chunker post-processing) | Stricter under `technical_manual` (the v2.7-era POS Boundary + heading promotion rules) | DO NOT modify per profile (`AGENT-SPATIAL-20`). |
| 20-unit vertical threshold | wherever it lives (search `20`-unit refinement) | Single threshold applied everywhere | DO NOT branch. |
| HARRY scanned-route assertion | `tests/test_classifier_fallback.py` | `is_scan=true` doesn't catch born-digital novels (HARRY routes via `_score_digital_literature`'s HARD REJECT scans) | Confirm Firearms's feature vector doesn't accidentally trip a `digital_literature` HARD REJECT path that misroutes elsewhere |
| `data_spreadsheet` already-questionable route | per `docs/PROGRESS_CHECKLIST.md` Workstream D | Routes to `technical_manual` (acceptable) | Unrelated; do not collateral-fix. |

**Approach:**

1. **Investigate the route flip (option (a)).**
   - Compare Firearms's diagnostic features (pre-v2.8 vs v2.8 fresh)
     by re-running `DocumentDiagnosticEngine` on the same PDF and
     dumping the feature vector. Diff against the prior baseline's
     `intelligence_metadata` line.
   - Trace the `_score_scanned` vs `_score_technical_manual` scores
     for that vector. Identify which signal moved.
   - The most likely root cause: the 2026-04-30 Milestone 1 fix
     changed the `digital fallback default` from `DIGITAL_MAGAZINE`
     to `TECHNICAL_MANUAL` (`docs/PROGRESS_CHECKLIST.md` Workstream D),
     which made `technical_manual` the catch-all for long-form
     non-magazine docs. Firearms (`is_scan=true`, 292 pages) may be
     tripping this catch-all instead of `scanned`.
2. **Apply the targeted scorer fix** in
   `src/mmrag_v2/orchestration/profile_classifier.py`. Acceptable
   patterns:
   - Add a stronger `is_scan=true` weight to `_score_scanned` so it
     wins on Firearms-shape inputs.
   - Add a HARD REJECT to `_score_technical_manual` that suppresses
     it for inputs with `is_scan=true` AND `image_density >= 1.0`
     AND `total_pages > 100` (Firearms's signature).
   - Both are profile-scorer adjustments, NOT spatial-threshold
     branches. `AGENT-SPATIAL-20` is respected.
3. **If (a) fails** (i.e. no scorer adjustment can route Firearms
   correctly without breaking another doc):
   - Stop, document the failure analysis with concrete numbers
     (which alternative routes were considered, which broke).
   - Propose an explicit `AGENT-SPATIAL-20` amendment in
     `AGENTS.md` + new `docs/DECISIONS.md` entry with the empirical
     case. **Get user sign-off before implementing the amendment.**
4. **Verify the smoke matrix and the full canonical corpus** —
   Firearms reaches HEADING ≥ 80% AND no other row regresses
   (especially the SCAN0013 form-lane row, which already routes to
   `scanned` and must continue to do so).

**Tests (red→green) — 5 new in `tests/test_classifier_firearms_route.py`:**

- `test_firearms_feature_vector_routes_to_scanned` — feature vector
  pulled from `output/Firearms/ingestion.jsonl:1` (intelligence
  metadata); assert classifier returns `scanned` (or
  `scanned_degraded`).
- `test_firearms_heading_coverage_at_least_80pct_post_fix` — load a
  v2.9 fresh re-conversion JSONL (env-gated `RUN_FIREARMS_VERIFY=1`,
  populated by Phase 5); compute
  `len([c for c in chunks if c.metadata.hierarchy.parent_heading]) /
  total_text_chunks`; assert ≥ 0.80.
- `test_scan0013_still_routes_to_scanned` — form's feature vector;
  assert `scanned` (non-regression — must still hit form lane).
- `test_earthship_still_routes_to_scanned` — non-regression (Earthship
  is the canonical scanned book).
- `test_harry_still_routes_to_digital_literature` — non-regression
  (the classifier-flip bug must not surface a HARRY regression).

**Done when:**
- 5 new tests green (or the corpus-verify test xfailed pending Phase 5).
- v2.9 fresh re-conversion of Firearms: `profile_type=scanned` (or
  `scanned_degraded`), HEADING coverage ≥ 80%, chunk count within
  ±2% of v2.8 fresh's 1690.
- Smoke matrix 11/11 GATE_PASS + UNIVERSAL_PASS.
- `AGENT-SPATIAL-20` constraint respected — no profile-specific
  spatial-threshold branch added. (If (b) was required, the
  amendment is checked into `AGENTS.md` + `docs/DECISIONS.md`
  with user sign-off recorded in this plan's §8 decision log.)

**Risk:** Medium–High. Classifier scorer adjustments have corpus-wide
blast radius. The 5 named regression tests are the contract surface;
a sixth (full smoke matrix run) must pass before this phase merges.

**Estimated effort:**
- Path (a): **3–6 h engineering** (mostly investigation +
  scorer-tuning + corpus verification).
- Path (b): **+1 day** if `AGENT-SPATIAL-20` amendment proves
  necessary (drafting, user sign-off, doc updates). Plan effort
  total assumes (a); flag if (b) is taken.

---

### Phase 5 — Broad reconversion + Qdrant migration + VLM enrichment + AFTER snapshot

**What:** With Phases 0–4 closed, run the broad reconversion that
verifies all four code fixes land on real corpus data, drop and
recreate the `mmrag_v2_8` Qdrant collection (Phase 1 chunk_id
migration callout), run the **Priority 1 VLM enrichment** of all
~5,500 image chunks via cloud `qwen3-vl-plus`, re-ingest the corpus
clean, and produce the v2.9 AFTER snapshot.

This phase is **not optional**. The Ayeva, Firearms, refiner-routing,
and chunk_id fixes are unverifiable without re-converting the corpus.
Per the prompt's mandatory shape, Phase 5 also runs the chunk_id
migration callout from Phase 1 (drop-and-recreate `mmrag_v2_8` to
absorb the new collision-free chunk_ids cleanly).

**VLM choice — locked to cloud `qwen3-vl-plus` only.** Local
`NuMarkdown-8B-Thinking-mlx-8bits` at `http://10.0.10.246:8000/v1`
is unreachable from off-network machines (project memory, confirmed
2026-05-04) and is deferred to v2.10 when network reachability
returns. The v2.9 enrichment script defaults to cloud and **does
NOT branch on local availability**. (May add a
`# v2.10: re-evaluate local` comment at the call site.)

#### Pre-flight checklist

- [ ] Phases 0-4 all merged and tagged.
- [ ] `pytest tests/ -q` reports ≥ **610 passed, 0 failed**.
- [ ] `bash scripts/smoke_multiprofile.sh` reports **11/11 GATE_PASS
      + 11/11 UNIVERSAL_PASS** with no waivers (`AGENT-VAL-01`,
      CLAUDE.md "Project Invariants", `docs/QUALITY_GATES.md`).
      `GATE_PASS [form: ...]` for SCAN0013 is the only acceptable
      `[form]` variant; per `AGENT-VAL-01` no other doc may use the
      form lane as a workaround.
- [ ] Alibaba DashScope API key reachable; Source Sanctity validated
      (PCWorld + Combat-style hallucination probes from
      `tests/fixtures/blind_set_manifest.json` re-run pre-flight to
      confirm the cloud endpoint still passes — text-reading hits
      ≤ 22.2%, hard fallbacks ≤ 21.4%, Combat-style hallucinations
      = 0).
- [ ] HARRY pages-1-30 acceptance fixture
      (`tests/test_docling_postprocessor_acceptance.py`) green
      against the most recent live HARRY conversion.

**Parallel-site audit (do this FIRST):**

| Site | File:line | Current behavior | Action |
|---|---|---|---|
| Conversion runner | `scripts/convert_books.sh` | 34 entries, `--vision-provider none --no-refiner --no-cache` | After Phase 2 lands AND clean-prose byte-stability is verified, drop `--no-refiner`. Keep `--vision-provider none --no-cache` (Phase 5's targeted enrichment script handles VLM, not the conversion). |
| Image chunk pending status | per-doc `output/<doc>/ingestion.jsonl` | `vision_status="pending"`, `visual_description="[Figure on page N] | Context: <breadcrumb>"`, `refined_content=null` for image chunks | The enrichment script is the consumer; it must read these fields and write back `vision_status="complete"`, real `visual_description`, populated `refined_content`. |
| Ingestion script payload writer | `scripts/ingest_to_qdrant.py:316-317, 399-400` | Pulls `metadata.visual_description` or top-level `visual_description` for the payload `visual_description` field | Confirm the enrichment script writes BOTH locations consistently (or pick one — the schema canonical is `metadata.visual_description`). |
| `point_id` derivation | `scripts/ingest_to_qdrant.py:42, 453` | `uuid5(_POINT_ID_NAMESPACE, chunk_id)` | After Phase 1 lands, the `chunk_id` value changes for the 427 affected chunks; re-ingest produces new `point_id`s for those points. Drop-and-recreate cleans this up. |
| Existing `mmrag_v2_8` state | Qdrant 22,137 points | **No production retrieval state has been built up** (per project memory, the collection was just created 2026-05-04). | Drop-and-recreate is safe — no rollback of consumer state needed. Recommended option per the prompt §2 Priority 5 migration consideration. |
| 17 sister `*_v2` per-doc collections | Qdrant containers | Pre-existing user-owned data; out of v2.9 scope | DO NOT TOUCH. Drop only `mmrag_v2_8`. |
| Vision Source Sanctity validator | `src/mmrag_v2/vision/vision_prompts.py`, `src/mmrag_v2/vision/vision_manager.py` | Existing text-reading detection + sanitizer + retry harness | The enrichment script MUST call through the existing `VisionManager` so Source Sanctity rules apply (no text transcription, visual-only prompt, retry on detected text-reading). |
| Embedding rebuild scope | nomic-embed-text via `scripts/ingest_to_qdrant.py` | 23 embed errors logged in v2.8 ingest (mostly Combat p66 reconstructed text + 4 long tables) | Spot-check whether v2.9 changes (refiner routing, chunk_id) shift the embed-error count; document any new errors. |

#### Steps

**5a. Broad reconversion (post-Phase-1-through-4 fixes).**

1. Run `bash scripts/convert_books.sh` with the v2.9 flag posture
   (drop `--no-refiner` if Phase 2 verification confirmed
   byte-stability for clean-prose docs; otherwise document why it
   stays).
2. Per-doc audit: `python scripts/qa_conversion_audit.py output/<v29_run>/<doc>/ingestion.jsonl`.
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

1. Author `scripts/enrich_image_chunks_v29.py` (new). Behavior:
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
   - **DO NOT branch on local-VLM availability.** Hardcode cloud.
     (Add `# v2.10: re-evaluate local NuMarkdown-8B endpoint`
     comment at the provider-selection line.)
2. Per-doc audit again: `python scripts/qa_conversion_audit.py
   ...` — `placeholder_ratio` for image chunks should drop to 0%
   for docs that completed enrichment.
3. Run `python scripts/vlm_quality_summary.py
   output/<v29_run>/<doc>/ingestion.jsonl --production` for at
   least the blind-set documents (Greenhouse, etc. — see
   `tests/fixtures/blind_set_manifest.json`).

**5c. Qdrant `mmrag_v2_8` drop-and-recreate.**

1. Confirm no consumer is using `mmrag_v2_8` (the v2.8 ingest
   evidence shows it has no production retrieval state — per
   project memory).
2. Drop the collection:
   ```bash
   curl -X DELETE http://localhost:6333/collections/mmrag_v2_8
   ```
3. Re-create with the v2.8 schema (vector dim, distance metric —
   confirmed in `scripts/ingest_to_qdrant.py`).
4. Re-ingest the v2.9 corpus:
   ```bash
   bash tmp/v29_ingest.sh   # (loop scripts/ingest_to_qdrant.py once per canonical doc)
   ```
   Use `--collection mmrag_v2_8 --model nomic-embed-text` (same as
   v2.8). The new chunk_ids from Phase 1 produce new uuid5
   `point_id`s; the re-ingest is a clean populate.
5. Verify:
   ```bash
   curl -sS http://localhost:6333/collections/mmrag_v2_8/points/count \
     -X POST -H "Content-Type: application/json" -d '{"exact":true}'
   # expected: count == (sum of unique chunk_ids across 34 v2.9 JSONLs) − (embed errors)
   ```
6. Spot-check image retrieval — query with a Source-Sanctity-safe
   prompt for a known image (e.g. wizard ornament from HARRY,
   F-35 photo from Combat) and confirm a hit on the corresponding
   image point.

**5d. v2.9 AFTER snapshot.**

1. Create `docs/QUALITY_SNAPSHOT_<v29_ship_date>_v2.9_after.md`
   following the v2.8 AFTER template
   (`docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md`):
   - Per-document audit table (BEFORE / AFTER / Delta).
   - Smoke matrix table.
   - Phase 1-4 empirical outcomes (target docs and their before/after
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

- The Phase 1-4 contract tests (already authored in those phases)
  must all pass.
- The corpus-scan parametrized test from Phase 1
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
  capped at the v2.8 cloud baseline of ~21.4%).

**Done when:**
- All 7 Goals from §2 are met empirically.
- v2.9 AFTER snapshot exists and `docs/PROJECT_STATUS.md` "Active
  Baseline" points at it.
- v2.8 AFTER snapshot
  (`docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md`) gets the
  `> ⚠ SUPERSEDED — historical reference only.` banner per the
  `docs/AGENT_GOVERNANCE.md` "Canonicality Rule (added 2026-05-04)".
- `mmrag_v2_8` Qdrant collection contains the v2.9 chunk_count;
  zero placeholder image points.
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
# 1. Smoke matrix — every row GATE_PASS + UNIVERSAL_PASS, no waivers
bash scripts/smoke_multiprofile.sh
# expected: 11/11 GATE_PASS + 11/11 UNIVERSAL_PASS.
# `GATE_PASS [form: ...]` for the SCAN0013 row is acceptable
# (per docs/QUALITY_GATES.md "Form / Invoice Acceptance Class")
# ONLY when document_type=form. Per AGENT-VAL-01 no other row
# may use the form lane as a workaround.

# 2. Full unit suite
pytest tests/ -q
# expected: ≥610 passed, 0 failed (596 baseline + 4 Phase 1 +
# 4 Phase 2 + 6 Phase 3 + 5 Phase 4 + 1 Phase 5 acceptance).

# 3. Per-doc audit — no FAIL row in the canonical 34
for doc in output/<v29_run>/*/ingestion.jsonl; do
  python scripts/qa_conversion_audit.py "$doc"
done
# expected: AUDIT_PASS (or FORM_AUDIT_PASS for the SCAN0013-class
# scanned forms) for all 34 canonical rows.

# 4. Universal invariants — zero hard fails on every output
for doc in output/<v29_run>/*/ingestion.jsonl; do
  python scripts/qa_universal_invariants.py "$doc"
done
# expected: UNIVERSAL_PASS on all 34.

# 5. Qdrant point-count verification
curl -sS http://localhost:6333/collections/mmrag_v2_8/points/count \
  -X POST -H "Content-Type: application/json" -d '{"exact":true}'
# expected: count == (unique chunk_ids across 34 v2.9 JSONLs) − (embed errors).

# 6. Image enrichment audit — zero placeholders
python scripts/vlm_quality_summary.py output/<v29_run>/<doc>/ingestion.jsonl --production
# expected for every doc: placeholder_ratio = 0%, vision_provider_used=qwen3-vl-plus
# for every non-fallback image chunk.

# 7. Tag
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

Apply each requirement explicitly to the v2.9.0 tag commit:

1. ✓ All §2 Goals met empirically (audit + Qdrant + smoke).
2. ✓ v2.9 AFTER snapshot is `tracked` in `docs/`; outputs are
   `local-run` with commands recorded in §3 / `convert_books.sh`.
3. ✓ Known limitations documented in v2.9 AFTER snapshot under
   "Known Limitations" (the deferred-conditional remote
   CodeFormulaV2 trigger; the v2.10 local-VLM swap; any
   Phase-5b hard-fallback image rate above zero).
4. ✓ Local VLM comparison is **explicitly removed from v2.9 scope**
   (deferred to v2.10 — see §2 Non-Goals; project memory pin).
5. ✓ `PROJECT_STATUS.md` "Active Baseline" points at the v2.9
   AFTER snapshot; `PROGRESS_CHECKLIST.md` flips updated; the
   v2.8 AFTER snapshot is banner-marked superseded per the
   Canonicality Rule.
6. ✓ A fresh agent reading `PROJECT_STATUS.md` →
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
  Phase 5d lands the AFTER snapshot.
- `docs/DECISIONS.md` — add entries for: (a) Phase 1 chunk_id
  generator includes position component, (b) Phase 2 refiner
  smart-routing gate on `has_encoding_corruption`, (c) Phase 4
  Firearms route resolution (path (a) or path (b) — record which).
- `CHANGELOG.md` — `[2.9.0]` entry summarizing all phases at the
  end (mirrors v2.8 [2.8.0] entry style).
- v2.8 AFTER snapshot
  (`docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md`) gets the
  superseded banner per the Canonicality Rule.
- `AGENTS.md` "Priority TODOs" list updated post-v2.9 to drop the
  closed items; document any `AGENT-SPATIAL-20` amendment if
  Phase 4 path (b) was taken.

**Test contract integrity** (per CLAUDE.md, `AGENTS.md` AGENT-TEST-01,
`docs/AGENT_GOVERNANCE.md` "Test Contract Rules"):

- The 7 existing `tests/test_classifier_digital_literature.py` tests
  must remain green through Phase 3.
- The 9 existing `tests/test_classifier_fallback.py` tests must
  remain green through Phase 4.
- The 6 existing `tests/test_qdrant_point_id_collision.py` tests
  must remain green through Phase 1 (the schema change must NOT
  break the Qdrant-layer `point_id` derivation contract).
- Workstream B negative tests
  (`tests/test_code_enrichment_decision.py`) are contracts: do not
  loosen.
- `tests/test_docling_postprocessor_acceptance.py` HARRY pages 13-30
  fixture passes through every phase.
- HARRY-class regression: HARRY MUST keep auto-routing to
  `digital_literature` after Phase 3 (Rule 0c tightening) and after
  Phase 4 (Firearms route fix).

**Upstream tracking:**
- HybridChunker per-item token guard remains an upstream-Docling ask.
- Remote CodeFormulaV2 (`RemoteCodeFormulaOptions` /
  `ApiCodeFormulaOptions`) — track Docling release notes for
  appearance.
- Local NuMarkdown-8B endpoint reachability — re-check before
  v2.10 planning.

## 7. Effort Summary

| Phase | Engineering estimate | Runtime / external | External dependency? |
|---|---|---|---|
| Phase 0 — Lock baseline | 30 min | — | Qdrant container up |
| Phase 1 — chunk_id collision fix | 1.5–2 h + 1 h tests | — | None |
| Phase 2 — Refiner smart-routing | 1–2 h + ~30 min verify | — | None |
| Phase 3 — Rule 0c tightening (Ayeva) | 2–4 h + ~10 min Ayeva re-conversion | — | None |
| Phase 4 — Firearms route fix | 3–6 h (path (a)); +1 day if (b) `AGENT-SPATIAL-20` amendment required | — | User sign-off (only if (b)) |
| Phase 5 — Broad reconversion + drop/recreate + VLM enrichment + AFTER snapshot | ~4 h script + snapshot | ~1–2 days conversion runtime; ~6–10 h cloud-VLM runtime; ~2.5 h CPU per code-heavy doc | Alibaba DashScope API + spend |
| **Total** | **~10–18 h engineering** (path (a)); **+1 day** if path (b) | **~2–3 days runtime** | Cloud VLM is the dominant external dependency; cost recorded on completion |

## 8. Decision Log

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
