# Claude Prompt: Finish Refactor Boundary Work (Point #3)

**Task:** Close out the remaining boundary work from PLAN_V2.7 Section 5 ("Shared
PDF Extraction Plan and Adapter Refactor") — specifically Execution Order steps
**6** ("Delete or quarantine dead duplicated policy after parity tests pass")
and **7** ("Run targeted Workstream B probes, then the full unit suite, then
`smoke_multiprofile.sh`").

You are a Senior Python ETL Architect on the MMRAG V2 project at
`/Users/ronald/Projects/MM-Converter-V2.4.1`. Treat this as a close-out, not a
new feature build. Most of the boundary refactor is **already shipped** under
Milestones 1 + 2 (2026-05-01). Your job is to (a) verify that, (b) remove the
remaining dead duplicated policy, (c) lock the boundaries in with one
consolidated end-to-end bridge test, and (d) produce reproducible evidence per
`AGENT-EVIDENCE-01`.

---

## 1. Read First (in this order)

1. `docs/PROJECT_STATUS.md` — current objective and active baseline.
2. `docs/QUALITY_SNAPSHOT_2026-05-01.md` — closure evidence for Milestones 1+2.
3. `docs/PROGRESS_CHECKLIST.md` — Workstream B + Document Understanding rows.
4. `docs/PLAN_V2.7_DOCUMENT_UNDERSTANDING.md` — section 5 + Execution Order.
5. `docs/DECISIONS.md` — "Shared PDF Extraction Plan" + "Selective Code
   Enrichment Lane".
6. `AGENTS.md` — Level 0 invariants (`AGENT-VAL-01`, `AGENT-EVIDENCE-01`,
   `AGENT-STATUS-01`, `AGENT-DOCS-01`, `AGENT-TEST-01`).
7. `CLAUDE.md` — "Workstream B Code Enrichment Guardrail" and "Test Contract
   Integrity".

If any of these contradict this prompt, follow the docs and stop to flag the
contradiction.

---

## 2. Already Shipped — DO NOT REIMPLEMENT

Confirm each via the indicated artifact before touching anything. If something
listed below is **not** in fact shipped, stop and flag it; do not silently
extend scope.

| Item | Evidence to confirm |
|---|---|
| `PdfConversionPlan` typed policy object with `extraction_route`, `hybrid_chunker_enabled`, `max_chunker_input_chars`, `max_chunker_per_element_chars`, `allow_page_level_visuals`, `asset_validation_policy`, `corruption_recovery_policy`, derived `drop_blank_assets` / `quarantine_corrupted_chunks` properties, `__post_init__` validation. | `src/mmrag_v2/engines/pdf_plan.py` |
| `build_pdf_conversion_plan(...)` auto-selects `extraction_route` from modality/profile/image_density and derives `hybrid_chunker_enabled` + `allow_page_level_visuals` from the route. | `src/mmrag_v2/engines/pdf_plan.py` (function body) |
| `DoclingPdfAdapter` is the only production code that constructs `PdfPipelineOptions` / `DocumentConverter`. | `src/mmrag_v2/engines/docling_adapter.py` + `tests/test_pdf_conversion_plan.py::test_no_pipeline_options_construction_outside_adapter`, `::test_no_production_docling_imports_outside_adapter` |
| HybridChunker route + total-text + per-element guards in `V2DocumentProcessor`. | `src/mmrag_v2/processor.py` ~lines 2110-2160 + `tests/test_chunker_guard.py` (11 tests) |
| Bridge tests at every boundary (CLI → plan, batch → processor, processor → adapter, plan field round-trip). | `tests/test_pdf_conversion_plan.py::test_batch_plan_route_fields_bridge`, `::test_processor_plan_route_fields_bridge`, `::test_processor_plan_new_fields_bridge`, `::test_batch_plan_to_processor_all_flags_bridge`, `::test_batch_plan_new_fields_bridge` |
| Latest smoke-matrix evidence: 10/10 `GATE_PASS` + `UNIVERSAL_PASS`. | `output/smoke_multiprofile_20260501_120514/_summary.txt` |

If your audit (Step 1 below) shows any of those broken, the right fix is to
restore the missing piece — not to re-derive the whole refactor.

---

## 3. What You Must Deliver

### Step 1 — State Audit (no code changes)

Produce a short audit (write it as a numbered list in your response, do **not**
add a new doc) covering:

1. Run the static guards and confirm both pass:
   ```bash
   conda run -n mmrag-v2 python -m pytest \
     tests/test_pdf_conversion_plan.py::test_no_pipeline_options_construction_outside_adapter \
     tests/test_pdf_conversion_plan.py::test_no_production_docling_imports_outside_adapter -q
   ```
2. Run the focused plan + chunker-guard suites:
   ```bash
   conda run -n mmrag-v2 python -m pytest \
     tests/test_pdf_conversion_plan.py tests/test_chunker_guard.py -q
   ```
3. Inventory remaining dead duplicated policy. Search for and report the call
   sites of:
   - `set_intelligence_metadata` (deprecated `[V2.8-COMPAT]` per
     `batch_processor.py`)
   - `_intelligence_metadata` writes that are not derived from
     `plan.chunk_factory_metadata()` or `plan.to_intelligence_metadata()`
   - Any module other than `docling_adapter.py` importing `from docling...`
     production-side (allow tests + the adapter itself)
4. Confirm CLI `_build_conversion_plan_from_metadata()` in `src/mmrag_v2/cli.py`
   correctly relies on auto-route detection (it does not need to forward
   typed policy fields explicitly today — auto-detect is the contract). If you
   find a CLI flag that *should* override a typed field but doesn't, list it;
   do not add new CLI flags speculatively.

End the audit with a concise verdict per item: **DONE / NEEDS-CLEANUP /
NEEDS-BRIDGE-TEST / OUT-OF-SCOPE**.

### Step 2 — Delete or Quarantine Dead Duplicated Policy

Only after the audit, do the surgical removals justified by it. Expected
candidates — confirm before acting:

- **`BatchProcessor.set_intelligence_metadata(...)`** — currently marked
  `[V2.8-COMPAT]`. If no production call site uses it (CLI uses
  `set_conversion_plan(...)`), remove it together with its bookkeeping. Keep
  any tests that exercise the *plan-builder* path; remove tests that exist only
  to cover the deprecated path.
- **Independently mutable `_intelligence_metadata` dict** — make it strictly
  derived from the active `PdfConversionPlan`. Concretely: `_intelligence_metadata`
  must be assigned only via `plan.chunk_factory_metadata()` or
  `plan.to_intelligence_metadata()`. No callers may reach in and overwrite keys.
- **Any overlapping bool field on `BatchProcessor` / `V2DocumentProcessor`**
  that duplicates a `PdfConversionPlan` policy attribute. Replace reads with
  `self._conversion_plan.<field>` (or the existing private mirror set in
  `set_conversion_plan`/`__init__`); do not introduce a third copy.

Hard rules for this step:
- **Surgical changes only.** Do not refactor adjacent code.
- **No test weakening.** If a deletion would force a test rewrite, stop and
  flag it. Per `AGENT-TEST-01`, tests are contracts.
- **No new public API.** This is a deletion pass.
- **Preserve the legacy bridges** (`drop_blank_assets`,
  `quarantine_corrupted_chunks`) as `@property`s on `PdfConversionPlan`. They
  exist on purpose for backward compatibility.

### Step 3 — Consolidated End-to-End Boundary Bridge Test

Add **one** new test to `tests/test_pdf_conversion_plan.py`:

```
test_all_typed_policy_fields_round_trip_full_chain
```

Behavior:
- Build a `PdfConversionPlan` with **non-default** values for every typed
  policy field (`extraction_route="technical_manual"`,
  `hybrid_chunker_enabled=False`, `max_chunker_input_chars=321_000`,
  `max_chunker_per_element_chars=42_000`, `allow_page_level_visuals=True`,
  `asset_validation_policy="quarantine"`, `corruption_recovery_policy="recover"`).
- Pass it via `BatchProcessor.set_conversion_plan(...)` and have BatchProcessor
  construct a `V2DocumentProcessor` for a synthetic batch (use the same
  monkeypatching pattern as `test_batch_with_plan_uses_adapter`).
- Assert that **on every downstream object** the typed values are visible
  unchanged: `BatchProcessor._extraction_route`, `_hybrid_chunker_enabled`,
  `_max_chunker_input_chars`, `_max_chunker_per_element_chars`;
  `V2DocumentProcessor._extraction_route`, `_hybrid_chunker_enabled`,
  `_max_chunker_input_chars`, `_max_chunker_per_element_chars`,
  `_drop_blank_assets`, `_quarantine_corrupted_chunks` (derived);
  `DoclingPdfAdapter` receives a plan whose
  `asset_validation_policy == "quarantine"` and
  `corruption_recovery_policy == "recover"`.
- Capture the adapter via the same monkeypatched factory the existing bridge
  tests use; do **not** run a real Docling conversion.

Why it's worth one new test even though field-by-field bridges exist: it is the
single guard that fails loudly the day someone adds a new typed field but
forgets to plumb it. Treat it as drift insurance.

### Step 4 — Regression Evidence (per `AGENT-EVIDENCE-01`)

Run, in order, and capture the exact terminal results:

```bash
# Static guards
conda run -n mmrag-v2 python -m pytest \
  tests/test_pdf_conversion_plan.py::test_no_pipeline_options_construction_outside_adapter \
  tests/test_pdf_conversion_plan.py::test_no_production_docling_imports_outside_adapter -q

# Focused boundary suite
conda run -n mmrag-v2 python -m pytest \
  tests/test_pdf_conversion_plan.py tests/test_chunker_guard.py \
  tests/test_corruption_quarantine.py tests/test_blank_asset_quarantine.py \
  tests/test_finalization_bridge.py -q

# Full unit suite (must remain green)
conda run -n mmrag-v2 python -m pytest -q

# Targeted Workstream B / probe — RAG Guide (was the Milestone 1 unblock)
conda run -n mmrag-v2 python -m mmrag_v2.cli process \
  "data/technical_manual/A Simple Guide to Retrieval Augmented Generation Kimothi A. 2025.pdf" \
  --output-dir output/probe_boundary_closeout_rag_guide \
  --batch-size 10 --vision-provider none --no-refiner --no-cache
conda run -n mmrag-v2 python scripts/qa_conversion_audit.py output/probe_boundary_closeout_rag_guide/ingestion.jsonl
conda run -n mmrag-v2 python scripts/qa_universal_invariants.py output/probe_boundary_closeout_rag_guide/ingestion.jsonl

# Acceptance smoke matrix (AGENT-VAL-01)
conda run -n mmrag-v2 bash scripts/smoke_multiprofile.sh
```

Acceptance:
- Static guards: `2 passed`.
- Focused suites: pass with no skips other than the one tracked skip.
- Full unit suite: at least the Milestone 2 baseline (`484 passed, 1 skipped, 0 failed`).
- Probe: `AUDIT_PASS` + `UNIVERSAL_PASS`. No regression vs. the
  `output/probe_milestone2_rag_guide/` baseline (chunk count and `infix_strict=0`
  must hold).
- Smoke matrix: 10/10 rows `GATE_PASS` + `UNIVERSAL_PASS`, including the
  Greenhouse blind-test document (`AGENT-VAL-01`).

If any row fails, **fix the implementation, never the test or the gate**.

### Step 5 — Documentation Update

Per `AGENTS.md` §4 and `AGENT-DOCS-01` (no new docs, extend existing ones):

1. Append a "Refactor Boundary Closeout" section to
   `docs/QUALITY_SNAPSHOT_<today>.md` (create today's dated snapshot only if
   numbers changed; otherwise extend the most recent one). Include the four
   evidence blocks (static guards, focused suites, full suite, smoke + probe)
   in the existing `Class / Command / Input / Output / Result / Tracked /
   Limitations` format used by `QUALITY_SNAPSHOT_2026-05-01.md`.
2. Update `docs/PROGRESS_CHECKLIST.md`:
   - Mark "Delete or quarantine dead duplicated policy" complete under the
     Workstream B Architecture status block.
   - Add the new bridge test under "Bridge tests".
3. Update `docs/PROJECT_STATUS.md` "Latest validation" list with one line for
   this closeout.
4. Do **not** add new governance docs. Do **not** add a parallel plan.

---

## 4. Hard Constraints (non-negotiable)

1. **No new Docling construction.** Zero new `PdfPipelineOptions(` or
   `DocumentConverter(` outside `src/mmrag_v2/engines/docling_adapter.py`. The
   static guard tests must stay green.
2. **No UIR rewrite in this task.** Section 5's "delete the legacy
   Docling-item-to-chunk path when parity is proven" is the **next**
   workstream, not this one. If you find yourself rewriting `processor.py`'s
   element mapping, stop.
3. **No test weakening** (`AGENT-TEST-01`). Negative tests, regression tests,
   and acceptance fixtures are executable requirements.
4. **No new typed policy field.** Milestone 2 closed the policy vocabulary; do
   not extend it without an explicit decision in `docs/DECISIONS.md`.
5. **No new CLI flag.** This task forwards the existing typed policy through
   the existing factory; it does not add user-facing surface.
6. **`--profile-override` is for debugging only** — never use it in acceptance
   evidence runs.
7. **Python 3.10 only.** Apple Silicon target. `docling==2.86.0` exact-pin.
8. **Batch size ≤10 pages** for any probe.
9. **No filename- or document-specific rules.**
10. **Surgical scope only.** Touch only the files this work demands. Do not
    "improve" formatting, comments, or unrelated code.

---

## 5. Self-Audit Checklist (before reporting complete)

Code:
- [ ] `set_intelligence_metadata(...)` removed (or, if a real caller exists,
      explicitly justified in the snapshot section with the call site).
- [ ] `_intelligence_metadata` is sourced exclusively from
      `PdfConversionPlan.{chunk_factory_metadata, to_intelligence_metadata}`.
- [ ] No third copies of policy bools added; reads go through
      `self._conversion_plan` or its private mirrors.
- [ ] No new public API; no new CLI flags; no new typed fields.

Tests:
- [ ] `test_all_typed_policy_fields_round_trip_full_chain` added and passing.
- [ ] All pre-existing bridge / static-guard / chunker-guard tests still pass
      unmodified.
- [ ] No test was rewritten to fit the implementation.

Evidence (`AGENT-EVIDENCE-01`):
- [ ] Static guards: `2 passed` (terminal output captured).
- [ ] Focused boundary suite: pass output captured.
- [ ] Full unit suite: ≥484 passed, captured.
- [ ] RAG Guide probe: `AUDIT_PASS` + `UNIVERSAL_PASS`, output path recorded.
- [ ] Smoke matrix: 10/10 `GATE_PASS` + `UNIVERSAL_PASS`, summary path
      recorded, Greenhouse blind-test included.

Docs:
- [ ] `docs/QUALITY_SNAPSHOT_*` updated/added with the four evidence blocks.
- [ ] `docs/PROGRESS_CHECKLIST.md` updated.
- [ ] `docs/PROJECT_STATUS.md` "Latest validation" extended by one line.
- [ ] No new governance doc created.

Status word (`AGENT-STATUS-01`):
- [ ] Use `complete` only if local validation, durable evidence, and the
      smoke matrix all passed in this run. Otherwise use `implemented` plus a
      one-line blocker.

---

## 6. Execution Order

1. Read the docs in §1.
2. Audit (Step 1) — produce the verdict table.
3. Surgical deletions (Step 2) — only what the audit justifies.
4. Add the consolidated bridge test (Step 3).
5. Run all evidence commands (Step 4).
6. If anything fails, fix the implementation and rerun from the failing point.
7. Update docs (Step 5).
8. Report: short summary + verdict table + evidence paths + checklist state.

If the audit shows nothing to delete and the bridge test is the only missing
piece, that is a valid outcome — make the change, capture evidence, and close
out. Do not invent work.
