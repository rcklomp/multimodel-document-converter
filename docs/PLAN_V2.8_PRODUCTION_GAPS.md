# Plan: v2.8 — Close Known Production Gaps Before Broad Reconversion

**Status:** Draft, ratified for execution 2026-05-03
**Owner:** ingestion pipeline
**Successor to:** `docs/PLAN_DOCLING_POSTPROCESSOR.md` (shipped 2026-05-03) and the v2.7 closure
**Related:** `docs/PROJECT_STATUS.md`, `docs/PROGRESS_CHECKLIST.md`,
`docs/QUALITY_GATES.md`, `docs/DECISIONS.md`

---

## 1. Why this plan exists

`docs/PROJECT_STATUS.md` describes the project's current strategic
objective as: *"quality-stabilization phase before broad reconversion
and Qdrant re-ingestion."* The post-Docling sanity pass + new
`digital_literature` profile shipped 2026-05-03 closed the most
visible quality bug class. What remains between today and broad
reconversion are three known production gaps and one architectural
debt:

| Workstream | Symptom | Last evidence |
|---|---|---|
| C — Magazine corruption | Combat Aircraft full output fails TEXT audit: `encoding_artifacts=48`, `high_corruption=79` | `output/Combat_Aircraft_full_promptfix_v2/ingestion.jsonl` |
| B — Code book fidelity | Chaubal_PyTorch_Projects fails CODE gate; remote CodeFormulaV2 lane undecided | `output/Chaubal_PyTorch_Projects/ingestion.jsonl` |
| F — Control-character artifact | `A_comprehensive_review_on_hybrid_electri` has one control-character chunk that fails audit | `output/A_comprehensive_review_on_hybrid_electri/ingestion.jsonl` |
| §5 followup | `processor.py:2072` was bypassing the adapter; static guard missed it | Patched in commit `3bdbe0f` (2026-05-03), but the static guard wasn't extended to prevent recurrence |

Closing all four leaves the corpus in a state where:
- Every document either passes audit or has a documented, accepted exception.
- The "single Docling options authority" architectural rule is enforceable, not aspirational.
- A broad reconversion + Qdrant re-ingestion can produce a clean baseline.

## 2. Goals & non-goals

**Goals**
- Close Workstreams B, C, F so the production corpus has zero
  unjustified audit failures.
- Promote the v2.7 §5 architectural rule from convention to enforced
  static guard.
- Produce a `QUALITY_SNAPSHOT_2026-05-03.md` baseline that today's
  shipped work + this plan's closures can be measured against.
- Reach a state where `bash scripts/smoke_multiprofile.sh` and a
  full-corpus audit report only documented exceptions.

**Non-goals**
- Local VLM swap (Workstream A): blocked on local inference server
  reachability; defer to v2.9.
- Older-output placeholder image cleanup (Workstream E): consumer of
  Workstream A; defer to v2.9.
- Automated baseline delta reporter (`scripts/qa_delta_report.py`):
  useful tooling but not a blocker; defer to v2.9.
- Broader UIR refactor (`PdfConversionPlan` → `UniversalDocument` →
  `ElementProcessor` flow): canonical target per CLAUDE.md but not
  required for broad reconversion; the legacy direct-to-chunk path
  remains acceptable as long as it doesn't expand.
- HybridChunker per-item token guard (Milestone 1 known limitation):
  requires upstream Docling work.
- New profile types or new post-Docling stages.

## 3. Phases

Phases ordered smallest → largest by complexity and dependency. Each
is independently mergeable per the same TDD red→green discipline used
in `docs/PLAN_DOCLING_POSTPROCESSOR.md`.

### Phase 1 — Workstream F: control-character cleanup

**What:** A single chunk in `A_comprehensive_review_on_hybrid_electri`
contains a literal control-character run that fails the audit's
`ctrl_chunks` check. Fix at the right pipeline boundary so legitimate
Unicode (line breaks in code, em-dashes, smart quotes) is not
collateral damage.

**Steps:**
1. Locate the exact chunk and source page in the failing JSONL.
2. Check whether the smoke output for the same document still has the
   artifact (some Milestone 1 cleanup may have already addressed it).
3. If still present, identify the source: PDF text layer, OCR output,
   refiner pass, or chunker concatenation.
4. Add a generic control-character cleanup at the appropriate seam
   (`validators/` if it's a chunk-level rule;
   `validators/corruption_interceptor.py` if it's the same class as
   encoding artifacts; `mapper.py` if it's an emission-time issue).
5. Add a regression test that covers the failure shape (not just the
   one chunk) so other documents with similar artifacts surface in
   the suite.

**Done when:**
- Full audit on `A_comprehensive_review_on_hybrid_electri` reports
  `ctrl_chunks: 0`.
- Smoke output for the same document still passes.
- A `tests/test_control_char_cleanup.py` (or equivalent) covers the
  artifact pattern.
- No legitimate Unicode is stripped (verified with a positive-case
  test on a known-good chunk that contains em-dashes, smart quotes,
  and code line breaks).

**Estimated effort:** 1-3 hours.

---

### Phase 2 — §5 followup: adapter-invocation static guard

**What:** Promote the v2.7 §5 rule "all Docling extraction goes
through `DoclingPdfAdapter`" from convention to enforced static guard.
Today the existing guard
(`tests/test_pdf_conversion_plan.py::test_no_pipeline_options_construction_outside_adapter`)
catches `PdfPipelineOptions(` and `DocumentConverter(` *construction*
outside the adapter. It misses raw `self._converter.convert(...)`
*invocation* on a cached converter — exactly how `processor.py:2072`
silently bypassed `apply_postprocessors` until the 2026-05-03 patch.

**Approach:**
1. Add `tests/test_pdf_conversion_plan.py::test_no_raw_converter_invocation_outside_adapter`
   — AST-walk every production `*.py` under `src/mmrag_v2/`; flag
   `Attribute.attr == "convert"` calls on identifiers whose name
   contains `_converter` (or matches a deny list of cached-converter
   attribute names) when the file isn't `engines/docling_adapter.py`.
2. Allow-list pattern: any call routed through `self._adapter.convert(...)`
   is fine because the adapter wraps it.
3. Add a positive-case test that the guard fires on a synthetic
   bad file.

**Done when:**
- Guard test exists, passes on current code (which is clean since
  2026-05-03), and fails on a synthetic regression.
- `docs/PLAN_DOCLING_POSTPROCESSOR.md` close-out section "A
  companion guard test should follow" is checked off.

**Estimated effort:** 1-2 hours.

---

### Phase 3 — Workstream C: magazine corruption interceptor

**What:** Combat Aircraft full output fails TEXT audit with
`encoding_artifacts=48` and `high_corruption=79`. Decide whether
`validators/corruption_interceptor.py` (per-bbox OCR patching for
encoding-corrupted chunks) needs magazine-specific handling, or
whether the corruption originates earlier (PDF text layer, table
extraction, magazine multi-column re-flow).

**Steps:**
1. **Inspect** the highest-corruption chunks in the existing failing
   output. Sample 5-10 chunks; record source page, modality, label,
   and the artifact pattern (PUA codepoints, replacement chars,
   missing whitespace, garbled em-dashes).
2. **Locate the source.** Run the failing pages through:
   - PyMuPDF text extraction (does the raw text layer have the
     artifact?)
   - Docling extraction with refiner OFF (does Docling introduce
     the artifact?)
   - The corruption interceptor in isolation (does its current rule
     touch the artifact at all?)
3. **Decide the intervention layer.**
   - If the artifact is in the PDF text layer and present in raw
     PyMuPDF: it's a source defect; the corruption interceptor must
     OCR-patch it. Likely needs magazine-aware bbox routing
     (multi-column pages need column-aware bbox crops).
   - If the artifact appears only after Docling: it's a Docling
     extraction issue; the post-Docling sanity pass might extend
     here, or the refiner threshold needs tuning.
   - If the artifact is in OCR output: refiner threshold or the OCR
     engine itself.
4. **Add regression tests for the discovered pattern** before
   changing production code (per CLAUDE.md test-contract rule).
5. **Re-run a page-range conversion** on the worst pages.
6. **Re-run full Combat conversion** only after page-range evidence
   improves.

**Done when:**
- `output/Combat_Aircraft_*/ingestion.jsonl` reports `AUDIT_PASS`.
- `placeholder_ratio=0%` (image side stays clean).
- No firearm/bolt/exploded-view hallucinations (Workstream A
  acceptance preserved).
- No magazine-specific hardcoded word lists (per PROGRESS_CHECKLIST
  acceptance signals).
- New regression test exists for the corruption pattern.

**Risk:** Magazine corruption may turn out to be the kind of
display-typography artifact that the post-Docling sanity pass touches
but doesn't fully cover. Be willing to extend the post-pass if the
investigation points there; don't force a fit into the existing
corruption interceptor architecture.

**Estimated effort:** 1-3 days (depends on what step 2 finds).

---

### Phase 4 — Workstream B: Chaubal CodeFormulaV2 remote lane

**What:** Chaubal_PyTorch_Projects still fails CODE gate
(indentation_fidelity low). The Workstream B architecture decision is
to use a selective Docling-native CodeFormulaV2 enrichment lane gated
on cheap code-evidence. Current state:

- `needs_code_enrichment` cheap-evidence trigger: shipped (Workstream B
  cheap-evidence trigger, complete).
- Document-level vs region-level remote inference: undecided.
- Remote inference target: undecided (local-network preferred, cloud
  optional, client-local MLX/transformers diagnostic-only).

**Steps:**
1. **Decide remote inference target/protocol.** Options:
   - Local-network: a stronger machine (e.g., a workstation with a
     larger GPU) running a CodeFormulaV2-compatible inference server.
   - Cloud: Docling's `ApiVlmOptions` / `CodeFormulaVlmOptions` with
     `enable_remote_services=True` pointing at a hosted endpoint.
   - Client-local MLX/transformers: documented as
     diagnostic/fallback only; not the production target.
2. **Decide region-level vs document-level inference.** Region-level
   sends only `CodeItem` / code-candidate crops; document-level sends
   the whole PDF. Region-level is preferred for cost + privacy; only
   fall back to document-level if Docling does not expose region-level
   for the chosen remote provider.
3. **Run targeted Chaubal pages** through the selected lane. Compare
   output side-by-side with the current
   `_has_fenced_flat_code` provisional fallback.
4. **Run Fluent Python as non-regression control** (already passes
   CODE gate; must continue to pass).
5. **Re-run Chaubal full conversion** only after page/region evidence
   improves.

**Hard constraints (from CLAUDE.md / DECISIONS.md):**
- Do not enable `do_code_enrichment` from `has_encoding_corruption`
  alone.
- Do not loosen Workstream B negative tests (incidental shell
  commands, sparse fenced snippets, non-code magazines).
- Use `code_enrichment.api_key` for remote services; no fallback to
  VLM/refiner/CLI keys.
- Keep `_has_fenced_flat_code` clearly marked as fallback; do not let
  it mask whether native/remote enrichment fixed the code.

**Done when:**
- Chaubal_PyTorch_Projects passes CODE gate.
- Fluent Python remains AUDIT_PASS (non-regression control).
- Code chunk count does not collapse artificially.
- Remote inference target is documented in `docs/DECISIONS.md` with
  the rationale and fallback policy.

**Risk:** External dependency on the selected remote inference
target. If neither local-network nor cloud can be stood up, this
phase blocks; the fallback is to ship v2.8 without Phase 4 and treat
Chaubal as a known limitation.

**Estimated effort:** 1-3 days for the inference target decision +
1 day for evidence runs. External-dependency risk dominates.

---

### Phase 5 — Acceptance: broad reconversion + Qdrant re-ingestion

**What:** With Phases 1-4 closed, run the broad reconversion that
`docs/PROJECT_STATUS.md` describes as the current strategic
objective. This produces the new corpus baseline and re-populates
Qdrant.

**Steps:**
1. **Pre-flight.** Confirm:
   - Full unit suite green (target: 600+ tests; today is 570).
   - `bash scripts/smoke_multiprofile.sh` reports GATE_PASS +
     UNIVERSAL_PASS for every row (including the SCAN0013 calibration
     issue from 2026-05-03 — either replace the smoke probe or
     accept the documented exception).
   - `bash scripts/acceptance_technical_manual.sh` clean.
   - `tests/test_docling_postprocessor_acceptance.py` passes against
     a live HARRY conversion.
2. **Snapshot the as-is baseline.** Create
   `docs/QUALITY_SNAPSHOT_2026-05-03.md` (or a later date) listing
   per-document audit status before the broad reconversion. Locks
   the before-state for delta comparison.
3. **Broad reconversion.** Process every document in `data/` through
   `mmrag-v2 batch` (or per-category equivalents). Record
   per-document audit status.
4. **Qdrant re-ingestion.** Run `scripts/ingest_to_qdrant.py` against
   the new outputs. Confirm contextual-retrieval embedding text is
   correctly built (`build_contextualized_text` invariants).
5. **Snapshot the new baseline.** Create
   `docs/QUALITY_SNAPSHOT_<date>.md` with per-document deltas vs the
   pre-reconversion snapshot.
6. **Tag v2.8.0** if all Phase 5 steps pass.

**Done when:**
- `docs/PROJECT_STATUS.md` "Active Baseline" pointer updated to the
  new snapshot.
- Qdrant collection contains contextualized chunks for every
  document.
- A v2.8.0 tag exists at the commit that reproduces the baseline.

**Estimated effort:** 1-2 days (mostly conversion runtime, not
engineering).

## 4. Acceptance gate (whole plan)

The plan is "done" when:

```bash
bash scripts/smoke_multiprofile.sh
# expected: every row GATE_PASS + UNIVERSAL_PASS, OR documented
# exception with rationale in docs/QUALITY_GATES.md.

pytest tests/ -v
# expected: 0 failed, 0 errored. The HARRY acceptance fixture
# (tests/test_docling_postprocessor_acceptance.py) is no longer
# env-gated by default — runs in CI against a cached JSONL.

python scripts/qa_conversion_audit.py output/<broad_reconversion>/<doc>/ingestion.jsonl
# expected for every document: AUDIT_PASS, OR a documented exception.

git tag v2.8.0
git push origin v2.8.0
```

Plus the human-readable check that:
- `output/<broad_reconversion>/Combat_Aircraft_*/ingestion.jsonl` has
  `encoding_artifacts: 0` and `high_corruption: 0`.
- `output/<broad_reconversion>/Chaubal_*/ingestion.jsonl` has
  `indentation_fidelity >= 0.85`.
- `output/<broad_reconversion>/A_comprehensive_review_on_hybrid_electri/ingestion.jsonl`
  has `ctrl_chunks: 0`.
- The Qdrant collection's chunk count matches the JSONL chunk count
  (no silent ingestion drops).

## 5. Out of scope (deferred to v2.9 or later)

| Item | Why deferred | Owner doc |
|---|---|---|
| Local VLM swap (Workstream A) | Blocked on local inference server reachability | PROGRESS_CHECKLIST Workstream A |
| Older-output placeholder cleanup (Workstream E) | Consumer of Workstream A | PROGRESS_CHECKLIST Workstream E |
| `scripts/qa_delta_report.py` automated delta reporter | Useful but not blocking | PROGRESS_CHECKLIST Baseline And Tracking |
| Broader UIR refactor (PdfConversionPlan → UniversalDocument → ElementProcessor) | Canonical target but not required for broad reconversion; legacy direct-to-chunk path acceptable if it doesn't expand | CLAUDE.md "Workstream B Code Enrichment Guardrail" |
| HybridChunker per-item token guard | Requires upstream Docling work | PROGRESS_CHECKLIST Milestone 1 known limitation |
| New profile types | None identified after `digital_literature` | — |
| New post-Docling stages | Reading-order, drop-cap, label-leak, OCR gating cover the observed Docling 2.86 failure modes | PLAN_DOCLING_POSTPROCESSOR.md |

## 6. Cross-phase concerns

**Documentation updates** (one PR per phase, batched into the
matching commit):
- `docs/PROGRESS_CHECKLIST.md` — flip `[ ]` items to `[x]` as each
  phase closes; record evidence path and test counts.
- `docs/PROJECT_STATUS.md` — refresh "Active Baseline" pointer when
  Phase 5 lands a new snapshot.
- `docs/DECISIONS.md` — Phase 3 (magazine corruption) and Phase 4
  (CodeFormulaV2 remote target) each warrant a new decision entry.
- `CHANGELOG.md` — `[v2.8.0]` entry summarising all five phases at
  the end.

**Test contract integrity** (per CLAUDE.md):
- The `_has_fenced_flat_code` provisional fallback must not be
  removed during Phase 4; it stays as fallback evidence even after
  remote enrichment lands.
- Workstream B negative tests are contracts: do not loosen.
- Phase 1's regression test must cover the failure pattern, not just
  the specific chunk.

**Upstream tracking:**
- Workstream B remote inference may end up depending on a Docling
  endpoint feature; record any such dependency in DECISIONS.md.
- HybridChunker per-item token guard remains an upstream-Docling
  ask; not in scope.

## 7. Effort summary

| Phase | Estimate | External dependency? |
|---|---|---|
| Phase 1 — control-character cleanup | 1-3 h | No |
| Phase 2 — adapter-invocation static guard | 1-2 h | No |
| Phase 3 — magazine corruption interceptor | 1-3 days | No |
| Phase 4 — Chaubal CodeFormulaV2 remote lane | 1-3 days + 1 day evidence | **Yes** (remote inference target availability) |
| Phase 5 — broad reconversion + re-ingestion | 1-2 days | Conversion runtime |
| **Total** | **~3-9 days engineering + reconversion runtime** | Phase 4 dominates risk |

## 8. Decision log

- **2026-05-03** — Plan ratified. Scope chosen as the production
  blockers + the v2.7 §5 followup; everything else deferred to v2.9
  or later. Rationale: project's stated strategic objective is
  "quality-stabilization phase before broad reconversion"
  (PROJECT_STATUS.md). v2.8 closes the known gaps so the broad
  reconversion produces a clean baseline. Tooling (delta reporter,
  static guards beyond the §5 followup) and dependent workstreams
  (local VLM, older-output cleanup) move to v2.9 because they
  multiply value but don't unblock reconversion.
