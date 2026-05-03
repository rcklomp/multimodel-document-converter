# Plan: v2.8 — Close Known Production Gaps Before Broad Reconversion

**Status:** Draft v2, ratified for execution 2026-05-03
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
debt. **Each gap was inspected before drafting this plan**; the
phases below reference concrete patterns, not generic steps.

| Workstream | Symptom | Concrete pattern (verified 2026-05-03) | Last evidence |
|---|---|---|---|
| F — control-char artifact | `A_comprehensive_review_on_hybrid_electri` chunk i=3 page 1 fails `ctrl_chunks` audit | Codepoint **0x01 (SOH)** used as a keyword separator: `"Hybrid electric vehicle \x01 Hybrid energy storage system \x01 Architec..."`. Single chunk, one codepoint. | `output/A_comprehensive_review_on_hybrid_electri/ingestion.jsonl` |
| C — Magazine ornament-glyph corruption | Combat Aircraft full output: 50 chunks contain `�` runs in roster/squadron tables | Structured magazine-typography glyphs that PyMuPDF/Docling can't decode: `"�[il : ltJ! nfr! Ill r!·!�l�l:.[lr!Jl 1 ]1 r!' Dl="`. The actual squadron data is intact (`"Wing/Group Squadron Location = ... Aircraft = ... TailCode = ..."`); only the ornament rendering breaks. Concentrated on a few pages (e.g. page 66). Existing `CorruptionInterceptor` detects `�` but doesn't heal these — gating or re-extraction failure to investigate. | `output/Combat_Aircraft_full_promptfix_v2/ingestion.jsonl` |
| B — Chaubal `>>>` REPL-prompt flat-code | Chaubal_PyTorch_Projects: 279 code-flavored chunks; many show flattened code with `>>>` prompts but no newlines / indentation | Pattern: `">>> # Num epochs to wait for improvement before stopping patience_counter = 0 # How many ... for epoch in range(num_epochs): print(...) # --- Training -- model.train() ..."`. Squashed Jupyter / REPL output where line breaks were stripped. Existing `_has_fenced_flat_code` detector at `batch_processor.py:302` covers this pattern; current handling does not reconstruct indentation. | `output/Chaubal_PyTorch_Projects/ingestion.jsonl` |
| §5 followup | `processor.py:2072` was bypassing the adapter via raw `self._converter.convert(...)`; current static guard didn't catch it | Patched in commit `3bdbe0f` (2026-05-03). Guard test at `tests/test_pdf_conversion_plan.py::test_no_pipeline_options_construction_outside_adapter` only covers *construction*, not raw *invocation* on cached converters. Recurrence prevention not yet enforced. | n/a (architectural) |

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
- **Remote CodeFormulaV2 inference target.** Originally scoped for
  Phase 4; deferred after the Chaubal inspection showed the failure
  pattern is `>>>` REPL-prompt flat-code, which is reconstructible
  without remote inference. Remote CodeFormulaV2 remains a v2.9+
  option for documents where reconstruction is insufficient.

## 3. Phases

Phases ordered to front-load the cheapest concrete fixes (Phase 0-2)
so the snapshot baseline + early-win commits land before the deeper
investigations (Phases 3-4). Each phase is independently mergeable
per the same TDD red→green discipline used in
`docs/PLAN_DOCLING_POSTPROCESSOR.md`.

### Phase 0 — Lock the as-is baseline

**What:** Before changing any production code, snapshot every known
audit result for every document in `data/`. This is the
before-state for measuring v2.8 deltas. Without it, "did we improve
anything" becomes hand-waving.

**Steps:**
1. Run `scripts/qa_conversion_audit.py` on every existing
   `output/*/ingestion.jsonl`. Record per-document `AUDIT_PASS` /
   `AUDIT_FAIL` and the failure reason if any.
2. Capture the smoke matrix result
   (`/tmp/smoke_post_dl_v2_20260503/_summary.txt` already exists from
   2026-05-03; just preserve it).
3. Create `docs/QUALITY_SNAPSHOT_2026-05-03.md` with:
   - Per-document audit table (BEFORE column).
   - Smoke matrix table (BEFORE column).
   - Pointer to the post-Docling sanity pass + `digital_literature`
     profile shipped 2026-05-03 commits (`3bdbe0f`, `2f51816`,
     `379a733`).

**Done when:**
- `docs/QUALITY_SNAPSHOT_2026-05-03.md` exists.
- `docs/PROJECT_STATUS.md` "Active Baseline" pointer references it.

**Estimated effort:** 1 hour.

---

### Phase 1 — Workstream F: 0x01 keyword separator

**What:** Replace stray control-character codepoints used as
in-document separators with normal punctuation. Concrete case found:
`A_comprehensive_review_on_hybrid_electri` chunk i=3 page 1 has 0x01
(SOH) between keyword tokens. The codepoint is used **intentionally**
as a separator in the PDF source — strip-only would lose structure;
replacing with `; ` preserves the keyword list.

**Where:** Add at the chunk export boundary in
`src/mmrag_v2/mapper.py` (or
`src/mmrag_v2/validators/quality_filter_tracker.py` if that's where
chunk-text normalization lives) so it runs once at finalization.

**Heuristic:**
```
Replace runs of [\x00-\x08\x0B\x0C\x0E-\x1F]+ with "; " when the
surrounding text is a keyword/list pattern (no leading "if " or "def "
in adjacent context); otherwise replace with " " so legitimate
contexts are not damaged.
```

**Tests (red→green):**
- `test_ctrl_char_keyword_separator_replaced` — synthetic chunk
  `"Hybrid electric vehicle \x01 Hybrid energy storage system \x01 Architec"`
  → `"Hybrid electric vehicle; Hybrid energy storage system; Architec"`.
- `test_ctrl_char_in_code_chunk_replaced_with_space` —
  `"x = 1\x012\x013"` → `"x = 1 2 3"` (no semicolons inside code).
- `test_legitimate_unicode_passthrough` — em-dashes, smart quotes,
  CJK characters, and code newlines must NOT be touched.
- `test_a_comprehensive_review_chunk_passes_audit` — load the actual
  failing chunk fixture, run normalization, assert
  `qa_conversion_audit` reports `ctrl_chunks: 0`.

**Done when:**
- All four tests green.
- Re-running `qa_conversion_audit.py` on
  `A_comprehensive_review_on_hybrid_electri` reports
  `ctrl_chunks: 0`.
- Smoke run shows no other documents regressed (positive cases pass).

**Risk:** Low. The replacement rule is narrow; no impact outside
chunks containing actual control codes.

**Estimated effort:** 1-2 hours.

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
   — AST-walk every production `*.py` under `src/mmrag_v2/`. Flag any
   `Call` whose `func` is an `Attribute` with `attr == "convert"`
   AND whose `value` is an `Attribute` matching
   `self._converter` / `self._docling_converter` (deny list of cached
   converter attribute names) when the file isn't
   `engines/docling_adapter.py`.
2. Allow-list pattern: `self._adapter.convert(...)` is fine because
   the adapter wraps it.
3. Add a positive-case test that the guard fires on a synthetic
   bad-pattern source string passed via AST parser.

**Tests (red→green):**
- `test_no_raw_converter_invocation_outside_adapter` — passes on
  current code (clean since 2026-05-03).
- `test_guard_fires_on_synthetic_bypass` — synthetic source
  `"result = self._converter.convert(path)"` triggers the guard.
- `test_guard_does_not_fire_on_adapter_routing` — synthetic
  `"result = self._adapter.convert(path)"` does NOT trigger.

**Done when:**
- All three tests green.
- `docs/PLAN_DOCLING_POSTPROCESSOR.md` close-out note "A companion
  guard test should follow" can be checked off (cross-reference
  added).
- `docs/DECISIONS.md` "Shared PDF Extraction Plan" updated to note
  the guard now covers both construction AND invocation.

**Risk:** Low. The guard is AST-based, not regex-based, so it won't
false-positive on string literals or comments mentioning
`._converter.convert`.

**Estimated effort:** 1-2 hours.

---

### Phase 3 — Workstream C: magazine ornament-glyph cleanup

**What:** Combat Aircraft full output has 50 chunks containing
`�` runs from magazine-typography glyphs that PyMuPDF/Docling
can't decode. Inspection shows:
- The actual data is intact (`"Wing/Group Squadron Location = ...,
  Aircraft = ..., TailCode = ..."`).
- The corruption is a **structured ornament-glyph run** between
  data fields: `"�[il : ltJ! nfr! Ill r!·!�l�l:.[lr!Jl 1 ]1 r!' Dl= r!'"`.
- Concentrated on roster/squadron pages (e.g. page 66 has 5+
  affected chunks).
- The existing `CorruptionInterceptor` (`src/mmrag_v2/validators/corruption_interceptor.py`)
  has `�` in its `CORRUPTION_PATTERNS` regex but is **not
  healing these chunks**. Why is the first investigation step.

**Steps:**
1. **Investigate why CorruptionInterceptor isn't catching them.**
   Possibilities:
   - Gating: the interceptor only runs when
     `quarantine_corrupted_chunks` is True — check the magazine
     plan's `corruption_recovery_policy`.
   - Re-extraction failure: the per-bbox OCR returns equally
     corrupted text and the original is kept.
   - Bbox routing: magazine multi-column pages need column-aware
     bbox crops; the current single-bbox approach may straddle
     ornament glyphs and data.
   - Threshold: the interceptor may consider 13 `�` chars in a
     500-char chunk below its threshold.
2. **Decide intervention layer based on step 1.** Options:
   - **A. Strip-and-keep (cheapest):** Add a regex that strips
     ornament-glyph runs (long sequences of `�` mixed with
     `[il`, `ltJ!`, `nfr!`, etc.) at chunk-export time, leaving the
     surrounding data field intact.
   - **B. Re-OCR per-bbox (current architecture):** Fix the gating
     so CorruptionInterceptor runs on these chunks; verify
     re-extraction returns clean output.
   - **C. Promote to table chunk:** The data is structurally
     tabular (`Wing/Group Squadron Location = ..., Aircraft = ...,
     TailCode = ...`); re-emit as a `table` modality chunk with
     parsed rows.
3. **Add regression tests for the ornament-glyph pattern** before
   changing production code (per CLAUDE.md test-contract rule).
   Fixture: a chunk containing the actual page-66 corruption
   pattern.
4. **Re-run a page-range conversion** on Combat pages 60-70.
5. **Re-run full Combat conversion** only after page-range evidence
   improves.

**Tests (red→green):**
- `test_combat_page_66_ornament_glyphs_stripped` — fixture chunk
  with the page-66 pattern; assert stripped/healed output.
- `test_squadron_data_preserved` — same fixture; assert
  `"Wing/Group Squadron Location = NineteenthAir Force"` is intact.
- `test_no_false_positive_on_arabic_or_cjk` — non-ornament
  `�` (e.g. genuinely missing glyph from a CJK character) must
  NOT be stripped.

**Done when:**
- Targeted Combat pages 60-70 audit clean.
- Full Combat audit reports `encoding_artifacts: 0` and
  `high_corruption: 0`.
- `placeholder_ratio: 0%` (image side stays clean).
- No firearm/bolt/exploded-view hallucinations (Workstream A
  acceptance preserved).
- No magazine-specific hardcoded word lists (per PROGRESS_CHECKLIST
  acceptance signals).
- New regression test exists for the ornament-glyph pattern.

**Risk:** Stripping ornament glyphs may also strip legitimate
`�` from missing-glyph rendering in CJK or specialty content. The
test `test_no_false_positive_on_arabic_or_cjk` is the contract that
prevents this. If the investigation finds the corruption originates
in PyMuPDF's text extraction and Docling's OCR fallback can decode
it, the right fix is option B (re-OCR), not option A (strip).

**Estimated effort:** 0.5-2 days. Step 1 (investigate gating) is the
critical-path unknown.

---

### Phase 4 — Workstream B: Chaubal `>>>` flat-code reconstruction

**What:** Chaubal_PyTorch_Projects has 279 code-flavored chunks;
many show the `>>>` REPL-prompt flat-code pattern: `">>> # Num
epochs ... patience_counter = 0 # How many ... for epoch in
range(num_epochs):"`. Code lines were concatenated, line breaks
stripped, indentation lost. The existing
`_has_fenced_flat_code` detector at `batch_processor.py:302` covers
this pattern; the existing handling does not reconstruct
indentation.

**Reframed scope (vs. v1 of this plan):** The original Phase 4
called for deciding a remote CodeFormulaV2 inference target. After
inspecting the actual Chaubal output, the failure pattern is
**reconstructible without remote inference** — Python's tokenizer
can identify statement boundaries and indentation depth from the
flattened text. Remote CodeFormulaV2 remains a v2.9+ option for
documents where reconstruction is insufficient (e.g., obfuscated /
minified code, or non-Python languages).

**Approach:**
1. **Build a `>>>`-flat-code reconstructor.** Input: a chunk text
   containing `>>>` prompts and Python tokens. Output: properly
   newlined and indented code.
   - Strategy A: regex-based — split on `>>>` markers + `\n`-insert
     at known statement-terminator tokens (`:`, `\n# `, etc.).
     Limitations: doesn't handle multi-line strings, lambdas.
   - Strategy B: AST-based — repeatedly try `ast.parse` on
     candidate splits; backtrack if invalid.
     Higher-fidelity but expensive on multi-KB chunks.
   - Recommendation: Strategy A first, with Strategy B fallback when
     A produces output that fails `ast.parse`.
2. **Wire into the chunk export path** behind a plan field
   `code_reconstruction: "off" | "regex" | "ast_fallback"`. Default
   `"off"`; `technical_manual` profile with code evidence opts into
   `"ast_fallback"`.
3. **Run targeted Chaubal pages** through the reconstructor. Verify
   `indentation_fidelity` improves without breaking existing chunks.
4. **Run Fluent Python as non-regression control** (already passes
   CODE gate; must continue to pass).
5. **Re-run Chaubal full conversion** only after page evidence
   improves.

**Hard constraints (from CLAUDE.md / DECISIONS.md):**
- Do not enable `do_code_enrichment` from `has_encoding_corruption`
  alone.
- Do not loosen Workstream B negative tests (incidental shell
  commands, sparse fenced snippets, non-code magazines).
- Keep `_has_fenced_flat_code` clearly marked as detector;
  reconstruction is the new layer behind it. Do not let
  reconstruction mask whether native enrichment fixed the code (i.e.
  log when reconstruction was applied).

**Tests (red→green):**
- `test_reconstructor_handles_for_loop_pattern` — input
  `">>> for epoch in range(num_epochs): print(...) model.train()"`;
  expect 3 lines with proper indentation under `for`.
- `test_reconstructor_handles_comment_inline` —
  `">>> x = 1 # comment y = 2"`; expect two statements.
- `test_reconstructor_preserves_multiline_string` — multi-line
  triple-quoted string must not be split.
- `test_reconstructor_falls_back_to_input_on_unparseable` — when
  AST fallback fails, return original text (do not corrupt).
- `test_chaubal_page_252_code_chunk_reconstructed` — load actual
  failing chunk fixture; assert reconstructed output passes
  `ast.parse` (or returns original on failure).
- `test_fluent_python_code_chunks_unchanged` — non-regression
  control; reconstructed output is byte-identical to input on
  already-clean code.

**Done when:**
- Chaubal_PyTorch_Projects passes CODE gate
  (`indentation_fidelity >= 0.85`).
- Fluent Python remains AUDIT_PASS (non-regression control).
- Code chunk count does not collapse artificially (chunks may merge
  if reconstruction reveals shared blocks, but no >20% drop).
- New reconstructor module + tests exist.

**Risk:** AST-based fallback on long flattened code can be slow
(seconds per chunk on KB-scale text). Mitigation: cap input length
and timeout to fall back to regex-only, marked in chunk metadata.

**Estimated effort:** 1-2 days. No external dependency.

---

### Phase 5 — Acceptance: broad reconversion + Qdrant re-ingestion

**What:** With Phases 0-4 closed, run the broad reconversion that
`docs/PROJECT_STATUS.md` describes as the current strategic
objective. This produces the new corpus baseline and re-populates
Qdrant.

**Pre-flight checklist:**
- [ ] Full unit suite green (target: 600+ tests; today is 570).
- [ ] `bash scripts/smoke_multiprofile.sh` reports GATE_PASS +
      UNIVERSAL_PASS for every row. The 2026-05-03 SCAN0013 row
      currently fails micro_non_label_ratio because a small business
      form is the wrong probe; **before Phase 5 acceptance**, either:
  - (a) replace SCAN0013 with a real scanned book PDF (preferred),
  - (b) document the form-class exception in `docs/QUALITY_GATES.md`,
        OR
  - (c) make the gate form-aware (likely v2.9 work; document as
        accepted exception for v2.8).
- [ ] `tests/test_docling_postprocessor_acceptance.py` passes
      against a live HARRY conversion.

**Steps:**
1. **Decide Qdrant migration strategy.** Two options:
   - **Side-by-side:** Create a new collection
     `mmrag_v2_8` and ingest into it; keep the old collection until
     v2.8 is validated, then drop. Safest.
   - **Drop-and-recreate:** Drop existing collection, ingest fresh.
     Simpler but loses rollback. Pick this only if disk constraints
     force it.
   - Document the decision in `docs/DECISIONS.md`.
2. **Broad reconversion.** Process every document in `data/` through
   `mmrag-v2 batch` (or per-category equivalents). Record
   per-document audit status as the AFTER column for the
   `QUALITY_SNAPSHOT_2026-05-03.md` baseline.
3. **Qdrant re-ingestion.** Run `scripts/ingest_to_qdrant.py`
   against the new outputs. Confirm contextual-retrieval embedding
   text is correctly built (`build_contextualized_text` invariants).
4. **Snapshot the new baseline.** Create
   `docs/QUALITY_SNAPSHOT_<post-reconversion-date>.md` with
   per-document deltas vs `QUALITY_SNAPSHOT_2026-05-03.md`.
5. **Tag v2.8.0** if all Phase 5 steps pass.

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
| Remote CodeFormulaV2 inference target | Reframed: Phase 4's `>>>`-flat-code reconstructor handles the observed Chaubal pattern without remote inference. Remote remains a v2.9+ option for failure modes reconstruction can't cover (obfuscated / non-Python). | DECISIONS.md "Selective Code Enrichment Lane" |
| New profile types | None identified after `digital_literature` | — |
| New post-Docling stages | Reading-order, drop-cap, label-leak, OCR gating cover the observed Docling 2.86 failure modes | PLAN_DOCLING_POSTPROCESSOR.md |
| Form-aware audit gate (`micro_non_label_ratio`) | Surfaced by SCAN0013; document as exception for v2.8 acceptance, fix in v2.9 if forms become a routine target | docs/QUALITY_GATES.md |

## 6. Cross-phase concerns

**Documentation updates** (one PR per phase, batched into the
matching commit):
- `docs/PROGRESS_CHECKLIST.md` — flip `[ ]` items to `[x]` as each
  phase closes; record evidence path and test counts.
- `docs/PROJECT_STATUS.md` — refresh "Active Baseline" pointer when
  Phase 0 lands the snapshot, then again when Phase 5 lands the
  post-reconversion snapshot.
- `docs/DECISIONS.md` — Phase 3 (magazine corruption intervention),
  Phase 4 (`>>>`-flat-code reconstructor), Phase 5 (Qdrant migration
  strategy) each warrant a new decision entry.
- `CHANGELOG.md` — `[v2.8.0]` entry summarising all six phases at
  the end.

**Test contract integrity** (per CLAUDE.md):
- The `_has_fenced_flat_code` detector must not be removed during
  Phase 4; it stays as the upstream signal that triggers
  reconstruction.
- Workstream B negative tests are contracts: do not loosen.
- Each phase's regression test must cover the failure pattern, not
  just the specific chunk.

**Upstream tracking:**
- HybridChunker per-item token guard remains an upstream-Docling
  ask; not in scope.

## 7. Effort summary

| Phase | Estimate | External dependency? |
|---|---|---|
| Phase 0 — snapshot baseline | 1 h | No |
| Phase 1 — 0x01 keyword separator | 1-2 h | No |
| Phase 2 — adapter-invocation static guard | 1-2 h | No |
| Phase 3 — magazine ornament-glyph cleanup | 0.5-2 days | No |
| Phase 4 — Chaubal `>>>` flat-code reconstructor | 1-2 days | No |
| Phase 5 — broad reconversion + re-ingestion | 1-2 days | Conversion runtime |
| **Total** | **~3-7 days engineering + reconversion runtime** | None — dropped Phase 4 remote inference dependency |

## 8. Decision log

- **2026-05-03 v1** — Plan ratified. Scope chosen as the production
  blockers + the v2.7 §5 followup; everything else deferred to v2.9
  or later. Original Phase 4 framed as remote CodeFormulaV2 lane
  decision.
- **2026-05-03 v2** — Plan revised after inspecting actual failing
  outputs:
  - Phase 1 reframed: 0x01 is a **keyword separator**, not random
    corruption — replacement preserves structure.
  - Phase 3 reframed: Combat corruption is **structured magazine
    ornament glyphs in roster tables**, not generic encoding noise —
    investigation step added to determine why existing
    `CorruptionInterceptor` doesn't already heal them; intervention
    layer (strip vs re-OCR vs table-promote) decided by step 1.
  - Phase 4 reframed: Chaubal pattern is **`>>>` REPL-prompt
    flat-code**, reconstructible from Python tokenization without
    remote inference. Remote CodeFormulaV2 dropped from v2.8 scope;
    deferred to v2.9. Removes the only external dependency from the
    plan.
  - Phase 0 added: snapshot the baseline before changing code so
    deltas are measurable.
  - Phase 5 acceptance pre-flight now explicitly addresses the
    SCAN0013 calibration issue and Qdrant migration strategy.
  - Total estimate revised from 3-9 days → 3-7 days; external
    dependency removed.
