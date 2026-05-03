# Plan: v2.8 — Close Known Production Gaps Before Broad Reconversion

**Status:** Draft v3.2, ratified for execution 2026-05-03
**Owner:** ingestion pipeline
**Successor to:** `docs/PLAN_DOCLING_POSTPROCESSOR.md` (shipped 2026-05-03) and the v2.7 closure
**Related:** `docs/PROJECT_STATUS.md`, `docs/PROGRESS_CHECKLIST.md`,
`docs/QUALITY_GATES.md`, `docs/DECISIONS.md`

> **v2 → v3 (2026-05-03):** Per the CLAUDE.md "Libraries first, custom
> code last" principle, every phase was re-audited against Docling 2.86
> features. **Phase 4 was completely reframed**: Docling's
> `do_code_enrichment` + CodeFormulaV2 takes the rendered IMAGE of a
> code region and reconstructs proper indentation, overwriting
> `item.text`. Empirically verified on Chaubal pages 250-260 (14 code
> items go from 0 newlines flat to 6/22/39 newlines properly indented;
> ~27 sec/page CPU). The custom regex/AST reconstructor proposed in v2
> is dropped — it duplicates a feature Docling already ships. Phase 3
> investigation step expanded with empirical Docling OCR-fallback
> findings.

> **v3 → v3.1 (2026-05-03):** Added the **Parallel-Site Audit**
> cross-cutting principle (§2b) and a per-phase parallel-site audit
> step. The lesson: today's earlier `processor.py:2072` bypass
> happened because a single-site fix (the v2.7 §5 adapter refactor)
> wasn't audited against parallel call sites. This rewrite found
> several **existing fixes that may already cover the v2.8 gaps**:
> - Phase 1: `batch_processor.py:730-743` already defines
>   `_ctrl_table` and applies it unconditionally on chunk-dict
>   export. The failing A_comprehensive_review output is dated
>   2026-04-12, BEFORE this code shipped (commit `3bdbe0f`,
>   2026-05-03). Phase 1 step 0 is now: **re-run conversion and check
>   if Phase 1 work is even needed.**
> - Phase 2: `processor.py:2079` STILL contains a fallback
>   `else: result = self._converter.convert(...)` after today's bypass
>   patch. The new static guard must understand this conditional or
>   the fallback should be removed.
> - Phase 3: `batch_processor.py:3052` already invokes
>   `patch_corrupted_chunks(...)`. Phase 3 step 1 (gating
>   investigation) must check if this line is reached for Combat
>   pages, OR if it runs but doesn't heal.
> - Phase 4: `cli.py:1112` AND `cli.py:1741` both call
>   `decide_code_enrichment_for_pdf(...)` — the cheap-evidence trigger
>   exists in TWO places (process command + batch command). Both must
>   be audited.

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

## 2b. Parallel-Site Audit (cross-cutting principle)

**Lesson learned 2026-05-03 (post-Docling sanity pass):** A
single-site fix is suspect until parallel call sites in the
pipeline are audited. The post-Docling sanity pass shipped without
catching `processor.py:2072` for half a day because:
- v2.7 §5's static guard banned Docling option *construction*
  outside the adapter,
- but `processor.py:2072` was *invocation* of an already-cached
  converter (`self._converter.convert(...)`),
- so the post-processors were silently bypassed for any path that
  went through `V2DocumentProcessor.process_to_jsonl_atomic`
  (i.e. the `process` CLI command without batching).

**Mandatory step for every production-code phase in this plan:**
before designing a fix, walk the parallel call sites that touch the
same data. Ask:

1. Does the issue ALREADY have a fix elsewhere in the pipeline that
   the failing data simply hasn't been re-run through? (Check output
   timestamps vs. relevant commit dates.)
2. Does the existing fix have too narrow a gate (e.g. fires on
   `\x00` only when the bug surface is `\x01`-`\x1F`)?
3. Are there parallel boundaries (CLI process vs. CLI batch;
   `BatchProcessor` vs. `V2DocumentProcessor`; `engines/pdf_engine.py`)
   that need the same change?
4. Is there an upstream library config (Docling, EasyOCR, OcrMac)
   that already addresses the issue without custom code? (Per
   "Libraries first, custom code last" — CLAUDE.md.)

Each phase below has an explicit **Parallel-site audit** step listing
the specific files / line ranges to inspect before writing code.
Skipping the audit is the recurring failure mode the
`processor.py:2072` incident exposed.

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

**Parallel-site audit (do this FIRST):**

| Site | File:line | Current behavior | Action |
|---|---|---|---|
| Existing ctrl-table | `batch_processor.py:730` | Defines `_ctrl_table = {c: None for c in range(32) if c not in (10, 9)} \| {127: None}` — covers 0x01 | Reuse, do not redefine |
| Conditional gate | `batch_processor.py:731-732` | Only translates when `"\x00" in chunk.content` — gate too narrow | If 0x01 alone is the bug surface, broaden to `if any(0 <= ord(c) <= 31 and c not in '\n\t' for c in chunk.content)` OR rely on the unconditional pass at 740-743 |
| Unconditional gate | `batch_processor.py:740-743` | Translates dumped chunk content unconditionally — already strips 0x01 | Verify this path is reached for the failing chunk |
| Output age check | `output/A_comprehensive_review_on_hybrid_electri/ingestion.jsonl` | Timestamp 2026-04-12; predates the `_ctrl_table` code (commit `3bdbe0f`, 2026-05-03) | **Step 0: re-run conversion. Phase 1 may be a no-op.** |
| Other export paths | `processor.py:2505` (`content=text.strip()`); `mapper.py` `create_*_chunk(...)` | Each constructs chunks; need to confirm they all flow through the existing `_ctrl_table` site | If any export path bypasses `batch_processor`'s sanitizer, replace = `; ` semantics needs to land there too |

**Where (only if parallel-site audit confirms work is needed):**
Replace logic should live where the existing `_ctrl_table` is
applied (`batch_processor.py:740-743`). If the failing chunk
doesn't go through that path, the fix is to route it there, not to
add a parallel sanitizer.

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

**Parallel-site audit (do this FIRST):**

| Site | File:line | Current behavior | Action |
|---|---|---|---|
| Bypass site fixed today | `processor.py:2076-2079` | `if self._adapter is not None: result = self._adapter.convert(...) else: result = self._converter.convert(...)` | The else-branch fallback is a guard violation. Either remove (adapter is always set in production) or whitelist the conditional in the guard. |
| Indirect cached-converter use | `processor.py:1950` | `method = getattr(self._converter, method_name, None)` followed by `method(...)` if not None | Dynamic method lookup; guard must catch `getattr(self._converter, "convert", ...)` or document why it's safe. |
| BatchProcessor adapter use | `batch_processor.py:1364` | `result = self._adapter.convert(batch_path)` | Already correct; positive-case sanity check. |
| BatchProcessor get_converter sites | `batch_processor.py:1363, 2433` | `self._docling_converter = self._adapter.get_converter()` | Caches the converter; downstream call sites for `self._docling_converter.convert(...)` must be guarded too. |
| PDFEngine | `engines/pdf_engine.py:147+` | Holds `self._adapter: Optional[DoclingPdfAdapter]` and `self._converter` | Same pattern as processor.py; verify no `self._converter.convert(...)` slips through. |

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
3. **Decide what to do with `processor.py:2079` else-branch.** Either:
   - Remove the fallback (the adapter is always set when the
     processor handles a PDF; the else-branch is dead code).
   - Or wrap the `else` in a `# nocoverage: guard-allow` comment
     that the AST walker recognizes.
4. Add a positive-case test that the guard fires on a synthetic
   bad-pattern source string passed via AST parser.
5. Add a guard for the indirect `getattr(self._converter, ...)`
   pattern at `processor.py:1950` — either whitelist explicitly
   (after confirming it's safe) or block it.

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

**Docling-first findings (verified 2026-05-03):**

| Approach | Result on Combat p66 | Verdict |
|---|---|---|
| Baseline (`do_ocr=False`, current digital_magazine plan) | Docling extracts 2 text items via the broken text layer; replacement chars present | Current failure mode |
| `do_ocr=True` + `OcrMacOptions(force_full_page_ocr=True)` | 0 replacement chars BUT only 2 text items (a section header + page footer) | **Loses data:** the layout model under full-OCR is more conservative; roster table not re-extracted |
| Existing `CorruptionInterceptor` (per-bbox OCR) | Not running on these chunks (verified — full Combat output still has 50 `�` chunks) | **Right architecture, wrong gating** |

The Docling OCR options (`force_full_page_ocr`, `bitmap_area_threshold`)
trade fidelity for cleanliness — running OCR on the full magazine
page loses table/figure structure the layout model would otherwise
preserve. The existing `CorruptionInterceptor` (per-bbox OCR
patching) is the correct granularity. The bug is that it isn't
firing on the Combat chunks; investigation step is unchanged from
v2.

**Parallel-site audit (do this FIRST):**

| Site | File:line | Current behavior | Action |
|---|---|---|---|
| Interceptor invocation | `batch_processor.py:3052` | `from .validators.corruption_interceptor import patch_corrupted_chunks; all_chunks = patch_corrupted_chunks(all_chunks, self._current_pdf_path)` | Verify this line is reached for Combat (add a logger.info at entry). If not reached → gating bug; if reached → re-extraction bug. |
| Interceptor diagnostic check | `batch_processor.py:826-834` | Imports `has_encoding_artifacts` for a different downstream check | Confirm this isn't independently filtering the chunks in a way that hides whether the interceptor ran. |
| `quarantine_corrupted_chunks` policy | `engines/pdf_plan.py` `corruption_recovery_policy` derived bool | Magazine plans default to `quarantine` (gate IS on) | Confirm the magazine route doesn't override to `keep` somewhere. |
| OCR engine availability | `validators/corruption_interceptor.py` imports | Uses `pytesseract` | Apple Silicon may lack `tesserocr`/`tesseract`; check whether `OcrMacOptions` (Docling 2.86 native) is a drop-in replacement. |
| `V2DocumentProcessor` non-batch path | `processor.py` finalization | Does NOT appear to call `patch_corrupted_chunks` | Verify; if non-batch path bypasses the interceptor, that's an independent gap (parallel-site lesson again). |

**Steps:**
1. **Investigate why CorruptionInterceptor isn't catching them.**
   Use the audit table above as the checklist:
   - Confirm `patch_corrupted_chunks` is reached for Combat
     conversions (add temporary logger.info or inspect existing
     logs).
   - If reached: per-bbox OCR is failing. Try switching OCR engine
     to `OcrMacOptions` (faster on Apple Silicon, ships with Docling
     2.86, works without `tesserocr`).
   - If NOT reached: gating bug. Trace back to the magazine plan's
     `corruption_recovery_policy`.
   - Threshold check: 13 `�` in a 500-char chunk should easily
     trip `has_encoding_artifacts` (the regex matches a single
     `�`); if it doesn't, the bug is in the call path, not
     the threshold.
   - **Confirm V2DocumentProcessor non-batch path also runs the
     interceptor.** If it doesn't, that's a parallel-site bug
     (Combat happens to go through batch, but a future doc may
     not).
2. **Decide intervention layer based on step 1.** Options:
   - **A. Fix CorruptionInterceptor gating + switch to OcrMac
     engine (preferred — Docling-native, fixes existing
     architecture):** Per-bbox OCR is the right granularity; the
     interceptor exists; just needs to actually fire.
   - **B. Strip-and-keep (fallback):** If OCR re-extraction also
     produces garbage on the ornament-glyph regions, add a regex
     that strips `�` runs adjacent to ornament-glyph noise
     (`[il`, `ltJ!`, `nfr!`) at chunk-export time, leaving the
     surrounding data field intact.
   - **C. Promote to table chunk (deferred to v2.9):** The data is
     structurally tabular (`Wing/Group Squadron Location = ...,
     Aircraft = ..., TailCode = ...`); re-emit as a `table`
     modality chunk with parsed rows. Out of v2.8 scope (would
     need new table-promotion logic).
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

### Phase 4 — Workstream B: enable Docling's CodeFormulaV2 for code-heavy documents

**What:** Chaubal_PyTorch_Projects has 279 code-flavored chunks;
many show the `>>>` REPL-prompt flat-code pattern (`">>> # Num
epochs ... patience_counter = 0 # How many ... for epoch in
range(num_epochs):"`) — code lines concatenated, line breaks
stripped, indentation lost.

**Docling-first finding (verified 2026-05-03):** Docling 2.86
already solves this. `PdfPipelineOptions.do_code_enrichment=True`
activates `CodeFormulaModel`, which takes the rendered IMAGE of each
`CodeItem` region (at 120 DPI) and runs the CodeFormulaV2
vision-language model on it. The model returns reconstructed code
text and the model overwrites `item.text` (per
`docling/models/code_formula_model.py:CodeFormulaModel.__call__`).
The architecture decision in CLAUDE.md / DECISIONS.md
"Selective Code Enrichment Lane" already accounts for this.

**Empirical verification (Chaubal pages 250-260):**

| Run | `do_code_enrichment` | First code item text head | Newlines |
|---|---|---|---|
| Baseline | `False` | `"x = self.max_pool3(x) x = self.flatten(x) x = self.fc1(x) x = self.relu_fc1(x) ..."` | **0** |
| With CodeFormulaV2 | `True` | `"x = self.max_pool3(x)\n            x = self.flatten(x)\n            x = self.fc1(x)\n            x = self.relu_fc1(x)\n            ..."` | **6** |

All 14 code items in the slice were reconstructed (newlines went
from 0 → 6/22/39, indentation preserved, comments preserved,
function signatures intact). This is exactly the behavior the
custom reconstructor proposed in v2 was meant to deliver — Docling
already does it.

**Cost (verified):** ~295 seconds for 11 pages on CPU = ~27 sec/page.
Docling 2.86 explicitly removes MPS from supported devices for
`CodeFormulaModel`
(`Removing MPS from available devices because it is not in
supported_devices=[CPU, CUDA]`). For Chaubal's 359 pages that is
~150 minutes of CPU per full conversion.

**Parallel-site audit (do this FIRST):**

| Site | File:line | Current behavior | Action |
|---|---|---|---|
| Process command trigger | `cli.py:1112` | `needs_code_enrichment, code_reason, code_score = decide_code_enrichment_for_pdf(...)` | Verify it fires on Chaubal. |
| Batch command trigger | `cli.py:1741` | Same call in the batch path | Same — verify both paths agree. |
| Plan field consumer (adapter) | `engines/docling_adapter.py` `if self.plan.needs_code_enrichment: options.do_code_enrichment = True` | Already wired | Confirm adapter is actually invoked (Phase 2 covers this). |
| Plan field reader (V2DocumentProcessor) | `processor.py:279, 303` | Reads `needs_code_enrichment` from plan AND from raw metadata pop | Two different read paths; confirm they agree. |
| Plan field reader (BatchProcessor) | `batch_processor.py:617` | Reads from `plan.needs_code_enrichment` in `set_conversion_plan` | Single source. |
| Cheap-evidence trigger logic | `tests/test_code_enrichment_decision.py` | Existing tests pin the decision boundaries | Add a Chaubal-shaped feature vector; assert trigger fires. |
| MmragChunkingSerializerProvider interaction | `engines/docling_serializers.py` | Custom serializer suppresses picture labels | Verify it doesn't strip enriched code text (CodeItem text shouldn't go through the picture serializer, but worth sanity-checking the chunker's serialization order). |

**Approach (Docling-native, no new code module):**
1. **Confirm `needs_code_enrichment` cheap-evidence trigger fires
   on Chaubal.** The trigger lives in CLI / batch plan-builder code
   (TWO places per the audit above); verify both paths set
   `needs_code_enrichment=True` for Chaubal based on `CodeItem`
   count, code chunk ratio, or sampled regions. If either does not
   fire, fix the trigger.
2. **Run Chaubal full conversion with `needs_code_enrichment=True`.**
   Accept the ~150 min CPU cost as a one-off for v2.8's broad
   reconversion. Confirm `indentation_fidelity >= 0.85`.
3. **Run Fluent Python as non-regression control.** Fluent already
   passes CODE gate; CodeFormulaV2 enrichment must not regress it.
4. **Document the architectural decision in DECISIONS.md.** Specifically:
   "client-local CPU is acceptable for Chaubal-class documents at
   v2.8 because (a) CodeFormulaV2 fixes the failure cleanly, (b) MPS
   is unsupported by the model, (c) ~27 sec/page on CPU is tolerable
   for one-off broad reconversion. Remote CodeFormulaV2 inference
   remains v2.9+ scope when broader code-heavy corpus growth makes
   the runtime cost prohibitive."

**Hard constraints (from CLAUDE.md / DECISIONS.md, unchanged):**
- Do not enable `do_code_enrichment` from `has_encoding_corruption`
  alone.
- Do not loosen Workstream B negative tests (incidental shell
  commands, sparse fenced snippets, non-code magazines).
- Keep `_has_fenced_flat_code` clearly marked as fallback; do not
  let it mask whether native enrichment fixed the code (i.e. log
  when both fired so the diagnostic is preserved).

**Tests (red→green):**
- `test_chaubal_cheap_evidence_trigger_fires` — synthetic Chaubal-style
  diagnostic input must produce `needs_code_enrichment=True`.
- `test_fluent_python_cheap_evidence_trigger_fires` — non-regression
  control; Fluent already triggers (per
  `tests/test_code_enrichment_decision.py` patterns).
- `test_combat_aircraft_cheap_evidence_does_not_fire` — magazine with
  encoding corruption must NOT trigger code enrichment (per
  Workstream B negative-test contract).
- `test_chaubal_page_252_code_item_reconstructed` — fixture: load the
  Docling document state from a saved Chaubal slice with
  `do_code_enrichment=True`; assert at least one CodeItem.text
  contains `\n` (proves CodeFormulaV2 ran). Pure-test fixture; no
  live model invocation in the unit suite.

**Done when:**
- `tests/test_code_enrichment_decision.py` passes (existing) plus
  the four new tests above.
- Chaubal full conversion (one-off, run as part of Phase 5 broad
  reconversion) reports `indentation_fidelity >= 0.85`.
- Fluent Python non-regression control still passes CODE gate.
- DECISIONS.md updated with the Docling-native + client-local-CPU
  acceptance note.

**Risk:** CodeFormulaV2 inference is CPU-bound and slow at corpus
scale. Mitigation: gate on `needs_code_enrichment` so it only runs
on documents where the cheap-evidence trigger fires (Chaubal,
Ayeva, Fluent — not Combat or HARRY). For v2.8 broad reconversion
this is acceptable; v2.9 should investigate remote inference (no
upstream `RemoteCodeFormulaOptions` exists in Docling 2.86 today).

**Estimated effort:** 0.5-1 day engineering (mostly verifying the
cheap-evidence trigger fires correctly on Chaubal) + ~150 min CPU
runtime per Chaubal conversion in Phase 5.

---

### Phase 5 — Acceptance: broad reconversion + Qdrant re-ingestion

**What:** With Phases 0-4 closed, run the broad reconversion that
`docs/PROJECT_STATUS.md` describes as the current strategic
objective. This produces the new corpus baseline and re-populates
Qdrant.

**Pre-flight checklist:**
- [ ] Full unit suite green (target: 600+ tests; today is 570).
- [ ] `bash scripts/smoke_multiprofile.sh` reports GATE_PASS +
      UNIVERSAL_PASS for **every** row. **No waivers** (per
      AGENTS.md AGENT-VAL-01, CLAUDE.md "Acceptance is not complete
      unless GATE_PASS + UNIVERSAL_PASS are reported across all
      document categories", QUALITY_GATES.md "Every row must show
      GATE_PASS + UNIVERSAL_PASS"). The 2026-05-03 SCAN0013 row
      currently fails `micro_non_label_ratio=0.294 > 0.22` because a
      small business form is the wrong probe for a prose-calibrated
      gate. **Before Phase 5 acceptance, the smoke matrix MUST
      cleanly pass.** Two acceptable paths (in priority order):
  - (a) **Preferred:** replace `0013_140302111325_001.pdf` in the
        `scanned` slot with a representative scanned book/manual
        PDF that the existing gate accepts. The smoke matrix loses
        no coverage and no doc/code change is needed.
  - (b) **Fallback if (a) is not feasible:** make the gate
        form-aware in v2.8 scope (the §5 "out of scope" row for
        this is removed; see decision log v3.1 → v3.2). Requires a
        new acceptance class (`form` → relaxed `micro_non_label_ratio`)
        in `scripts/qa_conversion_audit.py` and a corresponding
        QUALITY_GATES.md entry. Triples Phase 5 effort.
  - **Documenting an exception is NOT an option.** Both AGENT-VAL-01
    and CLAUDE.md "Project Invariants" forbid waivers.
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
5. **Tag v2.8.0** when all Phase 5 steps pass AND every requirement
   in `docs/AGENT_GOVERNANCE.md` "Completion Rules" is satisfied:
   (1) every acceptance signal met; (2) evidence is `tracked` or
   `snapshot`; (3) known limitations documented; (4) required
   local/cloud comparisons completed or removed from scope;
   (5) `PROJECT_STATUS.md`, `PROGRESS_CHECKLIST.md`, and snapshots
   agree; (6) a fresh coding session can reproduce the claim
   without chat history.

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
# expected: EVERY row GATE_PASS + UNIVERSAL_PASS. No waivers per
# AGENTS.md AGENT-VAL-01 and CLAUDE.md "Project Invariants". If a
# row fails, fix the probe or the gate; do NOT add an exception.

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
| Remote CodeFormulaV2 inference target | Docling 2.86 does not expose `RemoteCodeFormulaOptions` / `ApiCodeFormulaOptions`; only the inline (client-local) `CodeFormulaModel` ships. Phase 4 accepts client-local CPU runtime (~27 sec/page) for v2.8 broad reconversion. v2.9 should investigate either (a) a custom adapter that intercepts CodeItems and pushes to a remote endpoint, or (b) waiting for upstream Docling to add remote options. | DECISIONS.md "Selective Code Enrichment Lane" |
| Custom `>>>`-flat-code reconstructor (was v2 Phase 4) | Dropped — Docling's CodeFormulaV2 reconstructs the indentation natively. Empirical verification on Chaubal pages 250-260 in v3. | This file, decision log v2→v3 |
| New profile types | None identified after `digital_literature` | — |
| New post-Docling stages | Reading-order, drop-cap, label-leak, OCR gating cover the observed Docling 2.86 failure modes | PLAN_DOCLING_POSTPROCESSOR.md |
| ~~Form-aware audit gate (`micro_non_label_ratio`)~~ | **REMOVED from out-of-scope.** Per CLAUDE.md "no waivers" and AGENTS.md AGENT-VAL-01, accepting an exception is not an option. Phase 5 pre-flight option (a) — replace SCAN0013 with a passing scanned PDF — is the preferred path. If (a) is infeasible, making the gate form-aware moves into v2.8 scope. | docs/QUALITY_GATES.md |

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
| Phase 3 — magazine ornament-glyph: fix CorruptionInterceptor gating + OcrMac engine | 0.5-1 day | No |
| Phase 4 — enable Docling CodeFormulaV2 for code-heavy docs | 0.5-1 day engineering + ~150 min CPU per Chaubal run in Phase 5 | No (v2.9 may revisit remote inference) |
| Phase 5 — broad reconversion + re-ingestion | 1-2 days runtime; CodeFormulaV2 adds ~2.5 hrs per code-heavy doc | Conversion runtime + CPU cost |
| **Total** | **~2-4 days engineering + reconversion runtime** | No external blockers; CPU runtime cost is the new dominant factor |

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
    flat-code**, prescribed a custom regex/AST reconstructor.
  - Phase 0 added: snapshot the baseline before changing code so
    deltas are measurable.
  - Phase 5 acceptance pre-flight now explicitly addresses the
    SCAN0013 calibration issue and Qdrant migration strategy.
  - Total estimate revised from 3-9 days → 3-7 days; external
    dependency removed.

- **2026-05-03 v3.2** — Coherence sweep against Layer 0 control docs
  (CLAUDE.md / AGENTS.md / DECISIONS.md / QUALITY_GATES.md /
  AGENT_GOVERNANCE.md / TESTING.md) surfaced three contradictions:
  1. **SCAN0013 "documented exception" framing dropped.** Plan v3.1
     §4 + §5 allowed waivers under "OR documented exception with
     rationale in QUALITY_GATES.md". This contradicted AGENT-VAL-01
     ("GATE_PASS + UNIVERSAL_PASS across all document categories"),
     CLAUDE.md "QA-CHECK-01 tolerance target is 0.10 for all
     profiles (no waivers)", and QUALITY_GATES.md ("These must be
     zero. No profile-specific waivers."). Phase 5 pre-flight now
     forces a binary choice: replace SCAN0013 with a passing scanned
     PDF (preferred) OR pull the form-aware audit gate into v2.8
     scope. Documenting an exception is forbidden.
  2. **Phase 4 client-local CPU framing kept; control docs amended.**
     Plan v3.1 Phase 4 prescribed client-local CPU CodeFormulaV2
     for v2.8 broad reconversion. This contradicted DECISIONS.md
     "Selective Code Enrichment Lane" ("client-local diagnostic /
     fallback only") and CLAUDE.md "Workstream B Code Enrichment
     Guardrail". Resolution: **DECISIONS.md got an "Amendment
     2026-05-03"** clarifying the original anti-pattern targeted
     custom MLX / transformer setups, not Docling 2.86's bundled
     CodeFormulaV2 (which runs at ~27 sec/page on CPU — fine for
     one-off batch). CLAUDE.md amended to match. Plan stays as-is
     and now references the amended DECISIONS entry.
  3. **Phase 5 tag criteria now reference AGENT_GOVERNANCE Completion
     Rules.** Plan v3.1 said "Tag v2.8.0 if all Phase 5 steps pass"
     without explicitly invoking the 6 binding requirements in
     `docs/AGENT_GOVERNANCE.md`. Phase 5 step 5 now lists them
     verbatim so v2.8.0 isn't tagged before evidence/snapshot/agreement
     are all in place.
  4. (Bonus) **TESTING.md HARRY path** updated independently — line
     26 still pointed at `data/scanned/...` after today's earlier
     `digital_literature` move.

- **2026-05-03 v3.1** — Added §2b "Parallel-Site Audit" cross-cutting
  principle and per-phase parallel-site audit tables. Triggered by
  feedback that the post-Docling sanity pass had to be patched
  (`processor.py:2072` bypass) because a single-site fix wasn't
  audited against parallel call sites. Ground-truth findings from
  the audit:
  - Phase 1: an existing `_ctrl_table` in `batch_processor.py:730-743`
    already handles 0x01. The failing
    `A_comprehensive_review_on_hybrid_electri` output is dated
    2026-04-12, BEFORE this code shipped (commit `3bdbe0f`,
    2026-05-03). Phase 1 step 0 is now: **re-run the conversion
    and check if Phase 1 work is even needed.** If the current code
    already strips 0x01, Phase 1 is a no-op + smoke verify.
  - Phase 2: `processor.py:2079` STILL contains a fallback
    `else: result = self._converter.convert(...)` after today's
    bypass patch. The new static guard must understand this
    conditional or the fallback should be removed.
  - Phase 3: `batch_processor.py:3052` already invokes
    `patch_corrupted_chunks(...)`. The bug is gating, not absence.
    Audit also found that `V2DocumentProcessor` non-batch path may
    not run the interceptor — independent gap to verify.
  - Phase 4: `cli.py:1112` AND `cli.py:1741` both call
    `decide_code_enrichment_for_pdf(...)` — the cheap-evidence
    trigger exists in TWO places (process command + batch command).
    Both must be audited.

- **2026-05-03 v3** — Per CLAUDE.md "Libraries first, custom code
  last", every phase re-audited against Docling 2.86 features:
  - **Phase 4 completely reframed.** Empirical test on Chaubal
    pages 250-260 confirmed Docling's `do_code_enrichment=True` +
    CodeFormulaV2 reconstructs `>>>` flat-code natively (14 code
    items go from 0 → 6/22/39 newlines, indentation preserved). The
    custom regex/AST reconstructor proposed in v2 is **dropped** —
    it duplicates a Docling-native feature. Phase 4 is now: ensure
    the existing `needs_code_enrichment` cheap-evidence trigger
    fires on Chaubal, accept the ~27 sec/page CPU cost (Docling
    forces CPU; MPS unsupported by CodeFormulaV2), document the
    architectural decision in DECISIONS.md.
  - **Phase 3 expanded with empirical Docling OCR-fallback
    results.** Tested `do_ocr=True` + `OcrMacOptions(force_full_page_ocr=True)`
    on Combat p66: produces 0 replacement chars but only 2 text
    items (data loss; layout model conservative under full-OCR).
    Confirms the existing `CorruptionInterceptor` (per-bbox OCR) is
    the right granularity. Added the Docling-native option of
    switching the interceptor's OCR engine to `OcrMacOptions` (the
    current implementation uses `pytesseract`).
  - Total estimate revised from 3-7 days → 2-4 days engineering
    (Phase 4 eng work shrinks because there's no new module to
    build) + CPU runtime in Phase 5.
  - Decision log addition: Docling 2.86 does NOT expose
    `RemoteCodeFormulaOptions` / `ApiCodeFormulaOptions`; only the
    inline `CodeFormulaModel` ships. Remote inference for code
    enrichment requires a custom adapter (deferred to v2.9) or
    upstream Docling work.
