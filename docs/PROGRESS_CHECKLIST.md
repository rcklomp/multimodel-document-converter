# Progress Checklist

Last updated: 2026-04-29

Purpose: durable handoff checklist for any coding model. Update this file as work is completed.

Legend:

- `[x]` done
- `[~]` in progress or partially done
- `[ ]` not started
- `[!]` blocked or needs decision

Status vocabulary and evidence classes are defined in `docs/AGENT_GOVERNANCE.md`.

## First 15 Minutes For A New Session

- [ ] Read `docs/PROJECT_STATUS.md`.
- [ ] Read this checklist.
- [ ] Read `AGENTS.md` for hard invariants.
- [ ] Run `git status --short` and identify pre-existing dirty files.
- [ ] Check whether the task touches VLM, code blocks, classifier, or conversion QA.
- [ ] Do not start broad conversion until targeted tests for the touched area pass.

## Baseline And Tracking

- [x] Create dated quality snapshot.
  - Evidence: `docs/QUALITY_SNAPSHOT_2026-04-29.md`
- [x] Add compact documentation index.
  - Evidence: `docs/README.md`
- [x] Add current-state handoff document.
  - Evidence: `docs/PROJECT_STATUS.md`
- [x] Add durable progress checklist.
  - Evidence: this file
- [ ] Add an automated baseline delta reporter.
  - Desired output: document-by-document old/new audit comparison
  - Suggested path: `scripts/compare_quality_snapshots.py` or `scripts/qa_delta_report.py`
- [ ] Create next dated quality snapshot after the next substantive fix.

## Workstream A: Model-Agnostic VLM Source Sanctity

Goal: VLM descriptions must describe visual structure without transcribing visible text, regardless of model.

Current status: `[~]` — `validated-cloud`, `local-pending`.

Evidence summary:

- Cloud provider tested: `qwen3-vl-plus` (2026-04-29).
- Local provider comparison: pending until local inference server is reachable.
- Blind-set evidence class: `reproducible-fixture`. Manifest is tracked at `tests/fixtures/blind_set_manifest.json`; assets are regenerated into ignored `output/`.
- Metric fix applied: `vlm_quality_summary.py` now reports `clean` and `sanitized` separately.

Completed:

- [x] Strengthened generic visual-only prompt.
- [x] Added text-reading validator patterns.
- [x] Added corrective retry prompt.
- [x] Added sanitizer for salvageable text-reading responses.
- [x] Added prompt-aware vision cache.
- [x] Added prompt harness: `scripts/eval_vlm_image_prompt.py`.
- [x] Added detector/sanitizer tests: `tests/test_vlm_text_detection.py`.
- [x] Verified unit tests: `32 passed`.
- [x] Verified compile check for modified VLM files and harness.
- [x] Add a small blind non-magazine image set to the prompt harness.
  - Manifest moved to tracked path: `tests/fixtures/blind_set_manifest.json`.
  - 16 images from 7 non-magazine docs. Asset paths point at `output/` (gitignored, runtime-regenerated).
  - Manifest documents how to regenerate assets. chunk_id now includes asset stem (collision fix).
- [x] Add a machine-readable VLM quality summary command.
  - Evidence: `scripts/vlm_quality_summary.py` — reads harness or production JSONL, outputs table or `--json`.
  - Now reports `clean` and `sanitized` as separate classes (fix applied 2026-04-29).
- [x] Emit structured VLM issue metadata into output JSONL.
  - `vision_validation_issues` field in `ChunkMetadata`. Populated at JSONL export for `vision_status=done` chunks.
- [x] Re-run PCWorld prompt harness with latest sanitizer state.
  - Evidence: `output/PCWorld_eval_qwen3vl_latest.jsonl` (126 images, qwen3-vl-plus, 2026-04-29).
  - Text-reading: 36.5% → 22.2%; hard fallback: 37.3% → 21.4%; truly clean: 77 (61.1%); sanitized: 22 (17.5%).
- [x] Run Combat Aircraft prompt harness with latest sanitizer state.
  - Evidence: `output/Combat_Aircraft_eval_qwen3vl_latest.jsonl` (18 images, qwen3-vl-plus, 2026-04-29).
  - 77.8% clean, 5.6% hard fallback, 3 timeout errors. Zero Combat-style hallucinations.
- [x] Run blind-set harness evaluation with VLM enabled.
  - Evidence: `output/blind_set_eval_qwen3vl.jsonl` (16 images, qwen3-vl-plus, 2026-04-29).
  - Truly clean: 75.0% (12/16); sanitized: 12.5% (2/16); hard fallback: 12.5%.
- [x] Confirm no text-reading regressions with cloud Qwen.
  - qwen3-vl-plus text-reading rate 22.2% (PCWorld) vs local NuMarkdown baseline 36.5%.
- [ ] Later: compare cloud Qwen3/Qwen3.5-VL with locally hosted Qwen3.5-VL.

Acceptance signals:

- [x] PCWorld text-reading hits decrease from baseline: 36.5% → 22.2%.
- [x] Hard fallbacks decrease: 37.3% → 21.4%. Sanitizer success rate 78.6%.
- [x] Combat-style hallucinations remain zero on measured patterns.
- [x] Blind non-magazine image set: 75% truly clean, 12.5% sanitized. Manifest now tracked.
- [x] `scripts/smoke_multiprofile.sh` GATE_PASS + UNIVERSAL_PASS (10/10 rows, latest 2026-04-29, `output/smoke_multiprofile_20260429_232651/`).

Open completion blockers:

- [ ] Run local VLM comparison when local inference server is reachable.

Do not:

- [ ] Do not add document-specific or filename-specific VLM rules.
- [ ] Do not allow VLM text transcription as a fallback.
- [ ] Do not use `--profile-override` for acceptance.

## Workstream B: Code Block Fidelity

Goal: code-heavy books preserve indentation and code structure without damaging non-code prose.

Current status: `[~]` — in progress; Docling-native code enrichment works qualitatively, but integration must be selective and remote-capable before acceptance.

Known failures:

- [~] `Ayeva_Python_Patterns`: CODE fail; indentation fidelity low; two infix artifacts. Custom fenced-flat detection exists but is not accepted as the primary fix until Docling-native evidence is checked.
- [ ] `Chaubal_PyTorch_Projects`: CODE fail; indentation fidelity low. Needs the same Docling-native comparison.

Known passing control:

- [x] `Fluent_Python_full_codex`: CODE pass; indentation fidelity high.
- [x] `Fluent_Python_full_vlm_codex`: CODE pass; indentation fidelity high.

Docling-native priority (2026-04-29):

- Official Docling docs describe code enrichment through `PdfPipelineOptions.do_code_enrichment`, disabled by default, processing `CodeItem` and setting `code_language`.
- Official example uses `CodeFormulaVlmOptions.from_preset(...)`, `CodeItem`, and `CodeItem.text` for extracted code blocks.
- Local Docling 2.86.0 exposes `do_code_enrichment`, `do_formula_enrichment`, `code_formula_options`, and `CodeFormulaVlmOptions`.
- Targeted Ayeva evidence from Claude session: Docling already emits `CodeItem`; raw `CodeItem.text` is flat because the PDF text layer is flat; `do_code_enrichment=True`/CodeFormulaV2 restores multiline/indented code for 21/22 items in a 10-page probe.
- Local CPU performance is not viable: roughly 18 minutes real time for 10 pages / 22 code items. `mlx_lm` is not installed; Torch MPS is available but not used by Docling's default auto-inline path.
- References: Docling enrichment features, pipeline options, and code/formula example.

Required architecture before further conversion:

- [x] Back out or replace any broad trigger that enables CodeFormulaV2 from `has_encoding_corruption` alone.
- [x] Add a cheap code-evidence decision pass using `CodeItem` count, code-chunk ratio, or sampled code-candidate regions.
- [x] Emit/use explicit `needs_code_enrichment=True` metadata with reason/counts; do not rely on profile or encoding corruption alone.
- [x] Preserve structural pathology flags when passing metadata from `BatchProcessor` to `V2DocumentProcessor`; avoid dropping evidence needed by either path.
- [ ] Refactor duplicated PDF extraction policy into a shared `PdfConversionPlan` + Docling adapter. `batch_processor.py` and `processor.py` must stop independently constructing Docling `PdfPipelineOptions`.
- [ ] Add bridge tests for every plan boundary: CLI -> plan, batch -> processor, processor -> adapter, adapter -> Docling options.
- [ ] Prefer remote-capable CodeFormulaV2 inference on stronger local-network hardware; cloud endpoint is second choice if data/cost policy allows; client local inference is diagnostic/fallback only.
- [ ] If Docling only supports document-level enrichment, enable it only after the cheap code-evidence pass identifies a code-heavy/code-candidate document.
- [ ] If region-level remote inference is implemented, send only `CodeItem`/code-candidate crops, not whole documents.
- [ ] Keep `_has_fenced_flat_code` as a provisional fallback marker only; do not let it mask whether native/remote code enrichment fixed the code.
- [x] Preserve Workstream B negative tests as contracts. Incidental shell commands, sparse fenced snippets, non-code magazines, and encoding corruption alone must not trigger CodeFormulaV2. Do not weaken assertions or rewrite fixtures to make a broad heuristic pass.

Current custom-path evidence (not accepted as final fix yet):

- Ayeva/Chaubal code blocks have their body lines squished onto a single line inside backtick fences.
  E.g.: ` ```python\nclass Foo: def bar(self): return 1\n``` ` — all one line.
- The existing `_is_flat_code_candidate` / `_is_flat_code` conditions checked `"\n" not in txt`, which is False
  when fences add newlines at chunk boundaries. The rescue never fired for Ayeva/Chaubal.
- Fluent Python worked because its flat code was truly newline-free (different corruption pattern).
- Added module-level `_has_fenced_flat_code(txt)` helper in `src/mmrag_v2/batch_processor.py`.
  Detects code chunks with fence markers whose body line is >120 chars with ≥2 Python keyword hits.
- Extended both detection conditions in `_is_flat_code_candidate` and `_is_flat_code` to also fire
  when `_has_fenced_flat_code(txt)` returns True.
- Added 11 targeted tests: `tests/test_fenced_flat_code_detection.py` — all pass.
- Confirmed: 85/85 fenced-flat code chunks in existing Ayeva output are now detected (all have bboxes).
- Full test suite: 337 passed, 1 skipped — no regressions.

Next steps:

- [x] Inspect failing code chunks in Ayeva.
- [x] Inspect failing code chunks in Chaubal.
- [x] Compare extraction/chunking behavior against Fluent Python.
- [x] Identify whether loss occurs in Docling extraction, chunking, code detection, refinement, or JSONL export.
- [x] Run targeted Docling-native `do_code_enrichment=True` probe on failing Ayeva pages.
- [x] Confirm Docling emits `CodeItem` for Ayeva and CodeFormulaV2 restores multiline/indented output in the targeted probe.
- [x] Fix the trigger design: no broad `has_encoding_corruption` CodeFormulaV2 activation; add/select a `needs_code_enrichment` lane.
- [ ] Decide remote inference target/protocol for CodeFormulaV2 (local-network preferred; cloud optional; client local fallback only).
- [x] Review all Workstream B test edits for AGENT-TEST-01 compliance before accepting implementation.
- [ ] Run targeted Chaubal pages through the selected lane.
- [ ] Run Fluent Python as non-regression control through the selected lane or prove why it is not enriched.
- [ ] Compare selected-lane output against current custom fenced-flat repair output.
- [ ] Add/update targeted tests for the accepted path.
- [ ] Re-run Ayeva and Chaubal samples only after page/region evidence supports the selected path.
- [ ] Re-run Fluent Python as a non-regression control.

Acceptance signals:

- [ ] Ayeva passes CODE gate.
- [ ] Chaubal passes CODE gate.
- [ ] Fluent Python remains AUDIT_PASS.
- [ ] Code chunk count does not collapse artificially.
- [x] `scripts/smoke_multiprofile.sh` remains green (10/10 rows, 2026-04-29, `output/smoke_multiprofile_20260429_232651/`).

## Workstream C: Combat Aircraft Text Corruption

Goal: full Combat Aircraft conversion passes audit without weakening gates.

Current status: `[ ]`

Known baseline:

- [ ] `output/Combat_Aircraft_full_promptfix_v2/ingestion.jsonl` fails TEXT.
- [ ] Current measured issues: `encoding_artifacts=48`, `high_corruption=79`.
- [x] Image descriptions improved; measured Combat-style hallucinations are zero.

Next steps:

- [ ] Inspect the highest-corruption chunks.
- [ ] Determine whether corruption comes from source PDF encoding, OCR/refiner path, table extraction, or magazine text-layer handling.
- [ ] Test whether corruption interceptor should apply differently to magazine text chunks.
- [ ] Add targeted regression tests for the discovered corruption pattern.
- [ ] Re-run a page-range conversion.
- [ ] Re-run full Combat conversion only after page-range evidence improves.

Acceptance signals:

- [ ] Full Combat conversion reports `AUDIT_PASS`.
- [ ] Image side remains `placeholder_ratio=0%`.
- [ ] No return of firearm/bolt/exploded-view hallucination pattern.
- [ ] No magazine-specific hardcoded word list.

## Workstream D: Classifier Correctness

Goal: documents route through plausible profiles using `ProfileClassifier`, without acceptance overrides.

Current status: `[ ]`

Known questionable routes:

- [ ] `business_form` smoke sample routes as `academic_whitepaper`.
- [ ] `HarryPotter_and_the_Sorcerers_Stone` routes as `digital_magazine`.
- [ ] `data_spreadsheet` routes as `technical_manual`.
- [ ] Some technical books route as `digital_magazine` or `academic_whitepaper`.

Next steps:

- [ ] Inspect `src/mmrag_v2/orchestration/profile_classifier.py`.
- [ ] Build evidence-based classifier tests from document features, not filenames.
- [ ] Add or update tests before changing classifier logic.
- [ ] Ensure routing uses content evidence and structural diagnostics, not layout alone.
- [ ] Re-run smoke matrix without `--profile-override`.

Acceptance signals:

- [ ] Questionable routes improve.
- [ ] Existing correct routes do not regress.
- [ ] Smoke matrix remains `GATE_PASS` plus `UNIVERSAL_PASS`.

## Workstream E: Placeholder Image Descriptions In Older Outputs

Goal: representative non-magazine digital outputs have real VLM descriptions or intentionally disabled VLM status, not accidental placeholders.

Current status: `[ ]`

Known state:

- [ ] Many older digital outputs pass audit with no-VLM advisory placeholder image descriptions.
- [x] Scanned/manual outputs such as Earthship and Firearms have real image descriptions and pass image gates.

Next steps:

- [ ] Pick representative non-magazine digital documents.
- [ ] Run image prompt harness before full reconversion.
- [ ] Decide whether full reconversion is needed or image enrichment can be applied safely.
- [ ] Audit image placeholder ratio after rerun.

Acceptance signals:

- [ ] Representative non-magazine outputs have `placeholder_ratio=0%` when VLM is enabled.
- [ ] VLM descriptions pass Source Sanctity validation.
- [ ] Text/table quality does not regress.

## Workstream F: Isolated Control-Character Artifact

Goal: remove the known control-character audit failure without broad text damage.

Current status: `[ ]`

Known failure:

- [ ] `A_comprehensive_review_on_hybrid_electri`: one control-character chunk in full existing output.

Next steps:

- [ ] Locate exact chunk and source page.
- [ ] Determine whether the current code already fixes it in smoke output.
- [ ] If not fixed, add a generic control-character cleanup rule at the right pipeline boundary.
- [ ] Add a regression test.

Acceptance signals:

- [ ] Full output audit passes.
- [ ] Smoke output remains clean.
- [ ] No legitimate Unicode text is stripped incorrectly.

## Standard Verification Commands

Targeted tests:

```bash
conda run -n mmrag-v2 python -m pytest tests/test_vlm_text_detection.py -q
conda run -n mmrag-v2 python -m py_compile src/mmrag_v2/vision/vision_prompts.py src/mmrag_v2/vision/vision_manager.py scripts/eval_vlm_image_prompt.py
```

Single-output audit:

```bash
conda run -n mmrag-v2 python scripts/qa_conversion_audit.py output/<run_name>/ingestion.jsonl
conda run -n mmrag-v2 python scripts/qa_universal_invariants.py output/<run_name>/ingestion.jsonl
```

VLM quality summary (harness output or production JSONL):

```bash
# From harness output:
conda run -n mmrag-v2 python scripts/vlm_quality_summary.py output/<harness_output>.jsonl
# From production output:
conda run -n mmrag-v2 python scripts/vlm_quality_summary.py output/<run_name>/ingestion.jsonl --production
# With baseline comparison:
conda run -n mmrag-v2 python scripts/vlm_quality_summary.py output/<new>.jsonl --baseline output/<old>.jsonl
# Machine-readable JSON:
conda run -n mmrag-v2 python scripts/vlm_quality_summary.py output/<file>.jsonl --json
```

Blind-set VLM evaluation:

```bash
conda run -n mmrag-v2 python scripts/eval_vlm_image_prompt.py --blind-set tests/fixtures/blind_set_manifest.json --output output/blind_set_eval.jsonl
conda run -n mmrag-v2 python scripts/vlm_quality_summary.py output/blind_set_eval.jsonl
```

Cross-profile acceptance:

```bash
conda run -n mmrag-v2 bash scripts/smoke_multiprofile.sh
```

Use the shell form already established in the project if `conda run ... bash` is not supported locally.

## Handoff Protocol

At the end of a session:

- [ ] Update this checklist.
- [ ] Update `docs/PROJECT_STATUS.md` if the next recommended step changed.
- [ ] Create or update a dated quality snapshot if quality numbers changed.
- [ ] Record exact output paths for new conversions or harness runs.
- [ ] Record commands that passed and commands that failed.
- [ ] Do not leave only chat-history breadcrumbs.
