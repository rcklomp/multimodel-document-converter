# Progress Checklist

Last updated: 2026-04-29

Purpose: durable handoff checklist for any coding model. Update this file as work is completed.

Legend:

- `[x]` done
- `[~]` in progress or partially done
- `[ ]` not started
- `[!]` blocked or needs decision

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

Current status: `[~]`

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

Still needed:

- [ ] Re-run PCWorld prompt harness with latest sanitizer state.
- [ ] Run Combat Aircraft prompt harness with latest sanitizer state.
- [ ] Add a small blind non-magazine image set to the prompt harness.
- [ ] Add a machine-readable VLM quality summary command.
- [ ] Decide whether VLM enforcement should emit structured issue metadata into output JSONL.
- [ ] Run production conversion sample after latest sanitizer additions.
- [ ] Confirm no text-reading regressions with cloud Qwen.
- [ ] Later: compare cloud Qwen3/Qwen3.5-VL with locally hosted Qwen3.5-VL.

Acceptance signals:

- [ ] PCWorld text-reading hits decrease from the baseline in `docs/QUALITY_SNAPSHOT_2026-04-29.md`.
- [ ] Hard fallbacks decrease or remain justified; do not trade all leakage for useless descriptions.
- [ ] Combat-style hallucinations remain zero on measured patterns.
- [ ] Blind non-magazine image set does not regress.
- [ ] `scripts/smoke_multiprofile.sh` still reports `GATE_PASS` and `UNIVERSAL_PASS`.

Do not:

- [ ] Do not add document-specific or filename-specific VLM rules.
- [ ] Do not allow VLM text transcription as a fallback.
- [ ] Do not use `--profile-override` for acceptance.

## Workstream B: Code Block Fidelity

Goal: code-heavy books preserve indentation and code structure without damaging non-code prose.

Current status: `[ ]`

Known failures:

- [ ] `Ayeva_Python_Patterns`: CODE fail; indentation fidelity low; two infix artifacts.
- [ ] `Chaubal_PyTorch_Projects`: CODE fail; indentation fidelity low.

Known passing control:

- [x] `Fluent_Python_full_codex`: CODE pass; indentation fidelity high.
- [x] `Fluent_Python_full_vlm_codex`: CODE pass; indentation fidelity high.

Next steps:

- [ ] Inspect failing code chunks in Ayeva.
- [ ] Inspect failing code chunks in Chaubal.
- [ ] Compare extraction/chunking behavior against Fluent Python.
- [ ] Identify whether loss occurs in Docling extraction, chunking, code detection, refinement, or JSONL export.
- [ ] Add targeted tests for the failure class.
- [ ] Implement the smallest general fix.
- [ ] Re-run Ayeva and Chaubal samples.
- [ ] Re-run Fluent Python as a non-regression control.

Acceptance signals:

- [ ] Ayeva passes CODE gate.
- [ ] Chaubal passes CODE gate.
- [ ] Fluent Python remains AUDIT_PASS.
- [ ] Code chunk count does not collapse artificially.
- [ ] `scripts/smoke_multiprofile.sh` remains green.

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

