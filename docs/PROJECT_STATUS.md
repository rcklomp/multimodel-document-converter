# Project Status

Last updated: 2026-04-29

Purpose: fast orientation for a new coding session. Read this before deeper project docs.

## Current Objective

The project is in a quality-stabilization phase before broad reconversion and Qdrant re-ingestion.

Immediate goal: make conversion quality measurable and reproducible across document categories, then fix the highest-risk quality issues without overfitting to one document or one VLM.

## Active Baseline

The current quality reference point is:

- `docs/QUALITY_SNAPSHOT_2026-04-29.md`

Use that snapshot as the before-state for future comparisons.

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

### Non-Magazine PDFs

Existing non-magazine outputs: 14 of 17 pass `scripts/qa_conversion_audit.py`.

Known failures:

- `A_comprehensive_review_on_hybrid_electri`: one control-character text artifact
- `Ayeva_Python_Patterns`: code indentation fidelity and two infix artifacts
- `Chaubal_PyTorch_Projects`: code indentation fidelity

Known caveat:

- Some documents pass audit while routing through questionable profiles. Classifier correctness remains a separate workstream.

### Magazines

`Combat_Aircraft_full_promptfix_v2`:

- full 100-page conversion completed
- image side improved substantially
- Combat-style firearm/bolt/exploded-view hallucinations reduced to zero in the measured patterns
- still fails text audit because of encoding/high-corruption artifacts

`PCWorld_July_2025`:

- full existing output passes audit but uses placeholder image descriptions
- use it as an asset source, not as proof of VLM quality

`PCWorld_promptfix_pages1_20`:

- partial production smoke passes audit with VLM descriptions
- some residual text leaks were found before the latest sanitizer additions

## Active Engineering Direction

The VLM fix must be model-agnostic:

1. Prompt for visual-only descriptions.
2. Validate every VLM response for text-reading behavior.
3. Retry once with corrective instructions.
4. Sanitize salvageable visual descriptions.
5. Fall back to a neutral visual description when unsafe.

The target is not a perfect prompt. The guarantee must live outside the model.

## Immediate Next Work

Follow `docs/PROGRESS_CHECKLIST.md`.

Recommended sequence:

1. Stabilize and test the model-agnostic VLM enforcement layer.
2. Build a repeatable prompt-harness regression set.
3. Fix code-block fidelity for failing code-heavy books.
4. Fix Combat Aircraft text corruption.
5. Address classifier correctness.
6. Re-run broad conversion and update the quality snapshot.

## Must-Respect Constraints

- Python 3.10 only.
- Batch size must stay at or below 10 pages.
- Do not use `--profile-override` for acceptance runs.
- Do not add filename-specific or document-specific quality rules.
- OCR handles text; VLMs describe visuals only.
- BBoxes must remain normalized integer `[0,1000]`.
- Acceptance requires `GATE_PASS` plus `UNIVERSAL_PASS` across the smoke matrix.

