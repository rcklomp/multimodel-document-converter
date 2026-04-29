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

Workstream B direction:

- Code-block fidelity must use a selective Docling-native enrichment lane, not more regex/reflow heuristics.
- Docling 2.86.0 exposes `CodeItem` and `do_code_enrichment`; targeted Ayeva evidence shows CodeFormulaV2 can restore multiline/indented code, but local CPU runtime is not viable for broad conversion.
- Do **not** trigger CodeFormulaV2 from `has_encoding_corruption` alone. Encoding corruption also appears in Combat Aircraft and would pull Workstream C into expensive code inference.
- Preferred architecture: cheap code-evidence pass first (`CodeItem` count/code density/sampled code candidates), then `needs_code_enrichment=True` with reason/counts, then remote-capable CodeFormulaV2 on a stronger local-network host or cloud endpoint.
- Client-side MLX/transformers acceleration is diagnostic/fallback only, not the main production strategy.
- Next-phase refactor: move duplicated PDF extraction policy out of `batch_processor.py` / `processor.py` into a shared `PdfConversionPlan` and Docling adapter before broadening conversion work.
- The current fenced-flat detector work is provisional fallback evidence, not a completed Workstream B fix.

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

VLM Source Sanctity enforcement status: **validated-cloud / local-pending** (2026-04-29).

1. Visual-only prompt enforced for all VLM calls.
2. Every response validated for text-reading behavior.
3. Retry with corrective instructions on first failure.
4. Sanitizer salvages text-reading responses (77.4% success rate on PCWorld).
5. Hard fallback to neutral description when unsalvageable.
6. Structured `vision_validation_issues` emitted in production JSONL.
7. Machine-readable quality summary via `scripts/vlm_quality_summary.py`.

Cloud-tested with `qwen3-vl-plus`. PCWorld raw text-reading detections: 36.5% → 22.2%. Zero measured Combat-style hallucinations. Blind-set run: 87.5% final-valid by the current validator.

Known limitations:

- Local NuMarkdown/local Qwen comparison is pending because the local inference server is unavailable off-network.
- Blind-set evidence is a reproducible fixture: manifest tracked at `tests/fixtures/blind_set_manifest.json`; generated assets belong in ignored `output/` and must be regenerated locally before re-running the harness.
- `scripts/vlm_quality_summary.py` harness mode now reports `clean` and `sanitized` as separate classes. Production mode classifies images as `clean_or_sanitized` (combined) because raw and issues fields are not available there.

Next engineering focus: code-block fidelity (Workstream B) and Combat Aircraft text corruption (Workstream C).

Latest validation:

- Unit suite: `337 passed, 1 skipped` (2026-04-29).
- Multi-profile smoke: `scripts/smoke_multiprofile.sh` reports `GATE_PASS` + `UNIVERSAL_PASS` for all 10 rows (2026-04-29, `output/smoke_multiprofile_20260429_232651/`).

## Immediate Next Work

Follow `docs/PROGRESS_CHECKLIST.md`.

Recommended sequence:

1. Fix Workstream A evidence durability and metric labeling.
2. Run local VLM comparison when the local inference server is reachable.
3. Replace broad Workstream B code-enrichment triggers with a selective `needs_code_enrichment` decision lane and remote-capable inference plan.
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
