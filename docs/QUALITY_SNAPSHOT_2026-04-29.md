# Quality Snapshot 2026-04-29

Purpose: reference point before further work on model-agnostic VLM enforcement, code-block fidelity, classifier quality, and remaining conversion issues.

This snapshot records observed quality from existing outputs under `output/` plus the current prompt-harness experiments. It is not a final acceptance report. Use it as a before/after comparison target.

## Environment

- Date: 2026-04-29
- Project version: v2.7.0 / schema `2.7.0`
- Python environment used for checks: `conda run -n mmrag-v2`
- Current local VLM config: OpenAI-compatible local NuMarkdown endpoint
  - model: `NuMarkdown-8B-Thinking-mlx-8bits`
  - base URL: `http://10.0.10.246:8000/v1`
- Cloud VLM comparison tested through DashScope/OpenAI-compatible endpoint:
  - model: `qwen3-vl-plus`
- Refiner endpoint remains separate from local VLM endpoint.

## Current Code Delta Under Test

The working tree includes prompt/validation changes that are not yet committed:

- `src/mmrag_v2/vision/vision_prompts.py`
  - stronger visual-only prompt
  - generic text-reading detector
  - corrective retry prompt
  - response sanitizer for salvageable visual descriptions
- `src/mmrag_v2/vision/vision_manager.py`
  - prompt-aware vision cache
  - editorial/literature crops avoid diagram prompt routing
  - retry/sanitize/fallback flow for text-reading VLM responses
- `scripts/eval_vlm_image_prompt.py`
  - image-prompt evaluation harness
- `tests/test_vlm_text_detection.py`
  - text-reading detector and sanitizer tests

Verification already run:

```bash
conda run -n mmrag-v2 python -m pytest tests/test_vlm_text_detection.py -q
conda run -n mmrag-v2 python -m py_compile src/mmrag_v2/vision/vision_prompts.py src/mmrag_v2/vision/vision_manager.py scripts/eval_vlm_image_prompt.py
```

Result: `32 passed`; compile check passed.

## Cross-Profile Smoke Baseline

Source: `output/codex_smoke_multiprofile/_summary.txt`

The latest smoke matrix passed all listed rows:

| Category | Document | Detected profile | Gate | Universal |
|---|---|---|---|---|
| academic_journal | AIOS LLM Agent Operating System | academic_whitepaper | GATE_PASS | UNIVERSAL_PASS |
| academic_journal | A_comprehensive_review_on_hybrid_electri | academic_whitepaper | GATE_PASS | UNIVERSAL_PASS |
| business_form | betwistingsformulier_aankoop_niet_ontvangen | academic_whitepaper | GATE_PASS | UNIVERSAL_PASS |
| data_spreadsheet | CarOK voorraadtelling 2021-04 | technical_manual | GATE_PASS | UNIVERSAL_PASS |
| digital_magazine | PCWorld_July_2025_USA | digital_magazine | GATE_PASS | UNIVERSAL_PASS |
| scanned | HarryPotter_and_the_Sorcerers_Stone | digital_magazine | GATE_PASS | UNIVERSAL_PASS |
| technical_manual | Firearms | technical_manual | GATE_PASS | UNIVERSAL_PASS |
| technical_manual | Greenhouse Design and Control by Pedro Ponce | technical_manual | GATE_PASS | UNIVERSAL_PASS |
| technical_manual | Python Distilled David M. Beazley 2022 | technical_manual | GATE_PASS | UNIVERSAL_PASS |
| technical_report | ATZ Elektronik German | academic_whitepaper | GATE_PASS | UNIVERSAL_PASS |

Notes:

- This is a smoke test, not full-document acceptance.
- It exposes classifier concerns: `business_form`, `scanned/HarryPotter`, and `data_spreadsheet` pass gates but route through debatable profiles.
- `Greenhouse Design and Control by Pedro Ponce.pdf` remains the technical-manual blind-test document.

## Full/Existing Non-Magazine Output Matrix

Command used:

```bash
conda run -n mmrag-v2 python scripts/qa_conversion_audit.py <non-magazine ingestion.jsonl files>
```

Summary: 14 of 17 existing non-magazine outputs pass the unified conversion audit.

| Output | Profile | Class | Content | Chunks | Audit | Main issue |
|---|---|---:|---:|---:|---|---|
| `AIOS_LLM_Agent_Operating_System` | academic_whitepaper | digital | mixed_prose | 145 | PASS | code indentation advisory only |
| `ATZ_Elektronik_German` | academic_whitepaper | digital | non_code | 59 | PASS | heading fragmentation high, advisory |
| `A_comprehensive_review_on_hybrid_electri` | academic_whitepaper | digital | non_code | 192 | FAIL | 1 control-character chunk |
| `Adedeji_GenAI_Google_Cloud` | technical_manual | digital | mixed_prose | 895 | PASS | code indentation advisory only |
| `Ayeva_Python_Patterns` | digital_magazine | digital | code_heavy | 710 | FAIL | code indentation fidelity; 2 infix artifacts |
| `Bourne_RAG_2024` | technical_manual | digital | mixed_prose | 884 | PASS | code indentation advisory only |
| `Chaubal_PyTorch_Projects` | technical_manual | digital | code_heavy | 723 | FAIL | code indentation fidelity |
| `Cronin_GenAI_Models` | technical_manual | digital | non_code | 1288 | PASS | heading fragmentation high, advisory |
| `Devlin_LLM_Agents` | digital_magazine | digital | mixed_prose | 699 | PASS | classifier/profile questionable; code advisory |
| `Earthship_Vol1` | technical_manual | scanned | non_code | 709 | PASS | scanned output strong |
| `Firearms` | technical_manual | scanned | non_code | 1691 | PASS | heading coverage warning, audit still pass |
| `Fluent_Python_full_codex` | academic_whitepaper | digital | code_heavy | 1633 | PASS | image descriptions are placeholders/advisory |
| `Fluent_Python_full_vlm_codex` | academic_whitepaper | digital | code_heavy | 1634 | PASS | 470 high-corruption warnings; audit still pass |
| `HarryPotter_and_the_Sorcerers_Stone` | digital_magazine | digital | non_code | 443 | PASS | classifier/profile questionable |
| `IRJET_Modeling_of_Solar_PV` | academic_whitepaper | digital | non_code | 40 | PASS | small output; no major issue |
| `Kimothi_RAG_Guide` | technical_manual | digital | mixed_prose | 689 | PASS | code indentation advisory only |
| `Recent_Trends_in_Transportation` | digital_magazine | digital | non_code | 20 | PASS | classifier/profile questionable |

Interpretation:

- Text extraction and structural schema are mostly stable outside magazines.
- Code-heavy technical books remain the main non-magazine weakness.
- Some existing outputs pass quality gates despite questionable profile classification.
- Many older digital outputs have placeholder image descriptions; image quality is therefore less proven than text quality outside scanned/manual outputs.

## Magazine Output Matrix

Command used:

```bash
conda run -n mmrag-v2 python scripts/qa_conversion_audit.py \
  output/Combat_Aircraft_full_promptfix_v2/ingestion.jsonl \
  output/PCWorld_July_2025/ingestion.jsonl \
  output/PCWorld_promptfix_pages1_20/ingestion.jsonl
```

| Output | Scope | Chunks | Text | Images | Tables | Audit | Main issue |
|---|---|---:|---:|---:|---:|---|---|
| `Combat_Aircraft_full_promptfix_v2` | full 100 pages | 537 | 327 | 197 | 13 | FAIL | text encoding/high-corruption artifacts |
| `PCWorld_July_2025` | full 108 pages, older no-VLM output | 345 | 219 | 126 | 0 | PASS | image descriptions are placeholders/advisory |
| `PCWorld_promptfix_pages1_20` | partial 20-page production smoke | 65 | 42 | 23 | 0 | PASS | residual VLM text leaks existed before latest sanitizer additions |

Combat Aircraft image-side improvement from the prompt fix:

| Metric | Previous behavior | Current `Combat_Aircraft_full_promptfix_v2` |
|---|---:|---:|
| firearm/bolt/exploded-view hallucination union | 28 | 0 |
| `bolt_action` hits | 12 | 0 |
| `exploded_view` / `exploded_diagram` hits | 26 | 0 |

Interpretation:

- The magazine VLM prompt fix materially reduced the specific visual hallucination pattern.
- Combat still fails because of text corruption, not image coverage.
- PCWorld full output is useful as an image-asset source but not as a VLM-quality reference because it has placeholder image descriptions.

## PCWorld VLM Prompt-Harness Matrix

Source image set: `output/PCWorld_July_2025/ingestion.jsonl` image assets, 126 images.

Harness:

```bash
conda run -n mmrag-v2 python scripts/eval_vlm_image_prompt.py ...
```

| Run | Output file | Images | Text-reading hits | Hard fallbacks | Errors | Avg latency | Max latency | Combat-style hallucinations |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NuMarkdown before corrective retry | `output/PCWorld_prompt_eval_current.jsonl` | 126 | 46 | 46 | 0 | 4.48s | 14.90s | 0 |
| NuMarkdown with retry flow | `output/PCWorld_prompt_eval_retryfix.jsonl` | 126 | 34 | 34 | 0 | 2.07s | 6.35s | 0 |
| Qwen3-VL-Plus cloud | `output/PCWorld_prompt_eval_qwen3_vl_plus.jsonl` | 126 | 34 | 6 | 0 | 4.44s | 15.15s | 0 |

Interpretation:

- Qwen3-VL-Plus is visually richer and needs fewer hard fallbacks.
- Qwen still reads visible text, chart values, UI labels, titles, and product/brand names.
- The model-agnostic enforcement layer remains necessary for all VLMs.
- Current target architecture remains: prompt -> validate -> corrective retry -> sanitize -> fallback.

## Issue Backlog Snapshot

| Area | Current state | Evidence | Priority | Acceptance signal |
|---|---|---|---|---|
| Model-agnostic VLM Source Sanctity | Improved, not complete | PCWorld harness still finds text-reading hits across NuMarkdown and Qwen | High | Reduced text-reading hits and fallbacks across PCWorld, Combat, and a blind non-magazine set |
| Combat Aircraft text corruption | Failing | `encoding_artifacts=48`, `high_corruption=79` | High | `AUDIT_PASS` on full Combat conversion without profile override |
| Code block fidelity | Failing in some code-heavy books | Ayeva and Chaubal fail CODE gate | High | Ayeva and Chaubal `AUDIT_PASS`; no regression in Fluent Python |
| Classifier correctness | Mixed | several pass-but-questionable profile assignments | Medium | ProfileClassifier routes books/forms/scans/spreadsheets plausibly without `--profile-override` |
| Placeholder image descriptions in older outputs | Present | many digital non-magazine outputs show no-VLM advisory | Medium | representative reruns have image `placeholder_ratio=0%` without text-reading violations |
| Control-character artifact | Isolated known failure | `A_comprehensive_review_on_hybrid_electri` has 1 control-char chunk | Low/Medium | full output `AUDIT_PASS`; no new ctrl-char chunks in smoke |
| Heading fragmentation advisories | Present but non-blocking | ATZ/Cronin/Devlin high fragmentation warnings | Low | no hard gate failures; retrieval review if needed |

## Suggested Comparison Protocol For Next Iterations

For each fix, record before/after against this document:

1. Run targeted unit tests.
2. Run the relevant full conversion or prompt harness.
3. Run `scripts/qa_conversion_audit.py`.
4. Run `scripts/qa_universal_invariants.py` on changed outputs.
5. Run `scripts/smoke_multiprofile.sh` before declaring the change broadly valid.
6. Update this snapshot or create a new dated snapshot with deltas.

Do not use `--profile-override` for acceptance runs. It remains a diagnostic-only tool.

