# v2.14 Phase 0 Calibration — Local Judge vs qwen-max

> Date: 2026-05-23
> Ground truth: `output/soak/v2.13_p1_omlx/work.jsonl` (qwen-max judgments from v2.13 P1 soak)
> Local judge: `meta-llama/llama-3.1-70b-instruct` @ `https://openrouter.ai/api/v1`
> Sample size: 518 queries judged on both sides (0 parse/call failures excluded)

## Headline agreement (per axis)

| Axis | n | Exact match | ±1 | Binary (0 vs ≥1) | Binary (≤1 vs 2) |
|---|---:|---:|---:|---:|---:|
| relevance | 518 | **79.9%** | 99.8% | 95.6% | 84.2% |
| format | 518 | **87.5%** | 99.2% | 98.8% | 87.8% |
| faithfulness | 518 | **75.1%** | 99.2% | 94.4% | 79.9% |

## Disposition by exact-match %

| Threshold | Recommended use |
|---|---|
| ≥85% | Local judge trustworthy for exploration soaks (hyperparameter sweeps, prompt iteration) |
| 70-85% | Restrict to HyDE-only (weaker semantics still help retrieval) |
| <70% | Not usable; pick a stronger local model or stay on cloud judging |

**Per-axis verdicts:**

- **relevance**: 79.9% exact → ⚠ RESTRICTED — HyDE-only
- **format**: 87.5% exact → ✓ TRUSTWORTHY — use for exploration soaks
- **faithfulness**: 75.1% exact → ⚠ RESTRICTED — HyDE-only

## Confusion matrices (qwen-max → local)

### relevance

| qwen-max ↓ \ local → | 0 | 1 | 2 |
|---|---:|---:|---:|
| **0** | 43 | 22 | 0 |
| **1** | 0 | 54 | 79 |
| **2** | 1 | 2 | 317 |

### format

| qwen-max ↓ \ local → | 0 | 1 | 2 |
|---|---:|---:|---:|
| **0** | 2 | 0 | 4 |
| **1** | 2 | 1 | 59 |
| **2** | 0 | 0 | 450 |

### faithfulness

| qwen-max ↓ \ local → | 0 | 1 | 2 |
|---|---:|---:|---:|
| **0** | 85 | 6 | 3 |
| **1** | 19 | 37 | 99 |
| **2** | 1 | 1 | 267 |

## Notes

- 0 queries had a parse or call failure on the local side and are excluded from the agreement numbers.
- Identical JUDGE prompt structure used on both sides (`JUDGE_SYSTEM` + `JUDGE_USER_TEMPLATE` from `scripts/synthetic_soak.py`).
- Same retrieved chunks, same gold, same query texts — only the judge model differs.
- Cache: `output/soak/v2.13_p1_omlx/calibration_openrouter_llama_31_70b.json` (rerun with the same `--results-cache` to resume).
