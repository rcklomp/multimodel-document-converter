# v2.14 Phase 0 Calibration — Local Judge vs qwen-max

> Date: 2026-05-23
> Ground truth: `output/soak/v2.13_p1_omlx/work.jsonl` (qwen-max judgments from v2.13 P1 soak)
> Local judge: `qwen/qwen3-32b` @ `https://openrouter.ai/api/v1`
> Sample size: 308 queries judged on both sides (210 parse/call failures excluded)

## Headline agreement (per axis)

| Axis | n | Exact match | ±1 | Binary (0 vs ≥1) | Binary (≤1 vs 2) |
|---|---:|---:|---:|---:|---:|
| relevance | 308 | **78.9%** | 99.4% | 94.5% | 83.8% |
| format | 308 | **84.1%** | 99.4% | 98.7% | 84.7% |
| faithfulness | 308 | **77.3%** | 99.0% | 92.9% | 83.4% |

## Disposition by exact-match %

| Threshold | Recommended use |
|---|---|
| ≥85% | Local judge trustworthy for exploration soaks (hyperparameter sweeps, prompt iteration) |
| 70-85% | Restrict to HyDE-only (weaker semantics still help retrieval) |
| <70% | Not usable; pick a stronger local model or stay on cloud judging |

**Per-axis verdicts:**

- **relevance**: 78.9% exact → ⚠ RESTRICTED — HyDE-only
- **format**: 84.1% exact → ⚠ RESTRICTED — HyDE-only
- **faithfulness**: 77.3% exact → ⚠ RESTRICTED — HyDE-only

## Confusion matrices (qwen-max → local)

### relevance

| qwen-max ↓ \ local → | 0 | 1 | 2 |
|---|---:|---:|---:|
| **0** | 26 | 12 | 1 |
| **1** | 3 | 24 | 48 |
| **2** | 1 | 0 | 193 |

### format

| qwen-max ↓ \ local → | 0 | 1 | 2 |
|---|---:|---:|---:|
| **0** | 1 | 1 | 0 |
| **1** | 1 | 26 | 8 |
| **2** | 2 | 37 | 232 |

### faithfulness

| qwen-max ↓ \ local → | 0 | 1 | 2 |
|---|---:|---:|---:|
| **0** | 32 | 17 | 2 |
| **1** | 2 | 44 | 43 |
| **2** | 1 | 5 | 162 |

## Notes

- 210 queries had a parse or call failure on the local side and are excluded from the agreement numbers.
- Identical JUDGE prompt structure used on both sides (`JUDGE_SYSTEM` + `JUDGE_USER_TEMPLATE` from `scripts/synthetic_soak.py`).
- Same retrieved chunks, same gold, same query texts — only the judge model differs.
- Cache: `output/soak/v2.13_p1_omlx/calibration_openrouter_qwen3_32b.json` (rerun with the same `--results-cache` to resume).
