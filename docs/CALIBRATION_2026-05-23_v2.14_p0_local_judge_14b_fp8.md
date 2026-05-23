# v2.14 Phase 0 Calibration — Local Judge vs qwen-max

> Date: 2026-05-23
> Ground truth: `output/soak/v2.13_p1_omlx/work.jsonl` (qwen-max judgments from v2.13 P1 soak)
> Local judge: `RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic` @ `http://10.0.10.239:8000/v1`
> Sample size: 518 queries judged on both sides (0 parse/call failures excluded)

## Headline agreement (per axis)

| Axis | n | Exact match | ±1 | Binary (0 vs ≥1) | Binary (≤1 vs 2) |
|---|---:|---:|---:|---:|---:|
| relevance | 518 | **82.2%** | 100.0% | 96.7% | 85.5% |
| format | 518 | **90.7%** | 99.8% | 98.8% | 91.7% |
| faithfulness | 518 | **76.6%** | 99.4% | 93.6% | 82.4% |

## Disposition by exact-match %

| Threshold | Recommended use |
|---|---|
| ≥85% | Local judge trustworthy for exploration soaks (hyperparameter sweeps, prompt iteration) |
| 70-85% | Restrict to HyDE-only (weaker semantics still help retrieval) |
| <70% | Not usable; pick a stronger local model or stay on cloud judging |

**Per-axis verdicts:**

- **relevance**: 82.2% exact → ⚠ RESTRICTED — HyDE-only
- **format**: 90.7% exact → ✓ TRUSTWORTHY — use for exploration soaks
- **faithfulness**: 76.6% exact → ⚠ RESTRICTED — HyDE-only

## Confusion matrices (qwen-max → local)

### relevance

| qwen-max ↓ \ local → | 0 | 1 | 2 |
|---|---:|---:|---:|
| **0** | 57 | 8 | 0 |
| **1** | 9 | 55 | 69 |
| **2** | 0 | 6 | 314 |

### format

| qwen-max ↓ \ local → | 0 | 1 | 2 |
|---|---:|---:|---:|
| **0** | 1 | 4 | 1 |
| **1** | 1 | 29 | 32 |
| **2** | 0 | 10 | 440 |

### faithfulness

| qwen-max ↓ \ local → | 0 | 1 | 2 |
|---|---:|---:|---:|
| **0** | 70 | 21 | 3 |
| **1** | 9 | 65 | 81 |
| **2** | 0 | 7 | 262 |

## Notes

- 0 queries had a parse or call failure on the local side and are excluded from the agreement numbers.
- Identical JUDGE prompt structure used on both sides (`JUDGE_SYSTEM` + `JUDGE_USER_TEMPLATE` from `scripts/synthetic_soak.py`).
- Same retrieved chunks, same gold, same query texts — only the judge model differs.
- Cache: `output/soak/v2.13_p1_omlx/calibration_local_judgments_14b_fp8.json` (rerun with the same `--results-cache` to resume).
