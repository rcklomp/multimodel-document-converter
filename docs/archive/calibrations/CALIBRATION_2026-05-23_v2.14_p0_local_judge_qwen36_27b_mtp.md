# v2.14 Phase 0 Calibration — Local Judge vs qwen-max

> Date: 2026-05-23
> Ground truth: `output/soak/v2.13_p1_omlx/work.jsonl` (qwen-max judgments from v2.13 P1 soak)
> Local judge: `Qwen/Qwen3.6-27B-FP8` @ `http://10.0.10.239:8000/v1`
> Sample size: 518 queries judged on both sides (0 parse/call failures excluded)

## Headline agreement (per axis)

| Axis | n | Exact match | ±1 | Binary (0 vs ≥1) | Binary (≤1 vs 2) |
|---|---:|---:|---:|---:|---:|
| relevance | 518 | **82.0%** | 99.8% | 93.8% | 88.0% |
| format | 518 | **70.7%** | 98.6% | 96.3% | 73.0% |
| faithfulness | 518 | **78.8%** | 98.6% | 91.9% | 85.5% |

## Disposition by exact-match %

| Threshold | Recommended use |
|---|---|
| ≥85% | Local judge trustworthy for exploration soaks (hyperparameter sweeps, prompt iteration) |
| 70-85% | Restrict to HyDE-only (weaker semantics still help retrieval) |
| <70% | Not usable; pick a stronger local model or stay on cloud judging |

**Per-axis verdicts:**

- **relevance**: 82.0% exact → ⚠ RESTRICTED — HyDE-only
- **format**: 70.7% exact → ⚠ RESTRICTED — HyDE-only
- **faithfulness**: 78.8% exact → ⚠ RESTRICTED — HyDE-only

## Confusion matrices (qwen-max → local)

### relevance

| qwen-max ↓ \ local → | 0 | 1 | 2 |
|---|---:|---:|---:|
| **0** | 65 | 0 | 0 |
| **1** | 31 | 58 | 44 |
| **2** | 1 | 17 | 302 |

### format

| qwen-max ↓ \ local → | 0 | 1 | 2 |
|---|---:|---:|---:|
| **0** | 5 | 1 | 0 |
| **1** | 11 | 50 | 1 |
| **2** | 7 | 132 | 311 |

### faithfulness

| qwen-max ↓ \ local → | 0 | 1 | 2 |
|---|---:|---:|---:|
| **0** | 85 | 7 | 2 |
| **1** | 28 | 70 | 57 |
| **2** | 5 | 11 | 253 |

## Notes

- 0 queries had a parse or call failure on the local side and are excluded from the agreement numbers.
- Identical JUDGE prompt structure used on both sides (`JUDGE_SYSTEM` + `JUDGE_USER_TEMPLATE` from `scripts/synthetic_soak.py`).
- Same retrieved chunks, same gold, same query texts — only the judge model differs.
- Cache: `output/soak/v2.13_p1_omlx/calibration_local_judgments_qwen36_27b_mtp.json` (rerun with the same `--results-cache` to resume).
