# v2.14 Phase 0 Calibration — Local Judge vs qwen-max

> **⚠ SUPERSEDED 2026-05-23.** The GX10 endpoint was swapped from
> `Qwen2.5-14B-Instruct` to `Qwen/Qwen3.6-27B-FP8` with native MTP=3
> on 2026-05-23. The per-axis verdicts in this report apply ONLY to
> the 14B model and are no longer operative for Phase 4 gating.
>
> **Operative replacement (pending):**
> `docs/CALIBRATION_2026-05-23_v2.14_p0_local_judge_qwen36_27b_mtp.md`
> — run `scripts/calibrate_local_judge_vs_qwen_max.py` (defaults now
> point at the 27B endpoint) to produce it. Same fixture, same
> JUDGE prompt — only the judge model differs. Until the new report
> exists, Phase 4b/c/d/e remain blocked per `feedback_gx10_deployment_guardrails`.
>
> Body preserved below as historical evidence + harness validation.

> Date: 2026-05-22
> Ground truth: `output/soak/v2.13_p1_omlx/work.jsonl` (qwen-max judgments from v2.13 P1 soak)
> Local judge: `Qwen/Qwen2.5-14B-Instruct` @ `http://10.0.10.239:8000/v1` *(retired 2026-05-23)*
> Sample size: 518 queries judged on both sides (0 parse/call failures excluded)

## Headline agreement (per axis)

| Axis | n | Exact match | ±1 | Binary (0 vs ≥1) | Binary (≤1 vs 2) |
|---|---:|---:|---:|---:|---:|
| relevance | 518 | **81.7%** | 100.0% | 96.7% | 84.9% |
| format | 518 | **90.2%** | 99.8% | 98.8% | 91.1% |
| faithfulness | 518 | **76.1%** | 99.4% | 93.6% | 81.9% |

## Disposition by exact-match %

| Threshold | Recommended use |
|---|---|
| ≥85% | Local judge trustworthy for exploration soaks (hyperparameter sweeps, prompt iteration) |
| 70-85% | Restrict to HyDE-only (weaker semantics still help retrieval) |
| <70% | Not usable; pick a stronger local model or stay on cloud judging |

**Per-axis verdicts:**

- **relevance**: 81.7% exact → ⚠ RESTRICTED — HyDE-only
- **format**: 90.2% exact → ✓ TRUSTWORTHY — use for exploration soaks
- **faithfulness**: 76.1% exact → ⚠ RESTRICTED — HyDE-only

## Confusion matrices (qwen-max → local)

### relevance

| qwen-max ↓ \ local → | 0 | 1 | 2 |
|---|---:|---:|---:|
| **0** | 56 | 9 | 0 |
| **1** | 8 | 55 | 70 |
| **2** | 0 | 8 | 312 |

### format

| qwen-max ↓ \ local → | 0 | 1 | 2 |
|---|---:|---:|---:|
| **0** | 1 | 4 | 1 |
| **1** | 1 | 28 | 33 |
| **2** | 0 | 12 | 438 |

### faithfulness

| qwen-max ↓ \ local → | 0 | 1 | 2 |
|---|---:|---:|---:|
| **0** | 70 | 21 | 3 |
| **1** | 9 | 61 | 85 |
| **2** | 0 | 6 | 263 |

## Notes

- 0 queries had a parse or call failure on the local side and are excluded from the agreement numbers.
- Identical JUDGE prompt structure used on both sides (`JUDGE_SYSTEM` + `JUDGE_USER_TEMPLATE` from `scripts/synthetic_soak.py`).
- Same retrieved chunks, same gold, same query texts — only the judge model differs.
- Cache: `output/soak/v2.13_p1_omlx/calibration_local_judgments.json` (rerun with the same `--results-cache` to resume).
