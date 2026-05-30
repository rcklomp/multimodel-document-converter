# v2.14 Phase 0 Calibration — Local Judge vs qwen-max

> **⚠ SUPERSEDED 2026-05-23.** The GX10 endpoint was swapped from
> `Qwen2.5-14B-Instruct` to `Qwen/Qwen3.6-27B-FP8` with native MTP=3
> on 2026-05-23. The per-axis verdicts in this report apply ONLY to
> the 14B model and are no longer operative for Phase 4 gating.
>
> **Operative replacement (ACTIVE):**
> [`docs/CALIBRATION_2026-05-23_v2.14_p0_local_judge_qwen36_27b_mtp.md`](CALIBRATION_2026-05-23_v2.14_p0_local_judge_qwen36_27b_mtp.md)
> — same 518-query fixture, same JUDGE prompt, only the judge model
> differs. **All three axes RESTRICTED on the 27B**: rel 82.0% /
> format **70.7%** / faith 78.8%. Bias direction flipped (was lenient
> on rel/faith for the 14B, now strict on format for the 27B). The
> 14B's TRUSTWORTHY format verdict (90.2% below) does NOT carry over.
> See PLAN_V2.14.md "Phase outcomes" Phase 0 row for the updated
> Phase 4 PERMITTED list.
>
> First-attempt note (historical): the 2026-05-23 re-cal initially got
> 510/518 parse failures because the 27B is served with
> `--reasoning-parser qwen3` and defaulted to thinking mode (output
> routed into `message.reasoning`; `message.content` came back null).
> Commit `0c5e818` added `chat_template_kwargs={"enable_thinking": false}`
> to the harness payload — second run completed with 0/518 failures.
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
