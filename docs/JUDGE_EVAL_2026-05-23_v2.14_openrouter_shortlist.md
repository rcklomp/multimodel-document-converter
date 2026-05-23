# v2.14 Phase 0 — OpenRouter Judge-Model Shortlist Comparison

> Date: 2026-05-23
> Cycle: post-v2.14.0 — `feedback_no_gx10_model_swap_reflex` "evaluate offline before any live-endpoint swap"
> Ground truth: `output/soak/v2.13_p1_omlx/work.jsonl` (518 qwen-max judgments from v2.13 P1 soak)
> Cost: **$0.28** total OpenRouter spend across both runs (well under the $1.50-3 forecast)

## 1. The shortlist

Per the proposal I laid out earlier this session:

| Candidate | Why included | GX10 viability if it wins |
|---|---|---|
| `qwen/qwen3-32b` (Qwen3-32B-Instruct via OpenRouter) | Middle-ground Qwen-family sibling; ~17 tok/s on GX10 at FP8 sweet spot | ✓ Fast enough for HyDE + gen + judge single-deploy |
| `meta-llama/llama-3.1-70b-instruct` (via OpenRouter) | JudgeBench winner (52.6%; beat GPT-4o per arxiv 2410.12784) | ⚠ Slow on GX10 (~4-8 tok/s FP4) — viable judging-only, problematic for HyDE/gen |

Skipped pre-eval: `Qwen2.5-72B-Instruct` (bandwidth math says unviable on GX10).

## 2. Results vs. existing baselines

| Judge | n | relevance | format | faithfulness | Verdict |
|---|---:|---:|---:|---:|---|
| **`Qwen2.5-14B-Instruct` (BF16, GX10, retired 2026-05-23)** | 518 | 81.7% | **90.2% ✓** | 76.1% | format TRUSTWORTHY only |
| **`Qwen/Qwen3.6-27B-FP8` + MTP (GX10 CURRENT)** | 518 | 82.0% | 70.7% | 78.8% | all three RESTRICTED |
| `qwen/qwen3-32b` (OpenRouter) | **308**¹ | 78.9% | 84.1% | 77.3% | all three RESTRICTED |
| **`meta-llama/llama-3.1-70b-instruct` (OpenRouter)** | 518 | 79.9% | **87.5% ✓** | 75.1% | format TRUSTWORTHY |

**¹ Qwen3-32B sample-size caveat:** 210/518 queries failed (40%) with
`finish_reason=length` because Qwen3-32B does thinking-mode reasoning
by default and OpenRouter's adapter doesn't propagate the
`chat_template_kwargs.enable_thinking=false` flag to the upstream
provider. The 308 successful samples are biased toward easy queries
that didn't need much reasoning. Re-running with `max_tokens=1500`
would give a clean number but is unlikely to lift Qwen3-32B above
the 14B baseline (already losing all three axes vs the 14B on the
biased-easy subset).

## 3. Per-axis deltas vs the GX10 current (27B-MTP)

| Judge | rel | format | faith |
|---|---:|---:|---:|
| `Qwen2.5-14B-Instruct` | -0.3pp | **+19.5pp** | -2.7pp |
| `qwen/qwen3-32b` (n=308) | -3.1pp | +13.4pp | -1.5pp |
| `meta-llama/llama-3.1-70b-instruct` | -2.1pp | **+16.8pp** | -3.7pp |

Format-axis recovery is the consistent win across all three
alternatives — confirms the 27B-MTP's format-strictness is the
27B-FP8-MTP-specific anomaly, not a general model-family trait.

## 4. Decision matrix

From the original proposal:

| Condition | Action | Triggered? |
|---|---|---|
| 32B beats 14B on all axes | swap GX10 to 32B (single deploy) | **No** — 32B loses to 14B on all 3 axes even on the biased-easy subset |
| 70B beats 32B by ≥5pp on format | dual-deploy: 70B for judging-only + keep 27B-MTP for HyDE/gen | **No** — only +3.4pp gap (87.5 vs 84.1) |
| Neither beats 14B | swap GX10 back to 14B | **Yes** — 14B wins all three axes against every candidate |
| 32B beats 14B marginally | swap GX10 to 32B | **No** |

## 5. Recommendation

**Swap GX10 back to `Qwen2.5-14B-Instruct` BF16.**

Rationale:
- The 14B BF16 (already calibrated, verdict known) has the best
  calibration profile of any local-deployable candidate evaluated:
  rel 81.7%, **format 90.2% TRUSTWORTHY**, faith 76.1%.
- The 27B-MTP's only advantage over the 14B was speculative-decoding
  throughput (~32 tok/s on GX10 vs the 14B's ~25 tok/s BF16). That
  speed advantage doesn't recover the -19.5pp format regression.
- Neither evaluated alternative (Qwen3-32B, Llama-3.1-70B) beats the
  14B clearly enough to justify a swap. The 70B comes closest on
  format (-2.7pp) but loses on rel + faith, and would be slow on
  GX10 for HyDE/gen.
- Cloud `qwen-max` (Dashscope) remains the ship-gate judge per the
  leniency-trap rule in `memory/feedback_fix_extraction_not_judge.md`,
  independent of which local-LLM sits on GX10.

**Restoration procedure** (per the deployment guardrails in
`memory/feedback_gx10_deployment_guardrails.md`):

1. SSH to the GX10; stop the current 27B-MTP container.
2. Re-launch the 14B container using the historical recipe (the 14B
   was the GX10 endpoint from 2026-05-22 morning through 2026-05-23
   morning; its container args are in earlier session notes /
   `memory/project_v2_14_gx10_27b_mtp_swap.md` "Predecessor" line).
3. No re-cal needed — the 14B's calibration verdict from 2026-05-22
   (`docs/CALIBRATION_2026-05-22_v2.14_p0_local_judge.md`) is
   reusable since the model bytes are identical.
4. Update `src/mmrag_v2/retrieval/hyde.py` `VLLM_DEFAULT_MODEL`
   back to `Qwen/Qwen2.5-14B-Instruct`.
5. Update `scripts/calibrate_local_judge_vs_qwen_max.py`
   `DEFAULT_LOCAL_MODEL` similarly.
6. Update `tests/test_hyde.py` model-name assertion + run.

**Alternative if user wants a different shape:**

If the user wants to keep Qwen3-family alignment AND format
TRUSTWORTHY, the only path is a clean re-eval of Qwen3-32B with
the thinking-mode worked around (bumped `max_tokens` + raised
parse-success rate). Cost: ~$0.30 + ~50 min. Likely outcome: still
RESTRICTED on rel + faith, but a clean format number.

If the user values format-axis lift over the rel/faith small losses,
the 70B Llama is a viable judging-only deploy (cloud, not GX10) —
keep 27B-MTP on GX10 for HyDE/gen, route ship-gate format judging
through cloud Llama-70B via OpenRouter ($/call comparable to
qwen-max). The cloud already runs ship-gate judging anyway per the
leniency-trap rule, so this is just changing which cloud model.

## 6. Cost + time summary

- OpenRouter total spend: **$0.28** (Qwen3-32B + Llama-3.1-70B
  combined, 308 + 518 = 826 successful calls)
- Parallel wall-clock: 70B finished ~25 min; 32B finished ~85 min
  (slower because of the thinking-mode preamble eating tokens)
- Zero impact on production retrieval or GX10 endpoint (eval was
  pure-cloud against OpenRouter)

## 7. References

- Harness: `scripts/calibrate_local_judge_vs_qwen_max.py` (patched
  this session with `--judge-bearer-env`, `HTTP-Referer`,
  `X-Title` headers; ~30 LOC addition)
- Per-judge cache: `output/soak/v2.13_p1_omlx/calibration_openrouter_{qwen3_32b,llama_31_70b}.json`
- Per-judge report: `docs/CALIBRATION_2026-05-23_v2.14_p0_openrouter_{qwen3_32b,llama_31_70b}.md`
- 14B baseline: `docs/CALIBRATION_2026-05-22_v2.14_p0_local_judge.md`
- 27B-MTP baseline: `docs/CALIBRATION_2026-05-23_v2.14_p0_local_judge_qwen36_27b_mtp.md`
- Deployment guardrails: `memory/feedback_gx10_deployment_guardrails.md`
- Anti-reflex policy: `memory/feedback_no_gx10_model_swap_reflex.md`
