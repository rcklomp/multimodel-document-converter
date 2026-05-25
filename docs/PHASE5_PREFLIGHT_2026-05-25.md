# v2.16 Phase 5 — Dynamic Top-K Pre-Flight Verdict

> Generated: 2026-05-25
> Fixture set: 2 class file(s), 20 queries.
> drop_off_threshold=2.5, min_absolute_gap=0.05, min_top_n=1, baseline top_n_return=5.

## Verdict

**KILL permanently.** At least one gate leg failed.

## Gate evaluation

| Leg | Condition | Result | Detail |
|---|---|---|---|
| (a) | ≥20% of queries `would_truncate` | PASS | 5/20 = 25.0% |
| (b) | PASS-retention ≥ 0.97 | FAIL | static_pass=0, dyn_pass=0, retention=undefined (static=0) |
| (c) | No HIGH class drops >2pp | PASS | CarOK_voorraadtelling: static=0.0% dyn=0.0% Δ=+0.0pp; Fluent_Python: static=0.0% dyn=0.0% Δ=+0.0pp |

## Per-query truncation samples

| query_id | class | top-N scores | trunc_n | would_truncate | static_pass | dynamic_pass |
|---|---|---|---|---|---|---|
| Q01_acdelco_brake_pads_part_number | CarOK_voorraadtelling | [0.000, 0.000, 0.000, 0.000, 0.000] | 5 | · | · | · |
| Q02_febi_signal_switch_vw_passat | CarOK_voorraadtelling | [0.000, 0.000, 0.000, 0.000, 0.697] | 5 | · | · | · |
| Q03_opel_interieurfilter_count | CarOK_voorraadtelling | [0.872, 0.000, 0.835, 0.000, 0.000] | 1 | ✓ | · | · |
| Q04_castrol_motorolie_price | CarOK_voorraadtelling | [0.000, 0.957, 0.000, 0.000, 0.000] | 2 | ✓ | · | · |
| Q05_peugeot_205_parts | CarOK_voorraadtelling | [0.000, 0.000, 0.000, 0.000, 0.000] | 5 | · | · | · |
| Q06_gm_part_13356945 | CarOK_voorraadtelling | [0.788, 0.000, 0.000, 0.733, 0.000] | 1 | ✓ | · | · |
| Q07_renault_5_brake_pads | CarOK_voorraadtelling | [0.000, 0.000, 0.000, 0.000, 0.000] | 5 | · | · | · |
| Q08_audi_parts | CarOK_voorraadtelling | [0.000, 0.000, 0.000, 0.000, 0.000] | 5 | · | · | · |
| Q09_espace_clio_parts | CarOK_voorraadtelling | [0.000, 0.000, 0.000, 0.000, 0.000] | 5 | · | · | · |
| Q10_gm_part_24420728 | CarOK_voorraadtelling | [0.782, 0.000, 0.695, 0.000, 0.000] | 1 | ✓ | · | · |
| Q01_lru_cache_memoization | Fluent_Python | [0.954, 0.946, 0.905, 0.905, 0.902] | 5 | · | · | · |
| Q02_sentence_iter_class | Fluent_Python | [0.949, 0.949, 0.949, 0.947, 0.947] | 5 | · | · | · |
| Q03_asyncio_as_completed | Fluent_Python | [0.941, 0.937, 0.934, 0.931, 0.915] | 5 | · | · | · |
| Q04_sha256_concurrent_futures | Fluent_Python | [0.964, 0.881, 0.880, 0.873, 0.869] | 1 | ✓ | · | · |
| Q05_repr_dunder | Fluent_Python | [0.947, 0.942, 0.928, 0.924, 0.921] | 5 | · | · | · |
| Q06_fibonacci_clockdeco | Fluent_Python | [0.977, 0.939, 0.928, 0.915, 0.915] | 5 | · | · | · |
| Q07_generator_yield | Fluent_Python | [0.953, 0.953, 0.930, 0.902, 0.898] | 5 | · | · | · |
| Q08_oscon_schedule_test | Fluent_Python | [0.946, 0.942, 0.941, 0.929, 0.929] | 5 | · | · | · |
| Q09_taxi_process_simulation | Fluent_Python | [0.952, 0.944, 0.938, 0.936, 0.929] | 5 | · | · | · |
| Q10_regex_word_tokenize | Fluent_Python | [0.907, 0.906, 0.903, 0.903, 0.894] | 5 | · | · | · |

## Disposition

Per PLAN_V2.16.md §3 Phase 5, KILL is permanent — no opt-in middle ground. DECISIONS.md entry: "v2.16 Phase 5 KILL — pre-flight evidence shows dynamic top-k has no measurable upside on the corpus. Failed legs: (b) PASS-retention undefined or below 0.97."