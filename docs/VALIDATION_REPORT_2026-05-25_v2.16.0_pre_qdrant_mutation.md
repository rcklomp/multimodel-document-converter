# v2.16 Personal Validation Report — v2.16.0_pre_qdrant_mutation

> Generated: 2026-05-25T20:29:49
> Fixture mechanic: per-class `target_pass_rate` is the gate.
> Per-query PASS rule (ALL must hold): (a) gold doc in top-5,
> (b) format_constraint matches top-1 modality / ast.parse,
> (c) `expected_anchor_regexes` matches top-1 content.

## Summary

| Class | personal_importance | target | pass_rate | result |
|---|---|---:|---:|---|
| CarOK_voorraadtelling | HIGH | 85% | 0.0% (0/10) | FAIL |
| Fluent_Python | HIGH | 85% | 0.0% (0/10) | FAIL |

## CarOK_voorraadtelling

- personal_importance: **HIGH** ; target: 85% ; got: **0.0%** (0/10) — FAIL

| Query | doc top-5 | format | regex | gold-match | top-1 chunk | overall |
|---|---|---|---|---|---|---|
| Q01_acdelco_brake_pads_part_number | ✓ | ✗ | ✗ | — | `134b24_010_text_259c3435` | ✗ |
| Q02_febi_signal_switch_vw_passat | ✓ | ✗ | ✗ | — | `24_012_table_dac93a5a_o2` | ✗ |
| Q03_opel_interieurfilter_count | ✓ | ✗ | ✗ | — | `134b24_005_text_719fc7fe` | ✗ |
| Q04_castrol_motorolie_price | ✓ | ✗ | ✗ | — | `134b24_001_text_e3a6ae32` | ✗ |
| Q05_peugeot_205_parts | ✓ | ✗ | ✗ | — | `134b24_010_text_259c3435` | ✗ |
| Q06_gm_part_13356945 | ✓ | ✗ | ✗ | — | `134b24_004_text_9bdca6fe` | ✗ |
| Q07_renault_5_brake_pads | ✓ | ✗ | ✓ | — | `134b24_001_text_4d99acfb` | ✗ |
| Q08_audi_parts | ✓ | ✗ | ✓ | — | `34b24_012_table_dac93a5a` | ✗ |
| Q09_espace_clio_parts | ✓ | ✗ | ✓ | — | `134b24_001_text_4d99acfb` | ✗ |
| Q10_gm_part_24420728 | ✓ | ✗ | ✗ | — | `134b24_004_text_9bdca6fe` | ✗ |

### Failed query detail

**Q01_acdelco_brake_pads_part_number** — `What's the part number for the ACDelco brake pads set for Peugeot 205?`

- top-1 doc: `CarOK_voorraadtelling` modality=`text`
- top-5 docs: ['CarOK_voorraadtelling', 'CarOK_voorraadtelling', 'Greenhouse_Design', 'CarOK_voorraadtelling', 'Greenhouse_Design']

**Q02_febi_signal_switch_vw_passat** — `Which brand makes the directional signal switch for VW Passat in my stock?`

- top-1 doc: `CarOK_voorraadtelling` modality=`text`
- top-5 docs: ['CarOK_voorraadtelling', 'CarOK_voorraadtelling', 'CarOK_voorraadtelling', 'CarOK_voorraadtelling', 'CarOK_voorraadtelling']

**Q03_opel_interieurfilter_count** — `How many Opel Interieurfilter parts are in stock (article number 1808012)?`

- top-1 doc: `CarOK_voorraadtelling` modality=`text`
- top-5 docs: ['CarOK_voorraadtelling', 'CarOK_voorraadtelling', 'CarOK_voorraadtelling', 'CarOK_voorraadtelling', 'CarOK_voorraadtelling']

**Q04_castrol_motorolie_price** — `What's the price for the Castrol Edge FST 0W-30 motor oil (1 liter)?`

- top-1 doc: `CarOK_voorraadtelling` modality=`text`
- top-5 docs: ['CarOK_voorraadtelling', 'CarOK_voorraadtelling', 'CarOK_voorraadtelling', 'CarOK_voorraadtelling', 'CarOK_voorraadtelling']

**Q05_peugeot_205_parts** — `Which parts do I have for a Peugeot 205?`

- top-1 doc: `CarOK_voorraadtelling` modality=`text`
- top-5 docs: ['CarOK_voorraadtelling', 'CarOK_voorraadtelling', 'CarOK_voorraadtelling', 'Python_Distilled', 'CarOK_voorraadtelling']

**Q06_gm_part_13356945** — `What's General Motors part number 13356945?`

- top-1 doc: `CarOK_voorraadtelling` modality=`text`
- top-5 docs: ['CarOK_voorraadtelling', 'CarOK_voorraadtelling', 'CarOK_voorraadtelling', 'CarOK_voorraadtelling', 'CarOK_voorraadtelling']

**Q07_renault_5_brake_pads** — `Which brand brake pads do I have for the Renault 5 Super?`

- top-1 doc: `CarOK_voorraadtelling` modality=`text`
- top-5 docs: ['CarOK_voorraadtelling', 'CarOK_voorraadtelling', 'Greenhouse_Design', 'Greenhouse_Design', 'CarOK_voorraadtelling']
- matched regex: `Renault\s*5\s*Super`

**Q08_audi_parts** — `Welke onderdelen heb ik voor Audi?`

- top-1 doc: `CarOK_voorraadtelling` modality=`text`
- top-5 docs: ['CarOK_voorraadtelling', 'KI_En_ChatGPT_Praktische_Gids', 'CarOK_voorraadtelling', 'KI_En_ChatGPT_Praktische_Gids', 'CarOK_voorraadtelling']
- matched regex: `Audi`

**Q09_espace_clio_parts** — `Heb ik onderdelen voor de Renault Espace of Clio?`

- top-1 doc: `CarOK_voorraadtelling` modality=`text`
- top-5 docs: ['CarOK_voorraadtelling', 'CarOK_voorraadtelling', 'CarOK_voorraadtelling', 'CarOK_voorraadtelling', 'CarOK_voorraadtelling']
- matched regex: `Espace`

**Q10_gm_part_24420728** — `What's General Motors part number 24420728?`

- top-1 doc: `CarOK_voorraadtelling` modality=`text`
- top-5 docs: ['CarOK_voorraadtelling', 'CarOK_voorraadtelling', 'CarOK_voorraadtelling', 'CarOK_voorraadtelling', 'CarOK_voorraadtelling']

## Fluent_Python

- personal_importance: **HIGH** ; target: 85% ; got: **0.0%** (0/10) — FAIL

| Query | doc top-5 | format | regex | gold-match | top-1 chunk | overall |
|---|---|---|---|---|---|---|
| Q01_lru_cache_memoization | ✓ | ✗ | ✓ | — | `6164a3_226_text_e281447e` | ✗ |
| Q02_sentence_iter_class | ✓ | ✗ | ✓ | — | `6164a3_443_text_09588ac1` | ✗ |
| Q03_asyncio_as_completed | ✓ | ✗ | ✓ | — | `6164a3_604_text_e02cc374` | ✗ |
| Q04_sha256_concurrent_futures | ✓ | ✗ | ✓ | — | `6164a3_728_text_ace94d5a` | ✗ |
| Q05_repr_dunder | ✓ | ✗ | ✓ | — | `9a39ce_307_text_1d2a8b9f` | ✗ |
| Q06_fibonacci_clockdeco | ✓ | ✗ | ✓ | — | `6164a3_227_text_c47b9850` | ✗ |
| Q07_generator_yield | ✓ | ✗ | ✓ | — | `6164a3_443_text_d6d4e760` | ✗ |
| Q08_oscon_schedule_test | ✓ | ✗ | ✓ | — | `6164a3_738_text_481db4cb` | ✗ |
| Q09_taxi_process_simulation | ✓ | ✗ | ✗ | — | `6164a3_574_text_a3c0b940` | ✗ |
| Q10_regex_word_tokenize | ✓ | ✗ | ✗ | — | `f0bbb4_411_text_4be80929` | ✗ |

### Failed query detail

**Q01_lru_cache_memoization** — `How do I memoize a recursive function using functools.lru_cache?`

- top-1 doc: `Fluent_Python` modality=`text`
- top-5 docs: ['Fluent_Python', 'Ayeva_Python_Patterns', 'Ayeva_Python_Patterns', 'Ayeva_Python_Patterns', 'Ayeva_Python_Patterns']
- matched regex: `functools\.lru_cache`

**Q02_sentence_iter_class** — `Show me a Sentence class that implements __iter__ to yield words`

- top-1 doc: `Fluent_Python` modality=`text`
- top-5 docs: ['Fluent_Python', 'Fluent_Python', 'Fluent_Python', 'Fluent_Python', 'Fluent_Python']
- matched regex: `class\s+Sentence`

**Q03_asyncio_as_completed** — `How do I use asyncio.as_completed to handle concurrent download tasks?`

- top-1 doc: `Fluent_Python` modality=`text`
- top-5 docs: ['Fluent_Python', 'Fluent_Python', 'Fluent_Python', 'Fluent_Python', 'Fluent_Python']
- matched regex: `asyncio\.(as_completed|get_event_loop)`

**Q04_sha256_concurrent_futures** — `Show an example of SHA-256 hashing with concurrent.futures`

- top-1 doc: `Fluent_Python` modality=`text`
- top-5 docs: ['Fluent_Python', 'Fluent_Python', 'Fluent_Python', 'Fluent_Python', 'Fluent_Python']
- matched regex: `from\s+concurrent\s+import\s+futures`

**Q05_repr_dunder** — `How do I implement a custom __repr__ for a Python class?`

- top-1 doc: `Python_Distilled` modality=`text`
- top-5 docs: ['Python_Distilled', 'Fluent_Python', 'Python_Distilled', 'Python_Distilled', 'Fluent_Python']
- matched regex: `def\s+__repr__`

**Q06_fibonacci_clockdeco** — `Show me a faster fibonacci using lru_cache with the clockdeco timer decorator`

- top-1 doc: `Fluent_Python` modality=`text`
- top-5 docs: ['Fluent_Python', 'Fluent_Python', 'Fluent_Python', 'Fluent_Python', 'Fluent_Python']
- matched regex: `def\s+fibonacci`

**Q07_generator_yield** — `How do I write a generator function that yields items lazily?`

- top-1 doc: `Fluent_Python` modality=`text`
- top-5 docs: ['Fluent_Python', 'Fluent_Python', 'Python_Distilled', 'Fluent_Python', 'Fluent_Python']
- matched regex: `yield\s+`

**Q08_oscon_schedule_test** — `Show me a unit test for the OSCON conference schedule database`

- top-1 doc: `Fluent_Python` modality=`text`
- top-5 docs: ['Fluent_Python', 'Fluent_Python', 'Fluent_Python', 'Fluent_Python', 'Fluent_Python']
- matched regex: `schedule\.CONFERENCE`

**Q09_taxi_process_simulation** — `Show me the taxi fleet simulation example with yield Event`

- top-1 doc: `Fluent_Python` modality=`text`
- top-5 docs: ['Fluent_Python', 'Fluent_Python', 'Fluent_Python', 'Fluent_Python', 'Fluent_Python']

**Q10_regex_word_tokenize** — `Compile a regex that tokenizes words using re.compile`

- top-1 doc: `Python_Cookbook` modality=`text`
- top-5 docs: ['Python_Cookbook', 'Python_Cookbook', 'Python_Cookbook', 'Python_Cookbook', 'Fluent_Python']
