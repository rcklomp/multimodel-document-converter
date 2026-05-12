# Quality Snapshot 2026-05-10 — v2.9 Phase 5 Attempt

> **Status:** Phase 5a broad reconversion attempted for
> `v2.9.0-rc1`; blocked at 30/34 fresh canonical outputs. Phase 5b
> cloud VLM enrichment, Qdrant `mmrag_v2_8` drop/recreate, RC AFTER
> snapshot, and `v2.9.0-rc1` tag are intentionally not run until
> 34/34 fresh conversions exist.

## 1. Runner and environment

- Runner: `conda run -n mmrag-v2 python scripts/convert_books_v29.py`.
- Direct conda Python reports `torch.backends.mps.is_available() == True`.
- Nested `bash -lc` / `zsh -lc` wrappers report MPS unavailable in this
  environment, causing Docling to fall back to a stalled CPU path.
- `scripts/convert_books.sh` is therefore disabled for Phase 5 unless
  `MMRAG_ALLOW_LEGACY_SHELL_RUNNER=1` is set.
- `scripts/convert_books_v29.py` now starts each conversion in its own
  process group and kills the full group on timeout.

## 2. Commands run

```bash
conda run -n mmrag-v2 python scripts/convert_books_v29.py
conda run -n mmrag-v2 python scripts/convert_books_v29.py --keep-going --append-log --timeout 3600
conda run -n mmrag-v2 python scripts/convert_books_v29.py --only Devlin_LLM_Agents,Raieli_AI_Agents,Adedeji_GenAI_Google_Cloud,Cronin_GenAI_Models,Hao_ML_Platform,Nagasubramanian_Agentic_AI,Sekar_MCP_Standard,Python_Cookbook,ArcGIS_Python_Cookbook,Fluent_Python,Python_Distilled,Ayeva_Python_Patterns,Chaubal_PyTorch_Projects,Earthship_Vol1,Firearms,Greenhouse_Design,ChatGPT_Praktijk_handboek,KI_En_ChatGPT_Praktische_Gids --keep-going --append-log --timeout 3600 --force
conda run -n mmrag-v2 python scripts/convert_books_v29.py --only ChatGPT_Praktijk_handboek,KI_En_ChatGPT_Praktische_Gids --keep-going --append-log --timeout 3600 --force
```

Canonical log evidence: `output/_convert_books.log`.

## 3. Fresh output state

Fresh canonical outputs: **30/34**.

Fresh successes:

| Doc | JSONL lines |
|---|---:|
| `HarryPotter_and_the_Sorcerers_Stone` | 689 |
| `Form_0013_invoice` | 22 |
| `Form_betwistingsformulier` | 9 |
| `CarOK_voorraadtelling` | 82 |
| `AIOS_LLM_Agent_Operating_System` | 170 |
| `A_comprehensive_review_on_hybrid_electri` | 211 |
| `Hybrid_electric_vehicles` | 124 |
| `IRJET_Modeling_of_Solar_PV` | 50 |
| `Recent_Trends_in_Transportation` | 25 |
| `Combat_Aircraft_August_2025` | 724 |
| `PCWorld_July_2025` | 521 |
| `ATZ_Elektronik_German` | 64 |
| `Kimothi_RAG_Guide` | 854 |
| `Integra_manual` | 275 |
| `Jungjun_AI_Agent` | 767 |
| `Devlin_LLM_Agents` | 970 |
| `Adedeji_GenAI_Google_Cloud` | 1096 |
| `Cronin_GenAI_Models` | 1566 |
| `Nagasubramanian_Agentic_AI` | 1240 |
| `Sekar_MCP_Standard` | 509 |
| `Python_Cookbook` | 491 |
| `ArcGIS_Python_Cookbook` | 1084 |
| `Fluent_Python` | 2077 |
| `Python_Distilled` | 1271 |
| `Ayeva_Python_Patterns` | 716 |
| `Chaubal_PyTorch_Projects` | 799 |
| `Earthship_Vol1` | 991 |
| `Firearms` | 2184 |
| `ChatGPT_Praktijk_handboek` | 299 |
| `KI_En_ChatGPT_Praktische_Gids` | 4519 |

Blocked / missing fresh output:

| Doc | Phase 5a result |
|---|---|
| `Bourne_RAG_2024` | Timeout after 3600s |
| `Raieli_AI_Agents` | Timeout after 3600s |
| `Hao_ML_Platform` | Timeout after 3600s |
| `Greenhouse_Design` | Exceeded timeout path and was manually terminated; no clean fresh JSONL |

## 4. Gate impact

- Phase 5a is not complete because the broad reconversion contract is
  34/34 canonical docs.
- Phase 5b enrichment must not run over the broad corpus yet; it would
  produce partial evidence and leave the blind-test Greenhouse document
  missing.
- Qdrant `mmrag_v2_8` must not be dropped/recreated yet because the
  v2.9 corpus is incomplete.
- The RC AFTER snapshot and `v2.9.0-rc1` tag remain blocked.

## 5. Verification

- `conda run -n mmrag-v2 python -m py_compile scripts/convert_books_v29.py`
  passes after timeout hardening.
- Earlier Phase 5 prep validation in this turn:
  - `pytest tests/test_pdf_conversion_plan.py -q -k 'adapter or recovery_page_coverage'`
    passed: 17 passed.
  - `pytest tests/test_pdf_conversion_plan.py tests/test_chunk_id_collision_v29.py tests/test_v29_image_enrichment_acceptance.py -q`
    passed: 62 passed, 2 skipped.

## 6. Next engineering action

1. Rerun the four blocked documents with the hardened runner and a
   longer timeout, e.g. `--timeout 10800 --force --keep-going`.
2. If any still timeout, inspect whether the blocker is refiner
   routing, OCR/shadow extraction, CodeFormulaV2, or Docling layout
   conversion, and fix the responsible lane without filename-specific
   production logic.
3. Only after 34/34 fresh JSONLs exist, run Phase 5b cloud
   `qwen3-vl-plus` enrichment, then Qdrant drop/recreate, then the RC
   AFTER snapshot.
