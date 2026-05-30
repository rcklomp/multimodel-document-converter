# v2.16 Phase 0 — Corpus Expansion Inventory

> Generated: 2026-05-25
> Source PDFs: `data/raw/` (7 files); outputs under `output/<basename>/ingestion.jsonl`
> Classifier: `scripts/classify_corpus_v2_16_p0.py` (recalibrated thresholds; see §2 below).

## 1. Per-doc classification

| basename | profile | chunks | text | code | table | image | code_ratio | table_ratio | uniq_tbl | non_ascii | intent_ML | flags | probes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `Bevestigingsmiddelen` | academic_whitepaper | 4 | 4 | 0 | 0 | 0 | 0.000 | 0.000 | 0 | 0.0000 | 0.00 | general | — |
| `ATZ_Aerodynamik_Nutzfahrzeugen` | digital_magazine | 37 | 24 | 0 | 0 | 13 | 0.000 | 0.000 | 0 | 0.0115 | 0.96 | minority_language | — |
| `ATZ_ESF_Mercedes_2009` | academic_whitepaper | 57 | 41 | 0 | 0 | 16 | 0.000 | 0.000 | 0 | 0.0118 | 0.97 | minority_language | — |
| `Schwungradspeicher` | digital_magazine | 1191 | 905 | 6 | 52 | 228 | 0.007 | 0.044 | 37 | 0.0138 | 0.90 | minority_language | — |
| `Eliasz_Zephyr_RTOS` | digital_magazine | 1682 | 872 | 325 | 17 | 468 | 0.272 | 0.010 | 12 | 0.0004 | 0.00 | code_dense | — |
| `Grundlagen_Fahrzeug_Motorentechnik` | digital_magazine | 2649 | 2175 | 2 | 30 | 442 | 0.001 | 0.011 | 24 | 0.0099 | 0.70 | minority_language | — |
| `Digitale_Fotografie_Feb_2026` | scanned | 888 | 750 | 0 | 0 | 138 | 0.000 | 0.000 | 0 | 0.0157 | 0.80 | minority_language | A=ELIGIBLE_RE_EXTRACT |

## 2. Recalibrated thresholds (vs plan defaults)

Per PLAN_V2.16.md §3 Phase 0 step 2 (threshold pre-validation):
the plan's defaults (code 0.30, table 0.40) fail the canonical-34
sanity targets — Fluent_Python at 0.276 code-ratio misses 0.30 by
construction; CarOK at 0.136 table-ratio misses 0.40 by
construction. Recalibrated thresholds satisfy all 6 sanity
targets via step-3 rules alone (Probe A not needed for CarOK at
the new threshold).

| Class | Plan default | v2.16 recalibrated | Sanity verdict |
|---|---|---|---|
| code-dense | `code/(text+code) ≥ 0.30` | **`≥ 0.25`** | Fluent_Python (0.276) PASS; HarryPotter/CarOK still NOT |
| form-class | `table/total ≥ 0.40 AND uniq ≥ 3` | **`≥ 0.10 AND uniq ≥ 3`** | CarOK (0.136, uniq=10) PASS; Python_Distilled (0.039) still NOT |
| minority-language | unchanged | unchanged | ATZ_Elektronik PASS via intent (no umlauts); HarryPotter false-positive noted as intent-classifier limitation |

## 3. Probe coverage (per PLAN_V2.16.md §3 Phase 0 step 4)

- **Probe A** — form-class re-extract via `--force-table-vlm`
  fires only on docs with `profile ∈ {scanned, scanned_degraded}`
  AND `image_chunks > 0` AND `table_chunks == 0`. See per-doc
  flags above.
- **Probe B** — borderline minority-language (signal-only). Fires
  when intent classifier matches ≥1 chunk but total hit-rate <
  0.30 AND non-ASCII < 0.03. See per-doc flags above.
- **Probe C** — near-boundary classification (signal-only). Flags
  docs within 5pp of the code-dense / form-class / minority-
  language thresholds for user review.

## 4. Class composition feeding Phase 2/3/4 scoping

Per PLAN_V2.16.md §3 Phase 0 step 7:

- **Minority-language docs** in this expansion: classified per
  Phase 0 (German content count visible in §1 above).
  Phase 2 (omlx-deficit class-level test) verdict already
  recorded as multi-factor in
  `docs/DIAGNOSTIC_2026-05-25_v2.16_p2_omlx_deficit_root_cause.md`
  — replication would require a hot dashscope collection which
  was dropped in v2.14 P3. Phase 6 KILL is final.
- **Form-class docs** in this expansion: see §1. Phase 4
  generality measured via the programmatic dual gate
  (`suppression_count > 0` AND no same-page Jaccard ≥ 0.5) once
  the CarOK re-extract result lands.
- **Code-dense docs** in this expansion: see §1. Phase 3
  acceptance bar (≥85% Phase 1 validation on code-dense docs)
  is structurally blocked by the inert-on-current-corpus state
  of partial_code (see DECISIONS.md "v2.16 Phase 3 …").

## 5. User-acceptance notes

**Probe-flagged docs require explicit user acceptance** before
Phase 0 step 6 (Qdrant snapshot + dense append + BM25 rebuild)
runs. Acceptance is recorded by editing this report in place
with one of:

- `ACCEPTED: classification correct`
- `RECLASSIFY: override profile to <X>` (re-run ingest with
  `--profile-override <X>` before step 6.3)
- `RECALIBRATE: adjust threshold <X> from <Y> → <Z>` (re-run
  step 2 + step 3 with the new threshold).

Edit each row of §1 to add an `ACCEPTED` / `RECLASSIFY` /
`RECALIBRATE` column when reviewing.

## 6. Source PDFs

- `data/raw/<...>.pdf` → `output/Bevestigingsmiddelen/`
- `data/raw/<...>.pdf` → `output/ATZ_Aerodynamik_Nutzfahrzeugen/`
- `data/raw/<...>.pdf` → `output/ATZ_ESF_Mercedes_2009/`
- `data/raw/<...>.pdf` → `output/Schwungradspeicher/`
- `data/raw/<...>.pdf` → `output/Eliasz_Zephyr_RTOS/`
- `data/raw/<...>.pdf` → `output/Grundlagen_Fahrzeug_Motorentechnik/`
- `data/raw/<...>.pdf` → `output/Digitale_Fotografie_Feb_2026/`

## 7. Reproduce

```bash
# Re-ingest all 7:
bash scripts/_v2_16_p0_ingest_all.sh

# Re-classify:
python scripts/classify_corpus_v2_16_p0.py output/{Bevestigingsmiddelen,ATZ_Aerodynamik_Nutzfahrzeugen,ATZ_ESF_Mercedes_2009,Schwungradspeicher,Eliasz_Zephyr_RTOS,Grundlagen_Fahrzeug_Motorentechnik,Digitale_Fotografie_Feb_2026}/ingestion.jsonl

# Regenerate this report:
python scripts/_v2_16_p0_inventory_report.py \
    --classifier-json output/_v2_16_p0_logs/classify_new7.json \
    --output docs/CORPUS_EXPANSION_2026-05-25_v2.16_p0.md
```