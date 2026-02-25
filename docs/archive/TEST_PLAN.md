# Test Plan (Comprehensive, Current State)

## Purpose
Validate all major functions, CLI flags, and processing paths before adding more document formats. This plan is written to be executed by Codex in the Conda environment and is structured to produce inputs for a detailed compliance report.

## Environment
- Runner: `conda run -p ./env mmrag-v2`
- Set `TS=$(date +%Y%m%d%H%M%S)` and `OUT=/tmp/mmrag_full_tests/$TS`
- Endpoint (OpenAI-compatible): `http://192.168.10.11:1234/v1`
- Vision model: `llama-joycaption-beta-one-hf-llava-mmproj`
- Refiner model: `mistral-7b-instruct-v0.3-mixed-6-8-bit`
- Output base: `$OUT`

## Test Data (Existing)
Use these from `data/raw/`:
- Digital academic PDF: `Hybrid_electric_vehicles_and_their_challenges.pdf`
- Editorial/magazine PDF: `IRJET_Modeling_of_Solar_PV_system_under.pdf` or `PCWorld_July_2025_USA.pdf`
- Technical manual (scanned): `Firearms.pdf`
- Long scanned literature: `HarryPotter_and_the_Sorcerers_Stone.pdf`
- EPUB: `Falkner, Leonie - ChatGPT Praktijk-handboek.epub`
- HTML: `Train a Model Faster with torch.compile and Gradient Accumulation.html`
- DOCX: `DAF EMS hillab Tool description.docx`
- PPTX: `Keywords In ControlDesk.pptx`
- XLSX: `Eu06_simulation_specs.xlsx`

## Pre-Checks
1. Confirm env python:
   ```bash
   conda run -p ./env python -V
   ```
2. Create output root:
   ```bash
   mkdir -p $OUT
   ```
3. Version sanity (schema_version injection):
   ```bash
   conda run -p ./env python - <<'PY'
   from mmrag_v2.version import __schema_version__
   print("schema_version:", __schema_version__)
   PY
   ```

---

## Phase A — CLI Surface Coverage (Arguments)

Conventions:
- Use `conda run -p ./env mmrag-v2 ...` for all commands.
- Use `$OUT` as the output root for all commands.

### A1. process (baseline, digital PDF → OCR off by policy)
```bash
conda run -p ./env mmrag-v2 process \
  "data/raw/Hybrid_electric_vehicles_and_their_challenges.pdf" \
  --output-dir $OUT/process_baseline \
  --vision-provider none
```

### A2. process --batch-size
```bash
conda run -p ./env mmrag-v2 process \
  "data/raw/Hybrid_electric_vehicles_and_their_challenges.pdf" \
  --output-dir $OUT/process_batch \
  --batch-size 3 \
  --vision-provider none
```

### A3. batch --pattern
```bash
mkdir -p $OUT/batch_input
cp "data/raw/Hybrid_electric_vehicles_and_their_challenges.pdf" $OUT/batch_input/doc1.pdf
cp "data/raw/IRJET_Modeling_of_Solar_PV_system_under.pdf" $OUT/batch_input/doc2.pdf

conda run -p ./env mmrag-v2 batch $OUT/batch_input \
  --output-dir $OUT/batch_out \
  --vision-provider none
```

### A4. --sensitivity (min/max)
```bash
conda run -p ./env \
  mmrag-v2 process \
  "data/raw/IRJET_Modeling_of_Solar_PV_system_under.pdf" \
  --output-dir $OUT/sensitivity_low \
  --sensitivity 0.1 \
  --vision-provider none

conda run -p ./env \
  mmrag-v2 process \
  "data/raw/IRJET_Modeling_of_Solar_PV_system_under.pdf" \
  --output-dir $OUT/sensitivity_high \
  --sensitivity 1.0 \
  --vision-provider none
```

### A5. --strict-qa on/off
```bash
conda run -p ./env \
  mmrag-v2 process \
  "data/raw/Hybrid_electric_vehicles_and_their_challenges.pdf" \
  --output-dir $OUT/strict_qa_on \
  --strict-qa \
  --vision-provider none

conda run -p ./env \
  mmrag-v2 process \
  "data/raw/Hybrid_electric_vehicles_and_their_challenges.pdf" \
  --output-dir $OUT/strict_qa_off \
  --no-strict-qa \
  --vision-provider none
```

### A6. --semantic-overlap on/off
```bash
conda run -p ./env \
  mmrag-v2 process \
  "data/raw/Hybrid_electric_vehicles_and_their_challenges.pdf" \
  --output-dir $OUT/semantic_overlap_on \
  --semantic-overlap \
  --vision-provider none

conda run -p ./env \
  mmrag-v2 process \
  "data/raw/Hybrid_electric_vehicles_and_their_challenges.pdf" \
  --output-dir $OUT/semantic_overlap_off \
  --no-semantic-overlap \
  --vision-provider none
```

### A7. --vlm-context-depth (0 and 10)
```bash
conda run -p ./env \
  mmrag-v2 process \
  "data/raw/IRJET_Modeling_of_Solar_PV_system_under.pdf" \
  --output-dir $OUT/vlm_ctx_0 \
  --vlm-context-depth 0 \
  --vision-provider none

conda run -p ./env \
  mmrag-v2 process \
  "data/raw/IRJET_Modeling_of_Solar_PV_system_under.pdf" \
  --output-dir $OUT/vlm_ctx_10 \
  --vlm-context-depth 10 \
  --vision-provider none
```

### A8. --allow-fullpage-shadow on/off
```bash
conda run -p ./env \
  mmrag-v2 process \
  "data/raw/PCWorld_July_2025_USA.pdf" \
  --output-dir $OUT/fullpage_shadow_off \
  --no-fullpage-shadow \
  --vision-provider none

conda run -p ./env \
  mmrag-v2 process \
  "data/raw/PCWorld_July_2025_USA.pdf" \
  --output-dir $OUT/fullpage_shadow_on \
  --allow-fullpage-shadow \
  --vision-provider none
```

### A9. Cache flags
```bash
conda run -p ./env \
  mmrag-v2 process \
  "data/raw/Hybrid_electric_vehicles_and_their_challenges.pdf" \
  --output-dir $OUT/cache_on \
  --enable-cache \
  --vision-provider none

conda run -p ./env \
  mmrag-v2 process \
  "data/raw/Hybrid_electric_vehicles_and_their_challenges.pdf" \
  --output-dir $OUT/cache_off \
  --no-cache \
  --vision-provider none
```

### A10. OCR engine variants
> Note: Run OCR engine variants only on scanned/unknown PDFs. Digital PDFs now skip the cascade by policy.
```bash
conda run -p ./env mmrag-v2 process \
  "data/raw/HarryPotter_and_the_Sorcerers_Stone.pdf" \
  --output-dir $OUT/ocr_easyocr \
  --enable-ocr \
  --ocr-engine easyocr \
  --vision-provider none

conda run -p ./env mmrag-v2 process \
  "data/raw/HarryPotter_and_the_Sorcerers_Stone.pdf" \
  --output-dir $OUT/ocr_tesseract \
  --enable-ocr \
  --ocr-engine tesseract \
  --vision-provider none

conda run -p ./env mmrag-v2 process \
  "data/raw/HarryPotter_and_the_Sorcerers_Stone.pdf" \
  --output-dir $OUT/ocr_doctr \
  --enable-ocr \
  --ocr-engine doctr \
  --vision-provider none
```

### A11. OCR mode routing
```bash
conda run -p ./env mmrag-v2 process \
  "data/raw/HarryPotter_and_the_Sorcerers_Stone.pdf" \
  --output-dir $OUT/ocr_mode_auto \
  --ocr-mode auto \
  --enable-ocr \
  --vision-provider none

conda run -p ./env mmrag-v2 process \
  "data/raw/HarryPotter_and_the_Sorcerers_Stone.pdf" \
  --output-dir $OUT/ocr_mode_legacy \
  --ocr-mode legacy \
  --enable-ocr \
  --vision-provider none

conda run -p ./env mmrag-v2 process \
  "data/raw/HarryPotter_and_the_Sorcerers_Stone.pdf" \
  --output-dir $OUT/ocr_mode_layout \
  --ocr-mode layout-aware \
  --enable-ocr \
  --vision-provider none
```

### A12. OCR confidence threshold (low/high)
```bash
conda run -p ./env mmrag-v2 process \
  "data/raw/HarryPotter_and_the_Sorcerers_Stone.pdf" \
  --output-dir $OUT/ocr_conf_low \
  --ocr-mode layout-aware \
  --enable-ocr \
  --ocr-confidence-threshold 0.3 \
  --vision-provider none

conda run -p ./env mmrag-v2 process \
  "data/raw/HarryPotter_and_the_Sorcerers_Stone.pdf" \
  --output-dir $OUT/ocr_conf_high \
  --ocr-mode layout-aware \
  --enable-ocr \
  --ocr-confidence-threshold 0.9 \
  --vision-provider none
```

### A13. Doctr layer on/off
```bash
conda run -p ./env \
  mmrag-v2 process \
  "data/raw/HarryPotter_and_the_Sorcerers_Stone.pdf" \
  --output-dir $OUT/doctr_on \
  --ocr-mode layout-aware \
  --enable-ocr \
  --enable-doctr \
  --vision-provider none

conda run -p ./env \
  mmrag-v2 process \
  "data/raw/HarryPotter_and_the_Sorcerers_Stone.pdf" \
  --output-dir $OUT/doctr_off \
  --ocr-mode layout-aware \
  --enable-ocr \
  --no-doctr \
  --vision-provider none
```

### A14. Refiner (thresholds + max edit)
```bash
conda run -p ./env \
  mmrag-v2 process \
  "data/raw/HarryPotter_and_the_Sorcerers_Stone.pdf" \
  --output-dir $OUT/refiner_default \
  --enable-refiner \
  --refiner-provider openai \
  --refiner-model "mistral-7b-instruct-v0.3-mixed-6-8-bit" \
  --refiner-base-url "http://192.168.10.11:1234/v1" \
  --refiner-threshold 0.15 \
  --refiner-max-edit 0.35 \
  --vision-provider none

conda run -p ./env \
  mmrag-v2 process \
  "data/raw/HarryPotter_and_the_Sorcerers_Stone.pdf" \
  --output-dir $OUT/refiner_maxedit \
  --enable-refiner \
  --refiner-provider openai \
  --refiner-model "mistral-7b-instruct-v0.3-mixed-6-8-bit" \
  --refiner-base-url "http://192.168.10.11:1234/v1" \
  --refiner-threshold 0.05 \
  --refiner-max-edit 1.0 \
  --vision-provider none
```

### A15. --pages slicing (partial)
```bash
conda run -p ./env \
  mmrag-v2 process \
  "data/raw/Hybrid_electric_vehicles_and_their_challenges.pdf" \
  --output-dir $OUT/pages_subset \
  --pages 1,2,3 \
  --vision-provider none
```

---

## Phase B — Vision Provider Matrix

Run a small PDF with each provider, verifying image chunks include visual descriptions.

### B1. OpenAI (LM Studio endpoint)
```bash
conda run -p ./env \
  mmrag-v2 process \
  "data/raw/IRJET_Modeling_of_Solar_PV_system_under.pdf" \
  --output-dir $OUT/vision_openai \
  --vision-provider openai \
  --vision-model "llama-joycaption-beta-one-hf-llava-mmproj" \
  --vision-base-url "http://192.168.10.11:1234/v1" \
  --api-key "lm-studio"
```

### B2. Ollama
```bash
conda run -p ./env \
  mmrag-v2 process \
  "data/raw/IRJET_Modeling_of_Solar_PV_system_under.pdf" \
  --output-dir $OUT/vision_ollama \
  --vision-provider ollama
```

### B3. Anthropic
```bash
conda run -p ./env \
  mmrag-v2 process \
  "data/raw/IRJET_Modeling_of_Solar_PV_system_under.pdf" \
  --output-dir $OUT/vision_anthropic \
  --vision-provider anthropic
```

---

## Phase C — Intelligence Stack Parity

For each doc type: run `process` and `batch`, then compare profile/strategy.

1. Academic (digital): Hybrid_electric_vehicles...
2. Magazine/editorial: IRJET or PCWorld
3. Technical manual: Firearms
4. Scanned literature: HarryPotter

---

## Phase D — Non-PDF Format Coverage

Run `process` on:
- EPUB, HTML, DOCX, PPTX, XLSX

**Verify** JSONL exists and basic metadata fields appear.

---

## Phase E — Output Integrity

For each PDF run:
- Check metadata non-null:
  ```bash
  conda run -p ./env \
    python tests/test_bug009_metadata_propagation.py <ingestion.jsonl>
  ```
- Ensure assets folder exists when images/tables are present.
- Note QA warnings (token variance) without failing unless strict-qa is enabled.

---

## Pass/Fail Criteria
- All commands complete without fatal errors.
- JSONL produced for every run.
- Metadata propagation test passes for all PDF paths.
- Vision-provider runs produce visual descriptions (when enabled).
- Refiner runs complete and respect max edit budget.
- Non-PDF formats produce JSONL without crashes.
