#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs output
MASTER_LOG="logs/run_technical_manual_exact.runner.log"
: > "$MASTER_LOG"

log_msg() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$MASTER_LOG"
}

while IFS= read -r -d '' pdf; do
  stem="$(basename "$pdf" .pdf)"
  out="output/${stem}_refiner_fixed"
  doc_log="logs/${stem}_fixed_refiner_full.log"

  if [[ -f "$out/ingestion.jsonl" ]]; then
    log_msg "SKIP $pdf chunks=$(wc -l < "$out/ingestion.jsonl" | tr -d ' ')"
    continue
  fi

  log_msg "START $pdf"

  set +e
  conda run --no-capture-output -n mmrag-v2 python -u -m mmrag_v2.cli process "$pdf" \
    --output-dir "$out" \
    --profile-override technical_manual \
    --batch-size 3 \
    --enable-ocr --ocr-mode layout-aware --enable-doctr --ocr-confidence-threshold 0.4 \
    --vision-provider openai --vision-model "llama-joycaption-beta-one-hf-llava-mmproj" \
    --vision-base-url "http://192.168.10.11:1234/v1" --vlm-timeout 600 \
    --enable-refiner --refiner-provider openai \
    --refiner-model "mistral-7b-instruct-v0.3-mixed-6-8-bit" \
    --refiner-base-url "http://192.168.10.11:1234/v1" --refiner-max-edit 0.30 \
    --api-key "lm-studio" \
    --verbose > "$doc_log" 2>&1
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    log_msg "FAIL rc=$rc $pdf log=$doc_log"
  else
    if [[ -f "$out/ingestion.jsonl" ]]; then
      log_msg "OK $pdf chunks=$(wc -l < "$out/ingestion.jsonl" | tr -d ' ')"
    else
      log_msg "FAIL no-ingestion $pdf log=$doc_log"
    fi
  fi
done < <(find data/technical_manual -maxdepth 1 -type f -name '*.pdf' -print0)
