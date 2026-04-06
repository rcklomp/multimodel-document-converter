#!/usr/bin/env bash
set -euo pipefail

BASE="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$BASE/data/technical_manual"
LOG="$BASE/logs/rerun_scanned_$(date +%Y%m%d_%H%M%S).log"
MMRAG="/Users/ronald/miniforge3/envs/mmrag-v2/bin/mmrag-v2"

COMMON=(
  --profile-override technical_manual
  --vision-provider openai
  --vision-model "numarkdown-8b-thinking-mlxs"
  --vision-base-url "http://192.168.10.11:1234/v1"
  --api-key "lm-studio"
  --vlm-timeout 600
  --no-refiner
  --no-force-table-vlm
  --enable-ocr
)

run() {
  local file="$1" outdir="$2"
  shift 2
  echo "" | tee -a "$LOG"
  echo "========================================" | tee -a "$LOG"
  echo "START: $outdir  ($(date))" | tee -a "$LOG"
  echo "FILE:  $file" | tee -a "$LOG"
  echo "========================================" | tee -a "$LOG"
  "$MMRAG" process "$DATA/$file" \
    --output-dir "$BASE/output/$outdir" \
    "${COMMON[@]}" "$@" 2>&1 | tee -a "$LOG"
  echo "DONE:  $outdir  ($(date))" | tee -a "$LOG"
}

echo "Scanned rerun started: $(date)" | tee "$LOG"
echo "Log: $LOG"

run "Firearms.pdf"                        firearms_ocr       --batch-size 3
run "Earthship_Vol1_How to build your own.pdf" earthship_ocr --batch-size 2

echo "" | tee -a "$LOG"
echo "All done: $(date)" | tee -a "$LOG"
