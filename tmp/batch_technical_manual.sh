#!/usr/bin/env bash
set -euo pipefail

BASE="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$BASE/data/technical_manual"
LOG="$BASE/logs/batch_technical_manual_$(date +%Y%m%d_%H%M%S).log"

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

echo "Batch started: $(date)" | tee "$LOG"
echo "Log: $LOG"

# Ordered smallest → largest
run "Devlin M. Building LLM Agents with RAG, Knowledge Graphs and Reflection...2025.pdf" \
    devlin_llm_agents --batch-size 5

run "Python Cookbook  Everyone can cook delicious recipes with Python.pdf" \
    python_cookbook --batch-size 5

run "Programming ArcGIS with Python Cookbook.pdf" \
    arcgis_python --batch-size 5

run "Jungjun H. Build an AI Agent (From Scratch)...MEAP 2026.pdf" \
    jungjun_ai_agent --batch-size 5

run "Fluent Python Luciano Ramalho 2015.pdf" \
    fluent_python --batch-size 3

run "Bourne K. Unlocking Data with Generative AI and RAG 2024.pdf" \
    bourne_rag_2024 --batch-size 3

run "Greenhouse Design and Control by Pedro Ponce.pdf" \
    greenhouse_design --batch-size 3

run "Firearms.pdf" \
    firearms --batch-size 3

run "Python Distilled David M. Beazley 2022.pdf" \
    python_distilled --batch-size 2

run "Raieli S. Building AI Agents with LLMs, RAG, and Knowledge Graphs...2025.pdf" \
    raieli_ai_agents --batch-size 2

run "Earthship_Vol1_How to build your own.pdf" \
    earthship_vol1 --batch-size 2

echo "" | tee -a "$LOG"
echo "All done: $(date)" | tee -a "$LOG"
