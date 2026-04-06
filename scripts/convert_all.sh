#!/usr/bin/env bash
# Convert all documents with Qwen VLM + refiner
set -uo pipefail

API_KEY="sk-5813a0a803ca4b96ab8755b1068f10fd"
BASE_URL="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
VLM_MODEL="qwen-vl-max"
REFINER_MODEL="qwen-plus"
LOG="output/_convert_all.log"

mkdir -p output
: >"$LOG"

now() { date "+%Y-%m-%d %H:%M:%S"; }

convert() {
  local pdf="$1"
  local name="$2"
  local batch_size="${3:-10}"
  local extra="${4:-}"

  local outdir="output/$name"
  rm -rf "$outdir"

  echo "$(now) START $name" | tee -a "$LOG"

  python -m mmrag_v2.cli process "$pdf" \
    --output-dir "$outdir" \
    --vision-provider openai \
    --vision-model "$VLM_MODEL" \
    --api-key "$API_KEY" \
    --vision-base-url "$BASE_URL" \
    --vlm-timeout 120 \
    --enable-refiner \
    --refiner-provider openai \
    --refiner-model "$REFINER_MODEL" \
    --refiner-base-url "$BASE_URL" \
    --batch-size "$batch_size" \
    $extra \
    >>"$LOG" 2>&1

  local exit_code=$?
  if [ $exit_code -eq 0 ]; then
    local chunks=$(wc -l < "$outdir/ingestion.jsonl" 2>/dev/null | tr -d ' ')
    echo "$(now) DONE $name ($chunks chunks)" | tee -a "$LOG"
  else
    echo "$(now) FAIL $name (exit=$exit_code)" | tee -a "$LOG"
  fi
  return $exit_code
}

echo "$(now) === FULL CONVERSION RUN ===" | tee -a "$LOG"

# Academic journals (short, digital)
convert "data/academic_journal/AIOS LLM Agent Operating System.pdf" "AIOS_LLM_Agent_Operating_System" 10
convert "data/academic_journal/A_comprehensive_review_on_hybrid_electri.pdf" "A_comprehensive_review_on_hybrid_electri" 10
convert "data/academic_journal/Hybrid_electric_vehicles_and_their_challenges.pdf" "Hybrid_electric_vehicles" 10
convert "data/academic_journal/IRJET_Modeling_of_Solar_PV_system_under.pdf" "IRJET_Modeling_of_Solar_PV" 10
convert "data/academic_journal/Recent_Trends_in_Transportation_Technolo.pdf" "Recent_Trends_in_Transportation" 10

# Business forms (short, scanned)
convert "data/business_form/betwistingsformulier_aankoop_niet_ontvangen.pdf" "betwistingsformulier" 3
convert "data/business_form/0013_140302111325_001.pdf" "Levoil_invoice" 3

# Data spreadsheet
convert "data/data_spreadsheet/CarOK voorraadtelling 2021-04.pdf" "CarOK_voorraadtelling" 10

# Digital magazines (medium, image-heavy)
convert "data/digital_magazine/PCWorld_July_2025_USA.pdf" "PCWorld_July_2025" 10
convert "data/digital_magazine/Combat Aircraft - August 2025 UK.pdf" "Combat_Aircraft_August_2025" 10

# Technical report (short, German)
convert "data/technical_report/ATZ.Elektronik.-.Effizientere.Software.Entwicklung.GERMAN.RETAiL.eBOOk-PDFWriters.pdf" "ATZ_Elektronik_German" 10

# Technical manuals - short/medium
convert "data/technical_manual/integra_u_en.pdf" "Integra_manual" 10
convert "data/technical_manual/A Simple Guide to Retrieval Augmented Generation Kimothi A. 2025.pdf" "Kimothi_RAG_Guide" 10
convert "data/technical_manual/Jungjun H. Build an AI Agent (From Scratch)...MEAP 2026.pdf" "Jungjun_AI_Agent" 10

# Technical manuals - medium RAG/AI books
convert "data/technical_manual/Bourne K. Unlocking Data with Generative AI and RAG 2024.pdf" "Bourne_RAG_2024" 10
convert "data/technical_manual/Devlin M. Building LLM Agents with RAG, Knowledge Graphs and Reflection...2025.pdf" "Devlin_LLM_Agents" 10
convert "data/technical_manual/Raieli S. Building AI Agents with LLMs, RAG, and Knowledge Graphs...2025.pdf" "Raieli_AI_Agents" 10
convert "data/technical_manual/Python Cookbook  Everyone can cook delicious recipes with Python.pdf" "Python_Cookbook" 10

# Technical manuals - long books
convert "data/technical_manual/Greenhouse Design and Control by Pedro Ponce.pdf" "Greenhouse_Design" 10
convert "data/technical_manual/Firearms.pdf" "Firearms" 3
convert "data/technical_manual/Programming ArcGIS with Python Cookbook.pdf" "ArcGIS_Python_Cookbook" 10
convert "data/technical_manual/Fluent Python Luciano Ramalho 2015.pdf" "Fluent_Python" 10
convert "data/technical_manual/Python Distilled David M. Beazley 2022.pdf" "Python_Distilled" 10

echo ""
echo "$(now) === ALL CONVERSIONS COMPLETE ===" | tee -a "$LOG"

# Summary
echo ""
echo "=== RESULTS ==="
for d in output/*/; do
  if [ -f "$d/ingestion.jsonl" ]; then
    name=$(basename "$d")
    chunks=$(wc -l < "$d/ingestion.jsonl" | tr -d ' ')
    printf "  %-45s %s chunks\n" "$name" "$chunks"
  fi
done | sort
