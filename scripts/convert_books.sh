#!/usr/bin/env bash
# Convert every PDF/EPUB in data/ for the v2.9 broad reconversion.
# Forms (business_form/) and spreadsheets (data_spreadsheet/) are first-class
# RAG content per PLAN_V2.8 §5a — included.
# Uses ~/.mmrag-v2.yml for VLM + refiner credentials.
#
# v2.9 (2026-05-04): --no-refiner DROPPED. The CLI's smart-routing fix
# (cli._decide_enable_refiner; v2.9 Phase 2) only auto-enables the
# refiner when has_encoding_corruption=True. Clean prose docs (HARRY)
# no longer hammer qwen-plus per chunk. Encoding-corrupt docs
# (Combat-class) still get the heal-over without manual flag intervention.
set -uo pipefail

LOG="output/_convert_books.log"
: >"$LOG"
now() { date "+%Y-%m-%d %H:%M:%S"; }

PASS=0
FAIL=0

convert() {
  local src="$1"
  local name="$2"
  local batch_size="${3:-10}"
  local outdir="output/$name"

  # Always reconvert — never skip stale outputs
  if [ -d "$outdir" ]; then
    rm -rf "$outdir"
  fi

  echo "$(now) START $name" | tee -a "$LOG"
  # v2.9 broad-reconversion flags. VLM is OFF here because Phase 5b runs
  # a targeted image-only enrichment via scripts/enrich_image_chunks_v29.py
  # AFTER conversion completes. --no-refiner is no longer needed: the
  # CLI's smart-routing fix only auto-enables the refiner when
  # has_encoding_corruption=True (v2.9 Phase 2).
  # Use ``conda run -n mmrag-v2`` so this script works whether the env
  # is activated in the calling shell or not.
  conda run -n mmrag-v2 --no-capture-output python -m mmrag_v2.cli process \
    "$src" -o "$outdir" -b "$batch_size" \
    --vision-provider none --no-cache >>"$LOG" 2>&1
  local exit_code=$?

  if [ $exit_code -eq 0 ] && [ -f "$outdir/ingestion.jsonl" ]; then
    local chunks=$(wc -l < "$outdir/ingestion.jsonl" | tr -d ' ')
    echo "$(now) DONE $name ($chunks lines)" | tee -a "$LOG"
    PASS=$((PASS + 1))
  else
    echo "$(now) FAIL $name (exit=$exit_code)" | tee -a "$LOG"
    FAIL=$((FAIL + 1))
  fi
}

echo "$(now) === BROAD RECONVERSION v2.9.0 ===" | tee -a "$LOG"
echo ""

# === DIGITAL LITERATURE ===
convert "data/digital_literature/HarryPotter_and_the_Sorcerers_Stone.pdf" "HarryPotter_and_the_Sorcerers_Stone" 10

# === BUSINESS FORMS / INVOICES (form lane per QUALITY_GATES.md) ===
convert "data/business_form/0013_140302111325_001.pdf" "Form_0013_invoice" 10
convert "data/business_form/betwistingsformulier_aankoop_niet_ontvangen.pdf" "Form_betwistingsformulier" 10

# === DATA SPREADSHEETS ===
convert "data/data_spreadsheet/CarOK voorraadtelling 2021-04.pdf" "CarOK_voorraadtelling" 10

# === ACADEMIC JOURNALS ===
convert "data/academic_journal/AIOS LLM Agent Operating System.pdf" "AIOS_LLM_Agent_Operating_System" 10
convert "data/academic_journal/A_comprehensive_review_on_hybrid_electri.pdf" "A_comprehensive_review_on_hybrid_electri" 10
convert "data/academic_journal/Hybrid_electric_vehicles_and_their_challenges.pdf" "Hybrid_electric_vehicles" 10
convert "data/academic_journal/IRJET_Modeling_of_Solar_PV_system_under.pdf" "IRJET_Modeling_of_Solar_PV" 10
convert "data/academic_journal/Recent_Trends_in_Transportation_Technolo.pdf" "Recent_Trends_in_Transportation" 10

# === DIGITAL MAGAZINES ===
convert "data/digital_magazine/Combat Aircraft - August 2025 UK.pdf" "Combat_Aircraft_August_2025" 10
convert "data/digital_magazine/PCWorld_July_2025_USA.pdf" "PCWorld_July_2025" 10

# === TECHNICAL REPORT (German) ===
convert "data/technical_report/ATZ.Elektronik.-.Effizientere.Software.Entwicklung.GERMAN.RETAiL.eBOOk-PDFWriters.pdf" "ATZ_Elektronik_German" 10

# === TECHNICAL MANUALS — Short ===
convert "data/technical_manual/A Simple Guide to Retrieval Augmented Generation Kimothi A. 2025.pdf" "Kimothi_RAG_Guide" 10
convert "data/technical_manual/integra_u_en.pdf" "Integra_manual" 10
convert "data/technical_manual/Jungjun H. Build an AI Agent (From Scratch)...MEAP 2026.pdf" "Jungjun_AI_Agent" 10

# === TECHNICAL MANUALS — RAG/AI books ===
convert "data/technical_manual/Bourne K. Unlocking Data with Generative AI and RAG 2024.pdf" "Bourne_RAG_2024" 10
convert "data/technical_manual/Devlin M. Building LLM Agents with RAG, Knowledge Graphs and Reflection...2025.pdf" "Devlin_LLM_Agents" 10
convert "data/technical_manual/Raieli S. Building AI Agents with LLMs, RAG, and Knowledge Graphs...2025.pdf" "Raieli_AI_Agents" 10
convert "data/technical_manual/Adedeji A. GenAI on Google Cloud. Enterprise Generative AI Systems...Agents 2026.pdf" "Adedeji_GenAI_Google_Cloud" 10
convert "data/technical_manual/Cronin I. Building and Training Generative AI Models. A Practical Guide...2026.pdf" "Cronin_GenAI_Models" 10
convert "data/technical_manual/Hao B. Machine Learning Platform Engineering. Build...for ML and AI systems 2026.pdf" "Hao_ML_Platform" 10
convert "data/technical_manual/Nagasubramanian D. Agentic AI for Engineers.Architecting Goal-Driven System 2026.pdf" "Nagasubramanian_Agentic_AI" 10
convert "data/technical_manual/Sekar S. The MCP Standard. A Developer's Guide..Building Universal AI Tools 2026.pdf" "Sekar_MCP_Standard" 10

# === TECHNICAL MANUALS — Python books ===
# Ayeva + Chaubal trigger CodeFormulaV2 (PLAN_V2.8 §4) — ~27 sec/page CPU.
# Chaubal alone adds ~2.5 hrs to the broad reconversion.
convert "data/technical_manual/Python Cookbook  Everyone can cook delicious recipes with Python.pdf" "Python_Cookbook" 10
convert "data/technical_manual/Programming ArcGIS with Python Cookbook.pdf" "ArcGIS_Python_Cookbook" 10
convert "data/technical_manual/Fluent Python Luciano Ramalho 2015.pdf" "Fluent_Python" 10
convert "data/technical_manual/Python Distilled David M. Beazley 2022.pdf" "Python_Distilled" 10
convert "data/technical_manual/Ayeva K. Mastering Python Design Patterns...essential Python patterns...3ed 2024.pdf" "Ayeva_Python_Patterns" 10
convert "data/technical_manual/Chaubal S. AI Projects in PyTorch. Hands-On Projects in Vision, Text,...2025.pdf" "Chaubal_PyTorch_Projects" 10

# === TECHNICAL MANUALS — Domain-specific ===
convert "data/technical_manual/Earthship_Vol1_How to build your own.pdf" "Earthship_Vol1" 10
convert "data/technical_manual/Firearms.pdf" "Firearms" 10
convert "data/technical_manual/Greenhouse Design and Control by Pedro Ponce.pdf" "Greenhouse_Design" 10

# === EPUBs ===
convert "data/technical_manual/Falkner, Leonie - ChatGPT Praktijk-handboek.epub" "ChatGPT_Praktijk_handboek" 10
convert "data/technical_manual/Seffer, David - KI En ChatGPT, Praktische Gids Voor Online Business Met Digitale Producten.epub" "KI_En_ChatGPT_Praktische_Gids" 10

echo ""
echo "$(now) === ALL DONE ($PASS pass, $FAIL fail) ===" | tee -a "$LOG"

echo ""
echo "=== RESULTS ==="
for d in output/*/; do
  if [ -f "$d/ingestion.jsonl" ]; then
    name=$(basename "$d")
    chunks=$(wc -l < "$d/ingestion.jsonl" | tr -d ' ')
    # Check schema version
    ver=$(head -1 "$d/ingestion.jsonl" | python -c "import sys,json; print(json.load(sys.stdin).get('schema_version','?'))" 2>/dev/null)
    printf "  %-50s %4s lines  v%s\n" "$name" "$chunks" "$ver"
  fi
done | sort
