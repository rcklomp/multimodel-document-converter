#!/usr/bin/env bash
# Convert all ebooks (PDF/EPUB) excluding business_form, data_spreadsheet,
# presentation, and Jaaropgave folders.
# Uses ~/.mmrag-v2.yml for VLM + refiner credentials.
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
  python -m mmrag_v2.cli process "$src" -o "$outdir" -b "$batch_size" >>"$LOG" 2>&1
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

echo "$(now) === BOOK CONVERSION v2.7.0 ===" | tee -a "$LOG"
echo ""

# === SCANNED ===
convert "data/scanned/HarryPotter_and_the_Sorcerers_Stone.pdf" "HarryPotter_and_the_Sorcerers_Stone" 10

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

# === TECHNICAL MANUALS — Python books ===
convert "data/technical_manual/Python Cookbook  Everyone can cook delicious recipes with Python.pdf" "Python_Cookbook" 10
convert "data/technical_manual/Programming ArcGIS with Python Cookbook.pdf" "ArcGIS_Python_Cookbook" 10
convert "data/technical_manual/Fluent Python Luciano Ramalho 2015.pdf" "Fluent_Python" 10
convert "data/technical_manual/Python Distilled David M. Beazley 2022.pdf" "Python_Distilled" 10

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
