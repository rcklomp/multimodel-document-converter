#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Multi-profile smoke test.
#
# Runs one representative PDF from each document category and collects
# a cross-profile quality baseline.  Intentionally fast (few pages, no VLM)
# so it can be run frequently during development.
#
# Usage:
#   bash scripts/smoke_multiprofile.sh [output_root]
#
# Environment overrides:
#   PAGES        pages per doc        (default: 10)
#   BATCH_SIZE   batch size           (default: 3)
#   ENV_NAME     conda env name       (default: mmrag-v2)
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="${1:-output/smoke_multiprofile_$(date +%Y%m%d_%H%M%S)}"
PAGES="${PAGES:-10}"
BATCH_SIZE="${BATCH_SIZE:-3}"
ENV_NAME="${ENV_NAME:-mmrag-v2}"
mkdir -p "$ROOT"

SUMMARY="$ROOT/_summary.txt"
: >"$SUMMARY"
now() { date -u "+%Y-%m-%dT%H:%M:%SZ"; }

echo "smoke_multiprofile start=$(now)" | tee -a "$SUMMARY"
echo "ROOT=$ROOT  PAGES=$PAGES  BATCH_SIZE=$BATCH_SIZE" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

# ---------------------------------------------------------------------------
# Test matrix: category | pdf_path
# One representative PDF per non-empty category.
# Add rows here as the data/  corpus grows.
# ---------------------------------------------------------------------------
declare -a MATRIX=(
  "academic_journal|data/academic_journal/AIOS LLM Agent Operating System.pdf"
  "academic_journal|data/academic_journal/A_comprehensive_review_on_hybrid_electri.pdf"
  "business_form|data/business_form/betwistingsformulier_aankoop_niet_ontvangen.pdf"
  "data_spreadsheet|data/data_spreadsheet/CarOK voorraadtelling 2021-04.pdf"
  "digital_magazine|data/digital_magazine/PCWorld_July_2025_USA.pdf"
  "scanned_literature|data/scanned_literature/HarryPotter_and_the_Sorcerers_Stone.pdf"
  "technical_manual|data/technical_manual/Firearms.pdf"
  "technical_manual|data/technical_manual/Greenhouse Design and Control by Pedro Ponce.pdf"
  "technical_manual|data/technical_manual/Python Distilled David M. Beazley 2022.pdf"
  "technical_report|data/technical_report/ATZ.Elektronik.-.Effizientere.Software.Entwicklung.GERMAN.RETAiL.eBOOk-PDFWriters.pdf"
)

# Accumulate per-row result lines for the final summary table.
declare -a TABLE_ROWS=()

run_doc() {
  local category="$1"
  local pdf="$2"
  local stem out log
  stem="$(basename "$pdf" .pdf)"
  out="$ROOT/${category}__${stem}"
  log="$out/run.log"
  mkdir -p "$out"

  {
    echo "===== $category / $stem ====="
    echo "pdf=$pdf"
    echo "start=$(now)"
  } | tee -a "$SUMMARY"

  # Uniform conversion args — no profile override, no VLM, OCR in auto mode.
  set +e
  conda run -n "$ENV_NAME" python -m mmrag_v2.cli process "$pdf" \
    --output-dir "$out" \
    --batch-size "$BATCH_SIZE" \
    --pages "$PAGES" \
    --enable-ocr --ocr-mode auto --enable-doctr --ocr-confidence-threshold 0.4 \
    --vision-provider none \
    --no-refiner --no-cache \
    --verbose >"$log" 2>&1
  local exit_code=$?
  set -e

  echo "exit=$exit_code end=$(now)" | tee -a "$SUMMARY"

  if [[ $exit_code -ne 0 ]]; then
    echo "CONVERT_ERROR (see $log)" | tee -a "$SUMMARY"
    TABLE_ROWS+=("$category | $stem | CONVERT_ERROR | — | — | —")
    echo "" | tee -a "$SUMMARY"
    return 0
  fi

  local jsonl="$out/ingestion.jsonl"
  if [[ ! -f "$jsonl" ]]; then
    echo "MISSING_JSONL" | tee -a "$SUMMARY"
    TABLE_ROWS+=("$category | $stem | MISSING_JSONL | — | — | —")
    echo "" | tee -a "$SUMMARY"
    return 0
  fi

  # Extract detected profile from ingestion_metadata line.
  local detected_profile
  detected_profile=$(python3 -c "
import json, sys
for line in open('$jsonl'):
    o = json.loads(line.strip())
    if o.get('object_type') == 'ingestion_metadata':
        print(o.get('profile_type','unknown'))
        sys.exit(0)
print('no_metadata')
" 2>/dev/null || echo "error")

  # Run hygiene check.
  conda run -n "$ENV_NAME" python scripts/qa_ingestion_hygiene.py "$jsonl" \
    | tee -a "$SUMMARY" || true

  # Run structural gate (auto doc-class).
  local gate_result
  gate_result=$(conda run -n "$ENV_NAME" python scripts/evaluate_technical_manual_gates.py \
    "$jsonl" --doc-class auto 2>&1 | tee -a "$SUMMARY" \
    | grep -E "^GATE_(PASS|FAIL)" | head -1 || echo "GATE_ERROR")

  # Run universal invariant check.
  local universal_result
  universal_result=$(conda run -n "$ENV_NAME" python scripts/qa_universal_invariants.py \
    "$jsonl" 2>&1 | tee -a "$SUMMARY" \
    | grep -E "^UNIVERSAL_(PASS|FAIL)" | head -1 || echo "UNIVERSAL_ERROR")

  TABLE_ROWS+=("$category | $stem | $detected_profile | $gate_result | $universal_result")
  echo "" | tee -a "$SUMMARY"
}

# ---------------------------------------------------------------------------
# Run all entries in the matrix.
# ---------------------------------------------------------------------------
for entry in "${MATRIX[@]}"; do
  IFS='|' read -r cat pdf <<<"$entry"
  if [[ -f "$pdf" ]]; then
    run_doc "$cat" "$pdf"
  else
    echo "SKIP: missing $pdf" | tee -a "$SUMMARY"
    TABLE_ROWS+=("$cat | $(basename "$pdf" .pdf) | MISSING_FILE | — | —")
  fi
done

# ---------------------------------------------------------------------------
# Final summary table.
# ---------------------------------------------------------------------------
echo "" | tee -a "$SUMMARY"
echo "======================================================" | tee -a "$SUMMARY"
echo "CROSS-PROFILE SMOKE TEST SUMMARY" | tee -a "$SUMMARY"
echo "======================================================" | tee -a "$SUMMARY"
printf "%-22s | %-45s | %-25s | %-12s | %-16s\n" \
  "CATEGORY" "DOCUMENT" "DETECTED_PROFILE" "GATE" "UNIVERSAL" \
  | tee -a "$SUMMARY"
printf "%-22s-+-%-45s-+-%-25s-+-%-12s-+-%-16s\n" \
  "----------------------" "---------------------------------------------" \
  "-------------------------" "------------" "----------------" \
  | tee -a "$SUMMARY"
for row in "${TABLE_ROWS[@]}"; do
  IFS='|' read -r cat stem profile gate universal <<<"$row"
  printf "%-22s | %-45s | %-25s | %-12s | %-16s\n" \
    "$(echo "$cat" | xargs)" \
    "$(echo "$stem" | xargs | cut -c1-45)" \
    "$(echo "$profile" | xargs)" \
    "$(echo "$gate" | xargs)" \
    "$(echo "$universal" | xargs)" \
    | tee -a "$SUMMARY"
done
echo "" | tee -a "$SUMMARY"
echo "done=$(now)" | tee -a "$SUMMARY"
echo "summary=$SUMMARY"
