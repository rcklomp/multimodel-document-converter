#!/usr/bin/env bash
# v2.16 Phase 0 — serial ingestion of 7 new PDFs in data/raw/.
# Each doc lands at output/<basename>/ingestion.jsonl per PLAN_V2.16.md §3 Phase 0 step 1.
set -uo pipefail

cd "$(dirname "$0")/.."

LOG_DIR=output/_v2_16_p0_logs
mkdir -p "$LOG_DIR"

# Map source PDFs to canonical snake_case basenames matching existing output/ conventions.
declare -a JOBS=(
  "Bevestigingsmiddelen.pdf|Bevestigingsmiddelen"
  "ATZ.-.Design.und.Aerodynamik.bei.Nutzfahrzeugen.pdf|ATZ_Aerodynamik_Nutzfahrzeugen"
  "ATZ.-.Das.Experimentalsicherheitsfahrzeug.ESF.2009.von.Mercedes.Benz.pdf|ATZ_ESF_Mercedes_2009"
  "Schwungradspeicher in der Fahrzeugtechnik.pdf|Schwungradspeicher"
  "Eliasz A. Zephyr RTOS Embedded C Programming. Using Embedded RTOS POSIX API 2024.pdf|Eliasz_Zephyr_RTOS"
  "Grundlagen Fahrzeug- und Motorentechnik.pdf|Grundlagen_Fahrzeug_Motorentechnik"
  "Digitale-Fotografie - Das essentielle Handbuch Februar 2026.pdf|Digitale_Fotografie_Feb_2026"
)

for spec in "${JOBS[@]}"; do
  src="${spec%%|*}"
  base="${spec##*|}"
  echo "=== $(date -Iseconds) START $base ===" | tee -a "$LOG_DIR/_master.log"
  log="$LOG_DIR/${base}.log"
  start=$(date +%s)
  mmrag-v2 process "data/raw/${src}" \
      --output-dir "output/${base}" \
      --batch-size 10 \
      > "$log" 2>&1
  rc=$?
  end=$(date +%s)
  elapsed=$((end - start))
  echo "=== $(date -Iseconds) END $base rc=$rc elapsed=${elapsed}s ===" | tee -a "$LOG_DIR/_master.log"
done

echo "=== $(date -Iseconds) ALL DONE ===" | tee -a "$LOG_DIR/_master.log"
