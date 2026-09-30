#!/usr/bin/env bash
set -euo pipefail

# Fast, repeatable smoke run for technical manuals.
# Processes only the first N pages of each PDF and prints hygiene metrics.

ROOT="${1:-output/smoke_techmanual_$(date +%Y%m%d_%H%M%S)}"
PAGES="${PAGES:-40}"
BATCH_SIZE="${BATCH_SIZE:-3}"

mkdir -p "$ROOT"
SUMMARY="$ROOT/_summary.txt"
: >"$SUMMARY"

now() { date -u "+%Y-%m-%dT%H:%M:%SZ"; }

echo "ROOT=$ROOT" | tee -a "$SUMMARY"
echo "PAGES=$PAGES BATCH_SIZE=$BATCH_SIZE" | tee -a "$SUMMARY"
echo "start=$(now)" | tee -a "$SUMMARY"

find data/technical_manual -maxdepth 1 -type f -name '*.pdf' -print0 | sort -z | \
while IFS= read -r -d '' f; do
  stem="$(basename "$f" .pdf)"
  out="$ROOT/$stem"
  mkdir -p "$out"

  OCR_ARGS=(--no-ocr --ocr-mode legacy --no-doctr)
  case "$stem" in
    *Earthship*|*Firearms*) OCR_ARGS=(--enable-ocr --ocr-mode auto) ;;
  esac

  {
    echo ""
    echo "===== $(basename "$f") ====="
    echo "start=$(now)"
  } | tee -a "$SUMMARY"

  set +e
  conda run -n mmrag-v2 python -m mmrag_v2.cli process "$f" \
    -o "$out" \
    --batch-size "$BATCH_SIZE" \
    --pages "$PAGES" \
    --vision-provider none \
    --no-refiner \
    --profile-override technical_manual \
    --no-cache \
    "${OCR_ARGS[@]}" \
    >"$out/run.log" 2>&1
  code=$?
  set -e

  echo "exit=$code" | tee -a "$SUMMARY"
  echo "end=$(now)" | tee -a "$SUMMARY"
  if [[ "$code" -ne 0 ]]; then
    echo "ERROR (see $out/run.log)" | tee -a "$SUMMARY"
    continue
  fi

  conda run -n mmrag-v2 python scripts/qa_ingestion_hygiene.py "$out/ingestion.jsonl" | tee -a "$SUMMARY"
done

echo "" | tee -a "$SUMMARY"
echo "done=$(now)" | tee -a "$SUMMARY"
echo "summary=$SUMMARY"

