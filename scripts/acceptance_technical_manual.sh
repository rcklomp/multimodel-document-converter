#!/usr/bin/env bash
set -euo pipefail

# Acceptance benchmark for technical_manual conversion quality.
# Fast by default (20 pages, batch-size 3), with explicit pass/fail gates.

ROOT="${1:-output/acceptance_techmanual_$(date +%Y%m%d_%H%M%S)}"
PAGES="${PAGES:-20}"
BATCH_SIZE="${BATCH_SIZE:-3}"
ENV_NAME="${ENV_NAME:-mmrag-v2}"

mkdir -p "$ROOT"
SUMMARY="$ROOT/_summary.txt"
: >"$SUMMARY"

now() { date -u "+%Y-%m-%dT%H:%M:%SZ"; }

echo "ROOT=$ROOT" | tee -a "$SUMMARY"
echo "PAGES=$PAGES BATCH_SIZE=$BATCH_SIZE ENV_NAME=$ENV_NAME" | tee -a "$SUMMARY"
echo "start=$(now)" | tee -a "$SUMMARY"

# Representative set:
# - Firearms: scanned/manual with many figures
# - Earthship: noisy scanned technical manual
# - Python Distilled: native digital technical book
# - Greenhouse Design: BLIND TEST (not in dev-loop, must pass)
declare -a DOCS=(
  "data/technical_manual/Firearms.pdf"
  "data/technical_manual/Earthship_Vol1_How to build your own.pdf"
  "data/technical_manual/Python Distilled David M. Beazley 2022.pdf"
  "data/technical_manual/Greenhouse Design and Control by Pedro Ponce.pdf"
)

run_doc() {
  local pdf="$1"
  local stem out log
  stem="$(basename "$pdf" .pdf)"
  out="$ROOT/$stem"
  log="$out/run.log"
  mkdir -p "$out"

  local -a OCR_ARGS
  local -a EXTRA_ARGS

  # Uniform extraction settings for blind validation; avoid filename-driven behavior.
  local DOC_CLASS
  OCR_ARGS=(--enable-ocr --ocr-mode auto --enable-doctr --ocr-confidence-threshold 0.4)

  # Keep benchmark stable and lightweight.
  EXTRA_ARGS=(--vision-provider none --no-refiner --no-cache)

  {
    echo ""
    echo "===== $stem ====="
    echo "pdf=$pdf"
    echo "start=$(now)"
    echo "doc_class=pending (inferred from output metadata)"
    echo "ocr_args=${OCR_ARGS[*]}"
    echo "extra_args=${EXTRA_ARGS[*]}"
  } | tee -a "$SUMMARY"

  set +e
  conda run -n "$ENV_NAME" python -m mmrag_v2.cli process "$pdf" \
    --output-dir "$out" \
    --profile-override technical_manual \
    --batch-size "$BATCH_SIZE" \
    --pages "$PAGES" \
    "${OCR_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    --verbose >"$log" 2>&1
  local code=$?
  set -e

  echo "exit=$code end=$(now)" | tee -a "$SUMMARY"
  if [[ "$code" -ne 0 ]]; then
    echo "FAIL: process command failed ($log)" | tee -a "$SUMMARY"
    return 0
  fi

  local jsonl="$out/ingestion.jsonl"
  conda run -n "$ENV_NAME" python scripts/qa_ingestion_hygiene.py "$jsonl" | tee -a "$SUMMARY"

  DOC_CLASS="$(
    conda run -n "$ENV_NAME" python -c "import json,sys;from collections import Counter;p=sys.argv[1];mods=Counter();f=open(p,'r',encoding='utf-8');\
[mods.update([str(((json.loads(line).get('metadata') or {}).get('document_modality') or '')).lower()]) for line in f if str(((json.loads(line).get('metadata') or {}).get('document_modality') or '')).lower()];\
f.close();\
print('digital' if (not mods or mods.most_common(1)[0][0] in {'native_digital','image_heavy'}) else 'scanned')" "$jsonl" | grep -E "^(digital|scanned)$" | tail -n 1 | tr -d "[:space:]"
  )"
  echo "doc_class=$DOC_CLASS (inferred)" | tee -a "$SUMMARY"

  conda run -n "$ENV_NAME" python scripts/evaluate_technical_manual_gates.py \
    "$jsonl" \
    --doc-class "$DOC_CLASS" | tee -a "$SUMMARY"
}

for pdf in "${DOCS[@]}"; do
  if [[ -f "$pdf" ]]; then
    run_doc "$pdf"
  else
    echo "SKIP: missing $pdf" | tee -a "$SUMMARY"
  fi
done

echo "" | tee -a "$SUMMARY"
echo "done=$(now)" | tee -a "$SUMMARY"
echo "summary=$SUMMARY"
