#!/usr/bin/env bash
# Regression test: convert all PDFs in data/technical_manual/ with the smart
# classifier, then re-run any GATE_FAIL with --profile-override technical_manual.
set -euo pipefail

ROOT="${1:-output/regression_techmanual_$(date +%Y%m%d_%H%M%S)}"
PAGES="${PAGES:-30}"
BATCH_SIZE="${BATCH_SIZE:-3}"
ENV_NAME="${ENV_NAME:-mmrag-v2}"
VISION_PROVIDER="${VISION_PROVIDER:-none}"
SEM_MAX_IMAGE_PLACEHOLDER="${SEM_MAX_IMAGE_PLACEHOLDER:-1.0}"
SEM_MAX_TABLE_PLACEHOLDER="${SEM_MAX_TABLE_PLACEHOLDER:-0.20}"
SEM_MIN_TABLE_MARKDOWN="${SEM_MIN_TABLE_MARKDOWN:-0.80}"
SEM_MIN_CODE_INDENT="${SEM_MIN_CODE_INDENT:-0.90}"

mkdir -p "$ROOT"
SUMMARY="$ROOT/_summary.txt"
: > "$SUMMARY"

now() { date -u "+%Y-%m-%dT%H:%M:%SZ"; }

echo "ROOT=$ROOT" | tee -a "$SUMMARY"
echo "PAGES=$PAGES BATCH_SIZE=$BATCH_SIZE ENV_NAME=$ENV_NAME VISION_PROVIDER=$VISION_PROVIDER" | tee -a "$SUMMARY"
echo "start=$(now)" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

PASS_DOCS=()
FAIL_DOCS=()
RETRY_DOCS=()

run_doc() {
  local pdf="$1"
  local override="${2:-}"   # optional: --profile-override technical_manual
  local stem out log label

  stem="$(basename "$pdf" .pdf)"
  label="${stem}"
  if [[ -n "$override" ]]; then
    out="$ROOT/${stem}_override"
    label="${stem} [OVERRIDE]"
  else
    out="$ROOT/$stem"
  fi
  log="$out/run.log"
  mkdir -p "$out"

  local -a EXTRA=()
  EXTRA+=(--enable-ocr --ocr-mode auto --enable-doctr --ocr-confidence-threshold 0.4)
  EXTRA+=(--vision-provider "$VISION_PROVIDER")
  EXTRA+=(--no-refiner --no-cache)
  if [[ -n "$override" ]]; then
    EXTRA+=("$override")
  fi

  { echo ""; echo "===== $label ====="; echo "pdf=$pdf"; echo "start=$(now)"; } | tee -a "$SUMMARY"

  set +e
  conda run -n "$ENV_NAME" python -m mmrag_v2.cli process "$pdf" \
    --output-dir "$out" \
    --batch-size "$BATCH_SIZE" \
    --pages "$PAGES" \
    "${EXTRA[@]}" \
    --verbose > "$log" 2>&1
  local code=$?
  set -e

  echo "exit=$code end=$(now)" | tee -a "$SUMMARY"
  if [[ "$code" -ne 0 ]]; then
    echo "PROCESS_FAIL: see $log" | tee -a "$SUMMARY"
    FAIL_DOCS+=("$stem")
    return 0
  fi

  local jsonl="$out/ingestion.jsonl"

  # Print IngestionMetadata fields
  conda run -n "$ENV_NAME" python3 - "$jsonl" <<'PYEOF' | tee -a "$SUMMARY"
import json, sys
with open(sys.argv[1]) as f:
    m = json.loads(f.readline())
print(f"profile_type={m.get('profile_type')} domain={m.get('domain')} is_scan={m.get('is_scan')} total_pages={m.get('total_pages')} chunk_count={m.get('chunk_count')}")
PYEOF

  conda run -n "$ENV_NAME" python scripts/qa_ingestion_hygiene.py "$jsonl" | tee -a "$SUMMARY"

  conda run -n "$ENV_NAME" python scripts/evaluate_technical_manual_gates.py \
    "$jsonl" --doc-class auto | tee -a "$SUMMARY"

  conda run -n "$ENV_NAME" python scripts/qa_semantic_fidelity.py \
    "$jsonl" \
    --max-image-placeholder-ratio "$SEM_MAX_IMAGE_PLACEHOLDER" \
    --max-table-placeholder-ratio "$SEM_MAX_TABLE_PLACEHOLDER" \
    --min-table-markdown-ratio "$SEM_MIN_TABLE_MARKDOWN" \
    --min-code-indentation-fidelity "$SEM_MIN_CODE_INDENT" | tee -a "$SUMMARY"
}

# ── Phase 1: smart classifier pass ─────────────────────────────────────────
echo "=== PHASE 1: Smart classifier (no override) ===" | tee -a "$SUMMARY"
for pdf in data/technical_manual/*.pdf; do
  [[ -f "$pdf" ]] || continue
  run_doc "$pdf"
done

# ── Phase 2: re-run any GATE_FAIL with technical_manual override ────────────
echo "" | tee -a "$SUMMARY"
echo "=== PHASE 2: Check for GATE_FAILs and retry with override ===" | tee -a "$SUMMARY"

for pdf in data/technical_manual/*.pdf; do
  [[ -f "$pdf" ]] || continue
  stem="$(basename "$pdf" .pdf)"
  gate_result=$(grep -A3 "===== $stem =====" "$SUMMARY" 2>/dev/null | grep "GATE_" | tail -1 || true)
  if [[ "$gate_result" == *"GATE_FAIL"* ]]; then
    echo "RETRYING $stem with --profile-override technical_manual" | tee -a "$SUMMARY"
    run_doc "$pdf" "--profile-override technical_manual"
    RETRY_DOCS+=("$stem")
  fi
done

# ── Final summary table ──────────────────────────────────────────────────────
echo "" | tee -a "$SUMMARY"
echo "=== FINAL RESULTS TABLE ===" | tee -a "$SUMMARY"
printf "%-55s %-22s %-12s %-14s\n" "Document" "Profile" "Gate" "Semantic" | tee -a "$SUMMARY"
printf "%-55s %-22s %-12s %-14s\n" "--------" "-------" "----" "--------" | tee -a "$SUMMARY"

for pdf in data/technical_manual/*.pdf; do
  [[ -f "$pdf" ]] || continue
  stem="$(basename "$pdf" .pdf)"
  for suffix in "" "_override"; do
    outdir="$ROOT/${stem}${suffix}"
    jsonl="$outdir/ingestion.jsonl"
    [[ -f "$jsonl" ]] || continue
    profile=$(python3 -c "import json; m=json.loads(open('$jsonl').readline()); print(m.get('profile_type','?'))" 2>/dev/null || echo "?")
    label="${stem:0:50}${suffix:+[OVR]}"
    gate=$(grep -A40 "===== ${stem}${suffix:+ [OVERRIDE]} =====" "$SUMMARY" 2>/dev/null | grep "^GATE_" | tail -1 || echo "?")
    sem=$(grep -A40 "===== ${stem}${suffix:+ [OVERRIDE]} =====" "$SUMMARY" 2>/dev/null | grep "^SEMANTIC_" | tail -1 || echo "?")
    printf "%-55s %-22s %-12s %-14s\n" "$label" "$profile" "$gate" "$sem" | tee -a "$SUMMARY"
  done
done

echo "" | tee -a "$SUMMARY"
echo "done=$(now)" | tee -a "$SUMMARY"
echo "summary=$SUMMARY"
