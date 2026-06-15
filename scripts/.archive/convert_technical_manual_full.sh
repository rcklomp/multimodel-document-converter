#!/usr/bin/env bash
set -euo pipefail

# Full technical_manual conversion with robust process supervision.
# Uses the exact conversion flags requested by the user, but adds:
# - per-doc retries on rc=137 (hard kill)
# - stale-log timeout detection (kill + retry)
# - master progress log with timestamps

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
ENV_NAME="${ENV_NAME:-mmrag-v2}"
DOC_GLOB="${DOC_GLOB:-data/technical_manual/*.pdf}"
MAX_RETRIES="${MAX_RETRIES:-3}"
STALL_TIMEOUT_SEC="${STALL_TIMEOUT_SEC:-1800}" # 30 minutes without logfile growth
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-30}"
SHARD_SIZE="${SHARD_SIZE:-40}" # pages per shard for large docs (0 disables sharding)
SKIP_DONE="${SKIP_DONE:-1}"

VISION_BASE_URL="${VISION_BASE_URL:-http://192.168.10.11:1234/v1}"
VISION_MODEL="${VISION_MODEL:-llama-joycaption-beta-one-hf-llava-mmproj}"
REFINER_BASE_URL="${REFINER_BASE_URL:-http://192.168.10.11:1234/v1}"
REFINER_MODEL="${REFINER_MODEL:-mistral-7b-instruct-v0.3-mixed-6-8-bit}"
API_KEY="${API_KEY:-lm-studio}"

mkdir -p "$ROOT_DIR/logs" "$ROOT_DIR/output"
MASTER_LOG="$ROOT_DIR/logs/technical_manual_full_$(date +%Y%m%d_%H%M%S).log"

now() { date -u "+%Y-%m-%dT%H:%M:%SZ"; }

log() {
  printf "%s %s\n" "[$(now)]" "$*" | tee -a "$MASTER_LOG"
}

run_one_attempt() {
  local pdf="$1"
  local out_dir="$2"
  local log_file="$3"
  local pages_spec="${4:-}"

  # Run exactly in the requested style: conda run ... process ...
  local -a cmd=(
    conda run --no-capture-output -n "$ENV_NAME" python -u -m mmrag_v2.cli process "$pdf"
    --output-dir "$out_dir"
    --profile-override technical_manual
    --batch-size 3
    --enable-ocr --ocr-mode layout-aware --enable-doctr --ocr-confidence-threshold 0.4
    --vision-provider openai --vision-model "$VISION_MODEL"
    --vision-base-url "$VISION_BASE_URL" --vlm-timeout 600
    --enable-refiner --refiner-provider openai
    --refiner-model "$REFINER_MODEL"
    --refiner-base-url "$REFINER_BASE_URL" --refiner-max-edit 0.30
    --api-key "$API_KEY"
    --verbose
  )
  if [[ -n "$pages_spec" ]]; then
    cmd+=(--pages "$pages_spec")
  fi
  "${cmd[@]}" >"$log_file" 2>&1 &

  local cmd_pid="$!"
  local last_size=0
  local last_change_ts
  last_change_ts="$(date +%s)"

  while kill -0 "$cmd_pid" 2>/dev/null; do
    local size=0
    if [[ -f "$log_file" ]]; then
      size="$(wc -c <"$log_file" | tr -d ' ')"
    fi
    if [[ "$size" -gt "$last_size" ]]; then
      last_size="$size"
      last_change_ts="$(date +%s)"
    fi

    local now_ts
    now_ts="$(date +%s)"
    local idle=$((now_ts - last_change_ts))
    if [[ "$idle" -ge "$STALL_TIMEOUT_SEC" ]]; then
      log "[STALL] no log growth for ${idle}s, killing pid=$cmd_pid"
      kill -TERM "$cmd_pid" 2>/dev/null || true
      sleep 3
      kill -KILL "$cmd_pid" 2>/dev/null || true
      wait "$cmd_pid" 2>/dev/null || true
      return 124
    fi
    sleep "$POLL_INTERVAL_SEC"
  done

  wait "$cmd_pid"
  return $?
}

get_page_count() {
  local pdf="$1"
  conda run --no-capture-output -n "$ENV_NAME" python - <<'PY' "$pdf"
import fitz
import sys
doc = fitz.open(sys.argv[1])
print(len(doc))
doc.close()
PY
}

run_with_retries() {
  local idx="$1"
  local total="$2"
  local pdf="$3"
  local out_dir="$4"
  local log_prefix="$5"
  local pages_spec="${6:-}"
  local ctx="${7:-doc}"
  local success=0

  for attempt in $(seq 1 "$MAX_RETRIES"); do
    local log_file="${log_prefix}.attempt${attempt}.log"
    log "[DOC-START][$idx/$total][$ctx][attempt=$attempt/$MAX_RETRIES] $pdf"
    set +e
    run_one_attempt "$pdf" "$out_dir" "$log_file" "$pages_spec"
    local rc=$?
    set -e

    if [[ "$rc" -eq 0 && -f "$out_dir/ingestion.jsonl" ]]; then
      local chunks
      chunks="$(wc -l <"$out_dir/ingestion.jsonl" | tr -d ' ')"
      log "[DOC-OK][$idx/$total][$ctx] $pdf rc=0 chunks=$chunks log=$(basename "$log_file")"
      success=1
      break
    fi

    log "[DOC-FAIL][$idx/$total][$ctx] $pdf rc=$rc ingestion=$([[ -f "$out_dir/ingestion.jsonl" ]] && echo yes || echo no) log=$(basename "$log_file")"
    if [[ "$rc" -ne 137 && "$rc" -ne 124 ]]; then
      break
    fi
    sleep 5
  done

  [[ "$success" -eq 1 ]]
}

cd "$ROOT_DIR"
PDFS=()
if [[ "$DOC_GLOB" == "data/technical_manual/*.pdf" ]]; then
  while IFS= read -r -d '' line; do
    PDFS+=("$line")
  done < <(find data/technical_manual -maxdepth 1 -type f -name '*.pdf' -print0)
else
  while IFS= read -r line; do
    [[ -n "$line" ]] && PDFS+=("$line")
  done < <(compgen -G "$DOC_GLOB" || true)
fi
TOTAL="${#PDFS[@]}"

if [[ "$TOTAL" -eq 0 ]]; then
  log "[FATAL] no pdfs found for DOC_GLOB=$DOC_GLOB"
  exit 1
fi

log "[START] docs=$TOTAL env=$ENV_NAME"
log "[CFG] retries=$MAX_RETRIES stall_timeout=${STALL_TIMEOUT_SEC}s poll=${POLL_INTERVAL_SEC}s shard_size=$SHARD_SIZE"

idx=0
for pdf in "${PDFS[@]}"; do
  idx=$((idx + 1))
  stem="$(basename "$pdf" .pdf)"
  out_dir="$ROOT_DIR/output/${stem}_refiner_fixed"
  mkdir -p "$out_dir"
  if [[ "$SKIP_DONE" -eq 1 && -f "$out_dir/ingestion.jsonl" ]]; then
    chunks="$(wc -l <"$out_dir/ingestion.jsonl" | tr -d ' ')"
    log "[DOC-SKIP][$idx/$TOTAL] $pdf existing_ingestion=yes chunks=$chunks"
    continue
  fi
  set +e
  page_count="$(get_page_count "$pdf" 2>/dev/null | tr -d '[:space:]')"
  page_rc=$?
  set -e
  if [[ "$page_rc" -ne 0 || -z "$page_count" || ! "$page_count" =~ ^[0-9]+$ ]]; then
    log "[DOC-FAIL][$idx/$TOTAL] $pdf page_count_error rc=$page_rc value='${page_count:-}'"
    continue
  fi

  if [[ "$SHARD_SIZE" -gt 0 && "$page_count" -gt "$SHARD_SIZE" ]]; then
    shard_root="$ROOT_DIR/output/.shards/${stem}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$shard_root"
    log "[DOC-SHARD][$idx/$TOTAL] $pdf pages=$page_count shard_size=$SHARD_SIZE"

    shard_failed=0
    start=1
    while [[ "$start" -le "$page_count" ]]; do
      end=$((start + SHARD_SIZE - 1))
      [[ "$end" -gt "$page_count" ]] && end="$page_count"

      pages_spec="$(seq "$start" "$end" | paste -sd, -)"
      shard_out="$shard_root/${start}_${end}"
      mkdir -p "$shard_out"
      prefix="$ROOT_DIR/logs/${stem}_fixed_refiner_full.${start}_${end}"
      if ! run_with_retries "$idx" "$TOTAL" "$pdf" "$shard_out" "$prefix" "$pages_spec" "pages=${start}-${end}"; then
        shard_failed=1
        break
      fi
      start=$((end + 1))
    done

    if [[ "$shard_failed" -eq 1 ]]; then
      log "[DOC-GIVEUP][$idx/$TOTAL] $pdf (sharded run failed)"
      continue
    fi

    mkdir -p "$out_dir/assets"
    : > "$out_dir/ingestion.jsonl"
    start=1
    while [[ "$start" -le "$page_count" ]]; do
      end=$((start + SHARD_SIZE - 1))
      [[ "$end" -gt "$page_count" ]] && end="$page_count"
      shard_out="$shard_root/${start}_${end}"
      cat "$shard_out/ingestion.jsonl" >> "$out_dir/ingestion.jsonl"
      if [[ -d "$shard_out/assets" ]]; then
        cp -f "$shard_out/assets/"* "$out_dir/assets/" 2>/dev/null || true
      fi
      start=$((end + 1))
    done
    chunks="$(wc -l <"$out_dir/ingestion.jsonl" | tr -d ' ')"
    log "[DOC-OK][$idx/$TOTAL] $pdf mode=sharded chunks=$chunks"
  else
    prefix="$ROOT_DIR/logs/${stem}_fixed_refiner_full"
    if ! run_with_retries "$idx" "$TOTAL" "$pdf" "$out_dir" "$prefix" "" "full"; then
      log "[DOC-GIVEUP][$idx/$TOTAL] $pdf"
    fi
  fi
done

log "[DONE] full batch complete"
log "[MASTER-LOG] $MASTER_LOG"
