#!/usr/bin/env bash
# PLAN_FIDELITY_ORACLE_FIRST_V1 Phase 1: code-fidelity bake-off.
# One code-dense Fluent Python slice (p286-325, the densest 40-page window) run
# through every candidate engine, then scored by scripts/code_repo_oracle.py
# against the author repo (github.com/fluentpython/example-code).
# Relay env (proven from conda); cap1600 default. Sequential against M5/GX10.
set -u
cd /Users/Shared/Projects/MM-Converter-V2.4.1
PY=/Users/Shared/miniforge3/envs/mmrag-v2/bin/python
REPO=/tmp/fp_oracle_test
export MINERU_ENDPOINT=http://127.0.0.1:18001 MINERU_MODEL=MinerU2.5-2509-1.2B
export VLM_NATIVE_ENDPOINT=http://127.0.0.1:18000/v1 VLM_NATIVE_MODEL=mlx-community/Qwen3-VL-8B-Instruct-8bit VLM_NATIVE_API_KEY=EMPTY

OUT=output/p1_code_oracle
mkdir -p "$OUT/_src"
SRC="$OUT/_src/FluentPython_p286-325.pdf"

$PY - <<PYEOF
import fitz
d=fitz.open("data/technical_manual/Fluent Python Luciano Ramalho 2015.pdf")
o=fitz.open(); o.insert_pdf(d, from_page=285, to_page=324)
o.save("$SRC"); print("slice pages:", o.page_count)
PYEOF

run_engine () {
  local name="$1"; shift
  local out="$OUT/$name"
  echo "[$(date +%H:%M:%S)] === $name START ==="
  # reset engine flags each run
  unset USE_DOCLING_FAST USE_MINERU_ENGINE USE_VLM_ENGINE USE_HYBRID_ENGINE USE_MINERU_QWEN_HYBRID
  "$@" $PY -m mmrag_v2.cli process "$SRC" --output-dir "$out" --batch-size 10 --vision-provider none > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] === $name DONE rc=$rc ==="
  if [ -f "$out/ingestion.jsonl" ]; then
    $PY scripts/code_repo_oracle.py --jsonl "$out/ingestion.jsonl" --repo "$REPO" --label "$name" | tee -a "$OUT/_oracle_summary.txt"
    echo "" >> "$OUT/_oracle_summary.txt"
  fi
}

: > "$OUT/_oracle_summary.txt"
# fast-first: docling offline, then mineru, then vlm-only, then hybrid (the prod default)
run_engine docling_fast   env USE_DOCLING_FAST=1
run_engine mineru_only    env USE_MINERU_ENGINE=1
run_engine vlm_qwen_only  env USE_VLM_ENGINE=1
run_engine hybrid_default env
echo "[$(date +%H:%M:%S)] === P1 BAKEOFF ALL DONE ==="
echo "=== ORACLE SUMMARY ===" && cat "$OUT/_oracle_summary.txt"
