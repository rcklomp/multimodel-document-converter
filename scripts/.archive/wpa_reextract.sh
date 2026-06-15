#!/usr/bin/env bash
# PLAN_F1 WP-A acceptance re-extractions (sequential against M5).
# FluentPython 15pg slice (no-regression) -> Jungjun (no-regression) -> Chaubal (P2 gate).
# Relay env (proven from conda); prod hybrid config; cap1600 default.
set -u
cd /Users/Shared/Projects/MM-Converter-V2.4.1
PY=/Users/Shared/miniforge3/envs/mmrag-v2/bin/python
export MINERU_ENDPOINT=http://127.0.0.1:18001 MINERU_MODEL=MinerU2.5-2509-1.2B
export VLM_NATIVE_ENDPOINT=http://127.0.0.1:18000/v1 VLM_NATIVE_MODEL=mlx-community/Qwen3-VL-8B-Instruct-8bit VLM_NATIVE_API_KEY=EMPTY
unset USE_DOCLING_FAST USE_MINERU_ENGINE USE_VLM_ENGINE USE_HYBRID_ENGINE USE_MINERU_QWEN_HYBRID

OUT=output/wpa
mkdir -p "$OUT/_src"

# Slice FluentPython p60-74 (15 pages)
$PY - <<'PYEOF'
import fitz, os
src="data/technical_manual/Fluent Python Luciano Ramalho 2015.pdf"
d=fitz.open(src); o=fitz.open(); o.insert_pdf(d, from_page=59, to_page=73)
o.save("output/wpa/_src/FluentPython_p60-74.pdf"); print("FluentPython slice:", o.page_count, "pages")
PYEOF

run_one () {
  local name="$1"; local src="$2"; local out="$OUT/$name"
  echo "[$(date +%H:%M:%S)] === $name START ==="
  $PY -m mmrag_v2.cli process "$src" --output-dir "$out" --batch-size 10 --vision-provider none > "$OUT/$name.log" 2>&1
  echo "[$(date +%H:%M:%S)] === $name DONE rc=$? ==="
}

run_one FluentPython "output/wpa/_src/FluentPython_p60-74.pdf"
run_one Jungjun "data/technical_manual/Jungjun H. Build an AI Agent (From Scratch)...MEAP 2026.pdf"
run_one Chaubal "data/technical_manual/Chaubal S. AI Projects in PyTorch. Hands-On Projects in Vision, Text,...2025.pdf"
echo "[$(date +%H:%M:%S)] === WPA ALL DONE ==="
