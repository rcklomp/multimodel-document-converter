#!/usr/bin/env bash
# ===========================================================================
# env_cloud_vlm.sh - Cloud VLM route config (2026-09-29).
#
# Replaces the retired LOCAL route. The old local stack (omlx-server on
# 10.0.10.246:8000 serving NuMarkdown-8B-Thinking-mlx-8bits, and the
# Qwen3-VL-8B-Instruct-8bit instance on macbook-pro-m5.lan) was slow
# (~33 tok/s; dense pages hit the 8192-token cap at ~249 s/page) and gave
# weaker extraction. Both LAN servers are DOWN as of 2026-09-29. This file
# re-arms the pipeline on fast CLOUD OpenAI-compatible vision models:
# default = Dashscope-intl qwen3-vl-flash (verified live 2026-09-29:
# image+text round trip ~1 s).
#
# Usage:
#   source scripts/env_cloud_vlm.sh          # then run CLI/scripts
# Requires DASHSCOPE_API_KEY in the environment (never commit the key).
#
# Alternative (OpenRouter route) if the Dashscope key is unavailable:
#   export VLM_NATIVE_ENDPOINT=https://openrouter.ai/api/v1
#   export VLM_NATIVE_MODEL=qwen/qwen3-vl-235b-a22b-instruct   # or -30b-a3b-instruct,
#                                                              # deepseek/deepseek-v4-flash-vision-exp
#   export VLM_NATIVE_API_KEY=$OPENROUTER_API_KEY
# ===========================================================================

# --- V3 VLM-native extraction route (mmrag_v3.engines.vlm_provider) --------
export VLM_NATIVE_ENDPOINT="${VLM_NATIVE_ENDPOINT:-https://dashscope-intl.aliyuncs.com/compatible-mode/v1}"
export VLM_NATIVE_MODEL="${VLM_NATIVE_MODEL:-qwen3-vl-flash}"
if [ -z "${VLM_NATIVE_API_KEY:-}" ]; then
  if [ -n "${DASHSCOPE_API_KEY:-}" ]; then
    export VLM_NATIVE_API_KEY="$DASHSCOPE_API_KEY"
  else
    echo "[env_cloud_vlm] ERROR: set DASHSCOPE_API_KEY (or VLM_NATIVE_API_KEY) first." >&2
  fi
fi
# Dashscope compatible-mode is not openrouter.ai/openai.com, so from_env()
# treats it as self-hosted and would default repetition_penalty=1.1 (a
# non-standard param). Keep it off; JSON stays mandated via the prompt.
export VLM_NATIVE_REPETITION_PENALTY="${VLM_NATIVE_REPETITION_PENALTY:-off}"

# --- MinerU (LAN GX10) is DOWN as of 2026-09-29 ----------------------------
# Keep MINERU_ENDPOINT UNSET: with it unset, mmrag_v3.extract() defaults to the
# legacy HybridEngine (Docling + VLM pre-flight) and VLM-heavy pages route to
# qwen3-vl-flash above. When GX10 (10.0.10.239:8001) is powered back on,
# uncomment to re-arm the production MineruQwenHybrid default route:
# export MINERU_ENDPOINT="http://10.0.10.239:8001"

# --- Engine pinning (optional) ----------------------------------------------
# Default route with MINERU_ENDPOINT unset = legacy HybridEngine. Force one:
# export USE_VLM_ENGINE=1        # pure VLM-native (qwen3-vl-flash every page)
# export USE_DOCLING_FAST=1      # offline CI mode (no VLM) - NOT for acceptance

echo "[env_cloud_vlm] VLM_NATIVE_ENDPOINT=$VLM_NATIVE_ENDPOINT"
echo "[env_cloud_vlm] VLM_NATIVE_MODEL=$VLM_NATIVE_MODEL (key ${VLM_NATIVE_API_KEY:+SET})"
