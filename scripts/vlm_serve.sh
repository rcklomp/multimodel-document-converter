#!/usr/bin/env bash
# Explicit control for the M5 Qwen3-VL inference server (mlx-vlm).
# Deliberate load — you start/stop it; `status` shows the RAM footprint so a
# loaded model is never invisible. Survives your shell/session (detached), but
# does NOT auto-start at boot. See scripts/m5_max_setup.md.
set -euo pipefail

ENV="vlm-server"                                   # isolated conda env (§3 of the runbook)
MODEL="${VLM_MODEL:-mlx-community/Qwen3-VL-8B-Instruct-8bit}"
PORT="${VLM_PORT:-8000}"
HOST="${VLM_HOST:-0.0.0.0}"                         # LAN-reachable; set 127.0.0.1 for loopback-only
KEYFILE="$HOME/.mlx_api_key"
LOG="$HOME/Library/Logs/mlx-vlm.log"
PIDFILE="$HOME/.mlx-vlm.pid"
CONDA="${CONDA_EXE:-/opt/homebrew/bin/conda}"

pid_running() { [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }
# the process actually LISTENing on the port = the real model-holding server
# (not the `conda run` wrapper), so RSS reflects the true RAM footprint.
cur_pid()     { lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | head -1; }

status() {
  local pid; pid="$(cur_pid || true)"
  if [[ -z "$pid" ]]; then echo "● stopped (port $PORT free)"; return 1; fi
  local rss_kb rss_gb
  rss_kb="$(ps -o rss= -p "$pid" | tr -d ' ')"
  rss_gb="$(awk "BEGIN{printf \"%.1f\", ${rss_kb:-0}/1024/1024}")"
  echo "● running  pid=$pid  RAM=${rss_gb}GB  model=$MODEL"
  echo "  endpoint: http://${HOST/0.0.0.0/$(ipconfig getifaddr en0 2>/dev/null || echo localhost)}:${PORT}/v1"
  echo "  log:      $LOG"
}

start() {
  if [[ -n "$(cur_pid || true)" ]]; then echo "already running:"; status; return 0; fi
  [[ -f "$KEYFILE" ]] || { echo "no API key at $KEYFILE"; exit 1; }
  echo "loading $MODEL on :$PORT (host $HOST) — this pulls ~9GB into RAM…"
  MLX_API_KEY="$(cat "$KEYFILE")" nohup \
    "$CONDA" run -n "$ENV" python -m mlx_vlm.server \
      --model "$MODEL" --port "$PORT" --host "$HOST" \
      >>"$LOG" 2>&1 &
  echo $! >"$PIDFILE"
  for _ in $(seq 1 60); do
    curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q '"id"' && { echo "ready."; status; return 0; }
    sleep 1
  done
  echo "did not become ready in 60s — check $LOG"; exit 1
}

stop() {
  local pid; pid="$(cur_pid || true)"
  if [[ -z "$pid" ]]; then echo "not running."; rm -f "$PIDFILE"; return 0; fi
  echo "stopping pid $pid (frees the model from RAM)…"
  pkill -f "mlx_vlm.server .*--port $PORT" 2>/dev/null || true
  rm -f "$PIDFILE"; echo "stopped."
}

case "${1:-status}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; sleep 1; start ;;
  status)  status ;;
  *) echo "usage: $0 {start|stop|restart|status}"; exit 2 ;;
esac
