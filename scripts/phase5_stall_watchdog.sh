#!/usr/bin/env bash
# Kill a stalled phase5 extraction so the full-corpus run can advance. A hung
# request (e.g. GX10 MinerU degenerate-repetition with no logits processor + the
# relay's no-timeout forward) can block an extraction child indefinitely. If a
# `mmrag_v2.cli process` child is running but NOTHING under output/phase5_reextract
# has been written for STALL_MIN minutes, the extraction is hung -> SIGKILL it.
# The orchestrator's subprocess.run then returns, the doc is recorded NO_OUTPUT ->
# LADDER_FAIL, and the run continues. External (system bash) - no orchestrator restart.
cd "$(dirname "$0")/.."
STALL_MIN=12
while true; do
  pids=$(pgrep -f "mmrag_v2.cli process")
  if [ -n "$pids" ]; then
    if [ -z "$(find output/phase5_reextract -type f -newermt "-${STALL_MIN} minutes" 2>/dev/null | head -1)" ]; then
      echo "[watchdog] STALL: no output for ${STALL_MIN}min; killing extraction $pids $(date -Iseconds)"
      kill -9 $pids 2>/dev/null
    fi
  fi
  sleep 120
done
