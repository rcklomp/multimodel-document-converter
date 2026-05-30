#!/usr/bin/env bash
# =============================================================================
# v3_sequential_soak.sh — GX10 time-sliced V3 extraction + evaluation.
#
# The GX10 (NVIDIA GB10, aarch64, 121 GiB unified mem) runs ONE vLLM at a time
# on :8000. This orchestrator time-slices it:
#
#   boot Qwen3-VL (extraction)  ->  extract corpus  ->  KILL + flush VRAM
#   ->  boot Qwen2.5-14B-FP8 (judge)  ->  index + soak  ->  KILL.
#
# Extraction + soak run LOCALLY (this Mac) and call the GX10 vLLM over the LAN.
# vLLM lifecycle is driven over passwordless SSH (`ssh gx10`, docker --rm).
#
# Usage:
#   scripts/v3_sequential_soak.sh smoke   # 2 small docs, tiny soak (DEFAULT)
#   scripts/v3_sequential_soak.sh full    # entire data/ corpus
#
# Verify-before-convert (CLAUDE.md): run `smoke` and inspect the report BEFORE
# committing the GX10 to a multi-day `full` run.
# =============================================================================
set -euo pipefail

# ---- config -----------------------------------------------------------------
MODE="${1:-smoke}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-$HOME/miniforge3/envs/mmrag-v2/bin/python}"
GX10_SSH="${GX10_SSH:-gx10}"                       # ~/.ssh/config alias (passwordless)
GX10_IP="${GX10_IP:-10.0.10.239}"
ENDPOINT="http://${GX10_IP}:8000/v1"
COMPOSE_DIR="${COMPOSE_DIR:-/home/ronald/gx10-serving}"        # holds docker-compose.yml on the GX10
LOCAL_COMPOSE="$REPO_ROOT/scripts/gx10/docker-compose.yml"     # repo = single source of truth

# Compose service + container names (container_name is pinned in the compose file).
# Memory budgets, image pin, gpus, logging, restart policy all live in the compose file.
VLM_SERVICE="qwen-vlm";     VLM_CONTAINER="gx10-vllm-qwen3vl"
JUDGE_SERVICE="qwen-judge"; JUDGE_CONTAINER="gx10-vllm-14b-fp8"
VLM_SERVED="Qwen3-VL-8B-Instruct"                              # must match compose --served-model-name
JUDGE_SERVED="RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic"
VLM_BOOT_TIMEOUT="${VLM_BOOT_TIMEOUT:-1500}"                   # first boot may download ~16 GB
JUDGE_BOOT_TIMEOUT="${JUDGE_BOOT_TIMEOUT:-900}"                # post-reboot cold cache loads slower

LOG_DIR="output/v3_sequential_soak"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/orchestrator.log"

log()  { printf '%s [seq-soak] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$RUN_LOG" ; }
die()  { log "FATAL: $*"; exit 1; }

# ---- GX10 vLLM lifecycle (Tier-2: docker compose — supervised + leak-proof) -
gx10()    { ssh -o BatchMode=yes -o ConnectTimeout=10 "$GX10_SSH" "$@"; }
compose() { gx10 "cd '$COMPOSE_DIR' && docker compose $*"; }

ensure_compose_file() {  # push the repo compose file up so the GX10 always runs the version-controlled config
  [ -f "$LOCAL_COMPOSE" ] || die "missing $LOCAL_COMPOSE"
  gx10 "mkdir -p '$COMPOSE_DIR'"
  scp -q "$LOCAL_COMPOSE" "$GX10_SSH:$COMPOSE_DIR/docker-compose.yml" || die "failed to stage compose file"
  compose "config --quiet" || die "invalid compose file on GX10"
}

container_running() { gx10 "docker ps --format '{{.Names}}' | grep -qx '$1'"; }

compose_down() {  # deterministic teardown — works even if a prior run was SIGKILLed (no zombie-container leak)
  log "compose down (deterministic cleanup) ..."
  compose "down --remove-orphans" >/dev/null 2>&1 || true
  local t=0
  while gx10 "(ss -tln 2>/dev/null || netstat -tln 2>/dev/null) | grep -q ':8000 '"; do
    sleep 2; t=$((t+2))
    if [ "$t" -ge 60 ]; then die "port 8000 never freed on GX10"; fi
  done
  log "port 8000 closed; sleeping 15s for OS VRAM flush ..."
  sleep 15
}

wait_health() {  # poll the OpenAI /v1/models endpoint until 200 or timeout
  local timeout="$1" container="$2" t=0
  log "waiting for $ENDPOINT/models (timeout ${timeout}s) ..."
  while true; do
    if curl -s --max-time 5 -o /dev/null -w '%{http_code}' "$ENDPOINT/models" 2>/dev/null | grep -q 200; then
      log "endpoint healthy after ${t}s."; return 0
    fi
    if ! container_running "$container"; then
      log "container $container not running during boot — last 40 log lines:"
      gx10 "docker logs --tail 40 '$container' 2>&1" | tee -a "$RUN_LOG" || true
      die "vLLM container $container died before becoming healthy"
    fi
    sleep 5; t=$((t+5))
    if [ "$t" -ge "$timeout" ]; then
      gx10 "docker logs --tail 40 '$container' 2>&1" | tee -a "$RUN_LOG" || true
      die "health timeout for $container"
    fi
  done
}

boot_service() {  # $1 compose-service, $2 container-name, $3 health-timeout
  compose_down                       # clean slate (frees the GPU pool; heals any prior leak)
  log "compose up -d $1 ..."
  compose "up -d $1" | tee -a "$RUN_LOG"
  wait_health "$3" "$2"
}

boot_vlm()   { log "booting Qwen3-VL extraction model on GX10 ...";  boot_service "$VLM_SERVICE"   "$VLM_CONTAINER"   "$VLM_BOOT_TIMEOUT"; }
boot_judge() { log "booting judge model on GX10 ...";                boot_service "$JUDGE_SERVICE" "$JUDGE_CONTAINER" "$JUDGE_BOOT_TIMEOUT"; }

cleanup() {  # always tear down via compose (deterministic, no orphan leak)
  local rc=$?
  log "cleanup (exit $rc): docker compose down ..."
  compose "down --remove-orphans" >/dev/null 2>&1 || true
  exit $rc
}
trap cleanup EXIT INT TERM

ensure_compose_file

# ---- extraction target (smoke vs full) --------------------------------------
case "$MODE" in
  smoke)
    DATA_DIR="$(mktemp -d /tmp/v3seq_smoke.XXXX)/data"
    mkdir -p "$DATA_DIR/business_form" "$DATA_DIR/academic_journal"
    cp "data/business_form/0013_140302111325_001.pdf"                     "$DATA_DIR/business_form/"
    cp "data/academic_journal/IRJET_Modeling_of_Solar_PV_system_under.pdf" "$DATA_DIR/academic_journal/"
    export N_CHUNKS=8
    EXTRACT_FORCE="--force"   # re-extract through the GX10 VLM even if cached, so the smoke truly tests extraction
    log "MODE=smoke: 2 docs (Form_0013 table + IRJET figures), forced re-extract, tiny soak."
    ;;
  full)
    DATA_DIR="data"
    export N_CHUNKS=50
    EXTRACT_FORCE=""          # resume-safe: skip docs already extracted, process the remaining corpus
    log "MODE=full: entire data/ corpus (resume-safe) — multi-day GX10 commitment."
    ;;
  *) die "unknown MODE '$MODE' (use: smoke | full)";;
esac

# =============================================================================
log "=== PHASE 1: boot Qwen3-VL extraction model ==="
boot_vlm

log "=== PHASE 2: extract corpus through HybridEngine -> GX10 VLM ==="
# NOTE (2026-05-30): v3_execution_root was removed; its chunker no longer on the path.
# scripts/v3_batch_ingest.py still imports `mmrag_v3.chunking.chunker` from that
# sandbox and MUST be repointed to src/mmrag_v2/chunking/uir_chunker before this
# soak runs again. (Sandbox recoverable from ~/mmrag_v3_execution_root_backup_2026-05-30.tar.gz.)
USE_VLM_ENGINE=1 \
VLM_NATIVE_ENDPOINT="$ENDPOINT" \
VLM_NATIVE_MODEL="$VLM_SERVED" \
VLM_NATIVE_API_KEY="${VLM_NATIVE_API_KEY:-EMPTY}" \
PYTHONPATH="$REPO_ROOT/src" \
  "$PY" scripts/v3_batch_ingest.py --data-dir "$DATA_DIR" --out-dir output/v3_baselines ${EXTRACT_FORCE} \
  2>&1 | tee -a "$LOG_DIR/extract.log"

log "=== PHASE 3: kill Qwen3-VL + flush VRAM ==="
compose_down

log "=== PHASE 4: boot judge model ==="
boot_judge

log "=== PHASE 5: index + soak (judge + gen on GX10) ==="
PY="$PY" N_CHUNKS="$N_CHUNKS" bash scripts/v3_post_batch_pipeline.sh 2>&1 | tee -a "$LOG_DIR/soak.log"

log "=== PHASE 6: kill judge ==="
compose_down

log "=== DONE. Logs in $LOG_DIR/ ; soak report in output/v3_soak/report.v3.md ==="
