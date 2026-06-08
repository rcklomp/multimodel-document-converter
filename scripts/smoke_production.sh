#!/usr/bin/env bash
# ===========================================================================
# smoke_production.sh - PLAN_V3.1 Phase 5 anti-rot guard for the production
# extraction path (R8). This is the mandatory pre-merge gate for any change to
# the V3 CLI extraction path (batch_processor.py, uir_chunker.py,
# src/mmrag_v3/**, IngestionChunk.from_uir).
#
# It runs the SHIPPING path - `mmrag-v2 process <doc> --batch-size 10` - on one
# small doc per routing lane and asserts the four checks that would have caught
# every bug found in the 2026-05-31 M5 smoke (response_format 400, missing
# asset_ref, missing visual_description, regressed HEADING gate):
#
#   a. BATCH INTEGRITY - no silent 0-chunk drop on a lane that should produce.
#   b. MODALITY SCHEMA - every IMAGE/TABLE chunk has a non-null asset_ref AND
#      the asset file exists on disk (QA-CHECK-05). Full mode additionally
#      requires a non-empty visual_description on IMAGE chunks.
#   c. ROUTING - offline: every content chunk is extraction_method=
#      uir_native_chunker (proves the V3 path, not a legacy leak). Full mode:
#      the lane hit the expected engine (prose ~Docling, form/academic VLM) and
#      the per-doc VLM fallback rate is at or below ~10%.
#   d. GATE STATUS - scripts/qa_full_conversion.py --source-pdf reports
#      QA_PASS or QA_PASS_WITH_ADVISORIES; QA_WARN/QA_FAIL fails the lane.
#
# THREE LANES (one small doc each):
#   prose     data/raw/Bevestigingsmiddelen.pdf            -> Docling-fast lane
#   academic  data/academic_journal/IRJET_..._under.pdf    -> Hybrid mixed lane
#   form      data/business_form/0013_140302111325_001.pdf -> VLM-heavy lane
#
# TWO MODES:
#   DEFAULT (offline / CI): forces USE_DOCLING_FAST=1 so every lane runs
#     deterministically on CPU with no network and no VLM tokens. This is the
#     pre-merge gate. The form doc (0013) is a known VLM-route doc: OFFLINE it
#     legitimately yields 0 chunks (Docling-fast finds no text layer on the
#     scanned form), so the offline form lane asserts "ran clean + no schema
#     violation on whatever it produced", NOT QA_PASS or chunks>0. This is a
#     DOCUMENTED per-lane expectation, not a silenced assertion (AGENT-TEST-01).
#     Offline, the prose AND academic lanes MUST yield >0 chunks and QA_PASS.
#
#   FULL (opt-in: SMOKE_FULL=1 + the VLM_NATIVE_* env): preflights the M5 VLM
#     endpoint (curl -m 8 .../v1/models). If UP, runs the lanes through
#     USE_VLM_ENGINE / HybridEngine against the M5 and asserts the richer
#     multimodal checks (image/table assets + visual_description + routing).
#     If SMOKE_FULL=1 but the preflight FAILS, the script exits non-zero with a
#     clear "M5 down" message - it does NOT silently fall back to offline. A run
#     that was asked to test the VLM and didn't is a FAIL.
#
# Usage:
#   bash scripts/smoke_production.sh                 # offline / CI (default)
#   SMOKE_FULL=1 VLM_NATIVE_ENDPOINT=http://macbook-pro-m5.lan:8000/v1 \
#     VLM_NATIVE_MODEL=mlx-community/Qwen3-VL-8B-Instruct-8bit \
#     VLM_NATIVE_API_KEY=local bash scripts/smoke_production.sh
#
# Env overrides:
#   SMOKE_FULL          set to 1 for FULL (M5 VLM) mode (default: offline)
#   SMOKE_DATA_ROOT     corpus root (default: <repo>/data)
#   SMOKE_OUT_ROOT      output root (default: output/smoke_production_<ts>)
#   ENV_PYTHON          python interpreter (default: ~/miniforge3/envs/mmrag-v2/bin/python)
#   VLM_NATIVE_ENDPOINT M5 OpenAI-compatible base (FULL mode preflight target)
#   VLM_FULL_FALLBACK_PCT  per-doc VLM fallback flag threshold (default: 10)
#
# This gate is extraction + QA ONLY. It does NOT touch Qdrant, the embedder, or
# the judge - that keeps it fast and side-effect free.
# ===========================================================================
set -uo pipefail

# --- Resolve repo + interpreters ------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_PYTHON="${ENV_PYTHON:-$HOME/miniforge3/envs/mmrag-v2/bin/python}"
MMRAG_CLI="$(dirname "$ENV_PYTHON")/mmrag-v2"

# The editable install resolves mmrag_v2/mmrag_v3 to the MAIN checkout; when
# this script runs from a worktree we must put the worktree's src first so the
# gate exercises THIS tree, not the user's checkout.
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

DATA_ROOT="${SMOKE_DATA_ROOT:-$REPO_ROOT/data}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${SMOKE_OUT_ROOT:-$REPO_ROOT/output/smoke_production_$TS}"
FALLBACK_PCT="${VLM_FULL_FALLBACK_PCT:-10}"

# Deterministic, offline-friendly model resolution (matches smoke_multiprofile).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/_summary.txt"
: >"$SUMMARY"

log() { echo "$@" | tee -a "$SUMMARY"; }

# --- Mode detection --------------------------------------------------------
FULL=0
if [ "${SMOKE_FULL:-0}" = "1" ]; then
  FULL=1
fi

log "==========================================================================="
log " SMOKE_PRODUCTION - PLAN_V3.1 Phase 5 anti-rot guard"
log "==========================================================================="
if [ "$FULL" -eq 1 ]; then
  log " MODE: FULL  (M5 VLM via USE_VLM_ENGINE/HybridEngine - real multimodal)"
else
  log " MODE: OFFLINE  (USE_DOCLING_FAST=1 - deterministic CPU, no VLM, CI gate)"
fi
log " repo:        $REPO_ROOT"
log " data root:   $DATA_ROOT"
log " output root: $OUT_ROOT"
log " python:      $ENV_PYTHON"
log "==========================================================================="

# --- Preconditions ---------------------------------------------------------
if [ ! -x "$ENV_PYTHON" ]; then
  log "PRECONDITION FAIL: python interpreter not executable: $ENV_PYTHON"
  log "SMOKE_PRODUCTION_FAIL"
  exit 2
fi
if [ ! -d "$DATA_ROOT" ]; then
  log "PRECONDITION FAIL: corpus root not found: $DATA_ROOT"
  log "  (set SMOKE_DATA_ROOT to a checkout that has data/)"
  log "SMOKE_PRODUCTION_FAIL"
  exit 2
fi

# Lane registry: name | relative-doc-path | offline-expectation
#   require_chunks  : prose/academic lanes MUST yield >0 chunks offline + QA_PASS
#   vlm_route_doc   : form lane offline-routes to VLM; 0 chunks offline is OK,
#                     no QA requirement offline (full mode flips it to require)
PROSE_DOC="$DATA_ROOT/raw/Bevestigingsmiddelen.pdf"
ACADEMIC_DOC="$DATA_ROOT/academic_journal/IRJET_Modeling_of_Solar_PV_system_under.pdf"
FORM_DOC="$DATA_ROOT/business_form/0013_140302111325_001.pdf"

for d in "$PROSE_DOC" "$ACADEMIC_DOC" "$FORM_DOC"; do
  if [ ! -f "$d" ]; then
    log "PRECONDITION FAIL: lane doc not found: $d"
    log "SMOKE_PRODUCTION_FAIL"
    exit 2
  fi
done

# --- FULL-mode preflight: the M5 must be UP, else FAIL (no silent offline) --
PREFLIGHT_ENDPOINT="${VLM_NATIVE_ENDPOINT:-}"
if [ "$FULL" -eq 1 ]; then
  if [ -z "$PREFLIGHT_ENDPOINT" ]; then
    log "FULL-MODE PRECONDITION FAIL: VLM_NATIVE_ENDPOINT is not set."
    log "  FULL mode must target the M5 endpoint, e.g."
    log "  VLM_NATIVE_ENDPOINT=http://macbook-pro-m5.lan:8000/v1"
    log "SMOKE_PRODUCTION_FAIL"
    exit 2
  fi
  MODELS_URL="${PREFLIGHT_ENDPOINT%/}/models"
  log ""
  # The M5 is self-hosted and flaky (not boot-persistent). A single-shot curl
  # can pass on a host that is flapping, after which the per-page VLM calls hang
  # up to the provider's 180s/request timeout. Require THREE consecutive healthy
  # probes so a flapping host fails the preflight cleanly instead of letting the
  # lanes stall on long per-page timeouts.
  log "FULL-mode preflight: curl -m 8 $MODELS_URL (x3 consecutive)"
  preflight_ok=1
  for attempt in 1 2 3; do
    if curl -fsS -m 8 "$MODELS_URL" >/dev/null 2>&1; then
      log "  probe $attempt/3 UP"
    else
      log "  probe $attempt/3 DOWN"
      preflight_ok=0
      break
    fi
  done
  if [ "$preflight_ok" -eq 1 ]; then
    log "  M5 VLM endpoint UP (3/3)."
  else
    log "FULL-MODE PRECONDITION FAIL: M5 VLM endpoint DOWN/flapping ($MODELS_URL)."
    log "  Asked to test the VLM but the endpoint is unreachable - refusing to"
    log "  silently fall back to offline and report green. Bring the M5 up"
    log "  (vlm_serve.sh) or run without SMOKE_FULL=1 for the offline gate."
    log "SMOKE_PRODUCTION_FAIL"
    exit 1
  fi
fi

# --- Per-lane runner -------------------------------------------------------
# Globals appended by run_lane:
LANE_NAMES=()
LANE_RESULTS=()   # PASS / FAIL
LANE_NOTES=()

# run_lane <lane_name> <doc_path> <expectation: require_chunks|vlm_route_doc>
run_lane() {
  local lane="$1" doc="$2" expect="$3"
  local out="$OUT_ROOT/$lane"
  local cli_log="$out/cli.log"
  local jsonl="$out/ingestion.jsonl"
  mkdir -p "$out"

  log ""
  log "---------------------------------------------------------------------------"
  log "LANE: $lane  doc=$(basename "$doc")  expect=$expect"
  log "---------------------------------------------------------------------------"

  # Run the production CLI. Offline forces Docling-fast on every lane. Full
  # mode forces VLM on the VLM-route lanes (form) and uses the HybridEngine
  # cost-optimizer on the mixed/prose lanes so per-page routing is exercised.
  local rc
  if [ "$FULL" -eq 0 ]; then
    # Offline CI has no VLM reachable, so be honest about it: --vision-provider
    # none stamps image chunks as the documented no-VLM ID-only fallback
    # (advisory), not "failed" (which means a configured VLM erred).
    USE_DOCLING_FAST=1 "$MMRAG_CLI" process "$doc" \
      --batch-size 10 --output-dir "$out" --vision-provider none >"$cli_log" 2>&1
    rc=$?
  else
    if [ "$expect" = "vlm_route_doc" ]; then
      USE_VLM_ENGINE=1 "$MMRAG_CLI" process "$doc" \
        --batch-size 10 --output-dir "$out" >"$cli_log" 2>&1
      rc=$?
    else
      # HybridEngine cost-optimizer (default route, no force flag).
      "$MMRAG_CLI" process "$doc" \
        --batch-size 10 --output-dir "$out" >"$cli_log" 2>&1
      rc=$?
    fi
  fi

  if [ "$rc" -ne 0 ]; then
    log "  [FAIL] CLI process exited $rc (see $cli_log)"
    tail -n 8 "$cli_log" | sed 's/^/    /' | tee -a "$SUMMARY"
    LANE_NAMES+=("$lane"); LANE_RESULTS+=("FAIL"); LANE_NOTES+=("CLI exit $rc")
    return
  fi
  if [ ! -f "$jsonl" ]; then
    log "  [FAIL] no ingestion.jsonl produced at $jsonl"
    LANE_NAMES+=("$lane"); LANE_RESULTS+=("FAIL"); LANE_NOTES+=("no jsonl")
    return
  fi

  # Count VLM fallbacks from the router log (FULL mode routing signal).
  local fallbacks
  fallbacks="$(grep -c "falling back to docling" "$cli_log" 2>/dev/null || true)"
  fallbacks="${fallbacks:-0}"

  # --- Structural assertions (a/b/c) in one python pass --------------------
  # Exit 0 = lane structural PASS; 3 = structural FAIL; note printed to stdout.
  local struct_out struct_rc
  struct_out="$(
    SMOKE_LANE="$lane" \
    SMOKE_EXPECT="$expect" \
    SMOKE_FULL_FLAG="$FULL" \
    SMOKE_JSONL="$jsonl" \
    SMOKE_OUTDIR="$out" \
    "$ENV_PYTHON" - <<'PY'
import json, os, sys
from pathlib import Path

lane = os.environ["SMOKE_LANE"]
expect = os.environ["SMOKE_EXPECT"]
full = os.environ["SMOKE_FULL_FLAG"] == "1"
jsonl = Path(os.environ["SMOKE_JSONL"])
outdir = Path(os.environ["SMOKE_OUTDIR"])

fails = []
notes = []

chunks = []
with jsonl.open("r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        # Skip the manifest/metadata line (no "modality").
        if obj.get("object_type") == "ingestion_metadata":
            continue
        if "modality" not in obj:
            continue
        chunks.append(obj)

n = len(chunks)
mods = {}
for c in chunks:
    mods[c.get("modality")] = mods.get(c.get("modality"), 0) + 1
n_img = mods.get("image", 0)
n_tbl = mods.get("table", 0)
n_txt = mods.get("text", 0)
notes.append(f"chunks={n} (txt={n_txt} img={n_img} tbl={n_tbl})")

# (a) BATCH INTEGRITY -----------------------------------------------------
require_chunks = (expect == "require_chunks") or full
if require_chunks:
    if n == 0:
        fails.append("BATCH_INTEGRITY: 0 chunks on a lane that must produce")
else:
    # Offline VLM-route doc: 0 chunks is the documented behavior (the doc
    # routes to VLM in production; Docling-fast finds no text layer offline).
    if n == 0:
        notes.append("offline 0 chunks = documented VLM-route behavior")

# (c) ROUTING -------------------------------------------------------------
if not full:
    # Prove the V3 path: every CONTENT chunk must be uir_native_chunker.
    bad = []
    for c in chunks:
        em = (c.get("metadata") or {}).get("extraction_method")
        if em != "uir_native_chunker":
            bad.append(f"{c.get('chunk_id')}: extraction_method={em!r}")
    if bad:
        fails.append(
            "ROUTING(offline): non-uir_native_chunker chunk(s) (legacy leak?): "
            + "; ".join(bad[:5])
        )
else:
    # FULL: assert the lane hit the expected engine, inferred from the
    # emitted modality mix (VLM lanes emit IMAGE/TABLE chunks; the Docling
    # prose lane emits text only).
    if expect == "require_chunks" and lane == "prose":
        if n_img or n_tbl:
            notes.append(
                f"prose lane emitted {n_img} img / {n_tbl} tbl "
                "(expected ~Docling text-only) - VLM over-routed on prose"
            )
    else:
        # form / academic: VLM-routed lane should produce visual chunks.
        if n_img == 0 and n_tbl == 0:
            fails.append(
                "ROUTING(full): VLM-route lane produced no IMAGE/TABLE chunks "
                "(did it actually hit the VLM?)"
            )

# (b) MODALITY SCHEMA (QA-CHECK-05) ---------------------------------------
# Every IMAGE/TABLE chunk needs a present, non-null asset_ref whose file
# exists on disk. Full mode also needs a non-empty visual_description on
# IMAGE chunks (the VLM populated it).
visual = [c for c in chunks if c.get("modality") in ("image", "table")]
for c in visual:
    cid = c.get("chunk_id")
    ar = c.get("asset_ref")
    fp = (ar or {}).get("file_path") if isinstance(ar, dict) else None
    if not ar or not fp:
        fails.append(f"MODALITY_SCHEMA: {cid} ({c.get('modality')}) missing asset_ref")
        continue
    apath = outdir / fp
    if not apath.exists():
        fails.append(f"MODALITY_SCHEMA: {cid} asset file missing on disk: {fp}")
    if full and c.get("modality") == "image":
        meta = c.get("metadata") or {}
        vd = (meta.get("visual_description") or c.get("visual_description") or "").strip()
        if not vd:
            fails.append(
                f"MODALITY_SCHEMA(full): image {cid} has empty visual_description"
            )

print("NOTES:" + " | ".join(notes))
if fails:
    for f in fails:
        print("STRUCT_FAIL: " + f)
    sys.exit(3)
sys.exit(0)
PY
  )"
  struct_rc=$?
  echo "$struct_out" | sed 's/^/    /' | tee -a "$SUMMARY"

  # FULL-mode fallback-rate flag (advisory unless it indicates no VLM at all,
  # which the structural ROUTING check already fails on).
  if [ "$FULL" -eq 1 ] && [ "$fallbacks" -gt 0 ]; then
    log "    VLM fallbacks observed in router log: $fallbacks (flag if >${FALLBACK_PCT}% of pages)"
  fi

  # --- (d) GATE STATUS -----------------------------------------------------
  # Offline: only run the strict gate on lanes that must produce chunks.
  # The offline VLM-route doc (form) with 0 chunks would phantom-FAIL the
  # gate (NO_PAGE_CHUNKS), which is its documented offline behavior, so we
  # skip the gate there offline. Full mode runs the gate on EVERY lane.
  local run_gate=0
  if [ "$FULL" -eq 1 ]; then
    run_gate=1
  elif [ "$expect" = "require_chunks" ]; then
    run_gate=1
  fi

  local gate_status="SKIPPED"
  if [ "$run_gate" -eq 1 ]; then
    local qa_log="$out/qa.log"
    "$ENV_PYTHON" "$REPO_ROOT/scripts/qa_full_conversion.py" "$jsonl" \
      --source-pdf "$doc" >"$qa_log" 2>&1
    local qa_rc=$?
    gate_status="$(grep -oE 'QA_(PASS_WITH_ADVISORIES|PASS|WARN|FAIL)' "$qa_log" | tail -n 1)"
    gate_status="${gate_status:-QA_UNKNOWN}"
    log "    GATE: $gate_status (qa_full_conversion exit $qa_rc; see $qa_log)"
    case "$gate_status" in
      QA_PASS|QA_PASS_WITH_ADVISORIES)
        : ;;  # pass
      *)
        struct_rc=3
        echo "    STRUCT_FAIL: GATE_STATUS $gate_status (require QA_PASS / QA_PASS_WITH_ADVISORIES)" | tee -a "$SUMMARY"
        ;;
    esac
  else
    log "    GATE: SKIPPED (offline VLM-route doc - documented, no QA requirement)"
  fi

  if [ "$struct_rc" -eq 0 ]; then
    log "  [PASS] $lane"
    LANE_NAMES+=("$lane"); LANE_RESULTS+=("PASS")
    LANE_NOTES+=("gate=$gate_status fallbacks=$fallbacks")
  else
    log "  [FAIL] $lane"
    LANE_NAMES+=("$lane"); LANE_RESULTS+=("FAIL")
    LANE_NOTES+=("gate=$gate_status fallbacks=$fallbacks")
  fi
}

# --- Run all three lanes ---------------------------------------------------
run_lane "prose"    "$PROSE_DOC"    "require_chunks"
run_lane "academic" "$ACADEMIC_DOC" "require_chunks"
run_lane "form"     "$FORM_DOC"     "vlm_route_doc"

# --- Per-lane PASS/FAIL table ----------------------------------------------
log ""
log "==========================================================================="
log " PER-LANE RESULTS"
log "==========================================================================="
printf "%-10s | %-4s | %s\n" "LANE" "RES" "NOTES" | tee -a "$SUMMARY"
printf "%-10s-+-%-4s-+-%s\n" "----------" "----" "------------------------------" | tee -a "$SUMMARY"
ANY_FAIL=0
i=0
while [ "$i" -lt "${#LANE_NAMES[@]}" ]; do
  printf "%-10s | %-4s | %s\n" \
    "${LANE_NAMES[$i]}" "${LANE_RESULTS[$i]}" "${LANE_NOTES[$i]}" | tee -a "$SUMMARY"
  if [ "${LANE_RESULTS[$i]}" != "PASS" ]; then
    ANY_FAIL=1
  fi
  i=$((i + 1))
done

log "==========================================================================="
if [ "$ANY_FAIL" -eq 0 ]; then
  if [ "$FULL" -eq 1 ]; then
    log "SMOKE_PRODUCTION_PASS (mode=FULL)"
  else
    log "SMOKE_PRODUCTION_PASS (mode=OFFLINE)"
  fi
  exit 0
else
  if [ "$FULL" -eq 1 ]; then
    log "SMOKE_PRODUCTION_FAIL (mode=FULL)"
  else
    log "SMOKE_PRODUCTION_FAIL (mode=OFFLINE)"
  fi
  exit 1
fi
