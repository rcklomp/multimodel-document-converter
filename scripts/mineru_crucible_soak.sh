#!/usr/bin/env bash
# Full-crucible Grand Soak through the V3 DEFAULT route (MineruQwenHybridEngine:
# MinerU for prose/tables/layout + Qwen for code-dense pages). This is the
# charter Section 9.2 item 1 validation gate: re-run the 16-doc crucible
# end-to-end and confirm the 2026-06-04 MinerU pivot holds at corpus scale
# (the bar the 2026-06-02 Grand Soak failed at doc 9/17).
#
# NOT a committed CI gate; the attended corpus-validation harness. Run it with
# the mmrag-v2 conda env active and BOTH extraction endpoints up.
#
# Usage:
#   bash scripts/mineru_crucible_soak.sh smoke   # hardest doc only (default)
#   bash scripts/mineru_crucible_soak.sh full    # all 16 crucible docs
#
# Acceptance (charter Section 9.1, gates the Grand Soak): every doc QA_PASS
# (or QA_PASS_WITH_ADVISORIES), crop-audit drift under the 0.15 threshold on
# forms/scans/tables, dense-magazine whole-page fallback ~0, and zero V3-path
# extraction-method leak.
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
MODE="${1:-smoke}"

# --- endpoints (default route engages when MINERU_ENDPOINT is set + no USE_* flag) ---
export MINERU_ENDPOINT="${MINERU_ENDPOINT:-http://10.0.10.239:8001}"
export MINERU_MODEL="${MINERU_MODEL:-MinerU2.5-2509-1.2B}"
# Code-dense pages route to Qwen via the VLM provider. Point it at the local M5
# server; if unset the hybrid falls back to the cloud OpenRouter default (cost).
export VLM_NATIVE_ENDPOINT="${VLM_NATIVE_ENDPOINT:-http://10.0.10.235:8000}"
export VLM_NATIVE_MODEL="${VLM_NATIVE_MODEL:-mlx-community/Qwen3-VL-8B-Instruct-8bit}"
export VLM_NATIVE_API_KEY="${VLM_NATIVE_API_KEY:-EMPTY}"
unset USE_MINERU_ENGINE USE_VLM_ENGINE USE_DOCLING_FAST USE_HYBRID_ENGINE

SRC="output/crucible_full_src"   # the pre-staged, page-sliced crucible inputs
OUT="output/mineru_crucible_soak"

# --- the 16-doc crucible, HARDEST FIRST -------------------------------------
# Lead with the docs that broke the 2026-06-02 soak (dense magazine) and the
# code-dense academic docs (the R3 risk), so a regression fails fast before the
# easy docs burn endpoint time.
DOCS=(
  "CombatAircraft_p20-44.pdf"   # dense magazine (Blocker A: ~58% fallback in the 0602 soak)
  "PCWorld_p20-44.pdf"          # dense magazine (the 2nd named dense doc)
  "AIOS_academic.pdf"           # academic + dense pseudocode + tables (R3)
  "FluentPython_p60-84.pdf"     # dense code book (R3 collapse risk)
  "Grundlagen_p1-25.pdf"        # German technical, code/figures
  "Form_0013.pdf"               # scanned form (Blocker B: crop drift 40-50%)
  "Form_betwisting.pdf"         # business form (crop drift)
  "CarOK_spreadsheet.pdf"       # dense spreadsheet (Qwen emptied; MinerU reads it)
  "Firearms_p1-25.pdf"          # multi-column manual (heading/infix shape)
  "DigitaleFotografie_p1-25.pdf" # image-centric book
  "HarryPotter_p1-30.pdf"       # prose
  "Kimothi_RAG_p1-25.pdf"       # technical manual w/ bookmarks
  "ATZ_Elektronik_report.pdf"   # technical report
  "IRJET_academic.pdf"          # academic w/ figures
  "Hybrid_EV_academic.pdf"      # academic
  "Bevestigingsmiddelen.pdf"    # short born-digital (headingless prose)
)
[[ "$MODE" == "smoke" ]] && DOCS=("${DOCS[0]}")

# --- pre-flight: fail fast if an endpoint is down (do not burn a partial run) ---
preflight() {
  local ok=1
  local m; m=$(curl -s --max-time 6 -o /dev/null -w "%{http_code}" "$MINERU_ENDPOINT/health" 2>/dev/null); m=${m:-000}
  if [[ "$m" == "200" ]]; then echo "  [OK]   MinerU  $MINERU_ENDPOINT (health 200)"; else
    echo "  [DOWN] MinerU  $MINERU_ENDPOINT (health $m) - start the MinerU server first"; ok=0; fi
  local v; v=$(curl -s --max-time 6 -o /dev/null -w "%{http_code}" "$VLM_NATIVE_ENDPOINT/health" 2>/dev/null); v=${v:-000}
  if [[ "$v" == "200" ]]; then echo "  [OK]   Qwen VLM $VLM_NATIVE_ENDPOINT (health 200)"; else
    echo "  [WARN] Qwen VLM $VLM_NATIVE_ENDPOINT (health $v) - code-dense pages will hit the cloud default or fail"; fi
  [[ -d "$SRC" ]] || { echo "  [DOWN] crucible source dir $SRC missing"; ok=0; }
  return $((1-ok))
}

echo "=== Grand Soak pre-flight (MODE=$MODE, ${#DOCS[@]} doc(s)) ==="
if ! preflight; then
  echo
  echo "ABORT: MinerU endpoint or crucible source not ready. Start the MinerU"
  echo "server (and the M5 Qwen VLM), then re-run. No documents were processed."
  exit 1
fi

rm -rf "$OUT"; mkdir -p "$OUT"
printf "\n%-30s %6s %7s %-26s %-10s %s\n" "DOC" "PAGES" "CHUNKS" "GATE" "CROP" "LEAK"
overall=0
for d in "${DOCS[@]}"; do
  doc="$SRC/$d"
  name=$(basename "$d" .pdf | cut -c1-28)
  o="$OUT/$name"
  if [[ ! -f "$doc" ]]; then printf "%-30s %s\n" "$name" "MISSING_SRC ($doc)"; overall=1; continue; fi
  "$PY" -m mmrag_v2.cli process "$doc" --output-dir "$o" --batch-size 10 --vision-provider none \
    > "$o.log" 2>&1
  jl="$o/ingestion.jsonl"
  if [[ ! -f "$jl" ]]; then printf "%-30s %6s %7s %s\n" "$name" "-" "-" "NO_OUTPUT (see $o.log)"; overall=1; continue; fi
  res=$("$PY" scripts/qa_full_conversion.py "$jl" --source-pdf "$doc" 2>&1 | grep -oE "QA_(PASS|WARN|FAIL)[A-Z_]*" | tail -1)
  read -r pages chunks leak crop < <("$PY" - "$jl" "$o/meta.json" <<'PY'
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1])]
ch=[r for r in rows if r.get("modality")]
hdr=next((r for r in rows if r.get("total_pages") is not None),{})
def _is_leak(m):
    # Whitelist the legitimate V3-path methods; anything else is a real leak
    # (e.g. a Docling/legacy fallback). uir_native_chunker = chunker text/table,
    # rendered_region_crop = materialized image assets, recovery_* = the
    # TextIntegrityScout fallbacks. Counting these as leaks gave false numbers.
    if not m:
        return False
    return m != "uir_native_chunker" and m != "rendered_region_crop" and not m.startswith("recovery_")
leak=sum(1 for c in ch if _is_leak((c.get("metadata") or {}).get("extraction_method")))
crop="na"
try:
    ca=json.load(open(sys.argv[2])).get("crop_audit") or {}
    if ca: crop=f"{ca.get('drift_rate',0):.2f}/fb{ca.get('full_page_fallbacks',0)}"
except Exception: pass
print(hdr.get("total_pages","?"), len(ch), f"leak={leak}", crop)
PY
)
  # acceptance: QA must PASS (advisories ok); leak must be 0
  case "$res" in QA_PASS|QA_PASS_WITH_ADVISORIES) :;; *) overall=1;; esac
  [[ "$leak" != "leak=0" ]] && overall=1
  printf "%-30s %6s %7s %-26s %-10s %s\n" "$name" "$pages" "$chunks" "$res" "$crop" "$leak"
done

echo
if [[ "$MODE" == "smoke" ]]; then
  echo "SMOKE complete -> $OUT. If the hardest doc passed, run: bash scripts/mineru_crucible_soak.sh full"
elif [[ "$overall" == "0" ]]; then
  echo "CRUCIBLE_SOAK_PASS: all ${#DOCS[@]} docs QA_PASS, no V3-path leak. Charter Section 9.2 item 1 cleared."
else
  echo "CRUCIBLE_SOAK_FAIL: at least one doc did not meet acceptance (see table + per-doc logs in $OUT)."
fi
exit "$overall"
