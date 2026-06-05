#!/usr/bin/env bash
# MinerU corpus soak: run a cross-category sample through the V3 DEFAULT route
# (MINERU_ENDPOINT set, NO USE_* flag -> exercises the new MinerU default), then
# strict-gate each output. NOT a committed gate; an ad-hoc validation harness.
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"  # run with the mmrag-v2 conda env active
export MINERU_ENDPOINT="${MINERU_ENDPOINT:-http://10.0.10.239:8001}"
export MINERU_MODEL="${MINERU_MODEL:-MinerU2.5-2509-1.2B}"
unset USE_MINERU_ENGINE USE_VLM_ENGINE USE_DOCLING_FAST USE_HYBRID_ENGINE
OUT=output/mineru_corpus_soak
rm -rf "$OUT"; mkdir -p "$OUT"

DOCS=(
  "data/business_form/betwistingsformulier_aankoop_niet_ontvangen.pdf"
  "data/raw/Bevestigingsmiddelen.pdf"
  "data/academic_journal/Recent_Trends_in_Transportation_Technolo.pdf"
  "data/technical_report/ATZ.Elektronik.-.Effizientere.Software.Entwicklung.GERMAN.RETAiL.eBOOk-PDFWriters.pdf"
  "data/academic_journal/IRJET_Modeling_of_Solar_PV_system_under.pdf"
  "data/data_spreadsheet/CarOK voorraadtelling 2021-04.pdf"
  "data/academic_journal/Hybrid_electric_vehicles_and_their_challenges.pdf"
)

printf "%-46s %6s %7s %s\n" "DOC" "PAGES" "CHUNKS" "GATE"
for doc in "${DOCS[@]}"; do
  name=$(basename "$doc" .pdf | cut -c1-44)
  o="$OUT/$(echo "$name" | tr ' /' '__')"
  "$PY" -m mmrag_v2.cli process "$doc" --output-dir "$o" --batch-size 10 --vision-provider none \
    > "$o.log" 2>&1
  jl="$o/ingestion.jsonl"
  if [[ -f "$jl" ]]; then
    res=$("$PY" scripts/qa_full_conversion.py "$jl" --source-pdf "$doc" 2>&1 | grep -oE "QA_(PASS|WARN|FAIL)[^|]*" | tail -1)
    stats=$("$PY" - "$jl" <<'PY'
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1])]
ch=[r for r in rows if r.get("modality")]
hdr=next((r for r in rows if r.get("total_pages") is not None), {})
nonuir=sum(1 for c in ch if (c.get("metadata") or {}).get("extraction_method") not in ("uir_native_chunker", None) and (c.get("metadata") or {}).get("extraction_method"))
print(f'{hdr.get("total_pages","?")} {len(ch)} leak={nonuir}')
PY
)
    pages=$(echo "$stats" | awk '{print $1}'); chunks=$(echo "$stats" | awk '{print $2}'); leak=$(echo "$stats" | awk '{print $3}')
    printf "%-46s %6s %7s %s %s\n" "$name" "$pages" "$chunks" "$res" "$leak"
  else
    printf "%-46s %6s %7s %s\n" "$name" "-" "-" "NO_OUTPUT (see $o.log)"
  fi
done
echo "DONE -> $OUT"
