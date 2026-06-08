#!/usr/bin/env bash
# Full-multimodal crucible validation: extraction soak -> M5 Qwen image
# enrichment -> re-validate. Stage 1 reuses the extraction soak harness
# (--vision-provider none); Stage 2 describes every image via the LOCAL M5
# Qwen3-VL through the env-overridable enrich_image_chunks_v29.py lane; Stage 3
# re-runs the strict gate so IMAGE_NO_VLM is cleared to a clean QA_PASS.
#
# Untracked attended harness (like mineru_crucible_soak.sh). Run with the
# mmrag-v2 conda env active and BOTH extraction endpoints + the M5 VLM up.
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
SRC=output/crucible_full_src
OUT=output/mineru_crucible_soak

echo "########## STAGE 1: extraction soak ##########"
bash scripts/mineru_crucible_soak.sh full
soak_rc=$?
echo "soak exit=$soak_rc"

echo "########## STAGE 2: M5 Qwen image enrichment ##########"
export MMRAG_ENRICH_PROVIDER=openai
export MMRAG_ENRICH_MODEL=mlx-community/Qwen3-VL-8B-Instruct-8bit
export MMRAG_ENRICH_BASE_URL=http://10.0.10.235:8000/v1
export MMRAG_REFINER_API_KEY=dummy
for d in "$SRC"/*.pdf; do
  name=$(basename "$d" .pdf | cut -c1-28)
  jl="$OUT/$name/ingestion.jsonl"
  [ -f "$jl" ] || { echo "  SKIP $name (no output)"; continue; }
  echo "  --- enriching $name ---"
  "$PY" scripts/enrich_image_chunks_v29.py "$jl" 2>&1 | grep -E "enriched|hard fallback|image chunks" | sed 's/^/    /'
done

echo "########## STAGE 3: re-validate (post-enrichment) ##########"
pass_clean=0; pass_adv=0; fail=0
printf "%-30s %s\n" DOC GATE
for d in "$SRC"/*.pdf; do
  name=$(basename "$d" .pdf | cut -c1-28)
  jl="$OUT/$name/ingestion.jsonl"
  [ -f "$jl" ] || { printf "%-30s %s\n" "$name" "NO_OUTPUT"; fail=$((fail+1)); continue; }
  res=$("$PY" scripts/qa_full_conversion.py "$jl" --source-pdf "$d" 2>&1 | grep -oE "QA_(PASS|WARN|FAIL)[A-Z_]*" | tail -1)
  printf "%-30s %s\n" "$name" "$res"
  case "$res" in
    QA_PASS) pass_clean=$((pass_clean+1));;
    QA_PASS_WITH_ADVISORIES) pass_adv=$((pass_adv+1));;
    *) fail=$((fail+1));;
  esac
done
echo "-----"
echo "FINAL: clean_QA_PASS=$pass_clean  pass_with_advisories=$pass_adv  fail=$fail  / 16"
echo "CRUCIBLE_VLM_PIPELINE_DONE"
