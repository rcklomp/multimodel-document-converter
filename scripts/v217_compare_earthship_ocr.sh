#!/usr/bin/env bash
# v2.17 Item #9 reopen — Earthship multi-column OCR engine comparison.
#
# Background: Earthship_Vol1 is a 235-page scanned multi-column book.
# The v2.13 Phase 2 force_full_page_ocr fix improved chunk recovery
# (+73% text chunks) but Format quality on multi-column pages remained
# suppressed because the adapter silently hardcoded EasyOcrOptions
# regardless of CLI --ocr-engine. v2.17 ships the dispatch fix
# (engines/docling_adapter.py::_build_ocr_options).
#
# This script runs the same Earthship PDF through TWO OCR engines and
# emits a side-by-side report so the operator can decide whether to
# promote a non-EasyOCR default for scanned_degraded profiles.
#
# Wall time: ~20-30 min per engine on Apple Silicon. Skip --vision-provider
# to keep VLM out of the comparison axis.
#
# Usage:
#   bash scripts/v217_compare_earthship_ocr.sh
#
# Outputs:
#   output/Earthship_v217_easyocr/ingestion.jsonl  (baseline)
#   output/Earthship_v217_ocrmac/ingestion.jsonl   (challenger)
#   docs/V217_EARTHSHIP_OCR_COMPARISON.md          (decision evidence)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PDF="data/technical_manual/Earthship_Vol1_How to build your own.pdf"
if [[ ! -f "$PDF" ]]; then
  echo "FATAL: source PDF not found: $PDF" >&2
  exit 1
fi

echo "=== v2.17 Earthship OCR comparison ==="
echo "PDF: $PDF"
echo

for ENGINE in easyocr ocrmac; do
  OUT="output/Earthship_v217_${ENGINE}"
  if [[ -d "$OUT" ]]; then
    echo "[$ENGINE] output already exists at $OUT — skipping conversion"
    echo "         (delete the dir to force re-conversion)"
    continue
  fi
  echo "[$ENGINE] converting (~20-30 min wall time)..."
  conda run -n mmrag-v2 mmrag-v2 process "$PDF" \
    --output-dir "$OUT" \
    --vision-provider none \
    --enable-ocr \
    --ocr-engine "$ENGINE" \
    --batch-size 10 \
    || { echo "[$ENGINE] FAILED — see traceback above"; exit 1; }
  echo "[$ENGINE] done. Output: $OUT"
  echo
done

# Quick comparison report
echo "=== Comparison ==="
conda run -n mmrag-v2 python - <<'PYEOF'
import json
from collections import Counter
from pathlib import Path

def summarize(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"error": f"missing: {path}"}
    chunks = []
    with p.open() as f:
        for line in f:
            rec = json.loads(line.strip())
            if rec.get("chunk_id"):
                chunks.append(rec)
    methods = Counter(c["metadata"].get("extraction_method", "") for c in chunks)
    pages = set(c["metadata"].get("page_number") for c in chunks)
    text_chunks = [c for c in chunks if c["modality"] == "text"]
    short = sum(1 for c in text_chunks if len(c.get("content", "")) < 100)
    return {
        "total_chunks": len(chunks),
        "text_chunks": len(text_chunks),
        "pages_covered": len(pages),
        "short_text_chunks_<100": short,
        "methods": dict(methods.most_common(5)),
    }

easy = summarize("output/Earthship_v217_easyocr/ingestion.jsonl")
mac  = summarize("output/Earthship_v217_ocrmac/ingestion.jsonl")
print(json.dumps({"easyocr": easy, "ocrmac": mac}, indent=2))
PYEOF
