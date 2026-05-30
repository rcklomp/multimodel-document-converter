#!/usr/bin/env bash
# V3 Grand Soak — post-batch pipeline.
#
# Runs after scripts/v3_batch_ingest.py has populated output/v3_baselines/.
# Does: translate → canonical-layout → omlx embed + Qdrant index → soak vs V3 →
# soak vs V2.16 baseline. All work is idempotent on re-run.
#
# Required env (same as the batch ingest):
#     MLX_API_KEY (omlx-server auth)
#     DASHSCOPE_API_KEY (NOT required when --judge-provider=vllm)
#
# Outputs:
#     output/v3_baselines_v2shape/        — V3 chunks translated to V2 schema
#     output/v3_canonical/                — canonical-name layout for soak
#     Qdrant collection mmrag_v3__qwen3_local
#     output/v3_soak/work.v3.jsonl + report.v3.md
#     output/v3_soak/work.v216.jsonl + report.v216.md

set -euo pipefail

PY=${PY:-$HOME/miniforge3/envs/mmrag-v2/bin/python}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

V3_COLLECTION=${V3_COLLECTION:-mmrag_v3__qwen3_local}
SOAK_OUT=${SOAK_OUT:-output/v3_soak}
N_CHUNKS=${N_CHUNKS:-50}
SEED=${SEED:-7}

mkdir -p "$SOAK_OUT"

echo "=== Step A: V3 → V2-shape JSONL translation ==="
$PY scripts/v3_to_v2_jsonl.py \
    --in-dir output/v3_baselines \
    --out-dir output/v3_baselines_v2shape

echo
echo "=== Step B: canonical-name layout (for soak sample stage) ==="
# build_v3_canonical_layout.py exits 1 when any canonical doc is unmatched.
# With a partial corpus that is EXPECTED (38 canonical names vs N extracted
# docs) and not a failure — the matched docs are laid out fine. Don't let the
# informational non-zero trip `set -e` and abort indexing of the matched set.
$PY scripts/build_v3_canonical_layout.py \
    --v3-shape-dir output/v3_baselines_v2shape \
    --out-dir output/v3_canonical \
    || echo "[pipeline] canonical-layout reported unmatched docs (expected, partial corpus) — continuing"

echo
echo "=== Step C: embed + index V3 chunks into ${V3_COLLECTION} ==="
RECREATE=--recreate
for jsonl in output/v3_canonical/*/ingestion.jsonl; do
    [ -f "$jsonl" ] || continue
    doc_name=$(basename "$(dirname "$jsonl")")
    echo "  indexing: ${doc_name}"
    $PY scripts/ingest_to_qdrant.py "$jsonl" \
        --collection "$V3_COLLECTION" \
        --provider omlx \
        --no-contextual \
        $RECREATE
    RECREATE=
done

echo
echo "=== Step D: V3 soak — GX10 judge ==="
$PY scripts/synthetic_soak.py \
    --stage all \
    --provider omlx \
    --collection "$V3_COLLECTION" \
    --rerank-backend omlx \
    --docs-root output/v3_canonical \
    --judge-provider vllm \
    --gen-provider vllm \
    --n-chunks "$N_CHUNKS" \
    --seed "$SEED" \
    --work-path "$SOAK_OUT/work.v3.jsonl" \
    --report-path "$SOAK_OUT/report.v3.md"

echo
echo "=== Step E: V2.16 baseline soak — same judge for apples-to-apples ==="
$PY scripts/synthetic_soak.py \
    --stage all \
    --provider omlx \
    --collection mmrag_v2_8__qwen3_local \
    --rerank-backend omlx \
    --judge-provider vllm \
    --gen-provider vllm \
    --n-chunks "$N_CHUNKS" \
    --seed "$SEED" \
    --work-path "$SOAK_OUT/work.v216.jsonl" \
    --report-path "$SOAK_OUT/report.v216.md"

echo
echo "=== V3 post-batch pipeline complete ==="
echo "V3 report:  $SOAK_OUT/report.v3.md"
echo "V216 report: $SOAK_OUT/report.v216.md"
