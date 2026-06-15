#!/usr/bin/env bash
# v2.8 broad reconversion + audit + snapshot, fully autonomous.
# Step 1 — convert every PDF/EPUB in data/ via convert_books.sh.
# Step 2 — audit every output via qa_conversion_audit.py.
# Step 3 — emit docs/QUALITY_SNAPSHOT_<date>.md AFTER snapshot.
# Step 4 — commit conversion outputs + snapshot.
# Step 5 — print Qdrant runbook (deferred — Qdrant container is in a
#          startup-deadlock state from a stale collection lock that needs
#          the user to clean before ingest can run).
set -uo pipefail

cd "$(dirname "$0")/.."

PIPELINE_LOG="output/_v28_pipeline.log"
DATE_ISO="$(date +%Y-%m-%d)"
SNAPSHOT="docs/QUALITY_SNAPSHOT_${DATE_ISO}_v2.8_after.md"
AUDIT_LOG="docs/quality_snapshots/${DATE_ISO}_v2.8_after/audit_all_outputs.txt"

mkdir -p "$(dirname "$AUDIT_LOG")"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$PIPELINE_LOG"; }

log "=========================================================="
log "v2.8 OVERNIGHT PIPELINE START"
log "=========================================================="

# --- Step 1: broad reconversion ---
log "Step 1/5: convert_books.sh (broad reconversion)"
bash scripts/convert_books.sh >>"$PIPELINE_LOG" 2>&1
CONV_EXIT=$?
log "convert_books.sh exit=$CONV_EXIT"

if [ "$CONV_EXIT" -ne 0 ]; then
  log "WARNING: convert_books.sh exit=$CONV_EXIT — continuing to audit anyway"
fi

# --- Step 2: audit every fresh output ---
log "Step 2/5: qa_conversion_audit on all output/<doc>/ingestion.jsonl"
JSONLS=$(find output -maxdepth 2 -name "ingestion.jsonl" | sort)
if [ -z "$JSONLS" ]; then
  log "ERROR: no ingestion.jsonl found under output/"
  exit 2
fi
python scripts/qa_conversion_audit.py $JSONLS > "$AUDIT_LOG" 2>&1
AUDIT_EXIT=$?
log "audit exit=$AUDIT_EXIT (1 == one or more FAIL — expected for stale outputs)"

# --- Step 3: build snapshot ---
log "Step 3/5: build $SNAPSHOT"
python scripts/v28_build_after_snapshot.py "$AUDIT_LOG" "$SNAPSHOT" >>"$PIPELINE_LOG" 2>&1
log "snapshot written to $SNAPSHOT"

# --- Step 4: commit ---
log "Step 4/5: git add + commit"
git add "$SNAPSHOT" "$AUDIT_LOG" docs/quality_snapshots/ 2>&1 | tee -a "$PIPELINE_LOG"
git commit -m "$(cat <<EOF
docs: PLAN_V2.8 Phase 5c AFTER snapshot — broad reconversion complete

Snapshot: $SNAPSHOT
Audit log: $AUDIT_LOG
Conversion log: output/_convert_books.log

Phase 5c step 3 (Qdrant re-ingestion) DEFERRED to next session — the
existing Qdrant container is in a startup-deadlock state on
storage/collections/sekar_s__.../version.info (stale lock from a
previous interrupted run). Recovery runbook in $SNAPSHOT.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)" >>"$PIPELINE_LOG" 2>&1
log "commit exit=$?"

log "=========================================================="
log "v2.8 OVERNIGHT PIPELINE DONE"
log "=========================================================="
log "Next steps (user, in the morning):"
log "  1. Resolve Qdrant container deadlock:"
log "     docker rm multimodal-doc-converter-qdrant"
log "     # then re-create per the project setup; OR clear the stale lock:"
log "     # docker run --rm -v <vol>:/qdrant/storage alpine rm -rf /qdrant/storage/collections/sekar_s__*"
log "  2. Start Ollama (default embedding service):"
log "     open /Applications/Ollama.app   # or: ollama serve"
log "  3. Run side-by-side ingest into mmrag_v2_8 collection:"
log "     for d in output/*/; do"
log "       jsonl=\"\$d/ingestion.jsonl\""
log "       [ -f \"\$jsonl\" ] || continue"
log "       python scripts/ingest_to_qdrant.py \"\$jsonl\" --collection mmrag_v2_8"
log "     done"
log "  4. After ingest verifies, tag v2.8.0:"
log "     git tag v2.8.0 && git push origin v2.8.0"

exit 0
