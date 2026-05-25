#!/usr/bin/env bash
# v2.16 Phase 0 — post-ingest orchestration.
#
# Polls output/_v2_16_p0_logs/_master.log for "ALL DONE", then runs the
# autonomous-safe post-ingest steps in sequence. Production-index
# mutations (Phase 0 step 6.3 dense append + step 6.4 BM25 rebuild) are
# emitted as a user runbook at the end rather than auto-executed —
# those touch shared Qdrant state and warrant explicit user kick-off.
#
# Sequence:
#   1. Classify 7 new docs → output/_v2_16_p0_logs/classify_new7.{txt,json}
#   2. Auto-draft `docs/CORPUS_EXPANSION_2026-05-25_v2.16_p0.md` from
#      the classifier JSON.
#   3. Multi-profile smoke test (AGENT-VAL-01 hard tag-block gate).
#   4. CarOK Phase 4 re-extract with --force-table-vlm + new dedup.
#   5. Phase 1 validation re-run (post-Phase-3+4 retrieval-side numbers
#      on the current Qdrant; pre-Qdrant-mutation snapshot of v2.16
#      retrieval behavior).
#   6. Final full test suite.
#   7. Emit `output/_v2_16_p0_logs/_USER_RUNBOOK.md` with copy-paste
#      commands for step 6.1 Qdrant snapshot + 6.3 dense append + 6.4
#      BM25 rebuild + 6.5 re-run anti-drift test + v2.16.0 tag push.
#
# Each step is logged + tolerant: a failed step writes its log + the
# orchestration continues. The user can inspect failures via the
# per-step log files in output/_v2_16_p0_logs/.

set -uo pipefail

cd "$(dirname "$0")/.."
LOG_DIR=output/_v2_16_p0_logs
MASTER="$LOG_DIR/_master.log"
RUNBOOK="$LOG_DIR/_USER_RUNBOOK.md"

DOCS=(
  Bevestigingsmiddelen
  ATZ_Aerodynamik_Nutzfahrzeugen
  ATZ_ESF_Mercedes_2009
  Schwungradspeicher
  Eliasz_Zephyr_RTOS
  Grundlagen_Fahrzeug_Motorentechnik
  Digitale_Fotografie_Feb_2026
)

mark() { echo "=== $(date -Iseconds) $* ===" | tee -a "$LOG_DIR/_post_ingest.log"; }

mark "post-ingest: polling for ALL DONE"
until grep -q "ALL DONE" "$MASTER" 2>/dev/null; do
  sleep 30
done
mark "post-ingest: master ALL DONE detected"

# Verify each doc's ingestion.jsonl actually exists.
missing=()
paths=()
for d in "${DOCS[@]}"; do
  p="output/$d/ingestion.jsonl"
  if [[ -f "$p" ]]; then
    paths+=("$p")
  else
    missing+=("$d")
  fi
done
if (( ${#missing[@]} > 0 )); then
  mark "post-ingest: MISSING outputs: ${missing[*]}"
fi

# Step 1: classify the 7 new docs.
mark "STEP 1 classify"
python scripts/classify_corpus_v2_16_p0.py "${paths[@]}" \
    > "$LOG_DIR/classify_new7.txt" 2>&1 || true
python scripts/classify_corpus_v2_16_p0.py --json "${paths[@]}" \
    > "$LOG_DIR/classify_new7.json" 2>&1 || true

# Step 2: auto-draft inventory report from the classifier JSON.
mark "STEP 2 inventory report"
python scripts/_v2_16_p0_inventory_report.py \
    --classifier-json "$LOG_DIR/classify_new7.json" \
    --output "docs/CORPUS_EXPANSION_2026-05-25_v2.16_p0.md" \
    > "$LOG_DIR/inventory_report_stdout.txt" 2>&1 || true

# Step 3: multi-profile smoke (AGENT-VAL-01 hard gate).
mark "STEP 3 multi-profile smoke"
bash scripts/smoke_multiprofile.sh "output/_v2_16_p0_logs/smoke_multiprofile" \
    > "$LOG_DIR/smoke_multiprofile.txt" 2>&1 || true

# Step 4: CarOK Phase 4 re-extract with VLM dedup.
mark "STEP 4 CarOK Phase 4 re-extract"
CAROK_SRC="data/data_spreadsheet/CarOK voorraadtelling 2021-04.pdf"
if [[ -f "$CAROK_SRC" ]]; then
  mmrag-v2 process "$CAROK_SRC" \
      --output-dir "output/CarOK_v2_16_p4_dedup" \
      --batch-size 10 \
      > "$LOG_DIR/carok_v2_16_p4.log" 2>&1 || true
else
  mark "STEP 4 WARNING: CarOK source PDF not found at $CAROK_SRC"
fi

# Step 5: re-run Phase 1 validation (post-Phase-3+4 retrieval-side
# behavior on the *current* Qdrant — pre-mutation snapshot).
mark "STEP 5 Phase 1 validation re-run"
python scripts/run_personal_validation.py \
    --label "v2.16.0_pre_qdrant_mutation" \
    --output "docs/VALIDATION_REPORT_2026-05-25_v2.16.0_pre_qdrant_mutation.md" \
    > "$LOG_DIR/validation_v2_16_pre_mutation.txt" 2>&1 || true

# Step 6: final test suite.
mark "STEP 6 full test suite"
python -m pytest tests/ -q > "$LOG_DIR/pytest_final.txt" 2>&1 || true
tail -5 "$LOG_DIR/pytest_final.txt"

# Step 7: emit user runbook.
mark "STEP 7 emit user runbook"
cat > "$RUNBOOK" <<'RUNBOOK_EOF'
# v2.16.0 — User Runbook (Phase 0 step 6.3+6.4 + Final Tag Push)

The autonomous post-ingest orchestration completed. The Qdrant
production-index mutations + v2.16.0 tag push remain user-supervised.

## Status check first

```bash
# Verify all post-ingest steps landed clean:
cat output/_v2_16_p0_logs/_post_ingest.log

# Verify smoke gate PASSed (HARD tag-block):
grep "GATE" output/_v2_16_p0_logs/smoke_multiprofile.txt | head -20

# Confirm 7 new outputs exist:
for d in Bevestigingsmiddelen ATZ_Aerodynamik_Nutzfahrzeugen \
         ATZ_ESF_Mercedes_2009 Schwungradspeicher Eliasz_Zephyr_RTOS \
         Grundlagen_Fahrzeug_Motorentechnik Digitale_Fotografie_Feb_2026; do
  ls -la "output/$d/ingestion.jsonl" 2>&1 | head -1
done
```

## Step 6.1 — Qdrant snapshot (pre-mutation revert anchor)

```bash
# Create snapshot of mmrag_v2_8__qwen3_local. Record the returned
# snapshot name in the commit message of the upcoming append commit
# (PLAN_V2.16.md §3 Phase 0 step 6.1).
curl -sS -X POST http://localhost:6333/collections/mmrag_v2_8__qwen3_local/snapshots \
  | jq .
```

## Step 6.3 — Dense append (ingest 7 new docs into production collection)

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate mmrag-v2

for d in Bevestigingsmiddelen ATZ_Aerodynamik_Nutzfahrzeugen \
         ATZ_ESF_Mercedes_2009 Schwungradspeicher Eliasz_Zephyr_RTOS \
         Grundlagen_Fahrzeug_Motorentechnik Digitale_Fotografie_Feb_2026; do
  echo "=== ingest $d ==="
  python scripts/ingest_to_qdrant.py "output/$d/ingestion.jsonl" \
      --collection mmrag_v2_8__qwen3_local \
      --provider omlx \
      --batch-size 10
done
```

## Step 6.4 — BM25 sparse rebuild (against the renamed CANONICAL_DOCS)

```bash
# Rebuild the BM25 index against ALL 41 canonical docs (uses
# CANONICAL_DOCS from rebuild_mmrag_v2_8_for_rc1.py post-rename
# commit ed62429):
python scripts/build_bm25_index.py
python scripts/ingest_bm25_sparse.py
```

## Step 6.5 — Anti-drift bridge re-run (validates dense+sparse parallel mapping)

```bash
pytest tests/test_canonical_docs_consistency.py -v
pytest tests/test_personal_validation.py::test_canonical_docs_constant_loads -v
# Strict-gate refresh against the now-extended corpus:
bash scripts/smoke_multiprofile.sh output/smoke_multiprofile_v2_16_post_step6
```

## Step 6.6 — Revert procedure (only if any of 6.1-6.4 failed)

Per PLAN_V2.16.md §3 Phase 0 step 6.6:

```bash
# 1. Restore dense:
curl -sS -X POST http://localhost:6333/collections/mmrag_v2_8__qwen3_local/snapshots/recover \
  -H 'Content-Type: application/json' \
  -d '{"location":"<snapshot_name_from_step_6.1>"}'

# 2. Revert canonical-docs commit:
git revert ed62429

# 3. Re-run BM25 against restored 34-doc list:
python scripts/build_bm25_index.py
python scripts/ingest_bm25_sparse.py

# 4. Verify anti-drift test:
pytest tests/test_canonical_docs_consistency.py
```

## Phase 0 commit (after 6.3 + 6.4 succeed)

```bash
# Record snapshot ID + new chunk counts in the commit message:
git add -A
git commit -m "feat(v2.16 Phase 0 step 6.3-6.4): dense append + BM25 rebuild

Qdrant snapshot: <snapshot_name>
Dense collection mmrag_v2_8__qwen3_local: <pre_count> → <post_count> points
BM25 collection mmrag_v2_8__bm25_sparse: <pre_count> → <post_count> points

Verified:
- Anti-drift bridge test (tests/test_canonical_docs_consistency.py): PASS
- Multi-profile smoke: GATE_PASS + UNIVERSAL_PASS across all categories
- Phase 1 validation (post-step-6.3/4): see docs/VALIDATION_REPORT_2026-05-25_v2.16.0_post_qdrant_mutation.md
"
```

## v2.16.0 annotated tag + push

```bash
# Final retrieval-regression refresh (if production retrieval shape changed):
python scripts/retrieval_regression_v2_14.py

# Annotated tag with FINAL v2.X message:
git tag -a v2.16.0 -m "v2.16.0 — FINAL v2.X release (convergence cycle).

MM-Converter-V2 is feature-complete. Post-tag: only bug-fix patches
(v2.16.x); new features = re-charter as v3.0. v2.17 fires only on
PLAN_V2.16.md §7 safety-valve triggers.

Phase outcomes:
- SHIPPED: Phase 0 (corpus 34→41), Phase 1 (personal_importance overlay),
  Phase 3 (partial_code adjacency, inert on current corpus),
  Phase 4 (VLM-table IoU dedup).
- KILLed: Phase 5 (pre-flight retention undefined),
  Phase 6 (Phase 2 multi-factor verdict; 2nd dead lever),
  Phase 7 (default; no user opt-in).
- 8 carry-forward items closed; Item #11 (ColPali) declared v3.0 OUT-OF-SCOPE.
"

# Push to both remotes:
git push origin main
git push origin v2.16.0
git push github main
git push github v2.16.0
```

## Post-tag

- Update DECISIONS.md "v2.16 …" entries with final commit SHAs.
- Move PLAN_V2.16.md to docs/archive/plans/ (next predecessor reference).
- README banner already in place.

Per CLAUDE.md guardrails:
- Do NOT force-push to main.
- Verify each step's output before proceeding to the next.
- Revert procedure (§6.6) exists if any step fails.
RUNBOOK_EOF

mark "STEP 7 done — runbook at $RUNBOOK"
mark "ALL POST-INGEST STEPS DONE"
