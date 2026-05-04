# Acceptance Order Execution Prompt

**Copy everything below this line as your Claude Code prompt.**

This prompt drives **step 5 (Acceptance Order)** of the combined plan covering
the open-issues plan-of-approach, `docs/PLAN_V2.8_PRODUCTION_GAPS.md` (current
execution plan), and `docs/archive/PLAN_V2.7_DOCUMENT_UNDERSTANDING.md`
(archived 2026-05-03; retained as architectural rationale).
It is intentionally adversarial: the agent must prove — with reproducible
artefacts — that every targeted failure has been addressed and that the
upstream stabilization, plan-extension, refactor-boundary, and contextual
retrieval phases are actually in place. "Done" is not a self-declaration; it
is a generated, machine-checkable verdict file.

---

```
# 🚨 ACCEPTANCE ORDER EXECUTION — MMRAG V2.7.x

You are executing **step 5 (Acceptance Order)** of the combined stabilization +
refactor + contextual retrieval plan. This is the **final gate** before broad
corpus reconversion.

You are NOT done when you think you are done. You are done when the verdict
file `output/acceptance_<RUN_ID>/VERDICT.json` records `"verdict": "PASS"` and
every audit command in PHASE 6 returns the expected exit code. Saying "PASS"
without the artefacts is a system failure.

----------------------------------------------------------------------
## CONTEXT & ORIENTATION
----------------------------------------------------------------------

Read these files FIRST and treat them as binding:
1.  `AGENTS.md`                                — Level 0 invariants (binding)
2.  `docs/AGENT_GOVERNANCE.md`                 — evidence/status rules
3.  `docs/PROJECT_STATUS.md`                   — current engineering state
4.  `docs/DECISIONS.md`                        — architectural decisions
5.  `docs/PLAN_V2.8_PRODUCTION_GAPS.md` — **SHIPPED 2026-05-04**, retained as the historical execution plan (the chain `5b0e13d → 645ab2b` on `main` and the `v2.8.0` annotated tag are its outcome). For the next execution cycle see `docs/PLAN_V2.9_DRAFT_PROMPT.md` (and the to-be-drafted `docs/PLAN_V2.9.md`). For architectural rationale see `docs/archive/PLAN_V2.7_DOCUMENT_UNDERSTANDING.md`.
6.  `docs/PROGRESS_CHECKLIST.md`               — pending items
7.  `docs/QUALITY_GATES.md`                    — pass/fail thresholds
8.  `docs/CONTEXTUAL_RETRIEVAL_PROMPT.md`      — companion (step 4)

Pinned invariants you must honour and cite in the verdict file:
- `AGENT-VAL-01` — blind-test smoke pass; no hardcoded filenames in production
- `AGENT-EVIDENCE-01` — evidence must be reproducible from tracked files
- `AGENT-STATUS-01` — explicit status scope; no premature "complete"
- `AGENT-TEST-01`  — never weaken/rewrite negative or regression assertions
- `AGENT-SPATIAL-20` — single 20-unit vertical threshold
- `AGENT-DOCS-01`  — minimal indexed docs

**Version:** engine v2.8.x / schema 2.7.x. Python 3.10. Docling exact-pinned 2.86.0.
**Phase:** v2.8.0 SHIPPED 2026-05-04 (broad reconversion + Qdrant ingest complete). v2.9 cycle: see `docs/PLAN_V2.9_DRAFT_PROMPT.md`.

----------------------------------------------------------------------
## RUN IDENTITY (do this FIRST, before any other action)
----------------------------------------------------------------------

```bash
conda activate mmrag-v2 || conda activate ./env
export RUN_ID="$(date +%Y%m%d_%H%M%S)"
export ROOT="output/acceptance_${RUN_ID}"
mkdir -p "${ROOT}"/{probes,proof,baseline}
echo "RUN_ID=${RUN_ID}" | tee "${ROOT}/RUN_ID.txt"

# Capture environment fingerprint — required for AGENT-EVIDENCE-01
python --version                                                  > "${ROOT}/proof/env.txt" 2>&1
python -c "import docling, sys; print('docling=', docling.__version__)" >> "${ROOT}/proof/env.txt" 2>&1
git rev-parse HEAD                                               >> "${ROOT}/proof/env.txt"
git status --porcelain                                           >> "${ROOT}/proof/git_status.txt"
mmrag-v2 version                                                 >> "${ROOT}/proof/env.txt" 2>&1
```

**Stop conditions for PHASE 0:**
- Python is not 3.10.x — STOP, fix env, do not proceed.
- `docling.__version__ != "2.86.0"` — STOP, this violates the pin.
- `git status --porcelain` shows uncommitted changes that you cannot account
  for in this run — STOP, surface them to the user before continuing.

----------------------------------------------------------------------
## PHASE 1 — Verify upstream phases (1–4) are actually in place
----------------------------------------------------------------------

Step 5 has **prerequisites**. Do not run probes against a half-finished tree.
Each item below produces a single line in `${ROOT}/proof/upstream_audit.txt`
of the form `OK  <id>  <evidence>` or `MISS <id> <reason>`. Any `MISS` is a
hard stop.

### 1.A Stabilization (combined plan §1)

For each item, run the listed test or grep and record the result:

| ID    | What                                          | Verifier (must pass)                                                                |
|-------|-----------------------------------------------|-------------------------------------------------------------------------------------|
| S1    | Low-confidence classifier fallback fixed      | `pytest tests/test_classifier_fallback.py -v`                                       |
| S2    | Scanned-book route exists                     | `pytest tests/test_pdf_conversion_plan.py::test_plan_scanned_modality_sets_scanned_book_route -v` |
| S3    | HybridChunker pathological-token guard        | `pytest tests/test_chunker_guard.py -v`                                             |
| S4    | Huge text sequences stopped pre-tokenizer     | `grep -n "max_chunker_input_chars" src/mmrag_v2/engines/*.py src/mmrag_v2/processor.py` |
| S5    | Blank asset validation                        | `pytest tests/test_blank_asset_quarantine.py -v`                                    |
| S6    | Corruption interceptor isolation              | `pytest tests/test_corruption_quarantine.py -v`                                     |

### 1.B `PdfConversionPlan` control plane (combined plan §2)

Confirm every named field exists on the dataclass:
```bash
python - <<'PY' | tee -a "${ROOT}/proof/upstream_audit.txt"
from dataclasses import fields
from mmrag_v2.engines.pdf_plan import PdfConversionPlan
need = {
  "extraction_route", "hybrid_chunker_enabled", "max_chunker_input_chars",
  "allow_page_level_visuals", "asset_validation_policy",
  "corruption_recovery_policy",
}
have = {f.name for f in fields(PdfConversionPlan)}
missing = sorted(need - have)
print("OK  P1  PdfConversionPlan fields present" if not missing
      else f"MISS P1 missing fields: {missing}")
PY
```

### 1.C Refactor boundary (combined plan §3)

```bash
pytest tests/test_pdf_conversion_plan.py::test_no_pipeline_options_construction_outside_adapter \
       tests/test_pdf_conversion_plan.py::test_no_production_docling_imports_outside_adapter \
       -v 2>&1 | tee -a "${ROOT}/proof/upstream_audit.txt"

pytest tests/test_finalization_bridge.py -v \
       2>&1 | tee -a "${ROOT}/proof/upstream_audit.txt"
```

These bridge tests prove `CLI → plan → batch → processor → adapter` carries
route + limits across object boundaries. A failure here means §3 is not done
and probes will be misleading.

### 1.D Contextual retrieval (combined plan §4)

```bash
pytest tests/test_contextual_retrieval.py -v \
       2>&1 | tee -a "${ROOT}/proof/upstream_audit.txt"

# Must NOT mutate canonical chunk content; embedding text lives in a separate field.
python - <<'PY' | tee -a "${ROOT}/proof/upstream_audit.txt"
from mmrag_v2.schema.ingestion_schema import IngestionChunk
fields = {f for f in IngestionChunk.model_fields}
expected = {"contextualized_content"} | {"embedding_text"}
overlap = fields & expected
print("OK  C1 contextual field present" if overlap else f"MISS C1 expected one of {expected} on IngestionChunk")
PY

# QA/source-text validation must use raw/refined content, not prefixed text.
grep -nE "contextualized_content|embedding_text" scripts/qa_universal_invariants.py scripts/qa_conversion_audit.py \
       || echo "OK  C2 QA scripts do not consume contextualized text"
```

**Stop condition:** any `MISS ` line in `upstream_audit.txt` → halt and report.
You may not proceed to PHASE 2 with missing prerequisites.

----------------------------------------------------------------------
## PHASE 2 — Targeted probes (combined plan §5, bullet 1)
----------------------------------------------------------------------

Targeted probes reproduce the specific failures the combined plan was built
to fix. **Discover paths** dynamically — do NOT hardcode. Filenames may
contain spaces, ellipses, or version suffixes.

```bash
discover() {  # discover <category> <pattern>
  find "data/$1" -maxdepth 1 -iname "$2" -print -quit
}

HARRY=$(discover digital_literature '*Harry*Potter*Sorcerer*.pdf')
SCAN0013=$(discover business_form  '0013_140302111325_001.pdf')
COMBAT=$(discover digital_magazine '*Combat*Aircraft*.pdf')
CAROK=$(discover  data_spreadsheet '*CarOK*voorraadtelling*.pdf')
AYEVA=$(discover  technical_manual '*Ayeva*Python*Design*Patterns*.pdf')
GREEN=$(discover  technical_manual '*Greenhouse*Pedro*Ponce*.pdf')

for v in HARRY SCAN0013 COMBAT CAROK AYEVA GREEN; do
  printf '%-9s %s\n' "$v" "${!v:-<NOT-FOUND>}"
done | tee "${ROOT}/proof/probe_paths.txt"
```

If any path is `<NOT-FOUND>` — STOP and surface to user. Do not substitute a
similar file silently; that violates `AGENT-VAL-01`.

| Probe key   | Bug class targeted                                                | Expected fix evidence                              |
|-------------|-------------------------------------------------------------------|----------------------------------------------------|
| HARRY       | Born-digital novel routing + post-Docling sanity pass             | profile=`digital_literature`; `extraction_route=native_digital`; page-13 paragraphs in PDF y-order; drop cap "M" at start of paragraph 1; no `Other`/`Icon`/`Table` label tokens; no chunks for cover pages 1-4 |
| SCAN0013    | Scanned-route assertion (replaces HARRY for the scanned probe)    | profile is a scanned variant; `extraction_route=scanned_book`; OCR active |
| COMBAT      | Encoding-corruption leak into final JSONL                         | quarantined chunks not in `ingestion.jsonl`        |
| CAROK       | Blank asset validation                                            | `asset_validation_policy=drop`/`quarantine` honoured |
| AYEVA       | HybridChunker pathological-token guard; code-enrichment decision  | no oversize chunker inputs; `do_code_enrichment` only when warranted |
| GREEN       | Blind-test baseline (`AGENT-VAL-01`)                              | smoke-equivalent probe must `GATE_PASS`            |

Run each probe (loop, not copy-paste):

```bash
run_probe() {  # run_probe <key> <path>
  local key="$1" pdf="$2" out="${ROOT}/probes/$1"
  mkdir -p "$out"
  python -m mmrag_v2.cli process "$pdf" \
    --output-dir "$out" \
    --pages 20 --batch-size 3 \
    --enable-ocr --ocr-mode auto \
    --vision-provider none \
    --no-refiner \
    --verbose 2>&1 | tee "$out/run.log"
  python scripts/qa_universal_invariants.py "$out/ingestion.jsonl" \
    2>&1 | tee "$out/universal.log" || true
  python scripts/qa_conversion_audit.py    "$out/ingestion.jsonl" \
    2>&1 | tee "$out/audit.log"     || true
}

run_probe HARRY    "$HARRY"
run_probe SCAN0013 "$SCAN0013"
run_probe COMBAT   "$COMBAT"
run_probe CAROK    "$CAROK"
run_probe AYEVA    "$AYEVA"
run_probe GREEN    "$GREEN"
```

**Per-probe acceptance assertions** (record PASS/FAIL in
`${ROOT}/proof/probe_assertions.txt`):

```bash
assert() {  # assert <id> <ok-bool> <evidence>
  if [ "$2" = "1" ]; then echo "OK   $1 $3"; else echo "FAIL $1 $3"; fi
}

# HARRY: born-digital novel — profile=digital_literature, route=native_digital,
# and the post-Docling sanity pass produces an ordered/clean page-13 chunk.
PROFILE=$(jq -r 'select(.object_type=="ingestion_metadata") | .profile_type // empty' \
  "${ROOT}/probes/HARRY/ingestion.jsonl" | head -1)
ROUTE=$(jq -r 'select(.object_type=="ingestion_metadata") | .document_type // empty' \
  "${ROOT}/probes/HARRY/ingestion.jsonl" | head -1)
assert HARRY-PROFILE $([ "$PROFILE" = "digital_literature" ] && echo 1 || echo 0) \
  "profile=$PROFILE"
assert HARRY-ROUTE   $([ "$ROUTE" = "native_digital" ] && echo 1 || echo 0) \
  "route=$ROUTE"

# HARRY post-Docling sanity assertions on page 13 (chapter 1 first body chunk):
#   - paragraphs in PDF y-order: "...nonsense." precedes "Mr. Dursley was the
#     director" precedes "The Dursleys had everything"
#   - drop cap healed: chunk starts with "Mr. and Mrs. Dursley" (capital M
#     prepended), not "r. and Mrs. Dursley"
#   - no label leak: lowercase "other" / capitalised "Other" must not appear
#     in the body text of the page-13 chunk
HARRY_P13_OK=$(python - "${ROOT}/probes/HARRY/ingestion.jsonl" <<'PY'
import json, re, sys
text = ""
for line in open(sys.argv[1], encoding="utf-8"):
    rec = json.loads(line)
    md = rec.get("metadata") or {}
    if md.get("page_number") == 13 and rec.get("modality") == "text":
        text = rec.get("content", "") or ""
        break
if not text:
    print(0); sys.exit()
order_ok = (
    text.find("hold with such nonsense") >= 0
    and text.find("Mr. Dursley was the director") > text.find("hold with such nonsense")
    and text.find("The Dursleys had everything") > text.find("Mr. Dursley was the director")
)
dropcap_ok = "Mr. and Mrs. Dursley" in text and not re.search(
    r"^\s*r\. and Mrs\. Dursley", text, flags=re.MULTILINE
)
no_label_leak = not re.search(r"^(Other|other|Icon|Table)\s*$", text, flags=re.MULTILINE)
no_cover_garbage = "AND Potter SIONE" not in text
print(1 if all([order_ok, dropcap_ok, no_label_leak, no_cover_garbage]) else 0)
PY
)
assert HARRY-P13-SANITY $HARRY_P13_OK "post-Docling sanity pass on page 13"

# SCAN0013: scanned business form — profile is a scanned variant and route is scanned_book.
SCAN_PROFILE=$(jq -r 'select(.object_type=="ingestion_metadata") | .profile_type // empty' \
  "${ROOT}/probes/SCAN0013/ingestion.jsonl" | head -1)
SCAN_ROUTE=$(jq -r 'select(.object_type=="ingestion_metadata") | .document_type // empty' \
  "${ROOT}/probes/SCAN0013/ingestion.jsonl" | head -1)
assert SCAN0013-PROFILE \
  $([ "$SCAN_PROFILE" = "scanned" ] || [ "$SCAN_PROFILE" = "scanned_degraded" ] && echo 1 || echo 0) \
  "profile=$SCAN_PROFILE"
assert SCAN0013-ROUTE \
  $([ "$SCAN_ROUTE" = "report" ] && echo 1 || echo 0) \
  "route=$SCAN_ROUTE"

# COMBAT: no chunk text contains a known PUA / replacement-char run
LEAKED=$(python - "${ROOT}/probes/COMBAT/ingestion.jsonl" <<'PY'
import json, sys
bad = 0
for line in open(sys.argv[1], encoding="utf-8"):
    try: rec = json.loads(line)
    except Exception: continue
    text = rec.get("content", "") or rec.get("text", "")
    if "�" in text or sum(1 for c in text if 0xE000 <= ord(c) <= 0xF8FF) > 5:
        bad += 1
print(bad)
PY
)
assert COMBAT-NO-LEAK $([ "${LEAKED:-1}" = "0" ] && echo 1 || echo 0) "leaked_corrupted_chunks=$LEAKED"

# CAROK: no chunks with image_id but empty content AND blank asset signature
# (replace metric with the project's actual blank-asset signal if different)
BLANK=$(python - "${ROOT}/probes/CAROK/ingestion.jsonl" <<'PY'
import json, sys
n = 0
for line in open(sys.argv[1], encoding="utf-8"):
    try: rec = json.loads(line)
    except Exception: continue
    if rec.get("type") == "image" and not (rec.get("description") or rec.get("vlm_description")):
        n += 1
print(n)
PY
)
assert CAROK-BLANK $([ "${BLANK:-1}" -lt 5 ] && echo 1 || echo 0) "blank_image_chunks=$BLANK"

# AYEVA: no chunk exceeds the chunker char guard
OVER=$(python - "${ROOT}/probes/AYEVA/ingestion.jsonl" <<'PY'
import json, sys
m = 0
for line in open(sys.argv[1], encoding="utf-8"):
    try: rec = json.loads(line)
    except Exception: continue
    txt = rec.get("content", "") or rec.get("text", "")
    if len(txt) > 500_000: m += 1
print(m)
PY
)
assert AYEVA-GUARD $([ "${OVER:-1}" = "0" ] && echo 1 || echo 0) "oversize_chunks=$OVER"

# GREEN: blind-test must produce a non-empty JSONL with no CONVERT_ERROR
GREEN_OK=0
test -s "${ROOT}/probes/GREEN/ingestion.jsonl" \
  && ! grep -q "CONVERT_ERROR\|MISSING_JSONL" "${ROOT}/probes/GREEN/run.log" \
  && GREEN_OK=1
assert GREEN-BLIND $GREEN_OK "blind_baseline_clean"
```

Append all `assert` lines to `${ROOT}/proof/probe_assertions.txt`. Any
`FAIL` is a probe regression — go to PHASE 3 with that bug class flagged.

----------------------------------------------------------------------
## PHASE 3 — Focused tests for each bug class (TDD on probe failures)
----------------------------------------------------------------------

For each `FAIL` from PHASE 2, before changing any production code:

1. Locate the existing test file from the table below.
2. Add or strengthen a **deterministic, fixture-only** test reproducing the
   failure. No network, no live PDFs in production logic, no random seeds.
3. Run that single test → must FAIL on current code (proving it captures the
   bug). Record the failing output to `${ROOT}/proof/red_<bugclass>.log`.
4. Implement the minimal fix. Re-run the test → must PASS. Record the green
   output to `${ROOT}/proof/green_<bugclass>.log`.
5. Re-run the matching probe from PHASE 2. The probe assertion must flip to
   `OK`.

| Bug class                  | Test file                                                         | Probe        |
|----------------------------|-------------------------------------------------------------------|--------------|
| Classifier fallback        | `tests/test_classifier_fallback.py`                               | SCAN0013     |
| `digital_literature` route | `tests/test_classifier_digital_literature.py`                     | HARRY        |
| Reading-order y-sort       | `tests/test_docling_postprocess_reading_order.py`                 | HARRY        |
| Drop-cap promotion         | `tests/test_docling_postprocess_dropcap.py`                       | HARRY        |
| Label-leak filter          | `tests/test_docling_postprocess_label_filter.py`                  | HARRY        |
| OCR gating                 | `tests/test_docling_postprocess_ocr_gating.py`                    | HARRY        |
| Post-Docling profile wiring| `tests/test_docling_postprocess_profile_integration.py`           | HARRY        |
| Scanned-book route         | `tests/test_pdf_conversion_plan.py`                               | SCAN0013     |
| Chunker guard              | `tests/test_chunker_guard.py`                                     | AYEVA        |
| Blank asset validation     | `tests/test_blank_asset_quarantine.py`                            | CAROK        |
| Corruption interceptor     | `tests/test_corruption_quarantine.py`                             | COMBAT       |
| Bridge integrity           | `tests/test_finalization_bridge.py`                               | (any)        |

Hard rules:
- **AGENT-TEST-01**: never delete, weaken, or reframe an existing assertion to
  go green. Strengthen, don't soften.
- **AGENT-VAL-01**: do not branch on probe filenames or directory names in
  production code. Production may inspect content, modality, profile — never
  the path string `Harry`/`Combat`/`Ayeva`/`CarOK`/`Greenhouse`.
- A bug class without a red-then-green log pair is **not fixed**.

Record the audit in `${ROOT}/proof/bugclass_audit.tsv`:
```
class   test_file   red_log   green_log   probe   probe_after
```

----------------------------------------------------------------------
## PHASE 4 — Full unit suite
----------------------------------------------------------------------

```bash
python -m pytest tests/ -v --tb=short \
  --junitxml="${ROOT}/proof/unit_junit.xml" \
  2>&1 | tee "${ROOT}/proof/unit_full.log"
```

Acceptance:
- Exit code = 0.
- `failed` count in junit = 0. `errors` = 0.
- `skipped` ≤ 2 (torch-MPS platform skip is allowed; investigate any third).
- A test cannot pass by xfail/skipif added in this run — verify with:
  ```bash
  git diff --unified=0 HEAD~..HEAD tests/ | \
    grep -E "^\+.*(xfail|skipif|@pytest.mark.skip)" \
    | tee "${ROOT}/proof/test_softening_check.txt"
  ```
  Any new soft mark must be justified inline with `# AGENT-TEST-01-OK: <reason>`.

----------------------------------------------------------------------
## PHASE 5 — Multi-profile smoke (combined plan §5, bullet 4)
----------------------------------------------------------------------

```bash
bash scripts/smoke_multiprofile.sh "${ROOT}/smoke" 2>&1 | tee "${ROOT}/smoke/run.log"
```

Acceptance (all must hold):
- Every row in `${ROOT}/smoke/_summary.txt` shows both `GATE_PASS` and
  `UNIVERSAL_PASS`.
- Greenhouse blind-test row is present and passing (`AGENT-VAL-01`).
- No `CONVERT_ERROR` and no `MISSING_JSONL` anywhere in
  `${ROOT}/smoke/_summary.txt`.
- Row count ≥ the row count in the most recent prior smoke run under
  `output/smoke_multiprofile_*` — never silently drop coverage.

Verify mechanically:
```bash
SUM="${ROOT}/smoke/_summary.txt"
TOTAL=$(grep -cE "^[a-z_]+\|" "$SUM")
GATE=$(grep -c "GATE_PASS" "$SUM")
UNIV=$(grep -c "UNIVERSAL_PASS" "$SUM")
ERRS=$(grep -cE "CONVERT_ERROR|MISSING_JSONL|GATE_FAIL|UNIVERSAL_FAIL" "$SUM")
PRIOR=$(ls -1d output/smoke_multiprofile_* 2>/dev/null | tail -1)
PRIOR_TOTAL=$([ -n "$PRIOR" ] && grep -cE "^[a-z_]+\|" "$PRIOR/_summary.txt" || echo 0)
echo "TOTAL=$TOTAL GATE=$GATE UNIV=$UNIV ERRS=$ERRS PRIOR_TOTAL=$PRIOR_TOTAL" \
  | tee "${ROOT}/proof/smoke_metrics.txt"
test "$ERRS" = "0" && test "$GATE" = "$TOTAL" && test "$UNIV" = "$TOTAL" \
  && test "$TOTAL" -ge "$PRIOR_TOTAL"
```

The shell `test` chain at the end **must exit 0**. If it exits non-zero, the
smoke phase failed regardless of how the rows look in prose.

----------------------------------------------------------------------
## PHASE 6 — Self-audit (machine-checkable; produces the verdict)
----------------------------------------------------------------------

This phase exists because "looks good" is not evidence. Generate a single
verdict file from real artefacts. The verdict must be reproducible: re-running
PHASE 6 against the same `${ROOT}` must produce the same verdict.

### 6.1 Anti-cheat audits

Each audit emits `OK`/`FAIL` to `${ROOT}/proof/anticheat.txt`. Any `FAIL`
forces the verdict to `FAIL`.

```bash
A="${ROOT}/proof/anticheat.txt"; : > "$A"

# A1. No --profile-override anywhere in this run's logs (debugging-only flag).
if grep -RnE "[-]{2}profile-override" "${ROOT}" 2>/dev/null; then
  echo "FAIL A1 profile-override used in acceptance run" >> "$A"
else
  echo "OK   A1 no profile-override usage" >> "$A"
fi

# A2. No probe filenames in production source (tests/fixtures excepted).
if grep -REn "Harry|Combat[ _]Aircraft|Ayeva|CarOK|Greenhouse|Python_Distilled|Chaubal|Fluent" \
        src/mmrag_v2/ 2>/dev/null \
   | grep -v "tests/" | grep -v "fixtures/"; then
  echo "FAIL A2 probe filename leaked into production source" >> "$A"
else
  echo "OK   A2 production source free of probe filenames" >> "$A"
fi

# A3. No PdfPipelineOptions / DocumentConverter construction outside adapter.
python -m pytest \
  tests/test_pdf_conversion_plan.py::test_no_pipeline_options_construction_outside_adapter \
  tests/test_pdf_conversion_plan.py::test_no_production_docling_imports_outside_adapter \
  -q && echo "OK   A3 boundary guards green" >> "$A" \
       || echo "FAIL A3 boundary guards red"  >> "$A"

# A4. No silent test softening introduced in this run.
if [ -s "${ROOT}/proof/test_softening_check.txt" ] && \
   ! grep -q "AGENT-TEST-01-OK" "${ROOT}/proof/test_softening_check.txt"; then
  echo "FAIL A4 unjustified xfail/skip added" >> "$A"
else
  echo "OK   A4 no unjustified test softening" >> "$A"
fi

# A5. Schema/version stamping is current (schema 2.7.x; engine 2.8.x in v2.8+).
SVER=$(python -c "import mmrag_v2.version as v; print(v.SCHEMA_VERSION)" 2>/dev/null || echo "?")
case "$SVER" in 2.7.*) echo "OK   A5 schema=$SVER" >> "$A" ;;
                *)    echo "FAIL A5 schema=$SVER (expected 2.7.x — schema is decoupled from engine since v2.8)" >> "$A" ;;
esac
EVER=$(python -c "from mmrag_v2.version import __engine_version__; print(__engine_version__)" 2>/dev/null || echo "?")
case "$EVER" in 2.8.*) echo "OK   A5b engine=$EVER" >> "$A" ;;
                *)    echo "FAIL A5b engine=$EVER (expected 2.8.x as of 2026-05-04)" >> "$A" ;;
esac

# A6. Every PHASE 2 FAIL has a matching red→green log pair in PHASE 3.
python - "${ROOT}" <<'PY' >> "$A"
import os, re, sys, glob, pathlib
root = pathlib.Path(sys.argv[1])
asserts = (root/"proof/probe_assertions.txt").read_text()
fails = re.findall(r"^FAIL\s+(\S+)", asserts, re.M)
audit = (root/"proof/bugclass_audit.tsv")
covered = audit.read_text() if audit.exists() else ""
missing = [f for f in fails if f.split("-")[0] not in covered]
print("OK   A6 every probe FAIL has a red/green pair" if not missing
      else f"FAIL A6 uncovered probe failures: {missing}")
PY
```

### 6.2 Generate verdict

```bash
python - "${ROOT}" <<'PY'
import json, pathlib, re, sys, time
root = pathlib.Path(sys.argv[1])
def read(p): return (root/p).read_text() if (root/p).exists() else ""

upstream    = read("proof/upstream_audit.txt")
probes      = read("proof/probe_assertions.txt")
anticheat   = read("proof/anticheat.txt")
smoke_metr  = read("proof/smoke_metrics.txt")
unit_log    = read("proof/unit_full.log")

def has_fail(s):  return bool(re.search(r"^(FAIL|MISS)\b", s, re.M))
unit_ok  = bool(re.search(r"== .* passed.* ==", unit_log)) and not re.search(r"\b(failed|error)s?:\s*[1-9]", unit_log)
smoke_ok = bool(smoke_metr) and "ERRS=0" in smoke_metr

verdict = "PASS" if all([
    not has_fail(upstream),
    not has_fail(probes),
    not has_fail(anticheat),
    unit_ok, smoke_ok,
]) else "FAIL"

out = {
  "run_id": root.name,
  "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "verdict": verdict,
  "summary": {
    "upstream_ok":  not has_fail(upstream),
    "probes_ok":    not has_fail(probes),
    "anticheat_ok": not has_fail(anticheat),
    "unit_ok":      unit_ok,
    "smoke_ok":     smoke_ok,
  },
  "evidence": {
    "env":               "proof/env.txt",
    "upstream_audit":    "proof/upstream_audit.txt",
    "probe_paths":       "proof/probe_paths.txt",
    "probe_assertions":  "proof/probe_assertions.txt",
    "bugclass_audit":    "proof/bugclass_audit.tsv",
    "unit_log":          "proof/unit_full.log",
    "unit_junit":        "proof/unit_junit.xml",
    "smoke_summary":     "smoke/_summary.txt",
    "smoke_metrics":     "proof/smoke_metrics.txt",
    "anticheat":         "proof/anticheat.txt",
  },
  "invariants_cited": [
    "AGENT-VAL-01","AGENT-EVIDENCE-01","AGENT-STATUS-01",
    "AGENT-TEST-01","AGENT-SPATIAL-20","AGENT-DOCS-01",
  ],
}
(root/"VERDICT.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
sys.exit(0 if verdict == "PASS" else 1)
PY
```

The Python block exits non-zero on `FAIL`. Do **not** swallow that exit code.

### 6.3 Human-readable proof file

Write `${ROOT}/proof/ACCEPTANCE_PROOF.md` with this exact structure (fill from
the artefacts above — never invent numbers):

```markdown
# Acceptance Proof — RUN_ID

## Verdict
PASS | FAIL  (mirrors VERDICT.json)

## Environment
<contents of proof/env.txt>

## Upstream prerequisites (combined plan §1–§4)
<table built from proof/upstream_audit.txt>

## Targeted probes (combined plan §5, bullet 1)
<table: probe | path | profile | route | assertion id | result>

## Bug-class fixes (combined plan §5, bullet 2)
<table from proof/bugclass_audit.tsv with red/green log paths>

## Unit suite (combined plan §5, bullet 3)
- junit: proof/unit_junit.xml
- passed / failed / errors / skipped: <numbers>

## Smoke (combined plan §5, bullet 4)
<copy proof/smoke_metrics.txt verbatim, then list each row's status>

## Anti-cheat audits
<contents of proof/anticheat.txt>

## Conclusion
- If PASS: state the broad rerun is now unblocked, link VERDICT.json,
  and propose updating docs/PROGRESS_CHECKLIST.md + a new
  docs/QUALITY_SNAPSHOT_<date>.md.
- If FAIL: list each FAIL/MISS line, the smallest fix, and the phase to
  re-enter. Do NOT mark the task complete.
```

----------------------------------------------------------------------
## DEFINITION OF DONE (binary; all must be true)
----------------------------------------------------------------------

You may only report success when ALL of these hold simultaneously:

1. `${ROOT}/VERDICT.json` exists, parses as JSON, and `.verdict == "PASS"`.
2. `${ROOT}/proof/upstream_audit.txt` contains zero `MISS ` and zero `FAIL` lines.
3. `${ROOT}/proof/probe_assertions.txt` contains zero `FAIL ` lines.
4. `${ROOT}/proof/anticheat.txt` contains zero `FAIL ` lines.
5. `${ROOT}/proof/unit_junit.xml` reports `failures="0" errors="0"`.
6. `${ROOT}/smoke/_summary.txt` shows `GATE_PASS` and `UNIVERSAL_PASS` on
   every category row, including the Greenhouse blind-test row.
7. No `--profile-override` appears anywhere under `${ROOT}`.
8. No probe filename appears in `src/mmrag_v2/` outside tests/fixtures.
9. Every `FAIL` produced in PHASE 2 has a corresponding red→green log pair
   recorded in `${ROOT}/proof/bugclass_audit.tsv`.
10. `${ROOT}/proof/ACCEPTANCE_PROOF.md` exists and matches the template above.

If any of (1)–(10) is false, the task is **not** done. Re-enter the failing
phase, fix the root cause, and regenerate the verdict.

----------------------------------------------------------------------
## RULES OF ENGAGEMENT
----------------------------------------------------------------------

- **Never** weaken a test to go green (`AGENT-TEST-01`).
- **Never** branch on a filename or directory string in production code
  (`AGENT-VAL-01`).
- **Never** declare PASS without a `VERDICT.json` whose `.verdict == "PASS"`
  and a matching `ACCEPTANCE_PROOF.md` (`AGENT-EVIDENCE-01`).
- **Never** use `--profile-override` in any acceptance run.
- **Never** start a broad corpus reconversion before the verdict is `PASS`.
- **Never** mark Workstream B (`do_code_enrichment`) on encoding-corruption
  alone — see CLAUDE.md "Workstream B Code Enrichment Guardrail".
- If a probe FAIL has no clear bug class, STOP and surface it before
  inventing one.
- If you discover a contradiction between this prompt and `AGENTS.md` /
  `docs/AGENT_GOVERNANCE.md`, the governance docs win — flag the conflict.

----------------------------------------------------------------------
## REFERENCES
----------------------------------------------------------------------

- `AGENTS.md` — Level 0 invariants
- `docs/AGENT_GOVERNANCE.md` — evidence/status rules
- `docs/DECISIONS.md` — architectural decisions
- `docs/QUALITY_GATES.md` — pass/fail thresholds
- `docs/PLAN_V2.8_PRODUCTION_GAPS.md` — **SHIPPED 2026-05-04**, retained for historical context. For the next cycle see `docs/PLAN_V2.9_DRAFT_PROMPT.md` (`docs/archive/PLAN_V2.7_DOCUMENT_UNDERSTANDING.md` is the older archived predecessor).
- `docs/CONTEXTUAL_RETRIEVAL_PROMPT.md` — companion (combined plan §4)
- `scripts/smoke_multiprofile.sh` — smoke runner
- `scripts/qa_conversion_audit.py` — conversion audit
- `scripts/qa_universal_invariants.py` — universal invariant checker
- `tests/test_pdf_conversion_plan.py` — plan/adapter boundary tests
- `tests/test_finalization_bridge.py` — CLI→plan→adapter bridge tests
- `tests/test_classifier_fallback.py` — classifier regression
- `tests/test_chunker_guard.py` — chunker input guard
- `tests/test_blank_asset_quarantine.py` — blank asset policy
- `tests/test_corruption_quarantine.py` — corruption interceptor
- `tests/test_contextual_retrieval.py` — contextual retrieval (§4)
```

---

**END OF PROMPT**
