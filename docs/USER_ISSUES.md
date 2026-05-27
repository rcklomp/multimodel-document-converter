# User Issues — MM-Converter-V2

**Purpose**: append-only registry of user-filed issues against
documented-limitation document classes. Read by
`scripts/analyze_doc_class_telemetry.py` to feed the
`open_user_issues` signal into the v2.15 Option F telemetry
promotion rule (per
`docs/DECISIONS.md` "v2.15 Documented-Limitation Telemetry Threshold").

## How this file is used

Per `docs/CYCLE_OPEN_CHECKLIST.md`, every cycle-open includes a
"Review USER_ISSUES.md for new entries since prior tag" step.
`analyze_doc_class_telemetry.py` parses this file's table rows and
counts entries per `doc_class` since the prior cycle's tag date.
The count flows into the v2.15 telemetry promotion rule:

- **Standard promotion arm** fires on `hit-rate >= 5%` AND
  (`severe_defect_tag = True` OR `open_user_issues >= 1`).
- **Defect-override arm** fires on `severe_defect_tag = True` AND
  `hit-rate >= 1%` (independent of issue count).
- **Closure arm** blocked when `open_user_issues >= 1` (file
  evidence beats telemetry; class cannot auto-close if real
  complaints exist).

## Format

Append rows to the table below. **DO NOT delete or modify
existing rows** — this is append-only by design (audit trail).
If an issue is resolved, append a new row with the same
`doc_class` and `observed_behavior = "RESOLVED: <commit hash>"`
rather than editing in place. The analyzer counts all rows for
the per-class total; resolution rows are tallied separately
via grep when needed.

Column meanings:
- **date**: ISO `YYYY-MM-DD` of issue filing
- **doc_class**: must match a `name` entry in
  `src/mmrag_v2/retrieval/documented_limitations.py`
  (e.g. `CarOK_voorraadtelling`, `Fluent_Python`)
- **query**: the user query that surfaced the issue (or "n/a" if
  not query-driven — e.g. a corpus-ingestion bug)
- **observed_behavior**: what the user saw (one line)
- **expected_behavior**: what they expected (one line)

The parser is a regex against `| YYYY-MM-DD | <doc_class> | …`
so keep the date in the first column and the doc_class in the
second column. Other column orderings are fine for human reading
but won't be picked up by the analyzer.

## Active issues

| date | doc_class | query | observed_behavior | expected_behavior |
|---|---|---|---|---|

<!-- No issues filed yet. v2.15 documented-limitation entries:
     CarOK_voorraadtelling, Fluent_Python (both severe_defect_tag=True
     on entry per their prior-cycle defect history; promotion-eligible
     via defect-override arm regardless of issue count). -->

## Resolution log (informational)

Append resolution markers here as one-line entries. Not parsed
by the analyzer; for human audit only.

<!-- Format suggestion:
     YYYY-MM-DD <commit-hash> <doc_class>: <one-line resolution>
-->

## v2.17 regression-attack outcomes (2026-05-27)

Operator instruction "stop postponing fixes; start solving the
regression issues" — landed targeted fixes for the 4 ongoing
regressions documented at v2.16.0 ship. Status of each:

**R1 — Fluent_Python `partial_code_cross_page` flag inert on HybridChunker path.**
*Status: FIX SHIPPED, end-to-end validation BLOCKED by separate
infra issue.* The v2.16 Phase 3 adjacency-fetch mechanism was inert
because the HybridChunker cross-page emit site never set
`partial_code=True`. Surgical fix at `processor.py:3613` adds the
predicate `partial_code=True if (chunk_type==CODE AND
len(per_page_text)>1) else None`. Predicate truth-table tested at
`tests/test_partial_code_cross_page_hybrid.py` (10/10 PASS).
Baseline analysis of `output/Fluent_Python/ingestion.jsonl` finds
**300 cross-page CODE chunks** the fix WILL flag at the next
clean HybridChunker run, vs **0 in the baseline** — the mechanism
is no longer inert. End-to-end re-conversion in this session
fell back to recovery extraction methods due to a pre-existing
Docling/MPS `float64` dtype error (the same blocker recorded in
`PHASE_A_SKIP_AUDIT.md` 2026-05-27 §A8 for the Ayeva real-Docling
tests). Empirical Fluent_Python validation pass-rate measurement
needs the MPS infra issue addressed first.

**R2 — CarOK v2.14 P1 -26.9pp Format regression, v2.16 P4 dedup unvalidated.**
*Status: MECHANISM VERIFIED, production Qdrant re-ingest needed.*
The v2.16 Phase 4 IoU dedup mechanism is shipped + working
correctly. Comparing `output/CarOK_v2_14_p1_force_works/` (71
chunks, no dedup — the regression baseline) vs
`output/CarOK_v2_16_p4_dedup/` (55 chunks, dedup armed): **16/16
dropped chunks are TEXT modality on the 5 VLM-table pages
{1,2,3,4,11}, ZERO collateral damage**. Dropped chunks are
unmistakably prose-flattened duplicates of the inventory tables
(samples: "Castrol, ink.ex.BTW Titel = 4,44 Castrol Magnatec
10W-40...", "1 Febi Bilstein, merk = 21594..."). The dedup IS
the correct fix for the v2.14 P1 regression mode. Remaining gap:
the v2.16 P4 dedup-armed output was never re-ingested into the
production Qdrant collection `mmrag_v2_8__qwen3_local`, which
still indexes the v2.14 P1 buggy version. Operator action:
`bash scripts/rebuild_mmrag_v2_8_for_rc1.py` against the
`CarOK_v2_16_p4_dedup` output to land the Format recovery in
production retrieval.

**R3 — Earthship_Vol1 multi-column OCR layout damage.**
*Status: SILENT DISPATCH BUG FIXED, empirical OCR-engine
comparison shipped as operator-triggered script.* The CLI
advertises `--ocr-engine {tesseract|easyocr|doctr}` but the
adapter at `engines/docling_adapter.py` line 96–98 silently
hardcoded EasyOcrOptions regardless of the flag. The v2.13
Phase 2 force_full_page_ocr fix could therefore never test
alternative engines on Earthship's multi-column scanned pages.
Fix: `DoclingPdfAdapter._build_ocr_options` now dispatches on
`plan.ocr_engine` with quiet-fallback semantics for missing
deps. Added `ocrmac` (macOS Vision framework, Apple Silicon
native) to the `OCREngine` enum because Apple's text-recognition
is purpose-built for multi-column document layouts; available on
this system per `pip show ocrmac`. Truth-table tests at
`tests/test_docling_adapter_ocr_dispatch.py` (6/6 PASS).
Required test update: `test_adapter_ocr_enabled_uses_easyocr`
in `tests/test_pdf_conversion_plan.py` was pinning the silent-bug
as "correct behavior" — replaced with
`test_adapter_ocr_engine_dispatches_to_requested_class` per
CLAUDE.md Test Contract Integrity (documented requirement change
in test docstring). Operator-triggered empirical comparison:
`bash scripts/v217_compare_earthship_ocr.sh` runs Earthship
through both easyocr and ocrmac then emits a side-by-side
chunk-count / methods / short-chunk report. Did NOT run in this
autonomous session because the full conversion is ~20-30 min per
engine + carries the same MPS infra risk as R1's Fluent_Python
attempt.

**R4 — omlx Qwen3-Embedding-8B -12pp R@1 deficit on Python_Cookbook /
IRJET / Greenhouse / Hybrid_electric_vehicles.**
*Status: STRUCTURALLY BLOCKED, closure requires Phase C ColPali.*
Per `docs/archive/diagnostics/DIAGNOSTIC_2026-05-25_v2.16_p2_omlx_deficit_root_cause.md`
the Phase 2 diagnostic explicitly could not test hypotheses H1–H4
because the apples-to-apples Dashscope cloud baseline collection
(`mmrag_v2_8__qwen3_dashscope`) was dropped 2026-05-23 PM under
the v2.14 Phase 3 user override; cold-storage snapshot retained
but offline. Verdict was "multi-factor, not H2-dominant, not
H3-dominant" — i.e. neither OOV nor cross-lingual is the dominant
cause. Pure-text retrieval-side augmentation is exhausted: HyDE
falsified (v2.15 P1 dead lever); query-rewriting killed by Phase 2
verdict (2nd dead lever). The remaining lever is visual retrieval
(ColPali / Phase C) which depends on Phase A full-rewrite
completion (rebooked to v3.0.2 per
`docs/PHASE_A_SCOPE_NEGOTIATION.md` 2026-05-27 entry). Per
operator instruction to stop postponing: closure of this
regression IS Phase C, which IS scheduled (v3.0.2). The "stop
postponing" remedy is to execute Phase C, not to invent a new
v2.X lever — every such lever has been falsified.
