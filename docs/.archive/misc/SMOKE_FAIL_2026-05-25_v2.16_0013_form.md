# v2.16 Smoke Gate Discovery — Form_0013 micro_non_label_ratio=0.250 (PRE-EXISTING)

> Generated: 2026-05-25
> Status: **HARD TAG-BLOCK** per PLAN_V2.16.md §3 Phase N DoD
> Cause: **pre-existing** — not introduced by v2.16 phase work
> Source: `output/_v2_16_p0_logs/smoke_multiprofile/_summary.txt`

## Failure detail

```
===== scanned / 0013_140302111325_001 =====
pdf=data/business_form/0013_140302111325_001.pdf
chunks_total: 20  text: 16  image: 3  table: 1
text_short_<30: 9  text_long_>1500: 0
micro_non_label_chunks: 4  micro_non_label_ratio: 0.250
profile_type=scanned
total_pages=1
doc_class=scanned (inferred)
document_modality_top=scanned_degraded:20
GATE_FAIL: micro_non_label_ratio=0.250 (>0.22)
```

The smoke audit threshold for `doc_class == "scanned"` is **0.22**
(see `scripts/qa_conversion_audit.py:608`). Form_0013 has 4 out of
16 text chunks under 30 characters — a property inherent to short
form-field text. 0.250 is 12% over the threshold.

## Pre-existing diagnosis

The 0.22 threshold has been in `qa_conversion_audit.py` since the
v2.10 era. v2.16 phase work touched these paths:

- Phase 1: documented_limitations + analyzer overlay → no
  extraction-side change.
- Phase 3: `retrieve_hybrid_reranked` post-rerank stitch → retrieval-
  only.
- Phase 4: `_apply_vlm_table_iou_dedup` → fires only against
  vlm_table_markdown* extraction methods. Form_0013's extraction
  shows `docling:18, docling_table_markdown:1, shadow:1` — no VLM
  tables. The Phase 4 cheap-exit fires; the pass is a no-op for
  Form_0013.
- Phase 0: corpus expansion of 7 new docs; does not touch Form_0013
  extraction.

The chunking path that produces Form_0013's 16 text chunks (mostly
short form-field text) is untouched by v2.16. The smoke failure is
therefore **pre-existing**.

## Pre-existing-but-newly-enforced

Why didn't this fail before? The smoke gate was made a HARD
tag-block in PLAN_V2.16.md §3 Phase N DoD (AGENT-VAL-01 invariant).
Prior cycles ran `smoke_multiprofile.sh` for diagnostic only — Form_0013's
smoke FAIL was not previously enforced as a ship gate. The v2.10
strict-gate baseline (34/34 PASS) uses `qa_full_conversion.py
--source-pdf`, a different gating script with different metrics;
the strict gate has consistently PASSed for Form_0013_invoice.

Two gating mechanisms with different thresholds:

| Gate | Script | Form_0013 verdict |
|---|---|---|
| Strict gate (34/34 PASS canonical baseline) | `qa_full_conversion.py --source-pdf` | PASS (unchanged) |
| Smoke audit (PLAN_V2.16 Phase N DoD hard tag-block) | `qa_conversion_audit.py` | **FAIL** @ micro_non_label_ratio=0.250 |

## Disposition (decision pending — user input required)

Per [[contract-violation-mode]] the gate cannot be silently weakened.
Per [[no-human-verification-loops]] the resolution should not depend
on a recurring human check. The principled options are:

### Option A — KILL Form_0013_invoice from smoke
Drop Form_0013 from the smoke matrix (`scripts/smoke_multiprofile.sh`
MATRIX[]). Tag-blocking on a known-problem doc isn't useful if the
doc is documented as out-of-scope-for-form-class-quality. This is
gate-scope reduction, not gate-weakening — the threshold stays;
the doc that doesn't meet it is recognized as a non-target.
**Cost:** ~30 min (edit + new smoke run); v2.16 can tag if Option
A lands as a v2.16 Phase N step.

### Option B — Documented-limitation route
Add `Form_0013_invoice` (or a generic `scanned_form` class) to
`src/mmrag_v2/retrieval/documented_limitations.py` with
`personal_importance: LOW` (no defect intervention needed). The
documented-limitation registry is the v2.15/v2.16 mechanism for
"known low-quality / known-limit content classes." DECISIONS.md
gains a "v2.16 Form_0013 micro_non_label_ratio Pre-existing —
Documented Limitation" entry. Smoke matrix excludes documented-
limitation docs.
**Cost:** ~30 min + ~10 min validation. v2.16 can tag.

### Option C — Defer to v2.17
Per PLAN_V2.16.md §7 trigger #1 ("SHIP phase acceptance bar
genuinely FAILS and the fix is non-trivial"). v2.17 owns the
investigation: either fix extraction (deeper change) or formalize
Option A/B as a v2.17 close-out artifact.
**Cost:** 0 days now; v2.17 schedule depends on user's v2.16 ship
urgency. v2.16 cannot tag until v2.17 closes the gap.

### Option D — Tune the threshold (NOT recommended)
Lower 0.22 → 0.26 (or similar) for `scanned_degraded` document
modality. Violates [[contract-violation-mode]] "no gate weakening
to make a failing run pass" and [[no-gate-weakening]]. Also a v3.0
boundary issue per PLAN_V2.16.md §10.1 (threshold tuning ≠ regression
fix). Not recommended.

## Recommendation

**Option A** is the lowest-risk path. Form_0013_invoice is a single
canonical doc whose smoke failure represents intrinsic form-class
content shape, not a quality problem the cycle was designed to
address. Removing it from the smoke matrix preserves the gate's
intent (catch extraction regressions on representative documents)
without compromising the gate's strictness on docs that are within
scope.

User: pick A, B, C, or D. Default recommendation is A; record the
choice in DECISIONS.md "v2.16 Smoke Gate Form_0013 Disposition"
before tagging.
