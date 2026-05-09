# Quality Gates and Known Warnings

## Required Checks
- QA-CHECK-01: token balance variance within profile-specific gate tolerance (standard target: 10%).
- QA-CHECK-02: every `asset_ref.file_path` must exist on disk.
- QA-CHECK-03: `breadcrumb_path` depth must match `hierarchy.level`.
- QA-CHECK-04: all `bbox` values must be integer coordinates in range `[0,1000]` (REQ-COORD-01 compliance).
- QA-CHECK-05: image/table chunks must have `asset_ref`.
- REQ-COORD-02: `page_width`/`page_height` present for spatial metadata.

## QA-CHECK-01 Tolerance Policy (Pass/Fail Source)

| Scope | Tolerance | Gate Decision |
|------|-----------|---------------|
| All profiles | 10% (`0.10`) | Pass |
| Any profile above `0.10` | >10% | Fail |

The former 18% temporary waiver for `digital_magazine` was retired in v2.6 after
implementing IMAGE-bbox-aware source text extraction (commit 2ff6883). Chart/graph
label text in the PDF text layer is now correctly excluded from the source baseline,
bringing both tested magazines (PCWorld, Combat Aircraft) from ~-20% to under -9%.

## Chunk Quality Policy (Principle-Based)
- Chunk-size quality is profile-specific and empirically validated.
- There is no single hard global min/max that applies to every document type.
- Use acceptance metrics to decide whether a change improved conversion quality.

### Recommended Evaluation Signals
- `text_short_<30` and `text_long_>1500` from `scripts/qa_ingestion_hygiene.py` for shape anomalies.
- `micro_non_label_ratio` for short-chunk fragmentation with label/code exclusions.
- `oversize_ratio` for retrieval-hostile long chunks.
- `orphan_label_ratio` for label/body attachment quality in procedural manuals.
- `code_fragmentation_ratio` for code-like chunks that are too small and not preserved as code.
- Token balance variance (QA-CHECK-01) for coverage risk.
- `infix_strict` (mid-sentence list-number artifact count) for OCR/read-order regressions.
- Sample-based semantic review on representative pages (front matter, dense procedural pages, noisy scans).

### Profile-Oriented Targets (Guidance, not hard invariants)
- `technical_manual`:
  - Expect moderate chunk sizes with `infix_strict=0` and low page-number artifacts.
  - Suggested acceptance gates (scanned-heavy manuals): `micro_non_label_ratio <= 0.22`, `oversize_ratio <= 0.02`, `orphan_label_ratio <= 0.30`.
  - Minimize very short "noise" chunks unless they are valid field labels (e.g., "Origin:").
- `scanned_degraded`:
  - Accept somewhat higher short-chunk ratio, but require clean control chars and stable token balance.
  - Prioritize readability and artifact suppression over aggressive merging.
- `digital/native technical`:
  - Keep short/long outliers bounded via chunking logic.
  - Suggested acceptance gates: `micro_non_label_ratio <= 0.12`, `oversize_ratio <= 0.01`, `orphan_label_ratio <= 0.20`, `code_fragmentation_ratio <= 0.05`.

### Form / Invoice Acceptance Class (added 2026-05-04, PLAN_V2.8 §5)
Forms (invoices, receipts, claim forms, single-page business documents) are
first-class RAG content — production RAG corpora routinely include them. They
have a fundamentally different shape than prose documents: each chunk is a
short field key or value, there is no section-heading hierarchy, and the
prose-calibrated `micro_non_label_ratio` gate would reject every well-extracted
form.

**Detection rule** (matches `scripts/qa_conversion_audit.py` and
`scripts/evaluate_technical_manual_gates.py`):

```
total_pages > 0 AND total_pages <= 5
  AND doc_class == "scanned"
  AND heading_coverage < 0.10   # parent_heading set on <10% of text chunks
```

**Form gate** (the only checks that apply):
- `infix_strict == 0` (mid-sentence list-number artifact count, must be zero)
- `oversize_ratio <= 0.02` (no chunks exceed 1500 chars)

The following checks are **skipped** for forms because they have no semantic
meaning on this content shape:
- `micro_non_label_ratio` — forms are intentionally short labels + values
- `orphan_label_ratio` — forms have no body paragraphs to attach labels to
- `heading_coverage` thresholds — forms have no section hierarchy

**Why no waiver:** Per CLAUDE.md "Project Invariants" / AGENT-VAL-01, every
document category must satisfy `GATE_PASS + UNIVERSAL_PASS`. The form lane is
NOT a waiver — it is a separate, equally rigorous acceptance class with the
threshold appropriate to the content shape. Universal invariants (clean
control chars, integer `[0,1000]` bboxes, non-empty content, modality
present) still apply.

**Output naming:** the audit script reports `FORM_AUDIT_PASS` /
`FORM_AUDIT_FAIL` for forms, distinguishing them from prose `AUDIT_PASS`.
The smoke evaluator emits `GATE_PASS [form: micro_non_label + label-orphan
checks skipped]` so the form lane is always visible in the smoke summary.

### Acceptance Workflow
1. Run `scripts/smoke_multiprofile.sh` — this is the primary gate. Every row must show `GATE_PASS` + `UNIVERSAL_PASS`.
2. Run `scripts/acceptance_technical_manual.sh` for deep validation of the technical-manual category (4 docs × 20 pages).
3. Compare metrics against prior baseline (`_summary.txt`).
4. Approve only when regressions are absent or explicitly justified with impact notes.
5. Keep thresholds as guidance defaults; tune per profile only with documented before/after evidence.

### Universal Invariants (apply to all profiles)
Run `scripts/qa_universal_invariants.py` on any output JSONL.
Hard fails:
- Any text chunk with `chunk_type=null`
- Any `bbox` value outside integer `[0,1000]`
- Any text chunk with empty content
- Any chunk missing `modality`

These must be zero. No profile-specific waivers.

### Canonical Single-Doc Strict Gate (Phase 4 Step 1, 2026-05-09)
For per-document validation use `scripts/qa_full_conversion.py --source-pdf`:

```
python scripts/qa_full_conversion.py output/<run>/ingestion.jsonl \
  --source-pdf data/<category>/<file>.pdf
```

It runs the full chain (`qa_conversion_audit`, `qa_universal_invariants`,
`qa_ingestion_hygiene`, `qa_semantic_fidelity`) plus blank-page-aware
deterministic checks. Reports `QA_PASS` / `QA_WARN` / `QA_FAIL`.

**Why `--source-pdf` is canonical:** without it, blank source pages
(common in books and magazines as verso/section dividers) get reported
as `MISSING_PAGES` failures even though they correctly have no chunks.
With `--source-pdf`, those pages are classified as `MISSING_PAGES_BLANK`
(info, not failure) by checking the source PDF for actual content.

