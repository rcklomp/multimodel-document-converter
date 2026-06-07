# Quality Gates and Known Warnings

## Required Checks
- QA-CHECK-01: token balance variance within profile-specific gate tolerance (standard target: 10%).
- QA-CHECK-02: every `asset_ref.file_path` must exist on disk.
- QA-CHECK-03: `breadcrumb_path` depth must match `hierarchy.level`.
- QA-CHECK-04: all `bbox` values must be integer coordinates in range `[0,1000]` (REQ-COORD-01 compliance).
- QA-CHECK-05: image/table chunks must have `asset_ref`.
- REQ-COORD-02: `page_width`/`page_height` present for spatial metadata.

## R3 — Code-Indentation Fidelity (redesigned 2026-06-05)

R3 ("indentation fidelity + syntax preservation") is enforced by the shared
metric in `scripts/_code_quality.py`, imported by `qa_conversion_audit.py` (the
authoritative HARD gate) and `qa_semantic_fidelity.py` (advisory). It measures
code over the FULL population (`modality=="code"` + legacy `modality=="text"`
code), POSITIVELY identifies code (keywords / code punctuation / `>>>` REPL) so
extractor-mislabelled equations are excluded, and scores indentation ONLY on
judgeable code (multi-line `:`-suite / brace blocks; flat/single-statement and
REPL transcripts are exempt — no nesting to assess).

**Fidelity floor:** `0.90` (unchanged). **Gate verdict (Policy B):** at/above the
floor, or fewer than 3 judgeable chunks → PASS. Below the floor → each degraded
chunk is flagged, and the document hard-fails ONLY when judgeable-code density
`>= 0.04`; below that density it is an advisory WARN (code is always minority
content, so a prose document is not discarded over incidental mangled code).
The verdict is INDEPENDENT of `content_type`. Full rationale, anti-weakening
note, and the rejected alternatives: `docs/DECISIONS.md` "R3 Code-Indentation
Gate Redesign" and `docs/PLAN_R3_CODE_GATE_REDESIGN.md`.

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

### Multimodal Image/Table Hygiene (2026-06-08)
- This is a multimodal converter: IMAGE chunks are always retained. With
  `--vision-provider none` they ship as documented ID-only fallbacks
  (`vision_status=no_vlm`); the `IMAGE_NO_VLM` advisory (below) covers this.
  Image DESCRIPTION is a POST-conversion enrichment step, not conversion-time.
- Icon/glyph-class image regions (rendered <96px in BOTH dims AND <1.5KB) and
  empty-content tables are culled, but ALWAYS behind a page-coverage guard so no
  filter can manufacture `MISSING_PAGES`. An empty-content table that is the only
  chunk on its page is PROMOTED to IMAGE (its rendered crop) rather than dropped.
- New CONTENT-quality signals (furniture, garble, non-visual images, etc.) follow
  `AGENTS.md` AGENT-GATE-PROGRESSION: advisory-first, crucible-calibrated, frozen
  fixture, promoted to a hard gate only when earned. See `docs/PLAN_GATE_QUALITY_V1.md`.

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

### Advisory Warning Classes (Phase G, 2026-05-11)

The strict gate emits `QA_PASS_WITH_ADVISORIES` (a documented PASS
variant for v2.9.0-rc1 and v2.10 strict-gate accounting) when
**all** of the following hold:

1. Zero `FAIL`-severity issues.
2. Every `WARN`-severity issue's code is in the
   `_ALLOWED_ADVISORY_WARN_CODES` set in
   `scripts/qa_full_conversion.py`.
3. For the conditional code `VISION_HARD_FALLBACK_RATE`: every
   `hard_fallback` image chunk in the corpus must carry the F4
   sentinel `complex_asset_short_response_after_retry`. A
   non-F4-sentinelled hard_fallback indicates a real defect (asset
   missing, API failure, validator rejection) and continues to block
   PASS.

Allowed advisory codes and their rationale (per `docs/DECISIONS.md`
"Retrieval-Value Test"):

| Code | Rationale | Conditional? |
|---|---|---|
| `ASSET_TINY` | Publisher icon-class assets (<1 KB). Per the Retrieval-Value Test these are low-retrieval-value but valid; their presence in the JSONL is informational. | No |
| `PAGE_COUNT_UNKNOWN` | EPUB documents have no PDF-style page count. The EPUB lane provides a virtual page mapping via chunk order. | No |
| `SCRIPT_ADVISORY_FAIL` | `qa_semantic_fidelity.py` is advisory by design (exit 0); the authoritative R3 hard gate is `qa_conversion_audit.py` (Policy B, see "R3 — Code-Indentation Fidelity" above). A `SEMANTIC_FAIL` it prints — e.g. `code_indentation_fidelity` below 0.90 — is informational; the hard verdict comes from the audit. | No |
| `MISSING_CHAPTERS` | EPUB spine coverage found missing chapters, but every missing chapter is a contiguous leading/trailing low-content structural item (for example title page, cover, copyright/colophon stub, or blank wrapper) that Docling's HTML parser stripped before chunk emission. Internal gaps or content-bearing edge chapters remain `FAIL`. | Yes (edge + low-content structural only) |
| `VISION_HARD_FALLBACK_RATE` | Hard-fallback rate > 5 % when ALL hard_fallbacks have the F4 sentinel — documented "VLM legitimately can't describe this" cases (complex assets with terse responses after the Phase 3 detail-retry). | Yes (F4 condition above) |
| `IMAGE_NO_VLM` | The converter ran with `--vision-provider none`, so image chunks are retained as ID-only fallbacks (asset filename, no description). This is a multimodal converter — the image asset still ships; the missing description is a documented, user-chosen no-VLM state, not a defect. The CLI warns at run time that an image-bearing document converted without a VLM has limited multimodal retrieval value. | No |

The PASS variant is parallel to the SCAN0013 form-aware variant
(`GATE_PASS [form: ...]`) — both are explicit governance allowances
that let the strict gate accept documented edge cases without
weakening core checks.

**Adding a new advisory code requires:**
1. Adding the code to `_ALLOWED_ADVISORY_WARN_CODES`.
2. Documenting it in this section with explicit rationale tied to
   the Retrieval-Value Test or another DECISIONS.md governance rule.
3. A regression test in `tests/test_qa_advisory_promotion.py`
   pinning the positive (code is advisory) and negative (different
   code is not) classifications.
4. If conditional, a regression test verifying the condition fires.

**Anti-pattern explicitly forbidden:** adding a code to the
allowed-advisory set as a way to silence a `FAIL` that should be
fixed. The set is for code-classes whose `WARN` severity already
reflects "informational, not blocking" — the promotion makes the
gate's final status agree with the code-classification.

## Soak Format Judge Axis — Release-Window Pin (added 2026-05-20)

Separate from the strict gate above: the synthetic soak
(`scripts/synthetic_soak.py`) grades the **retrieved** chunk on three
axes — `relevance`, `format`, `faithfulness` — via Dashscope
`qwen-max` as judge. This is informational/release-tracking, NOT a
code-level pass/fail.

The **Format axis** is the only one with a tracked release-window pin
because v2.11.0 missed the original ≥96% pin by 6.2pp due to
coverage-reveal effects (the new dashscope embedder reaches
scanned/form chunks the v2.10 baseline never retrieved, and the judge
correctly grades their pre-existing OCR/scan imperfections).

| Window | Format pin | Source of truth |
|---|---:|---|
| **v2.11.0** (this release) | **≥ 85%** | 89.8% actual per `docs/archive/snapshots/QUALITY_SNAPSHOT_2026-05-20_v2.11_soak_qwen3.md` |
| **v2.11.1+** (recovery) | **≥ 95%** | Target after v2.11.x scanned/form chunk-content sanitization patch |
| **v2.12+** (revert) | **≥ 96%** | Original pin; reinstated after two consecutive recovery soaks pass |

Full rationale + 30-day rollback contract: `docs/DECISIONS.md`
"v2.11.0 Embedder Swap Executed — Format Gate Downgrade".

**Why this isn't in `_ALLOWED_ADVISORY_WARN_CODES`:** the soak Format
axis is not a `qa_full_conversion.py` warning code at all — different
measurement stack, different cadence (per-tag soak vs every PR), and
different unit (LLM-graded 0/1/2 vs deterministic invariant check).
The two systems coexist; this section exists so a reader searching
"Format" in this file finds both the strict-gate rules above and the
release-window soak pin.
