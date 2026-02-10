# Quality Gates and Known Warnings

## Required Checks
- QA-CHECK-01: token balance variance within tolerance (default 10%).
- QA-CHECK-05: image/table chunks must have `asset_ref`.
- REQ-COORD-02: `page_width`/`page_height` present for spatial metadata.

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

### Acceptance Workflow
1. Run `scripts/acceptance_technical_manual.sh` on representative docs.
2. Compare metrics against prior baseline (`_summary.txt`).
3. Approve only when regressions are absent or explicitly justified with impact notes.
4. Keep thresholds as guidance defaults; tune per profile only with documented before/after evidence.

## Known Warnings (Observed)

### Layout-aware OCR: `ocrmac` NoneType iterable
- Symptom: Docling OCR stage fails intermittently with `NoneType` from `ocrmac`.
- Behavior: Pipeline falls back to layout-aware OCR and completes.
- Status: Not blocking metadata propagation; track separately if OCR quality is a concern.

### Token variance warnings (QA-CHECK-01)
- Symptom: token variance around -15% seen in some runs.
- Behavior: Logs critical warning but continues (unless strict mode enabled).
- Action: Investigate only if content loss is suspected; not tied to metadata propagation.

### imagehash missing
- Symptom: `imagehash not installed, using fallback hash`.
- Behavior: Deduplication still runs with fallback hashing.
- Action: Optional dependency for stronger dedupe; not required for correctness.
