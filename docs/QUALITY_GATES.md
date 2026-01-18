# Quality Gates and Known Warnings

## Required Checks
- QA-CHECK-01: token balance variance within tolerance (default 10%).
- QA-CHECK-05: image/table chunks must have `asset_ref`.
- REQ-COORD-02: `page_width`/`page_height` present for spatial metadata.

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
