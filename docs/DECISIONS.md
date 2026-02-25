# Decisions and Guardrails

## OCR Cascade Order
**Decision:** Docling → Tesseract → Doctr for layout-aware OCR.
**Rationale:** Keeps Docling layout awareness first, with progressive fallback when confidence is low.

## OCR Confidence Threshold Governance
**Decision:** The default layout-aware OCR trigger threshold is **0.70** (`--ocr-confidence-threshold` default).

**Rationale:**
- The threshold is an empirical quality lever, not a fixed architectural invariant.
- The legacy 0.90 expectation was too aggressive in practice and increased unnecessary OCR escalation.
- Acceptance tuning showed 0.70 gives better balance between extraction fidelity and over-triggering.

**Operationalization:**
- SRS defines behavior and default, while this document records the decision basis and tuning policy.
- Any change to the default threshold must include before/after acceptance evidence.
- Validate changes with representative acceptance runs and QA outputs before adoption.

## VLM Orchestration Protocol
- Changes to classifier/orchestrator must include impact analysis against the core test matrix.
- No modality-crossing fallbacks (scanned must stay scanned; digital must stay digital).
- No hardcoded document-specific rules.

## Anti-Patterns (Explicitly Forbidden)
- Overfitting to specific filenames.
- Forcing digital_magazine as a “safe” fallback for scans.
- Treating metadata as ground-truth instead of diagnostic evidence.

## Chunk Size Governance
**Decision:** Chunk length is governed per profile and verified with acceptance metrics; no universal hard min/max.

**Rationale:**
- Different modalities and document classes need different chunking behavior.
- A single global threshold causes regressions (either fragmentation or oversized chunks).
- Quality must be demonstrated with repeatable benchmarks, not assumed from one document.

**Operationalization:**
- Use representative acceptance runs (e.g., `scripts/acceptance_technical_manual.sh`).
- Track both structural hygiene (`text_short_<30`, `text_long_>1500`, `infix_strict`) and coverage (`QA-CHECK-01`).
- Document any threshold/range change with baseline comparison in the run summary.
