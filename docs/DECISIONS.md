# Decisions and Guardrails

## OCR Cascade Order
**Decision:** Docling → Tesseract → Doctr for layout-aware OCR.
**Rationale:** Keeps Docling layout awareness first, with progressive fallback when confidence is low.

## VLM Orchestration Protocol
- Changes to classifier/orchestrator must include impact analysis against the core test matrix.
- No modality-crossing fallbacks (scanned must stay scanned; digital must stay digital).
- No hardcoded document-specific rules.

## Anti-Patterns (Explicitly Forbidden)
- Overfitting to specific filenames.
- Forcing digital_magazine as a “safe” fallback for scans.
- Treating metadata as ground-truth instead of diagnostic evidence.
