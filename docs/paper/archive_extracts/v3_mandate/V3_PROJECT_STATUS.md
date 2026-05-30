# V3 Project Status

**Updated:** 2026-05-29
**Phase:** Phase B (LLM Sanitization Baseline) COMPLETE — hypothesis
falsified. Transitioning to **Phase C: Vision-Native Extraction**.

---

## Completed Phases

### Phase A2 — V3 Engine Calibration — COMPLETE

UIR-in / chunk-out pipeline shipped behind the AST-audited contracts:

- `src/mmrag_v3/universal/document.py` — UIR types (`BoundingBox`
  enforces int `[0, 1000]`; `Element`, `UniversalPage`,
  `UniversalDocument`).
- `src/mmrag_v3/schema/ingestion_schema.py` — canonical
  `IngestionChunk` Pydantic model.
- `src/mmrag_v3/engines/docling.py` — sole Docling boundary. BOM
  (`﻿`) stripped in `_item_text`; bbox normalization is
  bit-identical to v2.16 `union_item_bboxes_for_uir`
  (`int(...)` truncation, `x_min=x0, x_max=x1, y_min=min(y0,y1),
  y_max=max(y0,y1)`); `EasyOcrOptions(use_gpu=False)` +
  `AcceleratorOptions(device=CPU)`.
- `src/mmrag_v3/chunking/uir_chunker.py` — pure UIR-native chunker
  (zero Docling imports). Replicates HybridChunker shape: TEXT
  elements buffer between IMAGE/TABLE boundaries; TABLE elements
  split per row; per-chunk BBox is the union of ONLY the elements
  whose char-spans overlap each emitted chunk's range (no
  whole-buffer page-wide BBoxes).
- `src/mmrag_v3/processor.py` — Docling-agnostic orchestrator.
- `tests/test_v3_security.py` — 13/13 passing; AST contracts intact.

### Phase B — LLM Sanitization Baseline — COMPLETE (hypothesis falsified)

Hypothesis under test: text-based LLM sanitization on top of
standard Docling extraction can reach v2.16 chunk-level parity
(Identity Gate aggregate < 5% delta) without re-introducing the
heuristic patches that v2.16 used.

**Verdict: FALSIFIED.** Identity Gate aggregate = **100.00%** over
3 reference documents.

### Identity Gate — Validated Architectural Finding (2026-05-29)

`scripts/run_identity_gate.py --no-sanitize`:

| Doc | v2 chunks | v3 chunks | delta | Root cause |
|---|---:|---:|---:|---|
| Bevestigingsmiddelen | 4 | 5 | 100.00% | Flush-grouping semantic loss (text-sim < 0.85 floor) |
| Form_betwistingsformulier | 8 | 0 | 100.00% | **Upstream SIGSEGV** in EasyOCR CRAFT detector |
| CarOK_voorraadtelling | 81 | 0 | 100.00% | **Upstream silent drop**: Docling cannot extract spreadsheet-PDF table text without OCR; with OCR it returns empty |

The three root causes are all **upstream of the V3 boundary**:

1. **Upstream SIGSEGV in EasyOCR/Docling on Apple Silicon.** The
   detector segfaults during model setup even with
   `EasyOcrOptions(use_gpu=False)`, `AcceleratorOptions(CPU)`,
   `OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`. Subprocess
   isolation (`scripts/_extract_doc_subprocess.py`) prevents the
   gate from aborting but does not recover the document.
2. **Upstream silent extraction failure on spreadsheet PDFs
   (CarOK) without OCR.** Docling emits one empty-content
   `TableItem` per page; the chunker correctly drops empty content.
   Enabling OCR routes back into cause #1.
3. **Flush-grouping semantic loss** on Bevestigingsmiddelen. The
   per-chunk BBox math fix landed (page-2 chunks now carry
   spatially distinct bboxes — left/middle/right columns; the
   page-wide union `(44,81,904,930)` is gone). The residual delta
   is the text-similarity floor (0.732 / 0.565 / 0.331 / 0.017) —
   the v2.16 HybridChunker grouped Docling items into different
   semantic chunks than our buffer-at-non-TEXT-boundary flush.
   Text-only LLM sanitization cannot reconstruct the original
   spatial grouping from joined text.

**Conclusion:** Text-based sanitization is insufficient.
Transitioning to **Phase C: Vision-Native Extraction**.

---

## Current Task

**Phase C — Vision-Native Extraction (initiating).**

The Phase C hypothesis is that a VLM operating directly on
rendered page images can produce a chunk shape that is robust to:
- macOS Apple Silicon EasyOCR detector segfaults (no torch-OCR
  cascade required),
- silent text-extraction failures on spreadsheet PDFs (the VLM
  sees the rendered cell text directly), and
- HybridChunker-style semantic grouping (the VLM can be prompted
  to emit per-column / per-row / per-section bbox+text records
  natively).

Phase C scope:
- [ ] VLM provider scaffolding under
  `src/mmrag_v3/engines/vlm.py` (returns `UniversalDocument`).
- [ ] Page-image rendering pipeline (PyMuPDF or `pypdfium2`).
- [ ] VLM prompt template emitting bbox-anchored chunk records
  directly into the canonical `IngestionChunk` shape.
- [ ] New Identity Gate threshold + fixture set tuned to
  VLM-native output.

---

## Active Constraints

- Python 3.10 only.
- Apple Silicon (MPS) preferred; OCR/CRAFT path explicitly avoided
  in Phase C — known SIGSEGV (validated 2026-05-29).
- Zero Docling imports in processor (AST audit).
- All engines return `UniversalDocument` (signature audit).
- All chunkers accept only `UniversalDocument` (signature audit).
- No backward shims, no v2x_to_v3_mapper.
- Phase A2 BBox math (v2.16-identical truncation) and Phase B
  per-chunk BBox attribution are LOCKED — Phase C must preserve
  them on any VLM-emitted bboxes.
