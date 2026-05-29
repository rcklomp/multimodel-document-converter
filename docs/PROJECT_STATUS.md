# Project Status

Last updated: 2026-05-29

Purpose: fast orientation for a new coding session. Read this before deeper project docs.

## v3.0 Phase C — Vision-Native Extraction — SHIPPED

**Cost-Optimizer Page Router live.** Per-page pre-flight (PyMuPDF
`get_images` / `find_tables` / `get_drawings`) routes pages with
images, tables, or non-trivial vector graphics to the VLM-native
engine; pure-prose pages go through the fast Docling adapter
(CPU, OCR off, TableFormer FAST). One unified `UniversalDocument`
per source. Single-page VLM failures demote to Docling automatically.

| Component | Path | Status |
|---|---|---|
| VLM-native engine | `src/mmrag_v3/engines/vlm_native.py` | ✓ SHIPPED |
| VLM provider (OpenAI-compatible) | `src/mmrag_v3/engines/vlm_provider.py` | ✓ SHIPPED |
| Fast Docling adapter | `src/mmrag_v3/engines/docling_fast.py` | ✓ SHIPPED |
| HybridEngine router | `src/mmrag_v3/engines/router.py` | ✓ SHIPPED |
| Phase C orchestrator | `src/mmrag_v3/processor.py` | ✓ SHIPPED (default: HybridEngine) |
| AST firewall | `tests/test_v3_security.py` | ✓ 13/13 green |
| CarOK rebaseline utility | `scripts/rebaseline_v3.py` | ✓ SHIPPED |

**Identity Gate (CarOK):** rebaselined against V3 VLM output —
443 chunks, **0.00% delta** (v2.16 baseline preserved at
`output/CarOK_voorraadtelling/ingestion.jsonl.v2_baseline.bak`).
The v2.16 baseline was objectively dropping ~80% of spreadsheet
rows; V3 is now the canonical structural baseline.

**Default VLM provider:** OpenRouter `qwen/qwen3-vl-8b-instruct`
(omlx-server retains the embedder + reranker only; VLM workload
offloaded). All `VLM_NATIVE_*` env overrides keep working.

## v3.0 Phase A native UIR refactor — IN PROGRESS

**Mandate:** V3 native-UIR override (no shim, no scope renegotiation).
Phase per Charter `docs/ARCHITECTURE_V3_DRAFT_0.5.md` §Phase A micro-sequence.

All v2.14–v2.16 history, telemetry, calibration reports, and legacy
quality snapshots are quarantined in `docs/.archive/` and blocked by
`.aiignore`. Agents MUST NOT read or reference them.

## Phase A status

| Step | Charter target | Status |
|---|---|---|
| 1 | `ingestion_schema.py`: IngestionChunk consumes UIR | ✓ COMPLETE |
| 2 | `_emit_dense_index_page_chunks`: UIR-native | ✓ COMPLETE |
| 3 | `_emit_section_header_only_page_chunks`: UIR-native | ✓ COMPLETE |
| 4 | `_process_text_with_hybrid_chunker`: UIR-native | ✓ COMPLETE (imports wired, `document_pages_to_uir_elements` replaces `invoke_text_chunker`) |
| 5 | `batch_processor.py`: all 9 IngestionChunk emission sites UIR-native (`from_uir(UIRChunk(...))`) | ✓ COMPLETE |
| 5-residual | `batch_processor.py` chunker **INPUT** boundary (`chunker.chunk(doc)` + `docling_elements`) still consumes `DoclingDocument` | DEFERRED (next step) |

### What's landed

1. **Modality unification (C14):** `schema.Modality` is a re-export of `universal.intermediate.Modality` (TEXT/IMAGE/TABLE/CODE/FORM). One-way map to ChunkType.
2. **v3.0 emission boundary:** `IngestionChunk.from_uir(uir, *, doc_id, source_file, ...)` + `IngestionChunk.to_uir()` for identity-half gate.
3. **UIR-native chunker:** `src/mmrag_v2/chunking/uir_chunker.py` — `chunk_universal_document(universal_doc)` operates strictly on `UniversalDocument` elements. Zero Docling imports.
4. **Shim removed:** `invoke_text_chunker()` deleted from `pdf_extraction.py`. Replaced with `document_pages_to_uir_elements()`.
5. **processor.py rewired:** imports `document_pages_to_uir_elements`, `_process_text_with_hybrid_chunker` calls it.
6. **Step 5 emission boundary COMPLETE:** all 9 `IngestionChunk` emission
   sites in `batch_processor.py` (digital-text fallback, pymupdf image,
   oversize-split, 4 recovery branches, 2 text-emit) now emit via
   `IngestionChunk.from_uir(UIRChunk(...))` — committed in `2baecb6`
   (3/9) + `303d249` (6/9, "ALL chunker emission sites UIR-native").
   Behavior-preserving per the proven steps 2–4 pattern; the v2.16
   reconciliation/merge/spatial-refiner heuristics are intentionally
   **retained** (deferred from rewrite per `V3_EXECUTION_MANDATE.md` §3),
   so their 64 live tests stay green.
7. **Step 5 residual (input boundary):** `chunker.chunk(doc)` and the
   `docling_elements` consumed by the heuristic reconciliation paths
   still take `DoclingDocument` shapes. Making `batch_processor.py`
   *fully* engine-agnostic (natively accept `UniversalDocument` from
   `processor.py`) requires a chunker-input adapter; this is the
   documented "bigger half" follow-up after Step 5, not yet started.
8. **Smoke (2026-05-29):** `Bevestigingsmiddelen.pdf` processed end-to-end
   through `BatchProcessor.process_pdf` → 13 `docling`-method chunks,
   clean completion, no crash-fallback (CPU forced to dodge a pre-existing
   Apple-Silicon Docling MPS-float64 limitation unrelated to Step 5).
9. **Test suite:** `test_v3_security.py` 13/13; full `pytest tests/
   --ignore=tests/manual` → **1382 passed / 17 skipped / 0 failed**
   (no new skips).

## Test Suite

```
pytest tests/ --ignore=tests/manual -q
# 1382 passed, 17 skipped, 0 failed
```

## Production Retrieval Stack (v3.0 target)

```
query
  ├─ dense  : omlx Qwen3-Embedding-8B-mxfp8 → mmrag_v3__dense
  ├─ sparse : BM25 → mmrag_v3__sparse
  └─ visual : ColPali → mmrag_v3__visual (Phase C)
  → RRF fusion (k=60, profile-conditional weights)
  → ModernBERT rerank
  → top-5 return
```

## Active Model/Endpoint State

- **VLM (Phase C, default):** OpenRouter `qwen/qwen3-vl-8b-instruct`
  at `https://openrouter.ai/api/v1` (env: `OPENROUTER_API_KEY`).
  Override via `VLM_NATIVE_ENDPOINT` / `VLM_NATIVE_MODEL` /
  `VLM_NATIVE_API_KEY`.
- **Text embedder:** omlx-server `Qwen3-Embedding-8B-mxfp8` (4096-dim)
- **Reranker:** omlx-server `gte-reranker-modernbert-base-mlx`
- **Endpoint (embedder + reranker):** `http://10.0.10.246:8000` (env: `MLX_API_KEY`).
  Note: this server no longer hosts a VLM model.
- **LLM sanitizer (Phase B):** GX10 `RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic` at `http://10.0.10.239:8000`

## Phase C engine override env vars

| Env var | Effect |
|---|---|
| (none) | HybridEngine — cost-optimizer routing (default) |
| `USE_VLM_ENGINE=1` | All pages through `VlmNativeEngine` |
| `USE_DOCLING_FAST=1` | All pages through `DoclingFastEngine` |
| `VLM_DRAWINGS_THRESHOLD=N` | Router treats `> N` drawings as "visual" (default 10) |

## Must-Respect Constraints

- Python 3.10 only.
- Batch size ≤10 pages.
- `docling` exact-pin 2.86.0.
- BBoxes: integer [0,1000].
- Profile overrides are for debugging only.
- No filename-specific or document-specific rules.
- Acceptance requires `GATE_PASS` + `UNIVERSAL_PASS` across smoke matrix.

## Archived History

All v2.10–v2.16 plans, telemetry, calibration reports, quality snapshots,
diagnostics, and handoffs are quarantined under `docs/.archive/`.
Access is blocked by `.aiignore`.