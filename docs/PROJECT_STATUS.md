# Project Status

Last updated: 2026-06-01

## V3 extraction hardening + pre-Crucible boundary audit (2026-06-01)

Four fixes landed on the V3 vision-native path (all UNCOMMITTED; HEAD still
`00ac15d`). Full suite 1355 passed / 99 skipped / 0 failed; offline
`SMOKE_PRODUCTION_PASS`. See `HANDOVER_2026-06-01.md` for the full picture.

1. **Schema compliance (the crucible-killer).** VLM-native IMAGE/TABLE chunks
   reached `from_uir` with no `asset_ref` and failed QA-CHECK-05 (0/18 baselines
   in the original soak). Fixed by a shared `universal/asset_materializer.py`
   that crops bbox regions to PNG + sets `asset_ref` (used by BOTH batch and soak
   so they cannot diverge), plus producer-side `visual_description` truncation in
   `from_uir`. Gates NOT weakened. Proven on real M5 VLM output (doc `0013`).
2. **Crop-audit instrumentation.** Per-crop drift signals (full-page-fallback /
   edge-clamp / low-information, reusing the v2.9 blank def) + doc-level
   `QA_WARN_CROP_DRIFT` gate at 15%, recorded in meta.json.
3. **VLM circuit breaker.** Infra/transport errors raise `VlmInfraError` and
   hard-fail instead of silently falling back to Docling; the soak harness
   defaults to a pause-and-poll breaker (poll 60s, hard-fail after 30min down or
   5 per-doc infra failures); `--strict-breaker` for attended runs.
4. **VLM code/form (Charter-respecting).** `ElementType` stays 3 (Charter §7.1,
   contract test unchanged); code/form are smuggled as TEXT + a
   `promoted_modality` tag and promoted to `Modality.CODE`/`FORM` in the chunker;
   unknown types degrade to TEXT with a warning instead of crashing the page.

**3-boundary audit (read-only):** serialization and legacy post-processing are
CLEAN for code/form (indentation survives; refiners modality-gated; soak runs
none; `docling_postprocess` not wired into V3). The one REAL gap is the router
pre-flight (`router.py::_classify_page`): object-presence only, no
code/text-complexity heuristic - on AIOS, 10 of 35 pages route to Docling. This
is the deferred "measure before fixing" item.

**Next step:** the AIOS single-doc smoke (M5 is up) - both the code-heavy schema
proof and the Boundary-1 measurement. The Crucible Subset has NOT been run.
Telemetry-1 (V3-vs-V2.16 deltas) is blocked on missing V2.16 baselines.

## PLAN_V3.1 reconvergence — P1 + P2 landed (2026-05-31)

**P1 (collapse to one extraction path) — DONE.** `scripts/v3_batch_ingest.py`
+ `scripts/rebaseline_v3.py` repointed off the retired `v3_execution_root`
sandbox chunker onto the production shipping path (`HybridEngine.extract` →
`uir_chunker.chunk_universal_document` → `IngestionChunk.from_uir`). The v2→v3
sandbox translation + importlib namespace trick are gone. `grep -rn
v3_execution_root scripts/` is empty; the baseline tool runs offline
(`USE_DOCLING_FAST=1`) and emits IngestionChunk JSONL.

**P2 (HEADING coverage, UIR-native) — DONE.** New
`uir_chunker._assign_headings` pass sets `parent_heading` + `breadcrumb_path`
per text chunk by precedence (in-page heading → cross-page carry-forward →
PyMuPDF TOC fallback), with `breadcrumb_path` built from the TOC hierarchy. The
TOC is extracted document-wide (`batch_processor._extract_toc_headings`) and
threaded into the chunker as plain data, shifted to batch-local pages by the
new `_toc_for_batch` helper (no Docling, no resurrected batch_processor
methods). `breadcrumb_path` is a new additive field on `UIRChunk`, consumed by
`IngestionChunk.from_uir`. Result on an academic doc with bookmarks (RAG guide,
pp.23-32): HEADING coverage **68% → 100%**, breadcrumb 0% → 100%, qa
`HEADING: PASS` / `AUDIT_PASS`. `tests/test_ocr_path_heading_propagation.py`
RESTORED (un-skipped, 76 green incl. 7 new `_assign_headings` contracts). Two
DECISIONS.md entries added: the OCR-lane production-wiring pin retirement and
the short-document HEADING-gate skip (a born-digital doc with no bookmarks +
no headings, e.g. `Bevestigingsmiddelen.pdf`, legitimately has no hierarchy;
the gate is corrected to SKIP that class rather than fabricate headings).

P3/P4/P5 NOT started (need re-baselining + user judgment). See
`docs/PLAN_V3.1_PIPELINE_RECONVERGENCE.md`.

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

## v3.0 Phase A native UIR refactor — COMPLETE

**Mandate:** V3 native-UIR override (no shim, no scope renegotiation).
Phase per Charter `docs/ARCHITECTURE_V3_DRAFT_0.5.md` §Phase A micro-sequence.

**Closed 2026-05-29.** `batch_processor.py` is engine-agnostic on both the
emission boundary (all chunks via `IngestionChunk.from_uir`) and the input
boundary (extraction delegated to `mmrag_v3.extract()` → `UniversalDocument`
→ `chunk_universal_document()`). Zero docling imports remain in
`batch_processor.py`; the legacy OCR/layout lanes (1384 lines) are deleted.
Two nested chunk-hygiene heuristics remain woven into kept quality-filter
methods and are formally carried as Phase B technical debt (see below).

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
| 5 | `batch_processor.py`: all 9 IngestionChunk emission sites UIR-native + INPUT boundary decoupled to `mmrag_v3.extract()`; OCR/layout lanes deleted (1384 lines); DoclingPdfAdapter severed (zero docling imports); top-level reconcile/heading/front-matter heuristics stripped | ✓ COMPLETE (`813b9ba`) |

## Phase B Technical Debt

Carried forward from Phase A close (2026-05-29). **Disposition update
(2026-05-31):** the original plan to retire these via a "Phase B LLM-Sanitization
Layer" is void - that layer's hypothesis was falsified (see
`docs/V3_DEFERRED_TESTS.md`). Per `docs/V3_EXECUTION_MANDATE.md` §3 these are now
DISPOSITIONED deferrals (adopt / restore / delete-by-decision), owned by
`docs/PLAN_V3.1_PIPELINE_RECONVERGENCE.md` P3, not open-ended debt. The
control-doc contradictions that this exposed are catalogued in
`docs/AUDIT_CONTROL_DOCS_2026-05-31.md`. Accepted as deliberate, owned debt to
preserve green gates and the `AGENT-SPATIAL-20` invariant; NOT to be silently
dropped.

1. **`_apply_spatial_refiner` — ADOPTED + GUARDED (R6 closed 2026-06-01).**
   Still executes live on the V3 path via `_apply_vertical_proximity_merger`;
   governed by `AGENT-SPATIAL-20` (single 20-unit vertical threshold). No longer
   debt: its invariant is now an executable guard,
   `tests/test_spatial_refiner_agent_spatial_20.py` (7 tests, mutation-verified).
   See `docs/DECISIONS.md` "R6 closed - AGENT-SPATIAL-20 is now an executable
   guard".
2. **`_merge_mid_sentence_chunks` — ADOPTED (P3 2026-06-01).** Still executes via
   `_apply_quality_filters`; its module `test_cross_chunk_semantic_stitching.py`
   was RESTORED (un-skipped, 9/9) when the orphaned final-boundary-repair bridge
   was re-wired into `process_pdf`. See `docs/DECISIONS.md` "Phase A orphaned the
   final-boundary-repair bridge - RE-WIRED".

Both heuristics are now adopted and test-guarded (no longer deferred). The
remaining deferred surface is the 3-module legacy V2-Docling-lane cluster
(`docs/V3_DEFERRED_TESTS.md`), dispositioned to the Phase 6 lane retirement.

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
7. **Step 5 INPUT boundary DECOUPLED (`813b9ba`):**
   `_process_single_batch` now delegates extraction to `mmrag_v3.extract()`
   (HybridEngine router → `UniversalDocument`) → `chunk_universal_document()`
   → `IngestionChunk.from_uir()`, with batch `page_offset` projection.
   The legacy OCR/layout lanes (`_process_batch_layout_aware`,
   `_extract_docling_layout_elements`, `_process_page_layout_aware`,
   `_classify_page`, `_render_page_to_image`, `_nuclear_code_fix`,
   `_flat_code_ocr_rescue`, `_flat_code_rescue_legacy_pass`) are DELETED
   (1384 lines) and `DoclingPdfAdapter` is severed — `batch_processor.py`
   has **zero docling imports** (11,109 → 9,595 lines). Top-level
   reconcile/heading/front-matter heuristics removed from finalize. New
   guard: `tests/test_v3_integration.py` (offline DoclingFast, asserts
   `extraction_method == "uir_native_chunker"`, in the mandatory block).
   **Caveat — offline floor:** with `USE_DOCLING_FAST=1` (no VLM), the
   multi-doc smoke shows prose extracts clean V3 chunks, but forms /
   figure-heavy papers fall to the kept PyMuPDF recovery net (chunks
   tagged `recovery_*`, still emitted via `from_uir`) and **Form_0013
   yields 0 chunks offline**. Production routes those to the VLM engine
   (proven in V3_OVERNIGHT_REPORT: Form_0013→11, IRJET→65) but the VLM
   path is currently untestable (OpenRouter weekly budget exhausted).
8. **Smoke (2026-05-29):** `Bevestigingsmiddelen.pdf` processed end-to-end
   through `BatchProcessor.process_pdf` → 13 `docling`-method chunks,
   clean completion, no crash-fallback (CPU forced to dodge a pre-existing
   Apple-Silicon Docling MPS-float64 limitation unrelated to Step 5).
9. **Test suite (after input-boundary decouple, `813b9ba`):**
   `test_v3_security.py` 13/13; full `USE_DOCLING_FAST=1 pytest tests/
   --ignore=tests/manual` → **1218 passed / 182 skipped / 0 failed**.
   The 165 new skips are 6 deferred test modules (heuristic / OCR-lane /
   adapter) marked `@pytest.mark.skip(reason="V3_DEFERRED - Legacy
   Heuristic")` — every test in them pinned a stripped behavior.
   Pre-decouple committed baseline was 1382 / 17 / 0.

## Test Suite

```
# After input-boundary decouple (813b9ba):
USE_DOCLING_FAST=1 pytest tests/ --ignore=tests/manual -q
# 1218 passed, 182 skipped, 0 failed
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