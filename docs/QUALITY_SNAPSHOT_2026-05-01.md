# Quality Snapshot 2026-05-01

## Milestone 1: Stabilize Extraction First

Status: `complete`

Closes the Milestone 1 workstream begun 2026-04-30. RAG Guide is unblocked, the per-element chunker guard is in, and Ayeva is re-converted.

Evidence:
- Class: tracked + local-run
- Command: `conda run -n mmrag-v2 python -m pytest tests/test_classifier_fallback.py tests/test_pdf_conversion_plan.py tests/test_chunker_guard.py tests/test_corruption_quarantine.py tests/test_finalization_bridge.py tests/test_blank_asset_quarantine.py -q`
- Input: Milestone 1 focused tests (classifier fallback, plan + bridge, chunker guards, corruption / blank-asset quarantine, finalization bridge).
- Output: terminal result
- Result: `81 passed`
- Tracked: yes
- Limitations: focused tests; corpus quality proven via probes + smoke below.

Evidence:
- Class: tracked + local-run
- Command: `conda run -n mmrag-v2 python -m pytest -q`
- Input: tracked full unit suite
- Output: terminal result
- Result: `467 passed, 1 skipped` (Milestone 1 close); after Milestone 2 close: `484 passed, 1 skipped, 0 failed`.
- Tracked: yes
- Limitations: unit suite does not prove corpus-level conversion behavior by itself.

Evidence:
- Class: local-run
- Command: `conda run -n mmrag-v2 bash scripts/smoke_multiprofile.sh`
- Input: smoke matrix in `scripts/smoke_multiprofile.sh`
- Output: `output/smoke_multiprofile_20260501_105836/_summary.txt`
- Result: 10/10 rows `GATE_PASS` + `UNIVERSAL_PASS`. Zero regressions vs. the 2026-04-30 baseline.
- Tracked: no
- Limitations: generated artifacts ignored; rerun to reproduce.

## RAG Guide (Kimothi) — previously blocked

Status: `complete`

Evidence:
- Class: local-run
- Command: `conda run -n mmrag-v2 python -m mmrag_v2.cli process "data/technical_manual/A Simple Guide to Retrieval Augmented Generation Kimothi A. 2025.pdf" --output-dir output/probe_rag_guide_guard --batch-size 10 --vision-provider none --no-refiner --no-cache`
- Input: 258-page technical manual that previously hung on batch 25 at 7200s outer timeout.
- Output: `output/probe_rag_guide_guard/ingestion.jsonl`
- Result: full conversion in ~5 min, 680 chunks (text=559, image=99, table=22). `qa_conversion_audit.py` AUDIT_PASS, `qa_universal_invariants.py` UNIVERSAL_PASS, `evaluate_technical_manual_gates.py` GATE_PASS (`infix_strict=0`). CODE [mixed_prose]: `indentation_fidelity=0.91`. Pathological batch 25 (pages 241-250) detected via SIGALRM 120s and fell back to element-by-element chunking, producing 40 + 10 chunks for batches 25 and 26.
- Tracked: no
- Limitations: per-batch SIGALRM is the rescuer; the per-element guard added today is defense-in-depth and did not fire for this document because the 1M-token sequence is built inside HybridChunker, not from a single Docling element.

## Ayeva Python Patterns — re-conversion

Status: `complete`

Evidence:
- Class: local-run
- Command: `conda run -n mmrag-v2 python -m mmrag_v2.cli process "data/technical_manual/Ayeva K. Mastering Python Design Patterns...essential Python patterns...3ed 2024.pdf" --output-dir output/ayeva_qa_20260501 --batch-size 3 --enable-ocr --ocr-mode auto --enable-doctr --ocr-confidence-threshold 0.4 --vision-provider none --no-refiner --no-cache`
- Input: 296-page code-heavy technical manual previously failing CODE + TEXT gates.
- Output: `output/ayeva_qa_20260501/ingestion.jsonl`
- Result: 627 chunks (text=603, image=23, table=1). AUDIT_PASS + GATE_PASS + UNIVERSAL_PASS. CODE [code_heavy]: `indentation_fidelity=0.93` (was 0.22 in stale `output/Ayeva_Python_Patterns/`), `infix_strict=0` (was 2). Profile reclassified `digital_magazine` → `technical_manual` after Milestone 1 classifier fix.
- Tracked: no
- Limitations: 8/603 text chunks have missing bbox (advisory; UNIVERSAL_PASS counts only invalid bboxes).

## Per-Element HybridChunker Guard

Status: `complete`

Adds `max_chunker_per_element_chars: int = 100_000` to `PdfConversionPlan`. Defense-in-depth for documents where Docling emits a single mega-element that would feed a multi-million-token sequence to the sentence-transformer tokenizer.

Evidence:
- Class: tracked
- Command: `conda run -n mmrag-v2 python -m pytest tests/test_chunker_guard.py tests/test_pdf_conversion_plan.py -q`
- Input: focused chunker-guard + plan-bridge tests (added 4 new in `test_chunker_guard.py`, propagation assertions in `test_pdf_conversion_plan.py`).
- Output: terminal result
- Result: `47 passed`
- Tracked: yes
- Limitations: the per-element guard did not fire for the RAG Guide pathology; the SIGALRM 120s did. The guard remains useful for the documented "one giant Docling text element" case.

## Milestone 2: Plan Control Plane

Status: `complete`

`PdfConversionPlan` promoted to a typed policy object. New fields: `extraction_route` ∈ {`native_digital`, `scanned_book`, `image_heavy_magazine`, `technical_manual`} (`scanned_degraded` is a valid explicit-override only); `hybrid_chunker_enabled`; `allow_page_level_visuals`; `asset_validation_policy` (`drop` | `keep` | `quarantine`); `corruption_recovery_policy` (`quarantine` | `keep` | `recover`). Legacy bools (`drop_blank_assets`, `quarantine_corrupted_chunks`) preserved as derived `@property` bridges. `__post_init__` validates policy values.

Evidence:
- Class: tracked + local-run
- Command: `conda run -n mmrag-v2 python -m pytest tests/test_pdf_conversion_plan.py tests/test_chunker_guard.py -q`
- Input: focused plan + chunker-guard suite (15 new tests across Milestone 2 work: 9 for the typed policy fields and route migration, 6 for derived-property bridges + `__post_init__` rejection).
- Output: terminal result
- Result: `64 passed`
- Tracked: yes
- Limitations: focused tests; corpus quality proven via probe + smoke below.

Evidence:
- Class: local-run
- Command: `conda run -n mmrag-v2 bash scripts/smoke_multiprofile.sh`
- Input: smoke matrix in `scripts/smoke_multiprofile.sh`
- Output: `output/smoke_multiprofile_20260501_120514/_summary.txt`
- Result: 10/10 rows `GATE_PASS` + `UNIVERSAL_PASS`. Zero regressions vs. the Milestone 1 close baseline.
- Tracked: no
- Limitations: generated artifacts ignored; rerun to reproduce.

Evidence:
- Class: local-run
- Command: `conda run -n mmrag-v2 python -m mmrag_v2.cli process "data/technical_manual/A Simple Guide to Retrieval Augmented Generation Kimothi A. 2025.pdf" --output-dir output/probe_milestone2_rag_guide --batch-size 10 --vision-provider none --no-refiner --no-cache`
- Input: 258-page technical manual; same regression target as Milestone 1.
- Output: `output/probe_milestone2_rag_guide/ingestion.jsonl`
- Result: AUDIT_PASS + GATE_PASS + UNIVERSAL_PASS. 559 text chunks, `infix_strict=0`. Confirms vocabulary rename and policy-derivation changes do not regress full-doc conversion.
- Tracked: no
- Limitations: smoke + single probe is sufficient evidence for a policy-shape change; not a full corpus rerun.

## Static Guards (regression check)

Status: `passing`

Evidence:
- Class: tracked
- Command: `conda run -n mmrag-v2 python -m pytest tests/test_pdf_conversion_plan.py::test_no_pipeline_options_construction_outside_adapter tests/test_pdf_conversion_plan.py::test_no_production_docling_imports_outside_adapter -q`
- Result: `2 passed`
- Tracked: yes
- Limitations: AST-level guards; do not protect against constructor calls reflected via getattr.

## Refactor Boundary Closeout

Status: `complete`

Closes Execution Order steps 6 and 7 of `docs/PLAN_V2.7_DOCUMENT_UNDERSTANDING.md` Section 5. Removes the `[V2.8-COMPAT]` `BatchProcessor.set_intelligence_metadata` deprecated path and the dead CLI fallback that called it; locks the typed-policy plumbing with one consolidated end-to-end bridge test that fails the day a new `PdfConversionPlan` field forgets a downstream wiring. No new public API, no new CLI flag, no new typed field, no new construction site for `PdfPipelineOptions` / `DocumentConverter`.

Removed:
- `BatchProcessor.set_intelligence_metadata(intelligence_metadata)` and its bookkeeping. `_intelligence_metadata` is now assigned exclusively from `plan.chunk_factory_metadata()` (via `set_conversion_plan`) or `plan.to_intelligence_metadata()` — no caller can overwrite raw keys.
- `cli.py` dead branch `if conversion_plan is None: processor.set_intelligence_metadata(intelligence_metadata)` inside the PDF-only `use_batching` path. `conversion_plan` is built unconditionally for `is_pdf` before the BatchProcessor branch, so the guarded fallback was unreachable.
- Tests that existed only to cover the deprecated path (5 total): `test_legacy_batch_metadata_builds_adapter_plan` (covered by `test_batch_with_plan_uses_adapter`); `test_encoding_corrupt_magazine_does_not_enable` (covered by `test_plan_encoding_alone_no_code`); `test_structural_flags_propagate` (pure deprecated-API test); `test_batch_legacy_path_passes_needs_code_enrichment_to_processor` and `test_batch_legacy_path_passes_has_encoding_corruption_to_processor` (both covered by `test_batch_plan_to_processor_all_flags_bridge`).

Added:
- `tests/test_pdf_conversion_plan.py::test_all_typed_policy_fields_round_trip_full_chain` — drift insurance. Builds a `PdfConversionPlan` with non-default values for every Milestone 2 typed policy field (`extraction_route="technical_manual"`, `hybrid_chunker_enabled=False`, `max_chunker_input_chars=321_000`, `max_chunker_per_element_chars=42_000`, `allow_page_level_visuals=True`, `asset_validation_policy="quarantine"`, `corruption_recovery_policy="recover"`), feeds it through `BatchProcessor.set_conversion_plan(...)`, runs `_process_single_batch(...)` against monkeypatched `DoclingPdfAdapter` + `V2DocumentProcessor`, and asserts every typed value reaches every downstream object unchanged. One bridge test that fails loudly the day a new typed field forgets a boundary, replacing the temptation to add many narrow ones later.

Evidence:
- Class: tracked
- Command: `conda run -n mmrag-v2 python -m pytest tests/test_pdf_conversion_plan.py::test_no_pipeline_options_construction_outside_adapter tests/test_pdf_conversion_plan.py::test_no_production_docling_imports_outside_adapter -q`
- Input: AST-level static guards covering all production Python files under `src/mmrag_v2/`.
- Output: terminal result
- Result: `2 passed`
- Tracked: yes
- Limitations: AST-level guards; do not protect against constructor calls reflected via getattr.

Evidence:
- Class: tracked
- Command: `conda run -n mmrag-v2 python -m pytest tests/test_pdf_conversion_plan.py tests/test_chunker_guard.py tests/test_corruption_quarantine.py tests/test_blank_asset_quarantine.py tests/test_finalization_bridge.py -q`
- Input: focused boundary suite (plan + chunker + quarantine + finalization bridges).
- Output: terminal result
- Result: `93 passed`
- Tracked: yes
- Limitations: focused tests; corpus quality proven via probe + smoke below.

Evidence:
- Class: tracked + local-run
- Command: `conda run -n mmrag-v2 python -m pytest -q`
- Input: tracked full unit suite after deletion of 5 deprecated-path tests and addition of `test_all_typed_policy_fields_round_trip_full_chain`.
- Output: terminal result
- Result: `480 passed, 1 skipped, 0 failed` (was 484 passed; net change −5 deprecated +1 round-trip = 480).
- Tracked: yes
- Limitations: unit suite does not prove corpus-level conversion behavior by itself.

Evidence:
- Class: local-run
- Command: `conda run -n mmrag-v2 python -m mmrag_v2.cli process "data/technical_manual/A Simple Guide to Retrieval Augmented Generation Kimothi A. 2025.pdf" --output-dir output/probe_boundary_closeout_rag_guide --batch-size 10 --vision-provider none --no-refiner --no-cache`
- Input: 258-page technical manual; same regression target as Milestone 1 + Milestone 2.
- Output: `output/probe_boundary_closeout_rag_guide/ingestion.jsonl`
- Result: AUDIT_PASS + UNIVERSAL_PASS. 680 chunks (text=559, image=99, table=22), CODE [mixed_prose] `indentation_fidelity=0.91`, `infix_strict=0`. Identical structural shape to Milestone 1+2 baselines (`output/probe_rag_guide_guard/`, `output/probe_milestone2_rag_guide/`). Confirms `set_intelligence_metadata` removal does not regress full-doc conversion.
- Tracked: no
- Limitations: generated artifacts ignored; rerun to reproduce.

Evidence:
- Class: local-run
- Command: `conda run -n mmrag-v2 bash scripts/smoke_multiprofile.sh`
- Input: smoke matrix in `scripts/smoke_multiprofile.sh`
- Output: `output/smoke_multiprofile_20260501_134909/_summary.txt`
- Result: 10/10 rows `GATE_PASS` + `UNIVERSAL_PASS`, including the `Greenhouse Design and Control by Pedro Ponce` blind-test document (`AGENT-VAL-01`). Zero regressions vs. the Milestone 2 close baseline.
- Tracked: no
- Limitations: generated artifacts ignored; rerun to reproduce.

## Contextual Retrieval (Anthropic approach)

Status: `complete`

Closes Combined Plan point #4 and `docs/PLAN_V2.7_DOCUMENT_UNDERSTANDING.md` Feature 6. Adds an embed-time builder (`mmrag_v2.chunking.contextual_retrieval.build_contextualized_text`) that prepends hierarchical breadcrumb + parent heading + prev/next neighbor + non-text modality markers to the embedding string. The canonical `IngestionChunk.content` is never mutated; `metadata.refined_content` is read first per refiner ordering (AGENT-CONTEXTUAL-06). The optional `IngestionChunk.contextualized_text` schema field carries pre-computed embedding text in JSONL when desired but is never read by QA / source-text validation. The ingestor wires text+table modalities through the builder; image chunks continue to use `embed_image()` with the visual description as fallback. `--no-contextual` on `scripts/ingest_to_qdrant.py` restores v2.7.0 byte-stable behavior. AGENT-CONTEXTUAL-01..07 invariants stated in `docs/DECISIONS.md` "Contextual Retrieval (Anthropic approach)".

Evidence:
- Class: tracked
- Command: `conda run -n mmrag-v2 python -m pytest tests/test_pdf_conversion_plan.py::test_no_pipeline_options_construction_outside_adapter tests/test_pdf_conversion_plan.py::test_no_production_docling_imports_outside_adapter -q`
- Input: AST-level static guards covering all production Python files under `src/mmrag_v2/`.
- Output: terminal result
- Result: `2 passed`
- Tracked: yes
- Limitations: AST-level guards; do not protect against constructor calls reflected via getattr.

Evidence:
- Class: tracked
- Command: `conda run -n mmrag-v2 python -m pytest tests/test_contextual_retrieval.py -q`
- Input: focused contextual-retrieval suite (32 cases): content immutability, prefix separation, missing-context handling, length bounds, QA isolation against `content`-only readers (audit, universal invariants, token validator), modality marker rules, integration with `SemanticContext`, edge cases (empty content, marker-in-content preservation, defensive `None` reads), ingestor boundary monkeypatched-embedder capture for `--no-contextual` byte-stability + default contextual flow + image-lane untouched + zero-marker payload guard, and the AST-level drift guard `test_no_contextual_marker_strings_in_production_code`.
- Output: terminal result
- Result: `32 passed`
- Tracked: yes
- Limitations: drift guard is AST-level over `src/mmrag_v2/` and `scripts/`; reflected/dynamically-built marker strings would not be caught.

Evidence:
- Class: tracked + local-run
- Command: `conda run -n mmrag-v2 python -m pytest -q`
- Input: tracked full unit suite after addition of 32 contextual cases.
- Output: terminal result
- Result: `512 passed, 1 skipped, 0 failed` (was 480; net change +32 contextual = 512).
- Tracked: yes
- Limitations: unit suite does not prove corpus-level conversion behavior by itself.

Evidence:
- Class: local-run
- Command: `conda run -n mmrag-v2 python -m mmrag_v2.cli process "data/technical_manual/A Simple Guide to Retrieval Augmented Generation Kimothi A. 2025.pdf" --output-dir output/probe_contextual_retrieval_rag_guide --batch-size 10 --vision-provider none --no-refiner --no-cache`
- Input: 258-page technical manual; same regression target as Milestones 1, 2 and the Boundary Closeout.
- Output: `output/probe_contextual_retrieval_rag_guide/ingestion.jsonl`
- Result: AUDIT_PASS + UNIVERSAL_PASS. 680 chunks (text=559, image=99, table=22), CODE [mixed_prose] `indentation_fidelity=0.91`, `infix_strict=0`. Byte-identical structural shape to `output/probe_boundary_closeout_rag_guide/`. Confirms contextualization is embed-time only and does not change conversion output.
- Tracked: no
- Limitations: generated artifacts ignored; rerun to reproduce.

Evidence:
- Class: local-run
- Command: `conda run -n mmrag-v2 bash scripts/smoke_multiprofile.sh`
- Input: smoke matrix in `scripts/smoke_multiprofile.sh`
- Output: `output/smoke_multiprofile_20260501_153101/_summary.txt`
- Result: 10/10 rows `GATE_PASS` + `UNIVERSAL_PASS`, including the `Greenhouse Design and Control by Pedro Ponce` blind-test document (`AGENT-VAL-01`). Zero regressions vs. the Boundary Closeout baseline.
- Tracked: no
- Limitations: generated artifacts ignored; rerun to reproduce.
