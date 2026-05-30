# V3 Deferred Tests

Tests from the v2.16 suite that pin chunk COUNT or CONTENT against
heuristic-patched v2.16 output, and therefore cannot be ported until
the LLM sanitization layer (which subsumes those heuristics) is
implemented. One line per deferred test, with the heuristic
dependence.

The companion prompt placeholders live at
`src/mmrag_v3/sanitization/prompts/`.

---

## Deferred to Phase C (VLM)

Phase B (text-based LLM sanitization on top of Docling extraction)
was completed on 2026-05-29 with a falsified hypothesis: Identity
Gate aggregate = 100.00% over the 3 reference docs. The residual
delta on every doc is **upstream of the V3 boundary** and cannot be
closed by any text-only post-processor. See
`docs/paper/archive_extracts/v3_mandate/V3_PROJECT_STATUS.md` "Identity Gate
— Validated Architectural Finding" (salvaged from the retired `v3_execution_root`).

The following gate / contract is therefore deferred to Phase C
(Vision-Native Extraction), where a VLM operating on rendered page
images replaces the Docling+OCR cascade as the primary engine:

- `scripts/run_identity_gate.py` (aggregate < 5% threshold) — frozen
  at 100.00% pending Phase C. Triple root cause:
  1. **EasyOCR/Docling SIGSEGV** on Apple Silicon during CRAFT
     detector setup (validated 2026-05-29 with
     `EasyOcrOptions(use_gpu=False)` + `AcceleratorOptions(CPU)` +
     `OMP_NUM_THREADS=1`); blocks
     `Form_betwistingsformulier` extraction. Phase C avoids the
     OCR cascade entirely.
  2. **Docling silent table-text drop** on
     `CarOK_voorraadtelling`: with OCR enabled the OCR path
     silently produces no text; with OCR disabled Docling emits
     1 empty-content `TableItem` per page. Phase C VLM reads
     rendered cell glyphs directly.
  3. **Flush-grouping semantic loss** on
     `Bevestigingsmiddelen`: HybridChunker's per-semantic-group
     emission cannot be reconstructed by joining elements between
     IMAGE/TABLE boundaries (text-similarity floor 0.732 → 0.017
     across page-2 chunks); per-chunk BBox attribution was
     successfully tightened (page-2 chunks now carry
     left/middle/right column bboxes — `(44,81,281,930)`,
     `(276,81,435,930)`, `(345,81,904,930)` — replacing the
     prior page-wide `(44,81,904,930)`), but text-only
     sanitization cannot regenerate the spatial grouping from a
     joined string. Phase C VLM emits chunk records directly.

When the Phase C VLM engine ships, the Identity Gate must be
re-tuned (new fixture set, new tolerances calibrated against
VLM-native output rather than v2.16 HybridChunker output).

---

## Deferred (heuristic-dependent shape/count tests)

- `tests/test_docling_postprocess_reading_order.py` — depends on heuristic y_sort_reading_order deferred to LLM sanitization
- `tests/test_docling_postprocess_dropcap.py` — depends on heuristic drop_cap_heal deferred to LLM sanitization
- `tests/test_docling_postprocess_label_filter.py` — depends on heuristic label_leak_filter deferred to LLM sanitization
- `tests/test_docling_postprocess_ocr_gating.py` — depends on heuristic ocr_heading_override deferred to LLM sanitization
- `tests/test_docling_postprocess_profile_integration.py` — depends on heuristic y_sort_reading_order + drop_cap_heal deferred to LLM sanitization
- `tests/test_docling_postprocessor_acceptance.py` — depends on heuristic y_sort_reading_order + drop_cap_heal + label_leak_filter deferred to LLM sanitization
- `tests/test_dense_back_index_detector.py` — depends on heuristic dense_index_detection deferred to LLM sanitization
- `tests/test_toc_cell_marker_sanitizer.py` — depends on heuristic regex_dom_patching deferred to LLM sanitization
- `tests/test_toc_index_page_contract.py` — depends on heuristic dense_index_detection deferred to LLM sanitization
- `tests/test_section_header_only_page_emit.py` — depends on heuristic ocr_heading_override + trailing_preposition_merger deferred to LLM sanitization
- `tests/test_cross_chunk_semantic_stitching.py` — depends on heuristic trailing_preposition_merger deferred to LLM sanitization
- `tests/test_hybrid_chunker_dense_page_router.py` — depends on heuristic dense_index_detection deferred to LLM sanitization
- `tests/test_hybrid_chunker_heading_propagation.py` — depends on heuristic ocr_heading_override + trailing_preposition_merger deferred to LLM sanitization
- `tests/test_heading_carryforward_across_batches.py` — depends on heuristic ocr_heading_override deferred to LLM sanitization
- `tests/test_ocr_path_heading_propagation.py` — depends on heuristic ocr_heading_override deferred to LLM sanitization
- `tests/test_reading_order_fix.py` — depends on heuristic y_sort_reading_order deferred to LLM sanitization
- `tests/test_vlm_table_dedup.py` — depends on heuristic bbox_iou dedup (v2.16 phase 4) — out-of-scope shape, defer
- `tests/test_chunk_corruption_filter.py` — depends on heuristic regex_dom_patching deferred to LLM sanitization
- `tests/test_corruption_quarantine.py` — depends on heuristic regex_dom_patching deferred to LLM sanitization
- `tests/test_corruption_quarantine_toc_exemption.py` — depends on heuristic regex_dom_patching + dense_index_detection deferred to LLM sanitization
- `tests/test_phash_dedup_page_coverage.py` — depends on heuristic bbox_iou dedup deferred to LLM sanitization
- `tests/test_partial_code_cross_page_hybrid.py` — depends on heuristic ocr_heading_override (partial_code detection) deferred to LLM sanitization
- `tests/test_fenced_flat_code_detection.py` — depends on heuristic regex code-fence detection deferred to LLM sanitization
- `tests/test_code_chunking.py` — depends on heuristic regex code-fence detection deferred to LLM sanitization
- `tests/test_code_enrichment_decision.py` — depends on heuristic regex code-fence detection deferred to LLM sanitization
- `tests/test_infix_step_number_repair.py` — depends on heuristic regex_dom_patching deferred to LLM sanitization
- `tests/test_oversize_splitter.py` — depends on heuristic regex_dom_patching deferred to LLM sanitization
- `tests/test_oversize_pua_fixes.py` — depends on heuristic regex_dom_patching deferred to LLM sanitization
- `tests/test_null_fixes.py` — depends on heuristic regex_dom_patching deferred to LLM sanitization
- `tests/test_quality_fixes.py` — depends on heuristic regex_dom_patching deferred to LLM sanitization
- `tests/test_text_integrity_scout_per_batch_trigger.py` — depends on heuristic regex_dom_patching deferred to LLM sanitization
- `tests/test_full_page_guard.py` — depends on heuristic full-page bbox dedup deferred to LLM sanitization
- `tests/test_blank_asset_quarantine.py` — depends on heuristic asset-policy heuristic (drop_blank_assets) deferred to LLM sanitization
- `tests/test_qa_intentionally_blank_pages.py` — depends on heuristic blank-page classifier deferred to LLM sanitization
- `tests/test_qa_near_blank_render.py` — depends on heuristic blank-page classifier deferred to LLM sanitization
- `tests/test_vision_aided_front_matter.py` — depends on heuristic VLM-aided front-matter pickup deferred to LLM sanitization
- `tests/test_vlm_text_detection.py` — depends on heuristic VLM-text-detection deferred to LLM sanitization
- `tests/test_chunk_id_collision_v29.py` — depends on v2.16 chunk_id format (v3 uses simpler `p<page>-r<order>` format)
- `tests/test_classifier_digital_literature.py` — depends on profile classifier (out of v3 scope)
- `tests/test_classifier_fallback.py` — depends on profile classifier (out of v3 scope)
- `tests/test_classifier_firearms_route.py` — depends on profile classifier (out of v3 scope)
- `tests/test_classifier_rule_0c_tightening.py` — depends on profile classifier (out of v3 scope)
- `tests/test_pdf_conversion_plan.py` — depends on v2.16 PdfConversionPlan with profile-specific policy fields (v3 engine uses a fixed policy)
- `tests/test_docling_adapter_ocr_dispatch.py` — depends on v2.16 ocr_engine dispatch (v3 uses default EasyOcrOptions)
- `tests/test_epub_extraction_lane.py` — EPUB engine not yet ported to v3
- `tests/test_retrieval_regression_v2_10.py` — depends on v2.10 chunk count baseline (corpus-specific)
- `tests/test_retrieval_regression_v2_11.py` — depends on v2.11 chunk count baseline (corpus-specific)
- `tests/test_retrieval_regression_v2_12.py` — depends on v2.12 chunk count baseline (corpus-specific)
- `tests/test_canonical_docs_consistency.py` — depends on v2.16 corpus-wide chunk count + content baselines
- `tests/test_universal_pipeline.py` — depends on v2 BoundingBox.from_raw helper + v2 ElementType API (covered by v3-native shape tests in v3_parity_smoke.py)
- `tests/test_v3_uir_contract.py` — depends on v2 dual-vocabulary UIR contract (Modality enum, UIRChunk dataclass) replaced by v3 simpler IngestionChunk in v3
- `tests/test_v3_phase_a_step2_dense_page_uir.py` — depends on heuristic dense_index_detection deferred to LLM sanitization
- `tests/test_v3_phase_a_step3_section_header_uir.py` — depends on heuristic ocr_heading_override deferred to LLM sanitization
- `tests/test_v3_identity_gate_and_fusion.py` — covered by v3-native scripts/run_identity_gate.py in task 8.4
- `tests/test_export_integrity.py` — depends on v2.16 IngestionMetadata + nested-metadata schema (v3 schema is flatter)
- `tests/test_ingestion_metadata.py` — depends on v2.16 IngestionMetadata object_type sentinel (v3 emits chunks only, no per-doc metadata record)
- `tests/test_chunker_guard.py` — depends on v2.16 HybridChunker invariants (v3 chunker is UIR-native)
- `tests/test_cross_page_split_page_attribution.py` — depends on v2.16 cross-page attribution heuristic; v3 covered by simpler continuation_group_id in v3_parity_smoke.py
- `tests/test_semantic_overlap.py` — depends on v2.16 SemanticOverlapManager (post-chunker heuristic out of v3 scope)
- `tests/test_contextual_retrieval.py` — depends on retrieval pipeline (out of v3 engine+chunker scope)
- `tests/test_retrieval_pipeline.py` — depends on retrieval pipeline (out of v3 engine+chunker scope)
- `tests/test_refiner.py` — depends on Semantic Text Refiner (out of v3 engine+chunker scope)
- `tests/test_hyde.py` — depends on HyDE retrieval lane (out of v3 engine+chunker scope)
- `tests/test_intent.py` — depends on intent classifier (out of v3 engine+chunker scope)
- `tests/test_low_recall_trigger.py` — depends on retrieval pipeline (out of v3 engine+chunker scope)
- `tests/test_sparse_bm25.py` — depends on BM25 sparse retrieval (out of v3 engine+chunker scope)
- `tests/test_qdrant_search_priority.py` — depends on Qdrant ingestion pipeline (out of v3 engine+chunker scope)
- `tests/test_qdrant_point_id_collision.py` — depends on Qdrant ingestion pipeline (out of v3 engine+chunker scope)
- `tests/test_rebuild_resume.py` — depends on Qdrant ingestion rebuild lane (out of v3 engine+chunker scope)
- `tests/test_doc_class_telemetry.py` — depends on telemetry/instrumentation lane (out of v3 engine+chunker scope)
- `tests/test_personal_validation.py` — depends on personal_importance overlay (out of v3 engine+chunker scope)
- `tests/test_qa_image_gate_calibration.py` — depends on QA image-gate (out of v3 engine+chunker scope)
- `tests/test_qa_advisory_promotion.py` — depends on QA advisory promotion (out of v3 engine+chunker scope)
- `tests/test_v29_image_enrichment_acceptance.py` — depends on VLM image enrichment (out of v3 engine+chunker scope)
- `tests/test_doctr_block_merge.py` — depends on DocTR OCR cascade (out of v3 engine+chunker scope)
- `tests/test_soak_disk_precheck.py` — depends on disk precheck lane (out of v3 engine+chunker scope)
- `tests/test_local_then_cloud_soak.py` — depends on soak runner (out of v3 engine+chunker scope)
- `tests/test_enrich_retry_harness.py` — depends on retry harness (out of v3 engine+chunker scope)
- `tests/test_token_validator.py` — depends on v2.16 TokenValidator (QA-CHECK-01); v3 has no equivalent post-chunk validator yet
- `tests/test_asset_complexity.py` — depends on asset extraction lane (out of v3 engine+chunker scope)
- `tests/test_cli_refiner_smart_routing.py` — depends on CLI refiner integration (out of v3 engine+chunker scope)
- `tests/test_v3_c_prespike_harness.py` — depends on v3 pre-spike harness (replaced by v3_parity_smoke.py)
- `tests/test_v3_conversion_plan.py` — depends on v2.16 PdfConversionPlan parent contract (replaced by simpler v3 engine policy)
- `tests/test_v3_omlx_scaffold.py` — depends on omlx scaffolding (out of v3 engine+chunker scope)
- `tests/test_v3_sanitization_scaffold.py` — depends on v2.16 sanitization scaffolding (replaced by v3 prompts/ placeholders)
- `tests/test_v3_uir_exporter.py` — depends on v2.16 UIR exporter (replaced by IngestionChunk.model_dump_json in v3)
- `tests/test_v3_ingestion_chunk_from_uir.py` — depends on v2.16 IngestionChunk.from_uir bridge (v3 emits IngestionChunk directly)
- `tests/test_domain_detection_parity.py` — depends on document_domain classifier (out of v3 engine+chunker scope)
- `tests/test_strategy_profiles.py` — depends on profile classifier (out of v3 engine+chunker scope)
- `tests/test_finalization_bridge.py` — depends on v2.16 finalization bridge (replaced by direct chunker emission)
- `tests/test_ingest_content_preference.py` — depends on refined_content preference (out of v3 engine+chunker scope)
- `tests/test_coordinate_normalization_audit.py` — covered by v3-native BoundingBox int [0,1000] tests in v3_parity_smoke.py
- `tests/test_bbox.py` — depends on v2.16 bbox_iou util (v3 has no equivalent yet)
