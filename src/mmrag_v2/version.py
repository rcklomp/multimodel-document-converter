"""
Centralized version definitions for MM-RAG V2.

Single source of truth for schema and engine versions to avoid hardcoded
scattering across the codebase.
"""

# Schema version for ingestion output (chunk-shape contract).
# v2.8 made no changes to the IngestionChunk JSON shape — the
# behavioral changes (keyword-aware control-char replacement,
# CodeFormulaV2 enrichment, form-aware audit gate, adapter
# invocation guard) all preserve the existing schema. v2.9 likewise
# preserves the JSON shape; the chunk_id *value* changes for the 427
# previously-colliding chunks (per-doc position is hashed in) but no
# chunk field is added or removed. Downstream consumers that key on
# chunk_id for cross-version mapping must rebuild from v2.9 outputs.
__schema_version__ = "2.7.0"

# Engine/runtime version. v2.14.0 ships the local-LLM accelerator
# stack on top of the v2.13.0 retrieval stack. NO RETRIEVAL-STACK
# CHANGES — production retrieval is byte-for-byte identical to
# v2.13.0 (omlx Qwen3-Embedding-8B-mxfp8 + BM25 + RRF + ModernBERT
# rerank against `mmrag_v2_8__qwen3_local`).
#
# Phases shipped (alphabetical / numerical, not chronological):
#   Phase 0 (calibration)   27B-MTP local judge: all axes RESTRICTED
#                           (rel 82.0% / format 70.7% / faith 78.8%;
#                           bias flipped vs the retired 14B);
#                           PERMITTED uses contracted to query-gen +
#                           HyDE + tie-breaker harness; ship-gate
#                           judging stays on cloud qwen-max.
#   Phase 4a (HyDE)         `provider="vllm"` knob + Qwen3 thinking-
#                           mode payload fix (chat_template_kwargs
#                           enable_thinking=False); live re-smoke OK.
#   Phase 4c (gen-provider) `synthetic_soak.py --gen-provider vllm`
#                           wires the local 27B for query generation
#                           at $0/query; 2.0s/query live smoke.
#   Phase 4d (tie-breaker)  `scripts/local_then_cloud_soak.py` two-
#                           tier judging: local-vLLM on all in-scope,
#                           cloud qwen-max re-judges contested only;
#                           provenance tagged via judgment.judge_source
#                           ∈ {local, cloud, local_fallback}.
#   Phase 4-Resilience      `hyde.generate_with_fallback` chains
#                           vllm → dashscope qwen3-max → literal query
#                           when primary is vllm.
#   Phase 5 (disk precheck) `_check_disk_headroom` in synthetic_soak
#                           aborts retrieve/judge below 10 GB free.
#
# Phases PARTIAL (code-side landed; data acceptance bar NOT met):
#   Phase 1 (form/table)    `--force-table-vlm` truly forces (was
#                           silently overridden by technical_manual
#                           profile); local NuMarkdown-8B VLM produces
#                           clean 5-col tables on 5/12 CarOK pages.
#                           BUT 30-query CarOK mini-soak measured
#                           Format -26.9pp regression because flat-
#                           prose chunks coexist with VLM tables and
#                           win retrieval 29/30 times. Production
#                           data ROLLED BACK to v2.13 baseline. v2.15
#                           needs same-page prose-VLM dedup.
#   Phase 6 (code chunking) Block-extension policy + `partial_code`
#                           schema field shipped on the
#                           `_chunk_text_with_overlap` (scanned_book)
#                           path. Fluent_Python's truncated-code
#                           defect is Docling-extraction-layer (prose
#                           + code intermixed at page boundaries);
#                           HybridChunker post-merge pass tested in
#                           isolation but doesn't fire in production
#                           (reverted this session). v2.15 needs
#                           upstream Docling-config or text-norm fix.
#
# Phase 3 (rollback drop)   PENDING — time-gated to 2026-06-19
#                           decision point; v2.14.1 candidate.
#
# Production retrieval byte-for-byte identical to v2.13.0:
#   omlx Qwen3-Embedding-8B-mxfp8 → mmrag_v2_8__qwen3_local (4096-dim,
#   31,371 pts) + BM25 sparse + RRF + local ModernBERT rerank.
# v2.13 retrieval fingerprint must still PASS (re-verified at close).
# v2.14.0 annotated tag is STAGED locally; user pushes after live-
# stack re-verification.
# Predecessor: v2.13.0 (2026-05-22, b77341a region; staged tag).
__engine_version__ = "2.14.0"
