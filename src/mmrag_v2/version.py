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

# Engine/runtime version. v2.12.0 adds the retrieval-side stack on
# top of v2.11.0's embedder swap:
#   Phase 0   content/refined_content preference fix (Format +partial)
#   Phase 1   cross-encoder reranker (local ModernBERT via omlx-server)
#   Phase 2   hybrid retrieval (BM25 sparse + dense + RRF fusion)
#   Phase 3   HyDE — measured but ships opt-in (default off)
# Final v2.12 vs v2.11.0:
#   Recall@1 chunk    35.5% → 67.8%   (+32.3pp, 1.9×)
#   Recall@5 chunk    66.8% → 90.2%   (+23.4pp, STRETCH met)
#   Recall@5 doc      91.7% → 98.6%   (STRETCH met)
#   Relevance         59.3% → 82.1%
#   Faithfulness      50.6% → 72.6%
#   Format            89.8% → 88.4%   (Phase 0 carry-forward to v2.13)
# v2.12.0 annotated tag is STAGED but not pushed by the autonomous
# run; the user pushes/tags after live-stack re-verification.
# Predecessor: v2.11.0 (2026-05-20, c2a461c).
__engine_version__ = "2.12.0"
