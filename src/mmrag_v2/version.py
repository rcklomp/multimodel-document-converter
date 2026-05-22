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

# Engine/runtime version. v2.13.0 closes out two parallel workstreams
# on top of the v2.12.0 retrieval stack:
#   Phase 1   local embedder swap — Qwen3-Embedding-8B-mxfp8 via
#             omlx-server replaces cloud text-embedding-v4 as the
#             production embedder (6/6-axis apples-to-apples win)
#   Phase 2   OCR auto-routing — `plan.force_full_page_ocr=True` for
#             scanned profiles, batch_processor auto-overrides
#             layout-aware -> legacy when set so Docling's flag is
#             honored (Earthship + Firearms Format recovery)
# v2.13 P1 apples-to-apples (same fixture, only embedder differs):
#   Recall@1 chunk    55.0% → 57.5%   (+2.5pp omlx)
#   Recall@5 chunk    72.6% → 78.0%   (+5.4pp omlx)
#   Recall@5 doc      93.1% → 95.2%   (+2.1pp omlx)
#   Relevance         74.1% → 74.6%   (noise)
#   Format            89.2% → 92.9%   (+3.7pp omlx)
#   Faithfulness      65.9% → 66.9%   (noise)
# (See `docs/QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_omlx_vs_dashscope.md`.
# Absolute numbers differ from v2.12.0's 6/6-axis canonical because the
# v2.13 P1 fixture was sampled fresh post-v2.13-P2.)
# Production embedder: omlx Qwen3-Embedding-8B-mxfp8 against
# `mmrag_v2_8__qwen3_local` (4096-dim, 31,371 pts). Dashscope
# `mmrag_v2_8__qwen3_dashscope` retained as 30-day rollback baseline
# through 2026-06-19. v2.13.0 annotated tag is STAGED but not pushed
# by the autonomous run; the user pushes after live-stack re-verification.
# Predecessor: v2.12.0 (2026-05-21, 5a2ce18).
__engine_version__ = "2.13.0"
