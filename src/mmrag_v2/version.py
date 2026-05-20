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

# Engine/runtime version. v2.10.0 shipped 2026-05-16 (annotated tag
# on commit db6527c; same tree as v2.10.0-rc1 / 82c3639). v2.11.0
# Phase 1 swapped the production embedder from Ollama llava (4096-dim
# multimodal) to Dashscope text-embedding-v4 (1024-dim text-only) —
# 10× lift on Recall@1/Relevance/Faithfulness; Format pin temporarily
# downgraded to ≥85% with v2.11.x recovery target ≥95% (see
# docs/DECISIONS.md "v2.11.0 Embedder Swap Executed — Format Gate
# Downgrade" and docs/QUALITY_SNAPSHOT_2026-05-20_v2.11_soak_qwen3.md).
# v2.11.0 annotated tag is STAGED but not pushed by the autonomous
# run; the user pushes/tags after live-stack re-verification.
__engine_version__ = "2.11.0"
