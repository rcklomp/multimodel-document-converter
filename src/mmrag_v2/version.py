"""
Centralized version definitions for MM-RAG V2.

Single source of truth for schema and engine versions to avoid hardcoded
scattering across the codebase.
"""

# Schema version for ingestion output (chunk-shape contract).
# v2.8 made no changes to the IngestionChunk JSON shape — the
# behavioral changes (keyword-aware control-char replacement,
# CodeFormulaV2 enrichment, form-aware audit gate, adapter
# invocation guard) all preserve the existing schema. So
# schema_version stays 2.7.0; engine version moves to 2.8.0.
__schema_version__ = "2.7.0"

# Engine/runtime version (PLAN_V2.8 SHIPPED 2026-05-04 → bump to 2.8.0).
__engine_version__ = "2.8.0"
