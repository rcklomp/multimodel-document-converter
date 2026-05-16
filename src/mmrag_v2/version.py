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

# Engine/runtime version. v2.9.0-rc1 was the prior shipped tag
# (3e06d1b, 2026-05-12). v2.10.0-rc1 closes the seven v2.10
# production-tag root-cause classes from the v2.9.0-rc1 signed
# deferrals through Phases 1-7 and a corpus-wide strict-gate
# re-verification + Qdrant rebuild in Phase 8.
# See docs/QUALITY_SNAPSHOT_2026-05-16_v2.10_after.md.
__engine_version__ = "2.10.0-rc1"
