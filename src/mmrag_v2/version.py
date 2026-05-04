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

# Engine/runtime version (PLAN_V2.9 closes the four v2.8 carry-overs
# — chunk_id collision, refiner smart-routing, Ayeva misroute,
# Firearms heading regression — and ships the cloud-VLM-enriched
# mmrag_v2_8 collection. Bump to 2.9.0.)
__engine_version__ = "2.9.0"
