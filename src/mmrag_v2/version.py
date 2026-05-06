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

# Engine/runtime version. v2.8.0 is the most recent shipped tag.
# A v2.9.0 tag was attempted on 2026-05-05 and removed on 2026-05-06
# after a user-driven QA review surfaced defects the loose gate had
# missed; v2.9 is in progress on `main` but no v2.9.0 git tag exists.
# The version string below reads `2.9.0-dev` to reflect that the
# code on `main` carries the v2.9 in-flight fixes (chunk_id, refiner
# routing, cross-page split, page-scoped dedup/merge, corruption
# interceptor extension, full-page defer, enrichment content-field)
# without claiming a release.
# See docs/QUALITY_SNAPSHOT_2026-05-06_v2.9_strict_gate.md.
__engine_version__ = "2.9.0-dev"
