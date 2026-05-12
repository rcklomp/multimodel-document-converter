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

# Engine/runtime version. v2.8.0 was the prior shipped tag.
# v2.9.0-rc1 tagged 2026-05-12 on commit 3e06d1b after the
# strict-gate corpus close (9 PASS / 8 WARN / 17 FAIL →
# 26 PASS / 0 WARN / 8 FAIL; all 8 FAILs are signed v2.10
# deferrals per docs/DECISIONS.md "v2.9.0-rc1 Signed Deferrals").
# v2.9.0-rc1 is the v2.9 ship state; no intermediate v2.9.0 final
# tag is planned. The 8 deferrals (Firearms HEADING, KI EPUB,
# Devlin HEADING, cross-page-split misattribution on
# Python_Cookbook / Python_Distilled, TextIntegrityScout full-doc
# sensitivity on Fluent_Python, text-label-TOC on Chaubal p11,
# full-doc picture dedup on Earthship) carry forward as v2.10
# production-tag blockers under the unchanged strict gate.
# See docs/QUALITY_SNAPSHOT_2026-05-11_v2.9.0-rc1_after.md.
__engine_version__ = "2.9.0-rc1"
