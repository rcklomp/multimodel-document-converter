"""
Centralized version definitions for MM-RAG V2.

Single source of truth for schema and engine versions to avoid hardcoded
scattering across the codebase.
"""

# Schema version for ingestion output
__schema_version__ = "2.4.2-stable"

# Engine/runtime version (align with release as needed)
__engine_version__ = __schema_version__
