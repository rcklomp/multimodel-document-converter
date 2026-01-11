"""
Schema module for MM-Converter-V2.
Contains Pydantic V2 models for the ingestion pipeline.

V3.0.0: Shadow modality and create_shadow_chunk REMOVED per ARCHITECTURE.md.
"""

from .ingestion_schema import (
    BoundingBox,
    ChunkType,
    FileType,
    HierarchyMetadata,
    IngestionChunk,
    Modality,
    SpatialMetadata,
    AssetReference,
    ChunkMetadata,
    SemanticContext,
    create_image_chunk,
    create_table_chunk,
    create_text_chunk,
)

__all__ = [
    "BoundingBox",
    "ChunkType",
    "FileType",
    "HierarchyMetadata",
    "IngestionChunk",
    "Modality",
    "SpatialMetadata",
    "AssetReference",
    "ChunkMetadata",
    "SemanticContext",
    "create_image_chunk",
    "create_table_chunk",
    "create_text_chunk",
]
