"""
Schema module for MM-Converter-V2.
Contains Pydantic V2 models for the ingestion pipeline.
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
    create_image_chunk,
    create_table_chunk,
    create_text_chunk,
    create_shadow_chunk,
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
    "create_image_chunk",
    "create_table_chunk",
    "create_text_chunk",
    "create_shadow_chunk",
]
