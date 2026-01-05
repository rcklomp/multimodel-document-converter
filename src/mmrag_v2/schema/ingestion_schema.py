"""
Ingestion Schema - Pydantic V2 Models for MM-Converter-V2
==========================================================
ENGINE_USE: Docling v2.66.0 (Native Layout Analysis)

This module defines the "Gold Standard" output schema for the Multimodal RAG
Ingestion Pipeline. All coordinates use normalized 0-1000 INTEGER scale.

REQ Compliance:
- REQ-SCHEMA-01: IngestionChunk is the canonical output format
- REQ-COORD-01: All bounding boxes normalized to 0-1000 integer scale
- REQ-CHUNK-03: VLM visual_description truncated to 400 chars max (NOT content)
- REQ-MM-02: Asset naming [DocHash]_[Page]_[Type]_[Index].png

SRS Section 6: Output Schema
"Each chunk MUST include: chunk_id, content, modality, metadata, and
optional asset_ref for multimodal elements."

Author: Claude 4.5 Opus (Architect)
Date: 2025-12-28
Updated: 2025-12-30 (Integer 0-1000 Scale)
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================================
# CONSTANTS
# ============================================================================

MAX_CONTENT_CHARS: int = 400
SCHEMA_VERSION: str = "2.0.0"
COORD_SCALE: int = 1000


# ============================================================================
# ENUMS
# ============================================================================


class FileType(str, Enum):
    """Supported input file types."""

    PDF = "pdf"
    EPUB = "epub"
    HTML = "html"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    MARKDOWN = "markdown"
    TXT = "txt"


class Modality(str, Enum):
    """Content modality types."""

    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    SHADOW = "shadow"


class ChunkType(str, Enum):
    """Semantic chunk types for text content."""

    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    QUOTE = "quote"
    CODE = "code"
    TITLE = "title"


# ============================================================================
# BOUNDING BOX MODEL
# ============================================================================


class BoundingBox(BaseModel):
    """
    Normalized bounding box with 0-1000 INTEGER coordinate scale.

    REQ-COORD-01: All coordinates MUST be normalized to 0-1000 integer range.

    Coordinate system:
    - (0, 0) is top-left corner
    - (1000, 1000) is bottom-right corner
    """

    l: int = Field(..., ge=0, le=COORD_SCALE, description="Left edge (0-1000)")
    t: int = Field(..., ge=0, le=COORD_SCALE, description="Top edge (0-1000)")
    r: int = Field(..., ge=0, le=COORD_SCALE, description="Right edge (0-1000)")
    b: int = Field(..., ge=0, le=COORD_SCALE, description="Bottom edge (0-1000)")

    @model_validator(mode="after")
    def validate_box(self) -> "BoundingBox":
        """Ensure box has valid dimensions."""
        if self.r <= self.l:
            raise ValueError(f"Invalid bbox: r ({self.r}) must be > l ({self.l})")
        if self.b <= self.t:
            raise ValueError(f"Invalid bbox: b ({self.b}) must be > t ({self.t})")
        return self

    def to_list(self) -> List[int]:
        """Convert to [l, t, r, b] list format."""
        return [self.l, self.t, self.r, self.b]

    @classmethod
    def from_list(cls, bbox: List[int]) -> "BoundingBox":
        """Create from [l, t, r, b] integer list."""
        if len(bbox) != 4:
            raise ValueError(f"BoundingBox requires 4 values, got {len(bbox)}")
        return cls(l=int(bbox[0]), t=int(bbox[1]), r=int(bbox[2]), b=int(bbox[3]))

    def area(self) -> int:
        """Calculate normalized area."""
        return (self.r - self.l) * (self.b - self.t)

    def iou(self, other: "BoundingBox") -> float:
        """Calculate Intersection over Union."""
        inter_l = max(self.l, other.l)
        inter_t = max(self.t, other.t)
        inter_r = min(self.r, other.r)
        inter_b = min(self.b, other.b)

        if inter_r <= inter_l or inter_b <= inter_t:
            return 0.0

        inter_area = (inter_r - inter_l) * (inter_b - inter_t)
        union_area = self.area() + other.area() - inter_area

        return float(inter_area) / float(union_area) if union_area > 0 else 0.0

    def to_float(self) -> List[float]:
        """Convert to 0.0-1.0 float scale for compatibility."""
        return [c / COORD_SCALE for c in self.to_list()]

    @classmethod
    def from_float(cls, bbox: List[float]) -> "BoundingBox":
        """Create from 0.0-1.0 float scale coordinates."""
        if len(bbox) != 4:
            raise ValueError(f"BoundingBox requires 4 values, got {len(bbox)}")
        return cls(
            l=int(round(bbox[0] * COORD_SCALE)),
            t=int(round(bbox[1] * COORD_SCALE)),
            r=int(round(bbox[2] * COORD_SCALE)),
            b=int(round(bbox[3] * COORD_SCALE)),
        )


# ============================================================================
# HIERARCHY METADATA
# ============================================================================


class HierarchyMetadata(BaseModel):
    """Hierarchical context metadata for breadcrumb tracking."""

    parent_heading: Optional[str] = Field(default=None)
    breadcrumb_path: List[str] = Field(default_factory=list)
    level: Optional[int] = Field(default=None, ge=1, le=6)


# ============================================================================
# SPATIAL METADATA
# ============================================================================


class SpatialMetadata(BaseModel):
    """Spatial positioning metadata. Coordinates use 0-1000 integer scale."""

    bbox: Optional[List[int]] = Field(default=None, description="[l, t, r, b] integers 0-1000")
    page_width: Optional[float] = Field(default=None)
    page_height: Optional[float] = Field(default=None)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: Optional[List[int]]) -> Optional[List[int]]:
        """Validate bbox is 0-1000 integers."""
        if v is None:
            return v
        if len(v) != 4:
            raise ValueError(f"bbox must have 4 values, got {len(v)}")
        for i, coord in enumerate(v):
            if not isinstance(coord, int):
                raise ValueError(f"bbox[{i}]={coord} must be int")
            if not 0 <= coord <= COORD_SCALE:
                raise ValueError(f"bbox[{i}]={coord} out of range [0, {COORD_SCALE}]")
        return v


# ============================================================================
# ASSET REFERENCE
# ============================================================================


class AssetReference(BaseModel):
    """Reference to extracted multimodal asset."""

    file_path: str = Field(..., description="Relative path to asset file")
    mime_type: str = Field(default="image/png")
    width_px: Optional[int] = Field(default=None)
    height_px: Optional[int] = Field(default=None)
    file_size_bytes: Optional[int] = Field(default=None)


# ============================================================================
# CHUNK METADATA
# ============================================================================


class ChunkMetadata(BaseModel):
    """Complete metadata for an ingestion chunk."""

    source_file: str = Field(...)
    file_type: FileType = Field(...)
    page_number: int = Field(..., ge=1)
    chunk_type: Optional[ChunkType] = Field(default=None)
    hierarchy: HierarchyMetadata = Field(default_factory=HierarchyMetadata)
    spatial: Optional[SpatialMetadata] = Field(default=None)
    prev_text: Optional[str] = Field(default=None, max_length=300)
    visual_description: Optional[str] = Field(default=None, max_length=MAX_CONTENT_CHARS)
    extraction_method: str = Field(default="docling")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    search_priority: str = Field(
        default="medium",
        description="Search ranking priority: 'high' for OCR text, 'medium' for tables, 'low' for VLM descriptions",
    )
    ocr_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="OCR confidence score (0.0-1.0) for text extraction quality",
    )
    debug_profile_applied: Optional[str] = Field(
        default=None,
        description="DEBUG: Profile used for processing (e.g., 'ScannedCleanProfile', 'DigitalMagazineProfile')",
    )


# ============================================================================
# INGESTION CHUNK (MAIN MODEL)
# ============================================================================


class IngestionChunk(BaseModel):
    """
    The canonical output model for the MM-Converter-V2 pipeline.

    REQ-SCHEMA-01: Every processed document element becomes an IngestionChunk.

    Note: REQ-CHUNK-03 (400 char limit) applies to visual_description only,
    NOT to content field which can contain full paragraphs/tables.
    """

    chunk_id: str = Field(...)
    doc_id: str = Field(...)
    content: str = Field(..., description="Text content (no length limit for text modality)")
    modality: Modality = Field(...)
    metadata: ChunkMetadata = Field(...)
    asset_ref: Optional[AssetReference] = Field(default=None)
    schema_version: str = Field(default=SCHEMA_VERSION)

    def to_embedding_text(self) -> str:
        """Generate text suitable for embedding."""
        parts = []
        if self.metadata.hierarchy.breadcrumb_path:
            path = " > ".join(self.metadata.hierarchy.breadcrumb_path)
            parts.append(f"[{path}]")
        parts.append(self.content)
        if self.modality == Modality.IMAGE and self.metadata.visual_description:
            parts.append(f"[Visual: {self.metadata.visual_description}]")
        return " ".join(parts)


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================


def _generate_chunk_id(doc_id: str, content: str, page: int) -> str:
    """Generate unique chunk ID using content hash."""
    hash_input = f"{doc_id}:{page}:{content}"
    content_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    return f"{doc_id}_{page:03d}_{content_hash}"


def create_text_chunk(
    doc_id: str,
    content: str,
    source_file: str,
    file_type: FileType,
    page_number: int,
    hierarchy: Optional[HierarchyMetadata] = None,
    chunk_type: ChunkType = ChunkType.PARAGRAPH,
    bbox: Optional[List[int]] = None,
    ocr_confidence: Optional[float] = None,
) -> IngestionChunk:
    """Factory function to create a TEXT modality chunk with HIGH search priority."""
    chunk_id = _generate_chunk_id(doc_id, content, page_number)

    spatial = None
    if bbox:
        spatial = SpatialMetadata(bbox=bbox)

    metadata = ChunkMetadata(
        source_file=source_file,
        file_type=file_type,
        page_number=page_number,
        chunk_type=chunk_type,
        hierarchy=hierarchy or HierarchyMetadata(),
        spatial=spatial,
        extraction_method="docling",
        search_priority="high",  # TEXT chunks get HIGH priority (OCR is ground truth)
        ocr_confidence=ocr_confidence,
    )

    return IngestionChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content=content,
        modality=Modality.TEXT,
        metadata=metadata,
    )


def create_image_chunk(
    doc_id: str,
    content: str,
    source_file: str,
    file_type: FileType,
    page_number: int,
    asset_path: str,
    bbox: Optional[List[int]] = None,
    hierarchy: Optional[HierarchyMetadata] = None,
    prev_text: Optional[str] = None,
    visual_description: Optional[str] = None,
    width_px: Optional[int] = None,
    height_px: Optional[int] = None,
) -> IngestionChunk:
    """Factory function to create an IMAGE modality chunk with LOW search priority."""
    chunk_id = _generate_chunk_id(doc_id, f"image:{asset_path}", page_number)

    spatial = None
    if bbox:
        spatial = SpatialMetadata(bbox=bbox)

    metadata = ChunkMetadata(
        source_file=source_file,
        file_type=file_type,
        page_number=page_number,
        hierarchy=hierarchy or HierarchyMetadata(),
        spatial=spatial,
        prev_text=prev_text[:300] if prev_text and len(prev_text) > 300 else prev_text,
        visual_description=visual_description,
        extraction_method="docling",
        search_priority="low",  # IMAGE chunks get LOW priority (VLM descriptions are supplementary)
    )

    asset_ref = AssetReference(
        file_path=asset_path,
        mime_type="image/png",
        width_px=width_px,
        height_px=height_px,
    )

    return IngestionChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content=content,
        modality=Modality.IMAGE,
        metadata=metadata,
        asset_ref=asset_ref,
    )


def create_table_chunk(
    doc_id: str,
    content: str,
    source_file: str,
    file_type: FileType,
    page_number: int,
    bbox: Optional[List[int]] = None,
    hierarchy: Optional[HierarchyMetadata] = None,
    asset_path: Optional[str] = None,
) -> IngestionChunk:
    """Factory function to create a TABLE modality chunk."""
    chunk_id = _generate_chunk_id(doc_id, f"table:{content[:50]}", page_number)

    spatial = None
    if bbox:
        spatial = SpatialMetadata(bbox=bbox)

    metadata = ChunkMetadata(
        source_file=source_file,
        file_type=file_type,
        page_number=page_number,
        hierarchy=hierarchy or HierarchyMetadata(),
        spatial=spatial,
        extraction_method="docling",
    )

    asset_ref = None
    if asset_path:
        asset_ref = AssetReference(file_path=asset_path, mime_type="image/png")

    return IngestionChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content=content,
        modality=Modality.TABLE,
        metadata=metadata,
        asset_ref=asset_ref,
    )


def create_shadow_chunk(
    doc_id: str,
    content: str,
    source_file: str,
    file_type: FileType,
    page_number: int,
    asset_path: str,
    bbox: Optional[List[int]] = None,
    hierarchy: Optional[HierarchyMetadata] = None,
    prev_text: Optional[str] = None,
    visual_description: Optional[str] = None,
    width_px: Optional[int] = None,
    height_px: Optional[int] = None,
) -> IngestionChunk:
    """Factory function to create a SHADOW modality chunk."""
    chunk_id = _generate_chunk_id(doc_id, f"shadow:{asset_path}", page_number)

    spatial = None
    if bbox:
        spatial = SpatialMetadata(bbox=bbox)

    metadata = ChunkMetadata(
        source_file=source_file,
        file_type=file_type,
        page_number=page_number,
        hierarchy=hierarchy or HierarchyMetadata(),
        spatial=spatial,
        prev_text=prev_text[:300] if prev_text and len(prev_text) > 300 else prev_text,
        visual_description=visual_description,
        extraction_method="shadow",
    )

    asset_ref = AssetReference(
        file_path=asset_path,
        mime_type="image/png",
        width_px=width_px,
        height_px=height_px,
    )

    return IngestionChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content=content,
        modality=Modality.SHADOW,
        metadata=metadata,
        asset_ref=asset_ref,
    )
