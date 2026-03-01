"""
Ingestion Schema - Pydantic V2 Models for MM-Converter-V2
==========================================================
ENGINE_USE: Docling v2.66.0 (Native Layout Analysis)

This module defines the "Gold Standard" output schema for the Multimodal RAG
Ingestion Pipeline per SRS v2.3.0. All coordinates use normalized 0-1000 INTEGER scale.

REQ Compliance:
- REQ-SCHEMA-01: IngestionChunk is the canonical output format
- REQ-COORD-01: All bounding boxes normalized to 0-1000 integer scale
- REQ-COORD-02: bbox values MUST be List[int] with exactly 4 elements
- REQ-COORD-03: Format [x_min, y_min, x_max, y_max] integers 0-1000
- REQ-MM-03: Contextual anchoring with prev/next text snippets (300 chars)
- REQ-MM-02: Asset naming [DocHash]_[Page]_[Type]_[Index].png
- REQ-HIER-04: Breadcrumb depth MUST match hierarchy.level value

SRS Section 6: Output Schema
"Each chunk MUST include: chunk_id, content, modality, metadata, and
optional asset_ref for multimodal elements."

Author: Claude 4.5 Opus (Architect)
Date: 2025-12-28
Updated: 2026-01-08 (SRS v2.3.0 Compliance)
"""

from __future__ import annotations

import hashlib
import warnings
from datetime import datetime, timezone
from enum import Enum
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

# Centralized versioning (single source of truth)
from ..version import __schema_version__


# ============================================================================
# CONSTANTS
# ============================================================================

MAX_CONTEXT_SNIPPET_CHARS: int = 300  # REQ-MM-03
SCHEMA_VERSION: str = __schema_version__  # V2.4: Intelligence Stack Observability (stable)
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
    """
    Content modality types per SRS Section 6.1 and ARCHITECTURE.md V3.0.0.

    Valid values: text, image, table
    SHADOW modality is REMOVED per V3.0.0 architecture (NEVER "shadow").
    """

    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"


class ExtractionMethod(str, Enum):
    """
    Extraction method identifiers per SRS Section 6.1.

    REQ: extraction_method must be one of: docling, shadow, ocr
    """

    DOCLING = "docling"
    SHADOW = "shadow"
    OCR = "ocr"


class ContentClassification(str, Enum):
    """Content classification types per SRS Section 6.1."""

    EDITORIAL = "editorial"
    TECHNICAL = "technical"
    ADVERTISEMENT = "advertisement"


class OCRConfidenceLevel(str, Enum):
    """OCR confidence levels per REQ-PATH-06."""

    HIGH = "high"  # >= 0.85
    MEDIUM = "medium"  # 0.70 - 0.85
    LOW = "low"  # < 0.70


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
    REQ-COORD-03: Format [x_min, y_min, x_max, y_max]

    Coordinate system:
    - (0, 0) is top-left corner
    - (1000, 1000) is bottom-right corner
    """

    l: int = Field(..., ge=0, le=COORD_SCALE, description="Left edge / x_min (0-1000)")
    t: int = Field(..., ge=0, le=COORD_SCALE, description="Top edge / y_min (0-1000)")
    r: int = Field(..., ge=0, le=COORD_SCALE, description="Right edge / x_max (0-1000)")
    b: int = Field(..., ge=0, le=COORD_SCALE, description="Bottom edge / y_max (0-1000)")

    @model_validator(mode="after")
    def validate_box(self) -> "BoundingBox":
        """Ensure box has valid dimensions."""
        if self.r <= self.l:
            raise ValueError(f"Invalid bbox: r ({self.r}) must be > l ({self.l})")
        if self.b <= self.t:
            raise ValueError(f"Invalid bbox: b ({self.b}) must be > t ({self.t})")
        return self

    def to_list(self) -> List[int]:
        """Convert to [x_min, y_min, x_max, y_max] list format per REQ-COORD-03."""
        return [self.l, self.t, self.r, self.b]

    @classmethod
    def from_list(cls, bbox: List[int]) -> "BoundingBox":
        """Create from [x_min, y_min, x_max, y_max] integer list."""
        if len(bbox) != 4:
            raise ValueError(f"BoundingBox requires 4 values, got {len(bbox)}")
        return cls(l=int(bbox[0]), t=int(bbox[1]), r=int(bbox[2]), b=int(bbox[3]))

    def area(self) -> int:
        """Calculate normalized area."""
        return (self.r - self.l) * (self.b - self.t)

    def area_ratio(self, page_area: int = 1000000) -> float:
        """
        Calculate area ratio for Full-Page Guard (REQ-MM-08).

        Returns ratio of asset area to page area (0.0-1.0).
        Full page = 1000 * 1000 = 1,000,000 normalized units.
        """
        return float(self.area()) / float(page_area)

    def is_full_page(self, threshold: float = 0.95) -> bool:
        """
        Check if bbox covers full page (IRON-07, REQ-MM-09).

        Returns True if area_ratio > threshold (default 0.95).
        """
        return self.area_ratio() > threshold

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
    """
    Hierarchical context metadata for breadcrumb tracking.

    REQ-HIER-04: Breadcrumb depth MUST match hierarchy.level value.
    """

    parent_heading: Optional[str] = Field(default=None)
    breadcrumb_path: List[str] = Field(default_factory=list)
    level: Optional[int] = Field(default=None, ge=1, le=5)

    @model_validator(mode="after")
    def sync_level_with_breadcrumbs(self) -> "HierarchyMetadata":
        """
        REQ-HIER-04: Auto-calculate level from breadcrumb depth if not set.
        """
        if self.breadcrumb_path and self.level is None:
            # Set level based on breadcrumb depth (capped at 5 per SRS)
            self.level = min(len(self.breadcrumb_path), 5)
        return self


# ============================================================================
# SPATIAL METADATA
# ============================================================================


class SpatialMetadata(BaseModel):
    """
    Spatial positioning metadata per SRS Section 6.1.

    REQ-COORD-01: Coordinates use 0-1000 integer scale.
    REQ-COORD-02: bbox MUST be List[int] with exactly 4 elements.
    """

    bbox: Optional[List[int]] = Field(
        default=None, description="[x_min, y_min, x_max, y_max] integers 0-1000 per REQ-COORD-03"
    )
    page_width: Optional[int] = Field(default=None, description="Page width in pixels")
    page_height: Optional[int] = Field(default=None, description="Page height in pixels")

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: Optional[List[int]]) -> Optional[List[int]]:
        """
        Validate bbox conforms to REQ-COORD-01 through REQ-COORD-05.

        - Must be List[int] with 4 elements
        - All values must be integers in range [0, 1000]
        - REQ-COORD-04: Floats, raw pixels, percentages PROHIBITED
        """
        if v is None:
            return v
        if len(v) != 4:
            raise ValueError(f"REQ-COORD-02: bbox must have 4 values, got {len(v)}")
        for i, coord in enumerate(v):
            if not isinstance(coord, int):
                raise ValueError(
                    f"REQ-COORD-04 VIOLATION: bbox[{i}]={coord} ({type(coord).__name__}) "
                    f"must be int. Floats/strings PROHIBITED."
                )
            if not 0 <= coord <= COORD_SCALE:
                raise ValueError(
                    f"REQ-COORD-01 VIOLATION: bbox[{i}]={coord} out of range [0, {COORD_SCALE}]"
                )
        return v


# ============================================================================
# SEMANTIC CONTEXT (REQ-MM-03)
# ============================================================================


class SemanticContext(BaseModel):
    """
    Contextual anchoring per REQ-MM-03.

    Provides surrounding text for contextual embedding of multimodal elements.
    """

    prev_text_snippet: Optional[str] = Field(
        default=None,
        max_length=MAX_CONTEXT_SNIPPET_CHARS,
        description="Text before this element (max 300 chars)",
    )
    next_text_snippet: Optional[str] = Field(
        default=None,
        max_length=MAX_CONTEXT_SNIPPET_CHARS,
        description="Text after this element (max 300 chars)",
    )
    parent_heading: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Nearest parent heading for contextual grounding",
    )
    breadcrumb_path: Optional[List[str]] = Field(
        default=None,
        description="Breadcrumb path for hierarchical context",
    )


# ============================================================================
# ASSET REFERENCE
# ============================================================================


class AssetReference(BaseModel):
    """
    Reference to extracted multimodal asset per REQ-MM-02.

    Naming: [DocHash]_[PageNum]_[Type]_[Index].png
    """

    file_path: str = Field(..., description="Relative path to asset file")
    mime_type: str = Field(default="image/png")
    width_px: Optional[int] = Field(default=None)
    height_px: Optional[int] = Field(default=None)
    file_size_bytes: Optional[int] = Field(default=None)


# ============================================================================
# CHUNK METADATA
# ============================================================================


class ChunkMetadata(BaseModel):
    """
    Complete metadata for an ingestion chunk per SRS Section 6.1.

    Required fields: source_file, file_type, page_number, extraction_method, created_at
    """

    source_file: str = Field(..., description="Original filename")
    file_type: FileType = Field(..., description="pdf|epub|html|docx|pptx|xlsx")
    page_number: int = Field(..., ge=1, description="1-indexed page number")
    chunk_type: Optional[ChunkType] = Field(
        default=None, description="paragraph|heading|list|caption|null"
    )
    hierarchy: HierarchyMetadata = Field(default_factory=HierarchyMetadata)
    spatial: Optional[SpatialMetadata] = Field(default=None)
    extraction_method: str = Field(
        default="docling", description="docling|shadow|ocr per SRS Section 6.1"
    )
    content_classification: Optional[str] = Field(
        default=None, description="editorial|technical|advertisement"
    )
    ocr_confidence: Optional[str] = Field(
        default=None, description="high|medium|low per REQ-PATH-06"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Internal fields (not in SRS but useful)
    search_priority: str = Field(
        default="medium", description="Search ranking: high (text), medium (tables), low (images)"
    )
    visual_description: Optional[str] = Field(
        default=None, max_length=400, description="VLM description for images (internal use)"
    )

    # REQ-REFINE-01: Refinement metadata (v18.2 - Semantic Text Refiner)
    # Original 'content' is NEVER overwritten; refined text stored here for opt-in use
    refined_content: Optional[str] = Field(
        default=None,
        description="LLM-refined text (Stage B output). Original content preserved in 'content' field.",
    )
    refinement_applied: bool = Field(
        default=False,
        description="True if Stage B LLM refinement was applied and accepted by Stage C validation",
    )
    corruption_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Stage A noise scanner corruption score (0.0=clean, 1.0=severely corrupted)",
    )
    refinement_provider: Optional[str] = Field(
        default=None,
        description="LLM provider used for refinement (ollama|openai|anthropic) - audit trail",
    )
    refinement_model: Optional[str] = Field(
        default=None,
        description="LLM model used for refinement (e.g., llama2, gpt-4o-mini) - audit trail",
    )

    # V2.4 INTELLIGENCE STACK METADATA (Observability Fix)
    # These fields provide proof that intelligent classification ran
    profile_type: Optional[str] = Field(
        default=None,
        description="Strategy profile used (e.g., academic_whitepaper, digital_magazine) - proves classification ran",
    )
    profile_sensitivity: Optional[float] = Field(
        default=None,
        ge=0.1,
        le=1.0,
        description="Profile-determined sensitivity value (0.1-1.0) - proves profile params were applied",
    )
    min_image_dims: Optional[str] = Field(
        default=None,
        description="Min image dimensions used (e.g., '30x30', '100x100') - proves threshold configuration",
    )
    confidence_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="VLM confidence threshold used - proves strategy configuration",
    )
    document_domain: Optional[str] = Field(
        default=None,
        description="Detected document domain (academic, editorial, technical) - diagnostic result",
    )
    document_modality: Optional[str] = Field(
        default=None,
        description="Detected modality (native_digital, scanned_clean, scanned_degraded) - diagnostic result",
    )


# ============================================================================
# INGESTION CHUNK (MAIN MODEL)
# ============================================================================


class IngestionChunk(BaseModel):
    """
    The canonical output model for MM-Converter-V2 per SRS v2.3.0 Section 6.1.

    REQ-SCHEMA-01: Every processed document element becomes an IngestionChunk.

    Required fields:
    - chunk_id, doc_id, modality, content
    - metadata.source_file, metadata.file_type, metadata.page_number
    - metadata.extraction_method, metadata.created_at
    - metadata.hierarchy.breadcrumb_path (may be empty list)
    - metadata.spatial.bbox (REQUIRED when modality is image or table)
    - asset_ref (REQUIRED when modality is image or table)
    - schema_version
    """

    chunk_id: str = Field(..., description="UUID_v4 or composite hash")
    doc_id: str = Field(..., description="12-char hex from file MD5")
    modality: Modality = Field(..., description="text|image|table")
    content: str = Field(..., description="Text content or VLM description")
    metadata: ChunkMetadata = Field(...)
    asset_ref: Optional[AssetReference] = Field(default=None)
    semantic_context: Optional[SemanticContext] = Field(
        default=None, description="REQ-MM-03: Contextual anchoring with surrounding text"
    )
    schema_version: str = Field(default=SCHEMA_VERSION)

    @computed_field
    @property
    def visual_description(self) -> Optional[str]:
        """Top-level accessor for the VLM image description.

        Returns metadata.visual_description for image chunks so that the field
        appears at the root level of the serialised JSON (not just nested in metadata).
        Non-image chunks return None.
        """
        if self.modality == Modality.IMAGE:
            return self.metadata.visual_description
        return None

    @computed_field
    @property
    def chunk_type(self) -> Optional[ChunkType]:
        """Top-level accessor for the chunk content type.

        Returns metadata.chunk_type so that the field appears at the root level
        of the serialised JSON (not just nested in metadata).  Image and table
        chunks do not have a text-classification type, so they return None.
        """
        return self.metadata.chunk_type

    @model_validator(mode="after")
    def validate_multimodal_requirements(self) -> "IngestionChunk":
        """
        Validate SRS Section 6.3 required fields for multimodal chunks.

        V3.0.0: Only text, image, table modalities are valid.
        Image/Table modalities MUST have spatial.bbox and asset_ref.
        """
        if self.modality in (Modality.IMAGE, Modality.TABLE):
            # REQ: Image/Table MUST have asset_ref
            if self.asset_ref is None:
                raise ValueError(
                    f"QA-CHECK-05 VIOLATION: modality={self.modality.value} "
                    f"requires asset_ref (chunk_id={self.chunk_id})"
                )

            # REQ: Image/Table MUST have spatial.bbox
            if self.metadata.spatial is None or self.metadata.spatial.bbox is None:
                raise ValueError(
                    f"REQ-COORD-01 VIOLATION: modality={self.modality.value} "
                    f"requires metadata.spatial.bbox (chunk_id={self.chunk_id})"
                )

        return self

    def to_embedding_text(self) -> str:
        """Generate text suitable for embedding."""
        parts = []
        if self.metadata.hierarchy.breadcrumb_path:
            path = " > ".join(self.metadata.hierarchy.breadcrumb_path)
            parts.append(f"[{path}]")
        parts.append(self.content)
        # NOTE: For image chunks, content IS already the VLM description (set by all
        # ingestion code paths). Do not append [Visual: ...] here — it would duplicate
        # the description, inflating the embedding vector for no benefit.
        return " ".join(parts)


# ============================================================================
# INGESTION METADATA (DOCUMENT-LEVEL RECORD)
# ============================================================================


class IngestionMetadata(BaseModel):
    """First record in every ingestion.jsonl — document-level summary.

    Written as the very first line before any IngestionChunk records.
    Downstream consumers MUST skip records where object_type == "ingestion_metadata"
    when iterating chunks.
    """

    model_config = ConfigDict(populate_by_name=True)

    object_type: Literal["ingestion_metadata"] = "ingestion_metadata"
    schema_version: str
    doc_id: str
    source_file: str
    profile_type: Optional[str] = None
    document_type: Optional[str] = None
    domain: Optional[str] = None
    is_scan: Optional[bool] = None
    total_pages: Optional[int] = None
    image_density: Optional[float] = None
    avg_text_per_page: Optional[float] = None
    has_flat_text_corruption: Optional[bool] = None
    has_encoding_corruption: Optional[bool] = None
    chunk_count: Optional[int] = None
    ingestion_timestamp: str = ""


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_ocr_confidence_level(confidence: float) -> str:
    """
    Convert numeric OCR confidence to level string per REQ-PATH-06.

    - >= 0.85: "high"
    - 0.70 - 0.85: "medium"
    - < 0.70: "low"
    """
    if confidence >= 0.85:
        return OCRConfidenceLevel.HIGH.value
    elif confidence >= 0.70:
        return OCRConfidenceLevel.MEDIUM.value
    else:
        return OCRConfidenceLevel.LOW.value


def calculate_hierarchy_level(breadcrumb_path: List[str]) -> Optional[int]:
    """
    REQ-HIER-04: Calculate hierarchy level from breadcrumb depth.

    Level is capped at 5 per SRS Section 3.4 hierarchy standard.
    """
    if not breadcrumb_path:
        return None
    return min(len(breadcrumb_path), 5)


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================


def _generate_chunk_id(doc_id: str, content: str, page: int, modality: str = "text") -> str:
    """Generate unique chunk ID using content hash."""
    hash_input = f"{doc_id}:{page}:{modality}:{content}"
    content_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
    return f"{doc_id}_{page:03d}_{modality}_{content_hash}"


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
    extraction_method: str = "docling",
    prev_text: Optional[str] = None,
    next_text: Optional[str] = None,
    refined_content: Optional[str] = None,
    refinement_applied: bool = False,
    corruption_score: Optional[float] = None,
    refinement_provider: Optional[str] = None,
    refinement_model: Optional[str] = None,
    asset_ref: Optional[AssetReference] = None,
    content_classification: Optional[str] = None,
    # V2.4 Intelligence Stack Metadata
    profile_type: Optional[str] = None,
    profile_sensitivity: Optional[float] = None,
    min_image_dims: Optional[str] = None,
    confidence_threshold: Optional[float] = None,
    document_domain: Optional[str] = None,
    document_modality: Optional[str] = None,
) -> IngestionChunk:
    """
    Factory function to create a TEXT modality chunk.

    TEXT chunks get HIGH search priority (OCR text is ground truth for RAG).
    """
    chunk_id = _generate_chunk_id(doc_id, content, page_number, "text")

    # Ensure hierarchy has level calculated
    if hierarchy is None:
        hierarchy = HierarchyMetadata()
    elif hierarchy.level is None and hierarchy.breadcrumb_path:
        hierarchy.level = calculate_hierarchy_level(hierarchy.breadcrumb_path)

    spatial = None
    if bbox:
        spatial = SpatialMetadata(bbox=bbox)

    # Convert numeric confidence to level string
    confidence_level = None
    if ocr_confidence is not None:
        confidence_level = get_ocr_confidence_level(ocr_confidence)

    metadata = ChunkMetadata(
        source_file=source_file,
        file_type=file_type,
        page_number=page_number,
        chunk_type=chunk_type,
        hierarchy=hierarchy,
        spatial=spatial,
        extraction_method=extraction_method,
        search_priority="high",
        ocr_confidence=confidence_level,
        refined_content=refined_content,
        refinement_applied=refinement_applied,
        corruption_score=corruption_score,
        refinement_provider=refinement_provider,
        refinement_model=refinement_model,
        asset_ref=asset_ref,
        content_classification=content_classification,
        # V2.4 Intelligence Stack Metadata
        profile_type=profile_type,
        profile_sensitivity=profile_sensitivity,
        min_image_dims=min_image_dims,
        confidence_threshold=confidence_threshold,
        document_domain=document_domain,
        document_modality=document_modality,
    )

    # Build semantic context if provided
    semantic_context = None
    if prev_text or next_text:
        semantic_context = SemanticContext(
            prev_text_snippet=prev_text[:300] if prev_text else None,
            next_text_snippet=next_text[:300] if next_text else None,
            parent_heading=hierarchy.parent_heading if hierarchy else None,
            breadcrumb_path=hierarchy.breadcrumb_path if hierarchy else None,
        )

    return IngestionChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content=content,
        modality=Modality.TEXT,
        metadata=metadata,
        semantic_context=semantic_context,
    )


def create_image_chunk(
    doc_id: str,
    content: str,
    source_file: str,
    file_type: FileType,
    page_number: int,
    asset_path: str,
    bbox: List[int],  # REQUIRED for images per REQ-COORD-01
    hierarchy: Optional[HierarchyMetadata] = None,
    prev_text: Optional[str] = None,
    next_text: Optional[str] = None,
    visual_description: Optional[str] = None,
    width_px: Optional[int] = None,
    height_px: Optional[int] = None,
    file_size_bytes: Optional[int] = None,
    extraction_method: str = "docling",
    page_width: Optional[int] = None,
    page_height: Optional[int] = None,
    # V2.4 Intelligence Stack Metadata (same as text chunks)
    profile_type: Optional[str] = None,
    profile_sensitivity: Optional[float] = None,
    min_image_dims: Optional[str] = None,
    confidence_threshold: Optional[float] = None,
    document_domain: Optional[str] = None,
    document_modality: Optional[str] = None,
) -> IngestionChunk:
    """
    Factory function to create an IMAGE modality chunk.

    REQUIRES: bbox (spatial coordinates) per SRS Section 6.3
    IMAGE chunks get LOW search priority (VLM descriptions are supplementary).

    REQ-COORD-02: page_width and page_height MUST be populated for UI overlay support.

    V2.4: Image chunks now carry intelligence metadata for full audit trail.
    """
    chunk_id = _generate_chunk_id(doc_id, f"image:{asset_path}", page_number, "image")

    # Ensure hierarchy has level calculated
    if hierarchy is None:
        hierarchy = HierarchyMetadata()
    elif hierarchy.level is None and hierarchy.breadcrumb_path:
        hierarchy.level = calculate_hierarchy_level(hierarchy.breadcrumb_path)

    # bbox is REQUIRED for images, page dimensions for REQ-COORD-02
    spatial = SpatialMetadata(bbox=bbox, page_width=page_width, page_height=page_height)

    metadata = ChunkMetadata(
        source_file=source_file,
        file_type=file_type,
        page_number=page_number,
        hierarchy=hierarchy,
        spatial=spatial,
        visual_description=visual_description,
        extraction_method=extraction_method,
        search_priority="low",
        # V2.4 Intelligence Stack Metadata
        profile_type=profile_type,
        profile_sensitivity=profile_sensitivity,
        min_image_dims=min_image_dims,
        confidence_threshold=confidence_threshold,
        document_domain=document_domain,
        document_modality=document_modality,
    )

    asset_ref = AssetReference(
        file_path=asset_path,
        mime_type="image/png",
        width_px=width_px,
        height_px=height_px,
        file_size_bytes=file_size_bytes,
    )

    # Build semantic context (REQ-MM-03)
    semantic_context = SemanticContext(
        prev_text_snippet=prev_text[:300] if prev_text else None,
        next_text_snippet=next_text[:300] if next_text else None,
        parent_heading=hierarchy.parent_heading if hierarchy else None,
        breadcrumb_path=hierarchy.breadcrumb_path if hierarchy else None,
    )

    return IngestionChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content=content,
        modality=Modality.IMAGE,
        metadata=metadata,
        asset_ref=asset_ref,
        semantic_context=semantic_context,
    )


def create_table_chunk(
    doc_id: str,
    content: str,
    source_file: str,
    file_type: FileType,
    page_number: int,
    bbox: List[int],  # REQUIRED for tables per REQ-COORD-01
    hierarchy: Optional[HierarchyMetadata] = None,
    asset_path: Optional[str] = None,
    # ✅ FIX 3B: Add semantic context parameters for REQ-MM-03 symmetry
    prev_text: Optional[str] = None,
    next_text: Optional[str] = None,
    width_px: Optional[int] = None,
    height_px: Optional[int] = None,
    extraction_method: str = "docling",
    page_width: Optional[int] = None,
    page_height: Optional[int] = None,
    # V2.4 Intelligence Stack Metadata (same as text/image chunks)
    profile_type: Optional[str] = None,
    profile_sensitivity: Optional[float] = None,
    min_image_dims: Optional[str] = None,
    confidence_threshold: Optional[float] = None,
    document_domain: Optional[str] = None,
    document_modality: Optional[str] = None,
) -> IngestionChunk:
    """
    Factory function to create a TABLE modality chunk.

    REQUIRES: bbox (spatial coordinates) per SRS Section 6.3
    REQ-COORD-02: page_width and page_height MUST be populated for UI overlay support.

    V2.4: Table chunks now carry intelligence metadata for full audit trail.
    """
    chunk_id = _generate_chunk_id(doc_id, f"table:{content[:50]}", page_number, "table")

    # Ensure hierarchy has level calculated
    if hierarchy is None:
        hierarchy = HierarchyMetadata()
    elif hierarchy.level is None and hierarchy.breadcrumb_path:
        hierarchy.level = calculate_hierarchy_level(hierarchy.breadcrumb_path)

    # bbox is REQUIRED for tables, page dimensions for REQ-COORD-02
    spatial = SpatialMetadata(bbox=bbox, page_width=page_width, page_height=page_height)

    metadata = ChunkMetadata(
        source_file=source_file,
        file_type=file_type,
        page_number=page_number,
        hierarchy=hierarchy,
        spatial=spatial,
        extraction_method=extraction_method,
        search_priority="medium",
        # V2.4 Intelligence Stack Metadata
        profile_type=profile_type,
        profile_sensitivity=profile_sensitivity,
        min_image_dims=min_image_dims,
        confidence_threshold=confidence_threshold,
        document_domain=document_domain,
        document_modality=document_modality,
    )

    # Asset ref is required if we have an asset_path
    asset_ref = None
    if asset_path:
        asset_ref = AssetReference(
            file_path=asset_path,
            mime_type="image/png",
            width_px=width_px,
            height_px=height_px,
        )

    # ✅ FIX 3B: Build semantic context if provided (REQ-MM-03 symmetry with images)
    semantic_context = None
    if prev_text or next_text:
        semantic_context = SemanticContext(
            prev_text_snippet=prev_text[:300] if prev_text else None,
            next_text_snippet=next_text[:300] if next_text else None,
            parent_heading=hierarchy.parent_heading if hierarchy else None,
            breadcrumb_path=hierarchy.breadcrumb_path if hierarchy else None,
        )

    return IngestionChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content=content,
        modality=Modality.TABLE,
        metadata=metadata,
        asset_ref=asset_ref,
        semantic_context=semantic_context,  # ✅ FIX 3B: Add semantic context
    )


def create_shadow_chunk(
    doc_id: str,
    content: str,
    source_file: str,
    file_type: FileType,
    page_number: int,
    asset_path: str,
    bbox: List[int],  # REQUIRED
    hierarchy: Optional[HierarchyMetadata] = None,
    prev_text: Optional[str] = None,
    next_text: Optional[str] = None,
    visual_description: Optional[str] = None,
    width_px: Optional[int] = None,
    height_px: Optional[int] = None,
    # V2.4 Intelligence Stack Metadata (for backwards compat)
    profile_type: Optional[str] = None,
    profile_sensitivity: Optional[float] = None,
    min_image_dims: Optional[str] = None,
    confidence_threshold: Optional[float] = None,
    document_domain: Optional[str] = None,
    document_modality: Optional[str] = None,
) -> IngestionChunk:
    """
    DEPRECATED: Factory function to create a SHADOW modality chunk.

    Use create_image_chunk with extraction_method="shadow" instead.
    This function will be removed in v3.0.0.
    """
    warnings.warn(
        "create_shadow_chunk is DEPRECATED. Use create_image_chunk with "
        "extraction_method='shadow' instead. Will be removed in v3.0.0.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Delegate to create_image_chunk with shadow extraction method
    return create_image_chunk(
        doc_id=doc_id,
        content=content,
        source_file=source_file,
        file_type=file_type,
        page_number=page_number,
        asset_path=asset_path,
        bbox=bbox,
        hierarchy=hierarchy,
        prev_text=prev_text,
        next_text=next_text,
        visual_description=visual_description,
        width_px=width_px,
        height_px=height_px,
        extraction_method="shadow",
        # V2.4: Pass through intelligence metadata
        profile_type=profile_type,
        profile_sensitivity=profile_sensitivity,
        min_image_dims=min_image_dims,
        confidence_threshold=confidence_threshold,
        document_domain=document_domain,
        document_modality=document_modality,
    )
