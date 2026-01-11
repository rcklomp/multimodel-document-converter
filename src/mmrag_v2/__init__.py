"""
MMRAG V2 - Multimodal RAG Document Converter
=============================================
ENGINE_USE: Docling v2.66.0 (Native Layout Analysis)

V3.0.0 Architecture:
- Universal Intermediate Representation (UIR)
- TEXT regions -> OCR cascade -> modality: "text"
- IMAGE regions -> VLM visual description -> modality: "image"
- TABLE regions -> Structure extraction -> modality: "table"
- NEVER "shadow" modality per ARCHITECTURE.md

Features:
- Docling v2.66.0 native layout analysis
- Hierarchical breadcrumb tracking (ContextStateV2)
- VLM-based image enrichment (Ollama, OpenAI, Anthropic)
- Memory-efficient batch processing
- Perceptual hash deduplication

Usage:
    # CLI
    mmrag-v2 process document.pdf --vision-provider ollama

    # Python API
    from mmrag_v2 import V2DocumentProcessor
    processor = V2DocumentProcessor(output_dir="./output")
    chunks = processor.process_document("document.pdf")

Author: Claude 4.5 Opus (Architect)
Date: 2025-12-28
Version: 3.0.0
"""

from __future__ import annotations

__version__ = "3.0.0"
__author__ = "Claude 4.5 Opus (Architect)"

# Core processor
from .processor import V2DocumentProcessor, create_processor

# Batch processor
from .batch_processor import BatchProcessor, create_batch_processor

# Mapper
from .mapper import DoclingToV2Mapper, create_mapper

# Schema (V3.0.0: No shadow imports)
from .schema import (
    BoundingBox,
    ChunkType,
    FileType,
    HierarchyMetadata,
    IngestionChunk,
    Modality,
    create_image_chunk,
    create_table_chunk,
    create_text_chunk,
)

# State
from .state import ContextStateV2, create_context_state

# Utils
from .utils import (
    ensure_normalized,
    normalize_bbox,
    PDFBatchSplitter,
    ImageHashRegistry,
)

# Vision
from .vision import VisionManager, create_vision_manager

# Orchestration (V3.0.0: Shadow extraction removed)
from .orchestration import (
    DocumentProfile,
    SmartConfigProvider,
    ExtractionStrategy,
    StrategyOrchestrator,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "V2DocumentProcessor",
    "create_processor",
    "BatchProcessor",
    "create_batch_processor",
    # Mapper
    "DoclingToV2Mapper",
    "create_mapper",
    # Schema
    "BoundingBox",
    "ChunkType",
    "FileType",
    "HierarchyMetadata",
    "IngestionChunk",
    "Modality",
    "create_image_chunk",
    "create_table_chunk",
    "create_text_chunk",
    # State
    "ContextStateV2",
    "create_context_state",
    # Utils
    "ensure_normalized",
    "normalize_bbox",
    "PDFBatchSplitter",
    "ImageHashRegistry",
    # Vision
    "VisionManager",
    "create_vision_manager",
    # Orchestration
    "DocumentProfile",
    "SmartConfigProvider",
    "ExtractionStrategy",
    "StrategyOrchestrator",
]
