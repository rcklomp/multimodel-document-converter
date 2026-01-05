"""
MMRAG V2 - Multimodal RAG Document Converter
=============================================
ENGINE_USE: Docling v2.66.0 (Native Layout Analysis)

High-fidelity ETL pipeline for multimodal document ingestion.
Converts PDF, EPUB, HTML, DOCX documents into validated JSONL
output suitable for downstream RAG systems.

Features:
- Docling v2.66.0 native layout analysis
- Hierarchical breadcrumb tracking (ContextStateV2)
- VLM-based image enrichment (Ollama, OpenAI, Anthropic)
- Shadow extraction for missed images
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
Version: 2.0.0
"""
from __future__ import annotations

__version__ = "2.0.0"
__author__ = "Claude 4.5 Opus (Architect)"

# Core processor
from .processor import V2DocumentProcessor, create_processor

# Batch processor
from .batch_processor import BatchProcessor, create_batch_processor

# Mapper
from .mapper import DoclingToV2Mapper, create_mapper

# Schema
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
    create_shadow_chunk,
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

# Orchestration
from .orchestration import (
    DocumentProfile,
    SmartConfigProvider,
    ExtractionStrategy,
    StrategyOrchestrator,
    ShadowExtractor,
    create_shadow_extractor,
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
    "create_shadow_chunk",
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
    "ShadowExtractor",
    "create_shadow_extractor",
]
