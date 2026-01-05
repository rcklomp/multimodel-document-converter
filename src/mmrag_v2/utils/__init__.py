"""
Utils module for MM-Converter-V2.
Contains utility functions for coordinate normalization, PDF splitting, and image hashing.
"""
from .coordinate_normalization import (
    ensure_normalized,
    normalize_bbox,
    validate_bbox_strict,
)
from .pdf_splitter import (
    BatchInfo,
    PDFBatchSplitter,
    SplitResult,
)
from .image_hash_registry import (
    DuplicateInfo,
    ImageHashRegistry,
    create_image_hash_registry,
    create_page1_validator,
)

__all__ = [
    # Coordinate normalization
    "ensure_normalized",
    "normalize_bbox",
    "validate_bbox_strict",
    # PDF splitting
    "BatchInfo",
    "PDFBatchSplitter",
    "SplitResult",
    # Image hash registry
    "DuplicateInfo",
    "ImageHashRegistry",
    "create_image_hash_registry",
    "create_page1_validator",
]
