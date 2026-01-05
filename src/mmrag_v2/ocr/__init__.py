"""
Enhanced OCR Module (Phase 1A + 1B)
===================================

Provides layout-aware OCR with confidence-based cascade:
- Layer 1: Docling (existing, for digital PDFs)
- Layer 2: Tesseract 5.x + Image Preprocessing
- Layer 3: Doctr (for degraded scans)

Components:
- ImagePreprocessor: Deskew, denoise, contrast enhancement
- EnhancedOCREngine: 3-layer cascade with confidence routing
- LayoutAwareOCRProcessor: Combines layout detection with OCR cascade
"""

from .image_preprocessor import ImagePreprocessor
from .enhanced_ocr_engine import EnhancedOCREngine, OCRLayer, OCRResult
from .layout_aware_processor import LayoutAwareOCRProcessor, Region, ProcessedChunk

__all__ = [
    "ImagePreprocessor",
    "EnhancedOCREngine",
    "OCRLayer",
    "OCRResult",
    "LayoutAwareOCRProcessor",
    "Region",
    "ProcessedChunk",
]
