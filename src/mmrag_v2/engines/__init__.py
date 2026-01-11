"""
Format Engines Module
======================

This module provides format-specific extraction engines that convert
documents to the Universal Intermediate Representation (UIR).

Available Engines:
    - PDFEngine: PDF extraction via Docling v2.66.0
    - EpubEngine: ePub extraction via EbookLib
    - HTMLEngine: HTML extraction via Trafilatura

Usage:
    from mmrag_v2.engines import PDFEngine

    engine = PDFEngine()
    uir = engine.convert("document.pdf")

Engine Contract:
    All engines inherit from FormatEngine ABC and implement:
    - detect(file_path) -> bool: Check if engine can handle file
    - convert(file_path) -> UniversalDocument: Extract to UIR
    - supported_extensions: List of supported file extensions

Author: Claude (Architect)
Date: January 2026
"""

from .base import FormatEngine

__all__ = [
    "FormatEngine",
]
