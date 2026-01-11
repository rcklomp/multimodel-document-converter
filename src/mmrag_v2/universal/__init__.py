"""
Universal Intermediate Representation (UIR) Module
===================================================

This module provides format-agnostic data structures for document processing.
All format engines (PDF, ePub, HTML, DOCX) convert their input to these
universal structures, enabling quality-based routing regardless of source format.

Key Components:
    - intermediate.py: Core data structures (UniversalDocument, UniversalPage, Element)
    - router.py: Format detection and engine routing
    - quality_classifier.py: Page/element quality assessment
    - element_processor.py: Quality-based element processing to IngestionChunks

Usage:
    from mmrag_v2.universal import (
        UniversalDocument,
        UniversalPage,
        Element,
        ElementType,
        PageClassification,
        FormatRouter,
        ElementProcessor,
    )

    # Format engines produce UIR
    doc = pdf_engine.convert("document.pdf")

    # Element processor routes by quality
    for chunk in element_processor.process(doc):
        yield chunk

Author: Claude (Architect)
Date: January 2026
"""

from .intermediate import (
    BoundingBox,
    Element,
    ElementType,
    ExtractionMethod,
    PageClassification,
    UniversalPage,
    UniversalDocument,
    DocumentMetadata,
    create_element,
    create_page,
    create_document,
)
from .router import FormatRouter, create_router, get_router
from .element_processor import ElementProcessor, ProcessingResult, create_element_processor
from .quality_classifier import (
    ConfidenceNormalizer,
    ConfidenceThreshold,
    QualityTier,
    ElementConfidence,
    ElementConfidenceCalculator,
    PageQuality,
    PageQualityClassifier,
)

__all__ = [
    # Data Structures
    "BoundingBox",
    "Element",
    "ElementType",
    "ExtractionMethod",
    "PageClassification",
    "UniversalPage",
    "UniversalDocument",
    "DocumentMetadata",
    # Factory Functions
    "create_element",
    "create_page",
    "create_document",
    # Router
    "FormatRouter",
    "create_router",
    "get_router",
    # Element Processor
    "ElementProcessor",
    "ProcessingResult",
    "create_element_processor",
    # Quality Classifier
    "ConfidenceNormalizer",
    "ConfidenceThreshold",
    "QualityTier",
    "ElementConfidence",
    "ElementConfidenceCalculator",
    "PageQuality",
    "PageQualityClassifier",
]
