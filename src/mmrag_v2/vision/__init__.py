"""
Vision module for MM-Converter-V2.
Contains VisionManager for VLM-based image enrichment.
"""

from .vision_manager import (
    VisionManager,
    create_vision_manager,
)
from .ocr_hint_engine import (
    OCRHintEngine,
    OCRHintResult,
    create_ocr_hint_engine,
    build_ocr_hint_prompt_section,
)

__all__ = [
    "VisionManager",
    "create_vision_manager",
    "OCRHintEngine",
    "OCRHintResult",
    "create_ocr_hint_engine",
    "build_ocr_hint_prompt_section",
]
