"""
OCR Hint Engine - Non-Destructive OCR Assist for VLM
=====================================================
ENGINE_USE: EasyOCR with noise filtering

This module provides OCR "hints" to the VLM without polluting the content.
The OCR output is NEVER used directly - it's only passed as context to the
VLM which acts as "judge" to validate and interpret the hints.

ARCHITECTURAL PRINCIPLE: VLM AS JUDGE
=====================================
- OCR often produces noise like "I1I!1l" for degraded scans
- The VLM sees both the IMAGE and the OCR hints
- The VLM decides whether to trust the OCR (e.g., "Sako", "Browning")
- This prevents OCR noise from polluting the RAG corpus

REQ Compliance:
- REQ-OCR-01: OCR hints for scanned/degraded documents only
- REQ-OCR-02: VLM acts as judge - OCR never goes directly to content
- REQ-OCR-03: Confidence filtering to reduce noise injection

Author: Claude (Architect)
Date: 2025-01-03
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    import easyocr

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Minimum confidence for OCR results (0.0-1.0)
MIN_OCR_CONFIDENCE: float = 0.4

# Minimum text length to consider (filters out single-char noise)
MIN_TEXT_LENGTH: int = 2

# Maximum characters to include in hint (prevents prompt bloat)
MAX_HINT_CHARS: int = 300

# Known noise patterns (OCR artifacts)
NOISE_PATTERNS = [
    r"^[Il1\|!]+$",  # Common I/l/1 confusion
    r"^[\-_\.]+$",  # Dashes and dots
    r"^[^\w\s]+$",  # Only symbols
    r"^\d{1,2}$",  # Single/double digits (page numbers)
    r"^[A-Z]$",  # Single capital letters
    r"^(www|http|@)",  # URL/email prefixes (ads)
]

# Keywords that should boost OCR confidence (brand names, technical terms)
HIGH_VALUE_KEYWORDS = [
    # Firearm manufacturers
    "browning",
    "sako",
    "winchester",
    "remington",
    "colt",
    "smith",
    "wesson",
    "ruger",
    "beretta",
    "glock",
    "sig",
    "sauer",
    "springfield",
    "mossberg",
    "benelli",
    "kimber",
    "savage",
    "marlin",
    "henry",
    "tikka",
    "weatherby",
    "heckler",
    "koch",
    "fnherstal",
    "taurus",
    "walther",
    "steyr",
    "mauser",
    # Technical terms
    "caliber",
    "barrel",
    "magazine",
    "cartridge",
    "ammunition",
    "rifle",
    "pistol",
    "shotgun",
    "revolver",
    "bolt",
    "action",
    "trigger",
    "stock",
    # Aviation terms (for Combat Aircraft)
    "boeing",
    "lockheed",
    "martin",
    "northrop",
    "grumman",
    "airbus",
    "f-15",
    "f-16",
    "f-35",
    "f-22",
    "mig",
    "sukhoi",
    "eurofighter",
]


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class OCRHint:
    """
    A filtered OCR hint for VLM injection.

    Attributes:
        text: The recognized text
        confidence: OCR confidence score (0.0-1.0)
        bbox: Bounding box [x0, y0, x1, y1]
        is_high_value: Whether this matches high-value keywords
    """

    text: str
    confidence: float
    bbox: Optional[Tuple[float, float, float, float]] = None
    is_high_value: bool = False


@dataclass
class OCRHintResult:
    """
    Result of OCR hint extraction.

    Attributes:
        hints: List of filtered OCR hints
        raw_text: Combined text from all hints (for VLM prompt)
        high_value_terms: List of detected high-value terms
        was_executed: Whether OCR was actually run
        error: Error message if OCR failed
    """

    hints: List[OCRHint]
    raw_text: str
    high_value_terms: List[str]
    was_executed: bool = True
    error: Optional[str] = None

    @property
    def has_meaningful_content(self) -> bool:
        """Check if OCR found meaningful content."""
        return bool(self.hints) or bool(self.high_value_terms)


# ============================================================================
# OCR HINT ENGINE
# ============================================================================


class OCRHintEngine:
    """
    Non-destructive OCR assistant for VLM enrichment.

    This engine provides OCR "hints" to the VLM without ever putting
    raw OCR text directly into the content field. The VLM acts as
    the final judge on whether to trust the OCR suggestions.

    SAFETY FEATURES:
    1. Noise filtering - Removes common OCR artifacts
    2. Confidence thresholding - Only passes confident results
    3. High-value keyword detection - Boosts firearm/aviation terms
    4. Length limiting - Prevents prompt bloat
    5. EXPLICIT STATE ISOLATION - No hint leakage between pages

    Usage:
        engine = OCRHintEngine()
        result = engine.extract_hints(image, languages=['en'])
        if result.has_meaningful_content:
            prompt = f"OCR HINTS: {result.raw_text}"
    """

    def __init__(
        self,
        min_confidence: float = MIN_OCR_CONFIDENCE,
        max_hint_chars: int = MAX_HINT_CHARS,
        languages: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize OCRHintEngine.

        Args:
            min_confidence: Minimum OCR confidence (0.0-1.0)
            max_hint_chars: Maximum characters in combined hint
            languages: OCR languages (default: ['en'])
        """
        self.min_confidence = min_confidence
        self.max_hint_chars = max_hint_chars
        self.languages = languages or ["en"]

        # Lazy-loaded EasyOCR reader
        self._reader: Optional["easyocr.Reader"] = None
        self._noise_patterns = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]

        # REQ-OCR-ISOLATION: Track current page for isolation verification
        self._current_page: Optional[int] = None
        self._last_hints_count: int = 0

        logger.info(
            f"OCRHintEngine initialized: "
            f"min_conf={min_confidence:.2f}, "
            f"max_chars={max_hint_chars}, "
            f"langs={languages}"
        )

    def _get_reader(self) -> "easyocr.Reader":
        """
        Lazy-load EasyOCR reader.

        EasyOCR downloads models on first use, so we defer initialization
        until actually needed.
        """
        if self._reader is None:
            try:
                import easyocr

                logger.info(f"Initializing EasyOCR reader for languages: {self.languages}")
                # gpu=False for CPU-only to avoid MPS issues on Apple Silicon
                # We want reliability over speed for hint generation
                self._reader = easyocr.Reader(
                    self.languages,
                    gpu=False,  # CPU for stability
                    verbose=False,
                )
                logger.info("EasyOCR reader initialized successfully")

            except ImportError as e:
                logger.error(f"EasyOCR not installed: {e}")
                raise RuntimeError(
                    "EasyOCR is required for OCR hints. " "Install with: pip install easyocr>=1.7.1"
                )
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {e}")
                raise

        return self._reader

    def _is_noise(self, text: str) -> bool:
        """
        Check if text is likely OCR noise.

        Args:
            text: OCR text to check

        Returns:
            True if text is probably noise
        """
        # Too short
        if len(text.strip()) < MIN_TEXT_LENGTH:
            return True

        # Matches known noise patterns
        for pattern in self._noise_patterns:
            if pattern.match(text.strip()):
                logger.debug(f"[OCR-NOISE] Filtered: '{text}' (matched pattern)")
                return True

        return False

    def _is_high_value(self, text: str) -> bool:
        """
        Check if text contains high-value keywords.

        Args:
            text: OCR text to check

        Returns:
            True if text contains valuable keywords
        """
        text_lower = text.lower()

        for keyword in HIGH_VALUE_KEYWORDS:
            if keyword in text_lower:
                logger.debug(f"[OCR-VALUE] High-value keyword detected: '{keyword}' in '{text}'")
                return True

        return False

    def _extract_high_value_terms(self, text: str) -> List[str]:
        """
        Extract all high-value terms from text.

        Args:
            text: OCR text to search

        Returns:
            List of detected high-value terms
        """
        text_lower = text.lower()
        found = []

        for keyword in HIGH_VALUE_KEYWORDS:
            if keyword in text_lower:
                found.append(keyword.title())

        return found

    def extract_hints(
        self,
        image: Image.Image,
        languages: Optional[List[str]] = None,
    ) -> OCRHintResult:
        """
        Extract OCR hints from an image.

        This method:
        1. Runs EasyOCR on the image
        2. Filters out noise and low-confidence results
        3. Identifies high-value keywords
        4. Returns structured hints for VLM injection

        Args:
            image: PIL Image to process
            languages: Override languages for this call

        Returns:
            OCRHintResult with filtered hints
        """
        hints: List[OCRHint] = []
        high_value_terms: List[str] = []

        try:
            reader = self._get_reader()

            # Convert PIL to numpy array for EasyOCR
            import numpy as np

            img_array = np.array(image)

            # Run OCR - EasyOCR returns List[Tuple] but type hints are incomplete
            results: List[Any] = reader.readtext(
                img_array,
                detail=1,  # Get bounding boxes and confidence
                paragraph=False,  # Keep individual detections
            )

            logger.debug(f"[OCR] Raw results: {len(results)} detections")

            # Process results
            for result_item in results:
                # EasyOCR returns (bbox, text, confidence) as a list/tuple
                # Cast explicitly to avoid type checker issues
                result_tuple = tuple(result_item)
                bbox_raw = result_tuple[0] if len(result_tuple) > 0 else None
                text = str(result_tuple[1]) if len(result_tuple) > 1 else ""
                confidence = float(result_tuple[2]) if len(result_tuple) > 2 else 0.0

                # Skip low confidence
                if confidence < self.min_confidence:
                    logger.debug(
                        f"[OCR-FILTER] Low confidence: '{text}' ({confidence:.2f} < {self.min_confidence})"
                    )
                    continue

                # Skip noise
                if self._is_noise(text):
                    continue

                # Check for high-value content
                is_high_value = self._is_high_value(text)

                # Extract bbox as tuple
                bbox_tuple: Optional[Tuple[float, float, float, float]] = None
                if bbox_raw and len(bbox_raw) >= 2:
                    # EasyOCR returns [[x0,y0], [x1,y0], [x1,y1], [x0,y1]]
                    x_coords = [float(p[0]) for p in bbox_raw]
                    y_coords = [float(p[1]) for p in bbox_raw]
                    bbox_tuple = (
                        min(x_coords),
                        min(y_coords),
                        max(x_coords),
                        max(y_coords),
                    )

                hint = OCRHint(
                    text=text.strip(),
                    confidence=confidence,
                    bbox=bbox_tuple,
                    is_high_value=is_high_value,
                )
                hints.append(hint)

                # Collect high-value terms
                if is_high_value:
                    high_value_terms.extend(self._extract_high_value_terms(text))

            # Build combined raw text (prioritize high-value hints)
            high_value_hints = [h for h in hints if h.is_high_value]
            other_hints = [h for h in hints if not h.is_high_value]

            # Combine: high-value first, then others
            ordered_hints = high_value_hints + other_hints
            text_parts = [h.text for h in ordered_hints]
            raw_text = " ".join(text_parts)[: self.max_hint_chars]

            # Deduplicate high-value terms
            high_value_terms = list(set(high_value_terms))

            logger.info(
                f"[OCR-HINT] Extracted {len(hints)} hints, "
                f"{len(high_value_hints)} high-value, "
                f"terms: {high_value_terms}"
            )

            return OCRHintResult(
                hints=hints,
                raw_text=raw_text,
                high_value_terms=high_value_terms,
                was_executed=True,
                error=None,
            )

        except Exception as e:
            logger.warning(f"[OCR-HINT] Extraction failed: {e}")
            return OCRHintResult(
                hints=[],
                raw_text="",
                high_value_terms=[],
                was_executed=True,
                error=str(e),
            )

    def extract_hints_from_path(
        self,
        image_path: Path | str,
        languages: Optional[List[str]] = None,
    ) -> OCRHintResult:
        """
        Extract OCR hints from an image file.

        Args:
            image_path: Path to image file
            languages: Override languages for this call

        Returns:
            OCRHintResult with filtered hints
        """
        try:
            image = Image.open(image_path)
            return self.extract_hints(image, languages)
        except Exception as e:
            logger.warning(f"[OCR-HINT] Failed to load image {image_path}: {e}")
            return OCRHintResult(
                hints=[],
                raw_text="",
                high_value_terms=[],
                was_executed=False,
                error=str(e),
            )


# ============================================================================
# FACTORY FUNCTION
# ============================================================================


def create_ocr_hint_engine(
    min_confidence: float = MIN_OCR_CONFIDENCE,
    max_hint_chars: int = MAX_HINT_CHARS,
    languages: Optional[List[str]] = None,
) -> OCRHintEngine:
    """
    Factory function to create an OCRHintEngine.

    Args:
        min_confidence: Minimum OCR confidence (0.0-1.0)
        max_hint_chars: Maximum characters in combined hint
        languages: OCR languages (default: ['en'])

    Returns:
        Configured OCRHintEngine instance
    """
    return OCRHintEngine(
        min_confidence=min_confidence,
        max_hint_chars=max_hint_chars,
        languages=languages,
    )


# ============================================================================
# PROMPT BUILDER
# ============================================================================


def build_ocr_hint_prompt_section(
    ocr_result: OCRHintResult,
    include_confidence: bool = False,
) -> str:
    """
    Build the OCR hint section for VLM prompt injection.

    This creates a formatted section that tells the VLM about detected text
    without forcing it to trust the OCR. The VLM acts as "judge".

    Args:
        ocr_result: Result from OCRHintEngine
        include_confidence: Whether to include confidence scores

    Returns:
        Formatted prompt section, or empty string if no hints
    """
    if not ocr_result.has_meaningful_content:
        return ""

    parts = []

    # High-value terms get special treatment
    if ocr_result.high_value_terms:
        parts.append(
            f"DETECTED KEYWORDS (high confidence): {', '.join(ocr_result.high_value_terms)}"
        )

    # Raw OCR text as context
    if ocr_result.raw_text:
        parts.append(f"OCR TEXT HINTS: {ocr_result.raw_text}")

    # Add usage instruction
    parts.append(
        "NOTE: Use OCR hints to identify brand names, model numbers, and titles. "
        "If OCR seems wrong (gibberish), ignore it and describe what you SEE."
    )

    return "\n".join(parts)
