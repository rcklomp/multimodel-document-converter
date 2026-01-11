"""
Quality Classifier - Normalize Confidence Across Formats
=========================================================

This module solves the critical problem of making confidence scores
comparable across different extraction formats. Each format uses
different confidence metrics:

- PDF (Docling): Font detection confidence (0.0-1.0)
- ePub: Text extraction success (character count based)
- HTML (Trafilatura): Content quality score (0-100)
- Office (Docx/Pptx): Structure extraction success

The ConfidenceNormalizer maps all these to a universal 0.0-1.0 scale
enabling quality-based routing regardless of source format.

SRS Compliance:
    - REQ-QUALITY-01: Confidence scores must be comparable across formats
    - REQ-OCR-01: Confidence < 0.7 triggers OCR cascade

Author: Claude (Architect)
Date: January 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIDENCE THRESHOLDS (Universal)
# ============================================================================


class ConfidenceThreshold:
    """Universal confidence thresholds for routing decisions."""

    # Above this: high-quality extraction, use directly
    HIGH = 0.85

    # Above this: medium quality, validate but use
    MEDIUM = 0.7

    # Above this: low quality, consider OCR enhancement
    LOW = 0.5

    # Below LOW: very poor quality, always needs OCR/VLM

    # Threshold for triggering OCR cascade
    OCR_TRIGGER = 0.7


class QualityTier(Enum):
    """Quality classification tiers."""

    EXCELLENT = "excellent"  # >= 0.85
    GOOD = "good"  # >= 0.70
    FAIR = "fair"  # >= 0.50
    POOR = "poor"  # < 0.50


# ============================================================================
# CONFIDENCE NORMALIZER
# ============================================================================


class ConfidenceNormalizer:
    """
    Normalize format-specific confidence to universal 0.0-1.0 scale.

    This class addresses the fundamental problem that confidence scores
    from different extraction engines are NOT directly comparable:

    - Docling: Measures font embedding and text layer quality
    - Tesseract: Measures OCR character confidence
    - Trafilatura: Measures content extraction quality
    - EbookLib: Binary success/failure with text length heuristic

    The normalizer maps all of these to a universal scale where:
    - 0.0 = Complete failure / no usable content
    - 0.5 = Borderline quality / may need enhancement
    - 0.7 = Good quality / usable with validation
    - 1.0 = Perfect extraction / high confidence

    Usage:
        normalizer = ConfidenceNormalizer()

        # PDF confidence
        pdf_conf = normalizer.normalize_pdf(docling_score=0.85, text_length=500)

        # ePub confidence
        epub_conf = normalizer.normalize_epub(text_length=1500, has_toc=True)

        # HTML confidence
        html_conf = normalizer.normalize_html(trafilatura_score=75, text_length=800)

        # Compare across formats
        best_source = max([pdf_conf, epub_conf, html_conf])
    """

    # ========================================================================
    # PDF NORMALIZATION
    # ========================================================================

    @staticmethod
    def normalize_pdf(
        docling_score: Optional[float] = None,
        text_length: int = 0,
        has_images: bool = False,
        page_image_ratio: float = 0.0,
    ) -> float:
        """
        Normalize PDF extraction confidence.

        Docling provides various confidence signals:
        - Text layer quality (font embedding)
        - Layout detection confidence
        - OCR fallback indicators

        Args:
            docling_score: Docling's native confidence (0.0-1.0)
            text_length: Extracted text character count
            has_images: Whether page contains images
            page_image_ratio: Ratio of page covered by images

        Returns:
            Normalized confidence (0.0-1.0)
        """
        # Start with Docling score if available
        if docling_score is not None:
            base_conf = float(docling_score)
        else:
            # Fallback: estimate from text length
            base_conf = ConfidenceNormalizer._text_length_heuristic(text_length)

        # Adjust for image-heavy pages (may have embedded text in images)
        if page_image_ratio > 0.5:
            # High image ratio may indicate scanned content
            base_conf *= 0.9

        # Adjust for very short text (may indicate extraction failure)
        if text_length < 50:
            base_conf *= 0.5
        elif text_length < 100:
            base_conf *= 0.7

        return max(0.0, min(1.0, base_conf))

    # ========================================================================
    # EPUB NORMALIZATION
    # ========================================================================

    @staticmethod
    def normalize_epub(
        text_length: int = 0,
        has_toc: bool = False,
        chapter_count: int = 0,
        image_count: int = 0,
        extraction_errors: int = 0,
    ) -> float:
        """
        Normalize ePub extraction confidence.

        ePub extraction is typically binary (success/failure), but
        quality varies based on:
        - Text content length
        - Table of contents presence
        - Chapter structure
        - Image extraction success

        Args:
            text_length: Total extracted text characters
            has_toc: Whether TOC was extracted
            chapter_count: Number of chapters found
            image_count: Number of images extracted
            extraction_errors: Count of extraction errors

        Returns:
            Normalized confidence (0.0-1.0)
        """
        # Start with text length heuristic
        base_conf = ConfidenceNormalizer._text_length_heuristic(text_length)

        # Boost for structural elements
        if has_toc:
            base_conf = min(1.0, base_conf + 0.1)

        if chapter_count > 3:
            base_conf = min(1.0, base_conf + 0.05)

        # Penalize for extraction errors
        if extraction_errors > 0:
            penalty = min(0.3, extraction_errors * 0.05)
            base_conf -= penalty

        return max(0.0, min(1.0, base_conf))

    # ========================================================================
    # HTML NORMALIZATION
    # ========================================================================

    @staticmethod
    def normalize_html(
        trafilatura_score: Optional[float] = None,
        text_length: int = 0,
        has_article_tag: bool = False,
        has_main_content: bool = False,
        boilerplate_ratio: float = 0.0,
    ) -> float:
        """
        Normalize HTML extraction confidence.

        Trafilatura returns a quality score (0-100) based on:
        - Content extraction success
        - Boilerplate removal effectiveness
        - Structural element detection

        Args:
            trafilatura_score: Trafilatura quality score (0-100)
            text_length: Extracted text character count
            has_article_tag: Whether <article> tag was found
            has_main_content: Whether main content was identified
            boilerplate_ratio: Ratio of removed boilerplate

        Returns:
            Normalized confidence (0.0-1.0)
        """
        if trafilatura_score is not None:
            # Trafilatura uses 0-100 scale
            base_conf = float(trafilatura_score) / 100.0
        else:
            # Fallback to text length heuristic
            base_conf = ConfidenceNormalizer._text_length_heuristic(text_length)

        # Boost for semantic structure
        if has_article_tag:
            base_conf = min(1.0, base_conf + 0.05)

        if has_main_content:
            base_conf = min(1.0, base_conf + 0.05)

        # Penalize high boilerplate ratio
        if boilerplate_ratio > 0.7:
            base_conf *= 0.9

        return max(0.0, min(1.0, base_conf))

    # ========================================================================
    # OFFICE FORMAT NORMALIZATION
    # ========================================================================

    @staticmethod
    def normalize_docx(
        text_length: int = 0,
        paragraph_count: int = 0,
        has_styles: bool = False,
        table_count: int = 0,
        image_count: int = 0,
    ) -> float:
        """
        Normalize DOCX extraction confidence.

        DOCX extraction via python-docx is generally reliable,
        but confidence varies with document complexity.

        Args:
            text_length: Extracted text characters
            paragraph_count: Number of paragraphs
            has_styles: Whether document has custom styles
            table_count: Number of tables
            image_count: Number of images

        Returns:
            Normalized confidence (0.0-1.0)
        """
        base_conf = ConfidenceNormalizer._text_length_heuristic(text_length)

        # DOCX extraction is generally reliable
        if text_length > 100:
            base_conf = max(base_conf, 0.85)

        # Boost for structured content
        if paragraph_count > 5:
            base_conf = min(1.0, base_conf + 0.05)

        if has_styles:
            base_conf = min(1.0, base_conf + 0.03)

        return max(0.0, min(1.0, base_conf))

    @staticmethod
    def normalize_pptx(
        slide_count: int = 0,
        text_length: int = 0,
        has_notes: bool = False,
        image_count: int = 0,
    ) -> float:
        """
        Normalize PPTX extraction confidence.

        PowerPoint slides have unique challenges:
        - Text often in text boxes (not linear)
        - Speaker notes may contain important content
        - Images may contain text

        Args:
            slide_count: Number of slides
            text_length: Extracted text characters
            has_notes: Whether speaker notes exist
            image_count: Number of images

        Returns:
            Normalized confidence (0.0-1.0)
        """
        if slide_count == 0:
            return 0.0

        # Base confidence on text per slide
        text_per_slide = text_length / slide_count if slide_count > 0 else 0

        if text_per_slide > 200:
            base_conf = 0.85
        elif text_per_slide > 50:
            base_conf = 0.7
        else:
            base_conf = 0.5  # Likely image-heavy presentation

        # Boost for speaker notes
        if has_notes:
            base_conf = min(1.0, base_conf + 0.1)

        return max(0.0, min(1.0, base_conf))

    @staticmethod
    def normalize_xlsx(
        cell_count: int = 0,
        text_length: int = 0,
        formula_count: int = 0,
        sheet_count: int = 1,
    ) -> float:
        """
        Normalize XLSX extraction confidence.

        Spreadsheets are primarily data, not narrative text.
        Confidence is based on data extraction success.

        Args:
            cell_count: Number of non-empty cells
            text_length: Total text characters
            formula_count: Number of formulas
            sheet_count: Number of sheets

        Returns:
            Normalized confidence (0.0-1.0)
        """
        if cell_count == 0:
            return 0.0

        # Spreadsheets are structural - high confidence if cells extracted
        if cell_count > 100:
            base_conf = 0.9
        elif cell_count > 10:
            base_conf = 0.8
        else:
            base_conf = 0.6

        return max(0.0, min(1.0, base_conf))

    # ========================================================================
    # OCR CONFIDENCE NORMALIZATION
    # ========================================================================

    @staticmethod
    def normalize_ocr_tesseract(
        tesseract_conf: float,
        word_count: int = 0,
    ) -> float:
        """
        Normalize Tesseract OCR confidence.

        Tesseract reports confidence per word (0-100), typically
        aggregated as mean confidence.

        Args:
            tesseract_conf: Mean word confidence (0-100)
            word_count: Number of words recognized

        Returns:
            Normalized confidence (0.0-1.0)
        """
        # Tesseract uses 0-100 scale
        base_conf = float(tesseract_conf) / 100.0

        # Penalize very short results
        if word_count < 5:
            base_conf *= 0.7

        return max(0.0, min(1.0, base_conf))

    @staticmethod
    def normalize_ocr_doctr(
        doctr_conf: float,
        line_count: int = 0,
    ) -> float:
        """
        Normalize Doctr OCR confidence.

        Doctr reports confidence per line (0.0-1.0).

        Args:
            doctr_conf: Mean line confidence (0.0-1.0)
            line_count: Number of lines recognized

        Returns:
            Normalized confidence (0.0-1.0)
        """
        # Doctr already uses 0-1 scale
        base_conf = float(doctr_conf)

        # Doctr tends to be more conservative
        # Slight boost for non-empty results
        if line_count > 0:
            base_conf = min(1.0, base_conf + 0.05)

        return max(0.0, min(1.0, base_conf))

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    @staticmethod
    def _text_length_heuristic(text_length: int) -> float:
        """
        Estimate confidence from text length alone.

        Used as fallback when no native confidence is available.

        Args:
            text_length: Character count of extracted text

        Returns:
            Estimated confidence (0.0-1.0)
        """
        if text_length > 2000:
            return 0.9  # Substantial text = likely good extraction
        elif text_length > 500:
            return 0.8
        elif text_length > 100:
            return 0.7
        elif text_length > 20:
            return 0.5
        else:
            return 0.2  # Very short = likely failed extraction

    @staticmethod
    def classify_tier(confidence: float) -> QualityTier:
        """
        Classify confidence into quality tier.

        Args:
            confidence: Normalized confidence (0.0-1.0)

        Returns:
            QualityTier enum value
        """
        if confidence >= ConfidenceThreshold.HIGH:
            return QualityTier.EXCELLENT
        elif confidence >= ConfidenceThreshold.MEDIUM:
            return QualityTier.GOOD
        elif confidence >= ConfidenceThreshold.LOW:
            return QualityTier.FAIR
        else:
            return QualityTier.POOR

    @staticmethod
    def needs_ocr(confidence: float) -> bool:
        """
        Determine if content needs OCR enhancement.

        Args:
            confidence: Normalized confidence

        Returns:
            True if OCR cascade should be triggered
        """
        return confidence < ConfidenceThreshold.OCR_TRIGGER


# ============================================================================
# ELEMENT CONFIDENCE CALCULATOR
# ============================================================================


@dataclass
class ElementConfidence:
    """
    Confidence scores for a document element.

    Attributes:
        extraction_confidence: How well content was extracted (0.0-1.0)
        layout_confidence: How accurate is position/bbox (0.0-1.0)
        classification_confidence: How certain is element type (0.0-1.0)
        combined_confidence: Weighted combination (0.0-1.0)
        tier: Quality tier classification
        needs_ocr: Whether OCR enhancement is recommended
    """

    extraction_confidence: float
    layout_confidence: float
    classification_confidence: float
    combined_confidence: float
    tier: QualityTier
    needs_ocr: bool


class ElementConfidenceCalculator:
    """
    Calculate comprehensive confidence for document elements.

    Combines multiple confidence signals into a single quality score
    that can be used for routing decisions.

    Usage:
        calculator = ElementConfidenceCalculator()

        conf = calculator.calculate(
            extraction_conf=0.85,
            layout_conf=0.9,
            classification_conf=0.95,
        )

        if conf.needs_ocr:
            run_ocr_cascade(element)
    """

    # Weights for combining confidence scores
    EXTRACTION_WEIGHT = 0.6
    LAYOUT_WEIGHT = 0.2
    CLASSIFICATION_WEIGHT = 0.2

    def calculate(
        self,
        extraction_conf: float,
        layout_conf: float = 1.0,
        classification_conf: float = 1.0,
    ) -> ElementConfidence:
        """
        Calculate combined element confidence.

        Args:
            extraction_conf: Content extraction confidence
            layout_conf: Position/bbox confidence
            classification_conf: Element type classification confidence

        Returns:
            ElementConfidence with all scores
        """
        # Weighted combination
        combined = (
            extraction_conf * self.EXTRACTION_WEIGHT
            + layout_conf * self.LAYOUT_WEIGHT
            + classification_conf * self.CLASSIFICATION_WEIGHT
        )

        return ElementConfidence(
            extraction_confidence=extraction_conf,
            layout_confidence=layout_conf,
            classification_confidence=classification_conf,
            combined_confidence=combined,
            tier=ConfidenceNormalizer.classify_tier(combined),
            needs_ocr=ConfidenceNormalizer.needs_ocr(extraction_conf),
        )


# ============================================================================
# PAGE QUALITY CLASSIFIER
# ============================================================================


@dataclass
class PageQuality:
    """
    Quality assessment for a document page.

    Attributes:
        avg_confidence: Average element confidence
        min_confidence: Lowest element confidence
        max_confidence: Highest element confidence
        element_count: Number of elements
        ocr_needed_count: Elements needing OCR
        tier: Overall quality tier
        recommendation: Processing recommendation
    """

    avg_confidence: float
    min_confidence: float
    max_confidence: float
    element_count: int
    ocr_needed_count: int
    tier: QualityTier
    recommendation: str


class PageQualityClassifier:
    """
    Assess overall page quality from element confidences.

    Usage:
        classifier = PageQualityClassifier()

        quality = classifier.assess([0.9, 0.85, 0.3, 0.92])

        print(quality.recommendation)
        # "3 of 4 elements high quality, 1 needs OCR"
    """

    def assess(self, element_confidences: List[float]) -> PageQuality:
        """
        Assess page quality from element confidences.

        Args:
            element_confidences: List of element confidence scores

        Returns:
            PageQuality assessment
        """
        if not element_confidences:
            return PageQuality(
                avg_confidence=0.0,
                min_confidence=0.0,
                max_confidence=0.0,
                element_count=0,
                ocr_needed_count=0,
                tier=QualityTier.POOR,
                recommendation="No elements to assess",
            )

        avg_conf = sum(element_confidences) / len(element_confidences)
        min_conf = min(element_confidences)
        max_conf = max(element_confidences)
        ocr_count = sum(1 for c in element_confidences if c < ConfidenceThreshold.OCR_TRIGGER)

        tier = ConfidenceNormalizer.classify_tier(avg_conf)

        # Generate recommendation
        if ocr_count == 0:
            recommendation = f"All {len(element_confidences)} elements high quality"
        elif ocr_count == len(element_confidences):
            recommendation = f"All {len(element_confidences)} elements need OCR enhancement"
        else:
            high_quality = len(element_confidences) - ocr_count
            recommendation = f"{high_quality} of {len(element_confidences)} elements high quality, {ocr_count} needs OCR"

        return PageQuality(
            avg_confidence=avg_conf,
            min_confidence=min_conf,
            max_confidence=max_conf,
            element_count=len(element_confidences),
            ocr_needed_count=ocr_count,
            tier=tier,
            recommendation=recommendation,
        )
