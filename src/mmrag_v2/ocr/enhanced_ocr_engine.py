"""
Enhanced OCR Engine with Confidence-Based Cascade
=================================================

3-Layer OCR cascade that auto-escalates on low confidence:
- Layer 1: Docling (existing, fastest)
- Layer 2: Tesseract 5.x + Image Preprocessing
- Layer 3: Doctr (transformer-based, most accurate)

Validated test results (January 3, 2026):
- Tesseract alone: INSUFFICIENT for vintage scans
- Doctr Layer 3: REQUIRED for acceptable accuracy

Author: Claude (Architect)
Date: January 3, 2026
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np

from .image_preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)


class OCRLayer(Enum):
    """OCR engine layers in the cascade."""

    DOCLING = "docling"
    TESSERACT = "tesseract"
    DOCTR = "doctr"


@dataclass
class OCRResult:
    """Result from OCR processing."""

    text: str
    confidence: float  # 0.0 - 1.0
    layer_used: OCRLayer
    word_confidences: Optional[List[float]] = None
    processing_time_ms: int = 0
    word_count: int = 0

    def __post_init__(self):
        """Calculate word count if not set."""
        if self.word_count == 0 and self.text:
            self.word_count = len(self.text.split())


class EnhancedOCREngine:
    """
    Cascade OCR engine that auto-escalates on low confidence.

    Architecture:
    1. Try Docling result (if provided) - fast path
    2. Try Tesseract with preprocessing - medium path
    3. Try Doctr - slow but accurate path

    Integration:
    - Called by BatchProcessor when Docling confidence < threshold
    - NOT a replacement for Docling, but an enhancement layer
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        enable_tesseract: bool = True,
        enable_doctr: bool = True,
        tesseract_psm: int = 3,  # Fully automatic page segmentation
        tesseract_lang: str = "eng",
    ):
        """
        Initialize the enhanced OCR engine.

        Args:
            confidence_threshold: Minimum acceptable confidence (0.0-1.0)
            enable_tesseract: Whether to use Tesseract as Layer 2
            enable_doctr: Whether to use Doctr as Layer 3
            tesseract_psm: Tesseract page segmentation mode
            tesseract_lang: Tesseract language code
        """
        self.confidence_threshold = confidence_threshold
        self.enable_tesseract = enable_tesseract
        self.enable_doctr = enable_doctr
        self.tesseract_psm = tesseract_psm
        self.tesseract_lang = tesseract_lang

        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor()

        # Lazy-load OCR engines
        self._tesseract_available: Optional[bool] = None
        self._doctr_model = None

    def process_page(
        self,
        page_image: np.ndarray,
        docling_result: Optional[OCRResult] = None,
    ) -> OCRResult:
        """
        Process a page through the OCR cascade.

        Args:
            page_image: RGB numpy array of the page (from PyMuPDF render)
            docling_result: Optional existing result from Docling

        Returns:
            Best OCRResult from the cascade
        """
        logger.info(f"[OCR-CASCADE] Starting cascade processing")

        # Layer 1: Check Docling result
        if docling_result and docling_result.confidence >= self.confidence_threshold:
            logger.info(
                f"[OCR-CASCADE] Layer 1 (Docling) accepted: "
                f"confidence={docling_result.confidence:.2f}"
            )
            return docling_result

        if docling_result:
            logger.info(
                f"[OCR-CASCADE] Layer 1 (Docling) insufficient: "
                f"confidence={docling_result.confidence:.2f} < {self.confidence_threshold}"
            )

        # Layer 2: Tesseract with preprocessing
        if self.enable_tesseract:
            tesseract_result = self._run_tesseract(page_image)

            if tesseract_result.confidence >= self.confidence_threshold:
                logger.info(
                    f"[OCR-CASCADE] Layer 2 (Tesseract) accepted: "
                    f"confidence={tesseract_result.confidence:.2f}"
                )
                return tesseract_result

            logger.info(
                f"[OCR-CASCADE] Layer 2 (Tesseract) insufficient: "
                f"confidence={tesseract_result.confidence:.2f} < {self.confidence_threshold}"
            )

        # Layer 3: Doctr (final fallback)
        if self.enable_doctr:
            doctr_result = self._run_doctr(page_image)
            logger.info(
                f"[OCR-CASCADE] Layer 3 (Doctr) final result: "
                f"confidence={doctr_result.confidence:.2f}"
            )
            return doctr_result

        # Fallback: Return best available result
        candidates = [r for r in [docling_result, tesseract_result] if r is not None]
        if candidates:
            best = max(candidates, key=lambda r: r.confidence)
            logger.warning(
                f"[OCR-CASCADE] No layer met threshold, returning best: "
                f"{best.layer_used.value} with confidence={best.confidence:.2f}"
            )
            return best

        # No result at all - return empty
        logger.error("[OCR-CASCADE] All layers failed, returning empty result")
        return OCRResult(
            text="",
            confidence=0.0,
            layer_used=OCRLayer.DOCLING,
            processing_time_ms=0,
        )

    def process_region(
        self,
        page_image: np.ndarray,
        bbox: tuple,
    ) -> OCRResult:
        """
        Process a specific region of a page.

        Use this for layout-aware OCR where you want to OCR
        only text regions, not the entire page.

        Args:
            page_image: Full page image
            bbox: Bounding box as (x0, y0, x1, y1)

        Returns:
            OCRResult for the region
        """
        # Crop the region with padding
        region_crop = self.preprocessor.crop_region(page_image, bbox, padding=10)

        # Process the cropped region
        return self.process_page(region_crop)

    def _run_tesseract(self, image: np.ndarray) -> OCRResult:
        """
        Run Tesseract OCR with preprocessing.

        Args:
            image: Input image (RGB or grayscale)

        Returns:
            OCRResult with text and confidence
        """
        import pytesseract

        start_time = time.perf_counter()

        try:
            # Preprocess image
            preprocessed = self.preprocessor.enhance_for_ocr(image)

            # Configure Tesseract
            config = f"--psm {self.tesseract_psm}"

            # Get detailed output with confidence
            data = pytesseract.image_to_data(
                preprocessed,
                lang=self.tesseract_lang,
                config=config,
                output_type=pytesseract.Output.DICT,
            )

            # Extract text and confidence
            words = []
            confidences = []

            for i, conf in enumerate(data["conf"]):
                conf_int = int(conf)
                text = data["text"][i].strip()

                if conf_int > 0 and text:  # Valid detection
                    words.append(text)
                    confidences.append(conf_int / 100.0)  # Convert to 0-1

            # Calculate average confidence
            avg_confidence = np.mean(confidences) if confidences else 0.0
            full_text = " ".join(words)

            elapsed_ms = int((time.perf_counter() - start_time) * 1000)

            logger.debug(
                f"[TESSERACT] Extracted {len(words)} words in {elapsed_ms}ms, "
                f"avg confidence: {avg_confidence:.2f}"
            )

            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                layer_used=OCRLayer.TESSERACT,
                word_confidences=confidences,
                processing_time_ms=elapsed_ms,
                word_count=len(words),
            )

        except Exception as e:
            logger.error(f"[TESSERACT] Failed: {e}")
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return OCRResult(
                text="",
                confidence=0.0,
                layer_used=OCRLayer.TESSERACT,
                processing_time_ms=elapsed_ms,
            )

    def _run_doctr(self, image: np.ndarray) -> OCRResult:
        """
        Run Doctr OCR (transformer-based).

        Doctr is slower but more accurate for degraded scans.
        Uses db_resnet50 for detection and CRNN for recognition.

        Args:
            image: Input image (RGB)

        Returns:
            OCRResult with text and confidence
        """
        start_time = time.perf_counter()

        try:
            # Lazy load doctr model
            if self._doctr_model is None:
                logger.info("[DOCTR] Loading model (first time)...")
                from doctr.io import DocumentFile
                from doctr.models import ocr_predictor

                # Use PyTorch backend (lighter than TensorFlow)
                self._doctr_model = ocr_predictor(pretrained=True)
                logger.info("[DOCTR] Model loaded successfully")

            # Convert image for doctr
            # Doctr expects RGB uint8 numpy array
            if len(image.shape) == 2:
                # Grayscale -> RGB
                image = np.stack([image] * 3, axis=-1)

            # Run OCR
            result = self._doctr_model([image])

            # Extract text and confidence from result
            words = []
            confidences = []

            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            words.append(word.value)
                            confidences.append(word.confidence)

            # Calculate average confidence
            avg_confidence = np.mean(confidences) if confidences else 0.0
            full_text = " ".join(words)

            elapsed_ms = int((time.perf_counter() - start_time) * 1000)

            logger.debug(
                f"[DOCTR] Extracted {len(words)} words in {elapsed_ms}ms, "
                f"avg confidence: {avg_confidence:.2f}"
            )

            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                layer_used=OCRLayer.DOCTR,
                word_confidences=confidences,
                processing_time_ms=elapsed_ms,
                word_count=len(words),
            )

        except Exception as e:
            logger.error(f"[DOCTR] Failed: {e}")
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return OCRResult(
                text="",
                confidence=0.0,
                layer_used=OCRLayer.DOCTR,
                processing_time_ms=elapsed_ms,
            )

    def get_layer_status(self) -> dict:
        """
        Get status of available OCR layers.

        Returns:
            Dict with layer availability status
        """
        status = {
            "docling": True,  # Assumed available via existing pipeline
            "tesseract": False,
            "doctr": False,
        }

        # Check Tesseract
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
            status["tesseract"] = True
        except Exception:
            pass

        # Check Doctr
        try:
            import doctr

            status["doctr"] = True
        except Exception:
            pass

        return status
