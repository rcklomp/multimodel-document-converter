"""
Image Preprocessor for OCR Enhancement
=======================================

Preprocessing pipeline optimized for vintage scanned documents (1900s-1950s).
Improves OCR accuracy by enhancing image quality before text extraction.

Pipeline:
1. Grayscale conversion
2. Deskewing (straighten rotated scans)
3. Noise removal (salt-and-pepper artifacts)
4. Adaptive thresholding (better than binary for old scans)
5. Contrast enhancement (CLAHE)

Author: Claude (Architect)
Date: January 3, 2026
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Image enhancement pipeline for OCR quality improvement.

    Specifically tuned for degraded historical document scans
    with age-related artifacts (yellowing, foxing, fading).
    """

    def __init__(
        self,
        # Adaptive threshold parameters
        adaptive_block_size: int = 11,
        adaptive_c: int = 2,
        # Denoise parameters
        denoise_strength: int = 10,
        # Deskew parameters
        deskew_threshold: float = 0.5,
        # CLAHE parameters
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
    ):
        """
        Initialize preprocessor with tunable parameters.

        Args:
            adaptive_block_size: Block size for adaptive thresholding (must be odd)
            adaptive_c: Constant subtracted from mean in adaptive threshold
            denoise_strength: Strength of noise removal (higher = more aggressive)
            deskew_threshold: Minimum rotation angle to trigger deskew (degrees)
            clahe_clip_limit: Clip limit for CLAHE contrast enhancement
            clahe_grid_size: Grid size for CLAHE
        """
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
        self.denoise_strength = denoise_strength
        self.deskew_threshold = deskew_threshold
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size

        # Create CLAHE object
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_grid_size
        )

    def enhance_for_ocr(
        self,
        image: np.ndarray,
        apply_deskew: bool = True,
        apply_denoise: bool = True,
        apply_threshold: bool = True,
        apply_contrast: bool = True,
    ) -> np.ndarray:
        """
        Full enhancement pipeline for OCR.

        Args:
            image: Input image (RGB or grayscale numpy array)
            apply_deskew: Whether to straighten rotated images
            apply_denoise: Whether to remove noise
            apply_threshold: Whether to apply adaptive thresholding
            apply_contrast: Whether to apply CLAHE contrast enhancement

        Returns:
            Enhanced grayscale image ready for OCR
        """
        logger.debug(f"[PREPROCESS] Starting enhancement pipeline on {image.shape} image")

        # Step 1: Convert to grayscale
        gray = self._to_grayscale(image)
        logger.debug(f"[PREPROCESS] Grayscale conversion complete")

        # Step 2: Deskew (straighten rotated scans)
        if apply_deskew:
            gray = self._deskew(gray)

        # Step 3: Contrast enhancement (CLAHE) - before thresholding
        if apply_contrast:
            gray = self._enhance_contrast(gray)
            logger.debug(f"[PREPROCESS] Contrast enhancement complete")

        # Step 4: Denoise
        if apply_denoise:
            gray = self._denoise(gray)
            logger.debug(f"[PREPROCESS] Noise removal complete")

        # Step 5: Adaptive thresholding (binarization)
        if apply_threshold:
            gray = self._adaptive_threshold(gray)
            logger.debug(f"[PREPROCESS] Adaptive thresholding complete")

        logger.debug(f"[PREPROCESS] Enhancement pipeline complete")
        return gray

    def enhance_light(self, image: np.ndarray) -> np.ndarray:
        """
        Light enhancement for already-clean images.

        Only applies grayscale + contrast. No thresholding.
        Use this for images that are already reasonably clean.

        Args:
            image: Input image (RGB or grayscale)

        Returns:
            Lightly enhanced grayscale image
        """
        gray = self._to_grayscale(image)
        enhanced = self._enhance_contrast(gray)
        return enhanced

    def enhance_aggressive(self, image: np.ndarray) -> np.ndarray:
        """
        Aggressive enhancement for severely degraded scans.

        Applies all enhancements with stronger parameters.

        Args:
            image: Input image (RGB or grayscale)

        Returns:
            Aggressively enhanced grayscale image
        """
        gray = self._to_grayscale(image)
        gray = self._deskew(gray)
        gray = self._enhance_contrast(gray)

        # Stronger denoising
        gray = cv2.fastNlMeansDenoising(gray, h=15)

        # More aggressive threshold
        gray = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=4,
        )

        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        return gray

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed."""
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                # RGBA -> RGB -> Gray
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct rotation in scanned documents.

        Uses minAreaRect on contours to detect skew angle.
        Only corrects if angle exceeds threshold.
        """
        try:
            # Find coordinates of all non-zero pixels
            coords = np.column_stack(np.where(image > 0))

            if len(coords) < 100:
                # Not enough pixels to detect angle
                return image

            # Get minimum area rectangle
            angle = cv2.minAreaRect(coords)[-1]

            # Adjust angle
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            # Only correct if angle is significant
            if abs(angle) < self.deskew_threshold:
                return image

            logger.debug(f"[PREPROCESS] Deskewing by {angle:.2f} degrees")

            # Rotate image
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )

            return rotated

        except Exception as e:
            logger.warning(f"[PREPROCESS] Deskew failed: {e}")
            return image

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Remove noise using Non-local Means Denoising."""
        return cv2.fastNlMeansDenoising(image, h=self.denoise_strength)

    def _adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive thresholding for binarization.

        Adaptive threshold works better than global threshold for
        documents with uneven lighting or age-related darkening.
        """
        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=self.adaptive_block_size,
            C=self.adaptive_c,
        )

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

        CLAHE works better than standard histogram equalization for
        documents because it prevents over-amplification of noise.
        """
        return self._clahe.apply(image)

    def crop_region(
        self, image: np.ndarray, bbox: Tuple[int, int, int, int], padding: int = 10
    ) -> np.ndarray:
        """
        Crop a region from the image with padding.

        Args:
            image: Source image
            bbox: Bounding box as (x0, y0, x1, y1)
            padding: Pixels to add around the crop (REQ-MM-01: 10px)

        Returns:
            Cropped region with padding
        """
        x0, y0, x1, y1 = bbox
        h, w = image.shape[:2]

        # Apply padding with bounds checking
        x0 = max(0, x0 - padding)
        y0 = max(0, y0 - padding)
        x1 = min(w, x1 + padding)
        y1 = min(h, y1 + padding)

        return image[y0:y1, x0:x1]

    def estimate_quality(self, image: np.ndarray) -> float:
        """
        Estimate image quality for OCR.

        Returns a score from 0.0 (terrible) to 1.0 (excellent).
        Uses multiple metrics:
        - Contrast (standard deviation)
        - Sharpness (Laplacian variance)
        - Noise level

        Args:
            image: Input image (RGB or grayscale)

        Returns:
            Quality score between 0.0 and 1.0
        """
        gray = self._to_grayscale(image)

        # Contrast: standard deviation of pixel values
        contrast = np.std(gray) / 128.0  # Normalize to ~1.0 for good contrast

        # Sharpness: variance of Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var() / 1000.0  # Normalize

        # Combine metrics (weighted average)
        quality = 0.5 * min(contrast, 1.0) + 0.5 * min(sharpness, 1.0)

        return min(max(quality, 0.0), 1.0)


# Convenience function for simple usage
def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Simple function to preprocess an image for OCR.

    Args:
        image: Input image (RGB or grayscale numpy array)

    Returns:
        Enhanced grayscale image ready for OCR
    """
    preprocessor = ImagePreprocessor()
    return preprocessor.enhance_for_ocr(image)
