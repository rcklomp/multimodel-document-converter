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
import gc
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

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

        # Geometric indentation reconstruction (for code-like OCR blocks).
        self._indent_spaces_per_level = 4
        self._max_indent_spaces = 20
        self._doctr_indent_threshold = 0.01  # normalized X offset
        self._doctr_indent_step = 0.01  # normalized X per indent level
        self._tesseract_indent_threshold_ratio = 0.02  # relative to crop width
        self._tesseract_indent_step_ratio = 0.02  # relative to crop width
        self._doctr_merge_max_vertical_gap = 0.02  # normalized Y gap
        self._doctr_merge_gap_height_multiplier = 1.25  # relative to line height
        self._doctr_merge_min_overlap_ratio = 0.35
        self._doctr_merge_left_tolerance = 0.03
        self._doctr_merge_max_indent_shift = 0.16
        self._doctr_merge_max_seed_lines = 4

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
        logger.debug(f"[OCR-CASCADE] Starting cascade processing")

        # Layer 1: Check Docling result
        if docling_result and docling_result.confidence >= self.confidence_threshold:
            logger.debug(
                f"[OCR-CASCADE] Layer 1 (Docling) accepted: "
                f"confidence={docling_result.confidence:.2f}"
            )
            return docling_result

        if docling_result:
            logger.debug(
                f"[OCR-CASCADE] Layer 1 (Docling) insufficient: "
                f"confidence={docling_result.confidence:.2f} < {self.confidence_threshold}"
            )

        # Layer 2: Tesseract with preprocessing
        if self.enable_tesseract:
            tesseract_result = self._run_tesseract(page_image)

            if tesseract_result.confidence >= self.confidence_threshold:
                logger.debug(
                    f"[OCR-CASCADE] Layer 2 (Tesseract) accepted: "
                    f"confidence={tesseract_result.confidence:.2f}"
                )
                return tesseract_result

            logger.debug(
                f"[OCR-CASCADE] Layer 2 (Tesseract) insufficient: "
                f"confidence={tesseract_result.confidence:.2f} < {self.confidence_threshold}"
            )

        # Layer 3: Doctr (final fallback)
        if self.enable_doctr:
            doctr_result = self._run_doctr(page_image)
            logger.debug(
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

            # Extract text and confidence.
            # Preserve OCR line boundaries so downstream code chunks can retain indentation/newlines.
            line_words: Dict[Tuple[int, int, int], List[str]] = {}
            line_order: List[Tuple[int, int, int]] = []
            confidences = []

            for i, conf in enumerate(data["conf"]):
                try:
                    conf_int = int(float(conf))
                except Exception:
                    continue
                text = data["text"][i].strip()

                if conf_int > 0 and text:  # Valid detection
                    block_num = int(data.get("block_num", [0])[i])
                    par_num = int(data.get("par_num", [0])[i])
                    line_num = int(data.get("line_num", [0])[i])
                    key = (block_num, par_num, line_num)
                    if key not in line_words:
                        line_words[key] = []
                        line_order.append(key)
                    line_words[key].append(text)
                    confidences.append(conf_int / 100.0)  # Convert to 0-1

            # Calculate average confidence
            avg_confidence = np.mean(confidences) if confidences else 0.0
            line_left_px: Dict[Tuple[int, int, int], int] = {}
            for i, conf in enumerate(data["conf"]):
                try:
                    conf_int = int(float(conf))
                except Exception:
                    continue
                text = data["text"][i].strip()
                if conf_int <= 0 or not text:
                    continue
                block_num = int(data.get("block_num", [0])[i])
                par_num = int(data.get("par_num", [0])[i])
                line_num = int(data.get("line_num", [0])[i])
                key = (block_num, par_num, line_num)
                left_vals = data.get("left", [0])
                try:
                    left = int(float(left_vals[i]))
                except Exception:
                    left = 0
                if key not in line_left_px:
                    line_left_px[key] = left
                else:
                    line_left_px[key] = min(line_left_px[key], left)

            block_lines: Dict[int, List[str]] = {}
            block_left_px: Dict[int, int] = {}
            lines: List[str] = []
            image_width = int(preprocessed.shape[1]) if len(preprocessed.shape) >= 2 else 0
            for key in line_order:
                if key not in line_words or not line_words[key]:
                    continue
                block_num = key[0]
                line_text = " ".join(line_words[key]).strip()
                block_lines.setdefault(block_num, []).append(line_text)
                if key in line_left_px:
                    if block_num not in block_left_px:
                        block_left_px[block_num] = line_left_px[key]
                    else:
                        block_left_px[block_num] = min(block_left_px[block_num], line_left_px[key])

            code_like_blocks = {
                block_num: self._looks_like_code_block_lines(text_lines)
                for block_num, text_lines in block_lines.items()
            }

            for key in line_order:
                if key not in line_words or not line_words[key]:
                    continue
                block_num = key[0]
                line_text = " ".join(line_words[key]).strip()
                prefix = ""
                if (
                    code_like_blocks.get(block_num, False)
                    and key in line_left_px
                    and block_num in block_left_px
                    and image_width > 0
                ):
                    offset_px = max(0.0, float(line_left_px[key] - block_left_px[block_num]))
                    spaces = self._spaces_from_tesseract_offset(offset_px, image_width)
                    if spaces > 0:
                        prefix = " " * spaces
                lines.append(prefix + line_text)

            full_text = "\n".join(lines)
            word_count = sum(len(line_words[key]) for key in line_order if key in line_words)

            elapsed_ms = int((time.perf_counter() - start_time) * 1000)

            logger.debug(
                f"[TESSERACT] Extracted {word_count} words in {elapsed_ms}ms, "
                f"avg confidence: {avg_confidence:.2f}"
            )

            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                layer_used=OCRLayer.TESSERACT,
                word_confidences=confidences,
                processing_time_ms=elapsed_ms,
                word_count=word_count,
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

            # Extract text and confidence from result.
            # Keep per-line structure so downstream code handling can preserve block formatting.
            line_texts: List[str] = []
            confidences = []
            word_count = 0

            for page in result.pages:
                page_blocks, page_word_count = self._collect_doctr_page_blocks(
                    page=page,
                    confidences=confidences,
                )
                word_count += page_word_count
                merged_blocks = self._merge_fragmented_doctr_blocks(page_blocks)
                for block in merged_blocks:
                    block_lines: List[Tuple[str, Optional[float]]] = block.get("lines", [])
                    if not block_lines:
                        continue
                    block_left = min((lx for _, lx in block_lines if lx is not None), default=None)
                    code_like_block = self._looks_like_code_block_lines([lt for lt, _ in block_lines])
                    for line_text, line_left in block_lines:
                        prefix = ""
                        if code_like_block and block_left is not None and line_left is not None:
                            offset = max(0.0, float(line_left - block_left))
                            spaces = self._spaces_from_doctr_offset(offset)
                            if spaces > 0:
                                prefix = " " * spaces
                        line_texts.append(prefix + line_text)

            # Calculate average confidence
            avg_confidence = np.mean(confidences) if confidences else 0.0
            full_text = "\n".join(line_texts)

            elapsed_ms = int((time.perf_counter() - start_time) * 1000)

            logger.debug(
                f"[DOCTR] Extracted {word_count} words in {elapsed_ms}ms, "
                f"avg confidence: {avg_confidence:.2f}"
            )

            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                layer_used=OCRLayer.DOCTR,
                word_confidences=confidences,
                processing_time_ms=elapsed_ms,
                word_count=word_count,
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

    def _looks_like_code_block_lines(self, lines: List[str]) -> bool:
        """
        Heuristic code detector for OCR line groups.
        """
        if not lines:
            return False
        score = 0
        for line in lines:
            t = (line or "").strip()
            if not t:
                continue
            if t.startswith((">>>", "...")):
                score += 3
            if re.search(
                r"\b(def|class|import|from|return|yield|lambda|try|except|finally|with|for|while|if|elif|else|pass)\b",
                t,
            ):
                score += 2
            if re.search(r"[{}()\[\];=:@]", t):
                score += 1
            if t.endswith(":"):
                score += 1
        return score >= max(3, len(lines))

    def _collect_doctr_page_blocks(
        self,
        page: object,
        confidences: List[float],
    ) -> Tuple[List[Dict[str, object]], int]:
        page_blocks: List[Dict[str, object]] = []
        page_word_count = 0

        for block in getattr(page, "blocks", []) or []:
            block_lines: List[Tuple[str, Optional[float]]] = []
            block_bbox = self._extract_bbox_from_geometry(getattr(block, "geometry", None))

            for line in getattr(block, "lines", []) or []:
                words_in_line: List[str] = []
                for word in getattr(line, "words", []) or []:
                    word_text = (getattr(word, "value", "") or "").strip()
                    if not word_text:
                        continue
                    words_in_line.append(word_text)
                    word_conf = getattr(word, "confidence", None)
                    if isinstance(word_conf, (int, float)):
                        confidences.append(float(word_conf))

                if not words_in_line:
                    continue

                line_text = " ".join(words_in_line)
                line_left = self._extract_line_left_x(line)
                block_lines.append((line_text, line_left))
                page_word_count += len(words_in_line)

                line_bbox = self._extract_bbox_from_geometry(getattr(line, "geometry", None))
                if line_bbox is None:
                    for word in getattr(line, "words", []) or []:
                        word_bbox = self._extract_bbox_from_geometry(
                            getattr(word, "geometry", None)
                        )
                        line_bbox = self._update_bbox_union(line_bbox, word_bbox)
                block_bbox = self._update_bbox_union(block_bbox, line_bbox)

            if block_lines:
                page_blocks.append(
                    {
                        "lines": block_lines,
                        "bbox": block_bbox,
                    }
                )

        return page_blocks, page_word_count

    def _merge_fragmented_doctr_blocks(
        self,
        block_records: List[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        if len(block_records) < 2:
            return block_records

        # Missing geometry means we cannot safely merge by spatial heuristics.
        if any(block.get("bbox") is None for block in block_records):
            return block_records

        sorted_blocks = sorted(
            block_records,
            key=lambda block: (block["bbox"][1], block["bbox"][0]),  # type: ignore[index]
        )
        merged: List[Dict[str, object]] = []
        merge_count = 0

        for block in sorted_blocks:
            if not merged:
                merged.append(
                    {
                        "lines": list(block.get("lines", [])),
                        "bbox": block.get("bbox"),
                    }
                )
                continue

            current = merged[-1]
            if self._should_merge_doctr_blocks(current, block):
                current_lines: List[Tuple[str, Optional[float]]] = current.get("lines", [])
                next_lines: List[Tuple[str, Optional[float]]] = block.get("lines", [])
                current_lines.extend(next_lines)
                current["bbox"] = self._update_bbox_union(
                    current.get("bbox"),
                    block.get("bbox"),
                )
                merge_count += 1
            else:
                merged.append(
                    {
                        "lines": list(block.get("lines", [])),
                        "bbox": block.get("bbox"),
                    }
                )

        if merge_count > 0:
            logger.debug(
                "[DOCTR] Merged %d fragmented blocks (%d -> %d)",
                merge_count,
                len(block_records),
                len(merged),
            )

        return merged

    def _should_merge_doctr_blocks(
        self,
        current: Dict[str, object],
        nxt: Dict[str, object],
    ) -> bool:
        c_bbox = current.get("bbox")
        n_bbox = nxt.get("bbox")
        if not isinstance(c_bbox, tuple) or not isinstance(n_bbox, tuple):
            return False
        if len(c_bbox) != 4 or len(n_bbox) != 4:
            return False

        c_x0, c_y0, c_x1, c_y1 = [float(v) for v in c_bbox]
        n_x0, n_y0, n_x1, n_y1 = [float(v) for v in n_bbox]

        # Avoid merging overlapping detections that are likely separate columns/regions.
        if n_y0 < c_y0:
            return False

        c_h = max(c_y1 - c_y0, 1e-6)
        n_h = max(n_y1 - n_y0, 1e-6)
        v_gap = n_y0 - c_y1
        max_v_gap = max(
            self._doctr_merge_max_vertical_gap,
            max(c_h, n_h) * self._doctr_merge_gap_height_multiplier,
        )
        if v_gap < -0.01 or v_gap > max_v_gap:
            return False

        c_w = max(c_x1 - c_x0, 1e-6)
        n_w = max(n_x1 - n_x0, 1e-6)
        overlap = max(0.0, min(c_x1, n_x1) - max(c_x0, n_x0))
        overlap_ratio = overlap / max(min(c_w, n_w), 1e-6)

        c_lines = len(current.get("lines", []))
        n_lines = len(nxt.get("lines", []))
        if c_lines > self._doctr_merge_max_seed_lines and n_lines > self._doctr_merge_max_seed_lines:
            return False

        if overlap_ratio >= self._doctr_merge_min_overlap_ratio:
            return True

        left_shift = n_x0 - c_x0
        if -self._doctr_merge_left_tolerance <= left_shift <= self._doctr_merge_max_indent_shift:
            if n_x0 <= c_x1 + self._doctr_merge_max_indent_shift:
                return True

        return False

    def _spaces_from_doctr_offset(self, offset: float) -> int:
        if offset <= self._doctr_indent_threshold:
            return 0
        levels = int(offset / max(self._doctr_indent_step, 1e-6))
        if levels <= 0:
            return 0
        return min(levels * self._indent_spaces_per_level, self._max_indent_spaces)

    def _spaces_from_tesseract_offset(self, offset_px: float, image_width: int) -> int:
        if image_width <= 0:
            return 0
        threshold = max(image_width * self._tesseract_indent_threshold_ratio, 1.0)
        if offset_px <= threshold:
            return 0
        step = max(image_width * self._tesseract_indent_step_ratio, 1.0)
        levels = int(offset_px / step)
        if levels <= 0:
            return 0
        return min(levels * self._indent_spaces_per_level, self._max_indent_spaces)

    def _extract_line_left_x(self, line: object) -> Optional[float]:
        """
        Extract normalized left-x from Doctr line geometry.
        """
        geometry = getattr(line, "geometry", None)
        left = self._extract_left_from_geometry(geometry)
        if left is not None:
            return left
        # Fallback to first-word geometry when line geometry is unavailable.
        words = getattr(line, "words", None) or []
        for word in words:
            w_left = self._extract_left_from_geometry(getattr(word, "geometry", None))
            if w_left is not None:
                return w_left
        return None

    def _extract_left_from_geometry(self, geometry: object) -> Optional[float]:
        if geometry is None:
            return None
        if not isinstance(geometry, (list, tuple)):
            return None
        xs: List[float] = []
        for point in geometry:
            if (
                isinstance(point, (list, tuple))
                and len(point) >= 2
                and isinstance(point[0], (int, float))
            ):
                xs.append(float(point[0]))
        if not xs:
            return None
        return min(xs)

    def _extract_bbox_from_geometry(
        self,
        geometry: object,
    ) -> Optional[Tuple[float, float, float, float]]:
        if geometry is None:
            return None
        if not isinstance(geometry, (list, tuple)):
            return None

        xs: List[float] = []
        ys: List[float] = []
        for point in geometry:
            if (
                isinstance(point, (list, tuple))
                and len(point) >= 2
                and isinstance(point[0], (int, float))
                and isinstance(point[1], (int, float))
            ):
                xs.append(float(point[0]))
                ys.append(float(point[1]))

        if not xs or not ys:
            return None

        return (min(xs), min(ys), max(xs), max(ys))

    def _update_bbox_union(
        self,
        current_bbox: Optional[Tuple[float, float, float, float]],
        new_bbox: Optional[Tuple[float, float, float, float]],
    ) -> Optional[Tuple[float, float, float, float]]:
        if new_bbox is None:
            return current_bbox
        if current_bbox is None:
            return new_bbox
        return (
            min(current_bbox[0], new_bbox[0]),
            min(current_bbox[1], new_bbox[1]),
            max(current_bbox[2], new_bbox[2]),
            max(current_bbox[3], new_bbox[3]),
        )

    def cleanup(self) -> None:
        """
        Release OCR runtime model references and torch caches.

        Doctr loads transformer weights lazily and keeps them resident; this
        method drops references so long-running jobs can reclaim memory.
        """
        doctr_model = self._doctr_model
        self._doctr_model = None
        self._tesseract_available = None

        if doctr_model is not None:
            try:
                del doctr_model
            except Exception:
                pass

        try:
            import torch  # type: ignore

            try:
                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            try:
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()  # type: ignore[attr-defined]
            except Exception:
                pass
        except Exception:
            pass

        gc.collect()

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
