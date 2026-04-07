"""
CorruptionInterceptor — Per-bbox OCR patching for encoding-corrupted chunks.

When Docling extracts text from PDFs with broken character mappings (CIDFont,
non-standard ToUnicode), the text contains /C211 or /uniFB01 placeholders.
Instead of discarding the entire HybridChunker output (losing structure) or
hoping the refiner fixes it (it often can't), this interceptor:

1. Detects chunks with encoding artifacts
2. Renders the chunk's bbox region from the PDF at high DPI
3. Runs OCR on just that region
4. Replaces ONLY the corrupted text, keeping all metadata intact

This is "Selective OCR Patching" — heal-over, not fail-over.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..schema.ingestion_schema import IngestionChunk

logger = logging.getLogger(__name__)

# Patterns that indicate encoding corruption in extracted text
CORRUPTION_PATTERNS = re.compile(
    r"/C\d{1,3}"           # /C211, /C1 — CIDFont glyph placeholders
    r"|/uni[A-F0-9]{4,}"   # /uniFB01 — Unicode escape leaks
    r"|\\x[0-9a-f]{2}"     # \xc3 — hex escape leaks
    r"|\ufffd"             # Unicode replacement character
)


def has_encoding_artifacts(text: str) -> bool:
    """Check if text contains encoding corruption artifacts."""
    return bool(CORRUPTION_PATTERNS.search(text))


def count_encoding_artifacts(text: str) -> int:
    """Count encoding corruption artifacts in text."""
    return len(CORRUPTION_PATTERNS.findall(text))


def patch_corrupted_chunks(
    chunks: List["IngestionChunk"],
    pdf_path: Optional[Path] = None,
) -> List["IngestionChunk"]:
    """Patch encoding-corrupted chunks using per-bbox OCR.

    For each chunk with encoding artifacts:
    1. Get its bbox and page number
    2. Render that region from the PDF
    3. OCR the region
    4. Replace the corrupted content (keep metadata)

    Args:
        chunks: List of IngestionChunk objects
        pdf_path: Path to the source PDF for rendering

    Returns:
        Same list with corrupted chunks patched in-place
    """
    if not pdf_path or not pdf_path.exists():
        logger.warning("[CORRUPTION-INTERCEPTOR] No PDF path — skipping OCR patching")
        return chunks

    try:
        import fitz
        import pytesseract
        from PIL import Image
        import numpy as np
    except ImportError as e:
        logger.warning(f"[CORRUPTION-INTERCEPTOR] Missing dependency: {e}")
        return chunks

    from ..schema.ingestion_schema import Modality

    # Find corrupted chunks
    corrupted = []
    for i, ch in enumerate(chunks):
        if ch.modality != Modality.TEXT or not ch.content:
            continue
        if has_encoding_artifacts(ch.content):
            corrupted.append(i)

    if not corrupted:
        return chunks

    logger.info(f"[CORRUPTION-INTERCEPTOR] Found {len(corrupted)} chunks with encoding artifacts")

    # Open PDF once for all patches
    doc = None
    try:
        doc = fitz.open(pdf_path)
        patched = 0

        for idx in corrupted:
            ch = chunks[idx]
            page_no = ch.metadata.page_number if ch.metadata else None
            bbox = ch.metadata.spatial.bbox if ch.metadata and ch.metadata.spatial else None

            if not page_no or not bbox or page_no > len(doc):
                continue

            try:
                page = doc.load_page(page_no - 1)  # 0-indexed
                pw, ph = page.rect.width, page.rect.height

                # Convert [0,1000] normalized bbox back to PDF points
                x0 = bbox[0] / 1000 * pw
                y0 = bbox[1] / 1000 * ph
                x1 = bbox[2] / 1000 * pw
                y1 = bbox[3] / 1000 * ph

                # Add padding (10% each side)
                pad_x = (x1 - x0) * 0.1
                pad_y = (y1 - y0) * 0.1
                clip = fitz.Rect(
                    max(0, x0 - pad_x),
                    max(0, y0 - pad_y),
                    min(pw, x1 + pad_x),
                    min(ph, y1 + pad_y),
                )

                # Render at 300 DPI
                mat = fitz.Matrix(300 / 72, 300 / 72)
                pix = page.get_pixmap(matrix=mat, clip=clip)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

                # OCR the region
                ocr_text = pytesseract.image_to_string(img).strip()

                if ocr_text and len(ocr_text) > 10:
                    # Verify OCR is cleaner (fewer artifacts)
                    old_artifacts = count_encoding_artifacts(ch.content)
                    new_artifacts = count_encoding_artifacts(ocr_text)

                    if new_artifacts < old_artifacts:
                        ch.content = ocr_text
                        if ch.metadata and ch.metadata.refined_content:
                            ch.metadata.refined_content = ocr_text
                        patched += 1
                        logger.debug(
                            f"[CORRUPTION-INTERCEPTOR] Patched pg {page_no}: "
                            f"{old_artifacts}→{new_artifacts} artifacts"
                        )

            except Exception as e:
                logger.debug(f"[CORRUPTION-INTERCEPTOR] Failed pg {page_no}: {e}")

        if patched:
            logger.info(f"[CORRUPTION-INTERCEPTOR] Patched {patched}/{len(corrupted)} corrupted chunks")

    finally:
        if doc:
            doc.close()

    return chunks
