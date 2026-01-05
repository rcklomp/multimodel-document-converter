"""
PDF Batch Splitter - Memory-Efficient Large PDF Processing
===========================================================
ENGINE_USE: PyMuPDF (fitz) for fast PDF manipulation

This module provides utilities for splitting large PDFs into manageable
batches for memory-efficient processing on 16GB RAM systems.

REQ Compliance:
- REQ-PDF-05: Memory hygiene through batch splitting
- REQ-PDF-06: Page offset tracking for correct numbering
- REQ-BATCH-01: Configurable batch size (default: 10 pages)

SRS Section 4.3: Batch Processing
"For documents exceeding 50 pages, the system MUST split processing
into batches to maintain memory efficiency."

Author: Claude 4.5 Opus (Architect)
Date: 2025-12-29
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_BATCH_SIZE: int = 10
MIN_BATCH_SIZE: int = 1
MAX_BATCH_SIZE: int = 100


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class BatchInfo:
    """
    Information about a single batch of pages.

    Attributes:
        batch_index: 0-indexed batch number
        batch_path: Path to the extracted batch PDF
        start_page: First page number in original document (1-indexed)
        end_page: Last page number in original document (1-indexed)
        page_count: Number of pages in this batch
        page_offset: Offset to add to batch page numbers for correct absolute page
    """

    batch_index: int
    batch_path: Path
    start_page: int
    end_page: int
    page_count: int
    page_offset: int

    @property
    def page_range_str(self) -> str:
        """Human-readable page range string."""
        return f"{self.start_page}-{self.end_page}"


@dataclass
class SplitResult:
    """
    Result of splitting a PDF into batches.

    Attributes:
        original_path: Path to original PDF
        original_hash: MD5 hash of original document
        total_pages: Total page count in original document
        batch_count: Number of batches created
        batches: List of BatchInfo for each batch
        temp_dir: Temporary directory containing batch files
    """

    original_path: Path
    original_hash: str
    total_pages: int
    batch_count: int
    batches: List[BatchInfo]
    temp_dir: Path


# ============================================================================
# PDF BATCH SPLITTER
# ============================================================================


class PDFBatchSplitter:
    """
    Splits large PDFs into smaller batch files for memory-efficient processing.

    Uses PyMuPDF (fitz) for fast, memory-efficient PDF manipulation.
    The splitter creates temporary batch files that should be cleaned up
    after processing.

    Usage:
        with PDFBatchSplitter(batch_size=10) as splitter:
            result = splitter.split("large_document.pdf")
            for batch in result.batches:
                process_batch(batch.batch_path)
        # Temp files automatically cleaned up

    Attributes:
        batch_size: Number of pages per batch
        temp_dir: Temporary directory for batch files (created on split)
    """

    def __init__(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        specific_pages: Optional[List[int]] = None,
    ) -> None:
        """
        Initialize the PDF batch splitter.

        Args:
            batch_size: Number of pages per batch (default: 10)
            specific_pages: List of specific page numbers to extract (e.g., [6, 21, 169, 241])
        """
        self.batch_size = max(MIN_BATCH_SIZE, min(MAX_BATCH_SIZE, batch_size))
        self.specific_pages = specific_pages
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None

        if specific_pages:
            logger.info(f"PDFBatchSplitter initialized: specific_pages={specific_pages}")
        else:
            logger.info(f"PDFBatchSplitter initialized: batch_size={self.batch_size}")

    def __enter__(self) -> "PDFBatchSplitter":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - cleanup temp files."""
        self.cleanup()

    def cleanup(self) -> None:
        """Clean up temporary files."""
        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
                logger.debug("Cleaned up batch splitter temp directory")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory: {e}")
            self._temp_dir = None

    def get_page_count(self, pdf_path: Path) -> int:
        """
        Get the page count of a PDF without loading it fully.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Number of pages in the PDF
        """
        doc = None
        try:
            doc = fitz.open(str(pdf_path))
            return len(doc)
        finally:
            if doc is not None:
                doc.close()

    def split(self, pdf_path: Path | str) -> SplitResult:
        """
        Split a PDF into batches.

        Args:
            pdf_path: Path to the PDF file to split

        Returns:
            SplitResult containing batch information

        Raises:
            FileNotFoundError: If PDF doesn't exist
            ValueError: If PDF is invalid
        """
        pdf_path = Path(pdf_path).resolve()

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Create temp directory for batches
        self._temp_dir = tempfile.TemporaryDirectory(prefix="mmrag_batch_")
        temp_dir = Path(self._temp_dir.name)

        logger.info(f"Splitting PDF: {pdf_path.name} (temp: {temp_dir})")

        # Open source PDF
        src_doc: Optional[fitz.Document] = None
        batches: List[BatchInfo] = []

        try:
            src_doc = fitz.open(str(pdf_path))
            total_pages = len(src_doc)

            if total_pages == 0:
                raise ValueError(f"PDF has no pages: {pdf_path}")

            # ================================================================
            # SPECIFIC PAGES MODE: Extract only specific pages
            # ================================================================
            if self.specific_pages:
                # Validate and filter specific pages
                valid_pages = [p for p in self.specific_pages if 1 <= p <= total_pages]
                invalid_pages = [p for p in self.specific_pages if p not in valid_pages]

                if invalid_pages:
                    logger.warning(
                        f"Ignoring invalid page numbers: {invalid_pages} "
                        f"(document has {total_pages} pages)"
                    )

                if not valid_pages:
                    raise ValueError(
                        f"No valid pages in specific_pages list. "
                        f"Document has {total_pages} pages, requested: {self.specific_pages}"
                    )

                # Sort pages for consistent processing
                valid_pages = sorted(valid_pages)

                logger.info(
                    f"SPECIFIC PAGES MODE: Extracting pages {valid_pages} "
                    f"from {total_pages}-page document"
                )

                # Create one batch per specific page for correct page offset handling
                for batch_idx, page_num in enumerate(valid_pages):
                    page_0idx = page_num - 1  # Convert to 0-indexed

                    # Create batch PDF with single page
                    batch_filename = f"batch_{batch_idx:03d}_p{page_num}.pdf"
                    batch_path = temp_dir / batch_filename

                    batch_doc = fitz.open()
                    try:
                        batch_doc.insert_pdf(
                            src_doc,
                            from_page=page_0idx,
                            to_page=page_0idx,
                        )
                        batch_doc.save(str(batch_path))
                        logger.debug(f"Created specific page batch: page {page_num}")
                    finally:
                        batch_doc.close()

                    # Page offset = page_num - 1 so that batch page 1 becomes absolute page_num
                    batch_info = BatchInfo(
                        batch_index=batch_idx,
                        batch_path=batch_path,
                        start_page=page_num,
                        end_page=page_num,
                        page_count=1,
                        page_offset=page_0idx,
                    )
                    batches.append(batch_info)

                batch_count = len(batches)
                pages_to_process = len(valid_pages)

            # ================================================================
            # NORMAL MODE: Split into fixed-size batches
            # ================================================================
            else:
                # Calculate batch count
                batch_count = (total_pages + self.batch_size - 1) // self.batch_size

                logger.info(
                    f"PDF has {total_pages} pages, "
                    f"splitting into {batch_count} batches of {self.batch_size} pages"
                )

                # Create batches
                for batch_idx in range(batch_count):
                    # Calculate page range (0-indexed for PyMuPDF)
                    start_page_0idx = batch_idx * self.batch_size
                    end_page_0idx = min(start_page_0idx + self.batch_size, total_pages) - 1

                    # Convert to 1-indexed for our tracking
                    start_page_1idx = start_page_0idx + 1
                    end_page_1idx = end_page_0idx + 1
                    page_count = end_page_0idx - start_page_0idx + 1

                    # Page offset for correct absolute numbering
                    # If batch starts at page 11 (start_page_1idx=11), offset = 10
                    # So batch page 1 + offset = absolute page 11
                    page_offset = start_page_0idx

                    # Create batch PDF
                    batch_filename = f"batch_{batch_idx:03d}_p{start_page_1idx}-{end_page_1idx}.pdf"
                    batch_path = temp_dir / batch_filename

                    # Extract pages to new PDF
                    batch_doc = fitz.open()
                    try:
                        batch_doc.insert_pdf(
                            src_doc,
                            from_page=start_page_0idx,
                            to_page=end_page_0idx,
                        )
                        batch_doc.save(str(batch_path))
                        logger.debug(
                            f"Created batch {batch_idx}: pages {start_page_1idx}-{end_page_1idx}"
                        )
                    finally:
                        batch_doc.close()

                    # Create batch info
                    batch_info = BatchInfo(
                        batch_index=batch_idx,
                        batch_path=batch_path,
                        start_page=start_page_1idx,
                        end_page=end_page_1idx,
                        page_count=page_count,
                        page_offset=page_offset,
                    )
                    batches.append(batch_info)

                pages_to_process = total_pages

            # Compute document hash for identification
            import hashlib

            hasher = hashlib.md5()
            with open(pdf_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            doc_hash = hasher.hexdigest()[:12]

            return SplitResult(
                original_path=pdf_path,
                original_hash=doc_hash,
                total_pages=pages_to_process,
                batch_count=batch_count,
                batches=batches,
                temp_dir=temp_dir,
            )

        finally:
            if src_doc is not None:
                src_doc.close()

    def iter_batches(
        self,
        pdf_path: Path | str,
    ) -> Generator[BatchInfo, None, None]:
        """
        Iterate over batches without loading all at once.

        This is a convenience method that splits and yields batches
        one at a time.

        Args:
            pdf_path: Path to PDF file

        Yields:
            BatchInfo for each batch
        """
        result = self.split(pdf_path)
        for batch in result.batches:
            yield batch


@contextmanager
def batch_split_pdf(
    pdf_path: Path | str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Generator[SplitResult, None, None]:
    """
    Context manager for splitting a PDF into batches.

    Automatically cleans up temporary files when done.

    Args:
        pdf_path: Path to PDF file
        batch_size: Pages per batch

    Yields:
        SplitResult with batch information

    Example:
        with batch_split_pdf("large.pdf", batch_size=10) as result:
            for batch in result.batches:
                process_batch(batch.batch_path)
    """
    splitter = PDFBatchSplitter(batch_size=batch_size)
    try:
        yield splitter.split(pdf_path)
    finally:
        splitter.cleanup()
