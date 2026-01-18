"""
Quality Filter Tracker - Token-Level Filtering Analytics
=========================================================

This module implements a utility class for tracking and categorizing
filtered tokens during quality filtering. It provides detailed analytics
on why chunks were skipped and how many tokens were filtered.

KEY DESIGN:
- Uses efficient token estimation (characters / 4) to avoid tiktoken overhead
- Tracks filtered tokens by category: too_short, noise_pattern, page_number,
  tiny_bbox, decoration, and other
- Provides per-document summaries for validation and observability

INTEGRATION:
- BatchProcessor._should_skip_chunk() returns (should_skip, reason)
- BatchProcessor calls tracker.track_filtered_chunk(chunk, reason)
- TokenValidator uses tracker summary to adjust validation logic

SRS Compliance:
- REQ-QUALITY-01: Confidence scores must be comparable across formats
- QA-CHECK-01: Token balance verification with filtering awareness

Author: Cline (Senior Architect)
Date: 2026-01-17
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, TYPE_CHECKING, Set

if TYPE_CHECKING:
    from ..schema.ingestion_schema import IngestionChunk

logger = logging.getLogger(__name__)


class FilterCategory(Enum):
    """Categories for filtered chunks."""

    TOO_SHORT = "too_short"  # Below minimum character threshold for profile
    NOISE_PATTERN = "noise_pattern"  # Academic noise (page numbers, Roman numerals, copyright)
    PAGE_NUMBER = "page_number"  # Pure numeric page indicators
    TINY_BBOX = "tiny_bbox"  # Very small bounding box with short text
    DECORATION = "decoration"  # Pure decorative elements (dashes, bullets, etc.)
    EMPTY = "empty"  # Empty or whitespace-only content
    OTHER = "other"  # Other reasons


@dataclass
class FilteredChunkRecord:
    """Record of a filtered chunk."""

    chunk_id: str
    page_number: int
    category: FilterCategory
    estimated_tokens: int
    content_preview: str  # First 50 chars for debugging


@dataclass
class QualityFilterSummary:
    """Summary of filtered tokens for a document."""

    total_filtered_tokens: int = 0
    total_filtered_chunks: int = 0
    tokens_by_category: Dict[FilterCategory, int] = field(
        default_factory=lambda: {cat: 0 for cat in FilterCategory}
    )
    chunks_by_category: Dict[FilterCategory, int] = field(
        default_factory=lambda: {cat: 0 for cat in FilterCategory}
    )
    filtered_chunks: List[FilteredChunkRecord] = field(default_factory=list)

    def add_filtered_chunk(self, record: FilteredChunkRecord) -> None:
        """Add a filtered chunk record to the summary."""
        self.total_filtered_chunks += 1
        self.total_filtered_tokens += record.estimated_tokens

        self.chunks_by_category[record.category] = (
            self.chunks_by_category.get(record.category, 0) + 1
        )
        self.tokens_by_category[record.category] = (
            self.tokens_by_category.get(record.category, 0) + record.estimated_tokens
        )

        self.filtered_chunks.append(record)

    def get_filtered_ratio(self, total_source_tokens: int) -> float:
        """Calculate the ratio of filtered tokens to total source tokens."""
        if total_source_tokens == 0:
            return 0.0
        return self.total_filtered_tokens / total_source_tokens

    def to_dict(self) -> Dict:
        """Convert summary to dictionary for logging."""
        return {
            "total_filtered_tokens": self.total_filtered_tokens,
            "total_filtered_chunks": self.total_filtered_chunks,
            "tokens_by_category": {
                cat.value: count for cat, count in self.tokens_by_category.items() if count > 0
            },
            "chunks_by_category": {
                cat.value: count for cat, count in self.chunks_by_category.items() if count > 0
            },
            "filtered_ratio": self.get_filtered_ratio(
                1
            ),  # Will be calculated with actual source tokens later
        }


class QualityFilterTracker:
    """
    Tracks and categorizes filtered chunks with efficient token estimation.

    This class is designed to be lightweight and efficient, using character-based
    token estimation (chars / 4) to avoid the overhead of tiktoken calls for
    every skipped chunk.

    Usage:
        tracker = QualityFilterTracker()

        # When a chunk is filtered
        should_skip, reason = processor._should_skip_chunk(chunk)
        if should_skip:
            tracker.track_filtered_chunk(chunk, reason)

        # Get summary
        summary = tracker.get_summary()
    """

    # Token estimation factor (conservative: ~4 chars per token)
    CHARS_PER_TOKEN_ESTIMATE = 4.0

    def __init__(self) -> None:
        """Initialize a new QualityFilterTracker."""
        self._summary = QualityFilterSummary()
        self._reset_after_summary = False
        self._tracked_chunk_ids: Set[str] = set()

    def track_filtered_chunk(
        self,
        chunk: "IngestionChunk",
        category: FilterCategory,
        content: Optional[str] = None,
    ) -> None:
        """
        Track a filtered chunk.

        Args:
            chunk: The filtered IngestionChunk
            category: Why the chunk was filtered
            content: Optional explicit content (if chunk.content is not reliable)
        """
        # Use provided content or chunk content
        chunk_content = content if content is not None else (chunk.content or "")

        # Estimate tokens efficiently (chars / 4)
        estimated_tokens = self._estimate_tokens(chunk_content)

        # Create record
        record = FilteredChunkRecord(
            chunk_id=chunk.chunk_id,
            page_number=chunk.metadata.page_number if chunk.metadata else 0,
            category=category,
            estimated_tokens=estimated_tokens,
            content_preview=chunk_content[:50].replace("\n", " "),
        )

        # Add to summary
        self._summary.add_filtered_chunk(record)

        # Log at debug level
        logger.debug(
            f"[QUALITY-FILTER] Filtered chunk {chunk.chunk_id}: "
            f"category={category.value}, "
            f"tokens≈{estimated_tokens}, "
            f"content='{record.content_preview}...'"
        )

    def track_filtered_content(
        self,
        content: str,
        page_number: int,
        category: FilterCategory,
        chunk_id: str = "unknown",
    ) -> None:
        """
        Track filtered content when no chunk object is available.

        This is useful for content filtered before chunk creation.

        Args:
            content: The filtered text content
            page_number: Page number where content appeared
            category: Why the content was filtered
            chunk_id: Optional chunk identifier
        """
        # Estimate tokens efficiently
        estimated_tokens = self._estimate_tokens(content)

        # Create record
        record = FilteredChunkRecord(
            chunk_id=chunk_id,
            page_number=page_number,
            category=category,
            estimated_tokens=estimated_tokens,
            content_preview=content[:50].replace("\n", " "),
        )

        # Add to summary
        self._summary.add_filtered_chunk(record)

        logger.debug(
            f"[QUALITY-FILTER] Filtered content on page {page_number}: "
            f"category={category.value}, tokens≈{estimated_tokens}"
        )

    def get_summary(self, reset: bool = False) -> QualityFilterSummary:
        """
        Get the current filtering summary.

        Args:
            reset: If True, reset the tracker after returning summary

        Returns:
            QualityFilterSummary with all filtered chunks
        """
        summary = self._summary

        if reset:
            self.reset()

        return summary

    def reset(self) -> None:
        """Reset the tracker for a new document."""
        self._summary = QualityFilterSummary()

    def log_summary(self, doc_name: str, total_source_tokens: int) -> None:
        """
        Log a comprehensive summary of filtering for a document.

        Args:
            doc_name: Document name for context
            total_source_tokens: Total tokens in source text (for ratio calculation)
        """
        if self._summary.total_filtered_chunks == 0:
            logger.info(f"[QUALITY-FILTER] No chunks filtered for {doc_name}")
            return

        # Calculate ratios
        filtered_ratio = self._summary.get_filtered_ratio(total_source_tokens)

        # Build category breakdown
        category_breakdown = []
        for cat in FilterCategory:
            tokens = self._summary.tokens_by_category.get(cat, 0)
            if tokens > 0:
                percent = (tokens / self._summary.total_filtered_tokens) * 100
                category_breakdown.append(f"{cat.value}: {tokens} tokens ({percent:.1f}%)")

        # Log summary
        logger.info(
            f"[QUALITY-FILTER] Summary for {doc_name}:\n"
            f"  Total filtered: {self._summary.total_filtered_tokens} tokens "
            f"({self._summary.total_filtered_chunks} chunks)\n"
            f"  Filtered ratio: {filtered_ratio:.1%} of source tokens\n"
            f"  Categories: {', '.join(category_breakdown)}"
        )

        # Log detailed breakdown if debug enabled
        if logger.isEnabledFor(logging.DEBUG):
            for record in self._summary.filtered_chunks[:10]:  # First 10 for brevity
                logger.debug(
                    f"    - {record.chunk_id} (page {record.page_number}): "
                    f"{record.category.value}, {record.estimated_tokens} tokens, "
                    f"'{record.content_preview}...'"
                )
            if len(self._summary.filtered_chunks) > 10:
                logger.debug(f"    ... and {len(self._summary.filtered_chunks) - 10} more")

    def _estimate_tokens(self, text: str) -> int:
        """
        Efficiently estimate token count using character-based heuristic.

        Uses conservative estimate of 4 characters per token (cl100k_base average).
        This avoids the overhead of tiktoken calls for every filtered chunk.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count (rounded up)
        """
        if not text:
            return 0

        # Remove whitespace for better estimation
        clean_text = text.strip()
        if not clean_text:
            return 0

        # Conservative estimate: 4 chars per token (round up)
        return max(1, int(len(clean_text) / self.CHARS_PER_TOKEN_ESTIMATE + 0.5))


def create_quality_filter_tracker() -> QualityFilterTracker:
    """
    Factory function to create a QualityFilterTracker.

    Returns:
        New QualityFilterTracker instance
    """
    return QualityFilterTracker()
