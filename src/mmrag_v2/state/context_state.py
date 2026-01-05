"""
Context State V2 - Hierarchical Breadcrumb Tracking for Document Processing
============================================================================
ENGINE_USE: Docling v2.66.0 (Native Layout Analysis)

This module implements the ContextStateV2 class for maintaining document
structure context during processing. It tracks the current position in the
document hierarchy through a breadcrumb path.

REQ Compliance:
- REQ-STATE: Hierarchical breadcrumb tracking throughout processing
- REQ-STATE-01: Breadcrumb path reflects current heading hierarchy
- REQ-STATE-02: State resets appropriately when entering new sections
- REQ-STATE-03: State can be serialized for batch processing continuity

SRS Section 5: Document Structure Tracking
"The system MUST maintain a hierarchical context state that tracks the
current position within the document's logical structure."

Author: Claude 4.5 Opus (Architect)
Date: 2025-12-28
Updated: 2025-12-29 (Batch Processing Continuity)
"""

from __future__ import annotations

import logging
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Maximum breadcrumb depth to prevent runaway hierarchies
MAX_BREADCRUMB_DEPTH: int = 10

# Minimum heading length to be considered valid
MIN_HEADING_LENGTH: int = 2

# Maximum heading length (truncate if longer)
MAX_HEADING_LENGTH: int = 100

# Pattern to detect noise/invalid headings
NOISE_PATTERN = re.compile(r"^[\s\-·•\.\d\*]+$")

# Pattern to detect page numbers or artifacts
PAGE_NUMBER_PATTERN = re.compile(r"^\d+$|^page\s*\d+$", re.IGNORECASE)


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================


def is_valid_heading(text: str) -> bool:
    """
    Validate if text is a legitimate heading.

    Filters out:
    - Empty or whitespace-only strings
    - Pure punctuation/symbols
    - Page numbers
    - Very short strings (< 2 chars)
    - Noise patterns (bullets, dashes only)

    Args:
        text: The heading text to validate

    Returns:
        True if text is a valid heading, False otherwise
    """
    if not text or not text.strip():
        return False

    text = text.strip()

    # Too short
    if len(text) < MIN_HEADING_LENGTH:
        return False

    # Noise pattern (only symbols, bullets, etc.)
    if NOISE_PATTERN.match(text):
        return False

    # Page number pattern
    if PAGE_NUMBER_PATTERN.match(text):
        return False

    # Must contain at least one letter
    if not any(c.isalpha() for c in text):
        return False

    return True


def _truncate_heading(text: str) -> str:
    """Truncate heading to maximum length."""
    if len(text) > MAX_HEADING_LENGTH:
        return text[: MAX_HEADING_LENGTH - 3] + "..."
    return text


# ============================================================================
# CONTEXT STATE V2
# ============================================================================


@dataclass
class ContextStateV2:
    """
    Maintains hierarchical document context during processing.

    REQ-STATE: Tracks the current position in the document hierarchy
    through a breadcrumb path of headings.

    The breadcrumb path is updated when:
    1. A new heading is encountered (appropriate level adjustment)
    2. Page changes (for multi-page documents)

    Attributes:
        doc_id: Document identifier
        source_file: Original source filename
        current_page: Current page number (1-indexed)
        breadcrumbs: List of heading texts forming the breadcrumb path
        heading_levels: Corresponding heading levels for breadcrumbs
        current_header_level: Level of the most recent heading
        _page_first_heading: Track first heading per page for context
    """

    doc_id: str = ""
    source_file: str = ""
    current_page: int = 1
    breadcrumbs: List[str] = field(default_factory=list)
    heading_levels: List[int] = field(default_factory=list)
    current_header_level: int = 0
    _page_first_heading: Dict[int, str] = field(default_factory=dict)

    def update_page(self, page_number: int) -> None:
        """
        Update the current page number.

        Args:
            page_number: New page number (1-indexed)
        """
        if page_number != self.current_page:
            logger.debug(f"Page transition: {self.current_page} → {page_number}")
            self.current_page = page_number

    def update_on_heading(self, heading_text: str, level: int) -> None:
        """
        Update the context state when a heading is encountered.

        REQ-STATE-01: The breadcrumb path is adjusted based on heading level:
        - Same or lower level: Pop back to parent, then add new heading
        - Higher level (nested): Add as child of current heading

        Args:
            heading_text: The heading text
            level: Heading level (1-6, where 1 is highest)
        """
        # Validate heading
        if not is_valid_heading(heading_text):
            logger.debug(f"Rejected invalid heading: '{heading_text}'")
            return

        # Truncate if too long
        heading_text = _truncate_heading(heading_text.strip())

        # Track first heading per page
        if self.current_page not in self._page_first_heading:
            self._page_first_heading[self.current_page] = heading_text

        # Level must be positive
        level = max(1, min(6, level))

        # CASE 1: Empty breadcrumbs - just add
        if not self.breadcrumbs:
            self.breadcrumbs.append(heading_text)
            self.heading_levels.append(level)
            self.current_header_level = level
            logger.debug(f"Initial heading: '{heading_text}' (L{level})")
            return

        # CASE 2: Same or lower level - pop back to appropriate depth
        # e.g., if current is H2 and we see H2 or H1, pop back
        while self.heading_levels and self.heading_levels[-1] >= level:
            popped = self.breadcrumbs.pop()
            self.heading_levels.pop()
            logger.debug(f"Popped heading: '{popped}'")

        # CASE 3: Add new heading
        self.breadcrumbs.append(heading_text)
        self.heading_levels.append(level)
        self.current_header_level = level

        # Enforce maximum depth
        while len(self.breadcrumbs) > MAX_BREADCRUMB_DEPTH:
            self.breadcrumbs.pop(0)
            self.heading_levels.pop(0)

        logger.debug(f"Updated breadcrumbs: {' > '.join(self.breadcrumbs)} (L{level})")

    def get_breadcrumb_path(self) -> List[str]:
        """
        Get the current breadcrumb path.

        Returns:
            List of heading texts from root to current position
        """
        return list(self.breadcrumbs)

    def get_parent_heading(self) -> Optional[str]:
        """
        Get the immediate parent heading (last in breadcrumb).

        Returns:
            Parent heading text, or None if no breadcrumbs
        """
        if self.breadcrumbs:
            return self.breadcrumbs[-1]
        return None

    def get_breadcrumb_string(self, separator: str = " > ") -> str:
        """
        Get breadcrumb path as a formatted string.

        Args:
            separator: String to join breadcrumbs

        Returns:
            Formatted breadcrumb string
        """
        return separator.join(self.breadcrumbs)

    def get_state_copy(self) -> "ContextStateV2":
        """
        Create a deep copy of the current state.

        REQ-STATE-03: Used for batch processing to preserve state
        across batch boundaries.

        Returns:
            Deep copy of this ContextStateV2
        """
        return ContextStateV2(
            doc_id=self.doc_id,
            source_file=self.source_file,
            current_page=self.current_page,
            breadcrumbs=list(self.breadcrumbs),
            heading_levels=list(self.heading_levels),
            current_header_level=self.current_header_level,
            _page_first_heading=dict(self._page_first_heading),
        )

    def reset(self) -> None:
        """Reset state to initial values (keeping doc_id and source_file)."""
        self.current_page = 1
        self.breadcrumbs = []
        self.heading_levels = []
        self.current_header_level = 0
        self._page_first_heading = {}

    def to_dict(self) -> Dict:
        """
        Serialize state to dictionary.

        Returns:
            Dictionary representation of state
        """
        return {
            "doc_id": self.doc_id,
            "source_file": self.source_file,
            "current_page": self.current_page,
            "breadcrumbs": self.breadcrumbs,
            "heading_levels": self.heading_levels,
            "current_header_level": self.current_header_level,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ContextStateV2":
        """
        Deserialize state from dictionary.

        Args:
            data: Dictionary containing state data

        Returns:
            ContextStateV2 instance
        """
        return cls(
            doc_id=data.get("doc_id", ""),
            source_file=data.get("source_file", ""),
            current_page=data.get("current_page", 1),
            breadcrumbs=data.get("breadcrumbs", []),
            heading_levels=data.get("heading_levels", []),
            current_header_level=data.get("current_header_level", 0),
        )


# ============================================================================
# FACTORY FUNCTION
# ============================================================================


def create_context_state(
    doc_id: str = "",
    source_file: str = "",
    initialize_breadcrumb: bool = True,
) -> ContextStateV2:
    """
    Factory function to create a new ContextStateV2.

    REQ-STATE: Initialize breadcrumbs with source_file to ensure context
    is never empty for downstream RAG systems.

    Args:
        doc_id: Document identifier
        source_file: Original source filename
        initialize_breadcrumb: If True, add source_file as initial breadcrumb

    Returns:
        Initialized ContextStateV2 instance with proper breadcrumb context
    """
    state = ContextStateV2(
        doc_id=doc_id,
        source_file=source_file,
    )

    # REQ-STATE: Initialize breadcrumb with document title (source_file without extension)
    if initialize_breadcrumb and source_file:
        # Extract document title from filename (remove path and extension)
        from pathlib import Path

        doc_title = Path(source_file).stem
        # Clean up common patterns: replace underscores/hyphens with spaces, title case
        doc_title = doc_title.replace("_", " ").replace("-", " ")
        # Remove multiple spaces
        doc_title = " ".join(doc_title.split())
        if doc_title:
            state.breadcrumbs.append(doc_title)
            state.heading_levels.append(0)  # Level 0 = document root
            logger.debug(f"Initialized breadcrumb with document title: '{doc_title}'")

    return state
