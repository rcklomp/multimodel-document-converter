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

# Pattern to detect section numbers in headings (e.g., "3.1", "2.", "1.2.3")
SECTION_NUMBER_PATTERN = re.compile(r"^(\d+(?:\.\d+)*)\s*\.?\s+(.+)$")


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

    # Structural filters — language-agnostic patterns that indicate
    # non-heading content (credit lines, copyright notices, TOC lines).
    import re as _re

    # "Role: Name" pattern (credit lines in any language)
    if _re.match(r"^[A-Za-z\s]{3,30}:\s+[A-Z]", text):
        return False

    # Copyright symbol anywhere
    if "©" in text:
        return False

    # ISBN/ISSN pattern
    if _re.search(r"\bISBN\b|\bISSN\b", text, _re.IGNORECASE):
        return False

    # TOC-like lines (many dots as fill characters)
    if text.count(".") > 5 and "...." in text:
        return False

    # Quoted text is dialogue/speech, not a heading (language-agnostic)
    if text.startswith(("'", '"', "\u2018", "\u201c")) and text.endswith(("'", '"', "\u2019", "\u201d", "!'", '?"', "!'")):
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
class HierarchyLevel:
    """Single source of truth for a heading level."""

    text: str
    level: int
    section_number: Optional[List[int]] = None


@dataclass
class ContextStateV2:
    """
    Maintains hierarchical document context during processing.

    REQ-STATE: Tracks the current position in the document hierarchy
    through a breadcrumb path of headings.

    ARCHITECTURE V2 (HARDCORE FIX):
    - Single source of truth: hierarchy_stack (no more sync issues)
    - Aggressive reset: Clear all levels ≥ new level before adding
    - No skipping: Inherently correct logic

    The breadcrumb path is updated when:
    1. A new heading is encountered (appropriate level adjustment)
    2. Page changes (for multi-page documents)

    Attributes:
        doc_id: Document identifier
        source_file: Original source filename
        current_page: Current page number (1-indexed)
        hierarchy_stack: Single list of HierarchyLevel objects (ONE SOURCE OF TRUTH)
        current_header_level: Level of the most recent heading
        _page_first_heading: Track first heading per page for context
    """

    doc_id: str = ""
    source_file: str = ""
    current_page: int = 1
    hierarchy_stack: List[HierarchyLevel] = field(default_factory=list)
    current_header_level: int = 0
    _page_first_heading: Dict[int, str] = field(default_factory=dict)

    # LEGACY COMPATIBILITY: These properties expose the old interface
    @property
    def breadcrumbs(self) -> List[str]:
        """Compatibility property: Extract text from hierarchy_stack."""
        return [h.text for h in self.hierarchy_stack]

    @property
    def heading_levels(self) -> List[int]:
        """Compatibility property: Extract levels from hierarchy_stack."""
        return [h.level for h in self.hierarchy_stack]

    @property
    def section_numbers(self) -> List[List[int]]:
        """Compatibility property: Extract section numbers from hierarchy_stack."""
        return [h.section_number if h.section_number else [] for h in self.hierarchy_stack]

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
        HARDCORE FIX: AGGRESSIVE RESET ARCHITECTURE

        Update the context state when a heading is encountered.

        NEW LOGIC:
        1. Validate heading text
        2. Clamp level to 1-5 (Pydantic constraint)
        3. AGGRESSIVELY POP: Remove ALL headings >= new level
        4. Add new heading to stack
        5. Enforce max depth

        NO MORE:
        - Complex section number logic that causes sync issues
        - Conditional popping that skips cleanup
        - Multiple lists that get out of sync

        Args:
            heading_text: The heading text
            level: Heading level (1-6, clamped to 1-5)
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

        # CRITICAL: Clamp level to 1-5 (Pydantic HierarchyMetadata.level constraint)
        level = max(1, min(5, level))

        # Extract section number (for logging only)
        section_number = self._extract_section_number(heading_text)

        # CASE 1: Empty stack - initialize
        if not self.hierarchy_stack:
            new_level = HierarchyLevel(
                text=heading_text, level=level, section_number=section_number
            )
            self.hierarchy_stack.append(new_level)
            self.current_header_level = level
            logger.debug(f"[HIERARCHY] Initial: '{heading_text}' L{level}")
            return

        # CASE 2: AGGRESSIVE RESET - Remove all headings at same or deeper level
        # This prevents the stack from growing beyond depth limit
        # Example: [H1, H2, H3] + new H2 → [H1, H2-new]
        original_depth = len(self.hierarchy_stack)

        # Find the cutoff point: keep all headings with level < new level
        keep_until = 0
        for i, h in enumerate(self.hierarchy_stack):
            if h.level < level:
                keep_until = i + 1
            else:
                break  # Found first heading at same or deeper level

        # Aggressive reset: keep only shallower headings
        if keep_until < len(self.hierarchy_stack):
            removed = self.hierarchy_stack[keep_until:]
            self.hierarchy_stack = self.hierarchy_stack[:keep_until]
            logger.debug(
                f"[AGGRESSIVE-RESET] Removed {len(removed)} headings at level >= {level}: "
                f"{[h.text[:30] for h in removed]}"
            )

        # CASE 3: Add new heading
        new_level_obj = HierarchyLevel(
            text=heading_text, level=level, section_number=section_number
        )
        self.hierarchy_stack.append(new_level_obj)
        self.current_header_level = level

        # CASE 4: Enforce maximum depth
        if len(self.hierarchy_stack) > MAX_BREADCRUMB_DEPTH:
            excess = len(self.hierarchy_stack) - MAX_BREADCRUMB_DEPTH
            logger.warning(
                f"[MAX-DEPTH] Stack exceeded {MAX_BREADCRUMB_DEPTH}, removing {excess} oldest headings"
            )
            self.hierarchy_stack = self.hierarchy_stack[-MAX_BREADCRUMB_DEPTH:]

        # Log final state
        breadcrumb_str = " > ".join([h.text for h in self.hierarchy_stack])
        logger.debug(
            f"[HIERARCHY] Updated: {breadcrumb_str} "
            f"(depth: {original_depth}→{len(self.hierarchy_stack)}, L{level})"
        )

    def _extract_section_number(self, heading_text: str) -> Optional[List[int]]:
        """
        Extract section number from heading text.

        Examples:
            "3.1 Introduction" → [3, 1]
            "2. Methods" → [2]
            "1.2.3 Details" → [1, 2, 3]
            "Appendix A." → None (no valid numeric section)
            "Introduction" → None

        Args:
            heading_text: The heading text to parse

        Returns:
            List of integers representing section number, or None if no section number
        """
        match = SECTION_NUMBER_PATTERN.match(heading_text.strip())
        if match:
            section_str = match.group(1)
            try:
                # Split by period and convert to integers
                numbers = [int(n) for n in section_str.split(".") if n]
                # CRITICAL: Only return if we have actual numbers
                # This prevents returning [] which could cause index errors later
                if numbers:
                    logger.debug(f"[SEC-NUM] Extracted {numbers} from '{heading_text}'")
                    return numbers
            except ValueError:
                pass
        return None

    def _should_pop_by_section_number(
        self,
        new_section: List[int],
        new_level: int,
    ) -> bool:
        """
        Determine if stack should be popped based on section numbering.

        Logic:
        - Compare section numbers at the same depth
        - If new section's first number is LOWER than previous at same level, POP
        - Example: Moving from "2.5" to "3.1" → DON'T pop (3 > 2)
        - Example: Moving from "3.5" to "2.1" → POP (2 < 3, likely error or new chapter)

        This prevents Chapter 2 hierarchy from persisting into Chapter 3.

        Args:
            new_section: Section number of new heading (e.g., [3, 1])
            new_level: Level of new heading

        Returns:
            True if stack should be popped, False otherwise
        """
        # CRITICAL: Guard against empty lists
        if not self.section_numbers or not new_section or not self.heading_levels:
            return False

        # Find the last heading at the same or shallower level
        for i in range(len(self.heading_levels) - 1, -1, -1):
            # PHANTOM BUG FIX: Ensure index is within bounds of section_numbers
            # section_numbers and heading_levels must be synchronized
            if i >= len(self.section_numbers):
                logger.warning(
                    f"[PHANTOM-BUG-FIX] Index {i} out of range for section_numbers "
                    f"(len={len(self.section_numbers)}). Skipping."
                )
                continue

            prev_level = self.heading_levels[i]
            prev_section = self.section_numbers[i]

            # Only compare if at same level and both have section numbers
            if prev_level == new_level and prev_section:
                # Compare the FIRST number in the section (chapter/major section)
                prev_major = prev_section[0] if prev_section else 0
                new_major = new_section[0] if new_section else 0

                # If new major section is LOWER, we've likely jumped back
                # This shouldn't happen in normal documents, but handle gracefully
                if new_major < prev_major:
                    logger.warning(
                        f"[SEC-POP] Section number decreased: {prev_major} → {new_major}. "
                        f"Popping stack to reset hierarchy."
                    )
                    return True

                # If at deeper nesting level (e.g., comparing 3.2 with 3.1)
                # Check if we need to pop due to subsection reset
                if len(new_section) > 1 and len(prev_section) > 1:
                    # Compare at the same depth
                    depth = min(len(new_section), len(prev_section))
                    for d in range(depth):
                        if new_section[d] < prev_section[d]:
                            logger.info(
                                f"[SEC-POP] Subsection reset detected: "
                                f"{'.'.join(map(str, prev_section))} → "
                                f"{'.'.join(map(str, new_section))}"
                            )
                            return True
                        elif new_section[d] > prev_section[d]:
                            # Normal progression
                            break

                break  # Found comparison point, exit loop

        return False

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
        # Deep copy hierarchy_stack
        copied_stack = [
            HierarchyLevel(
                text=h.text,
                level=h.level,
                section_number=list(h.section_number) if h.section_number else None,
            )
            for h in self.hierarchy_stack
        ]

        return ContextStateV2(
            doc_id=self.doc_id,
            source_file=self.source_file,
            current_page=self.current_page,
            hierarchy_stack=copied_stack,
            current_header_level=self.current_header_level,
            _page_first_heading=dict(self._page_first_heading),
        )

    def reset(self) -> None:
        """Reset state to initial values (keeping doc_id and source_file)."""
        self.current_page = 1
        self.hierarchy_stack = []
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
            "section_numbers": self.section_numbers,  # PHASE 2: Include in serialization
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
        # Reconstruct hierarchy_stack from legacy breadcrumbs/levels/sections
        breadcrumbs = data.get("breadcrumbs", [])
        heading_levels = data.get("heading_levels", [])
        section_numbers = data.get("section_numbers", [])

        hierarchy_stack = []
        for i in range(len(breadcrumbs)):
            text = breadcrumbs[i]
            level = heading_levels[i] if i < len(heading_levels) else 1
            sec_num = section_numbers[i] if i < len(section_numbers) else None
            # Convert empty list to None
            if sec_num is not None and len(sec_num) == 0:
                sec_num = None

            hierarchy_stack.append(HierarchyLevel(text=text, level=level, section_number=sec_num))

        return cls(
            doc_id=data.get("doc_id", ""),
            source_file=data.get("source_file", ""),
            current_page=data.get("current_page", 1),
            hierarchy_stack=hierarchy_stack,
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
            # Use hierarchy_stack directly instead of properties
            initial_level = HierarchyLevel(text=doc_title, level=0, section_number=None)
            state.hierarchy_stack.append(initial_level)
            logger.debug(f"Initialized breadcrumb with document title: '{doc_title}'")

    return state
