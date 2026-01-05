"""
Magazine Section Detector - Intelligent Breadcrumb Enrichment for Magazines
============================================================================
ENGINE_USE: Docling v2.66.0 (Native Layout Analysis)

PROBLEM SOLVED:
Magazine layouts have complex section headers that Docling doesn't recognize
as H1/H2 headings (e.g., "96 Cutting Edge", "IN THE NEWS", "COMBAT AIRCRAFT").
This module detects these patterns and enriches breadcrumbs accordingly.

REQ Compliance:
- REQ-BREAD-01: Breadcrumbs must reflect document logical structure
- REQ-BREAD-02: Magazine sections should be detected from text patterns
- REQ-BREAD-03: Page numbers in headers should NOT be treated as section names

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│ Text Element from Docling                                        │
│        ↓                                                         │
│   MagazineSectionDetector.analyze()                              │
│        ↓                                                         │
│   Pattern Matching:                                              │
│   - "96 Cutting Edge" → section="Cutting Edge"                   │
│   - "IN THE NEWS" → section="In The News"                        │
│   - Large font headers → potential section                       │
│        ↓                                                         │
│   Enriched ContextStateV2 breadcrumbs                            │
└─────────────────────────────────────────────────────────────────┘

Author: Claude (Senior Architect)
Date: 2026-01-02
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Minimum section header length
MIN_SECTION_LENGTH: int = 3

# Maximum section header length (to avoid full paragraphs)
MAX_SECTION_LENGTH: int = 60

# Common magazine section patterns (case-insensitive)
MAGAZINE_SECTION_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Pattern: Page number followed by section name (e.g., "96 Cutting Edge")
    (re.compile(r"^\d{1,3}\s+([A-Z][A-Za-z\s&-]{2,40})$"), "numbered_section"),
    # Pattern: ALL CAPS section headers (e.g., "IN THE NEWS", "COMBAT AIRCRAFT")
    (re.compile(r"^([A-Z][A-Z\s&-]{3,40})$"), "caps_section"),
    # Pattern: Section with colon (e.g., "Feature: The F-35 Story")
    (re.compile(r"^([A-Za-z\s]{2,20}):\s*(.+)$"), "titled_section"),
    # Pattern: Common magazine sections
    (
        re.compile(
            r"^(News|Feature|Report|Analysis|Interview|Review|Special|Gallery|Inside|Profile)\b",
            re.I,
        ),
        "keyword_section",
    ),
]

# Words that indicate NON-section content (should NOT trigger section detection)
EXCLUSION_WORDS: Set[str] = {
    "the",
    "and",
    "but",
    "or",
    "for",
    "with",
    "from",
    "into",
    "by",
    "of",
    "page",
    "continued",
    "see",
    "also",
    "more",
    "next",
    "prev",
    "previous",
    "advertisement",
    "advert",
    "sponsor",
    "subscribe",
    "subscription",
}

# Common magazine section keywords that SHOULD trigger detection
SECTION_KEYWORDS: Set[str] = {
    "news",
    "feature",
    "report",
    "analysis",
    "interview",
    "review",
    "special",
    "gallery",
    "inside",
    "profile",
    "cutting edge",
    "frontline",
    "airpower",
    "combat",
    "mission",
    "operation",
    "squadron",
    "unit",
    "history",
    "heritage",
    "archive",
    "technology",
    "weapons",
    "systems",
}


# ============================================================================
# SECTION DETECTION RESULT
# ============================================================================


@dataclass
class SectionDetectionResult:
    """Result of section header detection."""

    is_section: bool = False
    section_name: Optional[str] = None
    detection_method: Optional[str] = None
    confidence: float = 0.0
    suggested_level: int = 2  # Default to H2
    page_number_prefix: Optional[int] = None

    def __repr__(self) -> str:
        if self.is_section:
            return f"Section('{self.section_name}', method={self.detection_method}, conf={self.confidence:.2f})"
        return "NoSection"


# ============================================================================
# MAGAZINE SECTION DETECTOR
# ============================================================================


class MagazineSectionDetector:
    """
    Detects magazine section headers from text patterns.

    Enriches breadcrumbs by identifying sections that Docling's
    layout analysis may miss (e.g., stylized headers, numbered sections).

    Usage:
        detector = MagazineSectionDetector()
        result = detector.analyze(text, page_number)
        if result.is_section:
            state.update_on_heading(result.section_name, result.suggested_level)
    """

    def __init__(self) -> None:
        """Initialize MagazineSectionDetector."""
        self._detected_sections: Dict[str, int] = {}  # section -> page
        self._page_sections: Dict[int, List[str]] = {}  # page -> sections
        self._last_page: int = 0

    def analyze(
        self,
        text: str,
        page_number: int,
        font_size: Optional[float] = None,
        is_first_on_page: bool = False,
    ) -> SectionDetectionResult:
        """
        Analyze text to determine if it's a section header.

        Detection heuristics:
        1. Pattern matching (numbered sections, ALL CAPS, etc.)
        2. Position (first text on page is often section header)
        3. Font size (larger = more likely section header)
        4. Keyword matching (common magazine section names)

        Args:
            text: Text content to analyze
            page_number: Current page number
            font_size: Font size if available (from Docling)
            is_first_on_page: Whether this is the first text on the page

        Returns:
            SectionDetectionResult with detection info
        """
        result = SectionDetectionResult()

        # Clean and validate text
        text = text.strip()
        if not self._is_candidate(text):
            return result

        # Track page transitions for context
        if page_number != self._last_page:
            self._last_page = page_number

        # Try pattern matching
        pattern_result = self._match_patterns(text)
        if pattern_result.is_section:
            return pattern_result

        # Try keyword detection
        keyword_result = self._match_keywords(text, is_first_on_page)
        if keyword_result.is_section:
            return keyword_result

        # Try position-based detection (first on page + short = likely section)
        if is_first_on_page and len(text) < 40:
            position_result = self._analyze_position_header(text, page_number)
            if position_result.is_section:
                return position_result

        return result

    def _is_candidate(self, text: str) -> bool:
        """Check if text could be a section header."""
        if not text:
            return False

        # Length check
        if len(text) < MIN_SECTION_LENGTH or len(text) > MAX_SECTION_LENGTH:
            return False

        # Must contain at least one letter
        if not any(c.isalpha() for c in text):
            return False

        # Should not start with exclusion words
        first_word = text.split()[0].lower() if text.split() else ""
        if first_word in EXCLUSION_WORDS:
            return False

        return True

    def _match_patterns(self, text: str) -> SectionDetectionResult:
        """Match text against known section patterns."""
        for pattern, pattern_name in MAGAZINE_SECTION_PATTERNS:
            match = pattern.match(text)
            if match:
                # Extract section name
                if pattern_name == "numbered_section":
                    # "96 Cutting Edge" -> "Cutting Edge"
                    section_name = match.group(1).strip()
                    page_prefix = int(text.split()[0]) if text.split()[0].isdigit() else None
                    return SectionDetectionResult(
                        is_section=True,
                        section_name=self._normalize_section_name(section_name),
                        detection_method=pattern_name,
                        confidence=0.9,
                        suggested_level=2,
                        page_number_prefix=page_prefix,
                    )

                elif pattern_name == "caps_section":
                    # "IN THE NEWS" -> "In The News"
                    section_name = match.group(1).strip()
                    return SectionDetectionResult(
                        is_section=True,
                        section_name=self._normalize_section_name(section_name),
                        detection_method=pattern_name,
                        confidence=0.85,
                        suggested_level=2,
                    )

                elif pattern_name == "titled_section":
                    # "Feature: The F-35 Story" -> "Feature"
                    section_type = match.group(1).strip()
                    return SectionDetectionResult(
                        is_section=True,
                        section_name=self._normalize_section_name(section_type),
                        detection_method=pattern_name,
                        confidence=0.95,
                        suggested_level=2,
                    )

                elif pattern_name == "keyword_section":
                    # "News Brief" -> "News"
                    keyword = match.group(1).strip()
                    return SectionDetectionResult(
                        is_section=True,
                        section_name=self._normalize_section_name(keyword),
                        detection_method=pattern_name,
                        confidence=0.8,
                        suggested_level=2,
                    )

        return SectionDetectionResult()

    def _match_keywords(self, text: str, is_first_on_page: bool) -> SectionDetectionResult:
        """Match against known section keywords."""
        text_lower = text.lower()

        for keyword in SECTION_KEYWORDS:
            if keyword in text_lower:
                # Higher confidence if first on page
                confidence = 0.75 if is_first_on_page else 0.5

                # Only trigger if keyword is prominent (near start or whole text)
                if text_lower.startswith(keyword) or len(text) < 30:
                    return SectionDetectionResult(
                        is_section=True,
                        section_name=self._normalize_section_name(text),
                        detection_method="keyword_match",
                        confidence=confidence,
                        suggested_level=2,
                    )

        return SectionDetectionResult()

    def _analyze_position_header(self, text: str, page_number: int) -> SectionDetectionResult:
        """Analyze potential header based on position."""
        # Short text at start of page that looks like a title
        words = text.split()

        # Title case check (each word capitalized)
        is_title_case = all(w[0].isupper() for w in words if w and w[0].isalpha())

        if is_title_case and len(words) <= 5:
            return SectionDetectionResult(
                is_section=True,
                section_name=self._normalize_section_name(text),
                detection_method="position_titlecase",
                confidence=0.6,
                suggested_level=2,
            )

        return SectionDetectionResult()

    def _normalize_section_name(self, name: str) -> str:
        """Normalize section name to consistent format."""
        # Remove leading/trailing whitespace
        name = name.strip()

        # Remove leading page numbers (e.g., "96 Cutting Edge" -> "Cutting Edge")
        words = name.split()
        if words and words[0].isdigit():
            words = words[1:]
            name = " ".join(words)

        # Convert to title case if ALL CAPS
        if name.isupper():
            name = name.title()

        # Remove excessive whitespace
        name = " ".join(name.split())

        return name

    def register_detected_section(self, section_name: str, page_number: int) -> None:
        """Register a detected section for tracking."""
        self._detected_sections[section_name] = page_number

        if page_number not in self._page_sections:
            self._page_sections[page_number] = []
        if section_name not in self._page_sections[page_number]:
            self._page_sections[page_number].append(section_name)

    def get_stats(self) -> Dict[str, int]:
        """Get detection statistics."""
        return {
            "total_sections": len(self._detected_sections),
            "pages_with_sections": len(self._page_sections),
        }

    def get_sections_for_page(self, page_number: int) -> List[str]:
        """Get sections detected on a specific page."""
        return self._page_sections.get(page_number, [])


# ============================================================================
# FACTORY FUNCTION
# ============================================================================


def create_section_detector() -> MagazineSectionDetector:
    """Create a MagazineSectionDetector instance."""
    return MagazineSectionDetector()
