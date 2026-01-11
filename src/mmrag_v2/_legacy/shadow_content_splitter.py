"""
Shadow Content Splitter - Intelligent Chunking of Full-Page Shadow Assets
==========================================================================
ENGINE_USE: Docling v2.66.0 + VLM descriptions

PROBLEM SOLVED:
81% of shadow assets (13/16) are full-page captures with bbox [0,0,1000,1000].
These large shadows can make RAG answers too broad/vague. This module
splits full-page shadows into semantic sub-chunks based on VLM description.

REQ Compliance:
- REQ-SHADOW-01: Full-page shadows (>50% area) should be evaluated for splitting
- REQ-SHADOW-02: VLM descriptions are parsed for semantic segments
- REQ-SHADOW-03: Sub-chunks inherit parent spatial but get unique content

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│ Shadow Asset (full-page)                                         │
│   bbox: [0, 0, 1000, 1000]                                       │
│   visual_description: "Magazine page with article 'Cut-price     │
│   cruise missile'. Photo of space shuttle. Technical schematic   │
│   of missile. RV image in corner..."                             │
│        ↓                                                         │
│   ShadowContentSplitter.analyze_and_split()                      │
│        ↓                                                         │
│   Semantic Segments:                                             │
│   1. "Article: Cut-price cruise missile"                         │
│   2. "Photo: Space shuttle"                                      │
│   3. "Technical: Missile schematic"                              │
│   4. "Secondary: RV image"                                       │
│        ↓                                                         │
│   Multiple enriched chunks OR single chunk with structured       │
│   content for better RAG retrieval                               │
└─────────────────────────────────────────────────────────────────┘

Author: Claude (Senior Architect)
Date: 2026-01-02
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Threshold for "full-page" detection (as ratio of page area)
FULL_PAGE_THRESHOLD: float = 0.50  # 50% of page

# Maximum segments to extract from a single shadow
MAX_SEGMENTS: int = 5

# Minimum segment text length to be valid
MIN_SEGMENT_LENGTH: int = 15

# Segment detection patterns
SEGMENT_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Article/headline detection
    (re.compile(r"article\s+titled?\s*['\"]?([^'\"\.]+)", re.I), "article"),
    (re.compile(r"headline[:\s]+['\"]?([^'\"\.]+)", re.I), "headline"),
    # Image type detection
    (re.compile(r"photograph?\s+of\s+([^\.]+)", re.I), "photo"),
    (re.compile(r"photo\s+of\s+([^\.]+)", re.I), "photo"),
    (re.compile(r"image\s+of\s+([^\.]+)", re.I), "image"),
    (re.compile(r"picture\s+of\s+([^\.]+)", re.I), "image"),
    # Technical content
    (re.compile(r"schematic\s+(?:of\s+)?([^\.]+)", re.I), "schematic"),
    (re.compile(r"diagram\s+(?:of\s+)?([^\.]+)", re.I), "diagram"),
    (re.compile(r"technical\s+(?:drawing|illustration)\s*(?:of\s+)?([^\.]+)?", re.I), "technical"),
    # Layout elements
    (re.compile(r"magazine\s+(?:page|cover)\s+(?:with\s+)?(?:an\s+)?([^\.]+)?", re.I), "layout"),
    (re.compile(r"book\s+cover\s+(?:featuring?\s+)?([^\.]+)?", re.I), "cover"),
    # Aircraft/military specific (for Combat Aircraft magazine)
    (re.compile(r"fighter\s+jet\s+([^\.]+)?", re.I), "aircraft"),
    (re.compile(r"aircraft\s+([^\.]+)?", re.I), "aircraft"),
    (re.compile(r"jet\s+(?:fighter\s+)?([^\.]+)?", re.I), "aircraft"),
    (re.compile(r"helicopter\s+([^\.]+)?", re.I), "aircraft"),
    (re.compile(r"missile\s+([^\.]+)?", re.I), "weapon"),
    # QR/barcode (often noise, but can be useful for metadata)
    (re.compile(r"qr\s+code", re.I), "metadata"),
    (re.compile(r"barcode", re.I), "metadata"),
]

# Keywords that indicate content richness (worth keeping as shadow)
RICH_CONTENT_KEYWORDS: List[str] = [
    "schematic",
    "diagram",
    "technical",
    "illustration",
    "infographic",
    "cutaway",
    "cross-section",
    "blueprint",
    "specification",
    "annotated",
    "comparison",
    "timeline",
    "map",
    "chart",
    "graph",
    "table",
]


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class SemanticSegment:
    """A semantic segment extracted from VLM description."""

    segment_type: str  # article, photo, schematic, etc.
    content: str  # Extracted content description
    confidence: float  # 0.0 - 1.0
    source_match: Optional[str] = None  # Original matched text

    def to_searchable_text(self) -> str:
        """Convert to RAG-friendly searchable text."""
        if self.segment_type == "article":
            return f"Article: {self.content}"
        elif self.segment_type == "photo":
            return f"Photograph showing {self.content}"
        elif self.segment_type == "schematic":
            return f"Technical schematic: {self.content}"
        elif self.segment_type == "diagram":
            return f"Diagram illustrating {self.content}"
        elif self.segment_type == "aircraft":
            return f"Aircraft: {self.content}"
        elif self.segment_type == "weapon":
            return f"Weapon system: {self.content}"
        elif self.segment_type == "layout":
            return f"Page layout: {self.content}"
        else:
            return self.content


@dataclass
class SplitAnalysis:
    """Result of shadow content analysis."""

    should_split: bool = False
    is_full_page: bool = False
    area_ratio: float = 0.0
    segments: List[SemanticSegment] = field(default_factory=list)
    is_rich_content: bool = False  # True if contains technical diagrams, etc.
    recommendation: str = "keep_as_is"  # keep_as_is, split_segments, structure_content
    structured_content: Optional[str] = None  # Enhanced content for RAG

    def get_segment_count(self) -> int:
        return len(self.segments)


# ============================================================================
# SHADOW CONTENT SPLITTER
# ============================================================================


class ShadowContentSplitter:
    """
    Analyzes and optionally splits full-page shadow assets.

    For large shadows (>50% page area), this class:
    1. Parses VLM description for semantic segments
    2. Determines if content is "rich" (technical diagrams, schematics)
    3. Either splits into sub-chunks or structures content for better RAG

    Usage:
        splitter = ShadowContentSplitter()
        analysis = splitter.analyze(bbox, visual_description)
        if analysis.should_split:
            for segment in analysis.segments:
                create_sub_chunk(segment)
        elif analysis.structured_content:
            use_structured_content(analysis.structured_content)
    """

    def __init__(
        self,
        full_page_threshold: float = FULL_PAGE_THRESHOLD,
        enable_splitting: bool = True,
    ) -> None:
        """
        Initialize ShadowContentSplitter.

        Args:
            full_page_threshold: Area ratio threshold for "full-page" (default 0.5)
            enable_splitting: If False, only provides structured content
        """
        self.full_page_threshold = full_page_threshold
        self.enable_splitting = enable_splitting

        # Statistics
        self._analyzed_count = 0
        self._full_page_count = 0
        self._split_count = 0

    def analyze(
        self,
        bbox: Optional[List[int]],
        visual_description: Optional[str],
        page_number: int = 0,
    ) -> SplitAnalysis:
        """
        Analyze a shadow asset for potential splitting.

        Args:
            bbox: Normalized bbox [l, t, r, b] (0-1000 scale)
            visual_description: VLM-generated description
            page_number: Page number for context

        Returns:
            SplitAnalysis with recommendations
        """
        self._analyzed_count += 1
        analysis = SplitAnalysis()

        # Calculate area ratio
        if bbox and len(bbox) == 4:
            l, t, r, b = bbox
            area = (r - l) * (b - t)
            analysis.area_ratio = area / (1000 * 1000)
            analysis.is_full_page = analysis.area_ratio >= self.full_page_threshold

        if analysis.is_full_page:
            self._full_page_count += 1
            logger.debug(f"Full-page shadow detected: area={analysis.area_ratio:.2%}")

        # Parse VLM description for segments
        if visual_description:
            segments = self._extract_segments(visual_description)
            analysis.segments = segments

            # Check for rich content
            analysis.is_rich_content = self._is_rich_content(visual_description)

            # Determine recommendation
            if analysis.is_full_page and len(segments) >= 2:
                if self.enable_splitting and not analysis.is_rich_content:
                    # Split into separate chunks (not for rich technical content)
                    analysis.should_split = True
                    analysis.recommendation = "split_segments"
                    self._split_count += 1
                else:
                    # Structure content for better searchability
                    analysis.recommendation = "structure_content"
                    analysis.structured_content = self._create_structured_content(
                        segments, visual_description
                    )
            elif analysis.is_rich_content:
                # Rich content: keep as-is but with structured content
                analysis.recommendation = "keep_rich"
                analysis.structured_content = self._create_structured_content(
                    segments, visual_description
                )
            else:
                analysis.recommendation = "keep_as_is"

        logger.debug(
            f"Shadow analysis: page={page_number}, "
            f"full_page={analysis.is_full_page}, "
            f"segments={len(analysis.segments)}, "
            f"rich={analysis.is_rich_content}, "
            f"rec={analysis.recommendation}"
        )

        return analysis

    def _extract_segments(self, description: str) -> List[SemanticSegment]:
        """Extract semantic segments from VLM description."""
        segments: List[SemanticSegment] = []
        used_ranges: List[Tuple[int, int]] = []  # Prevent overlapping matches

        for pattern, segment_type in SEGMENT_PATTERNS:
            for match in pattern.finditer(description):
                # Check for overlapping matches
                start, end = match.span()
                if any(s <= start < e or s < end <= e for s, e in used_ranges):
                    continue

                used_ranges.append((start, end))

                # Extract content
                groups = match.groups()
                content = groups[0] if groups and groups[0] else match.group(0)
                content = content.strip() if content else ""

                if len(content) >= MIN_SEGMENT_LENGTH or segment_type in ("metadata",):
                    segment = SemanticSegment(
                        segment_type=segment_type,
                        content=content[:200],  # Limit length
                        confidence=0.8,
                        source_match=match.group(0)[:100],
                    )
                    segments.append(segment)

                    if len(segments) >= MAX_SEGMENTS:
                        return segments

        return segments

    def _is_rich_content(self, description: str) -> bool:
        """Check if description indicates rich technical content."""
        desc_lower = description.lower()
        return any(keyword in desc_lower for keyword in RICH_CONTENT_KEYWORDS)

    def _create_structured_content(
        self,
        segments: List[SemanticSegment],
        original_description: str,
    ) -> str:
        """Create structured content for better RAG retrieval."""
        parts: List[str] = []

        # Add segment summaries
        if segments:
            parts.append("CONTENT ELEMENTS:")
            for i, seg in enumerate(segments, 1):
                parts.append(f"  {i}. {seg.to_searchable_text()}")

        # Add condensed original description
        parts.append("")
        parts.append("FULL DESCRIPTION:")

        # Truncate long descriptions intelligently
        if len(original_description) > 400:
            # Find sentence boundaries for clean truncation
            sentences = re.split(r"[.!?]\s+", original_description)
            condensed = ""
            for sentence in sentences:
                if len(condensed) + len(sentence) < 380:
                    condensed += sentence + ". "
                else:
                    break
            parts.append(condensed.strip() + "...")
        else:
            parts.append(original_description)

        return "\n".join(parts)

    def get_stats(self) -> Dict[str, int]:
        """Get splitter statistics."""
        return {
            "total_analyzed": self._analyzed_count,
            "full_page_shadows": self._full_page_count,
            "split_recommendations": self._split_count,
            "split_ratio": round(self._split_count / max(1, self._full_page_count) * 100, 1),
        }


# ============================================================================
# FACTORY FUNCTION
# ============================================================================


def create_shadow_splitter(
    enable_splitting: bool = True,
    threshold: float = FULL_PAGE_THRESHOLD,
) -> ShadowContentSplitter:
    """
    Create a ShadowContentSplitter instance.

    Args:
        enable_splitting: If False, only structures content without splitting
        threshold: Area ratio threshold for full-page detection

    Returns:
        Configured ShadowContentSplitter
    """
    return ShadowContentSplitter(
        full_page_threshold=threshold,
        enable_splitting=enable_splitting,
    )
