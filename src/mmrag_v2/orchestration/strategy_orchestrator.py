"""
Strategy Orchestrator - Dynamic Extraction Strategy Configuration
==================================================================
ENGINE_USE: Profile-based extraction parameter optimization

This module creates extraction strategies based on document profiles
and user-specified sensitivity levels.

REQ Compliance:
- REQ-STRAT-01: Dynamic min_image_width/height based on profile
- REQ-STRAT-02: Sensitivity slider (0.1-1.0) for recall tuning
- REQ-STRAT-03: Shadow extraction toggle based on document type

SRS Section 9.2: Extraction Strategy
"The system MUST adjust extraction parameters based on document
profile to optimize precision/recall tradeoff."

Author: Claude 4.5 Opus (Architect)
Date: 2025-12-29
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from rich.console import Console

from .smart_config import DocumentProfile, DocumentType

if TYPE_CHECKING:
    from .strategy_profiles import ProfileParameters, ProfileType

logger = logging.getLogger(__name__)
console = Console()


# ============================================================================
# CONSTANTS
# ============================================================================

# Default minimum image dimensions (in pixels)
DEFAULT_MIN_WIDTH: int = 50
DEFAULT_MIN_HEIGHT: int = 50

# Magazine-optimized dimensions (larger to filter ads/icons)
MAGAZINE_MIN_WIDTH: int = 100
MAGAZINE_MIN_HEIGHT: int = 100

# Academic-optimized dimensions (smaller for diagrams)
ACADEMIC_MIN_WIDTH: int = 30
ACADEMIC_MIN_HEIGHT: int = 30

# Sensitivity bounds
MIN_SENSITIVITY: float = 0.1
MAX_SENSITIVITY: float = 1.0
DEFAULT_SENSITIVITY: float = 0.5


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class ExtractionStrategy:
    """
    Extraction strategy parameters.

    Attributes:
        sensitivity: User-specified sensitivity (0.1-1.0)
        min_image_width: Minimum image width to extract (pixels)
        min_image_height: Minimum image height to extract (pixels)
        extract_backgrounds: Whether to extract background images
        enable_shadow_extraction: Whether to run shadow extraction
        historical_median: Median image dimensions from profile
        document_type: Detected document type
    """

    sensitivity: float = DEFAULT_SENSITIVITY
    min_image_width: int = DEFAULT_MIN_WIDTH
    min_image_height: int = DEFAULT_MIN_HEIGHT
    extract_backgrounds: bool = True
    enable_shadow_extraction: bool = True
    historical_median: int = 0
    document_type: DocumentType = DocumentType.UNKNOWN

    def describe(self) -> str:
        """Get human-readable description of strategy."""
        parts = [
            f"Sensitivity: {self.sensitivity:.1f}",
            f"Min dimensions: {self.min_image_width}x{self.min_image_height}px",
            f"Backgrounds: {'Yes' if self.extract_backgrounds else 'No'}",
            f"Shadow extraction: {'Yes' if self.enable_shadow_extraction else 'No'}",
        ]
        return " | ".join(parts)


# ============================================================================
# STRATEGY ORCHESTRATOR
# ============================================================================


class StrategyOrchestrator:
    """
    Creates extraction strategies based on document profiles.

    The orchestrator takes a DocumentProfile and user sensitivity
    setting to produce an optimized ExtractionStrategy.

    Usage:
        orchestrator = StrategyOrchestrator()
        strategy = orchestrator.create_strategy(profile, sensitivity=0.7)
    """

    def __init__(self) -> None:
        """Initialize StrategyOrchestrator."""
        logger.info("StrategyOrchestrator initialized")

    # ========================================================================
    # PROFILE TYPE TO DOCUMENT TYPE MAPPING (V16 BULLETPROOF FIX)
    # ========================================================================
    # ProfileClassifier returns ProfileType (digital_magazine, scanned_literature, etc.)
    # ExtractionStrategy expects DocumentType (magazine, academic, literature, technical, etc.)
    #
    # V16 FIX (2026-01-10): STRICT MAPPING - each profile maps to its SEMANTIC type
    # =============================================================================
    # BEFORE: scanned_literature → MAGAZINE (WRONG! Books are not magazines!)
    # AFTER:  scanned_literature → LITERATURE (Correct semantic type)
    #
    # BEFORE: technical_manual → REPORT (WRONG! Manuals are technical docs!)
    # AFTER:  technical_manual → TECHNICAL (Correct semantic type)
    #
    # This mapping affects text segmentation and downstream processing:
    # - MAGAZINE: Column-based layout, editorial photos
    # - LITERATURE: Paragraph-based flow, chapter structure
    # - TECHNICAL: Parts diagrams, assembly sequences, callouts
    # - ACADEMIC: Multi-column scientific papers, citations
    PROFILE_TO_DOC_TYPE = {
        "digital_magazine": DocumentType.MAGAZINE,
        "scanned_magazine": DocumentType.MAGAZINE,
        "academic_whitepaper": DocumentType.ACADEMIC,
        "scanned_literature": DocumentType.LITERATURE,  # V16 FIX: Books → LITERATURE
        "scanned_clean": DocumentType.REPORT,
        "scanned_degraded": DocumentType.REPORT,
        "technical_manual": DocumentType.TECHNICAL,  # V16 FIX: Manuals → TECHNICAL
        "standard_digital": DocumentType.UNKNOWN,
    }

    def create_strategy(
        self,
        profile: DocumentProfile,
        sensitivity: float = DEFAULT_SENSITIVITY,
        profile_params: Optional["ProfileParameters"] = None,
        profile_type: Optional[str] = None,  # V2.2: ProfileType.value from classifier
    ) -> ExtractionStrategy:
        """
        Create extraction strategy from document profile.

        ARCHITECTURE UPGRADE (2026-01-10):
        Now accepts optional ProfileParameters from strategy profiles
        for direct parameter control (no heuristics override).

        V2.2 FIX (2026-01-10):
        Added profile_type parameter to correctly map ProfileClassifier
        result to DocumentType, preventing 'academic' ghost bug.

        Args:
            profile: DocumentProfile from SmartConfigProvider
            sensitivity: User sensitivity setting (0.1=strict, 1.0=max recall)
            profile_params: Optional ProfileParameters from selected profile
            profile_type: Optional ProfileType.value string from ProfileClassifier

        Returns:
            Configured ExtractionStrategy
        """
        # Clamp sensitivity
        sensitivity = max(MIN_SENSITIVITY, min(MAX_SENSITIVITY, sensitivity))

        # PRIORITY: If profile_params provided, use those directly
        # This ensures ScannedLiteratureProfile settings are respected
        if profile_params is not None:
            logger.info(
                f"Using profile-driven parameters: "
                f"sensitivity={profile_params.sensitivity:.2f}, "
                f"min={profile_params.min_image_width}x{profile_params.min_image_height}, "
                f"backgrounds={profile_params.extract_backgrounds}"
            )

            # Use profile median if available
            historical_median = max(
                profile.median_image_width,
                profile.median_image_height,
            )

            # V2.2 FIX: Determine document_type from profile_type if provided
            # This ensures digital_magazine -> magazine, not academic
            if profile_type is not None:
                doc_type = self.PROFILE_TO_DOC_TYPE.get(profile_type, profile.document_type)
                logger.info(f"Mapped profile_type '{profile_type}' -> doc_type '{doc_type.value}'")
            else:
                doc_type = profile.document_type

            strategy = ExtractionStrategy(
                sensitivity=profile_params.sensitivity,
                min_image_width=profile_params.min_image_width,
                min_image_height=profile_params.min_image_height,
                extract_backgrounds=profile_params.extract_backgrounds,
                enable_shadow_extraction=profile_params.enable_shadow_extraction,
                historical_median=historical_median,
                document_type=doc_type,
            )

            logger.info(f"Created strategy: {strategy.describe()} | " f"doc_type={doc_type.value}")

            return strategy

        # FALLBACK: Use legacy heuristic-based strategy (backward compatible)
        logger.info("Using legacy heuristic-based strategy (no profile params)")

        # Get base dimensions for document type
        base_width, base_height = self._get_base_dimensions(profile.document_type)

        # Apply sensitivity adjustment per REQ-SENS-01
        # ✅ FIX 4B: Use SRS formula instead of multiplicative scaling
        # REQ-SENS-01: min_dimension = 400px - (sensitivity * 300px)
        # Higher sensitivity = smaller minimum = more recall
        # sensitivity 0.1 → 400 - 30 = 370px (strict, fewer images)
        # sensitivity 0.5 → 400 - 150 = 250px (balanced)
        # sensitivity 1.0 → 400 - 300 = 100px (max recall, more images)
        base_dimension = 400
        sensitivity_range = 300
        min_width = int(base_dimension - (sensitivity * sensitivity_range))
        min_height = int(base_dimension - (sensitivity * sensitivity_range))

        # Use profile median if available, otherwise use computed minimum
        historical_median = max(
            profile.median_image_width,
            profile.median_image_height,
        )

        # Determine background extraction
        # Magazines often have editorial photos in backgrounds
        extract_backgrounds = profile.document_type in (
            DocumentType.MAGAZINE,
            DocumentType.PRESENTATION,
            DocumentType.UNKNOWN,
        )

        # Determine shadow extraction
        # Enable for image-heavy documents where Docling might miss images
        enable_shadow = profile.is_image_heavy() or sensitivity >= 0.7

        strategy = ExtractionStrategy(
            sensitivity=sensitivity,
            min_image_width=min_width,
            min_image_height=min_height,
            extract_backgrounds=extract_backgrounds,
            enable_shadow_extraction=enable_shadow,
            historical_median=historical_median,
            document_type=profile.document_type,
        )

        logger.info(
            f"Created strategy: {strategy.describe()} | " f"doc_type={profile.document_type.value}"
        )

        return strategy

    def _get_base_dimensions(
        self,
        doc_type: DocumentType,
    ) -> tuple[int, int]:
        """
        Get base minimum dimensions for document type.

        Args:
            doc_type: Document type classification

        Returns:
            Tuple of (min_width, min_height)
        """
        if doc_type == DocumentType.MAGAZINE:
            return MAGAZINE_MIN_WIDTH, MAGAZINE_MIN_HEIGHT
        elif doc_type == DocumentType.ACADEMIC:
            return ACADEMIC_MIN_WIDTH, ACADEMIC_MIN_HEIGHT
        elif doc_type == DocumentType.PRESENTATION:
            return ACADEMIC_MIN_WIDTH, ACADEMIC_MIN_HEIGHT  # Slides have small diagrams
        else:
            return DEFAULT_MIN_WIDTH, DEFAULT_MIN_HEIGHT

    def print_strategy_banner(self, strategy: ExtractionStrategy) -> None:
        """
        Print a banner showing the extraction strategy.

        Args:
            strategy: Strategy to display
        """
        console.print()
        console.print("[bold cyan]━━━━━ EXTRACTION STRATEGY ━━━━━[/bold cyan]")
        console.print(f"[cyan]Document Type:[/cyan] {strategy.document_type.value}")
        console.print(f"[cyan]Sensitivity:[/cyan] {strategy.sensitivity:.1f}")
        console.print(
            f"[cyan]Min Dimensions:[/cyan] "
            f"{strategy.min_image_width}x{strategy.min_image_height}px"
        )
        console.print(
            f"[cyan]Background Extraction:[/cyan] "
            f"{'Enabled' if strategy.extract_backgrounds else 'Disabled'}"
        )
        console.print(
            f"[cyan]Shadow Extraction:[/cyan] "
            f"{'Enabled' if strategy.enable_shadow_extraction else 'Disabled'}"
        )
        if strategy.historical_median > 0:
            console.print(f"[cyan]Historical Median:[/cyan] {strategy.historical_median}px")
        console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        console.print()

    def create_default_strategy(
        self,
        sensitivity: float = DEFAULT_SENSITIVITY,
    ) -> ExtractionStrategy:
        """
        Create a default strategy without document analysis.

        Args:
            sensitivity: User sensitivity setting

        Returns:
            Default ExtractionStrategy
        """
        sensitivity = max(MIN_SENSITIVITY, min(MAX_SENSITIVITY, sensitivity))

        # ✅ FIX 4B: Use SRS formula for consistency
        # REQ-SENS-01: min_dimension = 400px - (sensitivity * 300px)
        base_dimension = 400
        sensitivity_range = 300
        min_width = int(base_dimension - (sensitivity * sensitivity_range))
        min_height = int(base_dimension - (sensitivity * sensitivity_range))

        return ExtractionStrategy(
            sensitivity=sensitivity,
            min_image_width=min_width,
            min_image_height=min_height,
            extract_backgrounds=True,
            enable_shadow_extraction=sensitivity >= 0.7,
            historical_median=0,
            document_type=DocumentType.UNKNOWN,
        )
