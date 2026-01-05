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
from typing import Optional

from rich.console import Console

from .smart_config import DocumentProfile, DocumentType

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
    
    def create_strategy(
        self,
        profile: DocumentProfile,
        sensitivity: float = DEFAULT_SENSITIVITY,
    ) -> ExtractionStrategy:
        """
        Create extraction strategy from document profile.
        
        Args:
            profile: DocumentProfile from SmartConfigProvider
            sensitivity: User sensitivity setting (0.1=strict, 1.0=max recall)
            
        Returns:
            Configured ExtractionStrategy
        """
        # Clamp sensitivity
        sensitivity = max(MIN_SENSITIVITY, min(MAX_SENSITIVITY, sensitivity))
        
        # Get base dimensions for document type
        base_width, base_height = self._get_base_dimensions(profile.document_type)
        
        # Apply sensitivity adjustment
        # Higher sensitivity = smaller minimum = more recall
        # sensitivity 0.1 → multiply by 2.0 (larger min, fewer images)
        # sensitivity 1.0 → multiply by 0.5 (smaller min, more images)
        size_multiplier = 2.0 - (sensitivity * 1.5)
        
        min_width = int(base_width * size_multiplier)
        min_height = int(base_height * size_multiplier)
        
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
            f"Created strategy: {strategy.describe()} | "
            f"doc_type={profile.document_type.value}"
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
            console.print(
                f"[cyan]Historical Median:[/cyan] {strategy.historical_median}px"
            )
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
        
        size_multiplier = 2.0 - (sensitivity * 1.5)
        min_width = int(DEFAULT_MIN_WIDTH * size_multiplier)
        min_height = int(DEFAULT_MIN_HEIGHT * size_multiplier)
        
        return ExtractionStrategy(
            sensitivity=sensitivity,
            min_image_width=min_width,
            min_image_height=min_height,
            extract_backgrounds=True,
            enable_shadow_extraction=sensitivity >= 0.7,
            historical_median=0,
            document_type=DocumentType.UNKNOWN,
        )
