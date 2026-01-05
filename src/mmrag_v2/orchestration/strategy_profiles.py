"""
Strategy Profiles - Polymorphic Document Processing Strategies
===============================================================
ENGINE_USE: Polymorphism-based strategy selection (no if/else chains)

This module implements the "Strategy Profile" pattern to isolate
document-specific processing logic. Each profile encapsulates:
- Extraction parameters (sensitivity, min dimensions)
- VLM prompt configuration (hints, freedom level)
- Post-processing rules

ARCHITECTURAL PRINCIPLE: SEPARATION BY POLYMORPHISM
====================================================
- DigitalMagazineProfile: For born-digital PDFs (Combat Aircraft, etc.)
- ScannedDegradedProfile: For scanned legacy documents (Firearms, etc.)

Changes to scan processing NEVER affect digital processing and vice versa.
Each profile is a self-contained configuration unit.

REQ Compliance:
- REQ-PROFILE-01: Polymorphic profile selection based on diagnostics
- REQ-PROFILE-02: No cross-contamination between digital and scan flows
- REQ-PROFILE-03: Fallback to safe digital profile on uncertainty

Author: Claude (Architect)
Date: 2025-01-03
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .document_diagnostic import DiagnosticReport

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================


class ProfileType(str, Enum):
    """Strategy profile type identifier."""

    DIGITAL_MAGAZINE = "digital_magazine"
    SCANNED_DEGRADED = "scanned_degraded"
    SCANNED_CLEAN = "scanned_clean"
    UNKNOWN = "unknown"


class VLMFreedom(str, Enum):
    """VLM interpretation freedom level."""

    STRICT = "strict"  # Stay close to pixels, no interpretation
    MODERATE = "moderate"  # Some contextual interpretation allowed
    HIGH = "high"  # Interpret through artifacts, fill gaps


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class ProfileParameters:
    """
    Extraction and VLM parameters for a profile.

    These parameters control the extraction pipeline behavior.
    """

    # Extraction parameters
    sensitivity: float = 0.5
    min_image_width: int = 50
    min_image_height: int = 50
    extract_backgrounds: bool = True
    enable_shadow_extraction: bool = True

    # VLM parameters
    vlm_freedom: VLMFreedom = VLMFreedom.MODERATE
    inject_scan_hints: bool = False
    inject_historical_hints: bool = False
    confidence_threshold: float = 0.7

    # Post-processing
    strip_artifacts_from_text: bool = False
    aggressive_deduplication: bool = False

    # OCR Hint Engine (REQ-OCR-01: Non-destructive hybrid layer)
    enable_ocr_hints: bool = False
    ocr_min_confidence: float = 0.4
    ocr_languages: List[str] = None  # type: ignore

    # Dynamic DPI (REQ-DPI-01: Profile-based resolution)
    render_dpi: int = 150  # Default DPI for page rendering

    def __post_init__(self) -> None:
        """Initialize default values for mutable fields."""
        if self.ocr_languages is None:
            self.ocr_languages = ["en"]


@dataclass
class VLMPromptConfig:
    """
    VLM prompt configuration for a profile.

    Controls what context and hints are injected into VLM prompts.
    """

    base_hints: List[str] = field(default_factory=list)
    artifact_hints: List[str] = field(default_factory=list)
    domain_context: str = ""
    freedom_instruction: str = ""

    def build_diagnostic_hints(self) -> str:
        """Build the diagnostic hints string for VLM prompt injection."""
        parts = []

        if self.base_hints:
            parts.append("CONTEXT HINTS:")
            parts.extend(f"- {hint}" for hint in self.base_hints)

        if self.artifact_hints:
            parts.append("\nARTIFACT HANDLING:")
            parts.extend(f"- {hint}" for hint in self.artifact_hints)

        if self.domain_context:
            parts.append(f"\nDOMAIN: {self.domain_context}")

        if self.freedom_instruction:
            parts.append(f"\n{self.freedom_instruction}")

        return "\n".join(parts) if parts else ""


# ============================================================================
# BASE PROFILE (Abstract)
# ============================================================================


class BaseProfile(ABC):
    """
    Abstract base class for strategy profiles.

    Each profile implements specific behavior for a document type.
    Subclasses MUST implement all abstract methods.

    POLYMORPHISM PRINCIPLE:
    - No if/else chains in calling code
    - Profile selection happens once at initialization
    - All subsequent calls use the profile's methods directly
    """

    @property
    @abstractmethod
    def profile_type(self) -> ProfileType:
        """Return the profile type identifier."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable profile name."""
        pass

    @abstractmethod
    def get_parameters(self) -> ProfileParameters:
        """Return extraction and VLM parameters for this profile."""
        pass

    @abstractmethod
    def get_vlm_prompt_config(self) -> VLMPromptConfig:
        """Return VLM prompt configuration for this profile."""
        pass

    @abstractmethod
    def should_use_diagnostic_context(self) -> bool:
        """Whether to inject diagnostic context into VLM prompts."""
        pass

    def get_diagnostic_context(self) -> Dict[str, Any]:
        """
        Build diagnostic context dict for VLM prompt injection.

        Returns:
            Dict compatible with VisionManager.enrich_image_with_diagnostics()
        """
        params = self.get_parameters()
        prompt_config = self.get_vlm_prompt_config()

        return {
            "classification": self.profile_type.value,
            "is_scan": params.inject_scan_hints,
            "scan_hints": prompt_config.artifact_hints if params.inject_scan_hints else [],
            "content_domain": prompt_config.domain_context,
            "confidence_level": "high" if params.confidence_threshold >= 0.8 else "medium",
            "detected_features": prompt_config.base_hints,
        }

    def describe(self) -> str:
        """Get human-readable description of profile configuration."""
        params = self.get_parameters()
        return (
            f"Profile: {self.name} | "
            f"Sensitivity: {params.sensitivity:.1f} | "
            f"Min: {params.min_image_width}x{params.min_image_height}px | "
            f"VLM Freedom: {params.vlm_freedom.value} | "
            f"Scan Hints: {'Yes' if params.inject_scan_hints else 'No'}"
        )


# ============================================================================
# DIGITAL MAGAZINE PROFILE
# ============================================================================


class DigitalMagazineProfile(BaseProfile):
    """
    Profile for born-digital magazines and editorial content.

    CHARACTERISTICS:
    - High-fidelity digital source (Combat Aircraft, etc.)
    - Clean pixels, sharp text, professional layout
    - Many small assets (ads, logos, icons to filter)

    VLM STRATEGY:
    - STRICT freedom: Stay close to what's visible in pixels
    - NO scan hints: Never treat artifacts as expected
    - Low hallucination tolerance: Precise descriptions only

    EXTRACTION:
    - Higher min dimensions to filter out icons/ads
    - Background extraction enabled for editorial photos
    - Standard sensitivity (0.5-0.7)

    This profile preserves the successful magazine workflow.
    """

    @property
    def profile_type(self) -> ProfileType:
        return ProfileType.DIGITAL_MAGAZINE

    @property
    def name(self) -> str:
        return "High-Fidelity Digital Magazine"

    def get_parameters(self) -> ProfileParameters:
        return ProfileParameters(
            # Extraction - proven magazine values
            sensitivity=0.5,
            min_image_width=100,  # Filter small ads/icons
            min_image_height=100,
            extract_backgrounds=True,  # Editorial photos
            enable_shadow_extraction=True,
            # VLM - strict, pixel-accurate
            vlm_freedom=VLMFreedom.STRICT,
            inject_scan_hints=False,  # NEVER inject scan hints
            inject_historical_hints=False,
            confidence_threshold=0.8,  # High bar for descriptions
            # Post-processing
            strip_artifacts_from_text=False,
            aggressive_deduplication=False,
        )

    def get_vlm_prompt_config(self) -> VLMPromptConfig:
        return VLMPromptConfig(
            base_hints=[
                "This is a digital magazine with sharp, high-resolution images",
                "Describe visual elements precisely as they appear",
                "Distinguish editorial photos from advertisements",
            ],
            artifact_hints=[],  # NO artifact hints for digital
            domain_context="editorial/magazine",
            freedom_instruction=(
                "STRICT MODE: Describe ONLY what you can clearly see. "
                "Do not interpret or guess at obscured content. "
                "If an element is unclear, state that explicitly."
            ),
        )

    def should_use_diagnostic_context(self) -> bool:
        # Digital magazines use minimal diagnostic context
        # to prevent VLM from "trying too hard"
        return False


# ============================================================================
# SCANNED DEGRADED PROFILE
# ============================================================================


class ScannedDegradedProfile(BaseProfile):
    """
    Profile for scanned legacy documents with degradation.

    CHARACTERISTICS:
    - Scanned paper documents (Firearms manuals, historical docs)
    - Visible artifacts: grain, stains, foxing, yellowing
    - OCR may be poor or absent
    - Often single large image per page

    VLM STRATEGY:
    - HIGH freedom: Interpret through artifacts
    - INJECT scan hints: Tell VLM to ignore paper quality
    - Historical context: Expect dated typography

    EXTRACTION:
    - Lower min dimensions to catch degraded content
    - Higher sensitivity for recall
    - Shadow extraction critical for image recovery

    This profile enables heavy VLM treatment for legacy scans.
    """

    @property
    def profile_type(self) -> ProfileType:
        return ProfileType.SCANNED_DEGRADED

    @property
    def name(self) -> str:
        return "Legacy Scan Analyst"

    def get_parameters(self) -> ProfileParameters:
        return ProfileParameters(
            # Extraction - optimized for scans
            sensitivity=0.8,  # Higher recall
            min_image_width=30,  # Catch degraded content
            min_image_height=30,
            extract_backgrounds=True,
            enable_shadow_extraction=True,  # Critical for scans
            # VLM - VISUAL-ONLY mode (REQ-VLM-NOISE: No textual meta-language)
            vlm_freedom=VLMFreedom.STRICT,  # CHANGED: STRICT for visual-only descriptions
            inject_scan_hints=True,  # Tell VLM about artifacts
            inject_historical_hints=True,
            confidence_threshold=0.8,  # CHANGED: Higher bar (0.7 → clean profile)
            # Post-processing
            strip_artifacts_from_text=True,  # Clean OCR noise
            aggressive_deduplication=True,
            # OCR Hints - REQ-OCR-01: Enable for scanned documents
            enable_ocr_hints=True,  # Activate EasyOCR hint injection
            ocr_min_confidence=0.4,  # Lower threshold for degraded scans
            ocr_languages=["en"],  # English for Firearms manual
            # Dynamic DPI - REQ-DPI-01: Higher resolution for degraded scans
            render_dpi=300,  # 300 DPI for better OCR/VLM on scans
        )

    def get_vlm_prompt_config(self) -> VLMPromptConfig:
        return VLMPromptConfig(
            base_hints=[
                "This is a scanned historical/legacy document",
                "Focus on the CONTENT, not the paper quality",
                "Expect aged typography and printing imperfections",
            ],
            artifact_hints=[
                "IGNORE paper texture, grain, and discoloration",
                "IGNORE scan artifacts, dust specks, fold marks",
                "IGNORE stains, foxing, and yellowing",
                "Focus ONLY on the actual printed/drawn content",
                "Do NOT describe paper quality issues as content",
            ],
            domain_context="technical/historical",
            freedom_instruction=(
                "INTERPRETIVE MODE: You may interpret content through "
                "minor visual artifacts. If text or images are partially "
                "obscured by paper damage, provide your best interpretation "
                "while noting uncertainty. Prioritize content recovery."
            ),
        )

    def should_use_diagnostic_context(self) -> bool:
        # Scans benefit from full diagnostic context
        return True


# ============================================================================
# SCANNED CLEAN PROFILE
# ============================================================================


class ScannedCleanProfile(BaseProfile):
    """
    Profile for high-quality scanned documents (confidence 0.70+).

    CHARACTERISTICS:
    - Scanned paper documents with good quality
    - Minimal artifacts, clear text, sharp images
    - OCR should work well with layout-aware mode
    - Firearms manual quality level (confidence ~0.70)

    VLM STRATEGY:
    - STRICT freedom: Visual-only descriptions, no text interpretation
    - INJECT scan hints: Tell VLM to ignore minor paper artifacts
    - Layout-aware OCR is primary source of truth

    EXTRACTION:
    - Moderate min dimensions (50x50) - balance between recall and precision
    - Standard sensitivity (0.6)
    - Shadow extraction enabled but not primary

    This profile is the OPTIMAL choice for quality scans like Firearms manual.
    """

    @property
    def profile_type(self) -> ProfileType:
        return ProfileType.SCANNED_CLEAN

    @property
    def name(self) -> str:
        return "High-Quality Scan (Layout-First)"

    def get_parameters(self) -> ProfileParameters:
        return ProfileParameters(
            # Extraction - balanced for quality scans
            sensitivity=0.6,  # Moderate recall
            min_image_width=50,  # Standard minimum
            min_image_height=50,
            extract_backgrounds=True,
            enable_shadow_extraction=True,
            # VLM - STRICT visual-only mode (REQ-VLM-NOISE)
            vlm_freedom=VLMFreedom.STRICT,  # Visual descriptors only
            inject_scan_hints=True,  # Minimal artifact handling
            inject_historical_hints=False,  # Not needed for clean scans
            confidence_threshold=0.8,  # High bar for VLM descriptions
            # Post-processing
            strip_artifacts_from_text=False,  # OCR should be clean
            aggressive_deduplication=False,
            # OCR Hints - REQ-OCR-01: Enable for layout-aware hybrid
            enable_ocr_hints=True,  # EasyOCR hints for brand names
            ocr_min_confidence=0.5,  # Higher threshold for clean scans
            ocr_languages=["en"],
            # Dynamic DPI - REQ-DPI-01: Standard resolution for clean scans
            render_dpi=150,  # 150 DPI sufficient for clean scans
        )

    def get_vlm_prompt_config(self) -> VLMPromptConfig:
        return VLMPromptConfig(
            base_hints=[
                "This is a high-quality scanned document",
                "Layout-aware OCR has ALREADY extracted ALL text content",
                "Your job: Describe ONLY visual elements (diagrams, photos, technical illustrations)",
                "DO NOT describe or interpret text content",
            ],
            artifact_hints=[
                "IGNORE minor paper texture or grain",
                "Focus ONLY on non-textual visual content",
                "Text is handled by OCR - you handle diagrams and images",
            ],
            domain_context="technical/manual",
            freedom_instruction=(
                "STRICT VISUAL-ONLY MODE:\n"
                "1. Describe ONLY diagrams, illustrations, photographs\n"
                "2. NO text interpretation - OCR owns all text\n"
                "3. NO meta-language ('This image shows...', 'The page contains...')\n"
                "4. Direct visual descriptors: 'Exploded diagram of...', 'Technical schematic showing...'\n"
                "5. If region contains ONLY text (no diagram), respond: 'Text region - OCR processed'"
            ),
        )

    def should_use_diagnostic_context(self) -> bool:
        # Clean scans use diagnostic context for scan hints
        return True


# ============================================================================
# PROFILE MANAGER
# ============================================================================


class ProfileManager:
    """
    Manager for automatic profile selection based on document diagnostics.

    SELECTION LOGIC:
    1. If DiagnosticReport indicates scan → ScannedDegradedProfile
    2. If digital with high confidence → DigitalMagazineProfile
    3. If uncertain → FALLBACK to DigitalMagazineProfile (safe default)

    SAFETY PRINCIPLE:
    When in doubt, use the digital profile. It's safer to under-process
    a scan than to inject scan hints into a digital document.
    """

    # Registry of available profiles
    _profiles: Dict[ProfileType, type] = {
        ProfileType.DIGITAL_MAGAZINE: DigitalMagazineProfile,
        ProfileType.SCANNED_DEGRADED: ScannedDegradedProfile,
        ProfileType.SCANNED_CLEAN: ScannedCleanProfile,
    }

    @classmethod
    def select_profile(
        cls,
        diagnostic_report: Optional["DiagnosticReport"] = None,
        force_profile: Optional[ProfileType] = None,
    ) -> BaseProfile:
        """
        Select appropriate profile based on diagnostics.

        Args:
            diagnostic_report: Output from DocumentDiagnosticEngine
            force_profile: Override automatic selection

        Returns:
            Instantiated profile matching document characteristics

        Raises:
            ValueError: If diagnostic confidence is too low and no force_profile
        """
        # Manual override takes precedence
        if force_profile is not None:
            logger.info(f"[PROFILE] Forced profile: {force_profile.value}")
            profile_class = cls._profiles.get(force_profile, DigitalMagazineProfile)
            return profile_class()

        # No diagnostics → safe default
        if diagnostic_report is None:
            logger.warning("[PROFILE] No diagnostics provided, using safe digital default")
            return DigitalMagazineProfile()

        # Extract key indicators
        is_scan = diagnostic_report.physical_check.is_likely_scan
        confidence = diagnostic_report.confidence_profile.overall_confidence
        modality = diagnostic_report.physical_check.detected_modality

        logger.info(
            f"[PROFILE] Diagnostics: scan={is_scan}, "
            f"confidence={confidence:.2f}, modality={modality.value}"
        )

        # LOW CONFIDENCE HANDLING
        # If we're not sure, default to digital (safer)
        if confidence < 0.5:
            logger.warning(
                f"[PROFILE] Low confidence ({confidence:.2f}), "
                "falling back to safe digital profile"
            )
            return DigitalMagazineProfile()

        # PROFILE SELECTION
        if is_scan:
            # CRITICAL FIX: Check CONFIDENCE FIRST, not modality label
            # A scan with confidence 0.70+ should use ScannedCleanProfile
            # even if diagnostics label it as "degraded"
            from .document_diagnostic import DocumentModality

            # PRIMARY DECISION: Confidence threshold
            if confidence >= 0.70:
                # High confidence → Use layout-first clean profile
                logger.info(
                    f"[PROFILE] Selected: ScannedCleanProfile "
                    f"(confidence={confidence:.2f}, modality={modality.value})"
                )
                return ScannedCleanProfile()
            else:
                # Low confidence → Use degraded profile with interpretive VLM
                logger.info(
                    f"[PROFILE] Selected: ScannedDegradedProfile "
                    f"(confidence={confidence:.2f}, modality={modality.value})"
                )
                return ScannedDegradedProfile()
        else:
            # Not a scan → digital profile
            logger.info("[PROFILE] Selected: DigitalMagazineProfile (digital source)")
            return DigitalMagazineProfile()

    @classmethod
    def get_profile_by_type(cls, profile_type: ProfileType) -> BaseProfile:
        """
        Get profile instance by type.

        Args:
            profile_type: Desired profile type

        Returns:
            Instantiated profile
        """
        profile_class = cls._profiles.get(profile_type, DigitalMagazineProfile)
        return profile_class()

    @classmethod
    def list_available_profiles(cls) -> List[ProfileType]:
        """Return list of available profile types."""
        return list(cls._profiles.keys())
