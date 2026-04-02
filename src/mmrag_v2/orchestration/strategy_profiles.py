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
    ACADEMIC_WHITEPAPER = "academic_whitepaper"
    SCANNED_DEGRADED = "scanned_degraded"
    SCANNED = "scanned"  # Standard quality scans (replaces scanned_clean/literature/magazine)
    TECHNICAL_MANUAL = "technical_manual"
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
    vlm_table_enabled: bool = True
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

    # Batch size safety (V2.2: Prevent OOM for high-DPI profiles)
    # Lower batch size for profiles with higher DPI to prevent memory issues
    recommended_batch_size: int = 10  # Default: 10 pages per batch

    def __post_init__(self) -> None:
        """Initialize default values for mutable fields."""
        if self.ocr_languages is None:
            self.ocr_languages = ["en"]


@dataclass
class AdaptiveSettings:
    """
    Per-document adaptive overrides layered on top of ProfileParameters.

    Only non-None fields are applied; everything else stays as in the base profile.
    """

    sensitivity: Optional[float] = None
    min_image_width: Optional[int] = None
    min_image_height: Optional[int] = None
    ocr_confidence_threshold: Optional[float] = None
    enable_aggressive_ocr: Optional[bool] = None
    semantic_overlap_ratio: Optional[float] = None


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

    def get_adaptive_settings(
        self,
        diagnostics: "DiagnosticReport",
        base_params: ProfileParameters,
        doc_profile: Optional[Any] = None,
    ) -> Optional["AdaptiveSettings"]:
        """
        Optional per-document adaptive overrides. Defaults to no overrides.
        """
        return None

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

    def get_adaptive_settings(
        self,
        diagnostics: "DiagnosticReport",
        base_params: ProfileParameters,
        doc_profile: Optional[Any] = None,
    ) -> Optional["AdaptiveSettings"]:
        # If magazine is very image-heavy, loosen filters and allow more recall.
        image_cov = float(getattr(diagnostics.physical_check, "avg_image_coverage", 0.0) or 0.0)
        median_dim = 0
        if doc_profile is not None:
            median_dim = max(
                int(getattr(doc_profile, "median_image_width", 0) or 0),
                int(getattr(doc_profile, "median_image_height", 0) or 0),
            )
        if image_cov > 0.6 or median_dim > 200:
            return AdaptiveSettings(
                sensitivity=0.7,  # slightly higher recall
                min_image_width=40,
                min_image_height=40,
                semantic_overlap_ratio=0.10,
            )
        return None


# ============================================================================
# ACADEMIC WHITEPAPER PROFILE
# ============================================================================


class AcademicWhitepaperProfile(BaseProfile):
    """
    Profile for academic papers, research documents, and technical whitepapers.

    CHARACTERISTICS:
    - Born-digital PDFs with high text density (>3000 chars/page)
    - Multi-column layout (typical Arxiv/IEEE/ACM style)
    - Technical diagrams, charts, and architecture schematics
    - Few decorative images, no advertisements
    - Structured sections (Abstract, Introduction, Related Work, References)

    VLM STRATEGY:
    - STRICT freedom: Focus on technical accuracy of diagrams
    - NO scan hints: Clean digital source
    - Technical domain context: Prioritize system architectures and data flows

    EXTRACTION:
    - Smaller min dimensions (30x30) to capture small diagrams and equations
    - Background extraction disabled (no editorial photos)
    - Shadow extraction enabled for embedded technical figures

    This profile optimizes for technical RAG applications where diagram
    understanding is critical and textual hierarchy must be preserved.
    """

    @property
    def profile_type(self) -> ProfileType:
        return ProfileType.ACADEMIC_WHITEPAPER

    @property
    def name(self) -> str:
        return "Academic/Technical Whitepaper"

    def get_parameters(self) -> ProfileParameters:
        return ProfileParameters(
            # Extraction - optimized for technical diagrams
            sensitivity=0.6,  # Moderate-high recall for small diagrams
            min_image_width=30,  # Catch small equation images and flowcharts
            min_image_height=30,
            extract_backgrounds=False,  # No editorial backgrounds in papers
            enable_shadow_extraction=True,  # Catch embedded figures
            # VLM - strict technical mode
            vlm_freedom=VLMFreedom.STRICT,
            inject_scan_hints=False,  # Digital source
            inject_historical_hints=False,
            confidence_threshold=0.8,  # High bar for technical accuracy
            # Post-processing
            strip_artifacts_from_text=False,
            aggressive_deduplication=False,  # Preserve all technical content
            # OCR Hints - disabled for clean digital academic papers
            enable_ocr_hints=False,
            ocr_min_confidence=0.5,
            ocr_languages=["en"],
            # Dynamic DPI - standard for digital
            render_dpi=150,
        )

    def get_vlm_prompt_config(self) -> VLMPromptConfig:
        return VLMPromptConfig(
            base_hints=[
                "This is an academic/technical document with high information density",
                "Focus on technical diagrams, system architectures, and data flow charts",
                "Describe technical content with precision - this is for technical RAG",
                "Distinguish between architectural diagrams, flowcharts, graphs, and equations",
            ],
            artifact_hints=[],  # No artifacts in clean digital papers
            domain_context="academic/technical",
            freedom_instruction=(
                "TECHNICAL STRICT MODE:\n"
                "1. Describe ONLY technical visual elements (diagrams, charts, architectures)\n"
                "2. Use technical vocabulary appropriate to the domain\n"
                "3. Identify system components, data flows, and relationships\n"
                "4. For equations/formulas: Note their presence but defer to OCR for exact content\n"
                "5. Focus on WHAT is shown, not aesthetic qualities\n"
                "6. NO generic descriptions like 'This image shows...' - be direct and technical"
            ),
        )

    def should_use_diagnostic_context(self) -> bool:
        # Academic papers benefit from domain context
        return True

    def get_adaptive_settings(
        self,
        diagnostics: "DiagnosticReport",
        base_params: ProfileParameters,
        doc_profile: Optional[Any] = None,
    ) -> Optional["AdaptiveSettings"]:
        image_cov = float(getattr(diagnostics.physical_check, "avg_image_coverage", 0.0) or 0.0)
        median_dim = 0
        if doc_profile is not None:
            median_dim = max(
                int(getattr(doc_profile, "median_image_width", 0) or 0),
                int(getattr(doc_profile, "median_image_height", 0) or 0),
            )
        if image_cov > 0.25 or median_dim > 150:
            return AdaptiveSettings(
                min_image_width=30,
                min_image_height=30,
                sensitivity=min(0.8, base_params.sensitivity + 0.05),
                semantic_overlap_ratio=0.10,
            )
        return None


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

    def get_adaptive_settings(
        self,
        diagnostics: "DiagnosticReport",
        base_params: ProfileParameters,
        doc_profile: Optional[Any] = None,
    ) -> Optional["AdaptiveSettings"]:
        overall_conf = getattr(diagnostics.confidence_profile, "overall_confidence", 1.0)
        if overall_conf < 0.6:
            return AdaptiveSettings(
                enable_aggressive_ocr=True,
                ocr_confidence_threshold=0.3,
            )
        return None


# ============================================================================
# SCANNED PROFILE (consolidates scanned_clean, scanned_literature, scanned_magazine)
# ============================================================================


class ScannedProfile(BaseProfile):
    """
    Profile for standard-quality scanned documents.

    Replaces the three former profiles (scanned_clean, scanned_literature,
    scanned_magazine) which shared identical batch-processor behavior and
    only differed in minor extraction parameters.

    CHARACTERISTICS:
    - Scanned documents with reasonable quality (scan_confidence 0.60+)
    - Covers: scanned books, scanned magazines, clean scanned manuals
    - NOT for heavily degraded scans (use scanned_degraded)
    - NOT for technical manuals needing fine-detail extraction (use technical_manual)

    VLM STRATEGY:
    - STRICT freedom: Visual-only descriptions
    - INJECT scan hints: Ignore paper artifacts
    - Generic hints suitable for any scan type

    EXTRACTION:
    - min_image: 30px — catches small illustrations and decorative elements
    - sensitivity: 0.70 — good recall without excessive noise
    - render_dpi: 200 — good quality for most scans
    """

    @property
    def profile_type(self) -> ProfileType:
        return ProfileType.SCANNED

    @property
    def name(self) -> str:
        return "Scanned Document"

    def get_parameters(self) -> ProfileParameters:
        return ProfileParameters(
            sensitivity=0.70,
            min_image_width=30,
            min_image_height=30,
            extract_backgrounds=True,
            enable_shadow_extraction=True,
            vlm_freedom=VLMFreedom.STRICT,
            inject_scan_hints=True,
            inject_historical_hints=False,
            confidence_threshold=0.8,
            strip_artifacts_from_text=False,
            aggressive_deduplication=False,
            enable_ocr_hints=True,
            ocr_min_confidence=0.5,
            ocr_languages=["en"],
            render_dpi=200,
            recommended_batch_size=10,
        )

    def get_vlm_prompt_config(self) -> VLMPromptConfig:
        return VLMPromptConfig(
            base_hints=[
                "This is a scanned document",
                "OCR has extracted text content — describe ONLY visual elements",
                "Focus on diagrams, photos, illustrations, and artwork",
            ],
            artifact_hints=[
                "IGNORE paper texture, grain, and discoloration",
                "IGNORE scan artifacts, dust specks, and binding shadows",
                "Focus ONLY on intentional visual content",
            ],
            domain_context="scanned document",
            freedom_instruction=(
                "SCANNED DOCUMENT MODE:\n"
                "1. Describe ONLY visual elements (diagrams, photos, illustrations)\n"
                "2. NO text interpretation — OCR owns all text\n"
                "3. IGNORE paper and scan quality artifacts\n"
                "4. Use direct visual descriptors without meta-language"
            ),
        )

    def should_use_diagnostic_context(self) -> bool:
        return True

    def get_adaptive_settings(
        self,
        diagnostics: "DiagnosticReport",
        base_params: ProfileParameters,
        doc_profile: Optional[Any] = None,
    ) -> Optional["AdaptiveSettings"]:
        overall_conf = getattr(diagnostics.confidence_profile, "overall_confidence", 1.0)
        if overall_conf < 0.6:
            return AdaptiveSettings(
                enable_aggressive_ocr=True,
                ocr_confidence_threshold=0.3,
            )
        return None


# ============================================================================
# TECHNICAL MANUAL PROFILE (NEW)
# ============================================================================


class TechnicalManualProfile(BaseProfile):
    """
    Profile for technical manuals with small but critical visual elements.

    CRITICAL USE CASE: Firearms.pdf scenario
    =========================================
    Technical manuals contain SMALL but CRITICAL visual elements:
    - Small parts diagrams: pins, screws, springs (often < 50px)
    - Part numbers and callout labels
    - Exploded view diagrams with fine detail
    - Assembly/disassembly sequences with numbered steps

    Standard scanned_clean profile (50x50 min) would FILTER OUT:
    - Small spring diagrams
    - Tiny screw illustrations
    - Part callout numbers
    - Detail insets

    This is CATASTROPHIC for RAG on technical documentation!

    SOLUTION:
    - Min dimensions: 30x30px (catch every bolt and spring)
    - Sensitivity: 0.8 (high recall for small parts)
    - DPI: 300 (preserve fine detail)

    VLM STRATEGY:
    - STRICT freedom: Technical accuracy is paramount
    - INJECT scan hints: If scanned, handle artifacts
    - Technical domain context: Focus on parts, assembly, function

    This profile ensures NO technical detail is lost.
    """

    @property
    def profile_type(self) -> ProfileType:
        return ProfileType.TECHNICAL_MANUAL

    @property
    def name(self) -> str:
        return "Technical Manual (Detail-First)"

    def get_parameters(self) -> ProfileParameters:
        return ProfileParameters(
            # Extraction - AGGRESSIVE for small technical details
            sensitivity=0.8,  # HIGH recall - catch every small part
            min_image_width=30,  # SMALL - catch pins, screws, springs
            min_image_height=30,  # SMALL - small parts diagrams
            extract_backgrounds=True,  # Technical diagrams may be in background
            enable_shadow_extraction=True,  # Catch embedded schematics
            # VLM - STRICT technical mode
            vlm_freedom=VLMFreedom.STRICT,  # Technical accuracy is paramount
            vlm_table_enabled=False,  # Prevent table hallucinations; use docling/OCR path
            inject_scan_hints=True,  # Handle potential scan artifacts
            inject_historical_hints=False,  # Not historical (unless detected)
            confidence_threshold=0.8,  # High bar for technical descriptions
            # Post-processing
            strip_artifacts_from_text=False,  # Preserve all text
            aggressive_deduplication=False,  # Keep all technical content
            # OCR Hints - CRITICAL for part numbers and labels
            enable_ocr_hints=True,  # Capture part numbers
            ocr_min_confidence=0.4,  # Lower threshold - catch faint labels
            ocr_languages=["en"],
            # Dynamic DPI - HIGH for fine technical detail
            render_dpi=300,  # Maximum quality for small parts
            # V2.2 SAFETY: Lower batch size for 300 DPI to prevent OOM
            recommended_batch_size=3,  # 3 pages per batch at 300 DPI
        )

    def get_vlm_prompt_config(self) -> VLMPromptConfig:
        return VLMPromptConfig(
            base_hints=[
                "This is a TECHNICAL MANUAL with detailed parts diagrams",
                "Focus on: exploded views, assembly sequences, part callouts",
                "SMALL PARTS MATTER: pins, screws, springs, clips are critical",
                "Part numbers and labels are essential for technical RAG",
            ],
            artifact_hints=[
                "IGNORE paper texture and minor scan artifacts",
                "Focus on technical content: diagrams, schematics, part illustrations",
                "Small elements are intentional technical details, NOT noise",
            ],
            domain_context="technical/manual",
            freedom_instruction=(
                "TECHNICAL MANUAL MODE:\n"
                "1. Describe ALL technical visual elements - size doesn't indicate importance\n"
                "2. Small parts (pins, screws, springs) are CRITICAL - describe them\n"
                "3. Note exploded views, assembly sequences, cross-sections\n"
                "4. Identify part relationships and assembly order\n"
                "5. Part numbers and callouts are valuable - note their presence\n"
                "6. Use technical vocabulary: 'detent spring', 'retaining pin', 'sear', etc.\n"
                "7. NO generic descriptions - be specific and technical"
            ),
        )

    def should_use_diagnostic_context(self) -> bool:
        # Technical manuals benefit from diagnostic context
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
        ProfileType.ACADEMIC_WHITEPAPER: AcademicWhitepaperProfile,
        ProfileType.SCANNED_DEGRADED: ScannedDegradedProfile,
        ProfileType.SCANNED: ScannedProfile,
        ProfileType.TECHNICAL_MANUAL: TechnicalManualProfile,
    }

    @classmethod
    def select_profile(
        cls,
        diagnostic_report: Optional["DiagnosticReport"] = None,
        force_profile: Optional[ProfileType] = None,
        doc_profile: Optional[Any] = None,
    ) -> BaseProfile:
        """
        Select appropriate profile based on diagnostics.

        ARCHITECTURE V2.0 (2026-01-10): MULTI-DIMENSIONAL CLASSIFICATION
        ==================================================================
        This method now uses the ProfileClassifier for intelligent,
        feature-based profile selection instead of hardcoded if/else chains.

        Args:
            diagnostic_report: Output from DocumentDiagnosticEngine
            force_profile: Override automatic selection
            doc_profile: DocumentProfile from SmartConfigProvider (for classifier)

        Returns:
            Instantiated profile matching document characteristics
        """
        # Manual override takes precedence
        if force_profile is not None:
            logger.info(f"[PROFILE] Forced profile: {force_profile.value}")
            profile_class = cls._profiles.get(force_profile, DigitalMagazineProfile)
            return profile_class()

        # MULTI-DIMENSIONAL CLASSIFICATION PATH
        # If doc_profile provided, use the new intelligent classifier
        if doc_profile is not None:
            from .profile_classifier import ProfileClassifier
            from .profile_classifier import ProfileType as ClassifierProfileType

            logger.info("[PROFILE] Using multi-dimensional classifier")
            classifier = ProfileClassifier()
            selected_type = classifier.classify(doc_profile, diagnostic_report)

            # Map ClassifierProfileType to profile class
            type_mapping = {
                ClassifierProfileType.ACADEMIC_WHITEPAPER: AcademicWhitepaperProfile,
                ClassifierProfileType.DIGITAL_MAGAZINE: DigitalMagazineProfile,
                ClassifierProfileType.SCANNED: ScannedProfile,
                ClassifierProfileType.SCANNED_DEGRADED: ScannedDegradedProfile,
                ClassifierProfileType.TECHNICAL_MANUAL: TechnicalManualProfile,
                ClassifierProfileType.STANDARD_DIGITAL: DigitalMagazineProfile,
            }

            profile_class = type_mapping.get(selected_type, DigitalMagazineProfile)
            logger.info(f"[PROFILE] Classifier selected: {selected_type.value}")
            return profile_class()

        # FALLBACK: LEGACY HEURISTIC PATH (backward compatible)
        # This path is used when doc_profile is not provided
        logger.warning("[PROFILE] Using legacy heuristic selection (doc_profile not provided)")

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
        if confidence < 0.5:
            logger.warning(
                f"[PROFILE] Low confidence ({confidence:.2f}), "
                "falling back to safe digital profile"
            )
            return DigitalMagazineProfile()

        # LEGACY PROFILE SELECTION (kept for backward compatibility)
        if is_scan:
            # Confidence-based selection: degraded scans need extra OCR tolerance
            if confidence >= 0.70:
                logger.info(
                    f"[PROFILE] Selected: ScannedProfile "
                    f"(confidence={confidence:.2f}, modality={modality.value})"
                )
                return ScannedProfile()
            else:
                logger.info(
                    f"[PROFILE] Selected: ScannedDegradedProfile "
                    f"(confidence={confidence:.2f}, modality={modality.value})"
                )
                return ScannedDegradedProfile()
        else:
            from .document_diagnostic import ContentDomain

            content_domain = diagnostic_report.confidence_profile.detected_domain
            avg_text_per_page = diagnostic_report.physical_check.avg_text_per_page

            # Academic detection
            if (
                content_domain in (ContentDomain.ACADEMIC, ContentDomain.TECHNICAL)
                and avg_text_per_page > 3000
            ):
                logger.info(
                    f"[PROFILE] Selected: AcademicWhitepaperProfile "
                    f"(domain={content_domain.value}, text_density={avg_text_per_page:.0f} chars/page)"
                )
                return AcademicWhitepaperProfile()
            else:
                logger.info(
                    f"[PROFILE] Selected: DigitalMagazineProfile "
                    f"(domain={content_domain.value}, text_density={avg_text_per_page:.0f} chars/page)"
                )
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
