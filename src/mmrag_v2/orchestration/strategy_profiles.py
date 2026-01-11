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
    SCANNED_CLEAN = "scanned_clean"
    SCANNED_LITERATURE = "scanned_literature"  # For scanned books/novels
    SCANNED_MAGAZINE = "scanned_magazine"  # NEW: Scanned editorial content (Combat Aircraft)
    TECHNICAL_MANUAL = "technical_manual"  # NEW: Technical manuals (Firearms.pdf)
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

    # Batch size safety (V2.2: Prevent OOM for high-DPI profiles)
    # Lower batch size for profiles with higher DPI to prevent memory issues
    recommended_batch_size: int = 10  # Default: 10 pages per batch

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
# SCANNED LITERATURE PROFILE
# ============================================================================


class ScannedLiteratureProfile(BaseProfile):
    """
    Profile for scanned literature: novels, fiction, long-form books.

    V16 VLM HALLUCINATIE FIX (2026-01-10):
    ======================================
    CRITICAL: The VLM was hallucinating "revolvers" in Harry Potter because
    it didn't know it was analyzing a CHILDREN'S BOOK. The domain_context
    must be EXPLICIT about the document type to prevent such hallucinations.

    NEW VLM STRATEGY:
    - EXPLICIT domain context: "children's book / fiction / literature"
    - REDUCED sensitivity (0.65 → 0.75 was too aggressive)
    - STRICT warnings against modern/violent interpretations

    CHARACTERISTICS:
    - Scanned books with 50+ pages (Harry Potter, etc.)
    - Editorial domain with narrative text flow
    - Small decorative illustrations, chapter headings, ornamental elements
    - Page-filling scans but with embedded artistic elements

    VLM STRATEGY:
    - STRICT freedom: Focus on visual elements only
    - INJECT scan hints: Tell VLM to ignore paper artifacts
    - EXPLICIT literature context: Prevent hallucinations
    - Capture small illustrations that add context/atmosphere

    EXTRACTION:
    - LOWER min dimensions (25x25) to catch small illustrations
    - MODERATE sensitivity (0.65) - balanced recall vs false positives
    - Background extraction ENABLED (illustrations are part of scan layer)
    - Shadow extraction enabled for embedded artwork

    This profile optimizes for RAG where small illustrations provide
    important contextual markers (chapter breaks, decorative elements, etc.)
    """

    @property
    def profile_type(self) -> ProfileType:
        return ProfileType.SCANNED_LITERATURE

    @property
    def name(self) -> str:
        return "Scanned Literature/Fiction"

    def get_parameters(self) -> ProfileParameters:
        return ProfileParameters(
            # Extraction - OPTIMIZED FOR SMALL ILLUSTRATIONS
            sensitivity=0.65,  # V16 FIX: Reduced from 0.75 - balance recall vs false positives
            min_image_width=25,  # Catch small chapter illustrations
            min_image_height=25,  # Small ornamental elements
            extract_backgrounds=True,  # CRITICAL: Illustrations in scan layer
            enable_shadow_extraction=True,  # Embedded artwork
            # VLM - STRICT visual-only mode with EXPLICIT literature context
            vlm_freedom=VLMFreedom.STRICT,
            inject_scan_hints=True,  # Tell VLM about scan artifacts
            inject_historical_hints=False,  # Not typically historical
            confidence_threshold=0.75,  # V16 FIX: Raised from 0.7 - higher bar for descriptions
            # Post-processing
            strip_artifacts_from_text=False,
            aggressive_deduplication=False,
            # OCR Hints - Enable for scanned books
            enable_ocr_hints=True,
            ocr_min_confidence=0.5,
            ocr_languages=["en"],
            # Dynamic DPI - Higher for better small illustration capture
            render_dpi=200,  # Balance between quality and performance
        )

    def get_vlm_prompt_config(self) -> VLMPromptConfig:
        return VLMPromptConfig(
            base_hints=[
                "This is a SCANNED BOOK - likely FICTION or CHILDREN'S LITERATURE",
                "Focus on decorative illustrations, chapter headings, and artistic elements",
                "These are BOOK ILLUSTRATIONS - whimsical, artistic, fantastical",
                "OCR handles text content - you handle illustrations and artwork",
            ],
            artifact_hints=[
                "IGNORE paper texture, grain, and minor discoloration",
                "IGNORE page edges, binding artifacts, and scan shadows",
                "Focus on intentional visual elements: illustrations, decorations, symbols",
                "DO NOT interpret abstract shapes as modern objects (guns, electronics, etc.)",
            ],
            domain_context="fiction/literature/children's books",
            freedom_instruction=(
                "LITERATURE VISUAL MODE - ANTI-HALLUCINATION RULES:\n"
                "1. You are analyzing a SCANNED BOOK (fiction/literature)\n"
                "2. Describe decorative illustrations, chapter art, ornamental elements\n"
                "3. AVOID modern/technical interpretations - this is NOT a manual\n"
                "4. If you see ambiguous shapes, describe them as 'decorative elements'\n"
                "5. DO NOT describe weapons, electronics, or modern objects unless UNMISTAKABLY present\n"
                "6. When uncertain, use neutral terms: 'illustration', 'decorative motif', 'artistic element'\n"
                "7. These are BOOK illustrations - expect whimsy, fantasy, artistic abstraction\n"
                "8. NO text interpretation - focus on artwork and design only"
            ),
        )

    def should_use_diagnostic_context(self) -> bool:
        # Literature scans use diagnostic context for artifact handling
        return True


# ============================================================================
# SCANNED MAGAZINE PROFILE (NEW)
# ============================================================================


class ScannedMagazineProfile(BaseProfile):
    """
    Profile for scanned magazines and editorial photo content.

    CRITICAL USE CASE: Combat Aircraft scenario
    ============================================
    When a magazine is scanned, the classifier correctly identifies it as a scan,
    but we MUST preserve magazine-quality processing:
    - Large editorial photos (1928px+ median dimensions)
    - High image density (4+ images/page)
    - Editorial domain content

    Without this profile, scanned magazines fall into scanned_clean which:
    - Uses conservative 50x50 min dimensions (too coarse)
    - Uses 150 DPI (loses photo detail)
    - Uses 0.6 sensitivity (misses subtle editorial elements)

    This profile OVERRIDES scanned_clean when:
    - High image density (0.5+) + Large median_dim (200+) + Scan = True

    VLM STRATEGY:
    - STRICT freedom: Preserve visual fidelity of editorial photos
    - INJECT scan hints: Handle scan artifacts gracefully
    - Editorial domain context: Focus on photo content, not paper quality

    EXTRACTION:
    - KEEP 100x100 min dimensions (filter icons, preserve photos)
    - HIGHER DPI (200) for better photo quality
    - HIGHER sensitivity (0.7) for complete photo capture
    """

    @property
    def profile_type(self) -> ProfileType:
        return ProfileType.SCANNED_MAGAZINE

    @property
    def name(self) -> str:
        return "Scanned Magazine (Editorial Photo-First)"

    def get_parameters(self) -> ProfileParameters:
        return ProfileParameters(
            # Extraction - MAGAZINE-OPTIMIZED for scans
            sensitivity=0.7,  # Higher than scanned_clean (0.6) for photo recall
            min_image_width=100,  # Same as digital magazine - filter small ads
            min_image_height=100,
            extract_backgrounds=True,  # CRITICAL: Editorial photos often in background layer
            enable_shadow_extraction=True,  # Catch embedded magazine photos
            # VLM - STRICT but scan-aware
            vlm_freedom=VLMFreedom.STRICT,  # Preserve visual fidelity
            inject_scan_hints=True,  # Handle scan artifacts gracefully
            inject_historical_hints=False,  # Not historical content
            confidence_threshold=0.8,  # High bar for editorial descriptions
            # Post-processing
            strip_artifacts_from_text=False,  # OCR should handle text well
            aggressive_deduplication=False,  # Preserve all editorial content
            # OCR Hints - Enable for any scanned content
            enable_ocr_hints=True,
            ocr_min_confidence=0.5,
            ocr_languages=["en"],
            # Dynamic DPI - HIGHER for magazine photo quality
            render_dpi=200,  # Compromise between 150 (clean) and 300 (degraded)
        )

    def get_vlm_prompt_config(self) -> VLMPromptConfig:
        return VLMPromptConfig(
            base_hints=[
                "This is a SCANNED MAGAZINE with high-quality editorial photos",
                "Focus on the editorial content: aircraft, vehicles, equipment, people",
                "These are professional photographs - describe subjects and composition",
                "Distinguish editorial photos from advertisements and graphics",
            ],
            artifact_hints=[
                "IGNORE minor scan artifacts, paper texture, and binding shadows",
                "IGNORE dust specks and scan lines",
                "The content is professional photography - focus on SUBJECTS not paper quality",
            ],
            domain_context="editorial/magazine (scanned)",
            freedom_instruction=(
                "SCANNED MAGAZINE MODE:\n"
                "1. Describe editorial photos with professional precision\n"
                "2. Focus on subjects: aircraft, vehicles, people, equipment\n"
                "3. Note composition and context of photographs\n"
                "4. IGNORE scan artifacts - these are high-quality magazine photos\n"
                "5. Distinguish between editorial content and advertisements\n"
                "6. NO meta-language about paper or scan quality"
            ),
        )

    def should_use_diagnostic_context(self) -> bool:
        # Scanned magazines need diagnostic context for artifact handling
        return True


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
        ProfileType.SCANNED_CLEAN: ScannedCleanProfile,
        ProfileType.SCANNED_LITERATURE: ScannedLiteratureProfile,
        ProfileType.SCANNED_MAGAZINE: ScannedMagazineProfile,  # NEW
        ProfileType.TECHNICAL_MANUAL: TechnicalManualProfile,  # NEW
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
                ClassifierProfileType.SCANNED_LITERATURE: ScannedLiteratureProfile,
                ClassifierProfileType.SCANNED_CLEAN: ScannedCleanProfile,
                ClassifierProfileType.SCANNED_DEGRADED: ScannedDegradedProfile,
                ClassifierProfileType.SCANNED_MAGAZINE: ScannedMagazineProfile,  # NEW
                ClassifierProfileType.TECHNICAL_MANUAL: TechnicalManualProfile,  # NEW
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
            from .document_diagnostic import ContentDomain

            content_domain = diagnostic_report.confidence_profile.detected_domain
            total_pages = diagnostic_report.physical_check.total_pages

            # Literature detection
            if content_domain == ContentDomain.EDITORIAL and total_pages > 50:
                logger.info(
                    f"[PROFILE] Selected: ScannedLiteratureProfile "
                    f"(editorial domain + {total_pages} pages + scan → literature/fiction)"
                )
                return ScannedLiteratureProfile()

            # Confidence-based selection
            if confidence >= 0.70:
                logger.info(
                    f"[PROFILE] Selected: ScannedCleanProfile "
                    f"(confidence={confidence:.2f}, modality={modality.value})"
                )
                return ScannedCleanProfile()
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
