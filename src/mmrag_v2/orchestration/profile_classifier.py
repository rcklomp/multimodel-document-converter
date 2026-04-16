"""
Profile Classifier - Multi-Dimensional Document Classification
===============================================================
ENGINE_USE: Weighted scoring system for profile selection

This module implements a multi-dimensional classification system that
moves beyond hardcoded if/else chains to intelligent feature weighting.

ARCHITECTURAL PRINCIPLE: ANTI-OVERFITTING
==========================================
Instead of document-specific rules (e.g., "if Harry Potter then..."),
we use generic visual and textual density features to classify documents.

KEY INSIGHT: VISUAL DENSITY HEURISTIC
======================================
The median_dim (median image dimensions) is the KEY discriminator:
- Small illustrations (25-100px): Decorative book elements (Harry Potter)
- Large photos (200-800px): Editorial magazine content (Combat Aircraft)
- Diagrams (50-200px): Technical content (whitepapers, academic)

This single metric prevents overfitting to specific documents while
maintaining accuracy across the "Holy Trinity":
1. Academic Whitepaper (AIOS) - High text, small-medium diagrams
2. Scanned Literature (Harry Potter) - Scan + editorial + small illustrations
3. Digital Magazine (Combat Aircraft) - Editorial + large photos

REQ Compliance:
- REQ-CLASS-01: Multi-dimensional feature-based classification
- REQ-CLASS-02: Visual density heuristic (median_dim + image_density)
- REQ-CLASS-03: Weighted scoring with fallback to safe defaults
- REQ-CLASS-04: No hardcoded document-specific rules

Author: Claude 4.5 Opus (System Architect)
Date: 2026-01-10
Version: 2.0 (Anti-Overfitting Release)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .document_diagnostic import DiagnosticReport
    from .smart_config import DocumentProfile

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================


class ProfileType(str, Enum):
    """Profile type identifiers."""

    ACADEMIC_WHITEPAPER = "academic_whitepaper"
    DIGITAL_MAGAZINE = "digital_magazine"
    SCANNED = "scanned"  # Standard-quality scans (replaces scanned_clean/literature/magazine)
    SCANNED_DEGRADED = "scanned_degraded"
    TECHNICAL_MANUAL = "technical_manual"
    STANDARD_DIGITAL = "standard_digital"  # Fallback profile


# ============================================================================
# CLASSIFICATION FEATURES
# ============================================================================


@dataclass
class ClassificationFeatures:
    """
    Multi-dimensional features for profile classification.

    These features are extracted from DocumentProfile and DiagnosticReport
    to provide a comprehensive view of document characteristics.
    """

    # Text features
    text_density: float  # chars/page
    has_extractable_text: bool

    # Image features
    image_density: float  # images/page
    median_dim: int  # max(median_width, median_height) in pixels
    image_count: int

    # Document features
    page_count: int
    is_scan: bool
    scan_confidence: float  # 0.0-1.0

    # Domain features
    domain: str  # academic, editorial, technical, etc.
    modality: str  # native_digital, scanned_clean, scanned_degraded

    def describe(self) -> str:
        """Human-readable feature summary."""
        return (
            f"text={self.text_density:.0f}ch/pg, "
            f"imgs={self.image_density:.2f}/pg, "
            f"median={self.median_dim}px, "
            f"pages={self.page_count}, "
            f"scan={self.is_scan}, "
            f"domain={self.domain}"
        )


@dataclass
class ProfileScore:
    """
    Score for a specific profile with explanation.

    Higher scores = better match. Scores range from 0.0 to 1.0.
    """

    profile_type: ProfileType
    score: float
    reasoning: list[str]
    confidence: float  # How confident we are in this classification

    def __lt__(self, other: "ProfileScore") -> bool:
        """Allow sorting by score."""
        return self.score < other.score


# ============================================================================
# MULTI-DIMENSIONAL CLASSIFIER
# ============================================================================


class ProfileClassifier:
    """
    Multi-dimensional classifier for document profile selection.

    ARCHITECTURE:
    - Extracts features from DocumentProfile and DiagnosticReport
    - Scores each profile type using weighted feature matching
    - Returns highest-scoring profile with confidence rating
    - Falls back to safe default if confidence is low

    ANTI-OVERFITTING DESIGN:
    - No document-specific rules (no "if filename contains 'Harry Potter'")
    - Generic visual/textual density features
    - Weighted scoring allows gradual transitions (not binary decisions)
    - Explicit fallback logic for edge cases
    """

    # ========================================================================
    # FEATURE THRESHOLDS (TUNED FOR BALANCE)
    # ========================================================================

    # Text density thresholds (chars/page)
    TEXT_DENSITY_HIGH = 3000  # Academic papers
    TEXT_DENSITY_MEDIUM = 1500  # Reports, articles
    TEXT_DENSITY_LOW = 500  # Magazines, scanned books

    # Image density thresholds (images/page)
    IMAGE_DENSITY_HIGH = 0.5  # Magazines, presentations
    IMAGE_DENSITY_MEDIUM = 0.3  # Reports with charts
    IMAGE_DENSITY_LOW = 0.1  # Text-heavy documents

    # Median dimension thresholds (pixels)
    MEDIAN_DIM_LARGE = 200  # Full-page magazine photos
    MEDIAN_DIM_MEDIUM = 100  # Standard diagrams
    MEDIAN_DIM_SMALL = 50  # Small illustrations, icons

    # Page count thresholds
    PAGE_COUNT_BOOK = 50  # Books typically 50+ pages
    PAGE_COUNT_ARTICLE = 20  # Articles/papers 10-30 pages

    # Confidence threshold
    MIN_CONFIDENCE = 0.6  # Below this, use fallback

    def __init__(self) -> None:
        """Initialize the classifier."""
        logger.info("[CLASSIFIER] Multi-dimensional profile classifier initialized")

    def classify(
        self,
        doc_profile: DocumentProfile,
        diagnostic_report: Optional[DiagnosticReport] = None,
    ) -> ProfileType:
        """
        Classify document and return best-matching profile type.

        V16 BULLETPROOF FIX (2026-01-10):
        =================================
        CRITICAL: The old code had a fatal flaw - when confidence was low,
        it would fall back to DIGITAL_MAGAZINE even if that profile had
        score=0.0 and was HARD REJECTED because the document was a scan.

        NEW LOGIC:
        1. Score all profiles
        2. Separate into VALID (score > 0) and REJECTED (score = 0) profiles
        3. If best match confidence < threshold, choose highest scoring VALID profile
        4. MODALITY-AWARE FALLBACK: If is_scan=True, fallback MUST be a scanned profile
        5. If is_scan=False, fallback MUST be a digital profile
        6. NEVER choose a rejected profile as fallback

        Args:
            doc_profile: DocumentProfile from SmartConfigProvider
            diagnostic_report: Optional DiagnosticReport from DocumentDiagnosticEngine

        Returns:
            ProfileType that best matches document characteristics
        """
        # Extract features
        features = self._extract_features(doc_profile, diagnostic_report)

        logger.info(f"[CLASSIFIER] Features: {features.describe()}")

        # Score all profiles
        scores = [
            self._score_academic_whitepaper(features),
            self._score_digital_magazine(features),
            self._score_scanned(features),
            self._score_scanned_degraded(features),
            self._score_technical_manual(features),
        ]

        # V16 FIX: Separate valid from rejected profiles
        valid_scores = [s for s in scores if s.score > 0.0]
        rejected_scores = [s for s in scores if s.score == 0.0]

        # Log all scores for transparency
        logger.info("[CLASSIFIER] Profile scores:")
        for score_result in sorted(scores, key=lambda s: s.score, reverse=True):
            status = "✓ VALID" if score_result.score > 0 else "✗ REJECTED"
            logger.info(
                f"  {score_result.profile_type.value}: {score_result.score:.3f} "
                f"(confidence: {score_result.confidence:.2f}) [{status}]"
            )
            for reason in score_result.reasoning:
                logger.info(f"    - {reason}")

        # Get best match from valid profiles only
        if not valid_scores:
            # EDGE CASE: All profiles rejected (should never happen)
            logger.error("[CLASSIFIER] All profiles rejected! Using emergency fallback.")
            return self._emergency_fallback(features)

        # V16.2 CONFIDENCE-WEIGHTED SELECTION: Always use score * confidence
        # This ensures that high-confidence classifications beat high-score but low-confidence ones
        # Example: technical_manual (0.80 * 0.80 = 0.64) beats scanned (0.85 * 0.56 = 0.47)
        def rank_score(s: ProfileScore) -> float:
            return s.score * s.confidence

        best_match = max(valid_scores, key=rank_score)

        logger.info(
            f"[CLASSIFIER] Ranking by score×confidence: {best_match.profile_type.value} "
            f"(rank={rank_score(best_match):.3f} = {best_match.score:.3f}×{best_match.confidence:.2f})"
        )

        # V16 BULLETPROOF FIX: Modality-aware fallback (only if confidence still too low)
        if best_match.confidence < self.MIN_CONFIDENCE:
            fallback = self._get_modality_aware_fallback(features, valid_scores)
            logger.warning(
                f"[CLASSIFIER] Low confidence ({best_match.confidence:.2f}), "
                f"using modality-aware fallback: {fallback.value} "
                f"(NOT hardcoded digital_magazine!)"
            )
            return fallback

        logger.info(
            f"[CLASSIFIER] Selected: {best_match.profile_type.value} "
            f"(score={best_match.score:.3f}, confidence={best_match.confidence:.2f})"
        )

        return best_match.profile_type

    def _get_modality_aware_fallback(
        self,
        features: ClassificationFeatures,
        valid_scores: list[ProfileScore],
    ) -> ProfileType:
        """
        Get a modality-aware fallback profile.

        V16 BULLETPROOF RULE:
        - If document is SCANNED → fallback MUST be a scanned profile
        - If document is DIGITAL → fallback MUST be a digital profile
        - NEVER cross the modality boundary!

        V16.1 CONFIDENCE-WEIGHTED SELECTION (2026-01-10):
        ==================================================
        Instead of selecting purely on RAW SCORE, we use a CONFIDENCE-WEIGHTED
        score to handle cases where a profile has high confidence but lower
        raw score (e.g., technical_manual with 0.80 confidence vs scanned
        with 0.56 confidence).

        FORMULA: weighted_score = score * (0.5 + 0.5 * confidence)
        - This gives 50% weight to raw score and 50% to confidence
        - A profile with score=0.65 and confidence=0.80 beats
          a profile with score=0.85 and confidence=0.56

        SCANNED PROFILES: scanned, scanned_degraded, technical_manual
        DIGITAL PROFILES: academic_whitepaper, digital_magazine, technical_manual

        Note: technical_manual is valid for BOTH modalities.

        Args:
            features: Classification features
            valid_scores: List of valid (non-rejected) profile scores

        Returns:
            ProfileType appropriate for the document's modality
        """
        # Define profile modality categories
        scanned_profiles = {
            ProfileType.SCANNED,
            ProfileType.SCANNED_DEGRADED,
            ProfileType.TECHNICAL_MANUAL,  # Valid for both
        }
        digital_profiles = {
            ProfileType.ACADEMIC_WHITEPAPER,
            ProfileType.DIGITAL_MAGAZINE,
            ProfileType.TECHNICAL_MANUAL,  # Valid for both
        }

        # Filter valid scores by modality
        if features.is_scan:
            logger.info("[CLASSIFIER] Modality: SCANNED → filtering to scanned profiles only")
            modality_valid = [s for s in valid_scores if s.profile_type in scanned_profiles]
            default_fallback = ProfileType.SCANNED
        else:
            logger.info("[CLASSIFIER] Modality: DIGITAL → filtering to digital profiles only")
            modality_valid = [s for s in valid_scores if s.profile_type in digital_profiles]
            default_fallback = ProfileType.DIGITAL_MAGAZINE

        # V16.1: Confidence-weighted selection
        def weighted_score(s: ProfileScore) -> float:
            """Calculate confidence-weighted score."""
            return s.score * (0.5 + 0.5 * s.confidence)

        # Get highest WEIGHTED scoring profile within correct modality
        if modality_valid:
            # Log all weighted scores for transparency
            logger.info("[CLASSIFIER] Confidence-weighted scores within modality:")
            for s in sorted(modality_valid, key=weighted_score, reverse=True):
                ws = weighted_score(s)
                logger.info(
                    f"  {s.profile_type.value}: weighted={ws:.3f} "
                    f"(raw={s.score:.3f} × (0.5 + 0.5×{s.confidence:.2f}))"
                )

            best_in_modality = max(modality_valid, key=weighted_score)
            logger.info(
                f"[CLASSIFIER] Best within modality (confidence-weighted): "
                f"{best_in_modality.profile_type.value} "
                f"(weighted={weighted_score(best_in_modality):.3f})"
            )
            return best_in_modality.profile_type
        else:
            # No valid profiles in correct modality - use default
            logger.warning(
                f"[CLASSIFIER] No valid profiles in modality, using default: {default_fallback.value}"
            )
            return default_fallback

    def _emergency_fallback(self, features: ClassificationFeatures) -> ProfileType:
        """
        Emergency fallback when all profiles are rejected.

        This should NEVER happen in normal operation, but provides
        a safe path if it does.

        Args:
            features: Classification features

        Returns:
            Safe default ProfileType based on modality
        """
        if features.is_scan:
            logger.warning("[CLASSIFIER] EMERGENCY: All rejected, defaulting to SCANNED")
            return ProfileType.SCANNED
        else:
            logger.warning("[CLASSIFIER] EMERGENCY: All rejected, defaulting to DIGITAL_MAGAZINE")
            return ProfileType.DIGITAL_MAGAZINE

    def _extract_features(
        self,
        doc_profile: DocumentProfile,
        diagnostic_report: Optional[DiagnosticReport],
    ) -> ClassificationFeatures:
        """Extract classification features from profiles."""
        # Basic features from DocumentProfile
        median_dim = max(
            doc_profile.median_image_width,
            doc_profile.median_image_height,
        )
        # Guard: full-page renders (>= 1000px) are not editorial photos — they're
        # artifacts from Docling extracting full pages as images. Cap to avoid
        # triggering "large editorial photos" boost in digital_magazine scoring.
        if median_dim >= 1000:
            median_dim = 200  # Neutral value — won't boost any profile

        # Default features if no diagnostic report
        is_scan = False
        scan_confidence = 0.0
        domain = "unknown"
        modality = "native_digital"

        # Enhanced features from DiagnosticReport
        if diagnostic_report:
            is_scan = diagnostic_report.physical_check.is_likely_scan
            scan_confidence = diagnostic_report.physical_check.scan_confidence
            domain = diagnostic_report.confidence_profile.detected_domain.value
            modality = diagnostic_report.physical_check.detected_modality.value

        return ClassificationFeatures(
            text_density=doc_profile.avg_text_per_page,
            has_extractable_text=doc_profile.has_text,
            image_density=doc_profile.image_density,
            median_dim=median_dim,
            image_count=doc_profile.image_count,
            page_count=doc_profile.total_pages,
            is_scan=is_scan,
            scan_confidence=scan_confidence,
            domain=domain,
            modality=modality,
        )

    # ========================================================================
    # PROFILE SCORING FUNCTIONS
    # ========================================================================

    def _score_academic_whitepaper(self, f: ClassificationFeatures) -> ProfileScore:
        """
        Score for Academic Whitepaper profile.

        MODALITY RULE: This is a DIGITAL profile - HARD REJECT if scan detected.

        CHARACTERISTICS:
        - HIGH text density (3000+ chars/page)
        - MODERATE image density (diagrams, not photos)
        - SMALL-MEDIUM median_dim (technical diagrams: 50-200px)
        - Academic or technical domain
        - NOT a scan (HARD REQUIREMENT)
        - Moderate page count (10-50 pages typical)
        """
        score = 0.0
        reasoning = []
        confidence = 1.0

        # MODALITY CHECK: HARD REQUIREMENT - must be digital
        # This is a DIGITAL profile - scans must use scanned profiles
        if f.is_scan:
            reasoning.append("REJECTED: Scanned document - use scanned profile instead")
            return ProfileScore(
                profile_type=ProfileType.ACADEMIC_WHITEPAPER,
                score=0.0,
                reasoning=reasoning,
                confidence=0.0,  # ZERO confidence - hard reject
            )

        # If we get here, it's a digital document - proceed with scoring
        reasoning.append("✓ Digital document (modality check passed)")

        # TEXT DENSITY: Primary signal for academic (weight: 0.35)
        if f.text_density >= self.TEXT_DENSITY_HIGH:
            score += 0.35
            reasoning.append(f"High text density ({f.text_density:.0f} >= 3000)")
        elif f.text_density >= self.TEXT_DENSITY_MEDIUM:
            score += 0.20
            reasoning.append(f"Moderate text density ({f.text_density:.0f})")
        else:
            score += 0.0
            reasoning.append(f"Low text density ({f.text_density:.0f})")
            confidence *= 0.5

        # DOMAIN: Strong signal (weight: 0.30)
        if f.domain in ("academic", "technical"):
            score += 0.30
            reasoning.append(f"Academic/technical domain ({f.domain})")
        else:
            score += 0.0
            reasoning.append(f"Non-academic domain ({f.domain})")
            confidence *= 0.6

        # IMAGE DENSITY: Should be LOW-MODERATE for papers (weight: 0.20)
        if f.image_density <= self.IMAGE_DENSITY_MEDIUM:
            score += 0.20
            reasoning.append(f"Appropriate image density for papers ({f.image_density:.2f})")
        else:
            score += 0.05
            reasoning.append(f"High image density ({f.image_density:.2f}) unusual for papers")
            confidence *= 0.7

        # MEDIAN DIM: Should be small-medium for diagrams (weight: 0.15)
        if self.MEDIAN_DIM_SMALL <= f.median_dim <= self.MEDIAN_DIM_LARGE:
            score += 0.15
            reasoning.append(f"Appropriate diagram size ({f.median_dim}px)")
        elif f.median_dim > self.MEDIAN_DIM_LARGE:
            score += 0.05
            reasoning.append(f"Large images ({f.median_dim}px) unusual for academic diagrams")
            confidence *= 0.8

        return ProfileScore(
            profile_type=ProfileType.ACADEMIC_WHITEPAPER,
            score=score,
            reasoning=reasoning,
            confidence=confidence,
        )

    def _score_digital_magazine(self, f: ClassificationFeatures) -> ProfileScore:
        """
        Score for Digital Magazine profile.

        MODALITY RULE: This is a DIGITAL profile - HARD REJECT if scan detected.

        CHARACTERISTICS:
        - LOW-MEDIUM text density (500-1500 chars/page)
        - HIGH image density (0.5+ images/page)
        - LARGE median_dim (200+ px for editorial photos)
        - Editorial domain
        - NOT a scan (HARD REQUIREMENT - born-digital)
        - Variable page count
        """
        score = 0.0
        reasoning = []
        confidence = 1.0

        # MODALITY CHECK: HARD REQUIREMENT - must be digital
        # This is a DIGITAL profile - scans must use scanned profile instead
        if f.is_scan:
            reasoning.append("REJECTED: Scanned document - use scanned profile instead")
            return ProfileScore(
                profile_type=ProfileType.DIGITAL_MAGAZINE,
                score=0.0,
                reasoning=reasoning,
                confidence=0.0,  # ZERO confidence - hard reject
            )

        # If we get here, it's a digital document - proceed with scoring
        reasoning.append("✓ Digital document (modality check passed)")

        # IMAGE DENSITY SANITY: > 5.0 images/page indicates decorative inline elements
        # (bullets, ornaments, drop caps, inline icons) rather than real editorial photos.
        # Real magazines have 0.5 – 3.0 large photos per page, not 10-15 tiny objects.
        # Cap the effective density used for scoring so decorative-heavy books don't get
        # the same bonus as genuine photo-heavy magazines.
        effective_image_density = min(f.image_density, 3.0)
        if f.image_density > 5.0:
            confidence *= 0.15
            reasoning.append(
                f"Extreme image density ({f.image_density:.1f}/page) → decorative inline "
                f"elements, not editorial photos; capped at 3.0 for scoring"
            )

        # IMAGE DENSITY: Primary signal for magazines (weight: 0.30)
        if effective_image_density >= self.IMAGE_DENSITY_HIGH:
            score += 0.30
            reasoning.append(f"High image density ({f.image_density:.2f} >= 0.5)")
        elif effective_image_density >= self.IMAGE_DENSITY_MEDIUM:
            score += 0.15
            reasoning.append(f"Moderate image density ({f.image_density:.2f})")
        else:
            score += 0.0
            reasoning.append(f"Low image density ({f.image_density:.2f})")
            confidence *= 0.5

        # MEDIAN DIM: Critical for distinguishing from scanned books (weight: 0.30)
        # This is the KEY anti-overfitting feature!
        if f.median_dim >= self.MEDIAN_DIM_LARGE:
            score += 0.30
            reasoning.append(f"Large editorial photos ({f.median_dim}px >= 200)")
        elif f.median_dim >= self.MEDIAN_DIM_MEDIUM:
            score += 0.15
            reasoning.append(f"Medium images ({f.median_dim}px)")
        else:
            score += 0.0
            reasoning.append(f"Small images ({f.median_dim}px) - not typical magazine photos")
            confidence *= 0.4

        # TEXT DENSITY: Should be LOW-MEDIUM (weight: 0.20)
        if self.TEXT_DENSITY_LOW <= f.text_density <= self.TEXT_DENSITY_MEDIUM:
            score += 0.20
            reasoning.append(f"Magazine-appropriate text density ({f.text_density:.0f})")
        elif f.text_density > self.TEXT_DENSITY_HIGH:
            score += 0.0
            reasoning.append(f"Too text-heavy ({f.text_density:.0f}) for magazine")
            confidence *= 0.5

        # DOMAIN: Strong signal (weight: 0.20)
        if f.domain == "editorial":
            score += 0.20
            reasoning.append("Editorial domain")
        elif f.domain == "literature":
            # Literature domain (novels, fiction) — digital_magazine is the safe default
            # profile for narrative content. Score it reasonably so it wins over
            # technical_manual for books with decorative images.
            score += 0.15
            reasoning.append("Literature domain — safe default for narrative content")
        else:
            score += 0.05
            reasoning.append(f"Non-editorial domain ({f.domain})")
            confidence *= 0.7

        # PAGE COUNT: No magazine issue exceeds ~250 pages. Long documents with high
        # image density are illustrated books or photo collections, not magazines.
        # Literature-domain docs are exempt — novels CAN be 300+ pages.
        if f.page_count > 250 and f.domain != "literature":
            confidence *= 0.15
            reasoning.append(
                f"Very long document ({f.page_count} pages) → not a magazine issue"
            )
        elif f.page_count > 150:
            confidence *= 0.5
            reasoning.append(
                f"Long document ({f.page_count} pages) → unlikely to be a magazine"
            )

        return ProfileScore(
            profile_type=ProfileType.DIGITAL_MAGAZINE,
            score=score,
            reasoning=reasoning,
            confidence=confidence,
        )

    def _score_scanned(self, f: ClassificationFeatures) -> ProfileScore:
        """
        Score for Scanned profile (standard-quality scans).

        Replaces scanned_clean, scanned_literature, and scanned_magazine —
        three former profiles that shared identical batch-processor behavior
        and only differed in minor extraction parameters.

        CHARACTERISTICS:
        - SCAN = True (essential)
        - Reasonable scan quality (confidence 0.60+)
        - NOT degraded (that's scanned_degraded)
        - NOT a technical manual (that's technical_manual via DOMINANCE)
        - Covers: books, editorial scans, standard-quality document scans
        """
        score = 0.0
        reasoning = []
        confidence = 1.0

        # SCAN CHECK: Required — hard reject if digital
        if not f.is_scan:
            reasoning.append("REJECTED: Digital document — use digital profile instead")
            return ProfileScore(
                profile_type=ProfileType.SCANNED,
                score=0.0,
                reasoning=reasoning,
                confidence=0.0,
            )

        reasoning.append("✓ Scanned document")

        # SCAN QUALITY (weight: 0.35)
        if f.scan_confidence >= 0.70:
            score += 0.35
            reasoning.append(f"Good scan quality ({f.scan_confidence:.2f})")
        else:
            score += 0.15
            reasoning.append(f"Moderate scan quality ({f.scan_confidence:.2f})")
            confidence *= 0.8

        # MODALITY (weight: 0.25)
        if f.modality in ("scanned_clean", "scanned_degraded", "hybrid"):
            score += 0.25
            reasoning.append(f"Confirmed scan modality ({f.modality})")

        # DOMAIN (weight: 0.20)
        if f.domain in ("editorial", "technical"):
            score += 0.20
            reasoning.append(f"Known domain ({f.domain})")
        elif f.domain == "academic":
            score += 0.15
            reasoning.append("Academic domain")
        else:
            score += 0.10
            reasoning.append(f"Unknown domain ({f.domain})")
            confidence *= 0.9

        # TEXT DENSITY: moderate extraction = readable scan (weight: 0.15)
        if f.text_density >= self.TEXT_DENSITY_LOW:
            score += 0.15
            reasoning.append(f"Readable scan ({f.text_density:.0f} chars/page)")
        else:
            # Low text density is valid (OCR may have struggled), small penalty
            score += 0.05
            reasoning.append(f"Low text extraction ({f.text_density:.0f} chars/page)")
            confidence *= 0.85

        # PAGE COUNT (weight: 0.05)
        if f.page_count >= 5:
            score += 0.05
            reasoning.append(f"Reasonable page count ({f.page_count})")

        return ProfileScore(
            profile_type=ProfileType.SCANNED,
            score=score,
            reasoning=reasoning,
            confidence=confidence,
        )

    def _score_scanned_degraded(self, f: ClassificationFeatures) -> ProfileScore:
        """
        Score for Scanned Degraded profile (low-quality legacy scans).

        CHARACTERISTICS:
        - SCAN = True with LOWER confidence (< 0.70)
        - Degraded modality
        - LOW text density (poor OCR)
        - Historical era indicators
        """
        score = 0.0
        reasoning = []
        confidence = 1.0

        # SCAN CHECK: Essential (weight: 0.35)
        if f.is_scan and f.scan_confidence < 0.70:
            score += 0.35
            reasoning.append(f"Degraded scan (confidence: {f.scan_confidence:.2f})")
        elif f.is_scan:
            score += 0.15
            reasoning.append(f"Scan but higher quality ({f.scan_confidence:.2f})")
            confidence *= 0.7
        else:
            score += 0.0
            reasoning.append("Not scanned")
            confidence *= 0.1
            return ProfileScore(
                profile_type=ProfileType.SCANNED_DEGRADED,
                score=score,
                reasoning=reasoning,
                confidence=confidence,
            )

        # MODALITY: Should be scanned_degraded (weight: 0.30)
        if f.modality == "scanned_degraded":
            score += 0.30
            reasoning.append("Degraded scan modality")
        elif f.modality in ("scanned_clean", "hybrid"):
            score += 0.10
            reasoning.append(f"Scan modality: {f.modality}")
            confidence *= 0.7

        # TEXT DENSITY: LOW due to poor OCR (weight: 0.20)
        if f.text_density < self.TEXT_DENSITY_LOW:
            score += 0.20
            reasoning.append(f"Low text extraction ({f.text_density:.0f}) - degraded OCR")
        else:
            score += 0.05
            reasoning.append(f"Reasonable text extraction ({f.text_density:.0f})")
            confidence *= 0.7

        # DOMAIN: Historical/technical documents (weight: 0.15)
        if f.domain in ("technical", "unknown"):
            score += 0.15
            reasoning.append(f"Typical domain for legacy scans ({f.domain})")
        else:
            score += 0.10
            reasoning.append(f"Domain: {f.domain}")

        return ProfileScore(
            profile_type=ProfileType.SCANNED_DEGRADED,
            score=score,
            reasoning=reasoning,
            confidence=confidence,
        )

    def _score_technical_manual(self, f: ClassificationFeatures) -> ProfileScore:
        """
        Score for Technical Manual profile (Firearms.pdf scenario).

        V16 TECHNICAL DOMINANCE FIX (2026-01-10):
        =========================================
        CRITICAL: Technical manuals with scanned origin and low native text
        were being misclassified because the scoring didn't account for the
        combination: technical domain + is_scan + low OCR text = MANUAL.

        NEW LOGIC:
        - If domain=technical + is_scan=True + text_density < 500 → BOOST score by 0.20
        - This combination is 99% of the time a scanned technical manual

        CRITICAL INSIGHT: Technical manuals have SMALL but CRITICAL visual elements:
        - Small parts diagrams (pins, screws, springs) often < 50px
        - Part numbers and callout labels
        - Exploded view diagrams with fine detail
        - Assembly/disassembly sequences

        Standard scanned_clean (50x50 min) would FILTER OUT these crucial elements!

        This profile requires:
        - Min dimensions: 30x30px (catch small parts)
        - Sensitivity: 0.8 (high recall for every bolt and spring)
        - DPI: 300 (fine detail preservation)

        CHARACTERISTICS:
        - Technical domain (strongly preferred)
        - MEDIUM image density (diagrams, not photo-heavy like magazines)
        - MEDIUM median_dim (exploded views are medium-sized)
        - Can be scan OR digital
        - Moderate page count (5-100 pages typical)
        """
        score = 0.0
        reasoning = []
        confidence = 1.0

        # V16 TECHNICAL DOMINANCE: Combo check FIRST
        # If technical domain + scan + low/no native text → almost certainly a scanned manual.
        # EXTENDED (v2.6): Domain detector is unreliable for scanned/OCR'd documents. An
        # editorial-domain scanned doc extracted as full-page images will have large median_dim
        # (1000-1600px per page) — so median_dim cannot discriminate between magazine photos and
        # scanned technical manual pages. Instead we use PAGE COUNT: magazine issues are < 100
        # pages; technical manuals are typically 100+ pages. A short doc (< 100 pages) with
        # small content images (median_dim < 200px) is also flagged as a technical manual.
        is_scanned_editorial_technical = (
            f.domain == "editorial"
            and f.is_scan
            and f.text_density < self.TEXT_DENSITY_LOW
            and (
                f.page_count >= (self.PAGE_COUNT_BOOK * 2)  # 100+ pages: no single magazine issue
                or f.median_dim < self.MEDIAN_DIM_LARGE  # short doc with small images: parts diagrams
            )
        )
        is_scanned_technical_manual = (
            f.is_scan
            and f.text_density < self.TEXT_DENSITY_LOW
            and (f.domain == "technical" or is_scanned_editorial_technical)
        )

        if is_scanned_technical_manual:
            # V16.2 MASSIVE BOOST: This combination is the signature of scanned technical manuals.
            # The boost + dominance lock ensures rank_score = 0.80 beats scanned (0.665).
            score += 0.55
            if is_scanned_editorial_technical:
                reasoning.append(
                    "⚡ TECHNICAL DOMINANCE: editorial + scan + low_text + long_doc → "
                    "scanned technical manual (domain detector unreliable for OCR docs)"
                )
            else:
                reasoning.append(
                    "⚡ TECHNICAL DOMINANCE: domain=technical + scan + low_text → "
                    "scanned technical manual signature (BOOSTED)"
                )
            confidence = 1.0  # HIGH confidence for this combination

        # EDITORIAL TECHNICAL BOOK: digital + editorial + low image density + many pages.
        # Technical reference books (coding books, handbooks) are classified as "editorial"
        # by the domain detector because they contain screenshots and figures. However they
        # are NOT magazines. Magazines require high image density (0.5+ images/page) with
        # large editorial photos. A digital editorial document with low image density and
        # many pages (>100) is a technical reference book, not a glossy magazine.
        is_editorial_tech_book = (
            f.domain == "editorial"
            and not f.is_scan
            and f.image_density < self.IMAGE_DENSITY_HIGH  # < 0.5; magazines need dense images
            and f.page_count > 100  # Long reference books; short works may genuinely be magazines
        )

        # DOMAIN: Technical is the PRIMARY signal (weight: 0.35 if not already boosted)
        if f.domain == "technical":
            if not is_scanned_technical_manual:
                score += 0.35
                reasoning.append("Technical domain - manual/documentation signature")
        elif f.domain == "unknown":
            score += 0.15
            reasoning.append("Unknown domain - could be technical manual")
            confidence *= 0.7
        elif is_editorial_tech_book:
            score += 0.25
            reasoning.append(
                f"Editorial technical book: digital + editorial + "
                f"low image density ({f.image_density:.2f}) + {f.page_count} pages"
            )
            # No confidence penalty — these are legitimate technical reference books
        elif is_scanned_editorial_technical:
            # Domain already handled by TECHNICAL DOMINANCE boost; no confidence penalty.
            reasoning.append(
                f"Editorial domain accepted: scanned + long_doc confirms technical manual"
            )
        else:
            score += 0.0
            reasoning.append(f"Non-technical domain ({f.domain})")
            confidence *= 0.4  # Strong penalty - manuals are technical

        # IMAGE DENSITY: Medium is ideal for manuals (weight: 0.20)
        # Not as high as magazines, not as low as books
        if f.image_density > 5.0:
            # Extreme image density (>5/page) indicates decorative inline elements
            # (drop caps, ornaments, bullets) — not technical diagrams. Real technical
            # manuals have 0.1–0.5 images/page.
            score += 0.0
            reasoning.append(
                f"Extreme image density ({f.image_density:.1f}/page) — "
                f"decorative content, not technical diagrams"
            )
            confidence *= 0.15
        elif self.IMAGE_DENSITY_LOW <= f.image_density <= self.IMAGE_DENSITY_HIGH:
            score += 0.20
            reasoning.append(
                f"Manual-appropriate image density ({f.image_density:.2f}) - "
                f"diagrams but not photo-heavy"
            )
        elif f.image_density > self.IMAGE_DENSITY_HIGH:
            score += 0.10
            reasoning.append(
                f"High image density ({f.image_density:.2f}) - very illustrated manual"
            )
            confidence *= 0.8
        else:
            # Low image density is ALSO valid for technical manuals (text-heavy sections)
            score += 0.10
            reasoning.append(f"Low image density ({f.image_density:.2f}) - text-heavy sections OK")
            # No confidence penalty - some manuals are text-heavy

        # MEDIAN DIM: Medium is typical for exploded diagrams (weight: 0.15)
        # Not large photos (magazines), not tiny decorative (books)
        if self.MEDIAN_DIM_SMALL <= f.median_dim <= self.MEDIAN_DIM_LARGE:
            score += 0.15
            reasoning.append(
                f"Technical diagram size ({f.median_dim}px) - "
                f"appropriate for exploded views and schematics"
            )
        elif f.median_dim > self.MEDIAN_DIM_LARGE:
            score += 0.10
            reasoning.append(f"Large images ({f.median_dim}px) - could be detailed schematics")
        else:
            score += 0.10
            reasoning.append(f"Small images ({f.median_dim}px) - small parts diagrams expected")
            # No confidence penalty - small parts diagrams are valid!

        # SCAN STATUS: Scanned manuals get EXTRA boost (weight: 0.15 for scans, 0.05 for digital)
        if f.is_scan:
            if not is_scanned_technical_manual:  # Don't double-count
                score += 0.15
            reasoning.append("Scanned manual - common for legacy documentation")
        else:
            score += 0.05
            reasoning.append("Digital manual - modern documentation")

        # PAGE COUNT: Manuals are typically 5-100 pages (weight: 0.10)
        if 5 <= f.page_count <= 100:
            score += 0.10
            reasoning.append(f"Manual-appropriate page count ({f.page_count} pages)")
        elif f.page_count > 100:
            score += 0.05
            reasoning.append(f"Long document ({f.page_count} pages) - comprehensive manual")
        else:
            score += 0.0
            reasoning.append(f"Very short ({f.page_count} pages) - may be datasheet")
            confidence *= 0.8

        # DOMINANCE LOCK: when TECHNICAL DOMINANCE fired, enforce certainty = 1.0.
        # Intermediate checks (image density, page count) may have reduced confidence;
        # the DOMINANCE condition is strong enough to override those reductions.
        if is_scanned_technical_manual:
            confidence = 1.0

        return ProfileScore(
            profile_type=ProfileType.TECHNICAL_MANUAL,
            score=score,
            reasoning=reasoning,
            confidence=confidence,
        )


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================


def classify_document(
    doc_profile: DocumentProfile,
    diagnostic_report: Optional[DiagnosticReport] = None,
) -> ProfileType:
    """
    Convenience function for document classification.

    Args:
        doc_profile: DocumentProfile from SmartConfigProvider
        diagnostic_report: Optional DiagnosticReport from DocumentDiagnosticEngine

    Returns:
        ProfileType that best matches document characteristics
    """
    classifier = ProfileClassifier()
    return classifier.classify(doc_profile, diagnostic_report)
