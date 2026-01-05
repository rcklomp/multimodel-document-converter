"""
Document Diagnostic Layer - Pre-Processing Intelligence
=========================================================
ENGINE_USE: PyMuPDF (fitz) + VLM for document analysis

This module implements the "Scanned-Document Diagnostic Layer" that runs
BEFORE main conversion to build a confidence profile and detect document
characteristics that inform processing strategy.

ARCHITECTURAL IMPROVEMENTS (Gemini Audit):
1. PhysicalCheck: Heuristic validation for scanned documents
2. Dynamic Prompt Injection: Template-based prompts with context
3. Confidence Scoring: Hallucination threshold system
4. Diagnostic Layer: Pre-flight analysis for robust processing

REQ Compliance:
- REQ-DIAG-01: Pre-analyze first N pages before full conversion
- REQ-DIAG-02: Detect scanned documents via physical heuristics
- REQ-DIAG-03: Build confidence profile for VLM guidance

SRS Section 9.3: Document Diagnostics
"The system SHOULD analyze document characteristics before processing
to optimize extraction strategy and prevent misclassification."

Author: Claude (Architect)
Date: 2025-01-03
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# PhysicalCheck thresholds (REQ-DIAG-02)
SCAN_FILE_SIZE_THRESHOLD_MB: float = 1.0  # Files > 1MB with low text are likely scans
SCAN_TEXT_THRESHOLD_CHARS: int = 100  # Less than 100 chars per page = likely scan
SCAN_IMAGE_RATIO_THRESHOLD: float = 0.8  # >80% page area covered by images = scan

# Diagnostic sampling
DIAGNOSTIC_SAMPLE_PAGES: int = 5  # Analyze first 5 pages
DPI_LOW: int = 72  # Quick analysis
DPI_HIGH: int = 150  # Detailed analysis

# Confidence thresholds
HIGH_CONFIDENCE_THRESHOLD: float = 0.85
LOW_CONFIDENCE_THRESHOLD: float = 0.50


# ============================================================================
# ENUMS
# ============================================================================


class DocumentModality(str, Enum):
    """Detected document modality classification."""

    NATIVE_DIGITAL = "native_digital"  # Born-digital PDF with selectable text
    SCANNED_CLEAN = "scanned_clean"  # Clean scan with good OCR potential
    SCANNED_DEGRADED = "scanned_degraded"  # Aged/damaged scan, artifacts present
    HYBRID = "hybrid"  # Mix of digital and scanned pages
    IMAGE_HEAVY = "image_heavy"  # Digital but dominated by images
    UNKNOWN = "unknown"


class DocumentEra(str, Enum):
    """Estimated document era for prompt context."""

    MODERN = "modern"  # Post-2000, clean digital
    VINTAGE = "vintage"  # 1970-2000, early digital or good scans
    HISTORICAL = "historical"  # Pre-1970, likely degraded scans
    UNKNOWN = "unknown"


class ContentDomain(str, Enum):
    """Detected content domain for semantic context."""

    TECHNICAL = "technical"  # Manuals, specs, engineering docs
    EDITORIAL = "editorial"  # Magazines, newspapers, articles
    ACADEMIC = "academic"  # Research papers, textbooks
    COMMERCIAL = "commercial"  # Marketing, ads, brochures
    UNKNOWN = "unknown"


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class PageDiagnostic:
    """Diagnostic results for a single page."""

    page_number: int
    text_length: int
    text_density: float  # chars per page area
    image_count: int
    image_coverage: float  # 0.0-1.0 of page area
    has_ocr_artifacts: bool
    detected_noise_level: float  # 0.0-1.0
    dominant_colors: List[str] = field(default_factory=list)
    contains_tables: bool = False
    contains_diagrams: bool = False


@dataclass
class PhysicalCheckResult:
    """
    Results from physical/heuristic document analysis.

    REQ-DIAG-02: Detect scans without relying on VLM classification.
    """

    file_size_mb: float
    total_pages: int
    avg_text_per_page: float
    avg_image_coverage: float
    is_likely_scan: bool
    scan_confidence: float  # 0.0-1.0
    detected_modality: DocumentModality
    reasoning: str


@dataclass
class ConfidenceProfile:
    """
    Confidence profile for VLM guidance.

    REQ-DIAG-03: Provides context to prevent hallucinations.
    """

    overall_confidence: float  # 0.0-1.0
    classification_confidence: float
    detected_features: List[str]
    detected_era: DocumentEra
    detected_domain: ContentDomain
    warnings: List[str] = field(default_factory=list)

    def is_low_confidence(self) -> bool:
        """Check if confidence is below threshold."""
        return self.overall_confidence < LOW_CONFIDENCE_THRESHOLD

    def is_high_confidence(self) -> bool:
        """Check if confidence is above threshold."""
        return self.overall_confidence >= HIGH_CONFIDENCE_THRESHOLD


@dataclass
class DiagnosticReport:
    """
    Complete diagnostic report for a document.

    This report informs the processing strategy and VLM prompt construction.
    """

    source_file: str
    physical_check: PhysicalCheckResult
    confidence_profile: ConfidenceProfile
    page_diagnostics: List[PageDiagnostic]
    recommended_strategy: str
    prompt_context: Dict[str, Any] = field(default_factory=dict)

    def should_force_scan_mode(self) -> bool:
        """Determine if document should be processed as scan."""
        return self.physical_check.is_likely_scan

    def get_vlm_context_hints(self) -> List[str]:
        """Get context hints for VLM prompts."""
        hints = []

        if self.physical_check.is_likely_scan:
            hints.append("This is a scanned document - ignore paper artifacts, stains, and grain")

        if self.confidence_profile.detected_era == DocumentEra.HISTORICAL:
            hints.append("Historical document - expect aged appearance, yellowing, foxing")
        elif self.confidence_profile.detected_era == DocumentEra.VINTAGE:
            hints.append("Vintage document - may have dated typography and printing style")

        if self.confidence_profile.detected_domain == ContentDomain.TECHNICAL:
            hints.append("Technical document - focus on diagrams, specifications, procedures")
        elif self.confidence_profile.detected_domain == ContentDomain.EDITORIAL:
            hints.append("Editorial content - distinguish photos from advertisements")

        for warning in self.confidence_profile.warnings:
            hints.append(f"Warning: {warning}")

        return hints


# ============================================================================
# DOCUMENT DIAGNOSTIC ENGINE
# ============================================================================


class DocumentDiagnosticEngine:
    """
    Pre-processing diagnostic engine for document analysis.

    Runs BEFORE main Docling conversion to build a confidence profile
    and detect document characteristics that inform processing strategy.

    GEMINI AUDIT FIX: This layer prevents misclassification by:
    1. Using physical heuristics (file size, text density) as ground truth
    2. Building context for dynamic prompt injection
    3. Setting confidence thresholds to catch hallucinations

    Usage:
        engine = DocumentDiagnosticEngine()
        report = engine.analyze("document.pdf")

        if report.should_force_scan_mode():
            # Adjust processing strategy for scanned documents
            ...

        vlm_hints = report.get_vlm_context_hints()
        # Use hints in VLM prompts
    """

    def __init__(
        self,
        sample_pages: int = DIAGNOSTIC_SAMPLE_PAGES,
        enable_vlm_analysis: bool = False,
        vlm_provider: Optional[Any] = None,
    ) -> None:
        """
        Initialize DocumentDiagnosticEngine.

        Args:
            sample_pages: Number of pages to analyze
            enable_vlm_analysis: Whether to use VLM for detailed analysis
            vlm_provider: Optional VLM provider for enhanced analysis
        """
        self.sample_pages = sample_pages
        self.enable_vlm_analysis = enable_vlm_analysis
        self.vlm_provider = vlm_provider

        logger.info(
            f"DocumentDiagnosticEngine initialized: "
            f"sample_pages={sample_pages}, vlm={enable_vlm_analysis}"
        )

    def analyze(self, pdf_path: Path | str) -> DiagnosticReport:
        """
        Analyze a PDF document and return diagnostic report.

        Args:
            pdf_path: Path to PDF file

        Returns:
            DiagnosticReport with physical analysis and confidence profile
        """
        pdf_path = Path(pdf_path).resolve()

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"[DIAGNOSTIC] Starting analysis: {pdf_path.name}")

        # Step 1: Physical check (file size, page count)
        physical_check = self._perform_physical_check(pdf_path)

        # Step 2: Page-level diagnostics
        page_diagnostics = self._analyze_sample_pages(pdf_path)

        # Step 3: Build confidence profile
        confidence_profile = self._build_confidence_profile(
            physical_check, page_diagnostics, pdf_path
        )

        # Step 4: Determine recommended strategy
        recommended_strategy = self._determine_strategy(physical_check, confidence_profile)

        # Step 5: Build prompt context
        prompt_context = self._build_prompt_context(
            physical_check, confidence_profile, page_diagnostics
        )

        report = DiagnosticReport(
            source_file=pdf_path.name,
            physical_check=physical_check,
            confidence_profile=confidence_profile,
            page_diagnostics=page_diagnostics,
            recommended_strategy=recommended_strategy,
            prompt_context=prompt_context,
        )

        logger.info(
            f"[DIAGNOSTIC] Complete: modality={physical_check.detected_modality.value}, "
            f"confidence={confidence_profile.overall_confidence:.2f}, "
            f"strategy={recommended_strategy}"
        )

        return report

    def _perform_physical_check(self, pdf_path: Path) -> PhysicalCheckResult:
        """
        REQ-DIAG-02: Physical heuristic analysis.

        This is the "recall baseline" - if physical checks indicate scan,
        override VLM classification.
        """
        file_size_bytes = os.path.getsize(pdf_path)
        file_size_mb = file_size_bytes / (1024 * 1024)

        doc: Optional[fitz.Document] = None
        try:
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)

            # Sample text extraction
            total_text_length = 0
            total_image_coverage = 0.0
            pages_analyzed = min(self.sample_pages, total_pages)

            for page_num in range(pages_analyzed):
                page = doc.load_page(page_num)
                page_rect = page.rect
                page_area = page_rect.width * page_rect.height

                # Text analysis
                text_result = page.get_text()
                text = str(text_result) if text_result else ""
                total_text_length += len(text.strip())

                # Image coverage analysis
                image_list = page.get_images(full=True)
                image_area = 0.0

                for img in image_list:
                    try:
                        xref = img[0]
                        img_info = doc.extract_image(xref)
                        if img_info:
                            width = img_info.get("width", 0)
                            height = img_info.get("height", 0)
                            # Approximate: assume images are placed proportionally
                            image_area += min(width * height, page_area)
                    except Exception:
                        pass

                if page_area > 0:
                    total_image_coverage += min(image_area / page_area, 1.0)

            avg_text_per_page = total_text_length / pages_analyzed if pages_analyzed > 0 else 0
            avg_image_coverage = total_image_coverage / pages_analyzed if pages_analyzed > 0 else 0

            # ================================================================
            # PHYSICAL CHECK LOGIC (Gemini Audit Fix #1)
            # ================================================================
            # Rule: if text_length < 100 AND file_size > 1MB → force scanned
            is_likely_scan = False
            scan_confidence = 0.0
            reasoning_parts = []

            # Check 1: Low text + large file = scan
            if (
                avg_text_per_page < SCAN_TEXT_THRESHOLD_CHARS
                and file_size_mb > SCAN_FILE_SIZE_THRESHOLD_MB
            ):
                is_likely_scan = True
                scan_confidence = 0.9
                reasoning_parts.append(
                    f"Low text ({avg_text_per_page:.0f} chars/page) + "
                    f"large file ({file_size_mb:.1f}MB) indicates scanned document"
                )

            # Check 2: High image coverage = image-heavy or scan
            if avg_image_coverage > SCAN_IMAGE_RATIO_THRESHOLD:
                is_likely_scan = True
                scan_confidence = max(scan_confidence, avg_image_coverage)
                reasoning_parts.append(
                    f"High image coverage ({avg_image_coverage:.1%}) indicates "
                    f"scanned or image-heavy document"
                )

            # Check 3: Zero Docling assets (from SmartConfig) would trigger this
            # This is handled in the calling code

            # Determine modality
            if is_likely_scan:
                if avg_text_per_page > 50:
                    detected_modality = DocumentModality.SCANNED_CLEAN
                else:
                    detected_modality = DocumentModality.SCANNED_DEGRADED
            elif avg_image_coverage > 0.5:
                detected_modality = DocumentModality.IMAGE_HEAVY
            else:
                detected_modality = DocumentModality.NATIVE_DIGITAL

            if not reasoning_parts:
                reasoning_parts.append(
                    f"Normal digital document: {avg_text_per_page:.0f} chars/page, "
                    f"{avg_image_coverage:.1%} image coverage"
                )

            return PhysicalCheckResult(
                file_size_mb=file_size_mb,
                total_pages=total_pages,
                avg_text_per_page=avg_text_per_page,
                avg_image_coverage=avg_image_coverage,
                is_likely_scan=is_likely_scan,
                scan_confidence=scan_confidence,
                detected_modality=detected_modality,
                reasoning=" | ".join(reasoning_parts),
            )

        finally:
            if doc is not None:
                doc.close()

    def _analyze_sample_pages(self, pdf_path: Path) -> List[PageDiagnostic]:
        """Analyze individual sample pages for detailed diagnostics."""
        diagnostics: List[PageDiagnostic] = []

        doc: Optional[fitz.Document] = None
        try:
            doc = fitz.open(str(pdf_path))
            pages_to_analyze = min(self.sample_pages, len(doc))

            for page_num in range(pages_to_analyze):
                page = doc.load_page(page_num)
                page_rect = page.rect
                page_area = page_rect.width * page_rect.height

                # Text analysis
                text_result = page.get_text()
                text: str = str(text_result) if text_result else ""
                text_length = len(text.strip())
                text_density = text_length / page_area if page_area > 0 else 0

                # Image analysis
                image_list = page.get_images(full=True)
                image_count = len(image_list)

                # Calculate image coverage
                image_area = 0.0
                for img in image_list:
                    try:
                        xref = img[0]
                        img_info = doc.extract_image(xref)
                        if img_info:
                            width = img_info.get("width", 0)
                            height = img_info.get("height", 0)
                            image_area += width * height
                    except Exception:
                        pass

                image_coverage = min(image_area / page_area, 1.0) if page_area > 0 else 0

                # OCR artifact detection (simple heuristics)
                has_ocr_artifacts = self._detect_ocr_artifacts(text)

                # Noise level estimation
                noise_level = self._estimate_noise_level(text, text_density)

                # Table detection (simple)
                contains_tables = "table" in text.lower() or "|" in text

                diagnostics.append(
                    PageDiagnostic(
                        page_number=page_num + 1,
                        text_length=text_length,
                        text_density=text_density,
                        image_count=image_count,
                        image_coverage=image_coverage,
                        has_ocr_artifacts=has_ocr_artifacts,
                        detected_noise_level=noise_level,
                        contains_tables=contains_tables,
                    )
                )

            return diagnostics

        finally:
            if doc is not None:
                doc.close()

    def _detect_ocr_artifacts(self, text: str) -> bool:
        """Detect common OCR artifacts in text."""
        if not text:
            return False

        # Common OCR error patterns
        ocr_patterns = [
            "l1",  # l/1 confusion
            "0O",  # 0/O confusion
            "rn",  # rn/m confusion
            "|||",  # misread lines
            "...",  # ellipsis artifacts
        ]

        artifact_count = sum(1 for p in ocr_patterns if p in text)

        # Check for unusual character distributions
        non_alpha = sum(1 for c in text if not c.isalnum() and c not in " .,;:!?-'\"")
        if len(text) > 0 and non_alpha / len(text) > 0.1:
            artifact_count += 1

        return artifact_count >= 2

    def _estimate_noise_level(self, text: str, text_density: float) -> float:
        """Estimate text noise level (0.0 = clean, 1.0 = noisy)."""
        if not text:
            return 0.5

        noise_score = 0.0

        # Low text density = potential scan with poor OCR
        if text_density < 0.001:
            noise_score += 0.3

        # Short text chunks separated by many spaces
        words = text.split()
        if words:
            avg_word_len = sum(len(w) for w in words) / len(words)
            if avg_word_len < 3:
                noise_score += 0.2

        # Many special characters
        special_chars = sum(1 for c in text if c in "·•—–□■●○◊")
        if len(text) > 0 and special_chars / len(text) > 0.05:
            noise_score += 0.3

        return min(noise_score, 1.0)

    def _build_confidence_profile(
        self,
        physical_check: PhysicalCheckResult,
        page_diagnostics: List[PageDiagnostic],
        pdf_path: Path,
    ) -> ConfidenceProfile:
        """
        Build confidence profile for VLM guidance.

        This profile helps the VLM understand what kind of document it's analyzing
        and sets confidence thresholds to catch hallucinations.
        """
        detected_features: List[str] = []
        warnings: List[str] = []

        # Analyze page diagnostics
        avg_noise = (
            sum(p.detected_noise_level for p in page_diagnostics) / len(page_diagnostics)
            if page_diagnostics
            else 0
        )
        has_tables = any(p.contains_tables for p in page_diagnostics)
        has_ocr_issues = any(p.has_ocr_artifacts for p in page_diagnostics)

        # Detected features
        if physical_check.is_likely_scan:
            detected_features.append("scanned_document")

        if has_tables:
            detected_features.append("contains_tables")

        if physical_check.avg_image_coverage > 0.3:
            detected_features.append("image_heavy")

        if has_ocr_issues:
            detected_features.append("ocr_artifacts_present")
            warnings.append("OCR quality may be poor - verify extracted text")

        # Detect era based on file characteristics
        detected_era = self._estimate_document_era(physical_check, page_diagnostics)

        # Detect domain based on filename and content
        detected_domain = self._estimate_content_domain(pdf_path, page_diagnostics)

        # Calculate confidence scores
        classification_confidence = 1.0 - (avg_noise * 0.5)
        if physical_check.is_likely_scan:
            classification_confidence *= 0.9  # Scans have more uncertainty

        if has_ocr_issues:
            classification_confidence *= 0.8
            warnings.append("Low confidence due to OCR artifacts")

        overall_confidence = (
            classification_confidence * 0.6
            + physical_check.scan_confidence * 0.2
            + (1.0 - avg_noise) * 0.2
        )

        if overall_confidence < LOW_CONFIDENCE_THRESHOLD:
            warnings.append("Overall confidence is LOW - manual review recommended")

        return ConfidenceProfile(
            overall_confidence=overall_confidence,
            classification_confidence=classification_confidence,
            detected_features=detected_features,
            detected_era=detected_era,
            detected_domain=detected_domain,
            warnings=warnings,
        )

    def _estimate_document_era(
        self,
        physical_check: PhysicalCheckResult,
        page_diagnostics: List[PageDiagnostic],
    ) -> DocumentEra:
        """Estimate document era based on characteristics."""
        avg_noise = (
            sum(p.detected_noise_level for p in page_diagnostics) / len(page_diagnostics)
            if page_diagnostics
            else 0
        )

        # High noise + scan = likely historical
        if physical_check.is_likely_scan and avg_noise > 0.5:
            return DocumentEra.HISTORICAL

        # Scan with moderate quality = vintage
        if physical_check.is_likely_scan and avg_noise > 0.2:
            return DocumentEra.VINTAGE

        # Digital with high text = modern
        if not physical_check.is_likely_scan and physical_check.avg_text_per_page > 500:
            return DocumentEra.MODERN

        return DocumentEra.UNKNOWN

    def _estimate_content_domain(
        self,
        pdf_path: Path,
        page_diagnostics: List[PageDiagnostic],
    ) -> ContentDomain:
        """Estimate content domain based on filename and characteristics."""
        filename_lower = pdf_path.stem.lower()

        # Filename-based hints
        technical_keywords = ["manual", "guide", "spec", "handbook", "firearms", "weapons", "tech"]
        editorial_keywords = ["magazine", "news", "article", "review", "journal"]
        academic_keywords = ["paper", "thesis", "research", "study", "report"]
        commercial_keywords = ["catalog", "brochure", "ad", "marketing", "promo"]

        for kw in technical_keywords:
            if kw in filename_lower:
                return ContentDomain.TECHNICAL

        for kw in editorial_keywords:
            if kw in filename_lower:
                return ContentDomain.EDITORIAL

        for kw in academic_keywords:
            if kw in filename_lower:
                return ContentDomain.ACADEMIC

        for kw in commercial_keywords:
            if kw in filename_lower:
                return ContentDomain.COMMERCIAL

        # Content-based hints
        has_tables = any(p.contains_tables for p in page_diagnostics)
        high_images = any(p.image_coverage > 0.5 for p in page_diagnostics)

        if has_tables and not high_images:
            return ContentDomain.TECHNICAL

        if high_images:
            return ContentDomain.EDITORIAL

        return ContentDomain.UNKNOWN

    def _determine_strategy(
        self,
        physical_check: PhysicalCheckResult,
        confidence_profile: ConfidenceProfile,
    ) -> str:
        """Determine recommended processing strategy."""
        if physical_check.is_likely_scan:
            if physical_check.detected_modality == DocumentModality.SCANNED_DEGRADED:
                return "scan_degraded_high_ocr"
            else:
                return "scan_clean_standard_ocr"

        if physical_check.detected_modality == DocumentModality.IMAGE_HEAVY:
            return "image_heavy_high_recall"

        if confidence_profile.is_low_confidence():
            return "low_confidence_conservative"

        return "standard_digital"

    def _build_prompt_context(
        self,
        physical_check: PhysicalCheckResult,
        confidence_profile: ConfidenceProfile,
        page_diagnostics: List[PageDiagnostic],
    ) -> Dict[str, Any]:
        """
        Build dynamic prompt context for VLM injection.

        GEMINI AUDIT FIX #2: Template-based prompt variables.
        """
        context = {
            "classification": physical_check.detected_modality.value,
            "detected_features": confidence_profile.detected_features,
            "document_era": confidence_profile.detected_era.value,
            "content_domain": confidence_profile.detected_domain.value,
            "confidence_level": (
                "high"
                if confidence_profile.is_high_confidence()
                else ("low" if confidence_profile.is_low_confidence() else "medium")
            ),
            "is_scan": physical_check.is_likely_scan,
            "warnings": confidence_profile.warnings,
        }

        # Build artifact hints for scanned documents
        if physical_check.is_likely_scan:
            context["scan_hints"] = [
                "Ignore paper texture, grain, and discoloration",
                "Focus on printed content, not scan artifacts",
                "Stains and foxing are NOT content elements",
            ]

            if confidence_profile.detected_era == DocumentEra.HISTORICAL:
                context["scan_hints"].extend(
                    [
                        "This appears to be a historical document",
                        "Expect aged typography and printing imperfections",
                    ]
                )

        return context


# ============================================================================
# FACTORY FUNCTION
# ============================================================================


def create_diagnostic_engine(
    sample_pages: int = DIAGNOSTIC_SAMPLE_PAGES,
    enable_vlm: bool = False,
) -> DocumentDiagnosticEngine:
    """
    Factory function to create DocumentDiagnosticEngine.

    Args:
        sample_pages: Number of pages to analyze
        enable_vlm: Enable VLM-based analysis (requires provider)

    Returns:
        Configured DocumentDiagnosticEngine
    """
    return DocumentDiagnosticEngine(
        sample_pages=sample_pages,
        enable_vlm_analysis=enable_vlm,
    )
