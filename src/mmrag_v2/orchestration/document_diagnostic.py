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
DEEP_SCAN_SAMPLE_PAGES: int = 15  # V16: Deep scan for low confidence cases
DPI_LOW: int = 72  # Quick analysis
DPI_HIGH: int = 150  # Detailed analysis

# Confidence thresholds
HIGH_CONFIDENCE_THRESHOLD: float = 0.85
LOW_CONFIDENCE_THRESHOLD: float = 0.50
DEEP_SCAN_TRIGGER_THRESHOLD: float = 0.60  # V16: Trigger deep scan if below this


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

        V16 DEEP SCAN FEATURE (2026-01-10):
        ===================================
        If initial confidence < DEEP_SCAN_TRIGGER_THRESHOLD (0.6), the engine
        automatically performs a "Deep Scan" analyzing up to 15 pages using
        stratified sampling (beginning, middle, end) to build a more robust
        confidence profile.

        This catches documents with:
        - Long introductions with only images (then text-heavy content later)
        - Mixed modality (some pages scanned, some digital)
        - OCR that varies in quality across the document

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

        # Step 2: Page-level diagnostics (initial sample)
        page_diagnostics = self._analyze_sample_pages(pdf_path)

        # Step 3: Build confidence profile
        confidence_profile = self._build_confidence_profile(
            physical_check, page_diagnostics, pdf_path
        )

        # ================================================================
        # V16 DEEP SCAN: Automatic extended analysis for low confidence
        # ================================================================
        # If confidence is below threshold and document has enough pages,
        # perform a deep scan with stratified sampling for more accuracy.
        if (
            confidence_profile.overall_confidence < DEEP_SCAN_TRIGGER_THRESHOLD
            and physical_check.total_pages > self.sample_pages
        ):
            logger.warning(
                f"[DIAGNOSTIC] ⚠ LOW CONFIDENCE ({confidence_profile.overall_confidence:.2f} < "
                f"{DEEP_SCAN_TRIGGER_THRESHOLD}) - Triggering DEEP SCAN"
            )

            # Perform deep scan with stratified sampling
            deep_diagnostics = self._perform_deep_scan(pdf_path, physical_check.total_pages)

            # Merge diagnostics
            page_diagnostics = self._merge_diagnostics(page_diagnostics, deep_diagnostics)

            # Rebuild confidence profile with more data
            confidence_profile = self._build_confidence_profile(
                physical_check, page_diagnostics, pdf_path
            )

            # Update reasoning
            confidence_profile.warnings.append(
                f"Deep scan performed: {len(page_diagnostics)} pages analyzed "
                f"(stratified sampling across {physical_check.total_pages} total pages)"
            )

            logger.info(
                f"[DIAGNOSTIC] Deep scan complete: new confidence={confidence_profile.overall_confidence:.2f}"
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

    def _perform_deep_scan(
        self,
        pdf_path: Path,
        total_pages: int,
    ) -> List[PageDiagnostic]:
        """
        V16 DEEP SCAN: Stratified sampling for low-confidence documents.

        Performs extended analysis using stratified sampling:
        - First 5 pages (already analyzed in initial pass)
        - Middle 5 pages (random sample from middle third)
        - Last 5 pages (random sample from last third)

        This captures variation across the document that might be missed
        by only analyzing the first few pages.

        Args:
            pdf_path: Path to PDF file
            total_pages: Total number of pages in document

        Returns:
            List of PageDiagnostic for additional sampled pages
        """
        import random

        logger.info(
            f"[DEEP-SCAN] Starting stratified sampling on {total_pages} pages "
            f"(max {DEEP_SCAN_SAMPLE_PAGES} samples)"
        )

        # Calculate page ranges for stratified sampling
        # We want to sample from: beginning (already done), middle, end
        pages_per_section = max(1, (DEEP_SCAN_SAMPLE_PAGES - self.sample_pages) // 2)

        # Already have first N pages, so sample from middle and end
        middle_start = total_pages // 3
        middle_end = (2 * total_pages) // 3
        end_start = (2 * total_pages) // 3

        # Generate page indices to sample (0-indexed)
        middle_pages = list(range(middle_start, middle_end))
        end_pages = list(range(end_start, total_pages))

        # Random sample from each section
        sample_middle = random.sample(middle_pages, min(pages_per_section, len(middle_pages)))
        sample_end = random.sample(end_pages, min(pages_per_section, len(end_pages)))

        # Combine and sort
        pages_to_analyze = sorted(set(sample_middle + sample_end))

        logger.info(
            f"[DEEP-SCAN] Sampling pages: {pages_to_analyze} "
            f"(middle: {sample_middle}, end: {sample_end})"
        )

        # Analyze the additional pages
        diagnostics: List[PageDiagnostic] = []

        doc: Optional[fitz.Document] = None
        try:
            doc = fitz.open(str(pdf_path))

            for page_idx in pages_to_analyze:
                if page_idx >= len(doc):
                    continue

                page = doc.load_page(page_idx)
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

                # OCR artifact detection
                has_ocr_artifacts = self._detect_ocr_artifacts(text)

                # Noise level estimation
                noise_level = self._estimate_noise_level(text, text_density)

                # Table detection
                contains_tables = "table" in text.lower() or "|" in text

                diagnostics.append(
                    PageDiagnostic(
                        page_number=page_idx + 1,  # 1-indexed
                        text_length=text_length,
                        text_density=text_density,
                        image_count=image_count,
                        image_coverage=image_coverage,
                        has_ocr_artifacts=has_ocr_artifacts,
                        detected_noise_level=noise_level,
                        contains_tables=contains_tables,
                    )
                )

            logger.info(f"[DEEP-SCAN] Analyzed {len(diagnostics)} additional pages")
            return diagnostics

        finally:
            if doc is not None:
                doc.close()

    def _merge_diagnostics(
        self,
        initial: List[PageDiagnostic],
        deep_scan: List[PageDiagnostic],
    ) -> List[PageDiagnostic]:
        """
        Merge initial and deep scan diagnostics, removing duplicates.

        Args:
            initial: Initial page diagnostics
            deep_scan: Deep scan page diagnostics

        Returns:
            Merged list of unique page diagnostics
        """
        # Use dict to ensure unique pages (keyed by page_number)
        merged = {d.page_number: d for d in initial}
        for d in deep_scan:
            if d.page_number not in merged:
                merged[d.page_number] = d

        # Return sorted by page number
        return sorted(merged.values(), key=lambda d: d.page_number)

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
            # V2.4 FORENSICS: Detect "Sandwich PDFs" (OCR-over-Scan)
            # ================================================================
            # PROBLEM: Harry Potter is a SCAN but has an OCR layer on top.
            # This makes it look "digital" (791 chars/page, Acrobat metadata).
            # But the text is from OCR, not native vector text!
            #
            # SOLUTION: Forensic analysis to detect OCR-sandwich patterns:
            # 1. FONT ANALYSIS: OCR uses generic fonts (GlyphLess, Identity-H, T1)
            # 2. IMAGE-IS-PAGE: If 1 image fills the entire page = scan
            # 3. FONT DIVERSITY: Real digital docs have multiple fonts
            #
            # This check runs BEFORE other logic to catch sandwich PDFs early.
            # ================================================================
            is_sandwich_pdf = False
            sandwich_reasons = []

            # Collect font information from sample pages
            all_fonts = set()
            pages_with_fullpage_image = 0

            for page_num in range(pages_analyzed):
                page = doc.load_page(page_num)
                page_rect = page.rect
                page_area = page_rect.width * page_rect.height

                # FONT ANALYSIS: Collect all font names
                text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
                blocks = text_dict.get("blocks", []) if isinstance(text_dict, dict) else []
                for block in blocks:
                    if isinstance(block, dict) and block.get("type") == 0:  # Text block
                        lines = block.get("lines", [])
                        for line in lines:
                            if isinstance(line, dict):
                                spans = line.get("spans", [])
                                for span in spans:
                                    if isinstance(span, dict):
                                        font_name = str(span.get("font", "")).lower()
                                        if font_name:
                                            all_fonts.add(font_name)

                # IMAGE-IS-PAGE CHECK: Does one image fill the entire page?
                image_list = page.get_images(full=True)
                for img in image_list:
                    try:
                        xref = img[0]
                        img_info = doc.extract_image(xref)
                        if img_info:
                            img_width = img_info.get("width", 0)
                            img_height = img_info.get("height", 0)
                            img_area = img_width * img_height
                            # If image covers > 90% of page area, it's likely the scan itself
                            if page_area > 0 and img_area / page_area > 0.90:
                                pages_with_fullpage_image += 1
                                break  # Only count once per page
                    except Exception:
                        pass

            # ================================================================
            # SANDWICH PDF DETECTION RULES
            # ================================================================
            # OCR-generated fonts have distinctive names
            OCR_FONT_INDICATORS = [
                "glyphless",  # Common OCR invisible font
                "identity",  # Identity-H encoding (OCR)
                "cid",  # CID fonts from OCR
                "r0",  # Generic OCR font naming (R0001, R0002)
                "t1",  # Type1 placeholder
                "unknown",  # Generic fallback
                "notdef",  # Missing glyph font
            ]

            # Check 1: Suspicious OCR fonts
            ocr_font_count = sum(
                1 for f in all_fonts if any(ind in f for ind in OCR_FONT_INDICATORS)
            )
            total_fonts = len(all_fonts)

            # ================================================================
            # V2.5 FIX: SANDWICH DETECTION REQUIRES LOW NATIVE TEXT
            # ================================================================
            # CRITICAL: All sandwich PDF checks should ONLY apply when
            # avg_text_per_page is LOW (< 500). If there's substantial native
            # text, the document is DIGITAL regardless of other indicators.
            #
            # Combat Aircraft Bug: 2675 chars/page + 2 fonts = falsely flagged
            # as sandwich because of "low font diversity" check.
            #
            # FIX: Gate ALL sandwich checks behind a text threshold.
            # If avg_text_per_page >= 500, skip sandwich detection entirely.
            # ================================================================
            SANDWICH_TEXT_CEILING = 500  # If above this, skip sandwich detection

            if avg_text_per_page < SANDWICH_TEXT_CEILING:
                # Only check for sandwich PDFs when native text is LOW

                if total_fonts > 0 and ocr_font_count / total_fonts > 0.5:
                    is_sandwich_pdf = True
                    sandwich_reasons.append(
                        f"OCR fonts detected ({ocr_font_count}/{total_fonts} fonts are OCR-style)"
                    )

                # Check 2: Very low font diversity with some text = likely OCR
                # Real magazines have 5+ fonts; OCR scans often have 1-2
                # NOTE: Only applies when text is LOW (gated above)
                if total_fonts <= 2 and avg_text_per_page > 50:
                    is_sandwich_pdf = True
                    sandwich_reasons.append(
                        f"Suspiciously low font diversity ({total_fonts} fonts for {avg_text_per_page:.0f} chars/page)"
                    )

                # Check 3: Most pages have a single full-page image (the scan itself)
                fullpage_ratio = (
                    pages_with_fullpage_image / pages_analyzed if pages_analyzed > 0 else 0
                )
                if fullpage_ratio > 0.7:  # >70% of pages have full-page images
                    is_sandwich_pdf = True
                    sandwich_reasons.append(
                        f"Full-page images detected ({pages_with_fullpage_image}/{pages_analyzed} pages = "
                        f"{fullpage_ratio:.0%}) - likely scanned pages"
                    )
            else:
                # HIGH native text = definitely digital, skip sandwich detection
                logger.debug(
                    f"[DIAGNOSTIC] Skipping sandwich detection: {avg_text_per_page:.0f} chars/page "
                    f">= {SANDWICH_TEXT_CEILING} threshold"
                )

            # ================================================================
            # PHYSICAL CHECK LOGIC (V2.1 FIX: TEXT LAYER IS KING)
            # ================================================================
            # CRITICAL RULE: Native text layer presence OVERRIDES image coverage!
            #
            # Modern digital magazines (Combat Aircraft) often have:
            # - 100% image coverage (full-page photos, backgrounds)
            # - BUT ALSO native text as vectors on top
            # - This is NOT a scan! The text layer proves digital origin.
            #
            # Scanned documents have:
            # - 100% image coverage (because the whole page IS one image)
            # - VERY LOW native text (only what OCR extracted, if any)
            #
            # THE FIX: Check text layer FIRST. If substantial text exists,
            # it's DIGITAL regardless of image coverage.
            #
            # V2.2 ENHANCEMENT: Also check PDF metadata for digital origin proof.

            is_likely_scan = False
            scan_confidence = 0.0
            reasoning_parts = []

            # ================================================================
            # THRESHOLD: Native text layer indicates digital origin
            # ================================================================
            # If we have > 500 chars/page of native text, this is DIGITAL
            # even if image coverage is 100%. Modern magazines/PDFs have
            # text as vectors overlaid on images.
            #
            # NOTE: This threshold is applied to the AVERAGE of sampled pages.
            # A cover page with only 100 chars won't disqualify the document
            # if subsequent pages have 2000+ chars.
            NATIVE_TEXT_DIGITAL_THRESHOLD = 500  # chars/page (AVERAGE)

            # ================================================================
            # V2.2: PDF METADATA CHECK - Digital software as proof
            # ================================================================
            # If PDF was created by digital software, it's DEFINITELY digital
            # regardless of any other heuristic.
            pdf_metadata = doc.metadata or {}
            creator = str(pdf_metadata.get("creator", "") or "").lower()
            producer = str(pdf_metadata.get("producer", "") or "").lower()

            # Known digital PDF creators (proves born-digital origin)
            DIGITAL_CREATORS = [
                "indesign",  # Adobe InDesign
                "quartz",  # macOS/iOS native
                "acrobat",  # Adobe Acrobat (NOT scanned PDFs)
                "illustrator",  # Adobe Illustrator
                "distiller",  # Adobe Distiller
                "office",  # Microsoft Office
                "word",  # Microsoft Word
                "powerpoint",  # Microsoft PowerPoint
                "latex",  # LaTeX
                "pdflatex",  # pdfLaTeX
                "cairo",  # Cairo graphics library
                "skia",  # Skia graphics library
                "reportlab",  # ReportLab Python
                "pdfkit",  # wkhtmltopdf/PDFKit
                "prince",  # PrinceXML
                "weasyprint",  # WeasyPrint
            ]

            # Check for digital creator in metadata
            is_digital_by_metadata = any(dc in creator or dc in producer for dc in DIGITAL_CREATORS)

            # ================================================================
            # V2.3 FIX: METADATA CHECK REQUIRES TEXT LAYER CONFIRMATION
            # ================================================================
            # PROBLEM: Acrobat Distiller is also used to export SCANNED documents!
            # Harry Potter PDF has "Acrobat Distiller" in producer but only 8 chars/page.
            # This means the PDF is a SCAN exported through Acrobat, NOT born-digital.
            #
            # FIX: Metadata check only confirms DIGITAL if there's ALSO substantial text.
            # If avg_text_per_page < 100, the document is still a SCAN regardless of metadata.
            # Scanned documents have their text in the image, not as native text.
            #
            # THRESHOLD: 100 chars/page is the minimum for metadata confirmation.
            # (Same as SCAN_TEXT_THRESHOLD_CHARS)
            METADATA_REQUIRES_TEXT_THRESHOLD = 100  # chars/page

            # ================================================================
            # V2.4 SANDWICH PDF OVERRIDE
            # ================================================================
            # If we detected a sandwich PDF (OCR layer over scan), OVERRIDE
            # all other checks and mark as SCAN. This catches Harry Potter!
            # ================================================================
            if is_sandwich_pdf:
                is_likely_scan = True
                scan_confidence = 0.90
                reasoning_parts.append(f"⚠ SANDWICH PDF DETECTED: OCR layer over scanned pages")
                for reason in sandwich_reasons:
                    reasoning_parts.append(f"  → {reason}")
                reasoning_parts.append(
                    f"Text appears to be from OCR ({avg_text_per_page:.0f} chars/page), "
                    f"not native digital origin"
                )

            elif is_digital_by_metadata and avg_text_per_page >= METADATA_REQUIRES_TEXT_THRESHOLD:
                # Metadata + text layer = definitely digital (but NOT if sandwich PDF!)
                is_likely_scan = False
                scan_confidence = 0.0
                matched_creator = next(
                    (dc for dc in DIGITAL_CREATORS if dc in creator or dc in producer),
                    "unknown",
                )
                reasoning_parts.append(
                    f"✓ DIGITAL: PDF metadata ('{matched_creator}') + "
                    f"text layer ({avg_text_per_page:.0f} chars/page) confirms digital origin"
                )
                reasoning_parts.append(
                    f"Creator: '{pdf_metadata.get('creator', 'N/A')}' | "
                    f"Producer: '{pdf_metadata.get('producer', 'N/A')}'"
                )
            elif is_digital_by_metadata and avg_text_per_page < METADATA_REQUIRES_TEXT_THRESHOLD:
                # Metadata says digital, but NO text layer = likely a scanned document
                # exported through Acrobat. This is the Harry Potter scenario!
                is_likely_scan = True
                scan_confidence = 0.85
                matched_creator = next(
                    (dc for dc in DIGITAL_CREATORS if dc in creator or dc in producer),
                    "unknown",
                )
                reasoning_parts.append(
                    f"⚠ SCAN: PDF has '{matched_creator}' metadata BUT only "
                    f"{avg_text_per_page:.0f} chars/page native text"
                )
                reasoning_parts.append(
                    f"This is likely a SCANNED document exported through {matched_creator}"
                )

            # ================================================================
            # PRIORITY CHECK: Native text layer = DIGITAL (overrides all else)
            # ================================================================
            elif avg_text_per_page >= NATIVE_TEXT_DIGITAL_THRESHOLD:
                # SUBSTANTIAL NATIVE TEXT = ALWAYS DIGITAL
                is_likely_scan = False
                scan_confidence = 0.0
                reasoning_parts.append(
                    f"✓ DIGITAL: Native text layer detected "
                    f"({avg_text_per_page:.0f} chars/page avg >= {NATIVE_TEXT_DIGITAL_THRESHOLD} threshold)"
                )
                reasoning_parts.append(
                    f"Image coverage ({avg_image_coverage:.1%}) does NOT indicate scan - "
                    f"text layer proves digital origin"
                )
            else:
                # LOW/NO NATIVE TEXT - Now check other indicators
                # Check 1: Low text + large file = likely scan
                if (
                    avg_text_per_page < SCAN_TEXT_THRESHOLD_CHARS
                    and file_size_mb > SCAN_FILE_SIZE_THRESHOLD_MB
                ):
                    is_likely_scan = True
                    scan_confidence = 0.9
                    reasoning_parts.append(
                        f"Low text ({avg_text_per_page:.0f} chars/page < {SCAN_TEXT_THRESHOLD_CHARS}) + "
                        f"large file ({file_size_mb:.1f}MB) indicates scanned document"
                    )

                # Check 2: High image coverage + low text = scan
                # NOTE: We only reach here if text is already low (< 500 chars/page)
                if avg_image_coverage > SCAN_IMAGE_RATIO_THRESHOLD:
                    is_likely_scan = True
                    scan_confidence = max(scan_confidence, avg_image_coverage)
                    reasoning_parts.append(
                        f"High image coverage ({avg_image_coverage:.1%}) + "
                        f"low native text ({avg_text_per_page:.0f} chars/page) indicates scan"
                    )

            # ================================================================
            # DETERMINE MODALITY
            # ================================================================
            if is_likely_scan:
                if avg_text_per_page > 50:
                    detected_modality = DocumentModality.SCANNED_CLEAN
                else:
                    detected_modality = DocumentModality.SCANNED_DEGRADED
            elif avg_image_coverage > 0.5:
                # High images but also high text = digital magazine/editorial
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
        """
        Estimate content domain based on filename and characteristics.

        V2.6 FIX (2026-01-11): Expanded academic keyword detection
        =========================================================
        Academic papers often have descriptive titles like:
        - "Hybrid_electric_vehicles_and_their_challenges.pdf"
        - "Deep_Learning_for_Natural_Language_Processing.pdf"
        - "Analysis_of_Climate_Change_Effects_on_Biodiversity.pdf"

        These don't contain obvious keywords like "paper" or "thesis"
        but have patterns that indicate academic content:
        1. Underscores separating words (exported from academic systems)
        2. Technical/scientific terminology
        3. "and", "of", "for", "in" conjunctions typical of paper titles
        """
        filename_lower = pdf_path.stem.lower()

        # Replace underscores with spaces for better matching
        filename_normalized = filename_lower.replace("_", " ").replace("-", " ")

        # ================================================================
        # ACADEMIC KEYWORD DETECTION (V2.6 EXPANDED)
        # ================================================================
        # Technical/scientific terms that indicate academic content
        academic_keywords = [
            # Publication types
            "paper",
            "thesis",
            "dissertation",
            "research",
            "study",
            "report",
            "arxiv",
            "acm",
            "ieee",
            "whitepaper",
            "white paper",
            "survey",
            "review",
            "analysis",
            "evaluation",
            "assessment",
            "investigation",
            # Scientific domains
            "hybrid",
            "electric",
            "vehicle",
            "vehicles",
            "autonomous",
            "robot",
            "neural",
            "network",
            "deep learning",
            "machine learning",
            "algorithm",
            "model",
            "framework",
            "system",
            "architecture",
            "protocol",
            "method",
            "approach",
            "technique",
            "solution",
            "implementation",
            # Academic structure words
            "challenges",
            "opportunities",
            "applications",
            "implications",
            "performance",
            "efficiency",
            "optimization",
            "comparison",
            "simulation",
            "experiment",
            "results",
            "findings",
            "conclusions",
            # Technical fields
            "computer",
            "software",
            "hardware",
            "database",
            "network",
            "security",
            "privacy",
            "encryption",
            "blockchain",
            "quantum",
            "energy",
            "power",
            "battery",
            "fuel",
            "renewable",
            "sustainable",
            "climate",
            "environment",
            "ecology",
            "biology",
            "chemistry",
            "physics",
            "mathematics",
            "statistics",
            "economics",
            "finance",
            "medical",
            "health",
            "clinical",
            "pharmaceutical",
            "genomic",
            "llm",
            "agent",
            "transformer",
            "attention",
            "reinforcement",
            "operating system",
            "distributed",
            "parallel",
            "concurrent",
        ]

        # Editorial keywords (magazines, newspapers)
        editorial_keywords = ["magazine", "news", "daily", "weekly", "monthly", "issue"]

        # Technical manual keywords
        technical_keywords = [
            "manual",
            "guide",
            "spec",
            "handbook",
            "firearms",
            "weapons",
            "tech",
            "instruction",
            "documentation",
        ]

        # Commercial keywords
        commercial_keywords = ["catalog", "brochure", "ad", "marketing", "promo", "sale"]

        # ================================================================
        # SCORING SYSTEM: Count keyword matches
        # ================================================================
        # Instead of first-match, count ALL matches to determine domain
        academic_score = 0
        editorial_score = 0
        technical_score = 0
        commercial_score = 0

        for kw in academic_keywords:
            if kw in filename_normalized:
                academic_score += 1
                logger.debug(f"[DOMAIN-DETECT] Academic keyword match: '{kw}'")

        for kw in editorial_keywords:
            if kw in filename_normalized:
                editorial_score += 1
                logger.debug(f"[DOMAIN-DETECT] Editorial keyword match: '{kw}'")

        for kw in technical_keywords:
            if kw in filename_normalized:
                technical_score += 1
                logger.debug(f"[DOMAIN-DETECT] Technical keyword match: '{kw}'")

        for kw in commercial_keywords:
            if kw in filename_normalized:
                commercial_score += 1
                logger.debug(f"[DOMAIN-DETECT] Commercial keyword match: '{kw}'")

        # ================================================================
        # EDITORIAL BOOST: "magazine" keyword is very strong signal
        # ================================================================
        # If "magazine" is in the filename, give a STRONG boost to editorial
        # This prevents magazines from being misclassified as academic
        if "magazine" in filename_normalized:
            editorial_score += 5  # Strong boost for explicit magazine keyword
            logger.debug(f"[DOMAIN-DETECT] EDITORIAL BOOST: 'magazine' in filename (+5)")

        # ================================================================
        # ACADEMIC PATTERN DETECTION
        # ================================================================
        # Academic papers often have specific filename patterns
        # BUT only apply pattern boost if editorial_score is still 0
        # (magazines shouldn't get academic pattern boosts)
        import re

        if editorial_score == 0:
            academic_patterns = [
                # Underscores with conjunctions (common in exported papers)
                r"\w+_and_\w+",  # "vehicles_and_their"
                r"\w+_of_\w+",  # "analysis_of_climate"
                r"\w+_for_\w+",  # "learning_for_nlp"
                r"\w+_in_\w+",  # "advances_in_robotics"
                # Multiple underscores (academic export format)
                r"\w+_\w+_\w+_\w+",  # 4+ word titles
            ]

            for pattern in academic_patterns:
                if re.search(pattern, filename_lower):
                    academic_score += 2  # Boost for pattern match
                    logger.debug(f"[DOMAIN-DETECT] Academic pattern match: '{pattern}'")
                    break

        # Log scores
        logger.info(
            f"[DOMAIN-DETECT] Scores for '{pdf_path.name}': "
            f"academic={academic_score}, editorial={editorial_score}, "
            f"technical={technical_score}, commercial={commercial_score}"
        )

        # ================================================================
        # DETERMINE DOMAIN BY HIGHEST SCORE
        # ================================================================
        # Academic wins if it has >= 2 matches (to avoid false positives)
        if academic_score >= 2 and academic_score >= editorial_score:
            logger.info(f"[DOMAIN-DETECT] → ACADEMIC (score={academic_score})")
            return ContentDomain.ACADEMIC

        if editorial_score > 0 and editorial_score > academic_score:
            logger.info(f"[DOMAIN-DETECT] → EDITORIAL (score={editorial_score})")
            return ContentDomain.EDITORIAL

        if technical_score > 0 and technical_score >= academic_score:
            logger.info(f"[DOMAIN-DETECT] → TECHNICAL (score={technical_score})")
            return ContentDomain.TECHNICAL

        if commercial_score > 0:
            logger.info(f"[DOMAIN-DETECT] → COMMERCIAL (score={commercial_score})")
            return ContentDomain.COMMERCIAL

        # ================================================================
        # CONTENT-BASED FALLBACK
        # ================================================================
        has_tables = any(p.contains_tables for p in page_diagnostics)
        high_images = any(p.image_coverage > 0.5 for p in page_diagnostics)

        # Calculate average text density
        avg_text_density = (
            sum(p.text_density for p in page_diagnostics) / len(page_diagnostics)
            if page_diagnostics
            else 0
        )

        # Academic papers have:
        # 1. High text density (lots of text per page)
        # 2. Low-to-moderate image coverage (diagrams but not editorial photos)
        # 3. Often have tables
        if avg_text_density > 0.01 and not high_images:  # High text, low images
            logger.info(f"[DOMAIN-DETECT] → ACADEMIC (content-based: high text density)")
            return ContentDomain.ACADEMIC

        if has_tables and not high_images:
            logger.info(f"[DOMAIN-DETECT] → TECHNICAL (content-based: has tables)")
            return ContentDomain.TECHNICAL

        if high_images:
            logger.info(f"[DOMAIN-DETECT] → EDITORIAL (content-based: high images)")
            return ContentDomain.EDITORIAL

        logger.info(f"[DOMAIN-DETECT] → UNKNOWN (no matches)")
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
