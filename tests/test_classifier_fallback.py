"""
Classifier Fallback Regression Tests
=====================================
Validates that long-form literature, code-heavy books, and encoding-corrupt
documents do NOT misroute as digital_magazine. Exercises the low-confidence
fallback logic and modality-aware profile selection.

Regression targets:
- Harry Potter (327pp literature, extreme inline image density)
- Ayeva Python Patterns (code-heavy book, editorial domain)
- Long-form literature with extreme image density
- Encoding-corrupt long-form code book
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pytest

from mmrag_v2.orchestration.profile_classifier import (
    ClassificationFeatures,
    ProfileClassifier,
    ProfileType,
)
from mmrag_v2.orchestration.smart_config import DocumentProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile(
    *,
    total_pages: int = 100,
    image_density: float = 0.3,
    median_image_width: int = 150,
    median_image_height: int = 100,
    avg_text_per_page: float = 1500.0,
    has_text: bool = True,
    image_count: int = 30,
) -> DocumentProfile:
    return DocumentProfile(
        total_pages=total_pages,
        image_density=image_density,
        median_image_width=median_image_width,
        median_image_height=median_image_height,
        avg_text_per_page=avg_text_per_page,
        has_text=has_text,
        image_count=image_count,
    )


class FakeDiagnosticReport:
    """Minimal diagnostic report stub for classifier tests."""

    def __init__(
        self,
        is_likely_scan: bool = False,
        scan_confidence: float = 0.0,
        detected_domain: str = "unknown",
        detected_modality: str = "native_digital",
    ):
        self.physical_check = _FakePhysicalCheck(
            is_likely_scan=is_likely_scan,
            scan_confidence=scan_confidence,
            detected_modality=detected_modality,
        )
        self.confidence_profile = _FakeConfidenceProfile(
            detected_domain=detected_domain,
        )


class _FakePhysicalCheck:
    def __init__(self, is_likely_scan, scan_confidence, detected_modality):
        self.is_likely_scan = is_likely_scan
        self.scan_confidence = scan_confidence
        self.detected_modality = type("M", (), {"value": detected_modality})()


class _FakeConfidenceProfile:
    def __init__(self, detected_domain):
        self.detected_domain = type("D", (), {"value": detected_domain})()


# ---------------------------------------------------------------------------
# Negative tests: must NOT route as digital_magazine
# ---------------------------------------------------------------------------


class TestNeverDigitalMagazine:
    """Documents that must NEVER be classified as digital_magazine."""

    def test_harry_potter_like_literature(self):
        """Long-form literature with extreme inline image density.

        Regression: Harry Potter (327pp, literature, image_density=14.6)
        was misrouted as digital_magazine because literature domain was
        exempt from the >250 page penalty and got a magazine domain boost.
        """
        classifier = ProfileClassifier()
        profile = _make_profile(
            total_pages=327,
            image_density=14.6,
            median_image_width=50,
            median_image_height=30,
            avg_text_per_page=791.0,
            image_count=4770,
        )
        diag = FakeDiagnosticReport(
            detected_domain="literature",
            detected_modality="native_digital",
        )
        result = classifier.classify(profile, diag)
        assert result != ProfileType.DIGITAL_MAGAZINE, (
            "Long-form literature must not route as digital_magazine"
        )

    def test_code_heavy_book_editorial(self):
        """Code-heavy book with editorial domain classification.

        Regression: Ayeva Python Design Patterns — editorial domain +
        digital + high page count. Should route as technical_manual.
        """
        classifier = ProfileClassifier()
        profile = _make_profile(
            total_pages=450,
            image_density=0.3,
            median_image_width=100,
            median_image_height=80,
            avg_text_per_page=2000.0,
            image_count=135,
        )
        diag = FakeDiagnosticReport(
            detected_domain="editorial",
            detected_modality="native_digital",
        )
        result = classifier.classify(profile, diag)
        assert result != ProfileType.DIGITAL_MAGAZINE, (
            "Code-heavy editorial book must not route as digital_magazine"
        )

    def test_long_literature_extreme_images(self):
        """300+ page novel with 10+ images/page inline decorative elements."""
        classifier = ProfileClassifier()
        profile = _make_profile(
            total_pages=500,
            image_density=12.0,
            median_image_width=40,
            median_image_height=25,
            avg_text_per_page=1200.0,
            image_count=6000,
        )
        diag = FakeDiagnosticReport(
            detected_domain="literature",
            detected_modality="native_digital",
        )
        result = classifier.classify(profile, diag)
        assert result != ProfileType.DIGITAL_MAGAZINE

    def test_encoding_corrupt_long_code_book(self):
        """Long-form digital code book with encoding corruption.

        Must not misroute just because native text layer exists.
        """
        classifier = ProfileClassifier()
        profile = _make_profile(
            total_pages=380,
            image_density=0.2,
            median_image_width=120,
            median_image_height=90,
            avg_text_per_page=1800.0,
            image_count=76,
        )
        diag = FakeDiagnosticReport(
            detected_domain="technical",
            detected_modality="native_digital",
        )
        result = classifier.classify(profile, diag)
        assert result != ProfileType.DIGITAL_MAGAZINE


# ---------------------------------------------------------------------------
# Positive tests: correct routing
# ---------------------------------------------------------------------------


class TestCorrectRouting:
    """Verify expected profile assignments."""

    def test_genuine_magazine_still_works(self):
        """Short editorial document with large photos should still be magazine."""
        classifier = ProfileClassifier()
        profile = _make_profile(
            total_pages=80,
            image_density=1.5,
            median_image_width=400,
            median_image_height=300,
            avg_text_per_page=800.0,
            image_count=120,
        )
        diag = FakeDiagnosticReport(
            detected_domain="editorial",
            detected_modality="native_digital",
        )
        result = classifier.classify(profile, diag)
        assert result == ProfileType.DIGITAL_MAGAZINE

    def test_short_academic_paper(self):
        """Short text-heavy academic paper."""
        classifier = ProfileClassifier()
        profile = _make_profile(
            total_pages=12,
            image_density=0.1,
            median_image_width=80,
            median_image_height=60,
            avg_text_per_page=3500.0,
            image_count=2,
        )
        diag = FakeDiagnosticReport(
            detected_domain="academic",
            detected_modality="native_digital",
        )
        result = classifier.classify(profile, diag)
        assert result == ProfileType.ACADEMIC_WHITEPAPER

    def test_scanned_document_never_digital(self):
        """Scanned document must never route to a digital profile."""
        classifier = ProfileClassifier()
        profile = _make_profile(
            total_pages=50,
            image_density=0.5,
            median_image_width=200,
            median_image_height=150,
            avg_text_per_page=400.0,
            has_text=False,
            image_count=25,
        )
        diag = FakeDiagnosticReport(
            is_likely_scan=True,
            scan_confidence=0.85,
            detected_domain="technical",
            detected_modality="scanned_clean",
        )
        result = classifier.classify(profile, diag)
        assert result not in (
            ProfileType.DIGITAL_MAGAZINE,
            ProfileType.ACADEMIC_WHITEPAPER,
        )


# ---------------------------------------------------------------------------
# Fallback tests
# ---------------------------------------------------------------------------


class TestFallbackBehavior:
    """Verify modality-aware fallback selects safe profiles."""

    def test_digital_emergency_fallback_is_not_magazine(self):
        """Emergency fallback for digital must NOT be digital_magazine."""
        classifier = ProfileClassifier()
        features = ClassificationFeatures(
            text_density=100.0,
            has_extractable_text=True,
            image_density=0.01,
            median_dim=50,
            image_count=1,
            page_count=3,
            is_scan=False,
            scan_confidence=0.0,
            domain="unknown",
            modality="native_digital",
        )
        result = classifier._emergency_fallback(features)
        assert result != ProfileType.DIGITAL_MAGAZINE

    def test_scanned_emergency_fallback_is_scanned(self):
        """Emergency fallback for scanned stays in scanned modality."""
        classifier = ProfileClassifier()
        features = ClassificationFeatures(
            text_density=100.0,
            has_extractable_text=False,
            image_density=0.5,
            median_dim=200,
            image_count=25,
            page_count=50,
            is_scan=True,
            scan_confidence=0.8,
            domain="unknown",
            modality="scanned_clean",
        )
        result = classifier._emergency_fallback(features)
        assert result == ProfileType.SCANNED
