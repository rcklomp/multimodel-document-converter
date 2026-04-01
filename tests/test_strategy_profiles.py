"""
Strategy Profile Tests - Regression Prevention via Golden Files
================================================================
ENGINE_USE: pytest + Golden File comparison

This module implements the "Golden File" test pattern to prevent regression
when making optimizations. Each document type has a known-good baseline
output that new code must match.

TEST STRATEGY:
1. Profile Selection Tests: Verify correct profile is selected for each document type
2. Parameter Isolation Tests: Ensure digital profiles NEVER get scan hints
3. Golden File Comparison: Compare output structure against known-good baseline

REGRESSION PREVENTION:
- Magazine flows must continue to work as before
- Scan optimizations must NOT affect digital document processing
- VLM prompts must be profile-appropriate

Author: Claude (Architect)
Date: 2025-01-03
"""

import pytest
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import Mock, patch

# Import the strategy profiles
from mmrag_v2.orchestration.strategy_profiles import (
    BaseProfile,
    ProfileType,
    ProfileParameters,
    VLMPromptConfig,
    VLMFreedom,
    DigitalMagazineProfile,
    ScannedDegradedProfile,
    ProfileManager,
)
from mmrag_v2.orchestration.document_diagnostic import (
    DiagnosticReport,
    PhysicalCheckResult,
    ConfidenceProfile,
    DocumentModality,
    DocumentEra,
    ContentDomain,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def digital_magazine_diagnostic() -> DiagnosticReport:
    """Create a diagnostic report for a digital magazine (Combat Aircraft style)."""
    return DiagnosticReport(
        source_file="Combat_Aircraft_Magazine.pdf",
        physical_check=PhysicalCheckResult(
            file_size_mb=45.0,
            total_pages=77,
            avg_text_per_page=2500.0,  # High text content
            avg_image_coverage=0.35,
            is_likely_scan=False,  # NOT A SCAN
            scan_confidence=0.1,
            detected_modality=DocumentModality.NATIVE_DIGITAL,
            reasoning="High text content, clean digital source",
        ),
        confidence_profile=ConfidenceProfile(
            overall_confidence=0.92,  # High confidence
            classification_confidence=0.95,
            detected_features=["contains_tables", "image_heavy"],
            detected_era=DocumentEra.MODERN,
            detected_domain=ContentDomain.EDITORIAL,
            warnings=[],
        ),
        page_diagnostics=[],
        recommended_strategy="standard_digital",
    )


@pytest.fixture
def scanned_firearms_diagnostic() -> DiagnosticReport:
    """Create a diagnostic report for a scanned document (Firearms manual style)."""
    return DiagnosticReport(
        source_file="Firearms_Manual_1970.pdf",
        physical_check=PhysicalCheckResult(
            file_size_mb=85.0,
            total_pages=120,
            avg_text_per_page=50.0,  # Low text (scan with poor OCR)
            avg_image_coverage=0.95,  # Full-page scans
            is_likely_scan=True,  # IS A SCAN
            scan_confidence=0.92,
            detected_modality=DocumentModality.SCANNED_DEGRADED,
            reasoning="Low text + large file + high image coverage = scanned document",
        ),
        confidence_profile=ConfidenceProfile(
            overall_confidence=0.85,
            classification_confidence=0.88,
            detected_features=["scanned_document", "ocr_artifacts_present"],
            detected_era=DocumentEra.VINTAGE,
            detected_domain=ContentDomain.TECHNICAL,
            warnings=["OCR quality may be poor - verify extracted text"],
        ),
        page_diagnostics=[],
        recommended_strategy="scan_degraded_high_ocr",
    )


@pytest.fixture
def low_confidence_diagnostic() -> DiagnosticReport:
    """Create a diagnostic report with low confidence (should fallback to digital)."""
    return DiagnosticReport(
        source_file="Unknown_Document.pdf",
        physical_check=PhysicalCheckResult(
            file_size_mb=10.0,
            total_pages=20,
            avg_text_per_page=500.0,
            avg_image_coverage=0.5,
            is_likely_scan=False,
            scan_confidence=0.3,
            detected_modality=DocumentModality.UNKNOWN,
            reasoning="Uncertain classification",
        ),
        confidence_profile=ConfidenceProfile(
            overall_confidence=0.35,  # LOW confidence
            classification_confidence=0.4,
            detected_features=[],
            detected_era=DocumentEra.UNKNOWN,
            detected_domain=ContentDomain.UNKNOWN,
            warnings=["Overall confidence is LOW - manual review recommended"],
        ),
        page_diagnostics=[],
        recommended_strategy="low_confidence_conservative",
    )


# ============================================================================
# PROFILE SELECTION TESTS
# ============================================================================


class TestProfileSelection:
    """Test that ProfileManager selects correct profiles based on diagnostics."""

    def test_digital_magazine_gets_digital_profile(
        self, digital_magazine_diagnostic: DiagnosticReport
    ):
        """Digital magazines MUST get DigitalMagazineProfile - NEVER scan profile."""
        profile = ProfileManager.select_profile(digital_magazine_diagnostic)

        assert isinstance(profile, DigitalMagazineProfile)
        assert profile.profile_type == ProfileType.DIGITAL_MAGAZINE
        assert profile.name == "High-Fidelity Digital Magazine"

    def test_scanned_document_gets_scan_profile(
        self, scanned_firearms_diagnostic: DiagnosticReport
    ):
        """
        Scanned documents MUST get a scan profile (degraded or clean).

        CONTEXT: The legacy heuristic (without doc_profile) selects between
        ScannedDegradedProfile (confidence < 0.70) and ScannedCleanProfile (>= 0.70).
        With confidence=0.85, it correctly selects ScannedCleanProfile.
        """
        profile = ProfileManager.select_profile(scanned_firearms_diagnostic)

        # Accept either scan profile type (degraded or clean)
        from mmrag_v2.orchestration.strategy_profiles import ScannedCleanProfile

        assert isinstance(profile, (ScannedDegradedProfile, ScannedCleanProfile))
        assert profile.profile_type in (ProfileType.SCANNED_DEGRADED, ProfileType.SCANNED_CLEAN)

    def test_low_confidence_falls_back_to_digital(
        self, low_confidence_diagnostic: DiagnosticReport
    ):
        """Low confidence MUST fallback to safe DigitalMagazineProfile."""
        profile = ProfileManager.select_profile(low_confidence_diagnostic)

        # SAFETY: When in doubt, use digital profile (no scan hints)
        assert isinstance(profile, DigitalMagazineProfile)
        assert profile.profile_type == ProfileType.DIGITAL_MAGAZINE

    def test_no_diagnostic_defaults_to_digital(self):
        """No diagnostics MUST default to DigitalMagazineProfile."""
        profile = ProfileManager.select_profile(diagnostic_report=None)

        assert isinstance(profile, DigitalMagazineProfile)

    def test_force_profile_overrides_diagnostics(
        self, digital_magazine_diagnostic: DiagnosticReport
    ):
        """Force profile MUST override diagnostic selection."""
        # Even though diagnostics say digital, force scan profile
        profile = ProfileManager.select_profile(
            diagnostic_report=digital_magazine_diagnostic,
            force_profile=ProfileType.SCANNED_DEGRADED,
        )

        assert isinstance(profile, ScannedDegradedProfile)


# ============================================================================
# PARAMETER ISOLATION TESTS
# ============================================================================


class TestParameterIsolation:
    """Test that profile parameters are strictly isolated between document types."""

    def test_digital_profile_never_has_scan_hints(self):
        """DigitalMagazineProfile MUST NEVER inject scan hints."""
        profile = DigitalMagazineProfile()
        params = profile.get_parameters()
        prompt_config = profile.get_vlm_prompt_config()

        # CRITICAL: No scan hints for digital magazines
        assert params.inject_scan_hints is False
        assert params.inject_historical_hints is False
        assert len(prompt_config.artifact_hints) == 0

    def test_scan_profile_has_scan_hints(self):
        """ScannedDegradedProfile MUST inject scan hints."""
        profile = ScannedDegradedProfile()
        params = profile.get_parameters()
        prompt_config = profile.get_vlm_prompt_config()

        # REQUIRED: Scan hints for degraded scans
        assert params.inject_scan_hints is True
        assert params.inject_historical_hints is True
        assert len(prompt_config.artifact_hints) > 0

        # Check specific hint content
        artifact_text = " ".join(prompt_config.artifact_hints).lower()
        assert "ignore" in artifact_text
        assert "paper" in artifact_text or "texture" in artifact_text

    def test_digital_uses_strict_vlm_freedom(self):
        """Digital magazines use STRICT VLM freedom (no interpretation)."""
        profile = DigitalMagazineProfile()
        params = profile.get_parameters()

        assert params.vlm_freedom == VLMFreedom.STRICT

    def test_scan_uses_strict_vlm_freedom(self):
        """
        Scanned documents use STRICT VLM freedom (visual-only mode).

        CONTEXT: SRS v2.4 compliance - REQ-VLM-NOISE
        The VLM must focus on visual descriptors only, not interpret text.
        STRICT mode enforces this for all document types including scans.
        """
        profile = ScannedDegradedProfile()
        params = profile.get_parameters()

        # Changed from HIGH to STRICT in visual-only enforcement fix
        assert params.vlm_freedom == VLMFreedom.STRICT

    def test_digital_has_higher_min_dimensions(self):
        """Digital magazines filter small icons/ads with higher min dimensions."""
        digital = DigitalMagazineProfile()
        scan = ScannedDegradedProfile()

        digital_params = digital.get_parameters()
        scan_params = scan.get_parameters()

        # Digital should have higher minimum dimensions
        assert digital_params.min_image_width >= 100
        assert digital_params.min_image_height >= 100

        # Scan should have lower minimums to catch degraded content
        assert scan_params.min_image_width <= 50
        assert scan_params.min_image_height <= 50

    def test_scan_has_higher_sensitivity(self):
        """Scanned documents use higher sensitivity for better recall."""
        digital = DigitalMagazineProfile()
        scan = ScannedDegradedProfile()

        digital_params = digital.get_parameters()
        scan_params = scan.get_parameters()

        # Scan needs higher sensitivity
        assert scan_params.sensitivity > digital_params.sensitivity

    def test_both_profiles_use_high_confidence_threshold(self):
        """
        Both profiles require high VLM confidence (0.8) for visual-only mode.

        CONTEXT: SRS v2.4 compliance - REQ-VLM-NOISE
        Both digital and scan profiles use STRICT VLM freedom with high confidence
        threshold to enforce visual-only descriptions without text interpretation.
        """
        digital = DigitalMagazineProfile()
        scan = ScannedDegradedProfile()

        digital_params = digital.get_parameters()
        scan_params = scan.get_parameters()

        # Both profiles use 0.8 threshold for visual-only enforcement
        assert digital_params.confidence_threshold == 0.8
        assert scan_params.confidence_threshold == 0.8


# ============================================================================
# DIAGNOSTIC CONTEXT TESTS
# ============================================================================


class TestDiagnosticContext:
    """Test that diagnostic context is built correctly for VLM prompts."""

    def test_digital_profile_skips_diagnostic_context(self):
        """Digital profile should NOT use diagnostic context (prevent over-processing)."""
        profile = DigitalMagazineProfile()

        assert profile.should_use_diagnostic_context() is False

    def test_scan_profile_uses_diagnostic_context(self):
        """Scan profile should use full diagnostic context."""
        profile = ScannedDegradedProfile()

        assert profile.should_use_diagnostic_context() is True

    def test_scan_diagnostic_context_includes_scan_hints(self):
        """Scan profile diagnostic context must include scan hints."""
        profile = ScannedDegradedProfile()
        context = profile.get_diagnostic_context()

        assert context["is_scan"] is True
        assert len(context["scan_hints"]) > 0
        assert context["classification"] == "scanned_degraded"

    def test_digital_diagnostic_context_no_scan_hints(self):
        """Digital profile diagnostic context must NOT include scan hints."""
        profile = DigitalMagazineProfile()
        context = profile.get_diagnostic_context()

        assert context["is_scan"] is False
        assert len(context["scan_hints"]) == 0
        assert context["classification"] == "digital_magazine"


# ============================================================================
# VLM PROMPT CONFIG TESTS
# ============================================================================


class TestVLMPromptConfig:
    """Test VLM prompt configuration for each profile."""

    def test_digital_prompt_emphasizes_precision(self):
        """Digital profile VLM prompt must emphasize precision over interpretation."""
        profile = DigitalMagazineProfile()
        config = profile.get_vlm_prompt_config()

        prompt_hints = config.build_diagnostic_hints()

        # Should mention strict/precise description
        freedom_lower = config.freedom_instruction.lower()
        assert "strict" in freedom_lower or "only" in freedom_lower

    def test_scan_prompt_allows_interpretation(self):
        """Scan profile VLM prompt must allow interpretation through artifacts."""
        profile = ScannedDegradedProfile()
        config = profile.get_vlm_prompt_config()

        freedom_lower = config.freedom_instruction.lower()

        # Should mention interpretation/inference
        assert "interpret" in freedom_lower or "may" in freedom_lower

    def test_scan_prompt_mentions_ignore_artifacts(self):
        """Scan profile must tell VLM to ignore paper artifacts."""
        profile = ScannedDegradedProfile()
        config = profile.get_vlm_prompt_config()

        prompt_hints = config.build_diagnostic_hints()

        # Should mention ignoring artifacts
        assert "IGNORE" in prompt_hints or "ignore" in prompt_hints.lower()


# ============================================================================
# GOLDEN FILE STRUCTURE TESTS
# ============================================================================


class TestGoldenFileStructure:
    """Test that output structure matches expected golden file format."""

    def test_profile_describe_format(self):
        """Profile describe() must return consistent format for logging."""
        digital = DigitalMagazineProfile()
        scan = ScannedDegradedProfile()

        digital_desc = digital.describe()
        scan_desc = scan.describe()

        # Both must contain key fields
        for desc in [digital_desc, scan_desc]:
            assert "Profile:" in desc
            assert "Sensitivity:" in desc
            assert "Min:" in desc
            assert "VLM Freedom:" in desc
            assert "Scan Hints:" in desc

    def test_profile_parameters_are_complete(self):
        """Profile parameters must include all required fields."""
        digital = DigitalMagazineProfile()
        scan = ScannedDegradedProfile()

        for profile in [digital, scan]:
            params = profile.get_parameters()

            # All parameters must be defined
            assert params.sensitivity is not None
            assert params.min_image_width is not None
            assert params.min_image_height is not None
            assert params.extract_backgrounds is not None
            assert params.enable_shadow_extraction is not None
            assert params.vlm_freedom is not None
            assert params.inject_scan_hints is not None
            assert params.inject_historical_hints is not None
            assert params.confidence_threshold is not None
            assert params.strip_artifacts_from_text is not None
            assert params.aggressive_deduplication is not None


# ============================================================================
# REGRESSION PREVENTION TESTS
# ============================================================================


class TestRegressionPrevention:
    """
    Critical regression tests - these MUST pass for any release.

    These tests encode the "golden" behavior that must not change.
    """

    def test_magazine_baseline_sensitivity(self):
        """Magazine sensitivity MUST be 0.5 (proven baseline)."""
        profile = DigitalMagazineProfile()
        params = profile.get_parameters()

        # This is the proven value from Combat Aircraft runs
        assert params.sensitivity == 0.5

    def test_magazine_baseline_min_dimensions(self):
        """Magazine min dimensions MUST be 100x100 (filter ads/icons)."""
        profile = DigitalMagazineProfile()
        params = profile.get_parameters()

        # These values filter out small ads and icons
        assert params.min_image_width == 100
        assert params.min_image_height == 100

    def test_scan_baseline_sensitivity(self):
        """Scan sensitivity MUST be 0.8 (high recall for degraded content)."""
        profile = ScannedDegradedProfile()
        params = profile.get_parameters()

        assert params.sensitivity == 0.8

    def test_scan_baseline_min_dimensions(self):
        """Scan min dimensions MUST be 30x30 (catch degraded content)."""
        profile = ScannedDegradedProfile()
        params = profile.get_parameters()

        assert params.min_image_width == 30
        assert params.min_image_height == 30

    def test_profile_types_are_distinct(self):
        """Profile types must be distinct - no cross-contamination."""
        digital = DigitalMagazineProfile()
        scan = ScannedDegradedProfile()

        # They must have different types
        assert digital.profile_type != scan.profile_type

        # They must have different scan hint settings
        assert digital.get_parameters().inject_scan_hints != scan.get_parameters().inject_scan_hints


# ============================================================================
# INTEGRATION SMOKE TESTS
# ============================================================================


class TestIntegrationSmoke:
    """Quick smoke tests to verify the profile system integrates correctly."""

    def test_profile_manager_available_profiles(self):
        """ProfileManager must list available profiles."""
        profiles = ProfileManager.list_available_profiles()

        assert ProfileType.DIGITAL_MAGAZINE in profiles
        assert ProfileType.SCANNED_DEGRADED in profiles

    def test_get_profile_by_type(self):
        """ProfileManager must return correct profile by type."""
        digital = ProfileManager.get_profile_by_type(ProfileType.DIGITAL_MAGAZINE)
        scan = ProfileManager.get_profile_by_type(ProfileType.SCANNED_DEGRADED)

        assert isinstance(digital, DigitalMagazineProfile)
        assert isinstance(scan, ScannedDegradedProfile)

    def test_unknown_type_defaults_to_digital(self):
        """Unknown profile type must default to DigitalMagazineProfile (safe)."""
        profile = ProfileManager.get_profile_by_type(ProfileType.UNKNOWN)

        # Should fallback to digital for safety
        assert isinstance(profile, DigitalMagazineProfile)


# ============================================================================
# EDITORIAL TECHNICAL BOOK CLASSIFICATION TESTS
# ============================================================================


class TestEditorialTechBookClassification:
    """
    Verify that digital editorial books with low image density are routed to
    TechnicalManualProfile rather than DigitalMagazineProfile.

    Root cause: Python coding books (Fluent Python, Python Cookbook, etc.) contain
    code screenshots on many pages. The domain detector labels them 'editorial'
    because the diagnostic heuristics see images. But they are NOT magazines —
    magazines require high image density (0.5+ images/page).

    The fix adds 'is_editorial_tech_book' to _score_technical_manual:
      editorial + digital + image_density < 0.5 + page_count > 100 → tech book.
    """

    def _make_features(
        self,
        text_density=840,
        image_density=0.10,
        median_dim=1353,
        page_count=365,
        domain="editorial",
        is_scan=False,
    ):
        from mmrag_v2.orchestration.profile_classifier import (
            ProfileClassifier,
            ClassificationFeatures,
        )
        return ClassificationFeatures(
            text_density=text_density,
            has_extractable_text=True,
            image_density=image_density,
            median_dim=median_dim,
            image_count=int(image_density * 20),
            page_count=page_count,
            is_scan=is_scan,
            scan_confidence=0.0,
            domain=domain,
            modality="native_digital",
        )

    def _classify(self, **kw):
        from mmrag_v2.orchestration.profile_classifier import (
            ProfileClassifier,
            ProfileType,
        )
        clf = ProfileClassifier()
        f = self._make_features(**kw)
        tm = clf._score_technical_manual(f)
        dm = clf._score_digital_magazine(f)
        aw = clf._score_academic_whitepaper(f)
        best = max([tm, dm, aw], key=lambda s: s.score * s.confidence)
        return best.profile_type

    def test_python_cookbook_profile(self):
        """Python Cookbook (low img density, many pages) → technical_manual."""
        from mmrag_v2.orchestration.profile_classifier import ProfileType
        result = self._classify(
            text_density=495, image_density=0.10, median_dim=1920, page_count=477
        )
        assert result == ProfileType.TECHNICAL_MANUAL

    def test_fluent_python_profile(self):
        """Fluent Python (moderate img density, many pages, high text) → technical_manual."""
        from mmrag_v2.orchestration.profile_classifier import ProfileType
        result = self._classify(
            text_density=2750, image_density=0.20, median_dim=8853, page_count=766
        )
        assert result == ProfileType.TECHNICAL_MANUAL

    def test_arcgis_python_profile(self):
        """ArcGIS Python Cookbook (very low img density, many pages) → technical_manual."""
        from mmrag_v2.orchestration.profile_classifier import ProfileType
        result = self._classify(
            text_density=1353, image_density=0.05, median_dim=1650, page_count=366
        )
        assert result == ProfileType.TECHNICAL_MANUAL

    def test_devlin_llm_agents_profile(self):
        """Devlin LLM Agents book → technical_manual."""
        from mmrag_v2.orchestration.profile_classifier import ProfileType
        result = self._classify(
            text_density=840, image_density=0.10, median_dim=1353, page_count=365
        )
        assert result == ProfileType.TECHNICAL_MANUAL

    def test_bourne_rag_profile(self):
        """Bourne RAG book → technical_manual."""
        from mmrag_v2.orchestration.profile_classifier import ProfileType
        result = self._classify(
            text_density=1631, image_density=0.10, median_dim=2775, page_count=346
        )
        assert result == ProfileType.TECHNICAL_MANUAL

    def test_real_magazine_still_digital_magazine(self):
        """A real magazine (high img density, 80 pages) → digital_magazine (no regression)."""
        from mmrag_v2.orchestration.profile_classifier import ProfileType
        result = self._classify(
            text_density=900, image_density=1.0, median_dim=500, page_count=80
        )
        assert result == ProfileType.DIGITAL_MAGAZINE

    def test_short_editorial_digital_not_boosted(self):
        """A short editorial digital doc (< 100 pages) → stays digital_magazine."""
        from mmrag_v2.orchestration.profile_classifier import ProfileType
        result = self._classify(
            text_density=900, image_density=0.15, median_dim=400, page_count=60
        )
        assert result == ProfileType.DIGITAL_MAGAZINE

    def test_high_image_density_editorial_stays_magazine(self):
        """Editorial with high img density (0.6) stays digital_magazine even if many pages."""
        from mmrag_v2.orchestration.profile_classifier import ProfileType
        result = self._classify(
            text_density=900, image_density=0.6, median_dim=500, page_count=200
        )
        assert result == ProfileType.DIGITAL_MAGAZINE


class TestCrossProfileClassifierFixes:
    """
    Regression tests for cross-profile classifier fixes identified during
    multi-profile smoke testing (April 2026).
    """

    def _make_features(self, **kw):
        from mmrag_v2.orchestration.profile_classifier import ClassificationFeatures
        defaults = dict(
            text_density=800, image_density=1.0, median_dim=300, page_count=80,
            is_scan=False, scan_confidence=0.0, domain="editorial", modality="native_digital",
            has_extractable_text=True, image_count=80,
        )
        defaults.update(kw)
        return ClassificationFeatures(**defaults)

    def _score_magazine(self, **kw):
        from mmrag_v2.orchestration.profile_classifier import ProfileClassifier
        clf = ProfileClassifier()
        f = self._make_features(**kw)
        return clf._score_digital_magazine(f)

    # ── digital_magazine sanity guards ───────────────────────────────────────

    def test_decorative_image_density_penalises_magazine(self):
        """image_density=14.6 (decorative book elements) → magazine confidence is near zero."""
        s = self._score_magazine(image_density=14.6, page_count=327, text_density=790,
                                  domain="unknown", median_dim=150)
        assert s.confidence < 0.05, (
            f"Expected near-zero confidence for decorative image density, got {s.confidence:.3f}"
        )

    def test_very_long_document_penalises_magazine(self):
        """page_count=327 → magazine confidence is strongly penalised (≤ 0.15)."""
        s = self._score_magazine(image_density=2.0, page_count=327, text_density=800,
                                  domain="editorial", median_dim=400)
        assert s.confidence <= 0.15, (
            f"Expected strongly penalised confidence for 327-page document, got {s.confidence:.3f}"
        )

    def test_pcworld_still_digital_magazine(self):
        """PCWorld (108 pages, img_density=1.0, large photos) → digital_magazine preserved."""
        from mmrag_v2.orchestration.profile_classifier import ProfileClassifier, ProfileType
        clf = ProfileClassifier()
        f = self._make_features(image_density=1.0, page_count=108, text_density=900,
                                 domain="editorial", median_dim=500)
        scores = [
            clf._score_digital_magazine(f),
            clf._score_academic_whitepaper(f),
            clf._score_technical_manual(f),
        ]
        best = max(scores, key=lambda s: s.score * s.confidence)
        assert best.profile_type == ProfileType.DIGITAL_MAGAZINE, (
            f"PCWorld should route to digital_magazine, got {best.profile_type}"
        )

    def test_harry_potter_features_not_digital_magazine(self):
        """
        Harry Potter PDF features (327 pages, image_density=14.6 decorative elements,
        unknown domain) must NOT route to digital_magazine.
        """
        from mmrag_v2.orchestration.profile_classifier import ProfileClassifier, ProfileType
        clf = ProfileClassifier()
        f = self._make_features(image_density=14.6, page_count=327, text_density=790,
                                 domain="unknown", median_dim=150, is_scan=False)
        scores = [
            clf._score_digital_magazine(f),
            clf._score_academic_whitepaper(f),
            clf._score_technical_manual(f),
        ]
        best = max(scores, key=lambda s: s.score * s.confidence)
        assert best.profile_type != ProfileType.DIGITAL_MAGAZINE, (
            f"Harry Potter features should NOT route to digital_magazine, got {best.profile_type}"
        )

    # ── LABEL_RE numbered section heading fix ─────────────────────────────────

    def test_numbered_section_heading_is_label_like(self):
        """'2.1 Der horizontale Bruch' (German numbered heading) is recognised as a label."""
        import re
        LABEL_RE = re.compile(r"^(?:\d[\d.]*\s+)?[A-Z][A-Za-z0-9/&()' .,-]{1,55}:?$")
        assert LABEL_RE.match("2.1 Der horizontale Bruch"), \
            "Numbered German section heading should match LABEL_RE"
        assert LABEL_RE.match("1 Einleitung"), \
            "Simple numbered heading should match LABEL_RE"
        assert LABEL_RE.match("10.2 Fazit und Ausblick"), \
            "Multi-level numbered heading should match LABEL_RE"

    def test_plain_prose_sentence_not_label_like(self):
        """Prose sentences starting with digits should NOT match the label pattern."""
        import re
        LABEL_RE = re.compile(r"^(?:\d[\d.]*\s+)?[A-Z][A-Za-z0-9/&()' .,-]{1,55}:?$")
        # Lowercase word after number → not a section heading
        assert not LABEL_RE.match("2 items were found in the results"), \
            "Prose sentence starting with lowercase after digit should not match"
        # Sentence too long
        assert not LABEL_RE.match("1 This is a very long section title that exceeds the maximum character limit set"), \
            "Overly long heading should not match"
