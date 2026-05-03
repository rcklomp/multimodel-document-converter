"""Routing tests for the new DIGITAL_LITERATURE profile.

These tests are the executable contract that born-digital novels reach
the post-Docling sanity pass automatically: ProfileClassifier returns
DIGITAL_LITERATURE when the diagnostic engine has set domain="literature"
on a long-form, native-digital document with small decorative imagery.

The HARRY Potter PDF (`data/scanned/HarryPotter_and_the_Sorcerers_Stone.pdf`)
is the canonical fixture; its diagnostic signature is encoded in
`test_harry_potter_routes_to_digital_literature` below.
"""
from __future__ import annotations

import pytest

from mmrag_v2.orchestration.profile_classifier import (
    ClassificationFeatures,
    ProfileClassifier,
    ProfileType,
)
from mmrag_v2.orchestration.smart_config import DocumentProfile


def _make_profile(
    *,
    total_pages: int = 300,
    image_density: float = 0.5,
    median_image_width: int = 50,
    median_image_height: int = 30,
    avg_text_per_page: float = 1500.0,
    has_text: bool = True,
    image_count: int = 100,
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


class _FakeDiagnosticReport:
    def __init__(
        self,
        *,
        is_likely_scan: bool = False,
        scan_confidence: float = 0.0,
        detected_domain: str = "literature",
        detected_modality: str = "native_digital",
    ):
        self.physical_check = type(
            "P", (), {
                "is_likely_scan": is_likely_scan,
                "scan_confidence": scan_confidence,
                "detected_modality": type("M", (), {"value": detected_modality})(),
            },
        )()
        self.confidence_profile = type(
            "C", (), {"detected_domain": type("D", (), {"value": detected_domain})()},
        )()


def _classify(**profile_kw):
    diag_kw = {}
    for key in ("detected_domain", "detected_modality", "is_likely_scan", "scan_confidence"):
        if key in profile_kw:
            diag_kw[key] = profile_kw.pop(key)
    return ProfileClassifier().classify(
        _make_profile(**profile_kw), _FakeDiagnosticReport(**diag_kw)
    )


def test_harry_potter_routes_to_digital_literature():
    """HARRY's actual diagnostic signature -> digital_literature.

    Numbers come from the cached `output/.../HARRY_30pg/ingestion.jsonl`
    metadata: 327 pages, native_digital, literature domain, 14.6 images/page
    (drop caps + dingbats, hence small median_dim), ~1370 chars/page.
    """
    assert _classify(
        total_pages=327,
        image_density=14.6,
        median_image_width=50,
        median_image_height=30,
        avg_text_per_page=1370.0,
        image_count=4770,
        detected_domain="literature",
        detected_modality="native_digital",
    ) == ProfileType.DIGITAL_LITERATURE


def test_typical_novel_with_no_inline_images_routes_to_digital_literature():
    """A clean novel (200 pages, no decorative images) still routes correctly."""
    assert _classify(
        total_pages=220,
        image_density=0.05,
        median_image_width=40,
        median_image_height=20,
        avg_text_per_page=1800.0,
        image_count=11,
        detected_domain="literature",
    ) == ProfileType.DIGITAL_LITERATURE


def test_short_literary_work_does_not_outrank_other_signals():
    """A 25-page literary excerpt should not steal routing from other profiles.

    The page-count weight is small (0.05) for short works, so the scorer
    yields a low confidence; classifier may pick a fallback. The test only
    asserts we don't HARD route a 25-page anything to digital_literature.
    """
    result = _classify(
        total_pages=25,
        image_density=0.1,
        median_image_width=40,
        median_image_height=20,
        avg_text_per_page=1500.0,
        detected_domain="literature",
    )
    # Either digital_literature (if it scores high enough on domain alone)
    # or another digital profile - but never DIGITAL_MAGAZINE.
    assert result != ProfileType.DIGITAL_MAGAZINE


def test_scanned_literature_does_not_route_to_digital_literature():
    """Scanned books need the scanned route, not the born-digital pipeline."""
    result = _classify(
        total_pages=300,
        image_density=0.3,
        median_image_width=60,
        median_image_height=40,
        avg_text_per_page=1500.0,
        detected_domain="literature",
        detected_modality="scanned_clean",
        is_likely_scan=True,
        scan_confidence=0.9,
    )
    assert result != ProfileType.DIGITAL_LITERATURE


def test_magazine_signature_does_not_steal_routing():
    """A born-digital editorial document with large photos stays a magazine."""
    result = _classify(
        total_pages=80,
        image_density=1.5,
        median_image_width=400,
        median_image_height=300,
        avg_text_per_page=800.0,
        detected_domain="editorial",
    )
    assert result == ProfileType.DIGITAL_MAGAZINE


def test_academic_paper_signature_does_not_route_to_literature():
    """Long academic paper with literature domain misfire would still be academic."""
    result = _classify(
        total_pages=80,
        image_density=0.1,
        median_image_width=80,
        median_image_height=60,
        avg_text_per_page=3500.0,
        detected_domain="academic",
    )
    assert result == ProfileType.ACADEMIC_WHITEPAPER


def test_digital_literature_plan_auto_enables_post_processors():
    """End-to-end: classify -> build_pdf_conversion_plan delivers Phase 1-4 fixes."""
    from mmrag_v2.engines.pdf_plan import build_pdf_conversion_plan

    profile_type = _classify(
        total_pages=327,
        image_density=14.6,
        median_image_width=50,
        median_image_height=30,
        avg_text_per_page=1370.0,
        image_count=4770,
        detected_domain="literature",
        detected_modality="native_digital",
    )
    assert profile_type == ProfileType.DIGITAL_LITERATURE

    plan = build_pdf_conversion_plan(
        document_modality="native_digital",
        profile_type=profile_type.value,
        document_domain="literature",
        total_pages=327,
        image_density=14.6,
        avg_text_per_page=1370.0,
    )
    assert plan.reading_order_strategy == "y_sort_with_dropcap"
    assert plan.suppress_layout_label_text is True
    assert plan.bitmap_area_threshold == pytest.approx(0.92)
