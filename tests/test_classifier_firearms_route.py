"""
v2.9 Phase 4 — Firearms route regression tests.

Pins the contract that long-form scanned docs with full-page image
extraction (``image_density>=1.0`` + ``is_scan=true`` + ``page_count>100``)
route to the ``scanned`` (or ``scanned_degraded``) profile, not
``technical_manual``. Background: v2.8 broad reconversion routed
``Firearms`` (292 pages, scanned_degraded modality) to
``technical_manual``, where the chunker's stricter heading-inheritance
dropped HEADING coverage from 100% to 78% (under the 80% gate). Same
content fidelity, just less hierarchy annotation.

Path (a) of the plan: re-route via ``profile_classifier.py`` scorer
adjustment. AGENT-SPATIAL-20 unchanged — the new HARD REJECT is a
numeric threshold on (is_scan, image_density, page_count), not a
profile-specific spatial-threshold branch.

The full corpus verification (HEADING coverage ≥ 0.80 on a v2.9
re-conversion of Firearms) lands in Phase 5 as an env-gated test.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mmrag_v2.orchestration.profile_classifier import (
    ProfileClassifier,
    ProfileType,
)
from mmrag_v2.orchestration.smart_config import DocumentProfile


def _make_profile(
    *,
    total_pages: int,
    image_density: float,
    median_image_width: int = 200,
    median_image_height: int = 300,
    avg_text_per_page: float = 700.0,
    image_count: int = 1000,
    has_text: bool = True,
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


class _Diag:
    def __init__(
        self,
        *,
        is_likely_scan: bool,
        scan_confidence: float,
        detected_domain: str,
        detected_modality: str,
    ) -> None:
        self.physical_check = type(
            "P", (), {
                "is_likely_scan": is_likely_scan,
                "scan_confidence": scan_confidence,
                "detected_modality": type("M", (), {"value": detected_modality})(),
            },
        )()
        self.confidence_profile = type(
            "C", (), {
                "detected_domain": type("D", (), {"value": detected_domain})(),
            },
        )()


def test_firearms_feature_vector_routes_to_scanned() -> None:
    """Firearms's actual v2.8 fresh feature vector → must route to
    ``scanned`` (or ``scanned_degraded``), NOT ``technical_manual``.

    Numbers come from output/Firearms/ingestion.jsonl line 1:
    profile_type=technical_manual (the bug), is_scan=True,
    image_density=1.0, avg_text_per_page=686, total_pages=292,
    domain=editorial.
    """
    profile = _make_profile(
        total_pages=292,
        image_density=1.0,
        median_image_width=600,
        median_image_height=800,
        avg_text_per_page=686.0,
        image_count=292,
    )
    diag = _Diag(
        is_likely_scan=True,
        scan_confidence=0.85,
        detected_domain="editorial",
        detected_modality="scanned_degraded",
    )
    result = ProfileClassifier().classify(profile, diag)
    assert result in (ProfileType.SCANNED, ProfileType.SCANNED_DEGRADED), (
        f"Firearms must route to scanned/scanned_degraded, got {result}"
    )


def test_earthship_still_routes_to_scanned() -> None:
    """Earthship (236pp scanned book, image_density=1.0, editorial domain)
    is the canonical scanned book and MUST route to scanned-class.

    Numbers from output/Earthship_Vol1/ingestion.jsonl line 1.
    """
    profile = _make_profile(
        total_pages=236,
        image_density=1.0,
        median_image_width=600,
        median_image_height=800,
        avg_text_per_page=873.0,
        image_count=236,
    )
    diag = _Diag(
        is_likely_scan=True,
        scan_confidence=0.85,
        detected_domain="editorial",
        detected_modality="scanned_clean",
    )
    result = ProfileClassifier().classify(profile, diag)
    assert result in (ProfileType.SCANNED, ProfileType.SCANNED_DEGRADED), (
        f"Earthship must route to scanned-class, got {result}"
    )


def test_scan0013_still_routes_to_scanned() -> None:
    """SCAN0013 form: a small scanned form (1-2 pages) MUST still route
    to scanned. The HARD REJECT must not flip short scanned docs.
    """
    profile = _make_profile(
        total_pages=2,
        image_density=1.0,
        median_image_width=600,
        median_image_height=800,
        avg_text_per_page=300.0,
        image_count=2,
    )
    diag = _Diag(
        is_likely_scan=True,
        scan_confidence=0.85,
        detected_domain="unknown",
        detected_modality="scanned_clean",
    )
    result = ProfileClassifier().classify(profile, diag)
    assert result in (ProfileType.SCANNED, ProfileType.SCANNED_DEGRADED), (
        f"SCAN0013 must still route to scanned-class, got {result}"
    )


def test_harry_still_routes_to_digital_literature() -> None:
    """HARRY non-regression: a born-digital novel must still route to
    digital_literature. The Phase 4 HARD REJECT requires is_scan=True
    so HARRY's signature (is_scan=False) is untouched.
    """
    profile = _make_profile(
        total_pages=327,
        image_density=14.6,
        median_image_width=50,
        median_image_height=30,
        avg_text_per_page=1370.0,
        image_count=4770,
    )
    diag = _Diag(
        is_likely_scan=False,
        scan_confidence=0.0,
        detected_domain="literature",
        detected_modality="native_digital",
    )
    result = ProfileClassifier().classify(profile, diag)
    assert result == ProfileType.DIGITAL_LITERATURE


_FIREARMS_OUTPUT = Path(__file__).resolve().parent.parent / "output" / "Firearms" / "ingestion.jsonl"


@pytest.mark.skipif(
    os.environ.get("RUN_FIREARMS_VERIFY") != "1",
    reason="Set RUN_FIREARMS_VERIFY=1 after Phase 5 v2.9 re-conversion of Firearms",
)
def test_firearms_heading_coverage_at_least_80pct_post_fix() -> None:
    """Heading coverage gate: after the Phase 4 route fix and the
    Phase 5 v2.9 re-conversion, Firearms's text chunks should hit
    parent_heading coverage >= 0.80 (the v2.8 baseline gate).

    Env-gated because the assertion is only meaningful against a
    v2.9 conversion, not the v2.8 AFTER artifact still on disk.
    """
    if not _FIREARMS_OUTPUT.exists():
        pytest.skip(f"output not found: {_FIREARMS_OUTPUT}")

    import json

    text_chunks = 0
    with_heading = 0
    with _FIREARMS_OUTPUT.open() as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("modality") != "text":
                continue
            text_chunks += 1
            md = rec.get("metadata") or {}
            hierarchy = md.get("hierarchy") or {}
            if hierarchy.get("parent_heading"):
                with_heading += 1

    if text_chunks == 0:
        pytest.skip("no text chunks found")

    coverage = with_heading / text_chunks
    assert coverage >= 0.80, (
        f"Firearms heading coverage {coverage:.2%} < 80% — "
        f"({with_heading}/{text_chunks} chunks have parent_heading)"
    )
