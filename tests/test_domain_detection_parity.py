"""
Domain Detection Parity Tests
=============================
Validates the V2.7 content-first domain detection logic to prevent
process/batch profile mismatches caused by filename changes.
"""

from pathlib import Path

from mmrag_v2.orchestration.document_diagnostic import (
    ContentDomain,
    DocumentDiagnosticEngine,
    PageDiagnostic,
)


def _page_diag(
    *,
    text_density: float,
    image_coverage: float,
    contains_tables: bool = False,
) -> PageDiagnostic:
    return PageDiagnostic(
        page_number=1,
        text_length=1000,
        text_density=text_density,
        image_count=3,
        image_coverage=image_coverage,
        has_ocr_artifacts=False,
        detected_noise_level=0.0,
        dominant_colors=[],
        contains_tables=contains_tables,
        contains_diagrams=False,
    )


def test_domain_detection_filename_independent_academic() -> None:
    engine = DocumentDiagnosticEngine()
    page_diagnostics = [
        _page_diag(text_density=0.007, image_coverage=0.2),
        _page_diag(text_density=0.007, image_coverage=0.2),
    ]

    academic_name = Path("Hybrid_electric_vehicles_and_their_challenges.pdf")
    generic_name = Path("doc1.pdf")

    domain_academic = engine._estimate_content_domain(academic_name, page_diagnostics)
    domain_generic = engine._estimate_content_domain(generic_name, page_diagnostics)

    assert domain_academic == ContentDomain.ACADEMIC
    assert domain_generic == ContentDomain.ACADEMIC


def test_domain_detection_content_over_filename_editorial() -> None:
    engine = DocumentDiagnosticEngine()
    page_diagnostics = [
        _page_diag(text_density=0.003, image_coverage=0.7),
        _page_diag(text_density=0.003, image_coverage=0.7),
    ]

    misleading_name = Path("research_thesis_analysis.pdf")
    domain = engine._estimate_content_domain(misleading_name, page_diagnostics)

    assert domain == ContentDomain.EDITORIAL


def test_domain_detection_technical_tables_signal() -> None:
    engine = DocumentDiagnosticEngine()
    page_diagnostics = [
        _page_diag(text_density=0.0025, image_coverage=0.2, contains_tables=True),
        _page_diag(text_density=0.0025, image_coverage=0.2, contains_tables=True),
    ]

    generic_name = Path("doc1.pdf")
    domain = engine._estimate_content_domain(generic_name, page_diagnostics)

    assert domain == ContentDomain.TECHNICAL


def test_domain_detection_unknown_when_weak_signals() -> None:
    engine = DocumentDiagnosticEngine()
    page_diagnostics = [
        _page_diag(text_density=0.0002, image_coverage=0.1),
        _page_diag(text_density=0.0002, image_coverage=0.1),
    ]

    generic_name = Path("doc1.pdf")
    domain = engine._estimate_content_domain(generic_name, page_diagnostics)

    assert domain == ContentDomain.UNKNOWN
