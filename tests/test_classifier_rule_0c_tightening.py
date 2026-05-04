"""
v2.9 Phase 3 — Rule 0c (literature dialogue lane) tightening.

Pins the contract that the +0.4 weak-dialogue contribution to
``content_score_literature`` is suppressed when the sampled pages show
clear code patterns (line-starting Python keywords or fenced blocks)
on 2+ pages. Programming books like Ayeva's "Mastering Python Design
Patterns" used to misroute to ``digital_literature`` because Python
f-strings and string literals contain quote characters that look like
dialogue under the cheap heuristic.

Background: v2.8 broad reconversion's ``Ayeva_Python_Patterns`` ran
``profile_type=digital_literature`` and CODE FAIL at
``indentation_fidelity=0.83`` (just under the 0.85 hard gate).
Probe ``output/ayeva_qa_20260501/`` (2026-05-01) had instead routed to
``technical_manual`` with ``indentation_fidelity=0.93`` because the
classifier was driven differently. v2.9 Phase 3 closes the loop in
``document_diagnostic.py`` so the diagnostic engine itself stops
mis-classifying code-heavy books as literature.

The tests synthesize ``PageDiagnostic`` lists and call
``DocumentDiagnosticEngine._estimate_content_domain`` directly so the
contract is reproducible without re-running PyMuPDF on real PDFs.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from mmrag_v2.orchestration.document_diagnostic import (
    ContentDomain,
    DocumentDiagnosticEngine,
    PageDiagnostic,
)


def _engine() -> DocumentDiagnosticEngine:
    return DocumentDiagnosticEngine()


def _harry_signature_diagnostics() -> list[PageDiagnostic]:
    """5 pages of HARRY-class dialogue prose with NO code keyword starts.

    text_density: ~0.0028 ≈ 1370 chars on 612x792 page (HARRY's profile).
    page_text_sample: prose with double quotes (dialogue) but no Python.
    """
    sample_with_dialogue = (
        '"Hello there," said the wizard.\n'
        'Harry looked up. "Who are you?"\n'
        '"You may call me the Headmaster."\n'
        'A pause. "But why are you here?"\n'
    )
    sample_plain = (
        "The morning was bright and full of birdsong. He walked\n"
        "down the long lane toward the old castle, his boots crunching\n"
        "against the gravel underfoot.\n"
    )
    return [
        PageDiagnostic(
            page_number=i,
            text_length=400,
            text_density=0.00283,  # ~1370 chars on 612*792 = avg_text_per_page
            image_count=2,
            image_coverage=0.05,
            has_ocr_artifacts=False,
            detected_noise_level=0.0,
            page_text_sample=sample_with_dialogue if i % 2 == 1 else sample_plain,
        )
        for i in range(1, 6)
    ]


def _ayeva_signature_diagnostics() -> list[PageDiagnostic]:
    """5 pages of Ayeva-class output: front matter (no code) + body (Python).

    Mirrors the actual sampled-page contents — first page empty/title,
    pages 75/149/222 contain class/def starts, page 296 is back matter.
    Uses real-world quote density that misled v2.8 (Python f-strings,
    string literals).
    """
    title_sample = ""
    code_sample_1 = (
        "Creational Design Patterns\n"
        "class WizardCharacter:\n"
        "    def interact_with(self, obstacle):\n"
        '        msg = f"{self} battles {obstacle}!"\n'
        '        print(msg)\n'
    )
    code_sample_2 = (
        "Behavioral Design Patterns\n"
        "First, we define the Observer interface, which holds an update.\n"
        "class Subject:\n"
        '    """The thing being observed."""\n'
        "    def attach(self, observer):\n"
    )
    code_sample_3 = (
        'persons = []\n'
        "for _ in range(0, 20):\n"
        '    p = {"firstname": fake.first_name()}\n'
        "    persons.append(p)\n"
    )
    backmatter = "Download a free PDF copy of this book\nThanks for purchasing this book!\n"
    samples = [title_sample, code_sample_1, code_sample_2, code_sample_3, backmatter]
    return [
        PageDiagnostic(
            page_number=i + 1,
            text_length=len(samples[i]),
            text_density=0.00283,
            image_count=0,
            image_coverage=0.0,
            has_ocr_artifacts=False,
            detected_noise_level=0.0,
            page_text_sample=samples[i],
        )
        for i in range(5)
    ]


def _chaubal_signature_diagnostics() -> list[PageDiagnostic]:
    """PyTorch book sample — multiple pages with Python code starts."""
    code_sample = (
        "import torch\n"
        "from torch import nn\n"
        "class CNN(nn.Module):\n"
        "    def forward(self, x):\n"
        "        return self.layers(x)\n"
    )
    return [
        PageDiagnostic(
            page_number=i + 1,
            text_length=len(code_sample),
            text_density=0.00283,
            image_count=0,
            image_coverage=0.0,
            has_ocr_artifacts=False,
            detected_noise_level=0.0,
            page_text_sample=code_sample if i >= 1 else "",
        )
        for i in range(5)
    ]


def _fluent_python_signature_diagnostics() -> list[PageDiagnostic]:
    """Fluent Python — code-heavy book; non-regression control.

    Uses code without heavy string literals so the fixture exercises the
    code-evidence guard, not the full-novel +0.8 dialogue lane (which
    is independently triggered by ``_dq >= 4`` per page; Python triple-
    quoted docstrings + several short string literals can push a page
    over that threshold even when the page is clearly code).
    """
    code_sample = (
        "def fizzbuzz(n):\n"
        "    for i in range(1, n):\n"
        "        if i % 15 == 0:\n"
        "            yield i\n"
        "import sys\n"
        "from collections import deque\n"
    )
    return [
        PageDiagnostic(
            page_number=i + 1,
            text_length=len(code_sample),
            text_density=0.00283,
            image_count=0,
            image_coverage=0.0,
            has_ocr_artifacts=False,
            detected_noise_level=0.0,
            page_text_sample=code_sample if i >= 1 else "",
        )
        for i in range(5)
    ]


def _stub_total_pages(total: int):
    """Patch ``fitz.open`` so ``_estimate_content_domain`` reports
    ``total`` pages without needing a real PDF on disk."""

    class _StubDoc:
        def __init__(self, total: int) -> None:
            self._total = total

        def __len__(self) -> int:
            return self._total

        def close(self) -> None:
            pass

    return patch(
        "mmrag_v2.orchestration.document_diagnostic.fitz.open",
        return_value=_StubDoc(total),
    )


def test_rule_0c_fires_for_harry_dialogue_low_code() -> None:
    """HARRY signature: dialogue + zero code keyword starts → literature
    domain returned. The +0.4 contribution is the load-bearing signal
    for HARRY since its dialogue ratio (2/5 = 0.4) is under the
    +0.8 cutoff (>0.3 + >50pp) only because total_pages=29 in the
    30-page test slice."""
    engine = _engine()
    diagnostics = _harry_signature_diagnostics()
    with _stub_total_pages(29):
        domain = engine._estimate_content_domain(Path("dummy.pdf"), diagnostics)
    # Even with low total_pages, dialogue + no code → literature wins
    # vs other domains (technical/editorial/academic all score 0).
    assert domain == ContentDomain.LITERATURE


def test_rule_0c_suppressed_for_ayeva_code_heavy_book() -> None:
    """Ayeva signature: dialogue-looking quotes BUT 3 code-evidence
    pages → literature contribution suppressed. The fallback domain
    determined by the rest of the rules MUST NOT be literature."""
    engine = _engine()
    diagnostics = _ayeva_signature_diagnostics()
    with _stub_total_pages(296):
        domain = engine._estimate_content_domain(Path("dummy.pdf"), diagnostics)
    assert domain != ContentDomain.LITERATURE


def test_rule_0c_suppressed_for_chaubal_pytorch_book() -> None:
    """Chaubal signature: clear class/def/import keyword starts on
    multiple pages → literature contribution suppressed."""
    engine = _engine()
    diagnostics = _chaubal_signature_diagnostics()
    with _stub_total_pages(359):
        domain = engine._estimate_content_domain(Path("dummy.pdf"), diagnostics)
    assert domain != ContentDomain.LITERATURE


def test_rule_0c_suppressed_for_fluent_python_book() -> None:
    """Fluent Python signature: another code-heavy book; the
    classifier still routes it to a non-literature domain. Pinned to
    catch any regression that loosens the code-evidence threshold."""
    engine = _engine()
    diagnostics = _fluent_python_signature_diagnostics()
    with _stub_total_pages(770):
        domain = engine._estimate_content_domain(Path("dummy.pdf"), diagnostics)
    assert domain != ContentDomain.LITERATURE


def test_ayeva_routes_to_technical_manual_post_fix() -> None:
    """Full classifier output on Ayeva-shape input: with the diagnostic
    domain returning ``technical`` (or non-literature), the
    ProfileClassifier should route to ``technical_manual`` for a
    long-form code-heavy native-digital book.

    This crosses the diagnostic→classifier boundary so it would have
    failed under v2.8 (literature → DIGITAL_LITERATURE).
    """
    from mmrag_v2.orchestration.profile_classifier import (
        ProfileClassifier,
        ProfileType,
    )
    from mmrag_v2.orchestration.smart_config import DocumentProfile

    engine = _engine()
    diagnostics = _ayeva_signature_diagnostics()
    with _stub_total_pages(296):
        domain = engine._estimate_content_domain(Path("dummy.pdf"), diagnostics)
    assert domain != ContentDomain.LITERATURE

    profile = DocumentProfile(
        total_pages=296,
        image_density=0.05,
        median_image_width=50,
        median_image_height=30,
        avg_text_per_page=1500.0,
        has_text=True,
        image_count=15,
    )

    class _Diag:
        def __init__(self, dom_value: str) -> None:
            self.physical_check = type(
                "P", (), {
                    "is_likely_scan": False,
                    "scan_confidence": 0.0,
                    "detected_modality": type("M", (), {"value": "native_digital"})(),
                },
            )()
            self.confidence_profile = type(
                "C", (), {
                    "detected_domain": type("D", (), {"value": dom_value})(),
                },
            )()

    profile_type = ProfileClassifier().classify(profile, _Diag(domain.value))
    assert profile_type == ProfileType.TECHNICAL_MANUAL


def test_harry_routes_to_digital_literature_post_fix() -> None:
    """Non-regression: HARRY signature still ends up in
    ``digital_literature``. The Phase 3 fix must not flip canonical
    novels."""
    from mmrag_v2.orchestration.profile_classifier import (
        ProfileClassifier,
        ProfileType,
    )
    from mmrag_v2.orchestration.smart_config import DocumentProfile

    engine = _engine()
    diagnostics = _harry_signature_diagnostics()
    with _stub_total_pages(327):
        domain = engine._estimate_content_domain(Path("dummy.pdf"), diagnostics)
    assert domain == ContentDomain.LITERATURE

    profile = DocumentProfile(
        total_pages=327,
        image_density=14.6,
        median_image_width=50,
        median_image_height=30,
        avg_text_per_page=1370.0,
        has_text=True,
        image_count=4770,
    )

    class _Diag:
        physical_check = type(
            "P", (), {
                "is_likely_scan": False,
                "scan_confidence": 0.0,
                "detected_modality": type("M", (), {"value": "native_digital"})(),
            },
        )()
        confidence_profile = type(
            "C", (), {
                "detected_domain": type("D", (), {"value": "literature"})(),
            },
        )()

    profile_type = ProfileClassifier().classify(profile, _Diag())
    assert profile_type == ProfileType.DIGITAL_LITERATURE
