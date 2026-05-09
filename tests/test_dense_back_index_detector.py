"""Phase 4 Step 2 — content-shape back-index detector regression tests.

Covers `_is_back_index_page_by_lines` (pure-string heuristic) and
`_classify_dense_back_index_pages_by_source` (source-PDF traversal).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from mmrag_v2.processor import (
    _BACK_INDEX_ENTRY_RE,
    _BACK_INDEX_MARKER_RE,
    _classify_dense_back_index_pages_by_source,
    _extract_pdf_page_lines,
    _is_back_index_page_by_lines,
)


# ---------------------------------------------------------------------------
# Pattern-level regex sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "Argilla platform for A/B testing, 157",
        "evolution to Agentic RAG, 45-55",
        "AIOps, 257, 260, 268",
        "talent and culture, 246-250",
        "Cloud Build, 225",
        "BLEU (Bilingual Evaluation Understudy), 17, 160",
    ],
)
def test_back_index_entry_re_matches_real_entries(line: str) -> None:
    assert _BACK_INDEX_ENTRY_RE.match(line) is not None


@pytest.mark.parametrize(
    "line",
    [
        "This is a normal paragraph of body text without trailing digit.",
        "Section 3.2.1 introduces the concept",  # no comma+digit shape
        "Figure 4 shows the architecture diagram for the proposed system.",
        "    indented heading line",
        "5J()lll t-I::> Nult: 1",  # corruption fragments must NOT match
    ],
)
def test_back_index_entry_re_rejects_non_entries(line: str) -> None:
    assert _BACK_INDEX_ENTRY_RE.match(line) is None


@pytest.mark.parametrize(
    "marker",
    [
        "280 | Index",
        "Index | 281",
        "Index",
    ],
)
def test_back_index_marker_re_matches(marker: str) -> None:
    assert _BACK_INDEX_MARKER_RE.match(marker) is not None


@pytest.mark.parametrize(
    "marker",
    [
        "Page 280",
        "Sub-Index",
        "280",
    ],
)
def test_back_index_marker_re_rejects_non_markers(marker: str) -> None:
    assert _BACK_INDEX_MARKER_RE.match(marker) is None


# ---------------------------------------------------------------------------
# Page classification — _is_back_index_page_by_lines
# ---------------------------------------------------------------------------


def _entries(n: int) -> list[str]:
    return [f"topic {i}, {100 + i}" for i in range(n)]


def test_classifier_below_min_lines_returns_false() -> None:
    # 19 lines is just under the 20-line floor; the marker alone shouldn't promote.
    lines = _entries(18) + ["280 | Index"]  # 19 lines total
    assert _is_back_index_page_by_lines(lines) is False


def test_classifier_high_ratio_no_marker_passes() -> None:
    # 30 lines, all index entries — ratio 1.0 >= 0.65 even with no marker.
    lines = _entries(30)
    assert _is_back_index_page_by_lines(lines) is True


def test_classifier_mid_ratio_with_marker_passes() -> None:
    # 60% entries + Index marker → passes via the marker-assisted gate.
    lines = _entries(60) + ["random body line"] * 40 + ["280 | Index"]
    assert _is_back_index_page_by_lines(lines) is True


def test_classifier_mid_ratio_no_marker_fails() -> None:
    # Same ratio without an Index marker → does NOT pass.
    lines = _entries(60) + ["random body line"] * 40
    assert _is_back_index_page_by_lines(lines) is False


def test_classifier_low_ratio_with_marker_fails() -> None:
    # Marker present but only 10% of lines look like entries → no false fire.
    lines = _entries(10) + ["body sentence with no comma-digit shape"] * 90 + ["Index"]
    assert _is_back_index_page_by_lines(lines) is False


def test_classifier_body_text_safety() -> None:
    body = [
        "Chapter 4 introduces the agent runtime model.",
        "Each agent owns its tool registry and is invoked by the orchestrator.",
        "The session manager maintains conversation history across turns.",
        "Memory is persisted to BigQuery via the embedding pipeline.",
    ] * 10
    assert _is_back_index_page_by_lines(body) is False


# ---------------------------------------------------------------------------
# Source-PDF classifier — _classify_dense_back_index_pages_by_source
# ---------------------------------------------------------------------------


def test_classify_returns_empty_when_pdf_path_is_none() -> None:
    assert _classify_dense_back_index_pages_by_source(None, total_pages=10, exclude=set()) == set()


def test_classify_uses_extractor_and_excludes_known_pages() -> None:
    fake_pages = {
        1: ["body sentence one.", "body sentence two."],
        2: _entries(80) + ["280 | Index"],  # back-index
        3: _entries(80) + ["281 | Index"],  # back-index
        4: ["chapter title", "body sentence."],
    }

    def fake_extract(pdf_path: object, page_no: int) -> list[str]:
        return fake_pages.get(page_no, [])

    with patch("mmrag_v2.processor._extract_pdf_page_lines", side_effect=fake_extract):
        # exclude={3} mimics a Docling-classified back-index page already covered.
        result = _classify_dense_back_index_pages_by_source(
            pdf_path=Path("/tmp/fake.pdf"),
            total_pages=4,
            exclude={3},
        )
    assert result == {2}


def test_extract_pdf_page_lines_handles_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.pdf"
    assert _extract_pdf_page_lines(missing, 1) == []


# ---------------------------------------------------------------------------
# End-to-end verification with the real Adedeji PDF (skipped if absent)
# ---------------------------------------------------------------------------

ADEDEJI_PDF = Path(
    "data/technical_manual/Adedeji A. GenAI on Google Cloud. "
    "Enterprise Generative AI Systems...Agents 2026.pdf"
)


@pytest.mark.skipif(not ADEDEJI_PDF.exists(), reason="Adedeji PDF not available in CI sandbox")
def test_real_adedeji_back_index_span_detected() -> None:
    """Adedeji p298-316 are the back-index — the Phase 4 Step 2 target."""
    found = _classify_dense_back_index_pages_by_source(
        pdf_path=ADEDEJI_PDF,
        total_pages=320,
        exclude=set(),
    )
    # Expect the 19-page span (probe data: 298-316 inclusive).
    expected = set(range(298, 317))
    # Allow ±2 page slack at the boundaries (Docling-pdfium edge differences),
    # but require at least 18 of the 19 expected pages caught and zero hits
    # outside the span.
    assert (found & expected) and len(found & expected) >= 18, (
        f"Expected ≥18 of {expected} but found {sorted(found)[:5]}…"
    )
    assert not (found - expected - {297, 317}), (
        f"Unexpected back-index hits outside [296-318]: {found - expected - {297, 317}}"
    )
