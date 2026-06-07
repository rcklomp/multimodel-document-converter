"""R10 (PLAN_V3.1 P3) - direct contract net for the V3 chunker ENTRY point.

Why this module exists
----------------------
`chunk_universal_document` is the single entry into the UIR-native chunker and
the boundary every Phase-3 disposition will touch. Before this module it had NO
dedicated, full-contract unit test: it was reached only indirectly by
`test_v3_integration` (asserts the emitted ingestion.jsonl, not the chunker
output) and by one TOC-propagation case in
`test_ocr_path_heading_propagation`. That gap is exactly how a real
signature-drift break (an internal call passing kwargs the callee did not
accept) sailed through the entire ``-k chunk`` selection in a prior session.

Two layers, per the PLAN_V3.1 P2.5/R10 decision (synthetic + live):
  1. SYNTHETIC unit tests - a hand-built UniversalDocument via the public
     ``create_*`` factories. Fast, hermetic, total control of edge cases
     (TEXT/IMAGE/TABLE modality split, integer [0,1000] bboxes, heading
     precedence in-page > TOC-for-page > carry-forward, breadcrumb_path).
  2. LIVE reality test - runs the real offline DoclingFast engine
     (``USE_DOCLING_FAST=1``, CPU, no VLM, no network) to PRODUCE a genuine
     UniversalDocument, then feeds it straight to ``chunk_universal_document``
     and asserts the same invariants. This guards against the synthetic fixture
     drifting from what the engine actually emits.

All offline. No VLM, no OpenRouter, no Qdrant, no network.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mmrag_v2.chunking.uir_chunker import chunk_universal_document
from mmrag_v2.universal.intermediate import (
    CoordinateFrame,
    Element,
    ElementType,
    Modality,
    PageClassification,
    UniversalDocument,
    UniversalPage,
    create_element,
)

# COORD scale per REQ-COORD-01.
_COORD_MAX = 1000


# ---------------------------------------------------------------------------
# Synthetic builders (public factories only - no hand-rolled serialization)
# ---------------------------------------------------------------------------


def _page(page_number: int, elements: list[Element]) -> UniversalPage:
    return UniversalPage(
        page_number=page_number,
        elements=elements,
        classification=PageClassification.DIGITAL,
        dimensions=(612, 792),
    )


def _doc(pages: list[UniversalPage], *, doc_id: str = "r10doc") -> UniversalDocument:
    return UniversalDocument(
        doc_id=doc_id,
        source_file="r10.pdf",
        file_type="pdf",
        pages=pages,
    )


def _mixed_doc() -> UniversalDocument:
    """A 2-page doc spanning all three modalities.

    Page 1: a heading + a body paragraph + an image region.
    Page 2: a table + a trailing body paragraph (no in-page heading).
    """
    p1 = _page(
        1,
        [
            create_element(
                ElementType.TEXT,
                "Chapter One Overview",
                bbox=[50, 40, 950, 80],
                element_index=0,
                source_label="section_header",
            ),
            create_element(
                ElementType.TEXT,
                "Body paragraph introducing the chapter with enough text to chunk.",
                bbox=[50, 100, 950, 300],
                element_index=1,
            ),
            create_element(
                ElementType.IMAGE,
                "A diagram of the system architecture.",
                bbox=[100, 320, 500, 560],
                element_index=2,
            ),
        ],
    )
    p2 = _page(
        2,
        [
            create_element(
                ElementType.TABLE,
                "| A | B |\n|---|---|\n| 1 | 2 |",
                bbox=[60, 60, 900, 360],
                element_index=0,
            ),
            create_element(
                ElementType.TEXT,
                "Closing body text on the second page with no heading element here.",
                bbox=[50, 380, 950, 520],
                element_index=1,
            ),
        ],
    )
    return _doc([p1, p2])


# ---------------------------------------------------------------------------
# 1. Entry point runs + signature integrity (the regression that escaped)
# ---------------------------------------------------------------------------


def test_entry_point_runs_and_returns_chunks() -> None:
    """The documented call signature works end-to-end and yields chunks.

    A wrong internal call (kwargs the callee rejects) would raise TypeError
    here - the exact failure that bypassed the ``-k chunk`` suite before.
    """
    chunks = chunk_universal_document(_mixed_doc())
    assert chunks, "entry point returned no chunks"


def test_entry_point_accepts_documented_kwargs() -> None:
    """All documented keyword args are accepted (signature-drift guard)."""
    chunks = chunk_universal_document(
        _mixed_doc(),
        max_chars=1400,
        min_chars=20,
        extraction_engine_version="test",
        profile_type="academic_whitepaper",
        toc_headings=None,
    )
    assert chunks


# ---------------------------------------------------------------------------
# 2. Modality split - TEXT/IMAGE/TABLE elements map to their own chunk modality
# ---------------------------------------------------------------------------


def test_modality_split_preserves_all_three() -> None:
    chunks = chunk_universal_document(_mixed_doc())
    mods = {c.modality for c in chunks}
    assert Modality.IMAGE in mods, "IMAGE element did not yield an IMAGE chunk"
    assert Modality.TABLE in mods, "TABLE element did not yield a TABLE chunk"
    assert Modality.TEXT in mods, "TEXT elements did not yield a TEXT chunk"


def test_image_and_table_chunks_carry_their_content() -> None:
    chunks = chunk_universal_document(_mixed_doc())
    img = [c for c in chunks if c.modality == Modality.IMAGE]
    tbl = [c for c in chunks if c.modality == Modality.TABLE]
    assert img and "diagram" in img[0].content.lower()
    assert tbl and "|" in tbl[0].content  # markdown grid preserved


# ---------------------------------------------------------------------------
# 3. Integer [0,1000] bbox invariant (REQ-COORD-01) on every located chunk
# ---------------------------------------------------------------------------


def test_all_bboxes_are_integers_in_range() -> None:
    chunks = chunk_universal_document(_mixed_doc())
    located = [c for c in chunks if c.locator is not None and c.locator.bbox is not None]
    assert located, "expected at least one located chunk"
    for c in located:
        bbox = c.locator.bbox
        assert len(bbox) == 4, f"bbox not 4-tuple: {bbox!r}"
        for v in bbox:
            assert isinstance(v, int), f"bbox coord {v!r} not int (REQ-COORD-01)"
            assert 0 <= v <= _COORD_MAX, f"bbox coord {v} out of [0,1000]"


# ---------------------------------------------------------------------------
# 4. Heading precedence: in-page > TOC-for-page > carry-forward
# ---------------------------------------------------------------------------


def test_precedence_in_page_heading_wins() -> None:
    """A trusted in-page heading is used even when a different TOC entry
    covers the page (rule 1 beats rule 3)."""
    doc = _doc(
        [
            _page(
                5,
                [
                    create_element(
                        ElementType.TEXT,
                        "5.2 Local Section Title",
                        bbox=[50, 40, 950, 80],
                        element_index=0,
                        source_label="section_header",
                    ),
                    create_element(
                        ElementType.TEXT,
                        "Body under the in-page heading, long enough to survive.",
                        bbox=[50, 100, 950, 300],
                        element_index=1,
                    ),
                ],
            )
        ]
    )
    toc = {
        5: ["Part X", "Chapter Five"],
        "__heading_map__": {"Chapter Five": ["Part X", "Chapter Five"]},
    }
    chunks = chunk_universal_document(doc, toc_headings=toc)
    body = [c for c in chunks if c.modality == Modality.TEXT and "Body under" in c.content]
    assert body, "expected the body chunk"
    assert body[0].parent_heading == "5.2 Local Section Title"


def test_carry_in_heading_seeds_null_body_chunks() -> None:
    """Cluster B (2026-06-07): heading assignment runs per batch. A batch whose
    chunks have no heading of their own must inherit the last heading from the
    previous batch via carry_in_heading (HarryPotter ch.1 batch 3)."""
    doc = _doc(
        [
            _page(
                21,
                [
                    create_element(
                        ElementType.TEXT,
                        "Albus Dumbledore did not seem to realize, and went on smiling.",
                        bbox=[50, 100, 950, 300],
                        element_index=0,
                    ),
                    create_element(
                        ElementType.TEXT,
                        "It certainly seems so, said Dumbledore, with much to be thankful for.",
                        bbox=[50, 320, 950, 540],
                        element_index=1,
                    ),
                ],
            )
        ]
    )
    chunks = chunk_universal_document(
        doc,
        carry_in_heading="THE BOY WHO LIVED",
        carry_in_breadcrumb=["Harry Potter", "THE BOY WHO LIVED", "Page 20"],
    )
    body = [c for c in chunks if c.modality == Modality.TEXT]
    assert body, "expected body chunks"
    assert all(c.parent_heading == "THE BOY WHO LIVED" for c in body)


def test_carry_in_heading_overridden_by_own_heading() -> None:
    """A real in-page heading on the batch overrides the carried heading
    (precedence 1 still beats the carry seed)."""
    doc = _doc(
        [
            _page(
                30,
                [
                    create_element(
                        ElementType.TEXT,
                        "CHAPTER TWO The Vanishing Glass",
                        bbox=[50, 40, 950, 80],
                        element_index=0,
                        source_label="section_header",
                    ),
                    create_element(
                        ElementType.TEXT,
                        "Nearly ten years had passed since the Dursleys had woken up.",
                        bbox=[50, 100, 950, 300],
                        element_index=1,
                    ),
                ],
            )
        ]
    )
    chunks = chunk_universal_document(doc, carry_in_heading="THE BOY WHO LIVED")
    body = [c for c in chunks if c.modality == Modality.TEXT and "Nearly ten" in c.content]
    assert body and body[0].parent_heading != "THE BOY WHO LIVED"


def test_precedence_toc_for_page_beats_stale_carry() -> None:
    """When a page has no in-page heading but the TOC has an entry for that
    exact page, the per-page TOC wins over a heading carried from earlier
    (rule 3 beats rule 2)."""
    doc = _doc(
        [
            _page(
                10,
                [
                    create_element(
                        ElementType.TEXT,
                        "Chapter Ten",
                        bbox=[50, 40, 950, 80],
                        element_index=0,
                        source_label="section_header",
                    ),
                    create_element(
                        ElementType.TEXT,
                        "Body on page ten under chapter ten heading element.",
                        bbox=[50, 100, 950, 300],
                        element_index=1,
                    ),
                ],
            ),
            _page(
                11,
                [
                    create_element(
                        ElementType.TEXT,
                        "Body on page eleven with no heading element of its own.",
                        bbox=[50, 60, 950, 260],
                        element_index=0,
                    ),
                ],
            ),
        ]
    )
    toc = {
        10: ["Chapter Ten"],
        11: ["Chapter Eleven"],  # page 11 belongs to a different section per TOC
        "__heading_map__": {
            "Chapter Ten": ["Chapter Ten"],
            "Chapter Eleven": ["Chapter Eleven"],
        },
    }
    chunks = chunk_universal_document(doc, toc_headings=toc)
    p11 = [
        c
        for c in chunks
        if c.modality == Modality.TEXT and c.locator.page_number == 11
    ]
    assert p11, "expected a page-11 text chunk"
    # TOC-for-page-11 (Chapter Eleven) must win over carried Chapter Ten.
    assert p11[0].parent_heading == "Chapter Eleven"


def test_precedence_carry_forward_when_no_toc() -> None:
    """With no TOC, a heading on an earlier page carries to later
    heading-less pages (rule 2)."""
    doc = _doc(
        [
            _page(
                3,
                [
                    create_element(
                        ElementType.TEXT,
                        "Introduction Heading",
                        bbox=[50, 40, 950, 80],
                        element_index=0,
                        source_label="section_header",
                    ),
                    create_element(
                        ElementType.TEXT,
                        "First body paragraph under the introduction heading here.",
                        bbox=[50, 100, 950, 300],
                        element_index=1,
                    ),
                ],
            ),
            _page(
                4,
                [
                    create_element(
                        ElementType.TEXT,
                        "Second page body with no heading element of its own here.",
                        bbox=[50, 60, 950, 260],
                        element_index=0,
                    ),
                ],
            ),
        ]
    )
    chunks = chunk_universal_document(doc, toc_headings=None)
    p4 = [
        c
        for c in chunks
        if c.modality == Modality.TEXT and c.locator.page_number == 4
    ]
    assert p4, "expected a page-4 text chunk"
    assert p4[0].parent_heading == "Introduction Heading"


def test_no_toc_no_heading_leaves_parent_heading_null() -> None:
    """Honest null: a doc with neither TOC nor any in-page heading must NOT
    fabricate a parent_heading."""
    doc = _doc(
        [
            _page(
                1,
                [
                    create_element(
                        ElementType.TEXT,
                        "Plain body text with absolutely no heading anywhere in it.",
                        bbox=[50, 60, 950, 260],
                        element_index=0,
                    )
                ],
            )
        ]
    )
    chunks = chunk_universal_document(doc, toc_headings=None)
    text = [c for c in chunks if c.modality == Modality.TEXT]
    assert text
    assert all(c.parent_heading is None for c in text)


# ---------------------------------------------------------------------------
# 5. breadcrumb_path populated from the TOC hierarchy
# ---------------------------------------------------------------------------


def test_breadcrumb_path_built_from_toc() -> None:
    doc = _doc(
        [
            _page(
                23,
                [
                    create_element(
                        ElementType.TEXT,
                        "1 LLMs and the need for RAG",
                        bbox=[50, 40, 950, 80],
                        element_index=0,
                        source_label="section_header",
                    ),
                    create_element(
                        ElementType.TEXT,
                        "Body content under the chapter one heading on this page.",
                        bbox=[50, 100, 950, 320],
                        element_index=1,
                    ),
                ],
            )
        ]
    )
    toc = {
        23: ["Part 1 Foundations", "1 LLMs and the need for RAG"],
        "__heading_map__": {
            "1 LLMs and the need for RAG": [
                "Part 1 Foundations",
                "1 LLMs and the need for RAG",
            ]
        },
    }
    chunks = chunk_universal_document(doc, toc_headings=toc)
    body = [c for c in chunks if c.modality == Modality.TEXT and "Body content" in c.content]
    assert body, "expected the body chunk"
    crumb = body[0].breadcrumb_path
    assert crumb, "breadcrumb_path not populated from TOC"
    assert "Part 1 Foundations" in crumb
    assert any("LLMs and the need for RAG" in seg for seg in crumb)


# ---------------------------------------------------------------------------
# 6. LIVE reality test - real offline engine -> UniversalDocument -> chunker
# ---------------------------------------------------------------------------

_SMOKE_PDF = Path("data/raw/Bevestigingsmiddelen.pdf")


@pytest.mark.skipif(
    not _SMOKE_PDF.exists(), reason=f"corpus PDF missing: {_SMOKE_PDF}"
)
def test_live_docling_fast_extract_through_chunker(monkeypatch) -> None:
    """The real DoclingFast engine output flows through the chunker intact.

    Grounds the synthetic fixtures above against a genuine engine-produced
    UniversalDocument. Offline + deterministic via USE_DOCLING_FAST=1.
    """
    monkeypatch.setenv("USE_DOCLING_FAST", "1")
    from mmrag_v3.processor import extract as v3_extract

    udoc = v3_extract(str(_SMOKE_PDF))
    assert isinstance(udoc, UniversalDocument)
    assert udoc.pages, "engine produced no pages"

    chunks = chunk_universal_document(udoc)
    assert chunks, "chunker produced no chunks from real engine output"

    for c in chunks:
        # Every chunk has a real modality from the unified vocabulary.
        assert c.modality in (
            Modality.TEXT,
            Modality.IMAGE,
            Modality.TABLE,
            Modality.CODE,
            Modality.FORM,
        )
        # Located chunks honor REQ-COORD-01.
        if c.locator is not None and c.locator.bbox is not None:
            assert len(c.locator.bbox) == 4
            for v in c.locator.bbox:
                assert isinstance(v, int) and 0 <= v <= _COORD_MAX
