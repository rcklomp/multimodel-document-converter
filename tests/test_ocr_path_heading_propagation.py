"""HEADING_PROPAGATION contracts (Phase 6 PLAN_V2.10 + PLAN_V3.1 P2).

Two layers of contract live here:

1. The central heading validator (``is_valid_heading``) and the
   ``ContextStateV2`` carry/attribution helpers — the garbage-rejection +
   cross-page-carry behaviour shared by every heading-assignment path.

2. PLAN_V3.1 P2 — the UIR-native heading-assignment pass
   ``uir_chunker._assign_headings``, which is the CURRENT shipping path. It
   sets ``parent_heading`` + ``breadcrumb_path`` per text chunk by
   precedence: (1) nearest preceding in-page heading element, (2)
   carry-forward of the last active heading from earlier pages, (3) the
   PyMuPDF TOC entry whose page covers the chunk; breadcrumb_path is built
   from the TOC hierarchy. TOC bookmark titles are authoritative and are
   NOT re-validated through ``is_valid_heading`` (which is tuned to reject
   OCR noise / numbered body-step prose).

Re-enabled 2026-05-31 (was V3_DEFERRED). The two prior production-wiring
pins on the deleted OCR/layout lane were DELETED-by-decision; see
docs/DECISIONS.md "OCR-lane production-wiring pins retired (PLAN_V3.1 P2)".
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from mmrag_v2.batch_processor import BatchProcessor
from mmrag_v2.chunking.uir_chunker import _assign_headings, chunk_universal_document
from mmrag_v2.state.context_state import (
    ContextStateV2,
    create_context_state,
    is_valid_heading,
)
from mmrag_v2.universal.intermediate import (
    ConfidenceBreakdown,
    CoordinateFrame,
    Element,
    ElementType,
    Locator,
    LocatorType,
    Modality,
    PageClassification,
    UIRChunk,
    UniversalDocument,
    UniversalPage,
)


def _processor(tmp_path) -> BatchProcessor:
    return BatchProcessor(
        output_dir=str(tmp_path),
        batch_size=10,
        vision_provider="none",
        enable_ocr=False,
    )


def _element(label: str, text: str) -> SimpleNamespace:
    """Build a lightweight Docling-element mimic.

    Mirrors the ``SimpleNamespace`` shape produced by
    ``BatchProcessor._extract_docling_layout_elements`` (light element with
    ``label.value`` + ``text``).
    """
    return SimpleNamespace(label=SimpleNamespace(value=label), text=text)


# ---------------------------------------------------------------------------
# Helper-state contracts (the "skip doc-title" rule in get_section_heading)
# ---------------------------------------------------------------------------


def test_get_section_heading_skips_doc_title_initial_breadcrumb() -> None:
    """The level-0 doc-title breadcrumb seeded by ``create_context_state``
    must NOT leak into the OCR-lane ``parent_heading``.
    """
    state = create_context_state(doc_id="firearms", source_file="Firearms.pdf")
    # Only the doc-title (level=0) is in the stack at this point.
    assert state.hierarchy_stack and state.hierarchy_stack[0].level == 0
    assert state.get_section_heading() is None

    state.update_on_heading("MAUSER 1898", level=1)
    assert state.get_section_heading() == "MAUSER 1898"


def test_get_section_heading_returns_latest_level1_only() -> None:
    state = ContextStateV2()
    assert state.get_section_heading() is None
    state.update_on_heading("Chapter One", level=1)
    state.update_on_heading("Chapter Two", level=1)
    assert state.get_section_heading() == "Chapter Two"


# ---------------------------------------------------------------------------
# Positive contracts (plan-required 3)
# ---------------------------------------------------------------------------


def test_section_header_pushes_to_hierarchy_stack(tmp_path) -> None:
    """A Docling section_header item pushes a heading into state and
    subsequent body emissions (this page, next page, page after) all
    inherit it via state.get_section_heading().
    """
    processor = _processor(tmp_path)
    processor._context_state = create_context_state(
        doc_id="firearms", source_file="Firearms.pdf"
    )
    processor._doc_total_pages = 292  # multi-page → push gate open

    # Page N: one section_header + 2 body paragraphs (Docling order).
    page_n_elements = [
        _element("section_header", "Mauser 1898 Rifle"),
        _element("paragraph", "The Mauser 1898 is a bolt-action rifle..."),
        _element("paragraph", "Disassembly begins by..."),
    ]
    processor._promote_ocr_section_headers(page_n_elements)
    assert processor._context_state.get_section_heading() == "Mauser 1898 Rifle"

    # Pages N+1, N+2 have no section_header items — body only.
    for _ in range(2):
        processor._promote_ocr_section_headers(
            [_element("paragraph", "More disassembly steps...")]
        )

    # All 5 body chunks (2 on page N, 3 on later pages) inherit the heading
    # at the read site (state.get_section_heading()).
    body_count = 5
    for _ in range(body_count):
        assert (
            processor._context_state.get_section_heading() == "Mauser 1898 Rifle"
        )


def test_new_section_header_replaces_prior_heading(tmp_path) -> None:
    """A second section_header arriving mid-document replaces the prior
    heading for subsequent body chunks.
    """
    processor = _processor(tmp_path)
    processor._context_state = create_context_state(
        doc_id="firearms", source_file="Firearms.pdf"
    )
    processor._doc_total_pages = 292

    processor._promote_ocr_section_headers(
        [_element("section_header", "Mauser 1898 Rifle")]
    )
    assert processor._context_state.get_section_heading() == "Mauser 1898 Rifle"

    # Three body chunks pages later, then a new chapter.
    processor._promote_ocr_section_headers(
        [_element("paragraph", "...")] * 3
    )
    assert processor._context_state.get_section_heading() == "Mauser 1898 Rifle"

    processor._promote_ocr_section_headers(
        [_element("section_header", "Remington Model 700")]
    )
    assert processor._context_state.get_section_heading() == "Remington Model 700"


def test_chapter_continues_across_batch_boundary(tmp_path) -> None:
    """Section_header on batch 1 last page is still the carried heading on
    batch 2 first page. State is per-document (``self._context_state``) and
    is NOT reset between batches.
    """
    processor = _processor(tmp_path)
    processor._context_state = create_context_state(
        doc_id="firearms", source_file="Firearms.pdf"
    )
    processor._doc_total_pages = 292

    # Batch 1 (pages 1-10): chapter heading appears on page 8.
    for _ in range(7):  # pages 1-7: body, no section_header
        processor._promote_ocr_section_headers(
            [_element("paragraph", "front-matter or earlier body")]
        )
    processor._promote_ocr_section_headers(
        [_element("section_header", "Browning Auto-5 Shotgun")]
    )
    # Page 9, 10 body-only inside batch 1
    for _ in range(2):
        processor._promote_ocr_section_headers(
            [_element("paragraph", "body")]
        )

    assert (
        processor._context_state.get_section_heading() == "Browning Auto-5 Shotgun"
    )

    # Batch 2 starts (pages 11-20). NO new section_header on the boundary
    # page. The carried heading must remain.
    processor._promote_ocr_section_headers(
        [_element("paragraph", "first page of next batch — no heading element")]
    )
    assert (
        processor._context_state.get_section_heading() == "Browning Auto-5 Shotgun"
    )


# ---------------------------------------------------------------------------
# Negative contracts (the missing pin in Phase 5 v1)
#
# A garbage section_header MUST NOT push to the hierarchy_stack. Re-uses the
# central ``is_valid_heading`` validator inside ``update_on_heading``.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "garbage_heading",
    [
        # Repeated-token artefact (Phase 5 audit-named)
        "Type Type TypeTypeTypeType",
        # Code/JSON-shape (Phase 5 audit-named)
        "GRAPH DATA: {",
        # Code-like prefix
        "def foo():",
        # Empty after strip
        "   ",
        # Pure page-number artefact
        "42",
    ],
)
def test_garbage_section_header_does_not_push(tmp_path, garbage_heading: str) -> None:
    processor = _processor(tmp_path)
    processor._context_state = create_context_state(
        doc_id="firearms", source_file="Firearms.pdf"
    )

    # Pre-state: only the doc-title level-0 entry.
    assert processor._context_state.get_section_heading() is None

    processor._promote_ocr_section_headers(
        [_element("section_header", garbage_heading)]
    )

    # update_on_heading rejected the garbage via is_valid_heading; the read
    # at the chunk-emission site still returns None.
    assert processor._context_state.get_section_heading() is None
    # And the validator is the one we expect (centralized; no parallel
    # _is_valid_ocr_heading).
    if garbage_heading.strip() and garbage_heading != "   ":
        assert is_valid_heading(garbage_heading) is False


def test_garbage_does_not_displace_prior_valid_heading(tmp_path) -> None:
    """A garbage section_header arriving after a valid one must not pop or
    replace the prior real heading.
    """
    processor = _processor(tmp_path)
    processor._context_state = create_context_state(
        doc_id="firearms", source_file="Firearms.pdf"
    )
    processor._doc_total_pages = 292

    processor._promote_ocr_section_headers(
        [_element("section_header", "Mauser 1898 Rifle")]
    )
    processor._promote_ocr_section_headers(
        [_element("section_header", "Type Type TypeTypeTypeType")]
    )

    assert (
        processor._context_state.get_section_heading() == "Mauser 1898 Rifle"
    )


def test_title_label_also_pushes_to_hierarchy(tmp_path) -> None:
    """Docling emits ``title`` as well as ``section_header`` for top-level
    structural items; both must promote into state.
    """
    processor = _processor(tmp_path)
    processor._context_state = create_context_state(
        doc_id="firearms", source_file="Firearms.pdf"
    )
    processor._doc_total_pages = 292

    processor._promote_ocr_section_headers(
        [_element("title", "Firearms Assembly/Disassembly")]
    )

    assert (
        processor._context_state.get_section_heading()
        == "Firearms Assembly/Disassembly"
    )


def test_non_heading_labels_do_not_push(tmp_path) -> None:
    """Labels other than ``section_header`` / ``title`` (paragraph, caption,
    list_item, table, picture) must not push to state — even if the element
    text happens to look like a heading.
    """
    processor = _processor(tmp_path)
    processor._context_state = create_context_state(
        doc_id="firearms", source_file="Firearms.pdf"
    )

    for non_heading_label in ("paragraph", "caption", "list_item", "table", "picture"):
        processor._promote_ocr_section_headers(
            [_element(non_heading_label, "INTRODUCTION")]
        )

    assert processor._context_state.get_section_heading() is None


def test_empty_input_is_a_noop(tmp_path) -> None:
    processor = _processor(tmp_path)
    processor._context_state = create_context_state(
        doc_id="firearms", source_file="Firearms.pdf"
    )

    processor._promote_ocr_section_headers(None)
    processor._promote_ocr_section_headers([])

    assert processor._context_state.get_section_heading() is None


def test_no_context_state_is_a_noop(tmp_path) -> None:
    """If ``self._context_state`` is None (e.g. legacy entry points), the
    helper silently no-ops instead of crashing.
    """
    processor = _processor(tmp_path)
    processor._context_state = None

    # Should not raise.
    processor._promote_ocr_section_headers(
        [_element("section_header", "Some Heading")]
    )


def test_single_page_doc_skips_push(tmp_path) -> None:
    """On single-page documents (forms / invoices / posters) the OCR-lane
    push is a no-op. The fix targets inter-page heading propagation;
    pushing a single Docling-tagged section_header on a one-page form
    flips the downstream form-detection heuristic in
    ``scripts/qa_conversion_audit.py``. The ``self._doc_total_pages > 1``
    gate restricts the push to its intended multi-page scope.
    """
    processor = _processor(tmp_path)
    processor._context_state = create_context_state(
        doc_id="form0013", source_file="invoice.pdf"
    )
    processor._doc_total_pages = 1

    processor._promote_ocr_section_headers(
        [_element("section_header", "Gesamt-Brutto 1.949,60 EUR")]
    )

    assert processor._context_state.get_section_heading() is None


def test_unknown_total_pages_skips_push(tmp_path) -> None:
    """When ``self._doc_total_pages`` is None (not yet set), the helper
    conservatively no-ops. This protects against accidentally pushing on
    a legacy/test code path that does not initialise the field.
    """
    processor = _processor(tmp_path)
    processor._context_state = create_context_state(
        doc_id="firearms", source_file="Firearms.pdf"
    )
    processor._doc_total_pages = None

    processor._promote_ocr_section_headers(
        [_element("section_header", "Mauser 1898")]
    )

    assert processor._context_state.get_section_heading() is None


# ---------------------------------------------------------------------------
# Phase 6 (audit follow-up) — ordered per-chunk heading attribution
#
# These tests pin the contract enforced by
# ``BatchProcessor._attribute_ocr_chunk_heading``, which is invoked once
# per ``ProcessedChunk`` in the order returned by
# ``LayoutAwareOCRProcessor.process_page``. Heading-marked chunks
# (``ProcessedChunk.is_heading=True``) push their content into
# ``ContextStateV2`` BEFORE state is read for that chunk, so within-page
# ordering is preserved.
# ---------------------------------------------------------------------------


def _pchunk(content: str, is_heading: bool = False):
    """Build a lightweight ProcessedChunk mimic with the two fields the
    ordered-attribution helper reads (``is_heading``, ``content``).
    """
    return SimpleNamespace(is_heading=is_heading, content=content)


def test_ordered_body_before_heading_inherits_prior(tmp_path) -> None:
    """Body chunks emitted BEFORE the first heading on a page must
    inherit the previously-active heading (set on an earlier page),
    not the heading later on the same page.
    """
    processor = _processor(tmp_path)
    processor._context_state = create_context_state(
        doc_id="firearms", source_file="Firearms.pdf"
    )
    processor._doc_total_pages = 292  # multi-page → per-chunk push gate open
    # Prior page established a heading.
    processor._context_state.update_on_heading("Mauser 1898 Rifle", level=1)

    body_before = _pchunk("Body text emitted before a new section_header.")
    later_heading = _pchunk("Remington Model 700", is_heading=True)
    body_after = _pchunk("Body text emitted after the new heading.")

    results = [
        processor._attribute_ocr_chunk_heading(pc)
        for pc in (body_before, later_heading, body_after)
    ]

    # body_before inherits the prior page's heading, not the later one.
    assert results == [
        "Mauser 1898 Rifle",
        "Remington Model 700",
        "Remington Model 700",
    ]


def test_ordered_body_before_first_heading_is_null(tmp_path) -> None:
    """Pre-section-header chunks emitted before ANY heading has been
    pushed must attribute to ``None``, not to the document title
    (which is at level=0 in ``hierarchy_stack`` by convention).
    """
    processor = _processor(tmp_path)
    processor._context_state = create_context_state(
        doc_id="firearms", source_file="Firearms.pdf"
    )
    processor._doc_total_pages = 292

    front_matter_body = _pchunk("Some pre-section front-matter prose.")
    first_heading = _pchunk("Introduction", is_heading=True)
    body_after = _pchunk("Section body.")

    results = [
        processor._attribute_ocr_chunk_heading(pc)
        for pc in (front_matter_body, first_heading, body_after)
    ]

    assert results == [None, "Introduction", "Introduction"]


def test_ordered_multiple_headings_on_same_page_switch_attribution(tmp_path) -> None:
    """When two headings appear on the same page, chunks between them
    attribute to the first; chunks after the second attribute to the
    second. The pre-Phase-6-audit "promote all then read once" design
    would have given every chunk the LAST heading.
    """
    processor = _processor(tmp_path)
    processor._context_state = create_context_state(
        doc_id="firearms", source_file="Firearms.pdf"
    )
    processor._doc_total_pages = 292

    heading_a = _pchunk("Mauser 1898", is_heading=True)
    body_under_a = _pchunk("Body text under Mauser.")
    heading_b = _pchunk("Remington Model 700", is_heading=True)
    body_under_b = _pchunk("Body text under Remington.")

    results = [
        processor._attribute_ocr_chunk_heading(pc)
        for pc in (heading_a, body_under_a, heading_b, body_under_b)
    ]

    assert results == [
        "Mauser 1898",
        "Mauser 1898",
        "Remington Model 700",
        "Remington Model 700",
    ]


def test_ordered_garbage_heading_does_not_displace_prior(tmp_path) -> None:
    """A heading-marked chunk whose content fails ``is_valid_heading``
    (Phase 5 audit-named shapes: repeated tokens, code/JSON shapes;
    Phase 6 audit-named: terminal-period body prose, numbered body
    steps) must NOT push to state. Subsequent body chunks inherit the
    PRIOR valid heading.
    """
    processor = _processor(tmp_path)
    processor._context_state = create_context_state(
        doc_id="firearms", source_file="Firearms.pdf"
    )
    processor._doc_total_pages = 292

    valid_heading = _pchunk("Mauser 1898", is_heading=True)
    garbage_repeated = _pchunk("Type Type TypeTypeTypeType", is_heading=True)
    garbage_code_shape = _pchunk("GRAPH DATA: {", is_heading=True)
    garbage_terminal_period = _pchunk(
        "5. Drift out the trigger cross-pin toward the right.", is_heading=True
    )
    garbage_numbered_body = _pchunk("6. Remove the hammer downward", is_heading=True)
    body = _pchunk("Body text after the garbage.")

    results = [
        processor._attribute_ocr_chunk_heading(pc)
        for pc in (
            valid_heading,
            garbage_repeated,
            garbage_code_shape,
            garbage_terminal_period,
            garbage_numbered_body,
            body,
        )
    ]

    # First push succeeds; every subsequent garbage push is rejected by
    # is_valid_heading, so state remains "Mauser 1898".
    assert results == ["Mauser 1898"] * 6


def test_ordered_no_heading_chunks_returns_carried_state(tmp_path) -> None:
    """Pages that contain only body chunks (no ``is_heading`` markers
    reach the per-chunk loop) inherit the previously-active heading on
    every chunk. State is the carry mechanism across pages and batches.
    """
    processor = _processor(tmp_path)
    processor._context_state = create_context_state(
        doc_id="firearms", source_file="Firearms.pdf"
    )
    processor._context_state.update_on_heading("Carried Chapter", level=1)

    body1 = _pchunk("First body chunk on this page.")
    body2 = _pchunk("Second body chunk.")
    body3 = _pchunk("Third body chunk.")

    results = [
        processor._attribute_ocr_chunk_heading(pc) for pc in (body1, body2, body3)
    ]

    assert results == ["Carried Chapter"] * 3


def test_ordered_question_heading_promotes_and_carries(tmp_path) -> None:
    """Question-mark headings restored by the audit refinement push
    through ``is_valid_heading`` and carry to subsequent body chunks.
    """
    processor = _processor(tmp_path)
    processor._context_state = create_context_state(
        doc_id="rag-tutorial", source_file="rag.pdf"
    )
    processor._doc_total_pages = 100

    heading_q = _pchunk("Why Use Retrieval Augmented Generation?", is_heading=True)
    body = _pchunk("RAG combines retrieval and generation to...")

    results = [
        processor._attribute_ocr_chunk_heading(pc) for pc in (heading_q, body)
    ]

    assert results == [
        "Why Use Retrieval Augmented Generation?",
        "Why Use Retrieval Augmented Generation?",
    ]


def test_ordered_single_page_doc_skips_per_chunk_push(tmp_path) -> None:
    """The single-page-doc gate that protects form/invoice detection
    must apply to BOTH the fallback ``_promote_ocr_section_headers``
    AND the per-chunk ordered push in ``_attribute_ocr_chunk_heading``.
    The canonical 0013 invoice has 1 page and Docling tags a layout-
    prominent total ("Gesamt-Brutto 1.949,60 EUR") or field-label
    ("Kunden-Nr.: ...") line as section_header; the ordered-attribution
    helper must NOT push it, or the downstream form-detection
    heuristic in ``scripts/qa_conversion_audit.py`` flips and the
    micro_non_label_ratio gate fires on what should be a form-skipped
    doc.
    """
    processor = _processor(tmp_path)
    processor._context_state = create_context_state(
        doc_id="form0013", source_file="invoice.pdf"
    )
    processor._doc_total_pages = 1

    body = _pchunk("First body line on the invoice.")
    form_label = _pchunk("Gesamt-Brutto 1.949,60 EUR", is_heading=True)
    field_label = _pchunk("Kunden-Nr.: 181222213/ KID-181222213", is_heading=True)
    body_after = _pchunk("Trailing body line.")

    results = [
        processor._attribute_ocr_chunk_heading(pc)
        for pc in (body, form_label, field_label, body_after)
    ]

    # Single-page doc → all chunks attribute to None (state never
    # mutated), matching the pre-Phase-6 form-detection contract.
    assert results == [None, None, None, None]


def test_ordered_no_context_state_returns_none(tmp_path) -> None:
    """When ``self._context_state`` is None (legacy / test entry
    points), the helper returns None instead of crashing.
    """
    processor = _processor(tmp_path)
    processor._context_state = None

    heading = _pchunk("Some Heading", is_heading=True)
    body = _pchunk("Some body.")

    results = [
        processor._attribute_ocr_chunk_heading(pc) for pc in (heading, body)
    ]

    assert results == [None, None]


def test_multi_page_doc_pushes_normally(tmp_path) -> None:
    """On a multi-page document the gate does not interfere with normal
    propagation behaviour.
    """
    processor = _processor(tmp_path)
    processor._context_state = create_context_state(
        doc_id="firearms", source_file="Firearms.pdf"
    )
    processor._doc_total_pages = 292

    processor._promote_ocr_section_headers(
        [_element("section_header", "Mauser 1898 Rifle")]
    )

    assert processor._context_state.get_section_heading() == "Mauser 1898 Rifle"


# ---------------------------------------------------------------------------
# Structural pin — one OCR-lane production push site
# ---------------------------------------------------------------------------


def test_ocr_lane_heading_helpers_delegate_to_central_validator() -> None:
    """The OCR-lane heading helpers delegate through the central
    ``update_on_heading`` validator; no parallel mechanism.

    The two named push helpers — ``_attribute_ocr_chunk_heading``
    (per-chunk ordered attribution) and ``_promote_ocr_section_headers``
    (VLM-fullpage / Tesseract-fullpage fallback) — both push via
    ``ContextStateV2.update_on_heading`` (the only mutation API that runs
    ``is_valid_heading``). This is the live behavioural contract.

    NOTE (PLAN_V3.1 P2, 2026-05-31): the prior production-wiring
    assertions (``callers_attribute == 1`` / ``callers_promote == 1``,
    and the ``_propagate_headings`` call-count pin in process_pdf) were
    DELETED-by-decision — they pinned the OCR/layout lane production
    call sites that Phase A Step 5 removed (the lane is gone; heading
    propagation is now UIR-native in uir_chunker._assign_headings). See
    docs/DECISIONS.md "OCR-lane production-wiring pins retired (PLAN_V3.1 P2)".
    The validator-delegation contract below is retained because the
    helpers and the central validator are still LIVE.
    """
    import inspect

    from mmrag_v2.batch_processor import BatchProcessor

    # The two named push helpers exist.
    assert hasattr(BatchProcessor, "_attribute_ocr_chunk_heading")
    assert hasattr(BatchProcessor, "_promote_ocr_section_headers")

    # Both delegate through the central validator (``update_on_heading``
    # is the only mutation API on ContextStateV2 that runs
    # ``is_valid_heading``).
    attribute_source = inspect.getsource(BatchProcessor._attribute_ocr_chunk_heading)
    promote_source = inspect.getsource(BatchProcessor._promote_ocr_section_headers)
    assert "update_on_heading" in attribute_source
    assert "update_on_heading" in promote_source


def test_no_parallel_ocr_heading_validator() -> None:
    """The Phase 6 fix MUST NOT introduce a parallel
    ``_is_valid_ocr_heading`` / ``_validate_ocr_heading`` validator in
    batch_processor. The central validator
    ``state.context_state.is_valid_heading`` is reused via
    ``update_on_heading``.
    """
    import inspect

    from mmrag_v2 import batch_processor as bp_mod

    source = inspect.getsource(bp_mod)
    for forbidden in ("_is_valid_ocr_heading", "_validate_ocr_heading"):
        assert forbidden not in source, (
            f"Found parallel OCR-lane heading validator {forbidden!r}; "
            "validation must live in state/context_state.py"
        )


# ---------------------------------------------------------------------------
# Phase 6 — is_valid_heading sentence-shape tightening
#
# The OCR-lane fix exposed a class of Docling layout mis-classifications:
# numbered list items / wrapped body lines in technical manuals get tagged
# `section_header` (Firearms-shape canonical example). The fix tightens
# ``is_valid_heading`` centrally to reject sentence-shape (≥ 5 words ending
# in terminal `.`, `!`, `?`). The tests below pin the new shape AND verify
# the Phase 5 audit-named garbage rejections still hold AND verify real
# Phase 5 Devlin-shape numbered headings still pass.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body_shape_heading",
    [
        # Firearms top-N false-positive shapes (numbered instruction steps,
        # 5+ words ending in a terminal period).
        "5. Drift out the trigger cross-pin toward the right.",
        "6 Drift out the trigger pin toward the left, and removeit.",
        "9 Drift out the small cross-pin near the bolt release.",
        "9. Drift out the bolt head retaining pin.",
        "9. Remove the sear, sear spring; and trigger upward:.",
        # Generic body-prose ending in a terminal period.
        "This is a clear demonstration of the principle.",
    ],
)
def test_is_valid_heading_rejects_terminal_period_body_prose(
    body_shape_heading: str,
) -> None:
    """The terminal-period rule fires only on `.`, not on `?` or `!`,
    so legitimate question headings are not regressed (see the
    ``test_is_valid_heading_accepts_question_headings`` pin below).
    """
    assert is_valid_heading(body_shape_heading) is False


@pytest.mark.parametrize(
    "question_heading",
    [
        # Real question-mark headings that an earlier Phase 6 iteration
        # incorrectly rejected. The audit restored ``?`` to the accept
        # set (real headings frequently take a question form in
        # tutorials, FAQ docs, and explainer chapters).
        "What Is an AI Agent?",
        "Why Use Retrieval Augmented Generation?",
        "2. What are AI agents?",
        "How does RAG work?",
        "Why?",
        "What now?",
    ],
)
def test_is_valid_heading_accepts_question_headings(
    question_heading: str,
) -> None:
    assert is_valid_heading(question_heading) is True


@pytest.mark.parametrize(
    "exclamation_heading",
    [
        # Exclamation-shaped headings (uncommon but legitimate).
        "Stop! Read This First",
        "Look Out!",
        "Get Started!",
    ],
)
def test_is_valid_heading_accepts_exclamation_headings(
    exclamation_heading: str,
) -> None:
    """Like ``?``, the ``!`` terminator is no longer treated as a
    body-prose signal — real exclamation-form headings are valid.
    """
    assert is_valid_heading(exclamation_heading) is True


@pytest.mark.parametrize(
    "numbered_body_step_no_period",
    [
        # Firearms-shape: numbered imperative without a terminal period.
        # These slip past the terminal-punctuation rule but match the
        # "numbered prefix + 2+ lowercase content words" body-step shape.
        "6. Remove the hammer downward",
        "2. Remove the trigger housing downward",
        # Stronger: 3 lowercase content words.
        "4. Push out the safety pivot toward the left",
        # Body-step with only 2 lowercase content words after stop-word
        # filtering — the threshold matches.
        "3 Remove the bolt from the receiver downward",
    ],
)
def test_is_valid_heading_rejects_numbered_body_step_case_pattern(
    numbered_body_step_no_period: str,
) -> None:
    """The case-pattern rule catches Firearms numbered instruction steps
    where the body has Title-case verb followed by lowercase content
    nouns (canonical body-step shape) — even when the terminal-period
    sentence-shape rule doesn't fire.
    """
    assert is_valid_heading(numbered_body_step_no_period) is False


@pytest.mark.parametrize(
    "phase5_garbage",
    [
        # Phase 5 audit-named garbage that `is_valid_heading` itself
        # rejects. These must STILL be rejected after the Phase 6
        # tightening (regression guard).
        # "Start" is intentionally NOT in this list — it passes
        # is_valid_heading by design and is handled at the propagation
        # layer via `_GENERIC_CARRY_HEADINGS` in `_propagate_headings`.
        "Type Type TypeTypeTypeType",
        "GRAPH DATA: {",
    ],
)
def test_is_valid_heading_phase5_garbage_still_rejected(
    phase5_garbage: str,
) -> None:
    assert is_valid_heading(phase5_garbage) is False


@pytest.mark.parametrize(
    "phase5_devlin_real_heading",
    [
        # Phase 5 Devlin top-N — these are REAL chapter headings and must
        # NOT be rejected by the new sentence-shape rule. None of them
        # ends with `.`, `!`, or `?` — that is precisely the signal that
        # keeps the new rule conservative.
        "1 - The New Age of AI Agents",
        "2. Fine-Tuning: Teaching the Model to Specialize",
        "6. Automating Evaluation",
        "B. Parallelization and Scalability",
        "14. Linking to Memory and Context",
        "7. Memory Synchronization and Versioning",
        "3. Use-Cases, Strengths & Trade-Offs",
        "8. Common Knowledge Graph Standards",
    ],
)
def test_is_valid_heading_phase5_devlin_real_headings_still_pass(
    phase5_devlin_real_heading: str,
) -> None:
    assert is_valid_heading(phase5_devlin_real_heading) is True


@pytest.mark.parametrize(
    "real_short_heading_with_period",
    [
        # Short titles that DO end with a period must still pass — the
        # rule only fires when word_count >= 5. This protects academic
        # headings like "References." or "5. Conclusions.".
        "References.",
        "5. Conclusions.",
        "Methods.",
        "Et al.",
    ],
)
def test_is_valid_heading_short_headings_with_period_still_pass(
    real_short_heading_with_period: str,
) -> None:
    assert is_valid_heading(real_short_heading_with_period) is True


@pytest.mark.parametrize(
    "real_heading",
    [
        # Firearms-shape real headings — chapter titles in the actual
        # corpus. These must NOT regress.
        "Disassembly:",
        "Reassembly Tips:",
        "MAUSER 1898",
        "REMINGTON MODEL 700",
        "WINCHESTER MODEL 71",
        "BROWNING AUTO-5",
        "Mauser 1898 Rifle",
        "Front Matter",
    ],
)
def test_is_valid_heading_firearms_real_headings_pass(real_heading: str) -> None:
    assert is_valid_heading(real_heading) is True


# DELETED-by-decision (PLAN_V3.1 P2, 2026-05-31):
# ``test_hybrid_chunker_lane_propagate_headings_call_count_unchanged`` pinned
# a single ``_propagate_headings(`` production call site inside process_pdf.
# Phase A Step 5 stripped that finalize call (the v2.16 reconcile lane is
# gone); heading + breadcrumb propagation is now UIR-native in
# uir_chunker._assign_headings, fed the PyMuPDF TOC as plain data. The pin
# asserted removed wiring and can no longer hold. See docs/DECISIONS.md
# "OCR-lane production-wiring pins retired (PLAN_V3.1 P2)".


# ---------------------------------------------------------------------------
# PLAN_V3.1 P2 — UIR-native heading-assignment pass (uir_chunker._assign_headings)
#
# These are the live contract that keeps the TOC-driven heading + breadcrumb
# propagation fixed. They exercise the shipping path's heading pass directly
# on UIRChunks, threading the PyMuPDF TOC in as plain data.
# ---------------------------------------------------------------------------


def _text_chunk(page: int, content: str, parent_heading=None) -> UIRChunk:
    return UIRChunk(
        modality=Modality.TEXT,
        content=content,
        locator=Locator(
            type=LocatorType.BBOX,
            bbox=[0, 0, 1000, 100],
            page_number=page,
            coordinate_frame=CoordinateFrame.PDF_PAGE_PORTRAIT,
        ),
        confidence=ConfidenceBreakdown(),
        extraction_method="uir_native_chunker",
        extraction_engine_version="test",
        parent_heading=parent_heading,
    )


# Mirrors the BatchProcessor._extract_toc_headings shape: int page -> breadcrumb
# at end-of-page, plus a "__heading_map__" of normalized title -> breadcrumb.
_RAG_TOC = {
    23: ["Part 1 Foundations", "1 LLMs and the need for RAG"],
    24: ["Part 1 Foundations", "1 LLMs and the need for RAG", "1.1\tCurse of the LLMs"],
    26: [
        "Part 1 Foundations",
        "1 LLMs and the need for RAG",
        "1.1\tCurse of the LLMs",
        "1.1.1\tLLMs are not trained for facts",
    ],
    "__heading_map__": {
        "1 LLMs and the need for RAG": ["Part 1 Foundations", "1 LLMs and the need for RAG"],
        "1.1.1 LLMs are not trained for facts": [
            "Part 1 Foundations",
            "1 LLMs and the need for RAG",
            "1.1\tCurse of the LLMs",
            "1.1.1\tLLMs are not trained for facts",
        ],
    },
}


def test_assign_headings_carry_forward_across_pages() -> None:
    """A heading on page 23 carries to body chunks on pages 24/25 that have
    no heading of their own (precedence rule 2)."""
    chunks = [
        _text_chunk(23, "1 chapter heading", parent_heading="1 LLMs and the need for RAG"),
        _text_chunk(24, "body on page 24 with no heading element"),
        _text_chunk(25, "body on page 25 with no heading element"),
    ]
    _assign_headings(chunks, None, doc_title=None)  # no TOC → carry-forward only
    assert chunks[0].parent_heading == "1 LLMs and the need for RAG"
    assert chunks[1].parent_heading == "1 LLMs and the need for RAG"
    assert chunks[2].parent_heading == "1 LLMs and the need for RAG"


def test_assign_headings_toc_fallback_when_no_in_page_heading() -> None:
    """When a chunk has no in-page heading and nothing has been carried, the
    TOC entry whose page covers the chunk supplies parent_heading +
    breadcrumb (precedence rule 3)."""
    chunks = [_text_chunk(24, "body text, page 24, no heading element")]
    _assign_headings(chunks, _RAG_TOC, doc_title="A Simple Guide")
    assert chunks[0].parent_heading == "1.1 Curse of the LLMs"  # TOC leaf, tab normalized
    assert chunks[0].breadcrumb_path
    assert "1.1 Curse of the LLMs" in chunks[0].breadcrumb_path
    assert chunks[0].breadcrumb_path[0] == "A Simple Guide"
    assert chunks[0].breadcrumb_path[-1] == "Page 24"


def test_assign_headings_trusts_toc_title_rejected_by_is_valid_heading() -> None:
    """A numbered subsection title like ``1.1.1 ...`` is rejected by
    is_valid_heading (numbered body-step shape) but IS a real TOC bookmark.
    The pass must trust it (authoritative structure) and build its
    breadcrumb, rather than dropping the chunk's heading."""
    assert is_valid_heading("1.1.1 LLMs are not trained for facts") is False  # premise
    chunks = [
        _text_chunk(
            26,
            "heading element text",
            parent_heading="1.1.1 LLMs are not trained for facts",
        ),
        _text_chunk(26, "body under the subsection, no heading element"),
    ]
    _assign_headings(chunks, _RAG_TOC, doc_title="A Simple Guide")
    for ch in chunks:
        assert ch.parent_heading == "1.1.1 LLMs are not trained for facts"
        assert ch.breadcrumb_path
        assert "1.1.1 LLMs are not trained for facts" in ch.breadcrumb_path


def test_assign_headings_breadcrumb_depth_capped_at_five() -> None:
    """REQ-HIER-04: breadcrumb depth is capped at 5, keeping the doc root
    plus the deepest context."""
    chunks = [_text_chunk(26, "body, page 26, no heading element")]
    _assign_headings(chunks, _RAG_TOC, doc_title="A Simple Guide")
    bc = chunks[0].breadcrumb_path
    assert len(bc) <= 5
    assert bc[0] == "A Simple Guide"
    assert bc[-1] == "Page 26"


def test_assign_headings_no_toc_no_heading_stays_null() -> None:
    """Honest behaviour: a doc with no TOC and no detectable heading element
    (e.g. a parts-list leaflet) leaves parent_heading null. No fabrication."""
    chunks = [
        _text_chunk(1, "M5x40 M5x50 M3x6 grubscrew ... parts list row"),
        _text_chunk(2, "7991 verzonken 34805-1 ... another parts row"),
    ]
    _assign_headings(chunks, None, doc_title=None)
    assert all(c.parent_heading is None for c in chunks)
    assert all(c.breadcrumb_path == [] for c in chunks)


def test_assign_headings_garbage_in_page_heading_not_carried() -> None:
    """A chunker heuristic heading that is NOT a TOC title and fails
    is_valid_heading (e.g. body prose mis-detected) must not become the
    carried heading; later body chunks stay null when there is no TOC."""
    garbage = "This is a clear demonstration of the principle."  # 5+ words, terminal period
    assert is_valid_heading(garbage) is False  # premise
    chunks = [
        _text_chunk(1, "mis-detected heading chunk", parent_heading=garbage),
        _text_chunk(2, "later body, no heading element"),
    ]
    _assign_headings(chunks, None, doc_title=None)
    assert chunks[1].parent_heading is None


def test_chunk_universal_document_threads_toc_and_propagates() -> None:
    """End-to-end through the chunker entry point: a 2-page UniversalDocument
    where page 1 has a section_header element and page 2 has body only. With
    the TOC threaded in, the page-2 body chunk inherits the heading and gets
    a TOC breadcrumb."""

    def _el(etype, content, page, label=""):
        return Element(
            type=etype,
            content=content,
            bbox=None,
            confidence=0.95,
            extraction_method="native",
            element_index=0,
            source_label=label,
        )

    page1 = UniversalPage(
        page_number=1,
        elements=[
            _el(ElementType.TEXT, "1 LLMs and the need for RAG", 1, label="section_header"),
            _el(ElementType.TEXT, "Body paragraph introducing the chapter content.", 1),
        ],
        classification=PageClassification.DIGITAL,
        dimensions=(612, 792),
    )
    page2 = UniversalPage(
        page_number=2,
        elements=[
            _el(ElementType.TEXT, "Continuation body text on the next page, no heading here.", 2),
        ],
        classification=PageClassification.DIGITAL,
        dimensions=(612, 792),
    )
    doc = UniversalDocument(
        doc_id="testdoc",
        source_file="test.pdf",
        file_type="pdf",
        pages=[page1, page2],
    )
    toc = {
        1: ["1 LLMs and the need for RAG"],
        2: ["1 LLMs and the need for RAG"],
        "__heading_map__": {"1 LLMs and the need for RAG": ["1 LLMs and the need for RAG"]},
    }
    chunks = chunk_universal_document(doc, toc_headings=toc)
    text_chunks = [c for c in chunks if c.modality == Modality.TEXT]
    assert text_chunks, "expected at least one text chunk"
    # Every text chunk (both pages) inherits the chapter heading.
    assert all(c.parent_heading == "1 LLMs and the need for RAG" for c in text_chunks)
    # Page-2 body chunk carries a populated breadcrumb.
    page2_chunks = [c for c in text_chunks if c.locator.page_number == 2]
    assert page2_chunks and all(c.breadcrumb_path for c in page2_chunks)
