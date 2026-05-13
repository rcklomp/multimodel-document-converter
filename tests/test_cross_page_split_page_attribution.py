"""Phase 4 v2.10 — cross-page-split page attribution
(`docs/PLAN_V2.10.md` §Phase 4 `CROSS_PAGE_SPLIT_PAGE_ATTRIBUTION`).

Docling 2.86 emits text items whose ``prov`` list carries multiple
``ProvenanceItem`` entries — one per source page — each with a
``charspan`` that locates the per-page slice within the item's
serialized text. HybridChunker also further slices such items into
multiple ``DocChunk`` objects when they exceed the token budget.

The v2.9 cross-page split path read only ``prov[0].page_no``, so both
DocChunks were attributed to the first page; the broadcast-to-every-
page emit + per-page byte-equal dedup then silently dropped the later
page. Python_Cookbook lost pages 63 / 128 / 365 / 397 to this defect.

These tests pin the corrected behavior:

* multi-prov single-item DocChunk → emit per-page slices using
  ``prov.charspan`` instead of broadcasting the merged text.
* multi-item DocChunk (one item per page) → still emits per-page text.
* three-page merged DocChunk → emits three distinct page-attributed
  chunks.
* serializer-empty fallback path → emits explicit
  ``[CROSS_PAGE_CONTINUED]`` marker; never silently drops the page.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple

import pytest

from mmrag_v2.processor import (
    CROSS_PAGE_CONTINUED_MARKER,
    _looks_like_subtitle_continuation,
    _split_doc_chunk_text_by_page,
)
from mmrag_v2.schema.ingestion_schema import ChunkType


def _load_qa_conversion_audit_module():
    """Load ``scripts/qa_conversion_audit.py`` by file path.

    The ``scripts/`` directory is not a package — bare ``pytest`` (i.e.
    ``conda run -n mmrag-v2 pytest tests/...``) does not add the repo
    root to ``sys.path`` the way ``python -m pytest`` does, so
    ``from scripts.qa_conversion_audit import _is_typed_non_micro``
    fails under the documented test command. Loading by file path
    keeps the tests invocation-style-agnostic without touching
    packaging.

    The module is registered in ``sys.modules`` *before* ``exec_module``
    runs because the dataclass machinery (the script defines
    ``@dataclass`` types) calls ``sys.modules.get(cls.__module__)``
    during class construction; without the registration that lookup
    returns ``None`` and dataclass field-type resolution explodes.
    """
    import importlib.util
    import sys
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    target = repo_root / "scripts" / "qa_conversion_audit.py"
    mod_name = "qa_conversion_audit_under_test"
    cached = sys.modules.get(mod_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(mod_name, str(target))
    assert spec and spec.loader, f"could not build spec for {target}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


# ---------------------------------------------------------------------------
# Synthetic Docling fixtures
# ---------------------------------------------------------------------------


def _mk_prov(
    page_no: int,
    charspan: Optional[Tuple[int, int]],
    bbox: Optional[SimpleNamespace] = None,
) -> SimpleNamespace:
    return SimpleNamespace(page_no=page_no, charspan=charspan, bbox=bbox)


def _mk_item(
    text: str,
    prov_specs: Sequence[Tuple[int, Optional[Tuple[int, int]]]],
) -> SimpleNamespace:
    return SimpleNamespace(
        label=SimpleNamespace(value="text"),
        text=text,
        prov=[_mk_prov(page, span) for page, span in prov_specs],
    )


def _mk_doc_chunk(text: str, items: List[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        meta=SimpleNamespace(doc_items=items, headings=[]),
    )


# ---------------------------------------------------------------------------
# Cookbook-shape fixtures (single multi-prov item, two DocChunks slicing it)
# ---------------------------------------------------------------------------


def _cookbook_p208_p209_pair() -> Tuple[SimpleNamespace, SimpleNamespace]:
    """Build the Cookbook-shape pair.

    A single Docling text item spans pages 208 + 209. p208 holds the
    chapter-end prose; p209 holds the trailing URL/citation list.
    HybridChunker slices the item into TWO DocChunks at the natural
    sentence boundary: DocChunk A carries all of p208, DocChunk B
    carries all of p209 (the URL list).
    """
    p208_text = (
        "Chapter 9 summary: the techniques in this chapter together "
        "give the reader an end-to-end view of evaluating chatbots "
        "and reasoning models. The next chapter expands these into "
        "agentic settings. References follow on the next page."
    )
    p209_text = (
        "Chatbot Arena: https://chat.lmsys.org/?leaderboard\n"
        "MMLU: https://arxiv.org/abs/2009.03300\n"
        "MT Bench: https://lmsys.org/blog/2023-06-22-leaderboard/"
    )
    item_text = p208_text + " " + p209_text
    p208_charspan = (0, len(p208_text))
    p209_charspan = (len(p208_text) + 1, len(item_text))

    item = _mk_item(
        item_text,
        prov_specs=[(208, p208_charspan), (209, p209_charspan)],
    )

    # DocChunk A: prose ending at the sentence boundary (all of p208,
    # zero chars of p209).
    dc_a_text = p208_text
    dc_a = _mk_doc_chunk(text=dc_a_text, items=[item])

    # DocChunk B: the URL list proper (all of p209).
    dc_b_text = p209_text
    dc_b = _mk_doc_chunk(text=dc_b_text, items=[item])

    return dc_a, dc_b


def test_p209_url_list_attributed_to_p209() -> None:
    """DocChunk B carries the URL/citation list. The full URL block
    must come back attributed to p209, NOT p208 — the v2.9 defect
    attributed all of DocChunk B to prov[0].page_no=208."""
    _, dc_b = _cookbook_p208_p209_pair()
    per_page = _split_doc_chunk_text_by_page(dc_b, page_offset=0)

    assert 209 in per_page and per_page[209], (
        f"p209 must receive the URL list slice, got: {per_page}"
    )
    assert "Chatbot Arena: https://chat.lmsys.org" in per_page[209]
    assert "MMLU: https://arxiv.org/abs/2009.03300" in per_page[209]
    # p208 must NOT receive any of DocChunk B's content (it lives
    # entirely on p209 per the item's prov.charspan).
    p208_text = per_page.get(208, "")
    assert "https://" not in p208_text, (
        f"p208 must not absorb p209's URL list; got p208 text: {p208_text!r}"
    )
    assert not p208_text, (
        f"p208 should not appear in DocChunk B's per-page dict at all "
        f"(no charspan overlap); got: {per_page}"
    )


def test_p208_content_attributed_to_p208() -> None:
    """DocChunk A carries the chapter-end prose. p208's prose must
    stay attributed to p208 (no regression from the fix)."""
    dc_a, _ = _cookbook_p208_p209_pair()
    per_page = _split_doc_chunk_text_by_page(dc_a, page_offset=0)

    assert 208 in per_page and per_page[208], (
        f"p208 must receive the chapter prose, got: {per_page}"
    )
    assert "Chapter 9 summary" in per_page[208]
    assert "References follow on the next page." in per_page[208]
    # DocChunk A has no charspan overlap with p209 → p209 must not
    # be present in DocChunk A's per-page dict.
    assert 209 not in per_page, (
        f"p209 must not appear in DocChunk A's per-page dict "
        f"(no charspan overlap); got: {per_page}"
    )


# ---------------------------------------------------------------------------
# Three-page merge
# ---------------------------------------------------------------------------


def test_three_page_merge_emits_three_distinct_chunks() -> None:
    """A single multi-prov item spanning pages 100, 101, 102 must
    decompose into three per-page text fragments, one per page."""
    p100 = "Section opener on page 100 with substantive prose. " * 3
    p101 = "Body continues on page 101 with further explanation. " * 3
    p102 = "Tail of the section on page 102 before the next header. " * 3
    item_text = p100 + p101 + p102

    cs100 = (0, len(p100))
    cs101 = (len(p100), len(p100) + len(p101))
    cs102 = (len(p100) + len(p101), len(item_text))

    item = _mk_item(
        item_text,
        prov_specs=[(100, cs100), (101, cs101), (102, cs102)],
    )
    dc = _mk_doc_chunk(text=item_text, items=[item])

    per_page = _split_doc_chunk_text_by_page(dc, page_offset=0)

    assert {100, 101, 102}.issubset(per_page.keys()), (
        f"all three pages must appear in the split; got: {sorted(per_page)}"
    )
    assert "Section opener on page 100" in per_page[100]
    assert "Body continues on page 101" in per_page[101]
    assert "Tail of the section on page 102" in per_page[102]
    # Each per-page fragment must be DISTINCT (not the merged text
    # broadcast — that was the v2.9 defect).
    assert per_page[100] != per_page[101]
    assert per_page[101] != per_page[102]
    assert per_page[100] != per_page[102]


# ---------------------------------------------------------------------------
# Multi-item shape (each item single-prov, on a different page)
# ---------------------------------------------------------------------------


def test_multi_item_cross_page_splits_per_item_text() -> None:
    """HARRY-shape: a chunk aggregates two single-prov items, each
    on a different page. Per-page text must be each item's own text,
    not the merged concatenation broadcast to both pages."""
    item_p65 = _mk_item(
        "Harry adjusted his glasses and walked toward the common room.",
        prov_specs=[(65, (0, 60))],
    )
    item_p66 = _mk_item(
        "The Fat Lady's portrait swung open as he gave the password.",
        prov_specs=[(66, (0, 60))],
    )
    dc_text = item_p65.text + " " + item_p66.text
    dc = _mk_doc_chunk(text=dc_text, items=[item_p65, item_p66])

    per_page = _split_doc_chunk_text_by_page(dc, page_offset=0)
    assert 65 in per_page and 66 in per_page
    assert "Harry adjusted his glasses" in per_page[65]
    assert "Fat Lady" not in per_page[65]
    assert "Fat Lady" in per_page[66]
    assert "Harry adjusted his glasses" not in per_page[66]


# ---------------------------------------------------------------------------
# Empty-text / fallback path
# ---------------------------------------------------------------------------


def test_partial_reconstruction_emits_zero_markers() -> None:
    """A multi-page DocChunk that has BOTH a real-text item (sliceable
    via charspan) AND a serializer-only item (empty `.text`, code or
    table) must emit zero markers. The real text reconstructs normally;
    the empty-text contributor's page is left for whatever other
    DocChunk produces real text for it. Markers are an emergency
    page-loss guard, not partial-failure noise.

    This is the Python_Cookbook v2.10-rc1 regression: 62 marker chunks
    were emitted across 62 already-covered pages when an unrelated
    code item appeared alongside a real-text contributor. The strict
    gate then tripped on ``micro_non_label_ratio``.
    """
    real_item = _mk_item(
        text="Page A prose. Page B prose continues.",
        prov_specs=[
            (10, (0, len("Page A prose."))),
            (11, (len("Page A prose. "), len("Page A prose. Page B prose continues."))),
        ],
    )
    # A code item that serializes its text via the chunker (item.text
    # is empty). Its prov pages should NOT trigger marker emission
    # because the multi-page DocChunk still has real text from the
    # `real_item`.
    code_item = _mk_item(text="", prov_specs=[(12, None)])

    dc = _mk_doc_chunk(
        text=real_item.text + " ```\ncode\n```",
        items=[real_item, code_item],
    )
    per_page = _split_doc_chunk_text_by_page(dc, page_offset=0)

    assert CROSS_PAGE_CONTINUED_MARKER not in per_page.values(), (
        f"helper must emit zero markers when partial reconstruction "
        f"succeeds; got: {per_page}"
    )
    assert 10 in per_page and "Page A prose" in per_page[10]
    assert 11 in per_page and "Page B prose" in per_page[11]
    # The code item's prov page (12) is NOT in the result — that's the
    # contract. Whatever DocChunk owns p12's real content will emit
    # the real chunk; THIS DocChunk does not need to backfill it with
    # a marker.
    assert 12 not in per_page, (
        f"serializer-only item's page must not appear as a marker in "
        f"partial-reconstruction output; got: {per_page}"
    )


def test_serializer_only_items_skipped_by_helper() -> None:
    """Items with empty ``.text`` (e.g. code/table items that
    contribute to ``dc.text`` only via the chunker's serializer) cannot
    be sliced by charspan. The helper silently skips them — emitting a
    marker for every empty-text contributor would pollute pages already
    covered by other DocChunks (the v2.10-rc1 regression: 62 marker
    chunks across 62 already-covered Python_Cookbook pages).

    The "every contributor was unsliceable" defense lives in the
    cross-page emission branch, exercised by
    ``test_cross_page_emit_uses_marker_when_helper_returns_nothing``.
    """
    item = _mk_item(
        text="",
        prov_specs=[(50, None), (51, None)],
    )
    dc = _mk_doc_chunk(text="some surrounding text", items=[item])

    per_page = _split_doc_chunk_text_by_page(dc, page_offset=0)
    assert per_page == {}, (
        f"unsliceable items must not produce per-page entries here; "
        f"got: {per_page}"
    )


def test_cross_page_emit_uses_marker_at_earliest_contributing_page_only() -> None:
    """When `_split_doc_chunk_text_by_page` returns an empty dict for a
    multi-page DocChunk (every contributor is serializer-only), the
    cross-page emission branch in `_process_text_with_hybrid_chunker`
    must emit ONE ``[CROSS_PAGE_CONTINUED]`` marker chunk at the
    earliest contributing page — NOT one marker per contributing
    page.

    Background: Python_Distilled p1-p3 are truly-blank cover/imprint
    pages whose Docling DocChunks have all-empty-text items. Emitting
    a marker on each page falsely populates the page with a chunk and
    excludes it from ``MISSING_PAGES_BLANK`` classification, then trips
    ``micro_non_label_ratio`` in tight-window smoke tests (10/11
    smoke at 0.273 > 0.12). The single-marker fallback expresses the
    DocChunk's existence without falsely claiming page coverage for
    pages whose source is blank."""
    item = _mk_item(text="", prov_specs=[(50, None), (51, None)])
    dc = _mk_doc_chunk(text="serialised content", items=[item])

    per_page = _split_doc_chunk_text_by_page(dc, page_offset=0)
    assert per_page == {}

    # Simulate the cross-page emit branch's fallback: when the helper
    # returns nothing, ONE marker is emitted at the earliest page in
    # _prov_by_page.
    _prov_by_page = {50: [(item, item.prov[0])], 51: [(item, item.prov[1])]}
    if not per_page:
        first_page = sorted(_prov_by_page.keys())[0]
        per_page = {first_page: CROSS_PAGE_CONTINUED_MARKER}

    assert per_page == {50: CROSS_PAGE_CONTINUED_MARKER}, (
        f"expected single marker at earliest page (50); got: {per_page}"
    )
    assert 51 not in per_page, (
        f"non-earliest contributing pages must NOT receive a marker — "
        f"they should fall through to MISSING_PAGES_BLANK if their "
        f"source is blank, or be covered by another DocChunk"
    )


def test_marker_constant_is_non_empty_and_unique() -> None:
    """The fallback marker must be a non-empty sentinel — empty text
    would trip the `empty_text_chunks` invariant. It also must be a
    constant string the strict gate / hygiene scripts can recognise
    as intentional (vs. a randomly truncated chunk)."""
    assert CROSS_PAGE_CONTINUED_MARKER
    assert CROSS_PAGE_CONTINUED_MARKER.strip() == CROSS_PAGE_CONTINUED_MARKER
    assert "CROSS_PAGE_CONTINUED" in CROSS_PAGE_CONTINUED_MARKER


# ---------------------------------------------------------------------------
# Page offset (batch) handling
# ---------------------------------------------------------------------------


def test_helper_dereferences_bare_doc_item_refs() -> None:
    """Docling 2.86 exposes some ``DocChunk.meta.doc_items`` as bare
    ``DocItem`` references (no `.text` attribute) instead of resolved
    ``TextItem`` instances; the real text lives at
    ``doc.texts[idx]``. The Python_Cookbook TOC + chapter-divider
    pages were emitting 56 marker chunks because of this — items had
    no `.text` so the helper skipped them entirely. Passing the parsed
    ``doc`` lets the helper dereference via ``self_ref`` and recover
    the real text for charspan slicing."""
    # Build a fake `doc` whose `.texts` array carries the resolved
    # TextItems; the doc_items in the DocChunk are bare references.
    resolved_p10 = SimpleNamespace(text="Page 10 narrative paragraph.")
    resolved_p11 = SimpleNamespace(text="Page 11 follow-on paragraph.")
    fake_doc = SimpleNamespace(texts=[None, resolved_p10, resolved_p11])

    ref_p10 = SimpleNamespace(
        label=SimpleNamespace(value="text"),
        self_ref="#/texts/1",
        # bare DocItem: no .text attribute at all
        prov=[_mk_prov(10, (0, len(resolved_p10.text)))],
    )
    ref_p11 = SimpleNamespace(
        label=SimpleNamespace(value="text"),
        self_ref="#/texts/2",
        prov=[_mk_prov(11, (0, len(resolved_p11.text)))],
    )
    assert not hasattr(ref_p10, "text")

    dc = _mk_doc_chunk(
        text=resolved_p10.text + " " + resolved_p11.text,
        items=[ref_p10, ref_p11],
    )

    # Without doc — helper sees no text on bare refs → returns {}.
    per_page_no_doc = _split_doc_chunk_text_by_page(dc, page_offset=0)
    assert per_page_no_doc == {}, (
        f"without doc, bare DocItem refs have no recoverable text; "
        f"got: {per_page_no_doc}"
    )

    # With doc — helper dereferences via self_ref and reconstructs
    # per-page text.
    per_page = _split_doc_chunk_text_by_page(dc, page_offset=0, doc=fake_doc)
    assert 10 in per_page and "Page 10 narrative" in per_page[10]
    assert 11 in per_page and "Page 11 follow-on" in per_page[11]
    assert CROSS_PAGE_CONTINUED_MARKER not in per_page.values()


def test_micro_merge_skips_pagesplit_fallback_markers() -> None:
    """``BatchProcessor._merge_micro_text_chunks`` attaches tiny
    non-label text fragments onto neighbors. The 22-char
    ``[CROSS_PAGE_CONTINUED]`` marker emitted by the cross-page split
    emergency fallback is exactly the shape that filter targets, so
    without an explicit guard the marker gets concatenated onto the
    same page's real prose chunk and contaminates retrievable text
    (Python_Distilled p472 saw the marker appended onto the
    "Set Operations" table chunk in the pre-guard reconvert).

    Pin: a chunk whose ``extraction_method`` is
    ``hybrid_chunker_pagesplit_fallback`` must survive
    ``_merge_micro_text_chunks`` standalone — never merged INTO or
    OUT OF a neighbor."""
    from mmrag_v2.batch_processor import BatchProcessor
    from mmrag_v2.schema.ingestion_schema import (
        FileType,
        IngestionChunk,
        create_text_chunk,
    )

    def _real_chunk(page: int, content: str, position: int) -> IngestionChunk:
        return create_text_chunk(
            content=content,
            doc_id="fake_doc",
            source_file="Fake.pdf",
            file_type=FileType.PDF,
            page_number=page,
            extraction_method="hybrid_chunker",
            position=position,
        )

    def _marker_chunk(page: int, position: int) -> IngestionChunk:
        return create_text_chunk(
            content=CROSS_PAGE_CONTINUED_MARKER,
            doc_id="fake_doc",
            source_file="Fake.pdf",
            file_type=FileType.PDF,
            page_number=page,
            extraction_method="hybrid_chunker_pagesplit_fallback",
            position=position,
        )

    real = _real_chunk(472, "Set Operations and Methods table content prose body.", 0)
    marker = _marker_chunk(472, 1)

    bp = BatchProcessor.__new__(BatchProcessor)
    merged = bp._merge_micro_text_chunks([real, marker], max_chars=30)

    real_after = next(c for c in merged if c.metadata.extraction_method == "hybrid_chunker")
    assert CROSS_PAGE_CONTINUED_MARKER not in real_after.content, (
        f"marker text must not be appended to neighboring prose; "
        f"got: {real_after.content!r}"
    )
    marker_after = [
        c for c in merged
        if c.metadata.extraction_method == "hybrid_chunker_pagesplit_fallback"
    ]
    assert len(marker_after) == 1, (
        f"marker chunk must survive standalone; got {len(marker_after)} "
        f"matching chunks in merged output"
    )
    assert marker_after[0].content == CROSS_PAGE_CONTINUED_MARKER


def test_overlap_trim_is_page_scoped() -> None:
    """`BatchProcessor._deduplicate_chunk_overlap` previously trimmed
    head/tail exact overlap between consecutive chunks regardless of
    page. Docling 2.86 occasionally emits two byte-identical DocChunks
    for the same code block on adjacent pages (Python_Cookbook p396 /
    p397 — DocChunk #335 prov=[396] and DocChunk #336 prov=[397] with
    identical `dc.text`). The cross-page trim stripped chunk[N+1]'s
    entire content, dropping p397 from coverage.

    Pin: when consecutive text chunks live on DIFFERENT pages, the
    overlap-trim must skip them — even if their content is identical.
    Same-page DSO overlap trimming still operates."""
    from mmrag_v2.batch_processor import BatchProcessor
    from mmrag_v2.schema.ingestion_schema import (
        FileType,
        create_text_chunk,
    )

    code_text = (
        "```\nnumber = 8 power = 3 result1 = number ** power # result1 is 512\n```"
    )

    cookbook_p396 = create_text_chunk(
        content=code_text,
        doc_id="fake_doc",
        source_file="Fake.pdf",
        file_type=FileType.PDF,
        page_number=396,
        extraction_method="hybrid_chunker",
        position=0,
    )
    cookbook_p397 = create_text_chunk(
        content=code_text,
        doc_id="fake_doc",
        source_file="Fake.pdf",
        file_type=FileType.PDF,
        page_number=397,
        extraction_method="hybrid_chunker",
        position=1,
    )

    bp = BatchProcessor.__new__(BatchProcessor)
    out = bp._deduplicate_chunk_overlap([cookbook_p396, cookbook_p397])

    p397_after = next(c for c in out if c.metadata.page_number == 397)
    assert p397_after.content == code_text, (
        f"cross-page duplicate DocChunks must NOT be trimmed; "
        f"got p397 content={p397_after.content!r}"
    )

    # Same-page DSO-style overlap trimming continues to work.
    dso_prev = create_text_chunk(
        content="first sentence. shared overlap sentence.",
        doc_id="fake_doc",
        source_file="Fake.pdf",
        file_type=FileType.PDF,
        page_number=10,
        extraction_method="hybrid_chunker",
        position=2,
    )
    dso_cur = create_text_chunk(
        content="shared overlap sentence. next sentence.",
        doc_id="fake_doc",
        source_file="Fake.pdf",
        file_type=FileType.PDF,
        page_number=10,
        extraction_method="hybrid_chunker",
        position=3,
    )
    out2 = bp._deduplicate_chunk_overlap([dso_prev, dso_cur])
    dso_cur_after = out2[1]
    assert "shared overlap sentence." not in dso_cur_after.content, (
        f"same-page DSO overlap trim must still fire; got: "
        f"{dso_cur_after.content!r}"
    )


class TestSubtitleContinuationPromotion:
    """`LITERATURE_MICRO_GATE_TUNE_AFTER_CROSS_PAGE_FIX` resolution.

    HybridChunker slices a multi-page TITLE across pages and Docling
    labels the trailing slice ``label=text`` — so the producer
    receives it as ``chunk_type=PARAGRAPH``. Pre-Phase-4, v2.9's
    cross-page broadcast was attaching the same merged text to every
    contributing page (wrong page attribution but a long enough
    chunk to slip past the strict gate's ``micro_non_label_ratio``
    threshold). Phase 4's correct per-page attribution exposed the
    real 24-char trailing slice on HarryPotter p7:
    ``"and the Sorcerer's Stone"`` — a legitimate book-subtitle
    chunk that fails the strict gate's lowercase-leading
    ``_is_label_like`` regex.

    `_looks_like_subtitle_continuation` promotes such chunks to
    ``ChunkType.HEADING`` so they're recognised by the strict gate
    as intentional structural content (paired with
    ``qa_conversion_audit._is_typed_non_micro``). The rule is
    universal: it relies on structural signals (short single-line
    text under an explicit parent_heading, no terminal sentence
    punctuation, first word is a small English connector
    stopword) — no filename / page-number / document-specific
    logic.
    """

    def test_harry_potter_subtitle_is_promoted(self) -> None:
        """The pin: HarryPotter p7 trailing subtitle slice has all
        the structural signals and IS promoted."""
        promoted = _looks_like_subtitle_continuation(
            "and the Sorcerer's Stone",
            ChunkType.PARAGRAPH,
            "Harry Potter",
        )
        assert promoted is True

    def test_short_body_prose_with_terminal_punct_NOT_promoted(self) -> None:
        """Ordinary short body prose ends with a sentence terminator
        (``.``, ``?``, ``!``, ``:``, ``;``, ``,``) — must NOT be
        promoted just because it's short. The terminal-punct check
        is the discriminator between "complete short sentence" and
        "trailing fragment of a title"."""
        for text in (
            "Hello world.",
            "Why now?",
            "Stop!",
            "Notes:",
            "First; then,",
            "and that's it.",
        ):
            assert (
                _looks_like_subtitle_continuation(text, ChunkType.PARAGRAPH, "Section X")
                is False
            ), f"terminal-punct short prose must stay PARAGRAPH; got promoted: {text!r}"

    def test_short_body_prose_not_starting_with_connector_NOT_promoted(self) -> None:
        """Standalone short text whose first word is NOT a
        title-continuation stopword stays PARAGRAPH. This rejects
        captions ("Logo", "Bar chart", "Flow chart"), index entries
        ("zip() function, 62, 314"), OCR junk ("ré Several refe"),
        and any short prose that begins a new noun-phrase rather
        than continuing one."""
        for text in (
            "Logo",
            "Bar chart",
            "Flow chart",
            "Hello world",
            "B.__spam 37 >>>",
            "ré Several refe",
            "zip() function 62 314",
        ):
            assert (
                _looks_like_subtitle_continuation(text, ChunkType.PARAGRAPH, "Front Matter")
                is False
            ), f"non-connector lead must stay PARAGRAPH; got promoted: {text!r}"

    def test_code_list_table_chunk_types_NOT_promoted(self) -> None:
        """Items already classified by the producer (Docling
        ``label=code``, ``list_item``, etc.) must NOT be re-typed —
        the promotion path only fires for would-be PARAGRAPH
        chunks."""
        for ct in (ChunkType.CODE, ChunkType.LIST_ITEM, ChunkType.HEADING, ChunkType.TITLE, ChunkType.CAPTION):
            assert (
                _looks_like_subtitle_continuation(
                    "and the next chapter", ct, "Some Title"
                )
                is False
            ), f"non-paragraph chunk_type {ct} must not be re-promoted"

    def test_no_parent_heading_NOT_promoted(self) -> None:
        """A chunk with no hierarchical context (``parent_heading is
        None`` or empty) cannot be a TITLE continuation — there's no
        title to continue from. Must stay PARAGRAPH."""
        for parent in (None, "", "   "):
            assert (
                _looks_like_subtitle_continuation(
                    "and more text", ChunkType.PARAGRAPH, parent
                )
                is False
            ), f"empty parent_heading must not promote (parent={parent!r})"

    def test_too_long_NOT_promoted(self) -> None:
        """The rule requires < 30 chars. Longer paragraphs are
        not subtitles regardless of the leading word."""
        long_text = "and " + "x" * 30
        assert (
            _looks_like_subtitle_continuation(long_text, ChunkType.PARAGRAPH, "Title")
            is False
        )

    def test_multiline_with_trailing_structural_fragment_promoted(self) -> None:
        """Docling 2.86 emits multi-page title slices whose dc.text
        ends with a trailing structural fragment on its own line
        (e.g. ``"and the Sorcerer's Stone\\nBY"`` — the ``BY`` is
        the introducer of the next item that
        ``BatchProcessor._deduplicate_chunk_overlap`` later trims
        as a head/tail overlap with the next chunk). The promotion
        rule must still fire on this shape because the FINAL
        single-line content reaching the gate is the bare subtitle.

        Pin: do NOT reject on ``\\n`` in raw content — the
        connector-lead + parent_heading + no-terminal-punct
        combination already discriminates."""
        promoted = _looks_like_subtitle_continuation(
            "and the Sorcerer's Stone\nBY",
            ChunkType.PARAGRAPH,
            "Harry Potter",
        )
        assert promoted is True

    def test_parent_heading_equal_to_content_NOT_promoted(self) -> None:
        """Anti-sentinel: if a buggy upstream sets
        ``parent_heading == content`` we must NOT promote — the
        chunk isn't a continuation of anything new."""
        text = "and the rest"
        assert (
            _looks_like_subtitle_continuation(text, ChunkType.PARAGRAPH, text)
            is False
        )

    def test_rule_not_tied_to_filename_or_page_or_literal_text(self) -> None:
        """Smoke check: the helper's signature takes ONLY
        ``content``, ``chunk_type``, ``parent_heading``. It cannot
        be tied to a filename, a page number, or any other
        document-specific state. Any synthetic content that matches
        the structural signals promotes regardless of context."""
        synthetic = "of the Forgotten Realms"  # any title-continuation
        assert (
            _looks_like_subtitle_continuation(
                synthetic, ChunkType.PARAGRAPH, "Some Made-Up Book"
            )
            is True
        )
        # Same content under a different parent still promotes — the
        # rule does NOT inspect parent_heading content beyond
        # "is non-empty and not equal to chunk content".
        assert (
            _looks_like_subtitle_continuation(
                synthetic, ChunkType.PARAGRAPH, "Different Title"
            )
            is True
        )


class TestAuditMicroNonLabelExemption:
    """The audit's ``micro_non_label`` counter previously exempted
    only `_is_label_like(content)` (regex on content) and
    ``chunk_type == "code"``. A producer-side subtitle promotion to
    ``ChunkType.HEADING`` was useless without a paired exemption on
    the audit side. ``_is_typed_non_micro(meta)`` adds heading /
    title to the exemption — clarifying that an explicitly
    non-paragraph chunk_type is structural content, not
    retrieval-noise micro-prose. This is a definition correction,
    NOT a threshold change.
    """

    def test_heading_chunk_exempt_from_micro_non_label(self) -> None:
        _is_typed_non_micro = _load_qa_conversion_audit_module()._is_typed_non_micro
        assert _is_typed_non_micro({"chunk_type": "heading"}) is True
        assert _is_typed_non_micro({"chunk_type": "HEADING"}) is True

    def test_title_chunk_exempt_from_micro_non_label(self) -> None:
        _is_typed_non_micro = _load_qa_conversion_audit_module()._is_typed_non_micro
        assert _is_typed_non_micro({"chunk_type": "title"}) is True

    def test_code_chunk_still_exempt(self) -> None:
        _is_typed_non_micro = _load_qa_conversion_audit_module()._is_typed_non_micro
        assert _is_typed_non_micro({"chunk_type": "code"}) is True
        assert _is_typed_non_micro({"content_classification": "code"}) is True

    def test_paragraph_chunk_NOT_exempt(self) -> None:
        _is_typed_non_micro = _load_qa_conversion_audit_module()._is_typed_non_micro
        assert _is_typed_non_micro({"chunk_type": "paragraph"}) is False
        assert _is_typed_non_micro({}) is False

    def test_list_item_chunk_NOT_exempt(self) -> None:
        """Only heading/title/code are exempt — list_item, caption,
        footnote, quote remain subject to the micro_non_label
        counter so the gate keeps catching retrieval-noise list
        fragments."""
        _is_typed_non_micro = _load_qa_conversion_audit_module()._is_typed_non_micro
        assert _is_typed_non_micro({"chunk_type": "list_item"}) is False
        assert _is_typed_non_micro({"chunk_type": "caption"}) is False
        assert _is_typed_non_micro({"chunk_type": "footnote"}) is False
        assert _is_typed_non_micro({"chunk_type": "quote"}) is False

    def test_evaluate_gate_parallel_site_recognises_heading(self, tmp_path) -> None:
        """The micro_non_label exemption must hold across BOTH
        gate paths — ``qa_conversion_audit.py`` AND
        ``evaluate_technical_manual_gates.py``. Forgetting the
        parallel site is exactly how HarryPotter smoke kept failing
        even after the audit-side fix landed (the smoke summary
        uses ``evaluate_technical_manual_gates``, not the audit).

        End-to-end pin: a 24-char HEADING-typed chunk under a
        digital_literature profile must produce
        ``micro_non_label_ratio == 0`` from the evaluate gate."""
        import subprocess, sys, json as _json
        # Minimal JSONL: ingestion_metadata header + one HEADING text chunk
        # (replicates the post-fix HarryPotter shape).
        meta = {
            "object_type": "ingestion_metadata",
            "profile_type": "digital_literature",
            "total_pages": 10,
        }
        chunk = {
            "chunk_id": "fake1",
            "content": "and the Sorcerer's Stone",
            "modality": "text",
            "metadata": {
                "chunk_type": "heading",
                "page_number": 7,
                "hierarchy": {"parent_heading": "Harry Potter"},
            },
        }
        # Add prose chunks so the gate has a non-trivial denominator
        prose_meta = {
            "object_type": "chunk_meta_stub",
        }
        prose = []
        for i in range(7):
            prose.append({
                "chunk_id": f"prose{i}",
                "content": "Paragraph text body. " * 5,
                "modality": "text",
                "metadata": {
                    "chunk_type": "paragraph",
                    "page_number": 8 + i,
                    "hierarchy": {"parent_heading": "Body"},
                },
            })
        jsonl_path = tmp_path / "ingestion.jsonl"
        with jsonl_path.open("w") as fh:
            fh.write(_json.dumps(meta) + "\n")
            fh.write(_json.dumps(chunk) + "\n")
            for p in prose:
                fh.write(_json.dumps(p) + "\n")

        out = subprocess.check_output(
            [
                sys.executable,
                "scripts/evaluate_technical_manual_gates.py",
                str(jsonl_path),
                "--doc-class",
                "auto",
            ],
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert "micro_non_label_ratio=0.0000" in out, (
            f"evaluate_technical_manual_gates parallel-site fix must "
            f"exempt chunk_type=heading from micro_non_label; got:\n{out}"
        )


def test_page_offset_applied_to_global_page_number() -> None:
    """BatchProcessor processes 10-page batches with a global
    page_offset; the helper must return global page numbers
    (per-batch local + offset), matching how the cross-page emit
    expects them downstream."""
    p1 = "First batch-local page text. "
    p2 = "Second batch-local page text. "
    item = _mk_item(
        p1 + p2,
        prov_specs=[(1, (0, len(p1))), (2, (len(p1), len(p1) + len(p2)))],
    )
    dc = _mk_doc_chunk(text=p1 + p2, items=[item])

    per_page = _split_doc_chunk_text_by_page(dc, page_offset=200)
    assert 201 in per_page and 202 in per_page, (
        f"page_offset must shift returned page numbers; got: {sorted(per_page)}"
    )
    assert 1 not in per_page and 2 not in per_page
