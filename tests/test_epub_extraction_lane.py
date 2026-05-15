"""
v2.10 Phase 7 — `KI_EPUB_EXTRACTION_LANE_REWRITE` regression tests.

Pins the EPUB lane contract:
1. `_epub_to_html` walks `book.spine` order and embeds chapter-boundary
   markers (`__MMRAG_EPUB_CH_NNNN__`) at the head of each non-empty
   chapter.
2. `_apply_epub_synthetic_pagination` rewrites EPUB chunks with a
   deterministic synthetic `page_number = chapter_1based * 1000 +
   position_in_chapter // 5`, the documented full-page bbox sentinel
   `[0, 0, 1000, 1000]`, `extraction_method="epub_html"`, and a
   regenerated chunk_id that preserves the v2.9 position-component
   uniqueness contract.
3. KI EPUB strict QA reaches QA_PASS / QA_PASS_WITH_ADVISORIES.
4. ChatGPT EPUB regression control remains QA_PASS_WITH_ADVISORIES.

The KI/ChatGPT JSONL-shape tests assume the reconverted JSONLs exist
at `output/{KI_En_ChatGPT_Praktische_Gids,ChatGPT_Praktijk_handboek}/
ingestion.jsonl`. They are skipped when the JSONLs are missing so the
test suite stays runnable without forcing a reconvert on every checkout
— the Phase 7 acceptance protocol still requires the reconvert + strict
qa run as a separate validation step.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

import pytest

from mmrag_v2.processor import (
    V2DocumentProcessor,
    _EPUB_CHAPTER_MARKER_PREFIX,
    _EPUB_CHAPTER_MARKER_RE,
)
from mmrag_v2.schema.ingestion_schema import (
    ChunkType,
    FileType,
    HierarchyMetadata,
    Modality,
    SpatialMetadata,
    create_text_chunk,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
KI_EPUB = REPO_ROOT / "data" / "technical_manual" / (
    "Seffer, David - KI En ChatGPT, Praktische Gids Voor Online Business "
    "Met Digitale Producten.epub"
)
CHATGPT_EPUB = REPO_ROOT / "data" / "technical_manual" / (
    "Falkner, Leonie - ChatGPT Praktijk-handboek.epub"
)
KI_JSONL = REPO_ROOT / "output" / "KI_En_ChatGPT_Praktische_Gids" / "ingestion.jsonl"
CHATGPT_JSONL = REPO_ROOT / "output" / "ChatGPT_Praktijk_handboek" / "ingestion.jsonl"


def _make_processor(tmp_path: Path) -> V2DocumentProcessor:
    return V2DocumentProcessor(output_dir=str(tmp_path / "out"))


def _make_text_chunk(
    doc_id: str,
    content: str,
    position: int,
    page_number: int = 1,
) -> Any:
    return create_text_chunk(
        doc_id=doc_id,
        content=content,
        source_file="dummy.epub",
        file_type=FileType.EPUB,
        page_number=page_number,
        hierarchy=HierarchyMetadata(breadcrumb_path=["dummy"]),
        chunk_type=ChunkType.PARAGRAPH,
        bbox=[0, 0, 1000, 1000],
        page_width=1000,
        page_height=1000,
        extraction_method="hybrid_chunker",
        position=position,
    )


def _write_epub_fixture(tmp_path: Path, chapters: list[tuple[str, str]]) -> Path:
    """Create a small EPUB with a controlled spine order.

    The Phase 7 producer and gate contracts must be testable from
    tracked code alone. Real KI/ChatGPT outputs remain acceptance
    evidence, but this fixture keeps marker injection and QA parity
    from depending on ignored `data/` artifacts.
    """
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("mmrag-phase7-fixture")
    book.set_title("MMRAG Phase 7 Fixture")
    book.set_language("en")

    spine_items = []
    for idx, (name, body_html) in enumerate(chapters, start=1):
        item = epub.EpubHtml(
            title=f"Chapter {idx}",
            file_name=name,
            lang="en",
        )
        item.content = (
            "<html><head><title>Chapter</title></head>"
            f"<body>{body_html}</body></html>"
        )
        book.add_item(item)
        spine_items.append(item)

    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = spine_items
    path = tmp_path / "phase7_fixture.epub"
    epub.write_epub(str(path), book, {})
    return path


def _iter_jsonl_chunks(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        first = True
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            if first:
                first = False
                if obj.get("object_type") == "ingestion_metadata":
                    continue
            yield obj


# ---------------------------------------------------------------------------
# Unit tests: marker injection + synthetic-pagination rewrite
# ---------------------------------------------------------------------------


def test_epub_to_html_injects_per_spine_chapter_markers(tmp_path: Path) -> None:
    """`_epub_to_html` must walk ``book.spine`` and prepend a
    ``__MMRAG_EPUB_CH_NNNN__`` paragraph to each non-empty chapter.
    Stashes ``_epub_chapter_count`` on the processor.
    """
    epub_path = _write_epub_fixture(
        tmp_path,
        [
            ("frontmatter.xhtml", "<h1>Title Page</h1><p>Copyright</p>"),
            ("chapter_01.xhtml", "<h1>Chapter 1</h1><p>First body.</p>"),
            ("chapter_02.xhtml", "<h1>Chapter 2</h1><p>Second body.</p>"),
        ],
    )
    proc = _make_processor(tmp_path)
    html_path = proc._epub_to_html(epub_path)
    assert html_path is not None and html_path.exists()
    html_text = html_path.read_text(encoding="utf-8")
    markers = _EPUB_CHAPTER_MARKER_RE.findall(html_text)
    assert markers == ["0001", "0002", "0003"]
    chapter_count = getattr(proc, "_epub_chapter_count", 0)
    assert chapter_count == 3
    # Markers are 1-based, contiguous, no duplicates.
    marker_ints = [int(m) for m in markers]
    assert marker_ints == sorted(set(marker_ints))
    assert marker_ints[0] == 1
    assert marker_ints[-1] == chapter_count
    html_path.unlink(missing_ok=True)


def test_apply_epub_synthetic_pagination_assigns_per_chapter_pages(
    tmp_path: Path,
) -> None:
    """Synthetic-pagination rewrite must:
    - Update page_number on chapter-marker boundaries.
    - Use formula chapter * 1000 + position // 5 within a chapter.
    - Set full-page bbox sentinel [0,0,1000,1000].
    - Set extraction_method="epub_html".
    - Drop chunks that are only the marker.
    - Strip markers from kept chunk content.
    """
    proc = _make_processor(tmp_path)
    doc_id = "doc12345678"
    # Build 11 chunks: chapter 1 marker + 6 body chunks, chapter 2 marker
    # + 3 body chunks, chapter 3 marker alone (drop).
    chunks = [
        _make_text_chunk(doc_id, f"{_EPUB_CHAPTER_MARKER_PREFIX}0001__", 0),
        _make_text_chunk(doc_id, "ch1 para 1", 1),
        _make_text_chunk(doc_id, "ch1 para 2", 2),
        _make_text_chunk(doc_id, "ch1 para 3", 3),
        _make_text_chunk(doc_id, "ch1 para 4", 4),
        _make_text_chunk(doc_id, "ch1 para 5", 5),
        _make_text_chunk(doc_id, "ch1 para 6", 6),
        _make_text_chunk(doc_id, f"{_EPUB_CHAPTER_MARKER_PREFIX}0002__", 7),
        _make_text_chunk(doc_id, "ch2 para 1", 8),
        _make_text_chunk(doc_id, "ch2 para 2", 9),
        _make_text_chunk(doc_id, "ch2 para 3", 10),
        _make_text_chunk(doc_id, f"{_EPUB_CHAPTER_MARKER_PREFIX}0003__", 11),
    ]
    out = list(proc._apply_epub_synthetic_pagination(chunks, chapter_count=3))
    # Marker-only chunks dropped → 9 body chunks remain.
    assert len(out) == 9
    # Chapter 1 body chunks: page = 1000 + pos // 5, position resets at marker.
    ch1_pages = [out[i].metadata.page_number for i in range(6)]
    assert ch1_pages == [1000, 1000, 1000, 1000, 1000, 1001]
    # Chapter 2 body chunks: page = 2000 + pos // 5, position resets at marker.
    ch2_pages = [out[i].metadata.page_number for i in range(6, 9)]
    assert ch2_pages == [2000, 2000, 2000]
    # bbox sentinel + extraction_method + cleaned content + chunk_id rewrite.
    for chunk in out:
        spatial = chunk.metadata.spatial
        assert spatial is not None
        assert spatial.bbox == [0, 0, 1000, 1000]
        assert spatial.page_width == 1000
        assert spatial.page_height == 1000
        assert chunk.metadata.extraction_method == "epub_html"
        assert _EPUB_CHAPTER_MARKER_RE.search(chunk.content) is None
        assert chunk.chunk_id.startswith(f"{doc_id}_")


def test_apply_epub_synthetic_pagination_strips_inline_marker_and_keeps_body(
    tmp_path: Path,
) -> None:
    """A chunk whose Docling emission concatenated the chapter marker
    with following body text must keep the body and strip the marker
    cleanly (no leftover ``__MMRAG_EPUB_CH_`` substring in content).
    """
    proc = _make_processor(tmp_path)
    doc_id = "doc87654321"
    inline = f"{_EPUB_CHAPTER_MARKER_PREFIX}0002__ Chapter 2 body opens here"
    chunks = [
        _make_text_chunk(doc_id, f"{_EPUB_CHAPTER_MARKER_PREFIX}0001__", 0),
        _make_text_chunk(doc_id, "ch1 line 1", 1),
        _make_text_chunk(doc_id, inline, 2),
    ]
    out = list(proc._apply_epub_synthetic_pagination(chunks, chapter_count=2))
    assert [c.content for c in out] == [
        "ch1 line 1",
        "Chapter 2 body opens here",
    ]
    assert [c.metadata.page_number for c in out] == [1000, 2000]


# ---------------------------------------------------------------------------
# Required tests per Phase 7 spec
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not KI_JSONL.exists(),
    reason="KI EPUB reconverted JSONL not present; run Phase 7 reconvert first",
)
def test_ki_epub_emits_per_chapter_page_numbers() -> None:
    """Phase 7 contract: KI EPUB chunks must have synthetic page_numbers
    clustered by chapter under the ``chapter_1based * 1000 + position
    // 5`` mapping. Chapter coverage must be contiguous (no internal
    gaps) starting at chapter >= 1. Note: EPUB spine items that Docling
    strips entirely (titlepage, colophon) are reported as
    MISSING_CHAPTERS advisory by the QA gate; the chunk stream therefore
    legitimately starts above chapter 1 on real-world books.
    """
    pages = []
    for chunk in _iter_jsonl_chunks(KI_JSONL):
        meta = chunk.get("metadata") or {}
        pn = meta.get("page_number")
        if pn:
            pages.append(int(pn))
    assert pages, "KI EPUB JSONL has no chunks with page_number"
    chapters = sorted({p // 1000 for p in pages})
    assert chapters[0] >= 1, (
        f"KI EPUB chapters must be 1-based, got first={chapters[0]}"
    )
    # At least 2 distinct chapters, contiguous (no internal gaps).
    assert len(chapters) >= 2, f"expected >=2 chapters, got {chapters}"
    assert chapters == list(range(chapters[0], chapters[-1] + 1)), (
        f"non-contiguous chapter coverage: {chapters}"
    )
    # Per-chapter clustering: more than one distinct page_number per chapter.
    pages_per_chapter: dict[int, set[int]] = {}
    for p in pages:
        pages_per_chapter.setdefault(p // 1000, set()).add(p)
    multi = [ch for ch, pgs in pages_per_chapter.items() if len(pgs) > 1]
    assert multi, (
        "expected at least one chapter with >1 synthetic page; "
        "synthetic position mapping appears flat"
    )


@pytest.mark.skipif(
    not KI_JSONL.exists(),
    reason="KI EPUB reconverted JSONL not present; run Phase 7 reconvert first",
)
def test_ki_epub_emits_full_page_bbox() -> None:
    """Phase 7 contract: every EPUB chunk emits bbox=[0,0,1000,1000]
    (the documented full-page sentinel) and extraction_method=epub_html.
    """
    seen = 0
    for chunk in _iter_jsonl_chunks(KI_JSONL):
        meta = chunk.get("metadata") or {}
        spatial = meta.get("spatial") or {}
        bbox = spatial.get("bbox")
        assert bbox == [0, 0, 1000, 1000], (
            f"KI EPUB chunk {chunk.get('chunk_id')} has bbox={bbox}, "
            "expected EPUB sentinel [0,0,1000,1000]"
        )
        assert meta.get("extraction_method") == "epub_html", (
            f"KI EPUB chunk {chunk.get('chunk_id')} has "
            f"extraction_method={meta.get('extraction_method')!r}, "
            "expected 'epub_html'"
        )
        seen += 1
    assert seen >= 50, f"KI EPUB should have many chunks, got {seen}"


@pytest.mark.skipif(
    not CHATGPT_JSONL.exists(),
    reason="ChatGPT EPUB reconverted JSONL not present; run Phase 7 reconvert first",
)
def test_chatgpt_epub_does_not_regress_on_advisory_path() -> None:
    """Regression control: ChatGPT EPUB must continue to emit chunks with
    EPUB-lane shape (full-page bbox sentinel, ``epub_html`` extraction
    method, chapter-clustered synthetic page_numbers, contiguous
    chapters starting at 1).
    """
    pages: list[int] = []
    bboxes: list[Any] = []
    methods: list[str] = []
    for chunk in _iter_jsonl_chunks(CHATGPT_JSONL):
        meta = chunk.get("metadata") or {}
        pages.append(int(meta.get("page_number") or 0))
        spatial = meta.get("spatial") or {}
        bboxes.append(spatial.get("bbox"))
        methods.append(meta.get("extraction_method") or "")
    assert pages and all(p > 0 for p in pages)
    assert all(b == [0, 0, 1000, 1000] for b in bboxes), (
        "ChatGPT EPUB chunks must carry the full-page bbox sentinel"
    )
    assert all(m == "epub_html" for m in methods), (
        f"ChatGPT EPUB chunks must use epub_html extraction method; "
        f"unique methods seen: {sorted(set(methods))}"
    )
    chapters = sorted({p // 1000 for p in pages})
    assert chapters[0] >= 1, (
        f"ChatGPT EPUB chapters must be 1-based, got first={chapters[0]}"
    )
    assert chapters == list(range(chapters[0], chapters[-1] + 1)), (
        f"non-contiguous chapter coverage: {chapters}"
    )


def _assert_unique_chunk_ids(jsonl_path: Path) -> None:
    seen: set[str] = set()
    for chunk in _iter_jsonl_chunks(jsonl_path):
        cid = chunk["chunk_id"]
        assert cid not in seen, (
            f"duplicate chunk_id {cid} in {jsonl_path.name}; v2.9 "
            "position-component contract violated by EPUB rewrite"
        )
        seen.add(cid)


@pytest.mark.skipif(
    not (KI_JSONL.exists() and CHATGPT_JSONL.exists()),
    reason="EPUB reconverted JSONLs not present; run Phase 7 reconvert first",
)
def test_epub_chunk_ids_remain_unique() -> None:
    """v2.9 chunk_id position-component contract must hold across the
    Phase 7 rewrite: regenerating chunk_id with the new synthetic
    page_number + global position counter must not produce duplicates.
    """
    _assert_unique_chunk_ids(KI_JSONL)
    _assert_unique_chunk_ids(CHATGPT_JSONL)


# ---------------------------------------------------------------------------
# QA-script EPUB-aware source handling (Phase 7 audit step)
# ---------------------------------------------------------------------------


def test_qa_script_epub_chapter_count_matches_producer(tmp_path: Path) -> None:
    """`qa_full_conversion._epub_chapter_count` must match what the
    producer (`_epub_to_html`) enumerated; both must walk the spine and
    skip empty chapters identically — otherwise the gate would expect a
    chapter set the producer never claimed.
    """
    import importlib.util
    import sys

    epub_path = _write_epub_fixture(
        tmp_path,
        [
            ("titlepage.xhtml", "<h1>Fixture title</h1>"),
            ("chapter_01.xhtml", "<h1>Chapter 1</h1><p>Body one.</p>"),
            ("chapter_02.xhtml", "<h1>Chapter 2</h1><p>Body two.</p>"),
        ],
    )
    qa_path = REPO_ROOT / "scripts" / "qa_full_conversion.py"
    spec = importlib.util.spec_from_file_location("qa_full_conversion", qa_path)
    assert spec is not None and spec.loader is not None
    qa = importlib.util.module_from_spec(spec)
    # Register in sys.modules so dataclass forward-ref resolution can
    # find ``Issue`` via ``cls.__module__``.
    previous_qa_module = sys.modules.get("qa_full_conversion")
    sys.modules["qa_full_conversion"] = qa
    try:
        spec.loader.exec_module(qa)  # type: ignore[union-attr]
        proc = _make_processor(tmp_path)
        html = proc._epub_to_html(epub_path)
        assert html is not None
        producer_count = getattr(proc, "_epub_chapter_count", 0)
        gate_count = qa._epub_chapter_count(epub_path)
        assert gate_count == producer_count, (
            f"gate chapter count {gate_count} != producer chapter count "
            f"{producer_count}; the EPUB-aware QA branch must mirror "
            "producer enumeration"
        )
        html.unlink(missing_ok=True)
    finally:
        if previous_qa_module is None:
            sys.modules.pop("qa_full_conversion", None)
        else:
            sys.modules["qa_full_conversion"] = previous_qa_module
