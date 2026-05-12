"""Phase 4 Step 3 — strict-gate corruption filter regression tests.

Covers the chunk-level pre-write quarantine that drops chunks matching
the strict-gate `LOCALIZED_CORRUPTION` patterns. The Combat Aircraft
p66 magazine-font garble is the canonical case.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mmrag_v2.batch_processor import BatchProcessor
from mmrag_v2.schema.ingestion_schema import (
    FileType,
    HierarchyMetadata,
    create_image_chunk,
    create_table_chunk,
    create_text_chunk,
)


def _bp() -> BatchProcessor:
    bp = BatchProcessor.__new__(BatchProcessor)
    bp._intelligence_metadata = {}
    bp._doc_hash = "deadbeef1234"
    bp._current_pdf_path = Path("/tmp/test.pdf")
    return bp


# ---------------------------------------------------------------------------
# _is_corrupted_chunk_content — pattern-level positive / negative
# ---------------------------------------------------------------------------


def test_combat_p66_gibberish_signature_detected():
    """Plan v2.9 Phase E refinement (2026-05-11): a >30K-char table
    chunk with fewer than 10 four-letter words per 1K chars is
    classified as gibberish. Combat p66 after the 2026-05-11
    reconvert had this exact signature: 40568 chars, 363 four-letter
    words = 8.9 w/k. Real tables in the corpus measured 20-54 w/k."""
    # Synthesize a chunk that looks like the Combat p66 gibberish:
    # ~30K of short-letter-with-punctuation patterns, with a few real
    # words sprinkled in (squadron names that survived).
    base = "| [il :   ltJ!  nf r!  Ill  r!·! l l:  .[lr!  Jl  1 ]1   r!'  Dl=  r!'  | "
    real_words = " Aircraft Tail Code Squadron Force Wing Nineteenth Antonio Texas "
    # Build to ~35K chars; embed real words sparsely (1 set per ~3K chars)
    parts = []
    while sum(len(p) for p in parts) < 35000:
        parts.append(base * 30)
        parts.append(real_words)
    chunk_content = "".join(parts)
    assert BatchProcessor._is_corrupted_chunk_content(chunk_content) is True


def test_real_large_table_not_classified_gibberish():
    """Negative-control: a legitimate large table with word_density
    in the normal 20-54 w/k range must NOT trigger the structural
    gibberish check. Modeled on CarOK p5 shape (~13K chars, 272
    real 4+ letter words)."""
    # Pad to >30K chars (over the length threshold) with realistic
    # table content so word_density stays > 10 w/k.
    rows = []
    for i in range(800):
        rows.append(
            f"| Customer{i:03d} | Order{i:04d} | Product Description {i} | "
            f"Quantity {i} | Total {i*100} EUR | Status Complete |"
        )
    content = "\n".join(rows)
    assert len(content) > 30000
    assert BatchProcessor._is_corrupted_chunk_content(content) is False


@pytest.mark.parametrize(
    "content",
    [
        # Combat p66 — actual chunk content shape (the corruption signal lives a
        # few hundred chars in, not in the first 100 chars). Real fixture: a
        # snippet that hits multiple patterns (em-dashes 6+, CS 10+, trademark 2+).
        "Squadron Note ———————|PuebloMemorialAirport,Colorado———“‘“‘CS*S™S™*™~™~C™C™CSCANWCN §@",
        # 6+ em-dashes alone:
        "Body —————— fragment",
        # Repeated trademark:
        "Foo ™™ bar",
        # 10+ Cs:
        "CCCCCCCCCC fragment text",
        # Garbled ordinal:
        "S5th squadron at base",
        # Replacement-char ratio above 0.5%:
        "abcd�" * 50,  # 50/250 chars are replacement = 20% — well above 0.5%
    ],
)
def test_corrupted_content_detected(content: str) -> None:
    assert BatchProcessor._is_corrupted_chunk_content(content) is True


@pytest.mark.parametrize(
    "content",
    [
        # Normal English prose:
        "The quick brown fox jumps over the lazy dog. " * 10,
        # Code:
        "def foo():\n    return 42\n",
        # Index-style entry:
        "Argilla platform for A/B testing, 157",
        # Markdown table:
        "| col1 | col2 |\n|------|------|\n| a    | b    |",
        # Single em-dash usage (legitimate):
        "The book — published in 1990 — is a classic.",
        # Single trademark (legitimate):
        "Acme™ product line",
        # Empty / whitespace:
        "",
        "   \n  ",
        # German Umlaut + replacement char below threshold:
        "Bücher" + "�" + "und" + "x" * 1000,  # 1/1011 ≈ 0.1%
    ],
)
def test_clean_content_not_flagged(content: str) -> None:
    assert BatchProcessor._is_corrupted_chunk_content(content) is False


# ---------------------------------------------------------------------------
# _drop_corrupted_chunks_before_metadata — full pipeline
# ---------------------------------------------------------------------------


def _text(content: str, page: int = 1) -> object:
    return create_text_chunk(
        doc_id="deadbeef1234",
        content=content,
        source_file="t.pdf",
        file_type=FileType.PDF,
        page_number=page,
        hierarchy=HierarchyMetadata(breadcrumb_path=["Doc"], level=1),
    )


def _table(content: str, page: int = 1) -> object:
    return create_table_chunk(
        doc_id="deadbeef1234",
        content=content,
        source_file="t.pdf",
        file_type=FileType.PDF,
        page_number=page,
        bbox=[0, 0, 1000, 1000],
        page_width=612,
        page_height=792,
        asset_path="assets/table.png",
    )


def test_drops_corrupted_table_keeps_clean_text() -> None:
    bp = _bp()
    clean = _text("Real prose chunk content." * 3, page=10)
    corrupted_table = _table(
        # Real Combat p66 corruption shape: hits em-dashes-6+ pattern.
        "Squadron Note ———————|PuebloMemorialAirport,Colorado———“‘“‘CS*S™S™*™~™~C™C™CSCANWCN §@",
        page=66,
    )
    image = create_image_chunk(
        doc_id="deadbeef1234",
        content="[Figure on page 66]",
        source_file="t.pdf",
        file_type=FileType.PDF,
        page_number=66,
        bbox=[0, 0, 1000, 1000],
        page_width=612,
        page_height=792,
        asset_path="/tmp/img.png",
        width_px=800,
        height_px=600,
    )

    result = bp._drop_corrupted_chunks_before_metadata([clean, corrupted_table, image])

    assert clean in result
    assert image in result, "image chunks must NOT go through the text/table filter"
    assert corrupted_table not in result


def test_drops_corrupted_text_chunk() -> None:
    bp = _bp()
    clean = _text("paragraph one is fine.", page=1)
    corrupted = _text("foo—————— bar", page=2)
    result = bp._drop_corrupted_chunks_before_metadata([clean, corrupted])
    assert result == [clean]


def test_no_drops_on_clean_corpus() -> None:
    bp = _bp()
    chunks = [_text(f"clean paragraph {i}." * 3, page=i) for i in range(1, 11)]
    result = bp._drop_corrupted_chunks_before_metadata(chunks)
    assert result == chunks


def test_corrupted_chunk_with_no_page_number_still_dropped() -> None:
    """Defensive: a chunk with metadata.page_number=None must still drop cleanly."""
    bp = _bp()
    chunk = _text("foo ——————— bar", page=1)
    chunk.metadata.page_number = None  # type: ignore[union-attr]
    result = bp._drop_corrupted_chunks_before_metadata([chunk])
    assert result == []
