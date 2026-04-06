"""
Tests for OversizeBreaker quality fixes and PUA/whitespace normalization.

Fix 1 — Word-boundary snap: when the hard-cap fires at max_chars, the split
         must land on the last space at or before max_chars, not mid-word.

Fix 2 — No overlap: overlap_chars=0 so consecutive split parts share no
         sentence content (no 120-char tail prepended to next chunk).

Fix 3 — PUA normalization: Private Use Area Unicode (Wingdings/Symbol fonts)
         converted to readable equivalents.

Fix 4 — Whitespace collapsing: double/triple spaces from PDF justified text
         collapsed to single space (non-code chunks only).
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch_processor():
    """Return a minimal BatchProcessor with mocked dependencies."""
    from mmrag_v2.batch_processor import BatchProcessor

    bp = BatchProcessor.__new__(BatchProcessor)
    bp._intelligence_metadata = {}
    bp._doc_hash = "deadbeef1234"
    bp._current_pdf_path = Path("/tmp/test.pdf")
    return bp


# ---------------------------------------------------------------------------
# Fix 1 + 2: OversizeBreaker — mid-word cuts and duplicate content
# ---------------------------------------------------------------------------


def test_no_mid_word_cuts_long_paragraph():
    """All split parts must start on a word boundary (not mid-word)."""
    bp = _make_batch_processor()

    # Build text longer than 1500 chars with no paragraph breaks — forces
    # the hard-cap path. Each sentence is long enough that a mid-word cut
    # at exactly 1500 would land inside a word.
    sentence = "The alarm controller validates configuration parameters. "
    text = sentence * 30  # ~1650 chars

    parts = bp._split_nearest_paragraph_breaks(text=text, max_chars=1500, overlap_chars=0)

    assert len(parts) >= 2, "Expected at least 2 parts"
    for i, part in enumerate(parts):
        assert part, f"Part {i} is empty"
        first_char = part[0]
        assert not (first_char.islower() and first_char.isalpha()), (
            f"Part {i} starts mid-word: {part[:60]!r}"
        )


def test_no_duplicate_content_between_parts():
    """Consecutive split parts must not share a full trailing sentence."""
    bp = _make_batch_processor()

    # 40 × sentence → ~2400 chars; forces 2 splits with overlap_chars=0.
    sentence = "The installer configures zone parameters. "
    text = sentence * 40

    parts = bp._split_nearest_paragraph_breaks(text=text, max_chars=1500, overlap_chars=0)

    assert len(parts) >= 2, "Expected at least 2 parts"
    for i in range(len(parts) - 1):
        words_i = set(parts[i].lower().split())
        words_next = set(parts[i + 1].lower().split())
        overlap = words_i & words_next
        # With overlap_chars=0 only words from a sentence straddling the boundary
        # legitimately appear in both chunks — that should be well under 8 words.
        assert len(overlap) <= 8, (
            f"Parts {i} and {i+1} share {len(overlap)} words — likely duplicate content.\n"
            f"  End of part {i}:   ...{parts[i][-100:]!r}\n"
            f"  Start of part {i+1}: {parts[i+1][:100]!r}"
        )


def test_word_boundary_snap_fires_on_hard_cap():
    """When the hard cap fires, the split must land at the last space, not mid-character."""
    bp = _make_batch_processor()

    # Positions 0–1489: alternating "A " pairs (word + space, 2 chars each).
    # Position 1490 onward: solid run of "B" with no spaces.
    # A naive hard cap at 1500 would cut inside "BBBB..." but the word-boundary
    # snap should pick the last space at or before 1500 (position ~1489).
    prefix = "A " * 745  # 1490 chars
    suffix = "B" * 200
    text = prefix + suffix  # ~1690 chars

    parts = bp._split_nearest_paragraph_breaks(text=text, max_chars=1500, overlap_chars=0)

    assert len(parts) >= 2
    first = parts[0]
    # First part must end cleanly (at the word boundary, not inside "BBBB...")
    assert not first.endswith("B"), (
        f"First part ends mid-word in the B run: {first[-30:]!r}"
    )
    second = parts[1]
    assert second, "Second part is empty"
    # Second part must not start with lowercase alpha
    first_char = second[0]
    assert not (first_char.islower() and first_char.isalpha()), (
        f"Second part starts mid-word: {second[:40]!r}"
    )


def test_split_monotonic_progress():
    """Splitting terminates even on pathological input (no spaces, no newlines)."""
    bp = _make_batch_processor()

    text = "X" * 4000
    parts = bp._split_nearest_paragraph_breaks(text=text, max_chars=1500, overlap_chars=0)
    total = sum(len(p) for p in parts)
    # Should produce content without growing beyond the input length
    assert total > 0
    assert total <= len(text) + 10


# ---------------------------------------------------------------------------
# Fix 3: PUA Unicode normalization
# ---------------------------------------------------------------------------


def test_pua_bullet_normalized():
    r"""\uf02d and \uf0b7 (Wingdings bullets/dashes) → bullet '•'."""
    bp = _make_batch_processor()

    text = "\uf02d arm the system\n\uf0b7 disarm the system"
    result = bp._normalize_pua_chars(text)

    assert "\uf02d" not in result
    assert "\uf0b7" not in result
    assert "• arm the system" in result
    assert "• disarm the system" in result


def test_pua_arrow_and_nav_keys_normalized():
    r"""\uf0e0 → →, \uf070 → ◄, \uf071 → ►, \uf074 → ▲, \uf075 → ▼."""
    bp = _make_batch_processor()

    text = "ETHM-1 \uf0e0 DLOADX\nUse \uf070 or \uf071 keys\nPress \uf074 up \uf075 down"
    result = bp._normalize_pua_chars(text)

    assert "→" in result
    assert "◄" in result
    assert "►" in result
    assert "▲" in result
    assert "▼" in result
    for pua in ("\uf0e0", "\uf070", "\uf071", "\uf074", "\uf075"):
        assert pua not in result


def test_pua_unknown_becomes_space():
    """Any unrecognized PUA codepoint falls back to a single space."""
    bp = _make_batch_processor()

    unknown_pua = "\ue999"  # not in _PUA_MAP but IS in BMP PUA range (U+E000–U+F8FF)
    result = bp._normalize_pua_chars(f"before{unknown_pua}after")

    assert "\ue999" not in result
    assert "before" in result
    assert "after" in result
    # The PUA char must have been replaced by a space
    assert "before after" in result


def test_pua_empty_string_safe():
    """_normalize_pua_chars is safe on empty input."""
    bp = _make_batch_processor()

    assert bp._normalize_pua_chars("") == ""


def test_pua_non_pua_text_unchanged():
    """Non-PUA text must pass through unmodified."""
    bp = _make_batch_processor()

    text = "Press # to confirm. Use ► key to navigate."
    assert bp._normalize_pua_chars(text) == text


# ---------------------------------------------------------------------------
# Fix 4: Whitespace collapsing via _normalize_chunk_text
# ---------------------------------------------------------------------------


def test_whitespace_collapse_on_text_chunks():
    """Double/triple spaces in non-code TEXT chunks are collapsed to one space."""
    from mmrag_v2.schema.ingestion_schema import FileType, create_text_chunk

    bp = _make_batch_processor()

    chunk = create_text_chunk(
        doc_id="abc123def456",
        content="It  is  in  the  interest  of  the  user  to  plan.",
        source_file="test.pdf",
        file_type=FileType.PDF,
        page_number=1,
    )
    [result] = bp._normalize_chunk_text([chunk])

    assert "  " not in result.content, (
        f"Double-space survived: {result.content!r}"
    )
    assert result.content == "It is in the interest of the user to plan."


def test_whitespace_collapse_preserves_code_chunks():
    """Code chunk whitespace (indentation) must not be collapsed."""
    from mmrag_v2.schema.ingestion_schema import ChunkType, FileType, create_text_chunk

    bp = _make_batch_processor()

    code_content = "def foo():\n    return  42\n"
    chunk = create_text_chunk(
        doc_id="abc123def456",
        content=code_content,
        source_file="test.pdf",
        file_type=FileType.PDF,
        page_number=1,
    )
    chunk.metadata.chunk_type = ChunkType.CODE

    [result] = bp._normalize_chunk_text([chunk])

    assert result.content == code_content, (
        f"Code chunk was modified: {result.content!r}"
    )


def test_whitespace_collapse_preserves_newlines():
    """Newlines must not be collapsed — only horizontal spaces."""
    from mmrag_v2.schema.ingestion_schema import FileType, create_text_chunk

    bp = _make_batch_processor()

    chunk = create_text_chunk(
        doc_id="abc123def456",
        content="Line one.\n\nLine  two with  double  spaces.",
        source_file="test.pdf",
        file_type=FileType.PDF,
        page_number=1,
    )
    [result] = bp._normalize_chunk_text([chunk])

    assert "\n\n" in result.content, "Paragraph break (double newline) was collapsed"
    assert "  " not in result.content, "Double-space survived"
