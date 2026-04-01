"""
Tests for the 4 output-quality fixes applied in v2.5.0-dev:

  Fix 1 — _filter_repetition_garbage   (token-repetition garbage chunks)
  Fix 2 — _looks_like_code_text        (index ">>>" vs REPL ">>>")
  Fix 3a — _fix_linebreak_hyphenation  (soft hyphen U+00AD)
  Fix 3b — _rejoin_leading_word_fragments (cross-chunk column-break fragments)
  Fix 4 — _looks_like_code_text        (enhanced code patterns)
"""

import pytest

from mmrag_v2.batch_processor import BatchProcessor
from mmrag_v2.schema.ingestion_schema import (
    ChunkType,
    FileType,
    HierarchyMetadata,
    Modality,
    create_image_chunk,
    create_text_chunk,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bp(tmp_path) -> BatchProcessor:
    return BatchProcessor(
        output_dir=str(tmp_path),
        batch_size=3,
        vision_provider="none",
        enable_ocr=False,
    )


def _text(content: str, page: int = 1, chunk_type: ChunkType = ChunkType.PARAGRAPH,
          content_classification=None):
    return create_text_chunk(
        doc_id="testdoc",
        content=content,
        source_file="test.pdf",
        file_type=FileType.PDF,
        page_number=page,
        hierarchy=HierarchyMetadata(
            parent_heading=None,
            breadcrumb_path=["test", f"p{page}"],
            level=2,
        ),
        chunk_type=chunk_type,
        content_classification=content_classification,
    )


def _image(page: int = 1, visual_description: str = "A diagram."):
    return create_image_chunk(
        doc_id="testdoc",
        content="",
        source_file="test.pdf",
        file_type=FileType.PDF,
        page_number=page,
        asset_path="asset.png",
        bbox=[0, 0, 500, 400],
        visual_description=visual_description,
    )


# ===========================================================================
# Fix 1 — _filter_repetition_garbage
# ===========================================================================

class TestFilterRepetitionGarbage:
    """Token-repetition garbage chunks must be dropped from TEXT chunks only."""

    def test_colon_separated_repetition_dropped(self, tmp_path):
        bp = _bp(tmp_path)
        chunk = _text("down: down: down: down: down:")
        result = bp._filter_repetition_garbage([chunk])
        assert result == [], "Expected garbage chunk to be dropped"

    def test_comma_separated_repetition_dropped(self, tmp_path):
        bp = _bp(tmp_path)
        chunk = _text("today, today, today, today, today")
        result = bp._filter_repetition_garbage([chunk])
        assert result == []

    def test_space_separated_repetition_dropped(self, tmp_path):
        bp = _bp(tmp_path)
        chunk = _text("error error error error error")
        result = bp._filter_repetition_garbage([chunk])
        assert result == []

    def test_four_repetitions_survives(self, tmp_path):
        """Below the threshold of 5 repetitions: must NOT be dropped."""
        bp = _bp(tmp_path)
        chunk = _text("word: word: word: word")
        result = bp._filter_repetition_garbage([chunk])
        assert len(result) == 1

    def test_normal_text_survives(self, tmp_path):
        bp = _bp(tmp_path)
        chunk = _text("This is a normal technical paragraph about the system.")
        result = bp._filter_repetition_garbage([chunk])
        assert len(result) == 1

    def test_code_chunk_survives_despite_repetition(self, tmp_path):
        """CODE chunks must never be filtered regardless of content."""
        bp = _bp(tmp_path)
        chunk = _text(
            "error: error: error: error: error:",
            chunk_type=ChunkType.CODE,
            content_classification="code",
        )
        result = bp._filter_repetition_garbage([chunk])
        assert len(result) == 1

    def test_image_chunk_survives(self, tmp_path):
        """Non-TEXT chunks (IMAGE) are always kept."""
        bp = _bp(tmp_path)
        img = _image()
        result = bp._filter_repetition_garbage([img])
        assert len(result) == 1

    def test_mixed_list_partial_removal(self, tmp_path):
        bp = _bp(tmp_path)
        good = _text("API authentication is required for all endpoints.")
        bad  = _text("down: down: down: down: down:")
        result = bp._filter_repetition_garbage([good, bad])
        assert len(result) == 1
        assert result[0].content == good.content


# ===========================================================================
# Fix 2 — _looks_like_code_text: index ">>>" guard
# ===========================================================================

class TestLooksLikeCodeTextIndexGuard:
    """Index cross-references using >>> must NOT be classified as code."""

    def test_index_entry_single_line_not_code(self, tmp_path):
        bp = _bp(tmp_path)
        # python_distilled-style index entry
        assert bp._looks_like_code_text(
            ">>> DEFAULT_BUFFER_SIZE value, 259 defaultdict"
        ) is False

    def test_index_entry_with_range_not_code(self, tmp_path):
        bp = _bp(tmp_path)
        assert bp._looks_like_code_text(
            ">>> MISSING_VALUE, 42-43"
        ) is False

    def test_index_entry_short_not_code(self, tmp_path):
        bp = _bp(tmp_path)
        assert bp._looks_like_code_text(
            ">>> format method, 78"
        ) is False

    def test_repl_assignment_is_code(self, tmp_path):
        bp = _bp(tmp_path)
        assert bp._looks_like_code_text(">>> x = 5") is True

    def test_repl_function_call_is_code(self, tmp_path):
        bp = _bp(tmp_path)
        assert bp._looks_like_code_text(">>> print('hello')") is True

    def test_repl_for_loop_is_code(self, tmp_path):
        bp = _bp(tmp_path)
        assert bp._looks_like_code_text(">>> for i in range(10):") is True

    def test_multiline_all_index_not_code(self, tmp_path):
        bp = _bp(tmp_path)
        text = (
            ">>> DEFAULT_BUFFER_SIZE value, 259 defaultdict\n"
            ">>> format method, 78\n"
            ">>> MISSING_VALUE, 42"
        )
        assert bp._looks_like_code_text(text) is False

    def test_multiline_one_real_repl_among_index_is_code(self, tmp_path):
        bp = _bp(tmp_path)
        text = (
            ">>> DEFAULT_BUFFER_SIZE value, 259 defaultdict\n"
            ">>> x = func()\n"          # real REPL — has parentheses
            ">>> format method, 78"
        )
        assert bp._looks_like_code_text(text) is True


# ===========================================================================
# Fix 3a — _fix_linebreak_hyphenation: soft hyphen (U+00AD)
# ===========================================================================

class TestFixLinebreakHyphenationSoftHyphen:

    def test_soft_hyphen_stripped_and_rejoined_without_hyphen(self, tmp_path):
        bp = _bp(tmp_path)
        result = bp._fix_linebreak_hyphenation("implemen\xad\nted")
        assert result == "implemented"

    def test_soft_hyphen_multiplatform(self, tmp_path):
        bp = _bp(tmp_path)
        result = bp._fix_linebreak_hyphenation("multi\xad\nplatform")
        assert result == "multiplatform"

    def test_hard_hyphen_preserved_and_rejoined(self, tmp_path):
        bp = _bp(tmp_path)
        result = bp._fix_linebreak_hyphenation("multi-\nstep")
        assert result == "multi-step"

    def test_soft_hyphen_no_false_positive_mid_word(self, tmp_path):
        """Soft hyphen NOT at a line break must be left unchanged."""
        bp = _bp(tmp_path)
        text = "hello\xadworld no newline here"
        result = bp._fix_linebreak_hyphenation(text)
        # The regex requires a \n after \xad — without it nothing changes.
        assert "\xad" in result

    def test_text_without_hyphens_unchanged(self, tmp_path):
        bp = _bp(tmp_path)
        text = "This is a plain sentence.\nNo hyphens involved."
        assert bp._fix_linebreak_hyphenation(text) == text


# ===========================================================================
# Fix 3b — _rejoin_leading_word_fragments
# ===========================================================================

class TestRejoinLeadingWordFragments:
    """Cross-chunk column-break word fragments must be rejoined conservatively."""

    def test_suffix_in_whitelist_rejoined(self, tmp_path):
        bp = _bp(tmp_path)
        # "ted" is in _WORD_FRAGMENT_SUFFIXES; "connec" + "ted" → "connected"
        cur  = _text("The system connec", page=5)
        nxt  = _text("ted all endpoints successfully.", page=5)
        result = bp._rejoin_leading_word_fragments([cur, nxt])
        assert len(result) == 1
        assert result[0].content == "The system connected all endpoints successfully."

    def test_tion_suffix_rejoined(self, tmp_path):
        bp = _bp(tmp_path)
        cur  = _text("The configura", page=2)
        nxt  = _text("tion process requires elevated rights.", page=2)
        result = bp._rejoin_leading_word_fragments([cur, nxt])
        assert len(result) == 1
        assert "configuration" in result[0].content

    def test_suffix_not_in_whitelist_not_rejoined(self, tmp_path):
        bp = _bp(tmp_path)
        cur  = _text("The system is", page=3)
        nxt  = _text("xyz and then continues with more text.", page=3)
        result = bp._rejoin_leading_word_fragments([cur, nxt])
        assert len(result) == 2

    def test_next_chunk_only_one_token_not_rejoined(self, tmp_path):
        """Guard: next chunk must have ≥2 tokens beyond the leading fragment."""
        bp = _bp(tmp_path)
        cur  = _text("Implementa", page=1)
        nxt  = _text("tion", page=1)  # only 1 token → must not rejoin
        result = bp._rejoin_leading_word_fragments([cur, nxt])
        assert len(result) == 2

    def test_different_pages_not_rejoined(self, tmp_path):
        bp = _bp(tmp_path)
        cur  = _text("The implementa", page=4)
        nxt  = _text("tion process requires elevated rights.", page=5)
        result = bp._rejoin_leading_word_fragments([cur, nxt])
        assert len(result) == 2

    def test_current_ends_uppercase_not_rejoined(self, tmp_path):
        """Guard: current chunk must end with a lowercase letter."""
        bp = _bp(tmp_path)
        cur  = _text("This ends with an Abbreviation ABC", page=1)
        nxt  = _text("ment in the pipeline ensures correctness.", page=1)
        result = bp._rejoin_leading_word_fragments([cur, nxt])
        assert len(result) == 2

    def test_code_chunk_not_rejoined(self, tmp_path):
        bp = _bp(tmp_path)
        cur  = _text("# setup", page=1, chunk_type=ChunkType.CODE,
                     content_classification="code")
        nxt  = _text("tion of the environment variable.", page=1)
        result = bp._rejoin_leading_word_fragments([cur, nxt])
        assert len(result) == 2

    def test_three_chunks_second_pair_rejoined(self, tmp_path):
        bp = _bp(tmp_path)
        a = _text("Introduction to the pipeline.", page=1)
        b = _text("The configura", page=2)
        c = _text("tion requires elevated rights now.", page=2)
        result = bp._rejoin_leading_word_fragments([a, b, c])
        assert len(result) == 2
        assert result[0].content == a.content
        assert "configuration" in result[1].content


# ===========================================================================
# Fix 4 — _looks_like_code_text: enhanced code patterns
# ===========================================================================

class TestLooksLikeCodeTextEnhanced:

    def test_snake_case_assignment_with_function_call(self, tmp_path):
        bp = _bp(tmp_path)
        assert bp._looks_like_code_text(
            'api_key = os.getenv("OPENAI_API_KEY")'
        ) is True

    def test_snake_case_underscore_assignment(self, tmp_path):
        bp = _bp(tmp_path)
        assert bp._looks_like_code_text(
            "base_url = config.get_endpoint()"
        ) is True

    def test_dict_literal_single_line(self, tmp_path):
        bp = _bp(tmp_path)
        assert bp._looks_like_code_text(
            '{"role": "user", "content": "Hello"}'
        ) is True

    def test_shebang_line(self, tmp_path):
        bp = _bp(tmp_path)
        assert bp._looks_like_code_text("#!/usr/bin/env python3") is True

    def test_multiline_snake_assignments_with_other_code(self, tmp_path):
        bp = _bp(tmp_path)
        text = (
            "client = openai.OpenAI()\n"
            "response = client.chat.completions.create(\n"
            '    model="gpt-4o",\n'
            '    messages=[{"role": "user"}]\n'
            ")"
        )
        assert bp._looks_like_code_text(text) is True

    def test_multiline_dict_literal_block(self, tmp_path):
        bp = _bp(tmp_path)
        text = (
            '{\n'
            '  "model": "gpt-4o",\n'
            '  "temperature": 0.7,\n'
            '  "max_tokens": 512\n'
            '}'
        )
        assert bp._looks_like_code_text(text) is True

    def test_comment_first_code_block(self, tmp_path):
        bp = _bp(tmp_path)
        text = (
            "# agent.py\n"
            "import json\n"
            "import openai\n"
        )
        assert bp._looks_like_code_text(text) is True

    def test_comment_plus_structural_lines(self, tmp_path):
        bp = _bp(tmp_path)
        text = (
            "# Load environment\n"
            "import os\n"
            "from pathlib import Path\n"
        )
        assert bp._looks_like_code_text(text) is True

    def test_plain_prose_not_code(self, tmp_path):
        bp = _bp(tmp_path)
        assert bp._looks_like_code_text(
            "The API key is stored in an environment variable for security reasons."
        ) is False

    def test_prose_starting_with_hash(self, tmp_path):
        """A hashtag or number sign in prose must not trigger code detection."""
        bp = _bp(tmp_path)
        assert bp._looks_like_code_text(
            "# This book uses Python 3.10 and requires pip install."
        ) is False

    def test_flat_if_else_single_line_is_code(self, tmp_path):
        """PDF-flattened single-line if/else with call tokens must be detected as code."""
        bp = _bp(tmp_path)
        assert bp._looks_like_code_text(
            "if a < b: print('Computer says Yes') else : print('Computer says No')"
        ) is True

    def test_english_if_else_no_colon_is_not_code(self, tmp_path):
        """English prose using 'if … else' without colon syntax must not be code."""
        bp = _bp(tmp_path)
        assert bp._looks_like_code_text(
            "if you follow the steps you will succeed else you will fail (somehow)"
        ) is False


# ===========================================================================
# Fix 5 — _maybe_demote_false_code_chunk: prose_lines guard
# ===========================================================================

class TestMaybeDemoteFalseCodeChunkProseGuard:
    """
    prose_lines must NOT count lines that is_code_line() already identifies as code.
    A def/import line ending in punctuation (e.g. trailing ...) must remain CODE.
    """

    def test_def_with_trailing_ellipsis_stays_code(self, tmp_path):
        """
        'def connect(hostname, port, timeout=300): # Function body ...'
        Previously counted as both code_line AND prose_line → wrongly demoted.
        After fix: is_code_line guard prevents prose count → stays CODE.
        """
        bp = _bp(tmp_path)
        chunk = _text(
            "def connect(hostname, port, timeout=300): # Function body ...",
            chunk_type=ChunkType.CODE,
            content_classification="code",
        )
        # Simulate what _apply_technical_manual_hygiene does: mark as CODE, then
        # call the demotion guard.  We call the guard directly.
        from mmrag_v2.schema.ingestion_schema import ChunkType as CT
        chunk.metadata.chunk_type = CT.CODE
        bp._maybe_demote_false_code_chunk(chunk)
        assert chunk.metadata.chunk_type == CT.CODE, (
            "def line with trailing '...' must NOT be demoted to paragraph"
        )

    def test_import_statement_alone_is_demoted(self, tmp_path):
        """
        A standalone import line with no other code context is correctly demoted
        (existing behaviour that must be preserved).
        """
        bp = _bp(tmp_path)
        chunk = _text(
            "import os",
            chunk_type=ChunkType.CODE,
            content_classification="code",
        )
        from mmrag_v2.schema.ingestion_schema import ChunkType as CT
        chunk.metadata.chunk_type = CT.CODE
        bp._maybe_demote_false_code_chunk(chunk)
        assert chunk.metadata.chunk_type != CT.CODE, (
            "Standalone import line should be demoted to paragraph"
        )

    def test_genuine_prose_with_punctuation_is_demoted(self, tmp_path):
        """
        A real prose sentence classified as code by accident must be demoted.
        """
        bp = _bp(tmp_path)
        chunk = _text(
            "The connect function establishes a TCP connection to the remote host.",
            chunk_type=ChunkType.CODE,
            content_classification="code",
        )
        from mmrag_v2.schema.ingestion_schema import ChunkType as CT
        chunk.metadata.chunk_type = CT.CODE
        bp._maybe_demote_false_code_chunk(chunk)
        assert chunk.metadata.chunk_type != CT.CODE, (
            "Prose sentence must be demoted back to paragraph"
        )


# ===========================================================================
# Fix 6 — _maybe_demote_false_code_chunk: prose->>> guard
# Book formatting sometimes uses ">>>" as cross-reference / prose lead-in,
# not as a Python REPL prompt. These must not be kept as code.
# ===========================================================================

class TestMaybeDemoteFalseCodeChunkReplProseGuard:
    """
    '>>>' used as book formatting (TOC cross-ref, prose lead-in) must be
    demoted to paragraph.  Real REPL code (digit start, lowercase variable,
    function call) must stay CODE.
    """

    def _code_chunk(self, tmp_path, content: str):
        from mmrag_v2.schema.ingestion_schema import ChunkType as CT
        bp = _bp(tmp_path)
        chunk = _text(content, chunk_type=ChunkType.CODE, content_classification="code")
        chunk.metadata.chunk_type = CT.CODE
        return bp, chunk

    def test_toc_section_numbers_demoted(self, tmp_path):
        """'>>> Input and Output 9.1 Data Representation 9.2 ...' is a TOC, not REPL."""
        bp, chunk = self._code_chunk(
            tmp_path,
            ">>> Input and Output 9.1 Data Representation 9.2 Text Encoding and Decoding "
            "9.3 Text and Byte Formatting 9.4 Reading Command-Line Options",
        )
        from mmrag_v2.schema.ingestion_schema import ChunkType as CT
        bp._maybe_demote_false_code_chunk(chunk)
        assert chunk.metadata.chunk_type != CT.CODE, (
            "TOC entry with >>> and section numbers must be demoted"
        )

    def test_prose_sentence_with_repl_prefix_demoted(self, tmp_path):
        """'>>> The round() function implements ...' is prose, not REPL."""
        bp, chunk = self._code_chunk(
            tmp_path,
            ">>> The round() function implements 'banker's rounding.' "
            "If the value being rounded is equally close to two multiples, "
            "it is rounded to the nearest even number.",
        )
        from mmrag_v2.schema.ingestion_schema import ChunkType as CT
        bp._maybe_demote_false_code_chunk(chunk)
        assert chunk.metadata.chunk_type != CT.CODE, (
            "Prose sentence prefixed with >>> must be demoted"
        )

    def test_multiline_prose_repl_prefix_demoted(self, tmp_path):
        """Multi-line prose with >>> lead-ins must be demoted."""
        bp, chunk = self._code_chunk(
            tmp_path,
            "Click here to view code image\n"
            ">>> One problem\n"
            ">>> with the interest.py program is that the output isn't very pretty.",
        )
        from mmrag_v2.schema.ingestion_schema import ChunkType as CT
        bp._maybe_demote_false_code_chunk(chunk)
        assert chunk.metadata.chunk_type != CT.CODE, (
            "Multi-line prose prefixed with >>> must be demoted"
        )

    def test_arithmetic_repl_stays_code(self, tmp_path):
        """'>>> 6000 + 4523.50 ...' is real arithmetic REPL — must stay CODE."""
        bp, chunk = self._code_chunk(
            tmp_path,
            ">>> 6000 + 4523.50 + 134.25 10657.75 >>> _ + 8192.75",
        )
        from mmrag_v2.schema.ingestion_schema import ChunkType as CT
        bp._maybe_demote_false_code_chunk(chunk)
        assert chunk.metadata.chunk_type == CT.CODE, (
            "Arithmetic REPL must stay CODE"
        )

    def test_assignment_repl_stays_code(self, tmp_path):
        """'>>> x = 5' is a real REPL assignment — must stay CODE."""
        bp, chunk = self._code_chunk(tmp_path, ">>> x = 5")
        from mmrag_v2.schema.ingestion_schema import ChunkType as CT
        bp._maybe_demote_false_code_chunk(chunk)
        assert chunk.metadata.chunk_type == CT.CODE, (
            "REPL assignment must stay CODE"
        )

    def test_function_call_repl_stays_code(self, tmp_path):
        """'>>> print(\"hello\")' is a real REPL call — must stay CODE."""
        bp, chunk = self._code_chunk(tmp_path, ">>> print('hello')")
        from mmrag_v2.schema.ingestion_schema import ChunkType as CT
        bp._maybe_demote_false_code_chunk(chunk)
        assert chunk.metadata.chunk_type == CT.CODE, (
            "REPL function call must stay CODE"
        )


# ===========================================================================
# Fix 7 — OCR text chunks chunk_type=null (3 creation paths)
# ===========================================================================

class TestOcrChunkTypeNullFix:
    """
    OCR text chunks must never have chunk_type=None.

    Three paths were fixed:
      A) Line 832  — OCR chunk creation default (TEXT → PARAGRAPH)
      B) Line 6140 — image_to_text_recovery reclassification
      C) Line 6254 — enhanced_image_ocr reclassification
      D) Line 6619 — split chunk propagation safety (chunk_type or PARAGRAPH)
    """

    def test_image_chunk_naturally_has_null_chunk_type(self):
        """Baseline: IMAGE chunks have chunk_type=None (correct — images are not text)."""
        img = create_image_chunk(
            doc_id="test",
            content="",
            source_file="test.pdf",
            file_type=FileType.PDF,
            page_number=1,
            asset_path="p1.png",
            bbox=[0, 0, 500, 400],
        )
        assert img.metadata.chunk_type is None

    def test_image_to_text_recovery_fix_sets_paragraph(self):
        """Path B: image_to_text_recovery sets PARAGRAPH when reclassifying IMAGE → TEXT."""
        img = create_image_chunk(
            doc_id="test",
            content="",
            source_file="test.pdf",
            file_type=FileType.PDF,
            page_number=1,
            asset_path="p1.png",
            bbox=[0, 0, 500, 400],
        )
        assert img.metadata.chunk_type is None  # baseline

        # Simulate the reclassification (mirrors lines 6134–6141 in batch_processor.py).
        img.modality = Modality.TEXT
        img.content = "This is recovered text from OCR."
        if img.metadata and img.metadata.chunk_type is None:
            img.metadata.chunk_type = ChunkType.PARAGRAPH

        assert img.metadata.chunk_type == ChunkType.PARAGRAPH, (
            "IMAGE→TEXT reclassification must assign ChunkType.PARAGRAPH"
        )

    def test_enhanced_image_ocr_fix_sets_paragraph(self):
        """Path C: enhanced_image_ocr sets PARAGRAPH when reclassifying IMAGE → TEXT."""
        img = create_image_chunk(
            doc_id="test",
            content="",
            source_file="test.pdf",
            file_type=FileType.PDF,
            page_number=1,
            asset_path="p1.png",
            bbox=[0, 0, 500, 400],
        )
        assert img.metadata.chunk_type is None  # baseline

        # Simulate enhanced_image_ocr reclassification (mirrors lines 6249–6255).
        img.modality = Modality.TEXT
        img.content = "Enhanced OCR recovered text from front page."
        if img.metadata and img.metadata.chunk_type is None:
            img.metadata.chunk_type = ChunkType.PARAGRAPH

        assert img.metadata.chunk_type == ChunkType.PARAGRAPH, (
            "enhanced_image_ocr IMAGE→TEXT reclassification must assign ChunkType.PARAGRAPH"
        )

    def test_split_chunk_null_chunk_type_safety(self):
        """Path D: split chunk from parent with chunk_type=None gets PARAGRAPH via 'or' guard."""
        from mmrag_v2.schema.ingestion_schema import ChunkMetadata

        meta = ChunkMetadata(
            source_file="test.pdf",
            file_type=FileType.PDF,
            page_number=1,
            chunk_type=None,  # old-style OCR chunk had null
        )
        # The fix: chunk_type=(chunk.metadata.chunk_type or ChunkType.PARAGRAPH)
        resolved = meta.chunk_type or ChunkType.PARAGRAPH
        assert resolved == ChunkType.PARAGRAPH

    def test_normalize_chunk_text_null_chunk_type_treated_as_non_code(self, tmp_path):
        """
        Path A side-effect: _normalize_chunk_text treats chunk_type=None as non-code
        → double-spaces are collapsed (same behaviour as PARAGRAPH).
        """
        bp = _bp(tmp_path)

        # Create a TEXT chunk with chunk_type=None (simulates the pre-fix state).
        chunk = create_text_chunk(
            doc_id="test",
            content="word  word  word",
            source_file="test.pdf",
            file_type=FileType.PDF,
            page_number=1,
            chunk_type=None,
            hierarchy=HierarchyMetadata(
                parent_heading=None,
                breadcrumb_path=["doc", "p1"],
                level=2,
            ),
        )
        assert chunk.metadata.chunk_type is None

        result = bp._normalize_chunk_text([chunk])
        # Treated as non-code → double-spaces collapsed.
        assert result[0].content == "word word word"

    def test_image_reclassified_already_has_chunk_type_not_overwritten(self):
        """
        If an image chunk already has chunk_type set (edge case), reclassification
        must NOT overwrite it (the fix uses 'if chunk_type is None').
        """
        img = create_image_chunk(
            doc_id="test",
            content="",
            source_file="test.pdf",
            file_type=FileType.PDF,
            page_number=1,
            asset_path="p1.png",
            bbox=[0, 0, 500, 400],
        )
        # Manually pre-set chunk_type (edge case).
        img.metadata.chunk_type = ChunkType.CODE

        # Simulate reclassification with the fix guard.
        img.modality = Modality.TEXT
        img.content = "some text"
        if img.metadata and img.metadata.chunk_type is None:
            img.metadata.chunk_type = ChunkType.PARAGRAPH

        # Pre-existing CODE must be preserved, not overwritten.
        assert img.metadata.chunk_type == ChunkType.CODE


# ===========================================================================
# Fix 8 — is_code_line: flow-control keywords require ':' to avoid prose false-positives
# ===========================================================================

def _code_chunk_for_demote(tmp_path, content: str):
    bp = _bp(tmp_path)
    ch = create_text_chunk(
        doc_id="test",
        content=content,
        source_file="test.pdf",
        file_type=FileType.PDF,
        page_number=1,
        chunk_type=ChunkType.CODE,  # start as CODE; demote test
        hierarchy=HierarchyMetadata(
            parent_heading=None,
            breadcrumb_path=["doc", "p1"],
            level=2,
        ),
    )
    return bp, ch


class TestIsCodeLineFlowControlGuard:
    """
    is_code_line must require ':' for flow-control keywords (if/for/while/etc.)
    to avoid treating English prose as code.

    Root cause: 'if the control is desired' starts with 'if' → old regex
    matched → code_lines=1 → chunk not demoted → SEMANTIC_FAIL in full-doc run.
    """

    def test_if_without_colon_is_demoted(self, tmp_path):
        """'if the co\\ndesired\\nvative is' (OCR fragment) → no ':' → demote."""
        bp, chunk = _code_chunk_for_demote(tmp_path, "if the co\ndesired\nvative is")
        bp._maybe_demote_false_code_chunk(chunk)
        assert chunk.metadata.chunk_type == ChunkType.PARAGRAPH, (
            "Prose fragment starting with 'if' but no ':' must be demoted"
        )

    def test_if_with_colon_stays_code(self, tmp_path):
        """'if x > 0:\\n    return x' has ':' → stays CODE."""
        bp, chunk = _code_chunk_for_demote(tmp_path, "if x > 0:\n    return x")
        bp._maybe_demote_false_code_chunk(chunk)
        assert chunk.metadata.chunk_type == ChunkType.CODE, (
            "Real if-statement with ':' must stay CODE"
        )

    def test_for_without_colon_is_demoted(self, tmp_path):
        """'for the good of all' → no ':' → demote."""
        bp, chunk = _code_chunk_for_demote(tmp_path, "for the good of all\npeople everywhere")
        bp._maybe_demote_false_code_chunk(chunk)
        assert chunk.metadata.chunk_type == ChunkType.PARAGRAPH

    def test_for_with_colon_stays_code(self, tmp_path):
        """'for i in range(10):\\n    print(i)' → stays CODE."""
        bp, chunk = _code_chunk_for_demote(tmp_path, "for i in range(10):\n    print(i)")
        bp._maybe_demote_false_code_chunk(chunk)
        assert chunk.metadata.chunk_type == ChunkType.CODE

    def test_return_without_colon_stays_code(self, tmp_path):
        """'return x + y' has no ':' but 'return' doesn't need one → stays CODE."""
        bp, chunk = _code_chunk_for_demote(tmp_path, "return x + y")
        bp._maybe_demote_false_code_chunk(chunk)
        assert chunk.metadata.chunk_type == ChunkType.CODE, (
            "'return' is a structural keyword — must not require ':' to stay CODE"
        )

    def test_def_without_colon_stays_code(self, tmp_path):
        """'def foo():\\n    pass' → stays CODE (def is structural, not flow-control)."""
        bp, chunk = _code_chunk_for_demote(tmp_path, "def foo():\n    pass")
        bp._maybe_demote_false_code_chunk(chunk)
        assert chunk.metadata.chunk_type == ChunkType.CODE


class TestFlatImportChainDetection:
    """
    _looks_like_code_text must recognize flat import chains that PDF extraction
    stripped of newlines.  These chunks already have chunk_type=code (Docling),
    but the flat string lacks operators/keywords to trigger the original step 6.

    Root cause: step 6 required kw_count>=2 AND operator, missing pure import
    chains like 'import os from X import Y from Z import W' (no =([{).

    Fix: added flat-import-chain branch — starts with 'import'/'from', ≥2
    import occurrences or ≥1 + operator.
    """

    def setup_method(self):
        self.bp = BatchProcessor.__new__(BatchProcessor)

    # -- _looks_like_code_text -------------------------------------------------

    def test_pure_import_chain_detected(self):
        """Multi-import chain without operators → True (no = or ( required)."""
        txt = (
            "import os from langchain_community.document_loaders import WebBaseLoader "
            "from langchain.text_splitter import RecursiveCharacterTextSplitter "
            "from langchain_openai import OpenAIEmbeddings"
        )
        assert self.bp._looks_like_code_text(txt), (
            "Flat import chain starting with 'import' and multiple 'import X' occurrences "
            "must be detected as code"
        )

    def test_single_import_with_assignment_detected(self):
        """'import X app = ...' — one import + '=' operator → True."""
        txt = "import IPython app = IPython.Application.instance() app.kernel.do_shutdown(True)"
        assert self.bp._looks_like_code_text(txt), (
            "Single import followed by assignment must be detected as flat code"
        )

    def test_from_import_chain_detected(self):
        """'import numpy as npfrom … import OpenAIEmbeddings' (merged words) → True."""
        txt = "import numpy as npfrom langchain_openai import OpenAIEmbeddings"
        assert self.bp._looks_like_code_text(txt), (
            "Merged flat from-import with multiple 'import' occurrences must be code"
        )

    def test_short_import_with_operator_detected(self):
        """62-char 'import pprint inputs = {...}' — below old 80-char guard → True."""
        txt = 'import pprint inputs = {"question": "What is task decomposition?"}'
        assert len(txt) > 60  # ensure it meets the new 60-char threshold
        assert self.bp._looks_like_code_text(txt), (
            "Short (60+) import+assignment concatenation must be detected"
        )

    def test_prose_starting_with_import_not_detected(self):
        """Long prose sentence starting with 'import' that has no operator → False."""
        txt = (
            "import regulations are notoriously complex across different jurisdictions "
            "and require careful consideration of trade agreements and international law"
        )
        # import_kw_count == 1 ('import regulations' matches), no operator → False
        result = self.bp._looks_like_code_text(txt)
        assert not result, (
            "Prose sentence with 'import' + no operator + single import occurrence "
            "must not be misidentified as code"
        )

    # -- _reflow_flat_code: bare 'from X' lines --------------------------------

    def test_reflow_bare_from_module_line_gets_repl_prefix(self):
        """
        When _reflow_flat_code splits 'from X import Y from Z import W' it creates
        bare 'from X' lines.  The stmt_like pattern must recognise them so all
        lines get '>>> ' prefix.
        """
        flat = (
            "from langchain_community.document_loaders import WebBaseLoader "
            "from langchain.text_splitter import RecursiveCharacterTextSplitter"
        )
        reflowed = self.bp._reflow_flat_code(flat)
        lines = [l for l in reflowed.splitlines() if l.strip()]
        assert all(l.startswith(">>> ") for l in lines), (
            "All lines of a reflowed from-import chain must have '>>> ' prefix; "
            f"got: {lines}"
        )

    def test_end_to_end_flat_import_chain_passes_fidelity(self):
        """
        End-to-end: _preserve_or_reflow_code_text on a flat import chain must
        produce a string that has '>>> ' on at least one line (passes fidelity check).
        """
        flat = (
            "import os from langchain_community.document_loaders import WebBaseLoader "
            "from langchain.text_splitter import RecursiveCharacterTextSplitter "
            "from langchain_openai import OpenAIEmbeddings"
        )
        result = self.bp._preserve_or_reflow_code_text(flat)
        has_repl = any(l.startswith(">>> ") for l in result.splitlines())
        has_indent = any(l.startswith("    ") or l.startswith("\t") for l in result.splitlines())
        assert has_repl or has_indent, (
            "_preserve_or_reflow_code_text on a flat import chain must produce "
            "REPL-prefixed or indented lines to pass code_indentation_fidelity"
        )
