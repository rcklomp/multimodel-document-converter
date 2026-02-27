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
