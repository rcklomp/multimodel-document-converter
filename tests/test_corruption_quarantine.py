"""
Corruption Detection & Quarantine Tests
=========================================
Unit tests for the corruption pattern detector. Bridge tests for the
quarantine are in test_finalization_bridge.py (call actual BatchProcessor
methods, no duplicated logic).

Regression target: Combat Aircraft (22 encoding artifacts, 79 high
corruption chunks after patch_corrupted_chunks() only patched 25/47).
"""

from __future__ import annotations

import pytest

from mmrag_v2.validators.corruption_interceptor import (
    CORRUPTION_PATTERNS,
    count_encoding_artifacts,
    has_encoding_artifacts,
)


# ---------------------------------------------------------------------------
# Unit tests: corruption detection
# ---------------------------------------------------------------------------


class TestCorruptionDetection:
    """Core detection patterns must catch all known artifact types."""

    def test_cid_font_placeholder(self):
        assert has_encoding_artifacts("The /C211 jumped /C1 over the fence")

    def test_unicode_escape_leak(self):
        assert has_encoding_artifacts("The /uniFB01rst test")

    def test_hex_escape_leak(self):
        assert has_encoding_artifacts("Bad text \\xc3\\xa9 here")

    def test_replacement_char(self):
        assert has_encoding_artifacts("Broken \ufffd text")

    def test_clean_text_no_artifacts(self):
        assert not has_encoding_artifacts("This is perfectly clean text.")

    def test_clean_code_no_false_positive(self):
        """Code with /path/to/file must not trigger false positive."""
        assert not has_encoding_artifacts("Run /usr/bin/python to start")

    def test_count_multiple_artifacts(self):
        text = "The /C211 and /C1 with /uniFB01 and \ufffd"
        count = count_encoding_artifacts(text)
        assert count >= 3


# ---------------------------------------------------------------------------
# Negative tests: corrupted content must not export unchanged
# ---------------------------------------------------------------------------


class TestCorruptionNeverExported:
    """Chunks containing encoding artifacts cannot be exported as-is."""

    CORRUPTED_TEXTS = [
        "The /C211 jumped over the /C23 lazy dog",
        "This has /uniFB01 unicode escapes /uniFB02",
        "Replacement chars \ufffd here \ufffd and there",
        "Mixed /C1 and /uniFFFD corruption",
    ]

    @pytest.mark.parametrize("text", CORRUPTED_TEXTS)
    def test_corrupted_chunk_detected(self, text):
        """Each corruption pattern must be detectable."""
        assert has_encoding_artifacts(text), (
            f"Corruption not detected in: {text!r}"
        )
