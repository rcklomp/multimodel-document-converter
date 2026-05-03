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


# ---------------------------------------------------------------------------
# PLAN_V2.8 §3 — Combat Aircraft ornament-glyph contract
# Phase 3 step 1 verified empirically that a fresh Combat full conversion
# under current code (CorruptionInterceptor + quarantine + post-Docling
# sanity stages, all shipped 2026-05-03) leaves zero � chunks. These
# tests pin the input contract: the page-66 ornament-glyph pattern is
# detectable, so the interceptor + quarantine pipeline can act on it; and
# legitimate non-ornament Unicode passes through unchanged.
# ---------------------------------------------------------------------------


class TestPlanV28OrnamentGlyphContract:
    """PLAN_V2.8 §3: Combat Aircraft ornament-glyph corruption is recognized."""

    # Real page-66 squadron-table noise pattern lifted from
    # output/Combat_Aircraft_full_promptfix_v2/ingestion.jsonl (the BEFORE
    # row in QUALITY_SNAPSHOT_2026-05-03.md).
    _ORNAMENT_PATTERN = (
        "Wing/Group Squadron Location = NineteenthAir Force\n"
        "�[il : ltJ! nfr! Ill r!·!�l�l:.[lr!Jl 1 ]1 r!' Dl= r!'\n"
        "Aircraft = F-35C, TailCode = NE-200"
    )

    def test_combat_page_66_ornament_glyphs_are_detected(self):
        """Pattern triggers `has_encoding_artifacts` so the pipeline can act on it."""
        assert has_encoding_artifacts(self._ORNAMENT_PATTERN)

    def test_squadron_data_preserved_under_strip(self):
        """The structural data field stays intact when the pipeline strips ornament noise.

        The post-Docling pipeline either re-OCRs the chunk (replacing the whole
        text) or quarantines it. This test pins the no-mid-strip contract:
        substring-stripping the U+FFFD characters alone leaves the squadron data
        readable. Documents that the data IS present in the source PDF; what
        breaks is only the ornament rendering.
        """
        stripped = self._ORNAMENT_PATTERN.replace("�", "")
        assert "Wing/Group Squadron Location = NineteenthAir Force" in stripped
        assert "Aircraft = F-35C, TailCode = NE-200" in stripped

    def test_no_false_positive_on_arabic_or_cjk(self):
        """Legitimate non-ASCII text without U+FFFD must NOT be flagged.

        Magazine layouts and CJK-localized PDFs use exotic Unicode that the
        ornament-glyph detector must not mistake for corruption. The detector
        keys off `\\ufffd`/`/C\\d+`/`/uni`/`\\xHH` patterns, none of which are
        present in legitimate Arabic, CJK, or em-dash text.
        """
        clean_samples = [
            "العربية: نص نظيف بدون أعطال",       # Arabic
            "漢字 日本語 中文 한국어",                # CJK mix
            "Em—dash and en–dash with curly “quotes”",
            "Raptors are fifth-generation fighters.",
        ]
        for s in clean_samples:
            assert not has_encoding_artifacts(s), (
                f"False-positive ornament-glyph detection on legitimate text: {s!r}"
            )
