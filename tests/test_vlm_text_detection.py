"""
Test VLM Text Detection Validation
===================================

Tests for detect_text_reading() function that enforces REQ-VLM-SIGNAL.

Author: Cline (Senior Architect)
Date: January 7, 2026
"""

import pytest

from mmrag_v2.vision.vision_prompts import (
    STRICTER_VISUAL_PROMPT,
    detect_text_reading,
    validate_vlm_response,
    clean_vlm_response,
)


class TestDetectTextReading:
    """Test cases for detect_text_reading() function."""

    # =========================================================================
    # Valid Visual Descriptions (Should return False = acceptable)
    # =========================================================================

    def test_valid_diagram_description(self):
        """Valid diagram descriptions should pass."""
        valid_responses = [
            "Exploded diagram of bolt action mechanism with 12 labeled components",
            "Technical schematic showing trigger assembly spring positions",
            "Cross-sectional view of rifle barrel with rifling visible",
            "Detailed technical drawing with numbered callouts",
            "Assembly diagram for rifle stock with installation sequence",
        ]
        for response in valid_responses:
            assert not detect_text_reading(response), f"Should pass: {response}"

    def test_valid_photograph_description(self):
        """Valid photograph descriptions should pass."""
        valid_responses = [
            "Black and white photograph of workshop machinery",
            "Vintage advertisement layout with product grid",
            "Close-up of rifle barrel threading",
            "Historical firearms collection arranged on display table",
            "Author portrait in gun workshop surrounded by tools",
        ]
        for response in valid_responses:
            assert not detect_text_reading(response), f"Should pass: {response}"

    def test_valid_short_descriptions(self):
        """Short valid descriptions should pass."""
        valid_responses = [
            "Rifle diagram",
            "Technical schematic",
            "Workshop photo",
            "Exploded view",
        ]
        for response in valid_responses:
            assert not detect_text_reading(response), f"Should pass: {response}"

    def test_valid_text_only_page_response(self):
        """Text-only page fallback response should pass."""
        response = "Text-only page - visual content captured by OCR"
        assert not detect_text_reading(response)

    # =========================================================================
    # Invalid Text Transcriptions (Should return True = rejected)
    # =========================================================================

    def test_invalid_text_says_pattern(self):
        """'The text says' pattern should be detected."""
        invalid_responses = [
            "The text says 'INTRODUCTION'",
            "Text says 'Chapter 1'",
            "The text reads 'SAFETY INSTRUCTIONS'",
        ]
        for response in invalid_responses:
            assert detect_text_reading(response), f"Should reject: {response}"

    def test_invalid_caption_pattern(self):
        """Caption reading patterns should be detected."""
        invalid_responses = [
            "The caption reads 'Figure 1: Bolt Assembly'",
            "The caption says 'Diagram of mechanism'",
            "Caption reads: 'Technical illustration'",
        ]
        for response in invalid_responses:
            assert detect_text_reading(response), f"Should reject: {response}"

    def test_invalid_label_pattern(self):
        """Label reading patterns should be detected."""
        invalid_responses = [
            "The label indicates 'Component A'",
            "The label says 'Part 1'",
            "Labeled with the text 'CAUTION'",
        ]
        for response in invalid_responses:
            assert detect_text_reading(response), f"Should reject: {response}"

    def test_invalid_title_pattern(self):
        """Title reading patterns should be detected."""
        invalid_responses = [
            "The title is 'FIREARMS MAINTENANCE'",
            "The title says 'Introduction'",
            "The heading reads 'Safety First'",
        ]
        for response in invalid_responses:
            assert detect_text_reading(response), f"Should reject: {response}"

    def test_invalid_excessive_quotes(self):
        """Excessive quoted text should be detected."""
        invalid_responses = [
            'Shows "CHAPTER 1" and "INTRODUCTION" and "SAFETY"',
            "Text visible: 'WARNING' 'CAUTION' 'DANGER'",
        ]
        for response in invalid_responses:
            assert detect_text_reading(response), f"Should reject: {response}"

    def test_invalid_all_caps_transcription(self):
        """Multiple ALL CAPS words indicating transcription should be detected."""
        invalid_responses = [
            "Page shows INTRODUCTION CHAPTER SAFETY INSTRUCTIONS",
            "FIREARMS MAINTENANCE MANUAL EDITION visible",
        ]
        for response in invalid_responses:
            assert detect_text_reading(response), f"Should reject: {response}"

    def test_invalid_long_quoted_string(self):
        """Long quoted strings (>20 chars) should be detected."""
        response = 'The text says "This is a very long transcribed sentence from the page"'
        assert detect_text_reading(response)

    def test_invalid_written_on_pattern(self):
        """'Written on' patterns should be detected."""
        response = "Written on the page is 'IMPORTANT NOTICE'"
        assert detect_text_reading(response)

    def test_invalid_text_visible_pattern(self):
        """'Text visible' patterns should be detected."""
        invalid_responses = [
            "Text visible: CHAPTER ONE",
            "Text visible in the image shows 'Safety'",
        ]
        for response in invalid_responses:
            assert detect_text_reading(response), f"Should reject: {response}"

    # =========================================================================
    # Edge Cases
    # =========================================================================

    def test_empty_response(self):
        """Empty response should pass (handled by validate_vlm_response)."""
        assert not detect_text_reading("")
        assert not detect_text_reading(None)

    def test_allowed_technical_abbreviations(self):
        """Technical abbreviations in caps should be allowed."""
        valid_responses = [
            "PDF document layout with PDF icons visible",
            "OCR processing diagram showing OCR workflow",
            "RGB color chart with DPI measurements",
        ]
        for response in valid_responses:
            assert not detect_text_reading(response), f"Should pass: {response}"

    def test_borderline_quote_count(self):
        """Four quotes (2 pairs) should be allowed."""
        # Exactly 2 quote pairs should pass
        response = 'Diagram showing "bolt" and "trigger" components'
        assert not detect_text_reading(response)

    def test_borderline_caps_count(self):
        """Two ALL CAPS words should be allowed."""
        response = "Technical SCHEMATIC showing ASSEMBLY process"
        assert not detect_text_reading(response)


class TestValidateVLMResponse:
    """Test cases for validate_vlm_response() function."""

    def test_valid_response(self):
        """Valid responses should pass validation."""
        validation = validate_vlm_response(
            "Exploded diagram of bolt action mechanism with 12 labeled components"
        )
        assert validation.is_valid
        assert not validation.text_reading_detected
        assert len(validation.issues) == 0

    def test_invalid_text_reading(self):
        """Text reading should fail validation."""
        validation = validate_vlm_response("The text says 'INTRODUCTION'")
        assert not validation.is_valid
        assert validation.text_reading_detected
        assert "Text transcription detected" in validation.issues

    def test_empty_response(self):
        """Empty response should fail validation."""
        validation = validate_vlm_response("")
        assert not validation.is_valid
        assert "Empty response" in validation.issues

    def test_generic_fallback_response(self):
        """Generic fallback phrases should fail validation."""
        validation = validate_vlm_response("Unable to describe the image content")
        assert not validation.is_valid
        assert "Generic fallback response detected" in validation.issues

    def test_short_response_after_cleaning(self):
        """Very short responses after cleaning should fail."""
        validation = validate_vlm_response("This image shows X")
        # After cleaning "This image shows", only "X" remains
        assert not validation.is_valid
        assert "Response too short after cleaning" in validation.issues


class TestCleanVLMResponse:
    """Test cases for clean_vlm_response() function."""

    def test_removes_this_image_shows(self):
        """Should remove 'This image shows' prefix."""
        dirty = "This image shows an exploded diagram of a rifle"
        clean = clean_vlm_response(dirty)
        assert clean.startswith("An exploded diagram") or clean.startswith("Exploded")

    def test_removes_the_page_contains(self):
        """Should remove 'The page contains' prefix."""
        dirty = "The page contains a technical schematic"
        clean = clean_vlm_response(dirty)
        assert "page contains" not in clean.lower()

    def test_capitalizes_after_cleaning(self):
        """First letter should be capitalized after cleaning."""
        dirty = "This image shows a small diagram"
        clean = clean_vlm_response(dirty)
        assert clean[0].isupper()


class TestStricterPrompt:
    """Test that STRICTER_VISUAL_PROMPT is properly defined."""

    def test_stricter_prompt_exists(self):
        """Stricter prompt should be defined."""
        assert STRICTER_VISUAL_PROMPT is not None
        assert len(STRICTER_VISUAL_PROMPT) > 100

    def test_stricter_prompt_contains_rules(self):
        """Stricter prompt should contain strict rules."""
        assert "DO NOT transcribe" in STRICTER_VISUAL_PROMPT
        assert "DO NOT use quotes" in STRICTER_VISUAL_PROMPT
        assert "REJECTED" in STRICTER_VISUAL_PROMPT


class TestIntegrationExample:
    """Integration example showing how to use in ElementProcessor."""

    def test_retry_logic_example(self):
        """Example of retry logic for VLM text detection."""

        # Simulate VLM responses
        vlm_responses = [
            "The text says 'INTRODUCTION'",  # Invalid - retry (forbidden phrase)
            "The heading reads 'CHAPTER 1'",  # Invalid - retry (forbidden phrase)
            "Technical diagram of rifle bolt mechanism",  # Valid - accept
        ]

        max_retries = 3
        valid_response = None

        for attempt, response in enumerate(vlm_responses[:max_retries]):
            validation = validate_vlm_response(response)
            if validation.is_valid:
                valid_response = validation.cleaned_response
                break

        assert valid_response is not None
        assert "Technical diagram" in valid_response

    def test_all_retries_fail_fallback(self):
        """Example when all retries fail, use fallback."""

        # All responses contain text transcription
        vlm_responses = [
            "The text says 'INTRODUCTION'",
            "The caption reads 'Figure 1'",
            "The label indicates 'Component A'",
        ]

        max_retries = 3
        valid_response = None
        fallback = "Visual element (VLM description unavailable)"

        for attempt, response in enumerate(vlm_responses[:max_retries]):
            validation = validate_vlm_response(response)
            if validation.is_valid:
                valid_response = validation.cleaned_response
                break

        # If no valid response, use fallback
        if valid_response is None:
            valid_response = fallback

        assert valid_response == fallback


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
