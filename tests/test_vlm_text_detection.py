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
    _detect_first_match,
    detect_text_reading,
    sanitize_text_reading_response,
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
            # Note: "Labeled with the text..." is detected by quote count (> 4 quotes)
            # not by the label pattern alone
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

    def test_invalid_magazine_caption_and_overlay_patterns(self):
        """Magazine caption/overlay reading patterns should be detected."""
        invalid_responses = [
            "Editorial photograph of aircraft on a runway, with a caption identifying the squadron.",
            "Editorial photograph of aircraft lined up on a runway, with text overlay describing CVW-5 assets.",
            "Two-column layout with dense text; footer contains a website, date, and page number.",
            "Dense typographic layout with a caption below the main image.",
            "Advertisement collage featuring a magazine cover titled A400M Atlas.",
            "A QR code is centered on a white background. Below it, a black rectangle contains the number 20.",
            "Dense typographic layout with a stylized logo, company name, and copyright information.",
            "Badge labeled with a squadron name and date.",
            "QR code with a date and a partial URL at the bottom.",
            "Advertisement collage with QR code and ordering information.",
            'Red and black abstract background with white text "WAI THUN".',
            "Google search results page with product images and article snippets.",
            "Editorial photograph of a Lenovo ThinkPad laptop on a wooden surface.",
            "Bar chart showing CPU performance scores for various devices.",
            "Bar chart with longer bars indicating better performance.",
            "Advertisement collage with a laptop displaying a wave painting and text about its features.",
            "Editorial photograph of a tablet displaying a drawing app with a hand-drawn face and text.",
            "FOUNDRY O",
            "Person holding a gaming console with a touchscreen interface displaying game titles and a search bar.",
            "Editorial photograph of a ROG gaming controller displaying the full-screen software interface.",
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
        assert _detect_first_match("") is None
        assert _detect_first_match(None) is None

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
        """Quoted visible words should be rejected even when short."""
        response = 'Diagram showing "bolt" and "trigger" components'
        assert detect_text_reading(response)

    def test_borderline_caps_count(self):
        """Two ALL CAPS words should be allowed."""
        response = "Technical SCHEMATIC showing ASSEMBLY process"
        assert not detect_text_reading(response)

    def test_diagnostic_first_match_returns_pattern_name(self):
        """Diagnostic variant should identify the first firing pattern."""
        cases = [
            (
                "The provided image is a mostly blank white field with a thin vertical strip.",
                "P0_meta_terms",
            ),
            (
                "Text block with paragraph formatting and a hyperlink (fave.co/4n4knLo).",
                "P10_url_or_domain",
            ),
            (
                "Diagram with green section showing a linear flow (Input → LLM → Output).",
                "P14_flow_arrows",
            ),
        ]
        for response, expected in cases:
            assert _detect_first_match(response) == expected

    # =========================================================================
    # Phase 3 Step 0 empirical leak fixtures (qwen3-vl-plus, 2026-05-08)
    # — strengthening Patterns 7-11 closes the leak classes that slipped past
    # the prior validator on real cloud output.
    # =========================================================================

    def test_smart_quote_variants_are_caught(self):
        """Curly quotes / guillemets bypass the ASCII-quote pattern."""
        # PCWorld p24 leak: search-term encoded with curly quotes.
        leaked = 'Screenshot showing search results for “alan” with four games.'
        assert detect_text_reading(leaked)
        leaked = "Screenshot of «Compose» button highlighted in sidebar."
        assert detect_text_reading(leaked)

    def test_markdown_emphasis_proper_nouns_caught(self):
        """Markdown italics around proper nouns transcribe text via formatting."""
        # PCWorld p24 leak class.
        leaked = "Game library showing *Alan Wake*, *Alan Wake Remastered*, and *American Nightmare*."
        assert detect_text_reading(leaked)

    def test_parenthesized_label_list_caught(self):
        """Parenthesized comma-lists of capitalized labels are UI text-reading."""
        # PCWorld p26 leak: Gmail sidebar folders listed in parens.
        leaked = (
            "Software interface showing sidebar navigation with labeled folders "
            "(Inbox, Starred, Snoozed, Sent, Drafts)."
        )
        assert detect_text_reading(leaked)

    def test_url_in_description_caught(self):
        """Bare URLs / domain-with-path in description are always text-reading."""
        # PCWorld p9 leak: hyperlink rendered in the visual.
        leaked = "Text block with paragraph formatting and a hyperlink (fave.co/4n4knLo)."
        assert detect_text_reading(leaked)
        # https variant for completeness.
        leaked = "Diagram with a callout pointing to https://example.com/docs."
        assert detect_text_reading(leaked)

    def test_dotted_identifier_list_caught(self):
        """Three+ distinct dotted identifiers = field-list transcription."""
        # Hao p75 leak: YAML field names enumerated in description.
        leaked = (
            "System schematic showing labeled fields and arrows pointing to key "
            "parameters (apiVersion, kind, metadata.name, spec.containers.name, "
            "image, ports.containerPort)."
        )
        assert detect_text_reading(leaked)

    def test_strengthened_patterns_do_not_overfire_on_legitimate_visuals(self):
        """Negative shapes — strengthening must not reject genuine visual descriptions."""
        legitimate = [
            "A line chart with two trend lines on a grid.",
            "Photograph of a laptop on a wooden desk under warm lighting.",
            "Diagram showing two boxes connected by an arrow.",
            "Bar chart with colored bars, axes, and category labels.",
            # Parenthesized list of lowercase positional words must NOT fire (Pattern 9
            # protects via the capital/dot/camelCase filter).
            "Diagram with three nodes (top, middle, bottom) arranged horizontally.",
            "Color palette swatch (red, green, blue) on white background.",
            # A single dotted identifier in technical context is allowed (Pattern 11
            # requires three+ distinct).
            "Architecture diagram showing module.method as the entry point.",
        ]
        for r in legitimate:
            assert not detect_text_reading(r), f"False positive: {r!r}"

    # =========================================================================
    # Phase 3 Step 0 v2 empirical leak fixtures (qwen3-vl-plus, 2026-05-09)
    # — strengthening Patterns 0/0b/12/13/14 closes leak classes that slipped
    # past Patterns 7-11. Each fixture is the actual leaked phrase from the
    # real flagged chunk.
    # =========================================================================

    def test_provided_image_meta_caught(self):
        """Prompt/asset meta-references via 'the provided image' frame."""
        # PCWorld p31 / p45 leak: response opens with prompt-echo language.
        leaked = "The provided image is a mostly blank white field with a thin vertical strip."
        assert detect_text_reading(leaked)

    def test_instructional_self_reference_caught(self):
        """'Therefore, per the rules' / 'following the instructions' shapes."""
        # PCWorld p31 / p45 leak: model reasons about the prompt.
        leaked = "A mostly blank field. Therefore, per the rules: this is not a diagram."
        assert detect_text_reading(leaked)
        leaked = "Following the instructions, this image lacks a structured diagram."
        assert detect_text_reading(leaked)

    def test_class_noun_followed_by_list_caught(self):
        """'columns for prompts, reference trajectories, ...' — Adedeji p187."""
        leaked = (
            "Table of evaluation metrics with columns for prompts, reference "
            "trajectories, responses, latency, failure status, predicted "
            "trajectories, and scores."
        )
        assert detect_text_reading(leaked)

    def test_list_followed_by_class_noun_caught(self):
        """'Group, Name, Status, ... columns' — Hao p182."""
        leaked = (
            "Table showing Group, Name, Status, Status Name, Type, Node, Start "
            "Time, and Operation columns with status indicators."
        )
        assert detect_text_reading(leaked)

    def test_unicode_flow_arrows_caught(self):
        """'(Input → LLM → Output)' — Adedeji p35."""
        leaked = "Diagram with green section showing a linear flow (Input → LLM → Output)."
        assert detect_text_reading(leaked)

    def test_strengthened_v2_patterns_do_not_overfire(self):
        """Negative shapes for the v2 additions — must not reject legitimate descriptions."""
        legitimate = [
            # Lowercase comma-list of 3 colors with no class-noun anchor.
            "A flag with red, white, and blue stripes arranged horizontally.",
            # 'columns' but the list is 2 items + adjective, not a label list.
            "Bar chart with two columns showing measured and expected values.",
            # 'fields' as a regular plural noun, not a structural label list.
            "Photograph of green fields under a clear blue sky.",
            # 'labels' as part of natural prose without an enumerated list.
            "Diagram with axis labels and a legend in the lower right corner.",
            # Single ASCII arrow in technical context — should NOT fire (Pattern 14
            # requires 2+ ASCII arrows).
            "Code-flow diagram showing input -> output as a single transformation.",
            # Pre-existing legitimate shape — re-asserted to guard against regression.
            "Bar chart with colored bars, axes, and category labels.",
        ]
        for r in legitimate:
            assert not detect_text_reading(r), f"False positive: {r!r}"

    # =========================================================================
    # Phase 3 Step 0 v3 empirical leak fixtures (qwen3-vl-plus, 2026-05-09)
    # — Patterns 15-17 close additional leak classes surfaced by the v3
    # enrichment pass against the c23d3f6+a879e85 strengthened validator.
    # =========================================================================

    def test_class_noun_with_parenthesized_label_list_caught(self):
        """'labeled fields (apiVersion, kind, metadata, data)' — Hao p93."""
        leaked = (
            "Text-based schematic of a Kubernetes ConfigMap YAML definition "
            "showing labeled fields (apiVersion, kind, metadata, data) with "
            "explanatory annotations."
        )
        assert detect_text_reading(leaked)

    def test_class_noun_with_parenthesized_lowercase_phrase_list_caught(self):
        """'stages (data collection/preparation, ...)' — Hao p28."""
        leaked = (
            "System schematic showing a linear sequence of six labeled "
            "processing stages (data collection/preparation, data versioning, "
            "model training, model evaluation, model validation, model "
            "deployment) enclosed in a container."
        )
        assert detect_text_reading(leaked)

    def test_chapter_reference_in_parens_caught(self):
        """'(ch 12-13)' — Hao p34. Numbered chapter / figure / section refs."""
        leaked = (
            "Workflow diagram with components labeled (ch 12-13) and color-coded sections."
        )
        assert detect_text_reading(leaked)
        # Variants
        assert detect_text_reading("Diagram (chapter 5) showing the architecture.")
        assert detect_text_reading("Photograph (figure 3-4) of the apparatus.")
        assert detect_text_reading("Schematic (section 7.2) of the controller.")

    def test_named_flow_chain_caught(self):
        """'through Prometheus to Alertmanager and Grafana' — Hao p111."""
        leaked = (
            "System schematic showing data flow from two applications through "
            "Prometheus to Alertmanager and Grafana, alongside Helm command-"
            "line configuration."
        )
        assert detect_text_reading(leaked)

    # =========================================================================
    # Phase 3 Step 0 v4 RESIDUAL leak fixtures (qwen3-vl-plus, 2026-05-09)
    # — Pattern 18 closes the dense-brand-token-density class that v1-v3
    # patterns missed. These are the 10 chunks documented in §7 of the
    # Phase 3 baseline snapshot, sampled to one fixture per leak shape.
    # =========================================================================

    def test_dense_brand_token_density_caught(self):
        """4+ distinct mid-sentence Capitalized non-vocab tokens = transcription."""
        # Hao p182 leak: config field names from a Kubernetes manifest.
        leaked = (
            "System schematic of resource configuration settings showing "
            "labeled fields for replica count (Min/Max), CPU resources "
            "(Requests/Limits), and Memory."
        )
        assert detect_text_reading(leaked)

    def test_brand_names_in_prose_caught(self):
        """Hao p363 leak: Prometheus + monitoring tool labels."""
        leaked = (
            "Text-based alert notification of a Prometheus monitoring alert "
            "showing labeled fields (Labels, Annotations), a severity "
            "indicator, and Grafana dashboard reference."
        )
        assert detect_text_reading(leaked)

    def test_product_names_in_prose_caught(self):
        """PCWorld p70 leak: Qualcomm + Dell + benchmark names."""
        leaked = (
            "Text block describing battery life test results for Qualcomm "
            "hardware and Dell Plus 14 2-in-1 laptop, referencing PCMark and "
            "Cinebench benchmarks."
        )
        assert detect_text_reading(leaked)

    def test_pipeline_stage_names_caught(self):
        """Adedeji p277 leak: pipeline stage names in prose."""
        leaked = (
            "System schematic with boxed components showing a feedback loop "
            "from Metrics through Anomaly detection to Insights, "
            "Optimization, and Automated remediation."
        )
        assert detect_text_reading(leaked)

    def test_dense_brand_pattern_does_not_overfire(self):
        """Pattern 18 must not reject legitimate visual descriptions.

        Tests use: descriptions that mention 0-3 novel Capitalized tokens
        (allowed); descriptions whose Capitalized tokens are all in the
        common-vocabulary exclusion list (allowed); sentence-initial
        Capitalization on a brand-like word in a short description (the
        first-token-of-each-sentence carve-out applies).
        """
        legitimate = [
            # 0 novel-cap tokens
            "A line chart with two trend lines on a grid.",
            # 1 novel-cap token (single brand reference is allowed — would be
            # caught by Pattern 6/Pattern 17 only if other shapes hit too).
            "Photograph of a Sony camera on a wooden desk.",
            # All Capitalized tokens are in the common-vocabulary list.
            "System schematic showing Top, Middle, and Bottom layers with "
            "Red, Green, and Blue highlights.",
            # Brand-like word at sentence start (excluded as sentence-initial).
            "Prometheus is a metric collection tool. The diagram shows "
            "a clean architecture.",
            # Legitimate technical description — re-asserted to guard against
            # regression on prior layers.
            "Bar chart with colored bars, axes, and category labels.",
        ]
        for r in legitimate:
            assert not detect_text_reading(r), f"False positive: {r!r}"

    def test_v3_patterns_do_not_overfire(self):
        """Negative shapes for v3 patterns."""
        legitimate = [
            # 3-item parenthesized lowercase positional list — Pattern 15's
            # 4-item floor protects this.
            "Diagram with three sections (top, middle, bottom) arranged horizontally.",
            # 'fields' as everyday noun without parenthesized label list.
            "Photograph of green fields stretching to the horizon.",
            # Single capitalized stage name in flow — Pattern 17 requires 3+ chain.
            "Diagram showing data flow from input to Output stage.",
            # Two-stage flow — Pattern 17 requires 3+, so this is allowed.
            "Process moves from start to finish across two boxes.",
            # Generic chapter-like word without the abbreviation pattern.
            "Diagram showing chapters of a book on a shelf.",
            # Page reference using the full word "page" outside parens — not the
            # parenthesized abbrev shape.
            "Photo of a magazine open to a page about technology.",
        ]
        for r in legitimate:
            assert not detect_text_reading(r), f"False positive: {r!r}"


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

    def test_validation_rejects_raw_quotes_before_cleaning(self):
        """Raw quoted headings must be rejected before the cleaner strips quotes."""
        validation = validate_vlm_response(
            'Dense typographic layout with "Headlines" and "Crushing Iran nuclear ambitions".'
        )
        assert not validation.is_valid
        assert validation.text_reading_detected
        assert "Text transcription detected" in validation.issues

    def test_sanitize_preserves_visual_subject_without_read_text(self):
        """Text-leaking responses can be reduced to visual-only descriptions."""
        examples = [
            (
                "Editorial photograph of a Lenovo ThinkPad laptop on a wooden surface.",
                "Editorial photograph of a laptop on a wooden surface.",
            ),
            (
                "Bar chart showing CPU performance scores for various devices, with longer bars indicating better performance.",
                "Bar chart with colored bars, axes, and category labels.",
            ),
            (
                'Google search results page showing "best laptops" query with product images.',
                "search results page query with product images.",
            ),
        ]
        for raw, expected in examples:
            sanitized = sanitize_text_reading_response(raw)
            assert sanitized == expected
            assert not detect_text_reading(sanitized)

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
        # Check for key strict rules in the prompt
        assert "NEVER transcribe" in STRICTER_VISUAL_PROMPT
        assert "CRITICAL RULES" in STRICTER_VISUAL_PROMPT
        assert "VISUAL CONTENT ONLY" in STRICTER_VISUAL_PROMPT


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
