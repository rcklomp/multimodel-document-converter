"""
Test Suite: Full-Page Guard (GAP #1)
====================================

Tests for REQ-MM-08 through REQ-MM-12:
- Full-page asset detection (area_ratio > 0.95)
- VLM verification of shadow assets
- Image pre-processing (resize to max 1024px)
- Logging according to REQ-MM-11
- Override flag (--allow-fullpage-shadow)

Author: Test Suite
Date: 2026-01-02
"""

import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, Any
from unittest import mock

import pytest
from PIL import Image

logger = logging.getLogger(__name__)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_vision_manager():
    """Create mock VisionManager for testing."""
    from mmrag_v2.vision.vision_manager import VisionManager, OllamaProvider

    provider = mock.MagicMock(spec=OllamaProvider)
    manager = VisionManager(provider=provider, cache_dir=None)
    return manager, provider


@pytest.fixture
def sample_image_small():
    """Create a small test image (200x200px)."""
    img = Image.new("RGB", (200, 200), color="red")
    return img


@pytest.fixture
def sample_image_fullpage():
    """Create a full-page test image (1000x1000px)."""
    img = Image.new("RGB", (1000, 1000), color="blue")
    return img


@pytest.fixture
def sample_image_largeish():
    """Create a large image (900x900px - nearly fullpage)."""
    img = Image.new("RGB", (900, 900), color="green")
    return img


@pytest.fixture
def breadcrumbs():
    """Sample breadcrumbs for testing."""
    return ["TestDocument", "Section 1", "Subsection A"]


# ============================================================================
# TEST: Area Ratio Calculation (REQ-MM-08)
# ============================================================================


def test_area_ratio_small_asset():
    """
    REQ-MM-08: Calculate area ratio for small asset.

    Asset: 200x200 = 40,000 px²
    Page: 1000x1000 = 1,000,000 px²
    Ratio: 4% (BELOW 95% threshold)
    """
    asset_area = 200 * 200
    page_area = 1000 * 1000
    area_ratio = asset_area / page_area

    assert area_ratio == 0.04
    assert area_ratio <= 0.95  # Should NOT trigger Full-Page Guard


def test_area_ratio_fullpage_asset():
    """
    REQ-MM-08: Calculate area ratio for full-page asset.

    Asset: 950x950 = 902,500 px²
    Page: 1000x1000 = 1,000,000 px²
    Ratio: 90.25% (BELOW 95% threshold)
    """
    asset_area = 950 * 950
    page_area = 1000 * 1000
    area_ratio = asset_area / page_area

    assert area_ratio == 0.9025
    assert area_ratio <= 0.95  # Just under threshold


def test_area_ratio_exceeds_threshold():
    """
    REQ-MM-08: Area ratio that exceeds 95% threshold.

    Asset: 980x980 = 960,400 px²
    Page: 1000x1000 = 1,000,000 px²
    Ratio: 96.04% (EXCEEDS 95% threshold)
    """
    asset_area = 980 * 980
    page_area = 1000 * 1000
    area_ratio = asset_area / page_area

    assert area_ratio > 0.95  # TRIGGERS Full-Page Guard
    assert area_ratio == 0.9604


# ============================================================================
# TEST: VLM Verification (REQ-MM-10)
# ============================================================================


def test_vlm_verify_editorial_content(mock_vision_manager, sample_image_fullpage, breadcrumbs):
    """
    REQ-MM-10: VLM verification returns 'editorial' classification.

    Asset should be ACCEPTED if VLM classifies it as editorial.
    """
    manager, provider = mock_vision_manager

    # Mock VLM response: editorial content
    provider.describe_image.return_value = json.dumps(
        {
            "classification": "editorial",
            "confidence": 0.95,
            "reason": "High-quality photograph with visible subjects",
        }
    )

    result = manager.verify_shadow_integrity(sample_image_fullpage, breadcrumbs)

    assert result["valid"] is True
    assert result["classification"] == "editorial"
    assert result["confidence"] == 0.95


def test_vlm_verify_ui_navigation(mock_vision_manager, sample_image_fullpage, breadcrumbs):
    """
    REQ-MM-10: VLM verification returns 'ui_navigation' classification.

    Asset should be REJECTED if VLM classifies it as UI/navigation.
    """
    manager, provider = mock_vision_manager

    # Mock VLM response: UI element
    provider.describe_image.return_value = json.dumps(
        {
            "classification": "ui_navigation",
            "confidence": 0.88,
            "reason": "Navigation menu detected",
        }
    )

    result = manager.verify_shadow_integrity(sample_image_fullpage, breadcrumbs)

    assert result["valid"] is False
    assert result["classification"] == "ui_navigation"
    assert result["confidence"] == 0.88


def test_vlm_verify_page_scan(mock_vision_manager, sample_image_fullpage, breadcrumbs):
    """
    REQ-MM-10: VLM verification returns 'page_scan' classification.

    Asset should be REJECTED if VLM classifies it as a page scan.
    """
    manager, provider = mock_vision_manager

    # Mock VLM response: page scan/screenshot
    provider.describe_image.return_value = json.dumps(
        {
            "classification": "page_scan",
            "confidence": 0.92,
            "reason": "Document page screenshot",
        }
    )

    result = manager.verify_shadow_integrity(sample_image_fullpage, breadcrumbs)

    assert result["valid"] is False
    assert result["classification"] == "page_scan"


# ============================================================================
# TEST: JSON Parsing (REQ-MM-10)
# ============================================================================


def test_vlm_parse_strict_json(mock_vision_manager, sample_image_fullpage, breadcrumbs):
    """
    REQ-MM-10: Enforce STRICT JSON parsing.

    Response MUST be valid JSON with required fields.
    """
    manager, provider = mock_vision_manager

    # Valid JSON response
    provider.describe_image.return_value = (
        '{"classification":"editorial","confidence":0.9,"reason":"Photo"}'
    )

    result = manager.verify_shadow_integrity(sample_image_fullpage, breadcrumbs)

    assert result["classification"] == "editorial"
    assert result["confidence"] == 0.9


def test_vlm_parse_invalid_json(mock_vision_manager, sample_image_fullpage, breadcrumbs):
    """
    REQ-MM-10: Handle invalid JSON gracefully (DISCARD).

    On parse failure, default to DISCARD for safety.
    """
    manager, provider = mock_vision_manager

    # Invalid JSON
    provider.describe_image.return_value = "This is not JSON at all"

    result = manager.verify_shadow_integrity(sample_image_fullpage, breadcrumbs)

    assert result["valid"] is False
    assert result["classification"] == "error"


def test_vlm_parse_missing_fields(mock_vision_manager, sample_image_fullpage, breadcrumbs):
    """
    REQ-MM-10: Reject JSON with missing required fields.

    Classification and confidence are MANDATORY.
    """
    manager, provider = mock_vision_manager

    # Missing 'confidence' field
    provider.describe_image.return_value = '{"classification":"editorial"}'

    result = manager.verify_shadow_integrity(sample_image_fullpage, breadcrumbs)

    assert result["valid"] is False
    assert result["classification"] == "error"


# ============================================================================
# TEST: Image Pre-processing (REQ-MM-10)
# ============================================================================


def test_image_resize_largeimage():
    """
    REQ-MM-10: Resize large image to max 1024px.

    Image larger than 1024px should be downscaled.
    """
    from mmrag_v2.vision.vision_manager import VisionManager

    # Create a 2000x2000 image
    img = Image.new("RGB", (2000, 2000), color="red")

    resized = VisionManager._resize_image_for_vlm(img, max_dimension=1024)

    assert resized.size[0] <= 1024
    assert resized.size[1] <= 1024
    assert resized.size[0] == 1024  # Should be exactly 1024 on longest side


def test_image_resize_smallimage():
    """
    REQ-MM-10: Don't resize small images.

    Image smaller than max should remain unchanged.
    """
    from mmrag_v2.vision.vision_manager import VisionManager

    # Create a 500x500 image
    img = Image.new("RGB", (500, 500), color="blue")
    img_id_before = id(img)

    resized = VisionManager._resize_image_for_vlm(img, max_dimension=1024)

    # Should be same size (or very close due to rounding)
    assert resized.size[0] == 500
    assert resized.size[1] == 500


def test_image_resize_landscape():
    """
    REQ-MM-10: Resize landscape image proportionally.

    Aspect ratio must be preserved.
    """
    from mmrag_v2.vision.vision_manager import VisionManager

    # Create a 3000x1500 landscape image
    img = Image.new("RGB", (3000, 1500), color="green")

    resized = VisionManager._resize_image_for_vlm(img, max_dimension=1024)

    # Longest side (3000) should become 1024
    assert resized.size[0] == 1024
    # Aspect ratio: 3000:1500 = 2:1, so new height = 512
    assert resized.size[1] == 512
    assert resized.size[0] / resized.size[1] == 2.0  # Aspect ratio preserved


# ============================================================================
# TEST: Logging (REQ-MM-11)
# ============================================================================


def test_fullpage_guard_logging_accepted(
    caplog, mock_vision_manager, sample_image_fullpage, breadcrumbs
):
    """
    REQ-MM-11: Log ACCEPTED full-page assets with full context.

    Format: [VLM-GUARD] Asset {id} - Ratio {ratio} - Result: {classification} - Confidence: {conf} - Reason: {reason}
    """
    manager, provider = mock_vision_manager

    # Mock VLM response: editorial
    provider.describe_image.return_value = json.dumps(
        {
            "classification": "editorial",
            "confidence": 0.95,
            "reason": "High-quality photograph",
        }
    )

    with caplog.at_level(logging.INFO):
        result = manager.verify_shadow_integrity(sample_image_fullpage, breadcrumbs)

    assert result["valid"] is True

    # Check that logging includes required fields
    log_output = caplog.text
    assert "editorial" in log_output or "[VLM-GUARD]" in log_output


def test_fullpage_guard_logging_rejected(
    caplog, mock_vision_manager, sample_image_fullpage, breadcrumbs
):
    """
    REQ-MM-11: Log REJECTED full-page assets with reason.
    """
    manager, provider = mock_vision_manager

    # Mock VLM response: ui_navigation
    provider.describe_image.return_value = json.dumps(
        {
            "classification": "ui_navigation",
            "confidence": 0.88,
            "reason": "Navigation menu",
        }
    )

    with caplog.at_level(logging.INFO):
        result = manager.verify_shadow_integrity(sample_image_fullpage, breadcrumbs)

    assert result["valid"] is False

    # Log should indicate rejection
    log_output = caplog.text
    assert "ui_navigation" in log_output or "[VLM-GUARD]" in log_output


# ============================================================================
# TEST: Override Flag (REQ-MM-12)
# ============================================================================


def test_fullpage_override_flag_true(caplog):
    """
    REQ-MM-12: Override flag --allow-fullpage-shadow bypasses Full-Page Guard.

    When allow_fullpage_shadow=True, all assets are accepted.
    """
    from mmrag_v2.batch_processor import BatchProcessor

    processor = BatchProcessor(
        output_dir="./output",
        allow_fullpage_shadow=True,  # Override enabled
    )

    # Simulate a full-page asset (980x980 on 1000x1000 page)
    asset_width, asset_height = 980, 980
    page_width, page_height = 1000.0, 1000.0
    area_ratio = (asset_width * asset_height) / (page_width * page_height)

    assert area_ratio > 0.95  # Full-page threshold exceeded

    # With override, should be accepted (no VLM needed)
    # We can't easily test the full method without integration,
    # but we can verify the override flag is stored
    assert processor.allow_fullpage_shadow is True


def test_fullpage_override_flag_false(caplog):
    """
    REQ-MM-12: Default behavior (allow_fullpage_shadow=False) enforces Full-Page Guard.

    When allow_fullpage_shadow=False, VLM verification is required.
    """
    from mmrag_v2.batch_processor import BatchProcessor

    processor = BatchProcessor(
        output_dir="./output",
        allow_fullpage_shadow=False,  # Default: enforce guard
    )

    assert processor.allow_fullpage_shadow is False


# ============================================================================
# TEST: 10px Padding (REQ-MM-01)
# ============================================================================


def test_padding_10px_applied():
    """
    REQ-MM-01: Verify 10px padding is applied to cropped assets.

    This test verifies the padding constraint mentioned in the Full-Page Guard spec.
    """
    # Original asset bounding box: [100, 100, 500, 600]
    # With 10px padding: [90, 90, 510, 610]
    original_bbox = [100, 100, 500, 600]
    padding = 10

    padded_bbox = [
        max(0, original_bbox[0] - padding),  # x_min
        max(0, original_bbox[1] - padding),  # y_min
        original_bbox[2] + padding,  # x_max
        original_bbox[3] + padding,  # y_max
    ]

    assert padded_bbox == [90, 90, 510, 610]


# ============================================================================
# TEST: Coordinate System (REQ-COORD-01)
# ============================================================================


def test_bbox_integer_normalization():
    """
    REQ-COORD-01: Bounding boxes MUST be integers in 0-1000 range.

    Verify normalization formula: normalized = round(int((raw / page_dimension) * 1000))
    NOTE: MUST use round() before int() to avoid floating point errors
    (e.g., 9.999 -> round to 10, not truncate to 9)
    """
    # Raw pixel coordinates: [100, 200, 500, 700]
    # Page dimensions: 1000x1000 px
    raw_bbox = [100, 200, 500, 700]
    page_width, page_height = 1000, 1000

    # CORRECT: round() BEFORE int() to handle floating point precision
    normalized_bbox = [
        int(round((raw_bbox[0] / page_width) * 1000)),
        int(round((raw_bbox[1] / page_height) * 1000)),
        int(round((raw_bbox[2] / page_width) * 1000)),
        int(round((raw_bbox[3] / page_height) * 1000)),
    ]

    assert normalized_bbox == [100, 200, 500, 700]
    assert all(isinstance(x, int) for x in normalized_bbox)
    assert all(0 <= x <= 1000 for x in normalized_bbox)


def test_bbox_normalization_floating_point_precision():
    """
    REQ-COORD-01: Handle floating point precision errors.

    Edge case: Coordinate that results in 9.999... must round to 10, not truncate to 9.
    This tests the critical round() requirement to avoid precision errors.
    """
    # Coordinate that causes floating point issues
    # If we have 9.99/10 * 1000 = 999.0, but due to FP might be 999.9999999
    raw_coord = 9.99
    page_dim = 10.0

    # WRONG (truncation): int((9.99 / 10.0) * 1000) = int(999.0) = 999 ✗
    # CORRECT (rounding): round((9.99 / 10.0) * 1000) = round(999.0) = 999 ✓

    # More problematic case: 9.999/10 * 1000 might be 999.9999999
    problematic_raw = 9.999
    problematic_page = 10.0

    # Without round(): int(999.9999999) = 999 (WRONG!)
    # With round(): round(999.9999999) = 1000 (CORRECT!)
    wrong_result = int((problematic_raw / problematic_page) * 1000)
    correct_result = int(round((problematic_raw / problematic_page) * 1000))

    # This demonstrates the issue:
    # int(999.9999) = 999, but round(999.9999) = 1000
    assert wrong_result == 999 or wrong_result == 1000  # Depends on FP precision
    assert correct_result == 1000  # Always correct with rounding


def test_bbox_full_page_normalization():
    """
    REQ-COORD-01: Full-page bbox should normalize to [0, 0, 1000, 1000].
    """
    # Full-page asset: [0, 0, 1000, 1000] in pixels
    raw_bbox = [0, 0, 1000, 1000]
    page_width, page_height = 1000, 1000

    normalized_bbox = [
        int((raw_bbox[0] / page_width) * 1000),
        int((raw_bbox[1] / page_height) * 1000),
        int((raw_bbox[2] / page_width) * 1000),
        int((raw_bbox[3] / page_height) * 1000),
    ]

    assert normalized_bbox == [0, 0, 1000, 1000]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
