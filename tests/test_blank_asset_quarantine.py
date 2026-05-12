"""
Blank Asset Quarantine Tests
==============================
Validates that blank image/table assets are detected and removed before
final JSONL export. Table markdown content is preserved by promoting
the chunk to TEXT modality (IMAGE/TABLE require asset_ref per schema).

Regression target: CarOK spreadsheet — blank table asset
46d689134b24_012_table_02.png (mean=255, std=0.0, 1.4KB) caused
AUDIT_FAIL (IMAGE category).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mmrag_v2.batch_processor import BatchProcessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_blank_png(path: Path, width: int = 100, height: int = 50) -> Path:
    """Create a blank (all white) PNG file."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    img.save(path, "PNG")
    return path


def _create_content_png(path: Path, width: int = 100, height: int = 50) -> Path:
    """Create a PNG with actual visual content (not blank)."""
    arr = np.random.randint(0, 200, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(path, "PNG")
    return path


def _create_black_png(path: Path, width: int = 100, height: int = 50) -> Path:
    """Create an all-black PNG file (also blank)."""
    img = Image.new("RGB", (width, height), color=(0, 0, 0))
    img.save(path, "PNG")
    return path


# ---------------------------------------------------------------------------
# Unit tests: blank detection
# ---------------------------------------------------------------------------


class TestBlankAssetDetection:
    """BatchProcessor._is_blank_asset must correctly identify blank images."""

    def test_white_image_is_blank(self, tmp_path):
        path = _create_blank_png(tmp_path / "white.png")
        assert BatchProcessor._is_blank_asset(path) is True

    def test_black_image_is_blank(self, tmp_path):
        path = _create_black_png(tmp_path / "black.png")
        assert BatchProcessor._is_blank_asset(path) is True

    def test_content_image_is_not_blank(self, tmp_path):
        path = _create_content_png(tmp_path / "content.png")
        assert BatchProcessor._is_blank_asset(path) is False

    def test_nonexistent_file_is_not_blank(self, tmp_path):
        path = tmp_path / "missing.png"
        assert BatchProcessor._is_blank_asset(path) is False

    def test_carok_dimensions_blank(self, tmp_path):
        """CarOK regression: 1565x198 blank table asset."""
        path = _create_blank_png(tmp_path / "table.png", width=1565, height=198)
        assert BatchProcessor._is_blank_asset(path) is True

    def test_combat_p27_figure_36_shape_is_blank(self, tmp_path):
        """Plan v2.9 Phase E regression: Combat_Aircraft_August_2025 p27
        `figure_36` has mean=253, std=7.4 — a near-white image with a
        faint watermark / compression noise above the prior std<5 cap.
        After the threshold relaxation to std<10, this case is correctly
        classified as blank."""
        # Construct an image with mean ~253, std ~7 (faint watermark
        # pattern on near-white background).
        arr = np.full((100, 50, 3), 254, dtype=np.uint8)
        # Add tiny watermark pixels
        arr[50:55, 25:30] = 200
        img = Image.fromarray(arr)
        path = tmp_path / "combat_figure_36.png"
        img.save(path, "PNG")
        assert BatchProcessor._is_blank_asset(path) is True

    def test_real_photo_with_low_brightness_is_not_blank(self, tmp_path):
        """Negative-control: a dark photo (mean ~50, std ~30) must NOT
        be classified as blank — its std exceeds the new threshold."""
        rng = np.random.default_rng(seed=7)
        arr = rng.integers(0, 100, (100, 50, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        path = tmp_path / "dark_photo.png"
        img.save(path, "PNG")
        assert BatchProcessor._is_blank_asset(path) is False

    def test_near_white_but_clearly_content_is_not_blank(self, tmp_path):
        """Negative-control: a near-white image with substantive content
        variance (std=15) must NOT be classified as blank — its std
        exceeds the new threshold even though the mean is high."""
        rng = np.random.default_rng(seed=11)
        # mean ~240, std ~15 — clearly textured even if light
        arr = (rng.normal(loc=240, scale=15, size=(100, 50, 3))
               .clip(0, 255).astype(np.uint8))
        img = Image.fromarray(arr)
        path = tmp_path / "textured.png"
        img.save(path, "PNG")
        assert BatchProcessor._is_blank_asset(path) is False


# ---------------------------------------------------------------------------
# Schema contract tests
# ---------------------------------------------------------------------------


class TestSchemaContract:
    """Verify that blank asset handling respects the IngestionChunk schema."""

    def test_table_chunk_requires_asset_ref(self):
        """TABLE modality requires asset_ref — cannot just clear it."""
        from mmrag_v2.schema.ingestion_schema import (
            FileType,
            IngestionChunk,
            Modality,
            SpatialMetadata,
            ChunkMetadata,
            HierarchyMetadata,
        )

        with pytest.raises(ValueError, match="requires asset_ref"):
            IngestionChunk(
                chunk_id="test_001",
                doc_id="test",
                modality=Modality.TABLE,
                content="| A | B |",
                metadata=ChunkMetadata(
                    source_file="test.pdf",
                    file_type=FileType.PDF,
                    page_number=1,
                    spatial=SpatialMetadata(bbox=[0, 0, 1000, 1000]),
                    hierarchy=HierarchyMetadata(
                        breadcrumb_path=["Doc", "Page 1"],
                        level=1,
                    ),
                ),
            )

    def test_text_chunk_does_not_require_asset_ref(self):
        """TEXT modality does NOT require asset_ref — safe for promotion."""
        from mmrag_v2.schema.ingestion_schema import (
            FileType,
            IngestionChunk,
            Modality,
            SpatialMetadata,
            ChunkMetadata,
            HierarchyMetadata,
        )

        # Should not raise
        chunk = IngestionChunk(
            chunk_id="test_001",
            doc_id="test",
            modality=Modality.TEXT,
            content="| A | B |\n|---|---|\n| 1 | 2 |",
            metadata=ChunkMetadata(
                source_file="test.pdf",
                file_type=FileType.PDF,
                page_number=1,
                spatial=SpatialMetadata(bbox=[0, 0, 1000, 1000]),
                hierarchy=HierarchyMetadata(
                    breadcrumb_path=["Doc", "Page 1"],
                    level=1,
                ),
            ),
        )
        assert chunk.modality == Modality.TEXT
        assert chunk.asset_ref is None
