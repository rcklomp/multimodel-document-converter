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
