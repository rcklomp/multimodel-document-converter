"""
Finalization Bridge Tests
==========================
Tests that invoke the ACTUAL BatchProcessor methods for corruption
quarantine and blank asset handling. No production logic is duplicated;
each test calls bp._quarantine_corrupted_text_chunks() or
bp._filter_blank_assets() directly.

Regression targets:
- Combat Aircraft: corrupted chunks leak through OCR patching
- CarOK: blank table asset causes AUDIT_FAIL
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest
from PIL import Image

from mmrag_v2.batch_processor import BatchProcessor
from mmrag_v2.engines.pdf_plan import build_pdf_conversion_plan
from mmrag_v2.schema.ingestion_schema import (
    AssetReference,
    ChunkMetadata,
    ChunkType,
    FileType,
    HierarchyMetadata,
    IngestionChunk,
    Modality,
    SemanticContext,
    SpatialMetadata,
)
from mmrag_v2.validators.corruption_interceptor import has_encoding_artifacts


# ---------------------------------------------------------------------------
# Chunk factories
# ---------------------------------------------------------------------------


def _meta(page: int, content: str) -> ChunkMetadata:
    return ChunkMetadata(
        source_file="test.pdf",
        file_type=FileType.PDF,
        page_number=page,
        spatial=SpatialMetadata(bbox=[0, 0, 1000, 1000]),
        hierarchy=HierarchyMetadata(
            breadcrumb_path=["Doc", f"Page {page}"],
            level=1,
        ),
        chunk_type=ChunkType.PARAGRAPH,
        schema_version="2.7.0",
        refined_content=content,
    )


def _text_chunk(
    content: str,
    doc_id: str = "test_doc",
    page: int = 1,
    chunk_idx: int = 0,
) -> IngestionChunk:
    return IngestionChunk(
        chunk_id=f"{doc_id}_{page:03d}_{chunk_idx:04d}",
        doc_id=doc_id,
        modality=Modality.TEXT,
        content=content,
        metadata=_meta(page, content),
        semantic_context=SemanticContext(),
    )


def _table_chunk(
    content: str,
    asset_path: str,
    doc_id: str = "test_doc",
    page: int = 1,
    chunk_idx: int = 0,
) -> IngestionChunk:
    return IngestionChunk(
        chunk_id=f"{doc_id}_{page:03d}_{chunk_idx:04d}",
        doc_id=doc_id,
        modality=Modality.TABLE,
        content=content,
        asset_ref=AssetReference(file_path=asset_path),
        metadata=_meta(page, content),
        semantic_context=SemanticContext(),
    )


def _image_chunk(
    content: str,
    asset_path: str,
    doc_id: str = "test_doc",
    page: int = 1,
    chunk_idx: int = 0,
) -> IngestionChunk:
    return IngestionChunk(
        chunk_id=f"{doc_id}_{page:03d}_{chunk_idx:04d}",
        doc_id=doc_id,
        modality=Modality.IMAGE,
        content=content,
        asset_ref=AssetReference(file_path=asset_path),
        metadata=_meta(page, content),
        semantic_context=SemanticContext(),
    )


def _create_blank_png(path: Path) -> None:
    Image.new("RGB", (100, 50), color=(255, 255, 255)).save(path, "PNG")


def _create_content_png(path: Path) -> None:
    arr = np.random.randint(0, 200, (50, 100, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, "PNG")


def _build_bp(tmp_path: Path) -> BatchProcessor:
    output = tmp_path / "output"
    output.mkdir(exist_ok=True)
    (output / "assets").mkdir(exist_ok=True)
    bp = BatchProcessor(output_dir=str(output))
    plan = build_pdf_conversion_plan(
        profile_type="technical_manual",
        document_domain="technical",
    )
    bp.set_conversion_plan(plan)
    return bp


# ---------------------------------------------------------------------------
# Corruption quarantine — calls bp._quarantine_corrupted_text_chunks()
# ---------------------------------------------------------------------------


class TestCorruptionQuarantineBridge:
    """Invoke the real BatchProcessor quarantine method."""

    def test_corrupted_text_removed(self, tmp_path):
        """Chunks with /C211 etc. are removed by the production method."""
        bp = _build_bp(tmp_path)

        chunks = [
            _text_chunk("Clean text paragraph.", page=1, chunk_idx=0),
            _text_chunk("Still has /C211 and /C23 artifacts", page=2, chunk_idx=1),
            _text_chunk("Another clean chunk.", page=3, chunk_idx=2),
        ]

        result = bp._quarantine_corrupted_text_chunks(chunks)

        assert len(result) == 2
        for c in result:
            assert not has_encoding_artifacts(c.content)

    def test_combat_partial_patch_simulation(self, tmp_path):
        """Combat-like scenario: 25 patched + 22 still corrupted."""
        bp = _build_bp(tmp_path)

        chunks = []
        for i in range(25):
            chunks.append(_text_chunk(f"Patched clean text {i}", page=i + 1, chunk_idx=i))
        for i in range(22):
            chunks.append(
                _text_chunk(
                    f"Unpatched /C211 text /uniFB01 {i}",
                    page=i + 26,
                    chunk_idx=i + 25,
                )
            )

        result = bp._quarantine_corrupted_text_chunks(chunks)
        assert len(result) == 25

    def test_image_chunks_not_quarantined(self, tmp_path):
        """IMAGE chunks pass through even if content has artifact patterns."""
        bp = _build_bp(tmp_path)
        output = tmp_path / "output"
        _create_content_png(output / "assets" / "img.png")

        chunks = [
            _image_chunk("[Image with /C211 in caption]", "assets/img.png", page=1),
        ]
        result = bp._quarantine_corrupted_text_chunks(chunks)
        assert len(result) == 1

    def test_quarantine_disabled(self, tmp_path):
        """When disabled, corrupted chunks pass through."""
        bp = _build_bp(tmp_path)
        bp._quarantine_corrupted_chunks = False

        chunks = [_text_chunk("Has /C211 artifacts", page=1)]
        result = bp._quarantine_corrupted_text_chunks(chunks)
        assert len(result) == 1

    def test_replacement_char_collapsed_at_chunk_creation(self, tmp_path):
        """Plan v2.9 Phase B1 extension (2026-05-11): U+FFFD replacement
        characters are now collapsed at chunk-creation time by the
        universal `_collapse_replacement_chars` validator in
        `mmrag_v2.schema.ingestion_schema`. They no longer survive into
        chunks that reach the BatchProcessor, so the quarantine sees
        clean content and does NOT drop the chunk. The previous contract
        (BP quarantine drops U+FFFD chunks) was correct for v2.8 but is
        obsolete now that the producer cleans them universally.

        The quarantine remains responsible for other corruption
        signatures (CIDFont placeholders, em-dash/CS runs) \u2014 see
        `test_em_dash_runs_quarantined` and the
        `tests/test_corruption_quarantine_toc_exemption.py` suite for
        the current contract."""
        bp = _build_bp(tmp_path)
        chunks = [_text_chunk("Text with \ufffd replacement char", page=1)]
        # Verify the chunk arrived sanitized:
        assert "\ufffd" not in chunks[0].content
        result = bp._quarantine_corrupted_text_chunks(chunks)
        # Now clean \u2192 no longer dropped by the quarantine.
        assert len(result) == 1

    def test_em_dash_runs_quarantined(self, tmp_path):
        """Phase 4 Step 3 corruption signatures other than U+FFFD remain
        the quarantine's responsibility. This pins the remaining contract
        after the U+FFFD collapse was lifted into chunk creation."""
        bp = _build_bp(tmp_path)
        chunks = [_text_chunk("Squadron note \u2014\u2014\u2014\u2014\u2014\u2014|CCCCCCCCCC SSSSSSSSSS", page=1)]
        result = bp._quarantine_corrupted_text_chunks(chunks)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Blank asset filter — calls bp._filter_blank_assets()
# ---------------------------------------------------------------------------


class TestBlankAssetFilterBridge:
    """Invoke the real BatchProcessor blank asset filter method."""

    def test_blank_table_promoted_to_text(self, tmp_path):
        """TABLE with blank asset promoted to TEXT, preserving markdown."""
        bp = _build_bp(tmp_path)
        output = tmp_path / "output"
        _create_blank_png(output / "assets" / "table_01.png")

        table_md = "| Col A | Col B |\n|-------|-------|\n| Val 1 | Val 2 |"
        chunks = [_table_chunk(table_md, "assets/table_01.png", page=1)]

        result = bp._filter_blank_assets(chunks)

        assert len(result) == 1
        promoted = result[0]
        assert promoted.modality == Modality.TEXT
        assert promoted.asset_ref is None
        assert promoted.content == table_md
        # Verify promoted chunk passes schema validation:
        # TEXT modality does not require asset_ref
        assert promoted.modality == Modality.TEXT
        assert promoted.asset_ref is None

    def test_blank_image_dropped(self, tmp_path):
        """IMAGE with blank asset is dropped entirely."""
        bp = _build_bp(tmp_path)
        output = tmp_path / "output"
        _create_blank_png(output / "assets" / "img_01.png")

        chunks = [
            _image_chunk("[placeholder]", "assets/img_01.png", page=1),
            _text_chunk("Clean text", page=2, chunk_idx=1),
        ]

        result = bp._filter_blank_assets(chunks)
        assert len(result) == 1
        assert result[0].modality == Modality.TEXT

    def test_nonblank_asset_survives(self, tmp_path):
        """Non-blank assets pass through the filter unchanged."""
        bp = _build_bp(tmp_path)
        output = tmp_path / "output"
        _create_content_png(output / "assets" / "photo.png")

        chunks = [_image_chunk("A real photo", "assets/photo.png", page=1)]

        result = bp._filter_blank_assets(chunks)
        assert len(result) == 1
        assert result[0].modality == Modality.IMAGE
        assert result[0].asset_ref is not None

    def test_blank_asset_file_deleted(self, tmp_path):
        """The blank asset file itself is removed from disk."""
        bp = _build_bp(tmp_path)
        output = tmp_path / "output"
        asset_path = output / "assets" / "blank.png"
        _create_blank_png(asset_path)
        assert asset_path.exists()

        chunks = [_image_chunk("[x]", "assets/blank.png", page=1)]
        bp._filter_blank_assets(chunks)
        assert not asset_path.exists()

    def test_filter_disabled(self, tmp_path):
        """When disabled, blank assets pass through."""
        bp = _build_bp(tmp_path)
        bp._drop_blank_assets = False
        output = tmp_path / "output"
        _create_blank_png(output / "assets" / "blank.png")

        chunks = [_image_chunk("[x]", "assets/blank.png", page=1)]
        result = bp._filter_blank_assets(chunks)
        assert len(result) == 1

    def test_carok_blank_table_regression(self, tmp_path):
        """CarOK regression: 1565x198 blank table asset, markdown preserved."""
        bp = _build_bp(tmp_path)
        output = tmp_path / "output"
        asset = output / "assets" / "46d689_012_table_02.png"
        Image.new("RGB", (1565, 198), (255, 255, 255)).save(asset, "PNG")

        md = "| Merk | Type | Kenteken | Kleur |\n|------|------|----------|-------|\n| VW | Golf | AB-12-CD | Zwart |"
        chunks = [_table_chunk(md, "assets/46d689_012_table_02.png", page=12)]

        result = bp._filter_blank_assets(chunks)
        assert len(result) == 1
        assert result[0].modality == Modality.TEXT
        assert "Merk" in result[0].content
        assert not asset.exists()  # blank file cleaned up
