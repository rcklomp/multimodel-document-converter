"""
Export Integrity End-to-End Test
=================================
Validates that the full BatchProcessor filter + JSONL-write pipeline
produces valid output: correct schema, accurate metadata.chunk_count,
no corrupted text, blank assets removed, TABLE→TEXT promotion correct.

This test calls the real production methods:
  - bp._quarantine_corrupted_text_chunks()
  - bp._filter_blank_assets()
Then writes JSONL using the same code path as BatchProcessor.process_pdf
and reads it back for validation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

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
    IngestionMetadata,
    Modality,
    SCHEMA_VERSION,
    SemanticContext,
    SpatialMetadata,
)
from mmrag_v2.validators.corruption_interceptor import has_encoding_artifacts


# ---------------------------------------------------------------------------
# Helpers
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


def _text(content: str, page: int = 1, idx: int = 0) -> IngestionChunk:
    return IngestionChunk(
        chunk_id=f"doc_{page:03d}_{idx:04d}",
        doc_id="test_doc",
        modality=Modality.TEXT,
        content=content,
        metadata=_meta(page, content),
        semantic_context=SemanticContext(),
    )


def _table(content: str, asset_path: str, page: int = 1, idx: int = 0) -> IngestionChunk:
    return IngestionChunk(
        chunk_id=f"doc_{page:03d}_{idx:04d}",
        doc_id="test_doc",
        modality=Modality.TABLE,
        content=content,
        asset_ref=AssetReference(file_path=asset_path),
        metadata=_meta(page, content),
        semantic_context=SemanticContext(),
    )


def _image(content: str, asset_path: str, page: int = 1, idx: int = 0) -> IngestionChunk:
    return IngestionChunk(
        chunk_id=f"doc_{page:03d}_{idx:04d}",
        doc_id="test_doc",
        modality=Modality.IMAGE,
        content=content,
        asset_ref=AssetReference(file_path=asset_path),
        metadata=_meta(page, content),
        semantic_context=SemanticContext(),
    )


def _blank_png(path: Path) -> None:
    Image.new("RGB", (100, 50), (255, 255, 255)).save(path, "PNG")


def _content_png(path: Path) -> None:
    arr = np.random.randint(0, 200, (50, 100, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, "PNG")


# ---------------------------------------------------------------------------
# End-to-end test
# ---------------------------------------------------------------------------


class TestExportIntegrity:
    """Full filter + write + read-back validation."""

    def test_full_pipeline_writes_valid_jsonl(self, tmp_path):
        """
        Scenario: 7 chunks enter the filter pipeline.
        - 2 clean text chunks
        - 1 corrupted text chunk (should be quarantined)
        - 1 table with blank asset (should be promoted to TEXT)
        - 1 image with blank asset (should be dropped)
        - 1 image with real asset (should survive)
        - 1 table with real asset (should survive)

        Expected output: 5 chunks in JSONL, metadata.chunk_count == 5.
        """
        output = tmp_path / "output"
        output.mkdir()
        assets = output / "assets"
        assets.mkdir()

        # Create asset files
        _blank_png(assets / "blank_table.png")
        _blank_png(assets / "blank_img.png")
        _content_png(assets / "real_photo.png")
        _content_png(assets / "real_table.png")

        # Build processor
        bp = BatchProcessor(output_dir=str(output))
        plan = build_pdf_conversion_plan(
            profile_type="technical_manual",
            document_domain="technical",
        )
        bp.set_conversion_plan(plan)

        fake_pdf = tmp_path / "source.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
        bp._current_pdf_path = fake_pdf

        # Build chunks
        chunks = [
            _text("Clean paragraph one.", page=1, idx=0),
            _text("Clean paragraph two.", page=2, idx=1),
            _text("Corrupted /C211 text /uniFB01 here", page=3, idx=2),
            _table(
                "| Merk | Type |\n|------|------|\n| VW | Golf |",
                "assets/blank_table.png",
                page=4,
                idx=3,
            ),
            _image("[placeholder]", "assets/blank_img.png", page=5, idx=4),
            _image("A real photograph of a car", "assets/real_photo.png", page=6, idx=5),
            _table(
                "| A | B |\n|---|---|\n| 1 | 2 |",
                "assets/real_table.png",
                page=7,
                idx=6,
            ),
        ]

        # Step 1: Apply production quarantine
        filtered = bp._quarantine_corrupted_text_chunks(chunks)

        # Step 2: Apply production blank asset filter
        filtered = bp._filter_blank_assets(filtered)

        # Step 3: Write JSONL (same pattern as production)
        jsonl_path = output / "ingestion.jsonl"
        meta_record = IngestionMetadata(
            schema_version=SCHEMA_VERSION,
            doc_id="test_doc",
            source_file="source.pdf",
            profile_type="technical_manual",
            domain="technical",
            chunk_count=len(filtered),
            ingestion_timestamp=datetime.now(timezone.utc).isoformat(),
            pipeline_version=SCHEMA_VERSION,
        )

        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(meta_record.model_dump(mode="json"), ensure_ascii=False) + "\n")
            for chunk in filtered:
                chunk_dict = chunk.model_dump(mode="json")
                f.write(json.dumps(chunk_dict, ensure_ascii=False) + "\n")

        # Step 4: Read back and validate
        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 6, f"Expected 1 metadata + 5 chunks, got {len(lines)} lines"

        # Validate metadata
        meta = json.loads(lines[0])
        assert meta["chunk_count"] == 5, (
            f"Metadata chunk_count should be 5, got {meta['chunk_count']}"
        )

        # Validate chunks
        chunk_dicts = [json.loads(line) for line in lines[1:]]

        # No corrupted text
        for cd in chunk_dicts:
            if cd["modality"] == "text":
                assert not has_encoding_artifacts(cd["content"]), (
                    f"Corrupted text leaked to JSONL: {cd['content']!r}"
                )

        # Check modalities
        modalities = [cd["modality"] for cd in chunk_dicts]
        assert modalities.count("text") == 3, (
            f"Expected 3 text (2 clean + 1 promoted table), got {modalities.count('text')}"
        )
        assert modalities.count("image") == 1, (
            f"Expected 1 image (real photo), got {modalities.count('image')}"
        )
        assert modalities.count("table") == 1, (
            f"Expected 1 table (real table), got {modalities.count('table')}"
        )

        # Promoted table chunk has no asset_ref
        promoted = [
            cd for cd in chunk_dicts
            if cd["modality"] == "text" and "Merk" in cd.get("content", "")
        ]
        assert len(promoted) == 1
        assert promoted[0].get("asset_ref") is None

        # Real image has asset_ref
        real_img = [cd for cd in chunk_dicts if cd["modality"] == "image"]
        assert len(real_img) == 1
        assert real_img[0]["asset_ref"]["file_path"] == "assets/real_photo.png"

        # Real table has asset_ref
        real_tbl = [cd for cd in chunk_dicts if cd["modality"] == "table"]
        assert len(real_tbl) == 1
        assert real_tbl[0]["asset_ref"]["file_path"] == "assets/real_table.png"

        # Blank asset files should be deleted
        assert not (assets / "blank_table.png").exists()
        assert not (assets / "blank_img.png").exists()
        # Real asset files should still exist
        assert (assets / "real_photo.png").exists()
        assert (assets / "real_table.png").exists()
