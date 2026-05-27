"""V3.0 Phase A step 1 bridge tests: IngestionChunk.from_uir / to_uir.

Charter: docs/ARCHITECTURE_V3_DRAFT_0.5.md §3.2 Phase A step 1
(refactor IngestionChunk to consume UIR).

These tests pin:
  - Modality is unified (schema.Modality is universal.intermediate.Modality).
  - ChunkType narrows to Modality per C14 (one-direction reconciliation).
  - from_uir builds a valid IngestionChunk for TEXT / IMAGE / TABLE / CODE
    modalities given UIRChunks emitted by the (eventual) v3 chunker.
  - to_uir produces a UIRChunk whose content + locator + modality matches.
  - Round-trip via to_uir -> from_uir preserves identity-relevant fields
    (content, modality, chunk_id, page_number, bbox).
"""

from __future__ import annotations

import pytest

from mmrag_v2.schema.ingestion_schema import (
    CHUNKTYPE_TO_MODALITY,
    MODALITY_TO_DEFAULT_CHUNKTYPE,
    AssetReference,
    ChunkMetadata,
    ChunkType,
    FileType,
    HierarchyMetadata,
    IngestionChunk,
    Modality,
    SpatialMetadata,
    _generate_chunk_id,
)
from mmrag_v2.universal.intermediate import (
    ConfidenceBreakdown,
    CoordinateFrame,
    Locator,
    LocatorType,
    Modality as UIRModality,
    StructuralFlag,
    UIRChunk,
)


class TestModalityUnification:
    """Charter §3.2 + C14: Modality is the single source of truth."""

    def test_schema_modality_is_uir_modality(self):
        # Same class identity — no two-way shim, no duplicate enum.
        assert Modality is UIRModality

    def test_five_value_widened_vocabulary(self):
        names = {m.name for m in Modality}
        assert names == {"TEXT", "IMAGE", "TABLE", "CODE", "FORM"}

    def test_str_enum_compatible(self):
        # Subclassing str makes the enum Pydantic-friendly and JSON-clean.
        assert Modality.TEXT == "text"
        assert Modality.CODE == "code"
        assert Modality.FORM == "form"


class TestChunkTypeNarrowing:
    """C14: ChunkType narrows to Modality; no two-way map."""

    def test_every_chunktype_maps_to_modality(self):
        for ct in ChunkType:
            assert ct in CHUNKTYPE_TO_MODALITY, f"ChunkType.{ct.name} unmapped"

    def test_code_chunktype_maps_to_code_modality(self):
        assert CHUNKTYPE_TO_MODALITY[ChunkType.CODE] == Modality.CODE

    def test_default_chunktype_per_modality(self):
        # The reverse direction is many-to-one (Modality.TEXT has 7
        # candidate ChunkTypes); a default exists for v3.0 emission.
        assert MODALITY_TO_DEFAULT_CHUNKTYPE[Modality.TEXT] == ChunkType.PARAGRAPH
        assert MODALITY_TO_DEFAULT_CHUNKTYPE[Modality.CODE] == ChunkType.CODE
        # IMAGE/TABLE/FORM carry no text-classification chunk_type:
        assert MODALITY_TO_DEFAULT_CHUNKTYPE[Modality.IMAGE] is None
        assert MODALITY_TO_DEFAULT_CHUNKTYPE[Modality.TABLE] is None
        assert MODALITY_TO_DEFAULT_CHUNKTYPE[Modality.FORM] is None


def _make_text_uir(content: str = "Hello world.", page: int = 1) -> UIRChunk:
    return UIRChunk(
        modality=Modality.TEXT,
        content=content,
        locator=Locator(
            type=LocatorType.BBOX,
            bbox=[100, 200, 800, 250],
            page_number=page,
            coordinate_frame=CoordinateFrame.PDF_PAGE_PORTRAIT,
        ),
        confidence=ConfidenceBreakdown(
            text_extraction_confidence=0.99,
            applicable={"text_extraction_confidence"},
        ),
        extraction_method="docling_direct",
        extraction_engine_version="docling-2.86.0",
        parent_heading="Chapter 1",
    )


def _make_image_uir(page: int = 2) -> UIRChunk:
    return UIRChunk(
        modality=Modality.IMAGE,
        content="A diagram of the lifecycle management process.",
        locator=Locator(
            type=LocatorType.BBOX,
            bbox=[50, 100, 950, 700],
            page_number=page,
            coordinate_frame=CoordinateFrame.PDF_PAGE_PORTRAIT,
        ),
        confidence=ConfidenceBreakdown(
            classification_confidence=0.92,
            applicable={"classification_confidence"},
        ),
        extraction_method="vlm_enrichment",
        extraction_engine_version="docling-2.86.0",
        asset_ref="abc123_002_image_0.png",
    )


def _make_table_uir(page: int = 3) -> UIRChunk:
    return UIRChunk(
        modality=Modality.TABLE,
        content="| h1 | h2 |\n|---|---|\n| a | b |",
        locator=Locator(
            type=LocatorType.BBOX,
            bbox=[80, 300, 900, 600],
            page_number=page,
            coordinate_frame=CoordinateFrame.PDF_PAGE_PORTRAIT,
        ),
        confidence=ConfidenceBreakdown(),
        extraction_method="docling_direct",
        extraction_engine_version="docling-2.86.0",
        asset_ref="abc123_003_table_0.png",
    )


def _make_code_uir(page: int = 4) -> UIRChunk:
    return UIRChunk(
        modality=Modality.CODE,
        content="def hello():\n    print('hi')",
        locator=Locator(
            type=LocatorType.BBOX,
            bbox=[100, 400, 800, 500],
            page_number=page,
            coordinate_frame=CoordinateFrame.PDF_PAGE_PORTRAIT,
        ),
        confidence=ConfidenceBreakdown(),
        extraction_method="docling_direct",
        extraction_engine_version="docling-2.86.0",
        structural_flags={StructuralFlag.PARTIAL_CODE_IN_BLOCK},
    )


class TestFromUir:
    """Charter §3.2 Phase A step 1: IngestionChunk.from_uir is the v3.0
    native emission boundary."""

    def test_text_chunk_basic(self):
        uir = _make_text_uir()
        chunk = IngestionChunk.from_uir(
            uir, doc_id="abc123abc123", source_file="foo.pdf"
        )
        assert chunk.modality == Modality.TEXT
        assert chunk.content == "Hello world."
        assert chunk.metadata.page_number == 1
        assert chunk.metadata.chunk_type == ChunkType.PARAGRAPH
        assert chunk.metadata.spatial is not None
        assert chunk.metadata.spatial.bbox == [100, 200, 800, 250]
        assert chunk.metadata.hierarchy.parent_heading == "Chapter 1"
        assert chunk.chunk_id.startswith("abc123abc123_001_text_")

    def test_image_chunk_requires_asset(self):
        uir = _make_image_uir()
        chunk = IngestionChunk.from_uir(
            uir,
            doc_id="abc123abc123",
            source_file="foo.pdf",
            page_width=1000,
            page_height=1000,
        )
        assert chunk.modality == Modality.IMAGE
        assert chunk.asset_ref is not None
        assert chunk.asset_ref.file_path == "abc123_002_image_0.png"
        assert chunk.metadata.spatial.bbox == [50, 100, 950, 700]
        # validate_multimodal_requirements should pass (image has bbox+asset)

    def test_table_chunk(self):
        uir = _make_table_uir()
        chunk = IngestionChunk.from_uir(
            uir, doc_id="abc123abc123", source_file="foo.pdf"
        )
        assert chunk.modality == Modality.TABLE
        assert chunk.asset_ref is not None
        assert chunk.metadata.search_priority == "medium"

    def test_code_chunk_widening(self):
        uir = _make_code_uir()
        chunk = IngestionChunk.from_uir(
            uir, doc_id="abc123abc123", source_file="foo.pdf"
        )
        # CODE is the v3.0 widening — accepted, no asset/bbox required.
        assert chunk.modality == Modality.CODE
        assert chunk.metadata.chunk_type == ChunkType.CODE
        assert chunk.metadata.partial_code is True
        assert chunk.asset_ref is None

    def test_image_without_bbox_rejected(self):
        # An IMAGE locator with FLOW_OFFSET is impossible per Locator
        # validation, but we test the IngestionChunk validator: image must
        # have spatial.bbox + asset_ref per v2.16 contract.
        # We construct a UIRChunk with BBOX locator but missing asset
        # to confirm the validator catches the missing asset:
        bad = UIRChunk(
            modality=Modality.IMAGE,
            content="x",
            locator=Locator(
                type=LocatorType.BBOX,
                bbox=[1, 1, 10, 10],
                page_number=1,
                coordinate_frame=CoordinateFrame.PDF_PAGE_PORTRAIT,
            ),
            confidence=ConfidenceBreakdown(),
            extraction_method="docling_direct",
            extraction_engine_version="docling-2.86.0",
            asset_ref=None,  # missing
        )
        with pytest.raises(ValueError, match="QA-CHECK-05"):
            IngestionChunk.from_uir(
                bad, doc_id="abc123abc123", source_file="foo.pdf"
            )

    def test_page_number_required_on_locator(self):
        # FLOW_OFFSET locator without page_number is invalid for projection
        # — IngestionChunk requires page_number.
        bad = UIRChunk(
            modality=Modality.TEXT,
            content="x",
            locator=Locator(
                type=LocatorType.FLOW_OFFSET,
                path="anchor:1",
            ),
            confidence=ConfidenceBreakdown(),
            extraction_method="docling_direct",
            extraction_engine_version="docling-2.86.0",
        )
        with pytest.raises(ValueError, match="missing page_number"):
            IngestionChunk.from_uir(
                bad, doc_id="abc123abc123", source_file="foo.pdf"
            )


class TestToUir:
    """Reverse builder: IngestionChunk.to_uir for the identity-half gate."""

    def test_text_chunk_to_uir(self):
        # Build an IngestionChunk via the existing factory, then project.
        from mmrag_v2.schema.ingestion_schema import create_text_chunk

        chunk = create_text_chunk(
            doc_id="abc123abc123",
            content="Hello world.",
            source_file="foo.pdf",
            file_type=FileType.PDF,
            page_number=1,
            bbox=[100, 200, 800, 250],
        )
        uir = chunk.to_uir()
        assert uir.modality == Modality.TEXT
        assert uir.content == "Hello world."
        assert uir.locator.type == LocatorType.BBOX
        assert uir.locator.bbox == [100, 200, 800, 250]
        assert uir.locator.page_number == 1

    def test_image_chunk_to_uir(self):
        from mmrag_v2.schema.ingestion_schema import create_image_chunk

        chunk = create_image_chunk(
            doc_id="abc123abc123",
            content="A photograph.",
            source_file="foo.pdf",
            file_type=FileType.PDF,
            page_number=2,
            asset_path="abc123_002_image_0.png",
            bbox=[50, 100, 950, 700],
        )
        uir = chunk.to_uir()
        assert uir.modality == Modality.IMAGE
        assert uir.asset_ref == "abc123_002_image_0.png"
        assert uir.locator.bbox == [50, 100, 950, 700]


class TestRoundTrip:
    """Identity-half axis: content + modality + locator must survive round-trip."""

    def test_text_round_trip(self):
        from mmrag_v2.schema.ingestion_schema import create_text_chunk

        original = create_text_chunk(
            doc_id="abc123abc123",
            content="Round-trip preservation test.",
            source_file="foo.pdf",
            file_type=FileType.PDF,
            page_number=7,
            bbox=[10, 20, 990, 900],
        )
        rebuilt = IngestionChunk.from_uir(
            original.to_uir(),
            doc_id=original.doc_id,
            source_file=original.metadata.source_file,
            file_type=original.metadata.file_type,
        )
        # Identity-relevant axes (Charter §3.2 identity half) preserved:
        assert rebuilt.content == original.content
        assert rebuilt.modality == original.modality
        assert rebuilt.metadata.page_number == original.metadata.page_number
        assert rebuilt.metadata.spatial.bbox == original.metadata.spatial.bbox
        assert rebuilt.chunk_id == original.chunk_id

    def test_image_round_trip(self):
        from mmrag_v2.schema.ingestion_schema import create_image_chunk

        original = create_image_chunk(
            doc_id="abc123abc123",
            content="A photograph.",
            source_file="foo.pdf",
            file_type=FileType.PDF,
            page_number=2,
            asset_path="abc123_002_image_0.png",
            bbox=[50, 100, 950, 700],
        )
        rebuilt = IngestionChunk.from_uir(
            original.to_uir(),
            doc_id=original.doc_id,
            source_file=original.metadata.source_file,
            file_type=original.metadata.file_type,
        )
        assert rebuilt.content == original.content
        assert rebuilt.modality == original.modality
        assert rebuilt.asset_ref.file_path == original.asset_ref.file_path
        assert rebuilt.metadata.spatial.bbox == original.metadata.spatial.bbox
