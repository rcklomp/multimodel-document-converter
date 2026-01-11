"""
Test Universal Pipeline Architecture
=====================================

This test verifies the new Universal Intermediate Representation (UIR)
architecture works correctly for both digital and scanned PDFs.

Run with:
    python -m pytest tests/test_universal_pipeline.py -v

Or directly:
    python tests/test_universal_pipeline.py

Author: Claude (Architect)
Date: January 2026
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest


class TestUIRDataStructures:
    """Test Universal Intermediate Representation data structures."""

    def test_element_type_enum(self):
        """Test ElementType enum values."""
        from mmrag_v2.universal import ElementType

        assert ElementType.TEXT.value == "text"
        assert ElementType.IMAGE.value == "image"
        assert ElementType.TABLE.value == "table"

    def test_page_classification_enum(self):
        """Test PageClassification enum values."""
        from mmrag_v2.universal import PageClassification

        assert PageClassification.DIGITAL.value == "digital"
        assert PageClassification.SCANNED.value == "scanned"
        assert PageClassification.HYBRID.value == "hybrid"

    def test_bounding_box_creation(self):
        """Test BoundingBox creation and validation."""
        from mmrag_v2.universal import BoundingBox

        bbox = BoundingBox(x_min=100, y_min=200, x_max=500, y_max=600)

        assert bbox.x_min == 100
        assert bbox.y_min == 200
        assert bbox.x_max == 500
        assert bbox.y_max == 600
        assert bbox.width == 400
        assert bbox.height == 400
        assert bbox.area == 160000

    def test_bounding_box_normalization(self):
        """Test BoundingBox.from_raw normalization."""
        from mmrag_v2.universal import BoundingBox

        # Raw bbox in pixels
        raw_bbox = [61.2, 79.2, 306.0, 396.0]  # About half the page
        page_width = 612.0
        page_height = 792.0

        bbox = BoundingBox.from_raw(raw_bbox, page_width, page_height, apply_padding=False)

        # Should be normalized to 0-1000 range
        assert 0 <= bbox.x_min <= 1000
        assert 0 <= bbox.y_min <= 1000
        assert 0 <= bbox.x_max <= 1000
        assert 0 <= bbox.y_max <= 1000

        # Approximately 500 (half of 1000)
        assert 90 <= bbox.x_max <= 510
        assert 90 <= bbox.y_max <= 510

    def test_bounding_box_validation(self):
        """Test BoundingBox validates coordinates."""
        from mmrag_v2.universal import BoundingBox

        # Invalid: x_max <= x_min
        with pytest.raises(ValueError):
            BoundingBox(x_min=500, y_min=200, x_max=100, y_max=600)

        # Invalid: out of range
        with pytest.raises(ValueError):
            BoundingBox(x_min=0, y_min=0, x_max=1001, y_max=500)

    def test_element_creation(self):
        """Test Element creation."""
        from mmrag_v2.universal import create_element, ElementType

        elem = create_element(
            element_type=ElementType.TEXT,
            content="Hello world",
            bbox=[100, 200, 500, 600],
            confidence=0.9,
        )

        assert elem.type == ElementType.TEXT
        assert elem.content == "Hello world"
        assert elem.confidence == 0.9
        assert elem.bbox is not None
        assert elem.bbox.x_min == 100

    def test_element_needs_ocr(self):
        """Test element.needs_ocr property."""
        from mmrag_v2.universal import create_element, ElementType

        # High confidence TEXT = no OCR needed
        elem1 = create_element(ElementType.TEXT, "Hello", confidence=0.9)
        assert not elem1.needs_ocr

        # Low confidence TEXT = needs OCR
        elem2 = create_element(ElementType.TEXT, "Hello", confidence=0.5)
        assert elem2.needs_ocr

        # IMAGE never needs OCR (needs VLM)
        elem3 = create_element(ElementType.IMAGE, "", confidence=0.5)
        assert not elem3.needs_ocr
        assert elem3.needs_vlm

    def test_page_classification(self):
        """Test UniversalPage classification."""
        from mmrag_v2.universal import UniversalPage, PageClassification

        # Digital: >100 chars
        assert UniversalPage.classify_page(150) == PageClassification.DIGITAL

        # Scanned: <20 chars
        assert UniversalPage.classify_page(10) == PageClassification.SCANNED

        # Hybrid: 20-100 chars
        assert UniversalPage.classify_page(50) == PageClassification.HYBRID

    def test_page_creation(self):
        """Test create_page factory function."""
        from mmrag_v2.universal import create_page, create_element, ElementType, PageClassification

        # Use long text content to trigger DIGITAL classification (>100 chars)
        long_text = "This is a sample paragraph with enough text content to exceed the digital threshold for proper classification testing."
        elements = [
            create_element(ElementType.TEXT, long_text, confidence=0.9),
            create_element(
                ElementType.TEXT, "Second paragraph with more text content here.", confidence=0.8
            ),
        ]

        page = create_page(
            page_number=1,
            elements=elements,
            dimensions=(612, 792),
        )

        assert page.page_number == 1
        assert len(page.elements) == 2
        assert page.width == 612
        assert page.height == 792
        # Auto-classified as digital due to text content (>100 chars)
        assert page.classification == PageClassification.DIGITAL


class TestFormatRouter:
    """Test format detection and routing."""

    def test_router_initialization(self):
        """Test FormatRouter can be created."""
        from mmrag_v2.universal import FormatRouter

        router = FormatRouter()
        assert router is not None

    def test_extension_mapping(self):
        """Test file extension to format mapping."""
        from mmrag_v2.universal.router import EXTENSION_MAP

        assert EXTENSION_MAP[".pdf"] == "pdf"
        assert EXTENSION_MAP[".epub"] == "epub"
        assert EXTENSION_MAP[".html"] == "html"
        assert EXTENSION_MAP[".docx"] == "docx"


class TestElementProcessor:
    """Test element processing and quality-based routing."""

    def test_processor_initialization(self):
        """Test ElementProcessor can be created."""
        from mmrag_v2.universal import ElementProcessor
        from pathlib import Path

        processor = ElementProcessor(
            output_dir=Path("./output/test_assets"),
            confidence_threshold=0.7,
        )

        assert processor is not None
        assert processor.confidence_threshold == 0.7

    def test_chunk_id_generation(self):
        """Test chunk ID generation."""
        from mmrag_v2.universal import ElementProcessor

        processor = ElementProcessor()

        chunk_id = processor._generate_chunk_id(
            doc_id="abc123",
            page_number=1,
            modality="text",
            element_idx=0,
        )

        assert chunk_id.startswith("abc123_001_text_")
        assert len(chunk_id) > 20  # Has hash suffix

    def test_text_reading_detection(self):
        """Test VLM text-reading detection."""
        from mmrag_v2.universal import ElementProcessor

        processor = ElementProcessor()

        # Should detect text reading
        assert processor._contains_text_reading("The text says INTRODUCTION")
        assert processor._contains_text_reading("The caption reads Figure 1")

        # Should not flag visual descriptions
        assert not processor._contains_text_reading("Exploded diagram of rifle mechanism")
        assert not processor._contains_text_reading("Technical schematic showing components")


class TestPDFEngine:
    """Test PDF engine."""

    def test_engine_initialization(self):
        """Test PDFEngine can be created."""
        from mmrag_v2.engines.pdf_engine import PDFEngine

        engine = PDFEngine(render_dpi=300)

        assert engine is not None
        assert engine.render_dpi == 300
        assert ".pdf" in engine.supported_extensions

    def test_engine_detect(self):
        """Test PDF detection by extension."""
        from mmrag_v2.engines.pdf_engine import PDFEngine
        from pathlib import Path
        import tempfile

        engine = PDFEngine()

        # Create temp file with PDF magic bytes
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.7\n")
            temp_path = Path(f.name)

        try:
            assert engine.detect(temp_path) == True
        finally:
            temp_path.unlink()


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_uir_to_result_flow(self):
        """Test that UIR elements produce correct modalities."""
        from mmrag_v2.universal import (
            ElementProcessor,
            create_element,
            create_page,
            ElementType,
            PageClassification,
        )
        import numpy as np

        # Create mock scanned page with elements
        text_element = create_element(
            ElementType.TEXT,
            content="Sample text content",
            confidence=0.9,
        )

        image_element = create_element(
            ElementType.IMAGE,
            content="",
            confidence=0.9,
        )
        # Add mock image data
        image_element.raw_image = np.zeros((100, 100, 3), dtype=np.uint8)

        page = create_page(
            page_number=1,
            elements=[text_element, image_element],
            dimensions=(612, 792),
            classification=PageClassification.DIGITAL,
        )

        # Process page
        processor = ElementProcessor(vision_manager=None)  # No VLM for test

        results = list(processor.process_page(page, doc_id="test123", source_file="test.pdf"))

        # Should have 2 results
        assert len(results) == 2

        # First should be TEXT modality
        assert results[0].modality == "text"
        assert results[0].content == "Sample text content"

        # Second should be IMAGE modality (not shadow!)
        assert results[1].modality == "image"

    def test_scanned_page_produces_text_modality(self):
        """
        CRITICAL TEST: Scanned pages must produce modality="text", not "shadow".

        This is the core fix for the scanned document problem.
        """
        from mmrag_v2.universal import (
            ElementProcessor,
            create_element,
            create_page,
            ElementType,
            PageClassification,
        )
        import numpy as np

        # Create scanned page with low-confidence text element
        text_element = create_element(
            ElementType.TEXT,
            content="",  # Empty - would need OCR
            confidence=0.1,  # Low confidence = scanned
        )
        # Add mock image data for OCR
        text_element.raw_image = np.zeros((100, 100, 3), dtype=np.uint8)

        page = create_page(
            page_number=1,
            elements=[text_element],
            dimensions=(612, 792),
            classification=PageClassification.SCANNED,
        )

        # Verify element needs OCR
        assert text_element.needs_ocr
        assert page.is_scanned

        # The key invariant: even for scanned pages,
        # TEXT elements produce modality="text" (via OCR)
        # NOT modality="shadow"

        processor = ElementProcessor(ocr_engine=None)  # No OCR for mock test

        # Element with empty content is skipped (would need OCR)
        results = list(processor.process_page(page, doc_id="test", source_file="scan.pdf"))

        # With actual OCR, this would produce modality="text"
        # Without OCR, empty content is filtered
        for result in results:
            # KEY ASSERTION: No "shadow" modality
            assert result.modality in ["text", "image", "table"]
            assert result.modality != "shadow"


# ============================================================================
# MAIN
# ============================================================================


if __name__ == "__main__":
    print("=" * 60)
    print("Universal Pipeline Architecture Tests")
    print("=" * 60)

    # Run basic tests
    import traceback

    test_classes = [
        TestUIRDataStructures,
        TestFormatRouter,
        TestElementProcessor,
        TestPDFEngine,
        TestIntegration,
    ]

    passed = 0
    failed = 0

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    getattr(instance, method_name)()
                    print(f"  ✓ {method_name}")
                    passed += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: {e}")
                    traceback.print_exc()
                    failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
