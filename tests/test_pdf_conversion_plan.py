"""Tests for shared PDF conversion planning and Docling adapter boundaries."""

from __future__ import annotations

import ast
import pytest
from pathlib import Path
from types import SimpleNamespace

from mmrag_v2.engines.docling_adapter import DoclingPdfAdapter
from mmrag_v2.engines.pdf_plan import (
    INTELLIGENCE_METADATA_KEYS,
    PdfConversionPlan,
    build_pdf_conversion_plan,
)
from mmrag_v2.schema.ingestion_schema import CHUNK_FACTORY_METADATA_KEYS


def _digital_plan(**overrides):
    params = {
        "enable_ocr": False,
        "document_modality": "native_digital",
        "profile_type": "digital_magazine",
        "profile_sensitivity": 0.8,
        "min_image_dims": "100x100",
        "confidence_threshold": 0.7,
        "document_domain": "technical",
        "total_pages": 100,
        "image_density": 0.2,
        "avg_text_per_page": 500.0,
    }
    params.update(overrides)
    return build_pdf_conversion_plan(**params)


class FakeInputFormat:
    PDF = "pdf"


class FakeTableFormerMode:
    ACCURATE = "accurate"


class FakePictureDescriptionOptions:
    def __init__(self):
        self.classification_deny = []


class FakePdfPipelineOptions:
    latest = None

    def __init__(self):
        FakePdfPipelineOptions.latest = self
        self.picture_description_options = FakePictureDescriptionOptions()
        self.sort_by_reading_order = None
        self.do_table_structure = None
        self.do_cell_matching = None
        self.do_picture_classification = None
        self.do_formula_enrichment = None


class FakeEasyOcrOptions:
    created = 0

    def __init__(self):
        FakeEasyOcrOptions.created += 1


class FakeTableStructureOptions:
    latest = None

    def __init__(self, do_cell_matching, mode):
        self.do_cell_matching = do_cell_matching
        self.mode = mode
        FakeTableStructureOptions.latest = self


class FakePdfFormatOption:
    latest = None

    def __init__(self, pipeline_options):
        self.pipeline_options = pipeline_options
        FakePdfFormatOption.latest = self


class FakeDocumentConverter:
    latest = None

    def __init__(self, format_options):
        self.format_options = format_options
        self.converted = []
        FakeDocumentConverter.latest = self

    def convert(self, path):
        self.converted.append(path)
        return SimpleNamespace(path=path)


def _patch_docling_classes(monkeypatch):
    def fake_loader(self):
        return (
            FakeInputFormat,
            FakePdfPipelineOptions,
            FakeEasyOcrOptions,
            FakeTableStructureOptions,
            FakeTableFormerMode,
            FakeDocumentConverter,
            FakePdfFormatOption,
        )

    FakePdfPipelineOptions.latest = None
    FakeTableStructureOptions.latest = None
    FakePdfFormatOption.latest = None
    FakeDocumentConverter.latest = None
    FakeEasyOcrOptions.created = 0
    monkeypatch.setattr(DoclingPdfAdapter, "_load_docling_classes", fake_loader)


def test_plan_ayeva_code_heavy():
    plan = _digital_plan(needs_code_enrichment=True, code_enrichment_reason="code density")

    assert plan.needs_code_enrichment is True
    assert plan.do_formula_enrichment is False
    assert plan.do_ocr is False
    assert plan.has_encoding_corruption is False


def test_plan_combat_encoding_corrupt():
    plan = _digital_plan(
        has_encoding_corruption=True,
        needs_code_enrichment=False,
        document_domain="aviation",
    )

    assert plan.has_encoding_corruption is True
    assert plan.needs_code_enrichment is False


def test_plan_fluent_control():
    plan = _digital_plan(needs_code_enrichment=False)

    assert plan.needs_code_enrichment is False
    assert plan.generate_picture_images is True
    assert plan.sort_by_reading_order is True


def test_plan_scanned_degraded():
    plan = build_pdf_conversion_plan(
        enable_ocr=True,
        document_modality="scanned_degraded",
        profile_type="scanned_degraded",
    )

    assert plan.do_ocr is True
    assert plan.do_picture_classification is False


def test_plan_magazine_prose():
    plan = _digital_plan(profile_type="digital_magazine", document_domain="magazine")

    assert plan.do_picture_classification is True
    assert plan.needs_code_enrichment is False
    assert plan.generate_table_images is False


def test_plan_encoding_alone_no_code():
    plan = _digital_plan(has_encoding_corruption=True)

    assert plan.has_encoding_corruption is True
    assert plan.needs_code_enrichment is False


def test_plan_force_table_vlm_enables_table_images():
    plan = _digital_plan(force_table_vlm=True)

    assert plan.force_table_vlm is True
    assert plan.generate_table_images is True


def test_plan_default_table_images_false():
    assert _digital_plan().generate_table_images is False


def test_to_intelligence_metadata_has_all_keys():
    metadata = _digital_plan(
        needs_code_enrichment=True,
        has_encoding_corruption=True,
        has_flat_text_corruption=True,
        geometry_error_rate=0.12,
    ).to_intelligence_metadata()

    assert set(metadata) == set(INTELLIGENCE_METADATA_KEYS)
    assert isinstance(metadata["needs_code_enrichment"], bool)
    assert isinstance(metadata["geometry_error_rate"], float)
    assert isinstance(metadata["total_pages"], int)


def test_chunk_factory_metadata_has_only_safe_keys():
    metadata = _digital_plan().chunk_factory_metadata()

    assert set(metadata) == set(CHUNK_FACTORY_METADATA_KEYS)


def test_chunk_factory_metadata_excludes_structural_flags():
    metadata = _digital_plan(
        needs_code_enrichment=True,
        has_encoding_corruption=True,
        total_pages=10,
    ).chunk_factory_metadata()

    assert "has_encoding_corruption" not in metadata
    assert "needs_code_enrichment" not in metadata
    assert "total_pages" not in metadata


def test_adapter_sets_all_pipeline_options(monkeypatch):
    _patch_docling_classes(monkeypatch)
    plan = _digital_plan(
        enable_ocr=True,
        force_table_vlm=True,
        needs_code_enrichment=True,
        code_enrichment_reason="unit-test",
    )

    DoclingPdfAdapter(plan).get_converter()
    options = FakePdfPipelineOptions.latest

    assert options.images_scale == 2.0
    assert options.generate_page_images is True
    assert options.generate_picture_images is True
    assert options.generate_table_images is True
    assert options.sort_by_reading_order is True
    assert options.do_table_structure is True
    assert options.do_cell_matching is False
    assert options.do_ocr is True
    assert options.do_code_enrichment is True
    assert options.do_formula_enrichment is False
    assert options.picture_description_options.classification_deny == [
        "full_page_image",
        "page_thumbnail",
    ]


def test_adapter_caches_converter(monkeypatch):
    _patch_docling_classes(monkeypatch)
    adapter = DoclingPdfAdapter(_digital_plan())

    assert adapter.get_converter() is adapter.get_converter()


def test_adapter_ocr_disabled(monkeypatch):
    _patch_docling_classes(monkeypatch)
    DoclingPdfAdapter(_digital_plan(enable_ocr=False)).get_converter()
    options = FakePdfPipelineOptions.latest

    assert options.do_ocr is False
    assert not hasattr(options, "ocr_options")
    assert FakeEasyOcrOptions.created == 0


def test_adapter_ocr_enabled_uses_easyocr(monkeypatch):
    _patch_docling_classes(monkeypatch)
    DoclingPdfAdapter(_digital_plan(enable_ocr=True, ocr_engine="tesseract")).get_converter()

    assert FakePdfPipelineOptions.latest.do_ocr is True
    assert isinstance(FakePdfPipelineOptions.latest.ocr_options, FakeEasyOcrOptions)


def test_adapter_code_enrichment_on(monkeypatch):
    _patch_docling_classes(monkeypatch)
    DoclingPdfAdapter(_digital_plan(needs_code_enrichment=True)).get_converter()

    assert FakePdfPipelineOptions.latest.do_code_enrichment is True
    assert FakePdfPipelineOptions.latest.do_formula_enrichment is False


def test_adapter_picture_classification_off_for_scanned(monkeypatch):
    _patch_docling_classes(monkeypatch)
    plan = build_pdf_conversion_plan(
        enable_ocr=True,
        document_modality="scanned_clean",
        profile_type="scanned",
    )

    DoclingPdfAdapter(plan).get_converter()

    assert FakePdfPipelineOptions.latest.do_picture_classification is False
    # When picture classification is off, classification_deny stays at its default empty list.
    assert FakePdfPipelineOptions.latest.picture_description_options.classification_deny == []


def test_adapter_allow_page_level_visuals(monkeypatch):
    """When allow_page_level_visuals=True, full_page_image is removed from deny list."""
    _patch_docling_classes(monkeypatch)
    plan = build_pdf_conversion_plan(
        enable_ocr=False,
        document_modality="native_digital",
        profile_type="digital_magazine",
        image_density=3.0,  # triggers image_heavy_magazine route
    )

    assert plan.extraction_route == "image_heavy_magazine"
    assert plan.allow_page_level_visuals is True

    DoclingPdfAdapter(plan).get_converter()

    assert FakePdfPipelineOptions.latest.do_picture_classification is True
    deny = FakePdfPipelineOptions.latest.picture_description_options.classification_deny
    assert "full_page_image" not in deny
    assert "page_thumbnail" in deny


def test_adapter_default_deny_list(monkeypatch):
    """Default plan keeps both full_page_image and page_thumbnail in deny list."""
    _patch_docling_classes(monkeypatch)
    plan = build_pdf_conversion_plan(
        enable_ocr=False,
        document_modality="native_digital",
        profile_type="technical_manual",
    )

    DoclingPdfAdapter(plan).get_converter()

    assert FakePdfPipelineOptions.latest.do_picture_classification is True
    deny = FakePdfPipelineOptions.latest.picture_description_options.classification_deny
    assert "full_page_image" in deny
    assert "page_thumbnail" in deny


def test_adapter_table_structure_and_former_mode(monkeypatch):
    _patch_docling_classes(monkeypatch)
    DoclingPdfAdapter(_digital_plan()).get_converter()

    assert FakeTableStructureOptions.latest.do_cell_matching is False
    assert FakeTableStructureOptions.latest.mode == FakeTableFormerMode.ACCURATE


def test_pdf_engine_uses_adapter(monkeypatch):
    from mmrag_v2.engines.pdf_engine import PDFEngine

    captured = {}

    def fake_get_converter(self):
        captured["plan"] = self.plan
        return SimpleNamespace(convert=lambda _path: None)

    monkeypatch.setattr(DoclingPdfAdapter, "get_converter", fake_get_converter)

    plan = _digital_plan(enable_ocr=False)
    engine = PDFEngine(conversion_plan=plan)
    engine.initialize()

    assert engine._converter is not None
    assert captured["plan"] is plan


def test_pdf_engine_default_plan_has_table_structure(monkeypatch):
    from mmrag_v2.engines.pdf_engine import PDFEngine

    captured = {}

    def fake_get_converter(self):
        captured["plan"] = self.plan
        return SimpleNamespace(convert=lambda _path: None)

    monkeypatch.setattr(DoclingPdfAdapter, "get_converter", fake_get_converter)

    PDFEngine(enable_ocr=False).initialize()

    assert captured["plan"].do_table_structure is True
    assert captured["plan"].sort_by_reading_order is True
    assert captured["plan"].do_cell_matching is False


def test_processor_with_plan_uses_adapter(monkeypatch, tmp_path):
    from mmrag_v2.processor import V2DocumentProcessor

    captured = {}

    def fake_get_converter(self):
        captured["plan"] = self.plan
        return SimpleNamespace(convert=lambda _path: None)

    monkeypatch.setattr(DoclingPdfAdapter, "get_converter", fake_get_converter)
    plan = _digital_plan(has_encoding_corruption=True, needs_code_enrichment=True)

    proc = V2DocumentProcessor(
        output_dir=str(tmp_path),
        vision_provider="none",
        conversion_plan=plan,
    )

    assert proc._converter is not None
    assert captured["plan"] is plan


def test_processor_plan_chunk_factory_metadata_is_safe(monkeypatch, tmp_path):
    from mmrag_v2.processor import V2DocumentProcessor

    monkeypatch.setattr(
        DoclingPdfAdapter,
        "get_converter",
        lambda self: SimpleNamespace(convert=lambda _path: None),
    )
    plan = _digital_plan(has_encoding_corruption=True, needs_code_enrichment=True)

    proc = V2DocumentProcessor(
        output_dir=str(tmp_path),
        vision_provider="none",
        conversion_plan=plan,
    )

    assert set(proc._intelligence_metadata) == set(CHUNK_FACTORY_METADATA_KEYS)
    assert "has_encoding_corruption" not in proc._intelligence_metadata
    assert "needs_code_enrichment" not in proc._intelligence_metadata


def test_processor_plan_structural_flags_on_instance(monkeypatch, tmp_path):
    from mmrag_v2.processor import V2DocumentProcessor

    monkeypatch.setattr(
        DoclingPdfAdapter,
        "get_converter",
        lambda self: SimpleNamespace(convert=lambda _path: None),
    )
    plan = _digital_plan(
        has_encoding_corruption=True,
        has_flat_text_corruption=True,
        geometry_error_rate=0.2,
        needs_code_enrichment=True,
    )

    proc = V2DocumentProcessor(
        output_dir=str(tmp_path),
        vision_provider="none",
        conversion_plan=plan,
    )

    assert proc.has_encoding_corruption is True
    assert proc.has_flat_text_corruption is True
    assert proc.geometry_error_rate == 0.2
    assert proc.needs_code_enrichment is True


def test_batch_with_plan_uses_adapter(monkeypatch, tmp_path):
    from mmrag_v2.batch_processor import BatchProcessor

    converted = {}

    class FakeAdapter:
        def __init__(self, plan):
            self.plan = plan

        def get_converter(self):
            converted["plan"] = self.plan
            return SimpleNamespace()

        def convert(self, path):
            converted["path"] = path
            return SimpleNamespace(document=SimpleNamespace(iterate_items=lambda: []))

    monkeypatch.setattr("mmrag_v2.batch_processor.DoclingPdfAdapter", FakeAdapter)
    proc = BatchProcessor(output_dir=str(tmp_path))
    plan = _digital_plan()
    proc.set_conversion_plan(plan)

    batch_pdf = tmp_path / "batch.pdf"
    batch_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    proc._extract_docling_layout_elements(batch_pdf, page_offset=0)

    assert converted["plan"] is plan
    assert converted["path"] == batch_pdf


def test_batch_plan_to_processor_all_flags_bridge(monkeypatch, tmp_path):
    import mmrag_v2.processor as processor_module
    from mmrag_v2.batch_processor import BatchProcessor
    from mmrag_v2.utils.pdf_splitter import BatchInfo, SplitResult

    captured = {}

    class FakeProcessor:
        def __init__(self, *args, **kwargs):
            captured["intelligence_metadata"] = kwargs.get("intelligence_metadata")
            captured["conversion_plan"] = kwargs.get("conversion_plan")

        def process_document(self, _path):
            return iter(())

        def get_final_state(self):
            return None

        def cleanup(self):
            return None

    monkeypatch.setattr(processor_module, "V2DocumentProcessor", FakeProcessor)
    batch_pdf = tmp_path / "batch.pdf"
    batch_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    plan = _digital_plan(
        needs_code_enrichment=True,
        has_encoding_corruption=True,
        has_flat_text_corruption=True,
        geometry_error_rate=0.42,
        total_pages=321,
        image_density=0.33,
        avg_text_per_page=123.4,
    )

    proc = BatchProcessor(output_dir=str(tmp_path / "out"))
    proc.set_conversion_plan(plan)
    batch_info = BatchInfo(
        batch_index=0,
        batch_path=batch_pdf,
        start_page=1,
        end_page=1,
        page_count=1,
        page_offset=0,
    )
    split_result = SplitResult(
        original_path=batch_pdf,
        original_hash="abc",
        total_pages=1,
        batch_count=1,
        batches=[batch_info],
        temp_dir=tmp_path,
    )

    proc._process_single_batch(batch_info, split_result, "source.pdf")
    metadata = captured["intelligence_metadata"]

    assert metadata["needs_code_enrichment"] is True
    assert metadata["has_encoding_corruption"] is True
    assert metadata["has_flat_text_corruption"] is True
    assert metadata["geometry_error_rate"] == 0.42
    assert metadata["total_pages"] == 321
    assert metadata["image_density"] == 0.33
    assert metadata["avg_text_per_page"] == 123.4
    assert captured["conversion_plan"].needs_code_enrichment is True


def test_batch_plan_chunk_factory_metadata_is_safe(tmp_path):
    from mmrag_v2.batch_processor import BatchProcessor

    proc = BatchProcessor(output_dir=str(tmp_path))
    proc.set_conversion_plan(
        _digital_plan(has_encoding_corruption=True, needs_code_enrichment=True)
    )

    assert set(proc._intelligence_metadata) == set(CHUNK_FACTORY_METADATA_KEYS)
    assert "has_encoding_corruption" not in proc._intelligence_metadata
    assert "needs_code_enrichment" not in proc._intelligence_metadata


def test_force_table_vlm_process_command_plan():
    from mmrag_v2.cli import _build_conversion_plan_from_metadata

    plan = _build_conversion_plan_from_metadata(
        intelligence_metadata=_digital_plan().to_intelligence_metadata(),
        enable_ocr=False,
        ocr_engine="tesseract",
        force_table_vlm=True,
        needs_code_enrichment=False,
        code_enrichment_reason="",
        code_enrichment_score=0.0,
    )

    assert plan.generate_table_images is True


def test_batch_command_force_table_vlm_false():
    from mmrag_v2.cli import _build_conversion_plan_from_metadata

    plan = _build_conversion_plan_from_metadata(
        intelligence_metadata=_digital_plan().to_intelligence_metadata(),
        enable_ocr=False,
        ocr_engine="tesseract",
        force_table_vlm=False,
        needs_code_enrichment=False,
        code_enrichment_reason="",
        code_enrichment_score=0.0,
    )

    assert plan.generate_table_images is False


def _production_python_files():
    root = Path(__file__).resolve().parents[1] / "src" / "mmrag_v2"
    return [path for path in root.rglob("*.py") if "__pycache__" not in path.parts]


def test_no_pipeline_options_construction_outside_adapter():
    root = Path(__file__).resolve().parents[1]
    allowed = root / "src" / "mmrag_v2" / "engines" / "docling_adapter.py"
    violations = []

    for path in _production_python_files():
        if path == allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if "PdfPipelineOptions(" in text or "DocumentConverter(" in text:
            violations.append(str(path.relative_to(root)))

    assert violations == []


def test_no_production_docling_imports_outside_adapter():
    root = Path(__file__).resolve().parents[1]
    allowed = root / "src" / "mmrag_v2" / "engines" / "docling_adapter.py"
    violations = []

    for path in _production_python_files():
        if path == allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module in {
                "docling.datamodel.pipeline_options",
                "docling.document_converter",
            }:
                violations.append(f"{path.relative_to(root)}:{node.lineno}")

    assert violations == []


# ============================================================================
# PLAN_V2.8 §2 — Adapter-invocation static guard
# Promotes the v2.7 §5 rule (all Docling extraction goes through
# DoclingPdfAdapter) from "construction guarded" to "invocation guarded".
# ============================================================================

# Cached converter attribute names that MUST NOT have `.convert(...)` called
# on them outside the adapter. These are attributes that hold a reference to
# a `DocumentConverter` returned by `DoclingPdfAdapter.get_converter()`;
# calling `.convert()` directly bypasses the adapter's post-Docling sanity
# stages (reading-order y-sort, drop-cap promotion, label-leak filter, etc.).
_CACHED_CONVERTER_ATTRS = {"_converter", "_docling_converter"}


def _find_raw_converter_invocations(source: str):
    """Yield (lineno, attr) for each `self.<cached>.convert(...)` call."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Looking for: Call(func=Attribute(value=Attribute(value=Name('self'),
        #                                                  attr=<cached>),
        #                                  attr='convert'))
        if not (isinstance(func, ast.Attribute) and func.attr == "convert"):
            continue
        receiver = func.value
        if not isinstance(receiver, ast.Attribute):
            continue
        if receiver.attr not in _CACHED_CONVERTER_ATTRS:
            continue
        if not (isinstance(receiver.value, ast.Name) and receiver.value.id == "self"):
            continue
        yield (node.lineno, receiver.attr)


def test_no_raw_converter_invocation_outside_adapter():
    """No production file may call `.convert()` on a cached Docling converter.

    Doing so bypasses the adapter's post-Docling sanity stages (the failure
    mode that put `processor.py:2072` and `pdf_engine.py:206` on the v2.8
    plan). Adapter routing (`self._adapter.convert(...)`) is the only
    sanctioned invocation path.
    """
    root = Path(__file__).resolve().parents[1]
    allowed = root / "src" / "mmrag_v2" / "engines" / "docling_adapter.py"
    violations = []

    for path in _production_python_files():
        if path == allowed:
            continue
        source = path.read_text(encoding="utf-8")
        for lineno, attr in _find_raw_converter_invocations(source):
            violations.append(f"{path.relative_to(root)}:{lineno} (self.{attr}.convert)")

    assert violations == [], (
        "Found raw cached-converter invocations outside the adapter:\n  "
        + "\n  ".join(violations)
    )


def test_guard_fires_on_synthetic_bypass():
    """Positive-case: the AST walker must flag the bypass pattern."""
    bad = "def f(self, p):\n    return self._converter.convert(p)\n"
    findings = list(_find_raw_converter_invocations(bad))
    assert findings == [(2, "_converter")]


def test_guard_fires_on_synthetic_docling_converter_alias():
    """Positive-case: also catches the BatchProcessor's `_docling_converter` alias."""
    bad = "def f(self, p):\n    return self._docling_converter.convert(p)\n"
    findings = list(_find_raw_converter_invocations(bad))
    assert findings == [(2, "_docling_converter")]


def test_guard_does_not_fire_on_adapter_routing():
    """Negative-case: legitimate adapter calls must NOT trigger the guard."""
    good = "def f(self, p):\n    return self._adapter.convert(p)\n"
    findings = list(_find_raw_converter_invocations(good))
    assert findings == []


def test_guard_does_not_fire_on_unrelated_method_calls():
    """Negative-case: cleanup-style calls (`._converter.cleanup()`) are fine."""
    good = (
        "def f(self):\n"
        "    if self._converter is not None:\n"
        "        self._converter.cleanup()\n"
    )
    findings = list(_find_raw_converter_invocations(good))
    assert findings == []


# ============================================================================
# Milestone 1: Extraction route controls
# ============================================================================


def test_plan_extraction_route_defaults():
    """Standard digital plan defaults to extraction_route='native_digital'."""
    plan = _digital_plan()
    assert plan.extraction_route == "native_digital"
    assert plan.hybrid_chunker_enabled is True
    assert plan.max_chunker_input_chars == 500_000
    assert plan.max_chunker_per_element_chars == 100_000
    assert plan.allow_page_level_visuals is False
    assert plan.asset_validation_policy == "drop"
    assert plan.corruption_recovery_policy == "quarantine"
    assert plan.drop_blank_assets is True
    assert plan.quarantine_corrupted_chunks is True


def test_plan_scanned_modality_sets_scanned_book_route():
    """Scanned modality automatically selects scanned_book extraction route."""
    plan = build_pdf_conversion_plan(document_modality="scanned_clean")
    assert plan.extraction_route == "scanned_book"


def test_plan_explicit_route_override():
    """Explicit extraction_route overrides auto-detection."""
    plan = build_pdf_conversion_plan(
        document_modality="native_digital",
        extraction_route="scanned_degraded",
    )
    assert plan.extraction_route == "scanned_degraded"


def test_batch_plan_route_fields_bridge(tmp_path):
    """Bridge: extraction route fields flow from plan → BatchProcessor."""
    from mmrag_v2.batch_processor import BatchProcessor

    plan = build_pdf_conversion_plan(
        extraction_route="scanned_book",
        max_chunker_input_chars=200_000,
        max_chunker_per_element_chars=42_000,
    )
    proc = BatchProcessor(output_dir=str(tmp_path))
    proc.set_conversion_plan(plan)

    assert proc._extraction_route == "scanned_book"
    assert proc._max_chunker_input_chars == 200_000
    assert proc._max_chunker_per_element_chars == 42_000
    assert proc._drop_blank_assets is True
    assert proc._quarantine_corrupted_chunks is True


# ============================================================================
# Milestone 2: Plan Control Plane — new fields
# ============================================================================


def test_plan_scanned_disables_hybrid_chunker():
    """Scanned modality sets hybrid_chunker_enabled=False."""
    plan = build_pdf_conversion_plan(document_modality="scanned_clean")
    assert plan.extraction_route == "scanned_book"
    assert plan.hybrid_chunker_enabled is False


def test_plan_native_digital_enables_hybrid_chunker():
    """Native digital documents keep hybrid_chunker_enabled=True."""
    plan = _digital_plan()
    assert plan.hybrid_chunker_enabled is True


def test_plan_image_heavy_magazine_route():
    """High image_density on digital_magazine triggers image_heavy_magazine route."""
    plan = build_pdf_conversion_plan(
        document_modality="native_digital",
        profile_type="digital_magazine",
        image_density=3.0,  # above 2.0 threshold
    )
    assert plan.extraction_route == "image_heavy_magazine"
    assert plan.allow_page_level_visuals is True


def test_plan_low_density_magazine_stays_native():
    """Magazine below image_density threshold stays native_digital."""
    plan = _digital_plan(image_density=0.5)
    assert plan.extraction_route == "native_digital"
    assert plan.allow_page_level_visuals is False


def test_plan_technical_manual_route():
    """technical_manual profile without scanning selects technical_manual route."""
    plan = build_pdf_conversion_plan(
        document_modality="native_digital",
        profile_type="technical_manual",
    )
    assert plan.extraction_route == "technical_manual"


def test_plan_asset_validation_policy_defaults():
    """Default asset_validation_policy is 'drop'."""
    plan = _digital_plan()
    assert plan.asset_validation_policy == "drop"


def test_plan_corruption_recovery_policy_defaults():
    """Default corruption_recovery_policy is 'quarantine'."""
    plan = _digital_plan()
    assert plan.corruption_recovery_policy == "quarantine"


def test_batch_plan_new_fields_bridge(tmp_path):
    """Bridge: new policy fields flow from plan → BatchProcessor."""
    from mmrag_v2.batch_processor import BatchProcessor

    plan = build_pdf_conversion_plan(
        extraction_route="image_heavy_magazine",
        hybrid_chunker_enabled=False,
        allow_page_level_visuals=True,
    )
    proc = BatchProcessor(output_dir=str(tmp_path))
    proc.set_conversion_plan(plan)

    assert proc._extraction_route == "image_heavy_magazine"
    assert proc._hybrid_chunker_enabled is False
    assert proc._allow_page_level_visuals is True


def test_processor_plan_new_fields_bridge(monkeypatch, tmp_path):
    """Bridge: new policy fields flow from plan → V2DocumentProcessor."""
    import mmrag_v2.processor as processor_module

    _patch_docling_classes(monkeypatch)

    plan = build_pdf_conversion_plan(
        extraction_route="technical_manual",
        hybrid_chunker_enabled=True,
        allow_page_level_visuals=False,
    )

    proc = processor_module.V2DocumentProcessor(
        output_dir=str(tmp_path),
        conversion_plan=plan,
    )

    assert proc._extraction_route == "technical_manual"
    assert proc._hybrid_chunker_enabled is True
    assert proc._allow_page_level_visuals is False


def test_processor_plan_route_fields_bridge(monkeypatch, tmp_path):
    """Bridge: extraction route fields flow from plan → V2DocumentProcessor."""
    import mmrag_v2.processor as processor_module

    _patch_docling_classes(monkeypatch)

    plan = build_pdf_conversion_plan(
        extraction_route="scanned_degraded",
        max_chunker_input_chars=100_000,
        max_chunker_per_element_chars=25_000,
    )

    proc = processor_module.V2DocumentProcessor(
        output_dir=str(tmp_path),
        conversion_plan=plan,
    )

    assert proc._extraction_route == "scanned_degraded"
    assert proc._max_chunker_input_chars == 100_000
    assert proc._max_chunker_per_element_chars == 25_000
    assert proc._drop_blank_assets is True
    assert proc._quarantine_corrupted_chunks is True


# ============================================================================
# Milestone 2 Close-out: Derived property bridges + __post_init__ validation
# ============================================================================


def test_plan_asset_validation_policy_derives_drop_blank_assets():
    """asset_validation_policy='keep' → drop_blank_assets is False."""
    plan = PdfConversionPlan(asset_validation_policy="keep")
    assert plan.drop_blank_assets is False

    plan2 = PdfConversionPlan(asset_validation_policy="drop")
    assert plan2.drop_blank_assets is True

    plan3 = PdfConversionPlan(asset_validation_policy="quarantine")
    assert plan3.drop_blank_assets is False


def test_plan_corruption_recovery_policy_derives_quarantine():
    """corruption_recovery_policy maps to quarantine_corrupted_chunks correctly."""
    plan_q = PdfConversionPlan(corruption_recovery_policy="quarantine")
    assert plan_q.quarantine_corrupted_chunks is True

    plan_k = PdfConversionPlan(corruption_recovery_policy="keep")
    assert plan_k.quarantine_corrupted_chunks is False

    plan_r = PdfConversionPlan(corruption_recovery_policy="recover")
    assert plan_r.quarantine_corrupted_chunks is True  # recover still gates


def test_plan_post_init_rejects_invalid_asset_policy():
    """Invalid asset_validation_policy raises ValueError."""
    with pytest.raises(ValueError):
        PdfConversionPlan(asset_validation_policy="invalid")  # type: ignore[arg-type]


def test_plan_post_init_rejects_invalid_recovery_policy():
    """Invalid corruption_recovery_policy raises ValueError."""
    with pytest.raises(ValueError):
        PdfConversionPlan(corruption_recovery_policy="invalid")  # type: ignore[arg-type]


def test_batch_plan_policy_derived_bridge(tmp_path):
    """Bridge: asset_validation_policy='keep' propagates through BatchProcessor."""
    from mmrag_v2.batch_processor import BatchProcessor

    plan = PdfConversionPlan(
        asset_validation_policy="keep",
        corruption_recovery_policy="keep",
    )
    proc = BatchProcessor(output_dir=str(tmp_path))
    proc.set_conversion_plan(plan)

    assert proc._drop_blank_assets is False  # derived from "keep"
    assert proc._quarantine_corrupted_chunks is False  # derived from "keep"


def test_processor_plan_policy_derived_bridge(monkeypatch, tmp_path):
    """Bridge: policy fields propagate to V2DocumentProcessor legacy attrs."""
    import mmrag_v2.processor as processor_module

    _patch_docling_classes(monkeypatch)

    plan = PdfConversionPlan(
        asset_validation_policy="quarantine",
        corruption_recovery_policy="recover",
    )

    proc = processor_module.V2DocumentProcessor(
        output_dir=str(tmp_path),
        conversion_plan=plan,
    )

    assert proc._drop_blank_assets is False   # quarantine → not drop
    assert proc._quarantine_corrupted_chunks is True  # recover → still quarantines


def test_all_typed_policy_fields_round_trip_full_chain(monkeypatch, tmp_path):
    """Drift insurance: every typed policy field must reach every downstream object.

    Builds a plan with non-default values for every Milestone 2 typed policy
    field, plumbs it through BatchProcessor.set_conversion_plan, runs a
    synthetic single-batch process, and asserts that BatchProcessor,
    V2DocumentProcessor, and DoclingPdfAdapter all observe the values
    unchanged. Fails loudly the day someone adds a new typed field but forgets
    to plumb it through one of the boundaries.
    """
    import mmrag_v2.processor as processor_module
    from mmrag_v2.batch_processor import BatchProcessor
    from mmrag_v2.utils.pdf_splitter import BatchInfo, SplitResult

    captured = {}

    class FakeAdapter:
        def __init__(self, plan):
            captured["adapter_plan"] = plan
            self.plan = plan

        def get_converter(self):
            return SimpleNamespace()

        def convert(self, _path):
            return SimpleNamespace(document=SimpleNamespace(iterate_items=lambda: []))

    class FakeProcessor:
        def __init__(self, *args, **kwargs):
            captured["processor_kwargs"] = kwargs
            captured["processor_plan"] = kwargs.get("conversion_plan")
            captured["processor_plan_attrs"] = {
                "_extraction_route": kwargs["conversion_plan"].extraction_route,
                "_hybrid_chunker_enabled": kwargs["conversion_plan"].hybrid_chunker_enabled,
                "_max_chunker_input_chars": kwargs["conversion_plan"].max_chunker_input_chars,
                "_max_chunker_per_element_chars": (
                    kwargs["conversion_plan"].max_chunker_per_element_chars
                ),
                "_allow_page_level_visuals": (
                    kwargs["conversion_plan"].allow_page_level_visuals
                ),
                "_drop_blank_assets": kwargs["conversion_plan"].drop_blank_assets,
                "_quarantine_corrupted_chunks": (
                    kwargs["conversion_plan"].quarantine_corrupted_chunks
                ),
            }

        def process_document(self, _path):
            return iter(())

        def get_final_state(self):
            return None

        def cleanup(self):
            return None

    monkeypatch.setattr("mmrag_v2.batch_processor.DoclingPdfAdapter", FakeAdapter)
    monkeypatch.setattr(processor_module, "V2DocumentProcessor", FakeProcessor)

    # Non-default values for every typed policy field.
    plan = PdfConversionPlan(
        extraction_route="technical_manual",
        hybrid_chunker_enabled=False,
        max_chunker_input_chars=321_000,
        max_chunker_per_element_chars=42_000,
        allow_page_level_visuals=True,
        asset_validation_policy="quarantine",
        corruption_recovery_policy="recover",
    )

    proc = BatchProcessor(output_dir=str(tmp_path / "out"))
    proc.set_conversion_plan(plan)

    # BatchProcessor sees every typed value unchanged.
    assert proc._extraction_route == "technical_manual"
    assert proc._hybrid_chunker_enabled is False
    assert proc._max_chunker_input_chars == 321_000
    assert proc._max_chunker_per_element_chars == 42_000
    assert proc._allow_page_level_visuals is True
    assert proc._drop_blank_assets is False  # quarantine → not drop
    assert proc._quarantine_corrupted_chunks is True  # recover → still gates

    # Run one synthetic batch through the bridge.
    batch_pdf = tmp_path / "batch.pdf"
    batch_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    batch_info = BatchInfo(
        batch_index=0,
        batch_path=batch_pdf,
        start_page=1,
        end_page=1,
        page_count=1,
        page_offset=0,
    )
    split_result = SplitResult(
        original_path=batch_pdf,
        original_hash="abc",
        total_pages=1,
        batch_count=1,
        batches=[batch_info],
        temp_dir=tmp_path,
    )
    proc._process_single_batch(batch_info, split_result, "source.pdf")

    # V2DocumentProcessor receives a plan with every typed value unchanged.
    p_attrs = captured["processor_plan_attrs"]
    assert p_attrs["_extraction_route"] == "technical_manual"
    assert p_attrs["_hybrid_chunker_enabled"] is False
    assert p_attrs["_max_chunker_input_chars"] == 321_000
    assert p_attrs["_max_chunker_per_element_chars"] == 42_000
    assert p_attrs["_allow_page_level_visuals"] is True
    assert p_attrs["_drop_blank_assets"] is False
    assert p_attrs["_quarantine_corrupted_chunks"] is True

    # DoclingPdfAdapter receives a plan whose policy strings are unchanged.
    adapter_plan = captured["adapter_plan"]
    assert adapter_plan.asset_validation_policy == "quarantine"
    assert adapter_plan.corruption_recovery_policy == "recover"
    assert adapter_plan.extraction_route == "technical_manual"
    assert adapter_plan.hybrid_chunker_enabled is False
    assert adapter_plan.max_chunker_input_chars == 321_000
    assert adapter_plan.max_chunker_per_element_chars == 42_000
    assert adapter_plan.allow_page_level_visuals is True
