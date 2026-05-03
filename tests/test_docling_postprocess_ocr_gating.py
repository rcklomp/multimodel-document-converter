"""Phase 4 tests for OCR gating via the bitmap-area threshold.

Docling's default `bitmap_area_threshold` (0.05) triggers OCR whenever a
page has more than 5 percent of its area covered by bitmaps. On HARRY's
photographic cover pages this produces strings like
`"= 23555 AND Potter SIONE has the star of a Quidditch team"`. Raising
the threshold to 0.75 (default) and to 0.92 for `digital_literature` /
`digital_magazine` profiles keeps OCR off cover artwork while still
allowing OCR on legitimately scanned content.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from mmrag_v2.engines.docling_adapter import DoclingPdfAdapter
from mmrag_v2.engines.pdf_plan import PdfConversionPlan, build_pdf_conversion_plan


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_pdf_conversion_plan import (  # type: ignore  # noqa: E402
    FakePdfPipelineOptions,
    FakeEasyOcrOptions,
    FakeInputFormat,
    FakeTableStructureOptions,
    FakeTableFormerMode,
    FakeDocumentConverter,
    FakePdfFormatOption,
    _patch_docling_classes,
)


def test_pdf_conversion_plan_bitmap_area_threshold_default():
    """Default plan ships a 0.75 threshold (raised from Docling's 0.05)."""
    plan = PdfConversionPlan()
    assert plan.bitmap_area_threshold == 0.75


def test_plan_digital_literature_raises_bitmap_threshold():
    """`digital_literature` profile auto-raises the threshold to 0.92."""
    plan = build_pdf_conversion_plan(
        document_modality="native_digital",
        profile_type="digital_literature",
    )
    assert plan.bitmap_area_threshold == 0.92


def test_plan_digital_magazine_raises_bitmap_threshold():
    """`digital_magazine` profile also opts into the higher threshold."""
    plan = build_pdf_conversion_plan(
        document_modality="native_digital",
        profile_type="digital_magazine",
        image_density=3.0,
    )
    assert plan.bitmap_area_threshold == 0.92


def test_plan_technical_manual_keeps_default_threshold():
    """Technical manuals stay at the conservative 0.75 default."""
    plan = build_pdf_conversion_plan(
        document_modality="native_digital",
        profile_type="technical_manual",
    )
    assert plan.bitmap_area_threshold == 0.75


def test_plan_scanned_keeps_default_threshold():
    """Scanned documents legitimately need OCR; threshold stays at default."""
    plan = build_pdf_conversion_plan(
        document_modality="scanned_clean",
        profile_type="scanned",
    )
    assert plan.bitmap_area_threshold == 0.75


def test_plan_explicit_override_threshold_wins():
    """Explicit caller-supplied threshold overrides the auto-raise."""
    plan = build_pdf_conversion_plan(
        document_modality="native_digital",
        profile_type="digital_literature",
        bitmap_area_threshold=0.5,
    )
    assert plan.bitmap_area_threshold == 0.5


def test_adapter_passes_bitmap_threshold_to_pipeline_options(monkeypatch):
    """Bridge: adapter sets bitmap_area_threshold on the EasyOcrOptions."""
    _patch_docling_classes(monkeypatch)

    # Augment FakeEasyOcrOptions to record the bitmap threshold.
    captured: dict = {}

    class _RecordingEasyOcrOptions:
        def __init__(self):
            captured["instance"] = self
            self.bitmap_area_threshold = None

    # Patch only this test's loader to use the recording class.
    def _loader(self):
        return (
            FakeInputFormat,
            FakePdfPipelineOptions,
            _RecordingEasyOcrOptions,
            FakeTableStructureOptions,
            FakeTableFormerMode,
            FakeDocumentConverter,
            FakePdfFormatOption,
        )

    monkeypatch.setattr(DoclingPdfAdapter, "_load_docling_classes", _loader)

    plan = PdfConversionPlan(do_ocr=True, bitmap_area_threshold=0.92)
    DoclingPdfAdapter(plan).get_converter()

    assert captured["instance"].bitmap_area_threshold == pytest.approx(0.92)


def test_adapter_skips_threshold_when_ocr_disabled(monkeypatch):
    """No EasyOcrOptions are created (and no threshold set) when OCR is off."""
    _patch_docling_classes(monkeypatch)
    plan = PdfConversionPlan(do_ocr=False, bitmap_area_threshold=0.92)
    DoclingPdfAdapter(plan).get_converter()
    assert FakeEasyOcrOptions.created == 0


def test_processor_picks_up_bitmap_threshold_from_plan(monkeypatch, tmp_path):
    """Bridge: V2DocumentProcessor's adapter sees the threshold via the plan."""
    from mmrag_v2.processor import V2DocumentProcessor

    captured = {}

    def fake_get_converter(self):
        captured["plan"] = self.plan
        return SimpleNamespace(convert=lambda _path: None)

    monkeypatch.setattr(DoclingPdfAdapter, "get_converter", fake_get_converter)

    plan = PdfConversionPlan(bitmap_area_threshold=0.92)
    V2DocumentProcessor(
        output_dir=str(tmp_path),
        vision_provider="none",
        conversion_plan=plan,
    )

    assert captured["plan"].bitmap_area_threshold == pytest.approx(0.92)


def test_batch_processor_picks_up_bitmap_threshold_from_plan(tmp_path):
    """Bridge: BatchProcessor adapter sees the threshold via the plan."""
    from mmrag_v2.batch_processor import BatchProcessor

    plan = PdfConversionPlan(bitmap_area_threshold=0.92)
    proc = BatchProcessor(output_dir=str(tmp_path))
    proc.set_conversion_plan(plan)

    assert proc._adapter is not None
    assert proc._adapter.plan.bitmap_area_threshold == pytest.approx(0.92)
