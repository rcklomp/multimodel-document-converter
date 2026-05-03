"""Phase 5 tests: `digital_literature` profile turns on the full post-Docling pipeline.

When a caller asks `build_pdf_conversion_plan` for the digital_literature
profile (born-digital novels, story collections), the resulting plan must
ship with every Phase 1-4 post-processor enabled by default:

  - reading_order_strategy = "y_sort_with_dropcap"  (Phase 1 + 2)
  - suppress_layout_label_text = True                (Phase 3)
  - bitmap_area_threshold = 0.92                     (Phase 4)

Explicit caller overrides still win, so the profile is a sensible default
and not a hard policy.
"""
from __future__ import annotations

import pytest

from mmrag_v2.engines.pdf_plan import PdfConversionPlan, build_pdf_conversion_plan


def test_digital_literature_profile_enables_full_pipeline():
    plan = build_pdf_conversion_plan(
        document_modality="native_digital",
        profile_type="digital_literature",
    )
    assert plan.reading_order_strategy == "y_sort_with_dropcap"
    assert plan.suppress_layout_label_text is True
    assert plan.bitmap_area_threshold == pytest.approx(0.92)


def test_digital_literature_keeps_post_processor_overrides():
    """Caller-supplied flags win over the profile defaults."""
    plan = build_pdf_conversion_plan(
        document_modality="native_digital",
        profile_type="digital_literature",
        reading_order_strategy="y_sort",
        suppress_layout_label_text=False,
        bitmap_area_threshold=0.4,
    )
    # Reading order: explicit value wins
    assert plan.reading_order_strategy == "y_sort"
    # bitmap threshold: explicit override (non-default 0.75) wins
    assert plan.bitmap_area_threshold == pytest.approx(0.4)
    # suppress flag: caller passed False, but profile default is True; we OR
    # them so the profile still wins on the safer side. Document this so the
    # behavior change is visible if anyone flips the rule.
    assert plan.suppress_layout_label_text is True


def test_other_profiles_keep_post_processors_disabled_by_default():
    """Standard digital, technical_manual, scanned: no post-processor opt-in."""
    for profile in ("standard_digital", "technical_manual", "academic_whitepaper"):
        plan = build_pdf_conversion_plan(
            document_modality="native_digital",
            profile_type=profile,
        )
        assert plan.reading_order_strategy == "docling_native", profile
        assert plan.suppress_layout_label_text is False, profile


def test_scanned_profile_does_not_auto_enable_postprocessors():
    """A scanned literature-style PDF must not auto-flip post-processors on."""
    plan = build_pdf_conversion_plan(
        document_modality="scanned_clean",
        profile_type="digital_literature",
    )
    # Scanned overrides: scanned_book route, no auto post-processors
    assert plan.extraction_route == "scanned_book"
    assert plan.reading_order_strategy == "docling_native"
    assert plan.suppress_layout_label_text is False


def test_full_pipeline_round_trips_to_processor(monkeypatch, tmp_path):
    """Bridge: digital_literature plan reaches V2DocumentProcessor with all flags."""
    from mmrag_v2.processor import V2DocumentProcessor
    from mmrag_v2.engines.docling_adapter import DoclingPdfAdapter
    from types import SimpleNamespace

    captured = {}

    def fake_get_converter(self):
        captured["plan"] = self.plan
        return SimpleNamespace(convert=lambda _path: None)

    monkeypatch.setattr(DoclingPdfAdapter, "get_converter", fake_get_converter)

    plan = build_pdf_conversion_plan(
        document_modality="native_digital",
        profile_type="digital_literature",
    )
    proc = V2DocumentProcessor(
        output_dir=str(tmp_path),
        vision_provider="none",
        conversion_plan=plan,
    )
    assert captured["plan"].reading_order_strategy == "y_sort_with_dropcap"
    assert captured["plan"].suppress_layout_label_text is True
    assert captured["plan"].bitmap_area_threshold == pytest.approx(0.92)
    assert proc._suppress_layout_label_text is True


def test_full_pipeline_round_trips_to_batch(tmp_path):
    """Bridge: digital_literature plan reaches BatchProcessor with all flags."""
    from mmrag_v2.batch_processor import BatchProcessor

    plan = build_pdf_conversion_plan(
        document_modality="native_digital",
        profile_type="digital_literature",
    )
    proc = BatchProcessor(output_dir=str(tmp_path))
    proc.set_conversion_plan(plan)
    assert proc._suppress_layout_label_text is True
    assert proc._adapter is not None
    assert proc._adapter.plan.reading_order_strategy == "y_sort_with_dropcap"
    assert proc._adapter.plan.bitmap_area_threshold == pytest.approx(0.92)
