"""Unit tests for the V3.0 format-agnostic ConversionPlan parent class.

Charter: docs/ARCHITECTURE_V3_DRAFT_0.5.md §3.2.

The Phase A task A1 will refactor `PdfConversionPlan` to inherit from
this parent. The foundation session ships only the parent class plus
its contract; PdfConversionPlan stays untouched (still the v2.16
production construction site).
"""

from __future__ import annotations

import pytest

from mmrag_v2.universal.conversion_plan import (
    DEFAULT_RENDER_DPI,
    RENDER_DPI_MAX,
    RENDER_DPI_MIN,
    ConversionPlan,
)


def _make_plan(**overrides):
    defaults = dict(
        source_path="/x.pdf",
        file_type="pdf",
        doc_id="abc123def456",
        profile_type="technical_manual",
        extraction_strategy="digital_native",
        reading_order_strategy="docling_native",
    )
    defaults.update(overrides)
    return ConversionPlan(**defaults)


class TestConversionPlanConstruction:
    def test_minimal_happy_path(self):
        plan = _make_plan()
        assert plan.file_type == "pdf"
        assert plan.render_dpi == DEFAULT_RENDER_DPI == 200
        assert plan.batch_size == 10
        assert plan.engine_options == {}
        assert plan.modality_flags == {}
        assert plan.lang_hint is None

    def test_engine_options_opaque_blob(self):
        """Charter §3.2: engine-specific Docling toggles ride in engine_options."""
        plan = _make_plan(
            engine_options={
                "do_code_enrichment": True,
                "do_picture_classification": False,
                "ocr_engine": "easyocr",
                "force_full_page_ocr": True,
            }
        )
        assert plan.engine_options["do_code_enrichment"] is True
        assert plan.engine_options["ocr_engine"] == "easyocr"


class TestRenderDpiValidation:
    """Charter §3.2 (Draft 0.5): render_dpi validation range [72, 600]."""

    def test_min_accepted(self):
        plan = _make_plan(render_dpi=RENDER_DPI_MIN)
        assert plan.render_dpi == 72

    def test_max_accepted(self):
        plan = _make_plan(render_dpi=RENDER_DPI_MAX)
        assert plan.render_dpi == 600

    def test_below_min_rejected(self):
        with pytest.raises(ValueError, match=r"render_dpi must be in \[72, 600\]"):
            _make_plan(render_dpi=71)

    def test_above_max_rejected(self):
        with pytest.raises(ValueError, match=r"render_dpi must be in \[72, 600\]"):
            _make_plan(render_dpi=601)

    def test_zero_rejected(self):
        with pytest.raises(ValueError, match="render_dpi"):
            _make_plan(render_dpi=0)

    def test_negative_rejected(self):
        with pytest.raises(ValueError, match="render_dpi"):
            _make_plan(render_dpi=-1)

    def test_typical_production_value(self):
        # 200 DPI is the C-spike measurement value per Charter §4.2.
        plan = _make_plan(render_dpi=200)
        assert plan.render_dpi == 200

    def test_higher_production_value(self):
        # 300 DPI is the second C-spike measurement value per Charter §4.2.
        plan = _make_plan(render_dpi=300)
        assert plan.render_dpi == 300


class TestBatchSizeValidation:
    def test_min_accepted(self):
        plan = _make_plan(batch_size=1)
        assert plan.batch_size == 1

    def test_default_accepted(self):
        # AGENTS.md §1.4 invariant: PDF batch size at <=10 pages.
        # Parent class does not enforce upper bound (would require
        # knowing the engine); PdfConversionPlan subclass enforces.
        plan = _make_plan(batch_size=10)
        assert plan.batch_size == 10

    def test_zero_rejected(self):
        with pytest.raises(ValueError, match="batch_size must be >=1"):
            _make_plan(batch_size=0)

    def test_negative_rejected(self):
        with pytest.raises(ValueError, match="batch_size must be >=1"):
            _make_plan(batch_size=-1)


class TestRequiredFields:
    def test_empty_file_type_rejected(self):
        with pytest.raises(ValueError, match="file_type"):
            _make_plan(file_type="")

    def test_empty_doc_id_rejected(self):
        with pytest.raises(ValueError, match="doc_id"):
            _make_plan(doc_id="")


class TestModalityFlags:
    def test_arbitrary_keys_accepted(self):
        # Charter §3.2: modality_flags is a Dict[str, bool] for diagnostic
        # signals like "is_scanned", "has_encoding_corruption", etc. The
        # parent class does not pin the key set.
        plan = _make_plan(
            modality_flags={
                "is_scanned": True,
                "has_encoding_corruption": False,
                "has_flat_text_corruption": False,
            }
        )
        assert plan.modality_flags["is_scanned"] is True
        assert plan.modality_flags["has_encoding_corruption"] is False


class TestLangHint:
    def test_default_none(self):
        plan = _make_plan()
        assert plan.lang_hint is None

    def test_iso639_value(self):
        plan = _make_plan(lang_hint="de")
        assert plan.lang_hint == "de"
