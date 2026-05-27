"""v2.17 Item #9 reopen — OCR engine dispatch in DoclingPdfAdapter.

Charter background: the CLI advertises `--ocr-engine
{tesseract|easyocr|doctr|ocrmac}` and the field flows through to
`PdfConversionPlan.ocr_engine`, but the adapter at
`engines/docling_adapter.py::get_converter` previously hardcoded
`EasyOcrOptions()` regardless of the plan field. Earthship multi-column
OCR damage (one of the documented v2.16 regressions) survived the v2.13
Phase 2 `force_full_page_ocr` fix in part because EasyOCR was always
the engine, even when `--ocr-engine tesseract` or `--ocr-engine ocrmac`
was requested at the CLI.

These tests pin the dispatch contract: `_build_ocr_options` returns the
right Docling OcrOptions class for each engine string, and falls back
to EasyOcr on unknown / missing-dep engines (so the adapter never
crashes at construction time).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mmrag_v2.engines.docling_adapter import DoclingPdfAdapter
from mmrag_v2.engines.pdf_plan import PdfConversionPlan


def _adapter_for(engine: str) -> DoclingPdfAdapter:
    plan = PdfConversionPlan(ocr_engine=engine, do_ocr=True)
    return DoclingPdfAdapter(plan)


def test_default_easyocr_returns_easyocr_options():
    from docling.datamodel.pipeline_options import EasyOcrOptions
    adapter = _adapter_for("easyocr")
    opts = adapter._build_ocr_options(EasyOcrOptions)
    assert isinstance(opts, EasyOcrOptions)


def test_empty_engine_string_falls_back_to_easyocr():
    """Defensive: empty/None engine field uses EasyOcr (backward-compat)."""
    from docling.datamodel.pipeline_options import EasyOcrOptions
    plan = PdfConversionPlan(ocr_engine="", do_ocr=True)
    adapter = DoclingPdfAdapter(plan)
    opts = adapter._build_ocr_options(EasyOcrOptions)
    assert isinstance(opts, EasyOcrOptions)


@pytest.mark.parametrize("engine,expected_class_name", [
    ("ocrmac", "OcrMacOptions"),
    ("tesseract", "TesseractCliOcrOptions"),
])
def test_alternative_engines_dispatch_to_right_class(engine, expected_class_name):
    """When an alternative engine is requested AND Docling provides the
    class, the adapter must return an instance of that class — NOT
    EasyOcr."""
    from docling.datamodel.pipeline_options import EasyOcrOptions
    adapter = _adapter_for(engine)
    opts = adapter._build_ocr_options(EasyOcrOptions)
    assert opts.__class__.__name__ == expected_class_name, (
        f"engine={engine!r} should produce {expected_class_name}, got "
        f"{opts.__class__.__name__}; the v2.17 dispatch is broken."
    )


def test_unknown_engine_falls_back_to_easyocr_with_warning(caplog):
    from docling.datamodel.pipeline_options import EasyOcrOptions
    plan = PdfConversionPlan(ocr_engine="bogus_engine_name", do_ocr=True)
    adapter = DoclingPdfAdapter(plan)
    with caplog.at_level("WARNING"):
        opts = adapter._build_ocr_options(EasyOcrOptions)
    assert isinstance(opts, EasyOcrOptions)
    assert any("bogus_engine_name" in r.message for r in caplog.records), (
        "Unknown engine names must log a warning so users notice the "
        "silent fallback to EasyOcr."
    )


def test_missing_dep_falls_back_quietly(caplog):
    """When the user requests an alternative engine but Docling can't
    import it (e.g. a future Docling release drops OcrMacOptions),
    the adapter must fall back to EasyOcr rather than crash at
    converter-construction time. Models the import failure via patch."""
    from docling.datamodel.pipeline_options import EasyOcrOptions
    plan = PdfConversionPlan(ocr_engine="ocrmac", do_ocr=True)
    adapter = DoclingPdfAdapter(plan)
    # Simulate import failure by deleting OcrMacOptions from the module's
    # namespace temporarily — the adapter's `from ... import` inside
    # _build_ocr_options will then raise ImportError.
    import docling.datamodel.pipeline_options as _po
    real = getattr(_po, "OcrMacOptions", None)
    try:
        if real is not None:
            del _po.OcrMacOptions
        with caplog.at_level("WARNING"):
            opts = adapter._build_ocr_options(EasyOcrOptions)
        assert isinstance(opts, EasyOcrOptions), (
            "When the requested engine's class is missing, fallback must "
            "be EasyOcr (which is always available since it's the default)."
        )
        assert any("ocrmac" in r.message.lower() for r in caplog.records), (
            "Fallback path must log which engine was requested + that it "
            "couldn't be loaded."
        )
    finally:
        if real is not None:
            _po.OcrMacOptions = real
