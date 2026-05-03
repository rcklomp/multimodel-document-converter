"""Phase 3 tests for the post-Docling label-leak filter (custom serializer).

Docling's default Markdown serializer emits picture classification labels
(`other`, `icon`, `table`) as body text whenever the picture has a
classification annotation but no caption. On HARRY page 13 this produces
chunks that begin with `"Other\nTHE BOY WHO LIVED..."`. The custom
serializer suppresses that text without losing the underlying label,
which still flows through the picture item's metadata.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
)
from docling_core.types.doc.document import (
    BoundingBox,
    CoordOrigin,
    DoclingDocument,
    PictureClassificationClass,
    PictureClassificationData,
    ProvenanceItem,
)

from mmrag_v2.engines.docling_serializers import (
    MmragChunkingDocSerializer,
    MmragChunkingSerializerProvider,
    MmragMarkdownPictureSerializer,
)
from mmrag_v2.engines.pdf_plan import PdfConversionPlan, build_pdf_conversion_plan


def _make_doc_with_picture(
    *,
    classification: str = "other",
    confidence: float = 0.9,
    caption: str = "",
) -> tuple[DoclingDocument, Any]:
    doc = DoclingDocument(name="test")
    annotations = [
        PictureClassificationData(
            provenance="test",
            predicted_classes=[
                PictureClassificationClass(
                    class_name=classification, confidence=confidence
                )
            ],
        )
    ] if classification else None
    prov = ProvenanceItem(
        page_no=13,
        bbox=BoundingBox(
            l=10.0, t=10.0, r=110.0, b=110.0, coord_origin=CoordOrigin.TOPLEFT
        ),
        charspan=(0, 0),
    )
    pic = doc.add_picture(annotations=annotations, prov=prov)
    if caption:
        cap_item = doc.add_text(label="caption", text=caption, prov=prov)
        pic.captions = [cap_item.get_ref()]
    return doc, pic


def test_picture_serializer_emits_empty_when_only_classification_label():
    """Picture with only a classification label and no caption -> empty text."""
    doc, _ = _make_doc_with_picture(classification="other")
    serializer = MmragChunkingDocSerializer(doc=doc)
    result = serializer.serialize()
    assert "Other" not in result.text
    assert "other" not in result.text


def test_picture_serializer_emits_caption_when_present():
    """Picture with a caption emits the caption text only."""
    doc, _ = _make_doc_with_picture(
        classification="other", caption="Figure 1: Cover photograph"
    )
    serializer = MmragChunkingDocSerializer(doc=doc)
    result = serializer.serialize()
    assert "Figure 1: Cover photograph" in result.text


def test_default_chunking_serializer_does_leak_classification_label():
    """Sanity: confirm default ChunkingDocSerializer DOES leak the label.

    This pins the failure mode that the Mmrag serializer is meant to fix:
    if the upstream behavior changes (e.g. Docling adds skip_labels), the
    label would no longer appear and this test would fail loudly, prompting
    a review of whether our custom serializer is still needed.
    """
    doc, _ = _make_doc_with_picture(classification="other")
    serializer = MarkdownDocSerializer(doc=doc)
    text = serializer.serialize().text.lower()
    assert "other" in text


def test_serializer_provider_returns_mmrag_serializer():
    """The provider hands back an Mmrag-aware serializer instance."""
    doc, _ = _make_doc_with_picture(classification="icon")
    provider = MmragChunkingSerializerProvider()
    serializer = provider.get_serializer(doc)
    assert isinstance(serializer, MmragChunkingDocSerializer)
    assert isinstance(serializer.picture_serializer, MmragMarkdownPictureSerializer)


def test_plan_default_disables_suppression():
    plan = PdfConversionPlan()
    assert plan.suppress_layout_label_text is False


def test_plan_can_enable_suppression_through_builder():
    plan = build_pdf_conversion_plan(suppress_layout_label_text=True)
    assert plan.suppress_layout_label_text is True


def test_processor_picks_up_suppress_layout_label_text(monkeypatch, tmp_path):
    """Bridge: V2DocumentProcessor reads the flag from the conversion plan."""
    from mmrag_v2.processor import V2DocumentProcessor
    from mmrag_v2.engines.docling_adapter import DoclingPdfAdapter
    from types import SimpleNamespace

    monkeypatch.setattr(
        DoclingPdfAdapter,
        "get_converter",
        lambda self: SimpleNamespace(convert=lambda _path: None),
    )

    plan = PdfConversionPlan(suppress_layout_label_text=True)
    proc = V2DocumentProcessor(
        output_dir=str(tmp_path),
        vision_provider="none",
        conversion_plan=plan,
    )
    assert proc._suppress_layout_label_text is True


def test_batch_processor_picks_up_suppress_layout_label_text(tmp_path):
    """Bridge: BatchProcessor reads the flag from the conversion plan."""
    from mmrag_v2.batch_processor import BatchProcessor

    plan = PdfConversionPlan(suppress_layout_label_text=True)
    proc = BatchProcessor(output_dir=str(tmp_path))
    proc.set_conversion_plan(plan)
    assert proc._suppress_layout_label_text is True


def test_processor_default_suppression_is_false(monkeypatch, tmp_path):
    """Default plan keeps suppression off."""
    from mmrag_v2.processor import V2DocumentProcessor
    from mmrag_v2.engines.docling_adapter import DoclingPdfAdapter
    from types import SimpleNamespace

    monkeypatch.setattr(
        DoclingPdfAdapter,
        "get_converter",
        lambda self: SimpleNamespace(convert=lambda _path: None),
    )

    proc = V2DocumentProcessor(
        output_dir=str(tmp_path),
        vision_provider="none",
        conversion_plan=PdfConversionPlan(),
    )
    assert proc._suppress_layout_label_text is False
