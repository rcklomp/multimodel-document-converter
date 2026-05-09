"""Custom Docling serializers used by the chunker.

Picture items whose only "annotation" is a classification class name like
`other`, `icon`, or `table` cause Docling's default Markdown serializer to
emit the class name as body text - the "label leak" failure mode visible
on HARRY page 13 (`"Other\nTHE BOY WHO LIVED..."`).

In Docling 2.86 the leak has two origins:

  1. ``MarkdownPictureSerializer`` calls ``serialize_annotations`` for any
     legacy ``annotations`` (when ``item.meta`` is absent), which emits
     ``PictureClassificationData`` text.
  2. ``MarkdownMetaSerializer`` walks ``item.meta`` and emits the
     ``classification`` field's main prediction class name.

This module patches both origins:

  * a custom ``MmragMarkdownPictureSerializer`` short-circuits to empty
    output when a picture has only classification labels and no caption /
    description;
  * the chunking serializer's params ship with
    ``blocked_meta_names={"classification"}`` so the meta serializer
    drops the classification field across the board.

Items keep their full label metadata, so downstream consumers can read it
from the provenance and meta - only the body text is filtered.

Upstream tracking:
  https://github.com/docling-project/docling-serve/issues/448
"""
from __future__ import annotations

from typing import Any, Set

from pydantic import Field

from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.base import (
    BaseDocSerializer,
    SerializationResult,
)
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownParams,
    MarkdownPictureSerializer,
)
from docling_core.types.doc.document import (
    DescriptionAnnotation,
    DoclingDocument,
    ImageRefMode,
    PictureClassificationData,
    PictureItem,
    PictureMoleculeData,
)


_BLOCKED_META_NAMES: Set[str] = {"classification"}
_CLASSIFICATION_ONLY_TYPES = (PictureClassificationData,)


def _picture_has_text_caption(
    item: PictureItem,
    doc_serializer: BaseDocSerializer,
    **kwargs: Any,
) -> bool:
    """True when the picture has a non-empty caption from doc_serializer."""
    cap_res = doc_serializer.serialize_captions(item=item, **kwargs)
    return bool(cap_res.text)


def _picture_has_meaningful_annotation(item: PictureItem) -> bool:
    """True when the picture has a description or any non-classification annotation."""
    for ann in item.get_annotations():
        if isinstance(ann, (DescriptionAnnotation, PictureMoleculeData)):
            text = getattr(ann, "text", None) or getattr(ann, "smi", None)
            if text:
                return True
        elif not isinstance(ann, _CLASSIFICATION_ONLY_TYPES):
            return True
    return False


class MmragMarkdownPictureSerializer(MarkdownPictureSerializer):
    """Picture serializer that suppresses bare classification-label text.

    Two failure modes from Docling 2.86 that this class patches:

    1. Picture with classification annotation and NO caption -> the parent
       emits the class name (``"Other"``, ``"Icon"``, ``"Table"``) as body
       text. We short-circuit to empty output.
    2. Picture with classification annotation AND a caption -> the parent
       emits the caption AND the class name. We keep the caption, drop the
       annotation text by stripping classification annotations before
       delegating, then re-attach them so downstream metadata is intact.
    """

    def serialize(  # type: ignore[override]
        self,
        *,
        item: PictureItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        has_caption = _picture_has_text_caption(
            item=item, doc_serializer=doc_serializer, **kwargs
        )
        has_meaningful_annotation = _picture_has_meaningful_annotation(item)
        if not has_caption and not has_meaningful_annotation:
            return create_ser_result(text="", span_source=item)

        # Temporarily strip classification-only annotations so the parent
        # serializer cannot emit them as body text. Description / molecule
        # / chart annotations stay - those are real content, not labels.
        original_annotations = list(getattr(item, "annotations", []))
        filtered = [
            ann for ann in original_annotations
            if not isinstance(ann, _CLASSIFICATION_ONLY_TYPES)
        ]
        try:
            if len(filtered) != len(original_annotations):
                item.annotations = filtered
            result = super().serialize(
                item=item,
                doc_serializer=doc_serializer,
                doc=doc,
                **kwargs,
            )
        finally:
            if len(filtered) != len(original_annotations):
                item.annotations = original_annotations
        return result


class MmragChunkingDocSerializer(ChunkingDocSerializer):
    """ChunkingDocSerializer with the label-leak-suppressing picture serializer."""

    picture_serializer: Any = MmragMarkdownPictureSerializer()
    # default_factory rather than `set()` literal: ChunkingDocSerializer is a
    # Pydantic BaseModel which deep-copies the literal default per instance,
    # so the literal form is safe in practice — but the factory form is the
    # idiomatic Pydantic v2 spelling and protects against future migration
    # to a non-Pydantic base.
    skip_pages: set[int] = Field(default_factory=set)
    params: MarkdownParams = MarkdownParams(
        image_mode=ImageRefMode.PLACEHOLDER,
        image_placeholder="",
        escape_underscores=False,
        escape_html=False,
        blocked_meta_names=_BLOCKED_META_NAMES,
    )

    def serialize(self, *, item: Any = None, **kwargs: Any) -> SerializationResult:  # type: ignore[override]
        if item is not None and self.skip_pages:
            prov = getattr(item, "prov", None)
            if prov:
                first = prov[0] if isinstance(prov, list) else prov
                page_no = getattr(first, "page_no", None)
                if page_no in self.skip_pages:
                    return SerializationResult(text="", spans=[])
        return super().serialize(item=item, **kwargs)


class MmragChunkingSerializerProvider(ChunkingSerializerProvider):
    """Provider that hands MmragChunkingDocSerializer to the chunker."""

    def __init__(self, skip_pages: set[int] | None = None) -> None:
        self.skip_pages = set(skip_pages or set())

    def get_serializer(self, doc: DoclingDocument) -> BaseDocSerializer:  # type: ignore[override]
        return MmragChunkingDocSerializer(doc=doc, skip_pages=self.skip_pages)
