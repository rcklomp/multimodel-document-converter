"""Fast Docling extraction adapter — single Docling import boundary.

This is the ONLY V3 Phase C engine permitted to import `docling`. It
wraps Docling in a stripped-down configuration optimized for speed on
text/prose pages: OCR disabled, CPU accelerator (MPS float64 bug
workaround), no picture classification. Returns the same v2-UIR
``UniversalDocument`` shape every Phase C engine emits.

Used by the HybridEngine router for pages that do not need vision-
native parsing. Not intended for direct invocation outside the router.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from mmrag_v2.universal.intermediate import (
    DocumentMetadata,
    ElementType,
    ExtractionMethod,
    PageClassification,
    UniversalDocument,
    UniversalPage,
    create_document,
    create_element,
    create_page,
)

logger = logging.getLogger(__name__)

_COORD_SCALE = 1000
_DEFAULT_PAGE_W = 612.0
_DEFAULT_PAGE_H = 792.0


def _build_fast_converter() -> Any:
    """Construct a Docling DocumentConverter with the fast prose policy.

    No OCR, CPU accelerator, no picture classification. Table structure
    stays on so the router still benefits when a Docling-routed page
    happens to contain a simple table the heuristic missed.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableFormerMode,
        TableStructureOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_ocr = False
    options.generate_page_images = False
    options.generate_picture_images = False
    options.generate_table_images = False
    if hasattr(options, "do_picture_classification"):
        options.do_picture_classification = False
    if hasattr(options, "do_table_structure"):
        options.do_table_structure = True
    try:
        options.table_structure_options = TableStructureOptions(
            do_cell_matching=False,
            mode=TableFormerMode.FAST,
        )
    except Exception as exc:  # pragma: no cover
        logger.debug("Docling table options unavailable: %s", exc)

    if hasattr(options, "accelerator_options"):
        try:
            from docling.datamodel.pipeline_options import (
                AcceleratorDevice,
                AcceleratorOptions,
            )

            options.accelerator_options = AcceleratorOptions(
                device=AcceleratorDevice.CPU
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("Docling CPU accelerator unavailable: %s", exc)

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def _item_page_no(item: Any) -> Optional[int]:
    prov = getattr(item, "prov", None)
    if not prov:
        return None
    first = prov[0] if isinstance(prov, list) else prov
    page_no = getattr(first, "page_no", None)
    return int(page_no) if page_no else None


def _item_label(item: Any) -> str:
    label = getattr(item, "label", "")
    value = getattr(label, "value", label)
    return str(value or "").replace("-", "_").replace(" ", "_").lower()


def _item_text(item: Any) -> str:
    raw = str(getattr(item, "text", "") or "")
    return raw.replace("﻿", "").strip()


def _normalize_bbox(
    bbox_obj: Any, page_w: float, page_h: float
) -> Optional[List[int]]:
    if bbox_obj is None or page_w <= 0 or page_h <= 0:
        return None
    try:
        l = float(getattr(bbox_obj, "l", 0))
        t = float(getattr(bbox_obj, "t", 0))
        r = float(getattr(bbox_obj, "r", page_w))
        b = float(getattr(bbox_obj, "b", page_h))
    except (TypeError, ValueError):
        return None

    x0 = int(l / page_w * _COORD_SCALE)
    y0 = int(t / page_h * _COORD_SCALE)
    x1 = int(r / page_w * _COORD_SCALE)
    y1 = int(b / page_h * _COORD_SCALE)
    x_min = max(0, min(_COORD_SCALE, x0))
    x_max = max(0, min(_COORD_SCALE, x1))
    y_min = max(0, min(_COORD_SCALE, min(y0, y1)))
    y_max = max(0, min(_COORD_SCALE, max(y0, y1)))
    if x_min >= x_max:
        if x_max < _COORD_SCALE:
            x_max = x_min + 1
        else:
            x_min = x_max - 1
    if y_min >= y_max:
        if y_max < _COORD_SCALE:
            y_max = y_min + 1
        else:
            y_min = y_max - 1
    return [x_min, y_min, x_max, y_max]


def _classify_element_type(label: str) -> ElementType:
    if label in {"table", "table_cell", "tablecell", "tabular", "grid"}:
        return ElementType.TABLE
    if label in {"image", "figure", "picture", "pic", "img"}:
        return ElementType.IMAGE
    return ElementType.TEXT


def _page_dimensions(doc: Any) -> Dict[int, Tuple[float, float]]:
    dims: Dict[int, Tuple[float, float]] = {}
    pages = getattr(doc, "pages", None)
    if not pages:
        return dims
    iterator = pages.values() if isinstance(pages, dict) else pages
    for page in iterator:
        page_no = getattr(page, "page_no", None)
        size = getattr(page, "size", None)
        if size is not None:
            w = float(getattr(size, "width", 0) or 0)
            h = float(getattr(size, "height", 0) or 0)
        else:
            w = float(getattr(page, "width", 0) or 0)
            h = float(getattr(page, "height", 0) or 0)
        if page_no is None:
            continue
        dims[int(page_no)] = (w or _DEFAULT_PAGE_W, h or _DEFAULT_PAGE_H)
    return dims


class DoclingFastEngine:
    """Phase C fast Docling adapter (CPU, no OCR)."""

    def extract(self, file_path: str) -> UniversalDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")
        converter = _build_fast_converter()
        result = converter.convert(str(path))
        document = getattr(result, "document", None)
        if document is None:
            raise RuntimeError(
                f"Docling convert() returned no document for {file_path}"
            )

        dims = _page_dimensions(document)
        page_buckets: Dict[int, List[Any]] = {}
        for item, _level in document.iterate_items():
            page_no = _item_page_no(item)
            if page_no is None:
                continue
            text = _item_text(item)
            label = _item_label(item)
            elem_type = _classify_element_type(label)
            if elem_type == ElementType.TEXT and not text:
                continue
            pw, ph = dims.get(page_no, (_DEFAULT_PAGE_W, _DEFAULT_PAGE_H))
            prov = getattr(item, "prov", None)
            bbox_obj = None
            if prov:
                first = prov[0] if isinstance(prov, list) else prov
                bbox_obj = getattr(first, "bbox", None)
            bbox = _normalize_bbox(bbox_obj, pw, ph)
            bucket = page_buckets.setdefault(page_no, [])
            bucket.append(
                create_element(
                    element_type=elem_type,
                    content=text,
                    bbox=bbox,
                    confidence=1.0 if text else 0.5,
                    extraction_method=ExtractionMethod.NATIVE,
                    element_index=len(bucket),
                    source_label=label or "text",
                )
            )

        universal_pages: List[UniversalPage] = []
        all_page_nos = sorted(set(list(page_buckets.keys()) + list(dims.keys())))
        for page_no in all_page_nos:
            pw, ph = dims.get(page_no, (_DEFAULT_PAGE_W, _DEFAULT_PAGE_H))
            elements = page_buckets.get(page_no, [])
            universal_pages.append(
                create_page(
                    page_number=page_no,
                    elements=elements,
                    dimensions=(int(round(pw)), int(round(ph))),
                    classification=PageClassification.DIGITAL,
                )
            )

        metadata = DocumentMetadata(
            page_count=len(universal_pages),
            file_size_bytes=path.stat().st_size,
            has_text_layer=True,
        )
        return create_document(
            file_path=path,
            file_type="pdf",
            pages=universal_pages,
            metadata=metadata,
        )


def extract(file_path: Union[str, Path]) -> UniversalDocument:
    """Module-level convenience: ``extract(file_path) -> UniversalDocument``."""
    return DoclingFastEngine().extract(str(file_path))
