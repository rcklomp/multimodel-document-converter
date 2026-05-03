"""Docling PDF adapter and converter factory."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Tuple

from .docling_postprocess import apply_postprocessors
from .pdf_plan import PdfConversionPlan

logger = logging.getLogger(__name__)


class DoclingPdfAdapter:
    """Owns Docling PDF option and converter construction."""

    def __init__(self, plan: PdfConversionPlan) -> None:
        self.plan = plan
        self._converter: Any = None

    def _load_docling_classes(self) -> Tuple[Any, Any, Any, Any, Any, Any, Any]:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            EasyOcrOptions,
            PdfPipelineOptions,
            TableFormerMode,
            TableStructureOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption

        return (
            InputFormat,
            PdfPipelineOptions,
            EasyOcrOptions,
            TableStructureOptions,
            TableFormerMode,
            DocumentConverter,
            PdfFormatOption,
        )

    def get_converter(self) -> Any:
        """Return a cached Docling DocumentConverter."""
        if self._converter is not None:
            logger.info("[DOCLING-ADAPTER] Reusing cached DocumentConverter")
            return self._converter

        (
            InputFormat,
            PdfPipelineOptions,
            EasyOcrOptions,
            TableStructureOptions,
            TableFormerMode,
            DocumentConverter,
            PdfFormatOption,
        ) = self._load_docling_classes()

        options = PdfPipelineOptions()
        options.images_scale = self.plan.images_scale
        options.generate_page_images = self.plan.generate_page_images
        options.generate_picture_images = self.plan.generate_picture_images
        options.generate_table_images = self.plan.generate_table_images

        if hasattr(options, "sort_by_reading_order"):
            options.sort_by_reading_order = self.plan.sort_by_reading_order
        if hasattr(options, "do_table_structure"):
            options.do_table_structure = self.plan.do_table_structure
        if hasattr(options, "do_cell_matching"):
            options.do_cell_matching = self.plan.do_cell_matching

        try:
            mode = getattr(TableFormerMode, self.plan.table_former_mode)
            options.table_structure_options = TableStructureOptions(
                do_cell_matching=self.plan.do_cell_matching,
                mode=mode,
            )
        except Exception as exc:
            logger.debug("[DOCLING-ADAPTER] Table structure options unavailable: %s", exc)

        if hasattr(options, "do_picture_classification"):
            options.do_picture_classification = self.plan.do_picture_classification
        if self.plan.do_picture_classification and hasattr(options, "picture_description_options"):
            picture_options = options.picture_description_options
            if hasattr(picture_options, "classification_deny"):
                # Build deny list from plan, allowing page-level visuals when enabled.
                deny_list = list(self.plan.picture_classification_deny)
                if self.plan.allow_page_level_visuals:
                    # image_heavy_magazine route: remove full_page_image from deny list
                    # so that full-page editorial images can be classified and described.
                    deny_list = [d for d in deny_list if d != "full_page_image"]
                picture_options.classification_deny = deny_list

        options.do_ocr = self.plan.do_ocr
        if self.plan.do_ocr:
            # Existing call sites always fell back to EasyOcrOptions regardless
            # of the CLI engine string. Preserve that behavior in the adapter.
            ocr_options = EasyOcrOptions()
            # Phase 4: raise the bitmap-area threshold so photographic cover
            # pages on born-digital documents are not OCR'd into garbage.
            if hasattr(ocr_options, "bitmap_area_threshold"):
                ocr_options.bitmap_area_threshold = float(
                    self.plan.bitmap_area_threshold
                )
            options.ocr_options = ocr_options

        if self.plan.needs_code_enrichment:
            options.do_code_enrichment = True
            options.do_formula_enrichment = self.plan.do_formula_enrichment
            logger.info(
                "[CODE-ENRICH] Enabled Docling code enrichment (reason: %s)",
                self.plan.code_enrichment_reason,
            )
        elif hasattr(options, "do_formula_enrichment"):
            options.do_formula_enrichment = self.plan.do_formula_enrichment

        self._converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )
        logger.info("[DOCLING-ADAPTER] Created cached DocumentConverter")
        return self._converter

    def convert(self, pdf_path: str | Path) -> Any:
        """Convert a PDF path through the cached Docling converter.

        Runs registered post-Docling sanity stages (reading-order y-sort,
        future drop-cap promotion) on the returned document, gated by plan
        fields. The returned ConversionResult is mutated in place; callers
        keep the same object reference.
        """
        result = self.get_converter().convert(str(pdf_path))
        document = getattr(result, "document", None)
        if document is not None:
            apply_postprocessors(document, self.plan)
        return result
