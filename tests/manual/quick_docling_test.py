#!/usr/bin/env python3
"""
Quick Docling Layout Detection Test.
"""

from pathlib import Path


def main() -> None:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pdf_path = Path("data/raw/Firearms.pdf")

    print("Quick Docling Layout Test - Firearms.pdf")
    print("=" * 60)
    if not pdf_path.exists():
        print(f"Missing input PDF: {pdf_path}")
        return

    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 1.5
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = True
    pipeline_options.do_ocr = True

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    print("Converting PDF...")
    result = converter.convert(str(pdf_path))
    doc = result.document

    elements_by_page = {}
    for item_tuple in doc.iterate_items():
        element, _ = item_tuple
        page_num = 1
        if hasattr(element, "prov") and element.prov:
            prov = element.prov[0] if isinstance(element.prov, list) else element.prov
            if hasattr(prov, "page_no") and prov.page_no:
                page_num = prov.page_no
        label_obj = getattr(element, "label", None)
        label = str(label_obj.value) if (label_obj is not None and hasattr(label_obj, "value")) else str(label_obj or "unknown")
        elements_by_page.setdefault(page_num, []).append(label)

    total_elements = sum(len(v) for v in elements_by_page.values())
    print(f"Pages with elements: {len(elements_by_page)}")
    print(f"Total elements: {total_elements}")


if __name__ == "__main__":
    main()
