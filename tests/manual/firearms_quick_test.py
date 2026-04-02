#!/usr/bin/env python3
"""Quick test for Firearms.pdf layout detection."""

from pathlib import Path


def main() -> None:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pdf_path = Path("data/raw/Firearms.pdf")
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

    print("Converting Firearms.pdf...")
    result = converter.convert(str(pdf_path))
    doc = result.document

    text_count = 0
    picture_count = 0
    table_count = 0
    pages_seen = set()

    for item_tuple in doc.iterate_items():
        element, _ = item_tuple
        label = getattr(element, "label", None)
        if label:
            label = str(label.value) if hasattr(label, "value") else str(label)
            label = label.lower()
        else:
            label = "unknown"

        if hasattr(element, "prov") and element.prov:
            prov = element.prov[0] if isinstance(element.prov, list) else element.prov
            if hasattr(prov, "page_no") and prov.page_no:
                pages_seen.add(prov.page_no)

        if "text" in label or "paragraph" in label or "section" in label:
            text_count += 1
        elif "picture" in label or "figure" in label:
            picture_count += 1
        elif "table" in label:
            table_count += 1

    print()
    print("=" * 60)
    print("FIREARMS.PDF DOCLING LAYOUT DETECTION RESULTS")
    print("=" * 60)
    print(f"Pages with elements: {len(pages_seen)}")
    print(f"TEXT elements: {text_count}")
    print(f"PICTURE elements: {picture_count}")
    print(f"TABLE elements: {table_count}")
    print()
    if text_count > 0 or picture_count > 0:
        print("VERDICT: Docling detects elements on scanned Firearms.pdf")
    else:
        print("VERDICT: Docling returned 0 elements")


if __name__ == "__main__":
    main()
