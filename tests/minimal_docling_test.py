#!/usr/bin/env python3
"""
Minimal Docling Test - Fast API exploration.
"""

from pathlib import Path


def main() -> None:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pdf_path = Path("data/raw/HarryPotter_and_the_Sorcerers_Stone.pdf")

    print("Minimal Docling API Test")
    print("=" * 60)
    if not pdf_path.exists():
        print(f"Missing input PDF: {pdf_path}")
        return

    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 1.0
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = False
    pipeline_options.do_ocr = False

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    print(f"Converting {pdf_path}...")
    result = converter.convert(str(pdf_path))
    doc = result.document

    print("\nDocument attributes:")
    print(f"dir(doc): {[a for a in dir(doc) if not a.startswith('_')][:20]}")

    if hasattr(doc, "pages"):
        pages = doc.pages
        print(f"\nPages type: {type(pages)}")
        if isinstance(pages, dict):
            print(f"Page numbers: {list(pages.keys())[:10]}")

    print("\nIterating items (first 10):")
    count = 0
    for i, item_tuple in enumerate(doc.iterate_items()):
        if i >= 10:
            print("... and more items")
            break

        element, _ = item_tuple
        label = getattr(element, "label", "N/A")
        text = str(getattr(element, "text", ""))[:50]

        prov_info = "N/A"
        if hasattr(element, "prov") and element.prov:
            prov = element.prov[0] if isinstance(element.prov, list) else element.prov
            prov_info = f"page={getattr(prov, 'page_no', '?')}"

        print(f"[{i}] {label}: '{text}...' | {prov_info}")
        count += 1

    print(f"\nTotal items iterated: {count}+")


if __name__ == "__main__":
    main()
