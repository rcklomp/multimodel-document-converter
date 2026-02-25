#!/usr/bin/env python3
"""
Magazine Layout Test script.

Note: Script-style utility intentionally does not execute during pytest import.
"""

from pathlib import Path


def main() -> None:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pdf_path = Path("data/raw/PCWorld_July_2025_USA.pdf")
    print("Magazine Layout Test - PCWorld")
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

    print(f"Converting {pdf_path}...")
    result = converter.convert(str(pdf_path))
    doc = result.document

    page_stats = {}
    type_counts = {}
    for item_tuple in doc.iterate_items():
        element, _ = item_tuple
        label_obj = getattr(element, "label", None)
        label = str(label_obj.value) if (label_obj is not None and hasattr(label_obj, "value")) else str(label_obj or "unknown")
        label = label.lower()

        page_no = 1
        if hasattr(element, "prov") and element.prov:
            prov = element.prov[0] if isinstance(element.prov, list) else element.prov
            if hasattr(prov, "page_no") and prov.page_no:
                page_no = prov.page_no

        page_stats.setdefault(page_no, {"text": 0, "picture": 0, "table": 0, "other": 0})

        if "text" in label or "paragraph" in label or "title" in label or "header" in label:
            page_stats[page_no]["text"] += 1
            type_counts["text"] = type_counts.get("text", 0) + 1
        elif "picture" in label or "figure" in label or "image" in label:
            page_stats[page_no]["picture"] += 1
            type_counts["picture"] = type_counts.get("picture", 0) + 1
        elif "table" in label:
            page_stats[page_no]["table"] += 1
            type_counts["table"] = type_counts.get("table", 0) + 1
        else:
            page_stats[page_no]["other"] += 1
            type_counts[label] = type_counts.get(label, 0) + 1

    print("Type distribution:", type_counts)
    print("Pages with elements:", len(page_stats))


if __name__ == "__main__":
    main()
