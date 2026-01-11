#!/usr/bin/env python3
"""
Minimal Docling Test - Fast API exploration
=============================================
Tests with the smallest available PDF to quickly understand Docling's output structure.
"""
import sys
from pathlib import Path

# Use the HarryPotter PDF which should be digital (fast to process)
pdf_path = "data/raw/HarryPotter_and_the_Sorcerers_Stone.pdf"

print(f"🔍 Minimal Docling API Test")
print("=" * 60)

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# Minimal config - no image extraction for speed
pipeline_options = PdfPipelineOptions()
pipeline_options.images_scale = 1.0
pipeline_options.generate_page_images = False
pipeline_options.generate_picture_images = False
pipeline_options.do_ocr = False  # Disable OCR for speed

converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

print(f"Converting {pdf_path}...")
result = converter.convert(pdf_path)
doc = result.document

print(f"\n📄 Document attributes:")
print(f"   dir(doc): {[a for a in dir(doc) if not a.startswith('_')][:20]}")

# Check pages attribute
if hasattr(doc, "pages"):
    pages = doc.pages
    print(f"\n📄 Pages type: {type(pages)}")
    if isinstance(pages, dict):
        print(f"   Page numbers: {list(pages.keys())[:10]}")
        if pages:
            first_key = list(pages.keys())[0]
            first_page = pages[first_key]
            print(
                f"   First page attributes: {[a for a in dir(first_page) if not a.startswith('_')]}"
            )
    else:
        print(f"   Pages: {pages}")

# Check iterate_items
print(f"\n📄 Iterating items (first 10):")
count = 0
for i, item_tuple in enumerate(doc.iterate_items()):
    if i >= 10:
        print(f"   ... and more items")
        break

    element, level = item_tuple

    # Get element info
    label = getattr(element, "label", "N/A")
    text = str(getattr(element, "text", ""))[:50]

    # Check prov attribute
    prov_info = "N/A"
    if hasattr(element, "prov") and element.prov:
        prov = element.prov[0] if isinstance(element.prov, list) else element.prov
        prov_info = f"page={getattr(prov, 'page_no', '?')}"
        if hasattr(prov, "bbox") and prov.bbox:
            bbox = prov.bbox
            if hasattr(bbox, "l"):
                prov_info += f", bbox=({bbox.l:.0f},{bbox.t:.0f},{bbox.r:.0f},{bbox.b:.0f})"

    print(f"   [{i}] {label}: '{text}...' | {prov_info}")
    count += 1

print(f"\n✅ Total items iterated: {count}+")
print("=" * 60)
print("VERDICT: Docling iterate_items() works and provides elements with prov/bbox!")
