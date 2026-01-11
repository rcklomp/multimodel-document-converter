#!/usr/bin/env python3
"""
Quick Docling Layout Detection Test - First 3 pages only
=========================================================
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

print("🔍 Quick Docling Layout Test - Firearms.pdf (first 3 pages)")
print("=" * 60)

# Configure for fast processing
pipeline_options = PdfPipelineOptions()
pipeline_options.images_scale = 1.5
pipeline_options.generate_page_images = False
pipeline_options.generate_picture_images = True
pipeline_options.do_ocr = True

converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

print("Converting first 3 pages...")
result = converter.convert("data/raw/Firearms.pdf")
doc = result.document

print(f"\n📄 Document info:")
print(f"   Name: {doc.name}")

# Collect elements per page
elements_by_page = {}

for item_tuple in doc.iterate_items():
    element, _ = item_tuple

    # Get page number
    page_num = 1
    if hasattr(element, "prov") and element.prov:
        prov = element.prov[0] if isinstance(element.prov, list) else element.prov
        if hasattr(prov, "page_no") and prov.page_no:
            page_num = prov.page_no

    # Get label
    label_obj = getattr(element, "label", None)
    if label_obj is not None:
        label = str(label_obj.value) if hasattr(label_obj, "value") else str(label_obj)
    else:
        label = "unknown"

    # Get bbox
    bbox = None
    if hasattr(element, "prov") and element.prov:
        prov = element.prov[0] if isinstance(element.prov, list) else element.prov
        if hasattr(prov, "bbox") and prov.bbox:
            bbox_obj = prov.bbox
            if hasattr(bbox_obj, "l"):
                bbox = f"({bbox_obj.l:.0f},{bbox_obj.t:.0f},{bbox_obj.r:.0f},{bbox_obj.b:.0f})"

    # Get text preview
    text = str(getattr(element, "text", ""))[:60]

    if page_num not in elements_by_page:
        elements_by_page[page_num] = []
    elements_by_page[page_num].append({"label": label, "bbox": bbox, "text": text})

# Report
print("\n" + "=" * 60)
print("📊 RESULTS")
print("=" * 60)

total_elements = 0
for page in sorted(elements_by_page.keys()):
    elements = elements_by_page[page]
    total_elements += len(elements)

    # Count by type
    types = {}
    for e in elements:
        types[e["label"]] = types.get(e["label"], 0) + 1

    print(f"\n📄 Page {page}: {len(elements)} elements")
    print(f"   Types: {types}")

    # Show first 5 elements
    for i, e in enumerate(elements[:5]):
        print(f"   [{i+1}] {e['label']}: '{e['text'][:40]}...' bbox={e['bbox']}")
    if len(elements) > 5:
        print(f"   ... and {len(elements) - 5} more")

print("\n" + "=" * 60)
print("📋 VERDICT")
print("=" * 60)

if total_elements == 0:
    print("❌ NO ELEMENTS DETECTED - Docling layout fails on this scanned PDF")
    print("   → Need fallback strategy (LayoutParser or region-based splitting)")
elif total_elements < 10:
    print(f"⚠️ MARGINAL - Only {total_elements} elements found")
    print("   → Docling works partially, may need enhancement")
else:
    print(f"✅ SUCCESS - {total_elements} elements detected!")
    print("   → Opus's UIR design will work as specified")
    print(f"✅ SUCCESS - {total_elements} elements detected!")
