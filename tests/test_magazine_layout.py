#!/usr/bin/env python3
"""
Magazine Layout Test - Test Docling on PCWorld (modern magazine with images)
=============================================================================
"""
import sys
from pathlib import Path

pdf_path = "data/raw/PCWorld_July_2025_USA.pdf"

print(f"🔍 Magazine Layout Test - PCWorld")
print("=" * 60)

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# Enable picture detection
pipeline_options = PdfPipelineOptions()
pipeline_options.images_scale = 1.5
pipeline_options.generate_page_images = False
pipeline_options.generate_picture_images = True
pipeline_options.do_ocr = True  # Enable OCR for any scanned content

converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

print(f"Converting {pdf_path}...")
result = converter.convert(pdf_path)
doc = result.document

# Count elements by type and page
page_stats = {}
type_counts = {}

for item_tuple in doc.iterate_items():
    element, level = item_tuple

    # Get label
    label_obj = getattr(element, "label", None)
    if label_obj is not None:
        label = str(label_obj.value) if hasattr(label_obj, "value") else str(label_obj)
    else:
        label = "unknown"
    label = label.lower()

    # Get page
    page_no = 1
    if hasattr(element, "prov") and element.prov:
        prov = element.prov[0] if isinstance(element.prov, list) else element.prov
        if hasattr(prov, "page_no") and prov.page_no:
            page_no = prov.page_no

    # Track stats
    if page_no not in page_stats:
        page_stats[page_no] = {"text": 0, "picture": 0, "table": 0, "other": 0}

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

print(f"\n📊 RESULTS")
print("=" * 60)
print(f"Total pages with elements: {len(page_stats)}")
print(f"Type distribution: {type_counts}")

# Show first 10 pages
print(f"\nPer-page breakdown (first 10):")
for page_no in sorted(page_stats.keys())[:10]:
    stats = page_stats[page_no]
    total = sum(stats.values())
    print(
        f"   Page {page_no:3d}: {total:3d} elements | text={stats['text']:2d} picture={stats['picture']:2d} table={stats['table']:2d}"
    )

# Verdict
total_text = type_counts.get("text", 0)
total_picture = type_counts.get("picture", 0)

print(f"\n" + "=" * 60)
print("📋 VERDICT")
print("=" * 60)
if total_text > 50 and total_picture > 10:
    print(
        f"✅ SUCCESS - Docling detects TEXT ({total_text}) and PICTURE ({total_picture}) elements!"
    )
    print("   → UIR pipeline will work as designed")
elif total_text > 20:
    print(f"⚠️ PARTIAL - Docling detects text but limited pictures")
    print("   → May need image extraction enhancement")
else:
    print(f"❌ INSUFFICIENT - Only {total_text} text, {total_picture} picture elements")
    print("   → Need fallback strategy")
