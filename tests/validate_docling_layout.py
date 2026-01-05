#!/usr/bin/env python3
"""
WEEK 0 - Test 1: Validate Docling Layout Detection on Scanned PDFs
===================================================================

This test determines whether Docling's native layout detection works
on vintage scanned documents. If it fails, we'll need LayoutParser.

GO/NO-GO Criteria:
- GO:    >5 regions per page with correct types (text, picture, table)
- NO-GO: 0-2 regions OR all type="unknown" → Need LayoutParser

Usage:
    python tests/validate_docling_layout.py data/raw/Firearms.pdf --pages 5,21,100

Author: Claude (Architect)
Date: January 3, 2026
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def render_page_to_image(pdf_path: Path, page_num: int, dpi: int = 300):
    """Render a PDF page to PIL Image using PyMuPDF."""
    import fitz
    from PIL import Image
    import io

    doc = fitz.open(str(pdf_path))
    page = doc.load_page(page_num - 1)  # 0-indexed

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    img_data = pix.tobytes("ppm")
    image = Image.open(io.BytesIO(img_data)).convert("RGB")

    doc.close()
    return image


def test_docling_layout(pdf_path: Path, page_num: int) -> dict:
    """
    Test Docling's layout detection on a single page.

    Returns:
        dict with:
        - regions_found: int
        - region_types: List[str]
        - has_text_regions: bool
        - has_image_regions: bool
        - confidence: str (GO/NO-GO assessment)
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    print(f"\n{'='*60}")
    print(f"Testing Docling layout on page {page_num}")
    print(f"{'='*60}")

    # Configure Docling with layout analysis
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_table_images = True
    pipeline_options.do_ocr = True  # Enable OCR for scanned pages

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    # Convert document
    result = converter.convert(str(pdf_path))
    doc = result.document

    # Collect all elements from the target page
    regions = []
    region_types = []

    for item_tuple in doc.iterate_items():
        element, _ = item_tuple

        # Get page number from provenance
        element_page = 1
        if hasattr(element, "prov") and element.prov:
            prov = element.prov[0] if isinstance(element.prov, list) else element.prov
            if hasattr(prov, "page_no") and prov.page_no:
                element_page = prov.page_no

        if element_page != page_num:
            continue

        # Get element type
        label_obj = getattr(element, "label", None)
        if label_obj is not None:
            label = str(label_obj.value) if hasattr(label_obj, "value") else str(label_obj)
        else:
            label = "unknown"

        # Get bounding box
        bbox = None
        if hasattr(element, "prov") and element.prov:
            prov = element.prov[0] if isinstance(element.prov, list) else element.prov
            if hasattr(prov, "bbox") and prov.bbox:
                bbox_obj = prov.bbox
                if hasattr(bbox_obj, "l"):
                    bbox = [bbox_obj.l, bbox_obj.t, bbox_obj.r, bbox_obj.b]

        regions.append(
            {"type": label, "bbox": bbox, "text_preview": str(getattr(element, "text", ""))[:50]}
        )
        region_types.append(label.lower())

    # Analyze results
    has_text = any("text" in t or "paragraph" in t or "section" in t for t in region_types)
    has_image = any("picture" in t or "figure" in t or "image" in t for t in region_types)
    has_table = any("table" in t for t in region_types)

    print(f"\n📊 Results for page {page_num}:")
    print(f"   Total regions found: {len(regions)}")
    print(f"   Region types: {set(region_types)}")
    print(f"   Has text regions: {'✅' if has_text else '❌'}")
    print(f"   Has image regions: {'✅' if has_image else '❌'}")
    print(f"   Has table regions: {'✅' if has_table else '❌'}")

    print(f"\n   Detailed regions:")
    for i, r in enumerate(regions[:10]):  # Show first 10
        print(f"   [{i+1}] {r['type']}: {r['text_preview'][:30]}... (bbox: {r['bbox']})")

    if len(regions) > 10:
        print(f"   ... and {len(regions) - 10} more regions")

    # GO/NO-GO assessment
    if len(regions) >= 5 and (has_text or has_image):
        assessment = "✅ GO - Docling layout works on this page"
    elif len(regions) >= 3:
        assessment = "⚠️ MARGINAL - Docling found some regions but may miss content"
    else:
        assessment = "❌ NO-GO - Docling layout failed, need LayoutParser"

    print(f"\n   Assessment: {assessment}")

    return {
        "page": page_num,
        "regions_found": len(regions),
        "region_types": list(set(region_types)),
        "has_text_regions": has_text,
        "has_image_regions": has_image,
        "has_table_regions": has_table,
        "assessment": assessment,
    }


def main():
    parser = argparse.ArgumentParser(description="Test Docling layout detection on scanned PDFs")
    parser.add_argument("pdf_path", type=Path, help="Path to PDF file")
    parser.add_argument(
        "--pages",
        type=str,
        default="5,21,50",
        help="Comma-separated page numbers to test (default: 5,21,50)",
    )

    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"❌ Error: PDF not found: {args.pdf_path}")
        sys.exit(1)

    pages = [int(p.strip()) for p in args.pages.split(",")]

    print(f"🔍 Testing Docling Layout Detection")
    print(f"   PDF: {args.pdf_path}")
    print(f"   Pages: {pages}")

    results = []
    for page_num in pages:
        try:
            result = test_docling_layout(args.pdf_path, page_num)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error on page {page_num}: {e}")
            results.append({"page": page_num, "error": str(e), "assessment": "❌ ERROR"})

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")

    go_count = sum(1 for r in results if "GO" in r.get("assessment", ""))
    total = len(results)

    print(f"\n   Pages tested: {total}")
    print(f"   GO assessments: {go_count}/{total}")

    if go_count == total:
        print(f"\n✅ FINAL VERDICT: Docling layout works! Use native layout detection.")
    elif go_count >= total // 2:
        print(f"\n⚠️ FINAL VERDICT: Docling partially works. Consider LayoutParser as backup.")
    else:
        print(f"\n❌ FINAL VERDICT: Docling layout fails on scans. Use LayoutParser.")

    # Save results
    import json

    output_path = Path("tests/docling_layout_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n   Results saved to: {output_path}")


if __name__ == "__main__":
    main()
