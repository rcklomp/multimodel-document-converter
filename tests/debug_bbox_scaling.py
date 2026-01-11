#!/usr/bin/env python3
"""Debug script to test bbox scaling from Docling coords to rendered pixels."""

import sys
import fitz
import numpy as np
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def main():
    pdf_path = "data/raw/Firearms.pdf"
    page_num = 19  # 0-indexed, so page 20
    render_dpi = 150

    print(f"Testing bbox scaling for {pdf_path} page {page_num+1}")
    print(f"Render DPI: {render_dpi}")
    print("=" * 80)

    # Step 1: Render page with PyMuPDF
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)

    # Get PDF dimensions
    pdf_rect = page.rect
    print(f"\nPDF page dimensions: {pdf_rect.width}x{pdf_rect.height} points (72 DPI)")

    # Render at target DPI
    zoom = render_dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    # Convert to numpy
    img_data = pix.tobytes("ppm")
    from PIL import Image
    import io

    pil_image = Image.open(io.BytesIO(img_data)).convert("RGB")
    page_image = np.array(pil_image)

    page_h, page_w = page_image.shape[:2]
    print(f"Rendered image dimensions: {page_w}x{page_h} pixels")
    print(f"Scale factor: {zoom:.4f}")

    # Step 2: Get Docling elements
    print(f"\n{'='*80}")
    print("Running Docling layout analysis...")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.generate_page_images = False

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    result = converter.convert(pdf_path)
    docling_doc = result.document

    # Step 3: Analyze elements for target page
    print(f"\n{'='*80}")
    print(f"Docling elements on page {page_num+1}:")

    element_count = 0
    for item_tuple in docling_doc.iterate_items():
        element, _ = item_tuple

        # Get page number
        elem_page = 1
        if hasattr(element, "prov") and element.prov:
            prov = element.prov[0] if isinstance(element.prov, list) else element.prov
            if hasattr(prov, "page_no") and prov.page_no is not None:
                elem_page = prov.page_no

        if elem_page != page_num + 1:
            continue

        element_count += 1

        # Get label
        label_obj = getattr(element, "label", None)
        if label_obj:
            label = str(label_obj.value) if hasattr(label_obj, "value") else str(label_obj)
        else:
            label = "unknown"

        # Get bbox
        bbox_raw = None
        if hasattr(element, "prov") and element.prov:
            prov = element.prov[0] if isinstance(element.prov, list) else element.prov
            if hasattr(prov, "bbox") and prov.bbox:
                bbox_obj = prov.bbox
                if hasattr(bbox_obj, "l"):
                    bbox_raw = [bbox_obj.l, bbox_obj.t, bbox_obj.r, bbox_obj.b]

        if not bbox_raw:
            continue

        # Calculate scaled bbox
        scale_factor = render_dpi / 72.0
        bbox_scaled = [
            int(bbox_raw[0] * scale_factor),
            int(bbox_raw[1] * scale_factor),
            int(bbox_raw[2] * scale_factor),
            int(bbox_raw[3] * scale_factor),
        ]

        # Check if bbox is within page bounds
        in_bounds = (
            0 <= bbox_scaled[0] < page_w
            and 0 <= bbox_scaled[1] < page_h
            and 0 <= bbox_scaled[2] <= page_w
            and 0 <= bbox_scaled[3] <= page_h
        )

        # Calculate crop dimensions
        crop_w = bbox_scaled[2] - bbox_scaled[0]
        crop_h = bbox_scaled[3] - bbox_scaled[1]

        print(f"\nElement {element_count}: {label}")
        print(
            f"  Raw bbox (PDF points):  [{bbox_raw[0]:.1f}, {bbox_raw[1]:.1f}, {bbox_raw[2]:.1f}, {bbox_raw[3]:.1f}]"
        )
        print(
            f"  Scaled bbox (pixels):   [{bbox_scaled[0]}, {bbox_scaled[1]}, {bbox_scaled[2]}, {bbox_scaled[3]}]"
        )
        print(f"  Crop size: {crop_w}x{crop_h} pixels")
        print(f"  In bounds: {in_bounds}")

        if in_bounds and crop_w > 0 and crop_h > 0:
            # Try to crop
            try:
                x1, y1, x2, y2 = bbox_scaled
                crop = page_image[y1:y2, x1:x2]
                print(f"  Crop result: {crop.shape} {'✓' if crop.size > 0 else '✗ EMPTY'}")
            except Exception as e:
                print(f"  Crop failed: {e}")
        else:
            print(f"  ✗ Cannot crop: out of bounds or invalid dimensions")

    print(f"\n{'='*80}")
    print(f"Total elements on page {page_num+1}: {element_count}")

    doc.close()


if __name__ == "__main__":
    main()
