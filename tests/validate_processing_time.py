#!/usr/bin/env python3
"""
WEEK 0 - Test 3: Validate Processing Time Budget
=================================================

This test measures realistic processing times for the enhanced OCR pipeline
to determine acceptable performance budgets.

GO/NO-GO Criteria:
- GO:    <30s per page → Acceptable for initial release
- WARN:  30-60s per page → Consider optimization
- NO-GO: >60s per page → Too slow, need optimization before deploy

Measures:
- Page rendering time
- Image preprocessing time
- OCR time (Tesseract)
- Layout detection time (if Docling works)
- Total pipeline time

Usage:
    python tests/validate_processing_time.py data/raw/Firearms.pdf --pages 5,21,50

Author: Claude (Architect)
Date: January 3, 2026
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class TimingResult:
    """Timing results for a single page."""

    page: int
    render_ms: int
    preprocess_ms: int
    ocr_ms: int
    layout_ms: int
    total_ms: int
    page_size: tuple


def render_page_to_image(pdf_path: Path, page_num: int, dpi: int = 300):
    """Render a PDF page to numpy array using PyMuPDF."""
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
    return np.array(image)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Basic preprocessing for OCR."""
    import cv2

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=11, C=2
    )

    return binary


def run_tesseract_ocr(image: np.ndarray) -> str:
    """Run Tesseract OCR and return text."""
    import pytesseract

    return pytesseract.image_to_string(image, config="--psm 3")


def run_docling_layout(pdf_path: Path, page_num: int) -> int:
    """
    Test Docling layout detection timing.

    Returns number of regions found.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 2.0
    pipeline_options.do_ocr = False  # Skip OCR for timing

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    result = converter.convert(str(pdf_path))

    # Count elements on target page
    region_count = 0
    for item_tuple in result.document.iterate_items():
        element, _ = item_tuple
        if hasattr(element, "prov") and element.prov:
            prov = element.prov[0] if isinstance(element.prov, list) else element.prov
            if hasattr(prov, "page_no") and prov.page_no == page_num:
                region_count += 1

    return region_count


def time_full_pipeline(pdf_path: Path, page_num: int) -> TimingResult:
    """
    Time the full processing pipeline for a single page.

    Measures:
    1. Page rendering (PyMuPDF)
    2. Image preprocessing (OpenCV)
    3. OCR (Tesseract)
    4. Layout detection (Docling)
    """

    print(f"\n{'='*60}")
    print(f"Timing page {page_num}")
    print(f"{'='*60}")

    # 1. Render page
    print("   ⏱️  Rendering page...")
    start = time.perf_counter()
    image = render_page_to_image(pdf_path, page_num)
    render_ms = int((time.perf_counter() - start) * 1000)
    print(f"      Render: {render_ms}ms ({image.shape[1]}x{image.shape[0]}px)")

    # 2. Preprocessing
    print("   ⏱️  Preprocessing...")
    start = time.perf_counter()
    preprocessed = preprocess_image(image)
    preprocess_ms = int((time.perf_counter() - start) * 1000)
    print(f"      Preprocess: {preprocess_ms}ms")

    # 3. OCR
    print("   ⏱️  Running Tesseract OCR...")
    start = time.perf_counter()
    ocr_text = run_tesseract_ocr(preprocessed)
    ocr_ms = int((time.perf_counter() - start) * 1000)
    word_count = len(ocr_text.split())
    print(f"      OCR: {ocr_ms}ms ({word_count} words)")

    # 4. Layout detection (Docling)
    print("   ⏱️  Running Docling layout detection...")
    start = time.perf_counter()
    try:
        region_count = run_docling_layout(pdf_path, page_num)
        layout_ms = int((time.perf_counter() - start) * 1000)
        print(f"      Layout: {layout_ms}ms ({region_count} regions)")
    except Exception as e:
        layout_ms = 0
        print(f"      Layout: FAILED ({e})")

    # Total
    total_ms = render_ms + preprocess_ms + ocr_ms + layout_ms

    print(f"\n   📊 TOTAL: {total_ms}ms ({total_ms / 1000:.1f}s)")

    # Assessment
    if total_ms < 30000:
        print(f"   ✅ Within 30s budget - ACCEPTABLE")
    elif total_ms < 60000:
        print(f"   ⚠️ 30-60s - Consider optimization")
    else:
        print(f"   ❌ >60s - TOO SLOW, optimization required")

    return TimingResult(
        page=page_num,
        render_ms=render_ms,
        preprocess_ms=preprocess_ms,
        ocr_ms=ocr_ms,
        layout_ms=layout_ms,
        total_ms=total_ms,
        page_size=(image.shape[1], image.shape[0]),
    )


def main():
    parser = argparse.ArgumentParser(description="Validate processing time budget")
    parser.add_argument("pdf_path", type=Path, help="Path to PDF file")
    parser.add_argument(
        "--pages",
        type=str,
        default="5,21,50",
        help="Comma-separated page numbers to test (default: 5,21,50)",
    )
    parser.add_argument(
        "--skip-layout",
        action="store_true",
        help="Skip Docling layout detection (for speed)",
    )

    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"❌ Error: PDF not found: {args.pdf_path}")
        sys.exit(1)

    pages = [int(p.strip()) for p in args.pages.split(",")]

    print(f"⏱️  Processing Time Validation")
    print(f"   PDF: {args.pdf_path}")
    print(f"   Pages: {pages}")
    print(f"   Layout detection: {'disabled' if args.skip_layout else 'enabled'}")

    results = []
    for page_num in pages:
        try:
            result = time_full_pipeline(args.pdf_path, page_num)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error on page {page_num}: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print(f"TIMING SUMMARY")
    print(f"{'='*60}")

    if results:
        avg_render = np.mean([r.render_ms for r in results])
        avg_preprocess = np.mean([r.preprocess_ms for r in results])
        avg_ocr = np.mean([r.ocr_ms for r in results])
        avg_layout = np.mean([r.layout_ms for r in results])
        avg_total = np.mean([r.total_ms for r in results])

        print(f"\n   Average timings per page:")
        print(f"      Render:     {avg_render:>6.0f}ms")
        print(f"      Preprocess: {avg_preprocess:>6.0f}ms")
        print(f"      OCR:        {avg_ocr:>6.0f}ms")
        print(f"      Layout:     {avg_layout:>6.0f}ms")
        print(f"      ────────────────────")
        print(f"      TOTAL:      {avg_total:>6.0f}ms ({avg_total/1000:.1f}s)")

        # Breakdown percentages
        print(f"\n   Time breakdown:")
        print(f"      Render:     {avg_render/avg_total*100:>5.1f}%")
        print(f"      Preprocess: {avg_preprocess/avg_total*100:>5.1f}%")
        print(f"      OCR:        {avg_ocr/avg_total*100:>5.1f}%")
        print(f"      Layout:     {avg_layout/avg_total*100:>5.1f}%")

        # Final verdict
        if avg_total < 30000:
            print(f"\n✅ FINAL VERDICT: Processing time is acceptable (<30s/page)")
            print(f"   Estimated time for 300-page PDF: {300 * avg_total / 1000 / 60:.0f} minutes")
        elif avg_total < 60000:
            print(f"\n⚠️ FINAL VERDICT: Processing time is marginal (30-60s/page)")
            print(f"   Consider optimization before production use")
            print(f"   Estimated time for 300-page PDF: {300 * avg_total / 1000 / 60:.0f} minutes")
        else:
            print(f"\n❌ FINAL VERDICT: Processing time is TOO SLOW (>60s/page)")
            print(f"   MUST optimize before deployment")
            print(f"   Bottleneck appears to be: ", end="")
            bottleneck = max(
                [
                    ("Render", avg_render),
                    ("Preprocess", avg_preprocess),
                    ("OCR", avg_ocr),
                    ("Layout", avg_layout),
                ],
                key=lambda x: x[1],
            )
            print(f"{bottleneck[0]} ({bottleneck[1]:.0f}ms)")

    # Save results
    import json

    output_path = Path("tests/processing_time_results.json")
    results_dict = [
        {
            "page": r.page,
            "render_ms": r.render_ms,
            "preprocess_ms": r.preprocess_ms,
            "ocr_ms": r.ocr_ms,
            "layout_ms": r.layout_ms,
            "total_ms": r.total_ms,
            "page_size": r.page_size,
        }
        for r in results
    ]
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n   Results saved to: {output_path}")


if __name__ == "__main__":
    main()
