"""
Testing script for Layout-Aware OCR pipeline.

Tests the complete pipeline on sample pages and generates comparison reports.

Usage:
    # Test single page
    python scripts/test_layout_aware_ocr.py \
        --pdf vintage_catalog.pdf \
        --page 21 \
        --output ./test_results

    # Test multiple pages
    python scripts/test_layout_aware_ocr.py \
        --pdf vintage_catalog.pdf \
        --pages 5,10,21,50 \
        --output ./test_results
        
    # Batch test on directory
    python scripts/test_layout_aware_ocr.py \
        --pdf-dir ./test_pdfs \
        --sample-pages 5 \
        --output ./batch_results
"""

import sys
from pathlib import Path
import json
import time
from typing import List, Dict
import logging

import click
import pandas as pd
import numpy as np
import cv2
import fitz  # PyMuPDF

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mmrag_v2.ocr.layout_aware_processor import LayoutAwareOCRProcessor
from mmrag_v2.ocr.enhanced_ocr_engine import EnhancedOCREngine

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def render_page(pdf_path: Path, page_num: int, dpi: int = 300) -> np.ndarray:
    """Render PDF page to image."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    doc.close()
    return img


def run_single_page(
    pdf_path: Path,
    page_num: int,
    output_dir: Path,
    enable_doctr: bool = True,
) -> Dict:
    """
    Test layout-aware OCR on a single page.

    Returns:
        Dictionary with test results
    """
    logger.info(f"Testing page {page_num} from {pdf_path.name}")

    # Render page
    page_image = render_page(pdf_path, page_num)
    logger.info(f"Rendered page: {page_image.shape}")

    # Initialize processor
    processor = LayoutAwareOCRProcessor(
        ocr_confidence_threshold=0.7, enable_doctr=enable_doctr, output_dir=output_dir / "assets"
    )

    # Process page
    start_time = time.time()

    chunks = processor.process_page(
        page_image=page_image,
        page_number=page_num + 1,
        doc_id="test_doc",
    )

    processing_time = time.time() - start_time

    # Analyze results
    text_chunks = [c for c in chunks if c.modality == "text"]
    image_chunks = [c for c in chunks if c.modality == "image"]
    table_chunks = [c for c in chunks if c.modality == "table"]

    # Calculate average OCR confidence
    ocr_confidences = [c.ocr_confidence for c in chunks if c.ocr_confidence is not None]
    avg_confidence = np.mean(ocr_confidences) if ocr_confidences else 0.0

    # Count OCR layers used
    layer_counts = {}
    for chunk in chunks:
        if chunk.ocr_layer:
            layer_counts[chunk.ocr_layer] = layer_counts.get(chunk.ocr_layer, 0) + 1

    results = {
        "pdf": pdf_path.name,
        "page": page_num,
        "processing_time_seconds": round(processing_time, 2),
        "total_chunks": len(chunks),
        "text_chunks": len(text_chunks),
        "image_chunks": len(image_chunks),
        "table_chunks": len(table_chunks),
        "avg_ocr_confidence": round(avg_confidence, 3),
        "ocr_layer_distribution": layer_counts,
        "total_text_chars": sum(len(c.content) for c in text_chunks),
    }

    # Save chunks to JSON
    chunks_file = output_dir / f"page_{page_num}_chunks.json"
    with open(chunks_file, "w") as f:
        json.dump(
            [
                {
                    "chunk_id": c.chunk_id,
                    "modality": c.modality,
                    "content": c.content[:200] + "..." if len(c.content) > 200 else c.content,
                    "bbox": c.bbox,
                    "ocr_confidence": c.ocr_confidence,
                    "ocr_layer": c.ocr_layer,
                }
                for c in chunks
            ],
            f,
            indent=2,
        )

    logger.info(f"Results: {results}")
    logger.info(f"Saved chunks to {chunks_file}")

    return results


def run_multiple_pages(
    pdf_path: Path,
    page_numbers: List[int],
    output_dir: Path,
    enable_doctr: bool = True,
) -> pd.DataFrame:
    """Test multiple pages and generate comparison report."""

    results = []

    for page_num in page_numbers:
        try:
            result = run_single_page(pdf_path, page_num, output_dir, enable_doctr)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed on page {page_num}: {e}")
            results.append({"pdf": pdf_path.name, "page": page_num, "error": str(e)})

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save report
    report_file = output_dir / f"{pdf_path.stem}_report.csv"
    df.to_csv(report_file, index=False)
    logger.info(f"Saved report to {report_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(df.describe())
    print("\n")
    print("OCR Layer Distribution:")
    print(df["ocr_layer_distribution"].value_counts())

    return df


def compare_with_baseline(
    test_results: pd.DataFrame,
    baseline_confidence: float = 0.4,  # Typical for current pipeline
) -> None:
    """Compare new results with baseline confidence."""

    avg_new_confidence = test_results["avg_ocr_confidence"].mean()
    improvement = ((avg_new_confidence - baseline_confidence) / baseline_confidence) * 100

    print("\n" + "=" * 80)
    print("COMPARISON WITH BASELINE")
    print("=" * 80)
    print(f"Baseline avg confidence:  {baseline_confidence:.2f}")
    print(f"New avg confidence:       {avg_new_confidence:.2f}")
    print(f"Improvement:              +{improvement:.1f}%")
    print(
        f"Target (0.75-0.85):       {'✓ ACHIEVED' if avg_new_confidence >= 0.75 else '✗ NOT YET'}"
    )
    print("=" * 80 + "\n")


@click.command()
@click.option("--pdf", type=click.Path(exists=True), help="PDF file to test")
@click.option("--pdf-dir", type=click.Path(exists=True), help="Directory of PDFs to batch test")
@click.option("--page", type=int, help="Single page number to test (0-indexed)")
@click.option("--pages", type=str, help='Comma-separated page numbers (e.g., "5,10,21,50")')
@click.option(
    "--sample-pages",
    type=int,
    default=5,
    help="Number of random pages to sample per PDF (for batch mode)",
)
@click.option(
    "--output",
    type=click.Path(),
    default="./test_results",
    help="Output directory for test results",
)
@click.option("--no-doctr", is_flag=True, help="Disable Doctr Layer 3 (test Tesseract only)")
@click.option(
    "--baseline-confidence", type=float, default=0.4, help="Baseline confidence for comparison"
)
def main(
    pdf,
    pdf_dir,
    page,
    pages,
    sample_pages,
    output,
    no_doctr,
    baseline_confidence,
):
    """Test Layout-Aware OCR pipeline."""

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    enable_doctr = not no_doctr

    if pdf:
        pdf_path = Path(pdf)

        if page is not None:
            # Single page test
            result = run_single_page(pdf_path, page, output_dir, enable_doctr)
            print(json.dumps(result, indent=2))

        elif pages:
            # Multiple specific pages
            page_numbers = [int(p.strip()) for p in pages.split(",")]
            df = run_multiple_pages(pdf_path, page_numbers, output_dir, enable_doctr)
            compare_with_baseline(df, baseline_confidence)

        else:
            # Sample random pages
            import fitz

            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()

            page_numbers = np.random.choice(
                total_pages, min(sample_pages, total_pages), replace=False
            ).tolist()

            logger.info(f"Testing random pages: {page_numbers}")
            df = run_multiple_pages(pdf_path, page_numbers, output_dir, enable_doctr)
            compare_with_baseline(df, baseline_confidence)

    elif pdf_dir:
        # Batch test on directory
        pdf_dir_path = Path(pdf_dir)
        pdf_files = list(pdf_dir_path.glob("*.pdf"))

        logger.info(f"Found {len(pdf_files)} PDFs in {pdf_dir}")

        all_results = []

        for pdf_path in pdf_files:
            logger.info(f"\nTesting {pdf_path.name}")

            import fitz

            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()

            page_numbers = np.random.choice(
                total_pages, min(sample_pages, total_pages), replace=False
            ).tolist()

            df = run_multiple_pages(pdf_path, page_numbers, output_dir, enable_doctr)

            all_results.append(df)

        # Combine all results
        combined_df = pd.concat(all_results, ignore_index=True)

        combined_report = output_dir / "combined_report.csv"
        combined_df.to_csv(combined_report, index=False)

        logger.info(f"\nSaved combined report to {combined_report}")

        compare_with_baseline(combined_df, baseline_confidence)

    else:
        logger.error("Must specify either --pdf or --pdf-dir")
        sys.exit(1)


if __name__ == "__main__":
    main()
