#!/usr/bin/env python3
"""
WEEK 0 - Test 2: Validate Tesseract OCR Baseline on Vintage Scans
=================================================================

This test measures Tesseract's raw OCR confidence on vintage scanned documents
to determine if Doctr fallback is needed.

GO/NO-GO Criteria:
- GO:    Average confidence >0.6 → Tesseract is sufficient
- NO-GO: Average confidence <0.5 → Need Doctr as Layer 3

Prerequisites:
    brew install tesseract  # macOS
    tesseract --version     # Should be 5.x

Usage:
    python tests/validate_tesseract_baseline.py data/raw/Firearms.pdf --pages 5,21,100

Author: Claude (Architect)
Date: January 3, 2026
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_tesseract_installation() -> bool:
    """Verify Tesseract is installed and return version."""
    import subprocess

    try:
        result = subprocess.run(
            ["tesseract", "--version"], capture_output=True, text=True, check=True
        )
        version_line = result.stdout.split("\n")[0]
        print(f"✅ Tesseract installed: {version_line}")

        # Check for version 5.x
        if "tesseract 5" in version_line.lower():
            print("   Version 5.x detected - optimal for this project")
            return True
        elif "tesseract 4" in version_line.lower():
            print("   ⚠️ Version 4.x detected - consider upgrading to 5.x")
            return True
        else:
            print("   ⚠️ Unknown version - may have compatibility issues")
            return True

    except FileNotFoundError:
        print("❌ Tesseract NOT installed!")
        print("   Install with: brew install tesseract")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Tesseract error: {e}")
        return False


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
    """
    Basic preprocessing for OCR.

    Pipeline:
    1. Grayscale conversion
    2. Adaptive thresholding
    """
    import cv2

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Adaptive threshold (better for old scans than global threshold)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=11, C=2
    )

    return binary


def run_tesseract_with_confidence(image: np.ndarray) -> Dict[str, Any]:
    """
    Run Tesseract OCR and extract per-word confidence scores.

    Returns:
        dict with:
        - text: str (full extracted text)
        - avg_confidence: float (0-100)
        - word_count: int
        - low_conf_words: List[str] (words with <50% confidence)
        - processing_time_ms: int
    """
    import pytesseract

    start_time = time.perf_counter()

    # Run Tesseract with detailed output
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config="--psm 3")

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)

    # Extract confidence scores (filter out -1 which means no text)
    confidences = []
    words = []
    low_conf_words = []

    for i, conf in enumerate(data["conf"]):
        conf_int = int(conf)
        text = data["text"][i].strip()

        if conf_int > 0 and text:  # Valid detection
            confidences.append(conf_int)
            words.append(text)

            if conf_int < 50:
                low_conf_words.append(f"{text}({conf_int}%)")

    # Calculate statistics
    avg_confidence = np.mean(confidences) if confidences else 0.0
    full_text = " ".join(words)

    return {
        "text": full_text,
        "avg_confidence": avg_confidence,
        "word_count": len(words),
        "low_conf_words": low_conf_words[:20],  # First 20
        "processing_time_ms": elapsed_ms,
        "confidence_distribution": {
            "0-25%": sum(1 for c in confidences if c < 25),
            "25-50%": sum(1 for c in confidences if 25 <= c < 50),
            "50-75%": sum(1 for c in confidences if 50 <= c < 75),
            "75-100%": sum(1 for c in confidences if c >= 75),
        },
    }


def test_tesseract_on_page(pdf_path: Path, page_num: int, use_preprocessing: bool = True) -> dict:
    """Test Tesseract OCR on a single page."""

    print(f"\n{'='*60}")
    print(f"Testing Tesseract on page {page_num}")
    print(f"{'='*60}")

    # Render page
    print("   Rendering page...")
    image = render_page_to_image(pdf_path, page_num)
    print(f"   Page size: {image.shape[1]}x{image.shape[0]}px")

    # Test WITHOUT preprocessing
    print("\n   [RAW] Testing without preprocessing...")
    raw_result = run_tesseract_with_confidence(image)

    print(f"   RAW Results:")
    print(f"      Average confidence: {raw_result['avg_confidence']:.1f}%")
    print(f"      Words detected: {raw_result['word_count']}")
    print(f"      Processing time: {raw_result['processing_time_ms']}ms")
    print(f"      Confidence distribution: {raw_result['confidence_distribution']}")

    # Test WITH preprocessing
    if use_preprocessing:
        import cv2

        print("\n   [ENHANCED] Testing with preprocessing...")
        preprocessed = preprocess_image(image)
        enhanced_result = run_tesseract_with_confidence(preprocessed)

        print(f"   ENHANCED Results:")
        print(f"      Average confidence: {enhanced_result['avg_confidence']:.1f}%")
        print(f"      Words detected: {enhanced_result['word_count']}")
        print(f"      Processing time: {enhanced_result['processing_time_ms']}ms")
        print(f"      Confidence distribution: {enhanced_result['confidence_distribution']}")

        # Improvement calculation
        improvement = enhanced_result["avg_confidence"] - raw_result["avg_confidence"]
        print(f"\n   Preprocessing improvement: {improvement:+.1f}%")

    # GO/NO-GO assessment
    best_conf = (
        enhanced_result["avg_confidence"] if use_preprocessing else raw_result["avg_confidence"]
    )

    if best_conf >= 60:
        assessment = "✅ GO - Tesseract confidence is acceptable"
    elif best_conf >= 50:
        assessment = "⚠️ MARGINAL - Consider Doctr for problematic pages"
    else:
        assessment = "❌ NO-GO - Need Doctr as Layer 3 fallback"

    print(f"\n   Assessment: {assessment}")

    # Show sample text
    sample_text = (enhanced_result if use_preprocessing else raw_result)["text"][:200]
    print(f"\n   Sample text: {sample_text}...")

    # Show low confidence words
    low_conf = (enhanced_result if use_preprocessing else raw_result)["low_conf_words"]
    if low_conf:
        print(f"\n   ⚠️ Low confidence words: {', '.join(low_conf[:10])}")

    return {
        "page": page_num,
        "raw_confidence": raw_result["avg_confidence"],
        "enhanced_confidence": enhanced_result["avg_confidence"] if use_preprocessing else None,
        "word_count": raw_result["word_count"],
        "processing_time_ms": raw_result["processing_time_ms"],
        "assessment": assessment,
    }


def main():
    parser = argparse.ArgumentParser(description="Test Tesseract OCR baseline on scanned PDFs")
    parser.add_argument("pdf_path", type=Path, help="Path to PDF file")
    parser.add_argument(
        "--pages",
        type=str,
        default="5,21,50",
        help="Comma-separated page numbers to test (default: 5,21,50)",
    )
    parser.add_argument("--no-preprocess", action="store_true", help="Skip image preprocessing")

    args = parser.parse_args()

    # Check Tesseract installation
    print("🔍 Checking Tesseract installation...")
    if not check_tesseract_installation():
        sys.exit(1)

    if not args.pdf_path.exists():
        print(f"❌ Error: PDF not found: {args.pdf_path}")
        sys.exit(1)

    pages = [int(p.strip()) for p in args.pages.split(",")]

    print(f"\n🔍 Testing Tesseract OCR Baseline")
    print(f"   PDF: {args.pdf_path}")
    print(f"   Pages: {pages}")
    print(f"   Preprocessing: {'disabled' if args.no_preprocess else 'enabled'}")

    results = []
    for page_num in pages:
        try:
            result = test_tesseract_on_page(
                args.pdf_path, page_num, use_preprocessing=not args.no_preprocess
            )
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error on page {page_num}: {e}")
            import traceback

            traceback.print_exc()
            results.append({"page": page_num, "error": str(e), "assessment": "❌ ERROR"})

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")

    valid_results = [r for r in results if "enhanced_confidence" in r]

    if valid_results:
        avg_raw = np.mean([r["raw_confidence"] for r in valid_results])
        avg_enhanced = np.mean(
            [r["enhanced_confidence"] for r in valid_results if r["enhanced_confidence"]]
        )

        print(f"\n   Pages tested: {len(results)}")
        print(f"   Average RAW confidence: {avg_raw:.1f}%")
        print(f"   Average ENHANCED confidence: {avg_enhanced:.1f}%")

        go_count = sum(1 for r in results if "GO" in r.get("assessment", ""))
        print(f"   GO assessments: {go_count}/{len(results)}")

        if avg_enhanced >= 60:
            print(f"\n✅ FINAL VERDICT: Tesseract is sufficient for this corpus.")
            print(f"   Doctr fallback is optional (nice-to-have, not required).")
        elif avg_enhanced >= 50:
            print(f"\n⚠️ FINAL VERDICT: Tesseract works but consider Doctr for edge cases.")
        else:
            print(f"\n❌ FINAL VERDICT: Tesseract is insufficient. Doctr Layer 3 is REQUIRED.")

    # Save results
    import json

    output_path = Path("tests/tesseract_baseline_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n   Results saved to: {output_path}")


if __name__ == "__main__":
    main()
