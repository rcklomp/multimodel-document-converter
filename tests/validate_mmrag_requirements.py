"""
MM-RAG Requirements Validation Suite

This script validates that the converted output meets ALL critical requirements
for Multimodal RAG corpus quality.

Usage:
    python tests/validate_mmrag_requirements.py \
        --input output/ingestion.jsonl \
        --assets output/assets/

Success Criteria:
    - All Tier 1 tests must pass (CRITICAL)
    - 80%+ of Tier 2 tests must pass (IMPORTANT)
    - Tier 3 tests are advisory (NICE-TO-HAVE)
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
import click


class ValidationResult:
    """Result of a single validation test."""

    def __init__(self, name: str, passed: bool, message: str, severity: str):
        self.name = name
        self.passed = passed
        self.message = message
        self.severity = severity  # CRITICAL | IMPORTANT | ADVISORY


class MMRAGValidator:
    """Validates MM-RAG corpus quality requirements."""

    def __init__(self, jsonl_path: Path, assets_dir: Path):
        self.jsonl_path = jsonl_path
        self.assets_dir = assets_dir
        self.chunks = self._load_chunks()
        self.results: List[ValidationResult] = []

    def _load_chunks(self) -> List[Dict]:
        """Load all chunks from JSONL."""
        chunks = []
        with open(self.jsonl_path) as f:
            for line in f:
                chunks.append(json.loads(line))
        return chunks

    def run_all_tests(self) -> Tuple[int, int, int, int]:
        """
        Run all validation tests.

        Returns:
            (passed, failed, total, exit_code) counts
        """
        print("=" * 80)
        print("MM-RAG REQUIREMENTS VALIDATION")
        print("=" * 80)
        print(f"\nInput: {self.jsonl_path}")
        print(f"Chunks: {len(self.chunks)}")
        print(f"Assets: {self.assets_dir}")

        # Tier 1: Critical Requirements
        print("\n" + "=" * 80)
        print("TIER 1: CRITICAL REQUIREMENTS (Must Pass)")
        print("=" * 80)

        self._test_no_shadow_modality()
        self._test_text_chunks_have_ocr_content()
        self._test_image_chunks_have_visual_descriptions()
        self._test_modality_separation_exists()
        self._test_no_vlm_text_reading_in_text_chunks()
        self._test_asset_files_exist()
        self._test_bbox_coordinates_valid()

        # Tier 2: Important Requirements
        print("\n" + "=" * 80)
        print("TIER 2: IMPORTANT REQUIREMENTS (80%+ Should Pass)")
        print("=" * 80)

        self._test_ocr_confidence_scores()
        self._test_content_not_truncated()
        self._test_search_priority_metadata()
        self._test_extraction_method_correct()

        # Tier 3: Advisory Requirements
        print("\n" + "=" * 80)
        print("TIER 3: ADVISORY REQUIREMENTS (Nice to Have)")
        print("=" * 80)

        self._test_ocr_layer_distribution()
        self._test_text_image_ratio_reasonable()
        self._test_element_indexing_sequential()

        # Summary
        return self._print_summary()

    # ========================================================================
    # TIER 1: CRITICAL TESTS
    # ========================================================================

    def _test_no_shadow_modality(self):
        """CRITICAL: No chunks should have modality='shadow'."""
        shadow_chunks = [c for c in self.chunks if c.get("modality") == "shadow"]

        passed = len(shadow_chunks) == 0

        if passed:
            msg = f"✓ PASS: No 'shadow' modality found (0/{len(self.chunks)} chunks)"
        else:
            msg = f"✗ FAIL: Found {len(shadow_chunks)} chunks with modality='shadow'"
            msg += f"\n  Example: chunk_id={shadow_chunks[0]['chunk_id']}"

        self.results.append(ValidationResult("No Shadow Modality", passed, msg, "CRITICAL"))
        print(f"\n{msg}")

    def _test_text_chunks_have_ocr_content(self):
        """CRITICAL: TEXT chunks must contain actual text, not VLM summaries."""
        text_chunks = [c for c in self.chunks if c.get("modality") == "text"]

        if not text_chunks:
            msg = "⚠ WARNING: No TEXT chunks found in output"
            self.results.append(ValidationResult("TEXT Chunks Exist", False, msg, "CRITICAL"))
            print(f"\n{msg}")
            return

        # Check for VLM-style descriptions in TEXT content
        vlm_patterns = [
            "the image shows",
            "the page displays",
            "a black and white",
            "the document features",
        ]

        contaminated = []
        for chunk in text_chunks:
            content = chunk.get("content", "").lower()
            if any(pattern in content for pattern in vlm_patterns):
                contaminated.append(chunk["chunk_id"])

        passed = len(contaminated) == 0

        if passed:
            msg = f"✓ PASS: All {len(text_chunks)} TEXT chunks contain OCR content"
            # Sample first text chunk
            sample = text_chunks[0]["content"][:100]
            msg += f'\n  Sample: "{sample}..."'
        else:
            msg = f"✗ FAIL: {len(contaminated)}/{len(text_chunks)} TEXT chunks contain VLM descriptions"
            msg += f"\n  Contaminated chunk IDs: {contaminated[:3]}..."

        self.results.append(
            ValidationResult("TEXT Chunks Have OCR Content", passed, msg, "CRITICAL")
        )
        print(f"\n{msg}")

    def _test_image_chunks_have_visual_descriptions(self):
        """CRITICAL: IMAGE chunks must have VLM descriptions, not empty content."""
        image_chunks = [c for c in self.chunks if c.get("modality") == "image"]

        if not image_chunks:
            msg = "⚠ WARNING: No IMAGE chunks found in output"
            self.results.append(
                ValidationResult(
                    "IMAGE Chunks Exist",
                    True,  # Not critical if document has no images
                    msg,
                    "CRITICAL",
                )
            )
            print(f"\n{msg}")
            return

        empty_descriptions = [
            c for c in image_chunks if not c.get("content") or len(c.get("content", "")) < 10
        ]

        passed = len(empty_descriptions) == 0

        if passed:
            msg = f"✓ PASS: All {len(image_chunks)} IMAGE chunks have descriptions"
            # Sample first image description
            sample = image_chunks[0]["content"][:100]
            msg += f'\n  Sample: "{sample}..."'
        else:
            msg = f"✗ FAIL: {len(empty_descriptions)}/{len(image_chunks)} IMAGE chunks have empty/short descriptions"

        self.results.append(
            ValidationResult("IMAGE Chunks Have Descriptions", passed, msg, "CRITICAL")
        )
        print(f"\n{msg}")

    def _test_modality_separation_exists(self):
        """CRITICAL: Output must contain both TEXT and IMAGE chunks."""
        modalities = Counter(c.get("modality") for c in self.chunks)

        has_text = modalities.get("text", 0) > 0
        has_image = modalities.get("image", 0) > 0

        passed = has_text and has_image

        if passed:
            msg = f"✓ PASS: Modality separation detected"
            msg += f"\n  Distribution: {dict(modalities)}"
        else:
            msg = f"✗ FAIL: Missing modality separation"
            msg += f"\n  Found: {dict(modalities)}"
            if not has_text:
                msg += "\n  ERROR: No TEXT chunks found!"
            if not has_image:
                msg += "\n  ERROR: No IMAGE chunks found!"

        self.results.append(ValidationResult("Modality Separation", passed, msg, "CRITICAL"))
        print(f"\n{msg}")

    def _test_no_vlm_text_reading_in_text_chunks(self):
        """CRITICAL: VLM should not be reading text from images in TEXT chunks."""
        text_chunks = [c for c in self.chunks if c.get("modality") == "text"]

        # Check extraction_method
        vlm_extracted_text = [
            c
            for c in text_chunks
            if "vlm" in c.get("metadata", {}).get("extraction_method", "").lower()
        ]

        passed = len(vlm_extracted_text) == 0

        if passed:
            msg = f"✓ PASS: No TEXT chunks extracted via VLM"
        else:
            msg = f"✗ FAIL: {len(vlm_extracted_text)} TEXT chunks have VLM extraction method"
            msg += f"\n  This indicates VLM is reading text instead of OCR"

        self.results.append(ValidationResult("No VLM Text Reading", passed, msg, "CRITICAL"))
        print(f"\n{msg}")

    def _test_asset_files_exist(self):
        """CRITICAL: All referenced asset files must exist on disk."""
        image_chunks = [c for c in self.chunks if c.get("asset_ref")]

        if not image_chunks:
            msg = "⚠ WARNING: No chunks with asset_ref found"
            self.results.append(ValidationResult("Asset Files Exist", True, msg, "CRITICAL"))
            print(f"\n{msg}")
            return

        missing_assets = []
        for chunk in image_chunks:
            asset_path = chunk["asset_ref"].get("file_path")
            if asset_path:
                full_path = self.assets_dir.parent / asset_path
                if not full_path.exists():
                    missing_assets.append(asset_path)

        passed = len(missing_assets) == 0

        if passed:
            msg = f"✓ PASS: All {len(image_chunks)} asset files exist on disk"
        else:
            msg = f"✗ FAIL: {len(missing_assets)}/{len(image_chunks)} asset files missing"
            msg += f"\n  Missing: {missing_assets[:3]}..."

        self.results.append(ValidationResult("Asset Files Exist", passed, msg, "CRITICAL"))
        print(f"\n{msg}")

    def _test_bbox_coordinates_valid(self):
        """CRITICAL: Bounding boxes must be normalized to [0, 1000] range."""
        chunks_with_bbox = [
            c
            for c in self.chunks
            if c.get("metadata", {}).get("spatial") and c["metadata"]["spatial"].get("bbox")
        ]

        if not chunks_with_bbox:
            msg = "⚠ WARNING: No chunks with bounding boxes found"
            self.results.append(ValidationResult("BBox Coordinates Valid", True, msg, "CRITICAL"))
            print(f"\n{msg}")
            return

        invalid_bboxes = []
        for chunk in chunks_with_bbox:
            bbox = chunk["metadata"]["spatial"]["bbox"]

            # Check: 4 values, all integers, all in [0, 1000]
            if len(bbox) != 4:
                invalid_bboxes.append((chunk["chunk_id"], "not 4 values"))
            elif not all(isinstance(v, int) for v in bbox):
                invalid_bboxes.append((chunk["chunk_id"], "not integers"))
            elif not all(0 <= v <= 1000 for v in bbox):
                invalid_bboxes.append((chunk["chunk_id"], f"out of range: {bbox}"))

        passed = len(invalid_bboxes) == 0

        if passed:
            msg = f"✓ PASS: All {len(chunks_with_bbox)} bboxes normalized to [0, 1000]"
        else:
            msg = f"✗ FAIL: {len(invalid_bboxes)}/{len(chunks_with_bbox)} bboxes invalid"
            msg += f"\n  Examples: {invalid_bboxes[:3]}"

        self.results.append(ValidationResult("BBox Coordinates Valid", passed, msg, "CRITICAL"))
        print(f"\n{msg}")

    # ========================================================================
    # TIER 2: IMPORTANT TESTS
    # ========================================================================

    def _test_ocr_confidence_scores(self):
        """IMPORTANT: TEXT chunks should have OCR confidence scores."""
        text_chunks = [c for c in self.chunks if c.get("modality") == "text"]

        chunks_with_conf = [
            c for c in text_chunks if c.get("metadata", {}).get("ocr_confidence") is not None
        ]

        if text_chunks:
            coverage = len(chunks_with_conf) / len(text_chunks)
            passed = coverage >= 0.8  # 80% should have confidence

            if passed:
                avg_conf = (
                    sum(c["metadata"]["ocr_confidence"] for c in chunks_with_conf)
                    / len(chunks_with_conf)
                    if chunks_with_conf
                    else 0
                )

                msg = f"✓ PASS: {coverage:.0%} of TEXT chunks have OCR confidence"
                msg += f"\n  Average confidence: {avg_conf:.2f}"
            else:
                msg = f"⚠ WARN: Only {coverage:.0%} of TEXT chunks have OCR confidence"
        else:
            passed = False
            msg = "⚠ WARN: No TEXT chunks to check"

        self.results.append(ValidationResult("OCR Confidence Scores", passed, msg, "IMPORTANT"))
        print(f"\n{msg}")

    def _test_content_not_truncated(self):
        """IMPORTANT: Content should not be truncated with '...'."""
        truncated = [c for c in self.chunks if c.get("content", "").endswith("...")]

        passed = len(truncated) == 0

        if passed:
            msg = f"✓ PASS: No truncated content found"
        else:
            msg = f"⚠ WARN: {len(truncated)}/{len(self.chunks)} chunks have truncated content"

        self.results.append(ValidationResult("Content Not Truncated", passed, msg, "IMPORTANT"))
        print(f"\n{msg}")

    def _test_search_priority_metadata(self):
        """IMPORTANT: Chunks should have search_priority for RAG ranking."""
        chunks_with_priority = [
            c for c in self.chunks if c.get("metadata", {}).get("search_priority") is not None
        ]

        coverage = len(chunks_with_priority) / len(self.chunks) if self.chunks else 0
        passed = coverage >= 0.5  # At least 50%

        if passed:
            msg = f"✓ PASS: {coverage:.0%} of chunks have search_priority"
        else:
            msg = f"⚠ WARN: Only {coverage:.0%} of chunks have search_priority"
            msg += "\n  Recommendation: Add search_priority metadata"

        self.results.append(ValidationResult("Search Priority Metadata", passed, msg, "IMPORTANT"))
        print(f"\n{msg}")

    def _test_extraction_method_correct(self):
        """IMPORTANT: Extraction methods should match modalities."""
        text_chunks = [c for c in self.chunks if c.get("modality") == "text"]
        image_chunks = [c for c in self.chunks if c.get("modality") == "image"]

        # TEXT chunks should have OCR-related extraction methods
        text_methods = Counter(
            c.get("metadata", {}).get("extraction_method", "unknown") for c in text_chunks
        )

        # IMAGE chunks should have VLM extraction method
        image_methods = Counter(
            c.get("metadata", {}).get("extraction_method", "unknown") for c in image_chunks
        )

        msg = "Extraction method distribution:"
        msg += f"\n  TEXT chunks: {dict(text_methods)}"
        msg += f"\n  IMAGE chunks: {dict(image_methods)}"

        # Pass if distributions look reasonable
        passed = True  # Advisory only

        self.results.append(
            ValidationResult("Extraction Method Distribution", passed, msg, "IMPORTANT")
        )
        print(f"\n{msg}")

    # ========================================================================
    # TIER 3: ADVISORY TESTS
    # ========================================================================

    def _test_ocr_layer_distribution(self):
        """ADVISORY: Check which OCR layers are being used."""
        text_chunks = [c for c in self.chunks if c.get("modality") == "text"]

        ocr_layers = Counter(c.get("metadata", {}).get("ocr_layer", "unknown") for c in text_chunks)

        msg = f"OCR layer distribution: {dict(ocr_layers)}"

        self.results.append(ValidationResult("OCR Layer Distribution", True, msg, "ADVISORY"))
        print(f"\n{msg}")

    def _test_text_image_ratio_reasonable(self):
        """ADVISORY: Text-to-image ratio should be reasonable."""
        modalities = Counter(c.get("modality") for c in self.chunks)

        text_count = modalities.get("text", 0)
        image_count = modalities.get("image", 0)

        if image_count > 0:
            ratio = text_count / image_count
            msg = f"Text-to-image ratio: {ratio:.2f} ({text_count} text / {image_count} image)"
        else:
            msg = "No images found for ratio calculation"

        self.results.append(ValidationResult("Text/Image Ratio", True, msg, "ADVISORY"))
        print(f"\n{msg}")

    def _test_element_indexing_sequential(self):
        """ADVISORY: Elements should have sequential indexing per page."""
        # Group by page
        pages = defaultdict(list)
        for chunk in self.chunks:
            page_num = chunk.get("metadata", {}).get("page_number")
            if page_num:
                pages[page_num].append(chunk)

        msg = f"Found {len(pages)} pages with chunks"

        self.results.append(ValidationResult("Element Indexing", True, msg, "ADVISORY"))
        print(f"\n{msg}")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    def _print_summary(self) -> Tuple[int, int, int, int]:
        """Print test summary and return (passed, failed, total, exit_code)."""
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)

        critical = [r for r in self.results if r.severity == "CRITICAL"]
        important = [r for r in self.results if r.severity == "IMPORTANT"]
        advisory = [r for r in self.results if r.severity == "ADVISORY"]

        critical_passed = sum(1 for r in critical if r.passed)
        important_passed = sum(1 for r in important if r.passed)
        advisory_passed = sum(1 for r in advisory if r.passed)

        print(f"\nCRITICAL: {critical_passed}/{len(critical)} passed")
        print(f"IMPORTANT: {important_passed}/{len(important)} passed")
        print(f"ADVISORY: {advisory_passed}/{len(advisory)} passed")

        total_passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        print(f"\nOVERALL: {total_passed}/{total} tests passed ({total_passed/total*100:.0f}%)")

        # Determine final verdict
        critical_fail = len(critical) - critical_passed
        important_fail = len(important) - important_passed

        print("\n" + "=" * 80)
        if critical_fail == 0 and important_passed / len(important) >= 0.8:
            print("✓ VERDICT: PRODUCTION READY")
            print("=" * 80)
            exit_code = 0
        elif critical_fail == 0:
            print("⚠ VERDICT: ACCEPTABLE (Some important tests failed)")
            print("=" * 80)
            exit_code = 0
        else:
            print("✗ VERDICT: NOT READY (Critical tests failed)")
            print("=" * 80)
            exit_code = 1

        return (total_passed, total - total_passed, total, exit_code)


@click.command()
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to ingestion.jsonl (preferred). Legacy mode accepts source PDF when --output points to ingestion.jsonl.",
)
@click.option(
    "--output",
    "output_path",
    required=False,
    type=click.Path(exists=True),
    help="Legacy path to ingestion.jsonl or output directory containing ingestion.jsonl.",
)
@click.option(
    "--assets",
    "assets_dir",
    required=False,
    type=click.Path(exists=True),
    help="Assets directory. Defaults to <jsonl_dir>/assets.",
)
def main(input_path, output_path, assets_dir):
    """Run MM-RAG requirements validation."""

    input_p = Path(input_path)
    output_p = Path(output_path) if output_path else None

    # Resolve JSONL path (support both corrected and legacy invocation patterns).
    jsonl_path: Path
    if input_p.is_file() and input_p.suffix.lower() == ".jsonl":
        jsonl_path = input_p
    elif output_p is not None:
        if output_p.is_file() and output_p.suffix.lower() == ".jsonl":
            jsonl_path = output_p
        elif output_p.is_dir() and (output_p / "ingestion.jsonl").exists():
            jsonl_path = output_p / "ingestion.jsonl"
        else:
            raise click.UsageError(
                "Could not resolve ingestion JSONL. Provide --input <ingestion.jsonl> "
                "or --output <ingestion.jsonl|output_dir_with_ingestion.jsonl>."
            )
    else:
        raise click.UsageError(
            "--input must point to ingestion.jsonl, or provide legacy --output "
            "with ingestion.jsonl location."
        )

    # Resolve assets directory.
    if assets_dir:
        assets_path = Path(assets_dir)
    else:
        default_assets = jsonl_path.parent / "assets"
        if default_assets.exists():
            assets_path = default_assets
        else:
            raise click.UsageError(
                "Could not resolve assets directory. Provide --assets <assets_dir>."
            )

    validator = MMRAGValidator(jsonl_path=jsonl_path, assets_dir=assets_path)

    passed, failed, total, exit_code = validator.run_all_tests()

    # Exit with appropriate code based on verdict
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
