"""
Test Script for BUG-009: Intelligence Metadata Propagation
===========================================================

This test verifies that intelligence metadata (profile_type, min_image_dims,
confidence_threshold, document_domain, document_modality, profile_sensitivity)
is NON-NULL in all JSONL chunks for both process and batch execution paths.

Usage:
    python tests/test_bug009_metadata_propagation.py

Expected Result:
    - All chunks should have non-null intelligence metadata
    - No "null" values should appear in the 6 metadata fields
    - Both text, image, and table chunks should contain metadata

Author: Claude 4.5 Opus (BUG-009 Fix)
Date: 2026-01-16
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


def check_intelligence_metadata(jsonl_path: Path) -> Dict[str, any]:
    """
    Check JSONL file for null intelligence metadata.

    Args:
        jsonl_path: Path to ingestion.jsonl file

    Returns:
        Dict with test results
    """
    if not jsonl_path.exists():
        return {
            "success": False,
            "error": f"JSONL file not found: {jsonl_path}",
        }

    # Intelligence metadata keys per V2.4 schema
    INTELLIGENCE_KEYS = [
        "profile_type",
        "profile_sensitivity",
        "min_image_dims",
        "confidence_threshold",
        "document_domain",
        "document_modality",
    ]

    total_chunks = 0
    chunks_with_null_metadata = []
    metadata_stats = {key: {"null_count": 0, "non_null_count": 0} for key in INTELLIGENCE_KEYS}
    modality_breakdown = {"text": 0, "image": 0, "table": 0}

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    chunk = json.loads(line)
                    total_chunks += 1

                    # Track modality
                    modality = chunk.get("modality", "unknown")
                    if modality in modality_breakdown:
                        modality_breakdown[modality] += 1

                    # Check metadata
                    metadata = chunk.get("metadata", {})
                    chunk_id = chunk.get("chunk_id", f"line_{line_num}")

                    null_fields = []
                    for key in INTELLIGENCE_KEYS:
                        value = metadata.get(key)
                        if value is None:
                            metadata_stats[key]["null_count"] += 1
                            null_fields.append(key)
                        else:
                            metadata_stats[key]["non_null_count"] += 1

                    if null_fields:
                        chunks_with_null_metadata.append(
                            {
                                "chunk_id": chunk_id,
                                "line": line_num,
                                "modality": modality,
                                "null_fields": null_fields,
                            }
                        )

                except json.JSONDecodeError as e:
                    print(f"⚠️  [PARSE ERROR] Line {line_num}: {e}", file=sys.stderr)
                    continue

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to read JSONL: {e}",
        }

    # Determine test result
    has_null_metadata = len(chunks_with_null_metadata) > 0

    return {
        "success": not has_null_metadata,
        "total_chunks": total_chunks,
        "chunks_with_null_metadata": len(chunks_with_null_metadata),
        "null_metadata_details": chunks_with_null_metadata[:10],  # First 10 examples
        "metadata_stats": metadata_stats,
        "modality_breakdown": modality_breakdown,
    }


def print_test_results(results: Dict[str, any], jsonl_path: Path) -> None:
    """Print formatted test results."""
    print("\n" + "=" * 70)
    print("BUG-009 METADATA PROPAGATION TEST")
    print("=" * 70)
    print(f"File: {jsonl_path}")
    print("-" * 70)

    if not results.get("success"):
        print(f"❌ TEST FAILED: {results.get('error', 'Unknown error')}")
        return

    total = results["total_chunks"]
    null_count = results["chunks_with_null_metadata"]

    if null_count == 0:
        print(f"✅ TEST PASSED: All {total} chunks have complete intelligence metadata")
    else:
        print(f"❌ TEST FAILED: {null_count}/{total} chunks have null metadata fields")

    print("-" * 70)
    print("MODALITY BREAKDOWN:")
    for modality, count in results["modality_breakdown"].items():
        print(f"  • {modality}: {count} chunks")

    print("-" * 70)
    print("METADATA FIELD STATISTICS:")
    for key, stats in results["metadata_stats"].items():
        null_pct = (stats["null_count"] / total * 100) if total > 0 else 0
        status = "✅" if stats["null_count"] == 0 else "❌"
        print(
            f"  {status} {key}: {stats['non_null_count']} non-null, {stats['null_count']} null ({null_pct:.1f}%)"
        )

    if null_count > 0:
        print("-" * 70)
        print("NULL METADATA EXAMPLES (first 10):")
        for detail in results["null_metadata_details"]:
            print(
                f"  • Line {detail['line']} ({detail['modality']}): {', '.join(detail['null_fields'])}"
            )
            print(f"    chunk_id: {detail['chunk_id'][:50]}...")

    print("=" * 70 + "\n")


def main():
    """Main test entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Test BUG-009: Intelligence Metadata Propagation")
    parser.add_argument(
        "jsonl_path",
        type=str,
        nargs="?",
        default="output/ingestion.jsonl",
        help="Path to ingestion.jsonl file (default: output/ingestion.jsonl)",
    )
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl_path)

    print(f"\n🔍 Testing intelligence metadata propagation...")
    print(f"📁 Target: {jsonl_path}")

    results = check_intelligence_metadata(jsonl_path)
    print_test_results(results, jsonl_path)

    # Exit with appropriate code
    sys.exit(0 if results.get("success") else 1)


if __name__ == "__main__":
    main()
