#!/usr/bin/env python3
"""Machine-readable VLM quality summary from prompt-harness or production JSONL.

Reads harness output JSONL (from eval_vlm_image_prompt.py) or production
ingestion JSONL and reports Source Sanctity compliance metrics.

Usage:
    # From harness output:
    python scripts/vlm_quality_summary.py output/PCWorld_prompt_eval.jsonl

    # From production ingestion JSONL:
    python scripts/vlm_quality_summary.py output/Firearms/ingestion.jsonl --production

    # Machine-readable JSON output:
    python scripts/vlm_quality_summary.py output/eval.jsonl --json

    # Compare two runs:
    python scripts/vlm_quality_summary.py output/eval_v1.jsonl --baseline output/eval_v0.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow import from project source tree
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mmrag_v2.vision.vision_prompts import validate_vlm_response  # noqa: E402

# ---------------------------------------------------------------------------
# Sentinel / fallback patterns used in the enforcement pipeline
# ---------------------------------------------------------------------------
_HARD_FALLBACK_PREFIXES = (
    "Dense typographic layout",
    "[VLM_FAILED",
    "Visual element (VLM description unavailable)",
)
_PLACEHOLDER_PREFIXES = (
    "Image region from page",
    "[Figure on page",
    "Image extraction unavailable",
)


def _classify_description(desc: str) -> str:
    """Classify a VLM description by its final output value alone.

    Used for production JSONL (no raw/issues available).

    Returns one of:
        clean_or_sanitized — passed validation (may include sanitized; use _classify_harness_record for distinction)
        hard_fallback      — fell back to generic sentinel
        placeholder        — no VLM was attempted (pending enrichment)
        empty              — empty or missing description
    """
    if not desc or not desc.strip():
        return "empty"
    for pfx in _PLACEHOLDER_PREFIXES:
        if desc.startswith(pfx):
            return "placeholder"
    for pfx in _HARD_FALLBACK_PREFIXES:
        if desc.startswith(pfx):
            return "hard_fallback"
    return "clean_or_sanitized"


def _classify_harness_record(r: dict[str, Any]) -> str:
    """Classify a harness record using raw + issues for full fidelity.

    Returns one of:
        clean         — passed validation on first attempt (raw == new, or no text-reading issues)
        sanitized     — salvaged by sanitizer (text-reading detected, then fixed)
        hard_fallback — fell back to generic sentinel despite retry/sanitize
        error         — VLM call failed entirely
        empty         — empty final description
    """
    if r.get("error"):
        return "error"
    desc = r.get("new_description", "")
    if not desc or not desc.strip():
        return "empty"
    for pfx in _HARD_FALLBACK_PREFIXES:
        if desc.startswith(pfx):
            return "hard_fallback"
    issues = r.get("validation_issues", [])
    text_reading_in_issues = any(
        "text" in i.lower() or "Text" in i for i in issues
    )
    raw = r.get("raw_description", "")
    # Sanitized: text-reading was detected AND final description differs from raw
    if text_reading_in_issues and raw and desc and raw != desc:
        return "sanitized"
    return "clean"


def _load_harness_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def _analyze_harness(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze prompt-harness output JSONL."""
    total = len(records)
    if total == 0:
        return {"total": 0, "error": "no records"}

    text_reading_hits = 0
    sanitizer_attempts = 0
    sanitizer_saves = 0
    errors = 0
    classifications: dict[str, int] = {}
    elapsed_values: list[float] = []
    per_source: dict[str, dict[str, int]] = {}

    for r in records:
        cls = _classify_harness_record(r)
        classifications[cls] = classifications.get(cls, 0) + 1

        issues = r.get("validation_issues", [])
        if any("text" in i.lower() or "Text" in i for i in issues):
            text_reading_hits += 1
        if r.get("error"):
            errors += 1
        if r.get("elapsed_s"):
            elapsed_values.append(r["elapsed_s"])

        # Sanitizer accounting: attempts = text-reading detected; saves = sanitized cls
        if any("text" in i.lower() or "Text" in i for i in issues):
            sanitizer_attempts += 1
        if cls == "sanitized":
            sanitizer_saves += 1

        desc = r.get("new_description", "")

        # Per-source breakdown
        source = r.get("chunk_id", "unknown")
        # Extract source document from blind-set chunk IDs or from metadata
        if source.startswith("blind_"):
            parts = source.split("_")
            src_key = "_".join(parts[1:-1]) if len(parts) > 2 else source
        else:
            src_key = source.rsplit("_", 3)[0] if "_" in source else source
        if src_key not in per_source:
            per_source[src_key] = {}
        per_source[src_key][cls] = per_source[src_key].get(cls, 0) + 1

    avg_elapsed = sum(elapsed_values) / len(elapsed_values) if elapsed_values else 0.0

    return {
        "total": total,
        "classifications": classifications,
        "text_reading_hit_rate": round(text_reading_hits / total, 4) if total else 0,
        "text_reading_hits": text_reading_hits,
        "sanitizer_attempts": sanitizer_attempts,
        "sanitizer_saves": sanitizer_saves,
        "sanitizer_success_rate": (
            round(sanitizer_saves / sanitizer_attempts, 4) if sanitizer_attempts else None
        ),
        "hard_fallback_count": classifications.get("hard_fallback", 0),
        "hard_fallback_rate": round(
            classifications.get("hard_fallback", 0) / total, 4
        ) if total else 0,
        "errors": errors,
        "avg_elapsed_s": round(avg_elapsed, 3),
    }


def _analyze_production(path: Path) -> dict[str, Any]:
    """Analyze production ingestion JSONL for VLM quality."""
    total_images = 0
    classifications: dict[str, int] = {}
    vision_statuses: dict[str, int] = {}
    text_reading_hits = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("record_type") == "metadata":
                continue
            if r.get("modality") != "image":
                continue
            total_images += 1

            desc = r.get("content", "")
            cls = _classify_description(desc)
            classifications[cls] = classifications.get(cls, 0) + 1

            meta = r.get("metadata", {})
            vs = meta.get("vision_status", "unknown")
            vision_statuses[vs] = vision_statuses.get(vs, 0) + 1

            # Check validation issues if present
            vi = meta.get("vision_validation_issues")
            if vi and any("text_reading" in str(i).lower() for i in vi):
                text_reading_hits += 1

            # Also re-validate the description to detect residual text reading.
            # cls is "clean_or_sanitized" here (production mode has no raw/issues).
            if cls == "clean_or_sanitized":
                result = validate_vlm_response(desc)
                if not result.is_valid and result.text_reading_detected:
                    text_reading_hits += 1

    if total_images == 0:
        return {"total": 0, "error": "no image chunks"}

    return {
        "total": total_images,
        "classifications": classifications,
        "vision_statuses": vision_statuses,
        "text_reading_hits": text_reading_hits,
        "text_reading_hit_rate": round(text_reading_hits / total_images, 4),
        "hard_fallback_count": classifications.get("hard_fallback", 0),
        "hard_fallback_rate": round(
            classifications.get("hard_fallback", 0) / total_images, 4
        ),
        "placeholder_count": classifications.get("placeholder", 0),
        "placeholder_rate": round(
            classifications.get("placeholder", 0) / total_images, 4
        ),
    }


def _print_table(report: dict[str, Any], label: str) -> None:
    """Print a human-readable summary table."""
    print(f"\n{'='*60}")
    print(f"  VLM Quality Summary: {label}")
    print(f"{'='*60}")

    total = report.get("total", 0)
    print(f"  Total images evaluated:     {total}")

    cls = report.get("classifications", {})
    for k in ("clean", "sanitized", "clean_or_sanitized", "hard_fallback", "placeholder", "empty", "error"):
        count = cls.get(k, 0)
        if count == 0 and k in ("clean_or_sanitized", "error"):
            continue  # suppress unused combined key in harness mode
        pct = f"({count/total*100:5.1f}%)" if total else ""
        print(f"  {k:28s} {count:5d}  {pct}")

    print(f"  {'---':28s}")

    if "text_reading_hits" in report:
        print(f"  Text-reading detections:    {report['text_reading_hits']}")
        print(f"  Text-reading hit rate:      {report.get('text_reading_hit_rate', 0):.1%}")

    if report.get("sanitizer_attempts") is not None:
        print(f"  Sanitizer attempts:         {report['sanitizer_attempts']}")
        print(f"  Sanitizer saves:            {report['sanitizer_saves']}")
        sr = report.get("sanitizer_success_rate")
        print(f"  Sanitizer success rate:     {sr:.1%}" if sr is not None else "  Sanitizer success rate:     n/a")

    if "hard_fallback_rate" in report:
        print(f"  Hard fallback rate:         {report['hard_fallback_rate']:.1%}")

    if "avg_elapsed_s" in report:
        print(f"  Avg elapsed per image:      {report['avg_elapsed_s']:.2f}s")

    if "vision_statuses" in report:
        print(f"\n  Vision status breakdown:")
        for k, v in sorted(report["vision_statuses"].items()):
            print(f"    {k:24s} {v:5d}")

    if "errors" in report:
        print(f"  VLM errors:                 {report['errors']}")

    print(f"{'='*60}\n")


def _compare(current: dict[str, Any], baseline: dict[str, Any]) -> None:
    """Print a delta comparison between two reports."""
    print(f"\n{'='*60}")
    print("  DELTA (current vs baseline)")
    print(f"{'='*60}")

    for key in ("text_reading_hit_rate", "hard_fallback_rate", "placeholder_rate"):
        cur = current.get(key, 0)
        base = baseline.get(key, 0)
        delta = cur - base
        arrow = "+" if delta > 0 else ""
        color = " WORSE" if delta > 0 else (" BETTER" if delta < 0 else "")
        print(f"  {key:30s}  {base:.1%} -> {cur:.1%}  ({arrow}{delta:.1%}){color}")

    cur_cls = current.get("classifications", {})
    base_cls = baseline.get("classifications", {})
    # Support both harness (clean/sanitized) and production (clean_or_sanitized) keys
    for k in ("clean", "sanitized", "clean_or_sanitized", "hard_fallback", "placeholder"):
        cur_n = cur_cls.get(k, 0)
        base_n = base_cls.get(k, 0)
        if cur_n == 0 and base_n == 0:
            continue
        delta = cur_n - base_n
        arrow = "+" if delta > 0 else ""
        print(f"  {k + ' (count)':30s}  {base_n} -> {cur_n}  ({arrow}{delta})")

    print(f"{'='*60}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="VLM Source Sanctity quality summary"
    )
    parser.add_argument("input_jsonl", type=Path,
                        help="Harness output JSONL or production ingestion JSONL")
    parser.add_argument("--production", action="store_true",
                        help="Treat input as production ingestion JSONL")
    parser.add_argument("--json", action="store_true",
                        help="Output machine-readable JSON instead of table")
    parser.add_argument("--baseline", type=Path, default=None,
                        help="Baseline JSONL for delta comparison")
    args = parser.parse_args()

    if not args.input_jsonl.exists():
        print(f"ERROR: {args.input_jsonl} not found", file=sys.stderr)
        return 1

    if args.production:
        report = _analyze_production(args.input_jsonl)
    else:
        records = _load_harness_records(args.input_jsonl)
        report = _analyze_harness(records)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_table(report, str(args.input_jsonl.name))

    if args.baseline:
        if not args.baseline.exists():
            print(f"ERROR: baseline {args.baseline} not found", file=sys.stderr)
            return 1
        if args.production:
            baseline_report = _analyze_production(args.baseline)
        else:
            baseline_report = _analyze_harness(_load_harness_records(args.baseline))
        if args.json:
            print(json.dumps({"baseline": baseline_report, "delta": {
                k: report.get(k, 0) - baseline_report.get(k, 0)
                for k in ("text_reading_hit_rate", "hard_fallback_rate")
            }}, indent=2))
        else:
            _compare(report, baseline_report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
