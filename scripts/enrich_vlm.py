#!/usr/bin/env python3
"""
Post-conversion VLM enrichment for image chunks.

Reads an existing ingestion.jsonl, finds image chunks with
vision_status=pending, generates VLM descriptions, and writes
an updated JSONL in-place.

Usage:
    python scripts/enrich_vlm.py output/MyDoc/ingestion.jsonl \
        --provider openai --model llava-1.6 \
        --base-url http://localhost:1234/v1 --api-key lm-studio

    python scripts/enrich_vlm.py output/MyDoc/ingestion.jsonl \
        --provider ollama --model llava:latest

    # Dry run: show what would be enriched
    python scripts/enrich_vlm.py output/MyDoc/ingestion.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-conversion VLM enrichment")
    parser.add_argument("ingestion_jsonl", type=Path)
    parser.add_argument("--provider", choices=["openai", "ollama", "anthropic"], default="openai")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=180, help="VLM timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="Show pending chunks without enriching")
    args = parser.parse_args()

    path = args.ingestion_jsonl
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        return 2

    # Read all lines
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines]

    # Find pending image chunks
    pending = []
    for i, rec in enumerate(records):
        if rec.get("modality") != "image":
            continue
        md = rec.get("metadata") or {}
        status = md.get("vision_status", "")
        if status in ("pending", "failed", ""):
            pending.append(i)

    if not pending:
        print("No pending image chunks found. All images already enriched or no images present.")
        return 0

    print(f"Found {len(pending)} image chunks to enrich")

    if args.dry_run:
        for idx in pending:
            rec = records[idx]
            md = rec.get("metadata") or {}
            pg = md.get("page_number", "?")
            asset = (rec.get("asset_ref") or {}).get("file_path", "none")
            status = md.get("vision_status", "unknown")
            print(f"  [{status}] page {pg}: {asset}")
        return 0

    # Initialize VLM
    try:
        from mmrag_v2.vision.vision_manager import VisionManager
    except ImportError:
        print("Error: could not import VisionManager. Ensure mmrag_v2 is installed.", file=sys.stderr)
        return 1

    cache_dir = str(path.parent)
    vm = VisionManager(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.timeout,
        cache_dir=cache_dir,
    )

    enriched = 0
    failed = 0
    assets_dir = path.parent / "assets"

    for idx in pending:
        rec = records[idx]
        md = rec.get("metadata") or {}
        pg = md.get("page_number", "?")
        asset_ref = rec.get("asset_ref") or {}
        asset_path = asset_ref.get("file_path", "")

        if not asset_path:
            md["vision_status"] = "skipped"
            md["vision_error"] = "no asset file"
            continue

        full_asset_path = path.parent / asset_path
        if not full_asset_path.exists():
            md["vision_status"] = "failed"
            md["vision_error"] = f"asset not found: {asset_path}"
            failed += 1
            continue

        # Load image and call VLM
        try:
            from PIL import Image
            img = Image.open(full_asset_path)

            attempts = 0
            description: Optional[str] = None
            last_error: Optional[str] = None

            for attempt in range(args.max_retries + 1):
                attempts += 1
                try:
                    description = vm.enrich_image(
                        image=img,
                        state=None,
                        page_number=pg if isinstance(pg, int) else 1,
                    )
                    if description and not description.startswith("[VLM_FAILED"):
                        break
                    last_error = description
                    description = None
                except Exception as e:
                    last_error = str(e)
                    if attempt < args.max_retries:
                        time.sleep(2 ** attempt)  # exponential backoff

            if description:
                rec["content"] = description
                md["visual_description"] = description
                md["vision_status"] = "done"
                md["vision_provider_used"] = args.provider
                md["vision_attempts"] = attempts
                md.pop("vision_error", None)
                enriched += 1
                print(f"  ✓ page {pg}: {description[:60]}...")
            else:
                md["vision_status"] = "failed"
                md["vision_error"] = last_error or "unknown error"
                md["vision_attempts"] = attempts
                failed += 1
                print(f"  ✗ page {pg}: {last_error}")

        except Exception as e:
            md["vision_status"] = "failed"
            md["vision_error"] = str(e)
            failed += 1
            print(f"  ✗ page {pg}: {e}")

        rec["metadata"] = md

    # Flush VLM cache
    try:
        vm.flush_cache()
    except Exception:
        pass

    # Write updated JSONL
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDone: {enriched} enriched, {failed} failed, {len(pending) - enriched - failed} skipped")
    print(f"Updated: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
