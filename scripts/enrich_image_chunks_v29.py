#!/usr/bin/env python
"""
v2.9 Phase 5b — image-only VLM enrichment for the canonical corpus.

Re-reads each ``output/<doc>/ingestion.jsonl``, finds image chunks
whose ``visual_description`` is still a placeholder
(``[Figure on page N] | Context: ...`` or ``vision_status="pending"``),
and replaces them with a real cloud-VLM description from
``qwen3-vl-plus`` (Alibaba DashScope international endpoint).

Why a separate Phase 5b run instead of conversion-time VLM:

* Conversion-time VLM blocks the entire batch on cloud round-trip latency.
* The Phase 5a broad reconversion runs sequentially across 34 docs;
  re-running the conversion just to retry one timed-out image inflates
  runtime by hours.
* This script is restartable: atomic tmp-file write-back means a crash
  mid-doc leaves the original ``ingestion.jsonl`` untouched.

VLM choice locked to cloud per ``docs/PLAN_V2.9.md`` §Phase 5 decision e:
local ``NuMarkdown-8B-Thinking-mlx-8bits`` at ``http://10.0.10.246:8000/v1``
is unreachable from off-network machines (project memory, confirmed
2026-05-04). Re-evaluate the local lane in v2.10. The provider here is
``qwen3-vl-plus`` only; no auto-detect, no fallback to local.

Usage::

    python scripts/enrich_image_chunks_v29.py output/<doc>/ingestion.jsonl [...]
    python scripts/enrich_image_chunks_v29.py output/<doc>/ingestion.jsonl --dry-run
    python scripts/enrich_image_chunks_v29.py output/*/ingestion.jsonl

Behavior:

* ``--dry-run``: report the image-chunk count and estimated calls per
  file without making any API call. Use this to verify the work scope
  before burning cloud spend.
* Atomic write-back via ``ingestion.jsonl.v29tmp`` + ``os.replace``.
* On crash: the temp file is left as ``ingestion.jsonl.v29tmp.failed``
  and exit code is non-zero. Re-running skips already-completed image
  chunks.
* Source Sanctity: descriptions go through ``VisionManager.enrich_image``
  which carries the existing visual-only prompt + text-reading
  retry harness.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

# v2.10: re-evaluate local NuMarkdown-8B endpoint reachability.
_CLOUD_PROVIDER = "openai"  # qwen3-vl-plus is OpenAI-compatible via DashScope
_CLOUD_MODEL = "qwen3-vl-plus"
_CLOUD_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

_PLACEHOLDER_PREFIX = "[Figure on page"
_FSYNC_EVERY = 50  # fsync the tmp file every N enrichments

logger = logging.getLogger("enrich_v29")


@dataclass
class EnrichmentStats:
    total_chunks: int = 0
    image_chunks: int = 0
    placeholder_chunks: int = 0
    enriched: int = 0
    hard_fallback: int = 0
    skipped_already_complete: int = 0
    api_call_seconds: float = 0.0


def _is_placeholder(value: Optional[str]) -> bool:
    """Return True when the chunk's ``visual_description`` is still the
    pending placeholder format used by the conversion path when no VLM
    ran."""
    if not value:
        return True
    return value.lstrip().startswith(_PLACEHOLDER_PREFIX)


def _resolve_api_key() -> str:
    """Resolve the DashScope API key from MMRAG_REFINER_API_KEY or
    ``~/.mmrag-v2.yml`` refiner.api_key."""
    env_key = os.environ.get("MMRAG_REFINER_API_KEY") or os.environ.get(
        "DASHSCOPE_API_KEY"
    )
    if env_key:
        return env_key

    cfg_path = Path.home() / ".mmrag-v2.yml"
    if cfg_path.exists():
        try:
            import yaml  # type: ignore[import-not-found]

            data = yaml.safe_load(cfg_path.read_text()) or {}
            refiner = (data.get("refiner") or {})
            key = refiner.get("api_key")
            if key:
                return str(key)
        except Exception as exc:  # pragma: no cover - best-effort lookup
            logger.warning("Could not read API key from %s: %s", cfg_path, exc)

    raise SystemExit(
        "No DashScope API key found. Set MMRAG_REFINER_API_KEY env var or "
        "configure refiner.api_key in ~/.mmrag-v2.yml."
    )


def _iter_chunks(jsonl_path: Path) -> Iterator[Dict[str, Any]]:
    with jsonl_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _count_work(jsonl_path: Path) -> EnrichmentStats:
    stats = EnrichmentStats()
    for rec in _iter_chunks(jsonl_path):
        if rec.get("object_type") and rec.get("object_type") != "chunk":
            continue
        stats.total_chunks += 1
        if rec.get("modality") != "image":
            continue
        stats.image_chunks += 1
        md = rec.get("metadata") or {}
        vision_status = md.get("vision_status") or rec.get("vision_status")
        visual_description = (
            md.get("visual_description") or rec.get("visual_description")
        )
        if vision_status == "complete" and not _is_placeholder(visual_description):
            stats.skipped_already_complete += 1
            continue
        stats.placeholder_chunks += 1
    return stats


def _enrich_one(
    rec: Dict[str, Any],
    vision_manager: Any,
    output_dir: Path,
) -> tuple[Dict[str, Any], bool]:
    """Enrich one image chunk in place. Returns (record, success)."""
    from PIL import Image  # local import to avoid cost in dry-run

    md = rec.setdefault("metadata", {})

    asset_ref = rec.get("asset_ref") or md.get("asset_ref") or {}
    asset_path = asset_ref.get("file_path")
    if not asset_path:
        logger.warning(
            "Image chunk %s has no asset_ref.file_path; marking hard_fallback",
            rec.get("chunk_id"),
        )
        md["vision_status"] = "hard_fallback"
        md["vision_error"] = "missing asset_ref.file_path"
        md["vision_provider_used"] = _CLOUD_MODEL
        return rec, False

    full_path = (output_dir / asset_path).resolve()
    if not full_path.exists():
        logger.warning(
            "Asset %s not found on disk for chunk %s; marking hard_fallback",
            full_path,
            rec.get("chunk_id"),
        )
        md["vision_status"] = "hard_fallback"
        md["vision_error"] = f"asset not found: {asset_path}"
        md["vision_provider_used"] = _CLOUD_MODEL
        return rec, False

    try:
        image = Image.open(full_path).convert("RGB")
    except Exception as exc:
        logger.warning(
            "Failed to load %s for chunk %s: %s; marking hard_fallback",
            full_path,
            rec.get("chunk_id"),
            exc,
        )
        md["vision_status"] = "hard_fallback"
        md["vision_error"] = f"image load failed: {exc}"
        md["vision_provider_used"] = _CLOUD_MODEL
        return rec, False

    page_number = md.get("page_number") or rec.get("page_number") or 1
    semantic_context = rec.get("semantic_context") or {}
    prev_text = semantic_context.get("prev_text_snippet") or ""

    from mmrag_v2.state.context_state import create_context_state

    state = create_context_state(
        doc_id=rec.get("doc_id") or "unknown",
        source_file=md.get("source_file") or "",
    )

    t0 = time.perf_counter()
    try:
        description = vision_manager.enrich_image(
            image,
            state,
            page_number=int(page_number),
            anchor_text=prev_text or None,
        )
    except Exception as exc:
        logger.error(
            "VLM call failed for chunk %s (asset %s): %s",
            rec.get("chunk_id"),
            asset_path,
            exc,
        )
        md["vision_status"] = "hard_fallback"
        md["vision_error"] = f"vlm error: {exc}"
        md["vision_provider_used"] = _CLOUD_MODEL
        return rec, False

    elapsed = time.perf_counter() - t0

    rec["visual_description"] = description
    md["visual_description"] = description
    md["refined_content"] = description
    md["vision_status"] = "complete"
    md["vision_provider_used"] = _CLOUD_MODEL
    md["vision_attempts"] = int(md.get("vision_attempts") or 0) + 1
    md.pop("vision_error", None)
    md.pop("vision_validation_issues", None)
    logger.debug(
        "Enriched %s in %.2fs", rec.get("chunk_id"), elapsed
    )
    return rec, True


def _enrich_jsonl(
    jsonl_path: Path,
    vision_manager: Any,
    dry_run: bool,
) -> EnrichmentStats:
    stats = _count_work(jsonl_path)
    logger.info(
        "%s: %d chunks, %d image, %d placeholder, %d already-complete",
        jsonl_path,
        stats.total_chunks,
        stats.image_chunks,
        stats.placeholder_chunks,
        stats.skipped_already_complete,
    )
    if dry_run or stats.placeholder_chunks == 0:
        return stats

    output_dir = jsonl_path.parent
    tmp_path = jsonl_path.with_suffix(jsonl_path.suffix + ".v29tmp")
    failed_path = jsonl_path.with_suffix(jsonl_path.suffix + ".v29tmp.failed")

    if failed_path.exists():
        logger.warning(
            "Found previous failed temp at %s; resuming will use existing "
            "completed entries from the source jsonl",
            failed_path,
        )

    expected_lines = 0
    written_lines = 0

    try:
        with tmp_path.open("w") as out_fh:
            for idx, rec in enumerate(_iter_chunks(jsonl_path)):
                expected_lines += 1
                if rec.get("object_type") and rec.get("object_type") != "chunk":
                    out_fh.write(json.dumps(rec) + "\n")
                    written_lines += 1
                    continue
                if rec.get("modality") == "image":
                    md = rec.get("metadata") or {}
                    vision_status = md.get("vision_status") or rec.get("vision_status")
                    visual_description = (
                        md.get("visual_description") or rec.get("visual_description")
                    )
                    needs_enrich = (
                        vision_status != "complete"
                        or _is_placeholder(visual_description)
                    )
                    if needs_enrich:
                        rec, ok = _enrich_one(rec, vision_manager, output_dir)
                        if ok:
                            stats.enriched += 1
                        else:
                            stats.hard_fallback += 1
                out_fh.write(json.dumps(rec) + "\n")
                written_lines += 1
                if (idx + 1) % _FSYNC_EVERY == 0:
                    out_fh.flush()
                    os.fsync(out_fh.fileno())
            out_fh.flush()
            os.fsync(out_fh.fileno())
    except Exception:
        # Leave the partial work for inspection. Preserve original file.
        try:
            tmp_path.replace(failed_path)
        except OSError:
            pass
        raise

    if written_lines != expected_lines:
        tmp_path.replace(failed_path)
        raise RuntimeError(
            f"line count mismatch — wrote {written_lines}, expected {expected_lines}; "
            f"left tmp at {failed_path}"
        )

    os.replace(tmp_path, jsonl_path)
    logger.info(
        "%s: enriched=%d hard_fallback=%d",
        jsonl_path,
        stats.enriched,
        stats.hard_fallback,
    )
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "jsonl_paths",
        nargs="+",
        type=Path,
        help="ingestion.jsonl files to enrich (one per canonical document).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count placeholder image chunks only; do not call the VLM.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="DEBUG logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    api_key = None
    vision_manager = None
    if not args.dry_run:
        api_key = _resolve_api_key()
        from mmrag_v2.vision.vision_manager import create_vision_manager

        vision_manager = create_vision_manager(
            provider=_CLOUD_PROVIDER,
            api_key=api_key,
            model=_CLOUD_MODEL,
            base_url=_CLOUD_BASE_URL,
        )
        # Document domain: leave default. Per-doc override would tilt the
        # prompt; the v2.9 acceptance contract is a single visual-only
        # prompt across all docs.

    grand_total = EnrichmentStats()
    failed_files: List[str] = []

    for path in args.jsonl_paths:
        if not path.exists():
            logger.error("missing: %s", path)
            failed_files.append(str(path))
            continue
        try:
            stats = _enrich_jsonl(path, vision_manager, args.dry_run)
        except Exception as exc:
            logger.exception("enrichment failed for %s: %s", path, exc)
            failed_files.append(str(path))
            continue
        grand_total.total_chunks += stats.total_chunks
        grand_total.image_chunks += stats.image_chunks
        grand_total.placeholder_chunks += stats.placeholder_chunks
        grand_total.enriched += stats.enriched
        grand_total.hard_fallback += stats.hard_fallback
        grand_total.skipped_already_complete += stats.skipped_already_complete

    print()
    print("=" * 60)
    print("v2.9 ENRICHMENT SUMMARY")
    print("=" * 60)
    print(f"  files processed:        {len(args.jsonl_paths) - len(failed_files)}")
    print(f"  total chunks scanned:   {grand_total.total_chunks}")
    print(f"  image chunks:           {grand_total.image_chunks}")
    print(f"  placeholder chunks:     {grand_total.placeholder_chunks}")
    print(f"  already complete:       {grand_total.skipped_already_complete}")
    if not args.dry_run:
        print(f"  enriched (this run):    {grand_total.enriched}")
        print(f"  hard fallback:          {grand_total.hard_fallback}")
    if failed_files:
        print(f"  FAILED files:           {len(failed_files)}")
        for f in failed_files:
            print(f"    - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
