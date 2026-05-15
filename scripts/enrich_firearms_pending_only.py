#!/usr/bin/env python
"""Phase 6 audit follow-up — targeted VLM enrichment for Firearms.

Calls the existing
``scripts/enrich_image_chunks_v29.py::_enrich_one`` helper on **only**
image chunks whose ``metadata.vision_status == "pending"``. The
canonical script enriches every non-``complete`` chunk; Firearms's
post-Phase-6 OCR-lane reconvert emits 825 ``done``-status docling
per-region chunks (already non-placeholder, already passing the
``qa_conversion_audit.py`` image-placeholder check) plus 264
``pending``-status shadow-extraction full-page chunks (the actual
``IMAGE: FAIL`` source). Targeting only the pending subset gives the
same QA-gate outcome (placeholder ratio → 0) at ~25 % of the cloud
spend / wall time.

The script does NOT change anything else about the canonical
enrichment contract: VisionManager configuration, prompt, retry harness
and ``vision_provider_used`` stamping are all inherited from the
canonical helper.

Usage::

    python scripts/enrich_firearms_pending_only.py output/Firearms_phase6e/ingestion.jsonl
    python scripts/enrich_firearms_pending_only.py output/<doc>/ingestion.jsonl --dry-run

Atomic write-back via ``ingestion.jsonl.pending-tmp`` + ``os.replace``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Re-use the canonical script's helpers verbatim.
sys.path.insert(0, str(Path(__file__).parent))
from enrich_image_chunks_v29 import (  # noqa: E402
    _CLOUD_BASE_URL,
    _CLOUD_MODEL,
    _CLOUD_PROVIDER,
    _enrich_one,
    _iter_chunks,
    _resolve_api_key,
)

logger = logging.getLogger("enrich_firearms_pending_only")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl_paths", type=Path, nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Inspect first: how many pending chunks per file?
    pending_per_file: dict[Path, int] = {}
    for path in args.jsonl_paths:
        if not path.exists():
            logger.error("missing: %s", path)
            return 2
        pending = 0
        for rec in _iter_chunks(path):
            if rec.get("modality") != "image":
                continue
            md = rec.get("metadata") or {}
            if (md.get("vision_status") or rec.get("vision_status")) == "pending":
                pending += 1
        pending_per_file[path] = pending
        logger.info("%s: %d pending image chunks", path, pending)

    if args.dry_run:
        logger.info("DRY RUN — would enrich %d chunks total", sum(pending_per_file.values()))
        return 0

    # Initialise VisionManager exactly as the canonical script does.
    from mmrag_v2.vision.vision_manager import create_vision_manager
    api_key = _resolve_api_key()
    vision_manager = create_vision_manager(
        provider=_CLOUD_PROVIDER,
        api_key=api_key,
        model=_CLOUD_MODEL,
        base_url=_CLOUD_BASE_URL,
    )

    total_enriched = 0
    total_hard_fallback = 0
    failed_files: list[str] = []

    for path, pending_count in pending_per_file.items():
        if pending_count == 0:
            logger.info("%s: nothing to enrich", path)
            continue
        output_dir = path.parent
        tmp_path = path.with_suffix(path.suffix + ".pending-tmp")
        try:
            enriched = 0
            hard_fallback = 0
            with tmp_path.open("w") as out_fh:
                for rec in _iter_chunks(path):
                    if rec.get("object_type") and rec.get("object_type") != "chunk":
                        out_fh.write(json.dumps(rec) + "\n")
                        continue
                    if rec.get("modality") == "image":
                        md = rec.get("metadata") or {}
                        vision_status = (
                            md.get("vision_status") or rec.get("vision_status")
                        )
                        if vision_status == "pending":
                            rec, ok, _detail = _enrich_one(
                                rec, vision_manager, output_dir
                            )
                            if ok:
                                enriched += 1
                                logger.info(
                                    "  enriched (%d/%d): chunk_id=%s",
                                    enriched + hard_fallback,
                                    pending_count,
                                    rec.get("chunk_id"),
                                )
                            else:
                                hard_fallback += 1
                                logger.warning(
                                    "  hard_fallback (%d/%d): chunk_id=%s",
                                    enriched + hard_fallback,
                                    pending_count,
                                    rec.get("chunk_id"),
                                )
                    out_fh.write(json.dumps(rec) + "\n")
            os.replace(tmp_path, path)
            total_enriched += enriched
            total_hard_fallback += hard_fallback
            logger.info(
                "%s: enriched=%d hard_fallback=%d (of %d pending)",
                path, enriched, hard_fallback, pending_count,
            )
        except Exception:
            logger.exception("enrichment failed for %s", path)
            failed_path = tmp_path.with_suffix(tmp_path.suffix + ".failed")
            if tmp_path.exists():
                os.replace(tmp_path, failed_path)
            failed_files.append(str(path))

    logger.info(
        "TOTAL: enriched=%d hard_fallback=%d failed_files=%d",
        total_enriched, total_hard_fallback, len(failed_files),
    )
    return 0 if not failed_files else 1


if __name__ == "__main__":
    raise SystemExit(main())
