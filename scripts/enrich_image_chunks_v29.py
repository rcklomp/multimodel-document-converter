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
SHORT_DESC_THRESHOLD_CHARS = 20
SHORT_RESPONSE_SENTINEL = "complex_asset_short_response_after_retry"
_DETAIL_RETRY_COMPLEXITIES = {"complex", "text_heavy"}
_DETAIL_RETRY_SUFFIX = (
    "Provide a detailed description of the visual layout, components, color "
    "scheme, and their spatial relationships. Do not transcribe or paraphrase "
    "any text from the image."
)

logger = logging.getLogger("enrich_v29")


@dataclass
class EnrichmentStats:
    total_chunks: int = 0
    image_chunks: int = 0
    placeholder_chunks: int = 0
    enriched: int = 0
    hard_fallback: int = 0
    detail_retry_attempted: int = 0
    detail_retry_resolved: int = 0
    detail_retry_hard_fallback: int = 0
    skipped_already_complete: int = 0
    api_call_seconds: float = 0.0


@dataclass(frozen=True)
class DetailRetryResult:
    triggered: bool = False
    resolved: bool = False
    hard_fallback: bool = False


def _is_placeholder(value: Optional[str]) -> bool:
    """Return True when the chunk's ``visual_description`` is still the
    pending placeholder format used by the conversion path when no VLM
    ran."""
    if not value:
        return True
    return value.lstrip().startswith(_PLACEHOLDER_PREFIX)


def _build_detail_retry_prompt() -> str:
    """Build the one-shot detail retry prompt from the visual-only base."""
    from mmrag_v2.vision.vision_prompts import VISUAL_ONLY_PROMPT

    return f"{VISUAL_ONLY_PROMPT.rstrip()}\n\n{_DETAIL_RETRY_SUFFIX}"


def _is_short_description(value: Optional[str]) -> bool:
    if not value:
        return True
    stripped = value.strip()
    return (
        len(stripped) < SHORT_DESC_THRESHOLD_CHARS
        and "layout" not in stripped.lower()
    )


def _description_from_record(rec: Dict[str, Any]) -> str:
    md = rec.get("metadata") or {}
    return (
        md.get("visual_description")
        or rec.get("visual_description")
        or rec.get("content")
        or ""
    )


def _write_visual_description(rec: Dict[str, Any], description: str) -> None:
    md = rec.setdefault("metadata", {})
    rec["content"] = description
    rec["visual_description"] = description
    md["visual_description"] = description
    md["refined_content"] = description


def _mark_hard_fallback(
    rec: Dict[str, Any],
    reason: str,
    *,
    retry_attempted: bool = False,
) -> None:
    md = rec.setdefault("metadata", {})
    md["vision_status"] = "hard_fallback"
    md["vision_error"] = reason
    md["vision_provider_used"] = _CLOUD_MODEL
    if retry_attempted:
        md["vision_detail_retry_attempted"] = True


def _needs_detail_retry(rec: Dict[str, Any], output_dir: Path) -> bool:
    """Return True for complete, short descriptions on complex assets."""
    if rec.get("modality") != "image":
        return False
    md = rec.get("metadata") or {}
    vision_status = md.get("vision_status") or rec.get("vision_status")
    if vision_status != "complete":
        return False
    if md.get("vision_detail_retry_attempted"):
        return False
    description = _description_from_record(rec)
    if _is_placeholder(description) or not _is_short_description(description):
        return False

    from mmrag_v2.vision.asset_complexity import classify_asset_complexity

    result = classify_asset_complexity(rec, output_dir=output_dir)
    return result.complexity in _DETAIL_RETRY_COMPLEXITIES


def _maybe_retry_for_detail(
    rec: Dict[str, Any],
    vision_manager: Any,
    image: Any,
    output_dir: Path,
) -> DetailRetryResult:
    """Retry once when a complex asset got a valid but too-terse response."""
    if not _needs_detail_retry(rec, output_dir):
        return DetailRetryResult()

    md = rec.setdefault("metadata", {})
    prompt = _build_detail_retry_prompt()
    chunk_id = rec.get("chunk_id")
    logger.info("Detail retry for short complex image chunk %s", chunk_id)
    md["vision_detail_retry_attempted"] = True
    md["vision_attempts"] = int(md.get("vision_attempts") or 0) + 1

    try:
        response = vision_manager._provider.describe_image(  # script-owned direct call
            image,
            prompt,
        )
    except Exception as exc:
        logger.warning("Detail retry VLM call failed for chunk %s: %s", chunk_id, exc)
        _mark_hard_fallback(
            rec,
            SHORT_RESPONSE_SENTINEL,
            retry_attempted=True,
        )
        return DetailRetryResult(triggered=True, hard_fallback=True)

    from mmrag_v2.vision.vision_prompts import validate_vlm_response

    validation = validate_vlm_response(response)
    if (
        validation.is_valid
        and len(validation.cleaned_response.strip()) >= SHORT_DESC_THRESHOLD_CHARS
    ):
        description = validation.cleaned_response
        _write_visual_description(rec, description)
        md["vision_status"] = "complete"
        md["vision_provider_used"] = _CLOUD_MODEL
        md.pop("vision_error", None)
        md.pop("vision_validation_issues", None)
        logger.info(
            "Detail retry resolved chunk %s with %d chars",
            chunk_id,
            len(description.strip()),
        )
        return DetailRetryResult(triggered=True, resolved=True)

    issues = validation.issues if validation.issues else ["short response"]
    logger.info(
        "Detail retry exhausted for chunk %s: %s",
        chunk_id,
        ", ".join(issues),
    )
    _mark_hard_fallback(rec, SHORT_RESPONSE_SENTINEL, retry_attempted=True)
    if not validation.is_valid:
        md["vision_validation_issues"] = issues
    return DetailRetryResult(triggered=True, hard_fallback=True)


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
        for line_number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise json.JSONDecodeError(
                    f"{jsonl_path}:{line_number}: {exc.msg}",
                    exc.doc,
                    exc.pos,
                ) from exc


def _validate_jsonl(path: Path, expected_lines: int) -> None:
    actual_lines = 0
    with path.open() as fh:
        for line_number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            actual_lines += 1
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{path}:{line_number}: invalid JSON after enrichment: {exc}"
                ) from exc
    if actual_lines != expected_lines:
        raise RuntimeError(
            f"{path}: line count mismatch after enrichment validation — "
            f"parsed {actual_lines}, expected {expected_lines}"
        )


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
) -> tuple[Dict[str, Any], bool, DetailRetryResult]:
    """Enrich one image chunk in place. Returns (record, success, detail_retry)."""
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
        return rec, False, DetailRetryResult()

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
        return rec, False, DetailRetryResult()

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
        return rec, False, DetailRetryResult()

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
        return rec, False, DetailRetryResult()

    elapsed = time.perf_counter() - t0

    # VisionManager.enrich_image returns "[VLM_FAILED: ...]" when the
    # provider exception path fires (cloud timeout, 5xx, etc.). Treat
    # those exactly like the explicit-exception branch above so they
    # don't leak into visual_description and trigger downstream RAG
    # retrieval on a placeholder.
    if description and description.lstrip().startswith("[VLM_FAILED"):
        md["vision_status"] = "hard_fallback"
        md["vision_error"] = description
        md["vision_provider_used"] = _CLOUD_MODEL
        md["vision_attempts"] = int(md.get("vision_attempts") or 0) + 1
        return rec, False, DetailRetryResult()

    # Per IngestionChunk schema, ``content`` is the canonical
    # "Text content or VLM description" field. For image chunks the
    # convention is that ``content`` carries the visual description
    # post-enrichment. Earlier v2.9 runs forgot to update this field
    # → semantic_fidelity gate read the residual placeholder and
    # reported image_placeholder_ratio=1.0 even though the metadata
    # had real descriptions. Fix: write to all three positions.
    _write_visual_description(rec, description)
    md["vision_status"] = "complete"
    md["vision_provider_used"] = _CLOUD_MODEL
    md["vision_attempts"] = int(md.get("vision_attempts") or 0) + 1
    md.pop("vision_error", None)
    md.pop("vision_validation_issues", None)
    logger.debug(
        "Enriched %s in %.2fs", rec.get("chunk_id"), elapsed
    )
    detail_retry = _maybe_retry_for_detail(rec, vision_manager, image, output_dir)
    return rec, md.get("vision_status") == "complete", detail_retry


def _retry_existing_complete_short_description(
    rec: Dict[str, Any],
    vision_manager: Any,
    output_dir: Path,
) -> tuple[Dict[str, Any], DetailRetryResult]:
    """Retry an already-complete short description if the complexity gate requires it."""
    if not _needs_detail_retry(rec, output_dir):
        return rec, DetailRetryResult()

    from PIL import Image  # local import to avoid cost in dry-run

    md = rec.setdefault("metadata", {})
    asset_ref = rec.get("asset_ref") or md.get("asset_ref") or {}
    asset_path = asset_ref.get("file_path")
    if not asset_path:
        _mark_hard_fallback(
            rec,
            "missing asset_ref.file_path",
            retry_attempted=True,
        )
        return rec, DetailRetryResult(triggered=True, hard_fallback=True)

    full_path = (output_dir / asset_path).resolve()
    if not full_path.exists():
        _mark_hard_fallback(
            rec,
            f"asset not found: {asset_path}",
            retry_attempted=True,
        )
        return rec, DetailRetryResult(triggered=True, hard_fallback=True)

    try:
        image = Image.open(full_path).convert("RGB")
    except Exception as exc:
        _mark_hard_fallback(
            rec,
            f"image load failed: {exc}",
            retry_attempted=True,
        )
        return rec, DetailRetryResult(triggered=True, hard_fallback=True)

    return rec, _maybe_retry_for_detail(rec, vision_manager, image, output_dir)


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
    if dry_run:
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
                        rec, ok, detail_retry = _enrich_one(
                            rec,
                            vision_manager,
                            output_dir,
                        )
                        if ok:
                            stats.enriched += 1
                        else:
                            stats.hard_fallback += 1
                    else:
                        rec, detail_retry = _retry_existing_complete_short_description(
                            rec,
                            vision_manager,
                            output_dir,
                        )
                    if detail_retry.triggered:
                        stats.detail_retry_attempted += 1
                    if detail_retry.resolved:
                        stats.detail_retry_resolved += 1
                    if detail_retry.hard_fallback:
                        stats.detail_retry_hard_fallback += 1
                        if not needs_enrich:
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
    try:
        _validate_jsonl(tmp_path, expected_lines)
    except Exception:
        tmp_path.replace(failed_path)
        raise

    os.replace(tmp_path, jsonl_path)
    logger.info(
        "%s: enriched=%d hard_fallback=%d detail_retry=%d resolved=%d retry_hard_fallback=%d",
        jsonl_path,
        stats.enriched,
        stats.hard_fallback,
        stats.detail_retry_attempted,
        stats.detail_retry_resolved,
        stats.detail_retry_hard_fallback,
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
        grand_total.detail_retry_attempted += stats.detail_retry_attempted
        grand_total.detail_retry_resolved += stats.detail_retry_resolved
        grand_total.detail_retry_hard_fallback += stats.detail_retry_hard_fallback
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
        print(f"  detail retries:         {grand_total.detail_retry_attempted}")
        print(f"  detail retry resolved:  {grand_total.detail_retry_resolved}")
        print(f"  detail retry fallback:  {grand_total.detail_retry_hard_fallback}")
    if failed_files:
        print(f"  FAILED files:           {len(failed_files)}")
        for f in failed_files:
            print(f"    - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
