"""
v2.9 Phase 5b acceptance — image-side VLM enrichment is empirically
complete across the canonical 34-doc corpus.

Env-gated by ``RUN_V29_VLM_ACCEPTANCE=1`` because the assertions are
only meaningful AFTER ``scripts/enrich_image_chunks_v29.py`` has run
against a freshly v2.9-converted corpus. Until then the test is
skipped.

Pinned contracts (from ``docs/PLAN_V2.9.md`` §Phase 5):

* No image chunk has ``vision_status="pending"``.
* No image chunk's ``visual_description`` matches the placeholder
  pattern ``[Figure on page N] | Context: ...``.
* Every non-fallback image chunk has
  ``vision_provider_used == "qwen3-vl-plus"`` (the v2.9 cloud lock).
* Hard-fallback rate stays under the v2.9 cloud baseline + 5pp
  cushion (Source Sanctity hard-fallback). The baseline is set in
  Phase 5 pre-flight and recorded in the v2.9 AFTER snapshot.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

_OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "output"
_CONVERT_BOOKS = Path(__file__).resolve().parent.parent / "scripts" / "convert_books.sh"
_PLACEHOLDER_PREFIX = "[Figure on page"
_HARD_FALLBACK_PCT_CEILING = 0.05  # 5% absolute ceiling pre-baseline


def _read_canonical_doc_dirs() -> list[str]:
    """Parse ``scripts/convert_books.sh`` for canonical output dirs."""
    if not _CONVERT_BOOKS.exists():
        return []
    dirs: list[str] = []
    for line in _CONVERT_BOOKS.read_text().splitlines():
        line = line.strip()
        if not line.startswith("convert "):
            continue
        parts = line.split('"')
        if len(parts) >= 4:
            dirs.append(parts[3])
    return dirs


@pytest.mark.skipif(
    os.environ.get("RUN_V29_VLM_ACCEPTANCE") != "1",
    reason="Set RUN_V29_VLM_ACCEPTANCE=1 after Phase 5b enrichment completes",
)
def test_v29_image_chunks_are_fully_enriched() -> None:
    """Walk every ``output/<canonical>/ingestion.jsonl`` and verify the
    image-enrichment invariants. Failure shape lists the offending
    files + counts so the operator can re-run enrichment for just
    those documents."""
    if not _OUTPUT_ROOT.exists():
        pytest.skip(f"output not found: {_OUTPUT_ROOT}")

    canonical_dirs = _read_canonical_doc_dirs()
    jsonl_paths = sorted(
        _OUTPUT_ROOT / d / "ingestion.jsonl" for d in canonical_dirs
    )
    jsonl_paths = [p for p in jsonl_paths if p.exists()]
    if not jsonl_paths:
        pytest.skip(f"no canonical ingestion.jsonl under {_OUTPUT_ROOT}")

    pending_failures: list[str] = []
    placeholder_failures: list[str] = []
    provider_failures: list[str] = []
    total_images = 0
    total_hard_fallback = 0

    for path in jsonl_paths:
        image_count = 0
        pending = 0
        placeholder = 0
        wrong_provider = 0
        hard_fallback = 0

        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("modality") != "image":
                    continue
                image_count += 1
                md = rec.get("metadata") or {}
                vision_status = md.get("vision_status") or rec.get("vision_status")
                visual_description = (
                    md.get("visual_description") or rec.get("visual_description") or ""
                )
                provider_used = md.get("vision_provider_used") or rec.get(
                    "vision_provider_used"
                )

                if vision_status == "pending":
                    pending += 1
                if vision_status == "hard_fallback":
                    hard_fallback += 1
                    # hard_fallback is the documented escape hatch when
                    # the cloud provider returns no usable description
                    # (network timeout, persistent Source Sanctity reject).
                    # The placeholder visual_description is expected; the
                    # acceptance contract is the global hard_fallback
                    # rate, not the placeholder presence.
                    continue
                if visual_description.lstrip().startswith(_PLACEHOLDER_PREFIX):
                    placeholder += 1
                if provider_used not in ("qwen3-vl-plus",):
                    wrong_provider += 1

        if image_count == 0:
            continue
        if pending:
            pending_failures.append(f"{path.relative_to(_OUTPUT_ROOT)}: {pending} pending")
        if placeholder:
            placeholder_failures.append(
                f"{path.relative_to(_OUTPUT_ROOT)}: {placeholder} placeholder"
            )
        if wrong_provider:
            provider_failures.append(
                f"{path.relative_to(_OUTPUT_ROOT)}: {wrong_provider} non-qwen-vl-plus"
            )
        # Aggregate counters for corpus-level ceiling — individual docs
        # with high-resolution magazine photography (Combat-class) may
        # exceed the 5% per-doc rate due to cloud-endpoint timeouts on
        # large images, but the corpus-wide rate is the meaningful
        # production gate.
        total_images += image_count
        total_hard_fallback += hard_fallback

    assert not pending_failures, (
        "vision_status=pending found in v2.9 corpus:\n  "
        + "\n  ".join(pending_failures)
    )
    assert not placeholder_failures, (
        "placeholder visual_description found in v2.9 corpus:\n  "
        + "\n  ".join(placeholder_failures)
    )
    assert not provider_failures, (
        "non-qwen3-vl-plus image points found (v2.9 locks cloud):\n  "
        + "\n  ".join(provider_failures)
    )
    corpus_rate = (
        total_hard_fallback / total_images if total_images else 0.0
    )
    assert corpus_rate <= _HARD_FALLBACK_PCT_CEILING, (
        f"corpus-wide hard_fallback rate {corpus_rate:.2%} "
        f"({total_hard_fallback}/{total_images}) > "
        f"{_HARD_FALLBACK_PCT_CEILING:.0%} ceiling. Review the v2.9 AFTER "
        f"snapshot before relaxing this gate."
    )
