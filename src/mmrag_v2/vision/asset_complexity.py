"""Asset complexity classifier for VLM gate calibration.

Phase 3 Step 2 — classifies image assets into ``simple`` / ``complex`` /
``text_heavy`` based on cheap, asset-only signals (bbox area, disk
size). The QA gate (Step 3) consumes this to decide whether a short
VLM ``visual_description`` is acceptable: short on a simple asset is
WARN, short on a complex/text_heavy asset is FAIL.

Empirical basis (Phase 3 Step 0 v4 corpus, qwen3-vl-plus):
- 604 image chunks across 3 docs (Hao 252, Adedeji 128, PCWorld 224)
- mean description length ~130-160 chars; p50 ~115-160; p90 ~260-300
- bbox-area distribution (Hao): 10 % <5 %, 40 % 5-15 %, 28 % 15-30 %,
  18 % 30-50 %, 4 % >=50 %
- Docling did NOT stamp picture-class labels on this corpus (all
  metadata.image_class entries were absent), so the classifier
  cannot rely on that signal even when it is present in v2.10+.

The thresholds below derive from that empirical distribution. They
are tuned for "small bbox AND tiny file = certainly simple"
(unambiguous icons / decorative thumbnails) versus everything else
defaulting to ``complex`` (the safer side for retrieval — a complex
asset with a short description gets retried, an icon with a short
description does not).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

# Coordinate system for bbox: integer [0, 1000] normalized
# (REQ-COORD-01, AGENTS.md "Project Invariants").
_COORD_SCALE = 1000

# Bbox-area fractions of page area. Empirical bins on Hao show clear
# separation at 0.15 (icon/small-figure boundary) and 0.50 (full-page
# threshold).
_SMALL_BBOX_AREA_MAX = 0.15
_LARGE_BBOX_AREA_MIN = 0.50

# Asset disk-size thresholds. PNGs under 5 KB are almost always icons
# or decorative thumbnails; PNGs over 100 KB usually contain rendered
# text or dense visual structure.
_TINY_ASSET_BYTES_MAX = 5_000
_LARGE_ASSET_BYTES_MIN = 100_000

# When neither bbox nor disk-size data is available, the safer default
# is ``complex``: a complex misclassification of an icon costs one VLM
# retry (cheap); a simple misclassification of a real diagram lets a
# leak slip through (expensive).
_DEFAULT_COMPLEXITY = "complex"


@dataclass(frozen=True)
class AssetComplexityResult:
    """Outcome of classifying one image-chunk asset.

    Attributes
    ----------
    complexity:
        One of ``"simple"``, ``"complex"``, ``"text_heavy"``.
    reason:
        Short human-readable summary of which signals drove the
        decision. Useful for the Phase 3 baseline doc and for QA
        debugging.
    bbox_area_fraction:
        Bbox area as a fraction of normalized page area (range
        ``[0, 1]``). ``None`` if bbox or page dims missing.
    asset_size_bytes:
        Disk size of the asset PNG, if resolvable. ``None`` otherwise.
    """

    complexity: str
    reason: str
    bbox_area_fraction: Optional[float]
    asset_size_bytes: Optional[int]


def _bbox_area_fraction(metadata: Mapping[str, Any]) -> Optional[float]:
    spatial = metadata.get("spatial") or {}
    bbox = spatial.get("bbox") or []
    if len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = (int(c) for c in bbox)
    except (TypeError, ValueError):
        return None
    bw = max(0, x1 - x0)
    bh = max(0, y1 - y0)
    return (bw * bh) / (_COORD_SCALE * _COORD_SCALE)


def _resolve_asset_size(
    chunk: Mapping[str, Any],
    output_dir: Optional[Path] = None,
) -> Optional[int]:
    """Return asset PNG disk size in bytes, or ``None`` if unresolvable.

    The chunk's ``asset_ref.file_path`` is typically a relative path
    rooted at the conversion output directory (``output/<doc>/``). We
    resolve it against ``output_dir`` if given, else try the path
    as-is.
    """
    asset_ref = chunk.get("asset_ref") or {}
    fp = asset_ref.get("file_path")
    if not fp:
        return None
    candidates = []
    if output_dir is not None:
        candidates.append(Path(output_dir) / fp)
    candidates.append(Path(fp))
    for p in candidates:
        try:
            return p.stat().st_size
        except (OSError, FileNotFoundError):
            continue
    return None


def classify_asset_complexity(
    chunk: Mapping[str, Any],
    output_dir: Optional[Path] = None,
) -> AssetComplexityResult:
    """Classify a single image-modality chunk.

    Reads only metadata + (optionally) the asset PNG's disk size — no
    PNG pixel inspection, no model call. Designed to be cheap enough
    to run on every image chunk during QA + retry-decision passes.

    Parameters
    ----------
    chunk:
        A JSONL row decoded as a mapping. Expected to have at least
        ``metadata.spatial.bbox`` for bbox signal; ``asset_ref.file_path``
        is optional and enables the disk-size signal.
    output_dir:
        Directory the asset paths are relative to. Typically
        ``output/<doc>/``. If omitted, asset size is read from the
        path as-is.

    Returns
    -------
    AssetComplexityResult — see field docs.
    """
    metadata = chunk.get("metadata") or {}
    area = _bbox_area_fraction(metadata)
    size = _resolve_asset_size(chunk, output_dir=output_dir)

    is_large_bbox = area is not None and area >= _LARGE_BBOX_AREA_MIN
    is_small_bbox = area is not None and area < _SMALL_BBOX_AREA_MAX
    is_large_file = size is not None and size >= _LARGE_ASSET_BYTES_MIN
    is_tiny_file = size is not None and size <= _TINY_ASSET_BYTES_MAX

    # text_heavy: full-page bbox OR very large file. Either signals
    # that the asset likely contains rendered text or dense layout
    # which a 20-char description cannot adequately cover.
    area_str = f"{area:.2f}" if area is not None else "NA"
    size_str = str(size) if size is not None else "NA"

    if is_large_bbox or is_large_file:
        return AssetComplexityResult(
            complexity="text_heavy",
            reason=(
                f"bbox_area={area_str}; asset_size={size_str}; "
                "full-page bbox or large file"
            ),
            bbox_area_fraction=area,
            asset_size_bytes=size,
        )

    # simple: small bbox AND tiny file (or unknown file). Both signals
    # must agree before we treat short descriptions as acceptable.
    if is_small_bbox and (is_tiny_file or size is None):
        return AssetComplexityResult(
            complexity="simple",
            reason=(
                f"bbox_area={area_str}; asset_size={size_str}; "
                "small bbox + tiny/unknown file"
            ),
            bbox_area_fraction=area,
            asset_size_bytes=size,
        )

    # Default lane: complex. Includes medium bbox, conflicting signals,
    # missing bbox data, etc.
    if area is None and size is None:
        reason = "no bbox or asset_size signal — default complex"
    else:
        reason = (
            f"bbox_area={area_str}; asset_size={size_str}; "
            "default complex lane"
        )
    return AssetComplexityResult(
        complexity=_DEFAULT_COMPLEXITY,
        reason=reason,
        bbox_area_fraction=area,
        asset_size_bytes=size,
    )
