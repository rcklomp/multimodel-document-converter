"""Unit tests for src/mmrag_v2/vision/asset_complexity.py.

Phase 3 Step 2 — covers every classifier branch with synthetic
fixtures plus a missing-data fallback.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from mmrag_v2.vision.asset_complexity import (
    AssetComplexityResult,
    classify_asset_complexity,
)


def _chunk(*, bbox=None, page_w=612, page_h=792, asset_path=None):
    """Build a minimal image-chunk dict for the classifier."""
    spatial = {}
    if bbox is not None:
        spatial = {"bbox": list(bbox), "page_width": page_w, "page_height": page_h}
    md = {"spatial": spatial} if spatial else {}
    out = {"modality": "image", "metadata": md}
    if asset_path is not None:
        out["asset_ref"] = {"file_path": str(asset_path)}
    return out


# ---------------------------------------------------------------------------
# bbox-area branches (no asset_size signal)
# ---------------------------------------------------------------------------


def test_full_page_bbox_classified_text_heavy():
    """bbox area >= 50% of page → text_heavy."""
    # bbox values are in normalized [0, 1000] units.
    chunk = _chunk(bbox=(0, 0, 1000, 800))  # 80 % page area
    res = classify_asset_complexity(chunk)
    assert res.complexity == "text_heavy"
    assert res.bbox_area_fraction is not None
    assert res.bbox_area_fraction >= 0.5


def test_medium_bbox_classified_complex():
    """bbox area in [15 %, 50 %) → complex."""
    chunk = _chunk(bbox=(100, 100, 600, 500))  # ~20 % page area
    res = classify_asset_complexity(chunk)
    assert res.complexity == "complex"


def test_small_bbox_no_file_size_signal_defaults_simple():
    """small bbox without disk-size data → simple (the small bbox is enough)."""
    chunk = _chunk(bbox=(0, 0, 200, 200))  # 4 % page area
    res = classify_asset_complexity(chunk)
    assert res.complexity == "simple"


def test_small_bbox_with_tiny_file_classified_simple(tmp_path):
    """small bbox + tiny file (icon shape) → simple."""
    asset = tmp_path / "icon.png"
    asset.write_bytes(b"x" * 1000)  # 1 KB
    chunk = _chunk(bbox=(0, 0, 100, 100), asset_path="icon.png")
    res = classify_asset_complexity(chunk, output_dir=tmp_path)
    assert res.complexity == "simple"
    assert res.asset_size_bytes == 1000


def test_tiny_bbox_iconography_lane_overrides_file_size(tmp_path):
    """Plan v2.9 Phase D (2026-05-11): bbox area < 1 % of page is
    iconography (logo, certification mark, status glyph). At this
    size, the asset cannot carry substantive content regardless of
    file size — the tiny-bbox lane fires before the size-based
    decision. Regression for Hybrid_electric_vehicles p1 logo:
    bbox 0.68 %, file 5963 bytes (just above the 5 KB tiny-file
    cap)."""
    asset = tmp_path / "logo.png"
    asset.write_bytes(b"x" * 5963)  # 5.96 KB — would FAIL is_tiny_file
    chunk = _chunk(bbox=(0, 0, 80, 80), asset_path="logo.png")  # 0.64 % area
    res = classify_asset_complexity(chunk, output_dir=tmp_path)
    assert res.complexity == "simple"
    assert "tiny bbox iconography lane" in res.reason


def test_tiny_bbox_iconography_lane_with_large_file(tmp_path):
    """Boundary case: a tiny-bbox asset (<1 %) but with a >100 KB
    file. The text_heavy branch fires FIRST on file size, so the
    iconography lane does not override. This protects against
    misleading-bbox tiny-thumbnail-of-large-asset cases."""
    asset = tmp_path / "thumb.png"
    asset.write_bytes(b"x" * 200_000)
    chunk = _chunk(bbox=(0, 0, 80, 80), asset_path="thumb.png")
    res = classify_asset_complexity(chunk, output_dir=tmp_path)
    assert res.complexity == "text_heavy"


def test_just_above_tiny_bbox_threshold_NOT_iconography(tmp_path):
    """bbox at exactly 1 % is the boundary — must NOT fire the
    iconography lane. A 1 % bbox is small but still capable of
    showing a sub-figure."""
    # 100 * 100 / (1000 * 1000) = 0.01 = exactly 1%
    asset = tmp_path / "small_diagram.png"
    asset.write_bytes(b"x" * 20_000)  # 20 KB
    chunk = _chunk(bbox=(0, 0, 100, 100), asset_path="small_diagram.png")
    res = classify_asset_complexity(chunk, output_dir=tmp_path)
    assert res.complexity != "simple"


def test_small_bbox_with_large_file_overrides_to_text_heavy(tmp_path):
    """small bbox but >100 KB file → text_heavy.

    A small bbox that points at a content-rich asset is suspicious —
    the bbox may be misleading (e.g. cropped thumbnail of a UI
    screenshot). Large file size dominates.
    """
    asset = tmp_path / "big.png"
    asset.write_bytes(b"x" * 200_000)  # 200 KB
    chunk = _chunk(bbox=(0, 0, 100, 100), asset_path="big.png")
    res = classify_asset_complexity(chunk, output_dir=tmp_path)
    assert res.complexity == "text_heavy"
    assert res.asset_size_bytes == 200_000


# ---------------------------------------------------------------------------
# Missing-data fallback
# ---------------------------------------------------------------------------


def test_no_bbox_no_asset_defaults_complex():
    """No bbox + no asset path → default complex (safer than simple)."""
    chunk = {"modality": "image", "metadata": {}}
    res = classify_asset_complexity(chunk)
    assert res.complexity == "complex"
    assert res.bbox_area_fraction is None
    assert res.asset_size_bytes is None
    assert "default complex" in res.reason


def test_malformed_bbox_treated_as_missing():
    """bbox with wrong arity → treat as missing, not a 0-area surface."""
    chunk = _chunk(bbox=(0, 0, 1000))  # only 3 values
    res = classify_asset_complexity(chunk)
    assert res.complexity == "complex"
    assert res.bbox_area_fraction is None


def test_zero_area_bbox_treated_as_simple_when_no_file():
    """0-area bbox is technically < 15 % — falls into simple lane when no
    file-size signal contradicts it. Acceptable: the chunk has no
    visual extent, short description is fine."""
    chunk = _chunk(bbox=(50, 50, 50, 50))  # zero area
    res = classify_asset_complexity(chunk)
    assert res.complexity == "simple"
    assert res.bbox_area_fraction == 0.0


# ---------------------------------------------------------------------------
# Asset path resolution
# ---------------------------------------------------------------------------


def test_asset_path_resolves_relative_to_output_dir(tmp_path):
    """File path is resolved against ``output_dir`` when supplied."""
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    asset = asset_dir / "f.png"
    asset.write_bytes(b"x" * 6_000)  # 6 KB — above tiny threshold
    chunk = _chunk(bbox=(100, 100, 300, 300), asset_path="assets/f.png")
    res = classify_asset_complexity(chunk, output_dir=tmp_path)
    assert res.asset_size_bytes == 6_000
    # Small bbox (4 %) + non-tiny file → falls through both gates → complex.
    assert res.complexity == "complex"


def test_missing_asset_file_yields_no_size_signal(tmp_path):
    """Asset path that does not exist on disk → asset_size_bytes is None."""
    chunk = _chunk(bbox=(0, 0, 1000, 1000), asset_path="missing.png")
    res = classify_asset_complexity(chunk, output_dir=tmp_path)
    assert res.asset_size_bytes is None
    # bbox is full-page → still text_heavy.
    assert res.complexity == "text_heavy"


def test_asset_path_with_no_output_dir(tmp_path, monkeypatch):
    """When no output_dir is given, the asset path is tried as-is."""
    asset = tmp_path / "raw.png"
    asset.write_bytes(b"x" * 3_000)  # 3 KB
    chunk = _chunk(bbox=(0, 0, 100, 100), asset_path=str(asset))
    res = classify_asset_complexity(chunk)
    assert res.asset_size_bytes == 3_000
    assert res.complexity == "simple"


# ---------------------------------------------------------------------------
# Result type sanity
# ---------------------------------------------------------------------------


def test_result_is_frozen_dataclass():
    chunk = _chunk(bbox=(0, 0, 200, 200))
    res = classify_asset_complexity(chunk)
    assert isinstance(res, AssetComplexityResult)
    with pytest.raises(Exception):
        res.complexity = "simple"  # type: ignore[misc]
