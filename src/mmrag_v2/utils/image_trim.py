"""
Image trimming utilities.

Purpose: produce cleaner figure/table assets by removing large uniform white margins
that often come from page-render crops (especially in digital PDFs with vector
diagrams rendered to raster crops).

Design:
- Conservative trigger: only trims when the outer border is mostly near-white.
- Robust on scans: scanned pages usually have non-white borders, so trimming
  typically does nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TrimResult:
    image: Image.Image
    trimmed: bool
    bbox: Optional[Tuple[int, int, int, int]]  # (left, top, right, bottom) in px


def edge_ink_fractions(
    image: Image.Image,
    *,
    white_thresh: int = 250,
    edge_px: int = 3,
) -> Dict[str, float]:
    """
    Compute how much "ink" (non-white-ish pixels) touches each edge.

    Used to detect overly tight crops that cut off figures/tables.
    """
    im = image.convert("RGB")
    arr = np.asarray(im)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0}

    h, w = arr.shape[:2]
    if h <= 0 or w <= 0:
        return {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0}

    ep = max(1, min(edge_px, h // 2, w // 2))
    # Anything not near-white counts as ink.
    mask = (arr < white_thresh).any(axis=2)

    top = float(mask[:ep, :].mean())
    bottom = float(mask[h - ep :, :].mean())
    left = float(mask[:, :ep].mean())
    right = float(mask[:, w - ep :].mean())
    return {"top": top, "bottom": bottom, "left": left, "right": right}


def expand_crop_box_if_clipped(
    page_image: Image.Image,
    crop_box: Tuple[int, int, int, int],
    *,
    step_px: int = 20,
    max_extra_px: int = 80,
    white_thresh: int = 250,
    edge_px: int = 3,
    ink_trigger: float = 0.02,
) -> Tuple[int, int, int, int]:
    """
    Expand a crop box when content touches the crop edges (likely cut off).

    This is a pragmatic fix for inaccurate bboxes produced by layout tools:
    - If ink is present on an edge strip, expand that edge outward.
    - Stops once edges are "clean enough" or max expansion is reached.
    """
    if page_image is None:
        return crop_box

    im = page_image  # keep original mode
    img_w, img_h = im.size
    if img_w <= 0 or img_h <= 0:
        return crop_box

    l, t, r, b = crop_box
    l = max(0, min(l, img_w - 1))
    t = max(0, min(t, img_h - 1))
    r = max(l + 1, min(r, img_w))
    b = max(t + 1, min(b, img_h))
    cur = (l, t, r, b)

    step = max(1, int(step_px))
    max_extra = max(0, int(max_extra_px))
    expanded = 0

    while expanded < max_extra:
        crop = im.crop(cur)
        fracs = edge_ink_fractions(crop, white_thresh=white_thresh, edge_px=edge_px)

        # If all edges are clean, stop.
        if (
            fracs["top"] <= ink_trigger
            and fracs["bottom"] <= ink_trigger
            and fracs["left"] <= ink_trigger
            and fracs["right"] <= ink_trigger
        ):
            break

        l, t, r, b = cur
        nl, nt, nr, nb = l, t, r, b
        if fracs["left"] > ink_trigger:
            nl = max(0, l - step)
        if fracs["right"] > ink_trigger:
            nr = min(img_w, r + step)
        if fracs["top"] > ink_trigger:
            nt = max(0, t - step)
        if fracs["bottom"] > ink_trigger:
            nb = min(img_h, b + step)

        new = (nl, nt, nr, nb)
        if new == cur:
            break
        cur = new
        expanded += step

    return cur


def trim_white_margins(
    image: Image.Image,
    *,
    white_thresh: int = 250,
    border_ratio: float = 0.04,
    min_border_px: int = 6,
    whiteness_trigger: float = 0.85,
    pad_px: int = 2,
    min_side_px: int = 40,
) -> TrimResult:
    """
    Trim large near-white margins from an image.

    Args:
        image: PIL image.
        white_thresh: RGB channel threshold considered "white-ish".
        border_ratio: border thickness as ratio of min(h, w).
        min_border_px: minimum border thickness in pixels.
        whiteness_trigger: minimum fraction of border pixels that must be near-white
            before trimming is attempted.
        pad_px: padding to keep around detected content bbox.
        min_side_px: don't trim if the resulting image becomes too small.

    Returns:
        TrimResult with potentially cropped image.
    """
    if image is None:
        return TrimResult(image=image, trimmed=False, bbox=None)  # type: ignore[arg-type]

    im = image.convert("RGB")
    arr = np.asarray(im)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return TrimResult(image=image, trimmed=False, bbox=None)

    h, w = arr.shape[:2]
    if h < min_side_px or w < min_side_px:
        return TrimResult(image=image, trimmed=False, bbox=None)

    b = max(min_border_px, int(min(h, w) * border_ratio))
    b = min(b, h // 2, w // 2)
    if b <= 0:
        return TrimResult(image=image, trimmed=False, bbox=None)

    top = arr[:b, :, :].reshape(-1, 3)
    bottom = arr[h - b :, :, :].reshape(-1, 3)
    left = arr[:, :b, :].reshape(-1, 3)
    right = arr[:, w - b :, :].reshape(-1, 3)
    border = np.concatenate([top, bottom, left, right], axis=0)

    # Border is "white-ish" only if all channels are above threshold.
    border_white = float(((border >= white_thresh).all(axis=1)).mean())
    if border_white < whiteness_trigger:
        return TrimResult(image=image, trimmed=False, bbox=None)

    # Content mask: anything not near-white.
    mask = (arr < white_thresh).any(axis=2)
    if not bool(mask.any()):
        return TrimResult(image=image, trimmed=False, bbox=None)

    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    # Apply padding and clamp.
    x0 = max(0, x0 - pad_px)
    y0 = max(0, y0 - pad_px)
    x1 = min(w - 1, x1 + pad_px)
    y1 = min(h - 1, y1 + pad_px)

    # Ensure meaningful crop (avoid 1px/empty).
    if (x1 - x0 + 1) < min_side_px or (y1 - y0 + 1) < min_side_px:
        return TrimResult(image=image, trimmed=False, bbox=None)

    bbox = (x0, y0, x1 + 1, y1 + 1)  # PIL crop is exclusive of right/bottom
    cropped = im.crop(bbox)

    # If crop doesn't actually change much, keep original.
    if cropped.size == im.size:
        return TrimResult(image=image, trimmed=False, bbox=None)

    return TrimResult(image=cropped, trimmed=True, bbox=bbox)
