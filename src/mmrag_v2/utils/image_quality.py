"""
Image quality utilities for lightweight blur estimation.

Uses Laplacian variance as a proxy for sharpness. Avoids hard dependency on
OpenCV; falls back to numpy gradients if cv2 is unavailable.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence, List

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore


def estimate_blur_variance(image_path: str) -> Optional[float]:
    """Estimate blur via Laplacian variance; higher = sharper."""
    try:
        if cv2:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            return float(cv2.Laplacian(img, cv2.CV_64F).var())
        # PIL + numpy fallback
        if Image is None:
            return None
        with Image.open(image_path) as im:
            gray = im.convert("L")
            arr = np.asarray(gray, dtype=np.float32)
            # simple Laplacian approximation via gradients
            gx, gy = np.gradient(arr)
            lap = np.gradient(gx)[0] + np.gradient(gy)[1]
            return float(lap.var())
    except Exception:
        return None


def sample_blur_variance(
    assets: Sequence[Path], sample_size: int = 10
) -> List[float]:
    """Sample up to sample_size images and return blur variances."""
    scores: List[float] = []
    if not assets:
        return scores
    # evenly spaced sample
    step = max(1, math.floor(len(assets) / sample_size))
    for p in assets[::step][:sample_size]:
        s = estimate_blur_variance(str(p))
        if s is not None:
            scores.append(s)
    return scores
