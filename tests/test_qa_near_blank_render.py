"""Phase B4.a — render-based near-blank page classification
(`docs/PLAN_V2.9.md` §3 Phase B4, `docs/DECISIONS.md` Retrieval-Value Test).

`scripts/qa_full_conversion.py:_page_render_is_near_blank` rasterizes
a page at low DPI and classifies it as blank-equivalent when the
mean pixel intensity is above `_RENDER_BLANK_MEAN_MIN` (245) and the
standard deviation is below `_RENDER_BLANK_STD_MAX` (20).

The wrapping `_read_blank_pages_in_source` additionally requires the
page's text-layer content to be under `_RENDER_BLANK_TEXT_CAP` (200
chars) to avoid misclassifying a real prose page that happens to
render light.

Empirical validation (2026-05-11) on the v2.9 34-doc corpus:
- 0/15 sampled real body-text pages trigger
  (Harry p50/100/150, Bourne p100/150/250, Cronin p100/200/400,
  Adedeji p50/100/200, Combat p30/60/90).
- Catches Python_Distilled's ~697 publisher-template placeholder
  pages (mean ~253, std ~10-15, text 0-200 chars).
- Catches Devlin p2/p264 (mean=255, std=0.0; literally blank).
- Does NOT catch Earthship p109 (mean=128, std=25 — real diagram).

These tests use synthetic Pillow images converted into a minimal
PyMuPDF page surrogate so the threshold logic is tested without
shipping real PDF fixtures in the repository.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from qa_full_conversion import (  # noqa: E402
    _RENDER_BLANK_MEAN_MIN,
    _RENDER_BLANK_NO_TEXT_MEAN_MIN,
    _RENDER_BLANK_STD_MAX,
    _RENDER_BLANK_TEXT_CAP,
    _page_is_no_text_image_only_placeholder,
    _page_render_is_near_blank,
)


class _FakePix:
    """Minimal stand-in for a pymupdf Pixmap that returns the bytes of
    a Pillow image."""

    def __init__(self, pil_image):
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        self._png = buf.getvalue()

    def tobytes(self, fmt: str) -> bytes:
        return self._png


class _FakePage:
    """Minimal stand-in for a pymupdf Page whose get_pixmap returns a
    deterministic surrogate image."""

    def __init__(self, pil_image):
        self._pil = pil_image

    def get_pixmap(self, matrix=None):
        return _FakePix(self._pil)


def _solid_color_image(width: int, height: int, value: int):
    from PIL import Image  # local import — tests already have Pillow

    return Image.new("L", (width, height), color=value)


def _noisy_image(width: int, height: int, mean_value: int, noise_amplitude: int):
    """Image whose pixel values cluster near `mean_value` with uniform
    noise of the requested amplitude."""
    from PIL import Image
    import numpy as np

    rng = np.random.default_rng(seed=42)
    base = rng.integers(
        low=max(0, mean_value - noise_amplitude),
        high=min(255, mean_value + noise_amplitude) + 1,
        size=(height, width),
        dtype=np.uint8,
    )
    return Image.fromarray(base, mode="L")


# ---------------------------------------------------------------------------
# Positive cases — page IS classified as near-blank
# ---------------------------------------------------------------------------


def test_pure_white_page_is_near_blank() -> None:
    page = _FakePage(_solid_color_image(300, 400, 255))
    assert _page_render_is_near_blank(page) is True


def test_mostly_white_with_low_noise_is_near_blank() -> None:
    """Python_Distilled publisher template shape: mean ~253, std ~12.
    The threshold is mean>245 and std<20 — this case trips both."""
    page = _FakePage(_noisy_image(300, 400, mean_value=253, noise_amplitude=15))
    assert _page_render_is_near_blank(page) is True


def test_devlin_p2_shape_is_near_blank() -> None:
    """mean=255 std=0 (literally blank under one image element)."""
    page = _FakePage(_solid_color_image(300, 400, 255))
    assert _page_render_is_near_blank(page) is True


# ---------------------------------------------------------------------------
# Negative cases — page is NOT classified as near-blank
# ---------------------------------------------------------------------------


def test_earthship_p109_shape_NOT_near_blank() -> None:
    """Real content image: mean ~128, std ~25. Must NOT be flagged
    blank — it's a substantive figure."""
    page = _FakePage(_noisy_image(150, 110, mean_value=128, noise_amplitude=40))
    assert _page_render_is_near_blank(page) is False


def test_body_text_page_shape_NOT_near_blank() -> None:
    """Real body-text content: mean ~240, std ~33. Sufficiently
    inked to fail the threshold despite mean being just under 245."""
    page = _FakePage(_noisy_image(300, 400, mean_value=240, noise_amplitude=50))
    assert _page_render_is_near_blank(page) is False


def test_dark_magazine_ad_NOT_near_blank() -> None:
    """Combat-style dark magazine page: mean ~93, std ~85."""
    page = _FakePage(_noisy_image(300, 400, mean_value=93, noise_amplitude=80))
    assert _page_render_is_near_blank(page) is False


def test_light_but_textured_page_NOT_near_blank() -> None:
    """Mean is above the threshold but std exceeds the cap (busy
    page with light backdrop)."""
    page = _FakePage(_noisy_image(300, 400, mean_value=246, noise_amplitude=30))
    # std on 30-amp uniform noise ≈ 17 — borderline. Push the
    # amplitude higher to put std comfortably above 20.
    page2 = _FakePage(_noisy_image(300, 400, mean_value=246, noise_amplitude=45))
    assert _page_render_is_near_blank(page2) is False


def test_just_below_mean_threshold_NOT_blank() -> None:
    """Boundary: mean exactly at 244 should fail (>245 required)."""
    page = _FakePage(_solid_color_image(300, 400, 244))
    assert _page_render_is_near_blank(page) is False


def test_just_above_std_threshold_NOT_blank() -> None:
    """Boundary: std at 25 should fail (<20 required)."""
    page = _FakePage(_noisy_image(300, 400, mean_value=253, noise_amplitude=42))
    assert _page_render_is_near_blank(page) is False


# ---------------------------------------------------------------------------
# Threshold constants pinned (governance: changes require evidence)
# ---------------------------------------------------------------------------


def test_threshold_constants_are_pinned() -> None:
    """The thresholds are empirically tuned. Any change must be
    justified with a corpus-wide false-positive re-run. This test
    pins the values so the change is visible in diffs."""
    assert _RENDER_BLANK_MEAN_MIN == 245.0
    assert _RENDER_BLANK_STD_MAX == 20.0
    assert _RENDER_BLANK_TEXT_CAP == 200


# ---------------------------------------------------------------------------
# Defensive: missing PIL/numpy should not crash
# ---------------------------------------------------------------------------


def test_page_with_broken_pixmap_returns_false() -> None:
    """A page whose get_pixmap raises must be treated as not-near-blank
    rather than crashing the gate."""

    class _ExplodingPage:
        def get_pixmap(self, matrix=None):
            raise RuntimeError("pixmap unavailable")

    assert _page_render_is_near_blank(_ExplodingPage()) is False


# ---------------------------------------------------------------------------
# B4.a refinement — `_page_is_no_text_image_only_placeholder`
# ---------------------------------------------------------------------------


class _FakePageWithMetadata:
    """Stand-in that also exposes get_text() and get_images() so the
    `_page_is_no_text_image_only_placeholder` check can reason about
    the text-layer state."""

    def __init__(self, pil_image, text: str, image_count: int):
        self._pil = pil_image
        self._text = text
        self._image_count = image_count

    def get_pixmap(self, matrix=None):
        return _FakePix(self._pil)

    def get_text(self, mode: str):
        return self._text

    def get_images(self, full: bool = False):
        return [object()] * self._image_count


def test_placeholder_zero_text_high_mean_classified_blank() -> None:
    """Python_Distilled p543 shape: text_len=0, 1 image, render is
    almost-but-not-quite-uniformly white. The stricter zero-text
    rule catches this regardless of std (real placeholder pages may
    have a tiny watermark / page number adding pixel variance).

    NOTE: the synthetic fixture uses a near-solid color image so the
    actual rendered mean stays above the threshold. Production data
    (Python_Distilled p543 etc.) shows mean ~252 in real PDFs."""
    page = _FakePageWithMetadata(
        _noisy_image(300, 400, mean_value=253, noise_amplitude=2),
        text="",
        image_count=1,
    )
    assert _page_is_no_text_image_only_placeholder(page) is True


def test_zero_text_zero_image_NOT_classified() -> None:
    """A page with no text AND no image is handled by the regular
    blank check, not this one. The image-existence precondition
    prevents this rule from firing on the trivial-blank case."""
    page = _FakePageWithMetadata(
        _solid_color_image(300, 400, 255),
        text="",
        image_count=0,
    )
    assert _page_is_no_text_image_only_placeholder(page) is False


def test_real_image_only_diagram_NOT_classified() -> None:
    """Earthship p109 shape: text_len=0, 1 image, mean ~128. The
    mean precondition (mean>250) keeps real diagrams from being
    classified as placeholder."""
    page = _FakePageWithMetadata(
        _noisy_image(150, 110, mean_value=128, noise_amplitude=40),
        text="",
        image_count=1,
    )
    assert _page_is_no_text_image_only_placeholder(page) is False


def test_text_page_NOT_classified() -> None:
    """A page with text content must NEVER match this rule, even
    when its render happens to be near-white."""
    page = _FakePageWithMetadata(
        _solid_color_image(300, 400, 253),
        text="Real content paragraph that should not be dropped.",
        image_count=1,
    )
    assert _page_is_no_text_image_only_placeholder(page) is False


def test_placeholder_with_text_below_threshold_does_match_other_check() -> None:
    """Sanity: a page with text < 200 chars but otherwise blank-rendered
    is caught by `_page_render_is_near_blank` (the broader rule), not
    by `_page_is_no_text_image_only_placeholder` (which requires
    text_len == 0). Both rules are wired into
    `_read_blank_pages_in_source`; the test pins the responsibility
    split."""
    page_with_short_text = _FakePageWithMetadata(
        _solid_color_image(300, 400, 253),
        text="iv",  # a roman numeral, < 200 chars
        image_count=1,
    )
    # The no-text rule does not fire (text is non-empty).
    assert _page_is_no_text_image_only_placeholder(page_with_short_text) is False
    # The render-near-blank rule does fire (its caller checks the
    # text length cap; here we just confirm the render passes).
    assert _page_render_is_near_blank(page_with_short_text) is True


def test_no_text_threshold_constant_pinned() -> None:
    assert _RENDER_BLANK_NO_TEXT_MEAN_MIN == 250.0
