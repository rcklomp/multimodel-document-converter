"""VLM render longest-side cap (DECISIONS.md cap1600 interim, 2026-06-10).

Contract: ``render_page_png`` clamps the longest rendered side to
``VLM_RENDER_MAX_PX`` (default 1600) so a large-format page can never balloon
to the unbounded-DPI sizes (up to ~19k px) that trip the VLM into degenerate
repetition. Small pages keep pure-DPI rendering; ``VLM_RENDER_MAX_PX=0`` is
the rollback lever restoring uncapped behavior.
"""

import fitz
import pytest

from mmrag_v3.engines.vlm_native import (
    PAGE_RENDER_DPI,
    VLM_RENDER_MAX_PX,
    render_page_png,
)


def _page(width_pts: float, height_pts: float) -> "fitz.Page":
    doc = fitz.open()
    return doc.new_page(width=width_pts, height=height_pts)


def test_large_page_clamped_to_cap():
    # 2000x3000 pts would render 8333px tall at dpi200 without the cap.
    _, w, h = render_page_png(_page(2000, 3000))
    assert max(w, h) <= VLM_RENDER_MAX_PX
    # Aspect ratio preserved (uniform zoom).
    assert abs(w / h - 2000 / 3000) < 0.01


def test_small_page_keeps_pure_dpi_render():
    # 300x400 pts at dpi200 -> ~833x1111, under the cap: must NOT be upscaled
    # or otherwise altered by the cap (PyMuPDF ceils pixel dims, hence the
    # 1px tolerance).
    _, w, h = render_page_png(_page(300, 400))
    assert abs(w - 300 * PAGE_RENDER_DPI / 72) <= 1
    assert abs(h - 400 * PAGE_RENDER_DPI / 72) <= 1


def test_env_override_tightens_cap(monkeypatch):
    monkeypatch.setenv("VLM_RENDER_MAX_PX", "800")
    _, w, h = render_page_png(_page(2000, 3000))
    assert max(w, h) <= 800


def test_env_zero_disables_cap(monkeypatch):
    # The rollback lever: 0 restores unbounded pure-DPI rendering.
    monkeypatch.setenv("VLM_RENDER_MAX_PX", "0")
    _, w, h = render_page_png(_page(2000, 3000))
    assert abs(max(w, h) - 3000 * PAGE_RENDER_DPI / 72) <= 1


def test_env_garbage_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("VLM_RENDER_MAX_PX", "not-a-number")
    _, w, h = render_page_png(_page(2000, 3000))
    assert max(w, h) <= VLM_RENDER_MAX_PX


@pytest.mark.parametrize("dpi", [150, 200])
def test_cap_applies_across_dpi_settings(dpi):
    _, w, h = render_page_png(_page(2500, 2500), render_dpi=dpi)
    assert max(w, h) <= VLM_RENDER_MAX_PX
