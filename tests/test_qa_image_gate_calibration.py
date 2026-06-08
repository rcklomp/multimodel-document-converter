"""Phase 3 Step 3 — gate calibration regressions.

Two surfaces are calibrated together:

- ``scripts/qa_full_conversion.py`` ``_is_blankish_visual_description``
- ``scripts/qa_semantic_fidelity.py`` ``is_placeholder_image_or_table``

Both gain:
1. The F4 hard-fallback exemption: a chunk with
   ``vision_status="hard_fallback"`` AND both ``vision_error`` and
   ``vision_provider_used`` set is a documented no-VLM-signal state,
   not a placeholder row.
2. (qa_full_conversion only) Asset-complexity-aware short-description
   handling: ``simple`` assets accept short descriptions, others do not.

These tests pin both behaviors plus their backward-compatibility
fallback (when called with the old plain-string signature).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Module loaders — the QA scripts are not packaged; load them by file path.
# ---------------------------------------------------------------------------


def _load_script(name: str, relative_path: str):
    import sys
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        name, repo_root / relative_path
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so the module's @dataclass decorators can find
    # forward-reference annotations via sys.modules during type resolution.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def qa_full():
    return _load_script("qa_full_conversion", "scripts/qa_full_conversion.py")


@pytest.fixture(scope="module")
def qa_sem():
    return _load_script("qa_semantic_fidelity", "scripts/qa_semantic_fidelity.py")


# ---------------------------------------------------------------------------
# qa_full_conversion._is_blankish_visual_description
# ---------------------------------------------------------------------------


def test_blankish_returns_true_on_empty_string(qa_full):
    assert qa_full._is_blankish_visual_description("")
    assert qa_full._is_blankish_visual_description("   ")


def test_blankish_returns_true_on_short_string_no_chunk(qa_full):
    """Backward-compat: short string + no chunk → still blankish."""
    assert qa_full._is_blankish_visual_description("Pie chart.")


def test_blankish_returns_false_on_layout_keyword_short(qa_full):
    """Pre-Phase-3 carve-out for 'layout' descriptions stays."""
    assert not qa_full._is_blankish_visual_description("Two-column layout.")


def test_blankish_simple_asset_short_description_allowed(qa_full):
    """Simple asset (small bbox + tiny file) → short description NOT blankish."""
    chunk = {
        "modality": "image",
        "metadata": {
            "spatial": {"bbox": [0, 0, 100, 100], "page_width": 612, "page_height": 792},
        },
    }
    # 15 chars — would have been blankish under the old flat rule.
    assert not qa_full._is_blankish_visual_description(
        "Bullet icon.", chunk=chunk
    )


def test_blankish_complex_asset_short_description_still_blankish(qa_full):
    """Complex asset (medium bbox) → short description IS blankish."""
    chunk = {
        "modality": "image",
        "metadata": {
            "spatial": {"bbox": [100, 100, 600, 500], "page_width": 612, "page_height": 792},
        },
    }
    assert qa_full._is_blankish_visual_description("Pie chart.", chunk=chunk)


def test_blankish_text_heavy_asset_short_description_still_blankish(qa_full):
    """Full-page bbox → text_heavy → short description IS blankish."""
    chunk = {
        "modality": "image",
        "metadata": {
            "spatial": {"bbox": [0, 0, 1000, 800], "page_width": 612, "page_height": 792},
        },
    }
    assert qa_full._is_blankish_visual_description("Screenshot.", chunk=chunk)


def test_blankish_long_description_never_blankish(qa_full):
    """Descriptions of 20+ chars (without 'layout') are NOT blankish."""
    chunk = {
        "modality": "image",
        "metadata": {
            "spatial": {"bbox": [0, 0, 1000, 800], "page_width": 612, "page_height": 792},
        },
    }
    desc = "Detailed pie chart with three colored segments and external labels."
    assert not qa_full._is_blankish_visual_description(desc, chunk=chunk)


def test_blankish_hard_fallback_with_full_metadata_exempt(qa_full):
    """F4 exemption: hard_fallback + vision_error + vision_provider_used → not blankish."""
    chunk = {
        "modality": "image",
        "metadata": {
            "vision_status": "hard_fallback",
            "vision_error": "[VLM_FAILED: call error]",
            "vision_provider_used": "qwen3-vl-plus",
            "spatial": {"bbox": [100, 100, 600, 500], "page_width": 612, "page_height": 792},
        },
    }
    # The placeholder-shaped description on a hard_fallback chunk would
    # normally be blankish — the exemption returns False.
    assert not qa_full._is_blankish_visual_description(
        "[Figure on page 5]", chunk=chunk
    )


def test_blankish_hard_fallback_missing_error_NOT_exempt(qa_full):
    """F4 contract requires BOTH vision_error and vision_provider_used.
    Missing either → exemption does not apply."""
    chunk = {
        "modality": "image",
        "metadata": {
            "vision_status": "hard_fallback",
            "vision_error": "",  # missing
            "vision_provider_used": "qwen3-vl-plus",
            "spatial": {"bbox": [100, 100, 600, 500], "page_width": 612, "page_height": 792},
        },
    }
    assert qa_full._is_blankish_visual_description(
        "[Figure on page 5]", chunk=chunk
    )


def test_blankish_placeholder_pattern_always_blankish(qa_full):
    """The PLACEHOLDER_VISUAL_RE check still fires on '[Figure on page N]'
    for non-hard-fallback chunks regardless of complexity."""
    chunk = {
        "modality": "image",
        "metadata": {
            "vision_status": "complete",
            "spatial": {"bbox": [0, 0, 1000, 800], "page_width": 612, "page_height": 792},
        },
    }
    assert qa_full._is_blankish_visual_description("[Figure on page 5]", chunk=chunk)


# ---------------------------------------------------------------------------
# qa_semantic_fidelity.is_placeholder_image_or_table
# ---------------------------------------------------------------------------


def test_sem_placeholder_pre_phase3_logic_preserved(qa_sem):
    """Backward-compat: chunk-less call returns the same answers as before."""
    assert qa_sem.is_placeholder_image_or_table("[Figure on page 5]")
    assert qa_sem.is_placeholder_image_or_table("[Image extraction unavailable]")
    assert qa_sem.is_placeholder_image_or_table("[VLM_FAILED: call error]")
    assert not qa_sem.is_placeholder_image_or_table(
        "Detailed bar chart with three colored bars showing quarterly revenue."
    )


def test_sem_placeholder_table_calls_unaffected(qa_sem):
    """Tables don't have a vision pipeline; chunk-less call must still work."""
    assert qa_sem.is_placeholder_image_or_table("[Table on page 3]")
    assert not qa_sem.is_placeholder_image_or_table(
        "| Col A | Col B |\n| --- | --- |\n| 1 | 2 |"
    )


def test_sem_placeholder_hard_fallback_with_full_metadata_exempt(qa_sem):
    """F4 exemption applies the same way it does in qa_full_conversion."""
    chunk = {
        "modality": "image",
        "content": "[Figure on page 5]",
        "metadata": {
            "vision_status": "hard_fallback",
            "vision_error": "[VLM_FAILED: call error]",
            "vision_provider_used": "qwen3-vl-plus",
        },
    }
    assert not qa_sem.is_placeholder_image_or_table(chunk["content"], chunk=chunk)


def test_sem_placeholder_hard_fallback_missing_error_NOT_exempt(qa_sem):
    """Same conjunction requirement as qa_full_conversion."""
    chunk = {
        "modality": "image",
        "content": "[Figure on page 5]",
        "metadata": {
            "vision_status": "hard_fallback",
            "vision_error": "",  # missing — exemption denied
            "vision_provider_used": "qwen3-vl-plus",
        },
    }
    assert qa_sem.is_placeholder_image_or_table(chunk["content"], chunk=chunk)


def test_sem_placeholder_complete_status_not_exempt(qa_sem):
    """Only hard_fallback gets the exemption; complete chunks are evaluated
    on content as usual."""
    chunk = {
        "modality": "image",
        "content": "[Figure on page 5]",
        "metadata": {
            "vision_status": "complete",
            "vision_provider_used": "qwen3-vl-plus",
        },
    }
    assert qa_sem.is_placeholder_image_or_table(chunk["content"], chunk=chunk)


# ---------------------------------------------------------------------------
# Cluster D (2026-06-06): no-VLM image fallback is advisory, not a failure.
# A multimodal converter keeps image chunks even with --vision-provider none,
# retaining them as ID-only fallbacks (vision_status="no_vlm"). The strict gate
# must treat these as a documented advisory (IMAGE_NO_VLM WARN), never as
# VISION_PENDING / IMAGE_DESCRIPTION_UNUSABLE failures.
# ---------------------------------------------------------------------------


def _no_vlm_image(page: int, fname: str) -> dict:
    return {
        "modality": "image",
        "content": "",
        "asset_ref": {"file_path": f"assets/{fname}"},
        "visual_description": f"[image: {fname}]",
        "metadata": {
            "page_number": page,
            "vision_status": "no_vlm",
            "vision_provider_used": "none",
            "vision_error": "no vision provider configured (--vision-provider none)",
        },
    }


def test_no_vlm_images_are_advisory_not_failure(qa_full):
    issues = qa_full._image_issues(
        [_no_vlm_image(1, "a.png"), _no_vlm_image(2, "b.png")],
        max_hard_fallback_ratio=0.5,
        require_image_descriptions=True,
    )
    codes = {(i.severity, i.code) for i in issues}
    assert ("WARN", "IMAGE_NO_VLM") in codes
    # The no-VLM state must NOT raise the description/pending hard failures.
    assert not any(
        i.severity == "FAIL"
        and i.code in {"VISION_PENDING", "IMAGE_DESCRIPTION_UNUSABLE"}
        for i in issues
    )


def test_image_no_vlm_is_documented_advisory(qa_full):
    warn = qa_full.Issue("WARN", "IMAGE_NO_VLM", "2/2 image chunk(s) ID-only fallback")
    assert qa_full._warn_is_documented_advisory(warn, []) is True
