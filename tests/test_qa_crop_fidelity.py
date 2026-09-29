"""WP-A1: the advisory crop-fidelity report flags the crops a count-based check cannot see."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import fitz

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "qa_crop_fidelity", REPO / "scripts" / "qa_crop_fidelity.py"
)
qa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qa)


def rec(
    page=1, modality="image", source="geometric", reason="geometric_rescue", clip=(40, 40, 74, 74)
):
    return {
        "asset_ref": f"assets/x_{page}.png",
        "page": page,
        "modality": modality,
        "crop_source": source,
        "crop_reason": reason,
        "vlm_rect_pt": None,
        "clip_rect_pt": list(clip) if clip else None,
        "asset_px": [67, 68],
    }


def _prose_pdf(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    for i in range(30):
        page.insert_text(
            (72, 100 + i * 14), "Body prose fills this region of the page completely.", fontsize=10
        )
    out = tmp_path / "p.pdf"
    doc.save(str(out))
    doc.close()
    return fitz.open(str(out))


def test_a_tiny_geometric_rescue_is_flagged_as_a_sprite():
    out = qa.evaluate([rec()], None)
    assert [f["flag"] for f in out["flagged"]] == ["tiny_rescue"]


def test_a_large_geometric_rescue_is_not_flagged():
    out = qa.evaluate([rec(clip=(60, 60, 300, 300))], None)
    assert out["flagged"] == []


def test_a_crop_over_body_prose_is_flagged(tmp_path):
    doc = _prose_pdf(tmp_path)
    out = qa.evaluate(
        [rec(source="vlm", reason="vlm_kept_graphics_in_box", clip=(60, 90, 500, 500))], doc
    )
    assert [f["flag"] for f in out["flagged"]] == ["prose_dominated"]
    assert out["flagged"][0]["prose_fraction"] >= 0.5


def test_a_full_page_crop_is_flagged_and_tables_are_ignored():
    out = qa.evaluate(
        [rec(clip=None, source="full_page", reason=""), rec(modality="table", clip=(1, 1, 5, 5))],
        None,
    )
    assert [f["flag"] for f in out["flagged"]] == ["full_page"]


def test_cli_reports_advisory_and_exits_zero(tmp_path, capsys):
    side = tmp_path / "crop_audit.json"
    side.write_text(json.dumps({"crops": [rec()]}), encoding="utf-8")
    assert qa.main([str(side)]) == 0
    text = capsys.readouterr().out
    assert "CROP_FIDELITY_FLAG tiny_rescue" in text and "CROP_FIDELITY_ADVISORY" in text


def test_cli_reports_clean_when_nothing_is_flagged(tmp_path, capsys):
    side = tmp_path / "crop_audit.json"
    side.write_text(json.dumps({"crops": [rec(clip=(60, 60, 300, 300))]}), encoding="utf-8")
    assert qa.main([str(side)]) == 0
    assert "CROP_FIDELITY_CLEAN" in capsys.readouterr().out
