"""PLAN_QUALITY_REMEDIATION_V1 WP-0.3: the source-anchored acceptance script.

The script must pass a clean conversion, fail on each defect class it anchors, and never
count an unlabelled heading as a pass (N/A), so a run cannot look good by dodging.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO / "tests" / "fixtures" / "gold_anchor_specs" / "irjet.json"


def _load():
    spec = importlib.util.spec_from_file_location(
        "qa_gold_anchor_smoke", REPO / "scripts" / "qa_gold_anchor_smoke.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MINI = {
    "min_evaluable_section_anchors": 2,
    "sections": {"a": "^Alpha$", "b": "^Beta$"},
    "section_anchors": [
        {"id": "a1", "phrase": "first sentence", "section": "a"},
        {"id": "b1", "phrase": "second sentence", "section": "b"},
    ],
    "front_matter_anchors": [
        {"id": "abs", "phrase": "abstract words", "must_not_parent": "^Alpha$"}
    ],
    "furniture_absent": ["Journal Header"],
    "reference_anchors": [{"id": "r1", "all_of": ["Smith", "vol\\. 4, no\\. 8", "2013"]}],
    "table_anchors": [{"id": "t1", "all_of": ["37\\.08"], "min_rows": 3}],
}


def T(cid, content, parent=None, modality="text"):
    return {
        "chunk_id": cid,
        "modality": modality,
        "content": content,
        "metadata": {"page_number": 1, "hierarchy": {"parent_heading": parent}},
    }


GOOD = [
    T("f", "abstract words here", None),
    T("a", "Alpha body: the first sentence lives here.", "Alpha"),
    T("b", "Beta body: the second sentence lives here.", "Beta"),
    T("r", "[1] J. Smith, Title, Journal, vol. 4, no. 8, 2013.", "Beta"),
    T("t", "| Rated power | 37.08Wp |\n| --- | --- |\n| V | 16 |\n| I | 2 |", None, "table"),
]


def verdicts(rows, spec=MINI):
    return {aid: v for aid, v, _ in _load().evaluate(rows, spec)}


def test_clean_output_passes_every_anchor():
    v = verdicts(GOOD)
    assert set(v.values()) == {"PASS"}, v


def test_wrong_parent_fails_the_section_anchor():
    # another chunk still carries "Alpha" (the engine labelled it), but the anchored one does not
    rows = [dict(r) for r in GOOD] + [T("a2", "Other Alpha text.", "Alpha")]
    rows[1] = T("a", "Alpha body: the first sentence lives here.", "Beta")
    assert verdicts(rows)["section:a1"] == "FAIL"


def test_abstract_under_the_wrong_parent_fails_the_front_matter_anchor():
    rows = [dict(r) for r in GOOD]
    rows[0] = T("f", "abstract words here", "Alpha")
    assert verdicts(rows)["front:abs"] == "FAIL"


def test_furniture_string_in_any_chunk_fails():
    rows = GOOD + [T("h", "Journal Header e-ISSN 1234\nBody text", "Beta")]
    assert verdicts(rows)["furniture:Journal Header"] == "FAIL"


def test_reference_entry_split_across_chunks_fails_and_intact_passes():
    rows = [r for r in GOOD if r["chunk_id"] != "r"] + [
        T("r1", "[1] J. Smith, Title, Journal, vol. 4, no.", "Beta"),
        T("r2", "8, 2013.", "Beta"),
    ]
    assert verdicts(rows)["reference:r1"] == "FAIL"
    assert verdicts(GOOD)["reference:r1"] == "PASS"


def test_table_with_too_few_rows_fails():
    rows = [r for r in GOOD if r["chunk_id"] != "t"] + [
        T("t", "| Rated power | 37.08Wp |\n| --- | --- |", None, "table")
    ]
    assert verdicts(rows)["table:t1"] == "FAIL"


def test_unlabelled_heading_is_na_never_pass():
    # no chunk carries "Beta" as a parent any more: the engine did not label that heading
    rows = [r for r in GOOD if r["chunk_id"] not in ("b", "r")] + [
        T("b", "Beta body: the second sentence lives here.", None),
        T("r", "[1] J. Smith, Title, Journal, vol. 4, no. 8, 2013.", None),
    ]
    assert verdicts(rows)["section:b1"] == "NA"


def test_anchor_text_missing_is_a_content_loss_fail_even_when_unlabelled():
    rows = [r for r in GOOD if r["chunk_id"] != "b"]
    assert verdicts(rows)["section:b1"] == "FAIL"


def test_too_few_evaluable_section_anchors_is_inconclusive(tmp_path, capsys):
    mod = _load()
    rows = [
        T("a", "Alpha body: the first sentence lives here.", None),
        T("b", "Beta body: the second sentence lives here.", None),
        T("f", "abstract words here", None),
        T("r", "[1] J. Smith, vol. 4, no. 8, 2013.", None),
        T("t", "| Rated power | 37.08Wp |\n| --- | --- |\n| V | 16 |\n| I | 2 |", None, "table"),
    ]
    jl = tmp_path / "ingestion.jsonl"
    spec = tmp_path / "spec.json"
    jl.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    spec.write_text(json.dumps({**MINI, "furniture_absent": []}), encoding="utf-8")
    rc = mod.main([str(jl), "--spec", str(spec)])
    assert rc == 1
    assert "GOLD_ANCHOR_INCONCLUSIVE" in capsys.readouterr().out


def test_real_spec_is_valid_and_every_regex_compiles():
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    for rx in spec["sections"].values():
        re.compile(rx)
    for a in spec["section_anchors"]:
        re.compile(a["phrase"])
        assert a["section"] in spec["sections"]
    for a in spec.get("reference_anchors", []) + spec.get("table_anchors", []):
        for p in a["all_of"]:
            re.compile(p)
    assert len(spec["section_anchors"]) >= spec["min_evaluable_section_anchors"]


BASELINES = ["cloud_probe_irjet", "irjet_baseline_499a5fa", "vlmtest_IRJET"]


@pytest.mark.parametrize("name", BASELINES)
def test_pre_fix_outputs_fail_many_anchors_proving_discrimination(name):
    path = REPO / "output" / name / "ingestion.jsonl"
    if not path.exists():
        pytest.skip("gitignored local output not present")
    mod = _load()
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    results = mod.evaluate(mod.load_rows(path), spec)
    fails = [aid for aid, v, _ in results if v == "FAIL"]
    assert len(fails) >= 8, fails
    assert any(a.startswith("furniture:") for a in fails)
    assert any(a.startswith("reference:") for a in fails)


# ------------------------------------------------------------------ figure and priority anchors (Round 0b N-1, N-13)
FIG_SPEC = {
    "crop_zoom": 2.0,
    "figure_area_band": [0.5, 2.0],
    "figure_anchors": [
        {"id": "big", "page": 5, "gold_rect_pt": [300, 100, 560, 560]},
        {"id": "small", "page": 5, "gold_rect_pt": [40, 600, 250, 700]},
    ],
    "priority_anchors": [{"id": "abs", "phrase": "the abstract text", "must_not_priority": "low"}],
}


def IMG(cid, w, h, page=5):
    return {
        "chunk_id": cid,
        "modality": "image",
        "content": "d",
        "asset_ref": {"width_px": w, "height_px": h},
        "metadata": {"page_number": page, "hierarchy": {"parent_heading": None}},
    }


def fig_verdicts(rows):
    return {aid: (v, d) for aid, v, d in _load().evaluate(rows, FIG_SPEC)}


def test_full_size_crops_pass_the_figure_anchors():
    v = fig_verdicts([IMG("a", 520, 920), IMG("b", 420, 200)])
    assert v["figure:big"][0] == "PASS" and v["figure:small"][0] == "PASS"


def test_a_fragment_crop_fails_even_though_an_image_chunk_exists():
    # the IRJET shapes: a 420x43 text strip and a 66x68 logo where whole figures should be
    v = fig_verdicts([IMG("strip", 420, 43), IMG("logo", 66, 68)])
    assert v["figure:big"][0] == "FAIL" and "fragment" in v["figure:big"][1]
    assert v["figure:small"][0] == "FAIL"


def test_a_dropped_figure_fails_and_is_not_rescued_by_a_manifest():
    v = fig_verdicts([IMG("a", 520, 920)])
    assert v["figure:big"][0] == "PASS"
    assert v["figure:small"][0] == "FAIL" and "dropped or never emitted" in v["figure:small"][1]


def test_one_image_chunk_cannot_satisfy_two_figures():
    v = fig_verdicts([IMG("only", 520, 920)])
    assert [
        k for k, (verdict, _) in v.items() if k.startswith("figure:") and verdict == "PASS"
    ] == ["figure:big"]


def test_image_on_another_page_does_not_count():
    v = fig_verdicts([IMG("elsewhere", 520, 920, page=4)])
    assert v["figure:big"][0] == "FAIL"


def test_priority_anchor_fails_when_the_chunk_is_demoted_to_low():
    low = T("x", "the abstract text goes here")
    low["metadata"]["search_priority"] = "low"
    ok = T("y", "the abstract text goes here")
    ok["metadata"]["search_priority"] = "high"
    assert fig_verdicts([low])["priority:abs"][0] == "FAIL"
    assert fig_verdicts([ok])["priority:abs"][0] == "PASS"
