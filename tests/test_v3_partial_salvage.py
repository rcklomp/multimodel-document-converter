"""V3 partial-first-element salvage (crucible fix #3, 2026-06-04).

The crucible found the VLM gives up mid-generation on long dense tables
(premature EOS, finish_reason=stop): the one giant table element is left as an
unterminated JSON string, so there are ZERO complete elements and A4's
complete-element recovery returns nothing -> Docling fallback (which drops most
spreadsheet rows). The repair now salvages that partial first element's type +
content so the data survives, with a near-full-page (inset) bbox so a salvaged
TABLE still satisfies QA-CHECK-05 without tripping crop-audit edge-clamp.

Deterministic/offline: synthetic truncated UIR JSON, no VLM.
"""

from __future__ import annotations

from mmrag_v3.engines.vlm_native import (
    _recover_complete_elements,
    repair_truncated_json,
)

# A page whose single TABLE element is cut mid-row (unterminated content string,
# no closing quote / brace / bracket) - the real dense-spreadsheet failure shape.
_HEAD = (
    '{"page_number": 1, "width": 2000, "height": 1500, "classification": "scanned", "elements": ['
)
_TRUNCATED_TABLE = (
    _HEAD
    + '{"type": "table", "content": "| aant | merk | prijs |\\n|---|---|---|\\n'
    + "| 3 | Castrol | 6,55 |\\n| 4 | Gulf | 7,10 |\\n| 2 | Shell | "
)


def test_complete_recovery_finds_nothing_but_salvage_recovers():
    # Baseline: the complete-element pass recovers zero (the only element is cut).
    assert _recover_complete_elements(_TRUNCATED_TABLE) == []

    payload = repair_truncated_json(_TRUNCATED_TABLE)
    assert payload is not None
    assert len(payload["elements"]) == 1
    el = payload["elements"][0]
    assert el["type"] == "table"
    # The markdown table built so far is preserved (header + rows).
    assert "| aant | merk | prijs |" in el["content"]
    assert "Castrol" in el["content"] and "Gulf" in el["content"]
    assert el["content"].count("\n") >= 3


def test_salvaged_bbox_is_inset_not_edge_clamped():
    el = repair_truncated_json(_TRUNCATED_TABLE)["elements"][0]
    x0, y0, x1, y1 = el["bbox"]
    # Near-full-page but inset from every edge (page is 2000x1500).
    assert x0 > 0 and y0 > 0
    assert x1 < 2000 and y1 < 1500
    assert (x1 - x0) > 1800 and (y1 - y0) > 1300  # still essentially full page


def test_salvage_preserves_non_table_type():
    cut_text = _HEAD + '{"type": "text", "content": "A long running paragraph that was cut'
    el = repair_truncated_json(cut_text)["elements"][0]
    assert el["type"] == "text"
    assert el["content"].startswith("A long running paragraph")


def test_complete_elements_take_precedence_over_salvage():
    # When complete elements exist, they are returned (salvage is the empty-case
    # fallback only) - the trailing partial is dropped, as before.
    body = (
        _HEAD
        + '{"type": "text", "content": "first complete element"}, '
        + '{"type": "table", "content": "| a |\\n|---|\\n| cut'
    )
    payload = repair_truncated_json(body)
    assert payload is not None
    assert len(payload["elements"]) == 1
    assert payload["elements"][0]["content"] == "first complete element"


def test_unrecoverable_still_none():
    assert repair_truncated_json("total garbage, no json, no content key") is None
