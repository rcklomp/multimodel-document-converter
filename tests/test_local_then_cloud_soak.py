"""v2.14 Phase 4d — unit tests for the tie-breaker soak harness.

Pure-function coverage on the contested-detection + judgment-parsing
helpers. Network paths (`_call_vllm`, `_call_dashscope`) are exercised
in the live smoke described in `docs/PLAN_V2.14.md` §"Phase 4d".
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


@pytest.fixture(scope="module")
def harness():
    """Import the tie-breaker script as a module so we can call its
    helpers directly (it's a CLI script; no __init__.py)."""
    spec = importlib.util.spec_from_file_location(
        "local_then_cloud_soak", SCRIPTS / "local_then_cloud_soak.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── _is_contested ────────────────────────────────────────────────────


def test_is_contested_none_judgment_is_contested(harness):
    """Missing judgment must always be contested (need cloud re-judge)."""
    assert harness._is_contested(None, axis_floor=2) is True


def test_is_contested_perfect_judgment_not_contested(harness):
    """2/2/2 across all three axes is the uncontested case."""
    perfect = {"relevance": 2, "format": 2, "faithfulness": 2, "rationale": "ok"}
    assert harness._is_contested(perfect, axis_floor=2) is False


def test_is_contested_any_axis_below_floor_is_contested(harness):
    """Any single axis strictly below the floor triggers contested."""
    cases = [
        {"relevance": 1, "format": 2, "faithfulness": 2},
        {"relevance": 2, "format": 0, "faithfulness": 2},
        {"relevance": 2, "format": 2, "faithfulness": 1},
        {"relevance": 0, "format": 0, "faithfulness": 0},
    ]
    for c in cases:
        assert harness._is_contested(c, axis_floor=2) is True, c


def test_is_contested_floor_zero_only_flags_missing_axes(harness):
    """With floor=0, scores of 0 are NOT contested (0 >= 0). Only
    parse-failures / missing axes trip the contested check."""
    assert harness._is_contested({"relevance": 0, "format": 0, "faithfulness": 0},
                                 axis_floor=0) is False
    # Missing axis is still contested (the int() of -1 default < 0).
    assert harness._is_contested({"relevance": 0, "format": 0},
                                 axis_floor=0) is True


def test_is_contested_non_int_axis_value_is_contested(harness):
    """Non-int axis values (corruption / parse partial) must be flagged
    contested so the cloud tie-break re-runs them rather than silently
    treating them as 0."""
    assert harness._is_contested(
        {"relevance": "two", "format": 2, "faithfulness": 2}, axis_floor=2,
    ) is True


# ── _parse_judgment ──────────────────────────────────────────────────


def test_parse_judgment_none_input(harness):
    assert harness._parse_judgment(None) is None
    assert harness._parse_judgment("") is None


def test_parse_judgment_clean_json(harness):
    raw = '{"relevance": 2, "format": 1, "faithfulness": 2, "rationale": "ok"}'
    parsed = harness._parse_judgment(raw)
    assert parsed == {
        "relevance": 2, "format": 1, "faithfulness": 2, "rationale": "ok",
    }


def test_parse_judgment_with_markdown_fences(harness):
    """The judge sometimes wraps output in ```json fences; `_extract_json`
    in synthetic_soak strips them."""
    raw = '```json\n{"relevance": 1, "format": 2, "faithfulness": 0, "rationale": "x"}\n```'
    parsed = harness._parse_judgment(raw)
    assert parsed is not None
    assert parsed["relevance"] == 1
    assert parsed["faithfulness"] == 0


def test_parse_judgment_string_axis_returns_none(harness):
    """Non-castable axis values produce None (caller treats as contested)."""
    raw = '{"relevance": "two", "format": 2, "faithfulness": 2, "rationale": ""}'
    assert harness._parse_judgment(raw) is None


def test_parse_judgment_missing_relevance_returns_none(harness):
    """A judgment without `relevance` is rejected — required field."""
    raw = '{"format": 2, "faithfulness": 2, "rationale": ""}'
    assert harness._parse_judgment(raw) is None


# ── stage_cloud_tiebreak provenance ─────────────────────────────────


def test_cloud_tiebreak_skips_uncontested_writes_local_provenance(harness, monkeypatch):
    """An uncontested query (local rated 2/2/2) must NOT call cloud;
    its `judgment` must come from `judgment_local` with provenance
    `local`."""
    rows = [{
        "sample_id": "S1",
        "gold_content": "...",
        "queries": [{
            "query_id": "S1.Q1",
            "query_text": "anything",
            "retrieval": {"top_k": [{"content": "x", "source_file": "f", "page_number": 1}]},
            "judgment_local": {
                "relevance": 2, "format": 2, "faithfulness": 2, "rationale": "ok",
            },
            "_tiebreak_in_scope": True,
        }],
    }]
    # Sentinel that fires if cloud is called.
    called = []
    monkeypatch.setattr(harness, "_call_dashscope",
                        lambda *a, **k: called.append((a, k)) or "")
    cloud_calls, uncontested, contested = harness.stage_cloud_tiebreak(
        rows, api_key="sk-fake", axis_floor=2, cloud_model="qwen-max",
    )
    assert called == [], "uncontested case must not call cloud"
    assert cloud_calls == 0
    assert contested == 0
    assert uncontested == 1
    j = rows[0]["queries"][0]["judgment"]
    assert j["relevance"] == 2 and j["format"] == 2 and j["faithfulness"] == 2
    assert j["judge_source"] == "local"


def test_cloud_tiebreak_contested_calls_cloud_and_tags_provenance(harness, monkeypatch):
    """A contested query (local rated 1 on format) must trigger one
    cloud call; result tagged `cloud`."""
    rows = [{
        "sample_id": "S1",
        "gold_content": "...",
        "queries": [{
            "query_id": "S1.Q1",
            "query_text": "anything",
            "retrieval": {"top_k": [{"content": "x", "source_file": "f", "page_number": 1}]},
            "judgment_local": {
                "relevance": 2, "format": 1, "faithfulness": 2, "rationale": "borderline",
            },
            "_tiebreak_in_scope": True,
        }],
    }]
    cloud_response = '{"relevance": 2, "format": 2, "faithfulness": 2, "rationale": "actually fine"}'
    monkeypatch.setattr(harness, "_call_dashscope",
                        lambda *a, **k: cloud_response)
    cloud_calls, uncontested, contested = harness.stage_cloud_tiebreak(
        rows, api_key="sk-fake", axis_floor=2, cloud_model="qwen-max",
    )
    assert cloud_calls == 1
    assert contested == 1
    assert uncontested == 0
    j = rows[0]["queries"][0]["judgment"]
    assert j["relevance"] == 2 and j["format"] == 2 and j["faithfulness"] == 2
    assert j["judge_source"] == "cloud"


def test_cloud_tiebreak_respects_in_scope_filter(harness, monkeypatch):
    """Queries not flagged `_tiebreak_in_scope=True` must be ignored
    entirely (no cloud call, no provenance change)."""
    rows = [{
        "sample_id": "S1",
        "gold_content": "...",
        "queries": [{
            "query_id": "S1.Q1", "query_text": "anything",
            "retrieval": {"top_k": [{"content": "x", "source_file": "f", "page_number": 1}]},
            "judgment_local": {"relevance": 0, "format": 0, "faithfulness": 0,
                               "rationale": "very contested"},
            "_tiebreak_in_scope": False,
        }],
    }]
    monkeypatch.setattr(harness, "_call_dashscope",
                        lambda *a, **k: pytest.fail("cloud should not be called"))
    cloud_calls, uncontested, contested = harness.stage_cloud_tiebreak(
        rows, api_key="sk-fake", axis_floor=2, cloud_model="qwen-max",
    )
    assert cloud_calls == 0
    assert uncontested == 0
    assert contested == 0
    assert "judgment" not in rows[0]["queries"][0]


def test_cloud_tiebreak_cloud_parse_failure_keeps_local_as_fallback(harness, monkeypatch):
    """If cloud is called but returns unparseable junk, the local
    judgment is preserved with provenance `local_fallback` (caller
    can post-hoc filter / re-run)."""
    rows = [{
        "sample_id": "S1", "gold_content": "...",
        "queries": [{
            "query_id": "S1.Q1", "query_text": "anything",
            "retrieval": {"top_k": [{"content": "x", "source_file": "f", "page_number": 1}]},
            "judgment_local": {"relevance": 1, "format": 1, "faithfulness": 1,
                               "rationale": "uncertain"},
            "_tiebreak_in_scope": True,
        }],
    }]
    monkeypatch.setattr(harness, "_call_dashscope", lambda *a, **k: "garbled")
    cloud_calls, uncontested, contested = harness.stage_cloud_tiebreak(
        rows, api_key="sk-fake", axis_floor=2, cloud_model="qwen-max",
    )
    assert contested == 1
    assert cloud_calls == 0, "parse-failed cloud calls don't count as successful"
    j = rows[0]["queries"][0]["judgment"]
    assert j["judge_source"] == "local_fallback"
    assert j["relevance"] == 1
