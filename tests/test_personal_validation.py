"""v2.16 Phase 1 — tests for run_personal_validation.py.

Covers:
  - Fixture schema validation (load_fixture).
  - Per-query evaluator on synthetic retrieval results (no live retrieval).
  - Doc-id resolution from production qdrant payload shape.
  - render_report shape sanity.

Live retrieval is exercised by the runner itself (CI runs in dry-run);
these tests use mock retrieval to validate the gating logic in isolation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import run_personal_validation as rpv  # noqa: E402


def _mock_chunk(*, doc_id_hash: str, modality: str, content: str,
                chunk_id: str = "ck1", source_file: str = "X.pdf") -> dict:
    return {
        "id": chunk_id,
        "score": 0.9,
        "payload": {
            "chunk_id": chunk_id,
            "doc_id": doc_id_hash,
            "source_file": source_file,
            "modality": modality,
            "content": content,
        },
        "rerank_score": 1.0,
        "rerank_index": 0,
    }


def test_fixture_loader_rejects_missing_expected_anchor_regexes(tmp_path):
    fx = tmp_path / "bad.json"
    fx.write_text(json.dumps({
        "class": "Foo",
        "personal_importance": "HIGH",
        "target_pass_rate": 0.85,
        "queries": [{
            "id": "Q1",
            "query_text": "?",
            "expected": {"top_5_gold_doc": True},  # missing anchors
        }],
    }))
    with pytest.raises(ValueError, match="expected_anchor_regexes"):
        rpv.load_fixture(fx)


def test_fixture_loader_accepts_valid_shape(tmp_path):
    fx = tmp_path / "ok.json"
    fx.write_text(json.dumps({
        "class": "Foo",
        "personal_importance": "MED",
        "target_pass_rate": 0.85,
        "queries": [{
            "id": "Q1",
            "query_text": "?",
            "expected": {
                "top_5_gold_doc": True,
                "expected_anchor_regexes": ["x"],
            },
        }],
    }))
    d = rpv.load_fixture(fx)
    assert d["class"] == "Foo"
    assert d["personal_importance"] == "MED"


def test_evaluate_query_all_three_checks_pass(monkeypatch):
    """Gold doc in top-5 + table modality + regex match → PASS."""
    chunks = [_mock_chunk(doc_id_hash="HASH1", modality="table",
                          content="| Part | 4567 |\n| --- | --- |\n| Count | 12 |")]
    monkeypatch.setattr(rpv, "retrieve_hybrid_reranked", lambda q, **k: chunks)
    monkeypatch.setattr(rpv, "_DOC_ID_MAP", {"HASH1": "CarOK_voorraadtelling"})
    q = {
        "id": "Q1",
        "query_text": "What's the count for part 4567?",
        "expected": {
            "top_5_gold_doc": True,
            "format_constraint": "table_value",
            "expected_anchor_regexes": [r"\b4567\b"],
        },
    }
    r = rpv.evaluate_query(q, class_name="CarOK_voorraadtelling")
    assert r.pass_gold_doc is True
    assert r.pass_format is True
    assert r.pass_anchor_regex is True
    assert r.pass_overall is True


def test_evaluate_query_table_value_fails_when_modality_is_text(monkeypatch):
    """The v2.14 P1 CarOK failure mode: top-1 is the flat-prose duplicate."""
    chunks = [_mock_chunk(doc_id_hash="HASH1", modality="text",
                          content="Part number 4567 has 12 in stock")]
    monkeypatch.setattr(rpv, "retrieve_hybrid_reranked", lambda q, **k: chunks)
    monkeypatch.setattr(rpv, "_DOC_ID_MAP", {"HASH1": "CarOK_voorraadtelling"})
    q = {
        "id": "Q1",
        "query_text": "Part 4567?",
        "expected": {
            "top_5_gold_doc": True,
            "format_constraint": "table_value",
            "expected_anchor_regexes": [r"\b4567\b"],
        },
    }
    r = rpv.evaluate_query(q, class_name="CarOK_voorraadtelling")
    assert r.pass_gold_doc is True
    assert r.pass_format is False
    assert r.pass_anchor_regex is True
    assert r.pass_overall is False


def test_evaluate_query_runnable_code_strips_repl_prompts(monkeypatch):
    """Fluent_Python REPL-prefixed code should ast.parse cleanly after stripping."""
    content = ">>> def foo(x):\n>>>     return x + 1\n>>> foo(2)"
    chunks = [_mock_chunk(doc_id_hash="H", modality="text", content=content)]
    monkeypatch.setattr(rpv, "retrieve_hybrid_reranked", lambda q, **k: chunks)
    monkeypatch.setattr(rpv, "_DOC_ID_MAP", {"H": "Fluent_Python"})
    q = {
        "id": "Q1", "query_text": "?",
        "expected": {
            "top_5_gold_doc": True,
            "format_constraint": "runnable_code",
            "expected_anchor_regexes": [r"def\s+foo"],
        },
    }
    r = rpv.evaluate_query(q, class_name="Fluent_Python")
    assert r.pass_format is True
    assert r.pass_anchor_regex is True
    assert r.pass_overall is True


def test_evaluate_query_runnable_code_rejects_prose(monkeypatch):
    """Top-1 with prose preamble (the v2.14 P6 Fluent_Python failure mode)
    must fail ast.parse."""
    content = ("A very practical decorator is functools.lru_cache. "
               "It implements memoization. Example: def foo(): pass")
    chunks = [_mock_chunk(doc_id_hash="H", modality="text", content=content)]
    monkeypatch.setattr(rpv, "retrieve_hybrid_reranked", lambda q, **k: chunks)
    monkeypatch.setattr(rpv, "_DOC_ID_MAP", {"H": "Fluent_Python"})
    q = {
        "id": "Q1", "query_text": "?",
        "expected": {
            "top_5_gold_doc": True,
            "format_constraint": "runnable_code",
            "expected_anchor_regexes": [r"functools\.lru_cache"],
        },
    }
    r = rpv.evaluate_query(q, class_name="Fluent_Python")
    assert r.pass_format is False
    assert r.pass_overall is False


def test_doc_id_resolution_prefers_canonical_basename(tmp_path, monkeypatch):
    """When multiple output dirs share doc_id (dev snapshots), the canonical
    name from CANONICAL_DOCS must win."""
    (tmp_path / "Fluent_Python").mkdir()
    (tmp_path / "Fluent_Python.phase2_baseline").mkdir()
    (tmp_path / "fp_p6_smoke").mkdir()
    header = {"object_type": "ingestion_metadata", "doc_id": "1e7e436164a3"}
    for d in ("Fluent_Python", "Fluent_Python.phase2_baseline", "fp_p6_smoke"):
        (tmp_path / d / "ingestion.jsonl").write_text(json.dumps(header) + "\n")
    monkeypatch.setattr(rpv, "_load_canonical_basenames",
                        lambda: {"Fluent_Python"})
    m = rpv._build_doc_id_to_basename_map(tmp_path)
    assert m["1e7e436164a3"] == "Fluent_Python"


def test_run_class_aggregates_pass_rate(monkeypatch, tmp_path):
    chunks_pass = [_mock_chunk(doc_id_hash="H", modality="table", content="|4567|12|")]
    chunks_fail = [_mock_chunk(doc_id_hash="H", modality="text", content="nothing matches")]
    calls = {"n": 0}

    def fake_retrieve(query, **kwargs):
        calls["n"] += 1
        return chunks_pass if calls["n"] <= 2 else chunks_fail

    monkeypatch.setattr(rpv, "retrieve_hybrid_reranked", fake_retrieve)
    monkeypatch.setattr(rpv, "_DOC_ID_MAP", {"H": "Foo"})

    fx = tmp_path / "Foo.json"
    fx.write_text(json.dumps({
        "class": "Foo",
        "personal_importance": "HIGH",
        "target_pass_rate": 0.85,
        "queries": [
            {"id": f"Q{i}", "query_text": "?",
             "expected": {
                 "top_5_gold_doc": True,
                 "format_constraint": "table_value",
                 "expected_anchor_regexes": [r"\b4567\b"],
             }}
            for i in range(4)
        ],
    }))
    cls = rpv.run_class(fx)
    assert cls.n_total == 4
    assert cls.n_pass == 2
    assert cls.pass_rate == 0.5
    assert not cls.meets_target


def test_dry_run_skips_retrieval(monkeypatch, tmp_path):
    """dry_run path must not call retrieve_hybrid_reranked (verifies fixture
    syntax without live infra)."""
    called = {"n": 0}

    def boom(*a, **k):
        called["n"] += 1
        raise AssertionError("retrieve should not be called in dry_run")

    monkeypatch.setattr(rpv, "retrieve_hybrid_reranked", boom)
    fx = tmp_path / "Foo.json"
    fx.write_text(json.dumps({
        "class": "Foo",
        "personal_importance": "MED",
        "target_pass_rate": 0.85,
        "queries": [{"id": "Q1", "query_text": "?",
                     "expected": {
                         "top_5_gold_doc": True,
                         "expected_anchor_regexes": ["x"],
                     }}],
    }))
    cls = rpv.run_class(fx, dry_run=True)
    assert called["n"] == 0
    assert cls.n_total == 1
    assert cls.queries[0].note == "dry_run"


def test_render_report_includes_summary_table(tmp_path):
    cls = rpv.ClassResult(
        class_name="Foo", personal_importance="HIGH", target_pass_rate=0.85,
        n_pass=1, n_total=2, pass_rate=0.5,
        queries=[rpv.QueryResult(
            query_id="Q1", query_text="?",
            pass_overall=False, pass_gold_doc=True, pass_format=False,
            pass_anchor_regex=True, top_1_doc_id="Foo", top_1_modality="text",
            top_1_chunk_id="ck1", top_5_doc_ids=["Foo"],
            matched_regex="x", gold_chunk_ids_authored=[],
            top_1_chunk_id_matches_gold=None,
        )],
    )
    md = rpv.render_report([cls], generated_at="2026-05-25", label="test")
    assert "| Foo | HIGH | 85% | 50.0% (1/2) | FAIL |" in md
    assert "## Foo" in md


def test_canonical_docs_constant_loads():
    """The PHASE-0-renamed CANONICAL_DOCS list (or legacy CANONICAL_34)
    must be loadable from the rebuild script — required for the doc-id
    resolution preference logic."""
    names = rpv._load_canonical_basenames()
    assert len(names) >= 34, (
        f"expected ≥34 canonical doc names; got {len(names)}. "
        "Did Phase 0 step 6.2 rename + append complete cleanly?"
    )
    assert "Fluent_Python" in names
    assert "CarOK_voorraadtelling" in names
