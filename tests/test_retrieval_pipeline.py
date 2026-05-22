"""v2.12 Phase 1 — retrieval pipeline composition tests.

Mock-driven tests for `mmrag_v2.retrieval.pipeline.retrieve_reranked`.
No live Dashscope, Qdrant, or omlx calls — the tests inject mock
embedders, mock Qdrant search, and mock rerankers to pin the
composition shape:

  query → embed → qdrant top-K → reranker(K → N) → out

If any of these stages drift in behavior (e.g. someone forgets to
pass the rerank score back into the result), the test fails.

Live-stack integration is covered by
`tests/test_retrieval_regression_v2_11.py` (and a new
`test_retrieval_regression_v2_12_reranked.py` added at Phase 1
close-out).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmrag_v2.retrieval import (
    DashscopeReranker,
    LocalOmlxReranker,
    Reranker,
    RerankerError,
    get_reranker,
    retrieve_reranked,
)
from mmrag_v2.retrieval.reranker import _NullReranker


# ── Reranker provider unit tests ─────────────────────────────────────────────


class _FixedScoreReranker:
    """Test double — returns chunks in descending order of a per-chunk
    fixed score derived from chunk_id (so the test can verify the
    pipeline preserves rerank order).

    Mimics the Reranker protocol. Score for chunk_id "c00" is 1.00,
    "c01" is 0.99, ..., "c99" is 0.01 (descending by ID number).
    """
    name = "fixed-test"
    model = "fixed-test-model"

    def rerank(self, query, chunks, *, top_n=None):
        scored = []
        for idx, c in enumerate(chunks):
            cid = c["chunk_id"]
            n = int(cid.lstrip("c")) if cid.startswith("c") and cid[1:].isdigit() else idx
            score = max(0.0, 1.0 - n / 100.0)
            out = dict(c)
            out["rerank_score"] = score
            out["rerank_index"] = idx
            scored.append(out)
        # Sort by rerank_score descending.
        scored.sort(key=lambda c: c["rerank_score"], reverse=True)
        if top_n is not None:
            scored = scored[:top_n]
        return scored


def test_null_reranker_preserves_order():
    """`_NullReranker` returns chunks unchanged in vector-rank order
    with sentinel scores. This is the fallback when no reranker is
    available."""
    rr = _NullReranker()
    chunks = [{"chunk_id": f"c{i:02d}", "content": f"doc {i}"} for i in range(5)]
    result = rr.rerank("any query", chunks, top_n=3)
    assert len(result) == 3
    assert [c["chunk_id"] for c in result] == ["c00", "c01", "c02"]
    assert all(c["rerank_score"] == 0.0 for c in result)
    assert [c["rerank_index"] for c in result] == [0, 1, 2]


def test_dashscope_reranker_requires_api_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(RerankerError, match="DASHSCOPE_API_KEY"):
        DashscopeReranker()


def test_local_omlx_reranker_requires_api_key(monkeypatch):
    monkeypatch.delenv("MLX_API_KEY", raising=False)
    with pytest.raises(RerankerError, match="MLX_API_KEY"):
        LocalOmlxReranker()


def test_dashscope_reranker_uses_env_var(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-fake")
    r = DashscopeReranker()
    assert r.name == "dashscope"
    assert r.model == "gte-rerank"


def test_local_omlx_reranker_uses_env_var(monkeypatch):
    monkeypatch.setenv("MLX_API_KEY", "omlx-fake")
    r = LocalOmlxReranker()
    assert r.name == "omlx"
    assert r.model == "gte-reranker-modernbert-base-mlx"


# ── Factory tests ────────────────────────────────────────────────────────────


def test_factory_explicit_arg(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    monkeypatch.setenv("MLX_API_KEY", "omlx-test")
    monkeypatch.delenv("RERANKER_BACKEND", raising=False)
    assert get_reranker("null").name == "null"
    assert get_reranker("dashscope").name == "dashscope"
    assert get_reranker("omlx").name == "omlx"


def test_factory_env_var_fallback(monkeypatch):
    monkeypatch.setenv("RERANKER_BACKEND", "omlx")
    monkeypatch.setenv("MLX_API_KEY", "omlx-test")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    r = get_reranker()
    assert r.name == "omlx"


def test_factory_unknown_backend_raises(monkeypatch):
    monkeypatch.delenv("RERANKER_BACKEND", raising=False)
    with pytest.raises(ValueError, match="Unknown reranker backend"):
        get_reranker("bogus")


def test_factory_uses_compile_default_after_phase_1():
    """After the v2.12 Phase 1 shootout (2026-05-21), the compile-time
    default in `mmrag_v2.retrieval.config._COMPILE_DEFAULT` is `omlx`
    (local ModernBERT, the Phase 1 winner). With no arg + no env var,
    the factory returns the omlx reranker — so callers don't have to
    re-pick after every release."""
    import os
    from mmrag_v2.retrieval import config as cfg
    # Save / restore around the test to avoid leaking env state.
    saved_env = os.environ.pop("RERANKER_BACKEND", None)
    saved_mlx = os.environ.get("MLX_API_KEY")
    try:
        os.environ["MLX_API_KEY"] = "omlx-test-fake"
        assert cfg._COMPILE_DEFAULT == "omlx", (
            "v2.12 Phase 1 set the compile-time reranker default to 'omlx' "
            "(local ModernBERT, the shootout winner). If this assertion "
            "fails, the default reverted unexpectedly — re-verify against "
            "the Phase 1 soak data before changing."
        )
        r = get_reranker()
        assert r.name == "omlx"
    finally:
        if saved_env is not None:
            os.environ["RERANKER_BACKEND"] = saved_env
        if saved_mlx is None:
            os.environ.pop("MLX_API_KEY", None)


def test_factory_no_backend_resolvable_after_clearing_default(monkeypatch):
    """If the compile-time default is cleared AND no env var / arg is
    given, the factory raises. Pins the resolution-order contract:
    explicit > env > compile-default."""
    monkeypatch.delenv("RERANKER_BACKEND", raising=False)
    from mmrag_v2.retrieval import config as cfg
    monkeypatch.setattr(cfg, "_COMPILE_DEFAULT", None)
    with pytest.raises(ValueError, match="No reranker backend resolved"):
        get_reranker()


# ── End-to-end pipeline tests with mocks ─────────────────────────────────────


def _qdrant_result(idx: int, chunk_id: str, content: str, score: float = 0.5):
    """Build a Qdrant search result shaped like search_qdrant.search() returns."""
    return {
        "id": f"point-{idx}",
        "score": score,
        "payload": {
            "chunk_id": chunk_id,
            "content": content,
            "doc_id": "doc-1",
            "modality": "text",
            "page_number": idx,
        },
    }


def test_retrieve_reranked_composes_embed_qdrant_rerank(monkeypatch):
    """Happy path: embed returns a vector, qdrant returns 5 candidates,
    reranker reorders them, top-3 is returned with rerank scores
    attached."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")

    # Mock embedding.
    fake_vector = [0.1, 0.2, 0.3]
    with patch("mmrag_v2.retrieval.pipeline.embed_text_dashscope",
               return_value=fake_vector) as embed_mock, \
         patch("mmrag_v2.retrieval.pipeline.qdrant_search") as qd_mock:

        # Mock Qdrant returns 5 candidates with chunk_ids c04, c03, c02, c01, c00.
        # The mock reranker will reorder by chunk_id number descending so c04 stays
        # on top BUT the order proves the reranker — not Qdrant — controls output.
        qd_mock.return_value = [
            _qdrant_result(0, "c04", "doc 4", score=0.95),
            _qdrant_result(1, "c03", "doc 3", score=0.93),
            _qdrant_result(2, "c02", "doc 2", score=0.91),
            _qdrant_result(3, "c01", "doc 1", score=0.89),
            _qdrant_result(4, "c00", "doc 0", score=0.87),
        ]

        # Reverse the order via the fixed-score reranker (which scores
        # c00 highest, c04 lowest). Pinned to dashscope provider +
        # text-embedding-v4 model to match the mocked embedder (the
        # v2.13.0 library defaults are omlx + Qwen3-Embedding-8B-mxfp8;
        # tests that mock dashscope must opt in explicitly).
        result = retrieve_reranked(
            query="any query",
            collection="test-coll",
            top_k_retrieve=5,
            top_n_return=3,
            embed_provider="dashscope",
            embed_model="text-embedding-v4",
            reranker=_FixedScoreReranker(),
        )

    # Embed called exactly once with the query.
    embed_mock.assert_called_once()
    args = embed_mock.call_args
    assert args[0][0] == "any query"
    assert args[0][1] == "text-embedding-v4"

    # Qdrant called exactly once with limit=5.
    qd_mock.assert_called_once()
    qd_args = qd_mock.call_args
    assert qd_args[0][0] == fake_vector
    assert qd_args[0][1] == "test-coll"
    assert qd_args[1]["limit"] == 5

    # Reranker has reordered: c00 first (score 1.0), then c01, c02. c03/c04 dropped.
    assert len(result) == 3
    assert [r["payload"]["chunk_id"] for r in result] == ["c00", "c01", "c02"]
    # Rerank scores attached, original Qdrant fields preserved.
    assert result[0]["rerank_score"] == 1.0
    assert result[0]["score"] == 0.87  # original Qdrant score for c00
    assert result[0]["id"] == "point-4"
    # rerank_index = position in the candidate list (4 for c00 in our mock).
    assert result[0]["rerank_index"] == 4


def test_retrieve_reranked_falls_back_on_rerank_error(monkeypatch):
    """If the reranker raises RerankerError and fall_back_on_rerank_error
    is True (default), the pipeline returns vector-rank order with
    sentinel rerank fields."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")

    class _BrokenReranker:
        name = "broken"
        model = "broken"
        def rerank(self, q, chunks, *, top_n=None):
            raise RerankerError("simulated outage")

    with patch("mmrag_v2.retrieval.pipeline.embed_text_dashscope",
               return_value=[0.0]*3), \
         patch("mmrag_v2.retrieval.pipeline.qdrant_search") as qd_mock:
        qd_mock.return_value = [
            _qdrant_result(0, "c00", "doc 0", score=0.95),
            _qdrant_result(1, "c01", "doc 1", score=0.93),
            _qdrant_result(2, "c02", "doc 2", score=0.91),
        ]
        result = retrieve_reranked(
            query="q",
            collection="t",
            top_k_retrieve=3,
            top_n_return=2,
            embed_provider="dashscope",
            embed_model="text-embedding-v4",
            reranker=_BrokenReranker(),
            fall_back_on_rerank_error=True,
        )
    # Fallback: vector-rank order preserved, sentinel scores.
    assert len(result) == 2
    assert [r["payload"]["chunk_id"] for r in result] == ["c00", "c01"]
    assert all(r["rerank_score"] == 0.0 for r in result)
    assert [r["rerank_index"] for r in result] == [0, 1]


def test_retrieve_reranked_propagates_error_when_fallback_disabled(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")

    class _BrokenReranker:
        name = "broken"
        model = "broken"
        def rerank(self, q, chunks, *, top_n=None):
            raise RerankerError("simulated outage")

    with patch("mmrag_v2.retrieval.pipeline.embed_text_dashscope",
               return_value=[0.0]*3), \
         patch("mmrag_v2.retrieval.pipeline.qdrant_search") as qd_mock:
        qd_mock.return_value = [_qdrant_result(0, "c00", "x")]
        with pytest.raises(RerankerError):
            retrieve_reranked(
                query="q", collection="t",
                top_k_retrieve=1, top_n_return=1,
                embed_provider="dashscope",
                embed_model="text-embedding-v4",
                reranker=_BrokenReranker(),
                fall_back_on_rerank_error=False,
            )


def test_retrieve_reranked_empty_qdrant_returns_empty(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    with patch("mmrag_v2.retrieval.pipeline.embed_text_dashscope",
               return_value=[0.0]*3), \
         patch("mmrag_v2.retrieval.pipeline.qdrant_search",
               return_value=[]):
        result = retrieve_reranked(
            query="q", collection="t",
            top_k_retrieve=25, top_n_return=5,
            embed_provider="dashscope",
            embed_model="text-embedding-v4",
            reranker=_NullReranker(),
        )
    assert result == []


def test_retrieve_reranked_truncates_to_top_n(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    with patch("mmrag_v2.retrieval.pipeline.embed_text_dashscope",
               return_value=[0.0]*3), \
         patch("mmrag_v2.retrieval.pipeline.qdrant_search") as qd_mock:
        qd_mock.return_value = [
            _qdrant_result(i, f"c{i:02d}", f"d{i}") for i in range(20)
        ]
        result = retrieve_reranked(
            query="q", collection="t",
            top_k_retrieve=20, top_n_return=5,
            embed_provider="dashscope",
            embed_model="text-embedding-v4",
            reranker=_NullReranker(),
        )
    assert len(result) == 5
    # Null reranker preserves vector-rank order.
    assert [r["payload"]["chunk_id"] for r in result] == [
        "c00", "c01", "c02", "c03", "c04",
    ]


def test_retrieve_reranked_uses_factory_when_no_reranker_passed(monkeypatch):
    """When `reranker` is None, the pipeline calls `get_reranker(backend)`
    via the factory. This test pins that path."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    monkeypatch.setenv("RERANKER_BACKEND", "null")  # forces factory to return _NullReranker
    with patch("mmrag_v2.retrieval.pipeline.embed_text_dashscope",
               return_value=[0.0]*3), \
         patch("mmrag_v2.retrieval.pipeline.qdrant_search") as qd_mock:
        qd_mock.return_value = [
            _qdrant_result(i, f"c{i:02d}", f"d{i}") for i in range(3)
        ]
        result = retrieve_reranked(
            query="q", collection="t",
            top_k_retrieve=3, top_n_return=2,
            embed_provider="dashscope",
            embed_model="text-embedding-v4",
            reranker=None,
            reranker_backend=None,  # factory should read RERANKER_BACKEND env
        )
    assert len(result) == 2
    assert all(r["rerank_score"] == 0.0 for r in result)


# ── Score-attach helper tests ────────────────────────────────────────────────


def test_attach_scores_skips_out_of_range_indices():
    """Defensive: if a reranker returns an index out of bounds (broken
    server, garbage response), `_attach_scores` skips it silently
    rather than raising."""
    from mmrag_v2.retrieval.reranker import _attach_scores
    chunks = [{"chunk_id": "a"}, {"chunk_id": "b"}]
    results = [
        {"index": 0, "relevance_score": 0.9},
        {"index": 99, "relevance_score": 0.5},  # out of bounds
        {"index": 1, "relevance_score": 0.3},
        {"index": -1, "relevance_score": 0.1},  # negative
    ]
    out = _attach_scores(chunks, results)
    assert len(out) == 2
    assert [c["chunk_id"] for c in out] == ["a", "b"]
    assert [c["rerank_score"] for c in out] == [0.9, 0.3]
