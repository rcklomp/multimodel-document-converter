"""v2.12 Phase 2 — BM25 sparse + RRF unit tests.

Pin behavior of the in-house BM25 implementation against fixed-input
expectations, so corpus rebuilds or vocab churn don't silently
re-shape the retrieval signal.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmrag_v2.retrieval.sparse import (
    BM25_K1,
    BM25_B,
    BM25Index,
    rrf_fuse,
    tokenize,
)


# ── Tokenizer ────────────────────────────────────────────────────────────────


def test_tokenize_basic_ascii():
    assert tokenize("Hello, World!") == ["hello", "world"]


def test_tokenize_drops_single_char_and_digits():
    assert tokenize("a b cd 12 345 efg") == ["cd", "efg"]


def test_tokenize_handles_multilingual():
    """Dutch + German + accented characters survive lowercasing."""
    result = tokenize("hoe schrijf ik betere ChatGPT-prompts in het Nederlands")
    assert "schrijf" in result
    assert "chatgpt" in result
    assert "nederlands" in result
    result = tokenize("Größere Häuser haben mehr Räume")
    # Note: casefold() converts German ß -> ss (intended unicode
    # behavior — matches how a Dutch/English speaker would type
    # "grossere" or "grossere") which also helps cross-spelling
    # retrieval. Tokens that *did* contain ß are still recoverable
    # because both "größere" and "groessere" casefold to "grössere".
    assert "grössere" in result
    assert "häuser" in result
    assert "räume" in result


def test_tokenize_empty_or_whitespace():
    assert tokenize("") == []
    assert tokenize("   \t\n   ") == []


# ── BM25 index ───────────────────────────────────────────────────────────────


SAMPLE_DOCS = [
    "The Model Context Protocol connects LLMs to data sources and tools.",
    "Python decorators wrap functions with additional behavior.",
    "MCP is a protocol that standardizes how applications provide context to LLMs.",
    "Hoe schrijf ik betere prompts voor ChatGPT.",
]


@pytest.fixture
def index():
    return BM25Index.build_from_corpus(SAMPLE_DOCS)


def test_index_basic_stats(index):
    assert index.n_docs == 4
    assert len(index.vocab) > 0
    assert index.avgdl > 0
    # k1/b are the standard BM25 defaults
    assert index.bm25_k1 == BM25_K1
    assert index.bm25_b == BM25_B


def test_index_vocab_is_deterministic():
    """Building twice over the same corpus yields byte-identical vocab."""
    a = BM25Index.build_from_corpus(SAMPLE_DOCS)
    b = BM25Index.build_from_corpus(SAMPLE_DOCS)
    assert a.vocab == b.vocab
    assert a.idf == b.idf


def test_index_idf_higher_for_rare_terms(index):
    """Tokens appearing in fewer docs have higher IDF."""
    # "protocol" appears in docs 0 and 2 (df=2). "wrap" appears in doc 1 only.
    protocol_id = index.vocab["protocol"]
    wrap_id = index.vocab["wrap"]
    assert index.idf[wrap_id] > index.idf[protocol_id]


def test_encode_document_returns_nonempty_for_in_vocab(index):
    indices, values = index.encode_document(SAMPLE_DOCS[0])
    assert len(indices) == len(values)
    assert len(indices) > 0
    # All values should be positive (IDF * tf' is non-negative; in
    # practice >0 since IDF>0 for non-saturated terms).
    assert all(v > 0 for v in values)


def test_encode_document_oov_tokens_dropped(index):
    """Tokens not in the corpus vocab are silently dropped."""
    indices, values = index.encode_document("xyzzy nonsense unobtainium tokens")
    assert indices == []
    assert values == []


def test_encode_query_binary_presence(index):
    """Query encoding uses 1.0 per unique in-vocab token (binary presence)."""
    indices, values = index.encode_query("MCP and protocol and LLMs")
    # Dedup: "and" appears twice in the query but should encode once.
    assert len(indices) == len(set(indices))
    # All values are 1.0 (binary presence).
    assert all(v == 1.0 for v in values)
    # Common known tokens should be present.
    assert index.vocab["protocol"] in indices
    assert index.vocab["llms"] in indices


def test_bm25_dot_product_matches_classical_formula(index):
    """Dot product of query.encode_query() and doc.encode_document()
    must equal the classical BM25 score (manually computed).
    """
    query_text = "model context protocol"
    qi, qv = index.encode_query(query_text)
    # Compute BM25 directly for doc 0 and verify dot product equals it.
    doc_text = SAMPLE_DOCS[0]
    tokens = tokenize(doc_text)
    doclen = len(tokens)
    classical = 0.0
    for qt in set(tokenize(query_text)):
        tid = index.vocab.get(qt)
        if tid is None:
            continue
        tf = tokens.count(qt)
        if tf == 0:
            continue
        idf = index.idf[tid]
        norm = BM25_K1 * (1.0 - BM25_B + BM25_B * doclen / index.avgdl)
        classical += idf * (tf * (BM25_K1 + 1.0)) / (tf + norm)
    # Now dot product via the encoded sparse vectors.
    di, dv = index.encode_document(doc_text)
    dot = 0.0
    for tid, weight in zip(qi, qv):
        if tid in di:
            j = di.index(tid)
            dot += weight * dv[j]
    assert math.isclose(dot, classical, rel_tol=1e-9, abs_tol=1e-9), (
        f"BM25 dot-product mismatch: dot={dot} vs classical={classical}"
    )


def test_bm25_ranks_relevant_docs_higher(index):
    """Sanity: the most relevant doc to a query scores higher."""
    qi, qv = index.encode_query("model context protocol")
    scores = []
    for did, doc in enumerate(SAMPLE_DOCS):
        di, dv = index.encode_document(doc)
        score = sum(
            qv[i] * dv[di.index(qi[i])]
            for i in range(len(qi))
            if qi[i] in di
        )
        scores.append((did, score))
    # Doc 0 ("The Model Context Protocol...") and doc 2 ("MCP is a
    # protocol...") should score highest; doc 1 (Python decorators)
    # should score 0; doc 3 (Dutch ChatGPT) should score 0.
    by_score = sorted(scores, key=lambda kv: kv[1], reverse=True)
    assert by_score[0][0] in (0, 2)
    assert by_score[1][0] in (0, 2)
    # Irrelevant docs at the bottom.
    assert by_score[2][1] == 0
    assert by_score[3][1] == 0


def test_index_persistence_roundtrip(tmp_path, index):
    """save() + load() reproduces the same index byte-by-byte."""
    p = tmp_path / "idx.json"
    index.save(p)
    loaded = BM25Index.load(p)
    assert loaded.vocab == index.vocab
    assert loaded.idf == index.idf
    assert loaded.avgdl == index.avgdl
    assert loaded.n_docs == index.n_docs
    assert loaded.bm25_k1 == index.bm25_k1


# ── RRF fusion ───────────────────────────────────────────────────────────────


def test_rrf_empty_lists():
    assert rrf_fuse() == []


def test_rrf_single_list():
    fused = rrf_fuse(["a", "b", "c"])
    # Order preserved from the single input list.
    assert [cid for cid, _ in fused] == ["a", "b", "c"]
    # Scores are 1/(60+1), 1/(60+2), 1/(60+3).
    assert math.isclose(fused[0][1], 1 / 61, rel_tol=1e-9)
    assert math.isclose(fused[1][1], 1 / 62, rel_tol=1e-9)


def test_rrf_two_lists_with_overlap():
    """A doc that appears at rank 1 in both lists must beat one that
    appears at rank 1 in only one list."""
    fused = rrf_fuse(
        ["a", "b", "c"],
        ["a", "d", "e"],
    )
    fused_map = dict(fused)
    # "a" gets 1/61 from each list = 2/61 ≈ 0.0328
    # "b" gets 1/62 ≈ 0.0161 from list 1 only
    # "a" wins.
    assert fused[0][0] == "a"
    assert math.isclose(fused_map["a"], 2 / 61, rel_tol=1e-9)
    assert math.isclose(fused_map["b"], 1 / 62, rel_tol=1e-9)


def test_rrf_respects_weights():
    """A weighted list contributes proportionally more to the fused score."""
    # Equal weights: a wins (appears in both)
    fused_eq = rrf_fuse(
        ["a", "b", "c"],
        ["x", "y", "z"],
        weights=[1.0, 1.0],
    )
    # Heavily weight list 2: x wins
    fused_b = rrf_fuse(
        ["a", "b", "c"],
        ["x", "y", "z"],
        weights=[1.0, 100.0],
    )
    assert fused_eq[0][0] == "a"
    assert fused_b[0][0] == "x"


def test_rrf_weights_length_mismatch_raises():
    with pytest.raises(ValueError, match="weights length"):
        rrf_fuse(["a"], ["b"], weights=[1.0, 2.0, 3.0])
