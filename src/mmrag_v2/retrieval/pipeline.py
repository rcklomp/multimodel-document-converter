"""v2.12 retrieval pipeline — composable embed → Qdrant → rerank.

Two public entry points:

  retrieve_reranked()         dense-only:   embed → Qdrant → rerank
  retrieve_hybrid_reranked()  hybrid:       embed + sparse → RRF → rerank

Both compose the same building blocks. The hybrid path adds a BM25
sparse search against a side-collection
(`mmrag_v2_8__bm25_sparse`) and fuses the two ranked lists via RRF
before handing the top-K to the reranker.

The embed and Qdrant primitives are imported from
`scripts.search_qdrant` / `scripts.ingest_to_qdrant` to avoid
duplication; the retrieval module owns ONLY the composition + rerank
provider abstraction.

Production usage (dense-only, v2.12 Phase 1 default):

    from mmrag_v2.retrieval import retrieve_reranked

    chunks = retrieve_reranked(
        query="how do LLM agents call tools",
        collection="mmrag_v2_8__qwen3_dashscope",
        top_k_retrieve=25,
        top_n_return=5,
        reranker_backend="omlx",
    )

Hybrid usage (v2.12 Phase 2):

    from mmrag_v2.retrieval import retrieve_hybrid_reranked

    chunks = retrieve_hybrid_reranked(
        query="how do LLM agents call tools",
        dense_collection="mmrag_v2_8__qwen3_dashscope",
        sparse_collection="mmrag_v2_8__bm25_sparse",
        bm25_index_path="tests/fixtures/bm25_index_v2_12.json",
        top_k_retrieve=25,   # per leg, before RRF
        top_n_fuse=25,       # candidates the reranker sees (post-RRF)
        top_n_return=5,
    )
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from mmrag_v2.retrieval.config import get_reranker
from mmrag_v2.retrieval.reranker import Reranker, RerankerError

# Resolve scripts/ on sys.path so we can reuse the embed + search
# primitives without duplicating them.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from ingest_to_qdrant import embed_text_dashscope  # noqa: E402
from search_qdrant import embed as embed_ollama  # noqa: E402
from search_qdrant import search as qdrant_search  # noqa: E402


def _embed_query(
    text: str,
    provider: str,
    model: str,
    *,
    api_key: str = "",
    ollama_url: str = "http://localhost:11434",
) -> list[float]:
    """Embed a query through the chosen provider. Mirrors the dispatch
    logic in `scripts.retrieval_regression`."""
    if provider == "dashscope":
        return embed_text_dashscope(text, model, api_key)
    if provider == "ollama":
        return embed_ollama(text, model=model, ollama_url=ollama_url)
    raise ValueError(f"Unsupported embed provider: {provider!r}")


def retrieve_reranked(
    query: str,
    *,
    collection: str = "mmrag_v2_8__qwen3_dashscope",
    top_k_retrieve: int = 25,
    top_n_return: int = 5,
    embed_provider: str = "dashscope",
    embed_model: str = "text-embedding-v4",
    embed_api_key: str | None = None,
    qdrant_url: str = "http://localhost:6333",
    reranker: Reranker | None = None,
    reranker_backend: str | None = None,
    fall_back_on_rerank_error: bool = True,
    use_hyde: bool = False,
    hyde_api_key: str | None = None,
) -> list[dict]:
    """Embed → Qdrant top-K → rerank → top-N.

    Returns a list of dicts, each shaped as the upstream Qdrant search
    result (`{"id", "score", "payload"}`) plus the two rerank fields
    `rerank_score` and `rerank_index` added by the reranker.

    Arguments:

      query                Natural-language query string.
      collection           Qdrant collection name. Defaults to v2.11.0
                           production collection.
      top_k_retrieve       Number of candidates the reranker sees.
                           v2.12 default = 25 per the empirical
                           latency benchmark; may rise to 50 if the
                           Phase 1 soak doesn't clear Recall@5 ≥ 85%.
      top_n_return         Final list size returned to caller. Default 5.
      embed_provider       "dashscope" (v2.11.0+) or "ollama" (legacy
                           rollback through 2026-06-19).
      embed_model          Embed model name. Defaults match the
                           production v2.11.0 collection.
      embed_api_key        Override for Dashscope key. Default reads
                           DASHSCOPE_API_KEY env var.
      qdrant_url           Qdrant base URL.
      reranker             Pre-constructed Reranker instance. Useful
                           for tests and for keeping a reranker hot
                           across many calls. If None, one is created
                           via `get_reranker(reranker_backend)`.
      reranker_backend     "dashscope" | "omlx" | "null". Only used
                           when `reranker` is None.
      fall_back_on_rerank_error
                           If True (default), catch RerankerError and
                           return the unreranked Qdrant top-N (with
                           `rerank_score=0.0`, `rerank_index=i`). If
                           False, the RerankerError propagates.
    """
    if embed_provider == "dashscope" and embed_api_key is None:
        embed_api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not embed_api_key:
            raise ValueError(
                "Dashscope embed provider requires DASHSCOPE_API_KEY env "
                "var or explicit embed_api_key arg"
            )

    # Optional Step 0: HyDE — generate a hypothetical answer and embed
    # that instead of the literal query. Falls back to the literal
    # query on any HyDE failure (network, parse, refusal).
    embed_text = query
    if use_hyde:
        from mmrag_v2.retrieval.hyde import generate_with_fallback
        embed_text = generate_with_fallback(query, hyde_api_key or embed_api_key)

    # Step 1: embed the query (or the HyDE-generated answer).
    vector = _embed_query(
        embed_text, embed_provider, embed_model,
        api_key=embed_api_key or "",
    )

    # Step 2: Qdrant vector search → top-K candidates.
    candidates = qdrant_search(
        vector, collection,
        limit=top_k_retrieve, qdrant_url=qdrant_url,
    )
    if not candidates:
        return []

    # Step 3: reranker. The reranker takes the inline payload content
    # (we pass payload-only structures so the rerank API has the
    # content field at top level), then we lift its decisions back
    # onto the full Qdrant result dicts (preserving `id`, `score`,
    # `payload`).
    rerank_inputs = [
        {
            "chunk_id": (c.get("payload") or {}).get("chunk_id") or str(c.get("id")),
            "content": (c.get("payload") or {}).get("content") or "",
            "_qdrant": c,  # preserve the full upstream result
        }
        for c in candidates
    ]

    if reranker is None:
        try:
            reranker = get_reranker(reranker_backend)
        except (ValueError, RerankerError) as e:
            if not fall_back_on_rerank_error:
                raise
            # Fall back to vector-rank order.
            return _vector_rank_fallback(candidates, top_n_return)

    try:
        reranked = reranker.rerank(query, rerank_inputs, top_n=top_n_return)
    except RerankerError:
        if not fall_back_on_rerank_error:
            raise
        return _vector_rank_fallback(candidates, top_n_return)

    # Step 4: lift rerank decisions back onto full Qdrant result dicts.
    out = []
    for r in reranked:
        qd = r.get("_qdrant") or {}
        out.append({
            **qd,
            "rerank_score": r.get("rerank_score", 0.0),
            "rerank_index": r.get("rerank_index", -1),
        })
    return out


def _vector_rank_fallback(candidates: list[dict], top_n_return: int) -> list[dict]:
    """Fallback when the reranker can't be used: return upstream Qdrant
    candidates in their original order, sliced to top_n_return, with
    sentinel rerank fields so downstream consumers can detect this
    was a fallback (`rerank_index == position`, `rerank_score == 0`)."""
    out = []
    for i, c in enumerate(candidates[:top_n_return]):
        out.append({
            **c,
            "rerank_score": 0.0,
            "rerank_index": i,
        })
    return out


# ── Phase 2: hybrid retrieval (dense + sparse + RRF + rerank) ───────────────


def _sparse_search(qdrant_url: str, sparse_collection: str,
                   indices: list[int], values: list[float],
                   limit: int, sparse_vector_name: str = "bm25") -> list[dict]:
    """POST /collections/{name}/points/query with a named sparse vector.

    Returns Qdrant points in descending sparse-similarity (BM25) order.
    Each point dict has `id`, `score`, `payload` (the side-collection
    payload has just `{"chunk_id", "doc_dir"}`).
    """
    import json as _json
    import urllib.request as _urllib
    body = _json.dumps({
        "query": {
            "indices": indices,
            "values": values,
        },
        "using": sparse_vector_name,
        "with_payload": True,
        "limit": limit,
    }).encode("utf-8")
    req = _urllib.Request(
        f"{qdrant_url}/collections/{sparse_collection}/points/query",
        data=body, method="POST",
    )
    req.add_header("Content-Type", "application/json")
    with _urllib.urlopen(req, timeout=30) as resp:
        data = _json.loads(resp.read())
    return data.get("result", {}).get("points", []) or []


def _fetch_dense_points_by_chunk_id(
    qdrant_url: str, dense_collection: str, chunk_ids: list[str],
) -> dict[str, dict]:
    """Look up full Qdrant payloads from the dense collection by
    chunk_id (preserves content, doc_id, source_file, etc.).

    Returns map chunk_id → Qdrant point dict.
    """
    import json as _json
    import urllib.request as _urllib
    if not chunk_ids:
        return {}
    # Use scroll with filter on chunk_id ∈ {targets}.
    body = _json.dumps({
        "filter": {
            "must": [
                {"key": "chunk_id", "match": {"any": chunk_ids}}
            ]
        },
        "limit": len(chunk_ids),
        "with_payload": True,
        "with_vector": False,
    }).encode("utf-8")
    req = _urllib.Request(
        f"{qdrant_url}/collections/{dense_collection}/points/scroll",
        data=body, method="POST",
    )
    req.add_header("Content-Type", "application/json")
    with _urllib.urlopen(req, timeout=30) as resp:
        data = _json.loads(resp.read())
    points = data.get("result", {}).get("points", []) or []
    return {
        (p.get("payload") or {}).get("chunk_id") or str(p.get("id")): p
        for p in points
    }


def retrieve_hybrid_reranked(
    query: str,
    *,
    dense_collection: str = "mmrag_v2_8__qwen3_dashscope",
    sparse_collection: str = "mmrag_v2_8__bm25_sparse",
    bm25_index_path: str = "tests/fixtures/bm25_index_v2_12.json",
    top_k_retrieve: int = 25,
    top_n_fuse: int = 25,
    top_n_return: int = 5,
    rrf_k: int = 60,
    rrf_weights: tuple[float, float] = (1.0, 1.0),  # (dense, sparse)
    embed_provider: str = "dashscope",
    embed_model: str = "text-embedding-v4",
    embed_api_key: str | None = None,
    qdrant_url: str = "http://localhost:6333",
    reranker=None,
    reranker_backend: str | None = None,
    fall_back_on_rerank_error: bool = True,
    use_hyde: bool = False,
    hyde_api_key: str | None = None,
) -> list[dict]:
    """Dense + BM25 sparse + RRF + reranker.

    Pipeline:
      1. embed query (text-embedding-v4) → dense vector
      2. dense top-K search on `dense_collection`
      3. BM25 query encoding from `bm25_index_path`
      4. sparse top-K search on `sparse_collection`
      5. RRF fuse the two rankings → top-N candidates
      6. fetch full payloads from dense collection by chunk_id
      7. reranker → top-N return

    `top_k_retrieve` is the per-leg top-K (both dense and sparse).
    `top_n_fuse`     is the post-RRF candidate count passed to the reranker.
    `top_n_return`   is the final list returned to the caller.

    `rrf_weights` lets callers tilt fusion toward dense or sparse;
    defaults equal weight (1, 1). Higher dense weight = embedder-led;
    higher sparse weight = BM25-led.
    """
    import os as _os
    from pathlib import Path as _Path
    from mmrag_v2.retrieval.sparse import BM25Index, rrf_fuse

    if embed_provider == "dashscope" and embed_api_key is None:
        embed_api_key = _os.environ.get("DASHSCOPE_API_KEY", "")
        if not embed_api_key:
            raise ValueError(
                "Dashscope embed provider requires DASHSCOPE_API_KEY env var"
            )

    # Load BM25 index (consider caching across calls in production —
    # for now we load per call; ~50ms for the 2MB JSON).
    index_path_resolved = _Path(bm25_index_path)
    if not index_path_resolved.is_absolute():
        index_path_resolved = _REPO_ROOT / bm25_index_path
    bm25 = BM25Index.load(index_path_resolved)

    # Optional Step 0: HyDE for the DENSE leg only — the BM25 leg
    # always uses the literal query (BM25 is a keyword matcher; HyDE
    # would change the keyword distribution and isn't helpful there).
    dense_query = query
    if use_hyde:
        from mmrag_v2.retrieval.hyde import generate_with_fallback
        dense_query = generate_with_fallback(query, hyde_api_key or embed_api_key)

    # Step 1: embed (the HyDE answer or the literal query).
    vector = _embed_query(
        dense_query, embed_provider, embed_model,
        api_key=embed_api_key or "",
    )

    # Step 2: dense top-K.
    dense_hits = qdrant_search(
        vector, dense_collection,
        limit=top_k_retrieve, qdrant_url=qdrant_url,
    )

    # Step 3+4: sparse query (LITERAL query, never the HyDE answer) → sparse top-K.
    sparse_indices, sparse_values = bm25.encode_query(query)
    sparse_hits = (
        _sparse_search(qdrant_url, sparse_collection,
                       sparse_indices, sparse_values, top_k_retrieve)
        if sparse_indices else []
    )

    # Step 5: RRF fusion.
    dense_chunk_ids = [
        (h.get("payload") or {}).get("chunk_id") or str(h.get("id"))
        for h in dense_hits
    ]
    sparse_chunk_ids = [
        (h.get("payload") or {}).get("chunk_id") or str(h.get("id"))
        for h in sparse_hits
    ]
    fused = rrf_fuse(
        dense_chunk_ids, sparse_chunk_ids,
        k=rrf_k, weights=list(rrf_weights),
    )
    if not fused:
        return []
    fused_top_ids = [cid for cid, _score in fused[:top_n_fuse]]

    # Step 6: fetch full payloads from dense collection (some chunk_ids
    # may only come from sparse and not be in dense_hits).
    dense_by_chunk = {
        (h.get("payload") or {}).get("chunk_id") or str(h.get("id")): h
        for h in dense_hits
    }
    missing_ids = [cid for cid in fused_top_ids if cid not in dense_by_chunk]
    if missing_ids:
        fetched = _fetch_dense_points_by_chunk_id(
            qdrant_url, dense_collection, missing_ids,
        )
        dense_by_chunk.update(fetched)

    rerank_inputs: list[dict] = []
    for cid in fused_top_ids:
        hit = dense_by_chunk.get(cid)
        if not hit:
            continue
        payload = hit.get("payload") or {}
        rerank_inputs.append({
            "chunk_id": cid,
            "content": payload.get("content") or "",
            "_qdrant": hit,
        })
    if not rerank_inputs:
        return []

    # Step 7: reranker.
    if reranker is None:
        try:
            from mmrag_v2.retrieval.config import get_reranker
            reranker = get_reranker(reranker_backend)
        except (ValueError, RerankerError) as e:
            if not fall_back_on_rerank_error:
                raise
            # Fall back to RRF-rank order.
            return [
                {**(rerank_inputs[i].get("_qdrant") or {}),
                 "rerank_score": 0.0, "rerank_index": i}
                for i in range(min(top_n_return, len(rerank_inputs)))
            ]
    try:
        reranked = reranker.rerank(query, rerank_inputs, top_n=top_n_return)
    except RerankerError:
        if not fall_back_on_rerank_error:
            raise
        return [
            {**(rerank_inputs[i].get("_qdrant") or {}),
             "rerank_score": 0.0, "rerank_index": i}
            for i in range(min(top_n_return, len(rerank_inputs)))
        ]

    # Lift rerank decisions back onto full Qdrant result dicts.
    out = []
    for r in reranked:
        qd = r.get("_qdrant") or {}
        out.append({
            **qd,
            "rerank_score": r.get("rerank_score", 0.0),
            "rerank_index": r.get("rerank_index", -1),
        })
    return out


# Resolve REPO_ROOT once for the hybrid pipeline (used in the BM25 index path).
_REPO_ROOT = Path(__file__).resolve().parents[3]
