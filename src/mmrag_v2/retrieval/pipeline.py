"""v2.12 retrieval pipeline — composable embed → Qdrant → rerank.

Single public entry point: `retrieve_reranked(...)`. Composes the
embed step (Dashscope `text-embedding-v4` for the v2.11.0 production
collection), the Qdrant vector search, and the reranker stage.

The embed and Qdrant primitives are imported from
`scripts.search_qdrant` / `scripts.ingest_to_qdrant` to avoid
duplication; the retrieval module owns ONLY the composition + rerank
provider abstraction.

Production usage:

    from mmrag_v2.retrieval import retrieve_reranked

    chunks = retrieve_reranked(
        query="how do LLM agents call tools",
        collection="mmrag_v2_8__qwen3_dashscope",
        top_k_retrieve=25,         # candidates the reranker sees
        top_n_return=5,            # returned to the caller
        reranker_backend="omlx",   # or "dashscope" or None for env var
    )
    for c in chunks:
        print(c["chunk_id"], c["rerank_score"], c["payload"]["content"][:80])
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

    # Step 1: embed the query.
    vector = _embed_query(
        query, embed_provider, embed_model,
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
