"""v2.12 retrieval module — embedding + Qdrant + reranker pipeline.

Exposes a single composable retrieve function that producers/consumers
of the retrieval flow call into. Backed by:

  - `reranker.py`    — provider abstraction (LocalOmlxReranker,
                       DashscopeReranker) implementing a common
                       `rerank(query, chunks) -> list[dict]` interface.
  - `pipeline.py`    — composable retrieve → rerank → return.
  - `config.py`      — factory `get_reranker(name)` reading
                       `RERANKER_BACKEND` env var or explicit arg.

Public API:

    from mmrag_v2.retrieval import retrieve_reranked, get_reranker

    chunks = retrieve_reranked(
        query="what is MCP",
        collection="mmrag_v2_8__qwen3_local",  # v2.13.0 production default
        top_k_retrieve=25,
        top_n_return=5,
    )

See `docs/PLAN_V2.12.md` Phase 1 for the design rationale and
`docs/PLAN_V2.13.md` Phase 1 for the v2.13.0 embedder swap.
"""
from mmrag_v2.retrieval.pipeline import (  # noqa: F401
    retrieve_reranked,
    retrieve_hybrid_reranked,
)
from mmrag_v2.retrieval.reranker import (  # noqa: F401
    DashscopeReranker,
    LocalOmlxReranker,
    Reranker,
    RerankerError,
)
from mmrag_v2.retrieval.config import get_reranker  # noqa: F401
from mmrag_v2.retrieval.sparse import (  # noqa: F401
    BM25Index,
    rrf_fuse,
    tokenize,
)
from mmrag_v2.retrieval.hyde import (  # noqa: F401
    HydeError,
    generate_hypothetical_answer,
    generate_with_fallback,
)

__all__ = [
    "retrieve_reranked",
    "retrieve_hybrid_reranked",
    "get_reranker",
    "Reranker",
    "DashscopeReranker",
    "LocalOmlxReranker",
    "RerankerError",
    "BM25Index",
    "rrf_fuse",
    "tokenize",
    "HydeError",
    "generate_hypothetical_answer",
    "generate_with_fallback",
]
