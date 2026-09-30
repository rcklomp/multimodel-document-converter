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

Production usage (dense-only, v2.13.0 default — omlx local embedder):

    from mmrag_v2.retrieval import retrieve_reranked

    chunks = retrieve_reranked(
        query="how do LLM agents call tools",
        collection="mmrag_v2_8__qwen3_local",
        top_k_retrieve=25,
        top_n_return=5,
        reranker_backend="omlx",
    )

Hybrid usage (v2.12 Phase 2 retrieval shape, v2.13.0 embedder):

    from mmrag_v2.retrieval import retrieve_hybrid_reranked

    chunks = retrieve_hybrid_reranked(
        query="how do LLM agents call tools",
        dense_collection="mmrag_v2_8__qwen3_local",
        sparse_collection="mmrag_v2_8__bm25_sparse",
        bm25_index_path="tests/fixtures/bm25_index_v2_12.json",
        top_k_retrieve=25,   # per leg, before RRF
        top_n_fuse=25,       # candidates the reranker sees (post-RRF)
        top_n_return=5,
    )
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Optional

from mmrag_v2.endpoints import endpoint as _endpoint
from mmrag_v2.retrieval.config import get_reranker
from mmrag_v2.retrieval.reranker import Reranker, RerankerError

# Resolve scripts/ on sys.path so we can reuse the embed + search
# primitives without duplicating them.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from ingest_to_qdrant import embed_text_dashscope, embed_text_omlx  # noqa: E402
from search_qdrant import embed as embed_ollama  # noqa: E402
from search_qdrant import search as qdrant_search  # noqa: E402

# Resolved from the central endpoint registry (mmrag_v2.endpoints);
# default is the Mini-hosted oMLX embeddings server, env-overridable.
_OMLX_DEFAULT_URL = _endpoint("embed").embeddings_url

# Search-time HNSW exploration depth for dense retrieval. The Qdrant
# collection default gets trapped in the large near-identical
# empty-image-chunk cluster and silently fails to reach higher-cosine
# text chunks (the ~6% doc-recall floor: gold chunks at cosine 0.62
# returned ABSENT while 0.36-scored empty placeholders filled the
# top-100). 512 recovers them to near-exact-search quality at
# negligible latency on this corpus size: full-514 production
# gold-chunk@10 82.9% -> 87.7% (+4.9pp, McNemar 25 wins / 0 losses,
# p<1e-5, 2026-06-16). See docs/PLAN_DOC_RECALL_FLOOR_V1.md.
_DEFAULT_HNSW_EF = 512


def _embed_query(
    text: str,
    provider: str,
    model: str,
    *,
    api_key: str = "",
    ollama_url: str = "http://localhost:11434",
    omlx_url: str = _OMLX_DEFAULT_URL,
) -> list[float]:
    """Embed a query through the chosen provider. Mirrors the dispatch
    logic in `scripts.retrieval_regression`."""
    if provider == "dashscope":
        return embed_text_dashscope(text, model, api_key)
    if provider == "ollama":
        return embed_ollama(text, model=model, ollama_url=ollama_url)
    if provider == "omlx":
        return embed_text_omlx(text, model, api_key, url=omlx_url)
    raise ValueError(f"Unsupported embed provider: {provider!r}")


def _l2_normalize(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    if norm == 0.0:
        return v
    return [x / norm for x in v]


def _blend_vectors(
    query_vec: list[float], hyde_vec: list[float], weight: float = 0.5
) -> list[float]:
    """Blend the literal-query embedding with the HyDE-answer embedding.

    L2-normalizes each side, takes the weighted average (default equal
    weight), and re-normalizes. Blending BEATS replacing the query with
    the hypothetical (the original v2.12 HyDE behavior): keeping the
    query's exact-token signal protects identifier/specific queries while
    still adding the answer-space bridge (it loses only 7 queries where
    pure-replace lost 11).

    HOWEVER, HyDE itself is OPT-IN and stays OFF by default. Its apparent
    +3.5pp (judge-free gold-chunk@10, McNemar p=0.0021) was measured
    against the OLD default-ef baseline that suffered the HNSW empty-chunk
    trap. Once that bug is fixed (`_DEFAULT_HNSW_EF`=512), HyDE-blend adds
    only +1.4pp on top (won 12 / lost 5, p=0.14 NOT significant,
    2026-06-16) - it was largely recovering the same buried chunks ef=512
    now gets for free, and it costs a per-query LLM generation. Do NOT
    turn HyDE on by default on the strength of the +3.5pp figure. See
    memory project_doc_recall_floor + project_retrieval_findings.
    """
    qn = _l2_normalize(query_vec)
    hn = _l2_normalize(hyde_vec)
    return _l2_normalize([weight * a + (1.0 - weight) * b for a, b in zip(qn, hn)])


def retrieve_reranked(
    query: str,
    *,
    collection: str = "mmrag_v2_8__qwen3_local",
    top_k_retrieve: int = 25,
    top_n_return: int = 5,
    embed_provider: str = "omlx",
    embed_model: str = "Qwen3-Embedding-8B-mxfp8",
    embed_api_key: str | None = None,
    qdrant_url: str = "http://localhost:6333",
    reranker: Reranker | None = None,
    reranker_backend: str | None = None,
    fall_back_on_rerank_error: bool = True,
    use_hyde: bool = False,
    hyde_api_key: str | None = None,
    hyde_provider: str = "vllm",
    hnsw_ef: int | None = _DEFAULT_HNSW_EF,
) -> list[dict]:
    """Embed → Qdrant top-K → rerank → top-N.

    Returns a list of dicts, each shaped as the upstream Qdrant search
    result (`{"id", "score", "payload"}`) plus the two rerank fields
    `rerank_score` and `rerank_index` added by the reranker.

    Arguments:

      query                Natural-language query string.
      collection           Qdrant collection name. Defaults to v2.13.0
                           production collection (mmrag_v2_8__qwen3_local,
                           4096-dim, populated by local Qwen3-Embedding-8B).
      top_k_retrieve       Number of candidates the reranker sees.
                           v2.12 default = 25 per the empirical
                           latency benchmark; may rise to 50 if the
                           Phase 1 soak doesn't clear Recall@5 ≥ 85%.
      top_n_return         Final list size returned to caller. Default 5.
      embed_provider       "omlx" (v2.13.0 default — local
                           Qwen3-Embedding-8B-mxfp8 via omlx-server),
                           "dashscope" (v2.11.0-v2.12.0 prod, retained
                           as 30-day rollback through 2026-06-19), or
                           "ollama" (legacy v2.10 path).
      embed_model          Embed model name. Defaults to v2.13.0 prod
                           (Qwen3-Embedding-8B-mxfp8); must match how
                           the target collection was built.
      embed_api_key        Auto-resolved per provider: DASHSCOPE_API_KEY
                           for dashscope, MLX_API_KEY for omlx.
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
    if embed_provider == "omlx" and embed_api_key is None:
        embed_api_key = os.environ.get("MLX_API_KEY", "")
        if not embed_api_key:
            raise ValueError(
                "omlx embed provider requires MLX_API_KEY env var or "
                "explicit embed_api_key arg"
            )

    # Step 1: embed the literal query.
    vector = _embed_query(
        query, embed_provider, embed_model,
        api_key=embed_api_key or "",
    )

    # Optional Step 0b: HyDE — generate a hypothetical answer, embed it,
    # and BLEND it with the literal-query vector (v3 2026-06-16: blend
    # beats the original replace-the-query behavior — see `_blend_vectors`).
    # `generate_with_fallback` returns the literal query verbatim on any
    # HyDE failure (network, parse, refusal); the `!= query` guard then
    # skips the redundant second embed and the blend collapses to the
    # literal-query vector, preserving the no-HyDE result exactly.
    if use_hyde:
        from mmrag_v2.retrieval.hyde import generate_with_fallback
        hypo = generate_with_fallback(
            query, hyde_api_key or embed_api_key, provider=hyde_provider,
        )
        if hypo and hypo != query:
            hypo_vec = _embed_query(
                hypo, embed_provider, embed_model,
                api_key=embed_api_key or "",
            )
            vector = _blend_vectors(vector, hypo_vec)

    # Step 2: Qdrant vector search → top-K candidates.
    candidates = qdrant_search(
        vector, collection,
        limit=top_k_retrieve, qdrant_url=qdrant_url,
        hnsw_ef=hnsw_ef,
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
    dense_collection: str = "mmrag_v2_8__qwen3_local",
    sparse_collection: str = "mmrag_v2_8__bm25_sparse",
    bm25_index_path: str = "tests/fixtures/bm25_index_v2_12.json",
    top_k_retrieve: int = 25,
    top_n_fuse: int = 25,
    top_n_return: int = 5,
    rrf_k: int = 60,
    rrf_weights: tuple[float, float] = (1.0, 1.0),  # (dense, sparse)
    embed_provider: str = "omlx",
    embed_model: str = "Qwen3-Embedding-8B-mxfp8",
    embed_api_key: str | None = None,
    qdrant_url: str = "http://localhost:6333",
    reranker=None,
    reranker_backend: str | None = None,
    fall_back_on_rerank_error: bool = True,
    use_hyde: bool = False,
    hyde_api_key: str | None = None,
    auto_intent_hyde: bool = False,
    hyde_provider: str = "vllm",
    hnsw_ef: int | None = _DEFAULT_HNSW_EF,
) -> list[dict]:
    """Dense + BM25 sparse + RRF + reranker.

    Pipeline:
      1. embed query (Qwen3-Embedding-8B-mxfp8 via omlx-server) → dense vector
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

    HyDE knobs (v2.14; v3 2026-06-16: HyDE now BLENDS the hypothetical-
    answer embedding with the literal-query embedding rather than
    REPLACING it — see `_blend_vectors`):
      - `use_hyde=True`              — always-on HyDE (the original v2.12
                                       Phase 3 knob); applies to the dense
                                       leg only.
      - `auto_intent_hyde=True`      — v2.14 Phase 2: classifies the query
                                       via `mmrag_v2.retrieval.intent.classify_intent`
                                       and auto-enables HyDE WITH an
                                       intent-specific system prompt only
                                       when intent ∈ {`code`,
                                       `minority_language`}. English /
                                       general queries skip HyDE
                                       entirely → no latency hit. Targets
                                       the omlx per-doc deficits on German
                                       + code-dense docs (~-12pp R@1) at
                                       query time, no permanent embedder
                                       routing infra needed.
      - `hyde_provider`              — "vllm" (default — local GX10, $0;
                                       v3 2026-06-16) or "dashscope"
                                       ($ per call). Used by both manual
                                       and auto-intent paths.
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
    if embed_provider == "omlx" and embed_api_key is None:
        embed_api_key = _os.environ.get("MLX_API_KEY", "")
        if not embed_api_key:
            raise ValueError(
                "omlx embed provider requires MLX_API_KEY env var"
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
    #
    # v2.14 Phase 2: `auto_intent_hyde=True` overrides `use_hyde` for
    # queries whose intent matches the code/minority-language target
    # set. Default queries (intent=None) skip HyDE entirely → no
    # latency hit for the ~90% of queries that don't need it.
    intent: str | None = None
    if auto_intent_hyde:
        from mmrag_v2.retrieval.intent import classify_intent
        intent = classify_intent(query)
    effective_use_hyde = use_hyde or (intent is not None)

    # Step 1: embed the literal query for the dense arm.
    vector = _embed_query(
        query, embed_provider, embed_model,
        api_key=embed_api_key or "",
    )
    # HyDE BLENDS the hypothetical-answer embedding into the dense vector
    # (v3 2026-06-16: blend beats replace — see `_blend_vectors`). The
    # SPARSE arm always uses the literal query (Step 3 below) — HyDE would
    # distort BM25's keyword distribution. Falls back to the literal query
    # vector on any HyDE failure.
    if effective_use_hyde:
        from mmrag_v2.retrieval.hyde import generate_with_fallback
        hypo = generate_with_fallback(
            query,
            hyde_api_key or embed_api_key,
            provider=hyde_provider,
            intent=intent,
        )
        if hypo and hypo != query:
            hypo_vec = _embed_query(
                hypo, embed_provider, embed_model,
                api_key=embed_api_key or "",
            )
            vector = _blend_vectors(vector, hypo_vec)

    # Step 2: dense top-K.
    dense_hits = qdrant_search(
        vector, dense_collection,
        limit=top_k_retrieve, qdrant_url=qdrant_url,
        hnsw_ef=hnsw_ef,
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

    # v2.16 Phase 3: partial_code adjacency fetch.
    # For each result chunk flagged `partial_code=True`, deterministically
    # stitch up to one neighbor in each direction (text/code modalities
    # only) into the merged content. Original rerank_score is preserved.
    #
    # NOTE: in current production indexes, partial_code is set ONLY on the
    # `_chunk_text_with_overlap` scanned_book path (v2.14 P6). Chunks emitted
    # by Docling HybridChunker (the dominant path for academic_whitepaper /
    # technical_manual including Fluent_Python) never carry partial_code=True,
    # so this mechanism is INERT against the documented Fluent_Python failure
    # mode in v2.16.0. The mechanism stays in tree for future cycles + for
    # the scanned_book corpus where partial_code coverage is real. Item #9
    # routes to v2.17 per PLAN_V2.16.md §7 trigger #1.
    out = _apply_partial_code_adjacency(
        out,
        qdrant_url=qdrant_url,
        dense_collection=dense_collection,
    )

    return out


def _apply_partial_code_adjacency(
    results: list[dict],
    *,
    qdrant_url: str,
    dense_collection: str,
) -> list[dict]:
    """v2.16 Phase 3 — bounded post-rerank stitch of `partial_code=True`
    chunks with up to one text/code neighbor in each direction.

    Algorithm (per PLAN_V2.16.md §3 Phase 3):
      for each result chunk where payload.partial_code is True:
        prev = lookup(backward, filter={source_file, partial_code=True,
                                        modality ∈ {text, code}})
        next = lookup(forward,  filter={source_file, partial_code=True,
                                        modality ∈ {text, code}})
        if prev or next:
          merged.content = concat(prev?.content, current.content, next?.content)
          merged.metadata.partial_code_resolved = True
          merged.metadata.adjacency_source = [prev_id?, current_id, next_id?]
          # preserve original rerank_score / rerank_index
        else:
          merged.metadata.partial_code_resolved = False  # sole partial_code chunk

    Schema ordering (from Phase 3 step 1 verification spike):
      chunk_id format = `<doc_hash>_<page:03d>_<modality>_<content_hash8>`.
      Page-number is stable in the chunk_id; within-page order is NOT
      recoverable from chunk_id alone. Adjacency lookup uses (source_file,
      page_number, modality ∈ {text, code}) and the partial_code flag to
      identify the split halves of a single oversized code unit. When
      multiple candidate neighbors exist on a page, the FIRST one in the
      page-sorted list (lower page first, then any deterministic
      tiebreaker by chunk_id) is used.

    Non-partial_code chunks pass through unchanged. Same applies if no
    adjacent partial_code neighbor exists in either direction (sole-chunk
    case).
    """
    if not results:
        return results
    # Cheap exit when no result is partial_code-flagged (the common case).
    if not any(((r.get("payload") or {}).get("partial_code") is True) for r in results):
        return results

    out: list[dict] = []
    for r in results:
        payload = r.get("payload") or {}
        if payload.get("partial_code") is not True:
            out.append(r)
            continue
        source_file = payload.get("source_file") or ""
        page_number = payload.get("page_number")
        chunk_id = payload.get("chunk_id")
        if not source_file or page_number is None or not chunk_id:
            out.append(r)
            continue
        prev_chunk, next_chunk = _find_partial_code_neighbors(
            qdrant_url=qdrant_url,
            dense_collection=dense_collection,
            source_file=source_file,
            anchor_page=page_number,
            anchor_chunk_id=chunk_id,
        )
        if not prev_chunk and not next_chunk:
            # Sole partial_code chunk — annotate + pass through.
            merged_payload = {**payload, "partial_code_resolved": False}
            out.append({**r, "payload": merged_payload})
            continue
        parts: list[str] = []
        adjacency_source: list[str] = []
        if prev_chunk:
            parts.append((prev_chunk.get("content") or "").rstrip())
            adjacency_source.append(prev_chunk.get("chunk_id") or "")
        parts.append(payload.get("content") or "")
        adjacency_source.append(chunk_id)
        if next_chunk:
            parts.append((next_chunk.get("content") or "").lstrip())
            adjacency_source.append(next_chunk.get("chunk_id") or "")
        merged_payload = {
            **payload,
            "content": "\n".join(parts),
            "partial_code_resolved": True,
            "adjacency_source": adjacency_source,
        }
        out.append({**r, "payload": merged_payload})
    return out


def _find_partial_code_neighbors(
    *,
    qdrant_url: str,
    dense_collection: str,
    source_file: str,
    anchor_page: int,
    anchor_chunk_id: str,
) -> tuple[Optional[dict], Optional[dict]]:
    """Scroll Qdrant for partial_code=True chunks in `source_file` with
    page ∈ [anchor_page - 1, anchor_page + 1] (and modality ∈ {text, code}
    via the modality field on the payload — code chunks have
    `modality="text"` + `is_code=true` in the v2.7.0 schema, so we accept
    both modalities).

    Returns (prev, next) — each is the closest partial_code neighbor on
    the smaller / larger page; None when there is no eligible neighbor in
    that direction. Within a page, deterministic ordering is by chunk_id
    ASC (stable + cheap; the within-page split halves rarely conflict
    because partial_code emission already implies a single split sequence
    per oversized unit).
    """
    import json as _json
    import urllib.request as _urllib

    body = _json.dumps({
        "filter": {
            "must": [
                {"key": "source_file", "match": {"value": source_file}},
                {"key": "partial_code", "match": {"value": True}},
                {"key": "page_number",
                 "range": {"gte": max(0, anchor_page - 1),
                           "lte": anchor_page + 1}},
            ]
        },
        "limit": 50,  # bounded — partial_code clusters are small
        "with_payload": True,
        "with_vector": False,
    }).encode("utf-8")
    try:
        req = _urllib.Request(
            f"{qdrant_url}/collections/{dense_collection}/points/scroll",
            data=body, method="POST",
        )
        req.add_header("Content-Type", "application/json")
        with _urllib.urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read())
    except Exception:
        return (None, None)
    points = data.get("result", {}).get("points", []) or []
    # Filter to text/code modalities (skip tables/images) and exclude self.
    candidates = []
    for p in points:
        payload = p.get("payload") or {}
        cid = payload.get("chunk_id") or str(p.get("id") or "")
        if cid == anchor_chunk_id:
            continue
        modality = payload.get("modality")
        if modality and modality not in ("text", "code"):
            continue
        candidates.append(payload)
    # Sort by (page, chunk_id) ASC — deterministic.
    candidates.sort(key=lambda c: (c.get("page_number", 0), c.get("chunk_id", "")))
    prev_neighbor = None
    next_neighbor = None
    for c in candidates:
        pg = c.get("page_number", 0)
        if pg < anchor_page or (pg == anchor_page and (c.get("chunk_id") or "") < anchor_chunk_id):
            # Take the closest prev (last one seen with pg <= anchor_page).
            prev_neighbor = c
        elif pg > anchor_page or (pg == anchor_page and (c.get("chunk_id") or "") > anchor_chunk_id):
            # Take the first next.
            next_neighbor = c
            break
    return (prev_neighbor, next_neighbor)


# Resolve REPO_ROOT once for the hybrid pipeline (used in the BM25 index path).
_REPO_ROOT = Path(__file__).resolve().parents[3]
