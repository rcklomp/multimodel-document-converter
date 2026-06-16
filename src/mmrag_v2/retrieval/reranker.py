"""v2.12 reranker provider abstraction.

Two implementations of the same interface — pick at runtime via
`mmrag_v2.retrieval.config.get_reranker(name)`:

  - `DashscopeReranker`   cloud `gte-rerank` via Dashscope intl API.
  - `LocalOmlxReranker`   local `gte-reranker-modernbert-base-mlx`
                          (or any model-name-compatible) via omlx-server.

Both implement `rerank(query, chunks, top_n) -> list[dict]` returning
the input chunks in rerank order, each augmented with `rerank_score`
and `rerank_index` (original position in the candidate set). The
returned list is sliced to top_n if provided; if not, returns all
candidates in reranker order.

The chunks input must be a list of dicts with at least:

  {"chunk_id": str, "content": str}

Additional payload fields are preserved verbatim. The implementation
truncates `content` to `max_chunk_chars` (default 1500) before
sending to the API to bound request size + latency.

Error semantics: any HTTP / network error raises `RerankerError` with
the underlying cause. Callers can catch this and decide to fall back
to vector-rank order (i.e. return chunks unchanged) rather than
failing the whole retrieval. Both providers implement retry-on-429
and retry-on-5xx with exponential backoff up to `retries` attempts.

Empirical latency baselines (per `tests/fixtures/reranker_latency_*`,
2026-05-21):

  DashscopeReranker  K=25 p99 = 1.70 s     network-dominated
  LocalOmlxReranker  K=25 p99 = 0.55 s     compute on Mac Mini

Empirical quality side-by-side (per
`tests/fixtures/reranker_quality_modernbert_2026-05-21.json`,
2026-05-21): 15% top-1 agreement, 0.239 mean Jaccard. Two GTE-family
rerankers with different training data; the Phase 1 soak picks the
production winner.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from mmrag_v2.endpoints import endpoint as _endpoint


class RerankerError(RuntimeError):
    """Raised by Reranker implementations on unrecoverable HTTP / network
    errors after retries are exhausted. Callers may catch this and fall
    back to vector-rank order."""


class Reranker(Protocol):
    """Protocol for rerank providers. Stateless: configuration is fixed
    at construction; each `rerank()` call is independent."""

    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        *,
        top_n: int | None = None,
    ) -> list[dict]:
        """Reorder chunks by relevance to query.

        Returns a NEW list (does not mutate input). Each returned chunk
        is the input chunk dict with two added fields:

          - `rerank_score`  float relevance score
          - `rerank_index`  int   original position in the input list

        If `top_n` is given, the result is truncated to top_n; otherwise
        all input chunks are returned in rerank order.
        """
        ...


@dataclass(frozen=True)
class _RerankerCommon:
    """Shared knobs for both reranker implementations."""
    model: str
    api_key: str
    url: str
    max_chunk_chars: int = 1500
    timeout: int = 60
    retries: int = 4
    backoff_base_seconds: float = 1.0


def _truncate_docs(chunks: list[dict], max_chars: int) -> list[str]:
    return [
        ((c.get("content") or "")[:max_chars])
        for c in chunks
    ]


def _attach_scores(
    chunks: list[dict],
    results: list[dict],
    *,
    score_key: str = "relevance_score",
) -> list[dict]:
    """Map reranker output (list of {"index", score_key}) back onto the
    original chunks, adding `rerank_score` + `rerank_index`. Returns
    chunks in reranker-output order (which is typically descending by
    score)."""
    out: list[dict] = []
    for r in results:
        idx = r.get("index")
        if idx is None or idx < 0 or idx >= len(chunks):
            continue
        chunk = dict(chunks[idx])
        chunk["rerank_score"] = float(r.get(score_key) or 0.0)
        chunk["rerank_index"] = int(idx)
        out.append(chunk)
    return out


def _post_with_retries(
    common: _RerankerCommon,
    body: bytes,
    *,
    description: str,
) -> dict:
    """POST to `common.url` with the standard Bearer auth + JSON
    content-type, retry on 429/5xx with exponential backoff."""
    last_err: Exception | None = None
    for attempt in range(common.retries):
        try:
            req = urllib.request.Request(common.url, data=body, method="POST")
            req.add_header("Authorization", f"Bearer {common.api_key}")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=common.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(common.backoff_base_seconds * (2 ** attempt))
                continue
            # Non-retryable HTTP error.
            try:
                detail = e.read().decode("utf-8", errors="replace")[:200]
            except Exception:
                detail = ""
            raise RerankerError(
                f"{description}: HTTP {e.code} {detail}"
            ) from e
        except (urllib.error.URLError, ConnectionError, OSError, TimeoutError) as e:
            last_err = e
            time.sleep(common.backoff_base_seconds * (2 ** attempt))
            continue
    raise RerankerError(
        f"{description}: failed after {common.retries} retries; last error: {last_err}"
    ) from last_err


class DashscopeReranker:
    """Cloud `gte-rerank` via Dashscope intl endpoint. Same
    `DASHSCOPE_API_KEY` env var as the embedder."""

    DEFAULT_URL = (
        "https://dashscope-intl.aliyuncs.com/api/v1/services/rerank/"
        "text-rerank/text-rerank"
    )
    DEFAULT_MODEL = "gte-rerank"
    name = "dashscope"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str | None = None,
        url: str | None = None,
        max_chunk_chars: int = 1500,
        timeout: int = 60,
        retries: int = 4,
    ):
        resolved_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        if not resolved_key:
            raise RerankerError(
                "DashscopeReranker requires api_key arg or DASHSCOPE_API_KEY env var"
            )
        self._common = _RerankerCommon(
            model=model or self.DEFAULT_MODEL,
            api_key=resolved_key,
            url=url or self.DEFAULT_URL,
            max_chunk_chars=max_chunk_chars,
            timeout=timeout,
            retries=retries,
        )

    @property
    def model(self) -> str:
        return self._common.model

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        *,
        top_n: int | None = None,
    ) -> list[dict]:
        if not chunks:
            return []
        documents = _truncate_docs(chunks, self._common.max_chunk_chars)
        body = json.dumps({
            "model": self._common.model,
            "input": {"query": query, "documents": documents},
            "parameters": {
                "top_n": top_n if top_n is not None else len(documents),
                "return_documents": False,
            },
        }).encode("utf-8")
        payload = _post_with_retries(
            self._common, body,
            description=f"dashscope rerank model={self._common.model}",
        )
        results = (payload.get("output") or {}).get("results", []) or []
        return _attach_scores(chunks, results)


class LocalOmlxReranker:
    """Local cross-encoder via omlx-server's Cohere-style /v1/rerank
    endpoint. Default model: `gte-reranker-modernbert-base-mlx`."""

    # Resolved from the central endpoint registry (env-overridable);
    # default is the Mini-hosted oMLX ModernBERT reranker.
    DEFAULT_URL = _endpoint("rerank").rerank_url
    DEFAULT_MODEL = _endpoint("rerank").model
    name = "omlx"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str | None = None,
        url: str | None = None,
        max_chunk_chars: int = 1500,
        timeout: int = 120,  # local 8B-class models can be slower than cloud
        retries: int = 3,
    ):
        resolved_key = api_key or os.environ.get("MLX_API_KEY", "")
        if not resolved_key:
            raise RerankerError(
                "LocalOmlxReranker requires api_key arg or MLX_API_KEY env var"
            )
        self._common = _RerankerCommon(
            model=model or self.DEFAULT_MODEL,
            api_key=resolved_key,
            url=url or self.DEFAULT_URL,
            max_chunk_chars=max_chunk_chars,
            timeout=timeout,
            retries=retries,
        )

    @property
    def model(self) -> str:
        return self._common.model

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        *,
        top_n: int | None = None,
    ) -> list[dict]:
        if not chunks:
            return []
        documents = _truncate_docs(chunks, self._common.max_chunk_chars)
        body = json.dumps({
            "model": self._common.model,
            "query": query,
            "documents": documents,
            "top_n": top_n if top_n is not None else len(documents),
            "return_documents": False,
        }).encode("utf-8")
        payload = _post_with_retries(
            self._common, body,
            description=f"omlx rerank model={self._common.model}",
        )
        results = payload.get("results", []) or []
        return _attach_scores(chunks, results)


class _NullReranker:
    """No-op reranker — returns chunks in original (vector-rank) order,
    annotated with the original index. Used as a fallback when no
    reranker is configured, or by tests that want to verify pipeline
    composition without invoking a real backend."""
    name = "null"
    model = "(vector-rank)"

    def rerank(
        self,
        query: str,  # noqa: ARG002
        chunks: list[dict],
        *,
        top_n: int | None = None,
    ) -> list[dict]:
        out = []
        for idx, c in enumerate(chunks):
            chunk = dict(c)
            chunk["rerank_score"] = 0.0
            chunk["rerank_index"] = idx
            out.append(chunk)
        if top_n is not None:
            out = out[:top_n]
        return out
