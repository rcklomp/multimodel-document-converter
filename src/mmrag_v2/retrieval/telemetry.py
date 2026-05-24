"""v2.15 Phase 3 [F] — retrieval-side telemetry primitives.

Two helpers used by the soak harness (and any production caller that
wants to write to the rolling document-class-hits log):

  compute_document_class_hits(reranked, classes, top_k=5)
    → list of documented-limitation class names that appear in the
      first `top_k` reranked results. Doc identity is by `doc_id`
      (production qdrant payloads) or `doc_dir` (soak fixture rows).

  build_telemetry_record(query, reranked, classes, *, timestamp=None)
    → dict in the canonical telemetry-log row format expected by
      `scripts/analyze_doc_class_telemetry.py`. Stamps current
      timestamp by default.

Design choice: the pipeline (`retrieve_hybrid_reranked`) stays
side-effect-free; the soak harness owns the write to
`output/telemetry/document_class_hits.jsonl` via these helpers.
Keeps the production retrieval path uncoupled from telemetry I/O
while the harness composes both.
"""
from __future__ import annotations

import time
from typing import Iterable, Optional, Sequence


def _doc_id_of(chunk: dict) -> str:
    """Extract a doc-class identifier from a chunk record.

    Tries (in order):
      1. payload.doc_id            (production qdrant shape)
      2. payload.doc_dir           (soak harness sample shape)
      3. top-level doc_id          (fallback for already-flattened chunks)
      4. top-level doc_dir         (same)
      5. ""                        (missing-id sentinel)
    """
    payload = chunk.get("payload") or {}
    return (
        payload.get("doc_id")
        or payload.get("doc_dir")
        or chunk.get("doc_id")
        or chunk.get("doc_dir")
        or ""
    )


def compute_document_class_hits(
    reranked: Sequence[dict],
    documented_limitation_classes: Iterable[str],
    *,
    top_k: int = 5,
) -> list[str]:
    """Return the subset of documented-limitation classes that appear
    in the first `top_k` reranked results.

    Ordering of the returned list reflects rank position (first
    appearance wins). De-duplicated: each class appears at most once
    even if multiple top-k chunks come from it.

    `top_k=5` matches the production retrieval shape; production
    pipelines that surface a different top_n_return to the consumer
    should pass that value here.

    Empty input → empty output. Inputs longer than `top_k` are
    truncated. Inputs missing both `doc_id` and `doc_dir` are silently
    skipped (treated as no-class hit).
    """
    classes_set = set(documented_limitation_classes)
    if not classes_set:
        return []
    seen: list[str] = []
    seen_set: set[str] = set()
    for chunk in list(reranked)[:top_k]:
        doc_id = _doc_id_of(chunk)
        if doc_id in classes_set and doc_id not in seen_set:
            seen.append(doc_id)
            seen_set.add(doc_id)
    return seen


def build_telemetry_record(
    query: str,
    reranked: Sequence[dict],
    documented_limitation_classes: Iterable[str],
    *,
    timestamp: Optional[float] = None,
    top_k: int = 5,
) -> dict:
    """Build a single telemetry-log row in the canonical format the
    `analyze_doc_class_telemetry.py` analyzer expects.

    Schema:
      query                     : str
      timestamp                 : float (unix epoch seconds)
      document_class_hits       : list[str]
      rerank_top_5_doc_ids      : list[str]   (always length top_k or
                                              the actual rerank length
                                              if smaller)
      rerank_top_5_non_empty    : bool        (False = retrieval
                                              returned 0 results; the
                                              row is excluded from
                                              denominator counts in
                                              the analyzer)
    """
    if timestamp is None:
        timestamp = time.time()
    top = list(reranked)[:top_k]
    return {
        "query": query,
        "timestamp": timestamp,
        "document_class_hits": compute_document_class_hits(
            top, documented_limitation_classes, top_k=top_k,
        ),
        "rerank_top_5_doc_ids": [_doc_id_of(c) for c in top],
        "rerank_top_5_non_empty": len(top) > 0,
    }
