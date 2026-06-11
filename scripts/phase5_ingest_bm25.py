#!/usr/bin/env python3
"""Phase 5 BM25 sparse twin for the re-extracted bounded subset.

Builds a BM25 index over the 12 phase5 docs' text chunks and ingests sparse
vectors into ``mmrag_v3__bm25_sparse`` on the LOCAL Mini Qdrant, parallel to the
dense ``mmrag_v3__qwen3_local``. Hybrid RRF fusion joins dense+sparse by
``chunk_id`` (``src/mmrag_v2/retrieval/fusion_v3.py`` - leg scores keyed by
chunk_id), which both collections carry in payload; point IDs are computed in the
SAME namespace as the dense ingester for consistency (not required by fusion).

Self-contained (does NOT import the v2.16-pinned ``ingest_bm25_sparse.py``, whose
hardcoded ``CANONICAL_DOCS`` is the wrong corpus). Idempotent: drop + recreate.

Run: python scripts/phase5_ingest_bm25.py [--qdrant-url http://127.0.0.1:6333]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mmrag_v2.retrieval.sparse import BM25Index  # noqa: E402

# Match the dense ingester's point-id namespace (scripts/ingest_to_qdrant.py).
_POINT_ID_NAMESPACE = uuid.UUID("8b7c5e3a-1f4d-4b2a-9c1e-6d8a3f0b9c2e")
SPARSE_VECTOR_NAME = "bm25"
DEFAULT_COLLECTION = "mmrag_v3__bm25_sparse"
DEFAULT_QDRANT = "http://127.0.0.1:6333"
BASE_DIR = REPO_ROOT / "output" / "phase5_reextract"


def discover_docs() -> list[str]:
    """Every output/phase5_reextract/<base>/ that has an ingestion.jsonl, sorted.

    Auto-discovery so the sparse twin always covers whatever has been re-extracted
    (the full-corpus run grows this set; no hardcoded list to drift). Excludes the
    smoke/src helper dirs (leading underscore)."""
    return sorted(
        p.name
        for p in BASE_DIR.iterdir()
        if p.is_dir() and not p.name.startswith("_") and (p / "ingestion.jsonl").exists()
    )


DOCS = discover_docs()


def _http(method: str, url: str, body=None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def iter_text_chunks():
    """Yield (doc, chunk_id, content) for every text chunk across the 12 docs."""
    for doc in DOCS:
        jl = BASE_DIR / doc / "ingestion.jsonl"
        if not jl.exists():
            print(f"  MISSING {jl} - skip")
            continue
        with open(jl, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if i == 0 and obj.get("object_type") == "ingestion_metadata":
                    continue
                if obj.get("modality") != "text":
                    continue
                cid = obj.get("chunk_id")
                content = obj.get("content") or ""
                if cid and content:
                    yield doc, cid, content


def ensure_collection(qdrant_url: str, collection: str) -> None:
    try:
        _http("DELETE", f"{qdrant_url}/collections/{collection}")
        print(f"Dropped existing {collection!r}")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"  delete HTTP {e.code} (continuing)")
    _http(
        "PUT",
        f"{qdrant_url}/collections/{collection}",
        {"sparse_vectors": {SPARSE_VECTOR_NAME: {"index": {"on_disk": True}}}},
    )
    print(f"Created sparse collection {collection!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant-url", default=DEFAULT_QDRANT)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--batch-size", type=int, default=256)
    a = ap.parse_args()

    # 1. Collect text chunks.
    rows = list(iter_text_chunks())
    print(f"Collected {len(rows)} text chunks across {len(DOCS)} docs")
    if not rows:
        print("ERROR: no text chunks found", file=sys.stderr)
        return 1

    # 2. Build BM25 index over this corpus.
    t0 = time.perf_counter()
    index = BM25Index.build_from_corpus([c for _, _, c in rows])
    print(
        f"BM25 index: vocab={len(index.vocab)}, avgdl={index.avgdl:.1f}, "
        f"n_docs={index.n_docs} ({time.perf_counter()-t0:.1f}s)"
    )

    # 3. Recreate sparse collection.
    ensure_collection(a.qdrant_url, a.collection)

    # 4. Encode + upsert.
    batch, n_total, n_zero = [], 0, 0
    per_doc: dict[str, int] = {}
    for doc, cid, content in rows:
        n_total += 1
        per_doc[doc] = per_doc.get(doc, 0) + 1
        indices, values = index.encode_document(content)
        if not indices:
            n_zero += 1
            continue
        batch.append(
            {
                "id": str(uuid.uuid5(_POINT_ID_NAMESPACE, cid)),
                "vector": {SPARSE_VECTOR_NAME: {"indices": indices, "values": values}},
                "payload": {"chunk_id": cid, "doc_dir": doc},
            }
        )
        if len(batch) >= a.batch_size:
            _http(
                "PUT",
                f"{a.qdrant_url}/collections/{a.collection}/points?wait=true",
                {"points": batch},
            )
            batch = []
    if batch:
        _http(
            "PUT", f"{a.qdrant_url}/collections/{a.collection}/points?wait=true", {"points": batch}
        )

    info = _http("GET", f"{a.qdrant_url}/collections/{a.collection}")["result"]
    print(
        f"\n=== DONE: {info.get('points_count')} sparse points, " f"status={info.get('status')} ==="
    )
    print(f"text chunks={n_total}, zero-sparse(OOV)={n_zero}")
    for d in DOCS:
        print(f"  {d:<22} {per_doc.get(d,0)} text chunks")
    print("INGEST_SPARSE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
