#!/usr/bin/env python3
"""BM25 sparse twin for the dense-ingested corpus.

Builds a BM25 index over the text chunks of the docs that are ACTUALLY ingested
into the dense collection ``mmrag_v3__qwen3_local`` and ingests sparse vectors
into ``mmrag_v3__bm25_sparse`` on the LOCAL Mini Qdrant. Hybrid RRF fusion joins
dense+sparse by ``chunk_id`` (``src/mmrag_v2/retrieval/fusion_v3.py`` - leg scores
keyed by chunk_id), which both collections carry in payload; point IDs are
computed in the SAME namespace as the dense ingester for consistency.

Doc selection is SCOPED TO THE DENSE DOC SET (2026-06-13 fix): the script scrolls
the dense collection's doc_ids and indexes only ``output/phase5_reextract/<base>/``
dirs whose header doc_id is dense-ingested. This prevents the asymmetry where the
phase5_reextract scratch dir (which accumulates re-extractions of EXCLUDED /
never-ingested docs - code books, failed-QA docs) leaks those docs' text into a
BM25 index that has no dense counterpart, polluting RRF with sparse-only hits.

Self-contained (does NOT import the v2.16-pinned ``ingest_bm25_sparse.py``, whose
hardcoded ``CANONICAL_DOCS`` is the wrong corpus). Idempotent: drop + recreate.

Run: python scripts/phase5_ingest_bm25.py [--qdrant-url http://127.0.0.1:6333]
     [--dense-collection mmrag_v3__qwen3_local]
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
DEFAULT_DENSE_COLLECTION = "mmrag_v3__qwen3_local"
DEFAULT_QDRANT = "http://127.0.0.1:6333"
BASE_DIR = REPO_ROOT / "output" / "phase5_reextract"
# The query-side BM25 index MUST be persisted: hybrid retrieval
# (retrieve_hybrid_reranked) loads it to encode queries for the sparse leg.
# Earlier this index was built in-memory and discarded, so the sparse leg never
# had a matching query encoder and hybrid silently fell back to dense-only.
INDEX_PATH = REPO_ROOT / "tests" / "fixtures" / "bm25_index_v3.json"


def _http(method: str, url: str, body=None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def dense_doc_ids(qdrant_url: str, dense_collection: str) -> set[str]:
    """The set of doc_ids actually ingested into the dense collection.

    The sparse twin MUST mirror the dense doc set so RRF (fused by chunk_id) is
    aligned and excluded/un-ingested docs are not made BM25-retrievable. Scrolls
    the dense payloads (read-only)."""
    ids: set[str] = set()
    offset = None
    while True:
        body = {"limit": 1000, "with_payload": ["doc_id"], "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        res = _http("POST", f"{qdrant_url}/collections/{dense_collection}/points/scroll", body)[
            "result"
        ]
        for p in res["points"]:
            did = (p.get("payload") or {}).get("doc_id")
            if did:
                ids.add(did)
        offset = res.get("next_page_offset")
        if offset is None:
            break
    return ids


def _dir_doc_id(jsonl: Path) -> str | None:
    """The doc_id from a dir's ingestion.jsonl header (first line)."""
    try:
        with open(jsonl, encoding="utf-8") as f:
            return json.loads(f.readline()).get("doc_id")
    except (OSError, json.JSONDecodeError):
        return None


def discover_docs(dense_ids: set[str]) -> list[str]:
    """output/phase5_reextract/<base>/ dirs whose doc_id is dense-ingested, sorted.

    Scoped to the dense doc set (not the whole scratch dir): phase5_reextract
    accumulates re-extractions of docs that were never ingested (excluded code
    books, failed-QA docs), and indexing those into sparse would make their text
    BM25-retrievable while absent from dense - a dense/sparse asymmetry that
    pollutes RRF. Excludes smoke/src helper dirs (leading underscore)."""
    out = []
    for p in sorted(BASE_DIR.iterdir()):
        if not p.is_dir() or p.name.startswith("_"):
            continue
        jl = p / "ingestion.jsonl"
        if not jl.exists():
            continue
        if _dir_doc_id(jl) in dense_ids:
            out.append(p.name)
    return out


def iter_text_chunks(docs: list[str]):
    """Yield (doc, chunk_id, content) for every text chunk across the docs."""
    for doc in docs:
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
    ap.add_argument("--dense-collection", default=DEFAULT_DENSE_COLLECTION)
    ap.add_argument("--batch-size", type=int, default=256)
    a = ap.parse_args()

    # 0. Scope to the dense doc set (mirror dense; do NOT index the whole scratch dir).
    try:
        dense_ids = dense_doc_ids(a.qdrant_url, a.dense_collection)
    except Exception as e:
        print(f"ERROR: cannot read dense collection {a.dense_collection!r}: {e}", file=sys.stderr)
        return 1
    if not dense_ids:
        print(f"ERROR: dense collection {a.dense_collection!r} has no doc_ids", file=sys.stderr)
        return 1
    docs = discover_docs(dense_ids)
    print(f"Dense doc set: {len(dense_ids)} doc_ids; matched {len(docs)} phase5_reextract dirs")

    # 1. Collect text chunks.
    rows = list(iter_text_chunks(docs))
    print(f"Collected {len(rows)} text chunks across {len(docs)} docs")
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
    # 2a. PERSIST the query-side index so hybrid retrieval can encode queries
    # against the SAME vocab/IDF used to encode these chunks. Without this the
    # sparse leg is unusable and hybrid degrades to dense-only.
    index.save(INDEX_PATH)
    print(f"Saved query-side BM25 index -> {INDEX_PATH}")

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
    for d in docs:
        print(f"  {d:<22} {per_doc.get(d,0)} text chunks")
    print("INGEST_SPARSE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
