#!/usr/bin/env python3
"""v2.12 Phase 2 — ingest BM25 sparse vectors into a side collection.

Creates `mmrag_v2_8__bm25_sparse` — a sparse-only Qdrant collection
parallel to the dense `mmrag_v2_8__qwen3_dashscope` collection.
Point IDs match across collections (same uuid5(chunk_id)) so the
hybrid retrieval pipeline can fuse dense + sparse rankings at query
time via chunk_id.

Why a side collection rather than a named sparse vector on the dense
collection: Qdrant 1.17 doesn't support adding a sparse vector
schema to an existing collection via PATCH (sparse_vectors_config is
fixed at creation). A side collection keeps the dense collection
untouched (rollback-safe) and is cheap to ingest (~2 sec for 25k
chunks; no Dashscope re-embed needed).

Idempotent: re-running drops + recreates the side collection.

Image + table chunks are skipped (no BM25 vector; the dense lane
covers them via their VLM description text).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
SCRIPTS = REPO_ROOT / "scripts"
for p in (SRC, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from mmrag_v2.retrieval.sparse import BM25Index  # noqa: E402
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_rebuild_mod", SCRIPTS / "rebuild_mmrag_v2_8_for_rc1.py"
)
_rebuild_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rebuild_mod)
CANONICAL_DOCS = _rebuild_mod.CANONICAL_DOCS

QDRANT_URL_DEFAULT = "http://localhost:6333"
COLLECTION_DEFAULT = "mmrag_v2_8__bm25_sparse"
DENSE_COLLECTION = "mmrag_v2_8__qwen3_dashscope"  # source of truth for chunk_id→point_id
INDEX_PATH_DEFAULT = "tests/fixtures/bm25_index_v2_12.json"
SPARSE_VECTOR_NAME = "bm25"


def _http_json(method: str, url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _chunk_id_to_point_id(chunk_id: str) -> str:
    """Mirror scripts/ingest_to_qdrant.py — deterministic uuid5 from chunk_id."""
    import uuid
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def ensure_sparse_collection(qdrant_url: str, collection: str) -> None:
    """Create a sparse-only collection (idempotent: drop + recreate)."""
    # 1. Drop existing collection if present (idempotent reingest).
    try:
        _http_json("DELETE", f"{qdrant_url}/collections/{collection}")
        print(f"Dropped existing collection {collection!r}")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"  warning: delete returned HTTP {e.code} (continuing)")
    # 2. Create the sparse-only collection.
    body = {
        # No "vectors" field => the collection has no dense vector.
        # qdrant allows pure-sparse collections.
        "sparse_vectors": {
            SPARSE_VECTOR_NAME: {
                "index": {"on_disk": True}
            }
        }
    }
    result = _http_json(
        "PUT",
        f"{qdrant_url}/collections/{collection}",
        body,
    )
    print(f"Created collection {collection!r} (result={result.get('result')})")


def iter_text_chunks(output_dir: Path, canonical_docs: list[str]):
    """Yield (doc_name, chunk_id, content) for every text chunk."""
    for doc_name in canonical_docs:
        jsonl = output_dir / doc_name / "ingestion.jsonl"
        if not jsonl.exists():
            continue
        with open(jsonl, "r", encoding="utf-8") as f:
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
                    yield doc_name, cid, content


def upsert_sparse_batch(qdrant_url: str, collection: str,
                        points_batch: list[dict]) -> None:
    """Upsert a batch of points with sparse vector + chunk_id payload."""
    if not points_batch:
        return
    body = {"points": points_batch}
    url = f"{qdrant_url}/collections/{collection}/points?wait=true"
    _http_json("PUT", url, body)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--collection", default=COLLECTION_DEFAULT)
    parser.add_argument("--qdrant-url", default=QDRANT_URL_DEFAULT)
    parser.add_argument("--index-path", default=INDEX_PATH_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip the Qdrant writes; just print what would happen.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    index_path = Path(args.index_path)
    if not index_path.is_absolute():
        index_path = REPO_ROOT / index_path
    if not index_path.exists():
        print(f"ERROR: BM25 index missing at {index_path}; run "
              f"scripts/build_bm25_index.py first.", file=sys.stderr)
        return 1

    print(f"=== Phase 2 sparse ingest ===")
    print(f"Collection:  {args.collection}")
    print(f"Qdrant URL:  {args.qdrant_url}")
    print(f"BM25 index:  {index_path.relative_to(REPO_ROOT)}")
    print(f"Batch size:  {args.batch_size}")
    print(f"Dry run:     {args.dry_run}")
    print()

    # Step 1: ensure sparse-only side collection exists (drop + recreate).
    if not args.dry_run:
        ensure_sparse_collection(args.qdrant_url, args.collection)

    # Step 2: load BM25 index.
    index = BM25Index.load(index_path)
    print(f"Loaded BM25 index: vocab={len(index.vocab)}, "
          f"avgdl={index.avgdl:.2f}, n_docs={index.n_docs}")
    print()

    # Step 3: stream text chunks, encode, batch-upsert.
    t0 = time.perf_counter()
    batch: list[dict] = []
    n_total = 0
    n_zero_sparse = 0
    n_per_doc: dict[str, int] = {}
    for doc_name, cid, content in iter_text_chunks(output_dir, CANONICAL_DOCS):
        indices, values = index.encode_document(content)
        n_total += 1
        n_per_doc[doc_name] = n_per_doc.get(doc_name, 0) + 1
        if not indices:
            n_zero_sparse += 1
            continue  # nothing to upsert for this chunk
        point = {
            "id": _chunk_id_to_point_id(cid),
            "vector": {
                SPARSE_VECTOR_NAME: {
                    "indices": indices,
                    "values": values,
                }
            },
            # Side collection stores just chunk_id in payload so a hybrid
            # query can fuse sparse + dense rankings by chunk_id. The full
            # payload (content, doc_id, etc.) lives on the dense collection.
            "payload": {"chunk_id": cid, "doc_dir": doc_name},
        }
        batch.append(point)
        if len(batch) >= args.batch_size:
            if not args.dry_run:
                upsert_sparse_batch(args.qdrant_url, args.collection, batch)
            print(f"  ... upserted {n_total} chunks "
                  f"(~{n_total / (time.perf_counter() - t0):.0f}/sec)")
            batch = []
    if batch and not args.dry_run:
        upsert_sparse_batch(args.qdrant_url, args.collection, batch)
        print(f"  ... upserted {n_total} chunks (final batch)")

    elapsed = time.perf_counter() - t0
    print()
    print(f"=== Done ===")
    print(f"Total chunks processed: {n_total}")
    print(f"Chunks with zero sparse vector (OOV-only): {n_zero_sparse}")
    print(f"Wall time: {elapsed:.1f}s ({n_total / elapsed:.0f} chunks/sec)")

    # Verify a few sparse vectors landed.
    if not args.dry_run:
        print()
        print("=== Verification: scroll 1 point to confirm sparse vector landed ===")
        verify_body = {
            "limit": 1,
            "with_vector": True,
            "with_payload": False,
        }
        result = _http_json(
            "POST",
            f"{args.qdrant_url}/collections/{args.collection}/points/scroll",
            verify_body,
        )
        points = result.get("result", {}).get("points", [])
        if points:
            vec = points[0].get("vector") or {}
            sparse = vec.get(SPARSE_VECTOR_NAME) if isinstance(vec, dict) else None
            if sparse:
                idx = sparse.get("indices") or []
                vals = sparse.get("values") or []
                print(f"  sample point sparse vector: "
                      f"{len(idx)} nonzero entries, "
                      f"first index/value: {idx[:3]} / {[round(v,3) for v in vals[:3]]}")
            else:
                print(f"  WARNING: sparse vector not found on sample point")
                print(f"  full vector dict: {vec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
