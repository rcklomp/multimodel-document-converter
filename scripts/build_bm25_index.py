#!/usr/bin/env python3
"""v2.12 Phase 2 — build the BM25 index over the 34-doc canonical corpus.

Reads every `output/<doc>/ingestion.jsonl`, collects text-modality
chunks, tokenizes them, and writes the index to
`tests/fixtures/bm25_index_v2_12.json` (tracked in the repo).

The index is then consumed by:
  - `scripts/ingest_sparse_collection.py` — encodes each chunk as a
    sparse Qdrant vector and ingests into the side collection.
  - `mmrag_v2.retrieval.pipeline.retrieve_hybrid_reranked()` — at
    query time, encodes the query as a sparse vector for the
    BM25 leg of the hybrid retrieve.

Idempotent: re-running rebuilds the index from scratch. The vocab is
sorted deterministically so two builds over the same JSONLs produce
byte-identical output.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmrag_v2.retrieval.sparse import BM25Index  # noqa: E402


# Mirror the canonical 34-doc list from rebuild_mmrag_v2_8_for_rc1.py.
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "_rebuild_mod", SCRIPTS / "rebuild_mmrag_v2_8_for_rc1.py"
)
_rebuild_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_rebuild_mod)
CANONICAL_34 = _rebuild_mod.CANONICAL_34


def iter_text_chunks(jsonl_path: Path):
    """Yield (chunk_id, content) for every text chunk in a JSONL."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
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
                # BM25 over text only. Image chunks embed via their
                # visual_description in the dense lane; we don't add
                # sparse weights for them here.
                continue
            cid = obj.get("chunk_id")
            content = obj.get("content") or ""
            if cid and content:
                yield cid, content


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir", default="output",
        help="Directory containing <doc>/ingestion.jsonl per canonical doc.",
    )
    parser.add_argument(
        "--index-path",
        default="tests/fixtures/bm25_index_v2_12.json",
        help="Where to write the persisted index.",
    )
    parser.add_argument(
        "--bm25-k1", type=float, default=1.5,
        help="BM25 k1 hyperparameter (default 1.5).",
    )
    parser.add_argument(
        "--bm25-b", type=float, default=0.75,
        help="BM25 b hyperparameter (default 0.75).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    index_path = Path(args.index_path)
    if not index_path.is_absolute():
        index_path = REPO_ROOT / index_path

    print(f"=== BM25 index build ===")
    print(f"Source: {output_dir}/<doc>/ingestion.jsonl")
    print(f"Docs:   {len(CANONICAL_34)} canonical")
    print()

    # Gather all text chunks across the canonical corpus.
    docs: list[str] = []
    chunk_count_per_doc: dict[str, int] = {}
    missing_docs: list[str] = []
    t0 = time.perf_counter()
    for doc_name in CANONICAL_34:
        jsonl = output_dir / doc_name / "ingestion.jsonl"
        if not jsonl.exists():
            missing_docs.append(doc_name)
            chunk_count_per_doc[doc_name] = 0
            continue
        n = 0
        for cid, content in iter_text_chunks(jsonl):
            docs.append(content)
            n += 1
        chunk_count_per_doc[doc_name] = n
        print(f"  {doc_name:35s} {n:5d} text chunks")

    if missing_docs:
        print()
        print(f"WARNING: {len(missing_docs)} canonical docs missing:")
        for d in missing_docs:
            print(f"  - {d}")

    print()
    print(f"Total text chunks across corpus: {len(docs)}")

    # Build the index.
    print("Building BM25 index...")
    t1 = time.perf_counter()
    index = BM25Index.build_from_corpus(docs, k1=args.bm25_k1, b=args.bm25_b)
    t2 = time.perf_counter()
    print(f"  Vocab size: {len(index.vocab):,} unique tokens")
    print(f"  avgdl:      {index.avgdl:.2f} tokens/doc")
    print(f"  n_docs:     {index.n_docs}")
    print(f"  build time: {t2 - t1:.2f}s")

    # Persist.
    index.save(index_path)
    idx_size = index_path.stat().st_size
    print(f"\nIndex written to {index_path.relative_to(REPO_ROOT)} ({idx_size:,} bytes)")
    print(f"Total time: {time.perf_counter() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
