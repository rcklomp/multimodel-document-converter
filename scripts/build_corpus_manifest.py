#!/usr/bin/env python3
"""Backfill ``corpus_manifest.jsonl`` (PLAN_F1 WP-B / D1).

One row per ``data/**`` PDF, joining three sources:
  1. the source file (doc_id = md5(file)[:12], sha256, repo-relative path);
  2. the freshest extraction provenance found in ``output/**/ingestion.jsonl``
     headers (engine/route/schema/version/timestamp/pages/degraded), matched by
     doc_id and selected by latest ``ingestion_timestamp``;
  3. the live Qdrant collections (dense ``mmrag_v3__qwen3_local`` keyed on the
     ``doc_id`` payload; sparse ``mmrag_v3__bm25_sparse`` keyed on the
     ``chunk_id`` prefix).

Outcome is evidence-based and conservative:
  - INGESTED   : has dense points in Qdrant;
  - LADDER_FAIL: extracted with degraded/laddered pages, not ingested;
  - PENDING    : never extracted, or extracted but not yet ingested.
(CONTENT_FAIL is reserved for a recorded QA_FAIL; the reconciliation wave
writes it. Backfill never invents a QA verdict it did not observe.)

The schema is the WP-7 contract documented inline in HANDOVER_OVERNIGHT_0613.md.
Re-runnable; writes sorted by source_path for a stable diff. Requires a live
Qdrant (read-only) for the ingest columns; pass --no-qdrant to skip them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"
MANIFEST_PATH = REPO_ROOT / "corpus_manifest.jsonl"

DENSE_COLLECTION = "mmrag_v3__qwen3_local"
SPARSE_COLLECTION = "mmrag_v3__bm25_sparse"
QDRANT_URL = "http://localhost:6333"


def _file_hashes(path: Path) -> Tuple[str, str]:
    """Return (doc_id = md5(file)[:12], sha256 hexdigest) for a file."""
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            md5.update(chunk)
            sha.update(chunk)
    return md5.hexdigest()[:12], sha.hexdigest()


def _scan_extraction_headers() -> Dict[str, Dict[str, Any]]:
    """Index the freshest extraction header per doc_id from output/**.

    Reads only the first line of each ingestion.jsonl (the
    ``object_type=ingestion_metadata`` header). Keeps the record with the
    latest ``ingestion_timestamp`` per doc_id.
    """
    latest: Dict[str, Dict[str, Any]] = {}
    if not OUTPUT_DIR.exists():
        return latest
    for jsonl in OUTPUT_DIR.rglob("ingestion.jsonl"):
        try:
            with open(jsonl, "r", encoding="utf-8") as f:
                first = f.readline()
            if not first.strip():
                continue
            header = json.loads(first)
        except (OSError, json.JSONDecodeError):
            continue
        if header.get("object_type") != "ingestion_metadata":
            continue
        doc_id = header.get("doc_id")
        if not doc_id:
            continue
        ts = header.get("ingestion_timestamp") or ""
        prev = latest.get(doc_id)
        if prev is None or ts > (prev.get("ingestion_timestamp") or ""):
            header["_source_jsonl"] = str(jsonl.relative_to(REPO_ROOT))
            latest[doc_id] = header
    return latest


def _qdrant_dense_counts() -> Counter:
    """Dense points per doc_id (payload doc_id)."""
    import requests

    counts: Counter = Counter()
    offset = None
    while True:
        body: Dict[str, Any] = {
            "limit": 1000,
            "with_payload": ["doc_id"],
            "with_vector": False,
        }
        if offset is not None:
            body["offset"] = offset
        res = requests.post(
            f"{QDRANT_URL}/collections/{DENSE_COLLECTION}/points/scroll",
            json=body,
            timeout=30,
        ).json()["result"]
        for p in res["points"]:
            did = (p.get("payload") or {}).get("doc_id")
            if did:
                counts[did] += 1
        offset = res.get("next_page_offset")
        if offset is None:
            break
    return counts


def _qdrant_sparse_counts() -> Counter:
    """Sparse points per doc_id (parsed from the chunk_id prefix)."""
    import requests

    counts: Counter = Counter()
    offset = None
    while True:
        body: Dict[str, Any] = {
            "limit": 1000,
            "with_payload": ["chunk_id"],
            "with_vector": False,
        }
        if offset is not None:
            body["offset"] = offset
        res = requests.post(
            f"{QDRANT_URL}/collections/{SPARSE_COLLECTION}/points/scroll",
            json=body,
            timeout=30,
        ).json()["result"]
        for p in res["points"]:
            cid = (p.get("payload") or {}).get("chunk_id") or ""
            if "_" in cid:
                counts[cid.split("_", 1)[0]] += 1
        offset = res.get("next_page_offset")
        if offset is None:
            break
    return counts


def _outcome(points_dense: int, header: Optional[Dict[str, Any]]) -> str:
    if points_dense > 0:
        return "INGESTED"
    if header is not None:
        degraded = header.get("extraction_degraded_pages") or 0
        fallback = header.get("extraction_fallback")
        if degraded > 0 or fallback:
            return "LADDER_FAIL"
    return "PENDING"


def build_row(
    pdf: Path,
    header: Optional[Dict[str, Any]],
    points_dense: int,
    points_sparse: int,
) -> Dict[str, Any]:
    doc_id, sha256 = _file_hashes(pdf)
    extraction: Optional[Dict[str, Any]] = None
    pages: Optional[int] = None
    if header is not None:
        pages = header.get("total_pages")
        extraction = {
            "engine": header.get("extraction_engine"),
            "route": header.get("extraction_engine"),
            "schema_version": header.get("schema_version"),
            "engine_version": header.get("pipeline_version"),
            "extracted_at": header.get("ingestion_timestamp"),
        }
    outcome = _outcome(points_dense, header)
    provenance = (
        f"backfill 2026-06-13; extraction from {header.get('_source_jsonl')}"
        if header is not None
        else "backfill 2026-06-13; no extraction output found"
    )
    return {
        "doc_id": doc_id,
        "source_path": str(pdf.relative_to(REPO_ROOT)),
        "sha256": sha256,
        "pages": pages,
        "extraction": extraction,
        "ingest": {
            "collection": DENSE_COLLECTION,
            "points_dense": points_dense,
            "points_sparse": points_sparse,
            "ingested_at": (
                (extraction or {}).get("extracted_at") if outcome == "INGESTED" else None
            ),
            "outcome": outcome,
        },
        "provenance": provenance,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-qdrant", action="store_true", help="skip live Qdrant counts")
    ap.add_argument("--out", default=str(MANIFEST_PATH))
    args = ap.parse_args()

    pdfs = sorted(DATA_DIR.rglob("*.pdf")) + sorted(DATA_DIR.rglob("*.PDF"))
    pdfs = sorted(set(pdfs), key=lambda p: str(p.relative_to(REPO_ROOT)).lower())
    headers = _scan_extraction_headers()

    dense = Counter()
    sparse = Counter()
    if not args.no_qdrant:
        dense = _qdrant_dense_counts()
        sparse = _qdrant_sparse_counts()

    rows: List[Dict[str, Any]] = []
    for pdf in pdfs:
        doc_id, _ = _file_hashes(pdf)
        rows.append(
            build_row(pdf, headers.get(doc_id), dense.get(doc_id, 0), sparse.get(doc_id, 0))
        )

    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    ingested = sum(1 for r in rows if r["ingest"]["outcome"] == "INGESTED")
    ladder = sum(1 for r in rows if r["ingest"]["outcome"] == "LADDER_FAIL")
    pending = sum(1 for r in rows if r["ingest"]["outcome"] == "PENDING")
    print(f"wrote {len(rows)} rows -> {args.out}")
    print(f"  INGESTED={ingested}  LADDER_FAIL={ladder}  PENDING={pending}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
