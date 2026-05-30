#!/usr/bin/env python3
"""Translate V3 IngestionChunk JSONL → V2-shaped JSONL.

The existing ``scripts/ingest_to_qdrant.py`` expects the v2.x schema
(``modality`` at top level, ``metadata.page_number``, ``metadata.spatial``,
optional ``hierarchy.breadcrumb_path``). V3 chunks emit ``element_type``
+ top-level ``page_number``. Rather than fork the ingester, this script
maps V3 fields into the v2 shape on disk.

Usage:
    python scripts/v3_to_v2_jsonl.py \\
        --in-dir  output/v3_baselines \\
        --out-dir output/v3_baselines_v2shape

Produces ``<out-dir>/<rel-path>/ingestion.jsonl`` for every V3 baseline
discovered. ``meta.json`` files are passed through unchanged for
traceability.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent


def _translate_chunk(v3: Dict[str, Any]) -> Dict[str, Any]:
    """Map one V3 IngestionChunk dict → one V2-shaped chunk dict."""
    element_type = (v3.get("element_type") or "text").lower()
    # v2 uses ``modality`` at top level and a redundant ``chunk_type`` inside
    # metadata. Map both consistently.
    modality = element_type
    bbox = v3.get("bbox")
    page_number = int(v3.get("page_number") or 0)
    source_file = v3.get("source_file") or ""
    file_type = v3.get("file_type") or "pdf"
    extraction_method = v3.get("extraction_method") or "vlm"
    v3_meta = v3.get("metadata") or {}

    spatial: Dict[str, Any] = {}
    if isinstance(bbox, list) and len(bbox) == 4:
        spatial = {
            "bbox": [int(c) for c in bbox],
            "bbox_units": "uir_normalized_1000",
        }

    hierarchy: Dict[str, Any] = {}
    parent_heading = v3_meta.get("parent_heading")
    if parent_heading:
        hierarchy["breadcrumb_path"] = [str(parent_heading)]

    return {
        "chunk_id": v3.get("chunk_id"),
        "doc_id": v3.get("doc_id"),
        "source_file": source_file,
        # v2 ingest reads ``modality`` at top level for the embedder routing.
        "modality": modality,
        "content": v3.get("content") or "",
        # v2 schema version is whatever the producer claims; passing V3's
        # value through is honest. ``ingest_to_qdrant.py`` does not gate on
        # this field.
        "schema_version": v3.get("schema_version") or "3.0.0-alpha-v2shape",
        "metadata": {
            "source_file": source_file,
            "file_type": file_type,
            "page_number": page_number,
            "chunk_type": modality,
            "hierarchy": hierarchy,
            "spatial": spatial,
            "extraction_method": extraction_method,
            "confidence": float(v3.get("confidence") or 0.0),
            "v3_reading_order": v3_meta.get("reading_order"),
        },
    }


def _convert_one(jsonl_in: Path, jsonl_out: Path) -> int:
    jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with jsonl_in.open("r", encoding="utf-8") as fin, jsonl_out.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            v3 = json.loads(line)
            # Skip any stray ingestion_metadata header line if present
            # (V3 normally has none, but be defensive).
            if v3.get("object_type") == "ingestion_metadata":
                continue
            v2 = _translate_chunk(v3)
            fout.write(json.dumps(v2))
            fout.write("\n")
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in-dir",
        type=Path,
        default=REPO_ROOT / "output" / "v3_baselines",
        help="Root of V3 baseline outputs (default: output/v3_baselines/)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "output" / "v3_baselines_v2shape",
        help="Where to write V2-shaped JSONLs (default: output/v3_baselines_v2shape/)",
    )
    args = parser.parse_args(argv)
    in_root: Path = args.in_dir.resolve()
    out_root: Path = args.out_dir.resolve()
    if not in_root.exists():
        print(f"in-dir does not exist: {in_root}", file=sys.stderr)
        return 2

    jsonls = sorted(in_root.rglob("ingestion.jsonl"))
    if not jsonls:
        print(f"no ingestion.jsonl found under {in_root}", file=sys.stderr)
        return 2

    total_chunks = 0
    total_docs = 0
    for jsonl_in in jsonls:
        rel = jsonl_in.relative_to(in_root)
        jsonl_out = out_root / rel
        n = _convert_one(jsonl_in, jsonl_out)
        total_chunks += n
        total_docs += 1
        # Pass meta.json through untouched.
        meta_in = jsonl_in.parent / "meta.json"
        if meta_in.exists():
            shutil.copy2(meta_in, jsonl_out.parent / "meta.json")
        print(f"  {rel.parent}: {n} chunks")
    print(f"converted {total_chunks} chunks across {total_docs} docs → {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
