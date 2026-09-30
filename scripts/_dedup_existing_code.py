"""Sweep already-ingested data for within-page duplicate CODE chunks (2026-06-18).

The dedup fix (_dedupe_within_page_text now covers Modality.CODE) only affects new
conversions. This removes the degenerate code duplicates already in production (e.g.
Fluent Python p214: 90 identical best_promo blocks) using the SAME rule -- within one
page, drop later code chunks whose content matches an earlier one exactly OR (>=120
chars) whitespace-normalized; keep the first. Deletes the duplicate points from BOTH
the dense and sparse collections (POST /points/delete) and rewrites the on-disk jsonl.
Conservative: only EXACT / long-normalized within-page matches; never cross-page,
never short snippets. Idempotent.
"""
from __future__ import annotations

import glob
import json
import re
import urllib.request
import uuid

QDRANT = "http://localhost:6333"
DENSE = "mmrag_v3__qwen3_local"
SPARSE = "mmrag_v3__bm25_sparse"
_NS = uuid.UUID("8b7c5e3a-1f4d-4b2a-9c1e-6d8a3f0b9c2e")  # ingest_to_qdrant _POINT_ID_NAMESPACE
_MIN = 120


def _http(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{QDRANT}{path}", data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


def _is_code(pl: dict) -> bool:
    return (pl.get("modality") == "code"
            or str(pl.get("chunk_type", "")).lower() == "code"
            or str(pl.get("content_classification", "")).lower() == "code")


def _page(chunk_id: str):
    m = re.search(r"_(\d+)_", chunk_id or "")
    return m.group(1) if m else None


def duplicate_chunk_ids() -> list:
    """chunk_ids of within-page code duplicates to drop (later occurrences)."""
    # (doc_id, page) -> {exact:set, norm:set}
    seen, drop = {}, []
    off = None
    rows = []
    while True:
        body = {"limit": 1000, "with_payload": ["content", "chunk_id", "doc_id"], "with_vector": False}
        if off:
            body["offset"] = off
        r = _http("POST", f"/collections/{DENSE}/points/scroll", body)["result"]
        rows.extend((p.get("payload") or {}) for p in r["points"])
        off = r.get("next_page_offset")
        if not off:
            break
    # deterministic order: by chunk_id (encodes page + reading-order hash suffix)
    rows.sort(key=lambda pl: str(pl.get("chunk_id") or ""))
    for pl in rows:
        if not _is_code(pl):
            continue
        cid = pl.get("chunk_id")
        key = (pl.get("doc_id"), _page(cid))
        s = (pl.get("content") or "").strip()
        if not s or key[1] is None:
            continue
        e, n = seen.setdefault(key, (set(), set()))
        norm = re.sub(r"\s+", " ", s)
        if s in e or (len(norm) >= _MIN and norm in n):
            drop.append(cid)
            continue
        e.add(s)
        if len(norm) >= _MIN:
            n.add(norm)
    return drop


def delete_points(collection: str, chunk_ids: list) -> int:
    ids = [str(uuid.uuid5(_NS, cid)) for cid in chunk_ids]
    deleted = 0
    for i in range(0, len(ids), 256):
        batch = ids[i:i + 256]
        _http("POST", f"/collections/{collection}/points/delete?wait=true", {"points": batch})
        deleted += len(batch)
    return deleted


def rewrite_on_disk(drop: set) -> int:
    files = glob.glob("output/backlog/*/ingestion.jsonl") + \
        glob.glob("output/phase5_reextract/*/ingestion.jsonl") + \
        ["output/devlin_full_fixb/ingestion.jsonl"]
    removed = 0
    for f in files:
        rows = [json.loads(l) for l in open(f) if l.strip()]
        kept = [d for d in rows if d.get("chunk_id") not in drop]
        if len(kept) != len(rows):
            removed += len(rows) - len(kept)
            with open(f, "w", encoding="utf-8") as fh:
                for d in kept:
                    fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    return removed


def main() -> int:
    drop = duplicate_chunk_ids()
    print(f"within-page duplicate code chunks to drop: {len(drop)}")
    if not drop:
        print("DEDUP_SWEEP_DONE (nothing to do)")
        return 0
    d = delete_points(DENSE, drop)
    s = delete_points(SPARSE, drop)
    disk = rewrite_on_disk(set(drop))
    print(f"deleted dense={d} sparse={s} (sparse absent for non-text-lane) | on-disk removed={disk}")
    print("DEDUP_SWEEP_DONE")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
