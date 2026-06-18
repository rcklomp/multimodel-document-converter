"""Targeted re-clean of furniture in already-ingested code chunks (2026-06-18).

The forward fix (uir_chunker._strip_code_furniture) only affects new conversions.
This cleans the ~120 already-ingested code chunks IN PLACE, safely: it rewrites the
stored payload `content` (what retrieval returns to the model) via Qdrant set-payload
in BOTH the dense and sparse collections, and rewrites the on-disk jsonl files so a
future sparse rebuild / re-ingest stays consistent. Vectors are left as-is on purpose
-- the furniture is ~one caption/header line out of a multi-line code chunk, so its
effect on ranking is negligible; what the model SEES is now clean. No re-embed, no
deletes (set-payload only). Idempotent.
"""
from __future__ import annotations

import glob
import json
import urllib.request

from mmrag_v2.chunking.uir_chunker import _strip_code_furniture

QDRANT = "http://localhost:6333"
DENSE = "mmrag_v3__qwen3_local"
SPARSE = "mmrag_v3__bm25_sparse"


def _http(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{QDRANT}{path}", data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


def _is_code(pl: dict) -> bool:
    return (str(pl.get("chunk_type", "")).lower() == "code"
            or pl.get("modality") == "code"
            or str(pl.get("content_classification", "")).lower() == "code")


def affected_from_dense() -> dict:
    """chunk_id -> cleaned_content for every code chunk the stripper would change."""
    out, off = {}, None
    while True:
        body = {"limit": 1000, "with_payload": True, "with_vector": False}
        if off:
            body["offset"] = off
        r = _http("POST", f"/collections/{DENSE}/points/scroll", body)["result"]
        for p in r["points"]:
            pl = p.get("payload") or {}
            if not _is_code(pl):
                continue
            c = pl.get("content") or ""
            cleaned = _strip_code_furniture(c)
            if cleaned != c:
                cid = pl.get("chunk_id")
                if cid:
                    out[cid] = cleaned
        off = r.get("next_page_offset")
        if not off:
            break
    return out


def set_payload_content(collection: str, chunk_id: str, cleaned: str) -> int:
    """Set content for a chunk_id; returns points updated (0 if absent in collection)."""
    body = {"payload": {"content": cleaned},
            "filter": {"must": [{"key": "chunk_id", "match": {"value": chunk_id}}]}}
    _http("POST", f"/collections/{collection}/points/payload?wait=true", body)
    cnt = _http("POST", f"/collections/{collection}/points/count",
                {"filter": {"must": [{"key": "chunk_id", "match": {"value": chunk_id}}]}})
    return cnt["result"]["count"]


def rewrite_on_disk(affected: dict) -> int:
    """Apply the stripper to code chunks in the on-disk jsonl (backlog + reextract)."""
    files = glob.glob("output/backlog/*/ingestion.jsonl") + \
        glob.glob("output/phase5_reextract/*/ingestion.jsonl") + \
        ["output/devlin_full_fixb/ingestion.jsonl"]
    changed = 0
    for f in files:
        rows = [json.loads(l) for l in open(f) if l.strip()]
        dirty = False
        for d in rows:
            m = d.get("metadata") or {}
            if not _is_code({"chunk_type": m.get("chunk_type"), "modality": m.get("modality") or d.get("modality"),
                             "content_classification": m.get("content_classification")}):
                continue
            c = d.get("content") or ""
            cleaned = _strip_code_furniture(c)
            if cleaned != c:
                d["content"] = cleaned
                dirty = True
                changed += 1
        if dirty:
            with open(f, "w", encoding="utf-8") as fh:
                for d in rows:
                    fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    return changed


def main() -> int:
    affected = affected_from_dense()
    print(f"affected code chunks (dense): {len(affected)}")
    d_upd = s_upd = 0
    for cid, cleaned in affected.items():
        d_upd += 1 if set_payload_content(DENSE, cid, cleaned) else 0
        s_upd += 1 if set_payload_content(SPARSE, cid, cleaned) else 0
    print(f"dense payloads updated: {d_upd} | sparse payloads updated: {s_upd}")
    disk = rewrite_on_disk(affected)
    print(f"on-disk jsonl code-chunk contents rewritten: {disk}")
    print("RECLEAN_DONE")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
