#!/usr/bin/env python3
"""Pre-mutation Qdrant snapshots (PLAN_F1 WP-B / D3).

Creates a Qdrant snapshot for each target collection and downloads it to
``/Users/Shared/qdrant-backups/<UTC-date>/`` so a reconciliation wave can roll
back if an ingest corrupts a collection. ``snapshot_collections`` is importable
so WP-C calls it BEFORE the first mutation of the night.

Snapshots are created via ``POST /collections/{c}/snapshots`` and pulled to the
local disk via ``GET /collections/{c}/snapshots/{name}`` (no docker access
needed). Idempotent per call (Qdrant names snapshots by timestamp); existing
local files are not overwritten.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

DEFAULT_COLLECTIONS = ["mmrag_v3__qwen3_local", "mmrag_v3__bm25_sparse"]
DEFAULT_BACKUP_ROOT = Path("/Users/Shared/qdrant-backups")
QDRANT_URL = "http://localhost:6333"


def _utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def snapshot_collections(
    collections: Optional[List[str]] = None,
    backup_root: Path = DEFAULT_BACKUP_ROOT,
    qdrant_url: str = QDRANT_URL,
) -> List[Path]:
    """Snapshot each collection and download it under ``backup_root/<UTC-date>/``.

    Returns the list of local snapshot file paths written. Raises on a Qdrant
    error (fail-loud: a failed snapshot must block the mutation that depends on
    it, per the D3 contract).
    """
    import requests

    collections = collections or DEFAULT_COLLECTIONS
    dest_dir = backup_root / _utc_date()
    dest_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for coll in collections:
        resp = requests.post(f"{qdrant_url}/collections/{coll}/snapshots", timeout=300)
        resp.raise_for_status()
        name = resp.json()["result"]["name"]

        local = dest_dir / f"{coll}__{name}"
        if not local.exists():
            dl = requests.get(
                f"{qdrant_url}/collections/{coll}/snapshots/{name}",
                timeout=600,
                stream=True,
            )
            dl.raise_for_status()
            with open(local, "wb") as f:
                for chunk in dl.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        written.append(local)
        print(f"snapshot: {coll} -> {local} ({local.stat().st_size} bytes)")

    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collections", nargs="*", default=DEFAULT_COLLECTIONS)
    ap.add_argument("--backup-root", default=str(DEFAULT_BACKUP_ROOT))
    ap.add_argument("--qdrant-url", default=QDRANT_URL)
    args = ap.parse_args()

    paths = snapshot_collections(
        collections=args.collections,
        backup_root=Path(args.backup_root),
        qdrant_url=args.qdrant_url,
    )
    print(f"\n{len(paths)} snapshot(s) written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
