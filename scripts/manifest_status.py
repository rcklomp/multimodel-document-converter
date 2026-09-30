#!/usr/bin/env python3
"""Report the pending/stale set of ``corpus_manifest.jsonl`` (PLAN_F1 WP-B / D1).

Pure classification (no live deps) + a thin CLI that supplies the only IO: the
current sha256 of each ``data/**`` source file. The reconciliation wave (WP-C)
consumes the pending+stale set as its work-list.

Definitions (Phase 4 re-extraction policy):
  - PENDING: never ingested (no dense points / outcome PENDING), OR ingested
    against an extraction whose provenance is below the production standard
    (engine != ``mineru_qwen_hybrid``, or laddered/degraded pages > 0).
  - STALE  : ingested, but the current source sha256 differs from the manifest
    sha256 (the file on disk changed since extraction).
  - FAILED : a recorded QA failure outcome (CONTENT_FAIL / LADDER_FAIL).
  - CURRENT: ingested, fresh sha256, production-grade extraction.

A doc that is both stale and below-standard is reported STALE (file changed
dominates: it must be re-extracted regardless of engine).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "corpus_manifest.jsonl"

PRODUCTION_ENGINE = "mineru_qwen_hybrid"

STATUS_CURRENT = "CURRENT"
STATUS_PENDING = "PENDING"
STATUS_STALE = "STALE"
STATUS_FAILED = "FAILED"


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def current_sha256(path: Path) -> Optional[str]:
    """sha256 of a source file, or None if it is missing (IO; not unit-tested)."""
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_below_standard(row: Dict[str, Any]) -> bool:
    """True if the row's extraction provenance is below the production grade."""
    ext = row.get("extraction") or {}
    if ext.get("engine") != PRODUCTION_ENGINE:
        return True
    return False


def classify_row(row: Dict[str, Any], current_sha: Optional[str]) -> str:
    """Classify one manifest row. Pure; ``current_sha`` is the live file digest.

    Precedence: FAILED outcome first, then file-changed STALE, then PENDING
    (not ingested or below-standard extraction), else CURRENT.
    """
    outcome = (row.get("ingest") or {}).get("outcome")
    if outcome in ("CONTENT_FAIL", "LADDER_FAIL"):
        return STATUS_FAILED

    ingested = outcome == "INGESTED" and (row.get("ingest") or {}).get("points_dense", 0) > 0

    if ingested and current_sha is not None and current_sha != row.get("sha256"):
        return STATUS_STALE

    if not ingested:
        return STATUS_PENDING

    if _is_below_standard(row):
        return STATUS_PENDING

    return STATUS_CURRENT


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(MANIFEST_PATH))
    ap.add_argument(
        "--no-sha-check",
        action="store_true",
        help="skip re-hashing source files (report ingest-state only)",
    )
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    rows = load_manifest(Path(args.manifest))
    buckets: Dict[str, List[Dict[str, Any]]] = {
        STATUS_CURRENT: [],
        STATUS_PENDING: [],
        STATUS_STALE: [],
        STATUS_FAILED: [],
    }
    for row in rows:
        sha = None
        if not args.no_sha_check:
            sha = current_sha256(REPO_ROOT / row["source_path"])
        status = classify_row(row, sha)
        row["_status"] = status
        buckets[status].append(row)

    if args.json:
        print(json.dumps({k: [r["source_path"] for r in v] for k, v in buckets.items()}, indent=2))
        return 0

    for status in (STATUS_PENDING, STATUS_STALE, STATUS_FAILED, STATUS_CURRENT):
        items = buckets[status]
        print(f"\n=== {status} ({len(items)}) ===")
        for r in sorted(items, key=lambda r: (r.get("pages") or 0)):
            ext = r.get("extraction") or {}
            eng = ext.get("engine") or "NO-EXTRACTION"
            pages = r.get("pages")
            pstr = f"{pages}p" if pages else "?p"
            print(f"  {pstr:>6}  {eng:18s}  {r['source_path']}")

    print(
        f"\nTOTAL {len(rows)}: "
        f"PENDING={len(buckets[STATUS_PENDING])} "
        f"STALE={len(buckets[STATUS_STALE])} "
        f"FAILED={len(buckets[STATUS_FAILED])} "
        f"CURRENT={len(buckets[STATUS_CURRENT])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
