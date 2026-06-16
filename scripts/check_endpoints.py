#!/usr/bin/env python3
"""Health probe for every configured model endpoint + Qdrant.

One command to see what's alive across the distributed stack (GX10 / M5 /
Mini). Reads the single source of truth in `mmrag_v2.endpoints`, so
it always reflects the real configuration (incl. env overrides).

    python scripts/check_endpoints.py

Exit code 0 if every endpoint is reachable, 1 otherwise (so it can gate a
batch run or be wired into a pre-flight check).
"""
from __future__ import annotations

import json
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmrag_v2.endpoints import all_endpoints  # noqa: E402

QDRANT_URL = "http://localhost:6333"


def _get(url: str, timeout: float = 4.0) -> dict | None:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _tcp_open(url: str, timeout: float = 3.0) -> bool:
    p = urlparse(url)
    port = p.port or (443 if p.scheme == "https" else 80)
    try:
        with socket.create_connection((p.hostname, port), timeout=timeout):
            return True
    except OSError:
        return False


def _probe_models(base_url: str) -> list[str] | None:
    """Return served model ids via OpenAI /models, or None if unavailable."""
    for candidate in (f"{base_url.rstrip('/')}/models", f"{base_url.rstrip('/')}/v1/models"):
        d = _get(candidate)
        if d and isinstance(d.get("data"), list):
            return [m.get("id", "?") for m in d["data"]]
    return None


def main() -> int:
    rows = []
    all_ok = True
    for ep in all_endpoints():
        models = _probe_models(ep.base_url)
        if models is not None:
            served = ",".join(models)
            status = "UP" if ep.model in served or not ep.model else "UP (model not loaded?)"
        elif _tcp_open(ep.base_url):
            status, served = "UP (no /models)", "-"
        else:
            status, served = "DOWN", "-"
            all_ok = False
        rows.append((ep.role, ep.box, ep.base_url, ep.model, status, served))

    # Qdrant
    q = _get(f"{QDRANT_URL}/collections")
    if q and q.get("status") == "ok":
        n = len(q["result"]["collections"])
        rows.append(("qdrant", "Mini", QDRANT_URL, f"{n} collections", "UP", "-"))
    else:
        rows.append(("qdrant", "Mini", QDRANT_URL, "-", "DOWN", "-"))
        all_ok = False

    w = [max(len(str(r[i])) for r in rows) for i in range(5)]
    hdr = ("role", "box", "base_url", "model", "status")
    print("  ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
    print("  ".join("-" * w[i] for i in range(5)))
    for r in rows:
        line = "  ".join(str(r[i]).ljust(w[i]) for i in range(5))
        if len(r) > 5 and r[5] not in ("-", ""):
            line += f"  served={r[5]}"
        print(line)

    print("\nALL_ENDPOINTS_UP" if all_ok else "\nSOME_ENDPOINTS_DOWN")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
