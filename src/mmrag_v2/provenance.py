"""Output provenance helpers (PLAN_QUALITY_REMEDIATION_V1 WP-C4).

``config_hash`` makes "was this output produced under the current pipeline configuration" a
comparable value: a SHA-256 of the canonical JSON of the options that change the OUTPUT. It never
includes credentials or endpoints, and the helper refuses option names that look like them, so a
future caller cannot leak a secret into a committed JSONL header by accident.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

_FORBIDDEN_FRAGMENTS = ("key", "secret", "token", "password", "endpoint", "url", "host")


def compute_config_hash(options: Dict[str, Any]) -> str:
    """SHA-256 (hex) of the canonical JSON of ``options``.

    Equal option dictionaries always hash equal (sorted keys, fixed separators); any changed
    value changes the hash. Raises ``ValueError`` for an option name that looks like a credential
    or an endpoint.
    """
    for name in options:
        low = str(name).lower()
        if any(f in low for f in _FORBIDDEN_FRAGMENTS):
            raise ValueError(
                f"config_hash option {name!r} looks like a credential/endpoint; refuse"
            )
    canonical = json.dumps(
        options, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_sha256(path: Union[str, Path]) -> Optional[str]:
    """SHA-256 (hex) of a file, or None when it cannot be read."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 16), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None
