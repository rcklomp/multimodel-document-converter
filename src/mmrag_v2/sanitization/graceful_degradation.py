"""Endpoint-unreachable fallback policy + sentinel-chunk accounting.

Charter: docs/ARCHITECTURE_V3_DRAFT_0.5.md §3.3 (graceful degradation) +
§7.6 (failure-mode behavior).

Behavior (Charter §3.3 / §7.6):
    1. Log `[SANITIZE_SKIPPED: endpoint unreachable]` sentinel per chunk.
    2. Fall back to heuristic sanitization (dual-write retained per
       Charter Phase B).
    3. Emit build-level warning with unreachable-chunk count.
    4. Do NOT hard-fail the build (LLM sanitizer is not the only path).

Sentinel rate accounting (Charter §3.3): >5% sentinel rate in any soak
run marks the run LLM_SENTINEL_DEGRADED and excludes it from the
dominance-criterion "two consecutive soak" confirmation. The sentinel
counter lives in this module so any path emitting a sentinel updates
the same bookkeeping.

Foundation-session status: FUNCTIONAL. The reachability check uses a
short TCP connect (no LLM call); production replaces it with the omlx
scheduler's health endpoint per Charter §7.7.
"""

from __future__ import annotations

import logging
import socket
from dataclasses import dataclass, field
from typing import List
from urllib.parse import urlparse


logger = logging.getLogger(__name__)


SENTINEL_RATE_DEGRADED_THRESHOLD = 0.05  # 5% per Charter §3.3
DEFAULT_REACHABILITY_TIMEOUT_S = 2.0


@dataclass
class SentinelAccount:
    """Build-level sentinel-chunk accounting (Charter §3.3)."""

    total_chunks: int = 0
    sentinel_chunks: List[str] = field(default_factory=list)
    # chunk_id of each chunk that got a "skipped:endpoint_unreachable"
    # status. Per-chunk reason can be added in Phase B if needed.

    @property
    def sentinel_count(self) -> int:
        return len(self.sentinel_chunks)

    @property
    def sentinel_rate(self) -> float:
        if self.total_chunks == 0:
            return 0.0
        return self.sentinel_count / self.total_chunks

    @property
    def is_degraded(self) -> bool:
        """Charter §3.3: >5% sentinel rate marks soak LLM_SENTINEL_DEGRADED."""
        return self.sentinel_rate > SENTINEL_RATE_DEGRADED_THRESHOLD

    def record_chunk(self, chunk_id: str, sentinel: bool) -> None:
        self.total_chunks += 1
        if sentinel:
            self.sentinel_chunks.append(chunk_id)
            logger.warning(
                "[SANITIZE_SKIPPED: endpoint unreachable] chunk_id=%s "
                "(sentinel %d/%d, rate %.3f)",
                chunk_id,
                self.sentinel_count,
                self.total_chunks,
                self.sentinel_rate,
            )

    def soak_marker(self) -> str:
        """Per Charter §3.3: returns the soak marker string to record."""
        if self.is_degraded:
            return "LLM_SENTINEL_DEGRADED"
        return "LLM_OK"


def is_endpoint_reachable(
    endpoint_url: str,
    *,
    timeout_s: float = DEFAULT_REACHABILITY_TIMEOUT_S,
) -> bool:
    """TCP-level reachability check for the GX10 (or other) endpoint.

    Conservative: returns False on any socket error rather than crashing.
    Phase B may replace with an omlx scheduler health-endpoint probe
    that distinguishes "down" from "overloaded" per Charter §7.7.
    """
    try:
        parsed = urlparse(endpoint_url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if not host:
            return False
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except (OSError, ValueError) as exc:
        logger.info(
            "Endpoint reachability probe failed for %s: %s",
            endpoint_url,
            exc,
        )
        return False
