"""omlx co-residency monitoring.

Charter: docs/ARCHITECTURE_V3_DRAFT_0.5.md §7.6, §7.7, R6, R16.

When three models share the omlx server's GPU memory, one is evicted
per LRU if memory exceeded. This module exposes the eviction event
counter that the R6 fork-back trigger consults: if eviction rate
sustains >1/min, the trigger fires.

Foundation-session status: SCHEMA + COUNTERS. The actual eviction
detection requires either (a) omlx server emitting eviction events
to a log file or (b) periodic probing of GPU memory state. Both
land in Phase C task C2.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List


logger = logging.getLogger(__name__)


# Charter §7.6: "if eviction rate >1/min sustained, R6 fork-back triggered"
EVICTION_RATE_FORKBACK_THRESHOLD_PER_MIN = 1.0
EVICTION_RATE_WINDOW_S = 60.0  # rolling 60-second window


@dataclass(frozen=True)
class CoresidencyEvent:
    """A single co-residency telemetry event."""

    timestamp_s: float
    event_type: str  # "eviction" | "high_water_mark" | "request" | ...
    model: str
    detail: str = ""


@dataclass
class CoresidencyMonitor:
    """Rolling-window co-residency telemetry.

    Foundation-session: in-memory only. Phase C wires this to
    `logs/omlx_scheduling_<timestamp>.jsonl` per Charter §7.5
    observability.
    """

    events: List[CoresidencyEvent] = field(default_factory=list)
    _eviction_times: Deque[float] = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, event: CoresidencyEvent) -> None:
        with self._lock:
            self.events.append(event)
            if event.event_type == "eviction":
                self._eviction_times.append(event.timestamp_s)
                self._trim_locked(now_s=event.timestamp_s)

    def _trim_locked(self, *, now_s: float) -> None:
        cutoff = now_s - EVICTION_RATE_WINDOW_S
        while self._eviction_times and self._eviction_times[0] < cutoff:
            self._eviction_times.popleft()

    def evictions_per_min(self, *, now_s: float | None = None) -> float:
        """Eviction rate over the rolling 60-second window."""
        if now_s is None:
            now_s = time.monotonic()
        with self._lock:
            self._trim_locked(now_s=now_s)
            return float(len(self._eviction_times))  # window is 60s

    def is_forkback_triggered(self, *, now_s: float | None = None) -> bool:
        """Charter §7.6: eviction rate >1/min sustained triggers R6 fork-back."""
        return self.evictions_per_min(now_s=now_s) > EVICTION_RATE_FORKBACK_THRESHOLD_PER_MIN

    def eviction_count(self) -> int:
        with self._lock:
            return sum(1 for e in self.events if e.event_type == "eviction")

    def clear(self) -> None:
        with self._lock:
            self.events.clear()
            self._eviction_times.clear()
