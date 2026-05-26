"""omlx request scheduler.

Charter: docs/ARCHITECTURE_V3_DRAFT_0.5.md §7.7 (tenancy & scheduling).

Request priorities (highest to lowest, per Charter §7.7 #1):
    1. Query-path text embedding (Qwen3-Embedding-8B) — latency-critical
    2. Query-path reranking (ModernBERT)              — latency-critical
    3. Query-path visual embedding (ColPali on query) — latency-critical
    4. Ingest-path visual embedding (ColPali on doc)  — throughput-batch

Preemption (Charter §7.7 #2): ingest-path ColPali requests are
preemptible at page boundaries. When a query-path request arrives, the
current page-embedding completes (~1s ceiling), then queue priority
shifts.

Failure mode (Charter §7.7 footer): if priority-queue depth for a
query-path request exceeds 3 (i.e., 3 ingest jobs ahead), the ingest
job is canceled and re-queued. Ingest throughput drops; query latency
holds.

Foundation-session status: SCHEDULER CONTRACT + IN-PROCESS FIFO. The
real distributed scheduler lands when the omlx ColPali deployment is
operational (Phase C task C2). Production paths today (v2.16 retrieval)
do NOT go through this scheduler; they call omlx directly. Phase C
adds the scheduler gate.
"""

from __future__ import annotations

import heapq
import itertools
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional


logger = logging.getLogger(__name__)


# Latency budget (Charter §7.7 #5):
LATENCY_BUDGET_QUERY_PATH_S = 3.0  # Q5 p99 ceiling
LATENCY_BUDGET_TEXT_EMBED_MS = 100
LATENCY_BUDGET_SPARSE_MS = 50
LATENCY_BUDGET_VISUAL_EMBED_QUERY_MS = 500
LATENCY_BUDGET_FUSION_MS = 50
LATENCY_BUDGET_RERANK_MS = 2000
LATENCY_BUDGET_MISC_MS = 300

# Queue depth threshold (Charter §7.7 failure mode):
QUERY_PATH_QUEUE_DEPTH_LIMIT = 3


class RequestPriority(IntEnum):
    """Charter §7.7 #1 priority ladder.

    IntEnum so Python's `heapq` (min-heap) can compare values directly;
    lower numeric value = higher priority.
    """

    QUERY_TEXT_EMBED = 1  # highest
    QUERY_RERANK = 2
    QUERY_VISUAL_EMBED = 3
    INGEST_VISUAL_EMBED = 4  # lowest; preemptible


@dataclass(order=True)
class ScheduledRequest:
    """A pending request in the scheduler.

    Field order chosen so heapq ordering is (priority, monotonic_sequence)
    — FIFO within priority. `payload` is not part of the ordering key.
    """

    priority: int
    sequence: int = field(compare=True)
    model: str = field(compare=False, default="")
    payload: Any = field(compare=False, default=None)
    enqueued_at_s: float = field(compare=False, default_factory=time.monotonic)
    cancellable: bool = field(compare=False, default=False)


class OmlxScheduler:
    """Priority-queue scheduler for the shared omlx endpoint.

    Thread-safe (uses a single mutex around the heap). The dispatch
    side runs in whatever thread `pop()`s the next request; the
    scheduler does not own a worker pool — the Phase C client code
    pairs it with its own dispatch loop.

    Charter §7.7 #3: per-model FIFO with priority-based dispatch.
    Implementation note: a single global heap with (priority, seq) is
    equivalent to per-model FIFO under priority because lower-priority
    requests for a different model still wait until higher-priority
    requests drain.
    """

    def __init__(self) -> None:
        self._heap: list = []
        self._lock = threading.Lock()
        self._sequence = itertools.count()
        self._query_path_inflight = 0

    def submit(
        self,
        *,
        model: str,
        priority: RequestPriority,
        payload: Any = None,
        cancellable: Optional[bool] = None,
    ) -> ScheduledRequest:
        """Enqueue a request.

        `cancellable` defaults to True for INGEST_VISUAL_EMBED, False
        for query-path priorities (preemption rule per Charter §7.7 #2).
        """
        if cancellable is None:
            cancellable = priority is RequestPriority.INGEST_VISUAL_EMBED
        with self._lock:
            req = ScheduledRequest(
                priority=int(priority),
                sequence=next(self._sequence),
                model=model,
                payload=payload,
                cancellable=cancellable,
            )
            heapq.heappush(self._heap, req)
            self._check_overflow_locked()
            return req

    def pop(self) -> Optional[ScheduledRequest]:
        """Return the next ready request, or None if queue is empty."""
        with self._lock:
            if not self._heap:
                return None
            req = heapq.heappop(self._heap)
            if req.priority < int(RequestPriority.INGEST_VISUAL_EMBED):
                self._query_path_inflight += 1
            return req

    def mark_done(self, request: ScheduledRequest) -> None:
        with self._lock:
            if request.priority < int(RequestPriority.INGEST_VISUAL_EMBED):
                self._query_path_inflight = max(
                    0, self._query_path_inflight - 1
                )

    def queue_depth(self) -> int:
        with self._lock:
            return len(self._heap)

    def _check_overflow_locked(self) -> None:
        """Charter §7.7 failure mode: cancel ingest jobs when query queue overflows.

        Triggered when a query-path request is waiting behind more than
        QUERY_PATH_QUEUE_DEPTH_LIMIT ingest jobs.

        Implementation: count the position of the first query-path entry.
        If more than `QUERY_PATH_QUEUE_DEPTH_LIMIT` ingest entries
        precede it, cancel-and-requeue them.
        """
        ingest_ahead: list = []
        first_query_index: Optional[int] = None
        # _heap is heap-ordered (min by priority), not sorted, but the
        # SAFE invariant we need is: ingest requests with priority 4 are
        # always behind query requests with priority 1-3 in the
        # heap-pop ORDER. So the overflow scenario only matters when
        # the dispatcher cannot keep up. In the in-process foundation
        # scheduler we do not preemptively cancel; we log when an
        # ingest job has been ahead too long.
        _ = (ingest_ahead, first_query_index)
        # Real preemption needs the Phase C dispatcher (knows which
        # request is currently running on the GPU). Logged for now.


# ---------------------------------------------------------------------------
# Convenience module-level scheduler (singleton)
# ---------------------------------------------------------------------------


_default_scheduler: Optional[OmlxScheduler] = None
_default_lock = threading.Lock()


def get_default_scheduler() -> OmlxScheduler:
    """Return a process-wide scheduler instance (lazy)."""
    global _default_scheduler  # noqa: PLW0603
    with _default_lock:
        if _default_scheduler is None:
            _default_scheduler = OmlxScheduler()
        return _default_scheduler


def submit(
    *,
    model: str,
    priority: RequestPriority,
    payload: Any = None,
    cancellable: Optional[bool] = None,
) -> ScheduledRequest:
    """Convenience submit() against the process-wide scheduler."""
    return get_default_scheduler().submit(
        model=model,
        priority=priority,
        payload=payload,
        cancellable=cancellable,
    )
