"""Unit tests for the omlx tenancy & scheduling scaffold.

Charter: docs/ARCHITECTURE_V3_DRAFT_0.5.md §7.7, §7.6, R6, R16.
"""

from __future__ import annotations

import pytest

from mmrag_v2.omlx.coresidency_monitor import (
    EVICTION_RATE_FORKBACK_THRESHOLD_PER_MIN,
    EVICTION_RATE_WINDOW_S,
    CoresidencyEvent,
    CoresidencyMonitor,
)
from mmrag_v2.omlx.scheduler import (
    QUERY_PATH_QUEUE_DEPTH_LIMIT,
    OmlxScheduler,
    RequestPriority,
    ScheduledRequest,
)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class TestRequestPriority:
    def test_priority_ladder_order(self):
        # Charter §7.7 #1: lower numeric value = higher priority.
        assert RequestPriority.QUERY_TEXT_EMBED < RequestPriority.QUERY_RERANK
        assert RequestPriority.QUERY_RERANK < RequestPriority.QUERY_VISUAL_EMBED
        assert RequestPriority.QUERY_VISUAL_EMBED < RequestPriority.INGEST_VISUAL_EMBED

    def test_int_enum_int_comparison(self):
        assert int(RequestPriority.QUERY_TEXT_EMBED) == 1
        assert int(RequestPriority.INGEST_VISUAL_EMBED) == 4


class TestOmlxScheduler:
    def test_empty_pop_returns_none(self):
        sched = OmlxScheduler()
        assert sched.pop() is None

    def test_higher_priority_pops_first(self):
        sched = OmlxScheduler()
        sched.submit(
            model="colpali",
            priority=RequestPriority.INGEST_VISUAL_EMBED,
            payload="page1",
        )
        sched.submit(
            model="qwen3",
            priority=RequestPriority.QUERY_TEXT_EMBED,
            payload="query1",
        )
        first = sched.pop()
        assert first is not None
        assert first.model == "qwen3"  # higher priority
        assert first.priority == int(RequestPriority.QUERY_TEXT_EMBED)
        second = sched.pop()
        assert second is not None
        assert second.model == "colpali"

    def test_fifo_within_same_priority(self):
        sched = OmlxScheduler()
        for i in range(5):
            sched.submit(
                model="colpali",
                priority=RequestPriority.INGEST_VISUAL_EMBED,
                payload=f"page{i}",
            )
        for i in range(5):
            req = sched.pop()
            assert req is not None
            assert req.payload == f"page{i}"

    def test_queue_depth(self):
        sched = OmlxScheduler()
        assert sched.queue_depth() == 0
        sched.submit(model="m", priority=RequestPriority.QUERY_TEXT_EMBED)
        sched.submit(model="m", priority=RequestPriority.QUERY_RERANK)
        assert sched.queue_depth() == 2
        sched.pop()
        assert sched.queue_depth() == 1

    def test_ingest_default_cancellable(self):
        sched = OmlxScheduler()
        ingest = sched.submit(
            model="colpali",
            priority=RequestPriority.INGEST_VISUAL_EMBED,
        )
        assert ingest.cancellable is True

    def test_query_default_not_cancellable(self):
        sched = OmlxScheduler()
        query = sched.submit(
            model="qwen3",
            priority=RequestPriority.QUERY_TEXT_EMBED,
        )
        assert query.cancellable is False

    def test_queue_depth_limit_constant(self):
        # Charter §7.7 failure mode: queue depth >3 cancels ingest.
        assert QUERY_PATH_QUEUE_DEPTH_LIMIT == 3

    def test_mark_done_decrements_inflight(self):
        sched = OmlxScheduler()
        req = sched.submit(
            model="qwen3", priority=RequestPriority.QUERY_TEXT_EMBED
        )
        popped = sched.pop()
        assert popped is not None
        # Internal counter incremented on pop. mark_done balances.
        sched.mark_done(popped)
        # Should not raise / not go negative
        sched.mark_done(req)  # idempotent-ish (won't go below 0)


# ---------------------------------------------------------------------------
# Co-residency monitor
# ---------------------------------------------------------------------------


class TestCoresidencyMonitor:
    def test_empty_state(self):
        mon = CoresidencyMonitor()
        assert mon.eviction_count() == 0
        assert mon.evictions_per_min(now_s=100.0) == 0.0
        assert mon.is_forkback_triggered(now_s=100.0) is False

    def test_eviction_count(self):
        mon = CoresidencyMonitor()
        mon.record(
            CoresidencyEvent(
                timestamp_s=10.0, event_type="eviction", model="colpali"
            )
        )
        mon.record(
            CoresidencyEvent(
                timestamp_s=15.0, event_type="eviction", model="qwen3"
            )
        )
        assert mon.eviction_count() == 2

    def test_rolling_window_drops_old_events(self):
        mon = CoresidencyMonitor()
        # Two evictions at t=0
        mon.record(CoresidencyEvent(timestamp_s=0.0, event_type="eviction", model="a"))
        mon.record(CoresidencyEvent(timestamp_s=0.0, event_type="eviction", model="b"))
        # At t=10 still inside window
        assert mon.evictions_per_min(now_s=10.0) == 2.0
        # At t=120 outside the 60s window
        assert mon.evictions_per_min(now_s=120.0) == 0.0

    def test_forkback_triggers_above_threshold(self):
        mon = CoresidencyMonitor()
        # 2 evictions per minute > 1/min threshold
        mon.record(CoresidencyEvent(timestamp_s=0.0, event_type="eviction", model="a"))
        mon.record(CoresidencyEvent(timestamp_s=10.0, event_type="eviction", model="b"))
        assert mon.is_forkback_triggered(now_s=20.0) is True

    def test_threshold_constants(self):
        assert EVICTION_RATE_FORKBACK_THRESHOLD_PER_MIN == 1.0
        assert EVICTION_RATE_WINDOW_S == 60.0

    def test_non_eviction_events_not_counted(self):
        mon = CoresidencyMonitor()
        mon.record(
            CoresidencyEvent(
                timestamp_s=0.0, event_type="high_water_mark", model="colpali"
            )
        )
        assert mon.eviction_count() == 0
        assert mon.evictions_per_min(now_s=10.0) == 0.0

    def test_clear(self):
        mon = CoresidencyMonitor()
        mon.record(
            CoresidencyEvent(timestamp_s=0.0, event_type="eviction", model="a")
        )
        mon.clear()
        assert mon.eviction_count() == 0


# ---------------------------------------------------------------------------
# Type sanity checks
# ---------------------------------------------------------------------------


class TestScheduledRequestType:
    def test_dataclass_ordering(self):
        # Lower priority value wins; sequence breaks ties.
        a = ScheduledRequest(priority=1, sequence=10, model="x")
        b = ScheduledRequest(priority=2, sequence=5, model="y")
        c = ScheduledRequest(priority=1, sequence=20, model="z")
        assert a < b
        assert a < c
        assert c < b
