"""V3 resilient pause-and-poll breaker contract (PLAN_V3.1 Step 2, 2026-06-01).

The strict circuit breaker (Step 1) hard-fails the batch on the first
VlmInfraError - correct for a short attended run, a death sentence for a 25-hour
soak where a transient drop is statistically certain. The resilient breaker
instead polls the endpoint and resumes on recovery, with TWO bounded guards so
it never hangs:

* ``recovery_ceiling_s`` - give up after N seconds of continuous unreachability
  (the "truly dead machine").
* ``max_resume_attempts`` - give up after N infra failures on one doc (the
  "HTTP-up-but-inference-dead" flap, which the health probe cannot detect).

Pins: ``VlmProvider.probe_health``, ``_wait_for_vlm_recovery``, and
``_process_with_resilience`` (the strict/resilient decision logic). Fully
offline/deterministic: no real sleeps, no real network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

from mmrag_v3.engines.vlm_provider import (
    VlmInfraError,
    VlmProvider,
    VlmProviderConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from v3_batch_ingest import (  # noqa: E402
    _process_with_resilience,
    _wait_for_vlm_recovery,
)

_PDF = Path("doc.pdf")


# --------------------------------------------------------------------------
# VlmProvider.probe_health
# --------------------------------------------------------------------------


class _Resp:
    def __init__(self, status_code: int):
        self.status_code = status_code


def _provider() -> VlmProvider:
    return VlmProvider(
        VlmProviderConfig(endpoint="http://node.invalid/v1", model="m", api_key=None)
    )


def test_probe_health_true_on_200(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(200))
    assert _provider().probe_health() is True


def test_probe_health_false_on_transport_error(monkeypatch):
    def boom(*a, **k):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(requests, "get", boom)
    assert _provider().probe_health() is False


def test_probe_health_false_on_non_200(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(503))
    assert _provider().probe_health() is False


# --------------------------------------------------------------------------
# _wait_for_vlm_recovery
# --------------------------------------------------------------------------


def test_wait_returns_true_on_immediate_recovery():
    sleeps = []
    ok = _wait_for_vlm_recovery(
        lambda: True,
        poll_interval_s=60,
        recovery_ceiling_s=600,
        sleep_fn=sleeps.append,
    )
    assert ok is True
    assert sleeps == []  # recovered before any wait


def test_wait_returns_true_when_endpoint_comes_back():
    probes = iter([False, False, True])  # immediate, poll1, poll2
    sleeps = []
    ok = _wait_for_vlm_recovery(
        lambda: next(probes),
        poll_interval_s=60,
        recovery_ceiling_s=600,
        sleep_fn=sleeps.append,
    )
    assert ok is True
    assert len(sleeps) == 2


def test_wait_gives_up_at_ceiling():
    sleeps = []
    ok = _wait_for_vlm_recovery(
        lambda: False,
        poll_interval_s=60,
        recovery_ceiling_s=180,
        sleep_fn=sleeps.append,
    )
    assert ok is False
    assert len(sleeps) == 3  # 60, 120, 180 then stop


# --------------------------------------------------------------------------
# _process_with_resilience
# --------------------------------------------------------------------------


def _proc(fail_times: int):
    """A process_fn that raises VlmInfraError `fail_times` times, then succeeds."""
    state = {"n": 0}

    def fn(pdf: Path):
        if state["n"] < fail_times:
            state["n"] += 1
            raise VlmInfraError("M5 down")
        return {"status": "ok", "src": str(pdf)}

    return fn


def _no_sleep(_):
    return None


def test_success_first_try_never_probes():
    def probe():
        raise AssertionError("probe must not be called when the doc succeeds")

    entry, reason = _process_with_resilience(
        _PDF,
        _proc(0),
        strict=False,
        probe_fn=probe,
        poll_interval_s=1,
        recovery_ceiling_s=10,
        max_resume_attempts=3,
        sleep_fn=_no_sleep,
    )
    assert reason is None
    assert entry["status"] == "ok"


def test_strict_halts_on_first_infra_without_polling():
    def probe():
        raise AssertionError("strict mode must not poll")

    entry, reason = _process_with_resilience(
        _PDF,
        _proc(99),
        strict=True,
        probe_fn=probe,
        poll_interval_s=1,
        recovery_ceiling_s=10,
        max_resume_attempts=3,
        sleep_fn=_no_sleep,
    )
    assert entry is None
    assert "strict breaker" in reason


def test_resilient_resumes_after_recovery():
    entry, reason = _process_with_resilience(
        _PDF,
        _proc(1),  # one transient drop, then succeeds
        strict=False,
        probe_fn=lambda: True,
        poll_interval_s=1,
        recovery_ceiling_s=10,
        max_resume_attempts=3,
        sleep_fn=_no_sleep,
    )
    assert reason is None
    assert entry["status"] == "ok"


def test_resilient_halts_when_endpoint_dead_past_ceiling():
    entry, reason = _process_with_resilience(
        _PDF,
        _proc(99),
        strict=False,
        probe_fn=lambda: False,  # never recovers
        poll_interval_s=60,
        recovery_ceiling_s=120,
        max_resume_attempts=3,
        sleep_fn=_no_sleep,
    )
    assert entry is None
    assert "ceiling" in reason


def test_resilient_halts_on_flap_after_max_attempts():
    # Pathological: probe always says healthy, but inference always fails.
    # The ceiling never trips (probe is True); the resume cap must.
    entry, reason = _process_with_resilience(
        _PDF,
        _proc(99),
        strict=False,
        probe_fn=lambda: True,
        poll_interval_s=1,
        recovery_ceiling_s=10,
        max_resume_attempts=3,
        sleep_fn=_no_sleep,
    )
    assert entry is None
    assert "exceeded 3 resume attempts" in reason
