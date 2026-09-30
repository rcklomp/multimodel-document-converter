"""Hard wall-clock deadline for per-page extraction calls (engines/_deadline.py).

ROOT-CAUSE regression (2026-06-18): a per-page VLM call hung 3h in
``requests.post`` at ``socket.readinto`` because the mobile server accepted the TCP
connection but never replied, and ``requests``' ``timeout=`` is not a reliable TOTAL
wall-clock bound in that state. ``run_with_deadline`` is the guaranteed cap. The
integration test reproduces the EXACT scenario with a black-hole socket that accepts
connections and never responds, and proves the caller is freed within the deadline
instead of blocking forever.
"""
from __future__ import annotations

import socket
import threading
import time

import pytest

from mmrag_v3.engines._deadline import (
    DeadlineExceeded,
    deadline_seconds,
    run_with_deadline,
)


def test_returns_value_when_fast():
    assert run_with_deadline(lambda: 42, 5, "ok") == 42


def test_propagates_callee_exception():
    with pytest.raises(ZeroDivisionError):
        run_with_deadline(lambda: 1 / 0, 5, "boom")


def test_raises_deadline_when_overrun_and_returns_promptly():
    t0 = time.time()
    with pytest.raises(DeadlineExceeded):
        run_with_deadline(lambda: time.sleep(30), 1.0, "slow")
    # Caller is freed at ~the deadline, NOT after the full 30s sleep.
    assert time.time() - t0 < 5.0


def test_deadline_seconds_resolution(monkeypatch):
    monkeypatch.delenv("EXTRACT_PAGE_DEADLINE_BUFFER", raising=False)
    monkeypatch.delenv("VLM_PAGE_DEADLINE_SECONDS", raising=False)
    # default = request_timeout + 120 buffer
    assert deadline_seconds("VLM_PAGE_DEADLINE_SECONDS", 600.0) == 720.0
    # explicit override wins
    monkeypatch.setenv("VLM_PAGE_DEADLINE_SECONDS", "90")
    assert deadline_seconds("VLM_PAGE_DEADLINE_SECONDS", 600.0) == 90.0


def test_vlm_page_retries_on_stall_then_succeeds(tmp_path, monkeypatch):
    """A stalled VLM page is re-issued on a FRESH connection (the server serves new
    requests instantly); recovery keeps the page on the VLM instead of demoting."""
    import fitz
    from types import SimpleNamespace
    from mmrag_v3.engines import vlm_native as vn

    calls = {"n": 0}

    def fake_rwd(fn, seconds, label="call"):
        calls["n"] += 1
        if calls["n"] <= 2:  # first two attempts stall
            raise DeadlineExceeded("stall")
        return {"elements": []}  # third attempt (fresh connection) succeeds

    monkeypatch.setattr(vn, "run_with_deadline", fake_rwd)
    monkeypatch.setattr(vn.VlmNativeEngine, "_page_from_payload",
                        staticmethod(lambda payload, **k: "PAGE_OK"))
    doc = fitz.open(); doc.new_page()
    eng = SimpleNamespace(_provider=SimpleNamespace(config=SimpleNamespace(timeout_seconds=10)),
                          render_dpi=72)
    out = vn.extract_page_vlm(eng, doc[0], 1)
    doc.close()
    assert out == "PAGE_OK"
    assert calls["n"] == 3  # 2 stalls + 1 success (default 2 retries)


def test_vlm_page_stall_exhausts_retries_then_raises(tmp_path, monkeypatch):
    """If every attempt stalls (genuinely input-specific wedge), the stall surfaces
    after the bounded retries so the caller can demote — never an infinite loop."""
    import fitz
    from types import SimpleNamespace
    from mmrag_v3.engines import vlm_native as vn

    monkeypatch.setenv("VLM_PAGE_STALL_RETRIES", "1")  # 2 attempts total
    calls = {"n": 0}

    def always_stall(fn, seconds, label="call"):
        calls["n"] += 1
        raise DeadlineExceeded("stall")

    monkeypatch.setattr(vn, "run_with_deadline", always_stall)
    doc = fitz.open(); doc.new_page()
    eng = SimpleNamespace(_provider=SimpleNamespace(config=SimpleNamespace(timeout_seconds=10)),
                          render_dpi=72)
    with pytest.raises(DeadlineExceeded):
        vn.extract_page_vlm(eng, doc[0], 1)
    doc.close()
    assert calls["n"] == 2  # 1 retry => 2 attempts, then raise


def test_black_hole_socket_does_not_hang_forever():
    """The exact 3h-hang shape: a server that ACCEPTS the connection but never
    sends a byte. ``requests.post`` would block in readinto; the deadline frees us."""
    requests = pytest.importorskip("requests")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    accepted = []

    def _accept_and_stall():
        try:
            conn, _ = srv.accept()
            accepted.append(conn)  # hold it open, send nothing (black hole)
        except OSError:
            pass

    threading.Thread(target=_accept_and_stall, daemon=True).start()

    t0 = time.time()
    # requests timeout deliberately HUGE (mimics the unreliable 600s cap); the hard
    # deadline of 2s must bound it regardless.
    with pytest.raises(DeadlineExceeded):
        run_with_deadline(
            lambda: requests.post(f"http://127.0.0.1:{port}/v1/x", json={}, timeout=600),
            2.0,
            "black_hole",
        )
    elapsed = time.time() - t0
    assert elapsed < 5.0, f"deadline did not bound the stalled request (took {elapsed:.1f}s)"

    for c in accepted:
        c.close()
    srv.close()
