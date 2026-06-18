"""Hard wall-clock deadline for a blocking call.

ROOT CAUSE (2026-06-18): a per-page VLM extraction hung for 3h in
``requests.post`` at ``socket.readinto`` (reading the HTTP status line) when the
mobile M5 server accepted the TCP connection but never sent a response. The
``requests`` ``timeout=`` parameter is a per-socket-operation timeout, NOT a total
wall-clock bound, and in that stalled-socket state it never fired — so neither the
600s read-timeout nor the fail-closed ladder engaged, and the conversion blocked
indefinitely.

``run_with_deadline`` runs ``fn`` in a daemon thread and ABANDONS it if it overruns
the deadline, raising ``DeadlineExceeded`` in the caller. The leaked worker thread
is a daemon (it dies with the process) and its blocked socket eventually unwinds on
the OS/connection timeout; meanwhile the caller is freed to fail-closed. This is the
ONLY reliable total bound when the underlying client's timeout cannot be trusted.
"""
from __future__ import annotations

import os
import threading
from typing import Callable, TypeVar

T = TypeVar("T")


class DeadlineExceeded(Exception):
    """A bounded call overran its hard wall-clock deadline (treat as a per-page
    failure: demote/fall back; NOT a transport circuit-breaker halt)."""


def run_with_deadline(fn: "Callable[[], T]", seconds: float, label: str = "call") -> T:
    """Run ``fn`` with a hard wall-clock cap. Raise ``DeadlineExceeded`` on overrun.

    Re-raises any exception ``fn`` itself throws (so existing error handling is
    preserved). The worker is a daemon thread, so an abandoned (still-blocked) call
    never keeps the process alive.
    """
    box: dict = {}

    def _target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 — ferry to the caller thread
            box["error"] = exc

    worker = threading.Thread(target=_target, name=f"deadline-{label}", daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise DeadlineExceeded(
            f"{label} exceeded {seconds:.0f}s hard wall-clock deadline "
            f"(server accepted the request but did not respond)"
        )
    if "error" in box:
        raise box["error"]
    return box["value"]


def deadline_seconds(env_var: str, request_timeout: float, default_buffer: float = 120.0) -> float:
    """Resolve a per-call deadline: ``request_timeout + buffer``, or an explicit
    override via ``env_var`` (seconds). The buffer keeps the deadline ABOVE the
    client's own timeout so a legitimately slow page is bounded by the client first;
    the deadline only catches the pathological case where that timeout never fires."""
    raw = (os.environ.get(env_var) or "").strip()
    if raw:
        try:
            return max(30.0, float(raw))
        except ValueError:
            pass
    buf = default_buffer
    raw_buf = (os.environ.get("EXTRACT_PAGE_DEADLINE_BUFFER") or "").strip()
    if raw_buf:
        try:
            buf = max(0.0, float(raw_buf))
        except ValueError:
            pass
    return float(request_timeout) + buf
