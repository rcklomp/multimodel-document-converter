"""V3 VLM timeout wiring + read-timeout retry policy (2026-06-04).

Two behaviors, both offline/deterministic:

1. ``VLM_NATIVE_TIMEOUT`` env wires the per-call read timeout into the V3
   provider (the hardcoded 180s is too tight for dense interior magazine pages -
   2026-06-03 measurement median ~265s).
2. A *read* timeout gets a dedicated small attempt cap
   (``READ_TIMEOUT_MAX_ATTEMPTS``): the server is generating, just too slowly, so
   retrying repeats the full timeout for a page that is simply too heavy.
   Connect/connection faults (node down, possibly transient) keep the full
   ``max_retries``. All terminal cases still raise ``VlmInfraError`` so the B4
   circuit-breaker contract is unchanged.
"""

from __future__ import annotations

import pytest
import requests

from mmrag_v3.engines.vlm_provider import (
    READ_TIMEOUT_MAX_ATTEMPTS,
    VlmInfraError,
    VlmProvider,
    VlmProviderConfig,
)


def _provider(max_retries: int = 3) -> VlmProvider:
    return VlmProvider(
        VlmProviderConfig(
            endpoint="http://node.invalid/v1",
            model="m",
            api_key=None,
            timeout_seconds=1.0,
            max_retries=max_retries,
            retry_backoff_seconds=0.0,
        )
    )


# --------------------------------------------------------------------------
# 1. VLM_NATIVE_TIMEOUT env wiring
# --------------------------------------------------------------------------


def test_timeout_env_is_wired(monkeypatch):
    monkeypatch.setenv("VLM_NATIVE_ENDPOINT", "http://macbook-pro-m5.lan:8000/v1")
    monkeypatch.setenv("VLM_NATIVE_MODEL", "Qwen3-VL")
    monkeypatch.setenv("VLM_NATIVE_TIMEOUT", "600")
    assert VlmProviderConfig.from_env().timeout_seconds == 600.0


def test_timeout_env_default_and_invalid(monkeypatch):
    monkeypatch.setenv("VLM_NATIVE_ENDPOINT", "http://macbook-pro-m5.lan:8000/v1")
    monkeypatch.setenv("VLM_NATIVE_MODEL", "Qwen3-VL")
    monkeypatch.delenv("VLM_NATIVE_TIMEOUT", raising=False)
    assert VlmProviderConfig.from_env().timeout_seconds == 180.0
    monkeypatch.setenv("VLM_NATIVE_TIMEOUT", "not-a-number")
    assert VlmProviderConfig.from_env().timeout_seconds == 180.0
    # Floored at a sane minimum.
    monkeypatch.setenv("VLM_NATIVE_TIMEOUT", "0")
    assert VlmProviderConfig.from_env().timeout_seconds == 10.0


# --------------------------------------------------------------------------
# 2. Read-timeout attempt cap (vs full retries for connect faults)
# --------------------------------------------------------------------------


def test_read_timeout_is_not_retried(monkeypatch):
    """A ReadTimeout fails after READ_TIMEOUT_MAX_ATTEMPTS, not max_retries."""
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise requests.exceptions.ReadTimeout("read timed out")

    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(VlmInfraError):
        _provider(max_retries=3).describe(b"img", "p")
    # Capped well below max_retries=3 (would have wasted 2 extra full timeouts).
    assert calls["n"] == READ_TIMEOUT_MAX_ATTEMPTS


def test_connection_error_keeps_full_retries(monkeypatch):
    """A connection fault (node down, maybe transient) still uses max_retries."""
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise requests.exceptions.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(VlmInfraError):
        _provider(max_retries=3).describe(b"img", "p")
    assert calls["n"] == 3


def test_connect_timeout_keeps_full_retries(monkeypatch):
    """ConnectTimeout is a connect fault (recoverable), NOT capped as a read timeout."""
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise requests.exceptions.ConnectTimeout("connect timed out")

    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(VlmInfraError):
        _provider(max_retries=3).describe(b"img", "p")
    assert calls["n"] == 3
