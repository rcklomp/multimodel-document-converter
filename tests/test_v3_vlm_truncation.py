"""V3 VLM truncation contract (Charter Blocker A / A1, 2026-06-03).

Executable guard for the dense-page failure mode: a VLM that hits its
output-token ceiling returns HTTP 200 with non-empty but TRUNCATED content
and ``finish_reason == "length"``. The pre-A1 provider returned that body as
if it were a success; downstream ``json.loads`` then failed silently and the
page was mass-demoted to Docling (Combat Aircraft: ~25/43 pages).

A1 makes truncation TYPED, not silent:
  1. ``finish_reason == "length"`` is detected even with non-empty content.
  2. The output budget is escalated ONCE and the call retried.
  3. If still truncated, a :class:`VlmTruncationError` is raised carrying the
     longest partial body, for the bounded JSON-repair stage (A4) to salvage.

Every payload here is a DENSE page (a large multi-element UIR object cut
mid-array) - a pass on a trivial 1-element body would prove nothing about the
real failure mode. Fully offline/deterministic: ``requests.post`` is mocked.
"""

from __future__ import annotations

import json

import pytest
import requests

from mmrag_v3.engines.vlm_provider import (
    TRUNCATION_ESCALATION_CAP,
    VlmInfraError,
    VlmProvider,
    VlmProviderConfig,
    VlmTruncationError,
)


def _provider(max_completion_tokens: int = 4096) -> VlmProvider:
    return VlmProvider(
        VlmProviderConfig(
            endpoint="http://node.invalid/v1",
            model="test-model",
            api_key=None,
            timeout_seconds=1.0,
            max_retries=3,
            retry_backoff_seconds=0.0,
            max_completion_tokens=max_completion_tokens,
        )
    )


_HEAD = (
    '{"page_number": 1, "width": 1654, "height": 2339, '
    '"classification": "digital", "elements": ['
)


def _complete_elements(n: int = 40) -> list:
    return [
        json.dumps(
            {
                "type": "text",
                "content": f"Paragraph {i}: " + ("dense body text " * 12).strip(),
                "bbox": [80, 100 + i * 30, 1500, 128 + i * 30],
                "confidence": 0.95,
                "source_label": "paragraph",
            }
        )
        for i in range(n)
    ]


def _dense_truncated_body() -> str:
    """A large UIR page object cut off mid-``content`` of a late element.

    Mirrors the real failure: many complete elements, then a partial trailing
    one with an unterminated string - exactly what an output-token ceiling
    produces on a dense magazine/manual page.
    """
    partial = '{"type": "text", "content": "Paragraph 40: this final paragraph was cut off mid-sen'
    return _HEAD + ", ".join(_complete_elements()) + ", " + partial


def _dense_full_body() -> str:
    """The same dense page, but a syntactically complete UIR object."""
    return _HEAD + ", ".join(_complete_elements()) + "]}"


class _Resp:
    def __init__(self, status_code: int, body=None, text: str = ""):
        self.status_code = status_code
        self._body = body
        self.text = text

    def json(self):
        if self._body is None:
            raise ValueError("no json body")
        return self._body


def _length_body(content: str) -> dict:
    return {"choices": [{"message": {"content": content}, "finish_reason": "length"}]}


def _stop_body(content: str) -> dict:
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}


def test_persistent_truncation_raises_typed_with_partial(monkeypatch):
    """A dense page truncated on every attempt -> VlmTruncationError + partial."""
    dense = _dense_truncated_body()
    seen_budgets = []

    def fake_post(url, json=None, headers=None, timeout=None):
        seen_budgets.append(json["max_tokens"])
        return _Resp(200, body=_length_body(dense))

    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(VlmTruncationError) as ei:
        _provider(max_completion_tokens=4096).describe(b"img", "prompt")

    # Typed truncation (NOT silent json.loads failure, NOT infra).
    assert not isinstance(ei.value, VlmInfraError)
    # The full partial body is retained for A4 repair.
    assert ei.value.partial_content == dense
    assert ei.value.finish_reason == "length"
    # The budget was escalated exactly once: second call carries a larger
    # max_tokens than the first, capped at the escalation ceiling.
    assert len(seen_budgets) >= 2
    assert seen_budgets[0] == 4096
    assert seen_budgets[1] == 8192
    assert all(b <= TRUNCATION_ESCALATION_CAP for b in seen_budgets)


def test_truncation_recovers_on_escalated_retry(monkeypatch):
    """First attempt truncates; the escalated retry completes -> full content."""
    partial = _dense_truncated_body()
    full = _dense_full_body()
    # Sanity: the "full" body is valid JSON the caller can parse.
    json.loads(full)

    calls = {"n": 0}
    seen_budgets = []

    def fake_post(url, json=None, headers=None, timeout=None):
        seen_budgets.append(json["max_tokens"])
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp(200, body=_length_body(partial))
        return _Resp(200, body=_stop_body(full))

    monkeypatch.setattr(requests, "post", fake_post)

    result = _provider(max_completion_tokens=4096).describe(b"img", "prompt")

    assert result == full
    assert seen_budgets[0] == 4096
    assert seen_budgets[1] == 8192  # escalated before the successful retry


def test_complete_response_does_not_escalate(monkeypatch):
    """A non-truncated dense page returns immediately, no escalation retry."""
    full = _dense_full_body()
    json.loads(full)

    calls = {"n": 0}
    seen_budgets = []

    def fake_post(url, json=None, headers=None, timeout=None):
        seen_budgets.append(json["max_tokens"])
        calls["n"] += 1
        return _Resp(200, body=_stop_body(full))

    monkeypatch.setattr(requests, "post", fake_post)

    result = _provider(max_completion_tokens=4096).describe(b"img", "prompt")

    assert result == full
    assert calls["n"] == 1
    assert seen_budgets == [4096]


def test_escalation_respects_cap(monkeypatch):
    """Escalation never exceeds the OOM-safety ceiling even from a high base."""
    dense = _dense_truncated_body()
    seen_budgets = []

    def fake_post(url, json=None, headers=None, timeout=None):
        seen_budgets.append(json["max_tokens"])
        return _Resp(200, body=_length_body(dense))

    monkeypatch.setattr(requests, "post", fake_post)

    # Base already at the cap: 2x would overflow, so it must stay clamped.
    with pytest.raises(VlmTruncationError):
        _provider(max_completion_tokens=TRUNCATION_ESCALATION_CAP).describe(b"img", "prompt")

    assert all(b <= TRUNCATION_ESCALATION_CAP for b in seen_budgets)
