"""Phase 0.5 retry-before-fallback on the MinerU engine.

PLAN_EXTRACTION_FIDELITY_V1 Sections 5.1 + 7 Phase 0.5. The MinerU server's
intermittent ``broadcast_shapes`` 500 (fcd4207: "same page 500s on one call and
returns 1706 chars on the next") is a transient, RETRYABLE fault. Before this
wrapper the engine called ``two_step_extract`` once and dropped straight to the
cross-engine fail-closed ladder. These tests pin the contract:

  * a transient 500 is RETRIED in-engine and recovers WITHOUT engaging the
    ladder (engine stamp stays ``mineru``, ``extraction_fallback`` is None);
  * N consecutive 500s still exhaust the bounded attempts and fall THROUGH to
    the ladder (fail-closed preserved);
  * retry fires ONLY on transient classes (timeout / connection / 5xx), never on
    a 4xx;
  * the attempt cap + backoff mirror the vlm_provider policy.

All offline: a fake MinerU client raises a controlled exception sequence; no
network, no model, no real server.
"""

from __future__ import annotations

import fitz
import pytest

from mmrag_v2.universal.intermediate import UniversalDocument
from mmrag_v3 import processor
from mmrag_v3.engines import mineru_native
from mmrag_v3.engines.mineru_native import (
    MINERU_MAX_RETRIES,
    MineruNativeEngine,
    _mineru_retry_classification,
)


class _ServerError(RuntimeError):
    """Mirrors mineru_vl_utils ServerError: a RuntimeError whose message carries
    the HTTP status, matching the real http-client surface
    (``"Unexpected status code: [500], response body: ..."``)."""


def _status_error(status: int, body: str = "") -> _ServerError:
    return _ServerError(f"Unexpected status code: [{status}], response body: {body}")


class _SequencedClient:
    """Fake MinerUClient: each call raises or returns the next scripted action.

    An action is either an Exception instance (raised) or an element list
    (returned). A single trailing action repeats for every page of a multi-page
    doc so a per-page driver does not run off the end.
    """

    def __init__(self, actions):
        self._actions = list(actions)
        self.calls = 0

    def two_step_extract(self, image):
        self.calls += 1
        action = self._actions[min(self.calls - 1, len(self._actions) - 1)]
        if isinstance(action, BaseException):
            raise action
        return action


_GOOD_PAGE = [{"type": "text", "bbox": [0.1, 0.1, 0.9, 0.2], "content": "real body text"}]


def _make_pdf(path, n_pages=1):
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), f"page {i + 1} native text layer")
    doc.save(str(path))
    doc.close()


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Never burn real backoff seconds in the test suite."""
    monkeypatch.setattr(mineru_native.time, "sleep", lambda *_a, **_k: None)


# --------------------------------------------------------------------------- #
# classification (bounded: transient-only, never 4xx)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("status", sorted({408, 429, 500, 502, 503, 504}))
def test_5xx_and_transient_statuses_are_retryable(status):
    retryable, is_read_timeout = _mineru_retry_classification(_status_error(status))
    assert retryable is True
    assert is_read_timeout is False


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422])
def test_4xx_statuses_are_not_retryable(status):
    retryable, _ = _mineru_retry_classification(_status_error(status))
    assert retryable is False


def test_broadcast_shapes_500_is_retryable():
    # The exact fault class fcd4207 routes around.
    exc = _status_error(500, "broadcast_shapes error in batch predict")
    retryable, _ = _mineru_retry_classification(exc)
    assert retryable is True


def test_connection_failure_is_retryable():
    retryable, is_read_timeout = _mineru_retry_classification(
        _ServerError("Failed to connect to server http://m5:8000. Is it running?")
    )
    assert retryable is True
    assert is_read_timeout is False


def test_read_timeout_is_retryable_and_flagged():
    class _ReadTimeout(Exception):
        pass

    retryable, is_read_timeout = _mineru_retry_classification(_ReadTimeout("read timed out"))
    assert retryable is True
    assert is_read_timeout is True


def test_unknown_error_shape_is_not_retried():
    # A logic bug (KeyError, ValueError with no status) must NOT be masked by retry.
    retryable, _ = _mineru_retry_classification(KeyError("content"))
    assert retryable is False


# --------------------------------------------------------------------------- #
# engine-level: retry mechanics + attempt cap
# --------------------------------------------------------------------------- #
def test_transient_500_then_success_recovers_in_engine():
    client = _SequencedClient([_status_error(500, "broadcast_shapes"), _GOOD_PAGE])
    engine = MineruNativeEngine(client=client)
    out = engine.two_step_extract(object())  # image arg is opaque to the fake
    assert out == _GOOD_PAGE
    assert client.calls == 2  # one failure + one recovery


def test_n_consecutive_500s_exhaust_attempt_cap_then_raise():
    client = _SequencedClient([_status_error(500)])  # always 500
    engine = MineruNativeEngine(client=client)
    with pytest.raises(RuntimeError):
        engine.two_step_extract(object())
    assert client.calls == MINERU_MAX_RETRIES  # bounded — no infinite retry


def test_4xx_raises_immediately_without_retry():
    client = _SequencedClient([_status_error(400, "bad request")])
    engine = MineruNativeEngine(client=client)
    with pytest.raises(RuntimeError):
        engine.two_step_extract(object())
    assert client.calls == 1  # never retried


def test_read_timeout_not_retried_past_subcap():
    class _ReadTimeout(Exception):
        pass

    client = _SequencedClient([_ReadTimeout("read timed out"), _GOOD_PAGE])
    engine = MineruNativeEngine(client=client)
    with pytest.raises(_ReadTimeout):
        engine.two_step_extract(object())
    assert client.calls == 1  # read-timeout sub-cap is 1: do not repeat the wait


# --------------------------------------------------------------------------- #
# processor-level: retry PRECEDES the fail-closed ladder
# --------------------------------------------------------------------------- #
def _route_to_mineru(monkeypatch, engine, docling=None):
    for var in (
        "USE_MINERU_ENGINE",
        "USE_VLM_ENGINE",
        "USE_DOCLING_FAST",
        "USE_HYBRID_ENGINE",
        "USE_MINERU_QWEN_HYBRID",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("MINERU_ENDPOINT", raising=False)
    monkeypatch.setenv("USE_MINERU_ENGINE", "1")
    monkeypatch.setattr(processor, "MineruNativeEngine", lambda: engine)
    if docling is not None:
        monkeypatch.setattr(processor, "DoclingFastEngine", lambda: docling)


def test_processor_transient_500_recovers_without_fallback(tmp_path, monkeypatch):
    src = tmp_path / "doc.pdf"
    _make_pdf(src, n_pages=1)
    # Page 1: 500 once, then good -> in-engine retry recovers; ladder never runs.
    engine = MineruNativeEngine(
        client=_SequencedClient([_status_error(500, "broadcast_shapes"), _GOOD_PAGE])
    )

    class _Boom:
        def extract(self, _p):
            raise AssertionError("ladder must NOT be engaged when retry recovers")

    _route_to_mineru(monkeypatch, engine, docling=_Boom())

    out = processor.extract(str(src))
    assert isinstance(out, UniversalDocument)
    assert out.metadata.extra["extraction_engine"] == "mineru"
    assert out.metadata.extra["extraction_fallback"] is None


def _docling_doc(source: str) -> UniversalDocument:
    from mmrag_v2.universal.intermediate import (
        DocumentMetadata,
        Element,
        ElementType,
        PageClassification,
        UniversalPage,
    )

    page = UniversalPage(
        page_number=1,
        elements=[
            Element(type=ElementType.TEXT, content="docling recovered", bbox=None, confidence=1.0)
        ],
        classification=PageClassification.DIGITAL,
        dimensions=(1000, 1000),
    )
    return UniversalDocument(
        doc_id="d",
        source_file=source,
        file_type="pdf",
        pages=[page],
        metadata=DocumentMetadata(),
        total_pages=1,
    )


def test_processor_persistent_500_falls_through_to_ladder(tmp_path, monkeypatch):
    src = tmp_path / "doc.pdf"
    _make_pdf(src, n_pages=1)
    engine = MineruNativeEngine(client=_SequencedClient([_status_error(500)]))  # always 500

    class _Docling:
        def extract(self, _p):
            return _docling_doc(str(src))

    _route_to_mineru(monkeypatch, engine, docling=_Docling())

    out = processor.extract(str(src))
    # Bounded retry exhausted -> engine raised -> ladder served the doc.
    assert out.metadata.extra["extraction_engine"] == "mineru"
    assert out.metadata.extra["extraction_fallback"] == "docling_fast"
    assert engine._client.calls == MINERU_MAX_RETRIES  # retry happened before fallback
