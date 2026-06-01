"""Router Boundary-1: object-independent monospace (code) routing.

A pure-text page of source code carries NO image/table/drawing signal, so
the object-only pre-flight would route it to Docling and strip its
indentation. ``_classify_page`` now also routes on monospace-character
density. These tests pin that contract with a minimal fake page (no real
PDF, no PyMuPDF dependency at assert time).
"""

from mmrag_v3.engines.router import (
    DEFAULT_MONO_RATIO_THRESHOLD,
    HybridEngine,
)


class _FakePage:
    """Minimal stand-in for a ``fitz.Page`` for ``_classify_page``.

    Exposes exactly the four accessors the router calls: ``get_images``,
    ``find_tables``, ``get_drawings``, and ``get_text('dict')``.
    """

    def __init__(self, spans, n_images=0, n_tables=0, n_drawings=0):
        # spans: list of (text, font) for a single line/block.
        self._spans = spans
        self._n_images = n_images
        self._n_tables = n_tables
        self._n_drawings = n_drawings

    def get_images(self):
        return [object()] * self._n_images

    def find_tables(self):
        class _T:
            tables = [object()] * self._n_tables

        return _T()

    def get_drawings(self):
        return [object()] * self._n_drawings

    def get_text(self, kind):
        assert kind == "dict"
        return {
            "blocks": [{"lines": [{"spans": [{"text": t, "font": f} for (t, f) in self._spans]}]}]
        }


def _code_spans(mono_chars, prose_chars):
    spans = []
    if mono_chars:
        spans.append(("x" * mono_chars, "NimbusMonL-Regu"))
    if prose_chars:
        spans.append(("y" * prose_chars, "NimbusRomNo9L-Regu"))
    return spans


def test_pure_code_page_routes_to_vlm_without_objects():
    """A page that is >15% monospace chars and has ZERO objects -> VLM."""
    # 40 mono chars / 60 total = 0.40, well above the 0.10 default.
    page = _FakePage(_code_spans(mono_chars=40, prose_chars=60))
    engine = HybridEngine.__new__(HybridEngine)  # skip heavy __init__
    engine.drawings_threshold = 10
    engine.mono_ratio_threshold = DEFAULT_MONO_RATIO_THRESHOLD

    choice, reason = engine._classify_page(page)
    assert choice == "vlm", reason
    assert "code" in reason and "mono_ratio" in reason


def test_inline_monospace_prose_stays_docling():
    """A prose page with one inline mono token (a URL/var name) -> Docling."""
    # 3 mono chars / 200 total = 0.015, below threshold.
    page = _FakePage(_code_spans(mono_chars=3, prose_chars=200))
    engine = HybridEngine.__new__(HybridEngine)
    engine.drawings_threshold = 10
    engine.mono_ratio_threshold = DEFAULT_MONO_RATIO_THRESHOLD

    choice, reason = engine._classify_page(page)
    assert choice == "docling", reason


def test_nimbus_roman_prose_is_not_treated_as_mono():
    """Regression: the foundry prefix 'nimbus' must NOT match Nimbus Roman.

    Gemini's first token list used 'nimbus', which matched the prose face
    NimbusRomNo9L and drove every page to the VLM. Guard against reintroducing
    that bug: a 100%-NimbusRoman page must route to Docling.
    """
    page = _FakePage(_code_spans(mono_chars=0, prose_chars=300))
    engine = HybridEngine.__new__(HybridEngine)
    engine.drawings_threshold = 10
    engine.mono_ratio_threshold = DEFAULT_MONO_RATIO_THRESHOLD

    choice, _ = engine._classify_page(page)
    assert choice == "docling"


def test_object_signal_still_wins_over_mono_check():
    """An image-bearing page short-circuits to VLM before the mono check."""
    page = _FakePage(_code_spans(mono_chars=0, prose_chars=100), n_images=1)
    engine = HybridEngine.__new__(HybridEngine)
    engine.drawings_threshold = 10
    engine.mono_ratio_threshold = DEFAULT_MONO_RATIO_THRESHOLD

    choice, reason = engine._classify_page(page)
    assert choice == "vlm"
    assert "images" in reason


def test_threshold_is_configurable():
    """A high threshold demotes a borderline code page back to Docling."""
    # 12 mono / 100 = 0.12: VLM at default 0.10, Docling at 0.20.
    page = _FakePage(_code_spans(mono_chars=12, prose_chars=88))
    engine = HybridEngine.__new__(HybridEngine)
    engine.drawings_threshold = 10

    engine.mono_ratio_threshold = 0.10
    assert engine._classify_page(page)[0] == "vlm"

    engine.mono_ratio_threshold = 0.20
    assert engine._classify_page(page)[0] == "docling"
