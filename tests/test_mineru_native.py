"""Deterministic offline tests for the MinerU2.5 -> UIR converter.

No model, no network, no mineru-vl-utils: these exercise the pure converter
(`src/mmrag_v3/engines/mineru_native.py`) against a DENSE synthetic MinerU
payload mirroring the real ``two_step_extract`` output shape captured from the
golden set (firearms-spec table page: a ``CONTENTS`` title plus many numeric
rows, with code/image/caption/merge_prev elements mixed in).

Contract under test:
  * MinerU's 13-value type vocabulary -> the frozen 3-value ElementType,
    with ``code`` smuggled as TEXT + Modality.CODE promotion (Charter §7.1);
  * normalized [0,1] float bbox -> integer [0,1000] UIR frame (REQ-COORD-01),
    with BoundingBox invariants and clamping;
  * every MinerU type preserved as ``source_label`` provenance;
  * dense pages converted element-for-element (no silent drops);
  * document assembly (doc_id, page dims, has_images/has_text_layer).
"""

from __future__ import annotations

import typing

import fitz
import pytest

from mmrag_v2.universal.intermediate import ElementType, UniversalDocument
from mmrag_v3.engines.mineru_native import (
    MineruConfigError,
    MineruNativeEngine,
    _html_table_to_markdown,
    _mineru_bbox_to_uir,
    _mineru_element_to_element,
    mineru_page_to_universal_page,
    mineru_to_universal_document,
)


def _dense_table_page():
    """A dense MinerU table page (24 elements) plus code/image/caption.

    Coordinates and content mirror the probed ``table_spec_firearms`` page;
    25+ rows of numeric spec data are exactly the dense class MinerU was
    chosen to survive (Qwen emptied it).
    """
    elements = [
        {"type": "header", "bbox": [0.08, 0.043, 0.40, 0.054], "angle": 0, "content": "FIREARMS"},
        {"type": "title", "bbox": [0.08, 0.248, 0.535, 0.312], "angle": 0, "content": "CONTENTS"},
    ]
    # 24 dense numeric spec rows (the table body).
    for i in range(24):
        y = 0.34 + i * 0.016
        elements.append(
            {
                "type": "text",
                "bbox": [0.085, round(y, 3), 0.473, round(y + 0.015, 3)],
                "angle": 0,
                "content": f"Ruger 44 Carbine {120 + i}",
                "merge_prev": i % 2 == 1,
            }
        )
    elements.append(
        {
            "type": "table",
            "bbox": [0.5, 0.34, 0.95, 0.74],
            "angle": 0,
            # MinerU emits tables as HTML (the real CarOK shape, with empty cells).
            "content": (
                "<table><tr><td>Model</td><td>Cal</td></tr>"
                "<tr><td>Ruger 44 Carbine</td><td>.44</td></tr>"
                "<tr><td>Mini-14</td><td></td></tr></table>"
            ),
        }
    )
    elements.append(
        {"type": "image", "bbox": [0.5, 0.75, 0.95, 0.95], "angle": 0, "content": "rifle photo"}
    )
    elements.append(
        {
            "type": "image_caption",
            "bbox": [0.5, 0.955, 0.95, 0.97],
            "angle": 0,
            "content": "Figure 1. Ruger 44 Carbine",
        }
    )
    elements.append(
        {
            "type": "code",
            "bbox": [0.08, 0.80, 0.47, 0.95],
            "angle": 0,
            "content": "def fire():\n    if safety:\n        return\n    discharge()",
        }
    )
    return elements


def test_bbox_projection_scales_unit_floats_to_uir_frame():
    # 0.085 -> 85, 0.473 -> 473, exact rounding.
    assert _mineru_bbox_to_uir([0.085, 0.248, 0.473, 0.312]) == [85, 248, 473, 312]
    # Full-frame extremes clamp into [0, 1000].
    assert _mineru_bbox_to_uir([0.0, 0.0, 1.0, 1.0]) == [0, 0, 1000, 1000]
    # Out-of-range floats clamp, never exceed 1000 or go below 0.
    box = _mineru_bbox_to_uir([-0.01, -0.2, 1.04, 1.5])
    assert box == [0, 0, 1000, 1000]


def test_bbox_swapped_pairs_are_corrected():
    assert _mineru_bbox_to_uir([0.6, 0.6, 0.2, 0.2]) == [200, 200, 600, 600]


def test_bbox_degenerate_point_satisfies_strict_invariants():
    # A zero-area box must still yield x_max > x_min and y_max > y_min so
    # BoundingBox.__post_init__ accepts it.
    box = _mineru_bbox_to_uir([0.5, 0.5, 0.5, 0.5])
    assert box[2] > box[0] and box[3] > box[1]
    # Right/bottom edge degenerate: clamp inward, do not exceed 1000.
    box2 = _mineru_bbox_to_uir([1.0, 1.0, 1.0, 1.0])
    assert box2[2] == 1000 and box2[0] == 999
    assert box2[3] == 1000 and box2[1] == 999


def test_html_table_transcoded_to_markdown_grid():
    html = (
        "<table><tr><td>aant</td><td>merk</td><td>ink.ex.BTW</td></tr>"
        "<tr><td></td><td>Castrol</td><td>6,55</td></tr>"
        "<tr><td>3</td><td>Castrol</td><td>26,85</td></tr></table>"
    )
    md = _html_table_to_markdown(html)
    lines = md.splitlines()
    assert lines[0] == "| aant | merk | ink.ex.BTW |"
    assert lines[1] == "| --- | --- | --- |"  # the separator the gate requires
    assert lines[2] == "|  | Castrol | 6,55 |"  # empty cell preserved as blank
    assert lines[3] == "| 3 | Castrol | 26,85 |"


def test_html_table_colspan_and_pipe_escaping():
    html = (
        "<table><tr><th>A</th><th colspan='2'>B</th></tr>"
        "<tr><td>a|b</td><td>c</td><td>d</td></tr></table>"
    )
    md = _html_table_to_markdown(html)
    lines = md.splitlines()
    # colspan=2 -> header padded to 3 columns; separator has 3 dashes.
    assert lines[0] == "| A | B |  |"
    assert lines[1] == "| --- | --- | --- |"
    # a literal pipe in a cell is escaped so it cannot break the grid.
    assert r"a\|b" in lines[2]


def test_html_table_unparseable_returns_none():
    assert _html_table_to_markdown("no table here") is None


def test_table_element_content_becomes_markdown():
    el = _mineru_element_to_element(
        {
            "type": "table",
            "bbox": [0.1, 0.1, 0.9, 0.9],
            "content": "<table><tr><td>x</td><td>y</td></tr><tr><td>1</td><td>2</td></tr></table>",
        },
        0,
    )
    assert el.type is ElementType.TABLE
    assert "<table>" not in el.content
    assert "| x | y |" in el.content
    assert "| --- | --- |" in el.content


def test_type_mapping_table_image_text():
    table = _mineru_element_to_element(
        {"type": "table", "bbox": [0.1, 0.1, 0.9, 0.9], "content": "x"}, 0
    )
    image = _mineru_element_to_element(
        {"type": "image", "bbox": [0.1, 0.1, 0.9, 0.9], "content": "x"}, 1
    )
    title = _mineru_element_to_element(
        {"type": "title", "bbox": [0.1, 0.1, 0.9, 0.2], "content": "x"}, 2
    )
    header = _mineru_element_to_element(
        {"type": "header", "bbox": [0.1, 0.1, 0.9, 0.2], "content": "x"}, 3
    )
    page_no = _mineru_element_to_element(
        {"type": "page_number", "bbox": [0.4, 0.95, 0.6, 0.99], "content": "7"}, 4
    )
    assert table.type is ElementType.TABLE
    assert image.type is ElementType.IMAGE
    assert title.type is ElementType.TEXT
    assert header.type is ElementType.TEXT
    assert page_no.type is ElementType.TEXT


def test_unknown_type_degrades_to_text():
    # MinerU vocabulary evolves; an unseen type must not crash, it degrades.
    el = _mineru_element_to_element(
        {"type": "interline_equation", "bbox": [0.1, 0.1, 0.9, 0.2], "content": "E=mc^2"}, 0
    )
    assert el.type is ElementType.TEXT
    assert el.source_label == "interline_equation"


def test_code_is_smuggled_as_text_with_promotion_metadata():
    code_src = "def fire():\n    if safety:\n        return\n    discharge()"
    el = _mineru_element_to_element(
        {"type": "code", "bbox": [0.08, 0.8, 0.47, 0.95], "content": code_src}, 0
    )
    assert el.type is ElementType.TEXT
    assert el.metadata["promoted_modality"] == "code"
    assert el.metadata["original_vlm_type"] == "code"
    assert el.source_label == "code"
    # Code content must stay verbatim (exact indentation preserved).
    assert el.content == code_src


def test_source_label_preserves_mineru_type():
    for mtype in ("title", "header", "footer", "table_caption", "list", "aside_text"):
        el = _mineru_element_to_element(
            {"type": mtype, "bbox": [0.1, 0.1, 0.9, 0.2], "content": "x"}, 0
        )
        assert el.source_label == mtype
        assert el.type is ElementType.TEXT
        assert el.metadata is None or "promoted_modality" not in el.metadata


def test_merge_prev_preserved_in_metadata():
    on = _mineru_element_to_element(
        {"type": "text", "bbox": [0.1, 0.1, 0.9, 0.2], "content": "x", "merge_prev": True}, 0
    )
    off = _mineru_element_to_element(
        {"type": "text", "bbox": [0.1, 0.1, 0.9, 0.2], "content": "x", "merge_prev": False}, 0
    )
    assert on.metadata["merge_prev"] is True
    assert off.metadata is None or "merge_prev" not in off.metadata


def test_degenerate_repeat_collapsed_on_prose_not_code():
    looped = "Transporter 1.9 TD 68pk, " * 40
    prose = _mineru_element_to_element(
        {"type": "text", "bbox": [0.1, 0.1, 0.9, 0.2], "content": looped}, 0
    )
    assert prose.content.count("Transporter 1.9 TD 68pk,") < 40
    code_loop = "x = 1\n" * 40
    code = _mineru_element_to_element(
        {"type": "code", "bbox": [0.1, 0.1, 0.9, 0.2], "content": code_loop}, 0
    )
    assert code.content == code_loop  # code is never collapsed


def test_reading_order_preserved_as_element_index():
    page = mineru_page_to_universal_page(
        _dense_table_page(), page_number=3, dimensions=(1654, 2339)
    )
    assert [e.element_index for e in page.elements] == list(range(len(page.elements)))


def test_dense_page_converts_every_element():
    raw = _dense_table_page()
    page = mineru_page_to_universal_page(raw, page_number=3, dimensions=(1654, 2339))
    # No silent drops: element-for-element.
    assert len(page.elements) == len(raw)
    assert page.page_number == 3
    assert page.dimensions == (1654, 2339)
    # The dense numeric rows + caption survive (the Qwen-empties failure class).
    assert page.table_elements, "table element lost"
    assert page.image_elements, "image element lost"
    assert sum(1 for e in page.text_elements if e.content.startswith("Ruger 44 Carbine")) == 24


def test_document_assembly_stamps_metadata(tmp_path):
    src = tmp_path / "firearms.pdf"
    src.write_bytes(b"%PDF-1.4 dense firearms spec bytes for md5")
    doc = mineru_to_universal_document(
        [
            (1, _dense_table_page(), (1654, 2339)),
            (
                2,
                [{"type": "text", "bbox": [0.1, 0.1, 0.9, 0.2], "content": "page two prose"}],
                (1654, 2339),
            ),
        ],
        str(src),
    )
    assert doc.total_pages == 2
    assert len(doc.doc_id) == 12  # MD5[:12]
    assert doc.file_type == "pdf"
    assert doc.metadata.page_count == 2
    assert doc.metadata.has_images is True
    assert doc.metadata.has_text_layer is True
    assert doc.metadata.file_size_bytes == src.stat().st_size
    # Page dims carried for the chunker's Locator stamping.
    assert all(p.dimensions == (1654, 2339) for p in doc.pages)


def test_bbox_values_land_in_valid_uir_range_on_dense_page():
    page = mineru_page_to_universal_page(
        _dense_table_page(), page_number=1, dimensions=(1654, 2339)
    )
    for el in page.elements:
        if el.bbox is None:
            continue
        for coord in el.bbox.to_list():
            assert 0 <= coord <= 1000
        assert el.bbox.x_max > el.bbox.x_min
        assert el.bbox.y_max > el.bbox.y_min


# ---------------------------------------------------------------------------
# Engine (MineruNativeEngine) — offline, mocked transport
# ---------------------------------------------------------------------------


class _FakeMineruClient:
    """Stand-in for mineru_vl_utils.MinerUClient: no model, no network.

    Records the rendered images it is handed and returns a canned, per-page
    element list so the engine's render -> drive -> convert path is exercised
    end-to-end without the heavy MinerU stack.
    """

    def __init__(self, pages_elements):
        self._pages = list(pages_elements)
        self.calls = []

    def two_step_extract(self, image):
        self.calls.append(image)
        # One canned element list per call, in order.
        return self._pages[len(self.calls) - 1]


def _make_pdf(path, n_pages=2):
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page(width=595, height=842)  # A4 points
        page.insert_text((72, 72), f"page {i + 1}")
    doc.save(str(path))
    doc.close()


def test_engine_honors_v3_extraction_contract():
    extract = MineruNativeEngine.extract
    hints = typing.get_type_hints(extract)
    assert hints.get("file_path") is str
    assert hints.get("return") is UniversalDocument


def test_engine_renders_drives_and_assembles_uir(tmp_path):
    src = tmp_path / "doc.pdf"
    _make_pdf(src, n_pages=2)
    canned = [
        _dense_table_page(),  # page 1: dense table + code + image
        [{"type": "text", "bbox": [0.1, 0.1, 0.9, 0.2], "content": "second page prose"}],
    ]
    client = _FakeMineruClient(canned)
    engine = MineruNativeEngine(client=client)

    doc = engine.extract(str(src))

    # Rendered both pages and handed each to MinerU.
    assert len(client.calls) == 2
    # Renderer produced PIL images at 200 DPI (A4 595x842 pt -> ~1654x2339 px).
    first_img = client.calls[0]
    assert first_img.size[0] > 1600 and first_img.size[1] > 2300
    # Assembled a 2-page UIR with the canned content, real page dims stamped.
    assert isinstance(doc, UniversalDocument)
    assert doc.total_pages == 2
    assert doc.get_page(1).dimensions == first_img.size
    assert doc.get_page(1).table_elements and doc.get_page(1).image_elements
    assert doc.get_page(2).text_elements[0].content == "second page prose"
    # Page numbers come from the engine's index, not the model.
    assert [p.page_number for p in doc.pages] == [1, 2]


def test_engine_missing_file_raises():
    engine = MineruNativeEngine(client=_FakeMineruClient([]))
    with pytest.raises(FileNotFoundError):
        engine.extract("/no/such/file.pdf")


def test_engine_without_endpoint_raises_config_error(monkeypatch):
    monkeypatch.delenv("MINERU_ENDPOINT", raising=False)
    engine = MineruNativeEngine()  # no client injected, no endpoint
    with pytest.raises(MineruConfigError):
        _ = engine.client


# ---------------------------------------------------------------------------
# Processor routing (USE_MINERU_ENGINE)
# ---------------------------------------------------------------------------


def test_use_mineru_engine_flag_detected(monkeypatch):
    from mmrag_v3 import processor

    monkeypatch.setenv("USE_MINERU_ENGINE", "1")
    assert processor.is_mineru_route_enabled() is True
    monkeypatch.setenv("USE_MINERU_ENGINE", "0")
    assert processor.is_mineru_route_enabled() is False
    monkeypatch.delenv("USE_MINERU_ENGINE", raising=False)
    assert processor.is_mineru_route_enabled() is False


def test_mineru_route_takes_precedence_and_calls_engine(monkeypatch):
    from mmrag_v3 import processor

    monkeypatch.setenv("USE_MINERU_ENGINE", "1")
    # Even with other flags set, MinerU wins.
    monkeypatch.setenv("USE_DOCLING_FAST", "1")
    monkeypatch.setenv("USE_VLM_ENGINE", "1")

    seen = {}

    class _SpyEngine:
        def extract(self, file_path):
            seen["file_path"] = file_path
            return "SENTINEL_UIR"

    def _no(*_a, **_k):  # other engines must NOT be constructed
        raise AssertionError("non-MinerU engine constructed despite USE_MINERU_ENGINE=1")

    monkeypatch.setattr(processor, "MineruNativeEngine", lambda: _SpyEngine())
    monkeypatch.setattr(processor, "VlmNativeEngine", _no)
    monkeypatch.setattr(processor, "DoclingFastEngine", _no)
    monkeypatch.setattr(processor, "HybridEngine", _no)

    result = processor.extract("/some/doc.pdf")
    assert result == "SENTINEL_UIR"
    assert seen["file_path"] == "/some/doc.pdf"


def _route_spies(monkeypatch):
    """Patch all four engines to identifying sentinels; return the processor."""
    from mmrag_v3 import processor

    def _spy(name):
        class _E:
            def extract(self, _fp):
                return name

        return lambda: _E()

    monkeypatch.setattr(processor, "MineruNativeEngine", _spy("mineru"))
    monkeypatch.setattr(processor, "VlmNativeEngine", _spy("vlm"))
    monkeypatch.setattr(processor, "DoclingFastEngine", _spy("docling"))
    monkeypatch.setattr(processor, "HybridEngine", _spy("hybrid"))
    for var in ("USE_MINERU_ENGINE", "USE_VLM_ENGINE", "USE_DOCLING_FAST", "USE_HYBRID_ENGINE"):
        monkeypatch.delenv(var, raising=False)
    return processor


def test_default_route_is_mineru_when_endpoint_configured(monkeypatch):
    proc = _route_spies(monkeypatch)
    monkeypatch.setenv("MINERU_ENDPOINT", "http://10.0.10.239:8001")
    assert proc.extract("/d.pdf") == "mineru"


def test_default_route_falls_back_to_hybrid_without_endpoint(monkeypatch):
    proc = _route_spies(monkeypatch)
    monkeypatch.delenv("MINERU_ENDPOINT", raising=False)
    assert proc.extract("/d.pdf") == "hybrid"


def test_use_hybrid_engine_overrides_mineru_default(monkeypatch):
    proc = _route_spies(monkeypatch)
    monkeypatch.setenv("MINERU_ENDPOINT", "http://10.0.10.239:8001")  # would default to mineru
    monkeypatch.setenv("USE_HYBRID_ENGINE", "1")  # explicit legacy override
    assert proc.extract("/d.pdf") == "hybrid"


def test_docling_fast_overrides_mineru_default(monkeypatch):
    # The offline smoke relies on this: USE_DOCLING_FAST wins even with a
    # MinerU endpoint configured in the environment.
    proc = _route_spies(monkeypatch)
    monkeypatch.setenv("MINERU_ENDPOINT", "http://10.0.10.239:8001")
    monkeypatch.setenv("USE_DOCLING_FAST", "1")
    assert proc.extract("/d.pdf") == "docling"
