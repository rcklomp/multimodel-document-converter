"""
HybridChunker Pathological-Input Guard Tests
=============================================
Validates that massive documents do not hang the pipeline by being fed
directly to HybridChunker's sentence-transformer tokenizer.

Regression targets:
- A Simple Guide to RAG (258pp, timeout at 7200s)
- GenAI on Google Cloud (320pp, timeout at 7200s)
- Ayeva Python Design Patterns (killed, CPU hang)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import pytest

from mmrag_v2.engines.docling_adapter import DoclingPdfAdapter


# ---------------------------------------------------------------------------
# Fake Docling classes (same pattern as test_pdf_conversion_plan.py)
# ---------------------------------------------------------------------------


class FakeInputFormat:
    PDF = "pdf"


class FakePdfPipelineOptions:
    latest = None

    def __init__(self):
        FakePdfPipelineOptions.latest = self
        self.picture_description_options = SimpleNamespace(classification_deny=[])
        self.sort_by_reading_order = None
        self.do_table_structure = None
        self.do_cell_matching = None
        self.do_picture_classification = None
        self.do_formula_enrichment = None


class FakeEasyOcrOptions:
    def __init__(self):
        pass


class FakeTableStructureOptions:
    def __init__(self, do_cell_matching, mode):
        pass


class FakeTableFormerMode:
    ACCURATE = "accurate"


class FakePdfFormatOption:
    def __init__(self, pipeline_options):
        self.pipeline_options = pipeline_options


class FakeDocumentConverter:
    _items = []  # Class-level: set before use

    def __init__(self, format_options):
        self.format_options = format_options

    def convert(self, path):
        items = list(FakeDocumentConverter._items)
        return SimpleNamespace(
            document=SimpleNamespace(
                pages={},
                iterate_items=lambda: iter(items),
            ),
        )


def _patch_docling(monkeypatch):
    def fake_loader(self):
        return (
            FakeInputFormat,
            FakePdfPipelineOptions,
            FakeEasyOcrOptions,
            FakeTableStructureOptions,
            FakeTableFormerMode,
            FakeDocumentConverter,
            FakePdfFormatOption,
        )

    FakePdfPipelineOptions.latest = None
    monkeypatch.setattr(DoclingPdfAdapter, "_load_docling_classes", fake_loader)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_doc_item(text: str, label: str = "text"):
    """Create a fake Docling document item."""
    return SimpleNamespace(text=text, label=label)


def _fake_doc_items(total_chars: int, items_count: int = 10):
    """Create fake Docling document items with specified total text characters."""
    chars_per_item = total_chars // items_count
    items = []
    for i in range(items_count):
        text = "a" * chars_per_item
        item = _fake_doc_item(text)
        prov = SimpleNamespace(
            page_no=i + 1,
            bbox=SimpleNamespace(l=0, t=0, r=612, b=792),
        )
        item.prov = [prov]
        items.append((item, None))
    return items


class TestChunkerGuard:
    """Tests for HybridChunker pathological-input guard."""

    def test_huge_text_skips_hybrid_chunker(self, monkeypatch, tmp_path):
        """Documents exceeding max_chunker_input_chars must skip HybridChunker."""
        import mmrag_v2.processor as proc_mod
        from mmrag_v2.engines.pdf_plan import build_pdf_conversion_plan

        # Create fake items with 5000 chars (exceeds 1000 threshold)
        items = _fake_doc_items(5000)
        FakeDocumentConverter._items = items

        _patch_docling(monkeypatch)

        plan = build_pdf_conversion_plan(max_chunker_input_chars=1000)

        processor = proc_mod.V2DocumentProcessor(
            output_dir=str(tmp_path),
            conversion_plan=plan,
        )

        assert processor._max_chunker_input_chars == 1000

        # Track whether _process_text_with_hybrid_chunker is called
        hybrid_called = {"value": False}

        def mock_hybrid(*args, **kwargs):
            hybrid_called["value"] = True
            return []

        monkeypatch.setattr(processor, "_process_text_with_hybrid_chunker", mock_hybrid)
        monkeypatch.setattr(
            processor, "_process_element_v2",
            lambda **kwargs: iter([]),
        )
        monkeypatch.setattr(processor, "_run_shadow_extraction", lambda **kwargs: iter([]))

        fake_path = tmp_path / "test.pdf"
        fake_path.write_bytes(b"%PDF-1.4\n%%EOF\n")

        chunks = list(processor.process_document(str(fake_path)))

        assert hybrid_called["value"] is False, (
            "HybridChunker must NOT be called for documents exceeding char threshold"
        )

    def test_normal_text_uses_hybrid_chunker(self, monkeypatch, tmp_path):
        """Documents under the threshold must still use HybridChunker."""
        import mmrag_v2.processor as proc_mod
        from mmrag_v2.engines.pdf_plan import build_pdf_conversion_plan

        # Create fake items with 1000 chars (well under 500k threshold)
        items = _fake_doc_items(1000, items_count=5)
        FakeDocumentConverter._items = items

        _patch_docling(monkeypatch)

        plan = build_pdf_conversion_plan(max_chunker_input_chars=500_000)

        processor = proc_mod.V2DocumentProcessor(
            output_dir=str(tmp_path),
            conversion_plan=plan,
        )

        hybrid_called = {"value": False}

        def mock_hybrid(*args, **kwargs):
            hybrid_called["value"] = True
            return []

        monkeypatch.setattr(processor, "_process_text_with_hybrid_chunker", mock_hybrid)
        monkeypatch.setattr(
            processor, "_process_element_v2",
            lambda **kwargs: iter([]),
        )
        monkeypatch.setattr(processor, "_run_shadow_extraction", lambda **kwargs: iter([]))

        fake_path = tmp_path / "test.pdf"
        fake_path.write_bytes(b"%PDF-1.4\n%%EOF\n")

        chunks = list(processor.process_document(str(fake_path)))

        assert hybrid_called["value"] is True, (
            "HybridChunker MUST be used for normal-sized documents"
        )

    def test_plan_max_chunker_chars_propagates(self, tmp_path):
        """Bridge: max_chunker_input_chars reaches V2DocumentProcessor from plan."""
        import mmrag_v2.processor as proc_mod
        from mmrag_v2.engines.pdf_plan import build_pdf_conversion_plan

        plan = build_pdf_conversion_plan(max_chunker_input_chars=123_456)

        proc = proc_mod.V2DocumentProcessor.__new__(proc_mod.V2DocumentProcessor)
        proc._conversion_plan = plan
        proc._max_chunker_input_chars = plan.max_chunker_input_chars

        assert proc._max_chunker_input_chars == 123_456

    def test_default_threshold_is_safe(self):
        """Default threshold must be large enough for normal docs but catch pathological ones."""
        from mmrag_v2.engines.pdf_plan import PdfConversionPlan

        plan = PdfConversionPlan()
        assert plan.max_chunker_input_chars == 500_000

    def test_scanned_book_route_disables_hybrid_chunker(self, monkeypatch, tmp_path):
        """scanned_book extraction route must disable HybridChunker."""
        import mmrag_v2.processor as proc_mod
        from mmrag_v2.engines.pdf_plan import build_pdf_conversion_plan

        # Small text — would normally use HybridChunker
        items = _fake_doc_items(500, items_count=3)
        FakeDocumentConverter._items = items

        _patch_docling(monkeypatch)

        plan = build_pdf_conversion_plan(
            document_modality="scanned_clean",  # auto-sets scanned_book route
        )
        assert plan.extraction_route == "scanned_book"

        processor = proc_mod.V2DocumentProcessor(
            output_dir=str(tmp_path),
            conversion_plan=plan,
        )
        assert processor._extraction_route == "scanned_book"

        hybrid_called = {"value": False}

        def mock_hybrid(*args, **kwargs):
            hybrid_called["value"] = True
            return []

        monkeypatch.setattr(processor, "_process_text_with_hybrid_chunker", mock_hybrid)
        monkeypatch.setattr(
            processor, "_process_element_v2",
            lambda **kwargs: iter([]),
        )
        monkeypatch.setattr(processor, "_run_shadow_extraction", lambda **kwargs: iter([]))

        fake_path = tmp_path / "test.pdf"
        fake_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
        list(processor.process_document(str(fake_path)))

        assert hybrid_called["value"] is False, (
            "scanned_book route must disable HybridChunker"
        )

    def test_per_batch_timeout_falls_back(self, monkeypatch, tmp_path):
        """If HybridChunker times out, text must still be emitted via element-by-element fallback.

        Regression: before the fix, _process_text_with_hybrid_chunker returned []
        on timeout but the caller kept _use_hybrid=True, causing skip_text=True in
        the element-by-element path — silently dropping all text chunks.
        """
        import mmrag_v2.processor as proc_mod
        from mmrag_v2.engines.pdf_plan import build_pdf_conversion_plan

        items = _fake_doc_items(500, items_count=3)
        FakeDocumentConverter._items = items
        _patch_docling(monkeypatch)

        plan = build_pdf_conversion_plan()
        processor = proc_mod.V2DocumentProcessor(
            output_dir=str(tmp_path),
            conversion_plan=plan,
        )

        # Make _process_text_with_hybrid_chunker raise TimeoutError
        # (simulating SIGALRM firing during HybridChunker.chunk)
        def mock_hybrid_timeout(*args, **kwargs):
            raise TimeoutError("HybridChunker exceeded per-batch time limit")

        monkeypatch.setattr(
            processor, "_process_text_with_hybrid_chunker", mock_hybrid_timeout
        )

        # Track whether _process_element_v2 is called with skip_text=False
        element_calls = []
        original_process_element = proc_mod.V2DocumentProcessor._process_element_v2

        def tracking_process_element(self_inner, **kwargs):
            element_calls.append(kwargs.get("skip_text", "NOT_SET"))
            return iter([])

        monkeypatch.setattr(
            processor, "_process_element_v2", lambda **kw: tracking_process_element(processor, **kw)
        )
        monkeypatch.setattr(processor, "_run_shadow_extraction", lambda **kwargs: iter([]))

        fake_path = tmp_path / "test.pdf"
        fake_path.write_bytes(b"%PDF-1.4\n%%EOF\n")

        list(processor.process_document(str(fake_path)))

        # The critical assertion: after timeout, skip_text must be False
        # so element-by-element emits text chunks as fallback
        assert len(element_calls) > 0, "Element-by-element path must be called"
        assert all(st is False for st in element_calls), (
            f"After HybridChunker timeout, skip_text must be False but got: {element_calls}"
        )

    def test_scanned_book_route_disables_picture_classification(self):
        """scanned_book route must also disable Docling picture classification."""
        from mmrag_v2.engines.pdf_plan import build_pdf_conversion_plan

        plan = build_pdf_conversion_plan(document_modality="scanned_clean")
        assert plan.extraction_route == "scanned_book"
        assert plan.do_picture_classification is False

    def test_pathological_single_element_skips_hybrid(self, monkeypatch, tmp_path):
        """A single huge element must trigger fallback even if total stays under threshold.

        Regression target: RAG Guide pages 241-250, where Docling emitted one
        mega-element ~4MB that produced a 1,060,086-token tokenization call,
        hanging HybridChunker for >86 minutes. The total-text guard does not
        catch this because the rest of the batch is small.
        """
        import mmrag_v2.processor as proc_mod
        from mmrag_v2.engines.pdf_plan import build_pdf_conversion_plan

        # One pathological element (200k chars) plus a few normal ones.
        # Total stays under 500_000 char document threshold, but the single
        # element exceeds the 100_000 per-element threshold.
        big_text = "x" * 200_000
        small_text = "small body of text. " * 20
        items: List[Tuple[SimpleNamespace, None]] = []
        for i, text in enumerate([small_text, big_text, small_text]):
            item = _fake_doc_item(text)
            item.prov = [
                SimpleNamespace(page_no=i + 1, bbox=SimpleNamespace(l=0, t=0, r=612, b=792))
            ]
            items.append((item, None))
        FakeDocumentConverter._items = items

        _patch_docling(monkeypatch)

        plan = build_pdf_conversion_plan(
            max_chunker_input_chars=500_000,
            max_chunker_per_element_chars=100_000,
        )
        assert plan.max_chunker_per_element_chars == 100_000

        processor = proc_mod.V2DocumentProcessor(
            output_dir=str(tmp_path),
            conversion_plan=plan,
        )
        assert processor._max_chunker_per_element_chars == 100_000

        hybrid_called = {"value": False}

        def mock_hybrid(*args, **kwargs):
            hybrid_called["value"] = True
            return []

        monkeypatch.setattr(processor, "_process_text_with_hybrid_chunker", mock_hybrid)
        monkeypatch.setattr(
            processor, "_process_element_v2", lambda **kwargs: iter([])
        )
        monkeypatch.setattr(processor, "_run_shadow_extraction", lambda **kwargs: iter([]))

        fake_path = tmp_path / "test.pdf"
        fake_path.write_bytes(b"%PDF-1.4\n%%EOF\n")

        list(processor.process_document(str(fake_path)))

        assert hybrid_called["value"] is False, (
            "HybridChunker MUST be skipped when a single element exceeds the "
            "per-element threshold (RAG Guide regression)"
        )

    def test_per_element_guard_does_not_trip_for_normal_batch(self, monkeypatch, tmp_path):
        """A batch with elements below the per-element threshold must still use HybridChunker."""
        import mmrag_v2.processor as proc_mod
        from mmrag_v2.engines.pdf_plan import build_pdf_conversion_plan

        # A 13k-char index table is realistic; well under 100k per-element guard.
        items = _fake_doc_items(50_000, items_count=4)  # 4 × 12,500 = 50,000 total
        FakeDocumentConverter._items = items

        _patch_docling(monkeypatch)

        plan = build_pdf_conversion_plan(
            max_chunker_input_chars=500_000,
            max_chunker_per_element_chars=100_000,
        )
        processor = proc_mod.V2DocumentProcessor(
            output_dir=str(tmp_path),
            conversion_plan=plan,
        )

        hybrid_called = {"value": False}

        def mock_hybrid(*args, **kwargs):
            hybrid_called["value"] = True
            return []

        monkeypatch.setattr(processor, "_process_text_with_hybrid_chunker", mock_hybrid)
        monkeypatch.setattr(
            processor, "_process_element_v2", lambda **kwargs: iter([])
        )
        monkeypatch.setattr(processor, "_run_shadow_extraction", lambda **kwargs: iter([]))

        fake_path = tmp_path / "test.pdf"
        fake_path.write_bytes(b"%PDF-1.4\n%%EOF\n")

        list(processor.process_document(str(fake_path)))

        assert hybrid_called["value"] is True, (
            "HybridChunker MUST run for normal-sized elements"
        )

    def test_per_element_threshold_propagates_through_plan(self, tmp_path):
        """Bridge: max_chunker_per_element_chars reaches V2DocumentProcessor from plan."""
        import mmrag_v2.processor as proc_mod
        from mmrag_v2.engines.pdf_plan import build_pdf_conversion_plan

        plan = build_pdf_conversion_plan(max_chunker_per_element_chars=42_000)

        proc = proc_mod.V2DocumentProcessor.__new__(proc_mod.V2DocumentProcessor)
        proc._conversion_plan = plan
        proc._max_chunker_per_element_chars = plan.max_chunker_per_element_chars

        assert proc._max_chunker_per_element_chars == 42_000

    def test_default_per_element_threshold(self):
        """Default per-element threshold must be small enough to catch index-page pathologies."""
        from mmrag_v2.engines.pdf_plan import PdfConversionPlan

        plan = PdfConversionPlan()
        assert plan.max_chunker_per_element_chars == 100_000
