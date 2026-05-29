"""V3 input-boundary integration test (Phase A Step 5).

Deterministic, offline regression net for the batch_processor.py rewire that
made the chunker input boundary consume a UniversalDocument (UIR) produced by
the engine-agnostic ``mmrag_v3`` engines, instead of a DoclingDocument.

This test is the standing guard demanded by the Step-5 revised DoD:
  1. Uses the offline DoclingFast engine (CPU, no VLM, no network) via
     ``USE_DOCLING_FAST=1`` so it is fully deterministic and budget-free.
  2. Processes a real PDF end-to-end through ``BatchProcessor.process_pdf``.
  3. Asserts the emitted chunks conform to the V3 UIR-native shape — every
     content chunk carries ``extraction_method == "uir_native_chunker"`` and a
     valid 1-indexed ``page_number`` — i.e. zero DoclingDocument artifacts
     leaked through the (now severed) input boundary.

It must stay in the mandatory ``pytest tests/`` block.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mmrag_v2.batch_processor import BatchProcessor
from mmrag_v2.engines.pdf_plan import build_pdf_conversion_plan

# Smallest corpus PDF (2 pages, pure prose). Routed 100% through DoclingFast
# when USE_DOCLING_FAST=1 — no VLM call, no OpenRouter budget consumed.
SMOKE_PDF = Path("data/raw/Bevestigingsmiddelen.pdf")

# The single marker that proves a chunk came through the V3 UIR-native chunker
# rather than any legacy Docling-DOM path.
V3_EXTRACTION_METHOD = "uir_native_chunker"


@pytest.mark.skipif(not SMOKE_PDF.exists(), reason=f"corpus PDF missing: {SMOKE_PDF}")
def test_batch_processor_is_v3_uir_native_on_input(tmp_path, monkeypatch):
    """A document processed through BatchProcessor emits V3 UIR-native chunks."""
    # Force the offline, deterministic DoclingFast engine route.
    monkeypatch.setenv("USE_DOCLING_FAST", "1")

    bp = BatchProcessor(output_dir=str(tmp_path))
    bp.set_conversion_plan(
        build_pdf_conversion_plan(
            profile_type="digital_magazine",
            document_domain="general",
        )
    )

    result = bp.process_pdf(str(SMOKE_PDF))

    # End-to-end completion (no crash → no failure-fallback).
    assert result is not None
    assert result.total_chunks > 0, "rewired pipeline produced zero chunks"

    out = tmp_path / "ingestion.jsonl"
    assert out.exists(), "no ingestion.jsonl written"

    records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    content_chunks = [r for r in records if r.get("modality")]
    assert content_chunks, "no content chunks emitted"

    for chunk in content_chunks:
        meta = chunk.get("metadata", {})

        # V3 marker: every content chunk came through the UIR-native chunker.
        assert meta.get("extraction_method") == V3_EXTRACTION_METHOD, (
            f"chunk {chunk.get('chunk_id')} has extraction_method="
            f"{meta.get('extraction_method')!r}; expected {V3_EXTRACTION_METHOD!r} "
            "(a DoclingDocument artifact leaked through the input boundary)"
        )

        # Valid, 1-indexed absolute page number survived batch projection.
        page = meta.get("page_number")
        assert isinstance(page, int) and page >= 1, (
            f"chunk {chunk.get('chunk_id')} has invalid page_number={page!r}"
        )

        # No DoclingDocument / DocChunk artifact strings anywhere in metadata.
        blob = json.dumps(meta).lower()
        assert "doclingdocument" not in blob and "docchunk" not in blob, (
            f"chunk {chunk.get('chunk_id')} metadata references a Docling DOM type"
        )
