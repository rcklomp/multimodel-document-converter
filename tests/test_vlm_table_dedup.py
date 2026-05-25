"""v2.16 Phase 4 — VLM-table IoU dedup bridge tests.

Tests `BatchProcessor._apply_vlm_table_iou_dedup` in isolation: synthetic
IngestionChunk lists go in, expected post-dedup lists come out. The five
canonical scenarios from PLAN_V2.16.md §3 Phase 4 step 4 are covered
(IoU=0.95, IoU=0.50, IoU=0.0, page with no VLM tables, multi-table page).
"""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mmrag_v2.batch_processor import BatchProcessor  # noqa: E402
from mmrag_v2.engines.pdf_plan import PdfConversionPlan, build_pdf_conversion_plan  # noqa: E402
from mmrag_v2.schema.ingestion_schema import (  # noqa: E402
    AssetReference,
    ChunkType,
    HierarchyMetadata,
    IngestionChunk,
    Modality,
    SemanticContext,
    SpatialMetadata,
    create_table_chunk,
    create_text_chunk,
    FileType,
)


def _build_proc(plan: PdfConversionPlan | None = None) -> BatchProcessor:
    """Build a bare BatchProcessor for unit-testing the dedup pass.
    We only set the fields the dedup method touches (_conversion_plan)."""
    proc = BatchProcessor.__new__(BatchProcessor)
    proc._conversion_plan = plan
    return proc


def _text_chunk(*, page: int, bbox, content: str = "prose row 1 row 2",
                 position: int = 0) -> IngestionChunk:
    return create_text_chunk(
        doc_id="d",
        content=content,
        source_file="x.pdf",
        file_type=FileType.PDF,
        page_number=page,
        bbox=list(bbox),
        page_width=612,
        page_height=792,
        position=position,
    )


def _vlm_table_chunk(*, page: int, bbox, position: int = 0,
                      extraction_method: str = "vlm_table_markdown_forced",
                      content: str = "| col1 | col2 |\n|---|---|\n| a | b |") -> IngestionChunk:
    chunk = create_table_chunk(
        doc_id="d",
        content=content,
        source_file="x.pdf",
        file_type=FileType.PDF,
        page_number=page,
        bbox=list(bbox),
        page_width=612,
        page_height=792,
        position=position,
        extraction_method=extraction_method,
        asset_path="x.png",
    )
    return chunk


def test_iou_above_threshold_suppresses_text_chunk():
    """IoU=0.95 between text and VLM table → text suppressed."""
    plan = build_pdf_conversion_plan(dedup_vlm_table_iou_threshold=0.85)
    proc = _build_proc(plan)
    text = _text_chunk(page=1, bbox=[100, 100, 200, 200], position=1)
    table = _vlm_table_chunk(page=1, bbox=[100, 100, 199, 199], position=2)
    out = proc._apply_vlm_table_iou_dedup([text, table])
    # Table stays; text dropped.
    assert len(out) == 1
    assert out[0].modality == Modality.TABLE


def test_iou_below_threshold_preserves_text_chunk():
    """IoU=0.50 (≤ 0.85) → text chunk NOT suppressed."""
    plan = build_pdf_conversion_plan(dedup_vlm_table_iou_threshold=0.85)
    proc = _build_proc(plan)
    # 100x100 boxes overlapping by 50% on x = IoU ≈ 0.33 (< 0.85).
    text = _text_chunk(page=1, bbox=[0, 0, 100, 100], position=1)
    table = _vlm_table_chunk(page=1, bbox=[50, 0, 150, 100], position=2)
    out = proc._apply_vlm_table_iou_dedup([text, table])
    assert len(out) == 2  # both retained


def test_iou_zero_disjoint_bboxes_preserves_text():
    """IoU=0.0 (disjoint) → text NOT suppressed (negative case)."""
    plan = build_pdf_conversion_plan(dedup_vlm_table_iou_threshold=0.85)
    proc = _build_proc(plan)
    text = _text_chunk(page=1, bbox=[0, 0, 100, 100], position=1)
    table = _vlm_table_chunk(page=1, bbox=[500, 500, 600, 600], position=2)
    out = proc._apply_vlm_table_iou_dedup([text, table])
    assert len(out) == 2


def test_page_with_no_vlm_tables_passes_text_through():
    """No VLM tables on the page → all text chunks pass through (no false
    positives, cheap-exit guard)."""
    plan = build_pdf_conversion_plan(dedup_vlm_table_iou_threshold=0.85)
    proc = _build_proc(plan)
    text1 = _text_chunk(page=1, bbox=[0, 0, 100, 100], position=1)
    text2 = _text_chunk(page=1, bbox=[200, 200, 300, 300], position=2)
    out = proc._apply_vlm_table_iou_dedup([text1, text2])
    assert len(out) == 2


def test_two_vlm_tables_one_overlapping_text_suppresses():
    """Page with 2 VLM tables + 1 text chunk overlapping ONE table at 0.95 →
    text suppressed (any single overlap above threshold fires)."""
    plan = build_pdf_conversion_plan(dedup_vlm_table_iou_threshold=0.85)
    proc = _build_proc(plan)
    text = _text_chunk(page=1, bbox=[100, 100, 200, 200], position=1)
    table1 = _vlm_table_chunk(page=1, bbox=[100, 100, 199, 199], position=2)
    table2 = _vlm_table_chunk(page=1, bbox=[500, 500, 600, 600], position=3)
    out = proc._apply_vlm_table_iou_dedup([text, table1, table2])
    text_kept = [c for c in out if c.modality == Modality.TEXT]
    assert text_kept == []


def test_dedup_disabled_at_threshold_zero_preserves_all():
    """Threshold=0.0 disables the pass: even high-IoU overlaps pass through."""
    plan = build_pdf_conversion_plan(dedup_vlm_table_iou_threshold=0.0)
    proc = _build_proc(plan)
    text = _text_chunk(page=1, bbox=[100, 100, 200, 200], position=1)
    table = _vlm_table_chunk(page=1, bbox=[100, 100, 199, 199], position=2)
    out = proc._apply_vlm_table_iou_dedup([text, table])
    assert len(out) == 2


def test_dedup_uses_default_threshold_when_plan_missing():
    """Defensive: no plan attached → cheap-exit treats as default (0.85);
    behavior matches the spec'd default. Threshold lookup falls through
    via getattr(..., default=0.85)."""
    proc = _build_proc(plan=None)
    text = _text_chunk(page=1, bbox=[100, 100, 200, 200], position=1)
    table = _vlm_table_chunk(page=1, bbox=[100, 100, 199, 199], position=2)
    out = proc._apply_vlm_table_iou_dedup([text, table])
    # Without a plan, getattr falls through to default 0.85 → IoU=~0.97 > 0.85 → drop.
    assert len(out) == 1
    assert out[0].modality == Modality.TABLE


def test_dedup_only_fires_against_vlm_methods():
    """Tables extracted via standard `docling_table_markdown` are NOT VLM-
    sourced and should NOT trigger dedup of overlapping text (the v2.14 P1
    regression is specific to the VLM-table-emission path)."""
    plan = build_pdf_conversion_plan(dedup_vlm_table_iou_threshold=0.85)
    proc = _build_proc(plan)
    text = _text_chunk(page=1, bbox=[100, 100, 200, 200], position=1)
    docling_table = _vlm_table_chunk(
        page=1, bbox=[100, 100, 199, 199], position=2,
        extraction_method="docling_table_markdown",
    )
    out = proc._apply_vlm_table_iou_dedup([text, docling_table])
    assert len(out) == 2  # Docling-only path is unchanged
