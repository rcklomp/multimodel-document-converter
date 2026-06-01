"""AGENT-SPATIAL-20 adopt test (PLAN_V3.1 P3 / R6, 2026-06-01).

WHY THIS EXISTS
---------------
`BatchProcessor._apply_spatial_refiner` (aliased `_apply_vertical_proximity_merger`
/ `..._pagewise`) runs on every document at 6+ live call sites in `process_pdf`
finalize, yet had ZERO test coverage. Its core rule IS the project's hard
invariant AGENT-SPATIAL-20 (AGENTS.md §1.6 + CLAUDE.md "Project Invariants" +
PROJECT_STATUS Phase B debt #1):

    "Refinement logic must rely on a single 20-unit vertical threshold. No
     profile-specific or heading-specific branches allowed."

Before this module the 20 was a bare magic literal (`if 0 <= v_gap <= 20 ...`,
batch_processor.py ~L3517) guarded only by prose. A change to 25, or the
addition of a profile branch, would have shipped silently. This test makes the
invariant executable: it pins the threshold value, the geometry predicate, and
the no-cross-page / code-vs-prose / no-profile-branch rules.

If a future change INTENTIONALLY re-tunes the threshold, this test must be
updated WITH a docs/DECISIONS.md entry and the AGENTS.md/CLAUDE.md invariant text
- per AGENT-TEST-01, do not weaken it to make drift pass.

All offline, pure-Python (no Docling, no VLM, no network).
"""

from __future__ import annotations

import inspect

from mmrag_v2.batch_processor import BatchProcessor
from mmrag_v2.schema.ingestion_schema import (
    ChunkType,
    FileType,
    create_text_chunk,
)


def _processor(tmp_path) -> BatchProcessor:
    return BatchProcessor(
        output_dir=str(tmp_path),
        batch_size=3,
        vision_provider="none",
        enable_ocr=False,
    )


def _chunk(
    content: str,
    bbox: list[int],
    page: int = 1,
    chunk_type: ChunkType = ChunkType.PARAGRAPH,
):
    """A text chunk with a real spatial.bbox in [0,1000] page-portrait coords."""
    return create_text_chunk(
        doc_id="spatial20",
        content=content,
        source_file="doc.pdf",
        file_type=FileType.PDF,
        page_number=page,
        bbox=bbox,
        page_width=1000,
        page_height=1000,
        chunk_type=chunk_type,
    )


# bbox = [x_min, y_min, x_max, y_max]. v_gap = next.y_min - current.y_max.
# Horizontal overlap clause requires h_overlap/min_width > 0.4 - all fixtures
# below use the SAME x-span [100,400] so h_overlap == min_width (ratio 1.0),
# isolating the vertical-threshold behavior unless a test says otherwise.


def test_merges_when_vertical_gap_within_20(tmp_path):
    """gap == 20 (boundary, inclusive) AND horizontally aligned -> MERGE."""
    a = _chunk("First line of a wrapped sentence", [100, 100, 400, 200])
    b = _chunk("continues right below it", [100, 220, 400, 320])  # v_gap = 20
    out = _processor(tmp_path)._apply_spatial_refiner([a, b])
    text = [c for c in out if c.modality.value == "text"]
    assert len(text) == 1, "chunks 20 units apart and aligned must merge into one"
    assert "First line" in text[0].content and "continues" in text[0].content


def test_does_not_merge_when_vertical_gap_exceeds_20(tmp_path):
    """gap == 21 (just over threshold) -> NO MERGE. Pins the exact boundary."""
    a = _chunk("Heading paragraph", [100, 100, 400, 200])
    b = _chunk("A separate block far below", [100, 221, 400, 321])  # v_gap = 21
    out = _processor(tmp_path)._apply_spatial_refiner([a, b])
    text = [c for c in out if c.modality.value == "text"]
    assert len(text) == 2, "chunks 21 units apart exceed the 20-unit gate; stay split"


def test_does_not_merge_without_horizontal_overlap(tmp_path):
    """Within the 20-unit gap but side-by-side columns (no h-overlap) -> NO MERGE.

    The merge predicate is BOTH clauses: gap<=20 AND h_overlap/min_width>0.4.
    """
    a = _chunk("Left column text", [100, 100, 300, 200])
    b = _chunk("Right column text", [600, 210, 800, 310])  # v_gap=10 but disjoint x
    out = _processor(tmp_path)._apply_spatial_refiner([a, b])
    text = [c for c in out if c.modality.value == "text"]
    assert len(text) == 2, "non-overlapping columns must not merge even within 20 units"


def test_does_not_merge_across_pages(tmp_path):
    """Same geometry but different page numbers -> NO MERGE (page boundary)."""
    a = _chunk("End of page one", [100, 900, 400, 980], page=1)
    b = _chunk("Start of page two", [100, 10, 400, 90], page=2)
    out = _processor(tmp_path)._apply_spatial_refiner([a, b])
    text = [c for c in out if c.modality.value == "text"]
    assert len(text) == 2, "cross-page chunks must never merge regardless of geometry"


def test_does_not_merge_code_with_prose(tmp_path):
    """Code and prose chunks never merge even when geometrically adjacent."""
    code = _chunk("def f(x):", [100, 100, 400, 200], chunk_type=ChunkType.CODE)
    prose = _chunk("This explains the function.", [100, 215, 400, 315])  # v_gap=15
    out = _processor(tmp_path)._apply_spatial_refiner([code, prose])
    text = [c for c in out if c.modality.value in ("text", "code")]
    assert len(text) == 2, "code/prose boundary must be preserved (fidelity rule)"


def test_threshold_is_a_single_literal_20_no_profile_branches(tmp_path):
    """AGENT-SPATIAL-20 structural half: ONE 20-unit threshold, NO profile/heading
    branching in the refiner body.

    A source-level guard (cheap, durable) against the specific drift the
    invariant forbids: someone adding `if profile_type == ...:` or a second
    gap threshold inside the merge logic.
    """
    src = inspect.getsource(BatchProcessor._apply_spatial_refiner)
    # The 20-unit vertical gate is present exactly as the single threshold.
    assert "<= 20" in src, "the 20-unit vertical threshold literal is missing/changed"
    # No profile- or heading-conditional branching inside the geometry refiner.
    lowered = src.lower()
    for forbidden in ("profile_type", "document_domain", "parent_heading", "sensitivity"):
        assert forbidden not in lowered, (
            f"AGENT-SPATIAL-20 violated: '{forbidden}' branch found in the spatial "
            "refiner - the rule mandates a single geometry threshold with no "
            "profile/heading branches"
        )


def test_vertical_proximity_merger_is_a_thin_alias(tmp_path):
    """`_apply_vertical_proximity_merger` must delegate to the single refiner -
    so there is ONE merge implementation, not a drifting parallel copy."""
    a = _chunk("Alias line one", [100, 100, 400, 200])
    b = _chunk("alias line two", [100, 215, 400, 315])  # v_gap = 15 -> merge
    out = _processor(tmp_path)._apply_vertical_proximity_merger([a, b])
    text = [c for c in out if c.modality.value == "text"]
    assert len(text) == 1, "the alias must produce the same single-threshold merge"
