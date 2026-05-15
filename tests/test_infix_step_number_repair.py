"""Phase 6 (PLAN_V2.10) audit follow-up — infix step-number repair.

OCR layout failures on multi-column numbered-instruction-list manuals
(Firearms-shape canonical) can mash an instruction-step number into the
trailing word of the preceding paragraph
(``"release the trigger 12. forsemgvaupwaros"``). The audit detector
``scripts/qa_conversion_audit.py::_INFIX_RE`` flags this as
``infix_artifacts``; the production fix
``BatchProcessor._repair_infix_step_numbers`` inserts a newline between
the preceding word and the step number so the chunk content reflects
the source manual's paragraph structure.

The contract enforced here:

1. The production repair detects the same shape as the audit gate (no
   drift between producer and consumer).
2. The repair preserves content character-for-character except for the
   single inserted ``\\n``.
3. The repair respects the audit's short-word and stop-word
   exclusions, so chunks the audit ignores are also left untouched by
   the repair.
4. A repaired chunk re-scored by the audit detector reports
   ``infix_artifacts == 0`` for that text.
"""
from __future__ import annotations

import re

import pytest

from mmrag_v2.batch_processor import BatchProcessor
from mmrag_v2.schema.ingestion_schema import (
    ChunkType,
    FileType,
    HierarchyMetadata,
    create_text_chunk,
)


# ---------------------------------------------------------------------------
# Audit-side detector reproduced verbatim from
# ``scripts/qa_conversion_audit.py``. The production repair MUST keep
# parity with this detector; the test re-applies the audit logic
# directly so the contract is enforced from both directions.
# ---------------------------------------------------------------------------

_AUDIT_INFIX_RE = re.compile(
    r"(?<![\n\r])(?<!^)"
    r"\b(?P<prev>[a-z][a-z'\-]{0,24})\s+"
    r"(?P<num>(?:[1-9]|[12]\d|3\d|40))\.\s+"
    r"(?P<next>[a-z][A-Za-z'\-]*)"
)


def _audit_infix_count(text: str) -> int:
    """Re-implementation of ``_count_infix_artifacts`` from the audit
    script. Kept inline so the test repository is the durable contract.
    """
    n = 0
    for m in _AUDIT_INFIX_RE.finditer(text or ""):
        prev = m.group("prev")
        nxt = m.group("next")
        start = m.start()
        left = (text or "")[max(0, start - 2):start]
        if left.endswith(("\n", "\r", ". ", ": ", "; ", "! ", "? ")):
            continue
        between = (text or "")[m.start("prev"):m.start("num")]
        if "\n" in between:
            continue
        if len(prev) <= 1 or len(nxt) <= 1:
            continue
        if prev in ("bis", "to", "from", "through", "vom", "von", "and", "or"):
            continue
        if nxt in ("bis", "to", "through"):
            continue
        n += 1
    return n


def _processor(tmp_path) -> BatchProcessor:
    return BatchProcessor(
        output_dir=str(tmp_path),
        batch_size=10,
        vision_provider="none",
        enable_ocr=False,
    )


def _text_chunk(content: str):
    return create_text_chunk(
        doc_id="firearms",
        content=content,
        source_file="Firearms.pdf",
        file_type=FileType.PDF,
        page_number=1,
        hierarchy=HierarchyMetadata(
            parent_heading=None,
            breadcrumb_path=["Firearms", "Page 1"],
            level=2,
        ),
        chunk_type=ChunkType.PARAGRAPH,
        extraction_method="ocr",
        position=1,
    )


# ---------------------------------------------------------------------------
# Positive contracts — audit-detected shapes from real Firearms output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_text",
    [
        # Canonical Firearms hits sampled from
        # `output/Firearms_phase6d/ingestion.jsonl` (verified via the
        # diagnostic probe).
        "that the bolt is fully to the rear 2. from the front of the bolt, "
        "counter-clockwise",
        "the cross-pin will release the trigger 12. forsemgvaupwaros",
        "etaining spring is mounted on the 9. end of the floorplate by a "
        "cross-pin",
        "the buttplate to give access to the 10. stock mounting bolt.",
        " the sear cross-pin, move the sear 15. forward, then remove it upward.",
        "bolt in the bolt carrier, be sure 1. the flat between the bolt is on top",
    ],
)
def test_infix_repair_inserts_newline_for_real_firearms_hits(tmp_path, raw_text):
    processor = _processor(tmp_path)
    chunk = _text_chunk(raw_text)

    # Pre-condition: the audit detector flags it.
    assert _audit_infix_count(raw_text) >= 1

    processor._repair_infix_step_numbers([chunk])

    # Post-condition: the audit detector no longer flags any hit on
    # the repaired content, AND the repaired content has exactly one
    # additional newline per hit.
    assert _audit_infix_count(chunk.content) == 0
    assert chunk.content.count("\n") == raw_text.count("\n") + _audit_infix_count(raw_text)


def test_infix_repair_preserves_byte_content_apart_from_newline(tmp_path):
    processor = _processor(tmp_path)
    raw = "release the trigger 12. forsemgvaupwaros"
    chunk = _text_chunk(raw)
    processor._repair_infix_step_numbers([chunk])

    # The only difference is the inserted newline.
    assert chunk.content.replace("\n", " ") == raw


def test_infix_repair_preserves_existing_paragraph_break(tmp_path):
    """If the source text already has a paragraph break (``\\n\\n``)
    after the numbered-step marker, the repair must still fire (the
    audit detector still flags it because its post-filter only
    excludes a newline BETWEEN prev and num), but the substitution
    preserves the existing separator after the period so two
    line-breaks are not collapsed to a single space.
    """
    processor = _processor(tmp_path)
    raw = "r block safety will be 4.\n\nfreed for removal from the left side"
    chunk = _text_chunk(raw)

    assert _audit_infix_count(raw) == 1

    processor._repair_infix_step_numbers([chunk])

    # Audit no longer flags it.
    assert _audit_infix_count(chunk.content) == 0
    # The existing `\n\n` separator after "4." is preserved
    # byte-for-byte; only a newline is INSERTED before "4".
    assert "4.\n\nfreed" in chunk.content
    assert "be\n4." in chunk.content


# ---------------------------------------------------------------------------
# Negative contracts — audit exclusions, no false-positive repairs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text_audit_ignores",
    [
        # Short prev word (≤1 char) — audit excludes, repair must too.
        "I 5. apples follow",
        # Short next word (≤1 char) — audit excludes, repair must too.
        "release the trigger 12. x",
        # Stop-word prev — audit excludes.
        "from 5. apples follow",
        "to 6. bananas follow",
        "and 7. cherries follow",
        # Stop-word next — audit excludes.
        "the train 5. through tunnels",
        # Pre-sentence-ending punctuation — audit excludes, repair must too.
        "Then it stops. trigger 12. forward step starts",
        # Number out of audit range (>40).
        "the trigger 41. forward",
        # Uppercase next word — audit excludes (next is [a-z] only).
        "release 12. Forward step",
        # Newline between prev word and number — audit "between" check
        # excludes (the structure already reflects a paragraph break).
        "release the trigger\n12. forward step",
    ],
)
def test_infix_repair_respects_audit_exclusions(tmp_path, text_audit_ignores):
    processor = _processor(tmp_path)
    chunk = _text_chunk(text_audit_ignores)

    # Pre-condition: the audit detector ignores it.
    assert _audit_infix_count(text_audit_ignores) == 0, (
        f"Audit detector unexpectedly flagged {text_audit_ignores!r}; "
        "the negative test premise is wrong"
    )

    processor._repair_infix_step_numbers([chunk])

    # Post-condition: the production repair also left it untouched.
    assert chunk.content == text_audit_ignores


def test_infix_repair_idempotent(tmp_path):
    """Running the repair twice yields the same content as running it
    once. The newline insertion makes the lookbehind `(?<![\\n\\r])`
    fire, so the second pass produces no further edits.
    """
    processor = _processor(tmp_path)
    raw = "release the trigger 12. forsemgvaupwaros"
    chunk = _text_chunk(raw)

    processor._repair_infix_step_numbers([chunk])
    after_first = chunk.content
    processor._repair_infix_step_numbers([chunk])
    assert chunk.content == after_first
    assert _audit_infix_count(chunk.content) == 0


def test_infix_repair_no_op_on_non_text_chunks(tmp_path):
    """The repair is text-content-only — image chunks pass through
    unchanged even if their (rarely-set) content text would otherwise
    match the pattern.
    """
    from mmrag_v2.schema.ingestion_schema import Modality, create_image_chunk

    processor = _processor(tmp_path)
    image_chunk = create_image_chunk(
        doc_id="firearms",
        content="release the trigger 12. forsemgvaupwaros",
        source_file="Firearms.pdf",
        file_type=FileType.PDF,
        page_number=1,
        asset_path="/tmp/x.png",
        bbox=[0, 0, 1000, 1000],
        page_width=1000,
        page_height=1000,
        hierarchy=HierarchyMetadata(
            parent_heading=None,
            breadcrumb_path=["Firearms", "Page 1"],
            level=2,
        ),
        extraction_method="docling",
        position=1,
    )
    assert image_chunk.modality == Modality.IMAGE
    snapshot = image_chunk.content

    processor._repair_infix_step_numbers([image_chunk])

    assert image_chunk.content == snapshot


# ---------------------------------------------------------------------------
# Audit-detector parity contract
# ---------------------------------------------------------------------------


def test_production_repair_matches_audit_detector_parity(tmp_path):
    """The production repair must zero-out the audit's
    ``infix_artifacts`` counter on the repaired content. This pin is a
    direct contract between the producer
    (``_repair_infix_step_numbers``) and the consumer
    (``scripts/qa_conversion_audit.py::_count_infix_artifacts``):
    every hit the audit detects, the production repair must close.
    """
    processor = _processor(tmp_path)

    # A composite Firearms-shape paragraph with multiple hits.
    raw = (
        "Open the bolt and move it part-way to the rear. Use a 1. small "
        "tool to depress the trigger 12. forsemgvaupwaros and check the "
        "stock 10. mounting bolt position."
    )

    audit_before = _audit_infix_count(raw)
    assert audit_before >= 2, (
        "Premise: synthetic example must produce ≥2 audit hits"
    )

    chunk = _text_chunk(raw)
    processor._repair_infix_step_numbers([chunk])

    assert _audit_infix_count(chunk.content) == 0


def test_production_repair_handles_python_cookbook_code_comment_shape(tmp_path):
    """The Python_Cookbook corpus has a small number of audit hits in
    code-comment text where a value-trailing period bumps against a
    code-call ("with width 3. print(...)"). The production repair
    treats these the same way the audit does — inserting a newline at
    the boundary so the chunk content's structure matches the source
    code comment's intended line break.
    """
    processor = _processor(tmp_path)
    raw = "This is a rectangle with length 5 and width 3. print(\"Area:\")"
    chunk = _text_chunk(raw)

    assert _audit_infix_count(raw) >= 1
    processor._repair_infix_step_numbers([chunk])
    assert _audit_infix_count(chunk.content) == 0
    # Content preserved minus the inserted newline:
    assert chunk.content.replace("\n", " ") == raw


# ---------------------------------------------------------------------------
# Pipeline-wiring contract
# ---------------------------------------------------------------------------


def test_repair_is_called_from_apply_final_boundary_repairs() -> None:
    """The repair must run as part of the final boundary repair pass
    (before hierarchy inference / heading propagation), not as an
    ad-hoc downstream step.
    """
    import inspect

    method_source = inspect.getsource(BatchProcessor._apply_final_boundary_repairs)
    assert "_repair_infix_step_numbers(" in method_source, (
        "Expected _apply_final_boundary_repairs to invoke "
        "_repair_infix_step_numbers"
    )
