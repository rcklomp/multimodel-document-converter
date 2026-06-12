"""PLAN_F1 Phase 0(b) modality-seam fix.

The chunk-level PyMuPDF indentation recovery in BatchProcessor._apply_code_hygiene
gated its candidate loop on `modality == Modality.TEXT`. V3 PROMOTES code chunks to
Modality.CODE, so the entire promoted population was skipped by construction and
the recovery never fired on it ("recovered indentation for 0 chunks" on born-digital
code books). The fix admits Modality.CODE code chunks while:
  - still excluding non-code modalities (IMAGE/TABLE/...) and non-code TEXT,
  - NOT re-extracting code that is already indented (the TEXT-only stamping loop
    never set indentation_fidelity on CODE chunks, so the recovery loop derives the
    flat/indented signal inline before deciding to fire).

These pin the seam open for CODE chunks and pin the no-regression guards.

No live PDF inference: `_recover_code_indentation_from_pdf` is monkeypatched to
record which chunks it is OFFERED. The text-only reflow step
(`_preserve_or_reflow_code_text`) is also stubbed to identity so it cannot
re-indent a flat TEXT fixture before recovery sees it - this test isolates the
recovery loop's admission decision, which is what the fix changed.
"""

from __future__ import annotations

import pytest

from mmrag_v2.batch_processor import BatchProcessor
from mmrag_v2.schema.ingestion_schema import (
    ChunkMetadata,
    ChunkType,
    FileType,
    IngestionChunk,
    Modality,
    create_text_chunk,
)

_FLAT_CODE = (
    "class Scheduler:\n"
    "def __init__(self, llm):\n"
    "self.llm = llm\n"
    "def stop(self):\n"
    "self.active = False"
)
_INDENTED_CODE = (
    "class Scheduler:\n"
    "    def __init__(self, llm):\n"
    "        self.llm = llm\n"
    "    def stop(self):\n"
    "        self.active = False"
)
_PROSE = (
    "The quick brown fox jumped over the lazy dog while the sun set slowly "
    "behind the distant hills, casting long shadows across the quiet meadow."
)


def _code_chunk(cid, content, modality):
    return IngestionChunk(
        chunk_id=cid,
        doc_id="doc_f1_seam",
        modality=modality,
        content=content,
        metadata=ChunkMetadata(
            source_file="doc.pdf",
            file_type=FileType.PDF,
            page_number=7,
            chunk_type=ChunkType.CODE,
            content_classification="code",
            extraction_method="uir_native_chunker",
            created_at="2026-06-12T00:00:00Z",
        ),
    )


@pytest.fixture
def offered(tmp_path, monkeypatch):
    """Return the chunk_ids offered to PyMuPDF recovery for a chunk set."""
    bp = BatchProcessor(output_dir=str(tmp_path / "out"))
    pdf = tmp_path / "dummy.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    bp._current_pdf_path = pdf
    # Isolate the recovery-loop admission: stop the TEXT-only reflow step from
    # re-indenting a flat fixture before recovery sees it.
    monkeypatch.setattr(bp, "_preserve_or_reflow_code_text", lambda t: t)

    def _run(chunks):
        seen: list[str] = []
        monkeypatch.setattr(
            bp,
            "_recover_code_indentation_from_pdf",
            lambda ch: (seen.append(ch.chunk_id), False)[1],
        )
        monkeypatch.setattr(bp, "_recover_fenced_code_blocks", lambda ch: False)
        bp._apply_code_hygiene(chunks)
        return seen

    return _run


def test_flat_code_modality_is_offered_to_recovery(offered):
    # The seam: a flat Modality.CODE chunk must now reach recovery.
    seen = offered([_code_chunk("flat_code_modality", _FLAT_CODE, Modality.CODE)])
    assert "flat_code_modality" in seen


def test_flat_code_text_still_offered(offered):
    # No regression: code smuggled as TEXT still reaches recovery (reflow isolated).
    seen = offered([_code_chunk("flat_code_text", _FLAT_CODE, Modality.TEXT)])
    assert "flat_code_text" in seen


def test_already_indented_code_modality_is_skipped(offered):
    # Guard: well-indented CODE chunks must NOT be re-extracted. CODE chunks carry
    # no stamped fidelity, so the inline derivation must protect them.
    seen = offered([_code_chunk("indented_code_modality", _INDENTED_CODE, Modality.CODE)])
    assert "indented_code_modality" not in seen


def test_non_code_text_not_offered(offered):
    # is_code_chunk guard: prose must not be sent to code recovery.
    prose = create_text_chunk(
        doc_id="doc_f1_seam",
        content=_PROSE,
        source_file="doc.pdf",
        file_type=FileType.PDF,
        page_number=7,
        bbox=[100, 100, 900, 200],
        page_width=612,
        page_height=792,
        position=0,
    )
    assert offered([prose]) == []


def test_mixed_set_only_flat_code_offered(offered):
    seen = offered(
        [
            _code_chunk("flat_code_modality", _FLAT_CODE, Modality.CODE),
            _code_chunk("indented_code_modality", _INDENTED_CODE, Modality.CODE),
            _code_chunk("flat_code_text", _FLAT_CODE, Modality.TEXT),
        ]
    )
    assert set(seen) == {"flat_code_modality", "flat_code_text"}
