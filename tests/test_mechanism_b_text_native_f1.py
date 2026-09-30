"""PLAN_F1 WP-2: Mechanism B generalization (text-native code pages).

On a born-digital text_native_code page the PDF text layer is authoritative, so
EVERY code chunk on that page is re-served from the text-layer clip - including
already-indented chunks (overriding the c95950b "skip if already indented" guard).
OFF a text-native page, c95950b is retained: only flat chunks are attempted.

These build a REAL 2-page PDF (page 1 = Python code -> text_native; page 2 =
prose -> not native) and assert which code chunks the recovery is OFFERED. The
reconstruction itself is monkeypatched to record offers (the offer decision is
exactly what WP-2 changes); no live inference.
"""

from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz")

from mmrag_v2.batch_processor import BatchProcessor
from mmrag_v2.schema.ingestion_schema import (
    ChunkMetadata,
    ChunkType,
    FileType,
    IngestionChunk,
    Modality,
)

_INDENTED_CODE = (
    "class Scheduler:\n"
    "    def __init__(self, llm):\n"
    "        self.llm = llm\n"
    "    def stop(self):\n"
    "        self.active = False"
)
_FLAT_CODE = "class Scheduler:\ndef __init__(self, llm):\nself.llm = llm"


def _make_pdf(path):
    doc = fitz.open()
    # Page 1: dense Python code -> text_native_code fires (c2_kw high, depths>=2).
    p1 = doc.new_page()
    code = [
        "import torch", "import torch.nn as nn", "class Net(nn.Module):",
        "    def __init__(self):", "        super().__init__()",
        "        self.fc = nn.Linear(10, 2)", "    def forward(self, x):",
        "        return self.fc(x)", "def train(model, loader):",
        "    for batch in loader:", "        loss = model(batch)",
        "        return loss",
    ]
    y = 72
    for ln in code:
        p1.insert_text((72, y), ln, fontsize=10, fontname="cour")
        y += 14
    # Page 2: prose -> not text_native_code.
    p2 = doc.new_page()
    y = 72
    for ln in ("The aircraft entered service in 2005 and remains unmatched.",
               "Avionics include advanced radar and electronic warfare suites.",
               "The deployment marked the first carrier qualification cycle.") * 4:
        p2.insert_text((72, y), ln, fontsize=10)
        y += 14
    doc.save(str(path))
    doc.close()


def _code_chunk(cid, content, page):
    return IngestionChunk(
        chunk_id=cid,
        doc_id="doc_wp2",
        modality=Modality.CODE,
        content=content,
        metadata=ChunkMetadata(
            source_file="doc.pdf",
            file_type=FileType.PDF,
            page_number=page,
            chunk_type=ChunkType.CODE,
            content_classification="code",
            extraction_method="uir_native_chunker",
            created_at="2026-06-12T00:00:00Z",
        ),
    )


@pytest.fixture
def offered(tmp_path, monkeypatch):
    pdf = tmp_path / "book.pdf"
    _make_pdf(pdf)
    bp = BatchProcessor(output_dir=str(tmp_path / "out"))
    bp._current_pdf_path = pdf
    monkeypatch.setattr(bp, "_preserve_or_reflow_code_text", lambda t: t)
    monkeypatch.setattr(bp, "_recover_fenced_code_blocks", lambda ch: False)

    def _run(chunks):
        seen: list[str] = []
        monkeypatch.setattr(
            bp, "_recover_code_indentation_from_pdf",
            lambda ch: (seen.append(ch.chunk_id), False)[1],
        )
        bp._apply_code_hygiene(chunks)
        return seen

    return _run


def test_indented_code_on_text_native_page_is_re_served(offered):
    # WP-2 override: already-indented code on a code page IS offered (text layer wins).
    seen = offered([_code_chunk("p1_indented", _INDENTED_CODE, 1)])
    assert "p1_indented" in seen


def test_indented_code_on_prose_page_is_skipped(offered):
    # c95950b retained off text-native pages: already-indented code is NOT re-served.
    seen = offered([_code_chunk("p2_indented", _INDENTED_CODE, 2)])
    assert "p2_indented" not in seen


def test_flat_code_on_prose_page_still_offered(offered):
    # No regression: a flat code chunk is offered regardless of page nativeness.
    seen = offered([_code_chunk("p2_flat", _FLAT_CODE, 2)])
    assert "p2_flat" in seen


def test_mixed(offered):
    seen = offered([
        _code_chunk("p1_indented", _INDENTED_CODE, 1),
        _code_chunk("p2_indented", _INDENTED_CODE, 2),
        _code_chunk("p2_flat", _FLAT_CODE, 2),
    ])
    assert set(seen) == {"p1_indented", "p2_flat"}
