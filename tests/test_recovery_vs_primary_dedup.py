"""Recovery-vs-primary dedup in BatchProcessor.

The TextIntegrityScout pulls whole-page text from the flush-left PDF text layer
to rescue genuinely-missing text. On code/dense pages the VLM already extracted
the content cleanly, but the scout re-adds it as flush-left recovery chunks,
producing mangled duplicates (output bloat + R3 code-indentation pollution).
Observed live on AIOS via Qwen: each code listing appeared once as a clean
modality=code chunk AND again as recovery_gap_fill/recovery_scan text.

`_apply_recovery_vs_primary_dedup` drops a recovery text chunk only when >80% of
its unique tokens are already in the primary chunks on its page; genuinely-new
recovery text and recovery on VLM-dropped pages survive. These pin both.
"""

from __future__ import annotations

from pathlib import Path

from mmrag_v2.batch_processor import BatchProcessor
from mmrag_v2.schema.ingestion_schema import (
    ChunkMetadata,
    FileType,
    IngestionChunk,
    Modality,
)

_CLEAN_CODE = (
    "class Scheduler:\n"
    "    def __init__(self, llm, memory_manager, storage_manager, tool_manager):\n"
    "        self.llm = llm\n"
    "        self.memory_manager = memory_manager\n"
    "        self.storage_manager = storage_manager\n"
    "        self.tool_manager = tool_manager\n"
    "        self.active = True\n"
    "    def start_processors(self):\n"
    "        for name, processor in self.request_processors.items():\n"
    "            processor.start()\n"
    "    def stop(self):\n"
    "        self.active = False\n"
    "        return None"
)
# Same code, flush-left, with a leaked page header — what the scout re-pulls.
_RECOVERED_DUP = "AIOS: LLM Agent Operating System\n" + "\n".join(
    ln.lstrip() for ln in _CLEAN_CODE.splitlines()
)
_GENUINE_NEW = (
    "Figure 3 illustrates the end-to-end latency measured across the evaluated "
    "agent frameworks under increasing concurrency, with error bars over five runs."
)


def _chunk(cid, content, modality, method, page):
    return IngestionChunk(
        chunk_id=cid,
        doc_id="doc_recovery_test",
        modality=modality,
        content=content,
        metadata=ChunkMetadata(
            source_file="doc.pdf",
            file_type=FileType.PDF,
            page_number=page,
            extraction_method=method,
            created_at="2026-06-06T00:00:00Z",
        ),
    )


def _bp(tmp_path: Path) -> BatchProcessor:
    return BatchProcessor(output_dir=str(tmp_path / "out"))


def test_recovery_duplicate_of_primary_code_is_dropped(tmp_path):
    primary = _chunk("c_primary", _CLEAN_CODE, Modality.CODE, "uir_native_chunker", 18)
    dup = _chunk("c_recovery_dup", _RECOVERED_DUP, Modality.TEXT, "recovery_gap_fill", 18)
    out = _bp(tmp_path)._apply_recovery_vs_primary_dedup([primary, dup])
    ids = {c.chunk_id for c in out}
    assert "c_primary" in ids
    assert "c_recovery_dup" not in ids  # flush-left duplicate dropped


def test_genuine_new_recovery_text_is_kept(tmp_path):
    # Page has primary code, but the recovery chunk is genuinely-missing prose
    # (low token overlap) — it must survive.
    primary = _chunk("c_primary", _CLEAN_CODE, Modality.CODE, "uir_native_chunker", 18)
    new = _chunk("c_recovery_new", _GENUINE_NEW, Modality.TEXT, "recovery_scan", 18)
    out = _bp(tmp_path)._apply_recovery_vs_primary_dedup([primary, new])
    ids = {c.chunk_id for c in out}
    assert ids == {"c_primary", "c_recovery_new"}


def test_recovery_on_vlm_dropped_page_is_kept(tmp_path):
    # No primary chunk on page 42 (the VLM dropped it entirely) — the scout's
    # legitimate purpose; the recovery chunk must always survive.
    primary = _chunk("c_primary", _CLEAN_CODE, Modality.CODE, "uir_native_chunker", 18)
    rescue = _chunk("c_rescue", _RECOVERED_DUP, Modality.TEXT, "recovery_gap_fill", 42)
    out = _bp(tmp_path)._apply_recovery_vs_primary_dedup([primary, rescue])
    ids = {c.chunk_id for c in out}
    assert "c_rescue" in ids  # different page, no primary to dedup against
