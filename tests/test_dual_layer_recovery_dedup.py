"""Dual-layer recovery-vs-primary dedup contract (PR #4 Finding 4 follow-up).

The same domain fact - "the recovery scout re-pulls code the VLM already
extracted cleanly, producing a flush-left duplicate" - is defended in TWO places
with DIFFERENT algorithms:
  - PRODUCTION: BatchProcessor._apply_recovery_vs_primary_dedup (token-set
    overlap >= 85%, per page) drops the duplicate before it reaches the JSONL;
  - AUDIT: _code_quality._duplicates_primary (substring-window matching) excludes
    it from the R3 indentation metric for files already on disk.

The PR #4 review noted the cross-link is comments-only ("comments don't
enforce"). A "the two always agree" test would be WRONG - they are intentionally
different algorithms (defense in depth) and may legitimately diverge on edge
cases. The enforceable contract is instead: BOTH layers must still catch the
canonical duplicate. This regression test pins exactly that, so a change that
silently breaks either layer fails here.

Offline/deterministic.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from mmrag_v2.batch_processor import BatchProcessor
from mmrag_v2.schema.ingestion_schema import (
    ChunkType,
    FileType,
    Modality,
    create_text_chunk,
)

# The canonical case: a clean, indented code listing, and a flush-left recovery
# re-pull of the SAME code (same tokens/chars, mangled indentation).
CLEAN = (
    "def process(items):\n"
    "    for x in items:\n"
    "        if x.ready:\n"
    "            handle(x)\n"
    "    return done"
)
FLUSH = (
    "def process(items):\n"
    "for x in items:\n"
    "if x.ready:\n"
    "handle(x)\n"
    "return done"
)


def _load_code_quality():
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_code_quality", repo / "scripts" / "_code_quality.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_code_quality"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_production_layer_drops_recovery_duplicate(tmp_path):
    bp = BatchProcessor(output_dir=str(tmp_path), vision_provider="none")
    primary = create_text_chunk(
        doc_id="d", content=CLEAN, source_file="d.pdf", file_type=FileType.PDF,
        page_number=5, chunk_type=ChunkType.CODE, extraction_method="uir_native_chunker",
        position=0,
    )
    recovery = create_text_chunk(
        doc_id="d", content=FLUSH, source_file="d.pdf", file_type=FileType.PDF,
        page_number=5, extraction_method="recovery_scan", position=1,
    )
    out = bp._apply_recovery_vs_primary_dedup([primary, recovery])
    contents = [c.content for c in out]
    assert CLEAN in contents  # the clean primary survives
    assert FLUSH not in contents  # the flush-left recovery duplicate is dropped


def test_audit_layer_excludes_recovery_duplicate():
    cq_mod = _load_code_quality()
    rows = [
        {"chunk_id": "c1", "modality": "code", "content": CLEAN},
        {
            "chunk_id": "t1",
            "modality": "text",
            "content": FLUSH,
            "metadata": {"content_classification": "code"},
        },
    ]
    cq = cq_mod.code_quality(rows)
    # The flush-left text-as-code duplicate is excluded, not scored as a code
    # indentation failure.
    assert cq.n_duplicate_excluded == 1


def test_recovery_on_vlm_dropped_page_is_kept(tmp_path):
    # Guard the other half of the domain fact: a recovery chunk on a page with NO
    # primary (the VLM dropped the page) is the scout's legitimate rescue - kept.
    bp = BatchProcessor(output_dir=str(tmp_path), vision_provider="none")
    recovery = create_text_chunk(
        doc_id="d", content=CLEAN, source_file="d.pdf", file_type=FileType.PDF,
        page_number=9, extraction_method="recovery_scan", position=0,
    )
    primary_other = create_text_chunk(
        doc_id="d", content="Unrelated body.", source_file="d.pdf",
        file_type=FileType.PDF, page_number=1, extraction_method="uir_native_chunker",
        position=1,
    )
    out = bp._apply_recovery_vs_primary_dedup([primary_other, recovery])
    assert any(c.content == CLEAN for c in out)  # rescue kept (no primary on p9)
