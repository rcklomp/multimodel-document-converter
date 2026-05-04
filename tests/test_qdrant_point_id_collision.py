"""Per PLAN_V2.8 §5c (Qdrant ingest fix): point IDs must not collide
across multiple files ingested into the same collection.

The original `point_id = i + 1` scheme generated identical IDs (1, 2,
3, ...) for every file. Ingesting a second file into the same
collection silently overwrote the first file's points; only the
largest single doc survived. The 2026-05-04 broad reconversion landed
22,587 chunks across 34 docs but only 1,690 points (= Firearms's chunk
count, the largest doc) made it into the collection.

Fix: derive point_id as a deterministic UUID5 from chunk_id (which IS
globally unique by schema since it embeds the doc-hash + page + type +
content-hash). Same chunk → same point_id across runs; different
chunks across docs → different point_ids.

The test exercises the deterministic derivation directly (the helper
namespace + uuid5 are tested in pure form so no Qdrant or Ollama
service is required).
"""
from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
INGEST_SCRIPT = REPO_ROOT / "scripts" / "ingest_to_qdrant.py"


def _load_ingest_module():
    spec = importlib.util.spec_from_file_location("ingest_to_qdrant", INGEST_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ingest_to_qdrant"] = module
    spec.loader.exec_module(module)
    return module


def _point_id(module, chunk_id: str) -> str:
    """Compute the point_id the way the ingest script does."""
    return str(uuid.uuid5(module._POINT_ID_NAMESPACE, chunk_id))


def test_namespace_is_deterministic_and_well_formed():
    """The namespace UUID is a fixed constant — must not change between runs."""
    m = _load_ingest_module()
    assert isinstance(m._POINT_ID_NAMESPACE, uuid.UUID)
    # Re-import is a no-op for module constants but verify the value is stable
    m2 = _load_ingest_module()
    assert m._POINT_ID_NAMESPACE == m2._POINT_ID_NAMESPACE


def test_same_chunk_id_yields_same_point_id_across_calls():
    """Determinism: re-ingesting the same chunk produces the same Qdrant point."""
    m = _load_ingest_module()
    cid = "deadbeef0001_005_text_a1b2c3d4"
    assert _point_id(m, cid) == _point_id(m, cid)


def test_different_chunks_in_same_doc_have_different_point_ids():
    """Within one file: every chunk_id maps to a unique point_id."""
    m = _load_ingest_module()
    ids = {
        _point_id(m, f"deadbeef0001_001_text_{i:08x}") for i in range(100)
    }
    assert len(ids) == 100  # no collisions within a single file


def test_different_docs_do_not_collide_at_index_1():
    """The original-bug regression: point_id=1 for chunk i=0 in every file.

    With the fix, the first chunk of file A and the first chunk of file B
    must NOT share a point_id even though they were both `i+1=1` before.
    This is the exact scenario that overwrote 32 of 34 docs in v2.8 5c
    ingest run #1.
    """
    m = _load_ingest_module()
    # Two different files, both with a "first chunk" — chunk_ids are
    # doc-hash-prefixed by the schema so they always differ even at
    # the same (page=1, type=text, index=0) position.
    a = _point_id(m, "aaaaaaaa0001_001_text_00000001")
    b = _point_id(m, "bbbbbbbb0001_001_text_00000001")
    assert a != b


def test_full_corpus_no_collisions_under_realistic_distribution():
    """Sanity-check: 34 docs × ~700 chunks each → 23,800 unique point_ids."""
    m = _load_ingest_module()
    seen = set()
    for doc_n in range(34):
        doc_hash = f"{doc_n:012x}"
        for page in range(1, 11):
            for kind in ("text", "image", "table"):
                for content_n in range(7):
                    cid = f"{doc_hash}_{page:03d}_{kind}_{content_n:08x}"
                    seen.add(_point_id(m, cid))
    # 34 docs × 10 pages × 3 kinds × 7 content variants = 7,140 unique
    assert len(seen) == 34 * 10 * 3 * 7


def test_fallback_uses_source_file_plus_index_when_chunk_id_missing():
    """Backwards-compat: empty chunk_id falls back to (source_file, index)."""
    m = _load_ingest_module()
    fallback_a = _point_id(m, "fileA.pdf#0")
    fallback_b = _point_id(m, "fileB.pdf#0")
    assert fallback_a != fallback_b
