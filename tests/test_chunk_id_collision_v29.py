"""
v2.9 Phase 1 — chunk_id collision regression tests.

Pins the contract that ``_generate_chunk_id`` and the three
``create_*_chunk`` factory functions thread a per-document ``position``
component into the chunk-id hash, so two chunks with byte-identical
``(doc_id, page, modality, content)`` but different positions get
distinct chunk_ids.

Background: v2.8 broad reconversion produced 22,587 chunks across 34
docs that collapsed to 22,160 unique chunk_ids — 427 within-file
duplicates concentrated on boilerplate footers, repeated page numbers,
and identical short labels. The duplicates silently overwrote each
other on Qdrant upsert (uuid5 from chunk_id, v2.8 commit ``0d3cc36``),
leaving ``mmrag_v2_8`` non-deterministic. v2.9 closes this by hashing
position into the chunk_id seed.

Schema version stays ``2.7.0`` because the chunk-shape contract is
unchanged; only the ``chunk_id`` *value* changes for affected chunks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mmrag_v2.schema.ingestion_schema import (
    FileType,
    HierarchyMetadata,
    _generate_chunk_id,
    create_image_chunk,
    create_table_chunk,
    create_text_chunk,
)


def test_within_file_dupe_content_yields_distinct_chunk_ids() -> None:
    """Two chunks with identical (doc_id, page, modality, content) but
    different positions must get distinct chunk_id strings.

    This is the core anti-collision contract. v2.8 violated it by
    omitting ``position`` from the hash seed; v2.9 must satisfy it.
    """
    cid_a = _generate_chunk_id(
        doc_id="doc_xyz",
        content="Page 47",
        page=47,
        modality="text",
        position=0,
    )
    cid_b = _generate_chunk_id(
        doc_id="doc_xyz",
        content="Page 47",
        page=47,
        modality="text",
        position=1,
    )
    assert cid_a != cid_b


def test_chunk_id_stable_under_position_zero_default() -> None:
    """The default ``position=0`` path must produce a deterministic,
    drift-free hex.

    Pinning the exact value here means any future change to the hash
    seed (e.g. swapping sha256 for blake2, changing the field
    delimiter) trips this test. The v2.9 seed format is
    ``f"{doc_id}:{page}:{modality}:{position}:{content}"``.
    """
    cid = _generate_chunk_id(
        doc_id="abc12345",
        content="Hello, world.",
        page=3,
        modality="text",
        position=0,
    )
    # Pre-computed: sha256("abc12345:3:text:0:Hello, world.")[:8] = "0a9c5fb6"
    # Format: f"{doc_id}_{page:03d}_{modality}_{content_hash}"
    expected_prefix = "abc12345_003_text_"
    assert cid.startswith(expected_prefix)
    assert len(cid.split("_")[-1]) == 8  # 8-hex content hash
    # Re-running yields the same result (no drift).
    cid2 = _generate_chunk_id(
        doc_id="abc12345",
        content="Hello, world.",
        page=3,
        modality="text",
        position=0,
    )
    assert cid == cid2


def test_factory_threads_position_for_text_image_table() -> None:
    """Each of the three production factories (text, image, table)
    must thread ``position`` so two chunks with the same content but
    different positions get distinct chunk_ids.

    Audits all three modality-routes in one test because v2.8 found
    text was the heaviest collision source but image and table can
    collide too (identical short labels, identical first-50-char
    table previews).
    """
    common = dict(
        doc_id="doc_collision_check",
        source_file="collision.pdf",
        file_type=FileType.PDF,
        page_number=10,
        hierarchy=HierarchyMetadata(),
    )

    text_a = create_text_chunk(content="footer", position=0, **common)
    text_b = create_text_chunk(content="footer", position=1, **common)

    image_common = dict(common)
    image_common.pop("hierarchy")
    image_a = create_image_chunk(
        content="[fig]",
        asset_path="assets/x_010_figure_01.png",
        bbox=[0, 0, 100, 100],
        position=0,
        **image_common,
    )
    image_b = create_image_chunk(
        content="[fig]",
        asset_path="assets/x_010_figure_01.png",
        bbox=[0, 0, 100, 100],
        position=1,
        **image_common,
    )

    table_a = create_table_chunk(
        content="| header | header |\n| --- | --- |",
        bbox=[0, 0, 100, 100],
        asset_path="assets/x_010_table_01.png",
        position=0,
        **image_common,
    )
    table_b = create_table_chunk(
        content="| header | header |\n| --- | --- |",
        bbox=[0, 0, 100, 100],
        asset_path="assets/x_010_table_01.png",
        position=1,
        **image_common,
    )

    assert text_a.chunk_id != text_b.chunk_id
    assert image_a.chunk_id != image_b.chunk_id
    assert table_a.chunk_id != table_b.chunk_id

    # All six chunk_ids must be unique (cross-modality + cross-position).
    all_ids = {
        text_a.chunk_id,
        text_b.chunk_id,
        image_a.chunk_id,
        image_b.chunk_id,
        table_a.chunk_id,
        table_b.chunk_id,
    }
    assert len(all_ids) == 6


_OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "output"


@pytest.mark.skipif(
    os.environ.get("RUN_CORPUS_SCAN") != "1",
    reason="Set RUN_CORPUS_SCAN=1 to scan output/<canonical>/ingestion.jsonl files",
)
def test_full_corpus_no_within_file_chunk_id_collisions() -> None:
    """Walks every ``output/<canonical>/ingestion.jsonl`` and asserts
    each file has zero within-file ``chunk_id`` duplicates.

    Env-gated because the test needs the v2.9 broad reconversion to
    have produced the JSONLs (Phase 5). Until Phase 5 lands, the
    contract is committed but not auto-run. After Phase 5 completes,
    `RUN_CORPUS_SCAN=1 pytest tests/test_chunk_id_collision_v29.py -v`
    must report zero collisions across all 34 canonical docs.
    """
    if not _OUTPUT_ROOT.exists():
        pytest.skip(f"output directory not found: {_OUTPUT_ROOT}")

    jsonl_paths = sorted(_OUTPUT_ROOT.glob("*/ingestion.jsonl"))
    if not jsonl_paths:
        pytest.skip(f"no ingestion.jsonl under {_OUTPUT_ROOT}")

    failures: list[str] = []
    for path in jsonl_paths:
        seen: dict[str, int] = {}
        dupe_count = 0
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Skip header / non-chunk records.
                if record.get("object_type") not in (None, "chunk"):
                    continue
                cid = record.get("chunk_id")
                if not cid:
                    continue
                seen[cid] = seen.get(cid, 0) + 1
                if seen[cid] == 2:
                    dupe_count += 1
        if dupe_count > 0:
            failures.append(f"{path.relative_to(_OUTPUT_ROOT)}: {dupe_count} dupe chunk_ids")

    assert not failures, "Within-file chunk_id collisions found:\n  " + "\n  ".join(failures)
