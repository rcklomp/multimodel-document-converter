from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from mmrag_v2.schema.ingestion_schema import (
    ChunkType,
    FileType,
    IngestionMetadata,
    create_image_chunk,
    create_table_chunk,
    create_text_chunk,
)

SCRIPT_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
try:
    import qa_universal_invariants
finally:
    sys.path.pop(0)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _metadata(chunk_count: int) -> dict:
    return IngestionMetadata(
        schema_version="2.7.0",
        doc_id="coordaudit01",
        source_file="coordinate-audit.pdf",
        profile_type="technical_manual",
        chunk_count=chunk_count,
        ingestion_timestamp="2026-04-30T00:00:00+00:00",
    ).model_dump(mode="json")


def _text_row(chunk_id: str, spatial: dict | None = None) -> dict:
    metadata = {
        "source_file": "coordinate-audit.pdf",
        "file_type": "pdf",
        "page_number": 1,
        "chunk_type": "paragraph",
    }
    if spatial is not None:
        metadata["spatial"] = spatial
    return {
        "chunk_id": chunk_id,
        "doc_id": "coordaudit01",
        "modality": "text",
        # v2.9: per-fixture-distinct content so the within-page dupe
        # gate in qa_universal_invariants doesn't flag synthetic test
        # data as duplicate text emissions.
        "content": f"Text content for {chunk_id}.",
        "metadata": metadata,
    }


def test_bbox_distribution_stats_from_current_schema_bridge(tmp_path: Path) -> None:
    """Schema factory output uses metadata.spatial and must reach the audit stats."""
    text = create_text_chunk(
        doc_id="coordaudit01",
        content="A paragraph with page coordinates.",
        source_file="coordinate-audit.pdf",
        file_type=FileType.PDF,
        page_number=1,
        chunk_type=ChunkType.PARAGRAPH,
        bbox=[100, 100, 500, 300],
        page_width=612,
        page_height=792,
    )
    image = create_image_chunk(
        doc_id="coordaudit01",
        content="Visual description.",
        source_file="coordinate-audit.pdf",
        file_type=FileType.PDF,
        page_number=1,
        asset_path="assets/image.png",
        bbox=[50, 40, 250, 240],
        page_width=612,
        page_height=792,
    )
    table = create_table_chunk(
        doc_id="coordaudit01",
        content="| A | B |\n| - | - |\n| 1 | 2 |",
        source_file="coordinate-audit.pdf",
        file_type=FileType.PDF,
        page_number=1,
        bbox=[300, 400, 700, 650],
        asset_path="assets/table.png",
        page_width=612,
        page_height=792,
    )
    rows = [
        _metadata(3),
        text.model_dump(mode="json"),
        image.model_dump(mode="json"),
        table.model_dump(mode="json"),
    ]
    result = qa_universal_invariants.check(_write_jsonl(tmp_path / "ingestion.jsonl", rows))

    assert result.invalid_bbox == 0
    assert result.bbox_missing_dims == 0
    assert result.bbox_stats["text"].with_bbox == 1
    assert result.bbox_stats["image"].with_bbox == 1
    assert result.bbox_stats["table"].with_bbox == 1
    assert result.bbox_stats["text"].min_area == 80000
    assert result.bbox_stats["image"].avg_area == pytest.approx(40000.0)
    assert result.bbox_stats["table"].max_area == 100000


def test_bbox_audit_supports_legacy_spatial_metadata_key(tmp_path: Path) -> None:
    rows = [
        _metadata(1),
        {
            "chunk_id": "legacy_spatial",
            "doc_id": "coordaudit01",
            "modality": "image",
            "content": "Legacy JSONL image.",
            "metadata": {
                "page_number": 1,
                "spatial_metadata": {
                    "bbox": [10, 20, 110, 220],
                    "page_width": 612,
                    "page_height": 792,
                },
            },
        },
    ]
    result = qa_universal_invariants.check(_write_jsonl(tmp_path / "legacy.jsonl", rows))

    assert result.invalid_bbox == 0
    assert result.bbox_stats["image"].with_bbox == 1
    assert result.bbox_stats["image"].min_area == 20000


@pytest.mark.parametrize(
    "bbox",
    [
        [0.1, 0, 1, 1],
        [0, 0, 1001, 1],
        [10, 10, 10, 20],
        [10, 10, 20],
        "10,10,20,20",
        [True, 0, 20, 20],
    ],
)
def test_invalid_bbox_shapes_and_values_fail_universal_check(
    tmp_path: Path, bbox: object
) -> None:
    rows = [
        _metadata(1),
        _text_row("bad_bbox", {"bbox": bbox, "page_width": 612, "page_height": 792}),
    ]
    result = qa_universal_invariants.check(_write_jsonl(tmp_path / "bad.jsonl", rows))

    assert result.invalid_bbox == 1
    assert result.bbox_stats["text"].invalid_bbox == 1
    assert any("invalid bbox" in example for example in result.examples)


def test_bbox_audit_reports_missing_page_dimensions_as_warning_only(tmp_path: Path) -> None:
    rows = [
        _metadata(1),
        _text_row("missing_dims", {"bbox": [10, 20, 110, 220]}),
    ]
    result = qa_universal_invariants.check(_write_jsonl(tmp_path / "missing_dims.jsonl", rows))

    assert result.invalid_bbox == 0
    assert result.bbox_missing_dims == 1
    assert result.bbox_stats["text"].bbox_missing_dims == 1


def test_universal_invariants_cli_prints_bbox_distribution_stats(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [
        _metadata(2),
        _text_row("with_bbox", {"bbox": [10, 20, 110, 220], "page_width": 612, "page_height": 792}),
        _text_row("without_bbox"),
    ]
    jsonl = _write_jsonl(tmp_path / "cli.jsonl", rows)

    assert qa_universal_invariants.main(["qa_universal_invariants.py", str(jsonl)]) == 0
    out = capsys.readouterr().out

    assert "bbox_stats[text]: total=2 with_bbox=1 missing_bbox=1 invalid_bbox=0" in out
    assert "x_range=10..110" in out
    assert "area_min=20000 area_avg=20000.0 area_max=20000" in out
    assert "UNIVERSAL_PASS" in out


def test_universal_invariants_cli_fails_for_bad_bbox(tmp_path: Path) -> None:
    rows = [
        _metadata(1),
        _text_row("bad_bbox", {"bbox": [10, 20, 1200, 220], "page_width": 612, "page_height": 792}),
    ]
    jsonl = _write_jsonl(tmp_path / "cli_bad.jsonl", rows)

    assert qa_universal_invariants.main(["qa_universal_invariants.py", str(jsonl)]) == 1


@pytest.mark.parametrize(
    ("bbox", "expected"),
    [
        ([412, 636, 412, 644], [412, 636, 413, 644]),
        ([30, 1000, 1000, 1000], [30, 999, 1000, 1000]),
        ([1000, 1000, 1000, 1000], [999, 999, 1000, 1000]),
        ([0.5, 1.0, 1.0, 1.0], [500, 999, 1000, 1000]),
    ],
)
def test_ensure_normalized_repairs_zero_extent_without_leaving_canvas(
    bbox: list[int | float], expected: list[int]
) -> None:
    from mmrag_v2.utils.coordinate_normalization import ensure_normalized

    assert ensure_normalized(bbox, context="zero_extent_regression") == expected
