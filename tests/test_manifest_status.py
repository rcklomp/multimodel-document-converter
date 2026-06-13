"""Unit tests for scripts/manifest_status.py (PLAN_F1 WP-B / D1).

Pure classification only - no Qdrant, no filesystem reads of source files. The
single IO seam (current_sha256) is supplied to classify_row as an argument, so
every case here is deterministic and offline.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "manifest_status",
    Path(__file__).resolve().parents[1] / "scripts" / "manifest_status.py",
)
ms = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ms)  # type: ignore[union-attr]


def _row(outcome="INGESTED", points_dense=10, sha="aaa", engine="mineru_qwen_hybrid"):
    return {
        "doc_id": "deadbeef0000",
        "source_path": "data/x/y.pdf",
        "sha256": sha,
        "pages": 10,
        "extraction": {"engine": engine},
        "ingest": {"outcome": outcome, "points_dense": points_dense},
    }


def test_current_when_ingested_fresh_and_production_engine():
    assert ms.classify_row(_row(sha="aaa"), current_sha="aaa") == ms.STATUS_CURRENT


def test_stale_when_source_sha_differs():
    assert ms.classify_row(_row(sha="aaa"), current_sha="bbb") == ms.STATUS_STALE


def test_pending_when_not_ingested():
    assert (
        ms.classify_row(_row(outcome="PENDING", points_dense=0), current_sha="aaa")
        == ms.STATUS_PENDING
    )


def test_pending_when_zero_dense_points_despite_ingested_label():
    # outcome says INGESTED but no dense points -> not really ingested.
    assert (
        ms.classify_row(_row(outcome="INGESTED", points_dense=0), current_sha="aaa")
        == ms.STATUS_PENDING
    )


def test_pending_when_below_standard_engine():
    row = _row(engine="docling_fast")
    assert ms.classify_row(row, current_sha="aaa") == ms.STATUS_PENDING


def test_failed_outcomes_map_to_failed():
    assert ms.classify_row(_row(outcome="CONTENT_FAIL"), current_sha="aaa") == ms.STATUS_FAILED
    assert ms.classify_row(_row(outcome="LADDER_FAIL"), current_sha="aaa") == ms.STATUS_FAILED


def test_stale_dominates_below_standard():
    # File changed AND below-standard engine -> STALE (must re-extract regardless).
    row = _row(sha="aaa", engine="docling_fast")
    assert ms.classify_row(row, current_sha="bbb") == ms.STATUS_STALE


def test_no_sha_check_treats_ingested_as_current():
    # When current_sha is None (sha check skipped), an ingested production row
    # cannot be flagged stale - it stays CURRENT.
    assert ms.classify_row(_row(sha="aaa"), current_sha=None) == ms.STATUS_CURRENT


def test_missing_extraction_block_is_below_standard():
    row = {
        "sha256": "aaa",
        "ingest": {"outcome": "INGESTED", "points_dense": 5},
        "extraction": None,
    }
    assert ms.classify_row(row, current_sha="aaa") == ms.STATUS_PENDING
