"""Env-gated acceptance checks for Phase 1 TOC/index page-window probes."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


RUN_TOC_CONTRACT = os.environ.get("RUN_TOC_PAGE_CONTRACT") == "1"


def _require_contract_run() -> None:
    if not RUN_TOC_CONTRACT:
        pytest.skip("set RUN_TOC_PAGE_CONTRACT=1 to validate generated TOC probes")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_page_summary(relative_path: str):
    path = _repo_root() / relative_path
    if not path.exists():
        raise AssertionError(f"missing probe output: {relative_path}")

    pages = {}
    methods = {}
    empty_text_chunks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("object_type") == "ingestion_metadata":
            continue
        metadata = row.get("metadata") or {}
        page = metadata.get("page_number") or row.get("page")
        if not page:
            continue
        page = int(page)
        pages[page] = pages.get(page, 0) + 1
        method = metadata.get("extraction_method") or row.get("extraction_method")
        if method:
            methods.setdefault(page, set()).add(method)
        if row.get("modality") == "text" and not (row.get("content") or "").strip():
            empty_text_chunks.append(row.get("chunk_id"))

    return pages, methods, empty_text_chunks


@pytest.mark.parametrize(
    ("relative_path", "expected_pages"),
    [
        (
            "output/probe_kimothi_toc_contract_codex/ingestion.jsonl",
            set(range(1, 16)) | (set(range(245, 256)) - {248}),
        ),
        (
            "output/probe_hao_toc_contract_codex/ingestion.jsonl",
            set(range(1, 16)) | set(range(496, 504)),
        ),
        (
            "output/probe_python_cookbook_toc_contract_codex/ingestion.jsonl",
            set(range(1, 16)),
        ),
        (
            "output/probe_ayeva_index_contract_codex/ingestion.jsonl",
            set(range(281, 296)),
        ),
    ],
)
def test_toc_index_probe_pages_emit_chunks(relative_path, expected_pages):
    _require_contract_run()

    pages, methods, empty_text_chunks = _load_page_summary(relative_path)

    missing = sorted(page for page in expected_pages if pages.get(page, 0) == 0)
    recovery_pages = sorted(
        page for page, page_methods in methods.items()
        if "recovery_page_coverage" in page_methods
    )
    assert missing == []
    assert recovery_pages == []
    assert empty_text_chunks == []


def test_ayeva_dense_index_pages_use_pageskip_only():
    _require_contract_run()

    _, methods, _ = _load_page_summary(
        "output/probe_ayeva_index_contract_codex/ingestion.jsonl"
    )

    for page in (285, 286, 289, 290):
        assert methods.get(page) == {"hybrid_chunker_pageskip"}


def test_kimothi_toc_probe_page_counts_are_deterministic():
    _require_contract_run()

    first_pages, first_methods, _ = _load_page_summary(
        "output/probe_kimothi_toc_contract_codex/ingestion.jsonl"
    )
    rerun_pages, rerun_methods, _ = _load_page_summary(
        "output/probe_kimothi_toc_contract_codex_rerun/ingestion.jsonl"
    )
    expected_pages = set(range(1, 16)) | set(range(245, 256))

    assert set(first_pages) == set(rerun_pages)
    assert {page: first_pages.get(page, 0) for page in expected_pages} == {
        page: rerun_pages.get(page, 0) for page in expected_pages
    }
    assert all(
        "recovery_page_coverage" not in methods
        for methods in first_methods.values()
    )
    assert all(
        "recovery_page_coverage" not in methods
        for methods in rerun_methods.values()
    )
