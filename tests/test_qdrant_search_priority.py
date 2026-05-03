from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.parent / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import ingest_to_qdrant as ingestor  # noqa: E402
from ingest_to_qdrant import (  # noqa: E402
    build_qdrant_payload,
    find_literature_toc_page,
    infer_document_domain,
    read_ingestion_jsonl,
    resolve_search_priority,
)


def _chunk(
    content: str,
    *,
    page: int = 1,
    domain: str | None = None,
    priority: str = "high",
    chunk_type: str = "paragraph",
    parent_heading: str | None = None,
    breadcrumb: list[str] | None = None,
    modality: str = "text",
) -> dict:
    metadata = {
        "page_number": page,
        "search_priority": priority,
        "chunk_type": chunk_type,
        "hierarchy": {
            "parent_heading": parent_heading,
            "breadcrumb_path": breadcrumb or [],
        },
    }
    if domain:
        metadata["document_domain"] = domain
    return {
        "chunk_id": f"chunk_{page}_{chunk_type}_{abs(hash(content))}",
        "doc_id": "doc123",
        "modality": modality,
        "content": content,
        "metadata": metadata,
    }


def test_literature_pages_before_toc_are_low_priority() -> None:
    chunks = [
        _chunk("Copyright 2026", page=1),
        _chunk("Contents", page=4, chunk_type="heading"),
        _chunk("Chapter 1", page=5, chunk_type="heading"),
    ]
    toc_page = find_literature_toc_page(chunks)

    assert toc_page == 4
    assert resolve_search_priority(chunks[0], "literature", toc_page) == "low"
    assert resolve_search_priority(chunks[2], "literature", toc_page) == "high"


def test_literature_without_toc_does_not_demote_by_page_position() -> None:
    chunk = _chunk("Chapter 1", page=1, chunk_type="heading")

    assert resolve_search_priority(chunk, "literature", None) == "high"


def test_academic_references_section_is_medium_priority() -> None:
    body = _chunk("Smith et al. discuss the result.", page=18, parent_heading="References")

    assert resolve_search_priority(body, "academic") == "medium"


def test_academic_body_mention_of_references_is_not_section_demoted() -> None:
    body = _chunk("See references in the implementation notes.", page=8)

    assert resolve_search_priority(body, "academic") == "high"


def test_technical_appendix_and_index_context_are_medium_priority() -> None:
    appendix = _chunk("Pinout details", page=92, parent_heading="Appendix A")
    index = _chunk("adapter, 23", page=120, parent_heading="Index")

    assert resolve_search_priority(appendix, "technical") == "medium"
    assert resolve_search_priority(index, "technical") == "medium"


def test_technical_body_mention_of_appendix_is_not_section_demoted() -> None:
    body = _chunk("The appendix is attached with two screws.", page=22)

    assert resolve_search_priority(body, "technical") == "high"


def test_domain_rules_never_promote_existing_low_priority() -> None:
    boilerplate = _chunk("References", page=30, priority="low", chunk_type="heading")

    assert resolve_search_priority(boilerplate, "academic") == "low"


def test_bridge_metadata_record_domain_reaches_qdrant_payload_priority(tmp_path: Path) -> None:
    jsonl = tmp_path / "ingestion.jsonl"
    rows = [
        {
            "object_type": "ingestion_metadata",
            "schema_version": "2.7.0",
            "doc_id": "doc123",
            "source_file": "novel.pdf",
            "domain": "literature",
        },
        _chunk("Copyright page", page=1),
        _chunk("Contents", page=3, chunk_type="heading"),
    ]
    with jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    metadata_record, chunks = read_ingestion_jsonl(jsonl)
    domain = infer_document_domain(metadata_record, chunks)
    toc_page = find_literature_toc_page(chunks)
    payload = build_qdrant_payload(
        chunks[0],
        source_file="novel.pdf",
        document_domain=domain,
        literature_toc_page=toc_page,
    )

    assert domain == "literature"
    assert payload["document_domain"] == "literature"
    assert payload["search_priority"] == "low"


def test_bridge_chunk_domain_reaches_qdrant_payload_without_metadata_record() -> None:
    chunks = [
        _chunk("Smith 2024.", page=20, domain="academic", parent_heading="References"),
    ]

    domain = infer_document_domain(None, chunks)
    payload = build_qdrant_payload(
        chunks[0],
        source_file="paper.pdf",
        document_domain=domain,
    )

    assert domain == "academic"
    assert payload["search_priority"] == "medium"


def test_bridge_main_upserts_resolved_search_priority(
    tmp_path: Path,
    monkeypatch,
) -> None:
    jsonl = tmp_path / "ingestion.jsonl"
    rows = [
        {
            "object_type": "ingestion_metadata",
            "schema_version": "2.7.0",
            "doc_id": "doc123",
            "source_file": "manual.pdf",
            "domain": "technical",
        },
        _chunk("Torque values", page=80, parent_heading="Appendix B"),
    ]
    with jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    captured_points: list[dict] = []

    monkeypatch.setattr(sys, "argv", ["ingest_to_qdrant.py", str(jsonl)])
    monkeypatch.setattr(ingestor, "embed_text", lambda *args, **kwargs: [0.1, 0.2])
    monkeypatch.setattr(ingestor, "create_collection", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        ingestor,
        "upsert_batch",
        lambda collection, points, qdrant_url: captured_points.extend(points),
    )
    monkeypatch.setattr(
        ingestor,
        "qdrant_request",
        lambda *args, **kwargs: {"result": {"points_count": len(captured_points)}},
    )

    assert ingestor.main() == 0

    assert captured_points
    payload = captured_points[0]["payload"]
    assert payload["document_domain"] == "technical"
    assert payload["search_priority"] == "medium"
