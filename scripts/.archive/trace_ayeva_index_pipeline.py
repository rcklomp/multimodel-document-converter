"""Phase 1 trace harness for Ayeva back-index dense-page routing.

Slices Ayeva pages 281-290, converts them through the production Docling
adapter, prints the dense-page classifier result, then times the processor's
HybridChunker path. This is read-only against production code.

Usage:
    conda run -n mmrag-v2 python scripts/trace_ayeva_index_pipeline.py
"""

from __future__ import annotations

import json
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import fitz

from mmrag_v2.engines.docling_adapter import DoclingPdfAdapter
from mmrag_v2.engines.pdf_plan import build_pdf_conversion_plan
from mmrag_v2.processor import V2DocumentProcessor, _classify_dense_index_pages
from mmrag_v2.schema.ingestion_schema import FileType


AYEVA_PDF = Path(
    "data/technical_manual/"
    "Ayeva K. Mastering Python Design Patterns...essential Python patterns...3ed 2024.pdf"
)
PAGE_RANGE = (281, 290)


def slice_pdf(src: Path, dst: Path) -> None:
    src_doc = fitz.open(src)
    dst_doc = fitz.open()
    try:
        dst_doc.insert_pdf(
            src_doc,
            from_page=PAGE_RANGE[0] - 1,
            to_page=PAGE_RANGE[1] - 1,
        )
        dst_doc.save(dst, garbage=4, deflate=True)
    finally:
        dst_doc.close()
        src_doc.close()


def page_item_summary(doc: Any) -> dict[int, dict[str, Any]]:
    per_page: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"items": 0, "text_items": 0, "labels": Counter(), "examples": []}
    )
    for item, _level in doc.iterate_items():
        prov = getattr(item, "prov", None)
        first = prov[0] if isinstance(prov, list) and prov else prov
        page_no = int(getattr(first, "page_no", 0) or 0) if first else 0
        text = (getattr(item, "text", "") or "").strip()
        label = getattr(getattr(item, "label", ""), "value", getattr(item, "label", ""))
        bucket = per_page[page_no]
        bucket["items"] += 1
        bucket["labels"][str(label)] += 1
        if text:
            bucket["text_items"] += 1
        if text and len(bucket["examples"]) < 8:
            bucket["examples"].append(text[:100])
    return {
        page: {
            "items": value["items"],
            "text_items": value["text_items"],
            "labels": dict(value["labels"]),
            "examples": value["examples"],
        }
        for page, value in sorted(per_page.items())
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="ayeva_index_trace_") as tmp:
        batch_path = Path(tmp) / "ayeva_281_290.pdf"
        slice_pdf(AYEVA_PDF, batch_path)
        plan = build_pdf_conversion_plan(
            enable_ocr=False,
            needs_code_enrichment=False,
            has_encoding_corruption=False,
            has_flat_text_corruption=False,
            document_modality="technical_manual",
            profile_type="technical_manual",
        )
        doc = DoclingPdfAdapter(plan).convert(batch_path).document
        dense_pages = _classify_dense_index_pages(doc)
        processor = V2DocumentProcessor(output_dir=tmp, vision_provider="none")
        started = time.monotonic()
        chunks = processor._process_text_with_hybrid_chunker(
            doc=doc,
            doc_hash="ayeva_trace",
            source_file=AYEVA_PDF.name,
            file_type=FileType.PDF,
            page_dims={page: (612.0, 792.0) for page in range(1, 11)},
            page_offset=PAGE_RANGE[0] - 1,
        )
        elapsed = time.monotonic() - started
        report = {
            "source": str(AYEVA_PDF),
            "page_range": PAGE_RANGE,
            "dense_pages_raw": sorted(dense_pages),
            "dense_pages_source": sorted(page + PAGE_RANGE[0] - 1 for page in dense_pages),
            "hybrid_seconds": round(elapsed, 3),
            "chunk_methods": Counter(ch.metadata.extraction_method for ch in chunks),
            "page_item_summary": page_item_summary(doc),
        }
        report["chunk_methods"] = dict(report["chunk_methods"])
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
