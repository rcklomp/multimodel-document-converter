"""Phase 1 Step 1 trace harness — Hao p5–p7 pipeline disappearance.

This script is READ-ONLY against production code. It reproduces the
production technical_manual path bit-for-bit by:

  * slicing Hao to the same 10-page tmp PDF the BatchProcessor would
    create for batch 1;
  * driving the production `DoclingPdfAdapter` with a
    `PdfConversionPlan` whose fields match the technical_manual route;
  * dumping per-page evidence at four pipeline stages:
      A. Docling raw `iterate_items()` — by page, by label
      B. Pre-chunker serializer surface — which items the
         `ChunkingDocSerializer` actually emits text for
      C. HybridChunker DocChunks — the most informative log per the
         Phase 1 plan
      D. Cross-page DocChunk shape — which DocChunks span page 5/6/7
         and where they get attributed.

It does NOT mutate any production state. The output is a JSON report
written to `/tmp/hao_p5_trace/report.json` and a human-readable
summary printed to stdout.

Usage:
    conda run -n mmrag-v2 python scripts/trace_hao_p5_pipeline.py
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fitz  # PyMuPDF

# Production imports — exactly what the BatchProcessor uses.
from mmrag_v2.engines.docling_adapter import DoclingPdfAdapter
from mmrag_v2.engines.pdf_plan import build_pdf_conversion_plan


HAO_PDF = Path(
    "data/technical_manual/Hao B. Machine Learning Platform Engineering. "
    "Build...for ML and AI systems 2026.pdf"
)
OUT_DIR = Path("/tmp/hao_p5_trace")
PAGE_RANGE = (1, 10)  # production batch 1 = pages 1..10
FOCUS_PAGES = {5, 6, 7}


def slice_pdf(src: Path, first_page: int, last_page: int, dst: Path) -> None:
    """Slice [first_page..last_page] (1-based) into dst, like production batching."""
    src_doc = fitz.open(src)
    dst_doc = fitz.open()
    try:
        dst_doc.insert_pdf(src_doc, from_page=first_page - 1, to_page=last_page - 1)
        dst_doc.save(dst, garbage=4, deflate=True)
    finally:
        dst_doc.close()
        src_doc.close()


def build_technical_manual_plan(total_pages: int) -> Any:
    """Construct a PdfConversionPlan that mirrors the technical_manual production route.

    Mirrors the values BatchProcessor._build_legacy_conversion_plan would
    produce for a clean technical_manual run with --no-ocr --no-refiner
    --vision-provider none (matches Step 0 conditions).
    """
    plan = build_pdf_conversion_plan(
        enable_ocr=False,                 # match --no-ocr
        ocr_engine="easyocr",
        force_table_vlm=False,
        needs_code_enrichment=False,      # don't fire CPU CodeFormulaV2 for the trace
        code_enrichment_reason="trace",
        code_enrichment_score=0.0,
        has_encoding_corruption=False,
        has_flat_text_corruption=False,
        geometry_error_rate=0.0,
        total_pages=total_pages,
        image_density=0.0,
        avg_text_per_page=0.0,
        document_modality="technical_manual",
        profile_type="technical_manual",
        confidence_threshold=0.5,
        document_domain="technical_manual",
    )
    return plan


def labels_for_doc(doc: Any) -> Tuple[Counter, Dict[int, Counter], List[Dict[str, Any]]]:
    """Stage A — walk doc.iterate_items() and bin by (page, label).

    Returns:
        global_labels: Counter of label name -> count
        per_page_labels: page -> Counter(label name -> count)
        items_focus: per-page item dicts for FOCUS_PAGES (with text snippet).
    """
    global_labels: Counter = Counter()
    per_page_labels: Dict[int, Counter] = defaultdict(Counter)
    items_focus: List[Dict[str, Any]] = []

    for element, _level in doc.iterate_items():
        prov = getattr(element, "prov", None)
        prov0 = (prov[0] if isinstance(prov, list) and prov else prov) if prov else None
        page_no = int(getattr(prov0, "page_no", 0) or 0) if prov0 is not None else 0

        label_obj = getattr(element, "label", None)
        label_val = (
            str(getattr(label_obj, "value", label_obj)) if label_obj is not None else ""
        )

        global_labels[label_val] += 1
        per_page_labels[page_no][label_val] += 1

        if page_no in FOCUS_PAGES:
            text_attr = getattr(element, "text", None)
            text_snip = (text_attr or "").strip().replace("\n", " ")[:120]
            cls_name = type(element).__name__
            items_focus.append({
                "page": page_no,
                "kind": cls_name,
                "label": label_val,
                "text_len": len(text_attr) if isinstance(text_attr, str) else 0,
                "text_snip": text_snip,
            })

    return global_labels, per_page_labels, items_focus


def serializer_surface(doc: Any) -> Dict[int, Dict[str, Any]]:
    """Stage B — what does the chunker's serializer actually emit for each item?

    The HybridChunker uses a `ChunkingDocSerializer` (Markdown-style with empty
    image placeholders by default) to turn each item's `.text` into the chunk
    surface. We instantiate the serializer and ask it to serialize each item;
    if the result is empty for an item that *does* have text in the raw doc,
    that's evidence of a serializer-side filter dropping the page.
    """
    from docling_core.transforms.chunker.hierarchical_chunker import (
        ChunkingDocSerializer,
    )

    serializer = ChunkingDocSerializer(doc=doc)
    per_page: Dict[int, Dict[str, Any]] = defaultdict(
        lambda: {
            "items_total": 0,
            "items_with_text": 0,
            "items_serialized_nonempty": 0,
            "labels": Counter(),
            "examples": [],
        }
    )

    for element, _level in doc.iterate_items():
        prov = getattr(element, "prov", None)
        prov0 = (prov[0] if isinstance(prov, list) and prov else prov) if prov else None
        page_no = int(getattr(prov0, "page_no", 0) or 0) if prov0 is not None else 0
        bucket = per_page[page_no]
        bucket["items_total"] += 1
        text_attr = getattr(element, "text", None) or ""
        if isinstance(text_attr, str) and text_attr.strip():
            bucket["items_with_text"] += 1

        try:
            ser_res = serializer.serialize(item=element)
            ser_text = (ser_res.text if hasattr(ser_res, "text") else str(ser_res)) or ""
        except Exception:
            ser_text = ""

        if ser_text.strip():
            bucket["items_serialized_nonempty"] += 1

        label_obj = getattr(element, "label", None)
        label_val = (
            str(getattr(label_obj, "value", label_obj)) if label_obj is not None else ""
        )
        bucket["labels"][label_val] += 1

        if page_no in FOCUS_PAGES and len(bucket["examples"]) < 8:
            bucket["examples"].append({
                "kind": type(element).__name__,
                "label": label_val,
                "raw_text_len": len(text_attr) if isinstance(text_attr, str) else 0,
                "ser_text_len": len(ser_text),
                "raw_snip": (text_attr or "").strip()[:80],
                "ser_snip": ser_text.strip()[:80],
            })

    # Convert Counters to dicts for JSON.
    return {
        pg: {
            "items_total": v["items_total"],
            "items_with_text": v["items_with_text"],
            "items_serialized_nonempty": v["items_serialized_nonempty"],
            "labels": dict(v["labels"]),
            "examples": v["examples"],
        }
        for pg, v in sorted(per_page.items())
    }


def chunker_output(doc: Any) -> Dict[str, Any]:
    """Stage C — drive the same HybridChunker production builds."""
    from docling_core.transforms.chunker import HybridChunker

    # Same kwargs as src/mmrag_v2/processor.py:2396-2403 for technical_manual.
    chunker_kwargs: Dict[str, Any] = {
        "tokenizer": "sentence-transformers/all-MiniLM-L6-v2",
        "max_tokens": 350,
    }
    chunker = HybridChunker(**chunker_kwargs)
    doc_chunks = list(chunker.chunk(doc))

    by_page: Dict[int, int] = defaultdict(int)
    cross_page: List[Dict[str, Any]] = []
    focus_chunks: List[Dict[str, Any]] = []

    for i, dc in enumerate(doc_chunks):
        text = (dc.text or "").strip()
        items = dc.meta.doc_items if dc.meta and dc.meta.doc_items else []
        pages = sorted({
            int(getattr(it.prov[0] if isinstance(it.prov, list) else it.prov,
                        "page_no", 0) or 0)
            for it in items
            if getattr(it, "prov", None)
        })
        for p in pages:
            by_page[p] += 1
        if len(pages) > 1:
            cross_page.append({
                "i": i,
                "pages": pages,
                "n_items": len(items),
                "text_len": len(text),
                "text_snip": text[:80].replace("\n", " "),
            })

        # Capture chunks that touch FOCUS_PAGES.
        if FOCUS_PAGES.intersection(pages):
            label_set = sorted({
                str(getattr(getattr(it, "label", None), "value",
                            getattr(it, "label", "")))
                for it in items
            })
            focus_chunks.append({
                "i": i,
                "pages": pages,
                "labels": label_set,
                "text_len": len(text),
                "text_snip": text[:120].replace("\n", " "),
            })

    return {
        "n_chunks": len(doc_chunks),
        "chunks_by_page": dict(sorted(by_page.items())),
        "cross_page_chunks": cross_page,
        "focus_chunks": focus_chunks,
    }


def pages_with_visible_text(slice_pdf_path: Path) -> Dict[int, int]:
    """Sanity check — how much text PyMuPDF sees on each page of the slice."""
    doc = fitz.open(slice_pdf_path)
    out: Dict[int, int] = {}
    try:
        for i in range(len(doc)):
            text = (doc.load_page(i).get_text("text") or "").strip()
            out[i + 1] = len(text)
    finally:
        doc.close()
    return out


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    if not HAO_PDF.exists():
        raise SystemExit(f"Hao PDF not found: {HAO_PDF}")

    first, last = PAGE_RANGE
    slice_path = OUT_DIR / f"hao_p{first}-{last}.pdf"
    slice_pdf(HAO_PDF, first, last, slice_path)

    print(f"[trace] sliced Hao p{first}-{last} -> {slice_path} "
          f"(size={slice_path.stat().st_size} bytes)")

    pre_text = pages_with_visible_text(slice_path)
    print(f"[trace] PyMuPDF visible-text bytes per page: {pre_text}")

    plan = build_technical_manual_plan(total_pages=last - first + 1)
    print(f"[trace] plan: route={plan.extraction_route} "
          f"hybrid_chunker_enabled={plan.hybrid_chunker_enabled} "
          f"do_cell_matching={plan.do_cell_matching} "
          f"do_picture_classification={plan.do_picture_classification} "
          f"suppress_layout_label_text={plan.suppress_layout_label_text}")

    adapter = DoclingPdfAdapter(plan)
    result = adapter.convert(slice_path)
    doc = result.document

    # Stage A
    a_global, a_per_page, a_focus_items = labels_for_doc(doc)
    print("\n[stage A] doc.iterate_items()")
    print(f"  global label distribution: {dict(a_global)}")
    for pg in sorted(a_per_page):
        print(f"  page {pg}: {dict(a_per_page[pg])}")

    # Stage B
    b = serializer_surface(doc)
    print("\n[stage B] ChunkingDocSerializer surface")
    for pg, info in b.items():
        marker = "  ** " if pg in FOCUS_PAGES else "  "
        print(
            f"{marker}page {pg}: items={info['items_total']} "
            f"with_text={info['items_with_text']} "
            f"ser_nonempty={info['items_serialized_nonempty']} "
            f"labels={info['labels']}"
        )

    # Stage C
    c = chunker_output(doc)
    print("\n[stage C] HybridChunker.chunk(doc)")
    print(f"  n_chunks: {c['n_chunks']}")
    print(f"  chunks_by_page: {c['chunks_by_page']}")
    print(f"  cross-page chunks: {len(c['cross_page_chunks'])}")
    for cp in c["cross_page_chunks"][:8]:
        print(f"    pages={cp['pages']} n_items={cp['n_items']} "
              f"text_len={cp['text_len']} snip={cp['text_snip']!r}")
    print(f"  chunks touching FOCUS_PAGES {sorted(FOCUS_PAGES)}: "
          f"{len(c['focus_chunks'])}")
    for fc in c["focus_chunks"][:12]:
        print(f"    pages={fc['pages']} labels={fc['labels']} "
              f"snip={fc['text_snip']!r}")

    # Diagnostics summary.
    a_focus_pages = {pg for pg in FOCUS_PAGES if a_per_page.get(pg)}
    b_focus_pages = {pg for pg in FOCUS_PAGES if b.get(pg, {}).get("items_with_text", 0) > 0}
    c_focus_pages = {
        pg for pg in FOCUS_PAGES if c["chunks_by_page"].get(pg, 0) > 0
    }
    diagnosis = {
        "focus_pages": sorted(FOCUS_PAGES),
        "stage_A_pages_with_items": sorted(a_focus_pages),
        "stage_B_pages_with_text_items": sorted(b_focus_pages),
        "stage_C_pages_with_chunks": sorted(c_focus_pages),
        "lost_between_A_and_C": sorted(a_focus_pages - c_focus_pages),
    }
    print("\n[diagnosis]")
    print(json.dumps(diagnosis, indent=2))

    report = {
        "slice": {
            "pdf": str(slice_path),
            "first_page": first,
            "last_page": last,
            "pymupdf_text_bytes_per_page": pre_text,
        },
        "plan": {
            "extraction_route": plan.extraction_route,
            "hybrid_chunker_enabled": plan.hybrid_chunker_enabled,
            "do_cell_matching": plan.do_cell_matching,
            "do_picture_classification": plan.do_picture_classification,
            "suppress_layout_label_text": plan.suppress_layout_label_text,
            "do_ocr": plan.do_ocr,
            "needs_code_enrichment": plan.needs_code_enrichment,
            "bitmap_area_threshold": plan.bitmap_area_threshold,
        },
        "stage_A": {
            "global_labels": dict(a_global),
            "per_page_labels": {pg: dict(c) for pg, c in a_per_page.items()},
            "focus_items": a_focus_items,
        },
        "stage_B_serializer_surface": b,
        "stage_C_chunker": c,
        "diagnosis": diagnosis,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\n[trace] full report -> {OUT_DIR / 'report.json'}")


if __name__ == "__main__":
    main()
