#!/usr/bin/env python3
"""One-off probe for Phase 4 — inspect HybridChunker DocChunks
covering Python_Cookbook pages 125-130 to discover whether the
chunk containing p127+p128 content actually has multi-page
doc_items or whether items mis-report their prov[0].page_no.
"""
from pathlib import Path
from typing import Any, List

PDF = Path("data/technical_manual/Python Cookbook  Everyone can cook delicious recipes with Python.pdf")
PAGE_LOW, PAGE_HIGH = 125, 130


def main() -> None:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.transforms.chunker import HybridChunker

    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 1.0
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = False
    pipeline_options.do_ocr = False

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    print(f"Converting (full doc) {PDF.name} ...")
    result = converter.convert(str(PDF))
    doc = result.document

    print("Chunking with HybridChunker (matching production kwargs) ...")
    chunker = HybridChunker(
        tokenizer="sentence-transformers/all-MiniLM-L6-v2",
        max_tokens=350,
    )
    doc_chunks = list(chunker.chunk(doc))
    print(f"Total DocChunks: {len(doc_chunks)}")

    def _prov_pages(item: Any) -> List[int]:
        prov = getattr(item, "prov", None)
        if not prov:
            return []
        prov_list = prov if isinstance(prov, list) else [prov]
        return [int(getattr(p, "page_no", 0) or 0) for p in prov_list]

    for i, dc in enumerate(doc_chunks):
        items = (dc.meta.doc_items if dc.meta else []) or []
        page_set: set[int] = set()
        first_pages: set[int] = set()
        for it in items:
            pages = _prov_pages(it)
            for p in pages:
                page_set.add(p)
            if pages:
                first_pages.add(pages[0])
        if not (page_set & set(range(PAGE_LOW, PAGE_HIGH + 1))):
            continue
        print()
        print(f"=== DocChunk #{i} ===")
        print(f"  item count       = {len(items)}")
        print(f"  prov[0].page_no  = {sorted(first_pages)}")
        print(f"  ALL prov pages   = {sorted(page_set)}")
        text = getattr(dc, "text", "") or ""
        print(f"  text length      = {len(text)}")
        # Item-by-item breakdown
        for j, it in enumerate(items[:30]):
            label_obj = getattr(it, "label", "")
            label = getattr(label_obj, "value", label_obj)
            it_text = getattr(it, "text", "") or ""
            pages = _prov_pages(it)
            print(
                f"    item[{j:>2}] label={label!s:<15} pages={pages!s:<14} "
                f"text_len={len(it_text)} :: {it_text[:60]!r}"
            )
        print(f"  text[:400] = {text[:400]!r}")
        print(f"  text[-300:] = {text[-300:]!r}")


if __name__ == "__main__":
    main()
