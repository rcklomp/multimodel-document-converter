#!/usr/bin/env python3
"""Re-baseline a V3 reference doc to its current V3 VLM + chunker output.

Used after the V3 vision-native engine produces objectively superior
extraction than the legacy v2.16 baseline (e.g., CarOK_voorraadtelling
where Docling silently dropped ~80% of the table rows). Running this
script promotes the V3 chunker output to the authoritative baseline
JSONL.

Behavior:
    * Backs up the existing baseline to ``<path>.v2_baseline.bak``
      (only on the FIRST rebaseline; subsequent runs leave the backup
      alone so the original v2.16 reference is preserved).
    * Runs the V3 VLM-native engine → v2-UIR ``UniversalDocument``.
    * Translates to v3-UIR ``UniversalDocument`` (same translator the
      Identity Gate subprocess uses).
    * Runs the V3 chunker.
    * Writes one ``IngestionChunk.model_dump_json()`` per line to the
      baseline path. NO ingestion_metadata header line (the gate's
      loader filters those, and V3 IngestionChunks carry no
      ``object_type`` field).

Env requirements:
    USE_VLM_ENGINE          — must be "1" to use the VLM route (else
                              the script raises rather than silently
                              re-baselining against Docling output).
    VLM_NATIVE_ENDPOINT     — e.g. http://10.0.10.246:8000/v1
    VLM_NATIVE_MODEL        — e.g. Qwen2.5-VL-7B-Instruct-8bit
    VLM_NATIVE_API_KEY      — Bearer token for the endpoint (optional)
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import time
import types
from pathlib import Path
from typing import Any, List

REPO_ROOT = Path(__file__).resolve().parent.parent


REF_DOCS = {
    "CarOK_voorraadtelling": {
        "pdf": REPO_ROOT
        / "data"
        / "data_spreadsheet"
        / "CarOK voorraadtelling 2021-04.pdf",
        "baseline": REPO_ROOT
        / "output"
        / "CarOK_voorraadtelling"
        / "ingestion.jsonl",
    },
}


def _load_phase_c_engine() -> Any:
    """Load src/mmrag_v3/engines/vlm_native.py by absolute file path.

    Same trick the gate's subprocess uses to sidestep the namespace
    collision between the project src/mmrag_v3 and the v3_execution_root
    sandbox src/mmrag_v3.
    """
    engines_dir = REPO_ROOT / "src" / "mmrag_v3" / "engines"
    pkg = types.ModuleType("_phase_c_engine")
    pkg.__path__ = [str(engines_dir)]  # type: ignore[attr-defined]
    sys.modules["_phase_c_engine"] = pkg

    for module_name in ("vlm_provider", "vlm_native"):
        spec = importlib.util.spec_from_file_location(
            f"_phase_c_engine.{module_name}",
            engines_dir / f"{module_name}.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"_phase_c_engine.{module_name}"] = module
        spec.loader.exec_module(module)

    return sys.modules["_phase_c_engine.vlm_native"].VlmNativeEngine


def _translate_v2_to_v3(v2_doc: Any) -> Any:
    """Map a v2-UIR UniversalDocument into the v3-UIR shape.

    Mirrors the translator used by the Identity Gate's subprocess so
    rebaseline and gate produce structurally identical chunk streams.
    """
    from mmrag_v3.universal.document import (
        BoundingBox as V3BBox,
        DocumentMetadata as V3Meta,
        Element as V3Element,
        ElementType as V3ElementType,
        ExtractionMethod as V3ExtractionMethod,
        PageClassification as V3PageClassification,
        UniversalDocument as V3Doc,
        UniversalPage as V3Page,
    )

    method_map = {
        "native": V3ExtractionMethod.DIGITAL,
        "ocr_tesseract": V3ExtractionMethod.OCR,
        "ocr_docling": V3ExtractionMethod.OCR,
        "ocr_doctr": V3ExtractionMethod.OCR,
        "vlm": V3ExtractionMethod.VLM,
        "fallback": V3ExtractionMethod.UNKNOWN,
    }

    v3_pages = []
    for v2_page in v2_doc.pages:
        v3_elements = []
        for el in v2_page.elements:
            bbox = None
            if el.bbox is not None:
                bbox = V3BBox(
                    x_min=int(el.bbox.x_min),
                    y_min=int(el.bbox.y_min),
                    x_max=int(el.bbox.x_max),
                    y_max=int(el.bbox.y_max),
                )
            v3_elements.append(
                V3Element(
                    type=V3ElementType(el.type.value),
                    content=el.content,
                    bbox=bbox,
                    confidence=float(el.confidence),
                    raw_image=None,
                    extraction_method=method_map.get(
                        el.extraction_method.value, V3ExtractionMethod.UNKNOWN
                    ),
                    element_index=int(el.element_index),
                    source_label=str(el.source_label or ""),
                    metadata=dict(el.metadata or {}),
                )
            )
        v3_pages.append(
            V3Page(
                page_number=int(v2_page.page_number),
                elements=v3_elements,
                classification=V3PageClassification(v2_page.classification.value),
                dimensions=(int(v2_page.dimensions[0]), int(v2_page.dimensions[1])),
                raw_image=None,
                text_density=float(getattr(v2_page, "text_density", 0.0) or 0.0),
                avg_confidence=float(getattr(v2_page, "avg_confidence", 0.0) or 0.0),
            )
        )

    v2_meta = v2_doc.metadata
    v3_meta = V3Meta(
        title=getattr(v2_meta, "title", None),
        author=getattr(v2_meta, "author", None),
        language=getattr(v2_meta, "language", None),
        page_count=len(v3_pages),
    )

    return V3Doc(
        doc_id=str(v2_doc.doc_id),
        source_file=str(v2_doc.source_file),
        file_type=str(v2_doc.file_type),
        pages=v3_pages,
        metadata=v3_meta,
        total_pages=len(v3_pages),
    )


def rebaseline(doc_name: str) -> int:
    if doc_name not in REF_DOCS:
        print(
            f"unknown ref doc: {doc_name!r} (known: {sorted(REF_DOCS.keys())})"
        )
        return 2

    if os.environ.get("USE_VLM_ENGINE", "").strip() not in {"1", "true", "TRUE", "yes"}:
        print(
            "REFUSING TO REBASELINE: USE_VLM_ENGINE is not set — re-baselining "
            "must run through the V3 VLM engine, not the Docling fallback."
        )
        return 2

    ref = REF_DOCS[doc_name]
    pdf_path: Path = ref["pdf"]
    baseline_path: Path = ref["baseline"]
    if not pdf_path.is_file():
        print(f"source PDF missing: {pdf_path}")
        return 2

    # Preserve the v2.16 reference on the first rebaseline.
    bak_path = baseline_path.with_suffix(baseline_path.suffix + ".v2_baseline.bak")
    if baseline_path.exists() and not bak_path.exists():
        shutil.copy2(baseline_path, bak_path)
        print(f"backed up legacy baseline → {bak_path.name}")

    # Make v3_execution_root src importable so the chunker resolves.
    v3_src = REPO_ROOT / "v3_execution_root" / "src"
    if str(v3_src) not in sys.path:
        sys.path.insert(0, str(v3_src))

    from mmrag_v3.chunking.chunker import chunk as chunk_document

    VlmNativeEngine = _load_phase_c_engine()
    engine = VlmNativeEngine()

    print(f"running VLM extraction on {pdf_path.name} ...")
    t0 = time.time()
    v2_doc = engine.extract(str(pdf_path))
    document = _translate_v2_to_v3(v2_doc)
    chunks: List[Any] = chunk_document(document)
    elapsed = time.time() - t0
    print(
        f"extracted+chunked: {len(chunks)} chunks across "
        f"{document.total_pages} pages in {elapsed:.1f}s"
    )

    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with baseline_path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(chunk.model_dump_json())
            fh.write("\n")
    print(f"wrote new baseline → {baseline_path}")
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "doc",
        nargs="?",
        default="CarOK_voorraadtelling",
        help="Reference doc name to re-baseline (default: CarOK_voorraadtelling)",
    )
    args = parser.parse_args(argv)
    return rebaseline(args.doc)


if __name__ == "__main__":
    sys.exit(main())
