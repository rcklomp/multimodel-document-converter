#!/usr/bin/env python3
"""Re-baseline a V3 reference doc to its current V3 VLM + chunker output.

Used after the V3 vision-native engine produces objectively superior
extraction than the legacy v2.16 baseline (e.g., CarOK_voorraadtelling
where Docling silently dropped ~80% of the table rows). Running this
script promotes the V3 chunker output to the authoritative baseline
JSONL.

This script runs the SHIPPING path (PLAN_V3.1 P1 — single extraction
path): ``mmrag_v3.processor.extract`` (HybridEngine router) →
``mmrag_v2.chunking.uir_chunker.chunk_universal_document`` →
``IngestionChunk.from_uir``. No sandbox translation, no
retired-sandbox import.

Behavior:
    * Backs up the existing baseline to ``<path>.v2_baseline.bak``
      (only on the FIRST rebaseline; subsequent runs leave the backup
      alone so the original v2.16 reference is preserved).
    * Extracts via the V3 engine → ``UniversalDocument``.
    * Runs the UIR-native chunker → ``UIRChunk`` objects.
    * Writes one ``IngestionChunk`` JSON object per line to the baseline
      path (the same shape ``mmrag-v2 process`` emits). NO
      ingestion_metadata header line.

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
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List

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

    # Production shipping path (PLAN_V3.1 P1): engine-agnostic extract →
    # UIR-native chunker → IngestionChunk.from_uir.
    from mmrag_v3.processor import extract as v3_extract
    from mmrag_v2.chunking.uir_chunker import chunk_universal_document
    from mmrag_v2.schema.ingestion_schema import FileType, IngestionChunk

    print(f"running V3 extraction on {pdf_path.name} ...")
    t0 = time.time()
    universal_doc = v3_extract(str(pdf_path))
    uir_chunks = chunk_universal_document(universal_doc)
    doc_id = universal_doc.doc_id or universal_doc.compute_doc_id()
    chunks: List[IngestionChunk] = [
        IngestionChunk.from_uir(
            uir,
            doc_id=doc_id,
            source_file=pdf_path.name,
            file_type=FileType.PDF,
            position=position,
        )
        for position, uir in enumerate(uir_chunks)
    ]
    elapsed = time.time() - t0
    print(
        f"extracted+chunked: {len(chunks)} chunks across "
        f"{universal_doc.total_pages} pages in {elapsed:.1f}s"
    )

    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with baseline_path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False))
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
