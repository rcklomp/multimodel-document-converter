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

Targets:
    * ``--pdf <path>`` (+ optional ``--baseline <path>``, default
      ``output/<stem>/ingestion.jsonl``) re-baselines an arbitrary doc.
    * a positional named shortcut (e.g. ``CarOK_voorraadtelling``) uses
      the path pair from ``REF_DOCS``.

Route guard (replaces the old USE_VLM_ENGINE hard gate):
    Re-baselining must run through a VLM/hybrid vision route, NEVER the
    docling/offline fallback (that would promote inferior extraction).
    The effective route is resolved by mirroring
    ``mmrag_v3.processor._select_engine`` precedence; the script proceeds
    iff it resolves to ``mineru_qwen_hybrid`` or ``vlm_native`` (the
    explicitly-forced VLM route), and refuses (exit 2) otherwise.
    Set ``USE_VLM_ENGINE=1`` (+ ``VLM_NATIVE_*``) or configure
    ``MINERU_ENDPOINT`` (the ``mineru_qwen_hybrid`` default).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

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


# Routes that carry VLM-grade fidelity and are therefore valid rebaseline
# sources. Everything else (notably docling_fast and the legacy docling-based
# `hybrid` fallback) is refused — rebaselining against docling would promote the
# very output this script exists to replace.
ALLOWED_ROUTES = {"mineru_qwen_hybrid", "vlm_native"}


def resolve_route() -> str:
    """Return the effective engine route name WITHOUT instantiating an engine.

    Mirrors the precedence of ``mmrag_v3.processor._select_engine`` (first
    match wins) using that module's pure env-predicate helpers, so the guard
    reuses a single source of routing truth instead of re-parsing env vars.
    """
    from mmrag_v3.processor import (
        _default_route_is_mineru,
        is_docling_fast_route_enabled,
        is_hybrid_route_enabled,
        is_mineru_qwen_hybrid_route_enabled,
        is_mineru_route_enabled,
        is_vlm_route_enabled,
    )

    if is_mineru_route_enabled():
        return "mineru"
    if is_vlm_route_enabled():
        return "vlm_native"
    if is_docling_fast_route_enabled():
        return "docling_fast"
    if is_hybrid_route_enabled():
        return "hybrid"
    if is_mineru_qwen_hybrid_route_enabled():
        return "mineru_qwen_hybrid"
    if _default_route_is_mineru():
        return "mineru_qwen_hybrid"
    return "hybrid"


def rebaseline(pdf_path: Path, baseline_path: Path) -> int:
    route = resolve_route()
    if route not in ALLOWED_ROUTES:
        print(
            f"REFUSING TO REBASELINE: effective route is {route!r} — re-baselining "
            f"must run through a VLM vision route ({sorted(ALLOWED_ROUTES)}), not the "
            f"docling/offline fallback. Set USE_VLM_ENGINE=1 (+ VLM_NATIVE_*) or "
            f"configure MINERU_ENDPOINT (the mineru_qwen_hybrid default)."
        )
        return 2

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

    print(f"running V3 extraction on {pdf_path.name} (route={route}) ...")
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


def resolve_targets(args: argparse.Namespace) -> Optional[Tuple[Path, Path]]:
    """Resolve (pdf_path, baseline_path) from --pdf/--baseline or a named shortcut."""
    if args.pdf:
        pdf_path = Path(args.pdf).expanduser().resolve()
        if args.baseline:
            baseline_path = Path(args.baseline).expanduser().resolve()
        else:
            baseline_path = REPO_ROOT / "output" / pdf_path.stem / "ingestion.jsonl"
        return pdf_path, baseline_path

    doc_name = args.doc
    if doc_name not in REF_DOCS:
        print(f"unknown ref doc: {doc_name!r} (known: {sorted(REF_DOCS.keys())})")
        return None
    ref = REF_DOCS[doc_name]
    return ref["pdf"], ref["baseline"]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "doc",
        nargs="?",
        default="CarOK_voorraadtelling",
        help="named ref-doc shortcut (used when --pdf is not given; "
        "default: CarOK_voorraadtelling)",
    )
    parser.add_argument(
        "--pdf",
        default=None,
        help="arbitrary source PDF path (overrides the named shortcut)",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="baseline JSONL output path (default: output/<stem>/ingestion.jsonl)",
    )
    args = parser.parse_args(argv)
    targets = resolve_targets(args)
    if targets is None:
        return 2
    return rebaseline(*targets)


if __name__ == "__main__":
    sys.exit(main())
