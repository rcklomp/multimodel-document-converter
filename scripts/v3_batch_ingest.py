#!/usr/bin/env python3
"""V3 Grand Soak — full-corpus batch ingest through HybridEngine.

For each PDF in ``data/``:
    1. Extract via HybridEngine (cost-optimizer router → VLM for
       visually-complex pages, fast Docling for prose pages).
    2. Translate v2-UIR → v3-UIR (mirrors rebaseline_v3._translate_v2_to_v3).
    3. Run the V3 chunker.
    4. Write ``output/v3_baselines/<doc_stem>/ingestion.jsonl``.
    5. Record routing decisions + timings to ``meta.json``.

Per-document failures are caught, logged, and the run continues
(matches the unattended-execution protocol's catch+log+skip+continue
contract). A top-level ``manifest.json`` summarizes the run.

Resume: if ``ingestion.jsonl`` already exists for a doc and is
non-empty, it is skipped unless ``--force`` is passed.

Env requirements:
    USE_VLM_ENGINE       — must be "1" (else the script refuses)
    VLM_NATIVE_ENDPOINT  — OpenRouter base URL (default in vlm_provider)
    VLM_NATIVE_MODEL     — OpenRouter model id, e.g. qwen/qwen3-vl-8b-instruct
    VLM_NATIVE_API_KEY   — Bearer token for the endpoint
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("v3_batch_ingest")


def _load_phase_c_modules() -> Tuple[Any, Any]:
    """Load src/mmrag_v3/engines by absolute path → (VlmNativeEngine, HybridEngine).

    Matches rebaseline_v3._load_phase_c_engine to sidestep the namespace
    collision between project src/mmrag_v3 and v3_execution_root src/mmrag_v3.
    """
    engines_dir = REPO_ROOT / "src" / "mmrag_v3" / "engines"
    pkg = types.ModuleType("_phase_c_engine")
    pkg.__path__ = [str(engines_dir)]  # type: ignore[attr-defined]
    sys.modules["_phase_c_engine"] = pkg

    for module_name in (
        "vlm_provider",
        "vlm_native",
        "docling_fast",
        "router",
    ):
        spec = importlib.util.spec_from_file_location(
            f"_phase_c_engine.{module_name}",
            engines_dir / f"{module_name}.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"_phase_c_engine.{module_name}"] = module
        spec.loader.exec_module(module)

    return (
        sys.modules["_phase_c_engine.vlm_native"].VlmNativeEngine,
        sys.modules["_phase_c_engine.router"].HybridEngine,
    )


def _translate_v2_to_v3(v2_doc: Any) -> Any:
    """Map v2-UIR UniversalDocument → v3-UIR shape. Mirrors rebaseline_v3."""
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


def _path_for_manifest(p: Path) -> str:
    """Display absolute paths under REPO_ROOT as relative; everything else absolute."""
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _routing_summary(
    decisions: List[Tuple[int, str, str]],
) -> Dict[str, int]:
    counts = {"vlm": 0, "docling": 0, "docling_fallback": 0, "other": 0}
    for _, choice, _ in decisions:
        if choice in counts:
            counts[choice] += 1
        else:
            counts["other"] += 1
    return counts


def _collect_pdfs(
    data_dir: Path, limit: Optional[int], max_pages: Optional[int]
) -> List[Path]:
    pdfs = sorted(p for p in data_dir.rglob("*.pdf") if p.is_file())
    if max_pages is not None and max_pages > 0:
        import fitz  # local import — fitz already a script dep elsewhere
        kept: List[Path] = []
        for p in pdfs:
            try:
                with fitz.open(str(p)) as d:
                    n = d.page_count
            except Exception:
                kept.append(p)
                continue
            if n <= max_pages:
                kept.append(p)
            else:
                logger.info(
                    "skipping %s (%d pages > --max-pages %d)",
                    p.name, n, max_pages,
                )
        pdfs = kept
    if limit is not None and limit > 0:
        pdfs = pdfs[:limit]
    return pdfs


def _doc_outdir(out_root: Path, pdf: Path) -> Path:
    # Use parent-relative stem so docs from the same subdir don't collide
    # and the layout mirrors data/<category>/<stem>/.
    rel = pdf.relative_to(pdf.parents[1]) if len(pdf.parents) >= 2 else pdf.name
    # rel is like Path("technical_manual/Foo.pdf")
    stem_dir = Path(str(rel.with_suffix("")))
    return out_root / stem_dir


def _is_complete(doc_dir: Path) -> bool:
    jsonl = doc_dir / "ingestion.jsonl"
    meta = doc_dir / "meta.json"
    if not jsonl.exists() or not meta.exists():
        return False
    try:
        return jsonl.stat().st_size > 0
    except OSError:
        return False


def _process_one_pdf(
    pdf: Path,
    out_root: Path,
    hybrid_engine: Any,
    chunk_document: Any,
    force: bool,
) -> Dict[str, Any]:
    """Process a single PDF. Returns a manifest entry dict."""
    doc_dir = _doc_outdir(out_root, pdf)
    doc_dir.mkdir(parents=True, exist_ok=True)
    entry: Dict[str, Any] = {
        "source_pdf": _path_for_manifest(pdf),
        "out_dir": _path_for_manifest(doc_dir),
    }
    if not force and _is_complete(doc_dir):
        existing = doc_dir / "meta.json"
        try:
            cached = json.loads(existing.read_text(encoding="utf-8"))
            entry.update({"status": "skipped_complete", **cached})
            return entry
        except Exception:
            entry.update({"status": "skipped_complete"})
            return entry

    t0 = time.time()
    try:
        v2_doc = hybrid_engine.extract(str(pdf))
        decisions = list(getattr(hybrid_engine, "last_routing_decisions", []))
        v3_doc = _translate_v2_to_v3(v2_doc)
        chunks = chunk_document(v3_doc)
        jsonl_path = doc_dir / "ingestion.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(chunk.model_dump_json())
                fh.write("\n")
        elapsed = time.time() - t0
        routing = _routing_summary(decisions)
        meta: Dict[str, Any] = {
            "status": "ok",
            "source_pdf": entry["source_pdf"],
            "doc_id": v2_doc.doc_id,
            "page_count": len(v3_doc.pages),
            "chunk_count": len(chunks),
            "routing": routing,
            "routing_decisions": [
                {"page": pn, "engine": choice, "reason": reason}
                for pn, choice, reason in decisions
            ],
            "elapsed_seconds": round(elapsed, 3),
        }
        (doc_dir / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        logger.info(
            "OK %s | pages=%d chunks=%d vlm=%d docling=%d fallback=%d %.1fs",
            pdf.name,
            meta["page_count"],
            meta["chunk_count"],
            routing["vlm"],
            routing["docling"],
            routing["docling_fallback"],
            elapsed,
        )
        entry.update(
            {
                "status": "ok",
                "doc_id": meta["doc_id"],
                "page_count": meta["page_count"],
                "chunk_count": meta["chunk_count"],
                "routing": routing,
                "elapsed_seconds": meta["elapsed_seconds"],
            }
        )
    except Exception as exc:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        err_meta = {
            "status": "error",
            "source_pdf": entry["source_pdf"],
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": tb,
            "elapsed_seconds": round(elapsed, 3),
        }
        (doc_dir / "meta.json").write_text(
            json.dumps(err_meta, indent=2), encoding="utf-8"
        )
        logger.exception("FAIL %s | %s: %s", pdf.name, type(exc).__name__, exc)
        entry.update(
            {
                "status": "error",
                "error_type": err_meta["error_type"],
                "error": err_meta["error"],
                "elapsed_seconds": err_meta["elapsed_seconds"],
            }
        )
    return entry


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data",
        help="Corpus root (default: data/)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "output" / "v3_baselines",
        help="Output root (default: output/v3_baselines/)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after N PDFs (alphabetical). 0 or unset = full corpus.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process docs that already have ingestion.jsonl + meta.json.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Skip PDFs with more than N pages (budget guardrail for long-tail "
             "tech-manual books). Unset = no cap.",
    )
    args = parser.parse_args(argv)

    use_vlm = os.environ.get("USE_VLM_ENGINE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if not use_vlm:
        logger.error(
            "USE_VLM_ENGINE is not set — the grand soak must run through the "
            "VLM-routed HybridEngine. Set USE_VLM_ENGINE=1 and retry."
        )
        return 2

    for var in ("VLM_NATIVE_ENDPOINT", "VLM_NATIVE_MODEL", "VLM_NATIVE_API_KEY"):
        if not os.environ.get(var):
            logger.warning(
                "%s is not set in env. VlmProviderConfig.from_env() may fall "
                "back to defaults that hit the wrong endpoint.",
                var,
            )

    # BROKEN as of 2026-05-30: v3_execution_root was retired (sandbox removed).
    # TODO(repoint): this script must move off the sandbox chunker onto the
    # production one. The change is NOT a drop-in — it touches the output format:
    #   1. drop `_translate_v2_to_v3` (the v2->v3-shape translation, ~lines 81-178)
    #      and the `from mmrag_v3.universal.document import ...` it uses;
    #   2. chunk the v2 UniversalDocument directly:
    #        from mmrag_v2.chunking.uir_chunker import chunk_universal_document
    #        uir_chunks = chunk_universal_document(v2_doc, profile_type=...)
    #   3. fix serialization: uir_chunks are mmrag_v2 UIRChunk *dataclasses*
    #      (no .model_dump_json); emit JSONL via IngestionChunk.from_uir(...).
    #      model_dump_json() to keep the shape v3_to_v2_jsonl.py expects.
    # Validate offline (USE_DOCLING_FAST=1 on a 1-page doc) THEN through the full
    # soak. Sandbox recoverable from ~/mmrag_v3_execution_root_backup_2026-05-30.tar.gz.
    v3_src = REPO_ROOT / "v3_execution_root" / "src"
    if str(v3_src) not in sys.path:
        sys.path.insert(0, str(v3_src))
    from mmrag_v3.chunking.chunker import chunk as chunk_document  # type: ignore  # noqa: E501 (see TODO above — sandbox retired)

    _, HybridEngine = _load_phase_c_modules()
    hybrid_engine = HybridEngine()

    out_root: Path = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    data_dir: Path = args.data_dir.resolve()
    pdfs = _collect_pdfs(data_dir, args.limit, args.max_pages)
    if not pdfs:
        logger.error("No PDFs found under %s", args.data_dir)
        return 2

    logger.info(
        "Starting V3 grand soak: %d PDFs → %s",
        len(pdfs),
        out_root,
    )

    manifest_path = out_root / "manifest.json"
    entries: List[Dict[str, Any]] = []
    t_start = time.time()
    for i, pdf in enumerate(pdfs, 1):
        logger.info("[%d/%d] %s", i, len(pdfs), pdf.name)
        entry = _process_one_pdf(
            pdf,
            out_root,
            hybrid_engine,
            chunk_document,
            force=args.force,
        )
        entries.append(entry)
        # Persist manifest incrementally so a crash mid-run is recoverable.
        manifest = {
            "out_dir": _path_for_manifest(out_root),
            "started_at": t_start,
            "elapsed_seconds_so_far": round(time.time() - t_start, 3),
            "doc_count_total": len(pdfs),
            "doc_count_processed": i,
            "entries": entries,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    ok = sum(1 for e in entries if e.get("status") == "ok")
    skipped = sum(1 for e in entries if e.get("status") == "skipped_complete")
    failed = sum(1 for e in entries if e.get("status") == "error")
    total_chunks = sum(int(e.get("chunk_count", 0) or 0) for e in entries)
    routing_total = {"vlm": 0, "docling": 0, "docling_fallback": 0, "other": 0}
    for e in entries:
        r = e.get("routing") or {}
        for k in routing_total:
            routing_total[k] += int(r.get(k, 0) or 0)
    summary = {
        "ok": ok,
        "skipped": skipped,
        "failed": failed,
        "total_docs": len(pdfs),
        "total_chunks": total_chunks,
        "routing_total": routing_total,
        "elapsed_seconds": round(time.time() - t_start, 3),
    }
    manifest = {
        "out_dir": str(out_root.relative_to(REPO_ROOT)),
        "started_at": t_start,
        "summary": summary,
        "entries": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("=== V3 batch ingest done ===")
    logger.info("ok=%d skipped=%d failed=%d", ok, skipped, failed)
    logger.info(
        "total_chunks=%d routing_pages=%s elapsed=%.1fs",
        total_chunks,
        routing_total,
        summary["elapsed_seconds"],
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
