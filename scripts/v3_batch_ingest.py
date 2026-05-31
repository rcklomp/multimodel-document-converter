#!/usr/bin/env python3
"""V3 Grand Soak — full-corpus batch ingest through HybridEngine.

Runs the SHIPPING path (PLAN_V3.1 P1 — single extraction path):
``HybridEngine.extract`` → ``UniversalDocument`` →
``mmrag_v2.chunking.uir_chunker.chunk_universal_document`` →
``IngestionChunk.from_uir``. The output JSONL is byte-shape-identical
to what ``mmrag-v2 process`` emits (one ``IngestionChunk`` JSON object
per line). No sandbox translation, no retired-sandbox import.

For each PDF in ``data/``:
    1. Extract via HybridEngine (cost-optimizer router → VLM for
       visually-complex pages, fast Docling for prose pages).
    2. Run the UIR-native chunker → ``UIRChunk`` objects.
    3. Serialize via ``IngestionChunk.from_uir`` → one JSON object/line.
    4. Write ``output/v3_baselines/<doc_stem>/ingestion.jsonl``.
    5. Record routing decisions + timings to ``meta.json``.

The retired Phase-A sandbox chunker is no longer imported.

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
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("v3_batch_ingest")


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
    """Process a single PDF. Returns a manifest entry dict.

    Shipping path: ``hybrid_engine.extract`` → ``UniversalDocument`` →
    ``chunk_universal_document`` → ``IngestionChunk.from_uir``.
    """
    from mmrag_v2.schema.ingestion_schema import FileType, IngestionChunk

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
        universal_doc = hybrid_engine.extract(str(pdf))
        decisions = list(getattr(hybrid_engine, "last_routing_decisions", []))
        uir_chunks = chunk_document(universal_doc)
        doc_id = universal_doc.doc_id or universal_doc.compute_doc_id()
        chunks = [
            IngestionChunk.from_uir(
                uir,
                doc_id=doc_id,
                source_file=pdf.name,
                file_type=FileType.PDF,
                position=position,
            )
            for position, uir in enumerate(uir_chunks)
        ]
        jsonl_path = doc_dir / "ingestion.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False))
                fh.write("\n")
        elapsed = time.time() - t0
        routing = _routing_summary(decisions)
        meta: Dict[str, Any] = {
            "status": "ok",
            "source_pdf": entry["source_pdf"],
            "doc_id": doc_id,
            "page_count": universal_doc.total_pages,
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

    def _truthy(name: str) -> bool:
        return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}

    use_vlm = _truthy("USE_VLM_ENGINE")
    docling_fast = _truthy("USE_DOCLING_FAST")
    if not use_vlm and not docling_fast:
        logger.error(
            "Neither USE_VLM_ENGINE nor USE_DOCLING_FAST is set — the grand "
            "soak must run through the VLM-routed HybridEngine (set "
            "USE_VLM_ENGINE=1). For an offline smoke (no VLM credits), set "
            "USE_DOCLING_FAST=1 to route every page through the fast Docling "
            "engine."
        )
        return 2

    if use_vlm:
        for var in ("VLM_NATIVE_ENDPOINT", "VLM_NATIVE_MODEL", "VLM_NATIVE_API_KEY"):
            if not os.environ.get(var):
                logger.warning(
                    "%s is not set in env. VlmProviderConfig.from_env() may fall "
                    "back to defaults that hit the wrong endpoint.",
                    var,
                )

    # PLAN_V3.1 P1: single extraction path. Chunk the v2 UniversalDocument
    # directly with the production UIR-native chunker; serialization to
    # IngestionChunk JSONL happens in _process_one_pdf via from_uir.
    from mmrag_v2.chunking.uir_chunker import chunk_universal_document as chunk_document

    # Engine selection mirrors mmrag_v3.processor.extract precedence: the
    # full soak uses HybridEngine (cost-optimizer routing + routing
    # decisions); USE_DOCLING_FAST forces the offline Docling route.
    if docling_fast and not use_vlm:
        from mmrag_v3.engines.docling_fast import DoclingFastEngine

        hybrid_engine = DoclingFastEngine()
    else:
        from mmrag_v3.engines.router import HybridEngine

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
        "out_dir": _path_for_manifest(out_root),
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
