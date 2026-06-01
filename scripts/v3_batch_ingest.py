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
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    from mmrag_v2.universal.asset_materializer import materialize_visual_assets
    from mmrag_v3.engines.vlm_provider import VlmInfraError

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
        # Vision-native IMAGE/TABLE chunks describe regions but carry no binary
        # asset; render the region crops to <doc>/assets/ and set asset_ref so
        # they satisfy QA-CHECK-05. The production batch path does the same via
        # BatchProcessor._render_visual_assets -> the same shared helper. Without
        # this step the crucible soak produced 0 valid baselines. Whole-doc run,
        # so page_offset=0; doc_id is the stable asset-filename hash.
        crop_audit = materialize_visual_assets(
            uir_chunks,
            pdf,
            doc_dir / "assets",
            doc_hash=doc_id,
            page_offset=0,
        )
        if crop_audit.exceeds_threshold:
            logger.warning(
                "%s: %s - %d/%d crops show drift (edge-clamp/blank), rate=%.0f%% "
                "> %.0f%% threshold; suspect crops in meta.json "
                "crop_audit.suspect_assets",
                pdf.name,
                crop_audit.gate_status,
                crop_audit.drift_flagged,
                crop_audit.rendered,
                crop_audit.drift_rate * 100,
                crop_audit.warn_threshold * 100,
            )
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
            "crop_audit": crop_audit.to_dict(),
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
                "crop_audit_gate": crop_audit.gate_status,
                "elapsed_seconds": meta["elapsed_seconds"],
            }
        )
    except VlmInfraError:
        # CIRCUIT BREAKER. Infra/transport outage is NOT a per-doc data
        # error. Do NOT write an error stub and continue - every later doc
        # would silently degrade to Docling and produce corrupt baselines.
        # Propagate to main(), which halts the batch with exit 1.
        raise
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


def _wait_for_vlm_recovery(
    probe_fn: Callable[[], bool],
    *,
    poll_interval_s: float,
    recovery_ceiling_s: float,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> bool:
    """Poll the VLM endpoint until it recovers or the ceiling elapses.

    Returns True if ``probe_fn()`` returned True within ``recovery_ceiling_s``
    (resume the batch), or False once that many seconds of continuous
    unreachability have elapsed (hard-fail). Polls every ``poll_interval_s``.
    """
    # Immediate probe: the endpoint may have come back during the failed retries
    # that produced the VlmInfraError.
    if probe_fn():
        return True
    waited = 0.0
    while waited < recovery_ceiling_s:
        sleep_fn(poll_interval_s)
        waited += poll_interval_s
        if probe_fn():
            logger.warning("VLM endpoint recovered after ~%.0fs down; resuming soak.", waited)
            return True
        logger.warning(
            "VLM endpoint still down after ~%.0fs of %.0fs ceiling; re-probing in %.0fs.",
            waited,
            recovery_ceiling_s,
            poll_interval_s,
        )
    logger.critical(
        "VLM endpoint unreachable for the full %.0fs ceiling; giving up.",
        recovery_ceiling_s,
    )
    return False


def _process_with_resilience(
    pdf: Path,
    process_fn: Callable[[Path], Dict[str, Any]],
    *,
    strict: bool,
    probe_fn: Callable[[], bool],
    poll_interval_s: float,
    recovery_ceiling_s: float,
    max_resume_attempts: int,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Run ``process_fn(pdf)``, applying the breaker policy on VlmInfraError.

    Returns ``(entry, None)`` on success, or ``(None, halt_reason)`` when the
    breaker decides to hard-fail. Two bounded guards ensure the unattended soak
    never hangs:

    * ``recovery_ceiling_s`` - max continuous endpoint unreachability per wait
      (the "truly dead machine" case).
    * ``max_resume_attempts`` - max infra failures on a single doc before giving
      up (the "HTTP-up-but-inference-dead" flap, which the ceiling alone misses
      because the health probe keeps returning 200).

    ``strict=True`` restores the original behavior: hard-fail on the first
    VlmInfraError with no polling. On recovery the doc is retried from scratch
    (its partial in-memory work is discarded; completed docs on disk are skipped
    on resume).
    """
    from mmrag_v3.engines.vlm_provider import VlmInfraError

    attempts = 0
    while True:
        try:
            return process_fn(pdf), None
        except VlmInfraError as exc:
            if strict:
                return None, f"strict breaker: {type(exc).__name__}: {exc}"
            attempts += 1
            if attempts > max_resume_attempts:
                return None, (
                    f"exceeded {max_resume_attempts} resume attempts on this doc "
                    f"(endpoint flapping / inference dead?): "
                    f"{type(exc).__name__}: {exc}"
                )
            logger.warning(
                "RESILIENT BREAKER on %s: VLM infra failure (%s). Resume attempt "
                "%d/%d - polling endpoint every %.0fs (ceiling %.0fs) before "
                "resuming.",
                pdf.name,
                exc,
                attempts,
                max_resume_attempts,
                poll_interval_s,
                recovery_ceiling_s,
            )
            recovered = _wait_for_vlm_recovery(
                probe_fn,
                poll_interval_s=poll_interval_s,
                recovery_ceiling_s=recovery_ceiling_s,
                sleep_fn=sleep_fn,
            )
            if not recovered:
                return None, (
                    f"endpoint unreachable past {recovery_ceiling_s:.0f}s ceiling: "
                    f"{type(exc).__name__}: {exc}"
                )
            # else: loop and retry the same doc from scratch.


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
    parser.add_argument(
        "--strict-breaker",
        action="store_true",
        help="Hard-fail the batch on the FIRST VlmInfraError (no pause-and-poll). "
             "Default is the resilient breaker: poll the endpoint and resume on "
             "recovery. Use for short attended runs (e.g. the Crucible Subset).",
    )
    parser.add_argument(
        "--vlm-poll-interval-s",
        type=float,
        default=60.0,
        help="Resilient mode: seconds between endpoint health probes (default 60).",
    )
    parser.add_argument(
        "--vlm-recovery-ceiling-s",
        type=float,
        default=1800.0,
        help="Resilient mode: hard-fail after this many seconds of continuous "
             "endpoint unreachability (default 1800 = 30 min).",
    )
    parser.add_argument(
        "--vlm-max-resume-attempts",
        type=int,
        default=5,
        help="Resilient mode: max infra failures on a single doc before "
             "hard-failing (guards against an HTTP-up-but-inference-dead flap "
             "the ceiling alone would miss; default 5).",
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

    # Health-probe bound to the configured VLM endpoint, lazily built on the
    # first infra failure (never constructed in strict/docling-fast runs).
    _probe_state: Dict[str, Any] = {}

    def _probe_fn() -> bool:
        prov = _probe_state.get("provider")
        if prov is None:
            from mmrag_v3.engines.vlm_provider import VlmProvider, VlmProviderConfig

            prov = VlmProvider(VlmProviderConfig.from_env())
            _probe_state["provider"] = prov
        return prov.probe_health()

    def _halt(halted_on: Path, reason: str, processed: int) -> None:
        logger.critical(
            "HALTING batch on %s: %s. Completed docs are skipped automatically "
            "on resume.",
            halted_on.name,
            reason,
        )
        manifest_path.write_text(
            json.dumps(
                {
                    "out_dir": _path_for_manifest(out_root),
                    "started_at": t_start,
                    "status": "halted_circuit_breaker",
                    "halted_on": _path_for_manifest(halted_on),
                    "halt_reason": reason,
                    "elapsed_seconds_so_far": round(time.time() - t_start, 3),
                    "doc_count_total": len(pdfs),
                    "doc_count_processed": processed,
                    "entries": entries,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    breaker_mode = "strict" if args.strict_breaker else "resilient"
    logger.info(
        "VLM circuit breaker: %s (poll=%.0fs, ceiling=%.0fs, max-resume=%d)",
        breaker_mode,
        args.vlm_poll_interval_s,
        args.vlm_recovery_ceiling_s,
        args.vlm_max_resume_attempts,
    )

    for i, pdf in enumerate(pdfs, 1):
        logger.info("[%d/%d] %s", i, len(pdfs), pdf.name)
        entry, halt_reason = _process_with_resilience(
            pdf,
            lambda p: _process_one_pdf(
                p, out_root, hybrid_engine, chunk_document, force=args.force
            ),
            strict=args.strict_breaker,
            probe_fn=_probe_fn,
            poll_interval_s=args.vlm_poll_interval_s,
            recovery_ceiling_s=args.vlm_recovery_ceiling_s,
            max_resume_attempts=args.vlm_max_resume_attempts,
        )
        if halt_reason is not None:
            _halt(pdf, halt_reason, i - 1)
            return 1
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
