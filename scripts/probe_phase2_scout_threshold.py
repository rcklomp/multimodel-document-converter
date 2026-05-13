"""Probe Phase 2: per-batch TextIntegrityScout trigger threshold validation.

For each canonical JSONL/source-PDF pair in `output/<doc>/ingestion.jsonl`,
group emitted TEXT chunks into simulated batches of `--batch-size` pages and
compute the per-batch text-coverage shape:

  per_batch_variance = (source_chars - chunk_chars) / source_chars

A batch is reported as TRIGGER when EITHER of these holds (subject to a
non-trivial source-text floor):

  (A) per_batch_variance >= --variance-pct, AND
      source_chars in the batch >= --min-source-chars

  (B) the batch contains >= --min-missing-pages where the source has
      >= --min-page-source-chars characters and the emitted text chunks
      have < --min-page-chunk-chars characters, AND
      source_chars in the batch >= --min-source-chars

Both rules are universal page-shape rules (no filename or
profile-specific logic). The variance rule catches batches with broad
shortfall; the missing-page-count rule catches batches with isolated
zero-content pages where the rest of the batch is healthy enough to
mask variance.

This probe does NOT modify any chunks; it is read-only. Use it to defend
the per-batch threshold across the 34-doc canonical corpus before wiring
the trigger into BatchProcessor.

Usage:
    python scripts/probe_phase2_scout_threshold.py \
        --output-root output \
        --data-root data \
        --batch-size 10
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    print("ERROR: PyMuPDF (fitz) is required for the probe.", file=sys.stderr)
    sys.exit(2)


@dataclass
class BatchShape:
    batch_index: int
    start_page: int
    end_page: int
    source_chars: int
    chunk_chars: int
    variance_ratio: float
    missing_pages: List[int]


@dataclass
class DocProbe:
    doc_name: str
    jsonl_path: Path
    pdf_path: Optional[Path]
    total_pages: int
    batches: List[BatchShape]


def _load_jsonl_metadata_and_pages(jsonl_path: Path) -> Tuple[dict, Dict[int, int]]:
    """Return (header_metadata, page -> total_chunk_text_chars) for TEXT chunks."""
    header: dict = {}
    chunk_chars_by_page: Dict[int, int] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("object_type") == "ingestion_metadata":
                header = rec
                continue
            if rec.get("modality") != "text":
                continue
            content = rec.get("content") or ""
            md = rec.get("metadata") or {}
            page = md.get("page_number")
            if not isinstance(page, int) or page <= 0:
                continue
            chunk_chars_by_page[page] = chunk_chars_by_page.get(page, 0) + len(content)
    return header, chunk_chars_by_page


def _source_chars_per_page(pdf_path: Path) -> Dict[int, int]:
    """Return page_number (1-indexed) -> raw text length from PyMuPDF.

    Uses raw `get_text("text")` per page; matches what the scout's existing
    raw_text_per_page extraction sees.
    """
    out: Dict[int, int] = {}
    doc = fitz.open(pdf_path)
    try:
        for idx in range(len(doc)):
            page = doc.load_page(idx)
            txt = (page.get_text("text") or "").strip()
            out[idx + 1] = len(txt)
    finally:
        doc.close()
    return out


def _resolve_pdf_for(jsonl_path: Path, data_root: Path, header: dict) -> Optional[Path]:
    """Resolve the source PDF for a JSONL by matching the recorded source_file."""
    src_name = header.get("source_file")
    if not isinstance(src_name, str) or not src_name:
        return None
    if not src_name.lower().endswith(".pdf"):
        return None
    for candidate in data_root.rglob("*.pdf"):
        if candidate.name == src_name:
            return candidate
    return None


def _compute_batches(
    total_pages: int,
    src_chars_per_page: Dict[int, int],
    chunk_chars_by_page: Dict[int, int],
    batch_size: int,
    min_page_source_chars: int,
    min_page_chunk_chars: int,
) -> List[BatchShape]:
    """Bucket pages into simulated batches and compute per-batch shape."""
    out: List[BatchShape] = []
    if batch_size <= 0:
        batch_size = 10
    bi = 0
    for start in range(1, total_pages + 1, batch_size):
        end = min(start + batch_size - 1, total_pages)
        src = 0
        chk = 0
        missing: List[int] = []
        for p in range(start, end + 1):
            psrc = src_chars_per_page.get(p, 0)
            pchk = chunk_chars_by_page.get(p, 0)
            src += psrc
            chk += pchk
            if psrc >= min_page_source_chars and pchk < min_page_chunk_chars:
                missing.append(p)
        variance = ((src - chk) / src) if src > 0 else 0.0
        out.append(
            BatchShape(
                batch_index=bi,
                start_page=start,
                end_page=end,
                source_chars=src,
                chunk_chars=chk,
                variance_ratio=variance,
                missing_pages=missing,
            )
        )
        bi += 1
    return out


def _probe_doc(
    jsonl_path: Path,
    data_root: Path,
    batch_size: int,
    min_page_source_chars: int,
    min_page_chunk_chars: int,
) -> Optional[DocProbe]:
    header, chunk_chars_by_page = _load_jsonl_metadata_and_pages(jsonl_path)
    pdf_path = _resolve_pdf_for(jsonl_path, data_root, header)
    if pdf_path is None:
        return None
    src_chars_per_page = _source_chars_per_page(pdf_path)
    total_pages = (
        header.get("total_pages")
        or max(src_chars_per_page.keys(), default=0)
    )
    if not isinstance(total_pages, int) or total_pages <= 0:
        return None
    batches = _compute_batches(
        total_pages=total_pages,
        src_chars_per_page=src_chars_per_page,
        chunk_chars_by_page=chunk_chars_by_page,
        batch_size=batch_size,
        min_page_source_chars=min_page_source_chars,
        min_page_chunk_chars=min_page_chunk_chars,
    )
    return DocProbe(
        doc_name=jsonl_path.parent.name,
        jsonl_path=jsonl_path,
        pdf_path=pdf_path,
        total_pages=total_pages,
        batches=batches,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", default="output", help="Root with per-doc ingestion.jsonl outputs")
    p.add_argument("--data-root", default="data", help="Root with source PDFs")
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--variance-pct", type=float, default=0.30,
                   help="Per-batch token-balance variance threshold (0.30 = 30%%)")
    p.add_argument("--min-source-chars", type=int, default=500,
                   help="Skip batches whose source text is below this floor")
    p.add_argument("--min-missing-pages", type=int, default=2,
                   help="Minimum 'missing pages' per batch for the trigger to fire")
    p.add_argument("--min-page-source-chars", type=int, default=100,
                   help="A page counts as non-blank-source above this char floor")
    p.add_argument("--min-page-chunk-chars", type=int, default=50,
                   help="A page counts as missing-chunk-content below this char floor")
    p.add_argument("--report-dir", default="output/probe_phase2_scout_threshold")
    args = p.parse_args()

    output_root = Path(args.output_root)
    data_root = Path(args.data_root)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    jsonls = sorted(p for p in output_root.glob("*/ingestion.jsonl") if p.is_file())
    if not jsonls:
        print(f"No ingestion.jsonl files under {output_root}", file=sys.stderr)
        return 2

    rows: List[dict] = []
    print(
        f"Probing {len(jsonls)} JSONL outputs with batch_size={args.batch_size}, "
        f"variance>={args.variance_pct:.0%}, min_missing_pages>={args.min_missing_pages}",
        flush=True,
    )

    for jsonl in jsonls:
        probe = _probe_doc(
            jsonl_path=jsonl,
            data_root=data_root,
            batch_size=args.batch_size,
            min_page_source_chars=args.min_page_source_chars,
            min_page_chunk_chars=args.min_page_chunk_chars,
        )
        if probe is None:
            print(f"  - SKIP {jsonl.parent.name}: no PDF source resolvable", flush=True)
            continue

        triggers = [
            b
            for b in probe.batches
            if b.source_chars >= args.min_source_chars
            and (
                b.variance_ratio >= args.variance_pct
                or len(b.missing_pages) >= args.min_missing_pages
            )
        ]
        rows.append(
            {
                "doc": probe.doc_name,
                "pdf": str(probe.pdf_path),
                "total_pages": probe.total_pages,
                "batch_count": len(probe.batches),
                "trigger_count": len(triggers),
                "triggers": [
                    {
                        "batch_index": b.batch_index,
                        "start_page": b.start_page,
                        "end_page": b.end_page,
                        "source_chars": b.source_chars,
                        "chunk_chars": b.chunk_chars,
                        "variance_ratio": round(b.variance_ratio, 4),
                        "missing_pages": b.missing_pages,
                    }
                    for b in triggers
                ],
            }
        )

        status = "FIRE" if triggers else "clean"
        details = ""
        if triggers:
            details = "; ".join(
                f"batch {t.batch_index+1} pp{t.start_page}-{t.end_page} "
                f"var={t.variance_ratio:.2%} missing={t.missing_pages}"
                for t in triggers
            )
        print(f"  - {status:5s} {probe.doc_name}: {details or 'no batches trigger'}", flush=True)

    summary_path = report_dir / "probe_summary.json"
    summary_path.write_text(json.dumps(
        {
            "params": vars(args),
            "rows": rows,
        },
        indent=2,
    ))
    print(f"\nSummary written: {summary_path}", flush=True)

    n_fire = sum(1 for r in rows if r["trigger_count"] > 0)
    n_total = len(rows)
    print(
        f"\nDocs probed: {n_total}; docs with >=1 firing batch: {n_fire}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
