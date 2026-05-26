#!/usr/bin/env python3
"""V3 Phase A task A0 — per-doc spike on ATZ_Elektronik_German.

Charter: docs/ARCHITECTURE_V3_DRAFT_0.5.md Phase A task A0 (3 days
nominal). Acceptance: "Refactor proves out on one doc; semantic-
identity gate passes on this doc alone (both halves); intentional
deltas list ≤30 lines, OR Phase A is renegotiated per protocol above."

This spike's scope is deliberately bounded: it does NOT re-extract
the document through a v3.0 ElementProcessor (that's Phase A task
A2). Instead it:

  1. Loads the existing v2.16 ingestion.jsonl for ATZ_Elektronik_German
  2. Projects each v2.X chunk into a v3.0 UIRChunk via the A0 mapper
  3. Runs the v3 identity-half gate on (v2.X baseline) vs
     (UIRChunk projection re-serialized to v2.X-shape)
  4. Reports identity ratio + delta categories

If identity ratio is 100% (or very close), the v3.0 UIR contract
carries v2.X content losslessly — A2 can proceed with confidence
that the rewrite's semantic-identity target is achievable. If it's
materially below 100%, the gap categorizes the work A2 has to do
(e.g., extraction_method vocabulary normalization, structural_flag
enum gaps, confidence-breakdown sentinel encoding).

This spike does NOT need GPU, omlx, Qdrant, or any external service.
Pure CPU file I/O + Python dataclass projection + hash comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from mmrag_v2.universal.v2x_to_v3_mapper import (
    map_v2x_corpus_to_v3,
    uirchunk_to_identity_projection,
)
from mmrag_v2.v3_identity_gate import compare_for_identity


ATZ_INGESTION_PATH = Path(
    "output/ATZ_Elektronik_German/ingestion.jsonl"
)
ATZ_DOC_ID = "6fccda8bd625"


@dataclass
class A0Report:
    target_doc_id: str
    source_path: str
    v2x_chunk_count: int
    v3_uirchunk_count: int
    identity_ratio: float
    matched: int
    differing: int
    missing: int
    new_in_candidate: int
    sample_differing_keys: List[str] = field(default_factory=list)
    sample_missing_keys: List[str] = field(default_factory=list)
    mapper_errors: List[str] = field(default_factory=list)


def load_v2x_chunks(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("chunk_id"):
                out.append(rec)
    return out


def baseline_projection(chunk: Dict[str, Any], doc_id: str) -> Dict[str, Any]:
    """Project a v2.X chunk into the same identity-comparison shape
    the mapper output uses, so the comparison is apples-to-apples.

    The identity-half gate compares 'identity-relevant projections' per
    Charter §8.2. Both sides must be projected the same way for the
    comparison to be meaningful — otherwise we'd compare raw v2.X
    chunks (with all their metadata) to a stripped projection and
    diff would be spurious.
    """
    metadata = chunk.get("metadata") or {}
    spatial = metadata.get("spatial") or {}
    hierarchy = metadata.get("hierarchy") or {}
    payload: Dict[str, Any] = {
        "doc_id": doc_id,
        "content": chunk.get("content", ""),
        "modality": chunk.get("modality"),
        "structural_flags": [],  # v2.X carries no equivalent yet
    }
    page_number = metadata.get("page_number")
    if page_number is not None:
        payload["page_number"] = int(page_number)
    bbox = spatial.get("bbox") if isinstance(spatial, dict) else None
    if bbox is not None:
        payload["bbox"] = [int(round(float(c))) for c in bbox]
    parent_heading = hierarchy.get("parent_heading") if isinstance(hierarchy, dict) else None
    if parent_heading is not None:
        payload["parent_heading"] = parent_heading
    ocr_conf = metadata.get("ocr_confidence")
    if ocr_conf is not None:
        payload["confidence_breakdown"] = {"ocr_confidence": float(ocr_conf)}
    return payload


def run_a0_spike(
    *,
    ingestion_path: Path,
    target_doc_id: str,
) -> A0Report:
    log = logging.getLogger("v3_a0_atz_spike")

    log.info("Loading v2.16 ingestion.jsonl: %s", ingestion_path)
    v2x = load_v2x_chunks(ingestion_path)
    log.info("Loaded %d v2.X chunks", len(v2x))

    log.info("Projecting v2.X chunks -> v3.0 UIRChunks via A0 mapper")
    mapper_errors: List[str] = []
    try:
        uirchunks = map_v2x_corpus_to_v3(v2x)
    except ValueError as exc:
        mapper_errors.append(str(exc))
        uirchunks = []
    log.info("Mapped %d UIRChunks", len(uirchunks))

    log.info("Projecting both sides into identity-comparison shape")
    baseline = [baseline_projection(c, target_doc_id) for c in v2x]
    candidate = [uirchunk_to_identity_projection(c, target_doc_id) for c in uirchunks]

    log.info(
        "Running identity-half gate (baseline=%d, candidate=%d)",
        len(baseline), len(candidate),
    )
    report = compare_for_identity(
        baseline_chunks=baseline, candidate_chunks=candidate,
    )

    return A0Report(
        target_doc_id=target_doc_id,
        source_path=str(ingestion_path),
        v2x_chunk_count=len(v2x),
        v3_uirchunk_count=len(uirchunks),
        identity_ratio=report.identity_ratio,
        matched=report.matched,
        differing=len(report.differing_baseline_ids),
        missing=len(report.missing_baseline_ids),
        new_in_candidate=len(report.new_candidate_ids),
        sample_differing_keys=report.differing_baseline_ids[:5],
        sample_missing_keys=report.missing_baseline_ids[:5],
        mapper_errors=mapper_errors,
    )


def _format(report: A0Report) -> str:
    pass_threshold = 0.95
    verdict = "PASS" if report.identity_ratio >= pass_threshold else "FAIL"
    delta_count = report.differing + report.missing + report.new_in_candidate
    return "\n".join([
        f"V3 Phase A task A0 — per-doc spike on {report.target_doc_id}",
        f"  source:                  {report.source_path}",
        f"  v2.X chunks:             {report.v2x_chunk_count}",
        f"  v3.0 UIRChunks produced: {report.v3_uirchunk_count}",
        f"  mapper errors:           {len(report.mapper_errors)}",
        "",
        f"  Identity-half gate:",
        f"    matched:               {report.matched}/{report.v2x_chunk_count}",
        f"    differing:             {report.differing}",
        f"    missing in candidate:  {report.missing}",
        f"    new in candidate:      {report.new_in_candidate}",
        f"    identity ratio:        {report.identity_ratio:.4f}",
        f"    threshold:             ≥{pass_threshold:.2f}",
        f"    verdict:               {verdict}",
        "",
        f"  Total deltas to enumerate:  {delta_count}",
        f"  Charter A0 acceptance cap:  ≤30 lines in PHASE_A_INTENTIONAL_DELTAS.md",
        f"    delta-count under cap:    {'YES' if delta_count <= 30 else 'NO'}",
    ])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="V3 Phase A task A0 — per-doc spike on ATZ_Elektronik_German.",
    )
    parser.add_argument("--ingestion", type=Path, default=ATZ_INGESTION_PATH)
    parser.add_argument("--doc-id", type=str, default=ATZ_DOC_ID)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s | %(message)s",
    )

    if not args.ingestion.exists():
        parser.error(f"v2.16 ingestion.jsonl not found: {args.ingestion}")

    report = run_a0_spike(
        ingestion_path=args.ingestion,
        target_doc_id=args.doc_id,
    )

    print(_format(report))
    if args.json_out:
        args.json_out.write_text(
            json.dumps(asdict(report), indent=2),
            encoding="utf-8",
        )
        print(f"\nJSON written: {args.json_out}")

    delta_count = report.differing + report.missing + report.new_in_candidate
    if report.identity_ratio >= 0.95 and delta_count <= 30:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
