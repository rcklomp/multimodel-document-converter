#!/usr/bin/env python3
"""V3.0 UIR JSONL exporter CLI — Phase A task A2 (shim path).

Charter: docs/ARCHITECTURE_V3_DRAFT_0.5.md Phase A scope-negotiation
option (b) UIR-shim fallback (recorded 2026-05-27 in
docs/PHASE_A_SCOPE_NEGOTIATION.md).

Reads a v2.X ingestion.jsonl, projects every chunk to a v3.0 UIRChunk
via the proven-lossless `v2x_to_v3_mapper`, writes a parallel UIR
JSONL, and verifies the identity-half gate (Charter §3.2) on the
result. By construction the identity ratio should be 1.0000 unless
the mapper cannot handle some v2.X chunk shape — in which case the
exporter raises ValueError citing the mapper bug.

Single-doc usage:
    python scripts/v3_export_uir.py output/ATZ_Elektronik_German/ingestion.jsonl

Multi-doc spot-verification (A5 acceptance):
    python scripts/v3_export_uir.py \\
        --multi output/ATZ_Elektronik_German/ingestion.jsonl \\
        --multi output/Earthship_Vol1.phase3_baseline/ingestion.jsonl \\
        --multi output/Fluent_Python/ingestion.jsonl \\
        --multi output/HarryPotter_and_the_Sorcerers_Stone/ingestion.jsonl \\
        --report docs/V3_PHASE_A_A2_SHIM_REPORT.json

Pure CPU, no GPU, no omlx, no Qdrant.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from mmrag_v2.universal.uir_exporter import (
    ExportReport,
    export_v2x_jsonl_to_uir,
)


logger = logging.getLogger("v3_export_uir")


def _default_output_path(input_path: Path) -> Path:
    """For input `<dir>/ingestion.jsonl` write `<dir>/v3_uir.jsonl`."""
    return input_path.parent / "v3_uir.jsonl"


def _format_one(report: ExportReport) -> str:
    verdict = "PASS" if report.identity_passes else "FAIL"
    lines = [
        f"  source:                 {report.source_path}",
        f"  output:                 {report.output_path}",
        f"  doc_id:                 {report.doc_id}",
        f"  v2.X chunks in:         {report.v2x_chunk_count}",
        f"  v3 UIRChunks out:       {report.uir_chunk_count}",
        f"  mapper errors:          {len(report.mapper_errors)}",
    ]
    if report.identity_ratio is not None:
        lines.extend([
            f"  identity ratio:         {report.identity_ratio:.4f}",
            f"  identity matched:       {report.identity_matched}",
            f"  identity differing:     {report.identity_differing}",
            f"  identity missing:       {report.identity_missing}",
            f"  identity new:           {report.identity_new}",
            f"  verdict (>=0.95):       {verdict}",
        ])
    else:
        lines.append("  identity-half gate:     SKIPPED (--no-verify)")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "input", type=Path, nargs="?",
        help="Single v2.X ingestion.jsonl to convert (omit when using --multi)",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="UIR JSONL output path (default: <input-dir>/v3_uir.jsonl)",
    )
    p.add_argument(
        "--multi", action="append", default=[],
        type=Path, dest="multi",
        help="Run on multiple inputs (repeat the flag). Outputs default to "
             "<each-input-dir>/v3_uir.jsonl.",
    )
    p.add_argument(
        "--doc-id", type=str, default=None,
        help="Explicit doc_id override (default: infer from chunks)",
    )
    p.add_argument(
        "--no-verify", action="store_true",
        help="Skip identity-half gate verification (faster batch runs).",
    )
    p.add_argument(
        "--report", type=Path, default=None,
        help="Write aggregate JSON report to this path.",
    )
    p.add_argument(
        "--log-level", type=str, default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s | %(message)s",
    )

    inputs: List[Path] = list(args.multi)
    if args.input is not None:
        inputs.append(args.input)
    if not inputs:
        p.error("Must provide either a positional input or --multi <path> (repeatable)")

    if args.output is not None and len(inputs) != 1:
        p.error("--output is only valid with a single input; use default paths for --multi")

    reports: List[ExportReport] = []
    any_fail = False

    print(f"\nV3 Phase A task A2 — UIR-shim export ({len(inputs)} input(s))\n")

    for input_path in inputs:
        if not input_path.exists():
            logger.error("input not found: %s", input_path)
            any_fail = True
            continue
        output_path = (
            args.output if args.output is not None
            else _default_output_path(input_path)
        )
        print(f"[{input_path.name}]")
        try:
            report = export_v2x_jsonl_to_uir(
                input_path=input_path,
                output_path=output_path,
                doc_id=args.doc_id,
                verify_identity=not args.no_verify,
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.error("export failed for %s: %s", input_path, exc)
            print(f"  FAIL: {exc}")
            any_fail = True
            continue
        reports.append(report)
        print(_format_one(report))
        if not args.no_verify and not report.identity_passes:
            any_fail = True
        print()

    if args.report:
        agg = {
            "doc_count": len(reports),
            "all_pass": (not any_fail) and all(
                r.identity_passes if r.identity_ratio is not None else True
                for r in reports
            ),
            "reports": [asdict(r) for r in reports],
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(agg, indent=2), encoding="utf-8")
        print(f"Aggregate report written: {args.report}")

    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
