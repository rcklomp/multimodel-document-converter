#!/usr/bin/env python3
"""Advisory crop-fidelity report from the WP-A1 ``crop_audit.json`` sidecar.

PLAN_QUALITY_REMEDIATION_V1 WP-A1: the size of a crop is frame-dependent and a wrong-object crop
(the page-header logo standing in for a figure, a text-strip raster inside a flowchart) reads as
"an image was produced" to every count-based check. This report reads the frame-invariant sidecar
(PDF-point rectangles) and the source PDF and flags the crops a human should look at:

  tiny_rescue      the B1 geometric pick replaced the VLM box with an object under 3000 pt^2
                   (a logo / icon sprite; IRJET Fig 3 was a 67x68 header logo)
  prose_dominated  crop_prose_fraction >= 0.5: at least half of the text-layer words inside the
                   rendered rectangle sit on lines of >= 8 words, i.e. the crop is body prose and
                   the VLM box landed on the wrong region (images and tables)
  full_page        the crop degraded to a full-page render

Advisory only (exit 0); AGENT-GATE-PROGRESSION: a new gate starts advisory and is promoted only
after it has run clean on real output. Usage:

  python scripts/qa_crop_fidelity.py --crop-audit output/<run>/crop_audit.json \\
      --source-pdf data/<cat>/<file>.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from mmrag_v2.universal.asset_materializer import PROSE_DOMINATED_MIN_SHARE, prose_word_share

TINY_RESCUE_MAX_PT2 = 3000.0


def _area(rect: List[float]) -> float:
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def evaluate(records: List[Dict[str, Any]], doc: Optional[Any]) -> Dict[str, Any]:
    flagged: List[Dict[str, Any]] = []
    counts = {"vlm": 0, "geometric": 0, "full_page": 0}
    reasons: Dict[str, int] = {}
    for rec in records:
        counts[rec["crop_source"]] = counts.get(rec["crop_source"], 0) + 1
        reasons[rec.get("crop_reason", "")] = reasons.get(rec.get("crop_reason", ""), 0) + 1
        clip = rec.get("clip_rect_pt")
        if clip is None:
            flagged.append({**rec, "flag": "full_page"})
            continue
        if (
            rec["modality"] == "image"
            and rec.get("crop_reason") == "geometric_rescue"
            and _area(clip) < TINY_RESCUE_MAX_PT2
        ):
            flagged.append({**rec, "flag": "tiny_rescue"})
            continue
        if doc is not None and 0 < rec["page"] <= doc.page_count:
            import fitz

            frac = prose_word_share(doc[rec["page"] - 1], fitz.Rect(clip))
            if frac is not None and frac >= PROSE_DOMINATED_MIN_SHARE:
                flagged.append({**rec, "flag": "prose_dominated", "prose_fraction": round(frac, 2)})
    return {"total": len(records), "sources": counts, "reasons": reasons, "flagged": flagged}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--crop-audit", type=Path, required=True)
    ap.add_argument("--source-pdf", type=Path, default=None)
    args = ap.parse_args(argv)

    records = json.loads(args.crop_audit.read_text(encoding="utf-8"))["crops"]
    doc = None
    if args.source_pdf is not None:
        import fitz

        doc = fitz.open(str(args.source_pdf))
    report = evaluate(records, doc)

    print(f"crops={report['total']} sources={report['sources']} reasons={report['reasons']}")
    for item in report["flagged"]:
        extra = f" prose_fraction={item['prose_fraction']}" if "prose_fraction" in item else ""
        print(
            f"CROP_FIDELITY_FLAG {item['flag']} page={item['page']} asset={item['asset_ref']} "
            f"asset_px={item['asset_px']} clip_pt={item['clip_rect_pt']}{extra}"
        )
    print("CROP_FIDELITY_ADVISORY" if report["flagged"] else "CROP_FIDELITY_CLEAN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
