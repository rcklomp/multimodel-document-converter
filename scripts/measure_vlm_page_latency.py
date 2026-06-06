#!/usr/bin/env python
"""Measure per-page VLM extraction latency vs image density (calibration).

Drives the V3 HybridEngine router per page over a PDF slice, timing each
VLM-routed ``describe()`` call, to collect the density -> latency data needed to
calibrate adaptive batch-size / VLM-timeout thresholds (the 2026-06-03 dense-doc
follow-up). Read-only: it renders pages and calls the VLM but persists NO chunks
and writes only a per-page CSV/JSONL of measurements.

Honors the same env as the production path (``VLM_NATIVE_*``); structured output
follows the shipped default (off for self-hosted) unless
``VLM_NATIVE_STRUCTURED_OUTPUT`` is set.

Usage:
  VLM_NATIVE_ENDPOINT=http://macbook-pro-m5.lan:8000/v1 \
  VLM_NATIVE_MODEL=mlx-community/Qwen3-VL-8B-Instruct-8bit \
  VLM_NATIVE_API_KEY=local \
  python scripts/measure_vlm_page_latency.py <pdf> --from-page 20 --to-page 32 \
      --out output/live_check/latency_combat_interior.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import fitz

from mmrag_v3.engines.router import HybridEngine
from mmrag_v3.engines.vlm_native import (
    _build_schema_prompt,
    estimate_output_budget,
    render_page_png,
)
from mmrag_v3.engines.vlm_provider import (
    VlmInfraError,
    VlmProvider,
    VlmProviderConfig,
    VlmProviderError,
    VlmTruncationError,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--from-page", type=int, default=1, help="1-indexed, inclusive")
    ap.add_argument("--to-page", type=int, default=10, help="1-indexed, inclusive")
    ap.add_argument("--out", default=None, help="JSONL output path")
    args = ap.parse_args()

    engine = HybridEngine()
    cfg = VlmProviderConfig.from_env()
    provider = VlmProvider(cfg)
    print(
        f"# endpoint={cfg.endpoint} model={cfg.model} "
        f"structured={cfg.structured_output_mode} timeout={cfg.timeout_seconds}s "
        f"floor={cfg.max_completion_tokens}"
    )

    doc = fitz.open(args.pdf)
    lo = max(0, args.from_page - 1)
    hi = min(args.to_page, doc.page_count)
    rows = []
    for i in range(lo, hi):
        page = doc[i]
        choice, reason = engine._classify_page(page)
        budget = estimate_output_budget(page)
        row = {
            "page": i + 1,
            "route": choice,
            "reason": reason,
            "est_budget": budget,
            "n_images": len(page.get_images()),
            "n_drawings": len(page.get_drawings()),
            "text_chars": len(page.get_text("text")),
        }
        if choice == "vlm":
            img, pw, ph = render_page_png(page)
            prompt = _build_schema_prompt(pw, ph)
            t0 = time.time()
            status, out_chars = "ok", 0
            try:
                out = provider.describe(img, prompt, mime="image/png", max_tokens=budget)
                out_chars = len(out)
            except VlmTruncationError as exc:
                status, out_chars = "truncated", len(exc.partial_content)
            except VlmInfraError:
                status = "infra_or_timeout"
            except VlmProviderError:
                status = "semantic_fail"
            row["latency_s"] = round(time.time() - t0, 1)
            row["status"] = status
            row["out_chars"] = out_chars
        else:
            row["latency_s"] = 0.0
            row["status"] = "docling"
            row["out_chars"] = 0
        rows.append(row)
        print(json.dumps(row), flush=True)
    doc.close()

    vlm_rows = [r for r in rows if r["route"] == "vlm"]
    if vlm_rows:
        lat = sorted(r["latency_s"] for r in vlm_rows)
        over = [r["page"] for r in vlm_rows if r["latency_s"] >= cfg.timeout_seconds]
        print(
            f"# VLM pages={len(vlm_rows)} "
            f"min={lat[0]}s median={lat[len(lat)//2]}s max={lat[-1]}s "
            f"over_{int(cfg.timeout_seconds)}s={len(over)} pages={over}"
        )

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        print(f"# wrote {len(rows)} rows -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
