#!/usr/bin/env python3
"""Three-way MinerU serving probe (WP-B; HANDOVER_OVERNIGHT_RENDERCAP_AND_PROBES).

Per-box numbers for the SAME work so the MinerU serving home (M5 mlx vs GX10 vLLM
vs Mini mlx) can be chosen on evidence. Drives the SHIPPING engine path
(``MineruNativeEngine`` -> ``extract_page_mineru``), NOT raw HTTP, so the Phase 0.5
WP2 bounded retry-before-fallback is exercised on every page. Reads only; persists
no chunks - just a per-box results JSON the morning report turns into a table.

What it measures, for a fixed 5-page set spanning dense-table / form / magazine /
code-heavy / prose (the phase0a internal manifest pages):

  k=1 (sequential)  per-page wall time, element count, output sanity.
  k=2 / k=4         that many pages in flight via a thread pool around the engine,
                    with a HARD per-page timeout (default 300s) so a stalling
                    server cannot hang the run. Records completions, timeouts,
                    500s, and WP2 retry recoveries (counted off the engine's
                    retry-warning log).

Output sanity per page: non-empty content, element count (compared cross-box at
report time for "in family with M5"), and a degenerate-repetition flag (a run of a
repeated line/token - relevant on the GX10 vLLM box, which is served WITHOUT the
MinerU anti-repetition logits processor).

Endpoint is the MinerU base URL with NO trailing /v1 (the MinerUClient adds it;
PLAN_OMNIDOCBENCH_EVAL 13.2). Model id is the SERVED id per box (GX10 uses
``MinerU2.5-2509-1.2B``; M5/Mini use ``mlx-community/MinerU2.5-2509-1.2B-bf16``).

Usage
-----
  python scripts/mineru_serving_probe.py \
    --box gx10 --endpoint http://10.0.10.239:8001 --model MinerU2.5-2509-1.2B \
    --max-concurrency 4 --out output/wpb/gx10.json
  # M5: never exceed k=2 (k=2 stall already known); run only AFTER the WP-A sweep.
  python scripts/mineru_serving_probe.py \
    --box m5 --endpoint http://10.0.10.235:8000 \
    --model mlx-community/MinerU2.5-2509-1.2B-bf16 --max-concurrency 2 --out ...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz  # PyMuPDF

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from mmrag_v3.engines.mineru_native import (  # noqa: E402
    MineruNativeEngine,
    extract_page_mineru,
)

# Fixed 5-page set: the phase0a internal manifest pages, content-class labelled.
PROBE_PAGES = [
    {
        "klass": "dense_table",
        "pdf": "data/data_spreadsheet/CarOK voorraadtelling 2021-04.pdf",
        "page": 1,
    },
    {
        "klass": "magazine",
        "pdf": "data/digital_magazine/Combat Aircraft - August 2025 UK.pdf",
        "page": 8,
    },
    {
        # Corrected phantom premise: the phase0a internal manifest's code page
        # (index 60) is a BLANK section-break (PyMuPDF chars=0, VLM "no text on
        # this page"); it tests nothing for the code-heavy class. Index 65 is the
        # first genuine dense-code page (a 31-line indented JSON block).
        "klass": "code_heavy",
        "pdf": "data/technical_manual/Ayeva K. Mastering Python Design Patterns...essential Python patterns...3ed 2024.pdf",
        "page": 65,
    },
    {
        "klass": "form",
        "pdf": "data/business_form/betwistingsformulier_aankoop_niet_ontvangen.pdf",
        "page": 0,
    },
    {"klass": "prose_scanned", "pdf": "data/business_form/0013_140302111325_001.pdf", "page": 0},
]


class _RetryCounter(logging.Handler):
    """Counts MinerU transient-fault retry-warning lines off the engine logger so a
    page that logged a retry but still succeeded is recorded as a recovery."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if "transient fault" in record.getMessage():
            self.count += 1


def _page_text(universal_page) -> str:
    return "\n".join((e.content or "") for e in universal_page.elements)


def _repetition_flag(text: str) -> dict:
    """Degenerate-repetition detector: longest run of an identical non-empty line,
    and longest run of an identical whitespace token. Flags GX10-style loops."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    max_line_run = best = 1
    for i in range(1, len(lines)):
        best = best + 1 if lines[i] == lines[i - 1] and len(lines[i]) >= 8 else 1
        max_line_run = max(max_line_run, best)
    toks = text.split()
    max_tok_run = best = 1
    for i in range(1, len(toks)):
        best = best + 1 if toks[i] == toks[i - 1] else 1
        max_tok_run = max(max_tok_run, best)
    flagged = max_line_run >= 5 or max_tok_run >= 30
    return {"flagged": flagged, "max_line_run": max_line_run, "max_tok_run": max_tok_run}


def _extract_one(engine: MineruNativeEngine, spec: dict) -> dict:
    """Render + MinerU extract ONE page via the engine path. Returns timing+sanity."""
    pdf = REPO / spec["pdf"]
    if not pdf.exists():
        return {"klass": spec["klass"], "error": f"missing pdf: {spec['pdf']}"}
    doc = fitz.open(str(pdf))
    try:
        if spec["page"] >= doc.page_count:
            return {"klass": spec["klass"], "error": f"page {spec['page']} out of range"}
        t0 = time.time()
        up = extract_page_mineru(engine, doc[spec["page"]], spec["page"] + 1)
        dt = time.time() - t0
    except Exception as exc:  # noqa: BLE001 - a probe: classify in the table
        return {
            "klass": spec["klass"],
            "error": f"{type(exc).__name__}: {exc}",
            "is_5xx": "500" in str(exc) or "ServerError" in type(exc).__name__,
        }
    finally:
        doc.close()
    text = _page_text(up)
    return {
        "klass": spec["klass"],
        "total_s": round(dt, 2),
        "element_count": len(up.elements),
        "chars": len(text),
        "non_empty": bool(text.strip()),
        "repetition": _repetition_flag(text),
    }


def _run_k1(engine, counter) -> list[dict]:
    rows = []
    for spec in PROBE_PAGES:
        counter.count = 0
        print(f"  [k1] {spec['klass']} ...", flush=True)
        r = _extract_one(engine, spec)
        r["retries"] = counter.count
        rows.append(r)
        if r.get("error"):
            print(f"      ERROR: {r['error']}")
        else:
            print(
                f"      {r['total_s']}s  elements={r['element_count']} chars={r['chars']} "
                f"rep={r['repetition']['flagged']} retries={r['retries']}"
            )
    return rows


def _run_concurrent(engine, k: int, per_page_timeout: int) -> dict:
    """k pages in flight via a thread pool, HARD per-future timeout. A timed-out
    future is abandoned (the worker thread cannot be force-killed) and recorded."""
    print(
        f"  [k{k}] {len(PROBE_PAGES)} pages, {k} in flight, {per_page_timeout}s/page cap ...",
        flush=True,
    )
    completions = timeouts = errors = fivexx = 0
    per: list[float] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=k) as ex:
        futs = {ex.submit(_extract_one, engine, spec): spec for spec in PROBE_PAGES}
        for fut in as_completed(futs, timeout=None):
            try:
                r = fut.result(timeout=per_page_timeout)
            except Exception as exc:  # noqa: BLE001 - future timeout or worker raise
                timeouts += 1
                print(f"      TIMEOUT/err {futs[fut]['klass']}: {type(exc).__name__}")
                continue
            if r.get("error"):
                errors += 1
                fivexx += 1 if r.get("is_5xx") else 0
            else:
                completions += 1
                per.append(r["total_s"])
    wall = round(time.time() - t0, 2)
    return {
        "k": k,
        "wall_s": wall,
        "completions": completions,
        "timeouts": timeouts,
        "errors": errors,
        "fivexx": fivexx,
        "per_page_mean_s": round(sum(per) / len(per), 2) if per else None,
        "pages_per_hr": round(3600 * completions / wall, 1) if wall else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--box", required=True, help="box label for the results file (m5/gx10/mini)")
    ap.add_argument("--endpoint", required=True, help="MinerU base URL, NO trailing /v1")
    ap.add_argument("--model", required=True, help="served model id for this box")
    ap.add_argument("--max-concurrency", type=int, default=4, help="cap concurrency level (M5: 2)")
    ap.add_argument("--per-page-timeout", type=int, default=300, help="hard per-page cap (s)")
    ap.add_argument("--out", required=True, help="results JSON path")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    counter = _RetryCounter()
    logging.getLogger("mmrag_v3.engines.mineru_native").addHandler(counter)

    engine = MineruNativeEngine(server_url=args.endpoint.rstrip("/"), model_name=args.model)

    print(
        f"===== MinerU serving probe: box={args.box} endpoint={args.endpoint} model={args.model} ====="
    )
    print("\n--- k=1 sequential (per-page latency + output sanity) ---")
    k1 = _run_k1(engine, counter)

    concurrency = {}
    for k in (2, 4):
        if k > args.max_concurrency:
            print(f"\n--- k={k} SKIPPED (max-concurrency={args.max_concurrency}) ---")
            continue
        print(f"\n--- k={k} concurrent ---")
        concurrency[f"k{k}"] = _run_concurrent(engine, k, args.per_page_timeout)

    ok = [r for r in k1 if not r.get("error")]
    k1_lat = [r["total_s"] for r in ok]
    summary = {
        "box": args.box,
        "endpoint": args.endpoint,
        "model": args.model,
        "max_concurrency": args.max_concurrency,
        "k1_pages_ok": len(ok),
        "k1_pages_total": len(k1),
        "k1_mean_s": round(sum(k1_lat) / len(k1_lat), 2) if k1_lat else None,
        "k1_max_s": max(k1_lat) if k1_lat else None,
        "k1_pages_per_hr": round(3600 / (sum(k1_lat) / len(k1_lat)), 1) if k1_lat else None,
        "any_repetition": any(r.get("repetition", {}).get("flagged") for r in ok),
        "any_empty": any(not r.get("non_empty", True) for r in ok),
        "total_retries": sum(r.get("retries", 0) for r in k1),
        "k1": k1,
        "concurrency": concurrency,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print("\n===== PROBE SUMMARY =====")
    print(
        f"box={args.box}  k1 mean={summary['k1_mean_s']}s max={summary['k1_max_s']}s "
        f"pages/hr(k1)={summary['k1_pages_per_hr']}"
    )
    for k, c in concurrency.items():
        print(
            f"{k}: wall={c['wall_s']}s completions={c['completions']} timeouts={c['timeouts']} "
            f"500s={c['fivexx']} pages/hr={c['pages_per_hr']}"
        )
    print(
        f"repetition_flagged={summary['any_repetition']} empty_pages={summary['any_empty']} "
        f"retries={summary['total_retries']}"
    )
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
