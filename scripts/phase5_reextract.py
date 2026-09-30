#!/usr/bin/env python3
"""PLAN_EXTRACTION_FIDELITY_V1 Phase 4 WP-B / Phase 5 corpus re-extraction harness.

Re-extracts documents through the FORMALIZED production hybrid config (route
`mineru_qwen_hybrid`: GX10 MinerU :8001 + M5 Qwen :8000 code lane + cap1600) and
gates every doc with `qa_full_conversion.py --source-pdf`. Per doc it records the
Section 5.4 provenance aggregates (engine, fallback, degraded, recovered), chunk
counts by modality, a V3-path leak count, wall time, and the QA verdict.

It tracks the PRE-NAMED Phase 4 rollback condition over a rolling window of 10
consecutive docs and STOPS re-extraction if it fires:
  - QA_WARN+QA_FAIL rate > 20 percentage points (>= 3 of any 10 consecutive), OR
  - ladder-served pages > 2% of pages (degraded/total).
A fired condition is RECORDED, never "fixed" by rerunning (handover WP-2 step 5).

Modes:
  --smoke   re-extract the two hardest classes (code-dense FluentPython slice +
            Form_0013) and assert engine=mineru_qwen_hybrid + degraded=0 + QA_PASS*
            before any batch (CLAUDE.md verify-before-convert / smoke-the-hardest-case).
  --batch   re-extract the budget-bounded prioritized set, full docs.

No server reconfiguration. Outputs are gitignored (output/, logs/). Harness-only,
safe to commit. Mirrors scripts/phase4_shadow_window.py exactly for the prod env.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "output" / "phase5_reextract"
SMOKE_SRC = OUT_DIR / "_smoke_src"
PY = sys.executable

# The formalized production hybrid config (Phase 4 DECISIONS entry, 2026-06-11).
# Endpoints are overridable via PHASE5_MINERU_ENDPOINT / PHASE5_VLM_ENDPOINT so the
# run can be routed through a localhost relay (/tmp/phase5_relay.py) when the conda
# env cannot reach the LAN servers directly (the 2026-06-11 utun scoped-route fault).
# The relay is transport-only: the served models + routing are identical.
PROD_ENV = {
    "MINERU_ENDPOINT": os.environ.get("PHASE5_MINERU_ENDPOINT", "http://10.0.10.239:8001"),
    "MINERU_MODEL": "MinerU2.5-2509-1.2B",
    "VLM_NATIVE_ENDPOINT": os.environ.get("PHASE5_VLM_ENDPOINT", "http://10.0.10.235:8000/v1"),
    "VLM_NATIVE_MODEL": "mlx-community/Qwen3-VL-8B-Instruct-8bit",
    "VLM_NATIVE_API_KEY": "EMPTY",
    # cap1600 is the shipped VLM_RENDER_MAX_PX default - do NOT override.
    # no USE_* force flag - default precedence routes to mineru_qwen_hybrid.
}
ROUTE_KEYS = [
    "USE_DOCLING_FAST",
    "USE_MINERU_ENGINE",
    "USE_VLM_ENGINE",
    "USE_HYBRID_ENGINE",
    "USE_MINERU_QWEN_HYBRID",
    "MINERU_ENDPOINT",
    "MINERU_MODEL",
    "VLM_NATIVE_ENDPOINT",
    "VLM_NATIVE_MODEL",
    "VLM_NATIVE_API_KEY",
]

# Budget-bounded prioritized crucible subset (smallest + docling-QA_FAIL docs first).
# (name, source pdf relative to data/, canonical output base). Full doc, no slice.
BATCH = [
    ("Form_0013", "business_form/0013_140302111325_001.pdf", "Form_0013"),
    (
        "Form_betwisting",
        "business_form/betwistingsformulier_aankoop_niet_ontvangen.pdf",
        "Form_betwisting",
    ),
    ("Bevestigingsmiddelen", "raw/Bevestigingsmiddelen.pdf", "Bevestigingsmiddelen"),
    (
        "ATZ_Elektronik",
        "technical_report/ATZ.Elektronik.-.Effizientere.Software.Entwicklung.GERMAN.RETAiL.eBOOk-PDFWriters.pdf",
        "ATZ_Elektronik",
    ),
    (
        "IRJET_academic",
        "academic_journal/IRJET_Modeling_of_Solar_PV_system_under.pdf",
        "IRJET_academic",
    ),
    (
        "CarOK_spreadsheet",
        "data_spreadsheet/CarOK voorraadtelling 2021-04.pdf",
        "CarOK_spreadsheet",
    ),
    (
        "Hybrid_EV",
        "academic_journal/Hybrid_electric_vehicles_and_their_challenges.pdf",
        "Hybrid_EV",
    ),
    ("AIOS_academic", "academic_journal/AIOS LLM Agent Operating System.pdf", "AIOS_academic"),
    (
        "DigitaleFotografie",
        "raw/Digitale-Fotografie - Das essentielle Handbuch Februar 2026.pdf",
        "DigitaleFotografie",
    ),
    ("Firearms", "technical_manual/Firearms.pdf", "Firearms"),
    ("CombatAircraft", "digital_magazine/Combat Aircraft - August 2025 UK.pdf", "CombatAircraft"),
    ("PCWorld", "digital_magazine/PCWorld_July_2025_USA.pdf", "PCWorld"),
]

SMOKE = [
    # code-dense slice (p60-74) + the 1-page scanned form
    ("FluentPython_slice", "technical_manual/Fluent Python Luciano Ramalho 2015.pdf", (60, 74)),
    ("Form_0013", "business_form/0013_140302111325_001.pdf", None),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def slice_pdf(src: Path, dst: Path, window) -> int:
    doc = fitz.open(src)
    n = doc.page_count
    if window is None:
        first, last = 1, n
    else:
        first, last = window
        last = min(last, n)
    out = fitz.open()
    out.insert_pdf(doc, from_page=first - 1, to_page=last - 1)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst)
    pages = last - first + 1
    out.close()
    doc.close()
    return pages


def run_one(name: str, src_pdf: Path, out_base: str) -> dict:
    out = OUT_DIR / out_base
    env = {k: v for k, v in os.environ.items() if k not in ROUTE_KEYS}
    env.update(PROD_ENV)
    cmd = [
        PY,
        "-m",
        "mmrag_v2.cli",
        "process",
        str(src_pdf),
        "--output-dir",
        str(out),
        "--batch-size",
        "10",
        "--vision-provider",
        "none",
    ]
    logf = OUT_DIR / f"{out_base}.log"
    t0 = time.time()
    with open(logf, "w") as lf:
        rc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0

    jl = out / "ingestion.jsonl"
    res = {
        "doc": name,
        "out_base": out_base,
        "wall_s": round(wall, 1),
        "rc": rc,
        "verdict": "NO_OUTPUT",
        "engine": None,
        "fallback": None,
        "degraded": None,
        "recovered": None,
        "total_pages": None,
        "n_text": 0,
        "n_image": 0,
        "n_table": 0,
        "n_chunks": 0,
        "leak": None,
    }
    if not jl.exists():
        return res

    rows = [json.loads(line) for line in open(jl)]
    chunks = [r for r in rows if r.get("modality")]
    hdr = next((r for r in rows if r.get("total_pages") is not None), {})
    res["engine"] = hdr.get("extraction_engine")
    res["fallback"] = hdr.get("extraction_fallback")
    res["degraded"] = hdr.get("extraction_degraded_pages")
    res["recovered"] = hdr.get("extraction_recovered_pages")
    res["total_pages"] = hdr.get("total_pages")
    res["n_text"] = sum(1 for c in chunks if c.get("modality") == "text")
    res["n_image"] = sum(1 for c in chunks if c.get("modality") == "image")
    res["n_table"] = sum(1 for c in chunks if c.get("modality") == "table")
    res["n_chunks"] = len(chunks)

    def _is_leak(m):
        if not m:
            return False
        return m not in ("uir_native_chunker", "rendered_region_crop") and not m.startswith(
            "recovery_"
        )

    res["leak"] = sum(
        1 for c in chunks if _is_leak((c.get("metadata") or {}).get("extraction_method"))
    )

    qa = subprocess.run(
        [PY, "scripts/qa_full_conversion.py", str(jl), "--source-pdf", str(src_pdf)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    for line in reversed(qa.stdout.splitlines()):
        for tok in ("QA_PASS_WITH_ADVISORIES", "QA_PASS", "QA_WARN", "QA_FAIL"):
            if line.startswith(tok):
                res["verdict"] = tok
                break
        if res["verdict"] != "NO_OUTPUT":
            break
    return res


def _is_warn_fail(v: str) -> bool:
    return v in ("QA_WARN", "QA_FAIL")


def run_smoke() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, rel, window in SMOKE:
        src = ROOT / "data" / rel
        if not src.exists():
            log(f"PHANTOM: source missing {src}")
            return 2
        if window is not None:
            slc = SMOKE_SRC / f"{name}.pdf"
            pages = slice_pdf(src, slc, window)
            src_use = slc
        else:
            pages = fitz.open(src).page_count
            src_use = src
        log(f"=== SMOKE {name}: {pages}pg")
        r = run_one(name, src_use, f"_smoke_{name}")
        # engine must be the hybrid; degraded must be exactly 0 (the silent-ladder
        # prereq check - None means header absent -> fail); verdict must be a PASS.
        passed = (
            r["engine"] == "mineru_qwen_hybrid"
            and r["degraded"] == 0
            and r["verdict"] in ("QA_PASS", "QA_PASS_WITH_ADVISORIES")
        )
        log(
            f"  {name}: {r['verdict']} engine={r['engine']} deg={r['degraded']} "
            f"leak={r['leak']} chunks={r['n_chunks']}(t{r['n_text']}/i{r['n_image']}/tb{r['n_table']}) "
            f"{r['wall_s']}s -> {'PASS' if passed else 'FAIL'}"
        )
        ok = ok and passed
    log(f"SMOKE {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def run_batch() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = OUT_DIR / "reextract_results.jsonl"
    window10: list[str] = []
    served_pages = 0
    degraded_pages = 0
    log(f"Phase 5 re-extraction: {len(BATCH)} docs (prod hybrid config)")
    for name, rel, base in BATCH:
        src = ROOT / "data" / rel
        if not src.exists():
            log(f"PHANTOM: source missing {src} - skipping {name}")
            continue
        log(f"=== {name} ({base}) ...")
        r = run_one(name, src, base)
        with open(results, "a") as rf:
            rf.write(json.dumps(r) + "\n")
        log(
            f"  {name}: {r['verdict']} engine={r['engine']} deg={r['degraded']} "
            f"leak={r['leak']} chunks={r['n_chunks']}(t{r['n_text']}/i{r['n_image']}/tb{r['n_table']}) "
            f"{r['wall_s']}s"
        )
        # rolling rollback window
        window10.append(r["verdict"])
        window10 = window10[-10:]
        if r.get("total_pages"):
            served_pages += r["total_pages"]
            degraded_pages += r.get("degraded") or 0
        wf = sum(1 for v in window10 if _is_warn_fail(v))
        ladder_rate = (degraded_pages / served_pages) if served_pages else 0.0
        if wf >= 3:
            log(f"ROLLBACK FIRED: {wf}/10 consecutive docs QA_WARN/FAIL (>20pp). STOPPING.")
            log(f"  window={window10}")
            return 3
        if ladder_rate > 0.02:
            log(
                f"ROLLBACK FIRED: ladder-served {degraded_pages}/{served_pages} = {ladder_rate:.3%} > 2%. STOPPING."
            )
            return 3
    log(f"BATCH DONE. ladder-served {degraded_pages}/{served_pages} pages.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--batch", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        return run_smoke()
    if a.batch:
        return run_batch()
    ap.error("pass --smoke or --batch")


if __name__ == "__main__":
    raise SystemExit(main())
