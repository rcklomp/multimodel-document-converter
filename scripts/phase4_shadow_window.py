#!/usr/bin/env python3
"""PLAN_EXTRACTION_FIDELITY_V1 Phase 4 / WP-A - shadow window harness.

Runs the 16-doc crucible through BOTH production configs on an IDENTICAL,
pre-sliced 15-page window per doc (the crucible methodology - a sliced source
PDF keeps total_pages aligned so qa_full_conversion --source-pdf does not raise
phantom MISSING_PAGES on a page-subset). Records, per doc per arm:

  - QA verdict (qa_full_conversion.py --source-pdf)
  - Section 5.4 provenance aggregates (engine, fallback, degraded, recovered)
  - chunk counts by modality + V3-path leak count
  - wall time

Arm A: the interim default - USE_DOCLING_FAST=1 (offline floor).
Arm B: the hybrid production config - MINERU_ENDPOINT (GX10) + VLM_NATIVE_* (M5),
no USE_* force flag, so the route selects mineru_qwen_hybrid; cap1600 default.

No server reconfiguration. Outputs are gitignored (output/, logs/). This script
is harness-only and safe to commit.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "output" / "phase4_shadow_src"
OUT_DIR = ROOT / "output" / "phase4_shadow"
RESULTS = OUT_DIR / "shadow_results.jsonl"
PY = sys.executable

# (name, source pdf relative to data, 1-indexed inclusive page window or None=first15)
DOCS = [
    ("CombatAircraft", "digital_magazine/Combat Aircraft - August 2025 UK.pdf", (20, 34)),
    ("PCWorld", "digital_magazine/PCWorld_July_2025_USA.pdf", (20, 34)),
    ("AIOS_academic", "academic_journal/AIOS LLM Agent Operating System.pdf", None),
    ("FluentPython", "technical_manual/Fluent Python Luciano Ramalho 2015.pdf", (60, 74)),
    ("Grundlagen", "raw/Grundlagen Fahrzeug- und Motorentechnik.pdf", None),
    ("Form_0013", "business_form/0013_140302111325_001.pdf", None),
    ("Form_betwisting", "business_form/betwistingsformulier_aankoop_niet_ontvangen.pdf", None),
    ("CarOK_spreadsheet", "data_spreadsheet/CarOK voorraadtelling 2021-04.pdf", None),
    ("Firearms", "technical_manual/Firearms.pdf", None),
    (
        "DigitaleFotografie",
        "raw/Digitale-Fotografie - Das essentielle Handbuch Februar 2026.pdf",
        None,
    ),
    ("HarryPotter", "digital_literature/HarryPotter_and_the_Sorcerers_Stone.pdf", None),
    (
        "Kimothi_RAG",
        "technical_manual/A Simple Guide to Retrieval Augmented Generation Kimothi A. 2025.pdf",
        None,
    ),
    (
        "ATZ_Elektronik",
        "technical_report/ATZ.Elektronik.-.Effizientere.Software.Entwicklung.GERMAN.RETAiL.eBOOk-PDFWriters.pdf",
        None,
    ),
    ("IRJET_academic", "academic_journal/IRJET_Modeling_of_Solar_PV_system_under.pdf", None),
    ("Hybrid_EV", "academic_journal/Hybrid_electric_vehicles_and_their_challenges.pdf", None),
    ("Bevestigingsmiddelen", "raw/Bevestigingsmiddelen.pdf", None),
]

PAGE_CAP = 15

ARM_A_ENV = {"USE_DOCLING_FAST": "1"}
ARM_B_ENV = {
    "MINERU_ENDPOINT": "http://10.0.10.239:8001",
    "MINERU_MODEL": "MinerU2.5-2509-1.2B",
    "VLM_NATIVE_ENDPOINT": "http://10.0.10.235:8000/v1",
    "VLM_NATIVE_MODEL": "mlx-community/Qwen3-VL-8B-Instruct-8bit",
    "VLM_NATIVE_API_KEY": "EMPTY",
}
# env keys to scrub between arms so neither leaks into the other
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


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def slice_pdf(src: Path, dst: Path, window) -> int:
    """Write a <=15-page slice. window=(start,end) 1-indexed inclusive, or None=first 15."""
    doc = fitz.open(src)
    n = doc.page_count
    if window is None:
        first, last = 1, min(PAGE_CAP, n)
    else:
        first, last = window
        last = min(last, n)
        if last - first + 1 > PAGE_CAP:
            last = first + PAGE_CAP - 1
    out = fitz.open()
    out.insert_pdf(doc, from_page=first - 1, to_page=last - 1)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst)
    pages = last - first + 1
    out.close()
    doc.close()
    return pages


def run_arm(name: str, slice_pdf_path: Path, arm: str, arm_env: dict) -> dict:
    out = OUT_DIR / f"{name}__{arm}"
    env = {k: v for k, v in os.environ.items() if k not in ROUTE_KEYS}
    env.update(arm_env)
    cmd = [
        PY,
        "-m",
        "mmrag_v2.cli",
        "process",
        str(slice_pdf_path),
        "--output-dir",
        str(out),
        "--batch-size",
        "10",
        "--vision-provider",
        "none",
    ]
    logf = OUT_DIR / f"{name}__{arm}.log"
    t0 = time.time()
    with open(logf, "w") as lf:
        rc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0

    jl = out / "ingestion.jsonl"
    res = {
        "doc": name,
        "arm": arm,
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
        return (
            m != "uir_native_chunker"
            and m != "rendered_region_crop"
            and not m.startswith("recovery_")
        )

    res["leak"] = sum(
        1 for c in chunks if _is_leak((c.get("metadata") or {}).get("extraction_method"))
    )

    qa = subprocess.run(
        [PY, "scripts/qa_full_conversion.py", str(jl), "--source-pdf", str(slice_pdf_path)],
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


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    if RESULTS.exists():
        RESULTS.unlink()

    log(f"Phase 4 shadow window: {len(DOCS)} docs x 2 arms")
    for name, rel, window in DOCS:
        src = ROOT / "data" / rel
        if not src.exists():
            log(f"PHANTOM: source missing {src} - skipping {name}")
            with open(RESULTS, "a") as rf:
                rf.write(
                    json.dumps({"doc": name, "arm": "-", "verdict": "MISSING_SRC", "src": str(src)})
                    + "\n"
                )
            continue
        slc = SRC_DIR / f"{name}.pdf"
        pages = slice_pdf(src, slc, window)
        log(f"=== {name}: sliced {pages}pg -> {slc.name}")
        for arm, arm_env in (("A_docling", ARM_A_ENV), ("B_hybrid", ARM_B_ENV)):
            log(f"  arm {arm} ...")
            r = run_arm(name, slc, arm, arm_env)
            r["sliced_pages"] = pages
            with open(RESULTS, "a") as rf:
                rf.write(json.dumps(r) + "\n")
            log(
                f"  arm {arm}: {r['verdict']} engine={r['engine']} "
                f"deg={r['degraded']} leak={r['leak']} "
                f"chunks={r['n_chunks']}(t{r['n_text']}/i{r['n_image']}/tb{r['n_table']}) {r['wall_s']}s"
            )
    log("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
