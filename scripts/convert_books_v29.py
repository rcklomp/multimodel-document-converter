#!/usr/bin/env python
"""Phase 5 v2.9 corpus conversion runner.

This is the preferred Phase 5 runner on Apple Silicon. It must be invoked
from the target Python environment, for example:

    conda run -n mmrag-v2 python scripts/convert_books_v29.py

Do not wrap it in ``bash -lc`` / ``zsh -lc``. In this environment nested
shells make torch report ``mps=False`` and Docling falls back to a very slow
CPU path. This runner uses ``sys.executable`` for child conversions so the
MPS-visible Python runtime is preserved.
"""

from __future__ import annotations

import argparse
import os
import signal
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
LOG = OUTPUT / "_convert_books.log"


@dataclass(frozen=True)
class ConversionTarget:
    src: str
    name: str
    batch_size: int = 10


TARGETS: tuple[ConversionTarget, ...] = (
    ConversionTarget(
        "data/digital_literature/HarryPotter_and_the_Sorcerers_Stone.pdf",
        "HarryPotter_and_the_Sorcerers_Stone",
    ),
    ConversionTarget("data/business_form/0013_140302111325_001.pdf", "Form_0013_invoice"),
    ConversionTarget(
        "data/business_form/betwistingsformulier_aankoop_niet_ontvangen.pdf",
        "Form_betwistingsformulier",
    ),
    ConversionTarget("data/data_spreadsheet/CarOK voorraadtelling 2021-04.pdf", "CarOK_voorraadtelling"),
    ConversionTarget("data/academic_journal/AIOS LLM Agent Operating System.pdf", "AIOS_LLM_Agent_Operating_System"),
    ConversionTarget(
        "data/academic_journal/A_comprehensive_review_on_hybrid_electri.pdf",
        "A_comprehensive_review_on_hybrid_electri",
    ),
    ConversionTarget("data/academic_journal/Hybrid_electric_vehicles_and_their_challenges.pdf", "Hybrid_electric_vehicles"),
    ConversionTarget("data/academic_journal/IRJET_Modeling_of_Solar_PV_system_under.pdf", "IRJET_Modeling_of_Solar_PV"),
    ConversionTarget("data/academic_journal/Recent_Trends_in_Transportation_Technolo.pdf", "Recent_Trends_in_Transportation"),
    ConversionTarget("data/digital_magazine/Combat Aircraft - August 2025 UK.pdf", "Combat_Aircraft_August_2025"),
    ConversionTarget("data/digital_magazine/PCWorld_July_2025_USA.pdf", "PCWorld_July_2025"),
    ConversionTarget(
        "data/technical_report/ATZ.Elektronik.-.Effizientere.Software.Entwicklung.GERMAN.RETAiL.eBOOk-PDFWriters.pdf",
        "ATZ_Elektronik_German",
    ),
    ConversionTarget(
        "data/technical_manual/A Simple Guide to Retrieval Augmented Generation Kimothi A. 2025.pdf",
        "Kimothi_RAG_Guide",
    ),
    ConversionTarget("data/technical_manual/integra_u_en.pdf", "Integra_manual"),
    ConversionTarget("data/technical_manual/Jungjun H. Build an AI Agent (From Scratch)...MEAP 2026.pdf", "Jungjun_AI_Agent"),
    ConversionTarget("data/technical_manual/Bourne K. Unlocking Data with Generative AI and RAG 2024.pdf", "Bourne_RAG_2024"),
    ConversionTarget(
        "data/technical_manual/Devlin M. Building LLM Agents with RAG, Knowledge Graphs and Reflection...2025.pdf",
        "Devlin_LLM_Agents",
    ),
    ConversionTarget(
        "data/technical_manual/Raieli S. Building AI Agents with LLMs, RAG, and Knowledge Graphs...2025.pdf",
        "Raieli_AI_Agents",
    ),
    ConversionTarget(
        "data/technical_manual/Adedeji A. GenAI on Google Cloud. Enterprise Generative AI Systems...Agents 2026.pdf",
        "Adedeji_GenAI_Google_Cloud",
    ),
    ConversionTarget(
        "data/technical_manual/Cronin I. Building and Training Generative AI Models. A Practical Guide...2026.pdf",
        "Cronin_GenAI_Models",
    ),
    ConversionTarget(
        "data/technical_manual/Hao B. Machine Learning Platform Engineering. Build...for ML and AI systems 2026.pdf",
        "Hao_ML_Platform",
    ),
    ConversionTarget(
        "data/technical_manual/Nagasubramanian D. Agentic AI for Engineers.Architecting Goal-Driven System 2026.pdf",
        "Nagasubramanian_Agentic_AI",
    ),
    ConversionTarget(
        "data/technical_manual/Sekar S. The MCP Standard. A Developer's Guide..Building Universal AI Tools 2026.pdf",
        "Sekar_MCP_Standard",
    ),
    ConversionTarget("data/technical_manual/Python Cookbook  Everyone can cook delicious recipes with Python.pdf", "Python_Cookbook"),
    ConversionTarget("data/technical_manual/Programming ArcGIS with Python Cookbook.pdf", "ArcGIS_Python_Cookbook"),
    ConversionTarget("data/technical_manual/Fluent Python Luciano Ramalho 2015.pdf", "Fluent_Python"),
    ConversionTarget("data/technical_manual/Python Distilled David M. Beazley 2022.pdf", "Python_Distilled"),
    ConversionTarget(
        "data/technical_manual/Ayeva K. Mastering Python Design Patterns...essential Python patterns...3ed 2024.pdf",
        "Ayeva_Python_Patterns",
    ),
    ConversionTarget(
        "data/technical_manual/Chaubal S. AI Projects in PyTorch. Hands-On Projects in Vision, Text,...2025.pdf",
        "Chaubal_PyTorch_Projects",
    ),
    ConversionTarget("data/technical_manual/Earthship_Vol1_How to build your own.pdf", "Earthship_Vol1"),
    ConversionTarget("data/technical_manual/Firearms.pdf", "Firearms"),
    ConversionTarget("data/technical_manual/Greenhouse Design and Control by Pedro Ponce.pdf", "Greenhouse_Design"),
    ConversionTarget(
        "data/technical_manual/Falkner, Leonie - ChatGPT Praktijk-handboek.epub",
        "ChatGPT_Praktijk_handboek",
    ),
    ConversionTarget(
        "data/technical_manual/Seffer, David - KI En ChatGPT, Praktische Gids Voor Online Business Met Digitale Producten.epub",
        "KI_En_ChatGPT_Praktische_Gids",
    ),
)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    line = f"{now()} {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def mps_available() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def parse_only(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def convert(target: ConversionTarget, *, pages: str | None, timeout: int | None, force: bool) -> int:
    src = ROOT / target.src
    outdir = OUTPUT / target.name
    lock_path = OUTPUT / f".{target.name}.convert.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        log(f"FAIL {target.name} (conversion lock exists: {lock_path})")
        return 125

    os.write(lock_fd, str(os.getpid()).encode("ascii"))
    os.close(lock_fd)
    try:
        return _convert_locked(target, src=src, outdir=outdir, pages=pages, timeout=timeout, force=force)
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _convert_locked(
    target: ConversionTarget,
    *,
    src: Path,
    outdir: Path,
    pages: str | None,
    timeout: int | None,
    force: bool,
) -> int:
    if (outdir / "ingestion.jsonl").exists() and not force:
        lines = sum(1 for _ in (outdir / "ingestion.jsonl").open("r", encoding="utf-8"))
        log(f"SKIP {target.name} (existing {lines} lines)")
        return 0

    if outdir.exists():
        shutil.rmtree(outdir)

    log(f"START {target.name}")
    cmd = [
        sys.executable,
        "-m",
        "mmrag_v2.cli",
        "process",
        str(src),
        "-o",
        str(outdir),
        "-b",
        str(target.batch_size),
        "--vision-provider",
        "none",
        "--no-cache",
    ]
    if pages:
        cmd.extend(["--pages", pages])

    proc = subprocess.Popen(cmd, cwd=ROOT, start_new_session=True)
    try:
        returncode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        log(f"TIMEOUT {target.name} (>{timeout}s)")
        return 124

    if returncode == 0 and (outdir / "ingestion.jsonl").exists():
        lines = sum(1 for _ in (outdir / "ingestion.jsonl").open("r", encoding="utf-8"))
        log(f"DONE {target.name} ({lines} lines)")
        return 0

    log(f"FAIL {target.name} (exit={returncode})")
    return returncode or 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v2.9 Phase 5 broad corpus conversion")
    parser.add_argument("--only", help="Comma-separated output names to convert")
    parser.add_argument("--pages", help="Forwarded page limit for probes only")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed document")
    parser.add_argument("--append-log", action="store_true", help="Append to the existing run log")
    parser.add_argument("--force", action="store_true", help="Reconvert outputs that already have ingestion.jsonl")
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Per-document timeout in seconds; timed-out documents fail with exit 124",
    )
    args = parser.parse_args()

    OUTPUT.mkdir(exist_ok=True)
    if not args.append_log:
        LOG.write_text("", encoding="utf-8")
    log("=== BROAD RECONVERSION v2.9.0-rc1 ===")
    log(f"MPS_AVAILABLE={mps_available()}")
    if not mps_available():
        log("ABORT MPS is unavailable in this Python process")
        return 2

    selected = parse_only(args.only)
    passed = 0
    failed = 0
    for target in TARGETS:
        if selected is not None and target.name not in selected:
            log(f"SKIP {target.name} (--only)")
            continue
        rc = convert(target, pages=args.pages, timeout=args.timeout, force=args.force)
        if rc == 0:
            passed += 1
        else:
            failed += 1
            if not args.keep_going:
                log(f"ABORT fail-fast after {target.name}")
                return rc

    log(f"=== ALL DONE ({passed} pass, {failed} fail) ===")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
