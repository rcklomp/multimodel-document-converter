#!/usr/bin/env python
"""Phase I (Plan v2.9, 2026-05-11) — rebuild `mmrag_v2_8` Qdrant
collection from the v2.9.0-rc1 canonical 34-doc corpus.

The first doc is ingested with ``--recreate`` (drops the v2.8 collection
and creates a fresh one). Subsequent docs append to the same collection.

Run from project root:

    conda run -n mmrag-v2 --no-capture-output python scripts/rebuild_mmrag_v2_8_for_rc1.py

Expected wall time: 4-8 hours on local Ollama llava.
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
LOG = OUTPUT / "_logs" / "phase_i_rebuild.log"
COLLECTION = "mmrag_v2_8"

CANONICAL_34 = [
    "HarryPotter_and_the_Sorcerers_Stone",
    "Form_0013_invoice",
    "Form_betwistingsformulier",
    "CarOK_voorraadtelling",
    "AIOS_LLM_Agent_Operating_System",
    "A_comprehensive_review_on_hybrid_electri",
    "Hybrid_electric_vehicles",
    "IRJET_Modeling_of_Solar_PV",
    "Recent_Trends_in_Transportation",
    "Combat_Aircraft_August_2025",
    "PCWorld_July_2025",
    "ATZ_Elektronik_German",
    "Kimothi_RAG_Guide",
    "Integra_manual",
    "Jungjun_AI_Agent",
    "Bourne_RAG_2024",
    "Devlin_LLM_Agents",
    "Raieli_AI_Agents",
    "Adedeji_GenAI_Google_Cloud",
    "Cronin_GenAI_Models",
    "Hao_ML_Platform",
    "Nagasubramanian_Agentic_AI",
    "Sekar_MCP_Standard",
    "Python_Cookbook",
    "ArcGIS_Python_Cookbook",
    "Fluent_Python",
    "Python_Distilled",
    "Ayeva_Python_Patterns",
    "Chaubal_PyTorch_Projects",
    "Earthship_Vol1",
    "Firearms",
    "Greenhouse_Design",
    "ChatGPT_Praktijk_handboek",
    "KI_En_ChatGPT_Praktische_Gids",
]


def log(message: str) -> None:
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> int:
    log("=== PHASE I REBUILD mmrag_v2_8 v2.9.0-rc1 ===")
    total_start = time.time()
    for idx, name in enumerate(CANONICAL_34, start=1):
        jsonl = OUTPUT / name / "ingestion.jsonl"
        if not jsonl.exists():
            log(f"[{idx}/34] MISSING {name} — aborting")
            return 1
        log(f"[{idx}/34] START {name}")
        cmd = [
            sys.executable,
            "scripts/ingest_to_qdrant.py",
            str(jsonl),
            "--collection",
            COLLECTION,
        ]
        if idx == 1:
            cmd.append("--recreate")
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=ROOT)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            log(f"[{idx}/34] FAIL {name} (rc={proc.returncode}, {elapsed:.0f}s)")
            return proc.returncode
        log(f"[{idx}/34] DONE {name} ({elapsed:.0f}s)")
    total = time.time() - total_start
    log(f"=== ALL DONE — total wall time {total/60:.1f} min ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
