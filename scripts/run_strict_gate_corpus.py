#!/usr/bin/env python
"""Run `scripts/qa_full_conversion.py` across the 34 canonical
v2.9.0-rc1 corpus docs and emit a per-doc PASS / WARN / FAIL
summary.

Promoted from the ad-hoc `/tmp/run_strict_gate.py` wrapper used
during the 2026-05-11 v2.9 cycle. Reads
`output/<doc>/ingestion.jsonl` for each doc and calls the strict
gate with `--source-pdf` for PDFs and `--allow-warnings` so the
gate's exit code does not abort the loop.

Usage::

    conda run -n mmrag-v2 python scripts/run_strict_gate_corpus.py

The `QA_PASS_WITH_ADVISORIES` allowed PASS variant (per
`docs/QUALITY_GATES.md` "Advisory Warning Classes") counts as PASS
in the aggregate summary. Per-doc detail is printed inline so
specific failure codes are easy to inspect.
"""
import subprocess
import sys
import time
from pathlib import Path

TARGETS = [
    ("HarryPotter_and_the_Sorcerers_Stone", "data/digital_literature/HarryPotter_and_the_Sorcerers_Stone.pdf"),
    ("Form_0013_invoice", "data/business_form/0013_140302111325_001.pdf"),
    ("Form_betwistingsformulier", "data/business_form/betwistingsformulier_aankoop_niet_ontvangen.pdf"),
    ("CarOK_voorraadtelling", "data/data_spreadsheet/CarOK voorraadtelling 2021-04.pdf"),
    ("AIOS_LLM_Agent_Operating_System", "data/academic_journal/AIOS LLM Agent Operating System.pdf"),
    ("A_comprehensive_review_on_hybrid_electri", "data/academic_journal/A_comprehensive_review_on_hybrid_electri.pdf"),
    ("Hybrid_electric_vehicles", "data/academic_journal/Hybrid_electric_vehicles_and_their_challenges.pdf"),
    ("IRJET_Modeling_of_Solar_PV", "data/academic_journal/IRJET_Modeling_of_Solar_PV_system_under.pdf"),
    ("Recent_Trends_in_Transportation", "data/academic_journal/Recent_Trends_in_Transportation_Technolo.pdf"),
    ("Combat_Aircraft_August_2025", "data/digital_magazine/Combat Aircraft - August 2025 UK.pdf"),
    ("PCWorld_July_2025", "data/digital_magazine/PCWorld_July_2025_USA.pdf"),
    ("ATZ_Elektronik_German", "data/technical_report/ATZ.Elektronik.-.Effizientere.Software.Entwicklung.GERMAN.RETAiL.eBOOk-PDFWriters.pdf"),
    ("Kimothi_RAG_Guide", "data/technical_manual/A Simple Guide to Retrieval Augmented Generation Kimothi A. 2025.pdf"),
    ("Integra_manual", "data/technical_manual/integra_u_en.pdf"),
    ("Jungjun_AI_Agent", "data/technical_manual/Jungjun H. Build an AI Agent (From Scratch)...MEAP 2026.pdf"),
    ("Bourne_RAG_2024", "data/technical_manual/Bourne K. Unlocking Data with Generative AI and RAG 2024.pdf"),
    ("Devlin_LLM_Agents", "data/technical_manual/Devlin M. Building LLM Agents with RAG, Knowledge Graphs and Reflection...2025.pdf"),
    ("Raieli_AI_Agents", "data/technical_manual/Raieli S. Building AI Agents with LLMs, RAG, and Knowledge Graphs...2025.pdf"),
    ("Adedeji_GenAI_Google_Cloud", "data/technical_manual/Adedeji A. GenAI on Google Cloud. Enterprise Generative AI Systems...Agents 2026.pdf"),
    ("Cronin_GenAI_Models", "data/technical_manual/Cronin I. Building and Training Generative AI Models. A Practical Guide...2026.pdf"),
    ("Hao_ML_Platform", "data/technical_manual/Hao B. Machine Learning Platform Engineering. Build...for ML and AI systems 2026.pdf"),
    ("Nagasubramanian_Agentic_AI", "data/technical_manual/Nagasubramanian D. Agentic AI for Engineers.Architecting Goal-Driven System 2026.pdf"),
    ("Sekar_MCP_Standard", "data/technical_manual/Sekar S. The MCP Standard. A Developer's Guide..Building Universal AI Tools 2026.pdf"),
    ("Python_Cookbook", "data/technical_manual/Python Cookbook  Everyone can cook delicious recipes with Python.pdf"),
    ("ArcGIS_Python_Cookbook", "data/technical_manual/Programming ArcGIS with Python Cookbook.pdf"),
    ("Fluent_Python", "data/technical_manual/Fluent Python Luciano Ramalho 2015.pdf"),
    ("Python_Distilled", "data/technical_manual/Python Distilled David M. Beazley 2022.pdf"),
    ("Ayeva_Python_Patterns", "data/technical_manual/Ayeva K. Mastering Python Design Patterns...essential Python patterns...3ed 2024.pdf"),
    ("Chaubal_PyTorch_Projects", "data/technical_manual/Chaubal S. AI Projects in PyTorch. Hands-On Projects in Vision, Text,...2025.pdf"),
    ("Earthship_Vol1", "data/technical_manual/Earthship_Vol1_How to build your own.pdf"),
    ("Firearms", "data/technical_manual/Firearms.pdf"),
    ("Greenhouse_Design", "data/technical_manual/Greenhouse Design and Control by Pedro Ponce.pdf"),
    ("ChatGPT_Praktijk_handboek", None),  # EPUB
    ("KI_En_ChatGPT_Praktische_Gids", None),  # EPUB
]

results = []
for name, pdf in TARGETS:
    jsonl = Path(f"output/{name}/ingestion.jsonl")
    if not jsonl.exists():
        results.append((name, "MISSING_JSONL", ""))
        continue
    cmd = ["python", "scripts/qa_full_conversion.py", str(jsonl)]
    if pdf:
        cmd.extend(["--source-pdf", pdf])
    cmd.append("--allow-warnings")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        results.append((name, "TIMEOUT", ""))
        continue
    elapsed = time.time() - t0
    # The script prints the final status line; pick it up from stdout
    out = (proc.stdout + proc.stderr).strip()
    last_status = ""
    for line in out.splitlines():
        if any(k in line for k in ("QA_PASS", "QA_WARN", "QA_FAIL")):
            last_status = line.strip()
    failure_codes = []
    for line in out.splitlines():
        if "FAIL" in line and "QA_" not in line:
            failure_codes.append(line.strip()[:80])
    summary = last_status[:60] or f"rc={proc.returncode}"
    print(f"{name:42}  rc={proc.returncode}  {elapsed:5.1f}s  {summary}")
    if failure_codes[:3]:
        for fc in failure_codes[:3]:
            print(f"  -> {fc}")
    results.append((name, summary, "\n".join(failure_codes[:5])))

print("\n=== SUMMARY ===")
# Phase G allowed PASS variant: QA_PASS_WITH_ADVISORIES counts as PASS.
pwa = sum(1 for _, s, _ in results if "QA_PASS_WITH_ADVISORIES" in s)
ppure = sum(1 for _, s, _ in results if s.startswith("QA_PASS:"))
pass_count = pwa + ppure
warn_count = sum(1 for _, s, _ in results if s.startswith("QA_WARN:"))
fail_count = sum(1 for _, s, _ in results if "QA_FAIL" in s or "TIMEOUT" in s or "MISSING" in s)
print(f"PASS={pass_count}  WARN={warn_count}  FAIL={fail_count}  total={len(results)}")
print(f"  (PASS breakdown: QA_PASS={ppure}, QA_PASS_WITH_ADVISORIES={pwa})")
