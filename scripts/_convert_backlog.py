"""Convert the backlog code books with Fix (b). NO Qdrant writes (ingest is a
separate, permissioned step). Writes each to output/backlog/<base>/ and logs R3 +
repair stats + gate verdict per book. Resilient: logs and continues on failure.
Throwaway driver (underscore name).

Run (background):
  MINERU_ENDPOINT=... VLM_NATIVE_ENDPOINT=... python scripts/_convert_backlog.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import _code_quality as cq  # noqa: E402

MMRAG = "/Users/Shared/miniforge3/envs/mmrag-v2/bin/mmrag-v2"
OUT = Path("output/backlog")

# backlog code books (doc_id not in production), Devlin excluded (already done)
BOOKS = [
    "data/raw/Eliasz A. Zephyr RTOS Embedded C Programming. Using Embedded RTOS POSIX API 2024.pdf",
    "data/technical_manual/Adedeji A. GenAI on Google Cloud. Enterprise Generative AI Systems...Agents 2026.pdf",
    "data/technical_manual/Bourne K. Unlocking Data with Generative AI and RAG 2024.pdf",
    "data/technical_manual/Chaubal S. AI Projects in PyTorch. Hands-On Projects in Vision, Text,...2025.pdf",
    "data/technical_manual/Fluent Python Luciano Ramalho 2015.pdf",
    "data/technical_manual/Hao B. Machine Learning Platform Engineering. Build...for ML and AI systems 2026.pdf",
    "data/technical_manual/Jungjun H. Build an AI Agent (From Scratch)...MEAP 2026.pdf",
    "data/technical_manual/Nagasubramanian D. Agentic AI for Engineers.Architecting Goal-Driven System 2026.pdf",
    "data/technical_manual/Programming ArcGIS with Python Cookbook.pdf",
    "data/technical_manual/Python Cookbook  Everyone can cook delicious recipes with Python.pdf",
    "data/technical_manual/Python Distilled David M. Beazley 2022.pdf",
    "data/technical_manual/Raieli S. Building AI Agents with LLMs, RAG, and Knowledge Graphs...2025.pdf",
    "data/technical_manual/The_Complete_C__Python_Coding_Manual_-_21th_Edition_2025.pdf",
]


def _slug(pdf: str) -> str:
    return Path(pdf).stem.replace(" ", "_").replace(".", "_")[:50]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for pdf in BOOKS:
        label = Path(pdf).name[:45]
        if not Path(pdf).exists():
            print(f"\n=== {label} ===\n    SOURCE MISSING; skip", flush=True)
            summary.append((label, "NO_SOURCE", None))
            continue
        outdir = OUT / _slug(pdf)
        if (outdir / "ingestion.jsonl").exists():
            print(f"\n=== {label} -> {outdir} ===\n    SKIP (already converted)", flush=True)
            summary.append((label, "SKIP_DONE", None))
            continue
        print(f"\n=== {label} -> {outdir} ===", flush=True)
        r = subprocess.run(
            [MMRAG, "process", pdf, "--batch-size", "10",
             "--output-dir", str(outdir), "--vision-provider", "none"],
            capture_output=True, text=True,
        )
        jl = outdir / "ingestion.jsonl"
        if not jl.exists():
            print(f"    CONVERT FAILED rc={r.returncode}; tail:\n{r.stderr[-600:]}", flush=True)
            summary.append((label, "CONVERT_FAIL", None))
            continue
        rows = [json.loads(l) for l in open(jl)]
        hdr = rows[0]
        m = cq.code_quality([{"content": d.get("content") or "",
                              "modality": (d.get("metadata") or {}).get("modality") or d.get("modality"),
                              "metadata": d.get("metadata") or {}} for d in rows])
        verdict = cq.gate_verdict(m)
        risk = hdr.get("extraction_quality_risk_pages")
        rep = hdr.get("extraction_code_repaired_pages")
        print(f"    chunks={len(rows)} R3={m.indentation_fidelity:.3f} "
              f"judge={m.n_judgeable} fail={m.n_judgeable_fail} verdict={verdict.upper()} "
              f"repaired={rep}/{risk}", flush=True)
        summary.append((label, f"{verdict.upper()} R3={m.indentation_fidelity:.2f} rep={rep}/{risk}", len(rows)))

    print("\n===== BACKLOG CONVERSION SUMMARY =====", flush=True)
    for label, status, n in summary:
        print(f"  {label:46s} {status:34s} chunks={n}", flush=True)
    ok = sum(1 for _, s, _ in summary if not s.startswith(("CONVERT_FAIL", "NO_SOURCE")))
    print(f"\nBACKLOG_CONVERT_DONE ok={ok}/{len(BOOKS)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
