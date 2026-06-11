#!/usr/bin/env python3
"""Phase 5 full-corpus reconciliation - extract + ingest-as-you-go, fail-safe.

Re-extracts the REMAINING data/ corpus (everything not in the 12-doc bounded
subset) through the production hybrid (via the localhost relay when the conda env
cannot reach the LAN servers - set PHASE5_MINERU_ENDPOINT / PHASE5_VLM_ENDPOINT),
and dense-ingests each CLEAN doc into mmrag_v3__qwen3_local immediately so progress
is durable across interruptions. Sparse twin is rebuilt at checkpoints.

FAIL-SAFE: a doc is ingested ONLY if engine=mineru_qwen_hybrid AND degraded==0 AND
QA verdict is a PASS. A degraded doc (relay/network drop -> laddered to Docling) is
NOT ingested (it is Docling-quality, not hybrid); it is recorded and RETRIED once at
the end. So a network blip costs a retry, never a stale/corrupt index entry.

Order: smallest-first (quick durable wins first; the big code books last). The full
remaining corpus is ~32 docs / ~11.5k pages = many hours; this run makes incremental
progress and is safe to stop/resume (re-running skips docs already present + clean).

Run (with the relay up):
  PHASE5_MINERU_ENDPOINT=http://127.0.0.1:18001 PHASE5_VLM_ENDPOINT=http://127.0.0.1:18000/v1 \
  python scripts/phase5_full_corpus.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
# run_one reads PROD_ENV (incl. PHASE5_* relay overrides) at import - env must be set first.
from phase5_reextract import run_one, OUT_DIR  # noqa: E402

PY = sys.executable
QDRANT = "http://127.0.0.1:6333"
DENSE_COL = "mmrag_v3__qwen3_local"
RESULTS = OUT_DIR / "full_corpus_results.jsonl"

# The 12 bounded-subset source PDFs (already done) - excluded by source path.
DONE_SRC = {
    "business_form/0013_140302111325_001.pdf",
    "business_form/betwistingsformulier_aankoop_niet_ontvangen.pdf",
    "raw/Bevestigingsmiddelen.pdf",
    "technical_report/ATZ.Elektronik.-.Effizientere.Software.Entwicklung.GERMAN.RETAiL.eBOOk-PDFWriters.pdf",
    "academic_journal/IRJET_Modeling_of_Solar_PV_system_under.pdf",
    "data_spreadsheet/CarOK voorraadtelling 2021-04.pdf",
    "academic_journal/Hybrid_electric_vehicles_and_their_challenges.pdf",
    "academic_journal/AIOS LLM Agent Operating System.pdf",
    "raw/Digitale-Fotografie - Das essentielle Handbuch Februar 2026.pdf",
    "technical_manual/Firearms.pdf",
    "digital_magazine/Combat Aircraft - August 2025 UK.pdf",
    "digital_magazine/PCWorld_July_2025_USA.pdf",
}
DONE_BASES = {
    "Form_0013",
    "Form_betwisting",
    "Bevestigingsmiddelen",
    "ATZ_Elektronik",
    "IRJET_academic",
    "CarOK_spreadsheet",
    "Hybrid_EV",
    "AIOS_academic",
    "DigitaleFotografie",
    "Firearms",
    "CombatAircraft",
    "PCWorld",
}
PASS_VERDICTS = {"QA_PASS", "QA_PASS_WITH_ADVISORIES"}
SPARSE_CHECKPOINT_EVERY = 5


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def base_for(stem: str, taken: set) -> str:
    b = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")[:48] or "doc"
    cand, n = b, 1
    while cand in taken or cand in DONE_BASES:
        n += 1
        cand = f"{b}_{n}"
    return cand


def remaining_docs() -> list:
    pdfs = sorted(ROOT.glob("data/**/*.pdf")) + sorted(ROOT.glob("data/**/*.PDF"))
    out, taken = [], set()
    for p in pdfs:
        rel = str(p.relative_to(ROOT / "data"))
        if rel in DONE_SRC:
            continue
        try:
            n = fitz.open(p).page_count
        except Exception:
            n = 1 << 30
        base = base_for(p.stem, taken)
        taken.add(base)
        out.append((n, p, base))
    out.sort(key=lambda x: x[0])
    return out


def already_clean(base: str) -> bool:
    """Resume support: a doc whose output exists with degraded==0 is skipped."""
    jl = OUT_DIR / base / "ingestion.jsonl"
    if not jl.exists():
        return False
    try:
        hdr = next((json.loads(line) for line in open(jl) if '"total_pages"' in line), {})
        return (
            hdr.get("extraction_degraded_pages") == 0
            and hdr.get("extraction_engine") == "mineru_qwen_hybrid"
        )
    except Exception:
        return False


def ingest_dense(base: str) -> bool:
    jl = OUT_DIR / base / "ingestion.jsonl"
    r = subprocess.run(
        [
            PY,
            "scripts/ingest_to_qdrant.py",
            str(jl),
            "--collection",
            DENSE_COL,
            "--qdrant-url",
            QDRANT,
            "--provider",
            "omlx",
            "--model",
            "Qwen3-Embedding-8B-mxfp8",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        log(f"  DENSE INGEST FAIL rc={r.returncode}: {r.stderr[-300:]}")
    return r.returncode == 0


def rebuild_sparse() -> None:
    r = subprocess.run(
        [PY, "scripts/phase5_ingest_bm25.py", "--qdrant-url", QDRANT],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "?"
    log(f"  sparse rebuild: {tail}")


def process(name, src, base, retry=False) -> dict:
    r = run_one(name, src, base)
    clean = (
        r["engine"] == "mineru_qwen_hybrid" and r["degraded"] == 0 and r["verdict"] in PASS_VERDICTS
    )
    r["clean"] = clean
    r["ingested"] = False
    if clean:
        r["ingested"] = ingest_dense(base)
    tag = "retry " if retry else ""
    log(
        f"  {tag}{name}: {r['verdict']} deg={r['degraded']} chunks={r['n_chunks']} "
        f"{r['wall_s']}s -> {'INGESTED' if r['ingested'] else ('SKIP(deg>0)' if not clean else 'INGEST_FAIL')}"
    )
    with open(RESULTS, "a") as f:
        f.write(json.dumps(r) + "\n")
    return r


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    docs = remaining_docs()
    log(
        f"Full-corpus reconciliation: {len(docs)} remaining docs, "
        f"{sum(n for n, _, _ in docs)} pages (smallest-first)"
    )
    window, skipped = [], []
    done = 0
    for n, src, base in docs:
        if already_clean(base):
            log(f"=== {base} ({n}pg): already clean - ensuring ingested")
            ingest_dense(base)
            done += 1
            continue
        log(f"=== {base} ({n}pg) ...")
        r = process(base, src, base)
        window.append(r["verdict"])
        window = window[-10:]
        if not r["clean"]:
            skipped.append((base, src))
        else:
            done += 1
            if done % SPARSE_CHECKPOINT_EVERY == 0:
                rebuild_sparse()
        wf = sum(1 for v in window if v in ("QA_WARN", "QA_FAIL"))
        if wf >= 3:
            log(f"ROLLBACK FIRED: {wf}/10 docs QA_WARN/FAIL. STOPPING. window={window}")
            break

    # one retry pass for degraded (relay/network blip) docs
    if skipped:
        log(f"--- retry pass: {len(skipped)} degraded docs ---")
        for base, src in skipped:
            r = process(base, src, base, retry=True)
            if r["clean"]:
                done += 1

    rebuild_sparse()
    log(
        f"DONE. clean+ingested={done}, still-degraded={sum(1 for b,_ in skipped if not already_clean(b))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
