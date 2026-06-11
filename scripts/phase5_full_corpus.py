#!/usr/bin/env python3
"""Phase 5 full-corpus reconciliation - extract + ingest-as-you-go, fail-safe.

Re-extracts the REMAINING data/ corpus (everything not in the 12-doc bounded
subset) through the production hybrid (via the localhost relay when the conda env
cannot reach the LAN servers - set PHASE5_MINERU_ENDPOINT / PHASE5_VLM_ENDPOINT),
and dense-ingests each CLEAN doc into mmrag_v3__qwen3_local immediately so progress
is durable across interruptions. Sparse twin is rebuilt at checkpoints.

Per-doc outcome (classify()):
  INGESTABLE   - hybrid, QA PASS, laddered fraction <= 2% (Phase-4 bound) -> ingest.
  CONTENT_FAIL - QA_WARN/FAIL on a cleanly-extracted doc (R3 code-indentation,
                 LABEL, ...). NOT ingested; recorded as a needs-fix follow-up; not
                 retried (deterministic - it needs an extraction/gate fix, not a rerun).
  LADDER_FAIL  - laddered > 2% (relay/MinerU page fault). NOT ingested; retried once.

HALT only on a CONNECTIVITY COLLAPSE (>=3 consecutive ~fully-laddered docs = relay or
network down) - never on content-gate failures (a diverse corpus legitimately has some).

Order: smallest-first. Resumable from the ledger (full_corpus_results.jsonl): a doc
already ingested is skipped; a doc whose existing output now qualifies is ingested
without re-extracting; a CONTENT_FAIL is recorded without re-extracting. The remaining
corpus is ~32 docs / ~11.5k pages = many hours; progress is durable across stops.

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
# Ingest a doc if it PASSES QA and its laddered (Docling-served) page fraction is
# within the Phase-4 production bound (2%). A doc that ladders MORE than this, or
# fails a content gate (R3 code-indentation, LABEL, ...), is NOT ingested.
LADDER_TOL = 0.02
# Connectivity-collapse HALT: only when consecutive freshly-extracted docs ladder
# almost entirely (relay/network down) - NOT on per-doc content-gate failures.
COLLAPSE_RATIO = 0.9
COLLAPSE_HALT = 3


def ladder_ratio(r: dict) -> float:
    return (r.get("degraded") or 0) / (r.get("total_pages") or 1)


def classify(r: dict) -> str:
    """INGESTABLE | CONTENT_FAIL (content gate) | LADDER_FAIL (connectivity)."""
    if r.get("engine") != "mineru_qwen_hybrid":
        return "LADDER_FAIL"
    if ladder_ratio(r) > LADDER_TOL:
        return "LADDER_FAIL"
    if r.get("verdict") not in PASS_VERDICTS:
        return "CONTENT_FAIL"
    return "INGESTABLE"


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


def load_ledger() -> dict:
    """Last result row per out_base (resume support)."""
    led = {}
    if RESULTS.exists():
        for line in open(RESULTS):
            try:
                r = json.loads(line)
                led[r["out_base"]] = r
            except Exception:
                pass
    return led


def process(name, src, base, retry=False) -> dict:
    r = run_one(name, src, base)
    cls = classify(r)
    r["outcome_class"] = cls
    r["ingested"] = ingest_dense(base) if cls == "INGESTABLE" else False
    tag = "retry " if retry else ""
    out = "INGESTED" if r["ingested"] else (cls if cls != "INGESTABLE" else "INGEST_FAIL")
    log(
        f"  {tag}{name}: {r['verdict']} deg={r['degraded']}/{r['total_pages']} "
        f"chunks={r['n_chunks']} {r['wall_s']}s -> {out}"
    )
    with open(RESULTS, "a") as f:
        f.write(json.dumps(r) + "\n")
    return r


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    docs = remaining_docs()
    ledger = load_ledger()
    log(
        f"Full-corpus reconciliation: {len(docs)} remaining docs, "
        f"{sum(n for n, _, _ in docs)} pages (smallest-first); ledger has {len(ledger)}"
    )

    ingested, content_fail, ladder_retry = 0, [], []
    work = []  # docs needing extraction
    # Resume: reuse the ledger so we never re-extract a doc whose output already qualifies.
    for n, src, base in docs:
        prev = ledger.get(base)
        if prev and prev.get("ingested"):
            continue  # already in Qdrant
        if prev:
            cls = classify(prev)
            if cls == "INGESTABLE":  # existing output qualifies under the 2% policy
                if ingest_dense(base):
                    ingested += 1
                    log(
                        f"=== {base}: existing output qualifies (deg={prev['degraded']}) -> INGESTED"
                    )
                    continue
            elif cls == "CONTENT_FAIL":
                content_fail.append((base, prev.get("verdict")))
                continue  # deterministic content-gate fail; needs a fix, not a retry
        work.append((n, src, base))

    log(
        f"resume: {ingested} re-ingested from ledger, {len(content_fail)} content-fail, "
        f"{len(work)} to extract"
    )

    collapse = 0
    for i, (n, src, base) in enumerate(work):
        log(f"=== [{i+1}/{len(work)}] {base} ({n}pg) ...")
        r = process(base, src, base)
        cls = r["outcome_class"]
        if cls == "INGESTABLE" and r["ingested"]:
            ingested += 1
            if ingested % SPARSE_CHECKPOINT_EVERY == 0:
                rebuild_sparse()
        elif cls == "CONTENT_FAIL":
            content_fail.append((base, r["verdict"]))
        else:  # LADDER_FAIL
            ladder_retry.append((base, src))
        # Engine-collapse halt: consecutive TOTAL failures = the engine path is broken
        # (relay/network down, OR the GX10 MinerU wedged -> watchdog-killed NO_OUTPUT,
        # which has no page count so it must be counted explicitly here, not via ratio).
        dead = r.get("total_pages") is None or ladder_ratio(r) > COLLAPSE_RATIO
        collapse = collapse + 1 if dead else 0
        if collapse >= COLLAPSE_HALT:
            log(
                f"ENGINE COLLAPSE: {collapse} consecutive total-failure docs (NO_OUTPUT or "
                f"~fully laddered) - relay/network down or MinerU wedged. STOPPING (content "
                f"fails do NOT trigger this; restart MinerU / fix the relay, then re-run to resume)."
            )
            break

    # one retry pass for LADDER_FAIL docs (transient relay/MinerU page faults)
    if ladder_retry:
        log(f"--- retry pass: {len(ladder_retry)} laddered docs ---")
        for base, src in ladder_retry:
            r = process(base, src, base, retry=True)
            if r["outcome_class"] == "INGESTABLE" and r["ingested"]:
                ingested += 1
            elif r["outcome_class"] == "CONTENT_FAIL":
                content_fail.append((base, r["verdict"]))

    rebuild_sparse()
    log(
        f"DONE. ingested={ingested}  content_fail={len(content_fail)}  "
        f"still_laddered={sum(1 for b, _ in ladder_retry if not (load_ledger().get(b) or {}).get('ingested'))}"
    )
    if content_fail:
        log("content-gate failures (need extraction/gate fix, NOT ingested):")
        for b, v in content_fail:
            log(f"  {v:<26} {b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
