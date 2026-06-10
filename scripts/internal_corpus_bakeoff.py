#!/usr/bin/env python3
"""Internal-corpus half of the Phase 1 bake-off (PLAN_EXTRACTION_FIDELITY_V1 WP-3).

OmniDocBench is English+Chinese only; the internal crucible owns the classes it
cannot speak for (dense automotive tables, magazines, code-indentation, Dutch/
German forms + technical manuals). This runner drives a SMALL fixed subset (6
docs, ~15 pages each) through every registered engine END-TO-END on the shipping
path (`mmrag-v2 process` -> mmrag_v3.extract -> chunk -> JSONL), then leaves the
JSONL + provenance for the WP-3 scorers (built junk-presence gate signals; the R3
code-indentation gate; ladder/provenance stats; qualitative omission diff).

STANDALONE: reuses ONLY the engine-env table from omnidocbench_bakeoff (no
extraction imports); shells out to the mmrag-v2 CLI exactly like the OmniDocBench
adapter. Per the seeded-fault blindness report (FINDINGS_LOG 2026-06-10):
  - code class is judged by the R3 indentation gate, NEVER text-ED (text-ED is
    whitespace-blind);
  - the junk-presence gate signals detect ADDED junk, are BLIND to content
    OMISSION, and are reported as exactly that;
  - content-omission verdicts are QUALITATIVE (artifact diff), recorded as such.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from omnidocbench_bakeoff import ENGINES, _engine_env, MMRAG_CLI

HOME = Path.home()
PROJ = Path(__file__).resolve().parent.parent
OUT_ROOT = HOME / "omnidocbench-eval" / "internal_wp3"

# 6-doc fixed subset spanning the thesis classes. page_spec is the `--pages`
# argument (a max-count "15" or a comma list of interior pages). Interior ranges
# skip covers/ads; the code doc is anchored on design-patterns p66 (the
# morning-report-validated dense indented-code page).
DOCS = [
    {"key": "carok_table", "klass": "dense_automotive_table",
     "pdf": "data/data_spreadsheet/CarOK voorraadtelling 2021-04.pdf",
     "pages": "15"},
    {"key": "combat_magazine", "klass": "magazine",
     "pdf": "data/digital_magazine/Combat Aircraft - August 2025 UK.pdf",
     "pages": ",".join(str(p) for p in range(8, 23))},
    {"key": "designpatterns_code", "klass": "code_heavy",
     "pdf": "data/technical_manual/Ayeva K. Mastering Python Design Patterns...essential Python patterns...3ed 2024.pdf",
     "pages": ",".join(str(p) for p in range(66, 81))},
    {"key": "dutch_dispute_form", "klass": "dutch_form",
     "pdf": "data/business_form/betwistingsformulier_aankoop_niet_ontvangen.pdf",
     "pages": "15"},
    {"key": "scanned_form_0013", "klass": "scanned_form",
     "pdf": "data/business_form/0013_140302111325_001.pdf",
     "pages": "15"},
    {"key": "grundlagen_de_manual", "klass": "german_technical_manual",
     "pdf": "data/raw/Grundlagen Fahrzeug- und Motorentechnik.pdf",
     "pages": ",".join(str(p) for p in range(20, 35))},
]


def run_one(engine: str, doc: dict, force: bool) -> dict:
    env = _engine_env(engine)
    out_dir = OUT_ROOT / engine / doc["key"]
    ingestion = out_dir / "ingestion.jsonl"
    if ingestion.exists() and not force:
        return {"engine": engine, "doc": doc["key"], "status": "skipped"}
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = PROJ / doc["pdf"]
    if not pdf.exists():
        return {"engine": engine, "doc": doc["key"], "status": "missing_pdf", "pdf": str(pdf)}
    cmd = [str(MMRAG_CLI), "process", str(pdf), "--batch-size", "10",
           "--vision-provider", "none", "--pages", doc["pages"],
           "--output-dir", str(out_dir)]
    print(f"[{engine}/{doc['key']}] pages={doc['pages']} ...", flush=True)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    ok = proc.returncode == 0 and ingestion.exists()
    if not ok:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
        for line in tail:
            print(f"    | {line}", flush=True)
    # provenance from the JSONL header
    prov = {}
    if ingestion.exists():
        for line in ingestion.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                if obj.get("object_type") == "ingestion_metadata":
                    prov = {k: obj.get(k) for k in (
                        "extraction_engine", "extraction_fallback",
                        "extraction_degraded_pages", "extraction_recovered_pages",
                        "total_pages")}
                break
    return {"engine": engine, "doc": doc["key"], "status": "ok" if ok else "fail",
            "rc": proc.returncode, "provenance": prov}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engines", nargs="+", default=["docling_fast", "mineru", "qwen3vl", "hybrid"])
    ap.add_argument("--docs", nargs="+", default=[d["key"] for d in DOCS])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    docs = [d for d in DOCS if d["key"] in args.docs]
    results = []
    for engine in args.engines:
        if engine not in ENGINES:
            print(f"  unknown engine {engine}, skipping")
            continue
        for doc in docs:
            results.append(run_one(engine, doc, args.force))
    (OUT_ROOT / "run_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n===== WP-3 RUN SUMMARY =====")
    for r in results:
        prov = r.get("provenance", {})
        print(f"  {r['engine']:<14} {r['doc']:<22} {r['status']:<10} "
              f"engine={prov.get('extraction_engine')} fb={prov.get('extraction_fallback')} "
              f"deg={prov.get('extraction_degraded_pages')} pages={prov.get('total_pages')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
