#!/usr/bin/env python3
"""PLAN_F1 WP-C - Phase 3 reconciliation wave 1 (AUTO-INGEST authorized).

Per doc, sequentially (sequential against M5):
  1. extract via the shipping CLI (full doc, batch 10, prod hybrid config) into
     output/phase5_reextract/<base>/ (so the sparse rebuild's auto-discovery
     covers it);
  2. enrich image chunks against the M5 VLM (so the IMAGE gate sees real
     visual_descriptions, not no_vlm);
  3. gate with qa_full_conversion.py --source-pdf;
  4. on QA_PASS / QA_PASS_WITH_ADVISORIES: dense-ingest (omlx Qwen3-Embedding-8B,
     incremental upsert into mmrag_v3__qwen3_local) - sparse is rebuilt ONCE at
     the end (BM25 idf is corpus-level; phase5_ingest_bm25 drops+recreates).
     On QA_WARN/QA_FAIL: NO ingest, record outcome, continue.

Rolling guard (Phase 4 rollback condition, live in production): STOP if >= 3 of
any 10 consecutive docs are QA_WARN/QA_FAIL, or ladder-served pages > 2%. A fired
guard is RECORDED, never pushed through.

Scope is the manifest pending/stale set MINUS the handover exclusions (Devlin,
Earthship, C++ manual) and, unless --include-code-books, the born-digital code
books gated on WP-A acceptance. Smallest-doc-first. No server reconfiguration;
relay endpoints (proven from conda). Outputs gitignored; results log committed-safe.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
OUT_DIR = ROOT / "output" / "phase5_reextract"
WAVE_DIR = ROOT / "output" / "wpc"
MANIFEST = ROOT / "corpus_manifest.jsonl"
QDRANT = "http://localhost:6333"
DENSE_COLLECTION = "mmrag_v3__qwen3_local"

# Relay endpoints (proven from conda; A1-resilient via the supervisor).
PROD_ENV = {
    "MINERU_ENDPOINT": os.environ.get("WPC_MINERU_ENDPOINT", "http://127.0.0.1:18001"),
    "MINERU_MODEL": "MinerU2.5-2509-1.2B",
    "VLM_NATIVE_ENDPOINT": os.environ.get("WPC_VLM_ENDPOINT", "http://127.0.0.1:18000/v1"),
    "VLM_NATIVE_MODEL": "mlx-community/Qwen3-VL-8B-Instruct-8bit",
    "VLM_NATIVE_API_KEY": "EMPTY",
}
ENRICH_ENV = {
    "MMRAG_ENRICH_PROVIDER": "openai",
    "MMRAG_ENRICH_MODEL": "mlx-community/Qwen3-VL-8B-Instruct-8bit",
    "MMRAG_ENRICH_BASE_URL": os.environ.get("WPC_VLM_ENDPOINT", "http://127.0.0.1:18000/v1"),
    "MMRAG_REFINER_API_KEY": "local",
}
ROUTE_KEYS = list(PROD_ENV) + [
    "USE_DOCLING_FAST",
    "USE_MINERU_ENGINE",
    "USE_VLM_ENGINE",
    "USE_HYBRID_ENGINE",
    "USE_MINERU_QWEN_HYBRID",
]

# Handover exclusions (never in wave 1).
EXCLUDE_SUBSTR = [
    "Devlin",  # P0 flat-source defect, disposition open
    "Earthship",  # scanned-prose item, separate register
    "The_Complete_C__Python_Coding_Manual",  # C++ manual, P3, awaits Phase 2
]
# Born-digital code books gated on WP-A acceptance (Section 1.1).
CODE_BOOK_SUBSTR = ["Chaubal", "Jungjun", "Bourne"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _http(method: str, url: str, body=None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _base_name(source_path: str) -> str:
    """A filesystem-safe output base from the data-relative path stem."""
    stem = Path(source_path).stem
    safe = "".join(c if c.isalnum() else "_" for c in stem)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")[:48]


def _dense_count(doc_id: str) -> int:
    body = {
        "exact": True,
        "filter": {"must": [{"key": "doc_id", "match": {"value": doc_id}}]},
    }
    try:
        return _http("POST", f"{QDRANT}/collections/{DENSE_COLLECTION}/points/count", body)[
            "result"
        ]["count"]
    except Exception:
        return -1


def load_scope(include_code_books: bool) -> list[dict]:
    rows = [json.loads(line) for line in open(MANIFEST) if line.strip()]
    # PENDING/STALE set = not currently a production-grade ingest.
    pending = []
    for r in rows:
        ing = r.get("ingest") or {}
        ext = r.get("extraction") or {}
        is_ingested = ing.get("outcome") == "INGESTED" and ing.get("points_dense", 0) > 0
        below = ext.get("engine") != "mineru_qwen_hybrid"
        if is_ingested and not below:
            continue  # CURRENT
        sp = r["source_path"]
        if any(s in sp for s in EXCLUDE_SUBSTR):
            continue
        if not include_code_books and any(s in sp for s in CODE_BOOK_SUBSTR):
            continue
        pending.append(r)
    # smallest-doc-first; unknown pages (never extracted) sort last by file size.
    pending.sort(key=lambda r: (r.get("pages") or 10_000, r["source_path"]))
    return pending


def extract(src: Path, out: Path, logf: Path) -> int:
    env = {k: v for k, v in os.environ.items() if k not in ROUTE_KEYS}
    env.update(PROD_ENV)
    cmd = [
        PY, "-m", "mmrag_v2.cli", "process", str(src),
        "--output-dir", str(out), "--batch-size", "10", "--vision-provider", "none",
    ]
    with open(logf, "w") as lf:
        return subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode


def enrich(jsonl: Path, logf: Path) -> int:
    env = dict(os.environ)
    env.update(ENRICH_ENV)
    cmd = [PY, "scripts/enrich_image_chunks_v29.py", str(jsonl)]
    with open(logf, "a") as lf:
        lf.write("\n=== ENRICH ===\n")
        return subprocess.run(cmd, env=env, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT).returncode


def qa(jsonl: Path, src: Path) -> str:
    p = subprocess.run(
        [PY, "scripts/qa_full_conversion.py", str(jsonl), "--source-pdf", str(src)],
        cwd=ROOT, capture_output=True, text=True,
    )
    for line in reversed(p.stdout.splitlines()):
        for tok in ("QA_PASS_WITH_ADVISORIES", "QA_PASS", "QA_WARN", "QA_FAIL"):
            if line.startswith(tok):
                return tok
    return "NO_VERDICT"


def dense_ingest(jsonl: Path, logf: Path) -> int:
    cmd = [
        PY, "scripts/ingest_to_qdrant.py", str(jsonl),
        "--provider", "omlx", "--model", "Qwen3-Embedding-8B-mxfp8",
        "--collection", DENSE_COLLECTION, "--qdrant-url", QDRANT,
    ]
    with open(logf, "a") as lf:
        lf.write("\n=== DENSE INGEST ===\n")
        return subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT).returncode


def header_of(jsonl: Path) -> dict:
    with open(jsonl) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("object_type") == "ingestion_metadata":
                return obj
            return obj
    return {}


def run_wave(include_code_books: bool, max_docs: int, page_cap: int) -> int:
    WAVE_DIR.mkdir(parents=True, exist_ok=True)
    results = WAVE_DIR / "wave_results.jsonl"
    scope = load_scope(include_code_books)
    if page_cap:
        scope = [r for r in scope if (r.get("pages") or 10_000) <= page_cap]
    if max_docs:
        scope = scope[:max_docs]
    log(f"WP-C wave: {len(scope)} docs in scope (include_code_books={include_code_books})")
    for r in scope:
        log(f"  - {r.get('pages') or '?'}p  {r['source_path']}")

    window10: list[str] = []
    served = degraded_total = 0
    for r in scope:
        src = ROOT / r["source_path"]
        if not src.exists():
            log(f"PHANTOM: source missing {src} - skip")
            continue
        base = _base_name(r["source_path"])
        out = OUT_DIR / base
        logf = WAVE_DIR / f"{base}.log"
        doc_id = r["doc_id"]
        log(f"=== {base} ({r.get('pages') or '?'}p) START")
        t0 = time.time()

        rc_ext = extract(src, out, logf)
        jsonl = out / "ingestion.jsonl"
        if not jsonl.exists():
            log(f"  {base}: NO_OUTPUT rc={rc_ext} - skip")
            _record(results, {"doc": base, "doc_id": doc_id, "verdict": "NO_OUTPUT", "rc": rc_ext})
            continue
        hdr = header_of(jsonl)
        engine = hdr.get("extraction_engine")
        deg = hdr.get("extraction_degraded_pages") or 0
        pages = hdr.get("total_pages")

        enrich(jsonl, logf)
        verdict = qa(jsonl, src)
        dense_before = _dense_count(doc_id)

        ingested = False
        if verdict in ("QA_PASS", "QA_PASS_WITH_ADVISORIES"):
            rc_ing = dense_ingest(jsonl, logf)
            ingested = rc_ing == 0
        dense_after = _dense_count(doc_id)
        wall = round(time.time() - t0, 1)

        rec = {
            "doc": base, "doc_id": doc_id, "source_path": r["source_path"],
            "engine": engine, "degraded": deg, "pages": pages,
            "verdict": verdict, "ingested": ingested,
            "dense_before": dense_before, "dense_after": dense_after, "wall_s": wall,
        }
        _record(results, rec)
        log(
            f"  {base}: {verdict} engine={engine} deg={deg} ingested={ingested} "
            f"dense {dense_before}->{dense_after} {wall}s"
        )

        # rolling guard
        window10.append(verdict)
        window10 = window10[-10:]
        if pages:
            served += pages
            degraded_total += deg
        wf = sum(1 for v in window10 if v in ("QA_WARN", "QA_FAIL"))
        ladder_rate = (degraded_total / served) if served else 0.0
        if wf >= 3:
            log(f"ROLLBACK GUARD FIRED: {wf}/10 consecutive QA_WARN/FAIL. STOPPING. window={window10}")
            return 3
        if ladder_rate > 0.02:
            log(f"ROLLBACK GUARD FIRED: ladder-served {degraded_total}/{served}={ladder_rate:.3%}>2%. STOPPING.")
            return 3

    log("WAVE COMPLETE (no guard fired). Rebuild sparse next: scripts/phase5_ingest_bm25.py")
    return 0


def _record(path: Path, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--include-code-books", action="store_true",
                    help="include Chaubal/Jungjun/Bourne (only if WP-A acceptance passed)")
    ap.add_argument("--max-docs", type=int, default=0, help="cap docs this wave (0 = all in scope)")
    ap.add_argument("--page-cap", type=int, default=0, help="skip docs with more pages than this")
    ap.add_argument("--dry-run", action="store_true", help="print scope and exit")
    a = ap.parse_args()
    if a.dry_run:
        scope = load_scope(a.include_code_books)
        if a.page_cap:
            scope = [r for r in scope if (r.get("pages") or 10_000) <= a.page_cap]
        if a.max_docs:
            scope = scope[: a.max_docs]
        print(f"SCOPE ({len(scope)} docs):")
        for r in scope:
            print(f"  {r.get('pages') or '?':>5}p  {r['source_path']}")
        return 0
    return run_wave(a.include_code_books, a.max_docs, a.page_cap)


if __name__ == "__main__":
    raise SystemExit(main())
