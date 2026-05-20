#!/usr/bin/env python3
"""v2.10 retrieval-regression harness.

Pins the v2.10.0-rc1 retrieval shape against `mmrag_v2_8` so any future
drift (chunker change, embedder change, dedup-rule change, JSONL
reconvert, etc.) is visible in a side-by-side diff.

The script intentionally does NOT judge retrieval quality. Its job is
to detect *change* relative to a tracked baseline. Quality evaluation
belongs to the synthetic-soak / LLM-judge layer (see Phase 9 plan
when authored).

Two modes:

  --capture     embed each query, capture top-5 from `mmrag_v2_8`, and
                write `tests/fixtures/retrieval_regression_v2_10.json`.
                Use this once after a rebuild to refresh the baseline.

  default       verify against the tracked baseline.

                STRICT  top-3 chunk_ids match the baseline -> PASS.
                LOOSE   top-1 doc_id matches the baseline   -> PASS_WITH_DRIFT
                        (top-3 reshuffled within a stable head).
                FAIL    top-1 doc_id changed -> exit 1.

Queries exercise every major v2.10 processing lane (OCR-heading,
HybridChunker cross-page split, EPUB synthetic pagination, magazine,
form-class, code-heavy, multilingual). Twenty queries across the
34-doc corpus.

This script uses the same Ollama + Qdrant primitives as
`scripts/search_qdrant.py`; the rerank step is intentionally skipped
(deterministic vector-only signature, no DASHSCOPE_API_KEY dependence).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from search_qdrant import embed as _embed_ollama, search  # noqa: E402
from ingest_to_qdrant import embed_text_dashscope  # noqa: E402

FIXTURE_PATH_DEFAULT = REPO_ROOT / "tests" / "fixtures" / "retrieval_regression_v2_10.json"
COLLECTION_DEFAULT = "mmrag_v2_8"
EMBED_MODEL_DEFAULT_OLLAMA = "llava"
EMBED_MODEL_DEFAULT_DASHSCOPE = "text-embedding-v4"
TOP_K = 5
STRICT_K = 3  # top-N chunk_ids must equal the baseline for STRICT pass

# (query_id, query_text). The expected-primary-doc concept was removed:
# this harness pins the CURRENT retrieval shape, whatever it is, so
# drift is detectable. Semantic quality is the soak layer's job.
QUERIES: list[tuple[str, str]] = [
    # AI / RAG / Agent books
    ("Q01_mcp",          "what is the Model Context Protocol and how does an MCP client work"),
    ("Q02_rag_eval",     "how to evaluate a retrieval augmented generation system"),
    ("Q03_agent_tools",  "how do LLM agents call tools and reason over results"),
    ("Q04_rag_arch",     "architecture of a production RAG pipeline"),
    # Coding / Python books
    ("Q05_iterator",     "how to write a Python iterator with __iter__ and __next__"),
    ("Q06_decorator",    "Python decorator recipe with functools.wraps"),
    ("Q07_module_org",   "structuring Python modules and packages for a large project"),
    ("Q08_pytorch_cnn",  "PyTorch convolutional neural network for image classification"),
    ("Q09_arcgis",       "automate ArcGIS Pro workflow with a Python script"),
    ("Q10_patterns",     "design patterns and idioms in Python"),
    # EPUB lane (Dutch, synthetic-page mapping must survive)
    ("Q11_chatgpt_nl",   "hoe schrijf ik betere ChatGPT prompts in het Nederlands"),
    ("Q12_ki_epub_nl",   "digitale producten verkopen met AI en ChatGPT"),
    # Magazine lane
    ("Q13_pcworld",      "Windows 11 PC build advice and hardware recommendations"),
    ("Q14_combat",       "F-35 Lightning II stealth fighter operations"),
    # OCR / scanned-class
    ("Q15_firearms",     "firearm cleaning and safety procedure step by step"),
    # Engineering / domain
    ("Q16_earthship",    "Earthship passive solar design and rainwater harvesting"),
    ("Q17_greenhouse",   "greenhouse climate control and ventilation strategy"),
    # Academic
    ("Q18_aios",         "what is an LLM agent operating system"),
    ("Q19_solar_pv",     "modelling a photovoltaic solar PV system performance"),
    # Literature
    ("Q20_harry",        "Harry Potter Quidditch match against Slytherin"),
]


def _embed_query(text: str, provider: str, model: str,
                 ollama_url: str, api_key: str) -> list[float]:
    """Embed a query through the chosen provider."""
    if provider == "dashscope":
        return embed_text_dashscope(text, model, api_key)
    return _embed_ollama(text, model=model, ollama_url=ollama_url)


def _run_query(query_text: str, collection: str, qdrant_url: str,
               provider: str, model: str, ollama_url: str, api_key: str) -> list[dict]:
    vector = _embed_query(query_text, provider, model, ollama_url, api_key)
    return search(vector, collection, limit=TOP_K, qdrant_url=qdrant_url)


def _result_summary(result: dict) -> dict:
    payload = result.get("payload") or {}
    return {
        "chunk_id": payload.get("chunk_id") or str(result.get("id")),
        "doc_id": payload.get("doc_id"),
        "source_file": payload.get("source_file"),
        "modality": payload.get("modality"),
        "page_number": payload.get("page_number"),
        "score": round(float(result.get("score") or 0.0), 6),
    }


def capture(fixture_path: Path, collection: str, engine_version: str,
            qdrant_url: str, provider: str, model: str,
            ollama_url: str, api_key: str) -> None:
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {
        "schema": 2,
        "engine_version": engine_version,
        "collection": collection,
        "provider": provider,
        "embed_model": model,
        "top_k": TOP_K,
        "strict_k": STRICT_K,
        "queries": [],
    }
    for qid, qtext in QUERIES:
        print(f"  capture {qid:14s} {qtext!r}", flush=True)
        results = _run_query(qtext, collection, qdrant_url, provider, model,
                             ollama_url, api_key)
        out["queries"].append({
            "id": qid,
            "text": qtext,
            "top_k": [_result_summary(r) for r in results[:TOP_K]],
        })
    fixture_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nWrote fingerprint: {fixture_path}")
    print(f"Queries captured:  {len(out['queries'])}")


def _summarize_top1(entry_top_k: list[dict]) -> str:
    if not entry_top_k:
        return "<empty>"
    r = entry_top_k[0]
    src = (r.get("source_file") or "").split("/")[-1][:42]
    return f"score={r.get('score'):.3f} doc={r.get('doc_id')} p={r.get('page_number')} src={src!r}"


def verify(fixture_path: Path, collection: str, qdrant_url: str,
           provider: str, model: str, ollama_url: str, api_key: str) -> int:
    if not fixture_path.exists():
        print(f"ERROR: fingerprint missing at {fixture_path}; run with --capture first", file=sys.stderr)
        return 2
    baseline = json.loads(fixture_path.read_text())
    if baseline.get("collection") != collection:
        print(f"ERROR: fingerprint collection mismatch ({baseline.get('collection')} != {collection})", file=sys.stderr)
        return 2

    failures: list[str] = []
    drifts: list[str] = []
    rows: list[str] = []

    for entry in baseline["queries"]:
        qid = entry["id"]
        qtext = entry["text"]
        baseline_top = entry["top_k"]
        baseline_chunk_ids = [r["chunk_id"] for r in baseline_top]
        baseline_top1_doc = baseline_top[0]["doc_id"] if baseline_top else None

        results = _run_query(qtext, collection, qdrant_url, provider, model,
                             ollama_url, api_key)
        current_top = [_result_summary(r) for r in results[:TOP_K]]
        current_chunk_ids = [r["chunk_id"] for r in current_top]
        current_top1_doc = current_top[0]["doc_id"] if current_top else None

        strict_hit = current_chunk_ids[:STRICT_K] == baseline_chunk_ids[:STRICT_K]
        loose_hit = current_top1_doc == baseline_top1_doc

        if not loose_hit:
            status = "FAIL"
            failures.append(
                f"{qid}: top-1 doc_id changed\n"
                f"  baseline: {baseline_top1_doc} ({_summarize_top1(baseline_top)})\n"
                f"  current : {current_top1_doc} ({_summarize_top1(current_top)})"
            )
        elif not strict_hit:
            status = "DRIFT"
            drifts.append(
                f"{qid}: top-{STRICT_K} chunk_ids reshuffled (top-1 doc stable)\n"
                f"  baseline: {baseline_chunk_ids[:STRICT_K]}\n"
                f"  current : {current_chunk_ids[:STRICT_K]}"
            )
        else:
            status = "PASS"

        rows.append(f"  {qid:14s} {status:5s} {_summarize_top1(current_top)}")

    print("=" * 78)
    print(f"Retrieval regression — {collection} ({provider}/{model}, top-{TOP_K}, strict={STRICT_K})")
    print(f"Baseline:  {fixture_path}")
    print(f"Engine:    baseline={baseline.get('engine_version')}")
    print("=" * 78)
    for row in rows:
        print(row)
    print()
    if drifts:
        print(f"DRIFT (top-{STRICT_K} chunk_ids reshuffled, top-1 doc stable) — {len(drifts)} queries:")
        for d in drifts:
            print(f"  - {d}")
        print()
    if failures:
        print(f"FAILURES (top-1 doc_id changed) — {len(failures)} queries:")
        for f in failures:
            print(f"  - {f}")
        print("\nRETRIEVAL_REGRESSION: FAIL")
        return 1
    if drifts:
        print("RETRIEVAL_REGRESSION: PASS_WITH_DRIFT")
        return 0
    print("RETRIEVAL_REGRESSION: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--capture", action="store_true",
                        help="capture fingerprint (overwrites the target fixture)")
    parser.add_argument("--collection", default=COLLECTION_DEFAULT,
                        help=f"Target Qdrant collection (default: {COLLECTION_DEFAULT})")
    parser.add_argument("--fixture", default=None,
                        help="Fixture JSON path. Defaults: tests/fixtures/retrieval_regression_v2_10.json "
                             "(provider=ollama), tests/fixtures/retrieval_regression_v2_11_qwen3.json "
                             "(provider=dashscope).")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "dashscope"],
                        help="Embedding provider for query side. Must match how the target "
                             "collection was built.")
    parser.add_argument("--model", default=None,
                        help="Embedding model. Default 'llava' for ollama; "
                             "'text-embedding-v4' for dashscope.")
    parser.add_argument("--engine-version", default=None,
                        help="Engine version string written into the captured fixture. "
                             "Defaults: '2.10.0' (ollama) / '2.11.0-candidate' (dashscope).")
    parser.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--api-key", default=None,
                        help="Dashscope API key. Defaults to DASHSCOPE_API_KEY env var.")
    args = parser.parse_args()

    # Provider-aware defaults.
    if args.model is None:
        args.model = (EMBED_MODEL_DEFAULT_OLLAMA if args.provider == "ollama"
                      else EMBED_MODEL_DEFAULT_DASHSCOPE)
    if args.fixture is None:
        if args.provider == "ollama":
            args.fixture = str(FIXTURE_PATH_DEFAULT)
        else:
            args.fixture = str(REPO_ROOT / "tests" / "fixtures"
                              / "retrieval_regression_v2_11_qwen3.json")
    if args.engine_version is None:
        args.engine_version = ("2.10.0" if args.provider == "ollama"
                              else "2.11.0-candidate")
    if args.api_key is None:
        args.api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if args.provider == "dashscope" and not args.api_key:
        print("ERROR: --provider dashscope requires --api-key or DASHSCOPE_API_KEY", file=sys.stderr)
        return 2

    fixture_path = Path(args.fixture)
    if args.capture:
        capture(fixture_path, args.collection, args.engine_version,
                args.qdrant_url, args.provider, args.model,
                args.ollama_url, args.api_key)
        return 0
    return verify(fixture_path, args.collection, args.qdrant_url,
                  args.provider, args.model, args.ollama_url, args.api_key)


if __name__ == "__main__":
    raise SystemExit(main())
