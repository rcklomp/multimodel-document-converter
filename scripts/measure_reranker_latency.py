#!/usr/bin/env python3
"""v2.12 Phase 1 pre-work — empirical reranker latency budget.

Runs the 20 v2.11 retrieval-regression queries through the full
production retrieval pipeline (embed -> Qdrant -> rerank) at several
candidate-set sizes (top_k_retrieve in {10, 25, 50, 100}). Measures
each stage's latency separately and reports p50 / p95 / p99 / max
per K so the v2.12 Phase 1 §"latency budget" open question can be
answered with data rather than a guess.

Endpoint: Dashscope intl `gte-rerank` (the production reranker for
v2.12 Phase 1; the cn endpoint exposes the same model under the name
`gte-rerank-v2`).

Cost: ~$0.001 per rerank call * 20 queries * |K_values| * samples
      = ~$0.24 for the default config. Bounded.

Wall time: ~5 minutes for the default config.

Usage:
    DASHSCOPE_API_KEY=... python scripts/measure_reranker_latency.py
    python scripts/measure_reranker_latency.py --samples 5 --k-values 25,50
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from ingest_to_qdrant import embed_text_dashscope  # noqa: E402
from search_qdrant import search  # noqa: E402

DEFAULT_RERANK_URL_DASHSCOPE = "https://dashscope-intl.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
DEFAULT_RERANK_MODEL_DASHSCOPE = "gte-rerank"

DEFAULT_RERANK_URL_OMLX = "http://10.0.10.246:8000/v1/rerank"
DEFAULT_RERANK_MODEL_OMLX = "mlx-community_Qwen3-Reranker-8B-mxfp8"

# 20 queries lifted from scripts/retrieval_regression.py (single source).
QUERIES = [
    ("Q01_mcp", "what is the Model Context Protocol and how does an MCP client work"),
    ("Q02_rag_eval", "how to evaluate a retrieval augmented generation system"),
    ("Q03_agent_tools", "how do LLM agents call tools and reason over results"),
    ("Q04_rag_arch", "architecture of a production RAG pipeline"),
    ("Q05_iterator", "how to write a Python iterator with __iter__ and __next__"),
    ("Q06_decorator", "Python decorator recipe with functools.wraps"),
    ("Q07_module_org", "structuring Python modules and packages for a large project"),
    ("Q08_pytorch_cnn", "PyTorch convolutional neural network for image classification"),
    ("Q09_arcgis", "automate ArcGIS Pro workflow with a Python script"),
    ("Q10_patterns", "design patterns and idioms in Python"),
    ("Q11_chatgpt_nl", "hoe schrijf ik betere ChatGPT prompts in het Nederlands"),
    ("Q12_ki_epub_nl", "digitale producten verkopen met AI en ChatGPT"),
    ("Q13_pcworld", "Windows 11 PC build advice and hardware recommendations"),
    ("Q14_combat", "F-35 Lightning II stealth fighter operations"),
    ("Q15_firearms", "firearm cleaning and safety procedure step by step"),
    ("Q16_earthship", "Earthship passive solar design and rainwater harvesting"),
    ("Q17_greenhouse", "greenhouse climate control and ventilation strategy"),
    ("Q18_aios", "what is an LLM agent operating system"),
    ("Q19_solar_pv", "modelling a photovoltaic solar PV system performance"),
    ("Q20_harry", "Harry Potter Quidditch match against Slytherin"),
]


def rerank_call(query: str, documents: list[str], api_key: str,
                url: str, model: str, top_n: int = 5,
                timeout: int = 60) -> tuple[list[dict], float]:
    """Call a rerank endpoint. Returns (results, elapsed_seconds).

    Supports two payload shapes:
      - Dashscope `gte-rerank` (input/parameters wrapper)
      - Cohere-style `rerank` (top-level query/documents, e.g. omlx)
    """
    is_dashscope = "dashscope" in url
    if is_dashscope:
        body = json.dumps({
            "model": model,
            "input": {"query": query, "documents": documents},
            "parameters": {"top_n": top_n, "return_documents": False},
        }).encode("utf-8")
    else:
        body = json.dumps({
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": False,
        }).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    start = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=timeout)
    payload = json.loads(resp.read())
    elapsed = time.perf_counter() - start
    if is_dashscope:
        results = payload.get("output", {}).get("results", [])
    else:
        results = payload.get("results", [])
    return results, elapsed


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:7.1f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collection", default="mmrag_v2_8__qwen3_dashscope")
    parser.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--k-values", default="10,25,50,100",
                        help="Comma-separated top-K candidate-set sizes to test.")
    parser.add_argument("--samples", type=int, default=3,
                        help="Number of timing samples per (query, K).")
    parser.add_argument("--top-n", type=int, default=5,
                        help="top_n returned by the reranker (production default: 5).")
    parser.add_argument("--api-key", default=None,
                        help="Embed-side API key (Dashscope). Default: DASHSCOPE_API_KEY env.")
    parser.add_argument("--embed-model", default="text-embedding-v4")
    parser.add_argument("--rerank-backend", default="dashscope",
                        choices=["dashscope", "omlx"],
                        help="Reranker backend. 'dashscope' = cloud gte-rerank "
                             "(intl endpoint). 'omlx' = local mlx-omni-server "
                             f"at {DEFAULT_RERANK_URL_OMLX}.")
    parser.add_argument("--rerank-url", default=None,
                        help="Override rerank URL.")
    parser.add_argument("--rerank-model", default=None,
                        help="Override rerank model name.")
    parser.add_argument("--rerank-api-key-env", default=None,
                        help="Env var name to read the rerank API key from. "
                             "Default: DASHSCOPE_API_KEY (dashscope) or "
                             "MLX_API_KEY (omlx).")
    parser.add_argument("--max-chunk-chars", type=int, default=1500,
                        help="Truncate each chunk's content to this many chars "
                             "before sending to the reranker (matches typical "
                             "production payload shape).")
    parser.add_argument("--output-json", default=None,
                        help="Optional path to dump the full per-call timing data.")
    args = parser.parse_args()

    # Embed-side API key (Dashscope for production embedder).
    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        print("ERROR: DASHSCOPE_API_KEY env var required for embedding.", file=sys.stderr)
        return 2

    # Rerank-side endpoint, model, api key.
    if args.rerank_backend == "dashscope":
        rerank_url = args.rerank_url or DEFAULT_RERANK_URL_DASHSCOPE
        rerank_model = args.rerank_model or DEFAULT_RERANK_MODEL_DASHSCOPE
        rerank_api_key_env = args.rerank_api_key_env or "DASHSCOPE_API_KEY"
    else:  # omlx
        rerank_url = args.rerank_url or DEFAULT_RERANK_URL_OMLX
        rerank_model = args.rerank_model or DEFAULT_RERANK_MODEL_OMLX
        rerank_api_key_env = args.rerank_api_key_env or "MLX_API_KEY"
    rerank_api_key = os.environ.get(rerank_api_key_env, "")
    if not rerank_api_key:
        print(f"ERROR: {rerank_api_key_env} env var required for rerank backend "
              f"'{args.rerank_backend}'.", file=sys.stderr)
        return 2

    k_values = [int(k) for k in args.k_values.split(",")]
    print(f"=== Reranker latency benchmark — {rerank_model} ===")
    print(f"Backend:    {args.rerank_backend} ({rerank_url})")
    print(f"Collection: {args.collection}")
    print(f"K values:   {k_values}")
    print(f"Samples:    {args.samples} per (query, K)")
    print(f"top_n:      {args.top_n} (production default)")
    print(f"Queries:    {len(QUERIES)}")
    print(f"Total calls: {len(QUERIES) * len(k_values) * args.samples}")
    print(f"Est cost:   ~${len(QUERIES) * len(k_values) * args.samples * 0.001:.2f}")
    print()

    # Pre-fetch Qdrant top-100 per query once so we time only the reranker.
    print("--- Pre-fetching Qdrant candidates (one embed + one search per query) ---")
    candidates_by_query: dict[str, list[dict]] = {}
    embed_times: list[float] = []
    qdrant_times: list[float] = []
    max_k = max(k_values)
    for qid, qtext in QUERIES:
        t0 = time.perf_counter()
        vec = embed_text_dashscope(qtext, args.embed_model, api_key)
        embed_times.append(time.perf_counter() - t0)
        t1 = time.perf_counter()
        results = search(vec, args.collection, limit=max_k, qdrant_url=args.qdrant_url)
        qdrant_times.append(time.perf_counter() - t1)
        candidates_by_query[qid] = results
        print(f"  {qid:14s} embed={fmt_ms(embed_times[-1])}ms search={fmt_ms(qdrant_times[-1])}ms candidates={len(results)}")
    print()

    # Now time the reranker call separately, sweeping K.
    print("--- Measuring reranker latency per K ---")
    timings: dict[int, list[float]] = {k: [] for k in k_values}
    per_call: list[dict] = []
    for qid, qtext in QUERIES:
        cands = candidates_by_query[qid]
        for k in k_values:
            subset = cands[:k]
            docs = [
                (((c.get("payload") or {}).get("content") or "")[:args.max_chunk_chars])
                for c in subset
            ]
            for s in range(args.samples):
                try:
                    rerank_results, elapsed = rerank_call(
                        qtext, docs, rerank_api_key,
                        rerank_url, rerank_model, top_n=args.top_n,
                    )
                except urllib.error.HTTPError as e:
                    print(f"    ! HTTPError {e.code} on {qid} K={k} sample={s}: {e.read()[:120]}",
                          file=sys.stderr)
                    continue
                except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                    print(f"    ! Network error on {qid} K={k} sample={s}: {e}",
                          file=sys.stderr)
                    continue
                timings[k].append(elapsed)
                per_call.append({
                    "query_id": qid,
                    "k": k,
                    "sample": s,
                    "elapsed_seconds": elapsed,
                    "candidates_sent": len(docs),
                    "top_indices": [r.get("index") for r in rerank_results[:args.top_n]],
                    "top_scores": [r.get("relevance_score") for r in rerank_results[:args.top_n]],
                })
        print(f"  {qid:14s} samples for K in {k_values} captured")
    print()

    # Aggregate stats.
    print("=" * 78)
    print("Reranker latency distribution (ms) — collection top-K candidates -> reranked top-N")
    print("=" * 78)
    print(f"{'K':>5}  {'samples':>8}  {'min':>7}  {'p50':>7}  {'p95':>7}  {'p99':>7}  {'max':>7}  {'mean':>7}")
    print("-" * 78)
    for k in k_values:
        t = timings[k]
        if not t:
            print(f"{k:>5}  {'NO DATA':>8}")
            continue
        print(
            f"{k:>5}  {len(t):>8}  "
            f"{fmt_ms(min(t))}  "
            f"{fmt_ms(percentile(t, 0.50))}  "
            f"{fmt_ms(percentile(t, 0.95))}  "
            f"{fmt_ms(percentile(t, 0.99))}  "
            f"{fmt_ms(max(t))}  "
            f"{fmt_ms(statistics.mean(t))}"
        )
    print()

    print("--- Reference: per-stage baseline (no rerank) ---")
    print(f"{'stage':>10}  {'min':>7}  {'p50':>7}  {'p95':>7}  {'p99':>7}  {'max':>7}  {'mean':>7}")
    print("-" * 78)
    for label, t in [("embed", embed_times), ("qdrant", qdrant_times)]:
        print(
            f"{label:>10}  "
            f"{fmt_ms(min(t))}  "
            f"{fmt_ms(percentile(t, 0.50))}  "
            f"{fmt_ms(percentile(t, 0.95))}  "
            f"{fmt_ms(percentile(t, 0.99))}  "
            f"{fmt_ms(max(t))}  "
            f"{fmt_ms(statistics.mean(t))}"
        )
    print()

    # Estimated end-to-end p99 per K.
    print("--- Estimated end-to-end (embed + qdrant + rerank) p99 per K ---")
    embed_p99 = percentile(embed_times, 0.99)
    qdrant_p99 = percentile(qdrant_times, 0.99)
    print(f"  Stage baselines p99: embed={fmt_ms(embed_p99)}ms qdrant={fmt_ms(qdrant_p99)}ms")
    print(f"  {'K':>5}  {'rerank p99':>11}  {'total p99':>11}  {'vs 1500ms budget':>20}")
    for k in k_values:
        if not timings[k]:
            continue
        rerank_p99 = percentile(timings[k], 0.99)
        total = embed_p99 + qdrant_p99 + rerank_p99
        verdict = "PASS" if total <= 1.5 else f"OVER by {(total - 1.5)*1000:.0f}ms"
        print(f"  {k:>5}  {fmt_ms(rerank_p99)}ms  {fmt_ms(total)}ms  {verdict:>20}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "backend": args.rerank_backend,
            "rerank_url": rerank_url,
            "rerank_model": rerank_model,
            "embed_model": args.embed_model,
            "collection": args.collection,
            "k_values": k_values,
            "samples_per_combo": args.samples,
            "top_n_returned": args.top_n,
            "max_chunk_chars": args.max_chunk_chars,
            "embed_times_seconds": embed_times,
            "qdrant_times_seconds": qdrant_times,
            "rerank_timings_seconds": {str(k): v for k, v in timings.items()},
            "per_call": per_call,
        }, indent=2))
        print(f"\nFull data written to {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
