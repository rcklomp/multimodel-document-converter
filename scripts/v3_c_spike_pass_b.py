#!/usr/bin/env python3
"""V3 Phase C C-Spike — PASS B (reranker discrimination on bounded join).

Charter §4.2 step 2 #8 (TIGHTENED in 0.4):

    Reranker top-1 selection rate on visually-retrieved pages ≥60%.
    The simulated rerank must use:
      - The exact ModernBERT model that production uses
      - The exact production prompt for the reranker
      - Candidate set construction: for each query where visual
        retrieval placed the correct page in the top-5, construct
        a candidate set =
          (top-25 text chunks from the full corpus under v2.16
           text retrieval)
          ∪
          (the top-N=3 text chunks of the visually-retrieved page,
           selected by their text-leg score per §3.4 #4 bounded join)
      - Deduplicate by chunk_id
      - Gold-chunk-on-gold-page mapping from regression fixture

This script reads the run-1 JSON produced by `v3_c_spike.py`, builds
the candidate set per Charter, runs the production reranker, and
reports whether top-1 lands on the gold page for each qualifying
query.

Because the ATZ corpus has no per-chunk regression-fixture gold map
(see V3_C_PRESPIKE_REPORT.md §"Gold page selection"), we measure at
the page level: PASS B succeeds for a query iff the reranker top-1
chunk is on the gold page. This is looser than the Charter's per-
chunk-gold criterion but it's the strictest version computable
without a fixture build-out.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CHARTER_PASS_B_THRESHOLD = 0.60
DEFAULT_TEXT_TOP_N = 25
DEFAULT_VISUAL_PAGE_TOP_K = 5  # consult visual top-5 per §3.4 #4 bounded join
DEFAULT_CHUNKS_PER_VISUAL_PAGE = 3  # N=3 per Charter §3.4 #4


# ---------------------------------------------------------------------------
# Production-aligned text retrieval (top-25)
# ---------------------------------------------------------------------------


def run_text_top_n(query: str, *, top_n: int = DEFAULT_TEXT_TOP_N) -> List[Dict[str, Any]]:
    """Production v2.16 text retrieval, top-N chunks corpus-wide."""
    from mmrag_v2.retrieval.pipeline import retrieve_hybrid_reranked

    raw = retrieve_hybrid_reranked(
        query,
        top_n_return=top_n,
        top_k_retrieve=top_n,
        top_n_fuse=top_n,
    )
    chunks = []
    for idx, r in enumerate(raw):
        payload = r.get("payload") or {}
        # page_number is top-level in production payload (NOT nested
        # under metadata) per ingest_to_qdrant.py. Defend against either
        # shape so a future payload refactor doesn't silently zero this.
        page_number = payload.get("page_number")
        if page_number is None:
            meta = payload.get("metadata") or {}
            if isinstance(meta, dict):
                page_number = meta.get("page_number")
        chunks.append({
            "chunk_id": payload.get("chunk_id"),
            "doc_id": payload.get("doc_id"),
            "page_number": page_number,
            "content": payload.get("content") or "",
            "rerank_score": float(r.get("rerank_score", 0.0)),
            "score": float(r.get("score", 0.0)),
            "text_rank": idx,
            "source": "text_top_n",
        })
    return chunks


# ---------------------------------------------------------------------------
# Per-page bounded join: top-3 chunks of a given (doc, page) by text-leg score
# ---------------------------------------------------------------------------


def embed_query_omlx(query: str) -> List[float]:
    """Embed a query via the production omlx Qwen3-Embedding-8B endpoint."""
    import urllib.request
    api_key = os.environ.get("MLX_API_KEY")
    if not api_key:
        raise RuntimeError("MLX_API_KEY must be set for omlx embedder dispatch")
    body = json.dumps({
        "model": "Qwen3-Embedding-8B-mxfp8",
        "input": query,
    }).encode("utf-8")
    req = urllib.request.Request(
        "http://10.0.10.246:8000/v1/embeddings",
        data=body, method="POST",
    )
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["data"][0]["embedding"]


def chunks_for_doc_page(
    *,
    query_vector: List[float],
    doc_id: str,
    page_number: int,
    top_k: int,
    collection: str = "mmrag_v2_8__qwen3_local",
    qdrant_url: str = "http://localhost:6333",
) -> List[Dict[str, Any]]:
    """Top-K text-leg chunks restricted to (doc_id, page_number) via Qdrant filter."""
    import urllib.request
    # Filter on top-level page_number — production payload stores it
    # at the root of the payload (see ingest_to_qdrant.py and the
    # corresponding fix in run_text_top_n above).
    body = json.dumps({
        "query": query_vector,
        "limit": top_k,
        "with_payload": True,
        "filter": {
            "must": [
                {"key": "doc_id", "match": {"value": doc_id}},
                {"key": "page_number", "match": {"value": page_number}},
            ]
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{qdrant_url}/collections/{collection}/points/query",
        data=body, method="POST",
    )
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    points = data.get("result", {}).get("points", [])
    chunks = []
    for p in points:
        payload = p.get("payload") or {}
        page_num = payload.get("page_number")
        if page_num is None:
            meta = payload.get("metadata") or {}
            if isinstance(meta, dict):
                page_num = meta.get("page_number")
        chunks.append({
            "chunk_id": payload.get("chunk_id"),
            "doc_id": payload.get("doc_id"),
            "page_number": page_num,
            "content": payload.get("content") or "",
            "text_leg_score": float(p.get("score", 0.0)),
            "source": "visual_page_topN",
        })
    return chunks


# ---------------------------------------------------------------------------
# Bounded-join candidate set + rerank
# ---------------------------------------------------------------------------


def build_candidate_set(
    *,
    query: str,
    visual_top_pages: List[int],
    target_doc_id: str,
    text_top_n: int = DEFAULT_TEXT_TOP_N,
    visual_page_top_k: int = DEFAULT_VISUAL_PAGE_TOP_K,
    chunks_per_visual_page: int = DEFAULT_CHUNKS_PER_VISUAL_PAGE,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Charter §4.2 step 2 #8 candidate set:
        (text top-25) ∪ (top-3 chunks of each visual top-K page)
    Dedup by chunk_id. Returns (chunks, provenance_map).
    """
    text_chunks = run_text_top_n(query, top_n=text_top_n)

    qvec = embed_query_omlx(query)
    visual_chunks: List[Dict[str, Any]] = []
    for page in visual_top_pages[:visual_page_top_k]:
        page_chunks = chunks_for_doc_page(
            query_vector=qvec,
            doc_id=target_doc_id,
            page_number=page,
            top_k=chunks_per_visual_page,
        )
        visual_chunks.extend(page_chunks)

    # Dedup by chunk_id (Charter explicit) preserving first occurrence.
    seen = {}
    provenance = {}
    for c in text_chunks:
        cid = c["chunk_id"]
        if not cid:
            continue
        if cid not in seen:
            seen[cid] = c
            provenance[cid] = "text"
    for c in visual_chunks:
        cid = c["chunk_id"]
        if not cid:
            continue
        if cid not in seen:
            seen[cid] = c
            provenance[cid] = "visual_page"
        else:
            # Already present from text leg; mark provenance as both
            if provenance.get(cid) == "text":
                provenance[cid] = "both"

    return list(seen.values()), provenance


def rerank_candidate_set(
    *,
    query: str,
    candidates: List[Dict[str, Any]],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    from mmrag_v2.retrieval.reranker import LocalOmlxReranker
    reranker = LocalOmlxReranker()
    return reranker.rerank(query, candidates, top_n=top_n)


# ---------------------------------------------------------------------------
# PASS B aggregation
# ---------------------------------------------------------------------------


@dataclass
class PassBQueryResult:
    query_id: str
    query_text: str
    gold_page: int
    visual_top_pages: List[int]
    gold_in_visual_top5: bool
    candidate_set_size: int
    rerank_top1_chunk_id: Optional[str] = None
    rerank_top1_page: Optional[int] = None
    rerank_top1_doc_id: Optional[str] = None
    rerank_top1_provenance: Optional[str] = None  # text | visual_page | both
    rerank_top5: List[Dict[str, Any]] = field(default_factory=list)
    qualifies_for_pass_b: bool = False  # gold ∈ visual top-5 per Charter
    passes_pass_b: bool = False          # rerank top-1 is on gold page


@dataclass
class PassBReport:
    results: List[PassBQueryResult]
    target_doc_id: str

    @property
    def n_qualifying(self) -> int:
        return sum(1 for r in self.results if r.qualifies_for_pass_b)

    @property
    def n_passing(self) -> int:
        return sum(1 for r in self.results if r.passes_pass_b)

    @property
    def pass_b_rate(self) -> float:
        if self.n_qualifying == 0:
            return 0.0
        return self.n_passing / self.n_qualifying

    @property
    def pass_b_verdict(self) -> bool:
        return self.pass_b_rate >= CHARTER_PASS_B_THRESHOLD


def run_pass_b(
    *,
    run1_json: Path,
    target_doc_id: str,
) -> PassBReport:
    log = logging.getLogger("v3_c_spike_pass_b")
    payload = json.loads(run1_json.read_text(encoding="utf-8"))
    queries = payload["queries"]
    results: List[PassBQueryResult] = []

    for q in queries:
        visual_top_pages = [tup[0] for tup in q["visual_top_pages"]]
        gold = q["gold_page"]
        gold_in_visual_top5 = gold in visual_top_pages[:5]

        qresult = PassBQueryResult(
            query_id=q["query_id"],
            query_text=q["query_text"],
            gold_page=gold,
            visual_top_pages=visual_top_pages[:5],
            gold_in_visual_top5=gold_in_visual_top5,
            candidate_set_size=0,
            qualifies_for_pass_b=gold_in_visual_top5,
        )

        if not gold_in_visual_top5:
            log.info("Q%s gold=%d not in visual top-5 (%s) — does not qualify",
                     q["query_id"], gold, visual_top_pages[:5])
            results.append(qresult)
            continue

        candidates, provenance = build_candidate_set(
            query=q["query_text"],
            visual_top_pages=visual_top_pages,
            target_doc_id=target_doc_id,
        )
        qresult.candidate_set_size = len(candidates)

        if not candidates:
            log.warning("Q%s candidate set empty", q["query_id"])
            results.append(qresult)
            continue

        ranked = rerank_candidate_set(
            query=q["query_text"], candidates=candidates, top_n=5,
        )
        if ranked:
            top1 = ranked[0]
            qresult.rerank_top1_chunk_id = top1.get("chunk_id")
            qresult.rerank_top1_page = top1.get("page_number")
            qresult.rerank_top1_doc_id = top1.get("doc_id")
            qresult.rerank_top1_provenance = provenance.get(top1.get("chunk_id"))
            qresult.rerank_top5 = [
                {
                    "chunk_id": r.get("chunk_id"),
                    "doc_id": r.get("doc_id"),
                    "page_number": r.get("page_number"),
                    "rerank_score": float(r.get("rerank_score", 0.0)),
                    "provenance": provenance.get(r.get("chunk_id")),
                }
                for r in ranked
            ]
            qresult.passes_pass_b = (
                qresult.rerank_top1_doc_id == target_doc_id
                and qresult.rerank_top1_page == gold
            )

        log.info(
            "Q%s gold=%d visTop5=%s candidates=%d rerankTop1: page=%s doc=%s prov=%s -> %s",
            q["query_id"], gold, visual_top_pages[:5],
            qresult.candidate_set_size,
            qresult.rerank_top1_page,
            qresult.rerank_top1_doc_id, qresult.rerank_top1_provenance,
            "PASS" if qresult.passes_pass_b else "miss",
        )
        results.append(qresult)

    return PassBReport(results=results, target_doc_id=target_doc_id)


def _format(report: PassBReport) -> str:
    lines = [
        f"V3 Phase C C-Spike PASS B — {report.target_doc_id}",
        "",
        f"  {'ID':<5} {'gold':>4} {'vTop5':<20} {'cand':>5} {'rTop1pg':>7} {'rTop1doc':<14} {'prov':<11} verdict",
    ]
    for r in report.results:
        verdict = (
            "PASS" if r.passes_pass_b
            else ("miss" if r.qualifies_for_pass_b else "n/a")
        )
        vis5 = ",".join(str(p) for p in r.visual_top_pages)
        prov = r.rerank_top1_provenance or "-"
        rtop1pg = r.rerank_top1_page if r.rerank_top1_page is not None else "-"
        rtop1doc = (r.rerank_top1_doc_id or "-")[:14]
        lines.append(
            f"  {r.query_id:<5} {r.gold_page:>4} {vis5:<20} {r.candidate_set_size:>5} "
            f"{str(rtop1pg):>7} {rtop1doc:<14} {prov:<11} {verdict}"
        )
    lines += [
        "",
        f"Aggregate:",
        f"  Qualifying queries (gold in visual top-5): {report.n_qualifying}/{len(report.results)}",
        f"  Passing (rerank top-1 on gold page):       {report.n_passing}/{report.n_qualifying}",
        f"  PASS B rate:                                {report.pass_b_rate:.2%}",
        f"  Threshold:                                  ≥{CHARTER_PASS_B_THRESHOLD:.0%}",
        f"  Charter §4.2 step 2 PASS B verdict:         {'PASS' if report.pass_b_verdict else 'FAIL'}",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "V3 Phase C C-spike PASS B — bounded-join candidate set + "
            "rerank top-1 discrimination on visually-retrieved pages."
        )
    )
    parser.add_argument("--run1-json", type=Path,
                        default=Path("docs/V3_C_SPIKE_RUN1.json"),
                        help="Path to the v3_c_spike.py run-1 JSON output")
    parser.add_argument("--doc-id", type=str, default="6fccda8bd625",
                        help="ATZ_Elektronik_German doc_id")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s | %(message)s",
    )

    if not args.run1_json.exists():
        parser.error(f"run-1 JSON not found: {args.run1_json}")

    report = run_pass_b(
        run1_json=args.run1_json,
        target_doc_id=args.doc_id,
    )
    print(_format(report))

    if args.json_out:
        out_payload = {
            "target_doc_id": report.target_doc_id,
            "n_qualifying": report.n_qualifying,
            "n_passing": report.n_passing,
            "pass_b_rate": report.pass_b_rate,
            "pass_b_verdict": report.pass_b_verdict,
            "threshold": CHARTER_PASS_B_THRESHOLD,
            "results": [
                {
                    "query_id": r.query_id,
                    "query_text": r.query_text,
                    "gold_page": r.gold_page,
                    "visual_top_pages": r.visual_top_pages,
                    "gold_in_visual_top5": r.gold_in_visual_top5,
                    "qualifies_for_pass_b": r.qualifies_for_pass_b,
                    "candidate_set_size": r.candidate_set_size,
                    "rerank_top1_chunk_id": r.rerank_top1_chunk_id,
                    "rerank_top1_page": r.rerank_top1_page,
                    "rerank_top1_doc_id": r.rerank_top1_doc_id,
                    "rerank_top1_provenance": r.rerank_top1_provenance,
                    "rerank_top5": r.rerank_top5,
                    "passes_pass_b": r.passes_pass_b,
                }
                for r in report.results
            ],
        }
        args.json_out.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
        print(f"\nJSON written: {args.json_out}")
    return 0 if report.pass_b_verdict else 1


if __name__ == "__main__":
    sys.exit(main())
