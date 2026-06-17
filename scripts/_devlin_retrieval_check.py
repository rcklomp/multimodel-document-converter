"""Ad-hoc DoD check: do code queries about Devlin return the right chunk in top-10?

Uses the production retrieval path (omlx dense embed + ModernBERT rerank, hnsw_ef
tuned) against the isolated test collection built from the Fix-b conversion. Gold
chunk_ids were identified by distinctive correctly-indented code in the output.
Not a committed test; a one-off validation harness.
"""
from __future__ import annotations

import sys

from mmrag_v2.retrieval import get_reranker, retrieve_reranked

COLLECTION = "mmrag_v3__qwen3_local__devlin_fixb"

# (query, substring that MUST appear in the gold chunk, human label)
CASES = [
    ("How does the code call the OpenAI chat completions endpoint with httpx and raise for status?",
     "raise_for_status", "httpx OpenAI client (page 50, repaired)"),
    ("Python tool that fetches a Wikipedia page summary over HTTP using httpx",
     "WIKI_SUMMARY", "Wikipedia summary tool (page 50)"),
    ("FastAPI server that exposes the QA agent as a web endpoint",
     "FastAPI", "FastAPI serve.py (page 54)"),
    ("command line interface that reads the question from argv and prints the answer",
     "sys.argv", "CLI main (page 54)"),
    ("function llm_complete that sends messages to gpt-4o-mini",
     "llm_complete", "llm_complete (page 60)"),
]


def _chunk_id(hit: dict) -> str:
    p = hit.get("payload") or {}
    return str(p.get("chunk_id") or p.get("id") or "")


def _content(hit: dict) -> str:
    p = hit.get("payload") or {}
    return p.get("content") or p.get("text") or ""


def main() -> int:
    rr = get_reranker("omlx")
    passed = 0
    for query, needle, label in CASES:
        hits = retrieve_reranked(
            query, collection=COLLECTION, top_k_retrieve=50, top_n_return=10,
            embed_provider="omlx", reranker=rr, hnsw_ef=512,
        )
        rank = None
        for i, h in enumerate(hits, 1):
            if needle in _content(h):
                rank = i
                break
        ok = rank is not None and rank <= 10
        passed += ok
        flag = "PASS" if ok else "FAIL"
        print(f"[{flag}] {label}")
        print(f"       query: {query}")
        print(f"       needle {needle!r} found at rank: {rank} (of {len(hits)} returned)")
        if hits:
            top = _content(hits[0])
            print(f"       top-1 chunk: {' '.join(top.split())[:110]}")
        print()
    print(f"RETRIEVAL_TOP10: {passed}/{len(CASES)} code queries returned the gold chunk in top-10")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
