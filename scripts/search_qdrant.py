#!/usr/bin/env python3
"""
Search Qdrant collections using nomic-embed-text via Ollama.

Usage:
    python3 scripts/search_qdrant.py "how to build walls with tires"
    python3 scripts/search_qdrant.py "dragon hatching" -c harrypotter_and_the_sorcerers_stone_pdf
    python3 scripts/search_qdrant.py "solar heating" -n 10
    python3 scripts/search_qdrant.py "greenhouse" --modality image
    python3 scripts/search_qdrant.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import urllib.request

# ── ANSI colors ─────────────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RESET = "\033[0m"
BLUE = "\033[34m"

MIN_SCORE = 0.55  # Don't show results below this relevance

# Reranker config (Alibaba Dashscope)
RERANK_MODEL = "qwen3-rerank"
RERANK_API_KEY = "sk-5813a0a803ca4b96ab8755b1068f10fd"
RERANK_URL = "https://dashscope-intl.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"


# ── API helpers ──────────────────────────────────────────────────────────────

def embed(text: str, model: str = "nomic-embed-text", ollama_url: str = "http://localhost:11434") -> list[float]:
    data = json.dumps({"model": model, "input": text}).encode()
    req = urllib.request.Request(f"{ollama_url}/api/embed", data=data)
    req.add_header("Content-Type", "application/json")
    return json.loads(urllib.request.urlopen(req, timeout=30).read())["embeddings"][0]


def list_collections(qdrant_url: str = "http://localhost:6333") -> list[dict]:
    resp = urllib.request.urlopen(f"{qdrant_url}/collections", timeout=10)
    data = json.loads(resp.read())
    collections = []
    for c in data["result"]["collections"]:
        name = c["name"]
        info = json.loads(urllib.request.urlopen(f"{qdrant_url}/collections/{name}", timeout=10).read())
        collections.append({"name": name, "points": info["result"]["points_count"]})
    return collections


def search(query_vector: list[float], collection: str, limit: int = 5,
           modality: str | None = None, keyword: str | None = None,
           qdrant_url: str = "http://localhost:6333") -> list[dict]:
    body: dict = {"vector": query_vector, "limit": limit, "with_payload": True}
    must_filters = []
    if modality:
        must_filters.append({"key": "modality", "match": {"value": modality}})
    if keyword:
        # For short queries, require the keyword to appear in content
        must_filters.append({"key": "content", "match": {"text": keyword}})
    if must_filters:
        body["filter"] = {"must": must_filters}
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{qdrant_url}/collections/{collection}/points/search", data=data)
    req.add_header("Content-Type", "application/json")
    return json.loads(urllib.request.urlopen(req, timeout=30).read())["result"]


def rerank(query: str, results: list[dict], top_n: int = 5) -> list[dict]:
    """Rerank search results using qwen3-rerank for semantic precision."""
    if not results:
        return results

    documents = []
    for r in results:
        p = r["payload"]
        text = p.get("content", "")
        if p.get("modality") == "image":
            text = p.get("visual_description", "") or text
        documents.append(text[:500])  # Cap length for API

    body = {
        "model": RERANK_MODEL,
        "input": {
            "query": query,
            "documents": documents,
        },
        "parameters": {"top_n": min(top_n, len(documents))},
    }

    try:
        data = json.dumps(body).encode()
        req = urllib.request.Request(RERANK_URL, data=data)
        req.add_header("Authorization", f"Bearer {RERANK_API_KEY}")
        req.add_header("Content-Type", "application/json")
        resp = urllib.request.urlopen(req, timeout=30)
        rr = json.loads(resp.read())

        if "output" not in rr or "results" not in rr["output"]:
            return results[:top_n]

        reranked = []
        for item in rr["output"]["results"]:
            idx = item["index"]
            r = results[idx].copy()
            r["score"] = item["relevance_score"]
            r["reranked"] = True
            reranked.append(r)
        return reranked

    except Exception as e:
        print(f"  {DIM}Reranker unavailable ({e}), using vector scores{RESET}", file=sys.stderr)
        return results[:top_n]


# ── Display ──────────────────────────────────────────────────────────────────

def format_collection_name(name: str) -> str:
    """Make collection names readable."""
    return name.replace("_pdf", "").replace("_", " ").title()


def format_content(text: str, width: int = 80) -> str:
    """Clean and wrap content for display."""
    # Collapse whitespace but preserve paragraph breaks
    lines = text.strip().split("\n")
    cleaned = " ".join(l.strip() for l in lines if l.strip())
    return textwrap.fill(cleaned, width=width, initial_indent="    ", subsequent_indent="    ")


def score_bar(score: float, width: int = 20) -> str:
    """Visual relevance bar."""
    filled = int(score * width)
    bar = "█" * filled + "░" * (width - filled)
    if score >= 0.75:
        color = GREEN
    elif score >= 0.65:
        color = YELLOW
    else:
        color = DIM
    return f"{color}{bar}{RESET} {score:.0%}"


def display_result(r: dict, rank: int):
    """Display a single search result."""
    p = r["payload"]
    score = r["score"]
    mod = p.get("modality", "text")
    pg = p.get("page_number", "?")
    source = p.get("source_file", "")
    heading = p.get("parent_heading", "")

    # Score bar
    print(f"\n  {score_bar(score)}")

    # Header line
    if mod == "image":
        label = f"{MAGENTA}IMAGE{RESET}"
    elif mod == "table":
        label = f"{BLUE}TABLE{RESET}"
    else:
        label = f"{CYAN}TEXT{RESET}"

    location = f"page {pg}"
    if heading:
        location += f"  {DIM}({heading}){RESET}"

    print(f"  {label}  {location}")

    # Content
    if mod == "image":
        desc = p.get("visual_description", "") or p.get("content", "")
        print(format_content(desc))
        asset = p.get("asset_path", "")
        if asset:
            print(f"    {DIM}Asset: {asset}{RESET}")
    else:
        content = p.get("content", "")
        print(format_content(content[:300]))


def main():
    parser = argparse.ArgumentParser(
        description="Search your document collections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              search_qdrant.py "how to build walls with tires"
              search_qdrant.py "dragon hatching" -n 3
              search_qdrant.py "tire construction" --modality image
              search_qdrant.py --list
        """),
    )
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("-c", "--collection", default=None, help="Specific collection")
    parser.add_argument("-n", "--limit", type=int, default=5, help="Max results shown per collection")
    parser.add_argument("-m", "--modality", choices=["text", "image", "table"], help="Filter by type")
    parser.add_argument("-l", "--list", action="store_true", help="List collections")
    parser.add_argument("--no-rerank", action="store_true", help="Skip reranking (vector scores only)")
    parser.add_argument("--min-score", type=float, default=MIN_SCORE, help="Minimum relevance")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    args = parser.parse_args()

    if args.list or not args.query:
        collections = list_collections(args.qdrant_url)
        print(f"\n{BOLD}Collections:{RESET}")
        for c in collections:
            name = format_collection_name(c["name"])
            print(f"  {GREEN}{name}{RESET}  ({c['points']:,} chunks)")
        print()
        if not args.query:
            return 0

    query = args.query

    # Determine targets
    if args.collection:
        targets = [args.collection]
    else:
        targets = [c["name"] for c in list_collections(args.qdrant_url)]

    vector = embed(query, ollama_url=args.ollama_url)

    use_rerank = not args.no_rerank

    print(f"\n{BOLD}Searching:{RESET} {query}")
    if args.modality:
        print(f"{DIM}Filter: {args.modality} only{RESET}")
    if use_rerank:
        print(f"{DIM}Reranking: qwen3-rerank{RESET}")

    any_results = False

    for collection in targets:
        # Retrieve wide for reranking (4x the requested limit)
        retrieve_limit = args.limit * 8 if use_rerank else args.limit
        # Extract the most distinctive keyword for filtering.
        # Pick proper nouns and technical terms over common words.
        import re as _re
        _COMMON = {"the","a","an","of","to","in","for","and","or","is","it","on","by",
                    "how","what","why","can","do","this","that","with","was","were","are",
                    "been","being","has","had","have","not","but","from","they","them",
                    "would","could","should","will","shall","may","might","did","does"}
        _clean_words = [_re.sub(r"[^a-zA-Z0-9._-]", "", w) for w in query.strip().split()]
        _clean_words = [w for w in _clean_words if w.lower() not in _COMMON and len(w) > 2]
        # Prefer capitalized words (proper nouns, model names) over common words
        _proper = [w for w in _clean_words if w[0].isupper()] if _clean_words else []
        keyword = max(_proper, key=len) if _proper else (max(_clean_words, key=len) if _clean_words else None)
        results = search(vector, collection, retrieve_limit, args.modality, keyword, args.qdrant_url)
        # If keyword filter returns nothing, retry without it
        if not results and keyword:
            results = search(vector, collection, retrieve_limit, args.modality, None, args.qdrant_url)
        # Filter by minimum vector score
        results = [r for r in results if r["score"] >= args.min_score]
        if not results:
            continue

        # Rerank for semantic precision
        if use_rerank and len(results) > 1:
            results = rerank(query, results, top_n=args.limit)
        else:
            results = results[:args.limit]

        if not results:
            continue

        any_results = True
        name = format_collection_name(collection)
        print(f"\n{BOLD}{'━' * 60}{RESET}")
        print(f"{BOLD}{GREEN}{name}{RESET}")

        for i, r in enumerate(results):
            display_result(r, i + 1)

    if not any_results:
        print(f"\n{DIM}No results above {args.min_score:.0%} relevance.{RESET}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
