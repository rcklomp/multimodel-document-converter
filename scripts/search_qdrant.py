#!/usr/bin/env python3
"""
Search and inspect Qdrant collections — conversion quality testing tool.

Usage:
    python3 scripts/search_qdrant.py "what is MCP"
    python3 scripts/search_qdrant.py "what is MCP" -c sekar -n 3
    python3 scripts/search_qdrant.py "what is MCP" --json           # raw JSON for Gemini audit
    python3 scripts/search_qdrant.py --page 60 -c sekar             # browse page
    python3 scripts/search_qdrant.py --stats -c sekar               # collection quality stats
    python3 scripts/search_qdrant.py --stats                        # all collections
    python3 scripts/search_qdrant.py --list                         # list collections
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import urllib.request
from collections import Counter

# ── ANSI colors ─────────────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RED = "\033[31m"
RESET = "\033[0m"
BLUE = "\033[34m"

MIN_SCORE = 0.55

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


def find_collection(name_fragment: str, qdrant_url: str) -> str | None:
    """Find collection by partial name match."""
    collections = list_collections(qdrant_url)
    for c in collections:
        if name_fragment.lower() in c["name"].lower():
            return c["name"]
    return None


def search(query_vector: list[float], collection: str, limit: int = 5,
           modality: str | None = None, keyword: str | None = None,
           qdrant_url: str = "http://localhost:6333") -> list[dict]:
    body: dict = {"vector": query_vector, "limit": limit, "with_payload": True}
    must_filters = []
    if modality:
        must_filters.append({"key": "modality", "match": {"value": modality}})
    if keyword:
        must_filters.append({"key": "content", "match": {"text": keyword}})
    if must_filters:
        body["filter"] = {"must": must_filters}
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{qdrant_url}/collections/{collection}/points/search", data=data)
    req.add_header("Content-Type", "application/json")
    return json.loads(urllib.request.urlopen(req, timeout=30).read())["result"]


def scroll_by_page(collection: str, page: int, qdrant_url: str = "http://localhost:6333") -> list[dict]:
    """Get all chunks from a specific page."""
    body = {
        "filter": {"must": [{"key": "page_number", "match": {"value": page}}]},
        "limit": 50,
        "with_payload": True,
        "with_vector": False,
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{qdrant_url}/collections/{collection}/points/scroll", data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    result = json.loads(urllib.request.urlopen(req, timeout=30).read())
    return result.get("result", {}).get("points", [])


def scroll_all(collection: str, qdrant_url: str = "http://localhost:6333",
               limit: int = 100, offset: int | None = None) -> tuple[list[dict], int | None]:
    """Scroll through all points in a collection."""
    body: dict = {"limit": limit, "with_payload": True, "with_vector": False}
    if offset is not None:
        body["offset"] = offset
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{qdrant_url}/collections/{collection}/points/scroll", data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    result = json.loads(urllib.request.urlopen(req, timeout=30).read())["result"]
    next_offset = result.get("next_page_offset")
    return result.get("points", []), next_offset


def rerank(query: str, results: list[dict], top_n: int = 5) -> list[dict]:
    """Rerank using qwen3-rerank."""
    if not results:
        return results
    documents = []
    for r in results:
        p = r["payload"]
        text = p.get("content", "")
        if p.get("modality") == "image":
            text = p.get("visual_description", "") or text
        documents.append(text[:500])

    body = {
        "model": RERANK_MODEL,
        "input": {"query": query, "documents": documents},
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
            r = results[item["index"]].copy()
            r["score"] = item["relevance_score"]
            r["reranked"] = True
            reranked.append(r)
        return reranked
    except Exception as e:
        print(f"  {DIM}Reranker unavailable ({e}), using vector scores{RESET}", file=sys.stderr)
        return results[:top_n]


# ── Display ──────────────────────────────────────────────────────────────────

def format_collection_name(name: str) -> str:
    return name.replace("_pdf", "").replace("_", " ").title()


def score_bar(score: float, width: int = 20) -> str:
    filled = int(score * width)
    bar = "█" * filled + "░" * (width - filled)
    color = GREEN if score >= 0.75 else YELLOW if score >= 0.65 else DIM
    return f"{color}{bar}{RESET} {score:.0%}"


def display_result(r: dict, rank: int, verbose: bool = False):
    """Display a search result with metadata."""
    p = r["payload"]
    score = r["score"]
    mod = p.get("modality", "text")
    pg = p.get("page_number", "?")
    heading = p.get("parent_heading", "")
    breadcrumb = p.get("breadcrumb", "")
    chunk_type = p.get("chunk_type", "")
    source = p.get("source_file", "")

    # Modality label
    labels = {"image": f"{MAGENTA}IMAGE{RESET}", "table": f"{BLUE}TABLE{RESET}"}
    label = labels.get(mod, f"{CYAN}TEXT{RESET}")

    # Score + modality + location
    print(f"\n  {score_bar(score)}")
    location = f"pg {pg}"
    if chunk_type:
        location += f"  {DIM}[{chunk_type}]{RESET}"
    print(f"  {label}  {location}")

    # Heading / breadcrumb
    if heading:
        print(f"  {YELLOW}Section:{RESET} {heading}")
    if verbose and breadcrumb:
        print(f"  {DIM}Path: {breadcrumb}{RESET}")

    # Content
    if mod == "image":
        desc = p.get("visual_description", "") or p.get("content", "")
        print(textwrap.fill(desc[:300], width=80, initial_indent="    ", subsequent_indent="    "))
        asset = p.get("asset_path", "")
        if asset:
            print(f"    {DIM}Asset: {asset}{RESET}")
    else:
        content = p.get("content", "")
        print(textwrap.fill(content[:400], width=80, initial_indent="    ", subsequent_indent="    "))


def display_page(points: list[dict], page: int, collection: str):
    """Display all chunks from a page."""
    if not points:
        print(f"\n  {DIM}No chunks on page {page}{RESET}")
        return

    name = format_collection_name(collection)
    print(f"\n{BOLD}{'━' * 60}{RESET}")
    print(f"{BOLD}{GREEN}{name}{RESET} — Page {page}")
    print(f"{DIM}{len(points)} chunks{RESET}")

    # Sort by modality (text first, then table, then image)
    order = {"text": 0, "table": 1, "image": 2}
    points.sort(key=lambda p: order.get(p["payload"].get("modality", "text"), 9))

    for i, pt in enumerate(points):
        p = pt["payload"]
        mod = p.get("modality", "text")
        heading = p.get("parent_heading", "")
        chunk_type = p.get("chunk_type", "")

        labels = {"image": f"{MAGENTA}IMAGE{RESET}", "table": f"{BLUE}TABLE{RESET}"}
        label = labels.get(mod, f"{CYAN}TEXT{RESET}")

        type_info = f"  {DIM}[{chunk_type}]{RESET}" if chunk_type else ""
        print(f"\n  {label}{type_info}")
        if heading:
            print(f"  {YELLOW}Section:{RESET} {heading}")

        if mod == "image":
            desc = p.get("visual_description", "") or p.get("content", "")
            print(textwrap.fill(desc[:200], width=80, initial_indent="    ", subsequent_indent="    "))
            asset = p.get("asset_path", "")
            if asset:
                print(f"    {DIM}Asset: {asset}{RESET}")
        else:
            content = p.get("content", "")
            print(textwrap.fill(content[:300], width=80, initial_indent="    ", subsequent_indent="    "))


def display_stats(collection_name: str, qdrant_url: str):
    """Show conversion quality statistics for a collection."""
    name = format_collection_name(collection_name)
    print(f"\n{BOLD}{'━' * 60}{RESET}")
    print(f"{BOLD}{GREEN}{name}{RESET}")

    # Scroll all points to compute stats
    all_points = []
    offset = None
    while True:
        points, next_offset = scroll_all(collection_name, qdrant_url, limit=100, offset=offset)
        all_points.extend(points)
        if next_offset is None:
            break
        offset = next_offset

    total = len(all_points)
    print(f"  Total chunks: {total}")

    # Modality distribution
    modalities = Counter()
    chunk_types = Counter()
    pages = set()
    headings = Counter()
    null_headings = 0
    content_lengths = []
    extraction_methods = Counter()

    for pt in all_points:
        p = pt["payload"]
        mod = p.get("modality", "?")
        modalities[mod] += 1
        ct = p.get("chunk_type", "")
        if ct:
            chunk_types[ct] += 1
        pg = p.get("page_number", 0)
        if pg:
            pages.add(pg)
        h = p.get("parent_heading")
        if h:
            headings[h] += 1
        else:
            null_headings += 1
        content = p.get("content", "")
        if content:
            content_lengths.append(len(content))

    # Modality
    print(f"\n  {BOLD}Modality:{RESET}")
    for mod, count in modalities.most_common():
        pct = 100 * count / total
        print(f"    {mod:8s}  {count:5d}  ({pct:.0f}%)")

    # Chunk types
    if chunk_types:
        print(f"\n  {BOLD}Chunk types:{RESET}")
        for ct, count in chunk_types.most_common(8):
            print(f"    {ct:12s}  {count:5d}")

    # Page coverage
    if pages:
        print(f"\n  {BOLD}Pages:{RESET} {min(pages)}-{max(pages)} ({len(pages)} unique)")

    # Heading quality
    total_text = modalities.get("text", 0)
    headed = total_text - null_headings
    pct_headed = 100 * headed / total_text if total_text else 0
    print(f"\n  {BOLD}Heading coverage:{RESET}")
    if pct_headed >= 90:
        color = GREEN
    elif pct_headed >= 70:
        color = YELLOW
    else:
        color = RED
    print(f"    Text with heading: {color}{headed}/{total_text} ({pct_headed:.0f}%){RESET}")
    if null_headings:
        print(f"    Null headings: {null_headings}")

    # Top headings
    if headings:
        print(f"\n  {BOLD}Top sections:{RESET}")
        for h, count in headings.most_common(10):
            print(f"    {DIM}{count:3d}x{RESET}  {h[:70]}")

    # Content size distribution
    if content_lengths:
        import statistics
        med = statistics.median(content_lengths)
        avg = statistics.mean(content_lengths)
        short = sum(1 for l in content_lengths if l < 30)
        long_ = sum(1 for l in content_lengths if l > 1500)
        print(f"\n  {BOLD}Content size:{RESET}")
        print(f"    Median: {med:.0f} chars, Mean: {avg:.0f} chars")
        if short:
            print(f"    {YELLOW}Micro (<30): {short}{RESET}")
        if long_:
            print(f"    {YELLOW}Oversize (>1500): {long_}{RESET}")

    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Search and inspect Qdrant document collections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              search_qdrant.py "what is MCP"                    # search all
              search_qdrant.py "MCP server" -c sekar -n 3       # search one collection
              search_qdrant.py "MCP server" --json              # JSON output for audit
              search_qdrant.py --page 60 -c sekar               # browse page 60
              search_qdrant.py --stats -c sekar                 # quality stats
              search_qdrant.py --stats                          # stats for all
              search_qdrant.py --list                           # list collections
        """),
    )
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("-c", "--collection", default=None, help="Collection (partial name match)")
    parser.add_argument("-n", "--limit", type=int, default=5, help="Max results per collection")
    parser.add_argument("-m", "--modality", choices=["text", "image", "table"], help="Filter by type")
    parser.add_argument("-l", "--list", action="store_true", help="List collections")
    parser.add_argument("-p", "--page", type=int, help="Browse chunks on a specific page")
    parser.add_argument("-s", "--stats", action="store_true", help="Show collection quality stats")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show breadcrumbs and extra metadata")
    parser.add_argument("--json", action="store_true", help="Output raw JSON (for Gemini/ChatGPT audit)")
    parser.add_argument("--no-rerank", action="store_true", help="Skip reranking")
    parser.add_argument("--min-score", type=float, default=MIN_SCORE, help="Minimum relevance")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    args = parser.parse_args()

    # Resolve collection name from partial match
    resolved_collection = None
    if args.collection:
        resolved_collection = find_collection(args.collection, args.qdrant_url)
        if not resolved_collection:
            print(f"{RED}No collection matching '{args.collection}'{RESET}", file=sys.stderr)
            return 1

    # --list
    if args.list or (not args.query and not args.page and not args.stats):
        collections = list_collections(args.qdrant_url)
        print(f"\n{BOLD}Collections:{RESET}")
        for c in sorted(collections, key=lambda x: x["name"]):
            name = format_collection_name(c["name"])
            print(f"  {GREEN}{name:55s}{RESET}  {c['points']:>5,} chunks")
        print(f"\n  {DIM}Total: {sum(c['points'] for c in collections):,} points across {len(collections)} collections{RESET}\n")
        if not args.query and not args.page and not args.stats:
            return 0

    # --stats
    if args.stats:
        if resolved_collection:
            display_stats(resolved_collection, args.qdrant_url)
        else:
            for c in list_collections(args.qdrant_url):
                display_stats(c["name"], args.qdrant_url)
        return 0

    # --page
    if args.page is not None:
        if not resolved_collection:
            print(f"{RED}--page requires -c <collection>{RESET}", file=sys.stderr)
            return 1
        points = scroll_by_page(resolved_collection, args.page, args.qdrant_url)
        if args.json:
            for pt in points:
                print(json.dumps(pt["payload"], indent=2, ensure_ascii=False))
        else:
            display_page(points, args.page, resolved_collection)
        print()
        return 0

    # Search
    if not args.query:
        return 0

    query = args.query
    targets = [resolved_collection] if resolved_collection else [c["name"] for c in list_collections(args.qdrant_url)]

    vector = embed(query, ollama_url=args.ollama_url)
    use_rerank = not args.no_rerank

    if not args.json:
        print(f"\n{BOLD}Searching:{RESET} {query}")
        if args.modality:
            print(f"{DIM}Filter: {args.modality} only{RESET}")
        if use_rerank:
            print(f"{DIM}Reranking: qwen3-rerank{RESET}")

    any_results = False

    for collection in targets:
        retrieve_limit = args.limit * 8 if use_rerank else args.limit

        # Keyword extraction for filtering
        import re as _re
        _COMMON = {"the","a","an","of","to","in","for","and","or","is","it","on","by",
                    "how","what","why","can","do","this","that","with","was","were","are",
                    "been","being","has","had","have","not","but","from","they","them",
                    "would","could","should","will","shall","may","might","did","does"}
        _words = [_re.sub(r"[^a-zA-Z0-9._-]", "", w) for w in query.strip().split()]
        _words = [w for w in _words if w.lower() not in _COMMON and len(w) > 2]
        _proper = [w for w in _words if w[0].isupper()] if _words else []
        keyword = max(_proper, key=len) if _proper else (max(_words, key=len) if _words else None)

        results = search(vector, collection, retrieve_limit, args.modality, keyword, args.qdrant_url)
        if not results and keyword:
            results = search(vector, collection, retrieve_limit, args.modality, None, args.qdrant_url)

        results = [r for r in results if r["score"] >= args.min_score]
        if not results:
            continue

        if use_rerank and len(results) > 1:
            results = rerank(query, results, top_n=args.limit)
        else:
            results = results[:args.limit]

        if not results:
            continue

        any_results = True

        if args.json:
            for r in results:
                out = {"score": r["score"], "collection": collection, **r["payload"]}
                print(json.dumps(out, indent=2, ensure_ascii=False))
        else:
            name = format_collection_name(collection)
            print(f"\n{BOLD}{'━' * 60}{RESET}")
            print(f"{BOLD}{GREEN}{name}{RESET}")
            for i, r in enumerate(results):
                display_result(r, i + 1, verbose=args.verbose)

    if not any_results and not args.json:
        print(f"\n{DIM}No results above {args.min_score:.0%} relevance.{RESET}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
