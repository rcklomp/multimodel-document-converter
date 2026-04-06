#!/usr/bin/env python3
"""
Validate ingested Qdrant collections for RAG readiness.

Runs automated probe queries against each collection and reports quality.

Usage:
    python3 scripts/validate_qdrant.py
    python3 scripts/validate_qdrant.py -c firearms_pdf
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"

QDRANT = "http://localhost:6333"
OLLAMA = "http://localhost:11434"
RERANK_URL = "https://dashscope-intl.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
RERANK_KEY = "sk-5813a0a803ca4b96ab8755b1068f10fd"


def embed(text: str) -> list[float]:
    data = json.dumps({"model": "nomic-embed-text", "input": text}).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/embed", data=data)
    req.add_header("Content-Type", "application/json")
    return json.loads(urllib.request.urlopen(req, timeout=30).read())["embeddings"][0]


def search(vector: list[float], collection: str, limit: int = 10,
           keyword: str | None = None) -> list[dict]:
    body: dict = {"vector": vector, "limit": limit, "with_payload": True}
    if keyword:
        body["filter"] = {"must": [{"key": "content", "match": {"text": keyword}}]}
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{QDRANT}/collections/{collection}/points/search", data=data)
    req.add_header("Content-Type", "application/json")
    return json.loads(urllib.request.urlopen(req, timeout=30).read())["result"]


def rerank(query: str, docs: list[str], top_n: int = 3) -> list[dict]:
    try:
        body = {"model": "qwen3-rerank", "input": {"query": query, "documents": docs},
                "parameters": {"top_n": top_n}}
        data = json.dumps(body).encode()
        req = urllib.request.Request(RERANK_URL, data=data)
        req.add_header("Authorization", f"Bearer {RERANK_KEY}")
        req.add_header("Content-Type", "application/json")
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read())["output"]["results"]
    except Exception:
        return []


def get_collections() -> list[dict]:
    resp = urllib.request.urlopen(f"{QDRANT}/collections", timeout=10)
    data = json.loads(resp.read())
    result = []
    for c in data["result"]["collections"]:
        name = c["name"]
        info = json.loads(urllib.request.urlopen(f"{QDRANT}/collections/{name}", timeout=10).read())
        result.append({"name": name, "points": info["result"]["points_count"]})
    return result


def get_collection_stats(collection: str) -> dict:
    """Sample chunks to build statistics and auto-generate probe queries."""
    # Get a random sample of chunks
    body = {"limit": 100, "with_payload": True, "offset": 0}
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{QDRANT}/collections/{collection}/points/scroll", data=data)
    req.add_header("Content-Type", "application/json")
    resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
    points = resp["result"]["points"]

    stats = {
        "total": 0, "text": 0, "image": 0, "table": 0,
        "headings": set(), "pages": set(),
        "empty_content": 0, "short_content": 0,
        "has_refined": 0, "has_visual_desc": 0,
        "avg_content_len": 0,
    }

    content_lens = []
    for p in points:
        payload = p.get("payload", {})
        mod = payload.get("modality", "text")
        stats["total"] += 1
        stats[mod] = stats.get(mod, 0) + 1

        content = payload.get("content", "")
        content_lens.append(len(content))
        if not content.strip():
            stats["empty_content"] += 1
        elif len(content.strip()) < 20:
            stats["short_content"] += 1

        pg = payload.get("page_number")
        if pg:
            stats["pages"].add(pg)

        heading = payload.get("parent_heading", "")
        if heading and heading not in ("", "null"):
            stats["headings"].add(heading)

        if payload.get("visual_description"):
            stats["has_visual_desc"] += 1

    stats["avg_content_len"] = sum(content_lens) / max(len(content_lens), 1)
    stats["page_range"] = f"{min(stats['pages'])}-{max(stats['pages'])}" if stats["pages"] else "?"
    return stats


def generate_probes(stats: dict, collection: str) -> list[dict]:
    """Generate probe queries from collection metadata."""
    probes = []

    # Probe 1: search by heading names
    for heading in sorted(stats["headings"])[:5]:
        if len(heading) > 3:
            probes.append({
                "query": heading,
                "expect": f"heading '{heading}' in results",
                "keyword": heading.split()[0] if heading.split() else None,
            })

    # Probe 2: search for first and last page content
    if stats["pages"]:
        probes.append({
            "query": f"page 1 introduction",
            "expect": "page 1 or 2 in results",
            "keyword": None,
        })

    return probes


def run_probe(query: str, keyword: str | None, collection: str) -> dict:
    """Run a single probe query and return quality metrics."""
    vector = embed(query)
    results = search(vector, collection, limit=20, keyword=keyword)
    if not results and keyword:
        results = search(vector, collection, limit=20)

    if not results:
        return {"score": 0.0, "found": False, "top_content": ""}

    # Rerank
    docs = [r["payload"].get("content", "")[:500] for r in results[:10]]
    reranked = rerank(query, docs, top_n=3)

    if reranked:
        best_idx = reranked[0]["index"]
        best_score = reranked[0]["relevance_score"]
        best_content = docs[best_idx][:100]
    else:
        best_score = results[0]["score"]
        best_content = results[0]["payload"].get("content", "")[:100]

    return {
        "score": best_score,
        "found": best_score > 0.5,
        "top_content": best_content,
    }


def validate_collection(name: str) -> dict:
    """Run full validation on a collection."""
    stats = get_collection_stats(name)
    probes = generate_probes(stats, name)

    results = []
    for probe in probes:
        result = run_probe(probe["query"], probe.get("keyword"), name)
        result["query"] = probe["query"]
        result["expect"] = probe["expect"]
        results.append(result)

    return {"stats": stats, "probes": results}


def print_report(name: str, validation: dict):
    """Print formatted validation report."""
    stats = validation["stats"]
    probes = validation["probes"]

    display_name = name.replace("_pdf", "").replace("_", " ").title()
    print(f"\n{BOLD}{'━' * 60}{RESET}")
    print(f"{BOLD}{GREEN}{display_name}{RESET}")
    print(f"{BOLD}{'━' * 60}{RESET}")

    # Stats
    print(f"\n  {CYAN}Chunks:{RESET}  {stats['total']} total "
          f"({stats['text']} text, {stats['image']} image, {stats.get('table', 0)} table)")
    print(f"  {CYAN}Pages:{RESET}   {stats['page_range']}")
    print(f"  {CYAN}Avg len:{RESET} {stats['avg_content_len']:.0f} chars")
    print(f"  {CYAN}Sections:{RESET} {len(stats['headings'])} unique headings")

    # Quality flags
    issues = []
    if stats["empty_content"] > 0:
        issues.append(f"{RED}empty chunks: {stats['empty_content']}{RESET}")
    if stats["short_content"] > stats["total"] * 0.2:
        issues.append(f"{YELLOW}many short chunks: {stats['short_content']}{RESET}")
    if stats["avg_content_len"] < 50:
        issues.append(f"{RED}very short avg content: {stats['avg_content_len']:.0f}{RESET}")
    if not stats["headings"]:
        issues.append(f"{YELLOW}no section headings detected{RESET}")

    if issues:
        print(f"  {CYAN}Issues:{RESET}  {', '.join(issues)}")
    else:
        print(f"  {CYAN}Issues:{RESET}  {GREEN}none{RESET}")

    # Probe results
    if probes:
        print(f"\n  {BOLD}Retrieval Probes:{RESET}")
        passed = 0
        for p in probes:
            score = p["score"]
            if score >= 0.7:
                icon = f"{GREEN}✓{RESET}"
                passed += 1
            elif score >= 0.5:
                icon = f"{YELLOW}~{RESET}"
                passed += 0.5
            else:
                icon = f"{RED}✗{RESET}"
            bar_len = int(score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            content_preview = p["top_content"][:60].replace("\n", " ")
            print(f"    {icon} {bar} {score:.0%}  \"{p['query'][:30]}\"")
            print(f"      {DIM}→ {content_preview}{RESET}")

        total_probes = len(probes)
        score_pct = (passed / total_probes * 100) if total_probes else 0
        if score_pct >= 80:
            grade = f"{GREEN}RAG-READY{RESET}"
        elif score_pct >= 50:
            grade = f"{YELLOW}ACCEPTABLE{RESET}"
        else:
            grade = f"{RED}NEEDS WORK{RESET}"
        print(f"\n  {BOLD}Grade: {grade} ({passed:.0f}/{total_probes} probes passed){RESET}")


def main():
    parser = argparse.ArgumentParser(description="Validate Qdrant collections for RAG readiness")
    parser.add_argument("-c", "--collection", default=None, help="Specific collection")
    args = parser.parse_args()

    collections = get_collections()
    if args.collection:
        collections = [c for c in collections if c["name"] == args.collection]
        if not collections:
            print(f"Collection '{args.collection}' not found", file=sys.stderr)
            return 1

    print(f"{BOLD}Validating {len(collections)} collection(s)...{RESET}")

    for col in collections:
        try:
            validation = validate_collection(col["name"])
            print_report(col["name"], validation)
        except Exception as e:
            print(f"\n{RED}Error validating {col['name']}: {e}{RESET}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
