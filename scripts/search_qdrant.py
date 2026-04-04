#!/usr/bin/env python3
"""
Search Qdrant collections using nomic-embed-text via Ollama.

Usage:
    python scripts/search_qdrant.py "how to build walls with tires"
    python scripts/search_qdrant.py "dragon hatching" --collection harrypotter_and_the_sorcerers_stone_pdf
    python scripts/search_qdrant.py "solar heating" --limit 10
    python scripts/search_qdrant.py "greenhouse design" --modality image
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request


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
           modality: str | None = None, qdrant_url: str = "http://localhost:6333") -> list[dict]:
    body: dict = {"vector": query_vector, "limit": limit, "with_payload": True}
    if modality:
        body["filter"] = {"must": [{"key": "modality", "match": {"value": modality}}]}
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{qdrant_url}/collections/{collection}/points/search", data=data)
    req.add_header("Content-Type", "application/json")
    return json.loads(urllib.request.urlopen(req, timeout=30).read())["result"]


def main():
    parser = argparse.ArgumentParser(description="Search Qdrant with semantic queries")
    parser.add_argument("query", nargs="?", help="Search query text")
    parser.add_argument("--collection", "-c", default=None, help="Collection name (default: search all)")
    parser.add_argument("--limit", "-n", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--modality", "-m", choices=["text", "image", "table"], default=None, help="Filter by modality")
    parser.add_argument("--list", "-l", action="store_true", help="List all collections")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    args = parser.parse_args()

    if args.list or not args.query:
        collections = list_collections(args.qdrant_url)
        print("Collections:")
        for c in collections:
            print(f"  {c['name']}: {c['points']} points")
        if not args.query:
            return 0

    query = args.query
    vector = embed(query, ollama_url=args.ollama_url)

    # Determine which collections to search
    if args.collection:
        targets = [args.collection]
    else:
        targets = [c["name"] for c in list_collections(args.qdrant_url)]

    print(f'\nQuery: "{query}"')
    if args.modality:
        print(f"Filter: modality={args.modality}")
    print()

    for collection in targets:
        results = search(vector, collection, args.limit, args.modality, args.qdrant_url)
        if not results:
            continue

        print(f"── {collection} ──")
        for r in results:
            p = r["payload"]
            score = r["score"]
            mod = p.get("modality", "?")
            pg = p.get("page_number", "?")
            source = p.get("source_file", "")

            if mod == "image":
                content = p.get("visual_description", "") or p.get("content", "")
                asset = p.get("asset_path", "")
                print(f"  [{score:.3f}] IMAGE pg={pg} | {content[:100]}")
                if asset:
                    print(f"          asset: {asset}")
            else:
                content = p.get("content", "")
                heading = p.get("parent_heading", "")
                prefix = f"[{heading}] " if heading else ""
                print(f"  [{score:.3f}] {mod:5} pg={pg} | {prefix}{content[:100]}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
