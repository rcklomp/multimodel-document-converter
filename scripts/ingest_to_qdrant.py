#!/usr/bin/env python3
"""
Ingest ingestion.jsonl + assets into Qdrant with llava multimodal embeddings.

Usage:
    python scripts/ingest_to_qdrant.py output/HarryPotter_and_the_Sorcerers_Stone/ingestion.jsonl

Options:
    --collection    Collection name (default: derived from source_file)
    --qdrant-url    Qdrant URL (default: http://localhost:6333)
    --ollama-url    Ollama URL (default: http://localhost:11434)
    --model         Embedding model (default: llava)
    --batch-size    Chunks per embedding batch (default: 10)
    --recreate      Drop and recreate collection if it exists
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path


# ── Ollama embedding ────────────────────────────────────────────────────────

def embed_text(text: str, model: str, ollama_url: str) -> list[float]:
    """Get text embedding from Ollama."""
    data = json.dumps({"model": model, "input": text}).encode()
    req = urllib.request.Request(f"{ollama_url}/api/embed", data=data)
    req.add_header("Content-Type", "application/json")
    resp = urllib.request.urlopen(req, timeout=120)
    return json.loads(resp.read())["embeddings"][0]


def embed_image(image_path: Path, model: str, ollama_url: str, fallback_text: str = "") -> list[float]:
    """Get image embedding from Ollama (multimodal model required)."""
    if not image_path.exists():
        # Fallback to text embedding of the description
        return embed_text(fallback_text or "image", model, ollama_url)

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    data = json.dumps({
        "model": model,
        "input": " ",  # Ollama requires input field
        "images": [b64],
    }).encode()
    req = urllib.request.Request(f"{ollama_url}/api/embed", data=data)
    req.add_header("Content-Type", "application/json")
    resp = urllib.request.urlopen(req, timeout=120)
    return json.loads(resp.read())["embeddings"][0]


# ── Qdrant operations ──────────────────────────────────────────────────────

def qdrant_request(method: str, path: str, body: dict | None, qdrant_url: str) -> dict:
    """Make a request to Qdrant REST API."""
    url = f"{qdrant_url}{path}"
    if body is not None:
        data = json.dumps(body).encode()
    else:
        data = None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode() if e.fp else ""
        raise RuntimeError(f"Qdrant {method} {path}: {e.code} {body_text[:200]}") from e


def create_collection(name: str, dim: int, qdrant_url: str, recreate: bool = False):
    """Create a Qdrant collection with mmap storage for scalability."""
    # Check if exists
    try:
        resp = qdrant_request("GET", f"/collections/{name}", None, qdrant_url)
        if resp.get("result"):
            if recreate:
                print(f"  Dropping existing collection '{name}'...")
                qdrant_request("DELETE", f"/collections/{name}", None, qdrant_url)
            else:
                count = resp["result"].get("points_count", 0)
                print(f"  Collection '{name}' exists ({count} points). Use --recreate to reset.")
                return False
    except RuntimeError:
        pass  # Collection doesn't exist

    print(f"  Creating collection '{name}' ({dim} dims, mmap on disk)...")
    qdrant_request("PUT", f"/collections/{name}", {
        "vectors": {
            "size": dim,
            "distance": "Cosine",
            "on_disk": True,  # mmap: vectors on SSD, not RAM
        },
        "optimizers_config": {
            "memmap_threshold": 10000,  # Switch to mmap after 10K vectors
        },
        "on_disk_payload": True,  # Payloads on disk too
    }, qdrant_url)
    return True


def upsert_batch(collection: str, points: list[dict], qdrant_url: str):
    """Upsert a batch of points into Qdrant."""
    qdrant_request("PUT", f"/collections/{collection}/points", {
        "points": points,
    }, qdrant_url)


# ── Main ingestion ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest ingestion.jsonl into Qdrant")
    parser.add_argument("jsonl_path", type=Path, help="Path to ingestion.jsonl")
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    parser.add_argument("--model", type=str, default="llava")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    jsonl_path = args.jsonl_path
    if not jsonl_path.exists():
        print(f"Error: {jsonl_path} not found", file=sys.stderr)
        return 1

    assets_dir = jsonl_path.parent / "assets"

    # Read all chunks
    chunks = []
    metadata_record = None
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("object_type") == "ingestion_metadata":
                metadata_record = obj
                continue
            chunks.append(obj)

    source_file = metadata_record.get("source_file", jsonl_path.stem) if metadata_record else jsonl_path.stem
    collection_name = args.collection or source_file.replace(" ", "_").replace(".", "_").lower()
    # Clean collection name (Qdrant allows alphanumeric + underscore + hyphen)
    collection_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in collection_name)

    print(f"Ingesting into Qdrant")
    print(f"  Source: {jsonl_path}")
    print(f"  Collection: {collection_name}")
    print(f"  Chunks: {len(chunks)} ({sum(1 for c in chunks if c.get('modality')=='text')} text, "
          f"{sum(1 for c in chunks if c.get('modality')=='image')} image, "
          f"{sum(1 for c in chunks if c.get('modality')=='table')} table)")
    print(f"  Model: {args.model}")
    print()

    # Get embedding dimension
    print("  Testing embedding model...")
    test_emb = embed_text("test", args.model, args.ollama_url)
    dim = len(test_emb)
    print(f"  Embedding dimension: {dim}")

    # Create collection
    create_collection(collection_name, dim, args.qdrant_url, args.recreate)
    print()

    # Embed and upsert in batches
    total = len(chunks)
    points_buffer = []
    embedded = 0
    errors = 0
    t0 = time.time()

    for i, chunk in enumerate(chunks):
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        modality = chunk.get("modality", "text")
        # Prefer refined_content (has hyphenation fixes, OCR cleanup) over raw content
        content = chunk.get("metadata", {}).get("refined_content") or chunk.get("content", "")
        page = chunk.get("metadata", {}).get("page_number", 0)

        # Build embedding
        try:
            if modality == "image":
                # Try image embedding first, fall back to description text
                asset_ref = chunk.get("asset_ref", {})
                asset_path = assets_dir / Path(asset_ref.get("file_path", "")).name if asset_ref else None
                description = (chunk.get("metadata", {}).get("visual_description")
                             or chunk.get("visual_description", "")
                             or content)
                if asset_path and asset_path.exists():
                    vector = embed_image(asset_path, args.model, args.ollama_url, description)
                else:
                    vector = embed_text(description, args.model, args.ollama_url)
            else:
                # Text and table chunks: embed content directly
                text_to_embed = content
                # For richer embedding, prepend breadcrumb context
                hierarchy = chunk.get("metadata", {}).get("hierarchy", {})
                breadcrumb = " > ".join(hierarchy.get("breadcrumb_path", []))
                if breadcrumb:
                    text_to_embed = f"{breadcrumb}\n{content}"
                vector = embed_text(text_to_embed, args.model, args.ollama_url)
        except Exception as e:
            print(f"  ERROR embedding chunk {i}/{total} ({chunk_id}): {e}", file=sys.stderr)
            errors += 1
            continue

        # Build payload (metadata stored in Qdrant for retrieval)
        payload = {
            "chunk_id": chunk_id,
            "doc_id": chunk.get("doc_id", ""),
            "modality": modality,
            "content": content[:10000],  # Cap payload size
            "page_number": page,
            "source_file": chunk.get("metadata", {}).get("source_file", source_file),
        }

        # Add modality-specific fields
        if modality == "text":
            payload["chunk_type"] = chunk.get("metadata", {}).get("chunk_type", "")
            payload["parent_heading"] = chunk.get("metadata", {}).get("hierarchy", {}).get("parent_heading", "")
            payload["breadcrumb"] = " > ".join(
                chunk.get("metadata", {}).get("hierarchy", {}).get("breadcrumb_path", [])
            )
        elif modality == "image":
            payload["visual_description"] = (
                chunk.get("metadata", {}).get("visual_description")
                or chunk.get("visual_description", "")
            )
            asset_ref = chunk.get("asset_ref", {})
            payload["asset_path"] = asset_ref.get("file_path", "")

        # Use a stable integer ID (Qdrant needs int or UUID)
        point_id = i + 1

        points_buffer.append({
            "id": point_id,
            "vector": vector,
            "payload": payload,
        })

        embedded += 1

        # Upsert batch
        if len(points_buffer) >= args.batch_size:
            upsert_batch(collection_name, points_buffer, args.qdrant_url)
            elapsed = time.time() - t0
            rate = embedded / elapsed if elapsed > 0 else 0
            print(f"  [{embedded}/{total}] {rate:.1f} chunks/sec "
                  f"({modality} pg={page})", end="\r")
            points_buffer = []

    # Final batch
    if points_buffer:
        upsert_batch(collection_name, points_buffer, args.qdrant_url)

    elapsed = time.time() - t0
    print(f"\n\nDone!")
    print(f"  Embedded: {embedded}/{total} chunks ({errors} errors)")
    print(f"  Time: {elapsed:.1f}s ({embedded/elapsed:.1f} chunks/sec)")
    print(f"  Collection: {collection_name}")

    # Verify
    resp = qdrant_request("GET", f"/collections/{collection_name}", None, args.qdrant_url)
    count = resp["result"].get("points_count", 0)
    print(f"  Qdrant points: {count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
