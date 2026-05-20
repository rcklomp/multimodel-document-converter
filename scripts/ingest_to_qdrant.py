#!/usr/bin/env python3
"""
Ingest ingestion.jsonl + assets into Qdrant with llava multimodal embeddings.

V2.7.1: Uses contextualized text for embedding (Anthropic Contextual Retrieval).
The canonical ``content`` field is NOT mutated. A separate ``contextualized_text``
is built for embedding, and QA/source-text validation uses the raw content.

Usage:
    python scripts/ingest_to_qdrant.py output/HarryPotter_and_the_Sorcerers_Stone/ingestion.jsonl

Options:
    --collection    Collection name (default: derived from source_file)
    --qdrant-url    Qdrant URL (default: http://localhost:6333)
    --ollama-url    Ollama URL (default: http://localhost:11434)
    --model         Embedding model (default: llava)
    --batch-size    Chunks per embedding batch (default: 10)
    --recreate      Drop and recreate collection if it exists
    --no-contextual  Disable contextual retrieval (fallback to breadcrumb-only)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

# Deterministic UUID namespace for Qdrant point IDs derived from chunk_id.
# Using a fixed namespace (rather than NAMESPACE_DNS or random) so the same
# chunk_id always maps to the same point_id across runs and across machines.
# Critical for the multi-doc-per-collection workflow (e.g. PLAN_V2.8 broad
# reconversion -> single mmrag_v2_8 collection): without a stable per-chunk
# point ID, sequential `i+1` IDs collide across files and later docs
# overwrite earlier ones.
_POINT_ID_NAMESPACE = uuid.UUID("8b7c5e3a-1f4d-4b2a-9c1e-6d8a3f0b9c2e")

# Contextual Retrieval — builds embedding text from hierarchical + neighbor context.
from mmrag_v2.chunking.contextual_retrieval import build_contextualized_text


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


# ── Dashscope embedding (v2.11 Phase 1) ────────────────────────────────────

_DASHSCOPE_EMBED_URL = "https://dashscope-intl.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"


def embed_text_dashscope(text: str, model: str, api_key: str,
                        timeout: int = 30, retries: int = 4) -> list[float]:
    """Embed text via Dashscope native embedding API.

    Used by v2.11 Phase 1 as the cloud-hosted challenger to the v2.10
    Ollama llava baseline. text-embedding-v3 / v4 both return 1024-dim
    cosine vectors. Image chunks route through text-of-description
    (no multimodal embedding — distinct from llava's behavior).
    """
    import time
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY required for --provider dashscope")
    # Dashscope text embedding takes ~8K char inputs; truncate defensively.
    text = (text or "")[:8000] or " "
    body = json.dumps({
        "model": model,
        "input": {"texts": [text]},
        "parameters": {"text_type": "document"},
    }).encode()
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(_DASHSCOPE_EMBED_URL, data=body, method="POST")
            req.add_header("Authorization", f"Bearer {api_key}")
            req.add_header("Content-Type", "application/json")
            resp = urllib.request.urlopen(req, timeout=timeout)
            data = json.loads(resp.read())
            return data["output"]["embeddings"][0]["embedding"]
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            try:
                detail = e.read().decode("utf-8", errors="replace")[:200]
            except Exception:
                detail = ""
            raise RuntimeError(f"Dashscope HTTP {e.code}: {detail}") from e
        except (urllib.error.URLError, ConnectionError, OSError, TimeoutError) as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError(f"Dashscope embed failed after {retries} retries: {last_err}")


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


# ── Domain-specific retrieval priority ─────────────────────────────────────

_PRIORITY_RANK = {"low": 0, "medium": 1, "high": 2}
_DEFAULT_PRIORITY_BY_MODALITY = {"text": "high", "table": "medium", "image": "low"}
_TOC_RE = re.compile(r"^\s*(table\s+of\s+contents|contents)\s*$", re.IGNORECASE)
_REFERENCES_RE = re.compile(r"^\s*(references|bibliography|works\s+cited)\s*$", re.IGNORECASE)
_TECHNICAL_BACK_MATTER_RE = re.compile(
    r"^\s*(index|appendix(?:\s+[a-z0-9]+)?|appendices)\b",
    re.IGNORECASE,
)


def read_ingestion_jsonl(jsonl_path: Path) -> tuple[dict | None, list[dict]]:
    """Read an ingestion JSONL and separate the document metadata record."""
    chunks = []
    metadata_record = None
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("object_type") == "ingestion_metadata":
                metadata_record = obj
                continue
            chunks.append(obj)
    return metadata_record, chunks


def infer_document_domain(metadata_record: dict | None, chunks: list[dict]) -> str:
    """Prefer document-level domain, then fall back to chunk-level metadata."""
    if metadata_record:
        domain = metadata_record.get("domain") or metadata_record.get("document_domain")
        if domain:
            return str(domain).lower()

    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        domain = metadata.get("document_domain") or metadata.get("domain")
        if domain:
            return str(domain).lower()
    return "unknown"


def _normalized_priority(value: Any, modality: str) -> str:
    priority = str(value or "").lower()
    if priority in _PRIORITY_RANK:
        return priority
    return _DEFAULT_PRIORITY_BY_MODALITY.get(modality, "medium")


def _demote_at_most(current: str, candidate: str) -> str:
    """Return the lower of two priorities so strict converter signals survive."""
    if _PRIORITY_RANK[candidate] < _PRIORITY_RANK[current]:
        return candidate
    return current


def _page_number(chunk: dict) -> int:
    try:
        return int((chunk.get("metadata") or {}).get("page_number") or 0)
    except (TypeError, ValueError):
        return 0


def _hierarchy_strings(chunk: dict) -> list[str]:
    metadata = chunk.get("metadata") or {}
    hierarchy = metadata.get("hierarchy") or {}
    values: list[str] = []
    parent_heading = hierarchy.get("parent_heading")
    if parent_heading:
        values.append(str(parent_heading))
    breadcrumb = hierarchy.get("breadcrumb_path") or []
    if isinstance(breadcrumb, list):
        values.extend(str(item) for item in breadcrumb if item)
    return values


def _is_heading_chunk(chunk: dict) -> bool:
    chunk_type = chunk.get("chunk_type") or (chunk.get("metadata") or {}).get("chunk_type")
    return str(chunk_type or "").lower() == "heading"


def _heading_context(chunk: dict) -> list[str]:
    values = _hierarchy_strings(chunk)
    if _is_heading_chunk(chunk):
        content = chunk.get("content")
        if content:
            values.append(str(content))
    return values


def _is_toc_chunk(chunk: dict) -> bool:
    for value in _heading_context(chunk):
        if _TOC_RE.match(value.strip()):
            return True
    if _is_heading_chunk(chunk):
        first_line = str(chunk.get("content") or "").strip().splitlines()[0:1]
        return bool(first_line and _TOC_RE.match(first_line[0].strip()))
    return False


def find_literature_toc_page(chunks: list[dict]) -> int | None:
    """Return the first TOC page for literature front-matter priority rules."""
    pages = [_page_number(chunk) for chunk in chunks if _is_toc_chunk(chunk)]
    pages = [page for page in pages if page > 0]
    return min(pages) if pages else None


def _has_references_heading_context(chunk: dict) -> bool:
    return any(_REFERENCES_RE.match(value.strip()) for value in _heading_context(chunk))


def _has_technical_back_matter_context(chunk: dict) -> bool:
    return any(_TECHNICAL_BACK_MATTER_RE.match(value.strip()) for value in _heading_context(chunk))


def resolve_search_priority(
    chunk: dict,
    document_domain: str,
    literature_toc_page: int | None = None,
) -> str:
    """
    Resolve retrieval priority at ingestion time from domain and structure.

    Domain rules only demote existing priority:
    - literature pages before the first TOC are low priority
    - academic references/bibliography sections are medium priority
    - technical index/appendix sections are medium priority
    """
    modality = str(chunk.get("modality") or "text").lower()
    metadata = chunk.get("metadata") or {}
    priority = _normalized_priority(metadata.get("search_priority"), modality)
    domain = (document_domain or "unknown").lower()

    if domain == "literature" and literature_toc_page is not None:
        page = _page_number(chunk)
        if 0 < page < literature_toc_page:
            priority = _demote_at_most(priority, "low")
    elif domain == "academic" and _has_references_heading_context(chunk):
        priority = _demote_at_most(priority, "medium")
    elif domain == "technical" and _has_technical_back_matter_context(chunk):
        priority = _demote_at_most(priority, "medium")

    return priority


def build_qdrant_payload(
    chunk: dict,
    *,
    source_file: str,
    document_domain: str,
    literature_toc_page: int | None = None,
) -> dict:
    """Build Qdrant payload fields, including ingestor-owned search priority."""
    chunk_id = chunk.get("chunk_id", "")
    modality = chunk.get("modality", "text")
    metadata = chunk.get("metadata") or {}
    content = metadata.get("refined_content") or chunk.get("content", "")
    page = metadata.get("page_number", 0)
    hierarchy = metadata.get("hierarchy") or {}

    payload = {
        "chunk_id": chunk_id,
        "doc_id": chunk.get("doc_id", ""),
        "modality": modality,
        "content": content[:10000],  # Cap payload size
        "page_number": page,
        "source_file": metadata.get("source_file", source_file),
        "search_priority": resolve_search_priority(
            chunk,
            document_domain,
            literature_toc_page=literature_toc_page,
        ),
    }
    if document_domain and document_domain != "unknown":
        payload["document_domain"] = document_domain

    if modality == "text":
        payload["chunk_type"] = metadata.get("chunk_type", "")
        payload["parent_heading"] = hierarchy.get("parent_heading", "")
        payload["breadcrumb"] = " > ".join(hierarchy.get("breadcrumb_path", []))
    elif modality == "image":
        payload["visual_description"] = (
            metadata.get("visual_description")
            or chunk.get("visual_description", "")
        )
        asset_ref = chunk.get("asset_ref", {})
        payload["asset_path"] = asset_ref.get("file_path", "")

    return payload


# ── Main ingestion ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest ingestion.jsonl into Qdrant")
    parser.add_argument("jsonl_path", type=Path, help="Path to ingestion.jsonl")
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    parser.add_argument("--provider", type=str, default="ollama",
                        choices=["ollama", "dashscope"],
                        help="Embedding provider. 'ollama' = local Ollama (default, llava 4096-dim multimodal). "
                             "'dashscope' = Dashscope cloud text-embedding (v2.11 Phase 1 challenger; "
                             "1024-dim, text-only — image chunks embed via their VLM description).")
    parser.add_argument("--model", type=str, default=None,
                        help="Embedding model. Default 'llava' for ollama; 'text-embedding-v4' for dashscope.")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for dashscope provider. Defaults to DASHSCOPE_API_KEY env var.")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--no-contextual", action="store_true",
                        help="Disable contextual retrieval (fallback to breadcrumb-only)")
    args = parser.parse_args()
    # Provider-specific defaults.
    if args.model is None:
        args.model = "llava" if args.provider == "ollama" else "text-embedding-v4"
    if args.provider == "dashscope" and not args.api_key:
        args.api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not args.api_key:
            print("ERROR: --provider dashscope requires --api-key or DASHSCOPE_API_KEY env var", file=sys.stderr)
            return 2

    jsonl_path = args.jsonl_path
    if not jsonl_path.exists():
        print(f"Error: {jsonl_path} not found", file=sys.stderr)
        return 1

    assets_dir = jsonl_path.parent / "assets"

    # Read all chunks
    metadata_record, chunks = read_ingestion_jsonl(jsonl_path)

    source_file = metadata_record.get("source_file", jsonl_path.stem) if metadata_record else jsonl_path.stem
    document_domain = infer_document_domain(metadata_record, chunks)
    literature_toc_page = find_literature_toc_page(chunks) if document_domain == "literature" else None
    collection_name = args.collection or source_file.replace(" ", "_").replace(".", "_").lower()
    # Clean collection name (Qdrant allows alphanumeric + underscore + hyphen)
    collection_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in collection_name)

    print(f"Ingesting into Qdrant")
    print(f"  Source: {jsonl_path}")
    print(f"  Collection: {collection_name}")
    print(f"  Chunks: {len(chunks)} ({sum(1 for c in chunks if c.get('modality')=='text')} text, "
          f"{sum(1 for c in chunks if c.get('modality')=='image')} image, "
          f"{sum(1 for c in chunks if c.get('modality')=='table')} table)")
    print(f"  Domain: {document_domain}")
    print(f"  Model: {args.model}")
    print()

    # Provider-aware embed helpers (closures capture args).
    def _embed_text(text: str) -> list[float]:
        if args.provider == "dashscope":
            return embed_text_dashscope(text, args.model, args.api_key)
        return embed_text(text, args.model, args.ollama_url)

    def _embed_image(asset_path: Path | None, fallback_text: str) -> list[float]:
        if args.provider == "dashscope":
            # Dashscope text-embedding is text-only; use the VLM description.
            return embed_text_dashscope(fallback_text or "image", args.model, args.api_key)
        if asset_path and asset_path.exists():
            return embed_image(asset_path, args.model, args.ollama_url, fallback_text)
        return embed_text(fallback_text or "image", args.model, args.ollama_url)

    # Get embedding dimension
    print(f"  Provider: {args.provider}")
    print("  Testing embedding model...")
    test_emb = _embed_text("test")
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
        metadata = chunk.get("metadata") or {}
        # Prefer refined_content (has hyphenation fixes, OCR cleanup) over raw content
        content = metadata.get("refined_content") or chunk.get("content", "")
        page = metadata.get("page_number", 0)

        # Build embedding
        try:
            if modality == "image":
                # Provider-aware: ollama can embed image bytes directly via llava;
                # dashscope embeds the VLM description text only.
                asset_ref = chunk.get("asset_ref", {})
                asset_path = assets_dir / Path(asset_ref.get("file_path", "")).name if asset_ref else None
                description = (metadata.get("visual_description")
                             or chunk.get("visual_description", "")
                             or content)
                vector = _embed_image(asset_path, description)
            else:
                # Text and table chunks: use contextualized text for embedding.
                # Contextual retrieval prepends hierarchical + neighbor context.
                # Canonical `content` is NOT mutated — QA validation uses raw text.
                hierarchy = metadata.get("hierarchy") or {}
                semantic_ctx = chunk.get("semantic_context") or {}

                if args.no_contextual:
                    # Fallback: breadcrumb-only (pre-v2.7.1 behavior)
                    breadcrumb = " > ".join(hierarchy.get("breadcrumb_path", []))
                    text_to_embed = f"{breadcrumb}\n{content}" if breadcrumb else content
                else:
                    # Build contextualized text using the Anthropic approach.
                    # Uses breadcrumb, heading, and neighbor snippets for context.
                    breadcrumb = hierarchy.get("breadcrumb_path") or []
                    parent_heading = hierarchy.get("parent_heading")
                    prev_snippet = semantic_ctx.get("prev_text_snippet")
                    next_snippet = semantic_ctx.get("next_text_snippet")

                    text_to_embed = build_contextualized_text(
                        content,
                        breadcrumb_path=breadcrumb,
                        parent_heading=parent_heading,
                        prev_text_snippet=prev_snippet,
                        next_text_snippet=next_snippet,
                        modality=modality,
                    )

                vector = _embed_text(text_to_embed)
        except Exception as e:
            print(f"  ERROR embedding chunk {i}/{total} ({chunk_id}): {e}", file=sys.stderr)
            errors += 1
            continue

        # Build payload (metadata stored in Qdrant for retrieval)
        payload = build_qdrant_payload(
            chunk,
            source_file=source_file,
            document_domain=document_domain,
            literature_toc_page=literature_toc_page,
        )

        # Stable, collision-free point ID across multiple files in one
        # collection: deterministic UUID5 from the chunk's globally-unique
        # chunk_id. Falls back to (source_file + integer index) when chunk_id
        # is missing, which keeps single-file v2.7 behavior identical.
        seed = chunk_id if chunk_id else f"{source_file}#{i}"
        point_id = str(uuid.uuid5(_POINT_ID_NAMESPACE, seed))

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
