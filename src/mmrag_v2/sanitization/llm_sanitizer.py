"""LLM sanitizer client + content-pinning cache.

Charter: docs/ARCHITECTURE_V3_DRAFT_0.5.md §3.3, §7.4.

Foundation-session status: the cache-key computation is FUNCTIONAL.
The endpoint client + actual LLM call is STUB — `sanitize_via_llm()`
raises `NotImplementedError` so a Phase B mistake won't silently send
chunks to nowhere.

Cache-key contract (Charter §7.4):
    key = (content_hash, context_hash, model_id, prompt_version)
    context_hash = SHA-256(prev_chunk_content_first64bits
                         + next_chunk_content_first64bits
                         + detected_lang)

Cache lookup vs invocation: on cache hit, no LLM is invoked. Cache is
file-backed under `output/sanitization_cache/`, keyed by content hash
prefix + first 8 chars of context hash.

Prompt-template hash change cost (Charter §3.3 prompt-migration cost
note, Draft 0.5 audit C11 #3): when `prompt_version` (git hash of the
prompt template) changes, every chunk previously cached becomes a cache
miss, and the entire corpus must be re-sanitized at the B8 cold-cache
cost. The CI gate at `docs/PHASE_B_BUILD_TIMES.md` documents this.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Charter §7.4: first 64 bits of SHA-256 = 16 hex chars.
_CONTEXT_HASH_PREFIX_BYTES = 8  # 64 bits / 8 bits-per-byte


@dataclass(frozen=True)
class SanitizationCacheKey:
    """Charter §7.4 content-pinning cache key.

    Tuple-equivalent dataclass — used as a dict key in in-memory cache
    and as a filename in disk-backed cache.
    """

    content_hash: str  # Full SHA-256 hex of raw chunk content
    context_hash: str  # SHA-256 hex of context tuple
    model_id: str  # e.g., "RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic"
    prompt_version: str  # Git hash (short) of the prompt template

    def cache_filename(self) -> str:
        """Disk-backed cache filename per Charter §7.4.

        `content hash prefix + first 8 chars of context hash` keeps the
        path length bounded; collisions are resolved by re-checking the
        full key inside the JSON payload before returning a cache hit.
        """
        return f"{self.content_hash[:16]}_{self.context_hash[:8]}.json"


def compute_content_hash(content: str) -> str:
    """SHA-256 hex digest of the raw content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_context_hash(
    *,
    prev_chunk_content: Optional[str],
    next_chunk_content: Optional[str],
    detected_lang: Optional[str],
) -> str:
    """SHA-256 of the context-tuple per Charter §7.4.

    Each component is hashed first to its 64-bit prefix (16 hex chars)
    to keep the input compact, then those prefixes + the lang code are
    concatenated and hashed. Missing components become empty string.
    """

    def _short_hash(value: Optional[str]) -> str:
        if value is None:
            return ""
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        return digest[: _CONTEXT_HASH_PREFIX_BYTES * 2]  # 16 hex chars

    parts = [
        _short_hash(prev_chunk_content),
        _short_hash(next_chunk_content),
        detected_lang or "",
    ]
    payload = "|".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_cache_key(
    *,
    raw_content: str,
    prev_chunk_content: Optional[str],
    next_chunk_content: Optional[str],
    detected_lang: Optional[str],
    model_id: str,
    prompt_version: str,
) -> SanitizationCacheKey:
    """Build a Charter §7.4 cache key from chunk + context + model + prompt."""
    return SanitizationCacheKey(
        content_hash=compute_content_hash(raw_content),
        context_hash=compute_context_hash(
            prev_chunk_content=prev_chunk_content,
            next_chunk_content=next_chunk_content,
            detected_lang=detected_lang,
        ),
        model_id=model_id,
        prompt_version=prompt_version,
    )


class FileBackedSanitizationCache:
    """Charter §7.4 disk-backed cache under `output/sanitization_cache/`.

    Foundation-session status: FUNCTIONAL. Phase B uses this directly.
    The cache stores the full SanitizationResult JSON per key; on a
    cache hit the LLM is not invoked.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: SanitizationCacheKey) -> Path:
        return self.root / key.cache_filename()

    def get(self, key: SanitizationCacheKey) -> Optional[dict]:
        """Return cached payload or None on miss / collision-mismatch."""
        path = self._path(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Corrupted cache entry — treat as miss; Phase B will re-sanitize.
            return None
        # Collision check: the file name truncates the keys. Re-verify
        # the full key matches what we asked for.
        cached_key = payload.get("_cache_key", {})
        if (
            cached_key.get("content_hash") == key.content_hash
            and cached_key.get("context_hash") == key.context_hash
            and cached_key.get("model_id") == key.model_id
            and cached_key.get("prompt_version") == key.prompt_version
        ):
            return payload
        return None

    def put(self, key: SanitizationCacheKey, payload: dict) -> None:
        """Store payload at the cache slot for `key`."""
        full_payload = dict(payload)
        full_payload["_cache_key"] = {
            "content_hash": key.content_hash,
            "context_hash": key.context_hash,
            "model_id": key.model_id,
            "prompt_version": key.prompt_version,
        }
        path = self._path(key)
        path.write_text(json.dumps(full_payload, indent=2), encoding="utf-8")


def sanitize_via_llm(
    *,
    raw_content: str,
    context: Optional[dict] = None,
    cache: Optional[FileBackedSanitizationCache] = None,
    model_id: str = "RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic",
    prompt_version: str = "v3.0-foundation",
    endpoint_url: str = "http://10.0.10.239:8000/v1/chat/completions",
) -> str:
    """Send a chunk to the GX10 FP8 endpoint for sanitization.

    Foundation-session status: NOT IMPLEMENTED. The endpoint client +
    prompt template + guard pipeline land in Phase B (Charter Cycle 3.1).
    A guard against premature use is in place: this function raises
    NotImplementedError so no production path can silently bypass the
    Phase B contract.
    """
    _ = (raw_content, context, cache, model_id, prompt_version, endpoint_url)
    raise NotImplementedError(
        "LLM sanitization client lands in Phase B (Charter §3.3). "
        "Foundation session ships only the cache-key contract; the GX10 "
        "endpoint dispatch + prompt template + guard pipeline are not yet "
        "wired. Use SanitizationMode.OFF or .HEURISTIC during foundation."
    )
