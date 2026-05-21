"""v2.12 Phase 2 — BM25 sparse vector encoder for hybrid retrieval.

Implements Okapi BM25 as sparse vectors that Qdrant can store + search
via the named-sparse-vector mechanism. Pairs with the existing dense
embedding lane to give hybrid (BM25 + dense + RRF) retrieval.

BM25 formula:
    IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
    score(d, q) = sum_{t in q} IDF(t) * (tf(t,d)*(k1+1)) /
                                       (tf(t,d) + k1*(1 - b + b*|d|/avgdl))

Standard params: k1=1.5, b=0.75. We pre-compute the per-document BM25
weights so the Qdrant sparse search reduces to a simple dot product:

    query_sparse  = {(token_id, 1.0) for token in query_tokens}
    doc_sparse    = {(token_id, doc_bm25_weight) for token in doc_tokens}

    score = dot(query_sparse, doc_sparse) = BM25(d, q)   exactly.

The vocab is built once over the corpus and persisted to disk; the
index file is tracked in the repo so the same query embedding can
reproduce the same sparse vector across machines.

Tokenization is character-level multilingual-safe (lowercase,
non-alphanumeric → whitespace, drop empty/single-char tokens). The
v2.11 corpus is English + Dutch + German prose — no stemmer / no
stopword list, just literal token matching. Stemmers add language
dependency and the BM25 score remains useful without them.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Standard BM25 hyperparameters. Use whatever the literature says.
BM25_K1 = 1.5
BM25_B = 0.75

# Tokenizer: lowercase, non-alphanumeric → whitespace, split, drop
# single-character tokens. Works for English/Dutch/German on this
# corpus; not a stemmer (deliberately).
_TOKEN_RE = re.compile(r"[^\w]+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenize text into BM25 tokens.

    Multilingual-safe: lowercases via .casefold(), uses unicode word
    classes so non-ASCII letters (Dutch ë, German ß, etc.) pass through.
    Drops digits-only and single-character tokens to reduce vocab noise.
    """
    if not text:
        return []
    lowered = text.casefold()
    raw = _TOKEN_RE.split(lowered)
    return [t for t in raw if len(t) > 1 and not t.isdigit()]


@dataclass
class BM25Index:
    """BM25 vocabulary + IDF table + corpus statistics.

    Built once from a corpus, then used to encode any text (query OR
    chunk) into a sparse vector compatible with Qdrant.

    Fields:
      vocab            map token → token_id (stable across encodes)
      idf              IDF score per token_id (parallel array)
      avgdl            average document length, in tokens
      n_docs           total documents the index was built from
      bm25_k1, bm25_b  hyperparameters (default 1.5 / 0.75)
    """
    vocab: dict[str, int] = field(default_factory=dict)
    idf: list[float] = field(default_factory=list)
    avgdl: float = 0.0
    n_docs: int = 0
    bm25_k1: float = BM25_K1
    bm25_b: float = BM25_B

    @classmethod
    def build_from_corpus(
        cls,
        documents: Iterable[str],
        *,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> "BM25Index":
        """Build a fresh index over a corpus of strings."""
        df: dict[str, int] = {}  # document frequency per token
        total_len = 0
        n = 0
        # First pass: count docs and accumulate df.
        for text in documents:
            tokens = tokenize(text)
            n += 1
            total_len += len(tokens)
            seen: set[str] = set()
            for t in tokens:
                if t in seen:
                    continue
                seen.add(t)
                df[t] = df.get(t, 0) + 1

        # Build stable vocab (deterministic order via sorted tokens).
        vocab = {tok: i for i, tok in enumerate(sorted(df.keys()))}
        idf = [0.0] * len(vocab)
        for tok, dfreq in df.items():
            # Okapi BM25 IDF — the +1 inside the log keeps it non-negative.
            idf[vocab[tok]] = math.log((n - dfreq + 0.5) / (dfreq + 0.5) + 1.0)

        avgdl = (total_len / n) if n else 0.0
        return cls(
            vocab=vocab,
            idf=idf,
            avgdl=avgdl,
            n_docs=n,
            bm25_k1=k1,
            bm25_b=b,
        )

    def encode_document(self, text: str) -> tuple[list[int], list[float]]:
        """Encode a document's tokens into a BM25-weighted sparse vector.

        Returns (indices, values) where each index is a vocab token_id
        and each value is the BM25 weight: IDF(t) * tf'(t, d), where
        tf'(t, d) = tf*(k1+1) / (tf + k1*(1 - b + b*|d|/avgdl)).

        Tokens not in the vocab are dropped silently (out-of-vocab).
        Query-time documents (e.g. queries with novel terms) will see
        OOV; this is acceptable for hybrid retrieval since BM25 is a
        complement to dense, not a replacement.
        """
        tokens = tokenize(text)
        if not tokens:
            return [], []
        tf: dict[int, int] = {}
        for t in tokens:
            tid = self.vocab.get(t)
            if tid is None:
                continue
            tf[tid] = tf.get(tid, 0) + 1
        if not tf:
            return [], []
        doclen = len(tokens)
        # Avoid div-by-zero on empty corpora; in practice avgdl > 0.
        avgdl = self.avgdl if self.avgdl > 0 else 1.0
        norm = self.bm25_k1 * (1.0 - self.bm25_b + self.bm25_b * (doclen / avgdl))
        indices: list[int] = []
        values: list[float] = []
        for tid, freq in tf.items():
            weight = self.idf[tid] * (freq * (self.bm25_k1 + 1.0)) / (freq + norm)
            indices.append(tid)
            values.append(float(weight))
        return indices, values

    def encode_query(self, text: str) -> tuple[list[int], list[float]]:
        """Encode a query's tokens as a binary-presence sparse vector.

        For the BM25 dot-product to yield the BM25 score, the query
        side just needs `1.0` per token (presence). The IDF + length
        normalization were already baked into `encode_document`.

        Tokens not in the vocab are dropped (OOV).
        """
        tokens = tokenize(text)
        if not tokens:
            return [], []
        # Use deduplication to avoid double-counting query terms.
        seen_ids: set[int] = set()
        indices: list[int] = []
        values: list[float] = []
        for t in tokens:
            tid = self.vocab.get(t)
            if tid is None or tid in seen_ids:
                continue
            seen_ids.add(tid)
            indices.append(tid)
            values.append(1.0)
        return indices, values

    def to_json(self) -> dict:
        """Serialize the index to a JSON-safe dict."""
        return {
            "version": 1,
            "n_docs": self.n_docs,
            "avgdl": self.avgdl,
            "bm25_k1": self.bm25_k1,
            "bm25_b": self.bm25_b,
            # vocab + idf are large but the JSON form is fine for the
            # ~30K-chunk corpus (~hundreds of KB).
            "vocab": self.vocab,
            "idf": self.idf,
        }

    @classmethod
    def from_json(cls, data: dict) -> "BM25Index":
        return cls(
            vocab=dict(data["vocab"]),
            idf=list(data["idf"]),
            avgdl=float(data["avgdl"]),
            n_docs=int(data["n_docs"]),
            bm25_k1=float(data.get("bm25_k1", BM25_K1)),
            bm25_b=float(data.get("bm25_b", BM25_B)),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        return cls.from_json(json.loads(path.read_text(encoding="utf-8")))


# ── RRF fusion ───────────────────────────────────────────────────────────────


def rrf_fuse(
    *ranked_lists: list[str],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion across N ranked lists of chunk_ids.

    Each list is a top-K ranking; rank 1 contributes 1/(k+1), rank 2
    contributes 1/(k+2), etc. The fused score is the sum across lists;
    output is sorted descending by fused score.

    Standard k=60 (from the original Cormack et al. 2009 RRF paper).
    Optional per-list `weights` scale each list's contribution.

    Returns: list of (chunk_id, fused_score), descending.
    """
    if not ranked_lists:
        return []
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError(
            f"weights length ({len(weights)}) must match number of "
            f"ranked lists ({len(ranked_lists)})"
        )
    scores: dict[str, float] = {}
    for ranking, w in zip(ranked_lists, weights):
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + w * (1.0 / (k + rank + 1))
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return fused
