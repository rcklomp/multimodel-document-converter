"""Guard 8: Corpus-level dedup-ratio invariant.

Charter §3.3 guard table row 8 (NEW in 0.4): runs once per build, not
per chunk. Measures near-duplicate ratio (Jaccard ≥0.9 over shingles)
across the sanitized corpus and compares against the heuristic corpus's
ratio. If LLM-mode dedup-ratio exceeds heuristic by >5%, build emits a
`SANITIZATION_DEDUP_DRIFT` warning and the dominance criterion treats
this as a regression on Format.

What this catches that per-chunk guards cannot: inter-chunk consistency
drift — chunk A's heuristic-removed footer reappears as chunk B's
prepended caption after LLM sanitization. Per-chunk guards can never
see across chunks.

Foundation-session status: FUNCTIONAL.

The shingle algorithm is 3-gram (token-level) Jaccard similarity, with
near-duplicate detected at Jaccard ≥ 0.9. The corpus-level metric is
the fraction of (chunk_a, chunk_b) pairs that are near-duplicates,
divided by the total pair count. Naïve O(n²) implementation is fine for
the foundation-session ≤6,800-chunk corpus; Phase B may swap in MinHash
LSH when corpus scales.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Set

from .edit_distance import GuardResult


DEFAULT_DRIFT_TOLERANCE = 0.05  # 5% delta vs heuristic per Charter §3.3 row 8
DEFAULT_NEAR_DUP_JACCARD = 0.9  # Per Charter §3.3 row 8
DEFAULT_SHINGLE_K = 3  # 3-gram token shingles

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _shingles(text: str, k: int = DEFAULT_SHINGLE_K) -> Set[str]:
    """k-gram token shingles as a set of joined strings."""
    tokens = _TOKEN_RE.findall(text.lower())
    if len(tokens) < k:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


@dataclass(frozen=True)
class DedupRatioReport:
    """Corpus-level dedup-ratio measurement (Charter §3.3 row 8)."""

    near_duplicate_pairs: int
    total_pairs: int
    ratio: float  # near_duplicate_pairs / total_pairs


def compute_dedup_ratio(
    contents: List[str],
    *,
    near_dup_threshold: float = DEFAULT_NEAR_DUP_JACCARD,
    shingle_k: int = DEFAULT_SHINGLE_K,
) -> DedupRatioReport:
    """Compute the near-duplicate-pair ratio over `contents`.

    For C12 corpus size (≤1000 docs, ≤6800 chunks at v2.16) the O(n²)
    pair scan is acceptable (~23M pairs, ~minute on commodity CPU).
    Phase B may swap in MinHash LSH for larger corpora.
    """
    shingles_per_chunk = [_shingles(c, k=shingle_k) for c in contents]
    n = len(shingles_per_chunk)
    if n < 2:
        return DedupRatioReport(
            near_duplicate_pairs=0, total_pairs=0, ratio=0.0
        )
    total_pairs = n * (n - 1) // 2
    near = 0
    for i in range(n):
        si = shingles_per_chunk[i]
        if not si:
            continue
        for j in range(i + 1, n):
            sj = shingles_per_chunk[j]
            if not sj:
                continue
            if _jaccard(si, sj) >= near_dup_threshold:
                near += 1
    return DedupRatioReport(
        near_duplicate_pairs=near,
        total_pairs=total_pairs,
        ratio=(near / total_pairs) if total_pairs else 0.0,
    )


def evaluate(
    heuristic_contents: List[str],
    llm_contents: List[str],
    *,
    drift_tolerance: float = DEFAULT_DRIFT_TOLERANCE,
    near_dup_threshold: float = DEFAULT_NEAR_DUP_JACCARD,
    shingle_k: int = DEFAULT_SHINGLE_K,
) -> GuardResult:
    """Corpus-level guard: reject if LLM dedup-ratio exceeds heuristic + tolerance.

    Unlike guards 1-7, this guard takes two corpus-level lists rather
    than one chunk pair. The orchestrator runs it once per build (after
    all per-chunk guards have run) so the dominance criterion can roll
    its result into the Format dominance arithmetic.
    """
    heuristic_report = compute_dedup_ratio(
        heuristic_contents,
        near_dup_threshold=near_dup_threshold,
        shingle_k=shingle_k,
    )
    llm_report = compute_dedup_ratio(
        llm_contents,
        near_dup_threshold=near_dup_threshold,
        shingle_k=shingle_k,
    )
    delta = llm_report.ratio - heuristic_report.ratio
    if delta > drift_tolerance:
        return GuardResult(
            accepted=False,
            guard_name="dedup_ratio",
            reason=(
                f"corpus dedup-ratio drift {delta:.3f} exceeds "
                f"tolerance {drift_tolerance:.3f} "
                f"(heuristic={heuristic_report.ratio:.3f}, "
                f"llm={llm_report.ratio:.3f})"
            ),
            metric_value=delta,
        )
    return GuardResult(
        accepted=True,
        guard_name="dedup_ratio",
        reason="",
        metric_value=delta,
    )
