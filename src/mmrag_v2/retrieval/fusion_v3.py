"""V3 fusion helpers: leg-skip re-normalization + RetrievalDebugPayload.

Charter: docs/ARCHITECTURE_V3_DRAFT_0.5.md §3.4 #6 (leg-skip
re-normalization) + §7.8 (RetrievalDebugPayload per-query inspection).

These are added as a separate module (`fusion_v3.py`) instead of
extending `retrieval/pipeline.py` directly so that the v2.16
production pipeline stays untouched in the foundation session. Phase C
task C4 wires `retrieve_hybrid_visual()` and consumes these helpers.

Foundation-session status: FUNCTIONAL math + dataclass. No production
caller yet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Charter §3.4 #5 PROSE-default visual floor:
PROSE_DEFAULT_VISUAL_WEIGHT = 0.1


def renormalize_on_leg_skip(
    weights: Dict[str, float],
    *,
    skipped_legs: List[str],
) -> Dict[str, float]:
    """Charter §3.4 #6: L2-normalize remaining weights when a leg is skipped.

    Without re-normalization, scores from text-only fusion aren't
    comparable to scores when visual was present, which breaks any
    score-based threshold downstream.

    Example (Charter §3.4 #6):
        PROSE weights (1.0, 1.0, 0.1) with visual skipped
        → remaining (1.0, 1.0) → L2-normalized to (0.707, 0.707).

    Returns a new dict; does not mutate `weights`. Skipped legs are
    removed entirely from the returned dict (they would contribute 0
    to fusion either way; removing them prevents accidental re-use).
    """
    remaining = {
        leg: weight
        for leg, weight in weights.items()
        if leg not in skipped_legs
    }
    if not remaining:
        return {}
    # L2 norm
    squared_sum = sum(w * w for w in remaining.values())
    if squared_sum == 0.0:
        # Degenerate: all remaining weights are zero. Return as-is to
        # avoid divide-by-zero; downstream will surface this via an
        # empty top-K rather than crashing.
        return dict(remaining)
    norm = math.sqrt(squared_sum)
    return {leg: weight / norm for leg, weight in remaining.items()}


# ---------------------------------------------------------------------------
# RetrievalDebugPayload (Charter §7.8, Draft 0.5)
# ---------------------------------------------------------------------------


@dataclass
class RetrievalDebugPayload:
    """Per-query retrieval debug data. Charter §7.8.

    ~1–2 KB per query; not logged by default. Attached to each
    `RetrievalResult` only when `return_debug_payload=True`; the RAG
    application logs it on demand for specific queries without enabling
    global tracing (which v2.16 has via `--log-fusion-trace` but is
    off by default in production).

    Fields:
        leg_scores               — per-leg raw scores keyed by chunk_id /
                                   page_id. {"dense": {chunk_id: score, ...},
                                   "sparse": {chunk_id: score, ...},
                                   "visual": {page_id: score, ...}}
        weights_applied          — profile-conditional weights actually
                                   used (after leg-skip re-normalization)
        legs_skipped             — which legs were unavailable
        fusion_input             — (chunk_id, fused_score) before rerank
        bounded_join_decisions   — (page_id, chunk_id_selected, top_n_rank)
                                   — bounded join trace per Charter §3.4 #4
        rerank_input             — chunk_ids handed to ModernBERT
        rerank_output            — (chunk_id, rerank_score) after rerank
        fusion_vs_rerank_flips   — (fusion_rank, rerank_rank) for chunks
                                   that moved
        profile_used             — PROSE / DIAGRAM / FORM / TABLE
        visual_collection_pin    — e.g., "vidore/colqwen2.5-v0.2 @ render_dpi=200"
        timing_ms                — per-leg timing breakdown
    """

    leg_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    weights_applied: Dict[str, float] = field(default_factory=dict)
    legs_skipped: List[str] = field(default_factory=list)
    fusion_input: List[Tuple[str, float]] = field(default_factory=list)
    bounded_join_decisions: List[Tuple[str, str, int]] = field(default_factory=list)
    rerank_input: List[str] = field(default_factory=list)
    rerank_output: List[Tuple[str, float]] = field(default_factory=list)
    fusion_vs_rerank_flips: List[Tuple[int, int]] = field(default_factory=list)
    profile_used: Optional[str] = None
    visual_collection_pin: str = ""
    timing_ms: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Bounded page→chunk join (Charter §3.4 #4)
# ---------------------------------------------------------------------------


DEFAULT_TOP_N_CHUNKS_PER_PAGE = 3  # Charter §3.4 #4 default


def bounded_page_chunk_join(
    *,
    visual_page_ranks: List[Tuple[str, int]],
    chunks_by_page: Dict[str, List[Tuple[str, float]]],
    top_n: int = DEFAULT_TOP_N_CHUNKS_PER_PAGE,
) -> List[Tuple[str, int]]:
    """Charter §3.4 #4: bounded page→chunk join.

    For each visually-retrieved page, select the top-N chunks on that
    page by their text-leg score. The selected chunks inherit the
    visual page's rank.

    Args:
        visual_page_ranks  — [(page_id, visual_rank)] from MaxSim
        chunks_by_page     — {page_id: [(chunk_id, text_leg_score), ...]}
        top_n              — chunks to keep per page (default 3 per §3.4 #4)

    Returns:
        [(chunk_id, inherited_visual_rank)] — flat list across all pages.
        Ties broken by text-leg score per Charter §3.4 #4.
    """
    result: List[Tuple[str, int]] = []
    for page_id, visual_rank in visual_page_ranks:
        candidates = chunks_by_page.get(page_id, [])
        # Sort by text-leg score descending, take top N.
        candidates_sorted = sorted(
            candidates, key=lambda pair: pair[1], reverse=True
        )[:top_n]
        for chunk_id, _ in candidates_sorted:
            result.append((chunk_id, visual_rank))
    return result
