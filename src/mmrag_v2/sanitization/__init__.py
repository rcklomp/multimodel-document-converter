"""V3.0 LLM-native chunk sanitization package.

Charter: docs/ARCHITECTURE_V3_DRAFT_0.5.md §3.3 — Ingestion Pipeline
Stage 2.

Foundation-session status (2026-05-26): SCAFFOLDING ONLY. No production
code path consumes this package yet. Phase B (Charter Cycle 3.1, 18-22d)
implements the orchestrator + GX10 endpoint + prompt-engineering loop
to make this load-bearing.

The interface is fixed in the foundation session so Phase B work can
slot into the contract without redesigning it:

    sanitize_chunk(uir_chunk, mode, config) -> SanitizationResult

with mode ∈ {"off", "llm", "heuristic", "both-and-diff"}.

8-layer guard stack (Charter §3.3):

    1. Edit-distance ceiling     — guards/edit_distance.py    (FUNCTIONAL)
    2. Numeric/entity preserve   — guards/numeric_entity.py   (PARTIAL: regex only,
                                                              spaCy NER deferred)
    3. Code-span hashing         — guards/code_span.py        (FUNCTIONAL)
    4. Order preservation        — guards/order_preservation.py (FUNCTIONAL)
    5. Token-level alignment     — guards/token_alignment.py  (FUNCTIONAL)
    6. Prompt boundary           — guards/prompt_boundary.py  (FUNCTIONAL)
    7. Entity-relation triples   — guards/entity_relation.py  (STUB — spaCy)
    8. Corpus dedup-ratio        — guards/dedup_ratio.py      (FUNCTIONAL)
"""

from .orchestrator import (  # noqa: F401
    SanitizationMode,
    SanitizationResult,
    sanitize_chunk,
)
