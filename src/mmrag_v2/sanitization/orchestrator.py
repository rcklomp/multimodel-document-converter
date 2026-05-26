"""Sanitization orchestrator: dispatches LLM vs heuristic vs both-and-diff.

Charter: docs/ARCHITECTURE_V3_DRAFT_0.5.md §3.3.

Mode flag semantics (Charter §3.3 #4):
    - "off"             : no sanitization — raw UIR chunks emitted (v2.16
                          equivalent, used as a regression baseline).
    - "llm"             : LLM sanitization only — heuristics skipped
                          (Phase B target).
    - "heuristic"       : existing heuristic stack only — LLM skipped
                          (v2.16 behavior preserved; default during the
                          dual-write window).
    - "both-and-diff"   : both run, output compared; disagreement logged
                          (validation mode).

Diff predicate (Charter §3.3): two sanitization outputs "differ" when
their token-level Levenshtein distance exceeds 5% of the shorter
output's token count. Whitespace-only differences and Unicode
NFC/NFD normalization differences are excluded before comparison.

Foundation-session status: stub orchestrator. The "off" path is the
only branch that does anything real (returns the chunk unchanged).
The other branches return the raw content with sanitization_status =
"skipped:not_implemented_phase_a_foundation" — Phase B replaces
those branches with real implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class SanitizationMode(Enum):
    """Charter §3.3 #4: --sanitize-mode flag values."""

    OFF = "off"
    LLM = "llm"
    HEURISTIC = "heuristic"
    BOTH_AND_DIFF = "both-and-diff"


@dataclass
class SanitizationResult:
    """Result of sanitizing one UIR chunk.

    Maps onto UIRChunk's provenance contract (Charter §3.2):
        - `content` is the authoritative value to write back to the chunk.
        - `content_original` is the raw extraction (always populated when
          sanitization was attempted).
        - `content_sanitized` is the LLM's output (populated even on
          rejection for guard-stack debugging).
        - `status` follows the same vocabulary as UIRChunk.sanitization_status:
          "accepted" | "rejected:<guard>" | "skipped:<reason>" | "not_applied".
    """

    content: str
    status: str = "not_applied"
    content_original: Optional[str] = None
    content_sanitized: Optional[str] = None
    rejected_by_guards: List[str] = field(default_factory=list)
    model_id: Optional[str] = None
    prompt_version: Optional[str] = None


def sanitize_chunk(
    *,
    raw_content: str,
    mode: SanitizationMode = SanitizationMode.OFF,
    context: Optional[dict] = None,
) -> SanitizationResult:
    """Apply sanitization to a single chunk's raw content.

    Parameters
    ----------
    raw_content : str
        The extracted chunk content prior to sanitization.
    mode : SanitizationMode
        Dispatch mode (see Charter §3.3 #4).
    context : Optional[dict]
        Per-chunk context (previous-chunk snippet, next-chunk snippet,
        detected language, page breadcrumb). Foundation session: unused;
        Phase B: passed to the LLM prompt template and the content-pinning
        cache key.

    Returns
    -------
    SanitizationResult

    Notes
    -----
    Foundation-session implementation:
        - OFF              → return raw content unchanged, status="not_applied"
        - HEURISTIC        → return raw content, status="skipped:foundation_session"
        - LLM              → return raw content, status="skipped:foundation_session"
        - BOTH_AND_DIFF    → return raw content, status="skipped:foundation_session"

    Phase B replaces the LLM and BOTH_AND_DIFF branches with real
    implementations; HEURISTIC keeps the v2.16 heuristic stack as the
    fallback (dual-write).
    """
    _ = context  # foundation-session: reserved for Phase B prompt context

    if mode is SanitizationMode.OFF:
        return SanitizationResult(content=raw_content, status="not_applied")

    # Foundation-session stub: every non-OFF mode emits a sentinel so the
    # downstream pipeline can see that sanitization was requested but no
    # work was done yet. Phase B replaces this with real dispatch.
    return SanitizationResult(
        content=raw_content,
        status="skipped:foundation_session",
        content_original=raw_content,
    )
