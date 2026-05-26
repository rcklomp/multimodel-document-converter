"""Guard 5: Token-level alignment (Levenshtein, not just count delta).

Charter §3.3 guard table row 5: "Token-level alignment (Levenshtein
distance, not just count delta)". Catches reorderings that preserve
token count — the blind spot of guard 1's character-level edit-distance.

Foundation-session status: FUNCTIONAL.

Tokenization is intentionally simple: whitespace + light punctuation
split. Phase B may swap in a real tokenizer (the same one the omlx
embedder uses) to make the metric align with what downstream retrieval
sees. For the guard contract that does not change — only the
calibrated threshold.
"""

from __future__ import annotations

import re
from typing import List

import Levenshtein

from .edit_distance import GuardResult


DEFAULT_TOKEN_ALIGNMENT_CEILING = 0.30  # 30% token-level Levenshtein

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    """Whitespace + light-punctuation tokenization (foundation default)."""
    return _TOKEN_RE.findall(text)


def evaluate(
    original: str,
    sanitized: str,
    *,
    ceiling: float = DEFAULT_TOKEN_ALIGNMENT_CEILING,
) -> GuardResult:
    """Reject when token-level Levenshtein distance ratio exceeds ceiling.

    `Levenshtein.distance(seq1, seq2)` accepts list-of-string arguments
    and computes edit distance over the token sequence, treating each
    token as an atomic unit (no character-level recursion). Distance is
    normalized by max(len(original_tokens), len(sanitized_tokens)).
    """
    original_tokens = _tokenize(original)
    if not original_tokens:
        return GuardResult(
            accepted=True,
            guard_name="token_alignment",
            reason="original has no tokens",
            metric_value=0.0,
        )
    sanitized_tokens = _tokenize(sanitized)
    distance = Levenshtein.distance(original_tokens, sanitized_tokens)
    denominator = max(len(original_tokens), len(sanitized_tokens))
    ratio = distance / denominator if denominator else 0.0
    if ratio > ceiling:
        return GuardResult(
            accepted=False,
            guard_name="token_alignment",
            reason=(
                f"token-Levenshtein ratio {ratio:.3f} exceeds ceiling {ceiling:.3f}"
            ),
            metric_value=ratio,
        )
    return GuardResult(
        accepted=True,
        guard_name="token_alignment",
        reason="",
        metric_value=ratio,
    )
