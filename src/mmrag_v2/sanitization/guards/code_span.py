"""Guard 3: Code-span hashing.

Charter §3.3 guard table row 3: "Text inside ``` fences must be
byte-identical or rejected". Catches the LLM "fixing" code syntax or
reordering statements — silent code corruption.

Foundation-session status: FUNCTIONAL.

Matching rule: identify all triple-backtick fenced blocks in `original`,
extract their bodies, SHA-256 each body, then assert each hash appears
unmodified in `sanitized`. Pre-sanitization code blocks that are
*split* across fences (a single block becoming two) will fail this
guard — which is the correct behavior, because the chunker / extraction
should have already produced clean fenced blocks.

Phase B may relax to allow whitespace normalization inside fences,
provided the relaxation is paired with explicit guard-rejection rate
monitoring per Charter §6.2 OR-clause trigger (b).
"""

from __future__ import annotations

import hashlib
import re
from typing import List

from .edit_distance import GuardResult


# Triple-backtick code block; optional language tag on the opening fence.
# Non-greedy body match; DOTALL because fences span lines.
_FENCE_RE = re.compile(
    r"```[A-Za-z0-9_+\-]*\n(.*?)\n```",
    re.DOTALL,
)


def _extract_fenced_bodies(text: str) -> List[str]:
    return _FENCE_RE.findall(text)


def _hash_bodies(bodies: List[str]) -> List[str]:
    return [hashlib.sha256(b.encode("utf-8")).hexdigest() for b in bodies]


def evaluate(original: str, sanitized: str) -> GuardResult:
    """Reject when any fenced code body from original is mutated in sanitized."""
    original_bodies = _extract_fenced_bodies(original)
    if not original_bodies:
        return GuardResult(
            accepted=True,
            guard_name="code_span",
            reason="no fenced code blocks in original",
            metric_value=0.0,
        )
    sanitized_bodies = _extract_fenced_bodies(sanitized)
    original_hashes = set(_hash_bodies(original_bodies))
    sanitized_hashes = set(_hash_bodies(sanitized_bodies))
    missing = original_hashes - sanitized_hashes
    if missing:
        missing_ratio = len(missing) / len(original_hashes)
        return GuardResult(
            accepted=False,
            guard_name="code_span",
            reason=(
                f"{len(missing)} of {len(original_hashes)} fenced code "
                "block(s) mutated or removed"
            ),
            metric_value=missing_ratio,
        )
    return GuardResult(
        accepted=True,
        guard_name="code_span",
        reason="",
        metric_value=0.0,
    )
