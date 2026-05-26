"""Guard 4: Order preservation.

Charter §3.3 guard table row 4: "Regex-identified ordered-list markers
must appear in same sequence". Catches the LLM reordering procedural
steps, recipes, or algorithms — reordered instructions.

Foundation-session status: FUNCTIONAL.

Markers detected: arabic-numeral lists (1., 2., 3., ...), Roman-numeral
lists (I., II., III., ...), and letter lists (a), b), c)). The guard
checks that the *sequence* of marker tokens in `sanitized` matches the
sequence in `original` (subsequence-equal, not just same set).

Phase B may extend with Markdown-numbered headings (## 1. ... ## 2. ...)
or table-row identifiers depending on what the prompt template emits.
"""

from __future__ import annotations

import re
from typing import List

from .edit_distance import GuardResult


# Each pattern captures a single marker on a line-starting position.
# MULTILINE so ^ matches after newlines.
_ARABIC_RE = re.compile(r"^\s*(\d{1,3})[.)]\s+", re.MULTILINE)
_ROMAN_RE = re.compile(
    r"^\s*([IVXLCDM]{1,8})[.)]\s+(?=[A-Z])",  # uppercase only; require word follow
    re.MULTILINE,
)
_LETTER_RE = re.compile(r"^\s*([a-z])\)\s+", re.MULTILINE)


def _extract_markers(text: str) -> List[str]:
    """Extract ordered-list marker tokens in document order.

    Tags each marker with its kind ("a", "r", "l") to keep different
    families separate — an arabic "1" should not match a letter "a".
    """
    markers: List[tuple] = []  # (position, kind:marker)
    for kind, regex in (
        ("a", _ARABIC_RE),
        ("r", _ROMAN_RE),
        ("l", _LETTER_RE),
    ):
        for match in regex.finditer(text):
            markers.append((match.start(), f"{kind}:{match.group(1)}"))
    markers.sort(key=lambda pair: pair[0])
    return [m for _, m in markers]


def evaluate(original: str, sanitized: str) -> GuardResult:
    """Reject when ordered-list marker sequence differs (set OR order)."""
    original_markers = _extract_markers(original)
    if not original_markers:
        return GuardResult(
            accepted=True,
            guard_name="order_preservation",
            reason="no ordered-list markers in original",
            metric_value=0.0,
        )
    sanitized_markers = _extract_markers(sanitized)
    if original_markers != sanitized_markers:
        # Distinguish set-equal-but-reordered from missing-markers.
        if set(original_markers) == set(sanitized_markers):
            reason = "ordered-list markers reordered"
        else:
            missing = set(original_markers) - set(sanitized_markers)
            added = set(sanitized_markers) - set(original_markers)
            reason = (
                f"ordered-list markers changed: "
                f"missing={sorted(missing)[:5]}, added={sorted(added)[:5]}"
            )
        return GuardResult(
            accepted=False,
            guard_name="order_preservation",
            reason=reason,
            metric_value=1.0,
        )
    return GuardResult(
        accepted=True,
        guard_name="order_preservation",
        reason="",
        metric_value=0.0,
    )
