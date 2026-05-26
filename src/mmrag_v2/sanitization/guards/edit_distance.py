"""Guard 1: Edit-distance ceiling.

Charter §3.3 guard table row 1: ">30% token change → reject". Catches
gross rewrites and fabrications.

Foundation-session status: FUNCTIONAL. Uses python-Levenshtein
(already in project deps via pyproject.toml).

Phase B may tune the threshold per modality (CODE chunks should have
near-zero tolerance because guard 3 hashes them; PROSE chunks may
tolerate higher edit-distance because reflow is normal).
"""

from __future__ import annotations

from dataclasses import dataclass

import Levenshtein


DEFAULT_EDIT_DISTANCE_CEILING = 0.30  # 30% token-level change


@dataclass(frozen=True)
class GuardResult:
    """Outcome of a single guard evaluation.

    Shared across all 8 guards. The orchestrator collects these and
    short-circuits on the first `accepted == False`.
    """

    accepted: bool
    guard_name: str
    reason: str = ""
    metric_value: float = 0.0  # The numeric value the guard measured


def evaluate(
    original: str,
    sanitized: str,
    *,
    ceiling: float = DEFAULT_EDIT_DISTANCE_CEILING,
) -> GuardResult:
    """Reject sanitized output if its edit-distance ratio vs original exceeds ceiling.

    Distance is computed via python-Levenshtein on raw character strings
    (not tokens — token-level alignment is guard 5). The ratio is
    Levenshtein distance / max(len(original), len(sanitized)). A zero-
    length original is treated as "no edit distance applies"
    (accepted = True with metric_value = 0.0).
    """
    if not original:
        return GuardResult(
            accepted=True,
            guard_name="edit_distance",
            reason="original empty; nothing to compare",
            metric_value=0.0,
        )
    distance = Levenshtein.distance(original, sanitized)
    denominator = max(len(original), len(sanitized))
    ratio = distance / denominator if denominator else 0.0
    if ratio > ceiling:
        return GuardResult(
            accepted=False,
            guard_name="edit_distance",
            reason=f"edit-distance ratio {ratio:.3f} exceeds ceiling {ceiling:.3f}",
            metric_value=ratio,
        )
    return GuardResult(
        accepted=True,
        guard_name="edit_distance",
        reason="",
        metric_value=ratio,
    )
