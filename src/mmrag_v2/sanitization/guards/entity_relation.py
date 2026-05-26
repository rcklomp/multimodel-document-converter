"""Guard 7: Entity-relation triple preservation.

Charter §3.3 guard table row 7 (NEW in 0.4): extract (subject,
predicate, object) triples via spaCy dependency parse before and after;
any *added* triple in the sanitized output is rejected as a
hallucination signal. Removed triples are flagged as warnings (the LLM
may have removed an extraction artifact masquerading as a relation),
not auto-rejected.

Foundation-session status: STUB.

Rationale for stub: spaCy + `en_core_web_sm` is not in current project
dependencies. Adding it requires user sign-off (~150 MB download, new
runtime dependency, optional GPU-accelerated transformer variant).

When Phase B is ready to enable this guard:

    1. Add `spacy>=3.7` + `spacy-lookups-data` to `pyproject.toml`.
    2. Run `python -m spacy download en_core_web_sm` (or `xx_sent_ud_sm`
       for multilingual coverage given the German corpus presence).
    3. Replace this stub's `evaluate()` body with the real
       dependency-parse triple-extraction logic.
    4. Tune the "added vs removed" thresholds against the 50-chunk
       golden set built in Phase B task B2.

Until then, `evaluate()` always accepts (returns a passing GuardResult
with `metric_value=-1.0` to indicate "guard did not execute"). This
matches the Charter §3.3 sentinel-chunk accounting: a deferred guard
is observable in the result, not silent.
"""

from __future__ import annotations

from .edit_distance import GuardResult


GUARD_STATUS = "deferred:requires_spacy"


def evaluate(original: str, sanitized: str) -> GuardResult:
    """Stub: always accepts. Real implementation requires spaCy.

    See module docstring for the Phase B activation procedure.
    """
    _ = (original, sanitized)
    return GuardResult(
        accepted=True,
        guard_name="entity_relation",
        reason=GUARD_STATUS,
        metric_value=-1.0,
    )
