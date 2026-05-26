"""Guard 2: Numeric / entity preservation.

Charter §3.3 guard table row 2: "All numbers, dates, identifiers, named
entities must appear verbatim in sanitized output". Catches subtle
factual corruption within the edit-distance budget (e.g., "100 mg" →
"10 mg", date changes, ID swaps).

Foundation-session status: PARTIAL.
    - Numbers, percentages, currency, ISO dates: FUNCTIONAL via regex.
    - Named entities (PERSON, ORG, GPE): DEFERRED — requires spaCy NER
      which is not in project dependencies. Phase B can add `spacy` +
      `en_core_web_sm` as optional install or replace with a lighter
      NER (HuggingFace transformer pipeline running on omlx).

Charter §3.3 #2 vs Charter §3.3 #7 (entity-relation triples): #2 checks
that entity *tokens* survive verbatim; #7 checks that no new *relations*
between them appear. Both are needed for defense-in-depth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from .edit_distance import GuardResult


# Patterns calibrated for English/European number + date conventions.
# Phase B will extend with locale-aware variants per chunk's `lang` field.
_NUMBER_RE = re.compile(
    r"""(?x)
    (?<![\w\.])             # not preceded by word char or dot (avoid 'v1.2.3')
    [+-]?                   # optional sign
    (?:\d{1,3}(?:[,\.]\d{3})+|\d+)  # 1,234 / 1.234 / 1234
    (?:[\.,]\d+)?           # optional decimal
    (?:\s?%)?               # optional percent
    (?![\w\.])              # not followed by word char or dot
    """
)
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_HTTP_URL_RE = re.compile(r"https?://[\w\.\-/]+")
_EMAIL_RE = re.compile(r"\b[\w\.\-]+@[\w\.\-]+\.\w+\b")
# Crude identifier pattern: 6+ char alphanumeric token with at least one digit.
# Phase B should pin this per modality (CODE chunks have many such tokens that
# are legitimate identifiers, not data).
_IDENTIFIER_RE = re.compile(r"\b(?=\w*\d)[A-Za-z0-9_\-]{6,}\b")


def _extract_tokens(text: str) -> List[str]:
    """Extract numeric + entity-like tokens that must survive sanitization."""
    tokens: List[str] = []
    for pattern in (_NUMBER_RE, _ISO_DATE_RE, _HTTP_URL_RE, _EMAIL_RE):
        tokens.extend(pattern.findall(text))
    # Identifiers run last so we can drop those already captured by other
    # patterns (e.g., a date matches _ISO_DATE_RE AND _IDENTIFIER_RE).
    other_set = set(tokens)
    for tok in _IDENTIFIER_RE.findall(text):
        if tok not in other_set:
            tokens.append(tok)
    return tokens


def evaluate(original: str, sanitized: str) -> GuardResult:
    """Reject when any number/date/URL/email/identifier from original is missing.

    Foundation-session implementation is the regex tier of Charter §3.3 #2.
    Named-entity preservation (PERSON, ORG, GPE) lands when spaCy or an
    equivalent NER becomes available; that addition is purely additive
    to this guard.
    """
    original_tokens = _extract_tokens(original)
    if not original_tokens:
        return GuardResult(
            accepted=True,
            guard_name="numeric_entity",
            reason="no numeric/entity tokens in original",
            metric_value=0.0,
        )
    missing = [tok for tok in original_tokens if tok not in sanitized]
    missing_ratio = len(missing) / len(original_tokens)
    if missing:
        # Charter §3.3 row 2 is strict: any missing numeric/entity token =
        # reject. Phase B may tier (warn vs reject) per modality.
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", +{len(missing) - 5} more"
        return GuardResult(
            accepted=False,
            guard_name="numeric_entity",
            reason=f"{len(missing)} numeric/entity token(s) missing: {preview}{suffix}",
            metric_value=missing_ratio,
        )
    return GuardResult(
        accepted=True,
        guard_name="numeric_entity",
        reason="",
        metric_value=0.0,
    )
