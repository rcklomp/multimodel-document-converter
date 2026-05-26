"""8-layer guard stack for LLM sanitization.

Charter: docs/ARCHITECTURE_V3_DRAFT_0.5.md §3.3.

Each guard implements `evaluate(original: str, sanitized: str, **kwargs)
-> GuardResult` and either accepts or rejects the sanitized output.
The orchestrator runs them in order; the first rejection short-circuits
the chain (Phase B will decide whether parallel-vs-serial is the
production layout).

Status per guard:

    1. edit_distance.py        — FUNCTIONAL (Levenshtein-based)
    2. numeric_entity.py       — PARTIAL  (regex numbers/dates;
                                          spaCy NER deferred to Phase B
                                          with optional spaCy dependency)
    3. code_span.py            — FUNCTIONAL (regex fence + SHA-256)
    4. order_preservation.py   — FUNCTIONAL (regex ordered-list markers)
    5. token_alignment.py      — FUNCTIONAL (token Levenshtein)
    6. prompt_boundary.py      — FUNCTIONAL (XML delimiters + length cap)
    7. entity_relation.py      — STUB    (requires spaCy dep parse)
    8. dedup_ratio.py          — FUNCTIONAL (Jaccard shingles; corpus-level)
"""

from .edit_distance import GuardResult  # noqa: F401
from .edit_distance import evaluate as evaluate_edit_distance  # noqa: F401
from .code_span import evaluate as evaluate_code_span  # noqa: F401
from .order_preservation import evaluate as evaluate_order_preservation  # noqa: F401
from .token_alignment import evaluate as evaluate_token_alignment  # noqa: F401
from .prompt_boundary import evaluate as evaluate_prompt_boundary  # noqa: F401
from .numeric_entity import evaluate as evaluate_numeric_entity  # noqa: F401
from .entity_relation import evaluate as evaluate_entity_relation  # noqa: F401
from .dedup_ratio import evaluate as evaluate_dedup_ratio  # noqa: F401
