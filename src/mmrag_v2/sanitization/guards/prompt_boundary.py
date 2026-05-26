"""Guard 6: Content / prompt boundary delimiters.

Charter §3.3 guard table row 6: "XML-style tags separating instructions
from document content; input-length cap". Catches prompt injection
via document text — adversarial document rewriting its neighbors.

Foundation-session status: FUNCTIONAL.

Two checks:
    1. **Boundary tags survived:** the wrapped chunk content uses
       `<chunk_content>...</chunk_content>` delimiters. If the
       sanitized output contains a closing `</chunk_content>` tag
       (or any other guard-tag) NOT present in the original, an
       injection succeeded and the chunk is rejected.
    2. **Length cap:** input chunks longer than the configured byte
       limit are rejected upfront so a 100 KB chunk cannot smuggle a
       prompt past the context window.

Phase B may extend with allow-list of inert escape sequences or move
to a dedicated structured-prompt mechanism (e.g., role-based isolation
via the OpenAI chat format).
"""

from __future__ import annotations

from .edit_distance import GuardResult


CHUNK_OPEN_TAG = "<chunk_content>"
CHUNK_CLOSE_TAG = "</chunk_content>"
# Other guard tags the orchestrator may wrap context in:
_GUARD_TAGS = (
    "<system_instructions>",
    "</system_instructions>",
    "<prev_chunk>",
    "</prev_chunk>",
    "<next_chunk>",
    "</next_chunk>",
    "<page_breadcrumb>",
    "</page_breadcrumb>",
    "<lang>",
    "</lang>",
    CHUNK_OPEN_TAG,
    CHUNK_CLOSE_TAG,
)


DEFAULT_INPUT_BYTE_CAP = 16 * 1024  # 16 KB per chunk; conservative for FP8-14B 32K ctx


def evaluate(
    original: str,
    sanitized: str,
    *,
    input_byte_cap: int = DEFAULT_INPUT_BYTE_CAP,
) -> GuardResult:
    """Reject on (1) injected guard tag in sanitized, or (2) input exceeds cap."""
    original_bytes = original.encode("utf-8")
    if len(original_bytes) > input_byte_cap:
        return GuardResult(
            accepted=False,
            guard_name="prompt_boundary",
            reason=(
                f"input exceeds byte cap "
                f"({len(original_bytes)} > {input_byte_cap})"
            ),
            metric_value=float(len(original_bytes)),
        )
    # Injection check: any guard tag that appears in sanitized but not in
    # original is evidence of prompt-format escape.
    injected = [
        tag
        for tag in _GUARD_TAGS
        if tag in sanitized and tag not in original
    ]
    if injected:
        return GuardResult(
            accepted=False,
            guard_name="prompt_boundary",
            reason=f"guard tag injected by sanitizer: {injected[0]}",
            metric_value=float(len(injected)),
        )
    return GuardResult(
        accepted=True,
        guard_name="prompt_boundary",
        reason="",
        metric_value=0.0,
    )
