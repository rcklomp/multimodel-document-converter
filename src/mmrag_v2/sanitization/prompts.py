"""Versioned, language-aware prompt templates for LLM sanitization.

Charter: docs/ARCHITECTURE_V3_DRAFT_0.5.md §3.3.

The `prompt_version` field on UIRChunk + the content-pinning cache key
(Charter §7.4) refers to a git hash of the prompt template. This module
exposes the active template + a deterministic hash function so the
cache and the UIRChunk provenance fields agree on what was used.

Foundation-session status: SKELETON. The actual prompt body lands in
Phase B B2 (prompt-engineering spike on 100-chunk sample). Foundation
ships:

    - Template string with content/context placeholders + XML guard tags
      (compatible with guards/prompt_boundary.py).
    - `prompt_version()` returning a short git-hash of the template body
      (so the foundation-session string is itself identifiable).
    - `render(raw, prev, next, lang, page_breadcrumb)` returning the
      final prompt to send to the LLM.

Prompt-template hash change cost (Charter §3.3 prompt-migration cost
note, Draft 0.5): when the template changes, every cache entry becomes
a miss and the entire corpus must be re-sanitized at the B8 cold-cache
cost. This module's `prompt_version()` is the source of truth for that
hash; CI fails builds where the prompt has changed without a
corresponding `B8_COLD_CACHE_COST.md` update.
"""

from __future__ import annotations

import hashlib
from typing import Optional


# Foundation-session template. Phase B B2 replaces the body with a
# spike-validated version on the 100-chunk sample. Changing this body
# bumps the prompt_version hash; see module docstring on the cost.
PROMPT_TEMPLATE = """\
<system_instructions>
You are a document-sanitization assistant. The following XML-delimited
content was extracted from a PDF page. Your job: clean obvious
extraction artifacts (drop-cap orphans, picture classification labels,
OCR noise, broken hyphenation, spurious whitespace) and reconstruct
the text into well-formed Markdown.

Strict constraints:
  - Preserve ALL factual content. No summarization. No paraphrasing.
  - Preserve every number, date, identifier, and named entity verbatim.
  - Preserve every fenced code block byte-for-byte.
  - Preserve the order of any numbered or lettered list.
  - Do NOT emit any of the XML tags below (e.g. </chunk_content>).
  - Detected language: {lang}

The previous chunk's tail and next chunk's head are provided only as
context for heal-across-page-boundaries decisions; do NOT include
their content in your output.
</system_instructions>

<page_breadcrumb>{page_breadcrumb}</page_breadcrumb>
<lang>{lang}</lang>
<prev_chunk>{prev}</prev_chunk>
<next_chunk>{next}</next_chunk>
<chunk_content>
{raw}
</chunk_content>

Return ONLY the cleaned Markdown body of this chunk. Do not echo the
XML tags. Do not add a preamble or postscript.
"""


def prompt_version() -> str:
    """Deterministic short hash of the active prompt template.

    Used as `UIRChunk.sanitizer_prompt_version` AND as a component of
    the content-pinning cache key (Charter §7.4). A change in the
    template body changes this hash, which invalidates every cache
    entry — see module docstring on the B8 cold-cache cost.
    """
    digest = hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
    return digest[:12]


def render(
    *,
    raw: str,
    prev: Optional[str] = None,
    next: Optional[str] = None,  # noqa: A002 (intentional name parallel to "prev")
    lang: Optional[str] = None,
    page_breadcrumb: Optional[str] = None,
) -> str:
    """Render the prompt for one chunk.

    Missing context fields become empty strings rather than the literal
    "None" — that keeps the template predictable for the LLM and keeps
    the prompt-boundary guard contract intact.
    """
    return PROMPT_TEMPLATE.format(
        raw=raw,
        prev=prev or "",
        next=next or "",
        lang=lang or "und",  # ISO 639-3 "undetermined"
        page_breadcrumb=page_breadcrumb or "",
    )
