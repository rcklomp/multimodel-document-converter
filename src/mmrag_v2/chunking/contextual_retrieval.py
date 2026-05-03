"""Contextual Retrieval — embedding-text builder per Anthropic research.

Reference: https://www.anthropic.com/news/contextual-retrieval

Design invariants (mirror docs/DECISIONS.md "Contextual Retrieval"):

- AGENT-CONTEXTUAL-01 — Content immutability. The canonical
  ``IngestionChunk.content`` is never mutated. The prefixes live in a
  separate, optional embedding-time field (``contextualized_text``) that is
  never read by QA, source-text validation, refiner threshold logic, or
  any chunk creator.
- AGENT-CONTEXTUAL-02 — Single embed-time builder. The only function
  allowed to assemble contextualized text is
  :func:`build_contextualized_text`. Importers are: the embedding lane in
  ``scripts/ingest_to_qdrant.py``, ``tests/test_contextual_retrieval.py``,
  and (optionally) a future RAG adapter — nothing else.
- AGENT-CONTEXTUAL-03 — QA isolation. Markers ``[Context: …]``,
  ``[Heading: …]``, ``[Previous: …]``, ``[Next: …]``, and
  ``[Modality: …]`` MUST NOT appear in ``IngestionChunk.content``,
  ``metadata.refined_content``, the payload ``text`` field that goes into
  Qdrant, or any artifact that is fed back into ``qa_conversion_audit.py``,
  ``qa_universal_invariants.py``, or ``token_validator.py``.
- AGENT-CONTEXTUAL-04 — Length budget. Per Anthropic, target ~50–100
  tokens (~200–400 chars) of context. Cap each ``prev_text_snippet`` and
  ``next_text_snippet`` to ``MAX_CONTEXT_CHARS = 300``. Truncate; do not
  reflow. Truncation is on a Unicode code-point boundary (Python ``str``
  slicing operates on code points), so it never emits a bare UTF-8
  continuation byte.
- AGENT-CONTEXTUAL-05 — Image lane untouched. Image chunks already embed
  via ``embed_image()`` with the visual description as fallback. The
  contextualization path is for ``modality in {"text", "table"}`` only;
  the ingestor does not call this function for image chunks. Callers
  *may* pass ``modality="image"`` if they choose, but the production
  ingest lane does not.
- AGENT-CONTEXTUAL-06 — Refiner ordering. The refiner runs *before*
  contextualization. The ingestor reads ``metadata.refined_content``
  first and falls back to ``chunk["content"]``; the contextualized
  string is never re-fed into the refiner.
- AGENT-CONTEXTUAL-07 — Cache key safety. If/when an embedding cache is
  added downstream, it MUST key on the contextualized string actually
  sent to the embedder, not on raw ``content``. Otherwise toggling
  ``--no-contextual`` returns stale vectors. (No embedding cache exists
  in this repo today; only ``vision_manager`` caches VLM responses.)
"""

from __future__ import annotations

from typing import Optional


# Maximum characters of prev/next neighbor context to prepend.
# Per Anthropic research: target ~50–100 tokens ≈ 200–400 chars (English prose).
# Truncation applies to ``prev_text_snippet`` and ``next_text_snippet`` ONLY;
# breadcrumb levels and the parent heading are not truncated by this builder.
MAX_CONTEXT_CHARS: int = 300


def build_contextualized_text(
    content: str,
    *,
    breadcrumb_path: Optional[list[str]] = None,
    parent_heading: Optional[str] = None,
    prev_text_snippet: Optional[str] = None,
    next_text_snippet: Optional[str] = None,
    modality: str = "text",
) -> str:
    """Build embedding-time text by prepending hierarchical + neighbor context.

    Pure function: no I/O, no logging, no global state. Empty/whitespace-only
    fields are skipped silently — never raises. The canonical ``content`` is
    never mutated and appears verbatim as the final line of the result.

    Order of assembled prefixes (each on its own line, before ``content``):
      1. ``[Context: A > B > C]`` — non-empty breadcrumb levels, joined with
         ``" > "`` after stripping each level. Whitespace-only levels are
         dropped.
      2. ``[Heading: <parent_heading.strip()>]``
      3. ``[Previous: <prev_text_snippet.strip()[:MAX_CONTEXT_CHARS]>]``
      4. ``[Next: <next_text_snippet.strip()[:MAX_CONTEXT_CHARS]>]``
      5. ``[Modality: <modality>]`` only when ``modality not in {"", "text"}``.
      6. The canonical ``content`` (verbatim, last line; may be empty).

    UTF-8 safety: the snippet truncation slices the already-decoded ``str``,
    which is code-point indexed in CPython, so a multi-byte character is
    never split mid-sequence.

    Headings and breadcrumb levels are *not* truncated by design — chapter
    headings are usually concise; the schema already caps
    ``parent_heading`` at 200 chars and breadcrumb levels at 80 chars.

    Args:
        content: Canonical or refined chunk text. Empty string is allowed.
        breadcrumb_path: Hierarchical breadcrumb (e.g. ["Ch 1", "Sec 2"]).
            ``None`` or empty list is treated as no breadcrumb.
        parent_heading: Nearest parent heading for grounding.
        prev_text_snippet: Preceding chunk text for continuity.
        next_text_snippet: Following chunk text for continuity.
        modality: Chunk modality ("text", "table", "image"). Only emits a
            ``[Modality: …]`` marker for non-text, non-empty values.

    Returns:
        Contextualized embedding string. Always ends with the canonical
        ``content`` (which may itself be the empty string).
    """
    parts: list[str] = []

    # 1. Breadcrumb hierarchy — drop whitespace-only levels per-level.
    if breadcrumb_path:
        levels = [str(level).strip() for level in breadcrumb_path]
        levels = [level for level in levels if level]
        if levels:
            parts.append(f"[Context: {' > '.join(levels)}]")

    # 2. Parent heading
    if parent_heading and parent_heading.strip():
        parts.append(f"[Heading: {parent_heading.strip()}]")

    # 3. Previous neighbor context
    if prev_text_snippet and prev_text_snippet.strip():
        snippet = prev_text_snippet.strip()[:MAX_CONTEXT_CHARS]
        parts.append(f"[Previous: {snippet}]")

    # 4. Next neighbor context
    if next_text_snippet and next_text_snippet.strip():
        snippet = next_text_snippet.strip()[:MAX_CONTEXT_CHARS]
        parts.append(f"[Next: {snippet}]")

    # 5. Modality marker — emit only for explicit non-text modalities.
    if modality and modality != "text":
        parts.append(f"[Modality: {modality}]")

    # 6. Canonical content (last line; may be empty).
    parts.append(content)

    return "\n".join(parts)
