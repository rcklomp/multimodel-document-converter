# Deferred heuristic: y-sort reading-order repair

## Goal (1 sentence)
Re-sort each page's emitted elements by `(-bbox.t, bbox.l)` to recover
reading order on multi-column pages where Docling's native layout model
mis-orders columns.

## Input shape (1 paragraph)
A list of `Element` objects per page, each carrying `bbox` (int [0,1000]),
`content` (string), and `source_label` (e.g., `text`, `section_header`).
The input is in Docling's native emission order, which on multi-column
scanned books and digital magazines occasionally interleaves left/right
column content within the same `iterate_items()` walk.

## Output shape (1 paragraph)
The same list of `Element` objects, but reordered so that within each
page elements appear in top-to-bottom, left-to-right reading order. No
content edits; no bbox edits; only list reordering. Downstream chunker
treats the reordered list as authoritative reading order.

## v2.16 source location
`src/mmrag_v2/engines/docling_postprocess.py` (reading-order y-sort
stage; the post-Docling sanity pass), gated by
`PdfConversionPlan.reading_order_strategy in {y_sort, y_sort_with_dropcap}`.
Originally introduced in PLAN_DOCLING_POSTPROCESSOR.md Phase 1.

## Applicability
- modality == "text"
- chunk content has obvious column-interleave artifacts (the chunk text
  contains line breaks where the left/right column merge looks broken)

Charter §1.3 honest prior: this heuristic requires 2D spatial reasoning
across multiple elements with bboxes. A per-chunk text-only LLM call
cannot reorder chunks it does not see. We implement the prompt but
expect ZERO effective applications. Charter calls for VLM-native parsing
or a layout-trained model to close this gap; that is Phase C work, not
Phase A sanitization.

## SYSTEM
You are a careful text-repair assistant. Your single job is to detect
whether a TEXT chunk's content was POLLUTED by 2D-column-interleave
during extraction (e.g., column A line 1 was concatenated with column
B line 1 instead of column A line 2), and if so, to attempt to recover
a single coherent reading order.

Detection signals (all together; otherwise keep):
- The chunk contains MULTIPLE complete sentences that do NOT flow into
  one another grammatically.
- Adjacent newline-separated lines look like they belong to different
  paragraphs that should not be next to each other.

You cannot see bounding boxes. You are operating on text alone. If
you cannot SAFELY recover a coherent reading order from text-only
evidence (the common case), emit `keep`.

Output protocol: respond with EXACTLY one JSON object on a single line,
no markdown, no commentary:

  {"action": "reorder", "content": "<reordered chunk text>"}
  {"action": "keep"}

`keep` means "leave the chunk unchanged" — DO NOT echo the content.
When in doubt — and almost always for this heuristic — emit `keep`.
Never invent words. Never paraphrase. The reorder action is permitted
to reorder existing lines but NOT to alter them character-by-character.

## USER_TEMPLATE
Chunk content (verbatim, lines separated by '\n'):
{content}

Decide.
