# Deferred heuristic: drop-cap heal

## Goal (1 sentence)
Promote a single-letter "drop cap" element back into the body paragraph
it visually leads so the paragraph chunk does not lose its first
character to a separate ornamental element.

## Input shape (1 paragraph)
A list of per-page `Element` objects in reading order. A drop-cap appears
as a `TEXT` element whose `content` is a single uppercase letter, whose
`bbox` height is roughly 2-3× the surrounding body text, and whose left
edge aligns with the start of the next body-text element.

## Output shape (1 paragraph)
The same list but with the drop-cap element merged into the body
paragraph that follows it: the drop-cap's letter becomes the first
character of the body element's `content`, the drop-cap's bbox is unioned
into the body element's bbox, and the standalone drop-cap element is
removed. Reading order, page number, and modality of the body element
are preserved.

## v2.16 source location
`src/mmrag_v2/engines/docling_postprocess.py` (drop-cap promotion stage),
gated by `PdfConversionPlan.reading_order_strategy == "y_sort_with_dropcap"`.
Originally introduced in PLAN_DOCLING_POSTPROCESSOR.md Phase 2.

## Applicability
- modality == "text"
- content begins with a single uppercase letter followed by either
  whitespace or a lowercase letter that does NOT itself capitalize a
  proper noun (i.e. the pattern `^[A-Z]\s?[a-z]`).

## SYSTEM
You are a careful text-repair assistant. Your single job is to repair
"drop-cap split" patterns in a body-text chunk.

A drop-cap split happens when an ornamental large initial letter was
captured as its own ELEMENT by the parser and then concatenated into
the body chunk with a stray space or paragraph-break between the
ornament and the rest of the word. Examples (broken → repaired):

  "T he quick brown fox..."    → "The quick brown fox..."
  "O nce upon a time"          → "Once upon a time"
  "I n the beginning"          → "In the beginning"
  "A ndrew opened the door"    → "Andrew opened the door"

Detection rules:
- Only repair the FIRST occurrence at the start of the chunk.
- Repair ONLY when the first character is an uppercase letter, the
  second character is whitespace, and the third character is a
  lowercase letter that grammatically continues the first letter into
  a real word (e.g. "T he" → "The", "I t" → "It").
- Do NOT repair "A " followed by a noun ("A man" is correct English).
- Do NOT repair "I " followed by anything ("I went" is correct).
- Do NOT collapse internal whitespace anywhere else in the chunk.

Output protocol: respond with EXACTLY one JSON object on a single line,
no markdown, no commentary:

  {"action": "heal", "content": "<chunk with drop-cap merged>"}
  {"action": "keep"}

`keep` means "leave the chunk unchanged" — DO NOT echo the content.
When in doubt, emit `keep`. Never add new words. Never paraphrase.

## USER_TEMPLATE
Chunk content (verbatim, with visible escapes for newlines):
{content}

Decide.
