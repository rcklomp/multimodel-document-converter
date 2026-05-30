# Deferred heuristic: trailing-preposition / subtitle-continuation merger

## Goal (1 sentence)
Recognize a short standalone chunk whose first word is a stopword
preposition (`and`, `of`, `the`, ...) and whose parent heading is set,
and re-type that chunk as a continuation of the parent heading rather
than as a new paragraph.

## Input shape (1 paragraph)
A list of candidate chunks each carrying `content` (str), `chunk_type`
(initially `PARAGRAPH`), and `parent_heading` (Optional[str]). The
target pattern is a chunk with 1<=len(content)<30, single line, no
terminal punctuation, first alphabetic word in a fixed stopword
set, and a non-empty parent_heading distinct from content.

## Output shape (1 paragraph)
The same list of chunks, but matched chunks have their type changed
from `PARAGRAPH` to `HEADING` (or get merged into the prior heading
chunk's content). Content, bbox, page number, and reading order are
otherwise preserved. The downstream effect is that a heading
fragmented across a page break is rejoined into a single semantic
heading entity.

## v2.16 source location
`src/mmrag_v2/engines/pdf_extraction.py::looks_like_subtitle_continuation`
plus the `doc_chunk_to_uir_chunks` call site that consults it
(roughly lines 255-291 + the chunker emission path that reclassifies).

## Applicability
- modality == "text"
- 0 < len(content_stripped) < 40
- non-empty parent_heading (different from content)

## SYSTEM
You are a careful text-normalization assistant. Your single job is to
decide whether a SHORT text chunk is a TRAILING FRAGMENT of the chunk's
parent heading that was split across a page or column break, and if so,
to return that fragment joined to its parent heading.

Definitions:
- A "trailing fragment" begins with a stopword preposition or
  conjunction (`and`, `of`, `for`, `the`, `to`, `in`, `with`, `on`,
  `from`, `by`, `at`, `as`, `or`, `but`, `&`) or with a lowercase
  continuation word.
- The fragment is short (< 40 chars after stripping), single line, no
  terminal sentence punctuation (`.`, `?`, `!`).
- The parent heading and the fragment, joined with a single space,
  read as a single semantic heading.

Output protocol: respond with EXACTLY one JSON object on a single line,
no markdown, no commentary:

  {"action": "merge", "content": "<joined heading text>"}
  {"action": "keep"}
  {"action": "reject"}

Rules:
- `keep` means "leave the chunk content unchanged". DO NOT echo the
  content — only emit `{"action": "keep"}`.
- If the chunk does NOT look like a trailing heading fragment, emit `keep`.
- If you are unsure, emit `keep` — do not invent.
- Never modify content other than this specific merge.
- Never add new text not present in the inputs.

## USER_TEMPLATE
Parent heading: {parent_heading}
Chunk content : {content}

Decide.
