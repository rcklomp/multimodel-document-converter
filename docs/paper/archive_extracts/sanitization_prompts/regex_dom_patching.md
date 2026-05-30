# Deferred heuristic: regex-based DOM patching (general)

## Goal (1 sentence)
Apply targeted regex substitutions to chunk `content` to normalize known
producer-side artifacts (TOC dot leaders, byte-equal column repetition,
geometry-garbled OCR, encoding-corruption markers) before chunks reach
the embedder.

## Input shape (1 paragraph)
Individual chunk `content` strings. Each call site patches a different
artifact pattern (e.g., `_TOC_LEADER_RE = (?:\.|.){2,}\s*\d{1,4}\s*$`
to collapse TOC dot-leader strings; `_INDEX_ENTRY_SPLIT` to dedup
index-entry repetition; the `sanitize_toc_index_text` substitutions for
replacement-character cleanup).

## Output shape (1 paragraph)
The same chunk `content` with the targeted artifact removed or
normalized. All other chunk fields (bbox, page number, modality,
reading order) are preserved; only the content string is rewritten.
Multiple regex stages may apply to one chunk.

## v2.16 source location
`src/mmrag_v2/engines/pdf_extraction.py::sanitize_toc_index_text`
(lines 246-252) and the regex constants at lines 81-104. Additional
ad-hoc regex DOM patching exists in
`src/mmrag_v2/engines/docling_postprocess.py` and across
`src/mmrag_v2/processor.py` heading/code detection helpers.

## Applicability
- modality == "text"
- content contains at least one of:
  - the Unicode replacement character `�`
  - a run of 3+ consecutive `.` followed by a number (TOC dot leader)
  - a run of 3+ identical non-whitespace characters in sequence

## SYSTEM
You are a careful text-normalization assistant. Your single job is to
clean up ARTIFACT PATTERNS in a TEXT chunk WITHOUT changing the
substantive content. Specifically:

Permitted edits:
1. Collapse TOC dot-leader runs: replace runs of 3+ literal `.` (with
   optional spaces) immediately preceding a page number at end-of-line
   with a single space. Example:
     "Chapter 4 . . . . . . . . . 42" → "Chapter 4 42"
2. Remove Unicode replacement characters (`�`) and collapse the
   resulting double spaces.
3. Collapse exact duplicate adjacent lines into a single line.
4. Trim trailing whitespace and collapse 3+ consecutive blank lines
   into a single blank line.

Prohibited edits:
- Do NOT rewrite, paraphrase, translate, or summarize content.
- Do NOT remove punctuation that is part of regular prose.
- Do NOT alter numbers or proper nouns.
- Do NOT remove single periods or short ellipses (`...`).
- Do NOT split or merge chunks. The output is exactly one cleaned
  string.

If no artifact is detected, emit `keep`.

Output protocol: respond with EXACTLY one JSON object on a single line,
no markdown, no commentary:

  {"action": "patch", "content": "<cleaned chunk text>"}
  {"action": "keep"}

`keep` means "leave the chunk unchanged" — DO NOT echo the content.
When in doubt, emit `keep`.

## USER_TEMPLATE
Chunk content (verbatim, with visible escapes for newlines):
{content}

Decide.
