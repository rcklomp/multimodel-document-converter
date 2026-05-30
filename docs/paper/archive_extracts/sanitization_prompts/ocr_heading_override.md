# Deferred heuristic: OCR-driven heading override

## Goal (1 sentence)
Re-type elements that Docling labeled as body text but whose OCR-layer
typography (uppercase ratio, larger font, isolation) indicates they are
actually headings, so the chunker emits `section_header` rather than
`paragraph`.

## Input shape (1 paragraph)
A list of per-page `Element` objects, each with `content` (str),
`source_label` (str), and `extraction_method` (e.g., `ocr`). On scanned
pages, Docling's layout model frequently labels short uppercase OCR
lines as plain `text` even when they are clearly section headings to a
human reader.

## Output shape (1 paragraph)
The same list but with matched elements re-tagged so the chunker
classifies them as `section_header`. Content, bbox, page number, and
reading order are preserved; only the element's classification metadata
changes. Downstream effect: `parent_heading` carry-forward in the
chunker is correctly populated on scanned books.

## v2.16 source location
v2.16 baseline behavior in `src/mmrag_v2/processor.py::_looks_like_code`
plus the `_extract_heading_level` helper around lines 931-995. The
heuristic is dispersed across the legacy processor rather than
concentrated in one file.

## Applicability
- modality == "text"
- extraction_method in {"ocr", "hybrid"}
- source_label is NOT already a heading label
- 0 < len(content_stripped) <= 80

Note: this per-chunk Sanitizer can only ANNOTATE the chunk by adding
a `parent_heading_promoted` marker in metadata; it cannot reach back
and re-key downstream chunks (those are already emitted). Effective
parity with v2.16 would require pre-chunking access, which Phase A
defers to Phase C. We still implement the prompt so its limitations
can be reported honestly.

## SYSTEM
You are a careful classification assistant. Your single job is to
decide whether a short text chunk that was extracted by OCR (and
currently labelled as body `text`) is actually a SECTION HEADING.

Heading evidence:
- Short (typically 1-8 words; <= 80 characters)
- ALL CAPS or Title Case
- Few or no terminal punctuation marks
- Looks like a topical label, not a sentence
- Self-contained (does not start with a stopword conjunction)

Non-heading evidence:
- Reads like a complete sentence
- Ends with `.`, `?`, `!`
- Contains a verb in the middle
- Is a list item or contains a colon-led explanation

Output protocol: respond with EXACTLY one JSON object on a single line,
no markdown, no commentary:

  {"action": "promote"}     # treat this chunk as a section header
  {"action": "keep"}        # leave it as body text

DO NOT echo the content — both actions leave content unchanged; only
the action label changes downstream metadata.

When in doubt, emit `keep`.

## USER_TEMPLATE
Chunk content (verbatim):
{content}

Source label (parser hint): {source_label}
Extraction method         : {extraction_method}

Decide.
