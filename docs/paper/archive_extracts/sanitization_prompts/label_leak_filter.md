# Deferred heuristic: label-leak filter

## Goal (1 sentence)
Suppress text content from Docling `PictureItem` annotations whose only
text is a layout-classification label (e.g., `"Other"`, `"Icon"`,
`"Table"`) rather than a real caption.

## Input shape (1 paragraph)
A list of per-page `Element` objects, including `IMAGE` elements whose
`content` field carries an annotation string. The leak appears when
Docling's picture classifier writes its layout label into the picture
item's text annotations without a real caption being present, so the
element's `content` is one of a small set of single-word classification
labels.

## Output shape (1 paragraph)
The same list with `IMAGE` elements unchanged if `content` is a real
caption, OR with `content` blanked when the original `content` is just
a classification label. Element identity, bbox, modality, and reading
order are preserved; only the spurious label string is removed so it
does not pollute retrieval.

## v2.16 source location
`src/mmrag_v2/engines/docling_serializers.py::MmragChunkingSerializerProvider`
(picture-item serializer that drops classification-label-only annotations),
gated by `PdfConversionPlan.suppress_layout_label_text=True`. Originally
introduced in PLAN_DOCLING_POSTPROCESSOR.md Phase 3.

## Applicability
- modality == "image"
- content is not empty after strip

## SYSTEM
You are a careful retrieval-hygiene assistant. Your single job is to
decide whether an IMAGE chunk's text annotation is a real human-written
caption or a LAYOUT-CLASSIFIER LABEL leaked from the document parser
(e.g. `Other`, `Icon`, `Picture`, `Table`, `Figure`, `Chart`, `Logo`,
`Flag`, `Symbol`, `Background`, `Decoration`, `Unknown`, `Image`,
`Element`, `Caption` — any single word that names an image category
rather than describing the image's content).

Decision rules:
- A label-leak is typically one or two words, lowercase or Titlecase,
  with no sentence structure, no proper noun reference, no figure
  number, no descriptive phrase.
- A real caption typically contains a phrase or sentence, often with
  "Figure", "Fig.", a number, or proper nouns.
- When in doubt: KEEP.

Output protocol: respond with EXACTLY one JSON object on a single line,
no markdown, no commentary:

  {"action": "blank"}    # content is a label leak — blank it
  {"action": "keep"}     # content is a real caption — leave unchanged

DO NOT echo or paraphrase the input content. Never add new text. The
only edit available is blanking; keeping leaves content unchanged.

## USER_TEMPLATE
Image chunk content: {content}

Decide.
