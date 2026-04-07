# Plan: v2.7 — Document Understanding Layer

## Context

v2.6 delivered HybridChunker integration, VLM transcription for scanned docs,
multi-format support, and 10/10 smoke test pass. The remaining issues are
Docling extraction limitations that can't be fixed in post-processing without
overfitting. v2.7 moves from text-based heuristics to layout-aware structural
mapping.

Source: Gemini Pro audit recommendations, April 2026.

## Features

### 1. Cross-Chunk Semantic Stitching

**Problem:** Orphan prepositions at chunk boundaries ("BY" at end of credits
chunk, separated from "Mary GrandPré" in the next chunk).

**Approach:** When a chunk ends with a semantically "hungry" word (preposition:
BY, FOR, OF, WITH, FROM) and the next chunk starts with a proper noun or title,
pull the preposition into the next chunk.

**Why it's not overfitting:** Prepositions are structurally incomplete without
their object. "VOID" and "END" are self-contained. This is a syntactic rule,
not a character-count heuristic.

**Implementation:** Post-processing pass after mid-sentence merger. Check
last word of each chunk against a preposition list. Check first word of next
chunk for capitalization (proper noun signal). Language-agnostic — prepositions
exist in all languages.

### 2. Vision-Aided Front Matter Detection

**Problem:** Docling misidentifies author names and publisher names as headings
on front matter pages ("J. K. Rowling" as parent_heading).

**Approach:** When a potential heading is on a page that:
- Has `shadow` or `docling` figure extractions (cover art, logos)
- Is before the first "Chapter" heading
- Has no numbered section pattern

Default the parent heading to "Front Matter" instead of using the detected
text as a heading.

**Why it's not overfitting:** Every book has front matter (title, copyright,
dedication, TOC) before content starts. Detecting the boundary between front
matter and content is a structural pattern.

**Implementation:** After heading inference, scan for the first chapter-like
heading. Everything before it with non-chapter headings gets re-labeled as
"Front Matter".

### 3. Domain-Specific Search Priority

**Problem:** `search_priority: "high"` on all text chunks including copyright
notices, printer information, ISBN numbers.

**Approach:** Move priority logic from the converter to the ingestor. Use
`document_domain` to apply domain-specific rules:
- Literature: pages before TOC → "low"
- Academic: references section → "medium"
- Technical: index/appendix → "medium"

**Why it's not overfitting:** Priority is a RAG retrieval concern, not a
conversion concern. The converter shouldn't decide what's important — the
ingestor/RAG system should, based on document structure metadata.

**Implementation:** In `ingest_to_qdrant.py`, add a priority field to the
Qdrant payload based on page position, heading context, and domain.

### 4. Coordinate Normalization Audit

**Problem:** Potential inconsistency between text and image bounding boxes
reported by Gemini (later confirmed to both use [0,1000] — may be a
non-issue, but worth auditing).

**Approach:** Add a final validation pass that checks all bbox values are
in [0,1000] and flag any that aren't. Already partially done by
QA-CHECK-04 / REQ-COORD-01.

**Implementation:** Enhance `qa_universal_invariants.py` to report bbox
distribution statistics per modality.

### 5. Docling Pipeline Options Per Profile

**Problem:** Docling treats all PDFs identically. Our profiles only affect
post-processing, not extraction.

**Approach:** Pass profile-specific Docling options:
- `technical_manual`: `do_code_enrichment=True`, `TableFormerMode.ACCURATE`
- `digital_magazine`: `do_picture_classification=True`
- `scanned_degraded`: `force_full_page_ocr=True`
- All profiles: `do_cell_matching=False` (already done)

**Implementation:** In `processor.py _get_converter()`, read profile from
intelligence_metadata and configure pipeline_options accordingly.

### 6. Contextual Retrieval (Anthropic approach)

**Problem:** Chunks lose context when embedded in isolation.

**Approach:** Per Anthropic's research, prepend 50-100 tokens of
chunk-specific context before embedding. HybridChunker's `contextualize()`
does this partially — v2.7 should ensure the contextualized text is what
gets embedded, not overwritten by the refiner.

**Implementation:** Fix the refiner/contextualize ordering (partially done
in v2.6), ensure `ingest_to_qdrant.py` uses contextualized text.

## Success Criteria

| Metric | v2.6 | v2.7 Target |
|--------|------|-------------|
| Front matter heading accuracy | ~60% (Docling misdetects) | 95%+ |
| search_priority accuracy | 0% (all "high") | 80%+ |
| Orphan prepositions | Present | 0 |
| Smoke test | 10/10 | 10/10 |

## Timeline

- Features 1-3: 1-2 days
- Features 4-5: 0.5 day
- Feature 6: 0.5 day
- Testing + regression: 1 day

## References

- Gemini Pro audit, April 7, 2026
- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [Docling Pipeline Options](https://docling-project.github.io/docling/reference/pipeline_options/)
- [Docling Issue #287: Heading Hierarchy](https://github.com/docling-project/docling/issues/287)

## Updated Architecture (from Gemini Pro, April 7 2026)

### Principle: Stop deleting artifacts. Start validating multimodally.

The v2.6 approach of string-length rules and hardcoded skips is overfitting.
The v2.7 approach uses OCR confidence, VLM descriptions, and POS tagging
as validation signals — not just text pattern matching.

### 1. CorruptionInterceptor (replaces threshold-based refiner bypass)
- Per-bbox OCR patching: when a chunk contains /C211 or fails dictionary
  token ratio, re-extract ONLY that bbox via OCR
- Keep HybridChunker structure (headings, tables, hierarchy)
- Replace only the corrupted text spans

### 2. POS Boundary Logic (replaces character-count orphan stripping)
- Check if trailing word is a "Hungry Operator" (BY, FOR, OF, WITH)
- If next chunk starts with Proper Noun → merge operator into next chunk
- "END", "VOID", "N/A" are Nouns → stay where they are
- Language-agnostic: prepositions are structurally incomplete without objects

### 3. Vision-Gated Hierarchy (replaces quote/credit pattern filters)
- When heading detected on high-image-density page, check VLM description
- If description contains "Cover", "Logo", "Sketch" → demote heading to
  "Front Matter" via MetadataRefiner pass
- Uses multimodal signals, not text heuristics

### 4. Content-Type Classification (replaces hardcoded search_priority)
- Lightweight classifier or regex-weighted score per chunk
- High density of legal/boilerplate tokens → search_priority: "low"
- Works globally for all books and papers, not document-specific
