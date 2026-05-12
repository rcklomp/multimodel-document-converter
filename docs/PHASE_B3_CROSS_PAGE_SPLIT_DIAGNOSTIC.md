# Phase B3 — Cross-page-split defect (2026-05-11)

> **Status:** complete diagnostic; no production code change in this
> step. Names a real defect in the v2.9 "one IngestionChunk per source
> page" split logic that is **not** the section-header-only emission
> defect (Phase B3 Step 2). This document records the evidence and
> the safe halt so a band-aid fix isn't quietly written.

## 1. Symptom

After Phase B1 + B2 + B3 Step 2, two residual MISSING_PAGES rows
remain that are **not** section-header-only and **not** title /
dedication / boilerplate pages:

- `Bourne_RAG_2024` p209 — chapter-end URL/citation list ("Chatbot
  Arena: https://chat.lmsys.org/?leaderboard", "MMLU:
  https://arxiv.org/abs/2009.03300", "MT Bench: …"). Real semantic
  content; high retrieval value (citation queries).
- `Ayeva_Python_Patterns` p4 — author dedication ("I would like to
  thank my parents for their love and support. — Kamon Ayeva").
  Low-but-nonzero retrieval value.

## 2. Evidence of absorption (not omission)

Grep in `output/Bourne_RAG_2024/ingestion.jsonl`:

```
p208 method=hybrid_chunker_pagesplit content="...HotpotQA: https://hotpotqa.github.io/...Chatbot Arena: https://chat.lmsys.org..."
```

The p209 content ("Chatbot Arena: https://chat.lmsys.org…") is in
the JSONL — tagged as `page_number=208`, with
`extraction_method=hybrid_chunker_pagesplit`.

Grep in `output/Ayeva_Python_Patterns/ingestion.jsonl`:

```
p3 method=hybrid_chunker_pagesplit content="ISBN 978-1-83763-961-8\nwww.packtpub.com\nI would like to thank my parents for their love and support.\n- Kamon Ayeva"
```

The p4 dedication is in the JSONL — tagged as `page_number=3`, with
`extraction_method=hybrid_chunker_pagesplit`.

In both cases the **v2.9 Phase 4 cross-page page-split fired**
(extraction_method confirms it), but the resulting IngestionChunk
was assigned the **earlier** page number rather than emitting a
separate chunk for each source page.

## 3. Why this is not the section-header-only defect

The Phase B3 Step 2 fix (`_emit_section_header_only_page_chunks` in
`src/mmrag_v2/processor.py:_emit_section_header_only_page_chunks`)
emits chunks for pages whose **only** Docling items are
`section_header` / `title` labels. Devlin p170, Nagasubramanian p2,
and Sekar p2 fit that shape.

Bourne p209 and Ayeva p4 do **not** — both have `label=text` items
(URL lines on Bourne; dedication sentences on Ayeva). They pass
through the normal HybridChunker path. The defect is downstream:
the cross-page-split assigns the content to the wrong page_number.

A direct probe via `mmrag-v2 process --pages 3,4,5` (Ayeva)
confirms p4 emits one chunk **with `page_number=4`** when processed
in isolation. The misassignment only happens in the full-doc
context, suggesting the cross-page page-split's page-attribution
logic is sensitive to batch boundaries or to the size of adjacent
pages.

## 4. Why no band-aid

A finalize-stage "page-coverage backfill" that emits a synthetic
chunk for each absorbed page would:

1. Conflict with `docs/PLAN_V2.9.md` Phase 1's invariant: "no
   production code may contain or emit `recovery_page_coverage`".
   Phase 1 banned finalize-stage synthetic page-coverage chunks
   precisely because they paper over upstream defects.
2. Produce duplicate content (the p209 URLs would appear in both
   the existing p208-tagged chunk and the backfilled p209 chunk),
   polluting retrieval.
3. Hide the cross-page-split defect from future regression tests
   instead of fixing it at the source.

## 5. Root-cause investigation lane (deferred)

The cross-page-split is in `src/mmrag_v2/processor.py`. Search for
`hybrid_chunker_pagesplit` extraction_method assignment. The fix
likely involves:

- Identifying which Docling `doc_chunk.meta.doc_items` belong to
  each source page.
- For multi-page chunks, emitting one IngestionChunk per source
  page **with the correct page_number for each split slice**,
  rather than always tagging the merged content with the first
  page's number.
- Verifying with regression fixtures from Ayeva p3-5 and Bourne
  p208-210 that each source page receives exactly one chunk.

## 6. Retrieval-value classification for the two affected pages

Per `docs/DECISIONS.md` Retrieval-Value Test:

- **Bourne p209** (URL/citation list): keep the chunk. Citation
  queries are a plausible user query (`"What benchmarks does Bourne
  reference?"`). The chunk should land with `page_number=209`.
  Resolution depends on the cross-page-split fix above.
- **Ayeva p4** (dedication): low retrieval value (no plausible
  user query targets a book's dedication). Could be classified as
  blank-equivalent by extending `_is_intentionally_blank_text` in
  `scripts/qa_full_conversion.py` with a "dedication" pattern.
  However, this requires the same "tight regex with corpus-wide
  false-positive testing" rigor as Phase B3 Step 4 (title-page
  detection). Treat together.

## 7. Scope for v2.9.0-rc1

Bourne p209 (high retrieval value) blocks the broader cross-page-split
defect fix, which is a non-trivial change to a v2.9-shipped path. The
cost/benefit for `v2.9.0-rc1`:

- Affected pages: ~2 confirmed (Bourne p209, Ayeva p4). May affect
  more docs in similar adjacent-short-page shapes (unmeasured).
- Fix size: medium-to-large (touches the cross-page-split logic in
  the chunker output handling).
- Test surface: needs at least 5 fixtures with different
  adjacent-page patterns to validate the fix doesn't break
  currently-correct cross-page handling.

**Recommendation for `v2.9.0-rc1`:**
1. Land Phase B3 Step 2 (closes 3 of 7 missing pages — Devlin p170,
   Nagasubramanian p2, Sekar p2).
2. Land Phase B3 Step 4 (title/dedication blank-equivalent) for
   Greenhouse p2, Ayeva p4 (treat dedication as low-retrieval-value
   blank-equivalent), and any other title-page-shaped residuals.
3. Document Bourne p209 as a **signed v2.10 deferral**:
   `CROSS_PAGE_SPLIT_PAGE_ATTRIBUTION`. The content IS in the
   corpus (just on p208); citation retrieval still works via
   semantic search. The strict-gate `MISSING_PAGES=p209` is
   accurate but the defect is upstream and out of B3's scope.

Parallel to the Firearms `OCR_PATH_HEADING_PROPAGATION` and KI EPUB
`KI_EPUB_EXTRACTION_LANE_REWRITE` deferrals, this defer-with-sign-off
pattern is documented in `docs/DECISIONS.md` "No gate weakening to
make a failing run pass" §Operationalization.

## 8. Empirical lessons captured

- **Page-coverage gates can lie about which page chunks belong to.**
  A page reported as `MISSING_PAGES` may have its content present in
  the corpus under a different `page_number`. Diagnostic probes
  should grep the JSONL for the page's distinctive content before
  building a producer-side emitter to fill the gap. Phase B3 Step 3
  catches this; future phases should follow the same pattern.
- **`extraction_method` is the most useful provenance signal.**
  `hybrid_chunker_pagesplit` says "the cross-page split fired"; the
  presence of that method on chunks across the missing page's
  neighbors is the diagnostic.
- **Phase 1's "no synthetic backfill" invariant has real teeth.** The
  invariant prevents exactly the failure mode I almost wrote in Step
  3 — emit a synthetic page-coverage chunk that duplicates content
  already in the JSONL under a different page number. The Phase 1
  ban directly steered us to the correct halt.
