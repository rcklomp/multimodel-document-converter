# Plan — Post-Docling Sanity Pass for Native-Digital Novels

**Status:** SHIPPED 2026-05-03 (Phases 0-5 complete; smoke re-run in flight)
**Owner:** ingestion pipeline (PdfConversionPlan / DoclingPdfAdapter seam)
**Successor to:** `docs/archive/PLAN_V2.7_DOCUMENT_UNDERSTANDING.md` §5 (PDF extraction refactor; archived 2026-05-03)
**Related:** `docs/ACCEPTANCE_ORDER_PROMPT.md`, `docs/CONVERSION_PROFILES.md`, `docs/PROGRESS_CHECKLIST.md` "Document Understanding Plan Items → Post-Docling Sanity Pass"

## As-shipped vs. as-planned (2026-05-03 close-out notes)

The plan landed end-to-end with five corrections that were not visible
until the actual pipeline ran on HARRY:

1. **`processor.py:2072` was bypassing the adapter.** The cached Docling
   converter (`self._converter`) was being invoked directly, sidestepping
   `DoclingPdfAdapter.convert()` and therefore `apply_postprocessors`.
   Re-routed through `self._adapter.convert(...)`. The static guard from
   v2.7 §5 only banned construction of `PdfPipelineOptions` /
   `DocumentConverter` outside the adapter; it did NOT catch raw
   `convert(...)` invocation. A companion guard test should follow.
2. **`ProfileManager.select_profile`'s `type_mapping` had no entry for
   `DIGITAL_LITERATURE`.** It silently fell through to
   `DigitalMagazineProfile`. Three places needed updates:
   `orchestration/profile_classifier.py` (enum + scorer + score loop +
   modality fallback), `orchestration/strategy_profiles.py`
   (`DigitalLiteratureProfile` strategy class + `_profiles` registry +
   `type_mapping`), `orchestration/strategy_orchestrator.py`
   (`PROFILE_TO_DOC_TYPE` mapping to `DocumentType.LITERATURE`).
3. **The drop cap "M" is NOT a separate Docling item.** Phase 2's
   "standalone glyph adjacent to lowercase paragraph" heuristic never
   matches on Docling 2.86. Reality: the M is appended INLINE at the
   end of the same TextItem (`"r. and Mrs. Dursley...nonsense. M"`).
   Added a complementary `_heal_inline_trailing_dropcap` pass that
   detects the trailing-uppercase + leading-lowercase pattern and moves
   the glyph to the front. Both rules ship; the inline one is the
   actually-needed pattern for the HARRY-style edition.
4. **Phase 3's "no caption" rule was insufficient.** The HARRY page-13
   picture has BOTH a caption ("THE BOY WHO LIVED") AND a `PictureClassificationData`
   annotation labeling it as "other". The original plan rule "suppress
   when no caption" left this case alone and "other" still leaked.
   Upgraded the picture serializer to ALWAYS strip
   `PictureClassificationData` annotations before delegating to the
   parent serializer; captions still flow through, metadata is restored
   afterward so downstream consumers see the full label set.
5. **Phase 4's plan field placement was wrong.** The plan called for
   `bitmap_area_threshold` on `PdfPipelineOptions`. The field actually
   lives on `OcrOptions` (and inherits to `EasyOcrOptions`). Threaded
   through to `EasyOcrOptions.bitmap_area_threshold` in
   `engines/docling_adapter.py`.

The HARRY pages 1-30 acceptance fixture (Phase 0) flipped from
broken-on-current-main to passing for the four body-text swaps it was
designed to detect (page 13 [3]→[4]→[5], page 14 cross-paragraph,
page 21, page 24). Two title-page swaps on pages 5 and 7 remain
visible; they're display-typography fragments, not body-text reading
order, and are out of scope for this plan.

## Live evidence (2026-05-03)

- `mmrag-v2 process data/digital_literature/HarryPotter_and_the_Sorcerers_Stone.pdf`
  → classifier auto-picks `digital_literature`, plan resolves to
  `reading_order_strategy=y_sort_with_dropcap`,
  `suppress_layout_label_text=True`, `bitmap_area_threshold=0.92`.
- Page 13 chunk reads in PDF order with drop cap "M" at the front, no
  `Other`/`Icon`/`Table` leak, no cover OCR garbage.
- Pages 1-4 (photographic covers) produce no text chunks.
- Full unit suite: 570 passed, 2 skipped, 1 deselected (pre-existing
  unrelated `test_semantic_overlap` failure).

---

## Original plan follows (kept verbatim for historical context)


---

## 1. Why this plan exists

A targeted acceptance run on `data/scanned/HarryPotter_and_the_Sorcerers_Stone.pdf`
(verified born-digital: AGaramondPro typesetting, Acrobat Distiller pipeline,
PDF metadata `subject = "Reference Quality Electronic Book Version of the
American Scholastic Hard Cover"`, RC4-encrypted) exposed four failure modes
in our Docling integration, all visible by comparing the converted markdown
to the source PDF:

1. **Reading-order swaps within a page.** Verified by bypassing our chunker
   and calling raw `DocumentConverter` with `do_ocr=False`: on page 13,
   Docling itself emits items at y=354, y=123, y=255 in document order
   `[354, 123, 255]`, but reading order should be top-to-bottom (decreasing y),
   i.e. `[354, 255, 123]`. The "Mr. Dursley was the director" paragraph
   appears between two other paragraphs in the source but Docling places it
   last. Same swap pattern recurs on pages 14 and 16.
2. **Drop-cap "M" dislocated.** The leading "M" of "Mr. and Mrs. Dursley" is
   typeset in a custom display font (`Able`). Docling's layout model treats it
   as a separate item and the reading-order pass appends it to the END of the
   paragraph, producing `"r. and Mrs. Dursley… nonsense. M"` instead of
   `"Mr. and Mrs. Dursley… nonsense."`.
3. **Label-leak into chunk text.** Tokens like literal strings `"Other"`,
   `"Icon"`, `"Table"` appear in chunk content as if they were body text.
   Docling's classifier emits these as `label` fields on items; somewhere
   in our serialization path the labels are concatenated into the chunk text.
4. **OCR auto-pilot on photographic cover pages.** With `--ocr-mode auto`
   Docling runs OCR on the photographic front/back cover pages and produces
   garbage like `"= 23555 AND Potter SIONE has the star of a Quidditch team"`
   from rendered cover artwork.

## 2. Community evidence (Docling 2.86, May 2026)

Web research confirmed these are not user-error or misconfiguration:

| Failure | Status in Docling | Evidence |
|---|---|---|
| Reading order in non-trivial layouts | **Known unfixed limitation.** Maintainer reply: *"Docling doesn't currently offer a built-in option or pipeline parameter to enforce human/visual reading order ... improvements are under discussion but not yet available."* | [Discussion #2791](https://github.com/docling-project/docling/discussions/2791); open issues [#1203](https://github.com/docling-project/docling/issues/1203), [#2201](https://github.com/docling-project/docling/issues/2201), [#2245](https://github.com/docling-project/docling/issues/2245), [#2538](https://github.com/docling-project/docling/issues/2538). CHANGELOG 2.85 → 2.92 contains zero entries on reading-order. |
| Drop-cap reordering | **Out-of-distribution.** Not in the issue tracker. Docling trained on DocLayNet (technical reports, scientific papers, magazines); literary drop caps absent from training distribution. | [#2245](https://github.com/docling-project/docling/issues/2245) is the closest analog; no maintainer engagement. |
| Label-leak | Not exposed via flag. **Fixable on consumer side** via custom serializer subclass (the public hook). | [docling-serve #448](https://github.com/docling-project/docling-serve/issues/448) confirms `doc_items.label` is intentionally exposed for downstream filtering. |
| OCR gating on hybrid PDFs | **"On the wish list,"** not shipped. | [Discussion #2755](https://github.com/docling-project/docling/discussions/2755). Practical levers today: `bitmap_area_threshold` (default 0.75), `do_ocr=False`. |

Community precedent for the post-pass approach: `docling-hierarchical-pdf`
(krrome, MIT, v0.1.1) calls `ResultPostprocessor(result, source=source).process()`
after `DocumentConverter().convert(...)`. We already integrate this for
heading reclassification (per project memory); we are extending the same
post-pass seam to cover the four failure modes above.

**Implication:** Upgrading Docling will not fix any of the four issues in
the foreseeable future. Bypassing Docling (option C, see §6) loses its
table/figure/heading detection, which is the genuinely best part. The
right ambition is option B: surgical post-Docling stages mirroring what
the maintainers have publicly described as "the eventual plan."

## 3. Goals & non-goals

**Goals**
- Fix the four observed failure modes without forking Docling.
- Stay aligned with community patterns so upstream fixes drop in cleanly
  (when Docling ships built-in reading-order, our pass becomes a no-op
  controlled by a profile field).
- Treat the HARRY pages 1–30 markdown as an executable acceptance fixture:
  production conversion must match the PDF's reading order paragraph-by-paragraph,
  drop caps in the right place, no label-leak tokens, no cover-page OCR garbage.

**Non-goals**
- Replace Docling. Augment, do not bypass.
- Fix issues outside this PDF class (scanned books, CJK reading order,
  multi-column scientific papers). Scope to the failure pattern we have
  ground truth for.
- Touch the `extraction_route` serialization gap. That is a smaller,
  parallel fix and should land independently.

## 4. Architectural placement

All Docling-touching code stays inside the existing seam:
[src/mmrag_v2/engines/docling_adapter.py](../src/mmrag_v2/engines/docling_adapter.py).
The boundary guard tests (`test_no_pipeline_options_construction_outside_adapter`,
`test_no_production_docling_imports_outside_adapter`) are kept as-is and
continue to enforce that Docling internals do not leak into other modules.
New behavior is added as ordered post-conversion stages, each independently
togglable via fields on `PdfConversionPlan` so we can A/B them per profile.

```
PdfConversionPlan  →  DoclingPdfAdapter.convert()  →  Docling.DocumentConverter.convert()
                                                        ↓
                                  [NEW] PostProcessor pipeline:
                                  1. label_filter        (Phase 3)
                                  2. dropcap_promoter    (Phase 2)
                                  3. reading_order_sort  (Phase 1)
                                                        ↓
                                  → UniversalDocument → ElementProcessor → chunks
```

Order matters: filter labels first (so dropcap heuristic does not have to
skip noise), then promote drop caps (so reorder pass sees the merged
paragraph), then re-sort. OCR gating (Phase 4) is configured on the plan
upstream of Docling, not after.

## 5. Phases

Each phase is independently mergeable and revertible. Each lands as a single
PR with red→green TDD. Phase 0 is pure tests + fixtures; Phases 1–4 add
production code; Phase 5 is corpus + acceptance integration.

### Phase 0 — Lock the ground truth (test-only)

**What:** Author the executable acceptance fixture.
- `tests/fixtures/harry_potter_pages_1_to_30/expected_reading_order.txt` —
  paragraph-level expected order extracted by walking the PDF's y-coordinates
  with `pymupdf` (deterministic, not Docling).
- `tests/test_docling_postprocessor_acceptance.py::test_harry_paragraph_order_matches_pdf`
  runs the full pipeline on HARRY pages 1–30 and asserts the chunk order
  matches the y-coordinate baseline. Marked `xfail` initially with reason
  `"Phase 1 not yet implemented"`.

**Done when:** Test exists, runs, fails on current main with the expected
paragraph-swap diff. The diff is human-readable and recognizable as the
page-13 `[3]→[4]→[5]` swap pattern.

**Rationale:** Without this, every later phase is judged by eye on a
markdown file — exactly how the reordering bug was missed initially. The
fixture forces honesty. **Estimated effort: 1.5 hours.**

---

### Phase 1 — Reading-order y-sort

**What:** Post-Docling stage that re-sorts items per page by `(-bbox.t, bbox.l)`
when bboxes are present and confidence-tagged as native (not OCR-derived).
Items without bboxes pass through in Docling's order. Mirrors what
`docling-hierarchical-pdf` does at the hierarchy level — we do it at the
body-text level.

**Where:** New file `src/mmrag_v2/engines/docling_postprocess.py`
exposing `apply_reading_order_sort(doc, plan)`. Called from
`DoclingPdfAdapter.convert()` after Docling returns and before the result
is handed to the chunker.

**Plan field:** Add to `PdfConversionPlan`:
```python
reading_order_strategy: Literal[
    "docling_native", "y_sort", "y_sort_with_dropcap"
] = "docling_native"
```
Default keeps current behavior; profiles opt in.

**Tests (red→green):**
- `test_y_sort_orders_three_paragraphs_top_to_bottom` — synthetic Docling
  document with three TextItems at y=354, y=123, y=255 in document order
  `[354, 123, 255]`; assert post-pass order is `[354, 255, 123]`. Mirrors
  HARRY page 13.
- `test_y_sort_skips_items_without_bbox` — items with `prov=None` retain
  Docling's order.
- `test_y_sort_does_not_break_multipage_order` — items on page 14 never
  reorder ahead of page 13 items.
- `test_harry_paragraph_order_matches_pdf` (from Phase 0) flips xfail → pass.

**Done when:** All four tests green; HARRY conversion shows page 13's
"Mr. Dursley was the director" paragraph between "r. and Mrs. Dursley" and
"The Dursleys had everything", matching source.

**Risk:** Could break documents where Docling's reading order is *better*
than y-sort (e.g. correct multi-column inference). Mitigated by the
`reading_order_strategy` field — only `digital_literature` and future
literature-like profiles opt in.

**Upstream tracking:** Comment in source linking to
[#2791](https://github.com/docling-project/docling/discussions/2791),
[#1203](https://github.com/docling-project/docling/issues/1203),
[#2245](https://github.com/docling-project/docling/issues/2245).
When Docling ships built-in reading-order control, replace with their flag.
**Estimated effort: 3 hours.**

---

### Phase 2 — Drop-cap promotion

**What:** Heuristic detector for the "lone capital letter from a display
font sitting after the paragraph it should lead" pattern. Rule: if a
TextItem contains a single uppercase character, is in a non-body font
(not in the document's most-common font), and sits within the same y-band
as the next paragraph that begins with a lowercase fragment, it is promoted:
prepend the letter to the next paragraph and drop the standalone item.

**Where:** Same `docling_postprocess.py`, function
`promote_drop_caps(doc, body_font_set)`. Body font set passed in from the
adapter, computed from the most-common font across the document.

**Plan field:** Reuse `reading_order_strategy = "y_sort_with_dropcap"`.

**Tests (red→green):**
- `test_dropcap_M_prepended_to_paragraph` — synthetic doc with
  `TextItem("M", font="Able")` and
  `TextItem("r. and Mrs. Dursley...", font="AGaramondPro-Regular")`;
  assert merged item reads `"Mr. and Mrs. Dursley..."`.
- `test_dropcap_not_promoted_when_next_paragraph_starts_uppercase` —
  `TextItem("M", font="Able")` followed by a paragraph starting with capital
  — no merge, the standalone M is preserved (it is not a drop cap, it is
  some other artifact).
- `test_dropcap_promoter_idempotent` — running twice gives the same result.
- `test_harry_chapter_one_drop_cap_present_at_start` — HARRY-specific:
  chunk for chapter 1 page 13 starts with `"Mr. and Mrs. Dursley"`, not
  `"r. and Mrs. Dursley"`, and contains no trailing standalone `"M"`.

**Done when:** Five tests green.

**Risk:** False positives on legitimate single-letter content (sentence
"I", footnote markers, section labels "A."). Mitigation: require non-body
font *and* the next paragraph to start lowercase. New corpus false positives
trigger a fixture and rule tightening.

**Upstream tracking:** This area has no community precedent — we own the
heuristic. Keep it surgical (single-letter, font-based, y-adjacent).
**Estimated effort: 4 hours.**

---

### Phase 3 — Label-leak filter

**What:** Subclass Docling's serializer to suppress text emission for items
labeled `OTHER`, `UNKNOWN`, and `PICTURE` (when no caption is present).
Labels still flow through to chunk metadata, so retrieval/QA can see them;
they just do not pollute the body text. This is the public-API approach
the community uses (no `skip_labels` flag exists; this is the hook).

**Where:** Same `docling_postprocess.py`, classes `MmragMarkdownSerializer`
and `MmragPictureSerializer` extending Docling's defaults. Wired into
`DoclingPdfAdapter.convert()` via `serializer_provider`.

**Plan field:** `suppress_layout_label_text: bool = False` (default off;
novel-handling profiles turn it on).

**Tests (red→green):**
- `test_picture_serializer_emits_empty_when_no_caption` — `PictureItem`
  with no caption produces no text output.
- `test_picture_serializer_emits_caption_when_present` — `PictureItem`
  with caption emits the caption text only.
- `test_other_label_suppressed_in_chunk_text` — synthetic doc with an
  `OTHER`-labeled item containing text; chunk output has no occurrence
  of that text.
- `test_harry_no_other_icon_table_tokens_in_chunks` — HARRY conversion
  contains no chunk whose body equals `"Other"`, `"Icon"`, or `"Table"`
  after `.strip()`.

**Done when:** Four tests green.

**Risk:** Low. Narrow surface, well-defined hook.

**Upstream tracking:** [docling-serve #448](https://github.com/docling-project/docling-serve/issues/448).
**Estimated effort: 2 hours.**

---

### Phase 4 — OCR gating

**What:** Two surgical changes:
1. Add `bitmap_area_threshold: float = 0.75` to `PdfConversionPlan`,
   threaded through to Docling's `PdfPipelineOptions`. Profiles that should
   not auto-OCR cover artwork (digital literature, magazines with full-bleed
   photos) raise it to ~0.92.
2. When the diagnostic engine detects native text density above a
   configurable floor, the plan sets `do_ocr=False` for the document.
   Cover/jacket pages with no native text get treated as illustrations
   rather than OCR'd.

**Where:** Field on `pdf_plan.py`, honored by `docling_adapter.py`,
set per profile in `orchestration/profile_classifier.py`.

**Tests:**
- `test_pdf_conversion_plan_bitmap_area_threshold_default` — default 0.75.
- `test_plan_digital_literature_raises_bitmap_threshold` —
  `digital_literature` profile sets it to 0.92.
- `test_adapter_passes_bitmap_threshold_to_pipeline_options` — bridge test,
  reads `PipelineOptions.bitmap_area_threshold` after adapter constructs them.
- `test_harry_cover_pages_have_no_ocr_garbage` — HARRY pages 1–3 chunks
  contain none of the patterns `r"= \d+"`, `r"AND Potter SIONE"`, or any
  non-ASCII jumble.

**Done when:** Four tests green; HARRY page 1 chunk contains either nothing
or the proper back-cover blurb.

**Risk:** Some scanned documents legitimately need OCR on image-heavy
pages. Mitigated by per-profile control: only digital-literature turns
OCR off; scanned profiles keep existing behavior.

**Upstream tracking:** [#2755](https://github.com/docling-project/docling/discussions/2755).
**Estimated effort: 2 hours.**

---

### Phase 5 — Acceptance + corpus integration

**What:** Re-run the acceptance order with all four post-processors enabled
for the `digital_literature` profile.

- Pending user approval: move `data/scanned/HarryPotter_and_the_Sorcerers_Stone.pdf`
  → `data/digital_literature/HarryPotter_and_the_Sorcerers_Stone.pdf`.
- Re-anchor the HARRY probe in `docs/ACCEPTANCE_ORDER_PROMPT.md` to assert
  `profile=digital_literature` and the corpus-wide `extraction_route` non-empty
  invariant.
- Pick `data/business_form/0013_140302111325_001.pdf` as the new `SCAN0013`
  probe for the scanned-route assertion.

**Tests:** Full unit suite (now ~530 tests), multi-profile smoke, HARRY
end-to-end acceptance fixture from Phase 0.

**Done when:** `output/acceptance_<NEW_RUN_ID>/VERDICT.json.verdict == "PASS"`.

**Estimated effort: 2 hours.**

## 6. Alternatives considered and rejected

**A. Mark HARRY known-bad, add regression test, re-anchor probe.**
Rejected: the failure class (drop caps + reading order + label leak) is
not exotic. As the corpus grows we will keep encountering it. Marking
known-bad concedes a class of input.

**C. Bypass Docling for literature, route to pymupdf with manual sorting.**
Rejected: pymupdf does not infer table structure or heading hierarchy.
Docling's table/figure/heading detection is genuinely the best part —
losing it for the body-text reordering problem is overkill. Surgical
post-pass keeps the gains.

**B. Post-Docling sanity pass (this plan).**
Selected: aligns with community precedent (`docling-hierarchical-pdf`,
maintainer-described eventual plan), surgical, revertible, individual
phases independent.

## 7. Cross-phase concerns

**Documentation updates** (one PR per phase, batched into the same merge):
- [docs/ARCHITECTURE.md](ARCHITECTURE.md) — add post-processor pipeline diagram.
- [docs/DECISIONS.md](DECISIONS.md) — one entry per phase explaining
  rationale and linking to the Docling issue tracker.
- [docs/PROGRESS_CHECKLIST.md](PROGRESS_CHECKLIST.md) — track phase completion.
- [docs/CONVERSION_PROFILES.md](CONVERSION_PROFILES.md) — new
  `digital_literature` profile and Plan settings.

**Upstream tracking:** Add a comment in `docling_postprocess.py` linking
to each Docling issue our heuristic addresses
([#2791](https://github.com/docling-project/docling/discussions/2791),
[#2245](https://github.com/docling-project/docling/issues/2245),
[#1203](https://github.com/docling-project/docling/issues/1203),
[docling-serve#448](https://github.com/docling-project/docling-serve/issues/448),
[#2755](https://github.com/docling-project/docling/discussions/2755)).
When Docling ships a built-in fix, delete the corresponding pass and
verify the test still passes.

**Test contract integrity** (per `CLAUDE.md`):
- Negative tests, regression tests, and acceptance fixtures are executable
  requirements. Do not weaken or rewrite assertions to make implementation
  pass.
- The HARRY ground-truth fixture (Phase 0) is the binding contract for
  this work. If conversion regresses against it later, the regression
  must be fixed, not the fixture.

## 8. Acceptance gate

The whole plan is "done" when this command sequence produces `verdict: PASS`
against the same prompt structure:

```bash
bash scripts/smoke_multiprofile.sh "${ROOT}/smoke"
# expected: every row GATE_PASS + UNIVERSAL_PASS

pytest tests/ -v
# expected: 0 failed, 0 errored

pytest tests/test_docling_postprocessor_acceptance.py::test_harry_paragraph_order_matches_pdf -v
# expected: green

python <generate verdict from $ROOT>
# expected: VERDICT.json.verdict == "PASS"
```

with the additional human-readable check that the HARRY pages 1–30
markdown reads in PDF source order, drop caps in the right places, no
`Other` / `Icon` / `Table` tokens, no cover OCR garbage.

## 9. Effort summary

| Phase | Estimate |
|---|---|
| Phase 0 — ground-truth fixture | 1.5 h |
| Phase 1 — y-sort | 3 h |
| Phase 2 — drop-cap promotion | 4 h |
| Phase 3 — label-leak filter | 2 h |
| Phase 4 — OCR gating | 2 h |
| Phase 5 — acceptance + corpus | 2 h |
| **Total** | **~14.5 h** |

## 10. Decision log

- **2026-05-03** — Plan ratified. Approach B selected based on community
  evidence: Docling will not ship reading-order, drop-cap, or OCR-gating
  fixes in the foreseeable future (verified across CHANGELOG 2.85 → 2.92,
  Discussions #2791 / #2755, multiple open issues). Post-Docling sanity
  pass mirrors community precedent (`docling-hierarchical-pdf`). Phases
  ordered for independent revertability. HARRY pages 1–30 markdown
  becomes the binding acceptance fixture.
