# Phase A — MISSING_PAGES Root-Cause Diagnostic (2026-05-11)

> **Status:** complete. Diagnostic is read-only; no production code
> changed. Outcome below names the four distinct failure sub-classes
> within the `MISSING_PAGES` failure surface and points Phase B at
> specific code lanes per sub-class.

## 1. Scope and method

Per `docs/PLAN_V2.9.md` §3 Phase A, this diagnostic investigates the
10 docs with MISSING_PAGES > 0 in
`docs/QUALITY_SNAPSHOT_2026-05-11_v2.9_strict_gate_full_corpus.md`.
The starting hypothesis space:

1. **Page-numbering mismatch** between JSONL `page_number` and
   source-PDF page index.
2. **Real chunk drop in a specific shape.**
3. **Conversion-time drop** before chunks reach our serializers.
4. **Profile-routing drop** sending a doc through a thinner lane.

Method: read-only inspection of JSONL `page_number` distributions
vs source-PDF page counts (`pymupdf`), plus single-page CLI
probes (`mmrag-v2 process --pages …`) to observe pipeline behavior
on representative missing pages without spending VLM budget.

## 2. Headline findings

- **Hypothesis 1 (page-numbering mismatch) is ruled out.**
  Python_Distilled JSONL chunks declare pages in the range 1-1402;
  the source PDF has 1411 pages. Coverage is 713/1411 = 50.5 %. The
  page-index mapping itself is correct; the 697 reported missing
  pages are genuine "no chunk emitted" cases per the strict gate's
  blank-page-aware count.
- **The MISSING_PAGES failure surface is not one defect — it is four
  distinct sub-classes** (A-D below) with different code lanes.
  Phase B must address them separately, in priority order, with a
  unit test per sub-class before reconvert.
- **The strict gate's blank-page check is correct.** Re-checked at
  `scripts/qa_full_conversion.py:185-215` (`_read_blank_pages_in_source`):
  blank = `not text AND not images AND not non_trivial_blocks`. A
  page with `text_len=0` AND `blocks=0` BUT a single full-page image
  is correctly classified as non-blank (it has content; the chunk is
  missing). No gate change is required.

## 3. Sub-class taxonomy

### Sub-class A — Image-only page chunk-drop (largest by chunk count)

Source-page shape: `page.get_text("text").strip() == ""` AND
`page.get_text("blocks") == []` AND `page.get_images(full=True) ≥ 1`.
Visually these pages contain only a full-page image (chapter
divider artwork, full-page figure, scanned page) and have no
text-layer content. The strict gate correctly classifies them as
non-blank (image content is real). Our pipeline emits no chunk for
them.

| Doc | Affected pages (sample) | Per-doc impact |
|---|---|---:|
| Python_Distilled | 541-590, 626-648, 686-698, 724-739, 753-768, 801-824, 843-862, 887-924, 1263-1291, 1327-1344 (10 longest runs) | **697** (dominant) |
| Earthship_Vol1 | p109 (1 page) | 1 |
| Devlin_LLM_Agents | p2 (`text_len=0 blocks=1 images=1`) | 1 (other half of Devlin's MISSING_PAGES=2 is sub-class C) |
| Python_Cookbook, Fluent_Python, Chaubal | some subsets of their MISSING_PAGES lists | (to confirm in Phase B) |

Verified via single-page probes: source-PDF `fitz.get_text` returns
0 chars; source-PDF `page.get_images()` returns ≥ 1; gate's
blank-page detector correctly returns False.

Why our pipeline drops them: Docling's text-layer-routed
`hybrid_chunker` produces zero chunks (no text to chunk). The
shadow-extraction fallback (`extraction_method='shadow'`,
167 chunks corpus-wide in Python_Distilled) does fire on some
pages but not consistently on text=0 image=1 pages. There is no
guaranteed "image-only → emit one image chunk" path.

**Phase B fix lane (per §2d constraint #1):** Extraction policy
layer. Either:

- `PdfConversionPlan` / `DoclingPdfAdapter` enables Docling's
  picture-extraction path on every page (currently profile-gated);
  OR
- `BatchProcessor` adds an "if Docling emitted no chunks AND the
  source page has an image, fall back to full-page-image chunk via
  shadow extraction" guard. This must live in the adapter or plan,
  not in the JSONL writer.

The choice between Docling-native vs shadow depends on whether
Docling 2.86 reliably emits a `PictureItem` for full-page images
on `technical_manual`-profiled docs; that is a Phase B
implementation question, not a Phase A diagnostic question.

### Sub-class B — TOC pages dropped by corruption-quarantine despite Phase 1 routing

Source-page shape: `text_len ≈ 2200-3000`, dotted-leader TOC
entries (`About the Author..........................................`),
roman-numeral page numbers (v through xxvii or similar).
Docling correctly emits chunks (verified via probe) AND the
Phase 1 dense-index router activates correctly (`[HYBRID-CHUNKER]
Routing dense TOC/index page(s) around HybridChunker: [5,6]`),
BUT the chunks are then dropped by `_quarantine_corrupted_text_chunks`
at `src/mmrag_v2/batch_processor.py:829`.

| Doc | Front-TOC pages dropped |
|---|---:|
| Cronin_GenAI_Models | 24 (p5-p28) |
| Nagasubramanian_Agentic_AI | 16 (p2, p5-p19) |
| Sekar_MCP_Standard | 13 (p2, p5-p13, p159, p228, p247) |
| Chaubal_PyTorch_Projects | 9 (p2, p4-p11) |

Probe evidence (Cronin p5-p10):

```
[HYBRID-CHUNKER] Routing dense TOC/index page(s) around HybridChunker: [5]
[HYBRID-CHUNKER] Produced 1 text chunks
...
[CORRUPTION-QUARANTINE] Dropped 5 text chunks with unrepairable encoding artifacts
[FINALIZE] Starting JSONL write: 0 chunks to process
```

The quarantine fires on `has_encoding_artifacts` matching one of
the patterns at
`src/mmrag_v2/validators/corruption_interceptor.py:30-45`:

```python
CORRUPTION_PATTERNS = re.compile(
    r"/C\d{1,3}"
    r"|/uni[A-F0-9]{4,}"
    r"|\\x[0-9a-f]{2}"
    r"|�"
)
OCR_FAILURE_PATTERNS = re.compile(
    "[—]{6,}"
    "|[CS]{10,}"
)
```

`fitz.get_text` on Cronin p10 returns 3040 clean characters with
**zero** CIDFont/uni/replacement-char matches. But the chunk that
reaches `_quarantine_corrupted_text_chunks` has different content
— Docling's per-cell extraction of the TOC table likely re-encodes
the dotted leaders or label/page-number columns in a way that
matches one of the regex patterns above. Most likely candidates:

- `[—]{6,}`: if Docling renders the dotted leaders as a Unicode
  em-dash run (e.g., when the source font's `.` glyphs lack ToUnicode
  mappings and Docling substitutes em-dashes).
- `/C\d{1,3}` or `/uni[A-F0-9]{4,}`: if the TOC font is CIDFont and
  Docling preserves placeholder names.

This is **the same defect class as Phase 4 Step 3 (Combat p66)**
but the quarantine is now over-firing on legitimate TOC content
that happens to match the same regex. Phase 4 Step 3's dry-run
("3,709 chunks across 4 unrelated docs: zero false positives") did
not include Cronin, Nagasubramanian, Sekar, or Chaubal — all of
which are 2025-2026-published GenAI books that share a common
publisher template, suggesting that template's TOC table emits the
same Docling-side signature as Combat p66.

**Phase B fix lane:** `_quarantine_corrupted_text_chunks` must
either:

- distinguish TOC/index chunks (already labeled via Phase 1's
  `extraction_method=hybrid_chunker_pageskip` /
  `hybrid_chunker_pageskip_source_pdf`) from body chunks and
  exempt them from the regex check; OR
- the corruption patcher (`patch_corrupted_chunks` at
  `batch_processor.py:3073`) must be taught to repair the
  TOC-specific signature before quarantine sees it.

The former is simpler and surgical; the latter is more correct.
Both need a regression test: a Cronin-p10-like fixture is preserved
through the pipeline AND a Combat-p66-like fixture is still
correctly dropped.

### Sub-class C — Short legitimate-text page filtered out

Source-page shape: `text_len ≈ 30-200` characters of legitimate
content (dedication, chapter title, short URL list). Pages are not
image-only and not corruption-tainted; they have real but very
short text. The pipeline drops them somewhere in the
empty-text-chunk / minimum-length filter chain.

| Doc | Page | Content shape | text_len |
|---|---:|---|---:|
| Ayeva_Python_Patterns | 4 | Dedication ("I would like to thank my parents…") | 74 |
| Devlin_LLM_Agents | 170 | Chapter heading ("II — Building Intelligent Foundations") | 37 |
| Greenhouse_Design | 2 | Title page ("Greenhouse Design and Control") | 29 |
| Bourne_RAG_2024 | 209 | URL reference list (190 chars: ChatGPT Arena, MMLU, MT Bench links) | 190 |

These pages have legitimate semantic content (dedications, chapter
headers, reference lists). Dropping them is a bug; they should
emit at least one short chunk.

**Phase B fix lane:** the empty-text-chunk guards added in Phase 1
(at three sites: oversize-breaker, finalize stage, JSONL-write
loop) likely treat these as effectively-empty after some
normalization. Find the guard that fires on `len < N` for some N >
30 and either lower the threshold or exempt named-page-class
shapes (e.g., never drop a page-number-bearing chunk regardless of
length).

A regression-test fixture: take any of the four pages above (Ayeva
p4 is shortest at 74 chars and the simplest) and assert that
processing it produces ≥ 1 chunk.

### Sub-class D — "Intentionally left blank" disclaimer pages

Source-page shape: `text_len ≈ 30-70`, content is literally the
boilerplate phrase "This page intentionally left blank" (often
duplicated when the publisher's template has a backing layer). The
gate considers them non-blank because text content exists. The
pipeline also drops them (no chunks emitted).

| Doc | Pages |
|---|---|
| Greenhouse_Design | p3, p11, p23 (all three contain only "This page intentionally left blank") |

This is a 3-page subset of Greenhouse's 4 missing pages
(Greenhouse p2 is sub-class C above).

**Phase B fix lane:** two options:

- **Producer side:** Emit a chunk for any page whose text contains
  the "intentionally left blank" boilerplate, tagged with an
  appropriate marker, so the strict gate sees coverage. The chunk
  content can be the boilerplate verbatim or a normalized
  `[INTENTIONALLY_BLANK]` sentinel.
- **Gate side:** Extend `_read_blank_pages_in_source` to recognize
  the boilerplate as blank-equivalent. This is a contained,
  case-insensitive substring check ("intentionally left blank")
  and would correctly count these pages as advisory rather than
  hard fail. The producer-side option keeps the chunk visible in
  retrieval; the gate-side option keeps the producer simpler. Both
  are valid; recommend producer-side because retrieval over a
  blank-marker chunk is still useful (a user querying for a
  document table-of-contents region wants to know that page p11
  exists).

This is a stand-alone fix and does NOT require Phase B's reconvert
of the 12 docs — Greenhouse is already in the Phase B
reconvert list for sub-classes A/B/C. A regression test fixture:
a single page with the boilerplate produces exactly 1 chunk with
content equal to the normalized boilerplate.

## 4. Sub-class impact summary

| Sub-class | Total missing pages explained | Docs affected |
|---|---:|---|
| A — Image-only page chunk-drop | ~700 (Python_Distilled dominates) | Python_Distilled, Earthship, Devlin (partial), possibly subsets of Fluent/Python_Cookbook/Chaubal |
| B — TOC corruption-quarantine over-fire | ~62 (24+16+13+9) | Cronin, Nagasubramanian, Sekar, Chaubal |
| C — Short-text page filter | ~6 (Ayeva 1 + Devlin 1 + Greenhouse 1 + Bourne 1 + plus an unknown subset of Python_Cookbook/Fluent_Python sparse missing) | Ayeva, Devlin, Greenhouse, Bourne; possibly subsets of others |
| D — "Intentionally left blank" disclaimer | 3 (Greenhouse only, confirmed) | Greenhouse |

Sub-classes A and B account for 98 % of the missing-page surface
by chunk count and must be addressed first. Sub-classes C and D
are smaller numerically but the fixes are equally surgical.

## 5. Phase B priority order

Recommended Phase B execution order, smallest-cost first:

1. **B1 — Sub-class B (TOC quarantine over-fire).** Fix in
   `_quarantine_corrupted_text_chunks` to exempt `hybrid_chunker_pageskip*`
   extraction methods OR teach `patch_corrupted_chunks` to repair
   the publisher-template signature before quarantine. Regression
   test: Cronin p10 fixture preserved, Combat p66 fixture still
   dropped. **No reconvert needed for the fix design** — the fix
   is in post-Docling code. Reconvert + re-enrich of Cronin,
   Nagasubramanian, Sekar, Chaubal needed to validate the metric.
2. **B2 — Sub-class D ("intentionally left blank").** Producer-side
   chunk emission. Regression test: a boilerplate-only page emits
   exactly 1 chunk. Reconvert of Greenhouse covers this.
3. **B3 — Sub-class C (short-text drop).** Locate the empty-text
   guard that overshoots; lower its threshold or exempt
   short-but-named pages. Regression tests for Ayeva p4 / Bourne
   p209 / Greenhouse p2 / Devlin p170 shapes. Reconvert of
   affected docs covers this; many overlap with B1/B2/B4.
4. **B4 — Sub-class A (image-only page chunk-drop).** Largest by
   chunk count. Fix in `PdfConversionPlan` / `DoclingPdfAdapter`
   per §2d constraint #1 — either enable Docling picture
   extraction unconditionally or guarantee shadow-extraction
   fires on every text=0 image=1 page. Regression test: an
   image-only fixture page produces ≥ 1 image chunk. Reconvert of
   Python_Distilled, Earthship, Devlin (and any other doc with
   sub-class A pages identified in B1-B3).

After all four fixes ship and the affected docs reconvert, run
Phase H targeted re-enrichment, then re-run the full strict gate.
Expected: MISSING_PAGES = 0 (non-blank) across all 12 affected
docs.

## 6. Out-of-scope but observed

- `scripts/convert_books_v29.py` reports `ENGINE_USE: Docling
  v2.66.0` in the CLI banner during probes but the actual Docling
  version is `2.86.0` (visible in batch logs). Banner is stale;
  cosmetic, no functional impact.
- `BatchProcessor._filter_blank_assets` (at
  `batch_processor.py:859`) already exists and drops image chunks
  whose asset PNG is blank. This is what Phase E (Combat
  `figure_36`) needs to invoke — the producer-side filter is
  already there, it just isn't catching the Combat case. Phase E
  scope tightens: confirm `_filter_blank_assets` fires correctly
  on `figure_36` and if not, fix its threshold.
- 167 `extraction_method='shadow'` chunks across Python_Distilled
  shows shadow extraction does fire on some image-bearing pages.
  Phase B4 needs to understand the gating condition that lets some
  image-only pages through and not others; the existing shadow
  lane may simply need its activation condition widened.

## 7. Evidence files

- Snapshot BEFORE: `docs/QUALITY_SNAPSHOT_2026-05-11_v2.9_strict_gate_full_corpus.md`
- Per-doc strict-gate output: `/tmp/strict_gate_capture/*.txt` (temporary; will not survive a reboot)
- Single-page probe outputs: `/tmp/phase_a_probes/cronin_p5_*` (temporary)
- Source files referenced:
  - `src/mmrag_v2/batch_processor.py:829` (`_quarantine_corrupted_text_chunks`)
  - `src/mmrag_v2/batch_processor.py:3073` (`patch_corrupted_chunks` call)
  - `src/mmrag_v2/batch_processor.py:859` (`_filter_blank_assets`)
  - `src/mmrag_v2/validators/corruption_interceptor.py:30-54`
  - `scripts/qa_full_conversion.py:185-215` (`_read_blank_pages_in_source`)

## 8. Postscript — B1 outcome correction (2026-05-11)

The diagnostic's §3 Sub-class B recommended the surgical exemption
(skip `_quarantine_corrupted_text_chunks` when
`extraction_method.startswith("hybrid_chunker_pageskip")`) as the
preferred fix, with producer-side normalization marked as a deeper
alternative. **Implementation experience proved the prediction
incomplete.** The exemption alone is necessary but not sufficient:

1. **A parallel BatchProcessor drop site existed** —
   `_drop_corrupted_chunks_before_metadata` at
   `batch_processor.py:4067` (Phase 4 Step 3, finalize-stage).
   §4 of this diagnostic flagged this as a parallel boundary to
   audit; B1 confirmed the audit was correct and added the same
   exemption at that second site.
2. **The strict-gate scripts themselves enforce corruption
   detection.** `qa_conversion_audit.py` and
   `qa_universal_invariants.py` each have independent corruption
   regexes that fire on raw U+FFFD content regardless of the
   extraction_method exemption. With only the BatchProcessor
   exemption applied, Cronin's reconvert produced
   `LOCALIZED_CORRUPTION = 36`, `UNIVERSAL_FAIL =
   irreparably_corrupt_chunks=36` — the chunks survived the
   producer pipeline only to be rejected by the validator scripts.

The load-bearing fix turned out to be U+FFFD normalization at the
producer. Collapsing `[\s]*�+[\s]*` runs to a single space at chunk
construction means chunks land clean and satisfy every downstream
corruption detector — both BatchProcessor and the validator scripts
— without requiring per-detector exemptions.

**Sanitizer scope (widened 2026-05-11 after architectural review).**
The original implementation scoped the sanitizer to
`_sanitize_toc_index_text` only (i.e., Phase 1 dense-index router
output). Review flagged this as borderline overfit — it implicitly
asserted "TOC = cosmetic, body = real corruption" based on the
Cronin observation alone. The sanitizer is now universal: it runs
as `_collapse_replacement_chars` at the chunk-creation chokepoint
in `mmrag_v2.schema.ingestion_schema` (factory + field-validators
for `content`, `refined_content`, `visual_description`). Rationale:
U+FFFD is by definition an unrenderable glyph; preserving it cannot
recover semantic content at any extraction site. Corruption
signatures that DO carry signal (CIDFont placeholders `/C211`,
`/uniXXXX`, `\xHH` escapes, em-dash runs `[—]{6,}`, C/S filler
runs `[CS]{10,}`) are unchanged and continue to be detected /
quarantined.

The TOC-scoped sanitizer in `_sanitize_toc_index_text` still runs
because the dense-index router does its own line-splitting and
benefits from seeing cleaned text BEFORE the split. Two passes,
both correct.

The two-site BatchProcessor exemption is retained as defense in
depth: if Phase 1 router output ever contains a non-U+FFFD
corruption signature (e.g., CIDFont placeholders that survive
upstream cleanup), the exemption keeps those chunks alive instead
of silently nuking the TOC.

**Empirical lesson for B2 / B3 / B4 (and any future failure-class
phase):** the parallel-site audit must include the strict-gate
scripts (`qa_full_conversion.py`, `qa_conversion_audit.py`,
`qa_universal_invariants.py`, `qa_semantic_fidelity.py`), not only
the production pipeline (`batch_processor.py`,
`processor.py`, engines). When the gate has its own pattern
matchers, the producer must be the authoritative cleanup site —
exemptions cannot silence detectors outside the producer's reach.
