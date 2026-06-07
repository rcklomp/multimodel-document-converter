# PLAN_GATE_QUALITY_V1 - Content-Quality Gate Workstream

Status: PROPOSED (2026-06-07, rev. 2 - incorporates two external review rounds)
Owner: extraction + QA
Scope: close the proxy-vs-outcome gap in the QA gate suite exposed by the
2026-06-07 16-doc crucible content audit.

Review amendments folded in (2026-06-07): spatial-first signals (furniture by
bbox Y-position, verified); cross-page dedup hard-excludes table/form (multi-page
header trap); F4 resolved to "fence code"; F7 blank check moved pre-VLM. Rejected
with evidence: a projected `layout_weight` heading proxy (MinerU2.5 emits no such
signal; bbox-height does not separate headings - see Section 3).

## 0. Motivation

The 16-doc crucible passed every gate (16/16 `QA_PASS`, `leak=0`), yet a manual
read of the output surfaced six classes of real, retrieval-harming defects. The
gates missed them because they assert **structural proxies** rather than
**content outcomes**:

- "X% of chunks have a parent_heading" instead of "the headings are real titles"
- "the image has a non-placeholder description" instead of "the region is
  actually visual content worth a chunk"
- "no empty chunks / valid bbox / page coverage" instead of "the chunks are
  retrievable, non-junk content"

Root cause: this gate suite was built for the v2 crash-era failure modes
(missing pages, empty chunks, bad bboxes, OCR corruption). The V3 MinerU+VLM
pipeline produces output that is structurally clean but semantically noisy
(furniture, misclassified regions, hallucinated headings). The gates never
evolved to the new failure surface.

Unifying frame: QUALITY_GATES.md already justifies the `ASSET_TINY` advisory
with a "Retrieval-Value Test." Almost every finding is the same question: would
this chunk help answer a query? Folios, mastheads, non-visual "images", repeated
captions, and garbled covers all fail that test, and nothing measures it. This
plan generalizes the Retrieval-Value Test into a small set of calibrated
signals.

## 1. Principles

1. **Two tiers, kept separate.** Hard gates stay deterministic and unambiguous
   (schema invariants: asset_ref present, no empty chunks, page coverage, bbox
   in range). Fuzzy quality properties go in the advisory tier as scored ratios.
   Forcing a fuzzy check into a hard gate is what made the pipeline drop good
   dense pages in V3; we do not repeat that.
2. **Fix-and-guard.** Every new metric is paired with an extraction-side fix.
   The fix lowers the metric; the metric is the regression net that keeps it
   down. We are not adding gates to fail more runs; we are surfacing extraction
   defects so they get fixed (AGENTS.md "fix extraction, not the judge").
3. **Spatial-first, not string-first.** Prefer structural metadata (bbox
   geometry, element type, image luminance) over text heuristics. Verified on
   real data: the folio `'22 ... www.Key.Aero'` sits at bbox y=960-975 (bottom
   4% of the page); a text-pattern matcher is the wrong tool when geometry is
   decisive. Text heuristics misfire (the naive "long heading" rule flagged
   PCWorld's legitimate article titles).
4. **Advisory-first, promote only when earned.** A signal becomes a hard gate
   only after (a) the extraction side can pass it and (b) the threshold holds on
   the crucible corpus. This also protects the no-weaken rule from the other
   direction: never add a hard gate that tempts weakening extraction to pass it.
5. **Calibrate on the crucible.** The 16-doc corpus is the reference set. Each
   threshold passes the known-good docs and trips on the known-bad examples,
   which are frozen as regression fixtures.
6. **Fewer, well-calibrated signals beat many noisy ones.** Alert fatigue is a
   real failure (the IMAGE_NO_VLM noise complaint). Each signal earns its place
   by catching a real defect without flagging good docs.

## 2. Governance protocol (to be added to AGENTS.md)

> AGENT-GATE-PROGRESSION: A content-quality signal enters the suite as an
> ADVISORY metric in `qa_semantic_fidelity.py`, calibrated on the crucible
> corpus with a frozen regression fixture under `tests/`. It is promoted to a
> HARD gate only after (a) the extraction path can pass it on the full corpus
> and (b) the threshold is shown stable across doc classes. Hard gates are
> reserved for deterministic schema invariants. No hard gate may be introduced
> that would be passed by weakening extraction rather than improving it.

## 3. What the schema actually provides (verified 2026-06-07)

Available per-chunk metadata (`metadata.*`): `spatial.bbox` (integer [0,1000]
frame), `spatial.page_width`, `spatial.page_height` (source px), `chunk_type`,
`content_classification`, `corruption_score`, `ocr_confidence`,
`indentation_fidelity`, `vision_status`, `visual_description`,
`original_vlm_type`, `hierarchy.parent_heading`, `page_number`.

NOT available: font size / weight / family. Heading validation therefore cannot
use font hierarchy; it uses bbox spatial-isolation plus element type, and only
at the chunker layer (see Section 5).

CONSIDERED AND REJECTED - a `layout_weight` / `heading_level` proxy projected
from the extractor (review amendment 2026-06-07). Verified the raw MinerU2.5
element carries only `type`, `content`, `bbox`, `merge_prev`
(`mineru_native._mineru_element_to_element`); the VL output is semantic type
labels plus bbox plus rotation, with no numeric layout weight to project. A
derived proxy from bbox height also fails: on real data (IRJET) `paragraph`
elements span bbox-height 29 to 893 (multi-line blocks), so height cannot
separate a heading from a one-line paragraph. The structural hierarchy signal we
already have is the element TYPE (title / header / section_header vs paragraph),
which the chunker already consumes. Revisit only if the extractor is changed to
one that emits font or level metadata.

Existing reusable building blocks: `asset_materializer._is_low_information` /
`_luminance_from_png` (deterministic blank/low-info detection),
`batch_processor._filter_no_visual_images` (the "no distinct non-text visuals"
sentinel filter), `state/context_state.is_valid_heading`, `corruption_score`.

## 4. Issue register (audit finding -> fix-and-guard)

Each row: the defect, real evidence, the extraction-side FIX, and the advisory
GUARD metric.

### F1. Magazine furniture as body chunks and as headings  [Medium-High]
- Evidence: CombatAircraft folios `'22\nAugust 2025 // www.Key.Aero'`
  (bbox y=960-975) emitted as standalone chunks on p13/15/19/21/22/23/24;
  masthead `COMBATAIRCRAFT` and `SCAN THE QR CODE TO ORDER ... shop.keypubliking`
  promoted to headings and carried forward.
- FIX (chunker / batch_processor): drop running-header/footer furniture using
  the SPATIAL signal - a short text element in the top or bottom ~5% of the page
  (bbox y0 > 950 or y1 < 50) whose normalized text repeats across >= 3 pages, or
  matches a folio shape (page-number + masthead/URL). Reject such elements as
  heading candidates.
- GUARD: `furniture_chunk_ratio` (advisory).

### F2. Text-regions-misclassified-as-images survive enrichment  [Medium]
- Evidence: 9 image chunks across 5 docs described
  `"Dense typographic layout; no distinct non-text visuals"`
  (DigitaleFotografie 5). `_filter_no_visual_images` runs at conversion time but
  the description arrives post-conversion from enrichment, so they are never
  filtered.
- FIX: re-run the non-visual filter AFTER enrichment (the enrichment lane, or a
  post-enrichment pass), dropping image chunks whose final description
  self-declares non-visual. NOTE: this case CANNOT be moved pre-VLM - a
  text-as-image region has real entropy/text, so only the VLM can classify it as
  non-visual. It is post-enrichment by necessity. (Contrast F7, which is
  deterministic and IS moved pre-VLM.)
- GUARD: `non_visual_image_ratio` (advisory, `text_as_image`).

### F3. Garbage headings accepted by is_valid_heading  [Low-Med]
- Evidence: `会` (single CJK glyph) as the HarryPotter cover heading;
  `合DANCGING-WIITH`; URL/ad lines. Propagation is small (1-2 chunks each), so
  this is a validation-permissiveness issue, not a carry-forward amplification.
- FIX (chunker `_trusted_heading` / `is_valid_heading`): reject heading
  candidates that are CJK-only or symbol-only on an otherwise Latin-script doc,
  URL/email/masthead-shaped, or folio-shaped. Where the heading ELEMENT bbox is
  available, also require spatial isolation (a distinct block, not a margin
  strip).
- GUARD: `heading_sanity_ratio` (advisory, string-level - see Section 5).

### F4. VLM-path code chunks not markdown-fenced  [Low, RESOLVED: fence]
- Evidence: 0/24 (AIOS) and 0/29 (FluentPython) `modality=code` chunks are
  fenced; the MinerU path fences via `_fence_code`. Indentation is perfect
  (R3=1.0), so this is consistency/presentation, not corruption.
- DECISION (review amendment 2026-06-07): code MUST be fenced. Downstream
  generation models rely on explicit Markdown code boundaries (```...```) to
  switch attention from prose to syntax, and parity with the fenced MinerU path
  removes mixed-route inconsistency.
- FIX (`uir_chunker`): fence VLM-promoted code (the `promoted_modality=code`
  lane) the same way `mineru_native._fence_code` fences the MinerU lane;
  idempotent, content verbatim inside the fence.
- GUARD: `code_fence_consistency` (advisory) - fraction of `modality=code`
  chunks that are fenced; expect 1.0.

### F5. Decorative cover-page garble  [Low, narrow, extraction limit]
- Evidence: HarryPotter p1 -> `'DOWING 1 A HARORYPOWERS 会 ADTHUD A IVEINEOO
  SCHIOASTRO'`. Stylized cover typography neither engine reads.
- FIX: genuinely hard. Option: route low-confidence / high-garble pages to a
  second engine, or flag and down-prioritize. Likely ACCEPT for now.
- GUARD: `text_garble_ratio` (advisory, conservative threshold), reusing the
  existing `corruption_score` / `ocr_confidence` fields rather than a new
  tokenizer heuristic.

### F6. Cross-page duplicate captions / running headers  [Low]
- Evidence: AIOS repeats `"(a) Normalized throughput. Higher is better."` x5;
  ATZ repeats `"ENTWICKLUNGSPROZESSE\nLifecycle Management"` x3.
- FIX: extend dedup to collapse exact/near-exact content repeated across page
  boundaries (captions and running headers), distinct from the existing
  within-page dedup.
- HARD EXCLUSION (review amendment 2026-06-07): cross-page dedup applies ONLY to
  `modality=text` and `modality=image` (captions). It MUST NOT touch
  `modality=table` or `modality=form` - multi-page tables legitimately repeat
  their column-header row on every page, and stripping it from page 2+ destroys
  the table's structural integrity for the downstream LLM. (The existing
  within-page cross-chunk dedup is page-keyed so it does not hit this; the NEW
  cross-page pass is where the trap lives.)
- GUARD: `cross_page_dupe_ratio` (advisory, text+image only). Also catches VLM
  degenerate repetition loops (the CarOK-class failure), so this signal earns
  double duty.

### F7. Blank-image VLM hallucination  [pre-flight fix + guard, pre-emptive]
- Not observed in this crucible (the materializer B2 blank-render guard masks
  it), but a known failure (an 8B VLM hallucinating a gear pattern on a blank
  invoice). Do NOT trust the VLM self-report for blanks.
- FIX (review amendment 2026-06-07) - PRE-FLIGHT, not post: run the
  deterministic `_is_low_information` luminance/entropy check BEFORE the asset is
  routed to enrichment. A blank/low-info asset is dropped immediately, so it
  never burns a VLM call and cannot be hallucinated into a description. This
  inverts the naive "describe then filter" sequence. (`_filter_blank_assets`
  already drops blank-asset IMAGE chunks at conversion time, which is pre-
  enrichment; this amendment ensures the enrichment lane ALSO short-circuits on a
  deterministic blank check, and strengthens that check with an entropy test in
  addition to luminance.)
- GUARD: `blank_image_ratio` (advisory) using the same deterministic check,
  computed independently of the VLM description, as the regression net.

## 5. The chunker-vs-gate split for headings

Heading quality is enforced in two places because the post-conversion JSONL only
retains `parent_heading` as a string (the heading element's bbox is gone):

- STRUCTURAL validation -> chunker (`_assign_headings` / `_trusted_heading`),
  where the heading Element, its bbox, and its type still exist. Reject a single
  `会` glyph, a margin-strip masthead, or a non-isolated block here. This is the
  fix (F1 heading-rejection + F3).
- CONTENT-SANITY validation -> gate metric on the resulting strings (CJK-only,
  URL/email, folio shape). This is the cheap regression net (`heading_sanity_ratio`).

This is the fix-and-guard principle applied cleanly: fix with full structural
context, guard with a cheap string check.

## 6. Signal definitions (initial, to be calibrated)

All advisory, emitted by `qa_semantic_fidelity.py`. Thresholds are starting
points to calibrate against the crucible (Section 7); the calibration step sets
the final numbers.

| Signal | Primary mechanism | Provisional advisory threshold |
|---|---|---|
| `furniture_chunk_ratio` | bbox in top/bottom 5% AND repeats across >=3 pages | warn > 0.05 |
| `non_visual_image_ratio` | VLM self-declared non-visual (text_as_image; post-enrich) | warn > 0.05 |
| `blank_image_ratio` | deterministic low-info luminance/entropy (pre-flight + guard) | warn > 0.02 |
| `cross_page_dupe_ratio` | exact/near-exact text+image content across page boundary (NOT table/form) | warn > 0.03 |
| `heading_sanity_ratio` | CJK-only / URL / folio-shaped heading strings | warn > 0.05 |
| `code_fence_consistency` | fraction of `modality=code` chunks that are fenced | warn < 1.0 |
| `text_garble_ratio` | mean `corruption_score` / low `ocr_confidence` share | warn (corpus-calibrated) |

## 7. Calibration and regression fixtures

1. Run all six signals (read-only) over the existing 16 enriched crucible
   outputs to get baseline distributions per doc class.
2. Set each threshold so the known-good docs pass and the known-bad examples
   trip. Document any doc-class-specific bands (magazines legitimately carry
   long titles and more imagery than academic papers).
3. Freeze regression fixtures under `tests/` from the concrete bad examples:
   - furniture: the CombatAircraft folio at bbox y=960-975 plus a real heading
     (negative + positive).
   - non-visual: an image chunk described "no distinct non-text visuals".
   - heading-sanity: the `会` and `合DANCGING-WIITH` strings (reject) plus a real
     title (keep).
   - cross-page dupe: the AIOS caption x5.
   Each fixture asserts the signal fires on bad and stays quiet on good.

## 8. Sequencing

Phase 0 (runs FIRST, in parallel with F1): stand up the OmniDocBench fidelity
benchmark and record the current-pipeline baseline on the English subset
(PLAN_OMNIDOCBENCH_EVAL Phase 0). This must be in place before the fixes land so
the fidelity floor (Section 10) exists to check against at acceptance.

1. F1 furniture (highest retrieval-noise reducer; spatial signal is decisive and
   already verifiable on CombatAircraft; fixture is trivial to freeze).
2. F3 + heading-sanity (cheap, chunker-local, removes the CJK/ad headings).
3. F7 blank pre-flight + F2 non-visual post-enrichment filter (deterministic
   short-circuit before VLM; VLM-classification filter after).
4. F6 cross-page dedup, text+image only (also nets the VLM-loop class).
5. F4 code fencing in `uir_chunker` (contract resolved: fence) + consistency
   guard.
6. F5 cover garble - advisory guard only; accept the extraction limit for now.

Land each extraction fix and its paired advisory metric together, with the
fixture. Promote any signal to a hard gate only via the Section 2 protocol after
corpus calibration.

## 9. Out of scope

- LLM-judge as a hard gate. An LLM can detect garble/furniture/bad-headings, but
  as a pass/fail gate it is non-deterministic and prone to the lenient-judge
  trap. Permitted only as a sampling auditor (advisory), not a gate.
- Source-fidelity (output vs PDF ground truth) at scale. No labeled ground truth
  exists; spot-checkable via judge sampling only.

## 10. Acceptance (two-axis)

Retrieval-value axis (this plan, our crucible):
- All advisory signals implemented in `qa_semantic_fidelity.py`, each with a
  frozen regression fixture that fires on the known-bad example and is quiet on
  the known-good one.
- Baseline distributions recorded for the 16-doc corpus.
- AGENTS.md carries the AGENT-GATE-PROGRESSION protocol.
- The crucible re-run shows the extraction fixes (F1-F3, F6) measurably lower
  their metrics vs the 2026-06-07 baseline, with no regression to the 16/16
  `QA_PASS` hard-gate result.

Fidelity axis (PLAN_OMNIDOCBENCH_EVAL, ground truth):
- Before any gate-quality fix lands, the OmniDocBench English-subset baseline
  (text edit distance, TEDS, reading order) is recorded for the CURRENT pipeline
  (that plan's Phase 0). This is the fidelity FLOOR.
- After the F1-F7 fixes land, the OmniDocBench re-run must not regress below that
  floor. The point: the retrieval-value fixes (dropping furniture, filtering
  non-visual images, fencing code) must improve chunk hygiene WITHOUT degrading
  transcription fidelity against ground truth. The two axes are checked together
  at acceptance.
