# Phase 6 — `OCR_PATH_HEADING_PROPAGATION` (Firearms) Diagnostic

**Status:** `validated-local` (2026-05-15). Firearms full strict gate
returns **`QA_PASS: failures=0 warnings=0`** under
`scripts/qa_full_conversion.py --source-pdf … --allow-warnings`. The
original plan's Done-when item 1 is met. The earlier audit iteration
labelled this "HEADING-scope closed" because two scope-orthogonal
defects (TEXT `infix_artifacts` and VLM image placeholders) were
still failing; both have been closed in this iteration without
weakening any threshold:

* TEXT `infix_artifacts: 148 → 0` via the new
  `BatchProcessor._repair_infix_step_numbers` chunk-content repair
  that **behaviorally mirrors**
  `scripts/qa_conversion_audit.py::_INFIX_RE` **after the audit's
  newline / stop-word post-filters** (the production regex collapses
  the audit's ``\s+`` + ``"\n" in between`` post-filter into a single
  ``[ \t]+`` capture on the prev→num side; the audit-detector parity
  pin in `tests/test_infix_step_number_repair.py` re-applies the
  audit detector to repaired content and asserts the count drops to
  zero). Universal heuristic; not Firearms-specific; tested across
  docs.
* VLM `image_placeholder_ratio: 0.2424 → 0.0000` via targeted
  enrichment of the 264 `vision_status="pending"` shadow-extraction
  full-page chunks through the canonical Alibaba `qwen3-vl-plus`
  endpoint
  (`scripts/enrich_firearms_pending_only.py`, which delegates to
  `scripts/enrich_image_chunks_v29.py::_enrich_one` so prompt /
  retry / hard-fallback behaviour is unchanged).
**Plan:** `docs/PLAN_V2.10.md` §"Phase 6 — `OCR_PATH_HEADING_PROPAGATION` (Firearms)"
**Probe:** `tests/manual/inspect_firearms_heading_propagation.py`
**Tests:** `tests/test_ocr_path_heading_propagation.py` (70 cases) +
`tests/test_infix_step_number_repair.py` (23 cases, audit-fix iteration)

## Problem statement

Firearms is the canonical doc routed to the `scanned` profile. Its full-doc
strict-gate HEADING coverage lands at 0.72 vs the 0.80 floor. The OCR/
element-by-element extraction lane in `BatchProcessor`
(`_process_page_layout_aware`) does NOT promote Docling `section_header`
items into `ContextStateV2.hierarchy_stack`, so body chunks emitted
page-by-page do not inherit the chapter heading that appeared on an earlier
page.

This is structurally parallel to the Phase 5 HybridChunker-lane problem,
but the lanes are independent — the Phase 5 fix to `_propagate_headings`
does NOT touch the OCR lane.

## Root cause (Candidate A confirmed)

Three concrete observations from the BEFORE probe
(`output/Firearms/ingestion.jsonl`, run on 2026-05-14):

1. **`section_header` recognition exists but does not push state.**
   `BatchProcessor._extract_docling_layout_elements` (around
   [batch_processor.py:1499](../src/mmrag_v2/batch_processor.py#L1499))
   inspects `label_val in ("section_header", "title")` only to *downgrade*
   misclassified long body text to "paragraph" — it never calls
   `update_on_heading` on a real heading.

2. **`LayoutAwareOCRProcessor` flattens the label.** The downstream
   `_convert_docling_elements`
   ([ocr/layout_aware_processor.py:425-436](../src/mmrag_v2/ocr/layout_aware_processor.py#L425))
   maps `section_header` and `title` into the generic `"text"` Region type,
   so the heading signal is lost before chunk emission.

3. **OCR-lane chunks ship `parent_heading=None`.** The per-chunk hierarchy
   build in `_process_page_layout_aware`
   ([batch_processor.py:1232 pre-fix](../src/mmrag_v2/batch_processor.py#L1232))
   hardcoded `parent_heading=None`. The only fallback was
   `_infer_headings_from_text`'s regex heuristics, which promote body-text
   first lines (e.g. "Data: Remington Model 788", "Weight: 712 pounds…")
   into `parent_heading` — producing the garbage top-N values seen in the
   BEFORE probe (see "Before/After top-N attribution" below).

### Why Candidate B was rejected

Candidate B would be "state is pushed but a fresh `BatchProcessor` instance
exists per batch and the push doesn't survive". This is the Phase 5
HybridChunker-lane shape. For the OCR lane it does NOT apply:
`BatchProcessor` is one instance per document (see
`process_pdf` at [batch_processor.py:2623](../src/mmrag_v2/batch_processor.py#L2623)),
the per-batch loop in `_process_batch_layout_aware` shares the same
`self`, and `self._context_state` is initialised once per document
([batch_processor.py:2671](../src/mmrag_v2/batch_processor.py#L2671)) and
already threaded into the HybridChunker lane. Re-using it here is the
minimal-state path.

### Why Candidate C was rejected

Candidate C would be "`is_valid_heading` rejects Firearms' actual heading
shape". The BEFORE probe confirmed `is_valid_heading` accepts the canonical
Firearms headings ("MAUSER", "WINCHESTER", "REMINGTON", "INTRODUCTION",
"Mauser 1898", "BROWNING AUTO-5", etc.). No validator relaxation is
needed; no relaxation was performed.

## Fix shape (Candidate A)

Two surgical changes in one method (single push site on the OCR lane):

1. **`src/mmrag_v2/state/context_state.py`** — add
   `ContextStateV2.get_section_heading()`. Returns the latest
   `level >= 1` heading, skipping the level-0 doc-title breadcrumb seeded
   by `create_context_state`. The OCR lane reads this; if no real heading
   has been pushed yet, the read returns `None` (so `parent_heading=None`
   is the pre-heading default, and downstream
   `_infer_headings_from_text` / vision-aided front-matter remain free to
   seed front-matter pages).

2. **`src/mmrag_v2/batch_processor.py`** — two additions inside
   `_process_page_layout_aware`:

    a. A new helper `BatchProcessor._promote_ocr_section_headers(...)`
       (called once, at the top of `_process_page_layout_aware`) walks the
       raw `docling_elements` and, for each `section_header` / `title`
       label, calls `self._context_state.update_on_heading(text, level=1)`.
       `update_on_heading` re-uses the central `is_valid_heading`
       validator; garbage section_headers (repeated tokens, code/JSON
       shape, generic Docling buckets) are rejected at the validator
       boundary. No parallel OCR-lane validator was introduced.
    b. Chunk-emission reads `self._context_state.get_section_heading()`
       once per page and threads the returned heading into both
       `parent_heading` and the breadcrumb path. When `None`, the legacy
       `[doc_title, "Page N"]` breadcrumb shape is preserved.

This is one production push site on the OCR lane. The HybridChunker lane's
`_propagate_headings` call inside `process_pdf` remains untouched — the
structural pin in `tests/test_vision_aided_front_matter.py` ("one call
site") still holds, and a parallel pin
(`test_hybrid_chunker_lane_propagate_headings_call_count_unchanged`) was
added to `tests/test_ocr_path_heading_propagation.py` to catch any
regression.

## Tests (red → green)

`tests/test_ocr_path_heading_propagation.py` — 70 cases:

- Original plan-required contracts: section-header push, replacement,
  cross-batch carry-forward, garbage rejection, doc-title skip, title
  label handling, non-heading no-ops, empty/no-state no-ops, and OCR /
  HybridChunker structural pins.
- Audit-fix expansions: ordered per-chunk attribution (body before
  heading, body after heading, multiple same-page headings, garbage
  non-displacement), question/exclamation heading positives, terminal-
  period and numbered-body-step negatives, and single-page form gate
  coverage on both the per-chunk and fallback paths.

All 70 cases (collected as of 2026-05-14) pass under bare `pytest`.

## Before / After top-N attribution (BEFORE)

Probe output: `tests/manual/inspect_firearms_heading_propagation.py`
(run on `output/Firearms/ingestion.jsonl`, BEFORE the Phase 6 fix).

```
text_chunks=1094 attributed=790 null=304 null_ratio=0.278

count  valid  parent_heading
141    False  'Data: Remington Model 788'
 68    True   'SMLE NO.1,'
 67    False  'Data: Mossberg Model 479'
 57    True   'RUGER |'
 41    True   'U.S. 1903'
 38    True   'Weight: 712 pounds original version of this gun, the Model 740, was first'
 36    False  'Data: Ruger No. 1'
 34    True   'REMINGTON'
 32    False  'Data: Marlin Model 336'
 32    False  'Data: Winchester Model 94'
```

Failure mode: 6 of the top 10 attributions are body-text fragments
("Data: …" model-spec lines or sentence fragments like "Weight: 712 pounds
original version of this gun, the Model 740, was first") promoted by
`_infer_headings_from_text`'s regex heuristics. 5 of those 10 don't even
pass `is_valid_heading`; the other 5 are accepted by the validator but
are still body text, not real chapter headings — `is_valid_heading` is
designed to reject *structural garbage* (repeated tokens, code shape,
page numbers), not body-prose first-lines.

## Before / After top-N attribution (AFTER)

Probe output: `tests/manual/inspect_firearms_heading_propagation.py`
(run on `output/Firearms_phase6e/ingestion.jsonl`, the final audit-fix
close artifact — ordered per-chunk attribution + narrowed terminal-period
rule + single-page push gate applied on both call sites, with TEXT infix
repair and targeted VLM enrichment also applied).

```
text_chunks=1094 attributed=1091 null=3 null_ratio=0.003

count  valid  parent_heading
696    True   'Disassembly:'
145    True   'Reassembly Tips:'
 19    True   'General Instructions:'
 14    True   'WNCHESTER MODEL 70'        (real chapter; OCR typo for WINCHESTER)
 12    True   'SAVACE MODEL I10'           (real chapter; OCR typo for SAVAGE)
 11    True   'REMINGTON MODEL 700'
 10    True   'MOSSBERG MODEL 479'
 10    True   'REMINGTON ROLLING BLOCK'
 10    True   'WINCHESTER MODEL 71'
 10    True   'NOSTALGIA'
```

All 10 attributions are real chapter or section titles from the source
manual. The top-5 human-readability gate prints
``PASS: all top-5 headings pass is_valid_heading``.

The 3 NULL chunks are page-1 front-matter (book-cover blurb / TOC
intro lines) emitted before any Docling section_header on the page —
ordered attribution correctly leaves them unattributed rather than
inheriting a heading that arrives later in the page stream. This is
the exact contract pinned by
`test_ordered_body_before_first_heading_is_null`.

Compared to the pre-audit phase6c page-level "promote all then read
once" output (`670 / 171 / 21 / 14 / 12 / 11 / 10 / 10 / 10 / 10`),
ordered attribution shifts a modest number of chunks toward
``Disassembly:`` (the most-common chapter sub-section) because chunks
that previously inherited a later heading on the same page now
correctly inherit the prior chapter's `Disassembly:` heading. None of
the top-10 entries became garbage as a result.

## Verdict (corrected after audit 2026-05-14)

## Verdict (final, 2026-05-15)

`scripts/qa_full_conversion.py output/Firearms_phase6e/ingestion.jsonl
--source-pdf data/technical_manual/Firearms.pdf --allow-warnings`
returns **`QA_PASS: failures=0 warnings=0`** with:

* `HEADING: PASS  coverage 1091/1094` (3 NULL = page-1 front matter,
  correctly unattributed under ordered per-chunk attribution).
* `TEXT: PASS  infix_artifacts: 0` (was 148 before
  `_repair_infix_step_numbers`).
* `IMAGE: PASS  image_placeholder_ratio: 0.0000` (was 0.2424 before
  targeted VLM enrichment of the 264 pending shadow chunks).
* `UNIVERSAL_PASS  page_coverage=292/292`.
* `SEMANTIC_PASS`.

## Verdict (audit-iteration trail, 2026-05-14 — HISTORICAL)

> ⚠️ HISTORICAL ONLY. The findings below describe the pre-audit-fix
> state on `output/Firearms_phase6d/ingestion.jsonl`. The current
> validated-local state is the QA_PASS in the section above, against
> `output/Firearms_phase6e/ingestion.jsonl`. Retained as an audit
> trail to make the iteration sequence reviewable.

The first Phase 6 close iteration claimed "QA_FAIL only on VLM
image_placeholder_ratio". That was **inaccurate**. A fresh
`qa_full_conversion.py` run on the post-Phase-6 Firearms output
(``output/Firearms_phase6d/ingestion.jsonl``, pre-audit-fix) reported
`QA_FAIL: failures=3 warnings=2`, with three distinct findings:

- `HEADING: PASS  coverage 1091/1094 (100% reported, 99.7% actual)`
  — Phase 6's actual win. Three NULL parent_headings on page 1 are
  pre-section front-matter chunks (correctly unattributed under
  ordered attribution).
- `TEXT: FAIL  infix_artifacts: 148  micro_<30: 225 (advisory)` —
  **pre-existing**, not introduced by Phase 6. Re-running the same
  `qa_conversion_audit.py` against the BEFORE-state output
  (`output/Firearms/ingestion.jsonl`, pre-Phase-6) reports the
  identical `infix_artifacts: 148` value, confirming the TEXT failure
  is unrelated to OCR-lane heading propagation. *Closed in the
  audit-fix iteration by ``_repair_infix_step_numbers``; see the
  QA_PASS section at the top of this document.*
- `IMAGE: FAIL  image_placeholder_ratio=0.2424` (264 of 1089 image
  chunks lack a real VLM description) — **scope-orthogonal but not
  pre-existing**. The Firearms baseline at
  `output/Firearms/ingestion.jsonl` has
  `image_placeholder_ratio=0.0000` because it was produced by an
  earlier VLM-enriched run; the phase6d reconvert ran with
  `--vision-provider none` per the plan's "VLM re-enablement is a
  separate workstream" guidance, so the same image chunks had
  placeholder content. *Closed in the audit-fix iteration by
  ``scripts/enrich_firearms_pending_only.py``; see the QA_PASS
  section at the top of this document.*

## Quality acceptance (Phase 6 non-negotiables)

| Check | Required | Result |
|---|---|---|
| Top-5 attributed `parent_heading` values all human-readable | yes | yes — all top-5 are real chapter/sub-section titles |
| None of top-5 is a repeated-token string / code/JSON fragment / generic Docling bucket >25 % / content equal to chunk's own body | yes | yes — top entries `Disassembly:` / `Reassembly Tips:` are document-specific sub-section names (the firearms manual is structurally a series of disassembly procedures, so the high `Disassembly:` count reflects content shape, not a generic bucket); rifle-chapter all-caps entries are document-specific real headings |
| Production has exactly ONE additional propagation site on the OCR lane | yes | yes (`_promote_ocr_section_headers` called once from `_process_page_layout_aware`) |
| Heading validation lives in `state/context_state.py` only | yes | yes (no `_is_valid_ocr_heading` in `batch_processor.py`) |
| No `chunk_dict` mutation at write time anywhere | yes | yes (edits land on `IngestionChunk` before serialization, via the existing chunk-emission loop) |
| HybridChunker-lane `_propagate_headings` call count == 1 inside `process_pdf` | yes | yes (pinned by Phase 5 + Phase 6 tests) |
| `is_valid_heading` tightenings retain Phase 5 garbage rejections (Type Type TypeTypeTypeType, GRAPH DATA: {) | yes | yes (regression suite pinned in `tests/test_ocr_path_heading_propagation.py`) |

## `is_valid_heading` tightenings

The OCR-lane fix surfaced a class of Docling layout mis-classifications
that the pre-Phase-6 validator accepted. Two universal rules were added
to `state.context_state.is_valid_heading` (the central validator —
no parallel OCR-lane validator was introduced):

1. **Terminal-period sentence-shape.** Text ending with `.` and
   containing ≥ 5 words is body-prose, not a heading. Canonical
   Firearms shapes caught: `"5. Drift out the trigger cross-pin toward
   the right."`, `"6 Drift out the trigger pin toward the left, and
   removeit."`. The threshold is conservative: short titles ending in
   a period (`"References."`, `"5. Conclusions."`, `"Et al."`) pass.

   **Audit refinement (2026-05-14):** the original rule rejected
   ``.!?`` and false-negatived real question / exclamation headings
   (`"What Is an AI Agent?"`, `"Why Use Retrieval Augmented
   Generation?"`, `"2. What are AI agents?"`). The rule was narrowed
   to terminal `.` only; `?` and `!` are now accepted.
   `test_is_valid_heading_accepts_question_headings` and
   `test_is_valid_heading_accepts_exclamation_headings` pin the
   refinement.

2. **Numbered-prefix body-case shape.** Text matching `^\d+(?:\.\d+)*\.?\s+`
   with ≥ 2 lowercase content words after the prefix (excluding stop
   words) is a numbered body-step — a Title-case verb followed by
   lowercase nouns. Canonical Firearms shape caught:
   `"6. Remove the hammer downward"`. Phase 5 Devlin real headings
   (`"5.1 Notation and Preliminaries"`, `"14. Linking to Memory and
   Context"`, `"1 - The New Age of AI Agents"`, `"2. Fine-Tuning:
   Teaching the Model to Specialize"`) all have 0 lowercase content
   words after the prefix and pass.

Both rules are pinned by parameterised test cases in
`tests/test_ocr_path_heading_propagation.py`, including
positive-control cases for Devlin/Firearms real headings and explicit
re-verification that the Phase 5 audit-named garbage
(`"Type Type TypeTypeTypeType"`, `"GRAPH DATA: {"`) is still rejected.

## Ordered same-page attribution (audit fix 2026-05-14)

The first Phase 6 iteration did a page-level "promote all
section_headers then read state ONCE for the whole page" assignment,
which gave every chunk on a page the LAST heading on that page —
including chunks emitted before the heading appeared in the Docling
element stream. The audit called this out as a same-page attribution
weakness.

The corrected design preserves Docling's `section_header` / `title`
label through the OCR-lane pipeline by extending the lightweight
`Region` (in `src/mmrag_v2/ocr/layout_aware_processor.py`) and
`ProcessedChunk` dataclasses with an `is_heading: bool` field. The
label is set in `_convert_docling_elements` and propagated through
`_process_text_region` to the emitted `ProcessedChunk`. The
`BatchProcessor._attribute_ocr_chunk_heading` helper is then invoked
once per `ProcessedChunk` in the order returned by
`LayoutAwareOCRProcessor.process_page`; when it sees `is_heading=True`
it pushes the chunk's content into `ContextStateV2` via
`update_on_heading` (subject to the central `is_valid_heading`
validator), then reads `get_section_heading()` for that chunk's
`parent_heading`. Body chunks emitted BEFORE the first heading on a
page therefore inherit the previously-active heading (or `None` if
none exists yet), and chunks emitted between two same-page headings
attribute to the first heading while chunks after the second attribute
to the second — pinned by
`test_ordered_body_before_heading_inherits_prior`,
`test_ordered_body_before_first_heading_is_null`,
`test_ordered_multiple_headings_on_same_page_switch_attribution`,
`test_ordered_garbage_heading_does_not_displace_prior`,
`test_ordered_question_heading_promotes_and_carries`.

The fallback `_promote_ocr_section_headers` push remains in place but
is only invoked when **no** chunk in the stream carries
`is_heading=True` — i.e. on the VLM-fullpage and Tesseract-fullpage
emit paths in `LayoutAwareOCRProcessor.process_page` that collapse a
page into a single synthesized chunk. Without the fallback,
Docling-recognised headings on those rare pages would be silently
dropped from state, breaking cross-page propagation.

Both push entry points (per-chunk and fallback) respect the
single-page-doc gate (`self._doc_total_pages > 1`) so the 0013
form-detection contract holds — pinned by
`test_ordered_single_page_doc_skips_per_chunk_push`.

## Single-page push gate

The first Phase 6 iteration regressed the cross-profile smoke at
`scanned/0013_140302111325_001` (a single-page German invoice). On
that document Docling tagged the layout-prominent total line
`"Gesamt-Brutto 1.949,60 EUR"` as `section_header`; the OCR-lane fix
faithfully promoted it, which flipped the audit gate's form-detection
heuristic (`form := scanned + total_pages ≤ 5 + heading_coverage < 0.10`).
With `heading_coverage = 1.0` the doc was no longer treated as a form,
and the `micro_non_label_ratio` check then fired (single-page invoices
have many short German lines).

The structural fix: the Phase 6 OCR-lane push is for **inter-page**
heading propagation. On a single-page document there is no cross-page
propagation to do. `_promote_ocr_section_headers` now early-returns
when `self._doc_total_pages` is unset or ``<= 1``. The gate is purely a
document-shape constraint, not filename- or page-number-specific; it
applies uniformly to any single-page input (forms, invoices, posters,
single-page summaries). Pinned by
`test_single_page_doc_skips_push`,
`test_unknown_total_pages_skips_push`,
`test_multi_page_doc_pushes_normally`.

After the gate, the 0013 reconvert produced
`parent_heading=None` for all 17 text chunks, restored
`document_type=form`, and recovered `GATE_PASS [form: micro_non_label +
label-orphan checks skipped]` — matching the pre-Phase-6 v2.9.0-rc1
baseline.

## Regression evidence

| Doc | Profile | Run | HEADING coverage | Verdict | Notes |
|---|---|---|---|---|---|
| Firearms (292 pages) | scanned | BEFORE (pre-Phase 6) | 0.72 (304/1094 null) | strict-gate FAIL (HEADING + TEXT `infix_artifacts: 148` + VLM) | top-5 had 2 `Data: …` garbage; TEXT failure already present |
| Firearms (292 pages) | scanned | Phase 6 v1 (page-level promote) | 1.00 (0/1094 null) | HEADING PASS + UNIVERSAL_PASS; overall QA_FAIL on **same** pre-existing TEXT (`infix_artifacts: 148`) + VLM | first iteration; same-page attribution coarse |
| Firearms (292 pages) | scanned | Phase 6d (audit-iter, ordered attribution) | 0.997 (3/1094 null on page-1 front matter) | HEADING PASS + UNIVERSAL_PASS; overall QA_FAIL on **same** pre-existing TEXT (`infix_artifacts: 148`) + VLM | ordered per-chunk attribution; top-5 all real chapter/section titles |
| **Firearms (292 pages)** | scanned | **FINAL (Phase 6e, audit-fix close)** | **0.997 (3/1094 null on page-1 front matter)** | **`QA_PASS: failures=0 warnings=0`** (HEADING PASS + TEXT PASS + IMAGE PASS + UNIVERSAL_PASS + SEMANTIC_PASS) | infix repair → `infix_artifacts: 0`; targeted VLM enrichment → `image_placeholder_ratio: 0.0000`; top-5 unchanged from phase6d |
| 0013 invoice (1 page) | scanned | BEFORE | n/a (no text headings) | GATE_PASS [form] | form-detection heuristic active |
| 0013 invoice (1 page) | scanned | Phase 6 v1 (no gate) | 1.0 — `"Gesamt-Brutto 1.949,60 EUR"` | GATE_FAIL micro_non_label | form detection flipped — REGRESSION |
| 0013 invoice (1 page) | scanned | Phase 6 + gate on fallback only | 1.0 — `"Kunden-Nr.: …"` | GATE_FAIL micro_non_label | per-chunk path bypassed the gate — REGRESSION |
| 0013 invoice (1 page) | scanned | Phase 6 + gate on BOTH paths | n/a (push skipped) | GATE_PASS [form] | restored to baseline |
| Earthship_Vol1 (236 pages) | scanned | BEFORE | 0/548 null (100%) | (smoke OK) | Greenhouse-like regression control |
| Earthship_Vol1 (236 pages) | scanned | AFTER (Phase 6d) | 0/549 null (100%) | HEADING PASS + UNIVERSAL_PASS; overall QA_FAIL on VLM `image_placeholder_ratio=0.4357` (Phase 6 reconvert ran without VLM enrichment, same scope-orthogonal pattern as Firearms; not introduced by the heading-propagation fix); `micro_non_label_ratio=0.033` (limit 0.12) | no HEADING coverage regression; chunk counts within ±1; top-10 all real chapter/section titles or pre-existing OCR-garbled section names |
| Smoke (11 docs × 10 pages) | mixed | BEFORE | 11/11 GATE_PASS | | Phase 5 close baseline |
| Smoke (11 docs × 10 pages) | mixed | Phase 6 v1 (no gate) | 10/11 GATE_PASS | | 0013 regression |
| Smoke (11 docs × 10 pages) | mixed | Phase 6 v2 (gate on fallback only) | 10/11 GATE_PASS | | per-chunk path bypassed gate |
| Smoke (11 docs × 10 pages) | mixed | Phase 6 audit-fix (gate on both paths) | **11/11 GATE_PASS + 11/11 UNIVERSAL_PASS** | | regression resolved |

| Test suite | Count | Status |
|---|---|---|
| `tests/test_ocr_path_heading_propagation.py` (incl. ordered-attribution + question/exclamation + audit-fix pins) | 70 | passing |
| `tests/test_infix_step_number_repair.py` (audit-fix iteration; incl. audit-detector parity test) | 23 | passing |
| `tests/test_hybrid_chunker_heading_propagation.py` (Phase 5) | 7 | passing — no regression |
| `tests/test_cross_page_split_page_attribution.py` (Phase 4) | 27 | passing — no regression |
| `tests/test_vision_aided_front_matter.py` (structural pin) | 8 | passing — `_propagate_headings(` call count in `process_pdf` unchanged at 1 |
| Full bare `pytest tests/ -x --ignore=tests/manual -q` | 953 | passing, 14 skipped |
