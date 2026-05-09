# Plan: v2.9 Phase 4 — Localized strict-gate hard failures

**Status:** Draft (2026-05-09). Authored after Phase 3 closure (`51e897b`).
**Master plan:** `docs/PLAN_V2.9.md` Phase 4.
**Predecessors:** Phase 1 closed `df91061` (2026-05-07); Phase 2 closed `29a7242` (2026-05-08); Phase 3 closed `649c952`/`51e897b` (2026-05-09).
**Phase nature:** Localized fixes + quarantine policy. No broad-corpus operations.

## Why this plan exists

Per `docs/PLAN_V2.9.md` §3 Phase 4, the remaining v2.9 hard failures after Phases 1-3 are localized strict-gate breaks on specific docs/pages. The diagnostic session prior to drafting this plan confirmed the active failure set against the Phase 3-enriched JSONLs and against the Phase 2 Combat probe.

Diagnostic findings (2026-05-09):

| Failure class | Source | Verdict |
|---|---|---|
| Adedeji p301 — TABLE_CORRUPTION | `output/bridge_phase2plus/pdf_adedeji/ingestion.jsonl` (Phase 3 enriched) | **Real defect.** Page-level corruption: all 9 chunks (1 table + 8 text/list_items) carry the same repeated phrase. Source PDF p301 is a clean back-index page (page-280 marker at bottom). Phase 1 dense-index router missed because Docling did not label this page as `document_index`. |
| Combat Aircraft p66 — LOCALIZED_CORRUPTION | `output/probe_phase2_combat/ingestion.jsonl` | **Source-PDF defect.** Magazine PDF with native font/encoding corruption (replacement chars `�`, fragments like `5J()lll t-I::>`). Docling extracts garbage from garbage. Quarantine policy applies. |
| Hao MISSING_PAGES (5 pages: 23, 117, 201, 203, 379) | `output/bridge_phase2plus/pdf_hao/ingestion.jsonl` | **Not a defect.** All 5 are blank source pages (0 text chars in pypdfium2 extraction). Already classified as `MISSING_PAGES_BLANK` by `qa_full_conversion.py --source-pdf`. |
| Adedeji MISSING_PAGES (4 pages: 11, 21, 167, 197) | `output/bridge_phase2plus/pdf_adedeji/ingestion.jsonl` | **Not a defect.** Same — all 4 are blank source pages. |
| Adedeji code_indentation_fidelity 0.886 (<0.90) | semantic-fidelity advisory | **Carry-forward from Phase 2.** CodeFormulaV2 OFF in active config. `docs/DECISIONS.md` Amendment 2026-05-03 already permits selective enablement. |
| Firearms HEADING coverage 72% (target ≥80%) | `output/probe_phase2_firearms/ingestion.jsonl` | **Real defect, separate root cause.** 304/1094 paragraph chunks lack `parent_heading`. Sample chunks show heading text inline (e.g. `'ASSEMBLY/DISASSEMBLY'`) but Docling layout did not promote it. Magazine-style scanned-image layout edge case. |
| KI EPUB label/universal failure | strict-gate snapshot | **Pre-existing.** Phase 4 plan permits deferral with explicit user sign-off. |

## Non-negotiable contract

| Contract | Source | Hard requirement |
|---|---|---|
| **No filename-specific production logic** | Plan §3 Phase 4 rules | Fixes must be by content shape, profile, or class — never `if "Adedeji" in path:`. |
| **Negative tests / strict-gate assertions stay strict** | `CLAUDE.md` Test Contract Integrity | Do not loosen any assertion to make a current run pass. If a test expectation appears wrong, document the requirement change before editing. |
| **Quarantine over corruption** | Plan §3 Phase 4 rules | Prefer dropping a page than emitting known-corrupt content. Quarantine signals must be observable in the JSONL (e.g. `metadata.page_quarantined=true` + reason). |
| **`qa_full_conversion.py --source-pdf` is the canonical strict-gate command** | Diagnostic finding | Hao + Adedeji blank-page MISSING_PAGES "failures" disappear under `--source-pdf`. The doc-level invariants must reflect this. |
| **No re-conversion of broad corpus** | Phase 5 owns broad reconversion | Phase 4 only re-converts the 2-3 docs whose code paths it touches (Adedeji + Combat + Firearms). |

## Out of scope

- Broad 34-doc reconversion / Qdrant refresh — Phase 5.
- Local VLM lane — deferred to v2.10.
- New profile creation — Firearms stays in `technical_manual` profile (or its current routed profile); no `firearms_magazine` profile.
- Adedeji `code_indentation_fidelity` — assessed in Step 5 below, may be deferred to v2.10 if CodeFormulaV2 enablement turns out to be cross-cutting.

## Pre-flights (run BEFORE Step 1)

### P-1: full test suite green

```bash
conda run -n mmrag-v2 pytest tests/ -q
# Expect: 685 passed, 14 skipped (the post-Phase-3 baseline). Phase 4 must not drop the count.
```

### P-2: confirm `qa_full_conversion.py --source-pdf` is the canonical strict-gate command

```bash
# Hao + Adedeji should each report 0 failures from MISSING_PAGES once --source-pdf is used.
conda run -n mmrag-v2 python scripts/qa_full_conversion.py \
  output/bridge_phase2plus/pdf_hao/ingestion.jsonl \
  --source-pdf "data/technical_manual/Hao B. Machine Learning Platform Engineering. Build...for ML and AI systems 2026.pdf" \
  | tail -5
# Expect: MISSING_PAGES_BLANK info line, 0 failures, ASSET_TINY warning only.

conda run -n mmrag-v2 python scripts/qa_full_conversion.py \
  output/bridge_phase2plus/pdf_adedeji/ingestion.jsonl \
  --source-pdf "data/technical_manual/Adedeji A. GenAI on Google Cloud. Enterprise Generative AI Systems...Agents 2026.pdf" \
  | tail -5
# Expect: MISSING_PAGES_BLANK info line, 1 failure (TABLE_CORRUPTION p301).
```

**Stop condition:** Phase 4 doesn't start until both pre-flights are green.

---

## Step 1 — Document `--source-pdf` as canonical strict-gate

**Why:** `qa_universal_invariants.py` already advises `(advisory; see qa_full_conversion.py with --source-pdf for blank-page-aware check)`. The ambiguity has been costing diagnostic time across phases (Hao + Adedeji both reported phantom MISSING_PAGES failures under the no-flag form).

**What:**

1. Update `docs/QUALITY_GATES.md` to specify the canonical strict-gate invocation as `qa_full_conversion.py --source-pdf <PDF>`.
2. Update `CLAUDE.md` "Acceptance Gate" section the same way.
3. Update `scripts/smoke_multiprofile.sh` and `scripts/acceptance_technical_manual.sh` to pass `--source-pdf` automatically (resolve PDF path from JSONL `source_file` header field).

**Done when:** Doc updates committed; both smoke scripts run cleanly against current Phase 3 outputs and report the blank-page count as `INFO`, not `FAIL`.

**Tests:** No code changes that need new unit tests. Add a regression test in `tests/test_qa_full_conversion.py` (create if missing) that calls the script with `--source-pdf` against a fixture and asserts the `MISSING_PAGES_BLANK` info path is taken.

**Estimated effort:** 0.5 day. Documentation + 1 regression test.

---

## Step 2 — Adedeji p301: dense-back-index detection by content shape

**Why:** Phase 1's dense-index router relies on Docling's `document_index` label. Adedeji p301 is a textbook back-index page that Docling did not label as such, so the router did not fire. The result: all 9 emitted chunks carry the same `'talent and culture, 246-250 vision and leadership ...'` repetition — page-level garbage.

**Diagnostic data:**

```text
chunk_id                                  modality  chunk_type  len
131b7b54c411_301_table_34fb5aa5           table     None         68741   # corrupted single-column table
131b7b54c411_301_text_13861b85            text      list_item    1388   # 'talent and culture, 246-250 vision and leadership ...'
131b7b54c411_301_text_13861b85_o2         text      paragraph      19
131b7b54c411_301_text_add60009            text      list_item    1372   # same repetition
131b7b54c411_301_text_add60009_o2         text      list_item    1386
131b7b54c411_301_text_add60009_o3         text      list_item    1082
131b7b54c411_301_text_1d0b0199            text      list_item    1437
131b7b54c411_301_text_1d0b0199_o2         text      paragraph      42
131b7b54c411_301_text_1d0b0199_o3         text      paragraph    1206
```

The clean source-PDF text via pypdfium2:

```text
talent and culture, 246-250
vision and leadership, 240
AIOps, 257, 260, 268
AlloyDB vector storage on Google Cloud, 49
Amazon Textract, 31
... [continues as a normal multi-column index]
```

**Two candidate fix paths.** Decide via Step 2a probe.

**Step 2a — content-shape probe (cheap, 1 day):**

Add a content-shape detector that runs before HybridChunker emits a page's chunks. Heuristic candidates (whichever proves more selective on a 5-doc probe):

- `repeated_phrase_ratio`: longest n-gram (length ≥ 6 tokens) that recurs ≥ 5 times across the page's would-be emitted text → if > 0.30, page is corrupt.
- `index_entry_ratio`: fraction of lines on the page matching `^[A-Z][A-Za-z\s,\-/&]+,\s*\d+(-\d+)?$` (index entry shape) → if > 0.50 AND page text has a `Index` / `^\d+ \| Index$` page footer marker, page is a back-index.

Probe corpus: Adedeji p301 (positive); Hao p472-475 (Hao's clean back-index, currently chunked normally — must NOT trigger); Adedeji p3-5 (front-matter, must not trigger); Kimothi p253 (Phase 1 baseline, must trigger if `--profile-override technical_manual`).

**Step 2b — fix path A: route via Phase 1 emitter (preferred):**

If the content-shape detector classifies the page as `dense_back_index`, add it to the `skip_pages` set fed to `MmragChunkingSerializerProvider` and emit chunks via `_emit_dense_index_page_chunks` (already shipped in `processor.py`). Reuses the proven Phase 1 path.

**Step 2b — fix path B: quarantine (fallback):**

If content-shape detection is too noisy or the emitter misbehaves on Adedeji's structure, quarantine the page: emit no chunks for it, with `metadata.page_quarantined=true, reason="dense_back_index_detected_layout_failure"` recorded on the doc-level header.

**Done when:**

- Adedeji p301 emits chunks via the Phase 1 emitter OR is quarantined cleanly.
- `qa_full_conversion.py --source-pdf <Adedeji>` reports `TABLE_CORRUPTION=0` and `LOCALIZED_CORRUPTION=0`.
- Probe corpus negative tests pass: Hao p472-475 + Adedeji front matter must NOT be flagged.

**Tests:**

- `tests/test_dense_back_index_detector.py` (new) — 5 fixtures (positive: Adedeji p301 grid, Kimothi p253; negative: Hao back-index, Adedeji front matter, normal body text).
- Extend `tests/test_hybrid_chunker_dense_page_router.py` if fix path A is taken — assert the new content-shape path is exercised when Docling label is absent.

**Estimated effort:** 2 days (1 for probe + heuristic tuning; 1 for fix + tests).

---

## Step 3 — Combat p66: source-PDF font-corruption quarantine

**Why:** Combat Aircraft p66 source PDF has native font/encoding corruption: extracted text contains replacement chars (`�`) and fragments like `5J()lll t-I::> `Nult: 1`. No layout fix can recover meaningful content from already-garbled glyph data. The Phase 4 plan rule "prefer quarantine over emitting corrupted content" applies directly.

**Diagnostic data:**

- Source PDF p66 via pypdfium2: 5145 chars, but riddled with `�` and broken word boundaries.
- Output: 1 corrupted table (4554 chars of garbage) + 2 image placeholders. Strict gate: `LOCALIZED_CORRUPTION p66=1`.

**What:**

1. Add a per-page source-PDF hygiene check that runs upfront: if `replacement_char_ratio > 0.02` OR `non_ascii_garble_ratio > 0.10` (TBD calibration via Step 3a probe), mark the page for quarantine.
2. Quarantine = emit zero chunks for the page; record `metadata.page_quarantined=true, reason="source_pdf_font_corruption"` on the doc header.
3. Strict-gate behavior: `qa_full_conversion.py` should treat quarantined pages as `PAGE_QUARANTINED` info, not as MISSING_PAGES failure.

**Step 3a — probe:**

Run the candidate hygiene rule against:
- Combat p66 (must trigger).
- Combat p1-65 (must not trigger).
- Hao + Adedeji full corpus (must not trigger anywhere — these are clean PDFs).
- ATZ Elektronik (German magazine — must not trigger; its German text has high non-ASCII rate but legitimate).

If any false positives, raise the threshold or refine to require co-occurrence of `�` chars + glyph fragments.

**Done when:**

- Combat p66 produces zero chunks (or a single placeholder chunk with `page_quarantined=true`).
- `qa_full_conversion.py` reports `PAGE_QUARANTINED p66`, not `LOCALIZED_CORRUPTION p66`.
- No regression on the probe corpus.

**Tests:**

- `tests/test_source_pdf_hygiene.py` (new) — fixtures from Combat p66 (positive) + Hao/Adedeji/ATZ samples (negative).
- Extend `scripts/qa_full_conversion.py` to recognize the `page_quarantined` signal.

**Estimated effort:** 1.5 days.

---

## Step 4 — Firearms HEADING coverage drift

**Why:** Firearms HEADING coverage = 790/1094 = 72% (target ≥80%). Of the 304 paragraph chunks lacking `parent_heading`, sample inspection shows the heading text IS present inline in the chunk content (e.g. `'ASSEMBLY/DISASSEMBLY'` appears in the chunk body, but Docling did not promote it to a heading). Firearms is a scanned/OCR'd magazine with complex visual layout; Docling's heading detector is missing visual headings that a human reader would recognize.

**Two candidate fix paths.** Investigate via Step 4a probe.

**Step 4a — probe (cheap, 1 day):**

Determine the root cause:
1. Are these headings detected by Docling at all (as `SectionHeader`/`Title`)?
2. If detected, why not propagated to following paragraphs' `parent_heading`?
3. Is `docling-hierarchical-pdf` (font-based heading reclassifier per project memory) enabled for the Firearms profile? If not, does enabling it close the gap?

Output: a diagnostic note pinpointing whether the failure is detection (Docling missed the heading visually) or propagation (heading detected but breadcrumb logic dropped it).

**Step 4b — fix path A: enable `docling-hierarchical-pdf` for the routed profile:**

If the probe shows headings exist as visual elements but Docling classified them as paragraphs, the font-based reclassifier should rescue them. Project memory notes the integration is already in place; this would be enabling it for the routed profile (technical_manual).

**Step 4b — fix path B: accept and document:**

If Step 4a shows the headings are genuinely undetectable from the source PDF (e.g. heavy OCR artifacts hide them), the 72% coverage is the document's natural ceiling. Action: per `docs/PLAN_V2.9.md` Phase 4, document the gap with explicit user sign-off and snapshot rationale, then update the strict gate to allow Firearms-profile docs to have HEADING coverage ≥70% instead of ≥80%. **Do not weaken the gate globally.**

**Done when:**

- Firearms HEADING coverage either passes ≥80% via the reclassifier OR has documented user sign-off + a profile-scoped gate change.
- The fix or gate change is profile-scoped (no filename match).

**Tests:**

- `tests/test_firearms_heading_propagation.py` (new) — focused on the heading-detection path. Assert `parent_heading` is set on at least one paragraph chunk per page that follows a recognized SectionHeader.

**Estimated effort:** 1.5 days (mostly probe + reclassifier verification).

---

## Step 5 — Adedeji code_indentation_fidelity 0.886 — assess and route

**Why:** Phase 2 carry-forward. Active config has CodeFormulaV2 disabled. Decision Amendment 2026-05-03 permits selective enablement based on cheap code-evidence detection. Adedeji has 35 code chunks with 0.886 fidelity (one flat-code chunk on p299 is the visible failure shape).

**Step 5a — assess (0.5 day):**

1. Re-run Adedeji's diagnostic stage and check the `needs_code_enrichment` decision (ensure it triggers for Adedeji).
2. Estimate the cost of CodeFormulaV2 on Adedeji at 320 pages × ~27 sec/page CPU = ~2.4 hours (one-shot).
3. Decide:
   - If `needs_code_enrichment` already says yes for Adedeji and the cost is acceptable, queue it under Phase 4 itself.
   - If the assessment shows CodeFormulaV2 wouldn't fix this specific failure mode (the flat-code chunk on p299 might be Docling losing newlines elsewhere), defer to v2.10.

**Step 5b — execute or defer:**

- **Execute:** convert Adedeji once with CodeFormulaV2 enabled, verify `code_indentation_fidelity ≥ 0.90`, no regression on other gates.
- **Defer:** document the deferral in `docs/PROJECT_STATUS.md` v2.10 backlog with the cost estimate and the chunk-id of the visible failure.

**Done when:** Either fidelity ≥0.90 OR documented deferral with rationale.

**Estimated effort:** 0.5 day assessment + (1 day execute) OR (0.25 day defer).

---

## Step 6 — KI EPUB label/universal failure — sign-off or fix

**Why:** Plan §3 Phase 4 explicitly permits deferral only with user sign-off. This step is the sign-off gate. Investigation has not yet started in Phase 4.

**What:**

1. Re-run strict gate against the latest KI_En_ChatGPT EPUB output. Confirm the failure is the same shape as the strict-gate snapshot recorded.
2. If fix is cheap (a few-hour EPUB-specific extraction tweak), implement under Phase 4.
3. If fix is structural (EPUB extraction lane needs broader work), document deferral and request user sign-off.

**Done when:** Either KI EPUB passes strict gate OR explicit user sign-off recorded in `docs/PROJECT_STATUS.md` and `docs/QUALITY_SNAPSHOT_…_phase4_after.md`.

**Estimated effort:** 0.5 day assessment + (1-2 days fix) OR (0.25 day defer).

---

## Phase 4 Acceptance Gate

Before declaring Phase 4 closed, verify the following in `docs/QUALITY_SNAPSHOT_…_v2.9_phase4_after.md`:

- [ ] Adedeji p301 produces clean chunks (via Phase 1 emitter) OR is quarantined cleanly. `qa_full_conversion.py --source-pdf` reports `TABLE_CORRUPTION=0`, `LOCALIZED_CORRUPTION=0`.
- [ ] Combat p66 produces zero chunks (quarantined). `qa_full_conversion.py --source-pdf` reports `PAGE_QUARANTINED p66`, no `LOCALIZED_CORRUPTION`.
- [ ] Firearms HEADING coverage ≥80% via fix OR documented gate change with explicit user sign-off.
- [ ] Adedeji code_indentation_fidelity ≥0.90 OR documented v2.10 deferral.
- [ ] KI EPUB passes strict gate OR documented user sign-off.
- [ ] No filename-specific production logic added anywhere.
- [ ] No negative test loosened.
- [ ] `pytest tests/ -q` reports the new test count (Phase 4 adds tests; never drops them).
- [ ] `docs/QUALITY_GATES.md` and `CLAUDE.md` updated to specify `qa_full_conversion.py --source-pdf` as the canonical strict-gate command.

## Effort summary

| Step | Effort | Gate to next step |
|---|---|---|
| 1 — `--source-pdf` documentation | 0.5 day | Doc + smoke scripts updated |
| 2 — Adedeji p301 dense-back-index router | 2 days | TABLE_CORRUPTION=0 on Adedeji |
| 3 — Combat p66 quarantine | 1.5 days | LOCALIZED_CORRUPTION=0 on Combat |
| 4 — Firearms HEADING | 1.5 days | Coverage ≥80% OR sign-off |
| 5 — Adedeji code-indent | 0.5–1.25 days | Fidelity ≥0.90 OR defer |
| 6 — KI EPUB | 0.25–2 days | Pass OR sign-off |

**Phase 4 total:** ~6–8 days of focused engineering, parallelizable Steps 2/3/4 (independent code paths).

## Decision log (this plan)

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-09 | `--source-pdf` is the canonical strict-gate command | Diagnostic showed Hao + Adedeji "phantom" MISSING_PAGES failures disappear under the flag; advisory was already in `qa_universal_invariants.py` output. |
| 2026-05-09 | Adedeji p301 path is content-shape detection, not new Docling label dependency | Docling 2.86 does not label this page as `document_index`; relying on the label means accepting future page-level corruption when the label is absent. |
| 2026-05-09 | Combat p66 = quarantine, not repair | Source PDF font corruption is not recoverable by extraction logic. Plan §3 Phase 4 explicitly prefers quarantine over emitting corrupted content. |
| 2026-05-09 | Firearms HEADING route resolved via probe (Step 4a), not pre-decided | Without knowing whether headings are detected-but-not-propagated vs not-detected-at-all, the fix path is unknowable. Probe first, then decide. |
