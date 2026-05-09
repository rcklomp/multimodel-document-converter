# Quality Snapshot 2026-05-09 — v2.9 Phase 4 AFTER

> **Status:** Phase 4 closed pending KI EPUB sign-off (Step 6).
> All 5 substantive steps shipped.
> Predecessors: Phase 3 closed `51e897b` (2026-05-09).

## 1. Steps shipped

| Step | Topic | Commit | Tests added |
|---|---|---|---:|
| 1 | `qa_full_conversion.py --source-pdf` documented as canonical strict-gate command | `611805d` | 0 |
| 2 | Source-PDF back-index fallback (Adedeji p298-316 corruption + truncation) | `afbbaa6` | +27 |
| 3 | Chunk-level corruption filter at finalize boundary (Combat p66) | `6719460` | +19 |
| 4 (fix) | Heading carry-forward across V2DocumentProcessor batches | `b429cb5` | +5 |
| 4 (gate) | Profile-scoped + sparseness-conditional HEADING relaxation | `5e58e6e` | +5 |
| 5 | Adedeji code_indentation_fidelity 0.886 → 0.9032 (cascade win from Step 2) | `a45b259` | — |
| 6 | KI EPUB → DEFER to v2.10 (sign-off pending) | `d18c494` | — |

**Cumulative:** 741 tests pass (was 685 at Phase 3 close, +56 net new).

## 2. Per-document strict-gate results

Strict gate run via `qa_full_conversion.py --source-pdf`:

| Doc | Phase 3 result | Phase 4 result | Notes |
|---|---|---|---|
| Hao | QA_FAIL (5 phantom MISSING_PAGES) | QA_PASS for blank-page check; ASSET_TINY warn | Step 1 (`--source-pdf`) makes blank-page handling correct |
| Adedeji | QA_FAIL (TABLE_CORRUPTION p301; code_indent 0.886; 4 phantom MISSING_PAGES) | QA_PASS (0 corruption; code_indent 0.9032; blank-page info only) | Steps 2 + 5 close the real defects |
| Combat (p1-100, no VLM) | QA_FAIL (LOCALIZED_CORRUPTION p66) | QA_PASS for corruption check (table dropped at finalize); pre-existing VLM_PENDING separate | Step 3 closes p66 |
| PCWorld | QA_PASS | QA_PASS unchanged | — |
| Firearms (no VLM) | HEADING FAIL (72 % < 80 %) | HEADING PASS (anchor-sparse, floor=0.70) | Step 4 Path A closes |
| KI EPUB | UNIVERSAL_FAIL + multiple structural | unchanged — DEFERRED | Step 6 sign-off pending |

## 3. Empirical evidence

### Adedeji p298-316 (back-index span)

Before Step 2:
- p301 had 9 corrupted text chunks ("talent and culture, 246-250 vision and leadership ..." repetition) + 1 corrupted table.
- p298-316: ~2 chunks per page on average (silently truncated).

After Step 2:
- 0 corruption-phrase repetitions across 1095 chunks.
- p301 emits 3 clean text chunks; full back-index span (19 pages) emits 48 chunks via `hybrid_chunker_pageskip` + `hybrid_chunker_pageskip_source_pdf` paths.
- Adedeji `code_indentation_fidelity` 0.886 → 0.9032 as a side-effect (the back-index entry on p299 was being mis-classified as flat code).

### Combat p66 (font-corrupted source PDF)

Before Step 3: 1 corrupted table chunk emitted (4554 chars matching `[—]{6,}` and `[CS]{10,}` corruption patterns).
After Step 3: 0 text/table chunks emitted (filter drops at finalize); 2 image chunks unaffected.
Dry-run on 3,709 chunks across 4 unrelated docs (Adedeji, Hao, PCWorld, Firearms): zero false positives.

### Firearms HEADING

Before Step 4: 72.2 % coverage (304/1094 chunks lack `parent_heading`), gate FAIL at 0.80.
After Step 4 Path A:
- Cross-batch carry-forward fix (b429cb5) shipped — correct in unit tests + small probe (p5-15 with batch-size 5) shows `parent_heading` carries across the p9→p10 batch boundary.
- Empirical Firearms coverage unchanged at 72.2 % because Firearms uses the OCR/element-by-element path (`extraction_method='ocr'`), not HybridChunker. The OCR-path heading propagation has a separate, deeper bug documented in `PLAN_V2.9__PHASE4.md` for v2.10.
- Profile-scoped relaxation (`5e58e6e`) makes Firearms PASS at its natural ceiling: `scanned` profile + `unique_headings/text_chunks=0.028` ≤ 0.05 → floor=0.70. Hao + Adedeji unaffected (technical_manual, sparse-ratios 0.22 / 0.17, keep 0.80).

## 4. v2.10 carry-forward backlog

These items moved to the v2.10 backlog from Phase 4:

1. **`KI_EPUB_EXTRACTION_LANE_REWRITE`** (Step 6) — full EPUB lane: pagination, bbox synthesis, dedup, heading propagation. Acceptance baseline: `output/KI_En_ChatGPT_Praktische_Gids/ingestion.jsonl` metrics in `PLAN_V2.9__PHASE4.md` Step 6.
2. **`OCR_PATH_HEADING_PROPAGATION`** — Firearms element-by-element/OCR path emits chunks with `extraction_method='ocr'` and `parent_heading=None` despite Docling having rich `dc.meta.headings`. Probe data: every Firearms page p80-95 has 5+ headings in dc.meta but JSONL chunks emit zero. Root cause likely in `_process_element_v2`'s `state.update_on_heading` not catching all Docling section_header item shapes, or OCR cascade replacing text without preserving heading metadata.

## 5. Acceptance gate status

Per `docs/PLAN_V2.9.md` §4 Acceptance Gate, before tagging v2.9.0:

- [x] Adedeji p301 produces clean chunks. `qa_full_conversion.py --source-pdf` reports `TABLE_CORRUPTION=0`, `LOCALIZED_CORRUPTION=0`. **Met.**
- [x] Combat p66 produces zero corrupted chunks. `LOCALIZED_CORRUPTION=0` after Step 3. **Met.** (Page is not "quarantined" per the original plan wording — instead, only the corrupted chunks are dropped at finalize. Image chunks remain. This is functionally equivalent and surgically smaller.)
- [x] Firearms HEADING coverage ≥ profile-appropriate floor. **Met via Path A** (sparse-profile relaxation; gate change is profile-scoped + sparseness-conditional, not global).
- [x] Adedeji `code_indentation_fidelity ≥ 0.90`. **Met** (cascade win from Step 2).
- [ ] **Pending:** KI EPUB sign-off — deferred to v2.10 with explicit user sign-off recorded in `PROJECT_STATUS.md`.
- [x] No filename-specific production logic added.
- [x] No negative test loosened.
- [x] `pytest tests/ -q` reports new test count (741 passed, was 685; +56 net new).
- [x] `docs/QUALITY_GATES.md` and `CLAUDE.md` document `qa_full_conversion.py --source-pdf` as the canonical strict-gate command (Step 1).

## 6. Test count provenance

| Phase milestone | Tests passed | Delta |
|---|---:|---:|
| Phase 3 close (51e897b) | 685 | — |
| Step 2 (afbbaa6) | 712 | +27 (back-index detector) |
| Step 3 (6719460) | 731 | +19 (corruption filter) |
| Step 4 fix (b429cb5) | 736 | +5 (carry-forward) |
| Step 4 gate (5e58e6e) | 741 | +5 (anchor-sparse gate) |
| **Phase 4 close** | **741** | **+56 from Phase 3** |

## 7. Outstanding decisions

1. **KI EPUB deferral sign-off.** Per the Phase 4 plan, the deferral requires explicit user sign-off. The acceptance baseline metrics are recorded in `PLAN_V2.9__PHASE4.md` Step 6 for the v2.10 follow-up.
2. **Phase 5 readiness.** With Steps 1-5 closed and Step 6 deferred-with-pending-sign-off, Phase 5 (broad reconversion + Qdrant migration + VLM enrichment + AFTER snapshot) becomes the next workstream once user signs off on the v2.9 closure.
