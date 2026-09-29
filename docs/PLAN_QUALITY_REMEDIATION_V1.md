# PLAN_QUALITY_REMEDIATION_V1 - IRJET-class defects, governance drift, and the open-issue register

Status: RATIFICATION PENDING (rev. 2, 2026-09-29; rev. 1 was restructured after an internal Round 0, Section 13).
Execution of Waves 0-2 follows the owner's 2026-09-29 instruction to remediate all open issues; Wave 3 items are
gated on the owner decisions in Section 9 and are NOT executed until answered. This plan carries no "in-progress"
state (AGENT-STATUS-01): every finding ends as FIX, DECIDE, CLOSE or BLOCKED, each with an owner and, where owner-held,
an answer-by date and a default on expiry.
Base: `499a5fa` (origin/feat/omnidocbench-phase0, 2026-06-18). Work branch: `fix/quality-remediation-v1`.
Companion files: `docs/PLAN_QUALITY_REMEDIATION_V1_REGISTER.md` (one row per finding id, generated),
`docs/PLAN_QUALITY_REMEDIATION_V1_AUDIT_PROMPT.md` (external audit prompt).

## 1. Why this plan exists (evidence, re-checked on 499a5fa)

On 2026-09-29 the owner re-pointed the VLM endpoints to a cloud model (Dashscope-intl `qwen3-vl-flash`,
`scripts/env_cloud_vlm.sh`) because the LAN stack (omlx, M5 Qwen, GX10 MinerU) was down, converted
`IRJET_Modeling_of_Solar_PV_system_under.pdf` (7 pages, `data/academic_journal/`), and received an external verdict:
strong metadata and tables, weak structure, equations, figures, references. This plan treats that verdict as a
hypothesis set and verifies each claim before acting (CLAUDE.md Reliability Protocol).

### 1.1 Verdict claims vs verified reality

| Claim | Status | Receipt |
|---|---|---|
| Flowchart (Fig 7) asset is a 420x43 fragment described as the full flowchart; Fig 3 asset is 67x68 | CONFIRMED, root-caused | B1 geometric crop picks one text-strip raster (Fig 7) and the page-header LOGO (Fig 3): `asset_materializer.py:216-269, 356-375`; the asset/bbox area ratio is 0.08 / 0.07 in the render frame (frame-dependent, Section 6 WP-A1); reproduced on the real PDF |
| Figs 4,5,6,8,9,10 missing as image chunks | CONFIRMED, root-caused | the VLM returned them; B1 gave sprite crops (49x36 .. 80x47 px, <1500 B); `_filter_tiny_icon_images` deleted them (`batch_processor.py:1318-1401`); fresh-run log: "[FINALIZE] Dropped 7 icon-class image chunk(s)". Their descriptions survive as stale `next_text_snippet` (`_apply_lookahead_buffer` runs before the filters) |
| Manifest/asset-disk desync | NOT REPRODUCED | in `output/cloud_probe_irjet` all 4 image + 2 table `asset_ref`s exist 1:1 on disk (6 PNGs); the verdict compared against a different uploaded image set. The defect underneath is the two rows above |
| Running headers/footers inside chunks; abstract / PSO formulation / PWM / conclusion `search_priority: low` | CONFIRMED, root-caused | `low` comes ONLY from a boilerplate regex (>=2 hits of ISBN/ISSN/(c) year) at `batch_processor.py:3056-3065`; the glued header line contains "e-ISSN" and "p-ISSN". F1 furniture filter (`ad3d2ba`) is chunk-level, capped at 70 chars (header is 151), and runs after the glue |
| Headings embedded mid-chunk; wrong `parent_heading` | CONFIRMED, root-caused | `_partition_text_elements` (`uir_chunker.py:1010-1036`) joins all elements between visual elements and stamps the LAST heading on every part; 0 of 9 headings lead their chunk, 12 of 30 parents wrong; corpus median heading-in-body ratio 0.23 over 36 existing outputs (MinerU and VLM alike) |
| Two-column reading order broken (pages 2, 4, 7); references out of order | CONFIRMED, route-specific | the VLM emitted L-top, whole R column, L-bottom; nothing on the legacy route orders by geometry; MinerU order is correct on the same page (older run). Not an engine-agnostic chunker bug |
| Sentence split across chunks | CONFIRMED, two causes | (a) route-specific: emission order puts "-parallel to form..." away from its head; (b) engine-agnostic: `_SENTENCE_ENDS` (`uir_chunker.py:100`) includes `)` `;` `:`, so prose is cut after them on every route ("... (PSO) / algorithm ...", "... with a diode; / in practice ..."); `_merge_mid_sentence_chunks` treats the same set as terminal and cannot rejoin (WP-C3) |
| Reference entries split mid-citation | CONFIRMED, root-caused | `_split_at_sentence_boundaries` slices at the last ". " or ") " before 1400 chars; reproduced the four observed cuts. On the fresh run the right-column references are ONE element with inline labels `... 2011. [10] M. Shan ...` (no newline before the labels), so newline-based splitting alone does not fix it (WP-C2) |
| Equations garbled, LaTeX only once | CONFIRMED, root-caused, with a correction | page prompt has no equation contract and the guided-JSON enum has no formula type (`vlm_native.py:307-374`, `vlm_provider.py:81`). Correction: source equations are a garbled CambriaMath TEXT layer plus some sprites, not pure rasters; equation NUMBERS were lost for 10 of 11 equations |
| Tables are excellent | CONFIRMED for cell content; caption text lost on this route | Table I/II content clean; both captions absent |
| `content_classification`, oversize marker, bbox identical across chunks | CONFIRMED | classification is set only as a side effect of false-code demotion (4/30) and no consumer reads it null-vs-set (WP-B3 KILLs that sub-item); bbox is one group-union copied to every part (fixed by WP-B1); `[Oversize Split n/m]` is appended to `breadcrumb_path`, which feeds embedding text |
| `has_flat_text_corruption=false` is misleading | REFRAMED | the flag detects missing newlines, not reading order; do not extend it |

### 1.2 Meta-findings the verdict could not see

1. **The gates are blind to every defect above.** `qa_full_conversion.py --source-pdf` on the defective output prints
   `QA_PASS: failures=0 warnings=0`; `furniture_chunk_ratio=0.0` while 11 of 30 chunks contain furniture lines. The
   mandatory `smoke_production.sh` "academic" lane IS this document. This is the AGENT-INTEGRITY-01 proxy-vs-outcome failure.
2. **The defects predate the cloud switch.** The June IRJET runs (`vlmtest_IRJET`, `enrichtest_IRJET`, `icontest_IRJET_academic`;
   MinerU-era boxes in the correct frame) show the same 67x68 asset, 7 header-contaminated chunks and 12 low-priority chunks; the
   fresh run on the current tree shows the same asset sizes and missing figures. The 45 upstream commits fixed none of it.
3. **The cloud route adds a bbox regression.** Image bboxes from `qwen3-vl-flash` have IoU 0.00-0.11 with the PDF's true raster
   geometry (June MinerU-era runs: 0.93-0.96); every VLM bbox is shrunk by about 1000/1132 (x) and 1000/1600 (y). Mechanism INFERRED
   (model answers in a 0-1000 grid, adapter divides by pixel size, `vlm_native.py:307-375, 528-566`); page 7 of the same run is
   inconsistent (x scaled, y not), so a single global correction may be wrong. WP-0.4 measures it.
4. **The checkout was 45 commits stale and a third lineage exists.** `github/laptop-travel` (45 commits, July-August) forked before
   the MinerU pivot and is mentioned nowhere. None of the five IRJET defect classes has a fix on it (D-3).
5. **The settled engine cannot run.** With `MINERU_ENDPOINT` unset `mmrag_v3.extract()` selects the legacy `HybridEngine`
   (`processor.py:80-118`). No DECISIONS entry records the retirement or ratifies this route; the Phase 4 rollback, stale-corpus rule
   and FULL smoke all assume the retired stack.
6. **Governance drift is large.** Three read-only audits produced 43 + 41 + 56 findings; with the 83-item backlog the register has
   223 distinct ids (`docs/PLAN_QUALITY_REMEDIATION_V1_REGISTER.md`): 90 FIX, 113 DECIDE, 17 CLOSE, 3 BLOCKED.
7. **Test baseline:** 1767 passed / 99 skipped / 0 failed on 499a5fa; every doc count (1623/100, 1624/99, 1678, 1679) is stale.

## 2. Precedent check (AGENT-PRECEDENT-01)

Read: `docs/paper/FINDINGS_DIGEST.md` (SETTLED / DEAD ENDS / OPEN), DECISIONS "Settled Precedents", FINDINGS_LOG F12/F13.
This plan re-proposes NO measured-and-rejected approach. New ground, with new evidence: IRJET-class academic documents with
vector/composite figures, two-column layout and equation-heavy pages on the legacy route; the route change forced by the
2026-09-29 retirement.

| Precedent | Effect on this plan |
|---|---|
| SETTLED "Conversion is NOT the RAG bottleneck (measured twice)" | Every WP is a FIDELITY / trust / hygiene fix; no WP claims a retrieval lift. Acceptance is source-anchored (Section 8), not a lift promise. Fixes stay small |
| DEAD END "Filtering empty image chunks to help retrieval (+0.0pp)" | WP-A1/A2 RETAIN described figures for the LOCKED multimodal contract (AGENTS.md: IMAGE chunks always retained), not for retrieval gain |
| SETTLED "Engine = MinerU+Qwen hybrid"; DEAD END "engine-swap reflex / more scaffolding around a single general VLM" | No engine change. Route status is decision D-1. VLM-adapter work is decision-gated (WP-K1) |
| SETTLED "Code indentation solved; prompt property (R3 0.947)" | Any prompt edit is additive with a same-model before/after A/B (WP-Q1, decision-gated). (Digest wording disputed: D-17) |
| SETTLED cap1600 render | untouched |
| F12 "spatial boundary-repair bridge REJECTED on VLM-native" | No geometric re-merge, no bbox re-sorter, no merge veto. The furniture pass (WP-B2) removes elements; reading-order repair (WP-O1) is decision-gated and measurement-gated |
| DEAD END "chunker-contiguity fix fires 0x" | Firing rates are measured BEFORE each chunker WP (WP-M0, Section 6); a WP that fires 0x on the corpus is KILLed |
| F13 / AGENT-SPATIAL-20 | no second "20" literal; furniture position uses rank (top/bottom k), not an absolute band |
| Retrieval-Value Test | supports dropping repeating running headers/footers (WP-B2) |
| Charter 7.1 (ElementType frozen at 3) | no new ElementType/Modality/StructuralFlag; equation marker reuses `original_vlm_type` |
| User deferral 2026-06-18 (PROJECT_STATUS): VLM serving/model+prompt work only "once the first production-level release is achieved" | Engine-agnostic chunker/materializer/QA/governance work is not VLM work and proceeds. VLM prompt/adapter items (WP-Q1, WP-O1, WP-K1) are decision-gated on D-4; the frame PROBE (WP-0.4) is measurement only |

## 3. Binding constraints for every WP

1. AGENT-TEST-01: no test/assertion/fixture is removed, weakened or reframed. A design that conflicts with a pinned test STOPS and
   records the proposed requirement change (Section 11.2).
2. AGENT-GATE-PROGRESSION: new quality signals are ADVISORY first with a frozen fixture that fires on bad and stays quiet on good;
   never added to `seeded_fault_sensitivity.gate_quality_signal_vector` (pinned blind). New advisory code must be exception-safe: a
   missing file may never flip `qa_full_conversion` status (it treats "Traceback" / "SEMANTIC_FAIL" in output as an advisory failure).
3. Fix-and-guard, but the GUARD IS NOT THE ACCEPTANCE. A metric computed from the fix's own predicate or from the output alone
   (mirror metric) is a regression guard only. Acceptance is anchored in the SOURCE PDF (Section 8).
4. Red-first: the failing run on the CURRENT code is captured before each fix. A WP whose red test cannot fail today is KILLed.
5. No filename- or document-specific rules; thresholds come from a stated principle or calibration on local outputs, never from the
   failing document (memory: contract-violation-mode); provisional thresholds are set only after calibration is recorded.
6. AGENT-STATUS-01: no "watch item". A regression found after a commit is reverted, not watched.
7. AGENT-EVIDENCE-01: ignored `data/` and `output/` are never the sole evidence. Registers and reports of this cycle that a reader
   needs live in committed files (the register file, FINDINGS_LOG entry, tests/fixtures).
8. ASCII punctuation only. `docs/.archive/` and `tests/.archive_*` are never read.
9. Paid calls are limited to: WP-0.4 (frame probe: 3 pages x 2 born-digital, non-personal documents) and WP-V1 (one 7-page IRJET
   run, one rerun only if the validity preconditions fail). No batch/corpus conversion. Personal/financial documents
   (`data/business_form/`) are never sent to a cloud VLM by this plan (D-20).
10. Commit locally, atomic per WP, with the required trailer; NO push; the public `github` remote is never touched.
11. Owner decisions: unanswered decisions take their stated default on the answer-by date, EXCEPT that a default with an external or
    irreversible effect is always "do nothing" (Section 9). Defaults are recorded in DECISIONS as "applied by default, revocable".
12. Wave 1 edit policy (documents): only (a) facts verified by a command at edit time, (b) dated `STATUS 2026-09-29: ... see D-n`
    markers at a conflict, (c) new dated DECISIONS entries that record FACTS, (d) moves of existing text that never summarize an
    owner directive. Never replacement text for MUST-level, SETTLED, DEAD-END or Layer-0 policy content.

## 4. The two load-bearing artifacts

- **Disposition register** (`docs/PLAN_QUALITY_REMEDIATION_V1_REGISTER.md`, generated): one row per id with a disposition from the closed vocabulary.
  Two checks. (1) Integrity in a clean clone: `python3 scripts/check_plan_register.py` (committed; exits non-zero on a duplicate id, a disposition outside the
  closed vocabulary, a DECIDE/BLOCKED row without owner and answer-by date, a decision or work-package reference that is absent from this plan, or a wrong counts
  line). (2) Completeness against the sources: the generator (`gen_register.py`) parsed the four audit registers (local, gitignored evidence in
  `output/audit_2026-09-29/`), aborted on any unmapped or duplicate id and on a wrong per-source count (83 INV, 43 G0, 41 DC, 56 AP), and its output was
  committed; regenerate only from those local files.
- **Source-anchored acceptance harness** (WP-0.2, WP-0.3): the instrument that lets a "fixed" claim be falsified.

## 5. Baseline (captured 2026-09-29 before any change)

Suite: `pytest tests/ -q` = 1767 passed / 99 skipped / 0 failed (56 s). Strict gate on the cloud output: `QA_PASS failures=0 warnings=0`.
Ad-hoc structural numbers (informational; WP-0.2 replaces them with committed metrics computed by the definitions in Section 6 and
records the acceptance baseline, with the commit hash, in FINDINGS_LOG BEFORE any Wave 2 fix lands):

| Measure | `cloud_probe_irjet` (user run, stale tree) | `irjet_baseline_499a5fa` (fresh) | June MinerU-era run |
|---|---|---|---|
| chunks (text/image/table) | 36 (30/4/2) | 34 | 36 |
| image/table asset area vs bbox area (render frame; frame-dependent) | 2 of 6 at 0.07-0.08 | 2 of 6 at 0.08 | Fig 3 same 67x68 asset |
| figures referenced in text / image chunks | captions 1,2,3,4,6,8,9,10; 4 image chunks | 4 image chunks; 7 sprite figures dropped (log) | 5 kept of 15 |
| chunks containing running header/footer text | 13 | 12 | 7 |
| TEXT chunks `search_priority: low` (IMAGE chunks are low by default and rise when figures are retained) | 7 | 7 | n/m |
| chunks with a known heading line embedded after line 1 | 8 | 7 | n/m |
| header `pipeline_version` / `config_hash` | "2.7.0" / null | "2.7.0" / null | "2.7.0" / null |
| reference labels | [10..17, 1..9], entries cut | [10..17 as ONE inline element, 1..9], 3 entries cut | 1..17 in order |
| QA-CHECK-01 variance | -23.0% (academic allowance 25%) | -28% (own simulation) | n/m |

## 6. Work packages

Risk classes: H = hygiene/doc (no behavior change); B = behavior-changing; D = decision-gated; M = measurement only.

### Wave 0 - instrument and measure (no behavior change except WP-0.5)

**WP-M0 firing-rate measurement (M).** Before any chunker/materializer WP is written, measure read-only on the local outputs
(`output/*/ingestion.jsonl`, 46 runs) and by reconstructed element streams: how often each planned change would fire (heading sections, entry-aware
split, sentence-end ranking, furniture removal by label and by repetition, crop predicate changes). Record the table in FINDINGS_LOG. A WP with
firing rate 0 is KILLed (chunker-contiguity precedent).

**WP-0.2 regression-guard metrics (advisory).** Pure functions in `src/mmrag_v2/validators/structural_outcomes.py`, printed as ADVISORY lines by
`scripts/qa_semantic_fidelity.py` (exit code unchanged, exception-safe, not added to the seeded-fault vector):
- `orphan_snippet_chunks`: a chunk whose non-empty `next_text_snippet` differs from the actual successor's prefix, PLUS `snippet_coverage_gap`: a non-last chunk whose successor has content but whose snippet is null (so nulling every snippet cannot score 0).
- `heading_inside_body`: TEXT chunks where a line other than the first equals a heading string used as some chunk's `parent_heading` (a MIRROR of WP-B1's rule: regression guard only).
- `furniture_line_chunks`: TEXT chunks containing a line whose digit-normalized form repeats on >= 3 distinct pages (output-derived upper bound; guard only).
- `reference_entry_integrity`: inside a references-class section (parent heading matches references/bibliography/literature/works-cited, or >= 3 `[n]` labels), label runs are non-decreasing in chunk order, labels are found both at line starts and inline after a sentence end (`(?<=[.]) \[\d+\] [A-Z]`; body citations like "Reference [10] presented" are not counted), and no entry is split across chunks.
- `figure_deficit_pages`: pages where distinct standalone caption lines (`Fig.`/`Figure` N, line length <= 120, not a sentence with a verb after the number) exceed IMAGE chunks on the page.
- INV-062: frozen fire-on-bad / quiet-on-good fixtures for `furniture_chunk_ratio`, `cross_page_dupe_ratio`, `code_fence_consistency` (they have none).
- **Fix-induced-fault matrix** (`tests/test_structural_outcomes.py`): seeded faults that a fix could introduce (all snippets nulled, all parents nulled, an over-deleted repeated body line, a dropped reference entry, a displaced crop) with the recorded verdict of which metric moves; metrics that cannot see a fault are listed as KNOWN-BLIND in the test (never re-labelled).
The crop-size metric is NOT defined here: asset-vs-bbox area is frame-dependent (MinerU 200-dpi frames, cap1600 frames and PDF-point frames give median ratios 0.50 / 0.52 / 1.48 / 4.03 for healthy crops), so it is computed only from the frame-invariant crop-audit sidecar of WP-A1.
Calibration: run over every local output, record distributions and false-positive rates in FINDINGS_LOG, then set thresholds. Rollback: one commit. Effort M.

**WP-0.3 source-anchored acceptance spec.** `tests/fixtures/gold_anchor_specs/irjet.json` + `scripts/qa_gold_anchor_smoke.py`. Anchors are derived from the SOURCE PDF text, not from any run:
(1) the 9 section headings in source order and, for each of ~6 anchor phrases (e.g. "20kHz", "No. of particles", "Particle swarm optimization was originally"), the regex of the section it belongs to; (2) the source's own running header/footer strings (present on every page of the text layer): zero chunks may contain them (anchor `furniture_absent`); (3) reference entries by author (Mandour, Miyatake, Tsai) that must be intact in ONE chunk including their terminal tokens; (4) table anchors (Table I 8 rows, Table II 10 rows, Greek/subscript characters preserved); (5) after Wave 3 only: equation-number adjacency.
Anchors whose pass depends on the engine labelling a heading element (e.g. "IV. SIMULATION" was labelled in one run and not in the other) are marked `engine_labelled`: evaluated only when the heading element exists in the UIR, else recorded N/A with the reason (never counted as pass).
Baseline: the script is run on BOTH existing outputs BEFORE any fix; the anchors that fail today are recorded in Section 5 / FINDINGS_LOG (this proves discrimination). Effort S.

**WP-0.4 bbox-frame probe (M, paid, about 6 page calls).** For 3 pages of IRJET and 3 pages of one non-personal born-digital document, request bboxes from the cloud model with the current prompt, compare against PyMuPDF text-block geometry per page, and record per page whether the answer is in the pixel frame or a 0-1000 grid (and whether pages within a document disagree). Output: a table in FINDINGS_LOG. This decides whether WP-K1 is meaningful and whether figure-crop acceptance on the cloud route is claimable (Section 8.1). No code change.
**Result (executed 2026-09-29, 6 calls, `qwen3-vl-flash` via Dashscope, production prompt and call path):** on 5 of 6 pages the model's raw bbox extents match a 0-1000 grid, not the pixel frame the prompt asks for. IRJET p2/p3/p7 (render 1132x1600): raw-max / native-text-max = x 0.866 / 0.838 / 0.874 (grid expectation 0.883), y 0.617 / 0.615 / 0.613 (0.625); a second born-digital document (Schwungradspeicher, render 1120x1600) p2: 0.892 / 0.627 (0.893 / 0.625), p4: 0.887 / 0.624; its p3 is inconclusive on y (raw boxes include figure regions with no text block; x 0.879). Pixel-frame answers would read 1.0 on both axes. The page-7 inconsistency seen in the user's run did not reproduce (p7 here is grid on both axes). Conclusion: the cloud route shrinks every VLM bbox by about 0.88 (x) and 0.625 (y); WP-K1 is warranted on this model and is now only waiting on D-4. Raw numbers: `output/audit_2026-09-29/wp04_frame_probe_results.json` (local) and the FINDINGS_LOG entry.

**WP-0.5 UIR dump (B, behavior-neutral).** `MMRAG_DUMP_UIR=<dir>` makes `batch_processor` write each batch's `UniversalDocument` as JSON (loader in `mmrag_v2.universal`), so chunker changes are A/B-tested on a FIXED extraction (two live IRJET extractions are 84% identical, F12). Tests: dump/load round-trip on a synthetic document; env unset = byte-identical output. Effort S-M.

### Wave 1 - governance corrections (class H, documents only; edit policy = Section 3 rule 12)

Commit 0: track `scripts/env_cloud_vlm.sh` (D-12 default applied now: it is the only configuration of the operative route, it holds no secret - verified by grep - and Wave 1 documents cite it). Revocable by reverting one commit.
Execution order: WP-G2 first entry records the 2026-06-18 VLM deferral VERBATIM (from PROJECT_STATUS.md lines 20-27) as an owner-attributed DECISIONS entry, then WP-G3 restructures PROJECT_STATUS (moving text, never summarizing an owner directive).
- **WP-G1 Layer-0:** CLAUDE.md, AGENTS.md, docs/README.md, V3_EXECUTION_MANDATE.md, TESTING.md, QUALITY_GATES.md, `.clinerules`.
- **WP-G2 DECISIONS.md:** markers for superseded/lapsed entries; new dated FACT entries: (a) the retirement and the observed behavior of the running route (unmeasured), (b) the verbatim 2026-06-18 deferral, (c) shipped-but-unrecorded behaviors (hard per-page deadline, code-repair pass, code-furniture stripper, endpoint registry, hard EXTRACTION_DEGRADED_CODE verdict, retrieval default change), each "pending ratification D-22".
- **WP-G3 status/architecture/plans/digest/log:** PROJECT_STATUS.md to ONE current state (banner, test counts 1767/99, endpoints "as of 2026-09-29", false "not pushed" claims), Charter Sec. 9.1 / 4.3 / 5 / header corrections, ARCHITECTURE.md historical label, plan status lines per AGENT-STATUS-01, FINDINGS_DIGEST marker lines only (resolved OPEN items with commit refs; the G7 structure is untouched), a dated FINDINGS_LOG entry.
- **WP-G4 memory hygiene** (outside the repo): stale entries removed/rewritten; entries about the owner's preferences are NOT rewritten without the owner (only marked stale in the index).
Verification per commit: (a) every edited claim re-checked by a command; (b) an independent adversarial reviewer agent diffs the edits against code and tries to find a false or newly drifting sentence; (c) the integrity suite (G1-G7) is run BOTH in the working tree and in a clean `git worktree` of the commit (G5 checks the working tree, not the committed tree, so untracked referents would hide a dangling path). Effort M-L (mechanical).

### Wave 2a - engine-agnostic fixes with measured designs (class B)

Order avoids same-file conflicts: `uir_chunker.py`: B1 -> C2 -> (C3); `batch_processor.py`: A2 -> C4 -> B3 -> D7 consumers untouched; `processor.py`: none in this wave.

**WP-B1 heading sections** (`uir_chunker.py:987-1036`). Symptom: 0/9 headings lead a chunk; 12/30 wrong parents. Cause: group-last heading stamped on every part; headings never a boundary. Change: a heading-labelled element closes the current section and starts a new one; the section's `parent_heading` is that heading; elements before the first heading get `parent_heading=None` (carry/TOC fills them); per-section bbox instead of the group union; the splitter applies inside an over-budget section and every part inherits the SECTION heading. ~25 lines.
Tests (red-first, seeded generator >= 20 seeds; prototype fails 39/40 today): P1 no chunk contains a heading text that is not its first line; P2 `parent_heading` equals the nearest preceding heading element at the chunk's first element; P3 element text/order preserved; P5 per-section bbox. Pins that stay green: `test_chunk_universal_document_contract.py` (7), `test_ocr_path_heading_propagation.py` (prototype: full suite 1767/99 green).
Exit criterion (offline replay): reconstruct element streams from the local outputs (AIOS is borderline on the hard HEADING gate) and confirm no document that was >= 0.80 HEADING coverage falls below it; a drop is explained or the design is amended, never the gate.
Consumers to list in the DECISIONS entry: `_assign_headings`, F1 band check, `_merge_mid_sentence_chunks` (keeps `cur` metadata), ingest `resolve_search_priority`/`_has_references_heading_context` (priority changes for re-attributed chunks), `rag/advanced_pipeline.py` breadcrumb-depth boost, identity gate. Chunk boundaries/ids change (D-8). Effort S.

**WP-C2 reference entries** (`uir_chunker.py:1044-1101`). In sections that are references-class (Section 6 WP-0.2 definition), rank split candidates: a newline or blank line before an entry label, then a label that follows a sentence end inline (`(?<=[.]) (?=\[\d+\]\s+[A-Z])`, prototyped: Mandour entry intact, no body-citation cut, full suite 1767/99 green), then the existing rule. Outside references-class sections behavior is unchanged (blast radius = reference lists; WP-M0 records how many sections it touches). Tests: (i) newline-separated entries (15/17 intact today), (ii) INLINE-entries fixture from the fresh baseline shape (red today), (iii) body citation "Reference [10] presented" not cut, (iv) a numbered manual list unaffected. Effort S.

**WP-C3 sentence-end ranking (measure-first).** `_SENTENCE_ENDS` includes `)` `;` `:`; prose is cut after them on every route and `_merge_mid_sentence_chunks` cannot rejoin. Candidate: rank ". " / "! " / "? " above the weaker enders and require a non-lowercase next character after a weaker ender. WP-M0 measures how many chunk boundaries on the 46 outputs would change; the change alters chunk shape for essentially every document, so it lands only with D-8 answered "accept" and only if the firing rate justifies it (otherwise KILL). Effort S.

**WP-A2 stale snippets + drop manifest** (`batch_processor.py`). Recompute `next_text_snippet` with the SAME rule as `_apply_lookahead_buffer` (successor `content[:300]`) on the FINAL chunk list, i.e. AFTER every in-loop drop (pHash duplicate rejection at ~2972-3035 and the asset-metadata-mismatch skip at ~2958-2967 happen inside the write loop; the logo orphans in the cloud run are exactly this): buffer the export list, fix snippets, then write, or hoist those two decisions into a pre-pass (chosen at implementation; the red test covers both). `prev_text_snippet` stays as today (null on V3): no backfill, so ingest contextual text changes only where a snippet was stale. Drop manifest: itemized reason -> count + asset names as a WARNING summary (no header change) covering ALL IMAGE-dropping sites: blank, tiny-icon, thin-strip, pHash reject, no-visual, full-page editorial, asset-less drop, `_apply_full_page_guard` DISCARD, asset-metadata-mismatch skip, chunk_id dedup. Invariant test: IMAGE elements in = IMAGE chunks out + itemized drops.
Tests: [text, image(sprite), text] and [text, image(logo duplicate of an earlier page), text] -> refresh; every surviving snippet is None or equals its actual successor's prefix; coverage: no non-last chunk lost its snippet (fails today with 6 orphans on the real output). Guard: `orphan_snippet_chunks` + `snippet_coverage_gap`. Effort S-M.

**WP-C4 provenance stamps.** `pipeline_version := __engine_version__` (`batch_processor.py:2923`; the two other header writers `processor.py:4364, 4438` also stamp `pipeline_version` + `source_file_hash`); doc-level header field `extraction_vlm_model` (from the served VLM engine's config; replaces the per-chunk model attribution that is not recoverable on the hybrid route, where MinerU and Qwen elements both carry `ExtractionMethod.VLM`; `vision_provider_used` stays the CLI flag and is documented as such); `config_hash` = sha256 of canonical JSON of output-affecting options (engine + schema version, profile, route, VLM model id, render cap, batch size, chunk max_chars, furniture flag; never keys/endpoints; exact set D-11); `qa_conversion_audit.py:339` compares `pipeline_version` to `__engine_version__` via a SEPARATE constant from the schema-version check. Corpus manifest rows gain a `config_hash` passthrough; the staleness RULE stays route-based until D-1 (this WP makes staleness RECORDABLE, not computed). No schema bump (header VALUE + optional fields; precedent `extraction_*`).
Consumers named: `scripts/build_corpus_manifest.py:171`, `scripts/manifest_status.py`, `scripts/v3_identity_gate.py:54-70` (metadata-only), fixtures with "2.7.0" (`test_code_indentation_audit_gate.py`, `test_tabular_audit_gate.py`: PROVENANCE is not a verdict fail, prototype-verified green). Tests are CI-enforced: a synthetic-PDF integration test (`fitz`-built) in a new module, not only the `skipif`-gated `test_v3_integration.py`. Effort S + M.

**WP-B3 metadata hygiene.** Only: drop the `[Oversize Split n/m]` / `[Split i/n]` breadcrumb markers and the level bump (keep `_oN` ids; REQ-HIER-04 holds; `breadcrumb_path` feeds `to_embedding_text` and the advanced-pipeline depth boost). KILLED sub-items (rule 4 / no consumer): `content_classification` for TEXT chunks (no gate keys on null-vs-set; every consumer tests `== "code"`; a substring heuristic would fill an unread field; `test_null_fixes.py` pins only oversize sub-chunks) and the `docling-2.86.0` literal (never reaches the output). Effort S.

**WP-D7 ingest embedding text.** `scripts/ingest_to_qdrant.py:585-588` embeds the 400-char truncated `visual_description` mirror instead of the full `content` (the contractual cap is correct; the docstring claim "loses nothing retrievable" is false for this consumer). Refactor commit first (extract `_image_embedding_text`, pin the source-grep contract of `tests/test_ingest_content_preference.py`), then the red test and the fix (prefer `content` when the mirror ends with "..." and `content` starts with the mirror prefix). Other consumers of the mirror (Qdrant payload, search display, `rag/advanced_pipeline.py`) are unchanged and listed in the DECISIONS entry. No retrieval-lift claim. Effort S.

**WP-F1 Docling bbox origin** (`docling_fast.py:108-141`). `_normalize_bbox` ignores `coord_origin` (BOTTOMLEFT); IRJET p2 header logo comes out at y 907-949 (true 55-91). Use docling_core's `to_top_left_origin` (prototype: [57,847,112,890] -> [57,109,112,152]); test with a BOTTOMLEFT fixture. Identical in effect to `laptop-travel a7b30f5` (dropped from the D-3 cherry-pick set). Affects every `docling_fast` bbox, so every offline smoke lane changes crops (re-run in the ship gate). Effort S.

**WP-F2 validate_qdrant exit code** (`scripts/validate_qdrant.py:274-283`): a per-collection exception is printed and `main()` still returns 0; return non-zero. Effort S.

### Wave 2b - measure-first designs (class B; a design is chosen by measurement or the WP is KILLed)

**WP-A1 crop plausibility (design by measurement).** Symptom: B1 picks a text-strip raster (Fig 7) or the header logo via the no-overlap "largest remaining" fallback (Fig 3). A one-sided area floor (`candidate >= 0.5 x VLM clip`) is REJECTED as the design: it regresses correct rescues under oversized boxes (12 local crops, e.g. HarryPotter p2 VLM bbox [0,0,1000,865] with a correct 908x666 raster, ratio 0.30) and on the cloud frame turns wrong-OBJECT crops into wrong-REGION crops (Fig 3 -> body prose, Fig 7 -> page header) while every area-based metric reads success.
Step 1 - sidecar: persist the crop-audit report (today discarded at `batch_processor.py:1874`) as `crop_audit.json` next to the JSONL: per IMAGE/TABLE chunk `page`, `vlm_rect_pt`, `clip_rect_pt`, `crop_source`, `asset_px`. This makes size ratios frame-invariant (points).
Step 2 - offline design study on the 673 local crops + the real IRJET PDF with both bbox variants (correct frame, cloud frame): candidate predicates (a) area floor, (b) DOMINANCE: a geometric object replaces the VLM crop only if it is the dominant graphics object (area >= 0.5 x the union of rasters and drawing rects intersecting the clip) and, on the no-overlap fallback, only if dominant among all non-chrome graphics on the page, (c) exclusion of page-chrome objects (same xref at the same bbox on >= 3 pages). Record for each: crops changed, the changed list, IRJET Fig 3/7 outcome, the 12 oversized-box cases, the pinned tests (`test_b1_rescues_garbage_vlm_bbox_via_geometric_object` ratio 1.9, `test_b1_distributes_two_chunks_across_two_objects`, `test_geometric_crop_is_never_reextracted`, tiny-icon/thin-strip tests). Choose the predicate that fixes IRJET, regresses none of the oversized rescues and keeps every pin; if none does, KILL A1 and record the measured reasons.
Step 3 - implement + red tests (composite figure of small label rasters + header logo; pure-vector figure + logo; oversized garbage box with a real raster stays rescued).
Outcome check independent of the predicate (`scripts/qa_crop_fidelity.py --source-pdf ... --crop-audit crop_audit.json`): `crop_prose_fraction` = share of the PDF text-layer words inside the clip that belong to lines of >= 8 words; a figure crop dominated by body prose is the wrong region.
Sequencing: figure-crop acceptance on the cloud route requires WP-0.4's answer; if the frame is a 0-1000 grid and WP-K1 is not permitted (D-4), figure-crop acceptance on that route is NOT claimable and Section 8.1 says so. Stage-2 (drawing clusters as extra candidates) stays KILLed (Section 10). Effort M.

**WP-B2 furniture at element level (measure-first, label-aware).** Cause: F1 is chunk-level, capped at 70 chars, runs last; furniture is glued into body chunks before it runs.
Design: (1) PRIMARY signal = the engine's own furniture labels (`source_label` in header/footer/page_number/page_header/page_footer; the VLM emits `type:"footer"`), (2) FALLBACK = repetition: digit-normalized, whitespace-collapsed signature (difflib ratio >= 0.9) on >= 3 distinct pages at the same rank position (top/bottom k elements per page; not an absolute band, because VLM frames are compressed), (3) NEVER remove heading-labelled elements or elements equal to a TOC/in-page heading string (HarryPotter's chapter titles ARE running headers: 42 of 48 of its TEXT chunks take their parent heading from a line repeated on >= 3 pages, DECISIONS cluster B 62% -> 98%), (4) F1's single-occurrence masthead/URL rule keeps its band AND its 70-char cap, (5) drops are RETURNED through an optional out-parameter of `chunk_universal_document` (its entry-point contract and 9 test modules stay valid) and registered by `batch_processor` with the `QualityFilterTracker` as `FilterCategory.NOISE_PATTERN`, so QA-CHECK-01 subtracts intentionally removed tokens (without this, IRJET goes from -23% to about -31% against the 25% academic allowance and becomes invalid), plus an exact-set check (removed lines == the furniture signatures). F1 chunk-level filter stays as backstop.
NOT in this WP (measured, KILLed for this cycle): the `search_priority` boilerplate "hardening" to >= 50% of lines (it flips 36 of 55 currently-demoted chunks to high, including the genuine copyright pages HarryPotter p8 and Kimothi p4; the IRJET cause is removed by (1)-(3): low TEXT chunks 7 -> 1 in simulation) and the `_merge_mid_sentence_chunks` veto (fires 0x once headers are removed; a veto on absolute bands cannot fire on the compressed cloud frame; F12 precedent). Known limitation: a header fused into a body element (page 7 "... www.irjet.net conditions.") survives.
Measure-first (WP-M0): elements removed by label vs by repetition on the local outputs, and HEADING coverage before/after on HarryPotter/CombatAircraft/AIOS by element-stream replay (a needed check that costs no route or credit). Exit criterion: no HEADING coverage decrease; otherwise the design is amended, not the gate.
Tests: synthetic 5-page UIR with a wrapped running header (positive); heading repeated on 2 pages, non-repeating short heading, single margin citation, only-element-on-page, TABLE header rows (negatives); HarryPotter-shaped fixture (chapter-opening title + the same string as a running header on >= 3 later pages: parent-heading coverage unchanged); idempotence and token-subset properties; QA-CHECK-01 tracker registration test. Guards: `furniture_line_chunks` + source-anchored `furniture_absent`. Effort M.

### Wave 3 - decision-gated (class D; not executed until the decision is answered or its default applies)

- **WP-R1 route ratification** (D-1, D-2): DECISIONS entry with an acceptance bar INSIDE the decision (a measured baseline on the ratified route, not an assertion), rewritten Phase 4 rollback/stale rules, an instrumented rollback counter.
- **WP-Q1 equations** (D-9, D-4): additive prompt rule 9 (`type:"formula"`, LaTeX in `\[ \]`, keep `\tag{n}`, never `type:"code"` for math, no LaTeX in table cells, table captions as separate text elements), enum entry `vlm_provider.py:81`, MinerU marker parity (`original_vlm_type="formula"`), a `keeps_formulas` swap guard in `_repair_degraded_code`. Acceptance: SAME-MODEL before/after prompt A/B (the settled R3 0.947 was measured on Qwen3-VL-8B/mlx and is not a comparator for `qwen3-vl-flash`); fixed-UIR A/B for chunker effects; element-atomic packing only if a fixed-UIR fixture set (not one live run) shows cuts inside display equations (the replay reproduces cuts at 19 of 50 pad values).
- **WP-O1 reading order** (D-10): advisory column-interleave detector first; a reorder pass (no merge/split) only after a firing-rate measurement and a deterministic reading-order A/B; KILLED outright if D-1 restores MinerU.
- **WP-K1 bbox frame** (D-4, D-12): per-page frame detection (not a static global knob: page 7 of the same run mixes frames): rescale only pages whose text-bbox union vs the PyMuPDF text-layer union ratio matches 1000/pixel_dim within 3%; guard `bbox_frame_suspect` needs a source PDF (the host script has none: implemented in `qa_crop_fidelity.py`). Acceptance: WP-0.4's table.
- **WP-L1 lineage** (D-3): cherry-pick set from `github/laptop-travel`, archive the rest.
- **WP-P1 policy conflicts** (D-5, D-6, D-7, D-13, D-15, D-16, D-17, D-21, D-22, D-23): each answered decision becomes a DECISIONS entry plus the matching doc edit.

### Wave 4 - acceptance (class H; evidence only)

- **WP-V1 live IRJET run with validity preconditions:** `extraction_engine == "hybrid"`, `extraction_degraded_pages == 0`, `extraction_fallback` null, VLM model id recorded, API key present (`env_cloud_vlm.sh` continues after printing an ERROR when the key is absent, which would silently serve the run from the Docling tier); invalid -> one rerun, then FAILED. `MMRAG_DUMP_UIR` writes the fixed UIR. Chunker-owned targets are evaluated on the FIXED UIR through the final chunker and through the base-commit chunker (deterministic A/B); only extraction-dependent items are read from the live run, with the 84% run-to-run identity noted.
- **WP-V2 human acceptance artifact set:** rendered source pages 2, 3, 5, 6 next to their crops and chunks for ONE bounded owner review at the phase boundary (memory: human-acceptance-artifacts). Not a recurring loop and not a blocker for automated gates.
- **WP-V3 ship gate** (Section 8.2).
- **WP-V4 source-anchored anchors** (WP-0.3 spec) on the new output, compared with the recorded baseline.
- **WP-V5a offline before/after regression (free):** the pinned offline recipe (Section 8.2 item 5) run BEFORE (base commit in a clean `git worktree`, `PYTHONPATH` at that worktree's `src`) and AFTER the branch over the same PDF list; the 6 regression-guard metrics plus the hard gates per document; explained-delta review per Mandate Sec. 3.
- **WP-V5b live multi-document regression:** BLOCKED on D-20 (cost, data policy); the limitation is stated in Section 8.1.

## 7. Disposition register

`docs/PLAN_QUALITY_REMEDIATION_V1_REGISTER.md` holds one row per finding id (INV-, G0-, DC-, AP-): severity, source class, title, disposition (FIX / DECIDE / CLOSE / BLOCKED), reference (WP or D id), owner, answer-by date and note. It is generated by `gen_register.py` and verified for id-set equality (Section 4). Counts at generation: 223 ids = 90 FIX, 113 DECIDE, 17 CLOSE, 3 BLOCKED.
Mapping principles (each was a Round 0 finding; H-1, H-2, M-8): a source USER-DECISION is never executed as a FIX; a SETTLED / DEAD-END / Settled-Precedents / AGENTS-contract edit is always a DECIDE with a dated marker at most; every DECIDE/BLOCKED row has an owner, the answer-by date 2026-10-20 and the default in Section 9; there is no DEFER state; code-only items are FIX or CLOSE(KILL), never BLOCKED on an unrelated decision.
CLOSE reasons in the register are self-contained (KILL/SETTLED/DEAD-END/MOOT with the fact that justifies them). The three BLOCKED rows are INV-030, INV-033 (need a trustworthy code-fidelity measure / a reachable VLM; default on expiry: CLOSE as accepted limitation) and INV-039 (armed: build the carry cap in the same change that drops `--no-contextual`; default: CLOSE).

## 8. Acceptance and ship gate

### 8.1 Source-anchored outcome acceptance (measured in WP-V1/V4/V5a; thresholds of guard metrics only after WP-0.2 calibration)
- Figure crops: `crop_prose_fraction` < 0.5 on every IRJET IMAGE chunk (majority-prose crop = wrong region), and every figure the source shows is retained OR itemized in the drop manifest with a reason. CLAIMABLE only under a correct bbox frame (WP-0.4 says pixel, or WP-K1 is permitted and confirmed); otherwise this line reads "not claimable on the cloud route" and the owner review (WP-V2) is the only evidence.
- Headings: every source-anchored section anchor (WP-0.3) that is not `engine_labelled`-N/A has the expected `parent_heading`; 100% of the evaluable anchors.
- Furniture: the source's own header/footer strings appear in 0 chunks (documented exception: fused first-sentence furniture); TEXT chunks with `search_priority: low` on IRJET are only copyright-shaped chunks.
- References: the sampled reference anchors are each intact in ONE chunk.
- Snippets: `orphan_snippet_chunks` = 0 and `snippet_coverage_gap` = 0 on the final export.
- Provenance: header `pipeline_version == __engine_version__`, `config_hash` non-null, `extraction_vlm_model` present; a test that changes only the render cap or the model id changes the hash.
- No regression: Table I 8x2 and Table II 10x2 with Greek/subscript characters preserved; QA-CHECK-01 reported as the ABSOLUTE post-fix variance against both the documented 0.10 and the code allowance (D-7 is open: this plan chooses neither side); existing hard gates unchanged.
- Guard metrics (mirrors) are reported next to the anchors but are never the acceptance.
- If a target is missed the WP that owns it has FAILED (Mandate Sec. 4): it is fixed or reverted, never re-labelled. There is no "or explained by a route limit" escape: a route limit is stated as "not claimable" up front, above.
- Scope of the claim: live evidence is one 7-page document; regression evidence beyond IRJET is offline-only (WP-V5a) until WP-V5b (D-20) is answered.

### 8.2 Ship gate (enumerated against the governance invariants)
1. `pytest tests/ -q` exit 0 with the baseline 1767 passed / 99 skipped plus the new tests; no new unregistered skip (G6); integrity G1-G7 green in the working tree AND in a clean worktree of the final commit.
2. `pytest tests/test_v3_security.py` (AST firewall) exit 0; `batch_processor.py` gains no Docling import.
3. `bash scripts/smoke_production.sh` prints `SMOKE_PRODUCTION_PASS` (offline; the pinned offline recipe below).
4. `qa_full_conversion.py --source-pdf` on the new IRJET output: QA_PASS or documented advisories; `qa_universal_invariants.py` UNIVERSAL_PASS.
5. AGENT-VAL-01 (`bash scripts/smoke_multiprofile.sh`): run on the PINNED OFFLINE ROUTE only - `env -u OPENROUTER_API_KEY -u DASHSCOPE_API_KEY -u VLM_NATIVE_API_KEY -u VLM_NATIVE_ENDPOINT -u MINERU_ENDPOINT USE_DOCLING_FAST=1` (the script has no route pin: with a key in the shell it would send up to 11 documents, including business forms and a scanned invoice, to a paid cloud VLM, and `extract()` now runs a VLM code-repair pass even under `USE_DOCLING_FAST`) - BEFORE (base commit, clean worktree) and AFTER. Pass criterion: every row GATE_PASS + UNIVERSAL_PASS, or each failing row is identical before and after (explained delta, Mandate Sec. 3). The script's own defects (exit 0 with failing rows) are recorded under D-6 and do not soften this criterion. `acceptance_technical_manual.sh` is run the same way or its omission is left to D-6.
6. Workstream B negative tests and seeded-fault blindness tests unchanged and green.
7. Mandate Sec. 2 ADVISORY criterion (OmniDocBench fidelity delta, `scripts/omnidocbench_adapter.py`): NOT run - no route with a recorded baseline is reachable and B1/B2 move text edit distance; the omission is stated here as an advisory waiver, not hidden.
8. Evidence is committed: tests/fixtures + commit hashes + the FINDINGS_LOG entry + the acceptance baseline recorded by WP-0.2 (with its commit hash) and the final numbers.

## 9. Owner decisions (owner: project owner; answer-by 2026-10-20; default applies on expiry and is recorded "applied by default, revocable"; defaults with external or irreversible effect are always "do nothing")

| Id | Decision | Recommendation | Default on expiry |
|---|---|---|---|
| D-1 | Production route while MinerU is unreachable: (a) ratify legacy HybridEngine + cloud VLM as an explicit interim route, (b) restore MinerU (re-home the server) and treat cloud as emergency only, (c) both | (c), but the interim ratification carries a MEASURED bar (a baseline on the ratified route: OmniDocBench delta or the FULL smoke that can authenticate to a keyed endpoint) and an expiry trigger "MinerU re-homed" | do nothing: docs state the route as "operative but unratified" |
| D-2 | LAN stack permanently retired or temporarily down; decides MOOT-close vs BLOCKED for infra items | answer with a date | treated as "unavailable"; nothing deleted |
| D-3 | Lineage `github/laptop-travel`: (A) archive, (B) cherry-pick a named set then archive, (C) merge, (D) canonical | (B): `06d22f2` (TOC-fallback hunk only), `2ab37b3`, `b2e31fe`, `0c124b4`, `5aead96`. NOT `cbb4fcf` (its QUALITY_GATES sync hunk writes the looser code thresholds into the docs, the opposite of the D-7 stance) and NOT `a7b30f5` (ported natively by WP-F1). Do NOT merge: 19 conflicted files, it changes gate exit semantics and breaks 7 scripts, and it rejects this line's fail-closed ingestion | do nothing |
| D-4 | Does the 2026-06-18 VLM deferral cover prompt/adapter edits (WP-Q1, WP-O1, WP-K1)? Define "first production-level release" independently of this plan | the release is NOT satisfied by this plan's n=1 acceptance; define it as Wave 4 met AND WP-V5a passing AND D-1 answered AND the owner acceptance review done | the deferral stands; WP-Q1/O1/K1 are not executed |
| D-5 | AGENTS Principle B ("forbid VLM text transcription") contradicts V3 (the V3 prompt requires transcription; the "Text transcription detected" flag fires on 3 of 4 IRJET images) | rescope Principle B to the v2 enrichment lane; keep the flag advisory-only | dated marker at Principle B: "CONTRADICTED BY CODE 2026-09-29 - see D-5" |
| D-6 | Competing definitions of done: AGENT-VAL-01 (`smoke_multiprofile.sh`) vs Mandate Sec. 2 (`smoke_production.sh`) | the Mandate is the stated conflict authority; AGENT-VAL-01 becomes a periodic corpus check; fix the script's exit semantics | gate 5 stays as Section 8.2 defines it; AGENTS text untouched |
| D-7 | "QA-CHECK-01 tolerance 0.10 for all profiles, no waivers" is false in code (missing tokens pass up to the profile noise allowance: 25% academic, 15% technical manual; IRJET measured -23%) | owner call; contract-violation-mode applies | neither side changed; both numbers reported |
| D-8 | Accept chunk-boundary / chunk_id churn from WP-B1/C2/C3/B2 and the resulting "stale by config_hash" outputs | accept for B1/C2/B2; C3 only if WP-M0's firing rate justifies it; re-ingestion stays owner-scheduled | B1/C2/B2 land (they restore a tested contract); C3 not executed |
| D-9 | Equation lane: prompt contract (iii-a) vs Docling formula enrichment vs crop lane; marker carrier `original_vlm_type` vs a new numeric field | (iii-a) + `original_vlm_type` | not executed |
| D-10 | Reading order: build a reorder pass or rely on restoring MinerU | advisory detector first | not executed |
| D-11 | `config_hash` field set | the list in WP-C4 | the WP-C4 list |
| D-12 | Track `scripts/env_cloud_vlm.sh`; delete the two untracked audit-prompt duplicates | track it; delete the duplicates on request | env script tracked in commit 0 (revocable); duplicates untouched |
| D-13 | Reversals recorded nowhere: LOCKED code-fencing undone by a fence-strip in `batch_processor`, the refiner never runs on V3, plan flags not reaching extraction | record what the code does as the decision of record, or restore the locked behavior; per item | dated markers only |
| D-14 | Embedding continuity and topology: production collections 4096-dim via omlx (down); ingest default provider dashscope 1024-dim; retrieval API defaults omlx; Qdrant home; snapshot wiring; two endpoint registries | owner decision; blocks any re-ingestion | do nothing (no re-ingestion) |
| D-15 | OCR "dead end" premise ("its ceiling drove the V3 pivot") is chronologically impossible (pivot 2026-05-29, score first measured 2026-06-09) and compares different page sets; it closed the narrow OCR-on-fallback item | re-word; decide OCR-on-fallback on new evidence | dated marker |
| D-16 | Hybrid retrieval listed as DEAD END while PROJECT_STATUS calls it the measured production path | re-word to "dead end as default vs plain top-10" | dated marker |
| D-17 | "Code indentation SOLVED" broader than the evidence; assertion masking in `tests/test_semantic_overlap.py` (6 tests turn AssertionError into SKIP); hidden env dependency in `test_qdrant_search_priority.py` | narrow the wording; propose the test fixes for sign-off | dated marker; tests untouched |
| D-18 | Release definition: README "feature-complete v2.16.0", CHANGELOG stops at v3.0.0-phase-c, no V3 tag | owner call | untouched |
| D-19 | Public GitHub mirror carries internal addresses in 45 files and 14 tracked logs | owner call | do nothing (never touched by this plan) |
| D-20 | Data policy for sending personal/financial documents to a cloud VLM; a cloud cost ceiling; paid measurement runs (INV-022/032); WP-V5b | owner call | no personal documents to the cloud, no paid run beyond WP-0.4 and WP-V1 |
| D-21 | Layer-0 policy-vs-code conflicts (G0-04, 08, 09, 11, 19, 20, 21, 29, 32; DC-05; AP-19, AP-31): for each, move the document to the code or the code to the document | per item | dated `CONTRADICTED BY CODE` markers only |
| D-22 | DECISIONS.md reversals and stale thresholds needing ratification (DC-09, 12, 13, 14, 15, 19, 21, 27, 28, 36, 39, 40; AP-55; INV-035) | ratify what the code does, per entry | dated markers; new FACT entries in WP-G2 |
| D-23 | Settled Precedents index and digest content (DC-17, DC-37, G0-34, G0-36, AP-47, AP-50): define AGENT-PRECEDENT-01 in AGENTS.md; re-scope the lines that say only `DoclingPdfAdapter` may build Docling | edit with owner sign-off | dated markers; AGENTS.md untouched |
| D-24 | Plan/phase closure records and target constraints (AP-14, AP-26, AP-35, AP-39; INV-036) | record closure text | untouched |
| D-25 | Repair-lane budget (INV-081): the recorded 25%/page rule belongs to PLAN_F1 Phase 2 (commit 46284af says Phase 2 execution remains unauthorized); does it govern `_repair_degraded_code`, per document or per batch (extract() runs per 10-page batch), rounding (ceil is the only choice that keeps `test_v3_code_repair.py` green; floor breaks two pins), page order, and the consequence that more code books ship unrepaired and fail R3 | owner call | not executed |
| D-26 | (unused id: folded into the adjudication and completion rules below) | - | - |
| D-27 | Main-line integration and CI (INV-003: HEAD is 98 commits ahead of origin/main; Gitea CI has never run V3 code), stale git artifacts (a 2026-05-31 stash, Cline checkpoint refs) | owner call | do nothing |
| D-28 | Non-PDF inputs still run only on the legacy V2 lane (INV-051): retire or port | owner call | untouched |
| D-29 | Measurement gaps worth funding (INV-056/057/058/059) | owner call | CLOSE as roadmap, recorded |
| D-30 | Deleting dead code and scripts (INV-061/067/076/077/078/079; Mandate 3(b) delete-with-entry) | owner sign-off per group | nothing deleted |
Adjudication: the owner adjudicates any disagreement between auditors; the plan's author does not. Completion if the owner stays silent: the plan is complete when Waves 0-2 and Wave 4 (V1, V2 packaged, V3, V4, V5a) meet Section 8 and every DECIDE/BLOCKED row has been answered or has taken its default on 2026-10-20.

## 10. KILL / closed-as-not-to-do (self-contained rationales)

- **Drawing clusters (`page.cluster_drawings()`) as extra crop candidates.** With correct frames it gives near-exact regions, but with shrunk cloud bboxes and the no-overlap fallback it mis-assigns clusters (Fig 3 received Fig 6's cluster), a content-rich wrong-figure crop the audit cannot see; it adds geometric scaffolding against F12; and it conflicts with the pinned `test_b1_uses_vlm_bbox_when_no_geometric_object` (a vector drawing is not a `get_image_info` object). The same wrong-figure risk exists for the existing no-overlap "largest remaining" fallback, which WP-A1's design study addresses.
- **A bbox-aware or long-description exemption in the tiny-icon / thin-strip filters.** Conflicts with `test_icon_class_image_on_content_page_is_dropped` and `test_thin_strip_on_content_page_is_dropped` (both build a large bbox on purpose and assert the drop). Removing the cause (sprite crops) leaves the filters unchanged.
- **Skipping the text-transcription validator for V3 output.** Conflicts with AGENTS Principle B until D-5; harmless today (advisory, no acting consumer).
- **Raising the 400-char `visual_description` cap.** Contractual and pinned; the defect is a consumer (WP-D7).
- **Wiring `has_encoding_corruption` to OCR/formula enrichment.** Workstream B guardrail and the OCR dead end; the flag is inert on V3 and was raised by a one-page Jaccard test on a figure page.
- **A bbox re-sorter, geometric re-merge, or merge veto in the VLM/chunker path.** F12: net-negative on VLM substrate; the veto also fires 0x once headers are removed.
- **`search_priority` boilerplate hardening (>= 50% of lines).** Measured: flips 36 of 55 demoted chunks to high including real copyright pages (HarryPotter p8, Kimothi p4).
- **`content_classification` fill and the `docling-2.86.0` literal fix (WP-B3 sub-items).** No consumer / never reaches the output.
- **A per-chunk VLM model id.** Not recoverable on the hybrid route (both engines stamp `ExtractionMethod.VLM`); replaced by the header field in WP-C4.
- **A hard-fail preflight for a missing `mineru-vl-utils` (INV-029).** The fail-closed ladder is SETTLED; the silent case is visible through `extraction_fallback_reason` and the `EXTRACTION_LADDER_SERVED` advisory.
- **Extending `has_flat_text_corruption` to reading order.** Different signal; own advisory if D-10 proceeds.
- **A profile seam at `extract()`.** Reverted upstream in `2ec40f5` (missed Devlin, re-opened the dense-table regression).

## 11. Risks, rollback, completeness

11.1 **Rollback:** one commit per WP on `fix/quality-remediation-v1`; behavior WPs carry no flag (WP-K1's per-page detection is off unless enabled). A regression found after acceptance is reverted. Local HEAD before this work: `7b515e3`; base `499a5fa`.
11.2 **CONTRACT-CONFLICT list (design would need an assertion edited; none is executed):** WP-E1 as originally worded with floor rounding vs `tests/test_v3_code_repair.py::test_flagged_page_repaired_when_vlm_strictly_better` and `::test_table_guard_allows_swap_when_table_preserved` (now D-25); stage-2 drawing clusters vs `test_b1_uses_vlm_bbox_when_no_geometric_object`; a bbox-aware icon/strip filter vs `test_icon_class_image_on_content_page_is_dropped` and `test_thin_strip_on_content_page_is_dropped`; skipping the validator vs AGENTS Principle B; adding a metric to the seeded-fault vector vs `test_seeded_fault_sensitivity.py`. The recommended WPs need none (prototype runs: A1-area, B1+C2, B1+C2b kept the full suite at 1767/99).
11.3 **Merge/conflict order (real, per file):** `uir_chunker.py`: B1, C2, C3, B2 (element pass entry); `batch_processor.py`: A1 (crop-audit persistence at ~1874), B2 (tracker registration), A2 (snippet/drop manifest), C4 (stamps), B3 (oversize marker ~7397, SMART-SPLIT ~9998); `processor.py`: C4 header writers only. No parallel edits of one file.
11.4 **Security review of new paths:** the metrics module, `qa_gold_anchor_smoke.py`, `qa_crop_fidelity.py` and the UIR dump read/write local files only (no network, no eval, no shell); the dump is opt-in by env; `env_cloud_vlm.sh` never stores a key; no push, no remote touched.
11.5 **Test-coverage gates:** red-first proof per WP; each new module has fire-on-bad, quiet-on-good and fix-induced-fault tests.
11.6 **Documentation strategy:** every behavior WP adds a stand-alone DECISIONS entry (heading-boundary rule, reference-entry ranking, furniture pass with its rank-based position rule and heading exclusion, snippet refresh and drop manifest, provenance semantics, crop predicate, consumers listed); one FINDINGS_LOG entry with the data tables; FINDINGS_DIGEST touched only by dated markers.
11.7 **Post-cycle sustainability:** the metrics run whenever `qa_full_conversion.py` runs (not on every conversion); each has a promotion criterion (full offline corpus pass and stability across >= 3 document classes, per AGENT-GATE-PROGRESSION) and a KILL date of 2027-01-31 recorded in FINDINGS_LOG: a metric with no promotion decision by then is removed by a DECISIONS entry (the previous six advisory metrics never received a decision: INV-062).
11.8 **Migration:** existing Qdrant collections are untouched. Chunk-id churn (D-8) makes pre-branch outputs "stale by config_hash"; re-extraction and re-ingestion are owner-scheduled and need D-1 and D-14 first.
11.9 **Route dependence:** engine-agnostic WPs (B1, C2, B2, A1, A2, C4, D7, B3, F1, F2) are valid on every route. Route-specific defects (reading order, equations, bbox frame) are labelled and decision-gated; the plan does not claim them fixed.

## 12. Stopping rule and audit cadence

Audit until two consecutive rounds return 0 HIGH findings counted across all auditors of the round. Round 1 needs at least two auditors from different model families (a
single clean audit can be a blind spot). Where auditors disagree, the owner decides; disagreements are shown side by side with each auditor's evidence. Findings whose proposed
fix violates an owner constraint are flagged, not applied.

## 13. Round 0 (internal adversarial review) - what changed from rev. 1

Two independent reviewers (a full-prompt audit and a feasibility/consumer audit with runnable prototypes, full-suite runs under the described designs) returned 6 HIGH / 19 MED / 9 LOW and
4 UNSOUND WPs. Structural changes made:
- Section 7 (register) was incomplete (14 open ids absent, about 25 mis-mapped, source USER-DECISIONs executed as safe edits, ids unresolvable in a clean clone): now a generated, verified, committed register with owner/answer-by/default per row and D-21..D-30 added.
- Wave 1 no longer rewrites SETTLED / DEAD-END / Settled-Precedents / AGENTS text (Section 3 rule 12).
- `undersized_asset_ratio` and `heading_inside_body_ratio` were mirrors of their own fixes (zero by construction) and one was frame-dependent: acceptance is now source-anchored (Section 8.1); mirrors are guards only; the crop size check moved to the frame-invariant sidecar.
- WP-A1's one-sided area floor regressed correct rescues (12 local crops) and produced wrong-region crops on the cloud frame: now a measure-first design study with a predicate-independent outcome check; cloud-route figure acceptance is stated non-claimable unless the frame is confirmed.
- WP-B2 would have deleted HarryPotter's chapter-title source and broken QA-CHECK-01 (-23% to about -31% vs the 25% allowance): now label-aware, heading-excluding, tracker-registered, measure-first; the boilerplate hardening and the merge veto are KILLed with measurements.
- WP-C2 did not fix the fresh-run shape (inline entries): inline-label rank and inline-aware metric added; the sentence-end premise corrected and split into WP-C3.
- WP-E1 applied a decision recorded for a different lane (and a floor rounding breaks two pinned tests): now D-25.
- Gate 5 ("run as-is") would have sent personal documents to a paid VLM: now a pinned offline before/after run with a pass criterion; n=1 acceptance replaced by WP-V5a (free) plus the stated limit for WP-V5b.
- WP-A2's refresh missed in-loop drops and three drop sites; WP-D6 per-chunk attribution is impossible; WP-B3(b) and the docling literal KILLed; WP-C4 gained consumers, a CI-enforced test and an honest claim (recordable, not computed).
- Added WP-M0 (firing rates), WP-0.4 (frame probe), WP-0.5 (UIR dump), validity preconditions for the live run.

## Appendix A - reproduction receipts

- Verdict verification: `python3` over `output/cloud_probe_irjet/ingestion.jsonl` (chunk table, asset dims vs bbox, header-contamination and heading-embedding counts); PyMuPDF `get_image_info()` per page of the source PDF; rendered pages 3 and 5 at 80 dpi (viewed).
- Gate blindness: `qa_full_conversion.py output/cloud_probe_irjet/ingestion.jsonl --source-pdf data/academic_journal/IRJET_Modeling_of_Solar_PV_system_under.pdf` -> `QA_PASS: failures=0 warnings=0`, `furniture_chunk_ratio=0.0000`.
- Furniture filter: `batch_processor.py:1550` `_FURNITURE_MAX_CHARS = 70` vs 151-char header chunks; band `y1 < 80` / `y0 > 920`.
- Fresh baseline: `mmrag-v2 process ... --output-dir output/irjet_baseline_499a5fa` (config from `~/.mmrag-v2.yml`, env from `scripts/env_cloud_vlm.sh`, `MINERU_ENDPOINT` unset): 175.9 s, 34 chunks, log line "[FINALIZE] Dropped 7 icon-class image chunk(s)".
- Test baseline: `pytest tests/ -q -p no:cacheprovider` -> 1767 passed, 99 skipped, 0 failed (56.6 s) on 499a5fa. With the WP-A1-area, WP-B1 + WP-C2 and WP-B1 + WP-C2b prototypes monkeypatched in: still 1767 / 99.
- Lineage: `git rev-list --left-right --count HEAD...github/laptop-travel` -> 215 (HEAD side) / 45 (laptop side); `git merge-tree` -> 19 conflicted files.
- Round 0 reports and the audit registers: `output/audit_2026-09-29/` (gitignored, local); the register file and this plan are the committed record.
