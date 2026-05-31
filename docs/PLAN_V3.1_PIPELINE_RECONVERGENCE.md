# PLAN_V3.1 - Pipeline Reconvergence

Status: IN PROGRESS (proposed 2026-05-31). P0 anchored on main; P1 + P2 DONE and
verified green on branch `v3.1-reconvergence` (commits 4099475, e947627), pending
merge. P3-P5 remain.
Owner: next coding session
Supersedes: nothing; complements docs/PROJECT_STATUS.md "Phase B Technical Debt"

## 1. The diagnosis (why the pipeline feels FUBAR after V3)

The V3 refactor delivered real extraction-quality wins, but it optimized and
PROVED those wins on a path that is not the path that ships. The project now
has two divergent realities:

- Reality A - the EVIDENCE path (sandbox). The V3_OVERNIGHT_REPORT numbers, the
  rebaseline, and the canonical layout were all produced by
  scripts/v3_batch_ingest.py, which imported the v3_execution_root sandbox
  chunker and wrote chunks through a permissive sandbox schema. This path is
  now RETIRED (the sandbox was deleted 2026-05-30) and BROKEN (both
  v3_batch_ingest.py and rebaseline_v3.py crash on import).

- Reality B - the SHIPPING path (production CLI). mmrag-v2 process ->
  BatchProcessor.process_pdf -> mmrag_v3.extract (HybridEngine) ->
  chunk_universal_document (uir_chunker) -> IngestionChunk.from_uir. This path
  was wired in Phase A Step 5 but never validated end to end. The 2026-05-31 M5
  smoke was the FIRST real run of it on image-bearing docs, and it immediately
  hit three latent blockers (response_format 400, missing asset_ref, missing
  visual_description - all fixed in 455eac8) plus a regressed quality gate
  (HEADING coverage) that had been "deferred" into a Phase B that was never
  built.

Consequences of the A/B split:
1. The headline "V3 beats V2.16 on every axis" was measured on sandbox chunking,
   not on what the CLI emits. The CLI produces different (fewer, denser) chunks
   (invoice: 4 vs ~11; IRJET: 45 vs 65). The result may still hold, but it is
   not currently reproducible through the shipping path.
2. The baseline-production tooling is broken, so there is no one-command way to
   regenerate baselines from the path that ships.
3. Quality gates that used to pass (HEADING coverage, breadcrumb/contextual
   retrieval) regressed and were deferred, not fixed.
4. The real deferred surface is 6 unconditionally-skipped V3_DEFERRED modules
   (the heading-propagation contract test_ocr_path_heading_propagation.py among
   them) - NOT the "182 skipped" total, which is mostly legitimate runtime
   skipif (corpus / GPU / network). test_repo_integrity.py is NOT skipped: it
   RUNS GREEN and is the G6 enforcer that keeps the deferred set registered. The
   gate that regressed is HEADING coverage; the deferred contracts are the
   safety net to restore.

Root cause in one sentence: V3 was declared done on the strength of a sandbox
path while the shipping path was left partially wired, its gates deferred, and
its baselines unreproducible.

## 2. North star

ONE canonical extraction path (the production CLI), GREEN gates on that path,
baselines that are REPRODUCIBLE from that path, and a guard that keeps it from
rotting. No second reality.

Strategic unlock that makes this affordable NOW: the self-hosted M5 Max VLM
(proven working this session) removes the OpenRouter weekly-budget ceiling that
made re-baselining impractical. Re-baselining through the CLI is no longer
budget-bound - it is LAN-local and free.

## 3. Issue register (what must end up true)

| ID | Issue | End state |
|---|---|---|
| R1 | Two extraction paths; evidence != shipping | Single CLI path; sandbox tooling gone |
| R2 | v3_batch_ingest.py / rebaseline_v3.py crash (sandbox import) | Repointed to CLI chunker or deleted |
| R3 | HEADING coverage QA_FAIL on academic/prose (parent_heading not propagated) | qa_full_conversion HEADING PASS; test un-skipped |
| R4 | breadcrumb_path empty -> contextual retrieval disabled (--no-contextual) | breadcrumb emitted; contextual retrieval re-enabled |
| R5 | 6 unconditionally-skipped V3_DEFERRED modules (incl. heading); test_repo_integrity is GREEN, not skipped | Skips at documented minimum; each remaining skip restored or deleted-by-decision per MANDATE §3 |
| R6 | Two Phase B heuristics still execute, tests deferred | Formally adopted or removed; tests un-skipped |
| R7 | V3-vs-V2.16 numbers from sandbox path | Re-baselined through CLI; reproducible report |
| R8 | No guard against path regression | Committed production-CLI smoke as pre-merge gate |
| R9 | EPUB extraction unsupported (2 docs) | Backlog; documented, not blocking |
| R10 | `chunk_universal_document` (V3 chunker ENTRY) has NO direct unit test - only exercised via test_v3_integration; a broken edit slipped past the entire `-k chunk` suite | Direct unit test added in P3 |
| R11 | PyMuPDF TOC quality varies (incomplete / absent bookmarks / scanned docs) - heading fallback robustness | Precedence puts in-page headings first, TOC only third; no-TOC short docs handled via documented gate skip; confirm on real TOCs at scale in P4 soak |
| R12 | Unbounded carry-forward poisons contextual embeddings on NO-TOC docs: one early in-page heading smears as parent_heading/breadcrumb across every later heading-less page (uir_chunker `_assign_headings` branch 2, no distance cap). NOT live until R4 turns contextual retrieval on. | Cap carry-forward by page distance; land WITH R4 (see P3 spec). Currently documented by test_carry_forward_distance_is_unbounded_until_overridden, which must be UPDATED (not deleted) to assert the cap. |

## 4. Phased execution (each phase has a hard exit gate)

### Phase 0 - Anchor and truth-up (0.5 day)  [R1 partial]
- DONE: commit the three production-path fixes (455eac8).
- Run scripts/smoke_multiprofile.sh-style coverage through the CLI on one doc
  per lane (invoice / academic / prose; reuse the 2026-05-31 smoke set) and
  record real per-doc chunk/modality/routing/QA. This is the new "shipping
  reality" snapshot.
- Annotate V3_OVERNIGHT_REPORT.md and PROJECT_STATUS.md: mark the sandbox
  numbers as "sandbox-path, NOT reproducible via CLI - see PLAN_V3.1", so no one
  cites them as the shipping baseline.
- EXIT GATE: tree committed; a CLI-produced baseline table exists for >=3 docs
  spanning all routing lanes.

### Phase 1 - Collapse to one path (1 day)  [R1, R2]
STATUS: DONE on branch v3.1-reconvergence (commit 4099475) - v3_batch_ingest.py +
rebaseline_v3.py repointed off the sandbox onto the production chunker; `grep -rn
v3_execution_root scripts/` empty; runs offline emitting valid IngestionChunk
JSONL. Pending merge.
- Replace v3_batch_ingest.py and rebaseline_v3.py with thin wrappers over the
  production path (BatchProcessor / mmrag-v2 batch + uir_chunker +
  IngestionChunk.from_uir). No script may import a chunker the CLI does not use.
  Prefer DELETING v3_batch_ingest.py and folding its resume-safe batch loop into
  a single `scripts/v3_baseline.py` that calls the CLI internals.
- Remove all v3_execution_root references and dead TODOs from scripts/.
- EXIT GATE: `grep -rn v3_execution_root scripts/` is empty; the baseline tool
  runs and emits IngestionChunk JSONL byte-shape-identical to `mmrag-v2 process`
  on a 1-page doc (assert via a diff test).

### Phase 2 - Fix HEADING coverage for real (1-2 days)  [R3, R4]
STATUS: DONE on branch v3.1-reconvergence (commit e947627) - UIR-native TOC
heading + breadcrumb propagation in uir_chunker._assign_headings; on an academic
doc with bookmarks parent_heading 68%->100%, breadcrumb 0%->100%, HEADING gate
FAIL->PASS; test_ocr_path_heading_propagation restored (+7 contracts); full suite
1293/119/0. No Docling added to batch_processor. Pending merge.

This is the "should have been fixed long ago" item. It is NOT hard - the inputs
already exist.

Root cause: Phase A Step 5 stripped the heading-reconcile/front-matter
heuristics from batch_processor finalize. The PyMuPDF TOC is still extracted
(batch_processor._extract_toc_headings) but never applied to V3 chunks; the
uir_chunker detects only in-page headings (Element.source_label heuristics),
does not carry a heading across pages, and emits no breadcrumb_path.

Fix - a UIR-native heading-assignment pass (in uir_chunker, or a finalize step
fed the TOC), with this precedence per text chunk:
1. Nearest preceding in-page heading element on the same page (existing logic).
2. Carry-forward: the last active heading from earlier pages if none on-page.
3. Fallback: the PyMuPDF TOC entry whose page range covers this chunk's page.
4. breadcrumb_path: built from the TOC tree (level hierarchy) for that page.

Do it UIR-native; do NOT resurrect the deleted batch_processor methods (keeps
AGENT-SPATIAL-20 and the engine-agnostic boundary intact).

TOC-robustness note (R11): PyMuPDF TOC quality varies - some PDFs have no
bookmarks (born-digital leaflets, parts lists), poor producers emit flat/partial
trees, scanned docs have none. The precedence above de-risks this by design: an
in-page heading always wins over the TOC, and carry-forward covers pages between
TOC anchors. The genuine no-structure case (short born-digital doc, no bookmarks,
no headings) is handled by the documented `short_document` HEADING-gate skip
(DECISIONS.md), NOT by fabricating headings. Robustness on messy real-world TOCs
is exercised at scale in the P4 soak, not asserted from one fixture.

Bonus (R4): emitting breadcrumb_path lets contextual retrieval be re-enabled
(drop --no-contextual in the indexer), recovering the contextual-embedding
quality that the V3 index gave up.

Un-defer: re-enable tests/test_ocr_path_heading_propagation.py (and any heading
assertions in the deferred modules). They become the contract that keeps this
fixed.
- EXIT GATE: qa_full_conversion HEADING gate PASS on IRJET + the academic smoke
  docs (coverage >= profile threshold); test_ocr_path_heading_propagation.py
  runs GREEN (not skipped); breadcrumb_path populated on a sampled chunk.

### Phase 3 - Restore the gate wall (1-2 days)  [R5, R6, R10]
STATUS: not started. After P2 the deferred surface is 5 V3_DEFERRED modules (the
heading module was restored in P2; 2 dead-code wiring assertions inside it were
deleted-by-decision). test_repo_integrity.py is NOT a deferred test - it RUNS
GREEN and is the G1-G6 enforcer that keeps the deferred set registered; keep it
green, do not skip it.
- Walk the 5 remaining V3_DEFERRED modules. For each: (a) restore behavior +
  re-enable, or (b) DELETE the test with a DECISIONS.md entry recording the
  dropped behavior (MANDATE §3b). No permanent silent skips:
  - test_cross_chunk_semantic_stitching.py
  - test_docling_postprocess_ocr_gating.py
  - test_docling_postprocess_profile_integration.py
  - test_vision_aided_front_matter.py
  - test_pdf_conversion_plan.py
- NEW (R10) - add the missing unit test for the V3 chunker ENTRY:
  `chunk_universal_document` has NO direct unit test today; it is only exercised
  end-to-end by test_v3_integration, which is how a broken edit (a kwarg the
  callee did not accept) slipped past the entire `-k chunk` suite. Add a direct
  test that feeds a small in-memory UniversalDocument fixture (TEXT + IMAGE +
  TABLE elements across >=2 pages, with and without a TOC) and asserts: chunk
  count + modality split, int [0,1000] bboxes, parent_heading precedence
  (in-page > carry-forward > TOC), and breadcrumb_path. This is the unit-level
  safety net the entry point lacks - it would have caught both the kwarg break
  and any future signature drift.
- NEW (R12) - cap carry-forward distance, landed WITH R4 (contextual retrieval).
  Today uir_chunker._assign_headings carry-forward (branch 2) is unbounded:
  on a NO-TOC doc one early in-page heading is stamped as parent_heading +
  breadcrumb on every later heading-less page. Harmless while --no-contextual is
  set (R4 off), but the moment contextual retrieval is re-enabled those stale
  prefixes get embedded and degrade dense recall. Scope note: this only bites
  NO-TOC docs - when a TOC exists, _extract_toc_headings fills page_map for every
  page from the first anchor onward, so branch 3 (per-page TOC breadcrumb) wins
  over carry-forward. Spec:
    * Track `last_heading_page` = the page where last_heading was established by
      an in-page heading (branch 1) or TOC (branch 3).
    * Add a named, tunable constant `MAX_CARRY_FORWARD_PAGES` (proposed default
      3; only affects no-TOC docs). In branch 2, carry only when
      `page - last_heading_page <= MAX_CARRY_FORWARD_PAGES`.
    * Beyond the cap: parent_heading = None (honest - section unknown);
      breadcrumb_path = [doc_name, "Page N"] (doc identity + page, never a stale
      section). Null-section is better for embeddings than a wrong section.
    * Update test_carry_forward_distance_is_unbounded_until_overridden to assert
      the cap (the test already instructs this; rename + assert cap, do NOT
      delete - AGENT-TEST-01). Add a DECISIONS.md entry (stricter contract).
  Sequencing: implement in the same change that drops --no-contextual (R4), so
  the cap and its only consumer land together and are soak-validated in P4.
- Resolve R6: ADOPT (un-skip the test) or REMOVE the two still-executing
  heuristics (_apply_spatial_refiner vertical-proximity merger,
  _merge_mid_sentence_chunks). The "Phase B LLM-sanitization" basis is falsified
  (MANDATE §3 / V3_DEFERRED_TESTS.md) - do not keep deferring to it.
- EXIT GATE: deferred-module count reduced to a documented minimum; every
  remaining skip restored or deleted-by-decision; chunk_universal_document has a
  direct unit test; test_repo_integrity.py + full suite green.

### Phase 4 - Re-baseline and lock (1 day)  [R7]
- With the CLI clean and gates green, and the GX10 judge now a standing service
  (fixed 2026-05-31), re-run the synthetic soak THROUGH the production CLI path
  on the canonical doc set via the M5 VLM (no OpenRouter budget needed). Produce
  the real V3-vs-V2.16 head-to-head from the shipping path.
- Decide the table-chunk granularity question (1 dense chunk vs N row-chunks)
  with soak evidence (Recall@k on row-level queries), not assumption.
- Retrieval validation: report Recall@1/@5 (judge-free) AND the GX10 judge axes,
  reading Format as the trustworthy axis per the calibration memo.
- TOC robustness at scale (R11): the soak doc set spans varied TOC quality (rich
  bookmarks, flat/partial, none, scanned). Spot-check heading/breadcrumb coverage
  per doc-class; confirm the P2 precedence holds beyond the single P2 fixture and
  that no-TOC docs land in the documented short_document class rather than
  failing. Flag any class where coverage is low for a structured doc (= a real
  regression, not a gate artifact).
- EXIT GATE: a soak report reproducible from `mmrag-v2 ...` + the standing judge;
  PROJECT_STATUS reflects single-path reality and cites the new numbers.

### Phase 5 - Anti-rot guard (0.5 day, then ongoing)  [R8]
- Promote the 2026-05-31 smoke into scripts/smoke_production.sh: runs the
  production CLI on one doc per lane (offline USE_DOCLING_FAST for CI; M5 for the
  full check) and ASSERTS: no 0-chunk batches; every IMAGE/TABLE chunk has
  asset_ref + visual_description; routing matches per-lane expectation; QA_PASS
  (or documented WARN). This is the guard that would have caught all three bugs
  found this session.
- CLAUDE.md: name this smoke as the mandatory pre-merge check for any change to
  the extraction path (batch_processor, uir_chunker, mmrag_v3 engines, from_uir).
- EXIT GATE: one-command smoke; CLAUDE.md updated.

## 5. Sequencing rationale

Commit first (anchor). Collapse to one path (P1) BEFORE fixing quality (P2):
fixing HEADING on a path that may still change is wasted. Un-defer the gates
(P3) to lock the fixes so they cannot silently regress. Re-baseline (P4) only
once the path is trustworthy, else the numbers are noise. Guard last (P5) so the
guard encodes the now-correct expectations.

## 6. Risk register

| Risk | Mitigation |
|---|---|
| Heading-reconcile re-intro conflicts with AGENT-SPATIAL-20 / stripped finalize | Implement UIR-native in uir_chunker; never resurrect deleted batch_processor methods |
| Un-deferring tests surfaces more regressions | That is the goal; surface now, not in production. Time-box each module to restore-or-delete |
| Re-baseline cost | Removed - M5 self-hosted VLM is LAN-local and free; GX10 judge now standing |
| Granularity change churns the index | Decide once in P4 with soak evidence; freeze in DECISIONS.md |
| Scope creep into "Phase B LLM-sanitization" | P1.1 forbids it - decide adopt-or-remove the heuristics, do not build a new layer to justify the defer |
| TOC quality varies (incomplete / absent / scanned bookmarks) | Precedence puts in-page headings first, TOC only third; no-TOC short docs handled via documented gate skip; confirm at scale in P4 soak (R11) |
| V3 chunker entry untested at unit level (broke silently once) | P3 adds a direct chunk_universal_document unit test (R10); the P5 smoke is the integration-level backstop |

## 7. Definition of done (the whole plan)

- `grep -rn v3_execution_root scripts/` empty; one extraction entry point.
- qa_full_conversion QA_PASS (or documented WARN) on every doc in the
  multi-profile smoke, HEADING gate included.
- Skip count at documented minimum; test_repo_integrity.py and
  test_ocr_path_heading_propagation.py green.
- chunk_universal_document has a direct unit test (no longer only reachable via
  test_v3_integration).
- A V3-vs-V2.16 soak report reproducible from the production CLI + standing GX10
  judge.
- scripts/smoke_production.sh is the one-command pre-merge gate; CLAUDE.md
  points to it.

## 8. Out of scope / backlog
- EPUB extraction (2 docs) - needs a fitz-independent adapter. Track separately.
- Magazine-layout high VLM-fallback rates (page tiling / per-doc max_tokens).
- Long-tail >300p tech-manual books (now affordable via M5; schedule after P4).
