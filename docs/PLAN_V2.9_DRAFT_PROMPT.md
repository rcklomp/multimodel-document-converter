# Prompt for Writing the v2.9 Plan

**Instructions for Claude / AI Agent:**

You are a Senior Architect tasked with writing the production execution plan for **v2.9** of the MM-Converter multimodal document ingestion pipeline. This plan will be saved as `docs/PLAN_V2.9.md` and will drive the next development cycle.

Follow these instructions precisely. The plan must be actionable, evidence-based, and structured for TDD execution.

**Repository state at the start of v2.9:** v2.8.0 has shipped (annotated tag on `645ab2b`, pushed to GitHub `rcklomp/multimodel-document-converter`). The 7-commit v2.8 chain is `5b0e13d → c2e795e → 9e4b8f8 → 59994f9 → 2f94503 → 0d3cc36 → 9726b43 → 645ab2b`. All four PLAN_V2.8 production gaps (Workstreams B, C, F, §5) are empirically closed. Engine `__engine_version__=2.8.0`, schema `__schema_version__=2.7.0` (de-aliased; v2.8 made no chunk-shape change). Test suite: 596 passed, 2 skipped, 0 failed. Smoke matrix: 11/11 GATE_PASS + 11/11 UNIVERSAL_PASS.

---

## Step 1: Read Required Documentation

Before drafting the plan, read the following files in order. ENOT what I asked
ach provides essential context that the plan must respect:

1. **`docs/PROJECT_STATUS.md`** — Current state, active baseline, open items, known failures, completed work. This tells you what's broken and what's fixed. The "Active Baseline" pointer leads to QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md.
2. **`docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md`** — The v2.9 BEFORE state. Empirical Phase outcomes from v2.8, the 4 documented v2.9 followups (Ayeva, Firearms, chunk_id dupes, refiner smart-routing), and the Qdrant ingest evidence (22,137 / 22,160 unique chunks in `mmrag_v2_8`, all images placeholder).
3. **`docs/PROGRESS_CHECKLIST.md`** — Workstream-level status tracking. Workstreams B, C, F are now `[x] CLOSED 2026-05-04`. Read the v2.8 closure entries to understand what's already done before duplicating any item in v2.9.
4. **`docs/PLAN_V2.8_PRODUCTION_GAPS.md`** — Study the *structure* of this plan. It is the gold standard for execution planning in this project. Your v2.9 plan must follow its format: phases, parallel-site audits, TDD red→green, acceptance gates, decision log. Pay particular attention to §2b Parallel-Site Audit (the lesson from `processor.py:2072`).
5. **`docs/DECISIONS.md`** — Architectural decisions and their rationale. Your plan must not contradict any ratified decision. If a decision needs revisiting, document that as a phase investigation step. Note especially the 2026-05-03 amendment to "Selective Code Enrichment Lane" (client-local CPU CodeFormulaV2 is acceptable for one-off batch).
6. **`AGENTS.md`** — Hard invariants (numbered `AGENT-*`). These are non-negotiable constraints. Reference them in your plan where relevant. **Critical:** if your phase appears to require violating an invariant (e.g. AGENT-SPATIAL-20 forbids profile-specific spatial branches), do not silently violate it — make the conflict explicit, propose an investigation step, and either find a respecting fix or document the proposed amendment.
7. **`docs/QUALITY_GATES.md`** — Pass/fail thresholds. All phases must produce outputs that satisfy these gates. The form acceptance class (added 2026-05-04 in v2.8 Phase 5a) is documented here; FORM_PASS and "GATE_PASS [form: ...]" are first-class success states, NOT waivers.
8. **`docs/ARCHITECTURE.md`** — System architecture, UIR contracts, pipeline flow. Ensure your phases respect the architecture.
9. **`CHANGELOG.md`** — `[2.8.0] — 2026-05-04` section enumerates every v2.8 Added / Changed / Fixed item with code paths. Use this to avoid scoping work that's already done.

---

## Step 2: Analyze v2.9 Scope

From your document review, identify and prioritize the v2.9 work items. **All items below are open as of v2.8.0 ship.** Two work items previously in this list (SCAN0013 form gate; Adapter-invocation static guard) are intentionally absent — they shipped in v2.8 (Phase 5a and Phase 2 respectively). The agent must NOT re-scope shipped work. Cross-reference each candidate against `CHANGELOG.md` `[2.8.0]` and `docs/PROGRESS_CHECKLIST.md` `[x] CLOSED 2026-05-04` markers before adding it to a phase.

### Priority Work Items (in priority order, highest user impact first):

**1. VLM enrichment of the `mmrag_v2_8` Qdrant collection (Workstream E, NEW PRIORITY)**

The 2026-05-04 v2.8 broad reconversion used `--vision-provider none --no-refiner --no-cache` for apples-to-apples baseline matching. As a result, **all ~5,500 image chunks** in the `mmrag_v2_8` Qdrant collection have placeholder `visual_description="[Figure on page N] | Context: <breadcrumb>"`, `vision_status="pending"`, and `refined_content=null`. Image-side RAG retrieval is currently degraded — searching for "wizard ornament" or "F-35 photo" cannot match.

The cheapest path is **targeted image-only enrichment**: load each canonical `output/<doc>/ingestion.jsonl`, identify image chunks with `vision_status="pending"`, run the configured VLM per image, update `visual_description` + `refined_content` + `vision_status="complete"` in place, then re-ingest just the image points into `mmrag_v2_8`. No re-extraction. The point_id collision fix from `0d3cc36` (UUID5 from chunk_id) means the re-ingest correctly upserts each image point in place.

Empirical: 22,137 points currently in `mmrag_v2_8`; ~5,500 are images requiring enrichment. VLM choice (cloud qwen3-vl-plus vs local NuMarkdown) ties directly into Priority 4 below.

**2. Refiner smart-routing fix (`cli.py:686`)**

The CLI's config-default refiner-enable logic at `src/mmrag_v2/cli.py:683-687`:

```python
_refiner_explicitly_disabled = "--no-refiner" in _sys.argv
if not enable_refiner and cfg.refiner.enabled and not _refiner_explicitly_disabled:
    enable_refiner = True
```

Fires whenever `~/.mmrag-v2.yml` has `refiner.enabled=true`, **regardless of `has_encoding_corruption`**. The diagnostic-driven auto-override at `cli.py:1101-1102` only matters when refiner is still off at that point, so it's effectively dead code under the config-default path.

Empirical evidence: the v2.8 broad reconversion's first attempt left HARRY (clean prose, zero encoding corruption) hammering qwen-plus per chunk with refinements rejected ("Edit ratio 53.16% exceeds budget"). The remediation was the `--no-refiner` flag — masking the bug, not fixing it.

Surgical fix: gate the config-default enable on `has_encoding_corruption=True` (need to defer the decision until after diagnostic runs, OR move the auto-override logic in front of the config-default). Unblocks running `convert_books.sh` without explicit `--no-refiner`.

**3. ProfileClassifier rule 0c tightening (Ayeva misclassification)**

`Ayeva_Python_Patterns` in the v2.8 fresh re-conversion routes to `digital_literature` instead of `technical_manual`. The misclassification suppresses the `needs_code_enrichment` cheap-evidence trigger (CodeFormulaV2 doesn't auto-engage for the `digital_literature` profile). Empirical:

- BEFORE (Phase 0, `output/Ayeva_Python_Patterns/` 2026-04-12 + `output/ayeva_qa_20260501/` 2026-05-01): the older probe routed correctly to `technical_manual`, CODE PASS, `indentation_fidelity=0.93`.
- AFTER (v2.8 fresh, 2026-05-04 in `output/Ayeva_Python_Patterns/`): routes to `digital_literature` (rule 0c misfire on a code-heavy book), CODE FAIL, `indentation_fidelity=0.83` (just under the 0.85 hard gate).

Rule 0c is in `src/mmrag_v2/orchestration/document_diagnostic.py` (added 2026-05-03 in commit `2f51816`): `_dialogue_pages >= 1 AND total_pages > 20 AND not has_tables AND 500 < avg_text_per_page < 2500 → literature += 0.4`. Ayeva probably trips this because Python code blocks include strings with quotation marks that the dialogue heuristic misreads.

Proposed fix direction: require `code_evidence_pages < 2` (or equivalent code-density inverse signal) before allowing the literature route. Verify on Ayeva, Chaubal, Fluent, HARRY, and the SCAN0013 / business-form set so HARRY still routes correctly.

**4. Firearms heading regression — AGENTS.md conflict to resolve first**

`Firearms` v2.8 fresh re-conversion: profile changed `scanned` → `technical_manual` between baselines, and the chunker's heading-inheritance is stricter under `technical_manual`. HEADING coverage 100% → 78% (gate is ≥80%). 178 / 815 chunks now lack `parent_heading`. Same content fidelity, just less hierarchy annotation.

**Constraint conflict to resolve before designing the fix:** `AGENT-SPATIAL-20` says (paraphrasing) "keep the single 20-unit vertical threshold behavior; no profile-specific branching for that rule." The naive fix — relaxing the heading threshold for `technical_manual` on scanned-modality input — directly violates this. The v2.9 plan must NOT silently violate AGENT-SPATIAL-20. Two acceptable resolutions:

- (a) **Re-route Firearms to the `scanned` profile.** Investigate why the classifier flipped the route; the fix lives in `profile_classifier.py`, not in the spatial threshold. This respects AGENT-SPATIAL-20 unchanged.
- (b) **Propose an explicit AGENT-SPATIAL-20 amendment** in `AGENTS.md` + decision log entry, with empirical evidence that the single-threshold rule no longer serves the corpus. This requires user sign-off; do not auto-amend.

Default: (a). (b) only if (a) demonstrably fails.

**5. Within-file chunk_id collision fix (schema-level)**

The v2.8 broad reconversion produced 22,587 chunks with only 22,160 unique `chunk_id`s — **427 within-file duplicates**, largest contributor `KI_En_ChatGPT_Praktische_Gids` with 279, then `Devlin_LLM_Agents` 76, `Fluent_Python` 15. These are typically boilerplate page footers / repeated page numbers / identical short labels across pages.

`src/mmrag_v2/schema/ingestion_schema.py::_generate_chunk_id(doc_id, content, page_number, type)` hashes only `(doc_id, content, page, type)`. Two chunks on the same page with identical content collapse to the same id. They then collide on Qdrant upsert (the v2.8 ingest landed only the unique 22,160 minus 23 embed errors = 22,137 points; the 427 dupes silently overwrote each other).

Fix: include the chunk's per-document position index (or `i+1`) in the hash seed. Surgical change in one function. Schema_version stays 2.7.0 (the `chunk_id` *value* changes for affected chunks, but the field shape doesn't).

Add a regression test: build a doc with two visually-identical paragraphs on the same page, assert distinct `chunk_id`s.

**6. Local VLM comparison (Workstream A)**

Cloud-tested with `qwen3-vl-plus` (Source Sanctity validated). Local provider: `NuMarkdown-8B-Thinking-mlx-8bits` at `http://10.0.10.246:8000/v1`. Compare local vs cloud on the blind set manifest (`tests/fixtures/blind_set_manifest.json`). Determine if the local model meets Source Sanctity standards. **Direct dependency for Priority 1** — if local meets Source Sanctity, the `mmrag_v2_8` enrichment can use local for cost; otherwise cloud.

**7. Remote CodeFormulaV2 inference target (Workstream B followup)**

v2.8 accepted client-local CPU CodeFormulaV2 at ~27 sec/page for one-off batch (per `DECISIONS.md` "Selective Code Enrichment Lane → Amendment 2026-05-03"). The Amendment also says: "If reconversion of code-heavy docs becomes routine (more than once per week), invest in remote inference setup for v2.9."

Docling 2.86 does NOT expose `RemoteCodeFormulaOptions` / `ApiCodeFormulaOptions` — only the inline `CodeFormulaModel` ships. v2.9 options:

- (a) Custom adapter that intercepts `CodeItem`s post-Docling and POSTs them to a remote VLM endpoint.
- (b) Wait for upstream Docling to add remote options.

Decide based on actual code-heavy reconversion frequency observed since v2.8 ship. If still one-off, defer to v2.10.

### Items intentionally OUT of v2.9 scope:

- **Adapter-invocation static guard** — shipped in v2.8 Phase 2 (`tests/test_pdf_conversion_plan.py::test_no_raw_converter_invocation_outside_adapter`).
- **SCAN0013 form-aware gate** — shipped in v2.8 Phase 5a. SCAN0013 row in the smoke matrix now reports `GATE_PASS [form: micro_non_label + label-orphan checks skipped]`. The form acceptance class is documented in `docs/QUALITY_GATES.md`. Do NOT re-scope.
- **Qdrant ingest collision-free point_id** — shipped in v2.8 mid-Phase-5c (`fix(ingest): collision-free point_id` commit `0d3cc36`). 6 regression tests in `tests/test_qdrant_point_id_collision.py`.

---

## Step 3: Write the Plan

Create `docs/PLAN_V2.9.md` following this exact structure (modeled on `PLAN_V2.8_PRODUCTION_GAPS.md`):

### Header Section
```
# Plan: v2.9 — [Title]

**Status:** Draft v1.0
**Owner:** [workstream owner]
**Successor to:** `docs/PLAN_V2.8_PRODUCTION_GAPS.md` (shipped 2026-05-04)
**Related:** `docs/PROJECT_STATUS.md`, `docs/PROGRESS_CHECKLIST.md`,
`docs/QUALITY_GATES.md`, `docs/DECISIONS.md`
```

### Section 1: Why This Plan Exists
- Brief executive summary of v2.8 completion status (30/34 pass, 4 carryover items)
- Table of workstreams, symptoms, concrete patterns, last evidence
- What closing all phases achieves

### Section 2: Goals & Non-Goals
**Goals:** Clear, measurable outcomes.
**Non-goals:** Items explicitly deferred to v2.10 or later.

### Section 2b: Parallel-Site Audit (Cross-Cutting Principle)
- Replicate the §2b from v2.8 plan — this is now a permanent requirement for every phase.
- List the 4 questions every phase must answer before designing a fix.

### Section 3: Phases
For each phase, include:

- **Phase N — [Workstream]: [Title]**
- **What:** One-paragraph problem description with concrete evidence (file paths, patterns, numbers).
- **Parallel-site audit (do this FIRST):** Table listing file:line, current behavior, and action for each relevant call site.
- **Approach:** Steps to resolve, ordered from investigation to implementation.
- **Tests (red→green):** Specific test names and assertions. TDD discipline.
- **Done when:** Pass/fail criteria for the phase.
- **Risk:** Assessment (Low/Medium/High) and mitigation.
- **Estimated effort:** Time estimate.

### Section 4: Acceptance Gate
- Shell commands for smoke test, unit tests, audit.
- Expected outcomes for every command.
- Human-readable quality checks per document.

### Section 5: Out of Scope
Table with items deferred to v2.10+, with rationale and owner doc.

### Section 6: Cross-Phase Concerns
- Documentation updates per phase.
- Test contract integrity notes.
- Upstream tracking items.

### Section 7: Effort Summary
Table: Phase | Estimate | External dependency?

### Section 8: Decision Log
Chronological entries for plan revisions and rationale.

---

## Step 4: Quality Requirements for the Plan

Your draft must meet these standards:

1. **No generic steps.** Every phase must reference specific files, line numbers, code patterns, and empirical evidence from the current codebase.
2. **Parallel-site audit tables are mandatory.** Skipping them is a recurring failure mode (the `processor.py:2072` incident from v2.8 happened because v2.7 §5 only added a *construction* guard, missing the *invocation* path).
3. **TDD discipline.** Every phase must specify red→green test names and assertions before describing implementation.
4. **No silent violations of AGENTS.md.** Reference `AGENT-*` constraint IDs where relevant. If a phase appears to require violating an invariant (e.g. `AGENT-SPATIAL-20`, `AGENT-VAL-01`), make the conflict explicit and either find a respecting fix OR propose an explicit amendment for user sign-off. Do not silently violate.
5. **No contradictions with DECISIONS.md.** If a decision needs revisiting, make that an investigation step. Cite the relevant decision-log entry by date (e.g. "Selective Code Enrichment Lane → Amendment 2026-05-03").
6. **Acceptance is binary, but not narrow.** Every smoke row must show `GATE_PASS` + `UNIVERSAL_PASS`. The form acceptance class shipped in v2.8 Phase 5a IS a valid GATE_PASS variant — `GATE_PASS [form: ...]` and `FORM_AUDIT_PASS` count as acceptance, NOT as waivers (per `docs/QUALITY_GATES.md` "Form / Invoice Acceptance Class"). Per `AGENT-VAL-01`: no profile-specific waivers on the universal invariants either.
7. **Empirical evidence required.** Reference specific output JSONL files, chunk counts, audit scores, and timestamps. Cite v2.8 commit hashes (`5b0e13d`, `c2e795e`, `9e4b8f8`, `59994f9`, `2f94503`, `0d3cc36`, `9726b43`, `645ab2b`) where the v2.9 work depends on a v2.8 change.
8. **Effort estimates are realistic.** Calibration from the v2.8 plan: surgical schema edits 1-2 h; AST-based static guards 1-2 h; investigation-then-fix phases 0.5-2 days; corpus-wide reconversion 1-2 days runtime. Add CPU/runtime line items separately from engineering time when they dominate.
9. **Python 3.10 only.** No syntax/features from 3.11+. `pyproject.toml` is locked at `requires-python = ">=3.10,<3.11"`.
10. **PDF batch size ≤ 10 pages.** Resource ceilings other than batch size are not currently documented in `CLAUDE.md` — do not invent constraints. If memory cap matters for a phase, measure and document it as part of the phase.

---

## Step 5: Additional Context to Incorporate

### Current Environment State
- **Docling version:** 2.86.0 (exact-pinned in `pyproject.toml`)
- **Python:** 3.10
- **Conda env:** `mmrag-v2` (activate with `conda activate mmrag-v2`)
- **CLI entry:** `mmrag-v2` → `mmrag_v2.cli:main`
- **Schema version:** 2.7.0
- **Apple Silicon (M-series)** — prefer ARM64, use MPS when available

### Key File Locations
- Core pipeline: `src/mmrag_v2/`
- Profile classifier: `src/mmrag_v2/orchestration/profile_classifier.py`
- Document diagnostic: `src/mmrag_v2/orchestration/document_diagnostic.py`
- Batch processor: `src/mmrag_v2/batch_processor.py`
- CLI: `src/mmrag_v2/cli.py` (refiner routing at ~line 686)
- Tests: `tests/`
- Smoke test: `scripts/smoke_multiprofile.sh`
- QA audit: `scripts/qa_conversion_audit.py`
- Universal invariants: `scripts/qa_universal_invariants.py`
- VLM quality: `scripts/vlm_quality_summary.py`

### VLM State
- Cloud: `qwen3-vl-plus` (tested, Source Sanctity validated)
- Local: `NuMarkdown-8B-Thinking-mlx-8bits` at `http://10.0.10.246:8000/v1`
- Blind set manifest: `tests/fixtures/blind_set_manifest.json`
- Prompt harness: `scripts/eval_vlm_image_prompt.py`

### Known Document States
- **Ayeva:** `output/ayeva_qa_20260501/` (baseline, `technical_manual` route) — CODE PASS, `indentation_fidelity=0.93`. **v2.8 fresh** `output/Ayeva_Python_Patterns/` — routes to `digital_literature`, CODE FAIL, `indentation_fidelity=0.83`
- **Chaubal:** v2.8 reconversion — `indentation_fidelity=0.96`, CodeFormulaV2 engaged
- **Combat Aircraft:** v2.8 reconversion — `encoding_artifacts=0`, `high_corruption=0`
- **Hybrid Electric Review:** v2.8 reconversion — `ctrl_chunks=0`
- **Harry Potter:** `output/HarryPotter_and_the_Sorcerers_Stone/` — auto-routes to `digital_literature`
- **SCAN0013:** FORM_PASS (shipped v2.8 Phase 5a) — do NOT re-scope
- **Firearms:** v2.8 fresh — profile `scanned` → `technical_manual`, HEADING coverage 100% → 78%, 178/815 chunks null `parent_heading`
- **KI_En_ChatGPT:** 279 within-file chunk_id dupes (largest contributor)

---

## Step 6: Output

Write the complete plan to `docs/PLAN_V2.9.md`. The plan must be:

1. **Self-contained** — a new agent can execute it without chat history
2. **Evidence-linked** — every claim references a file, output, or test
3. **Phase-ordered** — cheapest/lowest-risk fixes first, deeper investigations later
4. **Mergeable** — each phase is independently testable and mergeable
5. **Bounded** — clear scope, clear out-of-scope, clear effort estimates

After writing the plan, verify it against this checklist:
- [ ] No generic steps; all phases reference specific files/patterns
- [ ] Every phase has a parallel-site audit table
- [ ] Every phase has red→green test specifications
- [ ] AGENTS.md invariants either respected OR conflict explicitly flagged with proposed amendment
- [ ] No contradictions with DECISIONS.md rulings (or investigation step + decision-log entry queued)
- [ ] Acceptance gate requires GATE_PASS + UNIVERSAL_PASS for all rows; FORM_PASS / `GATE_PASS [form: ...]` accepted as variants
- [ ] Effort estimates are included (engineering time + runtime broken out separately)
- [ ] Out-of-scope items are documented (especially shipped-in-v2.8 items must NOT appear)
- [ ] Decision log section is present
- [ ] The `mmrag_v2_8` Qdrant collection's image-placeholder gap is addressed in some phase (Priority 1 in §2)
- [ ] Phase sequencing front-loads cheap fixes (ratio of "1-2 h" to "0.5-2 day" phases mirrors v2.8: Phase 1+2 small, Phase 3+4 deeper, Phase 5 broad reconversion + ingest)
- [ ] No work item from "Items intentionally OUT of v2.9 scope" (§2) re-introduced

---

**BEGIN PLAN DRAFT NOW.**