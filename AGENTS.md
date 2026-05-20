# 🤖 AGENTS.md: MMRAG V2 Operational Protocol

**Target Agent:** Claude Code / Senior Python ETL Architect  
**Project:** MMRAG V2 (Multimodal RAG Converter)  
**Philosophy:** Principle-Based Engineering over Rigid Scripting

---

## 🧭 0. HOW GUIDANCE WORKS (Principles-first, minimal constraints)

This project uses a 3-level guidance model to avoid “rule bloat”:

### Level 0 — Invariants (MUST)
Non-negotiable constraints. If violated, the project breaks, drifts architecturally, or becomes unstable.
**All Level 0 items live in this AGENTS.md** (single source of truth).

### Level 1 — Guardrails (SHOULD)
Strong defaults that prevent recurring regressions.
Deviation is allowed if you:
- document rationale + impact, and
- keep changes small and testable.

### Level 2 — Heuristics (MAY)
Suggestions and patterns. Always optional.
If a heuristic becomes critical (breakages recur), promote it through the evolution process.

Companion docs:
- `docs/DECISIONS.md` — records all architectural decisions and their rationale
- `docs/QUALITY_GATES.md` — pass/fail thresholds per profile

---

## 🛑 1. TECHNICAL INVARIANTS (Hard Constraints)
1. **Runtime Integrity:** Python **3.10** only. Avoid 3.11+ syntax/features.
2. **Hardware Bound:** Optimize for **Apple Silicon (MPS)**; prefer `mps` for torch when available.
3. **Library Lockdown:** `docling` must be exact-pinned in `pyproject.toml`; do not bump without impact review.
4. **Resource Ceiling:** Target **≤8GB RAM** during runs; keep batch sizes ≤10 pages and call `gc.collect()` between batches.
5. **AGENT-VAL-01 (Blind Test Validation):** A code change is only valid if the multi-profile smoke test (`smoke_multiprofile.sh`) yields `GATE_PASS` + `UNIVERSAL_PASS` across all document categories. At least one document per category must be a "blind test" document not used during the fix dev-loop. The technical-manual blind test document is `Greenhouse Design and Control by Pedro Ponce.pdf`. Any pass based on hardcoded filenames or word-lists is a system failure.
6. **AGENT-SPATIAL-20:** Refinement logic must rely on a single `20-unit` vertical threshold. No profile-specific or heading-specific branches allowed.
7. **AGENT-EVIDENCE-01:** No task/workstream may be marked complete unless its evidence is reproducible from tracked files or a tracked snapshot. Ignored `data/` and `output/` artifacts cannot be the sole evidence for completion. See `docs/AGENT_GOVERNANCE.md`.
8. **AGENT-STATUS-01:** Project status documents must use explicit status scope: `implemented`, `validated-cloud`, `validated-local`, `blocked`, or `complete`. Do not use "complete" when local validation, durable fixtures, or required comparisons are still pending. See `docs/AGENT_GOVERNANCE.md`.
9. **AGENT-DOCS-01:** Keep documentation minimal and indexed. Do not add new governance docs when an existing contract can be extended; obey the documentation budget in `docs/AGENT_GOVERNANCE.md`.
10. **AGENT-TEST-01 (Test Contract Integrity):** Negative tests, regression tests, and acceptance fixtures are executable requirements. Do not remove, loosen, rewrite, or reframe their core assertions to match the current implementation. If such a test fails, fix the implementation or stop and document why the requirement is wrong. Any expectation change requires explicit rationale and must make the contract clearer or stricter, not easier.

**Numbering Note:** SRS IRON IDs remain canonical. Agent-local constraints use `AGENT-*` IDs to avoid collisions.

---

## 🏗️ 2. CORE PRINCIPLES (Navigation)
**A. Unify through Representation (UIR)**  
- Map every extractor (PDF/HTML/EPUB) into `UniversalDocument` before OCR/VLM refinement.

**B. Respect Modality Boundaries (Source Sanctity)**  
- OCR handles text; VLMs describe visuals only. Use `VISUAL_ONLY_PROMPT`; forbid VLM text transcription.

**C. Identity through Content (DNA over Visuals)**  
- Classify by text evidence (keywords/regex/semantic markers), not by layout alone.

**D. Stateless Pipeline Orchestration**  
- Keep Router → Engine → Processor separation; avoid monolithic BatchProcessor logic.

**E. Visual Primacy (Magazine Doctrine)** 
- In the digital_magazine profile, visual layout data overrides the native PDF text layer.

**F. Recover through Shadow (Information Retrieval)**
- Any shadow asset is a potential text source; use extraction_method=shadow_ocr to prevent information loss.

**G. Chunking by Profile, Validated by Evidence**
- Do not enforce one global "optimal" chunk size.
- Tune chunk-size behavior per profile (`technical_manual`, `scanned_degraded`, `scanned`, `digital_magazine`, `digital_literature`, `academic_whitepaper`).
- Treat chunk size as an empirical quality lever: changes require before/after acceptance metrics, not intuition.

---

## 🧬 3. CLASSIFICATION & UIR CONTRACT

- Use the **`ProfileClassifier`** in `orchestration/profile_classifier.py` for all automatic routing. Do not replace it with the V2.4.2 `DocumentClassifier` approach (different architecture, not compatible).
- `--profile-override` is a debugging and diagnostic tool only. **Never use it in acceptance runs** — correct classification by the ProfileClassifier is the goal, not a workaround for it.
- Extraction pathway (OCR vs direct) is determined by **structural integrity flags** (`has_flat_text_corruption`, `has_encoding_corruption`) from `DocumentDiagnosticEngine`, not by profile type. See `docs/DECISIONS.md`.
- BBoxes must be normalized to **int [0,1000]** before emission.
- Shadow assets: promote to `IMAGE` if visual signal exists; otherwise drop before final JSONL.

---

## 💾 4. AGENT MEMORY & CONTEXT PROTOCOL
1. Start sessions with the indexed handoff path:
   - `docs/PROJECT_STATUS.md`
   - `docs/README.md`
   - `docs/QUALITY_SNAPSHOT_2026-05-20_v2.11_soak_qwen3.md` (current canonical baseline; v2.11 Phase 1 challenger soak — 10× lift on Recall@1/Relevance/Faithfulness)
   - `docs/QUALITY_SNAPSHOT_2026-05-16_v2.10_after.md` (v2.10 strict-gate baseline; unchanged in v2.11 — retrieval-side swap only)
   - `docs/QUALITY_SNAPSHOT_2026-05-16_v2.10_soak.md` (v2.10 soak; baseline for v2.11 Phase 1 delta column)
   - `docs/PLAN_V2.11.md` (v2.11 plan; Draft v1.0 — Phase 1 swap staged 2026-05-20, tag pending user push)
   - `docs/PLAN_V2.10.md` (v2.10 execution history — closed; tag `v2.10.0` on `db6527c` 2026-05-16)
2. Use the three-layer documentation model:
   - Layer 0 contracts: this file, `CLAUDE.md`, `docs/AGENT_GOVERNANCE.md`, `docs/DECISIONS.md`, `docs/QUALITY_GATES.md`, `docs/ARCHITECTURE.md`, SRS.
   - Layer 1 current state: `docs/PROJECT_STATUS.md`, dated quality snapshots.
   - Layer 2 execution: active/draft plan docs, `docs/TESTING.md`, run logs, archive.
3. Cross-check nontrivial changes against `docs/ARCHITECTURE.md` for UIR compliance.
4. Before marking a task complete or expanding docs, apply `docs/AGENT_GOVERNANCE.md`.
5. When finishing a task, update `docs/PROJECT_STATUS.md` (current state + recommended next step) and create/update a dated quality snapshot if quality numbers changed.

---

## 📍 5. CURRENT STATE & DIRECTIVES (May 2026)

**Engine version:** `v2.11.0` in `src/mmrag_v2/version.py` and `pyproject.toml`. Annotated tag `v2.11.0` is **staged but NOT pushed** by the autonomous run — user pushes/tags after live-stack re-verification. Predecessor tag: `v2.10.0` SHIPPED 2026-05-16 on commit `db6527c`. Schema version `2.7.0` — chunk-shape contract unchanged since v2.7.
**Phase:** `v2.11.0` is the v2.11 ship state: Phase 1 swapped the production text-retrieval embedder from Ollama `llava` (4096-dim multimodal) to Dashscope `text-embedding-v4` (1024-dim text-only) after the shootout delivered 10× lift on Recall@1, Relevance, and Faithfulness vs the v2.10 baseline. Production Qdrant collection is now `mmrag_v2_8__qwen3_dashscope` (30,588 points, status green). Legacy `mmrag_v2_8` retained through 2026-06-19 for 30-day rollback (both regression tests must stay green during the window). Phase 2 (validated-cloud CI workflow) and Phase 3 (carry-forward dispositions) also shipped. Format judge axis temporarily downgraded for v2.11.0 to ≥85% (89.8% actual; coverage-reveal of pre-existing OCR/scan format imperfections in scanned/form docs that the baseline's hub-collapse had hidden); v2.11.x recovery target ≥95%, v2.12 reverts to original ≥96%. Current canonical baseline: `docs/QUALITY_SNAPSHOT_2026-05-20_v2.11_soak_qwen3.md` (v2.11 Phase 1 challenger soak — production). Predecessors (kept for delta): `docs/QUALITY_SNAPSHOT_2026-05-16_v2.10_after.md` + `docs/QUALITY_SNAPSHOT_2026-05-16_v2.10_soak.md` (v2.10 ship state); earlier snapshots are historical.

**Active architecture decisions:**
- PDF extraction pathway is determined by structural integrity pre-flight tests, not semantic profile. See `docs/DECISIONS.md` — "Structural Pathology over Semantic Profiling".
- `IngestionMetadata` record is written as the first JSONL line (v2.6+); QA scripts must skip it.
- VLM failure paths use differentiated sentinels (`[VLM_FAILED: response invalid]`, `[VLM_FAILED: call error]`, `[VLM_FAILED: parse error]`).
- **Image extraction** uses Docling layout model for all document types. PyMuPDF `page.get_images()` was tested but reverted (unreliable for magazines/academic papers). See `docs/DECISIONS.md` — "Image Extraction Routing".
- **Encoding corruption** uses heal-over strategy: `CorruptionInterceptor` per-bbox OCR + quarantine of unrepairable chunks (Workstream C closed in v2.8). See `docs/DECISIONS.md` — "Heal-Over for Encoding Corruption".
- **4 multimodal validation layers** (v2.7): CorruptionInterceptor, POS Boundary Logic, Vision-Gated Hierarchy, Content-Type Classification. See `docs/DECISIONS.md`.
- **Adapter-invocation guard** (v2.8 Phase 2): `tests/test_pdf_conversion_plan.py::test_no_raw_converter_invocation_outside_adapter` blocks any `self._converter.convert(...)` outside the adapter; promotes the v2.7 §5 rule from construction-only to construction+invocation.
- **Form acceptance class** (v2.8 Phase 5a): scanned forms / invoices route to a `FORM_AUDIT_PASS` lane that skips prose-calibrated `micro_non_label_ratio`. See `docs/QUALITY_GATES.md` "Form / Invoice Acceptance Class". This is a first-class acceptance variant, NOT a waiver per `AGENT-VAL-01`.

**QA policy:** All profiles use the standard 10% token variance tolerance. See `docs/QUALITY_GATES.md`.

### Priority TODOs (Open — v2.11.x recovery + v2.12 cycle)
Source: `docs/PLAN_V2.11.md` (Draft v1.0). v2.11.0 swap is staged
locally (commits `18bfbf2` + `c2a461c`); user pushes/tags after live-
stack re-verification. The next cycle's open items:

1. ~~**Embedder shootout (v2.11 Phase 1).**~~ **Done 2026-05-20** —
   Dashscope `text-embedding-v4` swapped in; 10× lift on Recall@1.
2. ~~**Validated-cloud checkpoint (v2.11 Phase 2).**~~ **Done
   2026-05-17** — CI workflow at `.github/workflows/v2_11_validate.yml`
   (GitHub-hosted lint/quick + self-hosted full validate).
3. ~~**Carry-forward non-goal dispositions (v2.11 Phase 3).**~~ **Done
   2026-05-17** — five rows in `docs/DECISIONS.md` "v2.11 Carry-
   Forward Decisions"; 3a/3b/3e documented dispositions, 3c PAUSED
   for user signoff, 3d design recorded + implementation deferred to
   v2.12.
4. **v2.11.x Format recovery (NEW carry-forward).** Scanned/form
   chunk-content sanitization for `CarOK_voorraadtelling` 68.8%,
   `Earthship_Vol1` 71.9%, `IRJET_Modeling_of_Solar_PV` 71.9%.
   Acceptance: next soak Format ≥95% without regressing the other
   five axes. Effort: ~1-2 days for the three named docs.
5. **30-day rollback window (NEW carry-forward).** Legacy `mmrag_v2_8`
   collection + `test_retrieval_regression_v2_10.py` retained
   through 2026-06-19. Both regression tests must stay green during
   the window. Drop date: 2026-06-19 (or sooner with user signoff).
6. **v2.12 plan to be drafted (NEW).** Per Phase 1 outcome analysis,
   absolute retrieval quality is still mediocre (Recall@5 chunk 66.8%
   is "mediocre", not "good"). v2.12 should focus on closing this
   gap via reranker → hybrid retrieval → query rewriting → per-doc-
   class chunking, in that bang-for-buck order. Target: Recall@5 ≥
   85%, Recall@1 ≥ 55%, Faithfulness ≥ 70%.

Existing non-goals carried forward to v2.12+: 3a NuMarkdown-8B local
VLM (proposed replacement: Qwen3-VL-8B on Mac Mini at
`http://10.0.10.246:1234`), 3b remote CodeFormulaV2 (upstream-blocked
on Docling 2.87+), 3c broader UIR refactor (PAUSED for user signoff
on `ConversionPlan` parent-class carve-out scope), 3d HybridChunker
per-item token guard (design recorded; implementation deferred), 3e
magazine rendered-region-crop (defer with soak-data rationale — the
ceiling is the embedder, not chunk shape).

### Recently Completed (Do Not Reopen)
1. `--force-ocr` override is implemented.
2. QA strictness knobs are implemented (`--qa-tolerance`, `--qa-noise-allowance`, `--strict-qa`).
3. `--profile-override` is implemented (debugging use only).
4. `IngestionMetadata` record implemented (v2.6).
5. Multi-profile smoke test + universal invariant checker implemented (`scripts/smoke_multiprofile.sh`, `scripts/qa_universal_invariants.py`).
6. `digital_magazine` 18% token variance waiver retired — IMAGE-bbox-aware source text extraction brings all magazines under 10%.
7. Docling upgrade 2.66.0 → 2.86.0 with picture classification and code/formula enrichment options.
8. TOC-based heading hierarchy (PDF bookmarks + content-based magazine TOC).
9. Output provenance (`pipeline_version`, `source_file_hash`, `config_hash`).
10. 4 multimodal validation layers replacing heuristic-loop patching.

---

## 📂 6. DIRECTORY AUTHORITY
- `src/mmrag_v2/` … core pipeline, validators, profile logic.
- `src/mmrag_v2/engines/` … format-specific extraction (Docling, etc.).
- `docs/` … SRS, architecture, audits (canonical references).

**END OF AGENTS.md**
