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
   - `docs/PROGRESS_CHECKLIST.md`
   - `docs/README.md`
2. Use the three-layer documentation model:
   - Layer 0 contracts: this file, `CLAUDE.md`, `docs/AGENT_GOVERNANCE.md`, `docs/DECISIONS.md`, `docs/QUALITY_GATES.md`, `docs/ARCHITECTURE.md`, SRS.
   - Layer 1 current state: `docs/PROJECT_STATUS.md`, dated quality snapshots.
   - Layer 2 execution: `docs/PROGRESS_CHECKLIST.md`, `docs/TESTING.md`, plans, run logs, archive.
3. Cross-check nontrivial changes against `docs/ARCHITECTURE.md` for UIR compliance.
4. Before marking a task complete or expanding docs, apply `docs/AGENT_GOVERNANCE.md`.
5. When finishing a task, update `docs/PROGRESS_CHECKLIST.md`; update `docs/PROJECT_STATUS.md` if the next recommended step changed; create/update a dated quality snapshot if quality numbers changed.

---

## 📍 5. CURRENT STATE & DIRECTIVES (May 2026)

**Engine version:** `v2.8.0` (schema version `2.7.0` — de-aliased in v2.8; the chunk-shape contract is unchanged since v2.7).
**Phase:** v2.8.0 SHIPPED 2026-05-04 (annotated tag on `645ab2b`). v2.9 planning underway. Current canonical baseline: `docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md`.

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

### Priority TODOs (Open — v2.9 scope)
Source: `docs/PLAN_V2.9_DRAFT_PROMPT.md` §2 (priorities 1-7). Top items:
1. VLM enrichment of the `mmrag_v2_8` Qdrant collection (~5,500 image chunks have placeholder descriptions because v2.8 broad reconversion ran with `--vision-provider none`).
2. Refiner smart-routing fix (`cli.py:686` config-default ignores `has_encoding_corruption`).
3. ProfileClassifier rule 0c tightening (Ayeva mis-routed to `digital_literature`, suppresses CodeFormulaV2 → 0.83 vs 0.85 gate).
4. Firearms heading regression — but FIX MUST RESPECT `AGENT-SPATIAL-20`; either re-route via `profile_classifier.py` (preferred) or propose explicit AGENT-SPATIAL-20 amendment.
5. Within-file chunk_id collision fix (`_generate_chunk_id` should include position index; 427 dupes across the v2.8 corpus).
6. Local VLM comparison (Workstream A — direct dependency for #1).
7. Remote CodeFormulaV2 inference target (only if code-heavy reconversions become routine).
5. Re-ingest to Qdrant after all documents pass.
6. Establish per-category blind-test baselines for all document categories in the smoke test matrix.

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
