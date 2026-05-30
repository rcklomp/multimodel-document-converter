# 🤖 AGENTS.md: MMRAG V2 Operational Protocol

**Target Agent:** Claude Code / Senior Python ETL Architect  
**Project:** MMRAG V2 (Multimodal RAG Converter)  
**Philosophy:** Principle-Based Engineering over Rigid Scripting

---

## 🧭 0. HOW GUIDANCE WORKS (Binary constraint model)

A rule is either a strict constraint or it is deleted. There is no
"soft" tier — every item in this document is non-negotiable.

### Strict Constraints (MUST)
If violated, the project breaks, drifts architecturally, or becomes unstable.
**All constraints live in this AGENTS.md** (single source of truth).

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
7. **AGENT-EVIDENCE-01:** No task/workstream may be marked complete unless `docs/V3_EXECUTION_MANDATE.md`'s programmatic gates pass. Ignored `data/` and `output/` artifacts cannot be the sole evidence for completion.
8. **AGENT-STATUS-01:** There is no "in-progress," "rebooked," or "implemented but not validated." A phase either passes the gates in `docs/V3_EXECUTION_MANDATE.md` or it has failed.
9. **AGENT-DOCS-01:** Keep documentation minimal and indexed. Do not add new governance docs. `docs/V3_EXECUTION_MANDATE.md` is the single governance file.
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
1. Start sessions with the indexed handoff path. The single source
   of truth for "what version is current and which baseline doc to
   read" is `CLAUDE.md`'s Read First list; this section lists only
   the always-load entry points:
   - `docs/PROJECT_STATUS.md` (current state including version,
     cycle status, and next-cycle plan pointer)
   - `docs/README.md` (doc index + read-order)
   - The current canonical baseline named in `CLAUDE.md`'s Read
     First list and in `PROJECT_STATUS.md`'s headline section.
     Prior-version baselines are quarantined in `docs/.archive/`
     and blocked by `.aiignore`; do not read or reference them.
2. Use the three-layer documentation model:
   - Layer 0 contracts: this file, `CLAUDE.md`, `docs/V3_EXECUTION_MANDATE.md`, `docs/DECISIONS.md`, `docs/QUALITY_GATES.md`, `docs/ARCHITECTURE.md`, `docs/ARCHITECTURE_V3_DRAFT_0.5.md`, SRS.
   - Layer 1 current state: `docs/PROJECT_STATUS.md`.
   - Layer 2 execution: active plan docs, `docs/TESTING.md`, run logs.
3. Cross-check nontrivial changes against `docs/ARCHITECTURE_V3_DRAFT_0.5.md` for V3 UIR compliance; `docs/ARCHITECTURE.md` is the v2.X production baseline being evolved.
4. Before marking a task complete or expanding docs, apply `docs/V3_EXECUTION_MANDATE.md`.
5. When finishing a task, update `docs/PROJECT_STATUS.md` (current state + recommended next step) and create/update a dated quality snapshot if quality numbers changed.

---

## 📍 5. CURRENT STATE & DIRECTIVES

**Engine + cycle state lives in `docs/PROJECT_STATUS.md`** —
single source of truth for current version, ship+push status,
cycle phase, production collections, and active engineering
direction. Read it at session start. The architecture decisions
below are stable contracts that survive cycle-to-cycle changes;
they are kept here because they govern future work regardless of
which v2.X is currently shipping.

**Active architecture decisions:**
- PDF extraction pathway is determined by structural integrity pre-flight tests, not semantic profile. See `docs/DECISIONS.md` — "Structural Pathology over Semantic Profiling".
- `IngestionMetadata` record is written as the first JSONL line (v2.6+); QA scripts must skip it.
- VLM failure paths use differentiated sentinels (`[VLM_FAILED: response invalid]`, `[VLM_FAILED: call error]`, `[VLM_FAILED: parse error]`).
- **Image extraction** uses Docling layout model for all document types *(legacy v2 path; the V3 default route is vision-native via `mmrag_v3.extract` — see V3 Phase C note below)*. PyMuPDF `page.get_images()` was tested but reverted (unreliable for magazines/academic papers). See `docs/DECISIONS.md` — "Image Extraction Routing".
- **Encoding corruption** uses heal-over strategy: `CorruptionInterceptor` per-bbox OCR + quarantine of unrepairable chunks (Workstream C closed in v2.8). See `docs/DECISIONS.md` — "Heal-Over for Encoding Corruption".
- **4 multimodal validation layers** (v2.7): CorruptionInterceptor, POS Boundary Logic, Vision-Gated Hierarchy, Content-Type Classification. See `docs/DECISIONS.md`.
- **Adapter-invocation guard** (v2.8 Phase 2 — **legacy v2 path; test currently deferred `V3_DEFERRED`**): `tests/test_pdf_conversion_plan.py::test_no_raw_converter_invocation_outside_adapter` blocked any `self._converter.convert(...)` outside the adapter. It is `@pytest.mark.skip`-ped because the V3 `BatchProcessor.process_pdf` path delegates extraction to `mmrag_v3.extract()` and constructs no converter; the V3 Docling boundary is `src/mmrag_v3/engines/docling_fast.py`, firewalled by `tests/test_v3_security.py` (V3 Phase C note below).
- **Form acceptance class** (v2.8 Phase 5a): scanned forms / invoices route to a `FORM_AUDIT_PASS` lane that skips prose-calibrated `micro_non_label_ratio`. See `docs/QUALITY_GATES.md` "Form / Invoice Acceptance Class". This is a first-class acceptance variant, NOT a waiver per `AGENT-VAL-01`.
- **V3 Phase C — vision-native extraction** (2026-05-29): Phase C engines live under `src/mmrag_v3/engines/`. The default route is `HybridEngine` (per-page pre-flight: tables, images, or `> VLM_DRAWINGS_THRESHOLD` drawings → VLM; else fast Docling). Single-page VLM failures fall back to Docling automatically. The VLM is not trusted for coordinate normalization or page-number assignment — the adapter projects bboxes to `[0,1000]` and stamps page numbers from its own index. All requests are capped at `max_completion_tokens=4096`. Default provider is OpenRouter (`qwen/qwen3-vl-8b-instruct`); override via `VLM_NATIVE_ENDPOINT` / `VLM_NATIVE_MODEL` / `VLM_NATIVE_API_KEY`. Engine-file imports are firewalled by `tests/test_v3_security.py` (vision/glue files banned from docling + v2 legacy; the docling-boundary file `docling_fast.py` may import docling but not v2 legacy). See `docs/DECISIONS.md` — "v3.0 Phase C — Vision-Native Extraction".

**QA policy:** All profiles use the standard 10% token variance tolerance. See `docs/QUALITY_GATES.md`.

### Open items + next-cycle plan

Per-cycle priority TODOs live in `docs/PROJECT_STATUS.md` under
"Other Carry-Forwards" and in the latest `docs/PLAN_V2.X.md`
disposition sections — they shift every cycle and are not stable
contracts. Don't duplicate that state here; it goes stale by next
cycle-open.

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
- `src/mmrag_v3/` … V3 Phase C vision-native namespace (`engines/vlm_native.py`, `engines/vlm_provider.py`, `engines/docling_fast.py`, `engines/router.py`, `processor.py`). UIR contract types are imported from `mmrag_v2.universal.intermediate`.
- `v3_execution_root/src/mmrag_v3/` … V3 Phase A sandbox (chunker, schema, sanitization, scripts). Separate `mmrag_v3` namespace; Phase C engine is loaded into the Identity-Gate subprocess by absolute file path under a private package alias.
- `docs/` … SRS, architecture, audits (canonical references).

**END OF AGENTS.md**
