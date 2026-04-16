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
- Tune chunk-size behavior per profile (`technical_manual`, `scanned_degraded`, `scanned`, `digital_magazine`, `academic_whitepaper`).
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
1. Start sessions by skimming `docs/` for current SRS/architecture notes (no `active_context.md` in this branch).
2. Cross-check nontrivial changes against `docs/ARCHITECTURE.md` for UIR compliance.
3. When finishing a task, record deltas and next steps in commit messages or a short note for the next engineer.

---

## 📍 5. CURRENT STATE & DIRECTIVES (April 2026)

**Version:** `v2.7.0` (schema version 2.7.0)  
**Phase:** Production acceptance — 17 of 31 documents AUDIT_PASS, 14 pending reconversion.

**Active architecture decisions:**
- PDF extraction pathway is determined by structural integrity pre-flight tests, not semantic profile. See `docs/DECISIONS.md` — "Structural Pathology over Semantic Profiling".
- `IngestionMetadata` record is written as the first JSONL line (v2.6+); QA scripts must skip it.
- VLM failure paths use differentiated sentinels (`[VLM_FAILED: response invalid]`, `[VLM_FAILED: call error]`, `[VLM_FAILED: parse error]`).
- **Image extraction** uses Docling layout model for all document types. PyMuPDF `page.get_images()` was tested but reverted (unreliable for magazines/academic papers). See `docs/DECISIONS.md` — "Image Extraction Routing".
- **Encoding corruption** uses heal-over strategy: keep HybridChunker for structure, force refiner on all chunks at `threshold=0.0`. See `docs/DECISIONS.md` — "Heal-Over for Encoding Corruption".
- **4 multimodal validation layers** (v2.7): CorruptionInterceptor, POS Boundary Logic, Vision-Gated Hierarchy, Content-Type Classification. See `docs/DECISIONS.md`.

**QA policy:** All profiles use the standard 10% token variance tolerance. See `docs/QUALITY_GATES.md`.

### Priority TODOs (Open)
1. Convert remaining 14 documents and achieve full AUDIT_PASS across all 31.
2. Re-ingest to Qdrant after all documents pass.
3. Establish per-category blind-test baselines for all document categories in the smoke test matrix.

### Recently Completed (Do Not Reopen)
1. `--force-ocr` override is implemented.
2. QA strictness knobs are implemented (`--qa-tolerance`, `--qa-noise-allowance`, `--strict-qa`).
3. `--profile-override` is implemented (debugging use only).
4. `IngestionMetadata` record implemented (v2.6).
5. Multi-profile smoke test + universal invariant checker implemented (`scripts/smoke_multiprofile.sh`, `scripts/qa_universal_invariants.py`).
6. `digital_magazine` 18% token variance waiver retired — IMAGE-bbox-aware source text extraction brings all magazines under 10%.
7. Docling upgrade 2.66.0 → 2.86.0 with picture classification.
8. TOC-based heading hierarchy (PDF bookmarks + content-based magazine TOC).
9. Output provenance (`pipeline_version`, `source_file_hash`, `config_hash`).
10. 4 multimodal validation layers replacing heuristic-loop patching.

---

## 📂 6. DIRECTORY AUTHORITY
- `src/mmrag_v2/` … core pipeline, validators, profile logic.
- `src/mmrag_v2/engines/` … format-specific extraction (Docling, etc.).
- `docs/` … SRS, architecture, audits (canonical references).

**END OF AGENTS.md**
