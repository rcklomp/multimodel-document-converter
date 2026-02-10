# 🤖 AGENTS.md: MMRAG V2.4.1 Operational Protocol (Aligned, V2.4.2 learnings without classifier swap)

**Target Agent:** Codex / Senior Python ETL Architect  
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
- AGENT_GUIDANCE_MANAGEMENT.md (format + organization)
- AGENT_GUIDANCE_EVOLUTION.md (when/how to add or promote guidance)

---

## 🛑 1. TECHNICAL INVARIANTS (Hard Constraints)
1. **Runtime Integrity:** Python **3.10** only. Avoid 3.11+ syntax/features.
2. **Hardware Bound:** Optimize for **Apple Silicon (MPS)**; prefer `mps` for torch when available.
3. **Library Lockdown:** `docling` **v2.66.0** pinned; do not bump without impact review.
4. **Resource Ceiling:** Target **≤8GB RAM** during runs; keep batch sizes ≤10 pages and call `gc.collect()` between batches.
5. **IRON-09 (Blind Test Validation):** A code change is only valid if the full `acceptance_suite` yields a `GATE_PASS`. This MUST include at least one "Blind Test" document (Greenhouse Design and Control by Pedro Ponce) that was not part of the dev-loop. Any pass based on hardcoded filenames or word-lists is a system failure.
6. **IRON-10:** Refinement logic must rely on a single `20-unit` vertical threshold. No profile-specific or heading-specific branches allowed.

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
- Tune chunk-size behavior per profile (`technical_manual`, `scanned_degraded`, `digital_magazine`, etc.).
- Treat chunk size as an empirical quality lever: changes require before/after acceptance metrics, not intuition.

---

## 🧬 3. CLASSIFICATION & UIR CONTRACT (V2.4.1 scope)
- **Do NOT enable the V2.4.2 DocumentClassifier**; stay on the V2.4.1 multi-dimensional profile classifier (e.g., `academic_whitepaper`, `digital_magazine`).
- Prefer manual profile overrides when certainty is high (planned flag: `--profile-override`).
- BBoxes must be normalized to **int [0,1000]** before emission.
- Shadow assets: promote to `IMAGE` if visual signal exists; otherwise drop before final JSONL.

---

## 💾 4. AGENT MEMORY & CONTEXT PROTOCOL
1. Start sessions by skimming `docs/` for current SRS/architecture notes (no `active_context.md` in this branch).
2. Cross-check nontrivial changes against `docs/ARCHITECTURE.md` for UIR compliance.
3. When finishing a task, record deltas and next steps in commit messages or a short note for the next engineer.

---

## 📍 5. CURRENT STATE & DIRECTIVES (Jan 24, 2026)
**Phase:** `v2.4.1-stable` with targeted hotfixes (no v2.4.2 classifier).  
**Recent Finding:** Significant token variance on AIOS PDF; recovery pipeline rescues missing text, but OCR guard disabled layout-aware OCR on digital PDFs.  
**Known debt:** `digital_magazine` → "Combat Aircraft - August 2025 UK" stabilizes around **-16% token variance** due to heavy text-in-graphics; treat as tolerated debt (see QA guidance below).
**Decision (known debt):** Do not add/maintain extra "text-in-graphics" complexity (digital-magazine layout-OCR / image-region OCR). Keep the digital PDF path simple and stable; only run OCR on digital-like PDFs when the user explicitly sets `--force-ocr`.
**QA policy update:** For `digital_magazine` only, QA tolerance is 18% (0.18). Do not raise tolerances for other profiles.

### Priority TODOs
1. Add **`--force-ocr`** override to bypass the digital-modality OCR guard.  
2. Expose **QA strictness knobs** (`--qa-tolerance`, `--qa-noise-allowance`, or preset `--strict-qa`).  
3. Fix **finalize chunk-count mismatch** (log vs. written JSONL).  
4. Optional: add **profile override** flag to sidestep classifier drift without pulling in v2.4.2 classifier.
5. Keep chunk sizing profile-driven and acceptance-tested (no universal hard min/max invariant).

---

## 📂 6. DIRECTORY AUTHORITY
- `src/mmrag_v2/` … core pipeline, validators, profile logic.
- `src/mmrag_v2/engines/` … format-specific extraction (Docling, etc.).
- `docs/` … SRS, architecture, audits (canonical references).

**END OF AGENTS.md**
