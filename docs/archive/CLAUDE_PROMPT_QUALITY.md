# Prompt for Claude Sonnet (Quality-First Architecture)

You are the lead architect for MM‑Converter‑V2. Your mission: **maximize conversion quality**. Architecture is only valuable if it raises output fidelity. If the current architecture deviates from SRS v2.4, you must justify each deviation with measurable quality gains, or revert it.

## Clean‑context briefing (read first)
- Goal: **Extremely high quality outputs** (asset fidelity, no missing assets, correct semantic context, QA within tolerance).
- Current architecture drifted from SRS v2.4 and quality regressed.
- You may change SRS only if it **improves** quality.

## Non‑negotiables (Quality First)
1. **No silent degradation.** If a critical asset path or page image is missing, fail fast and surface why (not just warn).
2. **Full‑page assets must be verified** (UI vs editorial) with VLM; false positives are unacceptable.
3. **Semantic context for assets must be complete** (prev+next text snippets), otherwise it’s data loss.
4. **Atomic writes for JSONL** to preserve integrity.
5. **If a change weakens quality, undo it** or replace it with a stronger safeguard.

## Evidence‑Based Failures (File/Line Anchors)
1) **IRON‑06 violated (must halt on missing image buffer)**  
`src/mmrag_v2/processor.py:594-616`  
Warns and falls back to `element.image` without padding when `page_images` missing.

2) **IRON‑08 violated (atomic writes not used)**  
`src/mmrag_v2/cli.py:894` and `src/mmrag_v2/cli.py:1166`  
Calls `process_to_jsonl()` instead of `process_to_jsonl_atomic()`.

3) **REQ‑MM‑05/06/07 removed (shadow extraction)**  
`src/mmrag_v2/batch_processor.py:386-395`  
Explicit comment: shadow extraction removed.

4) **IRON‑07 / REQ‑MM‑10/11 missing (Full‑Page Guard VLM verification)**  
`src/mmrag_v2/batch_processor.py:1816-1874` (`_apply_full_page_guard`)  
Only prefixes or filters, does **not** call VLM verification.  
`src/mmrag_v2/vision/vision_manager.py:1552+` (`verify_shadow_integrity`) exists but isn’t wired.

5) **Full‑page detection logic anchor**  
`src/mmrag_v2/batch_processor.py:1789-1814` (`_is_full_page_bbox`)

6) **Missing semantic_context for assets (no next_text)**  
`src/mmrag_v2/processor.py:1330-1345` (`create_image_chunk` call)

7) **Tables missing semantic_context**  
`src/mmrag_v2/schema/ingestion_schema.py:732-807` (`create_table_chunk`)

8) **REQ‑PDF‑04 mismatch (do_extract_images)**  
`src/mmrag_v2/processor.py:320-339` (PdfPipelineOptions)  
`src/mmrag_v2/engines/pdf_engine.py:164-177` (PdfPipelineOptions)

9) **REQ‑SENS drift in fallback strategy**  
`src/mmrag_v2/orchestration/strategy_orchestrator.py:225-233`

## Additional anchors (batch/vision paths)
- **Batch OCR classification threshold**  
`src/mmrag_v2/batch_processor.py:401-429` (`_classify_page`)
- **Full‑page guard output logging**  
`src/mmrag_v2/batch_processor.py:1871-1892`
- **VLM full‑page guard parsing**  
`src/mmrag_v2/vision/vision_manager.py:1659+` (`_parse_fullpage_guard_json`)

## Your Task
1. **Identify where architecture drifted from SRS and reduced quality.**
2. **Propose fixes that restore or exceed SRS quality.**
3. **Deliver a concrete patch plan**: files, functions, and logic changes—*minimal but high‑impact*.  
4. **Define acceptance criteria** with measurable checks (logs, QA gates, JSONL audits).

## Constraints
- Do **not** introduce new regressions or remove safeguards.
- Avoid “clean architecture” changes that don’t improve output.
- Keep diffs minimal and focused on quality.
- Provide a **no‑quality‑loss proof** for any change that could reduce fidelity, or reject the change.
- Do **not** change output schema, asset naming, or chunk IDs unless quality increases and you prove compatibility.
- Touch only the functions listed in the anchors unless strictly necessary.

## Acceptance Criteria (must all pass)
- No “Page image missing” warnings in normal PDF runs (or explicit failure if they occur).
- Full‑page assets are VLM‑verified or discarded (logged).
- `semantic_context.prev_text_snippet` and `next_text_snippet` are non‑null for assets when text exists.
- QA‑CHECK‑01 variance within tolerance for baseline PDFs.
- JSONL written atomically.
- Padding must be applied or processing fails (no silent fallback).
- Provide before/after metrics: token variance, missing assets count, full‑page guard decisions.

Deliver:
- Short root‑cause summary
- Fix plan (file/line, rationale)
- Risks and mitigations
- Test commands to validate quality improvements
