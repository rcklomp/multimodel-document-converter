# Quality Snapshot 2026-05-03

**Purpose:** Lock the as-is BEFORE state for v2.8 (`docs/PLAN_V2.8_PRODUCTION_GAPS.md`) so deltas from Phases 1-4 are measurable. This is the Phase 0 deliverable.

**Predecessor:** `docs/QUALITY_SNAPSHOT_2026-05-01.md` (Milestone 1 + 2 closure, contextual retrieval).

**Successor:** `docs/QUALITY_SNAPSHOT_<post-v2.8-reconversion>.md` (Phase 5 will produce this).

---

## 1. Recent shipped work (2026-05-03, baked into this baseline)

These commits are the implicit "current" code state against which v2.8 deltas are measured:

| Commit | Subject |
|---|---|
| `3bdbe0f` | feat: post-Docling sanity pass + `digital_literature` profile |
| `2f51816` | feat(diagnostic): rule 0c — moderate-length dialogue-rich docs → literature |
| `379a733` | docs+chore: post-Docling close-out, acceptance fixture, smoke matrix |

These shipped after most existing `output/*/ingestion.jsonl` files were generated (see "Output age" notes in §3 below) — re-running conversions may already close some flagged failures without further code changes (see Phase 1 step 0 in the plan).

---

## 2. Smoke matrix (BEFORE)

Source: `/tmp/smoke_post_dl_v2_20260503/_summary.txt` (preserved as `docs/quality_snapshots/2026-05-03/smoke_summary.txt`).

Run command: `bash scripts/smoke_multiprofile.sh` (PAGES=10, BATCH_SIZE=3).

| Category | Document | Detected profile | GATE | UNIVERSAL |
|---|---|---|---|---|
| academic_journal | AIOS LLM Agent Operating System | academic_whitepaper | GATE_PASS | UNIVERSAL_PASS |
| academic_journal | A_comprehensive_review_on_hybrid_electri | academic_whitepaper | GATE_PASS | UNIVERSAL_PASS |
| business_form | betwistingsformulier_aankoop_niet_ontvangen | academic_whitepaper | GATE_PASS | UNIVERSAL_PASS |
| data_spreadsheet | CarOK voorraadtelling 2021-04 | technical_manual | GATE_PASS | UNIVERSAL_PASS |
| digital_magazine | PCWorld_July_2025_USA | digital_magazine | GATE_PASS | UNIVERSAL_PASS |
| digital_literature | HarryPotter_and_the_Sorcerers_Stone | digital_literature | GATE_PASS | UNIVERSAL_PASS |
| scanned | 0013_140302111325_001 | scanned | **GATE_FAIL: micro_non_label_ratio=0.294 (>0.22)** | UNIVERSAL_PASS |
| technical_manual | Firearms | technical_manual | GATE_PASS | UNIVERSAL_PASS |
| technical_manual | Greenhouse Design and Control by Pedro Ponce | technical_manual | GATE_PASS | UNIVERSAL_PASS |
| technical_manual | Python Distilled David M. Beazley 2022 | technical_manual | GATE_PASS | UNIVERSAL_PASS |
| technical_report | ATZ.Elektronik...PDFWriters | academic_whitepaper | GATE_PASS | UNIVERSAL_PASS |

**10/11 GATE_PASS, 11/11 UNIVERSAL_PASS.** The single failing row (SCAN0013, a small business form) is the row Phase 5 pre-flight option (a)/(b) addresses — per CLAUDE.md "no waivers" / AGENT-VAL-01 it cannot ship as a documented exception.

---

## 3. Per-document audit (BEFORE)

Source: `python scripts/qa_conversion_audit.py output/*/ingestion.jsonl` (full report preserved as `docs/quality_snapshots/2026-05-03/audit_all_outputs.txt`). Verdict aggregate: **30/37 PASS, 7 FAIL.**

The table below collapses canonical per-document outputs (the row v2.8 Phase 5 will measure against) and notes auxiliary probe/exploration runs separately.

### 3a. Canonical per-document outputs

| Output dir | Source PDF | Date | Audit | Failing checks (BEFORE) | v2.8 phase |
|---|---|---|---|---|---|
| AIOS_LLM_Agent_Operating_System | AIOS LLM Agent Operating System.pdf | recent | PASS | — | — |
| ATZ_Elektronik_German | ATZ.Elektronik...PDFWriters.pdf | recent | PASS | — | — |
| **A_comprehensive_review_on_hybrid_electri** | A_comprehensive_review_on_hybrid_electri.pdf | 2026-04-12 | **FAIL (TEXT)** | `ctrl_char_chunks: 1 (total: 4)` — 0x01 keyword separator | **Phase 1** (output predates `_ctrl_table` ship date 2026-05-03) |
| Adedeji_GenAI_Google_Cloud | Adedeji A. GenAI on Google Cloud...2026.pdf | recent | PASS | — | — |
| Ayeva_Python_Patterns | Ayeva K. Mastering Python Design Patterns...2024.pdf | 2026-04-12 | **FAIL (TEXT, CODE)** | `infix_artifacts: 2`, `indentation_fidelity: 0.22` | superseded by `ayeva_qa_20260501` (PASS); fold canonical into Phase 5 reconversion |
| Bourne_RAG_2024 | Bourne K. Unlocking Data with Generative AI and RAG 2024.pdf | recent | PASS | — | — |
| **Chaubal_PyTorch_Projects** | Chaubal S. AI Projects in PyTorch...2025.pdf | 2026-04-12 | **FAIL (CODE)** | `indentation_fidelity: 0.54`, code_chunks=182 | **Phase 4** (CodeFormulaV2 enrichment) |
| **Combat_Aircraft_August_2025** | Combat Aircraft - August 2025 UK.pdf | recent | **FAIL (TEXT)** | `encoding_artifacts: 48` | **Phase 3** |
| **Combat_Aircraft_full_promptfix_v2** | Combat Aircraft - August 2025 UK.pdf | 2026-04-29 | **FAIL (TEXT)** | `encoding_artifacts: 48`, `high_corruption: 79` | **Phase 3** (this is the run referenced explicitly in `PLAN_V2.8` §1) |
| Cronin_GenAI_Models | Cronin I. Building and Training Generative AI Models...2026.pdf | recent | PASS | — | — |
| Devlin_LLM_Agents | Devlin M. Building LLM Agents...2025.pdf | recent | PASS | — | — |
| Earthship_Vol1 | Earthship_Vol1...pdf | recent | PASS | — | — |
| Firearms | Firearms.pdf | recent | PASS | — | — |
| Fluent_Python_full_codex | Fluent Python Luciano Ramalho 2015.pdf | recent | PASS | — | Phase 4 non-regression control |
| Fluent_Python_full_vlm_codex | Fluent Python Luciano Ramalho 2015.pdf | recent | PASS | — | Phase 4 non-regression control |
| HarryPotter_and_the_Sorcerers_Stone | HarryPotter_and_the_Sorcerers_Stone.pdf | recent | PASS | — | `digital_literature` profile, post-Docling sanity-pass acceptance fixture |
| IRJET_Modeling_of_Solar_PV | IRJET_Modeling_of_Solar_PV_system_under.pdf | recent | PASS | — | — |
| Kimothi_RAG_Guide | A Simple Guide to Retrieval Augmented Generation Kimothi A. 2025.pdf | recent | PASS | — | — |
| PCWorld_July_2025 | PCWorld_July_2025_USA.pdf | recent | PASS | — | — |
| Recent_Trends_in_Transportation | Recent_Trends_in_Transportation_Technolo.pdf | recent | PASS | — | — |
| ayeva_qa_20260501 | Ayeva K. Mastering Python Design Patterns...2024.pdf | 2026-05-01 | PASS | — | latest QA run; CODE [code_heavy] `indentation_fidelity=0.93` |

### 3b. Auxiliary / probe outputs (informational only)

| Output dir | Audit | Note |
|---|---|---|
| Combat_Aircraft_full_vlm_codex | FAIL (TEXT) | legacy VLM-enabled run; same Workstream C corruption pattern |
| Combat_Aircraft_promptfix_pages91_100 | PASS | partial-page probe |
| PCWorld_promptfix_pages1_20 | PASS | partial-page probe |
| carok_fix | PASS | regression probe |
| carok_verify | PASS | regression probe |
| frontmatter_acceptance_harry_probe | PASS | front-matter probe |
| probe_ayeva_p1_20 | PASS | partial-page probe |
| probe_boundary_closeout_rag_guide | PASS | Milestone 2 probe |
| probe_carok | **FAIL (HEADING)** | **0 chunks** — bad probe, exclude from Phase 5 |
| probe_carok_codex_verify | PASS | regression probe |
| probe_combat | PASS | partial-page probe |
| probe_contextual_retrieval_rag_guide | PASS | contextual retrieval probe |
| probe_harry_potter | PASS | digital_literature probe |
| probe_milestone2_rag_guide | PASS | Milestone 2 probe |
| probe_rag_guide | PASS | RAG Guide probe |
| probe_rag_guide_guard | PASS | chunker-guard probe |

---

## 4. v2.8 phase targets — concrete BEFORE numbers

The plan's per-phase acceptance criteria are calibrated against these BEFORE numbers:

**Phase 1 — `A_comprehensive_review_on_hybrid_electri`:**
- BEFORE: `ctrl_char_chunks: 1 (total: 4)` (one chunk holds 4 × 0x01 separators).
- TARGET: `ctrl_chunks: 0`.
- Output is dated 2026-04-12, BEFORE `_ctrl_table` shipped in commit `3bdbe0f` (2026-05-03). Phase 1 step 0 is to re-run the conversion and verify whether any code change is still needed.

**Phase 3 — `Combat_Aircraft_full_promptfix_v2`:**
- BEFORE: `encoding_artifacts: 48`, `high_corruption: 79` across 537 chunks.
- TARGET: `encoding_artifacts: 0`, `high_corruption: 0`, `placeholder_ratio: 0%`.

**Phase 4 — `Chaubal_PyTorch_Projects`:**
- BEFORE: `indentation_fidelity: 0.54` across 182 code chunks (`code_heavy` content type).
- TARGET: `indentation_fidelity >= 0.85`.
- Non-regression control: `Fluent_Python_full_codex` (currently PASS).

**Phase 5 — Smoke matrix:**
- BEFORE: 10/11 GATE_PASS, 11/11 UNIVERSAL_PASS (SCAN0013 is the lone GATE_FAIL).
- TARGET (per CLAUDE.md / AGENT-VAL-01): 11/11 GATE_PASS + 11/11 UNIVERSAL_PASS, **no waivers**. Plan §5 pre-flight option (a)/(b) determines whether SCAN0013 is replaced or the audit gate is made form-aware.

---

## 5. Documents in `data/` without a current canonical output

Phase 5 broad reconversion will produce outputs for these (not yet present in `output/<canonical_name>/`):

- `data/academic_journal/Hybrid_electric_vehicles_and_their_challenges.pdf` (note: an `output/Hybrid_electric_vehicles/` exists but was not in this audit run)
- `data/business_form/0013_140302111325_001.pdf` (only `/tmp/smoke_*` runs)
- `data/business_form/betwistingsformulier_aankoop_niet_ontvangen.pdf`
- `data/data_spreadsheet/CarOK voorraadtelling 2021-04.pdf` (only `carok_*` probes)
- `data/technical_manual/Greenhouse Design and Control by Pedro Ponce.pdf` (only smoke runs)
- `data/technical_manual/Hao B. Machine Learning Platform Engineering...2026.pdf`
- `data/technical_manual/Jungjun H. Build an AI Agent...MEAP 2026.pdf`
- `data/technical_manual/Nagasubramanian D. Agentic AI for Engineers...2026.pdf`
- `data/technical_manual/Programming ArcGIS with Python Cookbook.pdf`
- `data/technical_manual/Python Cookbook  Everyone can cook delicious recipes with Python.pdf`
- `data/technical_manual/Python Distilled David M. Beazley 2022.pdf` (only smoke runs)
- `data/technical_manual/Raieli S. Building AI Agents with LLMs, RAG, and Knowledge Graphs...2025.pdf`
- `data/technical_manual/Sekar S. The MCP Standard...2026.pdf`
- `data/technical_manual/integra_u_en.pdf`

Phase 5 acceptance requires AUDIT_PASS (or a documented exception per `docs/AGENT_GOVERNANCE.md`) for every entry above.

---

## 6. How to reproduce this baseline

```bash
# Per-doc audit
conda run -n mmrag-v2 python scripts/qa_conversion_audit.py $(find output -maxdepth 2 -name "ingestion.jsonl" | sort)

# Smoke matrix
conda run -n mmrag-v2 bash scripts/smoke_multiprofile.sh
```

Preserved evidence files:
- `docs/quality_snapshots/2026-05-03/audit_all_outputs.txt` (37-output audit, full reports)
- `docs/quality_snapshots/2026-05-03/smoke_summary.txt` (11-row cross-profile smoke)
