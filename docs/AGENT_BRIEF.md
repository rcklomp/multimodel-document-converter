# Agent Brief

Use this file to orient LLM agents quickly. Read in the order below.
This is a reading/orientation order, not a trust hierarchy; the SRS remains the authoritative source of truth.

1) `docs/PROJECT_STATUS.md`
- Current state, model setup, quality baseline, and immediate next work.

2) `docs/PROGRESS_CHECKLIST.md`
- Durable task tracker for picking up work across chat sessions.

3) `AGENTS.md`
- Hard invariants and non-negotiable architecture constraints.

4) `docs/README.md`
- Documentation map and layer model.

5) `docs/DECISIONS.md`
- Guardrails and anti-patterns (must-follow rules for agents).

6) `docs/TESTING.md`
- Canonical commands and verification steps.

7) `docs/QUALITY_GATES.md`
- QA checks and known warnings for pass/fail decisions.
- Includes current QA-CHECK-01 tolerance policy (10% for all profiles, no waivers).

8) `docs/ARCHITECTURE.md`
- System design and data flow (reference when changing logic).

9) `docs/SRS_Multimodal_Ingestion_V2.5.md`
- Authoritative contract (rarely needed, but final source of truth).
- **Note:** The SRS is at v2.5 while the codebase is at v2.7.0. Features added in v2.6–v2.7 (multimodal validation layers, TOC-based heading hierarchy, Docling picture classification, encoding heal-over) are not yet reflected in the SRS. Use `CHANGELOG.md` and `docs/DECISIONS.md` for v2.7 context.
