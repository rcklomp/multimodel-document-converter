# Agent Brief

Use this file to orient LLM agents quickly. Read in the order below.
This is a reading/orientation order, not a trust hierarchy. Current authority is:
`AGENTS.md` for invariants, `docs/DECISIONS.md` for accepted architecture decisions,
and `docs/ARCHITECTURE.md` for target data flow. The SRS is stale v2.5 reference
material only until it is rewritten.

1) `docs/PROJECT_STATUS.md`
- Current state, model setup, quality baseline, and immediate next work.

2) `docs/PROGRESS_CHECKLIST.md`
- Durable task tracker for picking up work across chat sessions.

3) `AGENTS.md`
- Hard invariants and non-negotiable architecture constraints.

4) `docs/README.md`
- Documentation map and layer model.

5) `docs/AGENT_GOVERNANCE.md`
- Rules for durable evidence, completion status, self-review, and documentation budget.

6) `docs/DECISIONS.md`
- Guardrails and anti-patterns (must-follow rules for agents).

7) `docs/TESTING.md`
- Canonical commands and verification steps.

8) `docs/QUALITY_GATES.md`
- QA checks and known warnings for pass/fail decisions.
- Includes current QA-CHECK-01 tolerance policy (10% for all profiles, no waivers).

9) `docs/ARCHITECTURE.md`
- System design and data flow (reference when changing logic).

10) `docs/SRS_Multimodal_Ingestion_V2.5.md`
- Historical contract reference only.
- **Note:** The SRS is at v2.5 while the codebase is at **v2.8.0** (engine; schema stays 2.7.0). Features added in v2.6–v2.8 (multimodal validation layers, TOC-based heading hierarchy, Docling 2.86.0 picture/code-enrichment options, encoding heal-over, output provenance, v2.8 form acceptance class, v2.8 adapter-invocation static guard, v2.8 keyword-aware C0/DEL replacement, v2.8 CodeFormulaV2 enable lane) are not reflected in the SRS. Use `AGENTS.md`, `docs/DECISIONS.md`, and `docs/ARCHITECTURE.md` for current behavior.
