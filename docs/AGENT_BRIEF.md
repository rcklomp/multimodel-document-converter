# Agent Brief

Use this file to orient LLM agents quickly. Read in the order below.
This is a reading/orientation order, not a trust hierarchy; the SRS remains the authoritative source of truth.

1) `docs/DECISIONS.md`
- Guardrails and anti-patterns (must-follow rules for agents).

2) `docs/TESTING.md`
- Canonical commands and verification steps.

3) `docs/QUALITY_GATES.md`
- QA checks and known warnings for pass/fail decisions.
- Includes current QA-CHECK-01 tolerance policy (10% for all profiles, no waivers).

4) `docs/ARCHITECTURE.md`
- System design and data flow (reference when changing logic).

5) `docs/SRS_Multimodal_Ingestion_V2.5.md`
- Authoritative contract (rarely needed, but final source of truth).
- **Note:** The SRS is at v2.5 while the codebase is at v2.7.0. Features added in v2.6–v2.7 (multimodal validation layers, TOC-based heading hierarchy, Docling picture classification, encoding heal-over) are not yet reflected in the SRS. Use `CHANGELOG.md` and `docs/DECISIONS.md` for v2.7 context.
