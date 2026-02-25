# Agent Brief

Use this file to orient LLM agents quickly. Read in the order below.
This is a reading/orientation order, not a trust hierarchy; the SRS remains the authoritative source of truth.

1) `docs/DECISIONS.md`
- Guardrails and anti-patterns (must-follow rules for agents).

2) `docs/TESTING.md`
- Canonical commands and verification steps.

3) `docs/QUALITY_GATES.md`
- QA checks and known warnings for pass/fail decisions.
- Includes current QA-CHECK-01 tolerance policy, including any temporary profile-specific waivers and the target return to 10%.

4) `docs/ARCHITECTURE.md`
- System design and data flow (reference when changing logic).

5) `docs/SRS_Multimodal_Ingestion_V2.4.md`
- Authoritative contract (rarely needed, but final source of truth).
