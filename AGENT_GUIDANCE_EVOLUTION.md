# Agent Guidance Evolution (Avoid Rule Bloat)

**Status:** guidance; code not yet implemented (v2.4.1-stable)

Guidance should improve outcomes without turning into bureaucracy.
This doc defines when to add, change, promote, or remove guidance.

---

## Improvement Triggers

Consider updating guidance when you see:
- recurring regressions (same failure mode repeating)
- repeated review feedback of the same type
- a stable pattern used in 3+ places
- new tooling used consistently
- performance/memory issues recurring in the pipeline
- architecture drift from Router → Engine → Processor

---

## The “No-Bloat Gate” (Required)

Before adding new guidance, answer:

### 1) Is this actually an invariant?
If violating it breaks runtime stability, architecture integrity, or resource limits:
✅ Put it into **AGENTS.md** (Level 0).
Do not create a separate doc just for it.

### 2) Is this preventing a recurring regression?
If yes:
✅ Add it as a **Level 1 guardrail**.
Must include: rationale + how to safely deviate.

### 3) Otherwise: keep it optional
✅ Document as a **Level 2 heuristic** or leave it as local docs/comments.

---

## Promote / Modify / Deprecate

### Promote to Level 0 (Invariant) only if:
- violations repeatedly break functionality, OR
- it is required by platform/runtime constraints, OR
- it preserves core architectural boundaries (UIR, modality separation, resource ceiling)

### Modify existing guidance when:
- better examples exist in the codebase
- edge cases were discovered
- implementation details changed
- guidance conflicts with AGENTS.md principles/invariants (AGENTS.md wins)

### Deprecate/remove when:
- pattern no longer exists in the codebase
- it causes agents to resist legitimate improvements
- it duplicates what AGENTS.md already states
- guidance text or examples diverge from the active code version (v2.4.1-stable) and cannot be reconciled quickly

---

## Quality Bar

- actionable and specific
- examples derived from real code patterns when possible
- minimal “MUST” statements
- consistent with AGENTS.md + docs/ARCHITECTURE.md

---

## Change Checklist (use for any new/modified guideline)
1) Owner: who maintains/enforces it  
2) Reason: regression or invariant it protects  
3) Scope: Level 0/1/2 and files it applies to  
4) Version tag: confirm alignment with current code (v2.4.1-stable)  
5) Rollback note: how to revert if it blocks progress
