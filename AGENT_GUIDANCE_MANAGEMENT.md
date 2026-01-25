# Agent Guidance Management Guide (Agent-Agnostic)

**Status:** guidance; code not yet implemented (v2.4.1-stable)

This repository is **principles-first**. Guidance exists to:
- enforce a small set of **non-negotiable invariants**
- prevent **repeated regressions**
- document **stable patterns** that improve consistency

If guidance would block improvement or experimentation, it should be a **heuristic** (optional), not a hard constraint.

---

## Guidance Levels (Balance Model)

### Level 0 — Invariants (MUST)
- breaks runtime, architecture integrity, or resource constraints if violated
- short, explicit, and testable
- **canonical source: AGENTS.md**

### Level 1 — Guardrails (SHOULD)
- prevents recurring failure modes
- deviation allowed with documented rationale + impact

### Level 2 — Heuristics (MAY)
- suggestions and patterns
- always optional; include an “escape hatch”

---

## Guidance Document Format

Guidance docs should be readable by humans **and** usable by any agent system.

### Recommended structure

````markdown
# Title

## Intent
What this guidance is for and why it exists.

## Do
- actionable instructions

## Don’t
- anti-patterns to avoid

## Examples (when helpful)
✅ Good
```python
def process_item(item: dict) -> dict:
    return {"id": item["id"], "name": item["name"]}
```

❌ Bad
```python
def process_item(id, name):
    return {"id": id, "name": name}
```
````

### Optional metadata (only if your agent runner supports it)
Some agent runners support front-matter metadata (glob targeting, always-apply flags).
If yours doesn’t, treat metadata as comments and ignore.

---

## Organization

Keep guidance in one dedicated location in the repo to avoid scattering and drift.

Recommended:
```yaml
PROJECT_ROOT/agent-guidance/
AGENTS.md
AGENT_GUIDANCE_MANAGEMENT.md
AGENT_GUIDANCE_EVOLUTION.md
(optional) patterns/
(optional) checklists/
```

If your chosen agent runner requires a different folder, keep a single “source of truth” folder and mirror/symlink from there.

### Branch vs Patch (AGENTS.md updates)
- Patch in place when change is version-aligned (v2.4.1-stable) and small/urgent.
- Branch when change depends on future code, spans multiple files, or alters invariants.
- Always tag AGENTS.md with the active version string.

### Logging deviations
- Record any intentional deviations in `docs/active_context.md` (what, why, owner, expiry).

### Pre-merge diff procedure
- Run a focused diff on AGENTS.md and guidance files before merge.
- Verify Level labels and version tags match the active code.
- If drift is found, update guidance or block the merge.

---

## Content Guidelines (Principles-First)

1. Prefer **principles + invariants** over checklists.
2. Add “MUST” items **only** when:
   - they already exist as invariants in AGENTS.md, or
   - they prevent a recurring breakage that has happened multiple times.
3. Include examples only when ambiguity is common.
4. When guidance touches architecture, link back to canonical docs:
   - AGENTS.md
   - docs/ARCHITECTURE.md

---

## Best Practices Checklist

- [ ] Is this guidance Level 0/1/2 explicitly labeled?
- [ ] If Level 0: is it in AGENTS.md instead?
- [ ] Is it short, testable, and project-specific?
- [ ] Does it include an escape hatch when not Level 0?
- [ ] Does it prevent a real, recurring problem?

---

## Change Checklist (use for any new/modified guideline)
1) Owner: who maintains/enforces it  
2) Reason: regression or invariant it protects  
3) Scope: Level 0/1/2 and files it applies to  
4) Version tag: confirm alignment with current code (v2.4.1-stable)  
5) Rollback note: how to revert if it blocks progress
