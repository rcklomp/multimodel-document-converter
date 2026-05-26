# Phase A Scope-Negotiation Log

**Charter:** [`ARCHITECTURE_V3_DRAFT_0.5.md`](ARCHITECTURE_V3_DRAFT_0.5.md)
"Phase A: UIR Foundation" — Scope-negotiation protocol; R22.

This log records any invocation of Phase A's scope-negotiation protocol.
The 24-day Phase A budget is a working assumption; if any of the negotiation
triggers fire, the budget is renegotiated (not silently extended) and the
decision is recorded here BEFORE Phase A continues.

## Triggers (Charter §Phase A)

1. **A0 exceeds 4 days** — per-doc spike on `ATZ_Elektronik_German`.
2. **First 5 days of A2 show <20% progress** on the ~12 `batch_processor.py`
   reconciliation paths.
3. **Semantic-identity gate's explained-delta half exceeds 5%** (more
   chunks differ than the audit table can enumerate).

## Negotiation options

Per Charter:

- **(a) Defer ~1/3 of `batch_processor.py` reconciliation paths to v3.0.1.**
- **(b) UIR-shim fallback** — convert `DoclingDocument` → `UniversalDocument`
  at the adapter boundary and keep downstream `processor.py` / `mapper.py` /
  `batch_processor.py` unchanged. UIR contract ships without the full
  downstream rewrite; the rewrite becomes a v3.0.2 cycle. ~10% performance
  overhead from translation layer in exchange for ~50% Phase A scope
  reduction. Preferred fallback because it preserves the C13/R15 `chunk_id`
  stability contract.
- **(c) Widen explained-delta tolerance to ≤10%** (with explicit
  `DECISIONS.md` entry documenting the loosened gate).
- **(d) Split A2 across cycles** (A2a in 3.0.0; A2b in 3.0.1).
- **Content-derived `chunk_id` as scope-negotiation option:** if A0 reveals
  >20% chunk rewrite-map churn, flip `chunk_id` derivation from positional
  to content-based (regret #4 in Charter §6.3).

## Entries

**EMPTY** at foundation-session start (2026-05-26). No negotiation triggers
have fired because Phase A code work has not begun.

Schema for entries (when they appear):

```
### YYYY-MM-DD — Trigger N fired

**Trigger:** [which of 1/2/3 above]
**Evidence:** [link to A0 report, A2 progress dashboard, or identity-gate
  output that justifies the trigger firing]
**Option chosen:** [one of a/b/c/d, or "continue with original budget +
  documented justification"]
**Rationale:** [why this option was preferred over the others]
**New budget:** [new day count for Phase A, if changed]
**Reviewer sign-off:** [user confirmation]
```
