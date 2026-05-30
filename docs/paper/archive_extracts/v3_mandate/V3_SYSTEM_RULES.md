# V3 System Rules

**Version:** 3.0.0-alpha
**Principle:** Binary gates only. If it doesn't pass, it doesn't ship.

---

## Definition of Done

A task is complete **if and only if** all three conditions hold:

1. **pytest gate:** `pytest tests/ -x -q` exits zero. Every test in
   `tests/` is executable, expected to pass, and is the authoritative
   specification for correctness.

2. **Identity gate:** The output ingestion JSONL validates against the
   canonical `IngestionChunk` Pydantic model without exception. The
   source-of-truth schema lives in `src/mmrag_v3/schema/`.

3. **Commit-ready:** The diff is self-contained — no commented-out code,
   no TODO markers, no placeholder strings. If code is present, it is
   either test-code (exercising a requirement) or production-code
   (satisfying a requirement).

---

## Binary Constraint Model

Every rule in this document is a **strict constraint**. There are no
"shoulds," "preferences," or "unless convenient" clauses. A violation
is a build failure.

- **UIR-in, Chunk-out:** Every engine MUST return `UniversalDocument`.
  Every chunker MUST accept only `UniversalDocument`. These contracts
  are enforced by `tests/test_v3_security.py`.

- **No backward shims:** No `v2x_to_v3_mapper`, no bridge adapters,
  no compatibility layers. V3 types replace V2 types. Migration is
  one-direction.

- **Processor is Docling-agnostic:** `src/mmrag_v3/processor.py`
  imports zero Docling symbols. AST audit enforces this.

- **Schema is canonical:** `IngestionChunk` from
  `src/mmrag_v3/schema/` is the single source of truth. No shadow
  type definitions in other modules.

---

## What Exists

Only files tracked in this repository exist. If it is not in:

- `docs/v3_mandate/V3_SYSTEM_RULES.md` (this file)
- `docs/v3_mandate/V3_ARCHITECTURE.md`
- `docs/v3_mandate/V3_PROJECT_STATUS.md`

...then it does not exist for the purpose of any engineering decision.
The legacy archive at `.legacy_archive/` is out-of-scope. Do not
reference it. Do not import from it. Do not read it to resolve
ambiguities — the V3 mandate is self-contained.

---

## Test Authority

`tests/test_v3_security.py` is the architecture enforcement layer.
If a change causes any test in that file to fail, the change is
invalid. The tests define the contract; the code satisfies it.
There is no negotiation.