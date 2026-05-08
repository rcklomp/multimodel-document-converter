# Agent Governance

Purpose: keep agents honest, reproducible, and concise. This file is the single contract for evidence, completion, review, and documentation growth.

## Status Vocabulary

Use only these statuses in current-state docs:

| Status | Meaning |
|---|---|
| `not-started` | No implementation work has begun. |
| `in-progress` | Investigation or implementation is underway. |
| `implemented` | Code/docs changed, validation incomplete. |
| `validated-cloud` | Validated against a cloud/remote provider only. |
| `validated-local` | Validated against the expected local runtime/provider. |
| `blocked` | Waiting on missing service, data, decision, or access. |
| `complete` | All acceptance signals met with durable evidence. |

Do not use vague substitutes like "done", "fixed", "good", or "validated" without scope.

## Evidence Rules

Evidence classes:

- `tracked`: committed or commit-ready files.
- `snapshot`: summarized in a tracked dated snapshot.
- `local-run`: generated results under ignored `output/`, with command and input recorded.
- `reproducible-fixture`: tracked manifest/config whose assets are regenerated from source into `output/`.
- `local-transient`: local services, untracked scratch files, or chat-only evidence.
- `external`: remote API/model/service result.
- `manual`: human visual/semantic judgment with sample IDs.

Completion may rely on `tracked`, `snapshot`, `local-run`, or `reproducible-fixture` evidence when commands, inputs, outputs, and limitations are recorded. `local-transient` evidence can support investigation only.

Hard rules:

1. `output/` is the correct place for generated conversion results and should remain ignored by Git.
2. A claim based on `output/` must record the command, input file, output path, and date.
3. Ignored `data/` scratch files and chat-only evidence cannot be the sole evidence for `complete`.
4. Blind-test manifests/configs used for acceptance must be tracked, preferably under `tests/fixtures/`; large generated assets may be regenerated into `output/`.
5. Quality metrics must separate raw-clean, sanitized, hard-fallback, placeholder, error, and final-invalid.
6. A script must not report a category it cannot actually classify.
7. If a required local service is unavailable, status is `blocked`, `validated-cloud`, or `local-pending`, never `complete`.
8. Standard smoke tests do not prove qualitative VLM/image quality unless they directly measure that claim.

Minimum evidence record:

```md
Evidence:
- Class:
- Command:
- Input:
- Output:
- Result:
- Tracked: yes/no
- Limitations:
```

## Completion Rules

A workstream may be marked `complete` only when:

1. Every listed acceptance signal is satisfied.
2. Evidence is durable (`tracked` or `snapshot`).
3. Known limitations are documented.
4. Required local/cloud comparisons are completed or explicitly removed from scope.
5. `PROJECT_STATUS.md` and snapshots agree (per-task history lives in `docs/archive/PROGRESS_CHECKLIST.md`; current task state belongs in `PROJECT_STATUS.md`).
6. A fresh coding session can reproduce the claim without chat history.

Never mark complete when evidence is ignored, provider-specific but generalized, local validation is pending, or a qualitative task was checked only with generic tests.

## Test Contract Rules

Tests are executable requirements, not implementation suggestions.

Hard rules:

1. Do not remove or weaken a negative/regression assertion to make current code pass.
2. Do not rewrite fixtures to avoid the behavior a test is supposed to catch.
3. If a test fails because the requirement is too strict or wrong, stop and document the proposed requirement change before editing the test.
4. Test expectation changes must make the contract clearer or stricter unless an explicit governance/decision doc explains why the requirement changed.
5. When reviewing test edits, ask: did this preserve the failure this test was meant to catch?

## Review Protocol

Before finalizing nontrivial work, check:

1. Did I use ignored files as evidence?
2. Did I overstate status beyond the evidence?
3. Did I rely on one model/document/page range while claiming generality?
4. Did I add document-specific or filename-specific logic?
5. Did I run the right validation for the actual claim?
6. Did I leave required fixtures untracked?
7. Did I weaken any negative/regression test instead of fixing the implementation?
8. Can the next agent reproduce this from tracked docs and commands?

## Documentation Budget

Keep docs useful by limiting growth:

- `docs/PROJECT_STATUS.md`: target <=150 lines.
- `docs/AGENT_GOVERNANCE.md`: target <=140 lines.
- `docs/archive/PROGRESS_CHECKLIST.md` is the historical task log; do not grow it (archived 2026-05-07).
- Dated snapshots: create only when quality state changes materially.
- Archive stale plans instead of duplicating them.

When adding more than 50 lines of documentation, remove, consolidate, or archive stale detail unless the new content is a durable contract. Current state and task state both belong in `PROJECT_STATUS.md`; evidence belongs in dated snapshots; per-task history lives in `docs/archive/PROGRESS_CHECKLIST.md`.

## Canonicality Rule (added 2026-05-04)

When a metric (e.g. `indentation_fidelity`, `encoding_artifacts`,
`ctrl_chunks`) appears in BOTH a layer-1 status doc (`PROJECT_STATUS.md`)
AND a layer-2 dated snapshot, **the latest dated snapshot is canonical.** The current canonical baseline is named
in `PROJECT_STATUS.md` "Active Baseline" (the first bullet).

When v2.N work supersedes a metric reported in a prior snapshot or
status entry:

1. The new snapshot becomes the canonical source for that metric.
2. The prior snapshot must get a `> ⚠ SUPERSEDED — historical
   reference only.` banner at the top, pointing at the new snapshot.
3. Any layer-1 reference to the old metric must be inline-annotated
   (e.g. `**⚠ Superseded YYYY-MM-DD** — see <new-snapshot>`) — do not
   delete the old number; future agents need to compare. Marker
   conventions:
   - `[x]` becomes `[~]` when an item is historically complete but
     superseded (still useful for archaeology, no longer current).
   - Inline `(was X)` is fine for short metric updates.
   - For longer narrative entries, append a `**⚠ Superseded ...**`
     sentence rather than rewriting from scratch.

This rule was added after the v2.9 plan-writing prompt picked up
stale `Ayeva indentation_fidelity=0.93` from layer-1 docs while the
canonical v2.8 fresh re-conversion read `0.83 FAIL`. The two numbers
are both correct in their respective contexts; the failure was that
layer-1 docs did not mark the v2.8 supersession.
