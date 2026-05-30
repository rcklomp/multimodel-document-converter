# Repo Integrity — the "Committed-Truth" convention

A lean, mostly-mechanical convention that prevents a specific class of
recurring failures observed 2026-05-30: a clean clone of `HEAD` was not sound
even though everything "worked" in a developer's dirty working tree.

**Principle.** Truth is what `git` would hand a fresh clone, not what happens to
sit in your working directory. Mechanically checkable invariants are enforced as
pytest guards that run in the always-on hosted CI job (no corpus / GPU /
network). Only the one genuinely-judgment item is left as a written rule.

## Mechanical guards — `tests/test_repo_integrity.py`

Each runs against `git ls-files` (the *committed* tree). Wired into CI's hosted
`quick` job next to `tests/test_v3_security.py`.

| Guard | Invariant | Failure it prevents |
|---|---|---|
| **G1** import closure | every local module a tracked top-level (unguarded) import names is itself tracked | clean-clone `ModuleNotFoundError` — tracked code importing an untracked module |
| **G2** governance tracked | the Read-First docs (`CLAUDE.md` §"Read First") are committed | a fresh clone / CI with zero governance |
| **G3** precedence applied | every "SUPERSEDED ... by &lt;doc&gt;" marker names a tracked doc | unresolved Layer-0 contradiction re-derived by the next agent |
| **G4** contract liveness | a doc that says "guarded by ``test_X.py``" has that test tracked and not unconditionally skipped | a documented contract whose guard was deleted or skipped (hollow guarantee) |
| **G5** no dangling paths | live `src/`/`tests/`/`scripts/` + top-level `docs/*.md` refs in current-state docs resolve | docs pointing at moved/deleted paths |
| **G6** skips registered | every unconditional `V3_DEFERRED` test skip is listed in `docs/V3_DEFERRED_TESTS.md` | behavioral coverage silently rotting off the books |

### Conventions the guards rely on (so authors can satisfy them)

- **Forward references** (a path that does not exist yet, e.g. a planned script)
  must be annotated **on the same line** with one of: `NOT YET BUILT`,
  `not yet`, `(planned)`, `to be built`. G5 then treats it as intentional.
- **Superseding** a Layer-0 statement: add a literal `SUPERSEDED <date/why> by`
  marker followed by the winning doc's path in backticks, placed *at the
  conflicting content* (not only as a global "X supersedes Y" rule elsewhere).
  See the live example in `docs/ARCHITECTURE_V3_DRAFT_0.5.md` (the Step-5
  callout deferring to the Execution Mandate). G3 enforces the named target
  exists; applying it at the conflict site is what actually stops re-derivation.
- **Deferring a test**: mark the whole module
  `pytestmark = pytest.mark.skip(reason="V3_DEFERRED - ...")` **and** add a line
  to `docs/V3_DEFERRED_TESTS.md`. Use `skipif` (not `skip`) for runtime
  absence (corpus/GPU/network) — those are legitimate and are not flagged.

G1's scope is precisely "what breaks `import <module>`": **module-level,
non-`try`-guarded** imports. Lazy/function-local and `try/except`-guarded
optional imports (the documented fail-graceful sibling-engine pattern) are not
flagged.

## The one written rule (genuine judgment, not mechanizable)

**Gates must assert OUTCOMES, not PROXIES.** A speed check, an exit code, or a
"no exception" check is a proxy. When you add or change a quality gate, it must
verify the *content* the pipeline is supposed to produce — not a cheaper signal
that happens to correlate. This rule exists because online-FP8 quantization once
gave a real 1.73× speedup on *blank-page garbage* (`docs/paper/FINDINGS_LOG.md`,
F4): a speed-only gate would have shipped a pipeline that silently dropped every
page. No regex can know whether a given assertion measures the real outcome —
hence this stays a rule for the human/agent writing the gate. G4 + G6 cover the
mechanical half of the same failure (a gate that is skipped or deleted), but
*choosing to assert the right thing* is on you.
