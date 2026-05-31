# Control-Document Audit - 2026-05-31

**Remediation status (2026-05-31): APPLIED.** C1, C2, C3, C4, O1, O2, G2, S1, S2
were fixed in the same session (MANDATE §2/§3 rewrite; V3_DEFERRED_TESTS.md
regenerated; AGENT-TEST-01 carve-out; AGENT-DOCS-01 + README "single governance
doc" correction; TESTING.md judge provider; Docling banner; AGENTS.md §6 TODO
moved). OV1/OV2 (the 10%-tolerance and form-class restatements) were left as
deliberate, consistent quick-references rather than churned. Integrity guards
(test_repo_integrity G1-G6) and the full suite stayed green. The findings below
are retained as the rationale of record.


Scope: the agent-binding control docs (CLAUDE.md, AGENTS.md,
docs/V3_EXECUTION_MANDATE.md, docs/QUALITY_GATES.md, docs/TESTING.md,
docs/README.md, docs/PROJECT_STATUS.md, docs/V3_DEFERRED_TESTS.md, plus the
relevant DECISIONS.md / ARCHITECTURE_V3_DRAFT_0.5 anchors). Goal: find gaps,
overlaps, contradictions, and over-strict rules introduced or left stale by the
V3 cleanup, so agents are guided enough not to break the system but free to fix
and improve it.

## The meta-finding (read this first)

The governance is not simply "too strict." It has split into two classes, and
the mix is the actual hazard:

- LOAD-BEARING constraints that genuinely prevent breakage and are honored:
  AST firewall (test_v3_security), QA-CHECK-05 (image/table need asset_ref),
  int [0,1000] bboxes, no-Docling-in-batch_processor, ProfileClassifier routing,
  "assert OUTCOMES not PROXIES" (AGENT-INTEGRITY-01), repo-integrity G1-G6
  (test_repo_integrity runs green).

- DEAD-LETTER strict rules that are unachievable or already violated, so the
  project ignores them while still printing them: the V3 Definition of Done
  (MANDATE §2), "all v2.16 heuristics permanently deferred" (§3), "the single
  governance file" (AGENT-DOCS-01), and an Identity Gate that was never built.

Dead-letter strict rules are the worst case. An agent that OBEYS them is blocked
from fixing real regressions (e.g. HEADING coverage is "permanently deferred").
An agent that sees the project IGNORE them (Phase A/C declared CLOSED/SHIPPED
despite failing the stated DoD) learns that all governance is optional - which
then erodes the load-bearing rules that must hold. The fix is not "loosen
everything"; it is to separate the two classes: keep the load-bearing rules
strict, and repair or delete the dead-letter rules so the rule set is true again
and explicitly authorizes the fixes the project needs.

## Findings (severity-ranked)

### C1 - CONTRADICTION (critical): the V3 Definition of Done is unachievable and already violated
- MANDATE §2 requires, for an architectural phase to be "done":
  (2.2) `pytest tests/` exit 0 with ZERO skipped tests added, and
  (2.3) "the Identity Gate script runs and outputs a < 5% delta."
- Reality: the V3 work ADDED 6 unconditionally-skipped V3_DEFERRED modules
  (verified by replicating tests/test_repo_integrity.py's own AST check;
  PROJECT_STATUS's "6" is accurate). Either way >0, so "zero added" is violated.
  The Identity Gate script
  `scripts/run_identity_gate.py` was NEVER BUILT (V3_DEFERRED_TESTS.md line 28
  states this explicitly), and the only identity tool, `rebaseline_v3.py`, is
  broken (imports the retired sandbox). Yet Phase A is "CLOSED" and Phase C
  "SHIPPED." Per AGENT-STATUS-01 ("a phase either passes the gates or it has
  failed") those statuses are, by the project's own rule, invalid.
- Internal self-contradiction: §2.3 demands "< 5% delta" while §3 says "chunk
  count and content parity are explicitly excluded ... the V3 chunker
  fundamentally alters chunking shape." A <5% delta vs v2.16 is impossible by
  design (invoice 4 vs ~11 chunks). The current "Identity Gate" is V3-vs-its-own-
  rebaseline (0.00% delta) - structurally circular, not a real gate.
- Fix: rewrite MANDATE §2 to a DoD that is real and achievable:
  (a) `tests/test_v3_security.py` exit 0;
  (b) full suite exit 0 with every skip REGISTERED (not "zero added") - see O1;
  (c) the production-CLI smoke (scripts/smoke_production.sh, PLAN_V3.1 P5) green
  on one doc per routing lane, asserting no 0-chunk batches + asset_ref +
  visual_description + QA_PASS. Delete the "<5% delta Identity Gate" criterion or
  redefine it as an explained-delta review (identity-half >=95% + explained-delta
  <=5%, the ARCHITECTURE_V3_DRAFT split), not a single number against v2.16.

### C2 - OVER-STRICT + CONTRADICTION (critical): "permanently deferred" blocks the HEADING fix, on a falsified basis
- MANDATE §3: "All v2.16 heuristic reconciliation paths are permanently deferred
  per V3_DEFERRED_TESTS.md." This is the rule that, read literally, FORBIDS fixing
  HEADING coverage - the regression the user says should have been fixed long ago.
- The basis is falsified by its own authority doc. V3_DEFERRED_TESTS.md states:
  "Phase B (text-based LLM sanitization ...) was completed on 2026-05-29 with a
  falsified hypothesis." The deferrals wait for an LLM-sanitization layer that
  was proven not to close the delta. So "deferred until Phase B subsumes them"
  defers to something that does not work - i.e. permanent limbo, which
  AGENT-STATUS-01 forbids.
- It also mis-scopes the HEADING fix. Heading propagation is TOC-driven (PyMuPDF
  bookmarks already extracted by batch_processor._extract_toc_headings), NOT a
  text-sanitization problem. The "needs LLM sanitization" rationale never applied
  to it. The deferral has no valid basis.
- Fix: replace "permanently deferred" with a DISPOSITIONED backlog. Each deferred
  behavior must be one of: (a) RESTORED, (b) DELETED by a DECISIONS.md entry, or
  (c) DEFERRED WITH an owner + an explicit un-defer trigger + a date. Explicitly
  un-block heading propagation (PLAN_V3.1 P2): it is TOC-driven and independent of
  the falsified Phase B.

### C3 - GAP (critical): the deferral-authority doc is a stale sandbox salvage that does not match reality
- MANDATE §3 binds to V3_DEFERRED_TESTS.md as the authority for what is deferred.
  That doc lists ~90 tests. Reality: only 6 modules are unconditionally
  skipped with a V3_DEFERRED marker (the real deferred surface).
  ~50+ of its "out of v3 scope" entries (retrieval, hyde, sparse_bm25,
  token_validator, contextual_retrieval, qdrant, refiner, intent, telemetry)
  STILL RUN AND PASS - verified: `pytest test_hyde test_sparse_bm25
  test_token_validator test_contextual_retrieval` = 79 passed. The doc was
  salvaged from the retired sandbox and never reconciled to the production suite.
- Why it passed unnoticed: the repo-integrity G6 guard enforces only
  skip ⊆ doc (every skip must be listed), NOT doc ⊆ skip. Over-listing is
  silently legal. So a binding contract points at a largely-fictional registry.
- Note: PROJECT_STATUS's "6 deferred test modules" is accurate; the fiction is
  V3_DEFERRED_TESTS.md's ~90-entry list.
- Fix: regenerate V3_DEFERRED_TESTS.md from the 6 actually-skipped modules;
  delete the out-of-scope/aspirational entries (those tests pass - they are not
  deferred). Demote the doc from "binding authority" to a Layer-2 execution
  registry, OR fold the real list into PROJECT_STATUS. Reconcile the "6" claim.

### C4 - CONTRADICTION (moderate): "the single governance file" is false
- AGENT-DOCS-01 and README: "Do not add new governance docs. V3_EXECUTION_MANDATE
  is the only one." But AGENTS.md §4.2 and README's own Layer-0 list enumerate
  SEVEN contract docs (AGENTS, CLAUDE, MANDATE, DECISIONS, QUALITY_GATES, the two
  ARCHITECTURE docs, SRS). And CLAUDE.md's Workstream-B guardrail invites new
  `docs/PLAN_V2.8_*` docs.
- Risk: an agent takes "do not add new governance docs" literally and refuses to
  write a needed plan/audit (this audit, PLAN_V3.1), or is confused about which
  doc actually rules. Reading the other direction, agents may treat the
  contradiction as license to ignore AGENT-DOCS-01 entirely.
- Fix: state the truth. The governance SET is the seven contract docs;
  V3_EXECUTION_MANDATE is the CONFLICT-RESOLUTION authority, not the only file.
  Plans, audits, and execution docs are explicitly allowed and are not
  "governance docs." Keep the real intent (do not proliferate overlapping
  CONTRACT docs) and drop the false "only one" claim.

### O1 - OVER-STRICT (moderate): AGENT-TEST-01 has no carve-out for deleting dead tests, and the skip-pattern violates its own rule
- AGENT-TEST-01: "Do not remove, loosen, rewrite, or reframe core assertions."
  Read literally this blocks deleting a test that pins behavior an approved
  decision removed - exactly PLAN_V3.1 P3's "restore-or-delete deferred tests."
- Self-inconsistency: skipping a test to "defer" a behavior IS a way to "reframe
  to match the current implementation," which AGENT-TEST-01 forbids. The V3
  cleanup's own V3_DEFERRED skip-pattern is in tension with the rule as written.
- Fix: add the carve-out. Removing a test is permitted ONLY when the behavior was
  removed by a DECISIONS.md-recorded decision and the removal is logged.
  Weakening/skipping a test to make a failing implementation pass remains
  forbidden. A V3_DEFERRED skip is legal only with a cited decision + un-defer
  trigger (ties to C2). This keeps the load-bearing intent (no hollow-green) while
  permitting honest cleanup.

### O2 - OVER-STRICT (moderate): batch_processor scope in MANDATE §3 is narrower than reality
- MANDATE §3: batch_processor is "strictly limited to engine-agnostic
  orchestration (batching, routing, JSONL writing)." Reality: it also runs
  spatial-refiner + mid-sentence merge + quality filters + (as of 455eac8)
  asset-crop rendering. A strict reading forbids the asset-rendering fix that was
  required to make vision-native image/table chunks satisfy QA-CHECK-05.
- Fix: broaden to "engine-agnostic orchestration + emission-side asset/quality
  finalization; NO source extraction and NO Docling construction." This still
  bans the thing that actually matters (Docling in batch_processor, guarded by
  test_v3_security + the AGENTS.md note) while matching what the file legitimately
  does.

### G2 - GAP (minor-moderate): TESTING.md judge/gen provider is stale
- TESTING.md: "DASHSCOPE_API_KEY required for synthetic-soak judge + query
  generation." Since the GX10 work, synthetic_soak supports `--judge-provider
  vllm` / `--gen-provider vllm` pointed at the GX10 (now a standing service), and
  the default judge path is the local 14B-FP8. Dashscope is no longer required.
- Fix: update TESTING.md to document the vllm judge/gen path + the local default;
  keep dashscope as the cloud alternative.

### OV1 - OVERLAP (minor): QA-CHECK-01 10% tolerance stated in 4 docs
- CLAUDE.md, PROJECT_STATUS.md, AGENTS.md §5, QUALITY_GATES.md all state the 10%
  tolerance. Currently consistent, but four copies of a number drift.
- Fix: QUALITY_GATES.md is the single source; the other three link to it instead
  of restating the number.

### OV2 - OVERLAP (minor): Form-acceptance "not a waiver" explained in 3 docs
- CLAUDE.md invariants, AGENTS.md §5, QUALITY_GATES.md each re-explain the form
  acceptance class and the "not a waiver" framing.
- Fix: QUALITY_GATES.md owns the definition; others point to it.

### S1 - STALE (minor, cosmetic but misleading): "Docling v2.66.0" banner
- Installed + pinned Docling is 2.86.0 (verified), but cli.py (lines 781, 1936)
  and __init__.py print "Docling v2.66.0". A reader trusting the banner would
  think the 2.86.0 pin is violated.
- Fix: update the three banner strings to 2.86.0 (or read from the package).

### S2 - WRONG-LAYER (minor): a transient TODO lives in a Layer-0 contract
- AGENTS.md §6 carries "scripts/v3_batch_ingest.py, rebaseline_v3.py need
  repointing to uir_chunker" - execution state inside a stable-contract doc.
- Fix: move it to PROJECT_STATUS / PLAN_V3.1 (P1). Contract docs should not carry
  per-cycle TODOs (AGENTS.md §5 itself says so).

## Recommended remediation order

1. C1 + C2 first - they are what block the reconvergence and what teach agents to
   ignore governance. Rewrite MANDATE §2 (real DoD) and §3 (dispositioned backlog
   + un-block heading). These two unlock PLAN_V3.1 P2/P3.
2. C3 - regenerate V3_DEFERRED_TESTS.md from reality; reconcile PROJECT_STATUS.
3. O1 + O2 + C4 - add the test-deletion carve-out, fix the batch_processor scope,
   and correct the "single governance file" claim.
4. G2 + OV1 + OV2 + S1 + S2 - stale/overlap/cosmetic cleanups.

## What to KEEP strict (do not touch)

AST firewall (test_v3_security), QA-CHECK-05, int [0,1000] bboxes,
no-Docling-in-batch_processor, ProfileClassifier-only routing, AGENT-VAL-01 blind
test, AGENT-INTEGRITY-01 (assert OUTCOMES not PROXIES) + repo-integrity G1-G6,
AGENT-SPATIAL-20. These are load-bearing and currently honored; the audit
recommends no change to them.

## Net
The system is under-guided where it matters for honesty (deferrals have no owners
or triggers; the deferral registry is fiction; the DoD references a script that
does not exist) and over-guided where it matters for progress (heading fix
"permanently deferred"; test deletion forbidden; batch_processor scope too narrow).
Both pull the same direction: agents cannot both follow the rules AND fix the
system. Repairing C1-C2-C3 makes the rules true again and gives the next agent
explicit license to execute PLAN_V3.1 without violating governance.
