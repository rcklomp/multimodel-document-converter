# Extraction-Fidelity Plan Audit Prompt (external, unbiased model)

> Hand this entire file to a fresh frontier-model session (e.g. GPT-5.x or
> Gemini, ideally one from a different model family than the plan's author) that
> has read access to the MM-Converter-V2 repository. It is a standalone audit
> prompt. It produces a remediation report; it does not edit anything.
>
> Run this against the document `docs/PLAN_EXTRACTION_FIDELITY_V1.md` (rev. 3,
> "APPROVED FOR EXECUTION 2026-06-10").

---

## 0. Role and quality bar

You are an external, unbiased principal engineer auditing the single plan that is
meant to take this PDF-to-JSONL multimodal ETL pipeline from "research-grade,
intermittently working" to **production-level quality**. You have no prior
involvement and no stake in any decision; your only loyalty is to whether this
plan, executed as written, actually delivers a production-ready extraction layer.

Apply scrutiny commensurate with: *"this is the plan the owner is counting on to
FINALLY ship a production system; if it optimizes the wrong thing or ships a
false-confidence decision, the project stays stuck for another cycle."*

### The load-bearing artifact

The load-bearing artifact of this plan is **the Phase 1 two-corpus selection
mechanism and its decision rule** - the procedure by which the production-default
extraction engine is chosen from (a) OmniDocBench labeled fidelity (text edit
distance / table TEDS / reading-order ED) and (b) the internal 16-doc "crucible"
retrieval-value axis. Every downstream phase (specialist-lane identification,
reliability correction, the default-route flip, the spec rewrite) inherits
whatever this mechanism decides. If the selection mechanism is statistically
under-powered, runs on an unrepresentative internal corpus, lacks a
pre-registered decision rule, or is corrupted by the very server fault it routes
around, then the plan picks the wrong default *with a fidelity report that makes
the choice look rigorous*. Weight your audit toward this artifact, not toward the
most-visible artifact (the phase list).

A secondary load-bearing artifact: **the quality-risk arbitration's
"action-on-flag" policy** (Section 5.3). The plan defines a new
`extraction_quality_risk` provenance flag but never states what *happens* to a
flagged page in production. A flag with no defined consumer is a relabel, not a
correction.

Your output is a remediation report precise enough that the next session can fix
every substantiated item with nothing left ambiguous.

---

## 1. Background context (so findings are concrete, not abstract)

**What this plan is.** A Layer-2 execution plan that re-evaluates the project's
extraction thesis. Current shipped behavior asserts VLM-native extraction is
primary with a deterministic engine as a blind degraded fallback. The plan
proposes a CANDIDATE thesis - a specialized OCR/parsing PIPELINE as primary, the
VLM as a targeted specialist - to be PROVEN OR REFUTED by a bake-off, plus a
reliability correction (retry-before-fallback, quality-risk arbitration, demote
the fail-closed ladder to last-resort).

**Project state vs plan state (verify against the repo, do not take on faith):**
- The fail-closed 3-tier extraction ladder is SHIPPED (`fcd4207`, `e73fbd8`,
  `1cd20fe`): tier 1 selected engine -> tier 2 offline `DoclingFastEngine` ->
  tier 3 PyMuPDF native-text terminal tier. It fixed a real silent-data-loss bug
  (MinerU server `broadcast_shapes` 500s returned an empty doc with `failed=0`).
- The OmniDocBench fidelity benchmark Phase 0 is DONE: full-755-page English
  baseline text ED **0.301** / TEDS **0.563** (`scripts/omnidocbench_adapter.py`).
- The Phase 1 extractor bake-off already RAN and came back **INCONCLUSIVE** -
  blocked by the same intermittent M5 `broadcast_shapes` server fault. Only
  `docling_fast` produced clean preds (text ED 0.277 / reading 0.307 / TEDS 0.441
  on the bake-off subset). See `docs/PLAN_OMNIDOCBENCH_EVAL.md` Section 13.
- The internal "crucible" retrieval-value corpus is **16 documents**
  (`docs/PLAN_GATE_QUALITY_V1.md`, "IN PROGRESS"); that plan also states plainly:
  *"Source-fidelity at scale: No labeled ground truth."*
- Phase 0A (VLM operational profiling) has NOT been run yet
  (`scripts/phase0a_render_sweep.py` exists, 227 lines; `logs/phase0a_sweep.log`
  is empty).
- Test state: `pytest tests/` ~1574 passed / 99 skipped, plus **1 known,
  pre-existing, unrelated failure** (`test_v3_vlm_code_form.py::
  test_code_smuggles_as_text_promotes_to_code_modality`).
- The M5 endpoint is a working Qwen3-VL-8B mlx server at `10.0.10.235:8000`. It is
  the project's only VLM serving path and is operationally fragile; the owner has
  a standing memory rule NEVER to reconfigure it autonomously (it risks the
  working Qwen serving).

**External-state dependencies and their failure modes (the auditor must weigh
these):**
- The bake-off depends on a healthy MinerU serving path. The plan routes around
  the fault (Phase 0 has a fallback to engines that serve reliably) - but if no
  alternative engine is actually wired, Phase 1 cannot render a verdict and the
  load-bearing artifact silently fails the same way it already did once.
- Phase 1's decisive axis depends on `PLAN_GATE_QUALITY_V1`'s retrieval-value
  scorer being built, trustworthy, and frozen. If it is not, the decisive corpus
  produces a verdict with no calibrated instrument behind it.
- Phase 0A inference runs against the live M5 endpoint. On-host GPU%/thermal
  sampling needs a sampler ON the M5 (powermetrics over ssh); client metrics need
  only the endpoint.

**Load-bearing constraints the auditor MUST respect (a fix that violates one of
these is not a valid fix - flag the tension instead):**
- NO human-in-the-loop as a fix for system fragility (the owner rejects "user
  reviews X" designs; humans are inconsistent/lazy). So "add manual review of
  flagged pages" is NOT an acceptable action-on-flag.
- Fix the EXTRACTION layer, never weaken the JUDGE/gate to mask a defect
  (AGENT-INTEGRITY-01: "assert OUTCOMES, not PROXIES"; no-weaken-gate rule).
- NO model-swap-as-reflex: do not recommend "just standardize on engine X"
  reflexively; each serving swap costs real load + re-calibration time. Engine
  choice must be evidence-driven.
- Libraries-first: prefer a Docling/MinerU/PaddleOCR native capability over
  custom heuristics.
- Locked invariants (do NOT propose changes to these): extraction confined to
  `mmrag_v3.extract`; `batch_processor` keeps ZERO Docling imports; the
  `tests/test_v3_security.py` AST firewall; ElementType frozen at 3 / Modality at
  5; QA-CHECK-01 tolerance 0.10 (no waivers); bbox emitted as integer [0,1000];
  Python pinned 3.10; docling pinned 2.86.0.

**Stopping rule.** This is one of >=2 planned Round-1 audits from different model
families. Round 1 is complete only after two consecutive rounds (summed across
all auditors) return 0 HIGH-severity findings. If you find 0 HIGH, say so plainly
- do not pad to a count. If you find structural problems, report exactly what you
can substantiate.

---

## 2. What NOT to re-flag (already catalogued or out of scope)

Do not spend tokens on these - they are known and dispositioned:
- The doc-hygiene findings F2/F4/F6/F7/F8/F9 from the 2026-06-09 governance audit
  (stale line in `V3_EXECUTION_MANDATE`, phantom `SRS` in AGENTS.md, stale
  test-command count, `shadow_ocr` legacy marking, 0.5-target tag wording). The
  plan already catalogues these in Sections 8-9 for Phase 5.
- The known 1 unrelated test failure (`test_v3_vlm_code_form`) - already
  registered to be fixed before Phase 3.
- The locked invariants listed in Section 1 - they are constraints, not
  candidates for change.
- The choice to phase the work at all, or the information-gain ordering
  (0A -> 0.5 -> 1 -> 0 -> 2+) - that ordering is deliberate and approved. You may
  challenge whether a SPECIFIC phase is mis-ordered, but not the principle.

## 2a. Open self-review observations (NOT yet incorporated into the plan)

A self-review by the plan's own author surfaced the seven observations below.
They are NOT yet folded into the plan text, so they remain open. Your job re:
these is twofold: (1) pressure-test each - is the diagnosis sound, is it HIGH or
overstated, is the implied fix compatible with the Section-1 constraints? and
(2) find the structural problems these SEVEN MISSED. Do not simply re-derive
them and stop. A self-review systematically under-weights areas the author is
uncertain about and produces additive fixes (add a section, add a flag) rather
than structural ones (this plan should not exist in its current shape; the
offline floor should just ship as the interim default now). Expect to find at
least 1-2 structural findings the self-review dodged.

- SR1: The plan is an extraction-engine selection plan, not a production-readiness
  plan; it equates "win the bake-off" with "production-ready" and never defines
  what production-level means end-to-end.
- SR2: The `extraction_quality_risk` flag (Section 5.3) has no defined consumer or
  action - arbitration is currently a relabel, not a correction.
- SR3: The decisive axis rests on the 16-doc internal crucible, the
  least-specified artifact; is 16 docs representative/frozen/large enough to
  decide a production default?
- SR4: No pre-registered decision rule (margin threshold + variance/significance
  handling) for a bake-off whose blocking fault was INTERMITTENT - risk of
  deciding on noise.
- SR5: No "shortest path to a defensible baseline" - the shipped offline floor
  could be declared the interim production default NOW, with the smart lane
  developed behind it.
- SR6: No production runtime story - observability of the flag in prod,
  re-extraction/migration when the engine or render setting changes (note
  `scripts/rebaseline_v3.py` is documented as broken), canary/rollback for the
  Phase 4 default flip.
- SR7: Specialist-lane complexity (Phase 2) is unbudgeted - lanes x tiers x retry
  x arbitration is a combinatorial test/failure surface with no cap.

---

## 3. Systematic-bias callout

The plan was written by the same agent that performed the self-review in Section
2a. Self-audits avoid restructuring the load-bearing artifact and instead produce
fixes that are ADDITIVE (add a Section 0, add an action-on-flag spec, add a
decision rule) rather than STRUCTURAL (re-sequence the decision, admit a phase is
premature, admit the plan is answering the wrong question, admit the offline
floor should ship as production NOW and the rest is optimization). Specifically
look for:
- Whether the plan's rigor is a form of avoidance: is "rigorous two-corpus
  bake-off before any decision" being used to defer SHIPPING a defensible
  production baseline that already exists (the offline floor)?
- Whether the CANDIDATE-thesis framing ("proven or refuted by Phase 1") is honest
  or is a hedge that lets the plan avoid committing to any falsifiable prediction.
- Whether any phase's Definition of Done asserts an OUTCOME ("the winning default
  named") that the phase's own instruments cannot actually produce (e.g. a verdict
  from an under-powered corpus or a fault-corrupted bake-off).

Expect at least 1-2 structural findings the self-review dodged.

---

## 4. Audit lenses (adversarial, specific to THIS plan)

Engage every numbered lens. For each, either give a finding or explicitly state
"audited, nothing to flag" with one sentence of why.

**L1 - Load-bearing decision-rule lens (the central question).** The plan says
the default is chosen when the winner "beats ... by a margin worth the change"
(Sections 1, 11) but never defines the margin, never states how per-page metric
variance is handled, and the blocking fault (`broadcast_shapes`) is intermittent
- so per-page numbers carry real noise. Walk the chain: can a mean-vs-mean
comparison on a ~few-hundred-page subset, with one engine intermittently
500-ing, distinguish a real win from sampling noise? Is there a pre-registered
decision rule (threshold + significance/variance treatment) set BEFORE the run,
or does the plan leave room to rationalize the result post-hoc? Is the baseline
(text ED 0.301 / TEDS 0.563) a full-corpus number being compared against
subset numbers (0.277 / 0.441) - i.e. is the comparison even apples-to-apples?

**L2 - Decisive-corpus validity lens.** Phase 1's tie-breaking, production-
deciding axis is the internal 16-doc crucible (I2 makes it MANDATORY, "the
benchmark never decides alone"). The plan never establishes the crucible is
frozen, retrieval-value-labeled with answer-level ground truth, representative
across all 17 `data/` classes, or large enough to be decisive. `PLAN_GATE_QUALITY
_V1` itself states "no labeled ground truth" for source fidelity at scale. Is the
single most decision-critical input in the plan actually trustworthy enough to
pick a production default, or is the whole thesis being decided on 16 documents
scored by an instrument the dependency plan admits is uncalibrated for fidelity?

**L3 - Metric-validity lens (distinct from calibration).** For each acceptance
metric, write the worst-case failure mode of the engine being gated and ask: can
this metric MOVE when that failure happens, by construction? Specifically: (a)
text edit distance and TEDS penalize functionally-equivalent formatting - the
plan acknowledges this (caveat (a), I5) and switches to "regression-vs-baseline."
Does regression-vs-baseline actually detect the failure modes the project cares
about (code-indentation loss, dropped wiring-diagram labels, flattened tables),
or can a page lose exactly those and still not regress the aggregate ED? (b) The
internal axis is "retrieval-value" - does retrieval-value move when extraction
drops a tiny label or mangles code indentation, or is it insensitive to
exactly the small-content failures (I6) the plan most fears?

**L4 - Action-on-flag lens (the secondary load-bearing artifact).** Section 5.3
introduces `extraction_quality_risk` and promises "never silently substitute a
lower-quality representation as equivalent" - but the only stated consumers are
"the offline fidelity gate and any human review." Human review is forbidden by
the no-human-in-the-loop constraint. The offline gate is a selection/regression
gate, not a per-conversion production actor. So in PRODUCTION, what happens to a
flagged page? If the answer is "it ships with a flag nobody consumes," is the
arbitration mechanism functionally identical to today's accept-on-presence, just
with extra metadata? What is the actual production action (auto-route to
specialist, exclude from index, force retry tier, accept-with-degraded-marking),
and is that action specified or merely implied?

**L5 - Retry-before-fallback honesty lens.** Phase 0.5 adds idempotent bounded
retry of the primary engine before any cross-engine move, "reusing the existing
`vlm_provider` retry machinery; extend it to the MinerU path." Verify against the
code: does `vlm_provider` retry machinery exist and is it actually reusable for
the MinerU HTTP path? Does retry against an intermittent `broadcast_shapes` fault
(a server-side batching bug, not a transient network blip) actually recover, or
does the same malformed batch 500 every time for a given page set - making
"retry" a no-op that masks an unfixed server bug (X6) the plan explicitly defers?
Is Phase 0.5 shippable independently as claimed, or does it secretly depend on
the MinerU fix?

**L6 - Trigger-gaming / unfireable-conditional lens.** The plan's central
conditional is "CANDIDATE thesis CONFIRMED or REFUTED in Phase 1." Could
"confirmed" be reached permissively (e.g. by comparing the pipeline against a
VLM hybrid that is itself degraded by the unfixed server fault, so the pipeline
"wins" only because its competitor is broken)? Is the bake-off fixture
structurally capable of REFUTING pipeline-primary - i.e. does the two-corpus set
contain enough chart/diagram-to-data, full-bleed-visual, and code-indentation
pages (the classes where VLMs are supposed to win) for the VLM lane to actually
demonstrate a win, or is the corpus stacked toward prose/tables where pipelines
always win? A conditional that cannot fire its "refuted" branch is dishonest.

**L7 - Trigger-input-intrinsicness lens.** Phase 2 cuts a specialist lane only if
it shows "no measured win on the benchmark AND no measured win on the internal
corpus." Does that evaluate inputs INTRINSIC to the technical question (does this
content class technically need a specialist?) or extrinsic ones (did the 16-doc
crucible happen to contain enough of that class this run)? If a wiring-diagram
specialist is cut because the crucible had only one wiring diagram, the decision
was made for the wrong reason. Is lane KEEP/CUT gated on the technical evidence
or on the accident of corpus composition?

**L8 - Safety-valve demotion lens.** The plan demotes the fail-closed 3-tier
ladder from "the reliability story" to "last-resort net." But it KEEPS it, and
keeps accepting tier-2/tier-3 output. Does demoting-but-keeping create a perverse
incentive: the smart lane can be shipped sloppy because the ladder absorbs the
cost? Under the new design, when retry + arbitration both fail, the ladder still
serves a possibly-degraded page - is that page provenance-stamped AND excluded
from any "fidelity passed" claim, or can ladder output silently satisfy the
structural gates exactly as the audit (X2) says it does today?

**L9 - Phase 0A measurement-completeness lens.** Phase 0A is the highest-info-gain
phase and the answer to the owner's actual pain (M5 load). It sweeps render
settings (200/150 DPI, 1600/1400px) and scores each on both corpora. Does the DoD
actually let the central hypothesis ("the problem is rendering every page at
~1700x2200, not the architecture") be CONFIRMED or REFUTED - i.e. will the output
table let someone conclude "a DPI cut relieves the M5 with negligible two-corpus
quality loss" with enough confidence to act before the routing redesign? Is the
fixed-page artifact-capture sufficient to SHOW how a setting fails (labels gone,
borders collapsed) rather than only assert a delta? Is anything missing from the
profiled set (e.g. cold-start vs warm latency, batch vs single-page, the actual
token-cost-per-page the owner is paying)?

**L10 - Production-runtime completeness lens (SWE-standard).** This is THE
production plan. Check for the production concerns a research plan omits: (a)
rollback/canary for the Phase 4 default flip - it is validated only on two finite
corpora; production traffic contains documents in neither; is there a
shadow/canary period and a rollback trigger, or is it a one-shot cutover? (b)
re-extraction/migration - when the engine or render setting changes, already-
ingested documents are stale and were produced by a different engine;
`rebaseline_v3.py` is documented as broken; is there a migration story? (c)
production observability - when `extraction_quality_risk` fires at scale, is there
a metric/alert, or only a per-doc field? (d) extraction-provenance versioning so
you can tell which doc was produced by which engine/config.

**L11 - Scope / shortest-path-to-production lens (the owner's actual goal).** The
owner's stated goal is to FINALLY reach production. The shipped offline floor
(Docling -> PyMuPDF) is a defensible, reliable, measured baseline that exists
TODAY. Does the plan anywhere offer the option to declare the offline floor the
interim PRODUCTION default NOW and develop the smart lane behind a shipping
baseline - converting "still researching" into "in production, optimizing"? If
not, is the plan's full-rigor-before-any-ship structure actually the fastest path
to production, or is it research rigor standing in the way of shipping? This is a
scope/altitude finding, not an additive one - state plainly whether the plan is
shaped wrong for its stated goal.

**L12 - META lens (audit-the-audit).** Look back at this plan's three revisions
and the self-review in Section 2a. Did ANY revision restructure the load-bearing
artifact (the selection mechanism / decision rule), or were all changes additive
(folded in review conditions, added a profiling phase, added proxies, reworded
objectives)? Three revisions of additive refinement around an undefined decision
rule is the signature of a self-audit dodging the hard question. If every prior
change was additive, name the structural change the plan has been avoiding.

---

## 5. Anti-escape-hatch "Don't" list

- Don't recommend "defer to a later phase / next plan" as a fix unless the item is
  genuinely out of this plan's scope. This plan exists to STOP the pattern of
  deferring the hard decision.
- Don't accept assertion-backed rationales where evidence is available. "The
  pipeline is primary in the literature" is not evidence the pipeline wins on THIS
  corpus; the plan correctly knows this - hold it to that standard everywhere.
- Don't let "CANDIDATE thesis," "advisory gate," or "quality-risk flag" become
  euphemisms that preserve plausible deniability while changing nothing. If a
  label disguises a no-op, name it.
- Don't recommend adding a process step or a new document section as a substitute
  for a structural fix. If the decision rule is undefined, the fix is to define it
  with a falsifiable threshold, not to "add a calibration discussion."
- Don't propose any fix that violates a Section-1 constraint (no human-in-loop, no
  gate-weakening, no model-swap-reflex, no invariant change). If the real fix
  needs one of these, REFRAME as "this constraint should be re-litigated" with
  explicit user-decision-required framing, and present the trade-off - do not
  silently recommend violating it.
- Don't suggest "more validation / more testing" without naming the specific
  failure mode the extra validation would catch on a specific input class.
- Don't verify a factual claim by plausibility - check it against the repo. The
  plan makes checkable claims (the ladder is shipped, `vlm_provider` retry exists,
  the crucible is 16 docs, `rebaseline_v3.py` is broken, the baseline is ED 0.301
  / TEDS 0.563). Sample 2-3 and verify against actual files.
- Don't treat "audited, nothing to flag" on a lens as proof the lens is empty - it
  may be your blind spot; a second auditor may flag it HIGH. Be explicit about
  confidence.
- Don't pad to a finding count. 0 HIGH is a legitimate, useful result.
- Don't conflate "the self-review's recommended fix is wrong" with "there is no
  problem." If you reject an SR fix, re-read its diagnosis and say whether a
  DIFFERENT fix would still be warranted.

---

## 6. Required output format

### Findings
For each finding:
- **ID** (e.g. A1, A2) and **severity** (HIGH / MED / LOW)
- **Plan section** (the exact section/line of `PLAN_EXTRACTION_FIDELITY_V1.md`)
- **Issue** (one sentence)
- **Concrete failure mode** (what specifically goes wrong in execution or in
  production, on what input class - not an abstract concern)
- **Recommended fix** (compatible with the Section-1 constraints; if it cannot
  be, say so and reframe as user-decision-required)
- **Confidence** (high / medium / low) and what evidence would raise it
- If it relates to an SR1-SR7 self-review item, say which and whether you
  confirm / sharpen / downgrade it

### Audit lenses with nothing to flag
List every lens L1-L12 you engaged that produced no finding, each with one
sentence of why it is clean. (This forces engagement with every lens.)

### Overall stance
- Is the load-bearing-artifact discipline correctly applied - i.e. is the Phase 1
  two-corpus selection mechanism + decision rule actually trustworthy enough to
  choose a production default, or does it produce a false-confidence verdict?
- Is the plan SHIPPABLE as the production roadmap as-is, NEEDS REVISION before the
  next audit round, or NEEDS SUBSTANTIVE RESTRUCTURING (e.g. it answers the wrong
  question / should ship the offline floor as the interim default now)?
- State your confidence that this plan, executed as written, delivers
  production-level extraction quality - and name the single change that would most
  raise that confidence.
- A vague "looks solid" stance is not acceptable.
