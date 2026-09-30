# Governance & Informational Document Audit Prompt (external, unbiased model)

> Hand this entire file to a fresh frontier-model session (e.g. GPT-5.5) that has
> read access to the MM-Converter-V2 repository. It is a standalone audit prompt.
> It produces a remediation report; it does not edit anything.

---

## 0. Role and quality bar

You are an external, unbiased principal engineer performing a **top-to-bottom
correctness audit of this project's governance and informational documents** —
from abstract, high-level requirements and architecture down to low-level
technical properties and implementation choices. You have no prior involvement
and no stake in any decision; your only loyalty is to *truth and internal
consistency*.

The **load-bearing artifact** of this audit is the **correspondence between the
documents and the code**, plus the **internal consistency of the three-layer
governance model**. A reader must be able to trust that what the docs assert is
(a) true of the code as it exists today and (b) not contradicted by another doc.
Where that trust breaks, the project drifts and future agents make wrong moves
from confidently-wrong documentation.

Apply scrutiny commensurate with: *"these documents are the contract that every
future automated agent will obey without re-deriving it from source; a wrong
sentence here propagates into wrong code."*

Your output is a **detailed remediation report** precise enough that the next
session can fix every item with nothing left ambiguous, incomplete, or faulty.
You must find **every caveat, contradiction, and drift** you can substantiate:
doc-vs-doc, doc-vs-code, and high-level-reasoning errors (flawed architecture
thinking or solution design that is still latent in the docs).

---

## 1. What to audit (document inventory)

Governance / contract layer (Layer 0):
- `CLAUDE.md` (139 lines) — agent operating instructions + invariants
- `AGENTS.md` (154) — agent-protocol contract
- `docs/V3_EXECUTION_MANDATE.md` (32) — **conflict-resolution authority for V3**
- `docs/DECISIONS.md` (3001) — locked decisions log (largest; sample + spot-check)
- `docs/QUALITY_GATES.md` (254) — gate definitions / acceptance bars
- `docs/TESTING.md` (128) — test conventions
- `tests/test_repo_integrity.py` (docstring) — Committed-Truth guards G1-G6,
  AGENT-INTEGRITY-01 ("assert outcomes, not proxies")

Architecture layer:
- `docs/ARCHITECTURE.md` (924) — v2.X production baseline (intentionally historical)
- `docs/ARCHITECTURE_V3_DRAFT_0.5.md` (1546) — **V3.0 TARGET architecture (canonical target)**
- `docs/ARCHITECTURE_V3.1_CHARTER.md` (445) — **V3.1 AS-BUILT + roadmap (current reality, status-tagged)**

Current-state / index layer (Layer 1):
- `docs/PROJECT_STATUS.md` (434) — current task state
- `docs/README.md` (55) — docs index + three-layer model
- `docs/V3_DEFERRED_TESTS.md` (72) — deferred-test register

Treat `PLAN_*` files as Layer-2 execution docs: in scope only for whether their
*status claims* contradict Layer-0/Layer-1 reality, not for plan-internal design.

**Hard constraint — quarantine:** `docs/.archive/**` is blocked by `.aiignore`
and CLAUDE.md. **Do NOT read, cite, or recommend reading anything under
`docs/.archive/`.** A finding that requires archive content is out of scope.

---

## 2. Source-of-truth precedence (use this to resolve every conflict)

When two sources disagree, the authoritative one is decided by this order. Name
which source you treated as authoritative for each finding.

1. **The actual code/tests as they exist now** — for any claim about behavior,
   APIs, file boundaries, version numbers, config knobs, or test outcomes.
   Verify by reading/grepping; do not trust prose.
2. **`docs/V3_EXECUTION_MANDATE.md`** — for V3 governance conflicts between docs.
3. **`docs/ARCHITECTURE_V3.1_CHARTER.md`** — for *as-built* V3 reality.
4. **`docs/ARCHITECTURE_V3_DRAFT_0.5.md`** — for *target* V3 design only (it is a
   target; describing something as a goal there is NOT drift unless the charter
   or code already shipped it AND the draft contradicts that shipped reality).
5. Other Layer-0 docs (CLAUDE.md, AGENTS.md, DECISIONS.md, QUALITY_GATES.md).

`docs/ARCHITECTURE.md` is explicitly the **final v2.X baseline** and is *not*
expected to describe V3. Do not flag it as "stale" merely for omitting V3; flag
it only if it makes a claim that is now false, or if it fails to label itself as
the historical v2.X baseline.

---

## 3. Background context (so findings are concrete, not abstract)

Project: a PDF-to-JSONL multimodal ETL/RAG extraction pipeline. Two code
namespaces coexist: `src/mmrag_v2/` (v2.X production baseline) and
`src/mmrag_v3/` (V3 vision-native extraction, now the default path for batch PDF
processing).

Recently shipped (verify each against code before relying on it):
- **Fail-closed 3-tier extraction ladder** in `src/mmrag_v3/processor.py`
  `extract()`: tier 1 selected engine (default `MineruQwenHybridEngine` when
  `MINERU_ENDPOINT` is set, else legacy `HybridEngine`; force via
  `USE_MINERU_ENGINE`/`USE_VLM_ENGINE`/`USE_DOCLING_FAST`/`USE_HYBRID_ENGINE`/
  `USE_MINERU_QWEN_HYBRID`), tier 2 offline `DoclingFastEngine`, tier 3 PyMuPDF
  native-text "terminal" tier. Provenance is stamped on `doc.metadata.extra`
  (`extraction_engine`, `extraction_fallback`, `extraction_degraded_pages`,
  `extraction_recovered_pages`, `extraction_fallback_reason`). A page is treated
  as recoverable only when it has no visual element, no text content, AND a cheap
  PDF text-layer probe confirms text exists; genuinely blank pages cost nothing;
  the terminal tier never fabricates text.
- `batch_processor.process_pdf` delegates extraction to `mmrag_v3.extract()` and
  has ZERO docling imports; docling is confined to
  `src/mmrag_v3/engines/docling_fast.py`, enforced by the AST firewall in
  `tests/test_v3_security.py`.
- OmniDocBench Phase 0+1 fidelity benchmark (branch `feat/omnidocbench-phase0`):
  full English baseline text edit-distance 0.301 / TEDS 0.563; Phase 1 extractor
  bake-off **INCONCLUSIVE** (intermittent M5 server faults, not a fidelity verdict).

Versions (verify): engine/runtime `2.16.0` (`pyproject.toml` + `version.py`
`__engine_version__`); schema `2.7.0` (`__schema_version__`). These two are
DIFFERENT numbers on purpose; any doc that conflates them is a finding.

Known-bad state you must NOT treat as healthy:
- Exactly one **pre-existing, unrelated FAILING test**:
  `tests/test_v3_vlm_code_form.py::test_code_smuggles_as_text_promotes_to_code_modality`
  (a code-fencing whitespace assertion). Any doc that claims a fully green suite
  / "0 failures" without recording this is a finding.

Locked invariants the docs must state consistently (verify each against code;
flag any doc that states them differently or omits one a gate depends on):
- QA-CHECK-01 tolerance target is `0.10` for ALL profiles, no waivers.
- `bbox` emitted as integer `[0,1000]` coordinates.
- AGENT-SPATIAL-20: a single 20-unit vertical threshold, no profile branching.
- PDF batch size `<= 10` pages.
- `ProfileClassifier` is the router; the v2.4.2 `DocumentClassifier` approach is
  forbidden; `--profile-override` is debug-only.
- `ElementType` enum is frozen at exactly 3 members (TEXT/IMAGE/TABLE) per Charter
  §7.1; `Modality` is the 5-value chunker-boundary enum (TEXT/IMAGE/TABLE/CODE/
  FORM) reached via a "smuggle code/form as TEXT + promote at chunking" path.
- Python pinned `>=3.10,<3.11`; `docling` exact-pinned `2.86.0`.
- Acceptance requires `GATE_PASS` + `UNIVERSAL_PASS` across all document
  categories plus per-category blind docs; `scripts/smoke_production.sh` must
  print `SMOKE_PRODUCTION_PASS` for any change to the V3 extraction path.

Load-bearing engineering/governance rules (do not propose fixes that violate
these; if a real gap can only be fixed by violating one, flag it as
USER-DECISION-REQUIRED instead of recommending the violation):
- Libraries-first before custom heuristics; surgical changes only.
- No Docling construction in `batch_processor.py`.
- Three-layer docs model must be preserved; do NOT recommend deleting or merging
  the target draft, the as-built charter, or the v2.X baseline into each other.
- ASCII punctuation only in docs/code/config (no em-dash, no smart quotes).

---

## 4. The systematic biases of the document authors (hunt for these)

Most of these docs were written or last touched by the SAME agent that wrote the
code, in append-only bursts. Expect exactly the failure modes a self-author
produces:

- **Append-don't-reconcile:** a new dated section is prepended (e.g. a new
  "Current state") while the older contradicting section, headline banner, "Last
  updated" line, or a downstream "Known Design Debt"/"Remaining Work" list is left
  intact and now lies. Sweep WHOLE files, not just the top section.
- **Recency-only accuracy:** the newest subsystem (fail-closed ladder,
  OmniDocBench) is documented in one place and never reconciled into the
  architecture docs, invariant lists, or gate definitions that should reference it.
- **Optimistic status tags:** `SHIPPED` / `PARTIAL` / `DONE` tags that overstate
  reality; "0 regressions / all green" claims that paper over a known red test.
- **Target-vs-as-built confusion:** the 0.5 *target* draft and the 3.1 *as-built*
  charter describing the same subsystem in mutually exclusive ways without either
  labeling which is aspirational.
- **Definition drift on shared constants:** the same invariant (a threshold, a
  version, an enum cardinality, an engine name, a default route) stated with
  different values/wording in different docs and/or in code.
- **Self-leniency on its own architecture:** the authors will not have flagged
  high-level reasoning errors in their own solution design. **Expect to find at
  least 1-2 structural/conceptual problems no prior pass surfaced** — a guarantee
  stated more absolutely than the implementation delivers, a metric that cannot
  detect the failure it gates, a boundary contract that two docs describe
  incompatibly, or an invariant that the code already violates.

A purely additive audit (every finding is "add a sentence") means you dodged the
hard layer. At least some findings must be structural: a claim that must be
*retracted or restructured*, not merely supplemented.

---

## 5. Audit lenses (adversarial, concrete, mandatory)

Engage every lens. For each, either record findings or explicitly write "audited,
nothing to flag" with one line on what you checked. Cite file:line and quote the
exact text for every finding.

**L1 — Doc-vs-code factual verifiability.** For every concrete claim in the docs
(version numbers, file paths, function/enum names, env-var flags, default routes,
thresholds, "X has zero Y imports", "confined to file Z", test counts/outcomes),
verify it against the actual repo by reading/grepping. Sample aggressively in
DECISIONS.md (3001 lines — spot-check at least the V3 / extraction / version /
gate entries). A claim you cannot verify true is a finding; quote the contradicting
code with its path.

**L2 — Cross-doc contradiction (shared-constant drift).** For each locked
invariant in §3 and each "source of truth" constant, grep its statement across ALL
governance docs AND code, and confirm one consistent value/wording. Build a small
table: constant -> every location it is stated -> value at each -> authoritative
value. Any divergence (version 2.7.0 vs 2.16.0 conflation, a threshold, an enum
cardinality, a default-engine name, a batch-size, a Python/docling pin) is a
finding. Include tests that hard-code a value (e.g. an enum length assertion).

**L3 — Three-layer-model integrity (target vs as-built vs current).** Verify the
0.5 draft is unambiguously labeled the *target* and the 3.1 charter the
*as-built*, that README's layer assignments match how CLAUDE.md and the files
themselves present, and that no subsystem is described in two layers with
contradictory *as-shipped* claims. Specifically check: does anything the charter
marks `SHIPPED` still appear as a future goal in a way that reads as not-yet-done?
Does the 0.5 draft assert something as the V3 design that the code/charter already
superseded?

**L4 — Ship-gate-vs-governance-invariant completeness.** Grep the governance docs
(CLAUDE.md, AGENTS.md, DECISIONS.md, QUALITY_GATES.md) for "must" / "required" /
"invariant" / "acceptance" / "shall" / "MANDATORY". For each named gate, suite, or
invariant, verify it is actually enumerated in the acceptance/gate definitions
(`QUALITY_GATES.md`, the smoke/acceptance commands) AND still exists in code. Flag
any acceptance criterion that is asserted as mandatory in one doc but absent from
the gate checklist, or that names a script/flag/gate that no longer exists.

**L5 — Fail-closed-ladder documentation accuracy + guarantee honesty.** The
extraction reliability ladder is the newest load-bearing mechanism. Verify the
docs that describe extraction reliability (charter, project status, AGENTS,
DECISIONS) describe the ACTUAL 3-tier behavior and the ACTUAL provenance keys, and
that any "guarantee" wording matches what the code delivers and its honest scope
limit (text-layer pages only; scanned-OCR-failure explicitly out of scope; blank
pages not fabricated). A guarantee stated more absolutely than `extract()`
implements is a HIGH finding. Confirm the stamped key names in the docs match the
exact strings in `processor.py`.

**L6 — Acceptance-bar / metric validity + calibration.** For each numeric
acceptance bar in QUALITY_GATES.md / TESTING.md / DECISIONS.md (tolerances, TEDS,
edit-distance floors, R3/indent gates, gate pass thresholds): (a) does the metric
*structurally* measure the failure mode it gates (could the metric move when that
failure happens, by construction?), and (b) is the measurement instrument
documented anywhere as having calibration limits for the content class it scores?
Flag bars that are unachievable-by-design or that gate a failure the metric cannot
see. The OmniDocBench Phase-0 floor and any "fidelity acceptance" wording are prime
targets given Phase 1 came back INCONCLUSIVE.

**L7 — Exhaustiveness depth (what inventories silently omit).** Direct, specific
checks:
- Superseded-but-undocumented: a solution the docs still present as live that a
  newer mechanism replaced (e.g. any markdown post-processor, OCR auto-pilot,
  refiner, or "shared Docling pipeline factory" debt item that V3 made obsolete).
- In-tree opt-in / research flags described as if default, or default paths
  described as opt-in.
- Implicit governance intent: a doc says "we always do X" / "X is required" that is
  never actually tracked, gated, or true in code.
- Dead/placeholder references: a doc cites a script, module, command, endpoint, or
  fixture that does not exist (grep to confirm).

**L8 — Self-contained rationale (DECISIONS.md durability).** Sample 3-5 decision
entries (weight toward V3 / extraction / version / gate decisions). Each locked
decision must stand alone for a future maintainer: no "per chat", "see session",
"per Round-N" dangling references; the factual premises ("X-only", "engine Y does
not exist", "library Z lacks feature W") must be grep-verifiable against the repo.
Flag false premises and dangling references separately.

**L9 — High-level reasoning / solution-design errors (use full reasoning here).**
Step back from line-level checks and evaluate the *thinking*. Are the abstract
requirements coherent and mutually consistent? Does the high-level architecture
actually satisfy the stated requirements, or is there a latent contradiction
(e.g. a reliability guarantee vs a cost/"healthy-pays-nothing" claim that cannot
both hold; a "vision-native" default vs an invariant that assumes a text layer; a
boundary contract that two docs define incompatibly; an enum frozen at 3 while a
doc reasons as if it were 5)? Name any place where the documented solution would
not achieve the documented goal even if implemented perfectly.

**L10 — META: did the last reconciliation pass actually reconcile?** This doc set
was just edited to fix a version conflation, a stale status headline, and an
engine-name docstring. Check whether those edits LEFT contradictions elsewhere
(the append-don't-reconcile failure): does any other doc still carry the old
version, the old "current state", or the old engine-name claim? Did the fix to one
file create a new disagreement with an unedited file? If every recent fix was
additive and none retracted a now-false sentence elsewhere, say so — that is the
tell that the reconciliation was shallow.

---

## 6. Anti-escape-hatch rules (do NOT do these)

- **Do not** flag `ARCHITECTURE.md` as "stale" solely for being v2.X-scoped; it is
  the intentional historical baseline. Flag only false claims or a missing
  self-label as the v2.X baseline.
- **Do not** recommend deleting/merging the target draft, as-built charter, or v2.X
  baseline; the three-layer split is a deliberate constraint. If they contradict,
  the fix is to correct the *false* statement, not collapse the layers.
- **Do not** accept "this is just a wording nit" to avoid engaging a real
  contradiction; and conversely do not inflate pure prose polish to HIGH. Severity
  must track reader-harm.
- **Do not** assert a contradiction you have not verified by opening BOTH sources
  (and the code where relevant). Every doc-vs-code finding must quote the code.
- **Do not** recommend "defer" / "revisit later" / "add monitoring" as a fix for a
  factual error. A wrong sentence gets corrected or retracted now; name the exact
  replacement text.
- **Do not** propose a fix that violates a §3 locked invariant or load-bearing
  rule. If the only correct fix would, mark it USER-DECISION-REQUIRED and present
  the trade-off; do not silently recommend the violation.
- **Do not** read or cite `docs/.archive/**`.
- **Do not** pad to a count. If a lens is clean, say "nothing to flag". If you find
  3 substantive items, report 3.
- **Do not** conflate "the recommended fix is wrong" with "no problem here": if you
  reject a fix, state whether the underlying gap is still open.
- **Do not** treat your own single clean pass as proof a lens is clean — note where
  a second auditor from a different model family should double-check (especially L6
  and L9, which are judgment-heavy).

---

## 7. Required output format (a remediation report the next session can execute)

### 7a. Findings (one block each, ordered HIGH -> MED -> LOW)
For every finding:
- **ID:** F1, F2, ...
- **Severity:** HIGH (doc-vs-code contradiction, cross-doc contradiction in a
  contract/invariant, false guarantee, or high-level reasoning error) / MED (drift,
  staleness, an unlabeled target-vs-as-built, an unverifiable claim) / LOW (caveat,
  wording, ASCII-punctuation, dangling reference with no behavioral impact).
- **Location(s):** exact `file:line` for every source involved (quote the text).
- **Authoritative source:** which source is correct, per the §2 precedence, and why.
- **The conflict / error:** what is wrong, stated as the delta between sources.
- **Concrete failure mode:** what wrong action a future agent takes if they trust
  the bad doc.
- **Recommended fix:** the EXACT replacement text or edit, in which file, so the
  next session can apply it without re-deriving it. If it touches more than one
  file (drift across N docs), list every file that must change for consistency.
- **Confidence:** high/med/low, with what would raise it.

### 7b. Per-lens engagement table
One row per lens L1-L10: lens -> findings IDs (or "nothing to flag") -> one line on
what was checked. No lens may be silently skipped.

### 7c. Drift map (shared-constant consistency table)
For each §3 invariant/constant: every location it appears, the value/wording at
each, the authoritative value, and OK / DRIFT. This is the single most useful
artifact for the fix session — make it complete.

### 7d. Overall stance (calibrated; vague approval forbidden)
- Is the doc-vs-code correspondence trustworthy as-is, or not?
- Is the three-layer model internally consistent?
- Total HIGH / MED / LOW counts.
- The 2-3 most dangerous findings, in priority order.
- An ordered **remediation checklist** (do F-x, then F-y, ...) the next session can
  execute top-to-bottom, grouped so that a single shared-constant fix updates all
  its locations together.
- Explicit statement of any USER-DECISION-REQUIRED items.

---

## 8. Stopping rule and cadence

- Audit until you can make **two consecutive full passes over the inventory that
  surface 0 new HIGH-severity findings.** Until then, keep going.
- This is intended as **one of >=2 Round-1 audits from different model families.**
  Note where your verdict is judgment-dependent (L6/L9 especially) so a second
  auditor's disagreement can be read as signal, not noise. Where you and a future
  auditor would plausibly differ, say so and state which evidence would settle it.
- Do not stop at "looks consistent." The job is to find what the self-authoring
  process structurally hides; a clean report is only credible if §7b shows every
  lens was actually exercised against the code.
