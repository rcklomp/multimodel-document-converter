# Plan: v2.15 — Strategic Decision + Retrieval-Side Wins

**Status:** **Draft v0.9** (2026-05-24). Supersedes Draft v0.8
(same day). Round-8 audit returned **6 findings** against v0.8 —
**0 HIGH**, 4 MED (omlx-cosine threshold mis-calibrated for
short-sequence inputs at `max_tokens=30`, AST identifier-
intersection blind to severed `import` dependencies, standard
promotion arm's `open_user_issues` signal has no defined
collection source, defect-override 1% floor still admits chronic
death-spiral classes), 2 LOW (Abort Teardown Mandate triggered
only on 8-day cap not on acceptance failure, no programmatic
verification that teardown's 4 items actually landed). All 6
accepted and incorporated below.

**Stopping-rule progress — RULE FIRES**: HIGH-count progression is
now 2 → 2 → 0 → 1 → 0 → 1 → 0 → **0**. Round 8's 0 HIGH brings
the consecutive-clean count to **2**. The Section 9 rule requires
**two consecutive rounds with 0 HIGH** before execution — that
condition is **MET** as of Draft v0.9. **The plan is now
executable** pending the strategic-decision input from §8 Q1
(default Option F per silent-default clause at T-24h before
planned tag).

The round-8 findings are all next-layer refinements on already-
addressed structural surfaces from rounds 5-7 (the omlx-cosine
swap, the AST adequacy gate, the compound promotion gate, and the
Phase 2 Abort Teardown Mandate). None are ship-blockers; all are
absorbed in v0.9 without changing the executable phase set, and
the budget delta is negligible (+~$0.10/cycle for the diagnostic-
injection trigger on chronic-defect classes). The audit cadence
has reached genuine diminishing returns: the rounds 5→8 progression
shows the prior-round fixes propagating to ever-narrower failure
modes, and round 8 surfaces concerns that exist only as second-
order interactions of the fixes themselves.

### Round-8 audit changes folded into v0.9

| # | Severity | Finding | Where applied |
|---|---|---|---|
| 1 | MED | Phase 5c.0 omlx pairwise cosine at `max_tokens=30` inherits an unaudited threshold: the 0.85 gate was calibrated against the embedder's typical distribution on **retrieval-chunk-length text** (256-512 tokens), not 30-token snippets. Short-sequence inputs collapse toward a length-conditioned manifold attractor that inflates pairwise cosine independent of paraphrase quality. Either diverse paraphrases falsely route to cloud (gate over-triggers) or degenerate samples slip below the inflated short-text floor (gate under-triggers). | Phase 5c.0 spike gains a **calibration baseline pass** (~30 min curation + ≤5 min compute): hand-curate 5 anchor queries with known qualitatively-distinct vs near-identical paraphrase pairs; generate `n=5` quintuplets at `max_tokens=30` per anchor; record the cosine distribution per cluster. **Threshold is set to the midpoint between the two empirical clusters**, not the pre-stated 0.85. If clusters fail to separate at `max_tokens=30`, the `max_tokens ≥ 60` fallback ([PLAN_V2.15.md §5c.0 Fallbacks]) becomes the **primary** path, not the documented fallback. |
| 2 | MED | Phase 4 AST + identifier-intersection check ([PLAN_V2.15.md §4 Sub-check B]) validates that nearby prose mentions snippet-internal identifiers, but a snippet like `cache = LRUCache(maxsize=128)` parses, has matching identifiers (`cache`/`LRUCache`/`maxsize`), AND can still be **functionally unusable** if the `from x import LRUCache` upstream is severed outside the ±500-char window. RAG consumers copy the snippet; downstream user fails to import. The gate validates topical-relevance but not consumer-usability for the exact Fluent_Python failure mode it exists to guard against. | Extend Sub-check B with an **`Import` / `ImportFrom` resolution requirement**: walk the AST snippet for `Name` nodes whose parent is a `Call` and that are NOT defined locally (no preceding `FunctionDef`/`ClassDef`/`Assign` in the snippet) and NOT in `builtins.__dict__`. For each such "imported-from-elsewhere" name, require either (a) an `import` / `from … import` statement *inside* the snippet, OR (b) the name appears in the ±500-char prose window in a backtick code-span (e.g. `` `LRUCache` ``). ~15 lines of Python, $0. Closes the import-severance class without an LLM call. |
| 3 | MED | The compound promotion arm `hit-rate ≥ 5% AND (severe_defect_tag OR open_user_issues ≥ 1)` treats `open_user_issues` as a load-bearing input, but neither `analyze_doc_class_telemetry.py` nor `CYCLE_OPEN_CHECKLIST.md` defines a collection source, schema, or query path for it. The signal defaults to 0 indefinitely and the standard arm permanently degrades to `severe_defect_tag AND hit-rate ≥ 5%` — making `severe_defect_tag` the sole load-bearing input and unenforcing the popular-but-fine guard Round 5 Finding 1 introduced. | Add a defined source in `CYCLE_OPEN_CHECKLIST.md`: `open_user_issues = count of entries in docs/USER_ISSUES.md tagged with the doc-class name since prior cycle close`. **`docs/USER_ISSUES.md`** (NEW Phase 3 deliverable) is an append-only markdown table with columns `{date, doc_class, query, observed_behavior, expected_behavior}`. `analyze_doc_class_telemetry.py` parses it (greplike regex; no external service dependency). DECISIONS.md telemetry entry updated. Cycle-open checklist gains "Review `docs/USER_ISSUES.md` for new entries since prior tag" item. |
| 4 | MED | Defect-override 1% floor still admits the **chronic death spiral**: a class whose extraction has been visibly degraded for ≥1 cycle can have 30-day hit-rate <1% precisely because users have written it off entirely. The 1% floor was set as a corpus-frequency baseline, not as a failure-mode threshold. Telemetry can't distinguish "no one needs this class" from "everyone needed it once, got garbage, stopped" — and both produce ~5 hits / 30 days at the floor. | Add a **diagnostic-injection trigger** to `analyze_doc_class_telemetry.py`: for any class with `severe_defect_tag == True AND 60-day hit-rate < 1%`, the analyzer auto-generates 10 synthetic queries from that doc's v2.13/v2.14 weakest-query rationales (already enumerated in `docs/QUALITY_SNAPSHOT_*.md`) and writes them to `output/telemetry/diagnostic_injection_<class>_<date>.jsonl`. The next cycle's acceptance soak MUST run those queries; if R@1 < 30%, the class routes to **explicit A/E adjudication at cycle open** — same routing as middle-band aging. Cost: +~$0.10 (10 queries × cloud judge) per triggered class per cycle. Distinguishes "truly dead" from "actively avoided due to defect". |
| 5 | LOW | The Abort Teardown Mandate's preamble specifies a single trigger condition ("on 8-day cap trigger"). Phase 2 can produce the identical failed-cycle outcome via acceptance-bar miss at day 6, POC-gate failure, or user-rescind — none of which mandate teardown despite producing identical production state (no pdfplumber routing). A maintainer reading the plan could leave partial adapters in `engines/` after an acceptance-fail path. | Reword the Mandate trigger to **"any Phase 2 termination that does NOT result in pdfplumber routing being promoted to production"** — covers 8-day cap, acceptance-bar failure, POC-gate failure, and user-rescind uniformly. 6-word change to the Mandate header; closes the path-mismatch loophole. |
| 6 | LOW | The Mandate enumerates 4 cleanup items but no programmatic gate verifies they all landed before the v2.15.0 tag. Phase 4 abort has a hard programmatic backstop (15-query `partial_code` soak); Phase 2 abort relies on maintainer discipline — exactly the failure mode that introduced the `phase_0_expiration_date > today` boolean after near-misses. | Phase N DoD gains a line item: **"If Phase 2 was aborted, `scripts/verify_phase2_teardown.py` must report PASS"**. ~20-line script asserts: (a) no `pdfplumber` import in `src/mmrag_v2/engines/` non-experimental tree, (b) `engines/experimental/README.md` exists with the Mandate text, (c) `tests/test_pdfplumber_adapter.py` marked `@pytest.mark.skip`, (d) `docs/DECISIONS.md` contains the "v2.15 Phase 2 abort + teardown" entry. Same enforcement model as the calibration-freshness boolean. |

### Round-7 audit changes folded into v0.8

| # | Severity | Finding | Where applied |
|---|---|---|---|
| 1 | MED | Phase 1 German subgroup gate (`n=50, delta ≥ +10pp`) has ~21% false-positive rate under null. At n=50 ≈25 effective Bernoulli trials, P(net ≥ +5 flips \| null) ≈ 0.21 — a null-effect German HyDE passes the gate ~1/5 of the time by chance. Gate conflates significance with effect-size flooring. | Bump German subgroup **n=50 → n=100** (+$0.30 incremental). At n=100, single-flip noise = ±1pp; +10pp = 10 flips; P(10+ flips under null) ≈ 5%. Gate becomes statistically defensible. Phase 1 total n: 130 → 180. |
| 2 | MED | Phase 4 AST adequacy gate's ±500-char prose-proximity check confirms prose EXISTS near code but not that it's semantically RELEVANT. Cross-references ("See Chapter 14") and transition sentences pass the check while providing no useful context for RAG consumers needing parameter docs / usage examples / import provenance. | Augment proximity check with **lightweight AST-identifier intersection**: extract identifiers from the AST-validated snippet via `ast.walk` (function names, parameter names, top-level variable names from first 3 lines), require ≥1 to appear in the ±500-char prose window. Deterministic; ~10 lines of Python; $0. Closes the "cross-reference passes as context" loophole without adding an LLM call. |
| 3 | MED | `severe_defect_tag` is a manual flag. A newly-added v2.16+ doc class with severe extraction defects but no manual tag could enter telemetry → users abandon → hit-rate <1% → 60-day auto-close fires → permanently documented as Option E with no human ever having reviewed it. Diagnosis gap is largest exactly when class is newest. | Add **2-cycle grace period for new classes**: documented-limitation config gains `added_cycle` field; `analyze_doc_class_telemetry.py` skips auto-closure where `current_cycle - added_cycle < 2`. Class can still be explicitly closed by user decision, but the 60-day auto-close timer doesn't start ticking until the class has survived 2 full cycles with telemetry visibility. DECISIONS.md telemetry entry updated. |
| 4 | MED | T-72h `cycle_slip.log` trigger fires on **re-cal execution wall-clock >24h** but misses the case where re-cal runs fast (≤24h) and then human review of the verdict takes 36h. End-to-end blocking work slips >24h but slip log doesn't register → T-48h notification fires against stale original tag date. | Extend T-72h slip-log trigger: append entry if EITHER (a) re-cal wall-clock >24h, OR **(b) 24h after re-cal initiation, no `CALIBRATION_*_v2.15_p0_*.md` file with mtime > re-cal-start exists**. Single timer-fired file check; no continuous monitoring. Closes the re-cal-fast / review-slow path. |
| 5 | LOW | Phase 2 POC page selection ("3 representative pages; suggested 1, 6, 11") is manual. A confirmation-biased user wanting Option A could pick structurally easy pages where pdfplumber trivially passes ≥95% row alignment, greenlighting Option A regardless of pdfplumber's real capability on hard pages. | Add **page-selection rule**: 3 POC pages MUST include the page with highest column-count drift from v2.13/v2.14 Docling extraction (identifiable from existing CarOK weakest-query rationales or one-line grep of ingestion log). If column-drift data unavailable, use median-complexity page by table-cell count. Programmatic; no visual judgment. Removes cherry-picking path. |

### Round-6 audit changes folded into v0.7

| # | Severity | Finding | Where applied |
|---|---|---|---|
| 1 | MED | Compound promotion gate `hit-rate ≥5% AND pain-signal` creates a suppression death spiral for severely degraded classes — as retrieval degrades, users abandon, hit-rate drops below 5%, defect-tagged classes can never trigger F→A | Promotion rule gains a **severe-defect override arm**: `(hit-rate ≥ 5% AND pain-signal) OR (severe_defect_tag == True AND hit-rate ≥ 1%)`. 1% floor filters truly-dead classes while letting suppressed-volume defect classes trigger. Closure rule also extended: `severe_defect_tag == True` blocks auto-closure (a defective class with 0 queries can't be quietly closed). |
| 2 | MED | German subgroup `delta ≥ +5pp` at `n=30` still inside binomial noise: single flip = 3.33pp, 95% CI ≈ ±18pp | Bump German fixture **n=30 → n=50** (+$0.15 soak) AND raise effect-size floor **+5pp → +10pp** (≈5 query flips at n=50). Layered improvement per the auditor's proposal. |
| 3 | MED | Phase 5c.0 token-set Jaccard on `max_tokens=30` sequences is dominated by trivial syntactic glue tokens (stopwords, prompt echoes); high-frequency boilerplate deflates measured diversity independent of paraphrase quality | Replace token-set Jaccard with **pairwise cosine similarity using the production omlx embedder** (~80ms LAN, $0). Gate: `mean pairwise cosine ≤ 0.85`. Falls back to bigram Jaccard (≤0.45) or `max_tokens ≥ 60` if omlx unavailable. Uses the actual semantic representation the retrieval stack runs against. |
| 4 | MED | Phase 4 abort AST adequacy gate validates syntactic completeness of code snippet only — blind to severed imports, usage examples, or parameter prose context that RAG consumers need | Augment AST gate with **deterministic context-proximity check**: any AST-valid chunk must also contain ≥1 non-empty prose line within ±500 chars of the code fence boundary (regex on raw chunk text, $0). Co-gate both checks for ≥10/15 pass. |
| 5 | MED | Phase 2 8-day abort halts dev but leaves partial pdfplumber adapters / routing flags / schema-mapping stubs in `engines/` and `tests/` — zombie infrastructure consumes review bandwidth + risks accidental future activation | New **Abort Teardown Mandate** in Phase 2: on 8-day cap trigger, mandatory cleanup commit that (1) reverts routing to Docling-only fallback, (2) moves partial adapter to `src/mmrag_v2/engines/experimental/`, (3) applies `@pytest.mark.skip(reason="Phase 2 abort")` to test files, (4) logs exact state in DECISIONS.md. Abort isn't done until teardown lands. |
| 6 | HIGH | T-72h pre-tag checkpoint + T-48h silent-default notification both anchored to **static** planned tag date — if T-72h re-cal blocks for >24h, T-48h fires Option F auto-activation while maintainer is in plan-authorized blocking work | T-48h **decoupled from static date**: dynamic deadline rule = "if a T-72h blocking step fires, automatically extend T-48h notification deadline to `new_projected_tag_date - 48h`." `docs/CYCLE_OPEN_CHECKLIST.md` gains a `cycle_slip.log` tracker that requires explicit timestamp confirmation before the silent-default notification dispatch script runs. Closes the chronological race. |

### Round-5 audit changes folded into v0.6

| # | Severity | Finding | Where applied |
|---|---|---|---|
| 1 | MED | Telemetry ≥5% trigger measures corpus-frequency ("appears in top-5"), not pain-frequency ("appears in failed/degraded retrievals") — high-volume-but-acceptable classes could trigger Option A investment while lower-volume catastrophic-defect classes never cross the raw-volume threshold | Promotion rule becomes a **compound gate**: `hit-rate ≥5% AND (severe-defect-tag OR ≥1 open quality issue)`. Severe-defect-tag is a manual flag in the documented-limitation config (CarOK_voorraadtelling and Fluent_Python both qualify on entry). DECISIONS.md telemetry entry updated. |
| 2 | MED | German subgroup gate at n=20 is statistically fragile — single 20-query subgroup; 1-2 query flips swing direction; closure as dead-lever could be made on sampling noise | Phase 1 fixture: bump German subgroup fixture **20 → 30 queries** (+$0.25 soak) AND raise positivity threshold from "delta > 0" to **"delta ≥ +5pp"** (effect-size floor). Defense in depth — bigger n AND meaningful-effect requirement. |
| 3 | MED | Option F middle-band (1% ≤ rate < 5%) rolls forward "through next cycle" indefinitely — classes can live for 6 months in telemetry limbo with no escalation | DECISIONS.md telemetry entry gains a **persistence trigger**: ≥3 consecutive cycles in middle band escalates to **explicit A/E adjudication** at next cycle open (not auto-trigger; forces a user decision rather than another defer). |
| 4 | MED | Phase 4 abort-path soak validates downstream safety ("doesn't crash") but not user-visible adequacy ("can users actually answer code queries from truncated chunks?") — `partial_code` Fluent_Python could "pass" while being practically unusable | Abort soak gains an **adequacy gate**: of the 15 code queries against `partial_code` chunks, ≥10/15 must return a syntactically complete code block in top-5. Below 10/15, the defect is NOT safe to defer — routes to Phase 4 reopen or [[contract-violation-mode]] DECISIONS.md entry, same as the safety-gate failure path. |
| 5 | MED | Phase 5c.0 gates latency AND diversity but never defines the **post-implementation retrieval-lift** threshold required to justify the permanent multi-query overhead — could ship paraphrase fusion that passes 5c.0 but produces null R@1 lift | Phase 5c gains a **post-implementation effectiveness gate**: ≥3pp R@1 lift on the target fixture vs single-query baseline. Below 3pp, Phase 5c is reverted (opt-in flag stays but default-off; not promoted to production) or deferred. Latency/diversity gates pass = "safe to test"; effectiveness gate pass = "worth shipping." |
| 6 | LOW | Phase 2 [A] POC ("visually confirm columns separate cleanly") is subjective and bias-prone before the 5-7 day Option A commitment | POC exit criteria concretized: **≥95% row alignment on 3 representative CarOK pages AND successful emission of pdfplumber output into a valid `IngestionChunk` without downstream parser exceptions**. Programmatic check; no visual judgment call. |

### Round-4 audit changes folded into v0.5

| # | Severity | Finding | Where applied |
|---|---|---|---|
| 1 | HIGH | Phase 3 telemetry trigger defined but no reader/process — without it, v2.16 inherits a log nobody runs analysis on, F continues by inertia indefinitely (the "wait-and-see by another name" pathology Phase 3 was added to prevent) | Phase 3 [F] gains two hard deliverables: (a) `scripts/analyze_doc_class_telemetry.py` (computes per-class hit-rates over the 30-day window, emits `docs/TELEMETRY_REPORT_<date>.md` with trigger-fired booleans), and (b) `docs/CYCLE_OPEN_CHECKLIST.md` (new doc shipped in Phase N — line item "Run analyze_doc_class_telemetry.py; record findings in opening plan"). Closes the loop. |
| 2 | MED | Trichotomy asymmetric — F→A trigger defined (≥5%), F→E trigger absent. F is operationally biased toward A; classes <5% live in telemetry-purgatory forever | DECISIONS.md "v2.15 Documented-Limitation Telemetry Threshold" entry gains a complementary **<1% / 60-day auto-closure rule**: a class with hit-rate <1% over a 60-day window AND zero open user issues converts to Option E documented-limitation closure in the next cycle. Plan Phase 3 cross-references. F becomes a real fork, not a perpetual-defer state. |
| 3 | MED | Phase 1 narrowing lacks a falsifiable mechanism hypothesis — v2.14 broad-soak falsified; no articulation of why narrowing should produce a different result rather than reconfirm null | Phase 1 Goal gains explicit dilution hypothesis: "expect per-doc HyDE-on R@1 > HyDE-off on the 5 deficit docs IF dilution explains the broad-soak null." Falsification rule: per-doc lift null on ≥3/5 closes HyDE bridging as dead lever via DECISIONS.md entry rather than carry-forward. Phase 1 now has a termination condition, not just an acceptance gate. |
| 4 | MED | Phase 1 ≥4/5 directional gate is structurally blind to the German subgroup — 4 code-dense + 1 German fixture lets German-null pass cleanly while masking the v2.14-documented -12.5pp deficit | Phase 1 acceptance table replaces flat ≥4/5 with **subgroup-aware compound**: ATZ_Elektronik (German) MUST be positive AND ≥3/4 code-dense docs must be positive. German-null failure routes to explicit DECISIONS.md "German HyDE bridging null; deferred to v2.16" entry rather than aggregate burial. |
| 5 | MED | Phase 5c.0 spike measures latency but not sibling diversity — vLLM `n=5` at max_tokens=30 may return near-identical samples, passing latency gate while delivering ≈1 effective paraphrase for RRF fusion | Phase 5c.0 spike spec gains pairwise token-set Jaccard sub-check: mean pairwise Jaccard ≤ 0.7 across 5 siblings. Diversity failure routes to cloud `qwen-max` (stronger sampling at low max_tokens) or defers, same as latency-budget failure. Five-line addition; $0. |
| 6 | MED | Phase 4 abort path "potentially gated on Docling version bump" but no carry-forward trigger watches the Docling changelog — abort becomes effectively permanent until someone coincidentally notices a release | Carry-forward 6.1 gets a concrete trigger written into Section 4: **"Re-evaluate Phase 4 Approach 2 when Docling minor version ≥2.87 OR every 90 days, whichever first"**. Owner: `docs/CYCLE_OPEN_CHECKLIST.md` (shared with Finding 1's checklist). 5-min changelog check per trigger. |
| 7 | LOW | Silent-default-to-F at T-24h is biased against an E-preferring maintainer who is silent for reasons other than "F is fine" (offline/sick/traveling) | DoD silent-default clause gains a **T-48h notification step**: terminal banner / commit pre-push hook / Slack message (whichever channel maintainer reads) warning "Silent default to Option F in 24h; override by recording A/E/F selection in DECISIONS.md before T-24h." Lowers the probability that "silent = preferred" is the wrong inference. |
| 8 | LOW | Section 6 Budget counts cloud $$ precisely but treats local LLM as "$0 (LAN)" without surfacing aggregate GX10 wall-clock — contention with other workloads silently 2-3× phase durations and could trigger the 8-day cap on contention rather than merit | Section 6 gains a **GX10 wall-clock row**: Option A path ≈6h aggregate inference time; budget +50% for contention if endpoint is shared during cycle window. Reframes 8-day cap as wall-clock (includes contention) not pure engineering days. |

### Round-3 audit changes folded into v0.4

| # | Severity | Finding | Where applied |
|---|---|---|---|
| 1 | MED | Phase 1 ≥8pp lift on n=50 mini-soak is statistically fragile — single query flip ≈ 2pp at n=50 so 8pp is ≈4-query noise window | Phase 1 acceptance bumped to **n=100 (20/doc) AND ≥6pp aggregate R@1 lift AND ≥4/5 target docs show positive per-doc R@1 delta**. Cheaper than 3× soak; adds directional-consistency floor that rejects flukey single-doc aggregates. |
| 2 | MED | Phase 4 abort path leaves truncated code in production behind a `partial_code` flag whose downstream consumer behavior was never end-to-end validated | Phase N gains a **gated 15-query `partial_code` validation soak** that fires IF and only IF Phase 4 abort path was taken. Validates that judge scoring + retrieval downstream don't degrade or escalate parse failures on flagged chunks. Without this soak, the abort path is `partial_code`-flag-trusting (untested). |
| 3 | LOW | Calibration freshness boolean only fires at tag time — reactive gap if window expires mid-cycle | Phase N gains a **T-72h pre-tag checkpoint**: if `phase_0_expiration_date - 72h ≤ today`, auto-schedule `scripts/calibrate_local_judge_vs_qwen_max.py` as a blocking pre-tag step. Removes the late-discovery scramble. |
| 4 | LOW | DoD requires explicit A/E/F decision but provides no fallback if maintainer is silent — cycle stalls indefinitely | Section 8 + Phase N DoD gain a **silent-default clause**: if no explicit A/E/F selection is recorded T-24h before planned tag, **Option F auto-activates** (Phase 3 telemetry becomes a hard DoD requirement). F is the recommended path with the lowest commitment, so auto-defaulting to F is safe. |
| 5 | LOW | Option A $7 worst-case ignores iterative UIR-schema mapping cycles — likely 2+ iteration loops × ~$2/30-query mini-soak | Phase 2 method gains explicit **2-iteration schema-mapping cap** (~$4 cloud cost ceiling); further iterations roll into the 8-working-day hard abort trigger from v0.3 Finding 3 rather than spending more budget. |

### Round-2 audit changes folded into v0.3

| # | Severity | Finding | Where applied |
|---|---|---|---|
| 1 | HIGH | Phase 3 [F] `X%` telemetry threshold was deferred to v2.16 — guarantees v2.16 inherits a dataset with no decision rule attached | Phase 3 now defines `≥5% of logged queries in a 30-day rolling window, with rerank top-5 non-empty` as the per-class trigger for Option A treatment. Recorded as pre-cycle proposal in DECISIONS.md "v2.15 Documented-Limitation Telemetry Threshold". |
| 2 | HIGH | Phase 5c `<1500ms p50` budget mathematically impossible at measured 15 tok/s bare-config 14B-FP8 (n=5 × 30 tokens ≈ 2.0s wall-clock minimum; n=5 × 50 tokens ≈ 3.3s) | Phase 5c restructured with mandatory **measurement spike (Phase 5c.0)** before any implementation. Budget is set post-measurement; if measured p50 exceeds 4000ms, 5c is auto-routed to cloud `qwen-max` or deferred to v2.16. |
| 3 | MED | Section 5 "parallel-safe" framing for Phase 2 + Phase 4 (Option A) ignores single-engineer reality; serial wall-clock is the honest framing | Section 5 + Section 6 Budget: Option A explicitly capped at **8 working days serial**; "parallel branches" language replaced with "branches that can interleave but are time-additive in practice". Risks row added. |
| 4 | LOW | Item 3c (UIR refactor) has been PAUSED for user signoff since v2.11 — perpetually-paused = technical debt accumulating cognitive load | Section 4 carry-forwards: 3c row changes to **FORCE-CLOSED in Phase N**; user must either re-charter as a fresh architecture proposal with a concrete trigger or close the item. |
| 5 | LOW | Phase 6 freshness check fires "before any local-judge soak" but cycle could ship on a calibration that expired during admin-days | Phase N DoD adds explicit boolean: `Phase 0 calibration expiration date > today` at tag time. |

### Draft v0.2 → v0.3 also retains the Draft v0.1 → v0.2 deltas

Three changes drove the v0.1 → v0.2 revision (preserved here for
continuity):

1. **GX10 endpoint swap to RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic**
   completed 2026-05-23 PM (commit `53ffc73`). Phase 0 verdict reclaimed
   format-axis TRUSTWORTHY (90.7%) that the retired 27B-MTP regressed.
   Every Draft v0.1 reference to "27B-MTP is the GX10 ceiling" or
   "local 27B-MTP" is now stale.
2. **Phase 3 rollback drop already executed** (commit `2527414`,
   2026-05-23 PM) under explicit user "full send" override of the
   2026-06-19 time gate. The "Phase 1 (rollback drop)" listed in
   Draft v0.1 is moot — repurposed as a closed-out reference, not
   an active phase.
3. **Gemini audit round 1 (2026-05-24)** flagged: Option A budget too
   tight, Phase 4 regex approach too risky, Phase 5c latency multiplier
   under-quantified, Option F missing telemetry mechanism, Phase 2
   and Phase 5 must not run in overlapping soaks. All five accepted
   and incorporated in v0.2.

**Predecessor:** [`docs/PLAN_V2.14.md`](PLAN_V2.14.md) — CLOSED 2026-05-23
with 6 SHIPPED / 2 PARTIAL / 4 deferred. Production retrieval byte-
identical to v2.13.0 (purely additive local-LLM infrastructure).

**Owner:** ingestion + retrieval + LLM-integration pipeline.

---

## 1. Why this plan exists

### Thesis

v2.14 confirmed two trends that v2.15 has to respond to:

1. **The extraction-tuning lever is exhausted on the current Docling
   architecture.** Phase 1 (CarOK form-class) tried two cheap Docling
   config knobs (`do_cell_matching=True`, `force_full_page_ocr=True`),
   both regressed, and the third escalation (force-table-vlm) shipped
   the code path but its data was rolled back. Phase 6 (code-block
   chunking) shipped observability + a scanned-book improvement but
   the actual production defect (HybridChunker prose+code mixing on
   Fluent_Python) is upstream of where v2.14 could reach.

2. **The retrieval-side lever keeps paying.** Cumulative across
   v2.11 → v2.13: **+32pp Recall@1**. Cumulative extraction tuning
   across v2.11 → v2.14: **+6.2pp Format on one document
   (Earthship)** plus universal observability flags. The ROI ratio
   is roughly 5:1 in favor of retrieval-side work — and that's
   conservative because retrieval gains apply corpus-wide while
   extraction wins are typically single-doc.

### Post-v0.1 developments folded into v0.2 reasoning

- **Format-axis local judging reopened** on GX10 since the 14B-FP8
  swap. Draft v0.1 was written assuming the 27B-MTP's format 70.7%
  (NOT USABLE) as the operative reference and routed format judging
  to cloud `qwen-max`. The 14B-FP8's **90.7% TRUSTWORTHY** verdict
  means Phase 4b's format-only carve-out is back on the table for
  v2.15 sub-phases that need it (Phase 5a top-k tuning, Phase 5c
  paraphrase fusion). Cost implication: shifts judge cost back to
  $0 for format-axis work; cloud `qwen-max` still owns
  Relevance/Faithfulness final calls.
- **Throughput ceiling on the 14B-FP8 endpoint is bare-config.**
  n-gram speculative decoding was tested and rejected (6.3%
  acceptance; commit-context recorded in
  [[project-v2-14-ngram-spec-rejected]]). No same-family draft
  model exists with matching vocab below the 7B-class, so
  draft-model spec is also out. Steady-state ≈2.0s/judge call,
  ≈9s/HyDE call — relevant constraint for Phase 5c latency budget.

### Strategic question (unchanged from v0.1)

> Should v2.15 formalize specialized extraction lanes per document
> class (Option A), or accept tiered extraction quality and invest
> the engineering hours in retrieval / quality-of-life wins
> elsewhere (Option E)?

Both are defensible. Picking one with eyes open is the v2.15
opening move. **This draft does NOT pre-decide the question** —
it sketches both paths so the user can pick the direction before
the phase set finalizes.

### Non-goals

- **No GX10 model swaps.** The 14B-FP8 endpoint is the operative
  reference (commit `53ffc73`). The no-GX10-model-swap-reflex memory
  applies with full force. Re-evaluation only via offline-eval-first
  process per `scripts/calibrate_local_judge_vs_qwen_max.py` with
  OpenRouter bearer-key flag (already wired in v2.14).
- **No further speculative-decoding experiments on the GX10 14B-FP8
  endpoint.** n-gram rejected (negative result documented). Draft-
  model spec gated on same-family <7B with matching vocab=152064;
  no such model exists in the Qwen2.5 family.
- **No retrieval-stack architectural changes.** v2.13's stack
  (hybrid + RRF + ModernBERT rerank + omlx embedder) stays
  unchanged. v2.15 work within it (HyDE expansion, query rewriting,
  per-class tuning) is in scope; replacing it isn't.
- **No silent gate weakening.** Per [[contract-violation-mode]].
- **No third round of "let's try one more Docling knob" on CarOK.**
  The two cheap experiments already ran; v2.14 documented the
  outcome; further extraction work on CarOK is gated on the
  Option A/E decision below.

---

## 2. Strategic decision point — Option A vs Option E vs Hybrid

This is the **gating decision** for v2.15's phase set. The plan
files conditional phase sketches below for each path; the chosen
path becomes Draft v0.3's authoritative scope.

### Option A — Formalize specialized extraction lanes

Pick this if: you believe the CarOK / Fluent_Python / future-magazine
classes represent persistent corpus contribution worth fixing with
non-Docling tools, and you're willing to maintain a fleet of
extractors (Docling for the 90%, pdfplumber for inventory forms,
custom code-chunker for programming books, eventual VLM lane for
magazines).

**Concretely entails:**
- Taxonomic classifier extension (current `ProfileClassifier`
  doesn't catch CarOK; needs broadened form/inventory detection)
- pdfplumber parallel lane (or equivalent) for form-class
  documents — slot into the existing `IngestionChunk` schema with
  `extraction_method="pdfplumber_table"` or similar
- HybridChunker post-process or replacement for code-dense docs —
  per Phase 6.1 deferral from v2.14
- Maintenance cost grows with each new lane; QA framework needs to
  understand multi-tool routing

**Revised budget per Gemini audit:** v0.1's "2-3 days" for the
pdfplumber lane was optimistic. pdfplumber is a coordinate-based
text/line extractor; it does NOT natively output the UIR or the
`IngestionChunk` schema. Bridging requires:
- Cell-span handling for non-rectangular tables
- Image-region passthrough (pdfplumber has no image extraction —
  needs PyMuPDF or pdf2image companion for any cell with embedded
  imagery)
- Multi-page table joining (Docling does this; pdfplumber doesn't)
- HybridChunker downstream-consumer alignment: the semantic overlap
  manager + vision orchestrator expect the Docling chunk shape

**Realistic effort: 5-7 days minimum**, plus a UIR-schema-mapping
sub-task that must be explicitly scoped before implementation.

### Option E — Accept tiered quality, invest in retrieval

Pick this if: the data on extraction-vs-retrieval ROI (5:1
favoring retrieval) is dispositive, and you're willing to
explicitly document "this document class is best-effort" for the
hard cases rather than keep grinding extraction.

**Concretely entails:**
- DECISIONS.md entry naming the corpus tier (CarOK, certain
  programming books, magazines) as documented-limitation rather
  than ship-blocker
- Engineering hours shift to retrieval-side phases (Phase 2 HyDE
  bridging from v2.14 carryover; possibly query rewriting,
  per-class top-k tuning, multi-query expansion)
- Phase 6 partial-code observability shipped in v2.14 already gives
  downstream consumers a flag to surface "this chunk is severed" —
  no need to fix the severance if downstream can route around it
- Maintenance footprint stays smaller; QA framework unchanged

### Option F — Pragmatic hybrid (recommended default; updated v0.2)

Pick this if: you want the retrieval-side wins NOW (low-risk,
proven ROI) and want to keep the extraction-lane option open for
v2.16+ if specific documents prove load-bearing for actual user
queries.

**Concretely entails:**
- v2.15 ships the retrieval-side phases that work regardless
  (Phase 1 HyDE bridging — was Phase 2 in v0.1, renumbered; Phase 4
  calibration freshness — was Phase 6)
- Defer the Option A / Option E commitment to v2.16, when there's
  empirical evidence of which problem documents users actually query
  against
- **Document-class query telemetry (NEW per Gemini audit):** the
  "wait for empirical evidence" stance requires actually capturing
  the evidence. Add a thin logging pass: when a query's reranked
  top-5 includes any chunk from a documented-limitation document
  class (CarOK, Fluent_Python, …), log the query + class to a
  rolling file. Without this, "wait for v2.16 user-query evidence"
  is a sunk-cost trap by another name — there will be no evidence
  because we're not collecting it. Implementation: ~30 lines in
  `retrieve_hybrid_reranked` returning a `document_class_hit` field
  to the caller, plus a log line in the soak harness. Adds Phase 3
  to the unconditional set.
- Stop iterating on extraction in v2.15 entirely (no new pdfplumber
  lane, no HybridChunker rework) — let Phase 6's `partial_code`
  observability ride for one cycle to gather real-world signal,
  paired with the new query telemetry

**This is the path that most respects the no-swap-reflex / fix-at-
the-right-layer / contract-violation memories simultaneously.** It
ships visible wins on the retrieval side without committing to
architectural complexity that may not be necessary, AND it builds
the measurement instrument that makes "defer to v2.16" a real
decision rather than a wait-and-see.

---

## 3. Phases (proposed — conditional on strategic decision)

The phase list below has THREE tracks. Phases marked **[U]** are
unconditional and execute regardless of the Option A/E/F decision.
Phases **[A]** apply only if Option A or F-leaning-A is chosen.
Phases **[E]** apply only if Option E or F-leaning-E is chosen.
Phases **[F]** are introduced in v0.2 specifically for the Option F
telemetry mechanism.

**Phase numbering changed in v0.2:** v0.1's Phase 1 (rollback drop)
was executed pre-cycle (commit `2527414`); it's now recorded as
"Pre-cycle completed" below and not assigned a phase number.

### Pre-cycle completed (formerly v0.1 Phase 1) — dashscope-rollback drop

**Status:** ✓ COMPLETED 2026-05-23 PM (commit `2527414`) under
explicit user "full send" override of the 2026-06-19 time gate.

**Disk reclaimed:** ~30 GB across `mmrag_v2_8__qwen3_dashscope`
(31,371 pts, 1024-dim) and `mmrag_v2_8` (30,454 pts, 4096-dim).
Cold-storage snapshots persisted on Docker volume
`multimodal-doc-converter_qdrant_snapshots` (90-day retention).

**Carryover work:** none.

### Phase 1 [U or E] — HyDE bridging for code + minority-language queries

**(Was Phase 2 in v0.1.)** Carryover from v2.14 Phase 2 (was
redefined in v2.14 Draft v0.3 from per-doc embedder routing to
targeted HyDE bridging). Executes regardless of A/E path, but with
different priority weights.

**Goal:** address the omlx embedder's per-doc R@1 regressions on
German (`ATZ_Elektronik` -12.5pp) and code-dense content
(`Python_Cookbook` -12.4, `IRJET` -12.5, `Hybrid_electric_vehicles`
-12.6, `Greenhouse_Design` -12.5) **at query time**, reusing the
shipped Phase 4a local HyDE infrastructure.

**Falsifiable mechanism hypothesis (NEW in v0.5 per Round-4
Finding 3):** v2.14 Phase 2 falsified the *broad-query* HyDE lift
hypothesis at the corpus level. v2.15 Phase 1 retries on a narrower
5-doc subset under an explicit dilution-vs-no-lift discriminator:

- **Hypothesis H1 (dilution)**: broad-soak null was masking real
  per-doc lift on the 5 deficit docs because the >100 non-deficit
  docs in the broad soak diluted the signal. Narrowing the
  fixture to the 5 docs should surface lift if H1 is the true
  story.
- **Hypothesis H0 (no lift)**: HyDE is fundamentally inert on
  these doc classes. Narrowing should reconfirm the broad-soak
  null. **In which case HyDE bridging is closed as a dead lever**
  via a DECISIONS.md entry "HyDE bridging dead-lever; not carried
  to v2.16" rather than becoming yet another perpetually-deferred
  carry-forward.
- **Falsification rule**: if per-doc R@1 lift is null (delta ≤ 0)
  on ≥3 of the 5 target docs, H0 is reconfirmed; Phase 1 ships a
  closure entry rather than a success entry. Phase 1 has a
  termination condition, not just an acceptance gate.

**Method:** see v2.14 PLAN Phase 2 detail. Summary:
1. Lightweight query-intent classifier (regex/heuristic; lang-id;
   code-keyword density) — deterministic, no LLM call. (ALREADY
   SHIPPED v2.14 P2 as opt-in; was FALSIFIED on the v2.14 broad
   soak. v2.15 re-targets it to the 5 specific docs above with a
   narrower mini-soak.)
2. When `intent=code` or `intent=minority_language`: enable
   targeted HyDE (Phase 4a local) with content-aware system prompt
3. Wire `hyde_provider` through `retrieve_hybrid_reranked`
4. Mini-soak: **n=180 queries** on the 5 affected docs (REVISED
   in v0.8 per Round-7 Finding 1): **100 for ATZ_Elektronik**
   (German subgroup, bumped from 50 in v0.7 — and from 30 in v0.6,
   from 20 in v0.4 — for statistical defensibility; n=100 brings
   single-flip noise to ±1pp, putting +10pp = 10 flips well
   outside the ~5% binomial false-positive rate under null) +
   **20 per code-dense doc × 4** (Python_Cookbook, IRJET,
   Hybrid_electric_vehicles, Greenhouse_Design). HyDE-off baseline
   vs HyDE-on test arms over the identical fixture.

**Isolation requirement (NEW per Gemini round-1 audit):** Phase 1
mini-soak MUST be executed and evaluated BEFORE any Phase 5
retrieval-tuning sub-phases. Running them in the same soak creates
confounding variables and makes per-axis attribution impossible.

**Acceptance (REVISED in v0.4 per Round-3 Finding 1 — statistical
rigor):** the v0.3 single-metric `≥8pp aggregate R@1 lift` gate on
n=50 was statistically fragile. At n=50 a single query flip is
±2pp, so ≥8pp is only ≈4 query flips — comfortably within the
binomial-variance noise floor for R@1 measurements on small
fixtures. v0.4 replaces it with a compound gate:

| Gate | Threshold | Rationale |
|---|---|---|
| Aggregate R@1 lift (HyDE-on vs HyDE-off, same fixture) | **≥6pp** | Lower bar than v0.3's 8pp because n=100 cuts the per-query noise floor in half; 6pp on n=100 is meaningfully outside the binomial CI |
| Per-doc directional consistency (REVISED in v0.8 per Round-7 Finding 1 — fourth iteration on German subgroup statistical power) | **Subgroup-aware compound**: `ATZ_Elektronik` (German subgroup, **n=100**) MUST show R@1 delta **≥ +10pp** AND **≥3/4 code-dense docs** must show positive R@1 delta (delta > 0) | v0.7's `n=50 + ≥+10pp` had ~21% false-positive rate under null per Round-7's binomial analysis (≈25 effective trials at baseline R@1 ≈ 57.5%; P(net ≥ +5 flips \| null) ≈ 0.21 via normal approximation). v0.8 bumps n=50→100; single-flip noise = ±1pp; +10pp = 10 flips; P(10+ flips under null) ≈ 5%. Gate finally on defensible statistical ground after four iterations. Code-dense gate keeps "delta > 0" because the 4-of-4 quorum provides directional-consistency robustness within that subgroup. |
| Format axis (judge: local FP8-14B, TRUSTWORTHY) | no regression | HyDE shouldn't introduce format defects; cheap check |
| Faithfulness axis (judge: cloud `qwen-max`) | no regression beyond -1pp | small variance budget for measurement noise |

All four gates must pass to ship. Aggregate-only or per-doc-only
wins are rejected as insufficient signal.

**Subgroup-failure routing (NEW in v0.5 per Round-4 Finding 4):**
if the German subgroup gate fails (ATZ_Elektronik delta ≤ 0) but
all other gates pass, do NOT ship aggregate as success. Either:
(a) iterate the HyDE system prompt for `intent=minority_language`
within Phase 1's wall-clock budget, OR (b) ship a closure with an
explicit DECISIONS.md entry "German HyDE bridging null; deferred
to v2.16" naming the subgroup-specific failure. Aggregate burial
is rejected.

**Cost:** ~$3-4 mini-soak (cloud `qwen-max` for rel + faith
axes on n=180; v0.8 +$0.30-0.50 over the n=130 v0.7 budget for
the German subgroup bump 50→100). Format axis runs on local
FP8-14B (TRUSTWORTHY) at $0.

### Phase 2 [A] — Specialized extraction lane for form-class documents

**Skip this phase entirely if Option E or F chosen.**

**(Was Phase 3 in v0.1.) Budget revised per Gemini audit: 5-7 days
not 2-3.**

**Goal:** wire pdfplumber as a parallel extraction lane for
documents that today produce structurally-degraded Docling table
output. Initial scope: CarOK_voorraadtelling; expand as evidence
warrants.

**Method:**
1. Extend `ProfileClassifier` with form-class detection that catches
   CarOK (12-page native_digital with dominant
   `extraction_method=docling_table_markdown` chunks showing
   per-page column-count drift). Possibly: extend `QUALITY_GATES.md`
   §"Form / Invoice Acceptance Class" classifier rule.
2. **UIR-schema mapping sub-task (explicit per Gemini audit).**
   pdfplumber output → `IngestionChunk` shape requires:
   - Cell-span handling for non-rectangular table regions
   - Image-region passthrough via PyMuPDF or pdf2image companion
     (pdfplumber alone has no image extraction)
   - Multi-page table joining (Docling does this natively;
     pdfplumber doesn't)
   - HybridChunker downstream-consumer alignment so the semantic
     overlap manager + vision orchestrator don't choke on the
     foreign chunk shape
   Estimate this sub-task at 2-3 days alone before any
   pdfplumber-specific work.
3. New extraction lane in `engines/`: pdfplumber-based table
   extractor returning chunks in the existing `IngestionChunk`
   schema with `extraction_method="pdfplumber_table"`,
   `chunk_type="table"`. Adapter pattern parallel to
   `DoclingPdfAdapter`.
4. Routing logic in `batch_processor.py` or
   `strategy_orchestrator.py`: when profile is form-class, run
   pdfplumber lane first; fall back to Docling if pdfplumber emits
   zero rows.
5. Re-extract CarOK; validate via 30-query mini-soak that "odd
   whitespace + truncation" rationales disappear from weakest-15.
6. Document the new lane in `ARCHITECTURE.md` + `DECISIONS.md`.

**Acceptance:**
- CarOK Format score recovers ≥85% on the existing prose-Format
  rubric on a 30-query mini-soak
- pdfplumber lane chunks pass the existing 34/34 strict-gate
  matrix (no regression on the corpus baseline)
- New `tests/test_pdfplumber_adapter.py` covers boundary cases +
  bridge tests proving the routing flag flows through

**Cost:** ~$0.50 per 30-query mini-soak (cloud qwen-max judge) +
pdfplumber CPU time. **Schema-mapping iteration cap (Round-3
Finding 5):** UIR-schema mapping rarely lands on first try; budget
allows **up to 2 iteration cycles** at $2 each ($4 ceiling for
schema-mapping soaks). If iteration 2 still fails acceptance, the
phase consumes its remaining 8-day wall-clock budget without
spending further cloud cost — additional iterations are a
time-cost not a money-cost issue, gated by the day-8 hard abort
trigger. This prevents cost overruns from masking real engineering
overruns.

**Engineering effort: 5-7 days serial** (revised from 2-3 in v0.1;
hard cap 8 working days per Round-2 Finding 3).

**Decision input for the user:** before implementation begins, a
one-shot pdfplumber proof-of-concept against the CarOK PDF runs
with **concrete exit criteria** (REVISED in v0.6 per Round-5
Finding 6; v0.5's "visually confirm the columns separate cleanly"
was subjective and bias-prone before the 5-7 day commitment).

**POC exit criteria — all must pass:**

1. **Row alignment**: on 3 representative CarOK pages,
   pdfplumber extraction produces tables where ≥95% of rows are
   correctly column-aligned against the source PDF. Measured
   programmatically: compare extracted-row column count against
   expected column count (5 for CarOK's inventory layout); rows
   with mismatched count are misalignments. **95% floor across
   the 3 sample pages combined.**

   **Page-selection rule (REVISED in v0.8 per Round-7 Finding 5
   — removes cherry-picking path)**: 3 POC pages are NOT user-
   picked. They are selected programmatically:
   - **Page 1 (mandatory)**: the page with **highest column-
     count drift** from the v2.13/v2.14 Docling extraction —
     identifiable from existing CarOK weakest-query rationales
     in `docs/QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_omlx.md`
     and `docs/PROJECT_STATUS.md` v2.14 P1 mini-soak data, OR
     from a one-line grep of the ingestion log for column-count
     anomalies. If column-drift data is unavailable, fall back
     to the median-complexity page by table-cell count (count
     `<td>`-equivalent extraction events per page).
   - **Pages 2-3**: random sample from the remaining pages,
     fixed seed = 42 for reproducibility.

   The mandatory hard-page coverage prevents a
   confirmation-biased user from greenlighting Option A on
   structurally easy pages (e.g. title page + simple header
   tables) while pdfplumber's real failure modes on inventory-
   form core pages remain untested.

2. **Schema compatibility**: pdfplumber output emits into a valid
   `IngestionChunk` via the planned adapter without raising any
   `ValidationError` or downstream parser exception. Test:
   ingest the 3 sample-page tables, run them through the
   existing `IngestionChunk` validator + the chunker downstream
   path that semantic-overlap-manager / vision-orchestrator
   consume. **Zero exceptions across the 3 pages.**

If EITHER criterion fails, Option A Phase 2 is null and the
strategic decision routes to Option E (documented limitation) or
Option F (telemetry-defer to v2.16). The POC takes ≤2 hours; its
$0 cost is trivial against the 5-7 day commitment it gates.

No subjective "looks mostly right" greenlight path.

**Abort Teardown Mandate (NEW in v0.7 per Round-6 Finding 5;
trigger broadened in v0.9 per Round-8 Finding 5):**

The 8-day wall-clock abort trigger (from Round-2 Finding 3) halts
development but, without explicit cleanup, leaves partial
infrastructure in the production codebase — pdfplumber adapters,
routing flags, schema-mapping stubs, failing or skipped tests.
This zombie state consumes review bandwidth indefinitely, increases
cognitive load on every future PR review touching `engines/`, and
risks accidental activation in future cycles (e.g., a v2.17
contributor sees the routing flag and assumes it works).

**On any Phase 2 termination that does NOT result in pdfplumber
routing being promoted to production** (REVISED in v0.9 per
Round-8 Finding 5; was "on 8-day cap trigger" in v0.7-0.8) — this
covers (a) 8-day wall-clock cap, (b) acceptance-bar failure at any
day, (c) POC-gate failure, (d) user-rescind — **the cycle MUST
include a teardown commit before close-out** that does ALL of the
following:

1. **Revert routing logic to Docling-only fallback**: the form-
   class detection in `ProfileClassifier` and the routing branch
   in `batch_processor.py` / `strategy_orchestrator.py` revert to
   their pre-Phase-2 state. Any pdfplumber-related call sites
   return to plain Docling extraction. Verifiable via git diff
   against the cycle-open commit.
2. **Move partial adapter code to `src/mmrag_v2/engines/experimental/`**:
   any pdfplumber-related adapter files in `engines/` move to a
   new `engines/experimental/` subdirectory with a README noting
   "v2.15 Phase 2 abort; preserved for v2.16+ if Option A
   reconsidered." `engines/__init__.py` does not import from
   `experimental/`. Code stays in tree (no destructive deletion)
   but is structurally isolated from production paths.
3. **Apply `@pytest.mark.skip(reason="Phase 2 abort; see DECISIONS.md")`
   to `tests/test_pdfplumber_adapter.py`** and any other
   pdfplumber-touching test files. Skipped (not deleted) so the
   tests remain as documentation of intended behavior + are
   trivially re-activatable in a future cycle.
4. **Log exact teardown state in `DECISIONS.md`**: new entry
   "v2.15 Phase 2 abort + teardown" recording: day-N when abort
   fired, what work was incomplete, which files moved to
   `experimental/`, which tests are skipped, and the explicit
   re-evaluation trigger (per carry-forward 6.1: Docling minor
   ≥2.87 OR every 90 days). The DECISIONS.md entry is the
   permanent record; v2.16+ contributors should see it before
   touching any `experimental/` content.

The 4-item teardown is itself ~2-4 hours of work and is **within
the 8-day cap** (the cap is the abort trigger, but teardown is
part of the abort itself — abort isn't done until teardown lands).

**Why moving to `experimental/` rather than deleting**: the abort
is a "this lever didn't work this cycle" signal, not a "this lever
is fundamentally wrong" signal. The Docling-version-trigger
re-eval in carry-forward 6.1 may re-open the question; preserving
the partial work in `experimental/` makes resumption cheaper than
re-deriving from scratch, while structural isolation prevents
accidental promotion to production.

### Phase 3 [F] — Document-class query telemetry (NEW in v0.2; threshold defined in v0.3)

**Applies only if Option F is chosen. Skip under Option A or E.**

**Goal:** instrument the retrieval pipeline so that "defer to v2.16
when there's user-query evidence" becomes a measurable proposition
rather than a wait-and-see. Without this phase, Option F silently
becomes Option E by neglect.

**Method:**
1. Maintain a JSON-list of "documented-limitation" document classes
   (initial entries: `CarOK_voorraadtelling`, `Fluent_Python`, and
   any other docs called out in `DECISIONS.md` under §Documented
   Limitations). One config file under `src/mmrag_v2/retrieval/`.
2. In `retrieve_hybrid_reranked`, after the reranker output, compute
   `document_class_hits: List[str]` enumerating which limitation-
   class docs appear in the top-5. Return as a new field on the
   retrieval result object.
3. Soak harness writes a one-line JSON record per query: `{query,
   timestamp, document_class_hits, rerank_top_5_doc_ids,
   rerank_top_5_non_empty: bool}` to a rolling log at
   `output/telemetry/document_class_hits.jsonl`. Rotate weekly,
   retain indefinitely (small file; query text + doc IDs only).
4. **v2.16 decision rule (set in v0.3 per Round-2 Finding 1;
   pain-signal coupling + middle-band aging added in v0.6 per
   Round-5 Findings 1 + 3):**
   - **Denominator**: queries logged in the 30-day window preceding
     v2.16 cycle open where `rerank_top_5_non_empty == True`.
   - **Per-class hit-rate**: `(queries with class in top-5) / denominator`.
   - **Promotion trigger (F→A, REVISED in v0.7 per Round-6
     Finding 1 — adds defect-tag override arm; `open_user_issues`
     source defined in v0.9 per Round-8 Finding 3; chronic-defect
     diagnostic injection added in v0.9 per Round-8 Finding 4)**:
     a class earns Option A treatment in v2.16 when ANY of:
     - **Standard arm**: `hit-rate ≥ 5%` over 30 days AND
       `severe_defect_tag == True OR open_user_issues ≥ 1`
       (the v0.6 compound gate; addresses popular-but-fine), OR
     - **Defect-override arm (NEW in v0.7)**: `severe_defect_tag
       == True AND hit-rate ≥ 1%` over 30 days, OR
     - **Chronic-defect adjudication arm (NEW in v0.9 per
       Round-8 Finding 4)**: `severe_defect_tag == True AND
       60-day hit-rate < 1% AND diagnostic-injection R@1 < 30%`.
       This arm fires explicit A/E adjudication at next cycle
       open (not auto-promotion to A); the routing matches
       middle-band aging, but the input is the auto-generated
       diagnostic-injection soak rather than 3 consecutive
       middle-band cycles. See Phase 3 §6 deliverable below for
       the injection mechanism.

     **`open_user_issues` source (NEW in v0.9 per Round-8
     Finding 3)**: defined as the count of entries in
     **`docs/USER_ISSUES.md`** tagged with the doc-class name and
     timestamped since the prior cycle close. `USER_ISSUES.md` is
     an append-only markdown table (NEW Phase 3 deliverable; see
     §7 below) with columns `{date, doc_class, query,
     observed_behavior, expected_behavior}`. The analyzer parses
     it via regex (no external service dependency). Without this
     specification, `open_user_issues` would default to 0
     indefinitely and the standard arm would collapse to
     `severe_defect_tag AND hit-rate ≥ 5%`, unenforcing the
     popular-but-fine guard Round 5 Finding 1 introduced.

     The defect-override arm closes the **suppression death
     spiral**: a severely degraded class's users abandon queries,
     hit-rate falls below 5%, the v0.6 compound `AND` gate would
     never fire promotion despite the known defect. The 1% floor
     (3× below the corpus-frequency baseline of ~3.1%) filters
     truly-dead classes while letting suppressed-volume defect
     classes trigger explicit adjudication. The chronic-defect
     adjudication arm closes the **sub-1% chronic spiral** that
     the 1% floor itself created (Round-8 Finding 4): a class
     with `hit-rate < 1%` can be either "truly dead" (no real
     demand) or "actively avoided due to defect" (latent demand
     suppressed by the defect); the diagnostic-injection R@1
     measurement is what distinguishes the two cases. Both entry
     classes (CarOK_voorraadtelling, Fluent_Python) qualify on
     entry because both have documented extraction defects from
     v2.13/v2.14 cycles.
   - **Closure trigger (F→E, NEW in v0.5 per Round-4 Finding 2;
     defect-tag protection added in v0.7 per Round-6 Finding 1;
     new-class grace period added in v0.8 per Round-7 Finding 3)**:
     a class with hit-rate **<1% over a 60-day window** AND zero
     opened user issues against it AND `severe_defect_tag == False`
     AND `current_cycle - added_cycle >= 2` converts automatically
     to Option-E documented-limitation closure in the next cycle.

     The `severe_defect_tag == False` clause (v0.7) prevents the
     death-spiral pattern on the closure side: a heavily-defective
     class that's been so bad users stopped querying it shouldn't
     be quietly closed at <1% hit-rate.

     The **new-class grace period** (v0.8 per Round-7 Finding 3)
     closes the symmetric failure mode for newly-added classes:
     a v2.16+ corpus addition with severe extraction defects but
     no manual `severe_defect_tag` (because nobody's diagnosed it
     yet) could enter telemetry → users abandon → hit-rate drops
     below 1% → 60-day auto-close fires → class is permanently
     documented as Option E with no human ever reviewing it.
     The 2-cycle grace period (`added_cycle` field in the
     documented-limitation config) ensures the auto-close timer
     doesn't start until the class has survived ≥2 cycles with
     telemetry visibility — giving human-in-the-loop review a
     chance to apply `severe_defect_tag = True` if warranted
     before the closure path silently fires. Classes can still
     be explicitly closed by user decision during the grace
     window; only the automatic path is gated.

     Severe-defect-tagged classes can only exit telemetry-
     tracking via promotion to Option A, or via explicit user
     adjudication (remove the defect tag because the user decides
     it's no longer load-bearing).
   - **Middle band aging (NEW in v0.6 per Round-5 Finding 3)**:
     1% ≤ rate < 5% rolls forward through the next cycle, BUT if
     a class persists in middle band for **≥3 consecutive cycles**,
     it escalates to **explicit A/E adjudication** at next cycle
     open — forces a user decision rather than another defer.
     Prevents indefinite telemetry-limbo for moderately-painful
     classes.
   - **Why these thresholds**: see DECISIONS.md "v2.15 Documented-
     Limitation Telemetry Threshold" entry.

**Hard deliverables (NEW in v0.5 per Round-4 Finding 1):**

The v0.4 plan defined the trigger but specified no reader/process
to fire it — making the entire telemetry phase theatrical. v0.5
closes the loop with two artifacts:

5. **`scripts/analyze_doc_class_telemetry.py`** — reads
   `output/telemetry/document_class_hits.jsonl` AND
   `docs/USER_ISSUES.md` (NEW input in v0.9 per Round-8
   Finding 3), applies the 30-day and 60-day windows, computes
   per-class hit-rates, applies the 2-cycle grace period (NEW in
   v0.8 per Round-7 Finding 3), AND (NEW in v0.9 per Round-8
   Finding 4) emits diagnostic-injection synthetic queries to
   `output/telemetry/diagnostic_injection_<class>_<date>.jsonl`
   when the chronic-defect-spiral condition is met (see below).
   Emits `docs/TELEMETRY_REPORT_<date>.md` with explicit
   trigger-fired booleans per class:
   ```
   ## CarOK_voorraadtelling
   - added_cycle: v2.15  (current: v2.18 → grace_period_elapsed: True)
   - severe_defect_tag: True
   - 30-day hit-rate: 7.2% (37 / 514 qualified queries)
   - 60-day hit-rate: 6.8%
   - open_user_issues: 0 (source: docs/USER_ISSUES.md grep for "CarOK_voorraadtelling")
   - PROMOTION TRIGGER (standard arm: ≥5% AND pain-signal): FIRED
   - PROMOTION TRIGGER (defect-override arm: defect-tag AND ≥1%): FIRED
   - CHRONIC-DEFECT ADJUDICATION (defect-tag AND <1% AND injection R@1 <30%): NOT FIRED (hit-rate ≥1%)
   - CLOSURE TRIGGER (<1% AND 0 issues AND no defect-tag AND grace elapsed): NOT FIRED
   - v2.16 disposition: Option A treatment (extraction-lane investment)
   ```

   **Diagnostic injection mechanism (NEW in v0.9 per Round-8
   Finding 4)**: when a class has `severe_defect_tag == True AND
   60-day hit-rate < 1%`, the analyzer:
   - Parses that doc's weakest-query rationales from the
     `docs/QUALITY_SNAPSHOT_*.md` files (already enumerated per
     doc as part of v2.13/v2.14 baseline reporting)
   - Selects 10 representative rationales spanning the doc's
     failure modes
   - Writes them as synthetic queries to
     `output/telemetry/diagnostic_injection_<class>_<date>.jsonl`
     in the same line-schema as the production retrieval log
   - Adds a `CHRONIC-DEFECT INJECTION QUEUED` line to the
     telemetry report
   - The injection queries MUST be included in the next cycle's
     acceptance soak; if R@1 < 30%, that class routes to
     **explicit A/E adjudication at cycle open** (same routing
     as middle-band aging). If R@1 ≥ 30%, the class converts to
     normal Option-E closure at the next 60-day window (latent
     demand is not severe).

   Pure-Python; no LLM calls in the analyzer itself; the
   diagnostic-injection soak (~10 queries × cloud qwen-max) costs
   ~$0.10 per triggered class per cycle. Test coverage:
   `tests/test_telemetry_analyzer.py` (mock-driven; ≥10 tests
   covering all three promotion arms + closure with defect-tag
   protection + closure with grace-period protection + middle
   band + middle-band-aging escalation + chronic-defect injection
   trigger + chronic-defect injection skipped when hit-rate ≥1%
   + zero-data edge).

6. **`docs/CYCLE_OPEN_CHECKLIST.md`** — new doc shipped in Phase N.
   Contains a line item per cycle-open trigger AND the
   `cycle_slip.log` tracker (NEW in v0.7 per Round-6 Finding 6):
   - "Run `scripts/analyze_doc_class_telemetry.py`; copy
     trigger-fired booleans into opening plan's Carry-Forwards
     table" (closes Round-4 Finding 1)
   - "Review `docs/USER_ISSUES.md` for new entries since prior
     tag; verify counts feed the analyzer's `open_user_issues`
     input correctly" (NEW in v0.9 per Round-8 Finding 3)
   - "If any class shows `CHRONIC-DEFECT INJECTION QUEUED` in
     the telemetry report, include
     `output/telemetry/diagnostic_injection_<class>_<date>.jsonl`
     in this cycle's acceptance soak; record R@1 in the cycle's
     opening plan" (NEW in v0.9 per Round-8 Finding 4)
   - "Check Docling release notes since last cycle; if minor
     version ≥2.87, reopen Phase 4 evaluation per carry-forward
     6.1" (closes Round-4 Finding 6 — shared checklist artifact)
   - "Phase 0 calibration freshness check: `phase_0_expiration_date
     - 72h ≤ today`? If yes, schedule re-cal as blocking
     pre-Phase-1 step" (was Phase 6 + DoD; checklist makes it
     proactive)
   - **`cycle_slip.log` tracker (NEW in v0.7 per Round-6
     Finding 6)**: append-only text-file log at
     `docs/cycle_slip.log` (per-cycle file rotated by tag) that
     records any T-72h or other blocking-step that extends the
     effective tag date. Format per line:
     ```
     {iso_timestamp} | {trigger_name} | elapsed_hours={N} |
     old_tag_date={D} | new_projected_tag_date={D'}
     ```
     The silent-default notification dispatch script (Phase N
     DoD) reads this log to compute `effective_tag_date`. Empty
     log = use original planned tag date. Multiple slips compose:
     latest `new_projected_tag_date` wins.
   - Other cycle-open items as they accumulate in future cycles.

   The checklist becomes the single owner of "things that should
   fire each cycle-open" — removing the failure mode where deferred
   items rot because no process watches them. The `cycle_slip.log`
   is the single source of truth for "is the tag date still where
   we planned" — used by both the silent-default notification and
   any other static-date-anchored process.

**Acceptance:**
- New field on retrieval result; downstream callers don't break
- Bridge test in `tests/test_retrieval_pipeline.py` proving the
  field is populated for a known-limitation doc
- Log file appears on first soak run; rotation works
- `analyze_doc_class_telemetry.py` ships with test coverage
  (including the diagnostic-injection trigger path per Round-8
  Finding 4)
- **`docs/USER_ISSUES.md` ships as empty seed file** with the
  schema header `| date | doc_class | query | observed_behavior |
  expected_behavior |` (NEW in v0.9 per Round-8 Finding 3); the
  file is append-only, parsed by the analyzer, and is the single
  defined source for the `open_user_issues` signal
- `docs/CYCLE_OPEN_CHECKLIST.md` ships with the 5 cycle-open
  items listed above (REVISED in v0.9 per Round-8 Findings 3 + 4;
  was 3 items in v0.7-0.8)
- DECISIONS.md entry reflects the threshold rules in force
  (proposal → active, with all promotion arms AND closure rule)
  before tag

**Cost:** $0 baseline; **+~$0.10 per triggered class per cycle**
for diagnostic-injection soak (NEW in v0.9 per Round-8 Finding 4;
fires only when `severe_defect_tag AND 60-day hit-rate < 1%`;
zero classes trigger in v2.15 because both entry classes are
above 1%). Engineering effort: ~1.5 days (was ~1 in v0.5-0.8;
+0.5 for diagnostic-injection mechanism + USER_ISSUES.md seed +
additional test coverage per Round-8 Findings 3 + 4).

### Phase 4 [A] — HybridChunker post-process for code-dense documents

**Skip this phase entirely if Option E or F chosen.**

**(Was Phase 4 in v0.1. Approach 1 demoted to abort-path per Gemini
audit.)**

**Goal:** fix the production "truncated code" defect on Fluent_Python
and similar Python tutorials by intervening between Docling's
HybridChunker output and the IngestionChunk emission stage.

**Method:** v2.14 Phase 6 investigation found the actual defect
shape is Docling-extraction-layer prose+code mixing, NOT classical
CODE+CODE severance.

**Approach 1 — REJECTED:** regex/heuristic splitting at prose/code
boundary. Per Gemini audit: "Writing a parser to second-guess
Docling's chunker based on colons and code-keywords will result in
massive false-positive rates on standard technical prose." Any
prose with colons (Python type hints `x: int`, dictionary syntax in
prose, etc.) or code-keywords (`for`, `if`, `def`, …) could trigger
spurious splits. Lenient-judge trap analog at the chunking layer.
Confirmed dead path; do NOT fall back to it.

**Approach 2 — ONLY VIABLE PATH:** Docling configuration tuning.
Preserve cleaner CodeItem boundaries by configuring the
HybridChunker differently (specific tokenizer / merging policy
that better separates CodeItem from neighboring prose). Per
[[libraries-first]]. Depends on Docling 2.86 supporting the needed
options.

**Hard gate (NEW in v0.2 per Gemini round-1 audit):** if a one-day
spike proves Docling 2.86 does NOT support cleaner CodeItem boundary
configuration natively, Phase 4 is ABORTED and the defect is
deferred to v2.16 (potentially gated on a Docling-version bump).
NO regex/heuristic fallback.

**Abort-path validation gate (NEW in v0.4 per Round-3 Finding 2):**
the v0.2 abort path "trusted" the existing `partial_code` schema
flag (shipped in v2.14 P6) to provide downstream consumers with the
information they need to handle truncated code chunks gracefully.
That trust was never end-to-end validated. If Phase 4 ABORTS, the
truncated-code defect stays in production behind a flag whose
downstream behavior is untested — violating the "fix at the right
layer" principle by masking with untested observability.

**Therefore: an abort triggers a mandatory Phase N
`partial_code` downstream-validation soak** (15 queries against
docs with `partial_code=True` chunks). Three gates must all pass:

1. **Safety gate** (v0.4 original): Retrieval pipeline must not
   error or escalate when a `partial_code` chunk wins a top-5
   slot. "Doesn't crash."
2. **Judge-axis gate** (v0.4 original): Judge scoring on
   `partial_code` chunks must show no systematic Format/Faith
   degradation vs neighboring full-code chunks (no
   hallucination-from-truncation signal). "Doesn't visibly
   degrade quality scores."
3. **Adequacy gate (NEW in v0.6 per Round-5 Finding 4; context-
   proximity sub-check added in v0.7 per Round-6 Finding 4)**: of
   the 15 code queries, **≥10/15 (67%) must return a chunk
   passing BOTH sub-checks in their top-5**:

   - **Sub-check A — Syntactic completeness**: programmatic
     check via a small AST parser (Python `ast.parse` for Python
     chunks; analogous lib for other languages). Validates the
     code snippet would be parseable / runnable on its own.
   - **Sub-check B — Context proximity + relevance + import
     resolution (REVISED in v0.8 per Round-7 Finding 2;
     import-resolution added in v0.9 per Round-8 Finding 2)**:
     the AST-valid chunk must satisfy ALL THREE:
     - **structurally near**: ≥1 non-empty prose line within
       **±500 characters** of the code fence boundary (regex
       on raw chunk text; $0), AND
     - **semantically relevant**: that prose line contains
       **≥1 identifier from the AST-validated snippet**.
       Identifiers are extracted via `ast.walk` on the validated
       snippet: function names, parameter names from function
       signatures, and top-level variable names from the first
       3 lines of the code block. Intersection check against
       the ±500-char prose window. $0; deterministic; ~10 lines
       of Python, AND
     - **imports resolved (NEW in v0.9 per Round-8 Finding 2)**:
       for any `Name` node in the AST whose parent is a `Call`
       and which is NOT defined locally (no preceding
       `FunctionDef` / `ClassDef` / `Assign` in the snippet)
       and NOT in `builtins.__dict__`, require EITHER:
       (a) an `import` / `from … import` statement *inside* the
       snippet that binds the name, OR
       (b) the name appears in the ±500-char prose window
       inside a backtick code-span (e.g. `` `LRUCache` ``;
       regex `` `[A-Za-z_][A-Za-z0-9_]*` ``).
       ~15 lines of Python, $0; closes the import-severance
       failure mode (snippet `cache = LRUCache(maxsize=128)`
       parses + identifies + topically matches "uses caching",
       but is unusable to a RAG consumer without the upstream
       `from functools import lru_cache` / `from my_lib import
       LRUCache` that was severed outside ±500 chars).

   The relevance sub-check addresses the v0.7 gap: structural
   proximity alone is not relevance. A chunk with AST-valid
   Python and a nearby prose line "See Chapter 14 for details" or
   "As discussed in §3.2, these patterns recur throughout the
   standard library" would pass proximity-only but fail
   identifier-intersection (no overlap with the code's actual
   identifiers). Closes the "cross-reference passes as context"
   loophole without adding an LLM call.

   The import-resolution sub-check (v0.9) addresses the
   Fluent_Python failure mode this gate was originally built for:
   the HybridChunker routinely splits at section boundaries that
   sever the import block from code that depends on it.
   Identifier-intersection on snippet-internal identifiers
   passes such chunks; import-resolution catches them.

   All three components of Sub-check B (proximity AND relevance
   AND import-resolution) AND Sub-check A (AST validity) must
   pass for a chunk to count toward the ≥10/15 threshold.
   Co-gating addresses the layered gap: safety alone isn't
   adequacy, syntactic completeness alone isn't context,
   contextual proximity alone isn't relevance, and topical
   relevance alone isn't consumer-usability.

If ANY of the three gates fails, abort is INVALID — the defect
must be addressed in v2.15 (re-open Phase 4 with sign-off for an
alternate approach) or DECISIONS.md gets an explicit "shipping
with known `partial_code` adequacy gap" entry under
[[contract-violation-mode]] discipline.

Cost: ~$0.30 (15-query qwen-max soak); the adequacy gate adds
$0 (programmatic syntactic-completeness check, no extra LLM
calls). Gated — only fires IF abort path taken.

**Acceptance:**
- Zero "truncated code" rationales in a 30-query mini-soak on the
  affected Python docs
- Chunk count growth ≤+15% per doc
- New tests cover the prose+code boundary cases in addition to the
  existing 9 in `tests/test_code_chunking.py`

**Cost:** ~$0.50 mini-soak + chunker CPU time + 1-day spike for
Approach 2 viability check, +1-2 days implementation if viable,
+0 days if aborted.

### Phase 5 [E] — Retrieval-side investments

**Skip this phase entirely if Option A chosen.**

**Goal:** with extraction tuning deprioritized, allocate the
engineering hours to retrieval-side phases with demonstrated ROI.

**Isolation rule (NEW per Gemini audit):** any Phase 5 sub-phase
MUST run its mini-soak SEPARATELY from Phase 1's HyDE bridging
mini-soak. Same fixture is fine and recommended (apples-to-apples);
overlapping the changes in a single soak run is not.

**Candidate sub-phases (pick a subset based on appetite):**

- **5a) Per-class top-k tuning.** Current production uses uniform
  `top_k=25` candidates per leg. A small per-doc-class sweep may
  show wins. ~30-query mini-soak per class. Uses cloud `qwen-max`
  for rel/faith per the Phase 4 FORBIDDEN-uses list (retrieval-
  breadth tuning belongs to cloud judging); local FP8-14B now
  available for the format axis at $0.
- **5b) Query rewriting pipeline.** Beyond HyDE: rewrite ambiguous
  or under-specified queries with the local FP8-14B before
  retrieval. Latency cost: +~2-3s per query at single-stream
  bare-config (no spec decoding — n-gram rejected, no draft-model
  available); gating: opt-in flag like HyDE.
- **5c) Multi-query / RAG-fusion expansion.** Expand each query
  into 3-5 paraphrases, retrieve for each, RRF-fuse results. Known
  technique; non-trivial implementation; potentially large recall
  lift.

  **Latency physics (Round-2 Finding 2):** the bare-config FP8-14B
  endpoint measures ≈15 tok/s steady-state. Naive sequential
  generation of 5 paraphrases at ~30 tokens each = 5 × 2.0s = 10s
  per query — unworkable. vLLM `n=5` sampling parallelizes the
  forward pass and yields all 5 sequences at roughly per-step
  cost ≈ 67ms (decode is memory-bandwidth-bound, not compute-
  bound at this n); so 30-token paraphrase × n=5 ≈ 30 × 67ms ≈
  2.0s wall-clock. The Draft v0.2 budget of <1500ms p50 was
  aspirational; per the round-2 audit's correct math, 2.0s is
  the bare-config floor for 30 tokens × n=5 — meaning v0.2's
  budget was set below the achievable floor.

  **Phase 5c.0 — Mandatory measurement spike (NEW in v0.3; sibling-
  diversity sub-check added in v0.5 per Round-4 Finding 5):**
  before any 5c implementation begins, run a one-shot probe
  against the live FP8-14B endpoint that issues a realistic
  paraphrase prompt with `n=5, max_tokens=30` × N=20 trials,
  reads `latency.p50` and `latency.p99` from the response timing,
  computes pairwise sibling diversity (see below), and writes
  a one-line report to
  `docs/SPIKE_<date>_v2.15_p5c0_paraphrase_latency.md`.

  **Latency gate** — compare measurement against three brackets:

  | Measured p50 | Action |
  |---|---|
  | ≤ 2500ms | Phase 5c proceeds with budget = `measured p50 + 50% headroom` (interactive-acceptable) |
  | 2500-4000ms | Phase 5c proceeds with budget = `measured p50 + 25% headroom` AND opt-in flag only (no default-on for interactive paths) |
  | > 4000ms | Phase 5c **routes paraphrase generation to cloud `qwen-max`** (interactive-acceptable at ~1.0-1.5s; cost ~$0.0002/query) OR defers to v2.16. User picks at spike-completion. |

  **Sibling-diversity gate (NEW in v0.5 per Round-4 Finding 5;
  metric replaced in v0.7 per Round-6 Finding 3)**:
  vLLM `n=5` at low `max_tokens` can return near-identical samples
  (low effective sampling diversity at short sequence length).
  Latency-passing-but-diversity-failing would ship a phase whose
  RRF fusion sees ≈1 effective paraphrase — the recall lift would
  be null while the wall-clock cost was already paid.

  v0.5 used token-set Jaccard but that metric is biased on short
  sequences: at `max_tokens=30`, the token set is dominated by
  trivial syntactic glue (stopwords, prompt echoes, boilerplate)
  whose shared presence between siblings inflates Jaccard
  similarity even when the semantic paraphrase is high-quality.
  Small denominator size `|A ∪ B|` magnifies the weight of shared
  trivial tokens.

  v0.7 replaces token-set Jaccard with **pairwise cosine
  similarity using the production omlx embedder** —
  `Qwen3-Embedding-8B-mxfp8` via the LAN endpoint
  `http://10.0.10.246:8000/v1/embeddings`. Uses the actual
  semantic representation the retrieval stack runs against.
  Cost: ~80ms LAN per sibling-embedding call, $0.

  **Calibration baseline pass (NEW in v0.9 per Round-8
  Finding 1)**: the v0.7 threshold of 0.85 was set by analogy to
  the embedder's typical distribution on **retrieval-chunk-
  length text** (256-512 tokens), not 30-token snippets. Short-
  sequence inputs collapse toward a length-conditioned manifold
  attractor that inflates pairwise cosine independent of
  paraphrase quality. **A pre-stated threshold against the wrong
  reference distribution may either over-trigger (cloud-routing
  diverse-but-short paraphrases) or under-trigger (passing
  degenerate-but-not-quite-identical samples).**

  Before the spike's main n=20 trials begin, run a **5-minute
  calibration baseline pass**:
  1. Hand-curate 5 anchor queries (~30 min one-time work;
     reused for all future spikes) with a known **diverse**
     paraphrase quintuplet (e.g., 5 syntactically distinct
     rewrites) AND a known **near-identical** paraphrase
     quintuplet (e.g., 5 trivial whitespace / synonym swaps).
  2. Encode each quintuplet via the omlx endpoint at
     `max_tokens=30` truncation; compute pairwise cosine
     per quintuplet.
  3. Record the empirical distribution: `diverse_cluster_p95`
     (95th percentile of pairwise cosine across all 5 diverse
     quintuplets) AND `near_identical_cluster_p5` (5th
     percentile across all 5 near-identical quintuplets).
  4. **Operative threshold for this spike = midpoint between
     the two cluster boundaries**, NOT the pre-stated 0.85.
     If clusters fail to separate (i.e., `diverse_p95 >
     near_identical_p5`), the omlx-on-30-token approach is
     **falsified at the embedder layer** — the spike auto-
     promotes the `max_tokens ≥ 60` fallback (see below) to
     **primary path**, not fallback.

  The calibration baseline is appended to the spike report at
  `docs/SPIKE_<date>_v2.15_p5c0_paraphrase_latency.md`. Once
  the 5-anchor quintuplet set is curated, future re-runs of the
  spike reuse the same anchors → calibration becomes a 5-min
  compute step.

  | Measured mean pairwise cosine | Action |
  |---|---|
  | ≤ calibrated threshold | Diversity acceptable; sibling fusion can produce meaningful recall lift |
  | > calibrated threshold | Diversity insufficient; siblings are semantically near-equivalent. Route to cloud `qwen-max` (stronger sampling at low max_tokens) OR defer to v2.16. Same routing as latency-bracket failure. |

  The 0.85 figure remains a **fallback default** ONLY if the
  calibration baseline cannot run for resource reasons (e.g.,
  omlx down during the spike → see fallback section). It is no
  longer the operative threshold for omlx-available paths.

  **Fallbacks if omlx embedder is unavailable** (e.g. omlx-server
  down during the spike) OR if the calibration baseline shows
  clusters failing to separate at `max_tokens=30` (Round-8
  Finding 1's falsification path):
  - **`max_tokens ≥ 60` with omlx cosine** (PROMOTED in v0.9
    from "fallback" to "primary path on calibration-failure" per
    Round-8 Finding 1): re-run the calibration baseline at the
    longer sequence length; the larger context window typically
    resolves the length-attractor compression
  - **Bigram Jaccard** with threshold ≤ 0.45 (filters stopwords
    by requiring shared bigrams; less biased than token-set on
    short sequences) — used only if both `max_tokens ≥ 60`
    omlx calibration AND `max_tokens = 30` omlx calibration
    both fail to separate clusters

  Primary metric is omlx cosine at the calibration-derived
  threshold; longer-sequence omlx is the calibration-failure
  fallback; bigram Jaccard is the embedder-unavailable fallback.

  **Cost:** spike alone is $0 + ~10 min wall-clock. The Jaccard
  computation adds ≤5 lines of Python. Implementation cost depends
  on bracket outcomes.

  **Phase 5c gate:** v0.2's "hard latency budget of <1500ms p50"
  is REPEALED in v0.3 as physically infeasible at the floor.
  Budget is now measurement-derived (latency + diversity), not
  aspiration-set.

  **Phase 5c post-implementation effectiveness gate (NEW in v0.6
  per Round-5 Finding 5):** the 5c.0 spike validates that
  paraphrase fusion CAN run within budget (latency) and CAN
  produce non-degenerate siblings (diversity). It does NOT
  validate that it actually improves retrieval. Without an
  effectiveness gate, 5c could ship paraphrase fusion that
  passes 5c.0 but produces ≈null R@1 lift while every interactive
  retrieval path permanently absorbs 2-4s extra latency — a
  per-query tax for no measurable benefit.

  **Post-implementation requirement**: after 5c implementation,
  run a 100-query mini-soak on the same fixture used for Phase 1
  (n=110 sample is reusable here — same docs, same intent
  distribution). Compare single-query R@1 vs paraphrase-fusion
  R@1 over identical retrieval stack settings.

  | Measured R@1 lift | Action |
  |---|---|
  | ≥ +3pp | Ship 5c with default-on for affected query intents |
  | +1pp to +3pp | Ship 5c **opt-in flag only**; don't default-on (lift is positive but below the "worth the latency tax" floor) |
  | ≤ +1pp | **Revert 5c** — code stays in tree behind an opt-in flag for opt-in research use, but is NOT promoted to production. DECISIONS.md entry naming the null result. |

  Cost: ~$2 (100-query mini-soak with cloud qwen-max for rel +
  faith axes). Format axis on local FP8-14B at $0. Worth the
  spend — it's the only way to know whether the wall-clock cost
  is justified.
- **5d) Documented-limitation policy.** DECISIONS.md entry naming
  the corpus tier (CarOK + others identified by the v2.13/v2.14
  weakest-query data) as best-effort rather than ship-blocker.
  Cheap to do; high signal-to-noise. **Note for Option F:** this
  entry is also a prerequisite for Phase 3 [F]'s telemetry
  config — Phase 3 reads its document-class list from the
  DECISIONS.md entries.

**Acceptance:** depends on which sub-phases ship; each carries its
own acceptance criteria.

**Cost:** $2-5 across multiple mini-soaks if 5a/5b/5c all ship;
5d is $0.

### Phase 4 → Phase 6 renumber — Calibration freshness check (Phase 6 [U])

**(Was Phase 6 in v0.1; v0.4 adds a T-72h pre-tag checkpoint per
Round-3 Finding 3.)**

**Goal:** per the v2.14 Phase N policy added in Draft v0.5: if
>30 days have elapsed since the last Phase 0 calibration run OR
the GX10 model has changed since, re-run the calibration before
any local-judge soak in v2.15.

**Status check at v2.15 open (updated v0.2):** Phase 0 FP8-14B cal
SHIPPED 2026-05-23 PM. 30-day window expires **2026-06-22**. If
v2.15 close-out is before that date AND the GX10 endpoint is
unchanged, no re-cal needed. The retired 27B-MTP cal verdict is
NOT operative — do not reference it as a baseline.

**T-72h pre-tag checkpoint (NEW in v0.4 per Round-3 Finding 3;
slip-aware in v0.7 per Round-6 Finding 6):**
v0.3 had two gates — freshness check at cycle start + DoD boolean
at tag time. That left a reactive gap: a calibration that's
fresh at cycle-start can expire mid-cycle, and discovery at
tag-time creates a scramble (emergency re-cal blocks the tag or
tempts a silent gate-weaken). v0.4 added an intermediate gate:

```
72 hours before the planned tag date, check:
  if phase_0_expiration_date - 72h <= today:
    record re_cal_start_timestamp = now()
    auto-schedule scripts/calibrate_local_judge_vs_qwen_max.py
    as a BLOCKING pre-tag step
    update operative docs/CALIBRATION_*.md reference
    update phase_0_expiration_date forward (today + 30d)

    # NEW in v0.7 per Round-6 Finding 6;
    # extended in v0.8 per Round-7 Finding 4 to cover the
    # re-cal-fast / review-slow case:
    SLIP TRIGGER fires if EITHER:
      (a) re-cal wall-clock execution > 24h, OR
      (b) at re_cal_start_timestamp + 24h, no
          docs/CALIBRATION_*_v2.15_p0_*.md file exists with
          mtime > re_cal_start_timestamp
          (single timer-fired file check — verifies the verdict
          file has actually landed; absent file = human review
          is the bottleneck, not re-cal compute)

    when SLIP TRIGGER fires:
      compute new_projected_tag_date = now() + 48h
        # 48h buffer from "we noticed the slip" forward; gives
        # the maintainer time to complete review without the
        # auto-activation breathing down their neck
      append entry to docs/cycle_slip.log:
        {trigger: "T-72h slip (cause: re-cal>24h | verdict-file-missing)",
         re_cal_start: T0,
         detected_at: T1,
         old_tag_date: D,
         new_projected_tag_date: D'}
      → DoD silent-default notification deadline reads from
        cycle_slip.log and uses new_projected_tag_date - 48h
        rather than original_tag_date - 48h
```

The 72h buffer is sized for the calibration run itself
(~90 min local on n=518) plus result review + DECISIONS.md
update (~half day) plus a contingency margin. Removes the
late-discovery pattern.

The **slip-aware extension** addresses the round-6 HIGH: without
it, a >24h re-cal would slide past the static T-48h deadline,
causing Option F to auto-activate mid-blocking-work. The
`cycle_slip.log` is the single source of truth for the
notification dispatch script — any plan-authorized blocking work
that slides the tag date must register there.

**Cost:** $0 (local LLM only). Pre-tag re-cal cost the same — $0.
`cycle_slip.log` is an append-only text file managed by the
checklist.

### Phase N — Cycle close-out

- Engine version bump `2.14.0` → `2.15.0` (or `2.15.1` if
  v2.14.1 patch tag was cut for the GX10 swap, TBD by user)
- v2.15 retrieval-regression fingerprint (re-capture if Phase 1,
  Phase 2, Phase 3, or Phase 5 changed production retrieval shape
  or chunk_ids; otherwise v2.14 fingerprint stays canonical)
- AFTER snapshot `docs/QUALITY_SNAPSHOT_<date>_v2.15_after.md`
- Layer-0/1 docs sweep per [[doc-sanitization-completeness]]
- Archive v2.15 Draft v0.1 + v0.2 archaeology block
- Archive overdue v2.14 Draft v0.1 archaeology block (carry-over)
- v2.15.0 annotated tag staged for user push

**Definition of Done (minimal bar for v2.15.0 tag):**

A v2.15.0 tag MAY ship when ALL of the following are true:

- ✓ Phase 1 (HyDE bridging) shipped OR documented as "evidence
  showed null effect, deferred"
- ✓ Phase 3 [F] telemetry shipped IF Option F is the chosen path
  (without it, Option F is functionally Option E by neglect)
- ✓ All shipped phases pass their own acceptance bars
- ✓ Full pytest suite green; v2.14 fingerprint still passes (or
  fresh v2.15 fingerprint captured if production retrieval changed)
- ✓ Strict-gate corpus state unchanged (34/34 PASS) or improved
- ✓ Strategic decision (Option A/E/F) recorded in DECISIONS.md with
  evidence — see "silent-default clause" below for what happens
  if no explicit selection lands.
- ✓ **Silent-default clause (NEW in v0.4 per Round-3 Finding 4;
  T-48h notification added in v0.5 per Round-4 Finding 7):**

  **T-48h notification step (slip-aware in v0.7 per Round-6
  Finding 6 HIGH)**: the deadline is **dynamic, not anchored to
  the original tag date**. The notification dispatch script reads
  `docs/CYCLE_OPEN_CHECKLIST.md` `cycle_slip.log` and computes:
  ```
  effective_tag_date = max(original_tag_date,
                           latest cycle_slip.log new_projected_tag_date)
  notification_fires_at = effective_tag_date - 48h
  ```
  This prevents the v0.6 race: T-72h pre-tag re-cal that takes
  >24h slides `effective_tag_date` forward, and the T-48h
  notification correspondingly waits.

  48 hours before the **effective** tag date, surface a
  notification via whichever channel the maintainer actually reads
  (terminal banner on next shell open / commit pre-push hook /
  Slack message). The notification text:
  > "v2.15 silent default to Option F in 24h (effective tag date:
  > {effective_tag_date}; slip due to: {latest cycle_slip.log
  > trigger if any}). Override by recording an A/E/F selection in
  > `docs/DECISIONS.md` before T-24h. Selection format: see
  > PLAN_V2.15.md §8 Q1."

  **T-24h auto-activation (also slip-aware)**: if no explicit
  A/E/F selection is recorded by `effective_tag_date - 24h`,
  **Option F auto-activates**, making Phase 3 telemetry a hard
  DoD requirement. The auto-activation gets a one-line
  DECISIONS.md entry naming the default-trigger rather than the
  explicit decision, so v2.16 has the audit trail.

  Rationale: F is the recommended path with the lowest commitment
  cost; auto-defaulting prevents indefinite Phase N stall on
  indecision/overload. The T-48h notification step lowers the
  probability that "silent = preferred F" is the wrong inference
  (the maintainer could be silent because offline/sick/traveling,
  not because F is the preferred path). The slip-aware deadline
  prevents auto-activation while plan-authorized blocking work
  is in flight.
- ✓ Item 3c (UIR refactor) disposition recorded (Round-2 Finding 4):
  EITHER (a) re-chartered as a fresh `docs/PLAN_*_UIR.md` proposal
  with concrete trigger, OR (b) closed from carry-forwards. Zombie
  status ends in v2.15.
- ✓ **Phase 0 calibration not expired at tag time** (Round-2
  Finding 5 + Round-3 Finding 3): boolean check —
  `phase_0_expiration_date > today`. Current expiration:
  **2026-06-22** (FP8-14B re-cal 2026-05-23 PM + 30-day window).
  T-72h pre-tag checkpoint (Round-3) auto-schedules re-cal if
  expiring within the buffer; tag-time boolean is the final
  backstop. If tag-time is after expiration OR the GX10 endpoint
  has changed, re-run `scripts/calibrate_local_judge_vs_qwen_max.py`
  before tagging and update the operative
  `docs/CALIBRATION_*_v2.15_p0_*.md` reference. No "calibration
  was fresh when we started the cycle" grandfather clause.
- ✓ **`partial_code` downstream-validation soak** (NEW in v0.4 per
  Round-3 Finding 2): IF the Phase 4 abort path was taken, the
  15-query validation soak must show no judge-axis degradation
  and no retrieval-pipeline error escalation on `partial_code`
  chunks. IF Phase 4 was NOT aborted (either Approach 2 shipped
  successfully or Option E/F was chosen and Phase 4 was out of
  scope), this gate is automatically satisfied.
- ✓ **Phase 2 Abort Teardown verification (NEW in v0.9 per
  Round-8 Finding 6)**: IF any Phase 2 termination path was
  taken (8-day cap / acceptance-bar failure / POC-gate failure /
  user-rescind — per Round-8 Finding 5 broadened trigger), then
  `scripts/verify_phase2_teardown.py` must report PASS. The
  ~20-line script asserts:
  - (a) no `pdfplumber` import in `src/mmrag_v2/engines/`
    non-experimental tree (`grep -rn "import pdfplumber\|from
    pdfplumber" src/mmrag_v2/engines/ | grep -v experimental/`
    returns empty)
  - (b) `src/mmrag_v2/engines/experimental/README.md` exists
    and contains the Mandate text (timestamped record of what
    was preserved)
  - (c) `tests/test_pdfplumber_adapter.py` (if it exists) is
    marked `@pytest.mark.skip(reason="Phase 2 abort; see
    DECISIONS.md")` — programmatic check via AST parse of the
    test file
  - (d) `docs/DECISIONS.md` contains an entry whose header
    matches `^## v2\.15 Phase 2 abort \+ teardown`
  IF Phase 2 was NOT aborted (either Option A succeeded and
  pdfplumber routing shipped, or Option E/F was chosen and
  Phase 2 was out of scope), this gate is automatically
  satisfied. Same enforcement model as the calibration-freshness
  boolean — closes the maintainer-discipline-only loophole on
  the 4-item teardown.

---

## 4. v2.11+ carry-forwards (still open at v2.15 open)

| Item | Source | v2.15 disposition |
|---|---|---|
| 3a NuMarkdown-8B / Qwen3-VL-8B local VLM | v2.11 plan | **PARTIAL in v2.14 Phase 1** (force-table-vlm code path shipped, data rolled back). Future use depends on Option A/E choice — if A, becomes the VLM lane for form-class fallback. |
| 3c UIR refactor (PAUSED) | v2.11 plan | **FORCE-CLOSED in v2.15 Phase N (Round-2 Finding 4).** Has been "PAUSED for user signoff" since v2.11 (≈5 cycles) with zero forward motion. Phase N requires user pick one: (a) re-charter as a fresh `docs/PLAN_<id>_UIR.md` proposal with a concrete trigger condition (e.g., "Docling extraction defects exceed X% of corpus"), or (b) close the item from carry-forwards entirely. Either way, the zombie status ends in v2.15. |
| 3e Magazine rendered-region-crop | v2.11 plan | Deferred again. Image-axis perf is OK without it per current soak data. Re-evaluate in v2.16 if specific magazine queries surface as a problem. |
| 1.1 Same-page prose/VLM dedup | v2.14 deferred | If Option A chosen, this is part of Phase 2's pdfplumber lane integration. If Option E/F, defer indefinitely. |
| 6.1 Docling prose+code disambiguation | v2.14 deferred | If Option A chosen, this is Phase 4 (gated on Approach 2 viability spike). If Option E/F, defer with concrete re-evaluation trigger per Round-4 Finding 6: **"Re-evaluate Phase 4 Approach 2 when Docling minor version increments to ≥2.87 OR every 90 days, whichever first."** Owner: `docs/CYCLE_OPEN_CHECKLIST.md` (shared artifact with Phase 3 telemetry trigger per Round-4 Finding 1). 5-min changelog check per cycle-open. Removes the failure mode where Approach 2 stays infeasible forever because no process watches the Docling changelog. |

---

## 5. Phase ordering rationale

Suggested execution order regardless of strategic decision:

```
Phase 6 (calibration freshness)  ← gate for any local-judge soaks
Phase 1 (HyDE bridging)          ← retrieval win, executes either path
                                   MUST complete in isolation before any Phase 5
[fork on strategic decision]
Phase 2 → Phase 4 [A]    OR    Phase 5 [E]    OR    Phase 3 [F]
                                   Phase 5 sub-phases each in isolation
Phase N (close-out)              ← terminal
```

**Critical-path:** Phase 6 (calibration freshness check) must clear
BEFORE any phase that depends on the Phase 0 verdict. Phase 1's
mini-soak uses cloud `qwen-max` for rel/faith judging (local FP8-14B
acceptable for format), so Phase 6 isn't strictly blocking for it —
but for any Phase 5 sub-phase that uses the local judge, freshness
must clear first.

**Isolation requirement (Gemini round-1 audit):** Phase 1 (HyDE
bridging) and any Phase 5 sub-phase MUST NOT run their mini-soaks in
the same execution. Running them as a single soak makes per-change
attribution impossible (confounding variables). Run Phase 1 first,
read the verdict, THEN run Phase 5a/b/c/d each in its own soak.

**Phase 2 + Phase 4 sequencing (REVISED in v0.3 per Round-2
Finding 3):** v0.2 described Phase 2 (pdfplumber lane, 5-7 days) and
Phase 4 (HybridChunker config tuning, 1-day spike + 1-2d
implementation if viable) as "parallel branches". That framing
assumed multiple engineers. The actual team is one maintainer plus
the AI agent — single-threaded in practice. Honest framing:

- Phase 4 spike comes FIRST (1 day) — it's a viability test that
  either kills Phase 4 (auto-abort if Approach 2 infeasible) or
  scopes its 1-2 day implementation.
- Phase 2 follows (5-7 days serial). UIR-schema mapping sub-task
  is the first 2-3 days; pdfplumber integration is the next 3-4.
- They are technically independent (different sub-systems) but
  **time-additive in practice**.

**Option A hard cap: 8 working days serial.** If Phase 4 + Phase 2
combined work has not produced shippable code at day 8, the cycle
**aborts to Option F** (retain v2.14 behavior + telemetry, defer
the extraction lane to v2.16). No "just one more day" extension.
The cap protects against cycle burnout — the prior week of cycles
(v2.10 → v2.14) averaged ≈1.5 working days each; an 8-day single-
phase commitment is already a significant departure from that
cadence.

---

## 6. Budget

- **Cost cap:** $25/cycle (unchanged from v2.13/v2.14)
- **Estimated spend (Option F path — recommended floor):** **$2-4**
  across Phase 1 mini-soak ($2-3 at n=100 per Round-3 Finding 1;
  was $1-2 at n=50 in v0.3) + Phase 6 re-cal (if needed) + Phase 3
  telemetry validation + Phase N close-out validation. Most of v2.15
  is $0 if F is chosen.
- **Estimated spend (Option A path):** **$4-9** added on top —
  Phase 2 schema-mapping at up to 2 iterations × ~$2 ($4 ceiling
  per Round-3 Finding 5) + Phase 4 mini-soak + Phase 4 abort-path
  `partial_code` validation soak (~$0.30 if abort taken; $0 if
  not) + possible cloud-VLM fallback budget for any form-class
  doc pdfplumber doesn't catch.
  **Wall-clock cap (Round-2 Finding 3): 8 working days serial.**
  If not shippable at day 8, auto-abort to Option F + telemetry.
  Further schema-mapping iterations beyond 2 are time-cost not
  money-cost (Round-3 Finding 5) — gated by the day-8 cap.
- **Estimated spend (Option E path):** **$2-6** added on top —
  Phase 5 sub-phases use cloud qwen-max more aggressively for
  retrieval-tuning soaks (rel + faith axes); format axis can use
  local FP8-14B at $0.
- **Worst case bound:** ~$15.60-16.60 (well under $25 cap
  regardless of path; v0.3 estimated ~$10, v0.4 added ~$3,
  v0.6 added ~$2, v0.7 added ~$0.15-0.30, v0.8 added ~$0.30-0.50
  for the German subgroup bump n=50→100 per Round-7 Finding 1,
  v0.9 adds ~$0.10 per triggered class for diagnostic-injection
  per Round-8 Finding 4 — zero classes trigger in v2.15 because
  both entry classes are above the 1% chronic-defect floor, so
  v0.9 worst-case adds $0 in-cycle and budgets for ~$0.10/cycle
  in future cycles if any chronic-defect class emerges)
- **Local LLM usage:** $0 (LAN)
- **GX10 wall-clock budget (NEW in v0.5 per Round-4 Finding 8):**
  the "$0 (LAN)" line above hides aggregate inference time on the
  shared GX10 endpoint, which can blow the 8-day Option A cap on
  contention rather than merit.

  | Path | GX10 inference time |
  |---|---|
  | Option F (recommended) | ≈2-3h aggregate: Phase 1 mini-soak HyDE generation (~30 min) + Phase 3 telemetry validation (~5 min) + Phase 6 re-cal if needed (~90 min on n=518) + Phase N close-out smoke (~15 min). No spec-decoding throughput recovery available (n-gram rejected; no same-vocab draft model). |
  | Option A | ≈6h aggregate: Option F baseline + Phase 4 viability spike (~1h including iteration) + Phase 2 schema-mapping iterations × up to 2 (~2h each including soak validation runs) + Phase 4 abort-validation soak if abort path (~30 min). |
  | Option E | ≈3-5h aggregate: Option F baseline + Phase 5a per-class top-k sweep (~1h) + Phase 5b query rewriting tests (~1h) + Phase 5c.0 measurement spike (~10 min) + 5c implementation soaks if bracket allows. |

  **Contention factor**: if the GX10 endpoint is shared with
  another workload (separate project, second Codex session, an
  unrelated soak) during the v2.15 cycle window, allocate
  **+50% wall-clock** to the above figures. The 8-day Option A
  hard cap from Round-2 Finding 3 is wall-clock, not pure
  engineering days — it INCLUDES GX10 contention time. If the
  endpoint is uncontended, Option A's 6h GX10 budget is trivially
  absorbed; if it's contended, ≈9h cuts into a day of effective
  engineering before any code is written.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| Option A chosen but pdfplumber POC fails on CarOK | Phase 2 includes explicit POC-first gate before integration; if POC fails, fall back to Option E or defer to v2.16 with cloud VLM |
| Option A chosen and 5-7 day budget overruns | Hard abort at 7 working days; fall back to Option F + telemetry to gather v2.16 user-query evidence; keep partial pdfplumber spike for reference |
| Option E chosen but user later regrets accepting CarOK limitation | DECISIONS.md entry should capture WHY (5:1 retrieval-vs-extraction ROI, no evidence of CarOK-class user queries) so reopening in v2.16 is principled, not impulsive |
| Option F chosen but Phase 3 telemetry not shipped | DoD blocks v2.15.0 tag — Phase 3 is a DoD requirement under Option F |
| Phase 1 HyDE bridging doesn't recover the -12.5pp deficit | Acceptance gate; revisit per-doc embedder routing as Phase 1-bis with concrete "HyDE didn't suffice" evidence — but this is a v2.16 question, not v2.15 |
| Phase 1 false-positive ship from n=50 noise (Round-3 Finding 1) | n bumped to 100 (20/doc); compound gate ≥6pp aggregate AND ≥4/5 directional per-doc consistency floor; aggregate-only wins rejected |
| Phase 4 abort leaves untested `partial_code` in production (Round-3 Finding 2) | Gated 15-query Phase N validation soak fires IF abort taken; failure routes to either Phase 4 reopen or explicit [[contract-violation-mode]] DECISIONS.md entry |
| Calibration window expires mid-cycle, discovered too late (Round-3 Finding 3) | T-72h pre-tag checkpoint auto-schedules re-cal; tag-time boolean is final backstop |
| Maintainer silent on A/E/F at close-out, cycle stalls (Round-3 Finding 4) | Silent-default clause auto-activates Option F at T-24h; DECISIONS.md gets default-trigger entry |
| Phase 2 schema-mapping iterations blow budget (Round-3 Finding 5) | 2-iteration cap at $2 each ($4 ceiling); further iterations are time-cost gated by 8-day wall-clock abort |
| Phase 3 telemetry collects data but nobody runs analysis (Round-4 Finding 1 HIGH) | `scripts/analyze_doc_class_telemetry.py` ships as Phase 3 deliverable; `docs/CYCLE_OPEN_CHECKLIST.md` ships in Phase N with explicit run-this-script line item |
| Telemetry classes stuck in F-purgatory indefinitely (Round-4 Finding 2) | DECISIONS.md telemetry-threshold entry gains <1% / 60-day auto-closure rule complementing the ≥5% promotion rule; F becomes a real fork with both arms defined |
| Phase 1 reconfirms broad-soak null with no termination (Round-4 Finding 3) | Falsification rule: per-doc R@1 null on ≥3/5 closes HyDE bridging as dead lever via DECISIONS.md entry — no carry-forward |
| Phase 1 flat 4/5 gate buries German subgroup failure (Round-4 Finding 4) | Subgroup-aware gate: ATZ_Elektronik (German) MUST be positive AND ≥3/4 code-dense docs positive; aggregate burial of German-null is rejected |
| Phase 5c.0 latency-passes-but-siblings-identical (Round-4 Finding 5) | Spike adds mean-pairwise-Jaccard ≤ 0.7 sub-check; diversity failure routes to cloud or defers same as latency failure |
| Phase 4 abort effectively permanent, no Docling-changelog watcher (Round-4 Finding 6) | Carry-forward 6.1 gets concrete trigger: "Docling minor ≥2.87 OR every 90 days, whichever first"; owner = `docs/CYCLE_OPEN_CHECKLIST.md` |
| Silent-default-to-F triggers on maintainer-offline rather than maintainer-prefers-F (Round-4 Finding 7) | T-48h notification step (terminal banner / pre-push hook / Slack) before T-24h auto-activation |
| GX10 contention silently 2-3× wall-clock, triggers 8-day cap on contention not merit (Round-4 Finding 8) | Section 6 GX10 wall-clock row makes contention explicit; 8-day cap reframed as wall-clock not pure engineering days; +50% contention budget if endpoint is shared |
| Telemetry promotes corpus-frequency over pain-frequency (Round-5 Finding 1) | Promotion rule is compound: `≥5% hit-rate AND (severe-defect-tag OR ≥1 open quality issue)`; popular-but-fine classes don't earn Option A investment for being popular |
| German subgroup gate passes/fails on 1-2 query flips (Round-5 Finding 2) | Bump German fixture 20→30 + raise positivity from "delta > 0" to "delta ≥ +5pp"; defense in depth |
| Middle-band classes age into telemetry-limbo (Round-5 Finding 3) | DECISIONS.md telemetry-threshold entry adds persistence trigger: ≥3 consecutive middle-band cycles escalates to explicit A/E adjudication |
| Phase 4 abort path passes safety but Fluent_Python practically unusable for code-copy (Round-5 Finding 4) | Abort soak adds adequacy gate: ≥10/15 code queries return syntactically complete code block in top-5 (programmatic AST check); below 10/15, abort INVALID |
| Phase 5c ships with null R@1 lift but permanent latency tax (Round-5 Finding 5) | Post-implementation effectiveness gate: ≥3pp R@1 lift required to default-on; +1-3pp ships opt-in only; ≤+1pp reverts |
| Phase 2 POC greenlit on subjective "looks mostly right" (Round-5 Finding 6) | POC exit criteria: ≥95% row alignment on 3 representative pages AND zero downstream parser exceptions on IngestionChunk emission |
| T-72h re-cal slides past T-48h auto-activation while maintainer is in blocking work (Round-6 Finding 6 HIGH) | T-48h decoupled from static tag date; reads from `docs/cycle_slip.log`; effective deadline = `latest cycle_slip new_projected_tag_date - 48h`. Closes the chronological race |
| Defect-tagged classes can't trigger F→A because suppression death spiral keeps hit-rate <5% (Round-6 Finding 1) | Promotion rule gains defect-override arm: `severe_defect_tag AND hit-rate ≥ 1%`; closure rule extended with `severe_defect_tag == False` clause to prevent silent closure of suppressed-defective classes |
| German subgroup gate still in binomial noise at n=30 / +5pp (Round-6 Finding 2) | Bump to n=50 + raise effect-size to +10pp; third iteration on this surface across rounds 4-6 |
| Phase 5c.0 token-set Jaccard inflated by stopwords/glue at short max_tokens (Round-6 Finding 3) | Replaced with omlx cosine similarity (production embedder, LAN, $0); gate ≤ 0.85; fallbacks documented if omlx down |
| Phase 4 AST adequacy gate passes context-dead chunks (Round-6 Finding 4) | Co-gated with ±500-char prose-proximity check; both must pass for ≥10/15 threshold |
| Phase 2 abort leaves zombie pdfplumber code in production tree (Round-6 Finding 5) | Mandatory Abort Teardown commit: revert routing + move to `engines/experimental/` + skip tests + DECISIONS.md entry; abort isn't done until teardown lands |
| German subgroup gate passes null-HyDE ~21% of the time at n=50/+10pp (Round-7 Finding 1) | Bump n=50→100; single-flip noise = ±1pp; false-positive rate drops to ~5%; gate finally on defensible statistical ground after 4 iterations |
| AST adequacy gate passes context-dead chunks where nearby prose is a cross-reference (Round-7 Finding 2) | Augment proximity check with identifier-intersection: AST-extracted identifiers must overlap with ±500-char prose window; ~10 lines Python, $0 |
| New v2.16+ doc classes silently auto-close via F→E before any human review (Round-7 Finding 3) | 2-cycle grace period: `added_cycle` field in config; auto-closure skipped where `current_cycle - added_cycle < 2`; explicit closure by user still permitted |
| T-72h slip-log misses re-cal-fast / review-slow case (Round-7 Finding 4) | Extended trigger: append slip entry if EITHER re-cal wall-clock >24h OR no CALIBRATION verdict file with mtime > re-cal-start exists 24h after re-cal initiation |
| Phase 2 POC page selection cherry-pickable for confirmation bias (Round-7 Finding 5) | Programmatic page-selection rule: page 1 = highest column-count drift (from existing v2.13/v2.14 data) OR median-complexity fallback; pages 2-3 = random seed=42 |
| Phase 5c.0 omlx cosine threshold mis-calibrated for `max_tokens=30` short sequences (Round-8 Finding 1) | Spike gains 5-min calibration baseline pass: 5 hand-curated anchor queries with diverse-vs-near-identical paraphrase quintuplets; threshold = midpoint between empirical clusters, NOT pre-stated 0.85. Cluster non-separation auto-promotes `max_tokens ≥ 60` to primary path |
| AST adequacy gate passes import-severed snippets that parse + identifier-match but are unusable to RAG consumer (Round-8 Finding 2) | Sub-check B extended with import-resolution: any AST `Name` parent-`Call` not locally-bound and not in `builtins.__dict__` must have an `import` inside the snippet OR appear in ±500-char prose window as a backtick code-span. ~15 lines Python, $0 |
| Standard promotion arm's `open_user_issues` signal has no defined collection source — defaults to 0 → gate degrades to defect-tag-only (Round-8 Finding 3) | NEW `docs/USER_ISSUES.md` (append-only markdown table) is the defined source; `analyze_doc_class_telemetry.py` parses via regex; cycle-open checklist gains "Review USER_ISSUES.md" line item |
| Defect-override 1% floor still admits chronic death-spiral: `severe_defect_tag AND hit-rate < 1%` cannot distinguish "truly dead" from "actively avoided due to defect" (Round-8 Finding 4) | NEW chronic-defect adjudication arm: when `severe_defect_tag AND 60-day hit-rate < 1%`, analyzer auto-generates 10 synthetic queries from weakest-query rationales; next cycle's acceptance soak runs them; R@1 < 30% routes to explicit A/E adjudication. Cost: ~$0.10/triggered class/cycle |
| Phase 2 acceptance-fail leaves zombie infrastructure because Mandate trigger was 8-day-cap-only (Round-8 Finding 5) | Mandate trigger broadened to "any Phase 2 termination that does NOT result in pdfplumber routing being promoted to production" — covers 8-day cap, acceptance-bar miss, POC-gate fail, user-rescind |
| No programmatic verification that Mandate's 4 cleanup items landed before tag (Round-8 Finding 6) | Phase N DoD gains `scripts/verify_phase2_teardown.py` gate: asserts (a) no pdfplumber import outside experimental/ tree, (b) experimental/README.md exists, (c) test file is `@pytest.mark.skip`, (d) DECISIONS.md entry exists. Same enforcement model as calibration-freshness boolean |
| Phase 4 Approach 2 (Docling config tuning) proves infeasible | Hard abort gate per Gemini audit — NO regex fallback; defer to v2.16 potentially gated on Docling version bump |
| Phase 5c paraphrase fusion latency physics (Round-2 Finding 2) | v0.2's <1500ms p50 budget was below the physical floor (≈2.0s for n=5 × 30 tokens at 15 tok/s). Phase 5c.0 measurement spike runs FIRST; budget set post-measurement; >4000ms p50 auto-routes to cloud `qwen-max` or defers to v2.16 |
| Option A scope creep past 8 working days (Round-2 Finding 3) | Hard wall-clock cap; auto-abort to Option F + telemetry at day 8. Phase 4 spike runs FIRST so its abort path is taken early (day 1) rather than late |
| Phase 3 telemetry collects data with no decision rule (Round-2 Finding 1) | Threshold (`≥5% per-class hit-rate in 30-day rolling window with rerank top-5 non-empty`) defined in v0.3 Phase 3 + DECISIONS.md pre-cycle proposal. v2.16 inherits a ready rule, not a "we'll figure it out" |
| Tag ships on expired calibration (Round-2 Finding 5) | DoD boolean: `phase_0_expiration_date > today` at tag time; no grandfather clause |
| Phase 1 + Phase 5 results conflated by overlapping soaks | Isolation rule: each runs in its own mini-soak; never combined |
| Calibration freshness drifts mid-cycle | Phase 6 enforces 30-day-OR-model-change rule; the [[no-gx10-model-swap-reflex]] rule keeps model-change incidence low |
| Engineering effort balloons on Option A's specialized-lane integration | Strict Phase 2+4 acceptance bars; if either phase blows >7 working days (revised), abort to Option F (retain v2.14 behavior + telemetry, defer the lane to v2.16) |
| Scope creep — new "let's try one more X" experiments | This plan's Non-goals section explicitly forbids: no GX10 swaps, no further spec-decoding experiments, no retrieval-stack changes, no third CarOK Docling-knob experiment |

---

## 8. Open questions for the user (decision inputs)

1. **The strategic decision: Option A, Option E, or Option F?** This
   gates the Phase 2/4 vs Phase 3 vs Phase 5 fork. Recommended
   default: **F** (ship Phase 1 + 3 + 6 + N this cycle; defer the
   A/E commitment to v2.16 once Phase 3 telemetry has collected
   user-query evidence of which problem documents are load-bearing).
   Gemini round 1 (2026-05-24) and Round-3 audit (2026-05-24)
   independently arrived at the same recommendation.
   **Silent-default fallback (Round-3 Finding 4):** if no explicit
   selection by T-24h before planned tag, Option F auto-activates.
   This is a feature, not a bug — Option F is the path you'd pick
   if you had to pick blind, so defaulting to it removes the
   indefinite-stall failure mode.
2. **Phase 1 priority weighting:** is the German-content regression
   (ATZ_Elektronik -12.5pp R@1) worth investing in this cycle, or
   defer to v2.16 if no user queries against it have surfaced?
3. **Phase 5 sub-phase selection** (if Option E chosen): which of
   5a / 5b / 5c / 5d are in scope? Note 5c is hard-gated on latency
   budget; 5d is a $0 documentation phase.
4. **Phase 3 telemetry retention** (if Option F chosen): v0.3 sets
   default to "rotate weekly + retain indefinitely" (small file;
   query text + doc IDs only; no PII risk surfaced by this kind of
   query corpus). Override if your retention policy differs.
5. **Item 3c (UIR refactor) disposition** (Round-2 Finding 4): pick
   one in Phase N — (a) re-charter as fresh proposal with concrete
   trigger condition, or (b) close from carry-forwards. No third
   "let it sit another cycle" option in v0.3.
6. **Optional:** SSH key onboarding for the GX10 (still pending —
   would unblock live-endpoint diagnostics if local LLM crashes
   mid-soak; cheap one-time cost).

---

## 9. Process notes

- v2.14 plan archaeology (Draft v0.1 preserved-history block) is
  STILL pending archive from v2.14 close-out. v2.15 Phase N picks
  up both v2.14 and v2.15 archaeology in one sweep.
- The [[no-gx10-model-swap-reflex]] rule applies to v2.15 with the
  same force as v2.14. The 14B-FP8 endpoint is the operative
  reference; do not propose a swap as the reflex response to any
  v2.15 perf disappointment. Re-evaluation gated on the offline-
  eval-first protocol via `scripts/calibrate_local_judge_vs_qwen_max.py`
  with OpenRouter bearer-key (already wired in v2.14).
- The "evaluate candidates offline before any live-endpoint swap"
  protocol from the same memory is now battle-tested (v2.14 used
  it to disprove Qwen3-32B and Llama-3.1-70B against the 14B-FP8
  baseline before any production swap).
- Gemini audit round 1 (2026-05-24) was the source of five Draft v0.2
  amendments: Option A budget bump (Phase 2), Phase 4 Approach 1
  rejection + hard abort gate, Phase 5c latency budget, Phase 1/5
  isolation rule, and the entirely-new Phase 3 [F] telemetry phase.
- **Round-2 audit (2026-05-24, same day; prompt at
  `docs/PLAN_V2.15_AUDIT_PROMPT.md`)** flagged five additional
  findings, all incorporated in v0.3 — see Round-2 changes table
  at the head of this document. The audit prompt design
  deliberately blinds the round-2 auditor to round-1 findings; the
  fact that round 2 returned 5 substantive items (2 HIGH) validates
  that single-pass audits leave structural gaps.
- **Round-3 audit (2026-05-24, same day; same prompt)** flagged
  five further findings against v0.3 — all MED/LOW, all
  incorporated in v0.4. Round 3 returning 0 HIGH initially suggested
  diminishing marginal returns.
- **Round-4 audit (2026-05-24, same day; same prompt)** returned
  **8 findings** — 1 HIGH (Phase 3 telemetry has a defined trigger
  but no defined reader/process), 5 MED, 2 LOW — all incorporated
  in v0.5. The HIGH was a structural Phase-3 invalidator: a
  trigger nobody fires is theatrical.
- **Round-5 audit (2026-05-24, same day; same prompt)** returned
  **6 findings** — **0 HIGH**, 5 MED, 1 LOW — all incorporated
  in v0.6.
- **Round-6 audit (2026-05-24, same day; same prompt)** returned
  **6 findings** — **1 HIGH** (chronological race between T-72h
  checkpoint and T-48h notification), 5 MED — all incorporated
  in v0.7.
- **Round-7 audit (2026-05-24, same day; same prompt)** returned
  **5 findings** against v0.7 — **0 HIGH**, 4 MED (German
  subgroup false-positive rate, prose-proximity relevance gap,
  new-class auto-closure gap, T-72h slip-log end-to-end coverage),
  1 LOW (POC page-selection cherry-picking) — all incorporated
  in v0.8.
- **Round-8 audit (2026-05-24, same day; same prompt)** returned
  **6 findings** against v0.8 — **0 HIGH**, 4 MED (omlx-cosine
  threshold mis-calibration at `max_tokens=30`, AST import-
  severance blind spot, `open_user_issues` collection-source
  undefined, defect-override 1% floor chronic-spiral residual),
  2 LOW (Teardown Mandate trigger too narrow, no programmatic
  teardown-verification gate) — all incorporated in v0.9.
- **HIGH-severity count progression (final)**: 2 → 2 → 0 → 1 → 0
  → 1 → 0 → **0**. Round 8's 0 HIGH brings consecutive-clean to
  **2**.
- **Stopping rule FIRES at v0.9**: "two consecutive rounds with
  0 HIGH" condition is **MET** (rounds 7 and 8 both 0 HIGH).
  The plan is now executable pending §8 Q1 strategic decision
  (Option A/E/F). No further audit rounds are required by the
  protocol, though the silent-default clause keeps the cycle
  unblocked even if Q1 is not answered before T-24h.
- **Meta-observation (final updated)**: HIGH-count progression
  (2 → 2 → 0 → 1 → 0 → 1 → 0 → 0) and MED+LOW count
  (5 → 5 → 5 → 8 → 6 → 6 → 5 → 6) over eight rounds. The
  cumulative HIGH count is 6 across rounds 1-8 (none in rounds
  3, 5, 7, 8). Three interpretations remain active, with v0.9
  evidence updates:
  - (a) The plan covers genuinely complex multi-component scope;
    structural gaps surface each round because each round's
    fixes create new surface area. **Most strongly supported by
    v0.9**: round 8's findings are all next-layer concerns on
    surfaces the prior rounds had already addressed (omlx-cosine
    swap was Round 6, AST identifier-intersection was Round 7,
    compound promotion was Round 5, Teardown Mandate was Round 6
    — and v0.9 closes second-order gaps in each). The fixes
    genuinely propagate; the propagation is now ≤1 round behind
    each fix.
  - (b) Auditor prompt may unintentionally encourage finding
    "exactly 5-8" items. **Still weakly supported**: round 8
    returned 6, comfortably within the historical range. Items
    remain substantive and pass the "concrete failure mode"
    smell test.
  - (c) Multiple parallel structural concerns the plan never
    fully resolves. **Moderately supported, now stable**: the
    German subgroup gate was iterated 4 times (rounds 4-7) and
    stabilized at n=100 in v0.8; the AST adequacy gate was
    iterated 3 times (rounds 5, 6, 7, with v0.9 import-
    resolution being the 4th); the compound promotion gate was
    iterated 4 times (rounds 5, 6, 7, 9). Some surfaces require
    multiple iterations — the audit cadence is what catches
    them.

  The stopping rule was the right floor. Eight rounds across
  one day produced ~750 added lines (plan + DECISIONS.md + new
  artifacts) closing ~40 concrete in-cycle failure modes. The
  audit cadence paid; the rule fires; execution is unblocked.

---

## Appendix A — Draft archaeology (preserved)

### Draft v0.8 → v0.9 (Round-8 audit, 2026-05-24)

Draft v0.8 contained:
- Phase 5c.0 omlx pairwise cosine threshold of 0.85 was
  calibrated against the embedder's distribution on retrieval-
  chunk-length text (256-512 tokens), not the spike's
  `max_tokens=30` snippets. Short-sequence inputs collapse
  toward a length-conditioned manifold attractor that inflates
  pairwise cosine independent of paraphrase quality. The gate
  could over-trigger (cloud-routing diverse paraphrases) or
  under-trigger (passing degenerate samples). **Round-8 Finding
  1: MED.** v0.9 adds a 5-min calibration baseline pass before
  the main spike trials: 5 anchor queries with diverse-vs-near-
  identical paraphrase quintuplets; operative threshold =
  midpoint between empirical clusters. Cluster non-separation
  promotes `max_tokens ≥ 60` to primary path, not fallback.
- Phase 4 AST + identifier-intersection check validated that
  nearby prose mentions snippet-internal identifiers, but a
  snippet like `cache = LRUCache(maxsize=128)` parses, has
  matching identifiers, AND can still be unusable if the
  `from x import LRUCache` upstream is severed outside the
  ±500-char window. Identifier-intersection on snippet-internal
  identifiers cannot catch import-severance. **Round-8 Finding
  2: MED.** v0.9 extends Sub-check B with import resolution:
  any AST `Name` parent-`Call` not locally-bound and not in
  `builtins.__dict__` must have an `import` inside the snippet
  OR appear in the ±500-char prose window as a backtick code-
  span. ~15 lines Python, $0.
- Compound promotion arm's `open_user_issues` had no defined
  collection source — the analyzer would default to 0
  indefinitely and the standard arm would degrade to
  `severe_defect_tag AND hit-rate ≥ 5%`, unenforcing Round 5
  Finding 1's popular-but-fine guard. **Round-8 Finding 3:
  MED.** v0.9 defines `docs/USER_ISSUES.md` as an append-only
  markdown table (schema: `date | doc_class | query |
  observed_behavior | expected_behavior`); analyzer parses it
  via regex; cycle-open checklist gains a "review USER_ISSUES.md"
  line item.
- Defect-override 1% floor admitted a sub-1% chronic spiral: a
  class with `severe_defect_tag` AND hit-rate <1% could be
  "truly dead" (no latent demand) OR "actively avoided due to
  defect" (latent demand suppressed by defect); telemetry alone
  cannot distinguish. **Round-8 Finding 4: MED.** v0.9 adds a
  chronic-defect adjudication arm: when `severe_defect_tag AND
  60-day hit-rate < 1%`, analyzer auto-generates 10 synthetic
  queries from the doc's weakest-query rationales; the next
  cycle's acceptance soak runs them; R@1 <30% routes to
  explicit A/E adjudication. Cost: ~$0.10/triggered class/cycle.
- Abort Teardown Mandate trigger was "on 8-day cap trigger" —
  Phase 2 could produce identical failed-cycle outcomes via
  acceptance-bar miss / POC-gate failure / user-rescind without
  mandating teardown. **Round-8 Finding 5: LOW.** v0.9
  reworded the trigger to "any Phase 2 termination that does
  NOT result in pdfplumber routing being promoted to production."
- Teardown Mandate enumerated 4 cleanup items but no
  programmatic gate verified they all landed before tag.
  Maintainer-discipline-only enforcement on a process step that
  runs when the maintainer is most exhausted. **Round-8 Finding
  6: LOW.** v0.9 added DoD line item: if Phase 2 aborted,
  `scripts/verify_phase2_teardown.py` must report PASS — 4
  programmatic assertions matching the Mandate's 4 cleanup
  items. Same enforcement model as the calibration-freshness
  boolean.

### Draft v0.7 → v0.8 (Round-7 audit, 2026-05-24)

Draft v0.7 contained:
- Phase 1 German subgroup gate at `n=50, delta ≥ +10pp` had
  ~21% false-positive rate under null (≈25 effective Bernoulli
  trials at baseline R@1 ≈ 57.5%; P(net ≥ +5 flips | null) ≈ 0.21
  via normal approximation). Gate conflated significance testing
  with effect-size flooring. **Round-7 Finding 1: MED.** v0.8
  bumps n=50→100; false-positive rate drops to ~5%; gate finally
  on defensible statistical ground after 4 iterations across
  rounds 4-7.
- Phase 4 AST adequacy gate's ±500-char prose-proximity check
  confirmed prose EXISTS but not RELEVANCE. Cross-references
  ("See Chapter 14") and transition sentences pass without
  providing usable context. **Round-7 Finding 2: MED.** v0.8
  augments with identifier-intersection check: AST-extracted
  identifiers (function names, params, top-3-line variables)
  must overlap with the ±500-char prose window. Deterministic;
  ~10 lines Python; $0.
- `severe_defect_tag` is a manual flag — newly-added v2.16+
  classes with severe defects but no manual tag could enter
  telemetry → users abandon → hit-rate <1% → 60-day auto-close
  fires → permanent Option E with no human review. Diagnosis
  gap is largest when class is newest. **Round-7 Finding 3: MED.**
  v0.8 adds 2-cycle grace period (`added_cycle` field; auto-
  closure skipped where `current_cycle - added_cycle < 2`).
- T-72h `cycle_slip.log` trigger fired on re-cal wall-clock >24h
  but missed the re-cal-fast / review-slow case (90-min re-cal +
  36-hour review delay would NOT trigger the slip log despite
  effective tag date slipping >24h). **Round-7 Finding 4: MED.**
  v0.8 extends trigger: append slip entry if EITHER re-cal >24h
  OR no CALIBRATION verdict file with mtime > re-cal-start exists
  24h after re-cal initiation. Single file-existence check.
- Phase 2 POC page selection ("3 representative pages, suggested
  1, 6, 11") was manual — confirmation-biased user could pick
  structurally easy pages where pdfplumber trivially passes.
  **Round-7 Finding 5: LOW.** v0.8 makes page selection
  programmatic: page 1 = highest column-count drift from
  v2.13/v2.14 data (or median-complexity fallback); pages 2-3 =
  random sample seed=42.

### Draft v0.6 → v0.7 (Round-6 audit, 2026-05-24)

Draft v0.6 contained:
- Compound promotion gate `hit-rate ≥ 5% AND pain-signal` had a
  suppression death spiral: as a class's extraction degrades,
  users abandon queries → hit-rate falls below 5% → the AND gate
  blocks promotion. Defect-tagged classes could never trigger F→A
  if their volume was suppressed by their own defects. **Round-6
  Finding 1: MED.** v0.7 adds defect-override arm:
  `severe_defect_tag AND hit-rate ≥ 1%`. Also extends closure
  rule with `severe_defect_tag == False` to prevent silent
  closure of suppressed-defective classes.
- German subgroup at `n=30 / +5pp` was still in binomial noise
  (single flip = 3.33pp; +5pp ≈ 1.5 flips). **Round-6 Finding 2:
  MED.** v0.7 bumps to n=50 + raises effect-size floor to +10pp
  — third iteration on this surface across rounds 4-6.
- Phase 5c.0 token-set Jaccard on `max_tokens=30` was dominated
  by trivial syntactic tokens (stopwords, prompt echoes,
  boilerplate); small denominator inflated similarity scores
  independent of paraphrase quality. **Round-6 Finding 3: MED.**
  v0.7 replaces with omlx cosine similarity (production
  embedder, LAN, $0); gate ≤ 0.85. Fallbacks documented if omlx
  unavailable.
- Phase 4 AST adequacy gate validated syntactic completeness of
  the code snippet only — blind to severed imports, usage
  examples, parameter prose. Context-dead chunks would pass.
  **Round-6 Finding 4: MED.** v0.7 co-gates AST validity with
  ±500-char prose-proximity check; both must pass for ≥10/15
  threshold.
- Phase 2 8-day abort halted dev but left partial pdfplumber
  adapters / routing flags / schema-mapping stubs in `engines/`
  + `tests/` — zombie infrastructure. **Round-6 Finding 5: MED.**
  v0.7 mandates Abort Teardown: revert routing, move partial
  code to `engines/experimental/`, skip test files, log state in
  DECISIONS.md. Abort isn't done until teardown lands.
- T-72h pre-tag checkpoint and T-48h silent-default notification
  were both anchored to the **static** planned tag date. If T-72h
  re-cal blocks for >24h, T-48h could fire Option F
  auto-activation while the maintainer is in plan-authorized
  blocking work — a directly contradictory dual-trigger state.
  **Round-6 Finding 6: HIGH.** v0.7 decouples T-48h from the
  static date: dynamic deadline computed from
  `docs/cycle_slip.log`; T-72h appends a slip entry if it blocks
  for >24h; notification dispatch reads the log. Closes the
  chronological race.

### Draft v0.5 → v0.6 (Round-5 audit, 2026-05-24)

Draft v0.5 contained:
- Phase 3 promotion trigger measured only corpus-frequency
  ("appears in top-5") — high-volume-but-acceptable classes could
  trigger Option A investment for being popular while lower-volume
  catastrophic-defect classes never crossed the raw-volume
  threshold. **Round-5 Finding 1: MED.** v0.6 promotion rule is
  compound: `hit-rate ≥ 5% AND (severe-defect-tag OR ≥1 open
  quality issue)`. Pain-signal coupled to corpus-frequency
  signal.
- Phase 1 German subgroup gate at n=20 with "delta > 0" was
  fragile to 1-2 query flips of sampling noise. **Round-5
  Finding 2: MED.** v0.6 bumps German fixture 20→30 AND raises
  positivity threshold to "delta ≥ +5pp" — defense in depth.
  Code-dense gate keeps "delta > 0" because its 4-of-4 quorum
  provides directional-consistency robustness within that
  subgroup.
- Option F middle-band (1% ≤ rate < 5%) rolled forward "through
  next cycle" indefinitely — classes could live for 6 months in
  telemetry-limbo. **Round-5 Finding 3: MED.** v0.6 DECISIONS.md
  telemetry entry adds persistence trigger: ≥3 consecutive
  middle-band cycles escalates to **explicit A/E adjudication**
  at next cycle open. Forces a user decision rather than another
  defer.
- Phase 4 abort-path soak validated downstream safety ("doesn't
  crash") and judge-axis non-degradation, but not user-visible
  adequacy ("can users actually answer code queries from
  truncated chunks?"). **Round-5 Finding 4: MED.** v0.6 adds
  adequacy gate: ≥10/15 code queries must return syntactically
  complete code block in top-5 (programmatic AST parse). Below
  threshold, abort is INVALID and the defect cannot be classified
  as safe to defer.
- Phase 5c gated latency AND sibling diversity but never defined
  the post-implementation retrieval-lift threshold — could ship
  paraphrase fusion that passes 5c.0 but produces ≈null R@1 lift
  with permanent latency tax. **Round-5 Finding 5: MED.** v0.6
  adds post-implementation effectiveness gate: ≥3pp R@1 lift
  required to default-on; +1-3pp ships opt-in only; ≤+1pp reverts.
- Phase 2 [A] POC ("visually confirm columns separate cleanly")
  was subjective and bias-prone before the 5-7 day commitment.
  **Round-5 Finding 6: LOW.** v0.6 concretizes exit criteria:
  ≥95% row alignment on 3 representative CarOK pages AND zero
  downstream parser exceptions on IngestionChunk emission.

### Draft v0.4 → v0.5 (Round-4 audit, 2026-05-24)

Draft v0.4 contained:
- Phase 3 [F] telemetry trigger defined precisely but no reader
  or process specified — the trigger could never fire because
  nothing was scheduled to compute hit-rates against the log.
  **Round-4 Finding 1: HIGH.** v0.5 ships
  `scripts/analyze_doc_class_telemetry.py` as a hard Phase 3
  deliverable + `docs/CYCLE_OPEN_CHECKLIST.md` (new doc) in Phase
  N with the "run this script" line item. The checklist also
  becomes the owner for Phase 4 abort re-evaluation (Finding 6)
  and the T-72h calibration freshness check.
- Trichotomy was asymmetric — F→A trigger defined (≥5%), no F→E
  trigger. Classes below 5% lived in telemetry-purgatory forever;
  F was operationally biased toward A. **Round-4 Finding 2: MED.**
  v0.5 DECISIONS.md telemetry-threshold entry gains the
  complementary `<1% over 60-day window AND 0 open user issues`
  closure rule. F is now a real fork.
- Phase 1 retried v2.14's falsified broad-soak hypothesis on a
  narrower fixture without articulating WHY narrowing should
  produce a different answer. No termination condition if the
  null reconfirmed. **Round-4 Finding 3: MED.** v0.5 Phase 1 Goal
  gains explicit dilution-vs-no-lift hypothesis + falsification
  rule (per-doc null on ≥3/5 closes HyDE bridging as dead lever
  via DECISIONS.md entry, not carry-forward).
- Phase 1 ≥4/5 directional gate was structurally blind to the
  German subgroup (1 German + 4 code-dense; all-code-dense-lift
  + German-null passes ≥4/5). **Round-4 Finding 4: MED.** v0.5
  replaces with subgroup-aware gate: German MUST be positive AND
  ≥3/4 code-dense positive. German-null routes to explicit
  closure entry, not aggregate burial.
- Phase 5c.0 spike measured latency but not sibling diversity —
  vLLM `n=5` at short max_tokens may produce ≈identical samples,
  passing latency gate while delivering ≈1 effective paraphrase
  for RRF fusion. **Round-4 Finding 5: MED.** v0.5 adds
  mean-pairwise-Jaccard ≤ 0.7 sub-check; diversity failure
  routes to cloud or defers same as latency failure.
- Phase 4 abort path "potentially gated on Docling version bump"
  but nothing watched the Docling changelog — abort effectively
  permanent. **Round-4 Finding 6: MED.** v0.5 carry-forward 6.1
  gains concrete trigger "Docling minor ≥2.87 OR every 90 days,
  whichever first" with owner = `docs/CYCLE_OPEN_CHECKLIST.md`
  (shared artifact with Finding 1's checklist).
- Silent-default-to-F at T-24h assumed silence = "prefer F" but
  silence could equally mean offline/sick/traveling. **Round-4
  Finding 7: LOW.** v0.5 adds T-48h notification step (terminal
  banner / pre-push hook / Slack) before T-24h auto-activation.
- Section 6 budget treated local LLM as "$0 (LAN)" without
  surfacing aggregate GX10 wall-clock — contention could blow
  the 8-day cap on contention rather than merit. **Round-4
  Finding 8: LOW.** v0.5 adds GX10 wall-clock row with per-Option
  estimates + +50% contention budget; reframes 8-day cap as
  wall-clock not pure engineering days.

### Draft v0.3 → v0.4 (Round-3 audit, 2026-05-24)

Draft v0.3 contained:
- Phase 1 acceptance: single ≥8pp R@1 lift gate on n=50 mini-soak.
  **Round-3 Finding 1: MED.** At n=50 a single query flip is
  ±2pp; 8pp is ≈4 flips — comfortably within binomial-variance
  noise floor. v0.4 replaces with compound gate: n=100 +
  aggregate ≥6pp + per-doc directional ≥4/5 + format no-regression
  + faith ≤-1pp.
- Phase 4 abort path "trusted" the v2.14 `partial_code` schema flag
  to provide downstream consumers with truncation handling info,
  but never end-to-end-validated the consumer behavior.
  **Round-3 Finding 2: MED.** v0.4 adds a gated 15-query
  `partial_code` Phase N validation soak that fires only if abort
  path taken; failure routes to Phase 4 reopen or a
  [[contract-violation-mode]] DECISIONS.md entry.
- Phase 6 freshness check ran at cycle-start; DoD boolean at tag
  time. Reactive gap for mid-cycle expiration. **Round-3 Finding 3:
  LOW.** v0.4 adds T-72h pre-tag checkpoint that auto-schedules
  re-cal as a blocking pre-tag step.
- DoD required explicit A/E/F decision but no fallback for silent
  maintainer. Cycle could stall indefinitely. **Round-3 Finding 4:
  LOW.** v0.4 adds silent-default clause — Option F auto-activates
  at T-24h if no explicit selection.
- Option A worst-case $7 ignored iterative UIR-schema-mapping
  cycles. **Round-3 Finding 5: LOW.** v0.4 caps schema-mapping at
  2 iterations × $2 ($4 ceiling); further iterations are time-cost
  not money-cost, gated by 8-day wall-clock abort.

### Draft v0.2 → v0.3 (Round-2 audit, 2026-05-24)

Draft v0.2 contained:
- Phase 3 [F] telemetry threshold `X%` deferred to v2.16
  with no decision rule attached. **Round-2 Finding 1: HIGH.
  Defined in v0.3 as ≥5% per-class hit-rate in 30-day rolling
  window with rerank top-5 non-empty; recorded as pre-cycle
  proposal in DECISIONS.md.**
- Phase 5c hard latency budget of `<1500ms p50 / <3000ms p99`
  on the bare-config FP8-14B endpoint. **Round-2 Finding 2: HIGH.
  Below the physical floor — n=5 × 30 tokens at measured 15 tok/s
  ≈ 2.0s wall-clock minimum. v0.3 repeals the aspirational budget
  and inserts a mandatory Phase 5c.0 measurement spike before any
  implementation; budget set post-measurement; >4000ms p50
  routes to cloud or defers to v2.16.**
- Section 5 "parallel-safe" framing for Phase 2 + Phase 4 (Option A).
  **Round-2 Finding 3: MED. Single-engineer team can't actually
  parallelize; v0.3 reframes as time-additive and adds an 8-working-
  day hard cap on Option A with auto-abort to Option F.**
- Item 3c (UIR refactor) listed as "PAUSED for user signoff" with
  no end condition. **Round-2 Finding 4: LOW. Has been PAUSED
  ≈5 cycles; v0.3 force-closes in Phase N with user picking
  re-charter-as-fresh-proposal vs. close-from-carry-forwards.**
- Phase 6 calibration-freshness gate said "before any local-judge
  soak" but didn't bind tag time. **Round-2 Finding 5: LOW.
  v0.3 DoD adds `phase_0_expiration_date > today` boolean at tag
  time; no grandfather clause.**

### Draft v0.1 → v0.2 (Gemini round 1 + stale-state delta, 2026-05-24)

Draft v0.1 (2026-05-23) contained:
- Phase 1 (rollback drop) at the unconditional top of the phase
  list with a "2026-06-19 time gate" acceptance bar. **Executed
  pre-cycle by user override; demoted in v0.2.**
- Section 2 Non-goals reference to "27B-MTP is the GX10 ceiling."
  **Stale as of 2026-05-23 PM endpoint swap; rewritten in v0.2.**
- Phase 5b reference to "local 27B-MTP" for query rewriting.
  **Stale; updated to FP8-14B with current latency profile.**
- Phase 6 calibration-status reference to "Phase 0 27B-MTP cal
  SHIPPED 2026-05-23." **Stale; updated to FP8-14B verdict
  2026-05-23 PM with 2026-06-22 freshness window.**
- No telemetry mechanism for Option F (silent gap; flagged by
  Gemini round-1 audit and addressed by new Phase 3 [F]).
- No isolation rule for Phase 1/5 mini-soak overlap (silent
  confounding-variable risk; flagged by Gemini round-1 audit and
  addressed in Sections 3 + 5).
- Phase 2 (pdfplumber lane) at 2-3 days. **Under-budgeted per
  Gemini round-1 audit; revised to 5-7 days with explicit
  UIR-schema mapping sub-task.**
- Phase 4 (HybridChunker post-process) with Approach 1
  (regex/heuristic) as a fallback. **Rejected as a chunking-
  layer analog of the lenient-judge trap per Gemini round-1
  audit; v0.2 removes the fallback and adds a hard abort gate.**
