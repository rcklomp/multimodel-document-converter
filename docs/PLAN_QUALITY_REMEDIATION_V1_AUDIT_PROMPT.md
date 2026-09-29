# Audit prompt: PLAN_QUALITY_REMEDIATION_V1 (external, adversarial, Round 1)

> Hand this whole file to a fresh frontier-model session that has READ access to the repository
> `MM-Converter-V2.4.1` (branch `fix/quality-remediation-v1`, base `499a5fa`) and to the attachment folder
> `output/audit_2026-09-29/` (gitignored; copy it into the auditor's session). The prompt is standalone.
> It produces an audit report; it edits nothing. Run it on at least TWO auditors from different model
> families; Round 1 is not complete until both have reported (Section 2).

---

## 0. Role and quality bar

You are an external, unbiased principal engineer auditing a **remediation-and-governance-convergence plan**
for a PDF-to-JSONL multimodal RAG ingestion pipeline: `docs/PLAN_QUALITY_REMEDIATION_V1.md`. You have no
stake in any decision. Your only loyalty is to whether the plan will actually leave the project in a better,
verifiably closed state, or whether it will produce the appearance of closure.

The plan's distinguishing feature: it is triggered by an external quality verdict on ONE document (an IRJET
solar-PV paper) whose conversion passed the project's own strict gate with `QA_PASS: failures=0 warnings=0`
while carrying figure crops of 67x68 px, missing figures, running headers glued into 12 of 34 chunks and wrong
headings on 12 of 30 chunks. The same proxy-versus-outcome failure can recur inside the plan's own acceptance.

The **load-bearing artifacts** are:
1. the **disposition register** (plan Sections 7 and 9-10: every open item ends as exactly one of FIX, DECIDE,
   CLOSE or BLOCKED with an owner and a trigger), and
2. the **outcome-metric harness** (WP-0.2 and WP-0.3, baseline in Section 5, acceptance in Section 8) that makes a
   "fixed" claim falsifiable.
Weight your scrutiny toward these two, not toward the most visible artifact (the work-package list).

Scrutiny bar: *"the maintainer who opens this repo in six months, without any of today's context, must be able to
verify every closure and must be unable to mistake a soft deferral for a fix; a plan that cannot fail its own
acceptance is worthless."*

You must NOT read, cite, or recommend reading anything under `docs/.archive/` or `tests/.archive_*` (blocked by
`.aiignore`). A finding that requires archive content is out of scope.

---

## 1. What to read

Plan under audit: `docs/PLAN_QUALITY_REMEDIATION_V1.md` (read fully, twice: once for structure, once adversarially).
Governance you must apply (read before judging any WP): `CLAUDE.md` (incl. "Reliability Protocol"), `AGENTS.md`
(esp. AGENT-STATUS-01, AGENT-TEST-01, AGENT-GATE-PROGRESSION, AGENT-INTEGRITY-01, AGENT-EVIDENCE-01,
AGENT-SPATIAL-20, the multimodal image contract, "Settled work + dead ends"), `docs/V3_EXECUTION_MANDATE.md`
(the conflict-resolution authority), `docs/paper/FINDINGS_DIGEST.md` (SETTLED / DEAD ENDS / OPEN), the "Settled
Precedents" index at the top of `docs/DECISIONS.md`, `docs/QUALITY_GATES.md`, `docs/PROJECT_STATUS.md`,
`docs/ARCHITECTURE_V3.1_CHARTER.md`.
Attachments (`output/audit_2026-09-29/`): the four read-only audit registers (`r_issue_inventory.md`,
`r_gov_layer0.md`, `r_gov_decisions.md`, `r_gov_arch_plans_status.md`), the three root-cause reports
(`r_rc_figures.md`, `r_rc_order_headings.md`, `r_rc_equations_refs.md`), the lineage report
(`r_lineage_laptop_travel.md`), the verified-facts note (`irjet_verified_facts.md`) and two IRJET outputs
(`irjet_cloud_probe_ingestion.jsonl`, `irjet_baseline_499a5fa.jsonl`). The source PDF `data/academic_journal/IRJET_Modeling_of_Solar_PV_system_under.pdf`
is gitignored: it may be absent on your machine; say so if you cannot run a PDF-dependent check rather than assuming.

---

## 2. Round structure and stopping rule

- This is Round 1 of the plan audit. A prior INTERNAL adversarial review ("Round 0") was run by a separate subagent
  in the authoring session; its findings and how each was fixed are in Section 3. Round 0 does not count toward the stopping rule.
- Stopping rule: audit until TWO CONSECUTIVE rounds return **0 HIGH-severity findings counted across all auditors of that round**.
  If auditor A reports 0 HIGH and auditor B reports 3 HIGH, the round counts as 3 HIGH. A single clean audit is not proof: it may be your blind spot.
- Where two auditors disagree on the same lens, the disagreement is signal; report your position with the specific evidence
  (file:line, command output) that anchors it so the owner can compare side by side.

---

## 3. Prior-finding exclusion list (do NOT re-flag; how each was fixed)

Established facts (verified with receipts in the plan, Appendix A and the attachments; do not re-derive them, audit the PLAN built on them):
- E1 IRJET asset/bbox area ratios 0.08 and 0.07 (B1 picks a text-strip raster and the header logo); Figs 4,5,6,8,9,10 dropped by `_filter_tiny_icon_images`; stale `next_text_snippet`.
- E2 `search_priority: low` comes only from the boilerplate regex at `batch_processor.py:3056-3065`; F1 furniture filter is chunk-level, capped at 70 chars.
- E3 `_partition_text_elements` stamps the group-last heading on every part; headings never lead a chunk.
- E4 References split by `_split_at_sentence_boundaries`; header glued by `_merge_mid_sentence_chunks`; order defect is VLM emission order (engine-specific).
- E5 Equation defect: no equation contract in the page prompt or enum; source equations are a garbled text layer plus sprites.
- E6 `pipeline_version` stamps the schema version; `config_hash` is never assigned.
- E7 Gate blindness: `QA_PASS 0/0` on the defective output; test baseline 1767 passed / 99 skipped / 0 failed.
- E8 The checkout was 45 commits stale and was fast-forwarded (HEAD 7b515e3 -> 499a5fa); `github/laptop-travel` is a diverged lineage.
Round 0 findings (internal review by two subagents on plan rev. 1; 6 HIGH / 19 MED / 9 LOW; their reports are `r0_audit_A.md` and
`r0_audit_B.md` in the attachment folder). Treat each as FIXED in plan rev. 2 unless you can show the stated fix does not hold; plan Section 13 lists the changes.
- R0-H1 register incomplete / mis-mapped / not resolvable in a clean clone -> now the generated, committed register file
  `docs/PLAN_QUALITY_REMEDIATION_V1_REGISTER.md` (223 ids, verified id-set equality with `gen_register.py`, D-21..D-30 added).
- R0-H2 SETTLED / DEAD-END / AGENTS-contract text rewritten as "safe" edits -> Section 3 rule 12: Wave 1 = verified facts + dated markers + fact entries only.
- R0-H3 `undersized_asset_ratio` mirrors the crop fix and A1's area floor regresses correct rescues / yields wrong-region crops on the cloud frame -> measure-first design study, frame-invariant sidecar, predicate-independent `crop_prose_fraction`, cloud-route figure acceptance stated non-claimable unless the frame is confirmed.
- R0-H4 parent-heading metric zero by construction and Q3 tracks VLM labelling -> source-anchored section spec with `engine_labelled` N/A handling.
- R0-H5 gate 5 "run as-is" -> pinned offline before/after with a pass criterion.
- R0-H6 WP-B2 deletes the HarryPotter heading source -> label-aware, heading-excluding, measure-first.
- R0-B (feasibility): frame-dependent asset metric, inline reference shape, QA-CHECK-01 -23% to about -31% after B2, boilerplate hardening regressing real copyright pages, E1 scope/rounding/per-batch, A2 in-loop drops and missing drop sites, D6 per-chunk attribution impossible, B3 sub-items without consumers.
Round 0b (verification that rev. 2 fixes hold) results, if any, are appended below by the plan author as `R0b-*`.

---

## 4. Systematic-bias callout (read this before you start)

The plan's author is the same agent that ran the investigations and wrote the fixes' designs. Self-audits also tend to avoid
restructuring the load-bearing artifact and instead produce fixes that are ADDITIVE (add text, add steps, add metrics) rather than STRUCTURAL
(re-dispose an item, merge or delete a work package, admit a premise is false). Expect to find at least one or two structural findings the
internal Round 0 dodged. Specific biases to hunt for in THIS plan:
1. **Closure bias**: wanting Waves 0-2 to read as "done" (the repo's own memory records a prior "make-the-failing-run-pass" incident).
2. **Additive bias**: six new advisory metrics plus a gold-anchor script plus 10 code WPs; is any of it unnecessary, or a proxy of the fix it guards?
3. **Scope inflation**: about 140 governance findings and 83 backlog items were reduced to clusters; did clustering hide items?
4. **Route bias**: fixes target engine-agnostic code while the settled engine is unreachable; did the plan quietly redefine "production"?
5. **Investigator optimism**: subagent reports mix VERIFIED and INFERRED claims; did the plan promote an INFERRED claim to a fact?
6. **Precedent evasion**: did any WP re-propose a measured-and-rejected approach under a new name?
7. **n=1 overfitting**: the live acceptance is a single 7-page document.

---

## 5. Background context

Project: multimodal PDF-to-JSONL ETL for RAG. Two namespaces: `src/mmrag_v2/` (baseline, chunker, materializer, QA) and `src/mmrag_v3/`
(vision-native extraction). Extraction entry `mmrag_v3.extract()` selects an engine: with `MINERU_ENDPOINT` set the documented production default is
the MinerU2.5 + Qwen-for-code hybrid; with it unset (the situation on 2026-09-29 because the LAN servers were down) it falls back to the legacy
`HybridEngine` (Docling prose + a VLM, here cloud `qwen3-vl-flash`). The IRJET run used the fallback.
Project state: `feat/omnidocbench-phase0` tip `499a5fa` (2026-06-18); work branch `fix/quality-remediation-v1` (local only, never pushed; the public `github` remote is never touched).
Why this plan: an external verdict on one conversion; verification showed most claims real, root-caused them, and showed the gates cannot see them.
Load-bearing owner constraints (a fix that violates one must be FLAGGED, not silently proposed): no test/assertion/fixture may be removed or weakened (AGENT-TEST-01);
no threshold relaxation to make a run pass; no filename- or document-specific rules; commit locally only, never push; no human-verification loops as a mitigation for system fragility
(one bounded human acceptance review at a phase boundary is the sanctioned exception); user-directed deferral of VLM serving/model+prompt work "until the first production-level release" (PROJECT_STATUS, 2026-06-18);
ASCII punctuation only; SETTLED/DEAD-END items in FINDINGS_DIGEST are not re-opened without new evidence (AGENT-PRECEDENT-01).
External-state failure modes the plan depends on: (a) a cloud VLM API key and credits (paid; results non-deterministic, two live IRJET extractions were 84% identical); (b) LAN servers (omlx, GX10, M5) down; (c) `data/` and `output/` are gitignored, so acceptance evidence that lives only there is invalid under AGENT-EVIDENCE-01; (d) the conda env `mmrag-v2` (Python 3.10) is required to run tests; (e) Qdrant/embedder availability is unknown.
Budget/scope: no batch or corpus conversion in this plan; single-document smokes only; no re-ingestion.

---

## 6. Audit lenses (numbered; you must engage with EACH, even to report "nothing to flag")

### L1. Disposition-register integrity (the load-bearing artifact) - adversarial
Did the author systematically avoid the items they were uncertain about? Take the four audit registers and the plan's Section 7.
Sample at least 25 finding ids across INV-, G0-, DC-, AP- and confirm each resolves to exactly one disposition with an owner and trigger. List every id that is
absent, double-counted, or dispositioned into a cluster whose stated disposition does not actually cover it (for example a USER-DECISION swept into a SAFE-EDIT cluster).
Is "BLOCKED (owner=user, trigger=D-2 answered)" a real closure or a deferral relabelled? If the owner never answers, what forces closure and when?

### L2. Did any prior audit dodge restructuring? (meta-lens)
Did ANY finding from the four registers or from Round 0 restructure the plan (delete/merge a WP, re-dispose an item, admit a premise false), or were all changes additive?
If all were additive, say which hard question was dodged.

### L3. Precedent honesty (AGENT-PRECEDENT-01)
For each Wave 2 WP, read the matching SETTLED / DEAD END lines in `FINDINGS_DIGEST.md`, the "Settled Precedents" index and FINDINGS_LOG F12/F13, and decide whether the WP re-proposes a
rejected approach under a new name. Specific suspects: WP-B2 (furniture pass + merge veto vs F12 "spatial boundary-repair bridge REJECTED on VLM-native" and the `_merge_mid_sentence_chunks` decision that keeps it live);
WP-A1/A2 (retaining images vs the +0.0pp dead end on filtering empty images); WP-K1 (VLM adapter knob vs "engine-swap reflex / more scaffolding around a single general VLM"); WP-Q1 (prompt edit vs the settled R3 prompt property).
Is the plan's "new ground" statement honest, or is it the old argument restated?

### L4. Outcome-metric validity, independence and calibration (metric-validity lens)
For each of the six metrics (`undersized_asset_ratio`, `orphan_snippet_ratio`, `heading_inside_body_ratio`, `furniture_line_ratio`, `reference_entry_violations`, `figure_deficit_pages`):
write the worst-case failure the paired fix could introduce, then decide whether the metric can move when that failure happens BY CONSTRUCTION (not just in practice). Concrete tests to run in your head or on the attachments:
- `undersized_asset_ratio` divides by the bbox area; the cloud route's bboxes are shrunk by about 0.55 in area. Can a displaced crop of the right size (wrong region) pass? Can a correct crop fail? Is the metric independent of the fix predicate (`rect.area >= 0.5 * vlm_clip.area`), i.e. does it merely re-measure the fix's own rule (AGENT-INTEGRITY-01 "mirrors its own fix")?
- `furniture_line_ratio` is described as an upper bound: legitimate repeated lines (figure captions, table header rows repeated per page) count. What false-positive rate on the local crucible outputs is acceptable before it is noise, and is the threshold to be set from principle or from the failing document?
- `heading_inside_body_ratio` uses "a known parent_heading string appears as a non-first line": after WP-B1 stamps section headings, is the metric trivially 0 by construction (a circular guard)?
Then apply the seeded-fault method: for each metric, name a fault it would NOT see (compare `output/seeded_fault_report.json` and `tests/test_seeded_fault_sensitivity.py` for how the project measured blindness).

### L5. Validation-fixture answer-correctness
WP-0.3's gold-anchor smoke: are the five anchors ANSWER-level (the equation intact in one chunk, the whole reference entry in one chunk) or doc/format-level? Q3's anchor ("`parent_heading` equals the nearest preceding heading in output order") is computed from the output itself: can it pass on a wrong chunk from the right document? Propose the missing anchor if any. Are synthetic fixtures (no copyrighted text) faithful enough to the defect shape to fail on current code, and did the plan require the red run to be recorded?

### L6. Trigger gaming, unfireable-by-design, input-intrinsicness (for the decision-gated WPs)
WP-Q1, WP-O1, WP-K1 and the "element-atomic packing only if live output shows cuts" clause are CONDITIONAL. For each: could the trigger be read permissively? Can the gating measurement actually produce the trigger (an IRJET-only firing rate cannot justify or refute a corpus-wide change)? Does the trigger evaluate inputs INTRINSIC to the technical question (the failure rate of the defect) or extrinsic ones (whether the owner happened to answer D-4, what documents happen to be on disk)?

### L7. Soft-state-in-disguise and cap-as-deferral
The plan forbids "watch items" (AGENT-STATUS-01) yet uses BLOCKED with owner-answer triggers and DEFER per Mandate 3(c). Count them. Are any of them the deferral pattern relabelled? The plan sets NO spend or wall-clock cap for the cloud route (D-20 is undecided) and defers the multi-document regression (WP-V5) to that answer: is the absence of a cap a hidden unlimited-spend risk, and does deferring the regression set turn "acceptance" into n=1? What is the convergence-compatible alternative (cycle fails / explicit KILL)?

### L8. Ship-gate vs governance-invariant consistency
Grep `CLAUDE.md`, `AGENTS.md`, `docs/V3_EXECUTION_MANDATE.md`, `docs/QUALITY_GATES.md`, `docs/TESTING.md` for "must", "required", "invariant", "acceptance", "shall". Verify every named gate/suite/invariant appears in plan Section 8.2. Known tension the plan lists but may mishandle: AGENT-VAL-01 (`smoke_multiprofile.sh`) vs the Mandate's `smoke_production.sh`, and the fact that `smoke_multiprofile.sh` exits 0 with failing rows. Is "run and report as-is" a real gate or an escape hatch? Is QA-CHECK-01 (docs say 0.10 for all profiles; code allows -25% for academic) handled without silently choosing a side?

### L9. Cross-file consumer completeness
For every canonical constant, schema field or list a WP changes, grep `src/`, `scripts/`, `tests/` and enumerate ALL consumers; list any the plan misses. Minimum set: `_HEADING_LABELS`, `pipeline_version` / `config_hash` consumers (`scripts/build_corpus_manifest.py`, `scripts/qa_conversion_audit.py`, tests with "2.7.0" fixtures, `manifest_status.py`), `breadcrumb_path` / `level` (`to_embedding_text`, ingest prefix, REQ-HIER-04), `content_classification` (which gates key on null vs set), `next_text_snippet` (contextual retrieval at ingest), `visual_description` (ingest embedding text), `_FURNITURE_*` constants and `count_running_furniture`, the whole-dict pin in `test_extraction_provenance_consumers.py`. Is a bridge test needed (memory: cross-object flags need call-site bridge tests)?

### L10. Exhaustiveness depth (lapsed conditionals and superseded solutions)
Walk back through the plan docs and the digest and identify every conditional or deferred disposition ("if X ships", "defer until", "time-boxed to", "revisit when"). For each, decide whether the gating condition fired and whether the plan gives it an explicit re-disposition. Check specifically: review follow-up #8 ("defer to F3"), the B2 furniture filter (memory says unbuilt, code says half-built), MAX_CARRY_FORWARD_PAGES, the Phase 4 rollback trigger, the Charter "cloud fallback and cost ceiling" (condition fired 2026-09-29), PLAN_F1 P2 "blocked by chunker" (falsified). Also superseded-but-undisposed solutions (F1 chunk-level furniture vs the new element pass; two endpoint registries; two heading helpers). Also in-tree opt-in code with no users and implicit governance intent ("we always intended X") that was never tracked.

### L11. Self-contained rationale and factual verifiability
Every KILL / CLOSE rationale in plan Section 10 and Section 7.3 must stand alone (no "per audit", no reference to gitignored files as the ONLY evidence). Sample at least 4 KILL/closure rationales and verify their factual premises against the repo with commands, for example: the pinned tests named in Section 10 exist and assert what is claimed (`test_b1_uses_vlm_bbox_when_no_geometric_object`, `test_icon_class_image_on_content_page_is_dropped`, `test_thin_strip_on_content_page_is_dropped`); commit `2ec40f5` reverted a profile seam; the merge conflict counts for `github/laptop-travel` (`git merge-tree`); `smoke_multiprofile.sh` exit behaviour. Any false premise invalidates the disposition regardless of the conclusion.

### L12. Operational fragility
Where is the plan methodologically clean but operationally fragile? Consider: acceptance depends on a paid, non-deterministic cloud call; the metrics are calibrated on local gitignored outputs that differ per machine; the plan says the chunker A/B uses a fixed UIR but the raw UIR of the failing run was never saved (Round 0 recommended persisting it: is that a WP with an owner?); key handling when `DASHSCOPE_API_KEY` is absent; what if the fresh acceptance run itself fails or degrades (`extraction_degraded_pages > 0`)?

### L13. SWE-standard completeness
Per WP: is there a rollback, is the merge-conflict order for the shared files (`uir_chunker.py`, `batch_processor.py`) real, is there a security review of new code paths, a test-coverage gate (red-first evidence), a documentation strategy, and a migration path for chunk-id churn (D-8)? Name any WP missing one.

### L14. Post-cycle sustainability
Who maintains the six advisory metrics in six months? They are advisory-only by AGENT-GATE-PROGRESSION: what is the promotion or KILL trigger so they do not become permanent decoration? Do the DECISIONS entries the plan requires explain WHY for a maintainer without tribal knowledge? Does the plan create any KEEP-active infrastructure that needs future cycle-opens to work (post-freeze lifecycle)?

### L15. Governance-edit safety (Wave 1)
Wave 1 rewrites Layer-0 documents from audit findings. Which SAFE-EDIT classifications are actually policy changes in disguise (for example G0-04, G0-05, G0-08, DC-05)? Is the rule "edit only what is re-verified against code" enforceable? Are there edits that touch a SETTLED item or the G7-enforced digest structure without owner sign-off? Is restructuring `PROJECT_STATUS.md` safe under the repo's "whole-file sweep" rule and G1-G7 (`tests/test_repo_integrity.py`)?

### L16. Route and decision structure
Are D-1..D-20 the right decisions, correctly owned, with defaults that do not smuggle a policy change? Is any decision missing (for example: what "done" means for this plan if D-1 is never answered; who decides the Round-1 disagreements)? Does the recommended D-1(c) "ratify an interim route with an expiry trigger" itself violate AGENT-STATUS-01 (an interim state with an unmeasured route)?

### Required meta-lens M1. Self-deception test
Take any three "FIX" WPs and ask: if the fix silently did nothing, which acceptance line would still pass? If the answer is "all of them", the acceptance is proxy-based; say so and propose the outcome check.
### M2. Which single WP would you cut, merge or KILL, and why? (Answering "none" requires evidence.)
### M3. What is the cheapest experiment that would falsify the plan's biggest assumption, and does the plan run it?

---

## 7. Anti-escape-hatch "Don't" list

- Don't suggest "defer to the next cycle / a later version" as a fix unless the item is genuinely out of scope for this product.
- Don't accept rationales that are assertion-backed rather than evidence-backed ("no demand signal" is weaker than "no query in N cycles surfaced this"). Challenge anything that sounds convincing without a cited command or file:line.
- Don't let labels become euphemisms: FIX / DECIDE / CLOSE / BLOCKED / KILL / SETTLED / DEAD-END / MOOT must each mean what they say; if a label disguises a soft deferral, name it.
- Don't recommend adding process steps ("add a checkpoint", "add monitoring") as a substitute for fixing a structural problem.
- Don't accept KILL or closure rationales with factually false premises; verify against the repo.
- Don't accept "KEEP active" on cycle-dependent infrastructure without a post-freeze lifecycle plan.
- Don't accept overflow-routing as a disposition for cap-exceeded work: exceeding a budget or wall-clock means the WP fails or is explicitly KILLed.
- Don't accept numeric thresholds without checking the measuring instrument is calibrated for the content class, and that the metric can move when the gated failure occurs.
- Don't recommend a fix that violates an owner constraint listed in Section 5; if the gap is real, reframe as "the constraint should be re-litigated, owner decision required".
- Don't treat one auditor's "nothing to flag" on a lens as proof the lens is empty.
- Don't ask for "more validation" or "more testing" without naming the specific failure mode the extra validation would catch and the input that triggers it.
- Don't pad findings to reach a count; 0 HIGH is a valid, useful result and is the stopping-rule signal.
- Don't conflate "the recommended fix is wrong" with "there is no problem": where a Round 0 or registry finding was rejected, re-read its DIAGNOSIS separately from its prescription; a different fix may still be needed.
- Don't suggest new features beyond the plan's scope; this plan converges open work, it does not add product capability.
- Don't critique decisions already locked in as constraints (Section 5 list); you may say a constraint itself deserves re-litigation, with owner-decision framing.

---

## 8. Output format (a calibrated verdict; vague approval is forbidden by the format)

For each finding:
`[HIGH|MED|LOW] - plan section - issue - concrete failure mode (input -> wrong outcome) - recommended fix - confidence (high/medium/low) - evidence (file:line or command output)`.
Severity: HIGH = the plan would leave a defect closed-on-paper, or would cause a wrong/irreversible action, or violates a governance invariant; MED = real gap that weakens acceptance; LOW = polish.

Then, mandatory sections:
1. **Audit lenses with nothing to flag**: one line for EACH of L1-L16 and M1-M3 that produced no finding, stating what you checked and how (so silent skipping is impossible).
2. **Structural changes I recommend** (cut, merge, re-disposition, reverse a premise) separated from additive suggestions; state how many of each.
3. **Disagreements with Round 0**: which Round 0 fixes do not hold.
4. **Overall stance** (all five required):
   a. Is the load-bearing artifact discipline (disposition register + outcome harness) correctly applied? (yes / partly / no, with the decisive reason)
   b. Is the plan shippable as-is, does it need revision before the next audit round, or does it need substantive restructuring?
   c. Which disposition or scope changes do you recommend, and why?
   d. Your confidence that the disposition register and the outcome metrics are a sound foundation for what the plan claims to deliver (a number, with the main uncertainty).
   e. Your HIGH / MED / LOW counts, so the owner can apply the stopping rule.
A stance of "looks good" without these five items is invalid.
