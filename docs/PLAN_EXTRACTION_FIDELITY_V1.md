# PLAN_EXTRACTION_FIDELITY_V1 - Extraction Thesis Re-evaluation (candidate: pipeline-primary, VLM-as-specialist)

Status: APPROVED FOR EXECUTION (2026-06-10, rev. 3 - external review APPROVED rev. 2
with 3 non-blocking refinements, folded in here). Layer-0 edits still gated on the
Phase 1 outcome.
Owner: extraction + QA
Decision authority: user (2026-06-09 thesis direction; 2026-06-10 review conditions +
rev. 2 approval).
Depends on / drives:
- `PLAN_OMNIDOCBENCH_EVAL` (its Phase 1 extractor bake-off becomes the SELECTION
  mechanism; must be unblocked first).
- `PLAN_GATE_QUALITY_V1` (its retrieval-value axis is the orthogonal partner to the
  fidelity outcome gate defined here, and the internal-corpus half of selection).
- Governance audit 2026-06-09 (this plan supersedes the audit's "reconcile docs to
  code" disposition for F1/F3/F5 - see Section 9).

This is a Layer-2 execution plan (AGENT-DOCS-01: plans may be added freely). It does
NOT itself edit any Layer-0 contract; Section 8 lists the Layer-0 edits it authorizes
ONCE the candidate thesis is proven or refuted.

**Rev. 2 review conditions folded in (2026-06-10):** (1) pipeline-primary is a
CANDIDATE pending the bake-off, not a foregone conclusion; (2) conversion-time
"fidelity arbitration" is reframed as calibrated QUALITY-RISK arbitration (proxies,
not measured fidelity); (3) selection requires the INTERNAL corpus alongside
OmniDocBench, never the benchmark alone; (4) a dedicated VLM operational-profiling
phase (resolution / latency / thermal / token cost) is added - the M5 load is the
operational pain point that started this and the architecture work must not bury it.

**Rev. 3 refinements folded in (2026-06-10, on rev. 2 approval):** (5) Phase 0A also
measures throughput/queue-depth (GPU busy %, tokens/sec, pages/hour, avg+peak queue
depth), not just per-page latency - a never-idle server profiles differently from a
bursty one; (6) Phase 0A scores every render setting on BOTH OmniDocBench AND a
representative internal-corpus subset (a DPI cut harmless to prose can destroy tiny
wiring-diagram labels); (7) Phase 0's objective is reworded to "enable a clean
bake-off" (fix MinerU *sufficiently for evaluation* OR substitute an alternative
serving path), NOT "fix MinerU to production-perfect" - the further-MinerU-investment
call is deferred to AFTER Phase 1 data exists.

---

## 0. Why this plan exists

A governance/code audit (2026-06-09) plus git archaeology established that the V3
extraction layer ships a reliability story that LOOKS solid on its gates while its
actual extraction FIDELITY under degradation is unmeasured and probably degraded. The
chain of evidence:

1. **A real bug was fixed, on top of unresolved problems.** Commit `fcd4207`: the M5
   MinerU server's intermittent `broadcast_shapes` 500s made `extract()` return an
   empty document with `failed=0` / zero chunks / no error - silent data loss. The
   fail-closed 3-tier ladder (`fcd4207`, `e73fbd8`, `1cd20fe`) fixed THAT. Good. But
   it was built over three corner-cuts (Section 4).

2. **The fragile component was routed-around, not fixed.** The root cause is an
   intermittent, RETRYABLE server fault ("same page 500s on one call and returns 1706
   chars on the next" - `fcd4207` message). The correct first response is idempotent
   retry of the PRIMARY engine and a fix to the MinerU batching fault. Instead
   `extract()` calls the engine once and drops straight to a weaker engine
   (`src/mmrag_v3/processor.py` tier-1 -> tier-2).

3. **Fallback is accepted on text-PRESENCE, not quality.** The only acceptance test
   is `_page_content_chars(fb_page) > 0` (`processor.py:361`). A Docling page that
   strips code indentation or flattens a table is accepted as a "recovery" and passes
   the structural gates. This is the project's own AGENT-INTEGRITY-01 failure ("assert
   OUTCOMES, not PROXIES") one layer up: the gates assert "chunks produced," not
   "faithful chunks produced."

4. **The "smart-when-healthy, reliable-always" default is asserted, not measured.**
   FINDINGS_LOG 2026-06-09: "the benchmark is now the decision oracle ... any 'smart'
   lane must beat the offline floor by a margin worth its failure surface or it gets
   cut." Sound principle - but the bake-off that would prove the MinerU+Qwen smart
   lane beats the cheap offline Docling floor came back INCONCLUSIVE (blocked by the
   same server fault the ladder routes around).

**The precise claim this evidence supports (and the one it does NOT):** it shows
VLM-primary is UNPROVEN, not that it is WRONG. OmniDocBench favours pipelines, the
bake-off was inconclusive, and some scanned pages favoured Docling - that is enough
to STOP asserting VLM-primary, not enough to CONCLUDE pipeline-primary. So the docs
must not be reconciled DOWN to the current code (that enshrines a corner-cut), the
old "halt, no fallback" docs were not better (they were implemented as silent data
loss), AND this plan must not swing to the opposite unproven assertion. The thesis is
a CANDIDATE; the bake-off decides it.

---

## 1. The candidate thesis (decided by Phase 1, not by this document)

**Current (unvalidated) thesis:** vision-native extraction is the primary; a
deterministic engine is a degraded fallback.

**Candidate thesis (to be PROVEN OR REFUTED in Phase 1):** a specialized
OCR/document-parsing PIPELINE is the primary extractor; the VLM is a TARGETED
SPECIALIST invoked only for the content-classes where a pipeline is MEASURED to lose
(e.g. charts/diagram-to-data, complex full-bleed visuals, code-indentation).
Extraction is COMPLEMENTARY and quality-ARBITRATED, not primary-plus-blind-fallback.

This candidate is ELEVATED to the architecture only after Phase 1 shows the chosen
pipeline beats the VLM hybrid on BOTH OmniDocBench AND the internal corpus by a
margin worth the change. If Phase 1 shows VLM-primary actually wins on our corpus,
the thesis is REFUTED and the default stays VLM-primary (now validated) - the only
non-negotiable outcomes are: retry-before-fallback, quality-risk arbitration, and a
measured (not asserted) default. The specific pipeline engine, if chosen, is selected
EMPIRICALLY - candidates already in scope: `DoclingFastEngine` (in-tree offline
floor), MinerU's pipeline/cascade backend, PaddleOCR-VL (served on M5).

---

## 2. Literature basis (verified 2026-06-09)

- **Specialized pipelines beat general VLMs on parsing fidelity.** OmniDocBench
  (CVPR 2025): "pipeline tools specifically designed for document parsing demonstrate
  superior performance across the board" - MinerU pipeline text edit distance 0.058
  (EN) vs GPT-4o 0.144. ([arXiv:2412.07626](https://arxiv.org/html/2412.07626v1))
- **2026 state of the art reinforces it:** specialized OCR models (GLM-OCR 94.6%,
  PaddleOCR-VL, DeepSeek-OCR2) now beat frontier VLMs (Gemini 3 Pro 90.3%) on
  OmniDocBench. ([LlamaIndex: OmniDocBench is saturated](https://www.llamaindex.ai/blog/omnidocbench-is-saturated-what-s-next-for-ocr-benchmarks))
- **VLM document-parsing failure modes are structural,** not incidental: multi-column
  reading-order errors, dropped/misaligned table content, rotated tables/text,
  colored backgrounds. ([arXiv:2412.07626](https://arxiv.org/html/2412.07626v1))
- **The accepted architecture is complementary, not replacement:** native text +
  per-element specialized extractors + VLM in concert, with structure preserved.
  ([NVIDIA NeMo Retriever](https://developer.nvidia.com/blog/how-to-build-a-document-processing-pipeline-for-rag-with-nemotron/),
  [NVIDIA RAG VLM](https://docs.nvidia.com/rag/latest/vlm.html))
- **Reliability pattern:** retry transient faults first; fall back to a degraded
  representation only when a degraded state is explicitly acceptable AND flagged -
  never silently as equivalent.
  ([LLM parsing-error handling](https://apxml.com/courses/prompt-engineering-llm-application-development/chapter-7-output-parsing-validation-reliability/handling-parsing-errors))
- **Measure fidelity, not presence; per-content-type.** OmniDocBench advocates
  attribute-level evaluation (TEDS for tables, edit distance for text, reading-order
  metric). ([arXiv:2412.07626](https://arxiv.org/html/2412.07626v1))

Two caveats this plan inherits: (a) edit-distance/TEDS penalize functionally
equivalent formatting - the goal is SEMANTIC correctness, so the gate is
regression-against-baseline, not an absolute floor; (b) OmniDocBench is English +
Chinese only - it cannot speak for our German/Dutch/automotive/wiring-diagram
corpus, which is why selection is two-corpus (Section 7, I2).

---

## 3. Target architecture (as-should-be, thesis-conditional)

Canonical flow stays UIR-centric and respects all existing boundaries (extraction in
`mmrag_v3.extract`, zero Docling in `batch_processor`, AST firewall, ElementType=3 /
Modality=5):

```
mmrag_v3.extract(path)
  -> route per page by CHEAP pre-flight signals (existing router)
       primary engine    (the bake-off winner; default for prose/tables/forms/scans)
       specialist lane    (only for measured-loss classes: charts, complex
                           visuals, code-indentation)
  -> per-page RETRY of the chosen engine on transient fault (idempotent, bounded)
  -> QUALITY-RISK ARBITRATION at the merge point:
       accept a page only if it clears its per-modality quality-RISK bar (proxies,
       Section 5.3); else QUARANTINE the page (provenance: extraction_quality_risk)
       - never silently substitute a lower-quality representation as equivalent
  -> fail-closed SAFETY NET (the existing 3-tier ladder) as LAST RESORT only,
     reached after retry + arbitration, and always provenance-stamped
  -> UniversalDocument -> chunk_universal_document -> from_uir -> JSONL
```

Key shifts from today:
- **Default engine = the bake-off-selected winner** (pipeline OR VLM hybrid - decided
  by Phase 1 on two corpora), not an asserted choice.
- **Specialist lanes are opt-in-by-evidence**, scoped to measured-loss content classes.
- **Retry precedes fallback.** The ladder is demoted from "the reliability story" to
  "the last-resort net under a retry-first, quality-gated design."
- **Arbitration is quality-RISK-aware, not fidelity-measuring.** Presence
  (`chars > 0`) is necessary, not sufficient; but with no ground truth at conversion
  time we can only ESTIMATE risk from proxies. A page that only a lower-quality tier
  could serve is FLAGGED so the offline fidelity gate and any human review can see it;
  it never passes as equivalent.

---

## 4. What is wrong today (mapped to the governance audit)

| ID | Defect | Evidence | Audit finding |
|---|---|---|---|
| X1 | No primary retry before cross-engine fallback | `processor.py` tier-1 single call | new (this plan) |
| X2 | Fallback accepted on presence, not quality | `processor.py:361` `chars > 0` | new (this plan) |
| X3 | Default route's superiority unmeasured | OmniDocBench Phase 1 INCONCLUSIVE | F5 |
| X4 | Reliability behavior contradicts the as-built docs | `processor.py:294` vs charter B4/section 4, PROJECT_STATUS:417 | F1 |
| X5 | As-built charter silent on the shipped ladder | charter section 3/4 | F3 |
| X6 | MinerU server `broadcast_shapes` fault unfixed | FINDINGS_LOG 2026-06-09 | new (this plan) |
| X7 | M5 thermal/throughput load unprofiled (every page rendered ~1700x2200 and shipped to the VLM) | operational pain point; no measurement on record | new (rev. 2) |

F2/F4/F6/F7/F8/F9 from the audit are doc-hygiene fixes independent of the thesis and
are folded into Section 8 (the spec rewrite). (Audit F-IDs and this plan's X7 are
distinct numbering spaces.)

---

## 5. Reliability + quality model correction (replaces the silent-fallback design)

1. **Retry-first.** On a transient fault (timeout / connection / 5xx incl. MinerU
   `broadcast_shapes`), retry the SAME page on the SAME engine with bounded backoff
   before any cross-engine move. Reuse the existing `vlm_provider` retry machinery;
   extend it to the MinerU path. Pulled EARLY (Phase 0.5) - it is a pure operational
   fix with no thesis dependency. (X1, X6.)
2. **Fix the fragile component.** File and fix the MinerU `broadcast_shapes` batching
   fault at the server (it batches different-sized block crops), or move MinerU to a
   GX10 vLLM endpoint as PLAN_OMNIDOCBENCH_EVAL section 13 already contemplates.
   Routing-around is the net, not the fix. (X6.)
3. **Quality-risk arbitration (NOT a fidelity claim).** At the merge point, a
   fallback/secondary page is accepted only if it clears a per-modality quality-RISK
   bar; otherwise the page is retained but PROVENANCE-FLAGGED `extraction_quality_risk`
   (new key alongside the existing `extraction_*` stamps). These bars are PROXIES
   (Section 5.3) - explicitly NOT measured fidelity, which needs ground truth we do
   not have at conversion time. Code/table pages - the specialist's reason to exist -
   are never silently replaced by an indentation-stripping/flattening tier. (X2.)
4. **Keep the safety net, demoted.** The 3-tier ladder stays as the LAST resort under
   retry + arbitration. It is strictly better than the prior silent empty-doc loss
   and must not be removed; it must stop being described as the primary reliability
   story. (X4, X5.)

### 5.3 The quality-risk proxies (honest about what they are)

No ground truth exists at conversion time, so these ESTIMATE quality risk; they are
calibrated against OmniDocBench LABELED edit distance (Section 6) so we know the
correlation, but they remain proxies. Candidates: table-structure validity (parses to
a rectangular grid), code-fence/indentation integrity, reading-order monotonicity,
degenerate-repetition score, empty-region ratio. A page failing a proxy is FLAGGED,
not declared low-fidelity. Do NOT replace `chars > 0` with `proxy_score > threshold`
and call it fidelity - that is the false-confidence trap; the flag means "elevated
risk, look here," and the offline OmniDocBench gate (Section 6, real ground truth) is
the only place a true fidelity verdict is issued.

No change violates a locked invariant: extraction stays in `mmrag_v3.extract`,
`batch_processor` keeps zero Docling, the AST firewall and ElementType=3/Modality=5
are untouched.

---

## 6. The fidelity OUTCOME gate (AGENT-GATE-PROGRESSION compliant)

This is the ONE place a true fidelity verdict is issued - it has labeled ground
truth, runs OFFLINE, and is distinct from the conversion-time quality-risk proxies
(Section 5.3). Follows the advisory->hard protocol in AGENTS.md
AGENT-GATE-PROGRESSION:

- **Instrument:** OmniDocBench labeled metrics (text edit distance, table TEDS,
  reading-order edit distance) via the already-built `scripts/omnidocbench_adapter.py`
  (PLAN_OMNIDOCBENCH_EVAL). English subset first.
- **Advisory phase:** report the fidelity delta vs the recorded Phase-0 baseline
  (full-755: text ED 0.301 / TEDS 0.563) as an advisory in the two-axis acceptance.
  It is NOT a per-conversion production gate (no ground truth at conversion time) -
  it is an offline selection + regression gate, per PLAN_OMNIDOCBENCH_EVAL section 2.
- **Promotion to hard regression gate:** only after (a) the bake-off can run cleanly
  end-to-end (X6 cleared) and (b) the chosen route is shown stable across doc classes
  on BOTH corpora. Per the literature caveat, the bar is REGRESSION-against-baseline
  (no fidelity loss vs the selected baseline), not an absolute exact-match floor.
- **Anti-weaken:** no hard gate may be passable by WEAKENING extraction (e.g.
  dropping hard pages). The fidelity gate pairs with PLAN_GATE_QUALITY_V1's
  retrieval-value axis so "clean transcription of junk furniture" cannot pass both.

---

## 7. Phasing (each phase mechanically checkable; V3_EXECUTION_MANDATE section 2 style)

- **Phase 0 - Enable a clean bake-off (objective is RESULTS, not a perfect MinerU).**
  Get a Phase 1 bake-off able to run clean end-to-end (X6) - EITHER by fixing MinerU
  serving *sufficiently for evaluation* OR by substituting an alternative serving path
  (the engines that already serve reliably: docling_fast, PaddleOCR-VL, Qwen). This is
  benchmark-capable, not production-perfect; the call on whether MinerU deserves
  further engineering investment is DEFERRED to after Phase 1 (do not sink weeks into
  MinerU only to have the bake-off pick another engine). DoD: `scripts/omnidocbench_bakeoff.py`
  completes for the available candidates with no server fault; `scripts/smoke_production.sh`
  -> `SMOKE_PRODUCTION_PASS`.
- **Phase 0A - VLM operational profiling (the M5-load question).** Before any
  architecture redesign, measure what the VLM path actually costs. Per-page:
  rendered page dimensions, image payload size, VLM latency/page, GPU utilization,
  thermal load on the M5. Throughput/saturation (rev. 3): GPU busy %, tokens/sec,
  pages/hour, average + peak request queue depth (a never-idle server profiles
  differently from one fed in bursts - distinguishes a model-bound from a
  preprocessing-bound bottleneck). Then sweep render settings (200 DPI, 150 DPI,
  1600px cap, 1400px cap) and record the extraction-quality impact of each on BOTH
  corpora (rev. 3): an OmniDocBench set AND a representative internal-corpus subset
  (Dutch manuals, wiring diagrams, automotive) - a DPI cut harmless to OmniDocBench
  prose can destroy tiny wiring-diagram labels. DoD: a recorded table mapping render
  setting -> (latency, GPU busy %, tokens/sec, pages/hour, peak queue depth, peak
  temp, token cost, OmniDocBench fidelity delta, internal-corpus delta) and a
  recommended default render setting whose impact is measured on both. Rationale:
  today every page is rendered at ~1700x2200 and shipped to Qwen; the real bottleneck
  may be resolution/token count, not architecture, and that must be known before
  weeks are spent on routing.
- **Phase 0.5 - Retry-first (pulled early).** Idempotent bounded retry of the PRIMARY
  engine on transient fault before any cross-engine move; extend to the MinerU path
  (Section 5.1). DoD: tests for retry-before-fallback; `SMOKE_PRODUCTION_PASS`. No
  thesis dependency - ships independently of the bake-off outcome.
- **Phase 1 - Select the default engine empirically, on TWO corpora.** Run the
  bake-off (this IS PLAN_OMNIDOCBENCH_EVAL Phase 1, now unblocked) AND the internal
  crucible corpus (German/Dutch/automotive/wiring classes OmniDocBench cannot cover,
  scored on PLAN_GATE_QUALITY_V1's retrieval-value axis). DoD: two comparison tables
  (OmniDocBench fidelity + internal retrieval-value) in FINDINGS_LOG; the candidate
  pipeline-primary thesis explicitly CONFIRMED or REFUTED with margins on both; the
  winning default named.
- **Phase 2 - Identify the specialist lanes by evidence.** From the same two-corpus
  bake-off, identify the content-classes where the chosen default loses and a
  specialist lane wins. DoD: the per-class routing table recorded; a lane is CUT only
  if it shows no measured win on the benchmark AND no measured win on the internal
  corpus (rev. 2 - never cut on OmniDocBench alone; that overfits the benchmark and
  can damage the real workload).
- **Phase 3 - Reliability + quality correction.** Implement quality-risk arbitration
  + demote the ladder (Sections 5.3, 5.4). DoD: new tests for quarantine-on-quality-
  risk and ladder-as-last-resort; full suite green (the existing 1 known
  `test_v3_vlm_code_form` failure fixed or registered first); `SMOKE_PRODUCTION_PASS`.
- **Phase 4 - Re-route the default.** Flip the default route ONLY to whatever Phase 1
  proved (pipeline-primary + evidence-based specialist lanes, or validated
  VLM-primary). DoD: routing tests assert the new default; bake-off-confirmed on both
  corpora; two-axis acceptance (fidelity advisory + retrieval-value) green.
- **Phase 5 - Spec rewrite.** Apply the Section 8 Layer-0 edits to match the
  now-PROVEN reality. DoD: `tests/test_repo_integrity.py` green; governance audit
  F1/F3/F5 closed.

---

## 8. Layer-0 edits this plan authorizes (apply in Phase 5, after the thesis is proven)

These describe whatever Phase 1 proves; the bracketed conditionals are resolved by
the bake-off, not pre-decided here.

- `docs/ARCHITECTURE_V3.1_CHARTER.md`: rewrite section 1 framing and section 3
  as-built flow to the [bake-off-proven] default + specialist lanes; rewrite section 4
  resilience to retry-first + quality-risk-arbitration + ladder-as-last-resort
  (closes F1/F3); add the fail-closed-ladder + `extraction_*` / `extraction_quality_risk`
  provenance subsection.
- `docs/V3_EXECUTION_MANDATE.md`: fix line 18 (`smoke_production.sh` is SHIPPED, not
  "not yet built"; `visual_description` conditional on FULL mode) - F2; add the
  fidelity outcome gate to the Definition of Done once promoted.
- `docs/QUALITY_GATES.md`: add the fidelity outcome gate (advisory first) per
  Section 6; document the conversion-time quality-risk proxies as advisory signals.
- `docs/DECISIONS.md`: new entry recording the [proven] thesis + the reliability/
  quality model correction (supersedes the "MinerU+Qwen-for-code hybrid default" and
  the circuit-breaker "no Docling fallback" decisions, with rationale, the two-corpus
  evidence, and the literature basis).
- `docs/PROJECT_STATUS.md`: fix the self-contradiction at lines 415-420 vs 5-27 (F1),
  the stale test-command count at 425 (F6), and soften the "two-axis acceptance /
  fidelity floor wired" claim at 96-97 (F5).
- `AGENTS.md`: add `ARCHITECTURE_V3.1_CHARTER.md` to the Layer-0 list, drop the
  phantom `SRS` (F4); mark Principle F `shadow_ocr` as legacy / point V3 recovery at
  the ladder (F8).
- `CLAUDE.md` + `docs/README.md`: clarify the 0.5 "(canonical target)" tag (F9).

---

## 9. Reconciliation with the 2026-06-09 governance audit

- **F1 (resilience contradiction):** the audit's default fix was "document the
  fallback as-is." This plan OVERRIDES that: the fallback is a corner-cut, so the fix
  is to correct the DESIGN (retry-first + quality-risk arbitration), then write the
  spec to the corrected design - not to the current code. F1's doc edits move to
  Phase 5.
- **F3 (charter missing the ladder):** still add it, but described as the demoted
  last-resort net under the new design, not as the reliability story.
- **F5 (OmniDocBench overclaim):** this plan is the vehicle that makes the fidelity
  axis real; until Phase 1 runs clean, the spec says "fidelity gate PROPOSED,
  benchmark-gated," not "wired."
- **F2/F4/F6/F7/F8/F9:** thesis-independent doc-hygiene; apply in Phase 5 (or sooner
  as a standalone hygiene commit if the user prefers - they are factual corrections
  that should not wait on the architecture work).

USER-DECISION-REQUIRED still open: whether to invest in fixing the MinerU server
(Phase 0) versus standardizing on an alternative engine that serves reliably today
(e.g. PaddleOCR-VL or Docling). Phase 0A (operational profiling) and Phase 1 (the
bake-off) both inform it, but the serving-reliability cost is a separate operational
call.

---

## 10. Issue / risk register

- I1: bake-off still blocked if MinerU serving is not fixed AND no alternative
  engine is wired -> Phase 0 has a fallback: select among the engines that DO serve
  reliably (docling_fast, PaddleOCR-VL, Qwen) rather than waiting on MinerU.
- I2 (STRENGTHENED rev. 2): OmniDocBench is English+Chinese only and is NOT our
  corpus (Dutch manuals, German technical PDFs, automotive diagnostics, wiring
  diagrams). Selecting on it alone would optimize the benchmark and damage real-world
  performance. MITIGATION (now mandatory, not optional): every selection and
  lane-cut decision (Phase 1, Phase 2) requires the INTERNAL crucible retrieval-value
  axis ALONGSIDE OmniDocBench; the benchmark never decides alone.
- I3: quality-risk arbitration has no ground truth at conversion time -> it is
  explicitly a PROXY/risk-flag system (Section 5.3), calibrated against OmniDocBench
  labeled ED so we know the correlation, never sold as measured fidelity. Do not
  block on a perfect online scorer.
- I4: thesis change is large -> phased, each phase independently shippable + gated;
  the safety-net ladder + retry-first (Phase 0.5) ship early so reliability never
  regresses below today regardless of the thesis outcome.
- I5: edit-distance/TEDS penalize equivalent formatting (literature caveat) -> the
  offline fidelity gate is regression-vs-baseline, never absolute exact-match.
- I6 (rev. 2/3): a DPI/resolution cut to relieve the M5 (Phase 0A) could silently
  lower extraction quality -> every render-setting in the sweep is scored on
  OmniDocBench labeled ED AND a representative internal-corpus subset (rev. 3), so the
  latency/thermal win is never bought with an unmeasured fidelity loss - especially on
  the small-label content (wiring diagrams, automotive) OmniDocBench does not contain.

---

## 11. Acceptance (of this workstream)

- The default engine is SELECTED by recorded two-corpus evidence (OmniDocBench
  fidelity + internal retrieval-value), not asserted; the candidate pipeline-primary
  thesis is explicitly confirmed or refuted.
- Retry-first (Phase 0.5) + quality-risk arbitration (Phase 3) shipped with tests;
  the ladder is the last-resort net, provenance-stamped, never a silent equivalent
  substitution.
- A recorded VLM operational profile (Phase 0A) including throughput/queue-depth, with
  a chosen render setting whose quality impact is measured on BOTH OmniDocBench and the
  internal corpus.
- The fidelity outcome gate is live (advisory, then promoted per Section 6).
- The Section 8 Layer-0 edits applied; governance audit F1/F3/F5 closed;
  `tests/test_repo_integrity.py` + `scripts/smoke_production.sh` green.
- No locked invariant violated (extraction boundary, zero-Docling batch_processor,
  AST firewall, ElementType=3/Modality=5, QA-CHECK-01 0.10, bbox [0,1000]).
