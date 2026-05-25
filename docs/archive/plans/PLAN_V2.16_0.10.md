# Plan: v2.16 — Convergence Cycle (Final v2.X Tag)

**Status:** **Draft v0.10** (2026-05-25). Supersedes Draft v0.9
(2026-05-25). External audit Round 8 returned 9 findings against
v0.6 (auditor reviewed an outdated draft — v0.6, not v0.9 — and
re-surfaced 7 findings already incorporated in Rounds 4-7 fixes
the auditor didn't see, plus 2 substantive new findings) —
**0 HIGH accepted as structural + 1 PARTIAL MED + 8
REJECTED/SUPERSEDED**. Per the disposition table in Appendix A,
the partial accept (F5: programmatic near-boundary flag added to
Phase 0 step 3b as Probe C) is the only v0.10 edit; the 8
rejected/superseded findings (F1 already fixed in Round 5 v0.7;
F3 already fixed in Round 4 v0.6; F8 already fixed in Round 7
v0.9; F2/F4/F7 re-litigate accepted dispositions from Rounds
3-4; F6 fix would create soft state; F9 superseded by Round 5
F5's [[no-human-verification-loops]] disposition) generated no
edits. **Round 8 is the second consecutive 0-HIGH round —
v2.15 §9 stopping rule FIRES; cycle is ready to execute Phase
0.** Predecessor drafts: v0.9 Round 7 (0 HIGH + 3 MED + 2 LOW);
v0.8 Round 6 (4 HIGH iter-fallout + 1 MED + 2 LOW); v0.7 Round 5
(2 HIGH iter-induced + 2 MED + 1 partial-MED + 3 LOW); v0.6
Round 4 Kimi (2 HIGH + 3 MED + 1 LOW + 1 partial + 1 moot); v0.5
Round 3 (5 HIGH + 3 MED + 1 partial); v0.4 Round 2 (3b/3d
structural + 5 MED/LOW); v0.3 Round 1 (4 accepted); v0.2 Round 0
self-audit (14 findings).

**Predecessor:** [`docs/PLAN_V2.15.md`](PLAN_V2.15.md) — CLOSED 2026-05-24
under Option F. v2.15 Phase 1 (HyDE bridging) CLOSED AS DEAD LEVER
post-tag. v2.15 Phase 3 telemetry suite SHIPPED. v2.15.0 tag PUSHED.

**Successor:** **v2.17 exists only as a safety valve** for unexpected
issues arising during convergence (per user disposition: "I'm
allowing for a v2.17 if unexpected issues arise"). Tight scope: see
§5 below.

**Owner:** ingestion + retrieval + LLM-integration pipeline.

### Round-0 self-audit changes folded into v0.2

| # | Severity | Finding | Fix in v0.2 |
|---|---|---|---|
| 1 | HIGH | Phases 1, 3, 4, 5 had empty Method sections — placeholder said "see prior Draft v0.1 §2 Phase X" but there IS no prior v0.1 (this IS v0.1) | All four Method sections now inlined with concrete implementation steps. Stale "see prior" markers removed. |
| 2 | HIGH | Phase 7 (D2 image re-read) trigger had no firing path — Phase 1 validation queries are for CarOK + Fluent_Python (neither image-heavy) so "≥3 image-content gaps" is structurally impossible | Item #8 reframed to **EXPECTED KILL** unless Phase 1 explicitly authors validation queries for an image-heavy class (user choice in §8a Q3 NEW). Default is KILL; Phase 7 stays in plan as opt-in conditional only if user adds the scope. |
| 3 | HIGH | `data/raw/` integration unspecified — soak harness expects `output/<doc_name>/ingestion.jsonl` per `CANONICAL_*` list; no spec for where Phase 0 output lands or how docs get added to the list | Phase 0 method specifies: per-doc invocation `mmrag-v2 process data/raw/<doc>.pdf --output-dir output/<doc_basename>` to land at canonical location; `CANONICAL_*` list update is part of acceptance bar; post-ingestion source PDFs can stay in `data/raw/` (location of source PDF isn't load-bearing — only the per-doc ingestion.jsonl in `output/` matters for downstream consumers). |
| 4 | HIGH | Phase 3 `chunk_index + 1` adjacency fetch assumed monotonic source-flow order; chunk_index ordering semantics aren't guaranteed (modality-interleaved chunks could mean +1 = wrong chunk) | Phase 3 method now uses **`(source_file, page_number, in-page position)` tuple ordering with text/code-modality filter on the "next" lookup**. Implementation includes a 30-min spike at start of Phase 3 to verify chunk_index semantics against the actual production schema; if non-monotonic, fall to the tuple-based ordering. |
| 5 | MED | Item #9 KILL was conditional on Phase 3 succeeding; Phase 3 Risk paragraph noted v2.17 "reopens #9" if Phase 3 fails — contradictory disposition | §2 Item #9 reframed as **CONDITIONAL KILL** (KILLed if Phase 3 passes acceptance bar; reopens to v2.17 safety valve if Phase 3 fails). Convergence-discipline-compliant: v2.17 IS the escape valve. |
| 6 | MED | "Code-dense doc" and "form-class doc" not operationalized — referenced in Phase 0/3/4 dependency rules but never defined programmatically | Phase 0 method adds explicit classification rules: **code-dense = ≥30% of chunks CODE modality OR profile=technical_manual AND has_code_evidence=True**; **form-class = ≥40% of chunks tables AND ≥3 unique table-template patterns**; **minority-language = mean non_ascii_ratio > 0.03 across sampled chunks OR intent classifier returns minority_language on ≥30% of sampled chunk-text**. |
| 7 | MED | §8 question ordering unclear — §8a Q1 (DoD threshold) gates Phase 1 (validation query authoring) but no explicit sequencing | Phase 1 method gains: **"BLOCKING PREREQUISITE: §8a Q1 must be answered before Phase 1 starts."** §8a Q2 can be answered at Phase N close-out. Pre-execution checklist explicit on this. |
| 8 | MED | Soft-state "watch-item for v2.17 if a form-class doc surfaces" language sneaked back in via Phase 4 generality acceptance | Removed. Phase 4 generality acceptance now: "if Phase 0 has <2 form-class docs, CarOK-only validation IS the final form of the test — no future-cycle watch." Either commit or accept; no watch-items. |
| 9 | MED | Phase 1 cost estimate was 0.5d impl + 0.5d/class — total likely 2-3 days if Phase 0 surfaces 3+ MED classes; §7 budget didn't reflect this | Phase 1 cost line + §7 budget revised: implementation 0.5d + class authoring N × 0.5d where N is unknown until Phase 0 inventory. Worst case 4 days total Phase 1 (impl + 7 classes). |
| 10 | LOW | Pre-execution checklist Q1-Q5 numbering didn't match §8a/8b restructure | Updated to reference §8a (real questions) + §8b (proposed defaults). |
| 11 | LOW | §1 DoD item 1 self-contradictory: "pipeline.py unmodified since v2.15.0 except for the v2.16 Phase 3 ... and Phase 5" | Reworded: "pipeline.py modified only for Phase 3 + Phase 5 (no other changes)." |
| 12 | LOW | §7 wall-clock said "8-15 working days" + "hard cap 12 working days" — math doesn't match | Range tightened to 8-12 days; cap stays 12. |
| 13 | LOW | Item #15 (Magazine rendered-region-crop) KILL rationale leaned on "5 cycles with zero query-evidence" but telemetry was just shipped — no actual evidence cycle | Rationale reworded: "no demand signal surfaced through validation queries or strict-gate failures; magazine content currently meets quality bars per existing soak data." Honest. |
| 14 | LOW | Phase N "2nd-dead-lever" terminology applied symmetrically to Phases 6 + 7, but Phase 7 has no prior dead-lever precedent | Phase N close-out language: Phase 6 outcome is "SHIPPED OR CLOSED-as-2nd-dead-lever / KILLed"; Phase 7 outcome is "SHIPPED OR KILLed" (no dead-lever framing for Phase 7). |

---

## 1. Project Definition of Done

MM-Converter-V2 is **feature-complete** when ALL of the following hold:

1. **Production retrieval is stable** — hybrid (omlx Qwen3-Embedding-8B-mxfp8 + BM25 + RRF + ModernBERT rerank) with the opt-in HyDE knob retained as v2.15 dead-lever infra; pipeline.py modified only for Phase 3 (`partial_code` adjacency fetch) and **Phase 5 (dynamic top-k IF the Phase 1 pre-flight gate fires — default-on if shipped; not in production code at all if KILL'd)** — no other changes. (Phase 5 reference clarified in v0.8 per External Audit Round 6 Finding 1 — previously said "opt-in dynamic-top-k knob," which contradicted v0.6's binary disposition.)
2. **Strict-gate corpus state** — 34/34 PASS minimum (current baseline since v2.10), extended to whatever count the v2.16 Phase 0 corpus expansion produces. No regression.
3. **Personal validation queries** — every documented-limitation class has a curated 10-20 query fixture; per-class pass rate **satisfies the threshold set by §8a Q1** on the v2.16-shipped retrieval stack. Q1's three legitimate answers are: keep at 75% (loose, current default — some failures will be known-and-accepted), tighten to 85% (single uniform bar), or per-class (HIGH 85% / MED 75%). **Default of 75% applies only if Q1 is not actively answered before Phase 1 starts**; Q1 is BLOCKING per the pre-execution checklist, so the production path is for the user to pick before validation queries are authored. (Parameterized in v0.9 per External Audit Round 7 Finding 5 — prior hardcoded "≥75%" was wrong at ship-gate time if Q1 picked tighten-to-85% or per-class. Cross-reference corrected in v0.7 per External Audit Round 5 Finding 1 — previously pointed at §8 Q2, an §8a re-ordering relic.)
4. **omlx -12pp deficit either fixed or explicitly accepted** — Phase 2 diagnostic verdict drives a binary outcome: a Phase escalation (folded into v2.16) OR a permanent DECISIONS.md entry recording "this is the embedder's limit on these classes; documented limitation."
5. **Every documented-limitation class has a permanent disposition** — SHIP'd a fix (Phase 3/4 close the file), KILL'd (closed in DECISIONS.md as out-of-scope), or marked permanent-limitation-with-rationale. No class in a "we'll see" state.
6. **Zero carry-forwards in soft states** — `PROJECT_STATUS.md` Other Carry-Forwards list either empty or holds only items with explicit v3.0-class trigger conditions (which means: not part of v2.X work, ever).
7. **README declares v2.16.0 as feature-complete** for the project's stated use case (solo-dev personal PDF→JSONL multimodal-RAG pipeline against a curated multilingual corpus including technical manuals, magazines, scanned books, forms, code-dense books).
8. **Post-v2.16.0**: only bug fixes (v2.16.x patches). No new features. New features = re-charter as a v3.0 project. **v2.16.x vs v3.0 boundary (REVISED in v0.7 per External Audit Round 5 Finding 4 — the "bug fix" definition was previously undefined, creating ambiguity with Phase 4's "tunable post-hoc" config knob)**: a config-knob or threshold change is a **v2.16.x patch** ONLY if it fixes a demonstrable regression from v2.16.0 behavior on the v2.16.0 corpus (i.e., something that worked at v2.16.0 ship now doesn't). Changes motivated by new documents, new use cases, or "the threshold could be better tuned for X" are **v3.0 re-charter**, not v2.16.x. This rule applies uniformly to Phase 4's IoU threshold, Phase 5's truncation parameters (if shipped), and every other tunable knob. Item #16's "thresholds FROZEN at v2.16.0 ship values" applies the same rule by another route. The full v3.0 re-charter conditions are in §10.

### Why this matters

The v2.10 → v2.15 cadence shipped real wins but accumulated soft state at every cycle. v2.15 closed with 14 active+deferred items (see Inventory below) — most "gated on future evidence." Each individual gating was rational; the cumulative pattern is "the product is never done."

Convergence cycle discipline: every open item gets a binary disposition in this plan. Nothing carries forward as "we'll think about it later." Items genuinely worth doing in v3.0 are explicitly marked OUT-OF-SCOPE for v2.X with a re-chartering note; everything else SHIPs in v2.16 or is KILLed.

---

## 2. Disposition matrix — every open item gets a label

Inventory of all open items across the v2.11 → v2.15 carry-forward
history, plus this cycle's natural extensions:

| # | Item | Source | Disposition |
|---|---|---|---|
| 1 | Corpus expansion (Phase 0 enablement) — 7 docs added to `data/raw/` by user 2026-05-24 PM; class identity withheld to avoid analysis bias | v2.16 NEW | **SHIP** |
| 2 | Decision-mechanism overlay (personal_importance + curated validation queries) | v2.16 NEW | **SHIP** |
| 3 | omlx -12pp deficit diagnostic | v2.13 P1 evidence; v2.15 P1 explicitly unaddressed | **SHIP** (Phase 2) |
| 4 | `partial_code`-aware retrieval (adjacency fetch) | v2.14 P6 obs landed; consumer never built | **SHIP** (Phase 3) |
| 5 | VLM-Table Dedup IoU>85% (A1) | v2.14 P1 attempt rolled back; this is the missing piece | **SHIP** (Phase 4) |
| 6 | Dynamic top-k from rerank logit drop-off (C2) | Gemini v2.16 proposal | **CONDITIONAL SHIP** (Phase 5) — REVISED in v0.6 per External Audit Round 4 Finding 3. Previously unconditional SHIP; re-disposed because the same "no demand signal in 5 cycles" pattern that KILLs Items #10 and #15 applies here (no validation query in v2.11→v2.15 surfaced rerank-flatness or top-5 context bloat as a failure mode). Binary outcome: gate the CODE on a Phase 1 baseline pre-flight check — apply the proposed dynamic-top-k logic to Phase 1's baseline rerank outputs without writing the production code; if ≥20% of validation queries would be truncated (meaningful context reduction signal) AND PASS-retention under truncation is ≥0.97, implement Phase 5 + SHIP default-on. Else **KILL** permanently (no opt-in middle ground — opt-in dead code is the failure mode for a feature-frozen product). |
| 7 | C1 Query rewriting (rewrite query rather than synthesize answer) | Gemini v2.16 proposal | **CONDITIONAL SHIP** — REVISED in v0.8 per External Audit Round 6 Finding 2 (v0.7 added a Phase 2 analytical pre-flight for Phase 6 but didn't propagate the requirement into this row). Trigger is now **compound**: ship as Phase 6 in v2.16 ONLY if Phase 2 diagnostic verdict is H2 (OOV/vocabulary) OR H3 (cross-lingual) AND class-level **AND** the Phase 2 Phase-6 analytical pre-flight shows ≥3pp R@1 lift on a 5-10 query rewrite sample from the deficit class. **KILL** permanently if either leg fails (verdict not H2/H3+class-level, OR pre-flight <3pp lift). Symmetric with Phase 5's pre-flight gate. |
| 8 | D2 Retrieval-time image re-read ("Look at Images Twice") | Gemini v2.16 proposal | **OPT-IN (default KILL)** — REVISED in v0.4 per External Audit Round 2 Finding 6: previously labeled "EXPECTED KILL / CONDITIONAL on §8a Q3," but the trigger is structurally unfireable without explicit user opt-in (Phase 1 validation scope = CarOK + Fluent_Python, neither image-heavy). "Conditional" implies an objective trigger that could plausibly fire; this can only fire via §8a Q3 opt-in. If user answers §8a Q3 = "ADD image-heavy class to Phase 1," Phase 7 activates as a real CONDITIONAL SHIP gated on the new fixture's failure pattern. Without that override, Item #8 KILLs at Phase N. |
| 9 | B1 Docling HybridChunker config hunt | v2.15 P4 deferred under Option F | **CONDITIONAL KILL** — REVISED in v0.7 per External Audit Round 5 Finding 2 (previous trigger said "Fluent_Python ≥7/10" which is only ONE leg of Phase 3's compound acceptance bar). **KILLed only if Phase 3 passes its FULL acceptance bar**: Fluent_Python validation ≥70% pass rate AND — if Phase 0 surfaces ≥2 code-dense docs — each generalization doc also ≥70%. (Bar phrased as a ratio in v0.9 per External Audit Round 7 Finding 2.) **Reopens to v2.17 safety valve if Phase 3 fails on ANY leg** (one of the four enumerated v2.17 triggers per §5). The pre-revision trigger text could have falsely KILL'd Item #9 when Fluent_Python passed but a generalization doc failed — leaving a real code-dense defect with no fix plan. |
| 10 | A2 HTML+summary split (Unstructured pattern) | v2.16 Gemini proposal | **KILL** — REVISED in v0.5 per External Audit Round 3 Finding 9. Previously labeled OUT-OF-SCOPE (v3.0) with rationale "embedder retraining or migration territory"; on re-examination the work is a chunk-emission pattern change in the ingestion path (emit long-form chunk for embedding + short summary chunk for display) — NOT embedder-retraining-class and NOT a vector-store schema change. It could be a 2-5 day v2.X feature. But: zero demand signal across v2.11→v2.15 (no validation queries surface a summary/long-form mismatch; no strict-gate failure attributable to chunk-shape) and the current single-chunk pattern works for documented use cases. Convergence-cycle discipline → KILL with the same logic as Item #15 (no demand signal). If summary-vs-content distinction ever becomes load-bearing, v3.0 re-charter — but the OUT-OF-SCOPE label was overstating the scope. |
| 11 | D1 ColPali / VisRAG visual retrieval | v2.16 Gemini proposal | **OUT-OF-SCOPE (v3.0)** — full vector-DB explosion + cold-start + architecture rewrite. v3.0 project, not a feature. |
| 12 | B2 Code-Rescue Middleware (heuristic stitching) | v2.16 Gemini proposal; v0.9 audit Round-1 explicitly REJECTED | **KILL** — Round-1 finding stands. |
| 13 | UIR refactor (3c) | v2.11 plan; PARKED 5 cycles; v2.15 PARKED-WITH-TRIGGERS (4 trigger conditions, none realistic for personal-corpus use) | **KILL** — the four trigger conditions (3rd engine, cross-engine defect, ≥500 LOC test boilerplate, external integration request) are realistic only in a multi-format or multi-engineer setting. Personal-corpus PDF-only project will never trigger them. Closing prevents 5 more cycles of zero-motion PARKED state. Re-charter as a fresh v3.0 architecture proposal if multi-format ever becomes needed. |
| 14 | VLM swap to alternative model (3a from v2.11) | v2.11 carry-forward | **KILL** — current NuMarkdown-8B-Thinking-mlx-8bits works for the use case; no swap needed. v2.14 P1 force_table_vlm validated VLM produces clean output; the failure was dedup (now Phase 4). |
| 15 | Magazine rendered-region-crop (3e from v2.11) | v2.11 carry-forward | **KILL** — 5 cycles with zero demand-evidence; image-axis perf has been adequate per repeated soak data; no class of queries surfaced the need. Magazine docs already work with current chunker. |
| 16 | Telemetry collection (v2.15 P3) | v2.15 active | **KEEP active** (not a carry-forward; ongoing infrastructure) — keep for objective sanity check + future-deployment readiness. v2.16 Phase 1's personal_importance overlay is the cycle-open decision authority; telemetry stays as a second-class signal. **Maintenance contract (NEW in v0.3 per External Audit Round 1 Finding 4)**: thresholds in `analyze_doc_class_telemetry.py` are FROZEN at v2.16.0 ship values per the convergence-cycle feature-freeze — no post-tag tuning permitted (changing them = re-charter as v3.0). Log retention: telemetry JSONL files in `logs/doc_class_telemetry/` rotate at 30-day age; rotation handled by the existing logrotate-style cleanup invoked by cycle-open checklist §6. If retention or threshold changes ever become necessary, that's a v2.17 trigger only if it surfaces as a Stop-the-Line condition (per §5 trigger #3); otherwise v3.0. |
| 17 | Phase 4-Resilience qwen3-max cloud fallback (v2.14) | v2.14 shipped | **KEEP active** — unchanged production infrastructure. |
| 18 | HyDE bridging | v2.15 P1 closed dead-lever | **CLOSED (no further action)** — already closed in DECISIONS.md 2026-05-24. Listed here for inventory completeness. **Why HyDE's switched-off opt-in knob is NOT a contradiction of Item #6's "opt-in dead code is the failure mode" principle (NEW in v0.9 per External Audit Round 7 Finding 3)**: HyDE's feature-flag surface is a single boolean at the retrieval entry point with zero entanglement in the live retrieval path — when `False`, the HyDE step is skipped entirely and `retrieve_hybrid_reranked` runs unchanged. Phase 5's would-be opt-in is structurally different: dynamic top-k would interleave with `partial_code` adjacency fetch inside `retrieve_hybrid_reranked` (per §6 serial-order rule), and an opt-in surface there means two live code paths that interact non-trivially. The convergence-cycle "no opt-in dead code" principle is calibrated for the entanglement-cost case, not for top-level toggles. v2.15's DEAD LEVER closure in DECISIONS.md is the authoritative do-not-enable signal for future maintainers; the knob remains as inert opt-in infra (one boolean, no maintenance cost) rather than being ripped out, because ripping it out would be a v2.16.x scope expansion the §10.1 boundary forbids (HyDE's presence is not a v2.16.0-corpus regression). |
| 19 | Phase 3 rollback collection drop | v2.14 P3 executed | **CLOSED (no further action)** — executed 2026-05-23 PM. Listed for inventory. |
| 20 | Phase 6 calibration freshness check | v2.15 ongoing | **KEEP active** — T-72h pre-tag checkpoint + cycle-open freshness review per CYCLE_OPEN_CHECKLIST.md. Stays through v2.16. |
| 21 | 3b Remote CodeFormulaV2 inference | v2.11 Phase 3b (defer-with-named-workaround); silently dropped from carry-forward list across v2.12→v2.15 (NEW in v0.4 per External Audit Round 2 Finding 1) | **KILL** — local CPU CodeFormulaV2 lane in Docling 2.86.0 (~27 sec/page, per CLAUDE.md) is sufficient for one-off batch reconversion in this project's solo-dev workflow. Original trigger ("Docling 2.87+ exposes `RemoteCodeFormulaOptions`") never fired in 5 cycles. Remote inference is a v3.0 optimization, not a v2.X requirement. |
| 22 | 3d HybridChunker per-item token guard | v2.11 Phase 3d (build opt-in `--strict-hybrid-guard` flag); v2.12 marked "subsumed by Phase 4 if shipped"; v2.12 Phase 4 never triggered; flag never built (zero hits in src/scripts/tests); silently dropped from v2.13→v2.15 (NEW in v0.4 per External Audit Round 2 Finding 1) | **KILL** — v2.10 element-by-element fallback already handles pathological-input chunking; the per-item guard was opt-in/default-off in v2.11 design (no behavior change baseline) and never built (4 cycles, zero demand signal). Quality-optimization on an edge case, not load-bearing for feature-complete. |

### Disposition summary

| Disposition | Count |
|---|---:|
| **SHIP this cycle** | 5 unconditional (Phases 0-4) + 2 conditional (Phase 5, Phase 6) + 1 opt-in (Phase 7) = 5-8 phases depending on which conditional/opt-in gates fire |
| **KILL** (closed permanently in DECISIONS.md) | 9 base items (v0.4: #21, #22; v0.5: #10 re-disposed from OUT-OF-SCOPE) + Item #6 if Phase 5 pre-flight fails + Item #7 if Phase 2 compound trigger fails + Item #8 unless user opts in via §8a Q3 → up to **12 KILL items total** in the worst case. (Formula updated in v0.8 per External Audit Round 6 Finding 4 — the prior "may grow to 10" undercounted the Item #7/#8 KILL paths.) |
| **OUT-OF-SCOPE (v3.0)** | 1 item (#11 ColPali only) |
| **KEEP active** (ongoing infrastructure) | 3 items |
| **Already closed** | 2 items |

After v2.16 ships, the Other Carry-Forwards list in
`PROJECT_STATUS.md` is **empty**. No PARKED, no deferred, no
gated-on-future-evidence. Either the work is done, or it's
explicitly out-of-scope for this product.

---

## 3. SHIP phases

### Phase 0 — Corpus expansion (gates Phases 2 + 4)

**Goal:** ingest the 7 documents the user added to `data/raw/`
during plan revision (2026-05-24 PM, post-Draft-v0.1). User-explicit
methodological discipline: documents added intentionally without
disclosing class identity so neither the dev nor the planning
process biases analysis toward specific document types. Class
composition is whatever the existing `ProfileClassifier` + pipeline
diagnostic output reveals — not a planning assumption.

**Method:**

1. **Pipeline-driven per-doc ingestion (REVISED in v0.2 per Round-0
   Finding 3 — data/raw/ integration spec)**:
   For each `data/raw/<doc>.pdf`, run:
   ```
   mmrag-v2 process data/raw/<doc>.pdf \
       --output-dir output/<doc_basename> \
       --batch-size 10
   ```
   where `<doc_basename>` is the filename without `.pdf` extension.
   This lands each doc's `ingestion.jsonl` at the canonical
   location (`output/<doc_basename>/ingestion.jsonl`) that the
   soak harness + downstream consumers expect via
   `DOCS_ROOT / doc_name / "ingestion.jsonl"`. Source PDFs stay
   in `data/raw/` (location of source isn't load-bearing for
   downstream consumers — only the ingestion output is).

   No `--profile-override` — pipeline's `ProfileClassifier`
   auto-routes per doc based on PDF content. This is the
   canonical unbiased classification.

2. **Threshold pre-validation against known corpus (NEW in v0.4
   per External Audit Round 2 Finding 5)**: BEFORE classifying the
   unknown 7-doc corpus, run the three classification rules
   (below) against the existing 34-doc canonical corpus where the
   class is known. Sanity targets: known code-dense docs
   (`Fluent_Python`, `Python_Cookbook`, `Python_Distilled`) must
   classify as code-dense; known non-code docs (`HARRY`,
   `CarOK_voorraadtelling`) must NOT classify as code-dense; the
   known form-class doc (`CarOK_voorraadtelling`) must classify as
   form-class. If any of these known-result expectations fail,
   pause and recalibrate the threshold before running Phase 0 on
   the unknown 7-doc corpus. Cost: ~30 min, $0. Output recorded as
   a one-line confirmation in the inventory report (or threshold
   adjustment + rationale if recalibration was needed). This
   preserves the bias-discipline (no source-PDF inspection of the
   7 unknown docs) while removing the "thresholds are untested
   assumptions" risk.

   **Abort condition (NEW in v0.7 per External Audit Round 5
   Finding 7)**: if no threshold setting can simultaneously make
   the known code-dense docs classify as code-dense AND the
   known non-code docs not classify as code-dense (i.e.,
   recalibration cannot satisfy the sanity targets at all), the
   classification rules are fundamentally broken — likely a
   dependency change between v2.15.0 and Phase 0 (e.g., Docling
   patch altering CODE modality emission). Phase 0 aborts. This
   is an external dependency break per §5 trigger #2 (v2.17
   stop-the-line condition); investigation is required before
   the cycle can proceed.

3. **Categorization rules (REVISED in v0.2 per Round-0 Finding 6 —
   operationalization)**: after all docs ingest, read the
   ingestion.jsonl files (NOT the source PDFs) and compute
   per-doc class assignments via explicit programmatic rules:

   - **Code-dense**: `code_chunks / total_text_chunks ≥ 0.30`
     OR (`profile == "technical_manual"` AND
     `diagnostic.has_code_evidence == True`)
   - **Form-class**: `table_chunks / total_chunks ≥ 0.40`
     AND `unique_table_template_patterns ≥ 3`
     (template pattern = (column_count, row_count) tuple; ≥3
     unique tuples = real form-class doc with multi-shape tables)
   - **Minority-language**: `mean_non_ascii_ratio > 0.03` across
     sampled text chunks (≥10 sampled per doc)
     OR `intent_classifier_minority_language_hit_rate ≥ 0.30`
     across sampled chunks (using the existing
     `src/mmrag_v2/retrieval/intent.py` classifier)
   - **Other**: anything that doesn't trigger code-dense /
     form-class / minority-language → class = `general` (no
     special Phase 2/3/4 handling)

   Classes are not mutually exclusive (a doc could be both
   code-dense AND minority-language; both flags get recorded).

3b. **Programmatic secondary probes for suspected misclassifications
    (NEW in v0.6 per External Audit Round 4 Finding 6; extended in
    v0.7 per External Audit Round 5 Finding 6 — second probe for
    minority-language misclassification with OCR-stripped
    diacritics).** Two patterns, both bias-discipline-compatible
    (operate on pipeline outputs, never on source PDFs):

    **Probe A — form-class misclassification** (form-class doc
    whose tables are image-based may be silently classified as
    general because OCR failed to produce table chunks):
    ```
    if profile in {"scanned", "scanned_degraded"}
       AND image_chunks > 0
       AND table_chunks == 0:
        run: mmrag-v2 process <doc> --force-table-vlm
                     --pages 1-3 --output-dir output/_probe_<basename>
        if probe produces table chunks:
            override classification → form-class
            record rationale in inventory report
    ```

    **Probe B — minority-language misclassification with
    OCR-stripped diacritics** (e.g., scanned Turkish manual where
    OCR strips non-ASCII glyphs, producing low non_ascii_ratio +
    intent classifier under-fires):
    ```
    if intent_classifier fires on ≥1 chunk
       BUT total hit-rate < 0.30
       AND mean_non_ascii_ratio < 0.03:
        flag in inventory report: "borderline minority-language —
        N/total chunks triggered intent classifier;
        non_ascii_ratio below 0.03 threshold may reflect
        OCR-stripped diacritics rather than absence of
        minority-language content"
    ```
    Probe B is **signal-only** (no automatic reclassification) —
    the pattern is harder to disambiguate from "truly not
    minority-language" without source-PDF inspection. Flag goes
    into the inventory report.

    **Phase 2 H3 sampling rule for Probe B docs (NEW in v0.8 per
    External Audit Round 6 Finding 5 — v0.7 said Phase 2 "can use"
    the flag without specifying how):**
    - Probe B docs ARE included in Phase 2's H3 (cross-lingual)
      sample with their normal soak protocol. They appear in
      the H3 measurement as `borderline_minority_language`
      candidates and contribute to the H3 evidence base.
    - Probe B docs do NOT count toward the ≥3-doc class-level
      threshold (which determines whether the Phase 2 verdict is
      "class-level" vs "doc-specific"). The threshold counts
      only docs whose classification is confidently minority-
      language (mean_non_ascii_ratio > 0.03 OR intent_classifier
      hit-rate ≥ 0.30). This avoids letting borderline-flagged
      docs falsely promote a doc-specific finding to a
      "class-level" verdict that would trigger Phase 6.

    **Probe C — near-boundary classification flag (NEW in v0.10
    per External Audit Round 8 Finding 5 — programmatic
    reformulation of Round 8's "manual arbitration" recommendation,
    which would have violated [[no-human-verification-loops]]):**
    a doc whose class-determining metric falls within ±5% of any
    boundary threshold is flagged in the inventory report. Same
    bias-discipline principle as Probe B (signal-only, no
    auto-reclassification; user sees the flag during the existing
    Phase 0 acceptance review of the inventory report):
    ```
    for each doc:
        if 0.25 <= code_chunks_ratio < 0.30: flag NEAR_BOUNDARY_CODE_DENSE
        if 0.35 <= table_chunks_ratio < 0.40: flag NEAR_BOUNDARY_FORM_CLASS
        if 0.025 <= mean_non_ascii_ratio <= 0.035: flag NEAR_BOUNDARY_MINORITY_LANGUAGE
    ```
    Rationale: thresholds at 30% / 40% / 0.03 are calibrated to
    the 5 known examples; an out-of-distribution doc landing at
    28% code or 38% tables could silently bypass the corresponding
    Phase 3 / Phase 4 lane. Flag does NOT auto-promote; user
    decides at Phase 0 acceptance whether to (a) accept the
    classification as-is, (b) re-extract with a profile override,
    or (c) recalibrate the threshold (which then triggers the
    step 2 pre-validation re-run against the known 34-doc corpus
    to verify the new threshold doesn't break known cases).

    Cost: ~5-10 min per probed doc for Probe A; ~0 min for
    Probe B + Probe C (both computed from existing inventory data).
    Probe A only fires when the classification pattern looks
    suspicious; Probe B + Probe C compute on every doc but are
    cheap.

4. **Generate Phase 0 inventory report** at
   `docs/CORPUS_EXPANSION_2026-05-24_v2.16_p0.md` listing per-doc:
   - File basename
   - Auto-routed profile (from `ProfileClassifier` output)
   - Chunk count (text / table / image / code breakdown)
   - Computed class flags (code-dense / form-class /
     minority-language / general)
   - Class-determining metrics (the actual numbers that triggered
     or didn't trigger each rule)
   - Probe flags (Probe A reclassifications; Probe B
     borderline-minority-language signal; **Probe C
     near-boundary flags** — NEW in v0.10 per External Audit
     Round 8 Finding 5)
   - Any extraction warnings (encoding corruption, low-confidence
     OCR, etc.)

5. **Append to production indexes** (REVISED in v0.3 per External
   Audit Round 1 Finding 3 — Qdrant snapshot before mutation):
   - **PRE-MUTATION SNAPSHOT**: capture a Qdrant snapshot of
     `mmrag_v2_8__qwen3_local` via `qdrant snapshot create` (or
     equivalent collection-level export) BEFORE appending new
     chunks. Stash snapshot ID + timestamp in the Phase 0
     commit message. Rationale: production indexes live outside
     git; if Phase 0 ingestion misbehaves, snapshot is the only
     non-destructive revert path. (Phase 4 re-extraction of
     CarOK is git-revertable via the ingestion.jsonl rewrite +
     re-append, so no snapshot needed there.)
   - Append new chunks to `mmrag_v2_8__qwen3_local` (dense) via
     existing ingest script
   - Rebuild `mmrag_v2_8__bm25_sparse` (BM25 needs full rebuild
     on corpus change)
   - **Update ALL CANONICAL_34 consumers (REVISED in v0.5 per
     External Audit Round 3 Finding 3 — previously enumerated
     only `synthetic_soak.py`; verified consumers include 2
     independent definitions + 2 import-consumers + 1 test
     with hard-coded length/first-element assertions)**:
     1. `scripts/synthetic_soak.py` (line 124): append each
        `<doc_basename>` to the local `CANONICAL_34` list.
     2. `scripts/rebuild_mmrag_v2_8_for_rc1.py` (line 61):
        append the SAME entries here — this is the canonical
        source that `build_bm25_index.py` and
        `ingest_bm25_sparse.py` import via `_rebuild_mod.CANONICAL_34`.
     3. `tests/test_rebuild_resume.py` (lines 73-81): update
        the hard-coded `len(mod.CANONICAL_34) == 34` assertion
        to the new count AND the rename-to-CANONICAL_DOCS if
        applied (the `[0] == "HarryPotter..."` assertion is
        ordering-stable since new docs append, not prepend).
     4. Rename `CANONICAL_34` → `CANONICAL_DOCS` in ALL above
        sites simultaneously (REVISED in v0.6 per External Audit
        Round 4 Finding 4 — previously conditional on "if count
        is now non-round" which is non-monotonic naming churn).
        The rename is unconditional: post-expansion count is by
        definition different from 34, and the constant name
        should describe its semantic role (the canonical
        docs list), not its cardinality. Partial rename across
        sites would break the `_rebuild_mod.CANONICAL_34` imports
        in BM25 scripts — all five sites rename together.
     5. **Anti-drift bridge test** (NEW): add
        `tests/test_canonical_docs_consistency.py` asserting
        `set(synthetic_soak.CANONICAL_*) == set(rebuild_mod.CANONICAL_*)`
        — prevents future Phase-0-style updates from re-introducing
        drift between the two definitions. ~10 lines of test code.

6. **Class composition drives Phase 2/3/4 scope** (per
   per-phase dependency rules):
   - ≥3 minority-language docs: Phase 2 class-level diagnostic
     test runs at scale (across original ATZ + new docs)
   - <3 such docs: Phase 2 class-level test verdict labeled
     "inconclusive on class-level vs doc-specific" per §3 Phase 2
   - ≥2 form-class docs: Phase 4 generality validates multi-doc
   - <2 such docs: Phase 4 CarOK-only IS the final form of the
     test (no future-cycle watch per Round-0 Finding 8)
   - ≥2 code-dense docs: Phase 3 generality multi-doc
   - <2 such docs: Phase 3 Fluent_Python-only IS the final form
     of the test

**Acceptance:**
- All 7 docs ingest cleanly (no PDF-extraction failures).
  **(REVISED in v0.5 per External Audit Round 3 Finding 6 —
  "planning continues with docs that DID ingest" was soft state.)**
  Any extraction failure is a **Phase 0 FAIL pending explicit user
  resolution**, recorded in the inventory report with one of two
  binary outcomes: (a) **drop from v2.16 scope** — user signs off
  on a one-line rationale + the doc is moved out of `data/raw/`
  + Phase 0 re-runs with the smaller set; or (b) **fix extraction
  and re-ingest** — root cause is addressed (CLI flag tweak, OCR
  routing, etc.) and Phase 0 re-runs. Silent partial ingestion is
  forbidden; the cycle does not advance to Phase 1 with an
  unresolved Phase 0 failure.
- BM25 sparse rebuild completes
- v2.14 retrieval fingerprint 20/20 PASS unchanged (new docs are
  additive; existing 20-query fingerprint queries don't shift)
- Inventory report shipped, listing per-doc profile +
  characteristics from pipeline output only

**Cost:** ~0.5-1 working day wall-clock (most time is pipeline
extraction at ~5-15 min/doc on Apple Silicon); $0 cloud spend.

**Methodological note:** the dev (and this plan) deliberately
do NOT examine `data/raw/` contents directly before pipeline
processing. The pipeline's classification IS the unbiased reading
of the documents. Any post-hoc adjustment to per-doc class
assignment requires recording the rationale in the inventory
report.

### Phase 1 — Decision-mechanism overlay

**Goal:** replace v2.15's telemetry-as-decision-mechanism with a
curated personal-importance overlay (telemetry stays as second-class
signal). Provides the validation-query mechanism that Phases 3 + 4
depend on for acceptance measurement.

**BLOCKING PREREQUISITE (NEW in v0.2 per Round-0 Finding 7):** §8a
Q1 (DoD validation pass-rate threshold) must be answered before
Phase 1 starts. Validation queries are authored against the
threshold; if it changes mid-Phase-1, queries may need re-authoring.

**Method:**

1. **Extend `src/mmrag_v2/retrieval/documented_limitations.py`**:
   - Add `personal_importance: Literal["HIGH", "MED", "LOW"]`
     field per registered class entry
   - Existing CarOK + Fluent_Python entries get `personal_importance: "HIGH"`
   - Default factory for new entries (added in future cycles or
     v2.16 Phase 0 surfaces): `personal_importance: "MED"`

2. **Update `scripts/analyze_doc_class_telemetry.py`** to incorporate
   personal_importance in disposition logic:
   - `HIGH` → forces Option A recommendation regardless of
     telemetry hit-rate (overrides defer / middle-band)
   - `MED` → existing telemetry rules apply (promotion / closure /
     middle-band per DECISIONS.md "v2.15 Documented-Limitation
     Telemetry Threshold (ACTIVE RULE)")
   - `LOW` → reduces NEW_CLASS_GRACE_CYCLES from 2 to 1 for
     auto-closure; still requires 0 issues + no defect-tag
   - Report renders the override transparently: per-class section
     shows both signals + which rule fired

3. **Create `tests/fixtures/personal_validation_queries/` directory**
   with one JSON file per HIGH or MED class. Schema (REVISED in
   v0.5 per External Audit Round 3 Finding 2 — previous schema
   could pass on the wrong chunk from the right doc):
   ```json
   {
     "class": "CarOK_voorraadtelling",
     "personal_importance": "HIGH",
     "target_pass_rate": 0.85,
     "queries": [
       {
         "id": "Q01_part_count_4567",
         "query_text": "What's the total count for part number 4567?",
         "expected": {
           "doc_contains_answer": true,
           "top_5_gold_doc": true,
           "format_constraint": "table_value",
           "gold_chunk_ids": ["CarOK_voorraadtelling__p6_t3_r12"],
           "expected_anchor_regexes": ["\\b4567\\b.*\\b\\d+\\b"]
         },
         "notes": "Page 6, column 3 of inventory table"
       },
       ...
     ]
   }
   ```
   **Answer-correctness fields**: `gold_chunk_ids` (preferred — at
   least one of the listed chunk_ids must appear in retrieved
   top-5 for PASS) AND/OR `expected_anchor_regexes` (the top-1
   chunk content must match at least one regex). At least one of
   the two must be present per query. The previous `top_5_gold_doc`
   check is retained as a coarse sanity filter but is no longer
   sufficient alone — a query that returns the wrong table from
   the right doc previously PASSED; under this schema it FAILs.

   Authoring rules: 10-20 queries per class; manually written by
   the dev (you know what you'd ask); ≥3 queries should test the
   class-specific failure mode (e.g., for CarOK: queries that
   require reading multi-row table values; for Fluent_Python:
   queries that should return runnable code with imports/usage
   context). Authoring step includes harvesting `gold_chunk_ids`
   from a known-good baseline run (Phase 1 step 6 baseline =
   v2.15.0 retrieval state) — chunks that genuinely answer the
   query, not just chunks in the right doc.

4. **Create `scripts/run_personal_validation.py`**:
   - Reads all fixtures from `tests/fixtures/personal_validation_queries/`
   - For each query: runs `retrieve_hybrid_reranked` with production
     defaults (no `--use-hyde` unless that's the validation target)
   - **Per-query PASS rule (REVISED in v0.5 per External Audit
     Round 3 Finding 2)**: ALL of the following must hold —
     (a) `top_5_gold_doc` (doc_id appears in retrieved top-5);
     (b) `format_constraint` (table_value → top-1 modality =
     `table`; runnable_code → top-1 content parses via `ast.parse`);
     (c) **answer-correctness**: at least one `gold_chunk_ids`
     entry appears in retrieved top-5 OR top-1 content matches at
     least one `expected_anchor_regexes` pattern (whichever the
     fixture provides; if both are provided, both must hold).
     This closes the "right doc, wrong chunk" false-PASS hole.
   - Emits `docs/VALIDATION_REPORT_<YYYY-MM-DD>.md` with per-class
     pass rate vs `target_pass_rate`; per-query PASS/FAIL detail
   - Returns nonzero exit code if any HIGH class drops below
     `target_pass_rate` (so CI/cycle-open can gate on it)
   - $0 cloud spend (retrieval only, no LLM judging)
   - ~5-10 min wall-clock per full validation run

5. **Update `docs/CYCLE_OPEN_CHECKLIST.md`**: add new line item §6
   (between current §5 UIR check and §6 cycle_slip review):
   "Review/update `personal_importance` flags on documented-
   limitation classes per current personal workflow needs
   (2-minute review)."

6. **Establish Phase 1 BASELINE validation**: run the new
   `run_personal_validation.py` once against the CURRENT v2.15.0
   retrieval stack BEFORE any Phase 3/4/5 work. Capture this as
   `docs/VALIDATION_REPORT_2026-05-24_v2.15.0_baseline.md`. This
   is the comparison anchor for Phase 3/4/5 acceptance bars (each
   phase's acceptance is delta vs this baseline, not absolute).

**Initial classifications:**
- `CarOK_voorraadtelling`: **HIGH**
- `Fluent_Python`: **HIGH**
- Any documented-limitation classes that emerge from Phase 0
  inventory: default MED unless explicitly flagged by user

**Acceptance:**
- `personal_importance` field on every documented-limitation class
  entry in `documented_limitations.py`
- Analyzer report shows both signals + transparent override logic
- Validation query fixtures exist for all HIGH + MED classes
  (≥10 queries each)
- `run_personal_validation.py` runs end-to-end + emits the dated
  report; baseline validation run captured for v2.15.0 reference
- Baseline report shows current pass rates per class (raw numbers
  for the comparison anchor)
- DECISIONS.md "v2.16 Decision-Mechanism Overlay" entry recorded
- ≥5 new unit tests in `tests/test_personal_validation.py` covering
  the validation runner + fixture-schema validation

**Cost:** 0.5 day implementation + ~0.5 day query authoring per
HIGH/MED class (so N=2 baseline → 1.5 days total; N=5 if Phase 0
surfaces 3 MED classes → 3 days total). $0 cloud spend.

### Phase 2 — omlx -12pp deficit diagnostic spike

**Goal:** answer the v2.13 P1 open question definitively. Drives
Phase 6 conditional ship (Item #7 — C1 query rewriting).

**Method:**
- Hypothesis tests H1 (truncation), H2 (OOV/vocabulary), H3
  (cross-lingual), H4 (chunk length distribution) against the
  original 5 deficit docs (ATZ_Elektronik_German, Python_Cookbook,
  IRJET_Modeling_of_Solar_PV, Hybrid_electric_vehicles,
  Greenhouse_Design)
- **Class-level vs doc-specific test (Phase-0-dependent)**:
  - If Phase 0's inventory report identifies ≥3 docs auto-
    classified as same-class-as-deficit-docs (e.g. German tech,
    code-dense), re-run the omlx-vs-Dashscope shootout on those
    new docs. If the deficit replicates → **class-level**. If it
    doesn't → original 5 docs were **doc-specific** quirks.
  - If Phase 0's inventory has <3 same-class docs, the class-
    level test runs at whatever n is available; verdict
    explicitly notes the statistical-power gap and labels the
    outcome "inconclusive on class-level vs doc-specific."
- Output: `docs/DIAGNOSTIC_<date>_v2.16_p2_omlx_deficit_root_cause.md`
  with verdict + binary outcome for Item #7

**Acceptance:**
- All 4 hypotheses tested with measurements on the original 5
  deficit docs
- Class-level vs doc-specific verdict (or explicit "inconclusive"
  label if Phase 0 didn't provide enough same-class docs)
- Binary outcome for Phase 6 (CONDITIONAL SHIP gate): YES C1
  will help, OR NO permanent KILL. **"Inconclusive class-level"
  routes to KILL** per convergence-cycle discipline (no soft-
  state defer; if evidence isn't there, we don't ship a guess).
- Report shipped in `docs/`

**Cost:** 1 working day; ~$0.50 cloud spend (optional re-embedding
confirmation).

**Phase 6 analytical pre-flight (NEW in v0.7 per External Audit
Round 5 Finding 3 — symmetric with Phase 5's binary pre-flight
gate; before Phase 6 ships 2-3 dev days of code, verify
empirically that the proposed fix would help):** if Phase 2's
verdict is H2 or H3 class-level (the trigger for Phase 6),
**add a sub-deliverable**: take 5-10 Phase 1 validation queries
from the affected deficit class; manually author the
query-rewrite variants (no production code); run them through
the existing `retrieve_hybrid_reranked` and RRF-fuse the candidate
pools analytically; check R@1 / Hit@5 delta on the gold_chunk_ids
fixtures. If ≥3pp R@1 lift on this 5-10-query sample → Phase 6
triggers; build production code. If <3pp lift → **Phase 6 KILLs
before implementation** (DECISIONS.md entry: "Phase 2 H2/H3
verdict positive but Phase 6 pre-flight insufficient lift on
deficit-class validation queries; query rewriting closed as 2nd
dead lever without build cost"). Cost: ~30 min analytical work
inside the Phase 2 day budget; saves 2-3 dev days when the
hypothesis doesn't translate to measurable downstream lift.

**Risk:** the verdict may be "multiple factors interact" — in
which case the binary becomes "NO, C1 won't reliably help" → C1
KILLs. Multi-factor verdicts ≠ ambiguous defer; they ≠ trigger
escalation. Convergence cycle discipline: anything not crisply
positive on the H2/H3+class-level criteria + the new
Phase 6 pre-flight KILLs C1.

### Phase 3 — `partial_code`-aware retrieval (adjacency fetch)

**Goal:** the elegant fix for Fluent_Python (and any future
cross-page code defect): use the existing `partial_code=True`
schema flag from v2.14 P6 to deterministically stitch adjacent
chunks at retrieval time. Sidesteps Docling HybridChunker
configuration work entirely (Item #9 KILL is conditional on this
phase actually working).

**Method:**

1. **Schema verification spike (NEW in v0.2 per Round-0 Finding 4)**:
   30-min initial spike — read `src/mmrag_v2/schema/ingestion_schema.py`
   and a sample of production ingestion.jsonl files to verify
   chunk_index ordering semantics. Possible outcomes:
   - (a) `chunk_index` IS monotonic in source-flow order across all
     modalities → use simple `chunk_index + 1` lookup
   - (b) `chunk_index` is monotonic per-modality (e.g., all text
     chunks first, then tables) → use `(source_file, page_number,
     chunk_index)` tuple sort with text/code-modality filter on
     "next" lookup
   - (c) `chunk_index` reflects only emission order with no
     guaranteed source correspondence → use `(source_file,
     page_number, char_offset_or_position_within_page)` as the
     canonical ordering key
   Spike output documented in the Phase 3 commit message.

2. **In `retrieve_hybrid_reranked` (after rerank stage)** — REVISED
   in v0.5 per External Audit Round 3 Finding 7. Previous forward-
   only fetch missed the middle-chunk case: per `processor.py`
   `_chunk_code_by_lines`, `partial_code=True` is set on EVERY
   chunk inside an oversized code unit, not just the leading one.
   A middle chunk needs both backward (imports/setup) AND forward
   (continuation) context. Logic is now bidirectional, bounded:
   - For each result chunk in the top-N output:
     - If `chunk.payload.get("partial_code") is True`:
       - Resolve **both "prev chunk" and "next chunk"** per the
         schema verification spike's chosen ordering rule
       - Filter both lookups to text/code modalities only (skip
         tables, images)
       - Filter the prev/next candidates to same `source_file`
         AND `partial_code=True` on the neighbor — only fetch a
         neighbor that is part of the same oversized code unit.
         Stop at the first non-`partial_code` neighbor in each
         direction (that's the boundary).
       - **Bounded window**: maximum 1 chunk backward + 1 chunk
         forward (= up to 3-chunk merge). Larger windows risk
         pulling unrelated code; if the code unit truly spans
         >3 chunks, the v2.17 safety valve (B1 Docling config
         hunt, Item #9) reopens.
       - If at least one neighbor found: build merged dict
         (concat in order: prev + current + next, omit
         not-found ones), preserving original rerank_score,
         setting `metadata.partial_code_resolved = True`, and
         recording all participating chunk_ids in
         `metadata.adjacency_source = [prev_id?, current_id, next_id?]`
       - If neither neighbor found (current is sole partial_code
         chunk — rare): mark `metadata.partial_code_resolved =
         False`; pass through unmodified
   - Top-N list output unchanged in length; merged chunks replace
     their originals in-place

3. **Bridge tests in `tests/test_retrieval_pipeline.py`** (REVISED
   in v0.5 per External Audit Round 3 Finding 7 — added
   middle-chunk + first-chunk-of-sequence cases):
   - Mock a Qdrant top-N response with a `partial_code=True` chunk
     at rank 1
   - Mock the adjacency-fetch response
   - Assert merged chunk in output with concatenated content +
     adjacency metadata
   - Assert original rerank_score preserved (no inflation)
   - Edge case 1 (**leading chunk of partial_code sequence**):
     `prev` neighbor is non-partial_code (boundary); next is
     partial_code → assert merge is `current + next` only.
   - Edge case 2 (**middle chunk of partial_code sequence**, NEW
     in v0.5): both prev and next are partial_code → assert
     merge is `prev + current + next` (3-chunk window).
   - Edge case 3 (**trailing chunk of partial_code sequence**):
     prev is partial_code; next is non-partial_code (or doesn't
     exist) → assert merge is `prev + current` only.
   - Edge case 4 (**sole partial_code chunk** — rare): no
     adjacent partial_code in either direction → assert
     pass-through + `partial_code_resolved=False`.
   - Edge case 5: prev/next is non-text modality (e.g., table) →
     assert it's skipped (treated as boundary), pass-through if
     no eligible candidate in that direction.

4. **Validation via Phase 1's Fluent_Python fixture**:
   - Run baseline (current `retrieve_hybrid_reranked` from v2.15.0)
     on the Fluent_Python validation queries — capture per-query
     PASS/FAIL via `ast.parse` on top-1 content
   - Run patched (with adjacency fetch) on the same queries
   - Compare: each query that improved from FAIL→PASS proves the
     fix; aggregate threshold per acceptance bar below

5. **No-regression check**: run
   `scripts/retrieval_regression_v2_14.py` end-to-end — must
   still pass 20/20. (Adjacency fetch triggers only on
   `partial_code=True` which the 20-query fingerprint queries
   don't currently trigger, so this should be a clean pass-through
   verification.)

**Acceptance:**
- Fluent_Python validation queries (Phase 1 deliverable): **≥70%
  pass rate** (computed across the full authored fixture, whatever
  its size in the 10-20 range per Phase 1 step 3) — top-1 must
  return a syntactically complete code block per AST.parse
  verification. (Wording revised in v0.9 per External Audit Round
  7 Finding 2 — prior "≥7/10" was ambiguous for fixtures authored
  at >10 queries; the bar is the ratio, not a literal count.)
- **Generalization test (Phase-0-dependent)**: if Phase 0's
  inventory contains ≥2 code-dense docs, run the same validation
  pattern on each — each must show **≥70% pass rate** (same ratio
  rule). If Phase 0 has <2 code-dense docs, generality is limited
  to Fluent_Python only; gap documented in Phase 3 outcome entry.
- v2.14 retrieval fingerprint: 20/20 PASS unchanged
- ≥4 new bridge tests in `tests/test_retrieval_pipeline.py`

**Cost:** 1-2 working days; $0 cloud spend.

**Risk:** low. If Fluent_Python pass rate falls short of the 70%
threshold, fall to safety-valve v2.17 (Item #9's B1 Docling config
hunt reopens) — but this is the kind of "unexpected issue" the
user explicitly allowed.

### Phase 4 — VLM-Table Dedup (A1: IoU>85% suppression)

**Goal:** resurrect v2.14 P1's force_table_vlm with the missing
dedup piece. Surgical, $0 incremental cost, no new dependencies.

**Method:**

1. **`bbox_iou` utility** added to a new module
   `src/mmrag_v2/utils/bbox.py` (or wherever bounding-box logic
   conventionally lives in the existing codebase):
   ```python
   def bbox_iou(a: BBox, b: BBox) -> float:
       """Standard Intersection-over-Union for normalized [0,1000]
       integer bbox tuples (AGENT-SPATIAL-20 invariant compliant).
       Returns 0.0 if either bbox is empty/invalid; returns float
       in [0.0, 1.0]."""
   ```
   Includes 5 unit tests covering: identical bboxes (IoU=1.0),
   disjoint (0.0), 50% overlap, contained (smaller inside larger),
   degenerate (zero-area input).

2. **Dedup logic in chunk-emission flow** (`processor.py` or
   wherever `ElementProcessor` chunk emission happens — verify
   actual location during implementation):
   - Group all chunks for a page by source page_number
   - For each page:
     - Collect VLM-table chunks (extraction_method ∈ {`vlm_table`,
       `vlm_table_markdown`, or whatever NuMarkdown-8B emits;
       verify in `processor.py`)
     - For each text chunk on the same page:
       - For each VLM-table chunk on the same page:
         - Compute `iou = bbox_iou(text.bbox, vlm_table.bbox)`
         - If `iou > dedup_vlm_table_iou_threshold` (default 0.85):
           mark text chunk for suppression (set
           `_suppress: True`)
       - Filter `_suppress=True` text chunks before emission
     - Log suppression count per page to extraction stats (for
       debuggability)

3. **Configuration knob** added to `PdfConversionPlan`:
   ```python
   dedup_vlm_table_iou_threshold: float = 0.85
   ```
   With docstring linking to this plan; flows through to the
   chunker via existing `PdfConversionPlan` → `DoclingPdfAdapter`
   → `ElementProcessor` path. **Bridge test** in
   `tests/test_pdf_conversion_plan.py` proves the knob threads
   through (existing test pattern for plan-flag-flow).

4. **Bridge tests** in `tests/test_processor.py` (or wherever
   chunk-emission tests live):
   - Mock page with 1 VLM table + 1 text chunk at IoU=0.95 →
     assert text chunk suppressed in output
   - Mock page with same shapes at IoU=0.50 → assert text chunk
     NOT suppressed
   - Mock page with VLM table + text chunk that doesn't overlap
     (IoU=0.0) → assert text chunk NOT suppressed (negative case)
   - Mock page with no VLM tables → assert all text chunks pass
     through (no false-positive on non-table pages)
   - Mock page with 2 VLM tables + 1 text chunk overlapping each
     at 0.90 → assert suppressed (any one overlap > threshold
     triggers)

5. **Re-extract CarOK** with `--force-table-vlm` + new dedup
   active:
   - Per-page chunk count comparison: with dedup vs current v2.13
     baseline; expect VLM-table count = original prose count for
     pages that had both
   - Sanity check: top-1 retrieval for a known multi-row CarOK
     query (from Phase 1 fixture) returns the VLM table, not
     the flat-prose chunk

6. **Generality across Phase 0 form-class docs** (Phase-0-dependent
   per the §3 acceptance section below).

**Acceptance:**
- CarOK Format axis ≥85% **measured on the same judge as the
  v2.13 P1 baseline (Dashscope qwen-max)** for apples-to-apples
  comparison against the 71.9% baseline / v2.14 P1's regressed
  45%. (REVISED in v0.3 per External Audit Round 1 Finding 2 —
  judge previously unspecified.) **Honest-evaluation note**: the
  v2.13 documented CarOK judge-calibration limitation on
  form-class content means the 85% target may be hitting a judge
  ceiling rather than a content ceiling; if Phase 4 lands at
  78-84% on Dashscope but Phase 1 CarOK validation queries (which
  use retrieval-only PASS/FAIL via `top_5_gold_doc` + `format_constraint`
  heuristics, no LLM judge) clear ≥75%, that's a valid SHIP per
  the second bullet below — the retrieval-fixture metric is the
  authoritative pass/fail signal; the Dashscope Format axis is
  secondary evidence. Do NOT switch judges to clear the bar
  (per [[fix-extraction-not-judge]]).
- Phase 1 CarOK validation queries: ≥75% pass rate **(authoritative
  pass/fail; retrieval-fixture-based, judge-independent)**
- **Generality (Phase-0-dependent; REVISED in v0.4 per External
  Audit Round 2 Finding 4 — replaced subjective "manual
  spot-check" with programmatic gates)**: if Phase 0's inventory
  identifies ≥2 form-class docs (per the operationalization rules
  in Phase 0 §3), re-extract each with `--force-table-vlm` + new
  dedup; validate clean output via two **programmatic** assertions
  per doc:
  1. `suppression_count_per_doc > 0` — proves dedup actually fired
     somewhere (sanity check; zero suppressions means either the
     doc had no overlapping prose+VLM tables, or the IoU threshold
     missed every overlap)
  2. `no two chunks on the same page have identical content`
     (byte-level dedup: catches the below-threshold-IoU duplicate
     case where IoU ∈ [0.5, 0.85] still produces duplicate content
     that the threshold lets through; ~10 lines of Python in the
     acceptance validation script `scripts/eval_phase4_generality.py`)

  No human inspection step (per [[no-human-verification-loops]]).
  If Phase 0 has <2 form-class docs, **CarOK-only validation IS
  the final form of the test**. No watch-item, no v2.17 deferral,
  no "if a form-class doc surfaces later" clause. Convergence-cycle
  discipline: the test runs against what we have or it doesn't run
  at all; future-corpus speculation is not a deferral mechanism.
- v2.10 strict-gate 34/34 PASS (now extended for Phase 0
  additions) unchanged
- Bridge tests pass

**Cost:** 2-3 working days; ~$0.50 cloud spend for validation
mini-soak.

**Risk:** IoU threshold (0.85) is a judgment call. Shipped as
config knob — **tunable during v2.16 execution** (before tag) and
**post-tag only under the §1 DoD item 8 / §10.1 v2.16.x boundary**
(i.e., a tuning change post-tag is a v2.16.x patch ONLY if it
fixes a demonstrable regression from v2.16.0 behavior on the
v2.16.0 corpus; tuning for better performance or for new docs
is v3.0 re-charter). (Wording corrected in v0.8 per External
Audit Round 6 Finding 7 — the previous "tunable post-hoc without
code change" was honest at v0.6 but conflicted with v0.7's
tightening of the v2.16.x boundary.)

### Phase 5 [CONDITIONAL] — Dynamic top-k from rerank logit drop-off

**Goal:** retrieval-time optimization that reduces LLM context size
+ hallucination risk on queries with sharp rerank drop-off — IF
the corpus actually exhibits such drop-offs.

**Disposition gate (NEW in v0.6 per External Audit Round 4
Finding 3):** before writing any Phase 5 production code, run a
pre-flight on Phase 1's BASELINE validation run (Phase 1 step 6,
v2.15.0 retrieval state). Apply the proposed dynamic-top-k logic
analytically to each baseline query's rerank outputs (no code
changes to production):
- Compute `would_truncate` for each query under the default
  parameters (`drop_off_threshold=2.5`, `min_absolute_gap=0.05`).
- Compute PASS-retention under the simulated truncation using
  Phase 1's gold-anchor fixtures (per F2 schema).

**Binary outcome:**
- **SHIP default-on** if ≥20% of Phase 1 validation queries
  `would_truncate` AND PASS-retention ≥ 0.97 across the full
  fixture set AND no HIGH-class fixture's simulated pass rate
  is more than 2pp below its static baseline pass rate (both
  computed from the same Phase 1 baseline run; relative bound,
  NOT against the fixture's authored `target_pass_rate` —
  REVISED in v0.9 per External Audit Round 7 Finding 1). The
  Phase 5 code below ships in v2.16 with `dynamic_top_k=True`
  as the default.
- **KILL permanently** if <20% would truncate (no meaningful
  context reduction even available) OR PASS-retention < 0.97
  (truncation hurts answer-bearing retrieval) OR any HIGH-class
  fixture's simulated rate is more than 2pp below its static
  baseline. DECISIONS.md entry: "v2.16 Phase 5 KILL —
  pre-flight evidence shows dynamic top-k has no measurable
  upside on the corpus."
- **No opt-in middle ground** — opt-in dead code is the
  feature-frozen-product failure mode the convergence cycle
  exists to prevent.

**Method (only if SHIP gate fires):**

1. **Drop-off detection logic** in `src/mmrag_v2/retrieval/pipeline.py`
   (`retrieve_hybrid_reranked`):
   - After reranker produces top-N logits:
     ```python
     logits = [r.rerank_score for r in reranked]
     if len(logits) < 2:
         return reranked  # no truncation possible
     gaps = [logits[i] - logits[i+1] for i in range(len(logits)-1)]
     mean_gap = sum(gaps) / len(gaps)
     # Find first gap that exceeds threshold * mean
     for i, gap in enumerate(gaps):
         if gap > drop_off_threshold * mean_gap and gap > min_absolute_gap:
             # Truncate at i+1 (keep rank 0 through i)
             return reranked[: max(min_top_n, i + 1)]
     return reranked  # no truncation; flat distribution
     ```
   - `drop_off_threshold = 2.5` default (3x mean gap is sharp)
   - `min_absolute_gap = 0.05` default (prevents truncation on
     uniformly-tiny logit deltas where 2.5×mean is still
     statistically meaningless)
   - Bounded: never return fewer than `min_top_n = 1` or more
     than `max_top_n = 5` (= existing `top_n_return` default)

2. **CLI / API surface (only if SHIP gate fires; default-on)**:
   - New `retrieve_hybrid_reranked` parameter:
     `dynamic_top_k: bool = True` (default ON per binary outcome
     above — the SHIP path means the corpus benefits from this).
     Callers can pass `dynamic_top_k=False` to bypass for
     diagnostic comparison.
   - New `synthetic_soak.py` flag: `--no-dynamic-top-k`
     (inverted vs v0.5 — diagnostic disable, not opt-in enable)
   - README "Retrieval options" subsection documenting the
     parameter with one-sentence description + link to Phase 5
     validation report.

3. **Bridge tests** in `tests/test_retrieval_pipeline.py`:
   - Synthetic logits `[10.0, 9.5, 9.0, 5.0, 4.9]` →
     expect truncation at index 3 (gap 9.0→5.0 is 4.0; mean gap
     across all = (0.5+0.5+4.0+0.1)/4 = 1.275; 4.0 > 2.5×1.275 →
     fires)
   - Synthetic logits `[10.0, 9.5, 9.0, 8.5, 8.0]` → expect NO
     truncation (flat; all gaps = 0.5; mean = 0.5; no gap
     exceeds 2.5×0.5 = 1.25)
   - Synthetic logits `[10.0, 10.0, 10.0, 10.0, 10.0]` → NO
     truncation (identical; all gaps = 0; no drop-off)
   - Edge: top-N of length 1 → return as-is
   - Diagnostic-disable boundary: `dynamic_top_k=False` → bypass
     logic entirely; legacy behavior preserved. Note: on the
     SHIP path, `True` is the default (v0.8 Round 6 F1 — earlier
     drafts called `False` the default, inconsistent with v0.6's
     binary SHIP-default-on disposition).

4. **Validation against Phase 1 fixtures**: run
   `run_personal_validation.py` twice — once with
   `dynamic_top_k=False` (baseline = Phase 1's baseline run),
   once with `dynamic_top_k=True`. Compare per-class pass rates +
   distribution of returned top-N sizes (histogram of how often
   1, 2, 3, 4, 5 chunks returned).

**Acceptance (only if SHIP gate fires):**
- Dynamic top-k produces variable top-N on Phase 1 validation
  queries; report distribution
- **PASS-retention bound** (carried from v0.5 Round 3 fix; HIGH-class
  invariant clarified in v0.9 per External Audit Round 7 Finding 1):
  the pre-flight gate already verifies `PASS_rate_dynamic /
  PASS_rate_static ≥ 0.97` aggregate + no HIGH-class fixture's
  simulated rate more than 2pp below its static baseline (relative
  bound; `target_pass_rate` is the fixture's authored bar for DoD
  item 3 / Phase 1 acceptance reporting, NOT a Phase 5 gate
  condition). Acceptance re-verifies these on the implemented
  code against a fresh Phase 1 run (not just the analytical
  pre-flight).
- ≥3 new bridge tests in `tests/test_retrieval_pipeline.py`
- The default-on promotion decision is ALREADY MADE by the
  disposition gate (binary SHIP/KILL); no Phase N "promotion at
  close-out" step remaining.

**Cost:** 1 working day; $0 cloud spend.

### Phase 6 [CONDITIONAL] — C1 Query Rewriting

**Triggers (REVISED in v0.8 per External Audit Round 6 Finding 2 —
both legs of the compound trigger must fire):**
1. Phase 2 diagnostic verdict identifies vocabulary mismatch (H2)
   OR cross-lingual degradation (H3) AND class-level pattern
   (not doc-specific); AND
2. Phase 2's "Phase 6 analytical pre-flight" sub-deliverable
   shows ≥3pp R@1 lift on the 5-10 manually-rewritten queries
   from the deficit class (no production code; analytical
   measurement only).

Both legs required. Either leg failing → **Phase 6 KILLs**
without implementation; DECISIONS.md entry per §3 Phase 6's
"If trigger does NOT fire" path.

**If trigger fires:** ship in v2.16 (not v2.17). Specifically:
- Use local FP8-14B to generate 2-3 rewritten queries per input
- Run retrieval against each rewrite, RRF-fuse the candidate pools
- Rerank with ModernBERT
- Acceptance: ≥3pp R@1 lift on the deficit-class subset OR KILL
  (no defer; this is the convergence cycle's discipline) — if the
  lift doesn't materialize after build, close as 2nd dead-lever
  (HyDE was the first)
- Cost: ~2-3 working days + ~$1 validation soak

**If trigger does NOT fire:** Item #7 KILLs permanently per the
disposition matrix. No deferral. DECISIONS.md gets the closure
entry.

### Phase 7 [CONDITIONAL] — D2 Retrieval-time image re-read

**Triggers:** Phase 1 validation queries surface ≥3 image-content
gaps the current pipeline cannot answer (e.g. image chunks in
top-5 but the actual answer requires re-reading the image with
the query in context).

**If trigger fires:** ship in v2.16. Specifically:
- `--enable-vision-reread` flag in retrieval / generation pipeline
- If `IMAGE` or `TABLE` chunk in final top-5: load original
  image crop, pass to local VLM (NuMarkdown-8B) with user's query,
  use VLM output as expanded chunk context
- Acceptance: image-gap validation queries: **≥70% pass rate**
  (ratio across the full authored fixture, whatever its size in
  the 10-20 range per Phase 1 step 3 — revised in v0.9 per
  External Audit Round 7 Finding 2)
- Cost: ~3-5 working days + minimal local VLM compute

**If trigger does NOT fire:** Item #8 KILLs permanently per the
disposition matrix. DECISIONS.md gets the closure entry.

### Phase N — Cycle close-out + v2.16.0 tag (FINAL v2.X TAG)

- Engine version bump `2.15.0` → `2.16.0`
- pyproject.toml + `tests/test_v2_10_release_baseline.py` version
  pin sync
- v2.16 retrieval-regression fingerprint — re-capture IF
  production retrieval shape changed. Phase 3 adjacency fetch
  triggers only on `partial_code=True` (existing fingerprint
  queries don't, so pass-through expected). **Phase 5 dynamic
  top-k** (REVISED in v0.8 per External Audit Round 6 Finding 1):
  if shipped per the Phase 1 pre-flight gate, it ships
  default-on and the fingerprint MUST be re-captured to reflect
  the new production shape. If KILL'd, no fingerprint change.
  No "opt-in promoted at close-out" intermediate case remains.
- **AFTER snapshot** `docs/QUALITY_SNAPSHOT_<date>_v2.16_after.md`
  with prominent "FEATURE-COMPLETE FOR v2.X PROJECT" banner
- DECISIONS.md entries (all required):
  - "v2.16 Decision-Mechanism Overlay" (Phase 1)
  - "v2.16 Phase 2 omlx Deficit Diagnostic Verdict" with binary
    outcome for Item #7
  - "v2.16 Phase 3 partial_code Adjacency Fetch — SHIPPED"
  - "v2.16 Phase 4 VLM-Table Dedup — SHIPPED"
  - "v2.16 Phase 5 Dynamic Top-K — SHIPPED DEFAULT-ON / KILLed-by-pre-flight" (binary; no opt-in case post v0.6)
  - Phase 6 outcome: one entry (SHIPPED OR CLOSED-as-2nd-dead-lever / KILLed — HyDE was the 1st dead lever in v2.15)
  - Phase 7 outcome: one entry (SHIPPED OR KILLed — no dead-lever framing; no prior precedent)
  - **Permanent closures** (Items #9, #10, #12, #13, #14, #15, #21,
    #22 — #10 added in v0.5 per Round 3 re-disposition): one
    DECISIONS.md "v2.16 Carry-Forward Closures" combined entry
    naming each KILL'd item + permanent-closure rationale
  - **OUT-OF-SCOPE declarations** (Item #11): one DECISIONS.md
    "v2.16 v3.0-Class Items Declared Out-of-Scope for v2.X" entry
- `PROJECT_STATUS.md` Other Carry-Forwards list: **empty**
  (verified by automated check or visual confirmation)
- **Dead-trigger process removal (NEW in v0.6 per External Audit
  Round 4 Finding 2)**: every cycle-open / cyclical process check
  whose item is KILL'd in v2.16 §2 must be removed from
  `docs/CYCLE_OPEN_CHECKLIST.md` as part of close-out. Specifically:
  - **§5 (UIR refactor trigger review)**: REMOVE entirely.
    Item #13 KILL means the four triggers are dead process;
    leaving the review in the checklist contradicts the KILL.
    Replace with a one-line confirmation: "UIR refactor: KILL'd
    permanently per `docs/PLAN_V2.16.md` §2 Item #13 +
    `docs/DECISIONS.md` v2.16 Carry-Forward Closures. If
    multi-format need arises, v3.0 re-charter, not v2.X reopen."
  - **§3 carry-forward 6.1 (Docling 2.87 OR 90-day watcher)**:
    REMOVE — Item #9 KILL closes B1 Docling config hunt
    (conditional on Phase 3 passing). If Phase 3 passes, the
    watcher is dead process; if Phase 3 fails, v2.17 safety
    valve picks up Item #9 (and the watcher would belong to
    v2.17's checklist scope, not v2.16's frozen checklist).
- README declares v2.16.0 as feature-complete for stated use case
- Layer-0/1 docs sweep per [[doc-sanitization-completeness]]
- Archive Draft v0.1 archaeology + audit-round changes to
  Appendix A
- **v2.16.0 annotated tag** with "FINAL v2.X release" message;
  pushed to origin + GitHub
- **Post-tag rollback procedure (NEW in v0.6 per External Audit
  Round 4 Finding 5)**: documented in `docs/DECISIONS.md` "v2.16
  Post-Tag Rollback Procedure" — Phase 3 (`partial_code` adjacency)
  reverts via `git revert <phase3_commit_sha>`; Phase 5 (dynamic
  top-k, only if shipped) reverts via `git revert
  <phase5_commit_sha>`. Both are independent commits on `pipeline.py`
  per §6 serial-order rule, so reverts don't conflict. This is
  the only documented revert path for feature-frozen
  post-tag bug fixes (per §1 DoD item 8, v2.16.x patches).

**Definition of Done — v2.16.0 ship gate:**

ALL of the following MUST be true (any failure routes to v2.17
safety valve per §5; no soft-state defer permitted):

- ✓ §1 Project Definition of Done items 1-8 all satisfied
- ✓ **All phases whose final disposition is SHIPPED passed their
  acceptance bars** (REVISED in v0.8 per External Audit Round 6
  Finding 3 — the prior "All SHIP phases (0-5)" wording assumed
  Phase 5 always ships, contradicting v0.6's binary disposition.
  Per-phase SHIP/KILL outcomes: Phases 0-4 unconditional SHIP;
  Phase 5 SHIP-default-on if Phase 1 pre-flight fires else KILL;
  Phase 6 SHIP if Phase 2 compound trigger fires else KILL;
  Phase 7 SHIP only on user opt-in via §8a Q3 else KILL.)
- ✓ **All KILLed conditional/opt-in phases have DECISIONS.md
  closure entries AND no production code path** — applies to
  Phase 5, Phase 6, Phase 7 when they take the KILL branch
- ✓ All §2 Disposition Matrix KILL items have DECISIONS.md closure entries
- ✓ All OUT-OF-SCOPE items have DECISIONS.md declaration entries
- ✓ `PROJECT_STATUS.md` Other Carry-Forwards: empty
- ✓ Full pytest suite green
- ✓ v2.14 retrieval fingerprint passes (or fresh v2.16
  fingerprint captured if Phase 3/5 changed production retrieval)
- ✓ Strict-gate corpus state (now incl. Phase 0 additions)
  unchanged or improved
- ✓ **Multi-profile smoke (NEW in v0.5 per External Audit Round 3
  Finding 1)**: `bash scripts/smoke_multiprofile.sh` reports
  `GATE_PASS` + `UNIVERSAL_PASS` for every document category +
  at least one per-category blind-test document. This is the
  AGENT-VAL-01 invariant from CLAUDE.md; previously omitted from
  the Phase N gate list. Failure here is a hard tag-block.
- ✓ Phase 0 calibration expiration_date > today (T-72h pre-tag
  checkpoint per v2.15 carry-over)
- ✓ README updated with "v2.16.0 feature-complete" banner

---

## 4. Permanent closures (KILLs)

For DECISIONS.md "v2.16 Carry-Forward Closures" entry. Each item
gets a one-paragraph rationale; no future-cycle reopen path.

**Item #9 — B1 Docling HybridChunker config hunt: CLOSED.** v2.16
Phase 3 (`partial_code` adjacency fetch) resolves the
Fluent_Python cross-page code defect deterministically at retrieval
time. (Closure trigger clarified in v0.7 per External Audit Round 5
Finding 2: KILL fires only when Phase 3 passes its FULL acceptance
bar — Fluent_Python ≥70% AND any generalization docs ≥70%. (Bar
phrased as a ratio in v0.9 per External Audit Round 7 Finding 2.) If
Phase 3 fails on either leg, the bidirectional adjacency-fetch
approach is insufficient for the corpus's code-dense distribution,
and Item #9 reopens to v2.17 for the Docling-side investigation
that was deferred under Option F.) The Docling-side fight is no
longer needed; if a future corpus class surfaces a different
chunking defect, that's a v3.0-class re-architecting decision,
not a v2.X carry-forward. Original carry-forward 6.1 trigger
("Docling minor ≥2.87 OR every 90 days") REMOVED from
`CYCLE_OPEN_CHECKLIST.md`.

**Item #12 — B2 Code-Rescue heuristic stitching middleware: CLOSED.**
(Rationale rewritten in v0.4 per External Audit Round 2 Finding 7 —
previously KILL-by-audit-reference; now self-contained for future
maintainers.) Heuristic regex-based stitching of truncated code
across chunk boundaries is a chunking-layer analog of the lenient-
judge trap — it masks the extraction defect (Docling intermixing
prose+code, producing `partial_code=True` chunks) rather than
fixing it at the right layer. Maintaining the stitching regexes
becomes its own debt burden as Docling output shape evolves. The
correct fix is v2.16 Phase 3's retrieval-time adjacency fetch
using the deterministic `partial_code` schema flag emitted at
extraction time — it preserves chunk provenance (no
content-mutation), avoids regex-maintenance debt, and is
triggered only on the chunks that need it. Rejected in v2.15
Round-1 audit and confirmed at v2.16 convergence cycle.

**Item #13 — 3c UIR refactor: CLOSED.** Five cycles of PARKED with
zero forward motion. The four "PARKED WITH TRIGGERS" conditions
(3rd document engine, cross-engine chunking defect, ≥500 LOC test
boilerplate, external integration request) are not realistic for
this project's solo-dev PDF-only use case. If multi-format ever
becomes load-bearing, that's a v3.0-class re-charter as a fresh
architecture proposal — not a perpetual carry-forward.

*Context for future maintainers (NEW in v0.3 per External Audit
Round 1 Finding 5):* the v2.15 PARKED-WITH-TRIGGERS DECISIONS.md
entry documents the four trigger conditions in detail and the
v2.11→v2.15 metrics that informed each (engine count = 1 [Docling
only]; cross-engine defect count = 0; test boilerplate LOC remained
well under the 500-LOC trigger across all five cycles; no external
integration request landed). A future maintainer encountering an
EPUB / Word / HTML ingest requirement should NOT treat this as a
v2.16 reopen — that pathway is `git log v2.15.0..` for the
historical metrics + a new v3.0 architecture proposal, since
multi-format support is a re-charter, not a refactor of v2.16's
PDF-locked pipeline.

**Item #14 — 3a VLM swap: CLOSED.** (Rationale rewritten in v0.6
per External Audit Round 4 Finding 7 — v0.5 said "works" without
citing evidence.) v2.14 Phase 1 `force_table_vlm` soak measured
NuMarkdown-8B-Thinking-mlx-8bits and found: clean 5-column
markdown tables on 5/12 CarOK pages, $0 cost, structurally correct
output. The Phase 1 mini-soak Format regression (-26.9pp) was
caused by VLM-table chunks coexisting with flat-prose duplicates
that won retrieval 29/30 times — a DEDUP defect, not a VLM-quality
defect. v2.16 Phase 4 ships the missing dedup piece (IoU>0.85
suppression). With dedup in place, the existing VLM is sufficient
for the documented use case. **Insufficient evidence of VLM
quality deficit to justify swap cost.**

**Item #15 — 3e Magazine rendered-region-crop: CLOSED.** (Rationale
rewritten in v0.6 per External Audit Round 4 Finding 1 — v0.5
admitted weak evidence then KILL'd anyway, which is the
rationalization pattern the convergence-cycle discipline forbids.)
**Honest framing**: insufficient evidence to justify investment.
Magazine content meets existing strict-gate quality bars
(Combat_Aircraft_August_2025 and PCWorld_July_2025 chunks pass;
image axes score consistently per AFTER snapshots), but the v2.15
telemetry infrastructure is too new to have accumulated a full
evidence cycle — "no documented complaints across v2.11→v2.15"
is NOT statistically-rigorous evidence-of-absence. Convergence-
cycle discipline requires permanent closure now (deferral is
forbidden); the disposition is "KILL on insufficient evidence
threshold," not "KILL on demonstrated absence." If magazine-class
validation queries authored in a future product (v3.0) surface
real gaps, that's a corpus-class re-charter; not a v2.X reopen.

**Item #10 — A2 HTML+summary split: CLOSED.** (Re-disposed in v0.5
per External Audit Round 3 Finding 9 from OUT-OF-SCOPE → KILL.)
The Unstructured-pattern proposal — emit a long-form chunk for
embedding alongside a short summary chunk for display — is a
chunk-emission pattern change in the ingestion path, not an
embedder retraining or vector-store schema rewrite. The original
OUT-OF-SCOPE label overstated the scope. On honest examination,
this is a 2-5 day v2.X-class feature that could have shipped. But:
zero demand signal across v2.11→v2.15 (no validation queries
surface a summary/long-form mismatch; no strict-gate failure
attributable to chunk-shape); the current single-chunk pattern
works for the documented corpus. Convergence-cycle KILL is
correct for the same reason Item #15 KILLs — no demand. If a
future user-authored validation query against summary-shaped
content surfaces a real gap, that's a v3.0 chunk-shape re-charter,
not a v2.X carry-forward.

**Item #21 — 3b Remote CodeFormulaV2 inference: CLOSED.** (NEW in
v0.4 per External Audit Round 2 Finding 1 — previously silently
dropped from carry-forward list v2.12→v2.15 without formal
disposition.) Original v2.11 disposition was "defer-with-named-
workaround" pending Docling 2.87+ exposing `RemoteCodeFormulaOptions`;
Docling remains pinned at 2.86.0 and the trigger never fired in 5
cycles. The local-CPU CodeFormulaV2 lane in Docling 2.86 (~27 sec/
page on Apple Silicon per CLAUDE.md "Workstream B Code Enrichment
Guardrail") is sufficient for this project's one-off batch
reconversion workflow. Remote inference is a throughput
optimization for large-corpus or multi-tenant workloads, neither
of which applies to a solo-dev personal corpus. v3.0 re-charter
if multi-tenant or large-batch needs ever emerge; not a v2.X
carry-forward.

**Item #22 — 3d HybridChunker per-item token guard: CLOSED.** (NEW
in v0.4 per External Audit Round 2 Finding 1 — previously silently
dropped from carry-forward list across v2.12→v2.15.) v2.11 design
called for an opt-in `--strict-hybrid-guard` flag in `ingest_to_
qdrant.py` (default-off, preserves v2.10 chunker shape); the flag
was never built (verified: zero hits in src/scripts/tests). v2.12
marked the item "subsumed by Phase 4 if shipped"; v2.12 Phase 4
(per-doc-class chunking) was NOT triggered (per v2.12 archaeology:
"Phases 1+2 already clear all embedder-attributable floors").
Result: 4 cycles of zero forward motion + zero demand signal. The
v2.10 element-by-element fallback path already handles pathological
chunker inputs without the per-item guard. Quality optimization on
an edge case that has never surfaced as a defect; closing is
convergence-cycle discipline.

---

## 5. v2.17 safety valve (tight scope)

User-authorized escape hatch for unexpected issues arising
**during convergence execution**. NOT a re-introduction of the
deferral pattern.

**v2.17 triggers (exhaustive list — no others permit defer):**

(REVISED in v0.5 per External Audit Round 3 Findings 4 + 5:
trigger #4 previously covered "post-tag CONDITIONAL phase trigger
fires" — DELETED as inconsistent with §3 Phase N ship gate's
"Both CONDITIONAL phases resolved (SHIPPED or KILLed — no third
state)." Post-tag KILL reversal is a v3.0 re-charter, never v2.17.
New trigger #4 makes the previously-hidden §7 day-12 overflow
mechanism explicit and gates it on user sign-off.)

1. **SHIP phase acceptance bar genuinely FAILS** and the fix is
   non-trivial (>2 dev days). Example: Phase 3 partial_code fetch
   produces ≤30% Fluent_Python pass rate (vs the ≥70% bar), AND
   investigation reveals a structural issue not addressable in the
   cycle. **Acceptance-bar failure**, not schedule overflow — those
   are separate (see trigger #4).
2. **External dependency breaks** during convergence: Docling
   pin breaks; omlx-server protocol incompatibility; Qdrant
   schema change forces re-ingestion at scale.
3. **Strict-gate regression**: any Phase 0/3/4/5 work causes
   34/34 PASS to drop. This is a stop-the-line condition; fix in
   v2.17, do not ship v2.16 with regressed strict-gate.
4. **Convergence-cycle schedule overflow with explicit user
   sign-off**: at the 12-day hard cap (per §7), if a SHIP phase
   has not completed AND the user signs off in writing
   (`docs/cycle_slip.log` or DECISIONS.md entry) that the
   remaining work cannot be compressed inside the cap, the
   unfinished phase routes to v2.17 with the sign-off rationale
   recorded. **Without explicit user sign-off, the day-12 overflow
   BLOCKS the v2.16 tag** — the default is "no tag," not
   "silent v2.17 absorption." Extending the cap is a legitimate
   alternative if the user chooses; that decision is also recorded.

**v2.17 NON-triggers (these route to KILL or wait, never to
v2.17 escalation):**

- "Let's also do X" — no, X is KILL or OUT-OF-SCOPE per §2.
- "Phase 5 turned out smaller than expected, let's add Y" — no,
  ship and tag.
- Audit-round findings discovered late in convergence — fold into
  current cycle, no v2.17 defer.
- New ideas from external review (Gemini, audit prompts) — fold
  in or KILL via the §2 disposition matrix; no v2.17 deferral.

**v2.17 budget if triggered:** narrow remediation only. Single
SHIP phase (the trigger's fix); single AFTER snapshot patch;
v2.17.0 tag. No new feature work in v2.17.

**Post-v2.17 (if it fires)**: same project DoD applies. v2.17 closes
to same target state (Other Carry-Forwards: empty).

---

## 6. Phase ordering

```
Phase 0  Corpus expansion           ← MUST land first (gates 2 + 4)
Phase 1  Decision-mechanism overlay  ← MUST land before 3 + 4 + 5 (gates acceptance + Phase 5 pre-flight)
Phase 2  omlx diagnostic spike      ← Drives Phase 6 trigger; can run parallel to 3-5
Phase 3  partial_code adjacency     ← Independent
Phase 4  VLM-Table Dedup            ← Depends on Phase 0 (form-class docs)
[Phase 5 CONDITIONAL] runs only if Phase 1 pre-flight (≥20% truncation + ≥0.97 PASS-retention) fires; else KILL
[Phase 6 CONDITIONAL C1] runs only if Phase 2 verdict triggers
[Phase 7 OPT-IN D2] runs only if user opts in via §8a Q3
Phase N  Close-out + tag            ← Terminal; FINAL v2.X tag
```

**Isolation rule** (carried from v0.9): each phase's validation
mini-soak runs separately; no confounding-variable conflation.

**Phase 3 + Phase 5 serial-order rule (NEW in v0.4 per External
Audit Round 2 Finding 10)**: both modify `retrieve_hybrid_reranked`
in `src/mmrag_v2/retrieval/pipeline.py` after the rerank stage —
the adjacency-fetch block (Phase 3) and the dynamic-top-k block
(Phase 5) touch overlapping post-rerank logic. **Merge order:
Phase 3 ships first; Phase 5 branches from post-Phase-3 main
before implementation starts.** Avoids a guaranteed merge conflict
in `retrieve_hybrid_reranked`. If Phase 3 fails acceptance and
routes to v2.17 (per §5 trigger #1), Phase 5 is unaffected and
proceeds independently (its changes are in a different region of
the function).

---

## 7. Budget

- **Cost cap:** $25/cycle (unchanged)
- **Estimated cloud spend (all phases, no conditional triggers):**
  - Phase 0: $0
  - Phase 1: $0
  - Phase 2: ~$0.50
  - Phase 3: $0
  - Phase 4: ~$0.50
  - Phase 5: $0
  - Phase N: ~$1
  - **Total: ~$2-3**
- **Estimated cloud spend (worst case — both conditional triggers fire):**
  - Above + Phase 6 ($1-2) + Phase 7 (minimal local VLM) = **~$4-6**
- **Wall-clock budget (REVISED in v0.2 per Round-0 Finding 9 + 12):**
  - Phase 0: 0.5-1 day
  - Phase 1: 0.5 day implementation + N × 0.5 day query authoring,
    where N = HIGH+MED classes (minimum 2 = CarOK + Fluent_Python;
    Phase 0 may add up to 5 MED classes); realistic range
    **1.5-4 days** total
  - Phase 2: 1 day
  - Phase 3: 1-2 days (incl. 30-min schema verification spike)
  - Phase 4: 2-3 days
  - Phase 5: 1 day
  - Phase 6 (if triggered): 2-3 days
  - Phase 7 (if triggered + opt-in via §8a Q3): 3-5 days
  - Phase N: 0.5-1 day
  - **No conditional triggers**: 7-12 working days
  - **Both conditional triggers fire**: 12-20 working days
- **Convergence cycle hard cap: 12 working days.** At day 12,
  any remaining SHIP phase aborts. Routing is **gated on
  explicit user sign-off per §5 trigger #4** (REVISED in v0.5
  per External Audit Round 3 Finding 4 — previously routed
  silently to trigger #1, which is dishonest because schedule
  overflow ≠ acceptance-bar failure). If the user signs off,
  the unfinished phase ships as v2.17 with sign-off rationale
  recorded in `docs/cycle_slip.log` or DECISIONS.md. Without
  sign-off, the v2.16 tag is BLOCKED — default behavior is "no
  tag," not "silent v2.17 absorption." Extending the cap is a
  legitimate alternative the user can choose; that decision is
  also recorded. The 12-day cap is wall-clock-inclusive (GX10
  contention budget per Round-4 Finding 8 v0.5 of the v2.15 plan).

---

## 8. Open questions and proposed defaults

### 8a. Real open questions (require user input before execution)

1. **DoD item 3 threshold** (validation query per-class pass rate):
   currently set at **≥75%** in §1 — loose because some failures
   will be known-and-accepted as documented limitations. Should
   this be tightened to **≥85%**, kept at 75%, or set per-class
   (HIGH classes 85%, MED classes 75%)?

   **BLOCKING gate**: must be answered before Phase 1 starts
   (validation queries are authored against this threshold per
   Phase 1 method step 3).

2. **RESOLVED in v0.7 per External Audit Round 5 Finding 1 —
   orphaned by Phase 5's v0.6 binary re-disposition.** The prior
   Q2 asked about Phase 5 promotion-to-default-on criteria with
   three options including "stay opt-in." v0.6 (Round 4 F3)
   eliminated the opt-in path entirely: Phase 5 disposition is
   now binary (SHIP default-on OR KILL) per the Phase 1 pre-flight
   gate documented in §3 Phase 5 and §2 Item #6. No promotion
   decision remains at Phase N close-out. The reference is kept
   as a placeholder (renumbering Q3 → Q2 would break external
   citations from Round 1-4 archaeology); answer is locked.

3. **(NEW in v0.2 per Round-0 Finding 2)** **Phase 7 scope
   opt-in**: Phase 7 (D2 retrieval-time image re-read) is
   currently EXPECTED KILL because the Phase 1 validation set
   (CarOK + Fluent_Python) doesn't naturally surface
   image-content gaps. Three options:
   - (a) **Accept EXPECTED KILL**: Phase 7 closes at Phase N
     with no test attempt; image re-read deferred to v3.0 if
     ever needed. (Recommended: lowest scope; image-content
     gaps haven't been a documented problem.)
   - (b) **Add an image-heavy class to Phase 1 scope**: e.g.,
     promote `PCWorld_July_2025` or `Combat_Aircraft_August_2025`
     to documented-limitation status with `personal_importance =
     HIGH`; author 10-20 validation queries that specifically
     test image-content questions; Phase 7 becomes real
     CONDITIONAL SHIP gated on those queries' failure pattern.
     (+ ~1 day to Phase 1 authoring)
   - (c) **Add multiple image-heavy classes** (PCWorld +
     Combat_Aircraft + a Phase 0 image-heavy addition if one
     surfaced): broader coverage; +1.5-2 days to Phase 1.

   **Reviewable, non-blocking (REVISED in v0.8 per External
   Audit Round 6 Finding 6 — v0.7 listed Q3 as both blocking and
   defaulted, which is internally contradictory)**: Phase 1
   starts on the default (option a — accept EXPECTED KILL) if the
   user has not actively answered before the cycle opens. The
   user can override before Phase 1 starts by selecting option
   (b) or (c); after Phase 1 fixture authoring begins, override
   requires a recorded rationale and an extension to Phase 1's
   budget per the cost line. Default-on-silence is consistent
   with the convergence-cycle KILL discipline.

### 8b. Proposed defaults (running with unless you object)

These are decisions I've made for the plan; flagging here so you
can override before execution. If you don't override, these stand.

*Defaults are numbered independently of §8a real questions
(renumbered in v0.9 per External Audit Round 7 Finding 4 — prior
3/4/5/6 numbering was an iteration artifact from when §8a had two
items).*

1. **Phase 1 personal_importance initial classification**:
   - `CarOK_voorraadtelling`: HIGH (you wanted dedup work in v2.14;
     v2.16 Phase 4 ships the missing piece)
   - `Fluent_Python`: HIGH (Python content is core workflow;
     v2.16 Phase 3 ships the partial_code fix)
   - Any documented-limitation classes that emerge from Phase 0
     inventory: default MED
   - You can override any of these before Phase 1 ships.

2. **README "feature-complete" banner**: at v2.16.0 ship, I'll
   add a banner near the top of README.md stating:
   > "**MM-Converter-V2 is feature-complete as of v2.16.0**.
   > Production retrieval is stable; documented limitations are
   > explicit; only bug-fix patches (v2.16.x) accepted
   > post-tag. New features = re-charter as v3.0."

   Override if you want different wording, no banner, or a
   different placement.

3. **Phase 0 inventory report format**: I'll generate
   `docs/CORPUS_EXPANSION_2026-05-24_v2.16_p0.md` listing per-doc
   profile + chunk counts + extraction warnings, derived from
   pipeline output only (not source-doc inspection per the
   bias-discipline). Override if you want a different report
   shape or location.

4. **§5 v2.17 trigger interpretation**: the four enumerated
   trigger conditions are exhaustive (per the convergence-cycle
   discipline). If an issue arises during execution that doesn't
   match any of the four, default behavior is "fold into v2.16,
   no defer." Override if you want a different default response
   to genuinely-unexpected issues.

---

## 9. Process notes

- v2.15 closed with 8 audit rounds against its plan. Convergence-
  cycle plan recommendation: audit until stopping rule fires
  (2 consecutive 0-HIGH rounds) just like v2.15. Expect
  4-6 revision rounds against this Draft v0.1.
- Audit-round process applies BEFORE phase execution. The
  v2.17 safety valve does NOT permit shipping un-audited fixes.
- The convergence-cycle frame is itself reviewable: if an audit
  round flags "this disposition matrix is too aggressive — some
  KILL items should defer instead," that's a legitimate finding to
  consider. But the bar is high: each KILL has rationale; ad-hoc
  "let's keep this open" preferences should not override §2.
- Memory rules carry forward unchanged: `no-gx10-model-swap-reflex`,
  `gx10-deployment-guardrails`, `fix-extraction-not-judge`,
  `contract-violation-mode`, `libraries-first`.
- Post-v2.16.0: the cycle-plan workflow ends. v2.17 only if a
  trigger from §5 fires. After v2.17 (if it fires): bug fixes
  only as v2.17.x patches. No more v2.X plans.

---

## 10. Post-convergence governance

(NEW in v0.7 per External Audit Round 5 Findings 4 + 5 + 8 —
post-tag operational concerns previously implicit are now
documented.)

### 10.1. v2.16.x patch lane

- **Scope**: bug fixes only, per §1 DoD item 8 and the
  v2.16.x-vs-v3.0 boundary defined there.
- **Versioning**: `v2.16.1`, `v2.16.2`, etc. — increments on
  each shipped patch.
- **No audit-round process** for v2.16.x patches: single-PR
  flow, reviewer is the user, ship as soon as the fix is
  verified against the affected v2.16.0-corpus regression.
- **What counts as a "bug fix" for v2.16.x** (operational
  examples): regression in extraction output on a v2.16.0-corpus
  doc that worked at tag time; broken external dependency (omlx
  / Qdrant / Docling) that production retrieval depends on; a
  config knob value that the v2.16.0 ship gate proved sufficient
  but a subsequent change in dependencies has invalidated.
- **What does NOT count as a "bug fix"**: tuning a threshold for
  better performance on the existing corpus; adding support for
  a new doc class; adding a new validation query; modifying
  acceptance bars. All of these are v3.0.

### 10.2. v3.0 re-charter conditions

A v3.0 project is opened when ANY of the following hold:

1. **New format support needed** (HTML / RTF / DOCX / etc.) —
   the UIR refactor that Item #13 KILL'd in v2.16 becomes a
   genuine v3.0 work item.
2. **New retrieval-quality class** — a category of queries that
   the v2.16.0 retrieval stack cannot handle within documented
   limitations (e.g., visual retrieval driving the Item #11
   ColPali OUT-OF-SCOPE reopen).
3. **Corpus growth ≥2× v2.16.0** — fundamentally different
   scale changes the cost/benefit of design decisions made for
   the v2.16.0-size corpus.
4. **User explicit re-charter decision** — convergence-cycle
   discipline was the user's choice; the user can re-charter
   for any reason, but the decision is recorded.

### 10.3. v3.0 re-charter process

1. Author `docs/PLAN_V3.0.md` (this plan's structure as
   template; new disposition matrix scoped to v3.0 work).
2. Re-evaluate every v2.16 KILL and OUT-OF-SCOPE item against
   v3.0 scope; explicit decision to keep KILL'd or revive.
3. Open an audit cycle against the v3.0 plan (same audit
   prompt structure; same stopping rule — two consecutive
   0-HIGH rounds).
4. v3.0 is a fresh feature cycle, not a continuation of v2.X.
   Re-baseline against the latest v2.16.x state (not v2.16.0).

### 10.4. Opt-in post-convergence health monitoring

(NEW in v0.7 per External Audit Round 5 Finding 5 partial-accept —
post-tag regression detection beyond user-reported bugs.)

- **What**: a periodic `scripts/run_personal_validation.py` run
  comparing PASS rates to the v2.16.0 baseline captured at Phase
  1 step 6.
- **When**: opt-in (not required); a maintainer who wants
  proactive regression detection can schedule this monthly or
  on corpus change. Default behavior is reactive (bug reports
  surface regressions when they affect actual workflows).
- **Threshold**: ≥5pp regression on any HIGH-class fixture is a
  v2.16.x candidate per the patch-lane scope above.
- **Why opt-in not required**: per [[no-human-verification-loops]],
  a "monthly manual check" required by process is the failure
  mode in a solo-dev feature-frozen product. The script exists
  for use; nobody is obligated to run it on a schedule.

---

## Appendix A — Audit-round archaeology

### External Audit Round 8 (2026-05-25) — 9 findings, 1 PARTIAL + 8 REJECTED/SUPERSEDED (0 HIGH accepted as structural)

Audit done against an **outdated Draft v0.6** (the auditor's
header explicitly says "Convergence-Cycle Plan v2.16 (Draft v0.6)"
and the overall stance recommends "v0.7 revision prior to the
final Round 6 execution" — three iterations behind the v0.9 state
the audit was supposed to evaluate). The mismatch produced 7
re-flagged findings already fixed in Rounds 4-7 (the auditor
couldn't see those fixes) plus 2 substantive new findings.

**Round-8 stopping-rule status: 0 HIGH accepted as structural** —
the precedent from Round 1 (3 HIGH-claimed findings rejected or
downgraded → "0 HIGH accepted as structural") applies. **Second
consecutive 0-HIGH round → v2.15 §9 stopping rule FIRES.** Cycle
clears the audit gate; Phase 0 execution can begin pending the
pre-execution checklist's user-input items (§8a Q1 blocking + §8b
defaults review).

| # | Sev (reviewer) | Sev (final) | Disposition | Resolution in v0.10 |
|---|---|---|---|---|
| 1 | HIGH | — | **REJECTED — already fixed in Round 5 F1 (v0.7)**. §8a Q2 is explicitly marked "RESOLVED in v0.7 per External Audit Round 5 Finding 1"; pre-execution checklist Q2 is checked `[x]` as RESOLVED. Auditor reviewed v0.6 which predated this fix. | None — fix already in plan since v0.7. |
| 2 | HIGH | — | **REJECTED — re-litigates Round 4 F3's accepted CONDITIONAL SHIP design** (the binary pre-flight gate itself). Audit prompt's "What NOT to do" explicitly excludes re-flagging Rounds 0-4 findings. Auditor's recommended fix (use 34-doc canonical baseline) is structurally unworkable — those docs don't have gold-anchor fixtures, so PASS-retention can't be measured against them. The pre-flight measures on Phase 1's curated validation fixtures BY DESIGN because those are the queries with gold-anchor ground truth. | None — re-litigation of locked Round 4 disposition. |
| 3 | HIGH | — | **REJECTED — already fixed in Round 4 F2 (v0.6)**. Phase N close-out explicitly contains the bullet "§3 carry-forward 6.1 (Docling 2.87 OR 90-day watcher): REMOVE — Item #9 KILL closes B1 Docling config hunt." Auditor read v0.6 but missed this Phase N "dead-trigger process removal" subsection. | None — fix already in plan since v0.6. |
| 4 | MED-confidence-HIGH | — | **REJECTED — Round 5 audit prompt's Lens 2 explicitly considered this**. The prompt asked: "What if one of the 7 docs is in a format Docling chokes on, forcing a 2nd engine?" Round 5's reviewer addressed it and did not produce a re-disposition finding. The 7 new docs are PDFs — existing scanned/scanned_degraded profile routing in the ProfileClassifier handles format variation without requiring a "2nd engine." Item #13's UIR refactor was about multi-engine routing across PDF + non-PDF formats; PDF format expansion is not the trigger. | None — Round 5 already considered. |
| 5 | HIGH | MED-PARTIAL | **PARTIAL ACCEPT** — the underlying concern (a doc landing at 28% code-chunks or 38% table-chunks could silently bypass the matching Phase 3 / Phase 4 lane) is real, even if Round 5's Probe B + the existing step-2 threshold pre-validation cover related cases. Auditor's recommended fix ("manual arbitration for borderline documents") was REJECTED as [[no-human-verification-loops]] violation. **Programmatic reformulation ACCEPTED**: Phase 0 step 3b gains Probe C — every doc whose class-determining metric falls within ±5% of any boundary threshold gets a `NEAR_BOUNDARY_*` flag in the inventory report. Signal-only; user decides at the existing Phase 0 acceptance review (no new human-loop step). Bias-discipline-compliant (computed from existing inventory data; no source-PDF inspection). | Phase 0 step 3b: new Probe C with explicit threshold-band logic + step-4 inventory-report fields updated. |
| 6 | MED | — | **REJECTED — hard cap on Phase 4 generality scope would create soft state** ("we didn't test this form-class doc because of the cap"), directly contradicting convergence-cycle discipline. §5 Trigger #4 (schedule-overflow-with-explicit-user-sign-off) already handles the overflow case the auditor describes; with 7 unknown docs and a Probe A misclassification possibility, the worst case for new Phase 4 generality docs is ~3-4 — manageable inside the 12-day cap. | None — §5 Trigger #4 already handles. |
| 7 | HIGH | — | **REJECTED — re-litigates Round 3 F4/F5 + Round 1 F7**. Round 1 F7 was already rejected on related grounds (would collapse Trigger #4 into Trigger #3, eliminating the legitimate post-tag CONDITIONAL phase pathway). The current Trigger #4 (schedule overflow with explicit user sign-off; default = tag BLOCK) is intentional for a solo-dev product where the user IS the project authority. Adding programmatic gates ("immutable cryptographic artifact") to override user authority is enterprise-process over-engineering. Audit prompt's "Don't recommend adding process steps as a substitute for fixing structural problems" applies. | None — re-litigation of locked Round 3 disposition. |
| 8 | HIGH | — | **REJECTED — re-litigates the Round 7 F2 fix and conflicts with the designed v2.17 fallback**. "≥7/10" wording was already fixed in Round 7 F2 (now "≥70% pass rate"); auditor read v0.6 which predated this. The substantive concern (queries needing >3 chunks fail by design) is the EXACT scenario the v2.17 safety valve was designed for — Phase 3 Risk paragraph explicitly says "If Fluent_Python pass rate falls short of the 70% threshold, fall to safety-valve v2.17 (Item #9's B1 Docling config hunt reopens)." Auditor's recommended fix (pre-filter acceptance queries by capability) would make the bar unfalsifiable — any failure could be retroactively excluded as "needed too much context." | None — v2.17 fallback IS the designed response; bar must stay falsifiable. |
| 9 | MED | — | **REJECTED — superseded by Round 5 F5's [[no-human-verification-loops]] disposition**. Round 5 F5 already considered active monitoring proposals and accepted only the §10.4 opt-in compromise after the reviewer's "monthly manual run" fix was rejected. Auditor's "lightweight daily/weekly automated script" requires the user to set up cron + maintain it indefinitely on a feature-frozen product — same memory violation by a different route. The concrete failure mode (BM25 sparse collection drift causing wrong-chunk merging) is real but extremely narrow, and §10.4's opt-in `run_personal_validation.py` would surface a corpus-level regression via pass-rate drop without requiring active hash verification infrastructure. | None — §10.4 opt-in compromise stands. |

### External Audit Round 7 (2026-05-25) — 5 findings, all 5 accepted (0 HIGH + 3 MED + 2 LOW)

Audit done against Draft v0.8. **Round-7 stopping-rule status:
FIRST round to return 0 HIGH-severity findings** — the
structural-finding rate has dropped to zero, exactly the
convergence-signal pattern the user predicted ("Round 7 should be
a tight regression-only audit"). All 5 accepted findings are
wording/parameterization fixes; no §2 disposition changes; no
phase Method changes. Per v2.15 §9 stopping rule (two consecutive
0-HIGH rounds), **Round 8 is required as a tight regression audit
on v0.9 to verify these 5 fixes don't introduce new orphans**. If
Round 8 returns 0 HIGH, the cycle clears the stopping rule and
Phase 0 execution can begin.

| # | Sev (reviewer) | Sev (final) | Disposition | Resolution in v0.9 |
|---|---|---|---|---|
| 1 | MED | MED | **ACCEPTED** — Phase 5 pre-flight gate's HIGH-class invariant ("no HIGH-class fixture drops below `target_pass_rate`") was absolute-vs-relative ambiguous: CarOK's `target_pass_rate=0.85` is the authored bar, not the baseline; if baseline is already 0.78 (a real possibility for a HIGH-class documented-limitation), the gate auto-KILLs Phase 5 even when simulated rate is neutral-to-positive (0.79). DECISIONS.md entry would then misleadingly attribute KILL to "no measurable upside" when the actual cause was an unsatisfiable gate. Two wording sites: §3 Phase 5 disposition gate AND §3 Phase 5 acceptance bar. | Phase 5 disposition gate + acceptance bar: HIGH-class invariant rewritten as relative bound (simulated rate within 2pp of static baseline). `target_pass_rate` clarified as fixture-authored bar for DoD item 3 / Phase 1 reporting, NOT a Phase 5 gate condition. |
| 2 | MED | MED | **ACCEPTED** — "≥7/10" wording (literal ratio) conflicts with Phase 1 step 3's 10-20-query authoring rule: a 14-query fixture's PASS/FAIL is ambiguous (9 passes = 64% but ≥7 absolute). Five sites across the plan: Phase 3 Fluent_Python acceptance, Phase 3 generalization acceptance, Phase 7 image-gap acceptance, §2 Item #9 trigger text, §4 Item #9 closure rationale, §5 v2.17 trigger #1 example, Phase 3 Risk paragraph. Auditor's recommendation also named Phase 4 generality but Phase 4's form-class acceptance is programmatic (suppression_count + byte-dedup), no ≥7/10 bar there. | All seven occurrences across the five sites converted to "≥70% pass rate (computed across the full authored fixture, whatever its size in the 10-20 range per Phase 1 step 3)" or the equivalent ratio framing. Phase 4 not in scope (already programmatic). |
| 3 | MED | MED | **ACCEPTED** — principle-application inconsistency, not a soft state: §2 Item #6 KILLs Phase 5's opt-in middle ground with "opt-in dead code is the failure mode for a feature-frozen product" — but §1 DoD item 1 retains the HyDE opt-in knob as switched-off dead-lever infra. A future maintainer or Round 8 auditor reading both would conclude either Phase 5 KILL is over-strict, or HyDE knob should be ripped out in v2.16.x (a §10.1 boundary violation), or convergence principles are applied selectively. Distinction is real (HyDE is a top-level boolean with zero entanglement; Phase 5 would interleave with adjacency fetch deep in `retrieve_hybrid_reranked`) but was implicit. | §2 Item #18 rationale extended with one paragraph explicating the entanglement-cost distinction: top-level toggles (HyDE) are not what the "no opt-in dead code" principle is calibrated for; the principle targets opt-in surfaces that introduce two interacting live code paths (Phase 5 would). HyDE knob remains inert; ripping it out would be a §10.1 v2.16.x scope expansion (not a v2.16.0-corpus regression fix). |
| 4 | LOW | LOW | **ACCEPTED** — §8b items numbered 3, 4, 5, 6 (an iteration artifact from when §8a had two items); §8a's three items + §8b's first item now happen to share number 3, creating mild reader friction. Pure cleanup, exactly the Lens 13 (iteration-introduced orphan) pattern. | §8b renumbered to 1/2/3/4 with explicit header note: "Defaults are numbered independently of §8a real questions." |
| 5 | LOW | LOW | **ACCEPTED** — §1 DoD item 3 hardcodes "≥75%" but §8a Q1 is BLOCKING with three legitimate answers (keep 75% / tighten to 85% / per-class HIGH 85% / MED 75%). If user picks anything other than 75%, the ship-gate check "§1 DoD items 1-8 all satisfied" becomes interpretively ambiguous — does CarOK at 0.82 PASS (≥75% baseline) or FAIL (<85% HIGH-class target)? Convergence cycle is supposed to eliminate exactly this kind of late-cycle interpretation friction. | DoD item 3 reworded as parameterized on Q1's answer: "per-class pass rate satisfies the threshold set by §8a Q1 (default 75% if Q1 not actively answered before Phase 1 starts)." Q1's three legitimate answers enumerated inline for clarity. |

### External Audit Round 6 (2026-05-25) — 7 findings, all 7 accepted (4 HIGH + 1 MED + 2 LOW)

Audit done against Draft v0.7. Round-6 stopping-rule status: **4
HIGH iteration-fallout findings accepted**. All 4 HIGHs were
propagation gaps from Round 5's v0.7 edits — exactly the pattern
Round 5's reviewer warned about ("Round 6 likely worthwhile only
as a short regression audit for orphaned references"). Does NOT
clear the v2.15 §9 stopping rule. Round 7 required.

| # | Sev (reviewer) | Sev (final) | Disposition | Resolution in v0.8 |
|---|---|---|---|---|
| 1 | HIGH | HIGH | **ACCEPTED** — Phase 5 default-off remnants across 3 sites contradicting v0.6's SHIP-default-on-OR-KILL binary disposition. §1 DoD item 1 said "opt-in dynamic-top-k knob"; Phase 5 bridge test boundary said `dynamic_top_k=False (default)`; Phase N fingerprint bullet said "opt-in unless promoted." All three left over from the pre-binary state. | §1 DoD item 1 rewritten; Phase 5 step 3 bridge-test boundary annotated; Phase N fingerprint bullet rewritten. |
| 2 | HIGH | HIGH | **ACCEPTED** — Round 5 F3 added a Phase 2 analytical pre-flight for Phase 6 but didn't propagate the requirement into §2 Item #7's trigger row or §3 Phase 6's Triggers section. Compound trigger now stated in both. | §2 Item #7 trigger rewritten as compound; §3 Phase 6 Triggers section restructured into two-leg compound. |
| 3 | HIGH | HIGH | **ACCEPTED** — Phase N ship gate said "All SHIP phases (0-5) passed acceptance" but Phase 5 may KILL under v0.6's binary disposition. The "0-5" framing predates v0.6's re-disposition. Rewritten to use "final disposition is SHIPPED" wording that handles all phase outcomes (unconditional SHIP, conditional SHIP, conditional KILL, opt-in SHIP, opt-in KILL). | Phase N ship-gate bullets restructured. |
| 4 | LOW | LOW | **ACCEPTED** — disposition summary KILL count formula corrected. The v0.6 "may grow to 10 if Phase 5 fails" wording undercounted: Item #7 can KILL if Phase 2/Phase 6 trigger fails, Item #8 defaults to KILL unless opt-in. Worst case: 12 KILLs. | §2 disposition summary formula expanded. |
| 5 | MED | MED | **ACCEPTED** — Phase 0 Probe B (v0.7 Round 5 F6) flagged borderline minority-language docs but didn't specify Phase 2's behavior. Explicit rule: included in H3 sample, NOT counted toward ≥3-doc class-level threshold (avoids letting borderline-flagged docs falsely promote a doc-specific verdict to class-level). | Phase 0 step 3b extended with Phase 2 H3 sampling rule. |
| 6 | HIGH | HIGH | **ACCEPTED** — §8a Q3 was both "blocking" (per pre-execution checklist) and "defaulted to (a) if no answer" (per Q3 text). Self-contradictory. Resolution: keep the default, demote from blocking to "reviewable, non-blocking" — consistent with convergence-cycle KILL discipline (silence = KILL path proceeds). | §8a Q3 rewritten; pre-execution checklist updated. |
| 7 | LOW | LOW | **ACCEPTED** — Phase 4 Risk wording ("tunable post-hoc without code change") was honest at v0.6 but conflicted with v0.7's tightening of the v2.16.x boundary. Amended: tunable during v2.16 execution; post-tag only under §1/§10.1 regression-fix boundary. | Phase 4 Risk amended. |

### External Audit Round 5 (2026-05-24, Deepseek V4) — 8 findings, 7 accepted (2 HIGH + 2 MED + 1 partial-MED + 3 LOW)

Audit done against Draft v0.6. Round-5 stopping-rule status: **2
HIGH iteration-introduced-inconsistency findings accepted** —
both surfaced exactly the Lens 13 pattern Round 5's prompt was
designed to catch (v0.6 edits left orphaned references and an
incomplete trigger update). Does NOT clear the v2.15 §9 stopping
rule. Round 6 required.

| # | Sev (reviewer) | Sev (final) | Disposition | Resolution in v0.7 |
|---|---|---|---|---|
| 1 | HIGH | HIGH | **ACCEPTED** — §8a Q2 (Phase 5 promotion-to-default-on) was orphaned by v0.6 Round 4 F3 re-disposition of Phase 5 to binary outcome. Q2 also still referenced R@1 (replaced by PASS-retention in v0.5) — two layers of stale. Also: §1 DoD item 3 had a "§8 Q2" cross-reference that was an §8a-reorder relic (should have been Q1, the threshold question). | §8a Q2 marked RESOLVED; pre-execution checklist updated; §1 DoD item 3 cross-reference corrected to §8a Q1. |
| 2 | HIGH | HIGH | **ACCEPTED** — §2 Item #9 trigger ("Fluent_Python ≥7/10") covered only one leg of Phase 3's compound acceptance bar. If Fluent_Python passed but generalization failed, Item #9 would be wrongly KILL'd while Phase 3 routed to v2.17 — contradictory dispositions. | §2 Item #9 trigger rewritten to require Phase 3's FULL acceptance bar (Fluent_Python AND any generalization docs). §4 Item #9 rationale updated to match. |
| 3 | MED | MED | **ACCEPTED** — symmetric demand-signal test: Round 4 required Phase 5 to pre-flight on Phase 1 baseline data; same test wasn't applied to Phase 6. Phase 6 ships 2-3 days of code on a 1-day hypothesis; 30-min analytical pre-flight (rewrite 5-10 deficit queries, check R@1 delta) catches "hypothesis right but no measurable lift" before the build cost. | Phase 2 method gains a new "Phase 6 analytical pre-flight" sub-deliverable; symmetric with Phase 5's. |
| 4 | MED | MED | **ACCEPTED** — Phase 4 said IoU threshold "tunable post-hoc," Item #16 said thresholds "FROZEN at v2.16.0," §1 DoD item 8 said "only bug fixes" without defining what's a bug fix. Three statements, internally inconsistent. Defined the v2.16.x-vs-v3.0 boundary uniformly. | §1 DoD item 8 expanded; new §10.1 + §10.2 + §10.3. |
| 5 | MED | LOW-partial | **PARTIAL ACCEPT** — post-tag dormant-monitoring concern is real, but the reviewer's "monthly manual run" fix violates [[no-human-verification-loops]]. Compromise: documented opt-in maintenance procedure (the script exists for use; no required schedule). | §10.4 added as opt-in recommendation. |
| 6 | LOW | LOW | **ACCEPTED** — Phase 0 probe extended to cover OCR-stripped-diacritics minority-language pattern. Probe B is signal-only (no auto-reclassification); Phase 2 uses the flag to decide H3 sample composition. | Phase 0 step 3b extended with Probe B. |
| 7 | LOW | LOW | **ACCEPTED** — Phase 0 pre-validation abort condition added for fundamentally-broken classification rules (dependency change scenario). Routes to v2.17 trigger #2. | Phase 0 step 2 abort condition. |
| 8 | LOW | LOW | **ACCEPTED** — v3.0 entry conditions + re-charter process explicit. Folded with F4 into the new §10 Post-Convergence Governance section. | §10.2 + §10.3 added. |

### External Audit Round 4 (2026-05-24, Kimi-K2.6) — 8 findings, 6 accepted + 1 partial + 1 moot

Audit done against Draft v0.5. Round-4 stopping-rule status:
**2 HIGH structural findings accepted** + **1 disposition change**
(Phase 5 unconditional SHIP → CONDITIONAL with binary outcome).
Does NOT clear the v2.15 §9 stopping rule. Round 5 required.

| # | Sev (reviewer) | Sev (final) | Disposition | Resolution in v0.6 |
|---|---|---|---|---|
| 1 | HIGH | HIGH | **ACCEPTED** — v0.5 Item #15 rationale admitted weak evidence then KILL'd anyway. Same KILL, honest rationale ("insufficient evidence to justify investment; telemetry too new to have accumulated a cycle") rather than overclaimed evidence ("5 cycles of zero demand signal"). | §4 Item #15 rewritten. |
| 2 | HIGH | HIGH | **ACCEPTED** — genuine contradiction: §2 Item #13 KILL'd UIR permanently but CYCLE_OPEN_CHECKLIST.md §5 still ran trigger review each cycle-open. Resolution: Phase N close-out must remove §5 (and the §3 Docling 2.87 watcher tied to Item #9 KILL). Cleanest path — checklist itself is current v2.15.x process; the v2.16 KILL takes effect at Phase N. | Phase N close-out new "dead-trigger process removal" bullet covering §5 + §3 of CYCLE_OPEN_CHECKLIST. |
| 3 | MED | MED (disposition change) | **ACCEPTED, structural** — convergence-cycle discipline applied consistently: same "no demand signal in 5 cycles" pattern that KILLs #10 and #15. Phase 5 re-disposed from unconditional SHIP → CONDITIONAL with binary outcome (SHIP default-on OR KILL — no opt-in middle ground). Trigger sharper than reviewer's: pre-flight on Phase 1's baseline rerank outputs without writing production code. | §2 Item #6 row updated; Phase 5 header + Goal + new "Disposition gate" section; Method/Acceptance gated; §6 phase ordering updated; Phase N DECISIONS list updated. |
| 4 | MED | HIGH | **ACCEPTED, graduated MED→HIGH** — naming should be semantic, not numeric. The "rename if non-round" rule was non-monotonic and conditional on an irrelevant property (count cardinality). Rename to CANONICAL_DOCS unconditionally. | Phase 0 step 5 sub-item 4 reworded. |
| 5 | MED | LOW-partial | **PARTIAL ACCEPT** — Round 1 reviewer's full rollback playbook was rejected as solo-dev process substitute; this narrower one-line revert procedure for the two pipeline.py phases is cheap and useful for feature-frozen maintenance (post-tag v2.16.x patches per §1 DoD item 8). Accepted the one-line addition; rejected any expansion. | Phase N close-out new bullet citing exact `git revert` commands. |
| 6 | MED | MED | **ACCEPTED** — solves the Round 2 / Round 3 bias-discipline blind spot. Reviewer's programmatic probe operates only on pipeline outputs (re-extraction with different flags), never on source PDFs — preserves bias-discipline. Triggers only when classification pattern looks suspicious (scanned + image>0 + tables=0). | Phase 0 method new step 3b. |
| 7 | LOW | LOW | **ACCEPTED** — Item #14 rationale rewritten from "works" assertion to cite v2.14 P1 evidence (5/12 CarOK pages clean output; dedup was the defect, not VLM quality). | §4 Item #14 rewritten. |
| 8 | LOW | — | **MOOT under F3 acceptance** — Phase 5 becomes binary (SHIP default-on or KILL); no opt-in middle ground remains, so the "opt-in feature nobody knows about" failure mode the finding addresses cannot occur post-v0.6. If F3 had been rejected, F8 would stand as a valid LOW. | Documentation requirement folded into Phase 5 CLI/API section as part of SHIP default-on case. |

### External Audit Round 3 (2026-05-24) — 10 findings, 9 accepted (5 HIGH + 3 MED + 1 partial)

Audit done against Draft v0.4 (reviewer correctly identified the
current version, didn't re-flag v0.3/v0.4 changes). Round-3
stopping-rule status: **5 HIGH structural findings accepted**.
Does NOT clear the v2.15 §9 stopping rule. Round 4 required.

| # | Sev (reviewer) | Sev (final) | Disposition | Resolution in v0.5 |
|---|---|---|---|---|
| 1 | HIGH | HIGH | **ACCEPTED** — CLAUDE.md AGENT-VAL-01 requires `smoke_multiprofile.sh` `GATE_PASS` + `UNIVERSAL_PASS` for acceptance; Phase N ship gate omitted it. | Phase N ship gate — explicit smoke-multiprofile bullet added. |
| 2 | HIGH | HIGH | **ACCEPTED** — Phase 1 step 4 PASS rule (doc-id membership + format heuristic) lets a query PASS on the wrong chunk from the right doc. | Phase 1 step 3 fixture schema: `gold_chunk_ids` + `expected_anchor_regexes`. Step 4 PASS rule: gold-anchor must hit. |
| 3 | HIGH | HIGH | **ACCEPTED** — verified factually: 5 CANONICAL_34 sites (2 definitions + 2 import-consumers + 1 test with hard-coded length assertion). Phase 0 updated only synthetic_soak.py. | Phase 0 step 5 — enumerate all 5 sites + atomic rename rule + new anti-drift bridge test. |
| 4 | HIGH | HIGH | **ACCEPTED** — §7 stretched §5 trigger #1 ("acceptance bar FAILS") to silently absorb schedule overflow. Schedule overflow ≠ acceptance failure. | §5 trigger #4 replaced with explicit schedule-overflow-with-sign-off; default = tag-block, not silent v2.17. §7 updated to match. |
| 5 | HIGH | HIGH | **ACCEPTED** — §5 trigger #4 (post-tag conditional firing) contradicted Phase N ship gate's "Both CONDITIONAL phases resolved at tag." Post-tag KILL reversal is v3.0 re-charter, not v2.17. | Old §5 trigger #4 DELETED; replaced (see Finding 4 above). |
| 6 | MED | MED | **ACCEPTED** — Phase 0 acceptance allowed "planning continues with docs that DID ingest" — soft state. | Phase 0 acceptance: any failure is a Phase 0 FAIL requiring explicit user resolution (drop-from-scope OR fix-and-re-ingest). |
| 7 | MED | MED | **ACCEPTED** — verified via processor.py docstrings: `partial_code=True` is set on EVERY chunk inside an oversized code unit, so middle chunks need bidirectional adjacency. Forward-only fetch produces "middle+tail" without imports. | Phase 3 step 2 now bidirectional (prev + current + next, bounded ≤3-chunk window). Step 3 bridge tests added: middle-chunk, leading-chunk, trailing-chunk, sole-partial-code cases. |
| 8 | MED | HIGH | **ACCEPTED, graduated MED→HIGH** — mathematically sharp: rerank-based truncation drops ranks 2-N, never rank 1. My v0.4 fix picked R@1 as the gate, which is structurally insensitive to the regression dynamic top-k is most likely to cause. PASS-retention (built on F2's gold-anchor fixture) is the right metric. | Phase 5 acceptance: PASS-retention bound (`PASS_dynamic / PASS_static ≥ 0.97`) + HIGH-class invariant + diagnostics. R@1 demoted to "reported, not gated." |
| 9 | MED | MED | **ACCEPTED** — Item #10 OUT-OF-SCOPE rationale was assertion-backed not evidence-backed. A2 HTML+summary is a chunk-emission change, not embedder retraining. Re-disposed as KILL (no demand signal in 5 cycles, parallels Item #15). | §2 Item #10 → KILL; §4 closure rationale added; §2 summary table + Phase N closure list updated. |
| 10 | MED | — | **REJECTED** — IoU > 0.85 means near-identical bboxes. Captions adjacent to a table have separate bbox regions; page-level chunks have a bbox covering the page (IoU with embedded table ≪ 0.85). The 0.85 threshold is itself the geometric defense for the reviewer's failure modes, and Phase 4 risk note already exposes it as a tunable config knob if a real case emerges. | None — finding rejected. |

### External Audit Round 2 (2026-05-24, Deepseek) — 10 findings, 7 accepted (1 HIGH + 4 MED + 2 LOW)

Audit done against Draft v0.2; findings re-evaluated against v0.3
state. Round-2 stopping-rule status: **1 HIGH structural finding
accepted** (missing v2.11 carry-forwards 3b/3d — exactly the
self-audit-dodged structural class the audit prompt anticipated).
Does NOT clear the v2.15 §9 stopping rule. Round 3 required.

| # | Sev (reviewer) | Sev (final) | Disposition | Resolution in v0.4 |
|---|---|---|---|---|
| 1 | HIGH | HIGH | **ACCEPTED** — verified factually: 3b documented in DECISIONS.md L690 (defer-with-named-workaround); 3d marked "subsumed by Phase 4 if shipped" in v2.12; v2.12 Phase 4 archaeology confirms "NOT triggered"; `--strict-hybrid-guard` flag absent from src/scripts/tests. Both silently dropped v2.12→v2.15. Matrix exhaustion claim was false. | §2 Items #21 (3b KILL) + #22 (3d KILL); §4 closure rationales; Phase N closure list updated. |
| 2 | HIGH | — | **REJECTED** — fix violates [[no-human-verification-loops]]. Even with the "human reads metrics, not source PDFs" framing, the reviewer's proposal is still a human verification step; humans are inconsistent and lazy at this. Finding 5's pre-validation against the known 34-doc corpus is the programmatic equivalent and addresses the same underlying concern without the human dependency. | None — superseded by Finding 5 acceptance. |
| 3 | MED | MED | **ACCEPTED** — "no R@1 regression" was genuinely undefined at the per-query boundary. Defined as `R@1_dynamic / R@1_static ≥ 0.95` aggregate AND no HIGH-class fixture below its `target_pass_rate`. | Phase 5 acceptance section. |
| 4 | MED | MED | **ACCEPTED** — "manual spot-check" is a [[no-human-verification-loops]] violation. Replaced with two programmatic gates: `suppression_count_per_doc > 0` + byte-level same-page duplicate detection. | Phase 4 generality acceptance. |
| 5 | MED | MED | **ACCEPTED** — classification thresholds were untested at draft time. Added a pre-Phase-0 validation step against the known 34-doc canonical corpus (Fluent_Python/Python_Cookbook/Python_Distilled must classify code-dense; HARRY/CarOK must not). Bias-discipline-compliant (no source-PDF peek on the 7 unknown docs). | Phase 0 method step 2 (new). |
| 6 | LOW | LOW | **ACCEPTED** — Item #8 labeling improved: "EXPECTED KILL / CONDITIONAL" → "OPT-IN (default KILL)". Disposition summary row also updated (1 conditional + 1 opt-in, not 2 conditional). | §2 Item #8; §2 summary table. |
| 7 | LOW | LOW | **ACCEPTED** — Item #12 KILL rationale was audit-reference-only. Replaced with self-contained reasoning (extraction-vs-retrieval-layer argument + regex-maintenance-debt argument + provenance-preservation argument). | §4 Item #12. |
| 8 | LOW | — | **REJECTED** — Phase 4's re-extraction operates only on docs Phase 0 already classified as form-class. The scope-oscillation scenario (a "general" doc reclassified to form-class after Phase 4 re-extraction) doesn't occur because "general" docs aren't re-extracted in Phase 4. | None — finding rejected. |
| 9 | LOW | — | **REJECTED** — reviewer missed §1 DoD item 8: "Post-v2.16.0: only bug fixes (v2.16.x patches). No new features." Post-tag corpus-level bug fixes (e.g., IoU threshold tuning, dedup-suppression-of-text-chunks defect) go to v2.16.x patches, not v2.17. v2.17 is for unexpected issues during convergence execution, not post-tag bugs. The lane the reviewer wanted already exists. | None — finding rejected. |
| 10 | LOW | LOW-partial | **PARTIAL ACCEPT** — merge-conflict-resolution-ownership framing remains rejected (solo-dev). But the narrow piece — "Phase 3 ships first, Phase 5 branches from post-Phase-3 main" — is a 3-line clarification with non-zero value even for solo-dev when branches overlap in flight. Added to §6. | §6 Phase 3+5 serial-order rule (new). |

### External Audit Round 1 (2026-05-24) — 7 findings, 4 accepted

Findings against Draft v0.2 from external review. Round-1 stopping-
rule status: **0 HIGH accepted as structural** (3 HIGH downgraded
to partial or rejected on examination), 2 MED accepted, 1 LOW
deferred. Does NOT yet clear the v2.15 §9 stopping rule (need two
consecutive 0-HIGH rounds); next round is Round 2.

| # | Sev (reviewer) | Sev (final) | Disposition | Resolution in v0.3 |
|---|---|---|---|---|
| 1 | HIGH | — | **REJECTED** — fix violates Phase 0 bias-discipline (explicit methodological constraint) + [[no-human-verification-loops]] memory. Underlying audit lens 10 concern (silent misclassification) already mitigated by Phase 4's "if <2 form-class docs, CarOK-only IS the final test" blast-radius framing. | None — finding rejected. |
| 2 | HIGH | MED-partial | **PARTIAL ACCEPT** — judge ambiguity in Phase 4 acceptance is real; dual-judge fix is over-engineered. Specified Dashscope qwen-max as the judge (apples-to-apples with v2.13 baseline) + added honest-evaluation note about CarOK judge-calibration limitation + clarified Phase 1 retrieval fixture is the authoritative pass/fail signal. | Phase 4 acceptance section. |
| 3 | HIGH | MED-partial | **PARTIAL ACCEPT** — merge-owner / rollback-playbook portions REJECTED (solo-dev: Phase 3+5 are sequential commits, not parallel branches; git revert + tags suffice). Qdrant snapshot before Phase 0 mutation ACCEPTED — Qdrant lives outside git, snapshot is the only revert path for index-level corruption. | Phase 0 method step 4 — pre-mutation snapshot. |
| 4 | MED | MED | **ACCEPTED** — telemetry "KEEP active" without maintenance contract is soft state in a feature-frozen product. CI-smoke portion of fix dropped as scope creep; threshold-freeze + retention-policy portions added. | §2 Item #16 row. |
| 5 | MED | LOW-partial | **PARTIAL ACCEPT** — "reopen procedure" framing rejected (fights convergence-cycle KILL discipline). Future-reader context portion accepted — added concrete metrics + pointer to v2.15 PARKED-WITH-TRIGGERS DECISIONS.md entry. | §4 Item #13 rationale. |
| 6 | MED | — | **REJECTED** — Phase N IS the close-out phase, not a deferral target. Promotion criteria are already concrete (zero R@1 regression + ≥20% truncation rate); deciding on Phase 5 validation evidence at Phase N close-out is normal phase ordering, not soft state. The reviewer's "decide immediately after Phase 5 validation" is cosmetic relabeling — same decision, same evidence, same person. | None — finding rejected. |
| 7 | LOW | — | **REJECTED** — recommended fix ("narrow to verified production regression") would collapse trigger #4 into trigger #3 and eliminate the legitimate post-tag CONDITIONAL phase pathway. Trigger #4 is already narrow: it names exactly Phase 6 or 7 conditions, not arbitrary reopens. | None — finding rejected. |

### Draft v0.1 → v0.2 — Round-0 self-audit (14 findings)

*(See changelog table at the top of this plan, lines 27-40.)*

---

## Pre-execution checklist

Before any phase begins:

- [x] Plan has passed two consecutive audit rounds with 0 HIGH-severity findings (v2.15 §9 stopping rule applies; Round-0 self-audit doesn't count toward the stopping-rule count — formal external audits do). **FIRED 2026-05-25 at Round 8**: Round 7 (5 findings: 0 HIGH + 3 MED + 2 LOW accepted against v0.8) was the first 0-HIGH round; Round 8 (9 findings: 0 HIGH accepted as structural + 1 PARTIAL MED + 8 rejected/superseded against v0.6 mis-versioned audit) was the second. Per Round 1 precedent ("0 HIGH accepted as structural" with all reviewer-HIGH downgraded to rejected), Round 8 qualifies even though the reviewer flagged 7 HIGH-claimed findings — the disposition table in Appendix A documents the verification-based rejection rationale.
- [ ] User has reviewed §8a real open questions:
  - [ ] Q1 (DoD validation pass-rate threshold) — **BLOCKS Phase 1** (validation queries are authored against this threshold)
  - [ ] Q3 (Phase 7 scope opt-in) — **REVIEWABLE, non-blocking** (REVISED in v0.8 per External Audit Round 6 Finding 6 — Q3 has a default of option (a), so silence is a valid answer; Phase 1 proceeds on the default unless user actively selects (b) or (c))
  - [x] Q2 (Phase 5 promotion criteria) — **RESOLVED in v0.7**: Phase 5 disposition is binary (SHIP default-on / KILL) per the Phase 1 pre-flight gate. No promotion decision required at Phase N.
- [ ] §8b proposed defaults reviewed — any to override?
- [ ] §2 disposition matrix accepted as-is OR amendments agreed (any KILL contested → re-discuss; final list locked before execution)

The cycle does not open until all gates above clear.
