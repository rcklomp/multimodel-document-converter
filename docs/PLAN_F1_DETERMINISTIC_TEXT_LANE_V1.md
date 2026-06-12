# PLAN F1 - Deterministic text-layer code extraction + the 5.4 quality-risk loop

**Status: rev. 3 RATIFIED 2026-06-12 (user direction to proceed, recorded by the
M1 session; the 2026-06-12 review three amendment conditions are incorporated and
BINDING). Phase 1 build AUTHORIZED with the Section 6 oracle as the non-negotiable
exit criterion. Phase 2 remains gated on Section 8 decision 3 (re-extraction
budget).** Phase 0 (read-only
attribution, a-d) was executed 2026-06-12; results in Section 1.1. The
2026-06-12 external review verified the modality seam and the P2 inference, and
imposed three amendment conditions, all applied in this revision: (a) an
independent code-fidelity oracle in Section 6 (the audit gate's per-chunk test
is a nesting-PRESENCE check the repair satisfies by construction); (b) Phase 3's
ingest goal re-scoped against the un-root-caused MISSING_PAGES blocker (now
Phase 0(f), in scope); (c) "B decisively" downgraded to "B is the right first
spike" with the oracle as the spike's exit criterion. Per the review's
recommendation, the items submitted for immediate ratification are Phase 0(e)+(f)
and the modality-seam fix; the Phase 1 build is gated on ratifying this rev. 3.
Author: M1 Claude session, from the issue register item F1 (user-verified
2026-06-12), the R3 3-cause diagnosis (Mini commit `8b47371`), and the Phase 5
full-corpus failure evidence. Companion governance: `PLAN_EXTRACTION_FIDELITY_V1`
Section 5.4 (quality-risk consumers, spec'd unbuilt), `ARCHITECTURE_V3.1_CHARTER`
(post-Phase-5 rewrite), DECISIONS "Phase 4 - hybrid default".

---

## 1. The measured problem

Phase 5 full-corpus run (2026-06-11/12): 7 of 11 processed docs QA_FAILed, all
technical/code books, none ingested (the gates correctly refused). Verified
failure shape (Devlin): `AUDIT_FAIL (CODE)`, code_indentation_fidelity 0.341
(gate 0.90), plus 8 MISSING_PAGES. Engine `mineru_qwen_hybrid`, `degraded=0` -
these are PRIMARY-path failures, not ladder artifacts.

Page-class scan of the 7 books (PyMuPDF, 2026-06-12; textlay = pages with >100
text-layer chars; imgonly = <=20 chars + images; mono10 = pages crossing the
router's monospace>=0.10 signal):

| doc | pages | textlay | imgonly | mono10 |
|---|--:|--:|--:|--:|
| The_Complete_C_Python_Coding_Manual | 148 | 0 | 148 | 0 |
| Earthship_Vol1 | 236 | 0 | 236 | 0 |
| Jungjun_Build_an_AI_Agent | 266 | 265 | 1 | 4 |
| Adedeji_GenAI_on_Google_Cloud | 320 | 316 | 0 | 67 |
| Bourne_Unlocking_Data | 346 | 335 | 1 | 163 |
| Chaubal_AI_Projects_PyTorch | 359 | 358 | 0 | 216 |
| Devlin_Building_LLM_Agents | 365 | 358 | 2 | 0 |

Three populations, three distinct mechanisms:

- **P1 - born-digital, router-blind (Devlin 0 mono pages, Jungjun 4).** The code
  is in the text layer with exact indentation, but set in fonts the monospace
  signal cannot see; every page routes to MinerU, which mangles indentation
  (0.34). The "font-blind" cause the `8b47371` diagnosis sized as tiny on
  FluentPython (11/766) is a WHOLE-BOOK failure class here.
- **P2 - born-digital, router-visible, still failing (Chaubal 216, Bourne 163,
  Adedeji 67 mono pages).** These pages route to the Qwen VLM lane - a raster
  round-trip of text that exists losslessly in the PDF. Whether the failing
  chunks come from the VLM pages, the sub-threshold MinerU pages, or both is
  UNKNOWN and is the first Phase 0 question.
- **P3 - image-only raster (C++ manual 148/148, Earthship 236/236).** No text
  layer exists; font signals are blind by construction; MinerU OCR mangles code
  indentation. The specified-but-unbuilt Section 5.4 bounded re-extraction is
  the named fix. (Earthship is in this population but is NOT a code book - its
  failure class must be attributed in Phase 0 before it is allowed to gate this
  plan.)

The architectural reading (register item F1): the pipeline already computes
DIGITAL/SCANNED per-document diagnostics but uses them only in the fallback
ladder. For born-digital pages, the highest-fidelity extraction SOURCE available
is the PDF text layer - deterministic input, zero inference cost. NOTE on
exactness (review finding 4): PDF text layers store glyph POSITIONS, not leading
whitespace, so indentation is INFERRED from x-coordinates - a positional
reconstruction, not a literal read. "Deterministic" in this plan means
reproducible-from-source; it does not promise exact indentation, which is why
Section 6 carries an independent fidelity oracle.

### 1.1 Phase 0 results (executed 2026-06-12, read-only; deliverables a-f DONE)

**(d) Population corrections - two books exit the code-failure set:**
- Earthship: `AUDIT_FAIL (LABEL)` on a 313-image scanned book - scanned-prose
  class, NOT code. Exits this plan (Section 8 decision 2 resolves to
  "separate register item" on evidence).
- Adedeji: `AUDIT_FAIL (IMAGE)` (placeholder ratio 0.897, no-VLM run) - its
  CODE gate passed. Exits the code set; re-test under a VLM-enabled run.

The code-failure set is therefore **4 born-digital books + the C++ manual (P3)**:

| book | gate fid | code chunks via Qwen | via MinerU | per-lane judgeable pass |
|---|--:|--:|--:|---|
| Devlin | 0.341 | 0 | 78 | MinerU 34% (29/44 fail) |
| Jungjun | 0.386 | 0 | 215 | MinerU 39% (81/132 fail) |
| Bourne | 0.686 | 342 | 26 | **Qwen 71%** (33/115), MinerU 17% (5/6) |
| Chaubal | 0.774 | 341 | 51 | **Qwen 78%** (36/161), MinerU 67% (1/3) |

**(a) Lane attribution: BOTH engines fail born-digital code.** P1 books fail
catastrophically in the MinerU lane (0.34-0.39). P2 books fail moderately INSIDE
the Qwen lane itself (0.71-0.78 at scale) - the AIOS 1.00 sample did not
generalize; per-lane numbers recompose into the gate values exactly. Consequence:
re-routing more pages to Qwen cannot fix P2; the raster round-trip itself loses
~25% of judgeable indentation. The text layer is the only exact source.

**(b) Recovery fire-rate root cause: a modality seam, not bbox.** Every code
chunk in all 4 books carries page_number + bbox (100%). The recovery loop
(`batch_processor.py:5738`) gates on `ch.modality != Modality.TEXT: continue` -
but V3 PROMOTES code chunks to `Modality.CODE`, so the promoted population is
skipped by construction. Same seam class as the QA-side metric fixed 2026-06-06
(R3 redesign); the recovery side was never updated. Mechanism B's mapping
prerequisite is therefore HEALTHY; only its trigger gate is dead.

**(c) Census:** flagged-candidate (mono) pages carrying images: 0-15 per book -
mechanism A's structure-loss cost would be small, but (a)+(b) favour B anyway.

**(f) MISSING_PAGES attribution (executed 2026-06-12, blank-aware, reusing the QA
`_read_blank_pages_in_source` detector + a PyMuPDF per-missing-page text/image
census).** The headline "8/7/4/1 = 20 missing pages" largely dissolves:
- **13/20 are BLANK source pages** (Bourne 7/7, Adedeji 4/4, Devlin 2 of 8) ->
  already `MISSING_PAGES_BLANK` advisory, NOT content loss. Bourne and Adedeji
  have ZERO genuine loss; their QA_FAIL is elsewhere (Adedeji = no-VLM IMAGE gate,
  per (d)). The Phase 5 driver DID gate with `qa_full_conversion --source-pdf`, so
  these were already advisory in that run.
- **6/20 are Devlin genuine drops, but PROSE** (pages 139/242/278/283/296/303;
  70-92 unique tokens each, numbered review/solutions sections; 0 images; text
  present). The extraction emitted no chunk for the whole page with NO error in
  the per-book log (silent drop). They are NOT code pages -> **OUT of F1's
  deterministic-code-lane scope**; this is a distinct silent prose-page-drop
  mechanism. Each carries a repeated leading running-header token (e.g.
  "Solutions" x N) - a candidate interaction with an anti-repetition/quality
  filter, to confirm in the separate item.
- **1/20 is a genuine CODE drop** (Chaubal p278: 1452 chars, 17 code keywords,
  `scheduler.step(...)`, 0 images) -> the only F1-lane-addressable missing page.
Consequence (sharpens review finding 2): MISSING_PAGES is NOT a large ingestion
co-blocker. After recognizing the 13 blanks (no action) and opening a SEPARATE
register item for the 6 Devlin silent prose drops, the F1 lane's missing-page
debt is a single code page. Phase 3's whole-doc `QA_PASS*` for Devlin still needs
the prose-drop item resolved, but that item is not code and not this lane.

**(e) Candidate-signal over-trigger measurement (executed 2026-06-12, read-only,
against the frozen Workstream B negatives + positive controls).** Per-channel
scores (c1=leading-whitespace ladder, c2=code-keyword-START fraction,
c3=short-ragged-right):

| fixture | class | c1_indent | c1_depths | c2_kw | c3_ragged |
|---|---|--:|--:|--:|--:|
| incidental_shell | NEG | 0.20 | 1 | 0.00 | 0.40 |
| sparse_fenced | NEG | 0.00 | 0 | 0.00 | 0.17 |
| magazine_prose | NEG | 0.00 | 0 | 0.00 | 0.25 |
| magazine_keyvalue | NEG | 0.00 | 0 | 0.00 | 0.78 |
| R2_nested_list | NEG | 0.75 | 2 | 0.00 | 0.75 |
| R2_poetry | NEG | 1.00 | 2 | 0.00 | 0.50 |
| chaubal_code | POS | 0.67 | 2 | 0.67 | 0.67 |
| fluent_fenced | POS | 0.25 | 1 | 0.50 | 0.75 |

Conclusions for the 4.1 calibration (thresholds still frozen for Phase 1):
- **C2 (code-keyword-START density) is the only clean discriminator**: every
  negative = 0.00, positives >= 0.50. The keyword-START semantics (not substring)
  are why incidental shell (`pip install`, `python check_sensors.py`) and
  magazines score 0. C2 must be a REQUIRED (necessary) channel, not a vote.
- **C1 (indentation ladder) over-triggers exactly as R2 predicted**: nested-list
  0.75/2-depths and poetry 1.00/2-depths are indistinguishable from code on
  indentation alone. C1 is CONFIRMING-ONLY, never sufficient, and must be gated
  behind C2.
- **C3 (short ragged-right) does not discriminate** (magazine key-value 0.78,
  nested-list 0.75 >= positives): DROP it as a positive vote.
- A safe candidate rule exists (`c2_kw >= ~0.30 AND c1_depths >= 2`; fenced code
  stays on the existing fence channel): all six negatives fail the C2 gate, so the
  over-trigger contract is satisfiable. This UNFREEZES the 4.1 threshold work for
  Phase 1.

**Modality-seam trigger-gate fix SHIPPED (commit `c95950b`, 2026-06-12):** the
recovery loop in `_apply_code_hygiene` now admits promoted `Modality.CODE` chunks
(previously TEXT-only), with an inline flat/indented guard so already-indented
code is not re-extracted; 5 unit tests; suite + offline smoke green. This is the
narrow (b) fix only - it makes the EXISTING recovery fire on the promoted
population. The Mechanism B generalization (all code chunks on text-native pages,
signal calibration, the Section 6 fidelity oracle) remains Phase 1, gated on
ratifying rev. 3.

## 2. Thesis and non-goals

**Thesis:** code content on born-digital pages must be served from the PDF text
layer deterministically (P1/P2); raster pages get the already-specified 5.4
quality-risk loop (P3). No engine replacement, no re-litigation of Phase 1/4:
MinerU stays the structure/layout/table engine, Qwen stays the raster-code
specialist. This plan only stops sending DIGITAL text through a raster
round-trip when its failure class is code.

**Non-goals (binding):**
- No change to the hybrid default, the rollback hatch, or the ladder.
- No gate relaxation anywhere; R3/code gates stay as-is (Test Contract
  Integrity). The fix must move the MEASUREMENT'S INPUT, not the bar.
- No new ElementType (frozen at 3, Charter 7.1) - code stays smuggled as TEXT +
  `promoted_modality`, the existing pattern.
- AGENT-SPATIAL-20 (single 20-unit vertical threshold) untouched.
- Scanned PROSE quality (Earthship, if Phase 0 attributes it as non-code) is
  OUT of scope - register it separately rather than scope-creep this plan.

## 3. Existing mechanisms (libraries/in-tree first - build on, do not duplicate)

1. **Chunk-level PyMuPDF indentation recovery EXISTS**
   (`batch_processor.py` ~5770-5875): maps a flat code chunk's [0,1000] bbox to
   PDF points, clips `get_text("dict")`, reconstructs indentation from x
   positions, stamps `code_repair_applied`. Observed firing on ~0-1 chunks per
   failed book ("recovered indentation for 0 chunks"). Phase 0 must answer WHY
   (bbox/page metadata missing on hybrid-path chunks? `indentation_fidelity=0`
   precondition never true? clip mismatch?) before any new mechanism is built -
   the cheapest win may be making the existing one fire.
2. **Router page signals** (`router.py:page_has_code_block`, monospace
   char-ratio >= 0.10, table-guarded, dilution-fixed). P1 proves the FONT
   channel is insufficient; the signal needs a font-independent channel, with
   the Workstream B negative tests as the over-trigger contract.
3. **Section 5.4 spec** (PLAN_EXTRACTION_FIDELITY_V1, rev. 4): flagged page ->
   ONE bounded specialist re-extraction -> ship flagged if still failing;
   QA_WARN gate consumer for flagged/ladder-served ratios; fleet observability
   aggregates. Spec'd, unbuilt, no-human-in-the-loop. Phase 2 builds it as
   written - this plan adds only the R3-derived page flag as a trigger source.
4. **Document diagnostics** (`DocumentDiagnosticEngine`) already classify
   DIGITAL/SCANNED; per-PAGE text-layer presence is computable with the same
   PyMuPDF machinery as the scan above.

## 4. Design

### 4.1 Page-level signal: `text_native_code` (new, font-independent)

A born-digital page is `text_native_code` when it has a real text layer
(>~100 chars) AND a code-shaped region detected by font-INDEPENDENT channels:
line-leading-whitespace structure (stable indent ladders), code-token density
(keywords/operators/brackets), and/or short-line + ragged-right blocks. The
existing monospace channel remains as a third vote. Exact features and
thresholds are a Phase 1 calibration deliverable, gated by the Workstream B
negative-test contract: incidental shell commands, sparse fenced snippets,
non-code magazines, and encoding corruption alone MUST NOT trigger (those
tests are frozen requirements; if one fails, fix the signal, never the test).
The feature set above is CANDIDATE only: thresholds freeze only AFTER Phase
0(e) measures the new font-independent channels against those fixtures (review
sequencing caveat - 0(e) is the only thing validating the over-trigger
contract for the new channels).

### 4.2 Lane decision for flagged born-digital pages - two candidate mechanisms

- **Mechanism A - route-level deterministic lane.** `text_native_code` pages
  bypass MinerU/Qwen entirely: extract the full page from the text layer
  (y-sorted lines, x-position indentation - the same reconstruction the
  recovery function already implements, applied page-wide), emit as TEXT/CODE
  UIR elements. Pros: deterministic, zero inference cost, immune to bbox
  mapping. Cons: loses MinerU structure (headings/tables) on those pages;
  acceptable only if flagged pages are overwhelmingly code+prose (Phase 0
  measures this).
- **Mechanism B - region-level patch.** Keep the engine output; for every code
  chunk on a text-layer page, replace its content from the text-layer clip
  (generalize the existing recovery from "flat chunks only" to "all code chunks
  on text-native pages", and fix whatever currently stops it from firing).
  Pros: surgical, keeps MinerU structure. Cons: inherits engine bbox fidelity;
  if Phase 0 shows recovery fails BECAUSE hybrid-path chunks lack usable
  bbox/page metadata, B is dead on arrival and A wins by default.

**Phase 0 evidence (Section 1.1) makes B the right FIRST SPIKE** - not a
decided winner (review finding 3): bbox/page coverage is 100% on all code
chunks in all 4 books and the existing recovery is dead only at its trigger
gate (the modality seam), so B is largely a repair + generalization of in-tree
code; and B patches code from BOTH lanes - required now that the Qwen lane
itself measures 0.71-0.78 on real books. But coverage is PRESENCE, not
reconstruction FIDELITY: the indent inference assumes a near-monospace
char-width grid (median span width, `batch_processor.py` ~5847), unvalidated on
exactly the P1 proportional-font population, and a mis-registered bbox yields
wrong-but-gate-passing indentation. Therefore the 2-book spike's EXIT CRITERION
is the Section 6 independent fidelity oracle, never the audit gate; named spike
checks: the latent double-indent bug (span text retaining leading spaces while
`indent_chars` is prepended) and the two-column gutter clip (R4). Mechanism A
stays the LIVE fallback. A-vs-B escalates to a **USER-DECISION** if the spike
contradicts this evidence (Section 8).

### 4.3 P3 raster pages: build Section 5.4 as specified

R3-style conversion-time risk scoring flags a raster code page ->
one bounded Qwen re-extraction of that page -> if still failing the risk bar,
ship flagged + chunked + indexed but never primary-equivalent; QA consumer
counts flagged + ladder-served pages into QA_WARN above a calibrated bound;
doc-level aggregates in the smoke/QA summary. No loops, no human review step.

## 5. Phases

- **Phase 0 - Attribution + instrument validation (offline, no servers).
  (a)-(f) DONE 2026-06-12, results in Section 1.1.** (e) found C2 (code-keyword
  density) is the required discriminator, C1 confirming-only, C3 dropped - the
  over-trigger contract is satisfiable, 4.1 thresholds unfrozen for Phase 1. (f)
  found 13/20 missing pages blank, 6 Devlin silent PROSE drops (separate register
  item, not code), 1 Chaubal CODE drop (F1-addressable). The narrow modality-seam
  trigger-gate fix shipped (`c95950b`); the Mechanism B generalization is Phase 1.
  **Exit gate: every build decision below cites Phase 0 numbers** - 4.2 does.
- **Phase 1 - Deterministic lane (P1/P2).** Build the chosen mechanism (B
  spike first, A live fallback - 4.2), calibrate the signal after 0(e), unit
  tests + negative tests, then re-run the 4 born-digital code books end-to-end.
  Target: the CODE gate green on all 4 with the gate UNCHANGED **AND the
  Section 6 independent oracle passing** - the audit gate alone is a
  nesting-presence check the repair satisfies by construction (review finding
  1), so it is never the sole pass criterion.
- **Phase 2 - 5.4 loop (P3).** Build the three spec'd consumers; re-run the
  C++ manual. Target: code fidelity at or above the bar via bounded
  re-extracts, OR shipped-flagged with the QA_WARN consumer live (the spec'd
  honest-failure path) - pre-registered as acceptable either way, since 5.4's
  contract is honesty, not magic.
- **Phase 3 - Acceptance + reconciliation (re-scoped per review finding 2).**
  Ingestion requires WHOLE-DOC `QA_PASS*`, which the code axis alone does not
  deliver (Devlin failed on code AND 8 missing pages). Phase 3 therefore
  bundles three prerequisites before any ingest: (i) the code fix (Phases 1-2),
  (ii) the Phase 0(f) MISSING_PAGES fix-or-attribution, (iii) a VLM-ENABLED
  production run (clears the no-VLM IMAGE class, including Adedeji). Then:
  pre-registered rule (Section 6), crucible + smoke regression, re-run the
  failed + ~12 unprocessed docs through the shipping CLI, ingest on whole-doc
  pass - closing the Phase 5 reconciliation debt.

Each phase is independently shippable; a Phase 1 pass alone turns the code
AXIS green on the 4 books - it does not by itself ingest them (review finding
2; Phase 3 bundles the remaining modes).

## 6. Pre-registered acceptance (written before any build)

- **Primary:** on the 4 born-digital code books (Devlin, Jungjun, Bourne,
  Chaubal - Section 1.1): code_indentation_fidelity >= 0.90 (the existing
  gate, unchanged) and the CODE gate passing; per-book, not averaged. (Full
  `QA_PASS*` additionally requires MISSING_PAGES - Phase 0(f), IN scope - and
  the no-VLM IMAGE mode - the Phase 3 VLM-enabled run. The code gate alone
  does not ingest a book; Phase 3 is scoped accordingly.) On the C++ manual:
  `QA_PASS*` on code OR flagged-shipped with the 5.4 QA_WARN consumer
  demonstrably counting it. Adedeji re-tests under a VLM-enabled run
  (image-gate class, not code).
- **Independent fidelity oracle (review finding 1 - the audit gate's per-chunk
  `indentation_ok` is a nesting-PRESENCE check, and the repair only commits
  when it has manufactured indentation, so gate-green is near-tautological for
  repaired chunks):** (i) on repair-touched, judgeable, Python-shaped chunks:
  `ast.parse` success rate >= 0.85 per book AND strictly above that book's
  pre-repair rate (parse status is already computed at repair time); (ii) a
  FIXED 10-page-per-book side-by-side artifact set (raw text-layer lines vs
  recovered chunk content) captured in the Phase 1 report - the Phase 0A
  artifact-capture precedent: the numbers say whether it passed, the artifacts
  say HOW. The oracle, not the gate, is the B-spike exit criterion (4.2).
- **No-regression axis 1 (crucible):** the 16-doc crucible re-run shows no doc
  losing its Phase 4 shadow verdict; FluentPython specifically must not
  regress (it currently passes - the new signal must not reroute it
  destructively).
- **No-regression axis 2 (fidelity):** OmniDocBench fixed-set text-ED/TEDS vs
  the recorded hybrid baseline within noise. Baseline provenance PINNED (review
  caveat): 0.2212 text-ED / 0.7933 TEDS = the MinerU+Qwen hybrid on the Phase 1
  bake-off's 158-page fixed paired set (2026-06-11; recorded in the DECISIONS
  Phase 4 entry as the regression reference). The comparison MUST re-run that
  SAME fixed set with the same hybrid config. Never compare against the
  full-755-page EN corpus reference (0.301/0.563) - different page set AND
  different engine (the legacy OCR-enabled offline route; see the Phase 1
  outcome provenance correction). Additionally: the deterministic lane must not
  fire on the benchmark's image-only pages at all (no text layer exists; a fire
  there is a bug).
- **Negative contract:** all Workstream B negative tests green, unmodified.
- **MISSING_PAGES:** the Devlin-class missing-page failures must be attributed
  in Phase 0 and either fixed by the lane (if they are flag-eligible pages) or
  registered as a separate issue - acceptance does not silently absorb them.
- These margins are pre-registered as of rev. 1; changing them after
  verdict-eligible data exists requires a recorded USER decision.

## 7. Risk / issue register

- R1: text-layer artifacts (ligatures, soft hyphens, missing spaces in some
  generators) make "deterministic" not "perfect" - mitigation: Phase 0 censuses
  artifact rates on the 5 books; the lane carries the same R3 scoring so a bad
  text layer still gets flagged, not trusted blindly.
- R2: signal over-trigger on prose with indent structure (poetry, transcripts,
  nested lists) - mitigation: the frozen negative fixtures + a new
  prose-with-indents fixture; over-trigger costs fidelity (A) or nothing (B),
  which also informs the A/B choice.
- R3 (sharpened by review finding 3): bbox PRESENCE (measured 100%) is not
  reconstruction FIDELITY. The indent inference assumes a near-monospace
  char-width grid (median span width) - unvalidated on exactly the P1
  proportional-font population - and a mis-registered or page-wide bbox yields
  wrong-but-gate-passing indentation. Mitigation: the Section 6 oracle is the
  spike exit; the double-indent latent bug and char_width behavior on
  proportional fonts are named Phase 1 spike checks.
- R4: reading order on mechanism A pages (y-sort vs columns) - reuse the
  post-Docling y-sort heuristics; two-column code books exist. For B
  specifically (review caveat): a code-region bbox spanning the gutter ingests
  the neighbour column - add a column-aware clip guard to the spike checklist.
- R5: acceptance-set contamination - the 4 books inform the design AND gate
  acceptance, and (review caveat) contamination x a weak presence-oracle is
  close to fitting-to-test. Mitigation: the independent oracle (Section 6)
  covers the repair-fidelity side, the negative fixtures cover the trigger
  side, and the no-regression axes are independent corpora - though neither of
  those corpora measures code-repair fidelity, which is why the oracle is
  non-negotiable. Accept the residual explicitly (these books ARE the
  production backlog).
- R6: cost - mechanism A REDUCES inference cost (deterministic pages skip
  VLM/MinerU); 5.4 re-extracts add bounded Qwen calls on raster code pages
  only. A cost line per book goes in the Phase 3 report.

## 8. USER-DECISION-REQUIRED (open)

1. Mechanism A vs B if Phase 0 evidence does not make it obvious (Section 4.2).
2. Earthship disposition if attributed non-code: separate scanned-prose issue
   (recommended) or pulled into this plan's scope.
3. Phase 2 cost posture: bounded re-extraction budget per doc (pages or
   wall-clock cap) before a doc ships flagged.
4. Whether Phase 3 reconciliation auto-ingests on `QA_PASS*` or queues for a
   user-reviewed batch (the no-human-in-the-loop default says auto-ingest;
   flagging because it writes production collections).

## 9. Layer-0 edits this plan authorizes (apply only after Phase 3 passes)

- Charter section 3: add the deterministic text-native lane to the as-built
  flow + the page-signal table; section 4: 5.4 consumers go SHIPPED.
- QUALITY_GATES: the flagged/ladder-served QA_WARN consumer semantics.
- DECISIONS: ratification entry with the Phase 0 attribution table and the
  acceptance evidence.
- Issue register: F1 closed; Earthship/scanned-prose item opened if decided.
