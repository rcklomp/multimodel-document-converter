# PLAN_FIDELITY_ORACLE_FIRST_V1 - Per-class fidelity oracles before per-class heuristics

Status: APPROVED FOR EXECUTION (2026-06-13, user directive: "oracles first, then decide").
Owner: extraction + QA.
Layer: 2 (execution plan; authorizes no Layer-0 edit until a class is measured).
Relationship to existing plans: this is the missing FIRST HALF of
`PLAN_EXTRACTION_FIDELITY_V1`. That plan correctly diagnosed that acceptance is
presence-not-fidelity (its X2, I7) and built the OmniDocBench outcome gate - but
that gate is code-free English only, so the classes that actually fail (code,
multilingual technical, scans, forms, REPL) still have no ground-truth oracle and
its Phase 1 bake-off came back "INCONCLUSIVE by construction (identical on a
code-free benchmark)". This plan closes that coverage hole. It does not replace
PLAN_EXTRACTION_FIDELITY_V1; it unblocks its Phase 1 verdict for the failing classes.

---

## 0. Root cause (why all document types have struggled for six weeks)

The forensic read of all 21 handovers (2026-06-03 .. 2026-06-13) shows ONE pattern,
not a run of unrelated bugs: run a soak, get green gates, a manual audit or a harder
corpus then reveals a NEW class of defect the gates never saw, patch that symptom
with a heuristic, ship when smoke passes, repeat. 17+ distinct per-class failure
modes, ~14 heuristic patches vs ~3 structural changes, and at least three documented
"gates passed while content was wrong" cases (tables at 0% markdown inside a "10/15
pass"; the dpi200 baseline silently looping; the silent-ladder stamping
`engine=mineru_qwen_hybrid` on docling output).

**The binding root cause:** acceptance and engine-selection signals do not measure
FIDELITY on the failing classes. The pipeline is accepted on presence/structure
proxies (`QA_PASS`, `GATE_PASS`, `chars > 0`, `ast.parse`) that are provably blind to
the failure modes that matter (omission, stripped/wrong indentation, flattened
tables, reading-order, token corruption). The one true fidelity oracle that was
finally built (OmniDocBench) covers only code-free English prose. So every decision
about code books, German/Dutch technical manuals, scans, forms, and REPL transcripts
is made on blind proxies and contested homegrown floors. That is why the
English-prose classes converged and the rest are in an endless patch loop:

- **Code never converges.** Chaubal sits at 0.828 against a homegrown `ast.parse`
  >= 0.85 floor that is itself a proxy. No authoritative measure says whether the
  extracted code matches the real code, so the lane is tuned by guesswork
  (this week's chunker-contiguity fix was built, correct, and fired 0 times - aimed
  by a since-falsified diagnosis).
- **The bake-off could not end the guessing.** Pipeline-vs-hybrid was identical on a
  code-free benchmark; the instrument built to make the engine decision is blind to
  the dominant failing class.
- **Clean soaks are false-green** (no fidelity measure), so fixes are symptom-patches
  (debugging blind), and the hybrid runs both engines because there was no per-class
  measure to choose between them.

Secondary, compounding: reliability is wired to the most fragile components (M5 /
MinerU servers) and the fail-closed ladder degrades silently with provenance that can
lie when a dependency is absent.

**The fix in one sentence:** give every failing class a ground-truth fidelity oracle
FIRST, freeze new per-class extraction heuristics until the class can be measured,
then decide engine/routing on measured evidence.

---

## 1. The unlock: code books ship their code

The most painful, least-measurable class has AUTHORITATIVE ground truth available:
the author's published source repository. Diffing extracted code against the real
source measures exact fidelity (indentation, identifiers, everything), not a parse
proxy.

Confirmed in-corpus and fetchable this session:
- `data/technical_manual/Fluent Python Luciano Ramalho 2015.pdf` ->
  `github.com/fluentpython/example-code` (1st ed, archived, 343 .py files, 98% Python).
  Exact edition match. GitHub is reachable from this host; `git clone --depth 1` works.

`scripts/code_repo_oracle.py` (shipped this session) implements it. For each
judgeable code chunk in an extraction JSONL it diffs every significant code line
against the indexed repo source and reports:

- `verbatim_fidelity`   - line present in ground truth WITH its leading indentation
- `deindented_fidelity` - line present after both sides are left-stripped
- `indentation_gap`     = deindented - verbatim (content right, indentation wrong:
  the silent fidelity killer ast.parse and text-ED cannot see)
- `content_loss`        = 1 - deindented (line absent even ignoring indentation:
  OCR/engine corruption, or genuine book-vs-repo divergence)

REPL/console transcript chunks are bucketed and excluded from the source-diff
denominator (they are not in .py files; counting them as loss would be dishonest).

### 1.1 Instrument validation (Section 7.3 seeded-fault, run 2026-06-13)

On `output/wpa/FluentPython/ingestion.jsonl` (hybrid) vs a copy with indentation
stripped from every code chunk:

| signal | original (hybrid) | seeded fault (indent stripped) | moved? |
|---|--:|--:|---|
| `ast.parse` rate (incumbent f1_oracle) | 1.000 (3/3) | 0.000 (0/3) | yes, but only over 3 of 22 code chunks |
| repo-oracle `verbatim_fidelity` | 0.531 | 0.406 | yes |
| repo-oracle `deindented_fidelity` | 0.625 | 0.625 | invariant (correct: content unchanged) |
| repo-oracle `indentation_gap` | 0.094 | 0.219 | yes (+0.125) |
| repo-oracle `content_loss` | 0.375 | 0.375 | invariant (correct attribution) |

Reading: the repo oracle moves on the seeded fault and attributes it cleanly to
indentation (gap up, content_loss flat) - a passing sensitivity test. `ast.parse`
catches a TOTAL strip but is binary, scores only the "judgeable multi-line Python"
sub-population (3 of 22 chunks here), exempts all flat/REPL code, and cannot see
content substitution that stays syntactically valid. On the original extraction it
reports 1.000 (perfect) on an extraction the ground truth scores far below that.

Honest limitation (calibration): absolute `content_loss` mixes extraction error with
legitimate book-vs-repo divergence (callout markers like circled glyphs appended to
lines, modified inline examples, partial snippets). So the CALIBRATION-FREE signals
are (a) `indentation_gap` (book annotations do not change indentation) and (b)
ENGINE-COMPARATIVE deltas on the same book/pages. Absolute fidelity floors require
the per-book divergence baseline in Phase 1.

### 1.2 Phase 1 result - Fluent Python code-dense slice, all four engines (2026-06-13)

Densest 40-page code window (p286-325, Vector2d/Vector chapters) extracted through
every candidate engine via the relay prod env (`--vision-provider none`), each scored
against the repo with callout normalization (`--strip-callouts`; publisher `# <n>`
markers are book-vs-repo divergence, common-mode to all arms, so they do not affect
the ranking). Harness: `scripts/p1_code_oracle_bakeoff.sh`. Outputs:
`output/p1_code_oracle/`.

| engine | verbatim (content+indent) | deindented (content) | indent_gap | content_loss | code lines |
|---|--:|--:|--:|--:|--:|
| docling_fast | **0.015** | 0.328 | 0.313 | 0.672 | 67 |
| mineru_only | 0.147 | 0.629 | 0.482 | 0.371 | 170 |
| vlm_qwen_only | 0.343 | 0.750 | 0.407 | 0.250 | 236 |
| **hybrid_default** | **0.371** | **0.768** | 0.397 | 0.232 | 237 |

Findings (measured, first time):
1. **Engine ranking on code: hybrid > qwen >> mineru >>> docling.** The production
   hybrid is the measured code winner; this validates the Qwen-for-code lane on ground
   truth (the slice is code-dense, so the hybrid routes to Qwen).
2. **The dominant residual code defect is INDENTATION, not content.** Hybrid recovers
   77% of code-line CONTENT but only 37% of lines correct WITH indentation (0.40 gap).
   For Python that is semantically fatal, and it is precisely what `ast.parse` (the old
   oracle) and text-ED both score as fine. The project optimized blind to it.
3. **docling is catastrophic for code** (verbatim 0.015, loses 2/3 of content outright,
   surfaces 1/3 the code lines). docling is the offline floor, the fail-closed tier-2,
   AND the Phase 0B interim default. A code page that silently ladders to docling is
   DESTROYED, not degraded -> this upgrades Phase 5 "kill the silent ladder" from
   hygiene to data-integrity (a laddered code page must hard-fail, not advise).

Calibration honesty: even after callout normalization, `content_loss` (0.232 on the
winner) still mixes three things the `--examples` dump confirmed - real line-merge
corruption, PROSE misclassified into code chunks (run-on paragraphs bucketed as code),
and legitimately-non-Python examples (Jython/Java/shell in the book, absent from the
repo). So the ABSOLUTE verbatim/content numbers are FLOORS on true fidelity; the
engine-comparative deltas are exact. NEW register item surfaced by this run:
prose-into-code-chunk misclassification (a code/prose boundary defect, distinct from
indentation and from the F1 code lane).

Code-fidelity floor decision (replaces the contested `ast.parse` 0.85): use the
repo-diff metrics, ADVISORY first (AGENT-GATE-PROGRESSION) - (a) `deindented_fidelity`
regression-vs-best-arm-baseline, (b) `indentation_gap` as the primary defect signal. A
hard ABSOLUTE floor is deferred until the prose-into-code contamination is removed
(else the floor punishes a code/prose defect as if it were code infidelity).

---

## 2. Per-class oracle coverage map (the work surface)

| class | example docs | ground-truth source | status |
|---|---|---|---|
| code (Python, repo'd) | Fluent Python | author repo `fluentpython/example-code`, line-diff | **LIVE + measured (1.2)** |
| code (no clean repo) | Python Distilled (repo is errata-only, no source), Programming ArcGIS (2nd-ed repo URL 404), Chaubal (PyTorch), Jungjun (MEAP), Hao, Raieli, Eliasz Zephyr-C | hand-label 5-10 dense code pages per book | Phase 2 (verified 2026-06-13: only Fluent Python has a clean source repo; `gh` unavailable for deeper search) |
| multilingual technical | Grundlagen, Handbuch (DE), Bevestigingsmiddelen (NL) | hand-label 5-10 pages per language | Phase 2 |
| scanned / forms | 0013, scanned_degraded | hand-label transcription + field set | Phase 2 |
| REPL / notebook transcript | Chaubal, Jungjun | structural oracle (input/output cell pairing intact) | Phase 2 |
| tables | spreadsheets, data_spreadsheet | OmniDocBench TEDS (have) + internal hand-label | partial (OmniDocBench EN only) |
| prose EN | OmniDocBench | OmniDocBench text-ED (have) | LIVE |

OmniDocBench covers only the last two rows partially. Everything above the line is
the coverage hole this plan fills.

---

## 3. Phasing (each phase mechanically checkable)

- **Phase 0 - FREEZE + instrument (this session).** DONE: root cause recorded;
  `scripts/code_repo_oracle.py` shipped + seeded-fault-validated on Fluent Python.
  FREEZE: no new per-class extraction heuristic ships for a class until that class
  has an oracle. (The existing chunker-contiguity fix stays; it is net-positive and
  non-regressing. No NEW code-lane heuristic until the code oracle gives a real
  book-level number.)
- **Phase 1 - Code class, fully measured.** Clone the verified author repos; run a
  FULL-BOOK extraction of each repo'd code book through each candidate engine
  (MinerU-only, hybrid, docling-fast, Qwen-only) via the relay prod env; score every
  arm with `code_repo_oracle.py`. DoD: a per-book, per-engine table of
  `verbatim_fidelity` / `indentation_gap` / `content_loss` with the book-vs-repo
  divergence baseline (best-arm) recorded, so an ABSOLUTE code-fidelity floor can be
  set from evidence instead of the contested 0.85 ast.parse proxy. This is the input
  PLAN_EXTRACTION_FIDELITY_V1 Phase 1 needed and lacked.
- **Phase 2 - No-repo classes.** Build the hand-labeled mini ground-truth sets
  (5-10 pages per class: no-repo code, DE/NL technical, scans/forms, REPL structural).
  Small + paired is decisive (Section 7.2 of PLAN_EXTRACTION_FIDELITY_V1). DoD: a
  fidelity scorer per class + a recorded baseline per engine.
- **Phase 3 - Wire as regression gates.** Add the per-class oracles as ADVISORY
  regression gates (AGENT-GATE-PROGRESSION), promoted to hard per class once stable.
  DoD: `qa_full_conversion.py` / smoke surface per-class fidelity deltas; frozen
  fixtures; suite green; `SMOKE_PRODUCTION_PASS`.
- **Phase 4 - Decide engine/routing on measured evidence.** Re-run the
  PLAN_EXTRACTION_FIDELITY_V1 Phase 1 bake-off, now verdict-ELIGIBLE on the failing
  classes. Cut/keep specialist lanes per the measured per-class winner; cap at the top
  2 lanes by loss magnitude. DoD: the per-class routing table; default named by
  evidence; Layer-0 edits per PLAN_EXTRACTION_FIDELITY_V1 Section 8.
- **Phase 5 - Close the operational lie (parallel, cheap).** Hard preflight that
  fails closed if `mineru-vl-utils` is absent (no silent ladder); surface
  `extraction_degraded_pages` in the gate summary so a laddered run cannot read as
  QA_PASS. DoD: tests + smoke.

---

## 4. Issue / risk register

- I1: book-vs-repo divergence inflates `content_loss` -> use `indentation_gap` +
  engine-comparative deltas as the calibration-free signals; set absolute floors only
  after the Phase 1 best-arm divergence baseline.
- I2: not every code book has a clean repo (Chaubal/Jungjun unverified) -> those drop
  to the Phase 2 hand-label lane; do not block the repo'd books on them.
- I3: full-book extractions need M5/GX10 -> use the relay prod env
  (`scripts/phase5_relay.py`); if a server is down, Phase 1 is a DRY RUN with no
  verdict authority (PLAN_EXTRACTION_FIDELITY_V1 Section 7.2 health guard applies).
- I4: line-set membership over-counts trivial lines -> `_MIN_SIG_LEN` filter (>=4
  stripped chars); comparative use is common-mode robust to residual inflation.
- I5: the FREEZE could be read as "stop all extraction work" -> it is scoped to NEW
  per-class heuristics on UNMEASURED classes; reliability/ops fixes (Phase 5) and
  measured-class work proceed.

---

## 5. Acceptance (of this workstream)

- Every failing class has a fidelity oracle with a recorded per-engine baseline
  before any new per-class extraction heuristic ships for it.
- The code-fidelity floor is set from repo-diff evidence, replacing the contested
  ast.parse 0.85 proxy.
- The engine/routing default is named by measured per-class evidence (this unblocks
  PLAN_EXTRACTION_FIDELITY_V1 Phase 1 for the failing classes).
- No locked invariant violated (extraction boundary, zero-Docling batch_processor,
  AST firewall, ElementType=3/Modality=5, bbox [0,1000], QA-CHECK-01 0.10).
