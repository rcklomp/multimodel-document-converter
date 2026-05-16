# Plan: v2.11 — Embedder shootout, validated-cloud, and the rc1/rc2 non-goal closure

**Status:** **Draft v0.2** — post-soak, post-Phase-0. Soak ran
2026-05-16 (Format 98.3%, Recall@1 2.1%, Recall@5 doc 54.2%, Relevance
5.9%, Faithfulness 4.7%). Phase 0 closed same day with zero v2.10.x
patches needed; `v2.10.0` final shipped. The soak's quantitative
finding sharpens Phase 1: the challenger embedder must beat
**specific** numeric floors on the embedder-attributable axes (see
Phase 1 §"Soak baseline to beat"). Promotion to Draft v1.0 happens
when the user signs off on starting Phase 1 (pulling the Qwen3
embedder + spinning up the second Qdrant collection).
**Predecessor:** [`docs/PLAN_V2.10.md`](PLAN_V2.10.md) — Phases 1-8
closed, **`v2.10.0` SHIPPED 2026-05-16** (annotated tag on commit
`db6527c`, public on GitHub). The RC tag `v2.10.0-rc1` (`82c3639`)
is also retained on GitHub.
**Owner:** ingestion pipeline.

---

## 1. Why this plan exists

### Thesis

v2.10 closed every named root-cause class from the rc1 deferral set
and produced a clean 34 PASS / 0 WARN / 0 FAIL corpus baseline. What
it did **not** close are the three structural questions about whether
the v2.10 baseline is the *right* baseline:

1. **Is `llava` actually the right embedder for this corpus?** The
   v2.10 soak quantified the weakness: Recall@1 (gold chunk in top-1)
   is **2.1%** (11/518); Recall@5 doc is **54.2%**; the qwen-max judge
   marks Relevance at 5.9% and Faithfulness at 4.7%. The pattern in
   the soak's weakest-15 list is unambiguous: queries about LLM /
   RAG / agent topics get routed to Devlin's book regardless of true
   topic; Dutch queries get dominated by whichever Dutch doc is bigger;
   "F-35 Lightning II" lands on cloud-GenAI content. **The chunker is
   right** (Format 98.3%); the embedder is the bottleneck. See
   `docs/QUALITY_SNAPSHOT_2026-05-16_v2.10_soak.md` for the raw data
   and `docs/DECISIONS.md` "v2.10 chunker-quality ceiling" for the
   architectural framing.
2. **Is the validation tied to one developer's machine?** v2.10 is
   `validated-local`. We have no proof that a clean checkout on a
   different machine reproduces the strict gate, the smoke, the
   regression test, and the soak. Without that, "validated-local" is
   doing more work than it should.
3. **Are the rc1 carry-forward non-goals still right to defer?** Five
   items (NuMarkdown-8B reachability, remote CodeFormulaV2, broader
   UIR refactor, HybridChunker per-item token guard, magazine
   rendered-region-crop) were explicit non-goals through v2.7-v2.10.
   Each carry needs an explicit yes/no for v2.11 — formal further-defer
   or in-scope.

v2.11 answers all three with data, not opinion.

### Where v2.10 ended

- **`v2.10.0` SHIPPED 2026-05-16** (commit `db6527c`, rc1 commit + soak report). RC tag `v2.10.0-rc1` (`82c3639`) retained on GitHub. The v2.10.0 annotated tag explicitly frames the release as a chunker baseline.
- Strict gate: 34 PASS / 0 WARN / 0 FAIL across the 34-doc canonical corpus (16 `QA_PASS` + 18 `QA_PASS_WITH_ADVISORIES`).
- Qdrant `mmrag_v2_8` rebuilt: `status: green`, `points_count: 30,454`, `indexed_vectors_count: 30,192`, 4096-dim cosine via Ollama `llava`.
- Test suite: 975 passed, 14 skipped, 0 failed.
- Smoke multiprofile: 11/11 GATE_PASS + 11/11 UNIVERSAL_PASS.
- Retrieval-regression baseline captured at `tests/fixtures/retrieval_regression_v2_10.json` (20 queries × top-5; pinned by `tests/test_retrieval_regression_v2_10.py`).
- Synthetic-soak harness shipped + run (`scripts/synthetic_soak.py`, report at `docs/QUALITY_SNAPSHOT_2026-05-16_v2.10_soak.md`).

### What v2.11 starts from

The v2.10 cycle delivered:

- A clean chunker contract (Format 98.3% certified by the qwen-max judge).
- Durable infrastructure for measuring future change: retrieval-regression fingerprint + synthetic-soak harness, both runnable against any future Qdrant collection.
- One concrete blocker that v2.10 couldn't address inside its scope: the embedder. Recall@1 2.1% / Faithfulness 4.7% is documented as the v2.11 Phase 1 starting point.

The harnesses are not single-use. They get re-run against every embedder candidate, every rebuild, and every annotated tag. They're the measurement substrate v2.11 stands on.

### Carry-forward register (Draft v0.2 — post-soak)

| # | Class | Source | v2.11 status | Notes |
|---|---|---|---|---|
| 1 | **Embedder choice** (llava → ?) | v2.10 soak: Recall@1 2.1%, Faith 4.7% | **`in-scope`** | Phase 1 below. Challenger: Qwen3-Embedding-4B; BGE-M3 as fallback if Qwen3 not pullable via Ollama. |
| 2 | **validated-cloud / CI** | rc1 §"validated-local only" | **`in-scope`** | Phase 2 below. |
| 3 | NuMarkdown-8B local VLM | v2.7 / v2.8 / v2.9 / v2.10 non-goal | **`investigate-then-decide`** | Phase 3a — check reachability, decide formal-defer-or-act. |
| 4 | Remote CodeFormulaV2 inference | v2.10 non-goal (Docling 2.86 doesn't expose option) | **`upstream-blocked, defer to v2.12`** | One-line confirmation of Docling status. |
| 5 | Broader UIR refactor | v2.7+ non-goal | **`scope decision needed`** | Big surgery; pick "small carve-out" vs "defer". |
| 6 | HybridChunker per-item token guard | v2.10 non-goal, upstream-blocked | **`upstream-blocked, defer`** | Confirm Docling status. |
| 7 | Magazine rendered-region-crop | v2.10 non-goal | **`defer`** | Magazines pass at current quality bar; not a v2.11 blocker. |
| 8 | EPUB engine rewrite (parallel `EpubEngine`) | v2.10 §5 (v2.11+ scope) | **`defer to v2.12+`** | Soak data shows EPUB recall is embedder-bound (Phase 1), not lane-bound. Marker workaround is durable. Phase 4 below has the rationale. |
| 9 | v2.10.x defect candidates from soak | Synthetic-soak 2026-05-16 | **`empty (closed)`** | Soak weakest-15 was entirely embedder-attributable wrong-doc retrievals, not chunk-shape defects. Zero v2.10.x patches needed. Phase 0 closed same day. |
| 10 | **99.9% Format chunker target** | v2.10 soak (Format 98.3% ceiling) | **`out of scope`** | See `docs/DECISIONS.md` "v2.10 chunker-quality ceiling — 99.9% Format not chased". |

---

## 2. Goals & Non-Goals

### Goals (measurable)

1. **Embedder decision recorded with evidence.** A side-by-side soak
   comparison of `llava` vs `Qwen3-Embedding-4B` (and optionally a
   third candidate). Decision in `docs/DECISIONS.md` with the metric
   delta. If the challenger wins, ship as v2.11.0; if not, llava
   stays and v2.11 is a smaller release.
2. **At least one validated-cloud checkpoint** — the strict gate +
   smoke + pytest + retrieval-regression all pass against a fresh
   checkout in a fresh Python env on a runner that is NOT the
   developer's daily machine. CI-on-self-hosted-runner is acceptable;
   "fresh conda env locally" is a stepping stone, not the final state.
3. **All v2.10.x defects surfaced by the soak either closed in code
   or formally deferred to v2.12** with explicit rationale in
   `docs/DECISIONS.md`. No silent acceptance.
4. **Every carry-forward non-goal has a recorded decision in
   `docs/DECISIONS.md`** — formal further-defer with rationale, or
   in-scope phase with done-when contract.
5. **Strict gate stays at 34 PASS / 0 WARN / 0 FAIL** corpus-wide
   throughout v2.11. No new advisory codes added without explicit
   sign-off per `docs/QUALITY_GATES.md`.

### Non-Goals (deferred beyond v2.11 unless promoted)

- Scaling beyond the 34-doc canonical corpus. Adding new test
  documents is a separate workstream.
- New chunk-shape fields (schema version stays `2.7.0`).
- New languages beyond what's already in the corpus (EN/NL/DE).
- Replacement of the synthetic-soak LLM judge with a different
  provider. Dashscope `qwen-max` works.

---

## 2b. Cross-phase principles (carried from v2.10 §2b-d)

Unchanged from v2.10:

- **Parallel-site audit before every fix** — find every site the same
  defect could manifest, not just the named one.
- **No QA threshold weakening** — `_ALLOWED_ADVISORY_WARN_CODES` is
  modified only with explicit sign-off + `docs/QUALITY_GATES.md`
  update + a pinned regression test.
- **Status discipline** — `implemented` → `validated-local` →
  (optionally) `validated-cloud` → `complete`. Phase 1 cannot move to
  `validated-local` until the challenger's strict gate AND soak both
  land green AND a corpus rebuild + fingerprint refresh both pass.
- **Doc budget** — extend existing contracts; no new governance docs.
- **Libraries first, custom code last** — the Phase 1 embedder swap
  is one config change in `scripts/ingest_to_qdrant.py` and a
  Qdrant rebuild; not a refactor.

---

## 3. Phases

### Phase 0 — v2.10.0 final ship + soak-driven v2.10.x patches  ✅ CLOSED 2026-05-16

**What this phase was.** The bridge from v2.10 (chunker baseline, validated-local) to v2.11 (embedder swap + validated-cloud). Read the soak, triage the weakest list into code-defect / embedder-weakness / format-defect classes, ship any code-defect patches as v2.10.x, then tag `v2.10.0` final.

**What actually happened.**

1. Soak ran 2026-05-16 ~20:23-20:53 UTC. Report at `docs/QUALITY_SNAPSHOT_2026-05-16_v2.10_soak.md`.
2. Triage of the weakest 15:
   - **Code defects:** 0.
   - **Format / chunk-shape defects:** 0 (Format 98.3% corpus-wide). Minor 1/2 Format scores in the weakest list are subjective "odd spacing" / "minor truncation" judge calls — not actionable as code fixes.
   - **Embedder weakness:** ~15 of 15. Every wrong-doc retrieval in the weakest list is the same pattern (LLM/RAG/agent queries → Devlin; Dutch queries → KI EPUB regardless of true topic; specific-product queries → cloud-GenAI book). Pushed to Phase 1.
3. **Zero v2.10.x patches issued.** `v2.10.0` annotated tag created on `db6527c` (rc1 commit `82c3639` + soak report commit) and pushed to GitHub the same day.
4. The 1.7% Format gap is documented as a known limitation with explicit cost/benefit analysis in `docs/DECISIONS.md` "v2.10 chunker-quality ceiling — 99.9% Format not chased (2026-05-16)". Three revisit triggers recorded; none currently met.

**Closure outputs (durable evidence).**

- v2.10 AFTER snapshot revised with §10 entry recording the SHIPPED promotion.
- DECISIONS.md "v2.10 chunker-quality ceiling" entry authored.
- AGENTS.md / CLAUDE.md / docs/README.md / docs/PROJECT_STATUS.md / root README all swept from "rc1 staged" to "v2.10.0 SHIPPED".
- This plan's status banner promoted from Draft v0.1 → v0.2.

**Takeaways for v2.11.**

The framing v2.10 ended on — "the chunker is right; the embedder is the bottleneck" — is the architectural premise Phase 1 is testing. Phase 1's success bar is therefore not "make retrieval better" (vague) but "beat these specific soak numbers" (see Phase 1 §"Soak baseline to beat").

---

### Phase 1 — Embedder shootout: `llava` vs `Qwen3-Embedding-4B`

**What.** Decide whether to swap the v2.10 embedder. The v2.10 soak quantified llava's weakness on this corpus. The candidate `Qwen3-Embedding-4B` is:

- Multilingual (100+ languages including Dutch + German — directly addresses the soak's Dutch-query dominance pattern).
- 2560-dim cosine — 1.6× smaller index than llava's 4096-dim.
- Same vendor family as the soak judge — same Dashscope ecosystem if the comparison wins.
- Locally hostable via Ollama at ~2.5 GB Q4_0 → leaves headroom on a 16 GB M1 even with llava resident for the VLM enrichment lane.

**Fallback challenger if Qwen3-Embedding-4B is not pullable via Ollama** at start-of-phase: **BGE-M3** (Ollama-pullable, 1024-dim, multilingual, ~400 MB Q4). BGE-M3 won't match Qwen3 ceiling but is a guaranteed-available baseline that still tests the "any modern dedicated text embedder beats a repurposed VLM" hypothesis. If Qwen3 is unavailable, run BGE-M3 as the v2.11 challenger and add Qwen3 as a v2.12 candidate.

We are not changing the VLM enrichment path (which legitimately wants a vision-language model). We are changing only the **text retrieval embedder**. The two roles are decoupled.

#### Soak baseline to beat

The v2.10 soak gives Phase 1 concrete numeric floors. The challenger collection's soak must beat these on at least three of the four embedder-attributable axes without regressing Format:

| Axis | v2.10 baseline (llava) | Phase 1 floor for challenger | Stretch target |
|---|---:|---:|---:|
| **Recall@1** (gold chunk_id is top-1) | 2.1% | **≥ 15%** | ≥ 30% |
| **Recall@5 chunk_id** | 6.8% | **≥ 25%** | ≥ 50% |
| **Recall@5 doc_id** | 54.2% | **≥ 70%** | ≥ 85% |
| **Relevance (judge)** | 5.9% | **≥ 30%** | ≥ 60% |
| **Faithfulness (judge)** | 4.7% | **≥ 25%** | ≥ 55% |
| **Format (judge) — NOT regressed** | 98.3% | **≥ 96%** | ≥ 98% |

Reasoning for the floors: at Recall@1 < 15% the user-visible improvement is too small to justify the migration cost; at Recall@5 doc < 70% the challenger isn't materially better than llava on the most lenient axis. The "stretch" column anchors what a well-tuned modern multilingual embedder typically achieves on a corpus this size; if Qwen3 hits stretch on 3+ axes, the swap is a clear win.

If the challenger fails to clear the floors on at least three of the four embedder axes, llava stays as the v2.11 baseline and Phase 1 closes with a `decision-recorded, no-swap` outcome. The harnesses we built are not wasted — they're permanent measurement infrastructure for the next embedder candidate (v2.12+).

**Schedule note.** Per release-cycle hygiene: this phase runs *after* v2.10.0 ships final (Phase 0, closed 2026-05-16). We did NOT swap mid-rc1; the regression test + soak baseline exist precisely so this swap can be done with rigor instead of in-flight.

**Approach.**
1. `ollama pull qwen3-embedding:4b` (or equivalent tag once published).
2. Spin up a **second** Qdrant collection — call it
   `mmrag_v2_8__qwen3_4b` — so the current `mmrag_v2_8` (llava)
   stays untouched as the comparison baseline.
3. Rebuild the second collection from the same 34 canonical JSONLs
   via `scripts/ingest_to_qdrant.py --model qwen3-embedding:4b --collection mmrag_v2_8__qwen3_4b`.
4. `scripts/retrieval_regression.py --capture` against the new
   collection → produces a *separate* fingerprint file
   `tests/fixtures/retrieval_regression_v2_11_qwen3.json` (do not
   overwrite the v2.10 fixture).
5. `scripts/synthetic_soak.py` with `--collection` pointed at the
   challenger → produces
   `docs/QUALITY_SNAPSHOT_<DATE>_v2.11_soak_qwen3_4b.md`.
6. Side-by-side comparison of the two soak reports + the two
   regression fingerprints.

**Tests (red→green).**
- `tests/test_embedder_shootout_v2_11.py::test_qwen3_collection_built` —
  pin the challenger collection's `points_count` parity with `mmrag_v2_8`
  (modulo ingest filter rate).
- `tests/test_embedder_shootout_v2_11.py::test_soak_metric_delta_recorded` —
  pin the per-axis delta (Recall@1, Recall@5 chunk, Recall@5 doc,
  Relevance, Format, Faithfulness) between the two soak reports so
  re-running the comparison can't silently drift.
- The existing `tests/test_retrieval_regression_v2_10.py` keeps
  passing (it asserts against the llava collection).

**Done when.**
- Both collections green; both fingerprints captured; both soak reports authored.
- A decision row in `docs/DECISIONS.md` titled "v2.11 Embedder Shootout" recording which embedder won on which axes, with the specific numeric deltas against the §"Soak baseline to beat" table, and which embedder ships.
- **If challenger clears the floors on at least 3 of 4 embedder axes (Recall@1, Recall@5 chunk, Recall@5 doc, Relevance/Faithfulness composite) AND Format ≥ 96%:** `scripts/ingest_to_qdrant.py` default flipped to the challenger; v2.10 llava collection retained as a 30-day rollback baseline; v2.11.0 tag is the embedder-swap release; AFTER snapshot records the headline soak deltas.
- **If challenger fails the floors:** decision recorded as `no-swap`. v2.11.0 ships the rest of the cycle's work (validated-cloud, carry-forward dispositions) without the embedder change; the challenger becomes a documented v2.12+ candidate. Harnesses remain in place for the next attempt.

**Risk.** Medium. The actual quality delta is unknown until the soak
reports compare side by side. The architectural risk is low — we have
deterministic chunk_ids, idempotent uuid5 point IDs, and a Qdrant
rebuild path proven by Phase 8. The schedule risk is the wall-time of
the second rebuild (~5-7 hours at the v2.10 rate, faster if Qwen3 is
quicker to embed than llava).

**Cost class.** Reconvert: no (chunks unchanged). Re-enrich: no (VLM
unchanged). Qdrant rebuild + retrieval-regression capture + soak run:
yes, all driven by existing scripts. Effort: ~1-2 days of wall time,
mostly waiting on the rebuild + soak.

---

### Phase 2 — validated-cloud checkpoint

**What.** Prove that the v2.11 validation reproduces on a clean
checkout in a fresh environment that is NOT the developer's daily
machine. This closes the "validated-local only" hole that v2.10 left
open.

**Approach (three options ordered by setup cost).**

**2a (cheapest, ~30 min).** Local fresh-env re-run. `conda env create
--name mmrag-v2-fresh --file environment.yml`; `pip install -e .[dev]`;
re-run pytest + strict-gate-corpus + smoke. Catches dep-version drift
and accidental editable-install dependencies on the developer's env.
Does NOT catch hardware / OS coupling — that's still the same M1.

**2b (medium, ~2-4 h setup + per-run minutes).** GitHub Actions on
self-hosted runner pointed at a local Qdrant cache. Catches everything
2a does plus dep-resolution-on-different-Python-image. Self-hosted
runner can read `data/`, which is the blocker that makes a
GitHub-hosted runner expensive (the 34 source PDFs/EPUBs sum to ~few
hundred MB; either commit them to a private bucket or wire LFS).

**2c (durable, ~4-8 h setup).** GitHub-hosted runner + private data
bucket (R2 / S3). Cleanest validated-cloud signal. Highest setup cost.

**Recommendation.** Start with 2a as a Phase 1 prerequisite (run it
*before* the embedder rebuild so we know the fresh env reproduces the
llava baseline first). Move to 2b in this phase. Defer 2c to v2.12
unless 2b proves insufficient.

**Done when.**
- A passing CI workflow file under `.github/workflows/` (or `.gitea/`,
  if mirroring) that runs on push to `main` + on every annotated tag,
  and the workflow includes pytest + strict-gate-corpus + smoke +
  retrieval-regression + (optionally) a small synthetic-soak
  sub-sample.
- At least one green run captured on the v2.11.0-rc1 commit.
- `docs/PROJECT_STATUS.md` "Active Baseline" section updated to
  reference the CI evidence link, not just the local snapshot.

**Risk.** Medium. The CI YAML is straightforward; the data + secret
management is the friction.

**Cost class.** No reconvert, no rebuild. CI runner minutes only (~10-20
min per run).

---

### Phase 3 — Carry-forward non-goals: explicit decisions

**What.** Each of the five rc1 non-goals gets a one-line decision in
`docs/DECISIONS.md` for v2.11:
- formal further-defer (with named reason that's still true), OR
- in-scope sub-phase here (Phase 3a / 3b / etc.).

**Approach (one row per item).**

| Item | Recommended v2.11 status | Action |
|---|---|---|
| **3a. NuMarkdown-8B local VLM** | Probe reachability; decide. | One probe attempt against the documented endpoint. If reachable: open Phase 3a. If not: formal further-defer to v2.12. |
| **3b. Remote CodeFormulaV2** | Formal-defer to v2.12. | One-line check: does `docling-2.87+` expose `RemoteCodeFormulaOptions`? If yes, in-scope; if no, defer with date-stamp. |
| **3c. Broader UIR refactor** | Carve-out only. | Pick the smallest defensible carve-out (e.g. unify `PdfConversionPlan` ↔ `EpubConversionPlan` into a parent `ConversionPlan`). Anything larger is v2.12+. |
| **3d. HybridChunker per-item token guard** | Formal-defer to v2.12. | Upstream Docling tracking issue; one-line citation update. |
| **3e. Magazine rendered-region-crop** | Formal-defer. | Magazines pass at current quality. Revisit only if soak surfaces a magazine-specific defect. |

**Done when.** Five rows in a new `docs/DECISIONS.md` §"v2.11
Carry-Forward Decisions" subsection, each citing the rationale and
the date.

**Risk.** Low. This is governance work.
**Cost class.** Documentation only unless Phase 3a or 3c opens up.

---

### Phase 4 — EPUB engine question — DEFERRED to v2.12+

**Soak verdict.** The v2.10 soak's per-doc table gave Phase 4 the data it needed to decide:

| Doc | Format | Recall@5 doc | Faithfulness |
|---|---:|---:|---:|
| `KI_En_ChatGPT_Praktische_Gids` | 96.9% | 75.0% | 0.0% |
| `ChatGPT_Praktijk_handboek` | 100% | 12.5% | 0.0% |

Format on both EPUBs is essentially perfect — the marker workaround + synthetic-pagination architecture (Phase 7 of v2.10) produces clean chunks. The Recall@5 doc collapse on ChatGPT (12.5%) is the known KI-EPUB-dominates pattern, which is an **embedder problem** (Recall@5 doc is dominated by chunk-volume; KI EPUB has 4,512 chunks vs ChatGPT's 298, and llava can't disambiguate on content). The Faithfulness 0% is the same llava-weakness story affecting every doc.

**Decision (recorded here, to be transcribed to `docs/DECISIONS.md` at Phase 3 closure):** EPUB engine rewrite stays deferred to **v2.12+**. The marker workaround is durable; the soak proves it's not the bottleneck. Revisit only if:

1. The Phase 1 embedder swap lifts retrieval everywhere *except* on EPUBs (a future soak shows EPUB-specific underperformance vs PDF on the same corpus), OR
2. A new EPUB enters the corpus that breaks the marker pattern (e.g., an EPUB with reflowable layout that defeats the spine-order chapter detection).

**What the parallel `EpubEngine` would do (for v2.12+ reference).** Add `src/mmrag_v2/engines/epub_engine.py` parallel to `engines/docling_adapter.py`; map ebooklib output → `UniversalDocument` directly; replace `processor._epub_to_html` + post-process markers with a direct path through the new engine; reconvert KI + ChatGPT EPUBs; re-run strict gate + retrieval regression + soak; compare. Estimated effort: 1-2 weeks if pursued in isolation; absorbs into the broader UIR refactor (Phase 3c here) if that's pursued.

**Risk of deferral.** Low. The marker workaround landed clean in v2.10 and has zero open defects.
**Cost of deferral.** Zero direct cost; carry-forward register row 8 keeps the work item visible.

---

### Phase N — Re-verification, AFTER snapshot, v2.11.0 tag

**What.** Same shape as v2.10 Phase 8 but smaller corpus delta.

**Approach.**
1. Corpus-wide strict-gate re-verification using the chosen embedder.
2. Smoke 11/11.
3. Full pytest + retrieval-regression + soak.
4. Qdrant green at the chosen collection.
5. AFTER snapshot at `docs/QUALITY_SNAPSHOT_<DATE>_v2.11_after.md`
   following the v2.10 template.
6. Tag `v2.11.0` (no rc-cycle this time if Phase 2 validated-cloud has
   landed and Phase 0/1 outcomes are clean).

**Done when.** All of the above green; tag pushed to GitHub.

---

## 4. Acceptance Gate

The v2.11.0 production tag bar:

- Strict gate corpus: 34 PASS / 0 WARN / 0 FAIL.
- Smoke multiprofile: 11/11.
- Pytest: prior baseline + every red→green test added by Phases 1-4.
- Retrieval regression: PASS against the chosen embedder's baseline.
- Soak: report authored, no v2.11-introduced regressions vs v2.10 soak.
- Validated-cloud: at least one green CI run on a non-developer machine.
- AFTER snapshot tracked.
- All five non-goals: explicit decision recorded in `docs/DECISIONS.md`.

---

## 5. Out of Scope (this draft)

These remain non-goals for v2.11 unless explicitly promoted:

- Replacing Qdrant with another vector DB.
- Replacing Ollama as the local model runtime.
- Adding a UI / web frontend.
- New file-format support beyond the current six (PDF/EPUB/HTML/DOCX/PPTX/XLSX).
- Schema evolution (chunk-shape stays `2.7.0`).
- **Format 99.9% chunker-quality target.** v2.10 landed at Format 98.3% (1018/1036 axis-points on the soak); going from 98.3% → 99.9% would require a UIR refactor + LLM-cleanup post-process (~3 months of focused work) and the 1.7% gap is masked anyway by the embedder weakness (Recall@1 2.1%). See `docs/DECISIONS.md` "v2.10 chunker-quality ceiling — 99.9% Format not chased (2026-05-16)" for the cost/benefit analysis and the three triggers that would revisit the decision.

---

## 6. Decision log (this plan)

| Date | Change |
|---|---|
| 2026-05-16 | Draft v0.1 authored mid-v2.10.0-rc1-soak. Embedder challenger (Qwen3-Embedding-4B) named per release-plan discussion. Five rc1 carry-forward non-goals captured with default-recommended dispositions. Phase 0 explicitly gates Phase 1 on v2.10.0 final ship. |
| 2026-05-16 | **Phase 0 closed same day.** `v2.10.0` SHIPPED on commit `db6527c` (rc1 commit + soak report). Zero v2.10.x patches needed — soak weakest-15 was dominated by embedder-attributable wrong-doc retrievals, not chunk-shape defects. New `docs/DECISIONS.md` entry "v2.10 chunker-quality ceiling — 99.9% Format not chased (2026-05-16)" documents the choice to not pursue Format 99.9% in v2.11/v2.12 and lists three triggers that would revisit. §5 Out of Scope updated with the outbound cross-reference. Draft stays at v0.1 — user has no inspiration to change other parts of the plan yet. |
| 2026-05-16 | **Promoted to Draft v0.2.** Substantive updates: (1) Thesis §1 quantifies the embedder weakness with soak numbers instead of just regression-test anecdotes. (2) "Where v2.10 ended" section rewritten in past tense; "What v2.10.0 final is waiting on" stanza deleted (obsolete). (3) Carry-forward register row 9 (TBD soak findings) replaced with "empty (closed)" outcome; new row 10 added for the 99.9% Format out-of-scope decision. (4) Phase 0 rewritten uniformly past-tense with the actual triage breakdown. (5) Phase 1 gains a "Soak baseline to beat" sub-section with concrete numeric floors (Recall@1 ≥ 15%, Recall@5 doc ≥ 70%, Relevance ≥ 30%, Faithfulness ≥ 25%, Format ≥ 96%) replacing the vague "+5%" criterion; BGE-M3 documented as fallback challenger if Qwen3-Embedding-4B isn't Ollama-pullable. (6) Phase 4 (EPUB engine) flipped from "decision point" to "deferred to v2.12+" with the soak data backing the deferral. (7) §7 Open questions: items 1 and 5 marked resolved; item 6 added as the new Phase 1 start gate. Promotion to Draft v1.0 happens when the user signs off on starting Phase 1. |

---

## 7. Open questions

Status as of Draft v0.2 (2026-05-16). Answered questions are kept for traceability.

1. ~~**Soak findings.**~~ **Resolved 2026-05-16.** Recall@1 2.1%; Format 98.3%; weakest-15 is entirely embedder-attributable wrong-doc retrievals. All scope moves to Phase 1; zero v2.10.x patches; Phase 0 closed same day.
2. **Qwen3-Embedding-4B availability via Ollama.** Open. Confirm at Phase 1 kickoff via `ollama pull qwen3-embedding:4b`. If unavailable: fall back to BGE-M3 (Ollama-pullable, multilingual; documented in Phase 1 §Fallback challenger).
3. **NuMarkdown-8B reachability.** Open. One probe attempt at Phase 3a kickoff; if endpoint still off-network, formal-defer to v2.12.
4. **CI runner topology.** Open. Self-hosted on the same network as Qdrant + Ollama is the recommended path (2b in Phase 2); GitHub-hosted with private data bucket (2c) deferred to v2.12 unless 2b proves insufficient.
5. ~~**EPUB engine.**~~ **Resolved 2026-05-16.** Soak data shows EPUB recall is embedder-bound (KI dominates ChatGPT because llava can't disambiguate, not because the EPUB lane is broken). Phase 4 deferred to v2.12+. The marker workaround stays.
6. **Phase 1 start gate.** Currently waiting on user signoff to: (a) confirm Qwen3-Embedding-4B availability on the local Ollama, (b) approve the second-collection rebuild (~5-7 h wall time), (c) approve the per-Phase-1 budget for re-running the soak via Dashscope (~$2-3 in qwen-max calls). No technical blocker; awaiting decision.

---

**END OF DRAFT v0.2.** Phase 0 is closed; v2.10.0 has shipped; the soak gave us concrete numeric floors for Phase 1. Next checkpoint: user signoff on Phase 1 start (Qwen3-Embedding-4B pull + second-collection rebuild + comparison soak). Promotion to **Draft v1.0** at that signoff. The Phase 2 (validated-cloud) and Phase 3 (carry-forward dispositions) phases are independent of Phase 1's outcome and can run in parallel or interleaved.
