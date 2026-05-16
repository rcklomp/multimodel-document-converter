# Plan: v2.11 — Embedder shootout, validated-cloud, and the rc1/rc2 non-goal closure

**Status:** **Draft v0.1** — pre-soak. The synthetic-soak run started
2026-05-16 will add v2.10.x defect candidates (Phase 0) and may shift
the embedder-shootout success bar (Phase 1). This draft will be promoted
to Draft v0.2 after the soak report lands, and to Draft v1.0 after
v2.10.0 final ships.
**Predecessor:** [`docs/PLAN_V2.10.md`](PLAN_V2.10.md) — Phases 1-8
closed, v2.10.0-rc1 tagged 2026-05-16 on commit `82c3639`, public on
GitHub.
**Owner:** ingestion pipeline.

---

## 1. Why this plan exists

### Thesis

v2.10 closed every named root-cause class from the rc1 deferral set
and produced a clean 34 PASS / 0 WARN / 0 FAIL corpus baseline. What
it did **not** close are the three structural questions about whether
the v2.10 baseline is the *right* baseline:

1. **Is `llava` actually the right embedder for this corpus?** The
   v2.10 retrieval-regression captures (`tests/fixtures/retrieval_regression_v2_10.json`)
   surfaced concrete failure modes: Dutch queries dominated by the
   larger Dutch doc regardless of semantic relevance; "F-35 Lightning II"
   → cloud-GenAI book; specific-product queries returning wrong-domain
   results. These are not bugs in chunking — they're embedder weakness
   on multilingual + domain-discriminative retrieval.
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

### Where the previous cycle left off

- `v2.10.0-rc1` tagged + pushed 2026-05-16 (commit `82c3639`).
- Strict gate: 34 PASS / 0 WARN / 0 FAIL across the 34-doc canonical
  corpus (16 `QA_PASS` + 18 `QA_PASS_WITH_ADVISORIES`).
- Qdrant `mmrag_v2_8` rebuilt: `status: green`, `points_count: 30,454`,
  `indexed_vectors_count: 30,192`, 4096-dim cosine via Ollama `llava`.
- Test suite: 975 passed, 14 skipped, 0 failed.
- Smoke multiprofile: 11/11 GATE_PASS + 11/11 UNIVERSAL_PASS.
- Retrieval-regression baseline captured: 20 queries × top-5 pinned in
  `tests/fixtures/retrieval_regression_v2_10.json`.
- Synthetic-soak harness shipped: `scripts/synthetic_soak.py` + soak
  report writer; first run in flight at draft-time.

What `v2.10.0` final is waiting on:

- Soak report (Phase 0 here).
- Any v2.10.x patches that soak findings justify (Phase 0 here).
- A retrieval-regression baseline that has survived at least one
  rebuild without drift (Phase 1 byproduct here).
- A validated-cloud checkpoint (Phase 2 here).

### Carry-forward register (Draft v0.1 — TBD = "soak will refine")

| # | Class | Source | v2.11 status | Notes |
|---|---|---|---|---|
| 1 | **Embedder choice** (llava → ?) | rc1 retrieval-regression; soak | **`in-scope`** | Phase 1 below. Challenger: Qwen3-Embedding-4B. |
| 2 | **validated-cloud / CI** | rc1 §"validated-local only" | **`in-scope`** | Phase 2 below. |
| 3 | NuMarkdown-8B local VLM | v2.7 / v2.8 / v2.9 / v2.10 non-goal | **`investigate-then-decide`** | Phase 3a — check reachability, decide formal-defer-or-act. |
| 4 | Remote CodeFormulaV2 inference | v2.10 non-goal (Docling 2.86 doesn't expose option) | **`upstream-blocked, defer to v2.12`** | One-line confirmation of Docling status. |
| 5 | Broader UIR refactor | v2.7+ non-goal | **`scope decision needed`** | Big surgery; pick "small carve-out" vs "defer". |
| 6 | HybridChunker per-item token guard | v2.10 non-goal, upstream-blocked | **`upstream-blocked, defer`** | Confirm Docling status. |
| 7 | Magazine rendered-region-crop | v2.10 non-goal | **`defer`** | Magazines pass at current quality bar; not a v2.11 blocker. |
| 8 | EPUB engine rewrite (parallel `EpubEngine`) | v2.10 §5 (v2.11+ scope) | **`scope decision needed`** | Phase 7's synthetic-pagination is a workaround. Real fix is a parallel engine. |
| 9 | v2.10.x defect candidates from soak | Synthetic-soak report (pending) | **`TBD`** | Phase 0. Promoted on a case-by-case basis. |

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

### Phase 0 — v2.10.0 final ship + soak-driven v2.10.x patches

**What.** Promote `v2.10.0-rc1` to `v2.10.0` final after:
- The soak report has been read and any defect candidates either
  closed in v2.10.x or formally deferred to v2.11/v2.12.
- The retrieval-regression baseline has survived one no-op
  re-capture (idempotency check) with zero drift.

**Approach.**
1. Read `docs/QUALITY_SNAPSHOT_2026-05-16_v2.10_soak.md` once the
   in-flight soak completes.
2. Triage the "weakest 15" list:
   - **Code defect** → fix in `src/mmrag_v2/...`, tag as v2.10.1,
     ship.
   - **Embedder weakness** (most likely the bulk) → defer to Phase 1
     here; don't try to fix mid-rc1.
   - **Format / chunk-shape defect** → fix in v2.10.x.
3. If any v2.10.x ships, re-run strict gate + smoke + retrieval
   regression + soak before tagging.
4. Once stable, tag `v2.10.0` annotated on the same commit that bears
   the last v2.10.x patch (or `82c3639` if no patches needed).

**Done when.**
- v2.10.0 annotated tag on `main`, pushed to GitHub.
- v2.10 AFTER snapshot revised with the soak summary in §6 or §7.
- All soak-surfaced defects either closed or in the v2.11/v2.12
  carry-forward register.

**Risk.** Low. Mostly a triage exercise.
**Cost class.** Read + small fixes. No reconverts unless a soak finding
forces one.

---

### Phase 1 — Embedder shootout: `llava` vs `Qwen3-Embedding-4B`

**What.** Decide whether to swap the v2.10 embedder. The v2.10
retrieval-regression captures already surfaced specific failure modes
attributable to llava (multilingual weakness, domain-discrimination
weakness, larger-doc dominance for short queries). The candidate
`Qwen3-Embedding-4B` is:
- Multilingual (100+ languages including Dutch + German).
- 2560-dim cosine — 1.6× smaller index than llava's 4096-dim.
- Same vendor family as the soak judge — same Dashscope ecosystem if
  the comparison wins.
- Locally hostable via Ollama at ~2.5 GB Q4_0 → leaves headroom on a
  16 GB M1 even with llava resident for the VLM enrichment lane.

We are not changing the VLM enrichment path (which legitimately wants
a vision-language model). We are changing only the **text retrieval
embedder**. The two roles are decoupled.

**Schedule note.** Per release-cycle hygiene: this phase runs *after*
v2.10.0 ships final (Phase 0 above). We do NOT attempt to swap mid-rc1;
the regression test + soak baseline exist precisely so this swap can
be done with rigor instead of in-flight.

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
- Both collections green; both fingerprints captured; both soak
  reports authored.
- A decision row in `docs/DECISIONS.md` titled "v2.11 Embedder
  Shootout" recording which embedder won on which axes and which
  embedder ships.
- If Qwen3 wins decisively (e.g., Recall@5 doc +5 % or more, no
  regressions on Format/Faithfulness): `scripts/ingest_to_qdrant.py`
  default flipped to `qwen3-embedding:4b`; v2.10 llava collection
  retained as a 30-day rollback baseline; v2.11.0 tag is the
  embedder-swap release.
- If llava wins or the tie is too close to justify the migration cost:
  decision recorded, no swap. v2.11.0 ships without the swap and the
  Qwen3 work becomes a documented v2.12 candidate. The harnesses we
  built are NOT wasted — they're now permanent infrastructure.

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

### Phase 4 — EPUB engine question

**What.** The Phase 7 EPUB lane uses a marker-based workaround
(`__MMRAG_EPUB_CH_NNNN__` injected into the HTML pre-Docling) to recover
chapter information. That's correct but fragile — Docling can strip
the markers (it strips KI's first three) and we recover with the
pre-marker-buffer fallback. A cleaner architecture is a parallel
`EpubEngine` that doesn't depend on Docling for EPUB at all (uses
ebooklib directly + a UIR mapping).

**Decision point.** Is the marker workaround robust enough for v2.11,
or does v2.11 ship the parallel engine? The soak's EPUB queries
(Q11/Q12 in the regression set) plus the soak's per-doc Recall on the
two EPUBs are the data inputs.

**Approach (if in-scope).**
1. Add `src/mmrag_v2/engines/epub_engine.py` parallel to
   `engines/docling_adapter.py` shape.
2. Map ebooklib output → `UniversalDocument` directly.
3. Replace `processor._epub_to_html` + post-process markers with a
   direct path through the new engine.
4. Reconvert KI + ChatGPT EPUBs; re-run strict gate + retrieval
   regression + soak; compare to v2.10 EPUB baselines.

**Done when.** Decision row in `docs/DECISIONS.md`:
- "EPUB engine kept on marker-workaround for v2.11" (defer), OR
- "v2.11 ships parallel EpubEngine; KI and ChatGPT EPUB reconverted;
  Phase 7 marker workaround removed; AFTER snapshot updated."

**Risk.** Medium-high if in-scope. The marker workaround works.
**Cost class.** New engine code + reconverts if in-scope.

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

---

## 6. Decision log (this plan)

| Date | Change |
|---|---|
| 2026-05-16 | Draft v0.1 authored mid-v2.10.0-rc1-soak. Embedder challenger (Qwen3-Embedding-4B) named per release-plan discussion. Five rc1 carry-forward non-goals captured with default-recommended dispositions. Phase 0 explicitly gates Phase 1 on v2.10.0 final ship. |

---

## 7. Open questions (resolve before Draft v0.2)

1. **Soak findings.** What does the 2026-05-16 soak report show? Will
   the weakest-15 list move scope into Phase 0 (v2.10.x patches) or
   directly into Phase 1 (embedder problem)?
2. **Qwen3-Embedding-4B availability via Ollama.** Is `qwen3-embedding:4b`
   pullable as of 2026-05-16, or do we ship through `transformers`
   directly? (If `transformers`-only, add a small embedding service
   wrapper.)
3. **NuMarkdown-8B reachability.** Has the local endpoint come back
   online since v2.10's "off-network" finding?
4. **CI runner.** Self-hosted on the same network as Qdrant /
   Ollama, or GitHub-hosted with a private data bucket?
5. **EPUB engine.** Does any soak finding on KI / ChatGPT EPUB
   justify v2.11 effort, or is the Phase 7 marker workaround durable?

---

**END OF DRAFT v0.1.** Next checkpoint: read the in-flight soak report
(due ~21:00 UTC 2026-05-16), promote to Draft v0.2, then to Draft v1.0
once v2.10.0 final ships and Phase 0 is closed.
