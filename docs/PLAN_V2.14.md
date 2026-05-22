# Plan: v2.14 — Carry-Forwards Closeout + Local LLM Integration

**Status:** **Draft v0.3** (2026-05-22). User audit of Draft v0.2 (this
same day) led to four structural revisions, captured below. Phase 0
+ Phase 4a + Phase 5 already shipped; pending phases (1, 2, 3, 4b,
4c, 6, N) reflect the revised scope.

## Draft v0.3 revision summary (user audit response)

The Draft v0.2 plan proposed two band-aids that the user correctly
flagged: a lenient `format_form` judge axis (Phase 1) and a per-doc
embedder routing lane (Phase 2). Both would have masked or
architecturally entrenched real extraction defects. Evidence in the
v2.13 weakest-query rationales (`docs/QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_omlx.md`)
confirms: CarOK chunks have "table truncation + odd whitespace",
Fluent_Python has "truncated code". The judge isn't wrong — the
extraction is. Saved as a feedback rule in
`memory/feedback_fix_extraction_not_judge.md`.

Four revisions:

1. **Phase 1** redefined: `format_form` judge axis → **Form/Table
   layout extraction recovery** (fix the data, not the rubric).
2. **Phase 2** redefined: per-doc embedder routing → **Targeted HyDE
   bridging for code + minority languages** (use the already-shipped
   Phase 4a local HyDE to bridge the semantic gap at query time;
   no new collections, no permanent routing infra).
3. **Phase 4** guardrails tightened: explicit list of permitted
   local-judge uses (Format-axis judging only; prompt A/B; pre-filter
   tie-breaker pattern). Forbidden: tuning RRF weights, top_k,
   sparse/dense balance, or anything that adjusts retrieval breadth —
   the local judge's upward leniency bias on relevance + faithfulness
   would push tuning toward noisier retrieval.
4. **New Phase 6**: **Code-block chunking hygiene** (fix the
   Fluent_Python / ArcGIS_Python_Cookbook truncation root cause).

## Phase outcomes so far

| Phase | Status | Outcome |
|---|---|---|
| Phase 0 (judge calibration) | **SHIPPED 2026-05-22** | Per-axis verdict: relevance 81.7% (RESTRICTED), **format 90.2% (TRUSTWORTHY)**, faithfulness 76.1% (RESTRICTED). ±1 agreement ~100% on all axes — local LLM has same ordinal scale, slightly more lenient. Report: `docs/CALIBRATION_2026-05-22_v2.14_p0_local_judge.md` |
| Phase 4a (local HyDE) | **SHIPPED 2026-05-22** | `src/mmrag_v2/retrieval/hyde.py` gained `provider="vllm"` knob. Defaults to GX10 at `http://10.0.10.239:8000` with `Qwen/Qwen2.5-14B-Instruct`. 5 unit tests added. End-to-end smoke OK (392-char hypothesis generated locally in ~2s, $0). Default provider remains `dashscope` — no behavior change for existing callers. |
| Phase 5 (disk precheck) | **SHIPPED 2026-05-22** | `_check_disk_headroom()` in `synthetic_soak.py` aborts retrieve/judge stages below 10 GB free. Override via `SOAK_DISK_HEADROOM_FLOOR_GB` env. 5 unit tests added. |
| Phase 1 (Form/Table extraction recovery) | **PENDING — Draft v0.3 redefinition** | See section 2 below |
| Phase 2 (Targeted HyDE bridging) | **PENDING — Draft v0.3 redefinition** | See section 2 below |
| Phase 3 (rollback drop) | pending (decision point 2026-06-19) | Unchanged |
| Phase 4b (local judge in soak — Format-only) | **PENDING — Draft v0.3 scope tightened** | Format-axis only; explicit forbidden list |
| Phase 4c (local query gen) | pending | Likely safe — generation isn't judging |
| Phase 6 (Code-block chunking hygiene) | **PENDING — NEW in Draft v0.3** | See section 2 below |
| Phase N (close-out) | pending | Engine bump to 2.14.0 when enough phases land to justify a tag |

---

**Original Draft v0.1** (2026-05-22, preserved below as cycle archaeology).
Authored after v2.13.0 SHIPPED (commit `a77758b`, annotated tag staged
for user push) as the planning artifact for the next cycle.

**Predecessor:** [`docs/PLAN_V2.13.md`](PLAN_V2.13.md) — CLOSED
2026-05-22 with `v2.13.0` annotated tag staged on commit `a77758b`,
pending user push to GitHub + Gitea.
**Owner:** ingestion + retrieval + LLM-integration pipeline.

---

## 1. Why this plan exists

### Thesis

v2.13 closed the embedder workstream (local omlx wins 6/6 axes
apples-to-apples vs cloud) and shipped OCR auto-routing (Earthship
+6.2pp Format). The retrieval-quality side is in good shape:
R@1 ~58%, R@5 chunk ~78%, R@5 doc ~95%, Format ~93%.

What's left:

1. **A handful of per-doc regressions** the omlx swap revealed:
   German content (ATZ_Elektronik -12.5pp R@1), code-dense docs
   (Python_Cookbook -12.4pp, IRJET -12.5pp, Hybrid_electric
   -12.6pp, Greenhouse -12.5pp). These are offset by aggregate
   wins but worth investigating.

2. **The CarOK form-class Format penalty** is a documented judge-
   calibration limitation, not a content defect. The soak protocol
   needs a `format_form` axis variant so future form-class docs
   don't drag the headline Format number.

3. **The 30-day rollback window closes on 2026-06-19.** Decision
   point: drop `mmrag_v2_8__qwen3_dashscope` (1024-dim, ~17 GB)
   and `mmrag_v2_8` (legacy llava) if no rollback fired during
   the window.

4. **Local LLM capability now available.** Asus Ascent GX10 (DGX
   Spark clone) at `10.0.10.239:8000` runs vLLM with
   `Qwen/Qwen2.5-14B-Instruct` (32K context). Free local
   experimentation accelerator unlocks soak workloads the
   $25/cycle cap currently caps: bigger eval sets, hyperparameter
   sweeps, prompt iteration, HyDE default-on.

5. **Open carry-forwards from v2.11** (3a VLM swap, 3c UIR
   refactor PAUSED, 3e magazine rendered-region-crop) still live
   if user re-prioritizes them.

### Non-goals

- **No retrieval-side architecture changes.** v2.13's stack
  (hybrid + RRF + ModernBERT rerank + omlx embedder) is the
  production shape; v2.14 doesn't reshuffle that.
- **No new embedder swap.** omlx Qwen3-Embedding-8B is the
  production embedder for the foreseeable future.
- **No silent gate weakening.** Per the contract-violation rule.

---

## 2. Phases (proposed)

Each phase is a small, soak-validated change. Each can run
autonomously if calibration data justifies it. Phase 0 is the
gating prerequisite for Phase 4.

### Phase 0 — Local-LLM judge calibration *(prerequisite for Phase 4)*

**Goal:** measure per-axis agreement between `Qwen/Qwen2.5-14B-Instruct`
(local on GX10) and `qwen-max` (cloud) using the v2.13 P1 soak's
518 already-judged queries as ground truth. Same JUDGE prompt
structure on both sides — only the model differs.

**Method:** `scripts/calibrate_local_judge_vs_qwen_max.py`
(added 2026-05-22). Reuses `JUDGE_SYSTEM` + `JUDGE_USER_TEMPLATE`
from `synthetic_soak.py` to keep prompts identical. Outputs
`docs/CALIBRATION_2026-05-22_v2.14_p0_local_judge.md` with
per-axis agreement %, confusion matrices, and a disposition
verdict per axis.

**Disposition thresholds:**

| Per-axis exact-match | Verdict | Usage |
|---|---|---|
| ≥85% | TRUSTWORTHY | Use local judge for exploration soaks (RRF weight sweeps, top_k sweeps, prompt iteration) |
| 70-85% | RESTRICTED | HyDE-only — weaker semantics still help retrieval, but not enough for go/no-go judging |
| <70% | NOT USABLE | Either pick a stronger local model (Qwen3.6-35B-A3B-FP8 once vLLM supports it) or stay on cloud |

**Output:** SHIP if all three axes ≥85%; otherwise document
the restricted-use envelope and either (a) accept reduced scope
for Phase 4, or (b) gate Phase 4 on a stronger local model.

**Cost:** $0 (local LLM only; reuses existing qwen-max judgments).

### Phase 1 — Form/Table layout extraction recovery (REDEFINED in Draft v0.3)

**Goal:** Fix the actual extraction defect for form-class documents
like `CarOK_voorraadtelling`. The v2.13 weakest-query rationales
explicitly cite "table is slightly truncated", "odd spacing and
truncation", "odd whitespace" — these are real defects, not judge
miscalibration. Lenient judging would mask defects that downstream
generators will still hallucinate against.

**Why this is the right layer:** CarOK's tabular inventory data
needs to land in chunks as well-structured rows (markdown table or
key-value pairs), not as whitespace-collapsed prose. Once the
chunks are clean, the existing prose-calibrated Format judge
scores them correctly — no rubric carve-out needed.

**Method:**
1. Inspect a sample of current CarOK chunks. Confirm the truncation
   + whitespace damage at the chunk-emission step (not just at
   render time).
2. Route form-class documents (per existing
   `docs/QUALITY_GATES.md` §"Form / Invoice Acceptance Class"
   classifier) through a structured-table extraction path:
   - Primary attempt: Docling's table-structure-recognition
     (`do_table_structure=True` in `PdfPipelineOptions`). Emits
     each table as a Markdown table.
   - Fallback: VLM-assisted table parse if Docling's TSR doesn't
     resolve the columns cleanly (Qwen3-VL-8B on omlx, or qwen-vl-max
     cloud). This was tagged 3a in v2.11 carry-forwards — finally
     justified by a concrete defect, not speculation.
3. Re-extract `CarOK_voorraadtelling` and any other form-class
   doc in the corpus. Re-ingest into both production
   (`mmrag_v2_8__qwen3_local`) and the dashscope rollback collection
   so the apples-to-apples symmetry is maintained.
4. Sample a 30-query mini-soak on the re-extracted CarOK chunks
   to confirm the prose-Format judge no longer cites truncation /
   whitespace defects.

**Acceptance:**
- CarOK Format score recovers ≥85% on the EXISTING prose-Format
  rubric (no judge change required) on a 30-query mini-soak.
- No regression on the other 33 corpus docs' chunk counts or
  strict-gate pass status.
- Evidence + decision in `docs/DECISIONS.md`.

**Cost:** ~$0.50 (mini-soak qwen-max judge cost) + Docling CPU
re-extraction time.

### Phase 2 — Targeted HyDE bridging for code + minority languages (REDEFINED in Draft v0.3)

**Goal:** Address the omlx embedder's per-doc regressions on
German (`ATZ_Elektronik` -12.5pp R@1) and code-dense (`Python_Cookbook`
-12.4pp, `IRJET` -12.5pp, `Hybrid_electric_vehicles` -12.6pp,
`Greenhouse_Design` -12.5pp) content **at query time**, NOT by
routing to multiple embedder collections.

**Why this is the right layer:** Per-doc embedder routing forces
permanent architectural fragmentation — parallel Qdrant collections,
language/content classifiers upstream of retrieval, broken
cross-corpus search for queries that span both classes. Targeted
HyDE bridging reuses the already-shipped Phase 4a local HyDE
infrastructure: when a query is detected as code-intent or
minority-language, generate a synthetic hypothetical answer in
the target style/language, embed THAT, and the omlx embedder's
semantic gap shrinks dramatically.

The 2-second HyDE latency cost only applies to the targeted
lanes (estimated 5-10% of production queries); the default
retrieval path stays at ~1.6s p99.

**Method:**
1. Add a lightweight query-intent classifier (regex/heuristic
   first — code keywords like `def`, `class`, language patterns,
   non-ASCII ratio, or simple lang-id). Keep it deterministic
   and cheap; no LLM call.
2. When the classifier flags `intent=code` or
   `intent=minority_language`, the retrieval path automatically
   sets `use_hyde=True` + `hyde_provider="vllm"` (using the
   Phase 4a infrastructure) with a content-aware HyDE system
   prompt variant ("generate a code snippet answer" / "generate
   the answer in {detected_lang}").
3. Wire `hyde_provider` through `retrieve_hybrid_reranked` so
   the soak harness can A/B test the targeted-HyDE setting.
4. Validate on a focused mini-soak: same 5 affected docs (ATZ,
   Python_Cookbook, IRJET, Hybrid_electric_vehicles,
   Greenhouse_Design), 50 queries total, omlx-with-targeted-HyDE
   vs omlx-no-HyDE. Target: recover the -12.5pp R@1 deficit to
   within -3pp of dashscope on the same queries (so the
   aggregate stays a win without permanent infra).

**Acceptance:**
- Targeted HyDE on the 5 affected docs lifts R@1 by ≥8pp vs
  the no-HyDE baseline (recovering most of the gap without
  re-introducing cloud dependence).
- No regression on the 28 other corpus docs (sample 30 of them
  in the mini-soak to confirm; the default retrieval path
  remains HyDE-off so this is a non-issue, but worth a smoke).
- HyDE is OPT-IN per-query via the intent classifier; default
  production retrieval path unchanged.

**Cost:** ~$1-2 mini-soak.

**Why this defers permanent routing:** if targeted HyDE recovers
the deficit, no parallel embedder collection is needed. If it
doesn't, Phase 2 follow-up can revisit routing — but with
concrete evidence the cheaper bridge didn't suffice.

### Phase 3 — 30-day rollback drop (decision point: 2026-06-19)

**Goal:** if no v2.13.0 rollback fired during the 30-day window
(2026-05-22 → 2026-06-19), drop the dashscope rollback collection
and reclaim ~17 GB.

**Method:**
```bash
curl -X DELETE http://localhost:6333/collections/mmrag_v2_8__qwen3_dashscope
# Also drop the v2.10 legacy llava collection if still present:
curl -X DELETE http://localhost:6333/collections/mmrag_v2_8
```
Update `scripts/retrieval_regression_v2_12.py` to skip with a
"rollback collection deleted, see docs/DECISIONS.md" message
rather than failing.

**Acceptance:** disk reclaim verified + `docs/PROJECT_STATUS.md`
"Qdrant collections" table updated.

**Cost:** $0.

**Skip condition:** if any production user reported a regression
during the window that suggested swapping back, defer this phase
to v2.15 and document the issue first.

### Phase 4 — Local-LLM exploration accelerator *(SCOPE TIGHTENED in Draft v0.3)*

**Goal:** wire the local LLM into the experimentation loop where
the Phase 0 calibration verdict says it's safe, with explicit
guardrails to prevent the **leniency trap** (a lenient judge biased
upward on relevance/faithfulness will reward configs that retrieve
more noise because it forgives noise; optimization against it →
noisier retrieval).

**Phase 0 verdict (2026-05-22):**
- Format: 90.2% exact → **TRUSTWORTHY**
- Relevance: 81.7% exact → **RESTRICTED** (systematic upward bias)
- Faithfulness: 76.1% exact → **RESTRICTED** (systematic upward bias)

**PERMITTED local-judge uses:**
1. **Format-axis judging on any soak** (the well-calibrated axis).
   `--judge-provider vllm` is allowed for Format scoring.
2. **Prompt A/B testing** — comparing two judge prompts or two
   query-generation prompts. The relative-ranking signal survives
   the leniency bias as long as both arms are judged by the same
   biased judge.
3. **Pre-filter / tie-breaker pattern** — local-judge runs first
   on the full fixture; queries with score deltas above a noise
   threshold (e.g. local rates 2 but a previous-cycle qwen-max
   judgment said 0) get re-judged by cloud `qwen-max` as the
   tie-breaker. Cuts cloud spend dramatically while preserving
   ground-truth integrity.
4. **Free large-fixture query generation** (`gen-provider vllm`)
   — generation isn't judging; leniency bias doesn't apply.

**FORBIDDEN local-judge uses** (the leniency trap):
- Tuning **RRF weights**, **top_k_retrieve**, **top_n_fuse**,
  or any retrieval-breadth parameter
- Tuning **sparse/dense weight balance** in hybrid retrieval
- Selecting **rerank backends** or **reranker hyperparameters**
- Cycle-close go/no-go judging on Relevance + Faithfulness axes
- Anything that adjusts how much / how broadly the retrieval
  stack pulls in candidates

The rule of thumb: if the change could legitimately retrieve more
or fewer chunks per query, the lenient judge will systematically
prefer "more" because it forgives the added noise. Use cloud
`qwen-max` for those decisions.

**Sub-deliverables:**
- 4a) Local HyDE provider — **SHIPPED 2026-05-22** (`src/mmrag_v2/retrieval/hyde.py`
  `provider="vllm"`)
- 4b) `scripts/synthetic_soak.py` gains `--judge-provider {dashscope,vllm}`
  + `--judge-url` / `--judge-model` flags. When `vllm`, the soak
  WARNS at startup that the run is restricted to PERMITTED uses
  per the list above. Default remains `dashscope` / `qwen-max`.
- 4c) `--gen-provider {dashscope,vllm}` for query generation (no
  leniency concern; safe by default once smoke-tested).
- 4d) Tie-breaker harness as a new script (`scripts/local_then_cloud_soak.py`)
  — local judges everything, cloud re-judges the high-delta
  subset. Targets ~80% cost reduction on big-fixture soaks.
- 4e) A 1500-query Format-only exploration soak demonstrates
  the new capability + produces denser Format-axis evidence
  (the only axis the local judge is fully trusted on).

**Acceptance:** 1500-query Format-only soak completes in <60 min
on local and matches the 518-query baseline within ±2pp on Format.

**Cost:** 4b/4c/4d wiring: $0 (code only). 4e demonstration soak:
$0 (Format-only on local).

### Phase 5 — Disk-headroom precheck in `synthetic_soak.py` *(QoL)*

**Goal:** prevent a repeat of the v2.13 P1 disk-full incident
(disk hit 100% mid-judge, crashed Qdrant). Add a precheck that
refuses to start a stage if free disk is below a configurable
threshold (default 10 GB).

**Method:** small helper at the top of `stage_retrieve` and
`stage_judge` — `shutil.disk_usage(repo_root)` + warning/abort
threshold.

**Acceptance:** unit test simulates low-disk + verifies the
precheck aborts cleanly.

**Cost:** $0.

### Phase 6 — Code-block chunking hygiene (NEW in Draft v0.3)

**Goal:** Fix the chunk-boundary policy so fenced code blocks
(```python … ```) and indented code regions are never severed
mid-block. The v2.13 weakest-query rationales explicitly cite
"contains truncated code" for `Fluent_Python` and the same
pattern likely affects `ArcGIS_Python_Cookbook`, `Python_Distilled`,
and `Ayeva_Python_Patterns`.

**Why this is the right layer:** A code snippet sliced at line 17
of 25 is useless for retrieval (the embedder doesn't know the
function signature ends three lines later) and worse for the
downstream generator (it sees an incomplete function and may
hallucinate the rest). The fix is in the chunker, not the
retriever or the judge.

**Method:**
1. Inspect the current chunking logic in
   `src/mmrag_v2/universal/element_processor.py` and
   `src/mmrag_v2/batch_processor.py`. Identify where chunk
   boundaries can fall.
2. Add a code-block-aware policy:
   - Detect fenced code (markdown ```), indented blocks (4+
     spaces leading 3+ consecutive lines), and Docling
     `CodeItem` elements.
   - If a chunk would otherwise close mid-block, extend the
     chunk to end-of-block, up to a safe maximum (e.g., +50%
     over the normal chunk-size limit). Beyond that, accept
     a code-split but mark the chunk with `chunk.metadata.partial_code=True`
     and emit a paired continuation chunk with the rest.
3. Add tests for the three shapes (fenced, indented, Docling
   `CodeItem`) covering: (a) chunk fits cleanly, (b) chunk
   extends to end-of-block, (c) block exceeds the safe max
   and gets split with partial-code metadata.
4. Re-extract the four affected Python docs and re-ingest.
5. Mini-soak: 30 queries on the re-extracted Python docs,
   confirm "truncated code" rationales no longer appear in
   the weakest-15.

**Acceptance:**
- Zero "truncated code" rationales in the mini-soak weakest-15
  across the four re-extracted docs.
- No regression in chunk count >+15% on any of them (some
  growth expected from end-of-block extension; >15% suggests
  the safe-max is too generous).
- Existing chunk schema unchanged except for the optional new
  `metadata.partial_code` boolean.

**Cost:** ~$0.50 (mini-soak qwen-max judge cost) + chunker
re-extraction CPU time.

### Phase N — Cycle close-out

- Engine version bump `2.13.0` → `2.14.0`
- v2.14 retrieval-regression fingerprint (re-capture if any of
  Phase 1, Phase 2, or Phase 6 changed the production retrieval
  shape or chunk_ids on production docs; otherwise the v2.13
  fingerprint stays canonical)
- AFTER snapshot `docs/QUALITY_SNAPSHOT_<date>_v2.14_after.md`
- Layer-0/1 docs sweep
- v2.14.0 annotated tag staged for user push

---

## 3. v2.11+ carry-forwards (still open)

| Item | Source | v2.14 disposition |
|---|---|---|
| 3a NuMarkdown-8B / Qwen3-VL-8B local VLM | v2.11 plan | **Promoted to Phase 1 fallback** — VLM-assisted table parse is the Phase 1 fallback when Docling TSR doesn't resolve form-class table columns. v2.14 finally justifies this work with a concrete defect. |
| 3c UIR refactor (PAUSED) | v2.11 plan | Still PAUSED for user signoff |
| 3e Magazine rendered-region-crop | v2.11 plan | Defer with soak-data rationale (image-axis perf is OK without it) |

---

## 4. Phase ordering rationale (REVISED in Draft v0.3)

```
Phase 0 (calibration)        ← SHIPPED 2026-05-22
Phase 4a (local HyDE)        ← SHIPPED 2026-05-22 (prereq for Phase 2)
Phase 5 (disk precheck)      ← SHIPPED 2026-05-22

Phase 6 (code chunking)      ← independent; fix the root cause first
Phase 1 (form/table extract) ← independent; same shape as Phase 6 (fix-then-validate)
Phase 2 (HyDE bridging)      ← depends on Phase 4a (shipped); validates targeted-HyDE recovers per-doc deficits
Phase 4b/c/d/e (local LLM)   ← guardrails tightened; ships when Phase 1/2/6 land or are scoped
Phase 3 (rollback drop)      ← time-gated to 2026-06-19
Phase N (close-out)          ← terminal
```

Suggested execution order:

1. **Phase 6 (code chunking hygiene)** — fixes a concrete defect
   the v2.13 P1 soak found; small in code; produces re-extracted
   Python docs ready for Phase 2's HyDE bridging.
2. **Phase 1 (form/table extraction)** — same shape: fix the
   extraction, re-extract CarOK, validate via existing prose-Format
   judge. No rubric change.
3. **Phase 2 (targeted HyDE bridging)** — once Python docs are
   re-chunked (Phase 6) and form docs are re-extracted (Phase 1),
   the per-doc regressions can be re-measured cleanly and the
   targeted-HyDE bridge can be A/B tested without confounders.
4. **Phase 4b/4c/4d/4e (local LLM in soak)** — guardrails tightened;
   ship Format-only local judging + tie-breaker harness; gated
   permitted-uses list.
5. **Phase 3 (rollback drop)** — wait for 2026-06-19.
6. **Phase N (close-out)** — bump engine, retag, AFTER snapshot.

---

## 5. Budget (REVISED in Draft v0.3)

- **Cost cap**: $25/cycle (same as v2.13)
- **Estimated spend**: $4-6 across Phase 1 mini-soak + Phase 2
  mini-soak + Phase 6 mini-soak + Phase 4e Format-only demo (most
  of which is cloud-free on the local path); Phase 4b/4c/4d are $0
  code work
- **Local LLM usage**: $0 (LAN)

## 6. Risks (REVISED in Draft v0.3)

| Risk | Mitigation |
|---|---|
| Phase 1 Docling TSR can't recover CarOK tables cleanly | VLM-assisted parse fallback (already on hand: Qwen3-VL-8B on omlx or qwen-vl-max cloud); 3a promoted |
| Phase 2 targeted HyDE doesn't recover the -12.5pp R@1 deficit | Revisit per-doc embedder routing as Phase 2-bis with concrete "HyDE didn't suffice" evidence (not speculative) |
| Phase 6 code-extension policy explodes chunk count | Acceptance bound: ≤+15% chunk growth per doc; safe-max forces a partial-code split beyond that with metadata flag |
| Local judge leniency trap masking real retrieval regression | Phase 4 guardrails: FORBIDDEN list of tuning uses; cloud `qwen-max` retained for all retrieval-breadth decisions and cycle-close go/no-go |
| 30-day rollback window had a real rollback need we didn't catch | Phase 3 skip rule + grace period; collection retained 30 days |
| Disk fills again during a soak | Phase 5 precheck (SHIPPED) prevents recurrence |

## 7. Open questions for the user (when back)

- (none required to start) — Phase 6, 1, 2, 4b–e are autonomously
  executable. Phase 3 needs your "no rollback fired, drop it" signoff
  on or after 2026-06-19.
- Optional: SSH key onboarding for the GX10 (still pending — only
  blocker for live diagnostics if the local LLM crashes mid-soak).
- Heads-up: Phase 1 may need the user to OK the choice between
  "Docling TSR alone" vs "Docling TSR + VLM fallback" if Docling's
  default doesn't resolve CarOK cleanly. I'll proceed with TSR
  first and escalate with evidence if it falls short.
